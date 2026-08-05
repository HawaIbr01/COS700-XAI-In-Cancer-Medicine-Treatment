"""
src/extraction/nsga.py — COS700 Part 4, Methodology §4.4.

OWNER: Sakhile (feat/nsga2-rule-extraction)

CONTRACT (docs/CONTRACTS.md Arm 4):
    Produces:
        artifacts/rules/nsga_baseline.json
        artifacts/rules/nsga_sam.json
    (same §0.4 schema as GA — this is what makes the GA-vs-NSGA-II
    comparison in Evaluation fair; written via schema.save_rule_pool,
    which validates before writing.)

DEAP NSGA-II rule extraction. Multi-objective search over discretised GDSC
features:
  objective 1 (maximise): fidelity macro-F1 vs the model's class_pred
  objective 2 (minimise): rule length (# active conditions)

Usage (from the project root):
    python -m src.extraction.nsga --condition baseline
    python -m src.extraction.nsga --condition sam
    python -m src.extraction.nsga --condition baseline --n-runs 3 --pop 12 --n-gen 10   # smoke test

Fully wired to src/common/{config,metrics,schema}.py and
src/extraction/common.py — no fallback stubs. If any of those modules'
signatures change, this file breaks loudly at import time rather than
silently drifting.

MERGE NOTE: this file's config access and rule-pool writing were updated
to match the canonical src/common/config.py (PipelineConfig dataclass,
attribute access) and src/common/schema.py (list-of-errors validate,
save_rule_pool validates-then-writes) resolved during the merge with
feat/nsga2-rule-extraction. Everything else — data loading, scoring,
DEAP toolbox, run loop, aggregation, CLI — is unchanged from Sakhile's
original implementation.
"""

from __future__ import annotations

import argparse
import copy
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

import pandas as pd
from deap import base, creator, tools

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.common.config import config_hash, load_config
from src.common.metrics import fidelity_f1
from src.common.schema import save_rule_pool
from src.extraction.common import (
    cxTwoPoint_partitions,
    decode_individual,
    mutate_partition,
    random_gene,
    resolve_genes,
)

FIDELITY_SPLIT = "train"  # rules are fit against train predictions; test is
                          # held out for Evaluation's final numbers (§4.6).


# --------------------------------------------------------------------------
# Data loading
# --------------------------------------------------------------------------

def load_condition_data(data_dir: Path, condition: str, split: str):
    """Returns (df, feature_cols, feature_map, class_pred_col)."""
    df = pd.read_parquet(data_dir / "gdsc_processed.parquet")

    with open(data_dir / "splits.json") as f:
        splits = json.load(f)
    with open(data_dir / "feature_map.json") as f:
        feature_map = json.load(f)

    row_ids = set(splits[split])
    df = df[df["row_id"].isin(row_ids)].reset_index(drop=True)

    preds_path = data_dir.parent / "predictions" / f"{condition}_{split}.parquet"
    preds = pd.read_parquet(preds_path)
    df = df.merge(preds[["row_id", "class_pred"]], on="row_id", how="inner")

    feature_cols = list(feature_map.keys())
    missing = [c for c in feature_cols if c not in df.columns]
    if missing:
        raise ValueError(
            f"feature_map.json lists columns not present in gdsc_processed.parquet: {missing}"
        )

    return df, feature_cols, feature_map, "class_pred"


# --------------------------------------------------------------------------
# Scoring — cached per genotype (crossover/mutation reproduce genotypes
# constantly, and the same rule often reappears across the 30 runs)
# --------------------------------------------------------------------------

def rule_mask(df: pd.DataFrame, feature_cols, individual):
    import numpy as np
    from src.extraction.common import PARTITION_LABELS
    mask = np.ones(len(df), dtype=bool)
    for i, val in enumerate(individual):
        if val != -1:
            mask &= (df[feature_cols[i]].values == PARTITION_LABELS[val])
    return mask


def _score_genotype(individual, df, feature_cols, class_pred_col, min_coverage):
    mask = rule_mask(df, feature_cols, individual)
    coverage = float(mask.mean())
    length = sum(1 for v in individual if v != -1)

    # length == 0 means every gene is -1 (feature excluded): decode_individual()
    # would return zero conditions, and rule_mask() with zero conditions matches
    # every row. Left unguarded, this "empty rule" scores as a maximally general,
    # zero-length rule that can dominate on both NSGA-II objectives for free and
    # make it into the output pool with empty 'conditions'/'genes' — which fails
    # schema validation at write time (caught via a real run against a smoke-test
    # fixture, not just from reading the code). Route it through the same
    # None-predicted_class path already used for zero-coverage genotypes, so
    # aggregate_runs()'s existing `if predicted_class is None: continue` filters
    # it out before it ever reaches the rule pool.
    if length == 0 or mask.sum() == 0:
        return 0.0, length, coverage, None

    covered_preds = df.loc[mask, class_pred_col]
    predicted_class = covered_preds.mode().iat[0]
    rule_preds = pd.Series(predicted_class, index=covered_preds.index)

    fidelity = fidelity_f1(rule_preds, covered_preds)

    # Optional support/coverage floor: without this, a rule matching 1-3 rows
    # trivially hits fidelity 1.0 (predicting the majority class among a
    # handful of rows is guaranteed to match) and dominates on the fidelity
    # axis for free. Zeroing fidelity below the floor means such genotypes
    # never survive selNSGA2's dominance check against any genotype with
    # real coverage and non-trivial fidelity. Off by default (0.0) to match
    # the branch contract exactly; opt in with --min-coverage.
    if min_coverage > 0.0 and coverage < min_coverage:
        fidelity = 0.0

    return float(fidelity), length, coverage, predicted_class


def make_scorer(df, feature_cols, class_pred_col, min_coverage=0.0):
    cache = {}

    def score(individual):
        key = tuple(individual)
        if key not in cache:
            cache[key] = _score_genotype(list(key), df, feature_cols, class_pred_col, min_coverage)
        return cache[key]

    return score


# --------------------------------------------------------------------------
# DEAP toolbox
# --------------------------------------------------------------------------

def build_toolbox(n_features, scorer):
    if not hasattr(creator, "FitnessMulti"):
        creator.create("FitnessMulti", base.Fitness, weights=(1.0, -1.0))
    if not hasattr(creator, "Individual"):
        creator.create("Individual", list, fitness=creator.FitnessMulti)

    toolbox = base.Toolbox()
    toolbox.register("attr_partition", random_gene)
    toolbox.register(
        "individual", tools.initRepeat, creator.Individual, toolbox.attr_partition, n=n_features
    )
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register("clone", copy.deepcopy)

    def evaluate(individual):
        fidelity, length, _coverage, _pred = scorer(individual)
        return fidelity, length

    toolbox.register("evaluate", evaluate)
    toolbox.register("mate", cxTwoPoint_partitions)
    toolbox.register("mutate", mutate_partition, indpb=0.05)
    toolbox.register("select", tools.selNSGA2)                 # non-dom sort + crowding
    toolbox.register("select_parents", tools.selTournamentDCD)  # crowded binary tournament

    return toolbox


# --------------------------------------------------------------------------
# Single NSGA-II run
# --------------------------------------------------------------------------

def run_nsga2(toolbox, pop_size, n_gen, cxpb, mutpb, seed):
    assert pop_size % 4 == 0, "selTournamentDCD requires pop_size % 4 == 0"
    random.seed(seed)

    pop = toolbox.population(n=pop_size)
    for ind in pop:
        ind.fitness.values = toolbox.evaluate(ind)
    pop = toolbox.select(pop, len(pop))  # assigns rank + crowding distance

    for _gen in range(n_gen):
        parents = toolbox.select_parents(pop, len(pop))
        offspring = [toolbox.clone(ind) for ind in parents]

        for c1, c2 in zip(offspring[::2], offspring[1::2]):
            if random.random() < cxpb:
                toolbox.mate(c1, c2)
                del c1.fitness.values, c2.fitness.values
            if random.random() < mutpb:
                toolbox.mutate(c1)
                del c1.fitness.values
            if random.random() < mutpb:
                toolbox.mutate(c2)
                del c2.fitness.values

        invalid = [ind for ind in offspring if not ind.fitness.valid]
        for ind in invalid:
            ind.fitness.values = toolbox.evaluate(ind)

        pop = toolbox.select(pop + offspring, pop_size)  # elitist mu+lambda

    front0 = tools.sortNondominated(pop, len(pop), first_front_only=True)[0]
    return front0


# --------------------------------------------------------------------------
# Aggregation across the seeded runs
# --------------------------------------------------------------------------

def aggregate_runs(all_run_fronts, scorer, feature_cols, feature_map):
    genotype_runs = defaultdict(list)
    genotype_individual = {}

    for run_id, front in all_run_fronts:
        for ind in front:
            key = tuple(ind)
            genotype_runs[key].append(run_id)
            genotype_individual.setdefault(key, ind)

    pooled = list(genotype_individual.values())
    final_front = tools.sortNondominated(pooled, len(pooled), first_front_only=True)[0]
    final_keys = {tuple(ind) for ind in final_front}

    rules = []
    for key in final_keys:
        individual = genotype_individual[key]
        fidelity, length, coverage, predicted_class = scorer(individual)
        if predicted_class is None:
            continue

        conditions = decode_individual(individual, feature_cols)
        rules.append({
            "conditions": conditions,
            "genes": resolve_genes(conditions, feature_map),
            "predicted_class": predicted_class,
            "length": length,
            "fidelity_f1": round(fidelity, 4),
            "coverage": round(coverage, 4),
            "run_ids": sorted(set(genotype_runs[key])),
            "non_dominated": True,
        })

    rules.sort(key=lambda r: (-r["fidelity_f1"], r["length"]))
    return rules


def write_rule_pool(out_path, condition, n_runs, config_hash_val, rules, feature_map):
    for i, r in enumerate(rules, start=1):
        r["id"] = f"nsga_{condition}_{i:04d}"

    payload = {
        "meta": {
            "extractor": "nsga",
            "condition": condition,
            "runs": n_runs,
            "config_hash": config_hash_val,
        },
        "rules": rules,
    }

    # save_rule_pool validates against schema.validate_rule_pool internally
    # and raises ValueError with the full list of problems if invalid, then
    # writes only on success. (Previously called validate_rule_pool() alone
    # expecting it to raise — it doesn't, it returns a list of errors — so
    # that call was silently doing nothing. save_rule_pool is the correct
    # validate-then-write entry point in the resolved schema.py.)
    save_rule_pool(payload, out_path, feature_map=feature_map)
    return out_path


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--condition", choices=["baseline", "sam"], required=True)
    ap.add_argument("--config", default="configs/pipeline.yaml")
    ap.add_argument("--data-dir", default="artifacts/data")
    ap.add_argument("--out-dir", default="artifacts/rules")
    ap.add_argument("--n-runs", type=int, default=None, help="override pipeline.yaml nsga.runs")
    ap.add_argument("--pop", type=int, default=None, help="override pipeline.yaml nsga.pop")
    ap.add_argument("--n-gen", type=int, default=None, help="override pipeline.yaml nsga.generations")
    ap.add_argument("--seed", type=int, default=None, help="override pipeline.yaml seed")
    ap.add_argument("--cxpb", type=float, default=0.9)
    ap.add_argument("--mutpb", type=float, default=0.2)
    ap.add_argument("--min-coverage", type=float, default=0.0,
                     help="zero out fidelity below this coverage fraction (0.0 = off, matches spec exactly)")
    args = ap.parse_args()

    config_path = Path(args.config)
    cfg = load_config(config_path)
    # cfg is a PipelineConfig dataclass (attribute access), not a dict —
    # matches the canonical src/common/config.py resolved during the merge.
    n_gen = args.n_gen or cfg.nsga.generations
    pop_size = args.pop or cfg.nsga.pop
    n_runs = args.n_runs or cfg.nsga.runs
    base_seed = args.seed if args.seed is not None else cfg.seed

    df, feature_cols, feature_map, class_pred_col = load_condition_data(
        Path(args.data_dir), args.condition, FIDELITY_SPLIT
    )
    scorer = make_scorer(df, feature_cols, class_pred_col, min_coverage=args.min_coverage)
    toolbox = build_toolbox(len(feature_cols), scorer)

    print(
        f"[nsga] condition={args.condition} split={FIDELITY_SPLIT} "
        f"features={len(feature_cols)} rows={len(df)} pop={pop_size} gens={n_gen} runs={n_runs} "
        f"min_coverage={args.min_coverage}"
    )

    all_run_fronts = []
    for run_id in range(1, n_runs + 1):
        seed = base_seed + run_id
        front = run_nsga2(toolbox, pop_size, n_gen, args.cxpb, args.mutpb, seed)
        all_run_fronts.append((run_id, front))
        print(f"[nsga] run {run_id}/{n_runs} seed={seed} front_size={len(front)}")

    rules = aggregate_runs(all_run_fronts, scorer, feature_cols, feature_map)
    config_hash_val = config_hash(config_path)
    out_path = Path(args.out_dir) / f"nsga_{args.condition}.json"
    write_rule_pool(out_path, args.condition, n_runs, config_hash_val, rules, feature_map)

    print(f"[nsga] wrote {len(rules)} non-dominated rules -> {out_path}")


if __name__ == "__main__":
    main()