"""
Single-objective Genetic Algorithm rule extraction (DEAP).

OWNER: Melisa (feat/ga-rule-extraction)
Methodology: §4.3

CONTRACT (docs/CONTRACTS.md Arm 3):
    Receives: artifacts/data/*, artifacts/checkpoints/{baseline,sam}/,
        artifacts/predictions/{baseline,sam}_*.parquet,
        src.common.metrics, src.extraction.common
    Produces:
        artifacts/rules/ga_baseline.json
        artifacts/rules/ga_sam.json
    (both validated against src.common.schema.validate_rule_pool
    before being written — via schema.save_rule_pool.)

Usage (from the project root):
    python -m src.extraction.ga --condition baseline
    python -m src.extraction.ga --condition sam
    python -m src.extraction.ga --condition baseline --n-runs 2 --pop 12 --n-gen 10   # smoke test

Mirrors nsga.py's structure deliberately (data loading, config/metrics/
schema wiring, aggregation-across-runs shape) so GA and NSGA-II stay
comparable per CONTRACTS.md Arm 3/4, differing only where the
methodology requires it: tournament selection instead of non-dominated
sorting, a scalarised (fidelity + accuracy + comprehensibility) fitness
instead of true multi-objective, plus rule pruning and aging-evolution
diversity (§4.3.3/§4.3.4, explicitly not required of NSGA-II).

Does NOT load a model checkpoint or run inference — per the Arm 1
hand-off note ("you do not need to run the model"), fidelity is scored
against the already-materialised predictions/{condition}_*.parquet
class_pred column.
"""

from __future__ import annotations

import argparse
import copy
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from deap import base, creator, tools

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.common.config import config_hash, load_config
from src.common.metrics import comprehensibility, fidelity_f1
from src.common.schema import save_rule_pool
from src.extraction.common import (
    PARTITION_LABELS,
    cxTwoPoint_partitions,
    decode_individual,
    mutate_partition,
    predict_with_rule,
    resolve_genes,
    sparse_individual,
)

ARTIFACTS_DIR = Path(__file__).resolve().parents[2] / "artifacts"

# Fidelity is fit on train (same split NSGA-II uses, for comparability);
# accuracy is scored on val, which is what "accuracy on held-out data"
# in the Arm 3 methodology spec means, and is why GA (unlike NSGA-II)
# needs a real ground-truth label per row.
FIDELITY_SPLIT = "train"
ACCURACY_SPLIT = "val"

# Scalarisation weights (§4.3.3). Provisional -- CONTRACTS.md Arm 3 flags
# "fitness weighting" as likely to change pending pilot + alignment with
# Yedjour et al.; override via CLI rather than editing these in place.
DEFAULT_W_FIDELITY = 0.45
DEFAULT_W_ACCURACY = 0.35
DEFAULT_W_COMPREHENSIBILITY = 0.20

# Aging-evolution diversity (§4.3.3): individuals older than this many
# generations are replaced by fresh random immigrants, preventing the
# population from stagnating on an early local optimum.
DEFAULT_MAX_AGE = 8


# --------------------------------------------------------------------------
# Data loading
# --------------------------------------------------------------------------

def load_condition_data(data_dir: Path, condition: str):
    """
    Returns (df_train, df_val, feature_cols, feature_map, true_class_by_row).

    df_train carries the condition's class_pred for the fidelity target;
    df_val carries both class_pred (unused by GA, kept for parity with
    nsga.py) and the true response class (for the accuracy objective),
    joined from response_classes.json's per_row_class.
    """
    df = pd.read_parquet(data_dir / "gdsc_processed.parquet")

    with open(data_dir / "splits.json") as f:
        splits = json.load(f)
    with open(data_dir / "feature_map.json") as f:
        feature_map = json.load(f)
    with open(data_dir / "response_classes.json") as f:
        response_classes = json.load(f)

    true_class_by_row = {
        int(row_id): cls for row_id, cls in response_classes["per_row_class"].items()
    }

    feature_cols = list(feature_map.keys())
    missing = [c for c in feature_cols if c not in df.columns]
    if missing:
        raise ValueError(
            f"feature_map.json lists columns not present in gdsc_processed.parquet: {missing}"
        )

    def _load_split(split_name):
        row_ids = set(splits[split_name])
        split_df = df[df["row_id"].isin(row_ids)].reset_index(drop=True)
        preds_path = data_dir.parent / "predictions" / f"{condition}_{split_name}.parquet"
        preds = pd.read_parquet(preds_path)
        split_df = split_df.merge(preds[["row_id", "class_pred"]], on="row_id", how="inner")
        return split_df

    df_train = _load_split(FIDELITY_SPLIT)
    df_val = _load_split(ACCURACY_SPLIT)
    df_val["true_class"] = df_val["row_id"].map(true_class_by_row)
    n_unresolved = df_val["true_class"].isna().sum()
    if n_unresolved:
        print(f"[ga] warning: {n_unresolved} val rows have no entry in "
              f"response_classes.json per_row_class -- dropping them from "
              f"the accuracy objective for this run.")
        df_val = df_val.dropna(subset=["true_class"]).reset_index(drop=True)

    return df_train, df_val, feature_cols, feature_map


# --------------------------------------------------------------------------
# Scoring — cached per genotype, same rationale as nsga.py: crossover and
# mutation reproduce genotypes constantly, and the same rule commonly
# reappears across the 30 independent runs.
# --------------------------------------------------------------------------

def _rule_mask(df, feature_cols, individual):
    """
    Boolean mask of rows in `df` satisfied by every active (non -1) gene
    in `individual`.

    NOTE: this used to live in src.extraction.common as a shared
    `rule_mask()`. Sakhile's real common.py deliberately doesn't export
    one -- his `predict_with_rule(covered_preds)` takes an already-masked
    Series and leaves masking to each caller (see common.py's docstring:
    "covered_preds ... the rows a rule's conditions match"). Rather than
    add rule_mask back to common.py (which risks a merge fight with
    nsga.py over shared-file ownership per docs/CONTRACTS.md), this stays
    local to ga.py -- a few duplicated lines, not a divergent contract.
    """
    mask = np.ones(len(df), dtype=bool)
    for i, val in enumerate(individual):
        if val != -1:
            mask &= (df[feature_cols[i]].values == PARTITION_LABELS[val])
    return mask


def _score_genotype(individual, df_train, df_val, feature_cols, w_fidelity,
                     w_accuracy, w_comprehensibility, min_coverage):
    """
    Returns (weighted_fitness, fidelity, accuracy, comp_score, length,
    train_coverage, predicted_class).

    fidelity  : from common.predict_with_rule() on the TRAIN rows the rule
                covers -- macro-F1 between that covered set's own
                class_pred majority vote and the covered class_pred values
                themselves (i.e. how internally consistent the model's
                predictions are within the rule's covered region; a rule
                that carves out a region the model classifies uniformly
                scores near 1.0, a rule straddling the model's decision
                boundary scores near 0.5).
    accuracy  : macro-F1 of that SAME predicted_class (fixed, from train)
                vs the real response_classes.json label on VAL rows the
                rule covers (0.0 if no val coverage). predict_with_rule()
                isn't reused here on purpose -- calling it on val's
                true_class would re-derive its own majority-vote class
                from true_class and self-compare, which measures label
                homogeneity, not whether the rule's train-derived
                prediction generalises to held-out ground truth. This
                needs a fixed comparison target, so it's a direct
                fidelity_f1() call instead.
    comp_score: 1 - length/n_features, a higher-is-better transform of
                src.common.metrics.comprehensibility()'s raw length, per
                that function's own docstring note that any such
                transform belongs in the calling arm, not in metrics.py.
    """
    n_features = len(feature_cols)
    length = sum(1 for v in individual if v != -1)
    if length == 0:
        return -1.0, 0.0, 0.0, 0.0, length, 0.0, None

    train_mask = _rule_mask(df_train, feature_cols, individual)
    train_coverage = float(train_mask.mean()) if len(df_train) else 0.0
    if train_mask.sum() == 0 or (min_coverage > 0.0 and train_coverage < min_coverage):
        return -1.0, 0.0, 0.0, 0.0, length, train_coverage, None

    covered_train_preds = df_train.loc[train_mask, "class_pred"]
    predicted_class, fidelity = predict_with_rule(covered_train_preds)
    if predicted_class is None:  # can't happen given the train_mask.sum()==0 guard above, but stay defensive
        return -1.0, 0.0, 0.0, 0.0, length, train_coverage, None

    val_mask = _rule_mask(df_val, feature_cols, individual)
    if val_mask.sum() == 0:
        accuracy = 0.0
    else:
        covered_true = df_val.loc[val_mask, "true_class"]
        rule_preds_val = pd.Series(predicted_class, index=covered_true.index)
        accuracy = fidelity_f1(rule_preds_val, covered_true)  # same macro-F1 metric, fixed target

    raw_length = comprehensibility({"conditions": [None] * length})  # matches metrics.py's contract
    comp_score = 1.0 - (raw_length / n_features)

    weighted = w_fidelity * fidelity + w_accuracy * accuracy + w_comprehensibility * comp_score
    return weighted, fidelity, accuracy, comp_score, length, train_coverage, predicted_class


def make_scorer(df_train, df_val, feature_cols, w_fidelity, w_accuracy,
                 w_comprehensibility, min_coverage):
    cache = {}

    def score(individual):
        key = tuple(individual)
        if key not in cache:
            cache[key] = _score_genotype(
                list(key), df_train, df_val, feature_cols,
                w_fidelity, w_accuracy, w_comprehensibility, min_coverage,
            )
        return cache[key]

    return score


# --------------------------------------------------------------------------
# DEAP toolbox — tournament selection, two-point crossover, mutation
# (§4.3.3), matched to NSGA-II's operators where the methodology allows,
# for comparability.
# --------------------------------------------------------------------------

def build_toolbox(n_features, scorer, tournament_size=3):
    if not hasattr(creator, "FitnessGA"):
        creator.create("FitnessGA", base.Fitness, weights=(1.0,))
    if not hasattr(creator, "GAIndividual"):
        creator.create("GAIndividual", list, fitness=creator.FitnessGA)

    toolbox = base.Toolbox()
    toolbox.register(
        "individual", tools.initIterate, creator.GAIndividual, lambda: sparse_individual(n_features)
    )
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register("clone", copy.deepcopy)

    def evaluate(individual):
        weighted, *_ = scorer(individual)
        return (weighted,)

    toolbox.register("evaluate", evaluate)
    toolbox.register("mate", cxTwoPoint_partitions)
    toolbox.register("mutate", mutate_partition, indpb=0.05)
    toolbox.register("select", tools.selTournament, tournsize=tournament_size)

    return toolbox


# --------------------------------------------------------------------------
# Rule pruning (§4.3.4): drop low-support conditions each generation
# rather than waiting for selection pressure alone to remove them.
# --------------------------------------------------------------------------

def prune_population(population, scorer, min_coverage):
    """Mutates rules in place: any individual whose train coverage falls
    below min_coverage has active conditions stripped, one at a time
    (randomly chosen), until it clears the floor or runs out of
    conditions. Invalidates fitness on anything changed."""
    if min_coverage <= 0.0:
        return
    for ind in population:
        length = sum(1 for v in ind if v != -1)
        if length <= 1:
            continue
        _, _, _, _, _, coverage, _ = scorer(ind)
        if coverage >= min_coverage:
            continue
        active_idx = [i for i, v in enumerate(ind) if v != -1]
        random.shuffle(active_idx)
        changed = False
        for idx in active_idx:
            ind[idx] = -1
            changed = True
            _, _, _, _, _, coverage, _ = scorer(ind)
            if coverage >= min_coverage:
                break
        if changed and ind.fitness.valid:
            del ind.fitness.values


# --------------------------------------------------------------------------
# Aging-evolution diversity (§4.3.3): individuals track how many
# generations they've survived; once past max_age they're replaced by
# fresh random immigrants rather than being bred from indefinitely, so
# the population doesn't stagnate around one early-found local optimum.
# --------------------------------------------------------------------------

def apply_aging(population, toolbox, max_age):
    if max_age is None or max_age <= 0:
        return population
    new_population = []
    for ind in population:
        age = getattr(ind, "age", 0)
        if age >= max_age:
            fresh = toolbox.individual()
            fresh.age = 0
            new_population.append(fresh)
        else:
            ind.age = age + 1
            new_population.append(ind)
    return new_population


# --------------------------------------------------------------------------
# Single GA run
# --------------------------------------------------------------------------

def run_ga(toolbox, scorer, pop_size, n_gen, cxpb, mutpb, min_coverage, max_age, seed, logbook=None):
    """logbook: optional tools.Logbook -- when given, records per-generation
    fitness stats (min/max/avg of the weighted scalar fitness) across the
    population. Purely additive/opt-in: callers that don't pass it (every
    real extraction run) get identical behaviour and RNG consumption to
    before this parameter existed."""
    random.seed(seed)
    stats = None
    if logbook is not None:
        stats = tools.Statistics(lambda ind: ind.fitness.values[0])
        stats.register("min", min)
        stats.register("max", max)
        stats.register("avg", lambda vals: sum(vals) / len(vals))

    population = toolbox.population(n=pop_size)
    for ind in population:
        ind.age = 0
        ind.fitness.values = toolbox.evaluate(ind)

    hof = tools.HallOfFame(10)
    hof.update(population)

    if logbook is not None:
        logbook.record(gen=0, popsize=len(population), **stats.compile(population))

    for _gen in range(n_gen):
        offspring = toolbox.select(population, len(population))
        offspring = [toolbox.clone(ind) for ind in offspring]

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

        prune_population(offspring, scorer, min_coverage)

        invalid = [ind for ind in offspring if not ind.fitness.valid]
        for ind in invalid:
            ind.fitness.values = toolbox.evaluate(ind)

        offspring = apply_aging(offspring, toolbox, max_age)
        invalid = [ind for ind in offspring if not ind.fitness.valid]
        for ind in invalid:
            ind.fitness.values = toolbox.evaluate(ind)

        population[:] = offspring
        hof.update(population)

        if logbook is not None:
            logbook.record(gen=_gen + 1, popsize=len(population), **stats.compile(population))

    # "Non-dominated" for a scalarised GA is filtered post-hoc across the
    # three underlying objectives (fidelity, accuracy, comprehensibility)
    # so the output shape matches NSGA-II's front, as CONTRACTS.md Arm 3
    # requires ("aggregate the non-dominated rules ... into one pool").
    candidates = list(hof) + population
    return _pareto_filter(candidates, scorer)


def _dominates(a, b):
    """a dominates b on (fidelity, accuracy, comp_score), all maximised."""
    return all(x >= y for x, y in zip(a, b)) and any(x > y for x, y in zip(a, b))


def _pareto_filter(candidates, scorer):
    seen = {}
    for ind in candidates:
        key = tuple(ind)
        if key not in seen:
            seen[key] = ind

    objs = {}
    for key, ind in seen.items():
        _, fidelity, accuracy, comp_score, _, _, predicted_class = scorer(ind)
        if predicted_class is None:
            continue
        objs[key] = (fidelity, accuracy, comp_score)

    keys = list(objs.keys())
    front = [k for k in keys if not any(_dominates(objs[o], objs[k]) for o in keys if o != k)]
    return [seen[k] for k in front]


# --------------------------------------------------------------------------
# Aggregation across the 30 seeded runs (mirrors nsga.py's aggregate_runs)
# --------------------------------------------------------------------------

def aggregate_runs(all_run_fronts, scorer, feature_cols, feature_map):
    genotype_runs = defaultdict(list)
    genotype_individual = {}

    for run_id, front in all_run_fronts:
        for ind in front:
            key = tuple(ind)
            genotype_runs[key].append(run_id)
            genotype_individual.setdefault(key, ind)

    pooled_keys = list(genotype_individual.keys())
    pooled_objs = {}
    for key in pooled_keys:
        _, fidelity, accuracy, comp_score, _, _, predicted_class = scorer(genotype_individual[key])
        if predicted_class is None:
            continue
        pooled_objs[key] = (fidelity, accuracy, comp_score)

    final_keys = [
        k for k in pooled_objs
        if not any(_dominates(pooled_objs[o], pooled_objs[k]) for o in pooled_objs if o != k)
    ]

    rules = []
    for key in final_keys:
        individual = genotype_individual[key]
        weighted, fidelity, accuracy, comp_score, length, coverage, predicted_class = scorer(individual)

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
            # accuracy/comp_score aren't part of the §0.4 schema's
            # required keys, but keeping them out entirely would throw
            # away the accuracy objective the methodology asks for and
            # Evaluation might want -- schema.validate_rule_pool only
            # checks REQUIRED_RULE_KEYS are *present*, extra keys pass.
            "accuracy": round(accuracy, 4),
        })

    rules.sort(key=lambda r: (-r["fidelity_f1"], r["length"]))
    return rules


def write_rule_pool(out_path, condition, n_runs, config_hash_val, rules, feature_map):
    for i, r in enumerate(rules, start=1):
        r["id"] = f"ga_{condition}_{i:04d}"

    payload = {
        "meta": {
            "extractor": "ga",
            "condition": condition,
            "runs": n_runs,
            "config_hash": config_hash_val,
        },
        "rules": rules,
    }
    save_rule_pool(payload, out_path, feature_map=feature_map)
    return out_path


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------

def extract_rules_ga(condition: str, cfg, args) -> list:
    df_train, df_val, feature_cols, feature_map = load_condition_data(
        Path(args.data_dir), condition
    )
    scorer = make_scorer(
        df_train, df_val, feature_cols,
        args.w_fidelity, args.w_accuracy, args.w_comprehensibility,
        args.min_coverage,
    )

    n_gen = args.n_gen or cfg.ga.generations
    n_runs = args.n_runs or cfg.ga.runs
    base_seed = args.seed if args.seed is not None else cfg.seed

    print(
        f"[ga] condition={condition} fidelity_split={FIDELITY_SPLIT} "
        f"accuracy_split={ACCURACY_SPLIT} features={len(feature_cols)} "
        f"train_rows={len(df_train)} val_rows={len(df_val)} gens={n_gen} runs={n_runs}"
    )

    all_run_fronts = []
    for run_id in range(1, n_runs + 1):
        seed = base_seed + run_id
        pop_size = random.Random(seed).randint(cfg.ga.pop_min, cfg.ga.pop_max)
        toolbox = build_toolbox(len(feature_cols), scorer)
        front = run_ga(
            toolbox, scorer, pop_size, n_gen, args.cxpb, args.mutpb,
            args.min_coverage, args.max_age, seed,
        )
        all_run_fronts.append((run_id, front))
        print(f"[ga] run {run_id}/{n_runs} seed={seed} pop={pop_size} front_size={len(front)}")

    return aggregate_runs(all_run_fronts, scorer, feature_cols, feature_map)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--condition", choices=["baseline", "sam"], required=True)
    ap.add_argument("--config", default="configs/pipeline.yaml")
    ap.add_argument("--data-dir", default="artifacts/data")
    ap.add_argument("--out-dir", default="artifacts/rules")
    ap.add_argument("--n-runs", type=int, default=None, help="override pipeline.yaml ga.runs")
    ap.add_argument("--n-gen", type=int, default=None, help="override pipeline.yaml ga.generations")
    ap.add_argument("--seed", type=int, default=None, help="override pipeline.yaml seed")
    ap.add_argument("--cxpb", type=float, default=0.8)
    ap.add_argument("--mutpb", type=float, default=0.2)
    ap.add_argument("--min-coverage", type=float, default=0.02,
                     help="prune/reject rules covering less than this fraction of train rows (C6: 2% floor)")
    ap.add_argument("--max-age", type=int, default=DEFAULT_MAX_AGE,
                     help="aging-evolution: replace individuals older than this many generations")
    ap.add_argument("--w-fidelity", type=float, default=DEFAULT_W_FIDELITY)
    ap.add_argument("--w-accuracy", type=float, default=DEFAULT_W_ACCURACY)
    ap.add_argument("--w-comprehensibility", type=float, default=DEFAULT_W_COMPREHENSIBILITY)
    args = ap.parse_args()

    config_path = Path(args.config)
    cfg = load_config(config_path)

    rules = extract_rules_ga(args.condition, cfg, args)
    config_hash_val = config_hash(config_path)
    n_runs = args.n_runs or cfg.ga.runs
    out_path = Path(args.out_dir) / f"ga_{args.condition}.json"

    with open(Path(args.data_dir) / "feature_map.json") as f:
        feature_map = json.load(f)

    write_rule_pool(out_path, args.condition, n_runs, config_hash_val, rules, feature_map)
    print(f"[ga] wrote {len(rules)} non-dominated rules -> {out_path}")


if __name__ == "__main__":
    main()
