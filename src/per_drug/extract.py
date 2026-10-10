"""Run one (drug, arm, extractor, fold) cell: n_seeds independent GA / NSGA-II runs
on the fold's extraction rows, pooled exactly as the global runs pool them
(ga.aggregate_runs / nsga.aggregate_runs), then scored on the held-out fold.

The search code (operators, scorers, pruning, aging, aggregation) is the
existing src.extraction.{ga,nsga}; only the data handed to them and the
coverage floor (min_cover_n) differ. The closed-form fast scorer from
run_logged is installed (identical results, verified by
check_scorer_equivalence.py).
"""
from __future__ import annotations

import random
import time

from src.extraction import ga as ga_mod
from src.extraction import nsga as nsga_mod
from src.extraction.common import decode_individual, resolve_genes
from src.extraction.coverage import coverage_floor
from src.extraction.run_logged import install_fast_scorer
from src.per_drug import data as D
from src.per_drug import scoring as S
from src.per_drug.config import DESIGN, design_hash, git_info

install_fast_scorer(ga_mod, nsga_mod)


def _front_rules(front, scorer_fn, feature_cols, feature_map):
    """Unique genotypes of one seed's front -> rule dicts (conditions, class, genes)."""
    out, seen = [], set()
    for ind in front:
        key = tuple(ind)
        if key in seen:
            continue
        seen.add(key)
        pred = scorer_fn(ind)[-1]          # predicted_class is last in both scorers' tuples
        if pred is None:
            continue
        conds = decode_individual(ind, feature_cols)
        out.append({"conditions": conds, "predicted_class": pred, "genes": resolve_genes(conds, feature_map)})
    return out


def run_cell(dd: D.DrugData, arm: str, extractor: str, fold: int, n_seeds: int,
             n_gen: int | None = None, min_cover_n: int | None = None) -> dict:
    """Returns the cell record (JSON-serialisable). Deterministic given its inputs
    apart from the wall-clock fields."""
    n_gen = DESIGN["n_gen"] if n_gen is None else n_gen
    min_cover_n = DESIGN["min_cover_n"] if min_cover_n is None else min_cover_n
    parts = D.fold_parts(dd, fold)
    cp = f"class_pred_{arm}"
    df_fid = parts.fid.copy()
    df_fid["class_pred"] = df_fid[cp]
    df_val = parts.inner_val.copy()
    df_val["class_pred"] = df_val[cp]
    floor = coverage_floor(0.0, min_cover_n, len(df_fid))
    fcols, fmap = dd.feature_cols, dd.feature_map
    t_cell = time.perf_counter()

    if extractor == "ga":
        g = DESIGN["ga"]
        scorer = ga_mod.make_scorer(df_fid, df_val, fcols, g["w_fidelity"], g["w_accuracy"],
                                    g["w_comprehensibility"], floor)
        toolbox = ga_mod.build_toolbox(len(fcols), scorer)
        score_one = scorer
    elif extractor == "nsga":
        scorer = nsga_mod.make_scorer(df_fid, fcols, "class_pred", min_coverage=floor)
        toolbox = nsga_mod.build_toolbox(len(fcols), scorer)
        score_one = scorer
    else:
        raise ValueError(extractor)

    fronts, seed_rows, per_seed_rules = [], [], {}
    for run_id in range(1, n_seeds + 1):
        seed = DESIGN["base_seed"] + run_id
        t0 = time.perf_counter()
        if extractor == "ga":
            pop = random.Random(seed).randint(g["pop_min"], g["pop_max"])
            front = ga_mod.run_ga(toolbox, scorer, pop, n_gen, g["cxpb"], g["mutpb"], floor, g["max_age"], seed)
        else:
            pop = DESIGN["nsga"]["pop"]
            front = nsga_mod.run_nsga2(toolbox, pop, n_gen, DESIGN["nsga"]["cxpb"], DESIGN["nsga"]["mutpb"], seed)
        dt = time.perf_counter() - t0
        fronts.append((run_id, front))
        rules = _front_rules(front, score_one, fcols, fmap)
        per_seed_rules[seed] = rules
        seed_rows.append({"run_id": run_id, "seed": seed, "pop_size": pop, "front_size": len(front),
                          "n_rules": len(rules), "seconds": round(dt, 3),
                          "genes": sorted({x for r in rules for x in r["genes"]})})

    agg = (ga_mod if extractor == "ga" else nsga_mod).aggregate_runs(fronts, scorer, fcols, fmap)
    pool = []
    for i, r in enumerate(agg, start=1):
        pool.append({"id": f"{dd.drug}|{arm}|{extractor}|f{fold}|{i:04d}", **r})

    cell = {
        "key": {"drug": dd.drug, "arm": arm, "extractor": extractor, "fold": fold, "n_seeds": n_seeds,
                "n_gen": n_gen, "min_cover_n": min_cover_n},
        "manifest": {**git_info(), "design_hash": design_hash(), "base_seed": DESIGN["base_seed"]},
        "sizes": {"fid": len(df_fid), "inner_val": len(df_val), "heldout": len(parts.heldout),
                  "floor_fraction": floor},
        "seed_rows": seed_rows,
        "per_seed_rules": {str(k): v for k, v in per_seed_rules.items()},
        "pool": pool,
        "wall_clock_s": round(time.perf_counter() - t_cell, 3),
    }
    cell["heldout"] = S.score_pool(pool, parts.heldout, arm)
    return cell
