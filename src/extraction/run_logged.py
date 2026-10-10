"""
Logged re-run of the GA / NSGA-II rule extractors (Stage 5, evaluation arm).

PURPOSE
    ga.py / nsga.py already support an opt-in per-generation `logbook`, but
    their `main()` never passes one and never saves the per-run fronts, so the
    original 30-run extractions left no record of how the search evolved. This
    driver re-runs the SAME extraction (same seeds, same operators, same
    parameters, same scorer) while saving, for every run and every generation:

      * the standard DEAP statistics (min / max / avg of each fitness value),
      * extra diagnostics computed from the population at that generation
        (distinct genotypes, mean rule length, share of the population that
        clears the 2% coverage floor, best rule's fidelity/length/coverage,
        NSGA-II front-0 size and distinct-genotype count, GA population age),
      * each run's final front (the rules that go into the pooled output).

    It does NOT modify ga.py, nsga.py or common.py. The extra diagnostics are
    collected by swapping deap.tools.Statistics for a subclass whose compile()
    forwards the population to a callback; the callback only reads individuals
    and calls the (cached) scorer, so it neither consumes random numbers nor
    changes any individual -- the search itself is untouched.

REPRODUCIBILITY CHECK
    The pooled rules are written through the modules' own aggregate_runs() /
    write_rule_pool() and compared with the previously shipped pools
    (--compare-with). If seeds and code are identical the pools must match
    exactly; the comparison result is written to <out-dir>/logs/.

Usage (project root):
    python -m src.extraction.run_logged --extractor ga   --condition baseline
    python -m src.extraction.run_logged --extractor nsga --condition sam
    python -m src.extraction.run_logged --extractor ga --condition sam --n-runs 2 --n-gen 10 --tag smoke
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter
from pathlib import Path

import deap.tools as dtools
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

# ---- parameters that the original CLI/config supplied (see ga.py / nsga.py
# defaults and the 2026-09-30 run logs: seed 42 -> run seeds 43..72, GA pop
# drawn from 35..45, NSGA-II pop 40, 50 generations, 30 runs, 2% floor) ------
BASE_SEED = 42
GA_POP_MIN, GA_POP_MAX = 35, 45
NSGA_POP = 40
N_GEN = 50
N_RUNS = 30
MIN_COVERAGE = 0.02
GA_CXPB, GA_MUTPB, GA_MAX_AGE = 0.8, 0.2, 8
GA_WEIGHTS = (0.45, 0.35, 0.20)
NSGA_CXPB, NSGA_MUTPB = 0.9, 0.2

FAST = [True]
DATA_DIR = ROOT / "data" / "processed" / "real_extraction" / "data"


# --------------------------------------------------------------------------
# Hook: forward the population to a callback every time stats.compile() runs
# --------------------------------------------------------------------------

class HookedStatistics(dtools.Statistics):
    hook = None

    def compile(self, data):
        rec = super().compile(data)
        if HookedStatistics.hook is not None:
            HookedStatistics.hook(list(data), rec)
        return rec



# --------------------------------------------------------------------------
# Optional fast scorer (identical results, ~10x quicker)
# --------------------------------------------------------------------------
# Profiling showed ~90% of the run time is sklearn's f1_score + pandas .mode()
# on 10^5-row string arrays, called once per novel genotype. In this project a
# rule ALWAYS predicts one constant class, for which macro-F1 has a closed
# form: with n covered rows of which k belong to the predicted class,
#     macro-F1 = 1.0              if k == n   (only one label present anywhere)
#              = k / (n + k)      otherwise   (other class has F1 = 0)
# which is exactly what sklearn computes (class-c F1 = 2k/(n+k), other = 0,
# averaged over the two labels). The majority vote tie-break replicates
# pandas' Series.mode().iat[0] (ties -> alphabetically first label =
# 'resistant'). Equivalence to the original functions is checked on random
# genotypes by check_scorer_equivalence.py, and end-to-end by comparing the
# pooled rules with the shipped pools.

import numpy as np

_LABELS = ("resistant", "sensitive")


def fast_fidelity_f1(rule_class_preds, model_class_preds):
    n = len(model_class_preds)
    if len(rule_class_preds) != n:
        raise ValueError("length mismatch")
    if n == 0:
        raise ValueError("cannot compute fidelity on an empty rule coverage set")
    c = rule_class_preds.iloc[0] if hasattr(rule_class_preds, "iloc") else rule_class_preds[0]
    vals = np.asarray(model_class_preds)
    k = int(np.count_nonzero(vals == c))
    if k == n:
        return 1.0
    return 0.5 * (2.0 * k / (n + k))


def fast_predict_with_rule(covered_preds):
    covered = pd.Series(covered_preds)
    n = len(covered)
    if n == 0:
        return None, 0.0
    vals = covered.to_numpy()
    k_res = int(np.count_nonzero(vals == "resistant"))
    k_sen = int(np.count_nonzero(vals == "sensitive"))
    if k_res + k_sen != n:
        raise ValueError("unexpected class label (fast scorer supports sensitive/resistant only)")
    predicted = "resistant" if k_res >= k_sen else "sensitive"
    k = k_res if predicted == "resistant" else k_sen
    fid = 1.0 if k == n else 0.5 * (2.0 * k / (n + k))
    return predicted, float(fid)


def install_fast_scorer(*modules):
    """Point the extractor modules' imported names at the closed-form versions."""
    for m in modules:
        if hasattr(m, "predict_with_rule"):
            m.predict_with_rule = fast_predict_with_rule
        if hasattr(m, "fidelity_f1"):
            m.fidelity_f1 = fast_fidelity_f1



def _f(x):
    return None if x is None else float(x)


def ensure_response_classes(data_dir: Path) -> None:
    """ga.py needs <data_dir>/response_classes.json (per_row_class). It is built
    exactly as the team's own generate_response_classes.py does: straight from
    the parquet's existing row_id / response_class columns, no re-inference."""
    out = data_dir / "response_classes.json"
    if out.exists():
        return
    df = pd.read_parquet(data_dir / "gdsc_processed.parquet", columns=["row_id", "response_class"])
    payload = {
        "threshold": "per_drug_median_train",
        "target": "ln_ic50",
        "classes": ["sensitive", "resistant"],
        "threshold_source": "per-drug train-split median (see gdsc_processed.parquet response_class)",
        "per_row_class": {str(int(r)): c for r, c in zip(df["row_id"], df["response_class"])},
    }
    with open(out, "w", encoding="utf-8") as f:
        json.dump(payload, f)


def read_config_hash(compare_with: Path | None) -> str:
    if compare_with and compare_with.exists():
        return json.load(open(compare_with))["meta"]["config_hash"]
    return "unknown"


# --------------------------------------------------------------------------
# GA
# --------------------------------------------------------------------------

def run_ga_logged(condition, n_runs, n_gen, out_dir, log):
    import random
    from src.extraction import ga as ga_mod
    if FAST[0]:
        install_fast_scorer(ga_mod)

    df_train, df_val, feature_cols, feature_map = ga_mod.load_condition_data(DATA_DIR, condition)
    scorer = ga_mod.make_scorer(df_train, df_val, feature_cols, *GA_WEIGHTS, MIN_COVERAGE)
    log(f"[ga] condition={condition} features={len(feature_cols)} train_rows={len(df_train)} "
        f"val_rows={len(df_val)} gens={n_gen} runs={n_runs}")

    gen_rows, run_rows, run_fronts = [], [], {}
    state = {"gen": -1, "run": None, "seed": None}

    def hook(pop, rec):
        state["gen"] += 1
        sc = [scorer(i) for i in pop]  # (weighted, fid, acc, comp, length, cov, pred)
        valid = [s for s in sc if s[6] is not None]
        lens = [s[4] for s in sc]
        best = max(sc, key=lambda s: s[0])
        row = {
            "run_id": state["run"], "seed": state["seed"], "gen": state["gen"], "popsize": len(pop),
            "fit_min": _f(rec["min"]), "fit_max": _f(rec["max"]), "fit_avg": _f(rec["avg"]),
            "n_unique_genotypes": len({tuple(i) for i in pop}),
            "mean_length": sum(lens) / len(lens), "max_length": max(lens),
            "frac_above_floor": len(valid) / len(pop),
            "mean_fidelity_valid": (sum(s[1] for s in valid) / len(valid)) if valid else None,
            "best_fidelity_valid": max((s[1] for s in valid), default=None),
            "mean_coverage_valid": (sum(s[5] for s in valid) / len(valid)) if valid else None,
            "best_ind_fidelity": best[1] if best[6] is not None else None,
            "best_ind_accuracy": best[2] if best[6] is not None else None,
            "best_ind_length": best[4],
            "best_ind_coverage": best[5],
            "mean_age": sum(getattr(i, "age", 0) for i in pop) / len(pop),
            "n_age0": sum(1 for i in pop if getattr(i, "age", 0) == 0),
        }
        gen_rows.append(row)

    HookedStatistics.hook = hook
    dtools.Statistics = HookedStatistics

    all_run_fronts = []
    for run_id in range(1, n_runs + 1):
        seed = BASE_SEED + run_id
        pop_size = random.Random(seed).randint(GA_POP_MIN, GA_POP_MAX)
        toolbox = ga_mod.build_toolbox(len(feature_cols), scorer)
        state.update(gen=-1, run=run_id, seed=seed)
        logbook = dtools.Logbook()
        t0 = time.time()
        front = ga_mod.run_ga(toolbox, scorer, pop_size, n_gen, GA_CXPB, GA_MUTPB,
                              MIN_COVERAGE, GA_MAX_AGE, seed, logbook=logbook)
        dt = time.time() - t0
        # cross-check: logbook == hook record for the standard stats
        mine = [r for r in gen_rows if r["run_id"] == run_id]
        assert len(mine) == len(logbook) == n_gen + 1
        for r, lb in zip(mine, logbook):
            assert abs(r["fit_max"] - lb["max"]) < 1e-12 and abs(r["fit_avg"] - lb["avg"]) < 1e-12
        all_run_fronts.append((run_id, front))
        run_fronts[run_id] = []
        for ind in front:
            w, fid, acc, comp, length, cov, pred = scorer(ind)
            run_fronts[run_id].append({
                "conditions": ga_mod.decode_individual(ind, feature_cols), "predicted_class": pred,
                "length": length, "fidelity_f1": fid, "accuracy": acc, "coverage": cov, "weighted_fitness": w,
            })
        run_rows.append({"run_id": run_id, "seed": seed, "pop_size": pop_size, "front_size": len(front),
                         "best_weighted": max(r["fit_max"] for r in mine),
                         "final_best_weighted": mine[-1]["fit_max"], "seconds": round(dt, 1)})
        log(f"[ga] run {run_id}/{n_runs} seed={seed} pop={pop_size} front_size={len(front)} ({dt:.0f}s)")

    rules = ga_mod.aggregate_runs(all_run_fronts, scorer, feature_cols, feature_map)
    return rules, feature_map, gen_rows, run_rows, run_fronts, ga_mod


# --------------------------------------------------------------------------
# NSGA-II
# --------------------------------------------------------------------------

def run_nsga_logged(condition, n_runs, n_gen, out_dir, log, pop_size=NSGA_POP):
    from src.extraction import nsga as nsga_mod
    if FAST[0]:
        install_fast_scorer(nsga_mod)

    df, feature_cols, feature_map, class_col = nsga_mod.load_condition_data(DATA_DIR, condition, "train")
    scorer = nsga_mod.make_scorer(df, feature_cols, class_col, min_coverage=MIN_COVERAGE)
    toolbox = nsga_mod.build_toolbox(len(feature_cols), scorer)
    log(f"[nsga] condition={condition} features={len(feature_cols)} rows={len(df)} pop={pop_size} "
        f"gens={n_gen} runs={n_runs} min_coverage={MIN_COVERAGE}")

    gen_rows, run_rows, run_fronts = [], [], {}
    state = {"gen": -1, "run": None, "seed": None}

    def hook(pop, rec):
        state["gen"] += 1
        sc = [scorer(i) for i in pop]  # (fidelity, length, coverage, pred)
        above = [s for s in sc if s[3] is not None and s[2] >= MIN_COVERAGE and s[1] > 0]
        lens = [s[1] for s in sc]
        front0 = dtools.sortNondominated(pop, len(pop), first_front_only=True)[0]
        best = max(above, key=lambda s: (s[0], -s[1])) if above else None
        row = {
            "run_id": state["run"], "seed": state["seed"], "gen": state["gen"], "popsize": len(pop),
            "fid_min": rec["min"][0], "fid_max": rec["max"][0], "fid_avg": rec["avg"][0],
            "len_min": rec["min"][1], "len_max": rec["max"][1], "len_avg": rec["avg"][1],
            "n_unique_genotypes": len({tuple(i) for i in pop}),
            "front0_size": len(front0), "front0_unique": len({tuple(i) for i in front0}),
            "frac_above_floor": len(above) / len(pop),
            "mean_fidelity_above": (sum(s[0] for s in above) / len(above)) if above else None,
            "mean_coverage_above": (sum(s[2] for s in above) / len(above)) if above else None,
            "best_above_fidelity": best[0] if best else None,
            "best_above_length": best[1] if best else None,
            "best_above_coverage": best[2] if best else None,
        }
        gen_rows.append(row)

    HookedStatistics.hook = hook
    dtools.Statistics = HookedStatistics

    all_run_fronts = []
    for run_id in range(1, n_runs + 1):
        seed = BASE_SEED + run_id
        state.update(gen=-1, run=run_id, seed=seed)
        logbook = dtools.Logbook()
        t0 = time.time()
        front = nsga_mod.run_nsga2(toolbox, pop_size, n_gen, NSGA_CXPB, NSGA_MUTPB, seed, logbook=logbook)
        dt = time.time() - t0
        mine = [r for r in gen_rows if r["run_id"] == run_id]
        assert len(mine) == len(logbook) == n_gen + 1
        for r, lb in zip(mine, logbook):
            assert abs(r["fid_max"] - lb["max"][0]) < 1e-12 and abs(r["len_avg"] - lb["avg"][1]) < 1e-12
        all_run_fronts.append((run_id, front))
        counts = Counter(tuple(i) for i in front)
        run_fronts[run_id] = []
        for key, n_copies in counts.items():
            ind = next(i for i in front if tuple(i) == key)
            fid, length, cov, pred = scorer(ind)
            run_fronts[run_id].append({
                "conditions": nsga_mod.decode_individual(ind, feature_cols), "predicted_class": pred,
                "length": length, "fidelity_f1": fid, "coverage": cov, "n_copies_in_front": n_copies,
            })
        run_rows.append({"run_id": run_id, "seed": seed, "pop_size": pop_size, "front_size": len(front),
                         "front_unique_genotypes": len(counts),
                         "best_fidelity_final": max((r["fidelity_f1"] for r in run_fronts[run_id]), default=None),
                         "seconds": round(dt, 1)})
        log(f"[nsga] run {run_id}/{n_runs} seed={seed} front_size={len(front)} "
            f"unique={len(counts)} ({dt:.0f}s)")

    rules = nsga_mod.aggregate_runs(all_run_fronts, scorer, feature_cols, feature_map)
    return rules, feature_map, gen_rows, run_rows, run_fronts, nsga_mod


# --------------------------------------------------------------------------
# Reproducibility comparison
# --------------------------------------------------------------------------

def _canon(rule):
    conds = tuple(sorted((c["feature"], c["partition"]) for c in rule["conditions"]))
    return conds, rule["predicted_class"], round(rule["fidelity_f1"], 4), round(rule["coverage"], 4), tuple(rule["run_ids"])


def compare_pools(new_path: Path, old_path: Path) -> str:
    new = json.load(open(new_path))["rules"]
    old = json.load(open(old_path))["rules"]
    a, b = {_canon(r) for r in new}, {_canon(r) for r in old}
    same = a == b
    lines = [f"compare {new_path.name} vs shipped {old_path.name}: "
             f"{'IDENTICAL rule sets' if same else 'DIFFERENT'} "
             f"(new={len(new)} rules, shipped={len(old)} rules)"]
    if not same:
        lines.append(f"  only in new: {len(a - b)}   only in shipped: {len(b - a)}")
        for x in sorted(a - b)[:5]:
            lines.append(f"    new-only: {x}")
        for x in sorted(b - a)[:5]:
            lines.append(f"    shipped-only: {x}")
    return "\n".join(lines)


# --------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--extractor", choices=["ga", "nsga"], required=True)
    ap.add_argument("--condition", choices=["baseline", "sam"], required=True)
    ap.add_argument("--n-runs", type=int, default=N_RUNS)
    ap.add_argument("--n-gen", type=int, default=N_GEN)
    ap.add_argument("--pop", type=int, default=NSGA_POP, help="NSGA-II population (GA draws 35..45 per seed)")
    ap.add_argument("--out-dir", default="outputs")
    ap.add_argument("--no-fast", action="store_true", help="use the original sklearn/pandas scorer (slow)")
    ap.add_argument("--tag", default="", help="suffix for smoke tests, e.g. 'smoke'")
    ap.add_argument("--compare-with", default=None,
                    help="shipped pool JSON to compare against (default artifacts/rules_orig/<ext>_<cond>.json)")
    args = ap.parse_args()

    FAST[0] = not args.no_fast
    out = Path(args.out_dir)
    for sub in ("logs", "rules", "runs"):
        (out / sub).mkdir(parents=True, exist_ok=True)
    name = f"{args.extractor}_{args.condition}" + (f"_{args.tag}" if args.tag else "")
    log_path = out / "logs" / f"{name}_rerun.log"
    logf = open(log_path, "w", encoding="utf-8")

    def log(msg):
        print(msg, flush=True)
        logf.write(msg + "\n")
        logf.flush()

    ensure_response_classes(DATA_DIR)
    compare_with = Path(args.compare_with) if args.compare_with else \
        ROOT / "artifacts" / "rules_orig" / f"{args.extractor}_{args.condition}.json"
    cfg_hash = read_config_hash(compare_with)

    t_start = time.time()
    if args.extractor == "ga":
        rules, feature_map, gen_rows, run_rows, run_fronts, mod = run_ga_logged(
            args.condition, args.n_runs, args.n_gen, out, log)
    else:
        rules, feature_map, gen_rows, run_rows, run_fronts, mod = run_nsga_logged(
            args.condition, args.n_runs, args.n_gen, out, log, pop_size=args.pop)

    pool_path = out / "rules" / f"{name}.json"
    mod.write_rule_pool(pool_path, args.condition, args.n_runs, cfg_hash, rules, feature_map)
    log(f"[{args.extractor}] wrote {len(rules)} non-dominated rules -> {pool_path}")

    pd.DataFrame(gen_rows).to_csv(out / "logs" / f"{name}_generations.csv", index=False)
    pd.DataFrame(run_rows).to_csv(out / "logs" / f"{name}_runs.csv", index=False)
    with open(out / "runs" / f"{name}_run_fronts.json", "w", encoding="utf-8") as f:
        json.dump(run_fronts, f, indent=1)

    if not args.tag:
        msg = compare_pools(pool_path, compare_with)
        log(msg)
        with open(out / "logs" / f"{name}_repro_check.txt", "w") as f:
            f.write(msg + "\n")
    log(f"[{args.extractor}] total wall time {time.time() - t_start:.0f}s")
    logf.close()


if __name__ == "__main__":
    main()
