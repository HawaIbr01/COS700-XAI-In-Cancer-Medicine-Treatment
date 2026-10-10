"""
Check that the closed-form fast scorer in run_logged.py returns EXACTLY the
same numbers as the original ga.py / nsga.py scorers, on a large set of random
genotypes (sparse, mutated-dense, crossover children) for both conditions.

Writes a short report to outputs/logs/scorer_equivalence.txt.
"""
import copy
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.extraction import ga as ga_mod, nsga as nsga_mod          # noqa: E402
from src.extraction import common                                   # noqa: E402
from src.extraction.run_logged import (DATA_DIR, GA_WEIGHTS, MIN_COVERAGE,  # noqa: E402
                                       fast_fidelity_f1, fast_predict_with_rule)

N = int(sys.argv[1]) if len(sys.argv) > 1 else 250
lines = []
orig_ga_pred, orig_ga_f1 = ga_mod.predict_with_rule, ga_mod.fidelity_f1
orig_ns_pred = nsga_mod.predict_with_rule


def genotypes(n_features, n, seed):
    rnd = random.Random(seed)
    random.seed(seed)
    out = []
    while len(out) < n:
        a = common.sparse_individual(n_features)
        b = common.sparse_individual(n_features)
        kind = rnd.random()
        if kind < 0.4:
            out.append(a)
        elif kind < 0.7:
            c1, c2 = common.cxTwoPoint_partitions(copy.deepcopy(a), copy.deepcopy(b))
            out += [c1, c2]
        else:
            m = copy.deepcopy(a)
            common.mutate_partition(m, indpb=0.05)
            out.append(m)
    return out[:n]


def compare(a, b):
    return all((x == y) if not isinstance(x, float) else (x == y) for x, y in zip(a, b)) and len(a) == len(b)


bad = 0
for cond in ("baseline", "sam"):
    # ---- GA scorer ----
    df_tr, df_va, fcols, fmap = ga_mod.load_condition_data(DATA_DIR, cond)
    gs = genotypes(len(fcols), N, 1)
    ga_mod.predict_with_rule, ga_mod.fidelity_f1 = orig_ga_pred, orig_ga_f1
    ref = [ga_mod._score_genotype(list(g), df_tr, df_va, fcols, *GA_WEIGHTS, MIN_COVERAGE) for g in gs]
    ga_mod.predict_with_rule, ga_mod.fidelity_f1 = fast_predict_with_rule, fast_fidelity_f1
    new = [ga_mod._score_genotype(list(g), df_tr, df_va, fcols, *GA_WEIGHTS, MIN_COVERAGE) for g in gs]
    n_cov = sum(1 for r in ref if r[6] is not None)
    diff = sum(0 if compare(r, n) else 1 for r, n in zip(ref, new))
    bad += diff
    lines.append(f"GA   {cond}: {len(gs)} genotypes ({n_cov} above the 2% floor): {diff} mismatches")

    # ---- NSGA-II scorer ----
    df, fcols, fmap, cc = nsga_mod.load_condition_data(DATA_DIR, cond, "train")
    gs = genotypes(len(fcols), N, 2)
    nsga_mod.predict_with_rule = orig_ns_pred
    ref = [nsga_mod._score_genotype(list(g), df, fcols, cc, MIN_COVERAGE) for g in gs]
    nsga_mod.predict_with_rule = fast_predict_with_rule
    new = [nsga_mod._score_genotype(list(g), df, fcols, cc, MIN_COVERAGE) for g in gs]
    n_cov = sum(1 for r in ref if r[3] is not None and r[2] >= MIN_COVERAGE)
    diff = sum(0 if compare(r, n) else 1 for r, n in zip(ref, new))
    bad += diff
    lines.append(f"NSGA {cond}: {len(gs)} genotypes ({n_cov} above the 2% floor): {diff} mismatches")

lines.append("RESULT: " + ("fast scorer is bit-identical to the original on every tested genotype"
                           if bad == 0 else f"{bad} MISMATCHES - do not use the fast scorer"))
out = ROOT / "outputs" / "logs"
out.mkdir(parents=True, exist_ok=True)
(out / "scorer_equivalence.txt").write_text("\n".join(lines) + "\n")
print("\n".join(lines))
