"""Per-generation convergence traces for one cell (monitoring only).

Hooked in without touching src/extraction: ga.run_ga / nsga.run_nsga2 accept an opt-in `logbook` and then call
`tools.Statistics(...).compile(population)` once per generation, generation 0 included (n_gen + 1 calls). We
swap deap.tools.Statistics for a subclass whose compile() hands the population to a Tracer. The Tracer only
READS: fitness values, the scorer's memoised results (every population member was already scored by
toolbox.evaluate, so these are cache hits) and the held-out fold's arrays. It makes no RNG calls and never
writes to an individual, so selection and every random draw are unchanged. Held-out values are logged only;
nothing reads them back.

Definitions (also written into every cell under traces.definitions):
  GA   best_/mean_fitness     weighted scalar fitness over the whole population
       best_fidelity/accuracy max over individuals clearing the coverage floor (None if none)
       mean_fidelity          mean over those individuals
       mean_len_best          mean rule length of the individuals tied for best fitness
       mean_len_pop           mean rule length over the population
  NSGA front0_*               first non-dominated front of the population (fitness = (fidelity, length))
       hypervolume            2-D, fidelity maximised from 0, length minimised to the reference length
  both unique_genotypes, evals_cumulative (= toolbox.evaluate calls so far in this seed, incl. gen 0)
       held_out_fidelity/coverage of the best-fitness (GA) / max-fidelity front member (NSGA; ties -> shorter
       rule, then first in front order): fidelity = macro-F1 of the rule's train-derived class vs the network's
       class_pred on the covered held-out rows (closed form, identical to the pool scoring); coverage = covered
       fraction of the held-out rows. None where undefined.
"""
from __future__ import annotations

from contextlib import contextmanager

import numpy as np
from deap import tools

from src.extraction.common import PARTITION_LABELS

ND = 6   # decimals kept for floats in the cell JSON


def _r(x):
    return None if x is None else round(float(x), ND)


class _Hooked(tools.Statistics):
    tracer = None

    def compile(self, data):
        rec = super().compile(data)
        if _Hooked.tracer is not None:
            _Hooked.tracer.on_generation(list(data))
        return rec


@contextmanager
def hooked():
    """Swap deap.tools.Statistics for the hooked subclass while a seed runs."""
    orig = tools.Statistics
    tools.Statistics = _Hooked
    try:
        yield
    finally:
        tools.Statistics = orig
        _Hooked.tracer = None


def hypervolume_2d(points, ref_len: float) -> float:
    """Area dominated by (fidelity max, length min) points w.r.t. reference (fidelity 0, length ref_len)."""
    pts = sorted((l, f) for f, l in points if l < ref_len and f > 0)
    hv, best = 0.0, 0.0
    for i, (l, f) in enumerate(pts):
        best = max(best, f)
        nxt = pts[i + 1][0] if i + 1 < len(pts) else ref_len
        hv += best * (nxt - l)
    return hv


class Tracer:
    def __init__(self, extractor, scorer, heldout_df, arm, feature_cols):
        self.extractor, self.scorer, self.feature_cols = extractor, scorer, feature_cols
        self.n_feat = len(feature_cols)
        self.held_n = len(heldout_df)
        self.held_pred = heldout_df[f"class_pred_{arm}"].to_numpy()
        self._held = heldout_df
        self._col = {}
        self._ho = {}
        self.records, self.evals, self.gen = [], 0, 0

    def start_seed(self):
        self.records, self.evals, self.gen = [], 0, 0
        _Hooked.tracer = self

    def count_evals(self, evaluate):
        def counted(ind):
            self.evals += 1
            return evaluate(ind)
        return counted

    # ---- held-out monitoring (read-only, memoised per genotype)
    def _heldout(self, ind, pred):
        if pred is None or self.held_n == 0:
            return None, None
        key = tuple(ind)
        if key not in self._ho:
            mask = np.ones(self.held_n, dtype=bool)
            for i, v in enumerate(key):
                if v != -1:
                    if i not in self._col:
                        self._col[i] = self._held[self.feature_cols[i]].to_numpy()
                    mask &= (self._col[i] == PARTITION_LABELS[v])
            n = int(mask.sum())
            if n == 0:
                self._ho[key] = (None, 0.0)
            else:
                k = int(np.count_nonzero(self.held_pred[mask] == pred))
                self._ho[key] = (1.0 if k == n else 0.5 * (2.0 * k / (n + k)), n / self.held_n)
        return self._ho[key]

    def on_generation(self, pop):
        rec = (self._ga if self.extractor == "ga" else self._nsga)(pop)
        rec = {"gen": self.gen, "evals_cumulative": self.evals, **rec}
        self.records.append(rec)
        self.gen += 1

    def _ga(self, pop):
        sc = [self.scorer(i) for i in pop]            # (weighted, fid, acc, comp, length, cov, pred)
        w = [s[0] for s in sc]
        bw = max(w)
        tied = [k for k, x in enumerate(w) if x == bw]
        valid = [s for s in sc if s[6] is not None]
        b = tied[0]
        hf, hc = self._heldout(pop[b], sc[b][6])
        return {
            "best_fitness": _r(bw), "mean_fitness": _r(sum(w) / len(w)),
            "best_fidelity": _r(max(s[1] for s in valid)) if valid else None,
            "mean_fidelity": _r(sum(s[1] for s in valid) / len(valid)) if valid else None,
            "best_accuracy": _r(max(s[2] for s in valid)) if valid else None,
            "mean_len_best": _r(sum(sc[k][4] for k in tied) / len(tied)),
            "mean_len_pop": _r(sum(s[4] for s in sc) / len(sc)),
            "unique_genotypes": len({tuple(i) for i in pop}),
            "held_out_fidelity": _r(hf), "held_out_coverage": _r(hc),
        }

    def _nsga(self, pop):
        front0 = tools.sortNondominated(pop, len(pop), first_front_only=True)[0]   # read-only
        fv = [i.fitness.values for i in front0]
        best = min(range(len(front0)), key=lambda k: (-fv[k][0], fv[k][1], k))
        hf, hc = self._heldout(front0[best], self.scorer(front0[best])[-1])
        allv = [i.fitness.values for i in pop]
        return {
            "front0_size": len(front0),
            "front0_max_fidelity": _r(max(f for f, _ in fv)), "front0_min_len": int(min(l for _, l in fv)),
            "hypervolume": _r(hypervolume_2d(fv, self.n_feat)),
            "mean_fidelity_pop": _r(sum(f for f, _ in allv) / len(allv)),
            "mean_len_pop": _r(sum(l for _, l in allv) / len(allv)),
            "unique_genotypes": len({tuple(i) for i in pop}),
            "held_out_fidelity": _r(hf), "held_out_coverage": _r(hc),
        }


def meta(extractor: str, n_gen: int, n_features: int) -> dict:
    m = {"version": 1, "n_records_per_seed": n_gen + 1,
         "definitions": "see src/per_drug/traces.py docstring; held_out_* are monitoring only"}
    if extractor == "nsga":
        m["hypervolume_reference_point"] = {"fidelity": 0.0, "length": n_features}
    return m
