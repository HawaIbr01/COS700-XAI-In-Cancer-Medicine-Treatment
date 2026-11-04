"""POST-HOC DIAGNOSTICS (decided after seeing the pilot results; not part of the frozen design).

Read-only analysis of the ORIGINAL global rule pools (artifacts/rules/{ga,nsga}_{baseline,sam}.json, 19 rules):
 1. per-rule agreement (accuracy vs the network's class_pred) and coverage on the train / val / test splits,
 2. per-pool share of test rows covered by >= 1 rule and by exactly one rule,
 3. matched random-rule baseline (same length, global coverage floor 2% of train rows, 200 draws per rule).
Output: artifacts/per_drug/diagnostics/posthoc_global_*.csv. Uses src.per_drug.diagnostics.fid_from_counts
(closed form of src.common.metrics.fidelity_f1).
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

from src.per_drug.diagnostics import fid_from_counts, LEVELS

DATA = Path("data/processed/real_extraction/data")
PRED = Path("data/processed/real_extraction/predictions")
RULES = Path("artifacts/rules")
OUT = Path("artifacts/per_drug/diagnostics")
POOLS = ["ga_baseline", "ga_sam", "nsga_baseline", "nsga_sam"]
FLOOR, N_RANDOM, CAP, SEED = 0.02, 200, 10_000, 20261012
POP = np.array([bin(i).count("1") for i in range(256)], dtype=np.int64)


def popcount(packed):                       # packed: (..., nbytes) uint8 -> (...,) counts
    return POP[packed].sum(axis=-1)


def load_codes(fcols):
    meta = pd.read_parquet(DATA / "gdsc_processed.parquet", columns=["row_id", "split"])
    codes = np.zeros((len(meta), len(fcols)), np.int8)
    for s in range(0, len(fcols), 40):
        chunk = fcols[s:s + 40]
        d = pd.read_parquet(DATA / "gdsc_processed.parquet", columns=chunk)
        for j, c in enumerate(chunk):
            codes[:, s + j] = d[c].map(LEVELS).to_numpy(np.int8)
        del d
    return meta, codes


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    fcols = list(json.loads((DATA / "feature_map.json").read_text()))
    fidx = {c: i for i, c in enumerate(fcols)}
    meta, codes = load_codes(fcols)
    splits = {s: np.where(meta["split"].to_numpy() == s)[0] for s in ("train", "val", "test")}
    cls = {}
    for cond in ("baseline", "sam"):
        p = pd.concat([pd.read_parquet(PRED / f"{cond}_{s}.parquet")[["row_id", "class_pred"]] for s in splits])
        cls[cond] = meta[["row_id"]].merge(p, on="row_id", how="left", validate="one_to_one")["class_pred"].to_numpy()
        assert not pd.isna(cls[cond]).any()
    # packed (feature, level) indicator vectors per split
    packed = {}
    for s, idx in splits.items():
        X = codes[idx]
        packed[s] = np.stack([np.stack([np.packbits(X[:, f] == l) for l in range(3)]) for f in range(len(fcols))])  # (F,3,nb)
    obs = {f: np.unique(codes[splits["train"], f]) for f in range(len(fcols))}
    nlev = np.array([len(obs[f]) for f in range(len(fcols))])
    obs_mat = np.zeros((len(fcols), 3), np.int8)
    for f in range(len(fcols)):
        obs_mat[f, :nlev[f]] = obs[f]
    n_split = {s: len(i) for s, i in splits.items()}
    n_train = n_split["train"]
    min_n = math.ceil(FLOOR * n_train - 1e-9)
    rng_all = np.random.SeedSequence(SEED)

    rule_rows, rand_rows, cover_rows = [], [], []
    for pi, pool in enumerate(POOLS):
        d = json.loads((RULES / f"{pool}.json").read_text())
        cond = d["meta"]["condition"]
        y_res = {s: np.packbits(cls[cond][splits[s]] == "resistant") for s in splits}
        rng = np.random.default_rng(np.random.SeedSequence([SEED, pi]))
        test_hits = np.zeros(n_split["test"], int)
        for r in d["rules"]:
            pm = {}
            for s in splits:
                m = None
                for c in r["conditions"]:
                    v = packed[s][fidx[c["feature"]], LEVELS[c["partition"]]]
                    m = v if m is None else (m & v)
                pm[s] = m
            rc = r["predicted_class"]
            rrow = {"id": r["id"], "pool": pool, "length": r["length"], "rule_class": rc,
                    "stored_train_coverage": r["coverage"], "stored_train_fidelity": r["fidelity_f1"]}
            for s in splits:
                n_cov = int(popcount(pm[s]))
                n_res = int(popcount(pm[s] & y_res[s]))
                n_c = n_res if rc == "resistant" else n_cov - n_res
                rrow[f"n_covered_{s}"] = n_cov
                rrow[f"coverage_{s}"] = n_cov / n_split[s]
                rrow[f"agreement_{s}"] = n_c / n_cov if n_cov else float("nan")
                rrow[f"fidelity_{s}"] = float(fid_from_counts(n_c, n_cov)) if n_cov else float("nan")
            assert abs(rrow["coverage_train"] - r["coverage"]) < 6e-5, (r["id"], rrow["coverage_train"], r["coverage"])
            assert abs(rrow["fidelity_train"] - r["fidelity_f1"]) < 6e-5, (r["id"], rrow["fidelity_train"], r["fidelity_f1"])
            rule_rows.append(rrow)
            test_hits += np.unpackbits(pm["test"])[: n_split["test"]].astype(int)
            # random baseline
            k = r["length"]
            res = {s: {"agr": [], "fid": []} for s in splits}
            have, attempts = 0, 0
            while have < N_RANDOM and attempts < CAP:
                B = min(300, CAP - attempts)
                feats = np.argpartition(rng.random((B, len(fcols))), k - 1, axis=1)[:, :k]
                lv = obs_mat[feats, (rng.random((B, k)) * nlev[feats]).astype(int)]
                mt = packed["train"][feats[:, 0], lv[:, 0]].copy()
                for j in range(1, k):
                    mt &= packed["train"][feats[:, j], lv[:, j]]
                cnt = popcount(mt)
                ok = np.where(cnt >= min_n)[0]
                take = ok[: N_RANDOM - have]
                attempts += B if have + len(ok) < N_RANDOM else int(take[-1]) + 1
                if len(take):
                    nres_t = popcount(mt[take] & y_res["train"])
                    cres = 2 * nres_t >= cnt[take]          # majority class on train rows; tie -> 'resistant'
                    for s in splits:
                        ms = packed[s][feats[take, 0], lv[take, 0]].copy()
                        for j in range(1, k):
                            ms &= packed[s][feats[take, j], lv[take, j]]
                        nc_ = popcount(ms)
                        nr = popcount(ms & y_res[s])
                        nc = np.where(cres, nr, nc_ - nr)
                        with np.errstate(invalid="ignore", divide="ignore"):
                            res[s]["agr"].append(np.where(nc_ > 0, nc / nc_, np.nan))
                        res[s]["fid"].append(fid_from_counts(nc, nc_))
                    have += len(take)
            row = {"id": r["id"], "pool": pool, "length": k, "n_draws": have, "n_attempts": attempts, "draw_shortfall": N_RANDOM - have}
            for s in splits:
                a = np.concatenate(res[s]["agr"]) if res[s]["agr"] else np.array([np.nan])
                f = np.concatenate(res[s]["fid"]) if res[s]["fid"] else np.array([np.nan])
                row[f"real_agreement_{s}"] = rrow[f"agreement_{s}"]
                row[f"rand_agreement_mean_{s}"] = float(np.nanmean(a))
                row[f"rand_agreement_p95_{s}"] = float(np.nanpercentile(a, 95))
                row[f"real_agreement_above_p95_{s}"] = bool(rrow[f"agreement_{s}"] > row[f"rand_agreement_p95_{s}"])
                row[f"real_fidelity_{s}"] = rrow[f"fidelity_{s}"]
                row[f"rand_fidelity_mean_{s}"] = float(np.nanmean(f))
                row[f"rand_fidelity_p95_{s}"] = float(np.nanpercentile(f, 95))
                row[f"real_fidelity_above_p95_{s}"] = bool(rrow[f"fidelity_{s}"] > row[f"rand_fidelity_p95_{s}"])
                row[f"n_draws_empty_{s}"] = int(np.isnan(a).sum())
            rand_rows.append(row)
        cover_rows.append({"pool": pool, "n_rules": len(d["rules"]), "n_test_rows": n_split["test"],
                           "share_test_covered_by_ge1_rule": float((test_hits >= 1).mean()),
                           "share_test_covered_by_exactly_1_rule": float((test_hits == 1).mean()),
                           "share_test_covered_by_ge2_rules": float((test_hits >= 2).mean()),
                           "max_rules_covering_a_test_row": int(test_hits.max())})
    pd.DataFrame(rule_rows).to_csv(OUT / "posthoc_global_rules_agreement_coverage.csv", index=False)
    pd.DataFrame(cover_rows).to_csv(OUT / "posthoc_global_pool_test_coverage.csv", index=False)
    rr = pd.DataFrame(rand_rows)
    rr.to_csv(OUT / "posthoc_global_random_baseline_per_rule.csv", index=False)
    agg = {"n_rules": ("id", "size"), "n_draw_shortfall_rules": ("draw_shortfall", lambda s: int((s > 0).sum())),
           "total_attempts": ("n_attempts", "sum")}
    for s in splits:
        agg.update({f"mean_real_agreement_{s}": (f"real_agreement_{s}", "mean"),
                    f"mean_random_agreement_{s}": (f"rand_agreement_mean_{s}", "mean"),
                    f"mean_p95_agreement_{s}": (f"rand_agreement_p95_{s}", "mean"),
                    f"n_real_above_own_p95_{s}": (f"real_agreement_above_p95_{s}", "sum"),
                    f"share_real_above_own_p95_{s}": (f"real_agreement_above_p95_{s}", "mean"),
                    f"mean_random_fidelity_{s}": (f"rand_fidelity_mean_{s}", "mean"),
                    f"share_real_fid_above_own_p95_{s}": (f"real_fidelity_above_p95_{s}", "mean")})
    summ = rr.groupby("pool").agg(**agg).reset_index()
    allrow = rr.assign(pool="ALL_19").groupby("pool").agg(**agg).reset_index()
    pd.concat([summ, allrow]).to_csv(OUT / "posthoc_global_random_baseline_summary.csv", index=False)
    print("min_n_cover_train", min_n, "done")


if __name__ == "__main__":
    main()
