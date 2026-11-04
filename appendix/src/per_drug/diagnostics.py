"""POST-HOC DIAGNOSTICS (decided AFTER seeing RESULTS.md; not part of the frozen design).

Read-only analysis of the existing pilot outputs (artifacts/per_drug/pilot/cells). No extraction is re-run.
Outputs go to artifacts/per_drug/diagnostics/ (all files prefixed posthoc_). See docs/per_drug_analysis_plan.md §10.

Items: 1 fidelity metric behaviour, 2 accuracy sensitivity metric, 3 matched random-rule baseline,
4 stability (set definitions, sizes, chance Jaccard), 5 length units + BH over all comparisons,
6 timing for 379 drugs, 7 positive-control feasibility.
"""
from __future__ import annotations

import inspect
import json
import math
import multiprocessing as mp
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import f1_score

from src.common import metrics as M
from src.per_drug import data as D
from src.per_drug.analyse import ARMS, EXTS, bh, _wilcoxon, BOOT_N, BOOT_SEED, load_cells
from src.per_drug.config import DESIGN
from src.per_drug.scoring import POSITIVE_CONTROLS

OUT = Path("artifacts/per_drug/diagnostics")
PILOT = Path("artifacts/per_drug/pilot")
LEVELS = {"Low": 0, "Intermediate": 1, "High": 2}
N_RANDOM, MIN_COVER, ATTEMPT_CAP, BASE_SEED = 200, DESIGN["min_cover_n"], 10_000, 20261011
N_UNIVERSE = DESIGN["kuncheva_n_features"]


# ----------------------------------------------------------------- fidelity helpers
def fid_from_counts(n_c, n_cov):
    """Closed form of src.common.metrics.fidelity_f1 for a rule that predicts ONE fixed class c.
    n_c = covered rows whose network class == c, n_cov = covered rows. NaN if n_cov == 0."""
    n_c = np.asarray(n_c, float)
    n_cov = np.asarray(n_cov, float)
    with np.errstate(divide="ignore", invalid="ignore"):
        p = n_c / n_cov
        out = np.where(n_c == n_cov, 1.0, np.where(n_c == 0, 0.0, p / (1 + p)))
    return np.where(n_cov > 0, out, np.nan)


def check_closed_form(rng, n=600):
    for _ in range(n):
        m = int(rng.integers(1, 40))
        q = rng.random()
        truth = rng.choice(["sensitive", "resistant"], size=m, p=[q, 1 - q])
        c = str(rng.choice(["sensitive", "resistant"]))
        ref = M.fidelity_f1(pd.Series(c, index=range(m)), pd.Series(truth))
        got = float(fid_from_counts((truth == c).sum(), m))
        assert abs(ref - got) < 1e-12, (truth, c, ref, got)


def encode(df, cols):
    return np.column_stack([df[c].map(LEVELS).to_numpy(np.int8) for c in cols])


def draw_random_rules(rng, Xfit, is_res_fit, XH, obs, nlev, k, n_need=N_RANDOM, min_cover=MIN_COVER, cap=ATTEMPT_CAP):
    """Random k-condition rules (distinct random features, level uniform over the feature's observed levels),
    rejection-sampled until the rule covers >= min_cover rows of Xfit. Returns the covered counts on the
    scoring matrix and the rule class (majority of is_res_fit over covered rows; tie -> 'resistant', the
    alphabetically first class, as pandas .mode().iat[0] does in the real extractors)."""
    n_f = Xfit.shape[1]
    got, attempts = [], 0
    n_have = 0
    while n_have < n_need and attempts < cap:
        B = min(2000, cap - attempts)
        feats = np.argpartition(rng.random((B, n_f)), k - 1, axis=1)[:, :k]
        lv = obs[feats, (rng.random((B, k)) * nlev[feats]).astype(int)]
        mf = np.ones((B, Xfit.shape[0]), bool)
        for j in range(k):
            mf &= Xfit[:, feats[:, j]].T == lv[:, j][:, None]
        cnt = mf.sum(1)
        ok = np.where(cnt >= min_cover)[0]
        take = ok[: n_need - n_have]
        used = B if n_have + len(ok) < n_need else int(take[-1]) + 1
        attempts += used
        if len(take):
            n_res = (mf[take] & is_res_fit[None, :]).sum(1)
            cls_res = 2 * n_res >= cnt[take]
            mh = np.ones((len(take), XH.shape[0]), bool)
            for j in range(k):
                mh &= XH[:, feats[take, j]].T == lv[take, j][:, None]
            got.append((mh, cls_res))
            n_have += len(take)
    return got, attempts, n_have


# ----------------------------------------------------------------- per-drug worker
def per_drug(args):
    drug, cells = args
    dd = D.load_drug_frame(drug)
    fcols = dd.feature_cols
    rng_check = np.random.default_rng(1)
    rule_rows, rand_rows, ctrl_rows = [], [], []
    obs_cache = {}
    for ai, arm in enumerate(ARMS):
        cp = f"class_pred_{arm}"
        for ei, ext in enumerate(EXTS):
            for fold in range(DESIGN["n_folds"]):
                c = cells[(drug, arm, ext, fold)]
                parts = D.fold_parts(dd, fold)
                held, fid = parts.heldout, parts.fid
                is_res_h = (held[cp].to_numpy() == "resistant")
                is_other = (held["net_split"].to_numpy() != "train")
                Xf, XH = encode(fid, fcols), encode(held, fcols)
                is_res_f = (fid[cp].to_numpy() == "resistant")
                lev_sets = [np.unique(Xf[:, j]) for j in range(Xf.shape[1])]
                nlev = np.array([len(s) for s in lev_sets])
                obs = np.zeros((Xf.shape[1], nlev.max()), np.int8)
                for j, s in enumerate(lev_sets):
                    obs[j, : len(s)] = s
                rng = np.random.default_rng(np.random.SeedSequence([BASE_SEED, hash_drug(drug), ai, ei, fold]))
                stored = {h["id"]: h for h in c["heldout"]}
                for r in c["pool"]:
                    mask = M.evaluate_rule_mask(held, r["conditions"])
                    rc = r["predicted_class"]
                    cls = held.loc[mask, cp].to_numpy()
                    n_cov = int(mask.sum())
                    n_c = int((cls == rc).sum())
                    fid_v = float(fid_from_counts(n_c, n_cov)) if n_cov else float("nan")
                    sref = stored[r["id"]]["fidelity"]
                    assert (math.isnan(fid_v) and math.isnan(sref)) or abs(fid_v - sref) < 1e-12, (r["id"], fid_v, sref)
                    oth = mask & is_other
                    trn = mask & ~is_other
                    n_t, n_o = int(trn.sum()), int(oth.sum())
                    nct = int((held.loc[trn, cp].to_numpy() == rc).sum())
                    nco = int((held.loc[oth, cp].to_numpy() == rc).sum())
                    # matched-size check: net_train fidelity on random subsets of the same size as the net_other coverage
                    matched = float("nan")
                    if n_o > 0 and n_t >= n_o:
                        tr_cls = (held.loc[trn, cp].to_numpy() == rc)
                        vals = []
                        for _ in range(20):
                            sub = rng.choice(n_t, size=n_o, replace=False)
                            vals.append(float(fid_from_counts(tr_cls[sub].sum(), n_o)))
                        matched = float(np.mean(vals))
                    rule_rows.append({
                        "id": r["id"], "drug": drug, "arm": arm, "extractor": ext, "fold": fold, "length": r["length"],
                        "rule_class": rc, "n_heldout_rows": len(held), "n_covered": n_cov,
                        "coverage_heldout": n_cov / len(held), "n_classes_covered": int(len(set(cls))),
                        "n_rule_class_covered": n_c, "fidelity": fid_v, "accuracy": n_c / n_cov if n_cov else float("nan"),
                        "n_cov_net_train": n_t, "n_cov_net_other": n_o,
                        "n_classes_net_train": int(len(set(held.loc[trn, cp]))), "n_classes_net_other": int(len(set(held.loc[oth, cp]))),
                        "fid_net_train": float(fid_from_counts(nct, n_t)) if n_t else float("nan"),
                        "fid_net_other": float(fid_from_counts(nco, n_o)) if n_o else float("nan"),
                        "fid_net_train_matched_size": matched,
                    })
                    # random baseline matched on number of conditions
                    got, attempts, n_have = draw_random_rules(rng, Xf, is_res_f, XH, obs, nlev, r["length"])
                    fids, accs = [], []
                    for mh, cls_res in got:
                        ncov = mh.sum(1)
                        nres = (mh & is_res_h[None, :]).sum(1)
                        nc = np.where(cls_res, nres, ncov - nres)
                        fids.append(fid_from_counts(nc, ncov))
                        with np.errstate(divide="ignore", invalid="ignore"):
                            accs.append(np.where(ncov > 0, nc / ncov, np.nan))
                    fids = np.concatenate(fids) if fids else np.array([])
                    accs = np.concatenate(accs) if accs else np.array([])
                    rand_rows.append({
                        "id": r["id"], "drug": drug, "arm": arm, "extractor": ext, "fold": fold, "length": r["length"],
                        "real_fidelity": fid_v, "real_accuracy": n_c / n_cov if n_cov else float("nan"),
                        "n_draws": n_have, "n_attempts": attempts, "draw_shortfall": N_RANDOM - n_have,
                        "n_draws_heldout_empty": int(np.isnan(fids).sum()),
                        "rand_fid_mean": float(np.nanmean(fids)) if np.isfinite(fids).any() else float("nan"),
                        "rand_fid_p95": float(np.nanpercentile(fids, 95)) if np.isfinite(fids).any() else float("nan"),
                        "rand_acc_mean": float(np.nanmean(accs)) if np.isfinite(accs).any() else float("nan"),
                        "rand_acc_p95": float(np.nanpercentile(accs, 95)) if np.isfinite(accs).any() else float("nan"),
                    })
    # item 7: positive-control feasibility on ALL rows of the drug
    if drug in POSITIVE_CONTROLS:
        ctrl_rows = control_feasibility(dd, drug)
    return pd.DataFrame(rule_rows), pd.DataFrame(rand_rows), pd.DataFrame(ctrl_rows)


def hash_drug(drug):
    return sum(ord(ch) * (i + 1) for i, ch in enumerate(drug))


def control_feasibility(dd, drug, n_rand=2000):
    df = dd.df
    X = encode(df, dd.feature_cols)
    lev_sets = [np.unique(X[:, j]) for j in range(X.shape[1])]
    nlev = np.array([len(s) for s in lev_sets])
    obs = np.zeros((X.shape[1], nlev.max()), np.int8)
    for j, s in enumerate(lev_sets):
        obs[j, : len(s)] = s
    rows = []
    for feat in POSITIVE_CONTROLS[drug]:
        has = (df[feat] == "High").to_numpy()
        for tname, col in [("net_adam", "class_pred_adam"), ("net_adam_wd", "class_pred_adam_wd"),
                           ("net_sam", "class_pred_sam"), ("true_class", "true_class")]:
            y = df[col].to_numpy()
            is_res = y == "resistant"
            rng = np.random.default_rng(np.random.SeedSequence([BASE_SEED, 7, hash_drug(drug), hash_drug(feat), hash_drug(tname)]))
            got, attempts, n_have = draw_random_rules(rng, X, is_res, X, obs, nlev, 1, n_need=n_rand)
            racc, rfid = [], []
            for mh, cls_res in got:
                ncov = mh.sum(1)
                nres = (mh & is_res[None, :]).sum(1)
                nc = np.where(cls_res, nres, ncov - nres)
                racc.append(nc / ncov)
                rfid.append(fid_from_counts(nc, ncov))
            racc, rfid = np.concatenate(racc), np.concatenate(rfid)
            n_hi = int(has.sum())
            for cls in ("sensitive", "resistant"):
                tp = int(((y == cls) & has).sum())
                row = {"drug": drug, "feature": feat, "target": tname, "rule_class": cls, "n_rows": len(df),
                       "n_feature_High": n_hi, "share_feature_High": n_hi / len(df),
                       "agreement_covered": tp / n_hi if n_hi else float("nan"),       # = precision of the class on covered rows
                       "precision": tp / n_hi if n_hi else float("nan"),
                       "recall": tp / int((y == cls).sum()) if (y == cls).any() else float("nan"),
                       "fidelity_f1": float(fid_from_counts(tp, n_hi)) if n_hi else float("nan"),
                       "base_rate_class": float((y == cls).mean())}
                rows.append(row)
            best = max((r for r in rows[-2:]), key=lambda r: (r["agreement_covered"] if r["agreement_covered"] == r["agreement_covered"] else -1))
            for r in rows[-2:]:
                r["is_best_direction"] = r is best
                r["n_random_draws"] = n_have
                r["rand_acc_p95"] = float(np.percentile(racc, 95))
                r["rand_acc_mean"] = float(racc.mean())
                r["rand_fid_p95"] = float(np.nanpercentile(rfid, 95))
                r["share_random_acc_ge_best"] = float((racc >= best["agreement_covered"] - 1e-12).mean()) if best["agreement_covered"] == best["agreement_covered"] else float("nan")
                r["share_random_fid_ge_best"] = float((rfid >= best["fidelity_f1"] - 1e-12).mean()) if best["fidelity_f1"] == best["fidelity_f1"] else float("nan")
                r["n_feature_High_below_cover_floor"] = bool(n_hi < MIN_COVER)
    return rows


# ----------------------------------------------------------------- comparisons
def comp_generic(dl, metric_cols):
    comps = {"SAM_minus_AdamWD": ("sam", "adam_wd"), "AdamWD_minus_Adam": ("adam_wd", "adam")}
    rng = np.random.default_rng(BOOT_SEED)
    rows = []
    for metric in metric_cols:
        for ext in EXTS:
            for cname, (a, b) in comps.items():
                wide = dl[dl.extractor == ext].pivot(index="drug", columns="arm", values=metric)
                d = (wide[a] - wide[b]).dropna()
                n = len(d)
                lo, hi = (np.percentile(np.median(rng.choice(d.to_numpy(), size=(BOOT_N, n)), axis=1), [2.5, 97.5]) if n else (np.nan, np.nan))
                rows.append({"metric": metric, "extractor": ext, "comparison": cname, "n_drugs_paired": n,
                             "median_diff": float(d.median()) if n else np.nan, "mean_diff": float(d.mean()) if n else np.nan,
                             "ci95_lo_median_diff": float(lo), "ci95_hi_median_diff": float(hi),
                             "n_zero_diffs": int((d == 0).sum()), "wilcoxon_p": _wilcoxon(d)})
    df = pd.DataFrame(rows)
    df["wilcoxon_p_bh_within_metric"] = np.nan
    for m, idx in df.groupby("metric").groups.items():
        df.loc[idx, "wilcoxon_p_bh_within_metric"] = bh(df.loc[idx, "wilcoxon_p"].to_numpy())
    return df


def exp_jaccard(a, b, N=N_UNIVERSE, _cache={}):
    """E[|A∩B| / |A∪B|] for independent uniformly random subsets of sizes a, b out of N (hypergeometric)."""
    if a == 0 or b == 0:
        return float("nan")
    key = (min(a, b), max(a, b))
    if key not in _cache:
        i = np.arange(0, min(a, b) + 1)
        pm = stats.hypergeom.pmf(i, N, a, b)
        _cache[key] = float((pm * i / (a + b - i)).sum())
    return _cache[key]


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    cells = load_cells(PILOT)
    check_closed_form(np.random.default_rng(0))
    drugs = sorted({k[0] for k in cells})
    with mp.get_context("fork").Pool(3) as pool:
        res = pool.map(per_drug, [(d, cells) for d in drugs], chunksize=1)
    ev = pd.concat([r[0] for r in res], ignore_index=True)
    rnd = pd.concat([r[1] for r in res], ignore_index=True)
    ctrl = pd.concat([r[2] for r in res if len(r[2])], ignore_index=True)
    ev.to_csv(OUT / "posthoc_rule_evaluations.csv", index=False)
    rnd.to_csv(OUT / "posthoc_random_baseline_per_rule.csv", index=False)
    ctrl.to_csv(OUT / "posthoc_positive_control_feasibility.csv", index=False)

    g = ["extractor", "arm"]
    # ---- item 1
    ev["single_class"] = ev.n_classes_covered == 1
    q = np.arange(0.1, 1.0, 0.1)
    it1 = ev.groupby(g).agg(n_rule_evals=("id", "size"), n_nan_fidelity=("fidelity", lambda s: int(s.isna().sum())),
                            share_fidelity_exactly_1=("fidelity", lambda s: float((s == 1.0).mean())),
                            share_fidelity_exactly_0=("fidelity", lambda s: float((s == 0.0).mean())),
                            share_single_network_class=("single_class", "mean")).reset_index()
    dec = ev.groupby(g).fidelity.quantile(q).unstack().reset_index()
    dec.columns = g + [f"p{int(round(x * 100))}" for x in q]
    it1 = it1.merge(dec, on=g)
    it1.to_csv(OUT / "posthoc_1_fidelity_behaviour.csv", index=False)
    ev["fid1_and_single"] = (ev.fidelity == 1.0) & ev.single_class
    cross = (ev.groupby(g).apply(lambda d: pd.Series({"fid==1 & single_class": int(((d.fidelity == 1) & d.single_class).sum()),
                                                      "fid==1 & not single_class": int(((d.fidelity == 1) & ~d.single_class).sum()),
                                                      "fid<=0.5 & single_class": int(((d.fidelity <= .5) & d.single_class).sum()),
                                                      "fid max when 2 classes covered": float(d.loc[~d.single_class, "fidelity"].max())}),
                                 include_groups=False).reset_index())
    cross.to_csv(OUT / "posthoc_1_fidelity_vs_single_class.csv", index=False)

    def split_stats(d):
        both = d[(d.n_cov_net_train > 0) & (d.n_cov_net_other > 0)]
        return pd.Series({
            "n_rules_total": len(d), "n_rules_both_subsets_covered": len(both),
            "n_rules_no_net_other_cover": int((d.n_cov_net_other == 0).sum()),
            "mean_n_cov_net_train": d.loc[d.n_cov_net_train > 0, "n_cov_net_train"].mean(),
            "mean_n_cov_net_other": d.loc[d.n_cov_net_other > 0, "n_cov_net_other"].mean(),
            "share_single_class_net_train": (d.loc[d.n_cov_net_train > 0, "n_classes_net_train"] == 1).mean(),
            "share_single_class_net_other": (d.loc[d.n_cov_net_other > 0, "n_classes_net_other"] == 1).mean(),
            "share_fid1_net_train": (d.loc[d.n_cov_net_train > 0, "fid_net_train"] == 1).mean(),
            "share_fid1_net_other": (d.loc[d.n_cov_net_other > 0, "fid_net_other"] == 1).mean(),
            "median_fid_net_train": d.fid_net_train.median(), "median_fid_net_other": d.fid_net_other.median(),
            "median_fid_net_train_matched_size": d.fid_net_train_matched_size.median(),
            "median_fid_net_train_same_rules_as_matched": d.loc[d.fid_net_train_matched_size.notna(), "fid_net_train"].median(),
            "median_fid_net_other_same_rules_as_matched": d.loc[d.fid_net_train_matched_size.notna(), "fid_net_other"].median(),
            "median_fid_net_train_matched_size_rules": d.fid_net_train_matched_size.median(),
        })
    it1b = ev.groupby(g).apply(split_stats, include_groups=False).reset_index()
    it1b.to_csv(OUT / "posthoc_1_net_train_vs_other.csv", index=False)

    # ---- item 2: accuracy sensitivity
    cell = ev.groupby(["drug", "arm", "extractor", "fold"]).agg(
        med_acc=("accuracy", "median"), med_cov=("coverage_heldout", "median"), med_fid=("fidelity", "median")).reset_index()
    dl2 = cell.groupby(["drug", "arm", "extractor"]).agg(accuracy=("med_acc", "mean"), coverage=("med_cov", "mean"),
                                                         fidelity=("med_fid", "mean")).reset_index()
    dl2.to_csv(OUT / "posthoc_2_drug_level_accuracy_coverage.csv", index=False)
    c2 = comp_generic(dl2, ["accuracy", "coverage"])
    c2.to_csv(OUT / "posthoc_2_comparisons_accuracy_coverage.csv", index=False)
    means2 = dl2.groupby(g)[["accuracy", "coverage", "fidelity"]].mean().reset_index()
    means2.to_csv(OUT / "posthoc_2_means.csv", index=False)

    # ---- item 3: random baseline summary
    rnd["above_p95"] = rnd.real_fidelity > rnd.rand_fid_p95
    rnd["at_or_above_p95"] = rnd.real_fidelity >= rnd.rand_fid_p95
    rnd["above_p95_acc"] = rnd.real_accuracy > rnd.rand_acc_p95
    rnd["diff_fid"] = rnd.real_fidelity - rnd.rand_fid_mean
    rnd["diff_acc"] = rnd.real_accuracy - rnd.rand_acc_mean
    ok = rnd[rnd.real_fidelity.notna() & rnd.rand_fid_mean.notna()]
    s3 = ok.groupby(g).agg(n_rules=("id", "size"), mean_random_fidelity=("rand_fid_mean", "mean"),
                           mean_of_per_rule_p95_fidelity=("rand_fid_p95", "mean"),
                           mean_real_fidelity=("real_fidelity", "mean"),
                           share_real_above_own_p95=("above_p95", "mean"), share_real_at_or_above_own_p95=("at_or_above_p95", "mean"),
                           mean_random_accuracy=("rand_acc_mean", "mean"), mean_of_per_rule_p95_accuracy=("rand_acc_p95", "mean"),
                           mean_real_accuracy=("real_accuracy", "mean"), share_real_acc_above_own_p95=("above_p95_acc", "mean")).reset_index()
    dd_ = ok.groupby(g + ["drug"]).agg(d=("diff_fid", "mean"), da=("diff_acc", "mean")).reset_index().groupby(g).agg(
        drug_level_mean_real_minus_random_fidelity=("d", "mean"), drug_level_mean_real_minus_random_accuracy=("da", "mean")).reset_index()
    fail = rnd.groupby(g).agg(n_rules_all=("id", "size"), n_rules_with_draw_shortfall=("draw_shortfall", lambda s: int((s > 0).sum())),
                              total_draw_shortfall=("draw_shortfall", "sum"), total_attempts=("n_attempts", "sum"),
                              total_draws=("n_draws", "sum"), draws_with_empty_heldout=("n_draws_heldout_empty", "sum")).reset_index()
    s3 = s3.merge(dd_, on=g).merge(fail, on=g)
    s3.to_csv(OUT / "posthoc_3_random_baseline_summary.csv", index=False)
    byk = ok.groupby(g + ["length"]).size().reset_index(name="n_rules")
    byk.to_csv(OUT / "posthoc_3_rules_by_length.csv", index=False)

    # ---- item 4: stability sets and chance
    rows4 = []
    for (drug, arm, ext), _ in ev.groupby(["drug", "arm", "extractor"]):
        fold_sets, seed_sizes, seed_pairs_obs, seed_pairs_exp = [], [], [], []
        for fold in range(DESIGN["n_folds"]):
            c = cells[(drug, arm, ext, fold)]
            fold_sets.append({x for r in c["pool"] for x in r["genes"]})
            ss = [set(s["genes"]) for s in c["seed_rows"]]
            seed_sizes += [len(s) for s in ss]
            for i in range(len(ss)):
                for j in range(i + 1, len(ss)):
                    if ss[i] and ss[j]:
                        seed_pairs_obs.append(len(ss[i] & ss[j]) / len(ss[i] | ss[j]))
                        seed_pairs_exp.append(exp_jaccard(len(ss[i]), len(ss[j])))
        fo, fe = [], []
        for i in range(5):
            for j in range(i + 1, 5):
                if fold_sets[i] and fold_sets[j]:
                    fo.append(len(fold_sets[i] & fold_sets[j]) / len(fold_sets[i] | fold_sets[j]))
                    fe.append(exp_jaccard(len(fold_sets[i]), len(fold_sets[j])))
        rows4.append({"drug": drug, "arm": arm, "extractor": ext, "mean_fold_set_size": np.mean([len(s) for s in fold_sets]),
                      "mean_seed_set_size": np.mean(seed_sizes), "obs_jaccard_folds": np.mean(fo) if fo else np.nan,
                      "chance_jaccard_folds": np.mean(fe) if fe else np.nan, "obs_jaccard_seeds": np.mean(seed_pairs_obs) if seed_pairs_obs else np.nan,
                      "chance_jaccard_seeds": np.mean(seed_pairs_exp) if seed_pairs_exp else np.nan})
    st = pd.DataFrame(rows4)
    st.to_csv(OUT / "posthoc_4_stability_drug_level.csv", index=False)
    st4 = st.groupby(g).mean(numeric_only=True).reset_index()
    st4.to_csv(OUT / "posthoc_4_stability_summary.csv", index=False)

    # ---- item 5
    comp = pd.read_csv("artifacts/per_drug/comparisons.csv")
    elig = ~((comp.metric == "stability_kuncheva_folds") & (comp.n_drugs_paired < 5))
    comp["wilcoxon_p_bh_within_metric"] = comp["wilcoxon_p_bh"]
    comp["in_bh_all_family"] = elig
    comp["wilcoxon_p_bh_all_comparisons"] = np.nan
    comp.loc[elig, "wilcoxon_p_bh_all_comparisons"] = bh(comp.loc[elig, "wilcoxon_p"].to_numpy())
    comp.drop(columns=["wilcoxon_p_bh"]).to_csv(OUT / "posthoc_5_comparisons_bh_all.csv", index=False)
    rules_all = []
    for (drug, arm, ext, fold), c in cells.items():
        rules_all += [{"arm": arm, "extractor": ext, "length": r["length"]} for r in c["pool"]]
    ra = pd.DataFrame(rules_all)
    len5 = ra.groupby(g).length.agg(n_rules="size", mean_length_all_rules="mean", median_length_all_rules="median", max_length="max").reset_index()
    dlen = pd.read_csv("artifacts/per_drug/results_drug_level.csv").groupby(g).length.mean().reset_index(name="drug_level_mean_of_fold_median_length")
    len5 = len5.merge(dlen, on=g)
    len5.to_csv(OUT / "posthoc_5_length.csv", index=False)

    # ---- item 6
    man = pd.read_json(PILOT / "manifest.jsonl", lines=True)
    cellwall = man.drop_duplicates(["drug", "arm", "extractor", "fold"])[["drug", "arm", "extractor", "fold", "cell_wall_clock_s"]]
    seedsum = man.groupby(["drug", "arm", "extractor", "fold"]).wall_clock_s.sum().rename("sum_seed_s").reset_index()
    cw = cellwall.merge(seedsum)
    cw["overhead_s"] = cw.cell_wall_clock_s - cw.sum_seed_s
    per = man.groupby("extractor").wall_clock_s.mean()
    ovh = cw.groupby("extractor").overhead_s.mean()
    rows6 = []
    for label, exts in (("GA+NSGA-II", EXTS), ("GA only", ["ga"]), ("NSGA-II only", ["nsga"])):
        for s in (10, 5, 3):
            cpu = sum(15 * (ovh[e] + s * per[e]) for e in exts)             # 3 arms x 5 folds cells per drug
            rows6.append({"plan": label, "seeds": s, "cpu_s_per_drug": cpu, "hours_379_drugs_3_workers": 379 * cpu / 3 / 3600,
                          "hours_379_drugs_1_worker_equiv": 379 * cpu / 3600})
    t6 = pd.DataFrame(rows6)
    t6.to_csv(OUT / "posthoc_6_timing_379_drugs.csv", index=False)
    pd.DataFrame({"mean_seed_s": per, "mean_cell_overhead_s": ovh}).to_csv(OUT / "posthoc_6_timing_inputs.csv")

    json.dump({"fidelity_f1_source": inspect.getsource(M.fidelity_f1), "compute_n_universe": N_UNIVERSE,
               "pilot_wall_clock_s_actual_run": 12232, "model_check_14_drugs_10_seeds_3_workers_s":
               float(14 * sum(15 * (ovh[e] + 10 * per[e]) for e in EXTS) / 3)},
              open(OUT / "posthoc_misc.json", "w"), indent=1)
    print("done")


if __name__ == "__main__":
    main()
