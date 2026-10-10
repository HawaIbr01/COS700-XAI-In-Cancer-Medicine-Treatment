"""Tests for the per-drug pilot code (Stage 5b). Synthetic data only.

(i)   drug filter == manual filter (ga.py, nsga.py, per_drug.data)
(ii)  no cell line is in both the extraction part and the held-out part of a fold
(iii) same seed -> identical output
(iv)  tiny dry run (2 generations, 2 seeds) completes for both extractors, resumes, refuses mixed designs
plus: min_cover_n exactness, held-out scoring, Jaccard / Kuncheva.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.common import metrics  # noqa: E402
from src.extraction import ga, nsga  # noqa: E402
from src.extraction.coverage import coverage_floor  # noqa: E402
from src.per_drug import data as D  # noqa: E402
from src.per_drug import extract as X  # noqa: E402
from src.per_drug import run as R  # noqa: E402
from src.per_drug import scoring as S  # noqa: E402

N_CELLS = 70
DRUGS = ["DrugA", "DrugB", "DrugC"]
FEATS = [f"G{i}_mut" for i in range(6)] + ["PathX_activity"]


@pytest.fixture(scope="module")
def synth(tmp_path_factory):
    root = tmp_path_factory.mktemp("synth")
    rng = np.random.RandomState(0)
    cells = [f"cell{i:03d}" for i in range(N_CELLS)]
    base = {c: {f: rng.choice(["Low", "High"], p=[0.6, 0.4]) for f in FEATS[:-1]} for c in cells}
    for c in cells:
        base[c]["PathX_activity"] = rng.choice(["Low", "Intermediate", "High"])
    rows, rid = [], 0
    for d_i, d in enumerate(DRUGS):
        for c in cells:
            if d == "DrugC" and rng.rand() < 0.1:       # drugs differ in which cells they were tested on
                continue
            resp = "sensitive" if (base[c]["G0_mut"] == "High") ^ (rng.rand() < 0.15) else "resistant"
            rows.append({"row_id": rid, "cell_name": c, "drug_name": d, "split": rng.choice(["train", "val", "test"], p=[.8, .1, .1]),
                         "response_class": resp, **base[c]})
            rid += 1
    df = pd.DataFrame(rows)
    for f in FEATS:
        df[f] = pd.Categorical(df[f])
    data_dir = root / "data"
    pred_dir = root / "predictions"
    data_dir.mkdir()
    pred_dir.mkdir()
    df.to_parquet(data_dir / "gdsc_processed.parquet")
    fmap = {f: {"gene_symbol": f.split("_")[0], "partition_edges": None} for f in FEATS}
    (data_dir / "feature_map.json").write_text(json.dumps(fmap))
    splits = {s: df.loc[df["split"] == s, "row_id"].tolist() for s in ("train", "val", "test")}
    (data_dir / "splits.json").write_text(json.dumps(splits))
    (data_dir / "response_classes.json").write_text(json.dumps(
        {"per_row_class": {str(r): c for r, c in zip(df["row_id"], df["response_class"])}}))
    arm_pred = {}
    for arm, prefix, noise in (("adam", "baseline", 0.10), ("sam", "sam", 0.2), ("adam_wd", "adam_wd", 0.15)):
        flip = rng.rand(len(df)) < noise
        pred = np.where(flip, np.where(df["response_class"] == "sensitive", "resistant", "sensitive"), df["response_class"])
        for s in ("train", "val", "test"):
            m = (df["split"] == s).to_numpy()
            pd.DataFrame({"row_id": df.loc[m, "row_id"].to_numpy(), "ln_ic50_pred": np.float32(0), "class_pred": pred[m]}
                         ).to_parquet(pred_dir / f"{prefix}_{s}.parquet")
        arm_pred[arm] = (pred_dir, prefix)
    return {"df": df, "data_dir": data_dir, "arm_pred": arm_pred}


def _load(synth, drug="DrugA"):
    return D.load_drug_frame(drug, data_dir=synth["data_dir"], arm_pred=synth["arm_pred"])


# ---------------------------------------------------------------- (i)
def test_drug_filter_equals_manual_filter(synth):
    df, dd_dir = synth["df"], synth["data_dir"]
    manual = df[df["drug_name"] == "DrugA"]
    # ga / nsga: condition data restricted to the drug, as a manual filter of the same rows would give
    (dd_dir.parent / "predictions" / "baseline_train.parquet").exists()
    g_train, g_val, *_ = ga.load_condition_data(dd_dir, "baseline", drug="DrugA")
    n_train, *_ = nsga.load_condition_data(dd_dir, "baseline", "train", drug="DrugA")
    assert set(g_train["drug_name"]) == {"DrugA"} == set(g_val["drug_name"]) == set(n_train["drug_name"])
    assert sorted(g_train["row_id"]) == sorted(manual.loc[manual["split"] == "train", "row_id"])
    assert sorted(g_val["row_id"]) == sorted(manual.loc[manual["split"] == "val", "row_id"])
    assert sorted(n_train["row_id"]) == sorted(g_train["row_id"])
    pd.testing.assert_frame_equal(
        n_train.drop(columns="class_pred").sort_values("row_id").reset_index(drop=True),
        manual[manual["split"] == "train"].sort_values("row_id").reset_index(drop=True), check_dtype=False)
    # per-drug frame: all splits kept, split column retained, one row per cell line
    dd = _load(synth)
    assert sorted(dd.df["row_id"]) == sorted(manual["row_id"])
    assert dd.df["cell_name"].is_unique
    m = manual.set_index("row_id")
    assert (dd.df.set_index("row_id")["net_split"] == m["split"].astype(str)).all()
    with pytest.raises(ValueError):
        ga.load_condition_data(dd_dir, "baseline", drug="NoSuchDrug")


# ---------------------------------------------------------------- (ii)
def test_folds_are_disjoint_by_cell_line(synth):
    dd = _load(synth)
    seen_held = []
    for k in range(5):
        p = D.fold_parts(dd, k)
        held, fid, val = set(p.heldout["cell_name"]), set(p.fid["cell_name"]), set(p.inner_val["cell_name"])
        assert not held & fid and not held & val and not fid & val
        assert held | fid | val == set(dd.df["cell_name"])
        n_train = len(fid) + len(val)
        assert abs(len(val) - 0.2 * n_train) <= 1
        seen_held += list(held)
    assert sorted(seen_held) == sorted(dd.df["cell_name"])           # every cell line held out exactly once
    sizes = [len(D.fold_parts(dd, k).heldout) for k in range(5)]
    assert max(sizes) - min(sizes) <= 1
    # fixed seeds -> identical assignment on reload
    assert _load(synth).fold_of == dd.fold_of


# ---------------------------------------------------------------- (iii)
@pytest.mark.parametrize("extractor", ["ga", "nsga"])
def test_same_seed_identical_output(synth, extractor):
    dd = _load(synth)
    kw = dict(n_seeds=2, n_gen=4, min_cover_n=5)
    a = X.run_cell(dd, "sam", extractor, 1, **kw)
    b = X.run_cell(dd, "sam", extractor, 1, **kw)
    for k in ("pool", "per_seed_rules", "heldout", "sizes"):
        assert json.dumps(a[k], sort_keys=True) == json.dumps(b[k], sort_keys=True)     # NaN-safe equality
    assert [r["genes"] for r in a["seed_rows"]] == [r["genes"] for r in b["seed_rows"]]
    c = X.run_cell(dd, "sam", extractor, 1, n_seeds=3, n_gen=4, min_cover_n=5)      # more seeds: first two unchanged
    assert {k: v for k, v in c["per_seed_rules"].items() if k in a["per_seed_rules"]} == a["per_seed_rules"]


# ---------------------------------------------------------------- (iv)
def test_dry_run_both_extractors_resume_and_refuse(synth, tmp_path, monkeypatch):
    real_loader = D.load_drug_frame
    monkeypatch.setattr(D, "load_drug_frame", lambda drug, **kw: real_loader(
        drug, data_dir=synth["data_dir"], arm_pred=synth["arm_pred"]))
    out = tmp_path / "out"
    argv = ["run", "--out-root", str(out), "--drugs", "DrugA", "--n-seeds", "2", "--n-gen", "2",
            "--min-cover-n", "5", "--folds", "0", "1", "--arms", "adam", "adam_wd", "sam"]
    monkeypatch.setattr(sys, "argv", argv)
    R.main()
    cells = sorted((out / "cells" / "DrugA").glob("*.json"))
    assert len(cells) == 3 * 2 * 2                                   # arms x extractors x folds
    lines = [json.loads(x) for x in (out / "manifest.jsonl").read_text().splitlines()]
    assert len(lines) == len(cells) * 2
    assert {"git_commit", "design_hash", "arm", "extractor", "drug", "fold", "seed", "wall_clock_s"} <= set(lines[0])
    for c in cells:
        j = json.loads(c.read_text())
        assert j["key"]["n_gen"] == 2 and len(j["seed_rows"]) == 2 and "heldout" in j
    mtimes = {c: c.stat().st_mtime_ns for c in cells}
    R.main()                                                        # resume: nothing to do, nothing rewritten
    assert {c: c.stat().st_mtime_ns for c in cells} == mtimes
    monkeypatch.setattr(sys, "argv", argv[:argv.index("--n-seeds")] + ["--n-seeds", "3"] + argv[argv.index("--n-seeds") + 2:])
    with pytest.raises(SystemExit):                                 # different seed count must not overwrite
        R.main()


# ------------------------------------------------------ coverage floor
def test_coverage_floor_exact():
    for n in (37, 523, 550, 817):
        for m in (0, 1, 25):
            f = coverage_floor(0.02, m, n)
            for c in range(0, n + 1):
                frac = float((np.arange(n) < c).mean())
                assert (frac >= f) == (c >= m), (n, m, c)
    assert coverage_floor(0.02, None, 100) == 0.02


def test_min_cover_n_applied_in_scorers(synth):
    dd = _load(synth)
    p = D.fold_parts(dd, 0)
    fid = p.fid.copy(); fid["class_pred"] = fid["class_pred_adam"]
    val = p.inner_val.copy(); val["class_pred"] = val["class_pred_adam"]
    ind = [-1] * len(dd.feature_cols)
    ind[0] = 2                                                       # G0_mut == High
    n_cov = int((fid["G0_mut"] == "High").sum())
    for m, ok in ((n_cov, True), (n_cov + 1, False)):
        fl = coverage_floor(0.0, m, len(fid))
        assert (nsga.make_scorer(fid, dd.feature_cols, "class_pred", fl)(ind)[0] > 0) == ok
        assert (ga.make_scorer(fid, val, dd.feature_cols, .45, .35, .2, fl)(ind)[0] > 0) == ok


# ------------------------------------------------------ scoring
def test_score_rule_matches_manual(synth):
    dd = _load(synth)
    held = D.fold_parts(dd, 0).heldout
    rule = {"id": "r", "length": 1, "predicted_class": "sensitive", "fidelity_f1": 0.9, "coverage": 0.1,
            "conditions": [{"feature": "G0_mut", "partition": "High"}]}
    s = S.score_rule(rule, held, "sam")
    cov = held[held["G0_mut"] == "High"]
    assert s["n_covered"] == len(cov)
    assert s["fidelity"] == pytest.approx(metrics.fidelity_f1(pd.Series("sensitive", index=cov.index), cov["class_pred_sam"]))
    tr = cov[cov["net_split"] == "train"]
    assert s["n_covered_net_train"] == len(tr) and s["n_covered_net_other"] == len(cov) - len(tr)
    assert s["fidelity_net_train"] == pytest.approx(metrics.fidelity_f1(pd.Series("sensitive", index=tr.index), tr["class_pred_sam"]))
    assert s["low_coverage_flag"] == (len(cov) < 10)
    empty = {**rule, "conditions": [{"feature": "G0_mut", "partition": "High"}, {"feature": "G0_mut", "partition": "Low"}]}
    z = S.score_rule(empty, held, "sam")
    assert z["n_covered"] == 0 and np.isnan(z["fidelity"])


def test_jaccard_and_kuncheva():
    assert S.jaccard({"a", "b"}, {"b", "c"}) == pytest.approx(1 / 3)
    assert np.isnan(S.jaccard(set(), {"a"})) and np.isnan(S.jaccard(set(), set()))
    n = 572
    assert S.kuncheva({"a", "b"}, {"a", "b"}, n) == pytest.approx(1.0)
    assert S.kuncheva({"a", "b"}, {"c", "d"}, n) == pytest.approx((0 * n - 4) / (2 * (n - 2)))
    assert np.isnan(S.kuncheva({"a"}, {"a", "b"}, n))                  # unequal sizes
    st = S.pairwise_stats([{"a", "b"}, {"a", "c"}, set()])
    assert st["n_empty_sets"] == 1 and st["n_pairs"] == 3 and st["n_pairs_jaccard_defined"] == 1
    assert st["jaccard"] == pytest.approx(1 / 3)


def test_control_hit():
    rules = [{"id": "x", "conditions": [{"feature": "EGFR_mut", "partition": "Low"}]},
             {"id": "y", "conditions": [{"feature": "EGFR_mut", "partition": "High"}]}]
    assert S.control_hit("Erlotinib", rules)["hit_rule_ids"] == ["y"]
    assert S.control_hit("Erlotinib", rules[:1])["hit"] is False
    assert S.control_hit("Cisplatin", rules) == {"is_control": False}
