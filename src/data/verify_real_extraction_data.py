"""
Part 5 of the real-data preparation task: sanity checks on the output of
src.data.prepare_real_extraction_data, run against the REAL consuming code
(src.extraction.nsga.load_condition_data) rather than re-implementing a
parallel loader here.

Usage:
    python -m src.data.verify_real_extraction_data
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from src.extraction.nsga import load_condition_data

ROOT = Path(__file__).resolve().parents[2]
OUT_ROOT = ROOT / "data" / "processed" / "real_extraction"
COMBINED_DIR = ROOT / "data" / "processed"
FULL_DATA = ROOT / "artifacts" / "data" / "gdsc_processed.parquet"
REPORT_OUT = ROOT / "artifacts" / "data_prep" / "part5_sanity_checks.json"


def verify() -> dict:
    meta = json.loads((OUT_ROOT / "meta.json").read_text())
    results: dict = {"checks": []}

    # ---- 1. load through the REAL extractor loader, both conditions -------
    for condition in meta["conditions_run"]:
        for split in ("train", "val", "test"):
            df, feature_cols, feature_map, class_pred_col = load_condition_data(
                OUT_ROOT / "data", condition, split
            )
            results["checks"].append({
                "check": "loads_via_real_nsga_loader",
                "condition": condition,
                "split": split,
                "n_rows": len(df),
                "n_feature_cols": len(feature_cols),
                "class_pred_balance": df[class_pred_col].value_counts().to_dict(),
                "ok": len(df) > 0 and len(feature_cols) == len(feature_map),
            })
    print("[verify] all conditions/splits load through src.extraction.nsga.load_condition_data unmodified")

    # ---- 2. row-count ballpark: shortlist filter reproduces exactly -------
    full = pd.read_parquet(FULL_DATA, columns=["drug_name"])
    shortlist = set(meta["shortlisted_drugs"])
    expected_rows = int(full["drug_name"].isin(shortlist).sum())
    ok = expected_rows == meta["n_rows"]
    results["checks"].append({
        "check": "row_count_matches_shortlist_filter",
        "expected": expected_rows, "actual": meta["n_rows"], "ok": ok,
    })
    print(f"[verify] row count: expected={expected_rows:,} actual={meta['n_rows']:,} ok={ok}")

    # ---- 3. per-drug class balance sanity (not wildly skewed) -------------
    combined = pd.read_parquet(COMBINED_DIR / "real_extraction_inputs_baseline.parquet")
    counts = combined.groupby("drug_name").size().sort_values()
    sample_drugs = list(counts.index[:3]) + list(counts.index[-3:])
    for d in sample_drugs:
        sub = combined[combined["drug_name"] == d]
        true_bal = sub["true_class"].value_counts(normalize=True).round(3).to_dict()
        pred_bal = sub["class_pred"].value_counts(normalize=True).round(3).to_dict()
        worst_true = max(true_bal.values())
        results["checks"].append({
            "check": "per_drug_class_balance_not_degenerate",
            "drug": d, "n": int(len(sub)),
            "true_class_balance": true_bal, "class_pred_balance": pred_bal,
            "ok": worst_true < 0.95,
        })
        print(f"[verify] {d} (n={len(sub)}): true_class={true_bal} class_pred={pred_bal}")

    all_ok = all(c["ok"] for c in results["checks"])
    results["all_ok"] = all_ok
    REPORT_OUT.parent.mkdir(parents=True, exist_ok=True)
    REPORT_OUT.write_text(json.dumps(results, indent=2))
    print(f"[verify] all_ok={all_ok} -> wrote {REPORT_OUT}")
    return results


if __name__ == "__main__":
    verify()
