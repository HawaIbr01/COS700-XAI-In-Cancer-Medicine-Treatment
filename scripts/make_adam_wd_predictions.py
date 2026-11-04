"""Stage 0C: predictions for the Adam + weight-decay checkpoint, same recipe as
src.data.prepare_real_extraction_data (run_inference -> discretise with the
per-drug TRAIN-median thresholds stored in real_extraction/meta.json).

Also re-derives the baseline and SAM(confirmatory) predictions with the same
code and compares them with the shipped files, to show the recipe reproduces
the other two arms' class_pred before the new arm is trusted.

Writes (new files only):
  artifacts/per_drug/data/predictions/adam_wd_{train,val,test}.parquet
  artifacts/per_drug/data/predictions/adam_wd_verification.json
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data.discretisation import discretise                                # noqa: E402
from src.data.prepare_real_extraction_data import run_inference               # noqa: E402

RE = ROOT / "data/processed/real_extraction"
OUT = ROOT / "artifacts/per_drug/data/predictions"
CK = {
    "baseline": ROOT / "artifacts/checkpoints/baseline",
    "sam": ROOT / "artifacts/sam_investigation/confirmatory",
    "adam_wd": ROOT / "artifacts/sam_investigation/adam_wd_control",
}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    meta = json.loads((RE / "meta.json").read_text())
    thresholds = meta["thresholds_per_drug"]
    classes = ["sensitive", "resistant"]
    df = pd.read_parquet(RE / "data/gdsc_processed.parquet", columns=["row_id", "drug_name", "split"])
    drug_of = dict(zip(df["row_id"], df["drug_name"]))
    wanted = {s: df.loc[df["split"] == s, "row_id"].to_numpy(dtype=np.int64) for s in ("train", "val", "test")}

    report = {}
    for cond, ckpt in CK.items():
        preds, ck_meta = run_inference(ckpt, wanted)
        rep = {"checkpoint": str(ckpt.relative_to(ROOT)), "meta_gap": ck_meta.get("generalisation_gap"), "splits": {}}
        for split, (row_ids, y_pred) in preds.items():
            drugs = np.array([drug_of[r] for r in row_ids])
            cls = discretise(y_pred, thresholds, drugs, classes)
            new = pd.DataFrame({"row_id": row_ids, "ln_ic50_pred": y_pred, "class_pred": cls})
            if cond == "adam_wd":
                new.to_parquet(OUT / f"adam_wd_{split}.parquet")
            else:
                old = pd.read_parquet(RE / f"predictions/{cond}_{split}.parquet")
                m = old.merge(new, on="row_id", suffixes=("_old", "_new"), validate="one_to_one")
                rep["splits"][split] = {
                    "n_old": len(old), "n_new": len(new), "n_matched": len(m),
                    "class_pred_mismatches": int((m["class_pred_old"] != m["class_pred_new"]).sum()),
                    "max_abs_ln_ic50_diff": float(np.abs(m["ln_ic50_pred_old"] - m["ln_ic50_pred_new"]).max()),
                }
        report[cond] = rep
        print(cond, json.dumps(rep, indent=1), flush=True)

    # schema + gap checks for the new arm vs the SAM arm's shipped files
    chk = {}
    for split in ("train", "val", "test"):
        a = pd.read_parquet(OUT / f"adam_wd_{split}.parquet")
        b = pd.read_parquet(RE / f"predictions/sam_{split}.parquet")
        chk[split] = {"columns_equal": list(a.columns) == list(b.columns),
                      "dtypes_equal": a.dtypes.astype(str).to_dict() == b.dtypes.astype(str).to_dict(),
                      "row_ids_equal": bool(np.array_equal(np.sort(a.row_id.values), np.sort(b.row_id.values))),
                      "class_counts": a.class_pred.value_counts().to_dict()}
    report["adam_wd_schema_check_vs_sam"] = chk
    (OUT / "adam_wd_verification.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(chk, indent=1))


if __name__ == "__main__":
    main()
