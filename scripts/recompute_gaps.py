"""Stage 0B: recompute train/val/test loss and generalisation gap for the three
Stage-5 checkpoints with ONE code path (src.model.baseline.run_epoch, eval mode,
no grad), so the numbers are comparable.

gap_val  = val_loss  - train_loss   (src.model.baseline.generalisation_gap; the
                                      definition every scorecard in the repo uses)
gap_test = test_loss - train_loss   (reported for completeness)

Two row sets: ALL rows the networks were trained on (artifacts/data, 413 drugs)
and the 379-drug subset used by the extraction (data/processed/real_extraction).

Writes artifacts/per_drug/data/gap_recompute.json. Read-only w.r.t. checkpoints.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Subset

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data.dataset import DeepCCDSDataset                    # noqa: E402
from src.data.prepare_real_extraction_data import _model_from_checkpoint  # noqa: E402
from src.model.baseline import run_epoch                         # noqa: E402

CKPTS = {
    "adam_baseline": ROOT / "artifacts/checkpoints/baseline",
    "adam_wd": ROOT / "artifacts/sam_investigation/adam_wd_control",
    "sam_confirmatory": ROOT / "artifacts/sam_investigation/confirmatory",
}
SUBSET_PARQUET = ROOT / "data/processed/real_extraction/data/gdsc_processed.parquet"
OUT = ROOT / "artifacts/per_drug/data/gap_recompute.json"


def main():
    torch.manual_seed(0)
    subset_ids = set(pd.read_parquet(SUBSET_PARQUET, columns=["row_id"])["row_id"].tolist())
    datasets = {s: DeepCCDSDataset(s) for s in ("train", "val", "test")}
    result = {}
    for name, ckpt in CKPTS.items():
        model, meta = _model_from_checkpoint(ckpt)
        losses = {"all_rows": {}, "extraction_379_drugs": {}}
        for split, ds in datasets.items():
            for key, idx in (("all_rows", list(range(len(ds)))),
                             ("extraction_379_drugs",
                              [i for i, r in enumerate(ds.row_ids) if int(r) in subset_ids])):
                loader = DataLoader(Subset(ds, idx), batch_size=1024, shuffle=False)
                loss, _, _, _ = run_epoch(model, loader, optimizer=None, device="cpu")
                losses[key][split] = {"loss": float(loss), "n": len(idx)}
        entry = {"checkpoint_dir": str(ckpt.relative_to(ROOT)),
                 "meta_epoch": meta.get("epoch"), "meta_n_epochs": meta.get("n_epochs"),
                 "meta_generalisation_gap": meta.get("generalisation_gap"),
                 "meta_git_commit": meta.get("git_commit"), "meta_produced_by": meta.get("produced_by"),
                 "meta_stopping_rule": meta.get("stopping_rule")}
        for key, d in losses.items():
            entry[key] = {
                **{f"{s}_loss": v["loss"] for s, v in d.items()},
                **{f"n_{s}": v["n"] for s, v in d.items()},
                "gap_val_minus_train": d["val"]["loss"] - d["train"]["loss"],
                "gap_test_minus_train": d["test"]["loss"] - d["train"]["loss"],
            }
        result[name] = entry
        print(name, json.dumps({k: entry[k] for k in ("all_rows", "extraction_379_drugs")}, indent=1), flush=True)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(result, indent=2))
    print("wrote", OUT)


if __name__ == "__main__":
    main()
