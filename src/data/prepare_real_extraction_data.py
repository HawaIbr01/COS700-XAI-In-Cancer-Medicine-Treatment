"""
Parts 3-4 of the real-data preparation task for Sakhile's (NSGA-II) and
Melisa's (GA) rule extraction. See artifacts/data_prep/report.md for the
full write-up; this module implements the recommendations made there.

*** PROVISIONAL: DISCRETISATION_MODE and MIN_SAMPLES_PER_DRUG (both set in
configs/data_prep.yaml) are recommendations, not settled decisions. Hawa
needs to confirm or override both before this data is treated as final. ***

IMPORTANT PREMISE CORRECTION (see report Part 4 / "schema" finding):
notebooks/01_synthetic_data_foundation.ipynb and 02_synthetic_rule_pools.ipynb
are the EVALUATION arm's own fixtures ("Section: Evaluation layer" in both
notebook headers) -- they do not establish the schema GA/NSGA-II actually
read. The real, already-working contract is docs/CONTRACTS.md sec 0.1 /
sec 0.4, implemented in src/extraction/nsga.py's `load_condition_data()`:
    artifacts/data/gdsc_processed.parquet + splits.json + feature_map.json
    artifacts/predictions/{condition}_{split}.parquet
This script therefore emits data in THAT contract shape (row_id-keyed, long
format, filtered to the drug shortlist, re-discretised per DISCRETISATION_MODE)
under data/processed/real_extraction/{data,predictions}/, directly runnable
via e.g.:
    python -m src.extraction.nsga --condition baseline \
        --data-dir data/processed/real_extraction/data
It additionally writes one flattened, human-readable parquet per condition
(data/processed/real_extraction_inputs_{baseline,sam_final}.parquet) for
manual inspection / handoff, matching the file names requested in the task.

No retraining: this script only runs inference (a forward pass) through the
two already-trained checkpoints named in the task:
    artifacts/checkpoints/baseline/model.pt              (Adam baseline)
    artifacts/sam_investigation/confirmatory/model.pt    (SAM rho=0.5, WD=1e-3)
The second is used for the "sam" condition here -- NOT the older, superseded
artifacts/checkpoints/sam/model.pt (rho=0.2, no weight decay) -- because the
confirmatory checkpoint's own meta.json records it was produced specifically
"to produce a loadable model.pt for the GA / NSGA-II rule-extraction arms"
(artifacts/sam_investigation/confirmatory/meta.json, "run_note").

Usage:
    python -m src.data.prepare_real_extraction_data
    python -m src.data.prepare_real_extraction_data --skip-sam   # Parts 1-3 only
"""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml
from torch.utils.data import DataLoader, Subset

from src.common.checkpoint import load_checkpoint, peek_meta
from src.common.config import load_config
from src.data.dataset import DeepCCDSDataset
from src.data.discretisation import compute_thresholds, discretise
from src.model.baseline import run_epoch
from src.model.deepccds import DeepCCDS

ROOT = Path(__file__).resolve().parents[2]
ARTIFACTS_DIR = ROOT / "artifacts"
DATA_DIR = ARTIFACTS_DIR / "data"
BASELINE_CKPT_DIR = ARTIFACTS_DIR / "checkpoints" / "baseline"
SAM_FINAL_CKPT_DIR = ARTIFACTS_DIR / "sam_investigation" / "confirmatory"
DATA_PREP_CONFIG_PATH = ROOT / "configs" / "data_prep.yaml"

OUT_ROOT = ROOT / "data" / "processed" / "real_extraction"       # real extractor contract
COMBINED_DIR = ROOT / "data" / "processed"                        # human-readable combined files

DEFAULT_ARCH_CONFIG = {"mutation_dim": 534, "drug_dim": 1024, "n_pathways": 38, "dropout": 0.0}
SPLITS = ("train", "val", "test")


def load_data_prep_config(path: Path = DATA_PREP_CONFIG_PATH) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def build_shortlist(df: pd.DataFrame, min_samples_per_drug: int, low_sanity_floor: int) -> tuple[list[str], pd.Series]:
    """Part 3: drugs with >= min_samples_per_drug tested cell lines (and,
    redundantly but explicitly, always excluding anything under the
    low-sanity floor regardless of min_samples_per_drug)."""
    counts = df.groupby("drug_name").size()
    effective_min = max(min_samples_per_drug, low_sanity_floor)
    shortlist = sorted(counts[counts >= effective_min].index.tolist())
    return shortlist, counts


def _model_from_checkpoint(checkpoint_dir: Path) -> tuple[DeepCCDS, dict]:
    meta = peek_meta(checkpoint_dir)
    arch_config = meta.get("arch_config", DEFAULT_ARCH_CONFIG)
    model = DeepCCDS(**arch_config)
    load_checkpoint(model, checkpoint_dir)
    model.eval()
    return model, meta


def run_inference(checkpoint_dir: Path, wanted_row_ids_by_split: dict[str, np.ndarray],
                   data_dir: Path = DATA_DIR, batch_size: int = 1024,
                   ) -> tuple[dict[str, tuple[np.ndarray, np.ndarray]], dict]:
    """Forward-pass inference only (no gradient updates, no retraining) for
    the given checkpoint, restricted to the requested row_ids per split.
    Reuses src.model.baseline.run_epoch (the same eval-mode code path
    baseline.py/sam.py use themselves) rather than reimplementing it.

    Returns {split: (row_ids, ln_ic50_pred)} and the checkpoint's meta.json.
    """
    model, meta = _model_from_checkpoint(checkpoint_dir)

    results = {}
    for split in SPLITS:
        ds = DeepCCDSDataset(split, artifacts_data_dir=data_dir)
        wanted = wanted_row_ids_by_split[split]
        mask = np.isin(ds.row_ids, wanted)
        idx = np.nonzero(mask)[0].tolist()
        if not idx:
            results[split] = (np.array([], dtype=np.int64), np.array([], dtype=np.float32))
            continue
        loader = DataLoader(Subset(ds, idx), batch_size=batch_size, shuffle=False)
        _, _y_true, y_pred, row_ids = run_epoch(model, loader, optimizer=None, device="cpu")
        results[split] = (row_ids, y_pred.cpu().numpy())
    return results, meta


def _git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, text=True
        ).strip()
    except Exception:
        return "unknown"


def prepare(skip_sam: bool = False, config_path: Path = DATA_PREP_CONFIG_PATH) -> dict:
    data_prep_cfg = load_data_prep_config(config_path)
    disc_cfg = data_prep_cfg["discretisation"]
    shortlist_cfg = data_prep_cfg["drug_shortlist"]
    mode = disc_cfg["mode"]
    fixed_cutoff = disc_cfg.get("fixed_cutoff_ln_ic50")
    min_samples_per_drug = shortlist_cfg["min_samples_per_drug"]
    low_sanity_floor = shortlist_cfg["low_sanity_floor"]

    pipeline_cfg = load_config()
    classes = pipeline_cfg.response_binning.classes  # ["sensitive", "resistant"]

    feature_map = json.loads((DATA_DIR / "feature_map.json").read_text())
    feature_cols = list(feature_map.keys())

    full_df = pd.read_parquet(DATA_DIR / "gdsc_processed.parquet")

    # ---- Part 3: drug shortlist -------------------------------------------
    shortlist, all_counts = build_shortlist(full_df, min_samples_per_drug, low_sanity_floor)
    print(f"[prepare] shortlist: {len(shortlist)}/{full_df['drug_name'].nunique()} drugs "
          f"(min_samples_per_drug={min_samples_per_drug}, low_sanity_floor={low_sanity_floor})")

    df = full_df[full_df["drug_name"].isin(shortlist)].reset_index(drop=True)
    print(f"[prepare] filtered rows: {len(df):,} / {len(full_df):,}")

    # ---- Part 2: discretisation threshold(s), from TRAIN rows only --------
    threshold = compute_thresholds(df, mode=mode, fixed_cutoff_ln_ic50=fixed_cutoff)
    df["response_class"] = discretise(
        df["ln_ic50"].to_numpy(), threshold, df["drug_name"].to_numpy(), classes
    )
    print(f"[prepare] discretisation mode={mode} "
          f"({'per-drug' if isinstance(threshold, dict) else f'single value={threshold:.4f}'})")

    # ---- write the real extractor contract (docs/CONTRACTS.md sec 0.1) ----
    out_data_dir = OUT_ROOT / "data"
    out_pred_dir = OUT_ROOT / "predictions"
    out_data_dir.mkdir(parents=True, exist_ok=True)
    out_pred_dir.mkdir(parents=True, exist_ok=True)

    df.to_parquet(out_data_dir / "gdsc_processed.parquet")
    (out_data_dir / "feature_map.json").write_text(json.dumps(feature_map, indent=2))

    splits_out = {s: df.loc[df["split"] == s, "row_id"].tolist() for s in SPLITS}
    (out_data_dir / "splits.json").write_text(json.dumps(splits_out))

    wanted_row_ids_by_split = {s: np.array(v, dtype=np.int64) for s, v in splits_out.items()}

    conditions_run = {}

    # ---- Part 4.2: baseline inference --------------------------------------
    baseline_preds, baseline_meta = run_inference(BASELINE_CKPT_DIR, wanted_row_ids_by_split)
    conditions_run["baseline"] = _write_condition(
        "baseline", df, baseline_preds, threshold, classes, feature_cols, out_pred_dir, baseline_meta,
        BASELINE_CKPT_DIR,
    )

    # ---- Part 4.2: SAM (confirmatory rho=0.5 + WD) inference ---------------
    sam_gate_ok = (SAM_FINAL_CKPT_DIR / "model.pt").exists()
    if skip_sam or not sam_gate_ok:
        missing_path = SAM_FINAL_CKPT_DIR / "model.pt"
        reason = "--skip-sam passed" if skip_sam else f"{missing_path} not found on this machine"
        print(f"[prepare] SKIPPING SAM-side predictions: {reason}. "
              f"Parts 1-3 and the baseline half of Part 4 are complete regardless.")
    else:
        sam_preds, sam_meta = run_inference(SAM_FINAL_CKPT_DIR, wanted_row_ids_by_split)
        # written under the "sam" condition name so src.extraction.nsga's
        # `--condition sam` (its only two valid choices are baseline/sam)
        # picks it up unmodified -- see module docstring for why this
        # points at the confirmatory checkpoint, not artifacts/checkpoints/sam.
        conditions_run["sam"] = _write_condition(
            "sam", df, sam_preds, threshold, classes, feature_cols, out_pred_dir, sam_meta,
            SAM_FINAL_CKPT_DIR, combined_name="sam_final",
        )

    # ---- meta.json: provenance + the PROVISIONAL flag ----------------------
    meta_out = {
        "PROVISIONAL": True,
        "provisional_note": (
            "DISCRETISATION_MODE and MIN_SAMPLES_PER_DRUG are recommendations, not "
            "settled decisions. Hawa must confirm or override both "
            "(configs/data_prep.yaml) before this data is treated as final."
        ),
        "discretisation_mode": mode,
        "discretisation_threshold": threshold if not isinstance(threshold, dict) else "per-drug (see thresholds field)",
        "thresholds_per_drug": threshold if isinstance(threshold, dict) else None,
        "min_samples_per_drug": min_samples_per_drug,
        "low_sanity_floor": low_sanity_floor,
        "n_drugs_shortlisted": len(shortlist),
        "n_drugs_total": int(full_df["drug_name"].nunique()),
        "shortlisted_drugs": shortlist,
        "n_rows": len(df),
        "n_rows_by_split": {s: len(v) for s, v in splits_out.items()},
        "pipeline_config_hash": pipeline_cfg.config_hash,
        "git_commit": _git_commit(),
        "conditions_run": conditions_run,
    }
    (OUT_ROOT / "meta.json").write_text(json.dumps(meta_out, indent=2))
    print(f"[prepare] wrote {OUT_ROOT / 'meta.json'}")

    return meta_out


def _write_condition(condition: str, df: pd.DataFrame, preds: dict[str, tuple[np.ndarray, np.ndarray]],
                      threshold, classes: list[str], feature_cols: list[str], out_pred_dir: Path,
                      ckpt_meta: dict, ckpt_dir: Path, combined_name: str | None = None) -> dict:
    """Writes artifacts/-contract predictions/{condition}_{split}.parquet plus
    the flattened, human-readable combined parquet for this condition."""
    combined_name = combined_name or condition
    combined_frames = []

    row_id_to_drug = dict(zip(df["row_id"], df["drug_name"]))
    row_id_to_true_class = dict(zip(df["row_id"], df["response_class"]))

    for split, (row_ids, y_pred) in preds.items():
        if len(row_ids) == 0:
            continue
        drug_names = np.array([row_id_to_drug[r] for r in row_ids])
        pred_class = discretise(y_pred, threshold, drug_names, classes)

        preds_df = pd.DataFrame({
            "row_id": row_ids,
            "ln_ic50_pred": y_pred,
            "class_pred": pred_class,
        })
        preds_df.to_parquet(out_pred_dir / f"{condition}_{split}.parquet")

        split_df = df.loc[df["row_id"].isin(row_ids), ["row_id", "cell_name", "drug_name", "split", "ln_ic50",
                                                          "response_class"] + feature_cols].copy()
        split_df = split_df.merge(preds_df, on="row_id", how="inner")
        combined_frames.append(split_df)

    combined = pd.concat(combined_frames, ignore_index=True) if combined_frames else pd.DataFrame()
    combined = combined.rename(columns={"ln_ic50": "true_ln_ic50", "response_class": "true_class"})
    combined_path = COMBINED_DIR / f"real_extraction_inputs_{combined_name}.parquet"
    combined.to_parquet(combined_path)
    print(f"[prepare] {condition}: wrote {len(combined):,} rows -> {combined_path}")

    return {
        "checkpoint_dir": str(ckpt_dir.relative_to(ROOT)),
        "checkpoint_optimiser": ckpt_meta.get("optimiser"),
        "checkpoint_rho": ckpt_meta.get("rho"),
        "checkpoint_weight_decay": ckpt_meta.get("weight_decay"),
        "checkpoint_generalisation_gap": ckpt_meta.get("generalisation_gap"),
        "n_rows": len(combined),
        "combined_output": str(combined_path.relative_to(ROOT)),
        "contract_predictions_dir": str(out_pred_dir.relative_to(ROOT)),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-sam", action="store_true",
                     help="Only build Parts 1-3 + baseline predictions (SAM checkpoint not required)")
    ap.add_argument("--config", default=str(DATA_PREP_CONFIG_PATH))
    args = ap.parse_args()
    prepare(skip_sam=args.skip_sam, config_path=Path(args.config))


if __name__ == "__main__":
    main()
