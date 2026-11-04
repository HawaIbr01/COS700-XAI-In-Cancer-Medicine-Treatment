"""
scripts/simulate_data.py

Generates the synthetic artifacts that stand in for Arm 1 (baseline-model)
and Arm 2 (sam-intervention) so the NSGA-II arm can be built and debugged
before real upstream data exists. Produces exactly the files listed in the
branch contract §0.1:

  artifacts/data/gdsc_processed.parquet
  artifacts/data/splits.json
  artifacts/data/feature_map.json
  artifacts/data/response_classes.json
  artifacts/checkpoints/{baseline,sam}/{model.pt, meta.json}
  artifacts/predictions/{baseline,sam}_{train,val,test}.parquet

Design note on the synthetic label structure: two "true" rules are planted
(EGFR_expr High & TP53_expr Low -> resistant; MYC_expr High -> resistant),
then a condition-specific noise rate is applied to class_pred. baseline
gets more noise (0.25) than sam (0.10) — modelling the fact that SAM
regularisation is supposed to reduce the fragility gap (§4.5), so a
higher-fidelity, more consistent rule pool from the sam condition is an
expected and checkable property once Evaluation compares the two.

Run: python scripts/simulate_data.py   (from the project root)
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.common.checkpoint import save_checkpoint
from src.common.config import config_hash, file_hash

ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "configs" / "pipeline.yaml"
DATA_DIR = ROOT / "artifacts" / "data"
CKPT_DIR = ROOT / "artifacts" / "checkpoints"
PRED_DIR = ROOT / "artifacts" / "predictions"

N_ROWS = 1000
FEATURE_GENES = [
    "EGFR", "TP53", "MYC", "KRAS", "PIK3CA", "PTEN", "BRAF", "RB1",
    "CDKN2A", "MDM2", "CCND1", "ERBB2", "NRAS", "STK11", "SMAD4",
]
NOISE_RATE = {"baseline": 0.25, "sam": 0.10}


def discretise(z: float, high_z: float, low_z: float) -> str:
    if z > high_z:
        return "High"
    if z < low_z:
        return "Low"
    return "Intermediate"


def build_gdsc_processed(cfg, rng: np.random.Generator) -> pd.DataFrame:
    high_z = cfg["discretisation"]["high_z"]
    low_z = cfg["discretisation"]["low_z"]

    row_ids = np.arange(1, N_ROWS + 1)
    data = {"row_id": row_ids}

    # Latent continuous expression (z-scores), biased so the planted rules
    # are actually learnable rather than pure noise.
    z_egfr = rng.normal(0, 1, N_ROWS)
    z_tp53 = rng.normal(0, 1, N_ROWS)
    z_myc = rng.normal(0, 1, N_ROWS)

    latent = {"EGFR_expr": z_egfr, "TP53_expr": z_tp53, "MYC_expr": z_myc}
    for gene in FEATURE_GENES:
        if gene in ("EGFR", "TP53", "MYC"):
            continue
        latent[f"{gene}_expr"] = rng.normal(0, 1, N_ROWS)

    for col, z in latent.items():
        data[col] = [discretise(v, high_z, low_z) for v in z]

    # Raw ln_ic50 — optional per the branch contract, included for realism.
    # Correlated with the same planted structure used for class_pred so the
    # dataset is internally consistent even though extraction never reads it.
    rule_a = (z_egfr > high_z) & (z_tp53 < low_z)
    rule_b = z_myc > high_z
    base_resistant = rule_a | rule_b
    data["ln_ic50"] = np.where(
        base_resistant,
        rng.normal(2.0, 0.6, N_ROWS),
        rng.normal(-1.0, 0.6, N_ROWS),
    )

    return pd.DataFrame(data)


def build_splits(cfg, df: pd.DataFrame, rng: np.random.Generator) -> dict:
    ratios = cfg["split"]
    row_ids = df["row_id"].to_numpy().copy()
    rng.shuffle(row_ids)

    n = len(row_ids)
    n_train = int(round(n * ratios["train"]))
    n_val = int(round(n * ratios["val"]))

    train_ids = row_ids[:n_train]
    val_ids = row_ids[n_train:n_train + n_val]
    test_ids = row_ids[n_train + n_val:]

    return {
        "train": train_ids.tolist(),
        "val": val_ids.tolist(),
        "test": test_ids.tolist(),
    }


def build_feature_map(cfg) -> dict:
    high_z = cfg["discretisation"]["high_z"]
    low_z = cfg["discretisation"]["low_z"]
    return {
        f"{gene}_expr": {
            "gene_symbol": gene,
            "partition_edges": {"High": high_z, "Low": low_z},
        }
        for gene in FEATURE_GENES
    }


def build_response_classes(cfg) -> dict:
    return {
        "threshold": cfg["response_binning"]["threshold"],
        "classes": cfg["response_binning"]["classes"],
        "per_row_class": {},
    }


def planted_class(df: pd.DataFrame, noise_rate: float, rng: np.random.Generator) -> np.ndarray:
    """True label from the two planted rules, then label noise applied
    independently per row."""
    rule_a = (df["EGFR_expr"] == "High") & (df["TP53_expr"] == "Low")
    rule_b = df["MYC_expr"] == "High"
    resistant = (rule_a | rule_b).to_numpy()

    labels = np.where(resistant, "resistant", "sensitive")
    flip = rng.random(len(df)) < noise_rate
    flipped = np.where(labels == "resistant", "sensitive", "resistant")
    return np.where(flip, flipped, labels)


def build_predictions(df: pd.DataFrame, splits: dict, condition: str,
                       rng: np.random.Generator) -> dict:
    """One parquet per split for this condition."""
    class_pred_all = planted_class(df, NOISE_RATE[condition], rng)
    class_by_row = dict(zip(df["row_id"], class_pred_all))

    out = {}
    for split_name, row_ids in splits.items():
        row_ids = np.array(row_ids)
        classes = np.array([class_by_row[r] for r in row_ids])
        ln_ic50_pred = np.where(
            classes == "resistant",
            rng.normal(2.0, 0.7, len(row_ids)),
            rng.normal(-1.0, 0.7, len(row_ids)),
        )
        out[split_name] = pd.DataFrame({
            "row_id": row_ids,
            "ln_ic50_pred": ln_ic50_pred,
            "class_pred": classes,
        })
    return out


def main():
    cfg = yaml.safe_load(CONFIG_PATH.read_text())
    seed = cfg["seed"]
    rng = np.random.default_rng(seed)

    for d in (DATA_DIR, PRED_DIR, CKPT_DIR / "baseline", CKPT_DIR / "sam"):
        d.mkdir(parents=True, exist_ok=True)

    # --- data/ ---
    df = build_gdsc_processed(cfg, rng)
    df.to_parquet(DATA_DIR / "gdsc_processed.parquet", index=False)

    splits = build_splits(cfg, df, rng)
    (DATA_DIR / "splits.json").write_text(json.dumps(splits, indent=2))

    feature_map = build_feature_map(cfg)
    (DATA_DIR / "feature_map.json").write_text(json.dumps(feature_map, indent=2))

    response_classes = build_response_classes(cfg)
    (DATA_DIR / "response_classes.json").write_text(json.dumps(response_classes, indent=2))

    print(f"[simulate] wrote data/ — {len(df)} rows, {len(feature_map)} features")
    print(f"[simulate] splits: train={len(splits['train'])} val={len(splits['val'])} test={len(splits['test'])}")

    # --- checkpoints/ ---
    cfg_hash = config_hash(CONFIG_PATH)
    split_hash = file_hash(DATA_DIR / "splits.json")

    for condition, optimiser in (("baseline", "adam"), ("sam", "sam")):
        meta = {
            "optimiser": optimiser,
            "arch": "DeepCCDS",
            "epoch": 42,
            "config_hash": cfg_hash,
            "split_hash": split_hash,
            "note": "synthetic placeholder checkpoint — no real weights",
        }
        save_checkpoint(model=None, meta=meta, out_dir=CKPT_DIR / condition)

    print("[simulate] wrote checkpoints/{baseline,sam}/{model.pt, meta.json}")

    # --- predictions/ ---
    for condition in ("baseline", "sam"):
        preds = build_predictions(df, splits, condition, rng)
        for split_name, split_df in preds.items():
            out_path = PRED_DIR / f"{condition}_{split_name}.parquet"
            split_df.to_parquet(out_path, index=False)
        rates = {k: (v["class_pred"] == "resistant").mean() for k, v in preds.items()}
        print(f"[simulate] wrote predictions/{condition}_{{train,val,test}}.parquet "
              f"(resistant rate: {rates})")

    print("[simulate] done.")


if __name__ == "__main__":
    main()
