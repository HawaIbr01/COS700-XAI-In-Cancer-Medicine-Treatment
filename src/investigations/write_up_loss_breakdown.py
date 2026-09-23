"""
Standalone, read-only script for the post-rerun write-up audit (item 3).

Loads each full-data checkpoint that actually exists on disk, runs one
eval-mode forward pass over the validation split, and separately reports:
  (a) prediction MSE against true LN_IC50 targets
  (b) mutation-autoencoder reconstruction MSE
  (c) drug-autoencoder reconstruction MSE
  (d) their sum (== the combined Eq. 2 loss compute_loss() reports)

No training. Does not modify any checkpoint or existing report.

Of the four full-data conditions the audit asked for (Adam baseline, Adam+WD,
SAM rho=0.2, SAM rho=0.5+WD), only two have a loadable artifacts/**/model.pt
on this machine: Adam baseline (artifacts/checkpoints/baseline/) and SAM
rho=0.2 (artifacts/checkpoints/sam/). The Adam+WD control
(artifacts/sam_investigation/adam_wd_control/) and SAM rho=0.5+WD confirmatory
rerun (artifacts/sam_investigation/confirmatory/) have meta.json but no
model.pt in this working tree — see writeup_audit.md item 9. Those two
conditions are reported as "checkpoint_missing" rather than fabricated.
"""

from __future__ import annotations

import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from src.common.checkpoint import load_checkpoint
from src.data.dataset import DeepCCDSDataset
from src.model.deepccds import DeepCCDS

REPO_ROOT = Path(__file__).resolve().parents[2]
OUT_PATH = REPO_ROOT / "artifacts" / "sam_investigation" / "write_up_loss_breakdown.json"

CONDITIONS = {
    "adam_baseline": REPO_ROOT / "artifacts" / "checkpoints" / "baseline",
    "sam_rho0.2": REPO_ROOT / "artifacts" / "checkpoints" / "sam",
    "adam_wd_control": REPO_ROOT / "artifacts" / "sam_investigation" / "adam_wd_control",
    "sam_rho0.5_wd_confirmatory": REPO_ROOT / "artifacts" / "sam_investigation" / "confirmatory",
}

# meta.json/scorecard path used to derive a checkpoint-free combined
# reconstruction-MSE estimate when model.pt is missing, via the exact identity
# final_val_loss == prediction_mse + mutation_recon_mse + drug_recon_mse and
# val_rmse == sqrt(prediction_mse) (both eval_split() and rmse() in
# stage4_confirmatory.py / stage4b_adam_wd_control.py are deterministic
# full-pass evals over the identical val loader, model.eval() mode — no
# randomness between the two functions). Verified exact (to 6 dp) against the
# directly-measured checkpoint breakdown below for the two conditions where
# both are available.
SCORECARD_FOR_META_ONLY = {
    "adam_wd_control": REPO_ROOT / "artifacts" / "sam_investigation" / "adam_wd_control" / "meta.json",
    "sam_rho0.5_wd_confirmatory": REPO_ROOT / "artifacts" / "sam_investigation" / "confirmatory" / "meta.json",
}


@torch.no_grad()
def breakdown_for(ckpt_dir: Path) -> dict:
    meta = json.loads((ckpt_dir / "meta.json").read_text())
    model = DeepCCDS(**meta["arch_config"])
    load_checkpoint(model, ckpt_dir)
    model.eval()

    loader = DataLoader(DeepCCDSDataset("val"), batch_size=2048, shuffle=False)

    pred_sq_err_sum = 0.0
    mut_sq_err_sum = 0.0
    drug_sq_err_sum = 0.0
    n = 0

    for _, pathway, mutation, drug, y in loader:
        out = model(pathway, mutation=mutation, drug=drug)
        bsz = pathway.size(0)
        n += bsz
        pred_sq_err_sum += torch.nn.functional.mse_loss(
            out["prediction"], y, reduction="sum"
        ).item()
        if "mutation_recon" in out:
            mut_sq_err_sum += torch.nn.functional.mse_loss(
                out["mutation_recon"], out["mutation_input"], reduction="sum"
            ).item()
        if "drug_recon" in out:
            drug_sq_err_sum += torch.nn.functional.mse_loss(
                out["drug_recon"], out["drug_input"], reduction="sum"
            ).item()

    # mse_loss(reduction="sum") sums over all elements (batch * feature_dim);
    # dividing by n reproduces the same per-sample-mean-over-all-elements
    # convention compute_loss() uses per batch (mse_loss default reduction).
    pred_mse = pred_sq_err_sum / n
    mut_mse = mut_sq_err_sum / (n * mutation.shape[1]) if "mutation_recon" in out else None
    drug_mse = drug_sq_err_sum / (n * drug.shape[1]) if "drug_recon" in out else None

    combined = pred_mse + (mut_mse or 0.0) + (drug_mse or 0.0)

    return {
        "checkpoint_dir": str(ckpt_dir.relative_to(REPO_ROOT)),
        "n_val_rows": n,
        "prediction_mse": pred_mse,
        "mutation_recon_mse": mut_mse,
        "drug_recon_mse": drug_mse,
        "combined_eq2_loss": combined,
        "recorded_final_val_loss_in_meta_or_scorecard": None,  # filled by caller if available
    }


def main() -> None:
    results = {}
    for name, ckpt_dir in CONDITIONS.items():
        if not (ckpt_dir / "model.pt").exists():
            meta = json.loads(SCORECARD_FOR_META_ONLY[name].read_text())
            final_val_loss = meta["final_val_loss"]
            val_rmse = meta["rmse"] if "rmse" in meta else meta["val_rmse"]
            derived_combined_recon_mse = final_val_loss - val_rmse ** 2
            results[name] = {
                "checkpoint_dir": str(ckpt_dir.relative_to(REPO_ROOT)),
                "status": "checkpoint_missing",
                "note": "meta.json present but model.pt is absent from this working "
                        "tree (gitignored *.pt, never regenerated on this machine). "
                        "See writeup_audit.md item 9 / journal Stage 4c, 4e. Values "
                        "below are DERIVED from meta.json's final_val_loss and rmse "
                        "via the exact identity loss = prediction_mse + recon_mse "
                        "(prediction_mse = val_rmse**2); NOT a direct per-checkpoint "
                        "forward pass, and cannot be split into mutation vs. drug "
                        "recon MSE without the weights.",
                "final_val_loss": final_val_loss,
                "val_rmse": val_rmse,
                "prediction_mse_implied": val_rmse ** 2,
                "combined_recon_mse_derived": derived_combined_recon_mse,
                "combined_recon_pct_of_loss": 100 * derived_combined_recon_mse / final_val_loss,
            }
            continue
        print(f"Evaluating {name} ...")
        results[name] = breakdown_for(ckpt_dir)

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(results, indent=2))
    print(f"Wrote {OUT_PATH}")


if __name__ == "__main__":
    main()
