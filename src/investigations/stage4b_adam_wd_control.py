"""
Stage 4b — Adam + WD control run (confound check for Stage 4).

Purpose
-------
Isolate whether the 84.6% generalisation-gap reduction reported in Stage 4
(SAM ρ=0.5, WD=1e-3 vs Adam baseline, WD=0) is attributable to SAM's
sharpness-seeking geometry, or whether it is mostly/entirely explained by
weight decay alone — a generic regulariser with nothing intrinsically to do
with SAM, and one the original Adam baseline never received.

This script trains a plain-Adam DeepCCDS model on the full dataset with
WD=1e-3, LR=0.001 (unchanged), everything else identical to the Stage 4 SAM
confirmatory run (same architecture from artifacts/checkpoints/baseline/
meta.json, same DeepCCDSDataset splits, same batch size, same eval-mode
final metrics). It can then be compared directly against Stage 4's SAM
ρ=0.5/WD=1e-3 result as a matched pair with weight decay as the single
differing variable.

Stopping rule
-------------
val-loss-plateau-only (patience=30, min_delta=1e-4, MAX_EPOCHS=250 cap) —
the SAME rule used by Stage 4's SAM confirmatory run, deliberately NOT
Angelique's reverse-early-stopping rule (which additionally requires train
loss to still be falling over the trailing window). The reverse-early-stopping
rule is built to force a deliberately-overfitting baseline to keep training
until it demonstrably overfits; it is the wrong instrument for a
WD-regularised model that may plateau without ever showing that signature —
the same mismatch that made the first two Stage 4 attempts run to the cap
without producing usable output. See journal Stage 4b, "Stopping-rule
decision", for the full argument, including the explicit consequence that
this result is a matched pair against Stage 4's SAM run, NOT a clean
three-way comparison against the original 0.3369 baseline (which was
produced under the reverse-early-stopping rule).

Persistence
-----------
Reuses Stage 4's fixes verbatim: per-epoch JSONL append with flush + fsync,
atomic progress-snapshot writes, and fsync'd log handlers. Stage 4's first
two attempts silently lost all output to a WSL2 /mnt/c (drvfs) hard kill
because logging never fsyncs and results were only written once at the end.
Do not reintroduce either failure mode.

Output: artifacts/sam_investigation/adam_wd_control/  (does NOT touch
Angelique's baseline or Bahiya's SAM checkpoints).

Checkpoint saving (added 2026-09-10): like the first Stage 4 confirmatory run,
the original 2026-09-05 Adam+WD control saved metrics only and no weights. The
matched-pair loss-geometry comparison the thesis wants (Adam+WD vs SAM ρ=0.5+WD
under the same stopping rule) needs the Adam+WD parameters. This script now
writes a contract-compliant checkpoint via src.common.checkpoint —
artifacts/sam_investigation/adam_wd_control/{model.pt, meta.json} — with a
periodic best-val-loss crash-recovery copy ("partial": true) overwritten at
clean completion by the final-epoch state ("partial": false). model.pt is
caught by the repo *.pt gitignore rule and must be transferred out of band;
meta.json is trackable. Mirrors src/investigations/stage4_confirmatory.py.

Run from repo root:
    python -m src.investigations.stage4b_adam_wd_control
"""
from __future__ import annotations

import json
import logging
import os
import platform
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from src.common.checkpoint import hash_split, save_checkpoint
from src.common.config import load_config
from src.data.dataset import DeepCCDSDataset
from src.model.deepccds import DeepCCDS, compute_loss

REPO_ROOT = Path(__file__).resolve().parents[2]
OUT_DIR   = REPO_ROOT / "artifacts" / "sam_investigation" / "adam_wd_control"
BASELINE_CHECKPOINT_DIR = REPO_ROOT / "artifacts" / "checkpoints" / "baseline"
SPLITS_FILE           = REPO_ROOT / "artifacts" / "data" / "splits.json"
RESPONSE_CLASSES_FILE = REPO_ROOT / "artifacts" / "data" / "response_classes.json"

OUT_DIR.mkdir(parents=True, exist_ok=True)   # must exist before the FileHandler below

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    force=True,
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(OUT_DIR / "adam_wd_control.log", mode="w"),
    ],
)
logger = logging.getLogger(__name__)


def _flush_log_handlers() -> None:
    """Force every log handler's buffer out to the OS and fsync file handlers.

    Python's logging flushes the stream on each emit but never fsyncs. On the
    WSL2 /mnt/c (drvfs) mount, buffered writes are not guaranteed to reach the
    Windows-backed file until fsync or a clean close(); a hard kill skips
    close(), which is why Stage 4's first two attempts left a 0-byte log.
    """
    for h in logging.getLogger().handlers:
        try:
            h.flush()
            stream = getattr(h, "stream", None)
            if stream is not None and hasattr(stream, "fileno"):
                os.fsync(stream.fileno())
        except (OSError, ValueError):
            pass


def _atomic_write_json(path: Path, obj) -> None:
    """Write JSON via a temp file + rename so a kill mid-write never truncates
    the previous good copy. fsync before rename so the bytes are durable."""
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)
        f.flush()
        os.fsync(f.fileno())
    tmp.replace(path)


def _hardware_string() -> str:
    """One-line description of the machine this run executes on, for the
    checkpoint meta (this control was rerun on different hardware)."""
    return (
        f"{platform.system()} {platform.release()} | "
        f"python {platform.python_version()} | torch {torch.__version__} | "
        f"cpu={platform.processor() or platform.machine()!r} | "
        f"cuda_available={torch.cuda.is_available()}"
    )


def _git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], cwd=REPO_ROOT, text=True
        ).strip()
    except Exception:      # noqa: BLE001 — provenance is best-effort
        return "unknown"


def _write_checkpoint(model, *, split_hash: str, config_hash: str, epoch: int,
                      gap: float, r2: float, rmse_: float, extra: dict) -> None:
    """Contract-compliant checkpoint write into OUT_DIR (optimiser="adam").
    Same path src/investigations/stage4_confirmatory.py uses for the SAM run."""
    save_checkpoint(
        model, OUT_DIR,
        optimiser="adam",
        config_hash=config_hash,
        split_hash=split_hash,
        epoch=epoch,
        generalisation_gap=gap,
        r2=r2,
        rmse=rmse_,
        extra_meta=extra,
    )


# Config — only WD differs from the original Adam baseline (which had WD=0).
LR = 0.001          # unchanged from original Adam baseline and SAM Stage 4
WD = 1e-3           # THE variable under test
RHO = None          # no SAM wrapper — plain Adam

BATCH_SIZE = 1024
MIN_DELTA  = 1e-4
PATIENCE   = 30      # matches Stage 4 SAM run
MAX_EPOCHS = 250     # safety cap, matches Stage 4 SAM run

# Reference points for the scorecard's three-way comparison block.
BASELINE_GAP = 0.3369              # original Adam baseline (WD=0), reverse-early-stopping rule
BASELINE_R2  = 0.8565
SAM_RHO05_WD1E3_GAP = 0.05173152819530791   # Stage 4 SAM confirmatory, val-loss-plateau rule
SAM_RHO05_WD1E3_R2  = 0.8449745774269104


class StoppingTracker:
    def __init__(self):
        self.best_val_loss     = float("inf")
        self.no_improve_streak = 0
        self.train_history: list[float] = []
        self.val_history:   list[float] = []

    def update(self, train_loss: float, val_loss: float) -> bool:
        """Returns True iff this epoch set a new best val loss."""
        self.train_history.append(train_loss)
        self.val_history.append(val_loss)
        if val_loss < self.best_val_loss - MIN_DELTA:
            self.best_val_loss = val_loss
            self.no_improve_streak = 0
            return True
        self.no_improve_streak += 1
        return False

    def stabilised(self) -> bool:
        # Val-loss plateau only — identical to Stage 4's SAM confirmatory run.
        # The paired "train loss must still be falling" condition is
        # deliberately dropped; see module docstring and journal Stage 4b.
        return self.no_improve_streak >= PATIENCE


def eval_split(model, loader, device):
    model.eval()
    total_loss, n = 0.0, 0
    with torch.no_grad():
        for _, pathway, mutation, drug, y in loader:
            pathway, mutation, drug, y = (
                pathway.to(device), mutation.to(device),
                drug.to(device), y.to(device),
            )
            out = model(pathway, mutation=mutation, drug=drug)
            total_loss += compute_loss(out, y).item() * pathway.size(0)
            n += pathway.size(0)
    return total_loss / n


def r_squared(model, loader, device):
    model.eval()
    all_true, all_pred = [], []
    with torch.no_grad():
        for _, pathway, mutation, drug, y in loader:
            pathway, mutation, drug, y = (
                pathway.to(device), mutation.to(device),
                drug.to(device), y.to(device),
            )
            out = model(pathway, mutation=mutation, drug=drug)
            all_true.append(y)
            all_pred.append(out["prediction"])
    y_true = torch.cat(all_true)
    y_pred = torch.cat(all_pred)
    ss_res = torch.sum((y_true - y_pred) ** 2)
    ss_tot = torch.sum((y_true - y_true.mean()) ** 2)
    return (1 - ss_res / ss_tot).item() if ss_tot > 0 else float("nan")


def rmse(model, loader, device):
    model.eval()
    all_true, all_pred = [], []
    with torch.no_grad():
        for _, pathway, mutation, drug, y in loader:
            pathway, mutation, drug, y = (
                pathway.to(device), mutation.to(device),
                drug.to(device), y.to(device),
            )
            out = model(pathway, mutation=mutation, drug=drug)
            all_true.append(y)
            all_pred.append(out["prediction"])
    y_true = torch.cat(all_true)
    y_pred = torch.cat(all_pred)
    return torch.sqrt(torch.mean((y_true - y_pred) ** 2)).item()


def train_epoch_adam(model, loader, optimizer, device):
    """Plain single-step Adam epoch — the one deliberate difference from
    Stage 4's two-step SAM epoch (no uphill perturbation, no
    _disabled_running_stats second forward pass)."""
    model.train()
    total_loss, n = 0.0, 0
    for _, pathway, mutation, drug, y in loader:
        pathway, mutation, drug, y = (
            pathway.to(device), mutation.to(device),
            drug.to(device), y.to(device),
        )
        optimizer.zero_grad()
        out  = model(pathway, mutation=mutation, drug=drug)
        loss = compute_loss(out, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * pathway.size(0)
        n += pathway.size(0)
    return total_loss / n


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info("Device: %s", device)
    logger.info(
        "Config: optimizer=Adam (no SAM)  LR=%s  WD=%s  patience=%d  min_delta=%s  MAX_EPOCHS=%d",
        LR, WD, PATIENCE, MIN_DELTA, MAX_EPOCHS,
    )

    with open(BASELINE_CHECKPOINT_DIR / "meta.json", encoding="utf-8") as f:
        baseline_meta = json.load(f)
    arch_config = baseline_meta["arch_config"]

    config_hash = load_config().config_hash
    with open(SPLITS_FILE, encoding="utf-8") as f:
        split_hash = hash_split(json.load(f))
    with open(RESPONSE_CLASSES_FILE, encoding="utf-8") as f:
        response_threshold = json.load(f)["threshold"]
    if baseline_meta.get("config_hash") != config_hash:
        logger.warning("config_hash mismatch: baseline=%s  this run=%s",
                       baseline_meta.get("config_hash"), config_hash)
    if baseline_meta.get("split_hash") != split_hash:
        logger.warning("split_hash mismatch: baseline=%s  this run=%s",
                       baseline_meta.get("split_hash"), split_hash)

    train_ds = DeepCCDSDataset("train")
    val_ds   = DeepCCDSDataset("val")
    test_ds  = DeepCCDSDataset("test")

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False)
    test_loader  = DataLoader(test_ds,  batch_size=BATCH_SIZE, shuffle=False)

    logger.info(
        "Dataset sizes — train: %d  val: %d  test: %d",
        len(train_ds), len(val_ds), len(test_ds),
    )
    _flush_log_handlers()   # startup lines durable before the first (long) epoch

    torch.manual_seed(42)
    np.random.seed(42)
    model     = DeepCCDS(**arch_config).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WD)
    tracker   = StoppingTracker()
    t0        = time.time()
    final_epoch = 0
    converged   = False
    interrupted = False

    epoch_history = []  # list of per-epoch records

    out_jsonl    = OUT_DIR / "adam_wd_control_history.jsonl"
    out_progress = OUT_DIR / "adam_wd_control_progress.json"
    out_jsonl.unlink(missing_ok=True)   # fresh run — drop any stale partial log

    # Provenance shared by every checkpoint write this run makes.
    ckpt_base = {
        "arch_config": arch_config,
        "optimizer": "adam",
        "rho": None,
        "weight_decay": WD,
        "learning_rate": LR,
        "batch_size": BATCH_SIZE,
        "response_threshold": response_threshold,
        "stopping_rule": (
            f"val_loss_plateau: patience={PATIENCE}, min_delta={MIN_DELTA}, "
            f"max_epochs={MAX_EPOCHS}"
        ),
        "produced_by": "src.investigations.stage4b_adam_wd_control",
        "produced_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "git_commit": _git_commit(),
        "hardware": _hardware_string(),
        "run_note": (
            "Adam + WD=1e-3 confound control for Stage 4 (matched stopping rule). "
            "The original 2026-09-05 run saved metrics only; this rerun adds a "
            "loadable model.pt so the Adam+WD vs SAM ρ=0.5+WD loss-geometry "
            "comparison can be done on real weights."
        ),
        "original_metrics_run": "artifacts/sam_investigation/adam_wd_control/adam_wd_control_scorecard.json",
    }

    for epoch in range(1, MAX_EPOCHS + 1):
        try:
            train_loss = train_epoch_adam(model, train_loader, optimizer, device)
            val_loss   = eval_split(model, val_loader, device)
        except KeyboardInterrupt:
            logger.warning("Interrupted during epoch %d — persisting progress and stopping.", epoch)
            interrupted = True
            break

        improved = tracker.update(train_loss, val_loss)
        final_epoch = epoch

        gap = val_loss - train_loss
        record = {
            "epoch": epoch,
            "train_loss": round(train_loss, 6),
            "val_loss":   round(val_loss,   6),
            "gap":        round(gap,         6),
            "no_improve_streak": tracker.no_improve_streak,
            "elapsed_s": round(time.time() - t0, 1),
        }
        epoch_history.append(record)

        # --- persist this epoch immediately -------------------------------- #
        with open(out_jsonl, "a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")
            f.flush()
            os.fsync(f.fileno())
        _atomic_write_json(out_progress, {
            "last_epoch": epoch,
            "max_epochs": MAX_EPOCHS,
            "latest": record,
            "best_val_loss": round(tracker.best_val_loss, 6),
            "converged": False,
        })

        # crash-recovery checkpoint on every val-loss improvement
        if improved:
            v_r2   = r_squared(model, val_loader, device)
            v_rmse = rmse(model, val_loader, device)
            _write_checkpoint(
                model, split_hash=split_hash, config_hash=config_hash,
                epoch=epoch, gap=gap, r2=v_r2, rmse_=v_rmse,
                extra={
                    **ckpt_base, "partial": True,
                    "checkpoint_kind": "best_val_loss_mid_run",
                    "epoch_of_this_save": epoch,
                    "best_val_loss": round(tracker.best_val_loss, 6),
                    "n_epochs": epoch, "converged": False, "interrupted": False,
                },
            )
            logger.info("  wrote crash-recovery checkpoint at epoch %d (val_loss=%.4f) -> %s",
                        epoch, val_loss, OUT_DIR / "model.pt")

        logger.info(
            "epoch %3d/%d | train=%.4f val=%.4f gap=%.4f | no_imp=%d | %.0fs elapsed",
            epoch, MAX_EPOCHS, train_loss, val_loss, gap,
            tracker.no_improve_streak, time.time() - t0,
        )
        _flush_log_handlers()

        if tracker.stabilised():
            logger.info(
                "Stopping criterion met at epoch %d (val loss plateaued for %d epochs).",
                epoch, tracker.no_improve_streak,
            )
            converged = True
            break
    else:
        logger.warning("Reached MAX_EPOCHS=%d without the val-loss plateau firing.", MAX_EPOCHS)

    # Final eval-mode metrics
    final_train = eval_split(model, train_loader, device)
    final_val   = eval_split(model, val_loader,   device)
    final_test  = eval_split(model, test_loader,  device)
    val_r2      = r_squared(model, val_loader, device)
    val_rmse    = rmse(model, val_loader, device)
    train_r2    = r_squared(model, train_loader, device)
    test_r2     = r_squared(model, test_loader, device)
    train_rmse  = rmse(model, train_loader, device)
    test_rmse   = rmse(model, test_loader, device)
    runtime     = time.time() - t0

    gen_gap = final_val - final_train

    logger.info(
        "FINAL | train=%.4f val=%.4f test=%.4f gap=%.4f val_r2=%.4f val_rmse=%.4f | "
        "epochs=%d conv=%s interrupted=%s runtime=%.0fs",
        final_train, final_val, final_test, gen_gap, val_r2, val_rmse,
        final_epoch, converged, interrupted, runtime,
    )

    # Matched-pair comparison against Stage 4's SAM run (same stopping rule);
    # gap_vs_original_baseline is informational only — that number used a
    # different stopping rule (see module docstring / journal Stage 4b).
    gap_reduction_vs_baseline = (BASELINE_GAP - gen_gap) / BASELINE_GAP * 100
    sam_extra_reduction = (gen_gap - SAM_RHO05_WD1E3_GAP)
    logger.info(
        "vs. original Adam baseline (WD=0, different stopping rule) | gap: %.4f → %.4f (%.1f%%)",
        BASELINE_GAP, gen_gap, gap_reduction_vs_baseline,
    )
    logger.info(
        "matched pair vs. Stage 4 SAM (ρ=0.5/WD=1e-3, same stopping rule) | "
        "this Adam+WD gap=%.4f  SAM gap=%.4f  SAM-specific extra reduction=%.4f",
        gen_gap, SAM_RHO05_WD1E3_GAP, sam_extra_reduction,
    )

    scorecard = {
        "config": {"optimizer": "adam", "rho": RHO, "lr": LR, "wd": WD},
        "n_epochs": final_epoch,
        "converged": converged,
        "interrupted": interrupted,
        "stopping_rule": (
            f"val_loss_plateau: patience={PATIENCE}, min_delta={MIN_DELTA}. "
            "Identical to Stage 4's SAM confirmatory run; deliberately NOT "
            "Angelique's reverse-early-stopping rule (see journal Stage 4b). "
            "This makes the result a matched pair against Stage 4's SAM run, "
            "not a clean three-way comparison against the original 0.3369 baseline."
        ),
        "final_train_loss": final_train,
        "final_val_loss":   final_val,
        "final_test_loss":  final_test,
        "generalisation_gap": gen_gap,
        "val_r2":   val_r2,
        "val_rmse": val_rmse,
        "train_r2": train_r2,
        "test_r2":  test_r2,
        "train_rmse": train_rmse,
        "test_rmse":  test_rmse,
        "comparison": {
            "original_adam_baseline_wd0": {
                "gap": BASELINE_GAP,
                "r2":  BASELINE_R2,
                "note": "different stopping rule (reverse-early-stopping); reference only",
                "gap_reduction_pct_informational": round(gap_reduction_vs_baseline, 2),
            },
            "sam_rho05_wd1e3_stage4": {
                "gap": SAM_RHO05_WD1E3_GAP,
                "r2":  SAM_RHO05_WD1E3_R2,
                "note": "matched pair — same stopping rule; WD is the only differing variable",
                "sam_specific_extra_gap_reduction": round(sam_extra_reduction, 6),
            },
        },
        "runtime_s": round(runtime, 1),
        "dataset_sizes": {
            "train": len(train_ds),
            "val":   len(val_ds),
            "test":  len(test_ds),
        },
        "max_epochs_cap": MAX_EPOCHS,
    }

    out_scorecard = OUT_DIR / "adam_wd_control_scorecard.json"
    out_history   = OUT_DIR / "adam_wd_control_history.json"

    _atomic_write_json(out_scorecard, scorecard)
    logger.info("Saved: %s", out_scorecard)

    _atomic_write_json(out_history, epoch_history)
    logger.info("Saved: %s (also streamed to %s)", out_history, out_jsonl)

    # Final, authoritative checkpoint (overwrites the last crash-recovery copy).
    _write_checkpoint(
        model, split_hash=split_hash, config_hash=config_hash,
        epoch=final_epoch, gap=gen_gap, r2=val_r2, rmse_=val_rmse,
        extra={
            **ckpt_base, "partial": False,
            "checkpoint_kind": "final_epoch_state",
            "n_epochs": final_epoch, "converged": converged, "interrupted": interrupted,
            "best_val_loss": round(tracker.best_val_loss, 6),
            "final_train_loss": final_train, "final_val_loss": final_val,
            "final_test_loss": final_test,
            "train_r2": train_r2, "test_r2": test_r2,
            "train_rmse": train_rmse, "test_rmse": test_rmse,
            "runtime_s": round(runtime, 1),
        },
    )
    logger.info("Saved checkpoint: %s  (+ meta.json)", OUT_DIR / "model.pt")
    _flush_log_handlers()


if __name__ == "__main__":
    main()
