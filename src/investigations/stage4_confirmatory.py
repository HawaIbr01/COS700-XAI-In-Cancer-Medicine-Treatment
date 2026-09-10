"""
Stage 4 — confirmatory full-data run: SAM at ρ=0.5, WD=1e-3, LR=0.001.

Validates the Stage 3 subsample finding on the full training set.

Stopping criterion (changed 2026-09-04): val-loss plateau only. The original
pipeline pairs the val-loss plateau with a "train loss must still be improving"
condition — that is Angelique's reverse-early-stopping logic, built to detect
when a deliberately-overfitting baseline has stopped overfitting. It is the wrong
tool for a configuration whose *purpose* is to reduce overfitting: at ρ=0.5 the
train and val losses plateau together, which is the desired outcome, not a
failure to converge. Keeping the paired condition made the loop run to
MAX_EPOCHS every time. See docs/investigations/sam-generalisation/journal.md,
Stage 4, for the full justification.

Progress is persisted every epoch (confirmatory_history.jsonl, fsync'd) so a
kill or crash does not lose the run. Results are saved to
artifacts/sam_investigation/confirmatory/ (NOT Bahiya's
artifacts/checkpoints/sam/ — that checkpoint must not be overwritten).

Checkpoint saving (added 2026-09-10): the original 2026-09-04/05 Stage 4 run
recorded metrics only and never wrote the trained weights. The GA / NSGA-II
rule-extraction arms need the actual parameters, not just the scorecard, so this
script now writes a contract-compliant checkpoint via src.common.checkpoint:

    artifacts/sam_investigation/confirmatory/{model.pt, meta.json}

- model.pt is `model.state_dict()` (same format as artifacts/checkpoints/
  {baseline,sam}/model.pt), loadable with
  `load_checkpoint(DeepCCDS(**arch_config), <that dir>)`.
- A crash-recovery copy is rewritten every time validation loss improves
  (meta.json carries "partial": true until the end-of-run save overwrites it
  with "partial": false and the full eval-mode metrics). A mid-run kill has
  cost this project two days once; the periodic save means a killed run still
  leaves a loadable checkpoint.
- meta.json records full provenance (produced_by / produced_at_utc / hardware /
  exact config / git commit) so this checkpoint is distinguishable from the
  original metrics-only run.

NOTE: model.pt is caught by the repo `*.pt` gitignore rule and is therefore not
git-trackable. meta.json under artifacts/sam_investigation/ is trackable. The
weights must be transferred to the rule-extraction arms out of band.

Run from repo root:
    python -m src.investigations.stage4_confirmatory
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
from torch.utils.data import DataLoader, Subset

from src.common.checkpoint import hash_split, save_checkpoint
from src.common.config import load_config
from src.data.dataset import DeepCCDSDataset
from src.model.deepccds import DeepCCDS, compute_loss
from src.model.sam import SAM, _disabled_running_stats

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    force=True,   # override any handler set by sam.py at import time
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(
            Path(__file__).resolve().parents[2]
            / "artifacts" / "sam_investigation" / "stage4_confirmatory.log",
            mode="w",
        ),
    ],
)
logger = logging.getLogger(__name__)


def _flush_log_handlers() -> None:
    """Force every log handler's buffer out to the OS and fsync file handlers.

    Python's logging flushes the stream on each emit, but never fsyncs. On the
    WSL2 /mnt/c (drvfs) mount, buffered writes are not guaranteed to reach the
    Windows-backed file until fsync or a clean close(); a hard kill skips
    close(), which is why the previous run left a 0-byte log after 10+ hours.
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


REPO_ROOT = Path(__file__).resolve().parents[2]
OUT_DIR   = REPO_ROOT / "artifacts" / "sam_investigation" / "confirmatory"
BASELINE_CHECKPOINT_DIR = REPO_ROOT / "artifacts" / "checkpoints" / "baseline"
SPLITS_FILE           = REPO_ROOT / "artifacts" / "data" / "splits.json"
RESPONSE_CLASSES_FILE = REPO_ROOT / "artifacts" / "data" / "response_classes.json"

# Recommended config from Stage 3
RHO = 0.50
LR  = 0.001
WD  = 1e-3

BATCH_SIZE = 1024
MIN_DELTA  = 1e-4
PATIENCE   = 30       # val-loss plateau: stop after this many epochs with no improvement
MAX_EPOCHS = 250      # safety cap (~90 s/epoch on CPU → ~6 h worst case)

# Stage 1 baseline reference (full-data, eval-mode)
BASELINE_GAP = 0.3369
BASELINE_R2  = 0.8565

# Original metrics-only Stage 4 confirmatory result (2026-09-04/05 run), for the
# post-run consistency check logged at the end.
ORIGINAL_STAGE4_GAP = 0.05173152819530791
ORIGINAL_STAGE4_R2  = 0.8449745774269104


def _hardware_string() -> str:
    """One-line description of the machine this run is executing on, recorded in
    the checkpoint meta so a future reader can tell it apart from the original
    Stage 4 run (which was on different hardware)."""
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


def _subsample(ds, frac: float, seed: int = 42):
    """Deterministic row subsample of a dataset, used only by the smoke test."""
    n = len(ds)
    k = max(1, int(round(n * frac)))
    idx = np.random.default_rng(seed).choice(n, size=k, replace=False)
    return Subset(ds, sorted(idx.tolist()))


class StoppingTracker:
    def __init__(self, patience: int = PATIENCE, min_delta: float = MIN_DELTA):
        self.patience = patience
        self.min_delta = min_delta
        self.best_val_loss    = float("inf")
        self.no_improve_streak = 0
        self.train_history: list[float] = []
        self.val_history:   list[float] = []

    def update(self, train_loss: float, val_loss: float) -> bool:
        """Returns True iff this epoch set a new best val loss."""
        self.train_history.append(train_loss)
        self.val_history.append(val_loss)
        if val_loss < self.best_val_loss - self.min_delta:
            self.best_val_loss = val_loss
            self.no_improve_streak = 0
            return True
        self.no_improve_streak += 1
        return False

    def stabilised(self) -> bool:
        # Val-loss plateau only. The paired "train loss must still be falling"
        # condition from the production pipeline is deliberately dropped here —
        # see module docstring and journal Stage 4.
        return self.no_improve_streak >= self.patience


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


def train_epoch_sam(model, loader, optimizer, device):
    model.train()
    total_loss, n = 0.0, 0
    for _, pathway, mutation, drug, y in loader:
        pathway, mutation, drug, y = (
            pathway.to(device), mutation.to(device),
            drug.to(device), y.to(device),
        )
        out  = model(pathway, mutation=mutation, drug=drug)
        loss = compute_loss(out, y)
        loss.backward()
        optimizer.first_step(zero_grad=True)
        with _disabled_running_stats(model):
            out2  = model(pathway, mutation=mutation, drug=drug)
            loss2 = compute_loss(out2, y)
        loss2.backward()
        optimizer.second_step(zero_grad=True)
        total_loss += loss.item() * pathway.size(0)
        n += pathway.size(0)
    return total_loss / n


def _write_checkpoint(model, checkpoint_dir: Path, *, split_hash: str,
                      config_hash: str, epoch: int, gap: float, r2: float,
                      rmse_: float, extra: dict) -> None:
    """Thin wrapper over src.common.checkpoint.save_checkpoint so the periodic
    (partial) save and the final save go through the identical contract path
    that artifacts/checkpoints/{baseline,sam}/ were written with."""
    save_checkpoint(
        model,
        checkpoint_dir,
        optimiser="sam",
        config_hash=config_hash,
        split_hash=split_hash,
        epoch=epoch,
        generalisation_gap=gap,
        r2=r2,
        rmse=rmse_,
        extra_meta=extra,
    )


def main(*, out_dir: Path = OUT_DIR, checkpoint_dir: Path | None = None,
         max_epochs: int = MAX_EPOCHS, patience: int = PATIENCE,
         subsample: float | None = None, tag: str = "confirmatory") -> dict:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = Path(checkpoint_dir) if checkpoint_dir is not None else out_dir
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info("Device: %s", device)
    logger.info("Hardware: %s", _hardware_string())
    logger.info("Config: rho=%.2f  LR=%s  WD=%s  MAX_EPOCHS=%d  patience=%d", RHO, LR, WD, max_epochs, patience)
    if subsample is not None:
        logger.warning("SMOKE MODE: training on a %.1f%% subsample, %d-epoch cap — NOT a real run",
                       subsample * 100, max_epochs)

    with open(BASELINE_CHECKPOINT_DIR / "meta.json", encoding="utf-8") as f:
        baseline_meta = json.load(f)
    arch_config = baseline_meta["arch_config"]

    cfg = load_config()
    config_hash = cfg.config_hash
    with open(SPLITS_FILE, encoding="utf-8") as f:
        split_hash = hash_split(json.load(f))
    with open(RESPONSE_CLASSES_FILE, encoding="utf-8") as f:
        response_threshold = json.load(f)["threshold"]

    # Cross-check against the baseline checkpoint: a mismatch means this SAM run
    # and the Adam baseline were built from different config / splits and are not
    # directly comparable. Warn rather than abort — we still want the weights.
    if baseline_meta.get("config_hash") != config_hash:
        logger.warning("config_hash mismatch: baseline=%s  this run=%s",
                       baseline_meta.get("config_hash"), config_hash)
    if baseline_meta.get("split_hash") != split_hash:
        logger.warning("split_hash mismatch: baseline=%s  this run=%s",
                       baseline_meta.get("split_hash"), split_hash)

    train_ds = DeepCCDSDataset("train")
    val_ds   = DeepCCDSDataset("val")
    test_ds  = DeepCCDSDataset("test")
    if subsample is not None:
        train_ds = _subsample(train_ds, subsample)
        val_ds   = _subsample(val_ds, subsample)
        test_ds  = _subsample(test_ds, subsample)

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
    optimizer = SAM(
        model.parameters(),
        base_optimizer_cls=torch.optim.Adam,
        rho=RHO, lr=LR, weight_decay=WD,
    )
    tracker   = StoppingTracker(patience=patience)
    t0        = time.time()
    final_epoch = 0
    converged   = False
    interrupted = False

    epoch_history = []  # list of {epoch, train_loss, val_loss, gap}

    # Per-epoch progress files. history.jsonl is append-only and fsync'd every
    # epoch so a kill/crash keeps everything up to the last completed epoch;
    # progress.json is a small human-readable "where is it" snapshot.
    out_jsonl    = out_dir / f"{tag}_history.jsonl"
    out_progress = out_dir / f"{tag}_progress.json"
    out_jsonl.unlink(missing_ok=True)   # fresh run — drop any stale partial log

    # Provenance block shared by every checkpoint write this run makes.
    provenance = {
        "produced_by": "src.investigations.stage4_confirmatory",
        "produced_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "git_commit": _git_commit(),
        "hardware": _hardware_string(),
        "run_note": (
            "Checkpoint-saving rerun of the Stage 4 confirmatory config "
            "(rho=0.5, WD=1e-3, LR=0.001). The original 2026-09-04/05 Stage 4 run "
            "recorded metrics only and saved no weights; this rerun exists to "
            "produce a loadable model.pt for the GA / NSGA-II rule-extraction "
            "arms. Different hardware from the original run; exact reproduction of "
            f"the original gap ({ORIGINAL_STAGE4_GAP:.4f}) / val R2 "
            f"({ORIGINAL_STAGE4_R2:.4f}) is not expected (FP nondeterminism)."
        ),
        "original_metrics_run": "artifacts/sam_investigation/confirmatory/confirmatory_scorecard.json",
        "is_smoke_test": subsample is not None,
    }
    ckpt_base = {
        "arch_config": arch_config,
        "rho": RHO,
        "weight_decay": WD,
        "learning_rate": LR,
        "batch_size": BATCH_SIZE,
        "base_optimiser": "adam",
        "response_threshold": response_threshold,
        "stopping_rule": (
            f"val_loss_plateau: patience={patience}, min_delta={MIN_DELTA}, "
            f"max_epochs={max_epochs}"
        ),
        **provenance,
    }

    for epoch in range(1, max_epochs + 1):
        try:
            train_loss = train_epoch_sam(model, train_loader, optimizer, device)
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
            "max_epochs": max_epochs,
            "latest": record,
            "best_val_loss": round(tracker.best_val_loss, 6),
            "converged": False,
        })

        # --- crash-recovery checkpoint on every val-loss improvement ------- #
        # Cheap: one extra eval-mode pass over val (a few hundred ms). Marks the
        # meta "partial" so a reader knows the full eval-mode metrics are not in
        # yet. Overwritten by the end-of-run save below.
        if improved:
            v_r2   = r_squared(model, val_loader, device)
            v_rmse = rmse(model, val_loader, device)
            _write_checkpoint(
                model, checkpoint_dir,
                split_hash=split_hash, config_hash=config_hash,
                epoch=epoch, gap=gap, r2=v_r2, rmse_=v_rmse,
                extra={
                    **ckpt_base,
                    "partial": True,
                    "checkpoint_kind": "best_val_loss_mid_run",
                    "epoch_of_this_save": epoch,
                    "best_val_loss": round(tracker.best_val_loss, 6),
                    "n_epochs": epoch,
                    "converged": False,
                    "interrupted": False,
                },
            )
            logger.info("  wrote crash-recovery checkpoint at epoch %d (val_loss=%.4f) -> %s",
                        epoch, val_loss, checkpoint_dir / "model.pt")

        logger.info(
            "epoch %3d/%d | train=%.4f val=%.4f gap=%.4f | no_imp=%d | %.0fs elapsed",
            epoch, max_epochs, train_loss, val_loss, gap,
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
        logger.warning("Reached MAX_EPOCHS=%d without the val-loss plateau firing.", max_epochs)

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

    # Compare with baseline
    gap_reduction = (BASELINE_GAP - gen_gap) / BASELINE_GAP * 100
    r2_delta      = val_r2 - BASELINE_R2
    logger.info(
        "vs. Adam baseline | gap: %.4f → %.4f (%.1f%%) | val_r2: %.4f → %.4f (Δ=%.4f)",
        BASELINE_GAP, gen_gap, gap_reduction, BASELINE_R2, val_r2, r2_delta,
    )
    # Consistency check against the original metrics-only Stage 4 run.
    logger.info(
        "vs. original Stage 4 (metrics-only) | gap: %.4f vs %.4f (Δ=%+.4f) | "
        "val_r2: %.4f vs %.4f (Δ=%+.4f)",
        gen_gap, ORIGINAL_STAGE4_GAP, gen_gap - ORIGINAL_STAGE4_GAP,
        val_r2, ORIGINAL_STAGE4_R2, val_r2 - ORIGINAL_STAGE4_R2,
    )
    if subsample is None and (gen_gap > 0.15 or val_r2 < BASELINE_R2 - 0.05):
        logger.warning(
            "POST-RUN CHECK: final gap=%.4f / val_r2=%.4f deviates materially from the "
            "original Stage 4 result (gap≈%.4f, r2≈%.4f) — investigate before handing "
            "this checkpoint downstream.",
            gen_gap, val_r2, ORIGINAL_STAGE4_GAP, ORIGINAL_STAGE4_R2,
        )

    scorecard = {
        "config": {"rho": RHO, "lr": LR, "wd": WD},
        "n_epochs": final_epoch,
        "converged": converged,
        "interrupted": interrupted,
        "stopping_rule": (
            f"val_loss_plateau: patience={patience}, min_delta={MIN_DELTA}. "
            "Train-loss 'still improving' condition removed 2026-09-04 — see journal Stage 4."
        ),
        "final_train_loss": final_train,
        "final_val_loss":   final_val,
        "final_test_loss":  final_test,
        "generalisation_gap": gen_gap,
        "val_r2":   val_r2,
        "val_rmse": val_rmse,
        "gap_vs_baseline": {
            "baseline_gap": BASELINE_GAP,
            "baseline_r2":  BASELINE_R2,
            "gap_reduction_pct": round(gap_reduction, 2),
            "r2_delta": round(r2_delta, 4),
        },
        "runtime_s": round(runtime, 1),
        "dataset_sizes": {
            "train": len(train_ds),
            "val":   len(val_ds),
            "test":  len(test_ds),
        },
        "max_epochs_cap": max_epochs,
        "is_smoke_test": subsample is not None,
    }

    out_scorecard = out_dir / f"{tag}_scorecard.json"
    out_history   = out_dir / f"{tag}_history.json"

    _atomic_write_json(out_scorecard, scorecard)
    logger.info("Saved: %s", out_scorecard)

    _atomic_write_json(out_history, epoch_history)
    logger.info("Saved: %s (also streamed to %s)", out_history, out_jsonl)

    # --- final, authoritative checkpoint ---------------------------------- #
    # Overwrites the last crash-recovery copy. This is the model as it stood at
    # the stopping point (epoch = final_epoch), which is the state the eval-mode
    # metrics above describe and the state the rule-extraction arms should load.
    _write_checkpoint(
        model, checkpoint_dir,
        split_hash=split_hash, config_hash=config_hash,
        epoch=final_epoch, gap=gen_gap, r2=val_r2, rmse_=val_rmse,
        extra={
            **ckpt_base,
            "partial": False,
            "checkpoint_kind": "final_epoch_state",
            "n_epochs": final_epoch,
            "converged": converged,
            "interrupted": interrupted,
            "best_val_loss": round(tracker.best_val_loss, 6),
            "final_train_loss": final_train,
            "final_val_loss": final_val,
            "final_test_loss": final_test,
            "train_r2": train_r2,
            "test_r2": test_r2,
            "train_rmse": train_rmse,
            "test_rmse": test_rmse,
            "runtime_s": round(runtime, 1),
        },
    )
    logger.info("Saved checkpoint: %s  (+ meta.json)", checkpoint_dir / "model.pt")
    _flush_log_handlers()
    return scorecard


if __name__ == "__main__":
    main()
