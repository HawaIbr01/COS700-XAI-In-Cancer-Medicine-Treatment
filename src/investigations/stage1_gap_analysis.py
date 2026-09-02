"""
Stage 1 — Measure the actual generalisation gap.

Produces:
    artifacts/sam_investigation/stage1_training_curves.png
    artifacts/sam_investigation/stage1_gap_summary.json

Run from repo root:
    python -m src.investigations.stage1_gap_analysis
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
REPORTS_DIR = REPO_ROOT / "artifacts" / "reports"
OUT_DIR = REPO_ROOT / "artifacts" / "sam_investigation"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def load_history(path: Path) -> list[dict]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_scorecard(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def main() -> None:
    # ------------------------------------------------------------------ #
    # Load all training histories
    # ------------------------------------------------------------------ #
    baseline_history = load_history(REPORTS_DIR / "baseline_training_history.json")

    rho_histories = {
        0.01: load_history(REPORTS_DIR / "sam_training_history_rho_0p01.json"),
        0.05: load_history(REPORTS_DIR / "sam_training_history_rho_0p05.json"),
        0.10: load_history(REPORTS_DIR / "sam_training_history_rho_0p1.json"),
        0.20: load_history(REPORTS_DIR / "sam_training_history_rho_0p2.json"),
    }

    baseline_sc = load_scorecard(REPORTS_DIR / "baseline_gap_scorecard.json")
    rho_scorecards = {
        0.01: load_scorecard(REPORTS_DIR / "sam_gap_scorecard_rho_0p01.json"),
        0.05: load_scorecard(REPORTS_DIR / "sam_gap_scorecard_rho_0p05.json"),
        0.10: load_scorecard(REPORTS_DIR / "sam_gap_scorecard_rho_0p1.json"),
        0.20: load_scorecard(REPORTS_DIR / "sam_gap_scorecard_rho_0p2.json"),
    }

    # ------------------------------------------------------------------ #
    # Plot 1: Training curves — baseline + all SAM rho values
    # Two panels: (a) train loss, (b) val loss
    # ------------------------------------------------------------------ #
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    def epochs(h):
        return [e["epoch"] for e in h]

    def train_losses(h):
        return [e["train_loss"] for e in h]

    def val_losses(h):
        return [e["val_loss"] for e in h]

    # Baseline
    axes[0].plot(epochs(baseline_history), train_losses(baseline_history),
                 "k-", linewidth=2, label="Adam baseline")
    axes[1].plot(epochs(baseline_history), val_losses(baseline_history),
                 "k-", linewidth=2, label="Adam baseline")

    # SAM runs
    colors = ["#e41a1c", "#ff7f00", "#4daf4a", "#984ea3"]
    for (rho, hist), color in zip(rho_histories.items(), colors):
        axes[0].plot(epochs(hist), train_losses(hist),
                     color=color, linewidth=1.5, label=f"SAM ρ={rho}")
        axes[1].plot(epochs(hist), val_losses(hist),
                     color=color, linewidth=1.5, label=f"SAM ρ={rho}")

    for ax, title in zip(axes, ["(a) Training loss", "(b) Validation loss"]):
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss (MSE-based, Eq. 2)")
        ax.set_title(title)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    fig.suptitle("Stage 1: Training curves — Adam baseline vs SAM rho sweep", fontsize=12)
    fig.tight_layout()
    out_path = OUT_DIR / "stage1_training_curves.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved: {out_path}")

    # ------------------------------------------------------------------ #
    # Plot 2: Generalisation gap across epochs (per-epoch train-mode metric)
    # ------------------------------------------------------------------ #
    fig, ax = plt.subplots(figsize=(9, 5))

    def gaps(h):
        return [e["generalisation_gap"] for e in h]

    ax.plot(epochs(baseline_history), gaps(baseline_history),
            "k-", linewidth=2, label="Adam baseline")

    for (rho, hist), color in zip(rho_histories.items(), colors):
        ax.plot(epochs(hist), gaps(hist),
                color=color, linewidth=1.5, label=f"SAM ρ={rho}")

    ax.axhline(0, color="gray", linestyle=":", linewidth=1)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Generalisation gap (val_loss − train_loss, train-mode)")
    ax.set_title("Stage 1: Generalisation gap per epoch — Adam vs SAM rho sweep")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    out_path = OUT_DIR / "stage1_gap_per_epoch.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved: {out_path}")

    # ------------------------------------------------------------------ #
    # Plot 3: Final scorecard gap bar chart (eval-mode gaps)
    # ------------------------------------------------------------------ #
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.5))

    configs = ["Adam\nbaseline"] + [f"SAM\nρ={r}" for r in [0.01, 0.05, 0.10, 0.20]]
    gap_vals = [baseline_sc["generalisation_gap"]] + [
        rho_scorecards[r]["generalisation_gap"] for r in [0.01, 0.05, 0.10, 0.20]
    ]
    val_r2_vals = [baseline_sc["val_r2"]] + [
        rho_scorecards[r]["val_r2"] for r in [0.01, 0.05, 0.10, 0.20]
    ]
    val_rmse_vals = [baseline_sc["val_rmse"]] + [
        rho_scorecards[r]["val_rmse"] for r in [0.01, 0.05, 0.10, 0.20]
    ]

    bar_colors = ["#2166ac"] + list(colors)

    for ax, values, ylabel, title in zip(
        axes,
        [gap_vals, val_r2_vals, val_rmse_vals],
        ["Gap (val_loss − train_loss, eval mode)", "Validation R²", "Validation RMSE"],
        ["(a) Generalisation gap", "(b) Validation R²", "(c) Validation RMSE"],
    ):
        bars = ax.bar(configs, values, color=bar_colors, edgecolor="black", linewidth=0.6)
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.grid(True, axis="y", alpha=0.3)
        for bar, val in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.002,
                    f"{val:.3f}", ha="center", va="bottom", fontsize=7)

    # Mark baseline gap with dashed line
    axes[0].axhline(baseline_sc["generalisation_gap"], color="gray",
                    linestyle="--", linewidth=1, alpha=0.7, label="Baseline gap")

    fig.suptitle("Stage 1: Final scorecard metrics — Adam baseline vs SAM (eval-mode)", fontsize=11)
    fig.tight_layout()
    out_path = OUT_DIR / "stage1_scorecard_comparison.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved: {out_path}")

    # ------------------------------------------------------------------ #
    # Numeric summary table
    # ------------------------------------------------------------------ #
    baseline_gap_val = baseline_sc["generalisation_gap"]
    summary = {
        "stage": 1,
        "description": "Generalisation gap measurement — existing training runs",
        "baseline": {
            "optimiser": "adam",
            "weight_decay": 0,
            "learning_rate": baseline_sc["learning_rate"],
            "batch_size": baseline_sc["batch_size"],
            "n_epochs": baseline_sc["n_epochs"],
            "final_train_loss_eval_mode": baseline_sc["final_train_loss"],
            "final_val_loss_eval_mode": baseline_sc["final_val_loss"],
            "generalisation_gap_eval_mode": baseline_gap_val,
            "train_r2": baseline_sc["train_r2"],
            "val_r2": baseline_sc["val_r2"],
            "test_r2": baseline_sc["test_r2"],
            "val_rmse": baseline_sc["val_rmse"],
        },
        "sam_rho_sweep": [],
    }

    for rho in [0.01, 0.05, 0.10, 0.20]:
        sc = rho_scorecards[rho]
        gap = sc["generalisation_gap"]
        pct_change = (baseline_gap_val - gap) / baseline_gap_val * 100
        summary["sam_rho_sweep"].append({
            "rho": rho,
            "n_epochs": sc["n_epochs"],
            "final_train_loss_eval_mode": sc["final_train_loss"],
            "final_val_loss_eval_mode": sc["final_val_loss"],
            "generalisation_gap_eval_mode": gap,
            "gap_pct_change_vs_baseline": round(pct_change, 2),
            "train_r2": sc["train_r2"],
            "val_r2": sc["val_r2"],
            "test_r2": sc["test_r2"],
            "val_rmse": sc["val_rmse"],
        })

    # Gap at epoch 50 for both (checking stability)
    def gap_at_epoch(h, target_epoch):
        for e in h:
            if e["epoch"] == target_epoch:
                return e["generalisation_gap"]
        # If not found, return last before target
        for e in reversed(h):
            if e["epoch"] <= target_epoch:
                return e["generalisation_gap"]
        return None

    summary["gap_at_epoch_50"] = {
        "baseline": gap_at_epoch(baseline_history, 50),
        "sam_rho_0p20": gap_at_epoch(rho_histories[0.20], 50),
    }

    summary["per_epoch_gap_note"] = (
        "Per-epoch gap = val_loss(eval_mode) - train_loss(train_mode). "
        "Scorecard gap = both in eval_mode. Scorecard gaps are larger because "
        "eval_mode train_loss is lower (BN uses running stats, not mini-batch stats)."
    )

    out_json = OUT_DIR / "stage1_gap_summary.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"Saved: {out_json}")

    # ------------------------------------------------------------------ #
    # Print human-readable table
    # ------------------------------------------------------------------ #
    print()
    print("=" * 70)
    print("STAGE 1 — Numeric Summary (eval-mode scorecard gaps)")
    print("=" * 70)
    print(f"{'Config':<20} {'Gap':>8} {'Δ%':>8} {'val_R²':>8} {'val_RMSE':>10} {'epochs':>7}")
    print("-" * 70)
    print(f"{'Adam baseline':<20} {baseline_gap_val:>8.4f} {'—':>8} {baseline_sc['val_r2']:>8.4f} {baseline_sc['val_rmse']:>10.4f} {baseline_sc['n_epochs']:>7}")
    for row in summary["sam_rho_sweep"]:
        label = f"SAM ρ={row['rho']}"
        print(f"{label:<20} {row['generalisation_gap_eval_mode']:>8.4f} {row['gap_pct_change_vs_baseline']:>+7.1f}% "
              f"{row['val_r2']:>8.4f} {row['val_rmse']:>10.4f} {row['n_epochs']:>7}")
    print("=" * 70)
    print()
    print("Note: positive Δ% = gap REDUCED vs baseline (better); negative = gap WIDENED (worse).")


if __name__ == "__main__":
    main()
