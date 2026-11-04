"""
Stage 3 analysis — read sweep results, produce plots and tables.

Run after stage3_joint_sweep.py has completed:
    python -m src.investigations.stage3_analysis

Reads from:  artifacts/sam_investigation/stage3_sweep/stage3_sweep_summary.json
Writes to:   artifacts/sam_investigation/stage3_*.png
             artifacts/sam_investigation/stage3_analysis.json
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
SWEEP_DIR = REPO_ROOT / "artifacts" / "sam_investigation" / "stage3_sweep"
OUT_DIR = REPO_ROOT / "artifacts" / "sam_investigation"
BASELINE_GAP = 0.3369  # Adam baseline full-data gap (from Stage 1)
BASELINE_R2  = 0.8565  # Adam baseline full-data val_r2


def load_summary() -> dict:
    path = SWEEP_DIR / "stage3_sweep_summary.json"
    if not path.exists():
        raise FileNotFoundError(
            f"Sweep summary not found at {path}.\n"
            "Run src/investigations/stage3_joint_sweep.py first."
        )
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def load_partial_results() -> list[dict]:
    """Read whatever per-config results exist so far (works mid-sweep).
    Handles both .json and the buggy pjson extension from the running sweep."""
    results = []
    for f in sorted(SWEEP_DIR.iterdir()):
        if f.name.startswith("config_") and (f.suffix == ".json" or f.name.endswith("pjson")):
            try:
                results.append(json.loads(f.read_text()))
            except Exception:
                pass
    return results


def main(partial: bool = False) -> None:
    if partial:
        results = load_partial_results()
        print(f"Partial mode: loaded {len(results)} completed configs from {SWEEP_DIR}")
        if not results:
            print("No completed config files found yet.")
            return
    else:
        summary = load_summary()
        results = summary["results"]
    n = len(results)
    print(f"Loaded {n} config results.")

    # ------------------------------------------------------------------ #
    # Extract arrays for plotting
    # ------------------------------------------------------------------ #
    rhos = [r["rho"] for r in results]
    lrs  = [r["lr"]  for r in results]
    wds  = [r["wd"]  for r in results]
    gaps = [r["generalisation_gap"] for r in results]
    r2s  = [r["val_r2"] for r in results]
    epochs = [r["n_epochs"] for r in results]
    converged = [r["converged"] for r in results]

    # ------------------------------------------------------------------ #
    # Plot 1: Gap vs rho, one line per (lr, wd) combination
    # ------------------------------------------------------------------ #
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    lr_wd_combos = sorted(set(zip(lrs, wds)))
    colors = ["#e41a1c", "#377eb8", "#4daf4a", "#984ea3"]
    markers = ["o", "s", "^", "D"]

    unique_rhos = sorted(set(rhos))

    for (lr_val, wd_val), color, marker in zip(lr_wd_combos, colors, markers):
        subset = [(r["rho"], r["generalisation_gap"], r["val_r2"])
                  for r in results if r["lr"] == lr_val and r["wd"] == wd_val]
        subset.sort(key=lambda x: x[0])
        rho_x, gap_y, r2_y = zip(*subset)
        label = f"LR={lr_val}, WD={wd_val}"
        axes[0].plot(rho_x, gap_y, color=color, marker=marker,
                     linewidth=1.8, markersize=7, label=label)
        axes[1].plot(rho_x, r2_y, color=color, marker=marker,
                     linewidth=1.8, markersize=7, label=label)

    for ax in axes:
        ax.set_xlabel("ρ (SAM perturbation radius)")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    axes[0].set_ylabel("Generalisation gap (val − train, eval-mode)")
    axes[0].set_title("(a) Gap vs ρ — by (LR, WD) combo")
    axes[1].set_ylabel("Validation R²")
    axes[1].set_title("(b) Val R² vs ρ — by (LR, WD) combo")

    # Add horizontal reference lines
    axes[0].axhline(0, color="gray", linestyle=":", linewidth=1)
    fig.suptitle(
        "Stage 3: Joint sweep results (10% subsample)\n"
        "Note: absolute gap values not comparable to full-data results; "
        "relative ordering is the key diagnostic.",
        fontsize=9,
    )
    fig.tight_layout()
    out = OUT_DIR / "stage3_gap_vs_rho.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"Saved: {out}")

    # ------------------------------------------------------------------ #
    # Plot 2: Heatmaps — gap and val_r2 as function of (rho, lr) for each WD
    # ------------------------------------------------------------------ #
    unique_wds = sorted(set(wds))
    unique_lrs = sorted(set(lrs), reverse=True)  # higher LR on top

    fig, axes = plt.subplots(len(unique_wds), 2, figsize=(11, 4 * len(unique_wds)))
    if len(unique_wds) == 1:
        axes = [axes]

    for row_idx, wd_val in enumerate(unique_wds):
        for col_idx, metric_key in enumerate(["generalisation_gap", "val_r2"]):
            ax = axes[row_idx][col_idx]
            mat = np.zeros((len(unique_lrs), len(unique_rhos)))
            for ri, rho_val in enumerate(unique_rhos):
                for li, lr_val in enumerate(unique_lrs):
                    match = [r for r in results
                             if r["rho"] == rho_val and r["lr"] == lr_val and r["wd"] == wd_val]
                    if match:
                        mat[li, ri] = match[0][metric_key]
                    else:
                        mat[li, ri] = float("nan")

            cmap = "RdYlGn_r" if metric_key == "generalisation_gap" else "RdYlGn"
            im = ax.imshow(mat, cmap=cmap, aspect="auto")
            ax.set_xticks(range(len(unique_rhos)))
            ax.set_xticklabels([f"ρ={r}" for r in unique_rhos], fontsize=8)
            ax.set_yticks(range(len(unique_lrs)))
            ax.set_yticklabels([f"LR={lr}" for lr in unique_lrs], fontsize=8)
            ax.set_title(f"WD={wd_val} | {metric_key.replace('_', ' ')}", fontsize=9)
            plt.colorbar(im, ax=ax, shrink=0.8)
            for ri in range(len(unique_rhos)):
                for li in range(len(unique_lrs)):
                    v = mat[li, ri]
                    if not np.isnan(v):
                        ax.text(ri, li, f"{v:.3f}", ha="center", va="center",
                                fontsize=7, color="black")

    fig.suptitle(
        "Stage 3 heatmaps: gap (lower=better) and val R² (higher=better)\n"
        "10% subsample — relative ordering only",
        fontsize=10,
    )
    fig.tight_layout()
    out = OUT_DIR / "stage3_heatmaps.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"Saved: {out}")

    # ------------------------------------------------------------------ #
    # Plot 3: Bar chart — top configs by gap
    # ------------------------------------------------------------------ #
    sorted_by_gap = sorted(results, key=lambda r: r["generalisation_gap"])

    top_n = min(8, len(sorted_by_gap))
    labels = [
        f"ρ={r['rho']}\nLR={r['lr']}\nWD={r['wd']}"
        for r in sorted_by_gap[:top_n]
    ]
    gap_vals = [r["generalisation_gap"] for r in sorted_by_gap[:top_n]]

    fig, ax = plt.subplots(figsize=(11, 4.5))
    bar_colors = plt.cm.RdYlGn_r(np.linspace(0.2, 0.8, top_n))
    bars = ax.bar(range(top_n), gap_vals, color=bar_colors, edgecolor="black", linewidth=0.6)
    ax.set_xticks(range(top_n))
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel("Generalisation gap (eval-mode, subsample)")
    ax.set_title(
        f"Stage 3: Top {top_n} configs by gap — joint sweep (10% subsample)\n"
        "Lower gap = better. Absolute values not comparable to full-data results."
    )
    ax.grid(True, axis="y", alpha=0.3)
    for bar, val in zip(bars, gap_vals):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.001,
                f"{val:.3f}", ha="center", va="bottom", fontsize=8)
    fig.tight_layout()
    out = OUT_DIR / "stage3_top_configs.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"Saved: {out}")

    # ------------------------------------------------------------------ #
    # Numeric summary table
    # ------------------------------------------------------------------ #
    print()
    print("=" * 90)
    print("STAGE 3 — Joint Sweep Results (10% subsample, eval-mode gaps)")
    print("WARNING: absolute gap values not comparable to full-data Stage 1 numbers.")
    print("=" * 90)
    print(f"{'ρ':>5} {'LR':>8} {'WD':>6} {'gap':>8} {'val_R²':>8} {'epochs':>7} {'conv':>5} {'t(s)':>6}")
    print("-" * 90)
    for r in sorted_by_gap:
        cv = "Y" if r["converged"] else "N"
        print(f"{r['rho']:>5.2f} {r['lr']:>8.5f} {r['wd']:>6.4f} "
              f"{r['generalisation_gap']:>8.4f} {r['val_r2']:>8.4f} "
              f"{r['n_epochs']:>7d} {cv:>5} {r['runtime_s']:>6.0f}")
    print("=" * 90)

    # ------------------------------------------------------------------ #
    # Marginal analysis: effect of each dimension
    # ------------------------------------------------------------------ #
    print()
    print("--- Marginal effects ---")
    for dim_name, dim_values, dim_key in [
        ("ρ",  unique_rhos,  "rho"),
        ("LR", sorted(set(lrs)), "lr"),
        ("WD", sorted(set(wds)), "wd"),
    ]:
        print(f"\n{dim_name} marginal (mean gap across other dims):")
        for val in dim_values:
            subset_gaps = [r["generalisation_gap"] for r in results if r[dim_key] == val]
            print(f"  {dim_name}={val}: mean_gap={np.mean(subset_gaps):.4f} "
                  f"(n={len(subset_gaps)}, range {min(subset_gaps):.4f}–{max(subset_gaps):.4f})")

    # ------------------------------------------------------------------ #
    # Find best config and interpret rho trend
    # ------------------------------------------------------------------ #
    best = sorted_by_gap[0]
    worst = sorted_by_gap[-1]
    rho_only_wd0_lr001 = sorted(
        [r for r in results if r["lr"] == 0.001 and r["wd"] == 0],
        key=lambda r: r["rho"]
    )
    print()
    print(f"Best config:  ρ={best['rho']}, LR={best['lr']}, WD={best['wd']} → gap={best['generalisation_gap']:.4f}")
    print(f"Worst config: ρ={worst['rho']}, LR={worst['lr']}, WD={worst['wd']} → gap={worst['generalisation_gap']:.4f}")
    print()
    print("Rho trend (LR=0.001, WD=0, to match Bahiya's sweep conditions):")
    for r in rho_only_wd0_lr001:
        print(f"  ρ={r['rho']}: gap={r['generalisation_gap']:.4f} (Stage1 full-data anchor: ρ=0.2→0.312)")

    # Check if trend is still monotonic
    if len(rho_only_wd0_lr001) > 1:
        gaps_in_order = [r["generalisation_gap"] for r in rho_only_wd0_lr001]
        still_decreasing = all(gaps_in_order[i] >= gaps_in_order[i+1]
                               for i in range(len(gaps_in_order) - 1))
        print(f"  Monotonically decreasing with ρ: {still_decreasing}")
        if gaps_in_order[-1] < gaps_in_order[0]:
            print(f"  → Supports rho under-perturbation hypothesis: "
                  f"gap still falls from ρ={rho_only_wd0_lr001[0]['rho']} to ρ={rho_only_wd0_lr001[-1]['rho']}")
        else:
            print(f"  → Rho trend non-monotonic or plateau reached; "
                  f"under-perturbation hypothesis may not fully explain the muted effect.")

    # ------------------------------------------------------------------ #
    # Save analysis JSON
    # ------------------------------------------------------------------ #
    analysis = {
        "stage": 3,
        "best_config": {k: best[k] for k in ["rho", "lr", "wd", "generalisation_gap", "val_r2", "n_epochs"]},
        "sorted_by_gap": [
            {k: r[k] for k in ["rho", "lr", "wd", "generalisation_gap", "val_r2", "n_epochs", "converged"]}
            for r in sorted_by_gap
        ],
        "rho_trend_lr001_wd0": [
            {"rho": r["rho"], "generalisation_gap": r["generalisation_gap"], "val_r2": r["val_r2"]}
            for r in rho_only_wd0_lr001
        ],
        "caveat": (
            "10% training subsample; absolute gap values are larger than full-data results "
            "and should NOT be compared directly with Stage 1 full-data numbers. "
            "Only relative ordering across configs is interpretable."
        ),
    }
    out_json = OUT_DIR / "stage3_analysis.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(analysis, f, indent=2)
    print(f"\nSaved: {out_json}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--partial", action="store_true",
                        help="Run on whatever config files exist so far (mid-sweep)")
    args = parser.parse_args()
    main(partial=args.partial)
