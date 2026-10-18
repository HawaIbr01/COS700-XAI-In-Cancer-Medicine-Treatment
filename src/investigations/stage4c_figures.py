"""
Stage 4c — consolidated thesis figures for the SAM generalisation-gap investigation.

Turns every numeric result the investigation produced into comparison figures,
spanning the full arc:

  pre-investigation  : Adam baseline (WD=0) + the rho=0.01..0.2 SAM sweep that
                       barely moved the gap (the "muted SAM effect" that
                       triggered the investigation).
  investigation      : Stage 3 (rho,lr,wd) subsample sweep; Stage 4 confirmatory
                       SAM rho=0.5 / WD=1e-3 (full-data, checkpoint rerun
                       2026-09-10); Stage 4b Adam+WD=1e-3 confound control.

All figures are written to artifacts/sam_investigation/figures/ together with a
figures_manifest.json describing each one (what it shows, which thesis section it
suits, and its caveats).

Data sources (read-only):
  artifacts/reports/baseline_gap_scorecard.json, baseline_training_history.json
  artifacts/reports/sam_gap_scorecard_rho_0p{01,05,1,2}.json + histories
  artifacts/sam_investigation/adam_wd_control/adam_wd_control_{scorecard,history}.json
  artifacts/sam_investigation/confirmatory/{confirmatory_scorecard,confirmatory_history,meta}.json
  artifacts/sam_investigation/stage3_sweep/stage3_sweep_summary.json
  artifacts/checkpoints/baseline/{model.pt,meta.json}
  artifacts/sam_investigation/confirmatory/{model.pt,meta.json}
  artifacts/checkpoints/sam/{model.pt,meta.json}          (SAM rho=0.2)
  artifacts/predictions/baseline_{val,test}.parquet
  artifacts/data/gdsc_processed.parquet

Run from repo root:
    python -m src.investigations.stage4c_figures            # all figures
    python -m src.investigations.stage4c_figures --no-torch # skip landscape / hessian / pred-vs-actual
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", force=True)
logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[2]
REPORTS   = REPO_ROOT / "artifacts" / "reports"
SAMINV    = REPO_ROOT / "artifacts" / "sam_investigation"
CKPT      = REPO_ROOT / "artifacts" / "checkpoints"
FIGDIR    = SAMINV / "figures"

# --------------------------------------------------------------------------- #
# Condition registry — every full-data DeepCCDS run in the story
# --------------------------------------------------------------------------- #
# phase: "pre" (before the investigation) or "inv" (during it)
# stop : stopping rule used (matters for cross-run comparability)
CONDITIONS = {
    "adam_base": dict(
        label="Adam baseline\n(WD=0)", short="Adam", phase="pre", rho=0.0, wd=0.0,
        color="#3f3f3f", stop="reverse-early-stop",
        scorecard=REPORTS / "baseline_gap_scorecard.json",
        history=REPORTS / "baseline_training_history.json",
    ),
    "sam_r001": dict(
        label="SAM ρ=0.01", short="ρ=0.01", phase="pre", rho=0.01, wd=0.0,
        color="#c6dbef", stop="reverse-early-stop",
        scorecard=REPORTS / "sam_gap_scorecard_rho_0p01.json",
        history=REPORTS / "sam_training_history_rho_0p01.json",
    ),
    "sam_r005": dict(
        label="SAM ρ=0.05", short="ρ=0.05", phase="pre", rho=0.05, wd=0.0,
        color="#9ecae1", stop="reverse-early-stop",
        scorecard=REPORTS / "sam_gap_scorecard_rho_0p05.json",
        history=REPORTS / "sam_training_history_rho_0p05.json",
    ),
    "sam_r01": dict(
        label="SAM ρ=0.1", short="ρ=0.1", phase="pre", rho=0.1, wd=0.0,
        color="#6baed6", stop="reverse-early-stop",
        scorecard=REPORTS / "sam_gap_scorecard_rho_0p1.json",
        history=REPORTS / "sam_training_history_rho_0p1.json",
    ),
    "sam_r02": dict(
        label="SAM ρ=0.2", short="ρ=0.2", phase="pre", rho=0.2, wd=0.0,
        color="#2171b5", stop="reverse-early-stop",
        scorecard=REPORTS / "sam_gap_scorecard_rho_0p2.json",
        history=REPORTS / "sam_training_history_rho_0p2.json",
    ),
    "adam_wd": dict(
        label="Adam+WD\n(4b control)", short="Adam+WD", phase="inv", rho=0.0, wd=1e-3,
        color="#f16913", stop="val-plateau",
        scorecard=SAMINV / "adam_wd_control" / "adam_wd_control_scorecard.json",
        history=SAMINV / "adam_wd_control" / "adam_wd_control_history.json",
    ),
    "sam_r05_wd": dict(
        label="SAM ρ=0.5+WD\n(Stage 4)", short="ρ=0.5+WD", phase="inv", rho=0.5, wd=1e-3,
        color="#238b45", stop="val-plateau",
        scorecard=SAMINV / "confirmatory" / "confirmatory_scorecard.json",
        history=SAMINV / "confirmatory" / "confirmatory_history.json",
        meta=SAMINV / "confirmatory" / "meta.json",
    ),
}

ORDER = ["adam_base", "sam_r001", "sam_r005", "sam_r01", "sam_r02", "adam_wd", "sam_r05_wd"]

_manifest: list[dict] = []


def _load(p: Path):
    with open(p, encoding="utf-8") as f:
        return json.load(f)


def metrics(key: str) -> dict:
    """Unified final eval-mode metric dict for a condition, filling from the
    scorecard and (for the confirmatory run) its checkpoint meta."""
    c = CONDITIONS[key]
    sc = _load(c["scorecard"])
    m = dict(
        gap=sc.get("generalisation_gap"),
        train_loss=sc.get("final_train_loss"),
        val_loss=sc.get("final_val_loss"),
        test_loss=sc.get("final_test_loss"),
        val_r2=sc.get("val_r2"), train_r2=sc.get("train_r2"), test_r2=sc.get("test_r2"),
        val_rmse=sc.get("val_rmse"), train_rmse=sc.get("train_rmse"), test_rmse=sc.get("test_rmse"),
        n_epochs=sc.get("n_epochs"),
    )
    if "meta" in c:  # confirmatory: pull train/test R2/RMSE from checkpoint meta
        mm = _load(c["meta"])
        for k in ("train_r2", "test_r2", "train_rmse", "test_rmse"):
            if m.get(k) is None:
                m[k] = mm.get(k)
    return m


def history(key: str):
    """Return (epochs, train_loss, val_loss, gap) arrays for a condition."""
    h = _load(CONDITIONS[key]["history"])
    ep = np.array([r["epoch"] for r in h], float)
    tr = np.array([r["train_loss"] for r in h], float)
    va = np.array([r["val_loss"] for r in h], float)
    gp = np.array([r.get("gap", r.get("generalisation_gap")) for r in h], float)
    return ep, tr, va, gp


def _save(fig, name: str, caption: str, section: str, caveats: str = ""):
    import time
    import matplotlib.pyplot as plt
    FIGDIR.mkdir(parents=True, exist_ok=True)
    out = FIGDIR / name
    last = None
    for attempt in range(5):  # WSL2 /mnt/c (drvfs) occasionally returns EINVAL on write
        try:
            fig.savefig(out, dpi=150, bbox_inches="tight")
            break
        except OSError as e:  # noqa: PERF203
            last = e
            logger.warning("savefig %s failed (attempt %d): %s — retrying", name, attempt + 1, e)
            time.sleep(0.5 * (attempt + 1))
    else:
        plt.close(fig)
        raise last
    plt.close(fig)
    _manifest.append(dict(file=name, caption=caption, thesis_section=section, caveats=caveats))
    logger.info("saved %s", out)


# ========================================================================= #
# GROUP 1 — the full story
# ========================================================================= #
def fig_gap_full_story():
    import matplotlib.pyplot as plt
    keys = ORDER
    gaps = [metrics(k)["gap"] for k in keys]
    colors = [CONDITIONS[k]["color"] for k in keys]
    labels = [CONDITIONS[k]["label"] for k in keys]
    fig, ax = plt.subplots(figsize=(11, 5.6))
    bars = ax.bar(range(len(keys)), gaps, color=colors, edgecolor="black", linewidth=0.6)
    for b, g in zip(bars, gaps):
        ax.text(b.get_x() + b.get_width() / 2, g + 0.006, f"{g:.3f}", ha="center", va="bottom", fontsize=9)
    ax.axhline(gaps[0], ls="--", lw=0.9, color="#3f3f3f", alpha=0.6)
    ax.set_xticks(range(len(keys)))
    ax.set_xticklabels(labels, fontsize=8.5, rotation=15, ha="right")
    ax.set_ylabel("generalisation gap  (val − train loss, eval-mode)")
    ax.set_title("Generalisation gap across every full-data DeepCCDS run")
    ax.axvspan(-0.5, 4.5, color="#2171b5", alpha=0.05)
    ax.axvspan(4.5, 6.5, color="#238b45", alpha=0.06)
    ax.set_ylim(0, max(gaps) * 1.18)
    ax.text(2.0, max(gaps) * 1.14, "before the investigation — SAM ρ≤0.2 barely moves the gap",
            ha="center", va="top", fontsize=8.5, style="italic", color="#2171b5")
    ax.text(5.5, max(gaps) * 1.14, "investigation\noutcome", ha="center", va="top",
            fontsize=8.5, style="italic", color="#238b45")
    _save(fig, "01_gap_full_story_bars.png",
          "Eval-mode generalisation gap for all seven full-data runs. Pre-investigation "
          "Adam baseline overfits (0.337) and SAM ρ=0.01–0.2 fails to help; the "
          "investigation's ρ=0.5+WD run collapses the gap to 0.049.",
          "Results — overview / Discussion",
          "Pre-investigation runs used the reverse-early-stopping rule; the two "
          "investigation runs used val-loss-plateau. Cross-phase gap comparison is "
          "directional; the clean matched pair is Adam+WD vs SAM ρ=0.5+WD.")


def fig_gap_vs_rho():
    import matplotlib.pyplot as plt
    pre = ["adam_base", "sam_r001", "sam_r005", "sam_r01", "sam_r02"]
    rho = [CONDITIONS[k]["rho"] for k in pre]
    gap = [metrics(k)["gap"] for k in pre]
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(rho, gap, "o-", color="#2171b5", label="full-data, WD=0 (pre-investigation)")
    for x, y in zip(rho, gap):
        ax.annotate(f"{y:.3f}", (x, y), textcoords="offset points", xytext=(0, 8), ha="center", fontsize=8)
    g05 = metrics("sam_r05_wd")["gap"]
    ax.plot([0.5], [g05], "*", ms=18, color="#238b45", label="SAM ρ=0.5 + WD=1e-3 (Stage 4)")
    ax.annotate(f"{g05:.3f}", (0.5, g05), textcoords="offset points", xytext=(0, 12), ha="center",
                fontsize=9, color="#238b45", fontweight="bold")
    ax.axvspan(0, 0.25, color="#d62728", alpha=0.06)
    ax.text(0.12, max(gap) * 0.55, "ρ under-perturbation\nregion", ha="center", fontsize=8.5,
            style="italic", color="#d62728")
    ax.set_xlabel("SAM neighbourhood radius  ρ")
    ax.set_ylabel("generalisation gap (eval-mode)")
    ax.set_title("Generalisation gap vs ρ  (full data)")
    ax.legend(fontsize=8.5)
    ax.set_ylim(0, max(gap) * 1.15)
    _save(fig, "02_gap_vs_rho_fulldata.png",
          "Full-data generalisation gap as a function of SAM's ρ. Below ρ≈0.2 the "
          "perturbation is too small to matter (gap ≈ baseline); ρ=0.5 (with WD) is "
          "where the gap collapses.",
          "Results — Stage 1 / Stage 4",
          "The ρ=0.5 point also switches WD from 0 to 1e-3 and uses a different "
          "stopping rule, so it is not a pure ρ-only comparison — see the decomposition figure.")


def fig_gap_decomposition():
    import matplotlib.pyplot as plt
    g_base = metrics("adam_base")["gap"]      # 0.3369
    g_wd = metrics("adam_wd")["gap"]          # 0.2729 (saved Adam+WD checkpoint)
    g_sam = metrics("sam_r05_wd")["gap"]      # 0.0485
    d_wd = g_base - g_wd
    d_sam = g_wd - g_sam
    fig, ax = plt.subplots(figsize=(8.5, 5.4))
    ax.bar(0, g_base, color="#3f3f3f", edgecolor="black", label="remaining gap")
    ax.bar(1, g_wd, color="#3f3f3f", edgecolor="black")
    ax.bar(1, d_wd, bottom=g_wd, color="#f16913", edgecolor="black", label="removed by weight decay")
    ax.bar(2, g_sam, color="#3f3f3f", edgecolor="black")
    ax.bar(2, d_sam, bottom=g_sam, color="#238b45", edgecolor="black", label="removed by SAM ρ=0.5")
    # connector lines
    ax.plot([0.4, 0.6], [g_base, g_base], color="grey", lw=1)
    ax.plot([1.4, 1.6], [g_wd, g_wd], color="grey", lw=1)
    ax.text(0, g_base + 0.008, f"{g_base:.3f}", ha="center", fontsize=9)
    ax.text(1, g_wd + d_wd / 2, f"−{d_wd:.3f}", ha="center", va="center", fontsize=9, color="white", fontweight="bold")
    ax.text(1, g_wd - 0.014, f"{g_wd:.3f}", ha="center", va="top", fontsize=9, color="white")
    ax.text(2, g_sam + d_sam / 2, f"−{d_sam:.3f}", ha="center", va="center", fontsize=9, color="white", fontweight="bold")
    ax.text(2, g_sam + 0.008, f"{g_sam:.3f}", ha="center", va="bottom", fontsize=9)
    ax.set_xticks([0, 1, 2])
    ax.set_xticklabels(["Adam baseline\n(WD=0)", "+ weight decay\n(Adam+WD=1e-3)", "+ SAM ρ=0.5\n(SAM+WD=1e-3)"])
    ax.set_ylabel("generalisation gap (eval-mode)")
    ax.set_title("Where the 86% gap reduction comes from (Stage 4b decomposition)")
    ax.legend(fontsize=8.5, loc="upper right")
    ax.set_ylim(0, g_base * 1.16)
    pct_wd = d_wd / (g_base - g_sam) * 100
    pct_sam = d_sam / (g_base - g_sam) * 100
    ax.text(0.5, g_base * 1.07, f"weight decay ≈ {pct_wd:.0f}% of the total reduction   ·   "
            f"SAM ρ=0.5 ≈ {pct_sam:.0f}%", ha="center", fontsize=8.5,
            bbox=dict(boxstyle="round", fc="#fff8e1", ec="grey"))
    _save(fig, "03_gap_decomposition_waterfall.png",
          f"Decomposition of the total gap reduction ({g_base:.3f} → {g_sam:.3f}): weight decay "
          f"accounts for ~{pct_wd:.0f}%, SAM ρ=0.5 for ~{pct_sam:.0f}%.",
          "Results — Stage 4b / Discussion (confound)",
          "The Adam-baseline → Adam+WD step also carries the stopping-rule change "
          "(reverse-early-stop → val-plateau); the Adam+WD → SAM+WD step is the clean "
          "matched pair (single variable: the SAM wrapper).")


def fig_tradeoff():
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(8, 5.5))
    for k in ORDER:
        m = metrics(k)
        if m["val_r2"] is None:
            continue
        c = CONDITIONS[k]
        ax.scatter(m["gap"], m["val_r2"], s=140, color=c["color"], edgecolor="black",
                   zorder=3, marker="*" if k == "sam_r05_wd" else "o")
        ax.annotate(c["short"], (m["gap"], m["val_r2"]), textcoords="offset points",
                    xytext=(7, 4), fontsize=8.5)
    ax.set_xlabel("generalisation gap (eval-mode)  →  lower is better")
    ax.set_ylabel("validation R²  →  higher is better")
    ax.set_title("Gap vs predictive quality trade-off")
    ax.invert_xaxis()
    ax.margins(x=0.14)
    ax.grid(alpha=0.25)
    ax.text(0.02, 0.03, "x-axis inverted → smaller gap is to the right;\n"
            "SAM ρ=0.5+WD trades ≈0.012 val R² for a ≈0.29 gap reduction",
            transform=ax.transAxes, fontsize=8, style="italic")
    _save(fig, "04_gap_vs_valr2_tradeoff.png",
          "Each run as a point in (gap, val R²) space. SAM ρ=0.5+WD moves far left "
          "(gap 0.049) for a ~0.012 R² cost vs the baseline; Adam+WD sits between.",
          "Results — overview / Discussion",
          "Mixed stopping rules across points (see per-condition table).")


# ========================================================================= #
# GROUP 2 — training dynamics
# ========================================================================= #
def _loss_curves(keys, name, title, caption, section, caveats):
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(9, 5.5))
    for k in keys:
        ep, tr, va, _ = history(k)
        c = CONDITIONS[k]
        ax.plot(ep, tr, "-", color=c["color"], lw=1.3, alpha=0.9)
        ax.plot(ep, va, "--", color=c["color"], lw=1.3, alpha=0.9,
                label=c["label"].replace("\n", " "))
    ax.plot([], [], "k-", label="train loss")
    ax.plot([], [], "k--", label="val loss")
    ax.set_xlabel("epoch")
    ax.set_ylabel("loss (Eq. 2, MSE-based)")
    ax.set_title(title)
    ax.legend(fontsize=8, ncol=2)
    ax.set_ylim(0.5, 2.2)
    _save(fig, name, caption, section, caveats)


def fig_loss_curves_pre():
    _loss_curves(
        ["adam_base", "sam_r001", "sam_r005", "sam_r01", "sam_r02"],
        "05_loss_curves_preinvestigation.png",
        "Training vs validation loss — Adam baseline and the ρ=0.01–0.2 SAM sweep",
        "Pre-investigation training curves. Adding SAM at ρ≤0.2 leaves the train/val "
        "separation essentially unchanged — the observation that opened the investigation.",
        "Results — Stage 1",
        "Train-mode per-epoch losses (not eval-mode); early epochs show spuriously "
        "negative gaps as BN statistics settle.")


def fig_loss_curves_inv():
    _loss_curves(
        ["adam_base", "adam_wd", "sam_r05_wd"],
        "06_loss_curves_investigation.png",
        "Training vs validation loss — baseline vs Adam+WD vs SAM ρ=0.5+WD",
        "Investigation training curves. Under SAM ρ=0.5+WD the train and val curves "
        "track each other closely for the whole run; Adam+WD narrows the separation "
        "partially; the baseline's curves stay far apart.",
        "Results — Stage 4 / 4b",
        "Different stopping rules and epoch counts (90 / 157 / 250). Train-mode losses.")


def fig_gap_per_epoch():
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=True)
    for ax, group, ttl in (
        (axes[0], ["adam_base", "sam_r001", "sam_r005", "sam_r01", "sam_r02"], "pre-investigation"),
        (axes[1], ["adam_base", "adam_wd", "sam_r05_wd"], "investigation"),
    ):
        for k in group:
            ep, _, _, gp = history(k)
            c = CONDITIONS[k]
            ax.plot(ep, gp, color=c["color"], lw=1.4, label=c["label"].replace("\n", " "))
        ax.axhline(0, color="k", lw=0.8)
        ax.set_xlabel("epoch")
        ax.set_title(ttl)
        ax.legend(fontsize=8)
        ax.grid(alpha=0.2)
    axes[0].set_ylabel("generalisation gap  (val − train, per epoch)")
    axes[0].set_ylim(-0.3, 0.6)
    fig.suptitle("Per-epoch generalisation gap over training")
    _save(fig, "07_gap_per_epoch.png",
          "Per-epoch gap trajectories. SAM ρ=0.5+WD (right) holds the gap near zero "
          "from early on; the pre-investigation SAM sweep (left) widens toward the "
          "baseline's ~0.3–0.4.",
          "Results — Stage 1 / Stage 4",
          "Train-mode per-epoch gap; noisier than the eval-mode final numbers and "
          "can be negative early while BN statistics stabilise.")


# ========================================================================= #
# GROUP 3 — final metrics
# ========================================================================= #
def _grouped_metric(metric_keys, ylabel, title, name, caption, section, caveats, ylim=None):
    import matplotlib.pyplot as plt
    keys = ORDER
    x = np.arange(len(keys))
    w = 0.8 / len(metric_keys)
    fig, ax = plt.subplots(figsize=(11, 5))
    for i, mk in enumerate(metric_keys):
        vals = [metrics(k).get(mk) for k in keys]
        vals = [np.nan if v is None else v for v in vals]
        ax.bar(x + i * w - 0.4 + w / 2, vals, w, label=mk.replace("_", " "),
               edgecolor="black", linewidth=0.4)
    ax.set_xticks(x)
    ax.set_xticklabels([CONDITIONS[k]["short"] for k in keys])
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend(fontsize=8.5)
    if ylim:
        ax.set_ylim(*ylim)
    ax.grid(alpha=0.2, axis="y")
    _save(fig, name, caption, section, caveats)


def fig_r2():
    _grouped_metric(["train_r2", "val_r2", "test_r2"], "R²  (axis zoomed to 0.80–0.92)",
                    "R² by split and condition", "09_r2_comparison.png",
                    "Train/val/test R² per condition (y-axis zoomed). Every pre-investigation "
                    "run and Adam+WD keep train R² ≈ 0.90 well above val ≈ 0.86; SAM ρ=0.5+WD "
                    "is the only run whose train R² (0.852) sits right on top of its val R² "
                    "(0.844) — the visual signature of the closed gap.",
                    "Results — Stage 4 / Discussion",
                    "Adam+WD (Stage 4b) recorded val R² only — train/test bars are absent for it.",
                    ylim=(0.80, 0.92))


def fig_rmse():
    _grouped_metric(["train_rmse", "val_rmse", "test_rmse"], "RMSE, ln IC50 units  (axis zoomed)",
                    "RMSE by split and condition", "10_rmse_comparison.png",
                    "Train/val/test RMSE per condition (y-axis zoomed). Same story as R²: under "
                    "SAM ρ=0.5+WD train RMSE rises toward the val level rather than val error "
                    "dropping toward train.",
                    "Results — Stage 4",
                    "Adam+WD recorded val RMSE only.",
                    ylim=(0.75, 1.15))


def fig_losses():
    _grouped_metric(["train_loss", "val_loss", "test_loss"], "final loss (eval-mode)",
                    "Final eval-mode loss by split and condition", "11_final_loss_comparison.png",
                    "Final eval-mode loss on each split. The train/val separation IS the "
                    "generalisation gap; it is visibly smallest for SAM ρ=0.5+WD.",
                    "Results — Stage 4 / 4b", "")


# ========================================================================= #
# GROUP 4 — Stage 3 subsample sweep
# ========================================================================= #
def _stage3():
    return _load(SAMINV / "stage3_sweep" / "stage3_sweep_summary.json")


def fig_stage3_heatmaps():
    import matplotlib.pyplot as plt
    d = _stage3()
    res = d["results"]
    rhos = sorted({r["rho"] for r in res})
    combos = [(lr, wd) for lr in sorted({r["lr"] for r in res}, reverse=True)
              for wd in sorted({r["wd"] for r in res})]
    def grid(field):
        g = np.full((len(rhos), len(combos)), np.nan)
        for r in res:
            i = rhos.index(r["rho"]); j = combos.index((r["lr"], r["wd"]))
            g[i, j] = r[field]
        return g
    import matplotlib.patheffects as pe
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    for ax, field, ttl, cmap in ((axes[0], "generalisation_gap", "generalisation gap", "viridis_r"),
                                 (axes[1], "val_r2", "validation R²", "viridis")):
        g = grid(field)
        im = ax.imshow(g, cmap=cmap, aspect="auto")
        ax.set_xticks(range(len(combos)))
        ax.set_xticklabels([f"lr={lr}\nwd={wd}" for lr, wd in combos], fontsize=8)
        ax.set_yticks(range(len(rhos)))
        ax.set_yticklabels([f"ρ={r}" for r in rhos])
        for i in range(len(rhos)):
            for j in range(len(combos)):
                ax.text(j, i, f"{g[i, j]:.2f}", ha="center", va="center", fontsize=8.5,
                        color="white", fontweight="bold",
                        path_effects=[pe.withStroke(linewidth=2, foreground="black")])
        ax.set_title(ttl)
        fig.colorbar(im, ax=ax, fraction=0.046)
    fig.suptitle("Stage 3 joint (ρ, lr, wd) sweep — 10% training subsample")
    _save(fig, "12_stage3_sweep_heatmaps.png",
          "Stage 3 sweep grid. Gap falls monotonically with ρ and with WD=1e-3; "
          "val R² also falls at high ρ on the subsample (a subsample artefact — see "
          "Stage 3 notes).",
          "Results — Stage 3",
          "10% subsample: absolute gap values are inflated and not comparable to "
          "full-data numbers; only the ordering across cells is used.")


def fig_stage3_gap_vs_rho():
    import matplotlib.pyplot as plt
    res = _stage3()["results"]
    fig, ax = plt.subplots(figsize=(8, 5))
    for wd, style in ((0.0, "o--"), (1e-3, "s-")):
        pts = sorted([(r["rho"], r["generalisation_gap"]) for r in res if r["wd"] == wd])
        xs = [p[0] for p in pts]
        # average the two LR values at each rho
        import collections
        agg = collections.defaultdict(list)
        for r in res:
            if r["wd"] == wd:
                agg[r["rho"]].append(r["generalisation_gap"])
        xs = sorted(agg)
        ys = [np.mean(agg[x]) for x in xs]
        ax.plot(xs, ys, style, label=f"WD={wd:g}", lw=1.5)
    ax.set_xlabel("SAM ρ")
    ax.set_ylabel("generalisation gap (10% subsample, mean over LR)")
    ax.set_title("Stage 3: gap decreases with ρ and with weight decay")
    ax.legend()
    ax.grid(alpha=0.25)
    _save(fig, "13_stage3_gap_vs_rho_subsample.png",
          "Stage 3 subsample gap vs ρ, split by weight decay. Both levers reduce the "
          "gap; they combine roughly additively, motivating the ρ=0.5 + WD=1e-3 pick.",
          "Results — Stage 3",
          "Subsample gaps are inflated vs full data; qualitative trend only.")


# ========================================================================= #
# GROUP 5 — loss geometry (needs torch + checkpoints)
# ========================================================================= #
def _load_model(ckpt_dir: Path):
    import torch
    from src.common.checkpoint import load_checkpoint
    from src.model.deepccds import DeepCCDS
    meta = _load(ckpt_dir / "meta.json")
    model = DeepCCDS(**meta["arch_config"])
    load_checkpoint(model, ckpt_dir)
    model.eval()
    return model, meta


def _subset_loader(split: str, n: int, seed: int = 0):
    import torch
    from torch.utils.data import DataLoader, Subset
    from src.data.dataset import DeepCCDSDataset
    ds = DeepCCDSDataset(split)
    idx = np.random.default_rng(seed).choice(len(ds), size=min(n, len(ds)), replace=False)
    return DataLoader(Subset(ds, sorted(idx.tolist())), batch_size=2048, shuffle=False)


def _eval_loss(model, loader, device="cpu"):
    import torch
    from src.model.deepccds import compute_loss
    tot, n = 0.0, 0
    with torch.no_grad():
        for _, p, mut, drug, y in loader:
            out = model(p, mutation=mut, drug=drug)
            tot += compute_loss(out, y).item() * p.size(0)
            n += p.size(0)
    return tot / n


def _filter_norm_direction(model, seed):
    import torch
    g = torch.Generator().manual_seed(seed)
    direction = []
    for pr in model.parameters():
        d = torch.randn(pr.shape, generator=g)
        d.mul_(pr.norm() / (d.norm() + 1e-12))
        direction.append(d)
    return direction


def _landscape_curve(model, loader, seeds, n_points, span):
    import torch
    alphas = np.linspace(-span, span, n_points)
    curves = []
    base = [p.detach().clone() for p in model.parameters()]
    for s in seeds:
        d = _filter_norm_direction(model, s)
        losses = []
        with torch.no_grad():
            for a in alphas:
                for p, b, di in zip(model.parameters(), base, d):
                    p.copy_(b + float(a) * di)
                losses.append(_eval_loss(model, loader))
            for p, b in zip(model.parameters(), base):
                p.copy_(b)
        curves.append(losses)
    return alphas, np.array(curves)


def fig_loss_landscape(seeds=(0, 1, 2), n_points=25, span=0.06, n_sub=8000):
    import matplotlib.pyplot as plt
    specs = [("adam_base", CKPT / "baseline", "#3f3f3f"),
             ("sam_r05_wd", SAMINV / "confirmatory", "#238b45")]
    loaders = {"train": _subset_loader("train", n_sub, 100),
               "val": _subset_loader("val", n_sub, 101)}
    fig, ax = plt.subplots(figsize=(8.5, 5.5))
    floor = np.inf
    for key, cdir, col in specs:
        model, _ = _load_model(cdir)
        for split, ls in (("train", "-"), ("val", "--")):
            alphas, curves = _landscape_curve(model, loaders[split], seeds, n_points, span)
            mu, sd = curves.mean(0), curves.std(0)
            floor = min(floor, mu.min())
            ax.plot(alphas, mu, ls, color=col, lw=1.7,
                    label=f"{CONDITIONS[key]['short']} — {split}")
            ax.fill_between(alphas, mu - sd, mu + sd, color=col, alpha=0.15)
    ax.axvline(0, color="grey", lw=0.7, ls=":")
    ax.set_xlabel("perturbation magnitude α  (filter-normalised random direction)")
    ax.set_ylabel("loss (Eq. 2, MSE-based)")
    ax.set_title("Filter-normalised loss landscape — Adam baseline vs SAM ρ=0.5+WD")
    ax.set_ylim(floor * 0.95, floor + 3.0)   # focus on the basin; sharp curves escape upward
    ax.legend(fontsize=8.5)
    _save(fig, "14_loss_landscape_stage4.png",
          f"1-D filter-normalised loss landscape (Li et al. 2018 style), mean ± sd over "
          f"{len(seeds)} random directions, on {n_sub}-row subsamples, span ±{span}. A "
          f"shallower rise away from α=0 and a smaller train–val separation indicate a "
          f"flatter minimum that generalises better. Curves that leave the top of the "
          f"axis have escaped the basin within this α range.",
          "Results — Stage 4 / Discussion (flat minima)",
          "BN running statistics are held fixed while weights are perturbed (Li et al. "
          "recommend recomputing them); per-tensor (not per-filter) normalisation, so α "
          "is only comparable within this figure; single 1-D slice family; subsampled "
          "data. Read alongside figure 15 (Hessian λ_max).")


def _landscape_grid(model, loader, seed1, seed2, n_points, span):
    """2-D filter-normalised loss grid over two independent random directions
    (Li et al. 2018 style, extended from the 1-D slice in _landscape_curve)."""
    import torch
    alphas = np.linspace(-span, span, n_points)
    betas = np.linspace(-span, span, n_points)
    base = [p.detach().clone() for p in model.parameters()]
    d1 = _filter_norm_direction(model, seed1)
    d2 = _filter_norm_direction(model, seed2)
    grid = np.zeros((n_points, n_points))
    with torch.no_grad():
        for i, a in enumerate(alphas):
            for j, b in enumerate(betas):
                for p, bp, di, dj in zip(model.parameters(), base, d1, d2):
                    p.copy_(bp + float(a) * di + float(b) * dj)
                grid[i, j] = _eval_loss(model, loader)
            logger.info("  2-D landscape row %d/%d done", i + 1, n_points)
        for p, bp in zip(model.parameters(), base):
            p.copy_(bp)
    return alphas, betas, grid


def fig_loss_landscape_2d(seed_pair=(10, 11), n_points=15, span=0.06, n_sub=4000):
    """Full 2-D extension of figure 14: one 3-D surface (+ base contour) per
    condition, train and val overlaid, across all four full-data checkpoints
    that have loadable weights (Adam baseline, SAM rho=0.2, Adam+WD control,
    SAM rho=0.5+WD)."""
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 — registers 3-D projection

    specs = [
        ("adam_base", CKPT / "baseline"),
        ("sam_r02", CKPT / "sam"),
        ("adam_wd", SAMINV / "adam_wd_control"),
        ("sam_r05_wd", SAMINV / "confirmatory"),
    ]
    loaders = {"train": _subset_loader("train", n_sub, 300),
               "val": _subset_loader("val", n_sub, 301)}

    fig = plt.figure(figsize=(14, 12))
    zmin, zmax = np.inf, -np.inf
    panels = []
    for key, cdir in specs:
        logger.info("2-D landscape: %s", key)
        model, _ = _load_model(cdir)
        alphas, betas, g_tr = _landscape_grid(model, loaders["train"], *seed_pair, n_points, span)
        _, _, g_va = _landscape_grid(model, loaders["val"], *seed_pair, n_points, span)
        zmin = min(zmin, g_tr.min(), g_va.min())
        zmax = max(zmax, np.percentile(g_tr, 90), np.percentile(g_va, 90))
        panels.append((key, cdir, alphas, betas, g_tr, g_va))

    for idx, (key, cdir, alphas, betas, g_tr, g_va) in enumerate(panels):
        ax = fig.add_subplot(2, 2, idx + 1, projection="3d")
        A, B = np.meshgrid(alphas, betas, indexing="ij")
        c = CONDITIONS[key]
        g_tr_c = np.clip(g_tr, zmin, zmax)
        g_va_c = np.clip(g_va, zmin, zmax)
        ax.plot_surface(A, B, g_tr_c, color=c["color"], alpha=0.55, linewidth=0, antialiased=True)
        ax.plot_wireframe(A, B, g_va_c, color="#d62728", alpha=0.6, linewidth=0.5, rstride=1, cstride=1)
        ax.contourf(A, B, g_tr_c, zdir="z", offset=zmin, cmap="Greys", alpha=0.35, levels=12)
        ax.set_zlim(zmin, zmax)
        ax.set_xlabel("α (dir. 1)", fontsize=8)
        ax.set_ylabel("β (dir. 2)", fontsize=8)
        ax.set_zlabel("loss", fontsize=8)
        ax.set_title(c["short"], fontsize=10)
        ax.tick_params(labelsize=7)

    fig.subplots_adjust(top=0.90, hspace=0.05, wspace=0.05)
    fig.legend(handles=[
        plt.Line2D([0], [0], color="#555555", lw=6, alpha=0.55, label="train loss surface"),
        plt.Line2D([0], [0], color="#d62728", lw=1.5, label="val loss (wireframe)"),
    ], loc="upper center", ncol=2, fontsize=9, bbox_to_anchor=(0.5, 0.965))
    fig.suptitle("2-D filter-normalised loss landscape — all four full-data checkpoints", y=0.995)
    _save(fig, "17_loss_landscape_2d.png",
          f"Full 2-D filter-normalised loss landscape (Li et al. 2018), one random "
          f"direction pair per checkpoint, {n_points}x{n_points} grid, span ±{span}, "
          f"{n_sub}-row subsamples. Train surface (shaded, per-condition colour) with "
          f"val surface overlaid as a red wireframe and a train-loss contour projected "
          f"onto the base plane. The Adam baseline and SAM ρ=0.2 panels show a "
          f"visibly curved bowl with train/val surfaces separating away from the "
          f"origin; the Adam+WD and SAM ρ=0.5+WD panels are markedly flatter with "
          f"the two surfaces closer together across the whole grid — the 2-D "
          f"generalisation of figure 14's 1-D result.",
          "Results — Stage 4 / Discussion (flat minima)",
          "Single direction pair per checkpoint (no averaging over seeds, unlike "
          "figure 14's 3-seed 1-D average) — a coarser, higher-variance estimate "
          "traded for the extra dimension. Same BN-frozen / per-tensor-normalised / "
          "subsampled caveats as figure 14. z-axis clipped to the shared 90th-"
          "percentile ceiling across all four panels for a comparable view; sharp "
          "curves may be truncated. Read alongside figures 14-15.")


def fig_hessian(n_iter=25, batch=8192):
    import matplotlib.pyplot as plt
    import torch
    from src.model.deepccds import compute_loss
    specs = [("adam_base", CKPT / "baseline", "#3f3f3f"),
             ("sam_r02", CKPT / "sam", "#2171b5"),
             ("sam_r05_wd", SAMINV / "confirmatory", "#238b45")]
    loader = _subset_loader("train", batch, 200)
    b = next(iter(loader))
    _, p, mut, drug, y = b
    tops = []
    for key, cdir, _ in specs:
        model, _ = _load_model(cdir)
        params = [q for q in model.parameters() if q.requires_grad]
        out = model(p, mutation=mut, drug=drug)
        loss = compute_loss(out, y)
        grads = torch.autograd.grad(loss, params, create_graph=True)
        v = [torch.randn_like(q) for q in params]
        vnorm = torch.sqrt(sum((vi ** 2).sum() for vi in v))
        v = [vi / vnorm for vi in v]
        eig = 0.0
        for _ in range(n_iter):
            Hv = torch.autograd.grad(grads, params, grad_outputs=v, retain_graph=True)
            eig = float(sum((hi * vi).sum() for hi, vi in zip(Hv, v)))
            hn = torch.sqrt(sum((hi ** 2).sum() for hi in Hv))
            v = [hi / (hn + 1e-12) for hi in Hv]
        tops.append(abs(eig))
        logger.info("top Hessian eigenvalue  %-10s = %.4g", key, abs(eig))
    fig, ax = plt.subplots(figsize=(7, 4.5))
    cols = [s[2] for s in specs]
    ax.bar([CONDITIONS[k]["short"] for k, _, _ in specs], tops, color=cols,
           edgecolor="black")
    for i, t in enumerate(tops):
        ax.text(i, t, f"{t:.3g}", ha="center", va="bottom", fontsize=9)
    ax.set_ylabel("top Hessian eigenvalue  (λ_max, power iteration)")
    ax.set_title("Local sharpness of the trained minimum")
    _save(fig, "15_hessian_top_eigenvalue.png",
          f"Largest Hessian eigenvalue of the training loss at each trained solution "
          f"(power iteration, {n_iter} steps, {batch}-sample batch). Lower = flatter "
          f"minimum. This is the quantitative companion to the landscape figure.",
          "Results — Discussion (flat minima)",
          "Single batch, single random init for power iteration; eigenvalue scale is "
          "relative, compare across bars not to an absolute reference. BN in eval mode.")


# ========================================================================= #
# GROUP 6 — predictions
# ========================================================================= #
def _predict(ckpt_dir: Path, split: str):
    import torch
    model, _ = _load_model(ckpt_dir)
    loader = _subset_loader(split, 10 ** 9, 0)  # full split
    rid, pred = [], []
    with torch.no_grad():
        for row_id, p, mut, drug, y in loader:
            out = model(p, mutation=mut, drug=drug)
            rid.append(row_id.numpy()); pred.append(out["prediction"].numpy())
    return np.concatenate(rid), np.concatenate(pred)


def fig_pred_vs_actual(split="val"):
    import matplotlib.pyplot as plt
    import pandas as pd
    truth = pd.read_parquet(REPO_ROOT / "artifacts" / "data" / "gdsc_processed.parquet",
                            columns=["row_id", "split", "ln_ic50"])
    truth = truth[truth.split == split].set_index("row_id")["ln_ic50"]
    base = pd.read_parquet(REPO_ROOT / "artifacts" / "predictions" / f"baseline_{split}.parquet")
    base = base.set_index("row_id")["ln_ic50_pred"]
    rid, pr = _predict(SAMINV / "confirmatory", split)
    sam = pd.Series(pr, index=rid)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.4), sharex=True, sharey=True)
    for ax, s, name, col in ((axes[0], base, "Adam baseline", "#3f3f3f"),
                             (axes[1], sam, "SAM ρ=0.5 + WD=1e-3", "#238b45")):
        j = truth.index.intersection(s.index)
        t, p = truth.loc[j].to_numpy(), s.loc[j].to_numpy()
        ax.hexbin(t, p, gridsize=45, cmap="Greys", mincnt=1)
        lo, hi = np.percentile(np.r_[t, p], [1, 99])
        ax.plot([lo, hi], [lo, hi], color=col, lw=1.5)
        ss_res = np.sum((t - p) ** 2); ss_tot = np.sum((t - t.mean()) ** 2)
        r2 = 1 - ss_res / ss_tot
        rmse = np.sqrt(np.mean((t - p) ** 2))
        ax.set_title(f"{name}\nR²={r2:.3f}  RMSE={rmse:.3f}")
        ax.set_xlabel("true ln IC50")
        lo2, hi2 = np.percentile(np.r_[t, p], [0.2, 99.8])
        ax.set_xlim(lo2, hi2); ax.set_ylim(lo2, hi2)
        ax.set_aspect("equal", adjustable="box")
    axes[0].set_ylabel("predicted ln IC50")
    fig.suptitle(f"Predicted vs actual — {split} split")
    _save(fig, "16_pred_vs_actual.png",
          f"Predicted vs true ln IC50 on the {split} split, Adam baseline vs SAM "
          f"ρ=0.5+WD. Comparable predictive accuracy; SAM's advantage is the "
          f"generalisation gap, not raw {split} accuracy.",
          "Results — Stage 4",
          "SAM ρ=0.5 predictions regenerated from the rerun checkpoint; baseline "
          "predictions are the committed parquet.")


# ========================================================================= #
def per_condition_table():
    rows = []
    for k in ORDER:
        c = CONDITIONS[k]; m = metrics(k)
        rows.append(dict(key=k, label=c["label"].replace("\n", " "), phase=c["phase"],
                         rho=c["rho"], wd=c["wd"], stopping_rule=c["stop"],
                         n_epochs=m["n_epochs"], gap=m["gap"], val_r2=m["val_r2"],
                         val_rmse=m["val_rmse"]))
    (FIGDIR).mkdir(parents=True, exist_ok=True)
    with open(FIGDIR / "conditions_table.json", "w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2)
    logger.info("wrote %s", FIGDIR / "conditions_table.json")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-torch", action="store_true", help="skip landscape / hessian / pred-vs-actual")
    args = ap.parse_args()

    jobs = [
        fig_gap_full_story, fig_gap_vs_rho, fig_gap_decomposition, fig_tradeoff,
        fig_loss_curves_pre, fig_loss_curves_inv, fig_gap_per_epoch,
        fig_r2, fig_rmse, fig_losses,
        fig_stage3_heatmaps, fig_stage3_gap_vs_rho,
    ]
    if not args.no_torch:
        jobs += [fig_loss_landscape, fig_loss_landscape_2d, fig_hessian, fig_pred_vs_actual]

    for j in jobs:
        try:
            j()
        except Exception as e:  # noqa: BLE001 — one bad figure must not sink the rest
            logger.exception("FAILED: %s (%s)", j.__name__, e)

    per_condition_table()
    with open(FIGDIR / "figures_manifest.json", "w", encoding="utf-8") as f:
        json.dump(_manifest, f, indent=2)
    logger.info("wrote %s  (%d figures)", FIGDIR / "figures_manifest.json", len(_manifest))


if __name__ == "__main__":
    main()
