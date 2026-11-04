"""
Part 1 of the real-data preparation task: per-drug tested-cell-line
distribution across the 413 GDSC drugs in artifacts/data/gdsc_processed.parquet.

Purely descriptive -- informs (but does not decide) the MIN_SAMPLES_PER_DRUG
choice made in src/data/prepare_real_extraction_data.py (Part 3). See
artifacts/data_prep/report.md for how these numbers were used.

Usage:
    python -m src.data.analyze_drug_distribution
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

ARTIFACTS_DIR = Path(__file__).resolve().parents[2] / "artifacts"
DATA_DIR = ARTIFACTS_DIR / "data"
OUT_DIR = ARTIFACTS_DIR / "data_prep"
DATA_PREP_CONFIG = Path(__file__).resolve().parents[2] / "configs" / "data_prep.yaml"


def compute_distribution(data_dir: Path = DATA_DIR) -> pd.Series:
    df = pd.read_parquet(data_dir / "gdsc_processed.parquet", columns=["drug_name"])
    return df.groupby("drug_name").size().sort_values()


def analyze(data_dir: Path = DATA_DIR, out_dir: Path = OUT_DIR,
            config_path: Path = DATA_PREP_CONFIG) -> dict:
    with open(config_path, "r", encoding="utf-8") as f:
        data_prep_cfg = yaml.safe_load(f)
    low_sanity_floor = data_prep_cfg["drug_shortlist"]["low_sanity_floor"]

    counts = compute_distribution(data_dir)

    quantiles = counts.quantile([0.01, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99])
    flagged = counts[counts < low_sanity_floor]

    summary = {
        "n_drugs": int(counts.size),
        "n_cell_lines_total": int(pd.read_parquet(data_dir / "gdsc_processed.parquet",
                                                    columns=["cell_name"])["cell_name"].nunique()),
        "n_rows_total": int(counts.sum()),
        "min": int(counts.min()),
        "max": int(counts.max()),
        "mean": float(counts.mean()),
        "std": float(counts.std()),
        "median": float(counts.median()),
        "quantiles": {str(q): float(v) for q, v in quantiles.items()},
        "low_sanity_floor": low_sanity_floor,
        "n_drugs_below_low_sanity_floor": int(flagged.size),
        "drugs_below_low_sanity_floor": {k: int(v) for k, v in flagged.items()},
        "smallest_10_drugs": {k: int(v) for k, v in counts.head(10).items()},
        "largest_5_drugs": {k: int(v) for k, v in counts.tail(5).items()},
    }

    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "per_drug_sample_distribution.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    _plot_histogram(counts, low_sanity_floor, out_dir / "per_drug_sample_distribution.png")

    print(f"n_drugs={summary['n_drugs']} min={summary['min']} max={summary['max']} "
          f"median={summary['median']} mean={summary['mean']:.1f}")
    print(f"drugs below low-sanity floor ({low_sanity_floor}): {summary['n_drugs_below_low_sanity_floor']} "
          f"-> {list(summary['drugs_below_low_sanity_floor'].keys())}")
    print(f"wrote {out_dir / 'per_drug_sample_distribution.json'}")
    print(f"wrote {out_dir / 'per_drug_sample_distribution.png'}")
    return summary


def _plot_histogram(counts: pd.Series, low_sanity_floor: int, out_path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(counts.values, bins=40, color="#4C72B0", edgecolor="white")
    ax.axvline(low_sanity_floor, color="crimson", linestyle="--",
                label=f"low-sanity floor ({low_sanity_floor})")
    ax.axvline(counts.median(), color="black", linestyle=":", label=f"median ({counts.median():.0f})")
    ax.set_xlabel("tested cell lines per drug")
    ax.set_ylabel("number of drugs")
    ax.set_title(f"Per-drug tested cell-line counts (n={counts.size} drugs)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    analyze()
