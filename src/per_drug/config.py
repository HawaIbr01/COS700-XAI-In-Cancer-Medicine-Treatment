"""Frozen design constants for the per-drug rule-extraction pilot (Stage 5b).

Everything that defines the experiment lives here and is hashed into every
manifest (`design_hash`). Changing any value changes the hash, so outputs made
under different designs can never be mixed silently. See
docs/per_drug_analysis_plan.md for the rationale.
"""
from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

ARMS = ("adam", "adam_wd", "sam")
EXTRACTORS = ("ga", "nsga")

DESIGN = {
    "arms": list(ARMS),
    "extractors": list(EXTRACTORS),
    "n_folds": 5,
    "fold_seed": 42,             # cell-line -> fold assignment
    "inner_val_frac": 0.20,      # share of the 4 training folds kept for the GA accuracy term
    "inner_seed": 4242,          # inner split seed (+ fold index)
    "min_cover_n": 25,           # coverage floor, in cell lines (replaces the 2% rule)
    "n_gen": 50,
    "base_seed": 42,             # run seed = base_seed + run_id (run_id = 1..n_seeds), as in the global runs
    "ga": {"pop_min": 35, "pop_max": 45, "cxpb": 0.8, "mutpb": 0.2, "max_age": 8,
           "w_fidelity": 0.45, "w_accuracy": 0.35, "w_comprehensibility": 0.20},
    "nsga": {"pop": 40, "cxpb": 0.9, "mutpb": 0.2},
    "min_heldout_rows_flag": 10,
    "kuncheva_n_features": 572,
}


def design_hash() -> str:
    return hashlib.sha256(json.dumps(DESIGN, sort_keys=True).encode()).hexdigest()[:12]


def git_info() -> dict:
    def run(*a):
        return subprocess.check_output(["git", *a], cwd=ROOT, text=True).strip()
    try:
        commit = run("rev-parse", "HEAD")
        changed = run("diff", "--ignore-cr-at-eol", "--name-only", "HEAD", "--", "src", "scripts").splitlines()
    except Exception:
        return {"git_commit": "unknown", "uncommitted_code_files": None}
    return {"git_commit": commit, "uncommitted_code_files": changed}
