"""Run-level provenance for the per-drug runner: sha256 of every input file,
environment details, and a check against configs/expected_hashes.json.

Computed once per run (not per cell). Nothing here touches DESIGN or design_hash.
"""
from __future__ import annotations

import hashlib
import json
import platform
import socket
import sys
from importlib import metadata
from pathlib import Path

from src.per_drug import data as D
from src.per_drug.config import ROOT

EXPECTED = ROOT / "configs" / "expected_hashes.json"
CHECKPOINTS = (ROOT / "artifacts/checkpoints/baseline",
               ROOT / "artifacts/sam_investigation/confirmatory",
               ROOT / "artifacts/sam_investigation/adam_wd_control")
PACKAGES = ("torch", "numpy", "pandas", "pyarrow", "scikit-learn", "deap", "pymcdm", "gseapy")


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def input_files() -> list:
    files = [d / "model.pt" for d in CHECKPOINTS]
    for arm, (d, prefix) in D.ARM_PRED.items():
        files += [d / f"{prefix}_{s}.parquet" for s in D.SPLITS]
    files += [D.REAL / "data" / "gdsc_processed.parquet", D.REAL / "data" / "feature_map.json"]
    return files


def _rel(p: Path) -> str:
    return p.resolve().relative_to(ROOT).as_posix()


def input_hashes() -> dict:
    return {_rel(p): sha256_file(p) for p in input_files()}


def _version(pkg: str) -> str:
    try:
        return metadata.version(pkg)
    except metadata.PackageNotFoundError:
        return "not installed"


def _device() -> str:
    try:
        import torch
        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu (torch unavailable)"


def check_expected(hashes: dict, expected_path: Path = EXPECTED) -> list:
    """Returns a list of human-readable problems (empty = all match)."""
    expected = {k: v for k, v in json.loads(expected_path.read_text()).items() if not k.startswith("_")}
    problems = [f"MISMATCH {k}: expected {v}, got {hashes[k]}" for k, v in expected.items()
                if k in hashes and hashes[k] != v]
    problems += [f"NOT EXPECTED (absent from {expected_path.name}): {k}" for k in hashes if k not in expected]
    problems += [f"EXPECTED BUT NOT HASHED: {k}" for k in expected if k not in hashes]
    return problems


def collect(expected_path: Path = EXPECTED) -> dict:
    """Hash all inputs and verify them. Raises SystemExit on any missing file or mismatch."""
    try:
        hashes = input_hashes()
    except FileNotFoundError as e:
        sys.exit(f"REFUSING to start: input file missing: {e.filename}")
    problems = check_expected(hashes, expected_path)
    if problems:
        sys.exit("REFUSING to start: inputs differ from configs/expected_hashes.json:\n  " + "\n  ".join(problems))
    return {
        "input_sha256": hashes,
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "packages": {p: _version(p) for p in PACKAGES},
        "device": _device(),
        "hostname": socket.gethostname(),
    }
