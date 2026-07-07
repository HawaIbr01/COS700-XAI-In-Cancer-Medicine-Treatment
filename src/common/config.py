"""
Loads configs/pipeline.yaml and exposes a stable hash of it.

Contract (docs/CONTRACTS.md §0.2 / §0.3):
    No arm hardcodes a split ratio, threshold, generation count, or
    seed — everything comes from here. Every artifact this pipeline
    produces should record config_hash() in its meta.json / meta
    block, so that if two people's artifacts were built from
    different pipeline.yaml settings, that mismatch is visible
    instead of silently corrupting a downstream comparison.

Usage:
    from src.common.config import load_config

    cfg = load_config()
    cfg.seed                  # 42
    cfg.split.train           # 0.8
    cfg.ga.generations        # 50
    cfg.config_hash           # short hash of the resolved YAML
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

DEFAULT_CONFIG_PATH = Path(__file__).resolve().parents[2] / "configs" / "pipeline.yaml"


@dataclass(frozen=True)
class SplitConfig:
    train: float
    val: float
    test: float


@dataclass(frozen=True)
class DiscretisationConfig:
    high_z: float
    low_z: float


@dataclass(frozen=True)
class ResponseBinningConfig:
    target: str
    threshold: Any  # "median" at pilot stage, or a fixed float once locked
    classes: list[str]


@dataclass(frozen=True)
class GAConfig:
    generations: int
    pop_min: int
    pop_max: int
    runs: int


@dataclass(frozen=True)
class NSGAConfig:
    generations: int
    pop: int
    runs: int


@dataclass(frozen=True)
class PipelineConfig:
    seed: int
    split: SplitConfig
    discretisation: DiscretisationConfig
    response_binning: ResponseBinningConfig
    ga: GAConfig
    nsga: NSGAConfig
    raw: dict = field(repr=False, compare=False)
    config_hash: str = field(compare=False)

    def as_dict(self) -> dict:
        return self.raw


def _hash_dict(d: dict) -> str:
    """Deterministic short hash of a config dict, independent of key order."""
    encoded = json.dumps(d, sort_keys=True).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()[:12]


def load_config(path: str | Path = DEFAULT_CONFIG_PATH) -> PipelineConfig:
    """Load configs/pipeline.yaml into a validated, hashable PipelineConfig."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"pipeline.yaml not found at {path}. "
            f"Every arm must read shared parameters from this file — "
            f"see docs/CONTRACTS.md §0.2."
        )

    with open(path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)

    required_top_level = {"seed", "split", "discretisation", "response_binning", "ga", "nsga"}
    missing = required_top_level - raw.keys()
    if missing:
        raise ValueError(f"pipeline.yaml is missing required keys: {sorted(missing)}")

    cfg = PipelineConfig(
        seed=raw["seed"],
        split=SplitConfig(**raw["split"]),
        discretisation=DiscretisationConfig(**raw["discretisation"]),
        response_binning=ResponseBinningConfig(**raw["response_binning"]),
        ga=GAConfig(**raw["ga"]),
        nsga=NSGAConfig(**raw["nsga"]),
        raw=raw,
        config_hash=_hash_dict(raw),
    )

    split_sum = cfg.split.train + cfg.split.val + cfg.split.test
    if abs(split_sum - 1.0) > 1e-6:
        raise ValueError(f"split ratios must sum to 1.0, got {split_sum}")

    return cfg


def config_hash(path: str | Path = DEFAULT_CONFIG_PATH) -> str:
    """Convenience accessor — same hash as load_config(path).config_hash."""
    return load_config(path).config_hash


if __name__ == "__main__":
    cfg = load_config()
    print(f"config_hash: {cfg.config_hash}")
    print(f"seed: {cfg.seed}")
    print(f"split: {cfg.split}")
    print(f"ga: {cfg.ga}")
    print(f"nsga: {cfg.nsga}")
