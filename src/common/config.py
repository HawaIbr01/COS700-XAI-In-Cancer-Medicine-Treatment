"""
src/common/config.py — Branch Contracts §0.3.

Single source of truth for pipeline.yaml. No arm should hardcode a split
ratio, threshold, or seed — always read it from here.
"""

import hashlib
from pathlib import Path

import yaml

DEFAULT_CONFIG_PATH = Path("configs/pipeline.yaml")

_cache: dict[Path, dict] = {}


def load_config(config_path=DEFAULT_CONFIG_PATH) -> dict:
    config_path = Path(config_path)
    if config_path not in _cache:
        with open(config_path) as f:
            _cache[config_path] = yaml.safe_load(f)
    return _cache[config_path]


def config_hash(config_path=DEFAULT_CONFIG_PATH) -> str:
    """Hash of the raw pipeline.yaml bytes. Every artifact carries this —
    if hashes don't match across arms, the runs being compared are
    incompatible (Branch Contracts §6)."""
    config_path = Path(config_path)
    return hashlib.sha256(config_path.read_bytes()).hexdigest()[:12]


def file_hash(path) -> str:
    """Generic file hash, used for split_hash etc."""
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()[:12]
