"""
Sanity tests for src.common.config. Run with: pytest tests/

Uses the real configs/pipeline.yaml so these also act as a smoke test
that the checked-in config file stays parseable and internally
consistent as people edit it.
"""

import pytest

from src.common.config import load_config, config_hash


def test_load_config_reads_expected_fields():
    cfg = load_config()
    assert cfg.seed == 42
    assert cfg.ga.generations == 50
    assert cfg.nsga.runs == 30


def test_split_ratios_sum_to_one():
    cfg = load_config()
    total = cfg.split.train + cfg.split.val + cfg.split.test
    assert abs(total - 1.0) < 1e-6


def test_config_hash_is_deterministic():
    assert config_hash() == config_hash()


def test_missing_file_raises_clear_error(tmp_path):
    missing = tmp_path / "does_not_exist.yaml"
    with pytest.raises(FileNotFoundError):
        load_config(missing)


def test_missing_required_key_raises(tmp_path):
    bad_yaml = tmp_path / "bad_pipeline.yaml"
    bad_yaml.write_text("seed: 42\n")  # missing split/discretisation/etc.
    with pytest.raises(ValueError):
        load_config(bad_yaml)
