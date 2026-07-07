"""
Sanity tests for src.common.schema — confirms the §0.4 rule-pool
validator actually catches malformed artifacts before they'd be
written to disk. Run with: pytest tests/
"""

import pytest

from src.common.schema import validate_rule_pool


def _valid_rule_pool() -> dict:
    return {
        "meta": {
            "extractor": "ga",
            "condition": "baseline",
            "runs": 30,
            "config_hash": "abc123def456",
        },
        "rules": [
            {
                "id": "ga_baseline_0001",
                "conditions": [
                    {"feature": "EGFR_expr", "partition": "High"},
                    {"feature": "TP53_expr", "partition": "Low"},
                ],
                "genes": ["EGFR", "TP53"],
                "predicted_class": "resistant",
                "length": 2,
                "fidelity_f1": 0.82,
                "coverage": 0.15,
                "run_ids": [1, 7, 22],
                "non_dominated": True,
            }
        ],
    }


def test_valid_pool_passes():
    assert validate_rule_pool(_valid_rule_pool()) == []


def test_missing_meta_key_fails():
    pool = _valid_rule_pool()
    del pool["meta"]["config_hash"]
    errors = validate_rule_pool(pool)
    assert any("config_hash" in e for e in errors)


def test_bad_extractor_value_fails():
    pool = _valid_rule_pool()
    pool["meta"]["extractor"] = "random_forest"
    errors = validate_rule_pool(pool)
    assert any("extractor" in e for e in errors)


def test_empty_genes_fails():
    pool = _valid_rule_pool()
    pool["rules"][0]["genes"] = []
    errors = validate_rule_pool(pool)
    assert any("genes" in e for e in errors)


def test_length_mismatch_fails():
    pool = _valid_rule_pool()
    pool["rules"][0]["length"] = 5  # actual conditions list has 2
    errors = validate_rule_pool(pool)
    assert any("length" in e for e in errors)


def test_fidelity_out_of_range_fails():
    pool = _valid_rule_pool()
    pool["rules"][0]["fidelity_f1"] = 1.5
    errors = validate_rule_pool(pool)
    assert any("fidelity_f1" in e for e in errors)


def test_duplicate_ids_fail():
    pool = _valid_rule_pool()
    pool["rules"].append(dict(pool["rules"][0]))  # same id twice
    errors = validate_rule_pool(pool)
    assert any("duplicate id" in e for e in errors)
