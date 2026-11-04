"""
Read/write + validation for every artifact under artifacts/.

Contract (docs/CONTRACTS.md §0.1 / §0.4):
    Arms communicate through files on disk, not by importing each
    other's code. This module is the enforcement layer for that —
    if an artifact doesn't validate here, it doesn't ship.

    The most important schema is the rule pool (§0.4): both GA and
    NSGA-II must emit exactly this shape, because Evaluation treats
    all four pools uniformly. If you change a schema here, that's a
    breaking change — announce it and update docs/CONTRACTS.md in
    the same PR.

Each artifact type gets:
    - a small dataclass / TypedDict-like validator
    - validate_<thing>(obj) -> list[str]   (empty list = valid)
    - load_<thing>(path) -> dict           (validates on load)
    - save_<thing>(obj, path)              (validates before writing)

Deliberately dependency-light (stdlib json only) so every arm can
import this without pulling in the full pipeline's dependencies.
validate_predictions() is the one exception: it takes an
already-loaded dataframe (the caller owns the pandas dependency),
so this module still never needs to `import pandas` itself.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

# --------------------------------------------------------------------------
# Rule pool schema (artifacts/rules/{ga|nsga}_{baseline|sam}.json)
# --------------------------------------------------------------------------

REQUIRED_RULE_META_KEYS = {"extractor", "condition", "runs", "config_hash"}
REQUIRED_RULE_KEYS = {
    "id", "conditions", "genes", "predicted_class",
    "length", "fidelity_f1", "coverage", "run_ids", "non_dominated",
}
VALID_EXTRACTORS = {"ga", "nsga", "cart"}
# "cart" added 2026-08-08 (Hawa, notebook 06 / CART baseline arm). This is a breaking
# change per CONTRACTS.md §6 (schema.py signature/constant change) -- announced in the
# same PR as 06_cart_baseline.ipynb. CART is a benchmark extractor (CONTRACTS.md Arm 5,
# "CART as global symbolic rule extractor benchmarked against ANN predictions, not a
# competing predictor"), so it needs to pass through the same rule-pool schema as GA/NSGA
# to be scored by the same metrics.py functions and ranked in the same unified PROMETHEE run.
VALID_CONDITIONS = {"baseline", "sam"}
VALID_PARTITIONS = {"Low", "Intermediate", "High"}


def validate_rule_pool(obj: dict, feature_map: dict | None = None) -> list[str]:
    """Validate a rule-pool dict against the §0.4 schema. Returns a list of
    problems found; empty list means the object is valid.

    Args:
        obj: the loaded rule-pool dict.
        feature_map: optionally pass the loaded feature_map.json dict to
            additionally check that every condition's feature and every
            rule's genes actually resolve — required before Evaluation's
            GSEA step can run on this pool (docs/CONTRACTS.md §0.4).
    """
    errors: list[str] = []

    if "meta" not in obj:
        return ["missing top-level 'meta' key"]
    if "rules" not in obj:
        return ["missing top-level 'rules' key"]

    meta = obj["meta"]
    missing_meta = REQUIRED_RULE_META_KEYS - meta.keys()
    if missing_meta:
        errors.append(f"meta missing keys: {sorted(missing_meta)}")
    if meta.get("extractor") not in VALID_EXTRACTORS:
        errors.append(f"meta.extractor must be one of {VALID_EXTRACTORS}, got {meta.get('extractor')!r}")
    if meta.get("condition") not in VALID_CONDITIONS:
        errors.append(f"meta.condition must be one of {VALID_CONDITIONS}, got {meta.get('condition')!r}")

    if not isinstance(obj["rules"], list):
        errors.append("'rules' must be a list")
        return errors

    all_genes: set = set()
    if feature_map is not None:
        all_genes = {v["gene_symbol"] for v in feature_map.values()}

    seen_ids = set()
    for i, rule in enumerate(obj["rules"]):
        missing = REQUIRED_RULE_KEYS - rule.keys()
        if missing:
            errors.append(f"rules[{i}] missing keys: {sorted(missing)}")
            continue

        rid = rule["id"]
        if rid in seen_ids:
            errors.append(f"rules[{i}] duplicate id: {rid!r}")
        seen_ids.add(rid)

        if not rule["genes"]:
            errors.append(f"rules[{i}] ({rid}): 'genes' must be non-empty — "
                           f"GSEA in the evaluation arm depends on it")
        if not isinstance(rule["conditions"], list) or not rule["conditions"]:
            errors.append(f"rules[{i}] ({rid}): 'conditions' must be a non-empty list")
        else:
            for cond in rule["conditions"]:
                if "feature" not in cond or "partition" not in cond:
                    errors.append(f"rules[{i}] ({rid}): malformed condition {cond!r} — "
                                   f"needs 'feature' and 'partition'")
                    continue
                if cond["partition"] not in VALID_PARTITIONS:
                    errors.append(f"rules[{i}] ({rid}): invalid partition {cond['partition']!r}, "
                                   f"must be one of {VALID_PARTITIONS}")
                if feature_map is not None and cond["feature"] not in feature_map:
                    errors.append(f"rules[{i}] ({rid}): feature {cond['feature']!r} "
                                   f"not found in feature_map")
        if rule.get("length") != len(rule.get("conditions", [])):
            errors.append(f"rules[{i}] ({rid}): 'length' does not match len(conditions)")
        if not (0.0 <= rule.get("fidelity_f1", -1) <= 1.0):
            errors.append(f"rules[{i}] ({rid}): 'fidelity_f1' must be in [0, 1]")
        if not (0.0 <= rule.get("coverage", -1) <= 1.0):
            errors.append(f"rules[{i}] ({rid}): 'coverage' must be in [0, 1]")
        if feature_map is not None and rule.get("genes"):
            unresolved = set(rule["genes"]) - all_genes
            if unresolved:
                errors.append(f"rules[{i}] ({rid}): genes not resolvable in feature_map: {unresolved}")

    return errors


def load_rule_pool(path: str | Path, feature_map: dict | None = None) -> dict:
    path = Path(path)
    with open(path, "r", encoding="utf-8") as f:
        obj = json.load(f)
    errors = validate_rule_pool(obj, feature_map=feature_map)
    if errors:
        raise ValueError(f"{path} failed rule-pool validation:\n  " + "\n  ".join(errors))
    return obj


def save_rule_pool(obj: dict, path: str | Path, feature_map: dict | None = None) -> None:
    errors = validate_rule_pool(obj, feature_map=feature_map)
    if errors:
        raise ValueError(f"refusing to save invalid rule pool to {path}:\n  " + "\n  ".join(errors))
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)


# --------------------------------------------------------------------------
# Checkpoint meta schema (artifacts/checkpoints/{baseline|sam}/meta.json)
# --------------------------------------------------------------------------
# NOTE: required keys here must stay in sync with what src/common/checkpoint.py
# actually writes (optimiser, config_hash, split_hash, epoch,
# generalisation_gap, r2, rmse) — do not add an "arch" requirement here
# unless checkpoint.py's save_checkpoint() is also updated to write it.

REQUIRED_CHECKPOINT_META_KEYS = {
    "optimiser", "config_hash", "split_hash", "epoch",
    "generalisation_gap", "r2", "rmse",
}
VALID_OPTIMISERS = {"adam", "sam"}


def validate_checkpoint_meta(obj: dict) -> list[str]:
    errors: list[str] = []
    missing = REQUIRED_CHECKPOINT_META_KEYS - obj.keys()
    if missing:
        errors.append(f"missing keys: {sorted(missing)}")
    if obj.get("optimiser") not in VALID_OPTIMISERS:
        errors.append(f"optimiser must be one of {VALID_OPTIMISERS}, got {obj.get('optimiser')!r}")
    return errors


def load_checkpoint_meta(path: str | Path) -> dict:
    path = Path(path)
    with open(path, "r", encoding="utf-8") as f:
        obj = json.load(f)
    errors = validate_checkpoint_meta(obj)
    if errors:
        raise ValueError(f"{path} failed checkpoint-meta validation:\n  " + "\n  ".join(errors))
    return obj


def save_checkpoint_meta(obj: dict, path: str | Path) -> None:
    errors = validate_checkpoint_meta(obj)
    if errors:
        raise ValueError(f"refusing to save invalid checkpoint meta to {path}:\n  " + "\n  ".join(errors))
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)


# --------------------------------------------------------------------------
# Data artifacts (splits.json, feature_map.json, response_classes.json)
# --------------------------------------------------------------------------

def validate_splits(obj: dict, all_row_ids: set | None = None) -> list[str]:
    """Validate a splits.json dict. If all_row_ids is provided (the full
    set of row ids present in the processed dataset), also checks that
    every id referenced in the splits actually exists in the dataset."""
    errors = []
    for key in ("train", "val", "test"):
        if key not in obj:
            errors.append(f"splits.json missing '{key}'")
        elif not isinstance(obj[key], list):
            errors.append(f"splits.json['{key}'] must be a list of row ids")
    if not errors:
        train, val, test = set(obj["train"]), set(obj["val"]), set(obj["test"])
        if train & val or train & test or val & test:
            errors.append("splits.json: train/val/test overlap — this must never happen")
        if all_row_ids is not None:
            all_ids = train | val | test
            unknown = all_ids - all_row_ids
            if unknown:
                errors.append(f"splits.json references {len(unknown)} row_ids not present in the dataset")
    return errors


def validate_feature_map(obj: dict) -> list[str]:
    """Structural check: every entry in the feature_map has the required fields."""
    errors = []
    for feature, info in obj.items():
        if "gene_symbol" not in info:
            errors.append(f"feature_map['{feature}'] missing 'gene_symbol'")
        if "partition_edges" not in info:
            errors.append(f"feature_map['{feature}'] missing 'partition_edges'")
    return errors


def check_feature_map_covers(feature_map: dict, feature_cols) -> list[str]:
    """Completeness check: every column actually present in the processed
    dataset (feature_cols) has a corresponding feature_map entry. Distinct
    from validate_feature_map() above, which checks the *shape* of entries
    already in the map; this checks *coverage* against real data columns."""
    errors = []
    for col in feature_cols:
        if col not in feature_map:
            errors.append(f"feature_map.json missing entry for column: {col}")
        elif "gene_symbol" not in feature_map[col]:
            errors.append(f"feature_map[{col!r}] missing 'gene_symbol'")
    return errors


def validate_response_classes(obj: dict) -> list[str]:
    errors = []
    if "threshold" not in obj:
        errors.append("response_classes.json missing 'threshold'")
    if "per_row_class" not in obj:
        errors.append("response_classes.json missing 'per_row_class'")
    return errors


def validate_predictions(df) -> list[str]:
    """Validate a predictions dataframe (artifacts/predictions/*.parquet,
    already loaded by the caller — this module never imports pandas
    itself, per the dependency-light design note above)."""
    errors = []
    for col in ("row_id", "ln_ic50_pred", "class_pred"):
        if col not in df.columns:
            errors.append(f"predictions file missing column: {col}")
    if "class_pred" in df.columns:
        valid = {"sensitive", "resistant"}
        bad = set(df["class_pred"].unique()) - valid
        if bad:
            errors.append(f"predictions contain class_pred values outside {valid}: {bad}")
    return errors


def _generic_load(path: str | Path, validator) -> dict:
    path = Path(path)
    with open(path, "r", encoding="utf-8") as f:
        obj = json.load(f)
    errors = validator(obj)
    if errors:
        raise ValueError(f"{path} failed validation:\n  " + "\n  ".join(errors))
    return obj


def load_splits(path: str | Path, all_row_ids: set | None = None) -> dict:
    path = Path(path)
    with open(path, "r", encoding="utf-8") as f:
        obj = json.load(f)
    errors = validate_splits(obj, all_row_ids=all_row_ids)
    if errors:
        raise ValueError(f"{path} failed validation:\n  " + "\n  ".join(errors))
    return obj


def load_feature_map(path: str | Path) -> dict:
    return _generic_load(path, validate_feature_map)


def load_response_classes(path: str | Path) -> dict:
    return _generic_load(path, validate_response_classes)