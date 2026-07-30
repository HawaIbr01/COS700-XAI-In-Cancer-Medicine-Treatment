"""
src/common/schema.py — Branch Contracts §0.4 and §6.

Validators raise AssertionError with a specific message on the first
violation found — that's the contract check every arm's output must pass
before a downstream consumer touches it.
"""

REQUIRED_META_FIELDS = ("extractor", "condition", "runs", "config_hash")
REQUIRED_RULE_FIELDS = {
    "id", "conditions", "genes", "predicted_class",
    "length", "fidelity_f1", "coverage", "run_ids", "non_dominated",
}
VALID_PARTITIONS = ("Low", "Intermediate", "High")


def validate_rule_pool(payload: dict, feature_map: dict | None = None) -> None:
    """Validates an artifacts/rules/*.json payload against §0.4.

    Pass feature_map (the loaded feature_map.json dict) to additionally
    check that every rule's genes resolve — required before Evaluation's
    GSEA step can run.
    """
    assert "meta" in payload, "rule pool missing 'meta'"
    assert "rules" in payload, "rule pool missing 'rules'"

    meta = payload["meta"]
    for key in REQUIRED_META_FIELDS:
        assert key in meta, f"meta missing required field: {key}"
    assert meta["extractor"] in ("ga", "nsga"), f"unknown extractor: {meta['extractor']}"
    assert meta["condition"] in ("baseline", "sam"), f"unknown condition: {meta['condition']}"

    all_genes = set()
    if feature_map is not None:
        all_genes = {v["gene_symbol"] for v in feature_map.values()}

    seen_ids = set()
    for rule in payload["rules"]:
        rid = rule.get("id", "<no id>")
        missing = REQUIRED_RULE_FIELDS - rule.keys()
        assert not missing, f"rule {rid}: missing fields {missing}"
        assert rid not in seen_ids, f"duplicate rule id: {rid}"
        seen_ids.add(rid)

        assert rule["length"] == len(rule["conditions"]), (
            f"rule {rid}: length {rule['length']} != len(conditions) {len(rule['conditions'])}"
        )
        assert 0.0 <= rule["fidelity_f1"] <= 1.0, f"rule {rid}: fidelity_f1 out of [0,1]"
        assert 0.0 <= rule["coverage"] <= 1.0, f"rule {rid}: coverage out of [0,1]"

        for cond in rule["conditions"]:
            assert "feature" in cond and "partition" in cond, (
                f"rule {rid}: malformed condition {cond}"
            )
            assert cond["partition"] in VALID_PARTITIONS, (
                f"rule {rid}: invalid partition {cond['partition']}"
            )
            if feature_map is not None:
                assert cond["feature"] in feature_map, (
                    f"rule {rid}: feature {cond['feature']} not in feature_map"
                )

        if feature_map is not None:
            unresolved = set(rule["genes"]) - all_genes
            assert not unresolved, f"rule {rid}: genes not resolvable in feature_map: {unresolved}"


def validate_splits(splits: dict, all_row_ids: set) -> None:
    for name in ("train", "val", "test"):
        assert name in splits, f"splits.json missing '{name}'"
    ids = [i for s in splits.values() for i in s]
    assert len(ids) == len(set(ids)), "splits.json has duplicate row_ids across splits"
    assert set(ids) <= all_row_ids, "splits.json references row_ids not present in the dataset"


def validate_feature_map(feature_map: dict, feature_cols) -> None:
    for col in feature_cols:
        assert col in feature_map, f"feature_map.json missing entry for {col}"
        assert "gene_symbol" in feature_map[col], f"feature_map[{col}] missing gene_symbol"


def validate_predictions(df) -> None:
    for col in ("row_id", "ln_ic50_pred", "class_pred"):
        assert col in df.columns, f"predictions file missing column: {col}"
    assert df["class_pred"].isin(["sensitive", "resistant"]).all(), (
        "predictions contain a class_pred outside {sensitive, resistant}"
    )


def validate_checkpoint_meta(meta: dict) -> None:
    for key in ("optimiser", "arch", "epoch", "config_hash", "split_hash"):
        assert key in meta, f"checkpoint meta.json missing field: {key}"
