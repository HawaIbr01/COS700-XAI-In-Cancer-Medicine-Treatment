"""
Shared chromosome encoding/decoding for rule extraction.

OWNERS: Melisa + Sakhile, jointly (feat/ga-rule-extraction,
feat/nsga2-rule-extraction)

CONTRACT (docs/CONTRACTS.md Arm 3 & Arm 4):
    GA and NSGA-II must use the SAME chromosome representation and
    the SAME decode-to-rule logic, so that a rule produced by either
    extractor is structurally comparable and both emit the identical
    §0.4 rule-pool schema. This file is the one place that encoding
    lives — ga.py and nsga.py both import it rather than each
    re-implementing their own.

    If either of you needs to change the encoding, coordinate first —
    a divergence here quietly breaks the GA-vs-NSGA-II comparison
    that the whole evaluation stage depends on.

TODO (Melisa + Sakhile, together):
    - Chromosome representation: vector of length = n_features, each
      element is either a partition index (0=Low, 1=Intermediate,
      2=High) or -1 meaning "feature excluded from this rule" (§4.3.3).
    - decode_chromosome(): turn a chromosome vector into the
      'conditions' list + 'genes' list used in the §0.4 rule schema
      (genes resolved via artifacts/data/feature_map.json).
    - predict_with_rule(): given a rule's conditions and a dataframe
      of discretised features, return the predicted class per
      covered row (used by src.common.metrics.fidelity_f1).
    - Mutation / crossover operators, if shared (two-point crossover
      is specified identically for both in §4.3.3 / §4.4).
"""

from __future__ import annotations

from dataclasses import dataclass

PARTITION_LOW = 0
PARTITION_INTERMEDIATE = 1
PARTITION_HIGH = 2
PARTITION_EXCLUDED = -1

PARTITION_NAMES = {
    PARTITION_LOW: "Low",
    PARTITION_INTERMEDIATE: "Intermediate",
    PARTITION_HIGH: "High",
}


@dataclass
class Chromosome:
    """A candidate rule, encoded as one partition choice per feature
    (or PARTITION_EXCLUDED if the feature isn't part of the rule)."""
    values: list[int]  # length == n_features
    feature_names: list[str]  # same length, e.g. ["EGFR_expr", "TP53_expr", ...]


def decode_chromosome(chromosome: Chromosome, feature_map: dict) -> dict:
    """
    Decode a chromosome into the 'conditions' + 'genes' fields of the
    §0.4 rule schema.

    Args:
        chromosome: the encoded candidate rule.
        feature_map: artifacts/data/feature_map.json, i.e.
            {feature_col: {"gene_symbol": str, "partition_edges": [...]}}.

    Returns:
        {"conditions": [{"feature": ..., "partition": ...}, ...],
         "genes": [...]}
    """
    conditions = []
    genes = []
    for value, feature in zip(chromosome.values, chromosome.feature_names):
        if value == PARTITION_EXCLUDED:
            continue
        conditions.append({"feature": feature, "partition": PARTITION_NAMES[value]})
        gene = feature_map.get(feature, {}).get("gene_symbol")
        if gene:
            genes.append(gene)

    if not conditions:
        raise ValueError("decoded chromosome has zero active conditions — "
                          "this should not be selected/kept as a candidate rule")

    return {"conditions": conditions, "genes": sorted(set(genes))}


def predict_with_rule(conditions: list[dict], discretised_df) -> list[str | None]:
    """
    Apply a rule's conditions to a dataframe of discretised features and
    return the predicted class for each row the rule covers, or None
    for rows it does not cover.

    Args:
        conditions: the rule's 'conditions' list (feature + partition pairs).
        discretised_df: a dataframe with one column per feature,
            containing values in {"Low", "Intermediate", "High"}.

    Returns:
        list of predicted classes (or None), one per row in discretised_df,
        for use with src.common.metrics.fidelity_f1 after filtering out
        the None (uncovered) rows.

    TODO (Melisa + Sakhile): implement row-wise condition matching
    (a row is "covered" if all conditions hold), and decide how
    'predicted_class' is assigned to a covered row — this should match
    whatever majority-vote / lookup logic populates 'predicted_class'
    in the rule-pool schema.
    """
    raise NotImplementedError("predict_with_rule: see TODO above")
