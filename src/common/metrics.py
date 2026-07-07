"""
Shared rule-quality metrics.

Contract (docs/CONTRACTS.md §0.3):
    These three functions are THE definition of fidelity,
    comprehensibility, and stability for this project. GA, NSGA-II,
    and Evaluation all call these — nobody re-implements their own
    version. If a metric definition needs to change, change it here
    and announce it; do not fork it locally in an extraction arm.

    stability() in particular is owned conceptually by the
    Evaluation arm (Hawa) — the signature below is fixed so
    extraction arms can call it too, but the exact aggregation used
    should be confirmed with her before Stage 2.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np
from sklearn.metrics import f1_score


def fidelity_f1(rule_class_preds: Sequence[str], model_class_preds: Sequence[str]) -> float:
    """
    Fidelity of a rule (or rule set) to the black-box model, measured as
    macro F1-score between the rule's predicted classes and the model's
    predicted classes on the same rows.

    Both sequences must already be mapped to the discrete response
    classes defined in artifacts/data/response_classes.json
    (see docs/CONTRACTS.md §0.4 and methodology §4.2.2 / §4.6.1) —
    this function does not do that mapping itself.

    Args:
        rule_class_preds: predicted class per row, from the rule.
        model_class_preds: predicted class per row, from DeepCCDS
            (baseline or SAM — whichever condition is being evaluated).

    Returns:
        macro F1-score in [0, 1].
    """
    if len(rule_class_preds) != len(model_class_preds):
        raise ValueError(
            f"length mismatch: rule_class_preds has {len(rule_class_preds)} rows, "
            f"model_class_preds has {len(model_class_preds)}"
        )
    if len(rule_class_preds) == 0:
        raise ValueError("cannot compute fidelity on an empty rule coverage set")

    return float(f1_score(model_class_preds, rule_class_preds, average="macro", zero_division=0))


def comprehensibility(rule: dict) -> int:
    """
    Comprehensibility proxy for a single rule: the number of antecedent
    conditions ("Rule Explosion", §3.6.1). Lower is more comprehensible.

    Args:
        rule: a rule dict conforming to the §0.4 rule-pool schema
            (must contain a 'conditions' list, or a precomputed 'length').

    Returns:
        the rule length (number of conditions).
    """
    if "length" in rule:
        return int(rule["length"])
    return len(rule["conditions"])


def stability(rule_class_preds_by_subgroup: dict[str, Sequence[str]],
              model_class_preds_by_subgroup: dict[str, Sequence[str]]) -> float:
    """
    Stability of a rule across GDSC cancer subtypes / tissue types
    (§3.6.1, §4.6.1): the variance of per-subgroup fidelity. Lower
    variance = more stable = the rule isn't just exploiting one subtype.

    NOTE: this is a provisional implementation (variance of per-subgroup
    F1). The exact aggregation is owned by the Evaluation arm — confirm
    with Hawa before relying on absolute values for the final report,
    per docs/CONTRACTS.md §5 "Likely to change".

    Args:
        rule_class_preds_by_subgroup: {subgroup_name: [rule predictions]}
        model_class_preds_by_subgroup: {subgroup_name: [model predictions]},
            same keys/row-order as above.

    Returns:
        variance (float, >= 0) of per-subgroup fidelity_f1. 0 = perfectly
        stable across subgroups.
    """
    if rule_class_preds_by_subgroup.keys() != model_class_preds_by_subgroup.keys():
        raise ValueError("subgroup keys must match between rule and model predictions")

    per_subgroup_fidelity = []
    for subgroup, rule_preds in rule_class_preds_by_subgroup.items():
        model_preds = model_class_preds_by_subgroup[subgroup]
        if len(rule_preds) == 0:
            continue  # no coverage in this subgroup — skip rather than divide by zero
        per_subgroup_fidelity.append(fidelity_f1(rule_preds, model_preds))

    if len(per_subgroup_fidelity) < 2:
        raise ValueError(
            "stability requires at least 2 subgroups with non-zero coverage; "
            f"got {len(per_subgroup_fidelity)}"
        )

    return float(np.var(per_subgroup_fidelity))


if __name__ == "__main__":
    # smoke test
    r = ["sensitive", "sensitive", "resistant", "resistant"]
    m = ["sensitive", "resistant", "resistant", "resistant"]
    print("fidelity_f1:", fidelity_f1(r, m))
    print("comprehensibility:", comprehensibility({"conditions": [1, 2, 3]}))
