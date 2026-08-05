"""
Shared rule-quality metrics.

Contract (docs/CONTRACTS.md §0.3):
    These three functions are THE definition of fidelity,
    comprehensibility, and stability for this project. GA, NSGA-II,
    and Evaluation all call these — nobody re-implements their own
    version. If a metric definition needs to change, change it here
    and announce it; do not fork it locally in an extraction arm.

    stability() in particular is owned conceptually by the
    Evaluation arm (Hawa) — the implementation below is the
    confirmed, non-stub version (variance of per-subgroup fidelity).

    evaluate_rule_mask() / rule_class_preds() are dataframe-masking
    convenience helpers (originally from feat/nsga2-rule-extraction):
    given a features dataframe and a rule's conditions, they derive
    the sliced (rule_preds, model_preds) sequences that fidelity_f1()
    expects, so GA/NSGA-II fitness functions and Evaluation's own
    per-subgroup stability slicing don't each reimplement row-masking.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score


def evaluate_rule_mask(df: pd.DataFrame, conditions: list[dict]) -> np.ndarray:
    """Boolean mask of rows in `df` satisfied by every condition in a rule.

    conditions: [{"feature": "EGFR_expr", "partition": "High"}, ...]
    per the rule-pool schema (docs/CONTRACTS.md §0.4). Assumes discretised
    feature columns store the partition label directly as a string.
    """
    mask = np.ones(len(df), dtype=bool)
    for cond in conditions:
        mask &= (df[cond["feature"]].values == cond["partition"])
    return mask


def rule_class_preds(df: pd.DataFrame, conditions: list[dict], predicted_class: str,
                      class_col: str = "class_pred"):
    """Returns (rule_preds, model_preds, mask) restricted to the rows a
    rule covers. rule_preds is the rule's single predicted_class repeated
    for every covered row; model_preds is the model's actual class_pred
    on those same rows. Feed rule_preds/model_preds straight into
    fidelity_f1() below."""
    mask = evaluate_rule_mask(df, conditions)
    model_preds = df.loc[mask, class_col]
    rule_preds = pd.Series(predicted_class, index=model_preds.index)
    return rule_preds, model_preds, mask


def fidelity_f1(rule_class_preds: Sequence[str], model_class_preds: Sequence[str]) -> float:
    """
    Fidelity of a rule (or rule set) to the black-box model, measured as
    macro F1-score between the rule's predicted classes and the model's
    predicted classes on the same rows.

    Both sequences must already be mapped to the discrete response
    classes defined in artifacts/data/response_classes.json
    (see docs/CONTRACTS.md §0.4 and methodology §4.2.2 / §4.6.1) —
    this function does not do that mapping itself. Use rule_class_preds()
    above to derive these sequences from a features dataframe.

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

    Note the direction: this returns a raw count (lower = better), not a
    normalised (0,1] score. If a fitness function needs a "higher is
    better" transform for scalarisation, do that transform locally in
    the extraction arm's own fitness code — don't change what this
    function returns, since PROMETHEE II's preference direction for this
    criterion is set independently against the raw length.

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

    Build the per-subgroup dicts by calling rule_class_preds() once per
    subgroup (e.g. grouping a features dataframe by cancer_type first).

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