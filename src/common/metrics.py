"""
src/common/metrics.py — Branch Contracts §0.3.

fidelity_f1, comprehensibility, and stability are the contract for rule
quality: GA, NSGA-II, and Evaluation all call the same functions. Only
fidelity_f1 is fully specified right now — it's the one the extractors'
fitness functions depend on directly. comprehensibility is a placeholder.
stability is owned by feat/evaluation-pipeline (Hawa, §4.6) and is not
needed for extraction — it raises NotImplementedError so a wiring mistake
fails loudly instead of silently returning a meaningless number.
"""

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

VALID_PARTITIONS = ("Low", "Intermediate", "High")


def evaluate_rule_mask(df: pd.DataFrame, conditions: list[dict]) -> np.ndarray:
    """Boolean mask of rows in `df` satisfied by every condition in a rule.

    conditions: [{"feature": "EGFR_expr", "partition": "High"}, ...]
    per the rule-pool schema (Branch Contracts §0.4). Assumes discretised
    feature columns store the partition label directly as a string.
    """
    mask = np.ones(len(df), dtype=bool)
    for cond in conditions:
        mask &= (df[cond["feature"]].values == cond["partition"])
    return mask


def rule_class_preds(df: pd.DataFrame, conditions: list[dict], predicted_class: str,
                      class_col: str = "class_pred"):
    """Returns (rule_preds, model_preds, mask) restricted to the rows the
    rule covers. rule_preds is the rule's single predicted_class repeated
    for every covered row; model_preds is the model's actual class_pred
    on those same rows."""
    mask = evaluate_rule_mask(df, conditions)
    model_preds = df.loc[mask, class_col]
    rule_preds = pd.Series(predicted_class, index=model_preds.index)
    return rule_preds, model_preds, mask


def fidelity_f1(rule_preds: pd.Series, model_preds: pd.Series) -> float:
    """Macro-F1 between the rule's predicted class and the model's
    class_pred, on the rows the rule covers.

    GA, NSGA-II, and Evaluation MUST call this exact function — do not
    reimplement locally (Branch Contracts §0.3).
    """
    if len(model_preds) == 0:
        return 0.0
    return float(f1_score(model_preds, rule_preds, average="macro", zero_division=0))


def comprehensibility(rule: dict) -> float:
    """Simple inverse-length comprehensibility score in (0, 1]. Shorter
    rules score higher. Placeholder pending Evaluation's PROMETHEE
    weighting scheme (§4.6) — Hawa may replace this definition."""
    length = rule.get("length", len(rule.get("conditions", [])))
    return 1.0 / (1.0 + length)


def stability(rule: dict, subgroups) -> float:
    """Variance of a rule's fidelity across GDSC cancer subtypes.

    Owned by feat/evaluation-pipeline (§4.6) — Hawa defines this precisely
    and publishes it back here per the branch contract's hand-off notes.
    Not needed for extraction; raises to fail loudly rather than silently
    returning a meaningless placeholder number.
    """
    raise NotImplementedError(
        "stability() is owned by feat/evaluation-pipeline (Hawa) — "
        "not required for rule extraction, only for PROMETHEE ranking."
    )
