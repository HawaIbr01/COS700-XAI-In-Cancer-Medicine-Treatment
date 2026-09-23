"""
Shared chromosome encoding/decoding and genetic operators for rule
extraction.

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

STATUS: this is Sakhile's real, working implementation (previously a
TODO scaffold — superseded now that his NSGA-II arm has landed and
actually calls these functions). Melisa: build your GA arm against
THESE function names, not the earlier scaffold's `decode_chromosome` —
that name was never implemented.

CLOSED GAP (Melisa, feat/ga-rule-extraction): the module previously
noted there was no shared `predict_with_rule()`, with the row-covered
majority-vote scoring logic living inline in nsga.py's
`_score_genotype()`. Added `rule_mask()` and `predict_with_rule()` below
as an additive, non-breaking change — nsga.py's own inline copy is left
untouched (coordinate with Sakhile before swapping it over, per the
note below), but ga.py now imports these rather than re-deriving its
own masking logic a third time.

Chromosome = vector of feature-partition indices, one gene per
discretised feature: -1 = feature excluded from the rule, 0/1/2 =
Low/Intermediate/High.
"""

import random

import numpy as np

PARTITION_CODES = [0, 1, 2]
PARTITION_LABELS = {0: "Low", 1: "Intermediate", 2: "High"}
N_PARTITIONS = len(PARTITION_CODES)


def random_gene() -> int:
    return random.randint(-1, N_PARTITIONS - 1)


def decode_individual(individual, feature_cols):
    """Chromosome -> list of {"feature", "partition"} conditions (Branch
    Contracts §0.4 rule-pool schema), skipping excluded (-1) genes."""
    return [
        {"feature": feature_cols[i], "partition": PARTITION_LABELS[val]}
        for i, val in enumerate(individual)
        if val != -1
    ]


def resolve_genes(conditions, feature_map):
    return sorted({feature_map[c["feature"]]["gene_symbol"] for c in conditions})


def cxTwoPoint_partitions(ind1, ind2):
    """Two-point crossover over the partition-index vector. Thin named
    wrapper around DEAP's built-in so both extractors call one shared
    function name even though the operator itself is a DEAP primitive."""
    from deap import tools
    return tools.cxTwoPoint(ind1, ind2)


def mutate_partition(individual, indpb=0.05):
    """Per-gene resample: with probability indpb, replace a gene with a
    fresh random value in {-1, 0, 1, 2}."""
    for i in range(len(individual)):
        if random.random() < indpb:
            individual[i] = random_gene()
    return (individual,)


def rule_mask(df, feature_cols, individual):
    """
    Boolean mask of rows in `df` satisfied by every active (non -1) gene
    in `individual`. Mirrors nsga.py's inline helper of the same name —
    this is the shared version; see CLOSED GAP note above.
    """
    mask = np.ones(len(df), dtype=bool)
    for i, val in enumerate(individual):
        if val != -1:
            mask &= (df[feature_cols[i]].values == PARTITION_LABELS[val])
    return mask


def predict_with_rule(df, feature_cols, individual, class_pred_col, min_coverage=0.0):
    """
    Scores one genotype against a features dataframe: coverage, rule
    length, the rule's own predicted class (majority vote of the target
    column among covered rows), and that class as a per-row Series ready
    for src.common.metrics.fidelity_f1 / accuracy scoring.

    Mirrors nsga.py's _score_genotype()'s masking/majority-vote logic so
    GA and NSGA-II score genotypes identically (the fairness requirement
    CONTRACTS.md Arm 3/4 flags). Deliberately does NOT compute fidelity
    itself (that needs a *model's* class_pred specifically) so this one
    function can be reused for both the fidelity target (train
    class_pred) and the accuracy target (a true-label column), just by
    passing a different class_pred_col.

    Returns:
        (predicted_class | None, length, coverage, covered_preds | None)
        predicted_class is None when the rule is empty or covers nothing
        (mirrors nsga.py's convention for filtering dead genotypes).
    """
    length = sum(1 for v in individual if v != -1)
    mask = rule_mask(df, feature_cols, individual)
    coverage = float(mask.mean()) if len(df) else 0.0

    if length == 0 or mask.sum() == 0:
        return None, length, coverage, None
    if min_coverage > 0.0 and coverage < min_coverage:
        return None, length, coverage, None

    covered_preds = df.loc[mask, class_pred_col]
    predicted_class = covered_preds.mode().iat[0]
    return predicted_class, length, coverage, covered_preds