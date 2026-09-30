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

RESOLVED (was KNOWN GAP): `predict_with_rule()` below is that extraction.
nsga.py's `_score_genotype()` now calls it instead of computing the
majority vote inline — same two lines, moved here so GA (§4.6) uses the
identical logic rather than a second implementation. It takes a pandas
Series (works with anything `pd.Series()` accepts, e.g. a plain list) so
tie-breaking matches pandas' `.mode()` exactly, rather than reimplementing
mode with stdlib and risking a different tie-break on an even split.

Chromosome = vector of feature-partition indices, one gene per
discretised feature: -1 = feature excluded from the rule, 0/1/2 =
Low/Intermediate/High.
"""

import random

import pandas as pd

from src.common.metrics import fidelity_f1

PARTITION_CODES = [0, 1, 2]
PARTITION_LABELS = {0: "Low", 1: "Intermediate", 2: "High"}
N_PARTITIONS = len(PARTITION_CODES)


def random_gene() -> int:
    return random.randint(-1, N_PARTITIONS - 1)


def sparse_individual(n_features: int, max_active: int = 5) -> list:
    """Build one whole chromosome with a small number (1..max_active) of
    active genes, everything else excluded (-1) -- C10 (sparse init).

    Uniform per-gene sampling via random_gene() can't produce this: at real
    feature counts (~40+), calling it independently per position makes a
    dense, near-unreadable chromosome overwhelmingly likely by construction,
    and sparsity is a whole-chromosome property random_gene() has no way to
    see (it doesn't know how many genes are already active, or how many
    positions remain). Ported from
    notebooks/02_synthetic_rule_pools.ipynb's sample_sparse_chromosome(),
    which the synthetic fixtures already relied on for this same reason.

    Register with DEAP via tools.initIterate (calls this once per
    individual), not tools.initRepeat (calls a per-gene function n_features
    times) -- the two are not interchangeable for this.
    """
    n_active = random.randint(1, min(max_active, n_features))
    active_positions = random.sample(range(n_features), n_active)
    individual = [-1] * n_features
    for pos in active_positions:
        individual[pos] = random.randint(0, N_PARTITIONS - 1)
    return individual


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


def predict_with_rule(covered_preds) -> tuple[str | None, float]:
    """Majority-vote prediction + fidelity for the rows a rule covers.

    covered_preds: the model's class_pred values for the rows a rule's
    conditions match (e.g. df.loc[mask, "class_pred"]) — anything
    pd.Series() accepts, so a plain list works too.

    Returns (predicted_class, fidelity_f1). If covered_preds is empty,
    returns (None, 0.0) rather than raising — callers that already guard
    zero-coverage upstream (nsga.py's _score_genotype does, for its own
    length==0 reasons) just never hit this branch; callers that don't
    guard it (a future GA caller, say) get a safe default instead of an
    IndexError out of pandas' .mode().
    """
    covered_preds = pd.Series(covered_preds)
    if len(covered_preds) == 0:
        return None, 0.0

    predicted_class = covered_preds.mode().iat[0]
    rule_preds = pd.Series(predicted_class, index=covered_preds.index)
    fidelity = fidelity_f1(rule_preds, covered_preds)
    return predicted_class, float(fidelity)


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