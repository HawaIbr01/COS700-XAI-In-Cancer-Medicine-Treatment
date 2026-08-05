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

KNOWN GAP: there is no shared `predict_with_rule()` here. The
row-covered majority-vote prediction logic currently lives inline in
nsga.py's `_score_genotype()`. If GA needs identical scoring logic for
a fair GA-vs-NSGA-II comparison (§4.6), that inline logic needs
extracting into a shared function here — flagged, not yet done.

Chromosome = vector of feature-partition indices, one gene per
discretised feature: -1 = feature excluded from the rule, 0/1/2 =
Low/Intermediate/High.
"""

import random

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