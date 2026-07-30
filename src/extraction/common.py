"""
src/extraction/common.py — shared rule-extraction encoding and operators.

PROTOTYPE STATUS: written solo for the NSGA-II arm's simulation phase, per
the branch contract's explicit allowance ("coordinate with Arm 3 on
common.py if you want the encoding to match, but you can prototype your
own"). Once Melisa's GA arm lands, diff this against hers and reconcile to
one shared file — the GA-vs-NSGA comparison in Evaluation (§4.6) depends on
both extractors using identical operators. If they end up staying
different, that's a documented deviation, not silent drift.

Chromosome = vector of feature-partition indices, one gene per discretised
feature: -1 = feature excluded from the rule, 0/1/2 = Low/Intermediate/High.
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
