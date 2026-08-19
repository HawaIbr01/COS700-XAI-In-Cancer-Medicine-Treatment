"""
ga_rule_extraction.py
======================
Genetic Algorithm (GA) rule extraction for the Red XAI diagnostic pipeline
described in Section 4.3 of the COS700 proposal
("Diagnosing and Repairing XAI Fragility: SAM-Optimised Rules for Cancer
Drug Response").

This module extracts symbolic if-then rules from a trained DeepCCDS
checkpoint (Stage 1: Adam baseline, or Stage 2: SAM-regularised model),
using a DEAP-based single-objective GA with a weighted-sum fitness over
fidelity, accuracy, and comprehensibility (Section 4.3.3), tournament
selection, two-point crossover, structural mutation, low-support rule
pruning (Section 4.3.4), and 30 independent runs (Section 4.3.5).

--------------------------------------------------------------------------
DATA CONTRACT ASSUMED BY THE GA (post-alignment)
--------------------------------------------------------------------------
- X_disc: np.ndarray of shape (n_pairs, n_genes), integer-coded per
  Section 4.2.2's discretisation, one row per cell line-drug pair:
      0 = Low Expression      (Z < -1)
      1 = Intermediate        (-1 <= Z <= 1)
      2 = High Expression     (Z > 1)
- y_true: np.ndarray of shape (n_pairs,), the discretised ground-truth
  response class, obtained by thresholding the split's real LN_IC50
  column (Section 4.2.2).
- y_black_box: np.ndarray of shape (n_pairs,), DeepCCDS's own predicted
  class (from thresholding its continuous LN IC50 prediction on the
  pathway/mutation/drug-fingerprint inputs), used as the fidelity target
  -- rules are graded against the MODEL, not ground truth, for fidelity.

Script runs twice -- once against `checkpoints/baseline_adam_real.pt`
(Stage 1) and once against Bahiya's SAM checkpoint (Stage 2) -- to
produce the two rule pools that Section 4.6.1 feeds into the unified
PROMETHEE II matrix.
"""

from __future__ import annotations

import argparse
import json
import random
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import List, Sequence, Tuple

import numpy as np
from sklearn.metrics import f1_score, accuracy_score
from deap import base, creator, tools

try:
    import torch
except ImportError:
    torch = None  # only required for real (non-synthetic) checkpoint/model use

try:
    from src.model.deepccds import DeepCCDS
    from src.data.precomputed_inputs import (
        load_pathway_activity,
        load_mutation_matrix,
        load_drug_fingerprints,
        load_split,
    )
    _REPO_IMPORTS_OK = True
except ImportError:
    DeepCCDS = None
    load_pathway_activity = load_mutation_matrix = load_drug_fingerprints = load_split = None
    _REPO_IMPORTS_OK = False

# --------------------------------------------------------------------------
# Constants matching Section 4.2.2 / 4.3.2 discretisation scheme
# --------------------------------------------------------------------------
DONT_CARE = -1          # padding value: attribute excluded from the rule
N_CATEGORIES = 3        # {Low, Intermediate, High}

# --------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------
@dataclass
class GAConfig:
    """Hyperparameters matching Section 4.3.5 / Yedjour et al. baseline."""
    n_generations: int = 50
    pop_min: int = 35
    pop_max: int = 45
    tournament_size: int = 3
    crossover_prob: float = 0.7
    mutation_prob: float = 0.2
    # weighted-sum fitness weights: fidelity, accuracy, comprehensibility
    w_fidelity: float = 0.45
    w_accuracy: float = 0.35
    w_comprehensibility: float = 0.20
    max_rule_length: int = 12          # upper bound on active conditions
    min_support: float = 0.02          # Section 4.3.4 pruning threshold
    n_independent_runs: int = 30       # Section 4.3.5
    random_seed_base: int = 42
    # Section 4.2.2: "the exact threshold [is] fixed at the implementation
    # stage" -- median split is a reasonable placeholder (balances classes)
    # until the team fixes a clinically-motivated LN_IC50 cutoff.
    ln_ic50_threshold: float | None = None


# --------------------------------------------------------------------------
# SEAM 1: load the real Stage 1 / Stage 2 DeepCCDS checkpoint
# --------------------------------------------------------------------------
def load_black_box_model(checkpoint_path: str, device: str = "cpu"):
    """
    Load a trained DeepCCDS checkpoint -- either
    `checkpoints/baseline_adam_real.pt` (Stage 1, Angelique's branch) or
    the equivalent SAM checkpoint (Stage 2, Bahiya's branch), since both
    are trained via `train_baseline_real()`-style loops against the
    identical architecture (BASELINE_README.md Section 8).

    TODO (whoever has src/model/deepccds.py open): confirm DeepCCDS's
    __init__ signature dimensions are fixed
    (38 pathways, 534 driver genes, 1024-bit fingerprints), so DeepCCDS()
    with no args is the likely case, but pass explicit dims below if the
    constructor requires them, e.g. DeepCCDS(pathway_dim=38,
    mutation_dim=534, drug_fp_dim=1024).
    """
    if not _REPO_IMPORTS_OK:
        raise ImportError(
            "Could not import src.model.deepccds.DeepCCDS. Run this script "
            "from the feat/baseline-model repo root (or add it to "
            "PYTHONPATH) so 'from src.model.deepccds import DeepCCDS' "
            "resolves. Not needed for --synthetic mode."
        )
    if torch is None:
        raise ImportError("PyTorch is required to load a real checkpoint (pip install torch).")

    model = DeepCCDS()  # TODO: pass constructor args if deepccds.py requires them
    checkpoint = torch.load(checkpoint_path, map_location=device)
    # train_baseline.py's summary/history JSONs suggest the .pt file may be
    # either a raw state_dict or a dict wrapping one under a known key --
    # handle both without guessing wrong silently.
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
    elif isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    else:
        state_dict = checkpoint
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model


# --------------------------------------------------------------------------
# Real precomputed-feature loading (pathway / mutation / drug fingerprint)
# --------------------------------------------------------------------------
def load_precomputed_split_arrays(
    data_root: str, split: str = "test"
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Loads the DeepCCDS authors' precomputed features (per
    `src/data/precomputed_inputs.py`) and the requested split's row list,
    then gathers per-pair pathway/mutation/drug-fingerprint tensors by
    indexing into the per-cell-line / per-drug matrices.

    Returns
    -------
    pathway_activity : (n_pairs, 38) float
    mutation         : (n_pairs, 534) float/binary
    drug_fingerprint : (n_pairs, 1024) float/binary
    cell_idx         : (n_pairs,) int   -- needed later to join gene-
                        expression bins onto each cell-drug pair row
    drug_idx         : (n_pairs,) int
    ln_ic50_true     : (n_pairs,) float -- ground-truth continuous target
    """
    if not _REPO_IMPORTS_OK:
        raise ImportError(
            "Could not import src.data.precomputed_inputs loaders. Run "
            "from the feat/baseline-model repo root. Not needed for "
            "--synthetic mode."
        )
    pathway_by_cell = np.asarray(load_pathway_activity(data_root))     # (878, 38)
    mutation_by_cell = np.asarray(load_mutation_matrix(data_root))     # (878, 534)
    fp_by_drug = np.asarray(load_drug_fingerprints(data_root))         # (413, 1024)
    split_rows = np.asarray(load_split(data_root, split))              # (n_pairs, 3)

    cell_idx = split_rows[:, 0].astype(int)
    drug_idx = split_rows[:, 1].astype(int)
    ln_ic50_true = split_rows[:, 2].astype(float)

    pathway_activity = pathway_by_cell[cell_idx]
    mutation = mutation_by_cell[cell_idx]
    drug_fingerprint = fp_by_drug[drug_idx]
    return pathway_activity, mutation, drug_fingerprint, cell_idx, drug_idx, ln_ic50_true


# --------------------------------------------------------------------------
# SEAM 2: discretised gene-expression bins for rule antecedents
# --------------------------------------------------------------------------
def load_gene_expression_bins(
    data_root: str = None,
    synthetic: bool = False,
    n_cell_lines: int = 878,
    n_genes: int = 200,
    seed: int = 0,
) -> Tuple[np.ndarray, List[str]]:
    """
    Load (or synthesise) the discretised Low/Intermediate/High gene-
    expression bins that GA rule antecedents are built from
    (Section 4.2.2).

    STATUS (per BASELINE_README.md Sections 4 & 8, as of 2 Aug 2026):
    BLOCKED. These bins come from `src/data/preprocessing.py`'s raw-GDSC
    path, which itself needs the RNA-seq expression file from Cell Model
    Passports -- not yet downloaded/available. That pipeline outputs
    (n_cell_lines, n_genes) bins keyed by cell line, matching this
    function's return contract.

    Until that file lands, set synthetic=True to develop and test the GA
    against realistically-shaped placeholder data (878 cell lines, per
    the precomputed pathway/mutation files -- same cell-line index space,
    so synthetic bins line up with real model predictions if you run in
    a mixed real-model / synthetic-bins configuration).

    Once real bins exist, replace the body below with, e.g.:

        from src.data.preprocessing import build_expression_bins
        gene_bins_by_cell_line, gene_names = build_expression_bins(data_root)
        return gene_bins_by_cell_line, gene_names

    Returns
    -------
    gene_bins_by_cell_line : np.ndarray (n_cell_lines, n_genes) int in {0,1,2}
    gene_names : list[str]
    """
    if synthetic:
        rng = np.random.default_rng(seed)
        gene_bins_by_cell_line = rng.integers(0, N_CATEGORIES, size=(n_cell_lines, n_genes))
        gene_names = [f"SYNTH_GENE_{i}" for i in range(n_genes)]
        return gene_bins_by_cell_line, gene_names

    raise NotImplementedError(
        "Discretised gene-expression bins are blocked pending the RNA-seq "
        "expression file (BASELINE_README.md Sections 4 and 8) -- "
        "src/data/preprocessing.py cannot complete without it. Pass "
        "synthetic=True to develop/test against placeholder data in the "
        "meantime, or check with Angelique/Melisa/Sakhile once the file "
        "lands and wire this function to preprocessing.py's real output."
    )


def align_rule_frame_to_predictions(
    gene_bins_by_cell_line: np.ndarray, cell_idx: np.ndarray
) -> np.ndarray:
    """
    Broadcasts each cell line's gene-expression bins across every
    cell-drug pair it appears in, per the split's cell_idx column, so
    X_disc lines up row-for-row with y_black_box / y_true (Section 4.2.2:
    "each row represents a unique cell line-drug combination"). Rule
    antecedents describe gene-expression state only; drug identity is not
    itself an antecedent feature, matching the proposal's rule format.
    """
    return gene_bins_by_cell_line[cell_idx]


# --------------------------------------------------------------------------
# SEAM 3: black-box prediction extraction (fidelity target)
# --------------------------------------------------------------------------
def get_black_box_predictions(
    model,
    pathway_activity: np.ndarray,
    mutation: np.ndarray,
    drug_fingerprint: np.ndarray,
    threshold: float,
    device: str = "cpu",
    batch_size: int = 4096,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Runs the trained DeepCCDS model over the three input branches in
    batches and thresholds its continuous LN IC50 output into the
    discrete response class used as the fidelity target (Section 4.2.2 /
    4.6.1).

    TODO (whoever has src/model/deepccds.py open): confirm forward()'s
    return signature. BASELINE_README.md's description of the combined
    loss (Eq. 2: prediction MSE + both autoencoders' reconstruction MSE)
    implies forward() returns a tuple of (ln_ic50_pred, mutation_recon,
    drug_recon) or similar -- the unpacking below assumes the prediction
    is either the sole return value or the first element of a tuple.
    Adjust `_unpack_prediction()` if that's wrong.

    Returns
    -------
    y_black_box : np.ndarray (n_pairs,) int, thresholded class predictions
    ln_ic50_pred : np.ndarray (n_pairs,) float, raw continuous predictions
    """
    if torch is None:
        raise ImportError("PyTorch is required to run the DeepCCDS forward pass.")

    model.eval()
    preds = []
    n = pathway_activity.shape[0]
    with torch.no_grad():
        for start in range(0, n, batch_size):
            end = min(start + batch_size, n)
            p = torch.tensor(pathway_activity[start:end], dtype=torch.float32, device=device)
            m = torch.tensor(mutation[start:end], dtype=torch.float32, device=device)
            d = torch.tensor(drug_fingerprint[start:end], dtype=torch.float32, device=device)
            out = model(p, m, d)
            preds.append(_unpack_prediction(out).cpu().numpy().ravel())
    ln_ic50_pred = np.concatenate(preds)
    y_black_box = (ln_ic50_pred < threshold).astype(int)  # lower LN_IC50 = more sensitive
    return y_black_box, ln_ic50_pred


def _unpack_prediction(model_output):
    """See TODO in get_black_box_predictions -- adjust if forward()'s
    tuple order differs from (ln_ic50_pred, mutation_recon, drug_recon)."""
    if isinstance(model_output, (tuple, list)):
        return model_output[0]
    return model_output


# --------------------------------------------------------------------------
# Rule representation and evaluation
# --------------------------------------------------------------------------
def rule_matches(rule: Sequence[int], X_disc: np.ndarray) -> np.ndarray:
    """
    Vectorised boolean mask of which rows of X_disc satisfy `rule`.
    A rule is a vector of length n_features with values in
    {DONT_CARE, 0, 1, 2}. A row matches if every non-DONT_CARE condition
    equals the corresponding discretised feature value (logical AND).
    """
    rule_arr = np.asarray(rule)
    active = rule_arr != DONT_CARE
    if not active.any():
        # An empty rule matches nothing (avoid trivially "faithful" rules)
        return np.zeros(X_disc.shape[0], dtype=bool)
    return np.all(X_disc[:, active] == rule_arr[active], axis=1)


def predict_with_rule(
    rule: Sequence[int], X_disc: np.ndarray, majority_class: int
) -> np.ndarray:
    """
    Predict the majority-vote class for matched rows and `majority_class`
    (the dataset's dominant class) for unmatched rows, so unmatched rows
    don't spuriously inflate or deflate fidelity/accuracy.
    """
    mask = rule_matches(rule, X_disc)
    preds = np.full(X_disc.shape[0], majority_class, dtype=int)
    return preds, mask


def rule_length(rule: Sequence[int]) -> int:
    """Number of active (non-DONT_CARE) antecedent conditions."""
    return int(np.sum(np.asarray(rule) != DONT_CARE))


def rule_support(rule: Sequence[int], X_disc: np.ndarray) -> float:
    """Fraction of the dataset the rule's antecedent covers."""
    return float(rule_matches(rule, X_disc).mean())


def rule_consequent_class(
    rule: Sequence[int], X_disc: np.ndarray, y_black_box: np.ndarray
) -> int:
    """
    The consequent (predicted class) of a rule is the black-box model's
    majority predicted class among the rows the rule's antecedent matches.
    Falls back to the global majority class if the rule matches nothing.
    """
    mask = rule_matches(rule, X_disc)
    if not mask.any():
        vals, counts = np.unique(y_black_box, return_counts=True)
        return int(vals[np.argmax(counts)])
    vals, counts = np.unique(y_black_box[mask], return_counts=True)
    return int(vals[np.argmax(counts)])


# --------------------------------------------------------------------------
# Fitness evaluation (Section 4.3.3 / 4.6.1)
# --------------------------------------------------------------------------
class RuleFitnessEvaluator:
    """
    Wraps the discretised dataset, ground truth, and black-box predictions
    needed to score a candidate rule chromosome on:
      - fidelity           : macro-F1 of rule predictions vs black-box preds
      - accuracy            : macro-F1 of rule predictions vs ground truth
      - comprehensibility   : inverse of active-condition count
    combined into a single scalar weighted-sum fitness for the GA
    (Section 4.3.3), while individually retaining each objective so a
    non-dominated (Pareto) subset can be filtered from the final
    population/hall-of-fame per Section 4.3.5.
    """

    def __init__(
        self,
        X_disc: np.ndarray,
        y_true: np.ndarray,
        y_black_box: np.ndarray,
        config: GAConfig,
    ):
        self.X_disc = X_disc
        self.y_true = y_true
        self.y_black_box = y_black_box
        self.config = config
        self.n_features = X_disc.shape[1]
        vals, counts = np.unique(y_black_box, return_counts=True)
        self.majority_class = int(vals[np.argmax(counts)])

    def evaluate(self, rule: Sequence[int]) -> Tuple[float, float, float, float]:
        """
        Returns (weighted_fitness, fidelity, accuracy, comprehensibility).
        DEAP requires evaluate() to return a tuple matching the
        creator.Fitness weights, so index [0] is used for selection and
        the remaining three are stashed for post-hoc Pareto filtering.
        """
        cfg = self.config
        length = rule_length(rule)

        # Hard constraints: empty rules and over-long rules are penalised
        if length == 0 or length > cfg.max_rule_length:
            return (-1.0, 0.0, 0.0, 0.0)

        mask = rule_matches(rule, self.X_disc)
        support = float(mask.mean())
        if support < cfg.min_support:
            # Rules that barely fire are unreliable and penalised, matching
            # the pruning philosophy of Section 4.3.4
            return (-1.0, 0.0, 0.0, 0.0)

        consequent = rule_consequent_class(rule, self.X_disc, self.y_black_box)
        preds = np.full(self.X_disc.shape[0], self.majority_class, dtype=int)
        preds[mask] = consequent

        fidelity = f1_score(
            self.y_black_box, preds, average="macro", zero_division=0
        )
        accuracy = f1_score(self.y_true, preds, average="macro", zero_division=0)
        comprehensibility = 1.0 - (length / self.n_features)

        weighted = (
            cfg.w_fidelity * fidelity
            + cfg.w_accuracy * accuracy
            + cfg.w_comprehensibility * comprehensibility
        )
        return (weighted, fidelity, accuracy, comprehensibility)


# --------------------------------------------------------------------------
# DEAP GA setup
# --------------------------------------------------------------------------
def _init_individual(icls, n_features: int, active_prob: float = 0.35):
    """
    Randomly initialise a chromosome: each gene is DONT_CARE with
    probability (1 - active_prob), otherwise a random category in
    {0, 1, 2}, matching the sparse "many attributes excluded" encoding of
    Section 4.3.3.
    """
    genes = [
        random.randint(0, N_CATEGORIES - 1) if random.random() < active_prob else DONT_CARE
        for _ in range(n_features)
    ]
    return icls(genes)


def _mutate_rule(individual, indpb: float):
    """
    Structural mutation (Section 4.3.3): with probability indpb per gene,
    either toggle a DONT_CARE gene into an active condition, drop an
    active condition back to DONT_CARE, or flip an active condition to a
    different category.
    """
    for i in range(len(individual)):
        if random.random() < indpb:
            if individual[i] == DONT_CARE:
                individual[i] = random.randint(0, N_CATEGORIES - 1)
            else:
                roll = random.random()
                if roll < 0.4:
                    individual[i] = DONT_CARE
                else:
                    individual[i] = random.randint(0, N_CATEGORIES - 1)
    return (individual,)


def _setup_toolbox(n_features: int, evaluator: RuleFitnessEvaluator, config: GAConfig):
    # Guard against re-registering creator classes across multiple runs
    if not hasattr(creator, "FitnessRule"):
        creator.create("FitnessRule", base.Fitness, weights=(1.0, 0.0, 0.0, 0.0))
    if not hasattr(creator, "RuleIndividual"):
        creator.create("RuleIndividual", list, fitness=creator.FitnessRule)

    toolbox = base.Toolbox()
    toolbox.register(
        "individual", _init_individual, creator.RuleIndividual, n_features
    )
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register("evaluate", evaluator.evaluate)
    toolbox.register("mate", tools.cxTwoPoint)
    toolbox.register("mutate", _mutate_rule, indpb=1.0 / n_features)
    toolbox.register("select", tools.selTournament, tournsize=config.tournament_size)
    return toolbox


def _prune_population(
    population, evaluator: RuleFitnessEvaluator, config: GAConfig
):
    """
    Section 4.3.4: after each generation, remove conditions with low
    confidence/support by re-checking each active gene's marginal
    contribution to support; genes that push support below min_support
    when added are stripped back to DONT_CARE. Operates in place.
    """
    for ind in population:
        length = rule_length(ind)
        if length <= 1:
            continue
        current_support = rule_support(ind, evaluator.X_disc)
        if current_support >= config.min_support:
            continue
        # progressively drop the least-informative active gene until the
        # rule regains adequate support or becomes empty
        active_idx = [i for i, g in enumerate(ind) if g != DONT_CARE]
        random.shuffle(active_idx)
        for idx in active_idx:
            ind[idx] = DONT_CARE
            if rule_support(ind, evaluator.X_disc) >= config.min_support:
                break
        del ind.fitness.values


# --------------------------------------------------------------------------
# Single GA run
# --------------------------------------------------------------------------
def run_single_ga(
    evaluator: RuleFitnessEvaluator,
    config: GAConfig,
    n_features: int,
    seed: int,
) -> List[dict]:
    """
    Executes one full GA run (Section 4.3.5) and returns the non-dominated
    ("Pareto front") subset of rules from the final population + hall of
    fame, each represented as a dict ready for JSON/CSV export.
    """
    random.seed(seed)
    toolbox = _setup_toolbox(n_features, evaluator, config)

    pop_size = random.randint(config.pop_min, config.pop_max)
    population = toolbox.population(n=pop_size)
    hof = tools.HallOfFame(10)

    fitnesses = map(toolbox.evaluate, population)
    for ind, fit in zip(population, fitnesses):
        ind.fitness.values = fit
    hof.update(population)

    for gen in range(config.n_generations):
        offspring = toolbox.select(population, len(population))
        offspring = list(map(toolbox.clone, offspring))

        for c1, c2 in zip(offspring[::2], offspring[1::2]):
            if random.random() < config.crossover_prob:
                toolbox.mate(c1, c2)
                del c1.fitness.values
                del c2.fitness.values

        for mutant in offspring:
            if random.random() < config.mutation_prob:
                toolbox.mutate(mutant)
                del mutant.fitness.values

        _prune_population(offspring, evaluator, config)

        invalid = [ind for ind in offspring if not ind.fitness.valid]
        fitnesses = map(toolbox.evaluate, invalid)
        for ind, fit in zip(invalid, fitnesses):
            ind.fitness.values = fit

        population[:] = offspring
        hof.update(population)

    candidates = list(hof) + population
    return _extract_pareto_front(candidates, evaluator)


def _dominates(a: Tuple[float, float, float], b: Tuple[float, float, float]) -> bool:
    """a dominates b if a is >= b on all three objectives and > on at least one
    (fidelity, accuracy, comprehensibility -- all maximised)."""
    return all(x >= y for x, y in zip(a, b)) and any(x > y for x, y in zip(a, b))


def _extract_pareto_front(
    candidates, evaluator: RuleFitnessEvaluator
) -> List[dict]:
    """
    Filters the final population/hall-of-fame down to the non-dominated
    set across (fidelity, accuracy, comprehensibility), satisfying
    Section 4.3.5's requirement that each run outputs a Pareto front of
    rule sets, then serialises each surviving rule to a dict.
    """
    seen = {}
    for ind in candidates:
        key = tuple(ind)
        if key not in seen:
            seen[key] = ind

    objs = {}
    for key, ind in seen.items():
        if not ind.fitness.valid:
            ind.fitness.values = evaluator.evaluate(ind)
        _, fid, acc, comp = ind.fitness.values
        objs[key] = (fid, acc, comp)

    keys = list(objs.keys())
    front = []
    for k in keys:
        if not any(_dominates(objs[o], objs[k]) for o in keys if o != k):
            front.append(k)

    results = []
    for key in front:
        rule = seen[key]
        fid, acc, comp = objs[key]
        consequent = rule_consequent_class(rule, evaluator.X_disc, evaluator.y_black_box)
        results.append(
            {
                "rule": list(rule),
                "consequent_class": consequent,
                "fidelity": fid,
                "accuracy": acc,
                "comprehensibility": comp,
                "rule_length": rule_length(rule),
                "support": rule_support(rule, evaluator.X_disc),
            }
        )
    return results


# --------------------------------------------------------------------------
# Multi-run orchestration (Section 4.3.5: 30 independent runs, aggregated)
# --------------------------------------------------------------------------
def extract_rules(
    X_disc: np.ndarray,
    y_true: np.ndarray,
    y_black_box: np.ndarray,
    feature_names: List[str],
    config: GAConfig = GAConfig(),
    model_condition: str = "baseline_adam",
) -> List[dict]:
    """
    Runs the GA rule-extraction pipeline `config.n_independent_runs` times
    with distinct seeds, aggregates the non-dominated rule sets from every
    run into a single pool (deduplicated by rule signature), and attaches
    feature names + model condition metadata for downstream PROMETHEE II
    ranking (Section 4.6.1).
    """
    evaluator = RuleFitnessEvaluator(X_disc, y_true, y_black_box, config)
    n_features = X_disc.shape[1]

    pooled: dict = {}
    for run_idx in range(config.n_independent_runs):
        seed = config.random_seed_base + run_idx
        run_results = run_single_ga(evaluator, config, n_features, seed)
        for r in run_results:
            key = tuple(r["rule"])
            if key not in pooled:
                r["run_seed"] = seed
                pooled[key] = r
        print(
            f"[{model_condition}] run {run_idx + 1}/{config.n_independent_runs} "
            f"complete -- pool size: {len(pooled)}"
        )

    rules = list(pooled.values())
    for r in rules:
        r["model_condition"] = model_condition
        r["rule_readable"] = readable_rule(r["rule"], feature_names, r["consequent_class"])

    rules.sort(key=lambda r: (r["fidelity"] + r["accuracy"]), reverse=True)
    return rules


def readable_rule(rule: Sequence[int], feature_names: List[str], consequent: int) -> str:
    """Renders a chromosome as a human-readable IF-THEN string."""
    category_names = {0: "Low", 1: "Intermediate", 2: "High"}
    conditions = [
        f"{feature_names[i]} = {category_names[val]}"
        for i, val in enumerate(rule)
        if val != DONT_CARE
    ]
    antecedent = " AND ".join(conditions) if conditions else "TRUE"
    return f"IF {antecedent} THEN response_class = {consequent}"


# --------------------------------------------------------------------------
# Export for downstream PROMETHEE II / GSEA stages (Section 4.6)
# --------------------------------------------------------------------------
def export_rules(rules: List[dict], output_path: str) -> None:
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(rules, f, indent=2)
    print(f"Exported {len(rules)} rules to {output_path}")


# --------------------------------------------------------------------------
# End-to-end pipeline: load real (or synthetic) inputs, align, run GA
# --------------------------------------------------------------------------
def run_pipeline(
    condition: str,
    checkpoint_path: str = None,
    data_root: str = None,
    split: str = "test",
    synthetic: bool = False,
    n_synthetic_pairs: int = 2000,
    n_synthetic_cell_lines: int = 878,
    n_synthetic_genes: int = 200,
    config: GAConfig = None,
    device: str = "cpu",
    seed: int = 0,
) -> List[dict]:
    """
    Orchestrates: load model + precomputed split arrays -> get black-box
    LN_IC50 predictions -> threshold to y_true/y_black_box -> load gene-
    expression bins -> align to pair-level rows -> run the 30-run GA.

    In --synthetic mode, everything (model, precomputed features, gene
    bins) is placeholder data of the correct shape, so this exercises the
    full GA pipeline end-to-end without needing the real checkpoint or
    the blocked RNA-seq-derived bins.
    """
    config = config or GAConfig()

    if synthetic:
        print(f"[{condition}] SYNTHETIC MODE -- placeholder data, not real results.")
        rng = np.random.default_rng(seed)
        cell_idx = rng.integers(0, n_synthetic_cell_lines, size=n_synthetic_pairs)
        ln_ic50_pred = rng.normal(size=n_synthetic_pairs)
        ln_ic50_true = ln_ic50_pred + rng.normal(scale=0.3, size=n_synthetic_pairs)
        threshold = config.ln_ic50_threshold
        if threshold is None:
            threshold = float(np.median(ln_ic50_pred))
        y_black_box = (ln_ic50_pred < threshold).astype(int)
        y_true = (ln_ic50_true < threshold).astype(int)
        gene_bins_by_cell_line, feature_names = load_gene_expression_bins(
            synthetic=True, n_cell_lines=n_synthetic_cell_lines,
            n_genes=n_synthetic_genes, seed=seed,
        )
        X_disc = align_rule_frame_to_predictions(gene_bins_by_cell_line, cell_idx)
    else:
        if checkpoint_path is None or data_root is None:
            raise ValueError("checkpoint_path and data_root are required outside --synthetic mode.")
        model = load_black_box_model(checkpoint_path, device=device)
        (pathway_activity, mutation, drug_fingerprint,
         cell_idx, drug_idx, ln_ic50_true) = load_precomputed_split_arrays(data_root, split)

        threshold = config.ln_ic50_threshold
        if threshold is None:
            threshold = float(np.median(ln_ic50_true))
            print(f"[{condition}] No --ln-ic50-threshold given; using median "
                  f"of true LN_IC50 ({threshold:.4f}) as a placeholder split "
                  f"-- confirm the real clinical cutoff before final results.")

        y_black_box, ln_ic50_pred = get_black_box_predictions(
            model, pathway_activity, mutation, drug_fingerprint, threshold, device=device,
        )
        y_true = (ln_ic50_true < threshold).astype(int)

        gene_bins_by_cell_line, feature_names = load_gene_expression_bins(data_root=data_root)
        X_disc = align_rule_frame_to_predictions(gene_bins_by_cell_line, cell_idx)

    return extract_rules(
        X_disc, y_true, y_black_box, feature_names,
        config=config, model_condition=condition,
    )


# --------------------------------------------------------------------------
# CLI entry point
# --------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="GA rule extraction (Section 4.3) for a DeepCCDS checkpoint."
    )
    parser.add_argument(
        "--condition", choices=["baseline_adam", "sam_regularised"], required=True,
        help="Which experimental condition this run represents.",
    )
    parser.add_argument(
        "--checkpoint", default=None,
        help="Path to the DeepCCDS checkpoint, e.g. checkpoints/baseline_adam_real.pt "
             "(Stage 1) or the equivalent SAM checkpoint (Stage 2). Ignored with --synthetic.",
    )
    parser.add_argument(
        "--data-root", default=None,
        help="Path to data/deepccds_precomputed/ (see BASELINE_README.md Section 5). "
             "Ignored with --synthetic.",
    )
    parser.add_argument("--split", default="test", choices=["train", "valid", "test"])
    parser.add_argument(
        "--synthetic", action="store_true",
        help="Run against placeholder data of the correct shapes instead of the real "
             "checkpoint/data (use this now, since the real gene-expression bins are "
             "blocked on the RNA-seq file per BASELINE_README.md).",
    )
    parser.add_argument(
        "--ln-ic50-threshold", type=float, default=None,
        help="LN_IC50 cutoff for sensitive/resistant classification (Section 4.2.2 -- "
             "'fixed at the implementation stage'). Defaults to the median of the split.",
    )
    parser.add_argument(
        "--output", default=None,
        help="Output JSON path (defaults to rules_<condition>.json).",
    )
    parser.add_argument("--generations", type=int, default=50)
    parser.add_argument("--runs", type=int, default=30)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if not args.synthetic and (args.checkpoint is None or args.data_root is None):
        parser.error("--checkpoint and --data-root are required unless --synthetic is set.")

    config = GAConfig(
        n_generations=args.generations,
        n_independent_runs=args.runs,
        random_seed_base=args.seed,
        ln_ic50_threshold=args.ln_ic50_threshold,
    )

    rules = run_pipeline(
        condition=args.condition,
        checkpoint_path=args.checkpoint,
        data_root=args.data_root,
        split=args.split,
        synthetic=args.synthetic,
        config=config,
        seed=args.seed,
    )

    output_path = args.output or f"rules_{args.condition}.json"
    export_rules(rules, output_path)


if __name__ == "__main__":
    main()