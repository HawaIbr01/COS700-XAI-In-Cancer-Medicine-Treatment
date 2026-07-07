"""
Multi-objective NSGA-II rule extraction (DEAP).

OWNER: Sakhile (feat/nsga2-rule-extraction)
Methodology: §4.4

CONTRACT (docs/CONTRACTS.md Arm 4):
    Receives: same as GA (Arm 3) plus src.extraction.common (shared
        encoding — keep it that way; coordinate with Melisa before
        diverging).
    Produces:
        artifacts/rules/nsga_baseline.json
        artifacts/rules/nsga_sam.json
    (same §0.4 schema as GA — this is what makes the GA-vs-NSGA-II
    comparison in Evaluation fair; use schema.save_rule_pool)

TODO (Sakhile):
    - Build the DEAP toolbox: individual = chromosome (see
      src.extraction.common — same representation as GA), TRUE
      multi-objective fitness: maximise fidelity_f1, minimise rule
      length (do not scalarise into one number).
    - Fast non-dominated sorting + crowding distance (deap.tools has
      building blocks for this — selNSGA2).
    - Binary tournament selection using the crowded-comparison
      operator; two-point crossover + mutation matched to GA's
      operators for comparability (§4.4).
    - Run cfg.nsga.generations generations, population = cfg.nsga.pop.
    - Repeat for cfg.nsga.runs (30) independent seeded runs; aggregate
      the non-dominated Pareto front across all runs.
    - Do this once per condition ("baseline" and "sam").
    - Decode + score exactly as GA does (reuse
      src.extraction.common.decode_chromosome /
      src.extraction.common.predict_with_rule — don't reimplement).
    - If any operator deviates from GA's (e.g. because the
      multi-objective design needs it), record that in meta and flag
      it to Melisa + the team.
"""

from __future__ import annotations

from pathlib import Path

from src.common.config import load_config
from src.common.schema import save_rule_pool
from src.extraction.common import decode_chromosome, predict_with_rule

ARTIFACTS_DIR = Path(__file__).resolve().parents[2] / "artifacts"


def extract_rules_nsga(condition: str) -> dict:
    """
    Run the full 30-run NSGA-II extraction for one condition.

    Args:
        condition: "baseline" or "sam".

    Returns:
        a dict conforming to the §0.4 rule-pool schema (not yet saved),
        where every rule's non_dominated field reflects genuine Pareto
        non-domination (no rule dominated on both fidelity and length).
    """
    cfg = load_config()

    # TODO: load artifacts/data/feature_map.json, response_classes.json
    # TODO: load artifacts/predictions/{condition}_*.parquet as fidelity target
    # TODO: build DEAP toolbox with selNSGA2 (see TODOs in module docstring)
    # TODO: for run_id in range(cfg.nsga.runs): run NSGA-II, collect the
    #       Pareto front with their run_id
    # TODO: decode + score all collected individuals into rule dicts
    # TODO: assemble {"meta": {...}, "rules": [...]} per §0.4 schema
    raise NotImplementedError("extract_rules_nsga: see TODOs above")


def main() -> None:
    for condition in ("baseline", "sam"):
        pool = extract_rules_nsga(condition)
        out_path = ARTIFACTS_DIR / "rules" / f"nsga_{condition}.json"
        save_rule_pool(pool, out_path)
        print(f"wrote {out_path} ({len(pool['rules'])} rules)")


if __name__ == "__main__":
    main()
