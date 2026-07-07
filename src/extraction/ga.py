"""
Single-objective Genetic Algorithm rule extraction (DEAP).

OWNER: Melisa (feat/ga-rule-extraction)
Methodology: §4.3

CONTRACT (docs/CONTRACTS.md Arm 3):
    Receives: artifacts/data/*, artifacts/checkpoints/{baseline,sam}/,
        artifacts/predictions/{baseline,sam}_*.parquet,
        src.common.metrics, src.extraction.common
    Produces:
        artifacts/rules/ga_baseline.json
        artifacts/rules/ga_sam.json
    (both validated against src.common.schema.validate_rule_pool
    before being written — use schema.save_rule_pool, don't
    json.dump directly)

TODO (Melisa):
    - Build the DEAP toolbox: individual = chromosome (see
      src.extraction.common), scalarised fitness combining
      fidelity_f1, comprehensibility penalty, and held-out accuracy.
    - Tournament selection, two-point crossover, mutation (add/drop a
      condition), rule pruning (drop low-confidence/low-support
      conditions each generation), aging evolution for diversity.
    - Run cfg.ga.generations generations, population sized between
      cfg.ga.pop_min and cfg.ga.pop_max.
    - Repeat for cfg.ga.runs (30) independent seeded runs; aggregate
      the non-dominated rules across all runs into one pool.
    - Do this once per condition ("baseline" and "sam" — i.e. against
      each checkpoint's predictions).
    - Decode each surviving chromosome via
      src.extraction.common.decode_chromosome, compute fidelity_f1 /
      comprehensibility / coverage, assemble into the §0.4 schema,
      and schema.save_rule_pool(...) to artifacts/rules/.

    Do NOT pre-rank beyond non-domination — PROMETHEE ranking is the
    Evaluation arm's job on the full raw pool.
"""

from __future__ import annotations

from pathlib import Path

from src.common.config import load_config
from src.common.schema import save_rule_pool
from src.extraction.common import decode_chromosome, predict_with_rule

ARTIFACTS_DIR = Path(__file__).resolve().parents[2] / "artifacts"


def extract_rules_ga(condition: str) -> dict:
    """
    Run the full 30-run GA extraction for one condition.

    Args:
        condition: "baseline" or "sam" — selects which checkpoint's
            predictions to extract against.

    Returns:
        a dict conforming to the §0.4 rule-pool schema (not yet saved).
    """
    cfg = load_config()

    # TODO: load artifacts/data/feature_map.json, response_classes.json
    # TODO: load artifacts/predictions/{condition}_*.parquet as fidelity target
    # TODO: build DEAP toolbox (see TODOs in module docstring)
    # TODO: for run_id in range(cfg.ga.runs): run the GA, collect
    #       non-dominated individuals with their run_id
    # TODO: decode + score all collected individuals into rule dicts
    # TODO: assemble {"meta": {...}, "rules": [...]} per §0.4 schema
    raise NotImplementedError("extract_rules_ga: see TODOs above")


def main() -> None:
    for condition in ("baseline", "sam"):
        pool = extract_rules_ga(condition)
        out_path = ARTIFACTS_DIR / "rules" / f"ga_{condition}.json"
        save_rule_pool(pool, out_path)
        print(f"wrote {out_path} ({len(pool['rules'])} rules)")


if __name__ == "__main__":
    main()
