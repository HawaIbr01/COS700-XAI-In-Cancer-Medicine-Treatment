"""
PROMETHEE II ranking, statistical testing, GSEA, CART baseline.

OWNER: Hawa (feat/evaluation-pipeline)
Methodology: §4.6

CONTRACT (docs/CONTRACTS.md Arm 5):
    Receives: artifacts/rules/{ga,nsga}_{baseline,sam}.json,
        artifacts/predictions/*, artifacts/data/feature_map.json,
        src.common.metrics
    Produces:
        artifacts/reports/  (ranked rule tables, phi values,
        Mann-Whitney results, pathway-enrichment tables, comparison
        figures — everything the final report pulls from)

    This arm is the terminal consumer of the pipeline. Acceptance
    criterion (§5 of CONTRACTS.md): a single command reproduces all
    scorecards from the artifacts on disk, no manual steps.

TODO (Hawa):
    - Load all four rule pools via src.common.schema.load_rule_pool.
    - PROMETHEE II (pymcdm) over {fidelity_f1, comprehensibility,
      stability}: any comparison (baseline vs SAM, evolutionary vs
      CART) must run as a SINGLE unified PROMETHEE execution so net
      flows (phi) are commensurable (§4.6.1) — do not rank pools
      separately and then compare phi values across separate runs.
    - Take top-10 rules per condition from the unified ranking.
    - Mann-Whitney U test (scipy.stats.mannwhitneyu) on the phi
      distributions of the two conditions being compared, two
      independent samples, alpha=0.05 (§4.6.2).
    - GSEA (GSEApy) over-representation of each rule pool's 'genes'
      against KEGG/Enrichr; report enrichment p-values (§4.6.3).
    - CART baseline (scikit-learn DecisionTreeClassifier, depth-bounded)
      fitted to each condition's model predictions; extract path
      rules, score them with the SAME src.common.metrics functions,
      and feed them into the SAME unified PROMETHEE run (§4.6.4).
    - Compute per-rule stability (src.common.metrics.stability) across
      GDSC cancer subtypes — you own the final aggregation choice
      here; if it changes from the provisional variance-based version
      in metrics.py, update that docstring too.
    - Publish your final PROMETHEE weight vector and stability
      definition back into configs/pipeline.yaml so other arms can
      see what "current" looks like.
"""

from __future__ import annotations

from pathlib import Path

from src.common.config import load_config
from src.common.schema import load_rule_pool
from src.common import metrics

ARTIFACTS_DIR = Path(__file__).resolve().parents[2] / "artifacts"


def run_evaluation() -> None:
    cfg = load_config()

    # TODO: load all four rule pools:
    #   ga_baseline = load_rule_pool(ARTIFACTS_DIR / "rules" / "ga_baseline.json")
    #   ga_sam      = load_rule_pool(ARTIFACTS_DIR / "rules" / "ga_sam.json")
    #   nsga_baseline = load_rule_pool(ARTIFACTS_DIR / "rules" / "nsga_baseline.json")
    #   nsga_sam      = load_rule_pool(ARTIFACTS_DIR / "rules" / "nsga_sam.json")
    # TODO: compute stability per rule (needs per-subgroup predictions —
    #       coordinate with Angelique/Bahiya on how subgroup/tissue-type
    #       labels are exposed in the predictions parquet)
    # TODO: unified PROMETHEE II run(s) — baseline vs SAM, GA vs NSGA,
    #       evolutionary vs CART — each as its own single unified matrix
    # TODO: Mann-Whitney U on phi distributions per comparison
    # TODO: GSEA over-representation per rule pool's genes
    # TODO: CART baseline: fit, extract rules, score, fold into
    #       the relevant unified PROMETHEE run
    # TODO: write all tables/figures to artifacts/reports/
    raise NotImplementedError("run_evaluation: see TODOs above")


if __name__ == "__main__":
    run_evaluation()
