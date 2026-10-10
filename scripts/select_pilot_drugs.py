"""Stage 0E: choose the per-drug pilot drugs BEFORE any extraction is run.

Groups (frozen design, docs/per_drug_analysis_plan.md):
  best_predicted : 5 drugs with the highest per-drug VALIDATION macro-F1 of the
                   network's class_pred vs the TRUE class, averaged over the
                   three arms (adam, adam_wd, sam). "Validation" = the network's
                   own val split. Ties broken alphabetically by drug name.
  positive_controls : Erlotinib, PLX-4720, Olaparib, Selumetinib.
  random_5       : 5 drugs drawn with random.Random(700).sample from the sorted
                   candidates NOT already in the two groups above.
A drug in two groups is listed once in `pilot_drugs`.

Only class_pred, the true class and the val split are used here: no rule is
extracted or inspected.

Run:  python scripts/select_pilot_drugs.py
"""
from __future__ import annotations

import json
import random
from pathlib import Path

import pandas as pd
from sklearn.metrics import f1_score

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data/processed/real_extraction"
NEW_PRED = ROOT / "artifacts/per_drug/data/predictions"
OUT = ROOT / "artifacts/per_drug/pilot_drugs.json"

RANDOM_SEED = 700
POSITIVE_CONTROLS = {
    "Erlotinib": ["EGFR_mut"],
    "PLX-4720": ["BRAF_mut"],
    "Olaparib": ["BRCA1_mut", "BRCA2_mut"],
    "Selumetinib": ["BRAF_mut", "KRAS_mut"],
}


def main():
    df = pd.read_parquet(DATA / "data/gdsc_processed.parquet", columns=["row_id", "drug_name", "response_class", "split"])
    val = df[df["split"] == "val"]
    arm_pred = {
        "adam": DATA / "predictions/baseline_val.parquet",
        "adam_wd": NEW_PRED / "adam_wd_val.parquet",
        "sam": DATA / "predictions/sam_val.parquet",
    }
    candidates = sorted(df["drug_name"].unique())
    f1 = {}
    for arm, path in arm_pred.items():
        m = val.merge(pd.read_parquet(path)[["row_id", "class_pred"]], on="row_id", validate="one_to_one")
        assert len(m) == len(val), f"{arm}: prediction rows do not cover the val split"
        f1[arm] = {d: float(f1_score(g["response_class"].astype(str), g["class_pred"], average="macro"))
                   for d, g in m.groupby("drug_name")}
    table = pd.DataFrame(f1).reindex(candidates)
    assert not table.isna().any().any(), "a candidate drug has no val rows in some arm"
    table["mean_val_macro_f1"] = table.mean(axis=1)
    table["n_val_rows"] = val.groupby("drug_name").size().reindex(candidates)

    ranked = table.reset_index(names="drug").sort_values(["mean_val_macro_f1", "drug"], ascending=[False, True])
    best5 = ranked["drug"].head(5).tolist()
    controls = list(POSITIVE_CONTROLS)
    missing = [c for c in controls if c not in candidates]
    assert not missing, f"positive controls missing from candidates: {missing}"

    already = set(best5) | set(controls)
    pool = sorted(d for d in candidates if d not in already)
    random5 = random.Random(RANDOM_SEED).sample(pool, 5)

    pilot, groups_of = [], {}
    for name, drugs in (("best_predicted", best5), ("positive_control", controls), ("random", random5)):
        for d in drugs:
            groups_of.setdefault(d, []).append(name)
            if d not in pilot:
                pilot.append(d)

    payload = {
        "n_candidates": len(candidates),
        "random_seed": RANDOM_SEED,
        "random_draw_pool_size": len(pool),
        "best_predicted": best5,
        "positive_controls": POSITIVE_CONTROLS,
        "random": random5,
        "pilot_drugs": pilot,
        "n_pilot_drugs": len(pilot),
        "groups_of": groups_of,
        "per_drug_val_macro_f1": {r.drug: {a: round(getattr(r, a), 6) for a in arm_pred} | {"mean": round(r.mean_val_macro_f1, 6),
                                                                                           "n_val_rows": int(r.n_val_rows)}
                                  for r in ranked.head(15).itertuples()},
        "note": "per_drug_val_macro_f1 lists only the 15 best-ranked drugs; full table in pilot_drug_val_f1_all.csv",
    }
    OUT.write_text(json.dumps(payload, indent=2))
    ranked.to_csv(OUT.with_name("pilot_drug_val_f1_all.csv"), index=False)
    print(json.dumps({k: payload[k] for k in ("best_predicted", "random", "pilot_drugs", "n_pilot_drugs", "groups_of")}, indent=1))


if __name__ == "__main__":
    main()
