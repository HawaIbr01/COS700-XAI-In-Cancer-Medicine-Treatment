# feat/baseline-model — Stage 1 (Angie's branch)

Implements Section 4.2 of the proposal: GDSC preprocessing, the DeepCCDS
architecture, Stage 1 Adam baseline training with deliberate overfitting, and the
Clever Hans qualitative sanity check.

## Files

| File | Proposal section | What it does |
|---|---|---|
| `src/data/preprocessing.py` | 4.2.2 | Merge GDSC sub-tables, clean, one-hot encode, Z-score normalise, discretise into Low/Intermediate/High bins, compute LN(IC50), write the **locked 80/10/10 split** everyone else's branch should reuse. |
| `src/data/dataset.py` | 4.2.2 | PyTorch `Dataset` that loads the processed parquet + split indices. |
| `src/model/deepccds.py` | 3.4.1 | DeepCCDS-style architecture: pathway-masked layer (38 pathways) + optional mutation/drug autoencoders + MLP head. |
| `src/model/train_baseline.py` | 4.2.3 | Adam, no regularisation, "reverse early stopping" (patience=30, stop once val loss has plateaued/risen for 10 consecutive epochs), computes the generalisation gap, R², RMSE, saves the checkpoint. |
| `src/model/clever_hans_audit.py` | 4.2.4 | Gradient-based (vanilla input-gradient) attribution sanity check — flags housekeeping genes / high sample-to-sample instability in the top-attributed genes. |
| `tests/test_baseline_model.py` | — | Unit tests using synthetic tensors, so they run without the real GDSC data. All pass. |

## Known gaps / things to fix before this is "done" (flagging honestly rather than
hiding them):

1. **Real GDSC file paths/columns.** `preprocessing.py` assumes raw files at
   `data/raw/gdsc_expression.csv` etc. with a `GENE_` column prefix and a
   `COSMIC_ID`/`DRUG_ID` join key — these are placeholders. Once we've pulled the
   actual GDSC export, update `RAW_*_PATH` and the merge keys in
   `merge_sub_datasets()`.
2. **Pathway mask.** `DeepCCDS.__init__` falls back to a fully-dense gene→pathway
   layer if no `pathway_mask` is supplied. `build_pathway_mask()` is ready to accept
   a real gene→pathway-index dict — we still need to source the actual 38-pathway
   gene membership (KEGG/Reactome, matching what Wu et al. 2025 used for DeepCCDS).
3. **Mutation/drug branches untested on real data.** The autoencoder branches work
   in the smoke test (synthetic tensors) but haven't been run against real mutation
   / drug-descriptor tables yet.
4. **LN_IC50 → sensitive/resistant threshold** for rule fidelity (needed by Melisa
   and Sakhile's branches) isn't fixed yet — Section 4.2.2 says it'll be "fixed at
   the implementation stage." Worth a team discussion once we've looked at the
   LN_IC50 distribution.

## How to run

```bash
pip install -r requirements.txt

# once data/raw/ has the real GDSC files:
python -m src.data.preprocessing

python -m src.model.train_baseline
python -m src.model.clever_hans_audit

pytest tests/ -v
```

I ran the whole pipeline end-to-end on synthetic data to make sure the plumbing
works (train R² ~0.71 vs val R² ~0.09 after 60 epochs of unregularised training —
exactly the overfitting signature we want as the Fragility Baseline). Once real
GDSC data lands, `train_baseline.py` and `clever_hans_audit.py` should run
unchanged against it.
