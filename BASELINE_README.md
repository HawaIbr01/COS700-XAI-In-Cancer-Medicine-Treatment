# feat/baseline-model — Stage 1 (Angie's branch)

Implements Section 4.2 of the proposal: GDSC preprocessing, the DeepCCDS
architecture, Stage 1 Adam baseline training with deliberate overfitting, and the
Clever Hans qualitative sanity check.

**Updated 28 July** against your real GDSC1/GDSC2 files and the actual DeepCCDS
paper (Wu et al., 2025) -- several things in the first version were placeholder
guesses that turned out to not match the paper. Corrected below.

## Files

| File | Proposal section | What it does |
|---|---|---|
| `src/data/preprocessing.py` | 4.2.2 | Merges **GDSC1 + GDSC2** (verified against your real files: 517,309 combined rows, 57,888 overlapping pairs resolved in favour of GDSC2 -- the exact dedup rule the paper states), cleans, one-hot encodes, Z-score normalises, bins expression, writes the **locked 80/10/10 split**. |
| `src/data/dataset.py` | 4.2.2 | PyTorch `Dataset` that loads the processed parquet + split indices. |
| `src/model/deepccds.py` | 3.4.1 | **Corrected** DeepCCDS architecture matching the paper's exact numbers (see below). |
| `src/model/train_baseline.py` | 4.2.3 | Adam (lr=1e-3, batch=1024, matching the paper's grid search), "reverse early stopping," Eq. 2 combined loss, generalisation gap / R² / RMSE. |
| `src/model/clever_hans_audit.py` | 4.2.4 | Gradient-based attribution check, updated for the new model interface. |
| `tests/test_baseline_model.py` | — | 10 passing unit tests on synthetic tensors. |

## What changed after reading the real paper (important -- please read this)

1. **Pathway activity is NOT a learnable layer.** I originally built a masked
   *learnable* linear layer mapping gene expression → 38 pathways. The paper
   actually computes this with **ssGSEA as a preprocessing step**, entirely
   outside the neural network -- the network just receives a 38-dim activity
   vector directly. `deepccds.py` is corrected to take `pathway_activity` as
   input rather than raw gene expression.
2. **Autoencoder architecture corrected to the paper's exact numbers**: encoder
   hidden dims (300, 100), bottleneck 30 (I'd guessed 32 and a single hidden
   layer). Sigmoid output + BCE pretraining loss (inputs are binary: mutation
   calls, fingerprint bits).
3. **Feedforward head corrected**: two hidden layers *the same size as the input
   layer* (98 = 38 pathway + 30 mutation + 30 drug, when all three branches are
   active), not the arbitrary (128, 64) I'd guessed.
4. **Combined loss (Eq. 2)** now implemented: `MSE(prediction) + MSE(mutation
   reconstruction) + MSE(drug reconstruction)`, jointly minimised -- not a
   frozen-autoencoder setup.
5. **GDSC1 + GDSC2 merge added**: per Hawa's note and the paper's own text, the
   paper pairs both releases and keeps the GDSC2 value on overlap. Verified this
   against your actual files.
6. **Batch size corrected to 1024** and learning rate confirmed at 1e-3 (both
   match the paper's grid search result, so no change needed there).

## Known gaps still open (being upfront, not hiding them)

1. **Mutation matrix (534 COSMIC CGC driver genes, binary) -- not yet built.**
   This requires GDSC's separate mutation annotation files (not the response or
   expression files you sent) filtered to the COSMIC Cancer Gene Census list.
2. **Drug fingerprints (1024-bit Morgan fingerprints) -- not yet built.** Needs
   PubChemPy to fetch SMILES per DRUG_NAME, then RDKit (Python equivalent of the
   paper's R `rcdk`) to compute Morgan fingerprints.
3. **Pathway activity (the real 38-pathway ssGSEA signal) -- not yet built.**
   This is the biggest remaining piece: PPI network → Random-Walk-with-Restart
   from 534 driver genes → GSEA against 323 KEGG pathways → keep 38 with ES>0,
   FDR<0.2 → ssGSEA to score each cell line. **Before re-deriving this from
   scratch, check the DeepCCDS GitHub (https://github.com/hanjunwei-lab/DeepCCDS)
   and the paper's Table S1 (Supporting Information) for a precomputed
   gene→pathway mapping** -- it's very likely already published there and would
   save significant time.
4. **`TemporaryPathwayProxy` in `train_baseline.py` is a stand-in** (a fixed
   random projection of Z-scored expression to 38 dims) so the training loop is
   runnable end-to-end right now. It is explicitly flagged in the code and the
   output JSON with a `WARNING` field -- results from it are NOT the real
   DeepCCDS signal and shouldn't be reported as final.
5. Cell line ID mapping (`Cell_Lines_Details.xlsx`) and the RMA expression file
   (`Cell_line_RMA_proc_basalExp.txt`) are still needed on top of what's been
   sent -- see `preprocessing.py`'s docstring for exact download links.
6. The LN_IC50 sensitive/resistant threshold for rule fidelity (needed by
   Melisa/Sakhile) still isn't fixed -- worth a team discussion.

## How to run

```bash
pip install -r requirements.txt

# place data/raw/GDSC1_fitted_dose_response_27Oct23.xlsx,
#       data/raw/GDSC2_fitted_dose_response_27Oct23.xlsx,
#       data/raw/Cell_line_RMA_proc_basalExp.txt,
#       data/raw/Cell_Lines_Details.xlsx

python -m src.data.preprocessing
python -m src.model.train_baseline
python -m src.model.clever_hans_audit

pytest tests/ -v
```

Ran the full pipeline end-to-end on synthetic data again after these changes --
all 10 tests pass, training loop runs cleanly with the corrected architecture and
Eq. 2 loss.
