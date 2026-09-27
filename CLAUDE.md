# CLAUDE.md — COS700 XAI-In-Cancer-Medicine-Treatment

COS700 Honours thesis project (UP, 2026). Tests whether replacing Adam with SAM in the DeepCCDS drug-response framework widens the generalisation gap enough to improve symbolically-extracted rules downstream.

## Commit conventions

Do not add a `Co-Authored-By: Claude` (or any Claude/Anthropic attribution) trailer to commit messages, and do not include a `Claude-Session` link. Commits should be authored as Hawa alone. Write plain, conventional commit messages (e.g. `feat: ...`, `fix: ...`, `docs: ...`) with no AI attribution footer.

## Environment setup

```bash
python3 -m venv .venv
source .venv/bin/activate            # Linux/macOS
# .venv\Scripts\activate             # Windows

# All packages (except torch):
pip install -r requirements.txt

# PyTorch — CPU-only build (project hardware has no GPU):
pip install torch --index-url https://download.pytorch.org/whl/cpu
```

The `.venv/` directory is gitignored. Every teammate runs this once after cloning.

## Raw data setup

The three raw GDSC input CSVs must be in `data/deepccds_precomputed/` before preprocessing:

- `GDSC_ssgsea_input.csv`
- `GDSC_mutation_input.csv`
- `GDSC_SMILE_input.csv`
- `GDSC_train_IC50_by_borh_cv00.csv`
- `GDSC_valid_IC50_by_borh_cv00.csv`
- `GDSC_test_IC50_by_borh_cv00.csv`

Download from: https://github.com/hanjunwei-lab/DeepCCDS (Code → Download ZIP → `data/` folder).

After downloading, run preprocessing to produce `artifacts/data/` and the cached `.npy` matrices:

```bash
python -m src.data.preprocess
```

The processed parquet (`gdsc_processed.parquet`, `splits.json`, etc.) are committed. The `.npy` matrix files are gitignored (`*.npy`) and must be regenerated locally.

## Running the pipeline

```bash
# Stage 1: Adam baseline
python -m src.model.baseline

# Stage 2: SAM
python -m src.model.sam

# Investigation scripts (feat/generalisation-gap-intervention)
python -m src.investigations.stage1_gap_analysis
python -m src.investigations.stage3_joint_sweep      # long-running, ~2-3h
python -m src.investigations.stage3_analysis         # after sweep completes
python -m src.investigations.stage3_analysis --partial  # mid-sweep check
```

## Branch structure

| Branch | Owner | Purpose |
|--------|-------|---------|
| `main` | All | Milestone merges only |
| `feat/baseline-model-2` | Angelique | Adam baseline, data preprocessing |
| `feat/sam-intervention` | Bahiya | SAM optimiser arm |
| `feat/ga-rule-extraction` | Melisa | GA rule extractor |
| `feat/nsga2-rule-extraction` | Sakhile | NSGA-II extractor |
| `feat/evaluation-pipeline` | Hawa | PROMETHEE II, statistics, GSEA |
| `feat/generalisation-gap-intervention` | Hawa | SAM generalisation-gap investigation |

See `docs/CONTRACTS.md` for the full arm-to-arm artifact contract.

## Key artifact locations

- `artifacts/data/` — preprocessed GDSC data (parquet + split JSON) — committed
- `artifacts/checkpoints/` — model weights (`.pt`) — **not committed** (`.pt` in gitignore)
- `artifacts/reports/` — scorecards and training histories — committed
- `artifacts/sam_investigation/` — investigation plots and summaries — committed
- `docs/investigations/sam-generalisation/journal.md` — running investigation journal
- `configs/pipeline.yaml` — single source of truth for shared hyperparameters

## Hardware context

Project hardware: ~8GB RAM, i5 CPU-only (no GPU). WSL2 on Windows or macOS.
- All training is on CPU; `torch.cuda.is_available()` returns False.
- Full SAM training run (~200 epochs on 256k samples): ~5 hours.
- Stage 3 sweep (16 configs, 10% subsample): ~2-3 hours.

## Testing

```bash
pytest tests/ -v
```

Tests use synthetic fixtures in `tests/fixtures/synthetic/` — never real data.
