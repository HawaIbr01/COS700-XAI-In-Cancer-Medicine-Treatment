# Stage 1 timing report (Erlotinib, 3 arms x 2 extractors x 5 folds x 3 seeds, 1 worker)

Run: `python -m src.per_drug.run --out-root artifacts/per_drug/stage1_timing --drugs Erlotinib --n-seeds 3 --workers 1`
30 cells, 90 seed runs, 0 errors, 449 s in the run loop (7:30 wall incl. imports). Peak RSS of the single process: 734,588 kB (0.70 GiB).
Per-seed timings: `stage1_timing/per_seed_timing.csv`; manifest: `stage1_timing/manifest.jsonl` (90 lines).

| extractor | mean s/seed | median | max | per-cell overhead (aggregate+score) |
|---|---|---|---|---|
| GA | 7.77 | 7.60 | 10.45 | ~1.5 s |
| NSGA-II | 0.95 | 0.90 | 1.85 | ~0.35 s |

Per drug at S seeds, single core: 15 x [(7.77 + 0.95) S + 1.85] ~= 130.8 S + 28 s (+ ~10 s data load once per drug in a process).

| plan | single core | 3 workers | 4 workers |
|---|---|---|---|
| 14 drugs, 30 seeds | 15.4 h | 5.1 h | 3.9 h |
| 14 drugs, 10 seeds | 5.2 h | 1.7 h | 1.3 h |
| 379 drugs, 10 seeds | 140.8 h | 47 h | 35 h |

Worker-scaling is assumed linear, not measured. 8 logical cores; 3.8 GB RAM total (2.9 GB free at start).
