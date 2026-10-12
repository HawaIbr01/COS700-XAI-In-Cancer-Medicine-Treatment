"""Resumable driver for the per-drug pilot.

One checkpoint file per (drug, arm, extractor, fold):
    <out-root>/cells/<drug-slug>/<arm>__<extractor>__fold<k>.json
A cell whose file exists with the same key (incl. n_seeds, n_gen, min_cover_n)
and design_hash is skipped on restart; a file from a different design/seed
count is NEVER overwritten (the run aborts) so stages cannot mix.
Every finished cell appends one line per seed to <out-root>/manifest.jsonl
(git commit, design hash, arm, extractor, drug, fold, seed, wall-clock s).

    nohup python -m src.per_drug.run --out-root artifacts/per_drug/pilot \
        --drugs-from artifacts/per_drug/pilot_drugs.json --n-seeds 30 --workers 5 \
        > artifacts/per_drug/pilot/run.log 2>&1 &
"""
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import re
import sys
import time
from pathlib import Path

# One thread per worker process: set before numpy/pandas are imported so the BLAS pools pick it up
# and N workers never oversubscribe the CPU.
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[_v] = "1"

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.per_drug.config import ARMS, DESIGN, EXTRACTORS, design_hash  # noqa: E402


def slug(drug: str) -> str:
    return re.sub(r"[^A-Za-z0-9]+", "-", drug).strip("-")


def cell_path(out_root: Path, drug: str, arm: str, extractor: str, fold: int) -> Path:
    return out_root / "cells" / slug(drug) / f"{arm}__{extractor}__fold{fold}.json"


def _same_design(existing: dict, key: dict) -> bool:
    return existing.get("key") == key and existing.get("manifest", {}).get("design_hash") == design_hash()


_CACHE: dict = {}
_INPUT_HASHES: dict = {}      # set once in main() before workers fork; embedded in every cell's manifest


def _work(task):
    drug, arm, extractor, fold, n_seeds, n_gen, min_cover_n, out_root = task
    from src.per_drug import data as D
    from src.per_drug import extract as X
    if _CACHE.get("drug") != drug:
        _CACHE.clear()
        _CACHE["drug"], _CACHE["dd"] = drug, D.load_drug_frame(drug)
    cell = X.run_cell(_CACHE["dd"], arm, extractor, fold, n_seeds, n_gen=n_gen, min_cover_n=min_cover_n)
    cell["manifest"]["input_sha256"] = _INPUT_HASHES
    path = cell_path(Path(out_root), drug, arm, extractor, fold)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(cell))
    os.replace(tmp, path)                        # atomic: a cell file is either complete or absent
    return cell["key"], cell["manifest"], cell["seed_rows"], cell["wall_clock_s"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-root", required=True)
    ap.add_argument("--drugs", nargs="*", default=None)
    ap.add_argument("--drugs-from", default=None, help="pilot_drugs.json (uses its pilot_drugs list)")
    ap.add_argument("--n-seeds", type=int, required=True, help="no default: state the seed count explicitly")
    ap.add_argument("--n-gen", type=int, default=DESIGN["n_gen"])
    ap.add_argument("--min-cover-n", type=int, default=DESIGN["min_cover_n"])
    ap.add_argument("--folds", type=int, nargs="*", default=list(range(DESIGN["n_folds"])))
    ap.add_argument("--arms", nargs="*", default=list(ARMS))
    ap.add_argument("--extractors", nargs="*", default=list(EXTRACTORS))
    ap.add_argument("--workers", type=int, default=1)
    args = ap.parse_args()

    drugs = list(args.drugs or [])
    if args.drugs_from:
        drugs += [d for d in json.loads(Path(args.drugs_from).read_text())["pilot_drugs"] if d not in drugs]
    if not drugs:
        ap.error("no drugs given")
    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    from src.per_drug import provenance as P          # once per run, before any cell is scheduled
    prov = P.collect()
    _INPUT_HASHES.update(prov["input_sha256"])
    now = time.gmtime()
    record_json = json.dumps({**prov, "design_hash": design_hash(), "argv": sys.argv[1:],
                              "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", now)}, indent=2)
    # run_provenance.json is the latest start; the timestamped copy is never overwritten, so resumes keep a history
    (out_root / "run_provenance.json").write_text(record_json)
    (out_root / f"run_provenance_{time.strftime('%Y%m%dT%H%M%SZ', now)}.json").write_text(record_json)

    tasks, skipped = [], 0
    for drug in drugs:
        for arm in args.arms:
            for ext in args.extractors:
                for fold in args.folds:
                    key = {"drug": drug, "arm": arm, "extractor": ext, "fold": fold, "n_seeds": args.n_seeds,
                           "n_gen": args.n_gen, "min_cover_n": args.min_cover_n}
                    p = cell_path(out_root, drug, arm, ext, fold)
                    if p.exists():
                        existing = json.loads(p.read_text())
                        if not _same_design(existing, key):
                            sys.exit(f"REFUSING to overwrite {p}: it was made with a different key/design "
                                     f"({existing.get('key')} / {existing.get('manifest', {}).get('design_hash')}). "
                                     f"Use a new --out-root or log a re-run.")
                        skipped += 1
                        continue
                    tasks.append((drug, arm, ext, fold, args.n_seeds, args.n_gen, args.min_cover_n, str(out_root)))
    print(f"[run] design_hash={design_hash()} cells to run={len(tasks)} already done={skipped} "
          f"workers={args.workers}", flush=True)

    manifest = out_root / "manifest.jsonl"
    t0 = time.time()
    done = 0

    def record(res):
        nonlocal done
        key, man, seed_rows, wall = res
        with open(manifest, "a", encoding="utf-8") as f:
            for r in seed_rows:
                f.write(json.dumps({"git_commit": man["git_commit"], "design_hash": man["design_hash"],
                                    "arm": key["arm"], "extractor": key["extractor"], "drug": key["drug"],
                                    "fold": key["fold"], "seed": r["seed"], "wall_clock_s": r["seconds"],
                                    "cell_wall_clock_s": wall}) + "\n")
        done += 1
        print(f"[run] {done}/{len(tasks)} {key['drug']} {key['arm']} {key['extractor']} fold{key['fold']} "
              f"cell={wall:.0f}s elapsed={time.time() - t0:.0f}s", flush=True)

    if args.workers <= 1:
        for t in tasks:
            record(_work(t))
    else:
        with mp.get_context("fork").Pool(args.workers, maxtasksperchild=None) as pool:
            for res in pool.imap_unordered(_work, tasks, chunksize=1):
                record(res)
    print(f"[run] finished in {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
