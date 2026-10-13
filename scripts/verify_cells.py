"""Audit a per-drug pilot output tree: <out-root>/cells/<drug-slug>/<arm>__<extractor>__fold<k>.json

Reports expected vs present cells, empty/unparseable files, stray .tmp files, cells whose key does not match
their path, mixed design_hash / n_seeds / git_commit across cells, cells missing manifest.input_sha256 or whose
input_sha256 differs from configs/expected_hashes.json, cells with no rows in <out-root>/manifest.jsonl (a kill
between the cell write and the manifest append), and, per extractor, which drugs are complete (all arms x folds
present and readable) and which are not. Only complete drugs count in the analysis.
Read-only. Exit code 1 if any problem is found (incomplete drugs count as a problem unless --allow-incomplete).

    python scripts/verify_cells.py --out-root artifacts/per_drug/pilot \
        --drugs-from configs/pilot2_slice_1.json --n-seeds 10
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.per_drug.config import ARMS, DESIGN, EXTRACTORS  # noqa: E402

EXPECTED_HASHES = ROOT / "configs" / "expected_hashes.json"


def slug(drug: str) -> str:
    return re.sub(r"[^A-Za-z0-9]+", "-", drug).strip("-")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-root", required=True)
    ap.add_argument("--drugs", nargs="*", default=None)
    ap.add_argument("--drugs-from", default=None, help="JSON file with a pilot_drugs list")
    ap.add_argument("--arms", nargs="*", default=list(ARMS))
    ap.add_argument("--extractors", nargs="*", default=list(EXTRACTORS))
    ap.add_argument("--folds", type=int, nargs="*", default=list(range(DESIGN["n_folds"])))
    ap.add_argument("--n-seeds", type=int, default=None, help="if given, any cell with another n_seeds is a problem")
    ap.add_argument("--expected-hashes", default=str(EXPECTED_HASHES))
    ap.add_argument("--allow-incomplete", action="store_true",
                    help="incomplete drugs / missing cells are reported but do not make the exit code non-zero")
    ap.add_argument("--max-list", type=int, default=15, help="max items printed per problem class")
    args = ap.parse_args()

    drugs = list(args.drugs or [])
    if args.drugs_from:
        drugs += [d for d in json.loads(Path(args.drugs_from).read_text())["pilot_drugs"] if d not in drugs]
    if not drugs:
        ap.error("no drugs given")
    out_root = Path(args.out_root)
    cells_dir = out_root / "cells"
    expected_hashes = {k: v for k, v in json.loads(Path(args.expected_hashes).read_text()).items()
                       if not k.startswith("_")}

    # manifest.jsonl rows per cell
    manifest_rows: Counter = Counter()
    manifest_path = out_root / "manifest.jsonl"
    bad_manifest_lines = 0
    if manifest_path.exists():
        for line in manifest_path.read_text(encoding="utf-8").splitlines():
            try:
                r = json.loads(line)
                manifest_rows[(r["drug"], r["arm"], r["extractor"], r["fold"])] += 1
            except (ValueError, KeyError):
                bad_manifest_lines += 1

    expected = {(d, a, e, f): cells_dir / slug(d) / f"{a}__{e}__fold{f}.json"
                for d in drugs for a in args.arms for e in args.extractors for f in args.folds}
    missing, empty, unparseable, mismatched, no_hash, bad_hash, no_manifest_rows = [], [], [], [], [], [], []
    dup_manifest_rows = []
    designs, seeds, commits = Counter(), Counter(), Counter()
    present = 0
    ok_cells = set()                     # present AND readable
    for k, p in expected.items():
        if not p.exists():
            missing.append(k)
            continue
        present += 1
        if p.stat().st_size == 0:
            empty.append(p)
            continue
        try:
            c = json.loads(p.read_text(encoding="utf-8"))
        except (ValueError, OSError):
            unparseable.append(p)
            continue
        ok_cells.add(k)
        key, man = c.get("key", {}), c.get("manifest", {})
        if (key.get("drug"), key.get("arm"), key.get("extractor"), key.get("fold")) != k:
            mismatched.append(p)
        designs[man.get("design_hash")] += 1
        seeds[key.get("n_seeds")] += 1
        commits[man.get("git_commit")] += 1
        hashes = man.get("input_sha256")
        if not hashes:
            no_hash.append(p)
        elif hashes != expected_hashes:
            diff = sorted({h for h in set(hashes) | set(expected_hashes) if hashes.get(h) != expected_hashes.get(h)})
            bad_hash.append(f"{p.name} ({p.parent.name}): differs on {diff[:3]}{'...' if len(diff) > 3 else ''}")
        n_rows = manifest_rows.get(k, 0)
        if n_rows == 0:
            no_manifest_rows.append(f"{k[0]} {k[1]} {k[2]} fold{k[3]}")
        elif key.get("n_seeds") is not None and n_rows != key["n_seeds"]:
            dup_manifest_rows.append(f"{k[0]} {k[1]} {k[2]} fold{k[3]}: {n_rows} rows, n_seeds={key['n_seeds']}")

    # per-extractor completeness
    per_drug_needed = len(args.arms) * len(args.folds)
    complete = {e: [] for e in args.extractors}
    incomplete = {e: [] for e in args.extractors}
    for e in args.extractors:
        for d in drugs:
            n_ok = sum(1 for a in args.arms for f in args.folds if (d, a, e, f) in ok_cells)
            (complete if n_ok == per_drug_needed else incomplete)[e].append((d, n_ok))

    expected_paths = set(expected.values())
    on_disk = sorted(cells_dir.rglob("*.json")) if cells_dir.exists() else []
    stray_tmp = sorted(cells_dir.rglob("*.tmp")) if cells_dir.exists() else []
    unexpected = [p for p in on_disk if p not in expected_paths]

    def show(title, items):
        print(f"{title}: {len(items)}")
        for x in items[: args.max_list]:
            print(f"    {x}")
        if len(items) > args.max_list:
            print(f"    ... {len(items) - args.max_list} more")

    print(f"out-root={args.out_root}  drugs={len(drugs)} arms={args.arms} extractors={args.extractors} "
          f"folds={args.folds}")
    print(f"expected={len(expected)} present={present} missing={len(missing)}")
    show("missing cells", [f"{d} {a} {e} fold{f}" for d, a, e, f in missing])
    show("empty files (0 bytes)", empty)
    show("unparseable JSON", unparseable)
    show("stray .tmp files", stray_tmp)
    show("key does not match path", mismatched)
    show("cells on disk outside the expected set (other drugs/arms/etc.)", unexpected)
    show("missing/empty manifest.input_sha256", no_hash)
    show(f"input_sha256 differs from {Path(args.expected_hashes).name}", bad_hash)
    show("cells with NO rows in manifest.jsonl", no_manifest_rows)
    show("WARNING (not an error): manifest row count != n_seeds (e.g. a corrupt cell re-run appends again)",
         dup_manifest_rows)
    if bad_manifest_lines:
        print(f"unparseable manifest.jsonl lines: {bad_manifest_lines}")
    print(f"design_hash values: {dict(designs)}")
    print(f"n_seeds values:     {dict(seeds)}")
    print(f"git_commit values:  {dict(commits)}")
    for e in args.extractors:
        print(f"[{e}] complete drugs: {len(complete[e])}/{len(drugs)}   incomplete drugs: {len(incomplete[e])}")
        if complete[e]:
            names = [d for d, _ in complete[e]]
            print(f"    complete: {', '.join(names[: args.max_list])}{' ...' if len(names) > args.max_list else ''}")
        for d, n_ok in incomplete[e][: args.max_list]:
            print(f"    incomplete: {d} ({n_ok}/{per_drug_needed} cells)")
        if len(incomplete[e]) > args.max_list:
            print(f"    ... {len(incomplete[e]) - args.max_list} more incomplete")

    integrity = bool(empty or unparseable or stray_tmp or mismatched or no_hash or bad_hash or no_manifest_rows
                     or bad_manifest_lines or len(designs) > 1 or len(seeds) > 1
                     or (args.n_seeds is not None and set(seeds) - {args.n_seeds}))
    incomplete_any = bool(missing or any(incomplete[e] for e in args.extractors))
    problems = integrity or (incomplete_any and not args.allow_incomplete)
    print(f"RESULT: {'PROBLEMS FOUND' if problems else 'OK'}; integrity={'BAD' if integrity else 'ok'}; "
          f"{'INCOMPLETE' if incomplete_any else 'complete'}")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
