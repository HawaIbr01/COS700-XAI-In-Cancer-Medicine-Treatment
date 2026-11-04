"""Compare two per-drug cell JSONs. Prints PASS or the first differing key path; exit 0 / 1.

Ignored: wall_clock_s, per-seed seconds, manifest.git_commit / uncommitted_code_files / input_sha256 and the
other run-environment fields. The "traces" key is compared only when BOTH files have it (a pilot cell has
none); --ignore-traces skips it even then.

    python scripts/compare_cells.py new_cell.json pilot_cell.json
"""
from __future__ import annotations

import argparse
import json
import sys

IGN_TOP = {"wall_clock_s"}
IGN_MANIFEST = {"git_commit", "uncommitted_code_files", "input_sha256", "python", "platform", "packages",
                "device", "hostname", "argv", "started_utc"}


def normalise(cell: dict, with_traces: bool) -> dict:
    c = json.loads(json.dumps(cell))
    for k in IGN_TOP:
        c.pop(k, None)
    for k in IGN_MANIFEST:
        c.get("manifest", {}).pop(k, None)
    for r in c.get("seed_rows", []):
        r.pop("seconds", None)
    if not with_traces:
        c.pop("traces", None)
    return c


def first_diff(a, b, path=""):
    if type(a) is not type(b):
        return f"{path or '/'}: type {type(a).__name__} vs {type(b).__name__}"
    if isinstance(a, dict):
        if a.keys() != b.keys():
            return f"{path or '/'}: keys differ {sorted(map(str, set(a) ^ set(b)))}"
        for k in a:
            d = first_diff(a[k], b[k], f"{path}/{k}")
            if d:
                return d
        return None
    if isinstance(a, list):
        if len(a) != len(b):
            return f"{path}: length {len(a)} vs {len(b)}"
        for i, (x, y) in enumerate(zip(a, b)):
            d = first_diff(x, y, f"{path}[{i}]")
            if d:
                return d
        return None
    return None if a == b else f"{path}: {a!r} vs {b!r}"


def compare(path_a: str, path_b: str, ignore_traces: bool = False):
    a, b = (json.load(open(p, encoding="utf-8")) for p in (path_a, path_b))
    both = "traces" in a and "traces" in b and not ignore_traces
    return first_diff(normalise(a, both), normalise(b, both)), both


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("a")
    ap.add_argument("b")
    ap.add_argument("--ignore-traces", action="store_true")
    args = ap.parse_args()
    diff, traces_compared = compare(args.a, args.b, args.ignore_traces)
    note = " (traces compared)" if traces_compared else " (traces not compared)"
    print("PASS" + note if diff is None else "FAIL first difference: " + diff)
    return 0 if diff is None else 1


if __name__ == "__main__":
    sys.exit(main())
