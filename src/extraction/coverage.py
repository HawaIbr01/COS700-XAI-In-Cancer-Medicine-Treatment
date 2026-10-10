"""
Coverage floor expressed as a fixed row count (`min_cover_n`) instead of a fraction.

ga.py / nsga.py / GA pruning all compare a rule's coverage FRACTION
(mask.mean() == count / n_rows) with `min_coverage`. For the per-drug pilot the
floor is a fixed number of cell lines (25), so callers convert it once with
`coverage_floor()` and pass the result wherever `min_coverage` went before.

Exactness: mask.mean() on a boolean array is float(count) / float(n_rows), and
this function returns float(min_cover_n) / float(n_rows) -- the same IEEE
division -- which is monotone in the numerator. So `coverage >= floor` holds
exactly when `count >= min_cover_n`, with no rounding slack (checked for every
count in tests/test_per_drug.py).
"""
from __future__ import annotations


def coverage_floor(min_coverage: float, min_cover_n: int | None, n_rows: int) -> float:
    """Fraction to use as `min_coverage`. If `min_cover_n` is None the legacy
    fractional floor is returned unchanged (existing global runs unaffected)."""
    if min_cover_n is None:
        return min_coverage
    if min_cover_n < 0:
        raise ValueError("min_cover_n must be >= 0")
    if n_rows <= 0:
        raise ValueError("n_rows must be positive")
    return float(min_cover_n) / float(n_rows)
