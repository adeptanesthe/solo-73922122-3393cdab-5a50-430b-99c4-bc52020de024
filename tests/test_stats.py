"""Exact integer statistics: correction math, huge-base variance, validation."""
import random

import pytest

from app.stats import (InvalidStats, Stats, apply_correction, cov_numerator,
                       mean_and_cov, merge, remove)


def raw_stats(rows):
    """Sufficient statistics computed directly from raw integer rows."""
    n = len(rows)
    sums = [sum(r[i] for r in rows) for i in range(3)]
    outer = [[sum(r[i] * r[j] for r in rows) for j in range(3)] for i in range(3)]
    return Stats.build(n, sums, outer)


def sample_rows(n, seed=1, base=(0, 0, 0), spread=(50, 30, 200)):
    rng = random.Random(seed)
    rows = []
    for _ in range(n):
        rows.append(tuple(base[i] + rng.randint(-spread[i], spread[i])
                          for i in range(3)))
    return rows


def test_apply_correction_matches_raw_shift():
    rows = sample_rows(500, seed=7)
    st = raw_stats(rows)
    for c in (0, 1, -3, 25, -1000):
        shifted = [(s, p + c, y) for s, p, y in rows]
        assert apply_correction(st, c) == raw_stats(shifted)


def test_merge_equals_raw_pooling():
    a = raw_stats(sample_rows(120, seed=1))
    b = raw_stats(sample_rows(80, seed=2))
    pooled = merge([a, b])
    assert pooled == raw_stats(sample_rows(120, seed=1) + sample_rows(80, seed=2))


def test_huge_base_value_keeps_nonzero_variance():
    # Readings around 1e12 with variance 2/3: the naive float64 identity
    # E[x^2]-E[x]^2 cancels to <= 0, the exact integer path must not.
    big = 10**12
    rows = [(big, 7, 1), (big + 1, 7, 1), (big + 2, 7, 1)]
    st = raw_stats(rows)
    mean, cov = mean_and_cov(st)
    assert mean[0] == pytest.approx(big + 1.0)
    assert cov[0][0] == pytest.approx(2.0 / 3.0, rel=1e-12)
    assert cov[0][0] > 0
    # sanity: the naive float computation really would lose it
    naive = sum(float(r[0]) ** 2 for r in rows) / 3 - (sum(float(r[0]) for r in rows) / 3) ** 2
    assert naive <= 0.0


def test_covariance_numerator_exact_for_large_values():
    big = 10**15
    rows = [(big + i, 2 * i, i % 3) for i in range(10)]
    st = raw_stats(rows)
    c = cov_numerator(st)
    n = len(rows)
    # exact check against Fraction-free integer arithmetic
    sx = sum(r[0] for r in rows)
    sxx = sum(r[0] * r[0] for r in rows)
    assert c[0][0] == n * sxx - sx * sx
    assert c[0][0] > 0


def test_validate_rejects_inconsistent_outer():
    # sum(x^2) too small for the claimed sum (Cauchy-Schwarz violated)
    with pytest.raises(InvalidStats):
        Stats.build(10, [100, 0, 0],
                    [[5, 0, 0], [0, 0, 0], [0, 0, 0]])


def test_validate_rejects_asymmetric_outer():
    with pytest.raises(InvalidStats):
        Stats.build(3, [1, 2, 3],
                    [[1, 2, 0], [9, 4, 0], [0, 0, 9]])


def test_validate_rejects_bad_n():
    with pytest.raises(InvalidStats):
        Stats.build(0, [0, 0, 0], [[0] * 3] * 3)


def test_remove_is_exact_inverse_of_merge():
    rng = random.Random(11)
    stats = [raw_stats(sample_rows(60 + 10 * i, seed=11 + i))
             for i in range(6)]
    total = merge(stats)
    for i, st in enumerate(stats):
        rest = merge(s for j, s in enumerate(stats) if j != i)
        assert remove(total, st) == rest
    # removing everything one by one leaves the final single factory
    pool = total
    for st in stats[:-1]:
        pool = remove(pool, st)
    assert pool == stats[-1]


def test_remove_rejects_empty_pool():
    only = raw_stats(sample_rows(10, seed=3))
    with pytest.raises(InvalidStats):
        remove(only, only)


def test_remove_keeps_huge_base_variance_exact():
    big = 10**12
    rows_a = [(big + i, 7, i) for i in range(20)]
    rows_b = [(big + 100 + i, 9, 2 * i) for i in range(15)]
    a, b = raw_stats(rows_a), raw_stats(rows_b)
    fold = remove(merge([a, b]), b)
    assert fold == a
    _, cov = mean_and_cov(fold)
    assert cov[0][0] > 0  # variance of the biggest-base factory survives
