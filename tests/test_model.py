"""Ridge/LOFO pipeline: equivalence with a raw-row reference implementation."""
import math
import random

import numpy as np
import pytest

from app.model import (LAMBDAS, MAX_FACTORIES, NEAR_TIE_REL, FoldInfo,
                       _FoldBasis, fit, mse_on, train)
from app.stats import Stats, apply_correction, mean_and_cov, merge, remove


def make_factory_rows(n, speed_mu, press_mu, coef, intercept, noise, seed,
                      press_shift=0):
    """Simulate raw integer readings; pressure stored raw, true model uses
    pressure + press_shift."""
    rng = random.Random(seed)
    rows = []
    for _ in range(n):
        s = int(round(rng.gauss(speed_mu, 25)))
        p = int(round(rng.gauss(press_mu, 12)))
        y = coef[0] * s + coef[1] * (p + press_shift) + intercept + rng.gauss(0, noise)
        rows.append((s, p, int(round(y))))
    return rows


def stats_from_rows(rows):
    n = len(rows)
    sums = [sum(r[i] for r in rows) for i in range(3)]
    outer = [[sum(r[i] * r[j] for r in rows) for j in range(3)] for i in range(3)]
    return Stats.build(n, sums, outer)


# ---------------------------------------------------------------- reference
def reference_lofo(factory_rows, lam):
    """LOFO with raw rows: standardise by training mean/pop-std, ridge."""
    errs = []
    for k in range(len(factory_rows)):
        train_rows = [r for j, f in enumerate(factory_rows) if j != k for r in f]
        test_rows = factory_rows[k]
        X = np.array([[r[0], r[1]] for r in train_rows])
        y = np.array([r[2] for r in train_rows], dtype=float)
        mu = X.mean(axis=0)
        sd = X.std(axis=0)  # population std
        kept = [i for i in range(2) if sd[i] > 0]
        if kept:
            Z = (X[:, kept] - mu[kept]) / sd[kept]
            w = np.linalg.solve(Z.T @ Z + lam * len(Z) * np.eye(len(kept)),
                                Z.T @ (y - y.mean()))
            b = y.mean()
        else:
            w, b = np.zeros(0), y.mean()
        Xt = np.array([[r[0], r[1]] for r in test_rows])
        yt = np.array([r[2] for r in test_rows], dtype=float)
        if kept:
            pred = b + ((Xt[:, kept] - mu[kept]) / sd[kept]) @ w
        else:
            pred = np.full(len(yt), b)
        errs.append(float(np.mean((yt - pred) ** 2)))
    return errs


def test_pipeline_matches_raw_row_reference():
    factory_rows = [
        make_factory_rows(300, 1450, 120, (1.8, 0.6), 40, 5, seed=10 + i)
        for i in range(4)
    ]
    stats = [stats_from_rows(r) for r in factory_rows]
    result = train(stats)

    ref_scores = {lam: max(reference_lofo(factory_rows, lam)) for lam in LAMBDAS}
    ref_best = min(LAMBDAS, key=lambda l: ref_scores[l])
    assert result.lam == ref_best
    for lam in LAMBDAS:
        assert result.lambda_scores[lam] == pytest.approx(ref_scores[lam], rel=1e-9)

    ref_errs = reference_lofo(factory_rows, result.lam)
    for got, want in zip(result.factory_errors, ref_errs):
        assert got == pytest.approx(want, rel=1e-9)

    # final model == ridge on all raw rows with the chosen lambda
    all_rows = [r for f in factory_rows for r in f]
    X = np.array([[r[0], r[1]] for r in all_rows])
    y = np.array([r[2] for r in all_rows], dtype=float)
    mu, sd = X.mean(axis=0), X.std(axis=0)
    Z = (X - mu) / sd
    w = np.linalg.solve(Z.T @ Z + result.lam * len(Z) * np.eye(2),
                        Z.T @ (y - y.mean()))
    assert result.model.intercept == pytest.approx(y.mean())
    assert result.model.weights == pytest.approx(list(w), rel=1e-9)
    # prediction matches the raw-row model
    for s, p, _ in all_rows[:20]:
        assert result.model.predict(s, p) == pytest.approx(
            y.mean() + float(((np.array([s, p]) - mu) / sd) @ w), rel=1e-9)


def test_correction_improves_miszeroed_factory():
    # F3's pressure sensor is off by +25 but reports correction 0.
    rows = [make_factory_rows(400, 1450, 120, (1.8, 0.6), 40, 5, seed=100 + i)
            for i in range(2)]
    rows.append(make_factory_rows(400, 1450, 120, (1.8, 0.6), 40, 5, seed=99,
                                  press_shift=25))
    rows.append(make_factory_rows(400, 1450, 120, (1.8, 0.6), 40, 5, seed=103))
    bad = [stats_from_rows(r) for r in rows]
    res_bad = train(bad)
    fixed = [apply_correction(s, 25) if i == 2 else s for i, s in enumerate(bad)]
    res_fixed = train(fixed)
    assert res_fixed.factory_errors[2] < res_bad.factory_errors[2] * 0.5


def test_zero_variance_feature_is_ignored():
    # speed constant everywhere -> only pressure used
    rng_rows = []
    for i in range(3):
        rows = [(1000, 100 + j % 7, 5 * (100 + j % 7) + 3) for j in range(50)]
        rng_rows.append(rows)
    stats = [stats_from_rows(r) for r in rng_rows]
    result = train(stats)
    assert result.model.kept == [1]
    # prediction must not depend on speed
    p1 = result.model.predict(1000, 105)
    p2 = result.model.predict(999999, 105)
    assert p1 == p2


def test_all_features_constant_predicts_mean():
    rows = [[(7, 7, 100 + i)] * 20 for i in range(3)]
    stats = [stats_from_rows(r) for r in rows]
    result = train(stats)
    assert result.model.kept == []
    assert result.model.predict(1, 2) == pytest.approx(101.0)


def test_lambda_tie_prefers_smaller():
    # constant features -> every lambda yields identical scores -> 0.01 wins
    rows = [[(5, 5, 10 * i + j % 3) for j in range(30)] for i in range(3)]
    stats = [stats_from_rows(r) for r in rows]
    result = train(stats)
    assert result.lam == 0.01
    assert len(set(result.lambda_scores.values())) == 1  # really tied


def test_huge_base_values_end_to_end():
    # all factories around 1e6 counts; variance must survive standardisation
    factory_rows = [
        make_factory_rows(200, 1_000_000 + 50 * i, 500_000 + 10 * i,
                          (0.02, 0.01), 1000, 3, seed=1000 + i)
        for i in range(4)
    ]
    stats = [stats_from_rows(r) for r in factory_rows]
    result = train(stats)
    assert result.model.kept == [0, 1]
    assert result.model.std[0] > 0 and result.model.std[1] > 0
    # LOFO errors should be tiny compared to the power scale (~1e5)
    assert max(result.factory_errors) < 1e4


# ------------------------------------------------------------- scaling (256)
def many_factory_stats(k, seed0=3000, n0=120):
    return [
        stats_from_rows(make_factory_rows(
            n0 + (i % 5) * 24, 1000 + (i % 64), 500 + (i % 32),
            (1.8, 0.6), 40, 5, seed=seed0 + i))
        for i in range(k)
    ]


def test_factory_limit_is_256():
    assert MAX_FACTORIES == 256
    stats = many_factory_stats(256)
    result = train(stats)
    assert len(result.folds) == 256
    assert len(result.factory_errors) == 256
    with pytest.raises(ValueError):
        train(many_factory_stats(257))
    with pytest.raises(ValueError):
        train(many_factory_stats(2))


def test_summary_work_is_linear_in_factory_count():
    for k in (3, 8, 64, 256):
        result = train(many_factory_stats(k))
        # K-1 merges into the total + one exact subtraction per fold
        assert result.summary_ops == 2 * k - 1
        assert result.legacy_summary_ops == k * (k - 1)
        # one factorisation per fold at most, reused for all three lambdas
        assert result.factorizations <= k


def test_remove_matches_remerge_exactly():
    stats = many_factory_stats(20)
    total = merge(stats)
    for j, st in enumerate(stats):
        slow = merge(s for i, s in enumerate(stats) if i != j)
        assert remove(total, st) == slow  # bit-for-bit integer equality


def test_fast_path_matches_independent_lu_reference_at_256():
    stats = many_factory_stats(256)
    result = train(stats)

    # fully independent O(K^2) reference with direct LU solves
    scores, ref_errs = {}, {}
    for lam in LAMBDAS:
        errs = []
        for j in range(256):
            fold = merge(s for i, s in enumerate(stats) if i != j)
            mean, cov = mean_and_cov(fold)
            errs.append(mse_on(stats[j], fit(mean, cov, lam)))
        ref_errs[lam] = errs
        scores[lam] = max(errs)

    # lambda selection consistent
    assert result.lam == min(LAMBDAS, key=lambda l: scores[l])
    # candidate max held-out MSEs differ only by float rounding
    for lam in LAMBDAS:
        assert result.lambda_scores[lam] == pytest.approx(
            scores[lam], rel=1e-9, abs=1e-8)
    # winning lambda's per-factory errors: rounding differences only
    for got, want in zip(result.factory_errors, ref_errs[result.lam]):
        assert got == pytest.approx(want, rel=1e-9, abs=1e-8)
    # refit on all factories: same prediction up to rounding
    mean, cov = mean_and_cov(merge(stats))
    ref_model = fit(mean, cov, result.lam)
    assert result.model.kept == ref_model.kept
    assert result.model.intercept == pytest.approx(ref_model.intercept)
    assert np.allclose(result.model.weights, ref_model.weights, rtol=1e-9)
    for s, p in ((1000, 500), (1063, 521), (950, 480)):
        assert result.model.predict(s, p) == pytest.approx(
            ref_model.predict(s, p), rel=1e-9)


def test_one_factorisation_per_fold_serves_three_lambdas():
    # the cached eigenbasis and the direct solve agree on every fold/lambda
    stats = many_factory_stats(12)
    total = merge(stats)
    for j, st in enumerate(stats):
        basis = _FoldBasis.build(remove(total, st))
        for lam in LAMBDAS:
            fast = basis.model(lam)
            direct = basis.recheck_model(lam)
            assert fast.kept == direct.kept
            assert np.allclose(fast.weights, direct.weights, rtol=1e-9, atol=1e-10)


# ------------------------------------------------- per-fold constant features
def test_constant_pressure_dropped_only_in_affected_fold():
    # five factories: only C2 has varying pressure; the rest are constant,
    # all at the same corrected value.
    rows = [
        make_factory_rows(50, 1000 + i, 700, (1.8, 0.6), 40, 4,
                          seed=700 + i) if i == 2
        else [(1000 + i, 700,
               int(round(1.8 * (1000 + i) + 0.6 * 700 + 40 + j % 5)))
              for j in range(50)]
        for i in range(6)
    ]
    stats = [stats_from_rows(r) for r in rows]
    result = train(stats, labels=[f"C{i}" for i in range(6)])
    assert isinstance(result.folds[0], FoldInfo)
    for j, fold in enumerate(result.folds):
        if j == 2:  # hold-out of the sole varying-pressure factory
            assert fold.dropped == ["pressure"]
            assert fold.kept == [0]
        else:
            assert fold.dropped == []
            assert fold.kept == [0, 1]


def test_one_constant_factory_keeps_variance_in_every_fold():
    # the largest factory has constant pressure; every LOFO training pool
    # still contains the other varying-pressure factories.
    rows = []
    for i in range(5):
        if i == 0:
            rng = random.Random(800)
            rows.append([(1000 + rng.randint(-20, 20), 700,
                          2200 + rng.randint(-3, 3)) for _ in range(500)])
        else:
            rows.append(make_factory_rows(80, 1000 + i, 700 + i,
                                          (1.8, 0.6), 40, 4, seed=810 + i))
    stats = [stats_from_rows(r) for r in rows]
    result = train(stats, labels=[f"D{i}" for i in range(5)])
    assert all(fold.dropped == [] for fold in result.folds)
    assert result.model.kept == [0, 1]
    # and the biggest-sample fold still keeps genuine non-zero speed variance
    assert result.folds[0].kept == [0, 1]


def test_correction_can_make_a_fold_pressure_constant():
    # three factories share one raw pressure constant but report different
    # corrections; a fourth varies.  Aligning corrections to 0 makes the
    # fourth factory's hold-out fold drop pressure.
    fixed = [(1000 + j % 9, 5000, 3000 + j % 9) for j in range(60)]
    base = [stats_from_rows(fixed) for _ in range(3)]
    base[1] = apply_correction(base[1], -40)
    base[2] = apply_correction(base[2], 10)
    varying = stats_from_rows(make_factory_rows(
        60, 1000, 5000, (1.8, 0.6), 40, 4, seed=820))
    before = train(base + [varying], labels=["A", "B", "C", "V"])
    assert all(f.dropped == [] for f in before.folds)
    aligned = [apply_correction(stats_from_rows(fixed), 0) for _ in range(3)]
    after = train(aligned + [varying], labels=["A", "B", "C", "V"])
    assert after.folds[3].dropped == ["pressure"]  # fold holding V out
    assert all(f.dropped == [] for j, f in enumerate(after.folds) if j != 3)


# ------------------------------------------------------------- near-tie logic
def test_near_tie_recheck_fires_on_indistinguishable_scores():
    # all-constant features: every lambda is exactly tied
    rows = [[(5, 5, 10 * i + j % 3) for j in range(30)] for i in range(4)]
    result = train([stats_from_rows(r) for r in rows])
    assert result.lam == 0.01
    assert result.tie_rechecks is not None
    assert result.tie_rechecks["winner"] == 0.01
    assert set(result.tie_rechecks["near_tie_lambdas"]) == set(LAMBDAS)
    assert result.tie_rechecks["relative_gap"] == NEAR_TIE_REL


def test_no_recheck_when_scores_are_separated():
    result = train(many_factory_stats(8))
    assert result.tie_rechecks is None
