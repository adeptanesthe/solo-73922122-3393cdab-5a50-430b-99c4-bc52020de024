"""Ridge regression with leave-one-factory-out lambda selection (NumPy).

Loss:  MSE + lambda * ||w||^2 , intercept unpenalised.

Scalable LOFO (up to ``MAX_FACTORIES`` factories):

* Fold summaries are pooled in **linear** work.  One pass merges the K
  factory summaries into the total (K-1 merges); each leave-one-out fold
  then derives its training summary by exact integer subtraction
  (``stats.remove``) instead of re-merging K-1 factories -- 2K-1 summary
  operations in total, independent of how many lambdas are scored.
* Per fold the standardised-feature correlation matrix C is factorised
  **once** (symmetric eigendecomposition).  The three ridge systems
  (C + lam I) w = c for every candidate lambda are then solved by
  matrix-vector products against the cached eigenbasis, so the three
  lambdas together cost at most one matrix factorisation per fold.
* A fold whose speed or corrected pressure is constant drops *that
  feature in that fold only*; the exact-integer covariance numerator
  keeps genuine non-zero variance intact even for the fold missing the
  factory with the most samples.
* When the best and a runner-up lambda are indistinguishable up to
  floating-point rounding, the affected folds are independently
  re-scored with a direct (LU) solve so lambda selection never hinges on
  a rounding accident.

The chosen lambda is refit on all factories.  Everything is computed
from sufficient statistics only -- the service never sees individual
readings.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from .stats import FEATURES, NAMES, TARGET, Stats, mean_and_cov, merge, remove

LAMBDAS = (0.01, 0.1, 1.0)
MIN_FACTORIES = 3
MAX_FACTORIES = 256

# Two max-held-out-MSE scores closer than this relative gap may have been
# reordered by floating-point rounding; trigger an independent re-score.
NEAR_TIE_REL = 1e-12


@dataclass
class RidgeModel:
    """Ridge model on standardised features."""

    mean: list          # pooled training mean of (speed, pressure, power)
    std: dict           # feature index -> population std (kept features only)
    weights: list       # aligned with `kept`
    intercept: float    # unpenalised
    kept: list = field(default_factory=list)  # kept feature indices

    def predict(self, speed: float, pressure: float) -> float:
        x = {0: float(speed), 1: float(pressure)}
        y = self.intercept
        for w, i in zip(self.weights, self.kept):
            y += w * (x[i] - self.mean[i]) / self.std[i]
        return y

    def to_dict(self) -> dict:
        return {
            "mean": list(self.mean),
            "std": {str(k): v for k, v in self.std.items()},
            "weights": list(self.weights),
            "intercept": self.intercept,
            "kept": list(self.kept),
            "features": [NAMES[i] for i in self.kept],
        }

    @staticmethod
    def from_dict(d: dict) -> "RidgeModel":
        return RidgeModel(
            mean=list(d["mean"]),
            std={int(k): float(v) for k, v in d["std"].items()},
            weights=list(d["weights"]),
            intercept=float(d["intercept"]),
            kept=list(d["kept"]),
        )


def _standardise(mean, cov):
    """Kept features, their population std, correlation C and Cov(z, y)."""
    kept = [i for i in FEATURES if cov[i][i] > 0.0]
    std = {i: math.sqrt(cov[i][i]) for i in kept}
    if not kept:
        return kept, std, None, None
    C = np.array(
        [[cov[i][j] / (std[i] * std[j]) for j in kept] for i in kept],
        dtype=np.float64,
    )
    c = np.array([cov[i][TARGET] / std[i] for i in kept], dtype=np.float64)
    return kept, std, C, c


def fit(mean, cov, lam: float) -> RidgeModel:
    """Closed-form ridge on standardised features via a direct LU solve.

    With population covariance C of the standardised features and
    c = Cov(z, y), minimising  (1/N)||y - b - Zw||^2 + lam*||w||^2
    gives  (C + lam*I) w = c  and  b = mean(y)  (intercept unpenalised).
    Zero-variance features are ignored.
    """
    kept, std, C, c = _standardise(mean, cov)
    intercept = mean[TARGET]
    if not kept:
        return RidgeModel(mean=list(mean), std={}, weights=[],
                          intercept=intercept, kept=[])
    A = C + lam * np.eye(len(kept))
    w = np.linalg.solve(A, c)  # independent LU path (also used to re-check)
    return RidgeModel(mean=list(mean), std=std, weights=[float(v) for v in w],
                      intercept=intercept, kept=kept)


@dataclass
class _FoldBasis:
    """One fold's standardisation, with C factorised once for all lambdas."""

    mean: list
    cov: list
    kept: list
    std: dict
    eigvals: np.ndarray | None   # eigenvalues of C (ascending)
    eigvecs: np.ndarray | None   # orthonormal eigenvectors of C
    c_rot: np.ndarray | None     # Q^T c, the rotated covariance with y

    @classmethod
    def build(cls, fold_stats: Stats) -> "_FoldBasis":
        mean, cov = mean_and_cov(fold_stats)
        kept, std, C, c = _standardise(mean, cov)
        if not kept:
            return cls(mean, cov, kept, std, None, None, None)
        # Single symmetric eigendecomposition reused for every candidate
        # lambda: (C + lam I)^-1 c = Q diag(1/(d_j+lam)) Q^T c.
        eigvals, eigvecs = np.linalg.eigh(C)
        return cls(mean, cov, kept, std, eigvals, eigvecs, eigvecs.T @ c)

    def weights(self, lam: float) -> np.ndarray:
        if self.eigvals is None:
            return np.zeros(0)
        w_rot = self.c_rot / (self.eigvals + lam)
        return self.eigvecs @ w_rot

    def model(self, lam: float) -> RidgeModel:
        return RidgeModel(
            mean=list(self.mean), std=dict(self.std),
            weights=[float(v) for v in self.weights(lam)],
            intercept=self.mean[TARGET], kept=list(self.kept),
        )

    def recheck_model(self, lam: float) -> RidgeModel:
        """Independent fit of the same fold via a direct LU solve."""
        return fit(self.mean, self.cov, lam)


def mse_on(st: Stats, m: RidgeModel) -> float:
    """Exact MSE of a model on one factory, from its sufficient statistics.

    With r = intercept - mean(y):   MSE = Var(y) + r^2 + w'E[zz']w
        - 2 w'Cov(z, y) + 2 r w'E[z]
    which only uses exact covariances and means (numerically stable).
    """
    mean, cov = mean_and_cov(st)
    var_y = cov[TARGET][TARGET]
    r = m.intercept - mean[TARGET]
    mse = var_y + r * r
    if m.kept:
        ez = np.array([(mean[i] - m.mean[i]) / m.std[i] for i in m.kept])
        czz = np.array(
            [[cov[i][j] / (m.std[i] * m.std[j]) for j in m.kept] for i in m.kept]
        )
        ezz = czz + np.outer(ez, ez)
        czy = np.array([cov[i][TARGET] / m.std[i] for i in m.kept])
        w = np.array(m.weights)
        mse += float(w @ ezz @ w) - 2.0 * float(w @ czy) + 2.0 * r * float(w @ ez)
    return max(mse, 0.0)


@dataclass
class FoldInfo:
    """Diagnostics for one leave-one-factory-out fold."""

    factory_id: object
    n_held_out: int
    kept: list          # feature indices with non-zero pooled variance
    dropped: list       # feature names constant in this fold's training pool

    def to_dict(self) -> dict:
        return {
            "factory_id": self.factory_id,
            "n_held_out": self.n_held_out,
            "kept": list(self.kept),
            "dropped_features": list(self.dropped),
        }


@dataclass
class TrainingResult:
    lam: float
    factory_errors: list      # LOFO MSE per factory (same order as input)
    lambda_scores: dict       # lambda -> max held-out MSE
    model: RidgeModel
    folds: list = field(default_factory=list)        # FoldInfo per fold
    summary_ops: int = 0       # exact-integer summary combine/remove ops
    legacy_summary_ops: int = 0  # what a per-fold re-merge would have cost
    factorizations: int = 0    # matrix factorisations across all folds
    tie_rechecks: dict | None = None  # populated only when rounding could flip


def train(factories, labels=None) -> TrainingResult:
    """Full scalable pipeline: LOFO lambda selection, then refit on all.

    ``labels`` optionally names the factories (used in fold diagnostics);
    defaults to their input position.
    """
    factories = list(factories)
    k = len(factories)
    if not MIN_FACTORIES <= k <= MAX_FACTORIES:
        raise ValueError(
            f"need between {MIN_FACTORIES} and {MAX_FACTORIES} factories")
    if labels is None:
        labels = list(range(k))
    else:
        labels = list(labels)

    # --- linear-work pooled summaries: K-1 merges + K exact subtractions ---
    total = merge(factories)
    fold_stats = [remove(total, f) for f in factories]
    summary_ops = (k - 1) + k
    legacy_summary_ops = k * (k - 1)  # one re-merge of K-1 stats per fold

    bases = [_FoldBasis.build(st) for st in fold_stats]
    factorizations = sum(1 for b in bases if b.eigvals is not None)

    folds = [
        FoldInfo(
            factory_id=labels[j],
            n_held_out=factories[j].n,
            kept=list(b.kept),
            dropped=[NAMES[i] for i in FEATURES if i not in b.kept],
        )
        for j, b in enumerate(bases)
    ]

    # --- one factorisation per fold serves every candidate lambda ----------
    errors_by_lambda = {}
    scores = {}
    for lam in LAMBDAS:
        errs = [mse_on(factories[j], bases[j].model(lam)) for j in range(k)]
        errors_by_lambda[lam] = errs
        scores[lam] = max(errs)

    # --- independent re-check when rounding could change the winner --------
    best_score = min(scores.values())
    near = [
        lam for lam in LAMBDAS
        if abs(scores[lam] - best_score)
        <= NEAR_TIE_REL * max(1.0, abs(best_score))
    ]
    tie_rechecks = None
    if len(near) > 1:
        rechecked = {}
        for lam in near:
            # direct LU solves (a different algorithm than the eigh path)
            rechecked[lam] = max(
                mse_on(factories[j], bases[j].recheck_model(lam))
                for j in range(k)
            )
        rechecked_best = min(rechecked.values())
        winner = min(
            (lam for lam in near if rechecked[lam] == rechecked_best),
            key=lambda l: LAMBDAS.index(l),
        )
        tie_rechecks = {
            "near_tie_lambdas": list(near),
            "relative_gap": NEAR_TIE_REL,
            "scores_before": {str(l): scores[l] for l in near},
            "scores_after_lu_solve": {str(l): rechecked[l] for l in near},
            "winner": winner,
        }
    else:
        # min() keeps the first (smallest) lambda on exact ties.
        winner = min(LAMBDAS, key=lambda l: scores[l])

    # --- refit on every factory; predictions always use the current model ---
    mean, cov = mean_and_cov(total)
    model = fit(mean, cov, winner)
    return TrainingResult(
        lam=winner,
        factory_errors=errors_by_lambda[winner],
        lambda_scores={lam: scores[lam] for lam in LAMBDAS},
        model=model,
        folds=folds,
        summary_ops=summary_ops,
        legacy_summary_ops=legacy_summary_ops,
        factorizations=factorizations,
        tie_rechecks=tie_rechecks,
    )
