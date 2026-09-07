"""Robust multivariate structural-mass diagnostics for Market Gravity.

This implements the canonical F6 idea without inventing weights:

    z_j = (x_j - median_j) / (1.4826 * MAD_j + eps)
    M_struct = sqrt(z.T @ Sigma^{-1} @ z)

`Sigma` is estimated from chronological training data with Ledoit-Wolf shrinkage.
Distance/reachability is deliberately excluded so it is not double-counted.

This module does not turn structural mass into a probability. Fitted parameters
must be frozen on training data and reused unchanged on calibration/test/live rows.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np
import pandas as pd
from sklearn.covariance import LedoitWolf


DEFAULT_FEATURES: tuple[str, ...] = ("gex", "dex", "vex", "chex", "oi")


@dataclass(frozen=True)
class StructuralMassModel:
    feature_names: tuple[str, ...]
    medians: tuple[float, ...]
    scales: tuple[float, ...]
    covariance: tuple[tuple[float, ...], ...]
    precision: tuple[tuple[float, ...], ...]
    shrinkage: float
    condition_number: float
    eigenvalues: tuple[float, ...]
    train_rows: int
    degenerate_features: tuple[str, ...]
    eps: float

    def score(self, row: pd.Series | dict[str, float]) -> dict[str, object]:
        values = np.asarray([float(row[name]) for name in self.feature_names], dtype=float)
        if not np.all(np.isfinite(values)):
            raise ValueError("structural-mass row contains non-finite values")
        med = np.asarray(self.medians, dtype=float)
        scale = np.asarray(self.scales, dtype=float)
        precision = np.asarray(self.precision, dtype=float)
        z = (values - med) / scale
        quadratic = float(z.T @ precision @ z)
        if quadratic < -1e-8:
            raise ValueError("negative Mahalanobis quadratic form indicates numerical failure")
        mass = float(np.sqrt(max(quadratic, 0.0)))
        return {
            "status": "FITTED_STRUCTURAL_MASS",
            "mass": mass,
            "z": {name: float(v) for name, v in zip(self.feature_names, z)},
            "distance_included": False,
            "probability": None,
            "quality_flags": [
                "STRUCTURAL_MASS_IS_NOT_A_PROBABILITY",
                "DISTANCE_REACHABILITY_EXCLUDED_TO_AVOID_DOUBLE_COUNTING",
                *( ["ZERO_MAD_FEATURES_PRESENT"] if self.degenerate_features else [] ),
            ],
        }


def _validate_history(df: pd.DataFrame, feature_names: Sequence[str], min_rows: int) -> pd.DataFrame:
    if min_rows < 2:
        raise ValueError("min_rows must be >= 2")
    names = tuple(feature_names)
    if len(names) < 2:
        raise ValueError("structural mass requires at least two features")
    if len(set(names)) != len(names):
        raise ValueError("feature_names must be unique")
    missing = set(names) - set(df.columns)
    if missing:
        raise ValueError(f"missing structural-mass columns: {sorted(missing)}")
    x = df.loc[:, list(names)].apply(pd.to_numeric, errors="coerce")
    if x.isna().any().any():
        raise ValueError("historical structural-mass matrix contains missing/non-numeric values")
    if len(x) < min_rows:
        raise ValueError(f"need at least {min_rows} chronological rows; received {len(x)}")
    arr = x.to_numpy(dtype=float)
    if not np.all(np.isfinite(arr)):
        raise ValueError("historical structural-mass matrix contains non-finite values")
    return x


def fit_structural_mass(
    history: pd.DataFrame,
    *,
    feature_names: Sequence[str] = DEFAULT_FEATURES,
    min_rows: int = 30,
    eps: float = 1e-9,
) -> StructuralMassModel:
    """Fit robust location/scale and shrinkage covariance on training data only."""
    if eps <= 0 or not np.isfinite(eps):
        raise ValueError("eps must be finite and > 0")
    names = tuple(feature_names)
    x = _validate_history(history, names, min_rows)
    arr = x.to_numpy(dtype=float)

    med = np.median(arr, axis=0)
    mad = np.median(np.abs(arr - med), axis=0)
    degenerate = tuple(name for name, m in zip(names, mad) if m == 0.0)
    scales = 1.4826 * mad + eps
    z = (arr - med) / scales

    estimator = LedoitWolf(assume_centered=False).fit(z)
    covariance = np.asarray(estimator.covariance_, dtype=float)
    precision = np.linalg.pinv(covariance, hermitian=True)
    eigenvalues = np.linalg.eigvalsh(covariance)
    positive = eigenvalues[eigenvalues > eps]
    condition = float(np.max(positive) / np.min(positive)) if positive.size else float("inf")

    if not np.all(np.isfinite(covariance)) or not np.all(np.isfinite(precision)):
        raise ValueError("non-finite covariance/precision after shrinkage fit")

    return StructuralMassModel(
        feature_names=names,
        medians=tuple(float(v) for v in med),
        scales=tuple(float(v) for v in scales),
        covariance=tuple(tuple(float(v) for v in row) for row in covariance),
        precision=tuple(tuple(float(v) for v in row) for row in precision),
        shrinkage=float(estimator.shrinkage_),
        condition_number=condition,
        eigenvalues=tuple(float(v) for v in eigenvalues),
        train_rows=int(len(x)),
        degenerate_features=degenerate,
        eps=float(eps),
    )


def rms_mass_from_z(z_values: Iterable[float]) -> dict[str, object]:
    """Parameter-free fallback for already-normalized layers only.

    This is intentionally *not* a substitute for the fitted covariance form. The
    caller must supply z-scores computed from causally available/frozen robust
    normalization parameters.
    """
    z = np.asarray(list(z_values), dtype=float)
    z = z[np.isfinite(z)]
    if z.size == 0:
        raise ValueError("at least one finite normalized layer is required")
    return {
        "status": "RMS_FALLBACK_EXPLORATORY",
        "mass": float(np.sqrt(np.mean(z ** 2))),
        "layers": int(z.size),
        "probability": None,
        "quality_flags": [
            "NOT_THE_FITTED_CANONICAL_COVARIANCE_FORM",
            "INPUTS_MUST_ALREADY_BE_ROBUSTLY_NORMALIZED",
            "STRUCTURAL_MASS_IS_NOT_A_PROBABILITY",
        ],
    }
