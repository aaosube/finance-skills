"""Implied-volatility surface state and change diagnostics.

This module adopts the *problem* of forward IV-surface dynamics without assuming
that LSTM/Transformers are the solution. It creates auditable surface state
features and future surface-change labels; model choice remains an OOS research
competition.

Feature snapshots must be known at decision time. Future snapshots produced by
``surface_change_outcome`` are labels only and must never be merged back into the
same decision row as features.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class SurfaceConfig:
    expiry_col: str = "expiry"
    strike_col: str = "strike"
    forward_col: str = "forward"
    iv_col: str = "iv"
    as_of_col: str = "as_of"
    min_points_for_slope: int = 3
    min_points_for_curvature: int = 5

    def validate(self) -> None:
        if self.min_points_for_slope < 2:
            raise ValueError("min_points_for_slope must be >= 2")
        if self.min_points_for_curvature < 3:
            raise ValueError("min_points_for_curvature must be >= 3")


def _prepare_snapshot(df: pd.DataFrame, config: SurfaceConfig) -> pd.DataFrame:
    config.validate()
    required = {config.expiry_col, config.strike_col, config.forward_col, config.iv_col, config.as_of_col}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"missing IV-surface columns: {sorted(missing)}")

    out = df.loc[:, list(required)].copy()
    out[config.as_of_col] = pd.to_datetime(out[config.as_of_col], errors="coerce", utc=True)
    out[config.expiry_col] = pd.to_datetime(out[config.expiry_col], errors="coerce", utc=True)
    for col in (config.strike_col, config.forward_col, config.iv_col):
        out[col] = pd.to_numeric(out[col], errors="coerce")
    out = out.dropna()
    if out.empty:
        raise ValueError("IV-surface snapshot has no complete rows")
    if out[config.as_of_col].nunique() != 1:
        raise ValueError("one surface snapshot must contain exactly one as_of timestamp")
    if (out[config.strike_col] <= 0).any() or (out[config.forward_col] <= 0).any():
        raise ValueError("strike and forward must be > 0")
    if ((out[config.iv_col] <= 0) | (out[config.iv_col] > 5)).any():
        raise ValueError("iv must be decimal in (0, 5]; percent-form IV is not accepted")
    if (out[config.expiry_col] <= out[config.as_of_col]).any():
        raise ValueError("expiry must be strictly after as_of for surface-state diagnostics")

    out["log_moneyness"] = np.log(out[config.strike_col] / out[config.forward_col])
    out["time_years"] = (
        (out[config.expiry_col] - out[config.as_of_col]).dt.total_seconds() / (365.0 * 24.0 * 3600.0)
    )
    return out.sort_values([config.expiry_col, config.strike_col]).reset_index(drop=True)


def _slice_factors(group: pd.DataFrame, config: SurfaceConfig) -> dict[str, Any]:
    x = group["log_moneyness"].to_numpy(dtype=float)
    y = group[config.iv_col].to_numpy(dtype=float)
    nearest_idx = int(np.argmin(np.abs(x)))
    factors: dict[str, Any] = {
        "points": int(len(group)),
        "time_years": float(group["time_years"].median()),
        "atm_iv_nearest": float(y[nearest_idx]),
        "atm_abs_log_moneyness": float(abs(x[nearest_idx])),
        "median_iv": float(np.median(y)),
        "min_log_moneyness": float(np.min(x)),
        "max_log_moneyness": float(np.max(x)),
        "skew_slope": None,
        "curvature": None,
    }
    if len(group) >= config.min_points_for_slope and np.ptp(x) > 0:
        factors["skew_slope"] = float(np.polyfit(x, y, deg=1)[0])
    if len(group) >= config.min_points_for_curvature and np.ptp(x) > 0:
        coeff = np.polyfit(x, y, deg=2)
        factors["curvature"] = float(2.0 * coeff[0])
    return factors


def describe_surface_snapshot(df: pd.DataFrame, config: SurfaceConfig = SurfaceConfig()) -> dict[str, Any]:
    snapshot = _prepare_snapshot(df, config)
    as_of = snapshot[config.as_of_col].iloc[0]
    expiries: dict[str, Any] = {}
    for expiry, group in snapshot.groupby(config.expiry_col, sort=True):
        expiries[pd.Timestamp(expiry).isoformat()] = _slice_factors(group, config)

    ordered = list(expiries.items())
    term_slope = None
    if len(ordered) >= 2:
        t = np.array([v["time_years"] for _, v in ordered], dtype=float)
        atm = np.array([v["atm_iv_nearest"] for _, v in ordered], dtype=float)
        if np.ptp(t) > 0:
            term_slope = float(np.polyfit(t, atm, deg=1)[0])

    return {
        "status": "SURFACE_STATE",
        "as_of": pd.Timestamp(as_of).isoformat(),
        "rows": int(len(snapshot)),
        "expiries": expiries,
        "term_structure_atm_slope": term_slope,
        "quality_flags": [
            "ATM_IS_NEAREST_FORWARD_MONEYNESS_POINT",
            "SKEW_AND_CURVATURE_ARE_DESCRIPTIVE_NOT_PROBABILITIES",
            "MODEL_FAMILY_NOT_ASSUMED",
        ],
    }


def surface_change_outcome(
    earlier: pd.DataFrame,
    later: pd.DataFrame,
    config: SurfaceConfig = SurfaceConfig(),
) -> dict[str, Any]:
    """Build future IV-surface *labels* from two chronological snapshots."""
    a = _prepare_snapshot(earlier, config)
    b = _prepare_snapshot(later, config)
    t0 = a[config.as_of_col].iloc[0]
    t1 = b[config.as_of_col].iloc[0]
    if not t1 > t0:
        raise ValueError("later surface snapshot must have a strictly later as_of timestamp")

    left = a[[config.expiry_col, config.strike_col, config.iv_col]].rename(columns={config.iv_col: "iv_earlier"})
    right = b[[config.expiry_col, config.strike_col, config.iv_col]].rename(columns={config.iv_col: "iv_later"})
    matched = left.merge(right, on=[config.expiry_col, config.strike_col], how="inner")
    matched["iv_change"] = matched["iv_later"] - matched["iv_earlier"]

    state_a = describe_surface_snapshot(a, config)
    state_b = describe_surface_snapshot(b, config)
    common_expiries = sorted(set(state_a["expiries"]) & set(state_b["expiries"]))
    slice_changes: dict[str, Any] = {}
    for expiry in common_expiries:
        fa, fb = state_a["expiries"][expiry], state_b["expiries"][expiry]
        slice_changes[expiry] = {
            "atm_iv_change": float(fb["atm_iv_nearest"] - fa["atm_iv_nearest"]),
            "skew_slope_change": None if fa["skew_slope"] is None or fb["skew_slope"] is None else float(fb["skew_slope"] - fa["skew_slope"]),
            "curvature_change": None if fa["curvature"] is None or fb["curvature"] is None else float(fb["curvature"] - fa["curvature"]),
        }

    return {
        "status": "FUTURE_LABEL_ONLY",
        "as_of_earlier": pd.Timestamp(t0).isoformat(),
        "as_of_later": pd.Timestamp(t1).isoformat(),
        "matched_contract_points": int(len(matched)),
        "median_matched_iv_change": float(matched["iv_change"].median()) if len(matched) else None,
        "mean_abs_matched_iv_change": float(matched["iv_change"].abs().mean()) if len(matched) else None,
        "slice_changes": slice_changes,
        "term_structure_atm_slope_change": None if state_a["term_structure_atm_slope"] is None or state_b["term_structure_atm_slope"] is None else float(state_b["term_structure_atm_slope"] - state_a["term_structure_atm_slope"]),
        "hard_guards": [
            "THIS_OUTPUT_IS_A_LABEL_NOT_A_DECISION_TIME_FEATURE",
            "NO_DEEP_LEARNING_MODEL_IS_ASSUMED",
            "MODEL_CHOICE_REQUIRES_CHRONOLOGICAL_OOS_COMPARISON",
            "NO_INTERPOLATION_ACROSS_UNMATCHED_CONTRACTS_IS_SILENTLY_PERFORMED",
        ],
    }
