"""Causal macro/news event-response diagnostics for Market Gravity.

This module adopts the useful problem behind Hawkes-style news-to-price research:
measure how quickly market activity decays after a *known* event. It deliberately
does not claim to fit a Hawkes process from coarse bars.

Why: the current research stack has scheduled macro/event timestamps and bar data,
but a true Hawkes model requires event-time point-process data (e.g. trades/quotes,
price-change events) at materially finer resolution plus a validated event corpus.
Using 5-minute bars as though they were tick events would create false precision.

The functions below therefore provide:
- a causal live absorption state using only data available up to ``as_of``;
- an offline exponential-decay diagnostic using a completed post-event window.

Neither output is a trading probability. The offline decay fit is research-only and
must not be merged into a decision row before its ``available_at`` timestamp.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from math import log

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class EventResponseConfig:
    activity_col: str
    pre_bars: int = 12
    post_bars: int = 24
    min_pre_bars: int = 6
    min_decay_points: int = 3

    def validate(self) -> None:
        if not self.activity_col:
            raise ValueError("activity_col is required")
        if self.pre_bars < 1 or self.post_bars < 1:
            raise ValueError("pre_bars and post_bars must be >= 1")
        if self.min_pre_bars < 2 or self.min_pre_bars > self.pre_bars:
            raise ValueError("min_pre_bars must be in [2, pre_bars]")
        if self.min_decay_points < 3:
            raise ValueError("min_decay_points must be >= 3")


def _prepare_bars(bars: pd.DataFrame, config: EventResponseConfig) -> pd.DataFrame:
    config.validate()
    if config.activity_col not in bars.columns:
        raise ValueError(f"missing activity column: {config.activity_col}")
    if not isinstance(bars.index, pd.DatetimeIndex):
        raise TypeError("bars.index must be a DatetimeIndex")
    if bars.index.tz is None:
        raise ValueError("bars.index must be timezone-aware")
    if not bars.index.is_monotonic_increasing or bars.index.has_duplicates:
        raise ValueError("bars must be unique and sorted chronologically")

    out = bars[[config.activity_col]].copy()
    out[config.activity_col] = pd.to_numeric(out[config.activity_col], errors="coerce")
    out = out.replace([np.inf, -np.inf], np.nan).dropna()
    if out.empty:
        raise ValueError("no finite event-activity observations")
    if (out[config.activity_col] < 0).any():
        raise ValueError("activity metric must be non-negative")
    return out


def _utc_timestamp(value: object, name: str) -> pd.Timestamp:
    ts = pd.Timestamp(value)
    if ts.tzinfo is None:
        raise ValueError(f"{name} must be timezone-aware")
    return ts.tz_convert("UTC")


def _baseline(pre: pd.Series, config: EventResponseConfig) -> float:
    if len(pre) < config.min_pre_bars:
        raise ValueError(
            f"insufficient pre-event history: need {config.min_pre_bars}, received {len(pre)}"
        )
    value = float(pre.median())
    if not np.isfinite(value) or value <= 0:
        raise ValueError("pre-event median activity must be finite and > 0")
    return value


def causal_event_absorption_state(
    bars: pd.DataFrame,
    *,
    event_time: object,
    as_of: object,
    config: EventResponseConfig,
) -> dict[str, object]:
    """Return a live, causal post-event absorption diagnostic.

    Only rows with timestamp <= ``as_of`` are used. The result can therefore be
    computed live after an event without future leakage. It is a state diagnostic,
    not a calibrated probability or an automatic NO_TRADE rule.
    """
    frame = _prepare_bars(bars, config)
    event = _utc_timestamp(event_time, "event_time")
    current = _utc_timestamp(as_of, "as_of")
    if current < event:
        raise ValueError("as_of must be >= event_time")

    frame = frame.tz_convert("UTC")
    pre = frame.loc[frame.index < event, config.activity_col].tail(config.pre_bars)
    baseline = _baseline(pre, config)
    post = frame.loc[(frame.index >= event) & (frame.index <= current), config.activity_col]
    if post.empty:
        return {
            "status": "INSUFFICIENT_POST_EVENT_DATA",
            "event_time": event.isoformat(),
            "as_of": current.isoformat(),
            "baseline_activity_median": baseline,
            "post_observations": 0,
            "probability": None,
        }

    ratio = post.astype(float) / baseline
    excess = np.maximum(ratio.to_numpy(dtype=float) - 1.0, 0.0)
    peak_excess = float(np.max(excess))
    current_excess = float(excess[-1])
    if peak_excess <= 0:
        absorbed_fraction = 1.0
    else:
        absorbed_fraction = float(np.clip(1.0 - current_excess / peak_excess, 0.0, 1.0))

    peak_pos = int(np.argmax(ratio.to_numpy(dtype=float)))
    peak_time = pd.Timestamp(ratio.index[peak_pos])
    return {
        "status": "CAUSAL_EVENT_ABSORPTION_DIAGNOSTIC",
        "event_time": event.isoformat(),
        "as_of": current.isoformat(),
        "baseline_activity_median": baseline,
        "post_observations": int(len(post)),
        "current_activity_ratio": float(ratio.iloc[-1]),
        "peak_activity_ratio_so_far": float(ratio.iloc[peak_pos]),
        "peak_time_so_far": peak_time.isoformat(),
        "absorbed_fraction_of_peak_excess_so_far": absorbed_fraction,
        "probability": None,
        "hard_guards": [
            "USES_ONLY_ROWS_AT_OR_BEFORE_AS_OF",
            "NOT_A_TRADING_PROBABILITY",
            "NO_AUTOMATIC_NO_TRADE_THRESHOLD",
            "ACTIVITY_METRIC_DEFINITION_MUST_BE_FROZEN_UPSTREAM",
        ],
    }


def completed_event_decay_diagnostic(
    bars: pd.DataFrame,
    *,
    event_time: object,
    config: EventResponseConfig,
) -> dict[str, object]:
    """Fit an offline exponential decay to post-event excess activity.

    The fit uses the completed ``post_bars`` window and is therefore a research
    label/diagnostic. It must not be available to a decision made before the final
    bar used by the fit.

    Model fitted after the observed peak:

        excess(t) = A * exp(-beta * t)
        half_life = ln(2) / beta

    This is *not* a Hawkes intensity fit and does not identify exogenous versus
    endogenous excitation coefficients.
    """
    frame = _prepare_bars(bars, config).tz_convert("UTC")
    event = _utc_timestamp(event_time, "event_time")
    pre = frame.loc[frame.index < event, config.activity_col].tail(config.pre_bars)
    baseline = _baseline(pre, config)
    post = frame.loc[frame.index >= event, config.activity_col].head(config.post_bars)
    if len(post) < config.min_decay_points:
        return {
            "status": "INSUFFICIENT_POST_EVENT_DATA",
            "event_time": event.isoformat(),
            "post_observations": int(len(post)),
            "required_decay_points": int(config.min_decay_points),
            "probability": None,
        }

    ratio = post.astype(float) / baseline
    excess = np.maximum(ratio.to_numpy(dtype=float) - 1.0, 0.0)
    peak_pos = int(np.argmax(excess))
    peak_time = pd.Timestamp(ratio.index[peak_pos])
    decay_excess = excess[peak_pos:]
    decay_index = ratio.index[peak_pos:]
    positive = decay_excess > 0

    available_at = pd.Timestamp(post.index[-1])
    base = {
        "event_time": event.isoformat(),
        "available_at": available_at.isoformat(),
        "baseline_activity_median": baseline,
        "pre_observations": int(len(pre)),
        "post_observations": int(len(post)),
        "peak_activity_ratio": float(ratio.iloc[peak_pos]),
        "peak_time": peak_time.isoformat(),
        "probability": None,
    }

    if int(np.sum(positive)) < config.min_decay_points:
        return {
            **base,
            "status": "NO_IDENTIFIABLE_POSITIVE_DECAY",
            "positive_decay_points": int(np.sum(positive)),
            "hard_guards": [
                "OFFLINE_RESEARCH_DIAGNOSTIC_ONLY",
                "NOT_A_HAWKES_PROCESS_FIT",
                "DO_NOT_BACKFILL_BEFORE_AVAILABLE_AT",
            ],
        }

    elapsed_minutes = np.asarray(
        [(pd.Timestamp(ts) - peak_time).total_seconds() / 60.0 for ts in decay_index],
        dtype=float,
    )[positive]
    y = np.log(decay_excess[positive])

    # Need distinct elapsed times for a meaningful slope.
    if np.ptp(elapsed_minutes) <= 0:
        return {
            **base,
            "status": "NO_IDENTIFIABLE_POSITIVE_DECAY",
            "positive_decay_points": int(len(y)),
            "hard_guards": [
                "OFFLINE_RESEARCH_DIAGNOSTIC_ONLY",
                "NOT_A_HAWKES_PROCESS_FIT",
                "DO_NOT_BACKFILL_BEFORE_AVAILABLE_AT",
            ],
        }

    slope, intercept = np.polyfit(elapsed_minutes, y, deg=1)
    fitted = intercept + slope * elapsed_minutes
    ss_res = float(np.sum((y - fitted) ** 2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 1.0

    if slope >= 0:
        return {
            **base,
            "status": "NO_DECAYING_EXPONENTIAL_FIT",
            "slope_per_minute": float(slope),
            "fit_r2": float(r2),
            "positive_decay_points": int(len(y)),
            "hard_guards": [
                "OFFLINE_RESEARCH_DIAGNOSTIC_ONLY",
                "NOT_A_HAWKES_PROCESS_FIT",
                "DO_NOT_BACKFILL_BEFORE_AVAILABLE_AT",
            ],
        }

    beta = float(-slope)
    half_life = float(log(2.0) / beta)
    return {
        **base,
        "status": "EXPONENTIAL_EVENT_DECAY_DIAGNOSTIC",
        "beta_per_minute": beta,
        "half_life_minutes": half_life,
        "fit_r2": float(r2),
        "positive_decay_points": int(len(y)),
        "config": asdict(config),
        "hard_guards": [
            "OFFLINE_RESEARCH_DIAGNOSTIC_ONLY",
            "NOT_A_HAWKES_PROCESS_FIT",
            "DO_NOT_BACKFILL_BEFORE_AVAILABLE_AT",
            "HAWKES_REQUIRES_FINER EVENT-TIME DATA AND SEPARATE VALIDATION",
            "NO_AUTOMATIC_NO_TRADE_THRESHOLD",
        ],
    }
