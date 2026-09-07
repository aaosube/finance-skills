"""Execution-quality evaluator for Market Gravity strategies.

This layer is downstream from hypothesis/model selection. It evaluates only
precomputed causal positions against forward returns; it never generates signals.
DSR is applied only to return-based strategy evaluation, not to classifier scores.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from math import sqrt
from statistics import NormalDist
from typing import Iterable

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class ExecutionConfig:
    position_col: str = "position"
    forward_return_col: str = "forward_return"
    regime_col: str | None = None
    periods_per_year: float = 252.0
    transaction_cost_bps: float = 0.0
    slippage_bps: float = 0.0
    dsr_confidence: float = 0.95

    def validate(self) -> None:
        if self.periods_per_year <= 0:
            raise ValueError("periods_per_year must be > 0")
        if self.transaction_cost_bps < 0 or self.slippage_bps < 0:
            raise ValueError("costs/slippage cannot be negative")
        if not (0.5 < self.dsr_confidence < 1):
            raise ValueError("dsr_confidence must be in (0.5, 1)")


def _clean_frame(df: pd.DataFrame, config: ExecutionConfig) -> pd.DataFrame:
    config.validate()
    required = {config.position_col, config.forward_return_col}
    if config.regime_col:
        required.add(config.regime_col)
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"missing required execution columns: {sorted(missing)}")
    if not isinstance(df.index, pd.DatetimeIndex):
        raise TypeError("execution dataframe must use DatetimeIndex")
    if not df.index.is_monotonic_increasing or df.index.has_duplicates:
        raise ValueError("execution dataframe must be unique and chronological")

    cols = [config.position_col, config.forward_return_col]
    if config.regime_col:
        cols.append(config.regime_col)
    out = df.loc[:, cols].copy()
    out[config.position_col] = pd.to_numeric(out[config.position_col], errors="coerce")
    out[config.forward_return_col] = pd.to_numeric(out[config.forward_return_col], errors="coerce")
    out = out.replace([np.inf, -np.inf], np.nan).dropna(subset=[config.position_col, config.forward_return_col])
    if out.empty:
        raise ValueError("no complete execution rows")
    if (out[config.position_col].abs() > 1.0 + 1e-12).any():
        raise ValueError("position must be normalized to [-1, 1]")
    return out


def apply_costs(df: pd.DataFrame, config: ExecutionConfig) -> pd.DataFrame:
    """Apply one-way turnover costs to causal positions and forward returns.

    position[t] is assumed known at decision time t and forward_return[t] is the
    subsequent realized return label. The evaluator never shifts labels itself.
    """
    out = _clean_frame(df, config)
    pos = out[config.position_col].astype(float)
    fwd = out[config.forward_return_col].astype(float)
    prev = pos.shift(1).fillna(0.0)
    turnover = (pos - prev).abs()
    cost_rate = (config.transaction_cost_bps + config.slippage_bps) / 10000.0

    out["gross_strategy_return"] = pos * fwd
    out["turnover"] = turnover
    out["execution_cost"] = turnover * cost_rate
    out["net_strategy_return"] = out["gross_strategy_return"] - out["execution_cost"]
    out["entry_event"] = (((prev == 0) & (pos != 0)) | ((prev * pos) < 0)).astype(int)
    out["active"] = (pos != 0).astype(int)
    return out


def _moments(x: np.ndarray) -> tuple[float, float]:
    if x.size < 3:
        return 0.0, 3.0
    centered = x - x.mean()
    m2 = float(np.mean(centered ** 2))
    if m2 <= 0:
        return 0.0, 3.0
    skew = float(np.mean(centered ** 3) / (m2 ** 1.5))
    kurtosis = float(np.mean(centered ** 4) / (m2 ** 2))
    return skew, kurtosis


def _max_drawdown(returns: pd.Series) -> float:
    equity = (1.0 + returns.astype(float)).cumprod()
    peak = equity.cummax()
    dd = equity / peak - 1.0
    return float(dd.min())


def _profit_factor(returns: pd.Series) -> float:
    pos = float(returns[returns > 0].sum())
    neg = float(-returns[returns < 0].sum())
    if neg == 0:
        return float("inf") if pos > 0 else 0.0
    return pos / neg


def deflated_sharpe_ratio(
    returns: Iterable[float],
    *,
    trial_sharpes_annualized: Iterable[float],
    periods_per_year: float,
) -> dict[str, float | int | bool]:
    """Bailey/Lopez-de-Prado style DSR diagnostic.

    trial_sharpes_annualized must contain the actually tested strategy-family
    Sharpes (or an explicitly justified effective-independent set), not an
    arbitrary placeholder such as N=100.
    """
    x = np.asarray(list(returns), dtype=float)
    x = x[np.isfinite(x)]
    trials = np.asarray(list(trial_sharpes_annualized), dtype=float)
    trials = trials[np.isfinite(trials)]
    if x.size < 3:
        raise ValueError("at least 3 return observations are required for DSR")
    if trials.size < 1:
        raise ValueError("at least one actually tested trial Sharpe is required")
    std = float(np.std(x, ddof=1))
    if std <= 0:
        return {
            "n_obs": int(x.size),
            "n_trials": int(trials.size),
            "observed_sharpe_annualized": 0.0,
            "deflated_benchmark_sharpe_annualized": 0.0,
            "dsr_probability": 0.0,
            "passes_95pct": False,
        }

    sr_period = float(np.mean(x) / std)
    sr_annual = sr_period * sqrt(periods_per_year)

    trial_period = trials / sqrt(periods_per_year)
    sigma_trials = float(np.std(trial_period, ddof=1)) if trials.size > 1 else 0.0
    euler_gamma = 0.5772156649015329
    normal = NormalDist()
    n_trials = int(trials.size)
    if n_trials <= 1 or sigma_trials == 0:
        benchmark_period = 0.0
    else:
        z1 = normal.inv_cdf(1.0 - 1.0 / n_trials)
        z2 = normal.inv_cdf(1.0 - 1.0 / (n_trials * np.e))
        benchmark_period = sigma_trials * ((1.0 - euler_gamma) * z1 + euler_gamma * z2)

    skew, kurtosis = _moments(x)
    denom_sq = 1.0 - skew * sr_period + ((kurtosis - 1.0) / 4.0) * (sr_period ** 2)
    denom_sq = max(denom_sq, 1e-12)
    z = (sr_period - benchmark_period) * sqrt(x.size - 1.0) / sqrt(denom_sq)
    probability = float(normal.cdf(z))

    return {
        "n_obs": int(x.size),
        "n_trials": n_trials,
        "observed_sharpe_annualized": sr_annual,
        "deflated_benchmark_sharpe_annualized": benchmark_period * sqrt(periods_per_year),
        "return_skew": skew,
        "return_kurtosis": kurtosis,
        "dsr_probability": probability,
        "passes_95pct": bool(probability >= 0.95),
    }


def _core_metrics(applied: pd.DataFrame, config: ExecutionConfig) -> dict[str, float | int]:
    r = applied["net_strategy_return"].astype(float)
    gross = applied["gross_strategy_return"].astype(float)
    std = float(r.std(ddof=1)) if len(r) > 1 else 0.0
    sharpe = float(r.mean() / std * sqrt(config.periods_per_year)) if std > 0 else 0.0
    active = applied.loc[applied["active"] == 1, "net_strategy_return"]
    years = len(applied) / config.periods_per_year
    entries = int(applied["entry_event"].sum())

    return {
        "observations": int(len(applied)),
        "gross_total_return_arithmetic": float(gross.sum()),
        "net_total_return_arithmetic": float(r.sum()),
        "cost_drag_arithmetic": float((gross - r).sum()),
        "sharpe_annualized": sharpe,
        "max_drawdown": _max_drawdown(r),
        "profit_factor": _profit_factor(r),
        "active_period_win_rate": float((active > 0).mean()) if len(active) else 0.0,
        "exposure_fraction": float(applied["active"].mean()),
        "entries": entries,
        "entries_per_year": float(entries / years) if years > 0 else 0.0,
        "turnover_total": float(applied["turnover"].sum()),
        "turnover_annualized": float(applied["turnover"].sum() / years) if years > 0 else 0.0,
    }


def evaluate_strategy(
    df: pd.DataFrame,
    config: ExecutionConfig,
    *,
    trial_sharpes_annualized: Iterable[float] | None = None,
) -> dict[str, object]:
    """Return cost-aware risk metrics, optional regime breakdown and optional DSR."""
    applied = apply_costs(df, config)
    metrics = _core_metrics(applied, config)

    regimes = {}
    if config.regime_col:
        for regime, sub in applied.groupby(config.regime_col, dropna=False):
            regimes[str(regime)] = _core_metrics(sub, config)

    dsr = None
    if trial_sharpes_annualized is not None:
        dsr = deflated_sharpe_ratio(
            applied["net_strategy_return"].values,
            trial_sharpes_annualized=trial_sharpes_annualized,
            periods_per_year=config.periods_per_year,
        )
        dsr["passes_configured_confidence"] = bool(dsr["dsr_probability"] >= config.dsr_confidence)

    return {
        "execution_config": asdict(config),
        "metrics": metrics,
        "regime_metrics": regimes,
        "deflated_sharpe": dsr,
        "hard_guards": [
            "POSITIONS_MUST_BE_CAUSAL_AND_PRECOMPUTED",
            "FORWARD_RETURNS_ARE_LABELS_NOT_FEATURES",
            "TRANSACTION_COSTS_AND_SLIPPAGE_APPLIED_TO_TURNOVER",
            "DSR_USES_ACTUAL_TESTED_TRIALS_NOT_ARBITRARY_N",
            "DSR_IS_FOR_RETURN_SERIES_NOT_CLASSIFIER_LOGLOSS",
            "UNDERLYING_AND_OPTION_CONTRACT_ECONOMICS_REMAIN_SEPARATE",
        ],
    }
