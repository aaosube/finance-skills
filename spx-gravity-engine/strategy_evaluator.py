"""Execution-quality evaluator for Market Gravity strategies.

This layer is downstream from hypothesis/model selection. It evaluates only
precomputed causal positions against forward returns; it never generates signals.
DSR is applied only to return-based strategy evaluation, not to classifier scores.
Tail, liquidity/capacity and stability diagnostics are explicit and fail closed
when their required inputs are not configured.
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
    tail_confidence_levels: tuple[float, ...] = (0.95, 0.99)
    stability_window: int | None = None
    spread_bps_col: str | None = None
    dollar_volume_col: str | None = None
    trade_notional_col: str | None = None
    max_spread_bps: float | None = None
    max_participation_rate: float | None = None

    def validate(self) -> None:
        if self.periods_per_year <= 0:
            raise ValueError("periods_per_year must be > 0")
        if self.transaction_cost_bps < 0 or self.slippage_bps < 0:
            raise ValueError("costs/slippage cannot be negative")
        if not (0.5 < self.dsr_confidence < 1):
            raise ValueError("dsr_confidence must be in (0.5, 1)")
        if not self.tail_confidence_levels:
            raise ValueError("tail_confidence_levels cannot be empty")
        if any(not (0.5 < float(q) < 1.0) for q in self.tail_confidence_levels):
            raise ValueError("tail confidence levels must be in (0.5, 1)")
        if self.stability_window is not None and self.stability_window < 5:
            raise ValueError("stability_window must be >= 5 when configured")
        if self.max_spread_bps is not None:
            if self.max_spread_bps < 0:
                raise ValueError("max_spread_bps cannot be negative")
            if not self.spread_bps_col:
                raise ValueError("max_spread_bps requires spread_bps_col")
        if self.max_participation_rate is not None:
            if not (0 < self.max_participation_rate <= 1):
                raise ValueError("max_participation_rate must be in (0, 1]")
            if not self.dollar_volume_col or not self.trade_notional_col:
                raise ValueError("max_participation_rate requires dollar_volume_col and trade_notional_col")


def _clean_frame(df: pd.DataFrame, config: ExecutionConfig) -> pd.DataFrame:
    config.validate()
    required = {config.position_col, config.forward_return_col}
    optional_numeric = []
    if config.regime_col:
        required.add(config.regime_col)
    for col in (config.spread_bps_col, config.dollar_volume_col, config.trade_notional_col):
        if col:
            required.add(col)
            optional_numeric.append(col)
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
    cols.extend(c for c in optional_numeric if c not in cols)
    out = df.loc[:, cols].copy()
    numeric_cols = [config.position_col, config.forward_return_col] + optional_numeric
    for col in numeric_cols:
        out[col] = pd.to_numeric(out[col], errors="coerce")
    out = out.replace([np.inf, -np.inf], np.nan).dropna(subset=[config.position_col, config.forward_return_col])
    if out.empty:
        raise ValueError("no complete execution rows")
    if (out[config.position_col].abs() > 1.0 + 1e-12).any():
        raise ValueError("position must be normalized to [-1, 1]")
    if config.spread_bps_col and (out[config.spread_bps_col].dropna() < 0).any():
        raise ValueError("spread_bps cannot be negative")
    if config.dollar_volume_col and (out[config.dollar_volume_col].dropna() <= 0).any():
        raise ValueError("dollar_volume must be > 0 when supplied")
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


def empirical_tail_risk(returns: Iterable[float], confidence_levels: Iterable[float]) -> dict[str, object]:
    """Distribution-free tail diagnostics on realized net returns."""
    x = np.asarray(list(returns), dtype=float)
    x = x[np.isfinite(x)]
    if x.size == 0:
        raise ValueError("tail diagnostics require at least one return")
    downside = np.minimum(x, 0.0)
    out: dict[str, object] = {
        "method": "EMPIRICAL_NO_NORMALITY_ASSUMPTION",
        "observations": int(x.size),
        "worst_period_return": float(np.min(x)),
        "loss_frequency": float(np.mean(x < 0)),
        "downside_semideviation": float(np.sqrt(np.mean(downside ** 2))),
        "levels": {},
    }
    losses = -x
    for q in confidence_levels:
        qf = float(q)
        var = float(np.quantile(losses, qf))
        tail = losses[losses >= var]
        es = float(np.mean(tail)) if tail.size else var
        out["levels"][f"{qf:.4f}"] = {
            "empirical_loss_var": var,
            "empirical_expected_shortfall": es,
        }
    return out


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
    max_dd = _max_drawdown(r)
    recovery = float("inf") if max_dd <= -1 else (1.0 / (1.0 + max_dd) - 1.0 if max_dd < 0 else 0.0)
    return {
        "observations": int(len(applied)),
        "gross_total_return_arithmetic": float(gross.sum()),
        "net_total_return_arithmetic": float(r.sum()),
        "cost_drag_arithmetic": float((gross - r).sum()),
        "sharpe_annualized": sharpe,
        "max_drawdown": max_dd,
        "recovery_gain_required_after_max_drawdown": recovery,
        "profit_factor": _profit_factor(r),
        "active_period_win_rate": float((active > 0).mean()) if len(active) else 0.0,
        "exposure_fraction": float(applied["active"].mean()),
        "entries": entries,
        "entries_per_year": float(entries / years) if years > 0 else 0.0,
        "turnover_total": float(applied["turnover"].sum()),
        "turnover_annualized": float(applied["turnover"].sum() / years) if years > 0 else 0.0,
    }


def _liquidity_feasibility(applied: pd.DataFrame, config: ExecutionConfig) -> dict[str, object]:
    configured = any((config.spread_bps_col, config.dollar_volume_col, config.trade_notional_col))
    if not configured:
        return {"status": "NOT_EVALUATED", "reason": "no liquidity/capacity columns configured"}

    out: dict[str, object] = {"status": "DIAGNOSTIC_ONLY", "violations": {}}
    limits_configured = False
    if config.spread_bps_col:
        spread = applied[config.spread_bps_col].dropna().astype(float)
        out["spread_bps"] = {
            "observations": int(len(spread)),
            "median": float(spread.median()) if len(spread) else None,
            "p95": float(spread.quantile(0.95)) if len(spread) else None,
            "max": float(spread.max()) if len(spread) else None,
        }
        if config.max_spread_bps is not None:
            limits_configured = True
            mask = applied[config.spread_bps_col].astype(float) > config.max_spread_bps
            out["violations"]["spread"] = int(mask.fillna(False).sum())

    if config.dollar_volume_col and config.trade_notional_col:
        valid = applied[[config.dollar_volume_col, config.trade_notional_col]].dropna().copy()
        participation = valid[config.trade_notional_col].abs() / valid[config.dollar_volume_col]
        out["participation_rate"] = {
            "observations": int(len(participation)),
            "median": float(participation.median()) if len(participation) else None,
            "p95": float(participation.quantile(0.95)) if len(participation) else None,
            "max": float(participation.max()) if len(participation) else None,
        }
        if config.max_participation_rate is not None:
            limits_configured = True
            out["violations"]["participation"] = int((participation > config.max_participation_rate).sum())

    if limits_configured:
        total_violations = sum(int(v) for v in out["violations"].values())
        out["status"] = "PASS" if total_violations == 0 else "FAIL"
    return out


def _stability_diagnostic(applied: pd.DataFrame, config: ExecutionConfig) -> dict[str, object]:
    window = config.stability_window
    if window is None:
        return {"status": "NOT_EVALUATED", "reason": "stability_window not configured"}
    if len(applied) < 2 * window:
        return {
            "status": "INSUFFICIENT_DATA",
            "required_rows": int(2 * window),
            "available_rows": int(len(applied)),
        }
    prior = applied.iloc[-2 * window:-window]
    recent = applied.iloc[-window:]
    prior_metrics = _core_metrics(prior, config)
    recent_metrics = _core_metrics(recent, config)
    return {
        "status": "DIAGNOSTIC_ONLY",
        "window": int(window),
        "prior": prior_metrics,
        "recent": recent_metrics,
        "delta": {
            "mean_net_return": float(recent["net_strategy_return"].mean() - prior["net_strategy_return"].mean()),
            "sharpe_annualized": float(recent_metrics["sharpe_annualized"] - prior_metrics["sharpe_annualized"]),
            "active_period_win_rate": float(recent_metrics["active_period_win_rate"] - prior_metrics["active_period_win_rate"]),
            "max_drawdown": float(recent_metrics["max_drawdown"] - prior_metrics["max_drawdown"]),
        },
        "note": "No automatic retraining or decay threshold is inferred from this diagnostic.",
    }


def evaluate_strategy(
    df: pd.DataFrame,
    config: ExecutionConfig,
    *,
    trial_sharpes_annualized: Iterable[float] | None = None,
) -> dict[str, object]:
    """Return cost, tail, liquidity, stability, regime and optional DSR diagnostics."""
    applied = apply_costs(df, config)
    metrics = _core_metrics(applied, config)
    tail_risk = empirical_tail_risk(applied["net_strategy_return"].values, config.tail_confidence_levels)
    liquidity = _liquidity_feasibility(applied, config)
    stability = _stability_diagnostic(applied, config)

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
        "tail_risk": tail_risk,
        "liquidity_feasibility": liquidity,
        "stability_diagnostic": stability,
        "regime_metrics": regimes,
        "deflated_sharpe": dsr,
        "hard_guards": [
            "POSITIONS_MUST_BE_CAUSAL_AND_PRECOMPUTED",
            "FORWARD_RETURNS_ARE_LABELS_NOT_FEATURES",
            "TRANSACTION_COSTS_AND_SLIPPAGE_APPLIED_TO_TURNOVER",
            "TAIL_RISK_IS_EMPIRICAL_NOT_NORMAL_ASSUMED",
            "LIQUIDITY_LIMITS_ONLY_APPLY_WHEN_EXPLICITLY_CONFIGURED",
            "LIQUIDITY_DIAGNOSTICS_DO_NOT_DOUBLE_COUNT_EXECUTION_COSTS",
            "STABILITY_DIAGNOSTIC_DOES_NOT_AUTO_RETRAIN_OR_AUTO_ADAPT",
            "DSR_USES_ACTUAL_TESTED_TRIALS_NOT_ARBITRARY_N",
            "DSR_IS_FOR_RETURN_SERIES_NOT_CLASSIFIER_LOGLOSS",
            "UNDERLYING_AND_OPTION_CONTRACT_ECONOMICS_REMAIN_SEPARATE",
        ],
    }
