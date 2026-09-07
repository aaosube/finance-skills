"""Deterministic shadow/counterfactual policy replay for research only.

Inspired by the useful idea behind Vibe-Trading's Shadow Account, but this
implementation deliberately removes generated strategy code and broker
execution. Policies are structured data evaluated through a small operator
whitelist; no ``eval``/``exec`` is used.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Mapping

import numpy as np
import pandas as pd


class ShadowPolicyError(ValueError):
    pass


_ALLOWED_OPS = {"gt", "ge", "lt", "le", "eq", "ne", "between", "in"}


@dataclass(frozen=True)
class ShadowPolicy:
    name: str
    long_when: Mapping[str, Any] | None = None
    short_when: Mapping[str, Any] | None = None
    exit_when: Mapping[str, Any] | None = None
    max_hold_bars: int | None = None

    def validate(self) -> None:
        if not self.name.strip():
            raise ShadowPolicyError("policy name cannot be empty")
        if self.long_when is None and self.short_when is None:
            raise ShadowPolicyError("policy must define long_when and/or short_when")
        if self.max_hold_bars is not None and self.max_hold_bars <= 0:
            raise ShadowPolicyError("max_hold_bars must be > 0")
        for expr in (self.long_when, self.short_when, self.exit_when):
            if expr is not None:
                _validate_expr(expr)


@dataclass(frozen=True)
class ShadowTrade:
    side: str
    entry_time: str
    exit_time: str
    entry_price: float
    exit_price: float
    bars_held: int
    gross_pnl_per_unit: float
    cost_per_unit: float
    net_pnl_per_unit: float
    exit_reason: str


def _validate_expr(expr: Mapping[str, Any]) -> None:
    if not isinstance(expr, Mapping) or not expr:
        raise ShadowPolicyError("each expression must be a non-empty mapping")
    combinators = [k for k in ("all", "any", "not") if k in expr]
    leaf = "feature" in expr or "op" in expr
    if combinators and leaf:
        raise ShadowPolicyError("expression cannot mix combinator and leaf fields")
    if len(combinators) > 1:
        raise ShadowPolicyError("expression can contain only one combinator")
    if combinators:
        key = combinators[0]
        child = expr[key]
        if key == "not":
            _validate_expr(child)
        else:
            if not isinstance(child, list) or not child:
                raise ShadowPolicyError(f"{key} requires a non-empty list")
            for item in child:
                _validate_expr(item)
        return

    feature = expr.get("feature")
    op = expr.get("op")
    if not isinstance(feature, str) or not feature:
        raise ShadowPolicyError("leaf expression requires feature")
    if op not in _ALLOWED_OPS:
        raise ShadowPolicyError(f"unsupported operator: {op}")
    if op == "between":
        value = expr.get("value")
        if not isinstance(value, (list, tuple)) or len(value) != 2:
            raise ShadowPolicyError("between requires value=[low, high]")
    elif op == "in":
        if not isinstance(expr.get("value"), (list, tuple, set)):
            raise ShadowPolicyError("in requires a collection value")
    elif "value" not in expr:
        raise ShadowPolicyError("leaf expression requires value")


def referenced_features(expr: Mapping[str, Any] | None) -> set[str]:
    if expr is None:
        return set()
    if "all" in expr:
        return set().union(*(referenced_features(x) for x in expr["all"]))
    if "any" in expr:
        return set().union(*(referenced_features(x) for x in expr["any"]))
    if "not" in expr:
        return referenced_features(expr["not"])
    return {str(expr["feature"])}


def _eval_leaf(value: Any, op: str, target: Any) -> bool:
    if pd.isna(value):
        return False
    if op == "gt": return value > target
    if op == "ge": return value >= target
    if op == "lt": return value < target
    if op == "le": return value <= target
    if op == "eq": return value == target
    if op == "ne": return value != target
    if op == "between":
        low, high = target
        return low <= value <= high
    if op == "in": return value in target
    raise ShadowPolicyError(f"unsupported operator: {op}")


def evaluate_expr(row: pd.Series, expr: Mapping[str, Any]) -> bool:
    if "all" in expr: return all(evaluate_expr(row, x) for x in expr["all"])
    if "any" in expr: return any(evaluate_expr(row, x) for x in expr["any"])
    if "not" in expr: return not evaluate_expr(row, expr["not"])
    feature = str(expr["feature"])
    if feature not in row.index:
        raise ShadowPolicyError(f"policy references missing feature: {feature}")
    return _eval_leaf(row[feature], str(expr["op"]), expr.get("value"))


def run_shadow_policy(frame: pd.DataFrame, policy: ShadowPolicy, *, price_col: str = "spot", cost_bps: float = 0.0) -> dict[str, Any]:
    """Replay a structured policy over a chronological feature frame."""
    policy.validate()
    if cost_bps < 0:
        raise ShadowPolicyError("cost_bps must be >= 0")
    if not isinstance(frame.index, pd.DatetimeIndex):
        raise ShadowPolicyError("frame index must be a DatetimeIndex")
    if not frame.index.is_monotonic_increasing or frame.index.has_duplicates:
        raise ShadowPolicyError("frame must be strictly chronological with unique timestamps")
    if price_col not in frame.columns:
        raise ShadowPolicyError(f"missing price column: {price_col}")

    features = set().union(referenced_features(policy.long_when), referenced_features(policy.short_when), referenced_features(policy.exit_when))
    missing = sorted(features - set(frame.columns))
    if missing:
        raise ShadowPolicyError(f"policy references missing features: {missing}")

    position: dict[str, Any] | None = None
    trades: list[ShadowTrade] = []

    for i, (ts, row) in enumerate(frame.iterrows()):
        price = float(row[price_col])
        exited_this_bar = False
        if not np.isfinite(price) or price <= 0:
            raise ShadowPolicyError(f"non-positive/non-finite price at {ts}")

        if position is not None:
            held = i - position["entry_i"]
            exit_signal = policy.exit_when is not None and evaluate_expr(row, policy.exit_when)
            max_hold = policy.max_hold_bars is not None and held >= policy.max_hold_bars
            if exit_signal or max_hold:
                side = position["side"]
                gross = price - position["entry_price"] if side == "long" else position["entry_price"] - price
                cost = (position["entry_price"] + price) * cost_bps / 10_000.0
                trades.append(ShadowTrade(side, position["entry_ts"].isoformat(), ts.isoformat(), position["entry_price"], price, held, gross, cost, gross - cost, "EXIT_SIGNAL" if exit_signal else "MAX_HOLD"))
                position = None
                exited_this_bar = True

        if position is None and not exited_this_bar:
            long_signal = policy.long_when is not None and evaluate_expr(row, policy.long_when)
            short_signal = policy.short_when is not None and evaluate_expr(row, policy.short_when)
            if long_signal and short_signal:
                raise ShadowPolicyError(f"conflicting long and short signals at {ts}")
            if long_signal or short_signal:
                position = {"side": "long" if long_signal else "short", "entry_i": i, "entry_ts": ts, "entry_price": price}

    if position is not None:
        ts = frame.index[-1]
        price = float(frame.iloc[-1][price_col])
        side = position["side"]
        gross = price - position["entry_price"] if side == "long" else position["entry_price"] - price
        cost = (position["entry_price"] + price) * cost_bps / 10_000.0
        trades.append(ShadowTrade(side, position["entry_ts"].isoformat(), ts.isoformat(), position["entry_price"], price, (len(frame) - 1) - position["entry_i"], gross, cost, gross - cost, "END_OF_SAMPLE"))

    pnl = np.array([t.net_pnl_per_unit for t in trades], dtype=float)
    cumulative = pnl.cumsum() if len(pnl) else np.array([], dtype=float)
    equity = np.r_[0.0, cumulative] if len(cumulative) else np.array([0.0])
    running_peak = np.maximum.accumulate(equity)
    max_drawdown = float(np.min(equity - running_peak))

    return {
        "mode": "RESEARCH_ONLY_COUNTERFACTUAL",
        "policy": asdict(policy),
        "price_assumption": f"fills at {price_col}",
        "cost_bps_round_trip_legs": float(cost_bps),
        "trades": [asdict(t) for t in trades],
        "summary": {"trade_count": len(trades), "net_pnl_per_unit": float(pnl.sum()) if len(pnl) else 0.0, "win_rate": float((pnl > 0).mean()) if len(pnl) else None, "max_drawdown_per_unit": max_drawdown},
        "hard_guards": ["NO_EXEC_OR_EVAL", "NO_BROKER_ORDERS", "NO_STAGE1_MUTATION", "COST_ASSUMPTION_EXPLICIT"],
    }
