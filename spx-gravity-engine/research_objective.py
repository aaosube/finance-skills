"""Auditable loss and cost functions for strategy research.

The objective ranks already-evaluated strategy candidates.  It never creates
signals, changes the validation split, or substitutes LLM judgement for measured
out-of-sample results.  Penalty terms are normalized upstream to [0, 1] so their
weights remain explicit and comparable.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from math import isfinite


@dataclass(frozen=True)
class ResearchLossWeights:
    validation_sharpe: float
    drawdown: float
    cost_drag: float
    train_validation_gap: float
    complexity: float
    data_scarcity: float
    overfit: float
    devil_fragility: float

    def validate(self) -> None:
        values = asdict(self)
        if any(not isfinite(v) or v < 0 for v in values.values()):
            raise ValueError("research-loss weights must be finite and non-negative")
        if not any(v > 0 for v in values.values()):
            raise ValueError("at least one research-loss weight must be positive")


@dataclass(frozen=True)
class ResearchLossInputs:
    validation_sharpe: float
    max_drawdown_penalty: float
    cost_drag_penalty: float
    train_validation_sharpe_gap: float
    complexity_penalty: float
    data_scarcity_penalty: float
    overfit_penalty: float
    devil_fragility_penalty: float

    def validate(self) -> None:
        if not isfinite(self.validation_sharpe):
            raise ValueError("validation_sharpe must be finite")
        penalties = {
            k: v for k, v in asdict(self).items() if k != "validation_sharpe"
        }
        if any(not isfinite(v) or not 0.0 <= v <= 1.0 for v in penalties.values()):
            raise ValueError("all research penalties must be normalized to [0, 1]")


def research_loss(inputs: ResearchLossInputs, weights: ResearchLossWeights) -> dict[str, object]:
    """Compute the agreed multi-objective candidate loss.

    L = -lambda_1*SR_val + lambda_2*DD + lambda_3*cost
        + lambda_4*generalization_gap + lambda_5*complexity
        + lambda_6*scarcity + lambda_7*overfit + lambda_8*devil_fragility

    Lower is better.  The function reports every contribution so a candidate
    cannot win through a hidden aggregate.
    """
    inputs.validate()
    weights.validate()
    contributions = {
        "validation_sharpe_reward": -weights.validation_sharpe * inputs.validation_sharpe,
        "drawdown_penalty": weights.drawdown * inputs.max_drawdown_penalty,
        "cost_drag_penalty": weights.cost_drag * inputs.cost_drag_penalty,
        "train_validation_gap_penalty": weights.train_validation_gap * inputs.train_validation_sharpe_gap,
        "complexity_penalty": weights.complexity * inputs.complexity_penalty,
        "data_scarcity_penalty": weights.data_scarcity * inputs.data_scarcity_penalty,
        "overfit_penalty": weights.overfit * inputs.overfit_penalty,
        "devil_fragility_penalty": weights.devil_fragility * inputs.devil_fragility_penalty,
    }
    return {
        "loss": float(sum(contributions.values())),
        "contributions": contributions,
        "inputs": asdict(inputs),
        "weights": asdict(weights),
        "selection_rule": "LOWER_IS_BETTER",
        "normalization_contract": "PENALTIES_MUST_BE_UPSTREAM_NORMALIZED_TO_0_1",
    }


@dataclass(frozen=True)
class TradeCostInputs:
    notional: float
    spread_bps_round_trip: float = 0.0
    slippage_bps_round_trip: float = 0.0
    market_impact_bps_round_trip: float = 0.0
    borrow_bps_holding_period: float = 0.0
    other_bps_round_trip: float = 0.0
    fixed_round_trip_cash: float = 0.0
    option_fees_cash: float = 0.0
    stress_multiplier: float = 1.0

    def validate(self) -> None:
        values = asdict(self)
        if any(not isfinite(v) for v in values.values()):
            raise ValueError("trade-cost inputs must be finite")
        if self.notional <= 0:
            raise ValueError("notional must be positive")
        if self.stress_multiplier < 1.0:
            raise ValueError("stress_multiplier must be >= 1")
        nonnegative = {k: v for k, v in values.items() if k not in {"notional", "stress_multiplier"}}
        if any(v < 0 for v in nonnegative.values()):
            raise ValueError("trade-cost components cannot be negative")


def trade_cost(inputs: TradeCostInputs) -> dict[str, float | str]:
    """Return transparent cash and rate costs for one completed trade."""
    inputs.validate()
    variable_bps = (
        inputs.spread_bps_round_trip
        + inputs.slippage_bps_round_trip
        + inputs.market_impact_bps_round_trip
        + inputs.borrow_bps_holding_period
        + inputs.other_bps_round_trip
    )
    variable_cash = inputs.notional * variable_bps / 10_000.0
    base_cash = variable_cash + inputs.fixed_round_trip_cash + inputs.option_fees_cash
    stressed_cash = base_cash * inputs.stress_multiplier
    return {
        "base_cost_cash": float(base_cash),
        "stressed_cost_cash": float(stressed_cash),
        "stressed_cost_rate": float(stressed_cash / inputs.notional),
        "variable_cost_bps_round_trip": float(variable_bps),
        "stress_multiplier": float(inputs.stress_multiplier),
        "unit_contract": "BPS_COMPONENTS_ARE_ROUND_TRIP; CASH_COMPONENTS_ARE_ABSOLUTE",
    }
