"""Cox-Ross-Rubinstein option-pricing diagnostic for Market Gravity.

SPX/SPXW index options are European-style, while many ETF and single-stock
options are American-style. A European Black-Scholes formula therefore cannot be
silently generalized across the expanded INDEX/ETF/STOCK universe.

This module provides a deterministic binomial-tree diagnostic with continuous
dividend yield and optional early exercise. It is a valuation/contract-risk tool,
not a directional signal and not a replacement for executable market quotes.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import exp, sqrt

import numpy as np


@dataclass(frozen=True)
class CRRInputs:
    spot: float
    strike: float
    time_years: float
    rate: float
    volatility: float
    option_type: str
    exercise_style: str = "american"
    dividend_yield: float = 0.0
    steps: int = 200


def _validate(inp: CRRInputs) -> tuple[str, str]:
    numeric = {
        "spot": inp.spot,
        "strike": inp.strike,
        "time_years": inp.time_years,
        "rate": inp.rate,
        "volatility": inp.volatility,
        "dividend_yield": inp.dividend_yield,
    }
    for name, value in numeric.items():
        if not np.isfinite(float(value)):
            raise ValueError(f"{name} must be finite")
    if inp.spot <= 0 or inp.strike <= 0:
        raise ValueError("spot and strike must be > 0")
    if inp.time_years <= 0:
        raise ValueError("time_years must be > 0")
    if inp.volatility <= 0:
        raise ValueError("volatility must be > 0")
    if not isinstance(inp.steps, int) or inp.steps < 2:
        raise ValueError("steps must be an integer >= 2")
    option_type = str(inp.option_type).lower()
    style = str(inp.exercise_style).lower()
    if option_type not in {"call", "put"}:
        raise ValueError("option_type must be call or put")
    if style not in {"european", "american"}:
        raise ValueError("exercise_style must be european or american")
    return option_type, style


def crr_price(inp: CRRInputs) -> dict[str, object]:
    """Price a European/American option and return tree-based Delta/Gamma."""
    option_type, style = _validate(inp)
    s = float(inp.spot)
    k = float(inp.strike)
    t = float(inp.time_years)
    r = float(inp.rate)
    q = float(inp.dividend_yield)
    sigma = float(inp.volatility)
    n = int(inp.steps)

    dt = t / n
    u = exp(sigma * sqrt(dt))
    d = 1.0 / u
    growth = exp((r - q) * dt)
    denominator = u - d
    if denominator <= 0:
        raise ValueError("invalid CRR up/down factors")
    p = (growth - d) / denominator
    if not 0.0 <= p <= 1.0:
        raise ValueError("CRR risk-neutral probability outside [0,1]; increase steps or inspect inputs")
    disc = exp(-r * dt)

    j = np.arange(n + 1, dtype=float)
    terminal_spot = s * (u ** j) * (d ** (n - j))
    if option_type == "call":
        values = np.maximum(terminal_spot - k, 0.0)
    else:
        values = np.maximum(k - terminal_spot, 0.0)

    layer_2 = None
    layer_1 = None
    early_exercise_nodes = 0

    for i in range(n - 1, -1, -1):
        continuation = disc * (p * values[1:i + 2] + (1.0 - p) * values[:i + 1])
        if style == "american":
            jj = np.arange(i + 1, dtype=float)
            node_spot = s * (u ** jj) * (d ** (i - jj))
            if option_type == "call":
                intrinsic = np.maximum(node_spot - k, 0.0)
            else:
                intrinsic = np.maximum(k - node_spot, 0.0)
            exercise = intrinsic > continuation + 1e-14
            early_exercise_nodes += int(np.sum(exercise))
            values = np.maximum(continuation, intrinsic)
        else:
            values = continuation

        if i == 2:
            layer_2 = values.copy()
        elif i == 1:
            layer_1 = values.copy()

    if layer_1 is None or layer_2 is None:
        raise ValueError("tree did not retain enough layers for Greeks")

    s_up = s * u
    s_down = s * d
    delta = float((layer_1[1] - layer_1[0]) / (s_up - s_down))

    s_uu = s * u * u
    s_ud = s
    s_dd = s * d * d
    delta_up = float((layer_2[2] - layer_2[1]) / (s_uu - s_ud))
    delta_down = float((layer_2[1] - layer_2[0]) / (s_ud - s_dd))
    gamma = float((delta_up - delta_down) / (0.5 * (s_uu - s_dd)))

    return {
        "status": "CRR_NUMERICAL_DIAGNOSTIC",
        "price": float(values[0]),
        "delta": delta,
        "gamma": gamma,
        "risk_neutral_up_probability": float(p),
        "up_factor": float(u),
        "down_factor": float(d),
        "steps": n,
        "option_type": option_type,
        "exercise_style": style,
        "early_exercise_nodes": int(early_exercise_nodes),
        "hard_guards": [
            "VALUATION_DIAGNOSTIC_NOT_DIRECTIONAL_SIGNAL",
            "MARKET_BID_ASK_REMAINS_EXECUTION_TRUTH",
            "VOLATILITY_RATE_DIVIDEND_AND_TIME_ARE_MODEL_INPUTS",
            "AMERICAN_STYLE_REQUIRES_EARLY_EXERCISE_AWARENESS",
        ],
    }
