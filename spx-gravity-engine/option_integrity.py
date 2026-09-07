"""Option-chain integrity diagnostics for Market Gravity.

European put-call parity is used as a data-quality/synchronization check, not as
a directional signal. American-style stock/ETF options do not satisfy the same
exact equality because of early exercise; for them this module uses admissible
no-arbitrage bounds instead of falsely applying European parity.

Pairing/synchronization must occur upstream with point-in-time as-of semantics.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import exp

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class ParityInputs:
    strike: float
    time_years: float
    rate: float
    call_mid: float
    put_mid: float
    spot: float | None = None
    dividend_yield: float | None = None
    forward: float | None = None
    call_bid: float | None = None
    call_ask: float | None = None
    put_bid: float | None = None
    put_ask: float | None = None
    exercise_style: str = "european"


def _finite(name: str, value: float | None, *, allow_none: bool = False) -> float | None:
    if value is None and allow_none:
        return None
    x = float(value)
    if not np.isfinite(x):
        raise ValueError(f"{name} must be finite")
    return x


def _exercise_style(value: str) -> str:
    style = str(value).lower()
    if style not in {"european", "american"}:
        raise ValueError("exercise_style must be european or american")
    return style


def theoretical_call_minus_put(inp: ParityInputs) -> float:
    """Exact European call-minus-put parity value."""
    if _exercise_style(inp.exercise_style) != "european":
        raise ValueError("exact call-put parity equality applies here only to European-style options")
    k = _finite("strike", inp.strike)
    t = _finite("time_years", inp.time_years)
    r = _finite("rate", inp.rate)
    if k <= 0 or t < 0:
        raise ValueError("strike must be > 0 and time_years must be >= 0")

    if inp.forward is not None:
        f = _finite("forward", inp.forward)
        if f <= 0:
            raise ValueError("forward must be > 0")
        return float(exp(-r * t) * (f - k))

    s = _finite("spot", inp.spot)
    q = _finite("dividend_yield", inp.dividend_yield if inp.dividend_yield is not None else 0.0)
    if s <= 0:
        raise ValueError("spot must be > 0 when forward is not supplied")
    return float(s * exp(-q * t) - k * exp(-r * t))


def american_call_minus_put_bounds(inp: ParityInputs) -> tuple[float, float]:
    """American-style no-arbitrage bounds for C_A - P_A.

    With continuous dividend yield q, a common admissible interval is:

        S exp(-qT) - K <= C_A - P_A <= S - K exp(-rT)

    We use it only as an integrity bound. Market quotes remain execution truth.
    """
    if _exercise_style(inp.exercise_style) != "american":
        raise ValueError("American bounds require exercise_style='american'")
    k = _finite("strike", inp.strike)
    t = _finite("time_years", inp.time_years)
    r = _finite("rate", inp.rate)
    s = _finite("spot", inp.spot)
    q = _finite("dividend_yield", inp.dividend_yield if inp.dividend_yield is not None else 0.0)
    if k <= 0 or s <= 0 or t < 0:
        raise ValueError("strike/spot must be > 0 and time_years must be >= 0")
    lower = float(s * exp(-q * t) - k)
    upper = float(s - k * exp(-r * t))
    if lower > upper + 1e-12:
        raise ValueError("American parity bounds inverted for supplied rate/dividend inputs")
    return lower, upper


def _validated_quotes(inp: ParityInputs) -> tuple[float, float, float, float] | None:
    quote_fields = (inp.call_bid, inp.call_ask, inp.put_bid, inp.put_ask)
    if not all(v is not None for v in quote_fields):
        return None
    cb = _finite("call_bid", inp.call_bid)
    ca = _finite("call_ask", inp.call_ask)
    pb = _finite("put_bid", inp.put_bid)
    pa = _finite("put_ask", inp.put_ask)
    if min(cb, ca, pb, pa) < 0 or cb > ca or pb > pa:
        raise ValueError("invalid bid/ask quote ordering")
    return float(cb), float(ca), float(pb), float(pa)


def audit_put_call_parity(inp: ParityInputs, *, max_abs_mid_residual: float | None = None) -> dict[str, object]:
    call_mid = _finite("call_mid", inp.call_mid)
    put_mid = _finite("put_mid", inp.put_mid)
    if call_mid < 0 or put_mid < 0:
        raise ValueError("option prices cannot be negative")
    observed = float(call_mid - put_mid)
    style = _exercise_style(inp.exercise_style)
    quotes = _validated_quotes(inp)

    if style == "american":
        if max_abs_mid_residual is not None:
            raise ValueError("American-style options use bounds; exact mid residual threshold is not applicable")
        lower, upper = american_call_minus_put_bounds(inp)
        mid_inside = bool(lower - 1e-12 <= observed <= upper + 1e-12)
        out: dict[str, object] = {
            "status": "DIAGNOSTIC_ONLY",
            "exercise_style": "american",
            "method": "AMERICAN_PARITY_BOUNDS",
            "observed_call_minus_put_mid": observed,
            "theoretical_call_minus_put_bounds": {"lower": lower, "upper": upper},
            "mid_inside_bounds": mid_inside,
            "quality_flags": [
                "AMERICAN_STYLE_USES_BOUNDS_NOT_EUROPEAN_EQUALITY",
                "DIVIDEND_YIELD_IS_MODEL_INPUT",
            ],
        }
        if not mid_inside:
            out["quality_flags"].append("AMERICAN_PARITY_MID_OUTSIDE_THEORETICAL_BOUNDS")

        if quotes is not None:
            cb, ca, pb, pa = quotes
            executable_lower = float(cb - pa)
            executable_upper = float(ca - pb)
            overlap = bool(max(executable_lower, lower) <= min(executable_upper, upper) + 1e-12)
            out["executable_call_minus_put_interval"] = {"lower": executable_lower, "upper": executable_upper}
            out["executable_interval_overlaps_theoretical_bounds"] = overlap
            out["status"] = "PASS" if overlap else "FAIL"
            if not overlap:
                out["quality_flags"].append("AMERICAN_PARITY_EXECUTABLE_INTERVAL_VIOLATION")
        return out

    theo = theoretical_call_minus_put(inp)
    residual = observed - theo
    out = {
        "status": "DIAGNOSTIC_ONLY",
        "exercise_style": "european",
        "theoretical_call_minus_put": theo,
        "observed_call_minus_put_mid": observed,
        "mid_residual": residual,
        "abs_mid_residual": abs(residual),
        "method": "FORWARD_PARITY" if inp.forward is not None else "SPOT_DIVIDEND_PARITY",
        "quality_flags": [],
    }

    if quotes is not None:
        cb, ca, pb, pa = quotes
        lower = float(cb - pa)
        upper = float(ca - pb)
        inside = bool(lower - 1e-12 <= theo <= upper + 1e-12)
        out["executable_parity_interval"] = {"lower": lower, "upper": upper}
        out["inside_executable_interval"] = inside
        out["status"] = "PASS" if inside else "FAIL"
        if not inside:
            out["quality_flags"].append("PUT_CALL_PARITY_EXECUTABLE_INTERVAL_VIOLATION")

    if max_abs_mid_residual is not None:
        threshold = _finite("max_abs_mid_residual", max_abs_mid_residual)
        if threshold < 0:
            raise ValueError("max_abs_mid_residual cannot be negative")
        out["configured_mid_residual_limit"] = threshold
        threshold_pass = abs(residual) <= threshold
        out["mid_residual_limit_pass"] = bool(threshold_pass)
        if out["status"] == "DIAGNOSTIC_ONLY":
            out["status"] = "PASS" if threshold_pass else "FAIL"
        elif not threshold_pass:
            out["status"] = "FAIL"
        if not threshold_pass:
            out["quality_flags"].append("PUT_CALL_PARITY_MID_RESIDUAL_LIMIT_VIOLATION")

    if inp.forward is None:
        out["quality_flags"].append("DIVIDEND_YIELD_IS_MODEL_INPUT")
    return out


def audit_parity_frame(
    df: pd.DataFrame,
    *,
    strike_col: str = "strike",
    time_years_col: str = "time_years",
    rate_col: str = "rate",
    call_mid_col: str = "call_mid",
    put_mid_col: str = "put_mid",
    spot_col: str | None = "spot",
    dividend_yield_col: str | None = "dividend_yield",
    forward_col: str | None = None,
    call_bid_col: str | None = None,
    call_ask_col: str | None = None,
    put_bid_col: str | None = None,
    put_ask_col: str | None = None,
    exercise_style_col: str | None = None,
    default_exercise_style: str = "european",
    max_abs_mid_residual: float | None = None,
) -> pd.DataFrame:
    """Audit already synchronized call/put rows.

    Pairing/synchronization must happen upstream with as-of semantics. This
    function deliberately does not join calls and puts by itself because doing so
    can create look-ahead or mismatched-quote errors.
    """
    required = {strike_col, time_years_col, rate_col, call_mid_col, put_mid_col}
    if forward_col:
        required.add(forward_col)
    elif spot_col:
        required.add(spot_col)
    if exercise_style_col:
        required.add(exercise_style_col)
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"missing parity columns: {sorted(missing)}")

    records = []
    for idx, row in df.iterrows():
        style = row[exercise_style_col] if exercise_style_col else default_exercise_style
        inp = ParityInputs(
            strike=row[strike_col],
            time_years=row[time_years_col],
            rate=row[rate_col],
            call_mid=row[call_mid_col],
            put_mid=row[put_mid_col],
            spot=row[spot_col] if spot_col and spot_col in df.columns else None,
            dividend_yield=row[dividend_yield_col] if dividend_yield_col and dividend_yield_col in df.columns else None,
            forward=row[forward_col] if forward_col and forward_col in df.columns else None,
            call_bid=row[call_bid_col] if call_bid_col and call_bid_col in df.columns else None,
            call_ask=row[call_ask_col] if call_ask_col and call_ask_col in df.columns else None,
            put_bid=row[put_bid_col] if put_bid_col and put_bid_col in df.columns else None,
            put_ask=row[put_ask_col] if put_ask_col and put_ask_col in df.columns else None,
            exercise_style=str(style),
        )
        result = audit_put_call_parity(inp, max_abs_mid_residual=max_abs_mid_residual)
        records.append({"index": idx, **result})
    return pd.DataFrame.from_records(records).set_index("index")
