"""Option-chain integrity diagnostics for Market Gravity.

Put-call parity is used here as a data-quality and synchronization check, not as
a directional trading signal. The audit can use either a supplied forward price
or spot/rate/dividend inputs. Executable bid/ask bounds provide a threshold-free
no-arbitrage consistency check when full quotes are available.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import exp
from typing import Iterable

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


def _finite(name: str, value: float | None, *, allow_none: bool = False) -> float | None:
    if value is None and allow_none:
        return None
    x = float(value)
    if not np.isfinite(x):
        raise ValueError(f"{name} must be finite")
    return x


def theoretical_call_minus_put(inp: ParityInputs) -> float:
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


def audit_put_call_parity(inp: ParityInputs, *, max_abs_mid_residual: float | None = None) -> dict[str, object]:
    call_mid = _finite("call_mid", inp.call_mid)
    put_mid = _finite("put_mid", inp.put_mid)
    if call_mid < 0 or put_mid < 0:
        raise ValueError("option prices cannot be negative")
    theo = theoretical_call_minus_put(inp)
    observed = float(call_mid - put_mid)
    residual = observed - theo

    out: dict[str, object] = {
        "status": "DIAGNOSTIC_ONLY",
        "theoretical_call_minus_put": theo,
        "observed_call_minus_put_mid": observed,
        "mid_residual": residual,
        "abs_mid_residual": abs(residual),
        "method": "FORWARD_PARITY" if inp.forward is not None else "SPOT_DIVIDEND_PARITY",
        "quality_flags": [],
    }

    quote_fields = (inp.call_bid, inp.call_ask, inp.put_bid, inp.put_ask)
    if all(v is not None for v in quote_fields):
        cb = _finite("call_bid", inp.call_bid)
        ca = _finite("call_ask", inp.call_ask)
        pb = _finite("put_bid", inp.put_bid)
        pa = _finite("put_ask", inp.put_ask)
        if min(cb, ca, pb, pa) < 0 or cb > ca or pb > pa:
            raise ValueError("invalid bid/ask quote ordering")
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
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"missing parity columns: {sorted(missing)}")

    records = []
    for idx, row in df.iterrows():
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
        )
        result = audit_put_call_parity(inp, max_abs_mid_residual=max_abs_mid_residual)
        records.append({"index": idx, **result})
    return pd.DataFrame.from_records(records).set_index("index")
