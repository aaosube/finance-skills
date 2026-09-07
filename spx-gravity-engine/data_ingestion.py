"""Auditable local-file ingestion for SPX Gravity research.

This module normalizes user-exported CSV/Parquet data without silently
inventing fields. It is intentionally source-agnostic at the file layer:
Barchart, ChartExchange, QuantWheel and other exports can be loaded through
explicit column maps while preserving provenance and data-quality diagnostics.

Hard rules
----------
- Never overwrite or fabricate a missing required field.
- Never silently drop rows.
- Ambiguous column mappings fail closed.
- Naive timestamps are localized explicitly and the assumption is reported.
- Parquet is optional; the caller must install a compatible engine.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Mapping
import re

import pandas as pd


class DataQualityError(ValueError):
    """Raised when a file cannot be normalized without guessing."""


_CANONICAL_ALIASES: dict[str, tuple[str, ...]] = {
    "timestamp": ("timestamp", "datetime", "date_time", "quote_time", "trade_time", "asof", "as_of"),
    "date": ("date", "trade_date", "quote_date"),
    "time": ("time", "trade_time_only", "quote_time_only"),
    "symbol": ("symbol", "ticker", "underlying", "underlying_symbol"),
    "expiration": ("expiration", "expiry", "expiration_date", "expiry_date"),
    "strike": ("strike", "strike_price"),
    "option_type": ("option_type", "type", "put_call", "call_put", "right"),
    "open_interest": ("open_interest", "openinterest", "oi"),
    "volume": ("volume", "vol"),
    "iv": ("iv", "implied_volatility", "impliedvolatility"),
    "gamma": ("gamma",),
    "delta": ("delta",),
    "vanna": ("vanna",),
    "charm": ("charm",),
    "bid": ("bid",),
    "ask": ("ask",),
    "last": ("last", "last_price", "price"),
    "spot": ("spot", "underlying_price", "underlying_last", "underlyingprice"),
}

_NUMERIC_COLUMNS = {
    "strike", "open_interest", "volume", "iv", "gamma", "delta", "vanna", "charm",
    "bid", "ask", "last", "spot",
}


def _slug(value: object) -> str:
    text = str(value).strip().lower()
    text = re.sub(r"[^a-z0-9]+", "_", text).strip("_")
    return text


def _alias_lookup() -> dict[str, str]:
    lookup: dict[str, str] = {}
    for canonical, aliases in _CANONICAL_ALIASES.items():
        for alias in aliases:
            key = _slug(alias)
            if key in lookup and lookup[key] != canonical:
                raise RuntimeError(f"internal alias collision for {key}")
            lookup[key] = canonical
    return lookup


@dataclass(frozen=True)
class DataQualityReport:
    source: str
    input_rows: int
    output_rows: int
    mapped_columns: dict[str, str]
    unmapped_columns: tuple[str, ...]
    required_columns: tuple[str, ...]
    invalid_numeric_counts: dict[str, int]
    invalid_timestamp_count: int
    duplicate_key_count: int
    timestamp_mode: str
    timezone_assumption: str | None

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def read_market_file(path: str | Path) -> pd.DataFrame:
    """Read CSV or Parquet. The function performs no semantic normalization."""
    p = Path(path)
    suffix = p.suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(p)
    if suffix in {".parquet", ".pq"}:
        try:
            return pd.read_parquet(p)
        except ImportError as exc:
            raise DataQualityError(
                "Parquet requested but no parquet engine is installed; install pyarrow or fastparquet explicitly"
            ) from exc
    raise DataQualityError(f"unsupported file type: {suffix or '<none>'}")


def _resolve_mapping(columns: Iterable[object], column_map: Mapping[str, str] | None) -> tuple[dict[object, str], tuple[str, ...]]:
    """Resolve input column -> canonical column, rejecting ambiguous targets."""
    explicit = {_slug(k): _slug(v) for k, v in (column_map or {}).items()}
    alias_lookup = _alias_lookup()
    resolved: dict[object, str] = {}
    target_to_source: dict[str, object] = {}
    unmapped: list[str] = []

    for original in columns:
        source_key = _slug(original)
        canonical = explicit.get(source_key) or alias_lookup.get(source_key)
        if not canonical:
            unmapped.append(str(original))
            continue
        if canonical in target_to_source:
            prior = target_to_source[canonical]
            raise DataQualityError(
                f"ambiguous mapping: both {prior!r} and {original!r} map to canonical column {canonical!r}"
            )
        target_to_source[canonical] = original
        resolved[original] = canonical

    return resolved, tuple(unmapped)


def _parse_timestamp(frame: pd.DataFrame, timezone: str | None) -> tuple[pd.Series, str, str | None]:
    if "timestamp" in frame.columns:
        raw = frame["timestamp"]
        mode = "TIMESTAMP_COLUMN"
    elif "date" in frame.columns and "time" in frame.columns:
        raw = frame["date"].astype(str).str.strip() + " " + frame["time"].astype(str).str.strip()
        mode = "DATE_PLUS_TIME"
    elif "date" in frame.columns:
        raw = frame["date"]
        mode = "DATE_ONLY"
    else:
        raise DataQualityError(
            "no unambiguous timestamp field found; provide a column_map to timestamp or date/time"
        )

    parsed = pd.to_datetime(raw, errors="coerce")
    valid = parsed.dropna()
    if valid.empty:
        return parsed, mode, None

    try:
        tz = parsed.dt.tz
    except AttributeError as exc:
        raise DataQualityError("timestamp values have mixed/incompatible timezone formats") from exc

    if tz is None:
        if not timezone:
            raise DataQualityError("naive timestamps require an explicit timezone")
        parsed = parsed.dt.tz_localize(timezone, ambiguous="NaT", nonexistent="NaT")
        assumption = timezone
    else:
        parsed = parsed.dt.tz_convert("UTC")
        assumption = None

    return parsed, mode, assumption


def normalize_market_frame(
    frame: pd.DataFrame,
    *,
    source: str,
    timezone: str | None = "America/New_York",
    column_map: Mapping[str, str] | None = None,
    required: Iterable[str] = (),
    duplicate_key: Iterable[str] = (),
) -> tuple[pd.DataFrame, DataQualityReport]:
    """Normalize a market export and return the untouched-row-count result + audit report."""
    if not source or not str(source).strip():
        raise DataQualityError("source is required for provenance")
    if not isinstance(frame, pd.DataFrame):
        raise TypeError("frame must be a pandas DataFrame")

    out = frame.copy()
    input_rows = len(out)
    mapping, unmapped = _resolve_mapping(out.columns, column_map)
    out = out.rename(columns=mapping)

    required_tuple = tuple(_slug(x) for x in required)
    missing = [c for c in required_tuple if c not in out.columns]
    if missing:
        raise DataQualityError(f"missing required canonical columns: {missing}")

    parsed_ts, timestamp_mode, tz_assumption = _parse_timestamp(out, timezone)
    invalid_timestamp_count = int(parsed_ts.isna().sum())
    if invalid_timestamp_count:
        raise DataQualityError(
            f"{invalid_timestamp_count} rows have invalid/ambiguous timestamps; normalization fails closed"
        )
    out["timestamp"] = parsed_ts.dt.tz_convert("UTC")

    invalid_numeric_counts: dict[str, int] = {}
    for column in sorted(_NUMERIC_COLUMNS.intersection(out.columns)):
        raw_non_null = out[column].notna()
        converted = pd.to_numeric(out[column], errors="coerce")
        invalid = int((raw_non_null & converted.isna()).sum())
        invalid_numeric_counts[column] = invalid
        out[column] = converted

    if "option_type" in out.columns:
        mapping_type = {
            "c": "call", "call": "call", "calls": "call",
            "p": "put", "put": "put", "puts": "put",
        }
        out["option_type"] = out["option_type"].map(
            lambda x: mapping_type.get(str(x).strip().lower(), str(x).strip().lower()) if pd.notna(x) else x
        )

    out["source"] = str(source).strip()

    key = ["timestamp"] + [_slug(x) for x in duplicate_key]
    missing_key = [c for c in key if c not in out.columns]
    if missing_key:
        raise DataQualityError(f"duplicate_key references missing columns: {missing_key}")
    duplicate_count = int(out.duplicated(subset=key, keep=False).sum()) if key else 0

    report = DataQualityReport(
        source=str(source).strip(),
        input_rows=input_rows,
        output_rows=len(out),
        mapped_columns={str(k): v for k, v in mapping.items()},
        unmapped_columns=unmapped,
        required_columns=required_tuple,
        invalid_numeric_counts=invalid_numeric_counts,
        invalid_timestamp_count=invalid_timestamp_count,
        duplicate_key_count=duplicate_count,
        timestamp_mode=timestamp_mode,
        timezone_assumption=tz_assumption,
    )
    return out, report


def load_market_file(path: str | Path, **kwargs: object) -> tuple[pd.DataFrame, DataQualityReport]:
    return normalize_market_frame(read_market_file(path), **kwargs)
