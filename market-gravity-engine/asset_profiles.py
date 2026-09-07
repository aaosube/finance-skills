"""Asset profiles for the Market Gravity Engine.

The engine is shared across INDEX / ETF / STOCK targets, but each profile has
explicit capabilities and data requirements. Futures are not target assets in
the current canonical scope.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Mapping


class AssetClass(str, Enum):
    INDEX = "INDEX"
    ETF = "ETF"
    STOCK = "STOCK"


class SessionPolicy(str, Enum):
    CASH_RTH = "CASH_RTH"
    LISTED_EXTENDED_HOURS = "LISTED_EXTENDED_HOURS"


@dataclass(frozen=True)
class AssetProfile:
    symbol: str
    asset_class: AssetClass
    session_policy: SessionPolicy
    options_enabled: bool
    supports_extended_hours_target_tape: bool
    supports_dark_pool_features: bool
    supports_finra_short_volume: bool
    supports_borrow_fee: bool
    supports_ftd: bool
    corporate_event_sensitive: bool
    reference_symbol: str | None = None
    notes: str = ""


PROFILES: Mapping[str, AssetProfile] = {
    "SPX": AssetProfile(
        symbol="SPX",
        asset_class=AssetClass.INDEX,
        session_policy=SessionPolicy.CASH_RTH,
        options_enabled=True,
        supports_extended_hours_target_tape=False,
        supports_dark_pool_features=False,
        supports_finra_short_volume=False,
        supports_borrow_fee=False,
        supports_ftd=False,
        corporate_event_sensitive=False,
        reference_symbol=None,
        notes="SPX/SPXW remains the first specialized 0DTE index profile.",
    ),
    "NDX": AssetProfile(
        symbol="NDX",
        asset_class=AssetClass.INDEX,
        session_policy=SessionPolicy.CASH_RTH,
        options_enabled=True,
        supports_extended_hours_target_tape=False,
        supports_dark_pool_features=False,
        supports_finra_short_volume=False,
        supports_borrow_fee=False,
        supports_ftd=False,
        corporate_event_sensitive=False,
        reference_symbol=None,
    ),
    "RUT": AssetProfile(
        symbol="RUT",
        asset_class=AssetClass.INDEX,
        session_policy=SessionPolicy.CASH_RTH,
        options_enabled=True,
        supports_extended_hours_target_tape=False,
        supports_dark_pool_features=False,
        supports_finra_short_volume=False,
        supports_borrow_fee=False,
        supports_ftd=False,
        corporate_event_sensitive=False,
        reference_symbol=None,
    ),
    "SPY": AssetProfile(
        symbol="SPY",
        asset_class=AssetClass.ETF,
        session_policy=SessionPolicy.LISTED_EXTENDED_HOURS,
        options_enabled=True,
        supports_extended_hours_target_tape=True,
        supports_dark_pool_features=True,
        supports_finra_short_volume=True,
        supports_borrow_fee=False,
        supports_ftd=True,
        corporate_event_sensitive=False,
        reference_symbol="SPX",
    ),
    "QQQ": AssetProfile(
        symbol="QQQ",
        asset_class=AssetClass.ETF,
        session_policy=SessionPolicy.LISTED_EXTENDED_HOURS,
        options_enabled=True,
        supports_extended_hours_target_tape=True,
        supports_dark_pool_features=True,
        supports_finra_short_volume=True,
        supports_borrow_fee=False,
        supports_ftd=True,
        corporate_event_sensitive=False,
        reference_symbol="NDX",
    ),
    "IWM": AssetProfile(
        symbol="IWM",
        asset_class=AssetClass.ETF,
        session_policy=SessionPolicy.LISTED_EXTENDED_HOURS,
        options_enabled=True,
        supports_extended_hours_target_tape=True,
        supports_dark_pool_features=True,
        supports_finra_short_volume=True,
        supports_borrow_fee=False,
        supports_ftd=True,
        corporate_event_sensitive=False,
        reference_symbol="RUT",
    ),
}


def stock_profile(symbol: str, reference_symbol: str | None = None) -> AssetProfile:
    symbol = symbol.upper().strip()
    if not symbol:
        raise ValueError("stock symbol is required")
    return AssetProfile(
        symbol=symbol,
        asset_class=AssetClass.STOCK,
        session_policy=SessionPolicy.LISTED_EXTENDED_HOURS,
        options_enabled=True,
        supports_extended_hours_target_tape=True,
        supports_dark_pool_features=True,
        supports_finra_short_volume=True,
        supports_borrow_fee=True,
        supports_ftd=True,
        corporate_event_sensitive=True,
        reference_symbol=reference_symbol,
        notes="Stock profile enables corporate-event, short, borrow and off-exchange modules when causal data exist.",
    )


def get_profile(symbol: str, asset_class: AssetClass | str | None = None, reference_symbol: str | None = None) -> AssetProfile:
    key = symbol.upper().strip()
    if key in PROFILES:
        profile = PROFILES[key]
        if asset_class is not None and AssetClass(asset_class) != profile.asset_class:
            raise ValueError(f"{key} is registered as {profile.asset_class.value}, not {AssetClass(asset_class).value}")
        return profile

    if asset_class is None:
        raise ValueError(f"Unknown symbol {key}; provide asset_class explicitly")

    cls = AssetClass(asset_class)
    if cls == AssetClass.STOCK:
        return stock_profile(key, reference_symbol=reference_symbol)

    raise ValueError(
        f"Unregistered {cls.value} {key}; add an explicit profile so index/ETF invariants are not guessed"
    )


def required_modules(profile: AssetProfile) -> tuple[str, ...]:
    modules = [
        "data_integrity",
        "instrument_session_state",
        "market_macro_regime",
        "structural_geometry",
        "direction",
        "reachability",
        "boundary_competition",
        "trigger",
        "execution_risk",
        "validation_learning",
    ]

    if profile.options_enabled:
        modules.extend(["derivatives_state", "contract_ev"])
    if profile.supports_dark_pool_features:
        modules.append("off_exchange_liquidity")
    if profile.supports_finra_short_volume:
        modules.append("lagged_short_volume")
    if profile.supports_borrow_fee:
        modules.append("borrow_state")
    if profile.supports_ftd:
        modules.append("ftd_state")
    if profile.corporate_event_sensitive:
        modules.append("corporate_event_regime")

    # Stable order without duplicates.
    return tuple(dict.fromkeys(modules))
