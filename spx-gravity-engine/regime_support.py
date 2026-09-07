"""Support audit for regime-conditioned strategy results.

Market Gravity already reports strategy metrics by an upstream regime label. This
module adds the missing denominator audit: a regime slice with very few rows or
very few active observations must not be presented as robust evidence.

No universal sample-size threshold is invented. If the caller does not provide a
minimum, the result is diagnostic only. When explicit minimums are provided, the
module returns PASS/FAIL support status without changing any trading signal.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import pandas as pd


@dataclass(frozen=True)
class RegimeSupportConfig:
    regime_col: str
    position_col: str = "position"
    min_observations: int | None = None
    min_active_observations: int | None = None

    def validate(self) -> None:
        if not self.regime_col:
            raise ValueError("regime_col is required")
        if not self.position_col:
            raise ValueError("position_col is required")
        for name, value in {
            "min_observations": self.min_observations,
            "min_active_observations": self.min_active_observations,
        }.items():
            if value is not None and (not isinstance(value, int) or value < 1):
                raise ValueError(f"{name} must be a positive integer when configured")


def audit_regime_support(df: pd.DataFrame, config: RegimeSupportConfig) -> dict[str, object]:
    config.validate()
    required = {config.regime_col, config.position_col}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"missing regime-support columns: {sorted(missing)}")
    if len(df) == 0:
        raise ValueError("regime support audit requires at least one row")

    work = df[[config.regime_col, config.position_col]].copy()
    work[config.position_col] = pd.to_numeric(work[config.position_col], errors="coerce")
    work = work.dropna(subset=[config.regime_col, config.position_col])
    if work.empty:
        raise ValueError("no complete regime-support rows")

    total = len(work)
    thresholds_configured = config.min_observations is not None or config.min_active_observations is not None
    regimes: dict[str, object] = {}
    failures = 0

    for regime, sub in work.groupby(config.regime_col, dropna=False, sort=True):
        obs = int(len(sub))
        active = int((sub[config.position_col].astype(float) != 0.0).sum())
        checks: dict[str, bool] = {}
        if config.min_observations is not None:
            checks["observations"] = obs >= config.min_observations
        if config.min_active_observations is not None:
            checks["active_observations"] = active >= config.min_active_observations
        status = "DIAGNOSTIC_ONLY"
        if thresholds_configured:
            status = "PASS" if all(checks.values()) else "FAIL"
            failures += int(status == "FAIL")
        regimes[str(regime)] = {
            "status": status,
            "observations": obs,
            "active_observations": active,
            "sample_fraction": float(obs / total),
            "active_fraction_within_regime": float(active / obs) if obs else 0.0,
            "configured_checks": checks,
        }

    overall = "DIAGNOSTIC_ONLY"
    if thresholds_configured:
        overall = "PASS" if failures == 0 else "FAIL"

    return {
        "status": overall,
        "config": asdict(config),
        "total_observations": int(total),
        "regimes": regimes,
        "hard_guards": [
            "REGIME_LABELS_MUST_BE_DEFINED_UPSTREAM_WITH_CAUSAL_INFORMATION",
            "SMALL_REGIME_SLICES_MUST_NOT_BE PRESENTED AS ROBUST EVIDENCE",
            "NO_SAMPLE_SIZE_THRESHOLD_IS_INFERRED_AUTOMATICALLY",
            "THIS_AUDIT_DOES_NOT_CHANGE_TRADING_SIGNALS",
        ],
    }
