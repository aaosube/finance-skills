"""Empirical path simulation for reachability and boundary competition.

This module deliberately does *not* assume GBM/normal returns. It resamples
chronological historical log-return blocks to preserve short-range dependence and
estimates finite-horizon first-hit frequencies for lower/upper barriers.

The result is a robustness/baseline diagnostic, not a calibrated production
probability. The caller must provide a causally eligible historical sample and a
block size chosen outside the untouched test outcome.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np


@dataclass(frozen=True)
class BlockBootstrapConfig:
    horizon_steps: int
    n_paths: int = 5000
    block_size: int = 5
    seed: int = 42

    def validate(self, history_size: int) -> None:
        if self.horizon_steps < 1:
            raise ValueError("horizon_steps must be >= 1")
        if self.n_paths < 1:
            raise ValueError("n_paths must be >= 1")
        if self.block_size < 1:
            raise ValueError("block_size must be >= 1")
        if self.block_size > history_size:
            raise ValueError("block_size cannot exceed historical sample length")


def _sample_moving_blocks(
    log_returns: np.ndarray,
    config: BlockBootstrapConfig,
) -> np.ndarray:
    n = len(log_returns)
    config.validate(n)
    rng = np.random.default_rng(config.seed)
    block_starts = np.arange(0, n - config.block_size + 1)
    if block_starts.size == 0:
        raise ValueError("historical sample is too short for the requested block_size")

    blocks_needed = int(np.ceil(config.horizon_steps / config.block_size))
    sampled = np.empty((config.n_paths, config.horizon_steps), dtype=float)
    for p in range(config.n_paths):
        cursor = 0
        for _ in range(blocks_needed):
            start = int(rng.choice(block_starts))
            block = log_returns[start:start + config.block_size]
            take = min(config.horizon_steps - cursor, len(block))
            sampled[p, cursor:cursor + take] = block[:take]
            cursor += take
            if cursor >= config.horizon_steps:
                break
    return sampled


def empirical_first_hit(
    historical_log_returns: Iterable[float],
    *,
    start_price: float,
    lower_barrier: float,
    upper_barrier: float,
    config: BlockBootstrapConfig,
) -> dict[str, object]:
    """Estimate upper/lower/neither first-hit frequencies by block bootstrap."""
    history = np.asarray(list(historical_log_returns), dtype=float)
    history = history[np.isfinite(history)]
    if len(history) < 2:
        raise ValueError("at least two finite historical log returns are required")
    if not np.isfinite(start_price) or start_price <= 0:
        raise ValueError("start_price must be finite and > 0")
    if not np.isfinite(lower_barrier) or not np.isfinite(upper_barrier):
        raise ValueError("barriers must be finite")
    if not lower_barrier < start_price < upper_barrier:
        raise ValueError("require lower_barrier < start_price < upper_barrier")

    draws = _sample_moving_blocks(history, config)
    prices = start_price * np.exp(np.cumsum(draws, axis=1))
    upper_hit = prices >= upper_barrier
    lower_hit = prices <= lower_barrier

    sentinel = config.horizon_steps + 1
    upper_first_step = np.where(upper_hit.any(axis=1), upper_hit.argmax(axis=1) + 1, sentinel)
    lower_first_step = np.where(lower_hit.any(axis=1), lower_hit.argmax(axis=1) + 1, sentinel)

    upper_first = upper_first_step < lower_first_step
    lower_first = lower_first_step < upper_first_step
    neither = (upper_first_step == sentinel) & (lower_first_step == sentinel)
    # A discrete path cannot newly cross both separated barriers on the same sampled
    # step, but keep a fail-closed check rather than silently assigning a tie.
    ties = (upper_first_step == lower_first_step) & (upper_first_step != sentinel)
    if np.any(ties):
        raise ValueError("simulated path produced an unresolved same-step dual barrier hit")

    def _mean_hit_step(mask: np.ndarray, steps: np.ndarray) -> float | None:
        vals = steps[mask]
        return float(np.mean(vals)) if vals.size else None

    p_upper = float(np.mean(upper_first))
    p_lower = float(np.mean(lower_first))
    p_neither = float(np.mean(neither))
    if not np.isclose(p_upper + p_lower + p_neither, 1.0, atol=1e-12):
        raise ValueError("first-hit frequencies do not sum to one")

    return {
        "status": "EMPIRICAL_BLOCK_BOOTSTRAP_UNCALIBRATED",
        "start_price": float(start_price),
        "lower_barrier": float(lower_barrier),
        "upper_barrier": float(upper_barrier),
        "horizon_steps": int(config.horizon_steps),
        "n_paths": int(config.n_paths),
        "block_size": int(config.block_size),
        "seed": int(config.seed),
        "upper_first_frequency": p_upper,
        "lower_first_frequency": p_lower,
        "neither_frequency": p_neither,
        "mean_upper_first_step": _mean_hit_step(upper_first, upper_first_step),
        "mean_lower_first_step": _mean_hit_step(lower_first, lower_first_step),
        "simulated_terminal_median": float(np.median(prices[:, -1])),
        "simulated_path_low_median": float(np.median(np.min(prices, axis=1))),
        "simulated_path_high_median": float(np.median(np.max(prices, axis=1))),
        "hard_guards": [
            "HISTORICAL_SAMPLE_MUST_BE_CAUSALLY_ELIGIBLE",
            "BLOCK_SIZE_MUST_NOT_BE_TUNED_ON_UNTOUCHED_TEST_OUTCOME",
            "NO_GBM_OR_NORMAL_RETURN_ASSUMPTION",
            "SIMULATION_FREQUENCIES_ARE_NOT_CALIBRATED_PRODUCTION_PROBABILITIES",
            "COMPARE_AGAINST_BROWNIAN_AND_EMPIRICAL_BASELINES_OOS",
        ],
    }
