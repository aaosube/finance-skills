"""Validated orchestration for the SPX AI research layer.

This wrapper keeps the existing TRAIN -> CALIBRATION -> TEST_ONCE protocol,
then compares the frozen winner with a train+calibration empirical-prior
benchmark on the untouched test set. The benchmark cannot influence candidate
generation or winner selection.
"""
from __future__ import annotations

from typing import Any

import pandas as pd

from ai_research_loop import ResearchConfig, chronological_split, run_ai_research
from validation import compare_to_prior, empirical_prior_metrics


def run_validated_ai_research(
    df: pd.DataFrame,
    config: ResearchConfig,
    client: Any | None = None,
) -> dict[str, Any]:
    """Run the existing research loop and attach a test-only baseline audit."""
    split = chronological_split(df, config)
    result = run_ai_research(df, config, client=client)

    train_plus_cal = pd.concat([split.train, split.calibration], axis=0)
    baseline = empirical_prior_metrics(
        train_plus_cal[config.target_col].tolist(),
        split.test[config.target_col].tolist(),
    )

    winner = result["winner"]
    comparison = compare_to_prior(
        model_log_loss=float(winner["test_log_loss"]),
        model_brier=float(winner["test_brier"]),
        prior_metrics=baseline,
    )

    out = dict(result)
    out["test_baseline"] = baseline
    out["test_baseline_comparison"] = comparison
    out["hard_guards"] = list(result.get("hard_guards", [])) + [
        "TEST_BASELINE_CANNOT_SELECT_WINNER",
        "MODEL_MUST_BE_REPORTED_AGAINST_NAIVE_PRIOR",
    ]
    return out
