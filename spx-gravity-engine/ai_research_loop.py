"""SPX Gravity Engine — AI hypothesis research layer.

Purpose
-------
Use an LLM only as a hypothesis generator over a frozen, causally-built feature
matrix. The LLM never executes code, never sees test results while proposing
hypotheses, never controls the train/calibration/test split, and never changes
SPX Gravity structural levels.

Expected dataframe
------------------
- DatetimeIndex in chronological order.
- One column per precomputed causal feature.
- A target column supplied by the engine, e.g. ``boundary_outcome`` with
  labels such as ``UPPER_FIRST``, ``LOWER_FIRST``, ``NEITHER``.

Optional columns used by downstream execution research are intentionally not
part of model selection here. Contract monetization remains a separate stage.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any, Iterable
import json
import os
import re

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


@dataclass(frozen=True)
class ResearchConfig:
    target_col: str
    feature_pool: tuple[str, ...]
    train_fraction: float = 0.60
    calibration_fraction: float = 0.20
    iterations: int = 10
    max_features_per_candidate: int = 8
    max_interactions_per_candidate: int = 4
    random_state: int = 42
    model_name: str = os.getenv("OPENAI_MODEL", "gpt-5.6-luna")

    def validate(self) -> None:
        if not (0 < self.train_fraction < 1):
            raise ValueError("train_fraction must be in (0, 1)")
        if not (0 < self.calibration_fraction < 1):
            raise ValueError("calibration_fraction must be in (0, 1)")
        if self.train_fraction + self.calibration_fraction >= 1:
            raise ValueError("train + calibration fractions must leave a test set")
        if self.iterations < 1:
            raise ValueError("iterations must be >= 1")
        if not self.feature_pool:
            raise ValueError("feature_pool cannot be empty")


@dataclass(frozen=True)
class HypothesisSpec:
    name: str
    features: tuple[str, ...]
    interactions: tuple[tuple[str, str], ...]
    rationale: str


@dataclass
class CandidateResult:
    spec: HypothesisSpec
    train_log_loss: float
    train_brier: float
    calibration_log_loss: float | None = None
    calibration_brier: float | None = None
    test_log_loss: float | None = None
    test_brier: float | None = None

    @property
    def train_objective(self) -> float:
        return self.train_log_loss + self.train_brier

    @property
    def calibration_objective(self) -> float:
        if self.calibration_log_loss is None or self.calibration_brier is None:
            return float("inf")
        return self.calibration_log_loss + self.calibration_brier


@dataclass(frozen=True)
class ChronologicalSplit:
    train: pd.DataFrame
    calibration: pd.DataFrame
    test: pd.DataFrame


def ensure_causal_frame(df: pd.DataFrame, config: ResearchConfig) -> pd.DataFrame:
    """Validate shape, chronology, target and feature availability."""
    config.validate()
    if not isinstance(df.index, pd.DatetimeIndex):
        raise TypeError("df.index must be a DatetimeIndex")
    if not df.index.is_monotonic_increasing:
        raise ValueError("df must be sorted chronologically before research")
    if df.index.has_duplicates:
        raise ValueError("duplicate timestamps are not allowed")

    required = set(config.feature_pool) | {config.target_col}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"missing required columns: {sorted(missing)}")

    clean = df.loc[:, list(config.feature_pool) + [config.target_col]].copy()
    clean = clean.replace([np.inf, -np.inf], np.nan).dropna()
    if len(clean) < 30:
        raise ValueError("at least 30 complete chronological observations are required")
    if clean[config.target_col].nunique() < 2:
        raise ValueError("target must contain at least two classes")
    return clean


def chronological_split(df: pd.DataFrame, config: ResearchConfig) -> ChronologicalSplit:
    """One-way chronological split. Never randomize market time series."""
    clean = ensure_causal_frame(df, config)
    n = len(clean)
    train_end = int(np.floor(n * config.train_fraction))
    cal_end = train_end + int(np.floor(n * config.calibration_fraction))
    if train_end < 10 or cal_end <= train_end or cal_end >= n:
        raise ValueError("split leaves an unusable train/calibration/test partition")
    return ChronologicalSplit(
        train=clean.iloc[:train_end].copy(),
        calibration=clean.iloc[train_end:cal_end].copy(),
        test=clean.iloc[cal_end:].copy(),
    )


def _strip_json(text: str) -> str:
    text = text.strip()
    text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\s*```$", "", text)
    first, last = text.find("{"), text.rfind("}")
    if first < 0 or last < first:
        raise ValueError("LLM response did not contain a JSON object")
    return text[first : last + 1]


def validate_hypothesis(payload: dict[str, Any], config: ResearchConfig) -> HypothesisSpec:
    """Hard whitelist: the LLM can only select known features and interactions."""
    allowed = set(config.feature_pool)
    name = str(payload.get("name", "unnamed")).strip()[:120]
    rationale = str(payload.get("rationale", "")).strip()[:2000]

    raw_features = payload.get("features", [])
    if not isinstance(raw_features, list):
        raise ValueError("features must be a list")
    features = tuple(dict.fromkeys(str(x) for x in raw_features))
    if not features:
        raise ValueError("candidate must use at least one feature")
    if len(features) > config.max_features_per_candidate:
        raise ValueError("candidate uses too many features")
    forbidden = [f for f in features if f not in allowed]
    if forbidden:
        raise ValueError(f"candidate requested forbidden/unknown features: {forbidden}")

    interactions: list[tuple[str, str]] = []
    raw_interactions = payload.get("interactions", [])
    if not isinstance(raw_interactions, list):
        raise ValueError("interactions must be a list")
    if len(raw_interactions) > config.max_interactions_per_candidate:
        raise ValueError("candidate uses too many interactions")
    for pair in raw_interactions:
        if not isinstance(pair, list) or len(pair) != 2:
            raise ValueError("each interaction must be [feature_a, feature_b]")
        a, b = str(pair[0]), str(pair[1])
        if a not in allowed or b not in allowed:
            raise ValueError(f"forbidden interaction: {(a, b)}")
        interactions.append((a, b))

    return HypothesisSpec(name=name, features=features, interactions=tuple(interactions), rationale=rationale)


def build_design_matrix(df: pd.DataFrame, spec: HypothesisSpec) -> pd.DataFrame:
    """Deterministic engine-owned feature construction; no generated Python is executed."""
    x = df.loc[:, list(spec.features)].astype(float).copy()
    for a, b in spec.interactions:
        col = f"INT__{a}__X__{b}"
        x[col] = df[a].astype(float) * df[b].astype(float)
    return x


def _multiclass_brier(y_true: Iterable[Any], proba: np.ndarray, classes: np.ndarray) -> float:
    y = np.asarray(list(y_true))
    one_hot = np.zeros_like(proba, dtype=float)
    class_to_idx = {c: i for i, c in enumerate(classes)}
    for row, label in enumerate(y):
        one_hot[row, class_to_idx[label]] = 1.0
    return float(np.mean(np.sum((proba - one_hot) ** 2, axis=1)))


def _make_model(config: ResearchConfig) -> Pipeline:
    return Pipeline(
        steps=[
            ("scale", StandardScaler()),
            (
                "model",
                LogisticRegression(
                    max_iter=3000,
                    class_weight="balanced",
                    random_state=config.random_state,
                ),
            ),
        ]
    )


def fit_candidate(
    spec: HypothesisSpec,
    train: pd.DataFrame,
    evaluate: pd.DataFrame,
    config: ResearchConfig,
) -> tuple[Pipeline, float, float]:
    x_train = build_design_matrix(train, spec)
    y_train = train[config.target_col]
    x_eval = build_design_matrix(evaluate, spec)
    y_eval = evaluate[config.target_col]

    model = _make_model(config)
    model.fit(x_train, y_train)
    proba = model.predict_proba(x_eval)
    classes = model.named_steps["model"].classes_

    loss = float(log_loss(y_eval, proba, labels=classes))
    brier = _multiclass_brier(y_eval, proba, classes)
    return model, loss, brier


def generate_hypothesis(
    client: Any,
    config: ResearchConfig,
    train_memory: list[dict[str, Any]],
) -> HypothesisSpec:
    """Ask the LLM for a JSON hypothesis only. It receives TRAIN results, never test metrics."""
    prompt = f"""
You are the hypothesis-generation layer for the SPX Gravity Engine.
You are NOT allowed to write or execute Python strategy code.
You may only choose from this frozen feature whitelist:
{json.dumps(list(config.feature_pool), indent=2)}

Target is fixed by the engine as: {config.target_col}
The engine, not you, controls chronological train/calibration/test splits,
model fitting, metrics, execution, and contract monetization.

Prior TRAIN-only research memory:
{json.dumps(train_memory, indent=2)}

Propose a new and materially different hypothesis that could improve calibrated
first-hit / boundary classification. Prefer economically interpretable session,
regime, dealer-geometry, reachability, and cross-market interactions. Avoid
needless feature count and avoid rephrasing an already-tested candidate.

Return ONLY one JSON object with exactly these keys:
{{
  "name": "short descriptive name",
  "features": ["feature_a", "feature_b"],
  "interactions": [["feature_a", "feature_b"]],
  "rationale": "why the interaction might exist"
}}
""".strip()

    response = client.responses.create(model=config.model_name, input=prompt)
    payload = json.loads(_strip_json(response.output_text))
    return validate_hypothesis(payload, config)


def run_ai_research(
    df: pd.DataFrame,
    config: ResearchConfig,
    client: Any | None = None,
) -> dict[str, Any]:
    """Leakage-controlled AI research loop.

    Procedure:
    1. Split chronologically once.
    2. LLM proposes hypotheses using feature names + TRAIN-only results.
    3. All candidates are fitted/scored on TRAIN only during generation.
    4. After generation ends, freeze candidates.
    5. Score the frozen set on CALIBRATION and select one winner.
    6. Evaluate that single winner on TEST exactly once.

    TEST metrics are never fed back to the LLM.
    """
    split = chronological_split(df, config)

    if client is None:
        from openai import OpenAI

        if not os.getenv("OPENAI_API_KEY"):
            raise RuntimeError("Set OPENAI_API_KEY in the environment; never hard-code it in source")
        client = OpenAI()

    train_memory: list[dict[str, Any]] = []
    candidates: list[CandidateResult] = []
    seen: set[tuple[tuple[str, ...], tuple[tuple[str, str], ...]]] = set()

    attempts = 0
    while len(candidates) < config.iterations and attempts < config.iterations * 4:
        attempts += 1
        spec = generate_hypothesis(client, config, train_memory)
        signature = (tuple(sorted(spec.features)), tuple(sorted(spec.interactions)))
        if signature in seen:
            train_memory.append({"name": spec.name, "status": "duplicate_rejected"})
            continue
        seen.add(signature)

        _, train_loss, train_brier = fit_candidate(spec, split.train, split.train, config)
        result = CandidateResult(
            spec=spec,
            train_log_loss=train_loss,
            train_brier=train_brier,
        )
        candidates.append(result)
        train_memory.append(
            {
                "name": spec.name,
                "features": list(spec.features),
                "interactions": [list(x) for x in spec.interactions],
                "train_log_loss": round(train_loss, 6),
                "train_brier": round(train_brier, 6),
                "train_objective": round(result.train_objective, 6),
            }
        )

    if not candidates:
        raise RuntimeError("no valid candidate was generated")

    # Candidate set is now frozen. No more LLM calls beyond this point.
    for result in candidates:
        _, cal_loss, cal_brier = fit_candidate(result.spec, split.train, split.calibration, config)
        result.calibration_log_loss = cal_loss
        result.calibration_brier = cal_brier

    winner = min(candidates, key=lambda r: r.calibration_objective)

    # Refit only on train+calibration after selection, then one untouched test evaluation.
    train_plus_cal = pd.concat([split.train, split.calibration], axis=0)
    _, test_loss, test_brier = fit_candidate(winner.spec, train_plus_cal, split.test, config)
    winner.test_log_loss = test_loss
    winner.test_brier = test_brier

    return {
        "research_protocol": "TRAIN_GENERATE__CALIBRATE_SELECT__TEST_ONCE",
        "target": config.target_col,
        "model_name": config.model_name,
        "rows": {
            "train": len(split.train),
            "calibration": len(split.calibration),
            "test": len(split.test),
        },
        "winner": {
            "spec": asdict(winner.spec),
            "train_log_loss": winner.train_log_loss,
            "train_brier": winner.train_brier,
            "calibration_log_loss": winner.calibration_log_loss,
            "calibration_brier": winner.calibration_brier,
            "test_log_loss": winner.test_log_loss,
            "test_brier": winner.test_brier,
        },
        "all_candidates": [
            {
                "spec": asdict(r.spec),
                "train_log_loss": r.train_log_loss,
                "train_brier": r.train_brier,
                "calibration_log_loss": r.calibration_log_loss,
                "calibration_brier": r.calibration_brier,
            }
            for r in candidates
        ],
        "hard_guards": [
            "NO_EXEC_OF_LLM_CODE",
            "FEATURE_WHITELIST_ONLY",
            "CHRONOLOGICAL_SPLIT_ONLY",
            "TEST_RESULTS_NEVER_FED_TO_LLM",
            "AI_CANNOT_CHANGE_TARGET_OR_METRICS",
            "AI_CANNOT_CHANGE_SPX_GRAVITY_STRUCTURAL_LEVELS",
            "CONTRACT_MONETIZATION_REMAINS_SEPARATE",
        ],
    }


def yfinance_smoke_test_frame(
    start: str,
    end: str,
    interval: str = "1d",
) -> pd.DataFrame:
    """Optional RTH smoke-test loader; NOT the 23h Gravity data source.

    This fixes the duplicate ``Close``-column bug in the original snippet.
    For production Gravity research, feed the engine-built 23h ES/NQ/SPX feature
    matrix instead of relying on Yahoo intraday history.
    """
    import yfinance as yf

    spx = yf.download("^GSPC", start=start, end=end, interval=interval, auto_adjust=False, progress=False)
    vix = yf.download("^VIX", start=start, end=end, interval=interval, auto_adjust=False, progress=False)

    if spx.empty or vix.empty:
        raise RuntimeError("yfinance returned no data for the requested window/interval")

    # yfinance can return a MultiIndex depending on version / number of symbols.
    def close_series(frame: pd.DataFrame, name: str) -> pd.Series:
        if isinstance(frame.columns, pd.MultiIndex):
            s = frame.xs("Close", axis=1, level=0).iloc[:, 0]
        else:
            s = frame["Close"]
        return s.rename(name)

    out = pd.concat([close_series(spx, "SPX"), close_series(vix, "VIX")], axis=1).dropna()
    out["SPX_return"] = out["SPX"].pct_change()
    out["VIX_change"] = out["VIX"].pct_change()
    return out.dropna()
