"""Market Gravity Engine — AI hypothesis research layer.

The LLM is a constrained hypothesis generator, not the backtest authority.
It never executes generated code, never controls targets/splits/metrics, and
never sees untouched test results while proposing candidates.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
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

from validation import purged_walk_forward_splits


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
    instrument: str = "SPX"
    asset_type: str = "index"
    inner_min_train: int = 20
    inner_test_size: int = 5
    inner_gap: int = 1
    inner_step: int | None = None

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
        if self.inner_min_train < 10:
            raise ValueError("inner_min_train must be >= 10")
        if self.inner_test_size < 2:
            raise ValueError("inner_test_size must be >= 2")
        if self.inner_gap < 0:
            raise ValueError("inner_gap must be >= 0")
        if self.asset_type not in {"index", "etf", "stock"}:
            raise ValueError("asset_type must be index, etf, or stock")


@dataclass(frozen=True)
class HypothesisSpec:
    name: str
    features: tuple[str, ...]
    interactions: tuple[tuple[str, str], ...]
    rationale: str
    mechanism: str = ""
    assumptions: tuple[str, ...] = ()
    expected_regimes: tuple[str, ...] = ()
    failure_modes: tuple[str, ...] = ()


@dataclass
class CandidateResult:
    spec: HypothesisSpec
    train_log_loss: float
    train_brier: float
    train_folds: int = 0
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
    return text[first:last+1]


def _clean_string_list(value: Any, *, limit: int = 8, max_len: int = 300) -> tuple[str, ...]:
    if value is None:
        return ()
    if not isinstance(value, list):
        raise ValueError("expected a list of strings")
    return tuple(str(x).strip()[:max_len] for x in value[:limit] if str(x).strip())


def validate_hypothesis(payload: dict[str, Any], config: ResearchConfig) -> HypothesisSpec:
    allowed = set(config.feature_pool)
    name = str(payload.get("name", "unnamed")).strip()[:120]
    rationale = str(payload.get("rationale", "")).strip()[:2000]
    mechanism = str(payload.get("mechanism", "")).strip()[:2000]
    assumptions = _clean_string_list(payload.get("assumptions", []))
    expected_regimes = _clean_string_list(payload.get("expected_regimes", []))
    failure_modes = _clean_string_list(payload.get("failure_modes", []))

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

    interactions = []
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

    if not mechanism:
        mechanism = rationale
    if not assumptions:
        raise ValueError("candidate must state at least one explicit model/market assumption")
    if not failure_modes:
        failure_modes = ("No explicit failure mode supplied; candidate should be treated conservatively.",)

    return HypothesisSpec(
        name=name,
        features=features,
        interactions=tuple(interactions),
        rationale=rationale,
        mechanism=mechanism,
        assumptions=assumptions,
        expected_regimes=expected_regimes,
        failure_modes=failure_modes,
    )


def build_design_matrix(df: pd.DataFrame, spec: HypothesisSpec) -> pd.DataFrame:
    x = df.loc[:, list(spec.features)].astype(float).copy()
    for a, b in spec.interactions:
        x[f"INT__{a}__X__{b}"] = df[a].astype(float) * df[b].astype(float)
    return x


def _multiclass_brier(y_true: Iterable[Any], proba: np.ndarray, classes: np.ndarray) -> float:
    y = np.asarray(list(y_true))
    one_hot = np.zeros_like(proba, dtype=float)
    class_to_idx = {c: i for i, c in enumerate(classes)}
    for row, label in enumerate(y):
        if label not in class_to_idx:
            raise ValueError(f"evaluation class absent from fitted model: {label}")
        one_hot[row, class_to_idx[label]] = 1.0
    return float(np.mean(np.sum((proba - one_hot) ** 2, axis=1)))


def _make_model(config: ResearchConfig) -> Pipeline:
    return Pipeline([
        ("scale", StandardScaler()),
        ("model", LogisticRegression(max_iter=3000, class_weight="balanced", random_state=config.random_state)),
    ])


def fit_candidate(spec: HypothesisSpec, train: pd.DataFrame, evaluate: pd.DataFrame, config: ResearchConfig):
    x_train = build_design_matrix(train, spec)
    y_train = train[config.target_col]
    x_eval = build_design_matrix(evaluate, spec)
    y_eval = evaluate[config.target_col]
    model = _make_model(config)
    model.fit(x_train, y_train)
    classes = model.named_steps["model"].classes_
    unknown = set(y_eval) - set(classes)
    if unknown:
        raise ValueError(f"evaluation contains unseen classes: {sorted(unknown, key=str)}")
    proba = model.predict_proba(x_eval)
    return model, float(log_loss(y_eval, proba, labels=classes)), _multiclass_brier(y_eval, proba, classes)


def inner_walk_forward_score(spec: HypothesisSpec, train: pd.DataFrame, config: ResearchConfig):
    """TRAIN-only purged walk-forward score used for AI feedback."""
    n = len(train)
    min_train = min(config.inner_min_train, max(10, n - config.inner_gap - config.inner_test_size))
    if min_train < 10:
        raise ValueError("not enough TRAIN rows for inner walk-forward scoring")
    max_test = n - min_train - config.inner_gap
    test_size = min(config.inner_test_size, max_test)
    if test_size < 2:
        raise ValueError("not enough TRAIN rows for an inner evaluation fold")
    step = config.inner_step or test_size
    windows = purged_walk_forward_splits(
        n, min_train=min_train, test_size=test_size, gap=config.inner_gap, step=step
    )
    if not windows:
        raise ValueError("inner walk-forward produced no usable folds")
    weighted_loss = weighted_brier = 0.0
    total_rows = used_folds = 0
    for window in windows:
        fit = train.iloc[window.train_start:window.train_end]
        evaluate = train.iloc[window.test_start:window.test_end]
        if fit[config.target_col].nunique() < 2:
            continue
        if set(evaluate[config.target_col]) - set(fit[config.target_col]):
            continue
        _, loss, brier = fit_candidate(spec, fit, evaluate, config)
        weight = len(evaluate)
        weighted_loss += loss * weight
        weighted_brier += brier * weight
        total_rows += weight
        used_folds += 1
    if total_rows == 0:
        raise ValueError("no class-compatible inner walk-forward folds")
    return weighted_loss / total_rows, weighted_brier / total_rows, used_folds


def generate_hypothesis(client: Any, config: ResearchConfig, train_memory: list[dict[str, Any]]) -> HypothesisSpec:
    prompt = f"""
You are QuantAI, the constrained senior-quant hypothesis layer inside Market Gravity Engine.

Instrument profile: {config.instrument}
Asset type: {config.asset_type}
Target fixed by the engine: {config.target_col}

Your role:
- identify a plausible market inefficiency / conditional mechanism;
- expose the assumptions needed for that mechanism to hold;
- state where it should work and why it should fail;
- propose a small, interpretable feature set and interactions.

Hard rules:
1. Do NOT write Python or executable code.
2. Do NOT claim profitability, Sharpe, or probability from intuition.
3. Use only the frozen causal feature whitelist below.
4. The engine controls model fitting, transaction costs, slippage, risk metrics,
   walk-forward evaluation, DSR, calibration, test data, and contract monetization.
5. Prefer mechanisms grounded in session structure, microstructure, volatility,
   dealer geometry, reachability, boundary competition, or relative-state effects.
6. State explicit assumptions; fewer justified assumptions are preferable, but the
   engine does not use an arbitrary assumption-count penalty.
7. Explicitly try to falsify the hypothesis: give at least one failure mode.
8. Untouched TEST results are never available to you.

Frozen feature whitelist:
{json.dumps(list(config.feature_pool), indent=2)}

Prior TRAIN-only INNER-WALK-FORWARD memory:
{json.dumps(train_memory, indent=2)}

Return ONLY one JSON object:
{{
  "name": "short descriptive name",
  "mechanism": "specific market inefficiency or conditional mechanism",
  "assumptions": ["assumption required for mechanism to remain valid"],
  "features": ["feature_a", "feature_b"],
  "interactions": [["feature_a", "feature_b"]],
  "expected_regimes": ["where the mechanism should be strongest"],
  "failure_modes": ["specific condition that should break the mechanism"],
  "rationale": "brief statistical/economic rationale"
}}
""".strip()
    response = client.responses.create(model=config.model_name, input=prompt)
    return validate_hypothesis(json.loads(_strip_json(response.output_text)), config)


def run_ai_research(df: pd.DataFrame, config: ResearchConfig, client: Any | None = None) -> dict[str, Any]:
    """Purged TRAIN feedback -> frozen CALIBRATION selection -> TEST once."""
    split = chronological_split(df, config)
    if client is None:
        from openai import OpenAI
        if not os.getenv("OPENAI_API_KEY"):
            raise RuntimeError("Set OPENAI_API_KEY in the environment; never hard-code it in source")
        client = OpenAI()

    train_memory = []
    candidates = []
    seen = set()
    attempts = 0
    while len(candidates) < config.iterations and attempts < config.iterations * 4:
        attempts += 1
        try:
            spec = generate_hypothesis(client, config, train_memory)
        except ValueError as exc:
            train_memory.append({"status": "hypothesis_schema_rejected", "reason": str(exc)})
            continue
        signature = (tuple(sorted(spec.features)), tuple(sorted(spec.interactions)))
        if signature in seen:
            train_memory.append({"name": spec.name, "status": "duplicate_rejected"})
            continue
        seen.add(signature)
        try:
            train_loss, train_brier, folds = inner_walk_forward_score(spec, split.train, config)
        except ValueError as exc:
            train_memory.append({"name": spec.name, "status": "inner_walk_forward_rejected", "reason": str(exc)})
            continue
        result = CandidateResult(spec=spec, train_log_loss=train_loss, train_brier=train_brier, train_folds=folds)
        candidates.append(result)
        train_memory.append({
            "name": spec.name,
            "features": list(spec.features),
            "interactions": [list(x) for x in spec.interactions],
            "mechanism": spec.mechanism,
            "assumptions": list(spec.assumptions),
            "expected_regimes": list(spec.expected_regimes),
            "failure_modes": list(spec.failure_modes),
            "train_metric_source": "PURGED_INNER_WALK_FORWARD",
            "train_folds": folds,
            "train_log_loss": round(train_loss, 6),
            "train_brier": round(train_brier, 6),
            "train_objective": round(result.train_objective, 6),
        })

    if not candidates:
        raise RuntimeError("no valid candidate was generated")

    # Freeze candidate set. No more LLM calls beyond this point.
    for result in candidates:
        _, result.calibration_log_loss, result.calibration_brier = fit_candidate(
            result.spec, split.train, split.calibration, config
        )
    winner = min(candidates, key=lambda r: r.calibration_objective)
    train_plus_cal = pd.concat([split.train, split.calibration], axis=0)
    _, winner.test_log_loss, winner.test_brier = fit_candidate(
        winner.spec, train_plus_cal, split.test, config
    )

    return {
        "research_protocol": "TRAIN_INNER_WALK_FORWARD_GENERATE__CALIBRATE_SELECT__TEST_ONCE",
        "instrument": config.instrument,
        "asset_type": config.asset_type,
        "target": config.target_col,
        "model_name": config.model_name,
        "rows": {"train": len(split.train), "calibration": len(split.calibration), "test": len(split.test)},
        "winner": {
            "spec": asdict(winner.spec),
            "train_metric_source": "PURGED_INNER_WALK_FORWARD",
            "train_folds": winner.train_folds,
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
                "train_metric_source": "PURGED_INNER_WALK_FORWARD",
                "train_folds": r.train_folds,
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
            "EXPLICIT_ASSUMPTION_REGISTRY_REQUIRED",
            "TRAIN_FEEDBACK_IS_PURGED_INNER_WALK_FORWARD_ONLY",
            "TEST_RESULTS_NEVER_FED_TO_LLM",
            "AI_CANNOT_CHANGE_TARGET_OR_METRICS",
            "AI_CANNOT_CHANGE_GRAVITY_STRUCTURAL_LEVELS",
            "RETURNS_DSR_COSTS_AND_SLIPPAGE_EVALUATED_DOWNSTREAM",
            "CONTRACT_MONETIZATION_REMAINS_SEPARATE",
        ],
    }


def yfinance_smoke_test_frame(start: str, end: str, interval: str = "1d") -> pd.DataFrame:
    """Optional RTH smoke-test loader; never the production 23h Gravity source."""
    import yfinance as yf
    spx = yf.download("^GSPC", start=start, end=end, interval=interval, auto_adjust=False, progress=False)
    vix = yf.download("^VIX", start=start, end=end, interval=interval, auto_adjust=False, progress=False)
    if spx.empty or vix.empty:
        raise RuntimeError("yfinance returned no data for the requested window/interval")

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
