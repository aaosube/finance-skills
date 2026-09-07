"""Time-series validation utilities for the SPX Gravity research layer."""
from __future__ import annotations

from dataclasses import asdict, dataclass
from math import log
from typing import Sequence


@dataclass(frozen=True)
class WalkForwardWindow:
    train_start: int
    train_end: int
    test_start: int
    test_end: int

    def to_dict(self) -> dict[str, int]:
        return asdict(self)


def purged_walk_forward_splits(
    n_rows: int,
    *,
    min_train: int,
    test_size: int,
    gap: int = 0,
    step: int | None = None,
    max_train: int | None = None,
) -> list[WalkForwardWindow]:
    """Return expanding/rolling chronological windows with an optional purge gap."""
    for name, value in {"n_rows": n_rows, "min_train": min_train, "test_size": test_size}.items():
        if not isinstance(value, int) or value <= 0:
            raise ValueError(f"{name} must be a positive integer")
    if gap < 0:
        raise ValueError("gap must be >= 0")
    if max_train is not None and max_train < min_train:
        raise ValueError("max_train cannot be smaller than min_train")
    if step is None:
        step = test_size
    if step <= 0:
        raise ValueError("step must be > 0")

    windows: list[WalkForwardWindow] = []
    train_end = min_train
    while True:
        test_start = train_end + gap
        test_end = test_start + test_size
        if test_end > n_rows:
            break
        train_start = 0 if max_train is None else max(0, train_end - max_train)
        windows.append(WalkForwardWindow(train_start, train_end, test_start, test_end))
        train_end += step
    return windows


def empirical_prior_metrics(y_train: Sequence[object], y_test: Sequence[object], *, epsilon: float = 1e-12) -> dict[str, object]:
    """Evaluate a constant class-prior forecast learned from train only."""
    train = list(y_train)
    test = list(y_test)
    if not train or not test:
        raise ValueError("train and test labels must be non-empty")
    classes = tuple(dict.fromkeys(train))
    unknown = sorted({x for x in test if x not in classes}, key=str)
    if unknown:
        raise ValueError(f"test contains classes absent from train: {unknown}")

    counts = {c: train.count(c) for c in classes}
    total = float(len(train))
    prior = {c: counts[c] / total for c in classes}

    log_loss = 0.0
    brier = 0.0
    for label in test:
        p_true = min(1.0 - epsilon, max(epsilon, prior[label]))
        log_loss -= log(p_true)
        brier += sum((prior[c] - (1.0 if c == label else 0.0)) ** 2 for c in classes)

    return {
        "classes": classes,
        "prior": prior,
        "log_loss": log_loss / len(test),
        "brier": brier / len(test),
        "n_train": len(train),
        "n_test": len(test),
        "benchmark": "TRAIN_EMPIRICAL_PRIOR",
    }


def compare_to_prior(*, model_log_loss: float, model_brier: float, prior_metrics: dict[str, object]) -> dict[str, object]:
    baseline_log_loss = float(prior_metrics["log_loss"])
    baseline_brier = float(prior_metrics["brier"])
    return {
        "model_log_loss": float(model_log_loss),
        "model_brier": float(model_brier),
        "baseline_log_loss": baseline_log_loss,
        "baseline_brier": baseline_brier,
        "log_loss_improvement": baseline_log_loss - float(model_log_loss),
        "brier_improvement": baseline_brier - float(model_brier),
        "beats_baseline_both": bool(float(model_log_loss) < baseline_log_loss and float(model_brier) < baseline_brier),
    }
