"""Regression metrics and the physics assertions.

P-12 requires MAE, RMSE, MAPE and R2 reported **per vessel type**, not only
overall, because an overall figure hides poor performance on minority classes. In
our own baseline the headline 20.5% MAPE concealed ro-pax at 37.3% with R2 ~ 0.03,
which is no better than that class's mean - so this is not a hypothetical concern.

Note on metric choice: these are regression metrics. Precision, recall and F1 are
classification metrics and are undefined for a continuous target - there is no
class to hit or miss.

P-10 adds interval metrics: a P10/P90 band is only useful if it has roughly the
coverage it claims, so ``interval_metrics`` reports empirical coverage against the
nominal 80%.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd

__all__ = [
    "RegressionMetrics",
    "regression_metrics",
    "metrics_by_group",
    "interval_metrics",
    "assert_speed_monotonicity",
    "assert_positive_at_zero_speed",
]


@dataclass(slots=True)
class RegressionMetrics:
    """Point-prediction metrics for one group."""

    n: int
    mae: float
    rmse: float
    mape: float
    r2: float
    bias: float
    """Mean signed error: positive means the model over-predicts."""

    median_ape: float
    """Median absolute percentage error - robust to the long tail MAPE suffers from."""

    def as_dict(self) -> dict[str, float | int]:
        return asdict(self)


def regression_metrics(
    y_true: np.ndarray | pd.Series,
    y_pred: np.ndarray | pd.Series,
) -> RegressionMetrics:
    """Compute metrics over finite, strictly-positive actuals.

    Rows with a non-finite prediction or a non-positive actual are excluded (MAPE
    is undefined at zero), and ``n`` reports how many rows actually counted, so an
    exclusion can never inflate a score invisibly.
    """
    actual = np.asarray(y_true, dtype="float64")
    predicted = np.asarray(y_pred, dtype="float64")
    if actual.shape != predicted.shape:
        raise ValueError(f"shape mismatch: {actual.shape} vs {predicted.shape}")

    usable = np.isfinite(actual) & np.isfinite(predicted) & (actual > 0)
    if usable.sum() == 0:
        raise ValueError("no rows with a finite prediction and a positive actual")
    actual, predicted = actual[usable], predicted[usable]

    error = predicted - actual
    absolute_pct = np.abs(error) / actual
    denominator = np.sum((actual - actual.mean()) ** 2)
    return RegressionMetrics(
        n=int(usable.sum()),
        mae=float(np.mean(np.abs(error))),
        rmse=float(np.sqrt(np.mean(error**2))),
        mape=float(100.0 * np.mean(absolute_pct)),
        r2=float(1.0 - np.sum(error**2) / denominator) if denominator > 0 else float("nan"),
        bias=float(np.mean(error)),
        median_ape=float(100.0 * np.median(absolute_pct)),
    )


def metrics_by_group(
    y_true: np.ndarray | pd.Series,
    y_pred: np.ndarray | pd.Series,
    groups: np.ndarray | pd.Series,
    min_count: int = 30,
) -> pd.DataFrame:
    """Per-group metrics plus an ``__overall__`` row (P-12).

    Groups smaller than ``min_count`` are still reported but flagged, rather than
    dropped: a class with 12 test rows is exactly the case a reader needs to know
    is unreliable.
    """
    frame = pd.DataFrame(
        {
            "group": np.asarray(groups),
            "actual": np.asarray(y_true, dtype="float64"),
            "predicted": np.asarray(y_pred, dtype="float64"),
        }
    )
    rows = []
    for group, part in frame.groupby("group", dropna=False):
        try:
            metrics = regression_metrics(part["actual"], part["predicted"])
        except ValueError:
            continue
        row = {"group": group, **metrics.as_dict()}
        row["reliable"] = metrics.n >= min_count
        rows.append(row)

    overall = regression_metrics(frame["actual"], frame["predicted"])
    rows.append({"group": "__overall__", **overall.as_dict(), "reliable": True})

    table = pd.DataFrame(rows)
    table = table.sort_values(
        ["group"], key=lambda s: s.eq("__overall__"), kind="mergesort"
    )
    return table.reset_index(drop=True)


def interval_metrics(
    y_true: np.ndarray | pd.Series,
    p10: np.ndarray | pd.Series,
    p90: np.ndarray | pd.Series,
    nominal: float = 0.80,
) -> dict[str, float]:
    """Empirical coverage and width of the prediction interval (P-10).

    An interval claiming 80% that actually covers 55% is worse than no interval,
    because the optimizer would treat the conservative bound as safe when it is not.
    """
    actual = np.asarray(y_true, dtype="float64")
    low = np.asarray(p10, dtype="float64")
    high = np.asarray(p90, dtype="float64")
    usable = np.isfinite(actual) & np.isfinite(low) & np.isfinite(high) & (actual > 0)
    if usable.sum() == 0:
        raise ValueError("no rows with finite actual and interval bounds")
    actual, low, high = actual[usable], low[usable], high[usable]

    inside = (actual >= low) & (actual <= high)
    return {
        "n": int(usable.sum()),
        "nominal_coverage": float(nominal),
        "empirical_coverage": float(np.mean(inside)),
        "coverage_error": float(np.mean(inside) - nominal),
        "mean_relative_width": float(np.mean((high - low) / actual)),
        "below_p10": float(np.mean(actual < low)),
        "above_p90": float(np.mean(actual > high)),
    }


def assert_speed_monotonicity(
    model,
    features: pd.DataFrame,
    speeds: np.ndarray | None = None,
    load_ratio: float = 1.0,
) -> pd.DataFrame:
    """P-02: power must never decrease as speed increases.

    Sweeps a speed range with everything else held fixed and checks the predicted
    power curve is non-decreasing for every row.

    Raises:
        AssertionError: naming the rows and speeds where power decreased.
    """
    grid = np.linspace(4.0, 24.0, 21) if speeds is None else np.asarray(speeds, dtype="float64")
    curves = np.column_stack(
        [model.predict_power_kw(features, speed_kn=float(v), load_ratio=load_ratio) for v in grid]
    )
    differences = np.diff(curves, axis=1)
    violations = differences < -1e-9
    if violations.any():
        row_index, step_index = np.nonzero(violations)
        examples = [
            f"row {int(r)} between {grid[s]:.1f} and {grid[s + 1]:.1f} kn"
            for r, s in list(zip(row_index, step_index, strict=True))[:5]
        ]
        raise AssertionError(
            f"P-02 violated: power decreased with speed in {violations.sum()} "
            f"cases, e.g. {examples}"
        )
    return pd.DataFrame(curves, columns=[f"{v:.1f}kn" for v in grid])


def assert_positive_at_zero_speed(model, features: pd.DataFrame) -> np.ndarray:
    """P-03: at zero speed, predict auxiliary load only - positive, never zero.

    Raises:
        AssertionError: if any row predicts non-positive power at rest.
    """
    power = model.predict_power_kw(features, speed_kn=0.0)
    if not np.all(np.isfinite(power)) or np.any(power <= 0):
        bad = int(np.sum(~(power > 0)))
        raise AssertionError(
            f"P-03 violated: {bad} row(s) predicted non-positive power at zero speed; "
            "a berthed vessel still runs auxiliaries"
        )
    return power
