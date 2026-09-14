"""Hyperparameter tuning for the energy model, plus the P-13 ablation.

Two things have to be right here or the whole result is worthless:

**The tuning loop must not leak.** Hyperparameters selected on folds that share
vessels with the evaluation set leak just as surely as a bad final split (P-01), so
every fold is a *grouped* fold on IMO.

**The comparison must be budget-matched.** QPSO is only "better" if it beats random
search given the same number of objective evaluations (B-01). ``run_ablation`` takes
the evaluation count QPSO actually used and hands random search exactly that, then
reports both plus the untuned default - including when the quantum-inspired tuner
does not win (B-07, P-13).
"""

from __future__ import annotations

import json
import logging
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from greenfleet.prediction.metrics import regression_metrics
from greenfleet.prediction.model import EnergyModel
from greenfleet.prediction.qpso import (
    Parameter,
    QPSOResult,
    SearchSpace,
    qpso_minimise,
    random_search,
)
from greenfleet.prediction.splits import grouped_folds
from greenfleet.reproducibility import RunContext

logger = logging.getLogger(__name__)

__all__ = ["DEFAULT_SPACE", "make_cv_objective", "run_ablation", "AblationResult"]

# Search space for the XGBoost residual model. Bounds are deliberately wide enough
# that the default configuration is not sitting at the centre of the space.
DEFAULT_SPACE = SearchSpace(
    (
        Parameter("n_estimators", 100, 900, "int"),
        Parameter("max_depth", 3, 10, "int"),
        Parameter("learning_rate", 0.01, 0.30, "log"),
        Parameter("subsample", 0.5, 1.0, "float"),
        Parameter("colsample_bytree", 0.5, 1.0, "float"),
        Parameter("min_child_weight", 0.5, 20.0, "log"),
        Parameter("reg_lambda", 0.1, 20.0, "log"),
    )
)

DEFAULT_PARAMS = {
    "n_estimators": 400,
    "max_depth": 6,
    "learning_rate": 0.05,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "min_child_weight": 1.0,
    "reg_lambda": 1.0,
}


def make_cv_objective(
    features: pd.DataFrame,
    target: np.ndarray,
    groups: pd.Series,
    n_folds: int = 3,
    seed: int = 1000,
    monotone: bool = True,
    metric: str = "mape",
) -> Callable[..., float]:
    """Build a grouped-CV objective returning the mean metric across folds.

    Folds are computed once and reused for every candidate, so all candidates and
    both algorithms see identical data - a difference in score is then a difference
    in the configuration, not in the split.
    """
    frame = features.reset_index(drop=True)
    values = np.asarray(target, dtype="float64")
    grouped = pd.DataFrame({"imo": groups.reset_index(drop=True)})
    folds = grouped_folds(grouped, n_splits=n_folds)

    def objective(**params: float | int) -> float:
        scores = []
        for train_idx, test_idx in folds:
            model = EnergyModel(
                residual_model="xgboost", monotone=monotone, seed=seed, **params
            )
            model.fit(frame.iloc[train_idx], values[train_idx])
            predicted = model.predict(frame.iloc[test_idx]).energy_per_nmi_mj
            result = regression_metrics(values[test_idx], predicted)
            scores.append(getattr(result, metric))
        return float(np.mean(scores))

    objective.n_folds = n_folds  # type: ignore[attr-defined]
    return objective


@dataclass
class AblationResult:
    """P-13: the quantum-inspired tuner against budget-matched alternatives."""

    qpso: QPSOResult
    random: QPSOResult
    default_score: float
    metric: str = "mape"
    n_folds: int = 3
    notes: list[str] = field(default_factory=list)

    @property
    def budget(self) -> int:
        return self.qpso.n_evaluations

    @property
    def qpso_beats_random(self) -> bool:
        return self.qpso.best_score < self.random.best_score

    @property
    def qpso_beats_default(self) -> bool:
        return self.qpso.best_score < self.default_score

    def verdict(self) -> str:
        """A sentence that is true whichever way the result went."""
        gap = self.random.best_score - self.qpso.best_score
        relative = 100.0 * gap / self.random.best_score if self.random.best_score else 0.0
        if self.qpso_beats_random:
            return (
                f"QPSO improved CV {self.metric} to {self.qpso.best_score:.3f} vs "
                f"{self.random.best_score:.3f} for budget-matched random search "
                f"({relative:+.1f}%), and {self.default_score:.3f} untuned."
            )
        return (
            f"QPSO did NOT beat random search at equal budget "
            f"({self.qpso.best_score:.3f} vs {self.random.best_score:.3f}, "
            f"{relative:+.1f}%). Untuned default was {self.default_score:.3f}. "
            "Reported as-is."
        )

    def as_dict(self) -> dict[str, object]:
        return {
            "metric": self.metric,
            "n_folds": self.n_folds,
            "budget_evaluations": self.budget,
            "qpso": {
                "best_score": self.qpso.best_score,
                "best_params": self.qpso.best_params,
                "diversity_final": self.qpso.diversity[-1] if self.qpso.diversity else None,
                "converged_early": self.qpso.converged_early(),
                "history": self.qpso.history,
            },
            "random_search": {
                "best_score": self.random.best_score,
                "best_params": self.random.best_params,
                "history": self.random.history,
            },
            "default": {"best_score": self.default_score, "params": DEFAULT_PARAMS},
            "qpso_beats_random": self.qpso_beats_random,
            "qpso_beats_default": self.qpso_beats_default,
            "verdict": self.verdict(),
            "notes": self.notes,
        }


def run_ablation(
    features: pd.DataFrame,
    target: np.ndarray,
    groups: pd.Series,
    space: SearchSpace = DEFAULT_SPACE,
    n_particles: int = 8,
    n_iterations: int = 6,
    n_folds: int = 3,
    seed: int = 1000,
    metric: str = "mape",
    output_path: Path | str | None = None,
) -> AblationResult:
    """Run QPSO, budget-matched random search, and the untuned default."""
    objective = make_cv_objective(
        features, target, groups, n_folds=n_folds, seed=seed, metric=metric
    )

    logger.info("evaluating untuned default configuration")
    default_score = objective(**DEFAULT_PARAMS)
    logger.info("default CV %s = %.4f", metric, default_score)

    logger.info("running QPSO (%d particles x %d iterations)", n_particles, n_iterations)
    qpso_result = qpso_minimise(
        objective, space, n_particles=n_particles, n_iterations=n_iterations, seed=seed
    )

    logger.info("running budget-matched random search (%d evaluations)",
                qpso_result.n_evaluations)
    random_result = random_search(
        objective, space, n_evaluations=qpso_result.n_evaluations, seed=seed
    )

    result = AblationResult(
        qpso=qpso_result,
        random=random_result,
        default_score=default_score,
        metric=metric,
        n_folds=n_folds,
    )
    if qpso_result.converged_early():
        result.notes.append(
            "Q-03: swarm diversity collapsed in the first half of the QPSO run."
        )
    logger.info(result.verdict())

    if output_path is not None:
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "run": RunContext(
                seed=seed,
                stage="prediction.tune",
                inputs={
                    "n_rows": int(len(features)),
                    "n_particles": n_particles,
                    "n_iterations": n_iterations,
                    "n_folds": n_folds,
                    "metric": metric,
                },
            ).as_dict(),
            "ablation": result.as_dict(),
        }
        path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
        logger.info("wrote %s", path)

    return result
