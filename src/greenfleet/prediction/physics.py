"""Physics baseline: the Admiralty relation as a fitted estimator.

Propulsion power scales with displacement and speed as

    P  ~  c * D^(2/3) * V^n,        n = 3 theoretically

which in logs is linear and therefore fittable by ordinary least squares per ship
type:

    log P = a_type + b * log D + n * log V

Two modes, both of which the benchmark needs (B-04 requires a physics-only baseline):

``theoretical``
    b and n are *fixed* at 2/3 and 3. Only the per-type intercept is fitted, so the
    model has one free parameter per ship type. This is the honest "physics-only"
    baseline.
``calibrated``
    b and n are fitted per ship type and reported. Comparing the fitted exponents
    against 2/3 and 3 is itself a validation result: if the data says n = 2.4 on
    annual means, that is the aggregation bias described in ``mrv/derive.py``, not
    a broken cube law.

Monotonicity (P-02) holds structurally: with n > 0, higher speed always predicts
higher power. The fit constrains n to stay positive, so the guarantee cannot be
lost to a bad fit on a sparse ship type.

Cold start (P-05): a ship type never seen in training falls back to pooled
coefficients fitted across all types, and ``predict`` reports which rows used the
fallback rather than failing or inventing a number.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin

logger = logging.getLogger(__name__)

__all__ = ["PhysicsBaseline", "TypeCoefficients", "THEORETICAL_DISPLACEMENT_EXPONENT",
           "THEORETICAL_SPEED_EXPONENT"]

THEORETICAL_DISPLACEMENT_EXPONENT = 2.0 / 3.0
THEORETICAL_SPEED_EXPONENT = 3.0

# Guard rails for the calibrated fit. A negative speed exponent would break P-02;
# an exponent far above 3 is overfitting a sparse, collinear group.
_SPEED_EXPONENT_BOUNDS = (0.5, 4.5)
_DISPLACEMENT_EXPONENT_BOUNDS = (0.0, 1.5)

FEATURE_SPEED = "mean_speed_kn"
FEATURE_SIZE = "size_proxy_t"
FEATURE_TYPE = "ship_type"


@dataclass(slots=True)
class TypeCoefficients:
    """Fitted coefficients for one ship type."""

    ship_type: str
    intercept: float
    displacement_exponent: float
    speed_exponent: float
    n_train: int
    pooled: bool = False
    """True when this type fell back to pooled coefficients (too few samples)."""

    def as_dict(self) -> dict[str, float | str | int | bool]:
        return {
            "ship_type": self.ship_type,
            "intercept": round(self.intercept, 6),
            "displacement_exponent": round(self.displacement_exponent, 4),
            "speed_exponent": round(self.speed_exponent, 4),
            "n_train": self.n_train,
            "pooled": self.pooled,
        }


class PhysicsBaseline(BaseEstimator, RegressorMixin):
    """Admiralty-relation power model, fitted in log space.

    Args:
        mode: ``"theoretical"`` fixes the exponents at 2/3 and 3; ``"calibrated"``
            fits them per ship type.
        min_group_size: ship types with fewer training rows than this use pooled
            coefficients instead of their own noisy fit.
        target: name of the target column, for error messages only.

    Attributes:
        coefficients_: per ship type, the fitted coefficients.
        pooled_: coefficients fitted across all types, used for cold start (P-05).
    """

    def __init__(
        self,
        mode: str = "calibrated",
        min_group_size: int = 30,
        target: str = "mean_power_kw",
    ):
        self.mode = mode
        self.min_group_size = min_group_size
        self.target = target

    # ---- internals --------------------------------------------------------

    @staticmethod
    def _design(frame: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        """Return ``(log_size, log_speed)``, both strictly finite."""
        size = pd.to_numeric(frame[FEATURE_SIZE], errors="coerce").to_numpy(dtype="float64")
        speed = pd.to_numeric(frame[FEATURE_SPEED], errors="coerce").to_numpy(dtype="float64")
        with np.errstate(divide="ignore", invalid="ignore"):
            return np.log(size), np.log(speed)

    def _fit_group(
        self, frame: pd.DataFrame, target: np.ndarray, ship_type: str
    ) -> TypeCoefficients | None:
        log_size, log_speed = self._design(frame)
        with np.errstate(divide="ignore", invalid="ignore"):
            log_power = np.log(target)
        valid = np.isfinite(log_size) & np.isfinite(log_speed) & np.isfinite(log_power)
        if valid.sum() < 3:
            return None

        log_size, log_speed, log_power = log_size[valid], log_speed[valid], log_power[valid]

        if self.mode == "theoretical":
            # Only the intercept is free: subtract the fixed physics terms.
            residual = (
                log_power
                - THEORETICAL_DISPLACEMENT_EXPONENT * log_size
                - THEORETICAL_SPEED_EXPONENT * log_speed
            )
            return TypeCoefficients(
                ship_type=ship_type,
                intercept=float(np.mean(residual)),
                displacement_exponent=THEORETICAL_DISPLACEMENT_EXPONENT,
                speed_exponent=THEORETICAL_SPEED_EXPONENT,
                n_train=int(valid.sum()),
            )

        design = np.column_stack([np.ones_like(log_size), log_size, log_speed])
        solution, *_ = np.linalg.lstsq(design, log_power, rcond=None)
        intercept, displacement_exponent, speed_exponent = solution

        # Clip to physically sensible ranges so a sparse or collinear group can
        # never produce a model that decreases in speed (P-02).
        clipped_speed = float(np.clip(speed_exponent, *_SPEED_EXPONENT_BOUNDS))
        clipped_size = float(np.clip(displacement_exponent, *_DISPLACEMENT_EXPONENT_BOUNDS))
        if clipped_speed != speed_exponent or clipped_size != displacement_exponent:
            logger.info(
                "%s: exponents clipped to physical bounds (size %.3f->%.3f, speed %.3f->%.3f)",
                ship_type,
                displacement_exponent,
                clipped_size,
                speed_exponent,
                clipped_speed,
            )
            # Re-fit the intercept so the clipped model stays unbiased in log space.
            intercept = float(
                np.mean(log_power - clipped_size * log_size - clipped_speed * log_speed)
            )

        return TypeCoefficients(
            ship_type=ship_type,
            intercept=float(intercept),
            displacement_exponent=clipped_size,
            speed_exponent=clipped_speed,
            n_train=int(valid.sum()),
        )

    # ---- sklearn API ------------------------------------------------------

    def fit(self, X: pd.DataFrame, y: np.ndarray | pd.Series) -> PhysicsBaseline:
        if self.mode not in {"theoretical", "calibrated"}:
            raise ValueError(f"mode must be 'theoretical' or 'calibrated', got {self.mode!r}")
        missing = {FEATURE_SPEED, FEATURE_SIZE, FEATURE_TYPE} - set(X.columns)
        if missing:
            raise ValueError(f"PhysicsBaseline.fit requires columns {sorted(missing)}")

        target = np.asarray(y, dtype="float64")
        if len(target) != len(X):
            raise ValueError(f"X has {len(X)} rows but y has {len(target)}")

        pooled = self._fit_group(X, target, ship_type="__pooled__")
        if pooled is None:
            raise ValueError(
                f"cannot fit: fewer than 3 rows with finite {self.target}, size and speed"
            )
        pooled.pooled = True
        self.pooled_ = pooled

        self.coefficients_: dict[str, TypeCoefficients] = {}
        for ship_type, group in X.groupby(FEATURE_TYPE, dropna=True, observed=True):
            label = str(ship_type)
            if len(group) < self.min_group_size:
                fallback = TypeCoefficients(
                    ship_type=label,
                    intercept=pooled.intercept,
                    displacement_exponent=pooled.displacement_exponent,
                    speed_exponent=pooled.speed_exponent,
                    n_train=len(group),
                    pooled=True,
                )
                self.coefficients_[label] = fallback
                continue
            fitted = self._fit_group(group, target[group.index.to_numpy()], label)
            self.coefficients_[label] = fitted if fitted is not None else pooled

        self.n_features_in_ = 3
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Predict mean power in kW."""
        if not hasattr(self, "coefficients_"):
            raise RuntimeError("PhysicsBaseline is not fitted; call fit first")

        log_size, log_speed = self._design(X)
        types = X[FEATURE_TYPE].astype("string").fillna("__unknown__").to_numpy()

        intercept = np.empty(len(X), dtype="float64")
        size_exp = np.empty(len(X), dtype="float64")
        speed_exp = np.empty(len(X), dtype="float64")
        used_fallback = np.zeros(len(X), dtype=bool)

        for index, ship_type in enumerate(types):
            coefficients = self.coefficients_.get(str(ship_type))
            if coefficients is None:
                coefficients = self.pooled_
                used_fallback[index] = True
            intercept[index] = coefficients.intercept
            size_exp[index] = coefficients.displacement_exponent
            speed_exp[index] = coefficients.speed_exponent

        self.used_cold_start_ = used_fallback
        if used_fallback.any():
            logger.info(
                "P-05 cold start: %d of %d rows used pooled coefficients "
                "(ship types unseen in training)",
                int(used_fallback.sum()),
                len(X),
            )

        with np.errstate(over="ignore", invalid="ignore"):
            prediction = np.exp(intercept + size_exp * log_size + speed_exp * log_speed)
        return prediction

    # ---- reporting --------------------------------------------------------

    def coefficient_table(self) -> pd.DataFrame:
        """Fitted coefficients per ship type, for the report and the deck."""
        if not hasattr(self, "coefficients_"):
            raise RuntimeError("PhysicsBaseline is not fitted; call fit first")
        rows = [c.as_dict() for c in self.coefficients_.values()]
        table = pd.DataFrame(rows).sort_values("n_train", ascending=False)
        return table.reset_index(drop=True)
