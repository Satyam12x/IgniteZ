"""The energy model: imposed physics for the speed response, ML for the level.

Architecture, and why it is this way
------------------------------------
MRV cannot identify the speed-fuel relationship, because the reconstructed speed
and the reconstructed consumption share a measured column (docs/FINDINGS_PHASE1.md
section 2). Trying to fit the speed exponent returns ~1 instead of 3 and would give
the optimizer a slow-steaming sensitivity that is wrong by roughly 3x.

So the two halves come from different places:

* **Physics supplies the shape.** Total power splits into a propulsion term that
  scales as displacement^(2/3) x speed^3, and an auxiliary/hotel term that does not
  scale with speed at all::

      P(V, D) = P_ref * [ (1-a) * (D/D_ref)^(2/3) * (V/V_ref)^3  +  a ]

  where ``a`` is the class auxiliary share. Energy per nautical mile is P/V::

      E_nmi(V, D) = E_ref * [ (1-a) * (D/D_ref)^(2/3) * (V/V_ref)^2  +  a * V_ref/V ]

* **Data supplies the level.** ``E_ref`` - the ship's energy intensity at its class
  reference speed - is predicted by ML from leakage-free attributes only.

Three edge cases then hold *structurally*, not by test:

``P-02``
    ``P(V)`` is strictly increasing in V, since both bracket terms are
    non-negative and the first is increasing. Higher speed can never predict lower
    fuel, whatever the ML component does, because speed is not an ML input.
``P-03``
    ``P(0) = a * P_ref > 0``: a vessel at anchor or berth predicts auxiliary load
    only - not zero, never negative.
``The slow-steaming optimum``
    ``E_nmi`` is *not* monotone in V, and should not be: the ``a * V_ref/V`` term
    means crawling burns more energy per mile because the hotel load runs longer.
    So the curve has a genuine minimum, which is exactly the trade-off the problem
    statement is about, and it correctly predicts that a high-hotel-load vessel
    (cruise ship, a = 0.35) gains far less from slow steaming than a bulk carrier
    (a = 0.06).

The ML component additionally carries a monotone constraint on design efficiency.
EIV is emissions per tonne-mile, so a large ship has a *low* EIV but a *high*
absolute energy per mile: the relationship is negative. Measured within every
vessel class in the panel at -0.27 to -0.71, so the constraint is a confirmed
physical prior rather than an assumption.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge

from greenfleet.prediction.features import (
    LEAKAGE_FREE_FEATURES,
    VesselClassMap,
    load_vessel_classes,
    one_hot_classes,
)

logger = logging.getLogger(__name__)

__all__ = [
    "EnergyModel", "EnergyPrediction", "PHYSICS_EXPONENTS", "SpeedCurve",
    "physics_power_kw", "energy_per_nmi_from_power",
]

PHYSICS_EXPONENTS = {
    "speed_power": 3.0,        # P ~ V^3
    "displacement": 2.0 / 3.0,  # P ~ D^(2/3)
}

# Speeds outside this band around the class reference are flagged as extrapolation
# and get a widened interval (the spirit of P-06, applied to speed).
_EXTRAPOLATION_BAND = (0.6, 1.4)
# Log-interval widening per unit |log(V/V_ref)|.
_EXTRAPOLATION_WIDENING = 1.5
# Fallback multiplier for cold-start rows, used only when there are too few cold
# rows in training to calibrate a band directly. Prefer the calibrated bands: a
# fixed constant under-covered badly under temporal shift (69-71% against a
# nominal 80%), and an interval that claims 80% while delivering 69% is worse than
# no interval, because the optimizer treats the bound as safe when it is not.
_COLD_START_WIDENING = 1.8
_MIN_ROWS_FOR_BAND = 50

_MIN_SPEED_KN = 0.01


def physics_power_kw(
    reference_intensity_mj_per_nmi: np.ndarray | float,
    reference_speed_kn: np.ndarray | float,
    aux_share: np.ndarray | float,
    speed_kn: np.ndarray | float,
    load_ratio: np.ndarray | float = 1.0,
) -> np.ndarray:
    """Shaft + auxiliary power from the imposed physics, in kW.

    The single place the speed law lives. ``EnergyModel.predict`` and the
    optimizer's cached per-vessel curves both call this, so P-02 (strictly
    increasing in speed) and P-03 (positive at zero speed) hold identically in
    both paths by construction.
    """
    reference = np.asarray(reference_intensity_mj_per_nmi, dtype="float64")
    reference_speed = np.asarray(reference_speed_kn, dtype="float64")
    aux = np.asarray(aux_share, dtype="float64")
    speed = np.asarray(speed_kn, dtype="float64")
    load = np.asarray(load_ratio, dtype="float64")

    # The true ratio, un-clamped, so that zero speed gives *exactly* the
    # auxiliary load rather than auxiliary-plus-epsilon (P-03).
    speed_ratio = speed / reference_speed
    load_term = load ** PHYSICS_EXPONENTS["displacement"]
    reference_power = reference * reference_speed / 3.6
    return reference_power * (
        (1.0 - aux) * load_term * speed_ratio ** PHYSICS_EXPONENTS["speed_power"] + aux
    )


def energy_per_nmi_from_power(power_kw: np.ndarray, speed_kn: np.ndarray) -> np.ndarray:
    """MJ per nautical mile at a given power and speed; infinite at zero speed."""
    speed = np.asarray(speed_kn, dtype="float64")
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(
            speed > _MIN_SPEED_KN,
            np.asarray(power_kw) * 3.6 / np.maximum(speed, _MIN_SPEED_KN),
            np.inf,  # at zero speed, energy per mile is undefined, not zero
        )


@dataclass(frozen=True, slots=True)
class SpeedCurve:
    """One vessel's fitted operating curve, evaluable at any speed without ML.

    The learned part of the model (the vessel-specific intensity level) is fixed
    once per vessel; only the imposed physics varies with speed. An optimizer that
    evaluates thousands of speed settings per vessel therefore calls the residual
    model once and this curve thereafter - same numbers, three orders of magnitude
    faster.
    """

    vessel_class: str
    reference_intensity_mj_per_nmi: float
    reference_speed_kn: float
    aux_share: float
    cold_start: bool

    def power_kw(self, speed_kn: float, load_ratio: float = 1.0) -> float:
        return float(physics_power_kw(
            self.reference_intensity_mj_per_nmi, self.reference_speed_kn,
            self.aux_share, speed_kn, load_ratio,
        ))

    def energy_per_nmi_mj(self, speed_kn: float, load_ratio: float = 1.0) -> float:
        power = self.power_kw(speed_kn, load_ratio)
        return float(energy_per_nmi_from_power(power, speed_kn))


@dataclass(slots=True)
class EnergyPrediction:
    """Energy and power predictions with intervals and provenance flags."""

    energy_per_nmi_mj: np.ndarray
    """P50 energy per nautical mile."""

    p10_energy_per_nmi_mj: np.ndarray
    p90_energy_per_nmi_mj: np.ndarray
    power_kw: np.ndarray
    speed_kn: np.ndarray
    cold_start: np.ndarray
    """P-05: this row fell back to a class or global default."""

    extrapolating: np.ndarray
    """Speed is outside the class reference band; interval widened."""

    vessel_class: np.ndarray

    def as_frame(self) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "vessel_class": self.vessel_class,
                "speed_kn": self.speed_kn,
                "energy_per_nmi_mj": self.energy_per_nmi_mj,
                "p10": self.p10_energy_per_nmi_mj,
                "p90": self.p90_energy_per_nmi_mj,
                "power_kw": self.power_kw,
                "cold_start": self.cold_start,
                "extrapolating": self.extrapolating,
            }
        )


@dataclass
class _ResidualQuantiles:
    """Multiplicative prediction intervals from log residuals (P-10).

    Bands are held separately for warm rows (a design certificate is available) and
    cold-start rows, because the two have genuinely different error distributions -
    cold rows are both more biased and more dispersed. Calibrating them separately
    replaces a hand-set widening constant with a measured one.
    """

    by_class: dict[str, tuple[float, float]] = field(default_factory=dict)
    global_band: tuple[float, float] = (-0.3, 0.3)
    cold_band: tuple[float, float] | None = None

    def band(self, vessel_class: str) -> tuple[float, float]:
        return self.by_class.get(vessel_class, self.global_band)


class EnergyModel:
    """Physics-informed energy model.

    Args:
        residual_model: ``"xgboost"``, ``"ridge"`` or ``"class_mean"``. The latter
            two are the conventional baselines the benchmark needs (B-04).
        monotone: apply the design-efficiency monotone constraint (XGBoost only).
        seed: RNG seed; the same seed must give the same model (P-11).
        params: passed to the residual model.
    """

    def __init__(
        self,
        residual_model: str = "xgboost",
        monotone: bool = True,
        seed: int = 1000,
        classes: VesselClassMap | None = None,
        **params: object,
    ):
        self.residual_model = residual_model
        self.monotone = monotone
        self.seed = seed
        self.classes = classes or load_vessel_classes()
        self.params = params

    # ---- internals --------------------------------------------------------

    def _design_matrix(self, features: pd.DataFrame) -> pd.DataFrame:
        matrix = one_hot_classes(features, self.classes)
        if hasattr(self, "design_columns_"):
            # Train and test must have identical columns in identical order,
            # otherwise XGBoost silently reads the wrong feature.
            matrix = matrix.reindex(columns=self.design_columns_, fill_value=0)
        # A single hand-built row (one vessel from the optimizer or the API) with a
        # missing certificate arrives as an object column holding None; XGBoost
        # refuses object dtype, so coerce every column to float with NaN for missing.
        return matrix.apply(pd.to_numeric, errors="coerce").astype("float64")

    def _build_residual_model(self, n_columns: int):
        if self.residual_model == "ridge":
            return Ridge(alpha=float(self.params.get("alpha", 1.0)), random_state=None)
        if self.residual_model == "class_mean":
            return None
        if self.residual_model != "xgboost":
            raise ValueError(
                f"residual_model must be 'xgboost', 'ridge' or 'class_mean', "
                f"got {self.residual_model!r}"
            )
        from xgboost import XGBRegressor

        constraints = None
        if self.monotone:
            # Only design efficiency is constrained, and it is decreasing.
            vector = []
            for column in self.design_columns_:
                vector.append(-1 if column == "design_efficiency_gco2_per_t_nmi" else 0)
            constraints = "(" + ",".join(str(v) for v in vector) + ")"

        defaults = {
            "n_estimators": 400,
            "max_depth": 6,
            "learning_rate": 0.05,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "min_child_weight": 1.0,
            "reg_lambda": 1.0,
            "tree_method": "hist",
            "n_jobs": 4,
        }
        defaults.update({k: v for k, v in self.params.items() if k != "alpha"})
        return XGBRegressor(
            random_state=self.seed, monotone_constraints=constraints, **defaults
        )

    def _class_reference_speed(self, vessel_class: np.ndarray) -> np.ndarray:
        default = float(np.median(list(self.classes.typical_speed_kn.values())))
        return np.array(
            [self.classes.typical_speed_kn.get(str(c), default) for c in vessel_class],
            dtype="float64",
        )

    def _class_aux_share(self, vessel_class: np.ndarray) -> np.ndarray:
        default = float(np.median(list(self.classes.aux_load_share.values())))
        return np.array(
            [self.classes.aux_load_share.get(str(c), default) for c in vessel_class],
            dtype="float64",
        )

    # ---- fitting ----------------------------------------------------------

    def fit(
        self,
        features: pd.DataFrame,
        target: np.ndarray | pd.Series,
        oof_residuals: np.ndarray | None = None,
    ) -> EnergyModel:
        """Fit the reference-intensity model.

        Args:
            features: leakage-free features from ``build_features``.
            target: energy per nautical mile (MJ/nmi), strictly positive.
            oof_residuals: optional out-of-fold log residuals used to calibrate the
                prediction intervals. When omitted, in-sample residuals are used and
                the intervals are optimistic - the honest path is to pass
                cross-validated residuals.
        """
        missing = set(LEAKAGE_FREE_FEATURES) - set(features.columns)
        if missing:
            raise KeyError(f"features frame is missing {sorted(missing)}")

        values = np.asarray(target, dtype="float64")
        if len(values) != len(features):
            raise ValueError(f"features has {len(features)} rows, target has {len(values)}")

        valid = np.isfinite(values) & (values > 0)
        if valid.sum() < 10:
            raise ValueError("need at least 10 rows with a positive finite target")
        if not valid.all():
            logger.info("dropping %d rows with non-positive or missing target", (~valid).sum())

        frame = features.loc[valid].reset_index(drop=True)
        log_target = np.log(values[valid])

        self.design_columns_ = list(one_hot_classes(frame, self.classes).columns)
        matrix = self._design_matrix(frame)

        self.class_levels_ = (
            pd.Series(log_target).groupby(frame["vessel_class"].to_numpy()).mean().to_dict()
        )
        self.global_level_ = float(np.mean(log_target))

        estimator = self._build_residual_model(matrix.shape[1])
        if estimator is None:
            self.estimator_ = None
            fitted = frame["vessel_class"].map(self.class_levels_).fillna(
                self.global_level_
            ).to_numpy(dtype="float64")
        else:
            # Ridge cannot take NaN; XGBoost handles it natively and NaN here is
            # meaningful (no design certificate), so only impute for Ridge.
            if self.residual_model == "ridge":
                matrix = matrix.fillna(matrix.median(numeric_only=True)).fillna(0.0)
            estimator.fit(matrix, log_target)
            self.estimator_ = estimator
            fitted = estimator.predict(matrix)

        residuals = oof_residuals if oof_residuals is not None else (log_target - fitted)
        residuals = np.asarray(residuals, dtype="float64")
        self.residual_quantiles_ = _ResidualQuantiles()
        finite = np.isfinite(residuals)
        if finite.sum() >= 20:
            low, high = np.quantile(residuals[finite], [0.10, 0.90])
            self.residual_quantiles_.global_band = (float(low), float(high))

        cold_mask = self._cold_start_mask(frame)[: len(residuals)]
        warm_mask = ~cold_mask & finite
        cold_finite = cold_mask & finite

        # Calibrate the cold band directly rather than scaling the warm one.
        if cold_finite.sum() >= _MIN_ROWS_FOR_BAND:
            low, high = np.quantile(residuals[cold_finite], [0.10, 0.90])
            self.residual_quantiles_.cold_band = (float(low), float(high))
            logger.info(
                "P-10: cold-start band calibrated on %d rows -> (%.3f, %.3f) vs "
                "warm global (%.3f, %.3f)",
                int(cold_finite.sum()), low, high,
                *self.residual_quantiles_.global_band,
            )

        # Per-class bands use warm rows only; mixing cold rows in would inflate the
        # band for every ship of that class, not just the ones we know least about.
        by_class = pd.DataFrame(
            {
                "cls": frame["vessel_class"].to_numpy()[: len(residuals)],
                "r": residuals,
                "warm": warm_mask,
            }
        )
        by_class = by_class[by_class["warm"]].dropna(subset=["r"])
        for cls, group in by_class.groupby("cls"):
            if len(group) >= _MIN_ROWS_FOR_BAND:
                low, high = np.quantile(group["r"].to_numpy(), [0.10, 0.90])
                self.residual_quantiles_.by_class[str(cls)] = (float(low), float(high))

        self.n_train_ = int(valid.sum())
        return self

    # ---- prediction -------------------------------------------------------

    def predict_reference_intensity(self, features: pd.DataFrame) -> np.ndarray:
        """Energy per nautical mile at the class reference speed and nominal load."""
        if not hasattr(self, "design_columns_"):
            raise RuntimeError("EnergyModel is not fitted; call fit first")
        matrix = self._design_matrix(features)
        if self.estimator_ is None:
            log_level = (
                features["vessel_class"]
                .map(self.class_levels_)
                .fillna(self.global_level_)
                .to_numpy(dtype="float64")
            )
        else:
            if self.residual_model == "ridge":
                matrix = matrix.fillna(0.0)
            log_level = self.estimator_.predict(matrix)
        return np.exp(log_level)

    def _cold_start_mask(self, features: pd.DataFrame) -> np.ndarray:
        """P-05: rows the model has no real basis for."""
        unknown_class = ~features["vessel_class"].isin(self.class_levels_).to_numpy()
        no_design = features["design_efficiency_gco2_per_t_nmi"].isna().to_numpy()
        return unknown_class | no_design

    def predict(
        self,
        features: pd.DataFrame,
        speed_kn: np.ndarray | float | None = None,
        load_ratio: np.ndarray | float = 1.0,
    ) -> EnergyPrediction:
        """Predict energy per nautical mile and power at a given speed and load.

        Args:
            features: leakage-free features.
            speed_kn: speed to evaluate at. ``None`` uses each row's class
                reference speed, which reproduces the fitted operating point.
            load_ratio: displacement relative to the reference condition. 1.0 is
                nominal; 0.5 would be a light/ballast condition.
        """
        reference = self.predict_reference_intensity(features)
        vessel_class = features["vessel_class"].astype("string").to_numpy()
        reference_speed = self._class_reference_speed(vessel_class)
        aux_share = self._class_aux_share(vessel_class)

        speed = (
            reference_speed.copy()
            if speed_kn is None
            else np.broadcast_to(
                np.asarray(speed_kn, dtype="float64"), reference.shape
            ).astype("float64")
        )
        if np.any(speed < 0):
            raise ValueError("speed_kn must be non-negative")
        load = np.broadcast_to(
            np.asarray(load_ratio, dtype="float64"), reference.shape
        ).astype("float64")
        if np.any(load <= 0):
            raise ValueError("load_ratio must be strictly positive")

        speed_ratio = speed / reference_speed

        # Power: strictly increasing in speed, and positive at zero speed (P-02, P-03).
        power = physics_power_kw(reference, reference_speed, aux_share, speed, load)

        # Energy per nautical mile: has a genuine minimum in speed, as it should.
        energy_per_nmi = energy_per_nmi_from_power(power, speed)

        cold_start = self._cold_start_mask(features)
        extrapolating = (speed_ratio < _EXTRAPOLATION_BAND[0]) | (
            speed_ratio > _EXTRAPOLATION_BAND[1]
        )

        bands = np.array(
            [self.residual_quantiles_.band(str(c)) for c in vessel_class], dtype="float64"
        )
        # log(0) is undefined, so the widening term uses a floored ratio. This only
        # affects interval width, never the point estimate.
        floored_ratio = np.maximum(speed_ratio, _MIN_SPEED_KN / reference_speed)
        widening = 1.0 + _EXTRAPOLATION_WIDENING * np.abs(np.log(floored_ratio))

        # Cold-start rows get the WIDER of the measured band and the multiplier
        # applied to their class band. Measuring alone is not enough: cold rows in a
        # later period are a different population from the cold rows seen in
        # training (in MRV, the offshore fleet that only entered scope in 2024), and
        # a band fitted to the latter under-covers the former badly - 60% against a
        # nominal 80% when measured alone. An interval the optimizer will treat as a
        # safe bound must fail conservative, so take the wider of the two.
        if cold_start.any():
            cold_band = self.residual_quantiles_.cold_band
            scaled_low = bands[cold_start, 0] * _COLD_START_WIDENING
            scaled_high = bands[cold_start, 1] * _COLD_START_WIDENING
            if cold_band is not None:
                bands[cold_start, 0] = np.minimum(scaled_low, cold_band[0])
                bands[cold_start, 1] = np.maximum(scaled_high, cold_band[1])
            else:
                bands[cold_start, 0] = scaled_low
                bands[cold_start, 1] = scaled_high
        p10 = energy_per_nmi * np.exp(bands[:, 0] * widening)
        p90 = energy_per_nmi * np.exp(bands[:, 1] * widening)

        if cold_start.any():
            # Debug, not info: predict() is called once per fold per candidate during
            # tuning, and at info level this line buries the actual results.
            logger.debug(
                "P-05: %d of %d rows used a cold-start fallback; intervals widened %.1fx",
                int(cold_start.sum()),
                len(cold_start),
                _COLD_START_WIDENING,
            )

        return EnergyPrediction(
            energy_per_nmi_mj=energy_per_nmi,
            p10_energy_per_nmi_mj=p10,
            p90_energy_per_nmi_mj=p90,
            power_kw=power,
            speed_kn=speed,
            cold_start=cold_start,
            extrapolating=extrapolating,
            vessel_class=vessel_class,
        )

    def predict_power_kw(
        self,
        features: pd.DataFrame,
        speed_kn: np.ndarray | float,
        load_ratio: np.ndarray | float = 1.0,
    ) -> np.ndarray:
        """Shaft+auxiliary power in kW. Strictly increasing in speed (P-02)."""
        return self.predict(features, speed_kn=speed_kn, load_ratio=load_ratio).power_kw

    def speed_curves(self, features: pd.DataFrame) -> list[SpeedCurve]:
        """One :class:`SpeedCurve` per row: the residual model evaluated once.

        Exact agreement with :meth:`predict` at every speed is asserted by
        ``test_speed_curve_matches_full_prediction`` in ``tests/unit/test_api.py``.
        """
        reference = self.predict_reference_intensity(features)
        vessel_class = features["vessel_class"].astype("string").to_numpy()
        reference_speed = self._class_reference_speed(vessel_class)
        aux_share = self._class_aux_share(vessel_class)
        cold_start = self._cold_start_mask(features)
        return [
            SpeedCurve(
                vessel_class=str(vessel_class[i]),
                reference_intensity_mj_per_nmi=float(reference[i]),
                reference_speed_kn=float(reference_speed[i]),
                aux_share=float(aux_share[i]),
                cold_start=bool(cold_start[i]),
            )
            for i in range(len(reference))
        ]

    def optimal_speed_kn(
        self,
        features: pd.DataFrame,
        load_ratio: np.ndarray | float = 1.0,
    ) -> np.ndarray:
        """Speed minimising energy per nautical mile, in closed form.

        Minimising ``(1-a)*L*(V/Vr)^2 + a*Vr/V`` over V gives

            V* = Vr * ( a / (2*(1-a)*L) )^(1/3)

        This is the analytic slow-steaming optimum implied by the physics, and it
        is a useful sanity check on whatever the optimizer returns for a single
        vessel with no schedule constraints.
        """
        vessel_class = features["vessel_class"].astype("string").to_numpy()
        reference_speed = self._class_reference_speed(vessel_class)
        aux_share = self._class_aux_share(vessel_class)
        load = np.broadcast_to(
            np.asarray(load_ratio, dtype="float64"), reference_speed.shape
        ).astype("float64")
        ratio = aux_share / (2.0 * (1.0 - aux_share) * load ** PHYSICS_EXPONENTS["displacement"])
        return reference_speed * np.cbrt(ratio)
