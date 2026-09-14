"""Physics guarantees of the energy model.

Maps to P-02 (monotonicity in speed), P-03 (auxiliary load at rest), P-05 (cold
start), P-08 (energy not mass) and P-10 (prediction intervals).

These are the tests that matter most on stage: a judge can ask "what does your
model say if I slow the ship down?" and the answer must be right for structural
reasons, not because it happened to fit.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from greenfleet.config.loader import load_fuel_registry
from greenfleet.prediction.features import LEAKAGE_FREE_FEATURES, load_vessel_classes
from greenfleet.prediction.metrics import (
    assert_positive_at_zero_speed,
    assert_speed_monotonicity,
)
from greenfleet.prediction.model import EnergyModel

pytestmark = pytest.mark.edgecase


def _features(vessel_class: str = "bulk_carrier", n: int = 1, eiv: float = 10.0):
    """A minimal valid feature frame covering every declared leakage-free feature."""
    return pd.DataFrame(
        {
            "vessel_class": [vessel_class] * n,
            "design_efficiency_gco2_per_t_nmi": [eiv] * n,
            "has_design_efficiency": [1] * n,
            "efficiency_metric_eiv": [1] * n,
            "efficiency_metric_eedi": [0] * n,
            "efficiency_metric_eexi": [0] * n,
            "has_ice_class": [0] * n,
            "ice_class_ordinal": [0.0] * n,
            "ice_time_share": [0.0] * n,
            "reporting_period": [2024.0] * n,
            "time_at_sea_h": [5000.0] * n,
        }
    )


@pytest.fixture(scope="module")
def fitted_model() -> EnergyModel:
    """A model fitted on a small synthetic but physically shaped training set."""
    rng = np.random.default_rng(1000)
    classes = list(load_vessel_classes().typical_speed_kn)[:8]
    rows, target = [], []
    for cls in classes:
        for _ in range(80):
            eiv = float(rng.uniform(3, 60))
            has_ice = int(rng.random() < 0.2)
            rows.append(
                {
                    "vessel_class": cls,
                    "design_efficiency_gco2_per_t_nmi": eiv,
                    "has_design_efficiency": 1,
                    "efficiency_metric_eiv": 1,
                    "efficiency_metric_eedi": 0,
                    "efficiency_metric_eexi": 0,
                    "has_ice_class": has_ice,
                    "ice_class_ordinal": float(has_ice * rng.integers(1, 5)),
                    "ice_time_share": 0.0,
                    "reporting_period": float(rng.integers(2018, 2026)),
                    "time_at_sea_h": float(rng.uniform(1000, 7000)),
                }
            )
            # Lower EIV (bigger ship) -> higher absolute intensity, as measured.
            target.append(3000.0 / eiv * float(rng.lognormal(0, 0.1)))
    return EnergyModel(seed=1000, n_estimators=60, max_depth=4).fit(
        pd.DataFrame(rows), np.array(target)
    )


class TestMonotonicityP02:
    def test_power_never_decreases_with_speed(self, fitted_model):
        assert_speed_monotonicity(fitted_model, _features(n=5))

    @pytest.mark.parametrize(
        "vessel_class", ["bulk_carrier", "container_ship", "cruise_ship", "ropax"]
    )
    def test_holds_for_every_class_including_high_hotel_load(
        self, fitted_model, vessel_class
    ):
        assert_speed_monotonicity(fitted_model, _features(vessel_class))

    def test_holds_over_an_extreme_speed_range(self, fitted_model):
        assert_speed_monotonicity(
            fitted_model, _features(n=3), speeds=np.linspace(0.5, 40.0, 60)
        )

    def test_doubling_speed_raises_power_sharply(self, fitted_model):
        """The cube law is imposed, so 2x speed is close to 8x propulsion power."""
        features = _features("bulk_carrier")
        low = fitted_model.predict_power_kw(features, speed_kn=7.0)[0]
        high = fitted_model.predict_power_kw(features, speed_kn=14.0)[0]
        aux = load_vessel_classes().aux_load_share["bulk_carrier"]
        # With a small aux share the ratio should sit near 8, not near 2.
        assert high / low > 5.0, f"ratio {high / low:.2f} is too shallow for a cube law"
        assert aux < 0.1


class TestZeroSpeedP03:
    def test_power_at_rest_is_positive(self, fitted_model):
        power = assert_positive_at_zero_speed(fitted_model, _features(n=4))
        assert np.all(power > 0)

    def test_power_at_rest_equals_the_auxiliary_share(self, fitted_model):
        features = _features("bulk_carrier")
        classes = load_vessel_classes()
        reference_speed = classes.typical_speed_kn["bulk_carrier"]
        aux = classes.aux_load_share["bulk_carrier"]
        at_rest = fitted_model.predict_power_kw(features, speed_kn=0.0)[0]
        at_reference = fitted_model.predict_power_kw(features, speed_kn=reference_speed)[0]
        assert pytest.approx(aux, rel=1e-6) == at_rest / at_reference

    def test_cruise_ship_has_much_larger_load_at_rest(self, fitted_model):
        """A hotel-load-dominated vessel keeps burning fuel alongside."""
        bulk = fitted_model.predict(_features("bulk_carrier"), speed_kn=0.0)
        cruise = fitted_model.predict(_features("cruise_ship"), speed_kn=0.0)
        bulk_share = load_vessel_classes().aux_load_share["bulk_carrier"]
        cruise_share = load_vessel_classes().aux_load_share["cruise_ship"]
        assert cruise_share > bulk_share * 3
        assert np.isinf(bulk.energy_per_nmi_mj[0]), "energy per mile at rest is undefined"
        assert np.isinf(cruise.energy_per_nmi_mj[0])


class TestSlowSteamingBehaviour:
    def test_energy_per_mile_has_an_interior_minimum(self, fitted_model):
        """Not monotone, and correctly so: crawling runs the hotel load longer."""
        features = _features("cruise_ship")
        speeds = np.linspace(4.0, 30.0, 80)
        curve = np.array(
            [fitted_model.predict(features, speed_kn=float(v)).energy_per_nmi_mj[0]
             for v in speeds]
        )
        best = int(np.argmin(curve))
        assert 0 < best < len(speeds) - 1, "minimum should be interior, not at a bound"

    def test_analytic_optimum_matches_the_numeric_minimum(self, fitted_model):
        features = _features("cruise_ship")
        analytic = fitted_model.optimal_speed_kn(features)[0]
        speeds = np.linspace(2.0, 30.0, 400)
        curve = np.array(
            [fitted_model.predict(features, speed_kn=float(v)).energy_per_nmi_mj[0]
             for v in speeds]
        )
        assert speeds[int(np.argmin(curve))] == pytest.approx(analytic, rel=0.05)

    def test_low_hotel_load_gains_more_from_slowing_down(self, fitted_model):
        """The key differentiating insight: aux share governs slow-steaming payoff."""
        def saving(vessel_class: str) -> float:
            features = _features(vessel_class)
            fast = fitted_model.predict(features, speed_kn=14.0).energy_per_nmi_mj[0]
            slow = fitted_model.predict(features, speed_kn=12.0).energy_per_nmi_mj[0]
            return 1.0 - slow / fast

        assert saving("bulk_carrier") > saving("ropax") > saving("cruise_ship")


class TestLoadResponseP04:
    def test_heavier_load_needs_more_power(self, fitted_model):
        features = _features()
        light = fitted_model.predict_power_kw(features, speed_kn=14.0, load_ratio=0.5)[0]
        heavy = fitted_model.predict_power_kw(features, speed_kn=14.0, load_ratio=1.0)[0]
        assert heavy > light

    def test_load_exponent_is_exactly_two_thirds(self, fitted_model):
        """Displacement enters as D^(2/3), so 8x load is exactly 4x *propulsion* power.

        Isolates the propulsion term by subtracting the auxiliary load, which is
        recoverable as the power at zero speed.
        """
        features = _features("bulk_carrier")
        reference_speed = load_vessel_classes().typical_speed_kn["bulk_carrier"]
        auxiliary = fitted_model.predict_power_kw(features, speed_kn=0.0)[0]
        at_one = fitted_model.predict_power_kw(
            features, speed_kn=reference_speed, load_ratio=1.0
        )[0]
        at_eight = fitted_model.predict_power_kw(
            features, speed_kn=reference_speed, load_ratio=8.0
        )[0]
        propulsion_ratio = (at_eight - auxiliary) / (at_one - auxiliary)
        assert propulsion_ratio == pytest.approx(8 ** (2 / 3), rel=1e-9)
        assert propulsion_ratio == pytest.approx(4.0, rel=1e-9)

    def test_speed_exponent_is_exactly_three(self, fitted_model):
        """Same isolation, for the imposed cube law: 2x speed is exactly 8x propulsion."""
        features = _features("bulk_carrier")
        auxiliary = fitted_model.predict_power_kw(features, speed_kn=0.0)[0]
        at_seven = fitted_model.predict_power_kw(features, speed_kn=7.0)[0]
        at_fourteen = fitted_model.predict_power_kw(features, speed_kn=14.0)[0]
        ratio = (at_fourteen - auxiliary) / (at_seven - auxiliary)
        assert ratio == pytest.approx(8.0, rel=1e-9)

    def test_non_positive_load_rejected(self, fitted_model):
        with pytest.raises(ValueError, match="load_ratio"):
            fitted_model.predict(_features(), speed_kn=14.0, load_ratio=0.0)

    def test_negative_speed_rejected(self, fitted_model):
        with pytest.raises(ValueError, match="non-negative"):
            fitted_model.predict(_features(), speed_kn=-1.0)


class TestColdStartP05:
    def test_unknown_class_does_not_crash(self, fitted_model):
        prediction = fitted_model.predict(_features("harbour_tug"))
        assert np.isfinite(prediction.energy_per_nmi_mj).all()
        assert prediction.energy_per_nmi_mj[0] > 0

    def test_unknown_class_is_flagged_as_cold_start(self, fitted_model):
        assert fitted_model.predict(_features("harbour_tug")).cold_start[0]

    def test_missing_design_efficiency_is_flagged(self, fitted_model):
        features = _features()
        features["design_efficiency_gco2_per_t_nmi"] = np.nan
        features["has_design_efficiency"] = 0
        assert fitted_model.predict(features).cold_start[0]

    def test_cold_start_widens_the_interval(self, fitted_model):
        warm = fitted_model.predict(_features("bulk_carrier"))
        cold = _features("bulk_carrier")
        cold["design_efficiency_gco2_per_t_nmi"] = np.nan
        cold_prediction = fitted_model.predict(cold)
        warm_width = warm.p90_energy_per_nmi_mj[0] / warm.p10_energy_per_nmi_mj[0]
        cold_width = (
            cold_prediction.p90_energy_per_nmi_mj[0]
            / cold_prediction.p10_energy_per_nmi_mj[0]
        )
        assert cold_width > warm_width


class TestIntervalsP10:
    def test_interval_brackets_the_point_estimate(self, fitted_model):
        prediction = fitted_model.predict(_features(n=6))
        assert np.all(prediction.p10_energy_per_nmi_mj <= prediction.energy_per_nmi_mj)
        assert np.all(prediction.energy_per_nmi_mj <= prediction.p90_energy_per_nmi_mj)

    def test_extrapolation_widens_the_interval(self, fitted_model):
        features = _features("bulk_carrier")
        reference = load_vessel_classes().typical_speed_kn["bulk_carrier"]
        at_reference = fitted_model.predict(features, speed_kn=reference)
        far = fitted_model.predict(features, speed_kn=reference * 2.2)

        def relative_width(prediction):
            return (
                prediction.p90_energy_per_nmi_mj[0] - prediction.p10_energy_per_nmi_mj[0]
            ) / prediction.energy_per_nmi_mj[0]

        assert relative_width(far) > relative_width(at_reference)

    def test_far_speed_is_flagged_as_extrapolating(self, fitted_model):
        reference = load_vessel_classes().typical_speed_kn["bulk_carrier"]
        assert fitted_model.predict(
            _features("bulk_carrier"), speed_kn=reference * 3
        ).extrapolating[0]

    def test_reference_speed_is_not_extrapolating(self, fitted_model):
        reference = load_vessel_classes().typical_speed_kn["bulk_carrier"]
        assert not fitted_model.predict(
            _features("bulk_carrier"), speed_kn=reference
        ).extrapolating[0]


class TestEnergyNotMassP08:
    def test_same_energy_gives_different_mass_per_fuel(self, fitted_model):
        """P-08: predict energy once, convert per fuel. Methanol is ~2x MGO by mass."""
        registry = load_fuel_registry()
        prediction = fitted_model.predict(_features("bulk_carrier"), speed_kn=14.0)
        energy = float(prediction.energy_per_nmi_mj[0])
        mgo = registry.energy_to_mass_tonnes(energy, "mgo")
        methanol = registry.energy_to_mass_tonnes(energy, "methanol")
        assert methanol / mgo == pytest.approx(42.7 / 19.9, rel=1e-6)

    def test_hydrogen_is_lightest_but_bulkiest(self, fitted_model):
        registry = load_fuel_registry()
        energy = float(
            fitted_model.predict(_features(), speed_kn=14.0).energy_per_nmi_mj[0]
        )
        assert registry.energy_to_mass_tonnes(energy, "hydrogen") < (
            registry.energy_to_mass_tonnes(energy, "mgo")
        )
        assert registry.tank_volume_m3(energy, "hydrogen") > (
            registry.tank_volume_m3(energy, "mgo") * 10
        )


class TestReproducibilityP11:
    def test_same_seed_same_predictions(self):
        rng = np.random.default_rng(5)
        frame = _features("bulk_carrier", n=100)
        frame["design_efficiency_gco2_per_t_nmi"] = rng.uniform(5, 50, 100)
        frame["time_at_sea_h"] = rng.uniform(1000, 7000, 100)
        target = rng.lognormal(7, 0.3, 100)
        a = EnergyModel(seed=42, n_estimators=40).fit(frame, target).predict(frame)
        b = EnergyModel(seed=42, n_estimators=40).fit(frame, target).predict(frame)
        np.testing.assert_array_equal(a.energy_per_nmi_mj, b.energy_per_nmi_mj)

    def test_missing_feature_column_raises(self, fitted_model):
        features = _features().drop(columns=["time_at_sea_h"])
        with pytest.raises(KeyError):
            EnergyModel(seed=1).fit(features, np.array([1.0]))

    def test_every_declared_feature_is_present_in_the_test_frame(self):
        """Guards against a new feature being declared but untested."""
        assert set(LEAKAGE_FREE_FEATURES) <= set(_features().columns)

    def test_no_declared_feature_is_contaminated(self):
        from greenfleet.prediction.features import CONTAMINATED_BY_TARGET

        assert not set(LEAKAGE_FREE_FEATURES) & CONTAMINATED_BY_TARGET
