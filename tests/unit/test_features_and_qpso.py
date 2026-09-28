"""Feature safety, vocabulary and quantum-inspired search tests.

Maps to D-12 (vocabulary), P-01 (contamination guard), P-13 (ablation fairness),
Q-03 (premature convergence), Q-06 (determinism) and B-01/B-02 (fair budgets,
enough runs).
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from greenfleet.prediction.features import (
    CONTAMINATED_BY_TARGET,
    TARGET,
    ContaminatedFeatureError,
    build_features,
    build_target,
    load_vessel_classes,
    one_hot_classes,
)
from greenfleet.prediction.metrics import (
    interval_metrics,
    metrics_by_group,
    regression_metrics,
)
from greenfleet.prediction.qpso import (
    Parameter,
    SearchSpace,
    qpso_minimise,
    random_search,
    summarise_runs,
)

pytestmark = pytest.mark.edgecase


@pytest.fixture
def raw_panel() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "imo": [1, 2, 3, 4],
            "ship_type": [
                "Bulk carrier",
                "bulker",
                "Passenger ship (Cruise Passenger ship)",
                "Totally Unknown Vessel",
            ],
            "reporting_period": [2018, 2024, 2024, 2025],
            "technical_efficiency_gco2_per_t_nmi": [10.0, 20.0, np.nan, 5.0],
            "ice_class": ["IA", None, None, None],
            "time_at_sea_h": [4000.0, 5000.0, 6000.0, 3000.0],
            "inferred_lhv_mj_per_kg": [40.2, 42.7, 40.5, 48.0],
            "fuel_per_distance_kg_per_nmi": [100.0, 200.0, 300.0, 50.0],
        }
    )


class TestVocabularyD12:
    def test_aliases_map_to_one_canonical_class(self, raw_panel):
        features = build_features(raw_panel)
        assert features.vessel_class.iloc[0] == "bulk_carrier"
        assert features.vessel_class.iloc[1] == "bulk_carrier"

    def test_later_year_subtype_maps_to_its_own_class(self, raw_panel):
        """EMSA split cruise ships out from 2023; they keep a distinct class."""
        assert build_features(raw_panel).vessel_class.iloc[2] == "cruise_ship"

    def test_unknown_label_falls_back_with_a_warning(self, raw_panel, caplog):
        with caplog.at_level("WARNING"):
            features = build_features(raw_panel)
        assert features.vessel_class.iloc[3] == "other"
        assert "unknown vessel class" in caplog.text.lower()

    def test_tug_classes_are_marked_out_of_mrv_scope(self):
        classes = load_vessel_classes()
        assert classes.in_mrv_scope["harbour_tug"] is False
        assert classes.in_mrv_scope["bulk_carrier"] is True

    def test_one_hot_width_is_fixed_by_config_not_data(self, raw_panel):
        """Train and test matrices must match even when a class is absent."""
        classes = load_vessel_classes()
        full = one_hot_classes(build_features(raw_panel), classes)
        subset = one_hot_classes(build_features(raw_panel.head(1)), classes)
        assert list(full.columns) == list(subset.columns)
        assert sum(c.startswith("class_") for c in full.columns) == len(
            classes.typical_speed_kn
        )


class TestContaminationGuardP01:
    @pytest.mark.parametrize(
        "column",
        ["mean_speed_kn", "cargo_carried_t", "mean_power_kw", "distance_nmi",
         "fuel_per_distance_kg_per_nmi"],
    )
    def test_contaminated_features_are_rejected(self, raw_panel, column):
        frame = raw_panel.assign(**{column: 1.0})
        with pytest.raises(ContaminatedFeatureError, match="share a component"):
            build_features(frame, extra_features=(column,))

    def test_error_says_how_to_fix_it(self, raw_panel):
        frame = raw_panel.assign(mean_speed_kn=1.0)
        with pytest.raises(ContaminatedFeatureError, match="Leave them out of extra_features"):
            build_features(frame, extra_features=("mean_speed_kn",))

    def test_the_target_itself_is_on_the_contaminated_list(self):
        assert TARGET in CONTAMINATED_BY_TARGET

    def test_a_safe_extra_feature_is_allowed(self, raw_panel):
        frame = raw_panel.assign(port_calls=[3, 4, 5, 6])
        assert "port_calls" in build_features(frame, extra_features=("port_calls",)).columns

    def test_unknown_extra_feature_raises(self, raw_panel):
        with pytest.raises(KeyError, match="not in the frame"):
            build_features(raw_panel, extra_features=("does_not_exist",))


class TestTargetP08:
    def test_target_is_energy_not_mass(self, raw_panel):
        target = build_target(raw_panel)
        expected = 40.2 * 100.0
        assert target.iloc[0] == pytest.approx(expected)
        assert target.name == TARGET

    def test_missing_inputs_raise(self):
        with pytest.raises(KeyError, match="derive_physics"):
            build_target(pd.DataFrame({"x": [1]}))


class TestMetrics:
    def test_perfect_prediction(self):
        y = np.array([1.0, 2.0, 3.0])
        result = regression_metrics(y, y)
        assert result.mae == 0.0
        assert result.mape == 0.0
        assert result.r2 == pytest.approx(1.0)

    def test_mape_matches_hand_calculation(self):
        result = regression_metrics(np.array([100.0, 200.0]), np.array([110.0, 180.0]))
        assert result.mape == pytest.approx(10.0)

    def test_bias_sign_is_positive_when_over_predicting(self):
        assert regression_metrics(np.array([100.0]), np.array([120.0])).bias == 20.0

    def test_zero_actual_is_excluded_because_mape_is_undefined(self):
        result = regression_metrics(np.array([0.0, 100.0]), np.array([5.0, 100.0]))
        assert result.n == 1

    def test_no_usable_rows_raises(self):
        with pytest.raises(ValueError, match="no rows"):
            regression_metrics(np.array([0.0]), np.array([np.nan]))

    def test_shape_mismatch_raises(self):
        with pytest.raises(ValueError, match="shape mismatch"):
            regression_metrics(np.array([1.0, 2.0]), np.array([1.0]))

    def test_per_group_table_includes_overall_row(self):
        y = np.array([100.0, 110.0, 200.0, 210.0])
        p = np.array([105.0, 108.0, 190.0, 215.0])
        groups = np.array(["a", "a", "b", "b"])
        table = metrics_by_group(y, p, groups, min_count=2)
        assert "__overall__" in table.group.values
        assert table.group.iloc[-1] == "__overall__"
        assert set(table.group) == {"a", "b", "__overall__"}

    def test_small_group_is_flagged_not_dropped(self):
        y = np.array([100.0, 110.0, 200.0])
        p = np.array([105.0, 108.0, 190.0])
        table = metrics_by_group(y, p, np.array(["a", "a", "b"]), min_count=2)
        assert not table.set_index("group").loc["b", "reliable"]
        assert table.set_index("group").loc["a", "reliable"]

    def test_interval_coverage_is_measured(self):
        y = np.array([100.0, 100.0, 100.0, 100.0])
        low = np.array([90.0, 90.0, 90.0, 200.0])
        high = np.array([110.0, 110.0, 110.0, 300.0])
        result = interval_metrics(y, low, high)
        assert result["empirical_coverage"] == pytest.approx(0.75)
        assert result["coverage_error"] == pytest.approx(-0.05)


class TestSearchSpace:
    def test_log_parameter_round_trips(self):
        parameter = Parameter("lr", 0.01, 0.3, "log")
        assert parameter.from_internal(parameter.to_internal(0.05)) == pytest.approx(0.05)

    def test_int_parameter_is_rounded(self):
        assert Parameter("n", 1, 10, "int").from_internal(4.6) == 5

    def test_log_parameter_needs_positive_low(self):
        with pytest.raises(ValueError, match="strictly positive"):
            Parameter("bad", 0.0, 1.0, "log")

    def test_inverted_bounds_rejected(self):
        with pytest.raises(ValueError, match="must exceed"):
            Parameter("bad", 5.0, 1.0)

    def test_bad_kind_rejected(self):
        with pytest.raises(ValueError, match="kind must be"):
            Parameter("bad", 0.0, 1.0, "categorical")

    def test_duplicate_names_rejected(self):
        with pytest.raises(ValueError, match="duplicate"):
            SearchSpace((Parameter("x", 0, 1), Parameter("x", 0, 1)))

    def test_empty_space_rejected(self):
        with pytest.raises(ValueError, match="empty"):
            SearchSpace(())

    def test_decode_clips_to_bounds(self):
        space = SearchSpace((Parameter("x", 0.0, 1.0),))
        assert space.decode(np.array([5.0]))["x"] == 1.0


def _sphere(**kwargs: float) -> float:
    return float(sum(v**2 for v in kwargs.values()))


class TestQPSO:
    @pytest.fixture
    def space(self) -> SearchSpace:
        return SearchSpace(tuple(Parameter(f"x{i}", -5.0, 5.0) for i in range(3)))

    def test_finds_the_optimum_of_a_convex_function(self, space):
        result = qpso_minimise(_sphere, space, n_particles=12, n_iterations=25, seed=1000)
        assert result.best_score < 0.05

    def test_is_deterministic_for_a_fixed_seed(self, space):
        a = qpso_minimise(_sphere, space, n_particles=8, n_iterations=10, seed=7)
        b = qpso_minimise(_sphere, space, n_particles=8, n_iterations=10, seed=7)
        assert a.best_score == b.best_score
        assert a.best_params == b.best_params

    def test_different_seeds_give_a_distribution(self, space):
        """Q-06: benchmarking needs a distribution, so seeds must actually differ."""
        scores = [
            qpso_minimise(_sphere, space, n_particles=8, n_iterations=8, seed=s).best_score
            for s in (1, 2, 3, 4, 5)
        ]
        assert len(set(scores)) > 1

    def test_budget_is_exactly_as_advertised(self, space):
        result = qpso_minimise(_sphere, space, n_particles=6, n_iterations=5, seed=1)
        assert result.n_evaluations == 6 + 6 * 5

    def test_diversity_is_tracked_every_iteration(self, space):
        result = qpso_minimise(_sphere, space, n_particles=8, n_iterations=12, seed=1)
        assert len(result.diversity) == 12
        assert all(d >= 0 for d in result.diversity)

    def test_diversity_decreases_as_the_swarm_converges(self, space):
        result = qpso_minimise(_sphere, space, n_particles=12, n_iterations=30, seed=1)
        assert result.diversity[-1] < result.diversity[0]

    def test_premature_convergence_detector_q03(self, space):
        result = qpso_minimise(_sphere, space, n_particles=10, n_iterations=20, seed=1)
        assert isinstance(result.converged_early(), bool)
        # A healthy run should not collapse in the first half.
        assert not result.converged_early(threshold=1e-9)

    def test_history_is_monotonically_improving(self, space):
        history = qpso_minimise(_sphere, space, n_particles=8, n_iterations=15, seed=1).history
        assert all(b <= a for a, b in zip(history, history[1:], strict=False))

    def test_a_crashing_objective_does_not_kill_the_search(self, space):
        def flaky(**kwargs: float) -> float:
            if kwargs["x0"] > 0:
                raise RuntimeError("simulated failure")
            return _sphere(**kwargs)

        result = qpso_minimise(flaky, space, n_particles=8, n_iterations=8, seed=1)
        assert math.isfinite(result.best_score)

    def test_nan_objective_cannot_win(self, space):
        result = qpso_minimise(lambda **kw: float("nan"), space, n_particles=4,
                               n_iterations=3, seed=1)
        assert math.isinf(result.best_score)

    @pytest.mark.parametrize(("start", "end"), [(2.0, 0.5), (0.5, 1.0), (1.0, 0.0)])
    def test_invalid_beta_rejected(self, space, start, end):
        with pytest.raises(ValueError, match="beta"):
            qpso_minimise(_sphere, space, beta_start=start, beta_end=end)

    def test_too_few_particles_rejected(self, space):
        with pytest.raises(ValueError, match="at least 2 particles"):
            qpso_minimise(_sphere, space, n_particles=1)


class TestAblationFairness:
    @pytest.fixture
    def space(self) -> SearchSpace:
        return SearchSpace(tuple(Parameter(f"x{i}", -5.12, 5.12) for i in range(5)))

    def test_budgets_can_be_matched_exactly(self, space):
        """B-01: equal number of objective evaluations, stated explicitly."""
        qpso = qpso_minimise(_sphere, space, n_particles=8, n_iterations=10, seed=1)
        random = random_search(_sphere, space, n_evaluations=qpso.n_evaluations, seed=1)
        assert random.n_evaluations == qpso.n_evaluations

    def test_qpso_beats_random_on_a_multimodal_function(self, space):
        """Rastrigin: the standard check that the search actually does something."""
        def rastrigin(**kwargs: float) -> float:
            x = np.array(list(kwargs.values()))
            return float(10 * len(x) + np.sum(x**2 - 10 * np.cos(2 * np.pi * x)))

        qpso_scores, random_scores = [], []
        for seed in range(1000, 1005):
            q = qpso_minimise(rastrigin, space, n_particles=15, n_iterations=25, seed=seed)
            qpso_scores.append(q.best_score)
            random_scores.append(
                random_search(rastrigin, space, n_evaluations=q.n_evaluations,
                              seed=seed).best_score
            )
        assert np.mean(qpso_scores) < np.mean(random_scores)

    def test_summarise_runs_reports_a_distribution(self, space):
        runs = [
            qpso_minimise(_sphere, space, n_particles=6, n_iterations=6, seed=s)
            for s in range(5)
        ]
        stats = summarise_runs(runs)
        assert stats["n_runs"] == 5
        assert stats["best"] <= stats["median"] <= stats["worst"]
        assert stats["std"] >= 0

    def test_summarise_rejects_all_infinite(self):
        from greenfleet.prediction.qpso import QPSOResult

        bad = [QPSOResult(best_params={}, best_score=math.inf, n_evaluations=1)]
        with pytest.raises(ValueError, match="no finite scores"):
            summarise_runs(bad)
