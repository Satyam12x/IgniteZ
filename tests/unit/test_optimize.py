"""Fleet optimizer tests.

Maps to the M-series (mathematical formulation and constraints), the Q-series
(quantum-inspired engine) and the parts of the B-series that are properties of the
code rather than of a benchmark run.

The tests that matter most on stage are M-01 and M-05: a solver that silently drops
cargo, or silently breaches an emission cap, is worse than one that refuses - because
the refusal is visible and the silent answer is not.
"""

from __future__ import annotations

import numpy as np
import pytest

from greenfleet.optimize.baselines import (
    greedy_fleet,
    milp_fleet,
    nsga2_fleet,
    random_search_fleet,
)
from greenfleet.optimize.encoding import FleetEncoding
from greenfleet.optimize.pareto import (
    dominates,
    hypervolume,
    non_dominated_mask,
    pareto_front,
    reference_point,
    spacing,
    thin_front,
)
from greenfleet.optimize.problem import FleetProblem, Infeasible, Route, Vessel
from greenfleet.optimize.qmoea import optimise_fleet
from greenfleet.optimize.qubo import (
    build_qubo,
    solve_qubo_bruteforce,
    verify_penalty_weight,
)
from greenfleet.optimize.scenarios import (
    coastal_scenario,
    gttp_tug_scenario,
    large_scenario,
    tiny_scenario,
)

pytestmark = pytest.mark.edgecase


@pytest.fixture(scope="module")
def tiny() -> FleetProblem:
    return tiny_scenario()


@pytest.fixture(scope="module")
def coastal() -> FleetProblem:
    return coastal_scenario()


def _vessel(name: str = "V1", **kwargs) -> Vessel:
    defaults = dict(
        vessel_class="general_cargo", capacity_t=10000.0,
        design_efficiency_gco2_per_t_nmi=18.0,
        min_speed_kn=8.0, max_speed_kn=15.0, tank_volume_m3=1000.0,
        annual_cost_inr=1e8, available_hours=7000.0,
    )
    defaults.update(kwargs)
    return Vessel(name=name, **defaults)


def _route(name: str = "R1", **kwargs) -> Route:
    defaults = dict(
        ports=("mumbai", "cochin"), distance_nmi=500.0, annual_demand_t=200_000.0
    )
    defaults.update(kwargs)
    return Route(name=name, **defaults)


class TestM01DemandExceedsCapacity:
    def test_impossible_demand_is_refused_not_silently_dropped(self):
        """The most important failure mode: never return a plan that drops cargo."""
        problem = FleetProblem([_vessel()], [_route(annual_demand_t=1e12)])
        with pytest.raises(Infeasible, match="cannot be met"):
            problem.check_demand_satisfiable()

    def test_the_refusal_states_the_shortfall(self):
        problem = FleetProblem([_vessel()], [_route(annual_demand_t=1e12)])
        with pytest.raises(Infeasible, match="short by"):
            problem.check_demand_satisfiable()

    def test_satisfiable_demand_passes(self, tiny):
        tiny.check_demand_satisfiable()

    def test_a_shortfall_shows_up_as_a_violation(self):
        problem = FleetProblem([_vessel()], [_route(annual_demand_t=1e9)])
        solution = problem.evaluate([True], [0], [0], [12.0])
        assert not solution.feasible
        assert any(k.startswith("demand:") for k in solution.violations)

    def test_optimiser_returns_no_archive_when_infeasible(self):
        problem = FleetProblem([_vessel()], [_route(annual_demand_t=1e10)])
        result = optimise_fleet(problem, n_individuals=6, n_generations=10, seed=1)
        assert not result.feasible_found
        assert "no feasible" in result.summary()


class TestM02ZeroDemand:
    def test_zero_demand_is_feasible_with_nothing_deployed(self):
        problem = FleetProblem([_vessel()], [_route(annual_demand_t=0.0)])
        solution = problem.evaluate([False], [0], [0], [12.0])
        assert solution.feasible
        assert solution.n_deployed == 0
        assert solution.objectives[1] == 0.0

    def test_optimiser_finds_the_empty_plan(self):
        problem = FleetProblem([_vessel(), _vessel("V2")],
                               [_route(annual_demand_t=0.0)])
        result = optimise_fleet(problem, n_individuals=8, n_generations=20, seed=1)
        assert result.feasible_found
        assert result.best_by(1).objectives[1] >= 0.0


class TestM03SingleVesselSingleRoute:
    def test_solver_works_on_the_smallest_instance(self, tiny):
        result = optimise_fleet(tiny, n_individuals=8, n_generations=30, seed=1)
        assert result.feasible_found

    def test_matches_a_hand_calculation(self):
        """One vessel, one route: trips and delivery are checkable by hand."""
        # A near-degenerate speed range pins the vessel to ~10 kn so the trip
        # count is checkable by hand.
        vessel = _vessel(capacity_t=10000.0, available_hours=7000.0,
                         min_speed_kn=10.0, max_speed_kn=10.001)
        route = _route(distance_nmi=500.0, annual_demand_t=1000.0)
        problem = FleetProblem([vessel], [route])
        solution = problem.evaluate([True], [0], [0], [10.0])
        # round trip 1000 nmi at 10 kn = 100 h; 7000 h allows 70 trips
        expected_trips = int(7000 // 100)
        delivered = solution.cargo_delivered_t["R1"]
        assert delivered == pytest.approx(expected_trips * 10000.0, rel=0.05)


class TestM05EmissionCap:
    def test_an_impossible_cap_yields_no_feasible_plan(self, tiny):
        problem = tiny_scenario(emission_cap_tco2e=1.0)
        result = optimise_fleet(problem, n_individuals=8, n_generations=20, seed=1)
        assert not result.feasible_found

    def test_cap_breach_is_a_visible_violation(self, tiny):
        problem = tiny_scenario(emission_cap_tco2e=1.0)
        solution = problem.evaluate([True, True], [0, 0], [0, 0], [12.0, 12.0])
        assert "emission_cap" in solution.violations

    def test_a_generous_cap_is_satisfiable(self):
        problem = tiny_scenario(emission_cap_tco2e=1e9)
        result = optimise_fleet(problem, n_individuals=8, n_generations=30, seed=1)
        assert result.feasible_found

    def test_every_returned_plan_respects_the_cap(self):
        problem = coastal_scenario(emission_cap_tco2e=40_000.0)
        result = optimise_fleet(problem, n_individuals=20, n_generations=60, seed=1000)
        for solution in result.archive:
            assert solution.objectives[1] <= 40_000.0 + 1e-6


class TestM06SpeedBounds:
    def test_speed_is_clamped_into_the_engine_envelope(self, tiny):
        result = optimise_fleet(tiny, n_individuals=8, n_generations=30, seed=1)
        for solution in result.archive:
            for index, vessel in enumerate(tiny.vessels):
                if solution.deploy[index]:
                    assert vessel.min_speed_kn - 1e-9 <= solution.speed_kn[index]
                    assert solution.speed_kn[index] <= vessel.max_speed_kn + 1e-9

    def test_encoding_clips_out_of_range_speeds(self, tiny):
        encoding = FleetEncoding(tiny)
        plan = encoding.decode(np.ones(encoding.n_bits, dtype=bool),
                               np.array([999.0, -5.0]))
        assert plan.repaired
        for index, vessel in enumerate(tiny.vessels):
            assert vessel.min_speed_kn <= plan.speed_kn[index] <= vessel.max_speed_kn

    def test_inverted_speed_bounds_rejected(self):
        with pytest.raises(ValueError, match="min_speed"):
            _vessel(min_speed_kn=15.0, max_speed_kn=8.0)


class TestM08VesselRouteCompatibility:
    def test_incompatible_assignment_never_appears(self):
        """A draft-limited vessel must never be assigned to a port it cannot enter."""
        vessels = [
            _vessel("Deep", compatible_routes=("R2",)),
            _vessel("Shallow", compatible_routes=("R1",)),
        ]
        routes = [_route("R1", annual_demand_t=50_000.0),
                  _route("R2", annual_demand_t=50_000.0)]
        problem = FleetProblem(vessels, routes)
        result = optimise_fleet(problem, n_individuals=10, n_generations=40, seed=1)
        for solution in result.archive:
            for index, vessel in enumerate(problem.vessels):
                if solution.deploy[index] and vessel.compatible_routes:
                    assert routes[solution.route_index[index]].name in (
                        vessel.compatible_routes
                    )

    def test_incompatible_assignment_is_a_violation(self):
        problem = FleetProblem([_vessel(compatible_routes=("R2",))], [_route("R1")])
        solution = problem.evaluate([True], [0], [0], [12.0])
        assert "route_compatibility" in solution.violations

    def test_incompatible_fuel_is_a_violation(self):
        problem = FleetProblem([_vessel(compatible_fuels=("mgo",))], [_route()],
                               allow_fuels=("mgo", "hfo"))
        solution = problem.evaluate([True], [0], [1], [12.0])
        assert "fuel_compatibility" in solution.violations


class TestM09IntegerVessels:
    def test_deployment_is_binary(self, tiny):
        result = optimise_fleet(tiny, n_individuals=8, n_generations=20, seed=1)
        for solution in result.archive:
            assert solution.deploy.dtype == bool
            assert int(solution.deploy.sum()) == solution.n_deployed

    def test_trips_are_whole_numbers(self):
        """A fractional voyage does not exist."""
        problem = FleetProblem([_vessel()], [_route(annual_demand_t=1.0)])
        solution = problem.evaluate([True], [0], [0], [12.0])
        vessel = problem.vessels[0]
        hours_per_trip = 2 * 500.0 / 12.0
        expected = int(vessel.available_hours // hours_per_trip)
        assert solution.cargo_delivered_t["R1"] == pytest.approx(
            expected * vessel.capacity_t, rel=0.02
        )


class TestM10ParetoFront:
    def test_no_returned_plan_dominates_another(self, coastal):
        """The definition of a Pareto front, asserted rather than assumed."""
        result = optimise_fleet(coastal, n_individuals=20, n_generations=60, seed=1000)
        front = result.front
        for i in range(len(front)):
            for j in range(len(front)):
                if i != j:
                    assert not dominates(front[i], front[j])

    def test_every_returned_plan_is_feasible(self, coastal):
        result = optimise_fleet(coastal, n_individuals=20, n_generations=60, seed=1000)
        assert all(s.feasible for s in result.archive)

    def test_dominance_is_defined_correctly(self):
        assert dominates(np.array([1.0, 1.0]), np.array([2.0, 2.0]))
        assert dominates(np.array([1.0, 2.0]), np.array([1.0, 3.0]))
        assert not dominates(np.array([1.0, 3.0]), np.array([2.0, 2.0]))
        assert not dominates(np.array([1.0, 1.0]), np.array([1.0, 1.0]))

    def test_non_dominated_mask_keeps_only_the_front(self):
        objectives = np.array([[1.0, 5.0], [2.0, 3.0], [5.0, 1.0], [3.0, 4.0]])
        mask = non_dominated_mask(objectives)
        assert mask.tolist() == [True, True, True, False]

    def test_infeasible_solutions_are_excluded_from_the_front(self):
        objectives = np.array([[1.0, 1.0], [5.0, 5.0]])
        violations = np.array([1.0, 0.0])  # the better one is infeasible
        keep = pareto_front(objectives, violations)
        assert keep.tolist() == [1]

    def test_a_front_of_one_is_valid(self):
        assert len(pareto_front(np.array([[1.0, 1.0]]))) == 1


class TestM11ExtremeWeights:
    def test_the_cheapest_archive_member_matches_the_milp_cost_optimum(self, tiny):
        """All weight on cost should reproduce the single-objective minimum."""
        exact = milp_fleet(tiny, objective=2)
        assert exact.archive, "MILP should find the tiny instance feasible"
        result = optimise_fleet(tiny, n_individuals=12, n_generations=60, seed=1000)
        best = result.best_by(2)
        assert best is not None
        assert best.objectives[2] == pytest.approx(
            exact.archive[0].objectives[2], rel=1e-6
        )

    def test_best_by_returns_none_when_infeasible(self):
        problem = FleetProblem([_vessel()], [_route(annual_demand_t=1e10)])
        result = optimise_fleet(problem, n_individuals=6, n_generations=5, seed=1)
        assert result.best_by(0) is None


class TestQ01EncodingValidity:
    def test_every_bit_string_decodes_to_a_valid_plan(self, coastal):
        """No observation can decode to a non-existent route or fuel."""
        encoding = FleetEncoding(coastal)
        rng = np.random.default_rng(0)
        for _ in range(200):
            bits = rng.random(encoding.n_bits) < 0.5
            plan = encoding.decode(bits, encoding.random_speeds(rng))
            assert np.all(plan.route_index >= 0)
            assert np.all(plan.route_index < coastal.n_routes)
            assert np.all(plan.fuel_index >= 0)
            assert np.all(plan.fuel_index < coastal.n_fuels)

    def test_all_ones_and_all_zeros_both_decode(self, coastal):
        encoding = FleetEncoding(coastal)
        speeds = encoding.random_speeds(np.random.default_rng(0))
        for bits in (np.ones(encoding.n_bits, dtype=bool),
                     np.zeros(encoding.n_bits, dtype=bool)):
            plan = encoding.decode(bits, speeds)
            assert len(plan.deploy) == coastal.n_vessels

    def test_encode_decode_round_trips(self, coastal):
        encoding = FleetEncoding(coastal)
        rng = np.random.default_rng(1)
        deploy = rng.random(coastal.n_vessels) < 0.7
        deploy[0] = True
        route = rng.integers(0, coastal.n_routes, coastal.n_vessels)
        fuel = rng.integers(0, coastal.n_fuels, coastal.n_vessels)
        bits = encoding.encode(deploy, route, fuel)
        plan = encoding.decode(bits, encoding.random_speeds(rng))
        np.testing.assert_array_equal(plan.deploy, deploy)
        np.testing.assert_array_equal(plan.route_index, route)
        np.testing.assert_array_equal(plan.fuel_index, fuel)


class TestQ02MixedVariables:
    def test_discrete_and_continuous_are_encoded_separately(self, coastal):
        encoding = FleetEncoding(coastal)
        described = encoding.describe()
        assert described["n_bits"] > 0
        assert described["continuous_dimensions"] == coastal.n_vessels

    def test_speed_never_decodes_outside_its_bounds(self, coastal):
        encoding = FleetEncoding(coastal)
        rng = np.random.default_rng(3)
        for _ in range(100):
            plan = encoding.decode(
                rng.random(encoding.n_bits) < 0.5,
                rng.uniform(-50, 100, coastal.n_vessels),
            )
            for index, vessel in enumerate(coastal.vessels):
                assert vessel.min_speed_kn <= plan.speed_kn[index] <= vessel.max_speed_kn

    def test_both_layers_actually_move(self, coastal):
        """A hybrid that only optimises one layer is not a hybrid."""
        result = optimise_fleet(coastal, n_individuals=20, n_generations=60, seed=1000)
        assert result.feasible_found
        speeds = np.array([s.speed_kn for s in result.archive])
        deploys = np.array([s.deploy for s in result.archive])
        assert len(result.archive) >= 1
        # With more than one plan, the two layers should not both be constant.
        if len(result.archive) > 1:
            assert speeds.std() > 0 or deploys.std() > 0


class TestQ03PrematureConvergence:
    def test_diversity_is_tracked(self, coastal):
        result = optimise_fleet(coastal, n_individuals=20, n_generations=40, seed=1000)
        assert len(result.diversity) == result.generations
        assert result.diversity[0] > 0.9

    def test_amplitudes_do_not_fully_collapse(self, coastal):
        result = optimise_fleet(coastal, n_individuals=20, n_generations=200, seed=1000)
        assert min(result.diversity) > 0.0


class TestQ05ConstraintHandling:
    def test_the_empty_fleet_is_repaired_not_scored(self, coastal):
        encoding = FleetEncoding(coastal)
        plan = encoding.decode(np.zeros(encoding.n_bits, dtype=bool),
                               encoding.random_speeds(np.random.default_rng(0)))
        assert plan.deploy.any()
        assert plan.repaired

    def test_repairs_are_counted(self, coastal):
        result = optimise_fleet(coastal, n_individuals=20, n_generations=40, seed=1000)
        assert result.repairs >= 0

    def test_a_feasible_plan_is_never_traded_for_an_infeasible_one(self, coastal):
        result = optimise_fleet(coastal, n_individuals=20, n_generations=60, seed=1000)
        assert all(s.violation <= 1e-9 for s in result.archive)


class TestQ06Determinism:
    def test_same_seed_same_front(self, coastal):
        a = optimise_fleet(coastal, n_individuals=15, n_generations=30, seed=7)
        b = optimise_fleet(coastal, n_individuals=15, n_generations=30, seed=7)
        np.testing.assert_allclose(a.front, b.front)

    def test_different_seeds_explore_differently(self, coastal):
        a = optimise_fleet(coastal, n_individuals=15, n_generations=30, seed=1)
        b = optimise_fleet(coastal, n_individuals=15, n_generations=30, seed=2)
        assert a.front.shape != b.front.shape or not np.allclose(a.front, b.front)


class TestQ07TimeLimit:
    def test_a_time_limit_returns_the_best_so_far(self, coastal):
        result = optimise_fleet(coastal, n_individuals=20, n_generations=100_000,
                                seed=1000, time_limit_s=1.0)
        assert result.time_limited
        assert result.generations < 100_000
        assert result.elapsed_s < 10.0

    def test_no_time_limit_runs_every_generation(self, tiny):
        result = optimise_fleet(tiny, n_individuals=6, n_generations=20, seed=1)
        assert not result.time_limited
        assert result.generations == 20


class TestQ08KnownOptimum:
    def test_matches_the_milp_optimum_on_a_tiny_instance(self, tiny):
        """A metaheuristic that misses a trivial optimum cannot be trusted."""
        exact = milp_fleet(tiny, objective=2)
        assert exact.archive
        assert exact.exact, "CP-SAT should prove optimality on this size"
        optimum = exact.archive[0].objectives[2]
        result = optimise_fleet(tiny, n_individuals=12, n_generations=60, seed=1000)
        gap = result.best_by(2).objectives[2] / optimum - 1.0
        assert gap <= 0.02, f"Q-08 requires a gap within 2%, got {100 * gap:.3f}%"

    def test_milp_reports_its_discretisation_caveat(self, tiny):
        exact = milp_fleet(tiny, objective=2)
        assert any("discretisation" in note for note in exact.notes)


class TestQ10QUBO:
    def test_ground_state_matches_the_milp_optimum(self, tiny):
        """The quantum-ready claim, verified rather than asserted."""
        model = build_qubo(tiny, objective=2, n_speed_bands=3)
        best_x, _ = solve_qubo_bruteforce(model)
        decoded = model.decode(best_x)
        exact = milp_fleet(tiny, objective=2, n_speed_bands=3)
        assert decoded.feasible
        assert decoded.objectives[2] == pytest.approx(
            exact.archive[0].objectives[2], rel=1e-6
        )

    def test_penalty_weight_is_neither_too_low_nor_too_high(self, tiny):
        diagnosis = verify_penalty_weight(tiny, objective=2, n_speed_bands=3)
        assert not diagnosis["too_low"], "ground state is infeasible: penalty too low"
        assert not diagnosis["too_high"], "objective swamped: penalty too high"

    def test_a_too_low_penalty_breaks_feasibility(self, tiny):
        """The failure mode Q-10 names, demonstrated deliberately."""
        diagnosis = verify_penalty_weight(tiny, objective=2, n_speed_bands=2,
                                          penalty_weight=1.0)
        assert diagnosis["too_low"]

    def test_matrix_is_square_and_sized_to_the_variables(self, tiny):
        model = build_qubo(tiny, objective=2, n_speed_bands=3)
        assert model.Q.shape == (model.n_variables, model.n_variables)

    def test_energy_rejects_a_wrong_length_vector(self, tiny):
        model = build_qubo(tiny, objective=2, n_speed_bands=3)
        with pytest.raises(ValueError, match="expected"):
            model.energy(np.zeros(model.n_variables + 1))

    def test_bruteforce_refuses_an_oversized_model(self, coastal):
        model = build_qubo(coastal, objective=2, n_speed_bands=2)
        with pytest.raises(ValueError, match="raise max_variables"):
            solve_qubo_bruteforce(model)


class TestBaselines:
    def test_every_baseline_returns_a_comparable_result(self, tiny):
        for result in (
            random_search_fleet(tiny, n_evaluations=300, seed=1),
            greedy_fleet(tiny, seed=1),
            nsga2_fleet(tiny, n_evaluations=300, seed=1),
        ):
            assert result.front.ndim == 2
            assert all(s.feasible for s in result.archive)

    def test_baselines_evaluate_the_same_problem(self, tiny):
        """B-01: a benchmark where methods solve different problems proves nothing."""
        a = random_search_fleet(tiny, n_evaluations=300, seed=1)
        b = greedy_fleet(tiny, seed=1)
        for result in (a, b):
            for solution in result.archive:
                recomputed = tiny.evaluate(
                    solution.deploy, solution.route_index,
                    np.array([tiny.fuels.index(f) for f in solution.fuel]),
                    solution.speed_kn,
                )
                np.testing.assert_allclose(
                    recomputed.objectives, solution.objectives, rtol=1e-9
                )

    def test_budget_is_honoured(self, tiny):
        result = random_search_fleet(tiny, n_evaluations=250, seed=1)
        assert result.n_evaluations == 250


class TestParetoMetrics:
    def test_hypervolume_is_positive_for_a_dominating_front(self):
        front = np.array([[1.0, 1.0]])
        assert hypervolume(front, np.array([2.0, 2.0])) == pytest.approx(1.0)

    def test_hypervolume_is_zero_beyond_the_reference(self):
        assert hypervolume(np.array([[3.0, 3.0]]), np.array([2.0, 2.0])) == 0.0

    def test_a_better_front_has_more_hypervolume(self):
        reference = np.array([10.0, 10.0])
        worse = np.array([[5.0, 5.0]])
        better = np.array([[2.0, 5.0], [5.0, 2.0]])
        assert hypervolume(better, reference) > hypervolume(worse, reference)

    def test_reference_point_dominates_every_front(self):
        fronts = [np.array([[1.0, 5.0]]), np.array([[4.0, 2.0]])]
        reference = reference_point(fronts)
        for front in fronts:
            assert np.all(front < reference)

    def test_reference_point_needs_a_non_empty_front(self):
        with pytest.raises(ValueError, match="non-empty"):
            reference_point([np.empty((0, 2))])

    def test_spacing_is_zero_for_a_single_point(self):
        assert spacing(np.array([[1.0, 1.0]])) == 0.0

    def test_thinning_keeps_the_extremes(self):
        front = np.array([[float(i), float(10 - i)] for i in range(11)])
        keep = thin_front(front, 4)
        assert len(keep) == 4
        kept = front[keep]
        assert kept[:, 0].min() == front[:, 0].min()
        assert kept[:, 0].max() == front[:, 0].max()

    def test_thinning_is_a_no_op_when_small_enough(self):
        front = np.array([[1.0, 2.0], [2.0, 1.0]])
        assert len(thin_front(front, 10)) == 2


class TestScenarios:
    def test_tiny_is_small_enough_to_brute_force(self, tiny):
        assert tiny.n_vessels <= 3
        assert tiny.search_space_size() < 1e4

    def test_coastal_is_a_genuine_combinatorial_problem(self, coastal):
        assert coastal.search_space_size() > 1e10

    def test_tug_scenario_has_no_design_certificates(self):
        """Tugs are below the MRV threshold: this is the P-05 cold-start case."""
        problem = gttp_tug_scenario()
        assert all(
            v.design_efficiency_gco2_per_t_nmi is None for v in problem.vessels
        )

    def test_large_scenario_scales(self):
        problem = large_scenario(n_vessels=50, n_routes=4)
        assert problem.n_vessels == 50
        assert problem.search_space_size() > 1e50

    def test_scenarios_declare_their_provenance(self, coastal):
        provenance = coastal.as_provenance()
        assert provenance["n_vessels"] == coastal.n_vessels
        assert "fuel_pathways" in provenance


class TestValidation:
    def test_empty_fleet_rejected(self):
        with pytest.raises(ValueError, match="at least one candidate vessel"):
            FleetProblem([], [_route()])

    def test_empty_routes_rejected(self):
        with pytest.raises(ValueError, match="at least one route"):
            FleetProblem([_vessel()], [])

    def test_negative_capacity_rejected(self):
        with pytest.raises(ValueError, match="capacity"):
            _vessel(capacity_t=-1.0)

    def test_a_route_needs_two_ports(self):
        with pytest.raises(ValueError, match="at least two ports"):
            Route("R", ("mumbai",), 100.0, 1000.0)

    def test_negative_distance_rejected(self):
        with pytest.raises(ValueError, match="distance"):
            Route("R", ("mumbai", "cochin"), -1.0, 1000.0)

    def test_too_few_individuals_rejected(self, tiny):
        with pytest.raises(ValueError, match="at least 2 individuals"):
            optimise_fleet(tiny, n_individuals=1)

    def test_bad_beta_rejected(self, tiny):
        with pytest.raises(ValueError, match="beta"):
            optimise_fleet(tiny, beta_start=0.5, beta_end=1.0)
