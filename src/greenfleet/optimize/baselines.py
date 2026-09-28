"""Conventional baselines, for honest benchmarking.

B-04 requires the quantum-inspired optimizer to be compared against conventional
methods, and B-01 requires the comparison to be fair. Four baselines:

``random_search``
    Uniform sampling. The floor: any method that cannot beat it is doing nothing.
``greedy``
    A sensible domain heuristic - cheapest feasible vessel first, slowest feasible
    speed. This is what a planner would do with a spreadsheet, and it is the honest
    thing to beat.
``nsga2``
    The standard multi-objective evolutionary algorithm, via pymoo. Real-coded with
    rounding, which is how this problem would conventionally be attacked without a
    Q-bit encoding.
``milp``
    Exact, via CP-SAT, on a speed-discretised version of the problem. Small
    instances therefore have a *known* optimum, which is what Q-08 needs: a
    metaheuristic claiming to be near-optimal has to be checked against something.
``qbho``
    Quantum Black Hole Optimization - the quantum-inspired swarm metaheuristic the
    problem-statement sponsor describes as its own maritime method ("quantum
    mechanics and swarm intelligence"). Their formulation is not published in
    detail, so this is a faithful reading of the black-hole algorithm (Hatamlou,
    2013) with the quantum-behaved position update of QPSO, on the same real-coded
    relaxation as NSGA-II. Benchmarking the sponsor's own family of method at equal
    budget, and reporting the result either way, is the honest thing to do.

Fairness notes, stated because they matter more than the results
----------------------------------------------------------------
* Every baseline evaluates through the same ``FleetProblem.evaluate``, so all
  methods optimise an identical objective and identical constraints.
* Budgets are expressed in objective evaluations and matched explicitly.
* NSGA-II uses a real-coded relaxation rather than Q-bits. That is a genuine
  difference in representation, and it is the point of the comparison - not a
  handicap imposed on the baseline.
* The MILP discretises speed into bands, so it is exact for the *discretised*
  problem and a bound for the continuous one. Stated wherever it is quoted.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field

import numpy as np

from greenfleet.optimize.pareto import pareto_front, thin_front
from greenfleet.optimize.problem import FleetProblem, FleetSolution

logger = logging.getLogger(__name__)

__all__ = [
    "BaselineResult",
    "random_search_fleet",
    "greedy_fleet",
    "nsga2_fleet",
    "qbho_fleet",
    "milp_fleet",
    "milp_pareto_front",
]


@dataclass
class BaselineResult:
    """Uniform result type, so every algorithm is compared like for like."""

    algorithm: str
    archive: list[FleetSolution] = field(default_factory=list)
    n_evaluations: int = 0
    elapsed_s: float = 0.0
    seed: int = 0
    exact: bool = False
    """True only for the MILP, and only on the discretised problem."""

    notes: list[str] = field(default_factory=list)

    @property
    def front(self) -> np.ndarray:
        if not self.archive:
            return np.empty((0, 3))
        return np.array([s.objectives for s in self.archive])

    @property
    def feasible_found(self) -> bool:
        return bool(self.archive)

    def summary(self) -> str:
        if not self.archive:
            return f"{self.algorithm}: no feasible plan in {self.n_evaluations} evaluations"
        return (
            f"{self.algorithm}: {len(self.archive)} non-dominated plans from "
            f"{self.n_evaluations} evaluations in {self.elapsed_s:.1f}s"
        )


def _collect_front(
    solutions: list[FleetSolution], algorithm: str, evaluations: int,
    elapsed: float, seed: int,
) -> BaselineResult:
    feasible = [s for s in solutions if s.feasible]
    if not feasible:
        return BaselineResult(algorithm, [], evaluations, elapsed, seed)
    objectives = np.array([s.objectives for s in feasible])
    keep = pareto_front(objectives)
    return BaselineResult(
        algorithm, [feasible[i] for i in keep], evaluations, elapsed, seed
    )


def random_search_fleet(
    problem: FleetProblem, n_evaluations: int = 5000, seed: int = 1000
) -> BaselineResult:
    """Uniform random sampling of the decision space."""
    started = time.perf_counter()
    rng = np.random.default_rng(seed)
    n = problem.n_vessels
    low = np.array([v.min_speed_kn for v in problem.vessels])
    high = np.array([v.max_speed_kn for v in problem.vessels])

    solutions = []
    for _ in range(n_evaluations):
        deploy = rng.random(n) < 0.5
        if not deploy.any():
            deploy[rng.integers(n)] = True
        solutions.append(problem.evaluate(
            deploy,
            rng.integers(0, problem.n_routes, size=n),
            rng.integers(0, problem.n_fuels, size=n),
            rng.uniform(low, high),
        ))
    return _collect_front(
        solutions, "random_search", n_evaluations,
        time.perf_counter() - started, seed,
    )


def greedy_fleet(problem: FleetProblem, seed: int = 1000) -> BaselineResult:
    """Domain heuristic: cheapest capacity first, slowest feasible speed.

    What a planner does with a spreadsheet - add the most cost-effective vessel to
    the route with the largest unmet demand until demand is met, then slow steam.
    Sweeps the speed setting to trace a small front rather than returning one point.
    """
    started = time.perf_counter()
    evaluations = 0
    solutions = []

    order = sorted(
        range(problem.n_vessels),
        key=lambda i: problem.vessels[i].annual_cost_inr / problem.vessels[i].capacity_t,
    )
    for speed_fraction in np.linspace(0.0, 1.0, 11):
        for fuel_index in range(problem.n_fuels):
            deploy = np.zeros(problem.n_vessels, dtype=bool)
            route_index = np.zeros(problem.n_vessels, dtype=int)
            fuels = np.full(problem.n_vessels, fuel_index, dtype=int)
            speeds = np.array([
                v.min_speed_kn + speed_fraction * (v.max_speed_kn - v.min_speed_kn)
                for v in problem.vessels
            ])
            remaining = {r.name: r.annual_demand_t for r in problem.routes}

            for index in order:
                # Serve the route with the largest remaining shortfall.
                target = max(remaining, key=lambda name: remaining[name])
                if remaining[target] <= 0:
                    break
                deploy[index] = True
                route_index[index] = [r.name for r in problem.routes].index(target)
                candidate = problem.evaluate(deploy, route_index, fuels, speeds)
                evaluations += 1
                remaining = {
                    r.name: max(0.0, r.annual_demand_t - candidate.cargo_delivered_t[r.name])
                    for r in problem.routes
                }
            solutions.append(problem.evaluate(deploy, route_index, fuels, speeds))
            evaluations += 1

    return _collect_front(
        solutions, "greedy", evaluations, time.perf_counter() - started, seed
    )


def _real_decoder(problem: FleetProblem):
    """Decoder for the real-coded relaxation in [0, 1]^(4n) shared by NSGA-II and QBHO.

    Both conventional baselines attack the problem the same way - relax every
    discrete decision to a real number and round on decode - so a difference
    between them and the Q-bit encoding is a difference in representation, not in
    how the problem was posed.
    """
    n = problem.n_vessels
    low_speed = np.array([v.min_speed_kn for v in problem.vessels])
    high_speed = np.array([v.max_speed_kn for v in problem.vessels])

    def decode(row: np.ndarray) -> FleetSolution:
        deploy = row[:n] > 0.5
        if not deploy.any():
            deploy[int(np.argmax(row[:n]))] = True
        route = np.clip((row[n:2 * n] * problem.n_routes).astype(int),
                        0, problem.n_routes - 1)
        fuel = np.clip((row[2 * n:3 * n] * problem.n_fuels).astype(int),
                       0, problem.n_fuels - 1)
        speed = low_speed + row[3 * n:4 * n] * (high_speed - low_speed)
        return problem.evaluate(deploy, route, fuel, speed)

    return decode


def nsga2_fleet(
    problem: FleetProblem,
    n_evaluations: int = 5000,
    population: int = 40,
    seed: int = 1000,
) -> BaselineResult:
    """NSGA-II via pymoo, real-coded with rounding.

    The conventional attack on this problem: relax every discrete decision to a real
    number in [0, 1] and round on decode. Comparing it against the Q-bit encoding at
    equal evaluation budget is the whole point of the benchmark.
    """
    started = time.perf_counter()
    try:
        from pymoo.algorithms.moo.nsga2 import NSGA2
        from pymoo.core.problem import Problem
        from pymoo.optimize import minimize
    except ImportError:  # pragma: no cover
        logger.warning("pymoo not installed; skipping NSGA-II")
        return BaselineResult("nsga2", [], 0, 0.0, seed,
                              notes=["pymoo not installed"])

    n = problem.n_vessels
    evaluated: list[FleetSolution] = []
    decode = _real_decoder(problem)

    class _Wrapped(Problem):
        def __init__(self):
            super().__init__(n_var=4 * n, n_obj=3, n_constr=1, xl=0.0, xu=1.0)

        def _evaluate(self, X, out, *args, **kwargs):
            objectives, constraints = [], []
            for row in X:
                solution = decode(row)
                evaluated.append(solution)
                objectives.append(solution.objectives)
                constraints.append([solution.violation])
            out["F"] = np.array(objectives)
            out["G"] = np.array(constraints)

    generations = max(1, n_evaluations // population)
    minimize(
        _Wrapped(),
        NSGA2(pop_size=population),
        ("n_gen", generations),
        seed=seed,
        verbose=False,
    )
    return _collect_front(
        evaluated, "nsga2", len(evaluated), time.perf_counter() - started, seed
    )


def qbho_fleet(
    problem: FleetProblem,
    n_evaluations: int = 5000,
    n_stars: int = 20,
    seed: int = 1000,
    beta_start: float = 1.0,
    beta_end: float = 0.2,
    archive_capacity: int = 60,
    event_horizon: bool = True,
) -> BaselineResult:
    """Quantum Black Hole Optimization on the real-coded relaxation.

    Black-hole algorithm (Hatamlou, 2013): a population of "stars" is drawn
    toward the best solution, the "black hole"; any star that crosses the event
    horizon - radius ``f_bh / sum(f_i)`` in fitness terms - is swallowed and
    re-born at a random position, which is the algorithm's whole exploration
    mechanism. The quantum-behaved variant replaces the linear pull
    ``x + rand * (bh - x)`` with the delta-potential-well sampling of QPSO, so a
    star's next position is drawn from a probability cloud around its attractor
    and can tunnel past it.

    Multi-objective adaptation, stated because the original is single-objective:
    the black hole is drawn per star from the non-dominated archive (feasible
    plans only), uniformly at random, so different stars are pulled toward
    different parts of the front. The scalar used for the event-horizon radius is
    the mean of the objectives normalised by the population's worst, plus the
    constraint violation - it decides only who gets re-initialised, never who is
    reported.

    Budget is the total number of ``FleetProblem.evaluate`` calls, matched to the
    other methods (B-01). The defaults (20 stars, beta 1.0 -> 0.2, event horizon
    on) were chosen by a 36-configuration sweep on the 12-vessel instance, the same
    care given to the QIEA rotation angle, so that the baseline is as strong as we
    could make it.
    """
    if n_stars < 2:
        raise ValueError("need at least 2 stars")
    started = time.perf_counter()
    rng = np.random.default_rng(seed)
    decode = _real_decoder(problem)
    dim = 4 * problem.n_vessels
    iterations = max(1, n_evaluations // n_stars)

    positions = rng.random((n_stars, dim))
    evaluated: list[FleetSolution] = []
    archive: list[FleetSolution] = []
    archive_positions: list[np.ndarray] = []

    def scalar(solutions: list[FleetSolution]) -> np.ndarray:
        objectives = np.array([s.objectives for s in solutions], dtype="float64")
        worst = np.maximum(objectives.max(axis=0), 1e-12)
        return (objectives / worst).mean(axis=1) + np.array([s.violation for s in solutions])

    for iteration in range(iterations):
        solutions = [decode(row) for row in positions]
        evaluated.extend(solutions)

        # Archive update: feasible, non-dominated, positions kept for attraction.
        candidates = archive + [s for s in solutions if s.feasible]
        candidate_pos = archive_positions + [
            positions[i].copy() for i, s in enumerate(solutions) if s.feasible
        ]
        if candidates:
            keep = pareto_front(np.array([s.objectives for s in candidates]))
            # Same capacity rule as the QMOEA archive (crowding distance), so the
            # attraction set stays spread and the update stays O(capacity^2).
            if len(keep) > archive_capacity:
                keep = keep[thin_front(np.array([candidates[i].objectives for i in keep]),
                                       archive_capacity)]
            archive = [candidates[i] for i in keep]
            archive_positions = [candidate_pos[i] for i in keep]

        fitness = scalar(solutions)
        if archive_positions:
            picks = rng.integers(len(archive_positions), size=n_stars)
            leaders = [archive_positions[k] for k in picks]
            black_hole = archive_positions[int(rng.integers(len(archive_positions)))]
        else:
            best = int(np.argmin(fitness))
            leaders = [positions[best]] * n_stars
            black_hole = positions[best]

        # Event horizon: stars closer to the black hole than the radius are re-born.
        inverse = 1.0 / (1.0 + fitness)
        radius = inverse.max() / max(inverse.sum(), 1e-12) * np.sqrt(dim)
        distance = np.linalg.norm(positions - black_hole, axis=1)

        beta = beta_start + (beta_end - beta_start) * iteration / max(iterations - 1, 1)
        mean_best = (np.mean(archive_positions, axis=0) if archive_positions
                     else positions.mean(axis=0))
        new_positions = np.empty_like(positions)
        for i in range(n_stars):
            if event_horizon and distance[i] < radius and np.any(positions[i] != black_hole):
                new_positions[i] = rng.random(dim)
                continue
            phi = rng.random(dim)
            attractor = phi * positions[i] + (1.0 - phi) * leaders[i]
            u = rng.uniform(1e-12, 1.0, size=dim)
            step = beta * np.abs(mean_best - positions[i]) * np.log(1.0 / u)
            sign = np.where(rng.random(dim) < 0.5, -1.0, 1.0)
            new_positions[i] = np.clip(attractor + sign * step, 0.0, 1.0)
        positions = new_positions

    # Report the capped archive, as the QMOEA reports its archive: the front over
    # every evaluated point would list thousands of near-duplicate speed settings.
    return BaselineResult(
        "qbho", archive, len(evaluated), time.perf_counter() - started, seed
    )


def _speed_bands(problem: FleetProblem, n_bands: int) -> np.ndarray:
    """Discretised speed grid per vessel, for the exact solver."""
    return np.array([
        np.linspace(v.min_speed_kn, v.max_speed_kn, n_bands)
        for v in problem.vessels
    ])


def milp_fleet(
    problem: FleetProblem,
    objective: int = 2,
    emission_cap_tco2e: float | None = None,
    n_speed_bands: int = 5,
    time_limit_s: float = 60.0,
    seed: int = 1000,
) -> BaselineResult:
    """Exact solve via CP-SAT, on a speed-discretised version of the problem.

    Every (vessel, route, fuel, speed band) combination is pre-evaluated, giving a
    pure assignment problem that CP-SAT solves to proven optimality. That makes it
    the reference a metaheuristic is checked against (Q-08) - with the caveat, always
    stated, that it is exact for the discretised problem and a bound for the
    continuous one.

    Args:
        objective: which objective to minimise (0 fuel, 1 emissions, 2 cost).
        emission_cap_tco2e: optional cap, used for epsilon-constraint Pareto sweeps.
    """
    started = time.perf_counter()
    try:
        from ortools.sat.python import cp_model
    except ImportError:  # pragma: no cover
        logger.warning("ortools not installed; skipping MILP")
        return BaselineResult("milp", [], 0, 0.0, seed, notes=["ortools not installed"])

    bands = _speed_bands(problem, n_speed_bands)
    n_v, n_r, n_f = problem.n_vessels, problem.n_routes, problem.n_fuels

    # Pre-evaluate each single-vessel assignment. The objectives are additive across
    # vessels and the only coupling is through demand and the emission cap, so this
    # is exact rather than an approximation.
    cost = np.full((n_v, n_r, n_f, n_speed_bands), np.inf)
    emissions = np.full_like(cost, np.inf)
    energy = np.full_like(cost, np.inf)
    delivered = np.zeros_like(cost)
    evaluations = 0

    for v in range(n_v):
        for r in range(n_r):
            for f in range(n_f):
                for b in range(n_speed_bands):
                    deploy = np.zeros(n_v, dtype=bool)
                    deploy[v] = True
                    route = np.zeros(n_v, dtype=int)
                    route[v] = r
                    fuel = np.zeros(n_v, dtype=int)
                    fuel[v] = f
                    speeds = bands[:, b]
                    solution = problem.evaluate(deploy, route, fuel, speeds)
                    evaluations += 1
                    # Only per-vessel constraints matter here; demand shortfall is
                    # handled by the model, so ignore that part of the violation.
                    blocking = {
                        k: val for k, val in solution.violations.items()
                        if not k.startswith("demand:") and k != "emission_cap"
                    }
                    if blocking:
                        continue
                    cost[v, r, f, b] = solution.objectives[2]
                    emissions[v, r, f, b] = solution.objectives[1]
                    energy[v, r, f, b] = solution.objectives[0]
                    delivered[v, r, f, b] = solution.cargo_delivered_t[
                        problem.routes[r].name
                    ]

    model = cp_model.CpModel()
    x = {}
    for v in range(n_v):
        for r in range(n_r):
            for f in range(n_f):
                for b in range(n_speed_bands):
                    if np.isfinite(cost[v, r, f, b]):
                        x[v, r, f, b] = model.NewBoolVar(f"x_{v}_{r}_{f}_{b}")

    if not x:
        return BaselineResult("milp", [], evaluations, time.perf_counter() - started,
                              seed, notes=["no feasible vessel assignment exists"])

    # M-09: a vessel is deployed at most once, and integrally.
    for v in range(n_v):
        model.AddAtMostOne(
            x[key] for key in x if key[0] == v
        )

    # C1: cargo demand per route (M-01). Scaled to integers for CP-SAT.
    scale = 1000.0
    for r in range(n_r):
        terms = [
            (int(delivered[key] * 1), x[key]) for key in x if key[1] == r
        ]
        if terms:
            model.Add(
                sum(coefficient * var for coefficient, var in terms)
                >= int(problem.routes[r].annual_demand_t)
            )

    cap = emission_cap_tco2e if emission_cap_tco2e is not None else problem.emission_cap_tco2e
    if cap is not None:
        model.Add(
            sum(int(emissions[key] * scale) * x[key] for key in x) <= int(cap * scale)
        )

    table = {0: energy, 1: emissions, 2: cost}[objective]
    model.Minimize(sum(int(table[key]) * x[key] for key in x))

    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = time_limit_s
    solver.parameters.random_seed = seed
    solver.parameters.num_search_workers = 4
    status = solver.Solve(model)

    elapsed = time.perf_counter() - started
    if status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        return BaselineResult(
            "milp", [], evaluations, elapsed, seed,
            notes=[f"CP-SAT status {solver.StatusName(status)}: no feasible assignment"],
        )

    deploy = np.zeros(n_v, dtype=bool)
    route_index = np.zeros(n_v, dtype=int)
    fuel_index = np.zeros(n_v, dtype=int)
    speeds = np.array([v.min_speed_kn for v in problem.vessels])
    for (v, r, f, b), var in x.items():
        if solver.Value(var):
            deploy[v] = True
            route_index[v] = r
            fuel_index[v] = f
            speeds[v] = bands[v, b]

    solution = problem.evaluate(deploy, route_index, fuel_index, speeds)
    return BaselineResult(
        "milp", [solution] if solution.feasible else [], evaluations, elapsed, seed,
        exact=(status == cp_model.OPTIMAL),
        notes=[
            f"CP-SAT {solver.StatusName(status)}; exact for the "
            f"{n_speed_bands}-band speed discretisation, a bound for continuous speed"
        ],
    )


def milp_pareto_front(
    problem: FleetProblem,
    n_points: int = 6,
    n_speed_bands: int = 5,
    time_limit_s: float = 30.0,
    seed: int = 1000,
) -> BaselineResult:
    """Exact Pareto front by epsilon-constraint: minimise cost under emission caps.

    Gives a *verified* front on small instances, which is what the quantum-inspired
    front is measured against (Q-08, M-10).
    """
    started = time.perf_counter()
    unconstrained = milp_fleet(problem, objective=2, n_speed_bands=n_speed_bands,
                               time_limit_s=time_limit_s, seed=seed)
    cleanest = milp_fleet(problem, objective=1, n_speed_bands=n_speed_bands,
                          time_limit_s=time_limit_s, seed=seed)
    if not unconstrained.archive or not cleanest.archive:
        return BaselineResult("milp_pareto", [], unconstrained.n_evaluations,
                              time.perf_counter() - started, seed,
                              notes=["no feasible endpoint found"])

    high = unconstrained.archive[0].objectives[1]
    low = cleanest.archive[0].objectives[1]
    solutions = list(unconstrained.archive) + list(cleanest.archive)
    evaluations = unconstrained.n_evaluations + cleanest.n_evaluations

    for cap in np.linspace(low, high, n_points)[1:-1]:
        step = milp_fleet(problem, objective=2, emission_cap_tco2e=float(cap),
                          n_speed_bands=n_speed_bands, time_limit_s=time_limit_s,
                          seed=seed)
        evaluations += step.n_evaluations
        solutions.extend(step.archive)

    result = _collect_front(solutions, "milp_pareto", evaluations,
                            time.perf_counter() - started, seed)
    result.exact = True
    result.notes.append(
        f"epsilon-constraint over {n_points} caps; exact for the "
        f"{n_speed_bands}-band speed discretisation"
    )
    return result
