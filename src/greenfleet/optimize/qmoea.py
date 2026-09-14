"""Quantum-inspired multi-objective fleet optimizer.

Deliverable 3: the core metaheuristic engine. It combines the two quantum-inspired
components built and verified in Phase 1 and applies them to the mixed-variable
fleet problem:

* **Q-bit encoding + rotation gates** for the discrete decisions - deploy, route and
  fuel. Verified against exhaustive optima up to 2^8 states.
* **QPSO** for the continuous speed vector, with velocity-free updates sampled from
  a delta potential well.

Both layers optimise one shared objective and are updated together each generation,
which is the hybrid encoding of Q-02.

Multi-objective handling
------------------------
Fuel, emissions and cost conflict, so there is no single best solution and the
algorithm maintains an **archive** of non-dominated plans rather than one incumbent
(M-10). The rotation gate needs a single target to rotate toward, so each generation
picks a *leader* from the archive by crowding distance, favouring sparse regions.
That is what spreads the front instead of collapsing it onto one corner.

Constraint handling (Q-05)
--------------------------
Constrained domination: a feasible solution always beats an infeasible one, and two
infeasible solutions are ranked by violation magnitude. The search can therefore
start infeasible and still find its way in, while a feasible plan is never traded
away for an infeasible one that happens to score better.

Time limits (Q-07)
------------------
A wall-clock budget returns the best archive found so far, marked ``time_limited``,
rather than crashing or returning nothing.
"""

from __future__ import annotations

import logging
import math
import time
from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np

from greenfleet.optimize.encoding import FleetEncoding
from greenfleet.optimize.pareto import crowding_distance
from greenfleet.optimize.problem import FleetProblem, FleetSolution
from greenfleet.prediction.qiea import (
    DEFAULT_PROBABILITY_FLOOR,
    observe,
)

logger = logging.getLogger(__name__)

__all__ = ["QMOEAResult", "optimise_fleet", "adaptive_rotation_angle"]

_HALF_PI = math.pi / 2.0


@dataclass
class QMOEAResult:
    """Outcome of a quantum-inspired multi-objective run."""

    archive: list[FleetSolution] = field(default_factory=list)
    """Non-dominated feasible plans - the Pareto front (M-10)."""

    n_evaluations: int = 0
    generations: int = 0
    history: list[int] = field(default_factory=list)
    """Archive size per generation."""

    best_violation: list[float] = field(default_factory=list)
    """Lowest violation seen per generation; shows the search reaching feasibility."""

    diversity: list[float] = field(default_factory=list)
    """Q-bit diversity per generation, 1.0 superposed to 0.0 collapsed (Q-03)."""

    hypervolume_history: list[float] = field(default_factory=list)
    time_limited: bool = False
    """Q-07: the budget ran out before the generation count did."""

    elapsed_s: float = 0.0
    seed: int = 0
    repairs: int = 0

    @property
    def front(self) -> np.ndarray:
        """Objective values of the archive, shape (n, 3)."""
        if not self.archive:
            return np.empty((0, 3))
        return np.array([s.objectives for s in self.archive])

    @property
    def feasible_found(self) -> bool:
        return bool(self.archive)

    def best_by(self, objective: int) -> FleetSolution | None:
        """The archive member minimising one objective.

        Useful as the M-11 check: with all weight on cost this should match the
        single-objective cost minimum.
        """
        if not self.archive:
            return None
        return min(self.archive, key=lambda s: s.objectives[objective])

    def summary(self) -> str:
        if not self.archive:
            return (
                f"no feasible plan found in {self.n_evaluations} evaluations "
                f"(best violation {min(self.best_violation, default=float('nan')):.4g})"
            )
        return (
            f"{len(self.archive)} non-dominated plans from {self.n_evaluations} "
            f"evaluations in {self.elapsed_s:.1f}s (seed {self.seed})"
        )


# Rotation-angle scaling constant, calibrated by sweep. See
# :func:`adaptive_rotation_angle` for what it means and how it was set.
_ANGLE_SCALE = 0.03 * math.pi
_ANGLE_MIN = 0.002 * math.pi
_ANGLE_MAX = 0.15 * math.pi


def adaptive_rotation_angle(n_bits: int, n_generations: int) -> float:
    """Rotation angle scaled to problem size and budget.

        delta = 0.03 pi * n_bits / n_generations

    **Why it depends on both.** A Q-bit saturates in a number of generations set by
    the angle alone. What changes with problem size is how many bits must be right
    *together*, and therefore how much budget each one gets. A 6-bit instance with
    60 generations can afford to explore slowly; a 360-bit instance with the same
    budget has to commit, because it will never resolve 360 amplitudes one at a
    time.

    **Why it is not one fixed value.** Measured, not assumed. A single angle tuned
    on the 14-bit knapsack of Phase 1 (0.005 pi) loses two orders of magnitude of
    hypervolume on a 60-vessel instance; an angle fast enough for 60 vessels misses
    the exact optimum on the 2-vessel instance in 5 of 30 seeds. Neither is usable
    alone, and the ratio between the two working values tracks ``n_bits /
    n_generations`` closely enough to use directly.

    The constant was calibrated on a saturation-fraction sweep across a 12-vessel
    and a 60-vessel instance, then checked against the Q-08 exact-optimum test on
    the 2-vessel instance. One rule, three sizes, no per-instance tuning.
    """
    if n_bits < 1 or n_generations < 1:
        raise ValueError("n_bits and n_generations must both be positive")
    return float(
        np.clip(_ANGLE_SCALE * n_bits / n_generations, _ANGLE_MIN, _ANGLE_MAX)
    )


def _constrained_better(a: FleetSolution, b: FleetSolution) -> bool:
    """Constrained domination: feasibility first, then Pareto dominance."""
    if a.feasible and not b.feasible:
        return True
    if not a.feasible and not b.feasible:
        return a.violation < b.violation
    if not a.feasible:
        return False
    return bool(
        np.all(a.objectives <= b.objectives) and np.any(a.objectives < b.objectives)
    )


def _update_archive(
    archive: list[FleetSolution], candidate: FleetSolution, capacity: int
) -> list[FleetSolution]:
    """Insert a feasible candidate, keeping only non-dominated plans."""
    if not candidate.feasible:
        return archive
    for member in archive:
        if np.all(member.objectives <= candidate.objectives) and np.any(
            member.objectives < candidate.objectives
        ):
            return archive  # dominated by something already held
    survivors = [
        m
        for m in archive
        if not (
            np.all(candidate.objectives <= m.objectives)
            and np.any(candidate.objectives < m.objectives)
        )
    ]
    # Exact duplicates add nothing to a decision-maker's choice.
    if any(np.allclose(m.objectives, candidate.objectives) for m in survivors):
        return survivors
    survivors.append(candidate)

    if len(survivors) > capacity:
        objectives = np.array([s.objectives for s in survivors])
        keep = np.argsort(-crowding_distance(objectives))[:capacity]
        survivors = [survivors[i] for i in sorted(keep)]
    return survivors


def _pick_leader(
    archive: list[FleetSolution],
    fallback: FleetSolution,
    rng: np.random.Generator,
) -> FleetSolution:
    """Choose the plan the rotation gate rotates toward.

    Biased to sparse regions of the front by crowding distance, so the search keeps
    widening the trade-off curve instead of piling onto one corner.
    """
    if not archive:
        return fallback
    if len(archive) <= 2:
        return archive[rng.integers(len(archive))]
    distances = crowding_distance(np.array([s.objectives for s in archive]))
    finite = np.isfinite(distances)
    if not finite.any():
        return archive[rng.integers(len(archive))]
    weights = np.where(finite, distances, distances[finite].max())
    total = weights.sum()
    if total <= 0:
        return archive[rng.integers(len(archive))]
    return archive[rng.choice(len(archive), p=weights / total)]


def optimise_fleet(
    problem: FleetProblem,
    n_individuals: int = 20,
    n_generations: int = 60,
    rotation_angle: float | None = None,
    probability_floor: float = DEFAULT_PROBABILITY_FLOOR,
    beta_start: float = 1.0,
    beta_end: float = 0.4,
    archive_capacity: int = 60,
    seed: int = 1000,
    time_limit_s: float | None = None,
    on_generation: Callable[[int, QMOEAResult], bool | None] | None = None,
) -> QMOEAResult:
    """Optimise fleet deployment with a hybrid Q-bit / QPSO search.

    Args:
        problem: the fleet problem; every solver evaluates through it.
        n_individuals: population size, shared by both layers.
        n_generations: rotation-gate updates.
        rotation_angle: Q-bit rotation step. Left unset it is derived from the
            generation budget so a Q-bit can saturate within a quarter of the run -
            see :func:`adaptive_rotation_angle`. A fixed angle tuned on a small
            instance is badly wrong on a large one.
        probability_floor: keeps amplitudes off 0 and 1 so tunnelling stays possible.
        beta_start, beta_end: QPSO contraction-expansion coefficient, decreasing.
        archive_capacity: maximum size of the Pareto archive.
        time_limit_s: wall-clock budget. Exceeding it returns the best archive so
            far marked ``time_limited`` (Q-07).
        on_generation: called after each generation with the generation index and
            the live result, so a caller can report progress. Returning ``False``
            stops the search early and returns the archive so far, marked
            ``time_limited``; this is how an interactive session cancels a run.

    Returns:
        A result whose ``archive`` is the Pareto front of feasible plans.
    """
    if n_individuals < 2:
        raise ValueError("need at least 2 individuals")
    if n_generations < 1:
        raise ValueError("need at least 1 generation")
    if not 0.0 < beta_end <= beta_start < 1.8:
        raise ValueError(
            f"require 0 < beta_end <= beta_start < 1.8, "
            f"got {beta_start} and {beta_end}"
        )

    started = time.perf_counter()
    rng = np.random.default_rng(seed)
    encoding = FleetEncoding(problem)
    result = QMOEAResult(seed=seed)

    if rotation_angle is None:
        rotation_angle = adaptive_rotation_angle(encoding.n_bits, n_generations)
        logger.debug(
            "adaptive rotation angle %.5f rad (%.4f pi) for %d generations",
            rotation_angle, rotation_angle / math.pi, n_generations,
        )

    low_speed, high_speed = encoding.speed_bounds

    theta_low = math.asin(math.sqrt(probability_floor))
    theta_high = _HALF_PI - theta_low
    theta = np.full((n_individuals, encoding.n_bits), math.pi / 4.0)
    speeds = rng.uniform(low_speed, high_speed, size=(n_individuals, problem.n_vessels))

    personal_best_speed = speeds.copy()
    personal_best: list[FleetSolution | None] = [None] * n_individuals
    incumbent: FleetSolution | None = None

    def evaluate(bits: np.ndarray, speed_row: np.ndarray) -> FleetSolution:
        plan = encoding.decode(bits, speed_row)
        if plan.repaired:
            result.repairs += 1
        solution = problem.evaluate(
            plan.deploy, plan.route_index, plan.fuel_index, plan.speed_kn
        )
        result.n_evaluations += 1
        return solution

    for generation in range(n_generations):
        if time_limit_s is not None and time.perf_counter() - started > time_limit_s:
            result.time_limited = True
            logger.info("Q-07: time limit reached at generation %d", generation)
            break

        observed = np.array([observe(theta[i], rng) for i in range(n_individuals)])
        solutions = [evaluate(observed[i], speeds[i]) for i in range(n_individuals)]

        for index, solution in enumerate(solutions):
            if personal_best[index] is None or _constrained_better(
                solution, personal_best[index]
            ):
                personal_best[index] = solution
                personal_best_speed[index] = speeds[index].copy()
            if incumbent is None or _constrained_better(solution, incumbent):
                incumbent = solution
            result.archive = _update_archive(result.archive, solution, archive_capacity)

        # Each individual gets its OWN leader, drawn from the archive by crowding
        # distance. A single shared leader per generation pulls the whole population
        # toward one corner of the front and is the main reason a naive
        # multi-objective QIEA under-spreads; per-individual leaders let different
        # individuals pursue different regions of the trade-off simultaneously.
        leaders = [_pick_leader(result.archive, incumbent, rng)
                   for _ in range(n_individuals)]

        # --- rotation gate on the discrete block (Han & Kim, conditional) ----
        for index, (solution, leader) in enumerate(zip(solutions, leaders, strict=True)):
            if _constrained_better(solution, leader):
                continue  # this individual is already at least as good; leave it
            leader_bits = encoding.encode(
                leader.deploy, leader.route_index,
                np.array([problem.fuels.index(f) for f in leader.fuel]),
            )
            disagrees = observed[index] != leader_bits
            direction = np.where(leader_bits, 1.0, -1.0)
            theta[index] = np.clip(
                theta[index] + rotation_angle * direction * disagrees,
                theta_low, theta_high,
            )

        # --- QPSO on the continuous speed vector ----------------------------
        beta = beta_start + (beta_end - beta_start) * (
            generation / max(n_generations - 1, 1)
        )
        mbest = personal_best_speed.mean(axis=0)
        for index in range(n_individuals):
            phi = rng.random(problem.n_vessels)
            attractor = (
                phi * personal_best_speed[index]
                + (1.0 - phi) * leaders[index].speed_kn
            )
            u = rng.uniform(1e-12, 1.0, size=problem.n_vessels)
            step = beta * np.abs(mbest - speeds[index]) * np.log(1.0 / u)
            sign = np.where(rng.random(problem.n_vessels) < 0.5, -1.0, 1.0)
            speeds[index] = np.clip(attractor + sign * step, low_speed, high_speed)

        probability = np.sin(theta) ** 2
        result.generations = generation + 1
        result.history.append(len(result.archive))
        result.diversity.append(float(np.mean(4.0 * probability * (1.0 - probability))))
        result.best_violation.append(
            float(min(s.violation for s in solutions))
            if incumbent is None or not incumbent.feasible
            else 0.0
        )
        result.elapsed_s = time.perf_counter() - started
        if on_generation is not None and on_generation(generation, result) is False:
            result.time_limited = True
            logger.info("search stopped by caller at generation %d", generation)
            break

    result.elapsed_s = time.perf_counter() - started
    if not result.archive and incumbent is not None:
        logger.warning(
            "no feasible plan found; best violation %.4g. Check demand against "
            "fleet capacity with problem.check_demand_satisfiable().",
            incumbent.violation,
        )
    logger.info(result.summary())
    return result
