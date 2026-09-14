"""Quantum-behaved Particle Swarm Optimization (QPSO), and its honest ablation.

What makes this "quantum-inspired", precisely
---------------------------------------------
Classical PSO moves each particle by a velocity vector. QPSO (Sun, Feng & Xu, 2004)
drops velocity entirely. It models each particle as a quantum particle bound in a
delta potential well centred on a local attractor ``p``. Solving the Schrodinger
equation for that well gives an exponential probability density for the particle's
position, and sampling it by inverse transform yields the update

    X(t+1) = p  +/-  beta * |mbest - X(t)| * ln(1/u),      u ~ U(0,1)

where ``mbest`` is the mean of all personal bests and ``beta`` is the
contraction-expansion coefficient. A particle therefore has a non-zero chance of
appearing anywhere in the space at every iteration - the tunnelling analogue - which
is what gives QPSO its global search behaviour and makes it harder to trap in a
local optimum than velocity-based PSO.

It is **not** a quantum computer and claims no quantum speedup. It is a classical
algorithm whose update rule is derived from a quantum model, and it runs on ordinary
hardware. That distinction is the one to state plainly to a quantum-expert judge.

This module is deliberately general. Phase 3 needs exactly this for the continuous
speed variables in the hybrid fleet encoding (Q-02), alongside Q-bit encoding for
the discrete vessel and fuel choices.

Honest benchmarking, built in
-----------------------------
``random_search`` is provided with an identical evaluation budget so the ablation
(P-13) is fair by construction rather than by good intentions (B-01). Diversity is
tracked every iteration so premature convergence is visible rather than inferred
(Q-03), and the same seed always reproduces the same run (Q-06).
"""

from __future__ import annotations

import logging
import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field

import numpy as np

logger = logging.getLogger(__name__)

__all__ = [
    "Parameter",
    "SearchSpace",
    "QPSOResult",
    "qpso_minimise",
    "random_search",
]


@dataclass(frozen=True, slots=True)
class Parameter:
    """One search dimension.

    Args:
        name: parameter name passed to the objective.
        low: lower bound (inclusive).
        high: upper bound (inclusive).
        kind: ``"float"``, ``"int"``, or ``"log"`` for a log-uniform float, which
            is the right treatment for learning rates and regularisation weights.
    """

    name: str
    low: float
    high: float
    kind: str = "float"

    def __post_init__(self) -> None:
        if self.kind not in {"float", "int", "log"}:
            raise ValueError(f"{self.name}: kind must be float, int or log, got {self.kind!r}")
        if self.high <= self.low:
            raise ValueError(f"{self.name}: high ({self.high}) must exceed low ({self.low})")
        if self.kind == "log" and self.low <= 0:
            raise ValueError(f"{self.name}: log parameters need a strictly positive low bound")

    # Internally every dimension is searched on a unit-free scale so that a single
    # beta applies sensibly to learning rate and tree depth alike.
    def to_internal(self, value: float) -> float:
        if self.kind == "log":
            return math.log(value)
        return float(value)

    def from_internal(self, value: float) -> float | int:
        if self.kind == "log":
            return float(math.exp(value))
        if self.kind == "int":
            return int(round(value))
        return float(value)

    @property
    def internal_bounds(self) -> tuple[float, float]:
        return self.to_internal(self.low), self.to_internal(self.high)


@dataclass(slots=True)
class SearchSpace:
    """An ordered collection of parameters."""

    parameters: tuple[Parameter, ...]

    def __post_init__(self) -> None:
        names = [p.name for p in self.parameters]
        if len(set(names)) != len(names):
            raise ValueError("duplicate parameter name in search space")
        if not self.parameters:
            raise ValueError("search space is empty")

    def __len__(self) -> int:
        return len(self.parameters)

    @property
    def lower(self) -> np.ndarray:
        return np.array([p.internal_bounds[0] for p in self.parameters], dtype="float64")

    @property
    def upper(self) -> np.ndarray:
        return np.array([p.internal_bounds[1] for p in self.parameters], dtype="float64")

    def decode(self, position: np.ndarray) -> dict[str, float | int]:
        """Internal vector -> keyword arguments for the objective."""
        clipped = np.clip(position, self.lower, self.upper)
        return {p.name: p.from_internal(v) for p, v in zip(self.parameters, clipped, strict=True)}

    def sample(self, rng: np.random.Generator, n: int) -> np.ndarray:
        return rng.uniform(self.lower, self.upper, size=(n, len(self.parameters)))


@dataclass
class QPSOResult:
    """Outcome of a search, including the trace needed for convergence plots."""

    best_params: dict[str, float | int]
    best_score: float
    n_evaluations: int
    history: list[float] = field(default_factory=list)
    """Best-so-far score per iteration (B-05: convergence plots)."""

    diversity: list[float] = field(default_factory=list)
    """Mean normalised spread of the swarm per iteration (Q-03)."""

    evaluations: list[tuple[dict[str, float | int], float]] = field(default_factory=list)
    algorithm: str = "qpso"
    seed: int = 0

    def converged_early(self, threshold: float = 1e-3, before_fraction: float = 0.5) -> bool:
        """Q-03: did swarm diversity collapse before the run was half done?

        A diversity of zero early means the rotation/contraction was too aggressive
        and the search stopped exploring - the failure mode to detect, not discover
        after the fact.
        """
        if not self.diversity:
            return False
        cutoff = max(1, int(len(self.diversity) * before_fraction))
        return bool(np.min(self.diversity[:cutoff]) < threshold)

    def summary(self) -> str:
        return (
            f"{self.algorithm}: best={self.best_score:.5f} after "
            f"{self.n_evaluations} evaluations (seed {self.seed})"
        )


def _beta(iteration: int, n_iterations: int, beta_start: float, beta_end: float) -> float:
    """Linearly decreasing contraction-expansion coefficient.

    Sun et al. recommend decreasing beta over the run: large early for exploration,
    small late for exploitation. Values above ~1.8 are known to diverge.
    """
    if n_iterations <= 1:
        return beta_end
    fraction = iteration / (n_iterations - 1)
    return beta_start + (beta_end - beta_start) * fraction


def qpso_minimise(
    objective: Callable[..., float],
    space: SearchSpace,
    n_particles: int = 12,
    n_iterations: int = 15,
    beta_start: float = 1.0,
    beta_end: float = 0.4,
    seed: int = 1000,
    on_evaluation: Callable[[int, dict[str, float | int], float], None] | None = None,
) -> QPSOResult:
    """Minimise ``objective`` over ``space`` with QPSO.

    The objective is called with the decoded parameters as keyword arguments and
    must return a finite score to minimise; a non-finite return is treated as
    ``+inf`` so a crashed configuration cannot win.

    Budget is ``n_particles * n_iterations`` evaluations, plus the initial swarm.
    """
    if n_particles < 2:
        raise ValueError("QPSO needs at least 2 particles")
    if n_iterations < 1:
        raise ValueError("n_iterations must be at least 1")
    if not 0 < beta_end <= beta_start < 1.8:
        raise ValueError(
            f"require 0 < beta_end <= beta_start < 1.8, got {beta_start} and {beta_end}"
        )

    rng = np.random.default_rng(seed)
    lower, upper = space.lower, space.upper
    span = np.where(upper > lower, upper - lower, 1.0)

    positions = space.sample(rng, n_particles)
    result = QPSOResult(best_params={}, best_score=math.inf, n_evaluations=0,
                        algorithm="qpso", seed=seed)

    def evaluate(position: np.ndarray) -> float:
        params = space.decode(position)
        try:
            raw = float(objective(**params))
        except Exception as exc:  # a bad configuration must not kill the search
            logger.warning("objective failed for %s: %s", params, exc)
            raw = math.inf
        score = raw if math.isfinite(raw) else math.inf
        result.n_evaluations += 1
        result.evaluations.append((params, score))
        if on_evaluation is not None:
            on_evaluation(result.n_evaluations, params, score)
        if score < result.best_score:
            result.best_score = score
            result.best_params = params
        return score

    personal_best = positions.copy()
    personal_best_score = np.array([evaluate(p) for p in positions], dtype="float64")
    best_index = int(np.argmin(personal_best_score))
    global_best = personal_best[best_index].copy()
    global_best_score = float(personal_best_score[best_index])

    for iteration in range(n_iterations):
        beta = _beta(iteration, n_iterations, beta_start, beta_end)
        mbest = personal_best.mean(axis=0)

        for index in range(n_particles):
            # Local attractor: a random convex blend of this particle's own best
            # and the global best.
            phi = rng.random(len(space))
            attractor = phi * personal_best[index] + (1.0 - phi) * global_best

            # Inverse-transform sample from the delta-well density. u is bounded
            # away from zero so ln(1/u) cannot overflow.
            u = rng.uniform(1e-12, 1.0, size=len(space))
            step = beta * np.abs(mbest - positions[index]) * np.log(1.0 / u)
            sign = np.where(rng.random(len(space)) < 0.5, -1.0, 1.0)
            candidate = np.clip(attractor + sign * step, lower, upper)

            score = evaluate(candidate)
            positions[index] = candidate
            if score < personal_best_score[index]:
                personal_best_score[index] = score
                personal_best[index] = candidate
            if score < global_best_score:
                global_best_score = score
                global_best = candidate.copy()

        result.history.append(float(result.best_score))
        spread = float(np.mean(np.std(positions, axis=0) / span))
        result.diversity.append(spread)
        logger.debug(
            "qpso iter %d/%d beta=%.3f best=%.5f diversity=%.4f",
            iteration + 1, n_iterations, beta, result.best_score, spread,
        )

    if result.converged_early():
        logger.warning(
            "Q-03: swarm diversity collapsed in the first half of the run "
            "(min %.2e). Consider a larger beta_end or more particles.",
            min(result.diversity),
        )
    logger.info(result.summary())
    return result


def random_search(
    objective: Callable[..., float],
    space: SearchSpace,
    n_evaluations: int,
    seed: int = 1000,
) -> QPSOResult:
    """Uniform random search, for a budget-matched ablation (B-01, P-13).

    This is the baseline that decides whether the quantum-inspired tuner earns its
    place. Give it exactly the evaluation count QPSO used and report the comparison
    either way.
    """
    if n_evaluations < 1:
        raise ValueError("n_evaluations must be at least 1")
    rng = np.random.default_rng(seed)
    result = QPSOResult(best_params={}, best_score=math.inf, n_evaluations=0,
                        algorithm="random_search", seed=seed)
    for _ in range(n_evaluations):
        params = space.decode(space.sample(rng, 1)[0])
        try:
            raw = float(objective(**params))
        except Exception as exc:
            logger.warning("objective failed for %s: %s", params, exc)
            raw = math.inf
        score = raw if math.isfinite(raw) else math.inf
        result.n_evaluations += 1
        result.evaluations.append((params, score))
        if score < result.best_score:
            result.best_score = score
            result.best_params = params
        result.history.append(float(result.best_score))
    logger.info(result.summary())
    return result


def summarise_runs(results: Sequence[QPSOResult]) -> dict[str, float]:
    """Mean/std/best across seeds, for the >=30-run reporting requirement (B-02)."""
    scores = np.array([r.best_score for r in results], dtype="float64")
    finite = scores[np.isfinite(scores)]
    if finite.size == 0:
        raise ValueError("no finite scores to summarise")
    return {
        "n_runs": int(scores.size),
        "mean": float(np.mean(finite)),
        "std": float(np.std(finite, ddof=1)) if finite.size > 1 else 0.0,
        "best": float(np.min(finite)),
        "worst": float(np.max(finite)),
        "median": float(np.median(finite)),
    }
