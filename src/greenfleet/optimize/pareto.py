"""Pareto fronts and the metrics used to compare them.

The problem statement is explicit that the output is a **Pareto front**, not one
weighted answer (M-10). Fuel, emissions and cost genuinely conflict: the cheapest
plan is not the cleanest, and collapsing them into a single weighted score hides the
trade-off that a decision-maker is actually being asked to make.

Metrics here are the ones B-03 requires for honest benchmarking:

``hypervolume``
    Volume of objective space dominated by a front, measured against a shared
    reference point. Bigger is better. The reference point must be identical across
    every algorithm compared, or the numbers are meaningless - so it is computed
    once from the union of all fronts and passed in explicitly.
``igd``
    Mean distance from a reference front to the nearest solution found. Smaller is
    better. Used when a true or best-known front is available.
``spacing``
    How evenly solutions are distributed along the front. A front of three clustered
    points is worse for a decision-maker than five spread ones, even at equal
    hypervolume.
"""

from __future__ import annotations

import logging

import numpy as np

logger = logging.getLogger(__name__)

__all__ = [
    "non_dominated_mask",
    "pareto_front",
    "dominates",
    "hypervolume",
    "reference_point",
    "igd",
    "spacing",
    "crowding_distance",
]


def dominates(a: np.ndarray, b: np.ndarray) -> bool:
    """True if ``a`` dominates ``b``: no worse on every objective, better on one.

    All objectives are minimised.
    """
    a, b = np.asarray(a, dtype="float64"), np.asarray(b, dtype="float64")
    return bool(np.all(a <= b) and np.any(a < b))


def non_dominated_mask(objectives: np.ndarray) -> np.ndarray:
    """Boolean mask of the non-dominated rows of an ``(n, m)`` objective array."""
    values = np.asarray(objectives, dtype="float64")
    if values.ndim != 2:
        raise ValueError(f"objectives must be 2-D (n, m), got shape {values.shape}")
    n = len(values)
    mask = np.ones(n, dtype=bool)
    for i in range(n):
        if not mask[i]:
            continue
        for j in range(n):
            if i == j or not mask[j]:
                continue
            if dominates(values[j], values[i]):
                mask[i] = False
                break
    return mask


def pareto_front(
    objectives: np.ndarray,
    violations: np.ndarray | None = None,
) -> np.ndarray:
    """Indices of the non-dominated, feasible solutions.

    Infeasible solutions are excluded before the comparison, not ranked against
    feasible ones: a plan that fails to deliver the cargo is not a trade-off, it is
    not a plan (M-01).
    """
    values = np.asarray(objectives, dtype="float64")
    indices = np.arange(len(values))
    if violations is not None:
        feasible = np.asarray(violations, dtype="float64") <= 1e-9
        if not feasible.any():
            return np.array([], dtype=int)
        indices = indices[feasible]
        values = values[feasible]
    mask = non_dominated_mask(values)
    return indices[mask]


def reference_point(fronts: list[np.ndarray], margin: float = 0.1) -> np.ndarray:
    """A shared hypervolume reference point, from the union of several fronts.

    B-03 requires the same reference point for every algorithm being compared.
    Computing it from the union and offering it back to each run is how that is
    guaranteed rather than assumed.
    """
    usable = [np.asarray(f, dtype="float64") for f in fronts if len(f)]
    if not usable:
        raise ValueError("no non-empty fronts to derive a reference point from")
    stacked = np.vstack(usable)
    worst = stacked.max(axis=0)
    best = stacked.min(axis=0)
    span = np.where(worst > best, worst - best, np.abs(worst) + 1.0)
    return worst + margin * span


def hypervolume(front: np.ndarray, reference: np.ndarray) -> float:
    """Hypervolume dominated by ``front``, relative to ``reference``.

    Exact for two objectives by sweep; for three or more it delegates to pymoo when
    available and otherwise falls back to Monte Carlo with a fixed seed, which is
    reported as approximate. Minimisation throughout.
    """
    points = np.asarray(front, dtype="float64")
    reference = np.asarray(reference, dtype="float64")
    if points.ndim != 2:
        raise ValueError(f"front must be 2-D, got shape {points.shape}")
    if points.shape[1] != len(reference):
        raise ValueError(
            f"front has {points.shape[1]} objectives but reference has {len(reference)}"
        )
    # Only points that actually beat the reference contribute.
    points = points[np.all(points < reference, axis=1)]
    if len(points) == 0:
        return 0.0

    if points.shape[1] == 2:
        order = np.argsort(points[:, 0])
        ordered = points[order]
        volume = 0.0
        previous_y = reference[1]
        for x, y in ordered:
            if y < previous_y:
                volume += (reference[0] - x) * (previous_y - y)
                previous_y = y
        return float(volume)

    try:
        from pymoo.indicators.hv import HV

        return float(HV(ref_point=reference)(points))
    except Exception:  # pragma: no cover - exercised only without pymoo
        logger.info("pymoo unavailable; hypervolume estimated by Monte Carlo")
        rng = np.random.default_rng(1000)
        low = points.min(axis=0)
        samples = rng.uniform(low, reference, size=(200_000, len(reference)))
        dominated = np.any(
            np.all(samples[:, None, :] >= points[None, :, :], axis=2), axis=1
        )
        box = float(np.prod(reference - low))
        return float(dominated.mean() * box)


def igd(front: np.ndarray, reference_front: np.ndarray) -> float:
    """Inverted generational distance: mean distance from reference to found."""
    found = np.asarray(front, dtype="float64")
    target = np.asarray(reference_front, dtype="float64")
    if len(found) == 0:
        return float("inf")
    distances = np.linalg.norm(target[:, None, :] - found[None, :, :], axis=2)
    return float(distances.min(axis=1).mean())


def spacing(front: np.ndarray) -> float:
    """Standard deviation of nearest-neighbour distances. Lower is more even."""
    points = np.asarray(front, dtype="float64")
    if len(points) < 2:
        return 0.0
    distances = np.linalg.norm(points[:, None, :] - points[None, :, :], axis=2)
    np.fill_diagonal(distances, np.inf)
    nearest = distances.min(axis=1)
    return float(np.std(nearest))


def crowding_distance(front: np.ndarray) -> np.ndarray:
    """NSGA-II crowding distance, used to thin a front without losing its spread."""
    points = np.asarray(front, dtype="float64")
    n, m = points.shape
    distance = np.zeros(n)
    if n <= 2:
        return np.full(n, np.inf)
    for objective in range(m):
        order = np.argsort(points[:, objective])
        distance[order[0]] = distance[order[-1]] = np.inf
        span = points[order[-1], objective] - points[order[0], objective]
        if span <= 0:
            continue
        for position in range(1, n - 1):
            distance[order[position]] += (
                points[order[position + 1], objective]
                - points[order[position - 1], objective]
            ) / span
    return distance


def thin_front(front: np.ndarray, max_points: int) -> np.ndarray:
    """Indices of at most ``max_points`` solutions, keeping the spread (U-06).

    A Pareto front with a thousand points does not render legibly; one with three
    hides the trade-off. Crowding distance keeps the extremes and the gaps.
    """
    points = np.asarray(front, dtype="float64")
    if len(points) <= max_points:
        return np.arange(len(points))
    distances = crowding_distance(points)
    return np.argsort(-distances)[:max_points]
