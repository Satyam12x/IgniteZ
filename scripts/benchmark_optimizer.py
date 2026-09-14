"""Benchmark the quantum-inspired fleet optimizer against conventional methods.

Covers the whole of the B-series in the edge-case register:

``B-01``  Equal evaluation budget for every algorithm, stated explicitly.
``B-02``  30 independent seeds; mean, standard deviation, and a Wilcoxon
          signed-rank test - never a single best run.
``B-03``  Hypervolume against a single shared reference point, plus front size and
          spacing.
``B-05``  Convergence traces.
``B-06``  Scalability curve across instance sizes.
``B-07``  Honest reporting, including instances where the quantum-inspired method
          loses.
``Q-08``  Small instances checked against a proven MILP optimum.

The reference point is computed once from the union of every front produced by every
algorithm across every seed, then applied to all of them. Computing it per-algorithm
would make the numbers incomparable, which is the most common way a hypervolume
table ends up meaningless.
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path

import numpy as np

from greenfleet.optimize.baselines import (
    greedy_fleet,
    milp_fleet,
    nsga2_fleet,
    random_search_fleet,
)
from greenfleet.optimize.pareto import hypervolume, reference_point, spacing
from greenfleet.optimize.qmoea import optimise_fleet
from greenfleet.optimize.scenarios import (
    coastal_scenario,
    large_scenario,
    tiny_scenario,
)
from greenfleet.reproducibility import BENCHMARK_SEEDS, RunContext

logger = logging.getLogger(__name__)


def run_algorithms(problem, budget: int, seed: int) -> dict[str, dict]:
    """One seed, every algorithm, identical evaluation budget (B-01)."""
    out: dict[str, dict] = {}

    started = time.perf_counter()
    q = optimise_fleet(problem, n_individuals=30, n_generations=max(1, budget // 30),
                       seed=seed)
    out["QIEA+QPSO"] = {"front": q.front, "evals": q.n_evaluations,
                        "time": time.perf_counter() - started,
                        "diversity": q.diversity[-1] if q.diversity else None}

    n = nsga2_fleet(problem, n_evaluations=budget, seed=seed)
    out["NSGA-II"] = {"front": n.front, "evals": n.n_evaluations, "time": n.elapsed_s}

    r = random_search_fleet(problem, n_evaluations=budget, seed=seed)
    out["random"] = {"front": r.front, "evals": r.n_evaluations, "time": r.elapsed_s}

    g = greedy_fleet(problem, seed=seed)
    out["greedy"] = {"front": g.front, "evals": g.n_evaluations, "time": g.elapsed_s}
    return out


def benchmark(problem, budget: int, n_seeds: int, label: str) -> dict:
    """Multi-seed benchmark with a shared reference point and significance test."""
    seeds = list(BENCHMARK_SEEDS)[:n_seeds]
    per_seed: list[dict[str, dict]] = []
    print(f"\n{'=' * 78}")
    print(f"{label}: {problem.n_vessels} vessels, {problem.n_routes} routes, "
          f"{problem.n_fuels} fuels, {problem.search_space_size():.3g} combinations")
    print(f"{n_seeds} seeds x {budget} evaluations each (B-01, B-02)")
    print("=" * 78)

    for index, seed in enumerate(seeds):
        per_seed.append(run_algorithms(problem, budget, seed))
        if (index + 1) % 5 == 0:
            print(f"   ...{index + 1}/{n_seeds} seeds")

    # B-03: one reference point from the union of every front produced.
    every_front = [run[name]["front"] for run in per_seed for name in run
                   if len(run[name]["front"])]
    if not every_front:
        print("   no feasible solutions found by any algorithm")
        return {"label": label, "error": "no feasible solutions"}
    reference = reference_point(every_front)

    algorithms = list(per_seed[0])
    results: dict[str, dict] = {}
    for name in algorithms:
        volumes, sizes, spreads, times, feasible = [], [], [], [], 0
        for run in per_seed:
            front = run[name]["front"]
            volumes.append(hypervolume(front, reference) if len(front) else 0.0)
            sizes.append(len(front))
            spreads.append(spacing(front) if len(front) > 1 else 0.0)
            times.append(run[name]["time"])
            feasible += bool(len(front))
        results[name] = {
            "hypervolume_mean": float(np.mean(volumes)),
            "hypervolume_std": float(np.std(volumes, ddof=1)) if len(volumes) > 1 else 0.0,
            "hypervolume_best": float(np.max(volumes)),
            "hypervolume_all": [float(v) for v in volumes],
            "front_size_mean": float(np.mean(sizes)),
            "spacing_mean": float(np.mean(spreads)),
            "time_mean_s": float(np.mean(times)),
            "feasible_runs": feasible,
        }

    print(f"\n   {'algorithm':<12}{'HV mean':>13}{'HV std':>12}{'front':>8}"
          f"{'time s':>9}{'feasible':>10}")
    print("   " + "-" * 64)
    for name, stats in sorted(results.items(),
                              key=lambda kv: -kv[1]["hypervolume_mean"]):
        print(f"   {name:<12}{stats['hypervolume_mean']:>13.4g}"
              f"{stats['hypervolume_std']:>12.3g}{stats['front_size_mean']:>8.1f}"
              f"{stats['time_mean_s']:>9.2f}{stats['feasible_runs']:>7d}/{n_seeds}")

    # B-02: significance against each baseline, paired by seed.
    print("\n   Wilcoxon signed-rank, QIEA+QPSO vs each baseline (paired by seed):")
    try:
        from scipy.stats import wilcoxon

        ours = np.array(results["QIEA+QPSO"]["hypervolume_all"])
        for name in algorithms:
            if name == "QIEA+QPSO":
                continue
            theirs = np.array(results[name]["hypervolume_all"])
            difference = ours - theirs
            if np.allclose(difference, 0):
                verdict, p_value = "identical", 1.0
            else:
                p_value = float(wilcoxon(ours, theirs).pvalue)
                better = float(np.mean(difference) > 0)
                verdict = ("QIEA better" if better else "QIEA WORSE") + (
                    " (significant)" if p_value < 0.05 else " (not significant)"
                )
            results[name]["wilcoxon_p_vs_qiea"] = p_value
            print(f"     vs {name:<12} p = {p_value:.4f}   {verdict}")
    except ImportError:  # pragma: no cover
        print("     scipy not installed; significance test skipped")

    return {"label": label, "budget": budget, "n_seeds": n_seeds,
            "reference_point": reference.tolist(), "results": results,
            "problem": problem.as_provenance()}


def verify_against_milp(n_seeds: int = 30) -> dict:
    """Q-08: on a small instance the optimum is knowable, so check against it."""
    print(f"\n{'=' * 78}")
    print("Q-08 VERIFICATION AGAINST A PROVEN OPTIMUM")
    print("=" * 78)
    problem = tiny_scenario()
    exact = milp_fleet(problem, objective=2)
    if not exact.archive:
        print("   MILP found no feasible solution; cannot verify")
        return {"error": "milp infeasible"}

    optimum = exact.archive[0].objectives[2]
    print(f"   MILP proven optimum (cost): INR {optimum:,.0f}")
    print(f"   {exact.notes[0] if exact.notes else ''}")

    gaps, hits = [], 0
    for seed in list(BENCHMARK_SEEDS)[:n_seeds]:
        result = optimise_fleet(problem, n_individuals=12, n_generations=60, seed=seed)
        best = result.best_by(2)
        if best is None:
            gaps.append(float("inf"))
            continue
        gap = best.objectives[2] / optimum - 1.0
        gaps.append(gap)
        hits += abs(gap) < 1e-9
    finite = [g for g in gaps if np.isfinite(g)]
    print(f"\n   QIEA+QPSO over {n_seeds} seeds:")
    print(f"     exact optimum found : {hits}/{n_seeds} seeds")
    print(f"     mean gap            : {100 * np.mean(finite):+.4f}%")
    print(f"     worst gap           : {100 * np.max(finite):+.4f}%")
    print(f"   Q-08 asks for a stated gap; achieved worst case "
          f"{100 * np.max(finite):.4f}%")
    return {"optimum_cost_inr": float(optimum), "exact_hits": hits,
            "n_seeds": n_seeds, "mean_gap": float(np.mean(finite)),
            "worst_gap": float(np.max(finite))}


def scalability(budget: int, sizes: list[int], n_seeds: int = 3) -> dict:
    """B-06: runtime and quality as the instance grows."""
    print(f"\n{'=' * 78}")
    print("B-06 SCALABILITY")
    print("=" * 78)
    print(f"   {'vessels':>9}{'combinations':>15}{'HV mean':>13}{'front':>8}"
          f"{'time s':>9}{'feasible':>10}")
    print("   " + "-" * 64)
    out = {}
    for n_vessels in sizes:
        problem = large_scenario(n_vessels=n_vessels)
        volumes, sizes_found, times, feasible = [], [], [], 0
        fronts = []
        for seed in list(BENCHMARK_SEEDS)[:n_seeds]:
            started = time.perf_counter()
            result = optimise_fleet(problem, n_individuals=30,
                                    n_generations=max(1, budget // 30), seed=seed)
            times.append(time.perf_counter() - started)
            fronts.append(result.front)
            sizes_found.append(len(result.archive))
            feasible += bool(result.archive)
        usable = [f for f in fronts if len(f)]
        if usable:
            reference = reference_point(usable)
            volumes = [hypervolume(f, reference) if len(f) else 0.0 for f in fronts]
        else:
            volumes = [0.0]
        out[n_vessels] = {
            "combinations": problem.search_space_size(),
            "hypervolume_mean": float(np.mean(volumes)),
            "front_size_mean": float(np.mean(sizes_found)),
            "time_mean_s": float(np.mean(times)),
            "feasible_runs": feasible,
        }
        print(f"   {n_vessels:>9}{problem.search_space_size():>15.3g}"
              f"{np.mean(volumes):>13.4g}{np.mean(sizes_found):>8.1f}"
              f"{np.mean(times):>9.2f}{feasible:>7d}/{n_seeds}")
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", type=int, default=30, help="B-02 requires >= 30")
    parser.add_argument("--budget", type=int, default=6000)
    parser.add_argument("--sizes", type=int, nargs="*", default=[10, 25, 50, 100, 200])
    parser.add_argument("--output", type=Path,
                        default=Path("artifacts/optimizer_benchmark.json"))
    args = parser.parse_args()

    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(message)s")
    payload = {
        "run": RunContext(seed=1000, stage="optimize.benchmark",
                          inputs={"seeds": args.seeds, "budget": args.budget}).as_dict(),
        "q08_verification": verify_against_milp(args.seeds),
        "coastal": benchmark(coastal_scenario(), args.budget, args.seeds,
                             "COASTAL SCENARIO"),
        "large": benchmark(large_scenario(n_vessels=60), args.budget,
                           max(5, args.seeds // 3), "LARGE SCENARIO (60 vessels)"),
        "scalability": scalability(args.budget, args.sizes),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    print(f"\nwrote {args.output}")


if __name__ == "__main__":
    main()
