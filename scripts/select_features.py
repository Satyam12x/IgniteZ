"""Quantum-inspired feature selection for the energy model (deliverable 1).

Feature selection is a genuine 2^n combinatorial problem, which is what the Q-bit
encoding and rotation gate are actually good at - unlike smooth hyperparameter
tuning, where the earlier QPSO ablation showed no advantage.

This also keeps the honesty requirement: QIEA, budget-matched random search, and
(because the space is small enough) **exhaustive search for the true global
optimum** all run on identical grouped CV folds. With the exhaustive optimum known,
"QIEA found the best subset" is a verified statement rather than a hopeful one.

Vessel class is structural, not selectable: the physics core needs it for the class
reference speed and auxiliary share. Unselected numeric features are replaced by a
constant column, which carries no information to a tree - equivalent to removal,
while keeping the model's required feature contract intact.
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path

import numpy as np
import pandas as pd

from greenfleet.prediction.features import build_features, build_target
from greenfleet.prediction.metrics import regression_metrics
from greenfleet.prediction.model import EnergyModel
from greenfleet.prediction.qiea import (
    exhaustive_search,
    qiea_minimise,
    random_bit_search,
)
from greenfleet.prediction.splits import grouped_folds, split_by_vessel
from greenfleet.reproducibility import RunContext

logger = logging.getLogger(__name__)

# Selectable features. vessel_class is excluded: it is structural.
SELECTABLE: tuple[str, ...] = (
    "design_efficiency_gco2_per_t_nmi",
    "has_design_efficiency",
    "efficiency_metric_eiv",
    "efficiency_metric_eedi",
    "efficiency_metric_eexi",
    "has_ice_class",
    "ice_class_ordinal",
    "ice_time_share",
    "reporting_period",
    "time_at_sea_h",
)


def mask_features(features: pd.DataFrame, bits: np.ndarray) -> pd.DataFrame:
    """Blank out unselected features with a constant column."""
    masked = features.copy()
    for selected, name in zip(bits, SELECTABLE, strict=True):
        if not selected:
            masked[name] = 0.0
    return masked


def make_objective(frame: pd.DataFrame, n_folds: int, seed: int, n_estimators: int):
    """Grouped-CV MAPE as a function of a feature-selection bit string."""
    features = build_features(frame).reset_index(drop=True)
    target = frame["_y"].to_numpy(dtype="float64")
    folds = grouped_folds(frame.reset_index(drop=True), n_splits=n_folds)

    def objective(bits: np.ndarray) -> float:
        masked = mask_features(features, bits)
        scores = []
        for train_idx, test_idx in folds:
            model = EnergyModel(seed=seed, n_estimators=n_estimators, max_depth=6)
            model.fit(masked.iloc[train_idx], target[train_idx])
            predicted = model.predict(masked.iloc[test_idx]).energy_per_nmi_mj
            scores.append(regression_metrics(target[test_idx], predicted).mape)
        return float(np.mean(scores))

    return objective


def repair(bits: np.ndarray) -> np.ndarray:
    """Q-05: the empty subset is the only infeasible string. Repair, do not penalise."""
    if not bits.any():
        repaired = bits.copy()
        # Design efficiency is the single most informative feature, so it is the
        # sensible minimal subset.
        repaired[0] = True
        return repaired
    return bits


def describe(bits: np.ndarray) -> list[str]:
    return [name for selected, name in zip(bits, SELECTABLE, strict=True) if selected]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--panel", type=Path, default=Path("data/interim/mrv_derived.parquet"))
    parser.add_argument("--ships", type=int, default=4000, help="vessel-grouped subsample")
    parser.add_argument("--folds", type=int, default=3)
    parser.add_argument("--estimators", type=int, default=150)
    parser.add_argument("--generations", type=int, default=40)
    parser.add_argument("--individuals", type=int, default=8)
    parser.add_argument("--seed", type=int, default=1000)
    parser.add_argument("--exhaustive", action="store_true",
                        help="also brute-force all 2^n subsets to verify the optimum")
    parser.add_argument("--output", type=Path, default=Path("artifacts/feature_selection.json"))
    args = parser.parse_args()

    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(message)s")

    panel = pd.read_parquet(args.panel)
    panel = panel[panel.is_usable].copy()
    panel["_y"] = build_target(panel)
    panel = panel[np.isfinite(panel._y) & (panel._y > 0)]
    train = split_by_vessel(panel.reset_index(drop=True), test_size=0.25, seed=args.seed).train

    ships = train.imo.dropna().unique()
    keep = set(np.random.default_rng(args.seed).choice(
        ships, size=min(args.ships, len(ships)), replace=False
    ))
    subset = train[train.imo.isin(keep)].reset_index(drop=True)
    print(f"selecting over {len(SELECTABLE)} features = {2 ** len(SELECTABLE)} subsets")
    print(f"objective: {args.folds}-fold grouped CV MAPE on "
          f"{len(subset)} rows / {subset.imo.nunique()} ships\n")

    objective = make_objective(subset, args.folds, args.seed, args.estimators)

    all_on = np.ones(len(SELECTABLE), dtype=bool)
    started = time.perf_counter()
    baseline = objective(all_on)
    per_eval = time.perf_counter() - started
    print(f"all features on        : {baseline:.4f}  ({per_eval:.2f}s per evaluation)")

    qiea = qiea_minimise(
        objective, len(SELECTABLE), n_individuals=args.individuals,
        n_generations=args.generations, repair=repair, seed=args.seed,
    )
    print(f"QIEA                   : {qiea.best_score:.4f}  "
          f"({qiea.n_evaluations} distinct evals, {qiea.repairs} repairs)")

    random = random_bit_search(
        objective, len(SELECTABLE), n_evaluations=qiea.n_evaluations,
        repair=repair, seed=args.seed,
    )
    print(f"random (equal budget)  : {random.best_score:.4f}")

    payload = {
        "run": RunContext(seed=args.seed, stage="prediction.feature_selection",
                          inputs={"rows": len(subset), "folds": args.folds,
                                  "features": list(SELECTABLE)}).as_dict(),
        "baseline_all_features": baseline,
        "qiea": {"score": qiea.best_score, "features": describe(qiea.best_bits),
                 "n_evaluations": qiea.n_evaluations, "repairs": qiea.repairs,
                 "diversity": qiea.diversity, "history": qiea.history,
                 "converged_early": qiea.converged_early()},
        "random_search": {"score": random.best_score, "features": describe(random.best_bits),
                          "n_evaluations": random.n_evaluations},
    }

    if args.exhaustive:
        print(f"\nexhaustive: {2 ** len(SELECTABLE)} subsets, "
              f"~{2 ** len(SELECTABLE) * per_eval / 60:.0f} min estimated")
        best_bits, best_score, evaluations = exhaustive_search(
            objective, len(SELECTABLE), repair=repair, max_bits=len(SELECTABLE)
        )
        gap = 100.0 * (qiea.best_score - best_score) / best_score
        print(f"exhaustive optimum     : {best_score:.4f} over {evaluations} subsets")
        print(f"QIEA gap to optimum    : {gap:+.3f}%  "
              f"{'(EXACT)' if abs(gap) < 1e-9 else ''}")
        payload["exhaustive"] = {
            "score": best_score, "features": describe(best_bits),
            "n_evaluations": evaluations,
            "qiea_gap_pct": gap,
            "qiea_found_optimum": bool(abs(gap) < 1e-9),
            "search_fraction": qiea.n_evaluations / evaluations,
        }

    print(f"\nQIEA selected ({int(qiea.best_bits.sum())}/{len(SELECTABLE)}):")
    for name in describe(qiea.best_bits):
        print(f"    {name}")
    dropped = [n for n in SELECTABLE if n not in describe(qiea.best_bits)]
    if dropped:
        print("  dropped:")
        for name in dropped:
            print(f"    {name}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    print(f"\nwrote {args.output}")


if __name__ == "__main__":
    main()
