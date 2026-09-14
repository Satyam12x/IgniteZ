"""Phase 1b: full leakage-safe evaluation of the energy model.

Produces the metrics table that the quantum-inspired optimizer phase will be
measured against, across all three split types and every baseline (B-04).
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from greenfleet.prediction.features import build_features, build_target
from greenfleet.prediction.metrics import (
    assert_positive_at_zero_speed,
    assert_speed_monotonicity,
    interval_metrics,
    metrics_by_group,
    regression_metrics,
)
from greenfleet.prediction.model import EnergyModel
from greenfleet.prediction.splits import (
    grouped_folds,
    split_by_time,
    split_by_vessel,
    split_by_vessel_and_time,
)
from greenfleet.reproducibility import RunContext

logger = logging.getLogger(__name__)

BASELINES = {
    "class mean (naive)": dict(residual_model="class_mean"),
    "ridge (log)": dict(residual_model="ridge"),
    "XGBoost": dict(residual_model="xgboost", monotone=False),
    "XGBoost + monotone": dict(residual_model="xgboost", monotone=True),
}

ABLATION_PATH = Path("artifacts/ablation_p13.json")
SELECTION_PATH = Path("artifacts/feature_selection.json")
FLAGSHIP = "XGBoost + monotone"


def with_qpso_tuned(baselines: dict) -> dict:
    """Add the QPSO-tuned configuration if the ablation has been run.

    Kept as a separate row rather than folded into the monotone model, so the
    effect of tuning stays separable from the effect of the monotone constraint.
    """
    if not ABLATION_PATH.exists():
        logger.warning("%s not found; skipping the QPSO-tuned row", ABLATION_PATH)
        return baselines
    payload = json.loads(ABLATION_PATH.read_text(encoding="utf-8"))
    params = payload["ablation"]["qpso"]["best_params"]
    return baselines | {
        "XGBoost + mono + QPSO": dict(residual_model="xgboost", monotone=True, **params)
    }


def load_panel(path: Path) -> pd.DataFrame:
    panel = pd.read_parquet(path)
    panel = panel[panel.is_usable].copy()
    panel["_y"] = build_target(panel)
    panel = panel[np.isfinite(panel._y) & (panel._y > 0)]
    return panel.reset_index(drop=True)


def out_of_fold_residuals(train: pd.DataFrame, seed: int, n_folds: int = 5,
                          **model_kwargs) -> np.ndarray:
    """Honest interval calibration: residuals from vessels the fold never saw."""
    features, target = build_features(train), train._y.values
    residuals = np.full(len(train), np.nan)
    for train_idx, test_idx in grouped_folds(train, n_splits=n_folds):
        model = EnergyModel(seed=seed, **model_kwargs)
        model.fit(features.iloc[train_idx], target[train_idx])
        predicted = model.predict(features.iloc[test_idx]).energy_per_nmi_mj
        with np.errstate(divide="ignore", invalid="ignore"):
            residuals[test_idx] = np.log(target[test_idx]) - np.log(predicted)
    return residuals


def qiea_selected_features() -> list[str] | None:
    """Feature subset chosen by the Q-bit feature selector, if it has been run."""
    if not SELECTION_PATH.exists():
        return None
    payload = json.loads(SELECTION_PATH.read_text(encoding="utf-8"))
    return list(payload["qiea"]["features"])


def mask_to_selection(features: pd.DataFrame, selected: list[str]) -> pd.DataFrame:
    """Blank unselected numeric features with a constant column.

    vessel_class is structural and always kept: the physics core needs it for the
    class reference speed and auxiliary share.
    """
    masked = features.copy()
    for column in features.columns:
        if column == "vessel_class" or column in selected:
            continue
        masked[column] = 0.0
    return masked


def evaluate(split, seed: int, calibrate: bool) -> dict:
    train, test = split.train, split.test
    Xtr, ytr = build_features(train), train._y.values
    Xte, yte = build_features(test), test._y.values

    selected = qiea_selected_features()
    variants = dict(with_qpso_tuned(BASELINES))
    if selected is not None:
        variants["XGB + mono + QIEA feats"] = dict(
            residual_model="xgboost", monotone=True, _selection=selected
        )

    rows, report = [], {"split": split.as_dict(), "models": {}}
    for name, raw_kwargs in variants.items():
        kwargs = {k: v for k, v in raw_kwargs.items() if k != "_selection"}
        subset = raw_kwargs.get("_selection")
        train_features = Xtr if subset is None else mask_to_selection(Xtr, subset)
        test_features = Xte if subset is None else mask_to_selection(Xte, subset)
        oof = None
        if calibrate and kwargs.get("residual_model") == "xgboost":
            oof = out_of_fold_residuals(train, seed, **kwargs)
        model = EnergyModel(seed=seed, **kwargs).fit(
            train_features, ytr, oof_residuals=oof
        )
        prediction = model.predict(test_features)
        metrics = regression_metrics(yte, prediction.energy_per_nmi_mj)
        entry = metrics.as_dict()
        try:
            entry["intervals"] = interval_metrics(
                yte, prediction.p10_energy_per_nmi_mj, prediction.p90_energy_per_nmi_mj
            )
        except ValueError:
            entry["intervals"] = None
        report["models"][name] = entry
        rows.append((name, metrics, entry["intervals"]))

        if name == FLAGSHIP:
            report["per_class"] = metrics_by_group(
                yte, prediction.energy_per_nmi_mj, Xte.vessel_class.values
            ).to_dict("records")
            assert_speed_monotonicity(model, test_features.head(2000))
            assert_positive_at_zero_speed(model, test_features.head(2000))
            report["physics_assertions"] = {"P-02": "pass", "P-03": "pass"}

            # P-05: a model that says "I do not know" on rows with no design
            # certificate is behaving correctly. Reporting warm and cold rows
            # together hides that, and makes the cold rows look like a defect
            # rather than a declared limitation.
            report["cold_start"] = {}
            print(f"    {'-' * 68}")
            print(f"    P-05 cold-start stratification "
                  f"({int(prediction.cold_start.sum())} of {len(Xte)} rows lack a "
                  f"design certificate)")
            for label, mask in [("warm (has certificate)", ~prediction.cold_start),
                                ("cold (no certificate)", prediction.cold_start)]:
                if mask.sum() < 10:
                    continue
                sub = regression_metrics(yte[mask], prediction.energy_per_nmi_mj[mask])
                sub_iv = interval_metrics(
                    yte[mask],
                    prediction.p10_energy_per_nmi_mj[mask],
                    prediction.p90_energy_per_nmi_mj[mask],
                )
                report["cold_start"][label] = {**sub.as_dict(), "intervals": sub_iv}
                print(f"    {label:<24}{sub.mae:>9.1f}{sub.rmse:>9.1f}{sub.mape:>8.1f}"
                      f"{sub.median_ape:>9.1f}{sub.r2:>8.3f}"
                      f"{100 * sub_iv['empirical_coverage']:>7.1f}")

    print(f"\n=== {split.kind}: {split.detail} ===")
    print(f"    train {split.n_train} rows / {split.train.imo.nunique()} ships  ->  "
          f"test {split.n_test} rows / {split.test.imo.nunique()} ships")
    print(f"    {'model':<24}{'MAE':>9}{'RMSE':>9}{'MAPE%':>8}{'medAPE%':>9}{'R2':>8}{'cov%':>7}")
    for name, m, iv in rows:
        cov = f"{100 * iv['empirical_coverage']:.1f}" if iv else "-"
        print(f"    {name:<24}{m.mae:>9.1f}{m.rmse:>9.1f}{m.mape:>8.1f}"
              f"{m.median_ape:>9.1f}{m.r2:>8.3f}{cov:>7}")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--panel", type=Path, default=Path("data/interim/mrv_derived.parquet"))
    parser.add_argument("--seed", type=int, default=1000)
    parser.add_argument("--output", type=Path, default=Path("artifacts/evaluation.json"))
    parser.add_argument("--calibrate-intervals", action="store_true",
                        help="calibrate P10/P90 on out-of-fold residuals (slower, honest)")
    args = parser.parse_args()

    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(message)s")
    panel = load_panel(args.panel)
    print(f"panel: {len(panel)} ship-years, {panel.imo.nunique()} ships, "
          f"periods {sorted(int(p) for p in panel.reporting_period.dropna().unique())}")

    splits = [
        split_by_vessel(panel, test_size=0.25, seed=args.seed),
        split_by_time(panel, train_through=2023),
        split_by_vessel_and_time(panel, train_through=2023, test_size=0.25, seed=args.seed),
    ]
    reports = [evaluate(s, args.seed, args.calibrate_intervals) for s in splits]

    print("\n=== per vessel class (XGBoost + monotone, unseen vessels) - P-12 ===")
    table = pd.DataFrame(reports[0]["per_class"])
    table = table[table.group != "__overall__"].sort_values("n", ascending=False)
    print(f"    {'class':<24}{'n':>7}{'MAE':>9}{'MAPE%':>8}{'R2':>8}  reliable")
    for _, r in table.iterrows():
        print(f"    {r.group:<24}{int(r.n):>7}{r.mae:>9.1f}{r.mape:>8.1f}{r.r2:>8.3f}"
              f"{'':>4}{'yes' if r.reliable else 'NO'}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({
        "run": RunContext(seed=args.seed, stage="prediction.evaluate",
                          inputs={"panel": str(args.panel), "rows": len(panel),
                                  "calibrated": args.calibrate_intervals}).as_dict(),
        "splits": reports,
    }, indent=2, default=str), encoding="utf-8")
    print(f"\nwrote {args.output}")


if __name__ == "__main__":
    main()
