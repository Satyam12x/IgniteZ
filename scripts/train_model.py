"""Train the flagship energy model on the full panel and persist it for the API.

    python scripts/train_model.py                 # artifacts/model/energy_model.joblib

The model served by the decision-support platform is this one: XGBoost + monotone
constraint, with the QPSO-tuned hyperparameters from the P-13 ablation when that
has been run, and P10/P90 intervals calibrated on out-of-fold residuals from
vessels the fold never saw. A model card is written next to it so the platform can
show exactly what it is serving - data version, row counts, parameters, and the
held-out accuracy from ``artifacts/evaluation.json``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from greenfleet.prediction.features import LEAKAGE_FREE_FEATURES, build_features, build_target
from greenfleet.prediction.model import EnergyModel
from greenfleet.prediction.splits import grouped_folds

logger = logging.getLogger(__name__)

MODEL_DIR = Path("artifacts/model")
ABLATION_PATH = Path("artifacts/ablation_p13.json")
EVALUATION_PATH = Path("artifacts/evaluation.json")


def tuned_params() -> dict:
    if not ABLATION_PATH.exists():
        logger.warning("%s not found; training with XGBoost defaults", ABLATION_PATH)
        return {}
    payload = json.loads(ABLATION_PATH.read_text(encoding="utf-8"))
    return dict(payload["ablation"]["qpso"]["best_params"])


def load_panel(path: Path) -> pd.DataFrame:
    panel = pd.read_parquet(path)
    panel = panel[panel.is_usable].copy()
    panel["_y"] = build_target(panel)
    panel = panel[np.isfinite(panel._y) & (panel._y > 0)]
    return panel.reset_index(drop=True)


def out_of_fold_residuals(panel: pd.DataFrame, seed: int, n_folds: int, **kwargs) -> np.ndarray:
    """Interval calibration on vessels the fold never saw, as in evaluate_model.py."""
    features, target = build_features(panel), panel._y.values
    residuals = np.full(len(panel), np.nan)
    for fold, (train_idx, test_idx) in enumerate(grouped_folds(panel, n_splits=n_folds)):
        started = time.perf_counter()
        model = EnergyModel(seed=seed, **kwargs).fit(features.iloc[train_idx], target[train_idx])
        predicted = model.predict(features.iloc[test_idx]).energy_per_nmi_mj
        with np.errstate(divide="ignore", invalid="ignore"):
            residuals[test_idx] = np.log(target[test_idx]) - np.log(predicted)
        print(f"  fold {fold + 1}/{n_folds} done in {time.perf_counter() - started:.0f}s")
    return residuals


def held_out_summary(model_name: str) -> dict | None:
    """The leakage-safe accuracy the platform should quote, from the evaluation run."""
    if not EVALUATION_PATH.exists():
        return None
    payload = json.loads(EVALUATION_PATH.read_text(encoding="utf-8"))
    try:
        by_vessel = next(r for r in payload["splits"] if r["split"]["kind"] == "by_vessel")
        row = by_vessel["models"].get(model_name) or by_vessel["models"]["XGBoost + monotone"]
    except (KeyError, StopIteration):
        return None
    intervals = row.get("intervals") or {}
    return {
        "split": "by_vessel: 25% of ships held out, no ship on both sides",
        "n_test_ship_years": by_vessel["split"].get("n_test"),
        "n_test_ships": by_vessel["split"].get("test_ships"),
        "mape_pct": row.get("mape"),
        "r2": row.get("r2"),
        "median_ape_pct": row.get("median_ape"),
        "interval_coverage_80": intervals.get("empirical_coverage"),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--panel", type=Path, default=Path("data/interim/mrv_derived.parquet"))
    parser.add_argument("--seed", type=int, default=1000)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--output-dir", type=Path, default=MODEL_DIR)
    parser.add_argument("--no-calibration", action="store_true",
                        help="skip out-of-fold interval calibration (faster, optimistic bands)")
    args = parser.parse_args()
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(message)s")

    panel = load_panel(args.panel)
    periods = sorted(int(p) for p in panel.reporting_period.dropna().unique())
    print(f"panel: {len(panel):,} ship-years, {panel.imo.nunique():,} ships, periods {periods}")

    params = dict(residual_model="xgboost", monotone=True, **tuned_params())
    oof = None
    if not args.no_calibration:
        print(f"calibrating intervals on {args.folds} grouped folds ...")
        oof = out_of_fold_residuals(panel, args.seed, args.folds, **params)

    print("fitting the flagship model on the full panel ...")
    started = time.perf_counter()
    features = build_features(panel)
    model = EnergyModel(seed=args.seed, **params).fit(features, panel._y.values, oof_residuals=oof)
    fit_s = time.perf_counter() - started
    print(f"  fitted in {fit_s:.0f}s")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    model_path = args.output_dir / "energy_model.joblib"
    joblib.dump(model, model_path)

    name = "XGBoost + mono + QPSO" if tuned_params() else "XGBoost + monotone"
    card = {
        "name": name,
        "trained_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "panel": {
            "path": str(args.panel),
            "sha256": hashlib.sha256(args.panel.read_bytes()).hexdigest(),
            "n_ship_years": int(len(panel)),
            "n_ships": int(panel.imo.nunique()),
            "reporting_periods": periods,
        },
        "features": list(LEAKAGE_FREE_FEATURES),
        "params": {k: (float(v) if isinstance(v, (int, float)) else v) for k, v in params.items()},
        "seed": args.seed,
        "intervals": (
            f"P10/P90 calibrated on {args.folds}-fold grouped out-of-fold residuals"
            if oof is not None else "in-sample residuals (optimistic)"
        ),
        "held_out": held_out_summary(name),
        "fit_seconds": round(fit_s, 1),
        "python": sys.version.split()[0],
    }
    (args.output_dir / "model_card.json").write_text(json.dumps(card, indent=2), encoding="utf-8")
    print(f"wrote {model_path} ({model_path.stat().st_size / 1e6:.1f} MB) and model_card.json")


if __name__ == "__main__":
    main()
