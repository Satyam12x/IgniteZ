"""Phase 1b: tune the energy model with QPSO and run the P-13 ablation."""
import logging

import numpy as np
import pandas as pd

from greenfleet.prediction.features import build_features, build_target
from greenfleet.prediction.splits import split_by_vessel
from greenfleet.prediction.tune import run_ablation

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

d = pd.read_parquet("data/interim/mrv_derived.parquet")
d = d[d.is_usable].copy()
d["_y"] = build_target(d)
d = d[np.isfinite(d._y) & (d._y > 0)].reset_index(drop=True)

sp = split_by_vessel(d, test_size=0.25, seed=1000)
train = sp.train
# Tune on a vessel-grouped subsample to keep the budget affordable; the final
# model is refitted on the full training set afterwards.
ships = train.imo.dropna().unique()
rng = np.random.default_rng(1000)
keep = set(rng.choice(ships, size=min(6000, len(ships)), replace=False))
tune_set = train[train.imo.isin(keep)].reset_index(drop=True)
logging.info("tuning on %d rows / %d ships", len(tune_set), tune_set.imo.nunique())

result = run_ablation(
    build_features(tune_set),
    tune_set._y.values,
    tune_set.imo,
    n_particles=8,
    n_iterations=6,
    n_folds=3,
    seed=1000,
    output_path="artifacts/ablation_p13.json",
)
print()
print("=" * 78)
print("P-13 ABLATION (CV MAPE, grouped folds, equal budget of %d evaluations)" % result.budget)
print("=" * 78)
print("  untuned default      : %.4f" % result.default_score)
print("  random search        : %.4f" % result.random.best_score)
print("  QPSO                 : %.4f" % result.qpso.best_score)
print()
print(result.verdict())
print()
print("QPSO best params:")
for k, v in sorted(result.qpso.best_params.items()):
    print("    %-20s %s" % (k, round(v, 5) if isinstance(v, float) else v))
