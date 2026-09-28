# Q Fleet

Fuel consumption prediction and multi-objective fleet planning for cleaner coastal shipping.

Q Fleet decides which ships to run, on which fuel, at what speed, and shows the cost and the
whole-lifecycle emissions of every option. It combines three engines behind one FastAPI service
and a browser dashboard:

- **Fuel predictor.** A physics-informed model (imposed propeller cube law plus a
  monotone-constrained XGBoost residual) trained on 100,528 verified EU MRV ship-years. It
  predicts energy per nautical mile at any speed, with calibrated P10 to P90 intervals.
- **Well-to-wake emissions engine.** Upstream production, combustion CO₂, measured methane
  slip, N₂O and pilot fuel, per production pathway (grey, blue, green, e-fuel, bio).
- **Fleet planner.** A quantum-inspired multi-objective search (Q-bit evolutionary algorithm
  for the discrete choices, QPSO for speeds) over deployment, route, fuel and speed. It
  respects cargo demand, schedules, emission caps, tank range and year-by-year fuel
  availability at Indian ports, and returns a Pareto front rather than one weighted answer.

Quantum-inspired means classical algorithms that borrow quantum concepts; everything runs
on ordinary hardware.

## Requirements

- Python 3.10 or newer
- About 1 GB of disk for the environment

The processed data panel and the trained model are included, so no download is needed to run.

## Quick start

With [uv](https://docs.astral.sh/uv/):

```bash
uv venv
uv pip install -e ".[api,optim]"
uv run python -m greenfleet.api
```

With plain pip:

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -e ".[api,optim]"
python -m greenfleet.api
```

Open http://127.0.0.1:8000. The interactive API reference is at http://127.0.0.1:8000/docs.
Use `--port` to pick another port and `--host 0.0.0.0` to serve on a network.

## Dashboard

| Page | What it shows |
|---|---|
| Home | The recommended plan for the example fleet, the three best options, model accuracy and key facts |
| Plan my fleet | Choose a fleet, year and fuels; get the lowest-cost, balanced and lowest-emission plans, ship by ship |
| Compare fuels | Lifecycle CO₂ of every fuel and pathway against diesel, and what a funnel-only tool would report |
| Fuel calculator | Fuel burned per 1,000 nautical miles at each speed for a ship type, with the likely range |
| Ports | Fuel and shore-power availability at eight Indian ports by year, and the grid carbon trend |
| How it works | Model accuracy on unseen ships, search statistics, and an on-demand comparison with NSGA-II, QBHO and greedy |

## API

| Method | Route | Purpose |
|---|---|---|
| GET | `/api/health` | Service status and loaded model |
| GET | `/api/meta` | Fuels, ports, vessel classes, scenarios, policy targets |
| POST | `/api/predict` | Energy and fuel per nautical mile for a vessel, at one or many speeds |
| POST | `/api/emissions/compare` | Lifecycle emissions of every fuel option for a given energy demand |
| GET | `/api/emissions/electrification` | Battery-electric vs diesel CO₂ by year on the Indian grid |
| POST | `/api/optimise` | Start a fleet optimisation job (returns a job id) |
| GET | `/api/optimise/{job_id}` | Job progress and result |
| POST | `/api/optimise/{job_id}/cancel` | Cancel a running job |

An infeasible request (for example, a fleet that cannot move the demanded cargo) returns
HTTP 422 with the engine's explanation.

## Tests

```bash
uv pip install -e ".[dev,api,optim]"
pytest -m "not slow"               # 381 tests, about 30 seconds
pytest                             # all 416, includes benchmarks and raw-data integration tests
```

The integration tests need the raw data (see below).

## Rebuilding the model

Only needed to reproduce the served model from the source data.

```bash
python -m greenfleet.data.mrv.download   # 8 EMSA workbooks (about 44 MB) into data/raw/mrv/
python scripts/build_panel.py            # clean and derive -> data/interim/mrv_derived.parquet
python scripts/train_model.py            # fit and save -> artifacts/model/
```

The download writes a `manifest.json` that pins the reporting period, version and SHA-256 of
each workbook, because EMSA republishes a period whenever a company corrects a filing.

## Results

Fuel model, on 6,263 ships held out entirely from training (25% of vessels, no ship on both
sides of the split):

| Metric | Value |
|---|---|
| Mean absolute percentage error | 20.3% |
| Median absolute percentage error | 13.7% |
| R² | 0.59 |
| Real values inside the 80% interval | 80% |

Within-ship year-to-year variation in the data is about 11%, which sets a floor for any method.

Lifecycle accounting changes the verdict on several fuels (per unit of engine work, 2030):

| Fuel | Funnel-only view | Whole lifecycle |
|---|---|---|
| LNG, fossil | lower CO₂ | 4% less than diesel |
| Methanol, grey | lower CO₂ | 10% more than diesel |
| Ammonia, grey | zero CO₂ | 40% more than diesel |
| Hydrogen, grey | zero CO₂ | 9% more than diesel |
| Ammonia, green | zero CO₂ | 79% less than diesel |

Optimizer, equal evaluation budgets, hypervolume compared with a paired Wilcoxon test:

| Instance | Result |
|---|---|
| 2 ships, 1 route | Finds the proven optimum (CP-SAT) in 30 of 30 seeds |
| 12 ships, 3 routes | 6% above NSGA-II (not significant, p = 0.14); 20% above QBHO (p = 0.031) |
| 60 ships | 19% above NSGA-II (p = 0.004); 3% above QBHO (p = 0.014) |
| 200 ships | Completes in about 25 seconds |

## Project layout

```
src/greenfleet/
  config/         emission factors, ports, vessel classes (every number with a source)
  data/mrv/       EU MRV download, schema validation, cleaning, derived physics
  prediction/     physics core, features, leakage-safe splits, model, QIEA, QPSO, tuning
  emissions/      well-to-wake engine, port availability, tank range, Harit Sagar KPIs
  optimize/       fleet problem, Q-bit encoding, multi-objective search, baselines, QUBO
  api/            FastAPI service, job runner, and the dashboard (static/)
scripts/          build_panel.py, train_model.py
tests/            unit and integration tests
artifacts/model/  the served model and its model card
data/interim/     the processed MRV panel used for training
```

## Data and scope

Fuel data comes from the EU MRV "Publication of information" dataset published by EMSA
(THETIS-MRV): annual per-ship fuel consumption and CO₂ for ships of 5,000 GT and above calling
at EEA ports, reporting periods 2018 to 2025. No credentials are required.

The example fleets are representative, shaped like Indian coastal shipping, not an operator's
data. Port fuel availability by year, grid projections after 2023 and default fuel prices are
planning assumptions to be confirmed with port authorities. EU MRV contains no harbour tugs
(below 5,000 GT), so tug figures are physics-based estimates.

## Attribution

EU MRV data © European Union / EMSA, republished under the THETIS-MRV public publication of
information. Emission factor sources are listed in `src/greenfleet/config/fuels.yaml`.
