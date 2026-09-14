# Green Fleet Planner

**SIH 2026 · Problem Statement SIH26138 · Egreen Quanta**
Quantum-Inspired Fuel Consumption Prediction and Green Fleet Optimization

> A planner that decides **which ships to run, on which fuel, at what speed** — and
> shows the cost and the whole-lifecycle emissions of every option — trained on
> 100,000+ audited ship-years, checked against Indian port constraints.

---

## Run it in 10 minutes (step by step)

You need **Python 3.10 or newer** and **Git**. Nothing else is required — the data
panel, the trained model and the compiled dashboard are all in the repository.

### 1. Get the code

```bash
git clone <this-repository-url>
cd IgniteZ
```

### 2. Create a virtual environment and install

**Option A — with [uv](https://docs.astral.sh/uv/) (fast, recommended):**

```bash
pip install uv                              # once, if you don't have uv
uv venv                                     # creates .venv with Python 3.10+
uv pip install -e ".[dev,optim,api]"        # ~2 minutes
```

**Option B — plain pip:**

```bash
python -m venv .venv
# Windows:            .venv\Scripts\activate
# macOS / Linux:      source .venv/bin/activate
pip install -e ".[dev,optim,api]"
```

From here on, either run commands with `uv run python ...` or activate the venv
(`.venv\Scripts\activate` on Windows, `source .venv/bin/activate` elsewhere) and use
`python ...` directly. The commands below assume the venv is activated.

### 3. Check that everything works

```bash
pytest -m "not slow" -q
```

Expected: 380 tests pass in about 30 seconds. (The full suite, including the
slow integration tests that parse the raw MRV workbooks, is `pytest -q`; it takes
about 10 minutes and needs the raw data from step 6.)

### 4. Start the platform

```bash
python -m greenfleet.api
```

Open **http://127.0.0.1:8000** in a browser. The page runs the example fleet on load
and shows a finished result in about 4 seconds. The API docs are at
`http://127.0.0.1:8000/docs`. Use `--port 8011` if 8000 is taken.

### 5. Learn to use it

- **[docs/USER_GUIDE.pdf](docs/USER_GUIDE.pdf)** — every screen, every control, every
  number, with screenshots. Read this first.
- **[docs/STUDY_GUIDE.pdf](docs/STUDY_GUIDE.pdf)** — what the project is and why, in
  plain language, with a glossary.
- **[docs/DEMO_GUIDE.md](docs/DEMO_GUIDE.md)** — what to say at each screen in a
  five-minute demo.
- **[docs/REPORT_FOR_PPT.pdf](docs/REPORT_FOR_PPT.pdf)** — all results and charts for
  the SIH deck.

### 6. Optional: rebuild everything from the raw data

Only needed if you want to reproduce the panel, the model or the results from scratch.

```bash
python -m greenfleet.data.mrv.download          # 8 EMSA workbooks, ~44 MB, needs internet
python scripts/build_panel.py                   # -> data/interim/*.parquet   (~5 min)
python scripts/train_model.py                   # -> artifacts/model/          (~3 min)
python scripts/evaluate_model.py                # leakage-safe accuracy table  (~10 min)
python scripts/emissions_demo.py                # lifecycle vs stack-only      (seconds)
python scripts/benchmark_optimizer.py --seeds 30  # optimizer vs baselines    (~40 min)
python scripts/make_charts.py                   # -> artifacts/charts/*.png
python scripts/make_report_pdf.py               # -> docs/REPORT_FOR_PPT.pdf (needs Chrome or Edge)
```

### 7. Optional: change the dashboard styling

The dashboard is Tailwind, compiled to a static file so it works offline. After
editing classes in `index.html` / `app.js`, rebuild (needs Node.js):

```bash
cd src/greenfleet/api/static
npx tailwindcss@3 -c tailwind.config.js -i src/tailwind.in.css -o tailwind.css --minify
```

### If something goes wrong

| Symptom | Fix |
|---|---|
| `uv: command not found` | `pip install uv`, or use Option B (plain pip). |
| `No module named greenfleet` | The venv is not activated, or step 2 was run with a different Python. Re-run step 2. |
| Yellow bar in the browser: *No trained model loaded* | `python scripts/train_model.py`, then restart the server. |
| Yellow bar: *Planning service unreachable* | The server is not running; run step 4 and reload. |
| `UnicodeEncodeError` on Windows when running scripts | Set `PYTHONIOENCODING=utf-8` (`set PYTHONIOENCODING=utf-8` in cmd, `$env:PYTHONIOENCODING="utf-8"` in PowerShell). |
| Port 8000 already in use | `python -m greenfleet.api --port 8011` and open that port. |
| `make_report_pdf.py`: *no Chrome or Edge found* | Install Chrome, or add your browser's path to `BROWSERS` in the script. |

---

## What this is

Two engines behind one decision-support platform:

1. **Fuel predictor** — a physics-informed model that predicts vessel **energy**
   demand (MJ) per nautical mile at any speed, then converts to fuel mass per fuel via
   lower heating value. Trained on 100,528 audited EU MRV ship-years; reported with
   calibrated P10/P50/P90 intervals and an explicit cold-start flag.
2. **Fleet planner** — a quantum-inspired multi-objective optimizer (Q-bit amplitude
   encoding + rotation-gate updates for the discrete choices, QPSO for speeds) over
   vessel deployment, route, fuel and speed, minimising fuel, cost and **well-to-wake**
   GHG subject to cargo demand, schedule, emission caps, fuel availability at Indian
   ports and tank range. Outputs a **Pareto front**, not one weighted answer.

Between them sits a **well-to-wake emissions engine** (upstream production, combustion
CO₂, measured methane slip, N₂O, pilot fuel, per production pathway) and an **India
layer** (port fuel availability by year, CEA grid factors, Harit Sagar targets).

Everything is benchmarked against conventional methods (XGBoost, physics-only,
NSGA-II, greedy, random, exact MILP on small instances) over 30 seeded runs with
equal evaluation budgets.

**This is not a quantum computer.** It is quantum-*inspired*: classical algorithms
borrowing quantum concepts, running on ordinary hardware. The discrete core is also
formulated as a QUBO, verified against the exact optimum, so it can move to annealers
or National Quantum Mission hardware when that becomes practical.

## Project status

| Phase | Scope | State |
|---|---|---|
| 0 | Scaffold, canonical units, emission-factor config, reproducibility harness | done |
| 1a | EU MRV ingest: 8 reporting periods, 106,922 ship-years, 25,967 ships, version-pinned | done |
| 1b | **ML POC**: physics core (imposed cube law) + monotone-constrained residual ML, leakage-safe splits, P10/P90 intervals, QPSO tuning + honest ablation | done |
| 1c | Expanded leakage-free features, cold-start stratified reporting, **QIEA** (Q-bit + rotation gate) verified against brute-force optima | done |
| 2 | **Well-to-wake emissions engine**: measured methane slip, N₂O, pilot fuel, tank/range limits, Indian port availability, CEA grid factor, local air pollutants, Harit Sagar KPIs | done |
| 3 | **Fleet optimizer**: formal formulation, Q-bit + QPSO hybrid, Pareto archive, NSGA-II/greedy/random/MILP baselines, 30-seed benchmark, QUBO verified against exact optimum | done |
| 4 | **Platform**: FastAPI over the real engines (predict / compare / ports / optimise as a cancellable job), served model with model card, equal-budget baseline comparison on demand | prototype done |
| 5 | **Dashboard**: Plan · Fuels · Calculator · Ports, Tailwind, offline-capable | prototype done |
| 6 | Scenario library, PDF export, audit trail, Docker packaging, Mormugao/GTTP case study with confirmed port data | planned |

## Key design decision

MRV cannot identify the speed–fuel relationship: the reconstructed speed and the
reconstructed consumption share a measured column, so regressing one on the other
has a coefficient of +1 baked in by algebra and returns a speed exponent near 1
instead of 3. Details and the verification in
[docs/FINDINGS_PHASE1.md](docs/FINDINGS_PHASE1.md) §2.

The model therefore splits the two halves:

- **Physics supplies the shape** — `P = P_ref·[(1-a)·(D/D_ref)^(2/3)·(V/V_ref)³ + a]`,
  with the cube law imposed from naval architecture, not fitted.
- **Data supplies the level** — ML predicts the ship's energy intensity at its class
  reference speed, from leakage-free attributes only.

Three edge cases then hold *structurally* rather than by test: P-02 (power strictly
increasing in speed), P-03 (positive auxiliary load at rest), and the slow-steaming
optimum (energy per mile has a genuine interior minimum, so a high-hotel-load vessel
correctly gains far less from slowing down than a bulk carrier).

## Current results

Energy intensity (MJ/nmi) on **unseen vessels** — 25% of ships fully held out, zero
IMO overlap asserted. 100,528 ship-years, 25,051 ships, 2018–2025.

| model | MAE | RMSE | MAPE | medAPE | R² |
|---|---|---|---|---|---|
| class mean (naive) | 1528.5 | 2780.8 | 32.6% | 23.1% | 0.354 |
| ridge (log) | 1482.6 | 2724.9 | 31.5% | 22.2% | 0.379 |
| XGBoost | 958.7 | 2178.7 | **19.9%** | **13.4%** | **0.603** |
| XGBoost + monotone (served) | 995.4 | 2220.0 | 20.6% | 14.0% | 0.588 |

The platform serves the monotone model: 0.4 points less accurate, but it can never
predict less energy for a worse design certificate. Stricter splits: 25.0% MAPE on
unseen *years*, 25.2% on unseen vessels **and** years.

Stratified by confidence (P-05), because the headline hides the split:

| | n | MAPE | R² | 80% interval coverage |
|---|---|---|---|---|
| warm (has design certificate) | 23,652 | **19.3%** | **0.697** | 79.8% |
| cold (no certificate) | 1,453 | 41.8% | 0.146 | 84.2% |

A vessel with no design certificate gets a wide, conservative interval rather than a
confident wrong answer. 96% of offshore vessels fall in this bucket, by construction.

**Context for the numbers:** within-ship year-to-year variation is σ ≈ 11%, so ~11%
MAPE is a hard floor for *any* method on this data — and the floor for an unseen
vessel is strictly higher. A reported unseen-vessel MAPE below 11% would be evidence
of leakage, not skill.

## Why lifecycle accounting changes the answer

Per 1 TJ of shaft work, scored the way a voyage-optimisation tool scores it and the
way physics scores it:

| fuel | stack-only says | lifecycle truth |
|---|---|---|
| LNG (fossil) | −24% | −4% |
| **Methanol (grey)** | −12% | **+10%** |
| **Ammonia (grey)** | **−100%** | **+40%** |
| **Hydrogen (grey)** | **−100%** | **+9%** |
| Ammonia (green) | −100% | −79% |

**Three fuels change sign.** A stack-only model scores grey ammonia as *perfect*
while it actually increases emissions by 40%. Full analysis in
[docs/FINDINGS_PHASE2.md](docs/FINDINGS_PHASE2.md).

Our methane slip figure is measured, not defaulted: the median across **745 LNG
carrier ship-years** in EU MRV 2024–2025 is **1.72%**, against the FuelEU default of
3.1%. The two disagree about whether LNG helps at all (−3.9% vs +5.5%).

And the electrification answer is the honest one: on the Indian grid a
battery-electric tug does not beat diesel on CO₂ until **~2035** — but it removes
**35.6 t of NOx, 1.25 t of SOx and 187 kg of particulates** from the port city per
tug per year, starting immediately. Local pollutants are tracked separately and
never folded into CO₂e.

## The optimizer, benchmarked honestly

30 seeds, equal evaluation budget, one shared hypervolume reference point, Wilcoxon
signed-rank paired by seed:

| instance | search space | QIEA+QPSO | NSGA-II | greedy | verdict |
|---|---|---|---|---|---|
| 2 vessels | 2⁶ | **exact, 30/30** | — | — | matches proven MILP optimum, 0.0000% gap |
| 12 vessels | 2.3×10¹³ | **4.195e21** | 3.938e21 | 1.710e21 | tie (p = 0.14) |
| **60 vessels** | **7.5×10⁸³** | **3.885e26** | 3.260e26 | 3.602e26 | **wins, p = 0.004** |

Scales to 200 vessels (10²⁷⁹ combinations) in 54 s. The QUBO ground state matches the
CP-SAT optimum exactly. The first benchmark had the quantum-inspired method *losing*;
the fix and how it was calibrated are in
[docs/FINDINGS_PHASE3.md](docs/FINDINGS_PHASE3.md) §2.

## The quantum-inspired components

**QIEA** (`prediction/qiea.py`) — the discrete core the problem statement asks for by
name. Each decision is a **Q-bit**, a pair of probability amplitudes
`(cos θ, sin θ)` representing a superposition of 0 and 1; observation collapses it,
and a **rotation gate** biases the amplitudes toward the best solution found.
Verified against exhaustive search: **exact optimum in 100% of seeds up to 2⁸
states** (Q-09), 96% at 2¹², against 26–70% for budget-matched random search.

**QPSO** (`prediction/qpso.py`) — the continuous core, for the speed variables.
Velocity-free updates sampled from a delta potential well. On 5-D Rastrigin it
scores 5.05 against random search's 28.16 at equal budget.

Honest finding, reported rather than buried: QPSO tuning of XGBoost hyperparameters
gave **no meaningful gain** over random search. The quantum-inspired advantage shows
up on rugged combinatorial landscapes, not smooth low-dimensional ones — which is why
the machinery is aimed at the fleet optimizer and at feature selection. See
[docs/FINDINGS_PHASE1.md](docs/FINDINGS_PHASE1.md) §8.

## What is real and what is representative

Real: the EU MRV data (106,922 verified ship-years, version-pinned with SHA-256), the
trained model and its held-out accuracy, every emission factor (tagged with source
and confidence in `config/fuels.yaml`), CEA grid factors to 2023, Harit Sagar targets,
the Indian port list.

Representative and labelled as such: the example fleets (ship specs shaped like Indian
coastal shipping, not an operator's data), per-year fuel availability at each port
(structure real, years to be confirmed with port authorities), grid projections after
2023, default fuel prices. MRV contains no harbour tugs (below 5,000 GT), so tug
figures are physics-derived and flagged *estimate* wherever they appear.

## Data

Real data comes from the **EU MRV "Publication of information"** dataset published by
EMSA on THETIS-MRV — annual per-ship fuel consumption and CO₂ for ships ≥ 5000 GT
calling at EEA ports, reporting periods 2018–2025. No credentials required.

The processed panel (`data/interim/mrv_panel.parquet`, `mrv_derived.parquet`, 22 MB)
is committed so a fresh clone works without downloading. The raw workbooks are not
committed; `python -m greenfleet.data.mrv.download` fetches them into `data/raw/mrv/`
with a `manifest.json` pinning the reporting period, version, generation date and
SHA-256 of each file. Versions matter: EMSA republishes a period whenever a company
corrects a filing, so results are tied to pinned versions.

## Layout

```
src/greenfleet/
  units.py              canonical units; rejects ambiguous units instead of guessing
  reproducibility.py    seeding + run provenance
  config/               emission factors, ports, policy — all data, no hard-coded numbers
  data/mrv/             THETIS-MRV download, schema validation, cleaning, derived physics
  prediction/
    physics.py          Admiralty baseline as a fitted estimator (B-04 physics-only)
    features.py         feature build; CONTAMINATED_BY_TARGET blocks the leak in code
    splits.py           by_vessel / by_time / by_vessel_and_time, all leakage-asserted
    model.py            imposed-physics core + monotone residual ML + P10/P90 + SpeedCurve
    metrics.py          MAE/RMSE/MAPE/R2 per class, plus the P-02/P-03 assertions
    qiea.py             Q-bit encoding + rotation gates
    qpso.py             quantum-inspired continuous search
    tune.py             grouped-CV tuning and the budget-matched P-13 ablation
  emissions/            well-to-wake engine, ports, tank/range, Harit Sagar KPIs
  optimize/             fleet problem, Q-bit encoding, QMOEA, baselines, QUBO, scenarios
  api/                  FastAPI service + dashboard
    app.py, service.py, schemas.py, jobs.py, state.py
    static/             index.html, app.js, tailwind.css (compiled), tailwind.config.js
scripts/                build_panel, train_model, evaluate_model, run_ablation,
                        select_features, emissions_demo, benchmark_optimizer,
                        make_charts, make_report_pdf
tests/                  unit + integration; markers map to docs/TEST_AND_EDGE_CASES.md IDs
artifacts/              model/ (served model + card), *.json (results), charts/ (figures)
data/interim/           the processed MRV panel (committed)
docs/                   problem statement, findings per phase, test register,
                        USER_GUIDE.pdf, STUDY_GUIDE.pdf, REPORT_FOR_PPT.pdf, DEMO_GUIDE.md
```

## Design rules that are not negotiable

These come from `docs/TEST_AND_EDGE_CASES.md` and are enforced by tests:

- **P-01** Splits are leakage-safe — by vessel *and* by time. Accuracy on unseen
  vessels is reported separately from overall accuracy.
- **P-08** Predict **energy**, not fuel mass. Mass follows from each fuel's LHV.
- **P-02** Fuel must rise monotonically with speed, roughly as the cube.
- **P-05** A vessel with no history falls back to physics with wider intervals; it
  never crashes and never returns nonsense.
- **F-01/F-02/F-03** Grey vs green pathways, LNG methane slip, and ammonia N₂O are all
  modelled. Grey ammonia *can* come out worse than diesel, and the model must show it.
- **F-06** The optimizer never assigns a fuel that is not bunkerable at every port on
  the route in that year; "planned" is not bunkerable.
- **F-09** Every emission factor lives in `config/`, with a `source` and a `confidence`
  tag. No number is hard-coded.
- **B-01/B-02/B-07** Equal budgets, ≥30 seeded runs, and results reported including
  the cases where the quantum-inspired method loses.

## Licence & data attribution

EU MRV data © European Union / EMSA, republished under the THETIS-MRV public
publication of information. Emission factors are attributed in `config/fuels.yaml`.
