# Phase 4 findings — the decision-support platform (prototype)

Scope: deliverable 4 (software platform) and the demonstrable half of deliverable 5.
A FastAPI service over the Phase 1–3 engines and a single-page dashboard. Nothing in
this phase computes an emission factor, a physics term or an objective value; every
number on screen is produced by the modules the Phase 1–3 tests assert on.

## 1. What was built

| Piece | Where | Notes |
|---|---|---|
| Served model | `scripts/train_model.py` → `artifacts/model/` | XGBoost + monotone + QPSO-tuned, fitted on the full usable panel (100,528 ship-years, 25,051 ships), P10/P90 calibrated on 5-fold grouped out-of-fold residuals. A model card records the panel SHA-256, parameters and the held-out accuracy from `evaluation.json`. |
| API | `src/greenfleet/api/` | `/api/health`, `/api/meta`, `/api/predict`, `/api/emissions/compare`, `/api/emissions/electrification`, `/api/optimise` (job), `/api/optimise/{id}`, `/api/optimise/{id}/cancel`. Strict request schemas (unknown fields → 422). |
| Optimisation jobs | `api/jobs.py` | Worker thread per job, progress per generation via a new `on_generation` hook in `optimise_fleet`; returning `False` from the hook cancels and the archive so far is returned marked `time_limited`. |
| Dashboard | `api/static/` | Four pages in a sidebar shell (*Plan · Fuels · Calculator · Ports*), Tailwind (compiled with the CLI to a static `tailwind.css`, so no CDN) in Poppins, full-width content. *Plan* is one screen: setup card (fleet, year, fuel chips, pathways, carbon tax, CO₂ limit, editor dialog, search settings, comparison methods) → three plan cards (lowest cost / balanced / lowest emissions, each with cost, CO₂, % vs diesel, fuel mix, Harit Sagar ✓/✗) → selected plan (trade-off chart with baseline fronts, assignments table, download) → *How the search performed* (evaluations, search space, hypervolume vs NSGA-II, repairs, progress and baseline charts, provenance). Nothing is behind a toggle. Plotly is served from the installed package; the page works with no internet. `?job=<id>` reopens a result. User guide with screenshots: `docs/USER_GUIDE.pdf`. |
| Tests | `tests/unit/test_api.py` | 16 tests: physical consistency through the API, cold-start flag, lifecycle flip, infeasible demand refused with the shortfall, cancellation returns a partial archive, scenario settings survive unless overridden, the speed-curve cache reproduces `predict()` exactly. Suite total 415. |

## 2. The served model is the monotone one, on purpose

`evaluation.json` (by-vessel split, 6,263 unseen ships):

| model | MAPE | R² |
|---|---|---|
| XGBoost, unconstrained | **19.9%** | **0.603** |
| XGBoost + monotone | 20.6% | 0.588 |
| XGBoost + monotone + QPSO-tuned (**served**) | 20.3% | 0.593 |

The unconstrained model is 0.4 points better and can predict *less* energy for a
*worse* design certificate. A planner that can be argued with on physics loses the
room; the platform serves the constrained model and the model card says so. The
report may quote 19.9% as the best accuracy achieved; the platform quotes 20.3% as
the accuracy of what it serves. Both are true; do not mix them on one slide.

## 3. Using the trained model inside the optimizer without paying for it

Phase 3 benchmarked the optimizer on the physics-only fallback because calling
XGBoost once per vessel per evaluation costs milliseconds for an answer that cannot
change: the residual model fixes a vessel's intensity *level*; only the imposed
physics varies with speed.

Fix: `EnergyModel.speed_curves()` returns one `SpeedCurve` per vessel (reference
intensity, reference speed, auxiliary share, cold-start flag). `FleetProblem` caches
one per vessel on first use. The physics formula was extracted into
`physics_power_kw()` / `energy_per_nmi_from_power()` so `predict()` and the curve
call the same code; `test_speed_curve_matches_full_prediction` asserts agreement to
1e-9 at every speed including the infinity at rest.

Effect: 12 vessels × 6,000 evaluations with the trained model in **3.4 s** (was ~45 s
extrapolated). The optimizer now inherits the measured intensity levels of real ship
classes rather than a nominal 2,500 MJ/nmi scaled by capacity.

## 4. Does the trained model change the benchmark verdict?

Five seeds, 30 individuals × 200 generations, NSGA-II at the same 6,000 evaluations,
shared reference point. Hypervolume means:

| energy source | year | QIEA+QPSO | NSGA-II | QIEA wins |
|---|---|---|---|---|
| physics fallback | 2030 | 4.7e17 | 2.2e16 | 4/5 |
| physics fallback | 2047 | 6.3e20 | 3.5e20 | 2/5 |
| trained model | 2030 | 1.5e20 | 2.6e18 | 4/5 |
| trained model | 2047 | 4.9e20 | 6.3e20 | 2/5 |

Same picture as Phase 3: a clear win on the constrained 2030 instance (few feasible
fuels, rugged landscape), a tie on the 2047 instance where the front is smooth and
wide. The verdict does not depend on which energy source is used. At the dashboard's
*previous* default budget (20 × 60 = 1,200) NSGA-II was ahead by an order of
magnitude; the default is now 30 × 200 so the screen shows the benchmarked regime.

## 5. Bugs found by building the UI

- **Request defaults overrode scenario settings.** `year: int = 2030` and
  `carbon_price: float = 0` in the request schema silently turned the 2047 e-methanol
  scenario into a 2030 fossil run. Fixed by making them `None` = "keep the scenario's
  value"; asserted by `test_optimise_keeps_scenario_settings_unless_overridden`.
- **Object dtype from a hand-built feature row.** One vessel with no certificate gives
  a `None` cell, which pandas types as `object` and XGBoost refuses. The design matrix
  now coerces every column to float64 with NaN for missing — this also affected the
  tug scenario through the optimizer, which had never run with the trained model.
- **numpy.bool in a JSON response.** `HaritSagarStatus.on_track` is a numpy bool;
  pydantic will not serialise it. Cast at the boundary.
- **Methanol has no "grey" pathway key** — its fossil pathway is `fossil`. The default
  comparison list was wrong for one fuel; the registry's error message caught it.

## 6. What the Harit Sagar check uses as its baseline

The target is a reduction from a baseline. For a plan the platform uses *the same
deployment on MGO* — same ships, routes and speeds, fuel forced to diesel — evaluated
through the same `FleetProblem`. This is deterministic and cannot be gamed by the
optimiser; it is stated on every plan card. It is **not** the port's historical 2023
intensity, which a real deployment would substitute.

## 7. Not done, and to be said as such

- PDF/CSV report export (a JSON download of the selected plan exists).
- Scenario library with save/compare; audit trail; users and roles.
- Docker packaging for on-premises deployment.
- Multi-user job queue (the store is a dict behind a lock; single process).
- Streaming progress (the page polls every 250 ms; adequate at these run times).
- A React build. The dashboard is plain JS by choice: no build step, no CDN, no
  failure mode on an offline port server.
