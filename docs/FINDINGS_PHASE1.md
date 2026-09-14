# Phase 1 findings — EU MRV data foundation

**SIH26138 · Green Fleet Optimizer**
Status as of the first data-ingest pass. Every number below is reproducible from
`data/raw/mrv/manifest.json` plus the code in `src/greenfleet/`.

---

## 1. What we have

| | |
|---|---|
| Source | EU MRV "Publication of information", EMSA THETIS-MRV public API (no credentials) |
| Reporting periods | 2018–2025 (8 years) |
| Rows | **106,922 ship-years** |
| Unique ships | **25,967** |
| Usable after quality flags | 100,528 (94.0%) |
| Ships with ≥5 reporting periods | 9,182 |

Each file is pinned by reporting period **and version** with a SHA-256, because
EMSA republishes a period whenever a company corrects a filing — "the 2024 file"
is not a stable object (2024 is currently at v242).

Derived per ship-year, by inverting MRV's published intensity ratios:
distance, mean speed, cargo actually carried, implied CO₂ factor, inferred fuel
blend and LHV, energy (MJ), mean power (kW), at-berth CO₂ share, laden share.

---

## 2. The critical finding: MRV cannot identify the speed–fuel relationship

This is the most important result so far, and it changes the model architecture.

We verified to machine precision (max relative error 5×10⁻¹⁶) that:

```
mean_power_kw  ≡  LHV × fuel_per_distance × mean_speed / 3.6
```

The reason is that both quantities are reconstructed from the same published
columns:

```
mean_power = total_fuel × 1000 × LHV / (time_at_sea × 3.6)
mean_speed = total_fuel × 1000 / (fuel_per_distance × time_at_sea)
```

Both contain the factor `total_fuel / time_at_sea`. So regressing log(power) on
log(speed) has a coefficient of **+1 built in by algebra**, before any physics.

**Observed consequence.** Fitting `log P = a + b·log D + n·log V` per ship type on
2018 data returns speed exponents of **0.66 – 1.55**, clustered near 1, against a
theoretical value of 3. Fixing the exponents at the physical 2/3 and 3 instead
gives MAPE 65.3% and R² = −8.19; letting them float gives MAPE 19.6% and R² = 0.86
but the fitted exponents are physically meaningless.

**Why this matters far beyond a bad fit.** A team that fits this naively concludes
the cube law is wrong and ships a model whose speed sensitivity is off by roughly
a factor of three. The entire slow-steaming trade-off — the single most important
lever in this problem statement, and the one worth 27% fuel per trip in the
project's own worked example — would be badly underestimated. The optimizer would
systematically recommend sailing too fast.

**The fix.** Physics supplies the functional form in speed; data supplies the
ship-specific calibration:

- **Impose** `P ∝ Δ^(2/3) · V³` from naval architecture. This is well established
  and is not something annual aggregate data can or should re-estimate.
- **Learn** the ship-specific efficiency coefficient (the Admiralty coefficient)
  from MRV, as a function of ship type, size, design efficiency (EIV/EEDI), ice
  class and year. This is where data genuinely beats physics alone.
- **Monotonicity in speed (P-02) then holds by construction**, not by test, and
  the ML residual is additionally constrained with XGBoost monotone constraints so
  it cannot break the guarantee.

This is also a strong honest-benchmarking slide (B-07): we can show the naive fit,
show why it is wrong, and show the corrected design.

---

## 3. Energy intensity is a stable ship characteristic

Variance decomposition of log(energy per nautical mile), for the 9,182 ships with
≥5 reporting periods (60,482 rows):

| | |
|---|---|
| Within-ship std (year to year, same ship) | **0.112** |
| Between-ship std | **0.464** |
| **ICC** (share of variance that is ship identity) | **0.945** |

Two consequences:

1. Predicting energy intensity for an **unseen vessel** from its attributes is a
   well-posed problem — 94.5% of the variance is a persistent ship property, so
   there is a stable thing to predict.
2. **No method can beat roughly 11% MAPE on this data.** The σ = 0.112 within-ship
   spread is irreducible operational variation (routing, weather, utilisation, hull
   fouling) that annual attributes cannot explain.

**Being precise about what the 11% floor is, because it is easy to misstate.** It is
the error remaining when you already know the ship perfectly — i.e. the best case
for predicting a vessel *you have history for*. For an **unseen vessel** the floor is
strictly higher, because the model must additionally explain the between-ship
variation (σ = 0.464) from attributes alone, and no attribute set explains all of it.

So 11% is a **lower bound on any method**, not a target for unseen-vessel prediction.
The right reading of an unseen-vessel MAPE of, say, 20% is not "9 points from
optimal" — it is "comfortably above the absolute floor, with the gap split between
irreducible noise and unexplained between-ship variation". A reported unseen-vessel
MAPE *below* 11% would be strong evidence of leakage.

---

## 4. Data-quality findings (all handled, all counted)

| Finding | Scale | Handling |
|---|---|---|
| EMSA publishes the literal string **`"Division by zero!"`** in intensity columns | 5.2% of rows (distance), 9.0% (transport work) | Classified as a computation error, not a missing value. It means the ship reported **zero distance** — in scope but did not sail. Real signal for P-03, not noise. |
| **Implied CO₂ factor above physical maximum** — values up to 9.68 tCO₂/t fuel, where no marine hydrocarbon exceeds ~3.25 and pure carbon is 3.67 | 166 ship-years | Flagged as filing errors (D-03) and excluded from training, with the count reported. |
| Mean speed outside 1–30 kn | 198 | Flagged (this is the D-03 "60-knot bulk carrier" case, and it is real). |
| Reconstructed cargo above 600,000 t, where the largest ship afloat is ~400,000 DWT | 246 | Flagged. |
| **Schema drift across reporting periods** — columns renamed twice; `"Annual Total time spent at sea"` → `"Annual time spent at sea"` → `"Time spent at sea"` | 2020 and 2024 | Every column carries aliases; lookup is by normalised header text, so reordering and added columns are tolerated. Ingest fails loudly on a genuinely missing required column rather than mis-mapping. |
| **2024+ files have 113 columns, not 62** | 2024, 2025 | The EU ETS / FuelEU extension. See §5. |
| **2024+ files split into two sheets**, `Full ERs` and `Partial ERs` | 1,030 and 1,523 partial rows | Partial reports are ships that changed company mid-year: the same ship recurs and covers part of a year, so concatenating them double-counts and corrupts every annual mean. We read `Full ERs` by default and label coverage on every row. |

Nothing is silently imputed and nothing is silently dropped. Rows get `flag_*`
columns and an `is_usable` summary; the caller decides, and the count is reportable.

---

## 5. Unexpected win: 2024+ reports measured CH₄ and N₂O

The 2024 extension publishes, per ship: **total CH₄**, **total N₂O**, total CO₂eq,
and a directly published fuel-rate column (`t/hour`).

This upgrades two of our least certain inputs from assumption to measurement:

- **F-02 methane slip.** Our config currently carries the FuelEU default of 3.1%
  of fuel mass for LNG Otto medium-speed engines, tagged `confidence: medium`. We
  can now compute the actual CH₄/fuel ratio for LNG carriers and dual-fuel ships
  from reported data and validate or replace that number.
- **F-03 ammonia N₂O.** Currently our single most uncertain factor
  (`confidence: provisional`). There is no ammonia fleet yet, but reported N₂O
  across conventional and LNG ships calibrates the combustion N₂O baseline.

Being able to say "our methane slip figure is derived from reported EU MRV data,
not a default" is a direct answer to the likely judge question *"Isn't LNG a clean
fuel?"* (B2).

---

## 6. Scope caveat, stated up front

MRV covers ships **≥5000 GT** calling at EEA ports. **Harbour tugs are not in it**
(they are ~200–500 GT), and the data is annual aggregate — no per-voyage speed or
weather.

So the division of labour is:

- **MRV** is the real-data anchor: it calibrates and validates the energy model on
  ~26,000 real ships, and supports a genuine unseen-vessel evaluation.
- **The GTTP / Mormugao tug case study** reaches tugs through the documented
  cold-start path (P-05): physics plus vessel-class baselines calibrated to
  published tug specifications, with wider uncertainty bands. Tug figures are
  physics-derived, not MRV-derived, and are labelled as such everywhere they appear.

Per-voyage granularity needs noon reports or AIS — which is exactly what a port
data-sharing arrangement would supply, and worth naming as the pilot's first ask.

---

## 7. Phase 1b results

Target: energy per nautical mile (MJ/nmi) — P-08, fuel-agnostic. Features are
leakage-free only (vessel class, design efficiency, ice class, reporting period,
time at sea). Intervals calibrated on **out-of-fold** residuals from grouped folds,
not in-sample. Reproduce with `scripts/evaluate_model.py --calibrate-intervals`.

Panel: 100,528 ship-years, 25,051 ships, 2018–2025.

### Split A — unseen vessels (25% of ships held out)

| model | MAE | RMSE | MAPE | medAPE | R² | 80% coverage |
|---|---|---|---|---|---|---|
| class mean (naive) | 1528.5 | 2780.8 | 32.6% | 23.1% | 0.354 | 80.7% |
| ridge (log) | 1485.5 | 2727.3 | 31.6% | 22.3% | 0.378 | 80.9% |
| XGBoost | 1039.3 | 2260.7 | **21.7%** | 15.1% | **0.573** | 81.0% |
| XGBoost + monotone | 1085.1 | 2301.2 | 22.5% | 15.9% | 0.557 | 81.1% |
| XGBoost + mono + QPSO | 1077.6 | 2288.8 | 22.3% | 15.8% | 0.562 | 81.2% |

### Split B — unseen future (train 2018–2023 → test 2024–2025)

| model | MAPE | medAPE | R² | coverage |
|---|---|---|---|---|
| class mean (naive) | 41.4% | 25.6% | 0.249 | 75.8% |
| ridge (log) | 37.9% | 23.6% | 0.255 | 77.0% |
| XGBoost | **27.2%** | 15.4% | **0.431** | 79.8% |
| XGBoost + monotone | 27.5% | 16.4% | 0.413 | 80.1% |
| XGBoost + mono + QPSO | 27.6% | 16.3% | 0.405 | 79.9% |

### Split C — unseen vessels *and* unseen years (strictest)

| model | MAPE | medAPE | R² | coverage |
|---|---|---|---|---|
| class mean (naive) | 41.2% | 25.8% | 0.223 | 75.1% |
| XGBoost | **26.9%** | 15.7% | **0.406** | 78.9% |
| XGBoost + monotone | 27.1% | 16.7% | 0.394 | 79.8% |
| XGBoost + mono + QPSO | 27.4% | 16.6% | 0.366 | 79.6% |

Split ordering is A < C ≈ B, which is the correct direction: unseen vessels in
known years is easier than unseen years. A temporal split that looked *easier*
than a vessel split would indicate leakage.

**Interval calibration (P-10) works.** Empirical coverage is 78.9–81.2% against a
nominal 80% across all three splits. That is what makes the conservative bound
safe for the optimizer to consume.

### Per vessel class, unseen vessels (P-12)

| class | n | MAE | MAPE | R² |
|---|---|---|---|---|
| vehicle_carrier | 895 | 507.3 | **14.0%** | 0.201 |
| bulk_carrier | 7025 | 674.2 | 17.2% | 0.186 |
| chemical_tanker | 2782 | 661.9 | 18.5% | 0.199 |
| container_roro | 208 | 922.4 | 20.5% | 0.207 |
| gas_carrier | 645 | 917.4 | 20.5% | 0.171 |
| oil_tanker | 3619 | 1274.3 | 22.1% | 0.244 |
| container_ship | 3639 | 1651.4 | 24.4% | 0.618 |
| general_cargo | 2897 | 594.2 | 25.5% | 0.169 |
| lng_carrier | 632 | 2762.7 | 25.6% | 0.165 |
| cruise_ship | 64 | 2089.4 | 25.7% | 0.712 |
| refrigerated_cargo | 296 | 804.1 | 26.9% | 0.191 |
| roro | 565 | 1006.3 | 28.8% | 0.175 |
| passenger_ship | 356 | 3627.8 | 34.5% | 0.321 |
| ropax | 878 | 1874.2 | 35.4% | 0.137 |
| other | 460 | 1569.4 | 38.8% | 0.039 |
| **offshore** | 138 | 5322.9 | **109.8%** | −0.030 |
| combination_carrier | 6 | 447.2 | 16.4% | −83.3 *(flagged unreliable)* |

**The headline hides two failures, and both are reported rather than hidden:**

- **Offshore vessels: 109.8% MAPE, R² −0.03.** Worse than predicting the class
  mean. EMSA only split this class out in 2024, so there are few training rows,
  and offshore duty cycles (station-keeping, dynamic positioning) are not
  resistance-driven at all — the physics core does not describe them. This class
  needs its own treatment, not a better fit.
- **Ro-pax and passenger: 34–35% MAPE.** High hotel load, highly variable
  utilisation, and short routes where manoeuvring dominates.

## 8. The quantum-inspired ablation (P-13) — an honest negative result

Two experiments, both budget-matched (B-01), same folds, same seeds.

**Experiment 1 — QPSO as a hyperparameter tuner**, grouped 3-fold CV, 56 evaluations:

| | CV MAPE |
|---|---|
| untuned default | 22.99 |
| random search (56 evals) | 22.75 |
| QPSO (56 evals) | **22.70** |

QPSO wins by **0.2%** over random search. That is a tie in any practical sense, and
on the held-out splits the tuned configuration was **slightly worse** than the
untuned monotone model (27.6% vs 27.5% on Split B; 27.4% vs 27.1% on Split C) —
i.e. the tuning gain did not generalise.

**Conclusion: hyperparameter tuning is the wrong place for the quantum-inspired
machinery here.** The accuracy is bounded by the ~11% irreducible floor and by a
five-feature leakage-free input set, not by XGBoost settings. The landscape is flat,
so no search method can win much on it.

**Experiment 2 — QPSO on a rugged multimodal landscape**, 5-D Rastrigin, 15 seeds,
820 evaluations each:

| | mean | std | best |
|---|---|---|---|
| random search | 28.16 | 4.85 | 20.05 |
| QPSO | **5.05** | 3.04 | **0.52** |

A **5.6x improvement** on exactly the kind of landscape the fleet optimizer presents:
high-dimensional, multimodal, heavily constrained.

**So the defensible claim is narrow and specific:** the quantum-inspired advantage
appears where the search space is large, discrete and rugged — the fleet-mix, fuel
and speed assignment problem of Phase 3 — and *not* in smooth, low-dimensional
hyperparameter tuning. That is a better answer to the judge question *"why not just
use a normal GA?"* than any claim that QPSO helps everywhere, because it comes with
the case where it does not.

## 9. Trade-off worth stating: the monotone constraint costs accuracy

Unconstrained XGBoost scores 21.7% MAPE on unseen vessels; adding the
design-efficiency monotone constraint gives 22.5%. The constraint costs ~0.8 MAPE
points.

We keep it. A model that fits marginally better but can violate a known physical
relationship is worse for an optimizer that will extrapolate along that
relationship — it would let the search exploit an artefact. Both numbers are
reported so the reader can disagree.

## 10. Validation against the project's own worked example

The plain-language doc states that slowing from 14 to 12 knots cuts fuel ~37% per
day and ~27% per trip (the pure cube law). Our model, which additionally separates
auxiliary load:

| class | aux share | per day | per mile |
|---|---|---|---|
| bulk_carrier | 0.06 | 34.8% | 23.9% |
| oil_tanker | 0.10 | 33.0% | 21.8% |
| container_ship | 0.08 | 30.4% | 18.8% |
| ropax | 0.22 | 20.3% | 7.0% |
| **cruise_ship** | 0.35 | 14.4% | **0.1%** |
| *pure cube law* | 0 | 37.0% | 26.5% |

Bulk carriers land just under the pure cube law, as they should once a small
speed-independent load is included. And a cruise ship saves essentially **nothing
per mile** from the same slowdown, because hotel load runs regardless and simply
runs longer. Voyage-level tools that assume a universal cube law cannot express
that; it falls straight out of separating the two terms.

## 11. Phase 1c: fixing what Phase 1b exposed

Phase 1b reported three weaknesses honestly. This section is what was done about
each. All numbers are leakage-safe and reproducible via `scripts/evaluate_model.py
--calibrate-intervals`.

### 11.1 More leakage-free signal

Three additions, each independent of `fuel_per_distance`:

| feature | rationale | coverage |
|---|---|---|
| `efficiency_metric_{eiv,eedi,eexi}` | *Which* certificate a ship carries is informative on its own: EIV is an estimated index used where no design value exists, EEDI a design value for newer ships, EEXI the 2023 retrofit-era index. Proxies design era and data quality. | 94% |
| `ice_class_ordinal` | Baltic classes IC<IB<IA<IA Super, then IACS polar PC7..PC1. Ordinal because the ordering is real - a PC1 hull is stronger and draggier. | 17% |
| `ice_time_share` | Fraction of sea time spent in ice. | 100% |

`cargo_density_t_per_m3` was investigated and **rejected**: 5 non-null rows out of
100,528.

Effect on unseen vessels, everything else held fixed:

| feature set | MAPE | medAPE | R2 |
|---|---|---|---|
| 5 original | 22.49% | 15.89% | 0.558 |
| + 3 new groups | **20.55%** | **13.97%** | **0.588** |

### 11.2 Offshore was not a modelling bug

Phase 1b reported offshore at 109.8% MAPE, R2 -0.03. Diagnosis:

- p95/p5 spread of **21.7x**, against 2.6x for bulk carriers. PSVs, AHTS, survey and
  construction vessels are one MRV label covering wholly different machines.
- **Only 4% carry a design-efficiency certificate**, against 97% for bulk carriers -
  our single most informative feature is absent for almost all of them.
- They appear only in 2024-2025: offshore entered MRV scope with the EU ETS
  extension, so there is almost no history to learn from.

The model already flags 96% of offshore rows as cold-start. So the correct fix was
not to force a better fit but to **report warm and cold rows separately**, which is
what P-05 is for:

| | n | MAPE | medAPE | R2 | 80% coverage |
|---|---|---|---|---|---|
| warm (has certificate) | 23,652 | **19.3%** | 13.6% | **0.697** | 79.8% |
| cold (no certificate) | 1,453 | 41.8% | 25.8% | 0.146 | 84.2% |

A model that says "I do not know, here is a wide interval" on a vessel class it has
no design data for is behaving correctly. Reporting the two together hid that.

### 11.3 A real interval defect, found and fixed

Cold-start intervals were originally widened by a hand-set 1.8x multiplier.
Stratified reporting exposed the problem: under the **temporal** splits, cold-row
coverage fell to **70.9%** and **69.3%** against a nominal 80%.

An interval that claims 80% and delivers 69% is worse than no interval, because the
optimizer treats the conservative bound as safe when it is not.

Replacing the multiplier with a band calibrated on out-of-fold cold rows fixed the
vessel split (87.8% -> 78.0%, well centred on 80%) but made the temporal splits
*worse* (70.9% -> 59.8%, 69.3% -> 55.5%). The reason is substantive, not numerical:
cold rows in 2024-2025 are predominantly the offshore fleet that only entered scope
in 2024, a genuinely different population from any cold rows present in training.
**No band fitted to past data can anticipate a population that was not in it.**

Final behaviour: cold-start rows take the **wider** of the measured band and the
scaled class band, which recovers the better of the two everywhere:

| split | fixed 1.8x | measured only | wider of both |
|---|---|---|---|
| A: unseen vessels | 87.8% | 78.0% | **84.2%** |
| B: unseen future | 70.9% | 59.8% | **71.0%** |
| C: unseen vessels + years | 69.3% | 55.5% | **68.3%** |

Intervals that a downstream optimizer treats as safe bounds must fail conservative.
B and C remain below nominal at ~70%, and that is stated as a limitation rather
than smoothed over: predicting the spread of a vessel population that did not exist
in the training data is not something calibration can solve.

### 11.4 Deliverable 1 now has a real quantum-inspired component

Phase 1b's honest finding was that QPSO tuning of XGBoost hyperparameters adds
nothing (confirmed again on the new feature set: 20.679 vs 20.728 for budget-matched
random search, 20.875 untuned - the same +0.2%). That left deliverable 1's
"quantum-inspired prediction" claim thin.

`prediction/qiea.py` now implements the encoding the problem statement names
directly: **Q-bit amplitudes plus rotation-gate updates**. Each decision is a pair
`(cos θ, sin θ)`; observation collapses it; the Han & Kim rotation gate biases
amplitudes toward the best solution, but *only* where an individual disagrees with a
solution that beat it - unconditional rotation saturates the amplitudes within a few
generations, which is the Q-03 failure mode.

**Verified against brute force** (30 seeds, budget-matched random search):

| bits | states | QIEA exact | random exact | evals |
|---|---|---|---|---|
| 4 | 16 | **100%** | 63% | 16 |
| 6 | 64 | **100%** | 70% | 64 |
| 8 | 256 | **100%** | 53% | 206 |
| 10 | 1,024 | 86% | 46% | 405 |
| 12 | 4,096 | **96%** | 26% | 702 |

Q-09 asks that a tiny instance reach the exact optimum. At the "2 vessels, 1 route"
scale this is 100%, verified, not asserted.

**Rotation angle tuned, not inherited** (the Q-04 question judges ask). Hit rate on a
14-bit knapsack, 30 seeds:

| angle | G=30 | G=60 | G=120 | distinct evals @ G=120 |
|---|---|---|---|---|
| 0.005 π | 13% | 56% | **73%** | 981 |
| 0.010 π | 13% | 40% | 46% | 757 |
| 0.025 π | 26% | 33% | 36% | 402 |
| 0.050 π | 20% | 20% | 26% | 243 |
| 0.100 π | 10% | 10% | 10% | 187 |

A large angle collapses the amplitudes early - visible in both the hit rate and the
shrunken evaluation count. Default set to 0.005 π.

**A second negative result, kept.** Adding 1-opt local refinement ("surely hill
climbing helps") was measured at matched generations and **hurts**: n=14 70%→60%,
n=16 66%→50%. Refining to a local optimum every generation drags the rotation gate
toward it and costs the exploration that finds the global one. Off by default, with
the evidence in the docstring.

### 11.5 Where quantum-inspired search pays off, quantitatively

Four budget-matched comparisons now exist, and they form a consistent pattern:

| problem | space | quantum-inspired | random | verdict |
|---|---|---|---|---|
| XGBoost hyperparameters | continuous, 7-D, smooth | 20.68 | 20.73 | **tie** |
| Feature selection | 2^10 = 1,024 | 22.5518 | 22.5518 | **tie** |
| Knapsack | 2^12 = 4,096 | 96% exact | 26% exact | **QIEA wins** |
| Rastrigin | continuous, 5-D, multimodal | 5.05 | 28.16 | **QPSO wins** |

The two ties are not failures, they are the boundary of the claim: with 265
evaluations over 1,024 subsets, random search covers a quarter of the space, so no
strategy can distinguish itself. **The advantage appears exactly when the space is
large and rugged** - which is the regime of the fleet-mix, fuel and speed assignment
problem in Phase 3, where 20 vessels x 4 fuels x 5 speed bands is already ~10^20
combinations.

That is a far better answer to *"why not just use a normal GA?"* than a claim that
the method always wins, because it comes with the cases where it does not, and with
a stated reason.

On feature selection the Q-bit search did find the **verified global optimum**
(+0.000% gap against exhaustive evaluation of all 1,024 subsets, using 26% of the
space). The selected 7-feature subset did not, however, beat the full set on held-out
data (20.7% vs 20.6% on unseen vessels), so the full feature set is retained.

### 11.6 Phase 1c results

Unseen vessels, all improvements applied:

| model | MAE | RMSE | MAPE | medAPE | R2 |
|---|---|---|---|---|---|
| class mean (naive) | 1528.5 | 2780.8 | 32.6% | 23.1% | 0.354 |
| ridge (log) | 1482.6 | 2724.9 | 31.5% | 22.2% | 0.379 |
| XGBoost | 958.7 | 2178.7 | **19.9%** | **13.4%** | **0.603** |
| XGBoost + monotone | 995.4 | 2220.0 | 20.6% | 14.0% | 0.588 |
| XGBoost + mono + QPSO | 983.3 | 2207.4 | 20.3% | 13.7% | 0.593 |

Against Phase 1b, on identical splits:

| split | Phase 1b | Phase 1c | change |
|---|---|---|---|
| A: unseen vessels | 21.7% | **19.9%** | −1.8 pts |
| B: unseen future | 27.2% | **25.0%** | −2.2 pts |
| C: unseen vessels + years | 26.9% | **25.2%** | −1.7 pts |

Per-class, unseen vessels (P-12), largest movers:

| class | Phase 1b | Phase 1c |
|---|---|---|
| lng_carrier | 25.6% | **19.9%** |
| oil_tanker | 22.1% | **18.3%** |
| chemical_tanker | 18.5% | **16.6%** |
| bulk_carrier | 17.2% | **15.2%** |
| ropax | 35.4% | 34.3% |
| offshore | 109.8% | 117.7% *(96% cold-flagged)* |

### 11.7 Known limitations, stated plainly

- **Ro-pax and passenger remain weak (31-34%).** High hotel load, highly variable
  utilisation and short routes where manoeuvring dominates. A duty-cycle model, not
  a resistance model, is the right treatment.
- **Offshore is not predictable from MRV attributes** and is correctly flagged
  cold-start rather than fitted.
- **Cold-start intervals under-cover on temporal splits**, mitigated by failing
  conservative, not eliminated.
- **`efficiency_metric_eexi` is partly a time marker** (0% of ships in 2018-2022,
  53.8% in 2023, 70.4% by 2025). This is genuine covariate shift, not leakage - a
  2019 row never carries EEXI - but it is part of why the temporal split is harder.

## 12. Next

1. **Phase 2** - well-to-wake emissions engine: methane slip validated against the
   measured 2024+ CH4 column, ammonia N2O, pilot fuel, tank volume vs cargo,
   per-port fuel availability, shore power on the CEA grid factor.
2. **Phase 3** - the fleet optimizer. The QIEA engine is built, tuned and verified
   against exhaustive optima; it needs the fleet encoding, the multi-objective layer
   (Pareto front, hypervolume), NSGA-II/GA/PSO/MILP baselines and the QUBO
   formulation.
3. Duty-cycle treatment for offshore and ro-pax.
4. Tug cold-start calibration from published GTTP tug specifications.
