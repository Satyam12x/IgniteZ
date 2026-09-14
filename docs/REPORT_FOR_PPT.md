# Green Fleet Optimizer — Results Report for the SIH 2026 Idea Submission

**Problem Statement SIH26138 · Egreen Quanta · Quantum-Inspired Fuel Consumption Prediction and Green Fleet Optimization**

This report is organised **slide by slide against the SIH six-slide template** (Title,
Proposed Solution, Technical Approach, Feasibility & Viability, Impact & Benefits,
Research & References). Each section gives you: the message, the numbers, which chart
to use, and where the number comes from. Every figure below is read from
`artifacts/*.json` or recomputed with a fixed seed — none is typed from memory, so
the deck cannot drift from the code (PPT-02).

**Charts:** `artifacts/charts/01..14_*.png`, 300 dpi, sized for a 16:9 slide.
Regenerate with `uv run python scripts/make_charts.py`.

---

## 0. The five numbers to lead with

If a judge reads nothing else, these are the ones:

| # | claim | number | source |
|---|---|---|---|
| 1 | **Real, audited data — not synthetic** | **106,922 ship-years · 25,967 real ships · 8 years** (EU MRV, third-party verified regulatory filings) | `data/raw/mrv/manifest.json` |
| 2 | **A voyage tool gets the sign wrong on three fuels** | Grey ammonia: stack-only says **−100%**, truth is **+40%** | `emissions_demo.json` |
| 3 | **Prediction accuracy on ships never seen in training** | **19.9% MAPE, R² 0.603** — leakage-safe, 6,263 held-out ships | `evaluation.json` |
| 4 | **The quantum-inspired optimizer is verified, then wins where it should** | **30/30 exact vs proven MILP optimum**; beats NSGA-II on the 60-vessel instance, **p = 0.004** | `optimizer_benchmark.json` |
| 5 | **The honest electrification answer for India** | Electric tugs beat diesel on CO₂ only from **2035** — but remove **35.6 t NOx per tug per year** from the port city immediately | `emissions_demo.json` |

Number 5 is the one that wins a government room: it is specific, uncomfortable, and
obviously not sales copy — which makes everything else you say more credible.

---

## SLIDE 1 — TITLE

Fill exactly as the portal specifies (PPT-01, PPT-07). Verified values:

| field | value |
|---|---|
| Problem Statement ID | **SIH26138** |
| Problem Statement Title | **Quantum-Inspired Fuel Consumption Prediction and Green Fleet Optimization** (verbatim from the PS PDF — do not paraphrase) |
| Organisation | Egreen Quanta |
| Theme | Transportation & Logistics *(confirm against the portal listing)* |
| PS Category | Software |
| Team name / ID | *(yours)* |

**Tagline for the title slide** (one line, under the team name):

> *India's first sovereign, quantum-ready planner that decides which vessels, which fuels, and which speeds — not just which route.*

---

## SLIDE 2 — PROPOSED SOLUTION

### The problem in our own words (not the PS text)

Ports and coastal operators must decide **which ships to buy, which fuels to commit
to, how fast to sail, and when** — under cargo demand, schedules, emission caps, and
fuel that is only bunkerable at some Indian ports in some years. With 20 ships, 4
fuels and 5 speed bands there are ~10²⁰ combinations. Existing tools (ZeroNorth,
Wärtsilä FOS, DeepSea) optimise **one voyage of one ship** and count only funnel
CO₂. Nobody decides the fleet.

### The solution in one sentence

A decision-support system with two engines: a **physics-informed fuel predictor**
trained on 26,000 real ships, and a **quantum-inspired fleet optimizer** that outputs
a **Pareto front** of feasible plans under **well-to-wake** lifecycle emissions and
**Indian port** constraints.

### What is unique — four things nobody else has

| uniqueness | proof point | chart |
|---|---|---|
| **Lifecycle, not funnel.** Well-to-wake CO₂e including methane slip, N₂O, pilot fuel, grey/blue/green pathways | **3 fuels flip sign.** Grey ammonia: −100% stack-only → **+40%** truth. Grey methanol −12% → **+10%**. Grey hydrogen −100% → **+9%** | **01** |
| **Measured, not defaulted.** Methane slip from **745 measured LNG carrier ship-years** | Regulatory default 3.1% says LNG is **+5.5% worse** than diesel; measured 1.72% says **−3.9% better**. They disagree on the sign. | **02** |
| **India-specific.** 8 major ports with year-indexed bunkering, shore power, CEA grid factor, Harit Sagar KPIs, GTTP tugs, Mormugao | Electric tug on the Indian grid: **+20% CO₂ in 2030, −4% in 2035**, crossover **2035** — while removing **35.6 t NOx/yr** from day one | **03, 04** |
| **Quantum-inspired and quantum-ready.** Q-bit encoding + rotation gates for discrete decisions, QPSO for continuous speed, QUBO formulation for future hardware | **30/30 seeds** hit the proven MILP optimum; **beats NSGA-II at 60 vessels, p = 0.004**; QUBO ground state **exactly** matches CP-SAT | **09, 11** |

### Competitor comparison table (PPT-08)

| capability | ZeroNorth / Wärtsilä / DeepSea | **This tool** |
|---|---|---|
| Decision level | one voyage, one vessel | **fleet mix, fuel, speed, over years** |
| Emissions basis | tank-to-wake CO₂ | **well-to-wake CO₂e: CH₄ + N₂O + pilot + upstream** |
| Methane slip | default, if at all | **measured from 745 ship-years** |
| Grey vs green pathways | not modelled | **modelled; grey ammonia comes out worse than diesel** |
| Local air pollutants | not reported | **NOx / SOx / PM tracked separately** |
| Indian port fuel availability | not modelled | **year-indexed, 8 ports** |
| Indian grid carbon | not modelled | **CEA factor + T&D losses + decarbonisation path** |
| Harit Sagar / GTTP alignment | none | **KPI glide path, on/off-track** |
| Quantum-readiness | none | **QUBO formulation, verified** |
| Data residency | foreign SaaS | **on-premises, offline-capable** |

Honest framing to use verbally: *"Those are strong products doing a different job.
They optimise a voyage. Nothing in them decides which vessels to buy and which fuels
to commit to — which is the decision a port under GTTP and Harit Sagar actually faces."*

---

## SLIDE 3 — TECHNICAL APPROACH

### Architecture — use chart **14**

```
EU MRV data ──► Fuel Predictor ──► Well-to-Wake ──► India Layer
 (real, audited)  (physics + ML)   (lifecycle GHG)  (ports, grid, KPIs)
                        │                │               │
                        └────────────────┴───────────────┘
                                         ▼
                        Quantum-Inspired Optimizer (Q-bit + QPSO)
                                         ▼
                        Decision Support: Pareto front, scenarios, reports
```

### The pipeline, step by step

| step | what happens | key number |
|---|---|---|
| **1 · Data** | EU MRV 2018–2025 from EMSA's public API, version-pinned with SHA-256. 12 data-quality gates (units, outliers, schema drift, duplicates). | 106,922 ship-years → **100,528 usable (94.0%)** |
| **2 · Predictor** | Physics core `P ∝ Δ^(2/3)·V³` imposed from naval architecture; XGBoost learns the ship-specific level from leakage-free design attributes only. Predicts **energy** (MJ), converts to fuel mass per fuel via LHV. | **19.9% MAPE**, R² 0.603, unseen vessels |
| **3 · Emissions** | Well-to-tank + CO₂ + CH₄ (slip) + N₂O + pilot fuel, per pathway, on **shaft energy** so fuels and batteries compare fairly | grey ammonia **+40%** vs diesel |
| **4 · Constraints** | Fuel bunkerable at every port on the route (year-indexed); tank range; cargo displaced by tank volume; shore power needs port AND vessel fitted | hydrogen retrofit range = **7%** of diesel |
| **5 · Optimizer** | Q-bit strings + Han-Kim rotation gates for deploy/route/fuel; QPSO for continuous speed; Pareto archive with constrained domination; repair not penalty | **30/30** exact on tiny instance; **+19%** hypervolume over greedy at 60 vessels |
| **6 · Verification** | Exact CP-SAT MILP on small instances; NSGA-II / greedy / random baselines at equal budget; 30 seeds; Wilcoxon; shared hypervolume reference | **399 tests**, every edge case in the register mapped |

### The encoding (judges will ask — B1 "explain your encoding")

Per vessel: `[deploy bit | route bits | fuel bits]`, concatenated across the fleet.
Each bit is a **Q-bit** — a pair of amplitudes `(cos θ, sin θ)` in superposition.
Observation collapses it; the **rotation gate** `θ ← θ ± δ` biases amplitudes toward
the best solution, but only where an individual disagrees with a solution that beat
it (Han & Kim 2002). Route and fuel bits decode **modulo** the option count, so every
bit pattern is a valid plan (Q-01). Speed is continuous and carried by **QPSO** — the
hybrid encoding of Q-02. A 60-vessel, 6-route, 4-fuel instance is **360 Q-bits**.

### Rotation angle (judges will ask — "what's your angle and how did you choose it?")

`δ = 0.03π · n_bits / n_generations` — scaled to problem size and budget, **calibrated
by sweep, not inherited**. A fixed angle tuned on a 14-bit problem lost two orders of
magnitude of hypervolume on a 360-bit fleet; the size-scaled rule satisfies the
2-vessel, 12-vessel and 60-vessel instances with one setting. Full sweep table in
`docs/FINDINGS_PHASE3.md` §2.

### Tech stack

Python 3.10 · NumPy · pandas · scikit-learn · **XGBoost** (monotone constraints) ·
**pymoo** (NSGA-II) · **OR-Tools CP-SAT** (exact MILP) · SciPy (Wilcoxon) · own
QIEA/QPSO/QUBO in NumPy · pytest (399 tests) · ruff · uv. Planned: FastAPI, React +
Plotly, PostgreSQL, Docker (on-prem, offline).

---

## SLIDE 4 — FEASIBILITY & VIABILITY

### Feasibility: it already runs on real data

| evidence | number |
|---|---|
| Data source | **EU MRV**, mandatory regulatory filings, third-party verified, **no synthetic rows** |
| Scale | 106,922 ship-years, 25,967 ships, 2018–2025 |
| Model accuracy (unseen vessels) | **19.9% MAPE**, medAPE 13.4%, R² 0.603 |
| Model accuracy (unseen *years*, 2024–25) | 24.4% MAPE |
| Interval calibration | nominal 80% → **empirical 80.0%** (out-of-fold) |
| Optimizer verified | **30/30 seeds exact** vs proven optimum, gap **0.0000%** |
| Optimizer scale | 200 vessels, **10²⁷⁹ combinations**, feasible in **54 s** |
| Test coverage | **399 tests**, all edge cases in the register mapped by ID |

Use charts **05** (accuracy vs baselines), **06** (per class), **11** (QIEA verification), **10** (scalability).

### Viability: a real programme with a real first user

- **Green Tug Transition Programme (GTTP):** four Major Ports (Deendayal, JNPA,
  Visakhapatnam, V.O. Chidambaranar) have placed electric-tug work orders. Each faces
  exactly our fleet-mix, fuel and duty-cycle decision.
- **Mormugao Port Authority** signed a strategic quantum-technology agreement with
  Egreen Quanta (July 2026) and runs **Harit Shrey**, India's first green-ship
  incentive scheme. Our emission scores plug straight into it. This is the sponsor's
  live pilot.
- **Harit Sagar:** ports must cut emissions per tonne of cargo **30% by 2030, 70% by
  2047**. Our KPI tracker reports the glide path and on/off-track status.
- **Business model:** licence to port authorities / MoPSW, on-premises deployment,
  annual support. Data stays in India.

### Risks and mitigations (list every one *with* a mitigation — PPT-09)

| risk | mitigation, already built |
|---|---|
| **Data scarcity for Indian coastal vessels and tugs** — MRV has no ships < 5000 GT | Physics + vessel-class **cold-start path** (P-05): the model flags what it can't see and widens intervals; cold rows get **84.2%** coverage vs 80% nominal. Tug figures are labelled physics-derived. |
| **"Quantum-inspired" is hype** | Benchmarked honestly at equal budget over 30 seeds. We publish the ties (12-vessel: p = 0.14) and the wins (60-vessel: p = 0.004). Never claim quantum speedup. |
| **Emission factors are uncertain** | Every factor lives in config with a `source` and `confidence` tag. CH₄ and N₂O are now **measured** from 26,621 ship-years. Every result reports its weakest input's confidence. |
| **Port fuel availability changes fast** | Year-indexed config; `"planned"` deliberately does not count as available. Specific years are placeholders to confirm with port authorities — stated openly. |
| **IMO Net-Zero Framework is unsettled** (adjourned Oct 2025, revisit late 2026) | Policy is a scenario toggle, not code. Harit Sagar and EU rules apply regardless. |
| **Coal-heavy grid makes shore power look worse** | We model it honestly (crossover 2035) and pair it with the local air-quality benefit that is immediate. |
| **Live demo failure** | Everything runs offline; pre-computed large scenario; recorded backup. |

### Honest limitations (say these before a judge finds them)

- Offshore vessels: **117.7% MAPE** — 96% lack a design certificate; flagged cold-start.
- Ro-pax / passenger: 30–34% MAPE — hotel-load-dominated; needs a duty-cycle model.
- Pareto front thins above ~50 vessels at a 6,000-evaluation budget.
- MILP verification is exact for the *speed-discretised* problem, a bound for continuous.
- No result has run on quantum hardware; the QUBO is the readiness claim, verified classically.

---

## SLIDE 5 — IMPACT & BENEFITS

Every number here has a stated scenario (PPT-02). Use charts **01, 03, 04, 12**.

### Quantified impact

| impact | number | scenario |
|---|---|---|
| **Avoids a wrong procurement** | A stack-only tool scores grey ammonia at **−100%**; the truth is **+40%**. A port buying on the wrong column would *increase* emissions and report reaching zero. | 1 TJ shaft work, AR6 GWP100 |
| **Local air quality, immediately** | Each electrified harbour tug removes **35.6 t NOx, 1.25 t SOx, 187 kg PM** from the port city per year | 12 TJ/yr tug, Mormugao |
| **Carbon on the honest timeline** | Electric tug CO₂: **+20%** vs diesel in 2030 → **−4%** in 2035 → **−60%** in 2047 as the grid decarbonises | CEA grid path |
| **Fleet-level trade-off made visible** | Coastal fleet, 2047: **−91% GHG for +45% cost**, with **14** feasible intermediate mixed-fuel plans between | 12 vessels, 3 routes, e-methanol available |
| **Slow steaming, correctly** | 14→12 kn saves **23.9%/mile** on a bulk carrier but **0.1%/mile** on a cruise ship — a universal cube law cannot express this | model with class hotel-load share |
| **Harit Sagar tracked** | 35% clean-fuel share puts a 1 Mt port **on track** for −30% by 2030; 85% for −70% by 2047 | tracker output |
| **Range reality for hydrogen** | Retrofit range **7%** of diesel → tugs and ferries only; **4,842 t** of cargo lost on a purpose-built 5,000 nmi tank | F-04 / F-05 |

### Who benefits

| beneficiary | benefit |
|---|---|
| **Port authorities (GTTP)** | A defensible fleet-transition plan with the CO₂ and air-quality cases separated, and infrastructure constraints respected |
| **Mormugao Port** | Emission scores that plug into Harit Shrey; the sponsor's own pilot site |
| **MoPSW / Harit Sagar** | Per-tonne-of-cargo intensity against the 2030 / 2047 targets, per port |
| **Coastal & inland operators (Harit Nauka)** | Which vessels, which fuels, which speeds — with a Pareto front, not one number |
| **National Quantum Mission** | A real maritime problem already formulated as QUBO, ready for domestic hardware |
| **Public health in port cities** | Quantified NOx / SOx / PM removal, separately from carbon |

### Alignment with national programmes

GTTP · Harit Sagar Green Port Guidelines · Harit Nauka (inland, full green transition
by 2047) · National Green Hydrogen Mission (Deendayal, Paradip, V.O. Chidambaranar as
hubs) · Maritime India Vision 2030 · National Quantum Mission · Mormugao–Egreen Quanta
strategic agreement.

---

## SLIDE 6 — RESEARCH & REFERENCES

### Data
- EU MRV "Publication of information", EMSA THETIS-MRV — https://mrv.emsa.europa.eu
  (reporting periods 2018–2025; pinned versions in `data/raw/mrv/manifest.json`)
- CEA CO₂ Baseline Database for the Indian Power Sector (grid emission factor)

### Standards and regulation
- IMO MEPC.308(73) — 2018 EEDI Guidelines (LHV and C_f values)
- IMO Fourth GHG Study 2020 (engine emission factors — our measured CH₄ reproduces its default exactly)
- FuelEU Maritime Annex I/II (well-to-tank defaults, C_slip values)
- IPCC AR5 / AR6 WG1 (GWP100 — and we recovered EMSA's AR5 use exactly from 28,125 filings)
- Harit Sagar Green Port Guidelines, MoPSW
- Green Tug Transition Programme, MoPSW
- PIB: Mormugao Port Authority – Egreen Quanta Strategic Quantum Technology agreement (July 2026)
- IMO: Net-Zero Framework talks to resume in 2026

### Methods
- Han & Kim (2002), *Quantum-inspired evolutionary algorithm for a class of combinatorial optimization*, IEEE TEC — the Q-bit / rotation-gate scheme
- Sun, Feng & Xu (2004), *Particle swarm optimization with particles having quantum behavior*, IEEE CEC — QPSO
- Deb et al. (2002), NSGA-II — the baseline
- Google OR-Tools CP-SAT — exact verification
- ScienceDirect: *Improved quantum genetic algorithm for ship speed optimization*

### Competitors reviewed
ZeroNorth voyage optimisation · Wärtsilä Fleet Optimisation Solution · DNV Alternative
Fuels Insight · Maritime Executive on quantum port scheduling (Feb 2026)

### Links to add
GitHub repository · demo video · live link *(when Phases 4–5 are done)*

---

## APPENDIX A — Every number, with its source

### A1. Data (`data/raw/mrv/manifest.json`, `docs/FINDINGS_PHASE1.md`)

| item | value |
|---|---|
| Reporting periods | 2018, 2019, 2020, 2021, 2022, 2023, 2024, 2025 (v275, v228, v209, v218, v241, v91, v242, v57) |
| Ship-years | 106,922 |
| Unique ships | 25,967 |
| Usable after quality flags | 100,528 (94.0%) |
| Ships with ≥5 reporting periods | 9,182 |
| Rows with EMSA's literal `"Division by zero!"` | 5.2% (distance), 9.0% (transport work) — zero-distance ships, classified not imputed |
| Implied CO₂ factors above the physical maximum | 166 ship-years (max 9.68 tCO₂/t; physics caps at ~3.25) — flagged |
| Within-ship year-to-year σ of log energy intensity | 0.112 (≈ 11%) — the irreducible floor for *any* method |
| Between-ship σ | 0.464 · **ICC 0.945** |
| 2024+ CH₄/N₂O measured rows | 31,339 |

### A2. Prediction model (`artifacts/evaluation.json`)

Split A — unseen vessels (75,423 train / 25,105 test rows; 18,788 / 6,263 ships):

| model | MAE | RMSE | MAPE | medAPE | R² | 80% cov |
|---|---|---|---|---|---|---|
| class mean (naive) | 1528.5 | 2780.8 | 32.6% | 23.1% | 0.354 | 80.3% |
| ridge (log) | 1482.6 | 2724.9 | 31.5% | 22.2% | 0.379 | 80.4% |
| **XGBoost** | 958.7 | 2178.7 | **19.9%** | **13.4%** | **0.603** | 80.0% |
| XGBoost + monotone | 995.4 | 2220.0 | 20.6% | 14.0% | 0.588 | 80.1% |

Split B (train ≤2023 → 2024–25): best 24.4% MAPE. Split C (unseen vessels **and** years): best 24.4%.

Cold-start stratification (Split A): warm n=23,652 → **19.3% MAPE, R² 0.697, cov 79.8%**;
cold n=1,453 → 41.8% MAPE, R² 0.146, **cov 84.2%** (conservative, as intended).

Per class (unseen vessels): vehicle carrier 14.1% · bulk 15.2% · chemical tanker 16.6% ·
oil tanker 18.3% · LNG carrier 19.9% · container 23.7% · ro-pax 34.3% · offshore 117.7% (cold).

Quantum-inspired ablation (`ablation_p13.json`): untuned 20.875 · random search 20.728 ·
QPSO 20.679 — **a tie; reported as such.** Feature selection (`feature_selection.json`):
QIEA found the **verified exhaustive optimum** (gap 0.0000%) using 26% of the space,
but random search tied it — space too small to discriminate.

### A3. Emissions (`artifacts/emissions_demo.json`, AR6 GWP100, per TJ shaft work)

| fuel / pathway | stack-only | lifecycle | stack says | truth |
|---|---|---|---|---|
| MGO fossil | 75.1 | 90.7 | 0% | 0% |
| HFO fossil | 77.5 | 92.2 | +3% | +2% |
| LNG fossil | 57.3 | 87.1 | −24% | **−4%** |
| Methanol grey | 66.3 | 100.2 | −12% | **+10%** |
| E-methanol | 0.0 | 8.6 | −100% | −91% |
| Ammonia grey | 0.0 | 126.5 | −100% | **+40%** |
| Ammonia blue | 0.0 | 44.7 | −100% | −51% |
| Ammonia green | 0.0 | 18.6 | −100% | −79% |
| Hydrogen grey | 0.0 | 99.1 | −100% | **+9%** |
| Hydrogen green | 0.0 | 7.2 | −100% | −92% |

Methane slip (LNG vs diesel): FuelEU default 3.10% → +5.5% · measured median 1.72% → **−3.9%** · best engine 0.20% → −14.3%.
Measured slip spread: p25 0.63%, p50 1.72%, p75 2.12%, p95 2.97% (745 ship-years).

Measured conventional-fuel factors: CH₄ **0.050 g/kg** (17,288 HFO + 9,333 MGO ship-years; reproduces IMO 4th GHG Study default exactly), N₂O **0.180 g/kg**.

EU GWP set recovered exactly: **CH₄ = 28.00, N₂O = 265.00**, max relative residual 1.6×10⁻¹⁴ over 28,125 filings.

Electrification (per TJ shaft, diesel = 201.5): 2026 +38% · 2030 +20% · 2033 +6% · **2035 −4%** · 2040 −28% · 2047 −60%.

Harbour tug, 12 TJ/yr: diesel 2,418 tCO₂e / 35,597 kg NOx / 1,249 kg SOx / 187 kg PM ·
LNG 2,323 / 5,556 / 0 / 11 · e-methanol 229 / 33,585 / 50 / 72 · green H₂ FC 192 / 0 / 0 / 0 ·
electric 2030 2,910 / 0 / 0 / 0 · electric 2035 2,328 / 0 / 0 / 0.

Tank / range (3,000 MJ/nmi vessel, MGO tank sized for 5,000 nmi = 489 m³): retrofit range
MGO 5,000 · methanol 1,803 (36%) · LNG 1,624 (32%) · ammonia 1,284 (26%) · **hydrogen 374 (7%)**.
Cargo displaced on a purpose-built 5,000 nmi tank: LNG 813 t · methanol 693 t · ammonia 1,132 t · **hydrogen 4,842 t**.

Harit Sagar (1 Mt cargo/yr, 120 TJ, diesel → e-methanol): 2026 @10% clean OFF track ·
2030 @35% **ON** · 2035 @50% ON · 2047 @85% ON.

### A4. Optimizer (`artifacts/optimizer_benchmark.json`)

Q-08: MILP proven optimum INR 108,523,810 · QIEA+QPSO **30/30 exact**, mean gap 0.0000%, worst 0.0000%.

Coastal, 12 vessels, 2.33×10¹³ combinations, 30 seeds × 6,000 evaluations:

| algorithm | HV mean | HV std | front | time | feasible | p vs QIEA |
|---|---|---|---|---|---|---|
| **QIEA+QPSO** | **4.195e21** | 4.59e20 | 6.4 | 2.87 s | 30/30 | — |
| NSGA-II | 3.938e21 | 6.67e20 | 6.2 | 2.25 s | 30/30 | 0.140 (tie) |
| greedy | 1.710e21 | 8.0e05 | 3.0 | 0.05 s | 30/30 | <0.001 |
| random | 6.346e20 | 6.24e20 | 1.7 | 1.82 s | 30/30 | <0.001 |

Large, 60 vessels, 7.52×10⁸³ combinations, 10 seeds × 6,000 evaluations:

| algorithm | HV mean | HV std | front | time | feasible | p vs QIEA |
|---|---|---|---|---|---|---|
| **QIEA+QPSO** | **3.885e26** | 1.07e25 | 4.8 | 15.3 s | 10/10 | — |
| greedy | 3.602e26 | 7.2e10 | 2.0 | 1.0 s | 10/10 | **0.002** |
| NSGA-II | 3.260e26 | 4.57e25 | 2.8 | 8.6 s | 10/10 | **0.004** |
| random | 8.381e24 | 1.55e25 | 0.6 | 11.9 s | 6/10 | 0.002 |

Scalability (3 seeds, 6,000 evals): 10 vessels 5.8 s · 25 → 9.4 s · 50 → 14.6 s · 100 → 23.7 s · **200 (3.9×10²⁷⁹) → 53.7 s**, feasible 3/3 at every size.

QUBO: 12 variables on the 2-vessel instance; ground state **exactly** matches CP-SAT; penalty weight neither too low (ground state feasible) nor too high (relative energy spread 3.7×10³ among feasible states); density 0.54.

QIEA vs random on knapsack with known optimum (30 seeds, equal budget): 4 bits **100%** vs 63% · 6 bits **100%** vs 70% · 8 bits **100%** vs 53% · 10 bits 86% vs 46% · 12 bits **96%** vs 26%.

### A5. Engineering

399 tests passing · ruff clean · 33 source modules · every edge-case ID in `docs/TEST_AND_EDGE_CASES.md` (D, P, M, Q, F, S, B series) mapped to a test.

---

## APPENDIX B — Chart index

| file | what it shows | slide |
|---|---|---|
| `01_stack_vs_lifecycle.png` | Stack-only vs lifecycle, per fuel; three sign flips marked | 2, 5 |
| `02_methane_slip.png` | FuelEU default vs measured slip, and what it does to LNG's sign | 2, 4 |
| `03_electrification_crossover.png` | Electric vs diesel on the Indian grid by year; crossover 2035 | 5 |
| `04_tug_carbon_vs_air_quality.png` | One tug: CO₂e and NOx side by side per option | 5 |
| `05_model_accuracy.png` | MAPE and R² vs baselines, unseen vessels | 4 |
| `06_per_vessel_class.png` | MAPE per class, weak classes visible | 4 |
| `07_cold_start.png` | Warm vs cold: MAPE, R², interval coverage | 4 |
| `08_slow_steaming_by_class.png` | 14→12 kn saving per day and per mile by class | 3, 5 |
| `09_optimizer_benchmark.png` | Hypervolume vs NSGA-II/greedy/random, both instances, with p-values | 3, 4 |
| `10_scalability.png` | Runtime vs fleet size to 200 vessels | 4 |
| `11_qiea_verification.png` | Exact-optimum hit rate vs brute force, QIEA vs random | 3, 4 |
| `12_pareto_front.png` | A real Pareto front: −91% GHG for +45% cost, 14 plans | 2, 5 |
| `13_where_quantum_inspired_pays_off.png` | All six comparisons: ties and wins, honestly | 3, 4 |
| `14_architecture.png` | End-to-end pipeline diagram | 3 |

---

## APPENDIX C — Judge questions, with the answer and the number

From `docs/TEST_AND_EDGE_CASES.md` Part B. Assign one owner per topic.

**"Is this actual quantum computing?"** No. Quantum-*inspired*: classical algorithms
using superposition-style Q-bit encoding and rotation-gate updates, on ordinary
hardware. The discrete core is also written as a QUBO, verified to match the exact
optimum, so it can move to annealers or National Quantum Mission hardware when
practical. Never say "quantum speedup".

**"Why not just use a normal GA?"** We benchmarked exactly that. At 12 vessels it's a
tie (p = 0.14). At 60 vessels the quantum-inspired optimizer beats NSGA-II with
p = 0.004 and greedy with p = 0.002 at equal budget. The advantage appears when the
space is large and rugged — chart 13 shows the ties too.

**"Explain your encoding."** Chart 14 + Slide 3 text. One Q-bit per binary decision;
route and fuel by binary expansion decoded modulo the option count so every string is
valid; speed continuous via QPSO.

**"What's the rotation angle and how did you choose it?"** `0.03π · n_bits / n_gens`,
calibrated by a documented sweep across three instance sizes. A fixed angle from a
small problem lost 100× hypervolume on a large one — we show the failure.

**"Have you run on a real quantum device?"** No. The QUBO's ground state matches
CP-SAT on small instances, classically. That is the readiness claim, no more.

**"How is this different from ZeroNorth / Wärtsilä?"** They optimise one voyage.
We decide which ships, which fuels, which speeds across a fleet over years, with
lifecycle emissions and Indian port constraints. Different decision level.

**"Where does your data come from?"** EU MRV — 106,922 verified regulatory filings,
25,967 ships, 2018–2025, from EMSA's public API, version-pinned. Plus CEA grid data
and IMO/FuelEU factors, each tagged with a confidence level. No synthetic rows.

**"How accurate is your fuel prediction?"** 19.9% MAPE on 6,263 ships never seen in
training, R² 0.603, with a leakage-safe split by vessel. Warm rows 19.3%; we report
cold rows (41.8%) separately and their intervals cover 84%.

**"Isn't LNG a clean fuel?"** Only partly. Measured methane slip across 745 LNG
carrier ship-years is 1.72% of fuel — with methane at 28–30× CO₂ that erases most
of LNG's 24% combustion advantage, leaving −4%. The regulatory default (3.1%) would
even make it worse than diesel.

**"Is shore power actually green in India?"** On carbon, not until ~2035 — the grid
is too coal-heavy for a 45%-efficient marine diesel to lose. On local air quality,
immediately: 35.6 t NOx per tug per year off the quayside. We show both.

**"What about ships with no sensors?"** Physics-informed model with vessel-class
baselines; the cold-start path flags them and widens the interval. Tugs are handled
exactly this way.

**"Who would actually use this?"** GTTP ports — Deendayal, JNPA, Visakhapatnam,
V.O. Chidambaranar have placed electric-tug orders. First case: Mormugao, the
sponsor's pilot site, plugging into Harit Shrey.

**"How does this scale to 1,000 vessels?"** Runtime is roughly linear: 200 vessels in
54 s at 6,000 evaluations. Front density thins above 50 at that budget; more budget
fixes it, and the cost is tabled.

---

## APPENDIX D — What NOT to claim

- Do **not** say "quantum" alone; always "quantum-inspired" (PPT-04).
- Do **not** claim quantum speedup or hardware execution.
- Do **not** quote the coastal-scenario benchmark as a win; it is a tie (p = 0.14).
- Do **not** present port fuel-availability years as confirmed; they are placeholders pending port-authority confirmation.
- Do **not** present tug figures as measured; they are physics-derived (no tug is in MRV).
- Do **not** present the 11% within-ship floor as a target for unseen-vessel accuracy; it is a lower bound for any method.
- Do **not** add NOx/SOx/PM into a CO₂e total.
- Do **not** cite offshore or ro-pax accuracy as representative; state them as known weak classes.

Every number in this report can be regenerated:
`uv run python scripts/evaluate_model.py --calibrate-intervals`,
`uv run python scripts/emissions_demo.py`,
`uv run python scripts/benchmark_optimizer.py --seeds 30`,
`uv run python scripts/make_charts.py`.
