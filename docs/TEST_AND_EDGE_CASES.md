# SIH 2026 — Test Cases & Edge Cases

**Problem Statement 2 (Egreen Quanta):** Quantum-Inspired Fuel Consumption Prediction and Green Fleet Optimization

This document covers three perspectives:

1. **Implementation**: what to test in the code, and the edge cases that will break a naive build.
2. **Pitching**: the hard questions judges will ask, and the situations that go wrong on stage.
3. **PPT**: slide-by-slide checks for the SIH idea submission deck.

**How to use the tables:** Every row has an ID so you can track it in your issue tracker. "Expected" is what a correct system should do. Anything marked ⭐ is high priority, because a judge could plausibly test or ask about it.

---

## Part A — Implementation Test Cases

### A1. Data Pipeline & Ingestion

| ID | Test / Edge case | Expected behaviour |
|---|---|---|
| D-01 ⭐ | Unit mismatch: speed in knots vs km/h, fuel in tonnes vs m³, consumption per day vs per hour | Convert all inputs to one canonical unit system on ingest. Reject or flag files with ambiguous units instead of guessing silently. |
| D-02 | Missing values in speed, draft, or weather columns | Impute with a documented method or drop the row, and log how many rows were affected. Never impute the target (fuel). |
| D-03 | Negative fuel, zero fuel while sailing, or impossible speed (e.g., 60 knots for a bulk carrier) | Flag as outliers and exclude from training. Show the count in a data-quality report. |
| D-04 | Sensor spikes (a single reading 10× its neighbours) | Filter with a rolling median or IQR rule. The test must confirm genuine high-load events such as heavy weather are *not* removed. |
| D-05 | AIS gaps: a vessel disappears for hours or days | Don't interpolate across long gaps. Split the voyage into segments. |
| D-06 | Duplicate records, or the same timestamp reported twice | Deduplicate deterministically and keep the higher-quality source. |
| D-07 | Timezone mix: UTC AIS data vs IST port logs | Normalise everything to UTC internally. Display local time only in the UI. |
| D-08 | Uploaded CSV has the wrong schema, extra columns, or reordered columns | Validate against the schema and return a human-readable error naming the bad column. Must not crash the app. |
| D-09 | Very large file (millions of AIS rows) | Stream or chunk the ingest without running out of memory. Show a progress indicator. |
| D-10 | Empty file, or header-only file | Return a clear "no data" error, not a stack trace. |
| D-11 | Weather data at a different resolution than vessel data (e.g., ERA5 hourly grid vs 15-min vessel logs) | Spatial and temporal join to the nearest grid point and time. Test that the join doesn't shift data by a timezone offset. |
| D-12 | Vessel type label inconsistency ("Bulk Carrier", "bulker", "BC") | Map to a controlled vocabulary. Unknown labels go to "Other" with a warning. |

### A2. Fuel Consumption Prediction Model

| ID | Test / Edge case | Expected behaviour |
|---|---|---|
| P-01 ⭐ | **Data leakage.** A random train/test split puts rows from the same voyage in both sets. | Split by vessel or by time (train on earlier data, test on later). Report accuracy on *unseen vessels* separately. This is the most common silent mistake in fuel-prediction projects. |
| P-02 ⭐ | Physics sanity: fuel should increase with speed, roughly with the cube of speed | Monotonicity test: with everything else fixed, higher speed must never predict lower fuel. Plot the predicted curve and check its shape. |
| P-03 | Zero speed, i.e., vessel at anchor or berthed | Predict only auxiliary/hotel load, not zero and not negative. |
| P-04 | Laden vs ballast voyage for the same vessel | Laden should predict higher fuel at the same speed. Draft or load must be a feature. |
| P-05 ⭐ | **Cold start:** a new vessel with no history | Fall back to the physics or vessel-type baseline with wider uncertainty bands. The model must not crash or return nonsense. |
| P-06 | Weather beyond the training range (cyclone-level wind or waves) | Flag as out-of-distribution. Widen the prediction interval or refuse to predict. Don't extrapolate silently. |
| P-07 | Hull fouling: vessel performance degrades over months | Include time since last dry-dock or hull cleaning as a feature, or test that predictions drift correctly over time. |
| P-08 | Alternative fuel vessel: methanol has roughly half the energy per kg of diesel | Predict **energy** consumption (MJ or kWh), then convert to fuel mass using each fuel's heating value. Predicting fuel *mass* directly across fuel types is wrong. |
| P-09 | Dual-fuel engines that need pilot fuel (ammonia and methanol engines burn some diesel) | Model the pilot fuel fraction separately. It adds to both cost and emissions. |
| P-10 | Prediction uncertainty | Output a prediction interval (e.g., P10/P50/P90), not just a point estimate. The optimizer should be able to use the conservative value. |
| P-11 | Model reproducibility | The same seed and same data must give the same model and the same metrics. |
| P-12 | Metrics reporting | Report MAE, RMSE, MAPE, and R² per vessel type, not only overall. Overall metrics hide poor performance on minority vessel types. |
| P-13 | The quantum-inspired component (QPSO tuning, or quantum-inspired feature selection) | Ablation test: the same model *without* the quantum-inspired part. If there's no difference, say so honestly. |

### A3. Mathematical Formulation & Constraints

| ID | Test / Edge case | Expected behaviour |
|---|---|---|
| M-01 ⭐ | Cargo demand exceeds total fleet capacity | Return "infeasible" with an explanation, and suggest the minimum extra capacity needed. Never return a "solution" that silently drops cargo. |
| M-02 | Zero cargo demand | Return a valid solution: no vessels deployed, or minimum mandated service. |
| M-03 | Only one vessel and one route | Solver still works. Result should match a hand calculation. |
| M-04 | All vessels identical | Symmetry: many equivalent solutions. The solver must not waste time cycling between them, and the output should be stable. |
| M-05 ⭐ | Emission cap so tight that no fuel/vessel combination satisfies it | Infeasible, with a message like: "Cap requires X% cut; best achievable with current fuel availability is Y%." |
| M-06 | Speed bounds: engines have minimum safe load (slow steaming limit) and a maximum speed | Optimized speed must always lie within `[v_min, v_max]` for that vessel. Test with the optimum pushing against both bounds. |
| M-07 | Schedule time windows at ports | Arrival must fall inside the window. Test a window that's impossible at max speed, which should be flagged infeasible. |
| M-08 | Vessel-route compatibility: draft limits at a shallow port, vessel too large for a berth | Incompatible assignments must never appear in the output. |
| M-09 | Integer constraints: vessel counts must be whole numbers | No fractional vessels in the final answer, even if the relaxation produces them. |
| M-10 | Conflicting objectives (cheapest vs cleanest) | Output a Pareto front. Test that no solution on the front is dominated by another. |
| M-11 | Extreme weights (100% on cost, 0% on emissions) | The result should match the single-objective minimum-cost solution. Use this as a correctness check. |
| M-12 | Round-trip consistency | A vessel ending at port B must start its next leg from port B. |

### A4. Quantum-Inspired Optimization Engine

| ID | Test / Edge case | Expected behaviour |
|---|---|---|
| Q-01 ⭐ | **Encoding validity:** every Q-bit string observed (collapsed) must decode to a valid fleet decision | Decoder test on random Q-bit states. Invalid decodes must be repaired or penalised, and you must document which approach you use. |
| Q-02 | Mixed variables: vessel choice and fuel type are discrete, speed is continuous | Test the hybrid encoding, e.g., Q-bits for discrete choices plus real-coded or QPSO for speed. Speed must never decode outside its bounds. |
| Q-03 ⭐ | **Premature convergence:** Q-bit probabilities collapse to 0/1 too early | Track population diversity per generation. If the rotation angle is too large, diversity dies early. Add a test that fails when diversity hits zero before X% of iterations. |
| Q-04 | Rotation angle too small | Convergence becomes extremely slow. Tune it and document the value, or use an adaptive angle. |
| Q-05 | Constraint handling | Compare penalty vs repair approaches. The final output must always be feasible if a feasible solution exists. |
| Q-06 | Determinism | Same seed gives the same result. Different seeds give a *distribution* of results, so run at least 30 seeds for benchmarking. |
| Q-07 | Time limit reached before convergence | Return the best solution found so far, marked as "time-limited." Don't crash or return nothing. |
| Q-08 | Known-optimum check | On small instances, solve exactly with MILP (OR-Tools or Gurobi) first. The quantum-inspired result must be within a stated gap, e.g., ≤2%. |
| Q-09 | Tiny instance (2 vessels, 1 route) | Must still find the exact optimum. If a metaheuristic fails on a trivial case, judges will lose trust immediately. |
| Q-10 | QUBO formulation (for the "quantum-ready" claim) | Test that the QUBO's lowest-energy state matches the MILP optimum on small instances. Test the penalty weight choice: too low gives infeasible minima, too high flattens the landscape. |

### A5. Alternative Fuels & Lifecycle Emissions

| ID | Test / Edge case | Expected behaviour |
|---|---|---|
| F-01 ⭐ | Grey vs green versions of the same fuel (e.g., ammonia from natural gas vs renewables) | Use different well-to-wake emission factors. Grey ammonia or hydrogen can be *worse* than diesel on a lifecycle basis, so the model must be able to show that. |
| F-02 ⭐ | Methane slip from LNG engines | Include unburnt methane in the GHG total, converted using its global warming potential. Without this, LNG looks misleadingly clean. |
| F-03 | N₂O from ammonia combustion | Include it. N₂O is a very potent greenhouse gas, and ignoring it overstates ammonia's benefit. |
| F-04 | Hydrogen range limit (very low energy per volume) | Hydrogen should only be feasible on short routes, such as harbour tugs, ferries, and short coastal legs. Test that a long-haul hydrogen assignment is rejected on tank capacity. |
| F-05 | Tank volume reduces cargo space for low-density fuels | Assigning methanol or ammonia to a vessel should reduce its effective cargo capacity or range. Test this trade-off explicitly. |
| F-06 ⭐ | Fuel not available at a port (no methanol bunkering there) | The optimizer must not assign a fuel the vessel can't refuel on that route. |
| F-07 ⭐ | Shore power on a coal-heavy grid | Shore-power emissions depend on the Indian grid emission factor (use the latest CEA CO₂ baseline database). Test that the tool shows shore power's benefit correctly, which may be smaller than expected for CO₂ but large for local air pollution at the port. |
| F-08 | Shore power only works if both port and vessel are equipped | Test combinations: port has OPS but vessel doesn't, vessel does but port doesn't, both do. |
| F-09 | Emission factor source changes (e.g., IMO updates its LCA guidelines) | Emission factors live in a config file or database, not hard-coded. Changing one number must update all results. |
| F-10 | Unit consistency | GHG totals reported in tonnes CO₂-equivalent, with a clearly stated time horizon (e.g., GWP100). |

### A6. Scenario & Policy Engine

| ID | Test / Edge case | Expected behaviour |
|---|---|---|
| S-01 ⭐ | IMO Net-Zero Framework adopted vs not adopted | Both scenarios run from the same data. Policy is a toggle, not a code change. |
| S-02 | Carbon price from 0 up to a high value | Fleet mix should shift toward cleaner fuels as the price rises. If it doesn't, something is broken. |
| S-03 | Fuel price shock (e.g., diesel +50%) | Results update consistently. Speeds should drop (slow steaming becomes more attractive). |
| S-04 | Green hydrogen or methanol becomes available at an Indian hub in a future year | Multi-year scenarios reflect fuel availability by year and port. |
| S-05 | Grid decarbonisation over time | The shore-power benefit grows in later years as the grid gets cleaner. |
| S-06 | Harit Sagar KPI check | Report emissions per tonne of cargo against the 2030 (−30%) and 2047 (−70%) targets. |
| S-07 | Contradictory scenario inputs (e.g., ammonia-only fleet but no ammonia supply anywhere) | Validation error before running, with a clear message. |

### A7. Decision Support System (UI, API, Reports)

| ID | Test / Edge case | Expected behaviour |
|---|---|---|
| U-01 | Invalid form input: negative capacity, speed of 0 for a sailing leg, text in number fields | Inline validation. The request never reaches the solver. |
| U-02 | Long-running optimization | Run asynchronously with a progress bar and a cancel button. The UI must not freeze. |
| U-03 | User closes the browser mid-run | The job continues or is cleanly cancelled. No orphaned processes. |
| U-04 | Concurrent users running different scenarios | Results don't leak between sessions. |
| U-05 | Report generation (PDF or Excel) | The report matches the on-screen numbers exactly. Test with an empty result and with a very large fleet. |
| U-06 | Pareto front visualisation with 1 point, or with 1,000 points | Both render legibly. |
| U-07 | API: missing auth token, malformed JSON, wrong API version | Correct HTTP status codes (401, 400, 404) with readable messages. |
| U-08 | Low-bandwidth or offline use at a port office | Core features work on-prem without internet. |
| U-09 | Explainability | For any recommendation, the user can see *why*: which constraint was binding, and what the trade-off was. |

### A8. Benchmarking (Required by the PS)

| ID | Test / Edge case | Expected behaviour |
|---|---|---|
| B-01 ⭐ | **Fair comparison:** every algorithm gets the same budget | Equal number of objective-function evaluations, or equal wall-clock time. State which one you used. |
| B-02 ⭐ | Statistical significance | At least 30 independent runs per algorithm. Report mean, standard deviation, and a Wilcoxon or similar test, not a single best run. |
| B-03 | Multi-objective metrics | Hypervolume (same reference point for all algorithms), IGD, and number of Pareto solutions. |
| B-04 | Baselines | Prediction: linear regression, XGBoost, physics-only model. Optimization: NSGA-II, standard GA, PSO, and MILP (exact on small instances). |
| B-05 | Convergence plots | Best objective vs iterations, averaged across runs with shaded variance bands. |
| B-06 | Scalability curve | Runtime and solution quality at, e.g., 10, 50, 200, and 1,000 vessels. |
| B-07 ⭐ | Honest reporting | If the quantum-inspired method loses on some instances, show it. Explain *where* it wins (e.g., larger, more constrained instances) and where it doesn't. |

### A9. Performance, Security & Deployment

| ID | Test / Edge case | Expected behaviour |
|---|---|---|
| X-01 | Large-scale demo: 500+ vessels, multiple routes, 10-year horizon | Completes in a stated time on stated hardware. |
| X-02 | Memory usage on the biggest instance | Stays within available RAM. No crash. |
| X-03 | Data sovereignty | Can run fully on-prem. No port data sent to external services by default. |
| X-04 | Role-based access (port admin vs analyst vs viewer) | Viewers can't change data or scenarios. |
| X-05 | Audit trail | Every scenario run is logged with inputs, version, and timestamp, so any result can be reproduced. |
| X-06 | Dependency failure (e.g., weather API down) | Fall back to cached data, with a visible warning. |

---

## Part B — Pitching: Judge Questions & How to Answer

Judges will likely include a quantum expert from the sponsor, a maritime or port official, and a general technical evaluator. Prepare for all three.

### B1. Questions on the Quantum Claim

| Likely question | How to answer |
|---|---|
| ⭐ "Is this actual quantum computing?" | No, and say it clearly. It's quantum-*inspired*: classical algorithms using quantum concepts such as superposition-style probabilistic encoding and rotation-gate updates. It runs on normal hardware today. The discrete core is also formulated as a QUBO, so it can move to real quantum hardware when that becomes practical. |
| ⭐ "Why not just use a normal genetic algorithm?" | Show your benchmark. Point to where the quantum-inspired version is better (e.g., diversity, convergence on larger instances) with numbers. If the gain is modest, say so, and highlight the QUBO-readiness as the strategic reason. |
| "Explain your encoding." | Have one diagram ready: how a Q-bit string maps to vessel choice, fuel type, and speed band. |
| "What's the rotation angle and how did you choose it?" | State the value and show the tuning experiment. Never say "we used the default." |
| "Have you run it on a real quantum device?" | If not, say so. Optionally mention you validated the QUBO on a simulator on small instances. |

### B2. Questions on the Maritime Domain

| Likely question | How to answer |
|---|---|
| ⭐ "How is this different from ZeroNorth or Wärtsilä?" | They optimize individual voyages (routing, speed, hull). We decide *which ships and which fuels* to deploy across the fleet, with lifecycle emissions and Indian fuel-availability constraints. Different decision level. |
| ⭐ "Where does your data come from?" | Name specific sources: EU MRV public fuel data, ERA5 weather, synthetic data calibrated to real vessel specs. If you have Mormugao data, mention the scope. Never be vague here. |
| "How accurate is your fuel prediction?" | Give MAPE on *unseen vessels*, not just on the test set. Mention your leakage-safe split. |
| ⭐ "Isn't LNG a clean fuel?" | Only partly. Methane slip can erase much of its climate benefit on a lifecycle basis. Our model accounts for it. This answer shows domain depth. |
| "Is shore power actually green in India?" | It removes local pollution at the port immediately. The CO₂ benefit depends on the grid mix, which our model uses, and it improves as India adds renewables. |
| "What about ships with no sensors?" | Physics-informed model with vessel-type baselines, which works from basic specs and noon reports. |
| "What if the IMO framework isn't adopted?" | Policy is a configurable scenario. The tool also supports Indian targets (Harit Sagar) and EU rules, which apply regardless. |

### B3. Questions on Adoption & Feasibility

| Likely question | How to answer |
|---|---|
| ⭐ "Who would actually use this?" | Port authorities under the Green Tug Transition Programme, coastal operators, and inland waterway operators. Name one concrete first user, e.g., a Mormugao tug-fleet transition case. |
| "How long to deploy?" | Give a phased timeline: pilot on one port's tug fleet, then coastal fleet, then multi-port. |
| "What's the business or sustainability model?" | Licence to ports or government, on-prem deployment, annual support. Mention it can plug into green-ship incentive schemes like Mormugao's Harit Shrey. |
| "What if the optimizer recommends something operationally impossible?" | Hard constraints block impossible outputs. Humans stay in the loop, and every recommendation is explained. |
| "How does this scale to 1,000 vessels?" | Show your scalability curve (B-06). |

### B4. Things That Go Wrong on Pitch Day

| Situation | Mitigation |
|---|---|
| ⭐ Live demo crashes | Keep a recorded video of the full demo, plus screenshots in the deck. Switch without apologising at length. |
| Internet fails | Run the demo locally. Don't depend on any cloud API during the pitch. |
| Optimization takes too long live | Use a pre-computed scenario, and run only a small instance live. |
| Time runs out | Rehearse to finish 30 seconds early. Know which slide to skip. |
| A judge interrupts with a hard question mid-flow | Answer briefly, then return to the flow. Designate one teammate per topic (quantum, maritime, software). |
| Judge is non-technical | Lead with impact: fuel saved, emissions cut, rupees saved. Keep algorithm detail for follow-up questions. |
| Judge is a quantum expert | Be precise with terms. Don't say "quantum speedup." Say "quantum-inspired search" and show benchmarks. |
| Overclaiming gets caught | Never quote a number you can't back up. "Up to 40% savings" with no source destroys credibility. |
| A teammate is absent | Every member should be able to present every slide. |

---

## Part C — SIH PPT Perspective

The SIH 2026 idea presentation format allows a **maximum of six slides**, submitted as a PDF. Always check the latest official template from your institute's SPOC or the SIH portal, because the exact format can change between stages.

### C1. Slide-by-Slide Checklist

| Slide | Must include | Common mistake |
|---|---|---|
| **1. Title** | PS ID (SIH26138), PS title exactly as given, theme, category (Software), team name and ID, organization (Egreen Quanta) | Wrong PS ID, or a paraphrased title |
| **2. Idea / Proposed Solution** | One-line USP; the problem in your own words; how the solution addresses it; what's unique (fleet-level + fuel transition + India-specific + quantum-ready) | Copying the PS text; no differentiation from existing tools |
| **3. Technical Approach** | Architecture diagram (data → prediction → optimizer → DSS); tech stack; the Q-bit encoding diagram; math formulation in brief | Walls of text; a tech-stack logo collage with no flow |
| **4. Feasibility & Viability** | Data sources; risks (data scarcity, quantum hype, fuel-data uncertainty) with a mitigation for each; early benchmark results if available | Listing risks with no mitigations; claiming zero risks |
| **5. Impact & Benefits** | Quantified impact (fuel %, CO₂ tonnes, ₹ saved) with the scenario behind each; alignment with GTTP, Harit Sagar, Maritime India Vision 2030 | Vague claims like "will help the environment" |
| **6. Research & References** | Key papers, regulations, data sources, competitor tools; team details if the template requires | Missing references; broken or fake links |

### C2. PPT Edge Cases & Quality Checks

| ID | Check | Why it matters |
|---|---|---|
| PPT-01 ⭐ | Six slides maximum, PDF format | Going over can mean disqualification or the extra slides being ignored. |
| PPT-02 ⭐ | Every number on a slide has a source or a stated scenario | Unsourced numbers are the fastest way to lose credibility. |
| PPT-03 | Diagrams readable when printed or viewed small | Evaluators skim quickly. Tiny text in diagrams is invisible. |
| PPT-04 | Consistent terminology ("quantum-inspired" everywhere, never just "quantum") | Inconsistency signals confusion or overclaiming. |
| PPT-05 | The deck works without a presenter | At idea-submission stage, judges read it alone. Each slide must stand on its own. |
| PPT-06 | File size within the portal limit; fonts embedded in the PDF | Missing fonts can break layouts on the reviewer's machine. |
| PPT-07 | File named per the portal's naming rule | Misnamed files can get lost or rejected. |
| PPT-08 | Competitor comparison present (even as a small table) | Without it, the idea looks like "just another version of an existing tool." |
| PPT-09 | Limitations acknowledged somewhere | Shows maturity. Sponsor judges respect honesty over hype. |
| PPT-10 | Spell-check, especially names: Mormugao, Deendayal, Paradip, Chidambaranar, Harit Sagar | Misspelt place or scheme names look careless to government judges. |
| PPT-11 | No copyrighted images or company logos used without reason | Use your own diagrams. Competitor logos only in a comparison context, if at all. |
| PPT-12 | Colour choices readable in greyscale | Some evaluators print decks. |

---

## Part D — Final Pre-Submission Checklist

**Code**
- [ ] Leakage-safe train/test split (P-01)
- [ ] Speed-fuel monotonicity test passes (P-02)
- [ ] Energy-based fuel conversion across fuel types (P-08)
- [ ] Infeasibility handled with explanations (M-01, M-05)
- [ ] Q-bit decoder always yields valid solutions (Q-01)
- [ ] Diversity tracked; no premature convergence (Q-03)
- [ ] Methane slip, N₂O, and grey vs green factors included (F-01 to F-03)
- [ ] Fuel availability per port enforced (F-06)
- [ ] 30-run benchmark with fair budgets and statistics (B-01, B-02)

**Demo**
- [ ] Runs fully offline
- [ ] Pre-computed large scenario ready
- [ ] Backup demo video recorded
- [ ] Mormugao or GTTP tug-fleet case study prepared

**Pitch**
- [ ] Every teammate can answer the top ⭐ questions in Part B
- [ ] One owner each for quantum, maritime, and software questions
- [ ] Rehearsed to finish early

**PPT**
- [ ] Six slides or fewer, PDF, correct file name
- [ ] Every number sourced
- [ ] Competitor comparison included
- [ ] Limitations acknowledged
