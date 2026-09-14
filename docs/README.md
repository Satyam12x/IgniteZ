# SIH 2026 — Problem Statement Analysis

**Problem Statement 2 (Egreen Quanta):** Quantum-Inspired Fuel Consumption Prediction and Green Fleet Optimization

This README summarises the key requirements, the competitive landscape, our USP, and why a government stakeholder would adopt our solution.

---

## 1. What We Must Follow (Non-Negotiables)

The evaluation will be scored against the 5-row deliverables table in the PS, so build to it.

- **Two engines, both "quantum-inspired."** We need a fuel prediction model and a fleet optimizer, and both need a quantum-inspired element. "Quantum-inspired" means classical code that borrows quantum ideas, not real quantum hardware. The PS explicitly asks for an *encoding scheme for fleet decisions* and *quantum update mechanisms*. So use Q-bit (probability-amplitude) encoding plus rotation-gate updates, as in QIEA or quantum GA, or use QPSO.
- **Formal math model.**
  - Decision variables: vessel mix, capacity, speed, fuel type.
  - Objectives: minimum fuel, emissions, and cost.
  - Constraints: cargo demand, schedule, emission caps.
  - Output a **Pareto front**, not one weighted answer.
- **Lifecycle (well-to-wake) emissions, not just exhaust.** Grey ammonia can be worse than diesel, while green ammonia is near-zero. Ignoring this makes the alt-fuel analysis wrong. Cover LNG, methanol, hydrogen, ammonia, and shore power.
- **Benchmarking is mandatory, and most teams skip it.**
  - Prediction baselines: XGBoost, physics-based models.
  - Optimization baselines: NSGA-II, PSO, MILP (OR-Tools or Gurobi).
  - Metrics: accuracy, convergence speed, hypervolume, scalability.
  - Report honestly, including cases where we lose. The sponsor is a quantum company and will probe hype.
- **A working decision support system.** It needs a UI or API, scenario simulation, visualisation of fleet allocation and emissions, and auto-generated reports.
- **A large-scale demo plus an implementation guide.** Use real or simulated data. The public EU MRV per-ship fuel data and ERA5 weather data are good free sources.
- **Physics-informed prediction.** Fuel scales roughly with speed cubed. A hybrid physics + ML model tuned by QPSO will beat pure black-box ML on sparse data.

---

## 2. Existing Solutions and What They Lack

| Player | What it does | Gap |
|---|---|---|
| ZeroNorth, Wärtsilä FOS, DeepSea | Voyage and vessel-level routing, speed, and hull performance | Operational only. Doesn't decide *which ships and which fuels* to deploy. Foreign SaaS that needs rich sensor data. |
| DNV Alternative Fuels Insight | Market data on fuel uptake and orderbooks | Information, not an optimizer for *your* fleet |
| Academic quantum GA papers | Speed optimization with quantum GA | Research code, single problem, no product |
| Quantum port pilots | Berth and crane scheduling | Port-side, not fleet and fuel transition |

**Existing tools**
- ZeroNorth's fuel model combines naval architecture, machine learning and historical performance, and its voyage optimizer claims 5–10% fuel savings per voyage.
- Wärtsilä FOS integrates navigational, technical and operational vessel data.
- These are strong tools, but they optimize one voyage at a time.

**Academic work**
- Quantum genetic algorithms for ship speed optimization exist, but most studies focus on a single policy rather than combining several.

**Quantum pilots**
- Current quantum maritime work targets port subproblems, such as combined crane, yard, and gate sequencing in a hybrid quantum-classical workflow.

**The white space:** No one does integrated *predict, then optimize fleet mix, speed, and fuel transition* with lifecycle emissions and Indian infrastructure constraints.

---

## 3. Our USP

> **India's first sovereign, quantum-ready planner that decides which vessels, which fuels, and which speeds, not just which route.**

What backs that line:

- **Strategic + operational in one loop.** Fleet composition and fuel-transition planning are coupled with speed decisions.
- **India-specific constraints.** Fuel availability is tied to real hubs: MNRE has recognised Deendayal, Paradip, and V.O. Chidambaranar ports as Green Hydrogen Hubs. The model also covers shore-power availability, INR costs, and Harit Sagar KPIs.
- **Quantum-ready.** Formulate the discrete core as a QUBO so the same model can later run on annealers or National Quantum Mission hardware.
- **Works with sparse data.** The physics-informed model handles coastal and inland vessels that lack sensors, which foreign tools depend on.
- **Explainable, on-prem, data stays in India.**
- **Future-proof policy engine.** Keep rules configurable, because global rules are in flux. The IMO Net-Zero Framework adoption was adjourned in October 2025 and is set to be reconsidered in late 2026.

---

## 4. Why the Government Would Pick Ours

### Target a real government program: GTTP
The Green Tug Transition Programme replaces diesel harbour tugs with electric, hybrid, and alternative-fuel tugs. Four Major Ports (Deendayal, JNPA, Visakhapatnam, V.O. Chidambaranar) have already placed work orders for electric tugs. Each port faces an actual fleet-mix, fuel-choice, and duty-scheduling problem, which is exactly our optimizer. The same logic extends to inland waterways under Harit Nauka, which targets a full transition to green inland vessels by 2047.

### Map outputs to Harit Sagar targets
Ports must cut carbon emissions per tonne of cargo by 30% by 2030 and 70% by 2047. Show that number moving on our dashboard.

### Build the demo around Mormugao (biggest insider tip)
In July 2026, Egreen Quanta signed a Strategic Quantum Technology agreement with Mormugao Port Authority to jointly develop and pilot quantum-enhanced maritime logistics optimization, with the port sharing non-sensitive operational datasets. Mormugao also runs **"Harit Shrey,"** India's first green ship incentive scheme, which discounts port charges based on ships' environmental performance.

A Mormugao-based case study, plugging our emission scores into Harit Shrey, fits straight into the sponsor's live pilot. That makes our solution easy for them to adopt.

---

## 5. Final Advice

Don't oversell "quantum." Win on honest benchmarks, Pareto trade-off visuals, and a GTTP/Mormugao use case the jury can picture deploying next month.

---

## Sources

- [PIB — Mormugao Port Authority signs Strategic Quantum Technology agreement with Egreen Quanta (July 2026)](https://www.pib.gov.in/PressReleasePage.aspx?PRID=2280824&reg=1&lang=1)
- [ZeroNorth — Voyage Optimisation](https://zeronorth.com/voyage-optimisation)
- [Wärtsilä — Fleet Optimisation Solution](https://www.wartsila.com/marine/products/fleet-optimisation)
- [DNV — Alternative Fuels Insight](https://www.dnv.com/services/alternative-fuels-insight-afi/)
- [ScienceDirect — Improved quantum genetic algorithm for ship speed optimization](https://www.sciencedirect.com/science/article/abs/pii/S0959652622053884)
- [Maritime Executive — Quantum computing for port scheduling (Feb 2026)](https://maritime-executive.com/editorials/quantum-computing-can-solve-the-hardest-port-scheduling-problems)
- [IMO — Net-zero shipping talks to resume in 2026](https://www.imo.org/en/mediacentre/pressbriefings/pages/imo-net-zero-shipping-talks-to-resume-in-2026.aspx)
- [DNV — IMO MEPC 84: Revisiting the net-zero framework](https://www.dnv.com/news/2026/imo-mepc-84-revisiting-the-net-zero-framework/)
- [SAFETY4SEA — India accelerates green transition across major ports (Apr 2026)](https://safety4sea.com/india-accelerates-green-transition-across-major-ports/)
- [DD News — India charts green maritime transition (Harit Sagar, Harit Shrey, Harit Nauka)](https://ddnews.gov.in/en/india-charts-green-maritime-transition-with-ports-shipping-reforms-and-global-partnerships/)
