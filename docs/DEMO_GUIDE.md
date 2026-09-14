# Demo guide — Green Fleet Optimizer platform

For the person driving the screen. Five minutes, four views, every number real.

## Before the session

```bash
uv pip install -e ".[dev,optim,api]"
python scripts/train_model.py        # once; ~2 min; writes artifacts/model/
python -m greenfleet.api             # http://127.0.0.1:8000
```

Open the browser **before** the judges arrive. The page runs the 2047 coastal case on
load (about 4 s) so the first thing anyone sees is a finished trade-off, not a blank
form. Check the footer reads *Predictions come from a model trained on 1,00,528 real, audited
ship-years*. If a yellow banner says no trained model is loaded, run
`scripts/train_model.py` and restart.

The dashboard needs no internet: Tailwind is precompiled and Plotly is served from the
installed package. Fonts fall back to system faces offline; nothing else changes.

## The screen, in one paragraph

Four pages in the sidebar: **Plan**, **Fuels**, **Calculator**, **Ports**. *Plan* is the
whole product in one screen: choose a fleet, a year and the fuels you can buy, press
**Find plans**, and pick one of three cards — Lowest cost, Balanced, Lowest emissions.
Every setting (carbon tax, CO₂ limit, how each fuel is made, search settings, comparison
methods) is on the same card, and the evidence a judge will ask for — evaluations,
hypervolume against NSGA-II, search progress, provenance — sits under the result as
**How the search performed**. Nothing is hidden. Full usage: `docs/USER_GUIDE.pdf`.

## The five-minute script

### 1 · Plan (2 min) — the decision no existing tool makes

The page opens on a finished run: the setup card, then three plan cards.

> "Twelve coastal ships, three Indian routes, four fuels. We searched 6,000 complete
> plans and kept the ones you cannot improve without a trade-off. Three worth looking at."

Point at the cards left to right: **Lowest cost** (all heavy fuel oil, ~₹70 cr,
~29,000 t, *misses* Harit Sagar) → **Balanced** → **Lowest emissions** (all e-methanol,
~₹114 cr, ~2,600 t, *meets* the 2047 target, 91% less than diesel).

> "Same ships, same cargo. Ninety-one percent less CO₂ for sixty percent more cost —
> and every step between. The port chooses; we don't hide the trade-off behind one number."

Click **Lowest emissions**. The detail card below shows the chart with that plan ringed
and the assignments table: which ship, which route, which fuel, what speed, and the
ships kept in reserve.

Change *Fleet* to **Coastal cargo fleet · 12 ships · 2030** and press **Find plans**. Every plan is grey/brown; every card *misses* the target.

> "In 2030 methanol is only *planned* at Mormugao and Cochin. The planner will not build
> a fleet around a bunkering facility that does not exist yet. That refusal is the
> India layer."

If asked "can I enter my own fleet?": press **Edit ships & routes**, change a capacity or
delete a ship, run again. Type 9,000,000 t of cargo on a route and the tool refuses
in plain words with the exact shortfall.

Now flip **Technical details** on (bottom of the sidebar): a second row of tiles appears (18 plans · 6,000 evaluations ·
3–4 s · 2.3×10¹³ combinations · ~1.06× NSGA-II), the search-progress chart, and
**Against conventional methods**. At twelve ships this is a *tie* with NSGA-II. Say
"tie". The advantage appears at 60 ships (p = 0.004) and is in the findings.

### 2 · Fuels (1 min) — the lifecycle flip

The bar chart is sorted cleanest to dirtiest with a dotted *diesel* line. Green bars are
cleaner, red dirtier, **amber = looks clean at the funnel but isn't**. Tick **Show the
funnel-only view**: black ticks land at zero for ammonia, hydrogen and electric.

> "A funnel-only tool scores grey ammonia as a perfect fuel. Made from natural gas it is
> 40% *worse* than diesel over its lifecycle. Same molecule, opposite verdict — the
> pathway is everything. Four of these ten options flip."

The three cards below say the rest: *4 fuels look clean but are dirtier*, *LNG −4% with
measured slip (1.72% from 745 ships, not the 3.1% default)*, *Electric +20% on the 2030
grid, beats diesel from 2035*.

> "On today's grid a battery tug emits more CO₂ than diesel. Electrify anyway: the
> exhaust at the quayside goes to zero on day one — 36 tonnes of NOx per tug per year.
> The carbon win arrives with the grid."

### 3 · Calculator (1 min) — the physics is imposed, not learned

Default: bulk carrier, certificate 5.2, diesel, 14 knots. Drag the **speed slider** to
12: the big number drops and the row says *Slow down to 12.0 kn · −24% per mile*.

> "The cube law is imposed from naval architecture; the model learns only this ship's
> efficiency level from its certificate. On 6,263 ships never seen in training it is
> within about 20%, and the likely range covers the truth 80% of the time."

Change *Ship type* to **Harbour tug**: the certificate clears, the band widens and the
badge reads *Estimate — no certificate, wide range*.

> "No tug is in the training data — the law stops at 5,000 GT. The tool says so instead
> of guessing confidently."

### 4 · Ports (30 s) — the constraints the planner obeys

Flip the year 2030 → 2047: ○ *planned* becomes ◐ *limited* or ✓ *available*.

> "The structure is real; the exact years are placeholders to confirm with each port —
> the first task of the Mormugao pilot."

## Numbers to have ready

| Ask | Answer |
|---|---|
| Data | 106,922 verified EU MRV ship-years, 25,967 ships, 2018–2025, EMSA public API, version-pinned (SHA-256) |
| Served model | XGBoost + monotone constraint, QPSO-tuned; 20.3% MAPE, R² 0.59, 80.1% interval coverage on 6,263 unseen ships |
| Why not the 19.9% model | The unconstrained XGBoost is 0.4 points better but can predict *less* energy for a *worse* design certificate; we chose physical consistency |
| Optimizer | Q-bit encoding + Han-Kim rotation gate for discrete choices, QPSO for speeds, Pareto archive, repair not penalty |
| Verification | 30/30 exact on the 2-vessel case vs CP-SAT; QUBO ground state = MILP optimum |
| Benchmark | 12 vessels: tie with NSGA-II (p = 0.14). 60 vessels: wins, p = 0.004. 200 vessels: feasible plan in 54 s |
| Quantum? | No. Quantum-*inspired*, runs on a laptop. QUBO form is annealer-ready; not yet run on hardware |
| Runtime | 6,000 evaluations in 3–4 s with the trained model (per-vessel curves cached, XGBoost called once per vessel) |

## If something goes wrong

- **A yellow banner saying the service is not reachable** — the API is down. Restart `python -m greenfleet.api`.
- **"No plan could deliver all the cargo with these fuels in this year"** — a ticked fuel
  is not bunkerable on one of the routes yet. Tick diesel back or move the year forward.
  This is a correct answer; say so.
- **A run is slow** — *Stop* returns what was found so far. With Technical details on,
  lower *Generations* to 100.
- **A question you cannot answer** — "That is in `docs/FINDINGS_PHASE*.md`; I'll confirm
  the exact figure rather than guess."

## What is real and what is not

Real: every emission factor (config-tagged with source and confidence), the trained
model and its held-out accuracy, the optimizer and baselines, the Indian port structure,
CEA grid factors to 2023.

Representative, labelled as such: the coastal fleet's vessel specs (shaped like Indian
coastal shipping, not an operator's data), grid projections after 2023, per-year
bunkering availability at each port.

Not done yet: PDF report export, user accounts, a saved scenario library, Docker
packaging. Say "next phase" — do not imply they exist.
