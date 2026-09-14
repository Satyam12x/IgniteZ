# Phase 2 findings — well-to-wake emissions engine

**SIH26138 · Green Fleet Optimizer**
Every number here is reproducible with `uv run python scripts/emissions_demo.py`
and traces to `config/fuels.yaml` and `config/ports.yaml`.

---

## 1. The result that decides the competitive argument

A voyage-optimisation tool counts what leaves the funnel. Here is the same set of
fuel choices scored that way, and scored over the full lifecycle. Per 1 TJ of
**shaft work**, AR6 GWP100:

| fuel / pathway | stack-only tCO₂e | lifecycle tCO₂e | stack says | truth |
|---|---|---|---|---|
| Marine gas oil | 75.1 | 90.7 | 0% | 0% |
| Heavy fuel oil | 77.5 | 92.2 | +3% | +2% |
| LNG (fossil) | 57.3 | 87.1 | **−24%** | **−4%** |
| **Methanol (grey)** | 66.3 | 100.2 | **−12%** | **+10%** ⚠ |
| E-methanol | 0.0 | 8.6 | −100% | −91% |
| **Ammonia (grey)** | **0.0** | **126.5** | **−100%** | **+40%** ⚠ |
| Ammonia (blue) | 0.0 | 44.7 | −100% | −51% |
| Ammonia (green) | 0.0 | 18.6 | −100% | −79% |
| **Hydrogen (grey)** | **0.0** | **99.1** | **−100%** | **+9%** ⚠ |
| Hydrogen (green) | 0.0 | 7.2 | −100% | −92% |

**Three fuels change sign.** Grey ammonia, grey hydrogen and grey methanol are all
scored as improvements by a stack-only model — grey ammonia and grey hydrogen as
*perfect*, at −100% — while each actually **increases** lifecycle emissions.

A port authority that procured a fleet on the stack-only column would have spent
public money increasing emissions by up to 40% and reported it as reaching zero.

*(Per-TJ figures for hydrogen reflect a fuel-cell conversion efficiency of 0.50 against a
diesel engine's 0.45, so hydrogen is compared on shaft work like every other option -
see §3. All other fuels share the diesel-engine efficiency and their ratios to diesel are
unchanged by the shaft-energy basis.)*
That is the single clearest statement of why this tool is needed, and it is
assertable because the test suite asserts it.

---

## 2. Methane slip, measured rather than assumed

Our LNG slip figure is not a regulatory default. It is the median CH₄-to-fuel ratio
across **745 LNG carrier ship-years** in EU MRV 2024–2025 — data that only became
available with the EU ETS extension, and which we found while building the Phase 1
ingest.

| assumption | slip | tCO₂e / TJ shaft | vs diesel |
|---|---|---|---|
| FuelEU default (Otto medium-speed) | 3.10% | 95.7 | **+5.5%** |
| **Measured fleet median** | **1.72%** | **87.1** | **−3.9%** |
| Best engine (diesel slow-speed) | 0.20% | 77.7 | −14.3% |

The regulatory default and the measured fleet **disagree about whether LNG helps at
all**. Using the worst-case default for a vessel whose engine is unknown would rule
LNG out on evidence the real fleet does not support; using the best case would
oversell it. We use the measured median when the engine is unknown and the
engine-specific regulatory value when it is known, and the config says so.

The measured spread is wide — p25 0.63% to p95 2.97%, a factor of 4.7 — so the
distribution is carried in config, not just the point value.

### Cross-checks that fell out of the same data

| finding | detail |
|---|---|
| **EU's GWP set recovered exactly** | Solving `CO₂e − CO₂ = a·CH₄ + b·N₂O` over 28,125 filings gives **a = 28.00, b = 265.00** with a maximum relative residual of **1.6×10⁻¹⁴**. EMSA uses AR5, not AR6. Available as the `eu_mrv_ar5` GWP set so our figures can reconcile with official ones. |
| **Two of our factors were wrong** | Measured CH₄ for conventional fuels is 0.050 g/kg (we had 0.012 — 4× low) and N₂O is 0.180 g/kg (we had 0.072 — 2.5× low). The CH₄ value reproduces the IMO Fourth GHG Study default exactly, which independently validates the extraction. Both corrected. |

---

## 3. A modelling bug that would have inverted the GTTP recommendation

The first version of the tug comparison showed battery-electric at **+146% worse
than diesel**. That was wrong, and the cause is worth recording because it is easy
to miss.

Emissions were being compared per unit of **input** energy. But a marine diesel
converts only ~45% of fuel energy into shaft work, while an electric drivetrain
converts ~92%. Comparing them on input energy penalises the battery by roughly a
factor of two.

The engine now works in **shaft energy** throughout and converts per fuel. The
Green Tug Transition Programme is the flagship Indian programme this project
targets; shipping a tool that told a port not to electrify, for a units reason,
would have been fatal to the pitch.

---

## 4. Electrification in India: the honest answer

With the physics corrected, the result is still not the comfortable one — and this
is the finding most worth presenting to a government audience, because it is
specific, actionable, and clearly not sales copy.

| year | grid (tCO₂/MWh delivered) | electric tCO₂e / TJ shaft | vs diesel |
|---|---|---|---|
| 2026 | 0.831 | 278.9 | **+38%** |
| 2030 | 0.723 | 242.5 | **+20%** |
| 2033 | 0.636 | 213.4 | +6% |
| **2035** | 0.578 | 194.0 | **−4%** ← crossover |
| 2040 | 0.434 | 145.5 | −28% |
| 2047 | 0.241 | 80.8 | −60% |

**A battery-electric tug in India does not beat diesel on CO₂ until about 2035.**
Against a large marine diesel at 45% efficiency, the current coal-heavy grid is not
clean enough.

That is only half the picture, and reporting only that half would produce the wrong
policy. The other half is what a port city breathes:

| option (harbour tug, 12 TJ shaft work/yr) | tCO₂e | NOx kg | SOx kg | PM kg |
|---|---|---|---|---|
| Diesel (today) | 2,418 | **35,597** | **1,249** | **187** |
| LNG | 2,323 | 5,556 | 0 | 11 |
| E-methanol | 229 | 33,585 | 50 | 72 |
| Green hydrogen (fuel cell) | 192 | 0 | 0 | 0 |
| **Electric, 2030 grid** | 2,910 (+20%) | **0** | **0** | **0** |
| **Electric, 2035 grid** | 2,328 (−4%) | **0** | **0** | **0** |

**The recommendation this supports:** electrify harbour tugs now for air quality —
each tug removes **35.6 tonnes of NOx, 1.25 tonnes of SOx and 187 kg of
particulates** from the port city every year, on day one — and the carbon benefit
arrives around 2035 as the grid decarbonises. Local pollutants are tracked
separately and never folded into CO₂e, because adding an air-quality pollutant into
a greenhouse-gas total is a category error.

Note also that **e-methanol cuts CO₂ by 91% but barely touches NOx**: its lower NOx
per kilogram is offset by needing 2.1× the fuel mass. A tool that reported only
carbon would present it as a clean-air solution. It is not.

---

## 5. India-specific constraints that foreign tools do not model

`config/ports.yaml` encodes eight major Indian ports with year-indexed fuel
bunkering, shore-power commissioning, the CEA grid factor and the Harit Sagar
targets.

**F-06, fuel availability.** A fuel is usable on a route only if *every* port on it
can supply it that year. `"planned"` deliberately does not count as available —
planning a fleet around announced-but-unbuilt infrastructure is the failure this
prevents. On a Mumbai → Mormugao → Cochin route:

| year | fuels usable across the whole route |
|---|---|
| 2024 | hfo, mgo |
| 2030 | electricity, hfo, mgo |
| 2047 | electricity, hfo, lng, methanol, mgo |

Refusals explain themselves: *"hydrogen is not bunkerable in 2030 at: mumbai (none),
mormugao (none), cochin (none)"*.

**F-04/F-05, tank volume.** Installed energy density — including insulation and
containment, not neat fuel — governs both range and cargo. Retrofitting a diesel
tank:

| fuel | range from an MGO-sized tank | cargo lost on a purpose-built 5,000 nmi tank |
|---|---|---|
| MGO | 5,000 nmi (100%) | — |
| Methanol | 1,803 nmi (36%) | 693 t |
| LNG | 1,624 nmi (32%) | 813 t |
| Ammonia | 1,284 nmi (26%) | 1,132 t |
| **Hydrogen** | **374 nmi (7%)** | **4,842 t** |

Hydrogen reaching 7% of diesel range on a retrofit is exactly why it is a harbour
tug and ferry fuel, not a long-haul one — F-04, quantified rather than asserted.
Retrofit and newbuild are separate questions and the API makes the caller say which.

**S-06, Harit Sagar.** Emissions per tonne of cargo against the −30%/2030 and
−70%/2047 targets, with the intermediate glide path interpolated so a port knows its
2027 position, not just its legislated milestones.

---

## 6. What is solid and what is not

Being explicit about this is itself a deliverable — the sponsor is a technical
organisation and will probe.

| input | confidence | basis |
|---|---|---|
| LHV, tank-to-wake CO₂ (C_f) | **high** | IMO MEPC.308(73), verbatim |
| CH₄ and N₂O for conventional fuels | **high** | Median of 26,621 measured ship-years |
| LNG methane slip | **high** | Median of 745 measured LNG carrier ship-years |
| EU regulatory GWP set | **high** | Recovered exactly from 28,125 filings |
| Well-to-tank factors | medium | FuelEU Maritime Annex II defaults, to re-verify |
| Engine conversion efficiencies | medium | Published marine engine ranges |
| Ammonia N₂O | **provisional** | No ammonia fleet exists; literature range is wide |
| Tank system factors | **provisional** | Engineering estimates |
| **Port fuel availability by year** | **provisional** | **Structure is real; specific years are placeholders pending port authority confirmation** |
| Grid factor, historical | medium | CEA CO₂ Baseline Database |
| Grid factor, projected | provisional | Glide consistent with the 500 GW non-fossil target |

Every `EmissionResult` reports the **weakest** confidence among the factors that
produced it, so a provisional input cannot hide inside a confident-looking total.

The port availability table is the item to fix before any published claim. The data
model is correct and the numbers are placeholders; confirming them with Mormugao
Port Authority would be the natural first ask of a pilot engagement.

---

## 7. Against the competition

| capability | ZeroNorth / Wärtsilä / DeepSea | This tool |
|---|---|---|
| Decision level | one voyage, one vessel | fleet mix, fuel and speed over years |
| Emissions basis | tank-to-wake CO₂ | **well-to-wake CO₂e incl. CH₄, N₂O, pilot fuel** |
| Methane slip | typically a default, if modelled | **measured from 745 ship-years** |
| Grey vs green pathways | not a fleet decision they make | **modelled; grey ammonia comes out worse than diesel** |
| Local air pollutants | not typically reported | **NOx, SOx, PM tracked separately** |
| Indian port fuel availability | not modelled | **year-indexed per port** |
| Indian grid carbon | not modelled | **CEA factor with T&D losses and a decarbonisation path** |
| Harit Sagar KPIs | not modelled | **glide path and on/off-track status** |
| Data residency | foreign SaaS | on-premises, offline-capable |

The honest framing for a judge: those are strong products doing a different job
well. They optimise a voyage. Nothing in them decides **which vessels to buy, which
fuels to commit to, and when** — which is the decision a port authority under GTTP
and Harit Sagar actually faces.

---

## 8. Test coverage

99 tests in `tests/unit/test_emissions.py`, mapped to the register: F-01 through
F-10, P-09, S-02, S-04, S-06. **324 tests pass across the project.**

The tests assert the claims rather than merely exercising the code — that grey
ammonia comes out worse than diesel, that methane slip erases most of LNG's
advantage, that green ammonia is not zero, that a fuel with no bunkering cannot be
assigned, that shore power needs both sides fitted, and that electric trades carbon
for air quality before the crossover.

## 9. Next

1. **Phase 3 — the fleet optimizer.** Everything is now in place for it: an energy
   model with correct speed sensitivity, a lifecycle objective, hard infrastructure
   constraints, and a tuned QIEA engine verified against exhaustive optima.
2. Confirm port fuel availability with Mormugao Port Authority.
3. Duty-cycle treatment for offshore and ro-pax (carried over from Phase 1).
