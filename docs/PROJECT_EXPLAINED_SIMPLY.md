# Green Fleet Optimizer — Explained Simply

**SIH 2026 · Problem Statement ID: SIH26138 · Organization: Egreen Quanta**
**Title:** Quantum-Inspired Fuel Consumption Prediction and Green Fleet Optimization

This document explains the whole idea in plain, easy words. No heavy jargon. Where a technical word is needed, it's explained right there.

---

## 1. The Problem (In Simple Words)

Ships carry most of the world's goods. India moves about 95% of its trade (by volume) through its ports.

Ships run on fuel, mostly heavy diesel-type oil. This creates two big problems:

1. **Fuel is expensive.** It's one of the biggest costs of running a ship.
2. **Fuel causes pollution.** Burning it releases greenhouse gases like CO₂, which cause climate change.

Governments and world bodies (like the IMO, the United Nations body for shipping) are now pushing ship owners to cut pollution. India has its own targets too, through the **Harit Sagar** green port guidelines.

There are cleaner fuels available: **LNG, methanol, hydrogen, and ammonia**. Ports can also give ships **shore power**, meaning a docked ship plugs into the electricity grid instead of running its engines.

**But here's the hard part.** A shipping company or port has to answer many questions at the same time:

- Which ships should we use?
- How big should they be?
- How fast should they go?
- Which fuel should each ship use?
- Can we still deliver all the cargo on time?
- Are we following the pollution rules?

Every answer affects the others. Going slower saves fuel but takes longer, so you may need more ships. A clean fuel may be available at one port but not another. Hydrogen is very clean but takes so much tank space that it only works for short trips.

With just 20 ships, 4 fuel options and 5 speed levels, there are already more combinations than anyone can check by hand. Normal planning tools struggle with this.

**The problem statement asks us to build software that:**

1. **Predicts** how much fuel a ship will use in different conditions (speed, cargo weight, weather, ship type).
2. **Finds the best plan** for the whole fleet: which ships, which fuels, which speeds, so that fuel, cost and pollution are as low as possible, while all cargo still gets delivered on time.
3. Uses **"quantum-inspired" algorithms** to do this (explained in Section 3).
4. **Proves** it works better by comparing it against normal methods.

---

## 2. The Overall Scenario (With Examples)

> **Note:** The numbers in these examples are made up to explain the idea simply. They are not real data from any company or port.

### Example 1: A coastal shipping company (the speed trade-off)

A company runs 5 cargo ships between Mumbai and Kochi. Each ship normally sails at 14 knots (a knot is about 1.85 km/h).

There's a simple rule in shipping: **fuel use per day goes up roughly with the cube of speed.** So a small slowdown saves a lot of fuel.

If the ships slow down from 14 to 12 knots:

- Fuel burned **per day** drops by about 37%.
- The trip takes longer, so fuel **per trip** drops by about 27%. Still a big saving.
- **But** each trip now takes about 17% longer.

Now the company has a new problem. With slower trips, 5 ships may not carry all the monthly cargo on time. Should they add a sixth ship? That ship also burns fuel and costs money.

So the real question isn't "should we go slower?" It's: **"What is the best speed for each ship, and how many ships do we need, so that all cargo arrives on time with the least fuel and cost?"**

Now add clean fuels. Maybe two of the ships can run on methanol. But methanol holds only about half the energy of diesel per kg, so those ships need bigger fuel tanks, which leaves less room for cargo. And is methanol even available for refuelling at Kochi?

Every choice pulls on every other choice. That's why this needs smart software.

### Example 2: A port's tug boat fleet (a real government program)

Tug boats are small, powerful boats that push and pull big ships into the harbour. They run their engines almost all day.

The Government of India has a program called the **Green Tug Transition Programme (GTTP)**. It aims to replace diesel tugs with electric, hybrid, or clean-fuel tugs.

Imagine a port with 10 old diesel tugs. The port authority must decide:

- How many tugs to replace, and when (money is limited each year)?
- Which type: electric, hybrid, methanol, or hydrogen?
- Which tugs to keep for busy hours, and which for quiet hours?
- Where will they charge or refuel?
- Is the electricity itself clean? If it comes mostly from coal power, an electric tug is less green than it looks.

Our software can take all of this in and show the port authority the best options, like:

- "Plan A: cheapest, cuts pollution by 25%."
- "Plan B: costs 15% more, cuts pollution by 60%."
- "Plan C: middle ground."

The decision-maker then picks the plan that fits their budget and goals. This set of best trade-off options is called a **Pareto front**: a list of plans where you can't improve one thing (like cost) without making another thing (like pollution) worse.

---

## 3. Our Solution

### In one line

> **A smart planning tool that tells ship and port operators which ships to use, which fuels to use, and how fast to sail, to save the most fuel, money and pollution, while still delivering all cargo on time.**

### What makes it different from existing tools

Companies like ZeroNorth and Wärtsilä already make tools that help **one ship on one trip** (best route, best speed). Those are good tools.

We go one level higher. We help plan the **whole fleet**, including the switch to clean fuels over the coming years. And we build it for **Indian conditions**: Indian ports, Indian fuel availability, Indian electricity, Indian rules like Harit Sagar.

### The four parts of our solution

**Part 1: Fuel Predictor**
It learns how much fuel a ship uses from past data: speed, cargo weight, weather, and ship type. We combine basic ship physics (like the speed-cube rule) with machine learning. This way it gives sensible answers even for ships with very little data.

**Part 2: Fleet Planner (the optimizer)**
This is the "brain." It searches through millions of possible plans and finds the best ones. It follows all the rules: deliver all cargo, stay on schedule, stay within pollution limits, only use fuels that are actually available.

**Part 3: The "Quantum-Inspired" Search Method**
This is what the problem statement specifically asks for. Here's what it means in simple words:

- **It is NOT a quantum computer.** It runs on a normal laptop or server.
- It borrows ideas from quantum physics to search smarter.
- In a normal search, each option is either "yes" or "no" at each step.
- In our search, each option starts as a **probability**, like "60% chance this ship uses methanol." The search keeps many possibilities open at once, the way a quantum bit (Q-bit) can be in a mix of states.
- As the search learns which plans are good, it slowly **rotates** these probabilities toward better choices (this is called a "rotation gate" update).
- This helps it explore many options early and avoid getting stuck on a "good but not best" answer too soon.

We also write the problem in a special format (called **QUBO**) so it can run on real quantum computers in the future, when they become practical. This makes the tool **quantum-ready**.

**Part 4: Dashboard (Decision Support System)**
A simple website where a port officer or company manager can:
- Enter or upload their fleet and cargo details.
- Try "what if" scenarios: "What if diesel price goes up 50%?" "What if green hydrogen becomes available at Paradip port in 2030?"
- See the best plans on charts and maps.
- Download a report.

### The honest part

The tool counts pollution over the **full life of the fuel**: making it, transporting it, and burning it. This is called **lifecycle** or **well-to-wake** emissions. This matters because:

- "Grey" ammonia (made from natural gas) can pollute *more* than diesel overall.
- "Green" ammonia (made using solar or wind power) is nearly clean.
- LNG engines leak some unburnt methane, which is a very strong greenhouse gas.

Many tools only count what comes out of the ship's chimney. We count everything.

---

## 4. How We Will Build It (Implementation)

### The flow in simple steps

```
 Ship data + Weather data + Fuel data
              │
              ▼
   ┌─────────────────────┐
   │  1. Clean the data   │
   └─────────────────────┘
              │
              ▼
   ┌─────────────────────┐
   │  2. Fuel Predictor   │  → "Ship X at 12 knots, loaded, in rough sea
   └─────────────────────┘      will burn about Y tonnes per day"
              │
              ▼
   ┌─────────────────────┐
   │  3. Fleet Planner    │  → Tries millions of plans using the
   │  (quantum-inspired)  │     quantum-inspired search
   └─────────────────────┘
              │
              ▼
   ┌─────────────────────┐
   │  4. Dashboard        │  → Shows best plans, charts, reports
   └─────────────────────┘
```

### Step-by-step plan

**Step 1: Collect and clean data**
- Public ship fuel data (the EU publishes yearly fuel and CO₂ data per ship).
- Weather data (ERA5, a free global weather dataset).
- Fuel facts: energy content and emissions of each fuel.
- Where needed, create realistic "simulated" data based on real ship specifications.
- Fix mistakes in the data: wrong units, missing values, impossible readings.

**Step 2: Build the Fuel Predictor**
- Start with a simple physics formula (fuel rises with speed cubed).
- Add a machine learning model (like XGBoost) to learn what physics misses: weather, hull condition, cargo weight.
- Use a quantum-inspired method (QPSO) to tune the model's settings.
- Test it on ships the model has **never seen before**, so we know it really works.

**Step 3: Write the problem as maths**
- **Choices** (decision variables): which ships, how big, what speed, which fuel.
- **Goals** (objectives): lowest fuel, lowest pollution, lowest cost.
- **Rules** (constraints): deliver all cargo, stay on schedule, stay under pollution limits, use only fuels available at each port.

**Step 4: Build the quantum-inspired Fleet Planner**
- Represent each plan using Q-bits (probabilities).
- Use rotation updates to steer toward better plans.
- Make sure every plan it gives out actually follows all the rules.
- Output a set of best trade-off plans (the Pareto front).

**Step 5: Compare with normal methods (benchmarking)**
- Run the same problems with normal methods: a standard genetic algorithm, NSGA-II, particle swarm, and exact solvers (for small problems).
- Compare: accuracy, speed, quality of answers, and how well each handles big problems.
- Run each method 30 times, because these methods involve randomness and one lucky run proves nothing.
- Report results honestly, including where we don't win.

**Step 6: Build the Dashboard**
- Simple forms to enter fleet details.
- "What if" scenario buttons.
- Charts: fuel, cost, pollution, fleet plan on a map.
- One-click report download.

**Step 7: Demo with a real-looking case**
- A large simulated fleet scenario.
- A port tug-fleet case study (like the GTTP example above).
- Ideally built around **Mormugao Port**, since our problem statement organization (Egreen Quanta) already has an agreement with Mormugao Port for quantum-based maritime projects.

### Suggested tools (tech stack)

| Part | Tools |
|---|---|
| Data handling | Python, Pandas, NumPy |
| Fuel Predictor | Scikit-learn, XGBoost (optionally PyTorch) |
| Quantum-inspired engine | Our own code in Python + NumPy |
| Comparison methods | pymoo (for NSGA-II), Google OR-Tools (exact solver) |
| Backend (server) | FastAPI |
| Dashboard (frontend) | React or Streamlit, with Plotly for charts |
| Database | PostgreSQL |
| Packaging | Docker (so it runs the same on any machine, even offline) |

---

## 5. Challenges We May Face

### Data challenges
- **Not enough real data.** Detailed ship fuel data is often private. Indian coastal ships and tugs may have very little recorded data.
  → *Plan:* Use public data, physics-based models, and realistic simulated data. Clearly say which is which.
- **Messy data.** Wrong units, missing values, sensor errors.
  → *Plan:* Strong data cleaning and checks in Step 1.
- **Very little data on new fuels.** Few ships run on ammonia or hydrogen today.
  → *Plan:* Use published engineering values for these fuels instead of learning them from data.

### Technical challenges
- **The quantum-inspired method may not always beat normal methods.** This is a real possibility.
  → *Plan:* Be honest. Show where it helps and where it doesn't. The quantum-ready (QUBO) design is still valuable for the future.
- **The search can get stuck too early** on a "good enough" answer.
  → *Plan:* Track how varied the options are during the search, and tune the rotation settings carefully.
- **Mixing different kinds of choices.** Ship and fuel choices are "pick one" options, but speed is a smooth number. Handling both together is tricky.
  → *Plan:* Use a hybrid design: Q-bits for choices, a separate method for speed.
- **Big problems take a long time to solve.**
  → *Plan:* Show a demo on a pre-solved big scenario and run small ones live.

### Domain (shipping) challenges
- **Pollution numbers depend on how fuel is made.** Grey vs green fuels give very different results.
  → *Plan:* Keep emission values in an easy-to-update settings file, based on official IMO and Indian sources.
- **Rules keep changing.** The IMO's global net-zero rules were delayed in 2025 and are still being negotiated.
  → *Plan:* Make rules a switch in the tool ("rule adopted" / "not adopted"), not something hard-coded.
- **Shore power is only as clean as the electricity grid.** India's grid still uses a lot of coal.
  → *Plan:* Use official Indian grid emission data and show this honestly.

### Hackathon challenges
- **Limited time.** Building everything perfectly in a hackathon is hard.
  → *Plan:* Build a working core first (predictor + planner + simple dashboard), then add features.
- **Explaining "quantum-inspired" to judges.** Some judges are quantum experts, some aren't.
  → *Plan:* Simple explanation for everyone, precise details ready for experts. Never claim it's a real quantum computer.
- **Live demo failure.**
  → *Plan:* Run everything offline, and keep a backup demo video.

---

## 6. Summary in 5 Lines

1. **Problem:** Ships burn a lot of costly, polluting fuel, and planning a cleaner fleet is too complex for normal tools.
2. **Scenario:** Operators must pick ships, speeds, and fuels together, while delivering cargo on time and meeting pollution rules.
3. **Solution:** A fuel predictor plus a quantum-inspired fleet planner, with a simple dashboard, built for Indian ports.
4. **Build:** Clean data → predict fuel → plan the fleet → compare with normal methods → show it all on a dashboard.
5. **Challenges:** Limited data, honest benchmarking, changing rules, and explaining "quantum-inspired" clearly.
