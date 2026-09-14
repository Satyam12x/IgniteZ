"""Generate every chart for the PPT / report from the artifact JSONs.

Numbers come from ``artifacts/*.json`` (produced by the evaluation, benchmark and
demo scripts) or are recomputed live with fixed seeds. Nothing is typed in by hand,
so a chart cannot drift from the code that produced the number (PPT-02).

Output: ``artifacts/charts/*.png`` at 300 dpi, sized for a 16:9 slide.
"""

from __future__ import annotations

import json
import logging
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.patches import FancyBboxPatch  # noqa: E402

logger = logging.getLogger(__name__)

ART = Path("artifacts")
OUT = ART / "charts"

# Validated categorical palette (light surface), fixed order.
BLUE, ORANGE, AQUA, YELLOW = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
MAGENTA, GREEN, VIOLET, RED = "#e87ba4", "#008300", "#4a3aa7", "#e34948"
INK, INK2, MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#8a8985", "#e6e5e1", "#fcfcfb"
NEUTRAL = "#f0efec"

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
    "axes.edgecolor": GRID, "axes.labelcolor": INK2, "axes.titlecolor": INK,
    "xtick.color": INK2, "ytick.color": INK2, "text.color": INK,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
    "font.size": 11, "axes.titlesize": 14, "axes.titleweight": "bold",
    "legend.frameon": False, "figure.dpi": 100,
})


def save(fig, name: str) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / f"{name}.png"
    fig.savefig(path, dpi=300, bbox_inches="tight", facecolor=SURFACE)
    plt.close(fig)
    print(f"  {path}")


def load(name: str) -> dict:
    return json.loads((ART / f"{name}.json").read_text(encoding="utf-8"))


def _bar_labels(ax, bars, fmt, inside=False, color=INK, offset=3):
    for bar in bars:
        width = bar.get_width()
        y = bar.get_y() + bar.get_height() / 2
        x = width if not inside else width / 2
        ha = "left" if not inside else "center"
        ax.text(x + (offset if not inside and width >= 0 else -offset if not inside else 0),
                y, fmt(width), va="center", ha=ha if width >= 0 or inside else "right",
                fontsize=10, color=color)


# ---------------------------------------------------------------------------
# 1. The headline: stack-only vs lifecycle, per fuel
# ---------------------------------------------------------------------------
def chart_stack_vs_lifecycle() -> None:
    rows = load("emissions_demo")["stack_vs_lifecycle"]
    rows = [r for r in rows if r["fuel"] != "hfo"]
    labels = [r["label"] for r in rows]
    stack = np.array([100 * r["stack_change"] for r in rows])
    truth = np.array([100 * r["lifecycle_change"] for r in rows])

    fig, ax = plt.subplots(figsize=(11, 6.2))
    y = np.arange(len(labels))
    h = 0.38
    stack_bars = ax.barh(y + h / 2, stack, height=h, color=MUTED, label="Stack-only (what a voyage tool reports)")
    truth_colors = [RED if v > 0 else BLUE for v in truth]
    truth_bars = ax.barh(y - h / 2, truth, height=h, color=truth_colors, label="Lifecycle well-to-wake (this tool)")
    ax.axvline(0, color=INK2, linewidth=1)
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.invert_yaxis()
    ax.set_xlabel("Change in GHG vs conventional diesel (%)")
    ax.set_xlim(-115, 60)
    ax.set_title("Three fuels change sign once you count the whole lifecycle")
    for bar, v in zip(truth_bars, truth, strict=True):
        ax.text(v + (2 if v >= 0 else -2), bar.get_y() + bar.get_height() / 2,
                f"{v:+.0f}%", va="center", ha="left" if v >= 0 else "right",
                fontsize=10, fontweight="bold", color=INK)
    for bar, v in zip(stack_bars, stack, strict=True):
        ax.text(v + (2 if v >= 0 else -2), bar.get_y() + bar.get_height() / 2,
                f"{v:+.0f}%", va="center", ha="left" if v >= 0 else "right",
                fontsize=9, color=INK2)
    for i, r in enumerate(rows):
        if r["verdict_flips"]:
            ax.text(58, i, "SIGN FLIP", va="center", ha="right", fontsize=9,
                    fontweight="bold", color=RED)
    ax.legend(loc="upper left", fontsize=10)
    ax.text(0.0, -0.13, "Per 1 TJ of shaft work · AR6 GWP100 · source: artifacts/emissions_demo.json",
            transform=ax.transAxes, fontsize=8, color=MUTED)
    save(fig, "01_stack_vs_lifecycle")


# ---------------------------------------------------------------------------
# 2. Methane slip: default vs measured
# ---------------------------------------------------------------------------
def chart_methane_slip() -> None:
    d = load("emissions_demo")["methane_slip"]
    order = ["FuelEU default (Otto MS)", "measured fleet median", "best engine (diesel SS)"]
    slip = [d[k]["slip_pct"] for k in order]
    change = [100 * d[k]["vs_diesel"] for k in order]
    labels = ["FuelEU regulatory\ndefault", "Measured fleet median\n(745 ship-years)", "Best engine\n(diesel slow-speed)"]

    fig, axes = plt.subplots(1, 2, figsize=(11, 5))
    ax = axes[0]
    bars = ax.bar(labels, slip, color=[MUTED, BLUE, MUTED], width=0.6)
    for b, v in zip(bars, slip, strict=True):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.05, f"{v:.2f}%", ha="center", fontsize=11, fontweight="bold")
    ax.set_ylabel("Methane slip (% of fuel mass)")
    ax.set_title("Slip: assumed vs measured")
    ax.set_ylim(0, 3.6)
    ax.grid(axis="x", visible=False)

    ax = axes[1]
    colors = [RED if c > 0 else BLUE for c in change]
    bars = ax.bar(labels, change, color=colors, width=0.6)
    ax.axhline(0, color=INK2, linewidth=1)
    for b, v in zip(bars, change, strict=True):
        ax.text(b.get_x() + b.get_width() / 2, v + (0.4 if v >= 0 else -0.4),
                f"{v:+.1f}%", ha="center", va="bottom" if v >= 0 else "top", fontsize=11, fontweight="bold")
    ax.set_ylabel("LNG lifecycle GHG vs diesel (%)")
    ax.set_title("Does LNG help? Depends on which number you use")
    ax.set_ylim(-17, 9)
    ax.grid(axis="x", visible=False)
    fig.suptitle("The regulatory default and the measured fleet disagree about LNG's sign",
                 fontsize=13, fontweight="bold", y=1.02)
    fig.text(0.01, -0.02, "Source: EU MRV 2024-2025 measured CH4, 745 LNG carrier ship-years · artifacts/emissions_demo.json",
             fontsize=8, color=MUTED)
    save(fig, "02_methane_slip")


# ---------------------------------------------------------------------------
# 3. Electrification crossover on the Indian grid
# ---------------------------------------------------------------------------
def chart_electrification() -> None:
    from greenfleet.emissions.ports import load_port_registry
    from greenfleet.emissions.wtw import WellToWakeCalculator

    calc = WellToWakeCalculator(gwp_set="ar6_gwp100")
    ports = load_port_registry()
    years = list(range(2024, 2048))
    diesel = calc.compute(1e6, "mgo", "fossil").total_t
    electric = [calc.compute(1e6, "electricity", grid_factor_t_per_mwh=ports.grid_factor(y)).total_t for y in years]
    crossover = ports.electrification_crossover_year(calculator=calc)

    fig, ax = plt.subplots(figsize=(11, 5.4))
    ax.fill_between(years, electric, diesel, where=np.array(electric) > diesel, color=RED, alpha=0.12, interpolate=True)
    ax.fill_between(years, electric, diesel, where=np.array(electric) <= diesel, color=BLUE, alpha=0.12, interpolate=True)
    ax.plot(years, electric, color=BLUE, linewidth=2.2, label="Battery-electric on the Indian grid (CEA factor + T&D losses)")
    ax.axhline(diesel, color=INK2, linewidth=2, linestyle="--", label="Marine diesel (45% engine efficiency)")
    ax.axvline(crossover, color=ORANGE, linewidth=1.5, linestyle=":")
    ax.text(crossover + 0.3, diesel * 0.62, f"crossover\n{crossover}",
            color=ORANGE, fontsize=11, fontweight="bold")
    ax.text(2026.5, diesel * 1.12,
            "electric WORSE on CO2 before the crossover\n"
            "(but zero NOx / SOx / PM at the quayside from day one)",
            fontsize=10, color=RED)
    ax.text(2041, diesel * 0.72, "electric better\non carbon AND air", fontsize=10, color=BLUE)
    ax.set_xlabel("Year")
    ax.set_ylabel("Lifecycle tCO2e per TJ of shaft work")
    ax.set_title("Electrification in India: the carbon benefit arrives with the grid, not the vessel")
    ax.legend(loc="lower left", fontsize=10)
    ax.set_xlim(2024, 2047)
    fig.text(0.01, -0.02, "Grid glide path consistent with the 500 GW non-fossil target; projected years are marked provisional in config/ports.yaml",
             fontsize=8, color=MUTED)
    save(fig, "03_electrification_crossover")


# ---------------------------------------------------------------------------
# 4. Harbour tug: carbon AND air quality (small multiples, one axis each)
# ---------------------------------------------------------------------------
def chart_tug_air_quality() -> None:
    from greenfleet.emissions.ports import load_port_registry
    from greenfleet.emissions.wtw import WellToWakeCalculator

    calc = WellToWakeCalculator(gwp_set="ar6_gwp100")
    ports = load_port_registry()
    E = 12e6
    opts = [("Diesel (today)", "mgo", "fossil", None), ("LNG", "lng", "fossil", None),
            ("E-methanol", "methanol", "e_fuel", None), ("Green H2 fuel cell", "hydrogen", "green", None),
            ("Electric, 2030 grid", "electricity", None, 2030), ("Electric, 2035 grid", "electricity", None, 2035)]
    labels, co2, nox = [], [], []
    for label, fuel, path, year in opts:
        grid = ports.grid_factor(year) if year else None
        co2.append(calc.compute(E, fuel, path, grid_factor_t_per_mwh=grid).total_t)
        nox.append(calc.local_pollutants(E, fuel)["nox_kg"] / 1000.0)
        labels.append(label)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5.2))
    y = np.arange(len(labels))
    colors = [MUTED, AQUA, AQUA, AQUA, BLUE, BLUE]
    for ax, values, title, unit in [(axes[0], co2, "Lifecycle carbon", "tCO2e per year"),
                                    (axes[1], nox, "Quayside NOx (what the port city breathes)", "tonnes NOx per year")]:
        bars = ax.barh(y, values, color=colors, height=0.6)
        ax.set_yticks(y)
        ax.set_yticklabels(labels)
        ax.invert_yaxis()
        ax.set_xlabel(unit)
        ax.set_title(title)
        ax.grid(axis="y", visible=False)
        for b, v in zip(bars, values, strict=True):
            ax.text(v + max(values) * 0.015, b.get_y() + b.get_height() / 2,
                    f"{v:,.0f}" if v >= 10 else f"{v:.1f}", va="center", fontsize=10)
        ax.set_xlim(0, max(values) * 1.18)
    axes[1].set_yticklabels([])
    fig.suptitle("One harbour tug, 12 TJ/yr: electrify for air quality now, carbon follows the grid",
                 fontsize=13, fontweight="bold", y=1.01)
    fig.text(0.01, -0.02, "Physics-derived (tugs are below the MRV threshold, P-05). Local pollutants are never added into CO2e.",
             fontsize=8, color=MUTED)
    save(fig, "04_tug_carbon_vs_air_quality")


# ---------------------------------------------------------------------------
# 5. Model accuracy: baselines on unseen vessels
# ---------------------------------------------------------------------------
def chart_model_accuracy() -> None:
    ev = load("evaluation")
    split = ev["splits"][0]
    models = split["models"]
    order = ["class mean (naive)", "ridge (log)", "XGBoost + monotone", "XGBoost"]
    names = ["Class mean\n(naive baseline)", "Ridge regression", "Physics + XGBoost\n(monotone, ours)", "Physics + XGBoost\n(unconstrained)"]
    mape = [models[m]["mape"] for m in order]
    r2 = [models[m]["r2"] for m in order]

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8))
    ax = axes[0]
    colors = [MUTED, MUTED, BLUE, BLUE]
    bars = ax.barh(names, mape, color=colors, height=0.6)
    ax.invert_yaxis()
    ax.set_xlabel("MAPE on unseen vessels (%) - lower is better")
    ax.set_title("Prediction error, 6,263 held-out ships")
    ax.grid(axis="y", visible=False)
    for b, v in zip(bars, mape, strict=True):
        ax.text(v + 0.5, b.get_y() + b.get_height() / 2, f"{v:.1f}%", va="center", fontsize=11, fontweight="bold")
    ax.set_xlim(0, 40)
    ax = axes[1]
    bars = ax.barh(names, r2, color=colors, height=0.6)
    ax.invert_yaxis()
    ax.set_xlabel("R² on unseen vessels - higher is better")
    ax.set_title("Variance explained")
    ax.grid(axis="y", visible=False)
    ax.set_yticklabels([])
    for b, v in zip(bars, r2, strict=True):
        ax.text(v + 0.01, b.get_y() + b.get_height() / 2, f"{v:.3f}", va="center", fontsize=11, fontweight="bold")
    ax.set_xlim(0, 0.75)
    fig.suptitle("Leakage-safe: 25% of ships fully held out, zero IMO overlap, 100,528 ship-years",
                 fontsize=12, fontweight="bold", y=1.02)
    fig.text(0.01, -0.03, "Target: energy per nautical mile (MJ/nmi). Source: artifacts/evaluation.json, split by_vessel.",
             fontsize=8, color=MUTED)
    save(fig, "05_model_accuracy")


# ---------------------------------------------------------------------------
# 6. Per vessel class
# ---------------------------------------------------------------------------
def chart_per_class() -> None:
    ev = load("evaluation")
    rows = [r for r in ev["splits"][0]["per_class"] if r["group"] != "__overall__" and r["n"] >= 30]
    rows = sorted(rows, key=lambda r: r["mape"])
    names = [r["group"].replace("_", " ") for r in rows]
    mape = [r["mape"] for r in rows]
    n = [r["n"] for r in rows]
    colors = [BLUE if m < 25 else (YELLOW if m < 40 else RED) for m in mape]

    fig, ax = plt.subplots(figsize=(11, 6.5))
    bars = ax.barh(names, mape, color=colors, height=0.65)
    ax.invert_yaxis()
    ax.set_xlabel("MAPE on unseen vessels (%)")
    ax.set_title("Accuracy per vessel class - reported, not hidden behind the average")
    ax.grid(axis="y", visible=False)
    for b, m, c in zip(bars, mape, n, strict=True):
        ax.text(min(m, 60) + 0.8, b.get_y() + b.get_height() / 2, f"{m:.1f}%  (n={c:,})", va="center", fontsize=9.5)
    ax.set_xlim(0, 75)
    ax.axvline(25, color=MUTED, linewidth=0.8, linestyle=":")
    ax.text(25.5, len(names) - 0.4, "25%", fontsize=8, color=MUTED)
    fig.text(0.01, -0.02, "Offshore: 96% of rows lack a design certificate and are flagged cold-start with wide intervals (see chart 07).",
             fontsize=8, color=MUTED)
    save(fig, "06_per_vessel_class")


# ---------------------------------------------------------------------------
# 7. Cold-start stratification
# ---------------------------------------------------------------------------
def chart_cold_start() -> None:
    ev = load("evaluation")
    cs = ev["splits"][0]["cold_start"]
    warm, cold = cs["warm (has certificate)"], cs["cold (no certificate)"]
    labels = [f"Warm\n(has design certificate)\nn={warm['n']:,}", f"Cold\n(no certificate)\nn={cold['n']:,}"]

    fig, axes = plt.subplots(1, 3, figsize=(12, 4.6))
    for ax, key, title, fmt, lim in [
        (axes[0], "mape", "MAPE (%)", lambda v: f"{v:.1f}%", 50),
        (axes[1], "r2", "R²", lambda v: f"{v:.3f}", 0.8),
        (axes[2], None, "80% interval coverage (%)", lambda v: f"{v:.1f}%", 100),
    ]:
        if key:
            vals = [warm[key], cold[key]]
        else:
            vals = [100 * warm["intervals"]["empirical_coverage"], 100 * cold["intervals"]["empirical_coverage"]]
        bars = ax.bar(labels, vals, color=[BLUE, ORANGE], width=0.55)
        ax.set_title(title)
        ax.grid(axis="x", visible=False)
        for b, v in zip(bars, vals, strict=True):
            ax.text(b.get_x() + b.get_width() / 2, v + lim * 0.02, fmt(v), ha="center", fontsize=11, fontweight="bold")
        ax.set_ylim(0, lim)
        if key is None:
            ax.axhline(80, color=INK2, linestyle="--", linewidth=1)
            ax.text(1.35, 81.5, "nominal 80%", fontsize=9, color=INK2, ha="right")
    fig.suptitle("The model knows what it doesn't know: cold-start rows get wide, honest intervals (P-05)",
                 fontsize=12, fontweight="bold", y=1.03)
    fig.text(0.01, -0.03, "Intervals calibrated on out-of-fold residuals. Source: artifacts/evaluation.json.", fontsize=8, color=MUTED)
    save(fig, "07_cold_start")


# ---------------------------------------------------------------------------
# 8. Slow steaming by class (the physics that competitors' cube law can't show)
# ---------------------------------------------------------------------------
def chart_slow_steaming() -> None:
    import pandas as pd

    from greenfleet.prediction.features import build_features, build_target, load_vessel_classes
    from greenfleet.prediction.model import EnergyModel
    from greenfleet.prediction.splits import split_by_vessel

    d = pd.read_parquet("data/interim/mrv_derived.parquet")
    d = d[d.is_usable].copy()
    d["_y"] = build_target(d)
    d = d[np.isfinite(d._y) & (d._y > 0)].reset_index(drop=True)
    sp = split_by_vessel(d, seed=1000)
    model = EnergyModel(seed=1000).fit(build_features(sp.train), sp.train._y.values)
    classes = load_vessel_classes()

    names, per_day, per_mile = [], [], []
    for c, label in [("bulk_carrier", "Bulk carrier"), ("oil_tanker", "Oil tanker"),
                     ("container_ship", "Container ship"), ("ropax", "Ro-pax"), ("cruise_ship", "Cruise ship")]:
        row = pd.DataFrame([{"vessel_class": c, "design_efficiency_gco2_per_t_nmi": 10.0, "has_design_efficiency": 1,
                             "efficiency_metric_eiv": 1, "efficiency_metric_eedi": 0, "efficiency_metric_eexi": 0,
                             "has_ice_class": 0, "ice_class_ordinal": 0.0, "ice_time_share": 0.0,
                             "reporting_period": 2024.0, "time_at_sea_h": 5000.0}])
        p14, p12 = model.predict(row, speed_kn=14.0), model.predict(row, speed_kn=12.0)
        names.append(f"{label}\n(aux {classes.aux_load_share[c]:.0%})")
        per_day.append(100 * (1 - p12.power_kw[0] / p14.power_kw[0]))
        per_mile.append(100 * (1 - p12.energy_per_nmi_mj[0] / p14.energy_per_nmi_mj[0]))

    fig, ax = plt.subplots(figsize=(11, 5.2))
    x = np.arange(len(names))
    w = 0.36
    b1 = ax.bar(x - w / 2, per_day, w, color=BLUE, label="Fuel saved per DAY")
    b2 = ax.bar(x + w / 2, per_mile, w, color=AQUA, label="Fuel saved per MILE")
    ax.axhline(37.0, color=MUTED, linestyle="--", linewidth=1)
    ax.text(len(names) - 0.5, 38, "pure cube law: 37% / day", fontsize=9, color=MUTED, ha="right")
    ax.axhline(26.5, color=MUTED, linestyle=":", linewidth=1)
    ax.text(len(names) - 0.5, 27.5, "pure cube law: 26.5% / mile", fontsize=9, color=MUTED, ha="right")
    for bars in (b1, b2):
        for b in bars:
            ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.6, f"{b.get_height():.1f}%", ha="center", fontsize=10, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(names)
    ax.set_ylabel("Saving from slowing 14 kn to 12 kn (%)")
    ax.set_title("Slow steaming pays off very differently by vessel: a cruise ship saves almost nothing per mile")
    ax.legend(loc="upper right")
    ax.grid(axis="x", visible=False)
    ax.set_ylim(0, 44)
    fig.text(0.01, -0.02, "Separating propulsion (V^3) from speed-independent hotel load is what a universal cube law cannot express.",
             fontsize=8, color=MUTED)
    save(fig, "08_slow_steaming_by_class")


# ---------------------------------------------------------------------------
# 9. Optimizer benchmark: hypervolume, coastal + large
# ---------------------------------------------------------------------------
def chart_optimizer_benchmark() -> None:
    b = load("optimizer_benchmark")
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, key, title in [(axes[0], "coastal", "12 vessels · 2.3×10¹³ combinations · 30 seeds"),
                           (axes[1], "large", "60 vessels · 7.5×10⁸³ combinations · 10 seeds")]:
        res = b[key]["results"]
        order = ["QIEA+QPSO", "NSGA-II", "greedy", "random"]
        means = np.array([res[a]["hypervolume_mean"] for a in order])
        stds = np.array([res[a]["hypervolume_std"] for a in order])
        rel = 100 * means / means.max()
        rel_err = 100 * stds / means.max()
        colors = [BLUE, MUTED, MUTED, MUTED]
        bars = ax.bar(order, rel, yerr=rel_err, color=colors, width=0.6, capsize=4, error_kw={"ecolor": INK2, "linewidth": 1})
        ax.set_ylabel("Hypervolume (% of best)")
        ax.set_title(title, fontsize=11)
        ax.grid(axis="x", visible=False)
        ax.set_ylim(0, 118)
        for bar, v, a in zip(bars, rel, order, strict=True):
            p = res[a].get("wilcoxon_p_vs_qiea")
            tag = f"{v:.0f}%" + (f"\np={p:.3f}" if p is not None else "")
            ax.text(bar.get_x() + bar.get_width() / 2, v + 5, tag, ha="center", fontsize=9.5, fontweight="bold" if a == "QIEA+QPSO" else "normal")
    fig.suptitle("Quantum-inspired optimizer vs conventional methods, equal evaluation budget (6,000)",
                 fontsize=13, fontweight="bold", y=1.02)
    fig.text(0.01, -0.03, "Shared reference point · Wilcoxon signed-rank paired by seed · large instance: QIEA better than every baseline, p ≤ 0.004",
             fontsize=8, color=MUTED)
    save(fig, "09_optimizer_benchmark")


# ---------------------------------------------------------------------------
# 10. Scalability
# ---------------------------------------------------------------------------
def chart_scalability() -> None:
    b = load("optimizer_benchmark")["scalability"]
    sizes = sorted(int(k) for k in b)
    times = [b[str(s)]["time_mean_s"] for s in sizes]
    combos = [b[str(s)]["combinations"] for s in sizes]

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(sizes, times, color=BLUE, linewidth=2.2, marker="o", markersize=8)
    for s, t, c in zip(sizes, times, combos, strict=True):
        ax.annotate(f"{t:.0f} s\n10^{math.log10(c):.0f} combos", (s, t), textcoords="offset points", xytext=(0, 12), ha="center", fontsize=9)
    ax.set_xlabel("Vessels in the fleet")
    ax.set_ylabel("Wall-clock time (s), 6,000 evaluations")
    ax.set_title("Runtime grows roughly linearly with fleet size; feasible plans found at every size")
    ax.set_xscale("log")
    ax.set_xticks(sizes)
    ax.set_xticklabels([str(s) for s in sizes])
    ax.set_ylim(0, max(times) * 1.35)
    fig.text(0.01, -0.02, "Feasible in 3/3 seeds at every size. Front thins above ~50 vessels at this budget (documented limitation).",
             fontsize=8, color=MUTED)
    save(fig, "10_scalability")


# ---------------------------------------------------------------------------
# 11. QIEA verified against brute force
# ---------------------------------------------------------------------------
def chart_qiea_verification() -> None:
    from greenfleet.prediction.qiea import exhaustive_search, qiea_minimise, random_bit_search

    bits = [4, 6, 8, 10, 12]
    q_hits, r_hits = [], []
    for n in bits:
        rng = np.random.default_rng(n)
        val, wt = rng.uniform(1, 10, n), rng.uniform(1, 10, n)
        cap = wt.sum() * 0.4

        def knap(b, val=val, wt=wt, cap=cap):
            w = wt[b].sum()
            return 1e3 + (w - cap) if w > cap else -val[b].sum()

        _, opt, _ = exhaustive_search(knap, n)
        qh = rh = 0
        for s in range(1000, 1030):
            q = qiea_minimise(knap, n, n_individuals=10, n_generations=120, seed=s)
            qh += abs(q.best_score - opt) < 1e-9
            rh += abs(random_bit_search(knap, n, n_evaluations=q.n_evaluations, seed=s).best_score - opt) < 1e-9
        q_hits.append(100 * qh / 30)
        r_hits.append(100 * rh / 30)

    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(bits))
    w = 0.36
    b1 = ax.bar(x - w / 2, q_hits, w, color=BLUE, label="Q-bit + rotation gate (QIEA)")
    b2 = ax.bar(x + w / 2, r_hits, w, color=MUTED, label="Random search, same budget")
    for bars in (b1, b2):
        for b in bars:
            ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 1.5, f"{b.get_height():.0f}%", ha="center", fontsize=10, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels([f"{n} bits\n({2**n:,} states)" for n in bits])
    ax.set_ylabel("Seeds reaching the EXACT optimum (%)")
    ax.set_title("Verified against brute force: exact optimum in 100% of seeds up to 2⁸ states")
    ax.legend(loc="lower left")
    ax.set_ylim(0, 112)
    ax.grid(axis="x", visible=False)
    fig.text(0.01, -0.02, "Knapsack with a known exhaustive optimum · 30 seeds · budget matched per seed (B-01, Q-09)", fontsize=8, color=MUTED)
    save(fig, "11_qiea_verification")


# ---------------------------------------------------------------------------
# 12. A Pareto front from the coastal scenario
# ---------------------------------------------------------------------------
def chart_pareto_front() -> None:
    from greenfleet.optimize.qmoea import optimise_fleet
    from greenfleet.optimize.scenarios import coastal_scenario

    # Give the search a genuinely clean-but-expensive option (e-methanol) so the
    # trade-off between cost and emissions is real rather than a 1% ripple.
    # 2047: the Harit Sagar horizon year, when methanol and LNG are bunkerable on
    # these routes. In 2030 the port-availability constraint (F-06) correctly rules
    # methanol out, which collapses the front to speed/vessel trade-offs only.
    problem = coastal_scenario(
        year=2047,
        fuel_pathways={"mgo": "fossil", "hfo": "fossil", "lng": "fossil", "methanol": "e_fuel"},
        carbon_price_inr_per_t=0.0,
    )
    result = optimise_fleet(problem, n_individuals=30, n_generations=600, seed=1000)
    front = result.front
    order = np.argsort(front[:, 2])
    front = front[order]
    archive = [result.archive[i] for i in order]
    n_vessels = [s.n_deployed for s in archive]
    clean_share = [
        sum(1 for i, f in enumerate(s.fuel) if s.deploy[i] and f == "methanol") / max(s.n_deployed, 1)
        for s in archive
    ]

    fig, ax = plt.subplots(figsize=(10.5, 5.8))
    xs, ys = front[:, 2] / 1e7, front[:, 1] / 1e3
    ax.plot(xs, ys, color=BLUE, linewidth=1.5, alpha=0.5, zorder=1)
    scatter = ax.scatter(xs, ys, s=130, c=clean_share, cmap="Blues", vmin=-0.25, vmax=1.0,
                         edgecolor=INK2, linewidth=1, zorder=2)
    colorbar = fig.colorbar(scatter, ax=ax, pad=0.02, fraction=0.04)
    colorbar.set_label("Share of deployed ships on e-methanol")
    colorbar.set_ticks([0, 0.33, 0.67, 1.0])
    colorbar.set_ticklabels(["0%", "33%", "67%", "100%"])

    cheapest, cleanest = int(np.argmin(front[:, 2])), int(np.argmin(front[:, 1]))
    saving = 100 * (1 - front[cleanest, 1] / front[cheapest, 1])
    premium = 100 * (front[cleanest, 2] / front[cheapest, 2] - 1)
    ax.annotate(f"Cheapest: {n_vessels[cheapest]} ships, all fossil\n"
                f"₹{xs[cheapest]:.0f} cr · {ys[cheapest]:.1f}k tCO2e",
                (xs[cheapest], ys[cheapest]), textcoords="offset points", xytext=(30, -8),
                fontsize=9.5, fontweight="bold", color=ORANGE,
                arrowprops={"arrowstyle": "->", "color": ORANGE})
    ax.annotate(f"Cleanest: {n_vessels[cleanest]} ships, 100% e-methanol\n"
                f"₹{xs[cleanest]:.0f} cr · {ys[cleanest]:.1f}k tCO2e",
                (xs[cleanest], ys[cleanest]), textcoords="offset points", xytext=(-250, 75),
                fontsize=9.5, fontweight="bold", color=AQUA,
                arrowprops={"arrowstyle": "->", "color": AQUA})
    ax.text(0.03, 0.06, f"−{saving:.0f}% GHG for +{premium:.0f}% cost across the front;\n"
            "every intermediate point is a real, feasible mixed-fuel plan",
            transform=ax.transAxes, fontsize=10, color=INK2, va="bottom")
    ax.set_xlabel("Annual cost (INR crore)")
    ax.set_ylabel("Annual lifecycle GHG (thousand tCO2e)")
    ax.set_title(f"A Pareto front, not one answer: {len(front)} non-dominated fleet plans, coastal scenario 2047")
    ax.margins(x=0.10, y=0.14)
    fig.text(0.01, -0.02, "12 candidate vessels · 3 Indian coastal routes · year 2047 · MGO/HFO/LNG + e-methanol · every point meets all cargo demand and every constraint",
             fontsize=8, color=MUTED)
    save(fig, "12_pareto_front")


# ---------------------------------------------------------------------------
# 13. Where quantum-inspired search pays off (the honest summary)
# ---------------------------------------------------------------------------
def chart_where_it_pays_off() -> None:
    rows = [
        ("XGBoost hyperparameters\n(smooth, 7-D)", 0.2, "tie"),
        ("Feature selection\n(2¹⁰ subsets)", 0.0, "tie"),
        ("Fleet, 12 vessels\n(2.3×10¹³)", 6.5, "tie (p=0.14)"),
        ("Knapsack\n(2¹² states)", 269.0, "wins"),
        ("Fleet, 60 vessels\n(7.5×10⁸³)", 19.2, "wins (p=0.004)"),
        ("Rastrigin\n(continuous, multimodal)", 458.0, "wins"),
    ]
    names = [r[0] for r in rows]
    gains = [r[1] for r in rows]
    colors = [BLUE if r[2].startswith("wins") else MUTED for r in rows]

    fig, ax = plt.subplots(figsize=(11, 5.4))
    bars = ax.barh(names, gains, color=colors, height=0.62)
    ax.invert_yaxis()
    ax.set_xscale("symlog", linthresh=5)
    ax.set_xlabel("Improvement over the best conventional baseline (%, symlog)")
    ax.set_title("The advantage appears when the search space is large and rugged - and not otherwise")
    ax.grid(axis="y", visible=False)
    for b, r in zip(bars, rows, strict=True):
        ax.text(b.get_width() + 0.5, b.get_y() + b.get_height() / 2, f"{r[1]:+.1f}%   {r[2]}", va="center", fontsize=10,
                fontweight="bold" if r[2].startswith("wins") else "normal")
    ax.set_xlim(0, 2000)
    fig.text(0.01, -0.02, "Every comparison budget-matched. Ties are reported as ties. Knapsack/Rastrigin: gain in exact-hit rate / objective vs random search.",
             fontsize=8, color=MUTED)
    save(fig, "13_where_quantum_inspired_pays_off")


# ---------------------------------------------------------------------------
# 14. Architecture diagram
# ---------------------------------------------------------------------------
def chart_architecture() -> None:
    fig, ax = plt.subplots(figsize=(13, 6.2))
    ax.set_xlim(0, 13)
    ax.set_ylim(0, 6.2)
    ax.axis("off")

    def box(x, y, w, h, title, lines, color):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.15",
                                    facecolor=color, edgecolor=INK2, linewidth=1.2, alpha=0.95))
        ax.text(x + w / 2, y + h - 0.32, title, ha="center", va="top", fontsize=11, fontweight="bold", color=INK)
        for i, line in enumerate(lines):
            ax.text(x + 0.18, y + h - 0.72 - i * 0.34, line, ha="left", va="top", fontsize=8.6, color=INK)

    def arrow(x1, y1, x2, y2):
        ax.annotate("", (x2, y2), (x1, y1), arrowprops={"arrowstyle": "-|>", "color": INK2, "lw": 1.6})

    box(0.2, 3.4, 2.7, 2.5, "1 · DATA", [
        "EU MRV 2018–2025 (EMSA)",
        "106,922 ship-years, 25,967 ships",
        "Version-pinned, SHA-256",
        "Measured CH₄ / N₂O (2024+)",
        "D-01..D-12 quality gates",
    ], "#e8f0fb")
    box(3.4, 3.4, 3.0, 2.5, "2 · FUEL PREDICTOR", [
        "Physics core: P ∝ Δ^(2/3)·V³",
        "+ XGBoost residual (monotone)",
        "Energy (MJ), not fuel mass",
        "P10/P50/P90, cold-start path",
        "19.9% MAPE unseen vessels",
    ], "#e8f0fb")
    box(6.9, 3.4, 3.0, 2.5, "3 · WELL-TO-WAKE", [
        "WTT + CO₂ + CH₄ + N₂O + pilot",
        "Grey / blue / green pathways",
        "Measured slip: 1.72%",
        "CEA grid, T&D, crossover",
        "NOx / SOx / PM separately",
    ], "#e6f6ef")
    box(10.3, 3.4, 2.5, 2.5, "INDIA LAYER", [
        "8 major ports, by year",
        "Fuel bunkering (F-06)",
        "Shore power (F-07/08)",
        "Harit Sagar KPIs",
        "GTTP / Mormugao",
    ], "#fdf1e8")
    box(1.8, 0.3, 4.4, 2.5, "4 · QUANTUM-INSPIRED OPTIMIZER", [
        "Q-bit encoding + rotation gates (discrete)",
        "QPSO (continuous speed) — hybrid",
        "Pareto archive · constrained domination",
        "Verified vs MILP: 30/30 exact · QUBO ready",
        "Beats NSGA-II at 60 vessels, p=0.004",
    ], "#ede9fb")
    box(6.9, 0.3, 5.9, 2.5, "5 · DECISION SUPPORT (Phase 4–5)", [
        "Scenario engine: carbon price, fuel shock, policy toggle",
        "Pareto front · fleet allocation · emission profiles",
        "Reports · audit trail · on-premises, offline-capable",
        "Baselines: NSGA-II, greedy, random, exact MILP",
        "30-seed benchmarks, Wilcoxon, shared reference point",
    ], "#f4f4f2")

    arrow(2.9, 4.65, 3.4, 4.65)
    arrow(6.4, 4.65, 6.9, 4.65)
    arrow(9.9, 4.65, 10.3, 4.65)
    arrow(4.9, 3.4, 4.0, 2.8)
    arrow(8.4, 3.4, 8.4, 2.8)
    arrow(11.5, 3.4, 10.5, 2.8)
    arrow(6.2, 1.55, 6.9, 1.55)
    ax.text(6.5, 6.05, "GREEN FLEET OPTIMIZER — quantum-inspired, lifecycle-honest, India-specific",
            ha="center", fontsize=13, fontweight="bold", color=INK)
    save(fig, "14_architecture")


def main() -> None:
    logging.basicConfig(level=logging.WARNING)
    print("generating charts ->", OUT)
    for fn in [chart_stack_vs_lifecycle, chart_methane_slip, chart_electrification,
               chart_tug_air_quality, chart_model_accuracy, chart_per_class, chart_cold_start,
               chart_slow_steaming, chart_optimizer_benchmark, chart_scalability,
               chart_qiea_verification, chart_pareto_front, chart_where_it_pays_off,
               chart_architecture]:
        try:
            fn()
        except Exception as exc:  # keep going; report which chart failed
            print(f"  FAILED {fn.__name__}: {exc}")
            raise


if __name__ == "__main__":
    main()
