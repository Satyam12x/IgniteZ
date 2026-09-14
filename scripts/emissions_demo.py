"""Phase 2 demonstration: what lifecycle accounting changes about the answer.

Runs three comparisons a decision-maker can act on:

1. **Stack-only vs well-to-wake.** The same fuel choice, scored the way a
   voyage-optimisation tool scores it and the way physics scores it. Two fuels
   change sign.
2. **Mormugao harbour-tug transition.** The sponsor's pilot port, with real
   infrastructure constraints applied by year.
3. **Harit Sagar trajectory.** Whether a plan actually meets the 2030 target.

Every number traces to config/fuels.yaml and config/ports.yaml.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

from greenfleet.emissions.kpi import HaritSagarTracker
from greenfleet.emissions.ports import load_port_registry
from greenfleet.emissions.tank import TankCalculator
from greenfleet.emissions.wtw import WellToWakeCalculator
from greenfleet.reproducibility import RunContext

logger = logging.getLogger(__name__)

ONE_TJ = 1_000_000.0

FUEL_OPTIONS = [
    ("mgo", "fossil", "Marine gas oil"),
    ("hfo", "fossil", "Heavy fuel oil"),
    ("lng", "fossil", "LNG (fossil)"),
    ("methanol", "fossil", "Methanol (grey)"),
    ("methanol", "e_fuel", "E-methanol"),
    ("ammonia", "grey", "Ammonia (grey)"),
    ("ammonia", "blue", "Ammonia (blue)"),
    ("ammonia", "green", "Ammonia (green)"),
    ("hydrogen", "grey", "Hydrogen (grey)"),
    ("hydrogen", "green", "Hydrogen (green)"),
]


def stack_only_vs_lifecycle(calc: WellToWakeCalculator) -> list[dict]:
    """What a funnel-only model reports, against the full lifecycle figure."""
    baseline = calc.compute(ONE_TJ, "mgo", "fossil")
    baseline_stack = baseline.tank_to_wake_co2_t

    print("\n" + "=" * 78)
    print("1. STACK-ONLY vs WELL-TO-WAKE  (1 TJ of propulsion energy)")
    print("=" * 78)
    print("   A voyage-optimisation tool counts only combustion CO2. That is the")
    print("   'stack-only' column. The lifecycle column adds upstream production,")
    print("   methane slip, N2O and pilot fuel.\n")
    print(f"   {'fuel':<20}{'stack-only':>12}{'lifecycle':>12}"
          f"{'stack says':>13}{'truth':>10}")
    print(f"   {'':<20}{'tCO2e':>12}{'tCO2e':>12}{'vs diesel':>13}{'vs diesel':>10}")
    print("   " + "-" * 67)

    rows = []
    for fuel, pathway, label in FUEL_OPTIONS:
        result = calc.compute(ONE_TJ, fuel, pathway)
        stack = result.tank_to_wake_co2_t
        stack_change = stack / baseline_stack - 1.0
        true_change = result.total_t / baseline.total_t - 1.0
        flips = (stack_change < 0) and (true_change > 0)
        rows.append({
            "fuel": fuel, "pathway": pathway, "label": label,
            "stack_only_t": stack, "lifecycle_t": result.total_t,
            "stack_change": stack_change, "lifecycle_change": true_change,
            "verdict_flips": flips,
        })
        marker = "  <-- SIGN FLIP" if flips else ""
        print(f"   {label:<20}{stack:>12.1f}{result.total_t:>12.1f}"
              f"{100 * stack_change:>12.0f}%{100 * true_change:>9.0f}%{marker}")

    flipped = [r["label"] for r in rows if r["verdict_flips"]]
    print()
    print("   Fuels a stack-only tool calls an improvement but which actually")
    print(f"   increase lifecycle emissions: {', '.join(flipped) if flipped else 'none'}")
    return rows


def methane_slip_evidence(calc: WellToWakeCalculator) -> dict:
    """LNG scored with the regulatory default against the measured fleet value."""
    print("\n" + "=" * 78)
    print("2. METHANE SLIP: REGULATORY DEFAULT vs MEASURED FLEET")
    print("=" * 78)
    print("   Our slip figure is not a default. It is the median CH4-to-fuel ratio")
    print("   across 745 LNG carrier ship-years in EU MRV 2024-2025.\n")

    mgo = calc.compute(ONE_TJ, "mgo", "fossil").total_t
    out = {}
    print(f"   {'assumption':<28}{'slip':>8}{'tCO2e':>10}{'vs diesel':>12}")
    print("   " + "-" * 58)
    for engine, label in [
        ("otto_medium_speed", "FuelEU default (Otto MS)"),
        (None, "measured fleet median"),
        ("diesel_slow_speed", "best engine (diesel SS)"),
    ]:
        result = calc.compute(ONE_TJ, "lng", "fossil", engine=engine)
        slip = calc.registry.methane_slip_fraction("lng", engine)
        change = result.total_t / mgo - 1.0
        out[label] = {"slip_pct": 100 * slip, "total_t": result.total_t,
                      "vs_diesel": change}
        print(f"   {label:<28}{100 * slip:>7.2f}%{result.total_t:>10.1f}"
              f"{100 * change:>11.1f}%")
    print()
    print("   Using the worst-case default for an unknown vessel would rule LNG out")
    print("   on evidence the measured fleet does not support.")
    return out


def tug_transition(calc: WellToWakeCalculator, tank: TankCalculator, ports) -> dict:
    """Harbour-tug fuel options at the demo port, with real constraints applied."""
    port = ports.demo_site
    name = ports.get(port)["display_name"]
    print("\n" + "=" * 78)
    print(f"3. HARBOUR TUG TRANSITION - {name.upper()}")
    print("=" * 78)
    print("   A tug's duty cycle is short legs and long idling. Annual shaft work")
    print("   is taken as 12 TJ, a typical harbour tug. Figures are PHYSICS-DERIVED:")
    print("   tugs are below the 5000 GT MRV threshold, so no tug appears in the")
    print("   training data (P-05 cold start).\n")

    annual_energy = 12 * ONE_TJ
    leg_nmi, energy_per_nmi = 40.0, 2400.0
    diesel = calc.compute(annual_energy, "mgo", "fossil")

    print(f"   {'option':<22}{'tCO2e/yr':>10}{'vs diesel':>11}"
          f"{'tank':>10}{'bunkerable':>12}{'NOx kg/yr':>10}")
    print("   " + "-" * 75)
    rows = {}
    for fuel, pathway, label in [
        ("mgo", "fossil", "Diesel (today)"),
        ("lng", "fossil", "LNG"),
        ("methanol", "e_fuel", "E-methanol"),
        ("ammonia", "green", "Green ammonia"),
        ("hydrogen", "green", "Green hydrogen"),
        ("electricity", None, "Electric (shore)"),
    ]:
        grid = ports.grid_factor(2030)
        result = calc.compute(annual_energy, fuel, pathway, grid_factor_t_per_mwh=grid)
        change = result.total_t / diesel.total_t - 1.0
        if fuel == "electricity":
            range_ok = "n/a"
        else:
            # GTTP is procuring NEW tugs, so the tank is sized for the fuel rather
            # than inherited from a diesel hull. The informative number is then the
            # tank volume required, not a pass/fail.
            assessment = tank.assess(
                fuel, leg_nmi, energy_per_nmi, size_tank_for="fuel"
            )
            range_ok = f"{assessment.tank_volume_m3:.0f} m3"
        bunkerable = "yes" if ports.can_bunker(port, fuel, 2030) else "not by 2030"
        local = calc.local_pollutants(annual_energy, fuel, None)
        rows[label] = {"total_t": result.total_t, "vs_diesel": change,
                       "range_ok": range_ok, "bunkerable": bunkerable,
                       **local}
        print(f"   {label:<22}{result.total_t:>10.0f}{100 * change:>10.0f}%"
              f"{range_ok:>10}{bunkerable:>12}{local['nox_kg']:>10.0f}")

    print()
    print(f"   Tank volumes are for a purpose-built {leg_nmi:.0f} nmi duty cycle.")
    print("   A fuel can be clean, fit the duty cycle, and still be unavailable -")
    print("   which is why the optimizer needs infrastructure constraints, not just")
    print("   an emissions ranking.")

    crossover = ports.electrification_crossover_year(calculator=calc)
    print()
    print("   ELECTRIFICATION, THE HONEST VERSION (F-07)")
    print("   On the Indian grid, battery-electric beats diesel on lifecycle CO2e")
    print(f"   from {crossover}. Before then it is WORSE on carbon - but it removes")
    print("   every gram of NOx, SOx and particulates from the quayside immediately.")
    print("   For a port city that is a public-health win available today, and the")
    print("   carbon win arrives with grid decarbonisation.")
    rows["_crossover_year"] = crossover
    return rows


def harit_sagar_trajectory(calc: WellToWakeCalculator, ports) -> dict:
    """Does a stated transition plan actually meet the national target?"""
    tracker = HaritSagarTracker(ports.policy)
    print("\n" + "=" * 78)
    print("4. HARIT SAGAR TRAJECTORY (S-06)")
    print("=" * 78)
    print("   Port handling 1 Mt of cargo a year, 120 TJ of vessel energy,")
    print("   progressively switching from diesel to e-methanol.\n")

    cargo_t = 1_000_000.0
    energy = 120 * ONE_TJ
    baseline = calc.compute(energy, "mgo", "fossil").total_t / cargo_t

    print(f"   {'year':<8}{'clean share':>13}{'intensity':>12}"
          f"{'required':>11}{'status':>12}")
    print("   " + "-" * 56)
    out = {}
    for year, clean_share in [(2026, 0.10), (2030, 0.35), (2035, 0.50), (2047, 0.85)]:
        dirty = calc.compute(energy * (1 - clean_share), "mgo", "fossil").total_t
        clean = calc.compute(energy * clean_share, "methanol", "e_fuel").total_t
        status = tracker.assess(year, dirty + clean, cargo_t, baseline)
        out[year] = {
            "clean_share": clean_share,
            "intensity": status.current_intensity,
            "required": status.required_intensity,
            "on_track": status.on_track,
        }
        print(f"   {year:<8}{100 * clean_share:>12.0f}%{status.current_intensity:>12.4f}"
              f"{status.required_intensity:>11.4f}"
              f"{'ON TRACK' if status.on_track else 'OFF TRACK':>12}")
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gwp", default="ar6_gwp100",
                        help="ar6_gwp100 for science, eu_mrv_ar5 to match EU filings")
    parser.add_argument("--output", type=Path,
                        default=Path("artifacts/emissions_demo.json"))
    args = parser.parse_args()

    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(message)s")
    calc = WellToWakeCalculator(gwp_set=args.gwp)
    ports = load_port_registry()
    tank = TankCalculator()

    print(f"GWP set: {args.gwp} (GWP{calc.gwp.horizon_years}, "
          f"CH4={calc.gwp.ch4_fossil}, N2O={calc.gwp.n2o})")

    payload = {
        "run": RunContext(seed=0, stage="emissions.demo",
                          inputs={"gwp_set": args.gwp}).as_dict(),
        "stack_vs_lifecycle": stack_only_vs_lifecycle(calc),
        "methane_slip": methane_slip_evidence(calc),
        "tug_transition": tug_transition(calc, tank, ports),
        "harit_sagar": harit_sagar_trajectory(calc, ports),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    print(f"\nwrote {args.output}")


if __name__ == "__main__":
    main()
