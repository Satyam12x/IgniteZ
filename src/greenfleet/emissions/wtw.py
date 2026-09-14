"""Well-to-wake greenhouse gas accounting.

This is the module that separates a fleet-transition planner from a voyage
optimizer. Counting only what leaves the funnel makes grey ammonia look perfect and
LNG look clean; neither is true, and a port authority that bought a fleet on those
numbers would have wasted public money.

Energy basis: SHAFT energy, always
----------------------------------
``energy_mj`` throughout this module is **shaft work delivered**, not fuel energy
consumed. The conversion is done per fuel from its ``conversion_efficiency``.

This is not a detail. A marine diesel converts ~45% of fuel energy to shaft work
and an electric drivetrain ~92%. Comparing them on input energy makes a battery
look roughly twice as bad as it is, which for an electrification programme is not
a rounding error - it inverts the recommendation.

What is counted
---------------
For one shaft-energy demand, one fuel, and one production pathway::

    well-to-tank    upstream production, processing and transport of the fuel
    tank-to-wake    CO2 from combustion
                  + CH4 from unburnt fuel slipping through the engine   (F-02)
                  + N2O from combustion                                 (F-03)
                  + pilot fuel, for dual-fuel engines, in full          (P-09)

each converted to CO2-equivalent with a stated GWP horizon (F-10).

The three results this produces that a stack-only model cannot
--------------------------------------------------------------
* **Grey ammonia is worse than diesel.** It is carbon-free at the funnel, so a
  tank-to-wake model scores it zero. Its upstream steam-methane-reforming footprint
  makes it worse than the fuel it replaces (F-01).
* **LNG's advantage is smaller than its CO2 figure suggests.** Methane has ~28x the
  warming effect of CO2, so a few percent slipping unburnt erases much of the
  benefit of a fuel that starts 12% better on carbon (F-02).
* **Ammonia is not zero even when green**, because N2O from combustion has a GWP of
  265-273 (F-03).

Every factor comes from ``config/fuels.yaml``; nothing is hard-coded (F-09).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

from greenfleet.config.loader import FuelRegistry, load_fuel_registry

logger = logging.getLogger(__name__)

__all__ = ["EmissionResult", "WellToWakeCalculator"]


@dataclass(frozen=True, slots=True)
class EmissionResult:
    """Emissions for one energy demand on one fuel/pathway, in tonnes CO2e."""

    fuel: str
    pathway: str
    energy_mj: float

    well_to_tank_t: float
    """Upstream production, processing and transport."""

    tank_to_wake_co2_t: float
    tank_to_wake_ch4_t: float
    """Methane slip, already in CO2e (F-02)."""

    tank_to_wake_n2o_t: float
    """Combustion N2O, already in CO2e (F-03)."""

    pilot_fuel_t: float
    """Total CO2e from pilot fuel on a dual-fuel engine (P-09)."""

    fuel_mass_t: float
    gwp_set: str
    horizon_years: int
    confidence: str
    """Weakest confidence among the factors used - the result is only as good as this."""

    notes: list[str] = field(default_factory=list)

    @property
    def tank_to_wake_t(self) -> float:
        return (
            self.tank_to_wake_co2_t
            + self.tank_to_wake_ch4_t
            + self.tank_to_wake_n2o_t
            + self.pilot_fuel_t
        )

    @property
    def total_t(self) -> float:
        """Well-to-wake total, tonnes CO2e."""
        return self.well_to_tank_t + self.tank_to_wake_t

    @property
    def intensity_gco2e_per_mj(self) -> float:
        """Well-to-wake intensity, the comparable number across fuels."""
        if self.energy_mj <= 0:
            return 0.0
        return self.total_t * 1e6 / self.energy_mj

    def breakdown(self) -> dict[str, float]:
        return {
            "well_to_tank": self.well_to_tank_t,
            "combustion_co2": self.tank_to_wake_co2_t,
            "methane_slip": self.tank_to_wake_ch4_t,
            "n2o": self.tank_to_wake_n2o_t,
            "pilot_fuel": self.pilot_fuel_t,
            "total": self.total_t,
        }


_CONFIDENCE_ORDER = {"high": 0, "medium": 1, "provisional": 2}


def _weakest(*values: str | None) -> str:
    present = [v for v in values if v]
    if not present:
        return "provisional"
    return max(present, key=lambda v: _CONFIDENCE_ORDER.get(v, 2))


class WellToWakeCalculator:
    """Computes lifecycle GHG for a given energy demand and fuel choice."""

    def __init__(
        self,
        registry: FuelRegistry | None = None,
        gwp_set: str | None = None,
    ):
        """
        Args:
            registry: fuel configuration; loaded from config if omitted.
            gwp_set: override the config's GWP set. Use ``"eu_mrv_ar5"`` when a
                figure must reconcile with an official EU MRV number, and
                ``"ar6_gwp100"`` for current climate science (F-10).
        """
        self.registry = registry or load_fuel_registry()
        self.gwp_set_name = gwp_set or self.registry.gwp_set_name
        if self.gwp_set_name not in self.registry.gwp_sets:
            raise ValueError(
                f"unknown GWP set {self.gwp_set_name!r}; "
                f"available: {sorted(self.registry.gwp_sets)}"
            )
        self.gwp = self.registry.gwp_sets[self.gwp_set_name]

    def compute(
        self,
        energy_mj: float,
        fuel: str,
        pathway: str | None = None,
        engine: str | None = None,
        include_pilot_fuel: bool = True,
        grid_factor_t_per_mwh: float | None = None,
    ) -> EmissionResult:
        """Well-to-wake emissions for ``energy_mj`` delivered by ``fuel``.

        Args:
            energy_mj: **shaft** energy delivered at the propeller/auxiliaries, in
                MJ. Converted to fuel or electrical energy per the fuel's
                conversion efficiency, so fuels and batteries are comparable.
            fuel: fuel key from the registry.
            pathway: production pathway (``grey``/``green``/``fossil``/...). Defaults
                to the fuel's only pathway when unambiguous, otherwise required -
                because for ammonia and hydrogen the pathway changes the answer by
                more than an order of magnitude (F-01).
            engine: engine technology, for methane slip (F-02).
            include_pilot_fuel: count the diesel a dual-fuel engine burns (P-09).
            grid_factor_t_per_mwh: required for electricity; see
                ``greenfleet.emissions.shore_power`` (F-07).

        Raises:
            ValueError: energy is negative, or a pathway is needed but not given.
        """
        if energy_mj < 0:
            raise ValueError(f"energy_mj must be non-negative, got {energy_mj}")
        props = self.registry.get(fuel)
        notes: list[str] = []

        if props.is_energy_carrier:
            return self._electricity(energy_mj, fuel, grid_factor_t_per_mwh, notes)

        pathway_key = self._resolve_pathway(fuel, pathway, props)
        pathway_cfg = self.registry.pathway(fuel, pathway_key)

        # Shaft energy -> fuel energy. Everything below is in fuel energy.
        input_energy = self.registry.input_energy_mj(energy_mj, fuel)
        if props.conversion_efficiency != 1.0:
            notes.append(
                f"{100 * props.conversion_efficiency:.0f}% conversion efficiency: "
                f"{input_energy:,.0f} MJ of fuel for {energy_mj:,.0f} MJ of shaft work"
            )

        # --- fuel mass and the pilot split ------------------------------------
        pilot_fraction = 0.0
        if include_pilot_fuel and props.pilot_fuel is not None:
            pilot_fraction = float(props.pilot_fuel["energy_fraction"])
        main_energy = input_energy * (1.0 - pilot_fraction)
        fuel_mass_t = self.registry.energy_to_mass_tonnes(main_energy, fuel)

        # --- well to tank -----------------------------------------------------
        well_to_tank_t = main_energy * pathway_cfg.wtt_gco2e_per_mj / 1e6

        # --- tank to wake: CO2 -----------------------------------------------
        # A biogenic pathway's combustion CO2 is balanced by uptake during growth,
        # so it is not counted at the funnel; its upstream burden already is.
        combustion_co2_t = 0.0 if pathway_cfg.biogenic_carbon else (
            fuel_mass_t * props.cf_co2_t_per_t
        )
        if pathway_cfg.biogenic_carbon:
            notes.append(
                "combustion CO2 treated as biogenic and excluded; upstream burden "
                "is counted in well-to-tank"
            )

        # --- tank to wake: methane (F-02) ------------------------------------
        slip_fraction = self.registry.methane_slip_fraction(fuel, engine)
        if slip_fraction > 0:
            slipped_t = fuel_mass_t * slip_fraction
            methane_t = slipped_t * self.gwp.ch4_fossil
            notes.append(
                f"methane slip {100 * slip_fraction:.2f}% of fuel mass "
                f"({'engine ' + engine if engine else 'fleet-measured default'})"
            )
        else:
            methane_t = main_energy * props.ch4_g_per_mj / 1e6 * self.gwp.ch4_fossil

        # --- tank to wake: N2O (F-03) ----------------------------------------
        n2o_t = main_energy * props.n2o_g_per_mj / 1e6 * self.gwp.n2o

        # --- pilot fuel, counted in full (P-09) ------------------------------
        pilot_t = 0.0
        if pilot_fraction > 0:
            pilot_name = str(props.pilot_fuel["fuel"])  # type: ignore[index]
            pilot_pathway = props.pilot_fuel.get("pathway", "fossil")  # type: ignore[union-attr]
            pilot_result = self.compute(
                energy_mj * pilot_fraction,  # shaft share; re-converted by the pilot fuel
                pilot_name,
                pathway=str(pilot_pathway),
                include_pilot_fuel=False,
            )
            pilot_t = pilot_result.total_t
            notes.append(
                f"{100 * pilot_fraction:.1f}% of energy from pilot {pilot_name}, "
                "counted in full"
            )

        return EmissionResult(
            fuel=fuel,
            pathway=pathway_key,
            energy_mj=energy_mj,
            well_to_tank_t=well_to_tank_t,
            tank_to_wake_co2_t=combustion_co2_t,
            tank_to_wake_ch4_t=methane_t,
            tank_to_wake_n2o_t=n2o_t,
            pilot_fuel_t=pilot_t,
            fuel_mass_t=fuel_mass_t,
            gwp_set=self.gwp_set_name,
            horizon_years=self.gwp.horizon_years,
            confidence=_weakest(
                pathway_cfg.confidence,
                props.confidence_ttw,
                props.confidence_non_co2,
                props.confidence_slip if slip_fraction > 0 else None,
            ),
            notes=notes,
        )

    def _resolve_pathway(self, fuel: str, pathway: str | None, props) -> str:
        if pathway is not None:
            return pathway
        if len(props.pathways) == 1:
            return next(iter(props.pathways))
        raise ValueError(
            f"fuel {fuel!r} has pathways {sorted(props.pathways)} and they differ by "
            "more than an order of magnitude, so one must be named explicitly (F-01)"
        )

    def _electricity(
        self,
        energy_mj: float,
        fuel: str,
        grid_factor_t_per_mwh: float | None,
        notes: list[str],
    ) -> EmissionResult:
        """Shore power / battery electric: all emissions are upstream (F-07)."""
        if grid_factor_t_per_mwh is None:
            raise ValueError(
                "electricity needs a grid emission factor: shore power is only as "
                "clean as the grid behind it (F-07). Use "
                "greenfleet.emissions.shore_power.grid_factor()."
            )
        if grid_factor_t_per_mwh < 0:
            raise ValueError("grid factor must be non-negative")
        props = self.registry.get(fuel)
        charge = props.charge_discharge_efficiency or 1.0
        # Shaft work -> electricity at the motor -> electricity at the shore
        # connection, which is what the grid actually has to generate.
        electrical_mj = self.registry.input_energy_mj(energy_mj, fuel)
        delivered_mwh = electrical_mj / 3600.0 / charge
        upstream_t = delivered_mwh * grid_factor_t_per_mwh
        notes.append(
            f"grid factor {grid_factor_t_per_mwh:.3f} tCO2/MWh; "
            f"{100 * props.conversion_efficiency:.0f}% drivetrain, "
            f"{100 * charge:.0f}% charging"
        )
        return EmissionResult(
            fuel=fuel,
            pathway="grid",
            energy_mj=energy_mj,
            well_to_tank_t=upstream_t,
            tank_to_wake_co2_t=0.0,
            tank_to_wake_ch4_t=0.0,
            tank_to_wake_n2o_t=0.0,
            pilot_fuel_t=0.0,
            fuel_mass_t=0.0,
            gwp_set=self.gwp_set_name,
            horizon_years=self.gwp.horizon_years,
            confidence=_weakest(props.confidence_ttw, props.confidence_efficiency),
            notes=notes,
        )

    def local_pollutants(
        self,
        energy_mj: float,
        fuel: str,
        engine: str | None = None,
    ) -> dict[str, float]:
        """Quayside NOx, SOx and particulates in kg, for ``energy_mj`` of shaft work.

        Deliberately kept out of the CO2e total: these are local air-quality
        pollutants, not greenhouse gases, and adding them together would be a
        category error. They are reported alongside because they are the other half
        of the electrification case (F-07) - on a coal-heavy grid a battery tug can
        raise CO2 in the near term while removing every gram of NOx, SOx and
        particulate matter from the quayside immediately.
        """
        props = self.registry.get(fuel)
        if props.is_energy_carrier:
            return {"nox_kg": 0.0, "sox_kg": 0.0, "pm_kg": 0.0}
        input_energy = self.registry.input_energy_mj(energy_mj, fuel)
        pilot_fraction = (
            float(props.pilot_fuel["energy_fraction"]) if props.pilot_fuel else 0.0
        )
        mass_t = self.registry.energy_to_mass_tonnes(
            input_energy * (1.0 - pilot_fraction), fuel
        )
        totals = {
            "nox_kg": mass_t * props.nox_g_per_kg,
            "sox_kg": mass_t * props.sox_g_per_kg,
            "pm_kg": mass_t * props.pm_g_per_kg,
        }
        if pilot_fraction > 0:
            pilot = self.local_pollutants(energy_mj * pilot_fraction,
                                          str(props.pilot_fuel["fuel"]))
            for key in totals:
                totals[key] += pilot[key]
        return totals

    # ---- comparison helpers ----------------------------------------------

    def compare(
        self,
        energy_mj: float,
        options: list[tuple[str, str | None]],
        grid_factor_t_per_mwh: float | None = None,
        engine: str | None = None,
    ) -> list[EmissionResult]:
        """Compute several fuel/pathway options, sorted cleanest first."""
        results = [
            self.compute(
                energy_mj, fuel, pathway,
                engine=engine, grid_factor_t_per_mwh=grid_factor_t_per_mwh,
            )
            for fuel, pathway in options
        ]
        return sorted(results, key=lambda r: r.total_t)

    def relative_to(
        self,
        energy_mj: float,
        fuel: str,
        pathway: str | None = None,
        baseline_fuel: str = "mgo",
        baseline_pathway: str | None = "fossil",
        engine: str | None = None,
        grid_factor_t_per_mwh: float | None = None,
    ) -> float:
        """Fractional change in lifecycle GHG against a baseline fuel.

        Negative is an improvement. A positive value for a "clean" fuel is the F-01
        case worth showing a decision-maker: switching would increase emissions.
        """
        candidate = self.compute(
            energy_mj, fuel, pathway, engine=engine,
            grid_factor_t_per_mwh=grid_factor_t_per_mwh,
        )
        baseline = self.compute(energy_mj, baseline_fuel, baseline_pathway)
        if baseline.total_t <= 0:
            raise ValueError(f"baseline {baseline_fuel!r} produced zero emissions")
        return candidate.total_t / baseline.total_t - 1.0
