"""Tank volume, range, and the cargo it displaces.

Low-density fuels do not simply cost more - they take space, and space on a ship is
cargo. This module turns that into hard constraints the optimizer must respect:

``F-04``  Hydrogen carries so little energy per installed litre that it is only
          feasible on short routes. A long-haul hydrogen assignment must be rejected
          on tank capacity, not quietly accepted.
``F-05``  Assigning a low-density fuel to a vessel reduces its effective cargo
          capacity, because the extra tank volume comes out of the hold. A fleet plan
          that ignores this under-counts the ships needed to move the same cargo.

The governing number is installed energy density - energy per litre of *tank system*,
including insulation and containment, not per litre of neat fuel. On that basis MGO
carries 38.0 MJ/L and liquid hydrogen 2.8 MJ/L, a factor of 13.6.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

from greenfleet.config.loader import FuelRegistry, load_fuel_registry

logger = logging.getLogger(__name__)

__all__ = ["TankAssessment", "TankCalculator"]

# Usable fraction of a tank: pumpable limits, expansion space, and an operating
# reserve. Bunkering to a hard zero is not an operational plan.
DEFAULT_USABLE_FRACTION = 0.95

# Fraction of nominal range held back as reserve for weather, diversion and queueing.
DEFAULT_RESERVE_FRACTION = 0.15


@dataclass(frozen=True, slots=True)
class TankAssessment:
    """Range and cargo consequences of one fuel choice on one vessel."""

    fuel: str
    tank_volume_m3: float
    energy_per_nmi_mj: float
    usable_energy_mj: float
    range_nmi: float
    """Range at the stated energy intensity, after usable-volume and reserve."""

    required_route_nmi: float
    feasible: bool
    reason: str
    cargo_displaced_t: float
    """Cargo capacity lost to tank volume relative to the reference fuel (F-05)."""

    reference_fuel: str

    @property
    def range_margin(self) -> float:
        """Range divided by the required leg. Below 1.0 means it cannot make it."""
        if self.required_route_nmi <= 0:
            return float("inf")
        return self.range_nmi / self.required_route_nmi


class TankCalculator:
    """Range and cargo-displacement calculations for alternative fuels."""

    def __init__(
        self,
        registry: FuelRegistry | None = None,
        usable_fraction: float = DEFAULT_USABLE_FRACTION,
        reserve_fraction: float = DEFAULT_RESERVE_FRACTION,
    ):
        if not 0.0 < usable_fraction <= 1.0:
            raise ValueError(f"usable_fraction must be in (0, 1], got {usable_fraction}")
        if not 0.0 <= reserve_fraction < 1.0:
            raise ValueError(f"reserve_fraction must be in [0, 1), got {reserve_fraction}")
        self.registry = registry or load_fuel_registry()
        self.usable_fraction = usable_fraction
        self.reserve_fraction = reserve_fraction

    def usable_energy_mj(self, tank_volume_m3: float, fuel: str) -> float:
        """Energy actually available from a tank of the given installed volume."""
        if tank_volume_m3 < 0:
            raise ValueError("tank volume must be non-negative")
        props = self.registry.get(fuel)
        density = props.installed_energy_mj_per_litre
        if density is None:
            raise ValueError(
                f"{fuel!r} is an energy carrier without a volumetric density; "
                "battery capacity is specified directly in MJ, not by tank volume"
            )
        return tank_volume_m3 * 1000.0 * density * self.usable_fraction

    def range_nmi(
        self, tank_volume_m3: float, energy_per_nmi_mj: float, fuel: str
    ) -> float:
        """Range in nautical miles, after usable-volume and operating reserve."""
        if energy_per_nmi_mj <= 0:
            raise ValueError("energy_per_nmi_mj must be positive")
        usable = self.usable_energy_mj(tank_volume_m3, fuel)
        return usable / energy_per_nmi_mj * (1.0 - self.reserve_fraction)

    def required_tank_volume_m3(
        self, route_nmi: float, energy_per_nmi_mj: float, fuel: str
    ) -> float:
        """Installed tank volume needed to cover ``route_nmi`` with reserve."""
        if route_nmi < 0:
            raise ValueError("route_nmi must be non-negative")
        props = self.registry.get(fuel)
        density = props.installed_energy_mj_per_litre
        if density is None:
            raise ValueError(f"{fuel!r} has no volumetric density")
        energy = route_nmi * energy_per_nmi_mj / (1.0 - self.reserve_fraction)
        return energy / (density * 1000.0) / self.usable_fraction

    def assess(
        self,
        fuel: str,
        route_nmi: float,
        energy_per_nmi_mj: float,
        tank_volume_m3: float | None = None,
        reference_fuel: str = "mgo",
        cargo_density_t_per_m3: float = 0.8,
        size_tank_for: str = "reference",
    ) -> TankAssessment:
        """Assess whether ``fuel`` can cover ``route_nmi``, and what it costs in cargo.

        Two genuinely different questions, and the answer differs, so the caller
        states which one it is asking. Both arise in a Green Tug Transition
        Programme decision:

        ``size_tank_for="reference"`` (default)
            **Retrofit.** Keep the existing tank and change the fuel. A tank sized
            for diesel holds far less energy as hydrogen, so range collapses - this
            is where F-04 bites, and where most conversions actually fail.
        ``size_tank_for="fuel"``
            **Newbuild.** Size the tank for this fuel so the route is always
            coverable, and report what that volume costs in cargo space (F-05).

        Args:
            tank_volume_m3: explicit installed volume; overrides ``size_tank_for``.
            cargo_density_t_per_m3: converts displaced volume into displaced cargo
                tonnes (F-05). Dry bulk is roughly 0.8 t/m3.
        """
        if size_tank_for not in {"reference", "fuel"}:
            raise ValueError(
                f"size_tank_for must be 'reference' or 'fuel', got {size_tank_for!r}"
            )
        if tank_volume_m3 is None:
            sizing_fuel = reference_fuel if size_tank_for == "reference" else fuel
            tank_volume_m3 = self.required_tank_volume_m3(
                route_nmi, energy_per_nmi_mj, sizing_fuel
            )
        achievable = self.range_nmi(tank_volume_m3, energy_per_nmi_mj, fuel)
        # Relative tolerance: when the tank is sized for exactly this route with
        # this fuel, range and requirement are equal in exact arithmetic, and a
        # bare >= then fails on floating-point rounding.
        feasible = achievable >= route_nmi * (1.0 - 1e-9)

        # F-05: extra volume this fuel needs beyond the reference comes out of cargo.
        needed = self.required_tank_volume_m3(route_nmi, energy_per_nmi_mj, fuel)
        reference_needed = self.required_tank_volume_m3(
            route_nmi, energy_per_nmi_mj, reference_fuel
        )
        displaced_volume = max(0.0, needed - reference_needed)
        cargo_displaced = displaced_volume * cargo_density_t_per_m3

        if feasible:
            reason = f"range {achievable:,.0f} nmi covers the {route_nmi:,.0f} nmi leg"
        else:
            shortfall = 100.0 * (1.0 - achievable / route_nmi) if route_nmi else 0.0
            reason = (
                f"range {achievable:,.0f} nmi is {shortfall:.0f}% short of the "
                f"{route_nmi:,.0f} nmi leg; {needed:,.0f} m3 of tank would be needed "
                f"against {tank_volume_m3:,.0f} m3 installed"
            )

        return TankAssessment(
            fuel=fuel,
            tank_volume_m3=tank_volume_m3,
            energy_per_nmi_mj=energy_per_nmi_mj,
            usable_energy_mj=self.usable_energy_mj(tank_volume_m3, fuel),
            range_nmi=achievable,
            required_route_nmi=route_nmi,
            feasible=feasible,
            reason=reason,
            cargo_displaced_t=cargo_displaced,
            reference_fuel=reference_fuel,
        )

    def max_route_nmi(
        self, fuel: str, tank_volume_m3: float, energy_per_nmi_mj: float
    ) -> float:
        """Longest leg this fuel can cover from the given tank. The F-04 boundary."""
        return self.range_nmi(tank_volume_m3, energy_per_nmi_mj, fuel)
