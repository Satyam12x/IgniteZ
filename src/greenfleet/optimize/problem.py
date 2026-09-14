"""The green fleet deployment problem, stated formally.

This is deliverable 2 of the problem statement: decision variables, objectives and
constraints, written so that every solver in this package - quantum-inspired,
evolutionary, or exact MILP - optimises the identical problem. A benchmark where
the methods solve subtly different problems proves nothing (B-01).

Formulation
-----------
Given a candidate fleet ``V``, routes ``R`` with annual cargo demand, a set of fuels
``F``, and a planning year:

**Decision variables**, per vessel ``v``::

    deploy[v]  in {0, 1}          is the vessel used at all
    route[v]   in {0..|R|-1}      which route it serves
    fuel[v]    in F               which fuel it burns
    speed[v]   in [v_min, v_max]  service speed, continuous

The first three are discrete and carried by Q-bit encoding; speed is continuous and
carried by QPSO. That split is the hybrid encoding of edge case Q-02.

**Objectives**, all minimised::

    f1  total fuel energy          MJ/year
    f2  well-to-wake GHG           tCO2e/year
    f3  total cost                 INR/year (fuel + charter + carbon)

These genuinely conflict - the cheapest plan is not the cleanest - so the result is
a Pareto front, never a single weighted answer (M-10).

**Constraints**::

    C1  cargo demand met on every route                     (M-01)
    C2  annual emissions within the cap, if one is set      (M-05)
    C3  speed within each vessel's engine limits            (M-06)
    C4  fuel bunkerable at every port on the route          (F-06)
    C5  vessel range sufficient for the route leg           (F-04)
    C6  vessel physically compatible with the route         (M-08)
    C7  round trips are whole numbers                       (M-09)

Feasibility is reported as a *violation magnitude* rather than a boolean, because a
search needs to know how badly a candidate misses in order to move toward feasible.
``evaluate`` returns both, and infeasible solutions are never returned as answers
(M-01, M-05).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from greenfleet.config.loader import FuelRegistry, load_fuel_registry
from greenfleet.emissions.ports import PortRegistry, load_port_registry
from greenfleet.emissions.tank import TankCalculator
from greenfleet.emissions.wtw import WellToWakeCalculator

logger = logging.getLogger(__name__)

__all__ = [
    "Vessel",
    "Route",
    "FleetProblem",
    "FleetSolution",
    "Infeasible",
    "OBJECTIVE_NAMES",
]

OBJECTIVE_NAMES = ("fuel_energy_mj", "emissions_tco2e", "cost_inr")

# Indicative delivered prices, INR per GJ of fuel energy. These are scenario inputs
# (S-03) that a user overrides, not facts; the platform exposes them for editing.
DEFAULT_FUEL_PRICES_INR_PER_GJ: dict[str, float] = {
    "hfo": 1100.0, "mgo": 1500.0, "lng": 1300.0,
    "methanol": 2600.0, "ammonia": 3200.0, "hydrogen": 6000.0,
}

HOURS_PER_YEAR = 8760.0


class Infeasible(ValueError):
    """No solution can satisfy the constraints as stated (M-01, M-05)."""


@dataclass(frozen=True, slots=True)
class Vessel:
    """A candidate vessel: existing hull or a possible acquisition."""

    name: str
    vessel_class: str
    capacity_t: float
    """Cargo capacity in tonnes, before any tank-volume penalty (F-05)."""

    design_efficiency_gco2_per_t_nmi: float | None
    min_speed_kn: float
    max_speed_kn: float
    """M-06: engines have a minimum safe load and a maximum output."""

    tank_volume_m3: float
    compatible_routes: tuple[str, ...] | None = None
    """M-08: draft or berth limits. ``None`` means compatible with all."""

    compatible_fuels: tuple[str, ...] | None = None
    """Which fuels this hull can burn. ``None`` means all configured fuels."""

    annual_cost_inr: float = 0.0
    """Charter, crew, maintenance - incurred only if deployed."""

    available_hours: float = HOURS_PER_YEAR * 0.85
    """Hours at sea per year after maintenance and port time."""

    def __post_init__(self) -> None:
        if self.min_speed_kn <= 0 or self.max_speed_kn <= self.min_speed_kn:
            raise ValueError(
                f"{self.name}: need 0 < min_speed ({self.min_speed_kn}) "
                f"< max_speed ({self.max_speed_kn})"
            )
        if self.capacity_t <= 0:
            raise ValueError(f"{self.name}: capacity must be positive")


@dataclass(frozen=True, slots=True)
class Route:
    """A trade lane with an annual cargo requirement."""

    name: str
    ports: tuple[str, ...]
    distance_nmi: float
    """One-way distance. A round trip is twice this (M-12)."""

    annual_demand_t: float
    max_transit_hours: float | None = None
    """M-07: schedule window. ``None`` means no deadline."""

    def __post_init__(self) -> None:
        if self.distance_nmi <= 0:
            raise ValueError(f"{self.name}: distance must be positive")
        if self.annual_demand_t < 0:
            raise ValueError(f"{self.name}: demand cannot be negative")
        if len(self.ports) < 2:
            raise ValueError(f"{self.name}: a route needs at least two ports")


@dataclass(slots=True)
class FleetSolution:
    """One complete deployment decision and everything it implies."""

    deploy: np.ndarray
    route_index: np.ndarray
    fuel: list[str]
    speed_kn: np.ndarray

    objectives: np.ndarray
    """(fuel_energy_mj, emissions_tco2e, cost_inr), all minimised."""

    violation: float
    """Total constraint violation. Zero means feasible."""

    violations: dict[str, float] = field(default_factory=dict)
    cargo_delivered_t: dict[str, float] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)

    @property
    def feasible(self) -> bool:
        return self.violation <= 1e-9

    @property
    def n_deployed(self) -> int:
        return int(self.deploy.sum())

    def summary(self) -> str:
        status = "feasible" if self.feasible else f"violation {self.violation:.3g}"
        return (
            f"{self.n_deployed} vessels, "
            f"{self.objectives[1]:,.0f} tCO2e, "
            f"INR {self.objectives[2]:,.0f}, {status}"
        )


class FleetProblem:
    """The optimisation problem. Every solver evaluates through this object."""

    def __init__(
        self,
        vessels: list[Vessel],
        routes: list[Route],
        year: int = 2030,
        emission_cap_tco2e: float | None = None,
        carbon_price_inr_per_t: float = 0.0,
        fuel_price_inr_per_gj: dict[str, float] | None = None,
        fuel_pathways: dict[str, str] | None = None,
        energy_model: Any | None = None,
        registry: FuelRegistry | None = None,
        ports: PortRegistry | None = None,
        allow_fuels: tuple[str, ...] | None = None,
    ):
        if not vessels:
            raise ValueError("problem needs at least one candidate vessel")
        if not routes:
            raise ValueError("problem needs at least one route")

        self.vessels = vessels
        self.routes = routes
        self.year = year
        self.emission_cap_tco2e = emission_cap_tco2e
        self.carbon_price_inr_per_t = carbon_price_inr_per_t
        self.registry = registry or load_fuel_registry()
        self.ports = ports or load_port_registry()
        self.calculator = WellToWakeCalculator(self.registry)
        self.tank = TankCalculator(self.registry)
        self.energy_model = energy_model

        self.fuels: tuple[str, ...] = allow_fuels or tuple(
            f for f in self.registry.fuels if f != "electricity"
        )
        # Grey pathways are the default only where a fuel has one pathway; for
        # ammonia and hydrogen the caller must choose, because it changes the
        # answer by more than 5x (F-01).
        self.fuel_pathways = fuel_pathways or self._default_pathways()
        self.fuel_price_inr_per_gj = fuel_price_inr_per_gj or self._default_prices()

        self._route_index = {r.name: i for i, r in enumerate(self.routes)}
        self._curves: dict[str, Any] = {}
        self._validate()

    # ---- setup ------------------------------------------------------------

    def _default_pathways(self) -> dict[str, str]:
        pathways = {}
        for fuel in self.fuels:
            available = self.registry.get(fuel).pathways
            if len(available) == 1:
                pathways[fuel] = next(iter(available))
            elif "fossil" in available:
                pathways[fuel] = "fossil"
            elif "grey" in available:
                pathways[fuel] = "grey"
            else:
                pathways[fuel] = next(iter(available))
        return pathways

    def _default_prices(self) -> dict[str, float]:
        """Indicative delivered fuel prices, INR per GJ. Scenario inputs (S-03)."""
        return dict(DEFAULT_FUEL_PRICES_INR_PER_GJ)

    def _validate(self) -> None:
        for fuel in self.fuels:
            self.registry.get(fuel)
            if fuel not in self.fuel_price_inr_per_gj:
                raise ValueError(f"no price configured for fuel {fuel!r}")
        for route in self.routes:
            for port in route.ports:
                self.ports.get(port)
        total_capacity = sum(v.capacity_t for v in self.vessels)
        total_demand = sum(r.annual_demand_t for r in self.routes)
        if total_capacity <= 0 and total_demand > 0:
            raise Infeasible("fleet has no capacity but demand is positive")

    # ---- energy -----------------------------------------------------------

    def vessel_features(self, vessel: Vessel) -> dict[str, Any]:
        """The leakage-free feature row the Phase 1 model expects for a vessel."""
        return {
            "vessel_class": vessel.vessel_class,
            "design_efficiency_gco2_per_t_nmi": vessel.design_efficiency_gco2_per_t_nmi,
            "has_design_efficiency": int(vessel.design_efficiency_gco2_per_t_nmi is not None),
            "efficiency_metric_eiv": 1, "efficiency_metric_eedi": 0,
            "efficiency_metric_eexi": 0,
            "has_ice_class": 0, "ice_class_ordinal": 0.0, "ice_time_share": 0.0,
            "reporting_period": float(self.year),
            "time_at_sea_h": vessel.available_hours,
        }

    def speed_curve(self, vessel: Vessel):
        """The trained model's curve for one vessel, evaluated once and cached.

        The residual model fixes a vessel's intensity level; only the imposed
        physics varies with speed. Calling XGBoost inside the optimizer's inner loop
        would cost milliseconds per evaluation for an answer that cannot change, so
        the curve is computed on first use and reused for every speed thereafter.
        """
        curve = self._curves.get(vessel.name)
        if curve is None:
            import pandas as pd

            features = pd.DataFrame([self.vessel_features(vessel)])
            curve = self.energy_model.speed_curves(features)[0]
            self._curves[vessel.name] = curve
        return curve

    def energy_per_nmi_mj(self, vessel: Vessel, speed_kn: float) -> float:
        """Shaft energy per nautical mile at a given speed.

        Uses the Phase 1 model when one is supplied, so the optimizer inherits the
        measured intensity level and the imposed cube-law speed response. Falls
        back to a physics-only estimate otherwise, which keeps the problem usable
        in tests without a trained model.
        """
        if self.energy_model is not None:
            return self.speed_curve(vessel).energy_per_nmi_mj(speed_kn)
        # Physics fallback: intensity scales with the square of speed at constant
        # displacement, anchored on a nominal reference.
        reference_speed, reference_intensity = 14.0, 2500.0
        scale = vessel.capacity_t / 20000.0
        return reference_intensity * scale * (speed_kn / reference_speed) ** 2

    # ---- evaluation -------------------------------------------------------

    def evaluate(
        self,
        deploy: np.ndarray,
        route_index: np.ndarray,
        fuel_index: np.ndarray,
        speed_kn: np.ndarray,
    ) -> FleetSolution:
        """Score one complete deployment decision.

        Returns objectives and a violation magnitude. The magnitude matters: a
        search needs to know how far from feasible a candidate is, not merely that
        it missed.
        """
        n = len(self.vessels)
        deploy = np.asarray(deploy, dtype=bool).reshape(n)
        route_index = np.asarray(route_index, dtype=int).reshape(n)
        fuel_index = np.asarray(fuel_index, dtype=int).reshape(n)
        speed_kn = np.asarray(speed_kn, dtype="float64").reshape(n)

        fuels = [self.fuels[i % len(self.fuels)] for i in fuel_index]
        routes = [self.routes[i % len(self.routes)] for i in route_index]

        total_energy = 0.0
        total_emissions = 0.0
        total_cost = 0.0
        delivered = dict.fromkeys((r.name for r in self.routes), 0.0)
        violations: dict[str, float] = {}
        notes: list[str] = []

        for index, vessel in enumerate(self.vessels):
            if not deploy[index]:
                continue
            route = routes[index]
            fuel = fuels[index]

            # C3: speed bounds are clamped, and the clamp is recorded (M-06).
            speed = float(np.clip(speed_kn[index], vessel.min_speed_kn, vessel.max_speed_kn))
            if not math_isclose(speed, speed_kn[index]):
                violations["speed_bounds"] = violations.get("speed_bounds", 0.0) + abs(
                    float(speed_kn[index]) - speed
                )

            # C6: vessel-route compatibility (M-08).
            if vessel.compatible_routes is not None and route.name not in vessel.compatible_routes:
                violations["route_compatibility"] = (
                    violations.get("route_compatibility", 0.0) + 1.0
                )
                continue
            if vessel.compatible_fuels is not None and fuel not in vessel.compatible_fuels:
                violations["fuel_compatibility"] = (
                    violations.get("fuel_compatibility", 0.0) + 1.0
                )
                continue

            # C4: the fuel must be bunkerable at every port on the route (F-06).
            if not self.ports.route_feasible(list(route.ports), fuel, self.year):
                violations["fuel_availability"] = (
                    violations.get("fuel_availability", 0.0) + 1.0
                )
                continue

            energy_per_nmi = self.energy_per_nmi_mj(vessel, speed)

            # C5: range on one leg, with the tank actually installed (F-04).
            assessment = self.tank.assess(
                fuel, route.distance_nmi, energy_per_nmi,
                tank_volume_m3=vessel.tank_volume_m3,
            )
            if not assessment.feasible:
                violations["range"] = violations.get("range", 0.0) + (
                    route.distance_nmi - assessment.range_nmi
                ) / route.distance_nmi
                continue

            # F-05: tank volume displaces cargo.
            effective_capacity = max(
                0.0, vessel.capacity_t - assessment.cargo_displaced_t
            )

            # C7: whole round trips in the available hours (M-09, M-12).
            round_trip_nmi = 2.0 * route.distance_nmi
            hours_per_trip = round_trip_nmi / speed
            if route.max_transit_hours is not None and (
                route.distance_nmi / speed > route.max_transit_hours
            ):
                violations["schedule"] = violations.get("schedule", 0.0) + (
                    route.distance_nmi / speed - route.max_transit_hours
                )
                continue
            trips = int(vessel.available_hours // hours_per_trip)
            if trips <= 0:
                violations["no_trips_possible"] = (
                    violations.get("no_trips_possible", 0.0) + 1.0
                )
                continue

            delivered[route.name] += trips * effective_capacity

            shaft_energy = trips * round_trip_nmi * energy_per_nmi
            emission = self.calculator.compute(
                shaft_energy, fuel, self.fuel_pathways.get(fuel)
            )
            fuel_energy_mj = self.registry.input_energy_mj(shaft_energy, fuel)

            total_energy += fuel_energy_mj
            total_emissions += emission.total_t
            total_cost += (
                fuel_energy_mj / 1000.0 * self.fuel_price_inr_per_gj[fuel]
                + vessel.annual_cost_inr
                + emission.total_t * self.carbon_price_inr_per_t
            )

        # C1: cargo demand on every route (M-01).
        for route in self.routes:
            shortfall = route.annual_demand_t - delivered[route.name]
            if shortfall > 1e-9:
                violations[f"demand:{route.name}"] = shortfall / max(
                    route.annual_demand_t, 1.0
                )

        # C2: emission cap (M-05).
        if self.emission_cap_tco2e is not None:
            excess = total_emissions - self.emission_cap_tco2e
            if excess > 1e-9:
                violations["emission_cap"] = excess / max(self.emission_cap_tco2e, 1.0)

        return FleetSolution(
            deploy=deploy,
            route_index=route_index,
            fuel=fuels,
            speed_kn=speed_kn,
            objectives=np.array([total_energy, total_emissions, total_cost]),
            violation=float(sum(violations.values())),
            violations=violations,
            cargo_delivered_t=delivered,
            notes=notes,
        )

    # ---- diagnostics ------------------------------------------------------

    def max_theoretical_capacity_t(self) -> dict[str, float]:
        """Best-case annual delivery per route, ignoring emissions and cost.

        Used to answer M-01 honestly: if demand exceeds this, no plan exists and the
        right response is to say so and state the shortfall, not to return a
        "solution" that quietly drops cargo.
        """
        best = dict.fromkeys((r.name for r in self.routes), 0.0)
        for route in self.routes:
            for vessel in self.vessels:
                if vessel.compatible_routes is not None and (
                    route.name not in vessel.compatible_routes
                ):
                    continue
                hours_per_trip = 2.0 * route.distance_nmi / vessel.max_speed_kn
                trips = int(vessel.available_hours // hours_per_trip)
                best[route.name] += max(0, trips) * vessel.capacity_t
        return best

    def check_demand_satisfiable(self) -> None:
        """Raise with a useful message if demand cannot be met by any plan (M-01).

        Raises:
            Infeasible: naming the route, the shortfall, and the extra capacity
                needed - so the user learns what to change.
        """
        best = self.max_theoretical_capacity_t()
        problems = []
        for route in self.routes:
            if route.annual_demand_t > best[route.name] + 1e-9:
                shortfall = route.annual_demand_t - best[route.name]
                problems.append(
                    f"{route.name}: demand {route.annual_demand_t:,.0f} t exceeds the "
                    f"fleet's best case {best[route.name]:,.0f} t "
                    f"(short by {shortfall:,.0f} t)"
                )
        if problems:
            raise Infeasible(
                "cargo demand cannot be met even ignoring emissions and cost. "
                + "; ".join(problems)
            )

    @property
    def n_vessels(self) -> int:
        return len(self.vessels)

    @property
    def n_routes(self) -> int:
        return len(self.routes)

    @property
    def n_fuels(self) -> int:
        return len(self.fuels)

    def search_space_size(self) -> float:
        """Number of discrete combinations, for the scalability story."""
        per_vessel = 1 + self.n_routes * self.n_fuels
        return float(per_vessel) ** self.n_vessels

    def as_provenance(self) -> dict[str, Any]:
        return {
            "n_vessels": self.n_vessels,
            "n_routes": self.n_routes,
            "fuels": list(self.fuels),
            "fuel_pathways": dict(self.fuel_pathways),
            "year": self.year,
            "emission_cap_tco2e": self.emission_cap_tco2e,
            "carbon_price_inr_per_t": self.carbon_price_inr_per_t,
            "search_space_size": self.search_space_size(),
        }


def math_isclose(a: float, b: float, tol: float = 1e-9) -> bool:
    return abs(a - b) <= tol
