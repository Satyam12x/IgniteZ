"""Port infrastructure: fuel availability, shore power, and grid carbon.

This is the constraint layer that makes a recommendation actionable in India. An
optimizer that assigns green hydrogen to a route with no hydrogen bunkering has
produced a number, not a plan (F-06). One that treats shore power as zero-emission
on a coal-heavy grid has produced a wrong number (F-07).

Three rules enforced here:

``F-06``  A fuel can only be used on a route if every port on that route can supply
          it in the relevant year. Availability is year-indexed, so a multi-year
          transition scenario sees infrastructure appear over time (S-04).
``F-07``  Shore-power emissions follow the grid factor and the T&D losses, not zero.
          On the Indian grid this is a real number, and it falls each year (S-05).
``F-08``  Shore power needs the port fitted *and* the vessel fitted. Either missing
          means the vessel runs its auxiliary engines at berth.
"""

from __future__ import annotations

import bisect
import logging
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Literal

import yaml

logger = logging.getLogger(__name__)

CONFIG_PATH = Path(__file__).parent.parent / "config" / "ports.yaml"

__all__ = [
    "Availability",
    "PortRegistry",
    "ShorePowerResult",
    "load_port_registry",
    "FuelUnavailableError",
]

Availability = Literal["none", "planned", "limited", "available"]

# Only these count as usable for bunkering. "planned" is explicitly NOT usable:
# planning a fleet around infrastructure that does not exist yet is the failure
# mode this rule prevents.
_USABLE: frozenset[str] = frozenset({"available", "limited"})

_RANK = {"none": 0, "planned": 1, "limited": 2, "available": 3}


class FuelUnavailableError(ValueError):
    """A fuel was requested at a port that cannot supply it (F-06)."""


@dataclass(frozen=True, slots=True)
class ShorePowerResult:
    """Outcome of a shore-power assessment for one berth call (F-07, F-08)."""

    available: bool
    reason: str
    energy_mj: float
    grid_emissions_t: float
    """CO2e if drawn from the grid (zero when shore power is unavailable)."""

    auxiliary_emissions_t: float
    """CO2e if the vessel runs its own auxiliary engines instead."""

    grid_factor_t_per_mwh: float

    @property
    def saving_t(self) -> float:
        """Positive means shore power is cleaner than running auxiliaries."""
        if not self.available:
            return 0.0
        return self.auxiliary_emissions_t - self.grid_emissions_t

    @property
    def saving_fraction(self) -> float:
        if not self.available or self.auxiliary_emissions_t <= 0:
            return 0.0
        return self.saving_t / self.auxiliary_emissions_t


class PortRegistry:
    """Typed access to port infrastructure and grid carbon intensity."""

    def __init__(self, payload: dict[str, Any], source_path: Path | None = None):
        self._raw = payload
        self.source_path = source_path
        self.schema_version = int(payload["meta"]["schema_version"])
        self.ports: dict[str, dict[str, Any]] = payload["ports"]
        self._grid = payload["grid"]
        self.policy = payload.get("policy", {})
        self._grid_years = sorted(int(y) for y in self._grid["emission_factor_t_per_mwh"])
        if not self._grid_years:
            raise ValueError("ports.yaml defines no grid emission factors")

    @classmethod
    def from_yaml(cls, path: Path | str = CONFIG_PATH) -> PortRegistry:
        path = Path(path)
        return cls(yaml.safe_load(path.read_text(encoding="utf-8")), source_path=path)

    # ---- ports ------------------------------------------------------------

    def get(self, port: str) -> dict[str, Any]:
        try:
            return self.ports[port]
        except KeyError:
            raise KeyError(
                f"unknown port {port!r}; configured: {sorted(self.ports)}"
            ) from None

    @property
    def gttp_ports(self) -> list[str]:
        """Ports in the Green Tug Transition Programme."""
        return sorted(k for k, v in self.ports.items() if v.get("gttp_port"))

    @property
    def green_hydrogen_hubs(self) -> list[str]:
        return sorted(k for k, v in self.ports.items() if v.get("green_hydrogen_hub"))

    @property
    def demo_site(self) -> str | None:
        for key, value in self.ports.items():
            if value.get("is_demo_site"):
                return key
        return None

    # ---- F-06: fuel availability -----------------------------------------

    def availability(self, port: str, fuel: str, year: int) -> Availability:
        """Availability of ``fuel`` at ``port`` in ``year``.

        Availability is declared at milestone years; a year between milestones
        takes the most recent declared value (step, not interpolation - a bunkering
        facility is either commissioned or it is not).
        """
        schedule = self.get(port).get("fuel_availability", {}).get(fuel)
        if schedule is None:
            return "none"
        years = sorted(int(y) for y in schedule)
        if year < years[0]:
            return "none"
        index = bisect.bisect_right(years, year) - 1
        return str(schedule[years[index]])  # type: ignore[return-value]

    def can_bunker(self, port: str, fuel: str, year: int) -> bool:
        """Whether a vessel can actually take ``fuel`` at ``port`` in ``year``.

        "planned" counts as no. Planning a fleet around infrastructure that has been
        announced but not built is exactly the mistake this guards against.
        """
        return self.availability(port, fuel, year) in _USABLE

    def route_feasible(self, ports: list[str], fuel: str, year: int) -> bool:
        """Whether ``fuel`` can be used across a whole route (F-06).

        Every port on the route must be able to supply it. One gap strands the
        vessel, so the constraint is an AND, not an average.
        """
        if not ports:
            raise ValueError("route must name at least one port")
        return all(self.can_bunker(p, fuel, year) for p in ports)

    def require_route(self, ports: list[str], fuel: str, year: int) -> None:
        """Raise a readable error naming the ports that cannot supply ``fuel``.

        Raises:
            FuelUnavailableError: listing each blocking port and its status, so the
                message explains *why* rather than just refusing.
        """
        blocking = [
            f"{p} ({self.availability(p, fuel, year)})"
            for p in ports
            if not self.can_bunker(p, fuel, year)
        ]
        if blocking:
            raise FuelUnavailableError(
                f"{fuel} is not bunkerable in {year} at: {', '.join(blocking)}. "
                f"Route: {' -> '.join(ports)}"
            )

    def fuels_available_on_route(self, ports: list[str], year: int) -> list[str]:
        """Every fuel usable across the whole route - the optimizer's feasible set."""
        candidates: set[str] = set()
        for port in ports:
            candidates |= set(self.get(port).get("fuel_availability", {}))
        return sorted(f for f in candidates if self.route_feasible(ports, f, year))

    def earliest_year(
        self, port: str, fuel: str, horizon: tuple[int, int] = (2024, 2050)
    ) -> int | None:
        """First year ``fuel`` becomes bunkerable at ``port``, if ever in horizon.

        Answers the scenario question "when does this option open up?" (S-04).
        """
        for year in range(horizon[0], horizon[1] + 1):
            if self.can_bunker(port, fuel, year):
                return year
        return None

    # ---- F-07: grid carbon -----------------------------------------------

    def grid_factor(self, year: int, include_losses: bool = True) -> float:
        """Grid emission factor in tCO2 per MWh delivered at the berth.

        Declared at milestone years and linearly interpolated between them, because
        grid decarbonisation is genuinely gradual - unlike a bunkering facility,
        which is a step change.

        Args:
            include_losses: add transmission and distribution losses, so the figure
                is per MWh *delivered* rather than per MWh generated. Shore power is
                consumed at the berth, so losses belong in the number.
        """
        table = {int(y): float(v) for y, v in self._grid["emission_factor_t_per_mwh"].items()}
        years = self._grid_years
        if year <= years[0]:
            factor = table[years[0]]
        elif year >= years[-1]:
            factor = table[years[-1]]
        else:
            index = bisect.bisect_right(years, year) - 1
            low, high = years[index], years[index + 1]
            span = high - low
            weight = (year - low) / span if span else 0.0
            factor = table[low] + weight * (table[high] - table[low])
        if include_losses:
            factor /= 1.0 - float(self._grid.get("t_and_d_loss_fraction", 0.0))
        return factor

    def shore_power_status(self, port: str, year: int) -> Availability:
        """Onshore power supply status at ``port`` in ``year``.

        Year-indexed and resolved as a step, like fuel availability: an OPS berth
        is commissioned or it is not.
        """
        status = self.get(port).get("shore_power", {}).get("status", "none")
        if not isinstance(status, dict):
            return str(status)  # type: ignore[return-value]
        years = sorted(int(y) for y in status)
        if not years or year < years[0]:
            return "none"
        index = bisect.bisect_right(years, year) - 1
        return str(status[years[index]])  # type: ignore[return-value]

    def shore_power_ports(self, year: int) -> list[str]:
        """Ports with operational shore power in ``year``."""
        return sorted(
            p for p in self.ports if self.shore_power_status(p, year) in _USABLE
        )

    # ---- F-08: shore power -----------------------------------------------

    def shore_power(
        self,
        port: str,
        year: int,
        energy_mj: float,
        vessel_equipped: bool,
        auxiliary_fuel: str = "mgo",
        calculator: Any | None = None,
    ) -> ShorePowerResult:
        """Assess a berth call against shore power (F-07, F-08).

        Shore power requires the port fitted **and** the vessel fitted. Both
        combinations of one-without-the-other fall back to auxiliary engines, and
        the reason is reported rather than silently returning zero.
        """
        if energy_mj < 0:
            raise ValueError(f"energy_mj must be non-negative, got {energy_mj}")
        from greenfleet.emissions.wtw import WellToWakeCalculator

        engine = calculator or WellToWakeCalculator()
        factor = self.grid_factor(year)
        auxiliary_t = engine.compute(energy_mj, auxiliary_fuel, "fossil").total_t

        port_status = self.shore_power_status(port, year)
        port_ready = port_status in _USABLE

        if not port_ready and not vessel_equipped:
            reason = f"neither fitted: port shore power is {port_status!r}, vessel not equipped"
        elif not port_ready:
            reason = f"port shore power is {port_status!r}, not operational"
        elif not vessel_equipped:
            reason = "vessel is not fitted for shore power"
        else:
            reason = "port and vessel both fitted"

        available = port_ready and vessel_equipped
        grid_t = (
            engine.compute(energy_mj, "electricity", grid_factor_t_per_mwh=factor).total_t
            if available
            else 0.0
        )
        return ShorePowerResult(
            available=available,
            reason=reason,
            energy_mj=energy_mj,
            grid_emissions_t=grid_t,
            auxiliary_emissions_t=auxiliary_t,
            grid_factor_t_per_mwh=factor,
        )

    def electrification_crossover_year(
        self,
        baseline_fuel: str = "mgo",
        horizon: tuple[int, int] = (2024, 2050),
        calculator: Any | None = None,
    ) -> int | None:
        """First year battery-electric beats ``baseline_fuel`` on lifecycle CO2e.

        On a coal-heavy grid this is not immediate, and pretending otherwise would
        mislead an electrification programme. Before the crossover the honest case
        for electrification is local air quality - NOx, SOx and particulates removed
        from the quayside on day one - not carbon. After it, both apply.

        Returns:
            The crossover year, or ``None`` if the grid never gets clean enough
            within the horizon.
        """
        from greenfleet.emissions.wtw import WellToWakeCalculator

        engine = calculator or WellToWakeCalculator()
        probe = 1_000_000.0
        diesel = engine.compute(probe, baseline_fuel, "fossil").total_t
        for year in range(horizon[0], horizon[1] + 1):
            electric = engine.compute(
                probe, "electricity", grid_factor_t_per_mwh=self.grid_factor(year)
            ).total_t
            if electric < diesel:
                return year
        return None

    def as_provenance(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "source_path": str(self.source_path) if self.source_path else None,
            "ports": sorted(self.ports),
            "confidence_availability": self._raw["meta"].get("confidence_availability"),
        }


@lru_cache(maxsize=2)
def load_port_registry(path: Path | str = CONFIG_PATH) -> PortRegistry:
    """Load and cache the port registry."""
    return PortRegistry.from_yaml(path)
