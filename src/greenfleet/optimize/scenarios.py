"""Built-in fleet scenarios, from trivial to large-scale.

Three sizes, each serving a different purpose:

``tiny``    2 vessels, 1 route. Small enough to brute-force, which is how Q-09 is
            verified: a metaheuristic that cannot solve a trivial instance exactly
            is not trustworthy on a large one.
``coastal`` 12 vessels, 3 Indian coastal routes. The realistic demonstration case.
``large``   configurable up to 500+ vessels, for the scalability curve (B-06, X-01).

Vessel and route figures are representative of Indian coastal shipping rather than
drawn from any operator's data, and are labelled as such. The point of a scenario is
to exercise the optimizer on a realistically shaped problem, not to publish a claim
about a particular fleet.
"""

from __future__ import annotations

import numpy as np

from greenfleet.optimize.problem import FleetProblem, Route, Vessel

__all__ = ["tiny_scenario", "coastal_scenario", "large_scenario", "gttp_tug_scenario"]


def tiny_scenario(**kwargs) -> FleetProblem:
    """2 vessels, 1 route: small enough that brute force gives the true optimum."""
    vessels = [
        Vessel(
            name="Coastal-A", vessel_class="general_cargo", capacity_t=8000.0,
            design_efficiency_gco2_per_t_nmi=18.0,
            min_speed_kn=8.0, max_speed_kn=15.0, tank_volume_m3=900.0,
            annual_cost_inr=9.0e7, available_hours=7000.0,
        ),
        Vessel(
            name="Coastal-B", vessel_class="general_cargo", capacity_t=6000.0,
            design_efficiency_gco2_per_t_nmi=22.0,
            min_speed_kn=8.0, max_speed_kn=14.0, tank_volume_m3=700.0,
            annual_cost_inr=7.5e7, available_hours=7000.0,
        ),
    ]
    routes = [
        Route(
            name="Mumbai-Cochin", ports=("mumbai", "cochin"),
            distance_nmi=560.0, annual_demand_t=250_000.0,
        )
    ]
    defaults = {"year": 2030, "allow_fuels": ("mgo", "hfo")}
    defaults.update(kwargs)
    return FleetProblem(vessels, routes, **defaults)


def coastal_scenario(**kwargs) -> FleetProblem:
    """12 vessels over 3 Indian coastal routes. The demonstration case."""
    rng = np.random.default_rng(1000)
    vessels = []
    for index in range(12):
        capacity = float(rng.choice([6000, 8000, 12000, 18000]))
        vessels.append(
            Vessel(
                name=f"Coastal-{index + 1:02d}",
                vessel_class="general_cargo" if capacity < 12000 else "bulk_carrier",
                capacity_t=capacity,
                design_efficiency_gco2_per_t_nmi=float(rng.uniform(12.0, 26.0)),
                min_speed_kn=8.0,
                max_speed_kn=float(rng.choice([14.0, 15.0, 16.0])),
                tank_volume_m3=capacity / 10.0,
                annual_cost_inr=float(capacity * 11_000),
                available_hours=float(rng.uniform(6500, 7400)),
            )
        )
    routes = [
        Route("Mumbai-Cochin", ("mumbai", "cochin"), 560.0, 900_000.0),
        Route("Mumbai-Mormugao", ("mumbai", "mormugao"), 230.0, 600_000.0),
        Route("Cochin-Thoothukudi", ("cochin", "vo_chidambaranar"), 320.0, 450_000.0),
    ]
    defaults = {
        "year": 2030,
        "allow_fuels": ("mgo", "hfo", "lng", "methanol"),
        "carbon_price_inr_per_t": 2000.0,
    }
    defaults.update(kwargs)
    return FleetProblem(vessels, routes, **defaults)


def gttp_tug_scenario(n_tugs: int = 10, **kwargs) -> FleetProblem:
    """Harbour-tug transition at the demonstration port.

    Tugs are below the MRV threshold, so every figure here is physics-derived and
    labelled as such (P-05 cold start). Short legs make hydrogen and electric drive
    genuinely feasible, which is the opposite of the long-haul case.
    """
    rng = np.random.default_rng(2000)
    vessels = [
        Vessel(
            name=f"Tug-{index + 1:02d}", vessel_class="harbour_tug",
            capacity_t=80.0,
            design_efficiency_gco2_per_t_nmi=None,  # tugs carry no EEDI/EIV
            min_speed_kn=6.0, max_speed_kn=12.0,
            tank_volume_m3=float(rng.uniform(60.0, 140.0)),
            annual_cost_inr=float(rng.uniform(2.2e7, 3.2e7)),
            available_hours=float(rng.uniform(3000, 4200)),
        )
        for index in range(n_tugs)
    ]
    routes = [
        Route("Harbour-duty", ("mormugao", "mormugao"), 25.0, 60_000.0),
    ]
    defaults = {"year": 2030, "allow_fuels": ("mgo", "methanol", "hydrogen")}
    defaults.update(kwargs)
    return FleetProblem(vessels, routes, **defaults)


def large_scenario(n_vessels: int = 100, n_routes: int = 6, **kwargs) -> FleetProblem:
    """A synthetic instance of any size, for the scalability curve (B-06, X-01)."""
    if n_vessels < 1 or n_routes < 1:
        raise ValueError("need at least one vessel and one route")
    rng = np.random.default_rng(3000)
    vessels = [
        Vessel(
            name=f"V{index:04d}",
            vessel_class="bulk_carrier" if index % 3 else "general_cargo",
            capacity_t=float(rng.uniform(5_000, 40_000)),
            design_efficiency_gco2_per_t_nmi=float(rng.uniform(8.0, 30.0)),
            min_speed_kn=8.0,
            max_speed_kn=float(rng.uniform(13.0, 18.0)),
            tank_volume_m3=float(rng.uniform(600, 3000)),
            annual_cost_inr=float(rng.uniform(6e7, 4e8)),
            available_hours=float(rng.uniform(6000, 7500)),
        )
        for index in range(n_vessels)
    ]
    port_pairs = [
        ("mumbai", "cochin"), ("mumbai", "mormugao"), ("cochin", "vo_chidambaranar"),
        ("deendayal", "mumbai"), ("paradip", "visakhapatnam"), ("jnpa", "cochin"),
    ]
    # Demand is scaled to the fleet so a larger instance stays satisfiable rather
    # than becoming trivially infeasible.
    capacity_scale = sum(v.capacity_t for v in vessels) / max(n_routes, 1)
    routes = [
        Route(
            name=f"R{index}",
            ports=port_pairs[index % len(port_pairs)],
            distance_nmi=float(rng.uniform(200, 900)),
            annual_demand_t=float(capacity_scale * rng.uniform(1.0, 2.0)),
        )
        for index in range(n_routes)
    ]
    defaults = {"year": 2030, "allow_fuels": ("mgo", "hfo", "lng", "methanol")}
    defaults.update(kwargs)
    return FleetProblem(vessels, routes, **defaults)
