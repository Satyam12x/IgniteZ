"""Glue between HTTP requests and the Phase 1-3 engines.

Each function here builds the inputs an engine expects, calls it, and shapes the
result for the dashboard. None of them computes an emission factor, a physics
term or an objective value itself - that would create a second source of truth
the tests do not cover.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd

from greenfleet.api import schemas
from greenfleet.api.jobs import Job
from greenfleet.api.state import AppState
from greenfleet.optimize import scenarios
from greenfleet.optimize.baselines import (
    BaselineResult,
    greedy_fleet,
    nsga2_fleet,
    qbho_fleet,
    random_search_fleet,
)
from greenfleet.optimize.encoding import FleetEncoding
from greenfleet.optimize.pareto import hypervolume, reference_point
from greenfleet.optimize.problem import (
    DEFAULT_FUEL_PRICES_INR_PER_GJ,
    FleetProblem,
    FleetSolution,
    Infeasible,
    Route,
    Vessel,
)
from greenfleet.optimize.qmoea import QMOEAResult, optimise_fleet

# The comparison set the dashboard opens with. Pathways are explicit because for
# ammonia and hydrogen the pathway changes the answer by more than 5x (F-01).
DEFAULT_COMPARISON: list[tuple[str, str | None, str]] = [
    ("mgo", "fossil", "Diesel (MGO)"),
    ("hfo", "fossil", "Heavy fuel oil"),
    ("lng", "fossil", "LNG (measured slip)"),
    ("methanol", "fossil", "Grey methanol (fossil)"),
    ("methanol", "e_fuel", "E-methanol"),
    ("ammonia", "grey", "Grey ammonia"),
    ("ammonia", "green", "Green ammonia"),
    ("hydrogen", "grey", "Grey hydrogen"),
    ("hydrogen", "green", "Green hydrogen"),
    ("electricity", None, "Battery-electric (grid)"),
]

MILESTONE_YEARS = (2024, 2030, 2040, 2047)


def _finite(value: float) -> float | None:
    value = float(value)
    return value if math.isfinite(value) else None


# ---- meta -----------------------------------------------------------------


def meta(state: AppState) -> dict[str, Any]:
    registry, ports, classes = state.registry, state.ports, state.classes
    fuels = {}
    for key, props in registry.fuels.items():
        fuels[key] = {
            "display_name": props.display_name,
            "lhv_mj_per_kg": props.lhv_mj_per_kg,
            "is_energy_carrier": props.is_energy_carrier,
            "conversion_efficiency": props.conversion_efficiency,
            "pathways": {
                name: {"label": p.label, "wtt_gco2e_per_mj": p.wtt_gco2e_per_mj,
                       "confidence": p.confidence}
                for name, p in props.pathways.items()
            },
            "default_engine": props.default_engine,
            "methane_slip_options": dict(props.methane_slip_pct_of_fuel_mass),
            "default_price_inr_per_gj": DEFAULT_FUEL_PRICES_INR_PER_GJ.get(key),
        }
    port_rows = {}
    for key, entry in ports.ports.items():
        port_rows[key] = {
            "display_name": entry.get("display_name", key),
            "state": entry.get("state"),
            "is_demo_site": bool(entry.get("is_demo_site")),
            "gttp_port": bool(entry.get("gttp_port")),
            "green_hydrogen_hub": bool(entry.get("green_hydrogen_hub")),
            "shore_power": {y: ports.shore_power_status(key, y) for y in MILESTONE_YEARS},
            "fuel_availability": {
                fuel: {y: ports.availability(key, fuel, y) for y in MILESTONE_YEARS}
                for fuel in registry.fuels
            },
        }
    grid_years = list(range(2023, 2048))
    vessel_classes = [
        {
            "key": key,
            "display_name": classes.display_name.get(key, key),
            "typical_speed_kn": classes.typical_speed_kn.get(key),
            "aux_load_share": classes.aux_load_share.get(key),
            "in_mrv_scope": classes.in_mrv_scope.get(key, True),
        }
        for key in sorted(classes.typical_speed_kn)
    ]
    return {
        "fuels": fuels,
        "gwp_set": registry.gwp_set_name,
        "ports": port_rows,
        "demo_site": ports.demo_site,
        "grid_factor_t_per_mwh": {y: ports.grid_factor(y) for y in grid_years},
        "electrification_crossover_year": ports.electrification_crossover_year(
            calculator=state.calculator
        ),
        "harit_sagar": {
            "baseline_year": state.harit_sagar.baseline_year,
            "targets": state.harit_sagar.targets,
        },
        "vessel_classes": vessel_classes,
        "scenarios": {name: scenario_spec(name) for name in SCENARIOS},
        "model": state.model_summary(),
    }


def scenario_spec(name: str) -> dict[str, Any]:
    """A built-in scenario as editable vessel/route rows."""
    problem = _scenario_problem(name)
    return {
        "year": problem.year,
        "fuels": list(problem.fuels),
        "fuel_pathways": dict(problem.fuel_pathways),
        "carbon_price_inr_per_t": problem.carbon_price_inr_per_t,
        "vessels": [
            {
                "name": v.name, "vessel_class": v.vessel_class, "capacity_t": v.capacity_t,
                "design_efficiency_gco2_per_t_nmi": v.design_efficiency_gco2_per_t_nmi,
                "min_speed_kn": v.min_speed_kn, "max_speed_kn": v.max_speed_kn,
                "tank_volume_m3": v.tank_volume_m3, "annual_cost_inr": v.annual_cost_inr,
                "available_hours": v.available_hours,
            }
            for v in problem.vessels
        ],
        "routes": [
            {
                "name": r.name, "ports": list(r.ports), "distance_nmi": r.distance_nmi,
                "annual_demand_t": r.annual_demand_t, "max_transit_hours": r.max_transit_hours,
            }
            for r in problem.routes
        ],
    }


def _coastal_2047(**kwargs) -> FleetProblem:
    """The coastal fleet once e-methanol is bunkerable: the report's Pareto case.

    Same 12 hulls and 3 routes; 2047 availability, methanol on the e-fuel pathway,
    no carbon price - so the trade-off shown is fuel cost against emissions alone.
    """
    defaults = {
        "year": 2047,
        "fuel_pathways": {"hfo": "fossil", "mgo": "fossil", "lng": "fossil", "methanol": "e_fuel"},
        "carbon_price_inr_per_t": 0.0,
    }
    defaults.update(kwargs)
    return scenarios.coastal_scenario(**defaults)


SCENARIOS = {
    "coastal_2047": _coastal_2047,
    "coastal": scenarios.coastal_scenario,
    "gttp_tug": scenarios.gttp_tug_scenario,
    "tiny": scenarios.tiny_scenario,
}


def _scenario_problem(name: str, **kwargs) -> FleetProblem:
    return SCENARIOS[name](**kwargs)


# ---- prediction -----------------------------------------------------------


def predict(state: AppState, request: schemas.PredictRequest) -> schemas.PredictResponse:
    if state.model is None:
        raise RuntimeError("no trained model loaded; run scripts/train_model.py")
    classes = state.classes
    vessel_class = classes.resolve(request.vessel_class)
    features = pd.DataFrame([{
        "vessel_class": vessel_class,
        "design_efficiency_gco2_per_t_nmi": request.design_efficiency_gco2_per_t_nmi,
        "has_design_efficiency": int(request.design_efficiency_gco2_per_t_nmi is not None),
        "efficiency_metric_eiv": 1, "efficiency_metric_eedi": 0, "efficiency_metric_eexi": 0,
        "has_ice_class": 0, "ice_class_ordinal": 0.0, "ice_time_share": 0.0,
        "reporting_period": float(request.year),
        "time_at_sea_h": request.time_at_sea_h,
    }])
    curve = state.model.speed_curves(features)[0]
    if request.speeds_kn is None:
        speeds = np.linspace(0.0, 1.25 * curve.reference_speed_kn, 26)
    else:
        speeds = np.asarray(request.speeds_kn, dtype="float64")
    prediction = state.model.predict(features.loc[features.index.repeat(len(speeds))]
                                     .reset_index(drop=True),
                                     speed_kn=speeds, load_ratio=request.load_ratio)
    fuels = [f for f in request.fuels if f in state.registry.fuels
             and not state.registry.get(f).is_energy_carrier]

    points = []
    for i, speed in enumerate(speeds):
        energy = _finite(prediction.energy_per_nmi_mj[i])
        mass = {}
        for fuel in fuels:
            if energy is None:
                mass[fuel] = None
            else:
                fuel_energy = state.registry.input_energy_mj(energy * 1000.0, fuel)
                mass[fuel] = state.registry.energy_to_mass_tonnes(fuel_energy, fuel)
        points.append(schemas.CurvePoint(
            speed_kn=float(speed),
            energy_per_nmi_mj=energy,
            p10_energy_per_nmi_mj=_finite(prediction.p10_energy_per_nmi_mj[i]),
            p90_energy_per_nmi_mj=_finite(prediction.p90_energy_per_nmi_mj[i]),
            power_kw=float(prediction.power_kw[i]),
            extrapolating=bool(prediction.extrapolating[i]),
            fuel_t_per_1000_nmi=mass,
        ))
    reason = None
    if curve.cold_start:
        reason = (
            "no design-efficiency certificate: the level is the class typical value and "
            "the interval is widened (P-05)"
            if request.design_efficiency_gco2_per_t_nmi is None
            else f"class {vessel_class!r} has no training data (below MRV scope or unseen)"
        )
    return schemas.PredictResponse(
        vessel_class=vessel_class,
        vessel_class_display=classes.display_name.get(vessel_class, vessel_class),
        cold_start=curve.cold_start,
        cold_start_reason=reason,
        reference_speed_kn=curve.reference_speed_kn,
        reference_intensity_mj_per_nmi=curve.reference_intensity_mj_per_nmi,
        aux_share=curve.aux_share,
        optimal_speed_kn=float(state.model.optimal_speed_kn(features, request.load_ratio)[0]),
        points=points,
        model={"source": state.energy_source, "held_out": state.model_card.get("held_out")},
    )


# ---- emissions ------------------------------------------------------------


def compare_emissions(
    state: AppState, request: schemas.EmissionsCompareRequest
) -> schemas.EmissionsCompareResponse:
    calc, registry, ports = state.calculator, state.registry, state.ports
    grid = ports.grid_factor(request.year)
    energy = request.shaft_energy_mj

    def run(option: schemas.FuelOption):
        return calc.compute(
            energy, option.fuel, option.pathway, engine=option.engine,
            include_pilot_fuel=request.include_pilot_fuel, grid_factor_t_per_mwh=grid,
        )

    baseline = run(request.baseline)
    options = request.options
    labels: dict[tuple[str, str | None], str] = {
        (f, p): label for f, p, label in DEFAULT_COMPARISON
    }
    if options is None:
        options = [schemas.FuelOption(fuel=f, pathway=p) for f, p, _ in DEFAULT_COMPARISON]

    rows = []
    for option in options:
        result = run(option)
        label = labels.get((option.fuel, option.pathway)) or (
            f"{registry.get(option.fuel).display_name}"
            + (f" ({result.pathway})" if result.pathway else "")
        )
        stack = result.tank_to_wake_co2_t
        stack_change = (stack / baseline.tank_to_wake_co2_t - 1.0
                        if baseline.tank_to_wake_co2_t > 0 else 0.0)
        change = result.total_t / baseline.total_t - 1.0
        rows.append(schemas.EmissionRow(
            fuel=option.fuel, pathway=result.pathway, label=label,
            breakdown=result.breakdown(), total_t=result.total_t, stack_only_t=stack,
            vs_baseline=change, stack_only_vs_baseline=stack_change,
            verdict_flips=bool(stack_change < 0 and change > 0),
            intensity_gco2e_per_mj=result.intensity_gco2e_per_mj,
            fuel_mass_t=result.fuel_mass_t,
            local_pollutants_kg=calc.local_pollutants(energy, option.fuel, option.engine),
            confidence=result.confidence, notes=list(result.notes),
        ))
    return schemas.EmissionsCompareResponse(
        shaft_energy_mj=energy, year=request.year, grid_factor_t_per_mwh=grid,
        gwp_set=registry.gwp_set_name, rows=rows,
    )


def electrification_series(state: AppState, baseline_fuel: str = "mgo") -> dict[str, Any]:
    """Lifecycle CO2e per TJ of shaft work, electric vs the baseline, by year."""
    calc, ports = state.calculator, state.ports
    energy = 1.0e6
    diesel = calc.compute(energy, baseline_fuel, "fossil").total_t
    years = list(range(2024, 2048))
    electric = [
        calc.compute(energy, "electricity", None,
                     grid_factor_t_per_mwh=ports.grid_factor(y)).total_t
        for y in years
    ]
    local = {
        "diesel": calc.local_pollutants(1.2e7, baseline_fuel),
        "electric": calc.local_pollutants(1.2e7, "electricity"),
    }
    return {
        "years": years, "electric_t_per_tj": electric, "baseline_t_per_tj": diesel,
        "baseline_fuel": baseline_fuel,
        "crossover_year": ports.electrification_crossover_year(baseline_fuel, calculator=calc),
        "local_pollutants_kg_per_tug_year": local,
    }


# ---- optimisation ---------------------------------------------------------


def build_problem(state: AppState, request: schemas.OptimiseRequest) -> FleetProblem:
    # Only settings the caller actually chose are passed, so a built-in scenario
    # keeps its own year, carbon price and pathways unless overridden.
    common: dict[str, Any] = {
        "emission_cap_tco2e": request.emission_cap_tco2e,
        "energy_model": state.model,
        "registry": state.registry,
        "ports": state.ports,
    }
    if request.year is not None:
        common["year"] = request.year
    if request.carbon_price_inr_per_t is not None:
        common["carbon_price_inr_per_t"] = request.carbon_price_inr_per_t
    if request.fuels:
        common["allow_fuels"] = tuple(request.fuels)
    if request.fuel_pathways:
        common["fuel_pathways"] = dict(request.fuel_pathways)
    if request.fuel_price_inr_per_gj:
        common["fuel_price_inr_per_gj"] = {
            **DEFAULT_FUEL_PRICES_INR_PER_GJ, **request.fuel_price_inr_per_gj
        }

    if request.scenario != "custom":
        problem = _scenario_problem(request.scenario, **common)
    else:
        if not request.vessels or not request.routes:
            raise ValueError("a custom scenario needs at least one vessel and one route")
        common.setdefault("year", 2030)
        vessels = [
            Vessel(
                name=v.name, vessel_class=state.classes.resolve(v.vessel_class),
                capacity_t=v.capacity_t,
                design_efficiency_gco2_per_t_nmi=v.design_efficiency_gco2_per_t_nmi,
                min_speed_kn=v.min_speed_kn, max_speed_kn=v.max_speed_kn,
                tank_volume_m3=v.tank_volume_m3,
                compatible_routes=tuple(v.compatible_routes) if v.compatible_routes else None,
                compatible_fuels=tuple(v.compatible_fuels) if v.compatible_fuels else None,
                annual_cost_inr=v.annual_cost_inr, available_hours=v.available_hours,
            )
            for v in request.vessels
        ]
        routes = [
            Route(name=r.name, ports=tuple(r.ports), distance_nmi=r.distance_nmi,
                  annual_demand_t=r.annual_demand_t, max_transit_hours=r.max_transit_hours)
            for r in request.routes
        ]
        problem = FleetProblem(vessels, routes, **common)

    # Pathways the user did not pin fall to the problem's defaults, but ammonia and
    # hydrogen default to grey - which is the F-01 trap. Say so in the provenance
    # rather than silently optimising a fuel the user may believe is green.
    problem.check_demand_satisfiable()
    return problem


def _bau_problem(state: AppState, problem: FleetProblem) -> FleetProblem:
    """The same fleet and routes restricted to diesel (MGO): the Harit Sagar baseline.

    The target is a reduction from a baseline. For a fleet plan the honest
    baseline is the same ships on the same routes at the same speeds with today's
    fuel - not an arbitrary number the plan is guaranteed to beat.
    """
    return FleetProblem(
        problem.vessels, problem.routes, year=problem.year,
        energy_model=state.model, registry=state.registry, ports=state.ports,
        allow_fuels=("mgo",), fuel_pathways={"mgo": "fossil"},
    )


def _bau_intensity(bau: FleetProblem, solution: FleetSolution) -> float | None:
    """Emissions per tonne of cargo if this deployment burned diesel."""
    result = bau.evaluate(
        solution.deploy, solution.route_index,
        np.zeros(len(solution.deploy), dtype=int), solution.speed_kn,
    )
    cargo = sum(result.cargo_delivered_t.values())
    return result.objectives[1] / cargo if cargo > 0 else None


def _plan_out(
    state: AppState, problem: FleetProblem, index: int, solution: FleetSolution,
    bau_intensity: float | None, curves_cold: dict[str, bool],
) -> schemas.PlanOut:
    assignments = []
    mix: dict[str, int] = {}
    for i, vessel in enumerate(problem.vessels):
        deployed = bool(solution.deploy[i])
        fuel = solution.fuel[i] if deployed else None
        if fuel:
            mix[fuel] = mix.get(fuel, 0) + 1
        assignments.append(schemas.VesselAssignment(
            name=vessel.name, vessel_class=vessel.vessel_class, deployed=deployed,
            route=problem.routes[int(solution.route_index[i]) % problem.n_routes].name
            if deployed else None,
            fuel=fuel, pathway=problem.fuel_pathways.get(fuel) if fuel else None,
            speed_kn=float(np.clip(solution.speed_kn[i], vessel.min_speed_kn, vessel.max_speed_kn))
            if deployed else None,
            cold_start=curves_cold.get(vessel.name, False),
        ))
    cargo = sum(solution.cargo_delivered_t.values())
    harit = None
    if bau_intensity and cargo > 0:
        status = state.harit_sagar.assess(
            problem.year, float(solution.objectives[1]), cargo, bau_intensity
        )
        harit = {
            "on_track": bool(status.on_track),
            "achieved_reduction": float(status.achieved_reduction),
            "required_reduction": float(status.required_reduction),
            "gap": float(status.gap),
            "baseline_intensity_t_per_t": float(status.baseline_intensity),
            "current_intensity_t_per_t": float(status.current_intensity),
            "summary": status.summary(),
            "baseline": "same deployment on MGO",
        }
    return schemas.PlanOut(
        index=index,
        fuel_energy_mj=float(solution.objectives[0]),
        emissions_tco2e=float(solution.objectives[1]),
        cost_inr=float(solution.objectives[2]),
        n_deployed=solution.n_deployed,
        fuel_mix=mix,
        cargo_delivered_t={k: float(v) for k, v in solution.cargo_delivered_t.items()},
        assignments=assignments,
        harit_sagar=harit,
    )


def _extremes(front: np.ndarray) -> dict[str, int]:
    """Cheapest, cleanest, least energy, and the knee (closest to the ideal point)."""
    if len(front) == 0:
        return {}
    out = {
        "least_energy": int(np.argmin(front[:, 0])),
        "cleanest": int(np.argmin(front[:, 1])),
        "cheapest": int(np.argmin(front[:, 2])),
    }
    span = front.max(axis=0) - front.min(axis=0)
    span[span == 0] = 1.0
    scaled = (front - front.min(axis=0)) / span
    out["balanced"] = int(np.argmin(np.linalg.norm(scaled[:, 1:], axis=1)))
    return out


def run_optimisation(state: AppState, job: Job) -> dict[str, Any]:
    request: schemas.OptimiseRequest = job.request
    problem = build_problem(state, request)
    encoding = FleetEncoding(problem)

    def on_generation(generation: int, live: QMOEAResult) -> bool | None:
        job.progress = {
            "generation": generation + 1,
            "n_generations": request.n_generations,
            "n_evaluations": live.n_evaluations,
            "archive_size": len(live.archive),
            "best_violation": live.best_violation[-1] if live.best_violation else None,
            "diversity": live.diversity[-1] if live.diversity else None,
            "elapsed_s": live.elapsed_s,
        }
        if job.cancel_event.is_set():
            return False
        return None

    result = optimise_fleet(
        problem, n_individuals=request.n_individuals, n_generations=request.n_generations,
        seed=request.seed, time_limit_s=request.time_limit_s, on_generation=on_generation,
    )

    baselines: list[BaselineResult] = []
    budget = max(result.n_evaluations, 1)
    for name in request.compare_with:
        if job.cancel_event.is_set():
            break
        if name == "greedy":
            baselines.append(greedy_fleet(problem, seed=request.seed))
        elif name == "nsga2":
            baselines.append(nsga2_fleet(problem, n_evaluations=budget, seed=request.seed))
        elif name == "qbho":
            baselines.append(qbho_fleet(problem, n_evaluations=budget, seed=request.seed))
        elif name == "random":
            baselines.append(random_search_fleet(problem, n_evaluations=budget, seed=request.seed))

    fronts = [result.front] + [b.front for b in baselines]
    non_empty = [f for f in fronts if len(f)]
    reference = reference_point(non_empty) if non_empty else np.ones(3)
    curves_cold = {
        v.name: bool(problem.speed_curve(v).cold_start) for v in problem.vessels
    } if state.model is not None else {}
    bau = _bau_problem(state, problem)

    plans = []
    order = np.argsort(result.front[:, 2]) if len(result.archive) else []
    for rank, archive_index in enumerate(order):
        solution = result.archive[int(archive_index)]
        plans.append(_plan_out(
            state, problem, rank, solution, _bau_intensity(bau, solution), curves_cold,
        ))
    sorted_front = result.front[order] if len(result.archive) else result.front

    payload = schemas.OptimiseResult(
        summary=result.summary(),
        n_evaluations=result.n_evaluations,
        generations=result.generations,
        elapsed_s=result.elapsed_s,
        time_limited=result.time_limited,
        repairs=result.repairs,
        hypervolume=float(hypervolume(result.front, reference)) if len(result.front) else 0.0,
        reference_point=[float(x) for x in reference],
        history={
            "archive_size": [float(x) for x in result.history],
            "best_violation": [float(x) for x in result.best_violation],
            "diversity": [float(x) for x in result.diversity],
        },
        plans=plans,
        extremes=_extremes(sorted_front),
        baselines=[
            schemas.BaselineOut(
                algorithm=b.algorithm, n_evaluations=b.n_evaluations, elapsed_s=b.elapsed_s,
                n_plans=len(b.archive),
                hypervolume=float(hypervolume(b.front, reference)) if len(b.front) else 0.0,
                front=[[float(x) for x in row] for row in b.front],
            )
            for b in baselines
        ],
        problem={
            **problem.as_provenance(),
            "n_bits": encoding.n_bits,
            "fuel_price_inr_per_gj": dict(problem.fuel_price_inr_per_gj),
            "routes": [r.name for r in problem.routes],
            "vessels": [v.name for v in problem.vessels],
            "grey_default_warning": [
                f for f, p in problem.fuel_pathways.items() if p == "grey"
            ],
        },
        energy_source=state.energy_source,
    )
    return payload.model_dump()


__all__ = [
    "Infeasible", "build_problem", "compare_emissions", "electrification_series",
    "meta", "predict", "run_optimisation", "scenario_spec",
]
