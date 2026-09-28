"""Request and response models for the decision-support API.

Requests are strict (unknown fields rejected) so a typo in a scenario file fails
loudly instead of silently taking a default. Responses carry provenance - which
model, which data version, which factors - because a number a port authority
cannot trace is a number it cannot act on.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


# ---- prediction -----------------------------------------------------------


class PredictRequest(_Strict):
    vessel_class: str = Field(description="vessel class key or alias, e.g. 'bulk_carrier'")
    design_efficiency_gco2_per_t_nmi: float | None = Field(
        default=None, gt=0,
        description="EEDI/EIV/EEXI certificate value; omit for a cold-start vessel",
    )
    speeds_kn: list[float] | None = Field(
        default=None,
        description="speeds to evaluate; defaults to a sweep around the class reference",
    )
    load_ratio: float = Field(default=1.0, gt=0, le=2.0)
    year: int = Field(default=2030, ge=2018, le=2050)
    time_at_sea_h: float = Field(default=6000.0, gt=0, le=8760)
    fuels: list[str] = Field(
        default_factory=lambda: ["mgo", "lng", "methanol", "ammonia", "hydrogen"]
    )

    @field_validator("speeds_kn")
    @classmethod
    def _speeds_non_negative(cls, value: list[float] | None) -> list[float] | None:
        if value is not None and any(v < 0 for v in value):
            raise ValueError("speeds must be non-negative")
        return value


class CurvePoint(BaseModel):
    speed_kn: float
    energy_per_nmi_mj: float | None
    p10_energy_per_nmi_mj: float | None
    p90_energy_per_nmi_mj: float | None
    power_kw: float
    extrapolating: bool
    fuel_t_per_1000_nmi: dict[str, float | None]


class PredictResponse(BaseModel):
    vessel_class: str
    vessel_class_display: str
    cold_start: bool
    cold_start_reason: str | None
    reference_speed_kn: float
    reference_intensity_mj_per_nmi: float
    aux_share: float
    optimal_speed_kn: float
    points: list[CurvePoint]
    model: dict[str, Any]


# ---- emissions ------------------------------------------------------------


class FuelOption(_Strict):
    fuel: str
    pathway: str | None = None
    engine: str | None = None


class EmissionsCompareRequest(_Strict):
    shaft_energy_mj: float = Field(
        default=1.2e7, gt=0, description="12 TJ is a typical harbour tug year"
    )
    year: int = Field(default=2030, ge=2018, le=2050)
    options: list[FuelOption] | None = None
    baseline: FuelOption = Field(default_factory=lambda: FuelOption(fuel="mgo", pathway="fossil"))
    include_pilot_fuel: bool = True


class EmissionRow(BaseModel):
    fuel: str
    pathway: str
    label: str
    breakdown: dict[str, float]
    total_t: float
    stack_only_t: float
    vs_baseline: float
    stack_only_vs_baseline: float
    verdict_flips: bool
    intensity_gco2e_per_mj: float
    fuel_mass_t: float
    local_pollutants_kg: dict[str, float]
    confidence: str
    notes: list[str]


class EmissionsCompareResponse(BaseModel):
    shaft_energy_mj: float
    year: int
    grid_factor_t_per_mwh: float
    gwp_set: str
    rows: list[EmissionRow]


# ---- fleet optimisation ---------------------------------------------------


class VesselSpec(_Strict):
    name: str
    vessel_class: str
    capacity_t: float = Field(gt=0)
    design_efficiency_gco2_per_t_nmi: float | None = Field(default=None, gt=0)
    min_speed_kn: float = Field(gt=0)
    max_speed_kn: float = Field(gt=0)
    tank_volume_m3: float = Field(gt=0)
    annual_cost_inr: float = Field(default=0.0, ge=0)
    available_hours: float = Field(default=7446.0, gt=0, le=8760)
    compatible_routes: list[str] | None = None
    compatible_fuels: list[str] | None = None


class RouteSpec(_Strict):
    name: str
    ports: list[str] = Field(min_length=2)
    distance_nmi: float = Field(gt=0)
    annual_demand_t: float = Field(ge=0)
    max_transit_hours: float | None = Field(default=None, gt=0)


class OptimiseRequest(_Strict):
    scenario: Literal["coastal_2047", "coastal", "gttp_tug", "tiny", "custom"] = "coastal_2047"
    vessels: list[VesselSpec] | None = Field(
        default=None, description="required when scenario='custom'"
    )
    routes: list[RouteSpec] | None = None
    year: int | None = Field(default=None, ge=2024, le=2050,
                             description="unset keeps the built-in scenario's year")
    fuels: list[str] | None = Field(default=None, description="fuels the optimizer may assign")
    fuel_pathways: dict[str, str] | None = None
    fuel_price_inr_per_gj: dict[str, float] | None = None
    carbon_price_inr_per_t: float | None = Field(default=None, ge=0)
    emission_cap_tco2e: float | None = Field(default=None, gt=0)
    n_individuals: int = Field(default=20, ge=2, le=200)
    n_generations: int = Field(default=60, ge=1, le=2000)
    seed: int = Field(default=1000, ge=0)
    time_limit_s: float | None = Field(default=None, gt=0, le=600)
    compare_with: list[Literal["nsga2", "qbho", "greedy", "random"]] = Field(default_factory=list)


class JobProgress(BaseModel):
    generation: int
    n_generations: int
    n_evaluations: int
    archive_size: int
    best_violation: float | None
    diversity: float | None
    elapsed_s: float


class VesselAssignment(BaseModel):
    name: str
    vessel_class: str
    deployed: bool
    route: str | None
    fuel: str | None
    pathway: str | None
    speed_kn: float | None
    cold_start: bool


class PlanOut(BaseModel):
    index: int
    fuel_energy_mj: float
    emissions_tco2e: float
    cost_inr: float
    n_deployed: int
    fuel_mix: dict[str, int]
    cargo_delivered_t: dict[str, float]
    assignments: list[VesselAssignment]
    harit_sagar: dict[str, Any] | None


class BaselineOut(BaseModel):
    algorithm: str
    n_evaluations: int
    elapsed_s: float
    n_plans: int
    hypervolume: float
    front: list[list[float]]


class OptimiseResult(BaseModel):
    summary: str
    n_evaluations: int
    generations: int
    elapsed_s: float
    time_limited: bool
    repairs: int
    hypervolume: float
    reference_point: list[float]
    history: dict[str, list[float]]
    plans: list[PlanOut]
    extremes: dict[str, int]
    baselines: list[BaselineOut]
    problem: dict[str, Any]
    energy_source: str


class JobStatus(BaseModel):
    job_id: str
    status: Literal["queued", "running", "done", "failed", "cancelled"]
    request: OptimiseRequest
    progress: JobProgress | None = None
    result: OptimiseResult | None = None
    error: str | None = None
