"""Typed access to the fuel / emission-factor configuration.

F-09: every emission factor lives in YAML, never in Python. This module is the only
way the rest of the codebase reads those numbers, so a factor change is always a
config change.

P-08: the prediction model predicts **energy** (MJ). Fuel mass is derived here via
each fuel's lower heating value. Predicting fuel mass directly across fuel types is
wrong, because methanol carries roughly half the energy per kg of diesel.
"""

from __future__ import annotations

import warnings
from functools import lru_cache
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

CONFIG_DIR = Path(__file__).parent
FUELS_PATH = CONFIG_DIR / "fuels.yaml"

Confidence = Literal["high", "medium", "provisional"]

__all__ = [
    "FuelRegistry",
    "FuelProperties",
    "Pathway",
    "GwpSet",
    "ProvisionalFactorWarning",
    "load_fuel_registry",
]


class ProvisionalFactorWarning(UserWarning):
    """Raised when a provisional factor is used. Never publish a number that warns."""


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class Pathway(_Strict):
    """One production pathway for a fuel (grey / blue / green / bio / e-fuel)."""

    label: str
    wtt_gco2e_per_mj: float = Field(ge=0.0)
    source: str = Field(min_length=3)
    confidence: Confidence
    biogenic_carbon: bool = False


class GwpSet(_Strict):
    """Global warming potentials for one assessment report and horizon (F-10)."""

    horizon_years: int = Field(gt=0)
    source: str
    confidence: Confidence
    co2: float
    ch4_fossil: float
    ch4_biogenic: float
    n2o: float


class FuelProperties(_Strict):
    """Physical and emission properties of one fuel."""

    display_name: str
    lhv_mj_per_kg: float | None = Field(default=None, gt=0.0)
    density_kg_per_m3: float | None = Field(default=None, gt=0.0)
    cf_co2_t_per_t: float = Field(ge=0.0)
    source_ttw: str
    confidence_ttw: Confidence
    pathways: dict[str, Pathway] = Field(default_factory=dict)

    tank_system_factor: float = Field(default=1.0, ge=1.0)
    source_tank: str | None = None
    confidence_tank: Confidence | None = None

    n2o_g_per_mj: float = Field(default=0.0, ge=0.0)
    ch4_g_per_mj: float = Field(default=0.0, ge=0.0)
    nox_g_per_kg: float = Field(default=0.0, ge=0.0)
    sox_g_per_kg: float = Field(default=0.0, ge=0.0)
    pm_g_per_kg: float = Field(default=0.0, ge=0.0)
    source_local: str | None = None
    confidence_local: Confidence | None = None
    source_non_co2: str | None = None
    confidence_non_co2: Confidence | None = None

    methane_slip_pct_of_fuel_mass: dict[str, float] = Field(default_factory=dict)
    default_engine: str | None = None
    source_slip: str | None = None
    confidence_slip: Confidence | None = None
    measured_slip_percentiles: dict[str, float] = Field(default_factory=dict)
    """Observed slip distribution, where measured data exists (F-02)."""

    pilot_fuel: dict[str, Any] | None = None

    is_energy_carrier: bool = False
    conversion_efficiency: float = Field(default=1.0, gt=0.0, le=1.0)
    """Shaft energy delivered per unit of fuel/electrical energy input."""

    charge_discharge_efficiency: float | None = Field(default=None, gt=0.0, le=1.0)
    source_efficiency: str | None = None
    confidence_efficiency: Confidence | None = None

    @model_validator(mode="after")
    def _check_internal_consistency(self) -> FuelProperties:
        if not self.is_energy_carrier and self.lhv_mj_per_kg is None:
            raise ValueError(f"{self.display_name}: a combustible fuel needs lhv_mj_per_kg")
        if self.methane_slip_pct_of_fuel_mass and self.default_engine is None:
            raise ValueError(
                f"{self.display_name}: methane slip is engine-specific (F-02), so "
                "default_engine must name one of the listed engine technologies"
            )
        if (
            self.default_engine is not None
            and self.methane_slip_pct_of_fuel_mass
            and self.default_engine not in self.methane_slip_pct_of_fuel_mass
        ):
            raise ValueError(
                f"{self.display_name}: default_engine {self.default_engine!r} is not in "
                f"{sorted(self.methane_slip_pct_of_fuel_mass)}"
            )
        if self.pilot_fuel is not None:
            missing = {"fuel", "energy_fraction"} - set(self.pilot_fuel)
            if missing:
                raise ValueError(f"{self.display_name}: pilot_fuel missing {sorted(missing)}")
            fraction = float(self.pilot_fuel["energy_fraction"])
            if not 0.0 <= fraction < 1.0:
                raise ValueError(
                    f"{self.display_name}: pilot energy_fraction must be in [0, 1), got {fraction}"
                )
        return self

    @property
    def volumetric_energy_mj_per_litre(self) -> float | None:
        """Energy per litre of *neat* fuel."""
        if self.lhv_mj_per_kg is None or self.density_kg_per_m3 is None:
            return None
        return self.lhv_mj_per_kg * self.density_kg_per_m3 / 1000.0

    @property
    def installed_energy_mj_per_litre(self) -> float | None:
        """Energy per litre of *installed tank system* volume.

        This, not the neat density, is what limits range and eats cargo space
        (F-04, F-05). Hydrogen loses roughly a further 3x here.
        """
        neat = self.volumetric_energy_mj_per_litre
        return None if neat is None else neat / self.tank_system_factor


class FuelRegistry:
    """Loaded fuel configuration with unit-safe derived quantities."""

    def __init__(self, payload: dict[str, Any], source_path: Path | None = None):
        self._raw = payload
        self.source_path = source_path
        self.schema_version = int(payload["meta"]["schema_version"])
        self.gwp_set_name: str = payload["meta"]["gwp_set"]
        self.gwp_sets = {name: GwpSet(**cfg) for name, cfg in payload["gwp"].items()}
        if self.gwp_set_name not in self.gwp_sets:
            raise ValueError(
                f"meta.gwp_set {self.gwp_set_name!r} is not defined; "
                f"available: {sorted(self.gwp_sets)}"
            )
        self.fuels = {name: FuelProperties(**cfg) for name, cfg in payload["fuels"].items()}
        self._validate_cross_references()

    def _validate_cross_references(self) -> None:
        for name, fuel in self.fuels.items():
            if fuel.pilot_fuel is None:
                continue
            pilot = fuel.pilot_fuel["fuel"]
            if pilot not in self.fuels:
                raise ValueError(f"fuel {name!r} names unknown pilot fuel {pilot!r}")
            if pilot == name:
                raise ValueError(f"fuel {name!r} lists itself as its own pilot fuel")
            pilot_pathway = fuel.pilot_fuel.get("pathway")
            if pilot_pathway is not None and pilot_pathway not in self.fuels[pilot].pathways:
                raise ValueError(
                    f"fuel {name!r} names pilot pathway {pilot_pathway!r}, which "
                    f"{pilot!r} does not have "
                    f"({sorted(self.fuels[pilot].pathways)})"
                )

    # ---- construction -----------------------------------------------------

    @classmethod
    def from_yaml(cls, path: Path | str = FUELS_PATH) -> FuelRegistry:
        path = Path(path)
        payload = yaml.safe_load(path.read_text(encoding="utf-8"))
        return cls(payload, source_path=path)

    # ---- lookup -----------------------------------------------------------

    @property
    def gwp(self) -> GwpSet:
        """The active GWP set."""
        return self.gwp_sets[self.gwp_set_name]

    def get(self, fuel: str) -> FuelProperties:
        try:
            return self.fuels[fuel]
        except KeyError:
            raise KeyError(
                f"unknown fuel {fuel!r}; configured: {sorted(self.fuels)}"
            ) from None

    def input_energy_mj(self, shaft_energy_mj: float, fuel: str) -> float:
        """Fuel (or electrical) energy needed to deliver ``shaft_energy_mj`` of work.

        This is the conversion that makes fuels and batteries comparable. A marine
        diesel needs about 2.2 MJ of fuel per MJ of shaft work; an electric
        drivetrain needs about 1.1 MJ of electricity. Comparing them on input
        energy instead would understate electrification by roughly a factor of two.
        """
        if shaft_energy_mj < 0:
            raise ValueError(f"shaft energy must be non-negative, got {shaft_energy_mj}")
        return shaft_energy_mj / self.get(fuel).conversion_efficiency

    def methane_slip_fraction(self, fuel: str, engine: str | None = None) -> float:
        """Fraction of fuel mass escaping unburnt as CH4 (F-02).

        Args:
            fuel: fuel key.
            engine: engine technology. When omitted, the fuel's ``default_engine``
                is used - which for LNG is the *measured fleet* value, not the
                worst-case regulatory default. Assuming the worst engine for an
                unknown vessel overstates LNG's methane penalty by ~1.8x.
        """
        props = self.get(fuel)
        if not props.methane_slip_pct_of_fuel_mass:
            return 0.0
        key = engine or props.default_engine
        try:
            return float(props.methane_slip_pct_of_fuel_mass[str(key)]) / 100.0
        except KeyError:
            raise KeyError(
                f"unknown engine {key!r} for fuel {fuel!r}; configured: "
                f"{sorted(props.methane_slip_pct_of_fuel_mass)}"
            ) from None

    def methane_slip_range(self, fuel: str) -> tuple[float, float, float] | None:
        """Measured (p25, p50, p95) slip fractions, if a distribution was observed."""
        pct = self.get(fuel).measured_slip_percentiles
        if not pct:
            return None
        return (pct["p25"] / 100.0, pct["p50"] / 100.0, pct["p95"] / 100.0)

    def pathway(self, fuel: str, pathway: str) -> Pathway:
        """Look up one production pathway, e.g. ``pathway("ammonia", "grey")``."""
        props = self.get(fuel)
        try:
            found = props.pathways[pathway]
        except KeyError:
            raise KeyError(
                f"fuel {fuel!r} has no pathway {pathway!r}; "
                f"available: {sorted(props.pathways)}"
            ) from None
        if found.confidence == "provisional":
            warnings.warn(
                f"{fuel}/{pathway} well-to-tank factor is provisional "
                f"({found.wtt_gco2e_per_mj} gCO2e/MJ, {found.source}). "
                "Do not publish this number without verification.",
                ProvisionalFactorWarning,
                stacklevel=2,
            )
        return found

    # ---- P-08: energy <-> mass -------------------------------------------

    def lhv(self, fuel: str) -> float:
        """Lower heating value in MJ/kg."""
        props = self.get(fuel)
        if props.lhv_mj_per_kg is None:
            raise ValueError(
                f"{fuel!r} is an energy carrier with no heating value; "
                "use the grid emission factor instead of a fuel mass"
            )
        return props.lhv_mj_per_kg

    def energy_to_mass_tonnes(self, energy_mj: float, fuel: str) -> float:
        """Convert predicted energy (MJ) to fuel mass (tonnes) for a given fuel.

        This is the P-08 conversion: the model predicts energy once, and each
        candidate fuel turns that same energy into a different mass.
        """
        if energy_mj < 0:
            raise ValueError(f"energy must be non-negative, got {energy_mj}")
        return energy_mj / self.lhv(fuel) / 1000.0

    def mass_tonnes_to_energy(self, mass_t: float, fuel: str) -> float:
        """Convert fuel mass (tonnes) to energy (MJ)."""
        if mass_t < 0:
            raise ValueError(f"mass must be non-negative, got {mass_t}")
        return mass_t * 1000.0 * self.lhv(fuel)

    def tank_volume_m3(self, energy_mj: float, fuel: str) -> float:
        """Installed tank volume (m3) needed to carry ``energy_mj`` of ``fuel``.

        Drives F-04 (hydrogen range limit) and F-05 (tank volume displaces cargo).
        """
        props = self.get(fuel)
        density = props.installed_energy_mj_per_litre
        if density is None:
            raise ValueError(f"{fuel!r} has no volumetric density (energy carrier)")
        return energy_mj / density / 1000.0

    def as_provenance(self) -> dict[str, Any]:
        """Digestable snapshot for the audit trail (X-05)."""
        return {
            "schema_version": self.schema_version,
            "gwp_set": self.gwp_set_name,
            "source_path": str(self.source_path) if self.source_path else None,
            "fuels": sorted(self.fuels),
        }


@lru_cache(maxsize=4)
def load_fuel_registry(path: Path | str = FUELS_PATH) -> FuelRegistry:
    """Load and cache the fuel registry."""
    return FuelRegistry.from_yaml(path)
