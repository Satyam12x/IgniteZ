"""Reconstruct physical quantities from MRV's published intensity metrics.

MRV publishes annual aggregates and intensity ratios, not operational variables.
The ratios are algebraically invertible, which is what makes a physics-informed
model possible from this dataset at all:

    distance [nmi]      = total_fuel [t] * 1000 / fuel_per_distance [kg/nmi]
    mean speed [kn]     = distance [nmi] / time_at_sea [h]
    cargo carried [t]   = fuel_per_distance [kg/nmi] * 1000
                          / fuel_per_work_mass [g/(t.nmi)]
    implied Cf [t/t]    = total_CO2 [t] / total_fuel [t]
    energy [MJ]         = total_fuel [t] * 1000 * LHV(blend inferred from Cf)
    mean power [kW]     = energy [MJ] / time_at_sea [h] / 3.6

Mean power is the regression target. It is a physical quantity, it is comparable
across fuels (P-08), and the Admiralty relation P ~ displacement^(2/3) * V^3 gives
it a genuine physics prior.

Aggregation bias, stated plainly
--------------------------------
The cube law applies to instantaneous speed, but MRV gives annual means. Since
E[V^3] >= E[V]^3 (Jensen), fitting a cube law to annual means underestimates the
speed exponent and absorbs the residual into the intercept. We therefore fit and
report the *annual-average* relation as exactly that, and never claim it is a
per-voyage law. Per-voyage granularity needs noon reports or AIS, which is what
port/operator data (e.g. a Mormugao pilot) would supply. The tug case study is
physics-derived and labelled separately (P-05).
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from greenfleet.config.loader import FuelRegistry, load_fuel_registry

logger = logging.getLogger(__name__)

__all__ = ["derive_physics", "QualityReport", "PLAUSIBLE", "infer_blend_lhv"]

# Physical plausibility bounds. Anything outside is flagged, never silently kept
# and never silently dropped (D-03).
PLAUSIBLE: dict[str, tuple[float, float]] = {
    # A marine hydrocarbon cannot exceed roughly 3.25 tCO2 per tonne of fuel;
    # pure carbon would be 3.67. Above 3.30 is a filing error, not a fuel.
    "implied_cf_t_per_t": (2.40, 3.30),
    # Merchant service speeds. A bulk carrier at 60 kn is the D-03 example.
    "mean_speed_kn": (1.0, 30.0),
    # Hours in a leap year.
    "time_at_sea_h": (0.0, 8784.0),
    "total_fuel_t": (0.0, 500_000.0),
    "mean_power_kw": (0.0, 120_000.0),
    # The largest ship afloat is roughly 400,000 DWT, so a reconstructed mean
    # cargo above 600,000 t is a filing error. The lower bound excludes
    # ballast-only years, where a displacement^(2/3) term would collapse to zero
    # and predict zero power.
    "cargo_carried_t": (1.0, 600_000.0),
}

_EFFICIENCY_RE = re.compile(
    r"(?P<metric>EIV|EEDI|EEXI)\s*\(\s*(?P<value>[\d.]+)\s*g", re.IGNORECASE
)

# Reference fuels for inferring the blend from the implied Cf. Ordered by Cf.
_BLEND_LADDER = ("lng", "hfo", "mgo")


def _parse_technical_efficiency(series: pd.Series) -> tuple[pd.Series, pd.Series]:
    """Split ``'EIV (45.57 gCO2/t.nm)'`` into a metric name and a value."""
    text = series.fillna("").astype(str)
    extracted = text.str.extract(_EFFICIENCY_RE)
    metric = extracted["metric"].str.upper().astype("string")
    value = pd.to_numeric(extracted["value"], errors="coerce")
    return metric, value


def infer_blend_lhv(
    implied_cf: pd.Series | np.ndarray,
    registry: FuelRegistry | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Infer a two-fuel blend from the implied CO2 factor, and return its LHV.

    MRV's public file reports total fuel and total CO2 but not the per-fuel
    breakdown, so the fuel mix is unobserved. The implied Cf pins it down for a
    two-fuel blend: solving ``Cf = w*Cf_a + (1-w)*Cf_b`` for the mass fraction
    ``w`` gives ``LHV = w*LHV_a + (1-w)*LHV_b``.

    The reference pair is chosen by where Cf falls on the LNG -> HFO -> MGO
    ladder. Values outside the ladder are clamped to the nearest pure fuel; the
    caller is expected to have flagged them already via ``PLAUSIBLE``.

    Returns:
        ``(lhv_mj_per_kg, mgo_equivalent_mass_fraction)``. The second array is
        the inferred mass fraction of the *higher-Cf* reference fuel, kept for
        diagnostics.
    """
    reg = registry or load_fuel_registry()
    cf_values = np.asarray(implied_cf, dtype="float64")
    cf_ref = np.array([reg.get(f).cf_co2_t_per_t for f in _BLEND_LADDER])
    lhv_ref = np.array([reg.lhv(f) for f in _BLEND_LADDER])

    lhv = np.full(cf_values.shape, np.nan)
    weight = np.full(cf_values.shape, np.nan)

    for lower in range(len(_BLEND_LADDER) - 1):
        cf_low, cf_high = cf_ref[lower], cf_ref[lower + 1]
        lhv_low, lhv_high = lhv_ref[lower], lhv_ref[lower + 1]
        # The lowest band is open below (pure LNG and cleaner), the rest are
        # closed at their lower reference fuel.
        if lower == 0:  # noqa: SIM108 - clearer than a ternary over two masks
            in_band = cf_values < cf_high
        else:
            in_band = cf_values >= cf_low
        band = in_band & np.isfinite(cf_values)
        if not band.any():
            continue
        fraction = np.clip((cf_values[band] - cf_low) / (cf_high - cf_low), 0.0, 1.0)
        lhv[band] = lhv_low + fraction * (lhv_high - lhv_low)
        weight[band] = fraction

    return lhv, weight


@dataclass
class QualityReport:
    """Counts behind the data-quality report (D-02, D-03)."""

    rows: int = 0
    rows_usable: int = 0
    flag_counts: dict[str, int] = field(default_factory=dict)
    no_distance_reported: int = 0
    no_cargo_reported: int = 0

    def as_dict(self) -> dict[str, object]:
        return {
            "rows": self.rows,
            "rows_usable": self.rows_usable,
            "usable_fraction": round(self.rows_usable / self.rows, 4) if self.rows else 0.0,
            "no_distance_reported": self.no_distance_reported,
            "no_cargo_reported": self.no_cargo_reported,
            "flag_counts": dict(sorted(self.flag_counts.items(), key=lambda kv: -kv[1])),
        }


def derive_physics(
    panel: pd.DataFrame,
    registry: FuelRegistry | None = None,
) -> tuple[pd.DataFrame, QualityReport]:
    """Add reconstructed physical quantities and quality flags to an MRV panel.

    Nothing is dropped and nothing is imputed. Rows that fail a plausibility test
    get a ``flag_*`` column set, and ``is_usable`` summarises them, so the caller
    decides what to exclude and the count is reportable.
    """
    reg = registry or load_fuel_registry()
    out = panel.copy()
    report = QualityReport(rows=len(out))

    fuel_t = out["total_fuel_t"]
    fuel_per_nmi = out["fuel_per_distance_kg_per_nmi"]
    time_h = out["time_at_sea_h"]

    # --- distance and speed -------------------------------------------------
    with np.errstate(divide="ignore", invalid="ignore"):
        out["distance_nmi"] = np.where(
            fuel_per_nmi > 0, fuel_t * 1000.0 / fuel_per_nmi, np.nan
        )
        out["mean_speed_kn"] = np.where(
            time_h > 0, out["distance_nmi"] / time_h, np.nan
        )

    # --- cargo actually carried --------------------------------------------
    # Ship types report transport work on different bases; take the mass basis
    # where available, then dwt, and record which basis was used.
    with np.errstate(divide="ignore", invalid="ignore"):
        cargo_mass = np.where(
            out["fuel_per_work_mass_g_per_t_nmi"] > 0,
            fuel_per_nmi * 1000.0 / out["fuel_per_work_mass_g_per_t_nmi"],
            np.nan,
        )
        cargo_dwt = np.where(
            out["fuel_per_work_dwt_g_per_dwt_nmi"] > 0,
            fuel_per_nmi * 1000.0 / out["fuel_per_work_dwt_g_per_dwt_nmi"],
            np.nan,
        )
    out["cargo_carried_t"] = np.where(np.isfinite(cargo_mass), cargo_mass, cargo_dwt)
    out["cargo_basis"] = np.where(
        np.isfinite(cargo_mass), "mass", np.where(np.isfinite(cargo_dwt), "dwt", "none")
    )

    # --- fuel blend, energy, power -----------------------------------------
    with np.errstate(divide="ignore", invalid="ignore"):
        out["implied_cf_t_per_t"] = np.where(fuel_t > 0, out["total_co2_t"] / fuel_t, np.nan)
    lhv, high_cf_fraction = infer_blend_lhv(out["implied_cf_t_per_t"], reg)
    out["inferred_lhv_mj_per_kg"] = lhv
    out["inferred_blend_fraction"] = high_cf_fraction
    out["energy_mj"] = fuel_t * 1000.0 * lhv
    with np.errstate(divide="ignore", invalid="ignore"):
        # MJ/h -> kW: 1 MJ/h = 1e6 J / 3600 s = 277.8 W
        out["mean_power_kw"] = np.where(time_h > 0, out["energy_mj"] / time_h / 3.6, np.nan)
        out["energy_per_nmi_mj"] = np.where(
            out["distance_nmi"] > 0, out["energy_mj"] / out["distance_nmi"], np.nan
        )

    # --- operating-profile shares ------------------------------------------
    with np.errstate(divide="ignore", invalid="ignore"):
        # P-03 / F-07: at-berth CO2 share is the auxiliary-load and shore-power proxy.
        out["berth_co2_share"] = np.where(
            out["total_co2_t"] > 0, out["co2_at_berth_t"] / out["total_co2_t"], np.nan
        )
        # P-04: laden voyages burn more at the same speed.
        out["laden_fuel_share"] = np.where(fuel_t > 0, out["laden_fuel_t"] / fuel_t, np.nan)
        out["ice_time_share"] = np.where(
            time_h > 0, out["time_at_sea_ice_h"].fillna(0.0) / time_h, np.nan
        )

    out["technical_efficiency_metric"], out["technical_efficiency_gco2_per_t_nmi"] = (
        _parse_technical_efficiency(out["technical_efficiency_raw"])
    )
    out["has_ice_class"] = out["ice_class"].notna()

    # --- plausibility flags (D-03) -----------------------------------------
    for column, (low, high) in PLAUSIBLE.items():
        values = out[column]
        flag = f"flag_{column}_implausible"
        out[flag] = values.notna() & ((values < low) | (values > high))
        count = int(out[flag].sum())
        if count:
            report.flag_counts[flag] = count

    out["flag_zero_fuel_while_sailing"] = (time_h > 0) & (fuel_t <= 0)
    out["flag_negative_fuel"] = fuel_t < 0
    out["flag_co2_fuel_inconsistent"] = out["flag_implied_cf_t_per_t_implausible"]
    for flag in ("flag_zero_fuel_while_sailing", "flag_negative_fuel"):
        count = int(out[flag].sum())
        if count:
            report.flag_counts[flag] = count

    report.no_distance_reported = int(out["distance_nmi"].isna().sum())
    report.no_cargo_reported = int((out["cargo_basis"] == "none").sum())

    flag_columns = [c for c in out.columns if c.startswith("flag_")]
    out["is_usable"] = (
        ~out[flag_columns].any(axis=1)
        & out["mean_power_kw"].notna()
        & out["mean_speed_kn"].notna()
        & (out["mean_power_kw"] > 0)
    )
    report.rows_usable = int(out["is_usable"].sum())

    logger.info(
        "derived physics for %d rows; %d usable (%.1f%%)",
        report.rows,
        report.rows_usable,
        100.0 * report.rows_usable / max(report.rows, 1),
    )
    return out, report
