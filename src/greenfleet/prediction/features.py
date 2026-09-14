"""Feature construction, with target contamination declared in code.

Why this module is defensive
----------------------------
MRV publishes intensity *ratios*, and most reconstructed quantities share a
component with the target (see docs/FINDINGS_PHASE1.md §2). The target is energy
per nautical mile, which is ``LHV x fuel_per_distance``; and

    distance_nmi    = total_fuel * 1000 / fuel_per_distance
    mean_speed_kn   = distance_nmi / time_at_sea
    mean_power_kw   = LHV * fuel_per_distance * mean_speed / 3.6
    cargo_carried_t = fuel_per_distance * 1000 / fuel_per_work_mass

every one of which contains ``fuel_per_distance``. Feed any of them to the model
and it partially reads the answer off the input. The resulting score looks
excellent and means nothing.

So contamination is not a comment, it is data: ``CONTAMINATED_BY_TARGET`` lists
them, and :func:`build_features` raises if one is requested. Adding a new derived
column means deciding which set it belongs to.

What is left is genuinely independent of the target: vessel class, the *design*
efficiency certificate (EIV/EEDI), ice class, reporting period, and time at sea.
Design efficiency carries most of the signal, since the EIV formula is a function
of deadweight and installed power - it encodes ship size without touching
consumption.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

logger = logging.getLogger(__name__)

CONFIG_PATH = Path(__file__).parent.parent / "config" / "vessel_classes.yaml"

__all__ = [
    "TARGET",
    "LEAKAGE_FREE_FEATURES",
    "CONTAMINATED_BY_TARGET",
    "ContaminatedFeatureError",
    "VesselClassMap",
    "build_features",
    "build_target",
    "load_vessel_classes",
]

TARGET = "energy_per_nmi_mj"

# Independent of fuel_per_distance, so safe to use as inputs.
LEAKAGE_FREE_FEATURES: tuple[str, ...] = (
    "vessel_class",
    "design_efficiency_gco2_per_t_nmi",
    "has_design_efficiency",
    "efficiency_metric_eiv",
    "efficiency_metric_eedi",
    "efficiency_metric_eexi",
    "has_ice_class",
    "ice_class_ordinal",
    "ice_time_share",
    "reporting_period",
    "time_at_sea_h",
)

# Ice classes ordered by capability: Baltic classes first, then IACS polar classes.
# Ordinal rather than one-hot because the ordering is real - a PC1 hull is stronger
# and draggier than an IC hull - and ordering lets a tree split on "at least IA".
ICE_CLASS_ORDINAL: dict[str, int] = {
    "IC": 1, "IB": 2, "IA": 3, "IA SUPER": 4,
    "PC7": 5, "PC6": 6, "PC5": 7, "PC4": 8, "PC3": 9, "PC2": 10, "PC1": 11,
}

# Which efficiency certificate a ship carries is itself informative: EIV is an
# *estimated* index used where no design value exists, EEDI is a design value for
# newer ships, and EEXI is the 2023 retrofit-era index. The metric type therefore
# proxies design era and data quality, independently of the value itself.
EFFICIENCY_METRICS: tuple[str, ...] = ("EIV", "EEDI", "EEXI")

# Share a component with the target. Never inputs. Kept for physics and reporting.
CONTAMINATED_BY_TARGET: frozenset[str] = frozenset(
    {
        "fuel_per_distance_kg_per_nmi",
        "co2_per_distance_kg_per_nmi",
        "distance_nmi",
        "mean_speed_kn",
        "mean_power_kw",
        "energy_mj",
        "energy_per_nmi_mj",
        "cargo_carried_t",
        "total_fuel_t",
        "total_co2_t",
        "total_co2eq_t",
        "implied_cf_t_per_t",
        "inferred_lhv_mj_per_kg",
        "admiralty_coefficient",
        "fuel_per_hour_t_per_h",
        "laden_fuel_per_distance_kg_per_nmi",
        "fuel_per_work_mass_g_per_t_nmi",
        "co2_per_work_mass_g_per_t_nmi",
    }
)


class ContaminatedFeatureError(ValueError):
    """A requested feature shares a component with the target (P-01)."""


@dataclass(slots=True)
class VesselClassMap:
    """Controlled vocabulary for vessel classes (D-12)."""

    canonical: dict[str, str] = field(default_factory=dict)
    """Lower-cased source label -> canonical class key."""

    typical_speed_kn: dict[str, float] = field(default_factory=dict)
    aux_load_share: dict[str, float] = field(default_factory=dict)
    display_name: dict[str, str] = field(default_factory=dict)
    in_mrv_scope: dict[str, bool] = field(default_factory=dict)
    fallback: str = "other"

    def resolve(self, label: object) -> str:
        """Map one source label to a canonical class, falling back with a warning."""
        if label is None or (isinstance(label, float) and np.isnan(label)):
            return self.fallback
        key = str(label).strip().lower()
        found = self.canonical.get(key)
        if found is None:
            logger.warning(
                "D-12: unknown vessel class label %r mapped to %r; "
                "add it to config/vessel_classes.yaml if it is a real class",
                label,
                self.fallback,
            )
            return self.fallback
        return found

    def resolve_series(self, series: pd.Series) -> pd.Series:
        """Vectorised resolve, warning once per unknown label rather than per row."""
        labels = series.astype("string")
        unique = labels.dropna().unique()
        mapping = {}
        unknown = []
        for label in unique:
            key = str(label).strip().lower()
            found = self.canonical.get(key)
            if found is None:
                unknown.append(label)
                found = self.fallback
            mapping[label] = found
        if unknown:
            logger.warning(
                "D-12: %d unknown vessel class label(s) mapped to %r: %s",
                len(unknown),
                self.fallback,
                sorted(unknown)[:10],
            )
        return labels.map(mapping).fillna(self.fallback).astype("string")


@lru_cache(maxsize=2)
def load_vessel_classes(path: Path | str = CONFIG_PATH) -> VesselClassMap:
    """Load the vessel-class vocabulary and reference operating points."""
    payload = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    mapping = VesselClassMap()
    for key, cfg in payload["classes"].items():
        mapping.typical_speed_kn[key] = float(cfg["typical_service_speed_kn"])
        mapping.aux_load_share[key] = float(cfg["aux_load_share"])
        mapping.display_name[key] = str(cfg["display_name"])
        mapping.in_mrv_scope[key] = bool(cfg.get("in_mrv_scope", True))
        mapping.canonical[key.lower()] = key
        mapping.canonical[str(cfg["display_name"]).strip().lower()] = key
        for alias in cfg.get("aliases", ()):
            alias_key = str(alias).strip().lower()
            existing = mapping.canonical.get(alias_key)
            if existing is not None and existing != key:
                raise ValueError(
                    f"alias {alias!r} is claimed by both {existing!r} and {key!r}"
                )
            mapping.canonical[alias_key] = key
    if mapping.fallback not in mapping.typical_speed_kn:
        raise ValueError(f"fallback class {mapping.fallback!r} is not defined")
    return mapping


def _optional_column(
    frame: pd.DataFrame, name: str, dtype: str = "float64"
) -> pd.Series:
    """Fetch a column, or an all-missing Series of the right length if absent.

    ``DataFrame.get`` returns ``None`` for a missing column, and passing ``None``
    into ``pd.to_numeric`` silently yields a *scalar* NaN rather than a Series -
    which then breaks on the next Series method with a confusing AttributeError.
    Columns are genuinely optional here (the 2018 MRV schema lacks several that
    2024 has), so this makes absence behave like all-missing.
    """
    if name in frame.columns:
        return frame[name]
    # np.nan for numeric dtypes; pd.NA cannot fill a plain float64 array.
    missing = pd.NA if dtype == "string" else np.nan
    return pd.Series(missing, index=frame.index, dtype=dtype)


def build_target(frame: pd.DataFrame) -> pd.Series:
    """Energy per nautical mile, in MJ/nmi.

    P-08: energy, not fuel mass. The same energy demand becomes a different mass
    for every candidate fuel, so the model must predict the quantity that is
    fuel-agnostic and let the fuel registry do the conversion.
    """
    required = {"inferred_lhv_mj_per_kg", "fuel_per_distance_kg_per_nmi"}
    missing = required - set(frame.columns)
    if missing:
        raise KeyError(f"build_target needs {sorted(missing)}; run derive_physics first")
    target = frame["inferred_lhv_mj_per_kg"] * frame["fuel_per_distance_kg_per_nmi"]
    return target.rename(TARGET)


def build_features(
    frame: pd.DataFrame,
    classes: VesselClassMap | None = None,
    extra_features: tuple[str, ...] = (),
) -> pd.DataFrame:
    """Build the leakage-free feature frame.

    Args:
        frame: an MRV panel that has been through ``derive_physics``.
        classes: vessel-class vocabulary; loaded from config if omitted.
        extra_features: additional columns to include. Checked against
            ``CONTAMINATED_BY_TARGET`` and rejected if unsafe.

    Raises:
        ContaminatedFeatureError: an extra feature shares a component with the
            target. The message names the shared quantity.
    """
    bad = sorted(set(extra_features) & CONTAMINATED_BY_TARGET)
    if bad:
        raise ContaminatedFeatureError(
            f"{bad} share a component with the target ({TARGET} is "
            "LHV x fuel_per_distance, and these are all derived from "
            "fuel_per_distance). Using them inflates the score without improving "
            "the model. See docs/FINDINGS_PHASE1.md section 2."
        )
    mapping = classes or load_vessel_classes()

    features = pd.DataFrame(index=frame.index)
    features["vessel_class"] = mapping.resolve_series(frame["ship_type"])

    design = pd.to_numeric(
        _optional_column(frame, "technical_efficiency_gco2_per_t_nmi"), errors="coerce"
    )
    # A non-positive certificate value is a parse artefact, not a real design value.
    design = design.where(design > 0)
    features["design_efficiency_gco2_per_t_nmi"] = design
    features["has_design_efficiency"] = design.notna().astype("int8")

    metric = (
        _optional_column(frame, "technical_efficiency_metric", "string")
        .astype("string")
        .str.upper()
        .str.strip()
    )
    for name in EFFICIENCY_METRICS:
        # A nullable-string comparison yields pd.NA for missing certificates; a
        # missing certificate simply is not that metric, so fill False.
        features[f"efficiency_metric_{name.lower()}"] = (
            (metric == name).fillna(False).astype("int8")
        )

    ice = (
        _optional_column(frame, "ice_class", "string")
        .astype("string")
        .str.upper()
        .str.strip()
    )
    features["has_ice_class"] = ice.notna().to_numpy().astype("int8")
    # No ice class means no ice strengthening, which is a real 0 on this scale,
    # not a missing value.
    features["ice_class_ordinal"] = pd.to_numeric(
        ice.map(ICE_CLASS_ORDINAL), errors="coerce"
    ).fillna(0.0).astype("float64")
    features["ice_time_share"] = (
        pd.to_numeric(_optional_column(frame, "ice_time_share"), errors="coerce")
        .fillna(0.0)
        .astype("float64")
    )

    features["reporting_period"] = pd.to_numeric(
        frame["reporting_period"], errors="coerce"
    ).astype("float64")
    features["time_at_sea_h"] = pd.to_numeric(
        frame["time_at_sea_h"], errors="coerce"
    ).astype("float64")

    for column in extra_features:
        if column not in frame.columns:
            raise KeyError(f"extra feature {column!r} is not in the frame")
        features[column] = frame[column]

    return features


def one_hot_classes(
    features: pd.DataFrame,
    classes: VesselClassMap | None = None,
) -> pd.DataFrame:
    """Expand ``vessel_class`` to a fixed one-hot block.

    The column set comes from the config, not from the data, so train and test
    matrices always have identical columns even when a class is absent from one
    side - which happens routinely under a by-vessel split for rare classes.
    """
    mapping = classes or load_vessel_classes()
    out = features.drop(columns=["vessel_class"]).copy()
    labels = features["vessel_class"].astype("string")
    for key in sorted(mapping.typical_speed_kn):
        out[f"class_{key}"] = (labels == key).astype("int8")
    return out
