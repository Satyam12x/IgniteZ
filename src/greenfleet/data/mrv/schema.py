"""Declared schema for the EU MRV "Publication of information" workbook.

Layout of the published file (verified against 2018 v275 .. 2025 v57):

    sheet name   the reporting period, e.g. "2018"
    row 1        group header ("Ship", "Monitoring", ...) - merged, ignored
    row 2        blank
    row 3        column headers
    row 4+       one row per ship per reporting period

The schema is not stable across reporting periods
-------------------------------------------------
EMSA has renamed columns twice in the published series, and widened the file
substantially for the 2024 period:

    2018-2019   62 columns,  "Annual Total time spent at sea [hours]"
    2020-2023   62 columns,  "Annual time spent at sea [hours]"
    2024-2025   113 columns, "Time spent at sea [hours]", and the
                "Annual average " prefix dropped from every intensity metric

The 2024 widening is the EU ETS / FuelEU Maritime extension: ships now report
**measured CH4 and N2O** alongside CO2, plus CO2-equivalent totals. Those columns
matter well beyond ingest - they let F-02 (methane slip) and F-03 (ammonia N2O) be
validated against reported data instead of resting on provisional defaults.

Every column therefore carries ``aliases``. The primary header is tried first, so
where a period publishes both the long and short form the canonical one wins.

Why units are declared here and not parsed from the headers
-----------------------------------------------------------
The published headers use ``[m tonnes]`` for metric tonnes. ``greenfleet.units``
deliberately rejects the whole ``mt``/``tonne``-abbreviation family as ambiguous
(D-01), because in emission inventories the same token means megatonne. Rather
than widen the unit registry and lose that protection, each column states its
dimension and unit explicitly below, and the loader asserts the published header
still matches. If EMSA changes a header or a unit, ingest fails loudly instead of
silently importing numbers that are off by a factor of a million.

Column lookup is by header text, not position, so extra or reordered columns are
tolerated and a missing column names itself in the error (D-08).
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from typing import Literal

__all__ = [
    "ColumnSpec",
    "MRV_COLUMNS",
    "HEADER_ROW",
    "FIRST_DATA_ROW",
    "NULL_TOKENS",
    "COMPUTATION_ERROR_TOKENS",
    "normalise_header",
    "MrvSchemaError",
    "column_by_name",
    "column_by_header",
]

HEADER_ROW = 3
FIRST_DATA_ROW = 4

# Values EMSA uses for "this metric does not apply to this ship type".
NULL_TOKENS = frozenset({"", "n/a", "na", "not applicable", "not available", "-", "none"})

# EMSA's own spreadsheet formulas leak error strings into the published file when a
# ship reports zero distance travelled. This is not a parse failure on our side and
# not a missing value either: it means the denominator was zero, i.e. the ship was
# in scope but did not sail (berth-only, laid up, or an incomplete filing).
# Roughly 5% of rows per reporting period. Tracked separately so it stays visible
# and can be used as a feature rather than silently imputed (D-02, D-03, P-03).
COMPUTATION_ERROR_TOKENS = frozenset(
    {"division by zero!", "#div/0!", "div/0!", "#value!", "#n/a", "nan"}
)


class MrvSchemaError(ValueError):
    """The workbook does not match the declared schema."""


# NFKC already folds subscripts and superscripts (CO₂ -> CO2, m³ -> m3, CH₄ -> CH4).
# The middle dot and dashes are not decomposed, so they are handled explicitly.
_SUBSTITUTIONS = {
    "·": ".",  # middle dot, as in t · nmi
    "–": "-",  # en dash
    "—": "-",  # em dash
    " ": " ",  # non-breaking space
}


def normalise_header(header: str | None) -> str:
    """Fold a published header to a stable ASCII-ish key."""
    if header is None:
        return ""
    text = unicodedata.normalize("NFKC", str(header))
    for src, dst in _SUBSTITUTIONS.items():
        text = text.replace(src, dst)
    text = re.sub(r"\s+", " ", text)
    return text.strip().lower()


Dimension = Literal["mass", "time", "distance", "emission", "rate", "identity", "category"]


@dataclass(frozen=True, slots=True)
class ColumnSpec:
    """One column we actually consume, with its unit stated explicitly."""

    name: str
    """Internal snake_case name."""

    header: str
    """Primary published header, as it appears in row 3 (pre-normalisation)."""

    dimension: Dimension
    unit: str | None
    """Unit as published. ``None`` for identity/category columns."""

    required: bool = True
    aliases: tuple[str, ...] = ()
    """Alternative headers used by other reporting periods, most recent first."""

    notes: str = ""

    @property
    def key(self) -> str:
        return normalise_header(self.header)

    @property
    def keys(self) -> tuple[str, ...]:
        """Primary key first, then aliases, so the canonical header wins."""
        return (self.key, *(normalise_header(a) for a in self.aliases))


MRV_COLUMNS: tuple[ColumnSpec, ...] = (
    # ---- identity ----
    ColumnSpec("imo", "IMO Number", "identity", None,
               notes="P-01: the grouping key for leakage-safe splits"),
    ColumnSpec("ship_name", "Name", "identity", None, required=False),
    ColumnSpec("ship_type", "Ship type", "category", None,
               notes="D-12: mapped to a controlled vocabulary downstream"),
    ColumnSpec("reporting_period", "Reporting Period", "identity", None,
               notes="P-01: the key for time-based splits"),
    ColumnSpec("technical_efficiency_raw", "Technical efficiency", "category", None,
               required=False, notes="free text: 'EIV (12.34 gCO2/t.nm)', 'Not Applicable'"),
    ColumnSpec("ice_class", "Ice Class", "category", None, required=False),

    # ---- fuel and emissions (the physical core) ----
    ColumnSpec("total_fuel_t", "Total fuel consumption [m tonnes]", "mass", "t",
               notes="metric tonnes; see module docstring on why this is declared"),
    ColumnSpec("laden_fuel_t", "Fuel consumptions assigned to On laden [m tonnes]",
               "mass", "t", required=False,
               aliases=("Fuel consumption assigned to On laden [m tonnes]",),
               notes="P-04: laden share. Blank for many ships."),
    ColumnSpec("total_co2_t", "Total CO2 emissions [m tonnes]", "emission", "tCO2",
               notes="implied Cf = total_co2_t / total_fuel_t identifies the fuel blend"),
    ColumnSpec("co2_at_berth_t",
               "CO2 emissions which occurred within ports under a MS jurisdiction at berth "
               "[m tonnes]", "emission", "tCO2", required=False,
               notes="P-03: proxy for auxiliary/hotel load; also the shore-power target (F-07)"),

    # ---- 2024+ only: measured non-CO2 greenhouse gases ----
    # These are the reason the 2024 file has 113 columns instead of 62. They turn
    # F-02 and F-03 from assumed factors into validatable quantities.
    ColumnSpec("total_ch4_t", "Total CH4 emissions [m tonnes]", "emission", "tCO2",
               required=False,
               notes="F-02: measured methane. Validates the LNG slip assumption."),
    ColumnSpec("total_n2o_t", "Total N2O emissions [m tonnes]", "emission", "tCO2",
               required=False,
               notes="F-03: measured N2O. The ammonia N2O factor is our least certain."),
    ColumnSpec("total_co2eq_t", "Total CO2eq emissions [m tonnes]", "emission", "tCO2e",
               required=False,
               notes="F-10: EMSA's own CO2e aggregation, for cross-checking our GWP maths"),
    ColumnSpec("fuel_per_hour_t_per_h", "Fuel consumption per time spent at sea "
               "[m tonnes / hour]", "rate", "t/h", required=False,
               notes="2024+: a directly published fuel rate, independent of distance"),

    # ---- time ----
    ColumnSpec("time_at_sea_h", "Annual Total time spent at sea [hours]", "time", "h",
               aliases=("Annual time spent at sea [hours]", "Time spent at sea [hours]"),
               notes="renamed twice across the published series"),
    ColumnSpec("time_at_sea_ice_h", "Total time spent at sea through ice [hours]",
               "time", "h", required=False,
               aliases=("Time spent at sea through ice [hours]",)),

    # ---- intensity metrics: these are what let us recover distance and cargo ----
    ColumnSpec("fuel_per_distance_kg_per_nmi",
               "Annual average Fuel consumption per distance [kg / n mile]",
               "rate", "kg/nmi",
               aliases=("Fuel consumption per distance [kg / n mile]",),
               notes="distance = total_fuel_t * 1000 / this; then speed = distance / time"),
    ColumnSpec("fuel_per_work_mass_g_per_t_nmi",
               "Annual average Fuel consumption per transport work (mass) "
               "[g / m tonnes . n miles]", "rate", "g/(t.nmi)", required=False,
               aliases=("Fuel consumption per transport work (mass) "
                        "[g / m tonnes . n miles]",),
               notes="cargo carried = fuel_per_distance * 1000 / this"),
    ColumnSpec("fuel_per_work_dwt_g_per_dwt_nmi",
               "Annual average Fuel consumption per transport work (dwt) "
               "[g / dwt carried . n miles]", "rate", "g/(dwt.nmi)", required=False,
               aliases=("Fuel consumption per transport work (dwt) "
                        "[g / dwt carried . n miles]",),
               notes="fallback displacement proxy for ship types reporting on dwt"),
    ColumnSpec("fuel_per_work_pax_g_per_pax_nmi",
               "Annual average Fuel consumption per transport work (pax) "
               "[g / pax . n miles]", "rate", "g/(pax.nmi)", required=False,
               aliases=("Fuel consumption per transport work (pax) [g / pax . n miles]",)),
    ColumnSpec("fuel_per_work_volume_g_per_m3_nmi",
               "Annual average Fuel consumption per transport work (volume) "
               "[g / m3 . n miles]", "rate", "g/(m3.nmi)", required=False,
               aliases=("Fuel consumption per transport work (volume) [g / m3 . n miles]",)),
    ColumnSpec("co2_per_distance_kg_per_nmi",
               "Annual average CO2 emissions per distance [kg CO2 / n mile]",
               "rate", "kgCO2/nmi", required=False,
               aliases=("CO2 emissions per distance [kg CO2 / n mile]",),
               notes="cross-check: should equal fuel_per_distance * implied Cf"),
    ColumnSpec("co2_per_work_mass_g_per_t_nmi",
               "Annual average CO2 emissions per transport work (mass) "
               "[g CO2 / m tonnes . n miles]", "rate", "gCO2/(t.nmi)", required=False,
               aliases=("CO2 emissions per transport work (mass) "
                        "[g CO2 / m tonnes . n miles]",),
               notes="S-06: the Harit Sagar KPI shape - emissions per tonne of cargo"),

    # ---- laden-voyage variants (P-04) ----
    ColumnSpec("laden_fuel_per_distance_kg_per_nmi",
               "Fuel consumption per distance on laden voyages [kg / n mile]",
               "rate", "kg/nmi", required=False),

    # ---- context ----
    ColumnSpec("through_ice_nmi", "Through ice [n miles]", "distance", "nmi", required=False,
               aliases=("Distance through ice [n miles]",)),
    ColumnSpec("cargo_density_t_per_m3",
               "Average density of the cargo transported [m tonnes / m3]",
               "rate", "t/m3", required=False),
)


_BY_NAME = {spec.name: spec for spec in MRV_COLUMNS}

# Every header variant -> spec. A variant claimed by two specs is a schema bug.
_BY_KEY: dict[str, ColumnSpec] = {}
for _spec in MRV_COLUMNS:
    for _key in _spec.keys:
        if _key in _BY_KEY and _BY_KEY[_key] is not _spec:
            raise MrvSchemaError(
                f"header {_key!r} is claimed by both {_BY_KEY[_key].name!r} and {_spec.name!r}"
            )
        _BY_KEY[_key] = _spec

if len(_BY_NAME) != len(MRV_COLUMNS):  # pragma: no cover - guards a typo at import
    raise MrvSchemaError("duplicate internal column name in MRV_COLUMNS")


def column_by_name(name: str) -> ColumnSpec:
    try:
        return _BY_NAME[name]
    except KeyError:
        raise MrvSchemaError(
            f"unknown MRV column {name!r}; declared: {sorted(_BY_NAME)}"
        ) from None


def column_by_header(header: str) -> ColumnSpec | None:
    """Return the spec for a published header, or ``None`` if we do not consume it."""
    return _BY_KEY.get(normalise_header(header))


ALL_HEADER_KEYS: frozenset[str] = frozenset(_BY_KEY)
REQUIRED_NAMES: frozenset[str] = frozenset(s.name for s in MRV_COLUMNS if s.required)
