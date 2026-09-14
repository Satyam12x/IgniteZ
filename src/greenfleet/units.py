"""Canonical unit system and safe conversion.

Edge case D-01: every numeric value entering the system is converted to one
canonical unit on ingest. Ambiguous unit tokens are *rejected*, never guessed.

Canonical units
---------------
speed       knot (kn)
distance    nautical mile (nmi)
time        hour (h)
mass        tonne (t)
volume      cubic metre (m3)
energy      megajoule (MJ)
power       kilowatt (kW)
emission    tonne CO2-equivalent (tCO2e)
"""

from __future__ import annotations

import re
from typing import Final

__all__ = [
    "CANONICAL",
    "AmbiguousUnitError",
    "UnknownUnitError",
    "convert",
    "to_canonical",
    "parse_header_unit",
    "kn_to_ms",
    "ms_to_kn",
]


class UnitError(ValueError):
    """Base class for unit problems."""


class UnknownUnitError(UnitError):
    """The unit token is not in the registry for this dimension."""


class AmbiguousUnitError(UnitError):
    """The unit token has more than one plausible meaning. Refuse to guess (D-01)."""


CANONICAL: Final[dict[str, str]] = {
    "speed": "kn",
    "distance": "nmi",
    "time": "h",
    "mass": "t",
    "volume": "m3",
    "energy": "MJ",
    "power": "kW",
    "emission": "tCO2e",
}

# factor: value_in_unit * factor == value_in_canonical
_FACTORS: Final[dict[str, dict[str, float]]] = {
    "speed": {
        "kn": 1.0,
        "knot": 1.0,
        "knots": 1.0,
        "m/s": 1.9438444924406046,
        "mps": 1.9438444924406046,
        "km/h": 0.5399568034557235,
        "kmh": 0.5399568034557235,
        "kph": 0.5399568034557235,
        "mph": 0.8689762419006479,
    },
    "distance": {
        "nmi": 1.0,
        "nm": 1.0,
        "km": 0.5399568034557235,
        "m": 0.0005399568034557235,
        "mi": 0.8689762419006479,
    },
    "time": {
        "h": 1.0,
        "hr": 1.0,
        "hour": 1.0,
        "hours": 1.0,
        "min": 1.0 / 60.0,
        "s": 1.0 / 3600.0,
        "sec": 1.0 / 3600.0,
        "d": 24.0,
        "day": 24.0,
        "days": 24.0,
    },
    "mass": {
        "t": 1.0,
        "tonne": 1.0,
        "tonnes": 1.0,
        "kg": 1e-3,
        "g": 1e-6,
        "lb": 0.00045359237,
    },
    "volume": {
        "m3": 1.0,
        "m^3": 1.0,
        "l": 1e-3,
        "litre": 1e-3,
        "liter": 1e-3,
        "bbl": 0.158987294928,
    },
    "energy": {
        "MJ": 1.0,
        "kJ": 1e-3,
        "GJ": 1e3,
        "J": 1e-6,
        "kWh": 3.6,
        "MWh": 3600.0,
    },
    "power": {
        "kW": 1.0,
        "W": 1e-3,
        "MW": 1e3,
        "hp": 0.7456998715822702,  # mechanical horsepower
        "bhp": 0.7456998715822702,
    },
    "emission": {
        "tCO2e": 1.0,
        "tCO2": 1.0,  # only valid where the figure is CO2-only; caller must be explicit
        "kgCO2e": 1e-3,
        "gCO2e": 1e-6,
        "ktCO2e": 1e3,
    },
}

# Tokens that a maritime dataset genuinely uses to mean different things.
# Seeing one of these is a hard error: the ingest must be told which it is (D-01).
_AMBIGUOUS: Final[dict[str, str]] = {
    "mt": "'mt' means metric tonne in bunker reports but megatonne in emission inventories",
    "ton": "'ton' is metric tonne (1000 kg), long ton (1016 kg) or short ton (907 kg)",
    "tons": "'tons' is metric tonne (1000 kg), long ton (1016 kg) or short ton (907 kg)",
    "t/d": "consumption per day vs per hour must be stated as a rate dimension, not a mass unit",
    "tpd": "consumption per day vs per hour must be stated as a rate dimension, not a mass unit",
    "cbm": "'cbm' is cubic metres of fuel; convert to mass with a stated density instead",
    "kt": "'kt' is kilotonne (mass) or knot (speed)",
}


def _normalise(token: str) -> str:
    return token.strip().strip("[]()").replace("\u00b3", "^3").strip()


def convert(value: float, from_unit: str, to_unit: str, dimension: str) -> float:
    """Convert ``value`` between two units of the same ``dimension``."""
    factors = _FACTORS.get(dimension)
    if factors is None:
        raise UnknownUnitError(f"unknown dimension {dimension!r}")
    src, dst = _normalise(from_unit), _normalise(to_unit)
    for tok in (src, dst):
        if tok.lower() in _AMBIGUOUS:
            raise AmbiguousUnitError(f"{tok!r}: {_AMBIGUOUS[tok.lower()]}")
    try:
        return value * factors[src] / factors[dst]
    except KeyError as exc:
        raise UnknownUnitError(
            f"unit {exc.args[0]!r} is not registered for dimension {dimension!r}; "
            f"known: {sorted(factors)}"
        ) from None


def to_canonical(value: float, unit: str, dimension: str) -> float:
    """Convert ``value`` from ``unit`` into the canonical unit of ``dimension``."""
    return convert(value, unit, CANONICAL[dimension], dimension)


_HEADER_UNIT_RE = re.compile(r"[\(\[]([^)\]]+)[\)\]]\s*$")


def parse_header_unit(header: str, dimension: str) -> str:
    """Extract a unit from a column header such as ``"Speed over ground (kn)"``.

    Raises:
        UnknownUnitError: the header carries no parenthesised unit, or the unit
            is not registered. Silent assumption is exactly what D-01 forbids.
        AmbiguousUnitError: the unit token has multiple plausible meanings.
    """
    match = _HEADER_UNIT_RE.search(header.strip())
    if match is None:
        raise UnknownUnitError(
            f"header {header!r} states no unit; refusing to assume "
            f"{CANONICAL[dimension]!r} (edge case D-01)"
        )
    token = _normalise(match.group(1))
    if token.lower() in _AMBIGUOUS:
        raise AmbiguousUnitError(f"{token!r} in header {header!r}: {_AMBIGUOUS[token.lower()]}")
    if token not in _FACTORS[dimension]:
        raise UnknownUnitError(
            f"unit {token!r} in header {header!r} is not registered for dimension {dimension!r}"
        )
    return token


def kn_to_ms(knots: float) -> float:
    """Knots to metres per second (used by the physics model)."""
    return knots / 1.9438444924406046


def ms_to_kn(mps: float) -> float:
    """Metres per second to knots."""
    return mps * 1.9438444924406046
