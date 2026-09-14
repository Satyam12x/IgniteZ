"""Unit-system tests. Maps to edge case D-01 (unit mismatch on ingest)."""

from __future__ import annotations

import math

import pytest

from greenfleet.units import (
    CANONICAL,
    AmbiguousUnitError,
    UnknownUnitError,
    convert,
    kn_to_ms,
    ms_to_kn,
    parse_header_unit,
    to_canonical,
)

pytestmark = pytest.mark.edgecase


class TestCanonicalConversion:
    def test_speed_km_h_to_knots(self):
        # 1 knot == 1.852 km/h exactly, by definition
        assert to_canonical(1.852, "km/h", "speed") == pytest.approx(1.0, rel=1e-9)

    def test_speed_ms_to_knots(self):
        assert to_canonical(1.0, "m/s", "speed") == pytest.approx(1.9438444924, rel=1e-9)

    def test_mass_kg_to_tonnes(self):
        assert to_canonical(1500.0, "kg", "mass") == pytest.approx(1.5)

    def test_energy_kwh_to_mj(self):
        # 1 kWh == 3.6 MJ exactly
        assert to_canonical(1.0, "kWh", "energy") == pytest.approx(3.6)

    def test_power_hp_to_kw(self):
        assert to_canonical(1.0, "hp", "power") == pytest.approx(0.74569987, rel=1e-6)

    def test_canonical_unit_is_identity(self):
        for dimension, unit in CANONICAL.items():
            assert to_canonical(42.0, unit, dimension) == pytest.approx(42.0)

    @pytest.mark.parametrize(
        ("dimension", "unit"),
        [
            ("speed", "km/h"),
            ("speed", "m/s"),
            ("mass", "kg"),
            ("energy", "kWh"),
            ("distance", "km"),
            ("time", "min"),
            ("power", "hp"),
        ],
    )
    def test_round_trip_is_lossless(self, dimension, unit):
        """Converting out and back must return the original value."""
        canonical_unit = CANONICAL[dimension]
        value = 37.5
        there = convert(value, unit, canonical_unit, dimension)
        back = convert(there, canonical_unit, unit, dimension)
        assert back == pytest.approx(value, rel=1e-12)


class TestPerDayVsPerHour:
    """D-01 explicitly calls out consumption per day vs per hour."""

    def test_days_to_hours(self):
        assert to_canonical(1.0, "day", "time") == pytest.approx(24.0)

    def test_a_daily_rate_converted_as_if_hourly_is_off_by_24(self):
        # Guards the reason D-01 exists: silent misreading is a 24x error.
        assert to_canonical(1.0, "day", "time") / to_canonical(1.0, "h", "time") == 24.0


class TestRejectsRatherThanGuesses:
    def test_unknown_unit_raises(self):
        with pytest.raises(UnknownUnitError, match="not registered"):
            to_canonical(1.0, "furlongs", "distance")

    def test_unknown_dimension_raises(self):
        with pytest.raises(UnknownUnitError, match="unknown dimension"):
            convert(1.0, "kg", "t", "luminosity")

    @pytest.mark.parametrize("token", ["mt", "ton", "tons", "kt", "tpd"])
    def test_ambiguous_mass_tokens_are_rejected(self, token):
        """D-01: reject or flag ambiguous units instead of guessing silently."""
        with pytest.raises(AmbiguousUnitError):
            to_canonical(1.0, token, "mass")

    def test_ambiguous_error_explains_the_ambiguity(self):
        with pytest.raises(AmbiguousUnitError) as excinfo:
            to_canonical(1.0, "mt", "mass")
        assert "metric tonne" in str(excinfo.value)


class TestHeaderParsing:
    @pytest.mark.parametrize(
        ("header", "dimension", "expected"),
        [
            ("Speed over ground (kn)", "speed", "kn"),
            ("Average speed [km/h]", "speed", "km/h"),
            ("Total fuel consumption (t)", "mass", "t"),
            ("Annual energy (MWh)", "energy", "MWh"),
            ("Distance travelled (nmi)", "distance", "nmi"),
        ],
    )
    def test_extracts_unit(self, header, dimension, expected):
        assert parse_header_unit(header, dimension) == expected

    def test_header_without_unit_is_rejected(self):
        """A bare 'Speed' column must not be assumed to be knots."""
        with pytest.raises(UnknownUnitError, match="states no unit"):
            parse_header_unit("Speed", "speed")

    def test_header_with_ambiguous_unit_is_rejected(self):
        with pytest.raises(AmbiguousUnitError):
            parse_header_unit("Fuel consumption (mt)", "mass")

    def test_header_with_wrong_dimension_unit_is_rejected(self):
        """A speed column labelled in kilograms is a schema error, not a conversion."""
        with pytest.raises(UnknownUnitError, match="not registered"):
            parse_header_unit("Speed (kg)", "speed")

    def test_superscript_cubic_metres_normalised(self):
        assert parse_header_unit("Tank volume (m³)", "volume") == "m^3"


class TestPhysicsHelpers:
    def test_kn_ms_round_trip(self):
        assert ms_to_kn(kn_to_ms(14.0)) == pytest.approx(14.0, rel=1e-12)

    def test_typical_service_speed(self):
        # 14 kn is a common coastal service speed; ~7.2 m/s
        assert kn_to_ms(14.0) == pytest.approx(7.2016, abs=1e-3)

    def test_no_nan_leakage(self):
        assert math.isnan(to_canonical(float("nan"), "kn", "speed"))
