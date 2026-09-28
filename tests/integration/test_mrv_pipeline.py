"""End-to-end pipeline tests against the real EU MRV data.

Skipped automatically when the data has not been downloaded, so a fresh clone can
still run the unit suite. To enable::

    uv run python -m greenfleet.data.mrv.download

These tests assert the properties that must survive contact with real data, and
several of them encode findings that came *from* the data - the published
"Division by zero!" strings, implied CO2 factors above the physical maximum, and
the schema rename across reporting periods.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from greenfleet.data.mrv.derive import PLAUSIBLE, infer_blend_lhv
from greenfleet.data.mrv.load import load_all, load_reporting_period
from greenfleet.data.mrv.schema import MRV_COLUMNS
from greenfleet.prediction.features import build_features, build_target
from greenfleet.prediction.model import EnergyModel
from greenfleet.prediction.splits import split_by_vessel

RAW_DIR = Path("data/raw/mrv")
DERIVED = Path("data/interim/mrv_derived.parquet")

pytestmark = [pytest.mark.integration, pytest.mark.slow]

_workbooks = sorted(RAW_DIR.glob("mrv_*.xlsx"))
requires_raw = pytest.mark.skipif(not _workbooks, reason="MRV workbooks not downloaded")
requires_derived = pytest.mark.skipif(
    not DERIVED.exists(), reason="derived panel not built"
)


@requires_raw
class TestManifestProvenance:
    def test_manifest_pins_version_and_checksum(self):
        """X-05: results must be tied to the exact published version."""
        manifest = json.loads((RAW_DIR / "manifest.json").read_text(encoding="utf-8"))
        assert manifest["files"], "manifest records no files"
        for period, entry in manifest["files"].items():
            assert entry["version"] > 0, f"{period} has no version"
            assert len(entry["sha256"]) == 64, f"{period} has no checksum"
            assert entry["bytes"] > 1_000_000

    def test_every_recorded_file_exists_with_the_recorded_size(self):
        manifest = json.loads((RAW_DIR / "manifest.json").read_text(encoding="utf-8"))
        for entry in manifest["files"].values():
            path = RAW_DIR / entry["local_name"]
            assert path.exists(), f"{path} is recorded but missing"
            assert path.stat().st_size == entry["bytes"]


@requires_raw
class TestSchemaAcrossReportingPeriods:
    @pytest.mark.parametrize("workbook", _workbooks, ids=lambda p: p.stem)
    def test_every_period_loads(self, workbook):
        """D-08: the schema changed twice; aliases must cover every period."""
        frame, report = load_reporting_period(workbook)
        assert len(frame) > 5000, f"{workbook.name} yielded only {len(frame)} ships"
        assert report.reporting_period is not None

    @pytest.mark.parametrize("workbook", _workbooks, ids=lambda p: p.stem)
    def test_no_unparseable_values(self, workbook):
        """Every non-numeric cell must be a recognised token, not a surprise."""
        _, report = load_reporting_period(workbook)
        assert report.unparseable_counts == {}, (
            f"{workbook.name} has unclassified cell content: {report.unparseable_counts}"
        )

    def test_2024_onward_reports_measured_ch4_and_n2o(self):
        """The EU ETS extension: F-02 and F-03 become validatable from 2024."""
        recent = [w for w in _workbooks if "2024" in w.name or "2025" in w.name]
        if not recent:
            pytest.skip("2024/2025 not downloaded")
        for workbook in recent:
            _, report = load_reporting_period(workbook)
            assert "total_ch4_t" not in report.missing_optional_columns
            assert "total_n2o_t" not in report.missing_optional_columns

    def test_partial_reports_are_a_separate_sheet(self):
        """Concatenating Full and Partial ERs would double-count ships."""
        recent = [w for w in _workbooks if "2024" in w.name]
        if not recent:
            pytest.skip("2024 not downloaded")
        full, full_report = load_reporting_period(recent[0], coverage="full")
        partial, partial_report = load_reporting_period(recent[0], coverage="partial")
        assert full_report.sheet != partial_report.sheet
        assert len(full) > len(partial)
        assert (full.report_coverage == "full").all()
        assert (partial.report_coverage == "partial").all()

    def test_unknown_coverage_rejected(self):
        with pytest.raises(ValueError, match="coverage must be"):
            load_reporting_period(_workbooks[0], coverage="everything")

    def test_declared_columns_have_unique_internal_names(self):
        names = [spec.name for spec in MRV_COLUMNS]
        assert len(set(names)) == len(names)


@requires_raw
class TestPanelIdentity:
    def test_panel_is_unique_on_ship_and_period(self):
        """D-06: one row per ship per reporting period, deterministically."""
        panel, _ = load_all()
        duplicated = panel.duplicated(subset=["imo", "reporting_period"])
        assert not duplicated.any(), f"{duplicated.sum()} duplicate ship-years"

    def test_panel_covers_multiple_periods_and_recurring_ships(self):
        panel, _ = load_all()
        assert panel.reporting_period.nunique() >= 2
        counts = panel.groupby("imo").size()
        assert (counts > 1).any(), "no ship recurs, so P-01 grouping is untested"


@pytest.fixture(scope="module")
def derived() -> pd.DataFrame:
    """The derived panel, loaded once for the whole module."""
    return pd.read_parquet(DERIVED)


@pytest.fixture(scope="module")
def prepared(derived: pd.DataFrame) -> pd.DataFrame:
    """A vessel-grouped subsample with the target attached, for the model tests."""
    panel = derived[derived.is_usable].copy()
    panel["_y"] = build_target(panel)
    panel = panel[np.isfinite(panel._y) & (panel._y > 0)]
    ships = panel.imo.dropna().unique()
    keep = set(np.random.default_rng(1000).choice(ships, 2500, replace=False))
    return panel[panel.imo.isin(keep)].reset_index(drop=True)


@requires_derived
class TestDerivedPhysics:

    def test_power_identity_holds_exactly(self, derived):
        """The circularity finding: power is algebraically tied to speed.

        Verified here so that if anyone ever 'fixes' derive.py to make speed an
        independent regressor, this test tells them why they cannot.
        """
        usable = derived[derived.is_usable]
        left = usable.mean_power_kw.to_numpy()
        right = (
            usable.inferred_lhv_mj_per_kg
            * usable.fuel_per_distance_kg_per_nmi
            * usable.mean_speed_kn
            / 3.6
        ).to_numpy()
        relative = np.abs(left - right) / left
        assert np.nanmax(relative) < 1e-9

    def test_usable_fraction_is_high(self, derived):
        assert derived.is_usable.mean() > 0.85

    def test_flagged_rows_are_excluded_from_usable(self, derived):
        flags = [c for c in derived.columns if c.startswith("flag_")]
        assert not derived.loc[derived.is_usable, flags].any(axis=1).any()

    def test_implied_co2_factor_is_physically_bounded_when_usable(self, derived):
        """D-03: the panel contains factors up to 9.68, which is impossible."""
        usable = derived[derived.is_usable].implied_cf_t_per_t.dropna()
        low, high = PLAUSIBLE["implied_cf_t_per_t"]
        assert usable.between(low, high).all()
        # And the raw data really does contain the impossible values.
        assert (derived.implied_cf_t_per_t.dropna() > high).any()

    def test_speeds_are_realistic_when_usable(self, derived):
        speeds = derived[derived.is_usable].mean_speed_kn.dropna()
        assert speeds.between(*PLAUSIBLE["mean_speed_kn"]).all()
        assert 8.0 < speeds.median() < 16.0, "median service speed looks wrong"

    def test_inferred_lhv_lies_between_the_reference_fuels(self, derived):
        lhv = derived[derived.is_usable].inferred_lhv_mj_per_kg.dropna()
        assert lhv.between(40.2, 48.0).all(), "blend LHV escaped the LNG-HFO-MGO ladder"

    def test_blend_inference_endpoints(self):
        """Pure-fuel CO2 factors must recover that fuel's own heating value."""
        lhv, _ = infer_blend_lhv(np.array([3.206, 3.114, 2.750]))
        assert lhv[0] == pytest.approx(42.7, rel=1e-6)   # MGO
        assert lhv[1] == pytest.approx(40.2, rel=1e-6)   # HFO
        assert lhv[2] == pytest.approx(48.0, rel=1e-6)   # LNG

    def test_energy_intensity_is_stable_within_ship(self, derived):
        """ICC ~0.94: intensity is a ship property, which is why P-05 works."""
        usable = derived[derived.is_usable].copy()
        usable["log_intensity"] = np.log(
            usable.inferred_lhv_mj_per_kg * usable.fuel_per_distance_kg_per_nmi
        )
        usable = usable[np.isfinite(usable.log_intensity)]
        counts = usable.groupby("imo").log_intensity.count()
        repeat = usable[usable.imo.isin(counts[counts >= 5].index)]
        within = repeat.groupby("imo").log_intensity.std().mean()
        between = repeat.groupby("imo").log_intensity.mean().std()
        icc = between**2 / (between**2 + within**2)
        assert icc > 0.85, f"ICC fell to {icc:.3f}; the ship-level signal weakened"


@requires_derived
class TestEndToEndModel:
    def test_pipeline_produces_sane_predictions(self, prepared):
        split = split_by_vessel(prepared, test_size=0.25, seed=1000)
        model = EnergyModel(seed=1000, n_estimators=120).fit(
            build_features(split.train), split.train._y.values
        )
        prediction = model.predict(build_features(split.test))
        assert np.isfinite(prediction.energy_per_nmi_mj).all()
        assert (prediction.energy_per_nmi_mj > 0).all()

    def test_beats_the_naive_class_mean(self, prepared):
        from greenfleet.prediction.metrics import regression_metrics

        split = split_by_vessel(prepared, test_size=0.25, seed=1000)
        Xtr, ytr = build_features(split.train), split.train._y.values
        Xte, yte = build_features(split.test), split.test._y.values
        naive = EnergyModel(residual_model="class_mean", seed=1000).fit(Xtr, ytr)
        learned = EnergyModel(seed=1000, n_estimators=200).fit(Xtr, ytr)
        naive_mape = regression_metrics(yte, naive.predict(Xte).energy_per_nmi_mj).mape
        learned_mape = regression_metrics(yte, learned.predict(Xte).energy_per_nmi_mj).mape
        assert learned_mape < naive_mape

    def test_accuracy_respects_the_irreducible_floor(self, prepared):
        """~11% within-ship variation is irreducible; beating it signals leakage."""
        from greenfleet.prediction.metrics import regression_metrics

        split = split_by_vessel(prepared, test_size=0.25, seed=1000)
        model = EnergyModel(seed=1000, n_estimators=200).fit(
            build_features(split.train), split.train._y.values
        )
        mape = regression_metrics(
            split.test._y.values,
            model.predict(build_features(split.test)).energy_per_nmi_mj,
        ).mape
        assert mape > 5.0, (
            f"MAPE of {mape:.1f}% on unseen vessels is below the plausible floor; "
            "suspect leakage before celebrating"
        )
        assert mape < 45.0, f"MAPE of {mape:.1f}% is worse than the class mean"
