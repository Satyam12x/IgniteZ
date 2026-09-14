"""Leakage-safe split tests. Maps to P-01 (the highest-priority edge case)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from greenfleet.prediction.splits import (
    LeakageError,
    assert_no_vessel_leakage,
    grouped_folds,
    split_by_time,
    split_by_vessel,
    split_by_vessel_and_time,
)

pytestmark = pytest.mark.edgecase


@pytest.fixture
def panel() -> pd.DataFrame:
    """A ship-year panel: 200 ships x 8 periods, like MRV."""
    rng = np.random.default_rng(0)
    ships = np.arange(1000000, 1000200)
    rows = [
        {"imo": imo, "reporting_period": period, "y": rng.normal(100, 10)}
        for imo in ships
        for period in range(2018, 2026)
    ]
    return pd.DataFrame(rows)


class TestVesselSplit:
    def test_no_vessel_appears_on_both_sides(self, panel):
        result = split_by_vessel(panel, test_size=0.25, seed=1000)
        assert not set(result.train.imo) & set(result.test.imo)

    def test_all_rows_of_a_test_ship_are_in_test(self, panel):
        """A partially held-out ship is the leak P-01 warns about."""
        result = split_by_vessel(panel, test_size=0.25, seed=1000)
        for imo in result.test.imo.unique():
            expected = (panel.imo == imo).sum()
            assert (result.test.imo == imo).sum() == expected

    def test_split_is_deterministic(self, panel):
        a = split_by_vessel(panel, test_size=0.25, seed=1000)
        b = split_by_vessel(panel, test_size=0.25, seed=1000)
        assert set(a.test.imo) == set(b.test.imo)

    def test_different_seed_gives_different_partition(self, panel):
        a = split_by_vessel(panel, test_size=0.25, seed=1000)
        b = split_by_vessel(panel, test_size=0.25, seed=2000)
        assert set(a.test.imo) != set(b.test.imo)

    def test_no_rows_are_lost(self, panel):
        result = split_by_vessel(panel, test_size=0.25, seed=1000)
        assert result.n_train + result.n_test == len(panel)

    @pytest.mark.parametrize("test_size", [0.0, 1.0, -0.1, 1.5])
    def test_invalid_test_size_rejected(self, panel, test_size):
        with pytest.raises(ValueError, match="test_size"):
            split_by_vessel(panel, test_size=test_size)

    def test_single_vessel_cannot_be_split(self):
        one = pd.DataFrame({"imo": [1, 1, 1], "reporting_period": [2018, 2019, 2020]})
        with pytest.raises(ValueError, match="at least 2 distinct vessels"):
            split_by_vessel(one)

    def test_missing_group_column_raises(self):
        with pytest.raises(KeyError, match="imo"):
            split_by_vessel(pd.DataFrame({"x": [1, 2, 3]}))


class TestTimeSplit:
    def test_train_is_strictly_earlier(self, panel):
        result = split_by_time(panel, train_through=2023)
        assert result.train.reporting_period.max() == 2023
        assert result.test.reporting_period.min() == 2024

    def test_empty_side_raises_with_available_periods(self, panel):
        with pytest.raises(ValueError, match="periods available"):
            split_by_time(panel, train_through=2050)

    def test_time_split_deliberately_shares_vessels(self, panel):
        """Not a leak: you do know your own fleet's history. Different question."""
        result = split_by_time(panel, train_through=2023)
        assert set(result.train.imo) & set(result.test.imo)


class TestVesselAndTimeSplit:
    def test_unseen_vessels_and_unseen_years(self, panel):
        result = split_by_vessel_and_time(panel, train_through=2023, test_size=0.25)
        assert not set(result.train.imo) & set(result.test.imo)
        assert result.train.reporting_period.max() <= 2023
        assert result.test.reporting_period.min() >= 2024

    def test_is_strictest_split(self, panel):
        """Should hold out fewer rows than either single split."""
        strict = split_by_vessel_and_time(panel, train_through=2023, test_size=0.25)
        vessel = split_by_vessel(panel, test_size=0.25)
        assert strict.n_train < vessel.n_train


class TestLeakageDetector:
    def test_detects_shared_vessel(self):
        train = pd.DataFrame({"imo": [1, 2, 3]})
        test = pd.DataFrame({"imo": [3, 4, 5]})
        with pytest.raises(LeakageError, match="appear in both"):
            assert_no_vessel_leakage(train, test)

    def test_error_names_the_offending_imo(self):
        train = pd.DataFrame({"imo": [9999999]})
        test = pd.DataFrame({"imo": [9999999]})
        with pytest.raises(LeakageError, match="9999999"):
            assert_no_vessel_leakage(train, test)

    def test_passes_when_disjoint(self):
        assert_no_vessel_leakage(pd.DataFrame({"imo": [1, 2]}), pd.DataFrame({"imo": [3]}))

    def test_nan_groups_do_not_count_as_shared(self):
        train = pd.DataFrame({"imo": [1, np.nan]})
        test = pd.DataFrame({"imo": [2, np.nan]})
        assert_no_vessel_leakage(train, test)


class TestGroupedFolds:
    def test_every_fold_is_leakage_free(self, panel):
        for train_idx, test_idx in grouped_folds(panel, n_splits=5):
            assert not set(panel.iloc[train_idx].imo) & set(panel.iloc[test_idx].imo)

    def test_folds_cover_all_rows_once_as_test(self, panel):
        folds = grouped_folds(panel, n_splits=5)
        seen = np.concatenate([test_idx for _, test_idx in folds])
        assert len(seen) == len(panel)
        assert len(set(seen)) == len(panel)

    def test_too_few_groups_raises(self):
        small = pd.DataFrame({"imo": [1, 2, 3]})
        with pytest.raises(ValueError, match="cannot be split"):
            grouped_folds(small, n_splits=5)
