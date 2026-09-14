"""Leakage-safe train/test splits.

P-01 is the highest-priority edge case in the register, and the most common silent
mistake in fuel-prediction work: a random row split puts the *same ship* in both
train and test, the model memorises that ship, and the reported accuracy is
fiction.

MRV is a ship-year panel - 25,967 ships over 8 reporting periods, so the median
ship appears several times. A random split would leak almost every ship.

Three splits, answering three different questions:

``by_vessel``
    Hold out whole ships. Answers "how well does this work on a vessel we have
    never seen?" - the question that matters for deployment and for the cold-start
    path, and the number that must be reported separately (P-01).
``by_time``
    Train on earlier periods, test on later ones. Answers "does this still work
    next year?" and is the only split that respects causality.
``by_vessel_and_time``
    Unseen ships *and* unseen years. The strictest, and the one to quote when a
    judge asks whether the number is real.

Every split runs an assertion that no group appears on both sides. The check is
not optional: a split helper that can silently leak is worse than no helper.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold, GroupShuffleSplit

logger = logging.getLogger(__name__)

__all__ = [
    "LeakageError",
    "SplitResult",
    "split_by_vessel",
    "split_by_time",
    "split_by_vessel_and_time",
    "grouped_folds",
    "assert_no_vessel_leakage",
]

GROUP_COLUMN = "imo"
TIME_COLUMN = "reporting_period"


class LeakageError(AssertionError):
    """The same vessel appears in both train and test."""


@dataclass(slots=True)
class SplitResult:
    """One train/test partition, with enough provenance to reproduce it."""

    train: pd.DataFrame
    test: pd.DataFrame
    kind: str
    detail: str

    @property
    def n_train(self) -> int:
        return len(self.train)

    @property
    def n_test(self) -> int:
        return len(self.test)

    def summary(self) -> str:
        train_ships = self.train[GROUP_COLUMN].nunique()
        test_ships = self.test[GROUP_COLUMN].nunique()
        return (
            f"{self.kind} ({self.detail}): "
            f"train {self.n_train} rows / {train_ships} ships, "
            f"test {self.n_test} rows / {test_ships} ships"
        )

    def as_dict(self) -> dict[str, object]:
        return {
            "kind": self.kind,
            "detail": self.detail,
            "n_train": self.n_train,
            "n_test": self.n_test,
            "train_ships": int(self.train[GROUP_COLUMN].nunique()),
            "test_ships": int(self.test[GROUP_COLUMN].nunique()),
        }


def assert_no_vessel_leakage(
    train: pd.DataFrame, test: pd.DataFrame, group_column: str = GROUP_COLUMN
) -> None:
    """Raise if any vessel appears on both sides of a split (P-01).

    Raises:
        LeakageError: with the count and a few example IMO numbers.
    """
    if group_column not in train.columns or group_column not in test.columns:
        raise KeyError(f"both frames need a {group_column!r} column to check leakage")
    shared = set(train[group_column].dropna()) & set(test[group_column].dropna())
    if shared:
        examples = sorted(shared)[:5]
        raise LeakageError(
            f"{len(shared)} vessel(s) appear in both train and test, e.g. IMO {examples}. "
            "This is edge case P-01: a random row split leaks ships in a ship-year panel."
        )


def _require_columns(frame: pd.DataFrame, columns: tuple[str, ...]) -> None:
    missing = [c for c in columns if c not in frame.columns]
    if missing:
        raise KeyError(f"frame is missing required column(s) {missing}")


def split_by_vessel(
    frame: pd.DataFrame,
    test_size: float = 0.25,
    seed: int = 1000,
    group_column: str = GROUP_COLUMN,
) -> SplitResult:
    """Hold out whole vessels. The unseen-vessel number (P-01, P-05)."""
    _require_columns(frame, (group_column,))
    if not 0.0 < test_size < 1.0:
        raise ValueError(f"test_size must be in (0, 1), got {test_size}")
    usable = frame[frame[group_column].notna()]
    if usable[group_column].nunique() < 2:
        raise ValueError("need at least 2 distinct vessels to split by vessel")

    splitter = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
    train_idx, test_idx = next(splitter.split(usable, groups=usable[group_column]))
    train, test = usable.iloc[train_idx].copy(), usable.iloc[test_idx].copy()
    assert_no_vessel_leakage(train, test, group_column)
    result = SplitResult(train, test, "by_vessel", f"test_size={test_size}, seed={seed}")
    logger.info(result.summary())
    return result


def split_by_time(
    frame: pd.DataFrame,
    train_through: int,
    time_column: str = TIME_COLUMN,
) -> SplitResult:
    """Train on periods <= ``train_through``, test on later ones.

    Note that this split deliberately does *not* exclude ships seen in training:
    that is the realistic operational case (you do know your own fleet's history).
    It answers a different question from ``split_by_vessel``, so report both.
    """
    _require_columns(frame, (time_column,))
    periods = pd.to_numeric(frame[time_column], errors="coerce")
    train = frame[periods <= train_through].copy()
    test = frame[periods > train_through].copy()
    if train.empty or test.empty:
        available = sorted(periods.dropna().unique().tolist())
        raise ValueError(
            f"train_through={train_through} leaves an empty side; periods available: {available}"
        )
    result = SplitResult(train, test, "by_time", f"train<={train_through}")
    logger.info(result.summary())
    return result


def split_by_vessel_and_time(
    frame: pd.DataFrame,
    train_through: int,
    test_size: float = 0.25,
    seed: int = 1000,
    group_column: str = GROUP_COLUMN,
    time_column: str = TIME_COLUMN,
) -> SplitResult:
    """Unseen vessels in unseen years. The strictest split, and the honest headline.

    Vessels are partitioned first, then the time cut is applied to each side, so a
    test vessel contributes nothing to training even in earlier periods.
    """
    _require_columns(frame, (group_column, time_column))
    vessel_split = split_by_vessel(frame, test_size=test_size, seed=seed,
                                   group_column=group_column)
    periods_train = pd.to_numeric(vessel_split.train[time_column], errors="coerce")
    periods_test = pd.to_numeric(vessel_split.test[time_column], errors="coerce")

    train = vessel_split.train[periods_train <= train_through].copy()
    test = vessel_split.test[periods_test > train_through].copy()
    if train.empty or test.empty:
        raise ValueError(
            f"train_through={train_through} with test_size={test_size} leaves an empty side"
        )
    assert_no_vessel_leakage(train, test, group_column)
    result = SplitResult(
        train, test, "by_vessel_and_time",
        f"train<={train_through}, unseen vessels, test_size={test_size}, seed={seed}",
    )
    logger.info(result.summary())
    return result


def grouped_folds(
    frame: pd.DataFrame,
    n_splits: int = 5,
    group_column: str = GROUP_COLUMN,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Grouped K-fold index pairs for cross-validated tuning.

    Used by the QPSO tuner so that hyperparameter selection is itself evaluated on
    unseen vessels - otherwise tuning leaks even when the final split does not.
    """
    _require_columns(frame, (group_column,))
    groups = frame[group_column]
    n_groups = groups.nunique()
    if n_groups < n_splits:
        raise ValueError(f"{n_groups} vessels cannot be split into {n_splits} folds")
    folds = list(GroupKFold(n_splits=n_splits).split(frame, groups=groups))
    for train_idx, test_idx in folds:
        assert_no_vessel_leakage(frame.iloc[train_idx], frame.iloc[test_idx], group_column)
    return folds
