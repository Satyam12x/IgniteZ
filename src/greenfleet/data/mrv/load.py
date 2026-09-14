"""Read the MRV workbooks into a typed, validated DataFrame.

Responsibilities, in order:

1. Validate the sheet layout and the presence of every required column (D-08).
2. Map published headers to internal names by normalised header text, so extra or
   reordered columns are tolerated and a missing one names itself.
3. Coerce EMSA's "N/A" / "Not Applicable" family to NaN (D-02) without ever
   touching the target column.
4. Deduplicate (IMO, reporting period) deterministically (D-06).

No physics and no imputation happen here. Reconstruction of distance, speed and
cargo lives in ``derive.py`` so that the raw-to-typed step stays auditable.
"""

from __future__ import annotations

import logging
import re
import warnings
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from .schema import (
    COMPUTATION_ERROR_TOKENS,
    FIRST_DATA_ROW,
    HEADER_ROW,
    MRV_COLUMNS,
    NULL_TOKENS,
    MrvSchemaError,
    normalise_header,
)

logger = logging.getLogger(__name__)

DEFAULT_RAW_DIR = Path("data/raw/mrv")

__all__ = ["load_reporting_period", "load_all", "LoadReport"]

_NUMERIC_DIMENSIONS = {"mass", "time", "distance", "emission", "rate"}


@dataclass
class LoadReport:
    """Per-file provenance and row accounting, for the data-quality report (D-02, D-06)."""

    path: str
    sheet: str
    reporting_period: int | None
    coverage: str = "full"
    rows_read: int = 0
    rows_after_dedup: int = 0
    duplicate_rows_dropped: int = 0
    blank_rows_dropped: int = 0
    unconsumed_columns: list[str] = field(default_factory=list)
    missing_optional_columns: list[str] = field(default_factory=list)
    null_counts: dict[str, int] = field(default_factory=dict)
    computation_error_counts: dict[str, int] = field(default_factory=dict)
    """Per column, how many cells EMSA published as a spreadsheet error string."""
    unparseable_counts: dict[str, int] = field(default_factory=dict)
    """Per column, cells that are neither a number, a null token, nor a known error."""
    resolved_aliases: dict[str, str] = field(default_factory=dict)
    """Columns matched via an alias rather than the canonical header."""

    def summary(self) -> str:
        return (
            f"{Path(self.path).name}: {self.rows_after_dedup} ships "
            f"(read {self.rows_read}, -{self.duplicate_rows_dropped} dup, "
            f"-{self.blank_rows_dropped} blank)"
        )


def _resolve_sheet(book: pd.ExcelFile, path: Path, coverage: str = "full") -> str:
    """Pick the data sheet.

    Through 2023 the workbook has a single sheet named after the reporting period
    ("2018"). From 2024 it has two, split by report coverage:

        "2024 Full ERs"      ships with a full-year emission report
        "2024 Partial ERs"   ships that changed company mid-year, so the year is
                             split across several partial reports

    Partial reports must not be concatenated with full ones: the same ship appears
    more than once and its annual totals cover part of a year, so any annual
    aggregate double-counts and any derived annual mean is wrong. Intensity ratios
    remain meaningful, so partial reports are loadable on request and always
    labelled in the ``report_coverage`` column.
    """
    names = list(book.sheet_names)
    if len(names) == 1:
        return names[0]

    numeric = [n for n in names if n.strip().isdigit()]
    if len(numeric) == 1:
        return numeric[0]

    wanted = "partial" if coverage == "partial" else "full"
    matches = [n for n in names if wanted in n.lower()]
    if len(matches) == 1:
        return matches[0]

    raise MrvSchemaError(
        f"{path.name}: cannot identify the {coverage!r} data sheet among {names}"
    )


def _coerce_numeric(series: pd.Series, column: str) -> tuple[pd.Series, int, int]:
    """Coerce to float, classifying every non-numeric cell.

    Three outcomes are kept apart, because conflating them is how a data-quality
    report ends up lying:

    * a recognised null token ("N/A", "Not Applicable") - the metric does not
      apply to this ship type;
    * a recognised spreadsheet error ("Division by zero!") - EMSA's own formula
      failed because the ship reported zero distance. Real information (P-03);
    * anything else - genuinely unparseable, and worth a warning.

    Returns ``(values, computation_errors, unparseable)``.
    """
    if pd.api.types.is_numeric_dtype(series):
        return series.astype("float64"), 0, 0

    text = series.astype("string").str.strip()
    lowered = text.str.lower()
    is_null = lowered.isin(NULL_TOKENS) | text.isna()
    is_error = lowered.isin(COMPUTATION_ERROR_TOKENS)
    # EMSA occasionally publishes thousands separators in text-formatted cells.
    numeric = pd.to_numeric(text.str.replace(",", "", regex=False), errors="coerce")

    computation_errors = int(is_error.sum())
    unparseable_mask = numeric.isna() & ~is_null & ~is_error
    unparseable = int(unparseable_mask.sum())
    if unparseable:
        examples = text[unparseable_mask].dropna().unique()[:3].tolist()
        logger.warning(
            "column %s: %d values are neither numeric nor a recognised token, e.g. %s",
            column,
            unparseable,
            examples,
        )
    return numeric.astype("float64"), computation_errors, unparseable


def load_reporting_period(
    path: Path | str, coverage: str = "full"
) -> tuple[pd.DataFrame, LoadReport]:
    """Load one MRV workbook into a typed DataFrame.

    Args:
        path: the workbook.
        coverage: ``"full"`` (default) or ``"partial"``, selecting which sheet to
            read for 2024+ files. See :func:`_resolve_sheet`.
    """
    if coverage not in {"full", "partial"}:
        raise ValueError(f"coverage must be 'full' or 'partial', got {coverage!r}")
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"{path} does not exist; run greenfleet.data.mrv.download")

    with warnings.catch_warnings():
        # EMSA's writer omits a default style; openpyxl warns on every open.
        warnings.filterwarnings("ignore", message="Workbook contains no default style")
        book = pd.ExcelFile(path, engine="openpyxl")
        sheet = _resolve_sheet(book, path, coverage)
        raw = book.parse(
            sheet_name=sheet,
            skiprows=HEADER_ROW - 1,  # header lands on the published header row
            header=0,
            dtype=object,
        )
        book.close()

    if raw.empty:
        raise MrvSchemaError(f"{path.name}: sheet {sheet!r} has no data rows (D-10)")

    period_match = re.search(r"(19|20)\d{2}", sheet)
    report = LoadReport(
        path=str(path),
        sheet=sheet,
        reporting_period=int(period_match.group(0)) if period_match else None,
        coverage=coverage,
    )
    report.rows_read = len(raw)

    # Map published headers -> internal names. Lookup is by normalised text, so a
    # reordered or newly inserted column does not shift anything (D-08).
    published: dict[str, str] = {}
    for actual in raw.columns:
        published.setdefault(normalise_header(actual), actual)

    frame = pd.DataFrame(index=raw.index)
    missing_required: list[str] = []
    resolved_aliases: dict[str, str] = {}
    for spec in MRV_COLUMNS:
        # Primary header first, then aliases, so the canonical name wins when a
        # period publishes both forms (D-08, and see schema.py on renames).
        actual = None
        for position, candidate in enumerate(spec.keys):
            actual = published.get(candidate)
            if actual is not None:
                if position:
                    resolved_aliases[spec.name] = str(actual)
                break
        if actual is None:
            if spec.required:
                missing_required.append(f"{spec.name} (header: {spec.header!r})")
            else:
                # Absent optional column: materialise it empty so downstream code
                # sees a uniform frame across reporting periods with different
                # schemas (the 2024 file has 113 columns, 2018 has 62).
                report.missing_optional_columns.append(spec.name)
                if spec.dimension in _NUMERIC_DIMENSIONS:
                    frame[spec.name] = np.full(len(raw), np.nan, dtype="float64")
                else:
                    frame[spec.name] = pd.Series(
                        pd.NA, index=raw.index, dtype="string"
                    )
            continue
        column = raw[actual]
        if spec.dimension in _NUMERIC_DIMENSIONS:
            values, errors, unparseable = _coerce_numeric(column, spec.name)
            frame[spec.name] = values
            if errors:
                report.computation_error_counts[spec.name] = errors
            if unparseable:
                report.unparseable_counts[spec.name] = unparseable
        else:
            frame[spec.name] = column.astype("string").str.strip()

    if missing_required:
        raise MrvSchemaError(
            f"{path.name}: required column(s) absent: {'; '.join(missing_required)}. "
            "The published schema may have changed - update greenfleet.data.mrv.schema."
        )

    report.resolved_aliases = resolved_aliases
    consumed = {key for spec in MRV_COLUMNS for key in spec.keys}
    report.unconsumed_columns = [
        str(actual) for key, actual in published.items() if key and key not in consumed
    ]

    # IMO number is the identity key for every split downstream (P-01).
    frame["imo"] = pd.to_numeric(frame["imo"], errors="coerce").astype("Int64")
    frame["reporting_period"] = (
        pd.to_numeric(frame["reporting_period"], errors="coerce").round().astype("Int64")
    )
    if report.reporting_period is not None:
        # Trust the sheet name over a per-row value that EMSA writes as a float.
        frame["reporting_period"] = frame["reporting_period"].fillna(report.reporting_period)

    before = len(frame)
    frame = frame[frame["imo"].notna() & frame["total_fuel_t"].notna()]
    report.blank_rows_dropped = before - len(frame)

    # D-06: deterministic dedup. Keep the row with the most populated fields, then
    # the last occurrence, so the result does not depend on row order.
    before = len(frame)
    if frame.duplicated(subset=["imo", "reporting_period"]).any():
        frame = (
            frame.assign(_completeness=frame.notna().sum(axis=1))
            .sort_values(["imo", "reporting_period", "_completeness"], kind="mergesort")
            .drop_duplicates(subset=["imo", "reporting_period"], keep="last")
            .drop(columns="_completeness")
        )
    report.duplicate_rows_dropped = before - len(frame)
    report.rows_after_dedup = len(frame)
    report.null_counts = {
        col: int(frame[col].isna().sum()) for col in frame.columns if frame[col].isna().any()
    }

    frame["report_coverage"] = coverage
    frame = frame.reset_index(drop=True)
    frame.attrs["source_file"] = path.name
    frame.attrs["coverage"] = coverage
    frame.attrs["sheet"] = sheet
    frame.attrs["first_data_row"] = FIRST_DATA_ROW
    logger.info(report.summary())
    return frame, report


def load_all(
    raw_dir: Path | str = DEFAULT_RAW_DIR,
    periods: list[int] | None = None,
    coverage: str = "full",
) -> tuple[pd.DataFrame, list[LoadReport]]:
    """Load every downloaded reporting period into one panel.

    The result is a ship-year panel: one row per (IMO, reporting period). Ships
    recur across years, which is exactly why P-01 requires grouping by IMO.
    """
    raw_dir = Path(raw_dir)
    files = sorted(raw_dir.glob("mrv_*.xlsx"))
    if not files:
        raise FileNotFoundError(
            f"no MRV workbooks in {raw_dir}; run `python -m greenfleet.data.mrv.download`"
        )

    frames, reports = [], []
    for path in files:
        frame, report = load_reporting_period(path, coverage=coverage)
        if periods is not None and report.reporting_period not in periods:
            continue
        frames.append(frame)
        reports.append(report)

    if not frames:
        raise ValueError(f"no reporting periods matched {periods}")

    panel = pd.concat(frames, ignore_index=True)
    panel = panel.sort_values(["imo", "reporting_period"], kind="mergesort").reset_index(drop=True)
    return panel, reports
