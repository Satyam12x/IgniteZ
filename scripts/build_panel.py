"""Build the ship-year panel the model trains on from the downloaded MRV workbooks.

    python -m greenfleet.data.mrv.download      # once: fetch the 8 EMSA workbooks (~44 MB)
    python scripts/build_panel.py               # -> data/interim/mrv_panel.parquet
                                                #    data/interim/mrv_derived.parquet

The two parquet files are committed to the repository so that training, evaluation
and the platform work without re-downloading. Re-run this only when EMSA publishes
a new version of a reporting period (the manifest in ``data/raw/mrv`` pins versions).
Parsing eight 100k-row workbooks takes several minutes.
"""

from __future__ import annotations

import argparse
import logging
import time
from pathlib import Path

from greenfleet.data.mrv.derive import derive_physics
from greenfleet.data.mrv.download import DEFAULT_RAW_DIR
from greenfleet.data.mrv.load import load_all

INTERIM = Path("data/interim")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", type=Path, default=DEFAULT_RAW_DIR)
    parser.add_argument("--out-dir", type=Path, default=INTERIM)
    parser.add_argument("--periods", type=int, nargs="*", default=None,
                        help="reporting periods to include (default: all downloaded)")
    args = parser.parse_args()
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(message)s")

    started = time.perf_counter()
    panel, reports = load_all(args.raw_dir, periods=args.periods)
    for report in reports:
        print(f"  {report.summary()}")
    print(f"loaded {len(panel):,} ship-years from {len(reports)} periods "
          f"in {time.perf_counter() - started:.0f}s")

    derived, quality = derive_physics(panel)
    print(f"derived physics: {int(derived.is_usable.sum()):,} usable of {len(derived):,} rows")
    for key, value in quality.as_dict().items():
        print(f"  {key}: {value}")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    panel.to_parquet(args.out_dir / "mrv_panel.parquet", index=False)
    derived.to_parquet(args.out_dir / "mrv_derived.parquet", index=False)
    print(f"wrote {args.out_dir / 'mrv_panel.parquet'} and {args.out_dir / 'mrv_derived.parquet'}")


if __name__ == "__main__":
    main()
