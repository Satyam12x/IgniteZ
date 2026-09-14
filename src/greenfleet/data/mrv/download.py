"""Download the EU MRV "Publication of information" dataset from THETIS-MRV.

THETIS-MRV (EMSA) publishes one XLSX per reporting period, versioned and
regenerated as companies file corrections. The public API needs no credentials:

    GET /api/public-emission-report/downloadable-files
        -> [{reportingPeriod, version, generationDate, fileName}, ...]
    GET /api/public-emission-report/reporting-period-document/binary/{period}/{version}
        -> the XLSX

Why we pin the version
----------------------
The same reporting period is republished with a higher version number whenever a
company corrects a filing, so "the 2024 file" is not a stable object. X-05 requires
any result to be reproducible, so we record period + version + SHA-256 in a manifest
and train against pinned versions. Re-running with a newer version is then a visible
data change, not a silent one.

Scope caveat (documented deliberately)
--------------------------------------
MRV covers ships >= 5000 GT calling at EEA ports. Harbour tugs are ~200-500 GT and
are NOT in this dataset, and the data is annual aggregate per ship - no per-voyage
speed or weather. MRV is therefore the real-data anchor for the energy model; the
tug case study reaches tugs through the cold-start physics path (P-05).
"""

from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import requests

logger = logging.getLogger(__name__)

API_ROOT = "https://mrv.emsa.europa.eu/api/public-emission-report"
DEFAULT_RAW_DIR = Path("data/raw/mrv")
MANIFEST_NAME = "manifest.json"

# A partial download must never be mistaken for a usable file.
_MIN_PLAUSIBLE_BYTES = 1_000_000
_XLSX_MAGIC = b"PK\x03\x04"
_CHUNK = 1 << 16

__all__ = ["MrvFile", "list_downloadable_files", "download_reporting_period", "download_all"]


@dataclass(frozen=True, slots=True)
class MrvFile:
    """One published reporting-period file, as advertised by the API."""

    reporting_period: int
    version: int
    generation_date: str
    file_name: str

    @property
    def local_name(self) -> str:
        return f"mrv_{self.reporting_period}_v{self.version}.xlsx"

    @property
    def binary_url(self) -> str:
        return f"{API_ROOT}/reporting-period-document/binary/{self.reporting_period}/{self.version}"


def _session(timeout: int = 60) -> requests.Session:
    sess = requests.Session()
    sess.headers.update(
        {
            "Accept": "application/json, application/octet-stream",
            "User-Agent": "greenfleet/0.1 (SIH26138 research; contact via repo)",
        }
    )
    sess.request_timeout = timeout  # type: ignore[attr-defined]
    return sess


def list_downloadable_files(session: requests.Session | None = None) -> list[MrvFile]:
    """Ask the API which reporting periods and versions are currently published."""
    sess = session or _session()
    resp = sess.get(f"{API_ROOT}/downloadable-files", timeout=60)
    resp.raise_for_status()
    payload = resp.json()
    files = [
        MrvFile(
            reporting_period=int(row["reportingPeriod"]),
            version=int(row["version"]),
            generation_date=str(row.get("generationDate", "")),
            file_name=str(row.get("fileName", "")),
        )
        for row in payload.get("results", [])
    ]
    return sorted(files, key=lambda f: f.reporting_period)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(_CHUNK), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_manifest(raw_dir: Path) -> dict[str, Any]:
    manifest_path = raw_dir / MANIFEST_NAME
    if not manifest_path.exists():
        return {"source": API_ROOT, "files": {}}
    return json.loads(manifest_path.read_text(encoding="utf-8"))


def _save_manifest(raw_dir: Path, manifest: dict[str, Any]) -> None:
    (raw_dir / MANIFEST_NAME).write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def download_reporting_period(
    mrv_file: MrvFile,
    raw_dir: Path = DEFAULT_RAW_DIR,
    session: requests.Session | None = None,
    force: bool = False,
) -> Path:
    """Download one reporting period, skipping work when the pinned version is present.

    Raises:
        OSError: the downloaded bytes are not a plausible XLSX (truncated transfer
            or an HTML error page served with a 200).
    """
    sess = session or _session()
    raw_dir.mkdir(parents=True, exist_ok=True)
    target = raw_dir / mrv_file.local_name
    manifest = _load_manifest(raw_dir)
    key = str(mrv_file.reporting_period)
    recorded = manifest["files"].get(key)

    if (
        not force
        and target.exists()
        and recorded
        and recorded.get("version") == mrv_file.version
        and recorded.get("bytes") == target.stat().st_size
    ):
        logger.info("MRV %s v%s already present, skipping", key, mrv_file.version)
        return target

    logger.info("downloading MRV %s v%s -> %s", key, mrv_file.version, target.name)
    tmp = target.with_suffix(".xlsx.part")
    with sess.get(mrv_file.binary_url, stream=True, timeout=600) as resp:
        resp.raise_for_status()
        with tmp.open("wb") as handle:
            for chunk in resp.iter_content(chunk_size=_CHUNK):
                if chunk:
                    handle.write(chunk)

    size = tmp.stat().st_size
    with tmp.open("rb") as handle:
        magic = handle.read(4)
    if magic != _XLSX_MAGIC or size < _MIN_PLAUSIBLE_BYTES:
        tmp.unlink(missing_ok=True)
        raise OSError(
            f"MRV {key} v{mrv_file.version}: got {size} bytes with magic {magic!r}; "
            "expected a multi-MB XLSX. Transfer was truncated or the API returned an error page."
        )

    tmp.replace(target)
    manifest["files"][key] = {
        "reporting_period": mrv_file.reporting_period,
        "version": mrv_file.version,
        "generation_date": mrv_file.generation_date,
        "published_name": mrv_file.file_name,
        "local_name": mrv_file.local_name,
        "bytes": size,
        "sha256": _sha256(target),
        "url": mrv_file.binary_url,
    }
    _save_manifest(raw_dir, manifest)
    return target


def download_all(
    raw_dir: Path = DEFAULT_RAW_DIR,
    periods: list[int] | None = None,
    force: bool = False,
) -> list[Path]:
    """Download every published reporting period (or just ``periods``)."""
    sess = _session()
    available = list_downloadable_files(sess)
    if periods is not None:
        wanted = set(periods)
        missing = wanted - {f.reporting_period for f in available}
        if missing:
            raise ValueError(
                f"reporting periods {sorted(missing)} are not published; "
                f"available: {[f.reporting_period for f in available]}"
            )
        available = [f for f in available if f.reporting_period in wanted]
    return [download_reporting_period(f, raw_dir, sess, force) for f in available]


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    paths = download_all()
    total = sum(p.stat().st_size for p in paths)
    print(f"\n{len(paths)} files, {total / 1e6:.1f} MB in {DEFAULT_RAW_DIR}")
