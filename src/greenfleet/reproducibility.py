"""Deterministic seeding and run provenance.

Edge cases P-11 (same seed, same model, same metrics) and Q-06 (same seed, same
optimizer result; benchmarking uses >=30 distinct seeds). X-05 requires every run
to be reproducible from its logged inputs, so ``RunContext`` is what we persist
alongside any result.
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
import random
import subprocess
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

__all__ = ["set_global_seed", "RunContext", "BENCHMARK_SEEDS", "config_digest"]

# Fixed seed list so every algorithm in the benchmark sees the identical set (B-02).
BENCHMARK_SEEDS: tuple[int, ...] = tuple(range(1000, 1030))  # 30 runs


def set_global_seed(seed: int) -> np.random.Generator:
    """Seed every RNG this project can reach and return a local Generator.

    Prefer the returned ``Generator`` over module-level ``np.random`` calls:
    passing it explicitly is what makes parallel runs reproducible.
    """
    if not isinstance(seed, (int, np.integer)) or isinstance(seed, bool):
        raise TypeError(f"seed must be an int, got {type(seed).__name__}")
    random.seed(seed)
    np.random.seed(seed % (2**32))
    os.environ["PYTHONHASHSEED"] = str(seed)
    return np.random.default_rng(seed)


def _git_revision() -> str | None:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
        return out.stdout.strip() or None
    except (OSError, subprocess.SubprocessError):
        return None


def config_digest(payload: Any) -> str:
    """Stable SHA-256 over a config/input payload, for the audit trail (X-05).

    Emission factors live in config (F-09); the digest is how a result is tied to
    the exact factor set that produced it.
    """
    blob = json.dumps(payload, sort_keys=True, default=str, separators=(",", ":"))
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:16]


@dataclass(slots=True)
class RunContext:
    """Everything needed to reproduce one run."""

    seed: int
    stage: str
    inputs: dict[str, Any] = field(default_factory=dict)
    package_version: str = "0.1.0"
    git_revision: str | None = field(default_factory=_git_revision)
    python_version: str = field(default_factory=platform.python_version)
    started_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def as_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["input_digest"] = config_digest(self.inputs)
        return payload

    def write(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.as_dict(), indent=2, default=str), encoding="utf-8")
        return path
