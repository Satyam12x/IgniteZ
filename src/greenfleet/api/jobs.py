"""In-memory job store for optimisation runs.

An optimisation takes seconds to minutes, so it runs on a worker thread and the
client polls for progress. The store is deliberately simple - a dict guarded by a
lock - because the platform is single-process and on-premises; a queue would be
the first thing to add for a multi-user deployment.
"""

from __future__ import annotations

import threading
import uuid
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any


@dataclass
class Job:
    job_id: str
    request: Any
    status: str = "queued"
    progress: dict[str, Any] | None = None
    result: dict[str, Any] | None = None
    error: str | None = None
    cancel_event: threading.Event = field(default_factory=threading.Event)


class JobStore:
    def __init__(self, max_jobs: int = 50):
        self._jobs: dict[str, Job] = {}
        self._lock = threading.Lock()
        self._max_jobs = max_jobs

    def submit(self, request: Any, runner: Callable[[Job], dict[str, Any]]) -> Job:
        job = Job(job_id=uuid.uuid4().hex[:12], request=request)
        with self._lock:
            self._evict_locked()
            self._jobs[job.job_id] = job

        def work() -> None:
            job.status = "running"
            try:
                result = runner(job)
            except Exception as exc:  # noqa: BLE001 - surfaced to the client as-is
                job.status = "failed"
                job.error = f"{type(exc).__name__}: {exc}"
                return
            job.result = result
            job.status = "cancelled" if job.cancel_event.is_set() else "done"

        threading.Thread(target=work, name=f"optimise-{job.job_id}", daemon=True).start()
        return job

    def get(self, job_id: str) -> Job | None:
        with self._lock:
            return self._jobs.get(job_id)

    def cancel(self, job_id: str) -> bool:
        job = self.get(job_id)
        if job is None or job.status not in ("queued", "running"):
            return False
        job.cancel_event.set()
        return True

    def _evict_locked(self) -> None:
        finished = [j for j in self._jobs.values() if j.status in ("done", "failed", "cancelled")]
        while len(self._jobs) >= self._max_jobs and finished:
            self._jobs.pop(finished.pop(0).job_id, None)
