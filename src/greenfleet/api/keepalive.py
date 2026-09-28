"""Keep a free-tier host awake by requesting the app's own public URL.

Free Render web services sleep after about 15 minutes without inbound traffic, and the
next visitor then waits about a minute for a cold start. While the app is running, a
daemon thread requests ``<public url>/api/health`` every ``QFLEET_KEEPALIVE_INTERVAL_S``
seconds (default 600). The request must go out through the public URL: that is what the
host counts as inbound traffic, and a request to localhost would not be.

The public URL comes from ``QFLEET_KEEPALIVE_URL``, or else from ``RENDER_EXTERNAL_URL``,
which Render sets on every web service. With neither set (a laptop, the test suite) the
pinger does not start. ``QFLEET_KEEPALIVE=0`` turns it off everywhere.

This keeps a running instance awake; it cannot wake one that has already been stopped.
"""

from __future__ import annotations

import logging
import os
import threading
import urllib.request
from collections.abc import Callable, Mapping

logger = logging.getLogger(__name__)

DEFAULT_INTERVAL_S = 600.0
HEALTH_PATH = "/api/health"


def _fetch(url: str) -> int:
    with urllib.request.urlopen(url, timeout=30) as response:  # noqa: S310 - our own URL
        return response.status


def target_url(env: Mapping[str, str] | None = None) -> str | None:
    """The health URL to ping, or None when keep-alive should not run."""
    env = os.environ if env is None else env
    if env.get("QFLEET_KEEPALIVE", "1").strip().lower() in ("0", "false", "no", "off"):
        return None
    base = (env.get("QFLEET_KEEPALIVE_URL") or env.get("RENDER_EXTERNAL_URL") or "").strip()
    if not base:
        return None
    return base.rstrip("/") + HEALTH_PATH


class KeepAlive:
    def __init__(self, url: str, interval_s: float = DEFAULT_INTERVAL_S,
                 fetch: Callable[[str], int] = _fetch):
        if interval_s <= 0:
            raise ValueError(f"interval must be positive, got {interval_s}")
        self.url, self.interval_s, self._fetch = url, interval_s, fetch
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, name="keepalive", daemon=True)

    def start(self) -> KeepAlive:
        self._thread.start()
        logger.info("keep-alive: requesting %s every %.0f s", self.url, self.interval_s)
        return self

    def stop(self) -> None:
        self._stop.set()
        self._thread.join(timeout=5)

    def _run(self) -> None:
        # Wait first: the app has just started, so it is awake already.
        while not self._stop.wait(self.interval_s):
            try:
                status = self._fetch(self.url)
                logger.debug("keep-alive: %s -> %s", self.url, status)
            except Exception as exc:  # a failed ping must never take the server down
                logger.warning("keep-alive: %s failed: %s", self.url, exc)


def start_from_env(env: Mapping[str, str] | None = None) -> KeepAlive | None:
    env = os.environ if env is None else env
    url = target_url(env)
    if url is None:
        return None
    interval = float(env.get("QFLEET_KEEPALIVE_INTERVAL_S", DEFAULT_INTERVAL_S))
    return KeepAlive(url, interval).start()
