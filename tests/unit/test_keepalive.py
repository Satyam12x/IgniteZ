"""Keep-alive pinger: when it runs, what it requests, and that failures are harmless."""

from __future__ import annotations

import threading

import pytest

from greenfleet.api.keepalive import KeepAlive, start_from_env, target_url


class TestTargetUrl:
    def test_off_without_a_public_url(self):
        assert target_url({}) is None

    def test_uses_render_url(self):
        env = {"RENDER_EXTERNAL_URL": "https://q-fleet.onrender.com"}
        assert target_url(env) == "https://q-fleet.onrender.com/api/health"

    def test_explicit_url_wins_and_trailing_slash_is_dropped(self):
        env = {"RENDER_EXTERNAL_URL": "https://a.onrender.com", "QFLEET_KEEPALIVE_URL": "https://b.example/"}
        assert target_url(env) == "https://b.example/api/health"

    @pytest.mark.parametrize("off", ["0", "false", "OFF", "no"])
    def test_can_be_switched_off(self, off):
        env = {"RENDER_EXTERNAL_URL": "https://a.onrender.com", "QFLEET_KEEPALIVE": off}
        assert target_url(env) is None

    def test_start_from_env_does_nothing_locally(self):
        assert start_from_env({}) is None


class TestKeepAlive:
    def test_requests_the_url_repeatedly(self):
        calls, enough = [], threading.Event()

        def fetch(url):
            calls.append(url)
            if len(calls) >= 3:
                enough.set()
            return 200

        pinger = KeepAlive("https://x/api/health", interval_s=0.01, fetch=fetch).start()
        assert enough.wait(2)
        pinger.stop()
        assert set(calls) == {"https://x/api/health"}

    def test_a_failed_request_does_not_stop_it(self):
        calls, enough = [], threading.Event()

        def fetch(url):
            calls.append(url)
            if len(calls) >= 2:
                enough.set()
            raise OSError("network down")

        pinger = KeepAlive("https://x/api/health", interval_s=0.01, fetch=fetch).start()
        assert enough.wait(2)
        pinger.stop()

    def test_waits_one_interval_before_the_first_request(self):
        calls = []
        fetch = lambda u: calls.append(u) or 200  # noqa: E731
        pinger = KeepAlive("https://x/api/health", interval_s=60, fetch=fetch).start()
        pinger.stop()
        assert calls == []

    def test_rejects_a_non_positive_interval(self):
        with pytest.raises(ValueError):
            KeepAlive("https://x/api/health", interval_s=0)
