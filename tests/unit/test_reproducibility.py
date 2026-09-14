"""Determinism and provenance tests. Maps to P-11, Q-06 and X-05."""

from __future__ import annotations

import json

import numpy as np
import pytest

from greenfleet.reproducibility import (
    BENCHMARK_SEEDS,
    RunContext,
    config_digest,
    set_global_seed,
)

pytestmark = pytest.mark.edgecase


class TestSeeding:
    def test_same_seed_gives_same_stream(self):
        """P-11: same seed, same data, same model."""
        a = set_global_seed(42).random(20)
        b = set_global_seed(42).random(20)
        np.testing.assert_array_equal(a, b)

    def test_different_seeds_give_different_streams(self):
        """Q-06: different seeds must produce a distribution, not one answer."""
        a = set_global_seed(1).random(20)
        b = set_global_seed(2).random(20)
        assert not np.array_equal(a, b)

    def test_returns_generator_not_legacy_state(self):
        rng = set_global_seed(7)
        assert isinstance(rng, np.random.Generator)

    def test_bool_is_not_accepted_as_seed(self):
        """bool is an int subclass; silently seeding with True would be a bug."""
        with pytest.raises(TypeError):
            set_global_seed(True)

    def test_non_int_seed_rejected(self):
        with pytest.raises(TypeError):
            set_global_seed(3.5)

    def test_large_seed_does_not_overflow(self):
        rng = set_global_seed(2**40)
        assert rng.random() is not None


class TestBenchmarkSeeds:
    def test_at_least_thirty_runs(self):
        """B-02: at least 30 independent runs per algorithm."""
        assert len(BENCHMARK_SEEDS) >= 30

    def test_seeds_are_unique(self):
        assert len(set(BENCHMARK_SEEDS)) == len(BENCHMARK_SEEDS)

    def test_seed_list_is_fixed(self):
        """B-01 fairness: every algorithm must see the identical seed set."""
        assert tuple(range(1000, 1030)) == BENCHMARK_SEEDS


class TestConfigDigest:
    def test_digest_is_stable_across_key_order(self):
        assert config_digest({"a": 1, "b": 2}) == config_digest({"b": 2, "a": 1})

    def test_digest_changes_when_a_factor_changes(self):
        """F-09: changing one emission factor must be visible in the audit trail."""
        base = {"ammonia": {"wtt_gco2e_per_mj": 121.0}}
        tweaked = {"ammonia": {"wtt_gco2e_per_mj": 120.0}}
        assert config_digest(base) != config_digest(tweaked)

    def test_digest_handles_nested_and_non_json_types(self):
        assert config_digest({"when": np.float64(1.5), "nested": {"x": [1, 2]}})


class TestRunContext:
    def test_captures_seed_and_stage(self):
        ctx = RunContext(seed=1234, stage="prediction.train")
        payload = ctx.as_dict()
        assert payload["seed"] == 1234
        assert payload["stage"] == "prediction.train"

    def test_includes_input_digest(self):
        ctx = RunContext(seed=1, stage="test", inputs={"gwp_set": "ar6_gwp100"})
        assert len(ctx.as_dict()["input_digest"]) == 16

    def test_identical_inputs_give_identical_digest(self):
        inputs = {"model": "residual_xgb", "gwp_set": "ar6_gwp100"}
        assert (
            RunContext(seed=1, stage="s", inputs=inputs).as_dict()["input_digest"]
            == RunContext(seed=2, stage="s", inputs=inputs).as_dict()["input_digest"]
        )

    def test_write_round_trips_to_json(self, tmp_path):
        """X-05: every run is logged so any result can be reproduced."""
        ctx = RunContext(seed=99, stage="optimize", inputs={"vessels": 20})
        path = ctx.write(tmp_path / "runs" / "ctx.json")
        assert path.exists()
        loaded = json.loads(path.read_text(encoding="utf-8"))
        assert loaded["seed"] == 99
        assert loaded["inputs"]["vessels"] == 20
        assert "started_at" in loaded
        assert "python_version" in loaded
