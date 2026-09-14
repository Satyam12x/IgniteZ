"""Phase 4: the decision-support API.

These exercise the HTTP surface end to end through the real engines. They do
not re-test the engines - the Phase 1-3 suites do that - but they do assert the
properties a dashboard user relies on: an infeasible fleet is refused with the
reason, a finished plan really delivers the demand, a cancelled run still
returns what it found.
"""

from __future__ import annotations

import time

import numpy as np
import pytest

fastapi = pytest.importorskip("fastapi")
pytest.importorskip("httpx")

from fastapi.testclient import TestClient  # noqa: E402

from greenfleet.api.app import create_app  # noqa: E402
from greenfleet.api.state import load_state  # noqa: E402


@pytest.fixture(scope="module")
def client():
    state = load_state()
    with TestClient(create_app(state)) as c:
        c.state_ = state
        yield c


def wait_for(client, job_id, timeout_s=60.0):
    deadline = time.perf_counter() + timeout_s
    while time.perf_counter() < deadline:
        status = client.get(f"/api/optimise/{job_id}").json()
        if status["status"] in ("done", "failed", "cancelled"):
            return status
        time.sleep(0.05)
    raise TimeoutError(f"job {job_id} did not finish")


# ---- health & meta ----------------------------------------------------------


def test_health_reports_model_and_ports(client):
    body = client.get("/api/health").json()
    assert body["status"] == "ok"
    assert "loaded" in body["model"]
    assert "mormugao" in body["ports"]


def test_meta_exposes_everything_the_dashboard_needs(client):
    meta = client.get("/api/meta").json()
    assert set(meta["fuels"]) >= {
        "hfo", "mgo", "lng", "methanol", "ammonia", "hydrogen", "electricity",
    }
    assert meta["fuels"]["ammonia"]["pathways"].keys() >= {"grey", "green"}
    assert len(meta["ports"]) == 8
    assert meta["demo_site"] == "mormugao"
    assert set(meta["scenarios"]) == {"coastal_2047", "coastal", "gttp_tug", "tiny"}
    assert meta["scenarios"]["coastal_2047"]["year"] == 2047
    assert meta["scenarios"]["coastal_2047"]["fuel_pathways"]["methanol"] == "e_fuel"
    assert isinstance(meta["electrification_crossover_year"], int)
    # Availability is reported per milestone year, and "planned" is a distinct state.
    assert meta["ports"]["mormugao"]["fuel_availability"]["methanol"]["2030"] == "planned"


# ---- prediction -------------------------------------------------------------


def test_predict_is_physically_consistent(client):
    if client.state_.model is None:
        pytest.skip("no trained model in artifacts/model")
    r = client.post("/api/predict", json={"vessel_class": "bulk_carrier",
                                          "design_efficiency_gco2_per_t_nmi": 5.0})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["cold_start"] is False
    power = [p["power_kw"] for p in body["points"]]
    assert power[0] > 0                                     # P-03: hotel load at rest
    assert all(np.diff(power) > 0)                          # P-02: strictly increasing
    energies = [p["energy_per_nmi_mj"] for p in body["points"]]
    assert energies[0] is None                              # undefined at zero speed
    finite = [e for e in energies if e is not None]
    assert min(finite) < finite[-1]                         # interior minimum exists
    for p in body["points"][1:]:
        assert p["p10_energy_per_nmi_mj"] <= p["energy_per_nmi_mj"] <= p["p90_energy_per_nmi_mj"]
        assert p["fuel_t_per_1000_nmi"]["methanol"] > p["fuel_t_per_1000_nmi"]["mgo"]  # lower LHV


def test_predict_flags_cold_start_for_tugs(client):
    if client.state_.model is None:
        pytest.skip("no trained model in artifacts/model")
    r = client.post("/api/predict", json={"vessel_class": "harbour_tug"})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["cold_start"] is True
    assert "certificate" in body["cold_start_reason"]
    mid = body["points"][10]
    width = mid["p90_energy_per_nmi_mj"] / mid["p10_energy_per_nmi_mj"]
    assert width > 1.5, "cold-start band should be visibly wide"


def test_predict_rejects_bad_input(client):
    r = client.post("/api/predict", json={"vessel_class": "bulk_carrier", "speeds_kn": [-1.0]})
    assert r.status_code == 422
    r = client.post("/api/predict", json={"vessel_class": "bulk_carrier", "unknown_field": 1})
    assert r.status_code == 422


# ---- emissions --------------------------------------------------------------


def test_compare_shows_the_lifecycle_flip(client):
    r = client.post("/api/emissions/compare", json={})
    assert r.status_code == 200, r.text
    rows = {row["label"]: row for row in r.json()["rows"]}
    diesel = rows["Diesel (MGO)"]
    grey_nh3 = rows["Grey ammonia"]
    assert grey_nh3["stack_only_t"] == 0.0                  # no carbon in the molecule
    assert grey_nh3["total_t"] > diesel["total_t"]          # but worse over the lifecycle
    assert grey_nh3["verdict_flips"] is True
    assert rows["Green ammonia"]["verdict_flips"] is False
    assert rows["Green ammonia"]["vs_baseline"] < -0.5
    assert rows["Battery-electric (grid)"]["local_pollutants_kg"]["nox_kg"] == 0.0
    assert diesel["local_pollutants_kg"]["nox_kg"] > 1000


def test_compare_rejects_unknown_pathway(client):
    r = client.post("/api/emissions/compare",
                    json={"options": [{"fuel": "methanol", "pathway": "grey"}]})
    assert r.status_code == 422
    assert "pathway" in r.json()["detail"]


def test_electrification_series(client):
    body = client.get("/api/emissions/electrification").json()
    assert len(body["years"]) == len(body["electric_t_per_tj"])
    assert body["crossover_year"] in body["years"]
    index = body["years"].index(body["crossover_year"])
    assert body["electric_t_per_tj"][index - 1] > body["baseline_t_per_tj"]
    assert body["electric_t_per_tj"][index] <= body["baseline_t_per_tj"]


# ---- optimisation -----------------------------------------------------------


def test_optimise_tiny_returns_feasible_plans_with_provenance(client):
    r = client.post("/api/optimise", json={"scenario": "tiny", "n_individuals": 8,
                                           "n_generations": 20, "compare_with": ["greedy"]})
    assert r.status_code == 202, r.text
    status = wait_for(client, r.json()["job_id"])
    assert status["status"] == "done", status.get("error")
    result = status["result"]
    assert result["n_evaluations"] == 160
    assert result["plans"], "the tiny instance is always satisfiable"
    demand = 250_000.0
    for plan in result["plans"]:
        assert sum(plan["cargo_delivered_t"].values()) >= demand - 1e-6
        assert plan["n_deployed"] == sum(a["deployed"] for a in plan["assignments"])
        assert plan["harit_sagar"]["baseline"] == "same deployment on MGO"
    assert set(result["extremes"]) == {"least_energy", "cleanest", "cheapest", "balanced"}
    cheapest = result["plans"][result["extremes"]["cheapest"]]
    assert cheapest["cost_inr"] == min(p["cost_inr"] for p in result["plans"])
    assert result["baselines"][0]["algorithm"] == "greedy"
    assert result["hypervolume"] > 0
    assert result["problem"]["year"] == 2030
    assert "energy_source" in result


def test_optimise_keeps_scenario_settings_unless_overridden(client):
    r = client.post("/api/optimise", json={"scenario": "coastal_2047", "n_individuals": 4,
                                           "n_generations": 2})
    status = wait_for(client, r.json()["job_id"])
    assert status["status"] == "done", status.get("error")
    problem = status["result"]["problem"]
    assert problem["year"] == 2047
    assert problem["fuel_pathways"]["methanol"] == "e_fuel"
    assert problem["carbon_price_inr_per_t"] == 0.0

    r = client.post("/api/optimise", json={"scenario": "coastal_2047", "year": 2030,
                                           "carbon_price_inr_per_t": 500,
                                           "n_individuals": 4, "n_generations": 2})
    status = wait_for(client, r.json()["job_id"])
    problem = status["result"]["problem"]
    assert problem["year"] == 2030
    assert problem["carbon_price_inr_per_t"] == 500.0


def test_optimise_refuses_impossible_demand_with_the_shortfall(client):
    r = client.post("/api/optimise", json={
        "scenario": "custom",
        "vessels": [{"name": "A", "vessel_class": "general_cargo", "capacity_t": 1000,
                     "min_speed_kn": 8, "max_speed_kn": 12, "tank_volume_m3": 100}],
        "routes": [{"name": "R", "ports": ["mumbai", "cochin"], "distance_nmi": 560,
                    "annual_demand_t": 9e9}],
    })
    assert r.status_code == 422
    detail = r.json()["detail"]
    assert detail.startswith("infeasible")
    assert "short by" in detail


def test_optimise_rejects_unknown_port_and_scenario(client):
    r = client.post("/api/optimise", json={
        "scenario": "custom",
        "vessels": [{"name": "A", "vessel_class": "general_cargo", "capacity_t": 8000,
                     "min_speed_kn": 8, "max_speed_kn": 12, "tank_volume_m3": 800}],
        "routes": [{"name": "R", "ports": ["mumbai", "atlantis"], "distance_nmi": 560,
                    "annual_demand_t": 1000}],
    })
    assert r.status_code == 422
    assert "atlantis" in r.json()["detail"]
    assert client.post("/api/optimise", json={"scenario": "mars"}).status_code == 422


def test_optimise_can_be_cancelled_and_still_returns_the_archive(client):
    r = client.post("/api/optimise", json={"scenario": "coastal", "n_individuals": 20,
                                           "n_generations": 2000})
    job_id = r.json()["job_id"]
    time.sleep(0.4)
    assert client.post(f"/api/optimise/{job_id}/cancel").status_code == 200
    status = wait_for(client, job_id)
    assert status["status"] == "cancelled"
    assert status["result"]["time_limited"] is True
    assert status["result"]["generations"] < 2000
    # Cancelling twice is not an error worth a 500; it is a 409.
    assert client.post(f"/api/optimise/{job_id}/cancel").status_code == 409


def test_unknown_job_is_404(client):
    assert client.get("/api/optimise/nope").status_code == 404


def test_dashboard_and_vendor_bundle_are_served(client):
    index = client.get("/")
    assert index.status_code == 200
    assert "Green Fleet Planner" in index.text
    assert client.get("/static/app.js").status_code == 200
    plotly = client.get("/vendor/plotly.min.js")
    assert plotly.status_code == 200
    assert len(plotly.content) > 1_000_000


# ---- the speed-curve cache the optimizer relies on ----------------------------


def test_speed_curve_matches_full_prediction(client):
    """The cached per-vessel curve must reproduce predict() exactly (P-02/P-03 by construction)."""
    if client.state_.model is None:
        pytest.skip("no trained model in artifacts/model")
    import pandas as pd

    from greenfleet.optimize.scenarios import coastal_scenario

    model = client.state_.model
    problem = coastal_scenario(energy_model=model)
    vessel = problem.vessels[0]
    features = pd.DataFrame([problem.vessel_features(vessel)])
    for speed in (0.0, 6.0, 12.0, 15.0):
        direct = model.predict(features, speed_kn=speed)
        curve = problem.speed_curve(vessel)
        assert np.isclose(curve.power_kw(speed), direct.power_kw[0], rtol=0, atol=1e-9)
        if speed > 0:
            assert np.isclose(curve.energy_per_nmi_mj(speed), direct.energy_per_nmi_mj[0],
                              rtol=0, atol=1e-9)
        else:
            assert np.isinf(curve.energy_per_nmi_mj(speed))
