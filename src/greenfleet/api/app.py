"""The decision-support API and the dashboard that sits on it.

    python -m greenfleet.api            # http://127.0.0.1:8000

Routes are thin: validate, call the service, return. Errors that mean "your input
cannot work" (an infeasible fleet, an unknown port) come back as 422 with the
engine's own message, because that message says what to change.
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from greenfleet import __version__
from greenfleet.api import schemas, service
from greenfleet.api.jobs import JobStore
from greenfleet.api.state import AppState, load_state
from greenfleet.optimize.problem import Infeasible

logger = logging.getLogger(__name__)

STATIC_DIR = Path(__file__).parent / "static"


def _plotly_js() -> Path | None:
    """The Plotly bundle shipped inside the installed package, served locally.

    The platform runs on a port's own server with no internet; a CDN script tag
    would leave the dashboard blank there.
    """
    try:
        import plotly
    except ImportError:
        return None
    candidate = Path(plotly.__file__).parent / "package_data" / "plotly.min.js"
    return candidate if candidate.exists() else None


def create_app(state: AppState | None = None) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.engines = state or load_state()
        app.state.jobs = JobStore()
        logger.info("energy source: %s", app.state.engines.energy_source)
        yield

    app = FastAPI(
        title="Green Fleet Optimizer",
        version=__version__,
        description="Quantum-inspired fuel prediction and green fleet optimisation (SIH26138).",
        lifespan=lifespan,
    )

    def engines() -> AppState:
        return app.state.engines

    def _status(job) -> schemas.JobStatus:
        return schemas.JobStatus(
            job_id=job.job_id, status=job.status, request=job.request,
            progress=job.progress, result=job.result, error=job.error,
        )

    # ---- health & metadata ------------------------------------------------

    @app.get("/api/health")
    def health() -> dict:
        s = engines()
        return {
            "status": "ok",
            "version": __version__,
            "model": s.model_summary(),
            "gwp_set": s.registry.gwp_set_name,
            "ports": sorted(s.ports.ports),
        }

    @app.get("/api/meta")
    def meta() -> dict:
        return service.meta(engines())

    # ---- prediction -------------------------------------------------------

    @app.post("/api/predict", response_model=schemas.PredictResponse)
    def predict(request: schemas.PredictRequest) -> schemas.PredictResponse:
        try:
            return service.predict(engines(), request)
        except RuntimeError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc
        except (KeyError, ValueError) as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

    # ---- emissions --------------------------------------------------------

    @app.post("/api/emissions/compare", response_model=schemas.EmissionsCompareResponse)
    def compare(request: schemas.EmissionsCompareRequest) -> schemas.EmissionsCompareResponse:
        try:
            return service.compare_emissions(engines(), request)
        except (KeyError, ValueError) as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

    @app.get("/api/emissions/electrification")
    def electrification(baseline_fuel: str = "mgo") -> dict:
        try:
            return service.electrification_series(engines(), baseline_fuel)
        except (KeyError, ValueError) as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

    # ---- optimisation -----------------------------------------------------

    @app.post("/api/optimise", response_model=schemas.JobStatus, status_code=202)
    def optimise(request: schemas.OptimiseRequest) -> schemas.JobStatus:
        # Validate the problem synchronously so an impossible request fails now,
        # with the engine's explanation, instead of as a failed job later.
        try:
            service.build_problem(engines(), request)
        except Infeasible as exc:
            raise HTTPException(status_code=422, detail=f"infeasible: {exc}") from exc
        except (KeyError, ValueError) as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        job = app.state.jobs.submit(request, lambda j: service.run_optimisation(engines(), j))
        return _status(job)

    @app.get("/api/optimise/{job_id}", response_model=schemas.JobStatus)
    def job_status(job_id: str) -> schemas.JobStatus:
        job = app.state.jobs.get(job_id)
        if job is None:
            raise HTTPException(status_code=404, detail=f"no job {job_id}")
        return _status(job)

    @app.post("/api/optimise/{job_id}/cancel")
    def cancel(job_id: str) -> dict:
        if not app.state.jobs.cancel(job_id):
            raise HTTPException(status_code=409, detail="job is not running")
        return {"job_id": job_id, "cancelling": True}

    # ---- dashboard --------------------------------------------------------

    plotly = _plotly_js()
    if plotly is not None:
        @app.get("/vendor/plotly.min.js", include_in_schema=False)
        def plotly_bundle() -> FileResponse:
            return FileResponse(plotly, media_type="application/javascript")

    @app.get("/", include_in_schema=False)
    def index() -> FileResponse:
        return FileResponse(STATIC_DIR / "index.html")

    @app.exception_handler(Infeasible)
    async def infeasible_handler(_, exc: Infeasible) -> JSONResponse:
        return JSONResponse(status_code=422, content={"detail": f"infeasible: {exc}"})

    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
    return app


app = create_app()
