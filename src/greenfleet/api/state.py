"""Process-wide engines: registries, the trained model, and their provenance.

Loaded once at startup. The model is the artefact written by
``scripts/train_model.py``; if it is absent the platform still runs on the
physics-only fallback, but says so in every response rather than pretending.
"""

from __future__ import annotations

import json
import logging
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import joblib

from greenfleet.config.loader import FuelRegistry, ProvisionalFactorWarning, load_fuel_registry
from greenfleet.emissions.kpi import HaritSagarTracker
from greenfleet.emissions.ports import PortRegistry, load_port_registry
from greenfleet.emissions.tank import TankCalculator
from greenfleet.emissions.wtw import WellToWakeCalculator
from greenfleet.prediction.features import VesselClassMap, load_vessel_classes
from greenfleet.prediction.model import EnergyModel

logger = logging.getLogger(__name__)

MODEL_DIR = Path("artifacts/model")


@dataclass
class AppState:
    registry: FuelRegistry
    ports: PortRegistry
    classes: VesselClassMap
    calculator: WellToWakeCalculator
    tank: TankCalculator
    harit_sagar: HaritSagarTracker
    model: EnergyModel | None = None
    model_card: dict[str, Any] = field(default_factory=dict)

    @property
    def energy_source(self) -> str:
        if self.model is None:
            return "physics-only fallback (no trained model found; run scripts/train_model.py)"
        panel = self.model_card.get("panel", {})
        return (
            f"{self.model_card.get('name', 'trained model')} trained on "
            f"{panel.get('n_ship_years', '?'):,} EU MRV ship-years "
            f"({panel.get('n_ships', '?'):,} ships, {min(panel.get('reporting_periods', [0]))}-"
            f"{max(panel.get('reporting_periods', [0]))})"
        )

    def model_summary(self) -> dict[str, Any]:
        if self.model is None:
            return {"loaded": False, "source": self.energy_source}
        return {"loaded": True, "source": self.energy_source, **self.model_card}


def load_state(model_dir: Path | str = MODEL_DIR) -> AppState:
    # ProvisionalFactorWarning exists to stop a provisional number being published
    # unnoticed. The API reports confidence on every emission row instead, so in a
    # server process the warning is noise on every request.
    warnings.filterwarnings("ignore", category=ProvisionalFactorWarning)
    registry = load_fuel_registry()
    ports = load_port_registry()
    model_dir = Path(model_dir)
    model, card = None, {}
    model_path = model_dir / "energy_model.joblib"
    if model_path.exists():
        model = joblib.load(model_path)
        card_path = model_dir / "model_card.json"
        if card_path.exists():
            card = json.loads(card_path.read_text(encoding="utf-8"))
        logger.info("loaded energy model from %s", model_path)
    else:
        logger.warning("no trained model at %s; serving the physics fallback", model_path)
    return AppState(
        registry=registry,
        ports=ports,
        classes=load_vessel_classes(),
        calculator=WellToWakeCalculator(registry),
        tank=TankCalculator(registry),
        harit_sagar=HaritSagarTracker(ports.policy),
        model=model,
        model_card=card,
    )
