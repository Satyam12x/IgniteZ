"""Policy KPIs: Harit Sagar targets and the carbon-price lever.

S-06 asks for emissions per tonne of cargo reported against the Harit Sagar targets
- 30% below baseline by 2030 and 70% by 2047. That is the number an Indian port
authority is actually measured on, so it is the number the dashboard should lead
with, not raw tonnes.

S-02 adds the carbon price: as the price rises the fleet mix should shift toward
cleaner fuels, and if it does not, something is broken.
"""

from __future__ import annotations

import bisect
import logging
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger(__name__)

__all__ = ["HaritSagarStatus", "HaritSagarTracker", "carbon_cost_inr"]


@dataclass(frozen=True, slots=True)
class HaritSagarStatus:
    """Progress against the Harit Sagar intensity target for one year."""

    year: int
    baseline_year: int
    baseline_intensity: float
    """tCO2e per tonne of cargo in the baseline year."""

    current_intensity: float
    required_intensity: float
    achieved_reduction: float
    """Fraction below baseline actually achieved."""

    required_reduction: float
    """Fraction below baseline the target demands for this year."""

    on_track: bool

    @property
    def gap(self) -> float:
        """Reduction still needed, as a fraction of baseline. Zero when on track.

        Snapped to zero below 1e-12 so that exactly meeting the target reports a
        clean zero rather than floating-point dust on a dashboard.
        """
        shortfall = self.required_reduction - self.achieved_reduction
        return 0.0 if shortfall < 1e-12 else shortfall

    @property
    def headroom_intensity(self) -> float:
        """How much intensity must still be removed, in tCO2e per tonne of cargo."""
        return max(0.0, self.current_intensity - self.required_intensity)

    def summary(self) -> str:
        verdict = "ON TRACK" if self.on_track else "OFF TRACK"
        return (
            f"{self.year}: {100 * self.achieved_reduction:.1f}% below "
            f"{self.baseline_year} baseline, target "
            f"{100 * self.required_reduction:.1f}% - {verdict}"
        )


class HaritSagarTracker:
    """Tracks emission intensity against the Harit Sagar glide path (S-06)."""

    def __init__(self, policy: dict[str, Any]):
        """
        Args:
            policy: the ``policy.harit_sagar`` block from ``config/ports.yaml``.
        """
        # Accept either the whole policy block or just the harit_sagar sub-block.
        config = policy.get("harit_sagar", policy)
        self.baseline_year = int(config["baseline_year"])
        self.targets = {int(y): float(v) for y, v in config["targets"].items()}
        if not self.targets:
            raise ValueError("Harit Sagar config defines no targets")
        self.source = config.get("source", "")
        self.confidence = config.get("confidence", "provisional")
        self._years = sorted(self.targets)

    def required_reduction(self, year: int) -> float:
        """Fraction below baseline required in ``year``.

        Interpolated linearly between milestone years: the targets are endpoints of
        a transition, not step changes, so a port needs to know its 2027 position
        even though only 2030 and 2047 are legislated.
        """
        if year <= self.baseline_year:
            return 0.0
        if year >= self._years[-1]:
            return self.targets[self._years[-1]]
        if year <= self._years[0]:
            # Between baseline and the first milestone, glide from zero.
            first = self._years[0]
            span = first - self.baseline_year
            weight = (year - self.baseline_year) / span if span else 1.0
            return weight * self.targets[first]
        index = bisect.bisect_right(self._years, year) - 1
        low, high = self._years[index], self._years[index + 1]
        span = high - low
        weight = (year - low) / span if span else 0.0
        return self.targets[low] + weight * (self.targets[high] - self.targets[low])

    def assess(
        self,
        year: int,
        emissions_t: float,
        cargo_t: float,
        baseline_intensity: float,
    ) -> HaritSagarStatus:
        """Compare achieved intensity against the required glide path.

        Args:
            year: the year being assessed.
            emissions_t: total CO2e for the period.
            cargo_t: tonnes of cargo handled in the period.
            baseline_intensity: tCO2e per tonne of cargo in the baseline year.
        """
        if cargo_t <= 0:
            raise ValueError(
                "cargo_t must be positive: emissions per tonne of cargo is undefined "
                "when no cargo moved (M-02 is the zero-demand case and is handled by "
                "the optimizer, not here)"
            )
        if baseline_intensity <= 0:
            raise ValueError("baseline_intensity must be positive")

        current = emissions_t / cargo_t
        required_reduction = self.required_reduction(year)
        required_intensity = baseline_intensity * (1.0 - required_reduction)
        achieved = 1.0 - current / baseline_intensity
        return HaritSagarStatus(
            year=year,
            baseline_year=self.baseline_year,
            baseline_intensity=baseline_intensity,
            current_intensity=current,
            required_intensity=required_intensity,
            achieved_reduction=achieved,
            required_reduction=required_reduction,
            on_track=achieved >= required_reduction - 1e-12,
        )

    def glide_path(self, years: list[int], baseline_intensity: float) -> dict[int, float]:
        """Required intensity per year, for plotting against actuals."""
        return {
            year: baseline_intensity * (1.0 - self.required_reduction(year))
            for year in years
        }


def carbon_cost_inr(
    emissions_t: float,
    price_inr_per_t: float,
) -> float:
    """Cost of emitting ``emissions_t`` at a given carbon price (S-02).

    Kept trivial deliberately: the carbon price is a scenario input, not a modelled
    quantity, and the useful test is whether raising it shifts the optimizer's fuel
    choice - which is an optimizer test, not an arithmetic one.
    """
    if emissions_t < 0:
        raise ValueError("emissions_t must be non-negative")
    if price_inr_per_t < 0:
        raise ValueError("carbon price must be non-negative")
    return emissions_t * price_inr_per_t
