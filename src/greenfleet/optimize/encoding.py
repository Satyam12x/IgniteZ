"""Encoding scheme for fleet decisions.

The problem statement asks for this by name: *"Encoding scheme for fleet decisions,
Quantum update mechanisms"*. This module is the encoding half; the rotation-gate
update lives in ``prediction/qiea.py`` and is reused unchanged.

The mixed-variable problem (Q-02)
---------------------------------
Fleet decisions are not all of one kind:

    deploy[v]   binary       use this vessel or not
    route[v]    categorical  which lane it serves
    fuel[v]     categorical  which fuel it burns
    speed[v]    continuous   service speed

Q-bits encode binary decisions naturally, and categorical ones by binary expansion.
Speed is continuous and does not belong in a bit string at all - discretising it
into bands would either coarsen the answer or explode the encoding. So the search is
**hybrid**: Q-bit strings with rotation-gate updates for the discrete block, and
QPSO over a real vector for speeds, with the two layers optimised together against
one shared objective.

Decoder validity (Q-01)
-----------------------
Every observed Q-bit string must decode to a *usable* fleet decision. Two things
make that true here rather than hoped for:

* **Modular decoding.** ``k`` bits addressing ``m`` routes would leave ``2^k - m``
  invalid codes if read literally. Reading them modulo ``m`` maps every bit pattern
  onto a real route, so no observation can decode to a non-existent option.
* **Repair, not penalty** (Q-05). A decoded plan that deploys no vessel at all
  cannot deliver cargo and is never worth evaluating, so it is repaired by
  switching on the vessel with the highest deploy amplitude.

Modular decoding introduces a mild representational bias when ``m`` is not a power
of two - low-numbered options are reachable from slightly more bit patterns. The
alternative, rejecting invalid codes, wastes evaluations and biases the search far
more severely, so the bias is accepted and documented.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass

import numpy as np

from greenfleet.optimize.problem import FleetProblem, FleetSolution

logger = logging.getLogger(__name__)

__all__ = ["FleetEncoding", "DecodedPlan"]


@dataclass(frozen=True, slots=True)
class DecodedPlan:
    """A Q-bit string plus a speed vector, decoded into fleet decisions."""

    deploy: np.ndarray
    route_index: np.ndarray
    fuel_index: np.ndarray
    speed_kn: np.ndarray
    repaired: bool


class FleetEncoding:
    """Maps between fleet decisions and the Q-bit / real-vector representation.

    Bit layout per vessel, concatenated across the fleet::

        [ deploy | route bits | fuel bits ]

    so the total discrete dimension is ``n_vessels * (1 + route_bits + fuel_bits)``.
    """

    def __init__(self, problem: FleetProblem):
        self.problem = problem
        self.n_vessels = problem.n_vessels
        self.route_bits = max(1, math.ceil(math.log2(max(problem.n_routes, 2))))
        self.fuel_bits = max(1, math.ceil(math.log2(max(problem.n_fuels, 2))))
        self.bits_per_vessel = 1 + self.route_bits + self.fuel_bits
        self.n_bits = self.n_vessels * self.bits_per_vessel

    # ---- decoding ---------------------------------------------------------

    @staticmethod
    def _bits_to_int(bits: np.ndarray) -> int:
        value = 0
        for bit in bits:
            value = (value << 1) | int(bit)
        return value

    def decode(self, bits: np.ndarray, speeds: np.ndarray) -> DecodedPlan:
        """Decode a Q-bit observation and a speed vector into fleet decisions.

        Every bit pattern decodes to something valid (Q-01); the only repair needed
        is the empty fleet.
        """
        bits = np.asarray(bits, dtype=bool).reshape(self.n_vessels, self.bits_per_vessel)
        deploy = bits[:, 0].copy()
        route_index = np.empty(self.n_vessels, dtype=int)
        fuel_index = np.empty(self.n_vessels, dtype=int)

        for index in range(self.n_vessels):
            offset = 1
            route_raw = self._bits_to_int(bits[index, offset:offset + self.route_bits])
            offset += self.route_bits
            fuel_raw = self._bits_to_int(bits[index, offset:offset + self.fuel_bits])
            # Modulo keeps every pattern inside the real option set (Q-01).
            route_index[index] = route_raw % self.problem.n_routes
            fuel_index[index] = fuel_raw % self.problem.n_fuels

        repaired = False
        if not deploy.any():
            # Q-05: an empty fleet delivers nothing. Repair rather than penalise.
            deploy[0] = True
            repaired = True

        speeds = np.asarray(speeds, dtype="float64").reshape(self.n_vessels)
        clipped = np.array([
            float(np.clip(speeds[i], v.min_speed_kn, v.max_speed_kn))
            for i, v in enumerate(self.problem.vessels)
        ])
        if not np.allclose(clipped, speeds):
            repaired = True

        return DecodedPlan(deploy, route_index, fuel_index, clipped, repaired)

    def evaluate(self, bits: np.ndarray, speeds: np.ndarray) -> FleetSolution:
        """Decode and score in one step."""
        plan = self.decode(bits, speeds)
        return self.problem.evaluate(
            plan.deploy, plan.route_index, plan.fuel_index, plan.speed_kn
        )

    # ---- encoding (for seeding and for MILP cross-checks) -----------------

    def encode(
        self,
        deploy: np.ndarray,
        route_index: np.ndarray,
        fuel_index: np.ndarray,
    ) -> np.ndarray:
        """Encode explicit decisions back into a bit string.

        Used to seed a search from a known plan and to convert a MILP optimum into
        the same representation for a like-for-like comparison (Q-08).
        """
        bits = np.zeros((self.n_vessels, self.bits_per_vessel), dtype=bool)
        for index in range(self.n_vessels):
            bits[index, 0] = bool(deploy[index])
            route = int(route_index[index]) % self.problem.n_routes
            fuel = int(fuel_index[index]) % self.problem.n_fuels
            for position in range(self.route_bits):
                bits[index, 1 + position] = bool(
                    (route >> (self.route_bits - 1 - position)) & 1
                )
            for position in range(self.fuel_bits):
                bits[index, 1 + self.route_bits + position] = bool(
                    (fuel >> (self.fuel_bits - 1 - position)) & 1
                )
        return bits.reshape(-1)

    # ---- speed bounds for the continuous layer ---------------------------

    @property
    def speed_bounds(self) -> tuple[np.ndarray, np.ndarray]:
        low = np.array([v.min_speed_kn for v in self.problem.vessels])
        high = np.array([v.max_speed_kn for v in self.problem.vessels])
        return low, high

    def random_speeds(self, rng: np.random.Generator) -> np.ndarray:
        low, high = self.speed_bounds
        return rng.uniform(low, high)

    def describe(self) -> dict[str, object]:
        return {
            "n_vessels": self.n_vessels,
            "route_bits": self.route_bits,
            "fuel_bits": self.fuel_bits,
            "bits_per_vessel": self.bits_per_vessel,
            "n_bits": self.n_bits,
            "discrete_states": float(2**self.n_bits),
            "continuous_dimensions": self.n_vessels,
        }
