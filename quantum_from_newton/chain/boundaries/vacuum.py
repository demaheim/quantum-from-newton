"""Nothing beyond the ends of the chain."""

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Vacuum:
    """Nothing beyond the ends: the ghosts sit at +-infinity."""

    def get_positions(self, x: np.ndarray) -> np.ndarray:
        return np.concatenate(([-np.inf], x, [np.inf]))

    def get_polarities(self, s: np.ndarray) -> np.ndarray:
        # outer polarities are repeated
        return np.pad(s, 1, mode="edge")

    def get_gaps(self, x: np.ndarray) -> np.ndarray:
        gaps = np.diff(self.get_positions(x))
        # omit infinite gaps: they stay constant:
        # do not help ``integrate`` for time step determination
        return gaps[np.isfinite(gaps)]

    def get_companion_gradient(self, grad: np.ndarray) -> np.ndarray:
        # ``grad`` arrives from acceleration: n+2 entries, one per
        # position including both ghosts. What the integrator needs
        # is one entry per real companion. Just drop the ghosts,
        # and apply no back-action because these ghost do not move.
        return grad[1:-1]
