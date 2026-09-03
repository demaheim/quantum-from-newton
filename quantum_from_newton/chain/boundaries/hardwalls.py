"""Hard walls at +-b, closing the chain off at both ends."""

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class HardWalls:
    """Reflection in hard walls at +-b, with the ghost polarity flipped,
    that is the hard wall is a domain wall pinned at +-b.
    """

    b: float

    def get_positions(self, x: np.ndarray) -> np.ndarray:
        return np.concatenate(([-2.0 * self.b - x[0]], x, [2.0 * self.b - x[-1]]))

    def get_polarities(self, s: np.ndarray) -> np.ndarray:
        # flip the outer polarities
        return np.concatenate(([-s[0]], s, [-s[-1]]))

    def get_gaps(self, x: np.ndarray) -> np.ndarray:
        gaps = np.diff(self.get_positions(x))
        # used in ``integrate`` for time step determination
        return gaps

    def get_companion_gradient(self, grad: np.ndarray) -> np.ndarray:
        # ``grad`` arrives from acceleration: n+2 entries, one per
        # position including both ghosts. What the integrator needs
        # is one entry per real companion. Drop the ghosts,
        # reflect the gradient at the ends
        gradient = grad[1:-1].copy()
        gradient[0] -= grad[0]
        gradient[-1] -= grad[-1]
        return gradient
