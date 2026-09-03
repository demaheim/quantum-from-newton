"""The contract every boundary rule satisfies."""

from typing import Protocol

import numpy as np


class Boundary(Protocol):
    """What lies beyond the ends of the chain.

    ``Vacuum`` and ``HardWalls`` satisfy this by shape alone, so a new rule for the
    outside only has to supply these four methods.
    """

    def get_positions(self, x: np.ndarray) -> np.ndarray:
        """Where the companions of the extended chain sit: the real ones, plus a
        ghost beyond each end. A ghost may sit at infinity.
        """
        ...

    def get_polarities(self, s: np.ndarray) -> np.ndarray:
        """The matching polarities of those companions."""
        ...

    def get_gaps(self, x: np.ndarray) -> np.ndarray:
        """The distance between each pair of neighbours. This is what ``integrate``
        calls at every step to determine the time step.
        """
        ...

    def get_companion_gradient(self, grad: np.ndarray) -> np.ndarray:
        """The gradient with respect to the real companions: one entry each."""
        ...
