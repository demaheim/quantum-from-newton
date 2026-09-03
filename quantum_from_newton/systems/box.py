"""Particle in a box: hard walls at +-b, no external force inside."""

from dataclasses import dataclass

import numpy as np

from ..chain.boundaries.hardwalls import HardWalls
from ..chain.constants import KAPPA


@dataclass(frozen=True)
class Box:
    """A box of half-width ``b`` (nm), empty inside."""

    b: float

    @property
    def boundary(self) -> HardWalls:
        return HardWalls(self.b)

    def get_uniform_start(self, n: int) -> np.ndarray:
        """``n`` companions spread evenly across the box."""
        return np.linspace(-self.b, self.b, n + 2)[1:-1].copy()

    def force(self, x: np.ndarray) -> np.ndarray:
        """No external force: the box is flat between its walls."""
        return np.zeros_like(x)

    def eigenstate_density(self, n: int, x: np.ndarray) -> np.ndarray:
        """|psi_n|^2 of the Schroedinger equation, scaled to a maximum of 1."""
        k = n * np.pi / (2 * self.b)
        wave = np.cos(k * x) if n % 2 else np.sin(k * x)
        return wave**2

    def eigenstate_energy(self, n: int) -> float:
        """Energy the Schroedinger equation gives one particle in state ``n``.

        n^2 pi^2 hbar^2 / 2 m L^2 with L = 2b, which in terms of the interaction's
        own coupling is 4 pi^2 kappa n^2 / L^2.
        """
        return n**2 * 4 * np.pi**2 * KAPPA / (2 * self.b) ** 2
