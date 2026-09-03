"""Harmonic oscillator: V = m w^2 x^2 / 2, with vacuum beyond the chain ends."""

from dataclasses import dataclass

import numpy as np

from ..chain.boundaries.vacuum import Vacuum
from ..chain.constants import HBAR, MASS


@dataclass(frozen=True)
class Harmonic:
    """An oscillator of angular frequency ``omega`` (1/ps)."""

    omega: float

    boundary = Vacuum()

    @property
    def scale(self) -> float:
        """Gaussian width sigma = sqrt(hbar / 2 m w) of the stationary states."""
        return float(np.sqrt(HBAR / (2 * MASS * self.omega)))

    def get_uniform_start(self, n: int, extent: float) -> np.ndarray:
        """``n`` companions spread evenly over ``[-extent, extent]``."""
        return np.linspace(-extent, extent, n)

    def potential(self, x: np.ndarray) -> np.ndarray:
        return 0.5 * MASS * self.omega**2 * np.asarray(x, dtype=float) ** 2

    def force(self, x: np.ndarray) -> np.ndarray:
        return -MASS * self.omega**2 * np.asarray(x, dtype=float)

    def eigenstate_density(self, n: int, x: np.ndarray) -> np.ndarray:
        """|psi_n|^2 of the Schroedinger equation, scaled to a maximum of 1."""
        y = np.asarray(x, dtype=float) / (np.sqrt(2.0) * self.scale)
        coefficients = np.zeros(n + 1)
        coefficients[n] = 1.0
        density = np.polynomial.hermite.hermval(y, coefficients) ** 2 * np.exp(-(y**2))
        return density / density.max()
