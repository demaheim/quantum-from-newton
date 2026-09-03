"""A Double slit analogy: two Gaussian packets that spread, overlap and interfere."""

from dataclasses import dataclass

import numpy as np
from scipy import optimize, special

from ..chain.boundaries.vacuum import Vacuum
from ..chain.constants import HBAR, MASS


@dataclass(frozen=True)
class GaussianPair:
    """Two packets of width ``width``, centred at ``+-separation``."""

    separation: float
    width: float

    boundary = Vacuum()

    def force(self, x: np.ndarray) -> np.ndarray:
        """No external force: the packets evolve freely."""
        return np.zeros_like(x)

    def get_init_pos(self, n: int, p_start: float, p_end: float) -> np.ndarray:
        """``n`` companions at equally spaced quantiles of the initial density.

        Equal quantiles means uniform distributed probabilities (linspace).
        For each probability, the position is found via a bisection of
        the inverse of the initial probability density.
        """
        x = np.zeros(n)
        left, right = -0.1, 0.1
        # bisection: get left boundary
        while self._inverse_init_density(left) - p_start > 0.0:
            left *= 10
        for i, p in enumerate(np.linspace(p_start, p_end, n)):
            # bisection: get right boundary
            while self._inverse_init_density(right) - p < 0.0:
                right *= 10
            x[i] = optimize.brentq(
                lambda z, p=p: self._inverse_init_density(z) - p,
                left,
                right,
                xtol=4 * np.finfo(float).eps,
            )
            # bisection: set right boundary
            left = x[i]
        half = x[(n + 1) // 2 :]  # for odd n the centre companion stays where it is
        x[: n // 2] = -half[::-1]
        return x

    def _inverse_init_density(self, x: float) -> float:
        """Inverse initial probability density, used to place the companions.

        The two packets interfere already at t = 0, so this function
        carries the overlap term as well.
        """
        overlap = self._overlap()
        scale = np.sqrt(2) * self.width
        return (
            1
            / 2
            * 1
            / (1 + overlap)
            * (
                special.erf((x - self.separation) / scale) / 2
                + special.erf((x + self.separation) / scale) / 2
                + overlap * special.erf(x / scale)
                + 1
                + overlap
            )
        )

    def density(self, t: float, x: np.ndarray) -> np.ndarray:
        """|psi(x, t)|^2 for free evolution from the same initial state as get_init_pos() uses."""
        return np.abs(self.psi(t, np.asarray(x, dtype=float))) ** 2

    def psi(self, t: float, x: np.ndarray) -> np.ndarray:
        upper = self._packet(t, x, -1)
        lower = self._packet(t, x, +1)
        return (upper + lower) / self._norm()

    def _packet(self, t: float, x: np.ndarray, side: int) -> np.ndarray:
        offset = x + side * self.separation
        return self._prefactor(t) * np.exp(
            -(offset**2) / (4 * self.width**2 * (1 + 1j * self._spread(t)))
        )

    def _norm(self) -> float:
        # |upper + lower|^2 integrated: the two packets plus their cross term.
        return np.sqrt(2 + 2 * self._overlap())

    def _overlap(self) -> float:
        """How much the two packets share at t = 0."""
        return np.exp(-(self.separation**2) / (2 * self.width**2))

    def _spread(self, t: float) -> float:
        """Dimensionless spreading of a free packet."""
        return HBAR * t / (2 * MASS * self.width**2)

    def _prefactor(self, t: float) -> complex:
        return 1 / np.sqrt(np.sqrt(2 * np.pi) * self.width * (1 + 1j * self._spread(t)))
