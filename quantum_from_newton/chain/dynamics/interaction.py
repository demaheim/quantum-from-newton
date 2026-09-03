"""The interaction between companion particles.

Each companion compares the signed inverse gap on its left with the one on its
right, and the interaction charges the mismatch:

    u_l      = s_l s_{l+1} / (x_{l+1} - x_l)              (signed inverse gap)
    V_int    = kappa * sum_k (u_k - u_{k-1})^2 .          kappa = hbar^2 / 8m

``acceleration`` is the negative analytic gradient of ``V_int`` / MASS.
"""

import numpy as np

from ..boundaries.boundary import Boundary
from ..constants import KAPPA, MASS


def get_link_quantity(x: np.ndarray, s: np.ndarray) -> np.ndarray:
    """Signed inverse gaps u_l = s_l s_{l+1} / (x_{l+1} - x_l), one per link."""
    return s[:-1] * s[1:] / np.diff(x)


def get_link_density(x: np.ndarray):
    """``(midpoints, P)`` for the chain ``x``, where P_k = 1/(x_{k+1} - x_k)."""
    dx = np.diff(x)
    return x[:-1] + dx / 2, 1.0 / dx


def energy(x: np.ndarray, s: np.ndarray, boundary: Boundary) -> float:
    """Interaction energy V_int of the chain."""
    _, _, _, u = _get_extended_links(x, s, boundary)
    return float(KAPPA * np.sum(np.diff(u) ** 2))


def acceleration(x: np.ndarray, s: np.ndarray, boundary: Boundary) -> np.ndarray:
    """Acceleration -dV_int/dx_k / m of every companion (analytic gradient)."""
    X, dX, pol_product, u = _get_extended_links(x, s, boundary)
    link_difference = np.diff(u)

    # dV/du_l, from the two squared link differences. A link out
    # to a ghost at infinity falls out below on its own: 1/dX^2 there is zero.
    dV_du = np.zeros(u.size)
    dV_du[1:] += 2.0 * KAPPA * link_difference
    dV_du[:-1] -= 2.0 * KAPPA * link_difference

    # du_l/ddX_l = -pol_product_l / dX_l^2, then ddX_l/dX_{l+1} = +1, ddX_l/dX_l = -1.
    dV_ddX = dV_du * (-pol_product / dX**2)
    grad = np.zeros(X.size)
    grad[1:] += dV_ddX
    grad[:-1] -= dV_ddX

    return -boundary.get_companion_gradient(grad) / MASS


def _get_extended_links(x: np.ndarray, s: np.ndarray, boundary: Boundary):
    """Positions, gaps, polarity products and link quantities of the extended chain."""
    X = boundary.get_positions(x)
    S = boundary.get_polarities(s)
    dX = np.diff(X)
    pol_prod = S[:-1] * S[1:]
    return X, dX, pol_prod, pol_prod / dX
