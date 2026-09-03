"""One module per physical system.

Each holds everything specific to that system in one place: the boundary
conditions, any external force, a possible starting positions, and the
solution of the Schroedinger equation the result is compared against. Adding a
system means adding one module here.
"""

from .box import Box
from .gaussian_pair import GaussianPair
from .harmonic import Harmonic

__all__ = ["Box", "GaussianPair", "Harmonic"]
