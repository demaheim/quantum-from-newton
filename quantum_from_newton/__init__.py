"""Newtonian companion-particle chains that reproduce the predictions of the
Schroedinger equation.

A line of companion particles interacts through the signed inverse gaps to its
neighbours (``chain.interaction``). Nothing else is added: the chain obeys
Newton's equations, and the stationary densities and interference patterns of
quantum mechanics come out of it.
"""

from . import chain, plotting, results, systems

__all__ = ["chain", "plotting", "results", "systems"]
