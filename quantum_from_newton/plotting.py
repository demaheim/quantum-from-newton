"""Shared look of the figures."""

import numpy as np

BLUE = "#4b63ec"
ORANGE = "#dd401d"
GREEN = "#019985"

MUTED = "0.7"  # for a curve that is context rather than result


def plot_trajectories(ax, t: np.ndarray, x_snaps: np.ndarray, every: int = 1) -> None:
    """One line per companion: its position against time.

    ``x_snaps`` is ``(n_snapshots, n_particles)`` as stored. ``every`` thins the
    lines for chains too dense to draw in full.
    """
    for trajectory in x_snaps.T[::every]:
        ax.plot(t, trajectory, c=GREEN, linewidth=0.5)


def plot_density(ax, x_link, P, x_ref, P_ref, *, initial=None) -> dict:
    """The chain's link density against the prediction of the Schroedinger equation.

    Returns the plotted handles keyed by label, so a caller can build one shared
    legend across several panels.
    """
    handles = {}
    if initial is not None:
        (handles["initial density"],) = ax.plot(x_ref, initial, c=MUTED, lw=0.9, ls=":")
    (handles["Schrödinger equation"],) = ax.plot(x_ref, P_ref, c=ORANGE, lw=1.2)
    (handles["companion chain"],) = ax.plot(x_link, P, "o", ms=3, c=GREEN)
    ax.set_ylabel(r"$P / P_{\mathrm{max}}$")
    return handles
