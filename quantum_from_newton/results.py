"""Storage for what an experiment produced.

Each experiment writes its result to
``data/<name>.npz`` and the figure builder reads it back. The name is scoped by
system, so ``experiments/hosc/hosc_n1.py`` saves under ``hosc/hosc_n1`` and the
two trees mirror each other. That file is the only contract between
``experiments/`` and ``figures/``, and it holds the two things every panel of
the manuscript shows:

===============  ==================  ===================================================
field            shape               meaning
===============  ==================  ===================================================
``t_snaps``      (n_snap,)           times at which the chain was recorded
``x_snaps``      (n_snap, N)         companion positions -- the trajectory panel
``x_link``       (N-1,)              link midpoints (x_k + x_{k+1}) / 2
``P``            (N-1,)              the chain's link density there, over its maximum
``x_ref``        (n_grid,)           fine grid for the comparison curve
``P_ref``        (n_grid,)           |psi|^2 of the Schroedinger equation on that grid
``P_ref_initial`` (n_grid,)          optional: |psi|^2 at t = 0, for a time evolution
===============  ==================  ===================================================

A sweep experiment holds a family of relaxed chains rather than one trajectory:
one relaxation per placement of a migrating domain wall. ``save_sweep`` writes
those under the same tree, with its own contract:

=====================  ==============  ==============================================
field                  shape           meaning
=====================  ==============  ==============================================
``num_in_new_domain``  (n_sweep,)      companions in the growing domain, one per
                                       wall placement
``x_relaxed``          (n_sweep, N)    relaxed companion positions at each placement
``energy_ratio``       (n_sweep,)      interaction energy averaged over companions,
                                       over the Schroedinger ground energy
=====================  ==============  ==============================================
"""

from pathlib import Path

import numpy as np

DATA_DIR = Path(__file__).resolve().parent.parent / "data"


def save(
    name: str,
    *,
    t_snaps: np.ndarray,
    x_snaps: np.ndarray,
    x_link: np.ndarray,
    P: np.ndarray,
    x_ref: np.ndarray,
    P_ref: np.ndarray,
    P_ref_initial: np.ndarray | None = None,
) -> Path:
    """Write one experiment's result to ``data/<name>.npz``."""
    path = DATA_DIR / f"{name}.npz"
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = {
        "t_snaps": t_snaps,
        "x_snaps": x_snaps,
        "x_link": x_link,
        "P": P,
        "x_ref": x_ref,
        "P_ref": P_ref,
    }
    if P_ref_initial is not None:
        fields["P_ref_initial"] = P_ref_initial
    np.savez(path, **fields)
    return path


def save_sweep(
    name: str, *, num_in_new_domain: np.ndarray, x_relaxed: np.ndarray, energy_ratio: np.ndarray
) -> Path:
    """Write one sweep's result to ``data/<name>.npz``."""
    path = DATA_DIR / f"{name}.npz"
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        path, num_in_new_domain=num_in_new_domain, x_relaxed=x_relaxed, energy_ratio=energy_ratio
    )
    return path


def load(name: str):
    """Read back what ``save`` wrote, by the same name."""
    path = DATA_DIR / f"{name}.npz"
    if not path.exists():
        raise FileNotFoundError(f"{path} is missing -- run experiments/{name}.py first")
    return np.load(path)
