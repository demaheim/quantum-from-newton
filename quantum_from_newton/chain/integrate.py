"""Time integration of the chain."""

from dataclasses import dataclass

import numpy as np

from .boundaries.boundary import Boundary
from .constants import MASS
from .dynamics.interaction import acceleration
from .dynamics.stepping import DEFAULT_STEP, Snapshots, StepControl, StepPolicy


@dataclass(frozen=True)
class Trajectory:
    """Result of a run: the final chain, plus the recorded history."""

    x: np.ndarray  # final companion positions,      (n_particles,)
    v: np.ndarray  # final companion velocities,     (n_particles,)
    t_snaps: np.ndarray  # times of the snapshots,   (n_snapshots,)
    x_snaps: np.ndarray  # positions at those times, (n_snapshots, n_particles)


def integrate(
    x0: np.ndarray,
    *,
    v0: np.ndarray | None = None,
    ext_force,
    boundary: Boundary,
    polarities: np.ndarray,
    t_end: float,
    friction: float = 0.0,
    step: StepPolicy = DEFAULT_STEP,
    snapshot_every: int = 200,
) -> Trajectory:
    """This integration evolves a chain with initial particle
    positions ``x0`` from time ``t=0`` to ``t_end`` under the chain's
    own interaction plus an external force ``ext_force(x)``.

    ``v0`` is the initial velocity of every particle

    ``ext_force(x)`` is a force that is imposed on the chain from outside.

    ``boundary`` says what lies past the ends (see ``boundaries/``).

    ``polarities`` is the array of binary polarities s_k = +-1, one per particle.

    ``friction > 0`` drains the transient and the chain settles into a stationary
    state, ``friction = 0`` makes the chain evolve freely.

    ``step`` is the step policy: how long a step may be and how the search walks
    toward it (see ``StepPolicy``). The default adapts the step to the gaps.

    ``snapshot_every`` records the chain every N accepted steps. t=0 and the final
    state are always recorded.

    Raises ``ValueError`` unless ``x0`` is ordered along the line and inside the
    boundary, and ``RuntimeError`` once a step comes out non-finite or collapses under
    ``dt_min``.
    """
    s = np.asarray(polarities)

    def total_acceleration(x: np.ndarray) -> np.ndarray:
        return acceleration(x, s, boundary) + ext_force(x) / MASS

    return _evolve(
        x0,
        v0=v0,
        acceleration=total_acceleration,
        get_gaps=boundary.get_gaps,
        t_end=t_end,
        friction=friction,
        step=step,
        snapshot_every=snapshot_every,
    )


def _evolve(
    x0: np.ndarray,
    *,
    v0: np.ndarray | None,
    acceleration,
    get_gaps,
    t_end: float,
    friction: float,
    step: StepPolicy,
    snapshot_every: int,
) -> Trajectory:
    """The stepper itself: Stoermer-Verlet, driven by ``acceleration`` and ``get_gaps``.
    This function can integrate without enforcing the internal force and without a
    boundary, which is what lets the integration be checked against a system with a known solution.

    E.g.: _evolve([1.0], lambda x: -omega**2 * x, get_gaps=lambda x: np.empty(0), ...)
    -> ``cos(omega t)``.

    ``acceleration(x)`` gives the acceleration of every particle.

    ``get_gaps(x)`` are the distances between neighboring particles. They are used to
     determine the next time step.

    Raises ``ValueError`` unless every gap in ``x0`` is positive, and ``RuntimeError``
    if the step collapses under ``dt_min`` or ``t`` stops advancing.
    """
    x = np.asarray(x0, dtype=float).copy()
    v = np.zeros(x.size) if v0 is None else np.asarray(v0, dtype=float).copy()
    if v.shape != x.shape:
        raise ValueError(f"v0 has shape {v.shape}, x0 has {x.shape}")
    gaps0 = get_gaps(x)
    # Only t=0 needs this: every later x came back from ``find_step``, which returns
    # a trial only after its gaps have all tested positive.
    if gaps0.size and gaps0.min() <= 0.0:
        worst = int(np.argmin(gaps0))
        raise ValueError(
            f"x0 has a non-positive gap {gaps0[worst]:g} nm at index {worst}: companions "
            f"must be ordered along the line and inside the boundary"
        )
    a = acceleration(x)

    dt_max = t_end / 100 if step.dt_max is None else step.dt_max
    dt_min = t_end * 1e-12 if step.dt_min is None else step.dt_min
    # Under the resolution of t the search could shrink forever and t never reach t_end.
    if dt_min <= np.spacing(t_end):
        raise ValueError(
            f"dt_min = {dt_min:g} ps is under the resolution of t_end = {t_end:g} ps "
            f"({np.spacing(t_end):g} ps): the step search could shrink without ever ending"
        )
    stepControl = StepControl(get_gaps, step, dt_max=dt_max, dt_min=dt_min)
    dt = min(step.dt_start, dt_max)
    snaps = Snapshots(x, every=snapshot_every)
    t = 0.0
    while t < t_end:
        x_next, dt = stepControl.find_step(x, v, a, dt, t)
        if t + dt <= t:  # dt is under the resolution of t; the loop would never end
            raise RuntimeError(f"time stopped advancing at t = {t:g} ps with dt = {dt:g} ps")
        t += dt
        v = ((x_next - x) / dt) * np.exp(-friction * dt)
        x = x_next
        a = acceleration(x)
        snaps.record(t, x)
    t_snaps, x_snaps = snaps.finish(t, x)
    return Trajectory(x=x, v=v, t_snaps=t_snaps, x_snaps=x_snaps)
