"""Bookkeeping around the Verlet loop: how long a step may be, and what is recorded."""

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class StepPolicy:
    """How long a step may be, and how the search walks toward it.

    ``adaptive_dt_band`` is the acceptance window on the fractional shrinkage of
    the fastest-closing gap over one step. The default (0.98, 0.99) means if the
    tightest gap loses more than 2% of its length per step the step is too long,
    less than 1% and it's too short.

    ``dt_shrink`` and ``dt_grow`` are what a step outside that band is multiplied
    by: pulled in by ``dt_shrink``, let out by ``dt_grow``, until it lands inside.
    Closer to 1 searches the step more finely, at the cost of more trials per step.

    ``dt_start`` is only the first trial step. A too-large first step could fling
    a companion past its neighbour.

    ``dt_max`` is the ceiling on the step, ``t_end / 100`` unless given.

    ``dt_min`` is the floor, ``t_end * 1e-12`` unless given.
    """

    adaptive_dt_band: tuple[float, float] = (0.98, 0.99)
    dt_shrink: float = 0.95
    dt_grow: float = 1.05
    dt_start: float = 1e-6
    dt_max: float | None = None
    dt_min: float | None = None


# Have to use this to avoid a linter error.
DEFAULT_STEP = StepPolicy()


def _smallest_gap(gaps: np.ndarray) -> float:
    """The tightest gap, or ``inf`` when there are none at all (a lone particle)."""
    return float(gaps.min()) if gaps.size else np.inf


@dataclass(frozen=True)
class StepControl:
    """The policy applied to one run: where the gaps are, and what the ceiling is.

    Built once per run, because none of it changes from step to step.

    ``get_gaps(x)`` are the distances between neighboring particles. They are what
    the step length is measured against.

    ``dt_max`` is the policy's ceiling resolved against the run: a bare number, 
    ``np.inf`` for no ceiling at all, since ``StepPolicy`` alone cannot know

    ``t_end``. ``dt_min`` is the matching floor, ``0.0`` for no floor at all.
    """

    get_gaps: Callable[[np.ndarray], np.ndarray]
    stepPolicy: StepPolicy
    dt_max: float = np.inf
    dt_min: float = 0.0

    def find_step(self, x: np.ndarray, v: np.ndarray, a: np.ndarray, dt: float, t: float):
        """Search for a step the gaps will tolerate.

        Returns ``(x_next, dt)``: the trial position that was accepted and the step
        that produced it.

        ``t`` is carried in for the error messages alone. Raises ``RuntimeError`` once
        a trial comes out non-finite, or once the step has collapsed under ``dt_min``.
        """
        gaps_before = self.get_gaps(x)
        while True:
            x_next = x + v * dt + a * dt * dt
            if not np.isfinite(x_next).all():
                # a too small gap can make the force (~1/gap**3)  overflow
                smallest = _smallest_gap(gaps_before)
                raise RuntimeError(
                    f"integration failed after t = {t:g} ps, smallest gap {smallest:g} nm"
                )
            gaps_after = self.get_gaps(x_next)
            if np.any(gaps_after <= 0.0):  # a gap closed completely; back off and retry
                dt = self._shrink_dt(dt, gaps_before, t)
                continue
            dt, acceptable = self._adjust_dt(gaps_before, gaps_after, dt, t)
            if acceptable:
                return x_next, dt

    def _shrink_dt(self, dt: float, gaps_before: np.ndarray, t: float) -> float:
        """Pull the step in, unless that takes it under ``dt_min``."""
        dt *= self.stepPolicy.dt_shrink
        if dt < self.dt_min:
            raise RuntimeError(
                f"step collapsed below dt_min = {self.dt_min:g} ps after t = {t:g} ps, "
                f"smallest gap {_smallest_gap(gaps_before):g} nm"
            )
        return dt

    def _adjust_dt(self, gaps_before: np.ndarray, gaps_after: np.ndarray, dt: float, t: float):
        """Adjust ``dt`` until the fastest closing gap shrinks into ``adaptive_dt_band``.

        Returns the step to use next and whether the trial step was acceptable.
        """
        lower_limit, upper_limit = self.stepPolicy.adaptive_dt_band
        # No gaps at all (a lone particle): ``ratio = inf``
        gaps_ratio = float(np.min(gaps_after / gaps_before)) if gaps_before.size else np.inf
        if gaps_ratio < lower_limit:
            return self._shrink_dt(dt, gaps_before, t), False
        if gaps_ratio > upper_limit and dt < self.dt_max:
            return min(dt * self.stepPolicy.dt_grow, self.dt_max), False
        return dt, True


class Snapshots:
    """The recorded history: ``every``-th step, plus both ends of the run."""

    def __init__(self, x0: np.ndarray, *, every: int):
        self.every, self.taken = every, 0
        self.t, self.x = [0.0], [x0.copy()]

    def record(self, t: float, x: np.ndarray) -> None:
        self.taken += 1
        if self.taken % self.every == 0:
            self.t.append(t)
            self.x.append(x.copy())

    def finish(self, t: float, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Close the history off and hand back ``(t_snaps, x_snaps)`` for the caller
        to package. The caller owns the result type; this only owns the recording.
        """
        if self.t[-1] != t:  # always end on the state the caller will measure
            self.t.append(t)
            self.x.append(x.copy())
        return np.asarray(self.t), np.asarray(self.x)
