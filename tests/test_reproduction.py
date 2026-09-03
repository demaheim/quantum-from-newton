"""End-to-end: do the chains actually land on the states the manuscript shows?"""

import numpy as np
import pytest

from quantum_from_newton.chain import (
    StepPolicy,
    Vacuum,
    get_link_density,
    integrate,
    list_polarities,
)
from quantum_from_newton.systems import Box, GaussianPair, Harmonic

NUM_COMPANIONS = 60
HALF_BOX_WIDTH = 50.0  # nm
OMEGA = 2 * np.pi / 60  # 1/ps


# Tolerances are the measured deviations with a little headroom, not round
# numbers: a regression that pushed any of these up by half would fail here.
@pytest.mark.parametrize(
    "state,flips,tolerance", [(1, [], 0.04), (2, [30], 0.05), (3, [20, 40], 0.06)]
)
def test_box_relaxes_onto_its_eigenstate(state, flips, tolerance):
    box = Box(HALF_BOX_WIDTH)
    polarities = list_polarities(NUM_COMPANIONS, flips)
    x_link, P = relax(box, box.get_uniform_start(NUM_COMPANIONS), polarities)
    P_ref = box.eigenstate_density(state, x_link)
    assert measure_deviation_away_from_nodes(P, P_ref, polarities) < tolerance


@pytest.mark.parametrize(
    "state,flips,tolerance", [(0, [], 0.02), (1, [30], 0.06), (2, [24, 36], 0.11)]
)
def test_oscillator_relaxes_onto_its_eigenstate(state, flips, tolerance):
    oscillator = Harmonic(OMEGA)
    start_extent = 80.0  # nm, half-width of the uniform start
    polarities = list_polarities(NUM_COMPANIONS, flips)
    x_link, P = relax(
        oscillator, oscillator.get_uniform_start(NUM_COMPANIONS, start_extent), polarities
    )
    P_ref = oscillator.eigenstate_density(state, x_link)
    assert measure_deviation_away_from_nodes(P, P_ref, polarities) < tolerance


def test_the_double_slit_companions_end_up_distributed_as_the_fringes():
    """The double-slit figure, compared as a distribution rather than point by point."""
    packets = GaussianPair(separation=50.0, width=10.0)  # nm
    num_companions, t_end = 300, 20.0  # ps
    edge_quantile = 1e-4  # the outermost companions sit at this quantile

    start = packets.get_init_pos(num_companions, edge_quantile, 1.0 - edge_quantile)
    chain = integrate(
        x0=start,
        ext_force=packets.force,
        boundary=Vacuum(),  # the whole chain, as experiments/dslit/dslit.py steps it
        polarities=list_polarities(num_companions),
        friction=0.0,  # free evolution
        step=StepPolicy(adaptive_dt_band=(0.99, 0.993)),  # as experiments/dslit/dslit.py runs it
        t_end=t_end,
        snapshot_every=10**9,
    )

    # Exact, not approximate: nothing in the stepper reduces across the chain, so the
    # symmetry of the start survives every one of the ~23000 steps intact.
    assert np.array_equal(chain.x, -chain.x[::-1])

    x = chain.x
    share_of_the_medium = (np.arange(x.size) + 0.5) / x.size
    assert np.abs(share_of_the_medium - exact_cdf(packets, t_end, x)).max() < 0.07
    assert np.abs(share_of_the_medium - exact_cdf(packets, 0.0, x)).max() > 0.15


def relax(system, start, polarities):
    """``(x_link, P)`` of the chain ``system`` relaxes into, P over its maximum."""
    chain = integrate(
        x0=start,
        ext_force=system.force,
        boundary=system.boundary,
        polarities=polarities,
        friction=0.4,
        step=StepPolicy(adaptive_dt_band=(0.995, 0.999)),  # as the experiments run it
        t_end=130.0,
        snapshot_every=10**9,
    )
    x_link, P = get_link_density(chain.x)
    return x_link, P / P.max()


def measure_deviation_away_from_nodes(P, P_ref, polarities):
    """The worst gap between chain and Schroedinger, off the links carrying a wall."""
    return np.abs(P - P_ref)[~find_domain_walls(polarities)].max()


def find_domain_walls(polarities):
    """Boolean mask over links (``len(polarities) - 1``), true where the polarity turns over."""
    return polarities[:-1] * polarities[1:] < 0


def exact_cdf(packets, t, x):
    """Where |psi(t)|^2 says the fraction of the medium left of ``x`` should be."""
    x_ref = np.linspace(-600.0, 600.0, 200001)
    P_ref = packets.density(t, x_ref)
    cumulative = np.concatenate(([0.0], np.cumsum((P_ref[1:] + P_ref[:-1]) / 2 * np.diff(x_ref))))
    return np.interp(x, x_ref, cumulative / cumulative[-1])
