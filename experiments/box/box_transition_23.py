"""A second wall enters from the boundary: the quasi-static path from box state n = 2 to n = 3."""

import numpy as np

from quantum_from_newton.chain import StepPolicy, energy, integrate, list_polarities
from quantum_from_newton.results import save_sweep
from quantum_from_newton.systems import Box

half_box_width = 50.0  # nm
num_companions = 60
wall_stride = 2  # companions between placements; see the note on even m below
box = Box(half_box_width)


def relax(pol):
    """The standard damped relaxation from a uniform start, as in box_n1/2/3."""
    chain = integrate(
        x0=box.get_uniform_start(num_companions),
        ext_force=box.force,
        boundary=box.boundary,
        polarities=pol,
        friction=0.4,
        step=StepPolicy(adaptive_dt_band=(0.995, 0.999)),
        t_end=130.0,
        snapshot_every=10**9,
    )
    return chain.x


def flips_on_path(m):
    """Domains of (m, c, N - m - c) companions with c = (N - m) // 2.

    The new domain of m companions grows at the boundary while the two others
    stay equal, so the existing centre wall migrates at half the pace; m = 0 is
    state n = 2 (a flip at 0 is a global turnover and changes nothing).
    """
    return [m, m + (num_companions - m) // 2]


# Even m only: it keeps the two trailing domains at equal companion counts, the
# configurations actually on (0, 1/2, 1/2) -> (1/3, 1/3, 1/3). Odd m costs
# nothing measurable but shuffles the relaxed positions into a staircase.
num_in_new_domain = np.arange(0, num_companions // 3 + 1, wall_stride)
x_relaxed = np.empty((len(num_in_new_domain), num_companions))
energies = np.empty(len(num_in_new_domain))

print(f"relaxing {num_companions} companions along the two-wall path ...", flush=True)
for i, m in enumerate(num_in_new_domain):
    pol = list_polarities(num_companions, flips=flips_on_path(m))
    x_relaxed[i] = relax(pol)
    energies[i] = energy(x_relaxed[i], pol, box.boundary)
    ratio = energies[i] / num_companions / box.eigenstate_energy(1)
    print(f"  {m:3d} companions in the new domain: <E>/E_1^Q = {ratio:.4f}", flush=True)

# The average over each companion possibly being the massive particle, against the
# Schroedinger ground energy, so that the chain's own shortfall -- the energy
# stored at its ends -- stays visible.
energy_ratio = energies / num_companions / box.eigenstate_energy(1)
print(f"monotone rising: {bool(np.all(np.diff(energy_ratio) > 0))}")
print(f"start of the climb: {energy_ratio[0]:.4f}  (state n = 2, against 4)")
print(f"end of the climb: {energy_ratio[-1]:.4f}  (state n = 3, against 9)")
print(
    "wrote",
    save_sweep(
        "box/box_transition_23",
        num_in_new_domain=num_in_new_domain,
        x_relaxed=x_relaxed,
        energy_ratio=energy_ratio,
    ),
)
