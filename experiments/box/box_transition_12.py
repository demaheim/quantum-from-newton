"""Sweep the domain wall over the links: the quasi-static path from box state n = 1 to n = 2."""

import numpy as np

from quantum_from_newton.chain import StepPolicy, energy, integrate, list_polarities
from quantum_from_newton.results import save_sweep
from quantum_from_newton.systems import Box

half_box_width = 50.0  # nm
num_companions = 60
wall_stride = 1  # links between placements: every link, the chain being small enough
box = Box(half_box_width)

# One damped relaxation per placement of the wall. A flip at link j leaves the j
# companions of the new domain left of the wall; j = 0 turns over the whole chain,
# which is the wall-free ground state again -- a global flip changes nothing.
num_in_new_domain = np.arange(0, num_companions // 2 + 1, wall_stride)
x_relaxed = np.empty((len(num_in_new_domain), num_companions))
energies = np.empty(len(num_in_new_domain))

print(f"relaxing {num_companions} companions with the wall held at each link ...", flush=True)
for i, j in enumerate(num_in_new_domain):
    pol = list_polarities(num_companions, flips=[j])
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
    x_relaxed[i] = chain.x
    energies[i] = energy(chain.x, pol, box.boundary)
    ratio = energies[i] / num_companions / box.eigenstate_energy(1)
    print(f"  {j:3d} companions in the new domain: <E>/E_1^Q = {ratio:.4f}", flush=True)

# The average over each companion possibly being the massive particle, against the
# Schroedinger ground energy, so that the chain's own shortfall -- the energy
# stored at its two ends -- stays visible.
energy_ratio = energies / num_companions / box.eigenstate_energy(1)
print(f"monotone rising: {bool(np.all(np.diff(energy_ratio) > 0))}")
print(f"foot of the climb: {energy_ratio[0]:.4f}  (ground state, against 1)")
print(f"top of the climb: {energy_ratio[-1]:.4f}  (state n = 2, against 4)")
print(
    "wrote",
    save_sweep(
        "box/box_transition_12",
        num_in_new_domain=num_in_new_domain,
        x_relaxed=x_relaxed,
        energy_ratio=energy_ratio,
    ),
)
