"""Particle in a box, first excited state (n = 2)."""

import numpy as np

from quantum_from_newton.chain import StepPolicy, get_link_density, integrate, list_polarities
from quantum_from_newton.results import save
from quantum_from_newton.systems import Box

half_box_width = 50.0  # nm
num_companions = 60
box = Box(half_box_width)
pol = list_polarities(num_companions, flips=[30])  # one wall, between the two central companions

print(f"relaxing {num_companions} companions to box state n = 2 ...", flush=True)
chain = integrate(
    x0=box.get_uniform_start(num_companions),
    ext_force=box.force,
    boundary=box.boundary,
    polarities=pol,
    friction=0.4,
    step=StepPolicy(adaptive_dt_band=(0.995, 0.999)),
    t_end=130.0,
)

x_link, P = get_link_density(chain.x)
x_ref = np.linspace(-half_box_width, half_box_width, 400)
print(
    "wrote",
    save(
        "box/box_n2",
        t_snaps=chain.t_snaps,
        x_snaps=chain.x_snaps,
        x_link=x_link,
        P=P / P.max(),
        x_ref=x_ref,
        P_ref=box.eigenstate_density(2, x_ref),
    ),
)
