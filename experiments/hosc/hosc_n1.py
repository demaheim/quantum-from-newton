"""Harmonic oscillator, first excited state (n = 1)."""

import numpy as np

from quantum_from_newton.chain import StepPolicy, get_link_density, integrate, list_polarities
from quantum_from_newton.results import save
from quantum_from_newton.systems import Harmonic

num_companions = 60
start_extent = 80.0  # nm, half-width of the uniform start
oscillator = Harmonic(omega=2 * np.pi / 60)  # 1/ps
pol = list_polarities(num_companions, flips=[30])  # one wall, between the two central companions

print(f"relaxing {num_companions} companions to oscillator state n = 1 ...", flush=True)
chain = integrate(
    x0=oscillator.get_uniform_start(num_companions, start_extent),
    ext_force=oscillator.force,
    boundary=oscillator.boundary,
    polarities=pol,
    friction=0.4,
    step=StepPolicy(adaptive_dt_band=(0.995, 0.999)),
    t_end=130.0,
)

x_link, P = get_link_density(chain.x)
x_ref = np.linspace(-start_extent, start_extent, 800)
print(
    "wrote",
    save(
        "hosc/hosc_n1",
        t_snaps=chain.t_snaps,
        x_snaps=chain.x_snaps,
        x_link=x_link,
        P=P / P.max(),
        x_ref=x_ref,
        P_ref=oscillator.eigenstate_density(1, x_ref),
    ),
)
