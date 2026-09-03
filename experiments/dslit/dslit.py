"""Double slit: two packets released at rest, evolving freely."""

import numpy as np

from quantum_from_newton.chain import (
    StepPolicy,
    Vacuum,
    get_link_density,
    integrate,
    list_polarities,
)
from quantum_from_newton.results import save
from quantum_from_newton.systems import GaussianPair

num_companions = 300
t_end = 20.0  # ps
edge_quantile = 1e-4  # the outermost companions sit at this quantile

packets = GaussianPair(separation=50.0, width=10.0)  # nm
boundary = Vacuum()

print(f"evolving {num_companions} companions for {t_end:g} ps ...", flush=True)
start = packets.get_init_pos(num_companions, edge_quantile, 1.0 - edge_quantile)
chain = integrate(
    x0=start,
    ext_force=packets.force,
    boundary=boundary,
    polarities=list_polarities(num_companions),  # single-valued: no domain wall anywhere
    friction=0.0,  # free evolution: energy is conserved
    step=StepPolicy(adaptive_dt_band=(0.99, 0.993)),
    t_end=t_end,
    snapshot_every=200,
)

x_link, P = get_link_density(chain.x)
x_ref = np.linspace(x_link.min(), x_link.max(), 600)
initial, final = packets.density(0.0, x_ref), packets.density(t_end, x_ref)
print(
    "wrote",
    save(
        "dslit/dslit",
        t_snaps=chain.t_snaps,
        x_snaps=chain.x_snaps,
        x_link=x_link,
        P=P / P.max(),
        x_ref=x_ref,
        P_ref=final / final.max(),
        P_ref_initial=initial / initial.max(),
    ),
)
