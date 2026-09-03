# quantum-from-newton

Simulation code for the manuscript ***From Newton's equations to the Schrödinger equation: domain walls as the quantization condition***.
It reproduces every figure in that paper and is meant to be built on.

A line of binary-polarized *companion particles* moves under Newton's equations. They interact only with their
nearest neighbours, through a force built from the inverse gaps between them. ***Nothing wave-like is
put in***: no wave function, no probability, no quantum postulate. What comes out are the stationary
densities of the ***particle in a box and the harmonic oscillator, and the fringe pattern of the
double slit***, matching the predictions of the Schrödinger equation.

## Reproducing the figures

Requires [uv](https://docs.astral.sh/uv/).

```bash
uv sync
uv run python run_all.py
```

Or one at a time:

```bash
uv run python experiments/box/box_n2.py             # writes data/box/box_n2.npz
uv run python figures/make_manuscript_figures.py    # writes figures/*.png
```

| Manuscript figure | Experiments | System |
|---|---|---|
| `fig_box_states.png` | `box/box_n1.py`, `box/box_n2.py`, `box/box_n3.py` | particle in a box, `b = 50 nm`, `N = 60` |
| `fig_hosc_states.png` | `hosc/hosc_n0.py`, `hosc/hosc_n1.py`, `hosc/hosc_n2.py` | harmonic oscillator, `ω = 2π/60 ps⁻¹`, `N = 60` |
| `fig_dslit.png` | `dslit/dslit.py` | double slit, `N = 300`, free evolution to 20 ps |
| `fig_box_transition_12.png` | `box/box_transition_12.py` | particle in a box, the wall swept over the links, `N = 60` |
| `fig_box_transition_23.png` | `box/box_transition_23.py` | particle in a box, two migrating walls, `N = 60` |

The figures are committed, so the manuscript builds from a fresh clone without running
anything. The two transition sweeps dominate the runtime of `run_all.py`, a few minutes each:
every point of them is a full relaxation of its own.

## Layout

```
quantum_from_newton/
├── chain/            the microscopic model — the same for every system
│   ├── constants.py      nm/ps units; kappa = ħ²/8m
│   ├── polarity.py       s_k = ±1, and where the domain walls sit
│   ├── boundaries/       what lies beyond the ends
│   │   ├── boundary.py       the Boundary protocol every rule satisfies
│   │   ├── vacuum.py         nothing out there: ghosts at ±infinity
│   │   └── hardwalls.py      reflection in hard walls at ±b
│   ├── dynamics/         what makes the chain move
│   │   ├── interaction.py    V_int and its exact analytic gradient
│   │   └── stepping.py       the compression-limited step, and the recorded history
│   └── integrate.py      Störmer-Verlet, friction, the Trajectory
├── systems/          one module per physical system
│   ├── box.py            hard walls           + Dirichlet eigenstates
│   ├── harmonic.py       V, F, σ              + Hermite eigenstates
│   └── gaussian_pair.py  initial condition    + free |ψ(x,t)|²
├── results.py        the data/*.npz contract between experiments and figures
└── plotting.py       shared colours and panels
```

Stationary states and time evolution differ only in the friction passed to `integrate`:
`friction > 0` drains the transient so the chain settles into a minimum, `friction = 0` conserves
energy.

## Building on it

**A new system.** Write one module in `systems/` exposing `boundary` (how the chain is closed
off), `force(x)`, and whatever reference solution you want to compare against. Copy
`systems/harmonic.py` — a Morse or quartic potential is a few lines' change, and nothing in
`chain/` needs to know about it.

**A new experiment.** Copy any script in `experiments/`. The whole of one is:

```python
system = Box(b=50.0)
chain = integrate(
    x0=system.get_uniform_start(60),
    ext_force=system.force,
    boundary=system.boundary,
    polarities=list_polarities(60, flips=[30]),
    friction=0.4,
    step=StepPolicy(adaptive_dt_band=(0.995, 0.999)),
    t_end=130.0,
)
```

## Tests

```bash
uv run pytest
```

## Licence and citation

MIT — see [LICENSE](LICENSE). If you use this code, please cite the manuscript as well as the
software; see [CITATION.cff](CITATION.cff).
