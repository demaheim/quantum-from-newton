"""Build the figures of the manuscript from the saved experiment data.

Run the experiments first (or ``run_all.py``, which does both). The state and
double-slit panels pair the companion trajectories with the chain's link
density against the prediction of the Schroedinger equation; the transition
figures instead plot a family of relaxed chains against the placement of a
migrating domain wall, not time.
"""

from pathlib import Path

import matplotlib.pyplot as plt
from quantum_from_newton.plotting import GREEN, MUTED, plot_density, plot_trajectories
import numpy as np

from quantum_from_newton.results import load

OUT = Path(__file__).resolve().parent
DPI = 300  # print resolution; journals commonly ask for 600 on raster figures

FAMILIES = {
    # the box is numbered from n = 1; n = 0 would give psi == 0
    "fig_box_states.png": [
        ("box/box_n1", r"$|\psi_1|^2$"),
        ("box/box_n2", r"$|\psi_2|^2$"),
        ("box/box_n3", r"$|\psi_3|^2$"),
    ],
    "fig_hosc_states.png": [
        ("hosc/hosc_n0", r"$|\psi_0|^2$"),
        ("hosc/hosc_n1", r"$|\psi_1|^2$"),
        ("hosc/hosc_n2", r"$|\psi_2|^2$"),
    ],
}


def build_family(filename, rows):
    """Three stationary states, one per row."""
    data = [load(name) for name, _ in rows]
    x_limit = 1.08 * max(np.abs(d["x_link"]).max() for d in data)

    fig, axes = plt.subplots(3, 2, figsize=(10, 8), sharex="col")
    handles = {}
    for (ax_trajectories, ax_density), d, (_, label) in zip(axes, data, rows):
        plot_trajectories(ax_trajectories, d["t_snaps"], d["x_snaps"])
        ax_trajectories.set_ylabel("x (nm)")

        handles = plot_density(ax_density, d["x_link"], d["P"], d["x_ref"], d["P_ref"])
        ax_density.set_xlim(-x_limit, x_limit)
        ax_density.text(0.03, 0.9, label, transform=ax_density.transAxes)

    axes[-1][0].set_xlabel("t (ps)")
    axes[-1][1].set_xlabel("x (nm)")
    fig.legend(handles.values(), handles.keys(), loc="upper center", ncol=3, frameon=False)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(OUT / filename, dpi=DPI)
    plt.close(fig)
    print(f"saved {filename}")


def build_double_slit(filename="fig_dslit.png", limit=400.0):
    """One row: a time evolution rather than a relaxation."""
    d = load("dslit/dslit")
    fig, (ax_trajectories, ax_density) = plt.subplots(1, 2, figsize=(10, 4))

    keep = select_symmetric_indices(d["x_snaps"].shape[1], 5)
    plot_trajectories(ax_trajectories, d["t_snaps"], d["x_snaps"][:, keep])
    ax_trajectories.set_xlabel("t (ps)")
    ax_trajectories.set_ylabel("x (nm)")
    ax_trajectories.set_ylim(-limit, limit)

    handles = plot_density(
        ax_density, d["x_link"], d["P"], d["x_ref"], d["P_ref"], initial=d["P_ref_initial"]
    )
    ax_density.set_xlim(-limit, limit)
    ax_density.set_xlabel("x (nm)")
    ax_density.legend(handles.values(), handles.keys(), fontsize=8, loc="upper right")

    fig.tight_layout()
    fig.savefig(OUT / filename, dpi=DPI)
    plt.close(fig)
    print(f"saved {filename}")


def select_symmetric_indices(n_particles, step):
    """Every ``step``-th trajectory, kept symmetric about the centre of the chain.

    The first kept companion sits half a step off the centre, so that mirroring
    leaves the same gap across the middle as everywhere else; starting at the
    centre itself would keep two neighbours and draw them as a tight pair.
    """
    upper = np.arange((n_particles + step) // 2, n_particles, step)
    return np.unique(np.concatenate(((n_particles - 1) - upper, upper)))


def draw_relaxed_positions(ax, num_in_new_domain, x_relaxed, *, dashed_levels=()):
    """Left panel of a transition figure: one relaxed chain per wall placement, not a time axis.

    Longer chains are thinned to about sixty lines; drawing every companion of a
    few hundred would fill the panel solid, and the stretched link at a wall
    stays just as visible either way.
    """
    n_companions = x_relaxed.shape[1]
    keep = select_symmetric_indices(n_companions, max(1, n_companions // 60))
    plot_trajectories(ax, num_in_new_domain, x_relaxed[:, keep])
    for level in (-50.0, 50.0):
        ax.axhline(level, color=MUTED, lw=0.8)
    for level in dashed_levels:
        ax.axhline(level, color="0.85", lw=0.8, ls="--")
    ax.set_xlabel("companions in the new domain")
    ax.set_ylabel("x (nm)")
    ax.set_ylim(-54, 54)


def draw_energy_ratio(ax, num_in_new_domain, energy_ratio, exact_ratio, num_ends):
    """Right panel of a transition figure: the measured climb, its plateau, and the exact ratio.

    Energies are averaged over companions and taken over the Schroedinger ground
    energy, so the gap between the two dashed lines is exactly what the domain ends
    store: ``num_ends`` of them, two per bulk wall and one per hard wall, all
    carrying the same energy while the domains hold equal shares.
    """
    ax.plot(num_in_new_domain, energy_ratio, "o", ms=3, c=GREEN)
    plateau = energy_ratio[-1]
    ax.axhline(plateau, color=MUTED, lw=0.8, ls="--")
    ax.axhline(exact_ratio, color=MUTED, lw=0.8, ls="--")

    # Label the gap with what fills it. Bar ends rather than arrowheads: the gap
    # is a few percent of the panel for the first transition, and arrowheads that
    # size merge into a blob.
    x_marker = 0.30
    ax.annotate(
        "",
        xy=(x_marker, exact_ratio),
        xytext=(x_marker, plateau),
        xycoords=ax.get_yaxis_transform(),
        textcoords=ax.get_yaxis_transform(),
        arrowprops={
            "arrowstyle": "|-|,widthA=0.25,widthB=0.25",
            "color": "0.45",
            "lw": 0.8,
            "shrinkA": 0,
            "shrinkB": 0,
        },
    )
    ax.text(
        x_marker + 0.025,
        (plateau + exact_ratio) / 2,
        rf"${num_ends}\,E_\mathrm{{end}}$",
        va="center",
        fontsize=8,
        color="0.45",
        transform=ax.get_yaxis_transform(),
    )

    bottom = ax.get_ylim()[0]
    ax.set_ylim(bottom, exact_ratio + 0.04 * (exact_ratio - bottom))
    ax.set_xlabel("companions in the new domain")
    ax.set_ylabel(r"$\langle E \rangle / E_1^{\mathrm{Q}}$")


def build_transition_12(filename="fig_box_transition_12.png"):
    """Ground to first excited state: the wall carried in from the boundary."""
    d = load("box/box_transition_12")
    fig, (ax_positions, ax_energy) = plt.subplots(1, 2, figsize=(10, 4))

    draw_relaxed_positions(ax_positions, d["num_in_new_domain"], d["x_relaxed"])
    draw_energy_ratio(ax_energy, d["num_in_new_domain"], d["energy_ratio"], 4.0, 4)

    fig.tight_layout()
    fig.savefig(OUT / filename, dpi=DPI)
    plt.close(fig)
    print(f"saved {filename}")


def build_transition_23(filename="fig_box_transition_23.png"):
    """Second to third state: a new wall enters while the centre wall migrates."""
    d = load("box/box_transition_23")
    fig, (ax_positions, ax_energy) = plt.subplots(1, 2, figsize=(10, 4))

    draw_relaxed_positions(
        ax_positions, d["num_in_new_domain"], d["x_relaxed"], dashed_levels=(-50.0 / 3, 50.0 / 3)
    )
    draw_energy_ratio(ax_energy, d["num_in_new_domain"], d["energy_ratio"], 9.0, 6)

    fig.tight_layout()
    fig.savefig(OUT / filename, dpi=DPI)
    plt.close(fig)
    print(f"saved {filename}")


def main():
    for filename, rows in FAMILIES.items():
        build_family(filename, rows)
    build_double_slit()
    build_transition_12()
    build_transition_23()


if __name__ == "__main__":
    main()
