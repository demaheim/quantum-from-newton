"""Physical constants in the units used throughout: nanometres, picoseconds, kilograms.

Lengths are in nm, times in ps, masses in kg. Energies therefore come out in
kg nm^2 / ps^2, and accelerations in nm / ps^2.
"""

import numpy as np

NM = 1e-9  # m per nm
PS = 1e-12  # s per ps

_PLANCK = 6.62606957e-34  # J s

#: Reduced Planck constant, scaled to nm and ps.
HBAR = _PLANCK / (2 * np.pi) / NM**2 * PS

#: Electron mass (kg). The mass appearing in the companions' equation of motion.
MASS = 9.10938291e-31

#: Coupling of the interaction, kappa = hbar^2 / 8m.
KAPPA = HBAR**2 / (8 * MASS)
