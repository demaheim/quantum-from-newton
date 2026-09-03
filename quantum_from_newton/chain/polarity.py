"""Binary polarities of the companion particles.

Each companion k carries a polarity s_k in {+1, -1}. A stretch of companions
sharing a polarity is a domain; a link joining opposite polarities is a domain
wall.
"""

import numpy as np


def list_polarities(n: int, flips=()) -> np.ndarray:
    """Polarities of ``n`` companions, reversing after each index in ``flips``.

    ``flips`` lists the particle indices at which the polarity turns over, so a
    domain wall lands in the gap between particle ``f - 1`` and particle ``f``.
    ``list_polarities(60)`` is a single domain; ``list_polarities(60, [30])``
    puts one wall in the central gap; ``list_polarities(60, [20, 40])`` splits
    the chain into three equal domains.
    """
    s = np.ones(n)
    for f in flips:
        s[f:] *= -1.0
    return s
