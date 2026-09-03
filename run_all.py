"""Run every experiment, then build the figures of the manuscript.

Run it as ``uv run python run_all.py``, which reproduces the committed figures.

To redo a single panel, run its script in
``experiments/`` and then ``figures/make_manuscript_figures.py``.
"""

import runpy
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
EXPERIMENTS = [
    "box/box_n1",
    "box/box_n2",
    "box/box_n3",
    "box/box_transition_12",
    "box/box_transition_23",
    "hosc/hosc_n0",
    "hosc/hosc_n1",
    "hosc/hosc_n2",
    "dslit/dslit",
]

for name in EXPERIMENTS:
    started = time.time()
    runpy.run_path(str(ROOT / "experiments" / f"{name}.py"), run_name="__main__")
    print(f"  ({time.time() - started:.0f} s)\n", flush=True)

sys.path.insert(0, str(ROOT / "figures"))
runpy.run_path(str(ROOT / "figures" / "make_manuscript_figures.py"), run_name="__main__")
