#!/usr/bin/env python3
"""Reproduce original results and revised analyses/figures; no raw audio required."""
from pathlib import Path
import subprocess, sys
root = Path(__file__).resolve().parent
steps = [
    [root / "original/analysis.py", "--output-dir", root / "original_results"],
    [root / "original/plots.py", "--output-dir", root / "original_figures"],
    [root / "revision_analysis.py"],
    [root / "order_sensitivity.py"],
    [root / "revision_figures.py"],
    [root / "baseline_ancova.py"],
    [root / "baseline_ancova_figure.py"],
]
for args in steps:
    print("Running " + str(args[0].relative_to(root)), flush=True)
    subprocess.run([sys.executable, *map(str, args)], cwd=root, check=True)
print("Complete: original_results/, results/ and figures/")
