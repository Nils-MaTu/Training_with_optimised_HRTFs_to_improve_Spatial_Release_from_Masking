# Training with optimised HRTFs to improve Spatial Release from Masking

Standalone reproduction package for the statistical results, figures, and optimised KEMAR HRTF reported in the manuscript.

## Run

Python 3.11 is recommended.

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install -r requirements.txt

# 1. Generate the optimised KEMAR HRTF
python3 optimisation/optimise_hrtf.py

# 2. Reproduce the manuscript statistics
python3 analysis.py

# 3. Reproduce the five manuscript figures
python3 plots.py
```

Outputs are written below `outputs/`. The HRTF optimisation is the original full-horizontal-plane PCA/gradient-descent procedure used for the experiment; only its machine-specific paths were made portable. It should increase mean modelled front--back SRM from approximately 1.36 to 7.91 dB. Small floating-point differences can occur across CPU, CUDA, and Apple Metal backends.

## Contents

- `optimisation/optimise_hrtf.py` and `optimisation/lavandier2022.py`: HRTF optimisation and its differentiable SRM model.
- `optimisation/KEMAR_*.sofa`: source KEMAR HRTF distributed under the CC BY-SA 3.0 licence recorded in its SOFA metadata.
- `analysis.py`: repeated-measures ANOVAs, planned contrasts, participant-clustered binomial GEE, and baseline/improvement correlations.
- `plots.py`: the five manuscript figures.
- `data/participants/sub-*/`: one anonymised CSV per participant and dataset (front--back trials, SRM trials, and better-ear SNR gains).
- `data/hrtf_spectra_summary.npz`: compact group-level spectral summary required for Figure 1; no participant identifiers or HRTF impulse responses.
- `highlights.tex`: journal highlights.

Participant IDs are pseudonyms (`sub-01`--`sub-18`) with no public lookup table. Timestamps, response times, local paths, and unused response/stimulus fields are excluded. All scripts resolve paths relative to this repository and can be run from any working directory.
