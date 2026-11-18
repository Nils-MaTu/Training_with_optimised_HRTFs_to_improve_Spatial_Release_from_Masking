# Reproduction record

## Run

Validated with Python 3.11. Install `requirements-analysis.txt` in an isolated environment, then run:

```sh
python3 final_reproducibility/reproduce.py
```

Run this from the repository root (or pass the script’s absolute path from any directory). It regenerates the archived original outputs, the revision analyses, the reported post-on-pre ANCOVA, and the final figures. It does not regenerate speech, BRIRs, the complete acoustic-prediction corpus or the optimised HRTF. The original `requirements.txt` additionally lists optimiser dependencies; they are unnecessary for the analysis command above.

## November Figure 1 provenance

Current Figure 1 plots the tested filter and all 18 directly measured SONICOM responses. The plot normalisation uses the binaural energy up to 8 kHz at each direction and applies one gain to both ears. The ERB ratios use these same normalised spectra. This record also documents the AES export for comparison and provenance.

- The AES PDF’s embedded Figure 1 matches the RGB pixels of `avarig_submission/hrtf_magnitude_original_vs_augmented.png` exactly. The archived plotting scripts read `avarig_submission/optimized_hrtf_new.sofa`. The sorted 36-direction impulse responses in this file and both local `optimized_hrtf_LOG.sofa` copies are sample-identical at 48 kHz. The accompanying NPZ records log-magnitude PCA and 1,000 iterations. Both the SOFA and NPZ are bundled in `inputs/`.
- The published Methods and archived AES implementation agree on log-magnitude PCA, the half-plane grid, the optimised coefficients and the Adam schedule. The archived code weights the pairwise ILD penalty by 0.1, penalises the squared ILD difference directly and adds a combined-ear RMS penalty. The printed equations specify baseline-referenced pair differences and omit the RMS penalty. The figure reads the archived export directly, so it does not depend on rerunning either formulation.
- The behavioural code uses the `aug` archive and the front/back test logs link to its digit files. The separate `aug_log` archive matches the AES filter. At 30° and 150°, the aligned per-ear direct-arrival segments of `aug` matched the earlier filter with correlations of 1.00000 (rounded to five places). Correlations against the AES candidate were 0.69976–0.87346. The `aug_log` BRIRs gave the reverse match, with correlations of 1.00000 for the AES candidate. Correlations are absolute, after maximising waveform cross-correlation over lag and scaling the aligned 256-sample segment. The SRM test logs store trial IDs rather than the root path. The experiment and original acoustic-prediction code both resolve these IDs against the `aug` archive.
- All 18 measured SONICOM free-field-compensated SOFAs contain both exact azimuth/elevation coordinates. The extracted samples match the previous revision’s measured arrays exactly. `figure1.py` takes the FFT directly, plots all 18 measured responses, applies display-only smoothing (1/48 octave for the measured traces and 1/24 octave for the optimised filter), and integrates ERB powers from the unsmoothed spectra.
- Relative to the AES NPZ baseline, the saved AES SOFA’s maximum absolute per-ear phase error at the original FFT bins from 200 to 8000 Hz is 2.87×10⁻⁶ rad across the plotted directions. A 200–1500 Hz cross-correlation audit on the AES data gave peak-lag shifts of +13.02 and −23.44 µs at 30° and 150°. It used the same search and interpolation as the earlier audit. The existing `results/timing_audit.csv` describes the earlier tested variant.

## What was verified

| Check | Result |
|---|---|
| Original statistics | All 14 original CSV tables match their archived counterparts (maximum absolute numerical discrepancy 1.78×10⁻¹⁴), including ANOVAs, contrasts, GEE, configuration ordering and baseline/change correlations. Full values: `original_results/`; comparison: `results/original_archive_comparison.csv`. |
| H1, independently reconstructed | Optimised minus individual pre-training SRM: 6.842105 pp; t(17)=2.428397; p=.026554. |
| H3 sensitivity | Original p=.04577; binomial marginal contrast p=.051502; symmetric-null exact sign-flip p=.048767. |
| H4 | T30M150 +1.3 pp, Holm p=.7406; T150M30 +9.1 pp, Holm p=.007053. The latter is primarily a baseline effect. |
| Original acoustic Figure 1 summary | Reconstructed directly from saved filters; maximum discrepancy 2.84×10⁻¹⁴ dB before the documented individual-file correction. |
| Archived-audio predictions | Twelve conditions (two mixtures × three configurations × two filters for one listener), rerun with MATLAB R2025b and AMT 1.6.0: maximum absolute discrepancy 2.31×10⁻¹⁴ dB. `results/prediction_validation.csv`. This is a sample validation, not a rerun of all 2,160 predictions. |
| Raw-data match | All 18 pseudonyms uniquely matched the complete ordered raw front/back records. Identity lookup retained outside this package. |
| Response integrity | 2,880 front/back trials; 4,306 retained speech trials. Speech cells: 203 with 20 responses, 12 with 19, one with 18. Original first-20 selection retained for the one extra-response cell; no imputation. |

The original baseline/change correlations and pooled prediction plots are archived solely to reproduce the first submission. Their interpretation is superseded by the revised analyses; reproducing a calculation does not validate its scientific interpretation. The final post-on-pre ANCOVA is reproduced by `baseline_ancova.py` and plotted by `baseline_ancova_figure.py`.

## Files and provenance

- `original/analysis.py`, `original/plots.py`, `original/data/`: unchanged copies from the standalone submission package. The original five-figure generators reproduce data and design, not a guaranteed pixel-identical rendering across software versions.
- `original/optimisation/`: original portable optimiser, differentiable acoustic model and source KEMAR SOFA. The SOFA metadata records CC BY-SA 3.0. Saved numerical outputs are in `original/outputs/optimisation/`. Code and recorded objective (approximately 1.3553 → 7.9080 dB) were inspected; the optimiser was not rerun.
- `inputs/tested_optimised_hrtf.sofa`: the filter in the experiment’s HRTF directory, used in the new diagnostics. The later standalone optimiser export is not numerically identical (direction-aligned impulse-response maximum difference 0.000122722; RMS 0.000002711). These are different files, not demonstrated deterministic reproductions of each other.
- `inputs/acoustic_inputs.npz`: anonymised individual, AES log-magnitude and tested optimised impulse responses at 30°/150°, sampled at 48 kHz. One individual’s legacy plot loader selected an older measurement although the stimulus directory used its v2 measurement. Revised Figure 1 uses v2; original behaviour and original Figure 1 archive are unchanged.
- `inputs/optimisation_timing_inputs.npz`: original complex KEMAR representation from the saved optimiser archive and the tested optimised responses. The timing check uses the documented band-limited cross-correlation; it is not a perceptual ITD estimate.
- `inputs/canonical_predictions.csv`: all 20 rendered-mixture predictions per participant/filter/configuration from the repository’s `lavandier_trial_tokens_better_ear_srm_results.csv`. The legacy column name contains “srm”, but its values are effective better-ear SNR. Revised predicted release subtracts matched colocation. The BMLD column is retained for provenance and is not used.
- `inputs/training_trials.csv`, `training_summary.csv`: anonymised raw training records and derived summaries. The successful-block counter is logged before processing the current response; the final successful digit response is included when reconstructing completion. Elapsed time includes any within-session pauses. Criterion/cap comparisons are descriptive and performance-selected.
- `inputs/srm_trial_identity.csv`: anonymised retained responses with triplet identity and presentation order for the order/material sensitivity analysis.
- `revision_analysis.py`, `order_sensitivity.py`, `revision_figures.py`: new analyses. `results/` contains their numerical outputs. New binomial models use participant clusters, bias-reduced covariance and t(17) inference. H3 uses a probability-scale contrast; the exact sign-flip test assumes symmetric participant contrasts under the null.

## Review-driven decisions

1. Keep the original confirmatory contrasts for continuity and label added analyses exploratory/sensitivity analyses.
2. Use binomial modelling for ceiling responses instead of assuming an arcsine transformation resolves the issue.
3. Replace the mathematically coupled baseline/change correlation with post-on-pre ANCOVA using participant-clustered covariance.
4. Replace pooled prediction fits with configuration-specific comparisons; do not interpret unmodelled variance as informational masking.
5. Retain separate HRTF/session baselines because they define the stated SRM contrast; show how baseline movement changes it.
6. Report directional response bias, actual acoustic cues and individual training progression without treating these as proof of veridical localisation or randomised training-dose effects.
7. Address head movement, reverberation and synthetic speech through concise, bounded discussion rather than proposing unsupported new experiments as completed work.
