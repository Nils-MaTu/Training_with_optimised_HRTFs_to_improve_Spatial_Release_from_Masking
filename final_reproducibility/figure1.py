"""Plot the Hearing Research filter against the exact measured SONICOM directions."""
import os
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter1d

ROOT = Path(__file__).resolve().parent
INDIVIDUAL_COLOR = '#147CE5'
OPTIMISED_COLOR = '#FF8A00'
INK = '#202124'
GRID = '#E5E5EA'
SPINE = '#C7C7CC'
N_FFT = 8192


def magnitude_db(spectrum):
    return 20 * np.log10(np.maximum(abs(spectrum), 1e-12))


def smooth_for_display(f, curves, fwhm_octaves):
    """Smooth the displayed dB magnitude on a log-frequency grid."""
    samples_per_octave = 192
    sigma = fwhm_octaves * samples_per_octave / np.sqrt(8 * np.log(2))
    grid = np.arange(np.log2(f[1]), np.log2(f[-1]), 1 / samples_per_octave)
    positive = np.log2(f[1:])
    return np.array([
        np.interp(positive, grid,
                  gaussian_filter1d(np.interp(grid, positive, trace[1:]), sigma))
        for trace in curves.reshape(-1, f.size)
    ]).reshape(*curves.shape[:-1], f.size - 1)


def band_ratios(f, spectrum, fs, target, masker):
    """Return HRTF-only better-ear ratios on a half-ERB-rate grid."""
    erb_high = 11.17 * np.log((fs / 2000 + .312) / (fs / 2000 + 14.675)) + 43
    erb = np.arange(1, round(erb_high) + .5, .5)
    x = np.exp((erb - 43) / 11.17)
    centres = np.round((14.675 * x - .312) / (1 - x) * 1000)
    midpoints = np.sqrt(centres[:-1] * centres[1:])
    edges = np.r_[centres[0]**2 / midpoints[0], midpoints,
                  centres[-1]**2 / midpoints[-1]]
    visible = (centres >= 200) & (centres <= 12000)
    powers = np.stack([
        np.mean(abs(spectrum[..., (f >= lo) & (f < hi)])**2, axis=-1)
        for lo, hi in zip(edges[:-1][visible], edges[1:][visible])
    ], axis=-1)
    snr = 10 * np.log10(powers[..., target, :, :] / powers[..., masker, :, :])
    return centres[visible], snr.max(axis=-2), snr.argmax(axis=-2)


def band_ratios(f, spectrum, fs, target, masker):
    # Same half-ERB-rate grid and integration edges for both filter types.
    erb_high = 11.17 * np.log((fs / 2000 + .312) / (fs / 2000 + 14.675)) + 43
    erb = np.arange(1, round(erb_high) + .5, .5)
    x = np.exp((erb - 43) / 11.17)
    centres = np.round((14.675 * x - .312) / (1 - x) * 1000)
    midpoints = np.sqrt(centres[:-1] * centres[1:])
    edges = np.r_[centres[0]**2 / midpoints[0], midpoints, centres[-1]**2 / midpoints[-1]]
    visible = (centres >= 200) & (centres <= 12000)
    powers = np.stack([
        np.mean(abs(spectrum[..., (f >= lo) & (f < hi)])**2, axis=-1)
        for lo, hi in zip(edges[:-1][visible], edges[1:][visible])
    ], axis=-1)
    snr = 10 * np.log10(powers[..., target, :, :] / powers[..., masker, :, :])
    return centres[visible], snr.max(axis=-2), snr.argmax(axis=-2)


def binaural_band_energy(spectrum, band):
    """Sum one-sided FFT energy across ears, including conjugate bins."""
    power = abs(spectrum)**2
    power[..., 1:-1] *= 2
    return power[..., band].sum(axis=(-2, -1)) / N_FFT


@plt.rc_context(plt.rcParamsDefault)
def render():
    plt.rcParams.update({
        'font.family': 'sans-serif', 'font.size': 8,
        'axes.titlesize': 8.5, 'axes.labelsize': 8,
        'xtick.labelsize': 7.5, 'ytick.labelsize': 7.5,
        'axes.spines.top': False, 'axes.spines.right': False,
        'axes.edgecolor': SPINE, 'axes.linewidth': .8,
        'axes.labelcolor': INK, 'xtick.color': INK, 'ytick.color': INK,
        'figure.facecolor': 'white', 'savefig.facecolor': 'white',
    })
    out = Path(os.environ.get('FIGURE_OUTPUT_DIR', ROOT.parent / 'figures'))
    out.mkdir(parents=True, exist_ok=True)
    data = np.load(ROOT/'inputs/acoustic_inputs.npz')
    fs = float(data['fs'])
    f = np.fft.rfftfreq(N_FFT, 1/fs)
    spectra = {}
    for name, key in [('measured', 'individual_ir'), ('hr', 'optimised_ir')]:
        spectrum = np.fft.rfft(data[key], N_FFT, axis=-1)
        spectra[name] = spectrum
    db = {k: magnitude_db(v) for k,v in spectra.items()}
    smooth = smooth_for_display(f, db['measured'], 1 / 48)
    smooth_hr = smooth_for_display(f, db['hr'], 1 / 24)
    shown = (f >= 200) & (f <= 12000)
    fig, axes = plt.subplots(3, 2, figsize=(7.48, 4.55), sharex=True, sharey='row',
                             gridspec_kw={'height_ratios': [.76, .76, 1]})
    fig.subplots_adjust(left=.075, right=.99, bottom=.105, top=.91,
                        wspace=.07, hspace=.17)
    for j, az in enumerate((30,150)):
        for ear, side in enumerate(('Ipsilateral', 'Contralateral')):
            axis = axes[ear, j]
            for trace in smooth[:, j, ear, :]:
                axis.plot(f[shown], trace[shown[1:]], color=INDIVIDUAL_COLOR, alpha=.55, lw=.45)
            axis.plot(f[shown], smooth_hr[j, ear, shown[1:]], color=OPTIMISED_COLOR, lw=1.2)
            panel = chr(65 + 2 * ear + j)
            axis.text(.02, .96, f'{panel}   {side} magnitude at {az}°',
                      transform=axis.transAxes, ha='left', va='top', fontsize=7.5,
                      color=INK)
        target, masker = j, 1-j
        cfg = ('T30M150', 'T150M30')[j]
        centres, individual, _ = band_ratios(f, spectra['measured'], fs, target, masker)
        _, optimised, optimised_ears = band_ratios(f, spectra['hr'], fs, target, masker)
        for trace in individual:
            axes[2,j].plot(centres, trace, color=INDIVIDUAL_COLOR, alpha=.55, lw=.45)
        axes[2,j].plot(centres, optimised, color=OPTIMISED_COLOR, lw=1.2)
        for ear, marker in enumerate(('o', 's')):
            selected = optimised_ears == ear
            axes[2,j].scatter(centres[selected], optimised[selected], s=10,
                              marker=marker, color=OPTIMISED_COLOR,
                              edgecolor='white', lw=.25, zorder=4)
        axes[2,j].text(.02, .96, f'{chr(69+j)}   {cfg}',
                       transform=axes[2,j].transAxes, ha='left', va='top',
                       fontsize=7.5, color=INK)
        axes[2,j].axhline(0, color='.60', lw=.65, zorder=0)
        axes[2,j].set_xlabel('Frequency / band centre (Hz)')
    for ax in axes.flat:
        ax.set_xscale('log')
        ax.set_xlim(200,12000)
        ax.set_xticks([200,1000,4000,8000,12000],['200','1k','4k','8k','12k'])
        ax.grid(axis='both', color=GRID, lw=.8)
        ax.set_axisbelow(True)
    for ax in axes[:, 1]:
        ax.spines['left'].set_visible(False)
        ax.tick_params(axis='y', left=False, labelleft=False)
    for row, label in enumerate(('Magnitude (dB)', 'Magnitude (dB)',
                                  'Better-ear ratio (dB)')):
        axes[row,0].set_ylabel(label)
    for metric, group in enumerate((axes[:2].flat, axes[2])):
        panels = list(group)
        ymin = min(ax.get_ylim()[0] for ax in panels)
        ymax = max(ax.get_ylim()[1] for ax in panels)
        if metric == 1:
            ymax += .20 * (ymax - ymin)
        for ax in panels:
            ax.set_ylim(ymin, ymax)
    fig.legend(handles=[
        Line2D([],[],color=INDIVIDUAL_COLOR,alpha=.55,label='Individual'),
        Line2D([],[],color=OPTIMISED_COLOR,label='Optimised'),
    ], loc='upper center', bbox_to_anchor=(.5,.99), ncol=2, frameon=False,
       fontsize=8, handlelength=2.4, columnspacing=1.8)
    for j in range(2):
        axes[2,j].legend(handles=[
            Line2D([], [], color=OPTIMISED_COLOR, marker='o', lw=0,
                   markersize=4, markeredgecolor='white', markeredgewidth=.25,
                   label='Left better ear'),
            Line2D([], [], color=OPTIMISED_COLOR, marker='s', lw=0,
                   markersize=4, markeredgecolor='white', markeredgewidth=.25,
                   label='Right better ear'),
        ], loc='upper right', frameon=False, fontsize=6.7, handletextpad=.3,
           handlelength=1, borderaxespad=.4)
    for fmt in ('pdf','png'):
        fig.savefig(out/f'figure1_acoustics.{fmt}',dpi=240,bbox_inches='tight')
    plt.close(fig)
    for label, smoothed in [('measured', smooth), ('hr', smooth_hr)]:
        delta = abs(smoothed[..., shown[1:]] - db[label][..., shown])
        median, p95 = np.percentile(delta, [50, 95])
        print(f'{label} smoothing, median / 95th absolute change: {median:.3f} / {p95:.3f} dB')


if __name__=='__main__':
    render()
