"""
Python port of AMT lavandier2022.m and its dependencies.

Computes the binaural 'effective' target-to-interferer ratio (two-ears benefit)
using weighted BMLD and better-ear advantage with SII weightings.

Ported from:
  amtoolbox-1.6.0/models/lavandier2022.m
  amtoolbox-1.6.0/common/f2erbrate.m, erbrate2f.m, auditoryfilterbank.m,
  bmld.m, f2siiweightings.m
  amtoolbox-1.6.0/mex/comp_auditoryfilterbank_singlefc.c (gammatone filter)

Reference:
  M. Lavandier, T. Vicente, L. Prud'homme. A series of snr-based speech
  intelligibility models in the auditory modeling toolbox. Acta Acustica, 2022.

Differentiable (PyTorch) path: use lavandier2022 with torch tensors and
differentiable=True, or call lavandier2022_torch(...). Forward pass matches
MATLAB when differentiable=False (argmax/max); when differentiable=True,
soft-max and soft-argmax approximations are used so gradients flow.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray

try:
    import torch
    import torch.nn.functional as F

    _TORCH_AVAILABLE = True
except ImportError:
    _TORCH_AVAILABLE = False

# -----------------------------------------------------------------------------
# f2erbrate / erbrate2f (moore1983) - from common/f2erbrate.m, erbrate2f.m
# -----------------------------------------------------------------------------


def f2erbrate(f: float | NDArray, model: str = "glasberg1990") -> float | NDArray:
    """Frequency to ERB rate (Cams). Only 'moore1983' is used by lavandier2022."""
    if model == "moore1983":
        f = np.asarray(f, dtype=float) / 1000.0
        return 11.17 * np.log((f + 0.312) / (f + 14.675)) + 43
    if model == "glasberg1990":
        f = np.asarray(f, dtype=float) / 1000.0
        return 21.366 * np.log10(4.368 * f + 1)
    raise ValueError("Unknown model for the conversion.")


def erbrate2f(erbrate: float | NDArray, model: str = "glasberg1990") -> float | NDArray:
    """ERB rate to frequency (Hz). Only 'moore1983' is used by lavandier2022."""
    erbrate = np.asarray(erbrate, dtype=float)
    if model == "moore1983":
        f = (0.312 - np.exp((erbrate - 43) / 11.17) * 14.675) / (
            np.exp((erbrate - 43) / 11.17) - 1
        )
        return f * 1000.0
    if model == "glasberg1990":
        f = (10.0 ** (erbrate / 21.366) - 1) / 4.368
        return f * 1000.0
    raise ValueError("Unknown model for the conversion.")


# -----------------------------------------------------------------------------
# Gammatone filter (comp_auditoryfilterbank_singlefc / gammatone_c)
# ERB bandwidth: Glasberg & Moore 24.7*(4.37e-3*cf+1), BW_CORRECTION 1.019
# -----------------------------------------------------------------------------

BW_CORRECTION = 1.0190
VERY_SMALL_NUMBER = 1e-200


def _erb_hz(cf: float) -> float:
    """ERB bandwidth in Hz (Glasberg & Moore, used in gammatone filter)."""
    return 24.7 * (4.37e-3 * cf + 1.0)


def comp_auditoryfilterbank_singlefc(
    insig: NDArray, fs: float, fc: float, hrect: int = 0
) -> NDArray:
    """
    Single-channel 4th-order gammatone filter (basilar membrane response).
    Port of comp_auditoryfilterbank_singlefc.c / gammatone_c.
    """
    x = np.asarray(insig, dtype=float).ravel()
    nsamples = x.size
    cf = float(fc)
    fs = float(fs)

    tpt = 2.0 * np.pi / fs
    tptbw = tpt * _erb_hz(cf) * BW_CORRECTION
    a = np.exp(-tptbw)
    gain = (tptbw**4) / 3.0

    a1 = 4.0 * a
    a2 = -6.0 * a * a
    a3 = 4.0 * a * a * a
    a4 = -a * a * a * a
    a5 = a * a

    p0r = p1r = p2r = p3r = p4r = 0.0
    p0i = p1i = p2i = p3i = p4i = 0.0

    coscf = np.cos(tpt * cf)
    sincf = np.sin(tpt * cf)
    qcos = 1.0
    qsin = 0.0

    bm = np.empty(nsamples, dtype=float)

    for t in range(nsamples):
        # Filter part 1 & shift down to d.c.
        p0r = qcos * x[t] + a1 * p1r + a2 * p2r + a3 * p3r + a4 * p4r
        p0i = qsin * x[t] + a1 * p1i + a2 * p2i + a3 * p3i + a4 * p4i

        if abs(p0r) < VERY_SMALL_NUMBER:
            p0r = 0.0
        if abs(p0i) < VERY_SMALL_NUMBER:
            p0i = 0.0

        u0r = p0r + a1 * p1r + a5 * p2r
        u0i = p0i + a1 * p1i + a5 * p2i

        p4r, p3r, p2r, p1r = p3r, p2r, p1r, p0r
        p4i, p3i, p2i, p1i = p3i, p2i, p1i, p0i

        bm[t] = (u0r * qcos + u0i * qsin) * gain
        if hrect == 1 and bm[t] < 0:
            bm[t] = 0.0

        oldcs = qcos
        qcos = coscf * qcos + sincf * qsin
        qsin = coscf * qsin - sincf * oldcs

    return bm


def auditoryfilterbank(
    insig: NDArray, fs: float, fc: float, flag: str = "lavandier2022"
) -> NDArray:
    """
    For lavandier2022: single centre frequency fc, returns filtered signal.
    insig: 1d array (one channel).
    """
    if flag != "lavandier2022":
        raise ValueError("This port only supports flag 'lavandier2022'.")
    return comp_auditoryfilterbank_singlefc(insig, fs, fc, hrect=0)


# -----------------------------------------------------------------------------
# PyTorch (differentiable) building blocks
# -----------------------------------------------------------------------------


def _gammatone_torch(
    insig: "torch.Tensor", fs: float, fc: float, hrect: int = 0
) -> "torch.Tensor":
    """
    Single-channel 4th-order gammatone (basilar membrane response), differentiable.
    Same recurrence as comp_auditoryfilterbank_singlefc; no clipping for grad flow.
    """
    x = insig.ravel()
    nsamples = x.shape[0]
    tpt = 2.0 * np.pi / fs
    tptbw = tpt * _erb_hz(fc) * BW_CORRECTION
    a = np.exp(-tptbw)
    gain = (tptbw**4) / 3.0

    a1 = 4.0 * a
    a2 = -6.0 * a * a
    a3 = 4.0 * a * a * a
    a4 = -a * a * a * a
    a5 = a * a
    coscf = np.cos(tpt * fc)
    sincf = np.sin(tpt * fc)

    device = x.device
    dtype = x.dtype
    p0r = torch.zeros(1, device=device, dtype=dtype)
    p1r = torch.zeros(1, device=device, dtype=dtype)
    p2r = torch.zeros(1, device=device, dtype=dtype)
    p3r = torch.zeros(1, device=device, dtype=dtype)
    p4r = torch.zeros(1, device=device, dtype=dtype)
    p0i = torch.zeros(1, device=device, dtype=dtype)
    p1i = torch.zeros(1, device=device, dtype=dtype)
    p2i = torch.zeros(1, device=device, dtype=dtype)
    p3i = torch.zeros(1, device=device, dtype=dtype)
    p4i = torch.zeros(1, device=device, dtype=dtype)
    qcos = torch.ones(1, device=device, dtype=dtype)
    qsin = torch.zeros(1, device=device, dtype=dtype)

    out = []
    for t in range(nsamples):
        xt = x[t : t + 1]
        p0r_new = qcos * xt + a1 * p1r + a2 * p2r + a3 * p3r + a4 * p4r
        p0i_new = qsin * xt + a1 * p1i + a2 * p2i + a3 * p3i + a4 * p4i
        u0r = p0r_new + a1 * p1r + a5 * p2r
        u0i = p0i_new + a1 * p1i + a5 * p2i
        bm_t = (u0r * qcos + u0i * qsin) * gain
        if hrect == 1:
            bm_t = F.relu(bm_t)
        out.append(bm_t)
        p4r, p3r, p2r, p1r = p3r.clone(), p2r.clone(), p1r.clone(), p0r_new.clone()
        p4i, p3i, p2i, p1i = p3i.clone(), p2i.clone(), p1i.clone(), p0i_new.clone()
        oldcs = qcos.clone()
        qcos = coscf * qcos + sincf * qsin
        qsin = coscf * qsin - sincf * oldcs
    return torch.cat(out, dim=0)


def _xcorr_coeff_torch(
    left: "torch.Tensor", right: "torch.Tensor", maxlag: int
) -> tuple["torch.Tensor", "torch.Tensor"]:
    """Cross-correlation with coeff normalization; returns (iacc, lags) as 1d tensors."""
    left = left.ravel()
    right = right.ravel()
    n = max(left.shape[0], right.shape[0])
    if left.shape[0] < n:
        left = F.pad(left, (0, n - left.shape[0]))
    if right.shape[0] < n:
        right = F.pad(right, (0, n - right.shape[0]))
    norm_prod = torch.sqrt(torch.sum(left**2) * torch.sum(right**2) + 1e-20)
    # R_xy(lag) = sum_n left[n]*right[n+lag]. Conv with kernel=right gives output[i] = R_xy((n-1)-i); slice and flip to get lags -maxlag..maxlag.
    full_corr = F.conv1d(
        left.unsqueeze(0).unsqueeze(0),
        right.unsqueeze(0).unsqueeze(0),
        padding=n - 1,
    ).squeeze()
    zero_idx = n - 1
    i0 = zero_idx - maxlag
    i1 = zero_idx + maxlag + 1
    # full_corr[i] = R_xy((n-1)-i), so full_corr[zero_idx+maxlag] = R_xy(-maxlag). We want iacc[0]=R_xy(-maxlag), so flip.
    iacc = torch.flip(full_corr[i0:i1], (0,)) / norm_prod
    lags = torch.arange(-maxlag, maxlag + 1, device=left.device, dtype=left.dtype)
    return iacc, lags


def _local_do_xcorr_torch(
    left: "torch.Tensor",
    right: "torch.Tensor",
    fs: float,
    fc: float,
    differentiable: bool = False,
    xcorr_temperature: float = 1e10,
) -> tuple["torch.Tensor", "torch.Tensor"]:
    """Returns (phase, coherence). If differentiable, use softmax over lags."""
    maxlag = max(0, int(round(fs / (fc * 2))))
    iacc, lags = _xcorr_coeff_torch(left, right, maxlag)
    if differentiable and xcorr_temperature < 1e8:
        # Soft argmax: w = softmax(temperature * (iacc - iacc.max()))
        iacc_shifted = iacc - iacc.max()
        w = F.softmax(xcorr_temperature * iacc_shifted, dim=0)
        coherence = (w * iacc).sum()
        lag_samp = (w * lags).sum()
        phase = fc * 2.0 * np.pi * lag_samp / fs
        return phase, coherence
    else:
        delay_idx = iacc.argmax().long()
        coherence = iacc[delay_idx]
        lag_samp = lags[delay_idx]
        phase = fc * 2.0 * np.pi * lag_samp / fs
        return phase, coherence


def _bmld_torch(
    coherence: "torch.Tensor",
    phase_target: "torch.Tensor",
    phase_int: "torch.Tensor",
    fc: float,
) -> "torch.Tensor":
    """Binaural masking level difference (dB), differentiable (ReLU at 0)."""
    k = (1 + 0.25**2) * np.exp((2 * np.pi * fc) ** 2 * 0.000105**2)
    ratio = (k - torch.cos(phase_target - phase_int)) / (k - coherence + 1e-12)
    bmld_out = 10.0 * torch.log10(ratio.clamp(min=1e-12))
    return F.relu(bmld_out)


def _better_ear_torch(
    left_snr: "torch.Tensor",
    right_snr: "torch.Tensor",
    differentiable: bool = False,
    softmax_temperature: float = 10.0,
) -> "torch.Tensor":
    """max(left_SNR, right_SNR). If differentiable, use logsumexp smooth max."""
    if differentiable and softmax_temperature < 1e6:
        # (1/τ)*log(exp(τ*L)+exp(τ*R)) → max as τ→∞
        tau = softmax_temperature
        return (1.0 / tau) * torch.logsumexp(
            torch.stack([tau * left_snr, tau * right_snr], dim=0), dim=0
        )
    return torch.maximum(left_snr, right_snr)


# -----------------------------------------------------------------------------
# Cross-correlation (local_do_xcorr) - as in lavandier2022.m
# xcorr(left, right, round(fs/(fc*2)), 'coeff'); [coherence, delay_samp] = max(iacc); phase = fc*2*pi*lags(delay_samp)/fs
# -----------------------------------------------------------------------------


def _xcorr_coeff(left: NDArray, right: NDArray, maxlag: int) -> tuple[NDArray, NDArray]:
    """Cross-correlation with 'coeff' normalization: R / (norm(left)*norm(right))."""
    left = np.asarray(left, dtype=float).ravel()
    right = np.asarray(right, dtype=float).ravel()
    n = max(len(left), len(right))
    # Pad to same length for consistent normalization (MATLAB requires same length for 'coeff')
    if len(left) < n:
        left = np.pad(left, (0, n - len(left)))
    if len(right) < n:
        right = np.pad(right, (0, n - len(right)))
    norm_prod = np.sqrt(np.sum(left**2) * np.sum(right**2))
    if norm_prod == 0:
        norm_prod = 1.0  # avoid div by zero; then iacc will be 0
    # R_xy(lag) = sum_n left[n]*right[n+lag]; we want lags from -maxlag to maxlag
    from scipy.signal import correlate

    full_corr = correlate(left, right, mode="full")
    # full_corr: index (n-1)+k gives R_xy(-k). So [i0:i1] = [R(maxlag)..R(-maxlag)]; reverse so iacc[0]=R(-maxlag) to match lags[0]=-maxlag (MATLAB xcorr order).
    zero_idx = n - 1
    i0 = zero_idx - maxlag
    i1 = zero_idx + maxlag + 1
    iacc = (full_corr[i0:i1] / norm_prod)[::-1]
    lags = np.arange(-maxlag, maxlag + 1, dtype=float)
    return iacc, lags


def local_do_xcorr(
    left: NDArray, right: NDArray, fs: float, fc: float
) -> tuple[float, float]:
    """Interaural cross-correlation; returns (phase, coherence)."""
    maxlag = int(round(fs / (fc * 2)))
    maxlag = max(0, maxlag)
    iacc, lags = _xcorr_coeff(left, right, maxlag)
    delay_idx = int(np.argmax(iacc))
    coherence = float(iacc[delay_idx])
    lag_samp = lags[delay_idx]
    phase = fc * 2.0 * np.pi * lag_samp / fs
    return phase, coherence


# -----------------------------------------------------------------------------
# bmld - from common/bmld.m
# -----------------------------------------------------------------------------


def bmld(coherence: float, phase_target: float, phase_int: float, fc: float) -> float:
    """Binaural masking level difference (dB)."""
    k = (1 + 0.25**2) * np.exp((2 * np.pi * fc) ** 2 * 0.000105**2)
    bmld_out = 10.0 * np.log10((k - np.cos(phase_target - phase_int)) / (k - coherence))
    if bmld_out < 0:
        bmld_out = 0.0
    return float(bmld_out)


# -----------------------------------------------------------------------------
# f2siiweightings - from common/f2siiweightings.m
# -----------------------------------------------------------------------------

_SII_BANDS = np.array([0, 100, 200, 300, 400, 4400, 5300, 6400, 7700, 9500])
_SII_WEIGHTS = np.array(
    [0, 0.0103, 0.0261, 0.0419, 0.0577, 0.0460, 0.0343, 0.0226, 0.0110, 0]
)


def f2siiweightings(fc: ArrayLike) -> NDArray:
    """SII weightings for centre frequencies fc; normalized to sum to 1."""
    fc = np.atleast_1d(np.asarray(fc, dtype=float))
    weightings = np.zeros(len(fc))
    for n in range(len(fc)):
        if fc[n] >= 9500:
            weightings[n] = 0.0
        else:
            # ii = find(bands > fc(n)); weightings(n) = weights(ii(1)-1)
            gt = np.where(_SII_BANDS > fc[n])[0]
            if len(gt) > 0:
                idx = gt[0]  # first index where bands > fc
                weightings[n] = _SII_WEIGHTS[idx - 1]
            else:
                weightings[n] = 0.0
    s = np.sum(weightings)
    if s > 0:
        weightings = weightings / s
    return weightings


# -----------------------------------------------------------------------------
# lavandier2022_torch - differentiable main model
# -----------------------------------------------------------------------------


def lavandier2022_torch(
    target_in: "torch.Tensor",
    int_in: "torch.Tensor",
    fs: float,
    differentiable: bool = False,
    xcorr_temperature: float = 1e10,
    better_ear_temperature: float = 10.0,
) -> tuple["torch.Tensor", "torch.Tensor", "torch.Tensor"]:
    """
    Lavandier 2022 in PyTorch. Returns (twoears_benefit, weighted_bmld, weighted_better_ear).

    When differentiable=False, forward matches MATLAB (argmax, max). Use
    .double() inputs for bit-level consistency with numpy/MATLAB. When
    differentiable=True, soft-max over lags and logsumexp over ears are used
    so gradients flow; xcorr_temperature and better_ear_temperature control
    the sharpness (high = closer to MATLAB forward).
    """
    if not _TORCH_AVAILABLE:
        raise ImportError("PyTorch is required for lavandier2022_torch")
    # Use float64 for consistency with numpy/MATLAB forward
    target_in = target_in.double()
    int_in = int_in.to(target_in.dtype)
    if target_in.dim() == 1:
        target_in = target_in.unsqueeze(1)
    if int_in.dim() == 1:
        int_in = int_in.unsqueeze(1)
    if target_in.shape[1] < 2:
        target_in = torch.cat([target_in, target_in], dim=1)
    if int_in.shape[1] < 2:
        int_in = torch.cat([int_in, int_in], dim=1)

    f_high = fs / 2.0
    erb_high = float(f2erbrate(f_high, "moore1983"))
    nerbs = np.arange(1.0, round(erb_high) + 0.5, 0.5)
    fc_list = np.array([round(erbrate2f(n, "moore1983")) for n in nerbs])
    weightings_np = f2siiweightings(fc_list)
    device = target_in.device
    dtype = target_in.dtype
    weightings = torch.from_numpy(weightings_np).to(device=device, dtype=dtype)

    bmld_pred = []
    better_ear_pred = []
    for n in range(len(nerbs)):
        fc = float(fc_list[n])
        targ_left = _gammatone_torch(target_in[:, 0], fs, fc, 0)
        targ_right = _gammatone_torch(target_in[:, 1], fs, fc, 0)
        int_left = _gammatone_torch(int_in[:, 0], fs, fc, 0)
        int_right = _gammatone_torch(int_in[:, 1], fs, fc, 0)

        int_phase, int_coherence = _local_do_xcorr_torch(
            int_left, int_right, fs, fc, differentiable, xcorr_temperature
        )
        target_phase, _ = _local_do_xcorr_torch(
            targ_left, targ_right, fs, fc, differentiable, xcorr_temperature
        )

        bmld_pred.append(_bmld_torch(int_coherence, target_phase, int_phase, fc))

        mean_tl = targ_left.square().mean()
        mean_il = int_left.square().mean()
        mean_tr = targ_right.square().mean()
        mean_ir = int_right.square().mean()
        eps = 1e-12
        left_snr = 10.0 * torch.log10(mean_tl / (mean_il + eps) + eps)
        right_snr = 10.0 * torch.log10(mean_tr / (mean_ir + eps) + eps)
        better_ear_pred.append(
            _better_ear_torch(
                left_snr, right_snr, differentiable, better_ear_temperature
            )
        )

    # bmld_vec = torch.stack(bmld_pred)
    better_ear_vec = torch.stack(better_ear_pred)
    # weighted_bmld = (bmld_vec * weightings).sum()
    weighted_bmld = 0.0
    weighted_better_ear = (better_ear_vec * weightings).sum()
    twoears_benefit = weighted_better_ear + weighted_bmld
    return twoears_benefit, weighted_bmld, weighted_better_ear


# -----------------------------------------------------------------------------
# Fast path: precompute gammatone per direction, then SRM matrix (no BMLD in current torch)
# -----------------------------------------------------------------------------

# Frequency-domain ERB approximation: band power = sum_f |H(f)|^2 * mask_c(f).
# Much cheaper than time-domain gammatone; preserves same fc_list and SII weightings.
_ERB_MASK_SIGMA_FACTOR = 0.5  # sigma = this * ERB(fc) for Gaussian band mask


def _erb_band_masks_np(freqs_hz: NDArray, fc_list: NDArray) -> NDArray:
    """Gaussian ERB-shaped masks [n_channels, n_freqs] for frequency-domain band power."""
    masks = np.zeros((len(fc_list), len(freqs_hz)), dtype=np.float64)
    for c, fc in enumerate(fc_list):
        bw = _erb_hz(float(fc)) * BW_CORRECTION
        sigma = bw * _ERB_MASK_SIGMA_FACTOR
        # Gaussian in linear freq (good enough for band power)
        masks[c, :] = np.exp(-0.5 * ((freqs_hz - fc) / (sigma + 1e-12)) ** 2)
    return masks


def _lavandier2022_fc_weightings(fs: float):
    """Return (fc_list, weightings_np) for Lavandier ERB scale and SII weightings (numpy)."""
    f_high = fs / 2.0
    erb_high = float(f2erbrate(f_high, "moore1983"))
    nerbs = np.arange(1.0, round(erb_high) + 0.5, 0.5)
    fc_list = np.array([round(erbrate2f(n, "moore1983")) for n in nerbs])
    weightings_np = f2siiweightings(fc_list)
    return fc_list, weightings_np


def _cooke_gammatone_ir_np(fc: float, fs: float, n_samples: int) -> NDArray:
    """Causal impulse response of the 4th-order Cooke gammatone (same as comp_auditoryfilterbank_singlefc)."""
    impulse = np.zeros(n_samples, dtype=np.float64)
    impulse[0] = 1.0
    return comp_auditoryfilterbank_singlefc(impulse, fs, fc, hrect=0)


def _make_gammatone_fir_filters_np(
    fc_list: NDArray, fs: float, filter_len: int
) -> NDArray:
    """Build FIR filters [n_channels, filter_len] for Lavandier fc_list (Cooke gammatone)."""
    filters = np.stack(
        [
            _cooke_gammatone_ir_np(min(float(fc), fs / 2.0 - 0.1), fs, filter_len)
            for fc in fc_list
        ],
        axis=0,
    )
    return filters.astype(np.float32)


# Cache for FIR filters: (fs, filter_len) -> filters np array, so we build once per (fs, filter_len).
_gammatone_fir_cache: dict[tuple[float, int], NDArray] = {}

# Default FIR filter length (samples). Long enough to approximate IIR; 512 ~= 10 ms at 48 kHz.
_GAMMATONE_FIR_LEN = 512


def gammatone_filterbank_all_directions_torch(
    hrir_l: "torch.Tensor",
    hrir_r: "torch.Tensor",
    fs: float,
    use_fir: bool = True,
    filter_len: int = _GAMMATONE_FIR_LEN,
    use_freq_approx: bool = False,
) -> tuple["torch.Tensor", "torch.Tensor", np.ndarray, "torch.Tensor"]:
    """
    Precompute gammatone-filtered signals for every direction and every ERB channel.
    Reduces cost when computing SRM for many (target, masker) pairs: gammatone runs
    once per direction per channel instead of per pair per channel.

    Filtering backends (cheapest first):
      use_freq_approx=True:  ERB band power from |H(f)|^2 with Gaussian masks; no
                             time-domain filter. Same fc_list and SII weightings;
                             slightly different band shape than Cooke gammatone.
      use_fir=True (default): Batched conv1d with precomputed Cooke gammatone FIR
                             (cached). Exact gammatone, fast.
      use_fir=False:          Sample-by-sample IIR (slow, legacy).

    hrir_l, hrir_r: [n_dirs, n_samples] (e.g. from irfft of H_L, H_R).
    Returns:
      filtered_left, filtered_right: [n_dirs, n_channels, n_samples]
      fc_list: 1d numpy array of centre frequencies (Hz)
      weightings: 1d tensor, SII weightings (sum 1), same device/dtype as filtered
    """
    if not _TORCH_AVAILABLE:
        raise ImportError("PyTorch is required")
    n_dirs = hrir_l.shape[0]
    n_samples = hrir_l.shape[1]
    device = hrir_l.device
    dtype = hrir_l.dtype
    fc_list, weightings_np = _lavandier2022_fc_weightings(fs)
    n_channels = len(fc_list)

    if use_freq_approx:
        # Cheapest: band power in frequency domain; output constant-in-time per band
        # so downstream (filtered**2).mean() = band power.
        freqs_hz = np.fft.rfftfreq(n_samples, 1.0 / float(fs))
        masks = _erb_band_masks_np(freqs_hz, fc_list)  # [n_channels, n_freqs]
        masks_t = torch.from_numpy(masks.astype(np.float32)).to(
            device=device, dtype=dtype
        )
        H_L = torch.fft.rfft(hrir_l.to(dtype), dim=1)  # [n_dirs, n_freqs]
        H_R = torch.fft.rfft(hrir_r.to(dtype), dim=1)
        mag_sq_l = H_L.real**2 + H_L.imag**2  # [n_dirs, n_freqs]
        mag_sq_r = H_R.real**2 + H_R.imag**2
        power_left = torch.mm(mag_sq_l, masks_t.T)  # [n_dirs, n_channels]
        power_right = torch.mm(mag_sq_r, masks_t.T)
        # Broadcast so (filtered**2).mean(dim=2) = power; use constant sqrt(power) in time
        filtered_left = (
            torch.sqrt(power_left.clamp(min=1e-20))
            .unsqueeze(2)
            .expand(n_dirs, n_channels, n_samples)
        )
        filtered_right = (
            torch.sqrt(power_right.clamp(min=1e-20))
            .unsqueeze(2)
            .expand(n_dirs, n_channels, n_samples)
        )
    elif use_fir:
        # Fast path: batched conv1d with precomputed Cooke gammatone FIR filters (cached per fs)
        cache_key = (float(fs), filter_len)
        if cache_key not in _gammatone_fir_cache:
            _gammatone_fir_cache[cache_key] = _make_gammatone_fir_filters_np(
                fc_list, float(fs), filter_len
            )
        fir_np = _gammatone_fir_cache[cache_key]
        filters = (
            torch.from_numpy(fir_np).to(device=device, dtype=dtype).unsqueeze(1)
        )  # [n_channels, 1, filter_len]
        # Inputs: [n_dirs, 1, n_samples]; causal pad so output length = n_samples
        k = filter_len
        pad_left = k - 1
        x_l = F.pad(
            hrir_l.to(dtype).unsqueeze(1), (pad_left, 0), mode="constant", value=0.0
        )
        x_r = F.pad(
            hrir_r.to(dtype).unsqueeze(1), (pad_left, 0), mode="constant", value=0.0
        )
        filtered_left = F.conv1d(
            x_l, filters, padding=0
        )  # [n_dirs, n_channels, n_samples]
        filtered_right = F.conv1d(x_r, filters, padding=0)
    else:
        # Slow path: sample-by-sample IIR (original behaviour)
        # MPS does not support float64; use float32 on MPS, float64 elsewhere for legacy behaviour
        slow_dtype = torch.float32 if device.type == "mps" else torch.float64
        hrir_l = hrir_l.to(slow_dtype)
        hrir_r = hrir_r.to(slow_dtype)
        filtered_left = torch.zeros(
            n_dirs, n_channels, n_samples, device=device, dtype=slow_dtype
        )
        filtered_right = torch.zeros(
            n_dirs, n_channels, n_samples, device=device, dtype=slow_dtype
        )
        for d in range(n_dirs):
            for n in range(n_channels):
                fc = float(fc_list[n])
                filtered_left[d, n, :] = _gammatone_torch(hrir_l[d], fs, fc, 0)
                filtered_right[d, n, :] = _gammatone_torch(hrir_r[d], fs, fc, 0)
        if dtype != filtered_left.dtype:
            filtered_left = filtered_left.to(dtype)
            filtered_right = filtered_right.to(dtype)

    weightings = torch.from_numpy(weightings_np).to(device=device, dtype=dtype)
    return filtered_left, filtered_right, fc_list, weightings


def lavandier2022_torch_srm_matrix_from_filtered(
    filtered_left: "torch.Tensor",
    filtered_right: "torch.Tensor",
    weightings: "torch.Tensor",
    fs: float,
    differentiable: bool = False,
    better_ear_temperature: float = 20.0,
) -> "torch.Tensor":
    """
    Compute SRM matrix [n_dirs, n_dirs] from precomputed gammatone-filtered signals.
    Matches lavandier2022_torch when BMLD is disabled (better-ear only, weighted by SII).
    No gammatone is run here; use gammatone_filterbank_all_directions_torch first.

    Fully vectorized so all (i, j, channel) work runs in parallel on device.

    filtered_left, filtered_right: [n_dirs, n_channels, n_samples]
    weightings: [n_channels], SII weightings (sum 1)
    """
    if not _TORCH_AVAILABLE:
        raise ImportError("PyTorch is required")
    n_dirs, n_channels, _ = filtered_left.shape
    eps = 1e-12
    # Per-band power [n_dirs, n_channels]
    power_left = filtered_left.square().mean(dim=2)
    power_right = filtered_right.square().mean(dim=2)
    # All (i, j) pairs: power_left[i] vs power_left[j] -> left_snr[i,j,n]
    # [n_dirs, 1, n_channels] / [1, n_dirs, n_channels] -> [n_dirs, n_dirs, n_channels]
    pl_i = power_left.unsqueeze(1)  # [n_dirs, 1, n_channels]
    pl_j = power_left.unsqueeze(0)  # [1, n_dirs, n_channels]
    pr_i = power_right.unsqueeze(1)
    pr_j = power_right.unsqueeze(0)
    left_snr = 10.0 * torch.log10(pl_i / (pl_j + eps) + eps)
    right_snr = 10.0 * torch.log10(pr_i / (pr_j + eps) + eps)
    # Better-ear (element-wise max or soft-max) [n_dirs, n_dirs, n_channels]
    if differentiable and better_ear_temperature < 1e6:
        tau = better_ear_temperature
        better_ear = (1.0 / tau) * torch.logsumexp(
            torch.stack([tau * left_snr, tau * right_snr], dim=0), dim=0
        )
    else:
        better_ear = torch.maximum(left_snr, right_snr)
    # Weighted sum over channels -> [n_dirs, n_dirs]
    S = (better_ear * weightings.view(1, 1, -1)).sum(dim=2)
    return S


# -----------------------------------------------------------------------------
# lavandier2022 - main model (numpy or torch dispatch)
# -----------------------------------------------------------------------------


def lavandier2022(
    target_in: ArrayLike | "torch.Tensor",
    int_in: ArrayLike | "torch.Tensor",
    fs: float,
    differentiable: bool = False,
    xcorr_temperature: float = 1e10,
    better_ear_temperature: float = 10.0,
) -> tuple[float, float, float] | tuple["torch.Tensor", "torch.Tensor", "torch.Tensor"]:
    """
    Compute the binaural 'effective' target-to-interferer ratio.

    Parameters
    ----------
    target_in : (N, 2) array or torch.Tensor
        Target at left and right ear (columns 0 and 1).
    int_in : (N, 2) array or torch.Tensor
        Interferer at left and right ear.
    fs : float
        Sampling frequency in Hz.
    differentiable : bool
        If True and inputs are torch tensors, use differentiable (soft) approximations
        so gradients flow; forward may differ slightly from MATLAB. If False, forward
        matches MATLAB (argmax, max).
    xcorr_temperature : float
        For differentiable xcorr: softmax temperature over lags. High (e.g. 1e10)
        approximates argmax; lower gives smoother gradients.
    better_ear_temperature : float
        For differentiable better-ear: logsumexp temperature. High approximates max.

    Returns
    -------
    twoears_benefit, weighted_bmld, weighted_better_ear
        Floats if numpy input; tensors if torch input.
    """
    if (
        _TORCH_AVAILABLE
        and isinstance(target_in, torch.Tensor)
        and isinstance(int_in, torch.Tensor)
    ):
        out = lavandier2022_torch(
            target_in,
            int_in,
            fs,
            differentiable=differentiable,
            xcorr_temperature=xcorr_temperature,
            better_ear_temperature=better_ear_temperature,
        )
        return out

    target_in = np.asarray(target_in, dtype=float)
    int_in = np.asarray(int_in, dtype=float)
    if target_in.ndim == 1:
        target_in = target_in.reshape(-1, 1)
    if int_in.ndim == 1:
        int_in = int_in.reshape(-1, 1)
    if target_in.shape[1] < 2:
        target_in = np.column_stack([target_in.ravel(), target_in.ravel()])
    if int_in.shape[1] < 2:
        int_in = np.column_stack([int_in.ravel(), int_in.ravel()])

    # ERB scale: 1:0.5:round(f2erbrate(fs/2,'moore1983'))
    f_high = fs / 2.0
    erb_high = f2erbrate(f_high, "moore1983")
    nerbs = np.arange(1.0, round(erb_high) + 0.5, 0.5)

    fc = np.zeros(len(nerbs))
    bmld_prediction = np.zeros(len(nerbs))
    better_ear_prediction = np.zeros(len(nerbs))

    for n in range(len(nerbs)):
        fc[n] = round(erbrate2f(nerbs[n], "moore1983"))

        targ_left = auditoryfilterbank(target_in[:, 0], fs, fc[n], "lavandier2022")
        targ_right = auditoryfilterbank(target_in[:, 1], fs, fc[n], "lavandier2022")
        int_left = auditoryfilterbank(int_in[:, 0], fs, fc[n], "lavandier2022")
        int_right = auditoryfilterbank(int_in[:, 1], fs, fc[n], "lavandier2022")

        int_phase, int_coherence = local_do_xcorr(int_left, int_right, fs, fc[n])
        target_phase, _ = local_do_xcorr(targ_left, targ_right, fs, fc[n])

        bmld_prediction[n] = bmld(int_coherence, target_phase, int_phase, fc[n])

        # better-ear SNR in dB: 10*log10(mean(sig.^2)/mean(other.^2)) (MATLAB behaviour)
        mean_tl = np.mean(targ_left**2)
        mean_il = np.mean(int_left**2)
        mean_tr = np.mean(targ_right**2)
        mean_ir = np.mean(int_right**2)
        if mean_il > 0:
            left_SNR = 10.0 * np.log10(mean_tl / mean_il)
        else:
            left_SNR = np.inf if mean_tl > 0 else np.nan
        if mean_ir > 0:
            right_SNR = 10.0 * np.log10(mean_tr / mean_ir)
        else:
            right_SNR = np.inf if mean_tr > 0 else np.nan
        better_ear_prediction[n] = max(left_SNR, right_SNR)

    weightings = f2siiweightings(fc)
    weighted_bmld = float(np.sum(bmld_prediction * weightings))
    weighted_better_ear = float(np.sum(better_ear_prediction * weightings))
    twoears_benefit = weighted_better_ear + weighted_bmld

    return twoears_benefit, weighted_bmld, weighted_better_ear


def lavandier2022_better_ear_per_band(
    target_in: NDArray, int_in: NDArray, fs: float
) -> tuple[NDArray, NDArray, list[str]]:
    """
    Better-ear SNR (dB) per auditory band and which ear is better.

    Parameters
    ----------
    target_in : (N, 2) array
        Target at left and right ear (columns 0 and 1).
    int_in : (N, 2) array
        Interferer at left and right ear.
    fs : float
        Sampling frequency in Hz.

    Returns
    -------
    fc : ndarray
        Centre frequency (Hz) for each band.
    better_ear_dB : ndarray
        Better-ear SNR in dB per band (max of left and right SNR).
    which_ear : list of str
        'L' or 'R' for each band indicating which ear had the higher SNR.
    """
    target_in = np.asarray(target_in, dtype=float)
    int_in = np.asarray(int_in, dtype=float)
    if target_in.ndim == 1:
        target_in = target_in.reshape(-1, 1)
    if int_in.ndim == 1:
        int_in = int_in.reshape(-1, 1)
    if target_in.shape[1] < 2:
        target_in = np.column_stack([target_in.ravel(), target_in.ravel()])
    if int_in.shape[1] < 2:
        int_in = np.column_stack([int_in.ravel(), int_in.ravel()])

    f_high = fs / 2.0
    erb_high = f2erbrate(f_high, "moore1983")
    nerbs = np.arange(1.0, round(erb_high) + 0.5, 0.5)
    n_bands = len(nerbs)

    fc = np.zeros(n_bands)
    better_ear_dB = np.zeros(n_bands)
    which_ear: list[str] = []

    for n in range(n_bands):
        fc[n] = round(erbrate2f(nerbs[n], "moore1983"))

        targ_left = auditoryfilterbank(target_in[:, 0], fs, fc[n], "lavandier2022")
        targ_right = auditoryfilterbank(target_in[:, 1], fs, fc[n], "lavandier2022")
        int_left = auditoryfilterbank(int_in[:, 0], fs, fc[n], "lavandier2022")
        int_right = auditoryfilterbank(int_in[:, 1], fs, fc[n], "lavandier2022")

        mean_tl = np.mean(targ_left**2)
        mean_il = np.mean(int_left**2)
        mean_tr = np.mean(targ_right**2)
        mean_ir = np.mean(int_right**2)
        if mean_il > 0:
            left_snr = 10.0 * np.log10(mean_tl / mean_il)
        else:
            left_snr = np.inf if mean_tl > 0 else np.nan
        if mean_ir > 0:
            right_snr = 10.0 * np.log10(mean_tr / mean_ir)
        else:
            right_snr = np.inf if mean_tr > 0 else np.nan

        if np.isfinite(left_snr) and np.isfinite(right_snr):
            if left_snr >= right_snr:
                better_ear_dB[n] = left_snr
                which_ear.append("L")
            else:
                better_ear_dB[n] = right_snr
                which_ear.append("R")
        elif np.isfinite(left_snr):
            better_ear_dB[n] = left_snr
            which_ear.append("L")
        elif np.isfinite(right_snr):
            better_ear_dB[n] = right_snr
            which_ear.append("R")
        else:
            better_ear_dB[n] = np.nan
            which_ear.append("L")

    return fc, better_ear_dB, which_ear
