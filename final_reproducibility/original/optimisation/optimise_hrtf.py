"""Generate the optimised KEMAR HRTF used in the experiment."""

from pathlib import Path

import h5py
import numpy as np
import torch
import torch.nn as nn
from lavandier2022 import (
    gammatone_filterbank_all_directions_torch,
    lavandier2022_torch_srm_matrix_from_filtered,
)
from sklearn.decomposition import PCA
from spaudiopy.io import sofa_to_sh
from spaudiopy.sph import sh_matrix

# Use frequency-domain ERB band power instead of time-domain gammatone for SRM.
# Faster and usually very close; set False for exact Cooke gammatone.
USE_FAST_ERB_APPROX = True

SCRIPT_DIR = Path(__file__).resolve().parent
INPUT_SOFA = SCRIPT_DIR / "KEMAR_GRAS_EarSim_LargeEars_FreeFieldComp_48kHz.sofa"
OUTPUT_DIR = SCRIPT_DIR.parent / "outputs" / "optimisation"


def _compute_srm_matrix_lavandier_torch(
    H_L: torch.Tensor,
    H_R: torch.Tensor,
    ir_length: int,
    fs: float,
    differentiable: bool = True,
    use_freq_approx: bool = False,
) -> torch.Tensor:
    """
    Compute full SRM matrix (n_azimuths, n_azimuths) from frequency-domain HRTFs
    using differentiable Lavandier 2022 (time-domain gammatone + better-ear).
    Fast path: gammatone is run once per direction per channel, then SRM from filtered
    signals (avoids recomputing gammatone for every pair).

    use_freq_approx: If True, use ERB band power from |H(f)|^2 (no gammatone filter).
                     Cheaper and often very close to the full gammatone SRM.
    H_L, H_R: [n_azimuths, n_freqs] complex.
    """
    hrir_l = torch.fft.irfft(H_L, n=ir_length, dim=1)
    hrir_r = torch.fft.irfft(H_R, n=ir_length, dim=1)
    filtered_left, filtered_right, _fc_list, weightings = (
        gammatone_filterbank_all_directions_torch(
            hrir_l, hrir_r, fs, use_freq_approx=use_freq_approx
        )
    )
    S = lavandier2022_torch_srm_matrix_from_filtered(
        filtered_left,
        filtered_right,
        weightings,
        fs,
        differentiable=differentiable,
    )
    return S


def load_hrtfs_for_azimuth_range(
    sofa_file: str,
    az_min: float = 0.0,
    az_max: float = 90.0,
    n_azimuths: int = 19,
    el: float = 0.0,
    N_sph: int = 16,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float]:
    """Load HRTFs for a range of azimuths from SOFA file."""
    print(f"Loading HRTFs from {sofa_file}...")
    sh_coeffs, fs = sofa_to_sh(sofa_file, N_sph, sh_type="real")
    n_ears, n_coeffs, n_time = sh_coeffs.shape
    sh_coeffs_freq = np.fft.rfft(sh_coeffs, axis=2)  # [ears, coeffs, freqs]

    # Generate azimuth grid
    # If range includes 180°, exclude it to avoid duplicate with -180°
    if abs(az_max - 180.0) < 1e-6:
        # Generate n_azimuths+1 points, then remove the last one (180°)
        az_deg_temp = np.linspace(az_min, az_max, n_azimuths + 1)
        az_deg = az_deg_temp[:-1]  # Exclude last point (180°)
    else:
        az_deg = np.linspace(az_min, az_max, n_azimuths)
    # Normalize all azimuths to [-180, 180) range
    az_deg = np.mod(az_deg + 180.0, 360.0) - 180.0
    az_rad = np.deg2rad(az_deg)
    el_rad = np.deg2rad(np.full_like(az_deg, el))

    # Reconstruct HRTFs for each azimuth
    zen = np.pi / 2.0 - el_rad
    Y = sh_matrix(N_sph, az_rad, zen, sh_type="real")  # [n_azimuths, n_coeffs]
    H = np.einsum("dc,ecf->def", Y, sh_coeffs_freq)  # [n_azimuths, ears, freqs]
    H_L = H[:, 0, :]  # [n_azimuths, freqs]
    H_R = H[:, 1, :]  # [n_azimuths, freqs]

    n_freqs = H_L.shape[1]
    ir_length = (n_freqs - 1) * 2
    freqs_hz = np.fft.rfftfreq(ir_length, d=1.0 / fs)

    print(f"Loaded {n_azimuths} HRTFs for azimuths {az_min}° to {az_max}°")
    print(f"Frequency range: {freqs_hz[0]:.1f} Hz to {freqs_hz[-1]:.1f} Hz")
    print(f"Sample rate: {fs} Hz")

    return H_L, H_R, az_deg, freqs_hz, fs


def apply_spatial_pca_to_magnitude(
    H_L: np.ndarray,
    H_R: np.ndarray,
    n_components: int | None = None,
    whiten: bool = False,
) -> tuple[PCA, np.ndarray, np.ndarray, np.ndarray]:
    """Apply Spatial PCA to HRTF magnitude spectra (full bandwidth)."""
    mag_L, mag_R = np.abs(H_L), np.abs(H_R)
    X = np.vstack([mag_L, mag_R])

    pca = PCA(n_components=n_components, whiten=whiten)
    transformed = pca.fit_transform(X)
    components = pca.components_
    explained_variance = pca.explained_variance_ratio_

    print("\nSpatial PCA Results:")
    print(f"  Input shape: {X.shape}")
    print(f"  Components shape: {components.shape}")
    print(f"  Explained variance (top 5): {explained_variance[:5] * 100}")
    print(f"  Cumulative variance (top 5): {np.cumsum(explained_variance[:5]) * 100}")

    return pca, components, explained_variance, transformed


def reconstruct_magnitude_from_pca(
    pca: PCA,
    transformed: np.ndarray,
    n_azimuths: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Reconstruct magnitude spectra from PCA."""
    mag_recon = pca.inverse_transform(transformed)
    mag_L_recon = mag_recon[:n_azimuths, :]
    mag_R_recon = mag_recon[n_azimuths:, :]
    return mag_L_recon, mag_R_recon


class PCAHRTFOptimizer(nn.Module):
    """PyTorch module for optimizing HRTF via PCA component weights."""

    def __init__(
        self,
        pca_mean: torch.Tensor,
        pca_components: torch.Tensor,
        transformed_baseline: torch.Tensor,
        n_azimuths: int,
        n_components_optimize: int,
        device: str = "cpu",
    ):
        super().__init__()
        self.pca_mean = pca_mean.to(device)
        self.pca_components = pca_components.to(device)
        self.transformed_baseline = transformed_baseline.to(device)
        self.n_azimuths = n_azimuths
        self.n_components_optimize = n_components_optimize
        self.weight_delta = nn.Parameter(
            torch.zeros(2 * n_azimuths, n_components_optimize, device=device)
        )

    def forward(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Reconstruct magnitude spectra from optimized PCA weights."""
        transformed_opt = self.transformed_baseline.clone()
        transformed_opt[:, : self.n_components_optimize] += self.weight_delta
        mag_recon = self.pca_mean.unsqueeze(0) + torch.matmul(
            transformed_opt, self.pca_components
        )
        mag_L = mag_recon[: self.n_azimuths, :]
        mag_R = mag_recon[self.n_azimuths :, :]
        return mag_L, mag_R


def _freq_band_mask(freqs_hz: torch.Tensor, f_min: float, f_max: float) -> torch.Tensor:
    """Boolean mask for frequency band [f_min, f_max] (for torch)."""
    return (freqs_hz >= f_min) & (freqs_hz <= f_max)


def compute_broadband_ild_torch(
    mag_L: torch.Tensor,
    mag_R: torch.Tensor,
    freqs_hz: torch.Tensor,
    f_min: float = 200.0,
    f_max: float = 8000.0,
    eps: float = 1e-10,
) -> torch.Tensor:
    """Broadband ILD in dB (PyTorch)."""
    m = _freq_band_mask(freqs_hz, f_min, f_max)
    rms_L = torch.sqrt(torch.mean(mag_L[:, m] ** 2, dim=1))
    rms_R = torch.sqrt(torch.mean(mag_R[:, m] ** 2, dim=1))
    return 20.0 * torch.log10((rms_L + eps) / (rms_R + eps))


def compute_combined_rms_torch(
    mag_L: torch.Tensor,
    mag_R: torch.Tensor,
    freqs_hz: torch.Tensor,
    f_min: float = 200.0,
    f_max: float = 8000.0,
) -> torch.Tensor:
    """Combined RMS over both ears per azimuth (linear scale)."""
    m = _freq_band_mask(freqs_hz, f_min, f_max)
    combined_power = mag_L[:, m] ** 2 + mag_R[:, m] ** 2
    return torch.sqrt(torch.mean(combined_power, dim=1))


def optimize_hrtf_with_gradient_descent(
    H_L: np.ndarray,
    H_R: np.ndarray,
    pca: PCA,
    components: np.ndarray,
    transformed: np.ndarray,
    az_deg: np.ndarray,
    freqs_hz: np.ndarray,
    fs: float,
    n_components_optimize: int = 5,
    n_steps: int = 500,
    lr: float = 1e-2,
    lambda_ild: float = 0.1,
    lambda_rms: float = 0.1,
    lambda_ild_preserve: float = 0.0,
    lambda_srm_penalty: float = 1.0,
    target_dirs: np.ndarray | None = None,
    masker_dirs: np.ndarray | None = None,
    device: str = "cpu",
) -> tuple[np.ndarray, np.ndarray, dict]:
    """Optimize HRTF via gradient descent to maximize SRM with optional ILD/RMS penalties.

    SRM loss: maximize mean SRM over front-back symmetric pairs only (e.g. 20°↔160°, 40°↔140°).
    Penalty = lambda_srm_penalty * mean(relu(baseline - current)) over all other pairs.
    """
    n_azimuths = len(az_deg)
    symm_pairs = find_symmetrical_azimuth_pairs(az_deg)
    if symm_pairs:
        print(f"\nFound {len(symm_pairs)} symmetrical front-back pairs (SRM loss):")
        for fi, bi in symm_pairs[:5]:
            print(f"  {az_deg[fi]:.0f}° <-> {az_deg[bi]:.0f}°")

    # Move data to device
    def to_f32(x):
        return torch.as_tensor(x, dtype=torch.float32, device=device)

    def to_c64(x):
        return torch.as_tensor(x, dtype=torch.complex64, device=device)

    pca_mean = to_f32(pca.mean_)
    pca_components = to_f32(components)
    transformed_baseline = to_f32(transformed)
    phase_L, phase_R = to_f32(np.angle(H_L)), to_f32(np.angle(H_R))
    freqs_t = to_f32(freqs_hz)
    mag_L_orig, mag_R_orig = to_f32(np.abs(H_L)), to_f32(np.abs(H_R))

    rms_combined_orig = compute_combined_rms_torch(mag_L_orig, mag_R_orig, freqs_t)
    ild_orig = (
        compute_broadband_ild_torch(mag_L_orig, mag_R_orig, freqs_t)
        if lambda_ild_preserve > 0
        else None
    )

    model = PCAHRTFOptimizer(
        pca_mean,
        pca_components,
        transformed_baseline,
        n_azimuths,
        n_components_optimize,
        device=device,
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # SRM: maximize mean SRM over front-back symmetric pairs only
    primary_mask, floor_mask = _srm_front_back_masks(symm_pairs, n_azimuths, device)
    n_primary = primary_mask.sum().item()
    n_floor = floor_mask.sum().item()
    n_freqs = H_L.shape[1]
    ir_length = (n_freqs - 1) * 2

    with torch.no_grad():
        srm_baseline = None
        srm_baseline_matrix = None
        if target_dirs is not None and masker_dirs is not None:
            S0 = _compute_srm_matrix_lavandier_torch(
                to_c64(H_L).to(device),
                to_c64(H_R).to(device),
                ir_length,
                fs,
                differentiable=True,
                use_freq_approx=USE_FAST_ERB_APPROX,
            )
            srm_baseline_matrix = S0
            if n_primary > 0:
                srm_baseline = S0[primary_mask].mean().item()
            print(
                f"  SRM (Lavandier 2022): maximize front-back symmetric pairs (n_primary={int(n_primary)}), "
                f"penalty for drop on other pairs (n_floor={int(n_floor)}), λ_penalty={lambda_srm_penalty}"
            )

    print(
        f"\nOptimizing ({n_steps} steps)... Baseline SRM (front-back): {srm_baseline or 'N/A'}"
    )
    if symm_pairs:
        print(f"  lambda_ild={lambda_ild}, lambda_rms={lambda_rms}")
    if lambda_ild_preserve > 0:
        print(f"  lambda_ild_preserve={lambda_ild_preserve}")

    history = []
    zero = torch.tensor(0.0, device=device)
    for step in range(1, n_steps + 1):
        optimizer.zero_grad()
        mag_L, mag_R = model()
        H_L_opt = mag_L * torch.exp(1j * phase_L)
        H_R_opt = mag_R * torch.exp(1j * phase_R)

        # Maximize SRM on front-back symmetric pairs only
        loss_srm = zero
        srm_mean = zero
        if target_dirs is not None and masker_dirs is not None:
            S = _compute_srm_matrix_lavandier_torch(
                H_L_opt,
                H_R_opt,
                ir_length,
                fs,
                differentiable=True,
                use_freq_approx=USE_FAST_ERB_APPROX,
            )
            if n_primary > 0:
                srm_mean = S[primary_mask].mean()
                loss_srm = -srm_mean
            if (
                lambda_srm_penalty > 0
                and n_floor > 0
                and srm_baseline_matrix is not None
            ):
                loss_srm = (
                    loss_srm
                    + lambda_srm_penalty
                    * torch.relu(srm_baseline_matrix[floor_mask] - S[floor_mask]).mean()
                )

        loss_ild = zero
        if symm_pairs and lambda_ild > 0:
            ild = compute_broadband_ild_torch(mag_L, mag_R, freqs_t)
            loss_ild = torch.mean(
                torch.stack([(ild[i] - ild[j]) ** 2 for i, j in symm_pairs])
            )

        loss_rms = zero
        if lambda_rms > 0:
            rms = compute_combined_rms_torch(mag_L, mag_R, freqs_t)
            loss_rms = torch.mean((rms - rms_combined_orig) ** 2)

        loss_ild_preserve = zero
        if lambda_ild_preserve > 0 and ild_orig is not None:
            loss_ild_preserve = torch.mean(
                (compute_broadband_ild_torch(mag_L, mag_R, freqs_t) - ild_orig) ** 2
            )

        loss = (
            loss_srm
            + lambda_ild * loss_ild
            + lambda_rms * loss_rms
            + lambda_ild_preserve * loss_ild_preserve
        )
        loss.backward()
        optimizer.step()

        srm_val = srm_mean.item() if hasattr(srm_mean, "item") else 0.0
        history.append(
            {
                "step": step,
                "srm": srm_val,
                "loss": loss.item(),
                "loss_ild": loss_ild.item() if symm_pairs else 0.0,
                "loss_rms": loss_rms.item() if lambda_rms > 0 else 0.0,
                "loss_ild_preserve": (
                    loss_ild_preserve.item() if lambda_ild_preserve > 0 else 0.0
                ),
            }
        )

        if step == 1 or step == n_steps or step % 2 == 0:
            extra = (
                [f"ILD: {loss_ild.item():.4f}"] if symm_pairs and lambda_ild > 0 else []
            )
            if lambda_rms > 0:
                extra.append(f"RMS: {loss_rms.item():.4f}")
            if lambda_ild_preserve > 0:
                extra.append(f"ILD_pres: {loss_ild_preserve.item():.4f}")
            suf = ", " + ", ".join(extra) if extra else ""
            print(
                f"Step {step:4d}/{n_steps}  SRM: {srm_val:6.2f} dB  loss: {loss.item():.4f}{suf}"
            )

    with torch.no_grad():
        mag_L_f, mag_R_f = model()
        H_L_final = mag_L_f * torch.exp(1j * phase_L)
        H_R_final = mag_R_f * torch.exp(1j * phase_R)
        srm_optimized = None
        if target_dirs is not None and masker_dirs is not None and n_primary > 0:
            S_f = _compute_srm_matrix_lavandier_torch(
                H_L_final,
                H_R_final,
                ir_length,
                fs,
                differentiable=True,
                use_freq_approx=USE_FAST_ERB_APPROX,
            )
            srm_optimized = S_f[primary_mask].mean().item()

    print("\nDone.")
    if srm_baseline is not None and srm_optimized is not None:
        print(
            f"  SRM (front-back): {srm_baseline:.2f} → {srm_optimized:.2f} dB (+{srm_optimized - srm_baseline:.2f})"
        )

    return (
        H_L_final.cpu().numpy(),
        H_R_final.cpu().numpy(),
        {
            "srm_baseline": srm_baseline,
            "srm_optimized": srm_optimized,
            "history": history,
            "n_components_optimized": n_components_optimize,
        },
    )


def compute_broadband_ild(
    H_L: np.ndarray,
    H_R: np.ndarray,
    freqs_hz: np.ndarray,
    f_min: float = 200.0,
    f_max: float = 8000.0,
) -> np.ndarray:
    """Broadband ILD in dB per azimuth (NumPy)."""
    mag_L = np.abs(H_L)
    mag_R = np.abs(H_R)
    m = (freqs_hz >= f_min) & (freqs_hz <= f_max)
    rms_L = np.sqrt(np.mean(mag_L[:, m] ** 2, axis=1))
    rms_R = np.sqrt(np.mean(mag_R[:, m] ** 2, axis=1))
    return 20 * np.log10((rms_L + 1e-10) / (rms_R + 1e-10))


def _freq_to_erb_np(f: float | np.ndarray) -> np.ndarray:
    """Hz → ERB scale."""
    f = np.asarray(f, dtype=float)
    return 9.26449 * np.log(1.0 + f / (9.26449 * 24.7))


def _erb_to_freq_np(erb: float | np.ndarray) -> np.ndarray:
    """ERB scale → Hz."""
    erb = np.asarray(erb, dtype=float)
    return 9.26449 * 24.7 * (np.exp(erb / 9.26449) - 1.0)


def create_erb_filterbank(
    freqs_hz: np.ndarray, f_min: float = 20.0, f_max: float = 1250.0, n_bands: int = 32
) -> tuple[np.ndarray, np.ndarray]:
    """ERB-spaced bands and band index per frequency bin."""
    erb_min, erb_max = _freq_to_erb_np(f_min), _freq_to_erb_np(f_max)
    erb_centers = np.linspace(erb_min, erb_max, n_bands)
    fc_erb = _erb_to_freq_np(erb_centers)
    freqs_erb = _freq_to_erb_np(freqs_hz)
    band_assignments = np.argmin(
        np.abs(freqs_erb[:, np.newaxis] - erb_centers[np.newaxis, :]), axis=1
    )
    return fc_erb, band_assignments


def magnify_ipd_between_symmetrical_pairs(
    H_L: np.ndarray,
    H_R: np.ndarray,
    freqs_hz: np.ndarray,
    az_deg: np.ndarray,
    symm_pairs: list[tuple[int, int]],
    f_max: float = 1250.0,
    ipd_magnification_factor: float = 1.5,
) -> tuple[np.ndarray, np.ndarray]:
    """Magnify IPD difference between front/back pairs: equalize IPD to mean, then alternate magnify/attenuate per ERB band."""
    H_L_mod = H_L.copy()
    H_R_mod = H_R.copy()
    freq_mask = freqs_hz <= f_max
    freqs_band = freqs_hz[freq_mask]
    if len(freqs_band) == 0:
        print(f"Warning: No frequencies below {f_max} Hz. Skipping IPD magnification.")
        return H_L_mod, H_R_mod

    fc_erb, band_assignments = create_erb_filterbank(
        freqs_band, f_min=20.0, f_max=f_max, n_bands=2
    )
    print(
        f"\nIPD magnification: {len(symm_pairs)} pairs, f<{f_max:.0f} Hz, {len(fc_erb)} ERB bands, factor={ipd_magnification_factor:.2f}"
    )

    for front_idx, back_idx in symm_pairs:
        H_L_f, H_R_f = H_L_mod[front_idx], H_R_mod[front_idx]
        H_L_b, H_R_b = H_L_mod[back_idx], H_R_mod[back_idx]

        ipd_f = np.angle(H_L_f * np.conj(H_R_f))[freq_mask]
        ipd_b = np.angle(H_L_b * np.conj(H_R_b))[freq_mask]
        mean_ipd = (ipd_f + ipd_b) / 2.0
        adj_f = np.exp(0.5j * (mean_ipd - ipd_f))
        adj_b = np.exp(0.5j * (mean_ipd - ipd_b))

        H_L_f_eq = H_L_f.copy()
        H_R_f_eq = H_R_f.copy()
        H_L_b_eq = H_L_b.copy()
        H_R_b_eq = H_R_b.copy()
        H_L_f_eq[freq_mask] = H_L_f[freq_mask] * adj_f
        H_R_f_eq[freq_mask] = H_R_f[freq_mask] * np.conj(adj_f)
        H_L_b_eq[freq_mask] = H_L_b[freq_mask] * adj_b
        H_R_b_eq[freq_mask] = H_R_b[freq_mask] * np.conj(adj_b)

        for band_idx in range(len(fc_erb)):
            band_mask_full = np.zeros(len(freqs_hz), dtype=bool)
            band_mask_full[freq_mask] = band_assignments == band_idx
            if not np.any(band_mask_full):
                continue

            H_L_fb = H_L_f_eq[band_mask_full]
            H_R_fb = H_R_f_eq[band_mask_full]
            H_L_bb = H_L_b_eq[band_mask_full]
            H_R_bb = H_R_b_eq[band_mask_full]
            ipd_fb = np.angle(H_L_fb * np.conj(H_R_fb))
            ipd_bb = np.angle(H_L_bb * np.conj(H_R_bb))

            if band_idx % 2 == 0:
                ipd_f_mod = ipd_fb * ipd_magnification_factor
                ipd_b_mod = ipd_bb / ipd_magnification_factor
            else:
                ipd_f_mod = ipd_fb / ipd_magnification_factor
                ipd_b_mod = ipd_bb * ipd_magnification_factor

            d_f = np.exp(0.5j * (ipd_f_mod - ipd_fb))
            d_b = np.exp(0.5j * (ipd_b_mod - ipd_bb))
            H_L_mod[front_idx, band_mask_full] = H_L_fb * d_f
            H_R_mod[front_idx, band_mask_full] = H_R_fb * np.conj(d_f)
            H_L_mod[back_idx, band_mask_full] = H_L_bb * d_b
            H_R_mod[back_idx, band_mask_full] = H_R_bb * np.conj(d_b)

    return H_L_mod, H_R_mod


def _srm_front_back_masks(
    symm_pairs: list[tuple[int, int]],
    n_azimuths: int,
    device: torch.device | str,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Primary = front-back symmetric (target, masker) pairs only, e.g. (20°, 160°), (40°, 140°).
    Floor = all other pairs (for optional penalty).
    Returns (primary_mask, floor_mask), each (n_azimuths, n_azimuths).
    """
    primary = np.zeros((n_azimuths, n_azimuths), dtype=bool)
    for fi, bi in symm_pairs:
        primary[fi, bi] = True
        primary[bi, fi] = True
    primary_t = torch.from_numpy(primary).to(device)
    return primary_t, ~primary_t


def find_symmetrical_azimuth_pairs(az_deg: np.ndarray) -> list[tuple[int, int]]:
    """Front-back symmetric pairs: front az ↔ back 180° - az (within 5°). Returns (front_idx, back_idx)."""
    az_n = ((az_deg + 180) % 360) - 180
    pairs = []
    seen = set()
    for i, az_f in enumerate(az_n):
        az_b = ((180 - az_f + 180) % 360) - 180
        diff = np.abs(az_n - az_b)
        j = np.argmin(diff)
        if diff[j] < 5.0 and i != j:
            key = (min(i, j), max(i, j))
            if key not in seen:
                seen.add(key)
                pairs.append((i, j))
    return pairs


def compute_front_back_spectral_difference(
    H_L: np.ndarray,
    H_R: np.ndarray,
    front_idx: int,
    back_idx: int,
    freqs_hz: np.ndarray,
) -> float:
    """RMS spectral difference in dB between front and back locations."""
    mag_f = (np.abs(H_L[front_idx]) + np.abs(H_R[front_idx])) / 2
    mag_b = (np.abs(H_L[back_idx]) + np.abs(H_R[back_idx])) / 2
    diff_db = 20 * np.log10((mag_f + 1e-10) / (mag_b + 1e-10))
    return float(np.sqrt(np.mean(diff_db**2)))


def reconstruct_sh_from_hrtfs(
    H_L: np.ndarray,
    H_R: np.ndarray,
    az_deg: np.ndarray,
    el: float,
    N_sph: int,
) -> np.ndarray:
    """Reconstruct SH coefficients from HRTFs via least squares. Returns [ears, coeffs, freqs]."""
    n_azimuths, n_freqs = H_L.shape
    n_coeffs = (N_sph + 1) ** 2
    az_rad = np.deg2rad(az_deg)
    zen = np.pi / 2.0 - np.deg2rad(np.full_like(az_deg, el))
    Y = sh_matrix(N_sph, az_rad, zen, sh_type="real")
    H = np.stack([H_L, H_R], axis=1)
    Y_pinv = np.linalg.pinv(Y)
    sh_coeffs_freq = np.zeros((2, n_coeffs, n_freqs), dtype=complex)
    for f in range(n_freqs):
        sh_coeffs_freq[:, :, f] = (Y_pinv @ H[:, :, f]).T
    return sh_coeffs_freq


def reconstruct_hrtfs_from_sh_on_grid(
    sh_coeffs_freq: np.ndarray,
    az_min: float,
    az_max: float,
    az_step: float,
    el: float,
    N_sph: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Reconstruct HRTFs from SH coefficients on a regular azimuth grid. Returns H_L, H_R, az_deg."""
    if abs(az_max - 180.0) < 1e-6:
        az_deg = np.arange(az_min, az_max + az_step, az_step)
        az_deg = az_deg[az_deg < 180.0]
    else:
        az_deg = np.arange(az_min, az_max + az_step, az_step)
    az_deg = np.mod(az_deg + 180.0, 360.0) - 180.0
    az_rad = np.deg2rad(az_deg)
    zen = np.pi / 2.0 - np.deg2rad(np.full_like(az_deg, el))
    Y = sh_matrix(N_sph, az_rad, zen, sh_type="real")
    H = np.einsum("dc,ecf->def", Y, sh_coeffs_freq)
    return H[:, 0, :], H[:, 1, :], az_deg


def save_hrtfs_to_sofa(
    H_L: np.ndarray,
    H_R: np.ndarray,
    az_deg: np.ndarray,
    el_deg: np.ndarray,
    fs: float,
    output_path: str,
    reference_sofa: str | None = None,
) -> None:
    """Save HRTFs (frequency domain) to SOFA as HRIRs. Optionally copy metadata from reference_sofa."""
    n_azimuths = len(az_deg)
    n_freqs = H_L.shape[1]
    ir_length = (n_freqs - 1) * 2

    print(f"Converting to time domain (IR length: {ir_length})...")
    hrir = np.stack(
        [
            np.fft.irfft(H_L, n=ir_length, axis=1),
            np.fft.irfft(H_R, n=ir_length, axis=1),
        ],
        axis=1,
    )
    source_positions = np.column_stack([az_deg, el_deg, np.full(n_azimuths, 1.0)])

    print(f"Saving SOFA to {output_path}...")
    with h5py.File(output_path, "w") as f:
        # Create required groups
        f.create_group("Data")
        f.create_group("Dimensions")
        f.create_group("Variables")

        # Write HRIR data (use Data/IR format - the script checks both)
        f["Data"].create_dataset("IR", data=hrir.astype(np.float32))

        # Write sampling rate (at root level with dot notation for compatibility)
        f.create_dataset("Data.SamplingRate", data=np.array([fs], dtype=np.float64))

        f["Dimensions"].create_dataset("M", data=np.array([n_azimuths], dtype=np.int32))
        f["Dimensions"].create_dataset("R", data=np.array([2], dtype=np.int32))
        f["Dimensions"].create_dataset("N", data=np.array([ir_length], dtype=np.int32))

        # Write source positions
        f.create_dataset("SourcePosition", data=source_positions.astype(np.float64))

        # Write SOFA conventions and attributes
        f.attrs["Conventions"] = "SOFA"
        f.attrs["Version"] = "1.0"
        f.attrs["SOFAConventions"] = "SimpleFreeFieldHRIR"
        f.attrs["DataType"] = "FIR"
        f.attrs["RoomType"] = "free field"
        f.attrs["ListenerShortName"] = "KEMAR"
        f.attrs["ListenerDescription"] = "KEMAR with optimized HRTFs for SRM"
        f.attrs["SourceDescription"] = "Optimized HRTFs via PCA and gradient descent"

        # Copy additional metadata from reference SOFA if provided
        if reference_sofa is not None:
            try:
                with h5py.File(reference_sofa, "r") as ref:
                    # Copy common attributes
                    for key in ["ListenerDescription", "ListenerShortName"]:
                        if key in ref.attrs and key not in f.attrs:
                            f.attrs[key] = ref.attrs[key]
                    # Copy receiver positions if available
                    if "ReceiverPosition" in ref:
                        f.create_dataset(
                            "ReceiverPosition", data=ref["ReceiverPosition"][...]
                        )
            except Exception as e:
                print(f"Warning: Could not copy metadata from reference SOFA: {e}")

    print(f"Successfully saved SOFA file with {n_azimuths} measurements")


def main():
    """Run HRTF load → PCA → magnitude optimization → optional IPD magnification → SH reconstruction → save."""
    # Config
    enable_magnitude_optimization = True
    enable_phase_magnification = False
    sofa_file = INPUT_SOFA
    az_min, az_max, n_azimuths = -180.0, 180.0, 36
    el, N_sph = 0.0, 32
    n_components = 20
    n_components_optimize, n_steps, lr = 3, 300, 1e-2
    lambda_ild, lambda_rms = 0.1, 1.0
    lambda_ild_preserve = 1.0
    lambda_srm_penalty = 1.0  # penalty for negative SRM change on non-front-back pairs
    # Prefer CUDA (NVIDIA) > MPS (Apple Metal) > CPU
    if torch.cuda.is_available():
        device = "cuda"
    elif (
        getattr(torch.backends, "mps", None) is not None
        and torch.backends.mps.is_available()
    ):
        device = "mps"
    else:
        device = "cpu"
    az_step = 10.0
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_sofa = OUTPUT_DIR / "optimized_hrtf_new.sofa"
    output_npz = OUTPUT_DIR / "optimized_hrtf_new.npz"

    print(
        f"Device: {device}  Magnitude opt: {enable_magnitude_optimization}  Phase mag: {enable_phase_magnification}"
    )

    H_L, H_R, az_deg, freqs_hz, fs = load_hrtfs_for_azimuth_range(
        sofa_file, az_min, az_max, n_azimuths, el, N_sph
    )
    pca, components, explained_variance, transformed = apply_spatial_pca_to_magnitude(
        H_L, H_R, n_components=n_components
    )
    n_dirs = len(az_deg)
    target_dirs = masker_dirs = np.arange(n_dirs)

    if enable_magnitude_optimization:
        print("\n=== Magnitude optimization ===")
        H_L_mag, H_R_mag, results = optimize_hrtf_with_gradient_descent(
            H_L,
            H_R,
            pca,
            components,
            transformed,
            az_deg,
            freqs_hz,
            fs,
            n_components_optimize=n_components_optimize,
            n_steps=n_steps,
            lr=lr,
            lambda_ild=lambda_ild,
            lambda_rms=lambda_rms,
            lambda_ild_preserve=lambda_ild_preserve,
            lambda_srm_penalty=lambda_srm_penalty,
            target_dirs=target_dirs,
            masker_dirs=masker_dirs,
            device=device,
        )
    else:
        H_L_mag, H_R_mag = H_L.copy(), H_R.copy()
        results = {"srm_baseline": None, "srm_optimized": None, "history": []}

    if enable_phase_magnification:
        print("\n=== Phase IPD magnification ===")
        symm_pairs = find_symmetrical_azimuth_pairs(az_deg)
        H_L_final, H_R_final = magnify_ipd_between_symmetrical_pairs(
            H_L_mag,
            H_R_mag,
            freqs_hz,
            az_deg,
            symm_pairs,
            f_max=1250.0,
            ipd_magnification_factor=1.5,
        )
    else:
        H_L_final, H_R_final = H_L_mag.copy(), H_R_mag.copy()

    # Reconstruct on dense grid and save SOFA
    print("\n=== SH reconstruction (dense grid) ===")
    sh_coeffs_opt = reconstruct_sh_from_hrtfs(H_L_final, H_R_final, az_deg, el, N_sph)
    H_L_dense, H_R_dense, az_deg_dense = reconstruct_hrtfs_from_sh_on_grid(
        sh_coeffs_opt,
        az_min,
        az_max,
        az_step,
        el,
        N_sph,
    )
    el_deg_dense = np.full_like(az_deg_dense, el)
    save_hrtfs_to_sofa(
        H_L_dense,
        H_R_dense,
        az_deg_dense,
        el_deg_dense,
        fs,
        output_sofa,
        reference_sofa=sofa_file,
    )
    print(f"Saved {len(az_deg_dense)} HRTFs to {output_sofa}")

    np.savez(
        output_npz,
        H_L_original=H_L,
        H_R_original=H_R,
        H_L_magnified=H_L_mag,
        H_R_magnified=H_R_mag,
        H_L_final=H_L_final,
        H_R_final=H_R_final,
        az_deg=az_deg,
        freqs_hz=freqs_hz,
        fs=fs,
        components=components,
        explained_variance=explained_variance,
        transformed=transformed,
        pca_mean=pca.mean_,
        srm_baseline=results.get("srm_baseline"),
        srm_optimized=results.get("srm_optimized"),
        n_components_optimized=n_components_optimize,
        history=results.get("history"),
    )
    print(f"Saved results to {output_npz}")

    print("\nSummary:")
    print(
        f"  Azimuths {az_min}°–{az_max}°, top component variance {explained_variance[0]*100:.1f}%"
    )
    if (
        enable_magnitude_optimization
        and results.get("srm_baseline") is not None
        and results.get("srm_optimized") is not None
    ):
        print(
            f"  SRM: {results['srm_baseline']:.2f} → {results['srm_optimized']:.2f} dB"
        )


if __name__ == "__main__":
    main()
