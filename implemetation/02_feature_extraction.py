"""
Feature Extraction for EEG Attention Analysis
----------------------------------------------
Computes a comprehensive set of features for attention detection.

Feature Categories:
1. Time Domain: RMS, Skewness, Kurtosis, ZCR, Hjorth Parameters, Line Length
2. Frequency Domain: 
   - Absolute/Relative Power (Delta, Theta, Alpha, Beta, Gamma)
   - Ratios (TBR, TAR, GAR, Engagement, Relaxation, Fatigue, Sleepiness)
   - Spectral Markers (Centroid, SEF95, Shannon/Renyi/Tsallis Entropy, Slope, PAF, MDF)
3. Complexity: HFD, SampEn, LZC, KFD, PeEn, Petrosian FD
4. Energy: TKEO (Broadband, Theta, Alpha, Beta)
5. Spatial: Channel-group aggregates (Frontal, Parietal, Occipital), Asymmetry (L/R)
6. Connectivity: Within-band coherence averages

Optimized for storage (Parquet) and speed (Numba/Joblib).
"""

import os
import glob
import re
import warnings
import math

import numpy as np
import pandas as pd
import scipy.stats
from scipy.integrate import trapezoid
from scipy.signal import welch, butter, sosfiltfilt, detrend

import mne
from joblib import Parallel, delayed

# --- RIEMANNIAN FEATURES (pyriemann) ---
try:
    from pyriemann.estimation import Covariances
    from pyriemann.tangentspace import TangentSpace
    HAS_PYRIEMANN = True
except ImportError:
    HAS_PYRIEMANN = False
    print("Warning: pyriemann not found. Riemannian features will be skipped.")

# --- OPTIMIZATION: Numba JIT ---
try:
    from numba import jit
    HAS_NUMBA = True
except ImportError:
    HAS_NUMBA = False
    print("Warning: Numba not found. Complexity features will be slower.")
    # Dummy decorator if numba is missing
    def jit(nopython=True):
        def decorator(func):
            return func
        return decorator

# ---------- CONFIG ----------
try:
    from importlib import import_module
    config = import_module('00_config')
    BANDS = config.BANDS
    FRONTAL_CHANNELS = config.FRONTAL_CHANNELS
    PARIETAL_CHANNELS = config.PARIETAL_CHANNELS
    TEMPORAL_CHANNELS = config.TEMPORAL_CHANNELS
    OCCIPITAL_CHANNELS = config.OCCIPITAL_CHANNELS
    LEFT_CHANNELS = config.LEFT_CHANNELS
    RIGHT_CHANNELS = config.RIGHT_CHANNELS
    
    # Paths
    PREPROCESSED_DIR = getattr(config, 'PREPROCESSED_DIR', 'Preprocessed')
    FEATURES_DIR = getattr(config, 'FEATURES_DIR', 'Features')
    FINAL_DATA_FOLDER = getattr(config, 'FINAL_DATA_FOLDER', os.path.join(PREPROCESSED_DIR, 'FinalData'))
    OUT_FOLDER = getattr(config, 'FEATURE_OUT_FOLDER', FEATURES_DIR)
    OUT_FILE = getattr(config, 'FEATURE_OUT_FILE', 'features_enhanced.parquet')
    if not os.path.isabs(OUT_FILE):
        OUT_FILE = os.path.join(OUT_FOLDER, OUT_FILE)
    RIEMANN_COV_OUT = getattr(config, 'RIEMANN_COV_OUT', os.path.join(FEATURES_DIR, 'covariances.npz'))
except Exception:
    # Fallback if config not available
    BANDS = {
        'delta': (0.5, 4.0),
        'theta': (4.0, 8.0),
        'alpha': (8.0, 13.0),
        'beta': (13.0, 30.0),
        'gamma': (30.0, 45.0)
    }
    FRONTAL_CHANNELS = ['AF3', 'F7', 'F3', 'FC5', 'F4', 'F8', 'AF4', 'FC6']
    PARIETAL_CHANNELS = ['P7', 'P8']
    TEMPORAL_CHANNELS = ['T7', 'T8']
    OCCIPITAL_CHANNELS = ['O1', 'O2']
    LEFT_CHANNELS = ['AF3', 'F7', 'F3', 'FC5', 'T7', 'P7', 'O1']
    RIGHT_CHANNELS = ['AF4', 'F8', 'F4', 'FC6', 'T8', 'P8', 'O2']

# Default Paths (Fallback)
if 'FINAL_DATA_FOLDER' not in globals():
    PREPROCESSED_DIR = 'Preprocessed'
    FEATURES_DIR = 'Features'
    FINAL_DATA_FOLDER = os.path.join(PREPROCESSED_DIR, "FinalData")
    OUT_FOLDER = FEATURES_DIR
    OUT_FILE = os.path.join(OUT_FOLDER, "features_enhanced.parquet")
    RIEMANN_COV_OUT = os.path.join(FEATURES_DIR, 'covariances.npz')

# Regex to parse filenames like "audio12_attention_final_epo.fif"
FNAME_PATTERN = re.compile(
    r'(?P<task>[a-zA-Z]+)(?P<subj>\d+)(?:_)(?P<cond>attention|relaxed)',
    flags=re.IGNORECASE
)

# ---------------- HELPERS ----------------

def ensure_dir(p):
    os.makedirs(p, exist_ok=True)

def safe_div(a, b):
    """Division with zero protection"""
    return a / b if b > 0 else 0.0

# ---------------- NUMBA OPTIMIZED FUNCTIONS ----------------

@jit(nopython=True)
def _higuchi_fd(x, kmax):
    """Higuchi Fractal Dimension"""
    n_times = x.size
    lk = np.zeros(kmax)
    x_reg = np.zeros(kmax)
    y_reg = np.zeros(kmax)

    for k in range(1, kmax + 1):
        lm = np.zeros((k,))
        for m in range(k):
            ll = 0.0
            n_max = int(np.floor((n_times - m - 1) / k))
            for j in range(1, n_max + 1):
                ll += abs(x[m + j * k] - x[m + (j - 1) * k])
            norm = (n_times - 1) / (n_max * k)
            lm[m] = (ll * norm) / k
        lk[k - 1] = np.mean(lm)
        x_reg[k - 1] = np.log(1.0 / k)
        y_reg[k - 1] = np.log(lk[k - 1])

    # Linear regression slope
    x_mean = np.mean(x_reg)
    y_mean = np.mean(y_reg)
    numerator = np.sum((x_reg - x_mean) * (y_reg - y_mean))
    denominator = np.sum((x_reg - x_mean) ** 2)
    
    if denominator == 0: return np.nan
    return numerator / denominator

@jit(nopython=True)
def _petrosian_fd(x):
    """Petrosian Fractal Dimension"""
    n = len(x)
    diff = np.diff(x)
    n_delta = 0
    for i in range(len(diff) - 1):
        if diff[i] * diff[i+1] < 0:
            n_delta += 1
    if n_delta == 0: return 1.0
    return np.log10(n) / (np.log10(n) + np.log10(n / (n + 0.4 * n_delta)))

@jit(nopython=True)
def _katz_fd(x):
    """Katz Fractal Dimension"""
    n = len(x)
    if n <= 0: return np.nan
    dists = np.abs(np.diff(x))
    L = np.sum(dists)
    d = np.max(np.abs(x[1:] - x[0]))
    if L == 0 or d == 0: return np.nan
    return np.log10(n) / (np.log10(d / L) + np.log10(n))

@jit(nopython=True)
def _lz_complexity_simple(bin_seq):
    """Simplified Lempel-Ziv Complexity (Numba friendly)"""
    n = len(bin_seq)
    c = 1
    l = 1
    i = 0
    k = 1
    k_max = 1
    while True:
        if c + k > n: break
        pattern_found = False
        for j in range(i):
            match = True
            for m in range(k):
                if bin_seq[j+m] != bin_seq[i+m]:
                    match = False
                    break
            if match:
                pattern_found = True
                break
        if pattern_found:
            k += 1
        else:
            c += 1
            i += k
            k = 1
        if i + k > n: break
    if n > 1:
        return (c * np.log2(n)) / n
    return np.nan

@jit(nopython=True)
def _permutation_entropy(x, order=3, delay=1):
    """Permutation Entropy (Order 3 specific optimization)"""
    n = len(x)
    n_out = n - (order - 1) * delay
    if n_out <= 0: return 0.0
    counts = np.zeros(6) # 3! = 6 permutations
    for i in range(n_out):
        v0 = x[i]
        v1 = x[i+delay]
        v2 = x[i+2*delay]
        if v0 < v1:
            if v1 < v2: idx = 0   # 012
            elif v0 < v2: idx = 1 # 021
            else: idx = 4         # 201
        else: 
            if v1 > v2: idx = 5   # 210
            elif v0 < v2: idx = 2 # 102
            else: idx = 3         # 120
        counts[idx] += 1
    probs = counts / n_out
    pe = 0.0
    for p in probs:
        if p > 0:
            pe -= p * np.log2(p)
    return -pe / np.log2(6.0) # Normalized

@jit(nopython=True)
def _sample_entropy(x, m, r):
    """Sample Entropy"""
    n = len(x)
    
    def _phi(k):
        c_i = 0
        for i in range(n - k):
            local_count = 0
            for j in range(n - k):
                if i == j: continue
                # Max dist (Chebyshev)
                dist = 0.0
                for l in range(k):
                    d = abs(x[i+l] - x[j+l])
                    if d > dist: dist = d
                if dist < r:
                    local_count += 1
            c_i += local_count
        return c_i

    A = _phi(m + 1)
    B = _phi(m)
    if B == 0: return 0.0
    return -np.log(A / B)

@jit(nopython=True)
def _renyi_entropy(p_norm, alpha=2):
    """Renyi Entropy (Generalization of Shannon)"""
    s = 0.0
    for val in p_norm:
        if val > 0:
            s += val ** alpha
    if s == 0: return 0.0
    if alpha == 1: return 0.0 # Should be Shannon, but avoiding div/0
    return (1.0 / (1.0 - alpha)) * np.log2(s)

@jit(nopython=True)
def _tsallis_entropy(p_norm, q=2):
    """Tsallis Entropy (Non-extensive systems)"""
    s = 0.0
    for val in p_norm:
        if val > 0:
            s += val ** q
    if q == 1: return 0.0 
    return (1.0 / (q - 1.0)) * (1.0 - s)


# ---------------- SPECTRAL HELPERS ----------------

def compute_spectral_entropy(psd):
    """Shannon entropy of the power spectrum (normalized)"""
    total = np.sum(psd)
    if total <= 0:
        return 0.0
    psd_norm = psd / total
    se = 0.0
    for p in psd_norm:
        if p > 0:
            se -= p * np.log2(p)
    return se / np.log2(len(psd)) if len(psd) > 1 else 0.0

def compute_spectral_slope(freqs, psd):
    """1/f Slope via linear regression on log-log"""
    valid_idx = (freqs >= 1.0) & (freqs <= 45.0) & (psd > 0)
    if np.sum(valid_idx) < 5: return 0.0
    log_f = np.log10(freqs[valid_idx])
    log_p = np.log10(psd[valid_idx])
    slope, _, _, _, _ = scipy.stats.linregress(log_f, log_p)
    return slope

# ---------------- MAIN EXTRACTION LOGIC ----------------

def compute_features_for_epoch(epoch_data, sfreq):
    x = epoch_data.astype(np.float64)
    
    # --- TIME DOMAIN ---
    feats = {
        'rms': np.sqrt(np.mean(x**2)),
        'skew': scipy.stats.skew(x),
        'kurt': scipy.stats.kurtosis(x),
        'line_length': np.sum(np.abs(np.diff(x)))
    }
    
    # Zero Crossing Rate (Center first)
    x_centered = x - np.mean(x)
    feats['zcr'] = np.mean(np.diff(np.signbit(x_centered)))
    
    # Hjorth
    var_x = np.var(x)
    dx = np.diff(x)
    var_dx = np.var(dx)
    ddx = np.diff(dx)
    var_ddx = np.var(ddx)
    
    feats['hjorth_activity'] = var_x
    feats['hjorth_mobility'] = np.sqrt(var_dx / var_x) if var_x > 0 else 0
    if var_dx > 0 and feats['hjorth_mobility'] > 0:
        feats['hjorth_complexity'] = np.sqrt(var_ddx / var_dx) / feats['hjorth_mobility']
    else:
        feats['hjorth_complexity'] = 0
        
    # --- FREQUENCY DOMAIN ---
    n_per_seg = max(128, min(512, len(x))) if len(x) > 128 else len(x)
    freqs, psd = welch(x, fs=sfreq, nperseg=n_per_seg)
    total_power = trapezoid(psd, freqs)
    
    # Normalize PSD for Entropies
    if total_power > 0:
        psd_norm = psd / np.sum(psd)
    else:
        psd_norm = np.zeros_like(psd)

    # 1. Entropies
    feats['spec_entropy'] = compute_spectral_entropy(psd)
    
    # NEW: Advanced Entropies for Fatigue/System State
    if HAS_NUMBA:
        feats['spec_renyi'] = _renyi_entropy(psd_norm, alpha=2)
        feats['spec_tsallis'] = _tsallis_entropy(psd_norm, q=2)
    else:
        feats['spec_renyi'] = 0.0
        feats['spec_tsallis'] = 0.0

    feats['spec_slope'] = compute_spectral_slope(freqs, psd)
    
    # Band Powers
    band_powers = {}
    for band, (fmin, fmax) in BANDS.items():
        idx = (freqs >= fmin) & (freqs <= fmax)
        bp = trapezoid(psd[idx], freqs[idx]) if idx.any() else 0.0
        band_powers[band] = bp
        feats[f'{band}_power_abs'] = bp
        feats[f'{band}_power_rel'] = bp / total_power if total_power > 0 else 0.0
        
    # 2. Ratios
    feats['ratio_theta_beta'] = safe_div(band_powers['theta'], band_powers['beta'])
    feats['ratio_theta_alpha'] = safe_div(band_powers['theta'], band_powers['alpha'])
    feats['ratio_gamma_alpha'] = safe_div(band_powers['gamma'], band_powers['alpha'])
    
    # Engagement Index = Beta / (Theta + Alpha)
    feats['engagement_idx'] = safe_div(band_powers['beta'], (band_powers['theta'] + band_powers['alpha']))
    
    # Relaxation Index = Theta / Delta 
    feats['relaxation_idx'] = safe_div(band_powers['theta'], band_powers['delta'])

    # NEW: Inverse Ratios for Linear Separation of Fatigue
    feats['fatigue_idx'] = safe_div((band_powers['theta'] + band_powers['alpha']), band_powers['beta'])
    feats['sleepiness_idx'] = safe_div(band_powers['delta'], band_powers['theta'])
    
    # Spectral Markers
    feats['spec_centroid'] = safe_div(np.sum(freqs * psd), np.sum(psd))
    
    # SEF95 & Median Freq
    cumsum = np.cumsum(psd)
    norm_cumsum = safe_div(cumsum, cumsum[-1])
    
    idx95 = np.searchsorted(norm_cumsum, 0.95)
    feats['sef95'] = freqs[min(idx95, len(freqs)-1)]
    
    idx50 = np.searchsorted(norm_cumsum, 0.50)
    feats['median_freq'] = freqs[min(idx50, len(freqs)-1)]
    
    # Peak Alpha Frequency (PAF)
    idx_alpha = (freqs >= 8.0) & (freqs <= 13.0)
    if np.any(idx_alpha):
        alpha_freqs = freqs[idx_alpha]
        alpha_psd = psd[idx_alpha]
        peak_idx = np.argmax(alpha_psd)
        feats['peak_alpha_freq'] = alpha_freqs[peak_idx]
    else:
        feats['peak_alpha_freq'] = 0.0
    
    # --- NONLINEAR / COMPLEXITY ---
    if HAS_NUMBA:
        feats['hfd'] = _higuchi_fd(x, kmax=10)
        feats['katz'] = _katz_fd(x)
        feats['petrosian'] = _petrosian_fd(x)
        feats['pe_en'] = _permutation_entropy(x, order=3, delay=1)
        
        # Sample Entropy (m=2, r=0.2*std)
        r = 0.2 * np.std(x)
        feats['samp_entropy'] = _sample_entropy(x, m=2, r=r)
        
        # LZC (Binarized)
        thresh = np.mean(x)
        x_bin = (x > thresh).astype(np.int8)
        feats['lzc'] = _lz_complexity_simple(x_bin)
    else:
        # Fallback placeholders if numba missing
        feats['hfd'] = 0.0
        feats['katz'] = 0.0
        feats['petrosian'] = 0.0
        feats['pe_en'] = 0.0
        feats['samp_entropy'] = 0.0
        feats['lzc'] = 0.0
    
    # --- ENERGY (TKEO) ---
    # 1. Broadband TKEO
    tkeo = x[1:-1]**2 - x[:-2] * x[2:]
    feats['tkeo_mean'] = np.mean(tkeo)
    
    # 2. Band-Specific TKEO (Instability Markers)
    
    # Theta (4-8Hz)
    sos_t = butter(4, [4, 8], btype='bandpass', fs=sfreq, output='sos')
    x_t = sosfiltfilt(sos_t, x)
    tkeo_t = x_t[1:-1]**2 - x_t[:-2] * x_t[2:]
    feats['tkeo_theta_mean'] = np.mean(tkeo_t)

    # NEW: Alpha (8-13Hz) - Critical for Alpha Bursts/Mind Wandering
    sos_a = butter(4, [8, 13], btype='bandpass', fs=sfreq, output='sos')
    x_a = sosfiltfilt(sos_a, x)
    tkeo_a = x_a[1:-1]**2 - x_a[:-2] * x_a[2:]
    feats['tkeo_alpha_mean'] = np.mean(tkeo_a)

    # NEW: Beta (13-30Hz) - Attention Maintenance
    sos_b = butter(4, [13, 30], btype='bandpass', fs=sfreq, output='sos')
    x_b = sosfiltfilt(sos_b, x)
    tkeo_b = x_b[1:-1]**2 - x_b[:-2] * x_b[2:]
    feats['tkeo_beta_mean'] = np.mean(tkeo_b)
    
    return feats

def compute_spatial_features(epoch_data, ch_names, sfreq):
    """
    Compute epoch-level spatial features:
    - Channel group averages (frontal theta, parietal alpha, etc.)
    - Left/Right asymmetry for each band
    
    Args:
        epoch_data: (n_channels, n_samples) array for one epoch
        ch_names: list of channel names
        sfreq: sampling frequency
    
    Returns:
        dict of spatial features
    """
    ch_idx = {ch: i for i, ch in enumerate(ch_names)}
    spatial_feats = {}
    
    def get_band_power(ch, band):
        """Get band power for a single channel"""
        if ch not in ch_idx:
            return None
        sig = epoch_data[ch_idx[ch], :]
        n_per_seg = max(128, min(512, len(sig))) if len(sig) > 128 else len(sig)
        freqs, psd = welch(sig, fs=sfreq, nperseg=n_per_seg)
        fmin, fmax = BANDS[band]
        idx = (freqs >= fmin) & (freqs <= fmax)
        return trapezoid(psd[idx], freqs[idx]) if idx.any() else 0.0
    
    def group_band_mean(channels, band):
        """Average band power across a channel group"""
        vals = [get_band_power(ch, band) for ch in channels if ch in ch_idx]
        vals = [v for v in vals if v is not None]
        return np.mean(vals) if vals else 0.0
    
    # Channel group averages for key bands
    spatial_feats['frontal_theta'] = group_band_mean(FRONTAL_CHANNELS, 'theta')
    spatial_feats['frontal_beta'] = group_band_mean(FRONTAL_CHANNELS, 'beta')
    spatial_feats['parietal_alpha'] = group_band_mean(PARIETAL_CHANNELS, 'alpha')
    spatial_feats['occipital_alpha'] = group_band_mean(OCCIPITAL_CHANNELS, 'alpha')
    spatial_feats['temporal_beta'] = group_band_mean(TEMPORAL_CHANNELS, 'beta')
    
    # Hemispheric asymmetry (log ratio: log(right) - log(left))
    for band in ['theta', 'alpha', 'beta']:
        left_power = group_band_mean(LEFT_CHANNELS, band)
        right_power = group_band_mean(RIGHT_CHANNELS, band)
        if left_power > 0 and right_power > 0:
            spatial_feats[f'asymmetry_{band}'] = np.log(right_power) - np.log(left_power)
        else:
            spatial_feats[f'asymmetry_{band}'] = np.nan  # Mark as missing data
    
    # Frontal/Parietal ratio (attention network marker)
    fp_alpha_left = group_band_mean(['F3', 'F7'], 'alpha')
    fp_alpha_right = group_band_mean(['F4', 'F8'], 'alpha')
    parietal_alpha = group_band_mean(PARIETAL_CHANNELS, 'alpha')
    frontal_alpha = (fp_alpha_left + fp_alpha_right) / 2 if (fp_alpha_left + fp_alpha_right) > 0 else 0.0
    spatial_feats['frontal_parietal_alpha_ratio'] = safe_div(frontal_alpha, parietal_alpha)
    
    return spatial_feats


def compute_riemannian_features(epoch_data_3d):
    """
    Compute Riemannian covariance features for all epochs.
    
    Per Final_implementation.md Section 5.2:
    - Save raw covariance matrices (SPD) for later fold-wise TangentSpace mapping
    - Do NOT perform TangentSpace().fit_transform() globally here (would cause leakage)
    
    Args:
        epoch_data_3d: (n_epochs, n_channels, n_samples) array
        
    Returns:
        tuple: (riem_dict, covmats)
            riem_dict: dict mapping epoch_idx to dict of cov_* features
            covmats: (n_epochs, n_channels, n_channels) raw SPD covariance matrices
    """
    if not HAS_PYRIEMANN:
        return {}, None
    
    n_epochs, n_channels, _ = epoch_data_3d.shape
    
    try:
        # Scale to microvolts (uV) for numerical stability and to match regularization scale
        # MNE data is in Volts (~1e-6), variances ~1e-12. 
        # Tikhonov factor 1e-5 is too large for Volts, but appropriate for uV (Variance ~100).
        epoch_data_3d_uv = epoch_data_3d * 1e6
        
        # attempt pyriemann OAS estimation first
        cov_estimator = Covariances(estimator='oas')
        covmats = cov_estimator.fit_transform(epoch_data_3d_uv)
        
    except Exception as e:
        print(f"[WARN] pyriemann Covariances failed: {e}. Falling back to LedoitWolf.")
        try:
            from sklearn.covariance import LedoitWolf
            lw = LedoitWolf(store_precision=False)
            covmats = np.stack([lw.fit(epoch_data_3d_uv[i].T).covariance_ for i in range(n_epochs)], axis=0)
        except Exception as e2:
            print(f"[ERROR] fallback covariance failed: {e2}")
            return {}, None

    # adaptive regularization and quality checks
    covmats_out = np.empty_like(covmats)
    
    for ei in range(n_epochs):
        C = covmats[ei]
        trace = np.trace(C)
        mean_var = np.mean(np.var(epoch_data_3d_uv[ei], axis=1))
        
        # Filter: Skip epochs that are effectively zero/flat
        if mean_var < 1e-14 or trace <= 0 or np.isnan(trace):
            print(f"[SKIP] Epoch {ei} low mean variance {mean_var:.2e} trace {trace:.2e}")
            covmats_out[ei] = np.eye(n_channels) * 1e-12 # Identity placeholder
            continue
            
        # Adaptive Tikhonov Regularization
        # Reg term proportional to the trace (average energy)
        # Floor at 1e-8 for numerical safety
        reg_coef = max(1e-8, 1e-3 * (trace / float(n_channels)))
        C_reg = C + reg_coef * np.eye(n_channels)
        
        # Ensure symmetry
        C_reg = (C_reg + C_reg.T) / 2.0
        
        # Diagnostic Check
        eigs = np.linalg.eigvalsh(C_reg)
        if eigs.min() < 1e-12:
            print(f"[WARN] epoch {ei} tiny eig {eigs.min():.2e} after reg (reg={reg_coef:.2e}) mean_var={mean_var:.2e}")
            
        covmats_out[ei] = C_reg

    covmats = covmats_out
    
    # Log Summary (Global)
    all_eigvals = []
    # Sample a subset to report
    sample_indices = np.linspace(0, n_epochs-1, min(20, n_epochs)).astype(int)
    for ei in sample_indices:
        eigs = np.linalg.eigvalsh(covmats[ei])
        all_eigvals.extend(eigs)
        
    if all_eigvals:
        print(f"  Sampled Eigenvalues (n={len(all_eigvals)}): Min={np.min(all_eigvals):.2e}, Median={np.median(all_eigvals):.2e}, Max={np.max(all_eigvals):.2e}")

    # Build dict for each epoch with raw covariance features
    riem_dict = {}
    for ei in range(n_epochs):
        feats = {}
        idx = 0
        for i in range(n_channels):
            for j in range(i, n_channels):
                # Save just the values
                feats[f'riem_cov_{i:02d}_{j:02d}'] = float(covmats[ei, i, j])
                idx += 1
        riem_dict[ei] = feats
    
    return riem_dict, covmats

def process_file_optimized(fpath):
    """Process a single epoch file and extract all features"""
    fname = os.path.basename(fpath)
    
    # Parse metadata from filename
    m = FNAME_PATTERN.search(fname)
    if not m:
        try:
            parts = fname.split('_')
            task_raw = parts[0]
            cond = parts[1]
            m_task = re.match(r'([a-zA-Z]+)', task_raw)
            m_subj = re.search(r'(\d+)', task_raw)
            task = m_task.group(1).lower() if m_task else 'unknown'
            if m_subj:
                subject = int(m_subj.group(1))
            else:
                assert False, f"Invalid subject ID parsed from {fname}"
            condition = cond.lower()
        except:
            return pd.DataFrame()
    else:
        task = m.group("task").lower()
        subject = int(m.group("subj"))
        condition = m.group("cond").lower()
        
    try:
        epochs = mne.read_epochs(fpath, preload=True, verbose=False)
    except Exception as e:
        print(f"Error reading {fname}: {e}")
        return pd.DataFrame()
        
    data = epochs.get_data(copy=False)
    sfreq = epochs.info['sfreq']
    ch_names = epochs.ch_names
    
    rows = []
    n_epochs, n_ch, _ = data.shape
    
    # Provenance Helper
    def save_manifest(out_dir, config_module=None):
        import datetime
        import json
        manifest = {
            'created_at': datetime.datetime.now().isoformat(),
            'files_processed': len(glob.glob(os.path.join(FINAL_DATA_FOLDER, "*_final_epo.fif"))),
            'config': {k: v for k, v in vars(config_module).items() if k.isupper()} if config_module else {},
            # 'git_sha': ... (would require gitpython or subprocess, optional)
        }
        with open(os.path.join(out_dir, 'manifest.json'), 'w') as f:
            json.dump(manifest, f, indent=2)
            
    # Save manifest once (checking if main) - actually better to do in main()

    for ei in range(n_epochs):
        epoch_data = data[ei, :, :]
        
        # Compute per-channel features
        for ci in range(n_ch):
            sig = epoch_data[ci, :]
            feats = compute_features_for_epoch(sig, sfreq)
            feats.update({
                'file': fname,
                'task': task,
                'subject': subject,
                'condition': condition,
                'epoch_idx': ei,
                'channel': ch_names[ci]
            })
            rows.append(feats)
    
    # Compute epoch-level spatial features (one row per epoch)
    epoch_spatial_rows = []
    for ei in range(n_epochs):
        epoch_data = data[ei, :, :]
        spatial_feats = compute_spatial_features(epoch_data, ch_names, sfreq)
        spatial_feats.update({
            'file': fname,
            'task': task,
            'subject': subject,
            'condition': condition,
            'epoch_idx': ei
        })
        epoch_spatial_rows.append(spatial_feats)
    
    # Compute Riemannian covariance features (epoch-level)
    riem_rows = []
    riem_dict, covmats = compute_riemannian_features(data)
    for ei in range(n_epochs):
        riem_feats = riem_dict.get(ei, {})
        riem_feats.update({
            'file': fname,
            'task': task,
            'subject': subject,
            'condition': condition,
            'epoch_idx': ei
        })
        riem_rows.append(riem_feats)
    
    # Return tuple: (per-channel df, spatial df, riemannian df, cov_meta)
    # cov_meta contains covariance matrices and per-epoch metadata for npz
    cov_meta = {
        'covmats': covmats,  # (n_epochs, n_channels, n_channels) or None
        'subjects': np.full(n_epochs, subject, dtype=np.int32),
        'epoch_ids': np.arange(n_epochs, dtype=np.int32),
        'conditions': np.array([condition] * n_epochs),
        'tasks': np.array([task] * n_epochs)
    } if covmats is not None else None
    
    return pd.DataFrame(rows), pd.DataFrame(epoch_spatial_rows), pd.DataFrame(riem_rows), cov_meta

def compute_baselines_from_df(final_df, condition_col='condition', baseline_label='relaxed'):
    """
    Compute mean/std for each subject from 'relaxed' baseline epochs.
    Used for reporting-time normalization (NOT for CV, which computes this on fold train).
    Z-scoring by baseline removes inter-subject offsets without leakage if used carefully.
    """
    baselines = {}
    subjects = final_df['subject'].unique()
    
    # Identify feature columns (numeric)
    feature_cols = final_df.select_dtypes(include=[np.number]).columns.tolist()
    IGNORED_COLS = ['subject', 'epoch_idx', 'unique_id', 'target']
    feature_cols = [c for c in feature_cols if c not in IGNORED_COLS]
    
    for subj in subjects:
        # Select relaxed epochs for this subject
        base_df = final_df[(final_df['subject'] == subj) & (final_df[condition_col] == baseline_label)]
        
        if len(base_df) < 2:
            baselines[int(subj)] = None
            continue
            
        # Compute stats
        stats = {}
        for feat in feature_cols:
            vals = base_df[feat].dropna()
            if len(vals) > 0:
                stats[feat] = {
                    'mean': float(vals.mean()),
                    'std': float(vals.std(ddof=1)) if len(vals) > 1 else 1.0
                }
        baselines[int(subj)] = stats
        
    return baselines

def main():
    mne.set_log_level('WARNING')
    ensure_dir(OUT_FOLDER)
    
    # Find epoch files
    files = sorted(glob.glob(os.path.join(FINAL_DATA_FOLDER, "*_final_epo.fif")))
    print(f"Found {len(files)} epoch files in {FINAL_DATA_FOLDER}")
    
    if not files:
        print("No files found. Run preprocessing first.")
        return

    # Parallel feature extraction
    try:
        from importlib import import_module
        config = import_module('00_config')
        n_jobs = getattr(config, 'DEFAULT_N_JOBS', max(1, os.cpu_count() - 2))
    except Exception:
        n_jobs = max(1, os.cpu_count() - 2)
        
    print(f"Extracting features with {n_jobs} cores...")
    
    results = Parallel(n_jobs=n_jobs)(
        delayed(process_file_optimized)(f) for f in files
    )
    
    # Separate per-channel, spatial, riemannian dataframes, and covariance metadata
    channel_dfs = []
    spatial_dfs = []
    riem_dfs = []
    all_cov_metas = []  # For covariances.npz
    
    for result in results:
        if isinstance(result, tuple) and len(result) == 4:
            ch_df, sp_df, ri_df, cov_meta = result
            if not ch_df.empty:
                channel_dfs.append(ch_df)
            if not sp_df.empty:
                spatial_dfs.append(sp_df)
            if not ri_df.empty:
                riem_dfs.append(ri_df)
            if cov_meta is not None and cov_meta.get('covmats') is not None:
                all_cov_metas.append(cov_meta)
        elif isinstance(result, tuple) and len(result) == 3:
            ch_df, sp_df, ri_df = result
            if not ch_df.empty:
                channel_dfs.append(ch_df)
            if not sp_df.empty:
                spatial_dfs.append(sp_df)
            if not ri_df.empty:
                riem_dfs.append(ri_df)
        elif isinstance(result, tuple) and len(result) == 2:
            # Backwards compatibility
            ch_df, sp_df = result
            if not ch_df.empty:
                channel_dfs.append(ch_df)
            if not sp_df.empty:
                spatial_dfs.append(sp_df)
    
    if not channel_dfs:
        print("Error: No features extracted.")
        return

    # Concatenate per-channel features
    print("Concatenating per-channel features...")
    full_df = pd.concat(channel_dfs, ignore_index=True)
    
    # Concatenate spatial features
    if spatial_dfs:
        print("Concatenating spatial features...")
        spatial_df = pd.concat(spatial_dfs, ignore_index=True)
    else:
        spatial_df = pd.DataFrame()

    # Concatenate Riemannian features
    riem_df = None
    if riem_dfs:
        print("Concatenating Riemannian features...")
        riem_df = pd.concat(riem_dfs, ignore_index=True)
    
    # --- PIVOT TO WIDE FORMAT (Critical for ML) ---
    print("Pivoting to wide format (Channels x Features)...")
    full_df['unique_id'] = full_df['file'] + '_' + full_df['epoch_idx'].astype(str)
    
    # Metadata columns to preserve
    meta_cols = ['unique_id', 'file', 'task', 'subject', 'condition', 'epoch_idx']
    meta_df = full_df[meta_cols].drop_duplicates(subset=['unique_id'])
    
    # Numeric feature columns
    feature_cols = [c for c in full_df.columns if c not in meta_cols + ['channel']]
    
    # Pivot: unique_id x channel -> features
    pv = full_df.set_index(['unique_id', 'channel'])[feature_cols]
    pv = pv.unstack(level='channel')
    # Flatten MultiIndex: (feature, channel) -> channel_feature (e.g., AF3_rms)
    pv.columns = [f"{ch}_{feat}" for feat, ch in pv.columns]
    
    # Join with metadata
    final_df = meta_df.set_index('unique_id').join(pv)
    
    # --- MERGE OTHER FEATURES ---
    # Merge spatial
    if not spatial_df.empty:
        print("Merging spatial features...")
        spatial_df['unique_id'] = spatial_df['file'] + '_' + spatial_df['epoch_idx'].astype(str)
        sp_feat_cols = [c for c in spatial_df.columns if c not in meta_cols]
        final_df = final_df.join(spatial_df[['unique_id'] + sp_feat_cols].set_index('unique_id'), how='left')

    # Merge Riemannian (Renaming with RIEM__ prefix)
    if riem_df is not None and not riem_df.empty:
        print("Merging Riemannian features...")
        riem_df['unique_id'] = riem_df['file'] + '_' + riem_df['epoch_idx'].astype(str)
        riem_cols = [c for c in riem_df.columns if c.startswith('riem_')]
        rename_map = {c: f"RIEM__{c}" for c in riem_cols}
        riem_subset = riem_df[['unique_id'] + list(rename_map.keys())].rename(columns=rename_map)
        final_df = final_df.join(riem_subset.set_index('unique_id'), how='left')

    # Reset index
    final_df = final_df.reset_index(drop=True)
    
    # --- SUBJECT EXCLUSION LOGGING (Risk 7) ---
    print("Generating subject exclusion log...")
    expected_subjects = set()
    for f in files:
        try:
            fname = os.path.basename(f)
            sid = int(re.search(r'(\d+)', fname).group(1))
            expected_subjects.add(sid)
        except:
            pass
            
    final_subjects = set(final_df['subject'].unique()) if 'subject' in final_df.columns else set()
    dropped_subjects = expected_subjects - final_subjects
    
    exclusion_log = []
    for s in dropped_subjects:
        exclusion_log.append({'subject': s, 'reason': '0 Epochs Yielded', 'stage': '02_feature_extraction'})
        
    for s in final_subjects:
        count = len(final_df[final_df['subject'] == s])
        if count < 5:
            exclusion_log.append({'subject': s, 'reason': f'Low Epoch Count (n={count})', 'stage': '02_warning'})
            
    if exclusion_log:
        pd.DataFrame(exclusion_log).to_csv(os.path.join(OUT_FOLDER, 'subject_exclusion_log.csv'), index=False)
        print(f"Dropped subjects: {dropped_subjects}")

    # --- SAVE OUTPUTS ---
    print(f"Saving {len(final_df)} epochs with {len(final_df.columns)} features...")
    final_df.to_parquet(OUT_FILE, index=False)
    print(f"Saved Parquet: {OUT_FILE}")

    # --- BASELINE STATS (Risk 8) ---
    print("Computing baseline statistics (diagnostic)...")
    baselines = compute_baselines_from_df(final_df)
    if isinstance(baselines, dict):
        baselines['intended_use'] = "diagnostic_only_not_for_training"
    
    import json
    with open(os.path.join(OUT_FOLDER, "baseline_stats.json"), 'w') as f:
        json.dump(baselines, f, indent=2)
    print("Saved baseline_stats.json (DIAGNOSTIC ONLY)")

    # Save Covariances NPZ (if enabled)
    if all_cov_metas and getattr(config, 'RIEMANN_SAVE_COV', False):
        cov_out = getattr(config, 'RIEMANN_COV_OUT', 'FeatureSelection/covariances.npz')
        ensure_dir(os.path.dirname(cov_out))
        
        # Concatenate all parts
        big_covs = np.concatenate([m['covmats'] for m in all_cov_metas], axis=0)
        big_subjs = np.concatenate([m['subjects'] for m in all_cov_metas], axis=0)
        big_epochs = np.concatenate([m['epoch_ids'] for m in all_cov_metas], axis=0)
        big_conds = np.concatenate([m['conditions'] for m in all_cov_metas], axis=0)
        big_tasks = np.concatenate([m['tasks'] for m in all_cov_metas], axis=0)
        
        np.savez_compressed(
            cov_out, 
            covs=big_covs,
            subjects=big_subjs,
            epoch_ids=big_epochs,
            conditions=big_conds,
            tasks=big_tasks
        )
        print(f"Saved Covariances NPZ with metadata: {cov_out}")

if __name__ == "__main__":
    main()
