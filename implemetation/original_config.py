"""
00_config.py - Centralized Configuration for EEG Preprocessing Pipeline

This file contains all parameters, configurations, and settings used across
the preprocessing pipeline stages. Any changes to methodology should be
made here rather than in individual stage files.

Reference: Preprocessing_Guide.md (Implementation Specification v1.0)
"""

import os

# =============================================================================
# RANDOM SEED (Reproducibility)
# =============================================================================
RANDOM_SEED = 42

# =============================================================================
# EEG HARDWARE CONFIGURATION (Emotiv EPOC X)
# =============================================================================
SFREQ = 256  # Sampling frequency in Hz (fixed for EPOC X)

EEG_CHANNELS = [
    'AF3', 'F7', 'F3', 'FC5', 'T7', 'P7', 'O1',
    'O2', 'P8', 'T8', 'FC6', 'F4', 'F8', 'AF4'
]  # Standard 14-channel EPOC X configuration

# =============================================================================
# SUBJECT EXCLUSIONS
# =============================================================================
# Subjects excluded from preprocessing (use raw subject IDs as integers)
EXCLUDE_SUBJECTS_PREPROCESSING = []

# Subjects excluded from downstream analysis (feature extraction, classification)
EXCLUDE_SUBJECTS_ANALYSIS = [9, 10]

# =============================================================================
# STAGE 0: RAW INGESTION & SEGMENTATION
# =============================================================================
# Marker semantics
RELAXED_END_MARKER = 3  # MarkerValueInt value marking end of relaxed baseline

# =============================================================================
# STAGE 1: PRE-ICA SIGNAL CLEANING
# =============================================================================
# Bandpass filter parameters (SPEC: 1.0-45.0 Hz)
BP_LOW_FREQ = 1.0   # High-pass cutoff (Hz) - removes DC drift
BP_HIGH_FREQ = 45.0  # Low-pass cutoff (Hz) - removes high-freq artifacts

# Notch filter frequencies (line noise removal)
NOTCH_FREQS = [50, 60]  # 50 Hz (EU) and 60 Hz (US) power line frequencies

# Bad channel detection thresholds
FLATLINE_STD_THRESHOLD = 5e-9   # Channels with STD below this are flat (in Volts)
HIGH_AMPLITUDE_THRESHOLD = 500e-6  # Peak-to-peak threshold (in Volts)
MAX_BAD_CHANNELS = 6  # If more than this many bad channels, skip subject

# =============================================================================
# STAGE 2: ICA FITTING & ARTIFACT REJECTION
# =============================================================================
# ICA configuration
HP_FOR_ICA = 1.0     # High-pass for ICA stability (Hz)
BP_HIGH_FOR_ICA = 45.0  # Low-pass for ICA fitting (Hz)
ICA_METHOD = 'infomax'
ICA_MAX_ITER = 3000

# ICA component selection
# Either use variance fraction OR n_channels - 1
# Spec: 0.95 (95% variance) is preferred
ICA_N_COMPONENTS = 0.95  # Use 95% variance (not min with n_channels!)

# ICLabel artifact rejection
ICLABEL_PROB_THR = 0.8  # Minimum probability to exclude component
ICLABEL_EXCLUDE_LABELS = ['muscle artifact', 'eye blink', 'heart beat', 
                           'line noise', 'channel noise']  # Labels to reject

# Coarse AutoReject before ICA (removes catastrophic segments)
USE_COARSE_AR_BEFORE_ICA = True
COARSE_AR_THRESHOLD_UV = 550e-6  # 500 µV threshold

# =============================================================================
# STAGE 3: EPOCHING & AUTOREJECT
# =============================================================================
# Epoch parameters
EPOCH_DUR = 6.0  # Epoch duration in seconds

# Overlap strategy (CRITICAL: must be identical for both conditions per Final_implementation.md)
# Set to 0 to avoid autocorrelation confounds. Augmentation via overlap only in training folds.
EPOCH_OVERLAP_RELAXED = 0.0   # 0% overlap for both conditions
EPOCH_OVERLAP_ATTENTION = 0.0  # 0% overlap for both conditions

# AutoReject parameters
MIN_EPOCHS_FOR_AR = 4  # Minimum epochs required to run AutoReject
AR_N_INTERPOLATE = [3, 4, 5, 6, 7]  # Grid for interpolation
AR_CONSENSUS_GRID = [0.4, 0.5, 0.6, 0.7]  # Consensus thresholds

# Yield threshold
MIN_YIELD_PCT = 20.0  # Minimum yield percentage to consider subject valid

# =============================================================================
# QC THRESHOLDS
# =============================================================================
MIN_EPOCHS_QC = 5  # Minimum epochs for QC pass
QC_AMPLITUDE_THRESHOLD_UV = 200  # Max amplitude in microvolts

# =============================================================================
# FOLDER STRUCTURE
# =============================================================================
PREPROCESSED_DIR = 'Preprocessed'
FEATURES_DIR = 'Features'
BIOMARKER_DIR = 'BIOMARKER DISCOVERY'

RAW_CSV_FOLDER = 'Attention_EEG'
STAGE0_OUT = os.path.join(PREPROCESSED_DIR, 'EEGData')      # Stage 0: Raw FIF files
STAGE1_OUT = os.path.join(PREPROCESSED_DIR, 'PreICA')       # Stage 1: Pre-ICA cleaned
STAGE2_OUT = os.path.join(PREPROCESSED_DIR, 'ICA')          # Stage 2: ICA solutions
STAGE3_OUT = os.path.join(PREPROCESSED_DIR, 'AutoReject')   # Stage 3: Final epochs
FINAL_DATA_FOLDER = os.path.join(PREPROCESSED_DIR, 'FinalData')
LOGS_FOLDER = os.path.join(PREPROCESSED_DIR, 'Logs')
QC_FOLDER = os.path.join(PREPROCESSED_DIR, 'QC')
CACHE_DIR = 'cache/epochs'  # Cleaned epoch cache per subject

# =============================================================================
# PARALLELISM & MEMORY
# =============================================================================

MEMORY_SAVING = False
CPU_COUNT = os.cpu_count() or 2
DEFAULT_N_JOBS = 1 if MEMORY_SAVING else max(1, min(15, CPU_COUNT - 1))

# =============================================================================
# SKIP TOKENS (files to ignore)
# =============================================================================
SKIP_TOKEN = 'intervalmarker'

# =============================================================================
# FEATURE EXTRACTION CONFIGURATION
# =============================================================================
# Reference: Feature_Extraction_Guide.md

# Frequency bands (Hz)
BANDS = {
    'delta': (0.5, 4.0),
    'theta': (4.0, 8.0),
    'alpha': (8.0, 13.0),
    'beta': (13.0, 30.0),
    'gamma': (30.0, 45.0)
}

# Channel groups for spatial features
FRONTAL_CHANNELS = ['AF3', 'F7', 'F3', 'FC5', 'F4', 'F8', 'AF4', 'FC6']
PARIETAL_CHANNELS = ['P7', 'P8']
TEMPORAL_CHANNELS = ['T7', 'T8']
OCCIPITAL_CHANNELS = ['O1', 'O2']
LEFT_CHANNELS = ['AF3', 'F7', 'F3', 'FC5', 'T7', 'P7', 'O1']
RIGHT_CHANNELS = ['AF4', 'F8', 'F4', 'FC6', 'T8', 'P8', 'O2']

# PSD settings
PSD_NPERSEG = 512  # Welch nperseg parameter
PSD_MIN_LEN_S = 2.0  # Minimum signal length for PSD

# Sample entropy parameters
SAMPEN_M = 2  # Embedding dimension
SAMPEN_R_FACTOR = 0.2  # r = r_factor * std(x)

# Higuchi Fractal Dimension
HIGUCHI_KMAX = 10

# LZC binarization
LZ_THRESHOLD = 'mean'  # 'mean' or 'median'

# Connectivity settings
COMPUTE_CONNECTIVITY = False  # PLI/coherence disabled (missing P3/P4 channels)
CONNECTIVITY_METHODS = ['pli']
PLI_MIN_POWER = 1e-12

# Feature availability thresholds
MIN_FEATURE_AVAILABILITY = 0.30  # Keep features available in >= 30% epochs
OPTIONAL_FEATURE_MISSING_RATE = 0.70  # Mark as optional if missing > 70%

# Numba acceleration
USE_NUMBA = True

# Feature extraction output
FEATURE_OUT_FOLDER = FEATURES_DIR
FEATURE_OUT_FILE = 'features_enhanced.parquet'

# Riemannian / Covariance features
USE_PYRIEMANN = True
RIEMANN_SHRINKAGE = 'oasis'  # 'oasis' or 'ledoit_wolf'
RIEMANN_SAVE_COV = True     # Saved to covariances.npz for later use
RIEMANN_COV_OUT = os.path.join(FEATURES_DIR, 'covariances.npz')

# =============================================================================
# REDUNDANCY PRUNING CONFIGURATION
# =============================================================================
# Reference: Redundancy_Pruning_and_Wasserstein_Distance_Guide.md

REDUNDANCY_ENABLED = True
CORR_METHOD = 'spearman'           # Spearman rank correlation
DISTANCE_FORMULA = '1-abs(rho)'    # Distance metric for clustering
CORR_CLUSTER_THRESH = 0.20         # Default clustering distance threshold (approx |rho|>0.8)
MIN_FEATURES_AFTER_PRUNING = 50    # Minimum features to keep after pruning

# Threshold sweep parameters for adaptive pruning
PRUNING_THRESH_RANGE = (0.05, 0.5)  # Scan range
PRUNING_THRESH_STEP = 0.01
PRUNING_STRATEGY = 'fixed'         # 'fixed' | 'adaptive' | 'elbow'
FIXED_PRUNING_THRESHOLD = 0.20     # Distance threshold (1 - |rho|) for fixed strategy

# Backwards-compatible keys used by pipeline scripts
PRUNER_THRESHOLD = CORR_CLUSTER_THRESH
WASSERSTEIN_RATIO_THRESHOLD = 1.0   # default ratio threshold (ratio < 1 means keep)
WASSERSTEIN_TOP_K = None            # optional: keep top-k by ratio
USE_STATISTICAL_SELECTOR_IN_PIPELINE = False

# =============================================================================
# WASSERSTEIN DISTANCE CONFIGURATION
# =============================================================================
WASSERSTEIN_MIN_EPOCHS = 5          # Minimum epochs for valid distance
WASSERSTEIN_BOOTSTRAPS = 200        # Number of bootstrap iterations
WASSERSTEIN_EPSILON = 1e-9          # Avoid division by zero
SENSITIVITY_QUANTILE = 0.25         # Sensitivity floor (25th percentile)
CONVERGENCE_RATIO_THRESHOLD = 1.0   # Ratio must be < this for selection
USE_GLOBAL_WASSERSTEIN_CANDIDATES = False  # Set True only for diagnostic comparison

# =============================================================================
# LOSO & STATISTICAL VALIDATION CONFIGURATION
# =============================================================================
# Reference: Statistical_Validation_loso_Guide.md

LOSO_RESAMPLES = None              # None => full LOSO (n_subjects); else integer
RESAMPLING_ITERATIONS = 100        # Subsampling iterations per fold
PERMUTATION_ITERATIONS = 1000      # Permutation iterations per test
SIG_ALPHA = 0.05                   # Significance level
FDR_Q = 0.05                       # FDR correction q-value
ROBUSTNESS_THRESHOLD = 0.75        # Feature must survive >= 75% folds
MIN_SUBJECTS_FOR_LOSO = 5          # Minimum subjects for valid LOSO

# Balanced sampling parameters
K_PERCENTILE = 10                  # Percentile for robust K selection
K_FLOOR = 10                       # Minimum K value

# Normalization method for LOSO
NORMALIZATION_METHOD = 'zscore'    # 'zscore' (default) or 'robust'. 'baseline' is for LOSO only.

# =============================================================================
# STATISTICAL METHODOLOGY
# =============================================================================
STATS_METHOD = 'hboot'       # 'lmm' (Linear Mixed Model), 'hboot', or 'wilcoxon'
POOL_RELAXED = True        # Pool relaxed epochs as common baseline
SMOKE_MODE = False         # Set to True for fast testing (fewer bootstraps)

# Statistical testing options (Granular control)
STAT_TEST = 'hboot'         # 'wilcoxon' | 'signflip' | 'hboot' (hier_boot)
HIER_BOOTSTRAP = True
HIER_BOOTSTRAP_REPS = 1000
N_BOOT_FINAL = 1000            # Default bootstrap iterations for validation scripts
LOMO_STRICT = True             # strict LOMO = exclude test modality from training entirely

# =============================================================================
# SUBJECT EXCLUSIONS (DOWNSTREAM)
# =============================================================================
# Subjects 9 and 10 lack audio modality data - exclude from wasserstein analysis

# =============================================================================
# SAFETY & PIPELINE CONTROL
# =============================================================================
# Safety flags (explicit; prevents accidental leakage)
DEBUG_SAFE_MODE = True                 # If True, preprocessing and training scripts will assert critical invariants
RIEMANN_COMPUTE_TS_GLOBAL = False      # NEVER fit TangentSpace globally unless explicitly overriding
PRUNING_NORMALIZE = True               # Z-score per-subject before redundancy pruning (avoids Simpson effect)
PRUNING_NORMALIZE_SCOPE = 'subject'    # 'subject' or 'subject_task'
BASELINE_NORMALIZE = True              # Use relaxed/baseline epochs per subject to normalize epochs (training-only)
BASELINE_METHOD = 'zscore_by_baseline' # 'subtract_mean' or 'zscore_by_baseline' or None
VERBOSE = True

# =============================================================================
# CLASSIFIER PIPELINE CONFIGURATION
# =============================================================================
PRUNER_THRESHOLD = 0.20                  # Redundancy clustering threshold
WASSERSTEIN_RATIO_THRESHOLD = 1.0
WASSERSTEIN_TOP_K = None
USE_STATISTICAL_SELECTOR_IN_PIPELINE = False
MIN_EFFECT_SIZE_FOR_STAT = 0.5
