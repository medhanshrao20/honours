"""
EEG Preprocessing Pipeline for Emotiv Epoc X

Enhanced with Engineering Metrics:
1. Bandpass filtering SNR attenuation (50Hz)
2. ICA artifact confidence probability (ICLabel if available)
3. Data yield (retention rate)

Pipeline (MODIFIED - ICA workflow fixed):
Stage0: CSV -> Raw FIF (relaxed + attention segments) - with subject exclusion
Stage1: PreICA filtering + bad channel MARKING (NO interpolation before ICA)
Stage2: ICA fit on non-bad channels, optional ICLabel exclusion, ALWAYS save ICA
Stage3: Apply ICA to continuous, interpolate bad channels AFTER ICA, 
        epoch with differential overlap (0% relaxed, 0% attention), AutoReject
QC: Session-level QC over saved epochs
"""

import os
import time
import traceback
import csv
import re
import warnings
from glob import glob

from joblib import Parallel, delayed
import numpy as np
import pandas as pd

import mne
from mne.preprocessing import ICA
from scipy.signal import welch
from sklearn.decomposition import PCA

# Optional libraries
try:
    from mne_icalabel import label_components
    IC_LABEL_AVAILABLE = True
except Exception:
    IC_LABEL_AVAILABLE = False

try:
    from autoreject import AutoReject
    AUTOREJECT_AVAILABLE = True
except Exception:
    AUTOREJECT_AVAILABLE = False

# ---------------- CONFIG ----------------
# Import from centralized config module
try:
    from importlib import import_module
    config = import_module('00_config')
    EEG_CHANNELS = config.EEG_CHANNELS
    SFREQ = config.SFREQ
    EXCLUDE_SUBJECTS = config.EXCLUDE_SUBJECTS_PREPROCESSING  # Empty for preprocessing
    EPOCH_OVERLAP_RELAXED = config.EPOCH_OVERLAP_RELAXED
    EPOCH_OVERLAP_ATTENTION = config.EPOCH_OVERLAP_ATTENTION
    EPOCH_DURATION_S = config.EPOCH_DUR
    HP_FOR_ICA = config.HP_FOR_ICA
    BP_HIGH_FOR_ICA = config.BP_HIGH_FOR_ICA
    ICLABEL_PROB_THR = config.ICLABEL_PROB_THR
    MAX_BAD_CHANNELS = config.MAX_BAD_CHANNELS
    RANDOM_SEED = config.RANDOM_SEED
    ICA_N_COMPONENTS = config.ICA_N_COMPONENTS
    ICA_METHOD = getattr(config, 'ICA_METHOD', 'infomax')
    ICA_MAX_ITER = getattr(config, 'ICA_MAX_ITER', 3000)
    ICLABEL_EXCLUDE_LABELS = getattr(config, 'ICLABEL_EXCLUDE_LABELS', ['muscle artifact', 'eye blink', 'heart beat', 'line noise', 'channel noise'])
    CACHE_DIR = getattr(config, 'CACHE_DIR', 'cache/epochs')
    MIN_EPOCHS_QC_CONFIG = getattr(config, 'MIN_EPOCHS_QC', 5)
    
    # Paths
    PREPROCESSED_DIR = getattr(config, 'PREPROCESSED_DIR', 'Preprocessed')
    RAW_CSV_FOLDER = getattr(config, 'RAW_CSV_FOLDER', 'Attention_EEG')
    STAGE0_OUT = getattr(config, 'STAGE0_OUT', os.path.join(PREPROCESSED_DIR, 'EEGData'))
    STAGE1_OUT = getattr(config, 'STAGE1_OUT', os.path.join(PREPROCESSED_DIR, 'PreICA'))
    STAGE2_OUT = getattr(config, 'STAGE2_OUT', os.path.join(PREPROCESSED_DIR, 'ICA'))
    STAGE3_OUT = getattr(config, 'STAGE3_OUT', os.path.join(PREPROCESSED_DIR, 'AutoReject'))
    FINAL_DATA_FOLDER = getattr(config, 'FINAL_DATA_FOLDER', os.path.join(PREPROCESSED_DIR, 'FinalData'))
    LOGS_FOLDER = getattr(config, 'LOGS_FOLDER', os.path.join(PREPROCESSED_DIR, 'Logs'))
    QC_FOLDER = getattr(config, 'QC_FOLDER', os.path.join(PREPROCESSED_DIR, 'QC'))
    
    print("[Config] Loaded from 00_config.py")
except Exception as e:
    print(f"[Config] Warning: Could not import 00_config.py ({e}), using defaults")
    # minimal fallback config object
    class _C: pass
    config = _C()
    EEG_CHANNELS = ['AF3', 'F7', 'F3', 'FC5', 'T7', 'P7', 'O1', 'O2', 'P8', 'T8', 'FC6', 'F4', 'F8', 'AF4']
    SFREQ = 256
    EXCLUDE_SUBJECTS = []  # Include all in preprocessing
    EPOCH_OVERLAP_RELAXED = 0.0
    EPOCH_OVERLAP_ATTENTION = 0.0
    EPOCH_DURATION_S = 6.0
    HP_FOR_ICA = 1.0
    BP_HIGH_FOR_ICA = 45.0
    ICLABEL_PROB_THR = 0.8
    MAX_BAD_CHANNELS = 6
    RANDOM_SEED = 42
    ICA_N_COMPONENTS = 0.95
    ICA_METHOD = 'infomax'
    ICA_MAX_ITER = 3000
    ICLABEL_EXCLUDE_LABELS = ['muscle artifact', 'eye blink', 'heart beat', 'line noise', 'channel noise']
    CACHE_DIR = 'cache/epochs'
    MIN_EPOCHS_QC_CONFIG = 5

# Defaults / Fallback Paths (if config load failed)
if 'STAGE0_OUT' not in globals():
    PREPROCESSED_DIR = 'Preprocessed'
    RAW_CSV_FOLDER = 'Attention_EEG'
    STAGE0_OUT = os.path.join(PREPROCESSED_DIR, 'EEGData')
    STAGE1_OUT = os.path.join(PREPROCESSED_DIR, 'PreICA')
    STAGE2_OUT = os.path.join(PREPROCESSED_DIR, 'ICA')
    STAGE3_OUT = os.path.join(PREPROCESSED_DIR, 'AutoReject')
    FINAL_DATA_FOLDER = os.path.join(PREPROCESSED_DIR, 'FinalData')
    LOGS_FOLDER = os.path.join(PREPROCESSED_DIR, 'Logs')
    QC_FOLDER = os.path.join(PREPROCESSED_DIR, 'QC')

# Memory & parallelism
MEMORY_SAVING = False
CPU_COUNT = (os.cpu_count() or 2)
DEFAULT_N_JOBS = 1 if MEMORY_SAVING else max(1, min(15, CPU_COUNT - 1))

# Epoching (direct 6s) - overlap is now condition-dependent
MIN_EPOCHS_FOR_AR = 4

# AutoReject grid
AR_N_INTERPOLATE = np.array([3, 4, 5, 6, 7])
AR_CONSENSUS_GRID = [0.4, 0.5, 0.6, 0.7]

# QC thresholds
SKIP_TOKEN = 'intervalmarker'
THRESH_AMP_V = 200e-6   # 200 uV
THRESH_FLAT_STD = 5e-9
NOTCH_FREQS = [50, 60]
MIN_EPOCHS_QC = 5

# ---------------- Helper Functions ----------------

def ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)

def _as_path(p):
    if isinstance(p, (list, tuple)):
        return str(p[0]) if len(p) > 0 else ''
    return str(p)

def ensure_list_unique(seq):
    return list(dict.fromkeys(seq))

def parse_task_subject_from_base(base_token: str):
    """Parse task and subject from filename base token.
    
    Returns:
        task: str - lowercase task name (e.g., 'audio', 'video', 'text')
        subject: int or None - subject ID as integer (enforces consistency)
    """
    token = str(base_token)
    m_task = re.match(r'([A-Za-z]+)', token)
    task = m_task.group(1).lower() if m_task else token.lower()
    m_subj = re.search(r'(\d+)', token)
    # CRITICAL: Always return integer subject ID for LOSO consistency
    subject = int(m_subj.group(1)) if m_subj else None
    return task, subject

class Logger:
    def __init__(self, log_path: str):
        ensure_dir(os.path.dirname(log_path) or '.')
        self.f = open(log_path, 'w', buffering=1)

    def log(self, msg: str):
        t = time.strftime('%Y-%m-%d %H:%M:%S')
        line = f"[{t}] {msg}"
        print(line)
        try:
            self.f.write(line + "\n")
            self.f.flush()
        except Exception:
            pass

    def close(self):
        try:
            self.f.close()
        except Exception:
            pass

def append_to_csv(filepath: str, row_dict: dict):
    ensure_dir(os.path.dirname(filepath) or '.')
    file_exists = os.path.isfile(filepath)
    with open(filepath, 'a', newline='') as f:
        w = csv.DictWriter(f, fieldnames=row_dict.keys())
        if not file_exists:
            w.writeheader()
        w.writerow(row_dict)

def safe_nanmax(x):
    x = np.asarray(x)
    if np.isfinite(x).any():
        return float(np.nanmax(np.abs(x)))
    return np.nan

def calculate_robust_snr_db(data_inst, target_freq=50.0, bandwidth=2.0):
    """
    Robust SNR in dB around target_freq using Welch PSD.
    Works for Raw (continuous) or Epochs (concatenated).
    """
    try:
        if isinstance(data_inst, mne.io.BaseRaw):
            data = data_inst.get_data()
            sfreq = float(data_inst.info['sfreq'])
        elif isinstance(data_inst, mne.BaseEpochs):
            data = np.hstack(data_inst.get_data())
            sfreq = float(data_inst.info['sfreq'])
        else:
            return np.nan

        n_ch, n_samp = data.shape
        if n_ch == 0 or n_samp < sfreq:
            return np.nan

        nperseg = int(min(n_samp, sfreq * 4))
        freqs, psd = welch(data, fs=sfreq, nperseg=nperseg, window='hann', average='mean')
        avg_psd = np.mean(psd, axis=0)

        idx_signal = np.where((freqs >= target_freq - bandwidth / 2) &
                              (freqs <= target_freq + bandwidth / 2))[0]
        idx_noise = np.where((freqs >= target_freq - 5.0) &
                             (freqs <= target_freq + 5.0))[0]
        idx_noise = np.setdiff1d(idx_noise, idx_signal)

        if len(idx_signal) == 0 or len(idx_noise) == 0:
            return np.nan

        power_signal = np.mean(avg_psd[idx_signal])
        power_noise = np.mean(avg_psd[idx_noise])
        if power_noise <= 0:
            return np.nan

        return 10 * np.log10(power_signal / power_noise)
    except Exception as e:
        # Warn instead of silent failure
        warnings.warn(f"SNR Calc Error: {e}")
        return np.nan

def make_fixed_epochs_from_raw(raw, duration, overlap=0.0, preload=True):
    overlap_duration = duration * overlap
    events = mne.make_fixed_length_events(raw, duration=duration, overlap=overlap_duration)
    epochs = mne.Epochs(
        raw, events, tmin=0.0, tmax=duration,
        baseline=None, reject_by_annotation=True,
        preload=preload, verbose=False
    )
    return epochs

# ---------------- Stage 0: CSV -> Raw FIF ----------------

class EEGPreprocessor:
    def __init__(self, input_folder, output_folder, eeg_channels, sfreq, n_jobs=1):
        self.input_folder = input_folder
        self.output_folder = output_folder
        self.eeg_channels = eeg_channels
        self.eeg_columns = [f'EEG.{ch}' for ch in eeg_channels]
        self.sfreq = sfreq
        self.n_jobs = n_jobs

        ensure_dir(self.output_folder)
        if not os.path.exists(self.input_folder):
            print(f"[Error] Input folder not found: {self.input_folder}")
        elif not glob(os.path.join(self.input_folder, "*")):
            print(f"[Warning] Input folder appears empty: {self.input_folder}")

    def find_csv_files(self):
        files = []
        excluded_count = 0
        for task in ['audio', 'text', 'video']:
            pattern = os.path.join(self.input_folder, f"{task}*")
            for path in sorted(glob(pattern)):
                if not os.path.isdir(path):
                    continue
                folder_name = os.path.basename(path)
                subj_digits = ''.join([c for c in folder_name if c.isdigit()])
                subject = int(subj_digits) if subj_digits else None

                # Subject exclusion check
                if subject in EXCLUDE_SUBJECTS:
                    excluded_count += 1
                    continue  # Skip this subject entirely

                for fname in os.listdir(path):
                    if SKIP_TOKEN.lower() in fname.lower():
                        continue
                    if fname.lower().endswith('.csv'):
                        files.append({'path': os.path.join(path, fname),
                                      'task': task, 'subject': subject, 'file': fname})
        
        if excluded_count > 0:
            print(f"[Stage0] Excluded {excluded_count} folders for subjects {EXCLUDE_SUBJECTS}")
        return files

    def create_raw(self, data):
        try:
            info = mne.create_info(self.eeg_channels, self.sfreq, ch_types='eeg')
            raw = mne.io.RawArray(data, info, verbose=False)
            # Attempt to set Emotiv-specific montage if available, else fallback
            try:
                # load optional emotiv montage file emotiv_14.sfp if provided
                emotiv_sfp = os.path.join(os.path.dirname(__file__), 'montages', 'emotiv_epoc_x_14.sfp')
                if os.path.exists(emotiv_sfp):
                    montage = mne.channels.read_custom_montage(emotiv_sfp)
                else:
                    montage = mne.channels.make_standard_montage('standard_1020')
                raw.set_montage(montage, on_missing='ignore', verbose=False)
            except Exception:
                raw.set_montage(mne.channels.make_standard_montage('standard_1020'), on_missing='ignore', verbose=False)
            raw.set_eeg_reference('average', projection=True, verbose=False)
            return raw
        except Exception as e:
            print(f"[Error create_raw]: {e}")
            return None

    def process_csv(self, info):
        fname = info['file']
        path = info['path']
        if SKIP_TOKEN.lower() in fname.lower():
            return []

        # SPEC: Emotiv CSVs have metadata in row 0, column headers in row 1
        # Try header=1 first (Emotiv default), then fallback to header=0
        df = None
        for header_row in [1, 0]:  # Try header=1 first for Emotiv format
            try:
                test_df = pd.read_csv(path, header=header_row, low_memory=False)
                test_df.columns = [str(c).strip().replace('\ufeff', '').strip('"').strip("'") for c in test_df.columns]
                # Check if EEG columns are present
                if all(c in test_df.columns for c in self.eeg_columns):
                    df = test_df
                    break
            except Exception:
                continue
        if df is None:
            print(f"[EXCLUDED] {fname} - EEG columns not found")
            return []

        # Column cleaning already done in header check loop
        missing = [c for c in self.eeg_columns if c not in df.columns]
        if missing:
            print(f"[EXCLUDED] Subject {info.get('subject')} ({fname}) missing channels: {missing}")
            return []

        try:
            df[self.eeg_columns] = df[self.eeg_columns].astype(float) * 1e-6
        except Exception:
            return []

        if 'MarkerValueInt' not in df.columns:
            return []

        marker_idx = df.index[df['MarkerValueInt'] == getattr(config, 'RELAXED_END_MARKER', 3)].tolist()
        if len(marker_idx) == 0:
            return []

        last_idx = marker_idx[-1]
        relaxed_df = df.iloc[:last_idx + 1]
        attention_df = df.iloc[last_idx + 1:]

        created_files = []
        for seg_name, seg_df in {'relaxed': relaxed_df, 'attention': attention_df}.items():
            if len(seg_df) == 0:
                continue
            raw = self.create_raw(seg_df[self.eeg_columns].values.T)
            if raw is None:
                continue

            subj = info['subject'] if info['subject'] is not None else ''
            fname_out = f"{info['task']}{subj}_{seg_name}_raw.fif"
            fpath = os.path.join(self.output_folder, fname_out)
            try:
                raw.save(fpath, overwrite=True, verbose=False)
                created_files.append(fpath)
            except Exception:
                pass

        return created_files

    def process_all(self):
        files = self.find_csv_files()
        print(f"Stage0: Found {len(files)} CSV files.")
        results = Parallel(n_jobs=self.n_jobs)(delayed(self.process_csv)(info) for info in files)
        all_created = [p for sub in results for p in sub]
        print(f"Stage0: Created {len(all_created)} raw FIF files.")
        return all_created

# ---------------- Stage 1: PreICA ----------------

class EEGStage1:
    def __init__(self, input_folder, output_folder, logs_folder, qc_folder, n_jobs=1, max_bad_channels=3):
        self.input_folder = input_folder
        self.output_folder = output_folder
        self.logs_folder = logs_folder
        self.qc_folder = qc_folder
        self.n_jobs = n_jobs
        self.max_bad_channels = max_bad_channels
        self.summary_file = os.path.join(qc_folder, 'summary_stage1_filtering.csv')

        ensure_dir(self.output_folder)
        ensure_dir(self.logs_folder)
        ensure_dir(self.qc_folder)

    def detect_bad_channels_eeg(self, raw):
        try:
            data = raw.get_data()
            if data.size == 0 or data.shape[1] < 2:
                return []
            chan_std = np.nanstd(data, axis=1)
            flat_thresh = getattr(config, 'FLATLINE_STD_THRESHOLD', 5e-9)
            high_thresh_abs = getattr(config, 'HIGH_AMPLITUDE_THRESHOLD', None)
            high_thresh = high_thresh_abs if high_thresh_abs is not None else (np.nanmedian(chan_std) * 15.0)
            bads = []
            for idx, std_val in enumerate(chan_std):
                if (not np.isfinite(std_val)) or (std_val < flat_thresh) or (std_val > high_thresh):
                    # NOTE: Data assumed to be in Volts (MNE standard checks apply)
                    bads.append(raw.ch_names[idx])
            return bads
        except Exception:
            return []

    def _process_file_inner(self, raw_fif):
        raw_fif = _as_path(raw_fif)
        if SKIP_TOKEN.lower() in os.path.basename(raw_fif).lower():
            return None

        base = os.path.basename(raw_fif).replace('_raw.fif', '')
        first_token = str(base).split('_')[0]
        task, subject = parse_task_subject_from_base(first_token)

        logname = os.path.basename(raw_fif).replace('_raw.fif', '_stage1') + f"_{os.getpid()}.log"
        logger = Logger(os.path.join(self.logs_folder, logname))

        try:
            logger.log(f"Stage1: {base}")
            raw = mne.io.read_raw_fif(raw_fif, preload=True, verbose=False)
            dur_raw_s = raw.n_times / float(raw.info.get('sfreq', 1.0))

            snr_pre_db = calculate_robust_snr_db(raw, target_freq=50.0)

            # Explicit DC removal
            try:
                raw.apply_function(lambda x: x - np.mean(x), channel_wise=True, verbose=False)
                logger.log("Applied explicit DC removal (mean subtraction).")
            except Exception as e:
                logger.log(f"DC removal warning: {e}")

            # Bad channel detection - MARK ONLY (do NOT interpolate before ICA)
            # Interpolation will happen in Stage3 AFTER ICA is applied
            bads = self.detect_bad_channels_eeg(raw)
            n_bads_marked = 0
            if bads:
                logger.log(f"Marking {len(bads)} bad channels (will exclude from ICA): {bads}")
                raw.info['bads'] = ensure_list_unique(list(raw.info.get('bads', [])) + bads)
                n_bads_marked = len(raw.info['bads'])
                # DO NOT call raw.interpolate_bads() here - moved to Stage3 after ICA

            # Filtering
            sfreq = float(raw.info.get('sfreq', 1.0))
            if raw.n_times >= int(2.0 * sfreq):
                try:
                    raw.notch_filter(getattr(config, 'NOTCH_FREQS', [50, 60]), phase='zero', fir_design='firwin', verbose=False)
                    # SPEC: Bandpass must be 1.0-45.0 Hz (not 0.5!)
                    raw.filter(l_freq=getattr(config, 'BP_LOW_FREQ', 1.0), h_freq=getattr(config, 'BP_HIGH_FREQ', 45.0), fir_design='firwin', phase='zero', verbose=False)
                    logger.log("Applied notch (50,60) and band-pass (1.0–45 Hz).")
                except Exception as e:
                    logger.log(f"Filtering warning: {e}")

            snr_post_db = calculate_robust_snr_db(raw, target_freq=50.0)
            attenuation = np.nan
            if not np.isnan(snr_pre_db) and not np.isnan(snr_post_db):
                attenuation = snr_pre_db - snr_post_db
                logger.log(f"50Hz Attenuation: {attenuation:.2f} dB")

            # Ensure CAR projection exists and applied
            try:
                has_avg_proj = any(p.get('desc', '') == 'Average EEG reference' for p in raw.info.get('projs', []))
                if not has_avg_proj:
                    raw.set_eeg_reference('average', projection=True, verbose=False)
                    logger.log("Set average reference projection.")
                # SPEC: Only apply_proj if projections exist (raw.proj is True)
                if raw.proj:
                    raw.apply_proj(verbose=False)
                    logger.log("Applied projections (CAR).")
            except Exception as e:
                logger.log(f"CAR warning: {e}")

            out_path = os.path.join(self.output_folder, os.path.basename(raw_fif).replace('_raw.fif', '_clean_raw.fif'))
            raw.save(out_path, overwrite=True, verbose=False)
            logger.log(f"Saved cleaned raw: {out_path}")

            log_row = {
                'file_base': base,
                'task': task,
                'subject': subject,
                'duration_s': round(dur_raw_s, 2),
                'n_bad_channels': len(raw.info.get('bads', [])),
                'snr_50hz_pre_db': round(snr_pre_db, 2) if not np.isnan(snr_pre_db) else np.nan,
                'snr_50hz_post_db': round(snr_post_db, 2) if not np.isnan(snr_post_db) else np.nan,
                'snr_attenuation_db': round(attenuation, 2) if not np.isnan(attenuation) else np.nan,
                'clean_integrity_check': 'PASS' if (np.isnan(attenuation) or attenuation > 0 or (not np.isnan(snr_post_db) and snr_post_db < 1.0)) else 'WARN'
            }
            append_to_csv(self.summary_file, log_row)

            logger.close()
            return out_path
        except Exception as e:
            logger.log(f"Stage1 failed: {e}\n{traceback.format_exc()}")
            logger.close()
            return None

    def process_all(self):
        fifs = [os.path.join(self.input_folder, f) for f in os.listdir(self.input_folder) if f.endswith('_raw.fif')]
        results = Parallel(n_jobs=self.n_jobs)(delayed(self._process_file_inner)(f) for f in fifs)
        return [r for r in results if r]

# ---------------- Stage 2: ICA ----------------

class EEGStage2:
    def __init__(self, stage1_folder, output_folder, logs_folder, qc_folder, n_jobs=1):
        self.stage1_folder = stage1_folder
        self.output_folder = output_folder
        self.logs_folder = logs_folder
        self.qc_folder = qc_folder
        self.n_jobs = n_jobs
        self.summary_file = os.path.join(qc_folder, 'summary_stage2_ica.csv')

        ensure_dir(self.output_folder)
        ensure_dir(self.logs_folder)
        ensure_dir(self.qc_folder)

    def _process_file_inner(self, clean_raw_fif):
        clean_raw_fif = _as_path(clean_raw_fif)
        base = os.path.basename(clean_raw_fif).replace('_clean_raw.fif', '')
        first_token = str(base).split('_')[0]
        task, subject = parse_task_subject_from_base(first_token)

        logger = Logger(os.path.join(self.logs_folder, f"{base}_stage2_{os.getpid()}.log"))

        try:
            logger.log(f"Stage2 start: {base}")
            raw_clean = mne.io.read_raw_fif(clean_raw_fif, preload=True, verbose=False)
            logger.log("Loaded cleaned raw.")

            raw_for_ica = raw_clean.copy()
            
            # Drop bad channels from ICA fit (they will be interpolated in Stage3 AFTER ICA)
            bad_channels = list(raw_for_ica.info.get('bads', []))
            if bad_channels:
                logger.log(f"Excluding {len(bad_channels)} bad channels from ICA fit: {bad_channels}")
                raw_for_ica.drop_channels(bad_channels)
            
            raw_for_ica.filter(l_freq=HP_FOR_ICA, h_freq=BP_HIGH_FOR_ICA, verbose=False)
            logger.log(f"Filtered for ICA fit ({HP_FOR_ICA}-{BP_HIGH_FOR_ICA} Hz) on {len(raw_for_ica.ch_names)} channels.")

            # --- Coarse AutoReject: Remove catastrophic segments before ICA ---
            # This helps ICA converge by removing extreme amplitude artifacts
            COARSE_REJECT_UV = 500e-6  # 500 µV threshold
            try:
                # Create temporary epochs for rejection check
                epochs_temp = mne.make_fixed_length_epochs(raw_for_ica, duration=EPOCH_DURATION_S, preload=True, verbose=False)
                
                # Find good epochs (below threshold)
                good_idx = []
                for i in range(len(epochs_temp)):
                    epoch_data = epochs_temp.get_data()[i]
                    peak_to_peak = np.ptp(epoch_data, axis=1).max()
                    if peak_to_peak < COARSE_REJECT_UV:
                        good_idx.append(i)
                
                n_rejected = len(epochs_temp) - len(good_idx)
                if n_rejected > 0:
                    logger.log(f"Coarse AR: Rejected {n_rejected}/{len(epochs_temp)} epochs with amplitude > {COARSE_REJECT_UV*1e6:.0f}µV")
                
                if len(good_idx) < 2:
                    logger.log("Warning: Too few good segments after coarse reject. Using all data for ICA.")
                else:
                    # Concatenate good epochs back to continuous for ICA
                    raw_for_ica = mne.concatenate_raws([mne.io.RawArray(
                        epochs_temp.get_data()[i], epochs_temp.info, verbose=False
                    ) for i in good_idx], verbose=False)
                    logger.log(f"Using {len(good_idx)} clean segments for ICA fit.")
            except Exception as e:
                logger.log(f"Coarse AR failed (continuing with all data): {e}")

            logger.log("Fitting ICA...")
            
            # Determine n_components from variance fraction (if float) or use int directly
            # Per FINAL_IMPLEMENTATION_GUIDE: convert fraction to int via PCA explained variance
            X = raw_for_ica.get_data().T  # samples x channels
            if isinstance(ICA_N_COMPONENTS, float) and 0 < ICA_N_COMPONENTS <= 1:
                pca = PCA().fit(X)
                cumsum = pca.explained_variance_ratio_.cumsum()
                n_components = int(np.searchsorted(cumsum, ICA_N_COMPONENTS) + 1)
                n_components = min(n_components, len(raw_for_ica.ch_names) - 1)
                logger.log(f"PCA variance analysis: {ICA_N_COMPONENTS:.0%} variance -> {n_components} components")
            else:
                n_components = min(int(ICA_N_COMPONENTS), len(raw_for_ica.ch_names) - 1)
                logger.log(f"Using n_components={n_components} (direct from config)")
            
            ica = ICA(method=ICA_METHOD, fit_params=dict(extended=True), max_iter=ICA_MAX_ITER, n_components=n_components, random_state=RANDOM_SEED)
            ica.fit(raw_for_ica, verbose=False)
            logger.log(f"ICA fit complete ({ica.n_components_} components).")

            ica_out = os.path.join(self.output_folder, os.path.basename(clean_raw_fif).replace('_clean_raw.fif', '_ica.fif'))

            mean_artifact_prob = np.nan
            std_artifact_prob = np.nan

            # Optional ICLabel exclusion
            if IC_LABEL_AVAILABLE:
                try:
                    raw_for_label = raw_clean.copy()
                    raw_for_label.filter(l_freq=1.0, h_freq=45.0, verbose=False)

                    epochs_label = make_fixed_epochs_from_raw(raw_for_label, duration=EPOCH_DURATION_S, overlap=0.0, preload=True)
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        labels_info = label_components(epochs_label, ica, method='iclabel')

                    probs = labels_info.get('y_pred_proba', [])
                    labels = labels_info.get('labels', [])
                    
                    # ICLabel canonical labels: brain, muscle, eye, heart, line_noise, chan_noise, other
                    # Map to our config labels (allows for different naming conventions)
                    label_to_canonical = {
                        'brain': 'brain',
                        'muscle artifact': 'muscle', 'muscle': 'muscle',
                        'eye blink': 'eye', 'eye': 'eye',
                        'heart beat': 'heart', 'heart': 'heart',
                        'line noise': 'line_noise', 'line_noise': 'line_noise',
                        'channel noise': 'chan_noise', 'chan_noise': 'chan_noise',
                        'other': 'other'
                    }

                    excluded_probs = []
                    excluded_labels = []
                    for i, lab in enumerate(labels):
                        try:
                            prob = float(np.max(probs[i]))
                        except Exception:
                            prob = 0.0
                        # Map label and check if it should be excluded
                        canonical_lab = label_to_canonical.get(lab.lower() if hasattr(lab, 'lower') else lab, lab)
                        # Exclude if prob >= threshold AND label is in exclude list OR not 'brain'/'other'
                        should_exclude = (prob >= ICLABEL_PROB_THR and 
                                         (any(excl.lower() in lab.lower() if hasattr(lab, 'lower') else False 
                                              for excl in ICLABEL_EXCLUDE_LABELS) or
                                          canonical_lab not in ['brain', 'other']))
                        if should_exclude:
                            ica.exclude.append(i)
                            excluded_probs.append(prob)
                            excluded_labels.append(canonical_lab)

                    logger.log(f"ICLabel: excluded {len(ica.exclude)} components (labels: {set(excluded_labels) if excluded_labels else 'none'}).")
                    if excluded_probs:
                        mean_artifact_prob = float(np.mean(excluded_probs))
                        std_artifact_prob = float(np.std(excluded_probs))
                except Exception as e:
                    logger.log(f"ICLabel failed (continuing without it): {e}")

            # ALWAYS save ICA (critical)
            ica.save(ica_out, overwrite=True)
            logger.log(f"Saved ICA: {ica_out}")

            n_comp_total = int(getattr(ica, 'n_components_', 0) or 0)
            n_comp_excl = int(len(getattr(ica, 'exclude', [])))
            pct_excl = round((n_comp_excl / max(1, n_comp_total)) * 100.0, 1)

            log_row = {
                'file_base': base,
                'task': task,
                'subject': subject,
                'n_components_total': n_comp_total,
                'n_components_excluded': n_comp_excl,
                'percent_excluded': pct_excl,
                'mean_artifact_prob': round(mean_artifact_prob, 3) if not np.isnan(mean_artifact_prob) else 0.0,
                'std_artifact_prob': round(std_artifact_prob, 3) if not np.isnan(std_artifact_prob) else 0.0,
                'iclabel_available': str(IC_LABEL_AVAILABLE)
            }
            append_to_csv(self.summary_file, log_row)

            logger.close()
            return ica_out
        except Exception as e:
            logger.log(f"Stage2 failed: {e}\n{traceback.format_exc()}")
            logger.close()
            return None

    def process_all(self):
        files = [os.path.join(self.stage1_folder, f) for f in os.listdir(self.stage1_folder) if f.endswith('_clean_raw.fif')]
        results = Parallel(n_jobs=self.n_jobs)(delayed(self._process_file_inner)(f) for f in files)
        return [r for r in results if r]

# ---------------- Stage 3: Direct 6s Epoching + AutoReject ----------------

class EEGStage3:
    """
    Simplified Stage 3: Apply ICA to continuous data, create 6s epochs directly
    with 0% overlap, then run AutoReject.
    """
    def __init__(self, stage1_folder, stage2_folder, output_folder, logs_folder, qc_folder, n_jobs=1, min_epochs_for_ar=4):
        self.stage1_folder = stage1_folder
        self.stage2_folder = stage2_folder
        self.output_folder = output_folder
        self.logs_folder = logs_folder
        self.qc_folder = qc_folder
        self.n_jobs = n_jobs
        self.min_epochs_for_ar = min_epochs_for_ar
        self.summary_file = os.path.join(qc_folder, 'summary_stage3_yield.csv')

        ensure_dir(self.output_folder)
        ensure_dir(self.logs_folder)
        ensure_dir(self.qc_folder)

    def _match_files(self):
        cr_map = {
            f.replace('_clean_raw.fif', ''): os.path.join(self.stage1_folder, f)
            for f in os.listdir(self.stage1_folder)
            if f.endswith('_clean_raw.fif')
        }
        pairs = []
        for f in os.listdir(self.stage2_folder):
            if f.endswith('_ica.fif'):
                base = f.replace('_ica.fif', '')
                if base in cr_map:
                    pairs.append((cr_map[base], os.path.join(self.stage2_folder, f)))
        return pairs

    def _process_pair_inner(self, clean_raw_fif, ica_fif):
        clean_raw_fif = _as_path(clean_raw_fif)
        ica_fif = _as_path(ica_fif)

        base = os.path.basename(clean_raw_fif).replace('_clean_raw.fif', '')
        first_token = str(base).split('_')[0]
        task, subject = parse_task_subject_from_base(first_token)

        logger = Logger(os.path.join(self.logs_folder, f"{base}_stage3_{os.getpid()}.log"))

        try:
            logger.log(f"Stage3 (Direct 6s Epoching) Start: {base}")

            # 1) LOAD + APPLY ICA TO CONTINUOUS
            raw_clean = mne.io.read_raw_fif(clean_raw_fif, preload=True, verbose=False)
            ica = mne.preprocessing.read_ica(ica_fif)

            logger.log("Applying ICA to continuous raw...")
            raw_ica = ica.apply(raw_clean, exclude=getattr(ica, 'exclude', []), verbose=False)

            # 1.5) INTERPOLATE BAD CHANNELS AFTER ICA (moved from Stage1)
            bad_channels = list(raw_ica.info.get('bads', []))
            if bad_channels:
                logger.log(f"Interpolating {len(bad_channels)} bad channels AFTER ICA: {bad_channels}")
                try:
                    raw_ica.interpolate_bads(reset_bads=True, verbose=False)
                    logger.log("Bad channel interpolation complete.")
                except Exception as e:
                    logger.log(f"Interpolation warning: {e}")

            # 2) DETERMINE EPOCH OVERLAP BASED ON CONDITION
            # Condition is in the filename: e.g., "audio1_relaxed" or "video5_attention"
            condition = 'relaxed' if 'relaxed' in base.lower() else 'attention'
            epoch_overlap = EPOCH_OVERLAP_RELAXED if condition == 'relaxed' else EPOCH_OVERLAP_ATTENTION
            
            logger.log(f"Condition: {condition} | Epoch overlap: {epoch_overlap * 100:.0f}%")
            logger.log(f"Creating {EPOCH_DURATION_S}s epochs with {epoch_overlap * 100:.0f}% overlap...")
            
            epochs = make_fixed_epochs_from_raw(
                raw_ica,
                duration=EPOCH_DURATION_S,
                overlap=epoch_overlap,
                preload=True
            )
            epochs.apply_baseline((None, None), verbose=False)

            n_epochs_initial = len(epochs)
            logger.log(f"Created {n_epochs_initial} epochs of {EPOCH_DURATION_S}s ({condition})")

            if n_epochs_initial == 0:
                logger.log("No epochs created.")
                logger.close()
                return None

            # 3) AUTOREJECT ON 6s EPOCHS
            epochs_clean = epochs
            ar_applied = False
            n_dropped = 0
            n_interpolated = 0

            if AUTOREJECT_AVAILABLE and n_epochs_initial >= self.min_epochs_for_ar:
                logger.log(f"Running AutoReject with {n_epochs_initial} epochs...")
                try:
                    ar = AutoReject(
                        n_interpolate=AR_N_INTERPOLATE,
                        consensus=AR_CONSENSUS_GRID,
                        cv=min(5, n_epochs_initial),
                        n_jobs=self.n_jobs,
                        random_state=42,
                        verbose=False
                    )
                    epochs_clean, reject_log = ar.fit_transform(epochs, return_log=True)
                    ar_applied = True

                    n_dropped = int(np.sum(reject_log.bad_epochs))
                    # reject_log.labels is 2D (n_epochs, n_channels), check if any channel was interpolated per epoch
                    # labels: 0=good, 1=bad (to be interpolated), 2=bad (not repairable)
                    interpolated_mask = np.any(reject_log.labels == 1, axis=1)  # Shape: (n_epochs,)
                    n_interpolated = int(np.sum(~reject_log.bad_epochs & interpolated_mask))
                    logger.log(f"AutoReject: dropped {n_dropped} epochs, interpolated {n_interpolated} epochs")
                except Exception as e:
                    logger.log(f"AutoReject failed: {e}. Using all epochs.")
                    epochs_clean = epochs
            else:
                logger.log("Skipping AutoReject (too few epochs or not available)")

            n_epochs_final = len(epochs_clean)

            # 4) YIELD (unique time coverage), clamp to raw duration
            dur_raw_s = raw_ica.n_times / float(raw_ica.info['sfreq'])
            stride_s = EPOCH_DURATION_S * (1.0 - epoch_overlap)

            if n_epochs_final > 0:
                seconds_covered = EPOCH_DURATION_S + (n_epochs_final - 1) * stride_s
            else:
                seconds_covered = 0.0

            seconds_covered = float(min(seconds_covered, dur_raw_s))
            yield_pct = (seconds_covered / dur_raw_s) * 100.0 if dur_raw_s > 0 else 0.0

            logger.log(f"Yield: {yield_pct:.1f}% ({seconds_covered:.1f}s / {dur_raw_s:.1f}s)")
            logger.log(f"Final: {n_epochs_final} epochs of {EPOCH_DURATION_S}s")

            # 5) FINAL SNR
            final_snr = np.nan
            if n_epochs_final > 0:
                final_snr = calculate_robust_snr_db(epochs_clean, target_freq=50.0)
                logger.log(f"Final SNR: {final_snr:.2f} dB")

            # 6) SAVE
            out = None
            if n_epochs_final > 0:
                out = os.path.join(self.output_folder, f"{base}_AutoReject-epo.fif")
                epochs_clean.save(out, overwrite=True, verbose=False)

                ensure_dir(FINAL_DATA_FOLDER)
                final_out = os.path.join(FINAL_DATA_FOLDER, f"{base}_final_epo.fif")
                epochs_clean.save(final_out, overwrite=True, verbose=False)
                logger.log(f"Saved: {final_out}")

                # Cache per subject for LOSO pipeline (per Final_implementation.md Section 5.1)
                ensure_dir(CACHE_DIR)

                # Save baseline stats for downstream per-subject normalization (if enabled)
                if condition == 'relaxed' and subject is not None and getattr(config, 'BASELINE_NORMALIZE', False):
                    try:
                        baseline_data = epochs_clean.get_data()  # shape (n_epochs, n_ch, n_times)
                        channel_means = baseline_data.mean(axis=(0,2)).tolist()
                        channel_stds = baseline_data.std(axis=(0,2)).tolist()
                        baseline_fname = os.path.join(CACHE_DIR, f"sub{int(subject):02d}_baseline_stats.json")
                        with open(baseline_fname, 'w') as fh:
                            import json
                            json.dump({'subject': int(subject), 'means': channel_means, 'stds': channel_stds}, fh)
                        logger.log(f"Saved baseline stats: {baseline_fname}")
                    except Exception as e:
                        logger.log(f"Warning: Failed to save baseline stats: {e}")

                if subject is None:
                    cache_fname = f"anon_{task}_{condition}_epochs_epo.fif"
                else:
                    cache_fname = f"sub{int(subject):02d}_{task}_{condition}_epochs_epo.fif"
                cache_out = os.path.join(CACHE_DIR, cache_fname)
                epochs_clean.save(cache_out, overwrite=True, verbose=False)
                logger.log(f"Cached: {cache_out}")
                
                # Append to epoch_counts.csv for QC audit
                qc_counts_file = os.path.join(self.qc_folder, 'epoch_counts.csv')
                qc_row = {
                    'subject': subject,
                    'task': task,
                    'condition': condition,
                    'n_epochs': int(n_epochs_final),
                    'file_base': base
                }
                append_to_csv(qc_counts_file, qc_row)

            # 7) LOG CSV
            log_row = {
                'file_base': base,
                'task': task,
                'subject': subject,
                'duration_raw_s': round(dur_raw_s, 1),
                'n_epochs_initial': int(n_epochs_initial),
                'n_epochs_final': int(n_epochs_final),
                'autoreject_available': str(AUTOREJECT_AVAILABLE),
                'autoreject_applied': str(ar_applied),
                'n_dropped': int(n_dropped),
                'n_interpolated': int(n_interpolated),
                'yield_pct': round(yield_pct, 2),
                'final_snr_db': round(final_snr, 2) if not np.isnan(final_snr) else 0.0,
                'status': 'VALID' if yield_pct >= 20.0 else 'DROPPED'
            }
            append_to_csv(self.summary_file, log_row)

            logger.close()
            return out

        except Exception as e:
            logger.log(f"Stage3 failed: {e}\n{traceback.format_exc()}")
            logger.close()
            return None

    def process_all(self):
        pairs = self._match_files()
        print(f"Stage3: Processing {len(pairs)} pairs...")
        results = Parallel(n_jobs=self.n_jobs)(delayed(self._process_pair_inner)(cr, ic) for cr, ic in pairs)
        return [r for r in results if r]

# ---------------- Session QC ----------------

def compute_average_psd_over_data(data, sfreq, nperseg=None):
    signals = []
    for ep in data:
        for ch in ep:
            if np.isfinite(ch).any():
                signals.append(ch)
    if not signals:
        return None, None

    psds = []
    freqs = None
    for sig in signals:
        seg = nperseg if nperseg is not None else min(256, sig.size)
        if seg < 8:
            seg = max(8, sig.size)
        f, p = welch(sig, fs=sfreq, nperseg=min(seg, sig.size))
        if freqs is None:
            freqs = f
        psds.append(p)

    avg_psd = np.nanmean(np.vstack(psds), axis=0)
    return freqs, avg_psd

def compute_file_qc(fpath):
    record = {'file': os.path.basename(fpath)}
    if SKIP_TOKEN.lower() in record['file'].lower():
        record.update({'skipped': True, 'reason': f"name contains {SKIP_TOKEN}"})
        return record

    try:
        epochs = mne.read_epochs(fpath, preload=True, verbose=False)
    except Exception as e:
        record.update({'error': f'read_error: {e}'})
        return record

    info = epochs.info
    sfreq = info.get('sfreq', np.nan)
    ch_names = list(epochs.ch_names)
    bads = set(info.get('bads', []))
    good_ch_names = [ch for ch in ch_names if ch not in bads]

    n_epochs = len(epochs)
    try:
        data = epochs.get_data(copy=True)
    except Exception as e:
        record.update({'error': f'get_data_error: {e}'})
        return record

    if data.size == 0 or not np.isfinite(data).any():
        record.update({'error': 'no_data_or_all_nan'})
        return record

    flat = data.reshape(-1)
    max_abs = safe_nanmax(flat)
    percent_nan = 100.0 * np.isnan(flat).sum() / flat.size

    try:
        good_idx = [ch_names.index(ch) for ch in good_ch_names]
        ch_std_all = np.nanstd(data[:, good_idx, :], axis=(0, 2))
        n_flat_good = int(np.sum(ch_std_all < THRESH_FLAT_STD))
        pct_flat_good = 100.0 * n_flat_good / max(1, len(good_ch_names))
    except Exception:
        pct_flat_good = 0.0

    n_high = int(np.sum(np.abs(flat) > THRESH_AMP_V))
    pct_high = 100.0 * n_high / flat.size

    try:
        ptp_epochs = np.ptp(data, axis=2)
        ptp_median_uv = float(np.median(ptp_epochs) * 1e6)
    except Exception:
        ptp_median_uv = np.nan

    flag = 'PASS'
    notes = []
    if n_epochs < MIN_EPOCHS_QC:
        flag = 'REVIEW'
        notes.append('few_epochs')
    if pct_flat_good > 18:
        flag = 'REVIEW'
        notes.append('flat_channels')
    if pct_high > 2:
        flag = 'REVIEW'
        notes.append('many_high_amp')
    if max_abs is not None and np.isfinite(max_abs) and max_abs > THRESH_AMP_V:
        flag = 'REVIEW'
        notes.append('very_high_amplitude')

    record.update({
        'n_epochs': int(n_epochs),
        'sfreq': float(sfreq) if np.isfinite(sfreq) else np.nan,
        'max_abs_uv': float(max_abs) * 1e6 if np.isfinite(max_abs) else np.nan,
        'pct_nan': float(percent_nan),
        'ptp_median_uv': ptp_median_uv,
        'quality_flag': flag,
        'notes': ';'.join(notes)
    })
    return record

def run_session_qc(post_folder=STAGE3_OUT, qc_out_folder=QC_FOLDER):
    ensure_dir(qc_out_folder)
    fif_files = sorted(glob(os.path.join(post_folder, '*_AutoReject-epo.fif')))
    if not fif_files:
        print(f"No .fif files found in {post_folder} for QC.")
        return None

    print(f"Session QC over {len(fif_files)} files …")
    qc_records = Parallel(n_jobs=DEFAULT_N_JOBS)(delayed(compute_file_qc)(f) for f in fif_files)
    df = pd.DataFrame(qc_records)
    out_csv = os.path.join(qc_out_folder, 'session_qc_detailed_summary.csv')
    df.to_csv(out_csv, index=False)
    print(f"Session QC summary saved to {out_csv}")
    return out_csv

# ---------------- Main ----------------

def main():
    mne.set_log_level('WARNING')

    print("=" * 60)
    print("EEG PREPROCESSING PIPELINE (SIMPLIFIED 6S EPOCHS)")
    print("=" * 60)
    print(f"Epoch Duration: {EPOCH_DURATION_S}s | Overlap: Relaxed={EPOCH_OVERLAP_RELAXED*100:.0f}%, Attention={EPOCH_OVERLAP_ATTENTION*100:.0f}%")
    print("Artifact Rejection: AutoReject (epoch-level repair/drop)")

    # Safety Check
    if getattr(config, 'DEBUG_SAFE_MODE', False):
        assert config.EPOCH_OVERLAP_RELAXED == config.EPOCH_OVERLAP_ATTENTION == 0.0, \
            "DEBUG_SAFE_MODE: epoch overlap must be zero in both conditions to avoid confound."

    # Stage0
    stage0 = EEGPreprocessor(RAW_CSV_FOLDER, STAGE0_OUT, EEG_CHANNELS, SFREQ, n_jobs=DEFAULT_N_JOBS)
    stage0.process_all()

    # Stage1
    stage1 = EEGStage1(STAGE0_OUT, STAGE1_OUT, LOGS_FOLDER, QC_FOLDER, n_jobs=DEFAULT_N_JOBS, max_bad_channels=MAX_BAD_CHANNELS)
    stage1.process_all()

    # Stage2
    stage2 = EEGStage2(STAGE1_OUT, STAGE2_OUT, LOGS_FOLDER, QC_FOLDER, n_jobs=DEFAULT_N_JOBS)
    stage2.process_all()

    # Stage3
    stage3 = EEGStage3(STAGE1_OUT, STAGE2_OUT, STAGE3_OUT, LOGS_FOLDER, QC_FOLDER, n_jobs=DEFAULT_N_JOBS, min_epochs_for_ar=MIN_EPOCHS_FOR_AR)
    stage3.process_all()

    # QC
    print("\n--- Session QC ---")
    run_session_qc()

    print("\n--- Pipeline Complete ---")
    print(f"Final valid epochs available in: {FINAL_DATA_FOLDER}")

if __name__ == '__main__':
    main()
