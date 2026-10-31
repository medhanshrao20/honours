"""Senior configuration with portable paths and explicit paper/senior profiles."""
from original_config import *
from pathlib import Path

_HERE = Path(__file__).resolve().parent
RUN_PROFILE = os.environ.get('EEG_PROFILE','paper')
RUN_DIR = str(Path(os.environ.get('EEG_RUN_DIR',str(_HERE/'runs'/RUN_PROFILE/'full'))).resolve())
RAW_CSV_FOLDER = str(_HERE.parent/'Honors'/'Attention_EEG')
PREPROCESSED_DIR = os.path.join(RUN_DIR,'Preprocessed')
FEATURES_DIR = os.path.join(RUN_DIR,'Features')
BIOMARKER_DIR = os.path.join(RUN_DIR,'BIOMARKER DISCOVERY')
STAGE0_OUT = os.path.join(PREPROCESSED_DIR,'EEGData')
STAGE1_OUT = os.path.join(PREPROCESSED_DIR,'PreICA')
STAGE2_OUT = os.path.join(PREPROCESSED_DIR,'ICA')
STAGE3_OUT = os.path.join(PREPROCESSED_DIR,'AutoReject')
FINAL_DATA_FOLDER = os.path.join(PREPROCESSED_DIR,'FinalData')
LOGS_FOLDER = os.path.join(PREPROCESSED_DIR,'Logs')
QC_FOLDER = os.path.join(PREPROCESSED_DIR,'QC')
CACHE_DIR = os.path.join(RUN_DIR,'cache','epochs')
FEATURE_OUT_FOLDER = FEATURES_DIR
FEATURE_OUT_FILE = os.path.join(FEATURES_DIR,'features_enhanced.parquet')
RIEMANN_COV_OUT = os.path.join(FEATURES_DIR,'covariances.npz')
EXCLUDE_SUBJECTS_DOWNSTREAM = EXCLUDE_SUBJECTS_ANALYSIS
DEFAULT_N_JOBS = int(os.environ.get('EEG_JOBS','1'))
if RUN_PROFILE=='paper':
    WASSERSTEIN_EPSILON = 1e-6
    PRUNING_STRATEGY = 'fixed'
    FIXED_PRUNING_THRESHOLD = 0.20
SMOKE_MODE = os.environ.get('EEG_SMOKE','0')=='1'
if SMOKE_MODE:
    WASSERSTEIN_BOOTSTRAPS = 5
    N_BOOT_FINAL = N_BOOT_SMOKE = HIER_BOOTSTRAP_REPS = 20
    LOSO_RESAMPLES = None
