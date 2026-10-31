"""
08_task_independence_check.py

Negative control:
Task / modality classification using FIXED attention biomarkers.

Goal:
- Test whether attention biomarkers encode task identity.
- Desired result: near-chance accuracy (task-invariant).

NO feature selection
NO hyperparameter tuning
NO relaxed condition
"""

import os
import logging
import glob
import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import balanced_accuracy_score
from sklearn.svm import SVC
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

# =========================
# CONFIG
# =========================
FEATURE_FILE = "Features/features_enhanced.parquet"
VALIDATION_DIR = "BIOMARKER DISCOVERY/Validation"

def get_latest_file(prefix, directory=VALIDATION_DIR):
    """Finds the latest CSV file with a given prefix in a directory."""
    files = glob.glob(os.path.join(directory, f"{prefix}*.csv"))
    if not files:
        return None
    return max(files, key=os.path.getmtime)

# Dynamically resolve feature set paths
FEATURE_SETS = {
    "FDR_ONLY": get_latest_file("fdr_only_features_hboot"),
    "FDR_SURVIVAL": get_latest_file("fdr_survival_features_hboot"),
    "FDR_EFFECT": get_latest_file("fdr_effect_features_hboot"),
    "FINAL": get_latest_file("final_selected_features_hboot"),
}

# Static paths for exploratory sets
REDUNDANCY_PATH = "BIOMARKER DISCOVERY/Redundancy/selected_features_redundancy_pruned.csv"
WASSERSTEIN_PATH = "BIOMARKER DISCOVERY/Wasserstein/selected_features_wasserstein.csv"

if os.path.exists(REDUNDANCY_PATH):
    FEATURE_SETS["REDUNDANCY"] = REDUNDANCY_PATH
if os.path.exists(WASSERSTEIN_PATH):
    FEATURE_SETS["WASSERSTEIN"] = WASSERSTEIN_PATH

# Filter out empty
FEATURE_SETS = {k: v for k, v in FEATURE_SETS.items() if v is not None}
intersection = os.path.join(VALIDATION_DIR, 'final_robust_intersection.csv')
if os.path.exists(intersection):
    FEATURE_SETS['FINAL'] = intersection

OUT_DIR = "BIOMARKER DISCOVERY/Task_Independence"
os.makedirs(OUT_DIR, exist_ok=True)

MIN_TRAIN = 15
MIN_TEST = 5
RANDOM_SEED = 42

# =========================
# LOGGING
# =========================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(os.path.join(OUT_DIR, "task_independence.log")),
        logging.StreamHandler()
    ]
)
LOG = logging.getLogger(__name__)

# =========================
# LOAD DATA
# =========================
df = pd.read_parquet(FEATURE_FILE)

# Use ATTENTION ONLY
df = df[df["condition"] == "attention"].reset_index(drop=True)

SUBJECTS = sorted(df["subject"].unique())
TASKS = sorted(df["task"].unique())

TASK_MAP = {t: i for i, t in enumerate(TASKS)}
LOG.info("Tasks: %s", TASK_MAP)

# =========================
# LOAD FEATURE SETS
# =========================
feature_sets = {}
for name, path in FEATURE_SETS.items():
    fs = pd.read_csv(path)
    if fs.empty:
        LOG.warning('Skipping empty feature set: %s', name)
        continue
    feature_sets[name] = fs["feature"].tolist()
    LOG.info("Loaded %s (%d features)", name, len(feature_sets[name]))

# =========================
# MODELS
# =========================
MODELS = {
    "SVM": SVC(kernel="rbf", C=1.0, class_weight="balanced", random_state=RANDOM_SEED),
    "LDA": LinearDiscriminantAnalysis()
}

# =========================
# LOSO TASK CLASSIFICATION
# =========================
results = []

for fs_name, features in feature_sets.items():
    LOG.info("=" * 70)
    LOG.info("FEATURE SET: %s", fs_name)

    for test_subject in SUBJECTS:
        df_test = df[df["subject"] == test_subject]
        df_train = df[df["subject"] != test_subject]

        if len(df_train) < MIN_TRAIN or len(df_test) < MIN_TEST:
            continue

        y_train = df_train["task"].map(TASK_MAP).values
        y_test = df_test["task"].map(TASK_MAP).values

        if len(np.unique(y_train)) < 2:
            continue

        X_train = df_train[features].values
        X_test = df_test[features].values

        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        for model_name, model in MODELS.items():
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)

            ba = balanced_accuracy_score(y_test, y_pred)

            results.append({
                "feature_set": fs_name,
                "subject": test_subject,
                "model": model_name,
                "balanced_accuracy": ba
            })

            LOG.info(
                "[%s] subj=%s model=%s task-BA=%.3f",
                fs_name, test_subject, model_name, ba
            )

# =========================
# SAVE RESULTS
# =========================
res_df = pd.DataFrame(results)
res_df.to_csv(
    os.path.join(OUT_DIR, "task_independence_results.csv"),
    index=False
)

LOG.info("Saved task-independence results")
LOG.info("DONE")
