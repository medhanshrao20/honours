import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import balanced_accuracy_score
from sklearn.svm import SVC
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

import glob

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

# Add redundancy and wasserstein directly if paths exist
if os.path.exists(REDUNDANCY_PATH):
    FEATURE_SETS["REDUNDANCY"] = REDUNDANCY_PATH
if os.path.exists(WASSERSTEIN_PATH):
    FEATURE_SETS["WASSERSTEIN"] = WASSERSTEIN_PATH

# Filter out any sets where the file wasn't found
FEATURE_SETS = {k: v for k, v in FEATURE_SETS.items() if v is not None}
intersection = os.path.join(VALIDATION_DIR, 'final_robust_intersection.csv')
if os.path.exists(intersection):
    FEATURE_SETS['FINAL'] = intersection

OUT_DIR = "BIOMARKER DISCOVERY/Invariance_Classification_Final"
PLOT_DIR = os.path.join(OUT_DIR, "Plots")
os.makedirs(PLOT_DIR, exist_ok=True)

USE_LOMO = True          # True = LOSO+LOMO, False = LOSO only
MIN_TRAIN = 10
MIN_TEST = 5
RANDOM_SEED = 42

LABEL_MAP = {"attention": 1, "relaxed": 0}

# =========================
# LOGGING
# =========================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(os.path.join(OUT_DIR, "run.log")),
        logging.StreamHandler()
    ]
)
LOG = logging.getLogger(__name__)

# =========================
# LOAD DATA
# =========================
LOG.info("Loading features")
df = pd.read_parquet(FEATURE_FILE)

SUBJECTS = sorted(df["subject"].unique())
TASKS = sorted(df[df["condition"] == "attention"]["task"].unique())

LOG.info("Subjects (%d): %s", len(SUBJECTS), SUBJECTS)
LOG.info("Tasks: %s", TASKS)

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
    LOG.info("Loaded %s: %d features", name, len(feature_sets[name]))

# =========================
# MODELS
# =========================
MODELS = {
    "SVM": SVC(kernel="rbf", C=1.0, class_weight="balanced", cache_size=500),
    "LDA": LinearDiscriminantAnalysis(solver="svd")
}

# =========================
# MAIN LOOP
# =========================
results = []

for fs_name, fs_features in feature_sets.items():
    LOG.info("=" * 80)
    LOG.info("FEATURE SET: %s", fs_name)

    for test_subject in SUBJECTS:
        df_test_subj = df[df["subject"] == test_subject]
        df_train_subj = df[df["subject"] != test_subject]

        for task in TASKS:
            # LOMO
            if USE_LOMO:
                df_train = df_train_subj[df_train_subj["task"] != task]
            else:
                df_train = df_train_subj.copy()

            df_test = df_test_subj[df_test_subj["task"] == task]

            if len(df_train) < MIN_TRAIN or len(df_test) < MIN_TEST:
                continue

            y_train = df_train["condition"].map(LABEL_MAP).values
            y_test = df_test["condition"].map(LABEL_MAP).values
            if len(np.unique(y_train)) < 2:
                continue
            if len(np.unique(y_test)) < 2:
                continue

            X_train = df_train[fs_features].values
            X_test = df_test[fs_features].values

            scaler = StandardScaler()
            X_train = scaler.fit_transform(X_train)
            X_test = scaler.transform(X_test)

            for model_name, model in MODELS.items():
                try:
                    model.fit(X_train, y_train)
                    y_pred = model.predict(X_test)
                    ba = balanced_accuracy_score(y_test, y_pred)

                    results.append({
                        "feature_set": fs_name,
                        "subject": test_subject,
                        "modality": task,
                        "model": model_name,
                        "balanced_accuracy": ba,
                        "n_train": len(df_train),
                        "n_test": len(df_test),
                    })

                    LOG.info(
                        "[%s] subj=%s task=%s model=%s BA=%.3f",
                        fs_name, test_subject, task, model_name, ba
                    )
                except Exception as e:
                    LOG.error(
                        "Model crash | fs=%s subj=%s task=%s model=%s error=%s",
                        fs_name, test_subject, task, model_name, str(e)
                    )
                    continue

# =========================
# SAVE RAW RESULTS
# =========================
res_df = pd.DataFrame(results)

raw_csv = os.path.join(OUT_DIR, "loso_lomo_fixed_feature_results.csv")
res_df.to_csv(raw_csv, index=False)
LOG.info("Saved raw fold-level results: %s", raw_csv)

# =========================
# PAPER-READY TABLES
# =========================

# Table 1: Feature set × Model (mean ± SD)
summary_rows = []
for (fs, model), g in res_df.groupby(["feature_set", "model"]):
    summary_rows.append({
        "feature_set": fs,
        "model": model,
        "mean_balanced_accuracy": g["balanced_accuracy"].mean(),
        "std_balanced_accuracy": g["balanced_accuracy"].std(),
        "n_folds": len(g),
    })

summary_df = pd.DataFrame(summary_rows)
summary_csv = os.path.join(
    OUT_DIR, "table_results_mean_std_by_feature_and_model.csv"
)
summary_df.to_csv(summary_csv, index=False)
LOG.info("Saved summary table: %s", summary_csv)

# Table 2: Feature-set only
fs_summary = (
    res_df
    .groupby("feature_set")
    .agg(
        mean_balanced_accuracy=("balanced_accuracy", "mean"),
        std_balanced_accuracy=("balanced_accuracy", "std"),
        n_folds=("balanced_accuracy", "count")
    )
    .reset_index()
)

fs_summary_csv = os.path.join(
    OUT_DIR, "table_results_by_feature_set.csv"
)
fs_summary.to_csv(fs_summary_csv, index=False)
LOG.info("Saved feature-set summary: %s", fs_summary_csv)

# Per-subject table (supplement)
subject_summary = (
    res_df
    .groupby(["feature_set", "model", "subject"])
    .agg(
        mean_balanced_accuracy=("balanced_accuracy", "mean"),
        std_balanced_accuracy=("balanced_accuracy", "std"),
        n_folds=("balanced_accuracy", "count")
    )
    .reset_index()
)

subject_csv = os.path.join(
    OUT_DIR, "per_subject_results.csv"
)
subject_summary.to_csv(subject_csv, index=False)
LOG.info("Saved per-subject summary: %s", subject_csv)

# Plot-ready CSV
plot_csv = os.path.join(
    OUT_DIR, "plot_ready_balanced_accuracy.csv"
)
res_df[
    ["feature_set", "model", "subject", "modality", "balanced_accuracy"]
].to_csv(plot_csv, index=False)
LOG.info("Saved plot-ready CSV: %s", plot_csv)

# =========================
# PLOTS (PUBLICATION READY)
# =========================
sns.set(style="whitegrid", context="paper")

# 1. Boxplot: Feature set comparison
plt.figure(figsize=(6, 4))
sns.boxplot(
    data=res_df,
    x="feature_set",
    y="balanced_accuracy"
)
plt.axhline(0.5, linestyle="--", color="gray", linewidth=1)
plt.ylabel("Balanced Accuracy")
plt.xlabel("Feature Set")
plt.title("LOSO+LOMO Robustness Across Feature Sets")
plt.tight_layout()
plt.savefig(os.path.join(PLOT_DIR, "boxplot_feature_sets.png"), dpi=300)
plt.close()

# 2. Boxplot: Model comparison
plt.figure(figsize=(6, 4))
sns.boxplot(
    data=res_df,
    x="model",
    y="balanced_accuracy"
)
plt.axhline(0.5, linestyle="--", color="gray", linewidth=1)
plt.ylabel("Balanced Accuracy")
plt.xlabel("Classifier")
plt.title("Classifier Comparison (All Feature Sets)")
plt.tight_layout()
plt.savefig(os.path.join(PLOT_DIR, "boxplot_models.png"), dpi=300)
plt.close()

# 3. Feature set × Model interaction
plt.figure(figsize=(7, 4))
sns.boxplot(
    data=res_df,
    x="feature_set",
    y="balanced_accuracy",
    hue="model"
)
plt.axhline(0.5, linestyle="--", color="gray", linewidth=1)
plt.ylabel("Balanced Accuracy")
plt.xlabel("Feature Set")
plt.legend(title="Model")
plt.tight_layout()
plt.savefig(os.path.join(PLOT_DIR, "boxplot_feature_by_model.png"), dpi=300)
plt.close()

# 4. Per-subject distribution (FINAL ROBUST only)
plt.figure(figsize=(7, 4))
sns.boxplot(
    data=res_df[res_df["feature_set"] == "FINAL"],
    x="subject",
    y="balanced_accuracy",
    hue="model"
)
plt.axhline(0.5, linestyle="--", color="gray", linewidth=1)
plt.ylabel("Balanced Accuracy")
plt.xlabel("Subject")
plt.title("Subject-wise Performance (Final Robust Biomarkers)")
plt.legend(title="Model", bbox_to_anchor=(1.02, 1), loc="upper left")
plt.tight_layout()
plt.savefig(os.path.join(PLOT_DIR, "subjectwise_final_robust.png"), dpi=300)
plt.close()

LOG.info("=" * 80)
LOG.info("DONE")
LOG.info("All results and plots saved to %s", OUT_DIR)
LOG.info("Subjects covered: %d / %d", res_df["subject"].nunique(), len(SUBJECTS))
