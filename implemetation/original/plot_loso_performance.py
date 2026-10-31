import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Create output directory for the plot
OUT_DIR = r"BIOMARKER DISCOVERY/Classifier_Training/TrainingLogs"
os.makedirs(OUT_DIR, exist_ok=True)

# Load data
CSV_PATH = os.path.join(OUT_DIR, "loso_summary.csv")
df = pd.read_csv(CSV_PATH)

# Filter for requested models only
MODELS_TO_PLOT = ["SVM", "LDA", "MLP"]
df_filtered = df[df["model"].isin(MODELS_TO_PLOT)].copy()

# Set up the aesthetic
sns.set_theme(style="whitegrid", palette="viridis")
plt.figure(figsize=(12, 7))

# Create the boxplot
ax = sns.boxplot(
    data=df_filtered,
    x="feature_set",
    y="balanced_acc",
    hue="model",
    palette="muted",
    showmeans=True,
    meanprops={"marker":"o", "markerfacecolor":"white", "markeredgecolor":"black", "markersize":"8"}
)

# Add overlay points for better granularity
sns.stripplot(
    data=df_filtered,
    x="feature_set",
    y="balanced_acc",
    hue="model",
    dodge=True,
    alpha=0.3,
    palette="dark",
    legend=False
)

# Title and labels
plt.title("LOSO Classification Performance: SVM vs LDA vs MLP", fontsize=16, fontweight='bold')
plt.xlabel("Feature Set", fontsize=14)
plt.ylabel("Balanced Accuracy", fontsize=14)
plt.ylim(0.3, 1.0)  # Standard range for comparison
plt.axhline(0.5, color='red', linestyle='--', alpha=0.5, label="Chance Level (0.5)")

# Customize legend
plt.legend(title="Model", title_fontsize='13', fontsize='11', loc='upper right')

plt.tight_layout()

# Save the plot
PLOT_PATH = os.path.join(OUT_DIR, "loso_performance_boxplot.png")
plt.savefig(PLOT_PATH, dpi=300)
print(f"Plot saved to: {PLOT_PATH}")
