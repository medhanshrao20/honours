"""
04_wasserstein_selection_exploratory.py

Stage 3b: Wasserstein Distance (Exploratory/Diagnostic)
-------------------------------------------
Computes Wasserstein 1-distance between 'Attention' and 'Relaxed' states
for each feature, using bootstrap resampling to estimate robustness.
Selects features based on modality invariance (low inter-modality distance)
and state divergence (high attention vs relaxed distance).

⚠️ ANALYSIS ONLY - Per classifier_modifications.md:
This script is for EXPLORATORY ANALYSIS and PAPER TABLES ONLY.
Do NOT use output CSVs (selected_features_wasserstein.csv) for training.
For training, use the WassersteinSelector transformer inside CV folds.

Reference: Redundancy_Pruning_and_Wasserstein_Distance_Guide.md
"""

import pandas as pd
import numpy as np
from scipy.stats import wasserstein_distance
import os
import sys
from itertools import combinations
import seaborn as sns
import warnings
import datetime
import json
import argparse

# ---------- CONFIG ----------
try:
    from importlib import import_module
    config = import_module('00_config')
    EXCLUDE_SUBJECTS = getattr(config, 'EXCLUDE_SUBJECTS_DOWNSTREAM', [])
    MIN_EPOCHS = getattr(config, 'WASSERSTEIN_MIN_EPOCHS', 5)
    N_BOOTSTRAP = getattr(config, 'WASSERSTEIN_BOOTSTRAPS', 200)
    EPSILON = getattr(config, 'WASSERSTEIN_EPSILON', 1e-9)
    SENSITIVITY_QUANTILE = getattr(config, 'SENSITIVITY_QUANTILE', 0.25)
    CONVERGENCE_RATIO_THRESHOLD = getattr(config, 'CONVERGENCE_RATIO_THRESHOLD', 1.0)
    RANDOM_SEED = getattr(config, 'RANDOM_SEED', 42)
    print(f"[Config] Loaded from 00_config.py (Bootstrap={N_BOOTSTRAP})")
except Exception as e:
    print(f"[Config] Fallback defaults: {e}")
    EXCLUDE_SUBJECTS = []
    MIN_EPOCHS = 5
    N_BOOTSTRAP = 200
    EPSILON = 1e-9
    SENSITIVITY_QUANTILE = 0.25
    CONVERGENCE_RATIO_THRESHOLD = 1.0
    RANDOM_SEED = 42

np.random.seed(RANDOM_SEED)

# Paths
# Paths
features_dir = getattr(config, 'FEATURES_DIR', 'Features')
biomarker_dir = getattr(config, 'BIOMARKER_DIR', 'BIOMARKER DISCOVERY')

INPUT_PATH = os.path.join(features_dir, "features_enhanced.parquet")
OUTPUT_DIR = os.path.join(biomarker_dir, "Wasserstein")
METADATA_COLS = ['file', 'task', 'subject', 'condition', 'epoch_idx', 'unique_id']


def load_data(filepath):
    """Load features from parquet or CSV."""
    print(f"Loading data: {filepath}")
    if filepath.endswith('.parquet'):
        df = pd.read_parquet(filepath)
    else:
        df = pd.read_csv(filepath)
    
    # Filter excluded subjects
    if 'subject' in df.columns and EXCLUDE_SUBJECTS:
        n_before = len(df)
        df = df[~df['subject'].isin(EXCLUDE_SUBJECTS)]
        n_after = len(df)
        print(f"Excluded subjects {EXCLUDE_SUBJECTS}: {n_before} -> {n_after} rows")
    
    return df


def normalize_features_subjectwise(df, feature_cols):
    """Z-score normalize features per subject."""
    print("Normalizing features per subject (Z-score)...")
    df_norm = df.copy()
    subjects = df['subject'].unique()
    
    for subj in subjects:
        subj_mask = df['subject'] == subj
        subset = df.loc[subj_mask, feature_cols]
        std = subset.std().replace(0, 1.0)
        df_norm.loc[subj_mask, feature_cols] = (subset - subset.mean()) / std
    
    return df_norm


def bootstrap_wasserstein(arr1, arr2, n_bootstrap=200, seed=None):
    """
    Compute bootstrap distribution of Wasserstein distance.
    Returns: (mean_dist, ci_low, ci_high)
    
    Args:
        arr1, arr2: Arrays to compute Wasserstein distance between
        n_bootstrap: Number of bootstrap iterations
        seed: Random seed for reproducibility (uses RANDOM_SEED if None)
    """
    if len(arr1) < 2 or len(arr2) < 2:
        warnings.warn("Wasserstein input arrays too short (<2 samples)")
        return np.nan, np.nan, np.nan
    
    if np.all(np.isnan(arr1)) or np.all(np.isnan(arr2)):
        warnings.warn("Wasserstein input contains all NaNs")
        return np.nan, np.nan, np.nan
    
    # Use reproducible RNG
    rng = np.random.default_rng(seed if seed is not None else RANDOM_SEED)
    
    dists = []
    for _ in range(n_bootstrap):
        idx1 = rng.choice(len(arr1), size=len(arr1), replace=True)
        idx2 = rng.choice(len(arr2), size=len(arr2), replace=True)
        w = wasserstein_distance(arr1[idx1], arr2[idx2])
        dists.append(w)
    
    return np.mean(dists), np.percentile(dists, 2.5), np.percentile(dists, 97.5)


def compute_distances(df, feature_cols):
    """
    Compute Wasserstein distances with bootstrap resampling.
    - Inter-modality distances (dynamic detection)
    - State divergence (attention vs relaxed)
    """
    print(f"Computing Wasserstein distances with {N_BOOTSTRAP} bootstrap iterations...")
    
    results = []
    results_per_subject = []
    subjects = df['subject'].unique()
    if len(subjects) < 5:
        raise ValueError("Need at least 5 subjects for reliable Wasserstein.")
    
    # Dynamic modality detection
    all_modalities = df[df['condition'] == 'attention']['task'].unique().tolist()
    print(f"Detected modalities: {all_modalities}")
    
    for feat_idx, feat in enumerate(feature_cols):
        if (feat_idx + 1) % 50 == 0:
            print(f"  Processing feature {feat_idx + 1}/{len(feature_cols)}...")
        
        w_modal_scores = []
        w_state_scores = []
        bootstrap_ratios = []
        
        for subj in subjects:
            s_data = df[df['subject'] == subj]
            att_data = s_data[s_data['condition'] == 'attention']
            rel_data = s_data[s_data['condition'] == 'relaxed'][feat].dropna().values
            
            # Extract modality data dynamically
            modality_data = {}
            for mod in all_modalities:
                vals = att_data[att_data['task'] == mod][feat].dropna().values
                if len(vals) >= MIN_EPOCHS:
                    modality_data[mod] = vals
            
            # --- 1. Inter-Modality Distances ---
            modal_dists = []
            modal_ci_lows = []
            modal_ci_highs = []
            for mod1, mod2 in combinations(modality_data.keys(), 2):
                arr1, arr2 = modality_data[mod1], modality_data[mod2]
                mean_d, ci_low, ci_high = bootstrap_wasserstein(arr1, arr2, N_BOOTSTRAP, seed=RANDOM_SEED + feat_idx)
                if not np.isnan(mean_d):
                    modal_dists.append(mean_d)
                    modal_ci_lows.append(ci_low)
                    modal_ci_highs.append(ci_high)
            
            w_modal = np.mean(modal_dists) if modal_dists else np.nan
            w_modal_ci_low = np.mean(modal_ci_lows) if modal_ci_lows else np.nan
            w_modal_ci_high = np.mean(modal_ci_highs) if modal_ci_highs else np.nan
            w_modal_scores.append(w_modal)
            
            # --- 2. State Divergence ---
            all_att = att_data[feat].dropna().values
            if len(all_att) >= MIN_EPOCHS and len(rel_data) >= MIN_EPOCHS:
                w_state, w_state_ci_low, w_state_ci_high = bootstrap_wasserstein(all_att, rel_data, N_BOOTSTRAP, seed=RANDOM_SEED + feat_idx)
            else:
                w_state = np.nan
                w_state_ci_low = np.nan
                w_state_ci_high = np.nan
            w_state_scores.append(w_state)
            
            # Compute ratio for this subject
            if not np.isnan(w_modal) and not np.isnan(w_state) and w_state > 0:
                ratio = w_modal / (w_state + EPSILON)
                bootstrap_ratios.append(ratio)
                
                results_per_subject.append({
                    'feature': feat,
                    'subject': subj,
                    'W_Modal': w_modal,
                    'W_Modal_CI_Low': w_modal_ci_low,
                    'W_Modal_CI_High': w_modal_ci_high,
                    'W_State': w_state,
                    'W_State_CI_Low': w_state_ci_low,
                    'W_State_CI_High': w_state_ci_high,
                    'Ratio': ratio
                })
        
        # Aggregate across subjects using MEDIAN (more robust)
        median_w_modal = np.nanmedian(w_modal_scores)
        median_w_state = np.nanmedian(w_state_scores)
        
        if np.isnan(median_w_modal) or np.isnan(median_w_state):
            continue
        
        # Compute ratio
        convergence_ratio = median_w_modal / (median_w_state + EPSILON)
        
        # Bootstrap p-value: fraction of subjects where ratio < 1
        n_valid = len(bootstrap_ratios)
        n_ratio_lt_1 = sum(1 for r in bootstrap_ratios if r < 1.0)
        p_value = n_ratio_lt_1 / n_valid if n_valid > 0 else 0.0
        
        results.append({
            'feature': feat,
            'W_Modal_Median': median_w_modal,
            'W_State_Median': median_w_state,
            'Convergence_Ratio': convergence_ratio,
            'Valid_Subjects': n_valid,
            'P_Value_RatioLT1': p_value,
            'Ratio_IQR': np.subtract(*np.percentile(bootstrap_ratios, [75, 25])) if bootstrap_ratios else np.nan
        })
    
    return pd.DataFrame(results), pd.DataFrame(results_per_subject)


def plot_distributions(df, features, out_dir, subdir="", max_plots=None):
    """Plot KDE distributions for selected features."""
    if max_plots is None:
        max_plots = len(features)
    print(f"Plotting distributions for {min(len(features), max_plots)} features...")
    plots_dir = os.path.join(out_dir, "Plots", subdir)
    os.makedirs(plots_dir, exist_ok=True)
    
    try:
        import matplotlib.pyplot as plt
        import seaborn as sns
    except ImportError:
        print("Warning: matplotlib/seaborn not available. Skipping plots.")
        return
    
    sns.set_theme(style="whitegrid")
    
    # FIX 1: Sort modalities to ensure consistent colors across all plots
    modalities = sorted(df[df['condition'] == 'attention']['task'].unique().tolist())
    colors = plt.cm.tab10.colors
    
    for feat in features[:max_plots]:
        try:
            plt.figure(figsize=(10, 6))
            
            # Plot each modality
            plotted = False
            for i, mod in enumerate(modalities):
                data = df[(df['condition'] == 'attention') & (df['task'] == mod)][feat].dropna()
                
                # FIX 2: Use MIN_EPOCHS directly (removed * 2 multiplier)
                if len(data) >= MIN_EPOCHS:
                    sns.kdeplot(
                        data, 
                        label=f'Attention ({mod.title()})', 
                        color=colors[i % len(colors)], 
                        fill=True, 
                        alpha=0.2,
                        common_norm=False  # Good practice for comparing distributions
                    )
                    plotted = True
            
            # Plot relaxed
            data_rel = df[df['condition'] == 'relaxed'][feat].dropna()
            
            # FIX 3: Use MIN_EPOCHS directly (removed * 2 multiplier)
            if len(data_rel) >= MIN_EPOCHS:
                sns.kdeplot(
                    data_rel, 
                    label='Relaxed (Baseline)', 
                    color='black',  # Changed to black for visibility
                    linestyle='--',
                    linewidth=1.5,
                    common_norm=False
                )
                plotted = True
            
            if plotted:
                plt.title(f"Feature: {feat}")
                plt.xlabel("Normalized Value (Z-score)")
                plt.ylabel("Density")
                plt.legend()
                plt.tight_layout()
                safe_name = str(feat).replace('/', '_').replace('\\', '_')[:50]
                plt.savefig(os.path.join(plots_dir, f"{safe_name}_dist.png"))
            
            plt.close()
        except Exception as e:
            print(f"  Skipping plot for {feat}: {e}")
            plt.close()


def main():
    parser = argparse.ArgumentParser(description="Wasserstein selection (exploratory) and plotting.")
    parser.add_argument("--plots-only", action="store_true",
                        help="Skip distance computation and only generate plots from existing CSVs.")
    args = parser.parse_args()

    try:
        base_dir = os.path.dirname(os.path.abspath(__file__))
    except NameError:
        base_dir = os.getcwd()

    # WARNING: This script produces a global ranking intended for discovery/figures
    # Do NOT use selected_features_wasserstein.csv as precomputed candidates for cross-validated classification
    # unless you explicitly re-run this script inside the LOSO training split and save with fold id.
    print("="*60)
    print("MODE: DIAGNOSTIC / DISCOVERY")
    print("Output CSVs are for global analysis only. Do NOT use for classification training.")
    print("="*60)
    
    input_path = os.path.join(base_dir, INPUT_PATH)
    out_dir = os.path.join(base_dir, OUTPUT_DIR)
    os.makedirs(out_dir, exist_ok=True)
    
    # Handle parquet/csv fallback
    if not os.path.exists(input_path):
        csv_path = input_path.replace('.parquet', '.csv')
        if os.path.exists(csv_path):
            input_path = csv_path
        else:
            print(f"Error: Input file not found at {input_path}")
            return
    
    df = load_data(input_path)
    
    # RISK 7: Log dropped/excluded subjects
    dropped_info = {
        'excluded_by_config': EXCLUDE_SUBJECTS,
        'n_subjects_after_load': int(df['subject'].nunique()) if not df.empty else 0,
        'timestamp': datetime.datetime.utcnow().isoformat()
    }
    with open(os.path.join(out_dir, "dropped_subjects.json"), "w") as f:
        json.dump(dropped_info, f, indent=2)
    
    # Detect feature columns
    feature_cols = [c for c in df.columns if c not in METADATA_COLS and pd.api.types.is_numeric_dtype(df[c])]
    # RISK 4: Explicitly exclude Riemannian features from biomarker discovery
    feature_cols = [c for c in feature_cols if not c.startswith('RIEM__')]
    print(f"Features (excluding RIEM__): {len(feature_cols)}")
    
    # Check for pruned feature list - MANDATORY for Biomarker Track
    pruned_list_path = os.path.join(base_dir, biomarker_dir, "Redundancy", "selected_features_redundancy_pruned.csv")
    if not os.path.exists(pruned_list_path):
        raise FileNotFoundError("Run 03_redundancy_pruning.py first for pruned features.")
    
    pruned_df = pd.read_csv(pruned_list_path)
    pruned_features = pruned_df['feature'].tolist() if 'feature' in pruned_df.columns else pruned_df.iloc[:, 0].tolist()
    available_pruned = [f for f in pruned_features if f in feature_cols]
    
    if not available_pruned:
        raise ValueError("No pruned features available.")

    feature_cols = available_pruned
    print(f"Using {len(feature_cols)} pruned features for Wasserstein analysis.")

    # Normalize once for plotting/computation
    df_norm = normalize_features_subjectwise(df, feature_cols)

    if args.plots_only:
        print("MODE: PLOTS-ONLY (no recomputation)")
        selected_path = os.path.join(out_dir, "selected_features_wasserstein.csv")
        full_path = os.path.join(out_dir, "wasserstein_full_metrics.csv")

        if not os.path.exists(selected_path):
            print(f"Error: Missing selected features file at {selected_path}")
            return

        selected_df = pd.read_csv(selected_path)
        selected_features = selected_df['feature'].tolist() if 'feature' in selected_df.columns else []

        if selected_features:
            plot_distributions(df_norm, selected_features, out_dir, subdir="Selected")
        else:
            print("No selected features found in CSV. Skipping Selected plots.")

        if os.path.exists(full_path):
            dist_df = pd.read_csv(full_path)
            if len(dist_df) > 0 and 'Convergence_Ratio' in dist_df.columns and 'feature' in dist_df.columns:
                high_ratio = dist_df.nlargest(3, 'Convergence_Ratio')['feature'].tolist()
                if high_ratio:
                    plot_distributions(df_norm, high_ratio, out_dir, subdir="Rejected_HighRatio")
            else:
                print("Full metrics CSV missing expected columns. Skipping Rejected plots.")
        else:
            print(f"Warning: Full metrics file not found at {full_path}. Skipping Rejected plots.")
        return

    print(f"\nAnalyzing {len(feature_cols)} features for {len(df['subject'].unique())} subjects...")

    # Compute distances
    dist_df, subject_df = compute_distances(df_norm, feature_cols)
    
    if dist_df.empty:
        print("Error: No valid distances computed. Saving empty selected features.")
        pd.DataFrame({'feature': []}).to_csv(os.path.join(out_dir, "selected_features_wasserstein.csv"), index=False)
        return

    # Early exit if no valid features
    if len(dist_df) == 0 or dist_df['Valid_Subjects'].sum() == 0:
        print("No features with sufficient valid subjects. Saving empty selected CSV.")
        pd.DataFrame({'feature': []}).to_csv(os.path.join(out_dir, "selected_features_wasserstein.csv"), index=False)
        return
    
    # Apply sensitivity floor
    state_threshold = dist_df['W_State_Median'].quantile(SENSITIVITY_QUANTILE)
    if np.isnan(state_threshold):
        state_threshold = 0.0
    
    print(f"\n{'='*60}")
    print("SELECTION CRITERIA:")
    print(f"  1. Sensitivity floor: W_State >= {state_threshold:.4f}")
    print(f"  2. Convergence ratio < {CONVERGENCE_RATIO_THRESHOLD}")
    print(f"{'='*60}")
    
    # Filter
    filtered_df = dist_df[dist_df['W_State_Median'] >= state_threshold].copy()
    filtered_df = filtered_df.sort_values('Convergence_Ratio', ascending=True)
    
    # Select features with ratio < threshold
    selected_df = filtered_df[filtered_df['Convergence_Ratio'] < CONVERGENCE_RATIO_THRESHOLD].copy()
    
    print(f"\nSelected {len(selected_df)} features (Ratio < {CONVERGENCE_RATIO_THRESHOLD})")
    
    if len(selected_df) == 0:
        print("Warning: No features passed strict threshold. Relaxing to top 20...")
        selected_df = filtered_df.head(20).copy()
    
    # Add metadata for safe usage (diagnostic only)
    meta_info = {
        'generation_mode': 'diagnostic_global',  # Explicit flag
        'generated_at': datetime.datetime.utcnow().isoformat(),
        'n_subjects': len(df['subject'].unique()),
        'n_features_analyzed': len(feature_cols),
        'note': 'DO NOT USE FOR CLASSIFICATION TRAINING - global discovery only',
        # Methodological Enforcement
        'intended_use': 'BIOMARKER_DISCOVERY_ONLY',
        'do_not_use_for_classification': True
    }
    for col, val in meta_info.items():
        selected_df[col] = val
        
    # Also add to full metrics if you want
    for col, val in meta_info.items():
        dist_df[col] = val
        
    # Save outputs
    full_path = os.path.join(out_dir, "wasserstein_full_metrics.csv")
    dist_df.sort_values('Convergence_Ratio').to_csv(full_path, index=False)
    
    print("\n" + "!"*60)
    print("WARNING: These outputs (wasserstein_*.csv) are for")
    print("BIOMARKER DISCOVERY ONLY. Do NOT use for classification training.")
    print("!"*60 + "\n")
    
    subj_path = os.path.join(out_dir, "wasserstein_subject_clustering_data.csv")
    subject_df.to_csv(subj_path, index=False)
    
    # Save subject counts (how many subjects had valid W_state > threshold per feature)
    # derived from filtered_df or dist_df
    counts_path = os.path.join(out_dir, "wasserstein_ranking_by_subject_count.csv")
    dist_df[['feature', 'Valid_Subjects', 'P_Value_RatioLT1']].sort_values('Valid_Subjects', ascending=False).to_csv(counts_path, index=False)

    selected_path = os.path.join(out_dir, "selected_features_wasserstein.csv")
    selected_df.to_csv(selected_path, index=False)
    
    print(f"\n[Saved] Full metrics: {full_path}")
    print(f"[Saved] Per-subject data: {subj_path}")
    print(f"[Saved] Selected features: {selected_path}")
    
    print(f"\n--- TOP 10 SELECTED FEATURES ---")
    print(selected_df[['feature', 'Convergence_Ratio', 'W_State_Median', 'P_Value_RatioLT1', 'Valid_Subjects']].head(10).to_string(index=False))
    print("\n[Biomarker Track] Output ready for 06_biomarker_discovery.py.")
    print("Output ready for 06_biomarker_discovery.py.")
    
    # Generate plots
    # Generate plots
    selected_features = selected_df['feature'].tolist()
    if selected_features:
        plot_distributions(df_norm, selected_features, out_dir, subdir="Selected")
    else:
        print("No features selected for plotting (empty after threshold). Skipping Selected plots.")
    
    # Plot rejected features for comparison
    if len(dist_df) > 0:
        high_ratio = dist_df.nlargest(3, 'Convergence_Ratio')['feature'].tolist()
        if high_ratio:
            plot_distributions(df_norm, high_ratio, out_dir, subdir="Rejected_HighRatio")
        else:
            print("No high-ratio features found. Skipping Rejected plots.")
    else:
        print("dist_df empty — no features to rank. Skipping all plots.")


if __name__ == "__main__":
    main()
