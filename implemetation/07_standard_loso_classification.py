#!/usr/bin/env python3
"""
LOSO Classification with Fold-Local Redundancy Pruning and Wasserstein Ranking

This script performs Leave-One-Subject-Out (LOSO) classification, applying redundancy pruning and Wasserstein-based
feature ranking on the training set of each fold, exactly as in 03_redundancy_pruning_exploratory.py and 
04_wasserstein_selection_exploratory.py. Subjects 9 and 10 are excluded completely.

Process:
- Exclude subjects 9 and 10.
- LOSO over remaining subjects (one test subject per fold).
- On training set (remaining subjects): 
  - Apply redundancy pruning as in 03 (baseline normalization using 'relaxed', hierarchical clustering, medoid selection).
  - On pruned features, apply Wasserstein ranking as in 04 (subject-wise z-score, bootstrap Wasserstein distances).
- Evaluate feature sets: 'PRUNED', 'WASS_TOP10', 'WASS_TOP50PCT', 'WASS_ALL'.
- For each feature set, train ML models and evaluate on test subject's all epochs (including relaxed).
- Save per-fold artifacts, plots, and master summary CSV.

Usage:
    python this_script.py --features Features/features_enhanced.parquet --out TrainingLogs --no-zip
"""

import os
import sys
import time
import json
import shutil
import argparse
import warnings
from datetime import datetime
from itertools import combinations
from collections import defaultdict

import numpy as np
import pandas as pd
from scipy.stats import wasserstein_distance, spearmanr
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform
import scipy.stats
import scipy.cluster.hierarchy as sch
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.metrics import balanced_accuracy_score, confusion_matrix, classification_report, roc_auc_score
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.base import clone
from sklearn.utils import resample
import joblib

# Constants
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
EXCLUDE_SUBJECTS = [9, 10]  # Excluded completely as per instructions
META_COLS = ['unique_id', 'file', 'task', 'subject', 'condition', 'epoch_idx', 'target', 'label', 'channel']
DEFAULT_FEATURES_PATH = os.path.join("Features", "features_enhanced.parquet")
DEFAULT_OUT_DIR = os.path.join("BIOMARKER DISCOVERY", "Classifier_Training", "TrainingLogs")

# Redundancy Pruning Configs (from 03)
CORR_CLUSTER_THRESH = 0.30
MIN_FEATURES_AFTER_PRUNING = 30
PRUNING_THRESH_RANGE = (0.05, 0.5)
PRUNING_THRESH_STEP = 0.01
PRUNING_STRATEGY = 'adaptive'

# Wasserstein Configs (from 04)
WASS_MIN_EPOCHS = 5
WASS_N_BOOTSTRAP = 200
WASS_EPSILON = 1e-9
WASS_SENSITIVITY_QUANTILE = 0.25
WASS_CONVERGENCE_RATIO_THRESHOLD = 1.0

# ML Models
ALGORITHMS = [
    ('SVM', SVC(kernel='rbf', probability=True, random_state=RANDOM_SEED)),
    ('RF', RandomForestClassifier(n_estimators=100, random_state=RANDOM_SEED)),
    ('LDA', LinearDiscriminantAnalysis()),
    ('MLP', MLPClassifier(hidden_layer_sizes=(100,), max_iter=500, random_state=RANDOM_SEED)),
    ('KNN', KNeighborsClassifier(n_neighbors=5)),
]

# ---------------------------- Utility Functions ----------------------------

def print_progress(msg):
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {msg}")

def load_features(path):
    print_progress(f"Loading features from {path}...")
    if path.endswith(".parquet"):
        df = pd.read_parquet(path)
    else:
        df = pd.read_csv(path)
    # Add target if not present
    if 'target' not in df.columns:
        df['target'] = (df['condition'] == 'attention').astype(int)
    return df

def plot_cm(cm, labels, outpath, title="Confusion matrix"):
    fig, ax = plt.subplots(figsize=(4,4))
    im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    ax.set_title(title)
    ticks = np.arange(len(labels))
    ax.set_xticks(ticks); ax.set_yticks(ticks)
    ax.set_xticklabels(labels); ax.set_yticklabels(labels)
    plt.xlabel("Predicted"); plt.ylabel("True")
    thresh = cm.max() / 2. if cm.max() > 0 else 0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, int(cm[i,j]), ha="center", va="center",
                    color="white" if cm[i,j] > thresh else "black")
    fig.tight_layout()
    fig.savefig(outpath)
    plt.close(fig)

def prepare_wide_features(df):
    print_progress("Preparing wide format (Samples x Features)...")
    if 'channel' in df.columns:
        print_progress("Pivoting narrow format...")
        index_cols = [c for c in META_COLS if c in df.columns and c != 'channel']
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        value_cols = [c for c in numeric_cols if c not in META_COLS]
        if not value_cols:
            raise ValueError("No numeric feature columns found.")
        if 'unique_id' not in df.columns:
            df['unique_id'] = df['file'].astype(str) + '_' + df['epoch_idx'].astype(str)
            index_cols = list(set(index_cols + ['unique_id']))
        pv = df.set_index(index_cols + ['channel'])[value_cols].unstack(level='channel')
        pv.columns = ['{}_{}'.format(ch, feat) for feat, ch in pv.columns]
        pv = pv.reset_index()
        meta_df = pv[[c for c in pv.columns if c in META_COLS]]
        feature_df = pv.select_dtypes(include=[np.number])
        feature_df = feature_df.drop(columns=[c for c in feature_df.columns if c in META_COLS], errors='ignore')
        # Filter out Riemannian/covariance features (anything with 'riem' or 'cov')
        feature_df = feature_df[[c for c in feature_df.columns if not ('riem' in c.lower() or 'cov' in c.lower())]]
        print_progress(f"Pivoted to {feature_df.shape[1]} features (Covariance/RIEM excluded)")
        return feature_df, meta_df
    else:
        print_progress("Already wide format.")
        meta_df = df[[c for c in df.columns if c in META_COLS]]
        feature_df = df.select_dtypes(include=[np.number])
        feature_df = feature_df.drop(columns=[c for c in feature_df.columns if c in META_COLS], errors='ignore')
        # Filter out Riemannian/covariance features (anything with 'riem' or 'cov')
        feature_df = feature_df[[c for c in feature_df.columns if not ('riem' in c.lower() or 'cov' in c.lower())]]
        print_progress(f"Features: {feature_df.shape[1]} (Covariance/RIEM excluded)")
        return feature_df, meta_df

def baseline_normalize(df_features, meta_df):
    print_progress("Applying subject-wise baseline normalization (using 'relaxed')...")
    df_norm = df_features.copy()
    df_norm = df_norm.reset_index(drop=True)
    meta_df = meta_df.reset_index(drop=True)
    feature_cols = df_features.columns
    subjects = meta_df['subject'].unique()
    for subj in subjects:
        subj_mask = meta_df['subject'] == subj
        baseline_mask = subj_mask & (meta_df['condition'] == 'relaxed')
        if os.environ.get('EEG_PROFILE') == 'paper':
            baseline_mask = subj_mask
        idx = meta_df.index[subj_mask]
        if len(idx) < 2:
            continue
        if not baseline_mask.any():
            print_progress(f"  No relaxed for {subj}; fallback z-score.")
            subset = df_features.loc[idx, feature_cols]
            mean = subset.mean()
            std = subset.std().replace(0, 1.0)
            df_norm.loc[idx, feature_cols] = (subset - mean) / std
            continue
        baseline_data = df_features.loc[meta_df.index[baseline_mask]]
        mu = baseline_data.mean()
        sigma = baseline_data.std().replace(0, 1.0)
        df_norm.loc[idx, feature_cols] = (df_features.loc[idx] - mu) / sigma
    return df_norm

def compute_correlation_distance(df_features):
    print_progress("Computing Spearman correlation matrix...")
    corr_matrix = df_features.corr(method='spearman')
    dist_matrix = 1 - np.abs(corr_matrix)
    dist_matrix = np.clip(dist_matrix, 0, 2)
    np.fill_diagonal(dist_matrix.values, 0.0)
    if dist_matrix.isnull().values.any():
        print_progress("Warning: NaNs in distance matrix; filling with 1.0.")
        dist_matrix = dist_matrix.fillna(1.0)
    return corr_matrix, dist_matrix

def perform_clustering(dist_matrix):
    print_progress("Performing hierarchical clustering (Ward)...")
    dist_array = dist_matrix.values.astype(np.float64)
    condensed_dist = squareform(dist_array)
    linkage_matrix = linkage(condensed_dist, method='ward')
    return linkage_matrix

def find_optimal_threshold(linkage_matrix, n_features, min_features=MIN_FEATURES_AFTER_PRUNING, strategy=PRUNING_STRATEGY):
    if os.environ.get('EEG_PROFILE') == 'paper':
        return 0.20
    print_progress(f"Finding optimal threshold (strategy={strategy}, min_features={min_features})...")
    best_t = CORR_CLUSTER_THRESH
    for t in np.arange(PRUNING_THRESH_RANGE[0], PRUNING_THRESH_RANGE[1], PRUNING_THRESH_STEP):
        clusters = fcluster(linkage_matrix, t, criterion='distance')
        n_clusters = len(np.unique(clusters))
        if n_clusters >= min_features:
            best_t = t
            print_progress(f"  Threshold {t:.2f} -> {n_clusters} clusters")
            break
    else:
        best_t = PRUNING_THRESH_RANGE[0]
        clusters = fcluster(linkage_matrix, best_t, criterion='distance')
        n_clusters = len(np.unique(clusters))
        print_progress(f"  Warning: Using t={best_t:.2f} -> {n_clusters} clusters")
    return best_t

def select_medoids(df_features, clusters, corr_matrix):
    print_progress("Selecting medoids...")
    features = df_features.columns.values
    unique_clusters = np.unique(clusters)
    selected_features = []
    cluster_info = []
    dist_vals = 1 - np.abs(corr_matrix.values)
    for clust_id in unique_clusters:
        indices = np.where(clusters == clust_id)[0]
        if len(indices) == 1:
            medoid_idx = indices[0]
            avg_dist = 0.0
        else:
            sub_dist = dist_vals[np.ix_(indices, indices)]
            sum_dists = np.sum(sub_dist, axis=1)
            medoid_local_idx = np.argmin(sum_dists)
            medoid_idx = indices[medoid_local_idx]
            avg_dist = np.mean(sub_dist[medoid_local_idx, :])
        selected_features.append(features[medoid_idx])
        cluster_info.append({
            'cluster': int(clust_id),
            'medoid': features[medoid_idx],
            'size': len(indices),
            'members': list(features[indices]),
            'distance_mean': avg_dist
        })
    return selected_features, cluster_info

def plot_dendrogram(linkage_matrix, threshold, out_path):
    print_progress("Plotting dendrogram...")
    plt.figure(figsize=(15, 8))
    sch.dendrogram(
        linkage_matrix,
        truncate_mode='lastp',
        p=50,
        show_leaf_counts=True,
        leaf_rotation=90.,
        leaf_font_size=10.,
        color_threshold=threshold
    )
    plt.axhline(y=threshold, color='r', linestyle='--', label=f'Threshold: {threshold:.2f}')
    plt.title('Hierarchical Clustering Dendrogram (Top 50 Clusters)')
    plt.xlabel('Cluster Size / Feature')
    plt.ylabel('Distance (1 - |Spearman|)')
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()
    print_progress(f"Dendrogram saved to {out_path}")

def redundancy_prune(train_df, fold_dir):
    print_progress("Performing redundancy pruning on train set...")
    feature_df, meta_df = prepare_wide_features(train_df)
    if feature_df.empty:
        raise ValueError("Failed to prepare features.")
    if feature_df.isnull().values.any():
        print_progress("Filling NaNs with mean.")
        feature_df = feature_df.fillna(feature_df.mean()).dropna(axis=1)
    all_cols = feature_df.columns
    feature_norm = baseline_normalize(feature_df, meta_df)
    variances = feature_norm.var()
    constant_feats = variances[variances < 1e-12].index
    if len(constant_feats) > 0:
        print_progress(f"Dropping {len(constant_feats)} constant features.")
        feature_norm = feature_norm.drop(columns=constant_feats)
    corr_matrix, dist_matrix = compute_correlation_distance(feature_norm)
    linkage_matrix = perform_clustering(dist_matrix)
    optimal_thresh = find_optimal_threshold(linkage_matrix, len(feature_norm.columns))
    clusters = fcluster(linkage_matrix, optimal_thresh, criterion='distance')
    selected_features, cluster_info = select_medoids(feature_norm, clusters, corr_matrix)
    n_total = len(feature_norm.columns)
    n_selected = len(selected_features)
    reduction = (1 - n_selected / n_total) * 100 if n_total > 0 else 0
    print_progress(f"Pruned: {n_total} -> {n_selected} ({reduction:.2f}% reduction)")
    # Save artifacts
    pd.Series(selected_features).to_csv(os.path.join(fold_dir, 'pruned_features.csv'), index=False)
    pd.DataFrame(cluster_info).to_csv(os.path.join(fold_dir, 'pruning_cluster_info.csv'), index=False)
    summary = {
        'original_count': n_total,
        'selected_count': n_selected,
        'reduction_pct': round(reduction, 2),
        'threshold': optimal_thresh,
        'strategy': PRUNING_STRATEGY,
        'n_clusters': len(np.unique(clusters))
    }
    with open(os.path.join(fold_dir, 'pruning_summary.json'), 'w') as f:
        json.dump(summary, f, indent=2)
    plot_dendrogram(linkage_matrix, optimal_thresh, os.path.join(fold_dir, 'dendrogram.png'))
    return selected_features

def normalize_features_subjectwise(df, feature_cols):
    print_progress("Subject-wise z-score normalization for Wasserstein...")
    df_norm = df.copy()
    for subj in df_norm['subject'].unique():
        mask = df_norm['subject'] == subj
        subset = df_norm.loc[mask, feature_cols]
        mean = subset.mean()
        std = subset.std().replace(0, 1.0)
        df_norm.loc[mask, feature_cols] = (subset - mean) / std
    return df_norm

def bootstrap_wasserstein(arr1, arr2, n_bootstrap=WASS_N_BOOTSTRAP, seed=RANDOM_SEED):
    if len(arr1) < 2 or len(arr2) < 2:
        return np.nan
    rng = np.random.default_rng(seed)
    dists = []
    for _ in range(n_bootstrap):
        idx1 = rng.choice(len(arr1), len(arr1), replace=True)
        idx2 = rng.choice(len(arr2), len(arr2), replace=True)
        w = wasserstein_distance(arr1[idx1], arr2[idx2])
        dists.append(w)
    return np.mean(dists)

def compute_distances(df, feature_cols):
    print_progress(f"Computing Wasserstein distances (bootstrap={WASS_N_BOOTSTRAP})...")
    results = []
    subjects = df['subject'].unique()
    all_modalities = df[df['condition'] == 'attention']['task'].unique().tolist()
    print_progress(f"Detected modalities: {all_modalities}")
    for feat_idx, feat in enumerate(feature_cols):
        if (feat_idx + 1) % 50 == 0:
            print_progress(f"  Feature {feat_idx + 1}/{len(feature_cols)}...")
        w_modal_scores = []
        w_state_scores = []
        for subj in subjects:
            s_data = df[df['subject'] == subj]
            att_data = s_data[s_data['condition'] == 'attention']
            rel_data = s_data[s_data['condition'] == 'relaxed'][feat].dropna().values
            modality_data = {mod: att_data[att_data['task'] == mod][feat].dropna().values 
                             for mod in all_modalities if len(att_data[att_data['task'] == mod][feat].dropna()) >= WASS_MIN_EPOCHS}
            modal_dists = []
            for mod1, mod2 in combinations(modality_data.keys(), 2):
                mean_d = bootstrap_wasserstein(modality_data[mod1], modality_data[mod2])
                if not np.isnan(mean_d):
                    modal_dists.append(mean_d)
            w_modal = np.mean(modal_dists) if modal_dists else np.nan
            w_modal_scores.append(w_modal)
            all_att = att_data[feat].dropna().values
            w_state = bootstrap_wasserstein(all_att, rel_data) if len(all_att) >= WASS_MIN_EPOCHS and len(rel_data) >= WASS_MIN_EPOCHS else np.nan
            w_state_scores.append(w_state)
        median_w_modal = np.nanmedian(w_modal_scores)
        median_w_state = np.nanmedian(w_state_scores)
        if np.isnan(median_w_modal) or np.isnan(median_w_state):
            continue
        convergence_ratio = median_w_modal / (median_w_state + WASS_EPSILON)
        results.append({
            'feature': feat,
            'W_Modal_Median': median_w_modal,
            'W_State_Median': median_w_state,
            'Convergence_Ratio': convergence_ratio,
            'Valid_Subjects': len(subjects)
        })
    return pd.DataFrame(results)

def wasserstein_rank(train_wide_pruned, pruned_features, fold_dir):
    print_progress("Performing Wasserstein ranking on pruned features...")
    train_pruned_norm = normalize_features_subjectwise(train_wide_pruned, pruned_features)
    dist_df = compute_distances(train_pruned_norm, pruned_features)
    if dist_df.empty:
        print_progress("No valid distances; skipping Wasserstein.")
        return None, None
    state_threshold = dist_df['W_State_Median'].quantile(WASS_SENSITIVITY_QUANTILE)
    print_progress(f"Selection: W_State >= {state_threshold:.4f}, Ratio < {WASS_CONVERGENCE_RATIO_THRESHOLD}")
    filtered_df = dist_df[dist_df['W_State_Median'] >= state_threshold].sort_values('Convergence_Ratio')
    selected_df = filtered_df[filtered_df['Convergence_Ratio'] < WASS_CONVERGENCE_RATIO_THRESHOLD]
    if selected_df.empty:
        print_progress("No features passed threshold; using top 20.")
        selected_df = filtered_df.head(20)
    print_progress(f"Selected {len(selected_df)} features.")
    # Save
    dist_df.to_csv(os.path.join(fold_dir, 'wasserstein_full_metrics.csv'), index=False)
    selected_df.to_csv(os.path.join(fold_dir, 'wasserstein_selected.csv'), index=False)
    return dist_df, selected_df['feature'].tolist()

def plot_distributions(df, features, out_dir, subdir="", max_plots=20):
    print_progress(f"Plotting distributions ({subdir}) for up to {max_plots} features...")
    plots_dir = os.path.join(out_dir, "Plots", subdir)
    os.makedirs(plots_dir, exist_ok=True)
    sns.set_theme(style="whitegrid")
    modalities = sorted(df[df['condition'] == 'attention']['task'].unique().tolist())
    colors = plt.cm.tab10.colors
    for feat in features[:max_plots]:
        try:
            plt.figure(figsize=(10, 6))
            plotted = False
            for i, mod in enumerate(modalities):
                data = df[(df['condition'] == 'attention') & (df['task'] == mod)][feat].dropna()
                if len(data) >= WASS_MIN_EPOCHS:
                    sns.kdeplot(data, label=f'Attention ({mod.title()})', color=colors[i % len(colors)], fill=True, alpha=0.2)
                    plotted = True
            data_rel = df[df['condition'] == 'relaxed'][feat].dropna()
            if len(data_rel) >= WASS_MIN_EPOCHS:
                sns.kdeplot(data_rel, label='Relaxed', color='black', linestyle='--', linewidth=1.5)
                plotted = True
            if plotted:
                plt.title(f"Feature: {feat}")
                plt.xlabel("Normalized Value")
                plt.ylabel("Density")
                plt.legend()
                plt.tight_layout()
                safe_name = feat.replace('/', '_').replace('\\', '_')[:50]
                plt.savefig(os.path.join(plots_dir, f"{safe_name}_dist.png"))
            plt.close()
        except Exception as e:
            print_progress(f"  Skip plot for {feat}: {e}")
            plt.close()

class SubjectRobustScaler:
    def __init__(self):
        self._means = {}
        self._stds = {}
        self._global_mean = None
        self._global_std = None
        self.feature_cols = None

    def fit(self, X):
        self.feature_cols = [c for c in X.columns if c not in ['subject', 'condition']]
        self._global_mean = X[self.feature_cols].mean()
        self._global_std = X[self.feature_cols].std(ddof=1).replace(0, 1.0)
        for s in X['subject'].unique():
            df = X[X['subject'] == s][self.feature_cols]
            self._means[s] = df.mean()
            self._stds[s] = df.std(ddof=1).replace(0, 1.0)
        return self

    def transform(self, X):
        out = X.copy()
        for s in out['subject'].unique():
            mask = out['subject'] == s
            mean = self._means.get(s, self._global_mean)
            std = self._stds.get(s, self._global_std)
            out.loc[mask, self.feature_cols] = (out.loc[mask, self.feature_cols] - mean) / std
        return out

# ---------------------------- Main LOSO Function ----------------------------

def train_loso(df, out_dir, zip_folds=True, keep_fold_folders=False):
    os.makedirs(out_dir, exist_ok=True)
    summary = []
    groups = df['subject']
    logo = LeaveOneGroupOut()
    fold_i = 0
    for train_idx, test_idx in logo.split(df, groups=groups):
        fold_start = time.time()
        train_df = df.iloc[train_idx].reset_index(drop=True)
        test_df = df.iloc[test_idx].reset_index(drop=True)
        test_subj = test_df['subject'].unique()[0]
        fold_dir = os.path.join(out_dir, f"fold_{fold_i}_test_subj_{test_subj}")
        os.makedirs(fold_dir, exist_ok=True)
        print_progress(f"Fold {fold_i}: Test subject {test_subj}, train subjects {train_df['subject'].nunique()}")
        
        # Redundancy Pruning on train
        pruned_features = redundancy_prune(train_df, fold_dir)
        
        # Prepare wide pruned train for Wasserstein
        train_feature, train_meta = prepare_wide_features(train_df)
        train_wide_pruned = train_meta.join(train_feature[pruned_features])
        
        # Wasserstein on pruned
        dist_df, wass_selected = wasserstein_rank(train_wide_pruned, pruned_features, fold_dir)
        
        # Feature sets
        feature_sets = {
            'PRUNED': pruned_features,
            'WASS_TOP10': wass_selected[:10] if wass_selected else [],
            'WASS_TOP50PCT': wass_selected[:max(1, len(wass_selected)//2)] if wass_selected else [],
            'WASS_ALL': wass_selected if wass_selected else []
        }
        
        # Prepare test wide
        test_feature, test_meta = prepare_wide_features(test_df)
        test_wide = test_meta.join(test_feature)
        
        for fs_name, feats in feature_sets.items():
            if not feats:
                print_progress(f"  Skipping {fs_name}: no features selected.")
                continue
            print_progress(f"  --- Strategy: {fs_name} ({len(feats)} features) ---")
            strategy_dir = os.path.join(fold_dir, fs_name)
            os.makedirs(strategy_dir, exist_ok=True)
            # Save selected
            pd.Series(feats).to_csv(os.path.join(strategy_dir, 'selected_features.csv'), index=False)
            # Plots (using train_wide_pruned norm for wass, but for pruned use baseline norm? Use subject zscore for consistency)
            plot_df = normalize_features_subjectwise(train_wide_pruned, pruned_features)
            plot_distributions(plot_df, feats, strategy_dir)
            
            # Prepare data for ML
            train_selected = train_wide_pruned[feats + ['subject', 'condition']]
            test_selected = test_wide[[f for f in feats if f in test_wide.columns] + ['subject', 'condition']]
            test_selected = test_selected.reindex(columns=train_selected.columns, fill_value=0)
            norm = SubjectRobustScaler().fit(train_selected)
            train_norm = norm.transform(train_selected).drop('subject', axis=1)
            test_norm = norm.transform(test_selected).drop('subject', axis=1)
            y_train = (train_norm['condition'] == 'attention').astype(int)
            y_test = (test_norm['condition'] == 'attention').astype(int)
            X_train = train_norm.drop('condition', axis=1)
            X_test = test_norm.drop('condition', axis=1)
            
            # Balance train
            X_att = X_train[y_train == 1]
            X_rel = X_train[y_train == 0]
            n_min = min(len(X_att), len(X_rel))
            if n_min < 5:
                print_progress(f"  Skip {fs_name}: insufficient samples (n_min={n_min})")
                continue
            X_att_sub = resample(X_att, n_samples=n_min, replace=False, random_state=RANDOM_SEED)
            X_rel_sub = resample(X_rel, n_samples=n_min, replace=False, random_state=RANDOM_SEED)
            X_train_bal = pd.concat([X_att_sub, X_rel_sub])
            y_train_bal = np.hstack([np.ones(n_min), np.zeros(n_min)])
            idx = np.arange(len(X_train_bal))
            np.random.shuffle(idx)
            X_train_bal = X_train_bal.iloc[idx].reset_index(drop=True)
            y_train_bal = y_train_bal[idx]
            
            # Final scale
            final_scaler = StandardScaler().fit(X_train_bal)
            X_train_arr = final_scaler.transform(X_train_bal)
            X_test_arr = final_scaler.transform(X_test)
            
            # Train models
            for name, clf in ALGORITHMS:
                try:
                    clf_clone = clone(clf)
                    clf_clone.fit(X_train_arr, y_train_bal)
                    y_pred = clf_clone.predict(X_test_arr)
                    bal_acc = balanced_accuracy_score(y_test, y_pred)
                    auc = roc_auc_score(y_test, clf_clone.predict_proba(X_test_arr)[:,1]) if hasattr(clf_clone, "predict_proba") and len(np.unique(y_test)) == 2 else None
                    joblib.dump(clf_clone, os.path.join(strategy_dir, f"model_{name}.joblib"))
                    cm = confusion_matrix(y_test, y_pred)
                    plot_cm(cm, ['relaxed', 'attention'], os.path.join(strategy_dir, f"cm_{name}.png"), f"{name} CM")
                    summary.append({
                        "fold": fold_i,
                        "test_subject": test_subj,
                        "feature_set": fs_name,
                        "model": name,
                        "balanced_acc": bal_acc,
                        "auc": auc,
                        "n_features": len(feats)
                    })
                except Exception as e:
                    print_progress(f"  Model {name} failed: {e}")
        
        elapsed = time.time() - fold_start
        print_progress(f"Fold {fold_i} done in {elapsed:.2f}s")
        if zip_folds:
            zip_path = shutil.make_archive(fold_dir, 'zip', fold_dir)
            print_progress(f"Zipped {fold_dir} to {zip_path}")
            if not keep_fold_folders:
                shutil.rmtree(fold_dir)
        fold_i += 1
    
    if summary:
        pd.DataFrame(summary).to_csv(os.path.join(out_dir, "loso_summary.csv"), index=False)
        print_progress("Saved summary.")

# ---------------------------- CLI ----------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--features", default=DEFAULT_FEATURES_PATH)
    parser.add_argument("--out", default=DEFAULT_OUT_DIR)
    parser.add_argument("--no-zip", action="store_true")
    parser.add_argument("--keep-fold-folders", action="store_true")
    args = parser.parse_args()
    
    df = load_features(args.features)
    df = df[~df['subject'].isin(EXCLUDE_SUBJECTS)].reset_index(drop=True)
    print_progress(f"Excluded subjects {EXCLUDE_SUBJECTS}; remaining subjects: {df['subject'].nunique()}")
    
    train_loso(df, args.out, zip_folds=not args.no_zip, keep_fold_folders=args.keep_fold_folders)
