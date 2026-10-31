#!/usr/bin/env python3
"""
09_diagnostic_subject_clustering_and_region_mapping.py
-----------------------------------------------------

DIAGNOSTIC / ANALYTIC SCRIPT ONLY

- Subject clustering on robust biomarkers (quality assurance / heterogeneity check)
- Feature → channel mapping (preserves ORIGINAL feature names)
- Optional region mapping (--map-regions flag) for appendix/reporting

⚠️ DIAGNOSTIC ONLY:
This script is for INTERPRETATION and PAPER SUPPLEMENTS only.
Outputs MUST NOT be used for feature selection, training, or validation.
All inputs come from prior biomarker discovery (post-stat validation).

Outputs:
 - subject_clustering_labels.csv
 - subject_clustering_plot.png
 - feature_channel_map.csv
 - neuro_mapping_report.csv (if --map-regions)
 - mapping_manifest.json
"""

import os
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json
import logging
import re
import glob
from datetime import datetime
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# Expert Audit Fix: Suppress KMeans memory leak warning on Windows
os.environ["OMP_NUM_THREADS"] = "1"

# ----------------------------------------
# Config: try to import global config
# ----------------------------------------
try:
    from importlib import import_module
    config = import_module('00_config')
    EXCLUDE_SUBJECTS = getattr(config, 'EXCLUDE_SUBJECTS_DOWNSTREAM', [])
    RANDOM_SEED = getattr(config, 'RANDOM_SEED', 42)
    CLUSTER_K_RANGE = getattr(config, 'CLUSTER_K_RANGE', [2, 3, 4])
    CLUSTER_BOOTSTRAPS = getattr(config, 'CLUSTER_BOOTSTRAPS', 1000)
    CLUSTER_SILHOUETTE_MIN = getattr(config, 'CLUSTER_SILHOUETTE_MIN', 0.35)
    CLUSTER_MIN_SUBJECTS = getattr(config, 'CLUSTER_MIN_SUBJECTS', 5)
    BIOMARKER_DIR = getattr(config, 'BIOMARKER_DIR', 'BIOMARKER DISCOVERY')
except Exception:
    EXCLUDE_SUBJECTS = []
    RANDOM_SEED = 42
    CLUSTER_K_RANGE = [2, 3, 4]
    CLUSTER_BOOTSTRAPS = 1000
    CLUSTER_SILHOUETTE_MIN = 0.35
    CLUSTER_MIN_SUBJECTS = 5
    BIOMARKER_DIR = 'BIOMARKER DISCOVERY'

np.random.seed(RANDOM_SEED)

# Default channel → Attention Network mapping (standard 10-20 for EPOC X)
DEFAULT_CHANNEL_TO_REGION = {
    # Dorsal Attention Network (DAN) - Goal-driven / Top-down
    'AF3': 'DAN (Frontal/FEF)', 'AF4': 'DAN (Frontal/FEF)',
    'F3':  'DAN (Frontal/FEF)', 'F4':  'DAN (Frontal/FEF)',
    'P7':  'DAN (Parietal/IPS)', 'P8':  'DAN (Parietal/IPS)',
    
    # Ventral Attention Network (VAN) - Stimulus-driven / Reorienting
    'T7':  'VAN (Temporal/TPJ)', 'T8':  'VAN (Temporal/TPJ)',
    'F7':  'VAN (Inf Frontal)',   'F8':  'VAN (Inf Frontal)',
    'FC5': 'VAN (Frontal-Central)', 'FC6': 'VAN (Frontal-Central)',
    
    # Sensory
    'O1':  'Visual (Occipital)', 'O2':  'Visual (Occipital)'
}

def get_timestamp():
    return datetime.now().strftime('%Y-%m-%dT%H:%M:%S')

def load_csv_or_parquet(path):
    if path.endswith('.parquet'):
        return pd.read_parquet(path)
    else:
        return pd.read_csv(path)

def load_feature_list(path):
    df = load_csv_or_parquet(path)
    if 'feature' in df.columns:
        return df['feature'].tolist()
    else:
        return df.iloc[:, 0].tolist()

def run_subject_clustering(subj_df, robust_features, out_dir):
    """Perform K-means clustering on subject-level feature profiles."""
    os.makedirs(out_dir, exist_ok=True)
    
    # Standardize column names for pivot
    df = subj_df.copy()
    col_map = {
        'test_subject': 'subject',
        'effect_mean': 'value',
        'Ratio': 'value',
        'convergence_ratio': 'value'
    }
    df = df.rename(columns=col_map)
    
    # Ensure mandatory columns
    if 'subject' not in df.columns or 'feature' not in df.columns:
        raise ValueError(f"Subject data missing mandatory columns ('subject'/'test_subject' AND 'feature'). Found: {df.columns.tolist()}")

    # Aggregate if multiple entries per (subject, feature) - e.g. from multiple LOSO resamples
    if df.duplicated(subset=['subject', 'feature']).any():
        print("Detected duplicate entries (e.g. resamples); aggregating numeric columns by mean.")
        # Select numeric columns to aggregate (preserve subject & feature)
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        # If 'value' present it will be included; otherwise other numeric columns (like effect_mean) will be used.
        if not numeric_cols:
            raise ValueError("No numeric columns available to aggregate for duplicates.")
        agg_df = df.groupby(['subject', 'feature'])[numeric_cols].mean().reset_index()
        # Keep non-numeric metadata if needed (not used for clustering here)
        df = agg_df

    # Pivot to subject × feature matrix
    if 'value' in df.columns:
        pivot = df.pivot(index='subject', columns='feature', values='value')
    else:
        # Fallback: use first numeric column that isn't metadata
        skip_cols = {'subject', 'fold_idx', 'resample_idx', 'feature'}
        numeric_cols = [c for c in df.select_dtypes(include=[np.number]).columns if c not in skip_cols]
        if not numeric_cols:
            raise ValueError("No numeric value column found for clustering")
        print(f"Using '{numeric_cols[0]}' as value column for clustering (fallback).")
        pivot = df.pivot(index='subject', columns='feature', values=numeric_cols[0])
    
    # Filter to robust features only
    common_feats = [f for f in robust_features if f in pivot.columns]
    if len(common_feats) == 0:
        raise ValueError("No overlap between robust features and subject data columns")
    X = pivot[common_feats].values
    
    # Exclude subjects with any NaNs across the chosen features
    subjects = pivot.index.tolist()
    valid_mask = ~np.isnan(X).any(axis=1)
    X = X[valid_mask]
    subjects = np.array(subjects)[valid_mask]
    
    if len(subjects) < CLUSTER_MIN_SUBJECTS:
        print(f"Warning: Only {len(subjects)} subjects available. Skipping clustering.")
        return None, {}
    
    # Find optimal K
    # Reset best_score to actual silhouette range [-1, 1]
    from sklearn.preprocessing import StandardScaler
    X_scaled = StandardScaler().fit_transform(X)
    # Cap outliers at 3*std to prevent Inf/Huge Ratios from dominating
    X_scaled = np.clip(X_scaled, -3, 3)

    best_k = 2
    actual_best_score = -1.0
    silhouette_scores = []
    
    for k in CLUSTER_K_RANGE:
        if k >= len(subjects):
            continue
        try:
            kmeans = KMeans(n_clusters=k, random_state=RANDOM_SEED, n_init=50)
            labels = kmeans.fit_predict(X_scaled)
            # silhouette requires at least 2 clusters and < n_samples clusters
            score = silhouette_score(X_scaled, labels)
            silhouette_scores.append((k, score))
            if score > actual_best_score:
                actual_best_score = score
                if score >= CLUSTER_SILHOUETTE_MIN:
                    best_k = k
        except Exception as e:
            print(f"  Skipping k={k} due to error: {e}")
            continue

    # Final clustering with best_k
    try:
        kmeans = KMeans(n_clusters=best_k, random_state=RANDOM_SEED, n_init=50)
        labels = kmeans.fit_predict(X_scaled)
    except Exception as e:
        raise RuntimeError(f"Final clustering failed for k={best_k}: {e}")
    
    # Save results
    cluster_df = pd.DataFrame({
        'subject': subjects,
        'cluster_label': labels
    })
    cluster_path = os.path.join(out_dir, "subject_clustering_labels.csv")
    cluster_df.to_csv(cluster_path, index=False)
    
    # Plot (PC space for visualization)
    from sklearn.decomposition import PCA
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)
    
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x=X_pca[:, 0], y=X_pca[:, 1], hue=labels, palette='deep', s=100)
    plt.title(f"Subject Clustering (k={best_k}, silhouette={actual_best_score:.3f})")
    plt.xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)")
    plt.ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)")
    plot_path = os.path.join(out_dir, "subject_clustering_plot.png")
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Subject clustering complete: k={best_k}, silhouette={actual_best_score:.3f}")
    if actual_best_score < CLUSTER_SILHOUETTE_MIN:
        print(f"  (Note: silhouette below threshold {CLUSTER_SILHOUETTE_MIN}; default k=2 selected)")
    print(f"Saved labels: {cluster_path}")
    print(f"Saved plot: {plot_path}")
    
    manifest = {
        'best_k': best_k,
        'silhouette_score': float(actual_best_score),
        'n_subjects_clustered': int(len(subjects)),
        'features_used': int(len(common_feats)),
        'pca_var_ratio': pca.explained_variance_ratio_.tolist()
    }
    return cluster_df, manifest

def run_feature_channel_mapping(robust_features, out_dir, map_regions=False, channel_to_region=None):
    """Extract channel from feature name and optionally map to region."""
    os.makedirs(out_dir, exist_ok=True)
    
    mapping_records = []
    for feat in robust_features:
        # Common patterns: "AF3_alpha_power", "F7_theta_relative", "Global_gamma_entropy", etc.
        channel_match = re.match(r'^([A-Za-z0-9]+)_', feat)
        channel = channel_match.group(1) if channel_match else "Global"
        region = None
        if map_regions:
            if channel_to_region and channel in channel_to_region:
                region = channel_to_region.get(channel)
            else:
                # Map unknown channels to "Unknown" when mapping was requested
                region = "Unknown"
        
        mapping_records.append({
            'feature': feat,
            'channel': channel,
            'region': region
        })
    
    map_df = pd.DataFrame(mapping_records)
    map_path = os.path.join(out_dir, "feature_channel_map.csv")
    map_df.to_csv(map_path, index=False)
    
    if map_regions:
        # Summary report by region
        region_summary = map_df[map_df['region'].notnull()]
        region_counts = region_summary['region'].value_counts().reset_index()
        region_counts.columns = ['region', 'feature_count']
        report_path = os.path.join(out_dir, "neuro_mapping_report.csv")
        region_counts.to_csv(report_path, index=False)
        print(f"Saved region report: {report_path}")
    
    print(f"Saved feature -> channel map ({len(robust_features)} features): {map_path}")
    return map_df

def process_source(subject_path, robust_path, base_out_dir, map_regions, label):
    """Orchestrate the analysis for a specific source."""
    print(f"\n>>> PROCESSING SOURCE: {label} <<<")
    out_dir = os.path.join(base_out_dir, label)
    os.makedirs(out_dir, exist_ok=True)

    # 1. Load data
    if not os.path.exists(subject_path):
        print(f"  Warning: Subject data not found: {subject_path}")
        return
    subj_df = load_csv_or_parquet(subject_path)
    
    if not os.path.exists(robust_path):
        print(f"  Warning: Robust features file not found: {robust_path}")
        return
    robust_features = load_feature_list(robust_path)
    print(f"  Loaded {len(robust_features)} robust features")
    
    # Filter subject data
    if 'feature' in subj_df.columns:
        subj_df = subj_df[subj_df['feature'].isin(robust_features)].copy()
    
    # 2. Run analyses
    cluster_df, cluster_manifest = run_subject_clustering(subj_df, robust_features, out_dir)
    
    run_feature_channel_mapping(
        robust_features, out_dir,
        map_regions=map_regions,
        channel_to_region=DEFAULT_CHANNEL_TO_REGION if map_regions else None
    )
    
    # 3. Save source-specific manifest
    manifest = {
        'source_label': label,
        'input_subject_data': subject_path,
        'input_robust_features': robust_path,
        'map_regions_enabled': map_regions,
        'created_at': get_timestamp(),
        **(cluster_manifest or {})
    }
    with open(os.path.join(out_dir, "mapping_manifest.json"), 'w') as f:
        json.dump(manifest, f, indent=2, default=str)

def main():
    parser = argparse.ArgumentParser(description="Diagnostic subject clustering and feature → attention network mapping")
    parser.add_argument('--subject_data', type=str, default=None,
                        help='Path to per-subject data CSV.')
    parser.add_argument('--robust_features', type=str, default=None,
                        help='Path to robust features file.')
    parser.add_argument('--map-regions', action='store_true',
                        help='Enable Attention Network mapping (DAN, VAN, etc.)')
    parser.add_argument('--auto-all', action='store_true',
                        help='Automatically detect and run for BOTH Stat and Wasserstein sources.')
    parser.add_argument('--output-dir', type=str, default=os.path.join(BIOMARKER_DIR, "DiagnosticMapping"),
                        help='Output base directory')
    
    args = parser.parse_args()
    base_out_dir = args.output_dir
    os.makedirs(base_out_dir, exist_ok=True)

    sources = []
    
    # Option A: Manual override
    if args.subject_data and args.robust_features:
        sources.append({
            'subj': args.subject_data,
            'robust': args.robust_features,
            'label': 'ManualRun'
        })
    
    # Option B: Auto-detection (Stat + Wasserstein)
    if args.auto_all or not sources:
        # Detect Stat
        val_dir = os.path.join(BIOMARKER_DIR, "Validation")
        stat_robust = glob.glob(os.path.join(val_dir, "final_selected_features_*.csv"))
        stat_subj = glob.glob(os.path.join(val_dir, "loso_fold_results_*.csv"))
        
        if stat_robust and stat_subj:
            sources.append({
                'subj': max(stat_subj, key=os.path.getctime),
                'robust': max(stat_robust, key=os.path.getctime),
                'label': 'StatisticalValidation'
            })
            
        # Detect Wasserstein
        wass_dir = os.path.join(BIOMARKER_DIR, "Wasserstein")
        wass_robust = os.path.join(wass_dir, "selected_features_wasserstein.csv")
        wass_subj = os.path.join(wass_dir, "wasserstein_subject_clustering_data.csv")
        
        if os.path.exists(wass_robust) and os.path.exists(wass_subj):
            sources.append({
                'subj': wass_subj,
                'robust': wass_robust,
                'label': 'WassersteinExploratory'
            })

    if not sources:
        print("No data sources found. Use --subject_data and --robust_features to specify them manually.")
        return

    # Run for all identified sources
    for src in sources:
        process_source(src['subj'], src['robust'], base_out_dir, args.map_regions, src['label'])
    
    print("\n" + "!"*80)
    print("WARNING: This script is DIAGNOSTIC ONLY.")
    print("Outputs are for interpretation (Attention Networks), figures, and supplements.")
    print("DO NOT use clustering labels for training/classification.")
    print("!"*80 + "\n")
    
    print("\n" + "=" * 60)
    print("SUBJECT CLUSTERING & NETWORK MAPPING COMPLETE")
    print("=" * 60)

if __name__ == "__main__":
    main()
