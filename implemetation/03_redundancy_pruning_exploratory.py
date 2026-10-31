"""
03_redundancy_pruning_exploratory.py

Stage 3a: Feature Redundancy Pruning (Exploratory)
--------------------------------------------------
Performs unsupervised feature selection by grouping highly correlated features
and selecting cluster medoids.

⚠️ ANALYSIS ONLY - Per classifier_modifications.md:
This script is for EXPLORATORY ANALYSIS and PAPER TABLES ONLY.
Do NOT use output CSVs (selected_features_redundancy_pruned.csv) for training.
For training, use the RedundancyPruner transformer inside CV folds.

Reference: Redundancy_Pruning_and_Wasserstein_Distance_Guide.md
"""

import pandas as pd
import numpy as np
import scipy.cluster.hierarchy as sch
from scipy.spatial.distance import squareform
import matplotlib.pyplot as plt
import os
import sys
import json

# Import reporting utilities
try:
    from utils.reporting import format_pruning_cluster, write_manifest, get_timestamp
except ImportError:
    # Fallback if utils not found
    def get_timestamp():
        from datetime import datetime
        return datetime.now().strftime('%Y-%m-%dT%H:%M:%S')
    def format_pruning_cluster(cluster_id, medoid, size, members, distance_mean):
        return {'cluster_id': cluster_id, 'medoid_feature': medoid, 'size': size,
                'member_count': len(members), 'member_features_json': json.dumps(members),
                'cluster_distance_mean': distance_mean, 'created_at': get_timestamp()}
    def write_manifest(d, p, f='m.json'):
        import json; json.dump(d, open(os.path.join(p, f), 'w'), indent=2)

# ---------- CONFIG ----------
try:
    from importlib import import_module
    config = import_module('00_config')
    EXCLUDE_SUBJECTS = getattr(config, 'EXCLUDE_SUBJECTS_DOWNSTREAM', [])
    CORR_CLUSTER_THRESH = getattr(config, 'CORR_CLUSTER_THRESH', 0.20)
    MIN_FEATURES_AFTER_PRUNING = getattr(config, 'MIN_FEATURES_AFTER_PRUNING', 30)
    PRUNING_THRESH_RANGE = getattr(config, 'PRUNING_THRESH_RANGE', (0.05, 0.5))
    PRUNING_THRESH_STEP = getattr(config, 'PRUNING_THRESH_STEP', 0.01)
    PRUNING_STRATEGY = getattr(config, 'PRUNING_STRATEGY', 'adaptive')
    RANDOM_SEED = getattr(config, 'RANDOM_SEED', 42)
    print(f"[Config] Loaded from 00_config.py")
except Exception as e:
    print(f"[Config] Fallback defaults: {e}")
    EXCLUDE_SUBJECTS = []
    CORR_CLUSTER_THRESH = 0.30
    MIN_FEATURES_AFTER_PRUNING = 30
    PRUNING_THRESH_RANGE = (0.05, 0.5)
    PRUNING_THRESH_STEP = 0.01
    PRUNING_STRATEGY = 'adaptive'
    RANDOM_SEED = 42

np.random.seed(RANDOM_SEED)

# Known metadata columns
METADATA_COLS = ['unique_id', 'file', 'task', 'subject', 'condition', 'epoch_idx', 'target', 'label', 'channel']


def load_features(filepath):
    """Load features from parquet or CSV."""
    print(f"Loading features from {filepath}...")
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
    
    print(f"Loaded data shape: {df.shape}")
    return df


def prepare_wide_features(df):
    """
    Prepare features in wide format (samples x features).
    If data is narrow (has 'channel' column), pivot to wide format.
    Returns: (feature_df, metadata_df)
    """
    print("Preparing data (Samples x Features)...")
    
    # Check if 'channel' exists -> Needs Pivoting
    if 'channel' in df.columns:
        print("Detected narrow format (channel column present). Pivoting using set_index().unstack()...")
        
        # Identify columns
        index_cols = [c for c in df.columns if c in METADATA_COLS and c != 'channel']
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        value_cols = [c for c in numeric_cols if c not in METADATA_COLS]
        
        if not value_cols:
            print("Error: No numeric feature columns found.")
            return pd.DataFrame(), pd.DataFrame()
        
        try:
            # Create unique_id if not present
            if 'unique_id' not in df.columns:
                df['unique_id'] = df['file'].astype(str) + '_' + df['epoch_idx'].astype(str)
                index_cols = ['unique_id'] + [c for c in index_cols if c != 'unique_id']
            
            # Use set_index().unstack() - correct approach
            pv = df.set_index(index_cols + ['channel'])[value_cols]
            pv = pv.unstack(level='channel')
            
            # Flatten MultiIndex: (feature, channel) -> channel_feature
            pv.columns = ['{}_{}'.format(ch, feat) for feat, ch in pv.columns]
            pv = pv.reset_index()
            
            # Separate metadata and features
            meta_df = pv[[c for c in pv.columns if c in METADATA_COLS]]
            feature_df = pv.select_dtypes(include=[np.number])
            feature_df = feature_df.drop(columns=[c for c in feature_df.columns if c in METADATA_COLS], errors='ignore')
            
            print(f"Pivoted to {feature_df.shape[1]} features")
            return feature_df, meta_df
            
        except Exception as e:
            print(f"Pivot failed: {e}")
            return pd.DataFrame(), pd.DataFrame()
    
    else:
        print("Detected wide format (already flattened). Selecting numeric feature columns...")
        # Separate metadata and features
        meta_df = df[[c for c in df.columns if c in METADATA_COLS]]
        feature_df = df.select_dtypes(include=[np.number])
        feature_df = feature_df.drop(columns=[c for c in feature_df.columns if c in METADATA_COLS], errors='ignore')
        
        print(f"Features found: {feature_df.shape[1]}")
        return feature_df, meta_df


def subjectwise_zscore(df_features, meta_df):
    """
    Safe per-subject z-score with index alignment.
    
    NOTE:
    This baseline normalization is for CORRELATION STRUCTURE ONLY.
    All downstream statistical tests and classifiers re-normalize
    per fold to avoid leakage. This step does NOT affect modeling.
    """
    # Reset and align indices
    df_features = df_features.reset_index(drop=True)
    meta_df = meta_df.reset_index(drop=True)
    
    df_norm = df_features.copy()
    feature_cols = df_features.select_dtypes(include=[np.number]).columns
    
    for subj in meta_df['subject'].unique():
        mask = meta_df['subject'] == subj
        idx = meta_df[mask].index
        if len(idx) < 2:  # Too few samples
            continue
        subset = df_features.loc[idx, feature_cols]
        mean = subset.mean()
        std = subset.std().replace(0.0, 1.0)
        df_norm.loc[idx, feature_cols] = (subset - mean) / std
    
    return df_norm


def compute_correlation_distance(df_features):
    """Compute Spearman correlation and convert to distance matrix."""
    print("Computing Spearman correlation matrix...")
    corr_matrix = df_features.corr(method='spearman')
    
    # Distance = 1 - |correlation|
    dist_matrix = 1 - np.abs(corr_matrix)
    
    # Fix edge cases
    dist_matrix = np.clip(dist_matrix, 0, 2)
    np.fill_diagonal(dist_matrix.values, 0.0)
    
    # Handle NaNs (constant features)
    if dist_matrix.isnull().values.any():
        print("Warning: NaNs in distance matrix (constant features). Filling with max distance (1.0).")
        dist_matrix = dist_matrix.fillna(1.0)
    
    return corr_matrix, dist_matrix


def perform_clustering(dist_matrix):
    """Perform hierarchical clustering using Ward linkage."""
    print("Performing hierarchical clustering (Ward linkage)...")
    
    # Convert to condensed form
    dist_array = dist_matrix.values.astype(np.float64)
    condensed_dist = squareform(dist_array)
    
    # Ward linkage
    linkage_matrix = sch.ward(condensed_dist)
    
    return linkage_matrix


def find_optimal_threshold(linkage_matrix, n_features, min_features=30, strategy='adaptive'):
    """
    Find optimal clustering threshold using adaptive or fixed strategy.
    """
    # Load fixed threshold from config default
    fixed_thresh = getattr(config, 'FIXED_PRUNING_THRESHOLD', 0.20)
    
    if strategy == 'fixed':
        print(f"Using FIXED pruning threshold: {fixed_thresh} (equivalent to |Spearman rho| > {1 - fixed_thresh:.2f})")
        return fixed_thresh
    
    print(f"Finding optimal threshold (strategy={strategy}, min_features={min_features})...")
    
    best_t = getattr(config, 'CORR_CLUSTER_THRESH', 0.30)
    
    # Sweep through thresholds
    for t in np.arange(PRUNING_THRESH_RANGE[0], 
                       PRUNING_THRESH_RANGE[1], 
                       PRUNING_THRESH_STEP):
        clusters = sch.fcluster(linkage_matrix, t, criterion='distance')
        n_clusters = len(np.unique(clusters))
        
        if n_clusters >= min_features:
            best_t = t
            print(f"  Threshold {t:.2f} -> {n_clusters} clusters (meets minimum)")
            break
    else:
        # If no threshold meets minimum, use the one giving most clusters
        best_t = PRUNING_THRESH_RANGE[0]
        clusters = sch.fcluster(linkage_matrix, best_t, criterion='distance')
        n_clusters = len(np.unique(clusters))
        print(f"  Warning: Could not reach {min_features} clusters. Using t={best_t:.2f} -> {n_clusters} clusters")
    
    return best_t


def select_medoids(df_features, clusters, corr_matrix):
    """Select medoid (most central feature) from each cluster."""
    print("Selecting medoids from clusters...")
    features = df_features.columns.values
    unique_clusters = np.unique(clusters)
    selected_features = []
    cluster_info = []
    
    # Pre-compute distance matrix
    dist_vals = 1 - np.abs(corr_matrix.values)
    
    for clust_id in unique_clusters:
        indices = np.where(clusters == clust_id)[0]
        
        if len(indices) == 1:
            medoid_idx = indices[0]
            avg_dist = 0.0
        else:
            # Medoid: feature with minimum sum of distances to others
            sub_dist = dist_vals[np.ix_(indices, indices)]
            sum_dists = np.sum(sub_dist, axis=1)
            medoid_local_idx = np.argmin(sum_dists)
            medoid_idx = indices[medoid_local_idx]
            # Compute mean distance from medoid to all members
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
    """Plot and save dendrogram."""
    print("Plotting dendrogram...")
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
    print(f"Dendrogram saved to: {out_path}")


def main():
    # Setup paths
    try:
        base_dir = os.path.dirname(os.path.abspath(__file__))
    except NameError:
        base_dir = os.getcwd()
    
    # Use config paths if available, else fallback
    features_dir = getattr(config, 'FEATURES_DIR', 'Features')
    biomarker_dir = getattr(config, 'BIOMARKER_DIR', 'BIOMARKER DISCOVERY')
    
    features_path = os.path.join(base_dir, features_dir, 'features_enhanced.parquet')
    output_dir = os.path.join(base_dir, biomarker_dir, 'Redundancy')
    os.makedirs(output_dir, exist_ok=True)
    
    # Check for file (parquet or csv fallback)
    if not os.path.exists(features_path):
        csv_path = features_path.replace('.parquet', '.csv')
        if os.path.exists(csv_path):
            features_path = csv_path
        else:
            print(f"Error: Features file not found at {features_path}")
            sys.exit(1)
    
    # Load data
    df = load_features(features_path)
    
    # Prepare wide format
    df_wide, meta_df = prepare_wide_features(df)
    if df_wide.empty:
        print("Failed to prepare features. Exiting.")
        return
    
    # Handle NaNs
    if df_wide.isnull().values.any():
        print("Warning: NaNs found. Filling with column mean.")
        df_wide = df_wide.fillna(df_wide.mean())
        df_wide = df_wide.dropna(axis=1)
    
    # Filter for Discovery Track features (Reviewer Feature Explosion Fix)
    # STRATEGY CHANGE: Keep all channel-level features. Exclude RIEM__.
    # Regional aggregation is disabled in 02, so REGION__ cols shouldn't exist anyway.
    
    all_cols = df_wide.columns
    riem_cols = [c for c in all_cols if c.startswith('RIEM__')]
    
    # Discovery Track: Keep channel-level, exclude Riemannian
    cols_to_keep = [c for c in all_cols if not c.startswith('RIEM__')]
    print(f"Discovery Mode: Selecting {len(cols_to_keep)} features (excluding {len(riem_cols)} Riemannian).")
    print("Excluding Riemannian features from redundancy pruning (biomarker track).")
    df_wide = df_wide[cols_to_keep]

    # =====================================================
    # CRITICAL FIX: Subject-wise Baseline Normalization
    # Per redundancy pruning plan / reviewer: Track A uses relaxed baseline
    # =====================================================
    # Calculate baseline stats per subject
    print("Applying Subject-wise Baseline Normalization...")
    subjects = meta_df['subject'].unique()
    df_norm = df_wide.copy()
    
    for subj in subjects:
        subj_mask = meta_df['subject'] == subj
        # Check condition 'relaxed' for baseline
        # We need condition info. metadata has it?
        # metadata prepared in prepare_wide_features includes 'condition'
        
        baseline_mask = subj_mask & (meta_df['condition'] == 'relaxed')
        if getattr(config, 'RUN_PROFILE', 'senior') == 'paper':
            baseline_mask = subj_mask  # Paper Section II-D: per-subject z-score.
        if not baseline_mask.any():
            print(f"  Warning: No relaxed epochs for subject {subj}. Using full-subject Z-score fallback.")
            # Z-score fallback
             # Verify alignment via index
            idx = meta_df.index[subj_mask]
            if len(idx) > 1:
                subset = df_wide.loc[idx]
                df_norm.loc[idx] = (subset - subset.mean()) / subset.std().replace(0, 1)
            continue
            
        # Compute baseline M/Std
        baseline_data = df_wide.loc[meta_df.index[baseline_mask]]
        mu = baseline_data.mean()
        sigma = baseline_data.std().replace(0, 1)
        
        # Apply to ALL epochs of this subject
        idx = meta_df.index[subj_mask]
        df_norm.loc[idx] = (df_wide.loc[idx] - mu) / sigma
        
    df_wide = df_norm
    print("Normalization complete.")
    # SKIP old zscore call
    # df_wide = subjectwise_zscore(df_wide, meta_df)
    
    # Drop constant features (variance=0)
    variances = df_wide.var()
    constant_feats = variances[variances < 1e-12].index
    if len(constant_feats) > 0:
        print(f"Dropping {len(constant_feats)} constant/variance-zero features")
        df_wide = df_wide.drop(columns=constant_feats)
    
    # Compute correlation and distance
    corr_matrix, dist_matrix = compute_correlation_distance(df_wide)
    
    # Perform clustering
    linkage_matrix = perform_clustering(dist_matrix)
    
    # Find optimal threshold
    optimal_thresh = find_optimal_threshold(
        linkage_matrix, 
        len(df_wide.columns), 
        MIN_FEATURES_AFTER_PRUNING, 
        PRUNING_STRATEGY
    )
    
    # Form clusters
    print(f"Forming clusters with distance threshold {optimal_thresh:.3f}...")
    clusters = sch.fcluster(linkage_matrix, optimal_thresh, criterion='distance')
    
    # Select medoids
    selected_features, cluster_info = select_medoids(df_wide, clusters, corr_matrix)
    
    # Report results
    n_total = len(df_wide.columns)
    n_selected = len(selected_features)
    reduction = (1 - n_selected / n_total) * 100
    
    print(f"\n{'='*50}")
    print(f"Original Features: {n_total}")
    print(f"Selected Features: {n_selected}")
    print(f"Reduction: {reduction:.2f}%")
    print(f"Threshold Used: {optimal_thresh:.3f}")
    print(f"{'='*50}\n")
    
    # Visualize
    dendro_path = os.path.join(output_dir, 'feature_dendrogram.png')
    plot_dendrogram(linkage_matrix, optimal_thresh, dendro_path)
    
    # Save selected features
    out_csv = os.path.join(output_dir, 'selected_features_redundancy_pruned.csv')
    pd.Series(selected_features, name='feature').to_csv(out_csv, index=False)
    print(f"Selected features saved to {out_csv}")
    
    # Save cluster mapping
    cluster_map_path = os.path.join(output_dir, 'cluster_mapping.csv')
    cluster_map_df = pd.DataFrame({
        'feature': df_wide.columns,
        'cluster': clusters
    })
    cluster_map_df.to_csv(cluster_map_path, index=False)
    print(f"Cluster mapping saved to {cluster_map_path}")
    
    # Save pruning_cluster_info.csv (canonical schema)
    cluster_info_rows = []
    for info in cluster_info:
        cluster_info_rows.append(format_pruning_cluster(
            cluster_id=info['cluster'],
            medoid=info['medoid'],
            size=info['size'],
            members=info['members'],
            distance_mean=info.get('distance_mean', np.nan)
        ))
    cluster_info_df = pd.DataFrame(cluster_info_rows)
    cluster_info_path = os.path.join(output_dir, 'pruning_cluster_info.csv')
    cluster_info_df.to_csv(cluster_info_path, index=False)
    print(f"Cluster info saved to {cluster_info_path}")
    
    # Save summary JSON
    summary = {
        'original_count': n_total,
        'selected_count': n_selected,
        'reduction_pct': round(reduction, 2),
        'threshold': optimal_thresh,
        'strategy': PRUNING_STRATEGY,
        'pruning_strategy_used': PRUNING_STRATEGY, # Added this line
        'n_clusters': len(np.unique(clusters)),
        'cluster_sizes': {int(k): int(v) for k, v in zip(*np.unique(clusters, return_counts=True))}
    }
    summary_path = os.path.join(output_dir, 'pruning_summary.json')
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"Summary saved to {summary_path}")
    
    # Save correlation and distance matrices for traceability (Step 8)
    try:
        corr_npy_path = os.path.join(output_dir, 'correlation_matrix.npy')
        dist_npy_path = os.path.join(output_dir, 'distance_matrix.npy')
        np.save(corr_npy_path, corr_matrix.values)
        np.save(dist_npy_path, dist_matrix.values)
        print(f"Saved matrices: {corr_npy_path}, {dist_npy_path}")
    except Exception as e:
        print(f"Warning: Could not save matrices: {e}")
        
    # Save manifest for provenance
    import datetime, subprocess
    try:
        git_commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], stderr=subprocess.DEVNULL).decode().strip()
    except:
        git_commit = "unknown"
        
    manifest = {
        'created_at': datetime.datetime.utcnow().isoformat(),
        'git_commit': git_commit,
        'n_original_features': n_total,
        'n_selected_features': n_selected,
        'threshold': optimal_thresh,
        'config': {k: v for k, v in vars(config).items() if k.isupper()} if 'config' in locals() else {},
        # Methodological Enforcement
        'intended_use': 'BIOMARKER_DISCOVERY_ONLY',
        'do_not_use_for_classification': True,
        'riemannian_features_excluded': True
    }
    with open(os.path.join(output_dir, 'manifest_pruning.json'), 'w') as f:
        json.dump(manifest, f, indent=2)
    print(f"Saved manifest: {os.path.join(output_dir, 'manifest_pruning.json')}")
    
    print("\n" + "!"*60)
    print("WARNING: This output (selected_features_redundancy_pruned.csv) is for")
    print("BIOMARKER DISCOVERY ONLY. Do NOT use for classification training.")
    print("!"*60 + "\n")


if __name__ == "__main__":
    main()
