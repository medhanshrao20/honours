"""
Centralized Reporting Utilities
--------------------------------
Provides standardized CSV schemas, manifest writing, and plot saving
for reproducible paper-ready outputs.

Reference: Reporting_csv_plots_guide_and_audit.md

⚠️ These utilities are used in EXPLORATORY and DIAGNOSTIC scripts ONLY.
Outputs support biomarker discovery and supplements — do NOT reuse in classification training.
"""

import pandas as pd
import numpy as np
import os
import json
from datetime import datetime
import hashlib

# Try to get git sha
try:
    import subprocess
    GIT_SHA = subprocess.check_output(['git', 'rev-parse', '--short', 'HEAD'], 
                                       stderr=subprocess.DEVNULL).decode().strip()
except:
    GIT_SHA = "unknown"


def get_timestamp():
    """Return ISO 8601 timestamp."""
    return datetime.now().strftime('%Y-%m-%dT%H:%M:%S')


def feature_list_hash(features):
    """Create short hash of feature list for reproducibility."""
    features_str = ','.join(sorted(features))
    return hashlib.md5(features_str.encode()).hexdigest()[:8]


def append_fold_metric(row_dict, output_dir, filename="classifier_fold_metrics.csv"):
    """
    Append a single fold metric row to classifier_fold_metrics.csv.
    Creates file with headers if it doesn't exist.
    """
    filepath = os.path.join(output_dir, filename)
    os.makedirs(output_dir, exist_ok=True)
    
    # Add timestamp
    row_dict['created_at'] = get_timestamp()
    
    # Define canonical columns (in order)
    canonical_cols = [
        'run_id', 'feature_set', 'feature_list_id', 'classifier', 'cv_type',
        'fold_id', 'test_unit', 'balanced_accuracy', 'sensitivity', 'specificity',
        'auc', 'f1', 'precision', 'recall', 'n_test_samples', 'seed', 'created_at'
    ]
    
    # Ensure all columns exist
    for col in canonical_cols:
        if col not in row_dict:
            row_dict[col] = None
    
    df = pd.DataFrame([row_dict])
    
    # Reorder columns
    cols_present = [c for c in canonical_cols if c in df.columns]
    extra_cols = [c for c in df.columns if c not in canonical_cols]
    df = df[cols_present + extra_cols]
    
    # Append or create
    if os.path.exists(filepath):
        df.to_csv(filepath, mode='a', header=False, index=False)
    else:
        df.to_csv(filepath, index=False)
    
    return filepath


def write_summary_csv(df, filepath, add_timestamp=True):
    """
    Write summary CSV with optional created_at column.
    """
    df = df.copy()
    if add_timestamp and 'created_at' not in df.columns:
        df['created_at'] = get_timestamp()
    
    df.to_csv(filepath, index=False)
    return filepath


def write_manifest(manifest_dict, output_dir, filename="run_manifest.json"):
    """
    Write run manifest JSON with config, seed, timestamp, git sha.
    """
    filepath = os.path.join(output_dir, filename)
    
    manifest = {
        'timestamp': get_timestamp(),
        'git_sha': GIT_SHA,
        **manifest_dict
    }
    
    with open(filepath, 'w') as f:
        json.dump(manifest, f, indent=2, default=str)
    
    return filepath


def write_pipeline_manifest(data_dict, output_dir, filename="manifest.json"):
    """
    General helper to write a pipeline stage manifest.
    Automatically adds timestamp and git_sha.
    """
    data_dict['created_at'] = get_timestamp()
    data_dict['git_sha'] = GIT_SHA
    
    os.makedirs(output_dir, exist_ok=True)
    filepath = os.path.join(output_dir, filename)
    
    with open(filepath, 'w') as f:
        json.dump(data_dict, f, indent=2, default=str)
    
    return filepath


def save_plot_with_caption(fig, filepath, caption, dpi=150):
    """
    Save matplotlib figure and a companion caption .txt file.
    """
    # Save figure
    fig.savefig(filepath, dpi=dpi, bbox_inches='tight')
    
    # Save caption
    caption_path = filepath.rsplit('.', 1)[0] + '_caption.txt'
    with open(caption_path, 'w') as f:
        f.write(caption)
    
    return filepath, caption_path


def create_classifier_summary(fold_df, group_cols=['run_id', 'feature_set', 'classifier', 'cv_type']):
    """
    Create summary metrics from fold-level data with CIs.
    """
    summary_rows = []
    
    for name, group in fold_df.groupby(group_cols):
        n_folds = len(group)
        
        # Calculate stats
        mean_bal_acc = group['balanced_accuracy'].mean()
        std_bal_acc = group['balanced_accuracy'].std()
        
        # Bootstrap CI (simple percentile)
        if n_folds >= 5:
            boot_means = []
            for _ in range(1000):
                sample = group['balanced_accuracy'].sample(n=n_folds, replace=True)
                boot_means.append(sample.mean())
            ci_low = np.percentile(boot_means, 2.5)
            ci_high = np.percentile(boot_means, 97.5)
        else:
            ci_low = mean_bal_acc - 1.96 * std_bal_acc / np.sqrt(n_folds)
            ci_high = mean_bal_acc + 1.96 * std_bal_acc / np.sqrt(n_folds)
        
        row = dict(zip(group_cols, name if isinstance(name, tuple) else [name]))
        row.update({
            'mean_bal_acc': mean_bal_acc,
            'std_bal_acc': std_bal_acc,
            'ci_low': ci_low,
            'ci_high': ci_high,
            'mean_auc': group['auc'].mean() if 'auc' in group.columns else None,
            'std_auc': group['auc'].std() if 'auc' in group.columns else None,
            'mean_f1': group['f1'].mean() if 'f1' in group.columns else None,
            'n_folds': n_folds,
            'seed': group['seed'].iloc[0] if 'seed' in group.columns else None,
            'created_at': get_timestamp()
        })
        summary_rows.append(row)
    
    return pd.DataFrame(summary_rows)


# ============================================================
# LOSO-specific schemas
# ============================================================

def format_loso_fold_result(feature, fold_idx, test_subject, n_train, K, p_value, 
                            median_resample_p, n_resamples, passed):
    """Format a single LOSO fold result row."""
    return {
        'feature': feature,
        'fold_idx': fold_idx,
        'test_subject': test_subject,
        'n_train_subjects': n_train,
        'K': K,
        'p_value_fold': p_value,
        'median_resample_p': median_resample_p,
        'n_resamples': n_resamples,
        'passed_fold_bool': passed,
        'fold_notes': '',
        'created_at': get_timestamp()
    }


def format_loso_summary(feature, total_folds, passed_folds, survival_rate, 
                        combined_p, median_p, adjusted_p, final_decision):
    """Format a single LOSO summary row."""
    return {
        'feature': feature,
        'total_folds': total_folds,
        'passed_folds': passed_folds,
        'survival_rate': survival_rate,
        'combined_p_stouffer': combined_p,
        'median_fold_p': median_p,
        'adjusted_p_fdr': adjusted_p,
        'final_decision': final_decision,
        'created_at': get_timestamp()
    }


# ============================================================
# Wasserstein-specific schemas  
# ============================================================

def format_wasserstein_feature(feature, w_modal_med, w_modal_ci, w_state_med, w_state_ci,
                               conv_ratio, n_subjects, p_val, ratio_iqr, selected, notes=''):
    """Format a single Wasserstein feature score row."""
    return {
        'feature': feature,
        'W_Modal_Median': w_modal_med,
        'W_Modal_CI_low': w_modal_ci[0] if w_modal_ci else None,
        'W_Modal_CI_high': w_modal_ci[1] if w_modal_ci else None,
        'W_State_Median': w_state_med,
        'W_State_CI_low': w_state_ci[0] if w_state_ci else None,
        'W_State_CI_high': w_state_ci[1] if w_state_ci else None,
        'Convergence_Ratio': conv_ratio,
        'Valid_Subjects': n_subjects,
        'P_Value_RatioLT1': p_val,
        'Ratio_IQR': ratio_iqr,
        'selection_flag': 'candidate' if selected else 'rejected',
        'notes': notes,
        'created_at': get_timestamp()
    }


def format_wasserstein_subject(feature, subject, w_modal, w_state, ratio, n_modalities):
    """Format a single Wasserstein per-subject row."""
    return {
        'feature': feature,
        'subject': subject,
        'W_Modal': w_modal,
        'W_State': w_state,
        'Ratio': ratio,
        'valid_modality_count': n_modalities,
        'created_at': get_timestamp()
    }


# ============================================================
# Pruning-specific schemas
# ============================================================

def format_pruning_cluster(cluster_id, medoid, size, members, distance_mean):
    """Format a single pruning cluster info row."""
    return {
        'cluster_id': cluster_id,
        'medoid_feature': medoid,
        'size': size,
        'member_count': len(members),
        'member_features_json': json.dumps(members),
        'cluster_distance_mean': distance_mean,
        'created_at': get_timestamp()
    }
