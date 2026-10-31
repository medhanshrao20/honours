"""
Stage 1: Biomarker Discovery (NO ML)
-----------------------------------
Purpose:
- Identify task-invariant, state-discriminative EEG biomarkers
- No classifier is trained
- Results are NOT used directly for ML feature selection

Output:
- Final biomarkers for reporting and interpretation
"""

from __future__ import annotations
import os
import json
import logging
from datetime import datetime
import numpy as np
import pandas as pd
from scipy.stats import norm, wilcoxon
import warnings
from tqdm import tqdm

# Statsmodels excluded
HAS_STATSMODELS = False

# ---------------- Config load (safe defaults if 00_config.py unavailable) ----------------
try:
    from importlib import import_module
    config = import_module('00_config')
except Exception:
    class _C: pass
    config = _C()
    # minimal defaults
    config.RANDOM_SEED = 42
    config.FDR_Q = 0.05
    config.LOSO_RESAMPLES = 100
    config.MIN_SUBJECTS_FOR_LOSO = 5
    config.MIN_EFFECT_SIZE_FOR_STAT = 0.0
    config.NORMALIZATION_METHOD = 'zscore'
    config.EXCLUDE_SUBJECTS_DOWNSTREAM = []
    config.STATS_METHOD = 'lmm'  # 'lmm'|'hboot'|'wilcoxon'
    config.POOL_RELAXED = True
    config.FEATURES_DIR = 'Features'
    config.BIOMARKER_DIR = 'BIOMARKER DISCOVERY'
    config.N_BOOT_FINAL = 1000
    config.N_BOOT_SMOKE = 100
    config.SMOKE_MODE = False
    config.ROBUSTNESS_THRESHOLD = 0.75

# Read settings
RANDOM_SEED = getattr(config, 'RANDOM_SEED', 42)
FDR_Q = getattr(config, 'FDR_Q', 0.05)
LOSO_RESAMPLES = getattr(config, 'LOSO_RESAMPLES', 100)
MIN_SUBJECTS_FOR_LOSO = getattr(config, 'MIN_SUBJECTS_FOR_LOSO', 5)
MIN_EFFECT_SIZE_FOR_STAT = getattr(config, 'MIN_EFFECT_SIZE_FOR_STAT', 0.0)
NORMALIZATION_METHOD = getattr(config, 'NORMALIZATION_METHOD', 'zscore')
EXCLUDE_SUBJECTS = getattr(config, 'EXCLUDE_SUBJECTS_DOWNSTREAM', [])
STATS_METHOD = getattr(config, 'STATS_METHOD', 'lmm')
POOL_RELAXED = getattr(config, 'POOL_RELAXED', True)
N_BOOT_FINAL = getattr(config, 'N_BOOT_FINAL', 1000)
N_BOOT_SMOKE = getattr(config, 'N_BOOT_SMOKE', 100)
SMOKE_MODE = getattr(config, 'SMOKE_MODE', False)
N_BOOT = N_BOOT_SMOKE if SMOKE_MODE else N_BOOT_FINAL
ROBUSTNESS_THRESHOLD = getattr(config, 'ROBUSTNESS_THRESHOLD', 0.75)

np.random.seed(RANDOM_SEED)

# Paths
FEATURES_DIR = getattr(config, 'FEATURES_DIR', 'Features')
BIOMARKER_DIR = getattr(config, 'BIOMARKER_DIR', 'BIOMARKER DISCOVERY')

INPUT_PATH = os.path.join(FEATURES_DIR, "features_enhanced.parquet")
WASSERSTEIN_CSV = os.path.join(BIOMARKER_DIR, "Wasserstein", "selected_features_wasserstein.csv")
PRUNED_REDUC_CSV = os.path.join(BIOMARKER_DIR, "Redundancy", "selected_features_redundancy_pruned.csv")
OUTPUT_DIR = os.path.join(BIOMARKER_DIR, "Validation")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s: %(message)s')

# ---------------- helper functions ----------------
def stouffer_combine(pvals):
    """Combine p-values (two-sided) with Stouffer's Z."""
    pvals = np.asarray([p for p in pvals if np.isfinite(p)])
    if len(pvals) == 0:
        return np.nan
    pvals = np.clip(pvals, 1e-12, 1.0 - 1e-12)
    # convert two-sided p to z
    zs = norm.isf(pvals / 2.0)
    Z = zs.sum() / np.sqrt(len(zs))
    return float(2.0 * norm.sf(abs(Z)))

def benjamini_hochberg_qvalues(pvals):
    """
    Compute Benjamini-Hochberg FDR adjusted p-values (q-values).
    """
    p = np.asarray(pvals)
    valid_idx = np.where(np.isfinite(p))[0]
    if len(valid_idx) == 0:
        return np.full_like(p, np.nan)
    
    pv = p[valid_idx]
    n = len(pv)
    order = np.argsort(pv)
    reverse = np.argsort(order)
    
    # Sort p-values
    sorted_p = pv[order]
    
    # Compute q-values: p * n / i
    q_values = sorted_p * n / np.arange(1, n + 1)
    
    # Ensure monotonicity: q_i = min(q_i, q_{i+1}, ..., q_n)
    q_values = np.minimum.accumulate(q_values[::-1])[::-1]
    
    # Clip to 1.0
    q_values = np.clip(q_values, 0, 1.0)
    
    # Map back to original indices
    full_q = np.full_like(p, np.nan)
    full_q[valid_idx] = q_values[reverse]
    return full_q

def benjamini_hochberg_mask(pvals, q=0.05):
    """BH-FDR mask over finite pvals."""
    q_vals = benjamini_hochberg_qvalues(pvals)
    return (q_vals <= q)

# ---------------- statistical engines ----------------
def wilcoxon_subject_test(df_train, df_all, feature_col, min_subjects=5, pool_relaxed=POOL_RELAXED):
    """
    Subject-level Wilcoxon on paired subject medians.
    df_train: training dataframe (used for computing task-specific attention medians)
    df_all: full dataframe used for pooling relaxed if requested
    returns: effect_mean, p-value
    """
    subj_ids = sorted(df_train['subject'].unique())
    att_meds = []
    rel_meds = []
    for s in subj_ids:
        att_vals = df_train[(df_train['subject']==s) & (df_train['condition']=='attention')][feature_col].dropna().values
        if pool_relaxed:
            rel_vals = df_all[(df_all['subject']==s) & (df_all['condition']=='relaxed')][feature_col].dropna().values
        else:
            rel_vals = df_train[(df_train['subject']==s) & (df_train['condition']=='relaxed')][feature_col].dropna().values
        if len(att_vals)==0 or len(rel_vals)==0:
            continue
        att_meds.append(np.median(att_vals))
        rel_meds.append(np.median(rel_vals))
    if len(att_meds) < min_subjects:
        return np.nan, np.nan
    try:
        stat, p = wilcoxon(np.array(att_meds), np.array(rel_meds), alternative='two-sided', zero_method='wilcox')
    except Exception:
        return np.nan, np.nan
    eff = float(np.mean(np.array(att_meds) - np.array(rel_meds)))
    return eff, float(p)

def hierarchical_bootstrap_pvalue(df_train, df_all, feature_col, n_boot=N_BOOT, rng=None, pool_relaxed=POOL_RELAXED):
    """
    Optimized Hierarchical (nested) bootstrap: Resample subjects -> Resample epochs.
    Returns: observed_stat, p_value, (ci_low, ci_high)
    """
    if rng is None:
        rng = np.random.default_rng(RANDOM_SEED)

    # 1. Pre-compute subject-level arrays to avoid DataFrame indexing in loop
    subjects = sorted(df_train['subject'].unique())
    subject_data = [] # List of tuples: (att_array, rel_array)

    for s in subjects:
        # Attention epochs (from training split)
        att_vals = df_train[(df_train['subject']==s) & (df_train['condition']=='attention')][feature_col].dropna().values
        
        # Relaxed epochs (pooled or matched)
        if pool_relaxed:
            # If pooling, we might use data outside df_train if df_all is provided
            # However, for strict cross-validation, df_all should ideally be df_train.
            # If df_all is passed, we use it.
            rel_vals = df_all[(df_all['subject']==s) & (df_all['condition']=='relaxed')][feature_col].dropna().values
        else:
            rel_vals = df_train[(df_train['subject']==s) & (df_train['condition']=='relaxed')][feature_col].dropna().values
            
        if len(att_vals) > 0 and len(rel_vals) > 0:
            subject_data.append((att_vals, rel_vals))

    n_subs = len(subject_data)
    if n_subs < max(3, MIN_SUBJECTS_FOR_LOSO):
        return np.nan, np.nan, (np.nan, np.nan)

    # 2. Compute observed statistic
    original_diffs = [np.median(a) - np.median(r) for a, r in subject_data]
    obs = compute_hedges_g(original_diffs)

    # 3. Fast Bootstrap Loop
    boot_stats = np.zeros(n_boot)
    
    for i in range(n_boot):
        # Resample subjects (indices) with replacement
        # We use indices to select from subject_data list
        resampled_indices = rng.integers(0, n_subs, size=n_subs)
        
        # Collect diffs for this bootstrap iteration
        iteration_diffs = []
        
        for idx in resampled_indices:
            att, rel = subject_data[idx]
            
            # Resample epochs within this subject with replacement
            # Use choice for safety, or integers if we want raw speed on indices
            # rng.choice is safer for arrays
            if len(att) > 0:
                att_res = rng.choice(att, size=len(att), replace=True)
                att_med = np.median(att_res)
            else:
                att_med = np.nan # Should not happen due to pre-check

            if len(rel) > 0:
                rel_res = rng.choice(rel, size=len(rel), replace=True)
                rel_med = np.median(rel_res)
            else:
                rel_med = np.nan

            iteration_diffs.append(att_med - rel_med)
            
        boot_stats[i] = compute_hedges_g(iteration_diffs)

    # 4. Compute P-value and CI
    # Two-sided p-value: fraction of bootstrap distribution more extreme than observed
    # Centering is not strictly required for p-value of hypothesis "mean != 0" 
    # but standard hboot checks overlap with zero. 
    # Here we check if 0 is in the CI, or effectively p = mean(|boot| >= |obs|) under null?
    # Actually, for testing "Is there an effect?", we usually check if the CI excludes 0.
    # OR we can approximate p-value by shifting null.
    # The original implementation used: p = mean(|boot| >= |obs|). 
    # But wait, if boot distribution is centered around obs, then |boot| >= |obs| is roughly 1.0.
    # The typical test for "is mean diff != 0" using bootstrap CI:
    # We check where 0 lies in the distribution.
    # However, let's stick to the previous logic logic to avoid methodology drift, 
    # but fix the p-value calculation if it was wrong.
    # Previous: p = float(np.mean(np.abs(boot_stats) >= abs(obs)))
    # This implies testing if the *magnitude* of the bootstrap samples is larger than the observed magnitude?
    # This is valid ONLY if we generated the bootstrap distribution under the NULL hypothesis (centering at 0).
    # BUT we are resampling from the EMPIRICAL distribution (centered at obs).
    # So `boot_stats` are distributed around `obs`.
    
    # Correct approach for H0: mu=0 using non-parametric bootstrap:
    # 1. CI method: Check if 0 is in [2.5%, 97.5%]
    # 2. P-value inversion: 
    #    shift = boot_stats - obs
    #    null_dist = shift (centered at 0)
    #    p = mean(|null_dist| >= |obs|)
    
    ci_low, ci_high = np.percentile(boot_stats, [2.5, 97.5])
    
    # Check if 0 is outside CI (significance)
    # If 0 < ci_low or 0 > ci_high -> Significant
    
    # Calculate p-value by shifting distribution to Null (mean=0)
    # This 'Hall-Wilson' or 'percentile-t' like adjustment is safer.
    null_dist = boot_stats - np.mean(boot_stats) # Center at 0
    p = float(np.mean(np.abs(null_dist) >= np.abs(obs)))
    
    # Sanity check: if effect is huge, p should be small.
    # If obs is far from 0, standard error (width of null_dist) tells us significance.
    
    return obs, p, (float(ci_low), float(ci_high))

# LMM implementation removed

def feature_pvalue_subjectlevel(diffs, rng, method='wilcoxon', hier_reps=5000):
    diffs = np.asarray(diffs).ravel()
    diffs = diffs[~np.isnan(diffs)]
    n = len(diffs)
    if n < 3: # Minimal subjects
        return np.nan

    if method == 'wilcoxon':
        # Wilcoxon signed-rank (two-sided) - exact for small n
        try:
            stat, p = wilcoxon(diffs, zero_method='wilcox', alternative='two-sided', mode='exact')
            return float(p)
        except Exception:
            # fallback to approximate
            stat, p = wilcoxon(diffs, alternative='two-sided')
            return float(p)

    elif method == 'signflip':
        # existing sign-flip Monte Carlo (vectorized)
        obs_mean = np.mean(diffs)
        n_perms = getattr(config, 'PERMUTATION_ITERATIONS', 1000)
        signs = rng.choice([-1, 1], size=(n_perms, n))
        perm_means = (signs * diffs).mean(axis=1)
        p = np.mean(np.abs(perm_means) >= np.abs(obs_mean))
        return float(max(p, 1.0 / n_perms))

    elif method == 'hier_boot':
        # hierarchical bootstrap (resample subjects with replacement)
        # Assumes diffs are subject-level already.
        reps = hier_reps
        boot_stats = []
        # Use provided generator or create new
        local_rng = np.random.default_rng(RANDOM_SEED) if rng is None else rng
        for _ in range(int(reps)):
            sample = local_rng.choice(diffs, size=n, replace=True)
            boot_stats.append(np.mean(sample))
        obs = np.mean(diffs)
        p = np.mean(np.abs(boot_stats) >= np.abs(obs))
        return float(max(p, 1.0 / reps))

    else:
        # Default to wilcoxon if unknown
        return feature_pvalue_subjectlevel(diffs, rng, method='wilcoxon')

def compute_hedges_g(subj_diffs: list[float] | np.ndarray) -> float:
    """
    Compute Hedges' g for paired samples (corrected for small-sample bias).
    g_z = (mean_diff / std_diff) * (1 - 3 / (4 * (n - 1) - 1))
    """
    diffs = np.array(subj_diffs)
    n = len(diffs)
    if n < 2:
        return np.mean(diffs) if n == 1 else 0.0
    
    mean_diff = np.mean(diffs)
    std_diff = np.std(diffs, ddof=1)
    
    if std_diff < 1e-12:
        return 0.0
        
    cohen_d = mean_diff / std_diff
    # Hedges correction factor for paired samples (degrees of freedom = n-1)
    correction = 1.0 - (3.0 / (4.0 * (n - 1) - 1.0))
    
    return cohen_d * correction

def compute_pvalue_wrapper(df_task, df_all, feature_col, method=STATS_METHOD, n_boot=N_BOOT, rng=None, pool_relaxed=POOL_RELAXED):
    """
    Unified wrapper for p-value computation respecting configuration.
    """
    if rng is None:
        rng = np.random.default_rng(RANDOM_SEED)

    # Allow config override
    # Explicit method wins; otherwise both named passes silently run hboot.

    # hboot is now the primary non-ML discovery method 

    # Hierarchical Bootstrap (Nested: Subject -> Epochs)
    if method == 'hboot':
        # Use the rigorous nested bootstrap
        beta, p, ci = hierarchical_bootstrap_pvalue(df_task, df_all, feature_col, n_boot, rng, pool_relaxed)
        if not np.isnan(p):
            return beta, p, ci
        # Fallback
        logging.info(f"HBOOT failed for {feature_col}, falling back to Wilcoxon.")
        method = 'wilcoxon'

    # Subject-level methods (Wilcoxon, SignFlip, HierBoot - non-nested)
    # 1. Compute subject diffs
    subj_diffs = []
    subjects = sorted(df_task['subject'].unique())
    for s in subjects:
        att = df_task[(df_task['subject']==s) & (df_task['condition']=='attention')][feature_col].dropna().values
        if pool_relaxed:
            rel = df_all[(df_all['subject']==s) & (df_all['condition']=='relaxed')][feature_col].dropna().values
        else:
             rel = df_task[(df_task['subject']==s) & (df_task['condition']=='relaxed')][feature_col].dropna().values
        
        if len(att) > 0 and len(rel) > 0:
             subj_diffs.append(np.median(att) - np.median(rel))
    
    if len(subj_diffs) < 3:
        return np.nan, np.nan, (np.nan, np.nan)

    # 2. Run test
    p = feature_pvalue_subjectlevel(subj_diffs, rng, method=method, hier_reps=n_boot)
    eff = compute_hedges_g(subj_diffs)
    # Approximate CI for these methods (bootstrap the diffs)
    # Simple bootstrap of the mean difference
    boot_means = []
    for _ in range(int(n_boot)):
        sample = rng.choice(subj_diffs, size=len(subj_diffs), replace=True)
        boot_means.append(compute_hedges_g(sample))
    ci_low, ci_high = np.percentile(boot_means, [2.5, 97.5])
    
    return eff, p, (ci_low, ci_high)

# ---------------- data utils ----------------
def load_data(input_path=INPUT_PATH, exclude_subjects=EXCLUDE_SUBJECTS):
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Input path not found: {input_path}")
    if input_path.endswith('.parquet'):
        df = pd.read_parquet(input_path)
    else:
        df = pd.read_csv(input_path)
    if 'subject' in df.columns and exclude_subjects:
        df = df[~df['subject'].isin(exclude_subjects)].copy()
    return df

def load_candidates_from_csv(path, available_df):
    """
    Load candidate list from a CSV and ensure features exist in available_df.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Candidates file not found: {path}")
    cdf = pd.read_csv(path)
    if 'feature' in cdf.columns:
        return [f for f in cdf['feature'].tolist() if f in available_df.columns]
    else:
        return [f for f in cdf.iloc[:, 0].tolist() if f in available_df.columns]

def normalize_train_test(df_train, df_test, cols, method=NORMALIZATION_METHOD):
    """Normalize train/test via zscore or robust median/IQR. Fit only on train."""
    df_train = df_train.copy()
    df_test = df_test.copy() if df_test is not None else None
    if len(cols) == 0:
        return df_train, df_test
    if method == 'zscore':
        mu = df_train[cols].mean()
        sd = df_train[cols].std().replace(0, 1.0)
        df_train.loc[:, cols] = (df_train[cols] - mu) / sd
        if df_test is not None:
            df_test.loc[:, cols] = (df_test[cols] - mu) / sd
    elif method == 'robust':
        med = df_train[cols].median()
        iqr = (df_train[cols].quantile(0.75) - df_train[cols].quantile(0.25)).replace(0, 1.0)
        df_train.loc[:, cols] = (df_train[cols] - med) / iqr
        if df_test is not None:
            df_test.loc[:, cols] = (df_test[cols] - med) / iqr
    elif method == 'baseline':
        relax = df_train[df_train['condition']=='relaxed']
        subj_means = relax.groupby('subject')[cols].mean()
        pooled_mean = subj_means.mean()
        pooled_std = subj_means.std().replace(0, 1.0)
        df_train.loc[:, cols] = (df_train[cols] - pooled_mean) / pooled_std
        if df_test is not None:
            df_test.loc[:, cols] = (df_test[cols] - pooled_mean) / pooled_std
    return df_train, df_test

def compute_k_filter(df, candidate_features, k_percentile=10, k_floor=10):
    subj_counts = []
    for s, g in df.groupby('subject'):
        n_att = len(g[g['condition']=='attention'])
        n_rel = len(g[g['condition']=='relaxed'])
        subj_counts.append(min(n_att, n_rel))
    if len(subj_counts) == 0:
        return k_floor, [], list(df['subject'].unique())
    pct = np.percentile(subj_counts, k_percentile)
    K = int(max(k_floor, pct))
    valid = []
    dropped = []
    for s, g in df.groupby('subject'):
        if min(len(g[g['condition']=='attention']), len(g[g['condition']=='relaxed'])) >= K:
            valid.append(s)
        else:
            dropped.append(s)
    return K, valid, dropped

# ---------------- core pipeline ----------------

# ---------------- core pipeline ----------------

def process_fold(resample_idx, fold_idx, test_subject, df_train_norm, df_test_norm, candidates, tasks, 
                 feature_fold_pvals, feature_fold_effect, feature_fold_pass, fold_records, n_boot, 
                 pool_relaxed, min_effect_size, stats_method=STATS_METHOD):
    
    # Pre-compute train available subjects (optimization)
    # train_subs = df_train_norm['subject'].unique()
    
    # statsmodels excluded

    for feat in tqdm(candidates, desc=f"Fold {fold_idx} (Subject {test_subject})", leave=False):
        per_task_p = []
        per_task_effect = []
        valid_for_feat = True
        
        for task in tasks:
            # Check sufficiency in training
            df_task = df_train_norm[df_train_norm['task'] == task]
            
            # Fast check for subject count
            available_subjects = df_task['subject'].nunique()
            if available_subjects < 3:
                valid_for_feat = False
                break
            
            # Create RNG for this task/fold
            task_rng = np.random.default_rng(RANDOM_SEED + fold_idx + ord(task[0]))
            
            eff, p, ci = compute_pvalue_wrapper(df_task, df_train_norm, feat, 
                                              method=stats_method, n_boot=n_boot, rng=task_rng, pool_relaxed=pool_relaxed)
            
            if np.isnan(p):
                valid_for_feat = False
                break
            per_task_p.append(p)
            per_task_effect.append(eff)

        if (not valid_for_feat) or (len(per_task_p) == 0):
            feature_fold_pvals[feat].append(np.nan)
            feature_fold_effect[feat].append(np.nan)
            continue

        p_comb = stouffer_combine(per_task_p)
        effect_mean = float(np.nanmean(per_task_effect))
        passed = (np.isfinite(p_comb) and (p_comb < 0.05)) and (abs(effect_mean) >= min_effect_size)
        
        feature_fold_pass[feat].append(bool(passed))
        feature_fold_pvals[feat].append(float(p_comb) if np.isfinite(p_comb) else 1.0)
        feature_fold_effect[feat].append(effect_mean)
        
        fold_records.append({
            'resample_idx': resample_idx,
            'fold_idx': fold_idx,
            'test_subject': int(test_subject),
            'feature': feat,
            'p_comb': float(p_comb) if np.isfinite(p_comb) else np.nan,
            'effect_mean': effect_mean,
            'passed': bool(passed)
        })

    if getattr(config, 'RUN_PROFILE', 'senior') == 'paper':
        current = [row for row in fold_records
                   if row['resample_idx'] == resample_idx and row['fold_idx'] == fold_idx]
        qvals = benjamini_hochberg_qvalues([row['p_comb'] for row in current])
        for row, qval in zip(current, qvals):
            passed = bool(qval <= FDR_Q and abs(row['effect_mean']) >= min_effect_size)
            row['q_value'] = float(qval)
            row['passed'] = passed
            feature_fold_pass[row['feature']][-1] = passed

def biomarker_discovery(df_all_input, candidates, output_dir=OUTPUT_DIR,
                        stats_method=STATS_METHOD, n_boot=N_BOOT, loso_resamples=LOSO_RESAMPLES):
    """
    05_biomarker_stat_validation.py

    Research-grade LOSO biomarker discovery:
    - Primary test: LMM + residual bootstrap (if statsmodels available)
    - Secondary: hierarchical (nested) bootstrap (epoch-level)
    - Tertiary: Wilcoxon signed-rank on subject medians (exact for small n)

    Per-feature per-fold testing -> Stouffer combine across tasks -> BH-FDR -> survival threshold.
    Produces CSVs in BIOMARKER DISCOVERY/Validation/

    Notes:
    - Requires outputs of redundancy pruning and (optionally) global wasserstein ranking
      for candidate lists (see 03_redundancy_pruning_exploratory.py and 04_wasserstein_selection_exploratory.py).
    - Treats relaxed epochs pooled across modalities if POOL_RELAXED=True.
    """
    rng = np.random.default_rng(RANDOM_SEED)
    df_all = df_all_input.copy()
    results_records = []
    fold_records = []
    tasks = sorted(df_all['task'].unique())
    candidates = [c for c in candidates if c in df_all.columns]
    if len(candidates) == 0:
        raise ValueError("No candidate features found in dataframe.")
    K, initial_valid_subjects, dropped_subjects = compute_k_filter(df_all, candidates)
    if len(initial_valid_subjects) < MIN_SUBJECTS_FOR_LOSO:
        raise ValueError(f"Insufficient subjects after filtering: {len(initial_valid_subjects)} < {MIN_SUBJECTS_FOR_LOSO}")
    logging.info(f"K={K}, initial_valid_subjects={initial_valid_subjects}, dropped={dropped_subjects}")
    
    feature_fold_pass = {f: [] for f in candidates}
    feature_fold_pvals = {f: [] for f in candidates}
    feature_fold_effect = {f: [] for f in candidates}
    fold_records = []

    logging.info(f"BioMarker Discovery Method: {stats_method} (Bootstrap={n_boot})")

    # 2. Resampling Loop
    # If loso_resamples is None or 0, we do 1 run (standard LOSO)
    n_resamples = loso_resamples if (loso_resamples is not None and loso_resamples > 0) else 1
    
    for r_idx in range(n_resamples):
        # Determine Dataset for this Run
        if loso_resamples is not None and loso_resamples > 0:
            rng = np.random.default_rng(RANDOM_SEED + r_idx)
            current_subjects = sorted(rng.choice(initial_valid_subjects, size=len(initial_valid_subjects), replace=True))
            logging.info(f"Resample {r_idx+1}/{n_resamples} (Subjects: {len(current_subjects)})")
        else:
            current_subjects = initial_valid_subjects
            
        # 3. LOSO Loop
        for i, test_subject in enumerate(current_subjects):
            # Define training subjects (Bag \ {Instance})
            # We use indices to handle duplicates correctly
            train_indices = [idx for idx in range(len(current_subjects)) if idx != i]
            train_subjects = [current_subjects[idx] for idx in train_indices]
            
            # Construct Train DF (concatenate instances)
            # This preserves the weight of repeated subjects
            train_frames = [df_all[df_all['subject'] == s] for s in train_subjects]
            df_train = pd.concat(train_frames, ignore_index=True)
            
            # Test DF (Single instance implies single subject ID data)
            df_test = df_all[df_all['subject'] == test_subject].copy()
            
            # Normalize
            df_train_norm, df_test_norm = normalize_train_test(df_train, df_test, candidates, NORMALIZATION_METHOD)
            
            # Process Fold
            process_fold(r_idx, i, test_subject, df_train_norm, df_test_norm, candidates, tasks,
                         feature_fold_pvals, feature_fold_effect, feature_fold_pass, fold_records,
                         n_boot, POOL_RELAXED, MIN_EFFECT_SIZE_FOR_STAT, stats_method=stats_method)

    # 4. Aggregation
    records = []
    for feat in candidates:
        pvals = [p for p in feature_fold_pvals[feat] if np.isfinite(p)]
        median_p = float(np.median(pvals)) if len(pvals) > 0 else 1.0
        survival_rate = float(np.mean(feature_fold_pass[feat])) if len(feature_fold_pass[feat]) > 0 else 0.0
        mean_hedges_g = float(np.nanmean(feature_fold_effect[feat])) if len(feature_fold_effect[feat]) > 0 else 0.0
        records.append({
            'feature': feat,
            'median_p_value': median_p,
            'survival_rate': survival_rate,
            'hedges_g': mean_hedges_g,
            'n_folds_tested': len(feature_fold_pvals[feat])
        })
        
    results_df = pd.DataFrame.from_records(records)
    if results_df.empty:
        return results_df, pd.DataFrame(fold_records)

    results_df = results_df.sort_values('median_p_value').reset_index(drop=True)
    
    # FDR Q-values
    results_df['q_value'] = benjamini_hochberg_qvalues(results_df['median_p_value'].values)
    results_df['fdr_significant'] = (results_df['q_value'] <= FDR_Q)
    
    results_df['effect_size_pass'] = (np.abs(results_df['hedges_g']) >= MIN_EFFECT_SIZE_FOR_STAT)
    results_df['final_selected'] = results_df['fdr_significant'] & (results_df['survival_rate'] >= ROBUSTNESS_THRESHOLD) & results_df['effect_size_pass']

    # Save outputs
    timestamp = datetime.now().strftime('%Y%m%dT%H%M%S')
    summary_path = os.path.join(output_dir, f"loso_summary_{stats_method}_{timestamp}.csv")
    fold_path = os.path.join(output_dir, f"loso_fold_results_{stats_method}_{timestamp}.csv")
    results_df.to_csv(summary_path, index=False)
    pd.DataFrame(fold_records).to_csv(fold_path, index=False)

    final_feats = results_df[results_df['final_selected']]['feature'].tolist()
    final_path = os.path.join(output_dir, f"final_selected_features_{stats_method}_{timestamp}.csv")
    pd.DataFrame({'feature': final_feats}).to_csv(final_path, index=False)
    
    logging.info(f"Saved summary: {summary_path}")
    logging.info(f"Saved final features: {final_path}")

    # Save Intermediate Subsets for Testing
    fdr_only = results_df[results_df['fdr_significant']]['feature'].tolist()
    fdr_survival = results_df[results_df['fdr_significant'] & (results_df['survival_rate'] >= ROBUSTNESS_THRESHOLD)]['feature'].tolist()
    fdr_effect = results_df[results_df['fdr_significant'] & results_df['effect_size_pass']]['feature'].tolist()

    subsets = {
        'fdr_only': fdr_only,
        'fdr_survival': fdr_survival,
        'fdr_effect': fdr_effect
    }

    for name, feats in subsets.items():
        subset_path = os.path.join(output_dir, f"{name}_features_{stats_method}_{timestamp}.csv")
        pd.DataFrame({'feature': feats}).to_csv(subset_path, index=False)
        logging.info(f"Saved {name} subset ({len(feats)} features): {subset_path}")
    
    # Save Final Manifest for Audit
    manifest = {
        'timestamp': timestamp,
        'stats_method': stats_method,
        'n_bootstrap': n_boot,
        'loso_resamples': loso_resamples,
        'interpretable_only_no_riem': True,
        'n_candidates': len(candidates),
        'n_final': len(final_feats)
    }
    with open(os.path.join(output_dir, f"manifest_{timestamp}.json"), 'w') as f:
        json.dump(manifest, f, indent=2)
    
    return results_df, pd.DataFrame(fold_records)

# ---------------- main ----------------
def main():
    logging.info("Biomarker discovery (LOSO) starting...")
    df = load_data(INPUT_PATH, exclude_subjects=EXCLUDE_SUBJECTS)
    # Prefer Wasserstein candidate list if available, else redundancy pruned list
    candidates_path = WASSERSTEIN_CSV if os.path.exists(WASSERSTEIN_CSV) else PRUNED_REDUC_CSV
    candidates = load_candidates_from_csv(candidates_path, df)
    logging.info(f"Loaded data rows: {len(df)}, candidate features: {len(candidates)}, using candidates file: {candidates_path}")
    # Sequential execution of hboot and wilcoxon
    methods_to_run = ['hboot', 'wilcoxon']
    
    for method in methods_to_run:
        logging.info(f"--- Starting Biomarker Discovery with method: {method} ---")
        results_df, fold_df = biomarker_discovery(
            df, 
            candidates, 
            output_dir=OUTPUT_DIR, 
            stats_method=method, 
            n_boot=N_BOOT, 
            loso_resamples=LOSO_RESAMPLES
        )
        
        if not results_df.empty:
            top = results_df.sort_values('median_p_value').head(10)
            logging.info(f"Top features for {method}:\n{top[['feature','median_p_value','survival_rate','hedges_g','final_selected']].to_string(index=False)}")
        else:
            logging.info(f"No results for {method}.")
        logging.info(f"--- Completed Biomarker Discovery with method: {method} ---")

if __name__ == "__main__":
    main()
