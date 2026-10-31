# EEG attention implementation

This folder packages your senior's source code with one entry point, local dependencies, the supplied feature table, and the saved discovery checkpoints. Your original folders are unchanged.

## Run in PowerShell

```powershell
cd D:\Honors\implemetation
.\run.ps1
```

This installs the pinned dependencies into `.venv` and runs pruning, Wasserstein selection, both statistical tests, cross-modal classification, fold-local LOSO, and the modality negative control. By default it starts from the senior's **existing feature table**, rather than recomputing features from the incomplete raw recordings in `D:\Honors\Honors`.

The default `paper` profile applies the explicit paper settings where the scripts disagree. Use the `senior` profile to retain the original numerical settings and original five LOSO classifiers:

```powershell
.\run.ps1 -Profile senior
```

Full bootstrap discovery and fold-local selection are computationally expensive. The full default analysis has not been run to completion during setup. Use this reduced end-to-end check first; its results are **not research results**:

```powershell
.\run.ps1 -Smoke
```

To reproduce Table V using the original saved biomarker lists, without repeating discovery or the expensive fold-local LOSO stage:

```powershell
.\run.ps1 -Stage cross-modal -Profile senior -UseReferenceResults
```

That exact analysis was run successfully during setup in `runs/reference_validation`. Its final-biomarker SVM result is **0.641246 ± 0.132872, 42 folds**, matching the paper's rounded **0.641 ± 0.133**. LDA gives **0.601391 ± 0.116262**. These are freshly trained/evaluated classifiers using inherited feature data and inherited feature lists; discovery was not repeated for this result.

If PowerShell blocks scripts, invoke Python directly; no activation is needed:

```powershell
.\.venv\Scripts\python.exe main.py
```

## Entry point options

```powershell
# Audit the packaged data and effective settings.
.\run.ps1 -Stage audit

# Recompute preprocessing and features from the available raw FIF files first.
.\run.ps1 -FromRaw

# Run an individual stage (requires preceding stage outputs).
.\run.ps1 -Stage validate
.\run.ps1 -Stage loso

# Small integration regression check.
.\.venv\Scripts\python.exe self_check.py

# Python offers a custom output directory and CSV ingestion too.
.\.venv\Scripts\python.exe main.py --from-raw --raw-source csv --output runs\from_csv
```

`--jobs`/`-Jobs` controls preprocessing and feature-extraction workers; the default is 1 to keep memory use predictable. Run `main.py --help` for all stages. The `classify` stage runs cross-modal evaluation, fold-local LOSO, and the negative control.

## Inputs and outputs

- `original/`: unchanged copies of the senior's scripts. `original_config.py` is also unchanged.
- `00_config.py`: the active central configuration; it imports every original numerical setting and supplies portable paths and the explicit profile overrides.
- `Features/features_enhanced.parquet`: byte-for-byte copy of the senior's table: **4,496 epochs, 14 subjects, 611 physiological features plus 105 covariance features and 5 metadata columns**. Subject IDs are 1–8 and 11–16; exclusions are 9 and 10.
- `reference_results/`: unchanged senior discovery outputs: **322 pruned features, 83 Wasserstein candidates, 26 final biomarkers**. They are copied into a run only with `-UseReferenceResults`.
- `runs/paper/full/` and `runs/senior/full/`: separate full-run outputs.
- `runs/paper/smoke/`: completed reduced end-to-end validation (849 epochs, 26 input features, 5 Wasserstein bootstraps, 20 statistical bootstraps).
- `runs/reference_validation/`: completed full-data cross-modal evaluation and modality negative control using the inherited discovery lists.
- `runs/preprocessing_validation/`: one real raw recording passed through the original preprocessing pipeline with the paper profile fixes.
- `archive_initial_draft/`: early inspection artifacts made before you supplied the source-code location; unused by this runner.

Each run writes `run_manifest.json` with the effective configuration, code hashes, completed stages, and whether discovery results were inherited. `audit.json`, `input_manifest.json`, and `coverage.csv` record the input data. Smoke and full outputs are kept separately. Individual stages reuse the staged feature table; use a new `--output` directory when changing the input data or settings. Stages do not automatically skip completed work.

Main result paths below each run:

| Output | Location |
|---|---|
| Discovery feature lists | `BIOMARKER DISCOVERY/Redundancy/` and `Wasserstein/` |
| Bootstrap/Wilcoxon statistics and intersection | `BIOMARKER DISCOVERY/Validation/` |
| Cross-modal fold scores, tables and plots | `BIOMARKER DISCOVERY/Invariance_Classification_Final/` |
| Fold-local LOSO scores and trained models | `BIOMARKER DISCOVERY/Classifier_Training/TrainingLogs/` |
| Attention-only modality classification | `BIOMARKER DISCOVERY/Task_Independence/` |

## Configuration and reproduction limits

The source and paper agree on 256 Hz, 14 channels, 1–45 Hz filtering, 50/60 Hz notch, extended Infomax, 95% variance, ICLabel threshold 0.8, 6 s epochs, no overlap, 200 Wasserstein bootstraps, 1,000 inference bootstraps, FDR 0.05, minimum absolute effect 0.5, and survival threshold 0.75. The original code supplies the omitted feature formulas, delta band 0.5–4 Hz, Welch segment length 512, seed 42, classifier defaults, and additional thresholds.

| Difference | `senior` profile | `paper` profile |
|---|---|---|
| Wasserstein epsilon | Original `1e-9` | Paper `1e-6` |
| LOSO pruning | Original adaptive sweep (the original function ignores its strategy argument) | Fixed distance 0.20 |
| Pruning normalization | Original relaxed-baseline normalization | All-epoch per-subject z-score |
| LOSO classifiers | SVM, RF, LDA, MLP, KNN | SVM and LDA |
| Six-second epoch endpoint | Original 1,537 samples at 256 Hz, sharing the boundary sample | 1,536 samples; no shared endpoint |
| Final common-average reference | Original preprocessing behavior | Explicitly ensured after epoch repair |
| Statistical survival | Original raw-p survival, then FDR across median p-values | Fold-wise BH-FDR survival and FDR across median p-values |

The following integration fixes apply to both profiles: NumPy 1.26-compatible trapezoidal integration; consistent subject-exclusion key; independent dispatch of the requested bootstrap/Wilcoxon method; skipping empty biomarker sets; excluding binary test folds with one class; and preferring the newly computed bootstrap/Wilcoxon intersection when available. Original scripts are preserved for comparison.

Other original choices remain visible in the source, including training-pooled normalization in statistical validation, baseline pooling, minimum-epoch filtering, and the Wasserstein sensitivity floor/ratio cutoff. The paper does not specify enough detail to establish that every original implementation choice is mathematically identical to the published description. Fresh discovery may produce different feature counts and scores; the runner does not force 322, 83, 26, or any target accuracy.

The available raw and old `PostICA` directories are incomplete. The old processed files contain 1-second epochs with a 0.1 Hz high-pass filter; they cannot be used as the paper's 6-second cleaned epochs. They are intentionally not used by the default run. `-FromRaw` processes the available continuous FIF data, but it cannot recreate missing recordings or prove the provenance of the senior's packaged feature table.

Validation performed: complete smoke analysis through all six analysis stages; full-data 42-fold cross-modal evaluation; full-data negative control; launcher audit; 611-feature extraction; 26 reference biomarker-name checks; exact 6-second epoch boundaries; BH-FDR; independent Wilcoxon dispatch; and training-only scaling. See `VERIFICATION.json` for the measured results.
