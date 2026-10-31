# Modality-invariant EEG attention biomarkers

Runnable implementation of the supplied research code, with the paper, configuration, feature table, and reference biomarker lists.

## Run on Windows

Install Python 3.11, then open PowerShell in the cloned repository:

```powershell
.\implemetation\run.ps1
```

The launcher creates a local virtual environment, installs dependencies, and runs the analysis from the included feature table. Full bootstrap discovery is computationally expensive.

For a reduced end-to-end validation:

```powershell
.\implemetation\run.ps1 -Smoke
```

To reproduce the paper's cross-modal evaluation using the supplied discovery checkpoints:

```powershell
.\implemetation\run.ps1 -Stage cross-modal -Profile senior -UseReferenceResults
```

The verified final-biomarker SVM balanced accuracy is **0.641246 ± 0.132872 across 42 folds**. This uses inherited feature data and biomarker lists; the full fresh bootstrap discovery has not been completed during setup.

See [implementation instructions](implemetation/README.md) for configuration profiles, individual stages, validation evidence, and paper/code differences.

Raw EEG recordings, virtual environments, and generated run directories are not included. The included feature table supports the default analysis. Reprocessing raw recordings requires restoring the original `Honors/EEGData` or `Honors/Attention_EEG` directories locally.
