"""Small regression checks for the entry point's critical integration fixes."""
import importlib
import os
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
os.environ.update(EEG_RUN_DIR=str(ROOT/'runs/self_check'),EEG_PROFILE='paper',MPLBACKEND='Agg')
cfg = importlib.import_module('00_config')
extract = importlib.import_module('02_feature_extraction')
pre = importlib.import_module('01_preprocessing')
validation = importlib.import_module('05_biomarker_stat_validation')
loso = importlib.import_module('07_standard_loso_classification')

rng = np.random.default_rng(42)
data = rng.normal(size=(14,1536))*1e-5
features = {f'{ch}_{key}':value for ch,x in zip(cfg.EEG_CHANNELS,data)
            for key,value in extract.compute_features_for_epoch(x,256).items()}
features.update(extract.compute_spatial_features(data,cfg.EEG_CHANNELS,256))
assert len(features)==611, len(features)
assert np.isfinite(list(features.values())).all()
selected = pd.read_csv(next((ROOT/'reference_results/Validation').glob('final_selected_features_hboot*.csv'))).feature
assert len(selected)==26 and set(selected)<=features.keys()

import mne
raw = mne.io.RawArray(rng.normal(size=(14,256*18))*1e-6,
    mne.create_info(cfg.EEG_CHANNELS,256,'eeg'),verbose=False)
epochs = pre.make_fixed_epochs_from_raw(raw,6,preload=True)
assert epochs.get_data().shape==(3,14,1536)
assert np.all(np.diff(epochs.events[:,0])==1536)
assert np.allclose(validation.benjamini_hochberg_qvalues([.01,.04,.03]),[.03,.04,.04])

frame = pd.DataFrame([dict(subject=s,condition=cond,f=rng.normal()+int(cond=='attention'))
    for s in range(6) for cond in ('attention','relaxed') for _ in range(10)])
old_boot = validation.hierarchical_bootstrap_pvalue
try:
    validation.hierarchical_bootstrap_pvalue = lambda *a,**k:(_ for _ in ()).throw(AssertionError('Wilcoxon dispatched to bootstrap'))
    _,p,_ = validation.compute_pvalue_wrapper(frame,frame,'f',method='wilcoxon',n_boot=5,rng=rng)
    assert np.isfinite(p)
finally:
    validation.hierarchical_bootstrap_pvalue = old_boot

train = frame[frame.subject!=0]
test = frame[frame.subject==0]
scaler = loso.SubjectRobustScaler().fit(train)
before = scaler._global_mean.copy()
changed = test.copy()
changed['f'] += 1e6
scaler.transform(changed)
assert scaler._global_mean.equals(before)
assert np.isfinite(scaler.transform(test).f).all()
print('PASS: 611 features; 26 original biomarkers; exact non-overlapping 6s epochs; BH-FDR; independent Wilcoxon dispatch; training-only scaling.')
