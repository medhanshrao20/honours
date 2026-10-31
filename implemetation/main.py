"""Single entry point for the supplied EEG research code (Python 3.11)."""
import argparse
import hashlib
import importlib
import json
import os
from pathlib import Path
import shutil
import sys
from datetime import datetime, timezone

HERE = Path(__file__).resolve().parent
STAGES = ['preprocess','features','prune','rank','validate','cross-modal','loso','control']


def save_json(path, value):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,indent=2,default=str),encoding='utf-8')


def prepare_features(cfg, run_dir, smoke):
    import pandas as pd
    source = HERE/'Features/features_enhanced.parquet'
    dest = Path(cfg.FEATURES_DIR)/'features_enhanced.parquet'
    if dest.exists():
        return
    df = pd.read_parquet(source)
    df = df[~df.subject.isin(cfg.EXCLUDE_SUBJECTS_ANALYSIS)].copy()
    if smoke:
        df = df.groupby(['subject','task','condition'],group_keys=False).head(12)
        reference = HERE/'reference_results/Validation'
        selected = pd.read_csv(sorted(reference.glob('final_selected_features_hboot*.csv'))[-1]).feature.tolist()
        df = df[['file','task','subject','condition','epoch_idx']+selected]
    dest.parent.mkdir(parents=True,exist_ok=True)
    df.to_parquet(dest,index=False)
    df.groupby(['subject','task','condition']).size().rename('epochs').to_csv(run_dir/'coverage.csv')
    save_json(run_dir/'input_manifest.json',dict(source=str(source),
        source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),rows=len(df),columns=len(df.columns),
        subjects=sorted(map(int,df.subject.unique())),smoke_validation_only=smoke,
        epoch_provenance='Inherited from supplied feature table; not independently verifiable'))


def seed_reference(cfg):
    for folder in ('Redundancy','Wasserstein','Validation'):
        target = Path(cfg.BIOMARKER_DIR)/folder
        target.mkdir(parents=True,exist_ok=True)
        for source in (HERE/'reference_results'/folder).glob('*'):
            if source.is_file() and not (target/source.name).exists():
                shutil.copy2(source,target/source.name)


def preprocess(cfg, source):
    mod = importlib.import_module('01_preprocessing')
    mod.DEFAULT_N_JOBS = cfg.DEFAULT_N_JOBS
    if not mod.IC_LABEL_AVAILABLE or not mod.AUTOREJECT_AVAILABLE:
        raise RuntimeError('ICLabel and AutoReject are required; install requirements.txt.')
    if source=='csv':
        mod.main()
    else:
        raw_source = HERE.parent/'Honors/EEGData'
        raw_files = list(raw_source.glob('*_raw.fif'))
        if not raw_files:
            raise FileNotFoundError(f'No raw FIF recordings in {raw_source}')
        Path(cfg.STAGE0_OUT).mkdir(parents=True,exist_ok=True)
        for path in raw_files:
            target = Path(cfg.STAGE0_OUT)/path.name
            if not target.exists():
                shutil.copy2(path,target)
        mod.EEGStage1(cfg.STAGE0_OUT,cfg.STAGE1_OUT,cfg.LOGS_FOLDER,cfg.QC_FOLDER,
                      n_jobs=cfg.DEFAULT_N_JOBS,max_bad_channels=cfg.MAX_BAD_CHANNELS).process_all()
        mod.EEGStage2(cfg.STAGE1_OUT,cfg.STAGE2_OUT,cfg.LOGS_FOLDER,cfg.QC_FOLDER,
                      n_jobs=cfg.DEFAULT_N_JOBS).process_all()
        mod.EEGStage3(cfg.STAGE1_OUT,cfg.STAGE2_OUT,cfg.STAGE3_OUT,cfg.LOGS_FOLDER,cfg.QC_FOLDER,
                      n_jobs=cfg.DEFAULT_N_JOBS,min_epochs_for_ar=cfg.MIN_EPOCHS_FOR_AR).process_all()
        mod.run_session_qc()
    if not list(Path(cfg.FINAL_DATA_FOLDER).glob('*_final_epo.fif')):
        raise RuntimeError('No final epochs produced. Inspect Preprocessed/Logs.')


def validate(cfg):
    import pandas as pd
    mod = importlib.import_module('05_biomarker_stat_validation')
    df = mod.load_data()
    if not Path(mod.WASSERSTEIN_CSV).exists():
        raise FileNotFoundError('Run --stage rank before validation.')
    candidates = mod.load_candidates_from_csv(mod.WASSERSTEIN_CSV,df)
    lists = []
    for method in ('hboot','wilcoxon'):
        summary,_ = mod.biomarker_discovery(df,candidates,output_dir=mod.OUTPUT_DIR,
            stats_method=method,n_boot=mod.N_BOOT,loso_resamples=cfg.LOSO_RESAMPLES)
        lists.append(set(summary.loc[summary.final_selected,'feature']))
    final = sorted(set.intersection(*lists))
    pd.DataFrame({'feature':final}).to_csv(Path(mod.OUTPUT_DIR)/'final_robust_intersection.csv',index=False)
    print(f'Bootstrap/Wilcoxon intersection: {len(final)} features',flush=True)


def loso(cfg,args):
    mod = importlib.import_module('07_standard_loso_classification')
    if args.profile=='paper' or args.smoke:
        mod.ALGORITHMS = [(n,c) for n,c in mod.ALGORITHMS if n in ('SVM','LDA')]
    if args.smoke:
        mod.WASS_N_BOOTSTRAP = cfg.WASSERSTEIN_BOOTSTRAPS
        mod.bootstrap_wasserstein.__defaults__ = (cfg.WASSERSTEIN_BOOTSTRAPS,cfg.RANDOM_SEED)
        mod.plot_distributions = lambda *a,**kw:None
    frame = mod.load_features(str(Path(cfg.FEATURES_DIR)/'features_enhanced.parquet'))
    frame = frame[~frame.subject.isin(cfg.EXCLUDE_SUBJECTS_ANALYSIS)].reset_index(drop=True)
    if args.profile=='paper':
        frame = frame[[c for c in frame if not c.startswith('RIEM__')]]
    mod.train_loso(frame,mod.DEFAULT_OUT_DIR,zip_folds=False,keep_fold_folders=True)


def audit(cfg,run_dir):
    import pandas as pd
    df = pd.read_parquet(Path(cfg.FEATURES_DIR)/'features_enhanced.parquet')
    metadata = ['file','task','subject','condition','epoch_idx']
    physiological = [c for c in df if c not in metadata and not c.startswith('RIEM__')]
    checks = dict(rows=len(df),subjects=df.subject.nunique(),physiological_features=len(physiological),
        riemannian_features=sum(c.startswith('RIEM__') for c in df),
        missing_values=int(df.isna().sum().sum()),duplicated_epochs=int(df.duplicated(metadata).sum()),
        exclusions=cfg.EXCLUDE_SUBJECTS_ANALYSIS,profile=cfg.RUN_PROFILE,epoch_duration_setting=cfg.EPOCH_DUR)
    save_json(run_dir/'audit.json',checks)
    print(json.dumps(checks,indent=2),flush=True)
    if checks['missing_values'] or checks['duplicated_epochs']:
        raise ValueError('Invalid feature table; inspect audit.json')
    if not cfg.SMOKE_MODE and checks['physiological_features']!=611:
        raise ValueError('Expected the supplied 611 physiological features.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage',choices=['all','audit','classify']+STAGES,default='all')
    parser.add_argument('--profile',choices=['paper','senior'],default='paper')
    parser.add_argument('--from-raw',action='store_true',help='Preprocess and extract before analysis')
    parser.add_argument('--raw-source',choices=['fif','csv'],default='fif')
    parser.add_argument('--use-reference-results',action='store_true',help='Explicitly seed inherited discovery results')
    parser.add_argument('--smoke',action='store_true',help='Reduced data and bootstraps; validation only')
    parser.add_argument('--jobs',type=int,default=1)
    parser.add_argument('--output',type=Path)
    args = parser.parse_args()
    if args.jobs<1:
        parser.error('--jobs must be >=1')
    run_dir = (args.output or HERE/'runs'/args.profile/('smoke' if args.smoke else 'full')).resolve()
    run_dir.mkdir(parents=True,exist_ok=True)
    os.environ.update(EEG_RUN_DIR=str(run_dir),EEG_PROFILE=args.profile,EEG_SMOKE='1' if args.smoke else '0',
        EEG_JOBS=str(args.jobs),MPLBACKEND='Agg',PYTHONUTF8='1',OMP_NUM_THREADS='1',
        MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
    os.chdir(run_dir)
    cfg = importlib.import_module('00_config')
    print(f'Run directory: {run_dir}',flush=True)
    settings = {k:v for k,v in vars(cfg).items() if k.isupper()}
    manifest_path = run_dir/'run_manifest.json'
    previous = json.loads(manifest_path.read_text(encoding='utf-8')) if manifest_path.exists() else {}
    if previous and previous['settings']!=json.loads(json.dumps(settings,default=str)):
        raise ValueError('Output directory has different settings. Choose a new --output directory.')
    manifest = dict(started=datetime.now(timezone.utc).isoformat(),settings=settings,
        source_hashes={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in HERE.glob('*.py')},
        inherited_discovery_results=args.use_reference_results or previous.get('inherited_discovery_results',False),
        completed_stages=previous.get('completed_stages',[]),exact_published_scores_guaranteed=False)
    save_json(manifest_path,manifest)
    if not args.from_raw and args.stage not in ('preprocess','features'):
        prepare_features(cfg,run_dir,args.smoke)
        audit(cfg,run_dir)
    if args.stage=='audit':
        return
    if args.use_reference_results:
        seed_reference(cfg)
    stages = (STAGES if args.from_raw else STAGES[2:]) if args.stage=='all' else (
        ['cross-modal','loso','control'] if args.stage=='classify' else [args.stage])
    for stage in stages:
        print(f'\n=== {stage.upper()} ===',flush=True)
        if stage=='preprocess':
            preprocess(cfg,args.raw_source)
        elif stage=='features':
            importlib.import_module('02_feature_extraction').main()
            feature_file = Path(cfg.FEATURES_DIR)/'features_enhanced.parquet'
            if not feature_file.exists():
                raise RuntimeError('No features extracted; run preprocessing first.')
            import pandas as pd
            df = pd.read_parquet(feature_file)
            df[~df.subject.isin(cfg.EXCLUDE_SUBJECTS_ANALYSIS)].to_parquet(feature_file,index=False)
            audit(cfg,run_dir)
        elif stage=='prune':
            importlib.import_module('03_redundancy_pruning_exploratory').main()
        elif stage=='rank':
            mod = importlib.import_module('04_wasserstein_selection_exploratory')
            if args.smoke:
                mod.plot_distributions = lambda *a,**kw:None
            old_args = sys.argv
            try:
                sys.argv = [str(HERE/'04_wasserstein_selection_exploratory.py')]
                mod.main()
            finally:
                sys.argv = old_args
        elif stage=='validate':
            validate(cfg)
        elif stage=='cross-modal':
            importlib.import_module('06_nested_loso_lomo_classification')
        elif stage=='loso':
            loso(cfg,args)
        elif stage=='control':
            importlib.import_module('08_task_independence_check')
        manifest['completed_stages'] = list(dict.fromkeys(manifest['completed_stages']+[stage]))
        save_json(manifest_path,manifest)
    manifest['finished'] = datetime.now(timezone.utc).isoformat()
    save_json(manifest_path,manifest)
    print(f'\nCompleted. Results: {run_dir}',flush=True)


if __name__=='__main__':
    main()
