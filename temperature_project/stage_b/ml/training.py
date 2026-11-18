"""Validation-only model selection. Test partition loaded only by evaluate_test."""
import hashlib
import importlib.metadata
import json
import platform
from pathlib import Path
import time
import joblib
import numpy as np
from .data import digest,read_json
from .evaluation import metrics,latency
from .models import build_model,ModelBundle
from .experiments import validate_experiments
from .preprocessing import load_prepared,new_output,write_json


def versions():
    out=dict(python=platform.python_version(),platform=dict(system=platform.system(),release=platform.release(),machine=platform.machine()))
    for name in ('numpy','scikit-learn','joblib','xgboost','xgboost-cpu'):
        try:out[name]=importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:out[name]=None
    return out


def train(prepared,output,experiments,seed=42,repeats=50):
    validate_experiments(experiments)
    # Preflight all requested estimators (including optional imports) before writing output.
    estimators=[build_model(e['kind'],e.get('parameters'),seed) for e in experiments]
    manifest,scaling,training=load_prepared(prepared,'train')
    other,_,validation=load_prepared(prepared,'validation')
    if manifest!=other:raise ValueError('Prepared manifest changed during loading')
    tx=(training['X']-scaling['x_mean'])/scaling['x_scale']
    ty=(training['y']-scaling['y_mean'])/scaling['y_scale']
    reports=[]
    with new_output(output) as out:
        for experiment,estimator in zip(experiments,estimators):
            start=time.perf_counter();estimator.fit(tx,ty);seconds=time.perf_counter()-start
            bundle=ModelBundle(estimator,scaling,manifest['feature_sha256'],manifest['core_order'],manifest['label_provenance'],manifest['source_csv_sha256'])
            folder=out/experiment['name'];folder.mkdir()
            model=folder/'model.joblib';joblib.dump(bundle,model,compress=3)
            report=dict(experiment=experiment,seed=seed,train_seconds=seconds,validation=metrics(validation['y'],bundle.predict(validation['X'])),
                        latency=latency(bundle,validation['X'],repeats),serialized_model_bytes=model.stat().st_size,
                        model_sha256=digest(model),versions=versions(),prepared_manifest_sha256=digest(Path(prepared)/'manifest.json'),
                        source_csv_sha256=manifest['source_csv_sha256'],split_floorplans=manifest['split_floorplans'],test_evaluated=False)
            write_json(folder/'validation_report.json',report);reports.append(report)
        best=min(reports,key=lambda r:(r['validation']['rmse_c'],r['experiment']['name']))
        name=best['experiment']['name']
        write_json(out/'selection.json',dict(selected=name,criterion='minimum validation RMSE in Celsius; name tie-break',
                   model_sha256=best['model_sha256'],prepared_manifest_sha256=digest(Path(prepared)/'manifest.json'),seed=seed,
                   feature_sha256=manifest['feature_sha256'],validation_rmse_c=best['validation']['rmse_c'],test_evaluated=False))
        write_json(out/'validation_comparison.json',reports)
    return reports


def evaluate_test(prepared,run):
    """Explicit final command: selected model only, exclusive new report, no tuning."""
    run=Path(run).resolve();selection=read_json(run/'selection.json')
    if selection['prepared_manifest_sha256']!=digest(Path(prepared)/'manifest.json'):raise ValueError('Wrong prepared dataset')
    name=selection['selected']
    if Path(name).name!=name or name in ('.','..'):raise ValueError('Unsafe selected model path')
    model=run/name/'model.joblib'
    if digest(model)!=selection['model_sha256']:raise ValueError('Selected model changed after validation selection')
    target=run/'final_test_report.json'
    if target.exists():raise FileExistsError('Final test report already exists; never overwrite or repeatedly tune on test results')
    # Joblib is executable Python serialization: only load artifacts you trained/trust.
    bundle=joblib.load(model)
    manifest,_,test=load_prepared(prepared,'test')
    if bundle.feature_sha256!=manifest['feature_sha256']:raise ValueError('Feature definition mismatch')
    report=dict(selected=name,test=metrics(test['y'],bundle.predict(test['X'])),model_sha256=digest(model),
                prepared_manifest_sha256=selection['prepared_manifest_sha256'],test_floorplans=manifest['split_floorplans']['test'],
                policy='One final holdout evaluation; no further tuning on these results')
    with target.open('x') as stream:json.dump(report,stream,indent=2,allow_nan=False)
    return report
