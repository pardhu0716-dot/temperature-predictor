"""Leakage-safe whole-floorplan splitting and training-only scaling."""
from contextlib import contextmanager
import hashlib
import json
from pathlib import Path
import shutil
import tempfile
import numpy as np
from sklearn.preprocessing import StandardScaler
from .data import CORE_IDS,digest,read_json
from .features import build_features,VERSION

PROJECT=Path(__file__).resolve().parents[2]


def write_json(path,value):
    Path(path).write_text(json.dumps(value,indent=2,sort_keys=True,allow_nan=False)+'\n')


def safe_output(path):
    path=Path(path).expanduser().resolve()
    if path.is_relative_to(PROJECT) and not path.is_relative_to(PROJECT/'stage_b/ml_runs'):
        raise ValueError('Repository-local outputs must be under stage_b/ml_runs, never original data/code')
    if PROJECT.is_relative_to(path):raise ValueError('Output cannot contain the project')
    return path


@contextmanager
def new_output(path):
    path=safe_output(path)
    if path.exists():raise FileExistsError(f'Refusing to overwrite {path}')
    path.parent.mkdir(parents=True,exist_ok=True)
    temp=Path(tempfile.mkdtemp(prefix='.ml-preparing-',dir=path.parent))
    try:
        yield temp
        if path.exists():raise FileExistsError(path)
        temp.rename(path)
    finally:
        if temp.exists():shutil.rmtree(temp)


def split_floorplans(ids,seed=42,counts=(53,12,12)):
    unique=np.asarray(sorted(set(ids)))
    if len(unique)!=sum(counts) or any(n<1 for n in counts):raise ValueError(f'Split counts must sum to {len(unique)} floorplans and be nonzero')
    order=np.random.default_rng(seed).permutation(unique)
    cuts=np.cumsum((0,*counts));parts={}
    for i,name in enumerate(('train','validation','test')):parts[name]=sorted(order[cuts[i]:cuts[i+1]].tolist())
    assert not set(parts['train'])&set(parts['validation']) and not set(parts['train'])&set(parts['test']) and not set(parts['validation'])&set(parts['test'])
    return parts


def prepare(dataset,output,seed=42,counts=(53,12,12),geometry_manifest=None):
    X,definitions=build_features(dataset)
    splits=split_floorplans(dataset.floorplan_ids,seed,counts)
    masks={name:np.isin(dataset.floorplan_ids,ids) for name,ids in splits.items()}
    xscale=StandardScaler().fit(X[masks['train']]);yscale=StandardScaler().fit(dataset.temperatures_c[masks['train']])
    scaling=dict(x_mean=xscale.mean_.tolist(),x_scale=xscale.scale_.tolist(),y_mean=yscale.mean_.tolist(),y_scale=yscale.scale_.tolist(),fitted_on='train only',n_train_samples=int(masks['train'].sum()))
    schema_hash=hashlib.sha256(json.dumps(definitions,sort_keys=True).encode()).hexdigest()
    contract=dataset.schema.get('simulation_contract')
    if contract is not None:
        from .external import check_compatibility
        contract=dict(contract)
        normalized=dict(geometry_unit='mm',temperature_unit='C',power_unit='W',ambient_c=dataset.schema['ambient_c'],core_order=list(CORE_IDS),label_source='HotSpot',feature_sha256=schema_hash)
        for key,value in normalized.items():
            if key in contract and contract[key]!=value:raise ValueError(f'Simulation contract conflicts with normalized feature/target definition: {key}')
            contract[key]=value
        check_compatibility(contract,contract)
    with new_output(output) as out:
        for name,mask in masks.items():
            np.savez_compressed(out/f'{name}.npz',X=X[mask],y=dataset.temperatures_c[mask],floorplan_ids=dataset.floorplan_ids[mask],sample_ids=dataset.sample_ids[mask])
        write_json(out/'scaling.json',scaling);write_json(out/'features.json',dict(version=VERSION,definitions=definitions,sha256=schema_hash))
        write_json(out/'schema.json',dataset.schema)
        write_json(out/'simulation_contract.json',contract)
        manifest=dict(version=1,seed=seed,split_floorplans=splits,split_row_counts={k:int(v.sum()) for k,v in masks.items()},
                      source_csv_sha256=dataset.source_sha256,core_order=list(CORE_IDS),feature_count=X.shape[1],feature_sha256=schema_hash,
                      geometry_manifest_sha256=digest(geometry_manifest) if geometry_manifest else None,
                      label_provenance=contract,ambient_c=dataset.schema['ambient_c'],
                      files={p.name:digest(p) for p in out.iterdir()},test_policy='No test data loaded by train; explicit evaluate-test command only')
        write_json(out/'manifest.json',manifest)
    return manifest


def load_prepared(path,part):
    path=Path(path).resolve();manifest=read_json(path/'manifest.json')
    if part not in ('train','validation','test'):raise ValueError('Unknown partition')
    for name in [f'{part}.npz','scaling.json','features.json','schema.json']:
        if digest(path/name)!=manifest['files'][name]:raise ValueError(f'Prepared artifact changed: {name}')
    with np.load(path/f'{part}.npz',allow_pickle=False) as data:result={k:data[k].copy() for k in data.files}
    return manifest,read_json(path/'scaling.json'),result
