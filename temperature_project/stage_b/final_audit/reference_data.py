"""Strict, named alignment of the original tracked Stage A workloads and labels."""
import json
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter_ns
import numpy as np
from predict import discover_floorplans,parse_lcf_active_flp,parse_flp_cores,locate_and_load_thermal_matrix
from stage_a_pipeline import parse_flp
from stage_b.ml.data import CORE_IDS,validate_geometry
PROJECT=Path(__file__).resolve().parents[2]


def parse_power(path):
    rows=[line.split() for line in Path(path).read_text().splitlines() if line.strip() and not line.lstrip().startswith('#')]
    if len(rows)!=2 or len(rows[0])!=len(rows[1]) or len(set(rows[0]))!=len(rows[0]):raise ValueError('Expected one uniquely named power workload')
    mapping={k:float(v) for k,v in zip(*rows)}
    if not set(CORE_IDS).issubset(mapping) or not np.isfinite(list(mapping.values())).all():raise ValueError('Missing/nonfinite core powers')
    if any(v!=0 for k,v in mapping.items() if k.startswith('FILL')):raise ValueError('Passive FILL power must be zero')
    if any(k not in CORE_IDS and not k.startswith('FILL') for k in mapping):raise ValueError('Unsupported power block')
    return np.array([mapping[k] for k in CORE_IDS])


def parse_temperatures(path):
    values={}
    for line in Path(path).read_text().splitlines():
        row=line.split()
        if not row or row[0].startswith('#'):continue
        if len(row)!=2:raise ValueError('Malformed HotSpot steady output')
        name=row[0]
        if name.startswith('layer_0_'):name=name[len('layer_0_'):]
        if name.startswith('Core_') and name not in CORE_IDS:raise ValueError('Unexpected HotSpot core identifier')
        if name in CORE_IDS:
            if name in values:raise ValueError('Duplicate HotSpot core label')
            values[name]=float(row[1])-273.15
    if set(values)!=set(CORE_IDS) or not np.isfinite(list(values.values())).all():raise ValueError('Missing/nonfinite HotSpot labels')
    return np.array([values[k] for k in CORE_IDS])


@dataclass
class Workload:
    floorplan:str
    sample_id:str
    geometry:np.ndarray
    die:np.ndarray
    powers:np.ndarray
    hotspot:np.ndarray
    matrix:np.ndarray
    ptrace:Path
    steady:Path


def original_workloads(project=PROJECT):
    catalog=sorted(discover_floorplans(str(Path(project)/'dataset/Sample_dataset/2D_floorplans')),key=lambda x:x['full_name'])
    if len(catalog)!=23:raise ValueError('Expected 23 original floorplans')
    for item in catalog:
        root=Path(item['path']);flp=parse_lcf_active_flp(str(root));cores,die,_=parse_flp_cores(flp)
        g=np.array([[cores[n][k] for k in ('width_mm','height_mm','left_x_mm','bottom_y_mm')] for n in CORE_IDS]);die=np.array(die)
        validate_geometry(g,die)
        blocks=parse_flp(flp);rects=[np.array([b['width'],b['height'],b['left_x'],b['bottom_y']])*1000 for b in blocks]
        for i,r in enumerate(rects):
            if np.any(r[:2]<=0) or np.any(r[2:]<0) or np.any(r[:2]+r[2:]>die+1e-8):raise ValueError('Original FLP block invalid/outside die')
            for s in rects[:i]:
                if np.all(np.minimum(r[2:]+r[:2],s[2:]+s[:2])-np.maximum(r[2:],s[2:])>1e-8):raise ValueError('Original FLP overlap')
        if not np.isclose(sum(r[0]*r[1] for r in rects),np.prod(die),rtol=0,atol=1e-6):raise ValueError('Original FLP FILL coverage incomplete')
        _,order,R,_,_=locate_and_load_thermal_matrix(str(root),item['floorplan'])
        if set(order)!=set(CORE_IDS) or len(order)!=8:raise ValueError('Invalid Stage A matrix core names')
        indices=[order.index(n) for n in CORE_IDS];matrix=np.asarray(R)[np.ix_(indices,indices)]
        powers_dir=root/'validation_samples/ptrace';steady_dir=root/'validation_samples/steady'
        expected={f'val_sample_{i:03d}' for i in range(1,21)}
        if {p.stem for p in powers_dir.glob('val_sample_*.ptrace')}!=expected or {p.stem for p in steady_dir.glob('val_sample_*.steady')}!=expected:raise ValueError('Expected identical 20 power/reference sample IDs')
        for sid in sorted(expected):
            pt=powers_dir/(sid+'.ptrace');st=steady_dir/(sid+'.steady');p=parse_power(pt);hot=parse_temperatures(st)
            big=g[:,0]==4
            if np.any(p<np.where(big,10,5)) or np.any(p>np.where(big,20,10)):raise ValueError('Original powers outside supported bounds')
            yield Workload(item['full_name'],sid,g,die,p,hot,matrix,pt,st)


def score(true,pred):
    true=np.asarray(true);pred=np.asarray(pred)
    if true.shape!=pred.shape or true.ndim!=2 or true.shape[1]!=8 or not len(true) or not np.isfinite(true).all() or not np.isfinite(pred).all():raise ValueError('Expected matching finite workload-by-eight arrays')
    error=pred-true
    return dict(mae_c=float(np.mean(abs(error))),rmse_c=float(np.sqrt(np.mean(error**2))),max_absolute_error_c=float(np.max(abs(error))),hottest_core_accuracy_pct=float(np.mean(np.argmax(true,axis=1)==np.argmax(pred,axis=1))*100))


def legacy_baseline_tables(project=PROJECT):
    """In-memory fallback for old diagnostic scripts; never writes historical CSVs."""
    from stage_b.web.service import Predictor,UNITS
    from app import app as stage_a_app
    client=stage_a_app.test_client();pipeline_times=[]
    predictor=Predictor();wls=list(original_workloads(project));gt=[];a=[];v1=[];times=[];rows=[]
    for w in wls:
        w.matrix@w.powers
        start=perf_counter_ns();a.append(45.+w.matrix@w.powers);times.append((perf_counter_ns()-start)/1e6)
        start=perf_counter_ns();response=client.post('/api/predict',json=dict(floorplan=w.floorplan,powers_by_core=dict(zip(CORE_IDS,w.powers.tolist()))));pipeline_times.append((perf_counter_ns()-start)/1e6)
        if response.status_code!=200:raise RuntimeError('Stage A baseline request failed')
        payload=dict(mode='custom',model_version='v1',units=UNITS,die=dict(zip(('width_mm','height_mm'),w.die.tolist())),cores=[dict(id=n,**dict(zip(('width_mm','height_mm','x_mm','y_mm'),g.tolist())),power_w=float(p)) for n,g,p in zip(CORE_IDS,w.geometry,w.powers)])
        r=predictor.predict(payload);v1.append([r['temperatures_c'][n] for n in CORE_IDS]);gt.append(w.hotspot)
        for i,n in enumerate(CORE_IDS):rows.append(dict(floorplan=w.floorplan,sample_id=w.sample_id,core_id=n,power_w=float(w.powers[i]),hotspot_temp_c=float(w.hotspot[i]),stage_b_temp_c=v1[-1][i],core_type='Big' if w.geometry[i,0]==4 else 'Little'))
    gt=np.array(gt);a=np.array(a);v1=np.array(v1);m=score(gt,a)
    ref=dict(m,max_abs_err_c=m['max_absolute_error_c'],hottest_core_acc_pct=m['hottest_core_accuracy_pct'],median_model_inference_ms=float(np.median(times)),median_pipeline_total_ms=float(np.median(pipeline_times)),p95_pipeline_total_ms=float(np.percentile(pipeline_times,95)),timing_boundary="Local Flask test-client request (no network); model timing is matrix calculation only")
    layouts=predictor.layouts;manifest=json.loads((Path(project)/'stage_b/models/final_lasso/manifest.json').read_text());train_die=np.array([layouts[f][1] for f in manifest['split_floorplans']['train']])
    fps={}
    for name in sorted({w.floorplan for w in wls}):
        mask=np.array([w.floorplan==name for w in wls]);w=next(w for w in wls if w.floorplan==name);m1=score(gt[mask],v1[mask]);ma=score(gt[mask],a[mask]);inside=bool(np.all(w.die>=train_die.min(axis=0)) and np.all(w.die<=train_die.max(axis=0)))
        fps[name]=dict(floorplan=name,die_width_mm=w.die[0],die_height_mm=w.die[1],inside_stage_b_die_box=str(inside),stage_a_mae_c=ma['mae_c'],stage_b_mae_c=m1['mae_c'],stage_b_hottest_acc_pct=m1['hottest_core_accuracy_pct'])
    per_core={n:dict(core_id=n,stage_a_mae_c=float(np.mean(abs(a[:,i]-gt[:,i])))) for i,n in enumerate(CORE_IDS)}
    return {'Stage A':ref},fps,per_core,rows
