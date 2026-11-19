"""Fresh aligned 23-floorplan comparison of frozen predictors. No fitting/export."""
import csv,hashlib,json,platform,sys
from pathlib import Path
from time import perf_counter_ns
import numpy as np
from stage_b.ml.data import CORE_IDS
from stage_b.web.service import Predictor,UNITS
from .reference_data import PROJECT,original_workloads,score


def markdown_table(rows):
    text='| Floorplan | Stage A MAE (°C) | Stage B v1 MAE (°C) | Stage B v2 MAE (°C) | Stage B v2 RMSE (°C) | Stage B v2 Max Error (°C) |\n|---|---:|---:|---:|---:|---:|\n'
    for r in rows:text+='| '+r['floorplan']+' | '+' | '.join(f'{r[k]:.6f}' for k in ['stage_a_mae_c','stage_b_v1_mae_c','stage_b_v2_mae_c','stage_b_v2_rmse_c','stage_b_v2_max_absolute_error_c'])+' |\n'
    return text


def run(output):
    output=Path(output);targets=[output/n for n in ['stage_a_vs_stage_b_23_floorplans.csv','stage_a_vs_stage_b_all_predictions.csv','comparison_summary.json','STAGE_A_VS_STAGE_B_23_FLOORPLANS.md']]
    if any(p.exists() for p in targets):raise FileExistsError('Refusing to replace any completed comparison output')
    service=Predictor();workloads=list(original_workloads());assert len(workloads)==460
    ground=[];preds={'stage_a':[],'stage_b_v1':[],'stage_b_v2':[]};timing={k:[] for k in preds};records=[];inputs={};feature_ms={'v1':[],'v2':[]}
    for w in workloads:
        payload=dict(mode='custom',units=UNITS,die=dict(zip(('width_mm','height_mm'),w.die.tolist())),cores=[dict(id=n,**dict(zip(('width_mm','height_mm','x_mm','y_mm'),g.tolist())),power_w=float(p)) for n,g,p in zip(CORE_IDS,w.geometry,w.powers)])
        # A uses the exact calibrated matrix and canonical row/column remapping.
        start=perf_counter_ns();a=45.+w.matrix@w.powers;timing['stage_a'].append((perf_counter_ns()-start)/1e6)
        one=service.predict({**payload,'model_version':'v1'});two=service.predict({**payload,'model_version':'v2'})
        v1=np.array([one['temperatures_c'][n] for n in CORE_IDS]);v2=np.array([two['temperatures_c'][n] for n in CORE_IDS])
        ground.append(w.hotspot)
        for key,value,r in [('stage_a',a,None),('stage_b_v1',v1,one),('stage_b_v2',v2,two)]:
            preds[key].append(value)
            if r is not None:timing[key].append(r['latency_ms'])
        feature_ms['v1'].append(one['feature_engineering_ms']);feature_ms['v2'].append(two['feature_engineering_ms'])
        for path in (w.ptrace,w.steady):inputs[str(path.relative_to(PROJECT))]=hashlib.sha256(path.read_bytes()).hexdigest()
        for i,n in enumerate(CORE_IDS):records.append(dict(floorplan=w.floorplan,sample_id=w.sample_id,core_id=n,core_type='Big' if w.geometry[i,0]==4 else 'Little',power_w=float(w.powers[i]),hotspot_temp_c=float(w.hotspot[i]),stage_a_temp_c=float(a[i]),stage_b_v1_temp_c=float(v1[i]),stage_b_v2_temp_c=float(v2[i]),stage_a_error_c=float(a[i]-w.hotspot[i]),stage_b_v1_error_c=float(v1[i]-w.hotspot[i]),stage_b_v2_error_c=float(v2[i]-w.hotspot[i])))
    gt=np.array(ground);preds={k:np.array(v) for k,v in preds.items()};summary={k:score(gt,p) for k,p in preds.items()};per_floor=[]
    for name in sorted({w.floorplan for w in workloads}):
        mask=np.array([w.floorplan==name for w in workloads]);row=dict(floorplan=name,workloads=int(mask.sum()))
        for key,p in preds.items():row.update({key+'_'+k:v for k,v in score(gt[mask],p[mask]).items()})
        per_floor.append(row)
    best=min(per_floor,key=lambda x:x['stage_b_v2_mae_c']);worst=max(per_floor,key=lambda x:x['stage_b_v2_mae_c'])
    improved=sum(r['stage_b_v2_mae_c']<r['stage_b_v1_mae_c'] for r in per_floor)
    # Same warmup + 50 single-sample calls as prior experiments; scaling is included.
    w=workloads[0];X1=service._features_v1(w.geometry,w.die,w.powers);X2=service.v2_model.build_features(w.geometry,w.powers,w.die)
    funcs={'stage_a':lambda:45.+w.matrix@w.powers,'stage_b_v1':lambda:service.bundle.predict(X1),'stage_b_v2':lambda:service.v2_model.predict_from_features(X2)}
    latencies={}
    for key,fn in funcs.items():
        fn();values=[]
        for _ in range(50):
            start=perf_counter_ns();fn();values.append((perf_counter_ns()-start)/1e6)
        latencies[key]=dict(median_ms=float(np.median(values)),p95_ms=float(np.percentile(values,95)),repeats=50,boundary='Prebuilt inputs through model and required scaling; excludes feature construction, validation and HTTP')
    tie_aware={k:float(np.mean(np.isclose(gt[np.arange(460),np.argmax(p,axis=1)],np.max(gt,axis=1),rtol=0,atol=1e-9))*100) for k,p in preds.items()}
    original=json.loads((PROJECT/'stage_b/models/v2_ridge/metadata.json').read_text())['audited_metrics']['external_23fp_460wl']
    outliers=sorted(records,key=lambda r:abs(r['stage_b_v2_error_c']),reverse=True)[:10]
    result=dict(workloads=460,core_observations=3680,floorplans=23,metrics=summary,improved_floorplans_v2_over_v1=improved,
                v2_mae_reduction_percent=(1-summary['stage_b_v2']['mae_c']/summary['stage_b_v1']['mae_c'])*100,
                best_v2_floorplan=best['floorplan'],worst_v2_floorplan=worst['floorplan'],latencies=latencies,
                mean_feature_ms={k:float(np.mean(v)) for k,v in feature_ms.items()},hottest_core_policy='Primary: first argmax in canonical core-ID order, matching historical convention',
                tie_aware_hottest_accuracy_pct=tie_aware,ground_truth_tied_hottest_workloads=int(np.sum(np.sum(np.isclose(gt,gt.max(axis=1)[:,None],rtol=0,atol=1e-9),axis=1)>1)),
                v2_nearest_policy='Mean power among nearest sources within absolute 1e-12 mm; scalers/coefficients unchanged',
                historical_v2_external_metrics=original,v2_delta_vs_historical={k:summary['stage_b_v2'][new]-original[k] for k,new in [('mae_c','mae_c'),('rmse_c','rmse_c'),('max_err_c','max_absolute_error_c')]},
                largest_v2_errors=outliers,source_input_sha256=inputs,model_sha256={'v1':service.models_info['v1']['model_sha256'],'v2':service.v2_model.artifact_sha256},python=platform.python_version(),training_performed=False)
    output.mkdir(parents=True,exist_ok=True)
    def save(path,rows):
        with path.open('x',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    save(targets[0],per_floor);save(targets[1],records);targets[2].write_text(json.dumps(result,indent=2,allow_nan=False))
    overall='| Predictor | MAE °C | RMSE °C | Max °C | Hottest-core accuracy |\n|---|---:|---:|---:|---:|\n'+''.join(f"| {k} | {m['mae_c']:.6f} | {m['rmse_c']:.6f} | {m['max_absolute_error_c']:.6f} | {m['hottest_core_accuracy_pct']:.2f}% |\n" for k,m in summary.items())
    text=f'''# Stage A vs Stage B: all 23 original floorplans

Fresh evaluation of fixed artifacts after bug fixes. These 23 layouts have been
used in earlier v1/v2 evaluations and are **not an untouched blind test set**.
No fitting, calibration, tuning, model replacement or HotSpot rerun occurred.

{markdown_table(per_floor)}
## Overall (460 workloads, 3,680 per-core observations per predictor)

{overall}

V2 improves MAE over v1 by {result['v2_mae_reduction_percent']:.2f}% and wins on
{improved}/23 floorplans. Best v2 MAE: {best['floorplan']} ({best['stage_b_v2_mae_c']:.6f}°C).
Worst v2 MAE: {worst['floorplan']} ({worst['stage_b_v2_mae_c']:.6f}°C).

Stage A uses a separately calibrated 8×8 resistance matrix for each of these
layouts; this is its in-distribution advantage. Stage B shares one geometry-aware
model across layouts and was trained on the distinct 77-layout dataset. Better
Stage A accuracy here does not imply that its matrices generalize to new layouts.

## Alignment and methods

Exactly val_sample_001…020 per floorplan were paired by sample ID, with core names
canonicalized independently for geometry, ptrace columns, steady labels and matrix
rows/columns. Kelvin labels were converted with C=K−273.15; Stage A applies 45+R×P
without retraining. Original core/FILL bounds, overlap/area coverage, zero FILL
powers, dimensions, finite values and matching trace/reference sets were checked.
All 920 trace/reference file hashes are saved in comparison_summary.json.

Metrics pool all 3,680 observations (not an average of rounded floorplan metrics).
Hottest-core accuracy uses canonical first-argmax as in historical reports; tie-aware
accuracy and the count of quantized reference ties are provided separately.

## Comparison with historical records

The v1 reference was previously MAE 1.491469°C / RMSE 1.917708°C / max 6.535131°C.
The historical v2 record was MAE {original['mae_c']:.6f}°C /
RMSE {original['rmse_c']:.6f}°C / max {original['max_err_c']:.6f}°C.
New v2 MAE differs by {result['v2_delta_vs_historical']['mae_c']:+.9f}°C and
RMSE by {result['v2_delta_vs_historical']['rmse_c']:+.9f}°C. Historical files were
not changed. The intentional inference correction averages tied nearest-neighbor
powers instead of choosing one by core index. Frozen coefficients and scalers are
unchanged; the original legacy feature rule is retained only for scaler-provenance
audits. This is a correctness fix, not a hyperparameter adjustment.

The ten largest observed v2 errors, each linked to layout/sample/core, are in
comparison_summary.json. Per-core CSV predictions preserve full floating-point
values. Extreme custom-layout extrapolation and v1's identity-dependent features
remain model limitations; no claim of universal geometric/rotation invariance is
made. Input-row reordering preserves named mapping in both versions, while physical
core relabeling is equivariant in corrected v2 and not guaranteed in frozen v1.

Latency uses one warmup and 50 single-workload calls including required scaling;
it excludes feature construction and HTTP. Model sizes are 11,197 B for v1 and
3,028 B for v2, not runtime RAM. These timing/storage records are measured separately
from accuracy and do not replace previous benchmarks.
'''
    targets[3].write_text(text);print(overall);print('Best:',best['floorplan'],'Worst:',worst['floorplan'],'Improved:',improved,'/23')
    return result

if __name__=='__main__':run(Path(__file__).resolve().parent)
