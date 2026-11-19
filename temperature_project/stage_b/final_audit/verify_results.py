"""Read-only independent arithmetic/API consistency checks of the saved comparison."""
import csv,json
from collections import Counter
from pathlib import Path
import numpy as np
from combined_app import app
from .reference_data import original_workloads,legacy_baseline_tables
ROOT=Path(__file__).resolve().parent

def verify():
 rows=list(csv.DictReader((ROOT/'stage_a_vs_stage_b_all_predictions.csv').open()))
 summary=json.loads((ROOT/'comparison_summary.json').read_text())
 counts=Counter((r['floorplan'],r['sample_id']) for r in rows)
 assert len(rows)==3680 and len(counts)==460 and set(counts.values())=={8}
 assert set(Counter(f for f,s in counts).values())=={20}
 for model in ('stage_a','stage_b_v1','stage_b_v2'):
  e=np.array([float(r[model+'_temp_c'])-float(r['hotspot_temp_c']) for r in rows])
  actual=(np.abs(e).mean(),np.sqrt((e*e).mean()),np.abs(e).max())
  saved=summary['metrics'][model]
  np.testing.assert_allclose(actual,[saved[k] for k in ('mae_c','rmse_c','max_absolute_error_c')],rtol=0,atol=1e-12)
 with app.test_client() as client:
  assert client.get('/').status_code==client.get('/stage-b').status_code==200
  for w in original_workloads():
   if w.sample_id!='val_sample_001':continue
   result=client.post('/api/predict',json=dict(floorplan=w.floorplan,powers_by_core={f'Core_{i}_L0':float(w.powers[i]) for i in range(8)}))
   assert result.status_code==200
   np.testing.assert_allclose([result.json['predictions_by_core'][f'Core_{i}_L0']['temp_c'] for i in range(8)],45+w.matrix@w.powers,rtol=0,atol=1e-6)
 comp,fps,cores,legacy=legacy_baseline_tables()
 assert len(fps)==23 and len(cores)==8 and len(legacy)==3680
 np.testing.assert_allclose(comp['Stage A']['mae_c'],summary['metrics']['stage_a']['mae_c'],rtol=0,atol=1e-12)
 assert all(np.isclose(float(r['stage_b_temp_c']),float(s['stage_b_v1_temp_c']),rtol=0,atol=1e-12) for r,s in zip(legacy,rows))
 print(json.dumps(dict(result='PASS',workloads=460,observations=3680,independent_metrics_models=3,stage_a_api_floorplans=23,legacy_fallback_rows=len(legacy)),indent=2))
if __name__=='__main__':verify()
