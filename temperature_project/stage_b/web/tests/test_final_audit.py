"""Confirmed-bug regressions; no training or benchmark rewriting."""
import ast,copy,hashlib,io,json,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
from werkzeug.test import EnvironBuilder
from combined_app import app
from stage_b.ml.features_v2 import build_per_core_features
from stage_b.ml.v2_model import load_v2_production_model,V2_MODEL_DIR
G=np.array([[4,4,0,0],[4,4,4.5,0],[2,2,9,0],[2,2,11.5,0],[2,2,14,0],[2,2,9,3],[2,2,11.5,3],[2,2,14,3]],float)
P=np.array([18,14,5.5,6,9,7,8,9.5]);D=np.array([16.,6.])
PROJECT=Path(__file__).resolve().parents[3]
class FinalAuditTests(unittest.TestCase):
 def setUp(self):
  self.client=app.test_client();self.service=app.extensions['stage_b_predictor'];f=self.service.floorplan('b_fp_006')
  self.payload=dict(mode='existing',units=f['units'],floorplan_id=f['id'],powers_w={x['id']:x['power_w'] for x in f['cores']})
 def test_symmetric_permutation_equivariance(self):
  m=self.service.v2_model;expected=m.predict(G,P,D)[0]
  for perm in [np.arange(8)[::-1],np.array([2,4,1,6,0,7,5,3]),np.random.default_rng(9).permutation(8)]:np.testing.assert_allclose(m.predict(G[perm],P[perm],D)[0],expected[perm],atol=1e-11,rtol=0)
 def test_nearest_tie_is_mean_not_first(self):
  X,defs=build_per_core_features(G,P,D);i=[x['name'] for x in defs].index('coupling.nearest_p_over_d')
  self.assertAlmostEqual(X[3,i],((P[2]+P[4])/2)/3,places=12)
 def test_production_experimental_features_match(self):
  from stage_b.experiments.stage_b_v2.features_v2 import build_per_core_features as exp
  m=self.service.v2_model;a,_=build_per_core_features(G,P,D,groups=m.feature_groups);b,_=exp(G,P,D,groups=m.feature_groups);np.testing.assert_array_equal(a,b)
 def test_bad_feature_shapes_rejected(self):
  for g,p,d in [(np.zeros((7,4)),P,D),(G,np.zeros(7),D),(G,P,np.ones(3))]:
   with self.assertRaises(ValueError):build_per_core_features(g,p,d)
  for x in [np.zeros((7,63)),np.zeros((8,62)),np.full((8,63),np.nan)]:
   with self.assertRaises(ValueError):self.service.v2_model.predict_from_features(x)
 def test_invalid_scalers_and_scalars_fail_closed(self):
  for field,value in [('x_scale',0.),('x_scale',-1.),('y_scale',0.),('y_mean',np.nan),('intercept',np.inf)]:
   with self.subTest(field=field),tempfile.TemporaryDirectory() as t:
    root=Path(t);meta=json.loads((V2_MODEL_DIR/'metadata.json').read_text())
    with np.load(V2_MODEL_DIR/'model_v2.npz',allow_pickle=False) as f:data={k:f[k].copy() for k in f.files}
    data[field]=np.full_like(data[field],value);np.savez(root/'model_v2.npz',**data);meta['artifact']['npz_sha256']=hashlib.sha256((root/'model_v2.npz').read_bytes()).hexdigest();(root/'metadata.json').write_text(json.dumps(meta))
    with self.assertRaises(RuntimeError):load_v2_production_model(root)
 def test_missing_tampered_artifacts(self):
  with tempfile.TemporaryDirectory() as t:
   root=Path(t)
   with self.assertRaises((RuntimeError,OSError)):load_v2_production_model(root)
   (root/'metadata.json').write_bytes((V2_MODEL_DIR/'metadata.json').read_bytes());(root/'model_v2.npz').write_bytes(b'corrupt')
   with self.assertRaises(RuntimeError):load_v2_production_model(root)
 def test_streaming_input_read_is_bounded(self):
  class Guard(io.BytesIO):
   def read(self,n=-1):
    if n<0 or n>100001:raise AssertionError('Unbounded input read')
    return super().read(n)
  env=EnvironBuilder(path='/api/stage-b/predict',method='POST',content_type='application/json').get_environ();env.pop('CONTENT_LENGTH',None);env['wsgi.input_terminated']=True;env['wsgi.input']=Guard(b'x'*200000)
  self.assertEqual(self.client.open(env).status_code,413)
 def test_batch_root_settings_rejected(self):
  for key,value in [('model_version','v1'),('ambient_c',20)]:
   self.assertEqual(self.client.post('/api/stage-b/predict',json={'batch':[self.payload],key:value}).status_code,400)
 def test_huge_invalid_powers_useful_errors(self):
  for power in [-1,0,10**300,float('nan'),float('inf')]:
   p=copy.deepcopy(self.payload);p['powers_w']['Core_0_L0']=power;r=self.client.post('/api/stage-b/predict',json=p);self.assertEqual(r.status_code,400);self.assertNotIn('ufunc',r.json['error'])
 def test_model_runtime_error_returns_json(self):
  with patch.object(self.service.v2_model.__class__,'predict_from_features',side_effect=RuntimeError('private path')):r=self.client.post('/api/stage-b/predict',json=self.payload)
  self.assertEqual(r.status_code,503);self.assertNotIn('private path',r.json['error'])
 def test_scripts_explicitly_select_v1(self):
  for name in ['independent_audit.py','evaluate_final.py']:
   tree=ast.parse((PROJECT/'stage_b/experiments/stage_b_v2'/name).read_text());calls=[]
   for n in ast.walk(tree):
    if isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute) and isinstance(n.func.value,ast.Name) and n.func.value.id in ('v1_web','v1_predictor') and n.func.attr=='predict':
     calls.append(n);p=n.args[0];keys={k.value:v for k,v in zip(p.keys,p.values) if isinstance(k,ast.Constant)};self.assertEqual(ast.literal_eval(keys.get('model_version',ast.Constant(None))),'v1')
   self.assertEqual(len(calls),2)
 def test_named_reference_parsers_reject_misalignment(self):
  from stage_b.final_audit.reference_data import parse_power,parse_temperatures,score
  names=[f'Core_{i}_L0' for i in range(8)]
  with tempfile.TemporaryDirectory() as t:
   root=Path(t);pt=root/'power';st=root/'steady'
   pt.write_text(' '.join(names[::-1])+'\n'+' '.join(str(i+1) for i in range(8)[::-1])+'\n')
   np.testing.assert_array_equal(parse_power(pt),np.arange(1,9))
   st.write_text(''.join(f'layer_0_{n} {300+i}\n' for i,n in enumerate(names)))
   np.testing.assert_allclose(parse_temperatures(st),np.arange(300,308)-273.15)
   st.write_text(st.read_text()+'layer_0_Core_0_L0 300\n')
   with self.assertRaises(ValueError):parse_temperatures(st)
   pt.write_text(' '.join(names+[names[0]])+'\n'+' '.join(['1']*9))
   with self.assertRaises(ValueError):parse_power(pt)
  truth=np.zeros((2,8));pred=np.ones((2,8))*2;metrics=score(truth,pred)
  self.assertEqual((metrics['mae_c'],metrics['rmse_c'],metrics['max_absolute_error_c']),(2.,2.,2.))

 def test_training_scalers_match_frozen_legacy_features(self):
  from stage_b.ml.data import load_csv
  from sklearn.preprocessing import StandardScaler
  meta=self.service.v2_model.metadata
  schema=json.loads((PROJECT/'stage_b/models/final_lasso/schema.json').read_text())
  data=load_csv(PROJECT/'stage_b/stage_b_training_dataset.csv',schema,PROJECT/'stage_b/dataset_v1/floorplan_manifest.json')
  splits=meta['split_floorplans'];self.assertEqual([len(splits[k]) for k in ('train','validation','test')],[53,12,12])
  self.assertEqual(sum(map(len,splits.values())),len(set().union(*map(set,splits.values()))))
  train=np.isin(data.floorplan_ids,splits['train']);X,_=build_per_core_features(data.geometry_mm[train],data.powers_w[train],data.die_mm[train],groups=self.service.v2_model.feature_groups,nearest_policy='legacy_first')
  scaler=StandardScaler().fit(X)
  np.testing.assert_allclose(scaler.mean_,self.service.v2_model.x_mean,rtol=0,atol=1e-12)
  np.testing.assert_allclose(scaler.scale_,self.service.v2_model.x_scale,rtol=0,atol=1e-12)
  y=data.temperatures_c[train].reshape(-1)
  self.assertAlmostEqual(float(y.mean()),self.service.v2_model.y_mean,places=12);self.assertAlmostEqual(float(y.std()),self.service.v2_model.y_scale,places=12)

 def test_reflection_equivariance_on_symmetric_layout(self):
  m=self.service.v2_model;g=G.copy();g[:,2]=D[0]-G[:,2]-G[:,0]
  np.testing.assert_allclose(m.predict(G,P,D),m.predict(g,P,D),atol=1e-11,rtol=0)


class AdditionalAuditTests(unittest.TestCase):
 def test_frozen_production_equivalence_and_all_geometry_permutations(self):
  from stage_b.ml.data import load_csv
  from stage_b.models.v2_ridge.export_v2_production import load_experimental_bundle,EXP_MODEL_PATH
  service=app.extensions['stage_b_predictor'];m=service.v2_model;exp=load_experimental_bundle(EXP_MODEL_PATH)
  data=load_csv(PROJECT/'stage_b/stage_b_training_dataset.csv',json.loads((PROJECT/'stage_b/models/final_lasso/schema.json').read_text()),PROJECT/'stage_b/dataset_v1/floorplan_manifest.json')
  X=m.build_features(data.geometry_mm,data.powers_w,data.die_mm)
  reference=(exp.estimator.predict((X-exp.x_mean)/exp.x_scale)*exp.y_scale+exp.y_mean).reshape(-1,8)
  np.testing.assert_allclose(m.predict_from_features(X),reference,rtol=0,atol=1e-12)
  perm=np.array([7,1,3,0,5,2,6,4])
  np.testing.assert_allclose(m.predict(data.geometry_mm[:,perm],data.powers_w[:,perm],data.die_mm),reference[:,perm],rtol=0,atol=1e-11)
  for key in ('coef','x_mean','x_scale'):np.testing.assert_array_equal(getattr(m,key),getattr(exp,'x_mean' if key=='x_mean' else 'x_scale') if key!='coef' else exp.estimator.coef_)
 def test_concurrent_mixed_version_predictions_are_deterministic(self):
  from concurrent.futures import ThreadPoolExecutor
  service=app.extensions['stage_b_predictor'];f=service.floorplan('b_fp_006');base=dict(mode='existing',floorplan_id=f['id'],units=f['units'],powers_w={c['id']:c['power_w'] for c in f['cores']})
  expected={v:service.predict({**base,'model_version':v})['temperatures_c'] for v in ('v1','v2')}
  def request(i):
   version=('v1','v2')[i%2]
   with app.test_client() as client:
    response=client.post('/api/stage-b/predict',json={**base,'model_version':version})
    self.assertEqual(response.status_code,200);result=response.json
   self.assertEqual(result['temperatures_c'],expected[version])
  with ThreadPoolExecutor(max_workers=4) as pool:list(pool.map(request,range(40)))

if __name__=='__main__':unittest.main()
