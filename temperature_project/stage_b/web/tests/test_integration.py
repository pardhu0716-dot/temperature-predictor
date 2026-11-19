"""Inference-only validation-reference/API tests; no training or held-out labels."""
import copy,csv,gzip
from pathlib import Path
import unittest
from unittest.mock import patch
import numpy as np
from flask import render_template
from combined_app import app
from app import _load_floorplan_bundle
from stage_b.ml.data import CORE_IDS,load_power_inputs
PROJECT=Path(__file__).resolve().parents[3]
UNITS={'geometry':'mm','power':'W','temperature':'C'}

class WebIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        app.testing=True;cls.client=app.test_client();cls.service=app.extensions['stage_b_predictor']
        cls.floor=cls.client.get('/api/stage-b/floorplans/b_fp_006').json
    def payload(self):
        f=self.floor
        return dict(mode='existing',floorplan_id=f['id'],units=UNITS,powers_w={r['id']:r['power_w'] for r in f['cores']})
    def custom(self):return dict(mode='custom',units=UNITS,die=copy.deepcopy(self.floor['die']),cores=copy.deepcopy(self.floor['cores']))

    def test_model_loaded_once_and_feature_contract(self):
        from stage_b.web import register_stage_b
        before_v1=id(self.service.bundle);before_v2=id(self.service.v2_model);register_stage_b(app)
        self.client.post('/api/stage-b/predict',json=self.payload())
        self.assertEqual(before_v1,id(app.extensions['stage_b_predictor'].bundle))
        self.assertEqual(before_v2,id(app.extensions['stage_b_predictor'].v2_model))
        info=self.client.get('/api/stage-b/model').json
        self.assertEqual((info['default_model_version'],info['alpha'],info['feature_count'],info['serialized_bundle_bytes']),('v2',.01,63,3028))
        v1=info['models']['v1']
        self.assertEqual((v1['alpha'],v1['feature_count'],v1['serialized_bundle_bytes']),(.001,252,11197))

    def test_all_77_geometries_and_predictions(self):
        cat=self.client.get('/api/stage-b/floorplans').json;self.assertEqual(cat['total_floorplans'],77)
        for f in cat['floorplans']:
            d=self.client.get('/api/stage-b/floorplans/'+f['id']).json;g,die=self.service.layouts[f['id']]
            np.testing.assert_array_equal([d['die'][k] for k in ('width_mm','height_mm')],die)
            np.testing.assert_array_equal([[r[k] for k in ('width_mm','height_mm','x_mm','y_mm')] for r in d['cores']],g)
            result=self.client.post('/api/stage-b/predict',json=dict(mode='existing',floorplan_id=f['id'],units=UNITS,powers_w={r['id']:r['power_w'] for r in d['cores']}))
            self.assertEqual(result.status_code,200);self.assertEqual(set(result.json['temperatures_c']),set(CORE_IDS))

    def test_240_saved_validation_cases_both_modes_no_retraining(self):
        refs={}
        with gzip.open(PROJECT/'stage_b/experiment_results/seven_models/validation_predictions.csv.gz','rt',newline='') as f:
            for r in csv.DictReader(f):
                if r['model']=='lasso':refs.setdefault((r['floorplan_id'],r['sample_id']),{})[r['core_id']]=float(r['predicted_c'])
        self.assertEqual(len(refs),240);powers=load_power_inputs(PROJECT/'stage_b/dataset_v1/floorplan_manifest.json')
        with patch.object(type(self.service.bundle.estimator),'fit',side_effect=AssertionError('Retraining forbidden')):
            for key,expected in refs.items():
                fid,sid=key;p=dict(mode='existing',model_version='v1',floorplan_id=fid,units=UNITS,powers_w=dict(zip(CORE_IDS,powers[key].tolist())))
                actual=self.client.post('/api/stage-b/predict',json=p);self.assertEqual(actual.status_code,200)
                np.testing.assert_allclose([actual.json['temperatures_c'][n] for n in CORE_IDS],[expected[n] for n in CORE_IDS],rtol=0,atol=1e-10)
                floor=self.service.floorplan(fid);rows=floor['cores'][::-1]
                for r in rows:r['power_w']=p['powers_w'][r['id']]
                custom=self.client.post('/api/stage-b/predict',json=dict(mode='custom',model_version='v1',units=UNITS,die=floor['die'],cores=rows))
                self.assertEqual(custom.status_code,200)
                np.testing.assert_allclose([custom.json['temperatures_c'][n] for n in CORE_IDS],[expected[n] for n in CORE_IDS],rtol=0,atol=1e-10)

    def test_all_eight_power_inputs(self):
        p=self.payload();baseline=self.client.post('/api/stage-b/predict',json=p).json['temperatures_c']
        for n in CORE_IDS:
            other=copy.deepcopy(p);other['powers_w'][n]+=.5
            result=self.client.post('/api/stage-b/predict',json=other)
            self.assertEqual(result.status_code,200);self.assertNotEqual(result.json['temperatures_c'],baseline)

    def test_invalid_geometry_numbers_ids_powers(self):
        changes=[lambda p:p.update(units={'geometry':'m','power':'W','temperature':'C'}),lambda p:p['cores'][0].update(x_mm=-1),lambda p:p['cores'][0].update(width_mm=3),lambda p:p['cores'][0].update(x_mm=999),lambda p:p['cores'][0].update(power_w=0),lambda p:p['cores'][0].update(power_w=True),lambda p:p['cores'][0].update(power_w='8'),lambda p:p['cores'][0].update(power_w=float('nan')),lambda p:p['cores'][0].update(x_mm=float('inf')),lambda p:p['cores'][0].update(id=p['cores'][1]['id']),lambda p:p['cores'].pop(),lambda p:p['die'].update(width_mm=0),lambda p:p['die'].update(width_mm=1001),lambda p:p['cores'][0].update(x_mm=p['cores'][1]['x_mm'],y_mm=p['cores'][1]['y_mm'])]
        for change in changes:
            p=self.custom();change(p)
            with self.subTest(payload=p):
                r=self.client.post('/api/stage-b/predict',json=p);self.assertEqual(r.status_code,400);self.assertIsInstance(r.json['error'],str)
        for p in ([],{}, {'mode':'nope'},{**self.payload(),'powers_w':{}}):self.assertEqual(self.client.post('/api/stage-b/predict',json=p).status_code,400)

    def test_api_formats_batches_limits(self):
        self.assertEqual(self.client.get('/api/stage-b/floorplans/missing').status_code,404)
        p=self.payload();p['floorplan_id']='missing';self.assertEqual(self.client.post('/api/stage-b/predict',json=p).status_code,404)
        self.assertEqual(self.client.post('/api/stage-b/predict',data='{',content_type='application/json').status_code,400)
        self.assertEqual(self.client.post('/api/stage-b/predict',data='x',content_type='text/plain').status_code,415)
        for b in ([],[self.payload()]*33):self.assertEqual(self.client.post('/api/stage-b/predict',json={'batch':b}).status_code,400)
        r=self.client.post('/api/stage-b/predict',json={'batch':[self.payload(),self.custom()]});self.assertEqual(r.status_code,200);self.assertEqual(len(r.json['predictions']),2)
        self.assertEqual(self.client.post('/api/stage-b/predict',data=' '*100001,content_type='application/json').status_code,413)

    def test_ambient_and_material_settings_are_fixed(self):
        for key in ('ambient_c','material','layers'):
            p=self.custom();p[key]=20
            self.assertEqual(self.client.post('/api/stage-b/predict',json=p).status_code,400)
        self.assertEqual(self.client.get('/api/stage-b/model').json['ambient_c'],45.)

    def test_stage_a_html_unchanged_except_navigation(self):
        with app.test_request_context('/'):
            original=render_template('index.html');nav=render_template('stage_navigation.html',current_stage='a')
        actual=self.client.get('/').text
        self.assertEqual(actual.replace('<link rel="stylesheet" href="/stage-b/assets/stage_b.css">','',1).replace(nav,'',1),original)
        self.assertIn('href="/stage-b"',actual)
        self.assertIn('/static/css/style.css',self.client.get('/stage-b').text)

    def test_all_23_stage_a_matrix_predictions_unchanged(self):
        cat=self.client.get('/api/floorplans').json;self.assertEqual(cat['total_floorplans'],23)
        for f in cat['floorplans']:
            b=_load_floorplan_bundle(f['style'],f['floorplan']);names=b['matrix_core_order']
            public=self.client.get('/api/floorplan/'+f['full_name']).json
            powers=np.array([c['midpoint_power_w'] for c in public['cores']]);expected=45+np.asarray(b['_internal']['R_rows'])@powers
            r=self.client.post('/api/predict',json=dict(floorplan=f['full_name'],powers=powers.tolist()))
            self.assertEqual(r.status_code,200)
            np.testing.assert_allclose([r.json['predictions_by_core'][n]['temp_c'] for n in names],expected,rtol=0,atol=1e-6)
            one=self.client.post('/api/predict',json=dict(floorplan=f['full_name'],powers=powers.tolist(),target=names[3]))
            self.assertEqual(one.status_code,200);self.assertAlmostEqual(one.json['selected_core']['temp_c'],expected[3],places=5)

if __name__=='__main__':unittest.main()
