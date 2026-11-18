"""Synthetic fixtures test software only; no real labels or ML results are generated."""
import copy
import contextlib
import io
import importlib.util
import csv
import json
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest.mock import patch
from types import SimpleNamespace
import numpy as np
from stage_b.ml.data import CORE_IDS,FIELDS,Dataset,load_csv,load_geometry_manifest,load_power_inputs,schema_template
from stage_b.ml.features import build_features
from stage_b.ml.preprocessing import prepare,split_floorplans,load_prepared
from stage_b.ml.models import build_model
from stage_b.ml.optimizers import DenseRegressor
from stage_b.ml.evaluation import metrics
from stage_b.ml.training import train,evaluate_test
from stage_b.ml.external import check_compatibility,evaluate_external
from stage_b.ml.experiments import configurations,validate_experiments
from stage_b.ml.__main__ import main

PROJECT=Path(__file__).resolve().parents[3]
GEOMETRY=PROJECT/'stage_b/dataset_v1/floorplan_manifest.json'


def synthetic_csv(path):
    """58-column fixture: 2 identifiers + 8×(core ID + 6 numeric fields)."""
    schema=schema_template();layouts=load_geometry_manifest(GEOMETRY);source_power=load_power_inputs(GEOMETRY)
    headers=[schema['floorplan_column'],schema['sample_column']]
    for slot in schema['cores']:headers += [slot['id_column'],*[slot[f] for f in FIELDS]]
    rng=np.random.default_rng(123);rows=[]
    for fid,(geometry,die) in layouts.items():
        for sample in range(1,21):
            row={schema['floorplan_column']:fid,schema['sample_column']:f'sample_{sample:03d}'}
            for slot,index in zip(schema['cores'],rng.permutation(8)):
                g=geometry[index];big=g[0]==4
                power=source_power[(fid,f'sample_{sample:03d}')][index]
                temperature=100+index+sample*.03  # Deliberately synthetic, not a thermal result.
                row[slot['id_column']]=CORE_IDS[index]
                for field,value in zip(FIELDS,[*g,power,temperature]):row[slot[field]]=value
            rows.append(row)
    with path.open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=headers);writer.writeheader();writer.writerows(rows)
    return schema,headers,rows


class DataTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp=tempfile.TemporaryDirectory();cls.path=Path(cls.temp.name)/'synthetic.csv'
        cls.schema,cls.headers,cls.rows=synthetic_csv(cls.path)
        cls.data=load_csv(cls.path,cls.schema,GEOMETRY)
    @classmethod
    def tearDownClass(cls):cls.temp.cleanup()

    def test_schema_units_and_core_alignment(self):
        d=self.data
        self.assertEqual(d.geometry_mm.shape,(1540,8,4));self.assertEqual(d.temperatures_c.shape,(1540,8))
        np.testing.assert_allclose(d.temperatures_c[0],100+np.arange(8)+.03)
        self.assertEqual(len(self.headers),58)
        rows=copy.deepcopy(self.rows)
        for row in rows:
            for slot in self.schema['cores']:
                for f in ('width','height','x','y'):row[slot[f]]=float(row[slot[f]])/1000
                row[slot['temperature']]=float(row[slot['temperature']])+273.15
        with tempfile.TemporaryDirectory() as folder:
            p=Path(folder)/'units.csv'
            with p.open('w',newline='') as stream:
                writer=csv.DictWriter(stream,fieldnames=self.headers);writer.writeheader();writer.writerows(rows)
            schema={**self.schema,'geometry_unit':'m','temperature_unit':'K'}
            other=load_csv(p,schema,GEOMETRY)
            np.testing.assert_allclose(d.geometry_mm,other.geometry_mm)
            np.testing.assert_allclose(d.temperatures_c,other.temperatures_c)

    def test_validation_rejects_bad_rows(self):
        changes=[('missing',lambda r:r.update({self.schema['cores'][0]['power']:''})),
                 ('nonfinite',lambda r:r.update({self.schema['cores'][0]['temperature']:'nan'})),
                 ('bad ID',lambda r:r.update({self.schema['cores'][0]['id_column']:'Core_999_L0'})),
                 ('bad power',lambda r:r.update({self.schema['cores'][0]['power']:-1})),
                 ('wrong geometry',lambda r:r.update({self.schema['cores'][0]['width']:3})),
                 ('wrong floorplan',lambda r:r.update({'floorplan_id':'Style_1/fp_001'})),
                 ('wrong sample',lambda r:r.update({'sample_id':'sample_999'}))]
        for label,change in changes:
            with self.subTest(label=label),tempfile.TemporaryDirectory() as folder:
                rows=copy.deepcopy(self.rows);change(rows[0]);p=Path(folder)/'bad.csv'
                with p.open('w',newline='') as stream:
                    writer=csv.DictWriter(stream,fieldnames=self.headers);writer.writeheader();writer.writerows(rows)
                with self.assertRaises(ValueError):load_csv(p,self.schema,GEOMETRY)
        for rows in [self.rows+[self.rows[0]],self.rows[1:]]:
            with tempfile.TemporaryDirectory() as folder:
                p=Path(folder)/'bad.csv'
                with p.open('w',newline='') as stream:
                    writer=csv.DictWriter(stream,fieldnames=self.headers);writer.writeheader();writer.writerows(rows)
                with self.assertRaises(ValueError):load_csv(p,self.schema,GEOMETRY)

    def test_floorplan_splits_and_train_only_scaling(self):
        parts=split_floorplans(self.data.floorplan_ids)
        self.assertEqual([len(v) for v in parts.values()],[53,12,12])
        self.assertEqual(parts,split_floorplans(self.data.floorplan_ids))
        self.assertNotEqual(parts,split_floorplans(self.data.floorplan_ids,seed=43))
        self.assertFalse(set(parts['train'])&set(parts['test']))
        X,defs=build_features(self.data);self.assertEqual(X.shape,(1540,252));self.assertEqual(len(defs),252)
        with tempfile.TemporaryDirectory() as folder:
            out=Path(folder)/'prepared';prepare(self.data,out,geometry_manifest=GEOMETRY)
            manifest,scaling,t=load_prepared(out,'train')
            self.assertEqual([manifest['split_row_counts'][k] for k in ('train','validation','test')],[1060,240,240])
            np.testing.assert_allclose(scaling['x_mean'],t['X'].mean(axis=0))
            np.testing.assert_allclose(scaling['y_mean'],t['y'].mean(axis=0))
            altered=copy.deepcopy(self.data)
            mask=np.isin(altered.floorplan_ids,parts['validation']+parts['test'])
            altered.temperatures_c[mask]+=1000;altered.powers_w[mask]+=1000
            prepare(altered,Path(folder)/'other',geometry_manifest=GEOMETRY)
            self.assertEqual(json.loads((out/'scaling.json').read_text()),json.loads((Path(folder)/'other/scaling.json').read_text()))
            with self.assertRaises(FileExistsError):prepare(self.data,out,geometry_manifest=GEOMETRY)

    def test_feature_no_target_leakage(self):
        first,_=build_features(self.data);changed=copy.deepcopy(self.data);changed.temperatures_c+=999
        second,_=build_features(changed);np.testing.assert_array_equal(first,second)
        pair_index=next(i for i,d in enumerate(build_features(self.data)[1]) if d['name']=='Core_0_L0.power_density')
        self.assertAlmostEqual(first[0,pair_index],self.data.powers_w[0,0]/np.prod(self.data.geometry_mm[0,0,:2]))

    def test_end_to_end_selection_and_test_gate(self):
        # Synthetic fixture training only. Never opens a real CSV or simulator.
        with tempfile.TemporaryDirectory() as folder:
            prepared=Path(folder)/'prepared';run=Path(folder)/'run'
            prepare(self.data,prepared,geometry_manifest=GEOMETRY)
            import stage_b.ml.training as training
            original=training.load_prepared;parts=[]
            def tracked(path,part):parts.append(part);self.assertNotEqual(part,'test');return original(path,part)
            with patch.object(training,'load_prepared',side_effect=tracked),patch('subprocess.run',side_effect=AssertionError('No simulator allowed')):
                reports=train(prepared,run,[dict(name='ridge',kind='ridge',parameters=dict(alpha=1.))],repeats=2)
            self.assertEqual(parts,['train','validation']);self.assertFalse(reports[0]['test_evaluated'])
            self.assertGreater(reports[0]['serialized_model_bytes'],0)
            self.assertFalse((run/'final_test_report.json').exists())
            final=evaluate_test(prepared,run);self.assertEqual(final['test']['samples'],240)
            with self.assertRaises(FileExistsError):evaluate_test(prepared,run)
            with self.assertRaises(FileExistsError):train(prepared,run,[dict(name='ridge',kind='ridge')])
            with (run/'ridge/model.joblib').open('ab') as stream:stream.write(b'changed')
            with self.assertRaises(ValueError):evaluate_test(prepared,run)

    def test_cli_smoke_on_synthetic_fixture_only(self):
        with tempfile.TemporaryDirectory() as folder,contextlib.redirect_stdout(io.StringIO()):
            schema=Path(folder)/'schema.json';schema.write_text(json.dumps(self.schema))
            prepared=Path(folder)/'prepared';run=Path(folder)/'ridge'
            self.assertEqual(main(['prepare','--csv',str(self.path),'--schema',str(schema),'--geometry-manifest',str(GEOMETRY),'--out',str(prepared)]),0)
            self.assertEqual(main(['train','--prepared',str(prepared),'--out',str(run),'--models','ridge','--alpha','.5','--latency-repeats','2']),0)
            self.assertTrue((run/'selection.json').is_file())
            self.assertFalse((run/'final_test_report.json').exists())

    def test_cli_missing_data_and_schema_template(self):
        self.assertEqual(main(['validate','--csv','/nonexistent/stage_b_training_dataset.csv','--schema','/nonexistent/schema.json']),1)
        with tempfile.TemporaryDirectory() as folder:
            p=Path(folder)/'schema.json'
            self.assertEqual(main(['schema-template','--out',str(p)]),0)
            self.assertEqual(main(['schema-template','--out',str(p)]),1)
            empty=Path(folder)/'empty.csv';empty.touch()
            self.assertEqual(main(['inspect-csv','--csv',str(empty)]),1)


class ModelTests(unittest.TestCase):
    def setUp(self):
        rng=np.random.default_rng(17);self.X=rng.normal(size=(40,4));self.Y=self.X@rng.normal(size=(4,8))

    def test_all_models_multi_output(self):
        configs=[('ols',{}),('ridge',{'alpha':1}),('lasso',{'alpha':.001}),('elastic_net',{'alpha':.001}),
                 ('random_forest',{'n_estimators':3,'max_depth':2}),
                 ('mlp',{'epochs':3})]
        for kind,params in configs:
            with self.subTest(kind=kind):
                model=build_model(kind,params)
                model.fit(self.X,self.Y);prediction=model.predict(self.X)
                self.assertEqual(prediction.shape,(40,8));self.assertTrue(np.isfinite(prediction).all())
                if kind=='ols':self.assertLess(np.mean((prediction-self.Y)**2),1e-20)

    @unittest.skipUnless(importlib.util.find_spec('xgboost'), 'Optional XGBoost CPU dependency not installed')
    def test_xgboost_multi_output(self):
        model=build_model('xgboost',dict(n_estimators=2,max_depth=2))
        model.fit(self.X,self.Y)
        self.assertEqual(model.predict(self.X).shape,(40,8))

    def test_optimizers_learn_linear_problem_reproducibly(self):
        for optimizer,rate in [('batch_gd',.1),('sgd',.02),('adam',.04)]:
            config=dict(optimizer=optimizer,learning_rate=rate,epochs=200,tolerance=0,patience=200,random_state=91)
            a=DenseRegressor(**config).fit(self.X,self.Y);b=DenseRegressor(**config).fit(self.X,self.Y)
            self.assertLess(a.loss_curve_[-1],a.loss_curve_[0]*.1)
            np.testing.assert_array_equal(a.predict(self.X),b.predict(self.X))
            mlp=DenseRegressor(hidden_layers=(8,),optimizer=optimizer,epochs=5,learning_rate=.01).fit(self.X,self.Y)
            self.assertEqual(mlp.predict(self.X).shape,(40,8))

    def test_metrics_and_regularization(self):
        report=metrics(np.zeros((3,8)),np.ones((3,8))*2)
        self.assertEqual(report['mae_c'],2);self.assertEqual(report['rmse_c'],2);self.assertEqual(report['max_absolute_error_c'],2)
        self.assertEqual(set(report['per_core']),set(CORE_IDS))
        for group in ('core','optimizers','regularization'):validate_experiments(configurations(group))
        with self.assertRaises(ValueError):validate_experiments([dict(name='wrong',kind='random_forest',parameters={'optimizer':'adam'})])
        plain=DenseRegressor(optimizer='adam',epochs=60,l2=0).fit(self.X,self.Y)
        regular=DenseRegressor(optimizer='adam',epochs=60,l2=10).fit(self.X,self.Y)
        self.assertLess(sum(np.linalg.norm(w) for w in regular.weights_),sum(np.linalg.norm(w) for w in plain.weights_))

    def test_external_contract_fail_closed(self):
        contract=dict(hotspot_config_sha256='a'*64,material_layers=[dict(layer=0,specific_heat=1.75e6,resistivity=.01,thickness=.00015)],
                      ambient_c=45,geometry_unit='mm',temperature_unit='C',power_unit='W',core_order=list(CORE_IDS),label_source='HotSpot',feature_sha256='b'*64)
        self.assertTrue(check_compatibility(contract,contract)['compatible'])
        for changed in [{**contract,'label_source':'Stage A prediction'},{**contract,'ambient_c':25},{**contract,'hotspot_config_sha256':None}]:
            with self.assertRaises(ValueError):check_compatibility(contract,changed)
        with self.assertRaises(ValueError):check_compatibility({},contract)
        bundle=SimpleNamespace(simulation_contract=contract,feature_sha256=contract['feature_sha256'])
        with patch('stage_b.ml.external.load_csv',return_value=SimpleNamespace(schema={'ambient_c':25})):
            with self.assertRaisesRegex(ValueError,'ambient differs'):
                evaluate_external(bundle,'unused.csv',{},'unused.json',contract,contract)

if __name__=='__main__':unittest.main()
