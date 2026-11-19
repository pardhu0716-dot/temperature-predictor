"""Focused Stage B v2 production loading, equivalence, permutation, and switching tests."""
import copy
import hashlib
import subprocess
import sys
from pathlib import Path
import unittest
import numpy as np
from combined_app import app
from stage_b.ml.data import CORE_IDS
from stage_b.web.service import MODEL_DIR, MODEL_SHA, V2_MODEL_PATH, V2_MODEL_SHA, UNITS

PROJECT = Path(__file__).resolve().parents[3]


class StageBV2IntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        app.testing = True
        cls.client = app.test_client()
        cls.service = app.extensions['stage_b_predictor']
        cls.floor = cls.client.get('/api/stage-b/floorplans/b_fp_006').json

    def existing_payload(self, model_version=None):
        p = dict(
            mode='existing',
            floorplan_id=self.floor['id'],
            units=UNITS,
            powers_w={r['id']: r['power_w'] for r in self.floor['cores']},
        )
        if model_version is not None:
            p['model_version'] = model_version
        return p

    def custom_payload(self, model_version=None):
        p = dict(
            mode='custom',
            units=UNITS,
            die=copy.deepcopy(self.floor['die']),
            cores=copy.deepcopy(self.floor['cores']),
        )
        if model_version is not None:
            p['model_version'] = model_version
        return p

    def test_v2_production_and_v1_artifact_integrity(self):
        self.assertEqual(V2_MODEL_PATH.name, 'model_v2.npz')
        self.assertEqual(V2_MODEL_PATH.stat().st_size, 3028)
        self.assertEqual(hashlib.sha256(V2_MODEL_PATH.read_bytes()).hexdigest(), V2_MODEL_SHA)
        v1_path = MODEL_DIR / 'model.joblib'
        self.assertEqual(v1_path.stat().st_size, 11197)
        self.assertEqual(hashlib.sha256(v1_path.read_bytes()).hexdigest(), MODEL_SHA)

    def test_v2_loads_in_fresh_python_process_without_experiments(self):
        code = (
            "import sys, numpy as np; "
            "from stage_b.ml.v2_model import load_v2_production_model; "
            "m = load_v2_production_model(); "
            "assert not any('experiments' in k for k in sys.modules); "
            "assert m.artifact_bytes == 3028 and len(m.feature_names) == 63; "
            "print('FRESH_LOAD_OK')"
        )
        proc = subprocess.run(
            [sys.executable, '-c', code],
            cwd=str(PROJECT),
            capture_output=True,
            text=True,
            check=True,
        )
        self.assertIn('FRESH_LOAD_OK', proc.stdout)

    def test_v2_is_default_and_uses_portable_npz(self):
        r = self.client.post('/api/stage-b/predict', json=self.existing_payload())
        self.assertEqual(r.status_code, 200)
        body = r.json
        self.assertEqual(body['model_version'], 'v2')
        self.assertEqual(body['model']['alpha'], 0.01)
        self.assertEqual(body['model']['feature_count'], 63)
        self.assertEqual(body['model']['serialized_bundle_bytes'], 3028)
        self.assertEqual(body['core_order'], list(CORE_IDS))
        self.assertEqual(set(body['temperatures_c']), set(CORE_IDS))

    def test_v2_and_v1_switching_on_identical_inputs(self):
        r_v2 = self.client.post('/api/stage-b/predict', json=self.existing_payload('v2'))
        r_v1 = self.client.post('/api/stage-b/predict', json=self.existing_payload('v1'))
        self.assertEqual(r_v2.status_code, 200)
        self.assertEqual(r_v1.status_code, 200)
        self.assertEqual(r_v2.json['model_version'], 'v2')
        self.assertEqual(r_v1.json['model_version'], 'v1')
        self.assertNotEqual(r_v2.json['temperatures_c'], r_v1.json['temperatures_c'])

    def test_invalid_model_version_rejected(self):
        for bad in ('v3', 'lasso', '', 2, None):
            p = self.existing_payload()
            p['model_version'] = bad
            r = self.client.post('/api/stage-b/predict', json=p)
            self.assertEqual(r.status_code, 400)

    def test_v2_existing_matches_custom_and_is_repeatable(self):
        r1 = self.client.post('/api/stage-b/predict', json=self.existing_payload('v2'))
        r2 = self.client.post('/api/stage-b/predict', json=self.existing_payload('v2'))
        rc = self.client.post('/api/stage-b/predict', json=self.custom_payload('v2'))
        self.assertEqual(r1.status_code, 200)
        self.assertEqual(r2.status_code, 200)
        self.assertEqual(rc.status_code, 200)
        t1 = [r1.json['temperatures_c'][n] for n in CORE_IDS]
        t2 = [r2.json['temperatures_c'][n] for n in CORE_IDS]
        tc = [rc.json['temperatures_c'][n] for n in CORE_IDS]
        np.testing.assert_array_equal(t1, t2)
        np.testing.assert_allclose(t1, tc, rtol=0, atol=1e-12)

    def test_v2_core_permutation_equivariance_via_api(self):
        base = self.custom_payload('v2')
        for i, c in enumerate(base['cores']):
            c['power_w'] = (11.0 + i * 0.7) if c['width_mm'] == 4 else (5.5 + i * 0.4)
        r_base = self.client.post('/api/stage-b/predict', json=base)
        self.assertEqual(r_base.status_code, 200)

        perm = [3, 6, 0, 5, 1, 7, 2, 4]
        reordered = copy.deepcopy(base)
        orig_cores = copy.deepcopy(base['cores'])
        for new_idx, old_idx in enumerate(perm):
            reordered['cores'][new_idx] = copy.deepcopy(orig_cores[old_idx])
            reordered['cores'][new_idx]['id'] = CORE_IDS[new_idx]

        r_perm = self.client.post('/api/stage-b/predict', json=reordered)
        self.assertEqual(r_perm.status_code, 200)
        for new_idx, old_idx in enumerate(perm):
            t_new = r_perm.json['temperatures_c'][CORE_IDS[new_idx]]
            t_old = r_base.json['temperatures_c'][orig_cores[old_idx]['id']]
            self.assertAlmostEqual(t_new, t_old, places=11)

    def test_stage_b_page_exposes_model_selector(self):
        html = self.client.get('/stage-b').text
        self.assertIn('id="b-model"', html)
        self.assertIn('Stage B v2', html)
        self.assertIn('3,028 B', html)
        self.assertIn('Stage B v1', html)


if __name__ == '__main__':
    unittest.main()
