"""Production-only checks; original Stage A regression tests remain unchanged."""
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import deployment_check
from wsgi import app


class ProductionTests(unittest.TestCase):
    def test_ready_and_no_debug(self):
        self.assertFalse(app.debug)
        with app.test_client() as client:
            response = client.get('/healthz')
            self.assertEqual(response.status_code, 200)
            self.assertEqual(response.json, dict(status='ok', stage='A', calibrated_floorplans=23))
            self.assertEqual(response.headers['X-Content-Type-Options'], 'nosniff')

    def test_missing_dataset_fails_closed(self):
        with patch.object(deployment_check, 'discover_floorplans', return_value=[]):
            with self.assertRaises(RuntimeError):
                deployment_check.check_runtime_assets()

    def test_nonfinite_matrix_fails_closed(self):
        original = deployment_check._load_floorplan_bundle
        def corrupt(*args):
            bundle = original(*args)
            bundle['thermal_matrix_k_per_w'][0][0] = float('nan')
            return bundle
        with patch.object(deployment_check, '_load_floorplan_bundle', side_effect=corrupt):
            with self.assertRaises(RuntimeError):
                deployment_check.check_runtime_assets()

    def test_arbitrary_working_directory_and_no_simulator(self):
        previous = os.getcwd()
        try:
            with tempfile.TemporaryDirectory() as directory:
                os.chdir(directory)
                with patch('subprocess.run', side_effect=AssertionError('Online prediction must not simulate')):
                    self.assertEqual(deployment_check.check_runtime_assets(), 23)
                    with app.test_client() as client:
                        floorplan = client.get('/api/floorplan/Style_2/fp_003').json
                        powers = {c['name']: c['midpoint_power_w'] for c in floorplan['cores']}
                        result = client.post('/api/predict', json=dict(floorplan='Style_2/fp_003', powers_by_core=powers))
                        self.assertEqual(result.status_code, 200)
                        self.assertEqual(len(result.json['predictions']), 8)
        finally:
            os.chdir(previous)

    def test_request_size_limit(self):
        with app.test_client() as client:
            response = client.post('/api/predict', data=' ' * 20000, content_type='application/json')
            self.assertEqual(response.status_code, 413)

if __name__ == '__main__':
    unittest.main()
