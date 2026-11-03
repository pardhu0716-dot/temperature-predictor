"""Read-only regression tests. Run: python3 -m unittest discover -s tests -v"""
import contextlib
import csv
import hashlib
import io
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import run_all_stage_a as batch
import stage_a_pipeline
from predict import discover_floorplans


def protected_hashes():
    files = list((ROOT / 'dataset').rglob('*'))
    files += list(ROOT.glob('stage_a_*.csv'))
    return {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in files if p.is_file()}


class SavedModelTests(unittest.TestCase):
    def test_all_23_models_ready_without_raw_training_archives(self):
        catalog = discover_floorplans(str(ROOT / 'dataset/Sample_dataset/2D_floorplans'))
        self.assertEqual(len(catalog), 23)
        for item in catalog:
            with self.subTest(floorplan=item['full_name']):
                meta = batch.inspect_floorplan_metadata(item['path'])
                self.assertTrue(batch.check_floorplan_completed(
                    item['path'], item['floorplan'], meta['num_cores']))

    def test_batch_never_calibrates_launches_hotspot_or_changes_saved_files(self):
        before = protected_hashes()
        with patch.object(stage_a_pipeline, 'run_stage_a_for_floorplan',
                          side_effect=AssertionError('Recalibration forbidden')), \
             patch('subprocess.run', side_effect=AssertionError('Subprocess forbidden')), \
             contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(batch.main([]), 0)
        self.assertEqual(before, protected_hashes())

    def test_missing_model_fails_without_overwriting_summary(self):
        before = protected_hashes()
        with patch.object(batch, 'check_floorplan_completed', return_value=False), \
             patch('subprocess.run', side_effect=AssertionError('Subprocess forbidden')), \
             contextlib.redirect_stdout(io.StringIO()), \
             contextlib.redirect_stderr(io.StringIO()):
            self.assertEqual(batch.main([]), 1)
        self.assertEqual(before, protected_hashes())

    def test_missing_and_invalid_matrices_are_not_ready(self):
        with tempfile.TemporaryDirectory() as folder:
            fp = Path(folder) / 'fp_001'
            gen = fp / 'generated_samples'
            gen.mkdir(parents=True)
            self.assertFalse(batch.check_floorplan_completed(str(fp), 'fp_001', 8))
            matrix = gen / 'thermal_matrix_fp001.csv'
            for duplicate, value in [(False, 'nan'), (True, '1'), (False, '1')]:
                names = ['Core_0_L0'] * 8 if duplicate else [f'Core_{i}_L0' for i in range(8)]
                with matrix.open('w', newline='') as stream:
                    writer = csv.writer(stream)
                    writer.writerow(['target'] + names)
                    for name in names:
                        writer.writerow([name] + [value] * 8)
                self.assertEqual(batch.check_floorplan_completed(str(fp), 'fp_001', 8),
                                 not duplicate and value == '1')

    def test_overwrite_option_is_rejected(self):
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as error:
            batch.main(['--force-overwrite'])
        self.assertEqual(error.exception.code, 2)


if __name__ == '__main__':
    unittest.main()
