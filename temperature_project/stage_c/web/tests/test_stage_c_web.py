"""
Automated tests for the isolated Stage C web package (stage_c/web/).

Uses Python's standard-library unittest so it runs with zero extra dependencies:
    cd temperature_project
    python -m unittest discover -s stage_c/web/tests -v
"""

from __future__ import annotations

import csv
import shutil
import tempfile
import unittest
from pathlib import Path
from typing import List, Tuple

import numpy as np
from flask import Flask, jsonify

from stage_c.web import StageCRegistry, register_stage_c, stage_c_bp
from stage_c.web.model_registry import STAGE_C_ROOT, floorplan_tag
from stage_c.web.standalone_app import create_standalone_app


def _load_active_ptrace(ptrace_path: Path, active_core_names: Tuple[str, ...]) -> List[List[float]]:
    with ptrace_path.open("r", encoding="utf-8") as f:
        lines = [ln.strip() for ln in f if ln.strip()]
    headers = lines[0].split()
    col_indices = [headers.index(c) for c in active_core_names]
    rows: List[List[float]] = []
    for ln in lines[1:]:
        vals = [float(x) for x in ln.split()]
        rows.append([vals[i] for i in col_indices])
    return rows


def _load_saved_prediction_csv(csv_path: Path, active_core_names: Tuple[str, ...]) -> np.ndarray:
    with csv_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        rows = [[float(row[f"{c}_pred_C"]) for c in active_core_names] for row in reader]
    return np.asarray(rows, dtype=np.float64)


def _resolve_workload_c_paths(full_name: str, cohort: str) -> Tuple[Path, Path]:
    if cohort == "original_pilot_c01d":
        return (
            STAGE_C_ROOT / "simulations" / "c01d" / "c01d_independent.ptrace",
            STAGE_C_ROOT / "validation" / "c01d" / "predictions" / "two_speed_Case3_Joint_AB_on_c01d.csv",
        )
    tag = floorplan_tag(full_name)
    if cohort == "step7_pilot":
        return (
            STAGE_C_ROOT / "multi_floorplan_pilot" / "simulations" / tag / "workload_c_validation" / f"{tag}_c_validation.ptrace",
            STAGE_C_ROOT / "multi_floorplan_pilot" / "validations" / "predictions" / f"{tag}_two_speed_on_workload_c.csv",
        )
    return (
        STAGE_C_ROOT / "full_floorplan_expansion" / "simulations" / tag / "workload_c_validation" / f"{tag}_c_validation.ptrace",
        STAGE_C_ROOT / "full_floorplan_expansion" / "validations" / "predictions" / f"{tag}_two_speed_on_workload_c.csv",
    )


class TestStageCWebIsolated(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.app = create_standalone_app()
        cls.app.config["TESTING"] = True
        cls.client = cls.app.test_client()
        cls.registry = StageCRegistry()

    def test_01_registry_loads_all_23_models_with_verified_sha256(self) -> None:
        self.assertEqual(self.registry.count, 23)
        summary = self.registry.aggregate_summary()
        self.assertEqual(summary["floorplans_verified"], 23)
        self.assertTrue(summary["all_sha256_verified"])
        self.assertEqual(summary["raw_float64_bytes_per_floorplan"], 704)
        self.assertEqual(summary["total_parameters_per_floorplan"], 88)
        self.assertEqual(summary["total_raw_float64_bytes_all_23"], 23 * 704)
        self.assertAlmostEqual(summary["mean_two_speed_unseen_rmse_c"], 1.678150, places=4)
        self.assertAlmostEqual(summary["mean_per_floorplan_rmse_reduction_pct"], 73.4128, places=3)

    def test_02_registry_fails_fast_on_sha256_tampering(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            fake_stage_c = Path(tmpdir) / "transient"
            src_param = STAGE_C_ROOT / "models" / "two_speed" / "parameters" / "case3_final_joint_AB.json"
            dst_param = fake_stage_c / "models" / "two_speed" / "parameters" / "case3_final_joint_AB.json"
            dst_param.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src_param, dst_param)
            dst_param.write_text(dst_param.read_text(encoding="utf-8") + "\n", encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "SHA-256 mismatch"):
                StageCRegistry(stage_c_root=fake_stage_c)

    def test_03_numerical_parity_all_23_floorplans_against_audited_workload_c(self) -> None:
        for full_name in self.registry.list_names():
            entry = self.registry.get(full_name)
            ptrace_path, saved_csv_path = _resolve_workload_c_paths(full_name, entry.cohort)

            if ptrace_path.is_file() and saved_csv_path.is_file():
                # Full local workspace with raw HotSpot traces + audited Workload C CSVs
                power_rows = _load_active_ptrace(ptrace_path, entry.spec.active_core_names)
                expected_mat = _load_saved_prediction_csv(saved_csv_path, entry.spec.active_core_names)
                tol = 1e-4  # Audited CSVs are rounded to 4 decimal places
            else:
                # Self-contained Git clone: verify exact API parity against frozen model on 200-step Workload B
                phases = [(0, 50, 10.0, 5.0), (50, 100, 20.0, 10.0), (100, 150, 10.0, 5.0), (150, 200, 15.0, 8.0)]
                mat = np.zeros((200, 8), dtype=np.float64)
                for idx, blk in enumerate(entry.spec.active_cores):
                    for r0, r1, p_big, p_lit in phases:
                        mat[r0:r1, idx] = p_big if blk.core_type == "Big" else p_lit
                power_rows = mat.tolist()
                expected_mat = entry.model.predict_sequence_c(mat, init_temp_c=45.0)
                tol = 1e-5

            self.assertEqual(len(power_rows), 200)
            resp = self.client.post(
                "/api/stage-c/predict",
                json={
                    "floorplan": full_name,
                    "mode": "trace",
                    "num_steps": 200,
                    "dt_seconds": 0.01,
                    "init_temp_c": 45.0,
                    "power_trace_w": power_rows,
                },
            )
            self.assertEqual(resp.status_code, 200, f"{full_name} failed: {resp.get_json()}")
            data = resp.get_json()
            self.assertEqual(data["domain_status"], "validated", f"Calibrated trace should be validated for {full_name}")
            self.assertEqual(data["model_footprint"]["raw_float64_bytes"], 704)
            self.assertEqual(data["model_footprint"]["param_sha256"], entry.param_sha256)

            api_mat = np.column_stack(
                [np.asarray(data["temperatures_c"][c], dtype=np.float64) for c in entry.spec.active_core_names]
            )
            self.assertEqual(api_mat.shape, (200, 8))
            self.assertEqual(expected_mat.shape, (200, 8))
            max_diff = float(np.max(np.abs(api_mat - expected_mat)))
            self.assertLess(max_diff, tol, f"Numerical parity mismatch for {full_name}: max_diff={max_diff}")

    def test_04_api_health_and_floorplans_endpoints(self) -> None:
        r_health = self.client.get("/api/stage-c/health")
        self.assertEqual(r_health.status_code, 200)
        h = r_health.get_json()
        self.assertEqual(h["status"], "ok")
        self.assertEqual(h["summary"]["floorplans_verified"], 23)
        self.assertTrue(h["summary"]["all_sha256_verified"])

        r_fps = self.client.get("/api/stage-c/floorplans")
        self.assertEqual(r_fps.status_code, 200)
        fps = r_fps.get_json()
        self.assertEqual(fps["count"], 23)
        self.assertEqual(len(fps["floorplans"]), 23)

        r_detail = self.client.get("/api/stage-c/floorplan/Style_2/fp_003")
        self.assertEqual(r_detail.status_code, 200)
        d = r_detail.get_json()
        self.assertEqual(d["full_name"], "Style_2/fp_003")
        self.assertEqual(len(d["cores"]), 8)
        self.assertGreaterEqual(len(d["presets"]), 4)

        r_missing = self.client.get("/api/stage-c/floorplan/Style_9/fp_999")
        self.assertEqual(r_missing.status_code, 404)

    def test_05_constant_and_schedule_modes_and_domain_classification(self) -> None:
        entry_s1 = self.registry.get("Style_1/fp_001")
        val_vec = [20.0 if b.core_type == "Big" else 8.0 for b in entry_s1.spec.active_cores]

        # 1. Validated constant mode
        r_val = self.client.post(
            "/api/stage-c/predict",
            json={
                "floorplan": "Style_1/fp_001",
                "mode": "constant",
                "powers_w": val_vec,
            },
        )
        self.assertEqual(r_val.status_code, 200)
        d_val = r_val.get_json()
        self.assertEqual(d_val["domain_status"], "validated")
        self.assertEqual(d_val["extrapolation_warnings"], [])
        self.assertEqual(len(d_val["plot_times_s"]), 201)
        self.assertEqual(d_val["plot_times_s"][0], 0.0)
        self.assertEqual(d_val["plot_times_s"][-1], 2.0)
        self.assertGreater(d_val["model_footprint"]["live_inference_time_us"], 0.0)

        # 2. Extrapolated constant mode (physically valid 2 W on one Little and 30 W on one Big)
        ext_vec = list(val_vec)
        first_big = next(i for i, b in enumerate(entry_s1.spec.active_cores) if b.core_type == "Big")
        first_lit = next(i for i, b in enumerate(entry_s1.spec.active_cores) if b.core_type == "Little")
        ext_vec[first_big] = 30.0
        ext_vec[first_lit] = 2.0
        r_ext = self.client.post(
            "/api/stage-c/predict",
            json={
                "floorplan": "Style_1/fp_001",
                "mode": "constant",
                "powers_w": ext_vec,
            },
        )
        self.assertEqual(r_ext.status_code, 200)
        d_ext = r_ext.get_json()
        self.assertEqual(d_ext["domain_status"], "extrapolated")
        self.assertEqual(len(d_ext["extrapolation_warnings"]), 2)

        # 3. Schedule mode (4 phases)
        r_sched = self.client.post(
            "/api/stage-c/predict",
            json={
                "floorplan": "Style_4/fp_010",
                "mode": "schedule",
                "segments": [
                    {"start_step": 1, "end_step": 50, "powers_w": [5, 5, 5, 5, 5, 5, 10, 10]},
                    {"start_step": 51, "end_step": 100, "powers_w": [10, 10, 10, 10, 10, 10, 20, 20]},
                    {"start_step": 101, "end_step": 150, "powers_w": [5, 5, 5, 5, 5, 5, 10, 10]},
                    {"start_step": 151, "end_step": 200, "powers_w": [8, 8, 8, 8, 8, 8, 15, 15]},
                ],
            },
        )
        self.assertEqual(r_sched.status_code, 200)
        d_sched = r_sched.get_json()
        self.assertEqual(d_sched["num_steps"], 200)
        self.assertEqual(len(d_sched["per_core_summary"]), 8)

    def test_06_validation_rejects_invalid_inputs(self) -> None:
        bad_payloads = [
            {"floorplan": "Style_1/fp_001", "mode": "constant", "powers_w": [-1, 8, 8, 8, 8, 8, 15, 15]},
            {"floorplan": "Style_1/fp_001", "mode": "constant", "powers_w": [8, 8, 8, 8, 8, 8, 55, 15]},
            {"floorplan": "Style_1/fp_001", "mode": "constant", "powers_w": [8, 8, 8]},
            {"floorplan": "Style_1/fp_001", "mode": "constant", "dt_seconds": 0.05, "powers_w": [8, 8, 8, 8, 8, 8, 15, 15]},
            {
                "floorplan": "Style_1/fp_001",
                "mode": "schedule",
                "segments": [
                    {"start_step": 1, "end_step": 120, "powers_w": [8, 8, 8, 8, 8, 8, 15, 15]},
                    {"start_step": 100, "end_step": 200, "powers_w": [8, 8, 8, 8, 8, 8, 15, 15]},
                ],
            },
            {
                "floorplan": "Style_1/fp_001",
                "mode": "schedule",
                "segments": [
                    {"start_step": 1, "end_step": 100, "powers_w": [8, 8, 8, 8, 8, 8, 15, 15]},
                ],
            },
        ]
        for payload in bad_payloads:
            resp = self.client.post("/api/stage-c/predict", json=payload)
            self.assertEqual(resp.status_code, 400, f"Expected 400 for {payload}, got {resp.status_code}")
            body = resp.get_json()
            self.assertIn("error", body)
            self.assertIn("code", body)

    def test_07_ui_html_and_static_assets(self) -> None:
        r_html = self.client.get("/stage-c")
        self.assertEqual(r_html.status_code, 200)
        html = r_html.get_data(as_text=True)
        self.assertIn("2D Transient Thermal Predictor", html)
        self.assertIn("floorplanSelect", html)
        self.assertIn("tempChart", html)
        self.assertIn("/stage-c/assets/stage_c/stage_c.css", html)
        self.assertIn("/stage-c/assets/stage_c/stage_c.js", html)

        with self.client.get("/stage-c/assets/stage_c/stage_c.css") as r_css:
            self.assertEqual(r_css.status_code, 200)
            self.assertIn("--sc-bg", r_css.get_data(as_text=True))

        with self.client.get("/stage-c/assets/stage_c/stage_c.js") as r_js:
            self.assertEqual(r_js.status_code, 200)
            self.assertIn("/api/stage-c/predict", r_js.get_data(as_text=True))

    def test_08_blueprint_isolation_preserves_stage_a_and_stage_b_routes(self) -> None:
        mock_app = Flask(__name__)

        @mock_app.get("/")
        def stage_a_home():
            return jsonify({"stage": "A", "untouched": True})

        @mock_app.get("/stage-b")
        def stage_b_home():
            return jsonify({"stage": "B", "untouched": True})

        register_stage_c(mock_app)

        rules = {rule.rule for rule in mock_app.url_map.iter_rules()}
        self.assertIn("/", rules)
        self.assertIn("/stage-b", rules)
        self.assertIn("/stage-c", rules)
        self.assertIn("/api/stage-c/predict", rules)

        stage_c_only_app = Flask(__name__)
        stage_c_only_app.register_blueprint(stage_c_bp)
        bp_rules = {r.rule for r in stage_c_only_app.url_map.iter_rules() if r.endpoint != "static"}
        for r in bp_rules:
            self.assertTrue(
                r.startswith("/stage-c") or r.startswith("/api/stage-c/"),
                f"Unexpected route in stage_c_bp: {r}",
            )

        with mock_app.test_client() as client:
            self.assertEqual(client.get("/").get_json(), {"stage": "A", "untouched": True})
            self.assertEqual(client.get("/stage-b").get_json(), {"stage": "B", "untouched": True})
            self.assertEqual(client.get("/stage-c").status_code, 200)
            self.assertEqual(client.get("/api/stage-c/health").status_code, 200)


if __name__ == "__main__":
    unittest.main()
