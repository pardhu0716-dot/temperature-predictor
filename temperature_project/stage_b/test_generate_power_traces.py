#!/usr/bin/env python3
"""Regression and unit tests for Stage B Power-Trace Generator.

Covers:
- All 5 pilot floorplans (b_fp_001 to b_fp_005) reference power assignment
- Non-standard Big/Little core numbering (Style_2 variations)
- Arbitrary floorplan block ordering and interleaving
- Exact .ptrace header-value alignment and delimiters
- Strict zero-power FILL enforcement
- Inconsistent and invalid geometry rejection
- Configurable random workloads and reproducibility
- HotSpot simulation execution and consistency against corrected reference
"""
import math
import os
from pathlib import Path
import subprocess
import tempfile
import unittest

from generate_power_traces import (
    BIG_CORE_DIM_M,
    BIG_CORE_REF_POWER_W,
    CoreType,
    DEFAULT_BIG_RANGE,
    DEFAULT_LITTLE_RANGE,
    FILL_REF_POWER_W,
    GeometryValidationError,
    LITTLE_CORE_DIM_M,
    LITTLE_CORE_REF_POWER_W,
    PowerAssignmentError,
    PowerTrace,
    TOTAL_REF_POWER_W,
    analyze_floorplan,
    format_ptrace,
    generate_custom_trace,
    generate_random_traces,
    generate_reference_trace,
    read_flp_lines,
    write_ptrace,
)

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parent
PILOT = HERE / "pilot"
HOTSPOT = Path("/home/pardhu/HotSpot/hotspot")
CONFIG = PROJECT / "hotspot.config"
CORRECTED_SIM_DIR = HERE / "corrected_simulations"


def make_flp_content(blocks):
    """Helper to create .flp string from list of (name, w, h, x, y)."""
    lines = ["# name width height left-x bottom-y (metres)"]
    for name, w, h, x, y in blocks:
        lines.append(f"{name} {w:.6f} {h:.6f} {x:.6f} {y:.6f}")
    return "\n".join(lines) + "\n"


class TestStageBPowerTraceGeneration(unittest.TestCase):

    def test_pilot_reference_workloads(self):
        """Test all 5 pilot floorplans: exactly 88 W total, correct Big/Little assignment."""
        for i in range(1, 6):
            pid = f"b_fp_{i:03d}"
            flp_path = PILOT / pid / f"{pid}_layer_0.flp"
            self.assertTrue(flp_path.is_file(), f"Missing pilot file: {flp_path}")

            profile = analyze_floorplan(flp_path)
            self.assertEqual(profile.num_big, 2)
            self.assertEqual(profile.num_little, 6)
            self.assertEqual(len(profile.blocks), profile.num_big + profile.num_little + profile.num_fills)

            trace = generate_reference_trace(profile)
            self.assertAlmostEqual(trace.total_power_w, TOTAL_REF_POWER_W, places=4)
            self.assertEqual(trace.headers, profile.header_order)
            self.assertEqual(len(trace.headers), len(trace.powers))

            p_map = trace.power_dict()
            for b in profile.big_cores:
                self.assertEqual(p_map[b], 20.0, f"{pid}: Big core {b} did not get 20W")
            for l in profile.little_cores:
                self.assertEqual(p_map[l], 8.0, f"{pid}: Little core {l} did not get 8W")
            for f in profile.fill_blocks:
                self.assertEqual(p_map[f], 0.0, f"{pid}: FILL block {f} did not get 0W")

            # Floorplan-specific checks
            if pid == "b_fp_002":
                # Big cores must be Core_6 and Core_7
                self.assertEqual(sorted(profile.big_cores), ["Core_6_L0", "Core_7_L0"])
                self.assertEqual(
                    sorted(profile.little_cores),
                    ["Core_0_L0", "Core_1_L0", "Core_2_L0", "Core_3_L0", "Core_4_L0", "Core_5_L0"]
                )
            elif pid == "b_fp_003":
                # Core_2 appears before Core_1 in header order
                idx_c2 = trace.headers.index("Core_2_L0")
                idx_c1 = trace.headers.index("Core_1_L0")
                self.assertLess(idx_c2, idx_c1, "In b_fp_003 Core_2_L0 should appear before Core_1_L0")
                self.assertEqual(trace.powers[idx_c2], 8.0, "Core_2_L0 is Little and must get 8.0 W")
                self.assertEqual(trace.powers[idx_c1], 20.0, "Core_1_L0 is Big and must get 20.0 W")

    def test_different_big_little_numbering(self):
        """Verify Big/Little assignment when Big cores are NOT Core_0 and Core_1."""
        test_cases = [
            # Case A: Big cores are 3 and 7 (like Style_2/fp_001)
            {"big": ["Core_3_L0", "Core_7_L0"], "little": ["Core_0_L0", "Core_1_L0", "Core_2_L0", "Core_4_L0", "Core_5_L0", "Core_6_L0"]},
            # Case B: Big cores are 3 and 5 (like Style_2/fp_002)
            {"big": ["Core_3_L0", "Core_5_L0"], "little": ["Core_0_L0", "Core_1_L0", "Core_2_L0", "Core_4_L0", "Core_6_L0", "Core_7_L0"]},
            # Case C: Big cores are 0 and 7
            {"big": ["Core_0_L0", "Core_7_L0"], "little": ["Core_1_L0", "Core_2_L0", "Core_3_L0", "Core_4_L0", "Core_5_L0", "Core_6_L0"]},
        ]

        with tempfile.TemporaryDirectory() as tmp_dir:
            for idx, tc in enumerate(test_cases):
                blocks = []
                x = 0.0
                for c in tc["big"]:
                    blocks.append((c, 0.004, 0.004, x, 0.0))
                    x += 0.004
                for c in tc["little"]:
                    blocks.append((c, 0.002, 0.002, x, 0.0))
                    x += 0.002
                blocks.append(("FILL_0_L0", 0.002, 0.002, x, 0.0))

                flp_path = Path(tmp_dir) / f"test_numbering_{idx}.flp"
                flp_path.write_text(make_flp_content(blocks))

                profile = analyze_floorplan(flp_path)
                self.assertEqual(sorted(profile.big_cores), sorted(tc["big"]))
                self.assertEqual(sorted(profile.little_cores), sorted(tc["little"]))

                trace = generate_reference_trace(profile)
                p_map = trace.power_dict()
                for c in tc["big"]:
                    self.assertEqual(p_map[c], 20.0)
                for c in tc["little"]:
                    self.assertEqual(p_map[c], 8.0)
                self.assertEqual(p_map["FILL_0_L0"], 0.0)
                self.assertAlmostEqual(trace.total_power_w, 88.0)

    def test_different_block_ordering(self):
        """Verify headers and values match floorplan block sequence regardless of order."""
        blocks = [
            ("FILL_0_L0", 0.002, 0.002, 0.0, 0.0),
            ("FILL_1_L0", 0.002, 0.002, 0.002, 0.0),
            ("Core_5_L0", 0.002, 0.002, 0.004, 0.0),
            ("Core_1_L0", 0.002, 0.002, 0.006, 0.0),
            ("Core_0_L0", 0.004, 0.004, 0.008, 0.0),
            ("Core_3_L0", 0.002, 0.002, 0.012, 0.0),
            ("Core_7_L0", 0.004, 0.004, 0.014, 0.0),
            ("Core_2_L0", 0.002, 0.002, 0.018, 0.0),
            ("Core_4_L0", 0.002, 0.002, 0.020, 0.0),
            ("Core_6_L0", 0.002, 0.002, 0.022, 0.0),
            ("FILL_2_L0", 0.002, 0.002, 0.024, 0.0),
        ]
        with tempfile.TemporaryDirectory() as tmp_dir:
            flp_path = Path(tmp_dir) / "scrambled.flp"
            flp_path.write_text(make_flp_content(blocks))

            profile = analyze_floorplan(flp_path)
            self.assertEqual(profile.header_order, [b[0] for b in blocks])

            trace = generate_reference_trace(profile)
            self.assertEqual(trace.headers, [b[0] for b in blocks])
            expected_powers = [
                0.0, 0.0, 8.0, 8.0, 20.0, 8.0, 20.0, 8.0, 8.0, 8.0, 0.0
            ]
            self.assertEqual(trace.powers, expected_powers)

    def test_header_alignment_and_formatting(self):
        """Test tab and space formatting of .ptrace files."""
        blocks = [
            ("Core_0_L0", 0.004, 0.004, 0.0, 0.0),
            ("Core_1_L0", 0.004, 0.004, 0.004, 0.0),
            ("Core_2_L0", 0.002, 0.002, 0.008, 0.0),
            ("Core_3_L0", 0.002, 0.002, 0.010, 0.0),
            ("Core_4_L0", 0.002, 0.002, 0.012, 0.0),
            ("Core_5_L0", 0.002, 0.002, 0.014, 0.0),
            ("Core_6_L0", 0.002, 0.002, 0.016, 0.0),
            ("Core_7_L0", 0.002, 0.002, 0.018, 0.0),
            ("FILL_0_L0", 0.002, 0.002, 0.020, 0.0),
        ]
        with tempfile.TemporaryDirectory() as tmp_dir:
            flp_path = Path(tmp_dir) / "test.flp"
            flp_path.write_text(make_flp_content(blocks))

            profile = analyze_floorplan(flp_path)
            trace = generate_reference_trace(profile)

            # Tab delimiter
            tab_str = format_ptrace(trace, delimiter="\t", float_format="{:.2f}")
            lines = tab_str.strip().splitlines()
            self.assertEqual(len(lines), 2)
            headers = lines[0].split("\t")
            vals = [float(v) for v in lines[1].split("\t")]
            self.assertEqual(headers, [b[0] for b in blocks])
            self.assertEqual(vals, [20.0, 20.0, 8.0, 8.0, 8.0, 8.0, 8.0, 8.0, 0.0])

            # Space delimiter
            space_str = format_ptrace(trace, delimiter=" ", float_format="{:.1f}")
            lines_s = space_str.strip().splitlines()
            self.assertEqual(lines_s[0].split(" "), [b[0] for b in blocks])
            self.assertEqual([float(v) for v in lines_s[1].split(" ")], [20.0, 20.0, 8.0, 8.0, 8.0, 8.0, 8.0, 8.0, 0.0])

            # Write file and verify
            out_file = Path(tmp_dir) / "out.ptrace"
            write_ptrace(out_file, trace, delimiter="\t")
            self.assertTrue(out_file.is_file())
            self.assertEqual(out_file.read_text(encoding="utf-8"), tab_str)

    def test_fill_power_assignment(self):
        """Test strict zero-power assignment to FILL blocks and rejection of non-zero fills."""
        blocks = [
            ("Core_0_L0", 0.004, 0.004, 0.0, 0.0),
            ("Core_1_L0", 0.004, 0.004, 0.004, 0.0),
            ("Core_2_L0", 0.002, 0.002, 0.008, 0.0),
            ("Core_3_L0", 0.002, 0.002, 0.010, 0.0),
            ("Core_4_L0", 0.002, 0.002, 0.012, 0.0),
            ("Core_5_L0", 0.002, 0.002, 0.014, 0.0),
            ("Core_6_L0", 0.002, 0.002, 0.016, 0.0),
            ("Core_7_L0", 0.002, 0.002, 0.018, 0.0),
            ("FILL_0_L0", 0.002, 0.002, 0.020, 0.0),
            ("FILL_1_L0", 0.002, 0.002, 0.022, 0.0),
            ("FILL_2_L0", 0.002, 0.002, 0.024, 0.0),
        ]
        with tempfile.TemporaryDirectory() as tmp_dir:
            flp_path = Path(tmp_dir) / "test.flp"
            flp_path.write_text(make_flp_content(blocks))
            profile = analyze_floorplan(flp_path)

            trace = generate_reference_trace(profile)
            p_map = trace.power_dict()
            self.assertEqual(p_map["FILL_0_L0"], 0.0)
            self.assertEqual(p_map["FILL_1_L0"], 0.0)
            self.assertEqual(p_map["FILL_2_L0"], 0.0)

            # Rejection of non-zero fill power in custom trace
            active_powers = {c: 10.0 for c in profile.big_cores + profile.little_cores}
            with self.assertRaises(PowerAssignmentError):
                generate_custom_trace(profile, active_powers, fill_power=1.0)

    def test_invalid_and_inconsistent_geometry_rejection(self):
        """Verify that any non-standard or corrupt floorplan is rejected."""
        valid_8_cores = [
            ("Core_0_L0", 0.004, 0.004, 0.0, 0.0),
            ("Core_1_L0", 0.004, 0.004, 0.004, 0.0),
            ("Core_2_L0", 0.002, 0.002, 0.008, 0.0),
            ("Core_3_L0", 0.002, 0.002, 0.010, 0.0),
            ("Core_4_L0", 0.002, 0.002, 0.012, 0.0),
            ("Core_5_L0", 0.002, 0.002, 0.014, 0.0),
            ("Core_6_L0", 0.002, 0.002, 0.016, 0.0),
            ("Core_7_L0", 0.002, 0.002, 0.018, 0.0),
        ]

        # Case 1: 3 Big cores, 5 Little cores
        bad_counts = list(valid_8_cores)
        bad_counts[2] = ("Core_2_L0", 0.004, 0.004, 0.008, 0.0)
        # Case 2: 1 Big core, 7 Little cores
        bad_counts_2 = list(valid_8_cores)
        bad_counts_2[0] = ("Core_0_L0", 0.002, 0.002, 0.0, 0.0)
        # Case 3: Only 7 cores total
        missing_core = valid_8_cores[:7]
        # Case 4: 9 cores total
        extra_core = valid_8_cores + [("Core_8_L0", 0.002, 0.002, 0.020, 0.0)]
        # Case 5: Core with weird dimensions (3x3 mm)
        weird_dim = list(valid_8_cores)
        weird_dim[0] = ("Core_0_L0", 0.003, 0.003, 0.0, 0.0)
        # Case 6: Unknown block prefix (e.g. SRAM_0)
        unknown_prefix = list(valid_8_cores) + [("SRAM_0", 0.002, 0.002, 0.020, 0.0)]
        # Case 7: Duplicate block name
        dup_name = list(valid_8_cores) + [("Core_0_L0", 0.002, 0.002, 0.020, 0.0)]

        with tempfile.TemporaryDirectory() as tmp_dir:
            for idx, bad_b in enumerate([bad_counts, bad_counts_2, missing_core, extra_core, weird_dim, unknown_prefix, dup_name]):
                flp_path = Path(tmp_dir) / f"invalid_{idx}.flp"
                flp_path.write_text(make_flp_content(bad_b))
                with self.assertRaises(GeometryValidationError, msg=f"Failed to reject case {idx}"):
                    analyze_floorplan(flp_path)

    def test_configurable_workloads(self):
        """Test random workload generation with custom ranges, sample counts, and seeds."""
        flp_path = PILOT / "b_fp_001/b_fp_001_layer_0.flp"
        profile = analyze_floorplan(flp_path)

        # 1. Custom ranges
        big_rng = (12.5, 17.5)
        little_rng = (6.0, 9.0)
        traces = generate_random_traces(
            profile,
            num_samples=10,
            seed=1234,
            big_range=big_rng,
            little_range=little_rng,
            include_reference=True
        )
        self.assertEqual(len(traces), 10)

        # Sample 1 must be reference (20W / 8W)
        self.assertEqual(traces[0].sample_id, "sample_001")
        self.assertAlmostEqual(traces[0].total_power_w, 88.0)

        # Samples 2..10 must fall within custom ranges
        for t in traces[1:]:
            p_map = t.power_dict()
            for b in profile.big_cores:
                self.assertGreaterEqual(p_map[b], big_rng[0])
                self.assertLessEqual(p_map[b], big_rng[1])
            for l in profile.little_cores:
                self.assertGreaterEqual(p_map[l], little_rng[0])
                self.assertLessEqual(p_map[l], little_rng[1])
            for f in profile.fill_blocks:
                self.assertEqual(p_map[f], 0.0)

        # 2. Seed determinism
        traces_repeat = generate_random_traces(
            profile,
            num_samples=10,
            seed=1234,
            big_range=big_rng,
            little_range=little_rng,
            include_reference=True
        )
        for t1, t2 in zip(traces, traces_repeat):
            self.assertEqual(t1.powers, t2.powers)

    @unittest.skipUnless(os.environ.get("STAGE_B_RUN_HOTSPOT_TESTS") == "1",
                         "Real HotSpot integration is opt-in; preparation tests never simulate")
    def test_hotspot_simulation_match(self):
        """Run HotSpot with freshly generated reference traces and verify physical temperature ranges."""
        if not HOTSPOT.is_file() or not CONFIG.is_file():
            self.skipTest("HotSpot executable or config not found.")

        # Reference temperatures calculated and verified across the 88W cooling stack
        expected_ranges = {
            "b_fp_001": {"min": 150.41, "max": 155.43, "hottest": "layer_0_Core_6_L0"},
            "b_fp_002": {"min": 151.18, "max": 158.00, "hottest": "layer_0_Core_1_L0"},
            "b_fp_003": {"min": 150.38, "max": 154.10, "hottest": "layer_0_Core_2_L0"},
            "b_fp_004": {"min": 148.63, "max": 151.84, "hottest": "layer_0_Core_4_L0"},
            "b_fp_005": {"min": 150.21, "max": 155.01, "hottest": "layer_0_Core_2_L0"},
        }

        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)

            for i in range(1, 6):
                pid = f"b_fp_{i:03d}"
                flp_path = PILOT / pid / f"{pid}_layer_0.flp"
                profile = analyze_floorplan(flp_path)
                trace = generate_reference_trace(profile)

                ptrace_file = tmp_path / f"{pid}_test.ptrace"
                write_ptrace(ptrace_file, trace, delimiter="\t")

                steady_file = tmp_path / f"{pid}_test_steady.temp"
                cmd = [
                    str(HOTSPOT),
                    "-c", str(CONFIG),
                    "-f", f"{pid}_layer_0.flp",
                    "-p", str(ptrace_file),
                    "-grid_layer_file", f"{pid}.lcf",
                    "-model_type", "grid",
                    "-steady_file", str(steady_file)
                ]
                res = subprocess.run(cmd, cwd=str(PILOT / pid), capture_output=True, text=True)
                self.assertEqual(res.returncode, 0, f"HotSpot failed on {pid}: {res.stderr}")

                def parse_t(p):
                    return {line.split()[0]: float(line.split()[1]) for line in p.read_text().splitlines() if line.strip()}

                test_temps = parse_t(steady_file)
                core_c = {k: v - 273.15 for k, v in test_temps.items() if "Core_" in k}
                min_c = min(core_c.values())
                max_c = max(core_c.values())
                hottest = max(core_c, key=core_c.get)

                exp = expected_ranges[pid]
                self.assertAlmostEqual(min_c, exp["min"], places=2, msg=f"{pid} min temp mismatch")
                self.assertAlmostEqual(max_c, exp["max"], places=2, msg=f"{pid} max temp mismatch")
                self.assertEqual(hottest, exp["hottest"], f"{pid} hottest block mismatch")

                # If corrected_simulations directory exists, also check byte-for-byte numerical precision
                ref_steady_file = CORRECTED_SIM_DIR / f"{pid}_corrected_steady.temp"
                if ref_steady_file.is_file():
                    ref_temps = parse_t(ref_steady_file)
                    max_diff = max(abs(test_temps[k] - ref_temps[k]) for k in test_temps)
                    self.assertLess(max_diff, 1e-4, f"{pid}: Discrepancy {max_diff:.6f} K with corrected pilot")


if __name__ == "__main__":
    unittest.main()
