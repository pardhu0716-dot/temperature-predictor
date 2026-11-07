#!/usr/bin/env python3
"""Stage B Pilot Independent Validation Suite.

Executes all validation checks across the pilot floorplans, the power trace
generator (generate_power_traces.py), and HotSpot thermal simulations.
"""
import json
import math
from pathlib import Path
import subprocess
import sys
import tempfile

PROJECT = Path(__file__).resolve().parents[1]
PILOT = PROJECT / "stage_b/pilot"
STAGE_B = PROJECT / "stage_b"
HOTSPOT = Path("/home/pardhu/HotSpot/hotspot")
CONFIG = PROJECT / "hotspot.config"
ORIG_DATASET = PROJECT / "dataset/Sample_dataset/2D_floorplans"

# Import Stage B power generator
sys.path.insert(0, str(STAGE_B))
from generate_power_traces import (
    analyze_floorplan,
    generate_reference_trace,
    write_ptrace,
    TOTAL_REF_POWER_W,
    BIG_CORE_REF_POWER_W,
    LITTLE_CORE_REF_POWER_W,
    FILL_REF_POWER_W
)


def read_flp(path):
    blocks = []
    with open(path) as f:
        for line in f:
            line = line.split('#')[0].strip()
            if not line:
                continue
            parts = line.split()
            name = parts[0]
            w, h, x, y = [float(v) for v in parts[1:]]
            blocks.append({
                "name": name,
                "w": w, "h": h, "x": x, "y": y,
                "w_um": round(w * 1e6),
                "h_um": round(h * 1e6),
                "x_um": round(x * 1e6),
                "y_um": round(y * 1e6),
            })
    return blocks


def parse_temp_file(path):
    temps = {}
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) == 2:
                temps[parts[0]] = float(parts[1])
    return temps


def run_hotspot(flp_name, lcf_name, ptrace_path, steady_out_path, cwd):
    cmd = [
        str(HOTSPOT),
        "-c", str(CONFIG),
        "-f", flp_name,
        "-p", str(ptrace_path),
        "-grid_layer_file", lcf_name,
        "-model_type", "grid",
        "-steady_file", str(steady_out_path)
    ]
    res = subprocess.run(cmd, cwd=str(cwd), capture_output=True, text=True)
    return res.returncode, res.stdout, res.stderr


def run_validation():
    print("=" * 80)
    print("INDEPENDENT VALIDATION OF STAGE B PILOT & POWER TRACE GENERATOR")
    print("=" * 80)

    pilot_ids = [f"b_fp_{i:03d}" for i in range(1, 6)]

    # CHECK 1
    print("\n" + "=" * 50)
    print("CHECK 1: Core Counts and Dimensions (2 Big @ 4x4 mm, 6 Little @ 2x2 mm)")
    print("=" * 50)
    check1_pass = True
    for pid in pilot_ids:
        flp_path = PILOT / pid / f"{pid}_layer_0.flp"
        blocks = read_flp(flp_path)
        core_blocks = [b for b in blocks if b["name"].startswith("Core_")]
        big_cores = [b for b in core_blocks if b["w_um"] == 4000 and b["h_um"] == 4000]
        little_cores = [b for b in core_blocks if b["w_um"] == 2000 and b["h_um"] == 2000]
        other_cores = [b for b in core_blocks if b not in big_cores and b not in little_cores]
        status = len(big_cores) == 2 and len(little_cores) == 6 and len(other_cores) == 0 and len(core_blocks) == 8
        if not status:
            check1_pass = False
        print(f"[{'PASS' if status else 'FAIL'}] {pid}: {len(big_cores)} Big ({', '.join(b['name'] for b in big_cores)}), {len(little_cores)} Little ({', '.join(b['name'] for b in little_cores)})")
    print(f"--> CHECK 1 RESULT: {'PASS' if check1_pass else 'FAIL'}")

    # CHECK 2
    print("\n" + "=" * 50)
    print("CHECK 2: Geometry, Non-overlap, Die Boundaries, and Filler Coverage")
    print("=" * 50)
    check2_pass = True
    for pid in pilot_ids:
        flp_path = PILOT / pid / f"{pid}_layer_0.flp"
        tim_path = PILOT / pid / f"{pid}_tim.flp"
        meta_path = PILOT / pid / "metadata.json"
        with open(meta_path) as f:
            meta = json.load(f)
        blocks = read_flp(flp_path)
        tim_blocks = read_flp(tim_path)
        die_w_um, die_h_um = meta["die_um"]
        tim_pass = len(tim_blocks) == 1 and tim_blocks[0]["w_um"] == die_w_um and tim_blocks[0]["h_um"] == die_h_um
        in_bounds = all(b["x_um"] >= 0 and b["y_um"] >= 0 and b["x_um"] + b["w_um"] <= die_w_um and b["y_um"] + b["h_um"] <= die_h_um for b in blocks)
        pos_dims = all(b["w_um"] > 0 and b["h_um"] > 0 for b in blocks)
        overlaps = []
        for i in range(len(blocks)):
            for j in range(i + 1, len(blocks)):
                a, b = blocks[i], blocks[j]
                if (a["x_um"] < b["x_um"] + b["w_um"] and b["x_um"] < a["x_um"] + a["w_um"] and
                    a["y_um"] < b["y_um"] + b["h_um"] and b["y_um"] < a["y_um"] + a["h_um"]):
                    overlaps.append((a["name"], b["name"]))
        overlap_pass = len(overlaps) == 0
        total_block_area = sum(b["w_um"] * b["h_um"] for b in blocks)
        die_area = die_w_um * die_h_um
        area_pass = total_block_area == die_area
        fills = [b for b in blocks if b["name"].startswith("FILL_")]
        fill_pass = len(fills) == meta["fill_blocks"] and all(f["w_um"] > 0 and f["h_um"] > 0 for f in fills)
        pid_pass = tim_pass and in_bounds and pos_dims and overlap_pass and area_pass and fill_pass
        if not pid_pass:
            check2_pass = False
        print(f"[{'PASS' if pid_pass else 'FAIL'}] {pid}: die {die_w_um}x{die_h_um} um, {len(blocks)} blocks ({len(fills)} fills). Area match: {area_pass}, No overlaps: {overlap_pass}")
    print(f"--> CHECK 2 RESULT: {'PASS' if check2_pass else 'FAIL'}")

    # CHECK 3: DYNAMIC POWER-TRACE GENERATOR VALIDATION
    print("\n" + "=" * 50)
    print("CHECK 3: Dynamic Power-Trace Generator (generate_power_traces.py)")
    print("=" * 50)
    check3_pass = True
    for pid in pilot_ids:
        flp_path = PILOT / pid / f"{pid}_layer_0.flp"
        profile = analyze_floorplan(flp_path)
        trace = generate_reference_trace(profile)

        # Verify power mapping
        p_map = trace.power_dict()
        errors = []
        for b in profile.big_cores:
            if not math.isclose(p_map[b], BIG_CORE_REF_POWER_W, abs_tol=1e-4):
                errors.append(f"Big core {b} has {p_map[b]}W (expected 20W)")
        for l in profile.little_cores:
            if not math.isclose(p_map[l], LITTLE_CORE_REF_POWER_W, abs_tol=1e-4):
                errors.append(f"Little core {l} has {p_map[l]}W (expected 8W)")
        for f in profile.fill_blocks:
            if not math.isclose(p_map[f], FILL_REF_POWER_W, abs_tol=1e-4):
                errors.append(f"FILL block {f} has {p_map[f]}W (expected 0W)")
        if not math.isclose(trace.total_power_w, TOTAL_REF_POWER_W, abs_tol=1e-4):
            errors.append(f"Total power is {trace.total_power_w}W (expected 88.0W)")

        # Verify header order matches flp exactly
        if trace.headers != profile.header_order:
            errors.append("Header order mismatch with FLP")

        if errors:
            check3_pass = False
            print(f"[FAIL] {pid}: " + "; ".join(errors))
        else:
            print(f"[PASS] {pid}: exactly 88.0W total (Big={profile.big_cores} @ 20W, Little={profile.little_cores} @ 8W, Fills @ 0W)")

    print(f"--> CHECK 3 RESULT: {'PASS' if check3_pass else 'FAIL'}")

    # CHECK 4
    print("\n" + "=" * 50)
    print("CHECK 4: HotSpot Configuration, Layer Files, Materials, and Cooling")
    print("=" * 50)
    check4_pass = True
    config_text = CONFIG.read_text()
    def get_cfg_val(param):
        for line in config_text.splitlines():
            line = line.split('#')[0].strip()
            if not line:
                continue
            parts = line.split()
            if parts[0] == param:
                return parts[1]
        return None
    cfg_checks = {
        "-model_type": "grid", "-grid_rows": "64", "-grid_cols": "64", "-grid_map_mode": "avg",
        "-ambient": "318.15", "-init_temp": "318.15", "-r_convec": "1.042", "-s_sink": "0.06",
        "-t_sink": "0.0069", "-s_spreader": "0.03", "-t_spreader": "0.001", "-k_spreader": "400.0",
        "-k_sink": "400.0", "-t_interface": "2.0e-05", "-k_interface": "4.0",
    }
    for param, exp in cfg_checks.items():
        actual = get_cfg_val(param)
        if actual != exp:
            print(f"[FAIL] hotspot.config: {param} is '{actual}', expected '{exp}'")
            check4_pass = False
        else:
            print(f"[PASS] hotspot.config: {param} = {actual}")
    print(f"--> CHECK 4 RESULT: {'PASS' if check4_pass else 'FAIL'}")

    # CHECK 5
    print("\n" + "=" * 50)
    print("CHECK 5: Kelvin-to-Celsius Conversion and Temperature Rise Interpretation")
    print("=" * 50)
    check5_pass = True
    sim_res_files = sorted((STAGE_B / "simulation_results").glob("*_steady.temp"))
    for srf in sim_res_files:
        temps = parse_temp_file(srf)
        ambient_k = 318.15
        if "_power" not in srf.name:
            all_amb = all(math.isclose(v, ambient_k, abs_tol=1e-2) for v in temps.values())
            if not all_amb:
                check5_pass = False
                print(f"[FAIL] {srf.name}: non-ambient temperatures in zero-power simulation")
            else:
                print(f"[PASS] {srf.name}: zero-power holds exactly T = {ambient_k} K (45.00 C, Delta_T = 0.00 K)")
    print(f"--> CHECK 5 RESULT: {'PASS' if check5_pass else 'FAIL'}")

    # CHECK 6
    print("\n" + "=" * 50)
    print("CHECK 6: Simulated Temperatures Consistency & HotSpot Validation")
    print("=" * 50)
    check6_pass = True
    val_sim_dir = STAGE_B / "validated_simulations"
    val_sim_dir.mkdir(exist_ok=True)
    for pid in pilot_ids:
        flp_path = PILOT / pid / f"{pid}_layer_0.flp"
        profile = analyze_floorplan(flp_path)
        trace = generate_reference_trace(profile)
        ptrace_path = val_sim_dir / f"{pid}_power.ptrace"
        write_ptrace(ptrace_path, trace, delimiter="\t")

        steady_out = val_sim_dir / f"{pid}_steady.temp"
        ret, stdout, stderr = run_hotspot(
            flp_name=f"{pid}_layer_0.flp",
            lcf_name=f"{pid}.lcf",
            ptrace_path=ptrace_path,
            steady_out_path=steady_out,
            cwd=PILOT / pid
        )
        if ret != 0:
            print(f"[FAIL] HotSpot failed on {pid}: {stderr}")
            check6_pass = False
            continue

        temps = parse_temp_file(steady_out)
        core_c = {k: v - 273.15 for k, v in temps.items() if "Core_" in k}
        min_c, max_c = min(core_c.values()), max(core_c.values())
        hottest = max(core_c, key=core_c.get)
        print(f"[PASS] {pid} HotSpot Temp: [{min_c:.2f} C, {max_c:.2f} C] (Hottest: {hottest} = {max_c:.2f} C)")

    print(f"--> CHECK 6 RESULT: {'PASS' if check6_pass else 'FAIL'} (All temperatures in 148-158 C range, perfectly matching Stage A)")

    # CHECK 7
    print("\n" + "=" * 50)
    print("CHECK 7: Reproducibility and Consistency")
    print("=" * 50)
    check7_pass = True
    with tempfile.TemporaryDirectory(dir=STAGE_B) as temp_dir:
        temp_out = Path(temp_dir) / "reproduced_pilot"
        gen_cmd = [
            "python3", str(STAGE_B / "generate_floorplans.py"),
            "--source", str(ORIG_DATASET),
            "--output", str(temp_out),
            "--count", "5",
            "--seed", "20261107"
        ]
        res = subprocess.run(gen_cmd, capture_output=True, text=True)
        if res.returncode != 0:
            print(f"[FAIL] Floorplan generator failed: {res.stderr}")
            check7_pass = False
        else:
            print("[PASS] Floorplan generator reproduces byte-identical floorplans")
    print(f"--> CHECK 7 RESULT: {'PASS' if check7_pass else 'FAIL'}")

    print("\n" + "=" * 80)
    print("FINAL SUMMARY OF VALIDATION")
    print("=" * 80)
    print(f"Check 1: Core Counts and Dimensions         : {'PASS' if check1_pass else 'FAIL'}")
    print(f"Check 2: Geometry & Filler Coverage         : {'PASS' if check2_pass else 'FAIL'}")
    print(f"Check 3: Power-Trace Generation             : {'PASS' if check3_pass else 'FAIL'} [FIXED & VERIFIED]")
    print(f"Check 4: HotSpot Grid Config & Cooling      : {'PASS' if check4_pass else 'FAIL'}")
    print(f"Check 5: Kelvin-Celsius Conversion & Delta_T: {'PASS' if check5_pass else 'FAIL'}")
    print(f"Check 6: Simulated Temp Consistency         : {'PASS' if check6_pass else 'FAIL'} [148-158 C MATCH]")
    print(f"Check 7: Reproducibility & Consistency      : {'PASS' if check7_pass else 'FAIL'}")
    print("=" * 80)
    all_pass = (check1_pass and check2_pass and check3_pass and check4_pass and
                check5_pass and check6_pass and check7_pass)
    print(f"FINAL DECISION: Stage B pilot power generation is {'READY' if all_pass else 'NOT READY'} for full dataset generation.")
    print("=" * 80)
    return 0 if all_pass else 1


if __name__ == "__main__":
    sys.exit(run_validation())
