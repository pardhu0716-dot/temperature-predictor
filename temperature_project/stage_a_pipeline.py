#!/usr/bin/env python3
"""
Stage A Reusable Calibration & Independent Validation Pipeline
Lightweight Temperature Predictor Project

Mathematical Model:
    Delta_T = R x P
    T_predicted = T_ambient + R x P
"""

import argparse
import csv
import glob
import math
import os
import platform
import random
import shutil
import subprocess
import sys
import time
from typing import Dict, List, Optional, Tuple

import numpy as np


AMBIENT_TEMP_K = 318.15
BIG_CORE_AREA_M2 = 0.004 * 0.004      # 16 mm^2 (4 mm x 4 mm)
LITTLE_CORE_AREA_M2 = 0.002 * 0.002   # 4 mm^2 (2 mm x 2 mm)
AREA_TOL = 1e-9

BIG_CORE_POWER_RANGE = (10.0, 20.0)
LITTLE_CORE_POWER_RANGE = (5.0, 10.0)


def to_wsl_path(win_or_posix_path: str) -> str:
    """Convert a Windows or UNC WSL path to a Linux path inside WSL Ubuntu."""
    p = os.path.abspath(win_or_posix_path).replace("\\", "/")
    for prefix in ["//wsl$/Ubuntu/", "//wsl.localhost/Ubuntu/"]:
        if p.lower().startswith(prefix.lower()):
            return "/" + p[len(prefix):].lstrip("/")
    return p


def parse_lcf(lcf_path: str) -> List[Dict[str, str]]:
    """Parse HotSpot .lcf file into a list of layer dicts."""
    if not os.path.exists(lcf_path):
        raise FileNotFoundError(f"LCF file not found: {lcf_path}")

    raw_lines = []
    with open(lcf_path, "r") as f:
        for line in f:
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            raw_lines.append(stripped)

    if len(raw_lines) == 0 or len(raw_lines) % 7 != 0:
        raise ValueError(
            f"Invalid .lcf format in {lcf_path}: expected a multiple of 7 non-comment lines, got {len(raw_lines)}"
        )

    layers = []
    for i in range(0, len(raw_lines), 7):
        layers.append({
            "layer_num": raw_lines[i],
            "lateral_heat_flow": raw_lines[i + 1],
            "power_dissipation": raw_lines[i + 2],
            "specific_heat": raw_lines[i + 3],
            "resistivity": raw_lines[i + 4],
            "thickness": raw_lines[i + 5],
            "flp_file": raw_lines[i + 6],
        })
    return layers


def parse_flp(flp_path: str) -> List[Dict[str, object]]:
    """Parse HotSpot .flp file into block metadata dicts."""
    if not os.path.exists(flp_path):
        raise FileNotFoundError(f"FLP file not found: {flp_path}")

    blocks = []
    with open(flp_path, "r") as f:
        for line_num, line in enumerate(f, start=1):
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            parts = stripped.split()
            if len(parts) != 5:
                raise ValueError(f"Invalid .flp line {line_num} in {flp_path}: '{stripped}'")
            name = parts[0]
            width = float(parts[1])
            height = float(parts[2])
            left_x = float(parts[3])
            bottom_y = float(parts[4])
            is_fill = name.upper().startswith("FILL")
            blocks.append({
                "name": name,
                "width": width,
                "height": height,
                "area": width * height,
                "left_x": left_x,
                "bottom_y": bottom_y,
                "is_fill": is_fill,
            })
    if not blocks:
        raise ValueError(f"No blocks found in floorplan file: {flp_path}")
    return blocks


def parse_steady_ptrace(ptrace_path: str) -> Tuple[List[str], Dict[str, float]]:
    """Parse steady-state .ptrace header ordering and initial power values."""
    if not os.path.exists(ptrace_path):
        raise FileNotFoundError(f"Steady ptrace file not found: {ptrace_path}")

    with open(ptrace_path, "r") as f:
        lines = [line.strip() for line in f if line.strip() and not line.strip().startswith("#")]

    if len(lines) < 2:
        raise ValueError(f"Steady ptrace file {ptrace_path} must contain header and at least 1 data row.")

    headers = lines[0].split()
    values = [float(x) for x in lines[1].split()]
    if len(headers) != len(values):
        raise ValueError(
            f"Column count mismatch in {ptrace_path}: {len(headers)} headers vs {len(values)} values."
        )

    return headers, dict(zip(headers, values))


def determine_core_power_range(
    block: Dict[str, object],
    init_power: float,
    allow_geometry_fallback: bool = False,
) -> Tuple[str, Tuple[float, float], Optional[str]]:
    """
    Determine core type and valid power range from documented geometry and initial steady ptrace.
    If there is any ambiguity or mismatch between geometry and initial ptrace, raise an error
    unless explicitly configured via allow_geometry_fallback.
    """
    w = float(block["width"])
    h = float(block["height"])
    area = float(block["area"])
    name = str(block["name"])

    is_big_geom = math.isclose(w, 0.004, abs_tol=1e-6) and math.isclose(h, 0.004, abs_tol=1e-6)
    is_little_geom = math.isclose(w, 0.002, abs_tol=1e-6) and math.isclose(h, 0.002, abs_tol=1e-6)

    if not is_big_geom and not is_little_geom:
        raise ValueError(
            f"Ambiguous core dimensions for '{name}': width={w} m, height={h} m (area={area*1e6:.2f} mm^2). "
            f"Documented core sizes are 4x4 mm (Big: 10-20 W) and 2x2 mm (Little: 5-10 W). "
            f"Stopping to request explicit power-range configuration instead of guessing."
        )

    if is_big_geom:
        geom_type = "Big (P-core)"
        power_range = BIG_CORE_POWER_RANGE
    else:
        geom_type = "Little (E-core)"
        power_range = LITTLE_CORE_POWER_RANGE

    p_min, p_max = power_range
    warning_msg = None
    if not (p_min <= init_power <= p_max):
        msg = (
            f"Ambiguity / Mismatch detected for core '{name}': floorplan geometry is {w*1e3:.1f}x{h*1e3:.1f} mm "
            f"({geom_type}, documented range {p_min}-{p_max} W), but initial steady ptrace has {init_power} W."
        )
        if not allow_geometry_fallback:
            raise ValueError(
                msg + " Stopping as required: please provide explicit configuration (--allow-geometry-power-override) "
                "rather than guessing."
            )
        warning_msg = msg + " Using geometry-based power range because override was explicitly enabled."

    return geom_type, power_range, warning_msg


def run_hotspot_simulation(
    hotspot_bin: str,
    config_file: str,
    flp_file: str,
    lcf_file: str,
    ptrace_path: str,
    steady_path: str,
    cwd_dir: str,
) -> None:
    """Execute a single HotSpot grid steady-state simulation inside WSL or native Linux."""
    if platform.system().lower().startswith("win"):
        wsl_cwd = to_wsl_path(cwd_dir)
        wsl_bin = hotspot_bin if hotspot_bin.startswith("/") or hotspot_bin.startswith("~") else to_wsl_path(hotspot_bin)
        wsl_cfg = to_wsl_path(config_file)
        wsl_ptrace = to_wsl_path(ptrace_path)
        wsl_steady = to_wsl_path(steady_path)

        bash_cmd = (
            f"cd '{wsl_cwd}' && {wsl_bin} -c '{wsl_cfg}' -f '{flp_file}' "
            f"-p '{wsl_ptrace}' -grid_layer_file '{lcf_file}' -model_type grid -steady_file '{wsl_steady}'"
        )
        cmd = ["wsl", "-e", "bash", "-c", bash_cmd]
        res = subprocess.run(cmd, capture_output=True, text=True)
    else:
        cmd = [
            os.path.expanduser(hotspot_bin),
            "-c", os.path.abspath(config_file),
            "-f", flp_file,
            "-p", os.path.abspath(ptrace_path),
            "-grid_layer_file", lcf_file,
            "-model_type", "grid",
            "-steady_file", os.path.abspath(steady_path),
        ]
        res = subprocess.run(cmd, cwd=cwd_dir, capture_output=True, text=True)

    if res.returncode != 0:
        raise RuntimeError(
            f"HotSpot simulation failed (exit code {res.returncode}) for {ptrace_path}:\n{res.stderr}\n{res.stdout}"
        )
    if not os.path.exists(steady_path) or os.path.getsize(steady_path) == 0:
        raise RuntimeError(f"HotSpot did not generate a non-empty steady file at: {steady_path}")


def run_hotspot_simulation_batch(
    hotspot_bin: str,
    config_file: str,
    flp_file: str,
    lcf_file: str,
    ptrace_steady_pairs: List[Tuple[str, str]],
    cwd_dir: str,
) -> None:
    """Execute a batch of HotSpot grid steady-state simulations in a single WSL session for low overhead."""
    if not ptrace_steady_pairs:
        return

    if platform.system().lower().startswith("win"):
        wsl_cwd = to_wsl_path(cwd_dir)
        wsl_bin = hotspot_bin if hotspot_bin.startswith("/") or hotspot_bin.startswith("~") else to_wsl_path(hotspot_bin)
        wsl_cfg = to_wsl_path(config_file)

        sh_file = os.path.join(cwd_dir, "_run_hotspot_batch.sh")
        wsl_sh = to_wsl_path(sh_file)
        with open(sh_file, "w", newline="\n") as f:
            f.write("#!/usr/bin/env bash\nset -e\n")
            f.write(f"cd '{wsl_cwd}'\n")
            for ptrace_path, steady_path in ptrace_steady_pairs:
                wsl_ptrace = to_wsl_path(ptrace_path)
                wsl_steady = to_wsl_path(steady_path)
                f.write(
                    f"{wsl_bin} -c '{wsl_cfg}' -f '{flp_file}' -p '{wsl_ptrace}' "
                    f"-grid_layer_file '{lcf_file}' -model_type grid -steady_file '{wsl_steady}' >/dev/null\n"
                )
        try:
            cmd = ["wsl", "-e", "bash", wsl_sh]
            res = subprocess.run(cmd, capture_output=True, text=True)
        finally:
            if os.path.exists(sh_file):
                os.remove(sh_file)
        if res.returncode != 0:
            raise RuntimeError(
                f"HotSpot batch simulation failed (exit code {res.returncode}) in {cwd_dir}:\n{res.stderr}\n{res.stdout}"
            )
    else:
        for ptrace_path, steady_path in ptrace_steady_pairs:
            run_hotspot_simulation(
                hotspot_bin=hotspot_bin,
                config_file=config_file,
                flp_file=flp_file,
                lcf_file=lcf_file,
                ptrace_path=ptrace_path,
                steady_path=steady_path,
                cwd_dir=cwd_dir,
            )

    for _, steady_path in ptrace_steady_pairs:
        if not os.path.exists(steady_path) or os.path.getsize(steady_path) == 0:
            raise RuntimeError(f"HotSpot did not generate a non-empty steady file at: {steady_path}")


def parse_steady_temperatures(steady_path: str, active_core_names: List[str], layer_num: str = "0") -> List[float]:
    """Extract steady-state temperatures for the active cores in order."""
    if not os.path.exists(steady_path):
        raise FileNotFoundError(f"Steady output file not found: {steady_path}")

    t_map: Dict[str, float] = {}
    with open(steady_path, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) == 2:
                t_map[parts[0]] = float(parts[1])

    temps = []
    for c_name in active_core_names:
        key = f"layer_{layer_num}_{c_name}"
        if key not in t_map:
            raise KeyError(f"Missing thermal node '{key}' in {steady_path}")
        val = t_map[key]
        if math.isnan(val) or math.isinf(val):
            raise ValueError(f"Invalid temperature ({val}) for '{key}' in {steady_path}")
        temps.append(val)
    return temps


def run_stage_a_for_floorplan(
    fp_dir: str,
    hotspot_bin: str = "~/HotSpot/hotspot",
    config_file: Optional[str] = None,
    num_train: int = 100,
    num_val: int = 20,
    train_seed: int = 42,
    val_seed: int = 2026,
    force_overwrite: bool = False,
    allow_geometry_fallback: bool = False,
) -> Dict[str, object]:
    """Execute complete Stage A calibration, validation, latency benchmarking, and reporting for a floorplan."""
    fp_dir = os.path.abspath(fp_dir)
    if not os.path.isdir(fp_dir):
        raise NotADirectoryError(f"Floorplan directory does not exist: {fp_dir}")

    fp_id = os.path.basename(os.path.normpath(fp_dir))
    style_id = os.path.basename(os.path.dirname(os.path.normpath(fp_dir)))
    full_fp_label = f"{style_id}/{fp_id}"

    # Protect existing fp_001 and fp_002 results unless explicitly forced
    gen_dir = os.path.join(fp_dir, "generated_samples")
    val_dir = os.path.join(fp_dir, "validation_samples")
    matrix_filename = f"thermal_matrix_{fp_id.replace('_', '')}.csv"
    matrix_path = os.path.join(gen_dir, matrix_filename)

    if not force_overwrite and os.path.exists(matrix_path):
        raise FileExistsError(
            f"Existing Stage A results found at {matrix_path} for {full_fp_label}. "
            f"Refusing to overwrite existing results without --force-overwrite."
        )

    if config_file is None:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        config_file = os.path.join(script_dir, "hotspot.config")
    config_file = os.path.abspath(config_file)
    if not os.path.exists(config_file):
        raise FileNotFoundError(f"HotSpot config file not found: {config_file}")

    # Locate .lcf and steady .ptrace files automatically
    lcf_candidates = glob.glob(os.path.join(fp_dir, "*.lcf"))
    if len(lcf_candidates) != 1:
        raise FileNotFoundError(f"Expected exactly 1 .lcf file in {fp_dir}, found {len(lcf_candidates)}")
    lcf_path = lcf_candidates[0]
    lcf_filename = os.path.basename(lcf_path)

    ptrace_candidates = glob.glob(os.path.join(fp_dir, "*_steady.ptrace"))
    if len(ptrace_candidates) != 1:
        raise FileNotFoundError(f"Expected exactly 1 *_steady.ptrace file in {fp_dir}, found {len(ptrace_candidates)}")
    orig_ptrace_path = ptrace_candidates[0]

    # Parse .lcf and identify power-dissipating silicon layer(s)
    layers = parse_lcf(lcf_path)
    active_layers = [lyr for lyr in layers if lyr["power_dissipation"].upper() == "Y"]
    if len(active_layers) != 1:
        raise ValueError(
            f"Stage A 2D pipeline expects 1 power-dissipating layer in {lcf_path}, found {len(active_layers)}"
        )

    active_layer = active_layers[0]
    active_layer_num = active_layer["layer_num"]
    flp_filename = active_layer["flp_file"]
    flp_path = os.path.join(fp_dir, flp_filename)

    # Verify all layer floorplan files exist
    for lyr in layers:
        lyr_flp = os.path.join(fp_dir, lyr["flp_file"])
        if not os.path.exists(lyr_flp):
            raise FileNotFoundError(f"Layer floorplan file referenced in .lcf not found: {lyr_flp}")

    # Parse .flp and steady .ptrace
    blocks = parse_flp(flp_path)
    ptrace_headers, init_power_map = parse_steady_ptrace(orig_ptrace_path)

    block_names_in_flp = [str(b["name"]) for b in blocks]
    if ptrace_headers != block_names_in_flp:
        if set(ptrace_headers) != set(block_names_in_flp):
            raise ValueError(
                f"Block names in {orig_ptrace_path} do not match {flp_path}!"
            )

    block_by_name = {str(b["name"]): b for b in blocks}

    # Preserve exact ptrace column ordering for active cores and passive FILL blocks
    active_cores: List[Dict[str, object]] = []
    fill_blocks: List[Dict[str, object]] = []

    for col_name in ptrace_headers:
        blk = block_by_name[col_name]
        init_p = init_power_map[col_name]
        if blk["is_fill"]:
            if not math.isclose(init_p, 0.0, abs_tol=1e-9):
                raise ValueError(f"Passive FILL block '{col_name}' has non-zero initial power {init_p} W!")
            fill_blocks.append(blk)
        else:
            c_type, p_range, warn = determine_core_power_range(blk, init_p, allow_geometry_fallback)
            blk_info = dict(blk)
            blk_info["core_type"] = c_type
            blk_info["power_range"] = p_range
            blk_info["init_power"] = init_p
            active_cores.append(blk_info)

    num_cores = len(active_cores)
    if num_cores == 0:
        raise ValueError(f"No active cores found in {flp_path}")

    active_core_names = [str(c["name"]) for c in active_cores]

    # Compute die dimensions from bounding box of all blocks
    die_width_m = max(float(b["left_x"]) + float(b["width"]) for b in blocks)
    die_height_m = max(float(b["bottom_y"]) + float(b["height"]) for b in blocks)

    print(f"============================================================")
    print(f"Stage A Pipeline: {full_fp_label}")
    print(f"============================================================")
    print(f"Floorplan file       : {flp_filename} ({die_width_m*1e3:.1f} mm x {die_height_m*1e3:.1f} mm)")
    print(f"Layer config file    : {lcf_filename} ({len(layers)} layers)")
    print(f"Active cores (N={num_cores}) : {', '.join(active_core_names)}")
    for c in active_cores:
        pr = c["power_range"]
        print(
            f"  - {c['name']:10s} : {c['core_type']:15s} | "
            f"{float(c['width'])*1e3:.1f}x{float(c['height'])*1e3:.1f} mm at "
            f"({float(c['left_x'])*1e3:.1f}, {float(c['bottom_y'])*1e3:.1f}) mm | "
            f"init={c['init_power']:.1f} W | range=[{pr[0]:.1f}, {pr[1]:.1f}] W"
        )
    print(f"Passive FILL blocks  : {len(fill_blocks)} ({', '.join(str(b['name']) for b in fill_blocks)})")

    # Prepare output directories
    train_ptrace_dir = os.path.join(gen_dir, "ptrace")
    train_steady_dir = os.path.join(gen_dir, "steady")
    val_ptrace_dir = os.path.join(val_dir, "ptrace")
    val_steady_dir = os.path.join(val_dir, "steady")

    for d in [train_ptrace_dir, train_steady_dir, val_ptrace_dir, val_steady_dir]:
        os.makedirs(d, exist_ok=True)

    # 7. Generate 100 unique training power configurations (including initial config when valid)
    seen_configs = set()
    train_configs: List[Tuple[int, Tuple[float, ...], bool]] = []

    init_tuple = tuple(round(float(c["init_power"]), 2) for c in active_cores)
    init_valid = all(
        c["power_range"][0] <= float(c["init_power"]) <= c["power_range"][1]
        for c in active_cores
    )
    if init_valid:
        seen_configs.add(init_tuple)
        train_configs.append((1, init_tuple, True))

    rng_train = random.Random(train_seed)
    while len(train_configs) < num_train:
        p_vals = []
        for c in active_cores:
            p_min, p_max = c["power_range"]
            p_vals.append(round(rng_train.uniform(p_min, p_max), 2))
        p_tuple = tuple(p_vals)
        if p_tuple not in seen_configs:
            seen_configs.add(p_tuple)
            train_configs.append((len(train_configs) + 1, p_tuple, False))

    # 8. Run HotSpot for all 100 training configurations
    t_sim_train_start = time.perf_counter()
    train_powers_list: List[List[float]] = []
    train_temps_list: List[List[float]] = []
    combined_csv_rows: List[List[str]] = []
    train_batch_pairs: List[Tuple[str, str]] = []

    for sample_idx, p_tuple, is_orig in train_configs:
        ptrace_path = os.path.join(train_ptrace_dir, f"sample_{sample_idx:03d}.ptrace")
        steady_path = os.path.join(train_steady_dir, f"sample_{sample_idx:03d}.steady")

        core_p_map = {active_core_names[i]: p_tuple[i] for i in range(num_cores)}
        with open(ptrace_path, "w", newline="\n") as f:
            f.write("\t".join(ptrace_headers) + "\n")
            row_strs = []
            for h in ptrace_headers:
                if h in core_p_map:
                    val = core_p_map[h]
                    row_strs.append(f"{val:.1f}" if is_orig else f"{val:.2f}")
                else:
                    row_strs.append("0.0")
            f.write("\t".join(row_strs) + "\n")

        train_batch_pairs.append((ptrace_path, steady_path))

    run_hotspot_simulation_batch(
        hotspot_bin=hotspot_bin,
        config_file=config_file,
        flp_file=flp_filename,
        lcf_file=lcf_filename,
        ptrace_steady_pairs=train_batch_pairs,
        cwd_dir=fp_dir,
    )
    print(f"  [Training] Completed all {num_train}/{num_train} HotSpot simulations...")

    for sample_idx, p_tuple, is_orig in train_configs:
        steady_path = os.path.join(train_steady_dir, f"sample_{sample_idx:03d}.steady")
        temps = parse_steady_temperatures(steady_path, active_core_names, active_layer_num)
        train_powers_list.append(list(p_tuple))
        train_temps_list.append(temps)

        p_fmt = [f"{p:.1f}" if is_orig else f"{p:.2f}" for p in p_tuple]
        t_fmt = [f"{t:.2f}" for t in temps]
        combined_csv_rows.append([f"sample_{sample_idx:03d}"] + p_fmt + t_fmt)

    t_sim_train_sec = time.perf_counter() - t_sim_train_start

    # 9. Save combined_100_samples.csv
    combined_csv_path = os.path.join(gen_dir, "combined_100_samples.csv")
    combined_headers = (
        ["sample_id"]
        + [f"{c}_power_W" for c in active_core_names]
        + [f"{c}_temp_K" for c in active_core_names]
    )
    with open(combined_csv_path, "w", newline="\n") as f:
        writer = csv.writer(f)
        writer.writerow(combined_headers)
        writer.writerows(combined_csv_rows)

    # 10. Calculate floorplan-specific N x N thermal resistance matrix R using NumPy least squares
    t_calib_start = time.perf_counter()
    P_train = np.array(train_powers_list, dtype=np.float64).T  # shape (N, 100)
    T_train = np.array(train_temps_list, dtype=np.float64).T   # shape (N, 100)
    Delta_T_train = T_train - AMBIENT_TEMP_K                   # shape (N, 100)

    rank_P = int(np.linalg.matrix_rank(P_train))
    cond_P = float(np.linalg.cond(P_train))
    if rank_P < num_cores:
        raise RuntimeError(
            f"Rank-deficient power matrix P: rank={rank_P} < N={num_cores}. Cannot reliably estimate R."
        )

    R_T, _, _, _ = np.linalg.lstsq(P_train.T, Delta_T_train.T, rcond=None)
    R = R_T.T  # shape (N, N)
    t_calib_sec = time.perf_counter() - t_calib_start

    if R.shape != (num_cores, num_cores):
        raise RuntimeError(f"Expected R shape ({num_cores}, {num_cores}), got {R.shape}")

    # Save thermal_matrix_<floorplan_id>.csv
    with open(matrix_path, "w", newline="\n") as f:
        writer = csv.writer(f)
        writer.writerow(["target_core \\ source_core"] + active_core_names)
        for i, row_vals in enumerate(R):
            writer.writerow([active_core_names[i]] + [f"{val:.6f}" for val in row_vals])

    # 11. Calculate training RMSE, MAE, and Max Error
    T_train_pred = AMBIENT_TEMP_K + (R @ P_train)
    train_abs_errors = np.abs(T_train - T_train_pred)
    train_rmse = float(np.sqrt(np.mean((T_train - T_train_pred) ** 2)))
    train_mae = float(np.mean(train_abs_errors))
    train_max_err = float(np.max(train_abs_errors))

    # 12. Generate 20 new validation power configurations with zero overlap with training set
    rng_val = random.Random(val_seed)
    val_configs: List[Tuple[int, Tuple[float, ...]]] = []
    while len(val_configs) < num_val:
        p_vals = []
        for c in active_cores:
            p_min, p_max = c["power_range"]
            p_vals.append(round(rng_val.uniform(p_min, p_max), 2))
        p_tuple = tuple(p_vals)
        if p_tuple not in seen_configs:
            seen_configs.add(p_tuple)
            val_configs.append((len(val_configs) + 1, p_tuple))

    # 13. Run HotSpot for the 20 validation configurations
    t_sim_val_start = time.perf_counter()
    val_powers_list: List[List[float]] = []
    val_temps_list: List[List[float]] = []
    val_batch_pairs: List[Tuple[str, str]] = []

    for val_idx, p_tuple in val_configs:
        sample_id = f"val_sample_{val_idx:03d}"
        ptrace_path = os.path.join(val_ptrace_dir, f"{sample_id}.ptrace")
        steady_path = os.path.join(val_steady_dir, f"{sample_id}.steady")

        core_p_map = {active_core_names[i]: p_tuple[i] for i in range(num_cores)}
        with open(ptrace_path, "w", newline="\n") as f:
            f.write("\t".join(ptrace_headers) + "\n")
            row_strs = [f"{core_p_map[h]:.2f}" if h in core_p_map else "0.0" for h in ptrace_headers]
            f.write("\t".join(row_strs) + "\n")

        val_batch_pairs.append((ptrace_path, steady_path))

    run_hotspot_simulation_batch(
        hotspot_bin=hotspot_bin,
        config_file=config_file,
        flp_file=flp_filename,
        lcf_file=lcf_filename,
        ptrace_steady_pairs=val_batch_pairs,
        cwd_dir=fp_dir,
    )
    print(f"  [Validation] Completed all {num_val}/{num_val} HotSpot simulations...")

    for val_idx, p_tuple in val_configs:
        sample_id = f"val_sample_{val_idx:03d}"
        steady_path = os.path.join(val_steady_dir, f"{sample_id}.steady")
        temps = parse_steady_temperatures(steady_path, active_core_names, active_layer_num)
        val_powers_list.append(list(p_tuple))
        val_temps_list.append(temps)

    t_sim_val_sec = time.perf_counter() - t_sim_val_start

    # 14. Load the saved matrix from disk without recalibrating it
    R_loaded_rows = []
    with open(matrix_path, "r", newline="") as f:
        reader = csv.reader(f)
        next(reader)  # skip header
        for row in reader:
            R_loaded_rows.append([float(x) for x in row[1 : num_cores + 1]])
    R_loaded = np.array(R_loaded_rows, dtype=np.float64)

    P_val = np.array(val_powers_list, dtype=np.float64).T  # shape (N, 20)
    T_val_actual = np.array(val_temps_list, dtype=np.float64).T  # shape (N, 20)

    # Predict validation temperatures
    T_val_pred = AMBIENT_TEMP_K + (R_loaded @ P_val)

    # 15. Calculate validation RMSE, MAE, and maximum absolute error
    val_abs_errors = np.abs(T_val_actual - T_val_pred)
    val_rmse = float(np.sqrt(np.mean((T_val_actual - T_val_pred) ** 2)))
    val_mae = float(np.mean(val_abs_errors))
    val_max_err = float(np.max(val_abs_errors))

    # Save validation comparisons CSV
    total_val_comparisons = num_val * num_cores
    val_comp_csv_path = os.path.join(val_dir, f"validation_comparisons_{total_val_comparisons}.csv")
    val_comp_rows = []
    for s_idx in range(num_val):
        sample_id = f"val_sample_{s_idx + 1:03d}"
        for c_idx in range(num_cores):
            act_v = float(T_val_actual[c_idx, s_idx])
            pred_v = float(T_val_pred[c_idx, s_idx])
            err_v = abs(act_v - pred_v)
            val_comp_rows.append([
                sample_id,
                active_core_names[c_idx],
                f"{act_v:.2f}",
                f"{pred_v:.6f}",
                f"{err_v:.6f}",
            ])

    with open(val_comp_csv_path, "w", newline="\n") as f:
        writer = csv.writer(f)
        writer.writerow([
            "Sample_ID",
            "Core_ID",
            "Actual_Temperature_K",
            "Predicted_Temperature_K",
            "Absolute_Error_K",
        ])
        writer.writerows(val_comp_rows)

    # 16. Measure prediction execution time separately using repeated in-memory predictions with time.perf_counter()
    num_bench_iters = 10000
    single_P_vec = P_val[:, 0:1].copy()  # shape (N, 1)
    # Warmup
    for _ in range(100):
        _ = AMBIENT_TEMP_K + (R_loaded @ single_P_vec)

    t_bench_start = time.perf_counter()
    for _ in range(num_bench_iters):
        _ = AMBIENT_TEMP_K + (R_loaded @ single_P_vec)
    t_bench_total_sec = time.perf_counter() - t_bench_start
    avg_pred_latency_sec = t_bench_total_sec / num_bench_iters
    avg_pred_latency_us = avg_pred_latency_sec * 1e6

    # 17. Measure model size (saved CSV file size in bytes and raw float64 size N * N * 8 bytes)
    csv_file_size_bytes = os.path.getsize(matrix_path)
    raw_float64_size_bytes = num_cores * num_cores * 8

    # 18 & 19. Save validation summary report and floorplan summary CSV
    val_report_path = os.path.join(val_dir, "validation_summary_report.txt")
    report_text = (
        "============================================================\n"
        f"Stage A Calibration & Validation Summary Report ({full_fp_label})\n"
        "============================================================\n"
        f"Floorplan Directory      : {fp_dir}\n"
        f"Die Dimensions           : {die_width_m*1e3:.2f} mm x {die_height_m*1e3:.2f} mm\n"
        f"Active Cores Detected    : {num_cores} ({', '.join(active_core_names)})\n"
        f"Passive FILL Blocks      : {len(fill_blocks)}\n"
        f"Thermal Matrix File      : {matrix_path}\n"
        f"Matrix Dimensions        : {num_cores}x{num_cores} (Units: K/W)\n"
        f"Power Matrix Rank        : {rank_P} (Full Rank: {rank_P == num_cores})\n"
        f"Power Matrix Cond Number : {cond_P:.4f}\n"
        "------------------------------------------------------------\n"
        f"Training Set ({num_train} samples, {num_train * num_cores} comparisons):\n"
        f"  - Training RMSE        : {train_rmse:.6f} K\n"
        f"  - Training MAE         : {train_mae:.6f} K\n"
        f"  - Training Max Error   : {train_max_err:.6f} K\n"
        "------------------------------------------------------------\n"
        f"Validation Set ({num_val} samples, {total_val_comparisons} comparisons, 0 overlap):\n"
        f"  - Validation RMSE      : {val_rmse:.6f} K\n"
        f"  - Validation MAE       : {val_mae:.6f} K\n"
        f"  - Validation Max Error : {val_max_err:.6f} K\n"
        "------------------------------------------------------------\n"
        "Runtime & Model Footprint:\n"
        f"  - Avg Prediction Time  : {avg_pred_latency_us:.4f} us ({avg_pred_latency_sec:.6e} s)\n"
        f"  - Training Sim Time    : {t_sim_train_sec:.2f} s ({t_sim_train_sec/num_train:.3f} s/run)\n"
        f"  - Validation Sim Time  : {t_sim_val_sec:.2f} s ({t_sim_val_sec/num_val:.3f} s/run)\n"
        f"  - Least-Squares Fit    : {t_calib_sec*1e3:.4f} ms\n"
        f"  - Saved CSV Model Size : {csv_file_size_bytes} bytes\n"
        f"  - Raw float64 Size     : {raw_float64_size_bytes} bytes ({num_cores}x{num_cores}x8)\n"
        "============================================================\n"
    )
    with open(val_report_path, "w", newline="\n") as f:
        f.write(report_text)

    summary_csv_path = os.path.join(fp_dir, f"stage_a_summary_{fp_id}.csv")
    summary_headers = [
        "floorplan_name",
        "num_cores",
        "matrix_dimensions",
        "power_matrix_rank",
        "power_matrix_cond",
        "train_samples",
        "val_samples",
        "train_rmse_K",
        "train_mae_K",
        "train_max_error_K",
        "val_rmse_K",
        "val_mae_K",
        "val_max_error_K",
        "avg_prediction_time_us",
        "avg_prediction_time_s",
        "saved_matrix_csv_bytes",
        "raw_float64_bytes",
    ]
    summary_row = [
        full_fp_label,
        num_cores,
        f"{num_cores}x{num_cores}",
        rank_P,
        f"{cond_P:.4f}",
        num_train,
        num_val,
        f"{train_rmse:.6f}",
        f"{train_mae:.6f}",
        f"{train_max_err:.6f}",
        f"{val_rmse:.6f}",
        f"{val_mae:.6f}",
        f"{val_max_err:.6f}",
        f"{avg_pred_latency_us:.4f}",
        f"{avg_pred_latency_sec:.6e}",
        csv_file_size_bytes,
        raw_float64_size_bytes,
    ]
    with open(summary_csv_path, "w", newline="\n") as f:
        writer = csv.writer(f)
        writer.writerow(summary_headers)
        writer.writerow(summary_row)

    print(report_text)
    print("Thermal Resistance Matrix R (K/W):")
    hdr = "          " + " ".join([f"{c:>10s}" for c in active_core_names])
    print(hdr)
    for i, r_vals in enumerate(R_loaded):
        print(f"{active_core_names[i]:10s}" + " ".join([f"{v:10.6f}" for v in r_vals]))

    return {
        "floorplan_name": full_fp_label,
        "num_cores": num_cores,
        "matrix_dimensions": f"{num_cores}x{num_cores}",
        "rank": rank_P,
        "cond": cond_P,
        "train_rmse": train_rmse,
        "train_mae": train_mae,
        "train_max_err": train_max_err,
        "val_rmse": val_rmse,
        "val_mae": val_mae,
        "val_max_err": val_max_err,
        "avg_pred_latency_us": avg_pred_latency_us,
        "csv_file_size_bytes": csv_file_size_bytes,
        "raw_float64_size_bytes": raw_float64_size_bytes,
        "summary_csv_path": summary_csv_path,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Stage A Calibration & Validation Pipeline for 2D Floorplans."
    )
    parser.add_argument(
        "floorplan_dir",
        help="Path to the target 2D floorplan directory (e.g. dataset/Sample_dataset/2D_floorplans/Style_1/fp_003)",
    )
    parser.add_argument(
        "--hotspot-bin",
        default="~/HotSpot/hotspot",
        help="Path to HotSpot executable inside WSL/Linux (default: ~/HotSpot/hotspot)",
    )
    parser.add_argument(
        "--config-file",
        default=None,
        help="Path to hotspot.config (defaults to ~/temperature_project/hotspot.config)",
    )
    parser.add_argument(
        "--num-train",
        type=int,
        default=100,
        help="Number of training power configurations (default: 100)",
    )
    parser.add_argument(
        "--num-val",
        type=int,
        default=20,
        help="Number of validation power configurations (default: 20)",
    )
    parser.add_argument(
        "--train-seed",
        type=int,
        default=42,
        help="Random seed for training power generation (default: 42)",
    )
    parser.add_argument(
        "--val-seed",
        type=int,
        default=2026,
        help="Random seed for validation power generation (default: 2026)",
    )
    parser.add_argument(
        "--force-overwrite",
        action="store_true",
        help="Allow overwriting existing Stage A outputs in the target floorplan folder.",
    )
    parser.add_argument(
        "--allow-geometry-power-override",
        action="store_true",
        help="Use floorplan geometry to assign Big/Little power ranges when initial ptrace values mismatch.",
    )

    args = parser.parse_args()
    try:
        run_stage_a_for_floorplan(
            fp_dir=args.floorplan_dir,
            hotspot_bin=args.hotspot_bin,
            config_file=args.config_file,
            num_train=args.num_train,
            num_val=args.num_val,
            train_seed=args.train_seed,
            val_seed=args.val_seed,
            force_overwrite=args.force_overwrite,
            allow_geometry_fallback=args.allow_geometry_power_override,
        )
    except Exception as exc:
        print(f"\n[STAGE A PIPELINE ERROR] {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
