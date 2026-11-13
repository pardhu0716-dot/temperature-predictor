#!/usr/bin/env python3
"""
Stage A CLI Temperature Predictor
Lightweight Temperature Predictor Project

Predicts steady-state core temperatures for any of the 23 calibrated 2D floorplans
across Style_1, Style_2, Style_3, and Style_4 using each floorplan's pre-calibrated
8x8 thermal resistance matrix R:

    Delta_T = R x P
    T_K     = 318.15 + Delta_T   (Kelvin)
    T_C     = T_K - 273.15       (Celsius)

Normal prediction NEVER invokes HotSpot, regenerates training samples, or recalibrates R.
"""

import argparse
import csv
import glob
import math
import os
import sys
import time
from typing import Dict, List, Optional, Tuple

try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False

AMBIENT_TEMP_K = 318.15
KELVIN_OFFSET = 273.15
AMBIENT_TEMP_C = AMBIENT_TEMP_K - KELVIN_OFFSET  # 45.00 °C


def discover_floorplans(floorplans_root: str) -> List[Dict[str, str]]:
    """Discover all available 2D floorplans sorted by Style_1..Style_4 and fp_001..fp_010."""
    styles = ["Style_1", "Style_2", "Style_3", "Style_4"]
    catalog: List[Dict[str, str]] = []
    idx = 1
    for style in styles:
        style_dir = os.path.join(floorplans_root, style)
        if not os.path.isdir(style_dir):
            continue
        fp_dirs = sorted(
            [
                d
                for d in os.listdir(style_dir)
                if os.path.isdir(os.path.join(style_dir, d)) and d.startswith("fp_")
            ],
            key=lambda x: int(x.split("_")[1]) if "_" in x and x.split("_")[1].isdigit() else x,
        )
        for fp_id in fp_dirs:
            fp_path = os.path.join(style_dir, fp_id)
            catalog.append(
                {
                    "index": str(idx),
                    "style": style,
                    "floorplan": fp_id,
                    "full_name": f"{style}/{fp_id}",
                    "path": fp_path,
                }
            )
            idx += 1
    return catalog


def parse_lcf_active_flp(fp_dir: str) -> str:
    """Locate the active silicon layer .flp file from the floorplan's .lcf file (or fallback to *_layer_0.flp)."""
    lcf_files = sorted(glob.glob(os.path.join(fp_dir, "*.lcf")))
    if lcf_files:
        lcf_path = lcf_files[0]
        active_lines: List[str] = []
        with open(lcf_path, "r", encoding="utf-8") as f:
            for raw in f:
                line = raw.strip()
                if not line or line.startswith("#"):
                    continue
                active_lines.append(line)
        # Each layer block in HotSpot .lcf has 7 lines:
        # 0: layer_num, 1: lateral_heat_flow, 2: power_dissipation (Y/N),
        # 3: thickness, 4: conductivity, 5: specific_heat, 6: flp_file
        if len(active_lines) >= 7 and len(active_lines) % 7 == 0:
            for i in range(0, len(active_lines), 7):
                if active_lines[i + 2].upper() == "Y":
                    candidate = os.path.join(fp_dir, active_lines[i + 6])
                    if os.path.isfile(candidate):
                        return candidate

    flp_candidates = sorted(glob.glob(os.path.join(fp_dir, "*_layer_0.flp")))
    if flp_candidates:
        return flp_candidates[0]
    raise FileNotFoundError(f"Could not find active silicon layer .flp file in {fp_dir}")


def parse_flp_cores(flp_path: str) -> Tuple[Dict[str, Dict[str, object]], Tuple[float, float], int]:
    """
    Read the floorplan .flp file to identify all active cores, their exact names,
    physical geometry (width, height, position), and Big/Little classification.

    Returns:
      - core_map: dict mapping core_name -> metadata dict
      - die_size_mm: (die_width_mm, die_height_mm)
      - num_fill_blocks: count of passive FILL_* blocks
    """
    core_map: Dict[str, Dict[str, object]] = {}
    max_x_m = 0.0
    max_y_m = 0.0
    num_fill = 0

    with open(flp_path, "r", encoding="utf-8") as f:
        for raw in f:
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) < 5:
                continue
            name = parts[0]
            width_m = float(parts[1])
            height_m = float(parts[2])
            left_x_m = float(parts[3])
            bottom_y_m = float(parts[4])

            max_x_m = max(max_x_m, left_x_m + width_m)
            max_y_m = max(max_y_m, bottom_y_m + height_m)

            if name.upper().startswith("FILL"):
                num_fill += 1
                continue

            w_mm = width_m * 1e3
            h_mm = height_m * 1e3
            area_mm2 = w_mm * h_mm

            # Geometry-based core classification per dataset specification:
            # - 4 mm x 4 mm (16 mm^2): Big core, power range 10-20 W
            # - 2 mm x 2 mm (4 mm^2): Little core, power range 5-10 W
            if math.isclose(w_mm, 4.0, rel_tol=1e-2) and math.isclose(h_mm, 4.0, rel_tol=1e-2):
                core_type = "Big (4x4 mm)"
                short_type = "Big"
                power_range = (10.0, 20.0)
            elif math.isclose(w_mm, 2.0, rel_tol=1e-2) and math.isclose(h_mm, 2.0, rel_tol=1e-2):
                core_type = "Little (2x2 mm)"
                short_type = "Little"
                power_range = (5.0, 10.0)
            elif area_mm2 >= 10.0:
                core_type = f"Big ({w_mm:.1f}x{h_mm:.1f} mm)"
                short_type = "Big"
                power_range = (10.0, 20.0)
            else:
                core_type = f"Little ({w_mm:.1f}x{h_mm:.1f} mm)"
                short_type = "Little"
                power_range = (5.0, 10.0)

            core_map[name] = {
                "name": name,
                "width_mm": w_mm,
                "height_mm": h_mm,
                "left_x_mm": left_x_m * 1e3,
                "bottom_y_mm": bottom_y_m * 1e3,
                "area_mm2": area_mm2,
                "core_type": core_type,
                "short_type": short_type,
                "power_range": power_range,
            }

    return core_map, (max_x_m * 1e3, max_y_m * 1e3), num_fill


def locate_and_load_thermal_matrix(
    fp_dir: str, fp_id: str
) -> Tuple[str, List[str], List[List[float]], int, int]:
    """
    Locate and load the pre-calibrated N x N thermal resistance matrix R (in K/W)
    from <fp_dir>/generated_samples/.

    Returns:
      - matrix_path: path to the loaded CSV file
      - col_core_names: list of core names in exact matrix column order
      - R_rows: N x N nested float list
      - csv_size_bytes: file size on disk in bytes
      - raw_float64_bytes: raw float64 parameter storage size (N * N * 8 bytes)
    """
    gen_dir = os.path.join(fp_dir, "generated_samples")
    expected_name = f"thermal_matrix_{fp_id.replace('_', '')}.csv"
    expected_path = os.path.join(gen_dir, expected_name)

    if os.path.isfile(expected_path):
        matrix_path = expected_path
    else:
        candidates = sorted(glob.glob(os.path.join(gen_dir, "thermal_matrix_*.csv")))
        if not candidates:
            raise FileNotFoundError(
                f"No calibrated thermal resistance matrix found in {gen_dir}"
            )
        matrix_path = candidates[0]

    csv_size_bytes = os.path.getsize(matrix_path)

    with open(matrix_path, "r", encoding="utf-8", newline="") as f:
        reader = list(csv.reader(f))

    if len(reader) < 2:
        raise ValueError(f"Thermal matrix file {matrix_path} is empty or malformed.")

    col_core_names = [c.strip() for c in reader[0][1:] if c.strip()]
    num_cores = len(col_core_names)
    if num_cores == 0:
        raise ValueError(f"No core columns found in header of {matrix_path}")

    row_core_names: List[str] = []
    R_rows: List[List[float]] = []

    for row in reader[1:]:
        if not row or not any(cell.strip() for cell in row):
            continue
        row_label = row[0].strip()
        row_core_names.append(row_label)
        vals = [float(x) for x in row[1 : num_cores + 1]]
        if len(vals) != num_cores:
            raise ValueError(
                f"Row '{row_label}' in {matrix_path} has {len(vals)} values, expected {num_cores}."
            )
        R_rows.append(vals)

    if len(R_rows) != num_cores:
        raise ValueError(
            f"Thermal matrix in {matrix_path} has {len(R_rows)} rows, expected {num_cores}."
        )

    if row_core_names != col_core_names:
        raise ValueError(
            f"Row core order {row_core_names} does not match column core order {col_core_names} in {matrix_path}."
        )

    raw_float64_bytes = num_cores * num_cores * 8
    return matrix_path, col_core_names, R_rows, csv_size_bytes, raw_float64_bytes


def print_floorplan_catalog(catalog: List[Dict[str, str]]) -> None:
    """Display all 23 available 2D floorplans grouped by style."""
    print("=" * 74)
    print("  STAGE A LIGHTWEIGHT TEMPERATURE PREDICTOR -- AVAILABLE 2D FLOORPLANS")
    print("=" * 74)
    by_style: Dict[str, List[Dict[str, str]]] = {}
    for item in catalog:
        by_style.setdefault(item["style"], []).append(item)

    for style, items in by_style.items():
        fp_labels = [f"[{it['index']:>2s}] {it['floorplan']}" for it in items]
        # Print in rows of up to 5 floorplans per line for readability
        print(f"  {style}:")
        for i in range(0, len(fp_labels), 5):
            chunk = "   ".join(f"{lbl:<12s}" for lbl in fp_labels[i : i + 5])
            print(f"    {chunk}")
    print("-" * 74)


def resolve_floorplan_selection(
    user_input: str, catalog: List[Dict[str, str]]
) -> Optional[Dict[str, str]]:
    """
    Resolve a user floorplan selection string:
      - Index number '1'..'23'
      - Full name 'Style_1/fp_001' (case-insensitive, '/' or '\\' or space separated)
      - Style name 'Style_1'..'Style_4' or '1'..'4' (handled by caller if style-only)
    """
    token = user_input.strip()
    if not token:
        return None

    # Check direct catalog index (1..23)
    for item in catalog:
        if token == item["index"]:
            return item

    # Normalize separators
    norm = token.replace("\\", "/").replace(" ", "/")
    for item in catalog:
        if norm.lower() == item["full_name"].lower():
            return item

    # Support shorthand like s1/fp_001 or style1/fp001
    if "/" in norm:
        parts = [p for p in norm.split("/") if p]
        if len(parts) == 2:
            st_part, fp_part = parts[0].lower(), parts[1].lower()
            if st_part.isdigit():
                st_part = f"style_{int(st_part)}"
            elif st_part.startswith("style") and not st_part.startswith("style_"):
                st_part = "style_" + st_part[5:]
            if fp_part.isdigit():
                fp_part = f"fp_{int(fp_part):03d}"
            elif fp_part.startswith("fp") and not fp_part.startswith("fp_"):
                 suffix = fp_part[2:]
                 if suffix.isdigit():
                     fp_part = f"fp_{int(suffix):03d}"
            candidate_full = f"{st_part}/{fp_part}"
            for item in catalog:
                if item["full_name"].lower() == candidate_full:
                    return item

    return None


def interactive_select_floorplan(catalog: List[Dict[str, str]]) -> Dict[str, str]:
    """Prompt the user to select one of the 23 floorplans interactively."""
    print_floorplan_catalog(catalog)
    while True:
        try:
            raw = input(
                "Select a floorplan by number [1-23] or name (e.g. Style_1/fp_001, or Style_1): "
            ).strip()
        except EOFError:
            print("\nERROR: Unexpected end of input while selecting floorplan.", file=sys.stderr)
            sys.exit(1)

        if not raw:
            print("  [Invalid Input] Please enter a floorplan number (1-23) or name (e.g. Style_1/fp_001).")
            continue

        # Check if the user entered a style name like 'Style_1' .. 'Style_4'
        norm_style = raw.strip().lower().replace(" ", "_")
        if norm_style in ("style1", "style2", "style3", "style4"):
            norm_style = "style_" + norm_style[-1]
        style_matches = [it for it in catalog if it["style"].lower() == norm_style]
        if style_matches:
            valid_fps = ", ".join(it["floorplan"] for it in style_matches)
            try:
                fp_raw = input(
                    f"  Selected {style_matches[0]['style']}. Enter floorplan ({valid_fps}): "
                ).strip()
            except EOFError:
                print("\nERROR: Unexpected end of input while selecting floorplan.", file=sys.stderr)
                sys.exit(1)
            combined = f"{style_matches[0]['style']}/{fp_raw}"
            resolved = resolve_floorplan_selection(combined, catalog)
            if resolved is not None:
                return resolved
            print(
                f"  [Invalid Input] '{fp_raw}' is not a valid floorplan in {style_matches[0]['style']}. "
                f"Valid choices: {valid_fps}."
            )
            continue

        resolved = resolve_floorplan_selection(raw, catalog)
        if resolved is not None:
            return resolved

        print(
            f"  [Invalid Input] '{raw}' is not a valid floorplan selection. "
            f"Enter an index from 1 to {len(catalog)} or '<Style_X>/<fp_YYY>' (e.g. Style_1/fp_001)."
        )


def validate_single_power_value(
    raw_val: str, core_name: str, core_info: Dict[str, object]
) -> Tuple[Optional[float], Optional[str]]:
    """
    Validate a single core power string against numeric format and geometry-based range:
      - Big cores (4x4 mm): 10.0 to 20.0 W
      - Little cores (2x2 mm): 5.0 to 10.0 W
    Returns (value, None) if valid, or (None, error_message) if invalid.
    """
    token = raw_val.strip()
    if not token:
        return None, f"Empty input for {core_name}. Please enter a numeric power value in Watts."

    try:
        val = float(token)
    except ValueError:
        return (
            None,
            f"Invalid non-numeric power '{token}' for {core_name} ({core_info['core_type']}). "
            f"Please enter a number between {core_info['power_range'][0]:.1f} and {core_info['power_range'][1]:.1f} W.",
        )

    if math.isnan(val) or math.isinf(val):
        return (
            None,
            f"Invalid power '{token}' for {core_name}. Power must be a finite number.",
        )

    p_min, p_max = core_info["power_range"]
    # Allow tiny floating-point representation tolerance (1e-9) at boundaries
    if val < p_min - 1e-9 or val > p_max + 1e-9:
        return (
            None,
            f"Out-of-range power {val:g} W for {core_name} ({core_info['core_type']}). "
            f"Allowed range for {core_info['short_type']} cores is {p_min:.1f}-{p_max:.1f} W.",
        )

    return val, None


def interactive_prompt_powers(
    col_core_names: List[str], core_map: Dict[str, Dict[str, object]]
) -> List[float]:
    """Prompt the user to enter power values for all 8 cores in exact matrix column order."""
    print("\nEnter steady-state power (W) for each core in matrix column order:")
    print("  Policy: Big cores (4x4 mm) = 10.0-20.0 W | Little cores (2x2 mm) = 5.0-10.0 W")
    print("-" * 74)

    powers: List[float] = []
    for idx, core_name in enumerate(col_core_names, start=1):
        c_info = core_map[core_name]
        p_min, p_max = c_info["power_range"]
        prompt_str = (
            f"  [{idx}/{len(col_core_names)}] {core_name:<10s} | {c_info['core_type']:<16s} "
            f"| Pos ({c_info['left_x_mm']:4.1f}, {c_info['bottom_y_mm']:4.1f}) mm "
            f"| Range [{p_min:.1f}-{p_max:.1f} W]: "
        )
        while True:
            try:
                raw_val = input(prompt_str)
            except EOFError:
                print(f"\nERROR: Unexpected end of input while reading power for {core_name}.", file=sys.stderr)
                sys.exit(1)

            val, err = validate_single_power_value(raw_val, core_name, c_info)
            if err is not None:
                print(f"    [Invalid Input] {err}")
                continue
            powers.append(float(val))
            break

    return powers


def parse_powers_from_cli_or_ptrace(
    powers_arg: Optional[str],
    ptrace_arg: Optional[str],
    col_core_names: List[str],
    core_map: Dict[str, Dict[str, object]],
) -> List[float]:
    """Parse and validate core powers supplied via --powers or --ptrace."""
    if ptrace_arg is not None:
        if not os.path.isfile(ptrace_arg):
            raise FileNotFoundError(f"Power trace file not found: {ptrace_arg}")
        with open(ptrace_arg, "r", encoding="utf-8") as f:
            lines = [ln.strip() for ln in f if ln.strip()]
        if len(lines) < 2:
            raise ValueError(f"Power trace file {ptrace_arg} must have a header line and a values line.")
        headers = lines[0].split()
        vals = lines[1].split()
        if len(headers) != len(vals):
            raise ValueError(f"Header/value column count mismatch in {ptrace_arg}.")
        p_lookup = dict(zip(headers, vals))
        powers: List[float] = []
        for core_name in col_core_names:
            if core_name not in p_lookup:
                raise ValueError(f"Core '{core_name}' missing from power trace {ptrace_arg}.")
            val, err = validate_single_power_value(p_lookup[core_name], core_name, core_map[core_name])
            if err is not None:
                raise ValueError(err)
            powers.append(float(val))
        return powers

    assert powers_arg is not None
    raw_parts = [p.strip() for p in powers_arg.replace(";", ",").split(",") if p.strip()]
    if len(raw_parts) == 1 and " " in raw_parts[0]:
        raw_parts = [p.strip() for p in raw_parts[0].split() if p.strip()]

    if len(raw_parts) != len(col_core_names):
        raise ValueError(
            f"Expected {len(col_core_names)} power values in matrix column order "
            f"({', '.join(col_core_names)}), but received {len(raw_parts)}."
        )

    powers = []
    for raw_val, core_name in zip(raw_parts, col_core_names):
        val, err = validate_single_power_value(raw_val, core_name, core_map[core_name])
        if err is not None:
            raise ValueError(err)
        powers.append(float(val))
    return powers


def predict_temperatures(
    R_rows: List[List[float]], powers: List[float]
) -> Tuple[List[float], List[float], List[float], float]:
    """
    Compute steady-state core temperatures:
        Delta_T = R x P
        T_K     = 318.15 + Delta_T
        T_C     = T_K - 273.15

    Measures in-memory prediction latency in microseconds using time.perf_counter().
    """
    if HAS_NUMPY:
        R_mat = np.array(R_rows, dtype=np.float64)
        P_vec = np.array(powers, dtype=np.float64)
        # Single warmup call to avoid measuring one-time BLAS dispatch init
        _ = AMBIENT_TEMP_K + (R_mat @ P_vec) - KELVIN_OFFSET
        t0 = time.perf_counter()
        delta_t_arr = R_mat @ P_vec
        t_k_arr = AMBIENT_TEMP_K + delta_t_arr
        t_c_arr = t_k_arr - KELVIN_OFFSET
        elapsed_us = (time.perf_counter() - t0) * 1e6
        return (
            [float(x) for x in delta_t_arr],
            [float(x) for x in t_k_arr],
            [float(x) for x in t_c_arr],
            elapsed_us,
        )
    else:
        # Single warmup call
        _ = [
            (AMBIENT_TEMP_K + sum(r_ij * p_j for r_ij, p_j in zip(row, powers))) - KELVIN_OFFSET
            for row in R_rows
        ]
        t0 = time.perf_counter()
        delta_t = [
            sum(r_ij * p_j for r_ij, p_j in zip(row, powers))
            for row in R_rows
        ]
        t_k = [AMBIENT_TEMP_K + dt for dt in delta_t]
        t_c = [tk - KELVIN_OFFSET for tk in t_k]
        elapsed_us = (time.perf_counter() - t0) * 1e6
        return delta_t, t_k, t_c, elapsed_us


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="predict.py",
        description=(
            "Stage A CLI Temperature Predictor -- Predicts steady-state core temperatures "
            "for any of the 23 calibrated 2D floorplans using T = 318.15 + R x P (K)."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  1. Interactive mode (menu selection + prompted core powers):\n"
            "     python3 predict.py\n\n"
            "  2. List all 23 available floorplans and exit:\n"
            "     python3 predict.py --list\n\n"
            "  3. Select floorplan via flag and enter powers interactively:\n"
            "     python3 predict.py --floorplan Style_1/fp_001\n\n"
            "  4. Non-interactive prediction with comma-separated powers (in matrix column order):\n"
            "     python3 predict.py --floorplan Style_1/fp_001 --powers \"16.39,10.25,6.38,6.12,8.68,8.38,9.46,5.43\"\n\n"
            "  5. Non-interactive prediction from a .ptrace file:\n"
            "     python3 predict.py --floorplan Style_3/fp_001 --ptrace dataset/Sample_dataset/2D_floorplans/Style_3/fp_001/validation_samples/ptrace/val_sample_001.ptrace\n"
        ),
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="Display all 23 available 2D floorplans across Style_1..Style_4 and exit.",
    )
    parser.add_argument(
        "-f",
        "--floorplan",
        type=str,
        default=None,
        help="Floorplan identifier by index (1-23) or name (e.g., Style_1/fp_001).",
    )
    parser.add_argument(
        "-p",
        "--powers",
        type=str,
        default=None,
        help="Comma- or space-separated core power values (W) in exact matrix column order.",
    )
    parser.add_argument(
        "--ptrace",
        type=str,
        default=None,
        help="Path to a .ptrace file from which to read core power values.",
    )
    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()

    project_root = os.path.dirname(os.path.abspath(__file__))
    floorplans_root = os.path.join(project_root, "dataset", "Sample_dataset", "2D_floorplans")

    if not os.path.isdir(floorplans_root):
        print(f"ERROR: Floorplans directory not found at {floorplans_root}", file=sys.stderr)
        sys.exit(1)

    catalog = discover_floorplans(floorplans_root)
    if not catalog:
        print(f"ERROR: No floorplans discovered under {floorplans_root}", file=sys.stderr)
        sys.exit(1)

    if args.list:
        print_floorplan_catalog(catalog)
        return

    # 1. Select floorplan (interactive menu if --floorplan is omitted, or display catalog + resolve)
    if args.floorplan is not None:
        print_floorplan_catalog(catalog)
        selected = resolve_floorplan_selection(args.floorplan, catalog)
        if selected is None:
            print(
                f"ERROR: Invalid floorplan '{args.floorplan}'. "
                f"Use --list to see all {len(catalog)} available floorplans.",
                file=sys.stderr,
            )
            sys.exit(1)
    else:
        selected = interactive_select_floorplan(catalog)

    fp_dir = selected["path"]
    fp_id = selected["floorplan"]
    full_name = selected["full_name"]

    # 2. Load pre-calibrated thermal resistance matrix R
    try:
        matrix_path, col_core_names, R_rows, csv_size_bytes, raw_float64_bytes = (
            locate_and_load_thermal_matrix(fp_dir, fp_id)
        )
    except Exception as exc:
        print(f"ERROR loading thermal matrix for {full_name}: {exc}", file=sys.stderr)
        sys.exit(1)

    # 3. Read floorplan .flp to get core geometry and Big/Little classification
    try:
        flp_path = parse_lcf_active_flp(fp_dir)
        core_map, (die_w_mm, die_h_mm), num_fill = parse_flp_cores(flp_path)
    except Exception as exc:
        print(f"ERROR reading floorplan .flp for {full_name}: {exc}", file=sys.stderr)
        sys.exit(1)

    # Verify all matrix cores exist in .flp
    for c_name in col_core_names:
        if c_name not in core_map:
            print(
                f"ERROR: Matrix core '{c_name}' not found in floorplan file {flp_path}",
                file=sys.stderr,
            )
            sys.exit(1)

    num_cores = len(col_core_names)
    num_big = sum(1 for c in col_core_names if core_map[c]["short_type"] == "Big")
    num_little = num_cores - num_big

    print("\n" + "=" * 74)
    print(f"  SELECTED FLOORPLAN : {full_name} (Index #{selected['index']})")
    print("=" * 74)
    print(f"  Floorplan File     : {os.path.basename(flp_path)} ({die_w_mm:.1f} mm x {die_h_mm:.1f} mm)")
    print(f"  Thermal Matrix     : {os.path.relpath(matrix_path, project_root)} ({num_cores}x{num_cores} K/W)")
    print(f"  Active Cores       : {num_cores} ({num_big} Big 4x4 mm, {num_little} Little 2x2 mm), {num_fill} FILL blocks")
    print(f"  Matrix Core Order  : {', '.join(col_core_names)}")
    print("=" * 74)

    # 4. Obtain and validate core power values in exact matrix column order
    if args.powers is not None or args.ptrace is not None:
        try:
            powers = parse_powers_from_cli_or_ptrace(
                args.powers, args.ptrace, col_core_names, core_map
            )
        except Exception as exc:
            print(f"\n[Invalid Input] {exc}", file=sys.stderr)
            sys.exit(1)
    else:
        powers = interactive_prompt_powers(col_core_names, core_map)

    # 5. Calculate steady-state temperatures: Delta_T = R x P, T_K = 318.15 + Delta_T, T_C = T_K - 273.15
    delta_t_list, temp_k_list, temp_c_list, pred_time_us = predict_temperatures(R_rows, powers)

    # 6. Identify maximum predicted temperature and hottest core
    max_idx = max(range(num_cores), key=lambda i: temp_c_list[i])
    hottest_core = col_core_names[max_idx]
    hottest_type = str(core_map[hottest_core]["core_type"])
    max_temp_c = temp_c_list[max_idx]
    max_temp_k = temp_k_list[max_idx]
    total_power_w = sum(powers)

    # 7. Display results table
    print("\n" + "=" * 86)
    print(f"  STEADY-STATE TEMPERATURE PREDICTION RESULTS - {full_name}")
    print("=" * 86)
    print(
        f"  {'Core Name':<11s} | {'Core Type':<16s} | {'Input Power (W)':>15s} "
        f"| {'Predicted Temp (°C)':>21s} | {'Predicted Temp (K)':>18s}"
    )
    print("-" * 86)
    for i, c_name in enumerate(col_core_names):
        c_type = str(core_map[c_name]["core_type"])
        p_val = powers[i]
        tc_val = temp_c_list[i]
        tk_val = temp_k_list[i]
        marker = " *" if i == max_idx else "  "
        tc_str = f"{tc_val:6.2f} ({tc_val:10.6f}){marker}"
        print(
            f"  {c_name:<11s} | {c_type:<16s} | {p_val:>15.2f} "
            f"| {tc_str:>21s} | {tk_val:>18.6f}"
        )
    print("-" * 86)
    print(f"  Total Active Core Power    : {total_power_w:.2f} W (Ambient: {AMBIENT_TEMP_C:.2f} °C / {AMBIENT_TEMP_K:.2f} K)")
    print(f"  Maximum Predicted Temp     : {max_temp_c:.2f} °C ({max_temp_c:.6f} °C | {max_temp_k:.6f} K)")
    print(f"  Hottest Core               : {hottest_core} ({hottest_type})")
    num_params_per_fp = num_cores * num_cores
    total_fps = len(catalog)
    total_params_all_fps = total_fps * num_params_per_fp
    total_raw_bytes_all_fps = total_fps * raw_float64_bytes
    total_csv_bytes_all_fps = total_fps * csv_size_bytes
    print(f"  Prediction Time            : {pred_time_us:.3f} us")
    print(
        f"  Model Size (Per Floorplan) : {num_params_per_fp} parameters (8x8), "
        f"{raw_float64_bytes} B raw float64 / {csv_size_bytes} B CSV on disk"
    )
    print(
        f"  Model Size (All 23 FPs)    : {total_params_all_fps:,} parameters (23x8x8), "
        f"{total_raw_bytes_all_fps:,} B raw float64 / {total_csv_bytes_all_fps:,} B CSV on disk"
    )
    print("=" * 86)


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    main()
