#!/usr/bin/env python3
"""
Read-only Batch Verification for Stage A (All 23 Calibrated 2D Floorplans)
Lightweight Temperature Predictor Project
"""

import argparse
import csv
import glob
import math
import os
import sys
import time
from typing import Dict, List, Tuple

import numpy as np

from predict import locate_and_load_thermal_matrix

from stage_a_pipeline import (
    AMBIENT_TEMP_K,
    determine_core_power_range,
    parse_flp,
    parse_lcf,
    parse_steady_ptrace,
)


def inspect_floorplan_metadata(fp_dir: str) -> Dict[str, object]:
    """Inspect floorplan .lcf, .flp, and _steady.ptrace to count Big, Little, FILL blocks and check ptrace consistency."""
    lcf_candidates = glob.glob(os.path.join(fp_dir, "*.lcf"))
    if len(lcf_candidates) != 1:
        raise FileNotFoundError(f"Expected 1 .lcf in {fp_dir}, found {len(lcf_candidates)}")
    lcf_path = lcf_candidates[0]

    ptrace_candidates = glob.glob(os.path.join(fp_dir, "*_steady.ptrace"))
    if len(ptrace_candidates) != 1:
        raise FileNotFoundError(f"Expected 1 *_steady.ptrace in {fp_dir}, found {len(ptrace_candidates)}")
    ptrace_path = ptrace_candidates[0]

    layers = parse_lcf(lcf_path)
    active_layers = [lyr for lyr in layers if lyr["power_dissipation"].upper() == "Y"]
    flp_path = os.path.join(fp_dir, active_layers[0]["flp_file"])

    blocks = parse_flp(flp_path)
    ptrace_headers, init_power_map = parse_steady_ptrace(ptrace_path)
    block_by_name = {str(b["name"]): b for b in blocks}

    num_big = 0
    num_little = 0
    num_fill = 0
    needs_override = False
    active_core_names = []

    for col_name in ptrace_headers:
        blk = block_by_name[col_name]
        init_p = init_power_map[col_name]
        if blk["is_fill"]:
            num_fill += 1
        else:
            active_core_names.append(col_name)
            c_type, p_range, warn = determine_core_power_range(blk, init_p, allow_geometry_fallback=True)
            if "Big" in c_type:
                num_big += 1
            else:
                num_little += 1
            if warn is not None:
                needs_override = True

    die_w_mm = max(float(b["left_x"]) + float(b["width"]) for b in blocks) * 1e3
    die_h_mm = max(float(b["bottom_y"]) + float(b["height"]) for b in blocks) * 1e3

    return {
        "num_big": num_big,
        "num_little": num_little,
        "num_fill": num_fill,
        "num_cores": num_big + num_little,
        "active_core_names": active_core_names,
        "needs_override": needs_override,
        "die_dimensions_mm": f"{die_w_mm:.1f}x{die_h_mm:.1f}",
    }


def check_floorplan_completed(fp_dir: str, fp_id: str, num_cores: int) -> bool:
    """Recognize a usable saved model without requiring raw simulation archives.

    Training traces, simulator outputs, and historical summaries are not runtime
    dependencies of an already-calibrated model. Never recalibrate here.
    """
    matrix_path = os.path.join(fp_dir, "generated_samples", f"thermal_matrix_{fp_id.replace('_', '')}.csv")
    if not os.path.isfile(matrix_path):
        return False
    try:
        _, names, matrix, _, _ = locate_and_load_thermal_matrix(fp_dir, fp_id)
        values = np.asarray(matrix, dtype=np.float64)
        return (
            len(names) == num_cores
            and len(set(names)) == num_cores
            and values.shape == (num_cores, num_cores)
            and bool(np.isfinite(values).all())
        )
    except (OSError, ValueError, TypeError):
        return False


def collect_metrics_from_completed_floorplan(fp_dir: str, style_id: str, fp_id: str, meta: Dict[str, object]) -> Dict[str, object]:
    """Compute/load full standardized metrics from an already completed floorplan without overwriting its data."""
    num_cores = int(meta["num_cores"])
    active_core_names = list(meta["active_core_names"])
    gen_dir = os.path.join(fp_dir, "generated_samples")
    val_dir = os.path.join(fp_dir, "validation_samples")
    matrix_filename = f"thermal_matrix_{fp_id.replace('_', '')}.csv"
    matrix_path = os.path.join(gen_dir, matrix_filename)
    train_csv_path = os.path.join(gen_dir, "combined_100_samples.csv")
    val_comp_path = os.path.join(val_dir, f"validation_comparisons_{20 * num_cores}.csv")

    # Load R
    R_rows = []
    with open(matrix_path, "r", newline="") as f:
        reader = csv.reader(f)
        next(reader)
        for row in reader:
            R_rows.append([float(x) for x in row[1 : num_cores + 1]])
    R = np.array(R_rows, dtype=np.float64)

    # Load training data
    power_cols = [f"{c}_power_W" for c in active_core_names]
    temp_cols = [f"{c}_temp_K" for c in active_core_names]
    p_list, t_list = [], []
    with open(train_csv_path, "r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            p_list.append([float(row[c]) for c in power_cols])
            t_list.append([float(row[c]) for c in temp_cols])
    P_train = np.array(p_list, dtype=np.float64).T
    T_train = np.array(t_list, dtype=np.float64).T

    rank_P = int(np.linalg.matrix_rank(P_train))
    cond_P = float(np.linalg.cond(P_train))

    T_train_pred = AMBIENT_TEMP_K + (R @ P_train)
    train_abs_err = np.abs(T_train - T_train_pred)
    train_rmse = float(np.sqrt(np.mean((T_train - T_train_pred) ** 2)))
    train_mae = float(np.mean(train_abs_err))
    train_max_err = float(np.max(train_abs_err))

    # Load validation comparisons
    val_abs_errs = []
    val_sq_errs = []
    with open(val_comp_path, "r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            act_v = float(row["Actual_Temperature_K"])
            pred_v = float(row["Predicted_Temperature_K"])
            err_v = abs(act_v - pred_v)
            val_abs_errs.append(err_v)
            val_sq_errs.append((act_v - pred_v) ** 2)

    val_rmse = float(np.sqrt(np.mean(val_sq_errs)))
    val_mae = float(np.mean(val_abs_errs))
    val_max_err = float(np.max(val_abs_errs))

    # Measure prediction latency if not already in stage_a_summary_<fp_id>.csv
    summary_csv_path = os.path.join(fp_dir, f"stage_a_summary_{fp_id}.csv")
    if os.path.exists(summary_csv_path):
        with open(summary_csv_path, "r", newline="") as f:
            reader = csv.DictReader(f)
            s_row = next(reader)
            avg_pred_us = float(s_row["avg_prediction_time_us"])
    else:
        single_P_vec = P_train[:, 0:1].copy()
        for _ in range(100):
            _ = AMBIENT_TEMP_K + (R @ single_P_vec)
        t0 = time.perf_counter()
        for _ in range(10000):
            _ = AMBIENT_TEMP_K + (R @ single_P_vec)
        avg_pred_sec = (time.perf_counter() - t0) / 10000.0
        avg_pred_us = avg_pred_sec * 1e6


    csv_file_size = os.path.getsize(matrix_path)
    raw_f64_size = num_cores * num_cores * 8

    return {
        "style": style_id,
        "floorplan": fp_id,
        "full_name": f"{style_id}/{fp_id}",
        "die_dimensions_mm": meta["die_dimensions_mm"],
        "num_big_cores": meta["num_big"],
        "num_little_cores": meta["num_little"],
        "num_fill_blocks": meta["num_fill"],
        "num_active_cores": num_cores,
        "train_samples": len(p_list),
        "val_samples": len(val_abs_errs) // num_cores,
        "matrix_dimensions": f"{num_cores}x{num_cores}",
        "matrix_rank": rank_P,
        "power_matrix_cond": f"{cond_P:.4f}",
        "train_rmse_K": f"{train_rmse:.6f}",
        "train_mae_K": f"{train_mae:.6f}",
        "train_max_error_K": f"{train_max_err:.6f}",
        "val_rmse_K": f"{val_rmse:.6f}",
        "val_mae_K": f"{val_mae:.6f}",
        "val_max_error_K": f"{val_max_err:.6f}",
        "avg_prediction_time_us": f"{avg_pred_us:.4f}",
        "saved_matrix_csv_bytes": csv_file_size,
        "raw_float64_bytes": raw_f64_size,
        "power_override_used": "Yes" if meta["needs_override"] else "No",
        "status": "COMPLETED",
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Verify existing Stage A models without running HotSpot, recalibrating "
            "matrices, or writing dataset/summary files."
        )
    )
    parser.parse_args(argv)
    project_root = os.path.dirname(os.path.abspath(__file__))
    floorplans_root = os.path.join(project_root, "dataset", "Sample_dataset", "2D_floorplans")
    styles = ["Style_1", "Style_2", "Style_3", "Style_4"]
    entries: List[Tuple[str, str, str]] = []
    for style in styles:
        style_dir = os.path.join(floorplans_root, style)
        if not os.path.isdir(style_dir):
            continue
        fp_names = sorted(
            (name for name in os.listdir(style_dir)
             if name.startswith("fp_") and os.path.isdir(os.path.join(style_dir, name))),
            key=lambda name: int(name.split("_")[1]),
        )
        entries.extend((style, fp_id, os.path.join(style_dir, fp_id)) for fp_id in fp_names)

    print(f"Discovered {len(entries)} total 2D floorplans.")
    failures: List[Tuple[str, str]] = []
    verified = 0
    for style, fp_id, fp_dir in entries:
        label = f"{style}/{fp_id}"
        try:
            meta = inspect_floorplan_metadata(fp_dir)
            if not check_floorplan_completed(fp_dir, fp_id, int(meta["num_cores"])):
                raise ValueError("Saved thermal matrix is missing or invalid; no calibration attempted.")
            _, names, _, _, _ = locate_and_load_thermal_matrix(fp_dir, fp_id)
            if set(names) != set(meta["active_core_names"]):
                raise ValueError("Saved matrix core labels do not match the floorplan.")
            verified += 1
            print(f"PASS {label}: existing {len(names)}x{len(names)} calibrated model ready.")
        except Exception as exc:
            failures.append((label, str(exc)))
            print(f"FAIL {label}: {exc}", file=sys.stderr)

    if len(entries) != 23:
        failures.append(("catalog", f"Expected 23 floorplans, found {len(entries)}."))
        print(f"FAIL catalog: {failures[-1][1]}", file=sys.stderr)
    print(f"Verified calibrated models: {verified} / 23")
    print("Existing matrices, training/validation data, and saved summaries were not changed.")
    print("HotSpot was not invoked. No calibration or result-file writes were performed.")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
