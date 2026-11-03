#!/usr/bin/env python3
"""
Batch Orchestrator & Final Consolidator for Stage A (All 23 2D Floorplans)
Lightweight Temperature Predictor Project
"""

import csv
import glob
import math
import os
import sys
import time
from typing import Dict, List, Tuple

import numpy as np

from stage_a_pipeline import (
    AMBIENT_TEMP_K,
    determine_core_power_range,
    parse_flp,
    parse_lcf,
    parse_steady_ptrace,
    run_stage_a_for_floorplan,
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
    """Verify if a floorplan already has 100 training samples, thermal matrix, and 20 validation samples."""
    gen_dir = os.path.join(fp_dir, "generated_samples")
    val_dir = os.path.join(fp_dir, "validation_samples")
    matrix_filename = f"thermal_matrix_{fp_id.replace('_', '')}.csv"
    matrix_path = os.path.join(gen_dir, matrix_filename)
    train_csv_path = os.path.join(gen_dir, "combined_100_samples.csv")
    val_comp_path = os.path.join(val_dir, f"validation_comparisons_{20 * num_cores}.csv")

    if not (os.path.exists(matrix_path) and os.path.exists(train_csv_path) and os.path.exists(val_comp_path)):
        return False

    # Verify 100 training steady files and 20 validation steady files exist
    train_steady = glob.glob(os.path.join(gen_dir, "steady", "*.steady"))
    val_steady = glob.glob(os.path.join(val_dir, "steady", "*.steady"))
    if len(train_steady) != 100 or len(val_steady) != 20:
        return False

    return True


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

        # Save stage_a_summary_<fp_id>.csv for fp_001 / fp_002 if not created yet
        csv_file_size = os.path.getsize(matrix_path)
        raw_f64_size = num_cores * num_cores * 8
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
        with open(summary_csv_path, "w", newline="\n") as f:
            writer = csv.writer(f)
            writer.writerow(summary_headers)
            writer.writerow([
                f"{style_id}/{fp_id}",
                num_cores,
                f"{num_cores}x{num_cores}",
                rank_P,
                f"{cond_P:.4f}",
                len(p_list),
                len(val_abs_errs) // num_cores,
                f"{train_rmse:.6f}",
                f"{train_mae:.6f}",
                f"{train_max_err:.6f}",
                f"{val_rmse:.6f}",
                f"{val_mae:.6f}",
                f"{val_max_err:.6f}",
                f"{avg_pred_us:.4f}",
                f"{avg_pred_us*1e-6:.6e}",
                csv_file_size,
                raw_f64_size,
            ])

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


def main():
    project_root = os.path.dirname(os.path.abspath(__file__))
    floorplans_root = os.path.join(project_root, "dataset", "Sample_dataset", "2D_floorplans")

    # Discover all 23 2D floorplans sorted by Style_1..Style_4 and fp_001..fp_010
    styles = ["Style_1", "Style_2", "Style_3", "Style_4"]
    all_fp_entries: List[Tuple[str, str, str]] = []
    for st in styles:
        st_dir = os.path.join(floorplans_root, st)
        if not os.path.isdir(st_dir):
            continue
        fp_names = sorted(
            [d for d in os.listdir(st_dir) if os.path.isdir(os.path.join(st_dir, d)) and d.startswith("fp_")],
            key=lambda x: int(x.split("_")[1]),
        )
        for fp_id in fp_names:
            all_fp_entries.append((st, fp_id, os.path.join(st_dir, fp_id)))

    print(f"Discovered {len(all_fp_entries)} total 2D floorplans.")

    # Step 1: Inspect and classify completed vs remaining floorplans
    completed_list = []
    remaining_list = []
    meta_map: Dict[str, Dict[str, object]] = {}

    for st, fp_id, fp_dir in all_fp_entries:
        key = f"{st}/{fp_id}"
        meta = inspect_floorplan_metadata(fp_dir)
        meta_map[key] = meta
        if check_floorplan_completed(fp_dir, fp_id, int(meta["num_cores"])):
            completed_list.append((st, fp_id, fp_dir))
        else:
            remaining_list.append((st, fp_id, fp_dir))

    print(f"Already completed ({len(completed_list)}): {[f'{s}/{f}' for s, f, _ in completed_list]}")
    print(f"Remaining to run  ({len(remaining_list)}): {[f'{s}/{f}' for s, f, _ in remaining_list]}")

    summary_rows: List[Dict[str, object]] = []
    failures: List[Tuple[str, str]] = []

    # Step 2: Run remaining floorplans sequentially
    for idx, (st, fp_id, fp_dir) in enumerate(remaining_list, start=1):
        key = f"{st}/{fp_id}"
        meta = meta_map[key]
        needs_override = bool(meta["needs_override"])
        print(
            f"\n>>> [{idx}/{len(remaining_list)}] Processing {key} "
            f"(Big={meta['num_big']}, Little={meta['num_little']}, FILL={meta['num_fill']}, "
            f"needs_geometry_override={needs_override}) <<<"
        )
        try:
            run_stage_a_for_floorplan(
                fp_dir=fp_dir,
                hotspot_bin="~/HotSpot/hotspot",
                config_file=os.path.join(project_root, "hotspot.config"),
                num_train=100,
                num_val=20,
                train_seed=42,
                val_seed=2026,
                force_overwrite=False,
                allow_geometry_fallback=needs_override,
            )
        except Exception as exc:
            err_str = f"{type(exc).__name__}: {exc}"
            print(f"ERROR on {key}: {err_str}", file=sys.stderr)
            failures.append((key, err_str))

    # Step 3: Collect consolidated results across all 23 floorplans in canonical order
    total_train_samples = 0
    total_val_samples = 0
    total_matrices = 0

    for st, fp_id, fp_dir in all_fp_entries:
        key = f"{st}/{fp_id}"
        meta = meta_map[key]
        if check_floorplan_completed(fp_dir, fp_id, int(meta["num_cores"])):
            row_dict = collect_metrics_from_completed_floorplan(fp_dir, st, fp_id, meta)
            summary_rows.append(row_dict)
            total_train_samples += int(row_dict["train_samples"])
            total_val_samples += int(row_dict["val_samples"])
            total_matrices += 1
        else:
            summary_rows.append({
                "style": st,
                "floorplan": fp_id,
                "full_name": key,
                "die_dimensions_mm": meta["die_dimensions_mm"],
                "num_big_cores": meta["num_big"],
                "num_little_cores": meta["num_little"],
                "num_fill_blocks": meta["num_fill"],
                "num_active_cores": meta["num_cores"],
                "train_samples": 0,
                "val_samples": 0,
                "matrix_dimensions": f"{meta['num_cores']}x{meta['num_cores']}",
                "matrix_rank": 0,
                "power_matrix_cond": "N/A",
                "train_rmse_K": "N/A",
                "train_mae_K": "N/A",
                "train_max_error_K": "N/A",
                "val_rmse_K": "N/A",
                "val_mae_K": "N/A",
                "val_max_error_K": "N/A",
                "avg_prediction_time_us": "N/A",
                "saved_matrix_csv_bytes": 0,
                "raw_float64_bytes": 0,
                "power_override_used": "Yes" if meta["needs_override"] else "No",
                "status": "FAILED",
            })

    out_summary_csv = os.path.join(project_root, "stage_a_all_floorplans_summary.csv")
    fieldnames = [
        "style",
        "floorplan",
        "full_name",
        "die_dimensions_mm",
        "num_big_cores",
        "num_little_cores",
        "num_fill_blocks",
        "num_active_cores",
        "train_samples",
        "val_samples",
        "matrix_dimensions",
        "matrix_rank",
        "power_matrix_cond",
        "train_rmse_K",
        "train_mae_K",
        "train_max_error_K",
        "val_rmse_K",
        "val_mae_K",
        "val_max_error_K",
        "avg_prediction_time_us",
        "saved_matrix_csv_bytes",
        "raw_float64_bytes",
        "power_override_used",
        "status",
    ]

    with open(out_summary_csv, "w", newline="\n") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(summary_rows)

    print("\n============================================================")
    print("STAGE A CONSOLIDATED FINAL VERIFICATION")
    print("============================================================")
    print(f"Total Floorplans Processed : {len(all_fp_entries)}")
    print(f"Completed Floorplans       : {total_matrices} / {len(all_fp_entries)}")
    print(f"Total Training Samples     : {total_train_samples} (Expected: 2300)")
    print(f"Total Validation Samples   : {total_val_samples} (Expected: 460)")
    print(f"Total Calibrated Matrices  : {total_matrices} (Expected: 23)")
    print(f"Failures Encountered       : {len(failures)}")
    if failures:
        for k, err in failures:
            print(f"  - FAILED: {k}: {err}")
    print(f"Consolidated Summary CSV   : {out_summary_csv}")
    print("============================================================")


if __name__ == "__main__":
    main()
