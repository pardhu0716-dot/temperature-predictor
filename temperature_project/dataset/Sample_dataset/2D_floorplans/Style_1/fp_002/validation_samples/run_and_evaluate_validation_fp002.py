#!/usr/bin/env python3
import csv
import math
import os
import shutil
import subprocess
import sys

def main():
    fp001_dir = os.path.expanduser("~/temperature_project/dataset/Sample_dataset/2D_floorplans/Style_1/fp_001")
    fp002_dir = os.path.expanduser("~/temperature_project/dataset/Sample_dataset/2D_floorplans/Style_1/fp_002")
    hotspot_bin = os.path.expanduser("~/HotSpot/hotspot")
    config_file = os.path.expanduser("~/temperature_project/hotspot.config")

    flp_file = "fp_002_layer_0.flp"
    lcf_file = "fp_002.lcf"
    tim_file = "fp_002_tim.flp"

    src_val_ptrace_dir = os.path.join(fp001_dir, "validation_samples", "ptrace")
    train_csv_fp002 = os.path.join(fp002_dir, "generated_samples", "combined_100_samples.csv")
    matrix_path = os.path.join(fp002_dir, "generated_samples", "thermal_matrix_fp002.csv")

    val_dir = os.path.join(fp002_dir, "validation_samples")
    dst_ptrace_dir = os.path.join(val_dir, "ptrace")
    dst_steady_dir = os.path.join(val_dir, "steady")

    out_csv_path = os.path.join(val_dir, "validation_comparisons_160.csv")
    out_report_path = os.path.join(val_dir, "validation_summary_report.txt")

    os.makedirs(dst_ptrace_dir, exist_ok=True)
    os.makedirs(dst_steady_dir, exist_ok=True)

    # Verify required files exist
    for f in [
        hotspot_bin,
        config_file,
        os.path.join(fp002_dir, flp_file),
        os.path.join(fp002_dir, lcf_file),
        os.path.join(fp002_dir, tim_file),
        src_val_ptrace_dir,
        train_csv_fp002,
        matrix_path,
    ]:
        if not os.path.exists(f):
            print(f"ERROR: Required file/directory not found: {f}", file=sys.stderr)
            sys.exit(1)

    core_names = [f"Core_{i}_L0" for i in range(8)]
    power_cols = [f"Core_{i}_L0_power_W" for i in range(8)]
    steady_keys = [f"layer_0_Core_{i}_L0" for i in range(8)]
    ambient_temp = 318.15

    # Load all 100 training configurations for fp_002 to verify zero overlap
    train_configs = set()
    with open(train_csv_fp002, "r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            p_tuple = tuple(round(float(row[c]), 2) for c in power_cols)
            train_configs.add(p_tuple)

    print(f"Loaded {len(train_configs)} training power configurations from fp_002.")

    # 1. Copy 20 validation power configurations and 2. Verify zero overlap with training set
    val_powers = {}
    for val_idx in range(1, 21):
        sample_id = f"val_sample_{val_idx:03d}"
        src_ptrace = os.path.join(src_val_ptrace_dir, f"{sample_id}.ptrace")
        dst_ptrace = os.path.join(dst_ptrace_dir, f"{sample_id}.ptrace")

        if not os.path.exists(src_ptrace):
            print(f"ERROR: Missing source validation ptrace: {src_ptrace}", file=sys.stderr)
            sys.exit(1)

        shutil.copyfile(src_ptrace, dst_ptrace)

        with open(dst_ptrace, "r") as f:
            lines = [line.strip() for line in f if line.strip()]
            headers = lines[0].split()
            vals = [float(x) for x in lines[1].split()]
            p_map = dict(zip(headers, vals))

        p_tuple = tuple(round(p_map[c], 2) for c in core_names)
        if p_tuple in train_configs:
            print(f"ERROR: Validation configuration {sample_id} overlaps with training set!", file=sys.stderr)
            sys.exit(1)
        val_powers[sample_id] = [p_map[c] for c in core_names]

    print("Verified: 0 overlaps between the 20 validation samples and the 100 fp_002 training samples.")

    # 3 & 4. Run HotSpot on all 20 configurations and save in fp_002/validation_samples/steady/
    errors = []
    for val_idx in range(1, 21):
        sample_id = f"val_sample_{val_idx:03d}"
        dst_ptrace = os.path.join(dst_ptrace_dir, f"{sample_id}.ptrace")
        dst_steady = os.path.join(dst_steady_dir, f"{sample_id}.steady")

        cmd = [
            hotspot_bin,
            "-c", config_file,
            "-f", flp_file,
            "-p", dst_ptrace,
            "-grid_layer_file", lcf_file,
            "-model_type", "grid",
            "-steady_file", dst_steady
        ]

        res = subprocess.run(cmd, cwd=fp002_dir, capture_output=True, text=True)
        if res.returncode != 0:
            err_msg = f"Validation sample {sample_id} failed with return code {res.returncode}: {res.stderr}"
            print(err_msg, file=sys.stderr)
            errors.append(err_msg)
        elif not os.path.exists(dst_steady) or os.path.getsize(dst_steady) == 0:
            err_msg = f"Validation sample {sample_id} did not produce a valid steady output file."
            print(err_msg, file=sys.stderr)
            errors.append(err_msg)

        if val_idx % 5 == 0:
            print(f"Completed {val_idx}/20 validation simulations for fp_002...")

    if errors:
        print(f"ERROR: {len(errors)} validation simulations failed!", file=sys.stderr)
        sys.exit(1)

    # 5. Load the existing matrix fp_002/generated_samples/thermal_matrix_fp002.csv without modifying it
    R = []
    with open(matrix_path, "r", newline="") as f:
        reader = csv.reader(f)
        header = next(reader)
        for row in reader:
            R.append([float(x) for x in row[1:9]])
    if len(R) != 8 or any(len(r) != 8 for r in R):
        print(f"ERROR: Expected R shape (8, 8)", file=sys.stderr)
        sys.exit(1)

    # 6 & 7. Predict temperatures and compare against HotSpot outputs across all 20 samples x 8 cores = 160 comparisons
    comparison_rows = []
    sq_errors = []
    abs_errors = []

    for val_idx in range(1, 21):
        sample_id = f"val_sample_{val_idx:03d}"
        dst_steady = os.path.join(dst_steady_dir, f"{sample_id}.steady")

        P_vec = [float(p) for p in val_powers[sample_id]]  # length 8
        T_pred = [ambient_temp + sum(R[i][j] * P_vec[j] for j in range(8)) for i in range(8)]

        t_map = {}
        with open(dst_steady, "r") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) == 2:
                    t_map[parts[0]] = float(parts[1])

        T_actual = [float(t_map[k]) for k in steady_keys]

        for i in range(8):
            actual_val = T_actual[i]
            pred_val = T_pred[i]
            abs_err = abs(actual_val - pred_val)
            sq_err = (actual_val - pred_val) ** 2

            abs_errors.append(abs_err)
            sq_errors.append(sq_err)

            comparison_rows.append([
                sample_id,
                core_names[i],
                f"{actual_val:.2f}",
                f"{pred_val:.6f}",
                f"{abs_err:.6f}"
            ])

    num_comparisons = len(comparison_rows)
    if num_comparisons != 160:
        print(f"ERROR: Expected 160 comparisons, got {num_comparisons}", file=sys.stderr)
        sys.exit(1)

    # 8. Calculate overall validation RMSE, MAE, and maximum absolute error
    rmse = math.sqrt(sum(sq_errors) / num_comparisons)
    mae = sum(abs_errors) / num_comparisons
    max_abs_err = max(abs_errors)

    # 9. Save comparison CSV and validation summary report
    with open(out_csv_path, "w", newline="\n") as f:
        writer = csv.writer(f)
        writer.writerow([
            "Sample_ID",
            "Core_ID",
            "Actual_Temperature_K",
            "Predicted_Temperature_K",
            "Absolute_Error_K"
        ])
        writer.writerows(comparison_rows)

    report_text = (
        "============================================================\n"
        "Stage A Validation Accuracy Summary Report (Style_1/fp_002)\n"
        "============================================================\n"
        f"Thermal Matrix File : {matrix_path}\n"
        f"Validation Samples  : 20 independent configurations\n"
        f"Overlap w/ Training : 0 configurations (verified)\n"
        f"Active Cores        : 8 (Core_0_L0 .. Core_7_L0)\n"
        f"Total Comparisons   : {num_comparisons} (20 samples x 8 cores)\n"
        f"Ambient Temperature : {ambient_temp:.2f} K\n"
        "------------------------------------------------------------\n"
        "Validation Error Metrics:\n"
        f"  - Overall Validation RMSE : {rmse:.6f} K\n"
        f"  - Mean Absolute Error     : {mae:.6f} K\n"
        f"  - Maximum Absolute Error  : {max_abs_err:.6f} K\n"
        "============================================================\n"
    )

    with open(out_report_path, "w", newline="\n") as f:
        f.write(report_text)

    print(report_text)

if __name__ == "__main__":
    main()
