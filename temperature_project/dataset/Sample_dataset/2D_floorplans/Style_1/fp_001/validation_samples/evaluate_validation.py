#!/usr/bin/env python3
import csv
import os
import sys
import numpy as np

def main():
    val_dir = os.path.dirname(os.path.abspath(__file__))
    fp_dir = os.path.abspath(os.path.join(val_dir, ".."))
    matrix_path = os.path.join(fp_dir, "generated_samples", "thermal_matrix_fp001.csv")
    ptrace_dir = os.path.join(val_dir, "ptrace")
    steady_dir = os.path.join(val_dir, "steady")

    out_csv_path = os.path.join(val_dir, "validation_comparisons_160.csv")
    out_report_path = os.path.join(val_dir, "validation_summary_report.txt")

    ambient_temp = 318.15
    core_names = [f"Core_{i}_L0" for i in range(8)]
    steady_keys = [f"layer_0_Core_{i}_L0" for i in range(8)]

    if not os.path.exists(matrix_path):
        print(f"ERROR: Matrix file not found: {matrix_path}", file=sys.stderr)
        sys.exit(1)

    # 1. Load the existing 8x8 thermal resistance matrix R
    R_rows = []
    with open(matrix_path, "r", newline="") as f:
        reader = csv.reader(f)
        header = next(reader)
        for row in reader:
            R_rows.append([float(x) for x in row[1:9]])
    R = np.array(R_rows, dtype=np.float64)
    if R.shape != (8, 8):
        print(f"ERROR: Expected R shape (8, 8), got {R.shape}", file=sys.stderr)
        sys.exit(1)

    comparison_rows = []
    sq_errors = []
    abs_errors = []

    for val_idx in range(1, 21):
        sample_id = f"val_sample_{val_idx:03d}"
        ptrace_path = os.path.join(ptrace_dir, f"{sample_id}.ptrace")
        steady_path = os.path.join(steady_dir, f"{sample_id}.steady")

        if not os.path.exists(ptrace_path) or not os.path.exists(steady_path):
            print(f"ERROR: Missing validation files for {sample_id}", file=sys.stderr)
            sys.exit(1)

        # 2. Extract 8 core power values into 8x1 vector P
        with open(ptrace_path, "r") as f:
            lines = [line.strip() for line in f if line.strip()]
            headers = lines[0].split()
            vals = [float(x) for x in lines[1].split()]
            p_map = dict(zip(headers, vals))

        P = np.array([[p_map[c]] for c in core_names], dtype=np.float64)  # shape (8, 1)

        # 3. Predict temperature rise: Delta_T_predicted = R x P
        Delta_T_pred = R @ P  # shape (8, 1)

        # 4. Add ambient temperature (318.15 K)
        T_pred = Delta_T_pred + ambient_temp  # shape (8, 1)

        # 5. Extract actual temperatures from .steady file
        t_map = {}
        with open(steady_path, "r") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) == 2:
                    t_map[parts[0]] = float(parts[1])

        T_actual = np.array([[t_map[k]] for k in steady_keys], dtype=np.float64)  # shape (8, 1)

        # 6. Compare predicted and actual temperatures for all 8 cores
        for i in range(8):
            actual_val = float(T_actual[i, 0])
            pred_val = float(T_pred[i, 0])
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

    # 7. Calculate error metrics
    rmse = float(np.sqrt(np.mean(sq_errors)))
    max_abs_err = float(np.max(abs_errors))
    mae = float(np.mean(abs_errors))

    # 8. Save CSV containing all 160 comparisons
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

    # 9. Save summary report
    report_text = (
        "============================================================\n"
        "Stage A Validation Accuracy Summary Report (Style_1/fp_001)\n"
        "============================================================\n"
        f"Thermal Matrix File : {matrix_path}\n"
        f"Validation Samples  : 20 independent configurations\n"
        f"Active Cores        : 8 (Core_0_L0 .. Core_7_L0)\n"
        f"Total Comparisons   : {num_comparisons} (20 samples x 8 cores)\n"
        f"Ambient Temperature : {ambient_temp:.2f} K\n"
        "------------------------------------------------------------\n"
        "Validation Error Metrics:\n"
        f"  - Overall Validation RMSE : {rmse:.6f} K\n"
        f"  - Maximum Absolute Error  : {max_abs_err:.6f} K\n"
        f"  - Mean Absolute Error     : {mae:.6f} K\n"
        "============================================================\n"
    )

    with open(out_report_path, "w", newline="\n") as f:
        f.write(report_text)

    print(report_text)

if __name__ == "__main__":
    main()
