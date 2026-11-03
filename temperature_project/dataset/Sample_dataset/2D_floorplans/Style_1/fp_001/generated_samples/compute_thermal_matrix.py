#!/usr/bin/env python3
import csv
import os
import sys
import numpy as np

def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(base_dir, "..", "..", "..", "..", ".."))
    csv_path = os.path.join(base_dir, "combined_100_samples.csv")
    out_matrix_path = os.path.join(base_dir, "thermal_matrix_fp001.csv")
    fp_matrix_path = os.path.abspath(os.path.join(base_dir, "..", "thermal_matrix_fp001.csv"))
    root_matrix_path = os.path.join(project_root, "thermal_matrix_fp001.csv")

    ambient_temp = 318.15

    power_cols = [f"Core_{i}_L0_power_W" for i in range(8)]
    temp_cols = [f"Core_{i}_L0_temp_K" for i in range(8)]

    powers_list = []
    temps_list = []

    with open(csv_path, "r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            p_vec = [float(row[c]) for c in power_cols]
            t_vec = [float(row[c]) for c in temp_cols]
            powers_list.append(p_vec)
            temps_list.append(t_vec)

    # Convert to 8 x 100 matrices (rows = cores 0..7, columns = samples 0..99)
    P = np.array(powers_list, dtype=np.float64).T  # shape (8, 100)
    T = np.array(temps_list, dtype=np.float64).T   # shape (8, 100)

    # Temperature rise above ambient (318.15 K)
    Delta_T = T - ambient_temp  # shape (8, 100)

    print(f"Power matrix P shape: {P.shape}")
    print(f"Temperature rise matrix Delta_T shape: {Delta_T.shape}")

    # Check rank and singular values of P
    rank_P = np.linalg.matrix_rank(P)
    singular_values = np.linalg.svd(P, compute_uv=False)
    cond_P = np.linalg.cond(P)
    print(f"Rank of power matrix P: {rank_P}")
    print(f"Singular values of P: {np.round(singular_values, 4)}")
    print(f"Condition number of P: {cond_P:.4f}")

    # Solve Delta_T = R @ P  <=>  P.T @ R.T = Delta_T.T using np.linalg.lstsq (no explicit inverse)
    R_T, residuals, rank_lstsq, s_lstsq = np.linalg.lstsq(P.T, Delta_T.T, rcond=None)
    R = R_T.T  # shape (8, 8)

    # Evaluate reconstruction error
    Delta_T_pred = R @ P
    max_abs_err = np.max(np.abs(Delta_T - Delta_T_pred))
    rmse = np.sqrt(np.mean((Delta_T - Delta_T_pred) ** 2))
    print(f"Max absolute error (K): {max_abs_err:.6e}")
    print(f"RMSE (K): {rmse:.6e}")

    # Save R as thermal_matrix_fp001.csv in generated_samples, fp_001, and project root
    core_names = [f"Core_{i}_L0" for i in range(8)]
    for path in [out_matrix_path, fp_matrix_path, root_matrix_path]:
        with open(path, "w", newline="\n") as f:
            writer = csv.writer(f)
            writer.writerow(["target_core \\ source_core"] + core_names)
            for i, row_vals in enumerate(R):
                writer.writerow([core_names[i]] + [f"{val:.6f}" for val in row_vals])

    print("\nThermal Coupling Matrix R (K/W):")
    header_str = "          " + " ".join([f"{c:>10s}" for c in core_names])
    print(header_str)
    for i, row_vals in enumerate(R):
        vals_str = " ".join([f"{val:10.6f}" for val in row_vals])
        print(f"{core_names[i]:10s}{vals_str}")

if __name__ == "__main__":
    main()
