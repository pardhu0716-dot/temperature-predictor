#!/usr/bin/env python3
import csv
import os
import sys
import numpy as np

def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    csv_path = os.path.join(base_dir, "combined_100_samples.csv")
    out_matrix_path = os.path.join(base_dir, "thermal_matrix_fp002.csv")

    ambient_temp = 318.15
    core_names = [f"Core_{i}_L0" for i in range(8)]
    power_cols = [f"Core_{i}_L0_power_W" for i in range(8)]
    temp_cols = [f"Core_{i}_L0_temp_K" for i in range(8)]

    if not os.path.exists(csv_path):
        print(f"ERROR: Training CSV not found: {csv_path}", file=sys.stderr)
        sys.exit(1)

    powers_list = []
    temps_list = []

    with open(csv_path, "r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            powers_list.append([float(row[c]) for c in power_cols])
            temps_list.append([float(row[c]) for c in temp_cols])

    # Arrange into 8x100 matrices (rows = cores 0..7, columns = samples 0..99)
    P = np.array(powers_list, dtype=np.float64).T  # shape (8, 100)
    T = np.array(temps_list, dtype=np.float64).T   # shape (8, 100)

    # Calculate temperature rise above ambient (318.15 K)
    Delta_T = T - ambient_temp  # shape (8, 100)

    # Check rank and condition number of P
    rank_P = int(np.linalg.matrix_rank(P))
    cond_P = float(np.linalg.cond(P))
    singular_values = np.linalg.svd(P, compute_uv=False)

    # Fit Delta_T = R @ P  <=>  P.T @ R.T = Delta_T.T via np.linalg.lstsq
    R_T, residuals, rank_lstsq, s_lstsq = np.linalg.lstsq(P.T, Delta_T.T, rcond=None)
    R = R_T.T  # shape (8, 8)

    if R.shape != (8, 8):
        print(f"ERROR: Expected R shape (8, 8), got {R.shape}", file=sys.stderr)
        sys.exit(1)

    # Calculate training RMSE and maximum absolute training error
    Delta_T_pred = R @ P
    T_pred = Delta_T_pred + ambient_temp
    abs_err = np.abs(T - T_pred)
    train_rmse = float(np.sqrt(np.mean((T - T_pred) ** 2)))
    max_abs_err = float(np.max(abs_err))
    train_mae = float(np.mean(abs_err))

    # Save R as fp_002/generated_samples/thermal_matrix_fp002.csv
    with open(out_matrix_path, "w", newline="\n") as f:
        writer = csv.writer(f)
        writer.writerow(["target_core \\ source_core"] + core_names)
        for i, row_vals in enumerate(R):
            writer.writerow([core_names[i]] + [f"{val:.6f}" for val in row_vals])

    print(f"Power matrix P shape        : {P.shape}")
    print(f"Temp rise matrix Delta_T    : {Delta_T.shape}")
    print(f"Thermal matrix R shape      : {R.shape} (Units: K/W)")
    print(f"Rank of power matrix P      : {rank_P}")
    print(f"Singular values of P        : {np.round(singular_values, 4)}")
    print(f"Condition number of P       : {cond_P:.4f}")
    print(f"Training RMSE               : {train_rmse:.6f} K ({train_rmse:.6e} K)")
    print(f"Maximum Absolute Error      : {max_abs_err:.6f} K ({max_abs_err:.6e} K)")
    print(f"Training MAE                : {train_mae:.6f} K ({train_mae:.6e} K)")

    print("\nThermal Resistance Matrix R for Style_1/fp_002 (K/W):")
    header_str = "          " + " ".join([f"{c:>10s}" for c in core_names])
    print(header_str)
    for i, row_vals in enumerate(R):
        vals_str = " ".join([f"{val:10.6f}" for val in row_vals])
        print(f"{core_names[i]:10s}{vals_str}")

if __name__ == "__main__":
    main()
