#!/usr/bin/env python3
import csv
import os
import random
import subprocess
import sys

def main():
    base_dir = os.path.expanduser("~/temperature_project/dataset/Sample_dataset/2D_floorplans/Style_1/fp_001")
    hotspot_bin = os.path.expanduser("~/HotSpot/hotspot")
    config_file = os.path.expanduser("~/temperature_project/hotspot.config")
    flp_file = "fp_001_layer_0.flp"
    lcf_file = "fp_001.lcf"

    train_csv = os.path.join(base_dir, "generated_samples", "combined_100_samples.csv")
    val_dir = os.path.join(base_dir, "validation_samples")
    ptrace_dir = os.path.join(val_dir, "ptrace")
    steady_dir = os.path.join(val_dir, "steady")

    os.makedirs(ptrace_dir, exist_ok=True)
    os.makedirs(steady_dir, exist_ok=True)

    # Verify required input files exist
    for f in [hotspot_bin, config_file, os.path.join(base_dir, flp_file), os.path.join(base_dir, lcf_file), train_csv]:
        if not os.path.exists(f):
            print(f"ERROR: Required file not found: {f}", file=sys.stderr)
            sys.exit(1)

    # Load all 100 existing training configurations to ensure zero overlap
    power_cols = [f"Core_{i}_L0_power_W" for i in range(8)]
    seen_configs = set()
    with open(train_csv, "r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            p_tuple = tuple(round(float(row[c]), 2) for c in power_cols)
            seen_configs.add(p_tuple)

    print(f"Loaded {len(seen_configs)} unique training power configurations.")

    header_blocks = [
        "Core_0_L0", "Core_1_L0", "Core_2_L0", "Core_3_L0",
        "Core_4_L0", "Core_5_L0", "Core_6_L0", "Core_7_L0",
        "FILL_0_L0", "FILL_1_L0", "FILL_2_L0"
    ]

    # Use a distinct seed for validation set generation and ensure none match training set
    random.seed(2026)
    val_configs = []
    while len(val_configs) < 20:
        b0 = round(random.uniform(10.0, 20.0), 2)
        b1 = round(random.uniform(10.0, 20.0), 2)
        l2 = round(random.uniform(5.0, 10.0), 2)
        l3 = round(random.uniform(5.0, 10.0), 2)
        l4 = round(random.uniform(5.0, 10.0), 2)
        l5 = round(random.uniform(5.0, 10.0), 2)
        l6 = round(random.uniform(5.0, 10.0), 2)
        l7 = round(random.uniform(5.0, 10.0), 2)
        p_tuple = (b0, b1, l2, l3, l4, l5, l6, l7)
        if p_tuple not in seen_configs:
            seen_configs.add(p_tuple)
            val_idx = len(val_configs) + 1
            val_configs.append((val_idx, p_tuple))

    errors = []
    for val_idx, p_tuple in val_configs:
        ptrace_path = os.path.join(ptrace_dir, f"val_sample_{val_idx:03d}.ptrace")
        steady_path = os.path.join(steady_dir, f"val_sample_{val_idx:03d}.steady")

        with open(ptrace_path, "w", newline="\n") as f:
            f.write("\t".join(header_blocks) + "\n")
            power_strs = [f"{p:.2f}" for p in p_tuple] + ["0.0", "0.0", "0.0"]
            f.write("\t".join(power_strs) + "\n")

        cmd = [
            hotspot_bin,
            "-c", config_file,
            "-f", flp_file,
            "-p", ptrace_path,
            "-grid_layer_file", lcf_file,
            "-model_type", "grid",
            "-steady_file", steady_path
        ]

        res = subprocess.run(cmd, cwd=base_dir, capture_output=True, text=True)
        if res.returncode != 0:
            err_msg = f"Validation sample {val_idx:03d} failed with return code {res.returncode}: {res.stderr}"
            print(err_msg, file=sys.stderr)
            errors.append(err_msg)
        elif not os.path.exists(steady_path) or os.path.getsize(steady_path) == 0:
            err_msg = f"Validation sample {val_idx:03d} did not produce a valid steady output file."
            print(err_msg, file=sys.stderr)
            errors.append(err_msg)

        if val_idx % 5 == 0:
            print(f"Completed {val_idx}/20 validation simulations...")

    if errors:
        print(f"ERROR: {len(errors)} validation simulations failed!", file=sys.stderr)
        sys.exit(1)

    print("Successfully generated and verified all 20 validation samples.")

if __name__ == "__main__":
    main()
