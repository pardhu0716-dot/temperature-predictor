#!/usr/bin/env python3
import csv
import os
import random
import shutil
import subprocess
import sys

def main():
    base_dir = os.path.expanduser("~/temperature_project/dataset/Sample_dataset/2D_floorplans/Style_1/fp_001")
    hotspot_bin = os.path.expanduser("~/HotSpot/hotspot")
    config_file = os.path.expanduser("~/temperature_project/hotspot.config")
    flp_file = "fp_001_layer_0.flp"
    lcf_file = "fp_001.lcf"
    orig_ptrace = os.path.join(base_dir, "fp_001_steady.ptrace")
    orig_steady = os.path.join(base_dir, "fp_001.steady")

    out_dir = os.path.join(base_dir, "generated_samples")
    ptrace_dir = os.path.join(out_dir, "ptrace")
    steady_dir = os.path.join(out_dir, "steady")
    os.makedirs(ptrace_dir, exist_ok=True)
    os.makedirs(steady_dir, exist_ok=True)

    # Verify required input files exist
    for f in [hotspot_bin, config_file, os.path.join(base_dir, flp_file), os.path.join(base_dir, lcf_file), orig_ptrace, orig_steady]:
        if not os.path.exists(f):
            print(f"ERROR: Required file not found: {f}", file=sys.stderr)
            sys.exit(1)

    header_blocks = [
        "Core_0_L0", "Core_1_L0", "Core_2_L0", "Core_3_L0",
        "Core_4_L0", "Core_5_L0", "Core_6_L0", "Core_7_L0",
        "FILL_0_L0", "FILL_1_L0", "FILL_2_L0"
    ]

    # Sample 001 is the existing sample: Big cores (0, 1) = 20.0W, Little cores (2..7) = 8.0W
    seen_configs = set()
    sample_1_powers = (20.0, 20.0, 8.0, 8.0, 8.0, 8.0, 8.0, 8.0)
    seen_configs.add(sample_1_powers)

    # Copy sample 001 files into generated_samples so all 100 samples are numbered consistently without overwriting originals
    s1_ptrace = os.path.join(ptrace_dir, "sample_001.ptrace")
    s1_steady = os.path.join(steady_dir, "sample_001.steady")
    shutil.copyfile(orig_ptrace, s1_ptrace)
    shutil.copyfile(orig_steady, s1_steady)

    # Generate 99 additional unique power configurations
    # Core_0_L0, Core_1_L0: Big cores (4mm x 4mm) -> range [10.0, 20.0] W
    # Core_2_L0 .. Core_7_L0: Little cores (2mm x 2mm) -> range [5.0, 10.0] W
    # All FILL blocks = 0.0 W
    random.seed(42)
    configs = [(1, sample_1_powers)]

    while len(configs) < 100:
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
            sample_idx = len(configs) + 1
            configs.append((sample_idx, p_tuple))

    # Write ptrace files and run HotSpot for samples 2..100
    errors = []
    for sample_idx, p_tuple in configs[1:]:
        ptrace_path = os.path.join(ptrace_dir, f"sample_{sample_idx:03d}.ptrace")
        steady_path = os.path.join(steady_dir, f"sample_{sample_idx:03d}.steady")

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
            err_msg = f"Sample {sample_idx:03d} failed with return code {res.returncode}: {res.stderr}"
            print(err_msg, file=sys.stderr)
            errors.append(err_msg)
        elif not os.path.exists(steady_path) or os.path.getsize(steady_path) == 0:
            err_msg = f"Sample {sample_idx:03d} did not produce a valid steady output file."
            print(err_msg, file=sys.stderr)
            errors.append(err_msg)

        if sample_idx % 10 == 0:
            print(f"Completed {sample_idx}/100 simulations...")

    if errors:
        print(f"ERROR: {len(errors)} simulations failed!", file=sys.stderr)
        sys.exit(1)

    # Parse all 100 steady files and create the combined CSV
    csv_path = os.path.join(out_dir, "combined_100_samples.csv")
    core_Steady_keys = [f"layer_0_Core_{i}_L0" for i in range(8)]

    csv_headers = (
        ["sample_id"]
        + [f"Core_{i}_L0_power_W" for i in range(8)]
        + [f"Core_{i}_L0_temp_K" for i in range(8)]
    )

    rows = []
    for sample_idx, p_tuple in configs:
        steady_path = os.path.join(steady_dir, f"sample_{sample_idx:03d}.steady")
        if not os.path.exists(steady_path):
            print(f"ERROR: Missing steady file for sample {sample_idx:03d}", file=sys.stderr)
            sys.exit(1)

        temp_map = {}
        with open(steady_path, "r") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) == 2:
                    temp_map[parts[0]] = float(parts[1])

        core_temps = []
        for k in core_Steady_keys:
            if k not in temp_map:
                print(f"ERROR: Missing temperature for {k} in {steady_path}", file=sys.stderr)
                sys.exit(1)
            core_temps.append(temp_map[k])

        row = [f"sample_{sample_idx:03d}"] + [f"{p:.2f}" if sample_idx > 1 else f"{p:.1f}" for p in p_tuple] + [f"{t:.2f}" for t in core_temps]
        rows.append(row)

    with open(csv_path, "w", newline="\n") as f:
        writer = csv.writer(f)
        writer.writerow(csv_headers)
        writer.writerows(rows)

    print(f"Successfully generated and verified all {len(rows)} samples.")
    print(f"Combined CSV saved to: {csv_path}")

if __name__ == "__main__":
    main()
