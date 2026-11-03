#!/usr/bin/env python3
import csv
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

    src_ptrace_dir = os.path.join(fp001_dir, "generated_samples", "ptrace")
    out_dir = os.path.join(fp002_dir, "generated_samples")
    dst_ptrace_dir = os.path.join(out_dir, "ptrace")
    dst_steady_dir = os.path.join(out_dir, "steady")

    os.makedirs(dst_ptrace_dir, exist_ok=True)
    os.makedirs(dst_steady_dir, exist_ok=True)

    # Verify required input files exist
    for f in [
        hotspot_bin,
        config_file,
        os.path.join(fp002_dir, flp_file),
        os.path.join(fp002_dir, lcf_file),
        os.path.join(fp002_dir, tim_file),
        src_ptrace_dir,
    ]:
        if not os.path.exists(f):
            print(f"ERROR: Required file/directory not found: {f}", file=sys.stderr)
            sys.exit(1)

    core_names = [f"Core_{i}_L0" for i in range(8)]
    core_steady_keys = [f"layer_0_Core_{i}_L0" for i in range(8)]

    errors = []
    configs = []

    for sample_idx in range(1, 101):
        src_ptrace = os.path.join(src_ptrace_dir, f"sample_{sample_idx:03d}.ptrace")
        dst_ptrace = os.path.join(dst_ptrace_dir, f"sample_{sample_idx:03d}.ptrace")
        dst_steady = os.path.join(dst_steady_dir, f"sample_{sample_idx:03d}.steady")

        if not os.path.exists(src_ptrace):
            print(f"ERROR: Missing source ptrace file: {src_ptrace}", file=sys.stderr)
            sys.exit(1)

        # Copy the power configuration from fp_001 to fp_002/generated_samples/ptrace/
        shutil.copyfile(src_ptrace, dst_ptrace)

        # Read the 8 core power values from dst_ptrace
        with open(dst_ptrace, "r") as f:
            lines = [line.strip() for line in f if line.strip()]
            headers = lines[0].split()
            vals = lines[1].split()
            p_map = dict(zip(headers, vals))
        p_strs = [p_map[c] for c in core_names]
        configs.append((sample_idx, p_strs))

        # Run HotSpot using fp_002 floorplan and layer configuration
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
            err_msg = f"Sample {sample_idx:03d} failed with return code {res.returncode}: {res.stderr}"
            print(err_msg, file=sys.stderr)
            errors.append(err_msg)
        elif not os.path.exists(dst_steady) or os.path.getsize(dst_steady) == 0:
            err_msg = f"Sample {sample_idx:03d} did not produce a valid steady output file."
            print(err_msg, file=sys.stderr)
            errors.append(err_msg)

        if sample_idx % 10 == 0:
            print(f"Completed {sample_idx}/100 simulations for fp_002...")

    if errors:
        print(f"ERROR: {len(errors)} simulations failed!", file=sys.stderr)
        sys.exit(1)

    # Build combined_100_samples.csv and check for any missing values
    csv_path = os.path.join(out_dir, "combined_100_samples.csv")
    csv_headers = (
        ["sample_id"]
        + [f"Core_{i}_L0_power_W" for i in range(8)]
        + [f"Core_{i}_L0_temp_K" for i in range(8)]
    )

    rows = []
    for sample_idx, p_strs in configs:
        dst_steady = os.path.join(dst_steady_dir, f"sample_{sample_idx:03d}.steady")
        temp_map = {}
        with open(dst_steady, "r") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) == 2:
                    temp_map[parts[0]] = float(parts[1])

        core_temps = []
        for k in core_steady_keys:
            if k not in temp_map or temp_map[k] != temp_map[k]:  # check missing or NaN
                print(f"ERROR: Missing or NaN temperature for {k} in {dst_steady}", file=sys.stderr)
                sys.exit(1)
            core_temps.append(f"{temp_map[k]:.2f}")

        row = [f"sample_{sample_idx:03d}"] + p_strs + core_temps
        if len(row) != 17 or any(val == "" or val is None for val in row):
            print(f"ERROR: Invalid row for sample_{sample_idx:03d}: {row}", file=sys.stderr)
            sys.exit(1)
        rows.append(row)

    with open(csv_path, "w", newline="\n") as f:
        writer = csv.writer(f)
        writer.writerow(csv_headers)
        writer.writerows(rows)

    print(f"Successfully generated and verified all {len(rows)} samples for fp_002 (0 missing values).")
    print(f"Combined CSV saved to: {csv_path}")

if __name__ == "__main__":
    main()
