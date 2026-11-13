#!/usr/bin/env python3
"""
Stage A Temperature Predictor Website — Flask Backend
Lightweight Temperature Predictor Project

Connects the interactive web dashboard to the pre-calibrated Stage A 8x8
thermal resistance matrices (R) across all 23 2D floorplans (Style_1..Style_4).

Physics model (steady-state linear superposition with full thermal coupling):
  - Ambient temperature: T_ambient = 318.15 K = 45.00 °C
  - Individual core i:   T_i (°C)  = 45.0 + sum_{j=0..7} R[i, j] * P[j]
  - All cores:           T   (°C)  = 45.0 + R @ P
                         T   (K)   = 318.15 + R @ P

Never invokes HotSpot or recalibrates R during prediction.
"""

import csv
import os
import re
import time
from typing import Any, Dict, List, Optional, Tuple

from flask import Flask, jsonify, render_template, request

try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False

from predict import (
    AMBIENT_TEMP_C,
    AMBIENT_TEMP_K,
    KELVIN_OFFSET,
    discover_floorplans,
    locate_and_load_thermal_matrix,
    parse_flp_cores,
    parse_lcf_active_flp,
    validate_single_power_value,
)
from stage_a_pipeline import parse_flp, parse_steady_ptrace

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
FLOORPLANS_ROOT = os.path.join(PROJECT_ROOT, "dataset", "Sample_dataset", "2D_floorplans")
SUMMARY_CSV_PATH = os.path.join(PROJECT_ROOT, "stage_a_all_floorplans_summary.csv")
BENCHMARK_PER_FP_CSV_PATH = os.path.join(PROJECT_ROOT, "stage_a_benchmark_per_floorplan.csv")
BENCHMARK_OVERALL_CSV_PATH = os.path.join(PROJECT_ROOT, "stage_a_benchmark_overall_summary.csv")

app = Flask(
    __name__,
    template_folder=os.path.join(PROJECT_ROOT, "templates"),
    static_folder=os.path.join(PROJECT_ROOT, "static"),
)


def _extract_core_number(core_name: str, fallback_idx: int) -> int:
    """Extract numeric core ID k from 'Core_k_L0' (e.g., 'Core_2_L0' -> 2)."""
    m = re.search(r"Core_(\d+)", core_name, re.IGNORECASE)
    return int(m.group(1)) if m else fallback_idx


def _load_summary_lookup() -> Dict[str, Dict[str, str]]:
    """Load precomputed Stage A summary metrics from stage_a_all_floorplans_summary.csv."""
    lookup: Dict[str, Dict[str, str]] = {}
    if os.path.isfile(SUMMARY_CSV_PATH):
        with open(SUMMARY_CSV_PATH, "r", encoding="utf-8", newline="") as f:
            for row in csv.DictReader(f):
                lookup[row["full_name"]] = row
    return lookup


def _load_benchmark_per_fp_lookup() -> Dict[str, Dict[str, Any]]:
    """Load precomputed Stage A per-floorplan evaluation benchmark metrics from CSV."""
    lookup: Dict[str, Dict[str, Any]] = {}
    if os.path.isfile(BENCHMARK_PER_FP_CSV_PATH):
        try:
            with open(BENCHMARK_PER_FP_CSV_PATH, "r", encoding="utf-8", newline="") as f:
                for row in csv.DictReader(f):
                    fname = row.get("full_name")
                    if fname:
                        lookup[fname] = row
        except Exception:
            pass
    return lookup


def _load_benchmark_overall() -> Dict[str, Any]:
    """Load precomputed Stage A overall evaluation benchmark summary metrics from CSV."""
    if os.path.isfile(BENCHMARK_OVERALL_CSV_PATH):
        try:
            with open(BENCHMARK_OVERALL_CSV_PATH, "r", encoding="utf-8", newline="") as f:
                reader = csv.DictReader(f)
                return next(reader, {})
        except Exception:
            pass
    return {}


def _load_floorplan_bundle(style: str, floorplan: str) -> Dict[str, Any]:
    """
    Load floorplan geometry (.flp), calibrated 8x8 thermal resistance matrix (R),
    exact matrix core ordering, passive FILL blocks, and default validation powers.
    """
    catalog = discover_floorplans(FLOORPLANS_ROOT)
    target_full = f"{style}/{floorplan}".lower()
    matched_entry = next(
        (item for item in catalog if item["full_name"].lower() == target_full),
        None,
    )
    if matched_entry is None:
        raise KeyError(f"Floorplan '{style}/{floorplan}' not found among the 23 calibrated 2D floorplans.")

    fp_dir = matched_entry["path"]
    fp_id = matched_entry["floorplan"]
    style_id = matched_entry["style"]
    full_name = matched_entry["full_name"]

    # 1. Load pre-calibrated thermal matrix R and exact core column ordering
    matrix_path, col_core_names, R_rows, csv_size_bytes, raw_float64_bytes = (
        locate_and_load_thermal_matrix(fp_dir, fp_id)
    )
    if HAS_NUMPY:
        R_mat = np.array(R_rows, dtype=np.float64)
        raw_float64_bytes = int(R_mat.nbytes)
    else:
        R_mat = None

    # 2. Load active silicon layer .flp geometry (both active cores and passive FILL blocks)
    flp_path = parse_lcf_active_flp(fp_dir)
    core_map, (die_w_mm, die_h_mm), num_fill = parse_flp_cores(flp_path)
    all_flp_blocks = parse_flp(flp_path)

    # 3. Load default valid powers from validation_samples/ptrace/val_sample_001.ptrace
    val_ptrace_1 = os.path.join(fp_dir, "validation_samples", "ptrace", "val_sample_001.ptrace")
    default_power_map: Dict[str, float] = {}
    if os.path.isfile(val_ptrace_1):
        _, default_power_map = parse_steady_ptrace(val_ptrace_1)

    # 4. Build ordered active core list matching exact matrix column order (col_core_names)
    ordered_cores: List[Dict[str, Any]] = []
    for mat_idx, c_name in enumerate(col_core_names):
        if c_name not in core_map:
            raise ValueError(f"Matrix core '{c_name}' missing from floorplan geometry {flp_path}")
        c_info = core_map[c_name]
        p_min, p_max = c_info["power_range"]
        mid_p = round((p_min + p_max) / 2.0, 2)
        def_p = round(float(default_power_map.get(c_name, mid_p)), 2)
        if def_p < p_min or def_p > p_max:
            def_p = mid_p

        ordered_cores.append(
            {
                "name": c_name,
                "core_number": _extract_core_number(c_name, mat_idx),
                "matrix_index": mat_idx,
                "core_type": c_info["core_type"],
                "short_type": c_info["short_type"],
                "width_mm": round(float(c_info["width_mm"]), 4),
                "height_mm": round(float(c_info["height_mm"]), 4),
                "left_x_mm": round(float(c_info["left_x_mm"]), 4),
                "bottom_y_mm": round(float(c_info["bottom_y_mm"]), 4),
                "area_mm2": round(float(c_info["area_mm2"]), 4),
                "min_power_w": float(p_min),
                "max_power_w": float(p_max),
                "default_power_w": def_p,
                "midpoint_power_w": mid_p,
            }
        )

    #Also build sorted list by core_number (Core 0 .. Core 7) for the 8 individual core buttons
    cores_by_number = sorted(ordered_cores, key=lambda c: c["core_number"])

    # 5. Extract passive FILL blocks for proportional neutral background rendering
    fill_blocks: List[Dict[str, Any]] = []
    for blk in all_flp_blocks:
        if blk["is_fill"]:
            w_mm = float(blk["width"]) * 1e3
            h_mm = float(blk["height"]) * 1e3
            fill_blocks.append(
                {
                    "name": str(blk["name"]),
                    "width_mm": round(w_mm, 4),
                    "height_mm": round(h_mm, 4),
                    "left_x_mm": round(float(blk["left_x"]) * 1e3, 4),
                    "bottom_y_mm": round(float(blk["bottom_y"]) * 1e3, 4),
                    "area_mm2": round(w_mm * h_mm, 4),
                    "power_w": 0.0,
                }
            )

    # 6. Load floorplan validation metrics from summary CSV if available
    summary_lookup = _load_summary_lookup()
    summary_row = summary_lookup.get(full_name, {})

    # 7. Load precomputed Stage A evaluation benchmark metrics if available
    bench_lookup = _load_benchmark_per_fp_lookup()
    bench_row = bench_lookup.get(full_name)
    num_params_per_fp = len(col_core_names) * len(col_core_names)
    total_fps = len(catalog)
    total_params_all_fps = total_fps * num_params_per_fp
    total_raw_bytes_all_fps = total_fps * raw_float64_bytes
    total_csv_bytes_all_fps = sum(
        int(r.get("serialized_csv_bytes", 771)) for r in bench_lookup.values()
    ) if bench_lookup else total_fps * csv_size_bytes

    if bench_row:
        benchmark_data = {
            "available": True,
            "full_name": full_name,
            "val_rmse_c": float(bench_row.get("val_rmse_C", bench_row.get("val_rmse_K", 0.0))),
            "val_mae_c": float(bench_row.get("val_mae_C", bench_row.get("val_mae_K", 0.0))),
            "val_max_error_c": float(bench_row.get("val_max_error_C", bench_row.get("val_max_error_K", 0.0))),
            "val_rmse_k": float(bench_row.get("val_rmse_K", 0.0)),
            "val_mae_k": float(bench_row.get("val_mae_K", 0.0)),
            "val_max_error_k": float(bench_row.get("val_max_error_K", 0.0)),
            "train_rmse_k": float(bench_row.get("train_rmse_K", 0.0)),
            "numpy_median_time_us": float(bench_row.get("numpy_median_time_us", 0.0)),
            "numpy_mean_time_us": float(bench_row.get("numpy_mean_time_us", 0.0)),
            "stdlib_median_time_us": float(bench_row.get("stdlib_median_time_us", 0.0)),
            "web_api_median_time_ms": float(bench_row["web_api_median_time_ms"]) if bench_row.get("web_api_median_time_ms") else None,
            "cli_e2e_median_time_ms": float(bench_row["cli_e2e_median_time_ms"]) if bench_row.get("cli_e2e_median_time_ms") else None,
            "hotspot_median_time_ms": float(bench_row.get("hotspot_median_time_ms", 0.0)),
            "hotspot_mean_time_ms": float(bench_row.get("hotspot_mean_time_ms", 0.0)),
            "speedup_vs_hotspot_median_x": float(bench_row.get("speedup_vs_hotspot_median_x", 0.0)),
            "num_parameters": num_params_per_fp,
            "raw_float64_bytes": int(bench_row.get("raw_float64_bytes", raw_float64_bytes)),
            "serialized_csv_bytes": int(bench_row.get("serialized_csv_bytes", csv_size_bytes)),
            "serialized_npy_bytes": int(bench_row.get("serialized_npy_bytes", 640)),
            "total_parameters_all_floorplans": total_params_all_fps,
            "total_raw_float64_bytes_all_floorplans": total_raw_bytes_all_fps,
            "total_csv_size_bytes_all_floorplans": total_csv_bytes_all_fps,
            "val_samples": int(bench_row.get("val_samples", 20)),
            "val_comparisons": int(bench_row.get("val_comparisons", 160)),
        }
    else:
        benchmark_data = {
            "available": False,
            "full_name": full_name,
            "message": "Benchmark unavailable for this floorplan.",
        }

    return {
        "index": int(matched_entry["index"]),
        "style": style_id,
        "floorplan": fp_id,
        "full_name": full_name,
        "flp_file": os.path.basename(flp_path),
        "matrix_file": os.path.relpath(matrix_path, PROJECT_ROOT).replace("\\", "/"),
        "die_width_mm": round(die_w_mm, 4),
        "die_height_mm": round(die_h_mm, 4),
        "die_area_mm2": round(die_w_mm * die_h_mm, 4),
        "ambient_temp_k": AMBIENT_TEMP_K,
        "ambient_temp_c": AMBIENT_TEMP_C,
        "num_active_cores": len(ordered_cores),
        "num_big_cores": sum(1 for c in ordered_cores if c["short_type"] == "Big"),
        "num_little_cores": sum(1 for c in ordered_cores if c["short_type"] == "Little"),
        "num_fill_blocks": num_fill,
        "matrix_core_order": col_core_names,
        "cores": ordered_cores,
        "cores_by_number": cores_by_number,
        "fill_blocks": fill_blocks,
        "thermal_matrix_k_per_w": R_rows,
        "model_parameters_per_floorplan": num_params_per_fp,
        "model_csv_size_bytes": csv_size_bytes,
        "model_raw_float64_bytes": raw_float64_bytes,
        "total_model_parameters_all_floorplans": total_params_all_fps,
        "total_raw_float64_bytes_all_floorplans": total_raw_bytes_all_fps,
        "total_csv_size_bytes_all_floorplans": total_csv_bytes_all_fps,
        "validation_metrics": {
            "train_rmse_k": float(summary_row["train_rmse_K"]) if "train_rmse_K" in summary_row else None,
            "val_rmse_k": float(summary_row["val_rmse_K"]) if "val_rmse_K" in summary_row else None,
            "val_mae_k": float(summary_row["val_mae_K"]) if "val_mae_K" in summary_row else None,
            "val_max_error_k": float(summary_row["val_max_error_K"]) if "val_max_error_K" in summary_row else None,
            "matrix_rank": int(summary_row["matrix_rank"]) if "matrix_rank" in summary_row else len(col_core_names),
            "power_matrix_cond": float(summary_row["power_matrix_cond"]) if "power_matrix_cond" in summary_row else None,
        },
        "benchmark": benchmark_data,
        "_internal": {
            "core_map": core_map,
            "R_rows": R_rows,
            "R_mat": R_mat,
        },
    }


@app.route("/", methods=["GET"])
def index():
    """Render the main Stage A Temperature Predictor engineering dashboard."""
    return render_template("index.html")


@app.route("/api/floorplans", methods=["GET"])
def api_list_floorplans():
    """Return all 23 available 2D floorplans grouped by style and as an ordered catalog."""
    catalog = discover_floorplans(FLOORPLANS_ROOT)
    summary_lookup = _load_summary_lookup()

    enriched_catalog: List[Dict[str, Any]] = []
    by_style: Dict[str, List[Dict[str, Any]]] = {}

    for item in catalog:
        full_name = item["full_name"]
        s_row = summary_lookup.get(full_name, {})
        entry = {
            "index": int(item["index"]),
            "style": item["style"],
            "floorplan": item["floorplan"],
            "full_name": full_name,
            "die_dimensions_mm": s_row.get("die_dimensions_mm", ""),
            "num_big_cores": int(s_row.get("num_big_cores", 2)),
            "num_little_cores": int(s_row.get("num_little_cores", 6)),
            "num_fill_blocks": int(s_row.get("num_fill_blocks", 0)),
            "val_rmse_k": float(s_row["val_rmse_K"]) if "val_rmse_K" in s_row else None,
        }
        enriched_catalog.append(entry)
        by_style.setdefault(item["style"], []).append(entry)

    return jsonify(
        {
            "total_floorplans": len(enriched_catalog),
            "ambient_temp_k": AMBIENT_TEMP_K,
            "ambient_temp_c": AMBIENT_TEMP_C,
            "styles": {
                style_name: {
                    "count": len(fps),
                    "floorplans": fps,
                }
                for style_name, fps in by_style.items()
            },
            "floorplans": enriched_catalog,
        }
    )


@app.route("/api/floorplan/<style>/<floorplan>", methods=["GET"])
def api_get_floorplan(style: str, floorplan: str):
    """Return full geometry, core ordering, power ranges, and model metadata for a floorplan."""
    try:
        bundle = _load_floorplan_bundle(style, floorplan)
    except KeyError as exc:
        return jsonify({"error": str(exc)}), 404
    except Exception as exc:
        return jsonify({"error": f"Failed to load floorplan '{style}/{floorplan}': {exc}"}), 500

    response_data = {k: v for k, v in bundle.items() if k != "_internal"}
    return jsonify(response_data)


@app.route("/api/benchmark/<style>/<floorplan>", methods=["GET"])
def api_get_benchmark(style: str, floorplan: str):
    """Return precomputed Stage A benchmark metrics for a floorplan."""
    full_name = f"{style}/{floorplan}"
    lookup = _load_benchmark_per_fp_lookup()
    row = lookup.get(full_name)
    if not row:
        return jsonify({"available": False, "message": f"Benchmark unavailable for {full_name}."}), 404
    bench_data = {
        "available": True,
        "full_name": full_name,
        "val_rmse_c": float(row.get("val_rmse_C", row.get("val_rmse_K", 0.0))),
        "val_mae_c": float(row.get("val_mae_C", row.get("val_mae_K", 0.0))),
        "val_max_error_c": float(row.get("val_max_error_C", row.get("val_max_error_K", 0.0))),
        "numpy_median_time_us": float(row.get("numpy_median_time_us", 0.0)),
        "hotspot_median_time_ms": float(row.get("hotspot_median_time_ms", 0.0)),
        "speedup_vs_hotspot_median_x": float(row.get("speedup_vs_hotspot_median_x", 0.0)),
        "num_parameters": 64,
        "raw_float64_bytes": int(row.get("raw_float64_bytes", 512)),
        "serialized_csv_bytes": int(row.get("serialized_csv_bytes", 771)),
        "total_parameters_all_floorplans": 1472,
        "total_raw_float64_bytes_all_floorplans": 11776,
        "total_csv_size_bytes_all_floorplans": 17733,
        "metrics": row,
    }
    return jsonify(bench_data)


@app.route("/api/benchmark/overall", methods=["GET"])
def api_get_benchmark_overall():
    """Return precomputed Stage A overall evaluation summary metrics."""
    data = _load_benchmark_overall()
    if not data:
        return jsonify({"available": False, "message": "Overall benchmark summary unavailable."}), 404
    resp = {
        "available": True,
        "total_floorplans": int(data.get("total_floorplans", 23)),
        "pooled_val_rmse_k": float(data.get("pooled_val_rmse_K", 0.0)),
        "speedup_numpy_median_x": float(data.get("speedup_numpy_median_x", 0.0)),
        "numpy_median_inference_time_us": float(data.get("numpy_median_inference_time_us", 0.0)),
        "hotspot_median_simulation_time_ms": float(data.get("hotspot_median_simulation_time_ms", 0.0)),
        "summary": data,
    }
    return jsonify(resp)


def _resolve_target_core_name(
    raw_target: Any, col_core_names: List[str]
) -> Tuple[str, Optional[str]]:
    """
    Resolve prediction target:
      - 'all' -> ('all', None)
      - exact core name 'Core_0_L0'..'Core_7_L0' -> ('single', 'Core_X_L0')
      - integer or string '0'..'7' or 'Core 0'..'Core 7' -> resolves to 'Core_<k>_L0' by actual name
    """
    if raw_target is None:
        return "all", None

    token = str(raw_target).strip()
    if not token or token.lower() in ("all", "predict_all", "all_cores"):
        return "all", None

    # Exact case-insensitive match against matrix core names
    for c_name in col_core_names:
        if token.lower() == c_name.lower():
            return "single", c_name

    # Match '0'..'7' or 'Core 0' or 'Core_0' to 'Core_<k>_L0'
    m = re.match(r"^(?:core[\s_]*)?(\d+)$", token, re.IGNORECASE)
    if m:
        core_num = int(m.group(1))
        candidate = f"Core_{core_num}_L0"
        for c_name in col_core_names:
            if c_name.lower() == candidate.lower():
                return "single", c_name

    raise ValueError(
        f"Invalid prediction target '{raw_target}'. Must be 'all' or one of {col_core_names}."
    )


@app.route("/api/predict", methods=["POST"])
def api_predict():
    """
    Predict steady-state core temperature(s) using the floorplan's pre-calibrated 8x8 matrix R.
    Supports both 'all' (Predict All Cores) and individual core targets ('Core_0_L0'..'Core_7_L0').
    Even for an individual core prediction, all 8 core powers are required and used due to thermal coupling:
        T_i (°C) = 45.0 + sum_{j=0..7} R[i, j] * P[j]
    """
    payload = request.get_json(silent=True)
    if not isinstance(payload, dict):
        return jsonify({"error": "Request body must be a JSON object."}), 400

    # 1. Identify floorplan
    fp_str = str(payload.get("floorplan", "")).strip()
    style_str = str(payload.get("style", "")).strip()
    if "/" in fp_str and not style_str:
        parts = [p.strip() for p in fp_str.split("/") if p.strip()]
        if len(parts) == 2:
            style_str, fp_str = parts[0], parts[1]

    if not style_str or not fp_str:
        return (
            jsonify(
                {
                    "error": "Missing 'floorplan' parameter (expected format 'Style_X/fp_YYY')."
                }
            ),
            400,
        )

    try:
        bundle = _load_floorplan_bundle(style_str, fp_str)
    except KeyError as exc:
        return jsonify({"error": str(exc)}), 404
    except Exception as exc:
        return jsonify({"error": f"Failed to load floorplan model: {exc}"}), 500

    col_core_names: List[str] = bundle["matrix_core_order"]
    core_map: Dict[str, Dict[str, Any]] = bundle["_internal"]["core_map"]
    R_rows: List[List[float]] = bundle["_internal"]["R_rows"]
    R_mat = bundle["_internal"]["R_mat"]
    num_cores = len(col_core_names)

    # 2. Resolve target ('all' vs single core name)
    try:
        mode, target_core_name = _resolve_target_core_name(
            payload.get("target", payload.get("target_core", "all")),
            col_core_names,
        )
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400

    # 3. Extract and validate all 8 core powers in exact matrix column order
    powers_by_core_raw = payload.get("powers_by_core")
    powers_list_raw = payload.get("powers")

    raw_power_entries: List[Tuple[str, Any]] = []
    if isinstance(powers_by_core_raw, dict):
        missing_cores = [c for c in col_core_names if c not in powers_by_core_raw]
        if missing_cores:
            return (
                jsonify(
                    {
                        "error": f"Missing power values for core(s): {', '.join(missing_cores)}. "
                        f"All {num_cores} active core powers are required due to thermal coupling."
                    }
                ),
                400,
            )
        for c_name in col_core_names:
            raw_power_entries.append((c_name, powers_by_core_raw[c_name]))
    elif isinstance(powers_by_core_raw, list):
        powers_list_raw = powers_by_core_raw

    if not raw_power_entries:
        if isinstance(powers_list_raw, dict):
            missing_cores = [c for c in col_core_names if c not in powers_list_raw]
            if missing_cores:
                return (
                    jsonify(
                        {
                            "error": f"Missing power values for core(s): {', '.join(missing_cores)}."
                        }
                    ),
                    400,
                )
            for c_name in col_core_names:
                raw_power_entries.append((c_name, powers_list_raw[c_name]))
        elif isinstance(powers_list_raw, list):
            if len(powers_list_raw) != num_cores:
                return (
                    jsonify(
                        {
                            "error": f"Expected {num_cores} power values in matrix column order "
                            f"({', '.join(col_core_names)}), but received {len(powers_list_raw)}."
                        }
                    ),
                    400,
                )
            for c_name, raw_v in zip(col_core_names, powers_list_raw):
                raw_power_entries.append((c_name, raw_v))
        else:
            return (
                jsonify(
                    {
                        "error": "Missing 'powers' (list of 8 numbers in matrix order) or "
                        "'powers_by_core' (object mapping core name to watts)."
                    }
                ),
                400,
            )

    validated_powers: List[float] = []
    field_errors: Dict[str, str] = {}
    for c_name, raw_v in raw_power_entries:
        str_v = "" if raw_v is None else str(raw_v)
        val, err = validate_single_power_value(str_v, c_name, core_map[c_name])
        if err is not None:
            field_errors[c_name] = err
        else:
            validated_powers.append(float(val))

    if field_errors:
        first_err = next(iter(field_errors.values()))
        return (
            jsonify(
                {
                    "error": first_err,
                    "field_errors": field_errors,
                }
            ),
            400,
        )

    total_power_w = float(sum(validated_powers))

    # 4. Execute linear thermal prediction using NumPy (or stdlib fallback) and measure live time in microseconds via perf_counter_ns
    if mode == "all":
        if HAS_NUMPY and R_mat is not None:
            P_vec = np.array(validated_powers, dtype=np.float64)
            # Warmup once so one-time BLAS dispatch is not included in timer
            _ = AMBIENT_TEMP_K + (R_mat @ P_vec) - KELVIN_OFFSET
            t0_ns = time.perf_counter_ns()
            delta_t_arr = R_mat @ P_vec
            t1_ns = time.perf_counter_ns()
            live_pred_time_us = (t1_ns - t0_ns) / 1000.0

            temp_k_arr = AMBIENT_TEMP_K + delta_t_arr
            temp_c_arr = temp_k_arr - KELVIN_OFFSET
            delta_t_list = [float(x) for x in delta_t_arr]
            temp_k_list = [float(x) for x in temp_k_arr]
            temp_c_list = [float(x) for x in temp_c_arr]
        else:
            _ = [
                (AMBIENT_TEMP_K + sum(r_ij * p_j for r_ij, p_j in zip(row, validated_powers))) - KELVIN_OFFSET
                for row in R_rows
            ]
            t0_ns = time.perf_counter_ns()
            delta_t_list = [
                sum(r_ij * p_j for r_ij, p_j in zip(row, validated_powers))
                for row in R_rows
            ]
            t1_ns = time.perf_counter_ns()
            live_pred_time_us = (t1_ns - t0_ns) / 1000.0

            temp_k_list = [AMBIENT_TEMP_K + dt for dt in delta_t_list]
            temp_c_list = [tk - KELVIN_OFFSET for tk in temp_k_list]

        max_idx = max(range(num_cores), key=lambda i: temp_c_list[i])
        min_idx = min(range(num_cores), key=lambda i: temp_c_list[i])
        hottest_core_name = col_core_names[max_idx]
        coolest_core_name = col_core_names[min_idx]

        predictions_list: List[Dict[str, Any]] = []
        predictions_by_core: Dict[str, Dict[str, Any]] = {}
        for i, c_name in enumerate(col_core_names):
            c_info = core_map[c_name]
            self_dt = float(R_rows[i][i] * validated_powers[i])
            coupled_dt = float(delta_t_list[i] - self_dt)
            entry = {
                "core_name": c_name,
                "core_number": _extract_core_number(c_name, i),
                "matrix_index": i,
                "core_type": c_info["core_type"],
                "short_type": c_info["short_type"],
                "power_w": round(validated_powers[i], 4),
                "delta_t_c": round(delta_t_list[i], 6),
                "self_heating_c": round(self_dt, 6),
                "cross_coupling_c": round(coupled_dt, 6),
                "temp_c": round(temp_c_list[i], 6),
                "temp_k": round(temp_k_list[i], 6),
                "is_hottest": i == max_idx,
            }
            predictions_list.append(entry)
            predictions_by_core[c_name] = entry

        return jsonify(
            {
                "mode": "all",
                "target": "all",
                "floorplan": bundle["full_name"],
                "style": bundle["style"],
                "floorplan_id": bundle["floorplan"],
                "ambient_temp_c": AMBIENT_TEMP_C,
                "ambient_temp_k": AMBIENT_TEMP_K,
                "matrix_core_order": col_core_names,
                "powers_w": [round(p, 4) for p in validated_powers],
                "powers_by_core_w": {
                    col_core_names[i]: round(validated_powers[i], 4) for i in range(num_cores)
                },
                "total_power_w": round(total_power_w, 4),
                "predictions": predictions_list,
                "predictions_by_core": predictions_by_core,
                "hottest_core": hottest_core_name,
                "hottest_core_type": core_map[hottest_core_name]["core_type"],
                "max_temp_c": round(temp_c_list[max_idx], 6),
                "max_temp_k": round(temp_k_list[max_idx], 6),
                "coolest_core": coolest_core_name,
                "min_temp_c": round(temp_c_list[min_idx], 6),
                "min_temp_k": round(temp_k_list[min_idx], 6),
                "live_prediction_time_us": round(live_pred_time_us, 3),
                "prediction_time_us": round(live_pred_time_us, 3),
                "model_parameters_per_floorplan": bundle["model_parameters_per_floorplan"],
                "model_csv_size_bytes": bundle["model_csv_size_bytes"],
                "model_raw_float64_bytes": bundle["model_raw_float64_bytes"],
                "total_model_parameters_all_floorplans": bundle["total_model_parameters_all_floorplans"],
                "total_raw_float64_bytes_all_floorplans": bundle["total_raw_float64_bytes_all_floorplans"],
                "total_csv_size_bytes_all_floorplans": bundle["total_csv_size_bytes_all_floorplans"],
            }
        )

    else:
        # Individual core prediction: T_i = 45 + sum_{j=0..7} R[i, j] * P[j]
        assert target_core_name is not None
        target_idx = col_core_names.index(target_core_name)
        row_R = R_rows[target_idx]

        if HAS_NUMPY and R_mat is not None:
            P_vec = np.array(validated_powers, dtype=np.float64)
            R_row_vec = R_mat[target_idx]
            # Warmup once
            _ = (AMBIENT_TEMP_K + float(np.dot(R_row_vec, P_vec))) - KELVIN_OFFSET
            t0_ns = time.perf_counter_ns()
            delta_t_val = float(np.dot(R_row_vec, P_vec))
            t1_ns = time.perf_counter_ns()
            live_pred_time_us = (t1_ns - t0_ns) / 1000.0

            temp_k_val = AMBIENT_TEMP_K + delta_t_val
            temp_c_val = temp_k_val - KELVIN_OFFSET
        else:
            _ = (AMBIENT_TEMP_K + sum(r_ij * p_j for r_ij, p_j in zip(row_R, validated_powers))) - KELVIN_OFFSET
            t0_ns = time.perf_counter_ns()
            delta_t_val = sum(r_ij * p_j for r_ij, p_j in zip(row_R, validated_powers))
            t1_ns = time.perf_counter_ns()
            live_pred_time_us = (t1_ns - t0_ns) / 1000.0

            temp_k_val = AMBIENT_TEMP_K + delta_t_val
            temp_c_val = temp_k_val - KELVIN_OFFSET

        c_info = core_map[target_core_name]
        self_dt = float(row_R[target_idx] * validated_powers[target_idx])
        coupled_dt = float(delta_t_val - self_dt)

        # Coupling contributions from each of the 8 active cores j -> target_idx
        coupling_contributions = [
            {
                "source_core": col_core_names[j],
                "source_type": core_map[col_core_names[j]]["core_type"],
                "r_ij_k_per_w": round(float(row_R[j]), 6),
                "power_w": round(validated_powers[j], 4),
                "temp_rise_c": round(float(row_R[j] * validated_powers[j]), 6),
                "is_self": j == target_idx,
            }
            for j in range(num_cores)
        ]

        single_entry = {
            "core_name": target_core_name,
            "core_number": _extract_core_number(target_core_name, target_idx),
            "matrix_index": target_idx,
            "core_type": c_info["core_type"],
            "short_type": c_info["short_type"],
            "power_w": round(validated_powers[target_idx], 4),
            "delta_t_c": round(delta_t_val, 6),
            "self_heating_c": round(self_dt, 6),
            "cross_coupling_c": round(coupled_dt, 6),
            "temp_c": round(temp_c_val, 6),
            "temp_k": round(temp_k_val, 6),
            "is_selected": True,
            "coupling_contributions": coupling_contributions,
        }

        return jsonify(
            {
                "mode": "single",
                "target": target_core_name,
                "floorplan": bundle["full_name"],
                "style": bundle["style"],
                "floorplan_id": bundle["floorplan"],
                "ambient_temp_c": AMBIENT_TEMP_C,
                "ambient_temp_k": AMBIENT_TEMP_K,
                "matrix_core_order": col_core_names,
                "powers_w": [round(p, 4) for p in validated_powers],
                "powers_by_core_w": {
                    col_core_names[i]: round(validated_powers[i], 4) for i in range(num_cores)
                },
                "total_power_w": round(total_power_w, 4),
                "selected_core": single_entry,
                "predictions": [single_entry],
                "predictions_by_core": {target_core_name: single_entry},
                "live_prediction_time_us": round(live_pred_time_us, 3),
                "prediction_time_us": round(live_pred_time_us, 3),
                "model_parameters_per_floorplan": bundle["model_parameters_per_floorplan"],
                "model_csv_size_bytes": bundle["model_csv_size_bytes"],
                "model_raw_float64_bytes": bundle["model_raw_float64_bytes"],
                "total_model_parameters_all_floorplans": bundle["total_model_parameters_all_floorplans"],
                "total_raw_float64_bytes_all_floorplans": bundle["total_raw_float64_bytes_all_floorplans"],
                "total_csv_size_bytes_all_floorplans": bundle["total_csv_size_bytes_all_floorplans"],
            }
        )


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
