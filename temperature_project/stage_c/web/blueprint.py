"""
Isolated Flask Blueprint for Stage C Transient Temperature Prediction.

Routes (strictly namespaced under /stage-c and /api/stage-c/*):
  - GET  /stage-c
  - GET  /api/stage-c/health
  - GET  /api/stage-c/floorplans
  - GET  /api/stage-c/floorplan/<style>/<floorplan>
  - POST /api/stage-c/predict

Never defines '/', '/stage-b', or any Stage A / Stage B route, and never mutates
shared Flask app configuration.
"""

from __future__ import annotations

import time
from typing import Any, Dict, List

import numpy as np
from flask import Blueprint, Flask, current_app, jsonify, render_template, request

from .model_registry import RegisteredFloorplanModel, StageCRegistry
from .validation import (
    HARD_MAX_POWER_W,
    HARD_MIN_POWER_W,
    VALIDATED_BIG_RANGE_W,
    VALIDATED_LITTLE_RANGE_W,
    StageCValidationError,
    validate_and_build_power_matrix,
)

MAX_STAGE_C_PAYLOAD_BYTES: int = 256 * 1024  # 256 KiB local cap

stage_c_bp = Blueprint(
    "stage_c_web",
    __name__,
    template_folder="templates",
    static_folder="static",
    static_url_path="/stage-c/assets",
)


def _get_registry() -> StageCRegistry:
    registry = current_app.extensions.get("stage_c_registry")
    if registry is None:
        registry = StageCRegistry()
        current_app.extensions["stage_c_registry"] = registry
    return registry


def _build_presets(entry: RegisteredFloorplanModel) -> List[Dict[str, Any]]:
    """Build floorplan-aware calibrated preset workloads for the UI."""
    core_types = tuple(b.core_type for b in entry.spec.active_cores)
    big_indices = [i for i, t in enumerate(core_types) if t == "Big"]

    def vec(p_big: float, p_lit: float) -> List[float]:
        return [p_big if t == "Big" else p_lit for t in core_types]

    alt_a = [
        (20.0 if i == big_indices[0] else 10.0) if t == "Big" else 7.0
        for i, t in enumerate(core_types)
    ]
    alt_b = [
        (20.0 if i == big_indices[-1] else 10.0) if t == "Big" else 7.0
        for i, t in enumerate(core_types)
    ]

    return [
        {
            "id": "workload_b_4phase",
            "name": "Workload B (4-Phase Calibrated Dynamic)",
            "description": "Exact 4-phase synchronous calibration schedule (10/5 → 20/10 → 10/5 → 15/8 W).",
            "domain": "validated",
            "segments": [
                {"start_step": 1, "end_step": 50, "powers_w": vec(10.0, 5.0)},
                {"start_step": 51, "end_step": 100, "powers_w": vec(20.0, 10.0)},
                {"start_step": 101, "end_step": 150, "powers_w": vec(10.0, 5.0)},
                {"start_step": 151, "end_step": 200, "powers_w": vec(15.0, 8.0)},
            ],
        },
        {
            "id": "big_core_burst",
            "name": "Big-Core Turbo Burst (0.50–1.20 s)",
            "description": "Idle-low baseline followed by a 0.70 s high-power Big-core burst and cooldown.",
            "domain": "validated",
            "segments": [
                {"start_step": 1, "end_step": 50, "powers_w": vec(11.0, 5.5)},
                {"start_step": 51, "end_step": 120, "powers_w": vec(20.0, 9.5)},
                {"start_step": 121, "end_step": 200, "powers_w": vec(11.0, 5.5)},
            ],
        },
        {
            "id": "alternating_big_migration",
            "name": "Alternating Big-Core Task Migration",
            "description": "Opposing high/low power migration between the two Big cores every 0.50 s.",
            "domain": "validated",
            "segments": [
                {"start_step": 1, "end_step": 50, "powers_w": alt_a},
                {"start_step": 51, "end_step": 100, "powers_w": alt_b},
                {"start_step": 101, "end_step": 150, "powers_w": alt_a},
                {"start_step": 151, "end_step": 200, "powers_w": alt_b},
            ],
        },
        {
            "id": "workload_a_constant",
            "name": "Workload A (20 W Big / 8 W Little Step)",
            "description": "Constant calibrated step load across all 200 steps (2.00 s).",
            "domain": "validated",
            "segments": [
                {"start_step": 1, "end_step": 200, "powers_w": vec(20.0, 8.0)},
            ],
        },
        {
            "id": "extrapolation_cool_to_spike",
            "name": "Extrapolation Demo (Low Idle 2 W → 28 W Spike)",
            "description": "Intentionally exercises out-of-domain power levels to demonstrate extrapolation warnings.",
            "domain": "extrapolated",
            "segments": [
                {"start_step": 1, "end_step": 60, "powers_w": vec(2.0, 1.0)},
                {"start_step": 61, "end_step": 130, "powers_w": vec(28.0, 12.0)},
                {"start_step": 131, "end_step": 200, "powers_w": vec(4.0, 2.0)},
            ],
        },
    ]


@stage_c_bp.get("/stage-c")
def stage_c_index():
    return render_template("stage_c/index.html")


@stage_c_bp.get("/api/stage-c/health")
def api_stage_c_health():
    registry = _get_registry()
    return jsonify(
        {
            "status": "ok",
            "stage": "C",
            "model_architecture": "generalized_two_speed_discrete_transient",
            "calibrated_domain": {
                "big_core_power_w": list(VALIDATED_BIG_RANGE_W),
                "little_core_power_w": list(VALIDATED_LITTLE_RANGE_W),
                "physical_power_bounds_w": [HARD_MIN_POWER_W, HARD_MAX_POWER_W],
                "dt_seconds": 0.01,
                "horizon_steps": 200,
                "horizon_seconds": 2.0,
                "init_temp_c": 45.0,
            },
            "summary": registry.aggregate_summary(),
        }
    )


@stage_c_bp.get("/api/stage-c/floorplans")
def api_stage_c_floorplans():
    registry = _get_registry()
    return jsonify(
        {
            "count": registry.count,
            "summary": registry.aggregate_summary(),
            "floorplans": registry.list_metadata(),
        }
    )


@stage_c_bp.get("/api/stage-c/floorplan/<style>/<floorplan>")
def api_stage_c_floorplan_detail(style: str, floorplan: str):
    registry = _get_registry()
    full_name = f"{style}/{floorplan}"
    try:
        entry = registry.get(full_name)
    except KeyError:
        return (
            jsonify(
                {
                    "error": f"Floorplan '{full_name}' not found among the 23 calibrated Stage C floorplans.",
                    "code": "floorplan_not_found",
                }
            ),
            404,
        )
    data = entry.to_metadata_dict()
    data["presets"] = _build_presets(entry)
    return jsonify(data)


@stage_c_bp.post("/api/stage-c/predict")
def api_stage_c_predict():
    if request.content_length is not None and request.content_length > MAX_STAGE_C_PAYLOAD_BYTES:
        return (
            jsonify(
                {
                    "error": f"Request payload exceeds {MAX_STAGE_C_PAYLOAD_BYTES} bytes.",
                    "code": "payload_too_large",
                }
            ),
            413,
        )

    payload = request.get_json(silent=True)
    if payload is None:
        return (
            jsonify(
                {
                    "error": "Request body must be valid JSON.",
                    "code": "invalid_json",
                }
            ),
            400,
        )

    registry = _get_registry()
    floorplan_name = str(payload.get("floorplan", "")).strip()
    if not floorplan_name and "style" in payload and "floorplan_id" in payload:
        floorplan_name = f"{payload['style']}/{payload['floorplan_id']}"

    try:
        entry = registry.get(floorplan_name)
    except KeyError:
        return (
            jsonify(
                {
                    "error": f"Floorplan '{floorplan_name}' not found among the 23 calibrated Stage C floorplans.",
                    "code": "floorplan_not_found",
                }
            ),
            404,
        )

    core_types_tuple = tuple(b.core_type for b in entry.spec.active_cores)
    try:
        validated = validate_and_build_power_matrix(
            payload=payload,
            core_names=entry.spec.active_core_names,
            core_types=core_types_tuple,
        )
    except StageCValidationError as err:
        return (
            jsonify(
                {
                    "error": err.message,
                    "code": err.code,
                    "details": err.details,
                }
            ),
            400,
        )

    # Pure frozen inference timing (zero I/O, zero recalibration)
    t0_ns = time.perf_counter_ns()
    preds_c = entry.model.predict_sequence_c(
        validated.power_matrix_w,
        init_temp_c=validated.init_temp_c,
    )
    t1_ns = time.perf_counter_ns()
    live_inference_us = (t1_ns - t0_ns) / 1000.0

    num_steps = validated.num_steps
    dt = validated.dt_seconds
    # Step timestamps: k = 1..num_steps -> 0.01..2.00 s; also prepend t = 0.00 s (45.0 °C) for plotting
    step_times_s = [round((k + 1) * dt, 4) for k in range(num_steps)]
    plot_times_s = [0.0] + step_times_s

    core_names = list(entry.spec.active_core_names)
    core_types = list(core_types_tuple)

    series_by_core: Dict[str, List[float]] = {}
    plot_series_by_core: Dict[str, List[float]] = {}
    power_series_by_core: Dict[str, List[float]] = {}
    per_core_summary: List[Dict[str, Any]] = []

    overall_peak_c = -1e9
    overall_peak_core = core_names[0]
    overall_peak_time_s = 0.0

    for idx, (c_name, c_type) in enumerate(zip(core_names, core_types)):
        col_t = preds_c[:, idx]
        col_p = validated.power_matrix_w[:, idx]
        peak_idx = int(np.argmax(col_t))
        peak_c = float(col_t[peak_idx])
        final_c = float(col_t[-1])
        peak_t_s = step_times_s[peak_idx]

        if peak_c > overall_peak_c:
            overall_peak_c = peak_c
            overall_peak_core = c_name
            overall_peak_time_s = peak_t_s

        t_list = [round(float(v), 6) for v in col_t]
        p_list = [round(float(v), 4) for v in col_p]
        series_by_core[c_name] = t_list
        plot_series_by_core[c_name] = [round(validated.init_temp_c, 6)] + t_list
        power_series_by_core[c_name] = p_list

        per_core_summary.append(
            {
                "index": idx,
                "core_name": c_name,
                "core_type": c_type,
                "min_power_w": round(float(np.min(col_p)), 4),
                "max_power_w": round(float(np.max(col_p)), 4),
                "mean_power_w": round(float(np.mean(col_p)), 4),
                "peak_temp_c": round(peak_c, 4),
                "peak_time_s": peak_t_s,
                "final_temp_c": round(final_c, 4),
                "delta_final_c": round(final_c - validated.init_temp_c, 4),
            }
        )

    meta = entry.to_metadata_dict()
    return jsonify(
        {
            "floorplan": entry.full_name,
            "style": entry.style,
            "floorplan_id": entry.floorplan,
            "mode": validated.mode,
            "num_steps": num_steps,
            "dt_seconds": dt,
            "horizon_seconds": round(num_steps * dt, 4),
            "init_temp_c": validated.init_temp_c,
            "core_names": core_names,
            "core_types": core_types,
            "domain_status": validated.domain_status,
            "extrapolation_warnings": validated.extrapolation_warnings,
            "calibrated_domain": {
                "big_core_power_w": list(VALIDATED_BIG_RANGE_W),
                "little_core_power_w": list(VALIDATED_LITTLE_RANGE_W),
            },
            "step_times_s": step_times_s,
            "plot_times_s": plot_times_s,
            "temperatures_c": series_by_core,
            "plot_temperatures_c": plot_series_by_core,
            "powers_w": power_series_by_core,
            "per_core_summary": per_core_summary,
            "overall_summary": {
                "hottest_core": overall_peak_core,
                "peak_temp_c": round(overall_peak_c, 4),
                "peak_time_s": overall_peak_time_s,
                "final_max_temp_c": round(max(s["final_temp_c"] for s in per_core_summary), 4),
            },
            "accuracy": meta["accuracy"],
            "model_footprint": {
                **meta["model_footprint"],
                "live_inference_time_us": round(live_inference_us, 3),
            },
        }
    )


def register_stage_c(app: Flask, registry: StageCRegistry | None = None) -> StageCRegistry:
    """Register the isolated Stage C Blueprint onto any Flask app without touching existing routes."""
    reg = registry if registry is not None else StageCRegistry()
    app.extensions["stage_c_registry"] = reg
    if "stage_c_web" not in app.blueprints:
        app.register_blueprint(stage_c_bp)
    return reg
