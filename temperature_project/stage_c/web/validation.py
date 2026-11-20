"""
Input validation and calibrated-domain vs. extrapolation classification for Stage C.

Calibrated Stage C workload domain (Workloads A, B, and held-out Workload C):
  - Big cores:    10.0 W to 20.0 W
  - Little cores:  5.0 W to 10.0 W
  - Horizon:       200 steps at dt = 0.01 s (2.00 s total)
  - Initial temp:  45.0 °C (ambient)

Hard physical rejection bounds (HTTP 400):
  - Non-finite (NaN / Inf) or non-numeric values
  - Negative power (< 0.0 W) or extreme non-physical power (> 50.0 W)
  - Invalid step count (< 1 or > 200) or non-0.01 s dt
  - Mismatched core names or wrong core count (!= 8)
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np

HARD_MIN_POWER_W: float = 0.0
HARD_MAX_POWER_W: float = 50.0

VALIDATED_BIG_RANGE_W: Tuple[float, float] = (10.0, 20.0)
VALIDATED_LITTLE_RANGE_W: Tuple[float, float] = (5.0, 10.0)

DEFAULT_NUM_STEPS: int = 200
MAX_NUM_STEPS: int = 200
EXPECTED_DT_SECONDS: float = 0.01
EXPECTED_INIT_TEMP_C: float = 45.0


class StageCValidationError(ValueError):
    """Structured validation error convertible to HTTP 400 JSON."""

    def __init__(self, message: str, code: str = "invalid_input", details: Dict[str, Any] | None = None):
        super().__init__(message)
        self.message = message
        self.code = code
        self.details = details or {}


@dataclass(frozen=True)
class ValidatedPredictionRequest:
    mode: str
    num_steps: int
    dt_seconds: float
    init_temp_c: float
    power_matrix_w: np.ndarray  # shape (num_steps, 8), float64
    domain_status: str          # "validated" or "extrapolated"
    extrapolation_warnings: List[Dict[str, Any]]


def _parse_finite_power(value: Any, label: str) -> float:
    if isinstance(value, bool) or value is None:
        raise StageCValidationError(
            f"Power value for {label} must be a finite number.",
            code="invalid_power_type",
            details={"field": label},
        )
    try:
        val = float(value)
    except (TypeError, ValueError) as exc:
        raise StageCValidationError(
            f"Power value for {label} must be a finite number.",
            code="invalid_power_type",
            details={"field": label},
        ) from exc

    if not math.isfinite(val):
        raise StageCValidationError(
            f"Power value for {label} must be finite (not NaN or Inf).",
            code="non_finite_power",
            details={"field": label},
        )
    if val < HARD_MIN_POWER_W or val > HARD_MAX_POWER_W:
        raise StageCValidationError(
            f"Power value for {label} ({val} W) is outside physical bounds "
            f"[{HARD_MIN_POWER_W}, {HARD_MAX_POWER_W}] W.",
            code="power_out_of_physical_bounds",
            details={
                "field": label,
                "value_w": val,
                "min_w": HARD_MIN_POWER_W,
                "max_w": HARD_MAX_POWER_W,
            },
        )
    return val


def _parse_8core_vector(
    container: Dict[str, Any],
    core_names: Sequence[str],
    context: str,
) -> np.ndarray:
    """Parse an 8-core power vector from either 'powers_by_core' (dict) or 'powers_w' (list)."""
    n = len(core_names)
    if "powers_by_core" in container and container["powers_by_core"] is not None:
        raw_map = container["powers_by_core"]
        if not isinstance(raw_map, dict):
            raise StageCValidationError(
                f"{context}: 'powers_by_core' must be a JSON object mapping core names to watts.",
                code="invalid_powers_by_core",
            )
        expected_set = set(core_names)
        provided_set = set(raw_map.keys())
        missing = sorted(expected_set - provided_set)
        extra = sorted(provided_set - expected_set)
        if missing or extra:
            raise StageCValidationError(
                f"{context}: core keys must match exact floorplan cores.",
                code="core_key_mismatch",
                details={"missing": missing, "extra": extra, "expected": list(core_names)},
            )
        return np.array(
            [_parse_finite_power(raw_map[c], f"{context}.{c}") for c in core_names],
            dtype=np.float64,
        )

    if "powers_w" in container and container["powers_w"] is not None:
        raw_list = container["powers_w"]
        if not isinstance(raw_list, list) or len(raw_list) != n:
            raise StageCValidationError(
                f"{context}: 'powers_w' must be a list of exactly {n} numbers.",
                code="invalid_powers_length",
                details={"expected_length": n},
            )
        return np.array(
            [_parse_finite_power(v, f"{context}.{core_names[i]}") for i, v in enumerate(raw_list)],
            dtype=np.float64,
        )

    raise StageCValidationError(
        f"{context}: must provide either 'powers_by_core' or 'powers_w'.",
        code="missing_powers",
    )


def validate_and_build_power_matrix(
    payload: Dict[str, Any],
    core_names: Sequence[str],
    core_types: Sequence[str],
) -> ValidatedPredictionRequest:
    """Validate incoming Stage C prediction payload and classify validated vs. extrapolated domain."""
    if not isinstance(payload, dict):
        raise StageCValidationError("Request body must be a JSON object.", code="invalid_json")

    mode = str(payload.get("mode", "constant")).strip().lower()
    if mode not in {"constant", "schedule", "trace"}:
        raise StageCValidationError(
            f"Unsupported mode '{mode}'. Must be one of: constant, schedule, trace.",
            code="invalid_mode",
        )

    raw_steps = payload.get("num_steps", DEFAULT_NUM_STEPS)
    if isinstance(raw_steps, bool) or not isinstance(raw_steps, int):
        raise StageCValidationError("'num_steps' must be an integer.", code="invalid_num_steps")
    if raw_steps < 1 or raw_steps > MAX_NUM_STEPS:
        raise StageCValidationError(
            f"'num_steps' must be between 1 and {MAX_NUM_STEPS} (got {raw_steps}).",
            code="num_steps_out_of_range",
        )
    num_steps = raw_steps

    raw_dt = payload.get("dt_seconds", EXPECTED_DT_SECONDS)
    try:
        dt_seconds = float(raw_dt)
    except (TypeError, ValueError) as exc:
        raise StageCValidationError("'dt_seconds' must be a number.", code="invalid_dt") from exc
    if not math.isclose(dt_seconds, EXPECTED_DT_SECONDS, abs_tol=1e-9):
        raise StageCValidationError(
            f"Calibrated Stage C models require dt_seconds = {EXPECTED_DT_SECONDS} s (got {dt_seconds}).",
            code="unsupported_dt",
        )

    raw_init = payload.get("init_temp_c", EXPECTED_INIT_TEMP_C)
    try:
        init_temp_c = float(raw_init)
    except (TypeError, ValueError) as exc:
        raise StageCValidationError("'init_temp_c' must be a number.", code="invalid_init_temp") from exc
    if not math.isclose(init_temp_c, EXPECTED_INIT_TEMP_C, abs_tol=1e-9):
        raise StageCValidationError(
            f"Frozen two-speed models require init_temp_c = {EXPECTED_INIT_TEMP_C} °C (ambient).",
            code="unsupported_init_temp",
        )

    n_cores = len(core_names)

    if mode == "constant":
        vec = _parse_8core_vector(payload, core_names, "constant")
        power_matrix = np.tile(vec, (num_steps, 1))

    elif mode == "schedule":
        segments = payload.get("segments")
        if not isinstance(segments, list) or len(segments) == 0:
            raise StageCValidationError(
                "Schedule mode requires a non-empty 'segments' list.",
                code="missing_segments",
            )
        if len(segments) > MAX_NUM_STEPS:
            raise StageCValidationError(
                f"Too many schedule segments ({len(segments)} > {MAX_NUM_STEPS}).",
                code="too_many_segments",
            )
        power_matrix = np.empty((num_steps, n_cores), dtype=np.float64)
        covered = np.zeros(num_steps, dtype=bool)
        for seg_idx, seg in enumerate(segments):
            if not isinstance(seg, dict):
                raise StageCValidationError(
                    f"Segment {seg_idx} must be a JSON object.",
                    code="invalid_segment",
                )
            s_start = seg.get("start_step")
            s_end = seg.get("end_step")
            if (
                isinstance(s_start, bool)
                or isinstance(s_end, bool)
                or not isinstance(s_start, int)
                or not isinstance(s_end, int)
            ):
                raise StageCValidationError(
                    f"Segment {seg_idx} start_step and end_step must be 1-based integers.",
                    code="invalid_segment_bounds",
                )
            if s_start < 1 or s_end > num_steps or s_start > s_end:
                raise StageCValidationError(
                    f"Segment {seg_idx} bounds [{s_start}, {s_end}] are invalid for num_steps={num_steps}.",
                    code="segment_out_of_range",
                )
            if np.any(covered[s_start - 1 : s_end]):
                raise StageCValidationError(
                    f"Segment {seg_idx} [{s_start}, {s_end}] overlaps a previous segment.",
                    code="overlapping_segments",
                )
            vec = _parse_8core_vector(seg, core_names, f"segments[{seg_idx}]")
            power_matrix[s_start - 1 : s_end, :] = vec
            covered[s_start - 1 : s_end] = True

        if not np.all(covered):
            missing_steps = [int(i + 1) for i in np.where(~covered)[0][:10]]
            raise StageCValidationError(
                f"Schedule segments must cover all steps 1..{num_steps} without gaps.",
                code="incomplete_schedule",
                details={"first_missing_steps": missing_steps},
            )

    else:  # mode == "trace"
        raw_trace = payload.get("power_trace_w")
        if not isinstance(raw_trace, list) or len(raw_trace) == 0:
            raise StageCValidationError(
                "Trace mode requires a non-empty 2D 'power_trace_w' list.",
                code="missing_trace",
            )
        if len(raw_trace) > MAX_NUM_STEPS:
            raise StageCValidationError(
                f"Trace exceeds maximum length of {MAX_NUM_STEPS} steps (got {len(raw_trace)}).",
                code="trace_too_long",
            )
        if "num_steps" in payload and len(raw_trace) != num_steps:
            raise StageCValidationError(
                f"Trace length ({len(raw_trace)}) does not match num_steps ({num_steps}).",
                code="trace_length_mismatch",
            )
        num_steps = len(raw_trace)
        power_matrix = np.empty((num_steps, n_cores), dtype=np.float64)
        for step_idx, row in enumerate(raw_trace):
            if not isinstance(row, list) or len(row) != n_cores:
                raise StageCValidationError(
                    f"Trace row {step_idx + 1} must be a list of exactly {n_cores} numbers.",
                    code="invalid_trace_row",
                )
            for c_idx, val in enumerate(row):
                power_matrix[step_idx, c_idx] = _parse_finite_power(
                    val, f"trace[step={step_idx + 1}].{core_names[c_idx]}"
                )

    # Classify validated calibrated domain vs. extrapolation
    warnings: List[Dict[str, Any]] = []
    for c_idx, (c_name, c_type) in enumerate(zip(core_names, core_types)):
        lo, hi = VALIDATED_BIG_RANGE_W if c_type == "Big" else VALIDATED_LITTLE_RANGE_W
        col = power_matrix[:, c_idx]
        c_min = float(np.min(col))
        c_max = float(np.max(col))
        if c_min < lo - 1e-9 or c_max > hi + 1e-9:
            warnings.append(
                {
                    "core_name": c_name,
                    "core_type": c_type,
                    "entered_min_w": round(c_min, 4),
                    "entered_max_w": round(c_max, 4),
                    "validated_min_w": lo,
                    "validated_max_w": hi,
                    "reason": (
                        f"{c_name} ({c_type}) power [{c_min:.2f}, {c_max:.2f}] W "
                        f"extends outside the calibrated {c_type} workload range [{lo:.1f}, {hi:.1f}] W."
                    ),
                }
            )

    if num_steps != DEFAULT_NUM_STEPS:
        warnings.append(
            {
                "field": "num_steps",
                "entered_steps": num_steps,
                "validated_steps": DEFAULT_NUM_STEPS,
                "reason": (
                    f"Horizon of {num_steps} steps ({num_steps * dt_seconds:.2f} s) differs from "
                    f"the full 200-step (2.00 s) audited validation horizon."
                ),
            }
        )

    domain_status = "validated" if len(warnings) == 0 else "extrapolated"

    return ValidatedPredictionRequest(
        mode=mode,
        num_steps=num_steps,
        dt_seconds=dt_seconds,
        init_temp_c=init_temp_c,
        power_matrix_w=power_matrix,
        domain_status=domain_status,
        extrapolation_warnings=warnings,
    )
