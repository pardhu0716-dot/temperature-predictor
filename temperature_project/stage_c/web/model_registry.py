"""
Read-only Stage C Model Registry for all 23 calibrated 2D floorplans.

Loads all 23 frozen two-speed transient models from their canonical cohort
locations without moving or modifying any existing Stage C artifacts, verifies
each frozen parameter JSON file's SHA-256 digest against the consolidated audit
manifest, and instantiates the audited GeneralizedTwoSpeedModel directly.
"""

from __future__ import annotations

import csv
import hashlib
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
STAGE_C_ROOT = PROJECT_ROOT / "stage_c" / "transient"
GENERALIZED_MODELS_DIR = STAGE_C_ROOT / "multi_floorplan_pilot" / "generalized_models"
CONSOLIDATED_SUMMARY_CSV = (
    STAGE_C_ROOT / "full_floorplan_expansion" / "consolidated" / "all_23_floorplans_summary.csv"
)

if str(GENERALIZED_MODELS_DIR) not in sys.path:
    sys.path.insert(0, str(GENERALIZED_MODELS_DIR))

from floorplan_metadata import FloorplanSpec, load_floorplan_spec  # noqa: E402
from transient_models import (  # noqa: E402
    AMBIENT_TEMP_C,
    DEFAULT_DT_SECONDS,
    GeneralizedTwoSpeedModel,
)


def compute_file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


def floorplan_tag(full_name: str) -> str:
    """Convert 'Style_2/fp_005' -> 'style2_fp005'."""
    style, fp = full_name.split("/")
    style_num = style.replace("Style_", "")
    fp_num = fp.replace("fp_", "")
    return f"style{style_num}_fp{fp_num}"


def resolve_two_speed_param_path(full_name: str, cohort: str, stage_c_root: Path = STAGE_C_ROOT) -> Path:
    """Resolve the canonical frozen two-speed JSON path for a floorplan without moving files."""
    if cohort == "original_pilot_c01d":
        return stage_c_root / "models" / "two_speed" / "parameters" / "case3_final_joint_AB.json"
    tag = floorplan_tag(full_name)
    if cohort == "step7_pilot":
        return stage_c_root / "multi_floorplan_pilot" / "calibrations" / f"{tag}_two_speed_joint_AB.json"
    if cohort.startswith("step8_batch"):
        return stage_c_root / "full_floorplan_expansion" / "calibrations" / f"{tag}_two_speed_joint_AB.json"
    raise ValueError(f"Unknown Stage C cohort '{cohort}' for floorplan '{full_name}'.")


@dataclass(frozen=True)
class RegisteredFloorplanModel:
    """Immutable bundle for one calibrated Stage C two-speed floorplan model."""

    full_name: str
    style: str
    floorplan: str
    tag: str
    cohort: str
    param_path: Path
    param_sha256: str
    sha256_verified: bool
    spec: FloorplanSpec
    model: GeneralizedTwoSpeedModel
    die_area_mm2: float
    num_passive_fill: int
    one_speed_unseen_rmse_c: float
    one_speed_unseen_mae_c: float
    one_speed_unseen_max_c: float
    two_speed_unseen_rmse_c: float
    two_speed_unseen_mae_c: float
    two_speed_unseen_max_c: float
    rmse_reduction_pct: float
    rmse_improvement_factor: float
    benchmark_median_200steps_us: float
    raw_bytes_with_r: int

    def to_metadata_dict(self) -> Dict[str, Any]:
        active_core_types = tuple(b.core_type for b in self.spec.active_cores)
        cores = [
            {
                "index": idx,
                "name": name,
                "core_type": ctype,
                "validated_power_min_w": 10.0 if ctype == "Big" else 5.0,
                "validated_power_max_w": 20.0 if ctype == "Big" else 10.0,
                "default_power_w": 15.0 if ctype == "Big" else 8.0,
            }
            for idx, (name, ctype) in enumerate(
                zip(self.spec.active_core_names, active_core_types)
            )
        ]
        return {
            "full_name": self.full_name,
            "style": self.style,
            "floorplan": self.floorplan,
            "tag": self.tag,
            "cohort": self.cohort,
            "num_active_cores": self.model.num_cores,
            "num_passive_fill": self.num_passive_fill,
            "die_area_mm2": self.die_area_mm2,
            "dt_seconds": self.model.dt_seconds,
            "ambient_temp_c": self.model.ambient_temp_c,
            "horizon_steps": 200,
            "horizon_seconds": round(200 * self.model.dt_seconds, 4),
            "cores": cores,
            "accuracy": {
                "two_speed_unseen_rmse_c": round(self.two_speed_unseen_rmse_c, 6),
                "two_speed_unseen_mae_c": round(self.two_speed_unseen_mae_c, 6),
                "two_speed_unseen_max_c": round(self.two_speed_unseen_max_c, 6),
                "one_speed_unseen_rmse_c": round(self.one_speed_unseen_rmse_c, 6),
                "one_speed_unseen_mae_c": round(self.one_speed_unseen_mae_c, 6),
                "one_speed_unseen_max_c": round(self.one_speed_unseen_max_c, 6),
                "rmse_reduction_pct": round(self.rmse_reduction_pct, 4),
                "rmse_improvement_factor": round(self.rmse_improvement_factor, 4),
            },
            "model_footprint": {
                "total_parameters": self.model.total_param_count,
                "steady_state_r_parameters": self.model.num_cores * self.model.num_cores,
                "transient_parameters": self.model.transient_param_count,
                "raw_float64_bytes": self.model.raw_float64_bytes,
                "benchmark_median_200steps_us": round(self.benchmark_median_200steps_us, 3),
                "param_sha256": self.param_sha256,
                "sha256_verified": self.sha256_verified,
            },
        }


class StageCRegistry:
    """Read-only in-memory registry of all 23 verified Stage C two-speed models."""

    def __init__(
        self,
        summary_csv: Path = CONSOLIDATED_SUMMARY_CSV,
        stage_c_root: Path = STAGE_C_ROOT,
        dataset_2d_root: Optional[Path] = None,
    ) -> None:
        self.summary_csv = Path(summary_csv)
        self.stage_c_root = Path(stage_c_root)
        self.dataset_2d_root = (
            Path(dataset_2d_root)
            if dataset_2d_root is not None
            else PROJECT_ROOT / "dataset" / "Sample_dataset" / "2D_floorplans"
        )
        self._models: Dict[str, RegisteredFloorplanModel] = {}
        self._ordered_names: List[str] = []
        self._load_and_verify()

    def _load_and_verify(self) -> None:
        if not self.summary_csv.is_file():
            raise FileNotFoundError(f"Stage C summary manifest not found: {self.summary_csv}")

        with self.summary_csv.open("r", encoding="utf-8", newline="") as f:
            rows = list(csv.DictReader(f))

        if len(rows) != 23:
            raise ValueError(f"Expected 23 floorplans in {self.summary_csv}, found {len(rows)}.")

        for row in rows:
            full_name = row["full_name"].strip()
            style, fp_id = full_name.split("/")
            cohort = row["cohort"].strip()
            expected_sha = row["two_speed_param_sha256"].strip()
            param_path = resolve_two_speed_param_path(full_name, cohort, self.stage_c_root)

            if not param_path.is_file():
                raise FileNotFoundError(
                    f"Frozen two-speed parameter file missing for {full_name}: {param_path}"
                )

            actual_sha = compute_file_sha256(param_path)
            if actual_sha != expected_sha:
                raise ValueError(
                    f"SHA-256 mismatch for {full_name} ({param_path}): "
                    f"expected {expected_sha}, got {actual_sha}."
                )

            with param_path.open("r", encoding="utf-8") as pf:
                payload = json.load(pf)

            spec = load_floorplan_spec(self.dataset_2d_root / style / fp_id, full_name)

            # Normalize minor schema difference between original Style_1/fp_001 and Step 7/8 cohorts
            json_core_names = tuple(
                payload.get("core_names_ordered") or payload.get("core_names") or ()
            )
            if json_core_names != spec.active_core_names:
                raise ValueError(
                    f"Core ordering mismatch for {full_name}: "
                    f"JSON {json_core_names} != .flp {spec.active_core_names}"
                )

            dt_seconds = float(payload.get("dt_seconds", DEFAULT_DT_SECONDS))
            ambient_temp_c = float(payload.get("ambient_temp_c", AMBIENT_TEMP_C))

            model = GeneralizedTwoSpeedModel.from_physical_params(
                r_matrix=spec.r_matrix,
                beta=np.asarray(payload["beta"], dtype=np.float64),
                tau_fast_seconds=np.asarray(payload["tau_fast_seconds"], dtype=np.float64),
                tau_slow_seconds=np.asarray(payload["tau_slow_seconds"], dtype=np.float64),
                core_names=spec.active_core_names,
                dt_seconds=dt_seconds,
                ambient_temp_c=ambient_temp_c,
            )

            expected_raw_bytes = int(row["two_speed_raw_bytes_with_R"])
            if model.raw_float64_bytes != expected_raw_bytes:
                raise ValueError(
                    f"Parameter byte-size mismatch for {full_name}: "
                    f"model={model.raw_float64_bytes}, manifest={expected_raw_bytes}"
                )

            entry = RegisteredFloorplanModel(
                full_name=full_name,
                style=style,
                floorplan=fp_id,
                tag=floorplan_tag(full_name),
                cohort=cohort,
                param_path=param_path,
                param_sha256=actual_sha,
                sha256_verified=True,
                spec=spec,
                model=model,
                die_area_mm2=float(row["die_area_mm2"]),
                num_passive_fill=int(row["num_passive_fill"]),
                one_speed_unseen_rmse_c=float(row["one_speed_unseen_rmse_c"]),
                one_speed_unseen_mae_c=float(row["one_speed_unseen_mae_c"]),
                one_speed_unseen_max_c=float(row["one_speed_unseen_max_c"]),
                two_speed_unseen_rmse_c=float(row["two_speed_unseen_rmse_c"]),
                two_speed_unseen_mae_c=float(row["two_speed_unseen_mae_c"]),
                two_speed_unseen_max_c=float(row["two_speed_unseen_max_c"]),
                rmse_reduction_pct=float(row["rmse_reduction_pct"]),
                rmse_improvement_factor=float(row["rmse_improvement_factor"]),
                benchmark_median_200steps_us=float(row["two_speed_median_200steps_us"]),
                raw_bytes_with_r=expected_raw_bytes,
            )
            self._models[full_name] = entry
            self._ordered_names.append(full_name)

    @property
    def count(self) -> int:
        return len(self._ordered_names)

    def list_names(self) -> List[str]:
        return list(self._ordered_names)

    def list_metadata(self) -> List[Dict[str, Any]]:
        return [self._models[name].to_metadata_dict() for name in self._ordered_names]

    def get(self, full_name: str) -> RegisteredFloorplanModel:
        if full_name not in self._models:
            raise KeyError(f"Unknown Stage C floorplan '{full_name}'.")
        return self._models[full_name]

    def aggregate_summary(self) -> Dict[str, Any]:
        entries = [self._models[n] for n in self._ordered_names]
        one_rmses = [e.one_speed_unseen_rmse_c for e in entries]
        two_rmses = [e.two_speed_unseen_rmse_c for e in entries]
        reductions = [e.rmse_reduction_pct for e in entries]
        latencies = [e.benchmark_median_200steps_us for e in entries]
        mean_one = float(np.mean(one_rmses))
        mean_two = float(np.mean(two_rmses))
        return {
            "floorplans_verified": len(entries),
            "all_sha256_verified": all(e.sha256_verified for e in entries),
            "mean_one_speed_unseen_rmse_c": round(mean_one, 6),
            "mean_two_speed_unseen_rmse_c": round(mean_two, 6),
            "mean_per_floorplan_rmse_reduction_pct": round(float(np.mean(reductions)), 4),
            "overall_mean_rmse_reduction_pct": round(100.0 * (mean_one - mean_two) / mean_one, 4),
            "median_200step_latency_us": round(float(np.median(latencies)), 3),
            "raw_float64_bytes_per_floorplan": 704,
            "total_parameters_per_floorplan": 88,
            "total_raw_float64_bytes_all_23": sum(e.raw_bytes_with_r for e in entries),
        }
