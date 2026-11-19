"""
Production Stage B v2 Model Loader (Portable, Pickle-Free NumPy .npz + JSON).

Loads the frozen Stage B v2 shared per-core Ridge regression model (alpha=0.01,
63 physics-informed features) from `stage_b/models/v2_ridge/model_v2.npz` and
`metadata.json` without pickle, without joblib, and without importing any
experimental training code.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple
import numpy as np
from .features_v2 import build_per_core_features, PRODUCTION_GROUPS

V2_MODEL_DIR = Path(__file__).resolve().parents[1] / "models/v2_ridge"


@dataclass(frozen=True)
class StageBV2ProductionModel:
    coef: np.ndarray        # shape (63,) float64
    intercept: float
    x_mean: np.ndarray      # shape (63,) float64
    x_scale: np.ndarray     # shape (63,) float64
    y_mean: float
    y_scale: float
    feature_groups: Tuple[str, ...]
    feature_names: Tuple[str, ...]
    metadata: Dict[str, Any]
    artifact_bytes: int
    artifact_sha256: str

    def build_features(
        self, geometry_mm: np.ndarray, powers_w: np.ndarray, die_mm: np.ndarray
    ) -> np.ndarray:
        X, defs = build_per_core_features(
            geometry_mm, powers_w, die_mm, groups=self.feature_groups
        )
        if tuple(d["name"] for d in defs) != self.feature_names:
            raise RuntimeError("Engineered v2 feature ordering differs from production metadata")
        return X

    def predict_from_features(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=np.float64)
        if X.ndim != 2 or X.shape[1] != len(self.coef) or not len(X) or len(X) % 8 or not np.isfinite(X).all():
            raise ValueError("Expected finite v2 features with shape (N*8,63)")
        with np.errstate(over='raise', invalid='raise', divide='raise'):
            Xs = (X - self.x_mean) / self.x_scale
            pred_scaled = Xs @ self.coef + self.intercept
            result = (pred_scaled * self.y_scale + self.y_mean).reshape(-1, 8)
        if not np.isfinite(result).all():
            raise RuntimeError("Nonfinite Stage B v2 prediction")
        return result

    def predict(
        self, geometry_mm: np.ndarray, powers_w: np.ndarray, die_mm: np.ndarray
    ) -> np.ndarray:
        X = self.build_features(geometry_mm, powers_w, die_mm)
        return self.predict_from_features(X)


def load_v2_production_model(model_dir: Path | str = V2_MODEL_DIR) -> StageBV2ProductionModel:
    """Load and integrity-check the portable Stage B v2 production artifact."""
    root = Path(model_dir).resolve()
    meta_path = root / "metadata.json"
    npz_path = root / "model_v2.npz"
    try:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        raw_bytes = npz_path.read_bytes()
    except (OSError, ValueError) as exc:
        raise RuntimeError("Cannot read Stage B v2 model/metadata artifacts") from exc
    actual_sha = hashlib.sha256(raw_bytes).hexdigest()
    expected_sha = meta["artifact"]["npz_sha256"]
    if actual_sha != expected_sha:
        raise RuntimeError(
            f"Stage B v2 production artifact SHA-256 mismatch: {actual_sha} != {expected_sha}"
        )

    with np.load(npz_path, allow_pickle=False) as data:
        coef = np.asarray(data["coef"], dtype=np.float64)
        intercept = float(np.asarray(data["intercept"], dtype=np.float64).reshape(()))
        x_mean = np.asarray(data["x_mean"], dtype=np.float64)
        x_scale = np.asarray(data["x_scale"], dtype=np.float64)
        y_mean = float(np.asarray(data["y_mean"], dtype=np.float64).reshape(()))
        y_scale = float(np.asarray(data["y_scale"], dtype=np.float64).reshape(()))

    feature_names = tuple(meta["feature_names"])
    if (len(feature_names) != 63 or not all(isinstance(n, str) and n for n in feature_names)
            or len(set(feature_names)) != 63
            or tuple(meta["feature_groups"]) != PRODUCTION_GROUPS
            or meta["units"] != {"geometry": "mm", "power": "W", "temperature": "C"}
            or meta["ambient_c"] != 45.0):
        raise RuntimeError("Unsupported Stage B v2 feature/unit/ambient metadata")
    if coef.shape != (len(feature_names),) or x_mean.shape != coef.shape or x_scale.shape != coef.shape:
        raise RuntimeError("Stage B v2 production weight/scaler shapes mismatch metadata")
    if not (np.isfinite(coef).all() and np.isfinite(x_mean).all() and np.isfinite(x_scale).all()):
        raise RuntimeError("Nonfinite parameter in Stage B v2 production artifact")
    if not np.isfinite([intercept, y_mean, y_scale]).all() or np.any(x_scale <= 0) or y_scale <= 0:
        raise RuntimeError("Invalid finite/positive Stage B v2 scalar or scaling parameter")

    return StageBV2ProductionModel(
        coef=coef,
        intercept=intercept,
        x_mean=x_mean,
        x_scale=x_scale,
        y_mean=y_mean,
        y_scale=y_scale,
        feature_groups=tuple(meta["feature_groups"]),
        feature_names=feature_names,
        metadata=meta,
        artifact_bytes=len(raw_bytes),
        artifact_sha256=actual_sha,
    )
