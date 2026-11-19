#!/usr/bin/env python3
"""
Export audited Stage B v2 experimental candidate (`model_v2.joblib`) into a
portable, pickle-free production artifact (`model_v2.npz` + `metadata.json`)
and verify exact floating-point prediction equivalence (< 1e-12 C).

Does NOT retrain the model and does NOT modify the experimental artifact.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

from joblib.numpy_pickle import NumpyUnpickler
import numpy as np

OUT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = OUT_DIR.parents[2]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from stage_b.ml.data import CORE_IDS, load_csv  # noqa: E402
from stage_b.ml.features_v2 import build_per_core_features  # noqa: E402
from stage_b.ml.v2_model import load_v2_production_model  # noqa: E402

EXP_MODEL_PATH = PROJECT_DIR / "stage_b/experiments/stage_b_v2/candidate_model/model_v2.joblib"
EXP_EXPECTED_SHA = "e2c39c6d1ca55351e67e051b5e35f2c17a107389fd6102b12badb0b22446507a"


@dataclass
class _ExpBundleShim:
    estimator: object
    x_mean: np.ndarray
    x_scale: np.ndarray
    y_mean: float
    y_scale: float
    feature_groups: Tuple[str, ...]
    feature_names: List[str]
    feature_version: int
    ambient_c: float
    model_name: str
    hyperparameters: Dict[str, Any]
    source_csv_sha256: str
    seed: int


def load_experimental_bundle(path: Path) -> _ExpBundleShim:
    class _Unpickler(NumpyUnpickler):
        def find_class(self, module: str, name: str):
            if module in ("__main__", "run_experiments") and name == "SharedCoreModelBundle":
                return _ExpBundleShim
            return super().find_class(module, name)

    with path.open("rb") as f:
        return _Unpickler(str(path), f, ensure_native_byte_order="auto").load()


def parse_flp(path: Path) -> Tuple[np.ndarray, np.ndarray]:
    blocks, cores = [], {}
    for line in path.read_text(encoding="utf-8").splitlines():
        toks = line.split("#", 1)[0].split()
        if not toks:
            continue
        vals = [float(x) * 1000.0 for x in toks[1:5]]
        blocks.append((toks[0], vals))
        if toks[0].startswith("Core_"):
            cores[toks[0]] = vals
    die = np.array([max(v[0] + v[2] for _, v in blocks), max(v[1] + v[3] for _, v in blocks)], dtype=float)
    g = np.array([cores[n] for n in CORE_IDS], dtype=float)
    return g, die


def parse_ptrace(path: Path) -> np.ndarray:
    lines = [l.split() for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]
    pmap = {k: float(v) for k, v in zip(lines[0], lines[1])}
    return np.array([pmap[n] for n in CORE_IDS], dtype=float)


def main() -> None:
    exp_sha = hashlib.sha256(EXP_MODEL_PATH.read_bytes()).hexdigest()
    assert exp_sha == EXP_EXPECTED_SHA, f"Experimental SHA mismatch: {exp_sha}"
    exp = load_experimental_bundle(EXP_MODEL_PATH)

    coef = np.asarray(exp.estimator.coef_, dtype=np.float64).reshape(-1)
    intercept = np.asarray(float(exp.estimator.intercept_), dtype=np.float64)
    x_mean = np.asarray(exp.x_mean, dtype=np.float64).reshape(-1)
    x_scale = np.asarray(exp.x_scale, dtype=np.float64).reshape(-1)
    y_mean = np.asarray(float(exp.y_mean), dtype=np.float64)
    y_scale = np.asarray(float(exp.y_scale), dtype=np.float64)

    npz_path = OUT_DIR / "model_v2.npz"
    np.savez(
        npz_path,
        coef=coef,
        intercept=intercept,
        x_mean=x_mean,
        x_scale=x_scale,
        y_mean=y_mean,
        y_scale=y_scale,
    )
    npz_bytes = npz_path.stat().st_size
    npz_sha = hashlib.sha256(npz_path.read_bytes()).hexdigest()

    v1_manifest = json.loads((PROJECT_DIR / "stage_b/models/final_lasso/manifest.json").read_text(encoding="utf-8"))
    metadata = {
        "model_version": "v2",
        "name": "Ridge Regression (Stage B v2 · Shared Physics)",
        "algorithm": "Ridge",
        "hyperparameters": {"alpha": float(exp.estimator.alpha)},
        "feature_version": int(exp.feature_version),
        "feature_set": "F4_plus_boundary_images",
        "feature_groups": list(exp.feature_groups),
        "feature_count": int(len(exp.feature_names)),
        "feature_names": list(exp.feature_names),
        "core_order": list(CORE_IDS),
        "ambient_c": float(exp.ambient_c),
        "units": {"geometry": "mm", "power": "W", "temperature": "C"},
        "seed": int(exp.seed),
        "source_csv_sha256": exp.source_csv_sha256,
        "split_floorplans": v1_manifest["split_floorplans"],
        "split_counts": {"train_floorplans": 53, "validation_floorplans": 12, "test_floorplans": 12},
        "artifact": {
            "format": "numpy_npz_no_pickle",
            "filename": "model_v2.npz",
            "npz_size_bytes": int(npz_bytes),
            "npz_sha256": npz_sha,
            "raw_float64_parameter_bytes": int((coef.size + 1) * 8),
            "raw_float64_total_arrays_bytes": int((coef.size + 1 + x_mean.size + x_scale.size + 2) * 8),
            "experimental_source_file": "stage_b/experiments/stage_b_v2/candidate_model/model_v2.joblib",
            "experimental_source_bytes": int(EXP_MODEL_PATH.stat().st_size),
            "experimental_source_sha256": exp_sha,
        },
        "audited_metrics": {
            "validation_12fp_240wl": {"mae_c": 0.419293, "rmse_c": 0.572021, "max_err_c": 2.246520, "hottest_core_acc_pct": 82.92},
            "held_out_test_12fp_240wl": {"mae_c": 0.350588, "rmse_c": 0.445882, "max_err_c": 1.512353, "hottest_core_acc_pct": 84.17},
            "external_23fp_460wl": {"mae_c": 0.460754, "rmse_c": 0.643017, "max_err_c": 2.731624, "hottest_core_acc_pct": 79.78},
            "disclosure": "Held-out test and external 23-floorplan sets were reused across v1/v2 evaluations and are historical reference metrics, not blind benchmarks or guarantees.",
        },
    }
    meta_path = OUT_DIR / "metadata.json"
    meta_path.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    # ------------------------------------------------------------------
    # Step 3: Exact Equivalence Verification (Experimental vs Production)
    # ------------------------------------------------------------------
    prod = load_v2_production_model(OUT_DIR)

    def exp_predict(g: np.ndarray, p: np.ndarray, d: np.ndarray) -> np.ndarray:
        X, _ = build_per_core_features(g, p, d, groups=exp.feature_groups)
        Xs = (X - exp.x_mean) / exp.x_scale
        return (exp.estimator.predict(Xs) * exp.y_scale + exp.y_mean).reshape(-1, 8)

    schema = json.loads((PROJECT_DIR / "stage_b/models/final_lasso/schema.json").read_text(encoding="utf-8"))
    ds = load_csv(
        PROJECT_DIR / "stage_b/stage_b_training_dataset.csv",
        schema,
        PROJECT_DIR / "stage_b/dataset_v1/floorplan_manifest.json",
    )

    diffs = {}
    for split_name in ("validation", "test"):
        mask = np.isin(ds.floorplan_ids, v1_manifest["split_floorplans"][split_name])
        p_exp = exp_predict(ds.geometry_mm[mask], ds.powers_w[mask], ds.die_mm[mask])
        p_prod = prod.predict(ds.geometry_mm[mask], ds.powers_w[mask], ds.die_mm[mask])
        diffs[split_name] = float(np.max(np.abs(p_exp - p_prod)))

    # External 23 Stage A floorplans (460 workloads)
    ext_max = 0.0
    sa_base = PROJECT_DIR / "dataset/Sample_dataset/2D_floorplans"
    for fp_dir in sorted([p for p in sa_base.glob("*/*") if p.is_dir()]):
        g, die = parse_flp(fp_dir / f"{fp_dir.name}_layer_0.flp")
        for pt in sorted((fp_dir / "validation_samples/ptrace").glob("val_sample_*.ptrace")):
            p = parse_ptrace(pt)
            ext_max = max(ext_max, float(np.max(np.abs(exp_predict(g, p, die) - prod.predict(g, p, die)))))
    diffs["external_23fp_460wl"] = ext_max

    # Custom + 20 random valid 8-core configurations
    rng = np.random.default_rng(2026)
    rand_max = 0.0
    custom_geom = np.array([
        [4,4,0.5,0.5],[4,4,5.0,0.5],[2,2,9.5,0.5],[2,2,12.0,0.5],
        [2,2,0.5,5.0],[2,2,3.0,5.0],[2,2,5.5,5.0],[2,2,8.0,5.0],
    ], dtype=float)
    custom_die = np.array([15.0, 8.0], dtype=float)
    for _ in range(20):
        rp = np.array([rng.uniform(10.0, 20.0) if custom_geom[i, 0] == 4 else rng.uniform(5.0, 10.0) for i in range(8)])
        rand_max = max(rand_max, float(np.max(np.abs(exp_predict(custom_geom, rp, custom_die) - prod.predict(custom_geom, rp, custom_die)))))
    diffs["custom_and_random_20"] = rand_max

    # Permutation equivariance on real floorplan geometry (b_fp_006, distinct neighbor distances)
    perm_max = 0.0
    g_real = ds.geometry_mm[0]
    p_real = ds.powers_w[0]
    d_real = ds.die_mm[0]
    base_out = prod.predict(g_real, p_real, d_real)[0]
    for _ in range(5):
        perm = rng.permutation(8)
        perm_out = prod.predict(g_real[perm], p_real[perm], d_real)[0]
        perm_max = max(perm_max, float(np.max(np.abs(perm_out - base_out[perm]))))
    diffs["permutation_equivariance"] = perm_max

    print(f"EXPORTED: {npz_path} ({npz_bytes} bytes, sha256={npz_sha})")
    for k, v in diffs.items():
        print(f"  EQUIV [{k:24s}]: max_abs_diff = {v:.3e} C")
        assert v < 1e-12, f"Equivalence check failed for {k}: {v}"
    print("EQUIVALENCE_VERIFIED: ALL < 1e-12 C")


if __name__ == "__main__":
    main()
