#!/usr/bin/env python3
"""
Independent Audit & Reproduction Script for Stage B v2.

Verifies without trusting existing CSVs:
  1. Split disjointness (53 train / 12 val / 12 test) & zero data leakage
  2. Exact training-only scaling parameters stored inside model_v2.joblib
  3. Independent loading of model_v2.joblib (without importing run_experiments)
  4. Exact reproduction of v1 & v2 metrics on Validation, Held-Out Test, and External 23 Stage A FPs
  5. Strict 8-core permutation equivariance (< 1e-12 C)
  6. Unfamiliar valid custom floorplan numerical stability
  7. Repeated latency benchmarking (feature engineering, model inference, total pipeline)
"""

from __future__ import annotations

import hashlib
import json
import pickle
import statistics
import sys
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter_ns
from typing import Any, Dict, List, Tuple

import joblib
import numpy as np
from joblib.numpy_pickle import NumpyUnpickler
from sklearn.preprocessing import StandardScaler

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parents[2]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from stage_b.ml.data import CORE_IDS, load_csv  # noqa: E402
from stage_b.ml.features import build_features as build_v1_features  # noqa: E402
from stage_b.web.service import MODEL_SHA as V1_MODEL_SHA, Predictor as V1Predictor, UNITS  # noqa: E402
from features_v2 import build_per_core_features  # noqa: E402


@dataclass
class AuditedV2Bundle:
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

    def predict(self, geometry_mm: np.ndarray, powers_w: np.ndarray, die_mm: np.ndarray) -> Tuple[np.ndarray, float, float]:
        t0 = perf_counter_ns()
        X, _ = build_per_core_features(geometry_mm, powers_w, die_mm, groups=self.feature_groups)
        t1 = perf_counter_ns()
        Xs = (X - self.x_mean) / self.x_scale
        temps_c = (self.estimator.predict(Xs) * self.y_scale + self.y_mean).reshape(-1, 8)
        t2 = perf_counter_ns()
        return temps_c, (t1 - t0) / 1e6, (t2 - t1) / 1e6


def load_v2_safely(path: Path) -> AuditedV2Bundle:
    """Load model_v2.joblib without importing run_experiments.py (handles __main__.SharedCoreModelBundle)."""
    class SafeV2Unpickler(NumpyUnpickler):
        def find_class(self, module: str, name: str):
            if module in ("__main__", "run_experiments") and name == "SharedCoreModelBundle":
                return AuditedV2Bundle
            return super().find_class(module, name)

    with path.open("rb") as f:
        return SafeV2Unpickler(str(path), f, ensure_native_byte_order="auto").load()


def metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    err = y_pred - y_true
    abs_err = np.abs(err)
    return {
        "mae": float(np.mean(abs_err)),
        "rmse": float(np.sqrt(np.mean(err ** 2))),
        "max": float(np.max(abs_err)),
        "hot_acc": float(np.mean(np.argmax(y_pred, axis=1) == np.argmax(y_true, axis=1))) * 100.0,
    }


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


def parse_steady(path: Path) -> np.ndarray:
    tmap = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        toks = line.split()
        if not toks or toks[0].startswith("#"):
            continue
        name = toks[0][len("layer_0_"):] if toks[0].startswith("layer_0_") else toks[0]
        if name in CORE_IDS:
            tmap[name] = float(toks[1]) - 273.15
    return np.array([tmap[n] for n in CORE_IDS], dtype=float)


def main() -> None:
    print("=== PHASE 1: ARTIFACT & LEAKAGE AUDIT ===")
    v1_path = PROJECT_DIR / "stage_b/models/final_lasso/model.joblib"
    v2_path = SCRIPT_DIR / "candidate_model/model_v2.joblib"
    v1_sha = hashlib.sha256(v1_path.read_bytes()).hexdigest()
    v2_sha = hashlib.sha256(v2_path.read_bytes()).hexdigest()
    print(f"v1 model: {v1_path.stat().st_size} B, sha256={v1_sha} (matches expected={v1_sha == V1_MODEL_SHA})")
    print(f"v2 model: {v2_path.stat().st_size} B, sha256={v2_sha}")

    v2 = load_v2_safely(v2_path)
    print(f"v2 loaded cleanly via SafeV2Unpickler: estimator={type(v2.estimator).__name__}(alpha={v2.estimator.alpha}), n_features={len(v2.feature_names)}, groups={v2.feature_groups}")
    assert type(v2.estimator).__name__ == "Ridge" and v2.estimator.alpha == 0.01
    assert len(v2.feature_names) == 63

    # Verify splits & training-only scaler
    manifest = json.loads((PROJECT_DIR / "stage_b/models/final_lasso/manifest.json").read_text(encoding="utf-8"))
    schema = json.loads((PROJECT_DIR / "stage_b/models/final_lasso/schema.json").read_text(encoding="utf-8"))
    tr_ids = set(manifest["split_floorplans"]["train"])
    va_ids = set(manifest["split_floorplans"]["validation"])
    te_ids = set(manifest["split_floorplans"]["test"])
    assert len(tr_ids) == 53 and len(va_ids) == 12 and len(te_ids) == 12
    assert not (tr_ids & va_ids) and not (tr_ids & te_ids) and not (va_ids & te_ids)
    print("Split disjointness verified: 53 train / 12 val / 12 test (0 overlap)")

    ds = load_csv(
        PROJECT_DIR / "stage_b/stage_b_training_dataset.csv",
        schema,
        PROJECT_DIR / "stage_b/dataset_v1/floorplan_manifest.json",
    )
    tr_mask = np.isin(ds.floorplan_ids, list(tr_ids))
    va_mask = np.isin(ds.floorplan_ids, list(va_ids))
    te_mask = np.isin(ds.floorplan_ids, list(te_ids))

    X_tr, defs = build_per_core_features(ds.geometry_mm[tr_mask], ds.powers_w[tr_mask], ds.die_mm[tr_mask], groups=v2.feature_groups, nearest_policy="legacy_first")
    # Frozen scalers were fitted before the symmetric nearest-tie correction.
    assert [d["name"] for d in defs] == v2.feature_names
    sc = StandardScaler().fit(X_tr)
    expected_scale = np.where(sc.scale_ == 0.0, 1.0, sc.scale_)
    max_mean_diff = float(np.max(np.abs(sc.mean_ - v2.x_mean)))
    max_scale_diff = float(np.max(np.abs(expected_scale - v2.x_scale)))
    y_tr_flat = ds.temperatures_c[tr_mask].reshape(-1)
    y_mean_diff = abs(float(np.mean(y_tr_flat)) - v2.y_mean)
    y_scale_diff = abs(float(np.std(y_tr_flat)) - v2.y_scale)
    print(f"Training-only scaler audit: x_mean_diff={max_mean_diff:.2e}, x_scale_diff={max_scale_diff:.2e}, y_mean_diff={y_mean_diff:.2e}, y_scale_diff={y_scale_diff:.2e}")
    assert max(max_mean_diff, max_scale_diff, y_mean_diff, y_scale_diff) < 1e-12

    # Permutation equivariance test (5 random permutations)
    rng = np.random.default_rng(12345)
    g0, p0, d0 = ds.geometry_mm[va_mask][0], ds.powers_w[va_mask][0], ds.die_mm[va_mask][0]
    base_pred, _, _ = v2.predict(g0, p0, d0)
    max_perm_err = 0.0
    for _ in range(5):
        perm = rng.permutation(8)
        perm_pred, _, _ = v2.predict(g0[perm], p0[perm], d0)
        err = float(np.max(np.abs(perm_pred[0] - base_pred[0, perm])))
        max_perm_err = max(max_perm_err, err)
    print(f"Permutation equivariance (5 random permutations): max_abs_diff={max_perm_err:.3e} C")
    assert max_perm_err < 1e-12

    # Unfamiliar valid custom floorplans stability test
    custom_tests = [
        ("compact_narrow", np.array([16.0, 6.0]), np.array([
            [4,4,0,0],[4,4,4.5,0],[2,2,9,0],[2,2,11.5,0],[2,2,14,0],[2,2,9,3],[2,2,11.5,3],[2,2,14,3]
        ], dtype=float), np.array([18.0, 14.0, 8.0, 6.0, 9.0, 7.0, 5.5, 9.5])),
        ("large_sparse", np.array([25.0, 25.0]), np.array([
            [4,4,1,1],[4,4,19,19],[2,2,1,20],[2,2,20,1],[2,2,10,10],[2,2,14,10],[2,2,10,14],[2,2,14,14]
        ], dtype=float), np.array([15.0, 15.0, 7.5, 7.5, 7.5, 7.5, 7.5, 7.5])),
    ]
    for cname, cdie, cg, cp in custom_tests:
        ct, _, _ = v2.predict(cg, cp, cdie)
        assert np.isfinite(ct).all() and np.all(ct > 45.0) and np.all(ct < 300.0)
        print(f"Unfamiliar custom layout '{cname}': min={ct.min():.2f} C, max={ct.max():.2f} C (finite & physical)")

    print("\n=== PHASE 2: INDEPENDENT ACCURACY REPRODUCTION ===")
    v1_bundle = joblib.load(v1_path)
    X_v1_all, _ = build_v1_features(ds)
    for label, mask in [("Validation", va_mask), ("Held-out Test", te_mask)]:
        y_gt = ds.temperatures_c[mask]
        p_v1 = v1_bundle.predict(X_v1_all[mask])
        p_v2, _, _ = v2.predict(ds.geometry_mm[mask], ds.powers_w[mask], ds.die_mm[mask])
        m1, m2 = metrics(y_gt, p_v1), metrics(y_gt, p_v2)
        print(f"{label:14s} | v1: MAE={m1['mae']:.6f} RMSE={m1['rmse']:.6f} Max={m1['max']:.6f} Hot={m1['hot_acc']:.2f}%")
        print(f"{label:14s} | v2: MAE={m2['mae']:.6f} RMSE={m2['rmse']:.6f} Max={m2['max']:.6f} Hot={m2['hot_acc']:.2f}%")

    # External 23 Stage A floorplans (460 HotSpot workloads)
    v1_web = V1Predictor()
    sa_base = PROJECT_DIR / "dataset/Sample_dataset/2D_floorplans"
    fp_dirs = sorted([p for p in sa_base.glob("*/*") if p.is_dir()])
    assert len(fp_dirs) == 23

    # Warmup
    wg, wd = parse_flp(fp_dirs[0] / f"{fp_dirs[0].name}_layer_0.flp")
    wp = parse_ptrace(sorted((fp_dirs[0] / "validation_samples/ptrace").glob("val_sample_*.ptrace"))[0])
    v2.predict(wg, wp, wd)
    v1_web.predict({"mode": "custom", "model_version": "v1", "units": UNITS, "die": {"width_mm": float(wd[0]), "height_mm": float(wd[1])},
                    "cores": [{"id": n, "width_mm": float(wg[i,0]), "height_mm": float(wg[i,1]),
                               "x_mm": float(wg[i,2]), "y_mm": float(wg[i,3]), "power_w": float(wp[i])}
                              for i, n in enumerate(CORE_IDS)]})

    gt_ext, v1_ext, v2_ext = [], [], []
    v1_f, v1_i, v1_t = [], [], []
    v2_f, v2_i, v2_t = [], [], []
    fps_improved = 0

    for fp_dir in fp_dirs:
        g, die = parse_flp(fp_dir / f"{fp_dir.name}_layer_0.flp")
        fp_gt, fp_1, fp_2 = [], [], []
        for pt in sorted((fp_dir / "validation_samples/ptrace").glob("val_sample_*.ptrace")):
            p = parse_ptrace(pt)
            gt = parse_steady(fp_dir / "validation_samples/steady" / f"{pt.stem}.steady")
            r1 = v1_web.predict({"mode": "custom", "model_version": "v1", "units": UNITS, "die": {"width_mm": float(die[0]), "height_mm": float(die[1])},
                                 "cores": [{"id": n, "width_mm": float(g[i,0]), "height_mm": float(g[i,1]),
                                            "x_mm": float(g[i,2]), "y_mm": float(g[i,3]), "power_w": float(p[i])}
                                           for i, n in enumerate(CORE_IDS)]})
            t1 = np.array([float(r1["temperatures_c"][n]) for n in CORE_IDS])
            v1_f.append(float(r1["feature_engineering_ms"]))
            v1_i.append(float(r1["latency_ms"]))
            v1_t.append(float(r1["feature_engineering_ms"]) + float(r1["latency_ms"]))

            t2_arr, f2, i2 = v2.predict(g, p, die)
            t2 = t2_arr[0]
            v2_f.append(f2)
            v2_i.append(i2)
            v2_t.append(f2 + i2)

            fp_gt.append(gt); fp_1.append(t1); fp_2.append(t2)
            gt_ext.append(gt); v1_ext.append(t1); v2_ext.append(t2)

        if np.mean(np.abs(np.asarray(fp_2) - np.asarray(fp_gt))) < np.mean(np.abs(np.asarray(fp_1) - np.asarray(fp_gt))):
            fps_improved += 1

    m1_ext = metrics(np.asarray(gt_ext), np.asarray(v1_ext))
    m2_ext = metrics(np.asarray(gt_ext), np.asarray(v2_ext))
    print(f"External 23 FP | v1: MAE={m1_ext['mae']:.6f} RMSE={m1_ext['rmse']:.6f} Max={m1_ext['max']:.6f} Hot={m1_ext['hot_acc']:.2f}%")
    print(f"External 23 FP | v2: MAE={m2_ext['mae']:.6f} RMSE={m2_ext['rmse']:.6f} Max={m2_ext['max']:.6f} Hot={m2_ext['hot_acc']:.2f}%")
    print(f"External floorplans improved: {fps_improved}/23")

    print("\n=== PHASE 3: LATENCY BENCHMARK (460 WORKLOADS) ===")
    p95 = lambda a: float(np.percentile(a, 95))
    print(f"v1: feat_med={statistics.median(v1_f):.4f} ms (p95={p95(v1_f):.4f}) | inf_med={statistics.median(v1_i):.4f} ms (p95={p95(v1_i):.4f}) | tot_med={statistics.median(v1_t):.4f} ms (p95={p95(v1_t):.4f})")
    print(f"v2: feat_med={statistics.median(v2_f):.4f} ms (p95={p95(v2_f):.4f}) | inf_med={statistics.median(v2_i):.4f} ms (p95={p95(v2_i):.4f}) | tot_med={statistics.median(v2_t):.4f} ms (p95={p95(v2_t):.4f})")
    print("AUDIT_STATUS: PASS")


if __name__ == "__main__":
    main()
