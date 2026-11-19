#!/usr/bin/env python3
"""
Phases 2 & 3 — Controlled Feature Ablation & Lightweight Model Comparison
for Stage B v2.

STRICT LEAKAGE PREVENTION:
  - Uses ONLY the original 53 training floorplans (1,060 workloads / 8,480 core rows)
    and 12 validation floorplans (240 workloads / 1,920 core rows).
  - NEVER loads or evaluates the 12 held-out Stage B test floorplans.
  - NEVER loads or evaluates the 23 external Stage A floorplans.

Outputs under stage_b/experiments/stage_b_v2/:
  - feature_ablation.csv
  - validation_results.csv
  - model_comparison.csv
  - candidate_model/ (frozen best candidate bundle + metadata)
"""

from __future__ import annotations

import csv
import hashlib
import io
import json
import statistics
import sys
import warnings
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter_ns
from typing import Any, Dict, List, Sequence, Tuple

import joblib
import numpy as np
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import ElasticNet, Lasso, LinearRegression, Ridge
from sklearn.preprocessing import StandardScaler

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parents[2]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from stage_b.ml.data import CORE_IDS, load_csv  # noqa: E402
from stage_b.ml.features import build_features as build_v1_features  # noqa: E402
from features_v2 import ALL_GROUPS, AMBIENT_C, VERSION as V2_FEATURE_VERSION, build_per_core_features  # noqa: E402


@dataclass
class SharedCoreModelBundle:
    """Self-contained, permutation-equivariant Stage B v2 model bundle."""
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

    def predict_temperatures_c(
        self,
        geometry_mm: np.ndarray,
        powers_w: np.ndarray,
        die_mm: np.ndarray,
    ) -> Tuple[np.ndarray, float, float]:
        """Return (temps_c shape (N, 8), feature_eng_ms, model_inference_ms)."""
        t0 = perf_counter_ns()
        X, _ = build_per_core_features(geometry_mm, powers_w, die_mm, groups=self.feature_groups)
        t1 = perf_counter_ns()
        Xs = (X - self.x_mean) / self.x_scale
        pred_scaled = self.estimator.predict(Xs)
        temps_c = (pred_scaled * self.y_scale + self.y_mean).reshape(-1, 8)
        t2 = perf_counter_ns()
        return temps_c, (t1 - t0) / 1e6, (t2 - t1) / 1e6


def eval_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """Compute MAE, RMSE, MaxErr, Centered MAE, and Hottest-Core Accuracy on (N, 8) arrays."""
    err = y_pred - y_true
    abs_err = np.abs(err)
    centered = err - np.mean(err, axis=1, keepdims=True)
    hot_acc = float(np.mean(np.argmax(y_pred, axis=1) == np.argmax(y_true, axis=1))) * 100.0
    return {
        "mae_c": float(np.mean(abs_err)),
        "rmse_c": float(np.sqrt(np.mean(err ** 2))),
        "max_err_c": float(np.max(abs_err)),
        "centered_mae_c": float(np.mean(np.abs(centered))),
        "hottest_core_acc_pct": hot_acc,
    }


def serialized_size_bytes(obj: object) -> int:
    buf = io.BytesIO()
    joblib.dump(obj, buf, compress=0)
    return buf.tell()


def fit_shared_model(
    X_tr: np.ndarray,
    y_tr: np.ndarray,
    X_va: np.ndarray,
    y_va: np.ndarray,
    kind: str,
    params: Dict[str, Any],
    seed: int = 42,
) -> Tuple[object, StandardScaler, float, float, Dict[str, float], Dict[str, float], bool]:
    """Fit a shared per-core model with training-only scaling and return train/val metrics."""
    scaler = StandardScaler().fit(X_tr)
    # Guard against any zero-variance feature
    scale = np.where(scaler.scale_ == 0.0, 1.0, scaler.scale_)
    X_tr_s = (X_tr - scaler.mean_) / scale
    X_va_s = (X_va - scaler.mean_) / scale

    y_tr_flat = y_tr.reshape(-1)
    y_mean = float(np.mean(y_tr_flat))
    y_scale = float(np.std(y_tr_flat))
    y_tr_s = (y_tr_flat - y_mean) / y_scale

    converged = True
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", ConvergenceWarning)
        if kind == "ols":
            est = LinearRegression(**params)
        elif kind == "ridge":
            est = Ridge(**params)
        elif kind == "lasso":
            est = Lasso(max_iter=30000, tol=1e-5, random_state=seed, **params)
        elif kind == "elastic_net":
            est = ElasticNet(max_iter=30000, tol=1e-5, random_state=seed, **params)
        else:
            raise ValueError(f"Unknown model kind: {kind}")
        est.fit(X_tr_s, y_tr_s)
        if any(issubclass(w.category, ConvergenceWarning) for w in caught):
            converged = False

    pred_tr = (est.predict(X_tr_s) * y_scale + y_mean).reshape(-1, 8)
    pred_va = (est.predict(X_va_s) * y_scale + y_mean).reshape(-1, 8)
    return est, scaler, y_mean, y_scale, eval_metrics(y_tr, pred_tr), eval_metrics(y_va, pred_va), converged


def save_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    model_dir = PROJECT_DIR / "stage_b/models/final_lasso"
    manifest = json.loads((model_dir / "manifest.json").read_text(encoding="utf-8"))
    schema = json.loads((model_dir / "schema.json").read_text(encoding="utf-8"))
    v1_bundle = joblib.load(model_dir / "model.joblib")

    # Load 77-floorplan dataset and strictly extract TRAIN + VALIDATION only
    ds = load_csv(
        PROJECT_DIR / "stage_b/stage_b_training_dataset.csv",
        schema,
        PROJECT_DIR / "stage_b/dataset_v1/floorplan_manifest.json",
    )
    train_ids = set(manifest["split_floorplans"]["train"])
    val_ids = set(manifest["split_floorplans"]["validation"])
    tr_mask = np.isin(ds.floorplan_ids, list(train_ids))
    va_mask = np.isin(ds.floorplan_ids, list(val_ids))

    assert int(tr_mask.sum()) == 1060 and int(va_mask.sum()) == 240
    assert len(train_ids & val_ids) == 0

    g_tr, p_tr, d_tr, y_tr = ds.geometry_mm[tr_mask], ds.powers_w[tr_mask], ds.die_mm[tr_mask], ds.temperatures_c[tr_mask]
    g_va, p_va, d_va, y_va = ds.geometry_mm[va_mask], ds.powers_w[va_mask], ds.die_mm[va_mask], ds.temperatures_c[va_mask]

    # Reference Stage B v1 baseline on exact validation split
    X_v1_all, _ = build_v1_features(ds)
    pred_v1_va = v1_bundle.predict(X_v1_all[va_mask])
    v1_va_metrics = eval_metrics(y_va, pred_v1_va)

    # ==================================================================
    # PHASE 2: Controlled Feature Ablation (Train 53 / Val 12 only)
    # ==================================================================
    ablation_specs: List[Tuple[str, Tuple[str, ...], str]] = [
        ("F1_self_global", ("self_global",), "Target core power/area/type + global chip power"),
        ("F2_plus_pairwise", ("self_global", "pairwise_coupling"), "+ Multi-scale 2D thermal coupling kernels"),
        ("F3_plus_die_spread", ("self_global", "pairwise_coupling", "die_spreading"), "+ Die spreading & aspect-ratio resistance"),
        ("F4_plus_boundary_images", ("self_global", "pairwise_coupling", "die_spreading", "boundary_images"), "+ Adiabatic die-wall method-of-images reflections"),
        ("F5_full_physics", ALL_GROUPS, "+ Local power concentration & contrast interactions"),
        ("F5_minus_die_spread", ("self_global", "pairwise_coupling", "boundary_images", "local_interactions"), "Leave-one-out: drop die_spreading"),
        ("F5_minus_boundary", ("self_global", "pairwise_coupling", "die_spreading", "local_interactions"), "Leave-one-out: drop boundary_images"),
        ("F5_minus_pairwise", ("self_global", "die_spreading", "boundary_images", "local_interactions"), "Leave-one-out: drop pairwise_coupling"),
    ]

    ablation_rows: List[Dict[str, Any]] = [
        {
            "feature_set": "F0_v1_baseline_252_slot",
            "groups": "v1_original_252",
            "description": "Stage B v1 baseline (8 separate Lasso alpha=0.001 models)",
            "num_features": 252,
            "reference_model": "v1_lasso_0.001",
            "train_mae_c": "",
            "val_mae_c": f"{v1_va_metrics['mae_c']:.6f}",
            "val_rmse_c": f"{v1_va_metrics['rmse_c']:.6f}",
            "val_max_err_c": f"{v1_va_metrics['max_err_c']:.6f}",
            "val_centered_mae_c": f"{v1_va_metrics['centered_mae_c']:.6f}",
            "val_hottest_core_acc_pct": f"{v1_va_metrics['hottest_core_acc_pct']:.2f}",
            "delta_val_mae_vs_v1_c": "0.000000",
        }
    ]

    best_feat_name = None
    best_feat_groups: Tuple[str, ...] = ALL_GROUPS
    best_feat_val_mae = float("inf")

    for name, groups, desc in ablation_specs:
        X_tr, defs = build_per_core_features(g_tr, p_tr, d_tr, groups=groups)
        X_va, _ = build_per_core_features(g_va, p_va, d_va, groups=groups)
        for ref_kind, ref_params, ref_label in [
            ("ridge", {"alpha": 1.0}, "shared_ridge_1.0"),
            ("lasso", {"alpha": 0.0001}, "shared_lasso_0.0001"),
        ]:
            _, _, _, _, tr_m, va_m, _ = fit_shared_model(X_tr, y_tr, X_va, y_va, ref_kind, ref_params)
            ablation_rows.append({
                "feature_set": name,
                "groups": "+".join(groups),
                "description": desc,
                "num_features": X_tr.shape[1],
                "reference_model": ref_label,
                "train_mae_c": f"{tr_m['mae_c']:.6f}",
                "val_mae_c": f"{va_m['mae_c']:.6f}",
                "val_rmse_c": f"{va_m['rmse_c']:.6f}",
                "val_max_err_c": f"{va_m['max_err_c']:.6f}",
                "val_centered_mae_c": f"{va_m['centered_mae_c']:.6f}",
                "val_hottest_core_acc_pct": f"{va_m['hottest_core_acc_pct']:.2f}",
                "delta_val_mae_vs_v1_c": f"{va_m['mae_c'] - v1_va_metrics['mae_c']:+.6f}",
            })
            if va_m["mae_c"] < best_feat_val_mae:
                best_feat_val_mae = va_m["mae_c"]
                best_feat_name = name
                best_feat_groups = groups

    save_csv(SCRIPT_DIR / "feature_ablation.csv", ablation_rows)
    print(f"PHASE 2 BEST FEATURE SET: {best_feat_name} ({'+'.join(best_feat_groups)}), val_mae={best_feat_val_mae:.6f} C")

    # ==================================================================
    # PHASE 3: Lightweight Model Sweep on Best Feature Set
    # ==================================================================
    X_tr, defs = build_per_core_features(g_tr, p_tr, d_tr, groups=best_feat_groups)
    X_va, _ = build_per_core_features(g_va, p_va, d_va, groups=best_feat_groups)
    feature_names = [d["name"] for d in defs]

    candidates: List[Tuple[str, str, Dict[str, Any]]] = [("ols", "ols", {})]
    for a in (1e-5, 1e-4, 1e-3, 1e-2, 0.1, 1.0, 10.0, 100.0):
        candidates.append((f"ridge_alpha_{a:g}", "ridge", {"alpha": a}))
    for a in (1e-5, 1e-4, 1e-3, 1e-2, 0.1, 1.0, 10.0):
        candidates.append((f"lasso_alpha_{a:g}", "lasso", {"alpha": a}))
    for a in (1e-5, 1e-4, 1e-3, 1e-2, 0.1, 1.0):
        for l1 in (0.2, 0.5, 0.8, 0.9, 0.95):
            candidates.append((f"enet_a{a:g}_l1_{l1:g}", "elastic_net", {"alpha": a, "l1_ratio": l1}))

    val_rows: List[Dict[str, Any]] = []
    best_by_family: Dict[str, Dict[str, Any]] = {}
    best_overall_record: Dict[str, Any] | None = None
    best_overall_bundle: SharedCoreModelBundle | None = None

    for cname, kind, params in candidates:
        est, scaler, y_mean, y_scale, tr_m, va_m, converged = fit_shared_model(
            X_tr, y_tr, X_va, y_va, kind, params, seed=42
        )
        scale = np.where(scaler.scale_ == 0.0, 1.0, scaler.scale_)
        bundle = SharedCoreModelBundle(
            estimator=est,
            x_mean=scaler.mean_,
            x_scale=scale,
            y_mean=y_mean,
            y_scale=y_scale,
            feature_groups=best_feat_groups,
            feature_names=feature_names,
            feature_version=V2_FEATURE_VERSION,
            ambient_c=AMBIENT_C,
            model_name=cname,
            hyperparameters={"kind": kind, **params},
            source_csv_sha256=ds.source_sha256,
            seed=42,
        )
        size_b = serialized_size_bytes(bundle)
        coef = np.asarray(est.coef_).reshape(-1)
        nnz = int(np.sum(np.abs(coef) > 1e-12))
        w_l2 = float(np.linalg.norm(coef))
        w_max = float(np.max(np.abs(coef)))

        # Quick latency check (20 single-sample calls)
        feat_ms_list, inf_ms_list = [], []
        for idx in range(20):
            _, fms, ims = bundle.predict_temperatures_c(g_va[idx], p_va[idx], d_va[idx])
            feat_ms_list.append(fms)
            inf_ms_list.append(ims)

        rec = {
            "candidate": cname,
            "family": kind,
            "feature_set": best_feat_name,
            "num_features": X_tr.shape[1],
            "nonzero_params": nnz,
            "hyperparameters": json.dumps(params, sort_keys=True),
            "converged": str(converged),
            "weight_l2_norm": f"{w_l2:.4f}",
            "max_abs_weight": f"{w_max:.4f}",
            "train_mae_c": f"{tr_m['mae_c']:.6f}",
            "val_mae_c": f"{va_m['mae_c']:.6f}",
            "val_rmse_c": f"{va_m['rmse_c']:.6f}",
            "val_max_err_c": f"{va_m['max_err_c']:.6f}",
            "val_centered_mae_c": f"{va_m['centered_mae_c']:.6f}",
            "val_hottest_core_acc_pct": f"{va_m['hottest_core_acc_pct']:.2f}",
            "serialized_bytes": size_b,
            "median_feat_ms": f"{statistics.median(feat_ms_list):.6f}",
            "median_inf_ms": f"{statistics.median(inf_ms_list):.6f}",
            "median_total_ms": f"{statistics.median([a + b for a, b in zip(feat_ms_list, inf_ms_list)]):.6f}",
        }
        val_rows.append(rec)

        if kind not in best_by_family or float(rec["val_mae_c"]) < float(best_by_family[kind]["val_mae_c"]):
            best_by_family[kind] = rec

        # Selection guard: require convergence and well-conditioned weights (max|w| <= 5.0)
        # to prevent collinearity blow-up on unseen/extrapolated floorplans.
        if converged and w_max <= 5.0:
            if best_overall_record is None or float(rec["val_mae_c"]) < float(best_overall_record["val_mae_c"]):
                best_overall_record = rec
                best_overall_bundle = bundle

    # Sort validation_results by val_mae_c ascending
    val_rows_sorted = sorted(val_rows, key=lambda r: float(r["val_mae_c"]))
    save_csv(SCRIPT_DIR / "validation_results.csv", val_rows_sorted)

    # Model comparison table: v1 baseline + best of each family
    model_comp_rows = [
        {
            "candidate": "stage_b_v1_frozen_lasso",
            "family": "v1_multioutput_lasso",
            "feature_set": "F0_v1_baseline_252_slot",
            "num_features": 252,
            "nonzero_params": int(np.sum(np.abs(v1_bundle.estimator.coef_) > 1e-12)),
            "hyperparameters": json.dumps({"alpha": 0.001}),
            "converged": "True",
            "weight_l2_norm": f"{float(np.linalg.norm(v1_bundle.estimator.coef_)):.4f}",
            "max_abs_weight": f"{float(np.max(np.abs(v1_bundle.estimator.coef_))):.4f}",
            "train_mae_c": "",
            "val_mae_c": f"{v1_va_metrics['mae_c']:.6f}",
            "val_rmse_c": f"{v1_va_metrics['rmse_c']:.6f}",
            "val_max_err_c": f"{v1_va_metrics['max_err_c']:.6f}",
            "val_centered_mae_c": f"{v1_va_metrics['centered_mae_c']:.6f}",
            "val_hottest_core_acc_pct": f"{v1_va_metrics['hottest_core_acc_pct']:.2f}",
            "serialized_bytes": (model_dir / "model.joblib").stat().st_size,
            "median_feat_ms": "",
            "median_inf_ms": "",
            "median_total_ms": "",
        }
    ]
    for fam in ("ols", "ridge", "lasso", "elastic_net"):
        if fam in best_by_family:
            model_comp_rows.append(best_by_family[fam])
    save_csv(SCRIPT_DIR / "model_comparison.csv", model_comp_rows)

    # Freeze best candidate model artifact under candidate_model/
    assert best_overall_bundle is not None and best_overall_record is not None
    cand_dir = SCRIPT_DIR / "candidate_model"
    cand_dir.mkdir(parents=True, exist_ok=True)
    bundle_path = cand_dir / "model_v2.joblib"
    joblib.dump(best_overall_bundle, bundle_path, compress=0)
    bundle_sha = hashlib.sha256(bundle_path.read_bytes()).hexdigest()

    meta = {
        "candidate_name": best_overall_record["candidate"],
        "family": best_overall_record["family"],
        "feature_set": best_feat_name,
        "feature_groups": list(best_feat_groups),
        "num_features": int(best_overall_record["num_features"]),
        "nonzero_params": int(best_overall_record["nonzero_params"]),
        "hyperparameters": best_overall_bundle.hyperparameters,
        "converged": best_overall_record["converged"] == "True",
        "serialized_bytes": bundle_path.stat().st_size,
        "model_sha256": bundle_sha,
        "seed": 42,
        "source_csv_sha256": ds.source_sha256,
        "validation_metrics": {
            "mae_c": float(best_overall_record["val_mae_c"]),
            "rmse_c": float(best_overall_record["val_rmse_c"]),
            "max_err_c": float(best_overall_record["val_max_err_c"]),
            "centered_mae_c": float(best_overall_record["val_centered_mae_c"]),
            "hottest_core_acc_pct": float(best_overall_record["val_hottest_core_acc_pct"]),
        },
        "v1_validation_metrics": v1_va_metrics,
        "selection_policy": "Selected strictly on original 53-train / 12-validation split; test split and 23 Stage A floorplans untouched.",
    }
    (cand_dir / "metadata.json").write_text(json.dumps(meta, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    print("\n=== MODEL COMPARISON (VALIDATION SPLIT, 12 FLOORPLANS / 240 WORKLOADS) ===")
    for r in model_comp_rows:
        print(
            f"  {r['candidate']:26s} | val_MAE={r['val_mae_c']} C | val_RMSE={r['val_rmse_c']} C | "
            f"val_Max={r['val_max_err_c']} C | HotCore={r['val_hottest_core_acc_pct']}% | "
            f"size={r['serialized_bytes']} B | nnz={r['nonzero_params']}"
        )
    print(f"\nFROZEN BEST CANDIDATE: {best_overall_record['candidate']} -> {bundle_path} ({bundle_path.stat().st_size} B, sha256={bundle_sha[:16]}...)")


if __name__ == "__main__":
    main()
