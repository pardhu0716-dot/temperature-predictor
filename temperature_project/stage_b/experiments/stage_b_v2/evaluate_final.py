#!/usr/bin/env python3
"""
Phase 6 & 7 — Final Frozen Evaluation of Stage B v2 vs Stage B v1 (and Stage A reference).

Evaluates the ALREADY-FROZEN candidate model (`candidate_model/model_v2.joblib`) ONCE on:
  1. Original held-out Stage B test split (12 floorplans, 240 workloads, 1,920 core predictions)
  2. External 23 original Stage A floorplans (460 HotSpot workloads, 3,680 core predictions)

Produces under stage_b/experiments/stage_b_v2/:
  - final_comparison.csv
  - per_floorplan_comparison.csv
  - per_core_comparison.csv
  - accuracy_comparison_v1_vs_v2.svg
  - latency_comparison_v1_vs_v2.svg
  - stage_b_v2_report.md
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import platform
import statistics
import sys
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter_ns
from typing import Any, Dict, List, Tuple

import joblib
import numpy as np
import sklearn

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parents[2]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from stage_b.ml.data import CORE_IDS, load_csv  # noqa: E402
from stage_b.ml.features import build_features as build_v1_features  # noqa: E402
from stage_b.web.service import Predictor as StageBV1Predictor, UNITS  # noqa: E402
from run_experiments import SharedCoreModelBundle, eval_metrics, save_csv  # noqa: E402

KELVIN_OFFSET = 273.15
DATASET_DIR = PROJECT_DIR / "dataset/Sample_dataset/2D_floorplans"


def percentile(vals: List[float], p: float) -> float:
    return float(np.percentile(np.asarray(vals, dtype=np.float64), p))


def parse_flp(flp_path: Path) -> Tuple[np.ndarray, np.ndarray]:
    blocks, cores = [], {}
    for line in flp_path.read_text(encoding="utf-8").splitlines():
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
            tmap[name] = float(toks[1]) - KELVIN_OFFSET
    return np.array([tmap[n] for n in CORE_IDS], dtype=float)


def write_svg_grouped_bars(
    out_path: Path,
    title: str,
    subtitle: str,
    y_label: str,
    categories: List[str],
    series: List[Tuple[str, str, List[float]]],
    log_scale: bool = False,
) -> None:
    n = len(categories)
    width = max(940, 42 * n + 180)
    height = 500
    ml, mr, mt, mb = 85, 35, 88, 118
    pw, ph = width - ml - mr, height - mt - mb

    all_vals = [v for _, _, vals in series for v in vals]
    if log_scale:
        pos = [v for v in all_vals if v > 0]
        min_e = math.floor(math.log10(min(pos)))
        max_e = math.ceil(math.log10(max(pos)))
        if max_e == min_e:
            max_e += 1

        def y_pos(v: float) -> float:
            return mt + ph * (1.0 - (math.log10(max(v, 10 ** min_e)) - min_e) / (max_e - min_e))

        ticks = [10 ** e for e in range(min_e, max_e + 1)]
    else:
        max_v = max(all_vals) * 1.15 if all_vals else 1.0

        def y_pos(v: float) -> float:
            return mt + ph * (1.0 - v / max_v)

        ticks = [max_v * i / 5.0 for i in range(6)]

    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        f'<text x="{ml}" y="30" font-family="Segoe UI, Arial, sans-serif" font-size="18" font-weight="700" fill="#0f172a">{title}</text>',
        f'<text x="{ml}" y="52" font-family="Segoe UI, Arial, sans-serif" font-size="12" fill="#475569">{subtitle}</text>',
    ]
    lx = ml
    for lbl, col, _ in series:
        lines.append(f'<rect x="{lx}" y="60" width="12" height="12" rx="2" fill="{col}"/>')
        lines.append(f'<text x="{lx + 18}" y="70" font-family="Segoe UI, Arial, sans-serif" font-size="12" fill="#1e293b">{lbl}</text>')
        lx += 18 + len(lbl) * 7 + 22

    for tv in ticks:
        yp = y_pos(tv)
        lines.append(f'<line x1="{ml}" y1="{yp:.1f}" x2="{ml + pw}" y2="{yp:.1f}" stroke="#e2e8f0" stroke-width="1"/>')
        lbl = f"{tv:g}" if log_scale else f"{tv:.2f}"
        lines.append(f'<text x="{ml - 10}" y="{yp + 4:.1f}" text-anchor="end" font-family="Segoe UI, Arial, sans-serif" font-size="11" fill="#475569">{lbl}</text>')

    lines.append(f'<line x1="{ml}" y1="{mt}" x2="{ml}" y2="{mt + ph}" stroke="#334155" stroke-width="1.5"/>')
    lines.append(f'<line x1="{ml}" y1="{mt + ph}" x2="{ml + pw}" y2="{mt + ph}" stroke="#334155" stroke-width="1.5"/>')
    lines.append(f'<text x="20" y="{mt + ph / 2:.1f}" transform="rotate(-90 20 {mt + ph / 2:.1f})" text-anchor="middle" font-family="Segoe UI, Arial, sans-serif" font-size="12" font-weight="600" fill="#1e293b">{y_label}</text>')

    group_w = pw / max(n, 1)
    ns = len(series)
    bar_w = min(28.0, (group_w * 0.78) / ns)
    tot_w = bar_w * ns + (ns - 1) * 2.0

    for idx, cat in enumerate(categories):
        cx = ml + (idx + 0.5) * group_w
        x0 = cx - tot_w / 2.0
        for s_idx, (lbl, col, vals) in enumerate(series):
            val = vals[idx]
            bx = x0 + s_idx * (bar_w + 2.0)
            by = y_pos(val)
            bh = max(1.0, (mt + ph) - by)
            lines.append(f'<rect x="{bx:.1f}" y="{by:.1f}" width="{bar_w:.1f}" height="{bh:.1f}" rx="2" fill="{col}"><title>{cat} ({lbl}): {val:.6f}</title></rect>')
            if n <= 8:
                lines.append(f'<text x="{bx + bar_w / 2:.1f}" y="{by - 5:.1f}" text-anchor="middle" font-family="Segoe UI, Arial, sans-serif" font-size="10" fill="#0f172a">{val:.4f}</text>')
        if n > 8:
            lines.append(f'<text x="{cx:.1f}" y="{mt + ph + 14}" transform="rotate(-45 {cx:.1f} {mt + ph + 14})" text-anchor="end" font-family="Segoe UI, Arial, sans-serif" font-size="10" fill="#334155">{cat}</text>')
        else:
            lines.append(f'<text x="{cx:.1f}" y="{mt + ph + 20}" text-anchor="middle" font-family="Segoe UI, Arial, sans-serif" font-size="11" fill="#334155">{cat}</text>')

    lines.append("</svg>")
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    cand_dir = SCRIPT_DIR / "candidate_model"
    cand_meta = json.loads((cand_dir / "metadata.json").read_text(encoding="utf-8"))
    v2_bundle: SharedCoreModelBundle = joblib.load(cand_dir / "model_v2.joblib")
    v2_size_b = (cand_dir / "model_v2.joblib").stat().st_size

    v1_dir = PROJECT_DIR / "stage_b/models/final_lasso"
    v1_manifest = json.loads((v1_dir / "manifest.json").read_text(encoding="utf-8"))
    v1_schema = json.loads((v1_dir / "schema.json").read_text(encoding="utf-8"))
    v1_bundle = joblib.load(v1_dir / "model.joblib")
    v1_size_b = (v1_dir / "model.joblib").stat().st_size
    v1_predictor = StageBV1Predictor()

    # Load 77-floorplan Stage B dataset to evaluate ONCE on held-out TEST split (12 floorplans, 240 workloads)
    ds = load_csv(
        PROJECT_DIR / "stage_b/stage_b_training_dataset.csv",
        v1_schema,
        PROJECT_DIR / "stage_b/dataset_v1/floorplan_manifest.json",
    )
    train_ids = set(v1_manifest["split_floorplans"]["train"])
    test_ids = set(v1_manifest["split_floorplans"]["test"])
    tr_mask = np.isin(ds.floorplan_ids, list(train_ids))
    te_mask = np.isin(ds.floorplan_ids, list(test_ids))
    assert int(te_mask.sum()) == 240

    # Training die bounds (strictly from the 53 training floorplans)
    tr_die = ds.die_mm[tr_mask]
    sb_w_min, sb_w_max = float(tr_die[:, 0].min()), float(tr_die[:, 0].max())
    sb_h_min, sb_h_max = float(tr_die[:, 1].min()), float(tr_die[:, 1].max())

    # 1. Held-out Stage B test split evaluation (v1 vs v2)
    X_v1_all, _ = build_v1_features(ds)
    y_te = ds.temperatures_c[te_mask]
    pred_v1_te = v1_bundle.predict(X_v1_all[te_mask])
    pred_v2_te, _, _ = v2_bundle.predict_temperatures_c(ds.geometry_mm[te_mask], ds.powers_w[te_mask], ds.die_mm[te_mask])

    v1_te_m = eval_metrics(y_te, pred_v1_te)
    v2_te_m = eval_metrics(y_te, pred_v2_te)

    # 2. External 23 original Stage A floorplans (460 HotSpot workloads, 3,680 core predictions)
    # Load Stage A reference results from our previous benchmark (identical 460 workloads)
    bench_dir = PROJECT_DIR / "stage_b/benchmarks/stage_a_vs_stage_b_results"
    bench_comp = {r["model"]: r for r in csv.DictReader((bench_dir / "comparison.csv").open(encoding="utf-8"))}
    bench_fp = {r["floorplan"]: r for r in csv.DictReader((bench_dir / "per_floorplan.csv").open(encoding="utf-8"))}
    bench_core = {r["core_id"]: r for r in csv.DictReader((bench_dir / "per_core.csv").open(encoding="utf-8"))}
    stage_a_ref = next(v for k, v in bench_comp.items() if "Stage A" in k)

    # Warmup v1 & v2
    fp_dirs = sorted([p for p in DATASET_DIR.glob("*/*") if p.is_dir()])
    assert len(fp_dirs) == 23
    g0, d0 = parse_flp(fp_dirs[0] / f"{fp_dirs[0].name}_layer_0.flp")
    p0 = parse_ptrace(sorted((fp_dirs[0] / "validation_samples/ptrace").glob("val_sample_*.ptrace"))[0])
    v2_bundle.predict_temperatures_c(g0, p0, d0)
    v1_predictor.predict({
        "mode": "custom", "units": UNITS,
        "die": {"width_mm": float(d0[0]), "height_mm": float(d0[1])},
        "cores": [{"id": n, "width_mm": float(g0[i, 0]), "height_mm": float(g0[i, 1]),
                   "x_mm": float(g0[i, 2]), "y_mm": float(g0[i, 3]), "power_w": float(p0[i])}
                  for i, n in enumerate(CORE_IDS)],
    })

    gt_all, v1_all, v2_all = [], [], []
    v1_feat_ms, v1_inf_ms, v1_tot_ms = [], [], []
    v2_feat_ms, v2_inf_ms, v2_tot_ms = [], [], []

    per_fp_rows: List[Dict[str, Any]] = []
    in_box_gt, in_box_v1, in_box_v2 = [], [], []
    out_box_gt, out_box_v1, out_box_v2 = [], [], []

    for fp_dir in fp_dirs:
        full_name = f"{fp_dir.parent.name}/{fp_dir.name}"
        g, die = parse_flp(fp_dir / f"{fp_dir.name}_layer_0.flp")
        ptraces = sorted((fp_dir / "validation_samples/ptrace").glob("val_sample_*.ptrace"))

        fp_gt, fp_v1, fp_v2 = [], [], []
        for pt in ptraces:
            st = fp_dir / "validation_samples/steady" / f"{pt.stem}.steady"
            p = parse_ptrace(pt)
            gt = parse_steady(st)

            # Stage B v1 via actual web service Predictor
            res_v1 = v1_predictor.predict({
                "mode": "custom", "units": UNITS,
                "die": {"width_mm": float(die[0]), "height_mm": float(die[1])},
                "cores": [{"id": n, "width_mm": float(g[i, 0]), "height_mm": float(g[i, 1]),
                           "x_mm": float(g[i, 2]), "y_mm": float(g[i, 3]), "power_w": float(p[i])}
                          for i, n in enumerate(CORE_IDS)],
            })
            t_v1 = np.array([float(res_v1["temperatures_c"][n]) for n in CORE_IDS], dtype=float)
            v1_feat_ms.append(float(res_v1["feature_engineering_ms"]))
            v1_inf_ms.append(float(res_v1["latency_ms"]))
            v1_tot_ms.append(float(res_v1["feature_engineering_ms"]) + float(res_v1["latency_ms"]))

            # Stage B v2 via frozen SharedCoreModelBundle
            t_v2_arr, f2, i2 = v2_bundle.predict_temperatures_c(g, p, die)
            t_v2 = t_v2_arr[0]
            v2_feat_ms.append(f2)
            v2_inf_ms.append(i2)
            v2_tot_ms.append(f2 + i2)

            fp_gt.append(gt)
            fp_v1.append(t_v1)
            fp_v2.append(t_v2)
            gt_all.append(gt)
            v1_all.append(t_v1)
            v2_all.append(t_v2)

        fp_gt_arr = np.asarray(fp_gt)
        fp_v1_arr = np.asarray(fp_v1)
        fp_v2_arr = np.asarray(fp_v2)
        m1 = eval_metrics(fp_gt_arr, fp_v1_arr)
        m2 = eval_metrics(fp_gt_arr, fp_v2_arr)

        # Match the exact 17 in-box vs 6 out-of-box grouping from the previous benchmark
        in_box = bench_fp[full_name]["inside_stage_b_die_box"] == "True"
        if in_box:
            in_box_gt.append(fp_gt_arr)
            in_box_v1.append(fp_v1_arr)
            in_box_v2.append(fp_v2_arr)
        else:
            out_box_gt.append(fp_gt_arr)
            out_box_v1.append(fp_v1_arr)
            out_box_v2.append(fp_v2_arr)

        per_fp_rows.append({
            "floorplan": full_name,
            "die_width_mm": f"{die[0]:.3f}",
            "die_height_mm": f"{die[1]:.3f}",
            "inside_stage_b_die_box": str(in_box),
            "stage_a_mae_c": bench_fp[full_name]["stage_a_mae_c"],
            "v1_mae_c": f"{m1['mae_c']:.6f}",
            "v2_mae_c": f"{m2['mae_c']:.6f}",
            "delta_mae_v2_minus_v1_c": f"{m2['mae_c'] - m1['mae_c']:+.6f}",
            "v1_rmse_c": f"{m1['rmse_c']:.6f}",
            "v2_rmse_c": f"{m2['rmse_c']:.6f}",
            "v1_max_err_c": f"{m1['max_err_c']:.6f}",
            "v2_max_err_c": f"{m2['max_err_c']:.6f}",
            "v1_hottest_acc_pct": f"{m1['hottest_core_acc_pct']:.2f}",
            "v2_hottest_acc_pct": f"{m2['hottest_core_acc_pct']:.2f}",
        })

    gt_all_arr = np.asarray(gt_all)
    v1_all_arr = np.asarray(v1_all)
    v2_all_arr = np.asarray(v2_all)
    ext_v1_m = eval_metrics(gt_all_arr, v1_all_arr)
    ext_v2_m = eval_metrics(gt_all_arr, v2_all_arr)

    in_v1_m = eval_metrics(np.vstack(in_box_gt), np.vstack(in_box_v1))
    in_v2_m = eval_metrics(np.vstack(in_box_gt), np.vstack(in_box_v2))
    out_v1_m = eval_metrics(np.vstack(out_box_gt), np.vstack(out_box_v1))
    out_v2_m = eval_metrics(np.vstack(out_box_gt), np.vstack(out_box_v2))

    # Per-core comparison (external 23 floorplans, 460 workloads)
    per_core_rows = []
    for i, n in enumerate(CORE_IDS):
        e1 = np.abs(v1_all_arr[:, i] - gt_all_arr[:, i])
        e2 = np.abs(v2_all_arr[:, i] - gt_all_arr[:, i])
        per_core_rows.append({
            "core_id": n,
            "samples": len(e1),
            "stage_a_mae_c": bench_core[n]["stage_a_mae_c"],
            "v1_mae_c": f"{float(np.mean(e1)):.6f}",
            "v2_mae_c": f"{float(np.mean(e2)):.6f}",
            "delta_mae_v2_minus_v1_c": f"{float(np.mean(e2) - np.mean(e1)):+.6f}",
            "v1_rmse_c": f"{float(np.sqrt(np.mean(e1**2))):.6f}",
            "v2_rmse_c": f"{float(np.sqrt(np.mean(e2**2))):.6f}",
            "v1_max_err_c": f"{float(np.max(e1)):.6f}",
            "v2_max_err_c": f"{float(np.max(e2)):.6f}",
        })

    # Final comparison CSV (all evaluation settings clearly separated)
    final_comp_rows = [
        {
            "evaluation_setting": "Stage B Held-Out Test Split (12 floorplans / 240 workloads)",
            "model": "Stage B v1 (Frozen Lasso alpha=0.001)",
            "workloads": 240,
            "mae_c": f"{v1_te_m['mae_c']:.6f}",
            "rmse_c": f"{v1_te_m['rmse_c']:.6f}",
            "max_err_c": f"{v1_te_m['max_err_c']:.6f}",
            "centered_mae_c": f"{v1_te_m['centered_mae_c']:.6f}",
            "hottest_core_acc_pct": f"{v1_te_m['hottest_core_acc_pct']:.2f}",
            "median_feat_ms": "",
            "median_inf_ms": "",
            "median_total_ms": "",
            "p95_total_ms": "",
            "serialized_bytes": v1_size_b,
        },
        {
            "evaluation_setting": "Stage B Held-Out Test Split (12 floorplans / 240 workloads)",
            "model": f"Stage B v2 ({cand_meta['candidate_name']}, F4 63-feature shared physics)",
            "workloads": 240,
            "mae_c": f"{v2_te_m['mae_c']:.6f}",
            "rmse_c": f"{v2_te_m['rmse_c']:.6f}",
            "max_err_c": f"{v2_te_m['max_err_c']:.6f}",
            "centered_mae_c": f"{v2_te_m['centered_mae_c']:.6f}",
            "hottest_core_acc_pct": f"{v2_te_m['hottest_core_acc_pct']:.2f}",
            "median_feat_ms": "",
            "median_inf_ms": "",
            "median_total_ms": "",
            "p95_total_ms": "",
            "serialized_bytes": v2_size_b,
        },
        {
            "evaluation_setting": "External 23 Stage A Floorplans — ALL 23 (460 workloads)",
            "model": "Stage A Reference (Calibrated 8x8 Matrix, in-distribution)",
            "workloads": 460,
            "mae_c": stage_a_ref["mae_c"],
            "rmse_c": stage_a_ref["rmse_c"],
            "max_err_c": stage_a_ref["max_abs_err_c"],
            "centered_mae_c": "",
            "hottest_core_acc_pct": stage_a_ref["hottest_core_acc_pct"],
            "median_feat_ms": "0.000000",
            "median_inf_ms": stage_a_ref["median_model_inference_ms"],
            "median_total_ms": stage_a_ref["median_pipeline_total_ms"],
            "p95_total_ms": stage_a_ref["p95_pipeline_total_ms"],
            "serialized_bytes": 17733,
        },
        {
            "evaluation_setting": "External 23 Stage A Floorplans — ALL 23 (460 workloads)",
            "model": "Stage B v1 (Frozen Lasso alpha=0.001)",
            "workloads": 460,
            "mae_c": f"{ext_v1_m['mae_c']:.6f}",
            "rmse_c": f"{ext_v1_m['rmse_c']:.6f}",
            "max_err_c": f"{ext_v1_m['max_err_c']:.6f}",
            "centered_mae_c": f"{ext_v1_m['centered_mae_c']:.6f}",
            "hottest_core_acc_pct": f"{ext_v1_m['hottest_core_acc_pct']:.2f}",
            "median_feat_ms": f"{statistics.median(v1_feat_ms):.6f}",
            "median_inf_ms": f"{statistics.median(v1_inf_ms):.6f}",
            "median_total_ms": f"{statistics.median(v1_tot_ms):.6f}",
            "p95_total_ms": f"{percentile(v1_tot_ms, 95):.6f}",
            "serialized_bytes": v1_size_b,
        },
        {
            "evaluation_setting": "External 23 Stage A Floorplans — ALL 23 (460 workloads)",
            "model": f"Stage B v2 ({cand_meta['candidate_name']}, F4 63-feature shared physics)",
            "workloads": 460,
            "mae_c": f"{ext_v2_m['mae_c']:.6f}",
            "rmse_c": f"{ext_v2_m['rmse_c']:.6f}",
            "max_err_c": f"{ext_v2_m['max_err_c']:.6f}",
            "centered_mae_c": f"{ext_v2_m['centered_mae_c']:.6f}",
            "hottest_core_acc_pct": f"{ext_v2_m['hottest_core_acc_pct']:.2f}",
            "median_feat_ms": f"{statistics.median(v2_feat_ms):.6f}",
            "median_inf_ms": f"{statistics.median(v2_inf_ms):.6f}",
            "median_total_ms": f"{statistics.median(v2_tot_ms):.6f}",
            "p95_total_ms": f"{percentile(v2_tot_ms, 95):.6f}",
            "serialized_bytes": v2_size_b,
        },
        {
            "evaluation_setting": "External Subset — 17 Floorplans Inside Stage B Die Box (340 workloads)",
            "model": "Stage B v1 (Frozen Lasso alpha=0.001)",
            "workloads": 340,
            "mae_c": f"{in_v1_m['mae_c']:.6f}",
            "rmse_c": f"{in_v1_m['rmse_c']:.6f}",
            "max_err_c": f"{in_v1_m['max_err_c']:.6f}",
            "centered_mae_c": f"{in_v1_m['centered_mae_c']:.6f}",
            "hottest_core_acc_pct": f"{in_v1_m['hottest_core_acc_pct']:.2f}",
            "median_feat_ms": "",
            "median_inf_ms": "",
            "median_total_ms": "",
            "p95_total_ms": "",
            "serialized_bytes": v1_size_b,
        },
        {
            "evaluation_setting": "External Subset — 17 Floorplans Inside Stage B Die Box (340 workloads)",
            "model": f"Stage B v2 ({cand_meta['candidate_name']}, F4 63-feature shared physics)",
            "workloads": 340,
            "mae_c": f"{in_v2_m['mae_c']:.6f}",
            "rmse_c": f"{in_v2_m['rmse_c']:.6f}",
            "max_err_c": f"{in_v2_m['max_err_c']:.6f}",
            "centered_mae_c": f"{in_v2_m['centered_mae_c']:.6f}",
            "hottest_core_acc_pct": f"{in_v2_m['hottest_core_acc_pct']:.2f}",
            "median_feat_ms": "",
            "median_inf_ms": "",
            "median_total_ms": "",
            "p95_total_ms": "",
            "serialized_bytes": v2_size_b,
        },
        {
            "evaluation_setting": "External Subset — 6 Floorplans Outside Stage B Die Box (120 workloads)",
            "model": "Stage B v1 (Frozen Lasso alpha=0.001)",
            "workloads": 120,
            "mae_c": f"{out_v1_m['mae_c']:.6f}",
            "rmse_c": f"{out_v1_m['rmse_c']:.6f}",
            "max_err_c": f"{out_v1_m['max_err_c']:.6f}",
            "centered_mae_c": f"{out_v1_m['centered_mae_c']:.6f}",
            "hottest_core_acc_pct": f"{out_v1_m['hottest_core_acc_pct']:.2f}",
            "median_feat_ms": "",
            "median_inf_ms": "",
            "median_total_ms": "",
            "p95_total_ms": "",
            "serialized_bytes": v1_size_b,
        },
        {
            "evaluation_setting": "External Subset — 6 Floorplans Outside Stage B Die Box (120 workloads)",
            "model": f"Stage B v2 ({cand_meta['candidate_name']}, F4 63-feature shared physics)",
            "workloads": 120,
            "mae_c": f"{out_v2_m['mae_c']:.6f}",
            "rmse_c": f"{out_v2_m['rmse_c']:.6f}",
            "max_err_c": f"{out_v2_m['max_err_c']:.6f}",
            "centered_mae_c": f"{out_v2_m['centered_mae_c']:.6f}",
            "hottest_core_acc_pct": f"{out_v2_m['hottest_core_acc_pct']:.2f}",
            "median_feat_ms": "",
            "median_inf_ms": "",
            "median_total_ms": "",
            "p95_total_ms": "",
            "serialized_bytes": v2_size_b,
        },
    ]

    save_csv(SCRIPT_DIR / "final_comparison.csv", final_comp_rows)
    save_csv(SCRIPT_DIR / "per_floorplan_comparison.csv", per_fp_rows)
    save_csv(SCRIPT_DIR / "per_core_comparison.csv", per_core_rows)

    # Comparison charts (SVG)
    write_svg_grouped_bars(
        SCRIPT_DIR / "accuracy_comparison_v1_vs_v2.svg",
        title="Stage B v1 vs Stage B v2 — Per-Floorplan MAE (°C) on Original 23 Stage A Floorplans",
        subtitle=f"External 23-FP MAE: v1 = {ext_v1_m['mae_c']:.4f} °C -> v2 = {ext_v2_m['mae_c']:.4f} °C | Hottest-Core Acc: {ext_v1_m['hottest_core_acc_pct']:.1f}% -> {ext_v2_m['hottest_core_acc_pct']:.1f}%",
        y_label="MAE (°C)",
        categories=[r["floorplan"] for r in per_fp_rows],
        series=[
            ("Stage B v1 (Frozen Lasso, 252 features)", "#ea580c", [float(r["v1_mae_c"]) for r in per_fp_rows]),
            (f"Stage B v2 ({cand_meta['candidate_name']}, 63 shared physics features)", "#16a34a", [float(r["v2_mae_c"]) for r in per_fp_rows]),
        ],
    )
    write_svg_grouped_bars(
        SCRIPT_DIR / "latency_comparison_v1_vs_v2.svg",
        title="Stage B v1 vs Stage B v2 — Prediction Pipeline Latency (ms)",
        subtitle="Measured across 460 external workloads: feature engineering vs model inference vs full pipeline",
        y_label="Latency (ms)",
        categories=["Feature Eng. (Median)", "Model Inference (Median)", "Full Pipeline (Median)", "Full Pipeline (p95)"],
        series=[
            ("Stage B v1 (Frozen Lasso)", "#ea580c", [
                statistics.median(v1_feat_ms), statistics.median(v1_inf_ms),
                statistics.median(v1_tot_ms), percentile(v1_tot_ms, 95),
            ]),
            (f"Stage B v2 ({cand_meta['candidate_name']})", "#16a34a", [
                statistics.median(v2_feat_ms), statistics.median(v2_inf_ms),
                statistics.median(v2_tot_ms), percentile(v2_tot_ms, 95),
            ]),
        ],
    )

    # Count how many floorplans improved vs regressed
    n_improved = sum(1 for r in per_fp_rows if float(r["v2_mae_c"]) < float(r["v1_mae_c"]))
    n_regressed = len(per_fp_rows) - n_improved

    now_utc = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    v1_va = cand_meta["v1_validation_metrics"]
    v2_va = cand_meta["validation_metrics"]

    report = [
        "# Stage B v2 Experimental Evaluation Report",
        "",
        f"- **Generated**: {now_utc}",
        f"- **Candidate**: `{cand_meta['candidate_name']}` (`{cand_meta['family']}`, `{cand_meta['hyperparameters']}`)",
        f"- **Feature Representation**: `{cand_meta['feature_set']}` (`{cand_meta['num_features']}` permutation-equivariant per-core physical features; groups: `{', '.join(cand_meta['feature_groups'])}`)",
        f"- **Serialized Model Artifact**: `candidate_model/model_v2.joblib` (`{v2_size_b:,}` bytes, SHA-256 `{cand_meta['model_sha256']}`)",
        f"- **Environment**: Python `{platform.python_version()}`, NumPy `{np.__version__}`, scikit-learn `{sklearn.__version__}`, joblib `{joblib.__version__}`, seed `42`",
        f"- **Splits**: Exact original `53` train (`1,060` workloads / `8,480` core rows) / `12` validation (`240` workloads) / `12` test (`240` workloads) from `stage_b/models/final_lasso/manifest.json`",
        "",
        "## 1. Summary Comparison Across All Evaluation Regimes",
        "",
        "| Evaluation Setting | Model | MAE (°C) | RMSE (°C) | Max Err (°C) | Hottest-Core Acc | Full Pipeline Median / p95 (ms) | Model Size (B) |",
        "| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |",
        f"| **Stage B Validation (12 FPs / 240 wl)** | Stage B v1 (Lasso `0.001`) | `{v1_va['mae_c']:.6f}` | `{v1_va['rmse_c']:.6f}` | `{v1_va['max_err_c']:.6f}` | `{v1_va['hottest_core_acc_pct']:.2f}%` | — | `{v1_size_b:,}` |",
        f"| **Stage B Validation (12 FPs / 240 wl)** | **Stage B v2 (`{cand_meta['candidate_name']}`)** | **`{v2_va['mae_c']:.6f}`** | **`{v2_va['rmse_c']:.6f}`** | **`{v2_va['max_err_c']:.6f}`** | **`{v2_va['hottest_core_acc_pct']:.2f}%`** | — | **`{v2_size_b:,}`** |",
        f"| **Stage B Held-Out Test (12 FPs / 240 wl)** | Stage B v1 (Lasso `0.001`) | `{v1_te_m['mae_c']:.6f}` | `{v1_te_m['rmse_c']:.6f}` | `{v1_te_m['max_err_c']:.6f}` | `{v1_te_m['hottest_core_acc_pct']:.2f}%` | — | `{v1_size_b:,}` |",
        f"| **Stage B Held-Out Test (12 FPs / 240 wl)** | **Stage B v2 (`{cand_meta['candidate_name']}`)** | **`{v2_te_m['mae_c']:.6f}`** | **`{v2_te_m['rmse_c']:.6f}`** | **`{v2_te_m['max_err_c']:.6f}`** | **`{v2_te_m['hottest_core_acc_pct']:.2f}%`** | — | **`{v2_size_b:,}`** |",
        f"| **External 23 Stage A FPs (460 wl)** | Stage A Reference (Calibrated) | `{stage_a_ref['mae_c']}` | `{stage_a_ref['rmse_c']}` | `{stage_a_ref['max_abs_err_c']}` | `{stage_a_ref['hottest_core_acc_pct']}%` | `{stage_a_ref['median_pipeline_total_ms']}` / `{stage_a_ref['p95_pipeline_total_ms']}` | `17,733` CSV (`11,776` raw) |",
        f"| **External 23 Stage A FPs (460 wl)** | Stage B v1 (Lasso `0.001`) | `{ext_v1_m['mae_c']:.6f}` | `{ext_v1_m['rmse_c']:.6f}` | `{ext_v1_m['max_err_c']:.6f}` | `{ext_v1_m['hottest_core_acc_pct']:.2f}%` | `{statistics.median(v1_tot_ms):.4f}` / `{percentile(v1_tot_ms, 95):.4f}` | `{v1_size_b:,}` |",
        f"| **External 23 Stage A FPs (460 wl)** | **Stage B v2 (`{cand_meta['candidate_name']}`)** | **`{ext_v2_m['mae_c']:.6f}`** | **`{ext_v2_m['rmse_c']:.6f}`** | **`{ext_v2_m['max_err_c']:.6f}`** | **`{ext_v2_m['hottest_core_acc_pct']:.2f}%`** | **`{statistics.median(v2_tot_ms):.4f}` / `{percentile(v2_tot_ms, 95):.4f}`** | **`{v2_size_b:,}`** |",
        "",
        "## 2. External 23-Floorplan In-Box vs Out-of-Box Breakdown",
        "",
        "| Subset | Model | MAE (°C) | RMSE (°C) | Max Err (°C) | Hottest-Core Acc |",
        "| :--- | :--- | :---: | :---: | :---: | :---: |",
        f"| **17 Floorplans Inside Stage B Die Box** (`340` wl) | Stage B v1 | `{in_v1_m['mae_c']:.6f}` | `{in_v1_m['rmse_c']:.6f}` | `{in_v1_m['max_err_c']:.6f}` | `{in_v1_m['hottest_core_acc_pct']:.2f}%` |",
        f"| **17 Floorplans Inside Stage B Die Box** (`340` wl) | **Stage B v2** | **`{in_v2_m['mae_c']:.6f}`** | **`{in_v2_m['rmse_c']:.6f}`** | **`{in_v2_m['max_err_c']:.6f}`** | **`{in_v2_m['hottest_core_acc_pct']:.2f}%`** |",
        f"| **6 Floorplans Outside Stage B Die Box** (`120` wl) | Stage B v1 | `{out_v1_m['mae_c']:.6f}` | `{out_v1_m['rmse_c']:.6f}` | `{out_v1_m['max_err_c']:.6f}` | `{out_v1_m['hottest_core_acc_pct']:.2f}%` |",
        f"| **6 Floorplans Outside Stage B Die Box** (`120` wl) | **Stage B v2** | **`{out_v2_m['mae_c']:.6f}`** | **`{out_v2_m['rmse_c']:.6f}`** | **`{out_v2_m['max_err_c']:.6f}`** | **`{out_v2_m['hottest_core_acc_pct']:.2f}%`** |",
        "",
        f"- **Floorplan-level wins**: Stage B v2 improved MAE on **`{n_improved} / 23`** original Stage A floorplans (regressed on `{n_regressed} / 23`).",
        f"- **Latency breakdown (460 workloads)**:",
        f"  - Stage B v1: Feature eng. median `{statistics.median(v1_feat_ms):.4f} ms`, Model inference median `{statistics.median(v1_inf_ms):.4f} ms`, Full pipeline median `{statistics.median(v1_tot_ms):.4f} ms` (p95 `{percentile(v1_tot_ms, 95):.4f} ms`).",
        f"  - Stage B v2: Feature eng. median `{statistics.median(v2_feat_ms):.4f} ms`, Model inference median `{statistics.median(v2_inf_ms):.4f} ms`, Full pipeline median `{statistics.median(v2_tot_ms):.4f} ms` (p95 `{percentile(v2_tot_ms, 95):.4f} ms`).",
        "",
        "## 3. Disclosure of Limitations & Leakage Hygiene",
        "",
        "- **Strict Selection Hygiene**: Feature ablation (`feature_ablation.csv`) and model/hyperparameter selection (`validation_results.csv`, including the numerical conditioning check `max|w| <= 5.0` that rejected unregularized OLS) used **strictly the 53 training and 12 validation floorplans**. Neither the 12 Stage B test floorplans nor the 23 Stage A floorplans were used for fitting, scaling, or hyperparameter tuning.",
        "- **Test-Reuse Disclosure**: The 12 held-out Stage B test floorplans were previously evaluated for v1 and now evaluated once for frozen v2; the 23 external Stage A floorplans had their v1 error patterns inspected during Phase 1, so they are no longer an untouched blind benchmark.",
        "",
        "## 4. Reproducibility Commands",
        "```bash",
        "source ~/.venvs/temperature-predictor-preview/bin/activate",
        "cd ~/temperature-predictor-preview/temperature_project",
        "PYTHONDONTWRITEBYTECODE=1 python stage_b/experiments/stage_b_v2/phase1_error_analysis.py",
        "PYTHONDONTWRITEBYTECODE=1 python stage_b/experiments/stage_b_v2/run_experiments.py",
        "PYTHONDONTWRITEBYTECODE=1 python stage_b/experiments/stage_b_v2/evaluate_final.py",
        "```",
    ]
    (SCRIPT_DIR / "stage_b_v2_report.md").write_text("\n".join(report) + "\n", encoding="utf-8")

    print("=== FINAL EVALUATION COMPLETE ===")
    print(f"STAGE B HELD-OUT TEST (12 FPs / 240 wl):")
    print(f"  v1: MAE={v1_te_m['mae_c']:.6f} C, RMSE={v1_te_m['rmse_c']:.6f} C, Max={v1_te_m['max_err_c']:.6f} C, Hot={v1_te_m['hottest_core_acc_pct']:.2f}%")
    print(f"  v2: MAE={v2_te_m['mae_c']:.6f} C, RMSE={v2_te_m['rmse_c']:.6f} C, Max={v2_te_m['max_err_c']:.6f} C, Hot={v2_te_m['hottest_core_acc_pct']:.2f}%")
    print(f"EXTERNAL 23 STAGE A FPs (460 wl):")
    print(f"  v1: MAE={ext_v1_m['mae_c']:.6f} C, RMSE={ext_v1_m['rmse_c']:.6f} C, Max={ext_v1_m['max_err_c']:.6f} C, Hot={ext_v1_m['hottest_core_acc_pct']:.2f}%, MedTot={statistics.median(v1_tot_ms):.4f} ms")
    print(f"  v2: MAE={ext_v2_m['mae_c']:.6f} C, RMSE={ext_v2_m['rmse_c']:.6f} C, Max={ext_v2_m['max_err_c']:.6f} C, Hot={ext_v2_m['hottest_core_acc_pct']:.2f}%, MedTot={statistics.median(v2_tot_ms):.4f} ms")
    print(f"  17 In-Box:  v1 MAE={in_v1_m['mae_c']:.4f} C -> v2 MAE={in_v2_m['mae_c']:.4f} C (Hot: {in_v1_m['hottest_core_acc_pct']:.1f}% -> {in_v2_m['hottest_core_acc_pct']:.1f}%)")
    print(f"   6 Out-Box: v1 MAE={out_v1_m['mae_c']:.4f} C -> v2 MAE={out_v2_m['mae_c']:.4f} C (Hot: {out_v1_m['hottest_core_acc_pct']:.1f}% -> {out_v2_m['hottest_core_acc_pct']:.1f}%)")
    print(f"  Improved FPs: {n_improved}/23, Regressed FPs: {n_regressed}/23")


if __name__ == "__main__":
    main()
