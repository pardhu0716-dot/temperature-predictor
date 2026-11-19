# Stage B v2 Experimental Evaluation Report

- **Generated**: 2026-11-19 09:48:10 UTC
- **Candidate**: `ridge_alpha_0.01` (`ridge`, `{'alpha': 0.01, 'kind': 'ridge'}`)
- **Feature Representation**: `F4_plus_boundary_images` (`63` permutation-equivariant per-core physical features; groups: `self_global, pairwise_coupling, die_spreading, boundary_images`)
- **Serialized Model Artifact**: `candidate_model/model_v2.joblib` (`4,267` bytes, SHA-256 `e2c39c6d1ca55351e67e051b5e35f2c17a107389fd6102b12badb0b22446507a`)
- **Environment**: Python `3.14.4`, NumPy `2.3.5`, scikit-learn `1.8.0`, joblib `1.5.3`, seed `42`
- **Splits**: Exact original `53` train (`1,060` workloads / `8,480` core rows) / `12` validation (`240` workloads) / `12` test (`240` workloads) from `stage_b/models/final_lasso/manifest.json`

## 1. Summary Comparison Across All Evaluation Regimes

| Evaluation Setting | Model | MAE (°C) | RMSE (°C) | Max Err (°C) | Hottest-Core Acc | Full Pipeline Median / p95 (ms) | Model Size (B) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Stage B Validation (12 FPs / 240 wl)** | Stage B v1 (Lasso `0.001`) | `1.144337` | `1.463817` | `5.068595` | `48.75%` | — | `11,197` |
| **Stage B Validation (12 FPs / 240 wl)** | **Stage B v2 (`ridge_alpha_0.01`)** | **`0.419293`** | **`0.572021`** | **`2.246520`** | **`82.92%`** | — | **`4,267`** |
| **Stage B Held-Out Test (12 FPs / 240 wl)** | Stage B v1 (Lasso `0.001`) | `1.070281` | `1.339064` | `4.974435` | `56.67%` | — | `11,197` |
| **Stage B Held-Out Test (12 FPs / 240 wl)** | **Stage B v2 (`ridge_alpha_0.01`)** | **`0.350588`** | **`0.445882`** | **`1.512353`** | **`84.17%`** | — | **`4,267`** |
| **External 23 Stage A FPs (460 wl)** | Stage A Reference (Calibrated) | `0.002598` | `0.003032` | `0.007784` | `100.00%` | `0.001450` / `0.002157` | `17,733` CSV (`11,776` raw) |
| **External 23 Stage A FPs (460 wl)** | Stage B v1 (Lasso `0.001`) | `1.491469` | `1.917708` | `6.535131` | `48.26%` | `0.7918` / `1.4062` | `11,197` |
| **External 23 Stage A FPs (460 wl)** | **Stage B v2 (`ridge_alpha_0.01`)** | **`0.460754`** | **`0.643017`** | **`2.731624`** | **`79.78%`** | **`0.4819` / `0.9048`** | **`4,267`** |

## 2. External 23-Floorplan In-Box vs Out-of-Box Breakdown

| Subset | Model | MAE (°C) | RMSE (°C) | Max Err (°C) | Hottest-Core Acc |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **17 Floorplans Inside Stage B Die Box** (`340` wl) | Stage B v1 | `1.252364` | `1.598901` | `5.029334` | `50.59%` |
| **17 Floorplans Inside Stage B Die Box** (`340` wl) | **Stage B v2** | **`0.355369`** | **`0.498052`** | **`2.731624`** | **`85.00%`** |
| **6 Floorplans Outside Stage B Die Box** (`120` wl) | Stage B v1 | `2.168934` | `2.618036` | `6.535131` | `41.67%` |
| **6 Floorplans Outside Stage B Die Box** (`120` wl) | **Stage B v2** | **`0.759346`** | **`0.939227`** | **`2.462725`** | **`65.00%`** |

- **Floorplan-level wins**: Stage B v2 improved MAE on **`23 / 23`** original Stage A floorplans (regressed on `0 / 23`).
- **Latency breakdown (460 workloads)**:
  - Stage B v1: Feature eng. median `0.6746 ms`, Model inference median `0.1114 ms`, Full pipeline median `0.7918 ms` (p95 `1.4062 ms`).
  - Stage B v2: Feature eng. median `0.4141 ms`, Model inference median `0.0628 ms`, Full pipeline median `0.4819 ms` (p95 `0.9048 ms`).

## 3. Disclosure of Limitations & Leakage Hygiene

- **Strict Selection Hygiene**: Feature ablation (`feature_ablation.csv`) and model/hyperparameter selection (`validation_results.csv`, including the numerical conditioning check `max|w| <= 5.0` that rejected unregularized OLS) used **strictly the 53 training and 12 validation floorplans**. Neither the 12 Stage B test floorplans nor the 23 Stage A floorplans were used for fitting, scaling, or hyperparameter tuning.
- **Test-Reuse Disclosure**: The 12 held-out Stage B test floorplans were previously evaluated for v1 and now evaluated once for frozen v2; the 23 external Stage A floorplans had their v1 error patterns inspected during Phase 1, so they are no longer an untouched blind benchmark.

## 4. Reproducibility Commands
```bash
source ~/.venvs/temperature-predictor-preview/bin/activate
cd ~/temperature-predictor-preview/temperature_project
PYTHONDONTWRITEBYTECODE=1 python stage_b/experiments/stage_b_v2/phase1_error_analysis.py
PYTHONDONTWRITEBYTECODE=1 python stage_b/experiments/stage_b_v2/run_experiments.py
PYTHONDONTWRITEBYTECODE=1 python stage_b/experiments/stage_b_v2/evaluate_final.py
```
