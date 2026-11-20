# Lightweight 2D Temperature Predictor (Stages A, B, and C)

An academic thermal modeling project providing three complementary, HotSpot-calibrated 2D multicore temperature predictors served locally through a single self-contained Flask application:

- **Stage A (`/`)** — **2D Steady-State Linear Predictor** (`T = 45.0 °C + R @ P`) calibrated per floorplan across 23 2D heterogeneous floorplans (`Style_1..Style_4`).
- **Stage B (`/stage-b`)** — **2D Geometry-Aware ML Predictor** predicting steady-state core temperatures across unseen 2D floorplan geometries using permutation-equivariant physics-informed features (`v2` Ridge default + `v1` Lasso selectable).
- **Stage C (`/stage-c`)** — **2D Transient Thermal Predictor** using a discrete-time two-speed (dual-RC) architecture calibrated per floorplan across all 23 original 2D floorplans over a `2.00 s` transient horizon (`200` steps at `dt = 0.01 s`).

HotSpot is used offline only for reference simulations during calibration and evaluation. Web and CLI inference never invoke HotSpot or recalibrate frozen models.

---

## Stage Overview & Audited Performance

| Stage | Route | Architecture | Supported Floorplans | Audited Accuracy | Inference Latency | Raw Model Size |
| :--- | :---: | :--- | :--- | :--- | :--- | :--- |
| **Stage A** | `/` | Steady-state linear superposition (`8×8` `R` matrix per die) | 23 calibrated 2D floorplans (`Style_1..Style_4`, `8` cores: `2` Big + `6` Little) | Pooled held-out RMSE **`0.00303 °C`** (`0.00260 °C` MAE; `460` workloads) | **`2.01 µs`** median NumPy (`1.66 ms` web API) | **`512 B`** / floorplan (`64` `float64` params; `11.5 KiB` all 23) |
| **Stage B** | `/stage-b` | Geometry-aware ML (`v2` 63-feature Ridge `.npz` default; `v1` 252-feature Lasso selectable) | 77 Stage B 2D layouts + custom 8-core 2D geometries | `v2` held-out test (`12` FPs / `240` wl): **`0.3506 °C`** MAE / **`0.4459 °C`** RMSE; external 23 Stage A FPs: **`0.4608 °C`** MAE / **`0.6430 °C`** RMSE | Sub-millisecond NumPy inference | **`3,028 B`** portable `.npz` (`512 B` Ridge weights + intercept) |
| **Stage C** | `/stage-c` | Discrete-time two-speed (dual-RC) transient model (`R` + `β`, `τ_fast`, `τ_slow`) | All 23 original 2D floorplans (`Style_1..Style_4`, `8` active cores) | Unseen held-out Workload C: mean RMSE **`1.6782 °C`** (`1.3516 °C` MAE; **`73.41%`** mean per-floorplan RMSE reduction vs. one-speed `6.4225 °C`; `184/184` cores improved) | **`405.91 µs`** median for 200 steps (`~2.03 µs`/step) | **`704 B`** / floorplan (`88` `float64` params: `64` `R` + `24` transient; `15.8 KiB` all 23) |

*Note: Latency figures depend on host hardware and measurement boundaries; reported values are audited benchmark measurements rather than universal guarantees.*

---

## Local Installation & Dependencies

Python **3.10+** (tested on `3.12` and `3.14`) on Linux / Ubuntu / WSL:

```bash
git clone https://github.com/pardhu0716-dot/temperature-predictor.git
cd temperature-predictor/temperature_project
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install -r stage_b/web/requirements.txt
```

Runtime dependencies (no HotSpot binary or cloud service required):
- `flask>=3.0.0`
- `numpy>=1.26.0` (Stage B pins `numpy==2.3.5`, `scikit-learn==1.8.0`, `joblib==1.5.3` in `stage_b/web/requirements.txt`)

Stage C requires only `flask` and `numpy` (no additional packages).

---

## Running the Combined Three-Stage Website Locally

From `temperature_project/` with your virtual environment active:

```bash
python -m flask --app combined_abc_app run --host 127.0.0.1 --port 5205
```

Open in your browser:
- **Stage A (Steady-State Linear):** `http://127.0.0.1:5205/`
- **Stage B (Geometry-Aware ML):** `http://127.0.0.1:5205/stage-b`
- **Stage C (2D Transient Two-Speed):** `http://127.0.0.1:5205/stage-c`

Alternative isolated entrypoints (all existing files remain untouched):
- Stage A only (`app.py`): `python app.py` (default port `5000`)
- Stage A + Stage B (`combined_app.py`): `python -m flask --app combined_app run --host 127.0.0.1 --port 5205`
- Stage C standalone: `python -m stage_c.web.standalone_app` (default port `5051`)

---

## Supported Floorplans & Calibrated Input Limitations

- **Ambient & Initial Temperature:** Fixed at **`45.0 °C` (`318.15 K`)** matching `hotspot.config`. Stage C transient models require `T_init = T_ambient = 45.0 °C` (`F[0] = S[0] = 0`).
- **Calibrated Power Domain (Stages A & C):**
  - **Big cores:** `10.0 W – 20.0 W`
  - **Little cores:** `5.0 W – 10.0 W`
  - **Stage C Horizon:** `200` steps at `dt = 0.01 s` (`2.00 s` total).
  - **Extrapolation Handling on `/stage-c`:** Physically valid power inputs within `0.0–50.0 W` that fall outside the calibrated Big (`10–20 W`) or Little (`5–10 W`) range are computed transparently and clearly badged as **EXTRAPOLATED** with per-core warnings. Non-finite, negative, or `> 50 W` values are rejected with HTTP `400`.
- **Stage B Custom Geometries:** Supports 8-core 2D floorplans following the HotSpot layer configuration; predictions on layouts far outside the 77-layout training distribution should be treated as extrapolation.
- **Model Integrity:** Stage C verifies live SHA-256 checksums of all 23 frozen parameter JSON files on startup against `stage_c/transient/full_floorplan_expansion/consolidated/all_23_floorplans_summary.csv`.

---

## Verification & Automated Tests

From `temperature_project/`:

```bash
# Stage A saved-model & integrity tests (5 tests)
PYTHONDONTWRITEBYTECODE=1 python -m unittest discover -s tests -v

# Stage B web integration, v2 production & final audit tests (33 tests)
PYTHONDONTWRITEBYTECODE=1 python -m unittest discover -s stage_b/web/tests -p 'test_*.py' -v

# Stage C isolated web, SHA-256 integrity, numerical parity & Blueprint isolation tests (8 tests)
PYTHONDONTWRITEBYTECODE=1 python -m unittest discover -s stage_c/web/tests -v
```

---

## Repository Structure (Key Entrypoints)

```text
temperature_project/
  app.py                              Stage A Flask app (unchanged)
  combined_app.py                     Combined Stage A + Stage B entrypoint (unchanged)
  combined_abc_app.py                 Combined Stage A + Stage B + Stage C entrypoint
  predict.py                          Stage A CLI predictor
  dataset/Sample_dataset/2D_floorplans/   23 calibrated 2D floorplans & Stage A R matrices
  stage_b/                            Stage B dataset, v1/v2 models, web Blueprint & tests
  stage_c/
    web/                              Isolated Stage C Flask Blueprint, registry, UI & tests
    transient/
      models/two_speed/parameters/    Frozen Style_1/fp_001 two-speed JSON
      multi_floorplan_pilot/          Generalized two-speed model + 3 frozen pilot JSONs
      full_floorplan_expansion/       19 frozen JSONs, 23-FP summary manifest & final audit report
```

---

## Explicit Scope Limitations

This repository implements **2D offline-calibrated steady-state and transient temperature prediction only**. The following are **not implemented**:
- **No 3D stacked-die thermal prediction** (any `3D_floorplans/` files in the sample dataset are uncalibrated dataset assets only).
- **No live CPU/hardware power or thermal sensor monitoring.**
- **No proactive Dynamic Thermal Management (DTM), DVFS, or task-scheduling control loop.**
- **No cloud deployment or hosted production service configuration.**
