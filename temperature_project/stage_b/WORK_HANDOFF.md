# Stage B v2 → ChatGPT Work Handoff

## 1. Repository & Branch Checkpoint

- **Repository**: `https://github.com/pardhu0716-dot/temperature-predictor`
- **Branch**: `stage-b-web-integration`
- **Finalized Stage B v2 Commit**: `dba114ddd7ac3be8ba711087b9e3392e19f76e70` (`Finalize Stage B v2 physics-informed Ridge predictor and web integration`)
- **Status**: Verified in a fresh isolated clone (`/tmp/stage-b-handoff-check`); committed locally, **not pushed, merged, or deployed**.

---

## 2. Production Architecture & Tracked Files

The combined Flask application (`temperature_project/combined_app.py`) serves Stage A at `/` and Stage B at `/stage-b`. Stage A and Stage C are untouched.

### Stage B v2 Production Model & Feature Modules (Default)
- `temperature_project/stage_b/models/v2_ridge/model_v2.npz`
  - **Format**: Portable, pickle-free NumPy archive (`allow_pickle=False`)
  - **Size**: `3,028` bytes (`512 B` raw float64 Ridge weights + intercept; `1,536 B` total raw float64 arrays including training-only scaler)
  - **SHA-256**: `edbae799105b57134ab99cfe3a68db2e4afd9c7aeb0ae78b3ec49476899c2175`
- `temperature_project/stage_b/models/v2_ridge/metadata.json` — hyperparameters (`Ridge`, `alpha=0.01`), 63 ordered feature names, `53/12/12` split IDs (`seed=42`), integrity hashes, and audited metrics
- `temperature_project/stage_b/models/v2_ridge/README.md` — standalone loading instructions and model card
- `temperature_project/stage_b/models/v2_ridge/export_v2_production.py` — reproducible export & `< 1e-12 °C` equivalence verifier
- `temperature_project/stage_b/ml/features_v2.py` — production 63-feature permutation-equivariant physics feature builder (`F4`: `self_global`, `pairwise_coupling`, `die_spreading`, `boundary_images`)
- `temperature_project/stage_b/ml/v2_model.py` — standalone NumPy-only loader (`load_v2_production_model()`)

### Stage B v1 Baseline Model (Preserved & Selectable)
- `temperature_project/stage_b/models/final_lasso/model.joblib` — `11,197` bytes, `SHA-256: db1b49992c8a11b95c85a98be861b95502b00b0d3ec680709778236484d9ae80` (`Lasso`, `alpha=0.001`, 252 features via `stage_b/ml/features.py`)

### Combined Web Application & Tests
- `temperature_project/combined_app.py` — entry point registering Stage B blueprint on unchanged Stage A Flask app
- `temperature_project/stage_b/web/service.py` — loads v2 (`model_v2.npz`, default) and v1 (`model.joblib`, selectable via `model_version: "v1"`); zero experimental imports, zero online fitting
- `temperature_project/stage_b/web/templates/stage_b.html` & `stage_b/web/static/stage_b.js` — Stage B UI with `#b-model` v2/v1 selector, 77-layout dropdown, custom geometry editor, SVG heatmap, and 8-bar chart
- `temperature_project/stage_b/web/README.md` — updated Stage B v1/v2 documentation
- `temperature_project/stage_b/web/tests/test_integration.py` & `test_v2_integration.py` — 17 automated integration tests (including 240 v1 replay cases, fresh-process v2 `.npz` loading, v1/v2 switching, and API permutation equivariance)

### Experimental & Audit Trail (Tracked for Reproducibility)
- `temperature_project/stage_b/experiments/stage_b_v2/` — Phase 1–3 scripts, `candidate_model/model_v2.joblib` (`4,267 B`, `SHA-256: e2c39c6d...`), ablation & validation CSVs, SVG charts, and `stage_b_v2_report.md`

---

## 3. Dependencies & Fresh-Clone Setup

Python `3.11+` is required (tested on `3.12.14` and `3.14.4`). From `temperature_project/` in a fresh clone:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r stage_b/web/requirements.txt
```

Pinned inference dependencies (`stage_b/web/requirements.txt` + `requirements.txt`):
- `flask>=3.0.0` (`3.1.3`)
- `numpy==2.3.5`
- `scikit-learn==1.8.0`
- `joblib==1.5.3`

Start the combined website locally (zero dependency on any other WSL directory or HotSpot):

```bash
cd temperature_project
python -m flask --app combined_app run --host 127.0.0.1 --port 5205
```

- Stage A: `http://127.0.0.1:5205/`
- Stage B: `http://127.0.0.1:5205/stage-b`
- Model metadata API: `GET http://127.0.0.1:5205/api/stage-b/model`

---

## 4. Verification & Test Commands (Verified on Fresh Clone)

```bash
cd temperature_project
# 1. Stage B web integration & v2 production tests (17 passed)
PYTHONDONTWRITEBYTECODE=1 python -m unittest discover -s stage_b/web/tests -v

# 2. Stage A saved-model & hash integrity tests (5 passed)
PYTHONDONTWRITEBYTECODE=1 python -m unittest discover -s tests -v

# 3. Stage B ML framework tests (11 passed, 1 skipped optional xgboost)
PYTHONDONTWRITEBYTECODE=1 python -m unittest discover -s stage_b/ml/tests -v

# 4. Production .npz vs experimental .joblib equivalence (0.000e+00 °C max diff)
PYTHONDONTWRITEBYTECODE=1 python stage_b/models/v2_ridge/export_v2_production.py

# 5. Optional Stage A frontend regression suite (requires Node + jsdom; 240 assertions)
node tests/frontend_regressions.cjs
```

---

## 5. Benchmark Datasets & Audited Results

All benchmark datasets required for evaluation are **tracked in Git**:
- **Stage B 77-floorplan dataset**: `temperature_project/stage_b/stage_b_training_dataset.csv` (`1,540` rows, `58` columns, `SHA-256: ad78ac968292966a6b68e558f61aff3a1e2e185a553037e23fcd39c63e8bbc70`) + hashed geometries under `stage_b/dataset_v1/`
  - Split (`seed=42`, from `stage_b/models/final_lasso/manifest.json`): `53` train (`1,060` workloads) / `12` validation (`240`) / `12` held-out test (`240`)
- **External 23 Stage A floorplans**: `temperature_project/dataset/Sample_dataset/2D_floorplans/<Style>/<fp>/validation_samples/{ptrace,steady}/val_sample_001..020.*` (`460` HotSpot workloads, `3,680` core observations)

| Dataset | Stage B v1 MAE | Stage B v2 MAE | Stage B v2 RMSE | Stage B v2 Hottest-Core Acc |
| :--- | :---: | :---: | :---: | :---: |
| Validation (`12` FPs / `240` wl) | `1.144337 °C` | `0.419293 °C` | `0.572021 °C` | `82.92%` |
| Held-out test (`12` FPs / `240` wl) | `1.070281 °C` | `0.350588 °C` | `0.445882 °C` | `84.17%` |
| External 23 Stage A FPs (`460` wl) | `1.491469 °C` | `0.460754 °C` | `0.643017 °C` | `79.78%` |

- **Disclosure**: Both the 12 held-out test floorplans and 23 external Stage A floorplans were reused across v1/v2 evaluations and are historical reference benchmarks, not untouched blind test sets.

---

## 6. Local-Only / Ignored Items & Caveats

Nothing essential to running or testing the combined website is missing from Git. Non-essential local-only items:
1. **`temperature_project/stage_b/benchmarks/`** (untracked local Stage A vs Stage B v1 comparison folder, including `491 KB` `predictions.csv`): **Not required** for the website or any unit tests. Only `stage_b/experiments/stage_b_v2/phase1_error_analysis.py` and `evaluate_final.py` read its CSVs if re-run.
2. **`node_modules/jsdom`** (gitignored): Optional; only needed to run `node tests/frontend_regressions.cjs` (install via `npm install jsdom` in a fresh clone).
3. **`stage_b/simulation_results/` & HotSpot binary** (gitignored / local-only): Not needed for inference or testing.
4. **Note on `stage_b/experiments/stage_b_v2/independent_audit.py`**: Written before v2 became the default in `stage_b/web/service.py`; its `V1Predictor().predict(...)` call omits `model_version="v1"`, so on current `HEAD` it invokes v2 for both external lines unless `model_version="v1"` is passed.
5. **Security & Size Audit**: `git grep` secret scan on `dba114d` found **0 secrets**; largest added file in `dba114d` is `32,308` bytes (`evaluate_final.py`).
