# Stage B final verification

**Result: PASS WITH DOCUMENTED LIMITATIONS — suitable to freeze the supported local 2D demonstration after reviewing these changes.** This is evidence from the current audit, not a claim that all possible bugs or extrapolation errors have been eliminated.

## Regression gate (completed before external comparison)

| Category | Actual outcome |
|---|---|
| Stage B web/v1/v2/model/geometry/API tests | 33 passed, including 16 added audit tests |
| ML framework tests | 12 passed, including optional XGBoost dependency |
| Dataset generation/preparation tests | 25 passed; 1 optional real HotSpot integration skipped |
| Stage A backend regression tests | 5 passed |
| Stage A frontend regression runner | 240 assertions passed |
| Desktop and mobile browser checks | PASS: inputs, custom editing, reset, overlap rejection, charts/heatmap, API error clearing, stale response rejection, navigation |
| Additional browser regression | PASS: pending geometry locks, unlock, v1/v2 selection and numerical display consistency |
| Browser errors / assets | No observed JavaScript errors or failed Stage B assets |
| Stage B responsive layout | No page-level horizontal overflow in tested desktop/mobile viewports; Stage A's existing scrolling was not redesigned |
| Concurrency | 40 mixed v1/v2 requests across 4 threads; deterministic temperatures |
| Production/reference equivalence | v2 portable model agrees with frozen experimental estimator using identical corrected features within 1e-12°C over 1,540 samples; v1 reproduces saved 240 validation predictions |
| Permutation/geometry | All 77 layouts validated; v2 permutations across 1,540 records within 1e-11°C, symmetric and reflected fixture passed |
| Dataset/scaler provenance | Strict real CSV validation; 53/12/12 disjoint floorplan split; original train-only v2 scaler statistics match frozen arrays within 1e-12 |
| Post-comparison independent check | 3,680 rows, 460 unique workloads, 20 per layout; pooled metrics independently recomputed; Stage A API checked on all 23; in-memory legacy v1 baseline matches |

The initial broad dataset discovery command lacked `PYTHONPATH=temperature_project` and failed importing the web package. The corrected invocation passed; this was an invocation issue, not a suppressed assertion. ML unit tests fit synthetic software fixtures; no production model was retrained.

## Fresh accuracy results

460 original validation workloads, 3,680 core observations per model; full rows and all per-floorplan MAE/RMSE/max metrics are in the two requested CSV files.

| Model | MAE °C | RMSE °C | Maximum °C | Hottest-core accuracy |
|---|---:|---:|---:|---:|
| Stage A | 0.002598 | 0.003032 | 0.007784 | 100.00% |
| Stage B v1 | 1.491469 | 1.917708 | 6.535131 | 48.26% |
| Stage B v2, corrected tie rule | 0.460701 | 0.642963 | 2.733158 | 79.78% |

V2 improves MAE by 69.11% over v1, on all 23 layouts. Best v2 layout: Style_4/fp_008 (0.157054°C MAE); worst: Style_3/fp_005 (0.855573°C MAE). Largest single v2 error: Style_1/fp_003, val_sample_018, Core_5_L0, reference 143.510°C versus prediction 146.243158°C. One workload has a tied quantized hottest reference; accepting either tied core gives v2 80.00% instead of the primary historical first-argmax convention's 79.78%.

Historical v1 metrics reproduce to published precision. Historical v2 was MAE 0.460754, RMSE 0.643017, max 2.731624°C. The corrected tie rule gives slightly lower MAE/RMSE and slightly higher maximum error; this is disclosed, with no historical CSV/report edits.

## Timing and storage

Fresh 50-call warm measurements with prebuilt features, including required model scaling but excluding feature construction, validation and HTTP: median Stage A 0.001494 ms; v1 0.060985 ms; v2 0.010248 ms. Mean measured feature construction over the comparison was v1 1.380074 ms, v2 0.898290 ms. These are local microbenchmarks, sensitive to host scheduling, not Render/network performance guarantees. `comparison_summary.json` retains p95 and exact boundaries.

Serialized inference artifacts: v1 11,197 B; v2 3,028 B. These are file sizes, not runtime RAM; they exclude supplemental metadata. Stage A calibrated parameter storage is 512 raw float64 bytes per layout, 11,776 B over 23 layouts.

## Integrity and scope

Base `b9ec778e68605f78e717829266e16b38c7e284c6` has 3,344 tracked files in this checkout. Nine existing **Stage B source** files were modified; 3,335 existing tracked files remain byte-identical; no tracked file was deleted. `integrity_verification.json` lists all nine and original/current model hashes. Stage A code/mathematics/matrices/website, Stage C, raw datasets, simulation inputs, production model files, fitted artifacts and historical benchmark records remain byte-identical. New deliverables are under `stage_b/final_audit/`; new helpers/tests are Stage B only. No GitHub commit/push/merge or Render action occurred.

## Limitations and freeze assessment

- These 23 layouts and the saved Stage B validation/test results have been reused in prior comparisons. They are historical evaluation sets, not a new blind test. No new tuning or candidate selection occurred here.
- Stage A is specifically calibrated on each original layout; Stage B shares geometry-aware parameters. Stage A's advantage on those same layouts does not demonstrate generalization to unseen layouts.
- v1 retains physical core-ID dependence; v2's tie correction changes features on tied geometries while keeping original trained parameters/scalers. Old metrics are explicitly historical.
- Supported Stage B scope: steady-state 2D, eight cores, two 4×4 mm Big and six 2×2 mm Little, fixed 45°C ambient and training thermal assumptions, supported 10–20 W / 5–10 W powers. Zero, negative and out-of-range powers fail validation rather than extrapolating. Valid custom geometry does not guarantee accuracy; rotations and extreme die dimensions have no universal accuracy guarantee.
- No new physical simulations, deployment/load test on Render, or live-site verification was performed. The optional real HotSpot integration remains NOT RUN. Historical report-writing mains were not run; their changed data reconstruction was verified without overwriting reports.
- Browser coverage uses automated Chromium desktop/mobile viewports, not every device/browser. Scrolling on the existing Stage A UI was left unchanged.

All confirmed defects described in the audit were addressed and the relevant checks pass. The supported local v2 demonstration is ready for a reviewed freeze with these limitations; this does not authorize deployment or imply universal model correctness.

## Reproduction

From `temperature_project/`, use Python 3.12 with the existing combined app and ML test dependencies (including scikit-learn 1.8.0, NumPy 2.3.5, joblib 1.5.3 and optional XGBoost). Browser checks additionally require Playwright/Chromium; the existing Stage A JS tests require Node and jsdom.

```bash
export PYTHONPATH="$PWD" OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1
python -m unittest discover -s stage_b/web/tests -v
python -m unittest discover -s stage_b/ml/tests -v
python -m unittest discover -s stage_b -v
python -m unittest discover -s tests -v
python -m stage_b.final_audit.verify_results
```

`compare_23.py` refuses to overwrite completed outputs. To independently repeat into a new scratch directory, without touching saved reports:

```bash
python -c "from pathlib import Path; from stage_b.final_audit.compare_23 import run; run(Path('/tmp/stage_b_comparison_repeat'))"
```

Browser commands require starting the local combined app on the selected port first; see `web/tests/browser_audit.py` and existing `browser_checks.py` CLI options. Localhost is a local-only address, not a public preview.
