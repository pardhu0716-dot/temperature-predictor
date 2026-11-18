# Stage B ML infrastructure verification

Verification date: 18 November 2026. This report describes checks performed for
this implementation; it does not reuse any claimed ML benchmarking results.

## Repository inspection and scope

Base: `stage-b-dataset-preparation`, commit
`ed91bf48c2e9849c97f8626c8cd497fa4d021c77`.
The combined `stage_b/stage_b_training_dataset.csv` was not found in the inspected
GitHub branches. The stated WSL file is unavailable to this cloud environment.
The actual CSV header, 1,540 real rows, 12,320 real labels, units and HotSpot output
provenance are therefore **NOT VERIFIED**. Existing geometry and power manifests
are available and are used for strict input validation. No temperatures were
invented as project data and no simulator was run.

The branch adds only `temperature_project/stage_b/ml/` and the output-ignore file
`temperature_project/stage_b/ml_runs/.gitignore`. All 3,229 tracked files in the
base commit retain their original paths, modes and contents. No existing Stage A
or Stage B file, deployment branch, dataset, matrix or website file was edited.
Main and the deployed Stage A branch were not changed. No Render action occurred.

## Tests performed

- New framework: **12 tests passed**. Temporary synthetic fixtures exercise all
  77 IDs / 1,540 rows / 58 columns, permuted core slots, geometry/power matching,
  unit conversion, invalid/missing/duplicate data rejection, deterministic
  53/12/12 floorplan partitions (1,060/240/240 rows), 252 feature columns,
  training-only scaler fitting, no target features, no-overwrite behavior,
  explicit final-test gating, artifact tampering rejection and CLI smoke paths.
- Tiny synthetic estimator checks cover OLS, Ridge, Lasso, Elastic Net, Random
  Forest, CPU XGBoost and MLP, each with eight outputs. Batch GD, SGD and Adam
  learn a synthetic linear problem reproducibly. Regularization and known-error
  metric calculations are checked. These are software tests, not thermal-model
  accuracy or performance benchmarks.
- Future external compatibility checks reject missing/changed provenance,
  ambient mismatches and Stage A predictions declared as labels. No external
  floorplan evaluation was performed.
- Existing Stage B suite: **26 tests, 25 passed, 1 skipped**. The skipped test
  requires an explicit real-HotSpot opt-in; no simulation was authorized or run.
- Existing Stage A backend suite: **5 tests passed**.
- Existing Stage A frontend suite: **240 assertions passed**, including the
  original response-order regression checks. These are local regression checks,
  not a new browser or live-site deployment verification.

Environment: Python 3.12.14, NumPy 2.3.5, scikit-learn 1.8.0, joblib 1.5.3 and
optional CPU-only XGBoost 3.4.1. Test dependencies were installed outside the
checkout. Fixture CSVs, fitted models and reports were temporary and are not
committed. Full real-data experiment groups and final holdout evaluation were
not run. No real MAE/RMSE, inference latency or model-size claims are made.

## Required WSL follow-up

Follow README.md to inspect the actual CSV, explicitly map its header and units,
verify sample IDs/powers against the original workload manifests, and document
actual simulator configuration/material provenance. If identifiers or schema do
not match, investigate the original export rather than editing labels or
loosening validation silently. Then run the single-Ridge smoke test. Preserve
the holdout until configurations are frozen. Original Stage A floorplans and
earlier five pilots remain excluded. The original-23 external benchmark requires
separate authoritative geometry and independent compatible HotSpot labels.
