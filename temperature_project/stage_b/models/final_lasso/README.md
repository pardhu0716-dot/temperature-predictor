# Approved final Stage B Lasso

This is the exact saved/tested sklearn Lasso with **alpha=0.001**, not a new fit.
The compressed bundle is **11,197 bytes** and has SHA-256
`db1b49992c8a11b95c85a98be861b95502b00b0d3ec680709778236484d9ae80`.

- `model.joblib`: estimator plus required X/Y scaling metadata, feature hash,
  canonical eight-core order and source CSV hash.
- `schema.json`, `features.json`, `scaling.json`: actual-header mapping, 252-feature
  definitions and fitted training-only preprocessing parameters.
- `manifest.json`, `split_assignments.csv`: exact seed-42 split and source hashes.
- `FREEZE.json`: original pre-test configuration freeze.
- `original_validation_selection.json`, `validation_report.json`: original
  selection and validation evidence; final test reports are under
  `../../experiment_results/final_evaluation/`.
- `simulation_contract.json`: original null provenance contract, not fabricated.

Raw train/validation/test NPZ arrays are intentionally not duplicated. The manifest
retains their original hashes as provenance references, not claims that those
files are stored in this directory. Use the existing dataset and pinned preparation
for reproducing arrays; do not retrain or retune from final test results.

`ModelBundle.predict(X)` expects raw engineered features with shape `(n,252)`,
not eight powers or raw CSV rows. It performs X scaling and inverse target scaling
internally and returns `(n,8)` °C in Core_0_L0 through Core_7_L0 order. Keep the
matching `temperature_project` code on PYTHONPATH and use compatible versions:
Python 3.12.14, NumPy 2.3.5, sklearn 1.8.0, joblib 1.5.3. Only deserialize trusted
joblib artifacts; they execute Python. This publication does not deploy the model
or change the Stage A application.
