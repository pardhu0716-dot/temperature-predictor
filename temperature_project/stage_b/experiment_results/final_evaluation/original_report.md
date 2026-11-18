# Final Stage B held-out evaluation: frozen Lasso

Evaluation date: 18 November 2026. Source branch stage-b-ml-framework, commit
39b5494b5d9da5a6956aa572fb6b58c399a6f5ef. No GitHub write or deployment occurred.

## Model recovery and freeze

The exact Lasso alpha=0.001 from the completed seven-model comparison was recovered
from runs/lasso/lasso/model.joblib, not retrained. Its saved 11,197-byte bundle is
also byte-identical to the repeated optimizer/regularization Lasso baseline.
Model SHA-256: db1b49992c8a11b95c85a98be861b95502b00b0d3ec680709778236484d9ae80.
Prepared manifest SHA-256: bcabf27285b98bbda358949ec9f3e72124029a9fca341fb59f870b851dde401d.
Source CSV SHA-256: ad78ac968292966a6b68e558f61aff3a1e2e185a553037e23fcd39c63e8bbc70.

FREEZE.json was written before any test.npz open. It pins the estimator's full
parameters, model/CSV/features/preparation hashes, seed, exact split IDs and policy.
Configuration: sklearn Lasso alpha=.001, max_iter=10000, tol=.0001,
selection=cyclic, fit_intercept=True, random_state=42. There was no new fit,
parameter change, model selection or post-test tuning. Lasso.fit was disabled
in the evaluation process. Deserialization was restricted to this Lasso only.

## Split/preprocessing verification

Seed 42 reproduces the saved disjoint 53/12/12 whole-floorplan groups exactly.
Training NPZ contains only the 53 training IDs (1,060 rows); validation contains
only the 12 validation IDs (240 rows). Saved X/Y StandardScaler parameters match
freshly computed training-only means/scales exactly. This verification recomputed
scaler statistics but did not change the scalers or refit the model.

Prior smoke/comparison/optimizer access logs record zero test-array access and
only train.npz/validation.npz opens. Their selection files record validation-only
selection. Existing source code implements training/selection without loading
test.npz. These records and data checks support exclusion of the 12 test layouts
from fitting, scaler fitting, tuning and model selection. Initial dataset-format
validation and preparation necessarily processed all rows to check integrity and
write partitions; that was not held-out performance evaluation or fitting.

The selected model was evaluated on 240 test rows, 20 per floorplan, eight targets
in canonical Core_0_L0 through Core_7_L0 order: **1,920 temperature comparisons**.
Test floorplans: b_fp_014, b_fp_018, b_fp_019, b_fp_020, b_fp_025, b_fp_041, b_fp_042, b_fp_053, b_fp_059, b_fp_070, b_fp_071, b_fp_072.

## Aggregate results

| Metric | Validation | Held-out test | Test minus validation |
|---|---:|---:|---:|
| MAE C | 1.144337 | 1.070281 | -0.074056 |
| RMSE C | 1.463817 | 1.339064 | -0.124753 |
| Maximum absolute error C | 5.068595 | 4.974435 | -0.094160 |


Test MAE is 6.47% lower than validation; RMSE is 8.52% lower; maximum error is
1.86% lower. This held-out split shows no aggregate generalization degradation.
It does not establish performance on every future geometry/material/power regime.
Per-core errors vary: Core_0, Core_4 and Core_5 have higher test RMSE than their
validation RMSE despite the lower aggregate RMSE. The largest test error belongs
to Core_5_L0. No application accuracy threshold was assumed or claimed satisfied.

## Per-core results (240 comparisons each)

| Core | Test MAE C | Test RMSE C | Test max C | Validation RMSE C |
|---|---:|---:|---:|---:|
| Core_0_L0 | 0.724619 | 1.035086 | 2.919328 | 0.731132 |
| Core_1_L0 | 0.928755 | 1.070084 | 2.473772 | 1.404832 |
| Core_2_L0 | 0.710144 | 0.864360 | 1.858043 | 1.492160 |
| Core_3_L0 | 1.158766 | 1.389485 | 3.197002 | 1.521636 |
| Core_4_L0 | 1.039741 | 1.255522 | 2.983195 | 1.192250 |
| Core_5_L0 | 1.560520 | 1.994197 | 4.974435 | 1.701209 |
| Core_6_L0 | 1.272423 | 1.463383 | 2.901194 | 1.898454 |
| Core_7_L0 | 1.167277 | 1.325074 | 2.756861 | 1.473892 |


## Inference and storage

- Single-row latency median: 0.0632965 ms.
- Single-row latency p95: 0.1411827 ms.
- Full 240-row batch: 0.319088 ms.
- Compressed serialized bundle: 11,197 bytes.

Method matches prior experiments: one warmup, 50 timed single-row calls, then
one full-batch timing, using perf_counter_ns. Boundary includes X scaling,
regressor prediction and inverse Y scaling, excluding CSV loading/feature
engineering. Timing is host-specific; no deployment speed guarantee is inferred.
Bundle size includes preprocessing and model metadata, not runtime RAM or only
raw coefficients. Python 3.12.14, NumPy 2.3.5, sklearn 1.8.0, joblib 1.5.3;
OPENBLAS_NUM_THREADS=1, OMP_NUM_THREADS=1, MKL_NUM_THREADS=1. Exact platform and
threadpool details are in evaluation.json. Training time was not remeasured
because retraining was prohibited; original validation_report.json is preserved.

## Integrity, limitations and final-test status

Only this selected model was evaluated. The framework's evaluate_test function
was invoked once successfully, wrote an exclusive final_test_report.json, and
loaded the test NPZ into memory once. Those same arrays were reused to save
predictions, verify metrics and measure inference; repeated prediction calls for
latency are not evaluations of other candidates. Four filesystem opens named
test.npz include its integrity hash, NPZ loading, and byte-preserving artifact
copy operations (including the destination write); they are not four independent
holdout evaluations. No other model was deserialized for evaluation.

A preliminary preflight attempt stopped before FREEZE/test access: its guard
mistook checksum reads of previous model files for candidate-model evaluation.
The artifact-only guard was corrected to restrict joblib deserialization while
allowing integrity checks. The original log is retained as preflight_attempt.log.
No test result existed or was used during that correction; source code was not
changed. The successful evaluation did not encounter prediction/validation errors.

All previous artifacts and the original selected model retain their hashes.
Source checkout remains clean; Stage A/Stage C, raw CSV, matrices and HotSpot
simulation inputs were unchanged. No simulator, refit, deploy, merge, commit or
push occurred. No original-23 external benchmark was performed.

The supplied labels retain the earlier provenance limitation: raw solver outputs
and a complete matching simulation contract were unavailable for independent
label authenticity checking. Metrics are measured against the supplied HotSpot
CSV labels. The 12-layout held-out sample is finite; no statistical confidence
interval or universal accuracy claim is made.

**The test holdout has now been consumed for final reporting. Do not use these
results to select/tune a new model and present a later test score as untouched.**
Future development needs a new independent holdout or a clearly revised protocol.

## Artifact map

- FREEZE.json: immutable pre-test model/configuration/split freeze.
- selected_run/lasso/model.joblib: byte-preserving copy of the selected bundle.
- selected_run/lasso/validation_report.json: original validation evidence.
- selected_run/selection.json and original_validation_selection.json: original
  validation-based selection records.
- selected_run/final_test_report.json: exclusive framework final test report.
- evaluation.json: aggregate/per-core test, validation comparison, latency,
  model size, versions and threadpools.
- test_predictions.csv: all 1,920 per-core truths/predictions/signed/absolute errors.
- per_core_metrics.csv and validation_test_comparison.csv: machine-readable tables.
- reproducibility/: unchanged train/validation/test NPZ, split manifest and
  assignments, X/Y scaling, schema and feature definitions/units/hashes.
- verification.json: model/split/scaler/source preservation and test-audit evidence.
- evaluate_frozen_lasso.py, execution.log and preflight_attempt.log: procedure/logs.
- SHA256SUMS.json: hashes of every other packaged file.

Only load trusted joblib artifacts. The pinned temperature_project code must be
on PYTHONPATH to deserialize ModelBundle. The evaluation runner refuses existing
outputs; it is retained as evidence, not an instruction to rerun test selection.
