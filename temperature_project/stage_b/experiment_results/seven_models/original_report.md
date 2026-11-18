# Stage B core-model validation comparison

Measured on 18 November 2026, from source commit
39b5494b5d9da5a6956aa572fb6b58c399a6f5ef. No source edits or GitHub writes.

## Results, ordered by validation RMSE

| Model | MAE C | RMSE C | Max C | Fit s | Median ms | p95 ms | Bundle bytes |
|---|---:|---:|---:|---:|---:|---:|---:|
| lasso | 1.144337 | 1.463817 | 5.068595 | 0.593554 | 0.066371 | 0.194949 | 11,197 |
| elastic_net | 1.256918 | 1.613383 | 5.211665 | 1.573639 | 0.064492 | 0.925249 | 12,464 |
| ridge | 1.851581 | 2.248562 | 6.226269 | 0.012841 | 0.060352 | 0.167673 | 19,882 |
| ols | 2.472769 | 3.116586 | 10.258925 | 0.035971 | 0.058962 | 0.159387 | 21,006 |
| mlp | 2.879924 | 3.543643 | 11.162182 | 0.888324 | 0.033972 | 0.054582 | 70,037 |
| xgboost | 2.937197 | 3.721698 | 12.133633 | 6.031680 | 2.802488 | 4.499606 | 237,588 |
| random_forest | 3.843019 | 4.868436 | 14.760765 | 13.681567 | 6.312047 | 12.072374 | 4,015,608 |


All seven models succeeded; zero captured warnings or failures. Validation used
240 samples from 12 whole floorplans, totaling 1,920 temperatures per model.
The same existing Ridge prepared directory was reused without regeneration:
seed 42, 53/12/12 floorplan groups, 1,060/240/240 rows, 252 features and training-only
X/Y scaling. All saved models were reloaded and their validation metrics independently
recomputed exactly. The comparison Ridge model is byte-identical to the original
smoke-test model, and all original Ridge artifacts retain their hashes.

## Recommendation

Choose **Lasso alpha=0.001** as the current validation-based tradeoff. It has the
lowest RMSE, MAE and maximum error, and the smallest bundle. Versus Ridge it reduces
validation RMSE by 34.90% and bundle size by 43.68%.
Its 0.066371 ms median inference is close to Ridge's 0.060352 ms on this host.
The MLP measured fastest at 0.033972 ms, but its RMSE is 3.543643 C and its bundle
is 70,037 bytes. These small latency differences are noisy microbenchmarks, not
proof of a deployment speed advantage. Lasso's validation maximum remains 5.068595 C;
no application accuracy threshold was assumed or claimed to be met.

## Exact existing configurations (no tuning)

- OLS: framework default least-squares LinearRegression.
- Ridge: alpha=1.
- Lasso: alpha=0.001, max_iter=10000, random_state=42.
- Elastic Net: alpha=0.001, l1_ratio=0.5, max_iter=10000, random_state=42.
- Random Forest: 128 trees, depth 10, min_samples_leaf=2, random_state=42, n_jobs=1.
- XGBoost: eight CPU XGBRegressor models, each 128 trees, depth 3, learning_rate=0.05,
  squared-error objective, histogram tree method, random_state=42, n_jobs=1.
- NumPy MLP: hidden layer 32 tanh units, eight linear outputs, Adam, learning_rate=0.003,
  max 300 epochs, l2=0.0001; other existing framework defaults unchanged.

Full explicit experiment configs are in requested_experiments.json. Framework/default
estimator settings are retained in each serialized bundle; convergence details are
in verification.json. Each model ran through the existing train() function in its
own output directory so a failed dependency/model would not suppress other models.
No extra optimizer/regularization grids or tuning runs were performed.

## Measurement boundaries and environment

Fit time is estimator.fit only. Inference includes input scaling, model prediction
and inverse target scaling, excluding CSV access and feature engineering. Fifty
single-row calls after warmup determine median/p95; one 240-row batch is timed.
Timing variation between models/runs can include scheduling noise. Reported size is
compressed joblib bundle storage including preprocessing, not RAM or raw coefficients.
Python 3.12.14, numpy 2.3.5, sklearn 1.8.0, joblib 1.5.3, CPU XGBoost 3.4.1.
OPENBLAS_NUM_THREADS=1, OMP_NUM_THREADS=1, MKL_NUM_THREADS=1. Threadpool metadata,
source CSV hash and prepared artifact hashes are recorded in comparison.json.

## Test protection, provenance and integrity

The 12 test floorplans were not evaluated, loaded for training, or accessed as test
arrays. A runtime guard rejected test.npz opens. It recorded only train.npz and
validation.npz access. Original Ridge artifact writes were also prohibited. No test
metrics are included, and no final test evaluation command was run. Split IDs and
already-existing file hashes are metadata only. This archive omits test.npz entirely.

No Stage A/Stage C, raw CSV, matrices, HotSpot inputs, algorithms, deployment or Git
branch was changed. Source checkout remains clean. No HotSpot execution occurred.
Labels retain the provenance limitation identified by the Ridge smoke test: they
are the user-supplied HotSpot labels, but raw solver outputs and a complete matching
simulation contract are unavailable for independent authenticity checking. The
future original-23 external benchmark remains blocked. These are validation results,
not an unbiased final-test accuracy estimate; no original Stage A/pilot data was used.

## Artifact map

comparison.json / comparison.csv: complete consolidated results and warnings.
selection.json: chosen Lasso and pinned model/prepared hashes.
runs/<name>/<name>/model.joblib: seven fitted models with preprocessing.
runs/<name>/<name>/validation_report.json: per-core metrics, time, size, versions.
validation_predictions.csv: all 13,440 validation truth/prediction/error records.
verification.json: independent saved-model metric and integrity checks.
reproducibility/: original train/validation arrays, scaler/schema/features, full
split manifest and assignments; test array intentionally excluded.
execution.log, run_comparison.py, consolidate.py: execution evidence/orchestration.
SHA256SUMS.json: hashes of every other file in this archive.

Do not load untrusted joblib artifacts. These bundles require the matching stage_b.ml
code on PYTHONPATH. Do not invoke evaluate-test during this comparison phase.
