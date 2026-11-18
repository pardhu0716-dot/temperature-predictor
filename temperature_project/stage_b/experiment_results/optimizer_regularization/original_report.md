# Stage B optimizer and regularization validation report

Measured on 18 November 2026. Source commit 39b5494b5d9da5a6956aa572fb6b58c399a6f5ef.
Existing framework groups were used unchanged; combinations require no source changes.

40 successful table entries, 32 unique newly measured fits,
8 explicitly reused duplicate entries, 0 failures.
Warnings: 0 table entries. Epoch-budget-limited unique runs: 16.
The experiment plan and penalty strengths were fixed before results were examined.

## Selection

Lowest validation RMSE: **lasso**, 1.463817 C,
MAE 1.144337 C, max error 5.068595 C,
median latency 0.067633 ms, bundle 11,197 bytes.
Historical Lasso alpha=.001 baseline RMSE: 1.463817 C.
Recommendation: retain Lasso alpha=.001. It remains the lowest-RMSE run
and also has the smallest bundle among these runs.
Linear Adam+L1 is a useful measured lower-latency alternative, but its larger bundle
and higher validation error make it a weaker overall fit for the current goal.
The repeated baseline has identical validation metrics and serialized model hash.

The accuracy/latency/size Pareto configurations measured in this phase are:
lasso, elastic_net, linear_gradient_adam_l1, linear_gradient_adam_l1_l2, ridge_weak, ols, linear_gradient_batch_gd_l1_l2. These are single-host microbenchmarks with scheduling noise;
small latency differences are not proven deployment speed improvements. Compare
accuracy and actual compressed bundle sizes as well as timing. No application
accuracy threshold was assumed or claimed to be met.

## Interpretation and limitations

The three dense optimizers use different fixed rates and update counts. These
results compare the requested configurations; they do not establish universal
optimizer superiority or constitute an equal-compute hyperparameter search.
Actual epochs, stopping limits and training loss are recorded for every fit.
All 300-epoch limits are disclosed rather than labeled converged. Adam restores
the best observed TRAINING objective after its patience-based stopping; validation
is not used inside fit. Native Lasso/ElasticNet convergence metadata are retained.

Dense L1 uses a sign subgradient at a nondifferentiable penalty. Dense L1+L2 is
reported as combined penalties, not as the sklearn ElasticNet estimator or equal
alpha convention. Its objective averages squared error over rows and eight outputs,
then adds l1*sum(abs(W))+l2*sum(W**2), with unpenalized biases. Raw strengths are
therefore not directly interchangeable with native sklearn alpha values.

See PLAN.md for every supported/unsupported pairing. In particular, trees do not
use these optimizers, and native sklearn OLS/Ridge/Lasso/ElasticNet do not expose
the DenseRegressor optimizer switch. RF depth/leaf controls are structural
regularization, not L1/L2. XGBoost native penalties are outside the fixed groups
and dense matrix; no additional XGBoost penalty sweep was claimed or performed.

## Data isolation, preservation and measurement

Exactly the prior prepared data were reused: seed 42; 53 training, 12 validation,
12 reserved test floorplans; 1060/240/240 rows; 252 features. All scalers were the
saved training-only scalers; no scaler refit or split regeneration occurred.
Validation metrics use 240 rows x 8 cores = 1,920 temperature comparisons per fit.
Every model was reloaded and validation metrics independently recomputed exactly.

A runtime guard rejected any test.npz read and any previous-experiment write.
Only training/validation NPZ files were opened. No test arrays are included in
this archive. Split assignments and pre-existing hashes are metadata only.
Previous Ridge and seven-model results retain their original hashes. No source,
raw CSV, matrices, Stage A/Stage C, HotSpot inputs, branches or deployments changed.
No simulator run, Git commit/push/merge, final test or original-23 evaluation occurred.

Latency covers engineered features through input scaling, model prediction and
inverse target scaling, excluding CSV reading and feature generation. Fifty
single-row repeats follow warmup. Fit time is estimator.fit only. Storage is
compressed joblib bundle bytes including preprocessing, not RAM or raw coefficients.
Timing is machine-specific: Python 3.12.14, numpy 2.3.5, sklearn 1.8.0, joblib 1.5.3;
BLAS/OpenMP each one thread. Library/threadpool metadata are in results.json.

Labels retain the prior provenance limitation: user-supplied HotSpot results;
raw solver outputs/full simulation contract absent for independent authentication.
Repeated validation-based comparisons may overfit model selection to this holdout.
Final test performance is still unknown; no claim of unbiased final accuracy is made.

## Tables and reproducibility files

- optimizers.md / .csv: existing optimizer group, all six entries.
- regularization.md / .csv: existing ten-entry regularization group.
- combinations.md / .csv: complete 24-entry dense combination matrix.
- overall_ranking.md / .csv: unique fits ordered by validation RMSE, with baseline.
- requested_experiments.json: unchanged group definitions plus run-only matrix.
- results.json: all reports, exact effective hyperparameters, rates, alpha/L1/L2,
  actual/max epochs, stopping details, warnings, measured timings and hashes.
- runs/<canonical>/<canonical>/model.joblib and validation_report.json: every fit.
- validation_predictions.csv: validation truth/prediction/errors for all unique fits.
- verification.json: independent metrics, preservation, source and test checks.
- reproducibility/: original train/validation arrays, scaler/feature/schema metadata,
  full split manifest and assignments. test.npz intentionally excluded.
- warnings_and_failures.json: failures, warnings and epoch caps, without invented values.
- PLAN.md and run_experiments.py: fixed design and repeatable orchestration.

Load only trusted joblib files and put the pinned temperature_project on PYTHONPATH.
The runner uses recorded absolute Work paths; adjust only paths for a future run,
choose a fresh output directory and preserve the exact prepared artifacts. Do not
invoke evaluate-test during this validation phase.

# Optimizers comparison

| Configuration | MAE C | RMSE C | Max C | Fit s | Median ms | Bytes |
|---|---:|---:|---:|---:|---:|---:|
| linear_gradient_batch_gd | 4.913047 | 6.131626 | 19.753439 | 0.470295 | 0.031195 | 22,783 |
| linear_gradient_sgd | 2.609734 | 3.339236 | 11.450524 | 14.884705 | 0.031707 | 22,794 |
| linear_gradient_adam | 2.042145 | 2.624305 | 8.210626 | 0.591499 | 0.032216 | 21,534 |
| mlp_batch_gd | 4.560223 | 5.641336 | 18.663507 | 0.854258 | 0.034923 | 71,615 |
| mlp_sgd | 3.487289 | 4.344457 | 13.000188 | 38.104134 | 0.035047 | 71,680 |
| mlp_adam | 2.972676 | 3.664374 | 12.429757 | 0.884102 | 0.034283 | 69,984 |

All metrics measured on the same 240 validation rows / 1,920 temperatures.
See PLAN.md for objective conventions, unsupported combinations and exact penalty strengths.
Rows marked reused in CSV share one identical effective configuration/fit with the canonical run.


# Regularization comparison

| Configuration | MAE C | RMSE C | Max C | Fit s | Median ms | Bytes |
|---|---:|---:|---:|---:|---:|---:|
| ols | 2.472769 | 3.116586 | 10.258925 | 0.019717 | 0.057389 | 21,006 |
| ridge_weak | 2.320226 | 2.864986 | 8.936102 | 0.005953 | 0.059860 | 19,951 |
| ridge | 1.851581 | 2.248562 | 6.226269 | 0.004817 | 0.062513 | 19,882 |
| lasso_weak | 1.572389 | 1.995122 | 6.667774 | 3.176018 | 0.074416 | 14,216 |
| lasso | 1.144337 | 1.463817 | 5.068595 | 0.544267 | 0.067633 | 11,197 |
| elastic_net | 1.256918 | 1.613383 | 5.211665 | 0.850404 | 0.061702 | 12,464 |
| rf_shallow | 4.345949 | 5.502127 | 17.543137 | 5.684812 | 6.390382 | 560,594 |
| random_forest | 3.843019 | 4.868436 | 14.760765 | 10.301781 | 6.992583 | 4,015,608 |
| mlp_no_penalty | 2.972676 | 3.664374 | 12.429757 | 0.884102 | 0.034283 | 69,984 |
| mlp_l2 | 2.744256 | 3.436108 | 10.922784 | 0.982698 | 0.035388 | 69,862 |

All metrics measured on the same 240 validation rows / 1,920 temperatures.
See PLAN.md for objective conventions, unsupported combinations and exact penalty strengths.
Rows marked reused in CSV share one identical effective configuration/fit with the canonical run.


# Combinations comparison

| Configuration | MAE C | RMSE C | Max C | Fit s | Median ms | Bytes |
|---|---:|---:|---:|---:|---:|---:|
| linear_gradient_batch_gd_none | 4.913047 | 6.131626 | 19.753439 | 0.470295 | 0.031195 | 22,783 |
| linear_gradient_batch_gd_l1 | 4.906363 | 6.122905 | 19.704412 | 0.640547 | 0.031027 | 22,801 |
| linear_gradient_batch_gd_l2 | 4.895775 | 6.109899 | 19.680248 | 0.428349 | 0.032512 | 22,808 |
| linear_gradient_batch_gd_l1_l2 | 4.889140 | 6.101253 | 19.631416 | 0.427506 | 0.030246 | 22,799 |
| linear_gradient_sgd_none | 2.609734 | 3.339236 | 11.450524 | 14.884705 | 0.031707 | 22,794 |
| linear_gradient_sgd_l1 | 2.278589 | 2.903142 | 10.535188 | 15.951352 | 0.031811 | 22,841 |
| linear_gradient_sgd_l2 | 2.321036 | 2.941863 | 10.660859 | 14.500878 | 0.031147 | 22,780 |
| linear_gradient_sgd_l1_l2 | 2.162696 | 2.715251 | 9.685952 | 15.158118 | 0.032546 | 22,789 |
| linear_gradient_adam_none | 2.042145 | 2.624305 | 8.210626 | 0.591499 | 0.032216 | 21,534 |
| linear_gradient_adam_l1 | 1.348842 | 1.739552 | 6.026972 | 0.501287 | 0.031745 | 21,566 |
| linear_gradient_adam_l2 | 2.184941 | 2.728169 | 9.023959 | 0.362902 | 0.031258 | 21,225 |
| linear_gradient_adam_l1_l2 | 1.992796 | 2.500118 | 8.384366 | 0.362256 | 0.030875 | 21,212 |
| mlp_batch_gd_none | 4.560223 | 5.641336 | 18.663507 | 0.854258 | 0.034923 | 71,615 |
| mlp_batch_gd_l1 | 4.554133 | 5.634341 | 18.650753 | 0.841162 | 0.035269 | 71,610 |
| mlp_batch_gd_l2 | 4.544467 | 5.622682 | 18.627883 | 1.053503 | 0.035544 | 71,648 |
| mlp_batch_gd_l1_l2 | 4.538443 | 5.615776 | 18.615274 | 0.870853 | 0.035238 | 71,613 |
| mlp_sgd_none | 3.487289 | 4.344457 | 13.000188 | 38.104134 | 0.035047 | 71,680 |
| mlp_sgd_l1 | 2.877274 | 3.632460 | 11.379377 | 36.425068 | 0.035571 | 71,995 |
| mlp_sgd_l2 | 2.559415 | 3.259206 | 10.930467 | 35.056840 | 0.034666 | 71,626 |
| mlp_sgd_l1_l2 | 2.210666 | 2.818907 | 9.448932 | 36.147255 | 0.035365 | 72,069 |
| mlp_adam_none | 2.972676 | 3.664374 | 12.429757 | 0.884102 | 0.034283 | 69,984 |
| mlp_adam_l1 | 2.197660 | 2.798260 | 10.514853 | 1.016105 | 0.035222 | 70,076 |
| mlp_adam_l2 | 2.744256 | 3.436108 | 10.922784 | 0.982698 | 0.035388 | 69,862 |
| mlp_adam_l1_l2 | 2.079679 | 2.672208 | 9.208479 | 0.832363 | 0.035376 | 70,071 |

All metrics measured on the same 240 validation rows / 1,920 temperatures.
See PLAN.md for objective conventions, unsupported combinations and exact penalty strengths.
Rows marked reused in CSV share one identical effective configuration/fit with the canonical run.


# Overall ranking

| Configuration | MAE C | RMSE C | Max C | Fit s | Median ms | Bytes |
|---|---:|---:|---:|---:|---:|---:|
| lasso | 1.144337 | 1.463817 | 5.068595 | 0.544267 | 0.067633 | 11,197 |
| elastic_net | 1.256918 | 1.613383 | 5.211665 | 0.850404 | 0.061702 | 12,464 |
| linear_gradient_adam_l1 | 1.348842 | 1.739552 | 6.026972 | 0.501287 | 0.031745 | 21,566 |
| lasso_weak | 1.572389 | 1.995122 | 6.667774 | 3.176018 | 0.074416 | 14,216 |
| ridge | 1.851581 | 2.248562 | 6.226269 | 0.004817 | 0.062513 | 19,882 |
| linear_gradient_adam_l1_l2 | 1.992796 | 2.500118 | 8.384366 | 0.362256 | 0.030875 | 21,212 |
| linear_gradient_adam | 2.042145 | 2.624305 | 8.210626 | 0.591499 | 0.032216 | 21,534 |
| mlp_adam_l1_l2 | 2.079679 | 2.672208 | 9.208479 | 0.832363 | 0.035376 | 70,071 |
| linear_gradient_sgd_l1_l2 | 2.162696 | 2.715251 | 9.685952 | 15.158118 | 0.032546 | 22,789 |
| linear_gradient_adam_l2 | 2.184941 | 2.728169 | 9.023959 | 0.362902 | 0.031258 | 21,225 |
| mlp_adam_l1 | 2.197660 | 2.798260 | 10.514853 | 1.016105 | 0.035222 | 70,076 |
| mlp_sgd_l1_l2 | 2.210666 | 2.818907 | 9.448932 | 36.147255 | 0.035365 | 72,069 |
| ridge_weak | 2.320226 | 2.864986 | 8.936102 | 0.005953 | 0.059860 | 19,951 |
| linear_gradient_sgd_l1 | 2.278589 | 2.903142 | 10.535188 | 15.951352 | 0.031811 | 22,841 |
| linear_gradient_sgd_l2 | 2.321036 | 2.941863 | 10.660859 | 14.500878 | 0.031147 | 22,780 |
| ols | 2.472769 | 3.116586 | 10.258925 | 0.019717 | 0.057389 | 21,006 |
| mlp_sgd_l2 | 2.559415 | 3.259206 | 10.930467 | 35.056840 | 0.034666 | 71,626 |
| linear_gradient_sgd | 2.609734 | 3.339236 | 11.450524 | 14.884705 | 0.031707 | 22,794 |
| mlp_l2 | 2.744256 | 3.436108 | 10.922784 | 0.982698 | 0.035388 | 69,862 |
| mlp_sgd_l1 | 2.877274 | 3.632460 | 11.379377 | 36.425068 | 0.035571 | 71,995 |
| mlp_adam | 2.972676 | 3.664374 | 12.429757 | 0.884102 | 0.034283 | 69,984 |
| mlp_sgd | 3.487289 | 4.344457 | 13.000188 | 38.104134 | 0.035047 | 71,680 |
| random_forest | 3.843019 | 4.868436 | 14.760765 | 10.301781 | 6.992583 | 4,015,608 |
| rf_shallow | 4.345949 | 5.502127 | 17.543137 | 5.684812 | 6.390382 | 560,594 |
| mlp_batch_gd_l1_l2 | 4.538443 | 5.615776 | 18.615274 | 0.870853 | 0.035238 | 71,613 |
| mlp_batch_gd_l2 | 4.544467 | 5.622682 | 18.627883 | 1.053503 | 0.035544 | 71,648 |
| mlp_batch_gd_l1 | 4.554133 | 5.634341 | 18.650753 | 0.841162 | 0.035269 | 71,610 |
| mlp_batch_gd | 4.560223 | 5.641336 | 18.663507 | 0.854258 | 0.034923 | 71,615 |
| linear_gradient_batch_gd_l1_l2 | 4.889140 | 6.101253 | 19.631416 | 0.427506 | 0.030246 | 22,799 |
| linear_gradient_batch_gd_l2 | 4.895775 | 6.109899 | 19.680248 | 0.428349 | 0.032512 | 22,808 |
| linear_gradient_batch_gd_l1 | 4.906363 | 6.122905 | 19.704412 | 0.640547 | 0.031027 | 22,801 |
| linear_gradient_batch_gd | 4.913047 | 6.131626 | 19.753439 | 0.470295 | 0.031195 | 22,783 |
