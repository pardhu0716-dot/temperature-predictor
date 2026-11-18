# Stage B experimental summary

The approved geometry-aware model is **Lasso regression, alpha=0.001**. This
publication preserves the original measured results; it performs no training,
tuning, model selection from test results, simulator run or deployment.

| Frozen Lasso metric | Validation | Final test |
|---|---:|---:|
| MAE (°C) | 1.144337 | 1.070281 |
| RMSE (°C) | 1.463817 | 1.339064 |
| Maximum absolute error (°C) | 5.068595 | 4.974435 |

The model bundle is **11,197 bytes**. Final test inference measured median
0.0632965 ms and p95 0.1411827 ms for one sample, plus 0.319088 ms for all 240
samples. These host-specific timings include scaling/prediction/inverse scaling,
exclude CSV loading and feature engineering, and do not represent runtime RAM.

## Scope and selection

The original seven-family comparison covered OLS, Ridge, Lasso, Elastic Net,
Random Forest, XGBoost and a small MLP. Lasso had the lowest validation RMSE
(1.463817°C) and smallest bundle. The fixed follow-up comprised **40 entries**:
6 optimizer, 10 regularization and 24 optimizer × penalty entries, supported by
**32 unique fits and 8 explicitly reused duplicates**. **16 unique fits reached
300 epochs**; convergence is not assumed. There were no recorded warnings or
failed fits in the completed comparisons.

The best custom gradient configuration was linear Adam + L1 (l1=0.0001), with
validation RMSE 1.739552°C, median latency 0.031745 ms and a 21,566-byte bundle.
Lasso remains the balanced choice for error and storage; small latency differences
are noisy microbenchmarks, not proof of deployment speed improvements.

Native sklearn ElasticNet (alpha=0.001, l1_ratio=0.5) is distinct from custom dense
L1+L2 penalties. Dense combinations used l1=0.0001 and/or l2=0.001, with rates
0.01/0.001/0.003 for Batch GD/SGD/Adam. Tree construction and native sklearn solvers
are not incorrectly labeled GD/SGD/Adam. Exact configurations, actual epochs,
penalty conventions, unsupported pairings and all measured metrics are preserved.

## Data and held-out policy

The 77 b_fp_006…082 layouts have 20 samples each: 1,540 CSV rows, 58 columns and
12,320 core labels. Units are mm, W and °C. Original Stage A and pilot layouts
were excluded. The exact seed-42 floorplan-wise split has 53/12/12 layouts
(1,060/240/240 samples); all 252-feature input and target scalers used training
rows only. Selection used validation RMSE; only the frozen Lasso underwent the
final test, covering 1,920 temperatures across 12 unseen layouts.

All original artifacts were recovered and checksum-verified. The published model
is byte-identical to the tested bundle and reproduces all 1,920 recorded final
predictions exactly without retraining. This is an artifact-integrity check, not
a new model-selection evaluation. The test holdout is now consumed and must not
be used for further tuning. Labels retain the provenance limitation that raw
HotSpot output records and a complete simulation contract were unavailable for
independent authentication. Results are against the user-supplied HotSpot CSV.

## Deliverables

- [Complete visual experimental report (PDF)](experiment_results/Stage_B_Complete_Experimental_Comparison.pdf)
- [Result index and machine-readable records](experiment_results/README.md)
- [Approved final Lasso and preprocessing](models/final_lasso/README.md)

Only new result/model files are added. Stage A, Stage C, the raw CSV, calibrated
matrices, HotSpot inputs, application code and deployment configuration are unchanged.
