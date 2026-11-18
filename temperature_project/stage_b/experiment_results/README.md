# Complete Stage B experiment results

Start with [the Stage B summary](../EXPERIMENT_SUMMARY.md) or the
[complete visual PDF](Stage_B_Complete_Experimental_Comparison.pdf).

| Record | Scope |
|---|---|
| `seven_models/original_comparison.json`, `comparison.csv`, `original_report.md` | Original seven-family measurements |
| `seven_models/effective_parameters.json` | Parameters and actual iteration metadata recovered from saved models without fitting |
| `optimizer_regularization/optimizers.*` | All six existing optimizer entries |
| `optimizer_regularization/regularization.*` | All ten existing regularization entries |
| `optimizer_regularization/combinations.*` | Complete 24-entry dense matrix, including aliases |
| `optimizer_regularization/original_results.json` | All 40 entries, 32 fits, 8 reuses, exact effective parameters and convergence details |
| `optimizer_regularization/overall_ranking.*` | Ranking of 32 unique fits against Lasso |
| `optimizer_regularization/PLAN.md` | Fixed design, native/custom penalty conventions and unsupported combinations |
| `final_evaluation/` | Original frozen reports, 1,920 test predictions and per-core metrics |
| `figure_data.json` | Exact plotted values, transformations and two-sentence captions for each figure |
| `artifact_verification.json`, `publication_verification.json` | Recovery, prediction reproduction, PDF numerical/layout checks |
| `SHA256SUMS.json` | Hashes of all other published Stage B result/model files |

CSV/JSON retain original floating-point measurements; PDF/Markdown values are
rounded only for display. Timing snapshots from the original seven-family and
later optimizer phase remain distinct. Native sklearn Elastic Net is not the
custom dense L1+L2 implementation. All 16 epoch-limited fits remain disclosed.

Validation prediction records are losslessly compressed as `validation_predictions.csv.gz`.
Complete comparison tables are uncompressed CSV/JSON/Markdown. Original narrative
reports are retained verbatim as historical evidence and can refer to the original
Work archive layout; the paths in this index describe the committed layout.

Only one estimator is published: [the tested Lasso](../models/final_lasso/README.md).
Intermediate bundles, NPZ arrays, caches, environments, rendered page previews and
old ZIP archives are intentionally omitted. The existing source CSV plus saved
schema/features/scalers/split manifest define exact preparation; no dataset or
simulation input is duplicated or changed.

## PDF reproduction

The PDF builder consumes only committed result records and existing prediction
CSV files. It does not import the training framework or load an estimator:

```bash
python -m pip install -r temperature_project/stage_b/experiment_results/reporting/requirements.txt
MPLCONFIGDIR=/tmp/stage-b-mpl python temperature_project/stage_b/experiment_results/reporting/build_report.py --work-dir /tmp/stage-b-report
```

It uses DejaVu fonts at `/usr/share/fonts/truetype/dejavu` on Linux. Build previews
and temporary figure PDFs belong outside the repository. Do not rerun held-out
selection or retrain as part of presentation reproduction. The held-out data has
already been consumed for final reporting.
