# Stage A vs Stage B: all 23 original floorplans

Fresh evaluation of fixed artifacts after bug fixes. These 23 layouts have been
used in earlier v1/v2 evaluations and are **not an untouched blind test set**.
No fitting, calibration, tuning, model replacement or HotSpot rerun occurred.

| Floorplan | Stage A MAE (°C) | Stage B v1 MAE (°C) | Stage B v2 MAE (°C) | Stage B v2 RMSE (°C) | Stage B v2 Max Error (°C) |
|---|---:|---:|---:|---:|---:|
| Style_1/fp_001 | 0.002655 | 1.495469 | 0.610157 | 0.734875 | 1.827384 |
| Style_1/fp_002 | 0.002478 | 1.271685 | 0.583705 | 0.754040 | 1.959689 |
| Style_1/fp_003 | 0.002640 | 1.773722 | 0.850910 | 1.064232 | 2.733158 |
| Style_2/fp_001 | 0.002569 | 1.755068 | 0.443629 | 0.563923 | 1.338447 |
| Style_2/fp_002 | 0.002690 | 1.597211 | 0.366728 | 0.452897 | 1.205568 |
| Style_2/fp_003 | 0.002665 | 3.003803 | 0.836334 | 1.061441 | 2.462725 |
| Style_2/fp_004 | 0.002701 | 1.432627 | 0.642939 | 0.779067 | 1.915871 |
| Style_2/fp_005 | 0.002496 | 1.801827 | 0.618870 | 0.768638 | 2.356518 |
| Style_3/fp_001 | 0.002682 | 1.515821 | 0.397370 | 0.569329 | 1.729476 |
| Style_3/fp_002 | 0.002751 | 1.667726 | 0.232189 | 0.273176 | 0.747574 |
| Style_3/fp_003 | 0.002611 | 1.836907 | 0.772870 | 0.956719 | 2.312895 |
| Style_3/fp_004 | 0.002573 | 2.691439 | 0.828292 | 0.989642 | 2.191328 |
| Style_3/fp_005 | 0.002585 | 2.247002 | 0.855573 | 1.035054 | 2.439305 |
| Style_4/fp_001 | 0.002838 | 0.786987 | 0.174581 | 0.211695 | 0.509216 |
| Style_4/fp_002 | 0.002616 | 1.328776 | 0.233434 | 0.282170 | 0.690683 |
| Style_4/fp_003 | 0.002682 | 0.988923 | 0.380534 | 0.433963 | 0.859933 |
| Style_4/fp_004 | 0.002440 | 1.511106 | 0.310970 | 0.383378 | 0.927553 |
| Style_4/fp_005 | 0.002539 | 1.195169 | 0.256319 | 0.344791 | 0.905213 |
| Style_4/fp_006 | 0.002530 | 0.852371 | 0.200596 | 0.242773 | 0.581869 |
| Style_4/fp_007 | 0.002430 | 0.905977 | 0.224947 | 0.269243 | 0.629339 |
| Style_4/fp_008 | 0.002412 | 1.026466 | 0.157054 | 0.208487 | 0.649163 |
| Style_4/fp_009 | 0.002569 | 0.632494 | 0.322235 | 0.396343 | 0.931226 |
| Style_4/fp_010 | 0.002599 | 0.985218 | 0.295894 | 0.379590 | 0.930640 |

## Overall (460 workloads, 3,680 per-core observations per predictor)

| Predictor | MAE °C | RMSE °C | Max °C | Hottest-core accuracy |
|---|---:|---:|---:|---:|
| stage_a | 0.002598 | 0.003032 | 0.007784 | 100.00% |
| stage_b_v1 | 1.491469 | 1.917708 | 6.535131 | 48.26% |
| stage_b_v2 | 0.460701 | 0.642963 | 2.733158 | 79.78% |


V2 improves MAE over v1 by 69.11% and wins on
23/23 floorplans. Best v2 MAE: Style_4/fp_008 (0.157054°C).
Worst v2 MAE: Style_3/fp_005 (0.855573°C).

Stage A uses a separately calibrated 8×8 resistance matrix for each of these
layouts; this is its in-distribution advantage. Stage B shares one geometry-aware
model across layouts and was trained on the distinct 77-layout dataset. Better
Stage A accuracy here does not imply that its matrices generalize to new layouts.

## Alignment and methods

Exactly val_sample_001…020 per floorplan were paired by sample ID, with core names
canonicalized independently for geometry, ptrace columns, steady labels and matrix
rows/columns. Kelvin labels were converted with C=K−273.15; Stage A applies 45+R×P
without retraining. Original core/FILL bounds, overlap/area coverage, zero FILL
powers, dimensions, finite values and matching trace/reference sets were checked.
All 920 trace/reference file hashes are saved in comparison_summary.json.

Metrics pool all 3,680 observations (not an average of rounded floorplan metrics).
Hottest-core accuracy uses canonical first-argmax as in historical reports; tie-aware
accuracy and the count of quantized reference ties are provided separately.

## Comparison with historical records

The v1 reference was previously MAE 1.491469°C / RMSE 1.917708°C / max 6.535131°C.
The historical v2 record was MAE 0.460754°C /
RMSE 0.643017°C / max 2.731624°C.
New v2 MAE differs by -0.000052685°C and
RMSE by -0.000054036°C. Historical files were
not changed. The intentional inference correction averages tied nearest-neighbor
powers instead of choosing one by core index. Frozen coefficients and scalers are
unchanged; the original legacy feature rule is retained only for scaler-provenance
audits. This is a correctness fix, not a hyperparameter adjustment.

The ten largest observed v2 errors, each linked to layout/sample/core, are in
comparison_summary.json. Per-core CSV predictions preserve full floating-point
values. Extreme custom-layout extrapolation and v1's identity-dependent features
remain model limitations; no claim of universal geometric/rotation invariance is
made. Input-row reordering preserves named mapping in both versions, while physical
core relabeling is equivariant in corrected v2 and not guaranteed in frozen v1.

Latency uses one warmup and 50 single-workload calls including required scaling;
it excludes feature construction and HTTP. Model sizes are 11,197 B for v1 and
3,028 B for v2, not runtime RAM. These timing/storage records are measured separately
from accuracy and do not replace previous benchmarks.
