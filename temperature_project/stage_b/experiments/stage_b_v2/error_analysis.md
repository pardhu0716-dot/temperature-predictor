# Stage B v1 Error Analysis Report

## 1. Two Evaluation Settings (Strictly Separated)

| Evaluation Setting | Floorplans | Workloads | Core Predictions | MAE (°C) | RMSE (°C) | Max Err (°C) | Mean Signed Bias (°C) | Hottest-Core Acc |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Original Stage B Validation Split** (in-distribution, 12 layouts) | `12` | `240` | `1,920` | `1.144337` | `1.463817` | `5.068595` | `+0.027521` | `48.75%` |
| **External 23 Stage A Floorplans** (unseen layouts, 460 HotSpot runs) | `23` | `460` | `3,680` | `1.491469` | `1.917708` | `6.535131` | `-0.904131` | `48.26%` |

> **Leakage Policy**: The 12 held-out Stage B test floorplans (`b_fp_014`–`b_fp_072`) were **NOT** loaded or inspected in this analysis. All feature and model selection decisions in Phases 2–5 will use strictly the 53 training and 12 validation floorplans.

## 2. Structural Audit of Stage B v1 (`features.py`, `models.py`, `scaling.json`)

- **Core Ordering & Scaling Check**: Verified `CORE_IDS = Core_0_L0..Core_7_L0`, canonical ordering in `data.py`, `StandardScaler` fitted strictly on the 53 training floorplans (`1,060` rows), and exact `mm` / `W` / `°C` unit conversions. **No bugs** were found in core ordering, scaling, geometry parsing, or target inversion.
- **Model Sparsity & Capacity**: Stage B v1 fits **8 separate Lasso models** (`alpha=0.001`) over `252` features (`2,016` total parameters; non-zero counts per core: `[72, 73, 76, 70, 71, 81, 79, 78]`, total `600` non-zero weights). Each core output only learns from `53` training floorplans (`1,060` rows).
- **Three Root Causes Identified in Stage B v1's Representation**:
  1. **Lack of Permutation Equivariance (Core-Slot Overfitting)**: In 2D heat conduction, `Core_0_L0` and `Core_6_L0` obey identical physics. Fitting 8 separate regressors wastes 8× of the effective training data and allows spurious core-index correlations (notice on Stage B's own validation split, `Core_0_L0` MAE is `0.61 °C` while `Core_6_L0` MAE is `1.63 °C`!).
  2. **Missing Power × Geometry Multiplicative Physics**: Steady-state heat conduction is strictly affine in power ($\Delta T_i = \sum_j R_{ij}(\text{geom}) P_j$). Stage B v1 includes raw `die.width`, `die.height`, `die.area`, `x`, `y`, `edge_gap` as *additive* terms independent of power! It lacks physically critical multiplicative terms such as `total_power / die_area`, `total_power / sqrt(die_area)`, `self_power * edge_proximity`, and multi-scale thermal coupling kernels (`P_j / (d^2 + c)`, `P_j * exp(-d / lambda)`, `P_j * log(1 + L / d)`).
  3. **Die Geometry Extrapolation**: Stage B's 53 training floorplans span width `6.717–19.869 mm`, height `6.207–20.025 mm`, area `80.9–350.4 mm²`, max aspect ratio `2.68`. Six of the 23 Stage A floorplans have compact or high-aspect-ratio dies (`6×14 mm`, `14×6 mm`, `12×6 mm`, `10×6 mm`, aspect ratio up to `2.33`, area down to `60 mm²`).

## 3. Worst-Performing Floorplans (External 23-Floorplan Benchmark)

| Rank | Floorplan | Die (mm) | Aspect Ratio | In Training Box? | Stage B MAE (°C) | Mean Signed Bias (°C) | Within-Chip Centered MAE (°C) | Hot-Core Acc |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| 1 | `Style_2/fp_003` | `12.0×6.0` | `2.00` | `False` | `3.0038` | `-2.3283` | `2.2070` | `25.00%` |
| 2 | `Style_3/fp_004` | `6.0×14.0` | `2.33` | `False` | `2.6914` | `-2.6914` | `1.2802` | `35.00%` |
| 3 | `Style_3/fp_005` | `14.0×6.0` | `2.33` | `False` | `2.2470` | `-2.2462` | `1.0624` | `45.00%` |
| 4 | `Style_3/fp_003` | `14.0×6.0` | `2.33` | `False` | `1.8369` | `-1.8241` | `0.8430` | `65.00%` |
| 5 | `Style_2/fp_005` | `10.0×6.0` | `1.67` | `False` | `1.8018` | `-1.6895` | `1.4309` | `25.00%` |
| 6 | `Style_1/fp_003` | `16.0×8.0` | `2.00` | `True` | `1.7737` | `-1.3287` | `1.0384` | `35.00%` |
| 7 | `Style_2/fp_001` | `10.0×8.0` | `1.25` | `True` | `1.7551` | `-0.6231` | `1.6553` | `25.00%` |
| 8 | `Style_3/fp_002` | `13.0×12.0` | `1.08` | `True` | `1.6677` | `-0.9991` | `1.0598` | `30.00%` |
| 9 | `Style_2/fp_002` | `9.0×10.0` | `1.11` | `True` | `1.5972` | `-1.0490` | `1.5101` | `60.00%` |
| 10 | `Style_3/fp_001` | `7.0×14.0` | `2.00` | `True` | `1.5158` | `-0.8390` | `1.3416` | `40.00%` |

## 4. Error Decomposition & Feature Correlations (External 3,680 Core Predictions)

- **Global Chip-Level Offset vs Within-Chip Thermal Coupling Error**:
  - On the **validation split**: overall MAE = `1.1443 °C`, mean absolute per-workload offset = `0.5564 °C`, within-chip centered MAE (after removing per-workload offset) = `1.0454 °C`.
  - On the **external 23 floorplans**: overall MAE = `1.4915 °C`, overall signed bias = `-0.9041 °C`, within-chip centered MAE = `1.1747 °C`.
  - **Diagnosis**: Both chip-level package/spreader resistance error (driven by die area & aspect ratio extrapolation) AND within-chip local thermal coupling error (`~1.05–1.17 °C` centered MAE) contribute substantially.
- **Big vs Little Core Error**:
  - Big cores (`4×4 mm`, `10–20 W`): MAE = `0.9559 °C`, signed bias = `-0.2427 °C`
  - Little cores (`2×2 mm`, `5–10 W`): MAE = `1.6700 °C`, signed bias = `-1.1246 °C`

### Pearson Correlations with Absolute Error (`|err|`) and Signed Error (`pred - gt`)

| Factor | Corr with `|err|` | Corr with `signed_err` | Interpretation |
| :--- | :---: | :---: | :--- |
| `die_width_mm` | `-0.188` | `+0.191` | Narrow/wide dies shift spreader spreading resistance |
| `die_height_mm` | `-0.286` | `+0.291` | Short dies (`6 mm`) show largest positive/negative bias |
| `die_area_mm2` | `-0.345` | `+0.355` | Smaller dies (`60–80 mm²`) have higher error |
| `die_aspect_ratio` | `+0.368` | `-0.393` | Strong positive correlation: high aspect ratios (`>1.8`) increase error |
| `edge_distance_mm` | `-0.265` | `+0.307` | Boundary proximity affects local spreading |
| `mean_core_distance_mm` | `-0.059` | `+0.096` | Core centrality / neighbor coupling distance |
| `min_core_distance_mm` | `-0.286` | `+0.328` | Tightly packed cores amplify nearest-neighbor coupling error |
| `is_big` | `-0.257` | `+0.226` | Core type asymmetry |
| `core_power_w` | `-0.193` | `+0.163` | Power level |
| `total_power_w` | `+0.043` | `-0.030` | Total chip power |
| `actual_temp_c` | `+0.285` | `-0.295` | Higher-temperature compact chips have larger error |

## 5. Hottest-Core Ranking Error Analysis

- **Correctly identified**: `222 / 460` (`48.26%`)
- **Misidentified**: `238 / 460` (`51.74%`)
- **True Temp Gap ($T_1 - T_2$) when CORRECT**: median = `1.170 °C`, mean = `1.370 °C`
- **True Temp Gap ($T_1 - T_2$) when WRONG**: median = `0.850 °C`, mean = `1.005 °C`
- **True Temp Penalty when WRONG (T_true_max - T_predicted_hot)**: median = `1.395 °C`, mean = `1.522 °C`
- **Key Insight**: In these 8-core layouts, the top two cores are often separated by only `~0.5–1.5 °C`, whereas Stage B v1's within-chip relative MAE is `~1.05 °C`. Reducing within-chip relative thermal coupling error directly improves hottest-core identification accuracy.
