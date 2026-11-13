# Stage A Evaluation Benchmark Report — Lightweight Temperature Predictor

## Executive Summary

This report presents the reproducible Stage A evaluation benchmark across all **23 calibrated 2D silicon floorplans** (`Style_1`, `Style_2`, `Style_3`, and `Style_4`) for the three mandatory evaluation metrics:

| Mandatory Evaluation Metric | Measured Result (All 23 Floorplans) | Highlights & Physical Interpretation |
| :--- | :--- | :--- |
| **1. Predictor Accuracy** (Held-out Validation: 460 samples, 3,680 core comparisons) | **RMSE**: `0.003032 K` (`0.003032 °C`)<br>**MAE**: `0.002598 K` (`0.002598 °C`)<br>**Max Error**: `0.007784 K` (`0.007784 °C`) | Matches HotSpot steady-state solver up to HotSpot's 2-decimal (`0.01 K`) output rounding precision ($\sigma_{\text{quant}} = 0.01/\sqrt{12} \approx 0.002887\text{ K}$). |
| **2. Prediction Time & Speedup** | **Pure Matrix Inference (NumPy `R @ P`)**: `2.0096 µs` median (`2.3545 µs` mean)<br>**Pure Matrix Inference (Python stdlib)**: `17.4910 µs` median<br>**HotSpot Steady-State Simulation**: `1821.255 ms` median (`2400.135 ms` mean)<br>**Median Speedup Factor**: **`906,277.2×`** (`1,019,392.4×` mean) | Over **5 orders of magnitude faster** than running a HotSpot grid steady-state simulation (`~0.2 s` vs. `~1 µs`). |
| **3. Model Size (Parameter Storage — Not Runtime RAM)** | **Per Floorplan (`8×8` = `64` params)**: `512 B` raw `float64` (`771 B` CSV on disk / `640 B` `.npy`)<br>**All 23 Floorplans (`23×8×8` = `1,472` params)**: `11,776 B` (`11.50 KiB`) raw `float64` (`17,733 B` = `17.32 KiB` CSV on disk / `14,720 B` = `14.38 KiB` `.npy`) | Measures actual storage size of the calibrated thermal resistance matrix parameters (not runtime RAM consumption). |

---

## 1. Predictor Accuracy vs. HotSpot

### Mathematical Model
For each 2D floorplan with $N = 8$ active cores (`2 Big` $4\times4\text{ mm}$ cores in `[10, 20] W` and `6 Little` $2\times2\text{ mm}$ cores in `[5, 10] W`) and ambient heatsink temperature $T_{\text{ambient}} = 318.15\text{ K}$ ($45.00\text{ }^\circ\text{C}$):

$$\hat{T}_{\text{predicted}} = T_{\text{ambient}} + R \times P$$

where $R \in \mathbb{R}^{8\times8}$ is the floorplan-specific pre-calibrated thermal resistance matrix (in $\text{K/W}$) and $P \in \mathbb{R}^{8}$ is the active core power vector in exact matrix column order.

### Overall Accuracy Summary

| Dataset Split | Floorplans | Samples / FP | Total Samples | Total Core Comparisons | Pooled RMSE (K / °C) | Pooled MAE (K / °C) | Max Absolute Error (K / °C) | Per-FP RMSE Range (K) |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| **Training Set** | 23 | 100 | 2,300 | 18,400 | `0.002767` | `0.002358` | `0.006720` | `0.002632 – 0.002830` |
| **Held-Out Validation Set** | 23 | 20 | 460 | 3,680 | **`0.003032`** | **`0.002598`** | **`0.007784`** | `0.002844 – 0.003262` |

### Physical Explanation of the `~0.003 K` Residual Error
- Steady-state heat conduction in HotSpot with constant material properties is governed by a linear system $G \cdot \Delta T = P$.
- HotSpot formats steady-state output temperatures in `.steady` files to **2 decimal places** (`0.01 K` resolution, e.g., `413.38`).
- Uniform rounding quantization on $[-0.005, +0.005]\text{ K}$ has theoretical RMSE $\sigma = \frac{0.01}{\sqrt{12}} \approx 0.002887\text{ K}$ and theoretical MAE $= 0.002500\text{ K}$.
- Our measured held-out validation RMSE (`0.003032 K`) and MAE (`0.002598 K`) match this theoretical quantization floor, confirming that the calibrated $8\times8$ matrix $R$ captures the exact steady-state thermal physics of each floorplan.

---

## 2. Prediction Time & Speedup vs. HotSpot

### Timing Methodology
- **Pure In-Memory Matrix Inference (`NumPy R @ P`)**: Timed using `time.perf_counter_ns()` across `5,000` inferences per floorplan (`115,000` total runs across 23 floorplans) after 200 warmup iterations, using the pre-loaded $8\times8$ `float64` matrix $R$.
- **Pure In-Memory Python Stdlib Inference**: Timed across `2,000` runs per floorplan using pure Python list comprehensions (`predict_temperatures()` in `predict.py`) with no third-party dependencies.
- **End-to-End Web API Request (`POST /api/predict`)**: Timed across `10` requests per floorplan via the Flask application (includes JSON deserialization, Big/Little power range validation, matrix multiplication, coupling breakdown calculation, and JSON response serialization).
- **End-to-End CLI Subprocess (`python3 predict.py`)**: Timed across `3` full subprocess invocations per floorplan (includes OS process creation, Python interpreter startup, `.lcf`/`.flp`/`.csv` file parsing, and formatted table printing).
- **Equivalent HotSpot Steady-State Simulation**: Timed using `time.perf_counter_ns()` across `3` held-out validation `.ptrace` configurations per floorplan (`69` total grid steady-state runs) using `~/HotSpot/hotspot -c hotspot.config -f <flp> -p <ptrace> -grid_layer_file <lcf> -model_type grid -steady_file <out>`. Every benchmarked HotSpot output was verified to match the saved validation `.steady` temperatures with `0.000000 K` difference.

### Latency & Speedup Comparison Table

| Execution Mode | What Is Included | Median Latency | Mean Latency | Speedup vs. HotSpot (Median) |
| :--- | :--- | ---: | ---: | ---: |
| **1. Pure Matrix Inference (NumPy `R @ P`)** | In-memory `318.15 + R @ P` (`8×8` float64) | **`2.0096 µs`** | `2.3545 µs` | **`906,277.2×`** |
| **2. Pure Matrix Inference (Python stdlib)** | In-memory pure-Python dot product | **`17.4910 µs`** | — | **`104,125.2×`** |
| **3. End-to-End Web API (`POST /api/predict`)** | HTTP JSON parse + validation + `R @ P` + JSON serialize | **`1.6608 ms`** (`1660.8 µs`) | — | **`1,096.6×`** |
| **4. End-to-End CLI (`python3 predict.py`)** | Subprocess spawn + Python init + disk I/O + `R @ P` + print | **`225.47 ms`** | — | **`8.08×`** |
| **5. HotSpot Grid Steady-State Solver** | Full 6-layer 3D grid construction + steady-state solver | **`1821.255 ms`** (`1.8213 s`) | `2400.135 ms` | `1.0×` (Baseline) |

---

## 3. Model Size (Parameter Storage — Not Runtime RAM)

Model size is defined as the **storage size of the calibrated model parameters** required for prediction ($\hat{T} = T_{\text{ambient}} + R \times P$), rather than runtime RAM usage. Each floorplan uses an $8\times8$ thermal resistance matrix $R$ ($64$ `float64` parameters), giving $23 \times 64 = 1,472$ total parameters across all 23 floorplans:

| Representation Format | Parameters (Per FP / All 23 FPs) | Size per Floorplan (Bytes) | Total Size Across All 23 Floorplans | Notes |
| :--- | :---: | ---: | ---: | :--- |
| **Raw Parameter Storage (`8×8 float64`)** | **`64`** / **`1,472`** | **`512 B`** | **`11,776 B`** (`11.50 KiB`) | $8 \text{ rows} \times 8 \text{ cols} \times 8 \text{ bytes/float64} = 512 \text{ bytes}$ per floorplan |
| **Serialized CSV File (`thermal_matrix_fpXXX.csv`)** | **`64`** / **`1,472`** | **`771 B`** | **`17,733 B`** (`17.32 KiB`) | On-disk CSV file with row/column core headers and 6-decimal `K/W` values |
| **Serialized NumPy Binary (`.npy`)** | **`64`** / **`1,472`** | **`640 B`** | **`14,720 B`** (`14.38 KiB`) | `128 B` standard `.npy` array header + `512 B` raw `float64` payload |

---

## 4. Style-Level Summary (`Style_1` – `Style_4`)

| Style | Floorplans | Die Dimensions (mm) | Val RMSE (K) | Val MAE (K) | Val Max Error (K) | NumPy Median (µs) | HotSpot Median (ms) | Median Speedup | Raw Size (B) | CSV Size (B) |
| :--- | ---: | :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| **Style_1** | 3 | `16.0x8.0, 9.0x10.0, 9.0x11.0` | `0.003047` | `0.002591` | `0.006844` | `2.2400` | `1583.29` | **`706,825×`** | `1,536` | `2,313` |
| **Style_2** | 5 | `10.0x6.0, 10.0x8.0, 12.0x6.0, 9.0x10.0` | `0.003050` | `0.002624` | `0.006581` | `1.3295` | `2755.25` | **`2,072,395×`** | `2,560` | `3,855` |
| **Style_3** | 5 | `13.0x12.0, 14.0x6.0, 6.0x14.0, 7.0x14.0` | `0.003097` | `0.002640` | `0.007784` | `2.0880` | `4425.87` | **`2,119,669×`** | `2,560` | `3,855` |
| **Style_4** | 10 | `14.0x14.0` | `0.002985` | `0.002566` | `0.006614` | `2.0215` | `1333.39` | **`659,605×`** | `5,120` | `7,710` |

---

## 5. Per-Floorplan Evaluation Table (All 23 Floorplans)

| # | Floorplan | Die (mm) | FILL | Train RMSE (K) | Val RMSE (K) | Val MAE (K) | Val Max (K) | NumPy Med (µs) | Stdlib Med (µs) | HotSpot Med (ms) | Speedup (×) | Raw / CSV (B) |
| ---: | :--- | :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | `Style_1/fp_001` | `9.0x10.0` | 3 | `0.002763` | `0.003143` | `0.002655` | `0.006844` | `2.2400` | `20.4260` | `1583.29` | **`706,825×`** | `512 / 771` |
| 2 | `Style_1/fp_002` | `9.0x11.0` | 3 | `0.002729` | `0.002914` | `0.002478` | `0.006325` | `2.4259` | `20.3250` | `1152.14` | **`474,942×`** | `512 / 771` |
| 3 | `Style_1/fp_003` | `16.0x8.0` | 3 | `0.002741` | `0.003080` | `0.002640` | `0.006107` | `1.2255` | `9.7645` | `2949.32` | **`2,406,628×`** | `512 / 771` |
| 4 | `Style_2/fp_001` | `10.0x8.0` | 2 | `0.002718` | `0.002989` | `0.002569` | `0.006581` | `1.3295` | `10.6450` | `1466.22` | **`1,102,880×`** | `512 / 771` |
| 5 | `Style_2/fp_002` | `9.0x10.0` | 4 | `0.002794` | `0.003099` | `0.002690` | `0.006459` | `1.2515` | `9.8620` | `912.19` | **`728,907×`** | `512 / 771` |
| 6 | `Style_2/fp_003` | `12.0x6.0` | 1 | `0.002829` | `0.003115` | `0.002665` | `0.006254` | `1.2673` | `10.0920` | `5349.36` | **`4,220,900×`** | `512 / 771` |
| 7 | `Style_2/fp_004` | `10.0x6.0` | 1 | `0.002812` | `0.003139` | `0.002701` | `0.006157` | `2.1253` | `18.1550` | `2755.25` | **`1,296,435×`** | `512 / 771` |
| 8 | `Style_2/fp_005` | `10.0x6.0` | 1 | `0.002759` | `0.002903` | `0.002496` | `0.005775` | `1.9198` | `18.2010` | `3684.29` | **`1,919,152×`** | `512 / 771` |
| 9 | `Style_3/fp_001` | `7.0x14.0` | 9 | `0.002743` | `0.003226` | `0.002682` | `0.007784` | `2.5065` | `20.9635` | `4425.87` | **`1,765,792×`** | `512 / 771` |
| 10 | `Style_3/fp_002` | `13.0x12.0` | 12 | `0.002749` | `0.003181` | `0.002751` | `0.006252` | `2.0880` | `18.3620` | `2078.19` | **`995,301×`** | `512 / 771` |
| 11 | `Style_3/fp_003` | `14.0x6.0` | 3 | `0.002723` | `0.002978` | `0.002611` | `0.005560` | `2.0096` | `20.9140` | `5636.94` | **`2,805,005×`** | `512 / 771` |
| 12 | `Style_3/fp_004` | `6.0x14.0` | 3 | `0.002830` | `0.003026` | `0.002573` | `0.005772` | `2.1159` | `17.3720` | `5623.04` | **`2,657,579×`** | `512 / 771` |
| 13 | `Style_3/fp_005` | `14.0x6.0` | 3 | `0.002783` | `0.003067` | `0.002585` | `0.006175` | `1.9094` | `17.2595` | `3533.90` | **`1,850,790×`** | `512 / 771` |
| 14 | `Style_4/fp_001` | `14.0x14.0` | 17 | `0.002771` | `0.003262` | `0.002838` | `0.006614` | `2.3670` | `19.2810` | `1811.07` | **`765,116×`** | `512 / 771` |
| 15 | `Style_4/fp_002` | `14.0x14.0` | 12 | `0.002803` | `0.003030` | `0.002616` | `0.005643` | `2.0408` | `17.8020` | `1165.76` | **`571,239×`** | `512 / 771` |
| 16 | `Style_4/fp_003` | `14.0x14.0` | 16 | `0.002773` | `0.003064` | `0.002682` | `0.005769` | `1.2428` | `9.8020` | `1305.28` | **`1,050,231×`** | `512 / 771` |
| 17 | `Style_4/fp_004` | `14.0x14.0` | 16 | `0.002764` | `0.002870` | `0.002440` | `0.005711` | `1.9777` | `17.4910` | `1843.79` | **`932,313×`** | `512 / 771` |
| 18 | `Style_4/fp_005` | `14.0x14.0` | 14 | `0.002789` | `0.002985` | `0.002539` | `0.006343` | `1.9228` | `17.0685` | `2069.58` | **`1,076,310×`** | `512 / 771` |
| 19 | `Style_4/fp_006` | `14.0x14.0` | 16 | `0.002729` | `0.002982` | `0.002530` | `0.006454` | `2.5226` | `17.3970` | `1361.50` | **`539,722×`** | `512 / 771` |
| 20 | `Style_4/fp_007` | `14.0x14.0` | 10 | `0.002632` | `0.002859` | `0.002430` | `0.005828` | `2.0022` | `20.7665` | `1662.65` | **`830,432×`** | `512 / 771` |
| 21 | `Style_4/fp_008` | `14.0x14.0` | 16 | `0.002803` | `0.002844` | `0.002412` | `0.005852` | `2.0417` | `10.4775` | `1247.38` | **`610,953×`** | `512 / 771` |
| 22 | `Style_4/fp_009` | `14.0x14.0` | 13 | `0.002811` | `0.002967` | `0.002569` | `0.006357` | `2.3627` | `21.0690` | `1296.33` | **`548,663×`** | `512 / 771` |
| 23 | `Style_4/fp_010` | `14.0x14.0` | 11 | `0.002788` | `0.002965` | `0.002599` | `0.006082` | `1.3721` | `9.8580` | `1007.28` | **`734,090×`** | `512 / 771` |

---

## 6. Verification & Consistency Checks

1. **Zero Discrepancy vs. Saved Validation Results**: Recomputing $\hat{T} = 318.15 + R \times P$ from each floorplan's saved `thermal_matrix_fpXXX.csv` and held-out `validation_samples/ptrace/val_sample_001..020.ptrace` matched `validation_comparisons_160.csv` and `stage_a_all_floorplans_summary.csv` with `0.000000 K` discrepancy across all 23 floorplans.
2. **Zero Discrepancy on Fresh HotSpot Benchmark Runs**: All `69` timed HotSpot steady-state runs produced core temperatures identical (`0.000000 K` difference) to the previously saved `validation_samples/steady/val_sample_*.steady` outputs.
3. **Non-Sequential Core Ordering (`Style_3`)**: All five `Style_3` floorplans (`Style_3/fp_001` through `Style_3/fp_005`) list cores in non-sequential order in their `.flp` / `.ptrace` headers; exact core-name matching (`Core_X_L0`) was preserved throughout.
4. **HotSpot Simulation Time Variation Across Floorplans**: HotSpot execution time scales slightly with aspect ratio and number of floorplan blocks (`FILL` count from 1 to 17), whereas the calibrated Stage A predictor always performs a fixed $8\times8$ matrix-vector product (`512 bytes`) in `~1 µs` regardless of die dimensions or passive `FILL` block count.

---

## 7. How to Reproduce This Benchmark

Run the benchmark script inside the WSL Ubuntu project environment:

```bash
cd /home/pardhu/temperature_project
source .venv/bin/activate
python3 benchmark_stage_a.py
```

Optional flags:
```bash
# Customize number of timed HotSpot runs per floorplan (default: 5) and predictor batches (default: 1000 x 10 = 10,000):
python3 benchmark_stage_a.py --hotspot-runs-per-fp 5 --predictor-batches-per-fp 1000
```

Generated machine-readable and human-readable outputs:
- `stage_a_benchmark_per_floorplan.csv` — Per-floorplan accuracy, latency, speedup, and model size metrics for all 23 floorplans.
- `stage_a_benchmark_overall_summary.csv` — Aggregated overall Stage A benchmark metrics.
- `STAGE_A_EVALUATION_REPORT.md` — This Markdown evaluation summary report.
