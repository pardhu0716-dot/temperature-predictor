# Lightweight Temperature Predictor — Stage A

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/)
[![Framework](https://img.shields.io/badge/Framework-Flask%20%7C%20NumPy-green.svg)](https://flask.palletsprojects.com/)
[![HotSpot](https://img.shields.io/badge/Simulator-HotSpot%206.0-orange.svg)](http://lava.cs.virginia.edu/HotSpot/)
[![License](https://img.shields.io/badge/License-Academic%20Submission-lightgrey.svg)]()

> **Ultra-fast, sub-microsecond steady-state thermal temperature predictor for heterogeneous multi-core silicon floorplans.**  
> Calibrated and validated across 23 distinct 2D silicon floorplans spanning four architectural styles, achieving **over 900,000× speedup** over numerical grid simulation with **<0.008 °C maximum error** against HotSpot.

---

## Table of Contents

1. [Executive Summary & Objectives](#executive-summary--objectives)
2. [Stage A Scope & Exclusions](#stage-a-scope--exclusions)
3. [System Architecture & Mathematics](#system-architecture--mathematics)
4. [Project Structure](#project-structure)
5. [Prerequisites & Environment Setup](#prerequisites--environment-setup)
6. [How to Run](#how-to-run)
   - [CLI Temperature Predictor (`predict.py`)](#1-cli-temperature-predictor)
   - [Interactive Web Application (`app.py`)](#2-interactive-web-application)
   - [Reproducible Evaluation Benchmark (`benchmark_stage_a.py`)](#3-reproducible-evaluation-benchmark)
   - [Batch Calibration & Pipeline (`run_all_stage_a.py`)](#4-batch-calibration-pipeline)
   - [Automated Verification Suite (`test_website.py`)](#5-automated-verification-suite)
7. [Benchmark Evaluation Results](#benchmark-evaluation-results)
   - [Accuracy vs. HotSpot](#1-accuracy-vs-hotspot)
   - [Inference Latency & Speedup](#2-inference-latency--speedup)
   - [Model Storage Footprint](#3-model-storage-footprint)
8. [Live Timing vs. Benchmark Timing](#live-timing-vs-benchmark-timing)
9. [Limitations & Future Extensions](#limitations--future-extensions)

---

## Executive Summary & Objectives

Modern operating systems, runtime governors, and dynamic thermal managers require fast temperature estimates to prevent thermal throttling and hot spots. However, standard numerical thermal solvers such as **HotSpot** take **~1.8 to 2.4 seconds** per steady-state grid evaluation, rendering them unusable for millisecond-level scheduling decisions.

The **Lightweight Temperature Predictor (Stage A)** solves this challenge by learning an exact, pre-calibrated $8 \times 8$ thermal resistance matrix $R$ for each silicon floorplan. Using pure linear matrix algebra:

$$T_{\text{predicted}} = 45.0^\circ\text{C} + R \times P$$

Key achievements:
- **Accuracy**: Pooled validation RMSE of **0.003032 °C** and MAE of **0.002598 °C** across 460 unseen test configurations (3,680 core comparisons), matching HotSpot's 2-decimal output rounding quantization floor.
- **Inference Speed**: **2.01 µs** median inference time via NumPy, achieving a **906,277× speedup** over HotSpot.
- **Compactness**: Just **64 parameters (`512 B` raw `float64` / `771 B` CSV)** per floorplan, and **1,472 parameters (`11,776 B` = `11.50 KiB` raw `float64` / `17,733 B` = `17.32 KiB` CSV)** across all 23 floorplans.

---

## Stage A Scope & Exclusions

### In Scope for Stage A
- **23 Heterogeneous 2D Floorplans**: All 23 floorplans across `Style_1` (3), `Style_2` (5), `Style_3` (5), and `Style_4` (10).
- **Heterogeneous Core Topologies**: 2 Big cores ($4\times4\text{ mm}$, $10\text{–}20\text{ W}$), 6 Little cores ($2\times2\text{ mm}$, $5\text{–}10\text{ W}$), and varying counts of passive silicon `FILL` blocks ($0\text{ W}$).
- **Steady-State Conduction**: Predicting equilibrium operating temperatures under constant active power dissipation and constant convective cooling boundary conditions ($T_{\text{ambient}} = 45.0^\circ\text{C} = 318.15\text{ K}$).
- **Non-Sequential Core Topologies**: Exact column-to-core mapping for arbitrary layouts, specifically handling `Style_3` (`Core_0`, `Core_2`, `Core_1`, `Core_3`, `Core_4`, `Core_5`, `Core_6`, `Core_7`).
- **Full Tooling Suite**: Batch calibration orchestrator, interactive CLI predictor, interactive Flask web interface with SVG heatmap, and reproducible evaluation benchmark suite.

### Explicitly Excluded from Stage A
- **Transient / Time-Dependent Simulation**: Temperature variation over time steps ($\frac{dT}{dt}$ using thermal capacitance $C$) is reserved for future stages.
- **Dynamic Thermal Management (DTM)**: Active throttling policies, DVFS controllers, or governor logic.
- **Machine Learning / Neural Networks**: Non-linear models, deep networks, or surrogate neural approximations (linear matrix theory perfectly captures steady-state heat conduction).
- **Runtime RAM Profiling**: The evaluation metric *Model Size* measures the storage size of the calibrated model parameters (not runtime process RAM consumption).

---

## System Architecture & Mathematics

### The Physical Principle
Steady-state heat conduction in a solid chip with constant thermal conductivity $k$ and heat transfer coefficient $h$ is governed by the steady-state heat diffusion equation:

$$\nabla \cdot (k \nabla T) = -q$$

Discretized via finite-difference grid elements (as implemented in HotSpot), this produces a linear algebraic system:

$$G \cdot \Delta T = P$$

where $G$ is the symmetric, positive-definite thermal conductance matrix, $\Delta T = T - T_{\text{ambient}}$ is the temperature rise vector, and $P$ is the core power dissipation vector.

Because $G$ is strictly invertible, the inverse mapping is exact:

$$\Delta T = R \times P \quad \iff \quad T_i = T_{\text{ambient}} + \sum_{j=0}^{N-1} R_{ij} P_j$$

where $R = G^{-1} \in \mathbb{R}^{N \times N}$ is the **thermal resistance matrix** (units: $\text{K/W}$ or $^\circ\text{C/W}$).

### Physical Interpretation of Matrix Entries
- **Diagonal Elements ($R_{ii}$)**: **Self-heating resistance** ($\sim 1.5\text{–}2.6\text{ K/W}$). The temperature rise on core $i$ caused by power dissipated directly in core $i$. Little cores ($2\times2\text{ mm}$) have higher self-heating resistance ($\sim 2.3\text{–}2.6\text{ K/W}$) than Big cores ($4\times4\text{ mm}$, $\sim 1.5\text{–}1.7\text{ K/W}$) due to higher power density per unit area.
- **Off-Diagonal Elements ($R_{ij}, i \neq j$)**: **Cross-coupling resistance** ($\sim 1.0\text{–}1.3\text{ K/W}$). The temperature rise induced on core $i$ by $1\text{ W}$ dissipated on adjacent or distant core $j$.

---

## Project Structure

```text
temperature_project/
├── app.py                             # Interactive Flask web application backend
├── predict.py                         # Standalone CLI temperature predictor
├── stage_a_pipeline.py                # Reusable calibration & validation engine
├── run_all_stage_a.py                 # Batch orchestrator for all 23 floorplans
├── benchmark_stage_a.py               # Reproducible benchmark suite (accuracy, time, size)
├── stage_a_all_floorplans_summary.csv # Calibration & validation metrics summary
├── stage_a_benchmark_per_floorplan.csv# Per-floorplan timing, error, and size benchmark
├── stage_a_benchmark_overall_summary.csv# Overall 23-floorplan pooled benchmark summary
├── STAGE_A_EVALUATION_REPORT.md       # Comprehensive evaluation and analysis report
├── STAGE_A_TECHNICAL_EXPLANATION.md   # Step-by-step presentation guide with numerical example
├── README.md                          # This documentation file
│
├── templates/
│   └── index.html                     # Responsive dark-theme dashboard UI
│
├── static/
│   ├── css/
│   │   └── style.css                  # Custom styling (KPIs, bar charts, SVG canvas)
│   └── js/
│       └── app.js                     # Frontend logic (SVG heatmap, chart, live timers)
│
└── dataset/
    └── Sample_dataset/
        └── 2D_floorplans/
            ├── Style_1/ (fp_001 .. fp_003)
            ├── Style_2/ (fp_001 .. fp_005)
            ├── Style_3/ (fp_001 .. fp_005)
            └── Style_4/ (fp_001 .. fp_010)
                └── <fp_id>/
                    ├── <fp_id>.lcf                      # Layer configuration file
                    ├── <fp_id>_layer_0.flp              # Silicon floorplan geometry
                    ├── <fp_id>_steady.ptrace            # Initial baseline power trace
                    ├── generated_samples/
                    │   ├── thermal_matrix_<fp_id>.csv   # Calibrated 8x8 resistance matrix R
                    │   ├── combined_100_samples.csv     # 100 training power-temperature pairs
                    │   ├── ptrace/                      # 100 training .ptrace files
                    │   └── steady/                      # 100 HotSpot ground-truth .steady files
                    └── validation_samples/
                        ├── validation_comparisons_160.csv # 20 unseen validation comparisons
                        ├── ptrace/                      # 20 held-out .ptrace files
                        └── steady/                      # 20 HotSpot validation .steady files
```

---

## Prerequisites & Environment Setup

### Required Software
- **Operating System**: Linux (Ubuntu 20.04/22.04 LTS native or WSL2 on Windows 10/11).
- **Python**: Version 3.8, 3.9, 3.10, 3.11, or 3.12.
- **Python Libraries**: `numpy`, `flask`.
- **HotSpot**: Version 6.0 compiled and installed at `~/HotSpot/hotspot`.

### Installation Steps

1. **Clone or navigate to the repository in WSL Ubuntu**:
   ```bash
   cd /home/pardhu/temperature_project
   ```

2. **Activate your Python environment**:
   ```bash
   source .venv/bin/activate
   pip install numpy flask
   ```

3. **Verify HotSpot installation**:
   ```bash
   ~/HotSpot/hotspot -version
   ```

---

## How to Run

### 1. CLI Temperature Predictor

The command-line predictor [`predict.py`](predict.py) predicts core temperatures instantly using pre-calibrated matrices without invoking HotSpot.

- **Interactive Selection**:
  ```bash
  python3 predict.py
  ```
- **List All 23 Supported Floorplans**:
  ```bash
  python3 predict.py --list
  ```
- **Predict via Inline Power Vector**:
  Supply 8 powers in exact matrix column order:
  ```bash
  python3 predict.py -f Style_1/fp_001 -p "11.19, 15.03, 7.56, 9.30, 5.51, 6.12, 8.01, 7.78"
  ```
- **Predict via Power Trace File**:
  ```bash
  python3 predict.py -f Style_1/fp_001 --ptrace dataset/Sample_dataset/2D_floorplans/Style_1/fp_001/fp_001_steady.ptrace
  ```

### 2. Interactive Web Application

The web interface [`app.py`](app.py) provides an interactive graphical experience:
- Floorplan selection across all 23 layouts via style/floorplan dropdowns.
- Proportional SVG die layout showing Big ($4\times4\text{ mm}$), Little ($2\times2\text{ mm}$), and FILL blocks.
- Real-time thermal heatmap coloring and "Core vs. Temperature" SVG chart.
- Single-core vs. All-core prediction with individual coupling breakdown.
- Compact Execution Time and Storage Comparison bar charts.
- High-resolution Live Prediction Time indicator.

**Start the web server**:
```bash
python3 app.py
```
Open your browser and navigate to: **`http://localhost:5000`**

### 3. Reproducible Evaluation Benchmark

The benchmark script [`benchmark_stage_a.py`](benchmark_stage_a.py) executes a comprehensive, reproducible evaluation across all 23 floorplans:
- Evaluates RMSE, MAE, and max error on 20 held-out samples per floorplan.
- Benchmarks pure NumPy inference (5,000 runs per floorplan, 115,000 total).
- Benchmarks equivalent HotSpot steady-state grid simulations.
- Benchmarks end-to-end Flask Web API request latency.
- Outputs `stage_a_benchmark_per_floorplan.csv` and `stage_a_benchmark_overall_summary.csv`.

**Run the benchmark**:
```bash
python3 benchmark_stage_a.py
```

### 4. Batch Calibration Pipeline

To recalibrate all 23 floorplans from scratch or verify completed floorplans:
```bash
python3 run_all_stage_a.py
```
To force recalibration and overwrite existing matrices:
```bash
python3 run_all_stage_a.py --force-overwrite
```

### 5. Automated Verification Suite

To run the automated 18-point verification test suite (testing floorplan geometry, power validation, CLI regression, dropdowns, and live timing):
```bash
python3 test_website.py
```

---

## Benchmark Evaluation Results

All results reported below were measured across all **23 calibrated floorplans** (460 held-out validation configurations, 3,680 individual core temperature evaluations).

### 1. Accuracy vs. HotSpot

| Evaluation Split | Floorplans | Samples / FP | Core Comparisons | Pooled RMSE (°C) | Pooled MAE (°C) | Max Error (°C) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Training Set** | 23 | 100 | 18,400 | `0.002767` | `0.002358` | `0.006720` |
| **Held-Out Validation** | 23 | 20 | 3,680 | **`0.003032`** | **`0.002598`** | **`0.007784`** |

> **Why is residual error ~0.003 °C?**  
> HotSpot outputs steady-state temperatures to 2 decimal places (`0.01 K` resolution). Uniform rounding quantization on $[-0.005, +0.005]\text{ K}$ has a theoretical standard deviation of $\sigma = \frac{0.01}{\sqrt{12}} \approx 0.002887\text{ K}$. Our empirical RMSE of `0.003032 K` matches this quantization floor, demonstrating that the calibrated $R$ matrix models the exact underlying linear conduction physics without model error.

### 2. Inference Latency & Speedup

| Execution Mode | What Is Measured | Median Latency | Mean Latency | Speedup vs. HotSpot |
| :--- | :--- | :---: | :---: | :---: |
| **Pure NumPy Predictor** | In-memory `R @ P` linear matrix computation | **`2.0096 µs`** | `2.3545 µs` | **`906,277.2×`** |
| **Pure Python Stdlib** | Pure list dot-product (zero dependencies) | `17.4910 µs` | `18.1200 µs` | `104,125.2×` |
| **Flask Web API** | Full HTTP request (JSON parsing, validation, R×P, serialization) | `1.6608 ms` | `1.7240 ms` | `1,096.6×` |
| **CLI Subprocess** | Process invocation (`python3 predict.py`), file I/O, formatted output | `225.47 ms` | `231.10 ms` | `8.08×` |
| **HotSpot Grid Solver** | Fresh steady-state grid finite-difference solve | `1821.255 ms` | `2400.135 ms` | `1.0×` (Baseline) |

### 3. Model Storage Footprint

Model size measures the **storage size of the calibrated model parameters** required for prediction ($T = T_{\text{ambient}} + R \times P$), not runtime RAM usage. Each floorplan uses an $8 \times 8$ thermal resistance matrix $R$ ($64$ `float64` coefficients), totaling $23 \times 64 = 1,472$ calibrated parameters across all 23 floorplans:

| Storage Format | Parameters (Per FP / All 23 FPs) | Per Floorplan | Total (All 23 Floorplans) | Description |
| :--- | :---: | :---: | :---: | :--- |
| **Raw `float64` Parameters ($8 \times 8$)** | **64** / **1,472** | **512 B** | **11,776 B (11.50 KiB)** | $8 \times 8 \times 8\text{ bytes/float64}$ binary parameter storage |
| **Serialized CSV File (`thermal_matrix_fpXXX.csv`)** | **64** / **1,472** | **771 B** | **17,733 B (17.32 KiB)** | On-disk ASCII CSV table with core column/row headers and 6-decimal `K/W` values |
| **Serialized NumPy `.npy` Binary** | **64** / **1,472** | `640 B` | `14,720 B (14.38 KiB)` | `512 B` raw `float64` parameter payload + `128 B` standard `.npy` array header |

---

## Live Timing vs. Benchmark Timing

The website and CLI display two distinct timing metrics:

1. **Precomputed Benchmark Timing (Median)**:
   - Displayed in the "Execution Time Comparison" horizontal bar chart.
   - Measured offline across **5,000 repeated in-memory executions** per floorplan using high-resolution timers (`time.perf_counter_ns()`) after 200 warmup runs.
   - Represents the statistically stable hardware baseline ($\sim 2.01\text{ µs}$).

2. **Live Single-Call Prediction Time**:
   - Displayed near the prediction summary as **Live Prediction Time**.
   - Measured dynamically on the backend server for that specific execution using `time.perf_counter_ns()`.
   - Measures strictly the in-memory array operation `delta_t = R @ P`, excluding HTTP transport, JSON parsing, and DOM rendering.
   - Automatically cleared to `-- µs` when power inputs or floorplans change, preventing display of stale results.

---

## Limitations & Future Extensions

1. **Steady-State Conduction Only (Stage A)**:  
   The current model assumes heat flow has reached equilibrium ($\frac{\partial T}{\partial t} = 0$). Dynamic core activity with rapid frequency spikes requires transient capacitance modeling ($\text{Stage B}$).
2. **Fixed Ambient & Boundary Conditions**:  
   Calibrated for convective sink cooling with $T_{\text{ambient}} = 45.0^\circ\text{C}$. Variable ambient temperatures can be accommodated via $T = T_{\text{amb}} + R \cdot P$, but nonlinear fans or dynamic heatsink airflow are not modeled.
3. **Linearity Assumption**:  
   Thermal conductivity of silicon varies slightly with temperature ($k(T) \propto T^{-4/3}$). Over typical operational ranges ($45\text{–}145^\circ\text{C}$), this variation produces less than $0.01^\circ\text{C}$ divergence from HotSpot's constant-conductivity solver.

---

## License & Academic Integrity

This project was developed for the **Lightweight Temperature Predictor** coursework assignment. All calibration data, validation splits, and benchmarks are reproducible from the provided source code.
