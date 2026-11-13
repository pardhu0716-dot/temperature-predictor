# Thermal Management — Temperature Predictor

Production deployment preparation and local Gunicorn commands: [Render deployment guide](DEPLOYMENT.md). The live demo link will be added after a public deployment is approved and verified.

An academic project implementing **Stage A: a steady-state linear thermal predictor** for heterogeneous multicore floorplans. HotSpot supplies reference temperatures during calibration and evaluation; prediction uses saved thermal coupling matrices.

## Model and calibration

For a fixed floorplan, ambient temperature, material properties, and thermal configuration:

**ΔT = R × P**, and **T = T_ambient + R × P**.

`P` contains active-core powers in watts; `R[i,j]` describes the temperature rise at core `i` caused by power at core `j`, in K/W. The implementation uses an ambient temperature of **318.15 K (45 °C)**. Core order must match the matrix CSV headers. Passive FILL blocks have zero power.

`stage_a_pipeline.py` generates power samples, runs HotSpot in grid steady-state mode, forms temperature-rise data, and estimates the coupling matrix using NumPy least squares. Defaults are 100 training samples and 20 independent validation samples per floorplan. Calibration is specific to each floorplan and simulation configuration; changing these conditions requires recalibration.

## Existing project structure

The uploaded project's enclosing `temperature_project/` directory is preserved. Its original README and source files are unchanged. This repository-level README supplies concise setup guidance.

```text
temperature_project/
  README.md                          Original project documentation
  STAGE_A_TECHNICAL_EXPLANATION.md
  STAGE_A_EVALUATION_REPORT.md
  app.py                             Flask dashboard and prediction API
  predict.py                         Interactive/noninteractive CLI predictor
  stage_a_pipeline.py                Calibration and validation pipeline
  run_all_stage_a.py                 Batch processing and summary generation
  benchmark_stage_a.py               Accuracy, latency, and model-size evaluation
  requirements.txt                   Flask and NumPy dependencies
  hotspot.config                     Reference simulator configuration
  stage_a_all_floorplans_summary.csv
  stage_a_benchmark_per_floorplan.csv
  stage_a_benchmark_overall_summary.csv
  templates/index.html
  static/css/style.css
  static/js/app.js
  dataset/
    Sample_dataset/
      README.md
      2D_floorplans/Style_1...Style_4/
        fp_*/
          *.lcf, *.flp, *.png         Floorplan/layer definitions and images
          *_steady.ptrace            Original sample input
          *_transient.ptrace         Supplied input; not a transient predictor
          *.steady                   Existing reference output, where supplied
          stage_a_summary_*.csv
          generated_samples/
            combined_100_samples.csv
            thermal_matrix_*.csv
            *.py                     Original helper scripts, where present
          validation_samples/
            ptrace/, steady/         Held-out inputs and reference temperatures
            *.csv, *.txt, *.py       Existing comparisons/reports/helpers
      3D_floorplans/                  Original small sample inputs; not calibrated by Stage A
```

Directory contents vary by floorplan; the runtime matrix is in each floorplan's `generated_samples/` directory. The two redundant matrix copies omitted from upload are listed in the repository review.

## Installation

Use Python 3.10 or newer as a practical environment with the declared NumPy dependency. From the repository root, on Linux or Ubuntu/WSL:

```bash
cd temperature_project
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install -r requirements.txt
```

The uploaded virtual environment is intentionally excluded; recreate it locally. Dependencies are `flask>=3.0.0` and `numpy>=1.26.0` as declared by the existing project.

## Run predictions

Run these commands from `temperature_project/`:

```bash
python3 predict.py --list
python3 predict.py
python3 predict.py --floorplan Style_1/fp_001 --powers "16.39,10.25,6.38,6.12,8.68,8.38,9.46,5.43"
python3 predict.py --floorplan Style_1/fp_001 --ptrace dataset/Sample_dataset/2D_floorplans/Style_1/fp_001/validation_samples/ptrace/val_sample_001.ptrace
```

The CLI validates powers and uses saved matrices. The current dataset convention is 10–20 W for Big cores and 5–10 W for Little cores; follow the selected floorplan's core ordering and validation rules.

For the existing web dashboard:

```bash
python3 app.py
```

Open `http://localhost:5000`. The `PORT` environment variable overrides the default port. The development server binds to `0.0.0.0`; use it on a trusted local machine. Normal dashboard and CLI prediction do not invoke HotSpot or recalibrate matrices.

## Calibration and evaluation

Install and build HotSpot separately in Linux/Ubuntu/WSL. The default executable path is `~/HotSpot/hotspot`. No HotSpot installation or executable is included. Use the supplied configuration and floorplan layer files consistently.

Example calibration invocation (existing outputs are protected unless explicitly overwritten):

```bash
python3 stage_a_pipeline.py dataset/Sample_dataset/2D_floorplans/Style_1/fp_001 --hotspot-bin "$HOME/HotSpot/hotspot" --config-file hotspot.config
```

Use a separate working copy for recalibration, or review `python3 stage_a_pipeline.py --help` before using `--force-overwrite`. Regenerating calibration restores the excluded raw calibration traces and solver files. Geometry/power mismatches can be handled by the existing `--allow-geometry-power-override` option after reviewing the input.

```bash
python3 run_all_stage_a.py
python3 benchmark_stage_a.py --hotspot-bin "$HOME/HotSpot/hotspot"
```

The batch script uses the default HotSpot path and processes the 2D catalog. The benchmark uses retained combined training CSVs, held-out trace/steady files, matrices, and fresh HotSpot runs. It writes evaluation results and the evaluation report. Execution times depend on the machine and measurement boundary; saved benchmark values are historical results, not universal performance guarantees.

## Current status and future work

The upload contains calibration/validation outputs and 8×8 matrices for **23 2D floorplans**, a CLI, Flask dashboard, and benchmark/reporting code. Saved reports cover accuracy, prediction time, and model size. This repository preparation verified source preservation and prediction loading; it did not rerun HotSpot or independently reproduce the historical simulator benchmark.

Potential future work includes broader floorplan/configuration validation, transient prediction, proactive dynamic thermal management, and real workload power integration. These are future directions, not implemented capabilities. Included 3D floorplans and transient traces are dataset assets, not evidence of a 3D or transient predictor.

## Upload scope

The original Python environment, bytecode caches, regenerable raw calibration `ptrace/` and `steady/` directories, and two byte-identical redundant matrix copies are excluded. Combined training data, all runtime matrices, held-out validation inputs/reference outputs, summaries, original helper scripts, documentation, and small original dataset assets are retained. Existing source files have not been modified.
