# Stage B geometry-aware ML training infrastructure

**Implementation only; no real-data training or ML performance claims yet.**
The development branch starts from `stage-b-dataset-preparation` commit
`ed91bf48c2e9849c97f8626c8cd497fa4d021c77`. Existing Stage A/Stage B code,
inputs, matrices and metrics are preserved. Render and deployment branches are
not changed. No HotSpot execution, frontend integration or full training sweep
is part of this work.

## Data availability and the schema boundary

The user reports a WSL CSV with 1,540 rows and 58 columns, generated from 77
new floorplans (`b_fp_006`–`b_fp_082`), 20 real HotSpot configurations per
floorplan, and 12,320 core labels. That combined CSV and its actual header are
**not present in the inspected GitHub branches**. The repository contains the
original geometry/power inputs, not those new temperature outputs.

This package never invents missing labels or reconstructs them from Stage A.
You must provide the existing local `stage_b_training_dataset.csv`. Since the
header and units are unavailable, **there is no guessed automatic CSV schema**.
Inspect the actual file, generate an example JSON mapping, then have Antigravity
map the actual column names and declare the actual units. Keep the original CSV
unchanged. Output and any explicit adaptation copies belong under `stage_b/ml_runs/`.

`schema-template` emits an **example**, not a claim about your actual header.
Its 58-column example consists of two identifiers and eight slots with a core ID,
width, height, x, y, power and temperature. Other 58-column layouts are supported
by mapping their columns explicitly. Unused columns are not features, but must
still contain values. If each core has fixed named columns instead of an ID
column, replace a slot's `id_column` with `core_id: "Core_0_L0"` (one or the other,
never both). Dynamic per-row IDs are canonicalized to `Core_0_L0` … `Core_7_L0`
before features or labels are constructed. Big/Little is determined by dimensions,
not IDs. This preserves correct targets even when row slots are permuted.

Required schema declarations:

- `version: 1`; `floorplan_column`; `sample_column`; `expected_columns: 58`.
- `geometry_unit`: `m`, `mm`, or `um`; `temperature_unit`: `C` or `K`;
  `power_unit: "W"`; `label_source: "HotSpot"`; `ambient_c: 45.0` for this dataset.
- Eight core mappings, each with `core_id` or `id_column`, plus column names for
  `width`, `height`, `x`, `y`, `power`, and `temperature`.
- `max_temperature_c` is a configurable validation ceiling (default 500°C), not
  a new thermal assumption or a measured limit. The lower check permits 0.05°C
  below the declared ambient to accommodate solver rounding.

The loader requires all 77 new IDs and all 20 sample IDs per layout, 1,540 rows,
unique headers and floorplan/sample pairs, finite complete values, 2 Big 4×4 mm
and 6 Little 2×2 mm cores, in-bounds nonoverlapping geometry, fixed geometry per
floorplan, and canonical target ordering. It reads and hashes committed silicon
FLPs/LCFs; validates FILL coverage and original silicon/TIM materials; compares
CSV geometry to the named core rectangles; and compares every power vector to
its original hashed workload ptrace. Big powers must be 10–20 W and Little
powers 5–10 W, with unique configurations per layout. Original FILL powers must
be zero. Sample identifiers must be `sample_001` … `sample_020`. If the combined
CSV used different identifiers, make the relationship explicit in an audited
adapter/copy; never silently relabel or overwrite the original data.

Declaring `label_source: "HotSpot"` does not independently prove provenance.
The loader validates numbers, units, identity and input alignment, but cannot
reproduce absent solver outputs. Retain and review your original successful
HotSpot run records before real training. In particular, a Kelvin value wrongly
declared Celsius can be numerically plausible; units must come from provenance.

## Geometry-aware features and leakage prevention

Internal geometry units are mm, powers W, targets °C. There are **252 fixed
features**:

- Per core: power, width, height, x/y, area, power density, center x/y, Big flag
  (80 features).
- Die width, height, area, passive-area fraction (4).
- For 28 unordered pairs: signed center dx/dy, center distance, edge gap (112).
- For 56 directed core pairs: source power / (center distance + 1 mm) (56).

The 1 mm softening is a documented feature convention, not a learned or assumed
thermal resistance. Floorplan/sample IDs, temperature labels, Stage A predictions
and calibrated matrices are never features. Geometry comes from each layout,
not from a fitted summary of all layouts.

`prepare` shuffles sorted **floorplan IDs**, with seed 42, into **53 training,
12 validation, 12 test** layouts. Every layout's 20 configurations stay together:
1,060 / 240 / 240 rows. It does not randomly split rows. The same prepared
partition is reused for every algorithm and experiment. Canonical cores use
one stable eight-target ordering across partitions.

StandardScaler means/scales for both X and Y are fitted **exclusively on the
training rows**; Celsius is restored for every reported metric/prediction.
Prepared artifacts include three separate NPZ files, complete split IDs/counts,
feature names/units/version/hash, schema/observed header, scaler parameters,
source CSV SHA-256, and geometry-manifest SHA-256. Test arrays are not opened by
`train`. Dataset validation necessarily reads all labels to reject invalid data;
it does not use holdout labels for fitting or selection.

## Models and experiments

| Model | Implementation / regularization |
|---|---|
| Ordinary linear regression | scikit-learn least squares, eight outputs |
| Ridge | L2, configurable alpha |
| Lasso | independent-output L1, configurable alpha |
| Elastic Net | L1/L2 with configurable alpha and l1_ratio |
| Random Forest | native multi-output; depth, leaf size and tree count controls |
| XGBoost | eight CPU regressors via MultiOutputRegressor; depth, trees, learning rate, penalties configurable |
| Small MLP | NumPy, one default 32-unit tanh layer, eight linear outputs; configurable hidden layers and L1/L2 |

The NumPy differentiable estimator also supports a linear model without hidden
layers. **Batch GD** uses the whole training set per step; **SGD** uses one row
per step with seeded shuffling; **Adam** uses minibatches and bias-corrected first/
second moments. All optimize mean squared error on scaled targets. L1 uses the
standard sign subgradient; L2 is smooth. Biases are not penalized. Defaults
include gradient clipping, training-loss tolerance/patience and best-training-loss
weight restoration; no test-dependent or validation-dependent fitting occurs.
The least-squares model is the reference baseline for linear optimizer experiments.

Named groups are deliberately small:

- `core`: 7 configurations, one per requested model family.
- `optimizers`: 6 configurations (linear/MLP × batch GD/SGD/Adam).
- `regularization`: 10 configurations covering unregularized OLS, weak/default
  Ridge and Lasso, Elastic Net, shallow/default RF, and MLP with/without L2.

These are meaningful sets, not an unnecessary Cartesian product. SGD/Adam are
never assigned to trees. `--models` selects exact names from a group. A custom
`--experiments JSON_FILE` accepts a list of `{name, kind, parameters}` objects
for individual experiments or a compact custom set. Relevant parameters are
passed to their estimator; wrong names are errors, not silently ignored.

Examples of a custom differentiable experiment:

```json
[
  {"name": "small_mlp_adam", "kind": "mlp", "parameters": {
    "hidden_layers": [16], "optimizer": "adam", "learning_rate": 0.002,
    "epochs": 200, "batch_size": 32, "tolerance": 0.0000001,
    "patience": 20, "l1": 0.0, "l2": 0.001
  }}
]
```

`--epochs`, `--learning-rate`, `--batch-size`, `--tolerance` and `--alpha` also
provide appropriate CLI overrides. Unsupported optimizer/model combinations
are rejected. Optional XGBoost is imported only when requested; a missing
installation fails preflight before creating a training run, rather than silently
omitting an algorithm. The CPU-only package avoids CUDA/NCCL dependencies on WSL.
PyTorch/TensorFlow are unnecessary for this small MLP.

## Selection, reporting and the final test gate

Each experiment records aggregate and per-core MAE/RMSE/max absolute error in
°C, actual fit time, single-sample median/p95 latency, batch latency, actual
compressed joblib size, parameters, seed, source/split/feature hashes, library
versions and platform metadata. Latency includes X scaling, prediction and inverse
Y scaling; it excludes CSV loading and feature construction. Warmup is separate.
Training times and latency naturally vary between machines/runs.

Selection minimizes **validation RMSE**, with experiment-name tie-breaking.
`selection.json` pins the chosen model and prepared dataset hashes. Training
reports contain no test metrics. `evaluate-test` is a separate, explicit final
command: it loads only the selected model, verifies hashes and writes one new
`final_test_report.json`, refusing overwrite. Freeze configurations and select
across your validation experiments before invoking it. Running separate final
evaluations across many groups and tuning from them would still leak the holdout;
software cannot prevent a person from creating a new experiment directory.

All model families use the same scaled training/validation data. A saved
`ModelBundle.predict(raw_engineered_X)` restores temperatures in °C, with core
order and feature hash attached. Serialized model size includes the estimator
and required preprocessing metadata; it is not raw parameter storage or process
RAM. Joblib files execute Python during loading: load only your own trusted
training artifacts. No downloads or external model files are loaded automatically.

## Future original-23 external benchmark

Original Stage A layouts and pilots `b_fp_001`–`b_fp_005` are excluded from this
training dataset. The selected 77 plus 23 reserved originals give the intended
100-layout scope; earlier pilot artifacts remain preserved in the repository.

`external.py` provides a future evaluation API, and `external-check` only checks
contracts. **No original-23 evaluation has been run.** Before any future external
comparison, obtain independent HotSpot labels and an authoritative external
geometry manifest. Core IDs/counts, Big/Little sizes, die coverage, powers and
feature encoding must pass the same validation. Contracts must also match the
actual HotSpot config SHA-256, explicit material layers, ambient, normalized
mm/W/°C units, canonical core order, label source and feature-definition hash.
Stage A predictions/matrices are never accepted as ground-truth sources.

A schema may include `simulation_contract` with the **actual**
`hotspot_config_sha256` and `material_layers` (specific heat, resistivity,
thickness and flow/power flags for each layer). Preparation adds normalized
units, ambient, core order, label source and feature hash and saves the complete
`simulation_contract.json`. If this provenance is absent, training is possible
with the absence recorded, but the external-evaluation API fails closed.
Do not invent a hash or label provenance just to pass the gate.

## Antigravity: synchronize and run the first real-data smoke test later

The CSV is local and untracked; keep it backed up outside Git. From the user's
WSL checkout (adjust only if that directory is not the actual checkout):

```bash
cd /home/pardhu/temperature_project
git fetch origin
git switch stage-b-ml-framework
cd "$(git rev-parse --show-toplevel)/temperature_project"
python3 --version                      # Python 3.11+ recommended, tested on 3.12
python3 -m venv stage_b/ml_runs/venv
source stage_b/ml_runs/venv/bin/activate
python -m pip install -r stage_b/ml/requirements.txt

python -m stage_b.ml inspect-csv --csv /home/pardhu/temperature_project/stage_b/stage_b_training_dataset.csv
python -m stage_b.ml schema-template --out stage_b/ml_runs/schema.json
```

**Stop and map `stage_b/ml_runs/schema.json` to the printed real header, physical
units and provenance.** Antigravity should inspect the original CSV and HotSpot
records; the package cannot infer the absent header. Preserve the source CSV.
Then validate, prepare and train **Ridge only** as the first smoke test:

```bash
python -m stage_b.ml validate \
  --csv /home/pardhu/temperature_project/stage_b/stage_b_training_dataset.csv \
  --schema stage_b/ml_runs/schema.json
python -m stage_b.ml prepare \
  --csv /home/pardhu/temperature_project/stage_b/stage_b_training_dataset.csv \
  --schema stage_b/ml_runs/schema.json \
  --out stage_b/ml_runs/prepared-v1 --seed 42 --split-counts 53 12 12
python -m stage_b.ml train --prepared stage_b/ml_runs/prepared-v1 \
  --out stage_b/ml_runs/ridge-smoke --models ridge --alpha 1 --seed 42
```

Inspect `ridge-smoke/ridge/validation_report.json` and `ridge-smoke/selection.json`.
Do not evaluate the test partition yet. Paths above use the user's stated local
CSV location; that file and its actual schema cannot be verified from this cloud
machine. Existing directories are never overwritten; choose a new run name when
changing configuration. Outputs inside the project are restricted to the new
Git-ignored `stage_b/ml_runs/` tree. No original files are modified by these commands.

For later, separately authorized comparisons (not run during implementation):

```bash
python -m pip install -r stage_b/ml/requirements-xgboost.txt
python -m stage_b.ml train --prepared stage_b/ml_runs/prepared-v1 --out stage_b/ml_runs/core-v1 --group core
python -m stage_b.ml train --prepared stage_b/ml_runs/prepared-v1 --out stage_b/ml_runs/optimizers-v1 --group optimizers
python -m stage_b.ml train --prepared stage_b/ml_runs/prepared-v1 --out stage_b/ml_runs/regularization-v1 --group regularization
# Only after final selection and configuration freeze:
python -m stage_b.ml evaluate-test --prepared stage_b/ml_runs/prepared-v1 --run stage_b/ml_runs/core-v1
# Future contract check only, no original-floorplan prediction/evaluation:
python -m stage_b.ml external-check --training-contract stage_b/ml_runs/prepared-v1/simulation_contract.json --external-contract /path/to/verified-external-contract.json
```

## Software verification

From `temperature_project/`, after installing the appropriate test dependencies:

```bash
PYTHONDONTWRITEBYTECODE=1 python -m unittest discover -s stage_b/ml/tests -v
PYTHONDONTWRITEBYTECODE=1 python -m unittest discover -s stage_b -v
PYTHONDONTWRITEBYTECODE=1 python -m unittest discover -s tests -v
```

The first suite constructs explicitly synthetic temperatures in temporary CSVs,
using original geometry/power inputs. Those labels are software fixtures, not
HotSpot results, and are deleted afterward. Tiny estimator fits and a Ridge
CLI integration test validate plumbing only. No synthetic dataset, fitted model
or fictitious ML benchmark is committed. The existing real-HotSpot test remains
opt-in and is not run. See `VALIDATION.md` for the actual verification results.
