# Stage B input dataset and later HotSpot execution

Prepared from `stage-b-pilot` commit `c8cc4e27a332af168b9e539aae80da615735eeca`
on the separate `stage-b-dataset-preparation` branch. `main` and `stage-b-pilot`
are preserved. No Stage A code, floorplans, traces, matrices, CSVs, or metrics
were modified; the five pilot layouts and their metadata are unchanged.

## Counts and scope

| Item | Count |
|---|---:|
| Original Stage A floorplans, preserved | 23 |
| Existing Stage B pilots, preserved | 5 |
| New layouts, b_fp_006 through b_fp_082 | 77 |
| Total distinct floorplans across these groups | 105 |
| Configurations per NEW layout | 20 |
| Reference configurations, 20 W Big / 8 W Little | 77 |
| Independent random configurations | 1,463 |
| Expected NEW HotSpot steady-state simulations | 1,540 |
| Real simulations executed during preparation | 0 |
| Temperature labels generated during preparation | 0 |

The request explicitly specified approximately 77 additional floorplans.
23 + 5 + 77 = 105, rather than exactly 100; the explicit additional count was
used. No new power traces are generated for Stage A or the existing pilots.
The new simulation manifest selects only the 77 new layouts.

## Files

- `prepare_stage_b_dataset.py`: deterministic, atomic preparation and read-only
  dataset validation. Reuses the original floorplan parser, geometry validator,
  candidate generator, FILL tessellation, and diversity descriptor.
- `generate_power_traces.py`: reused geometry-aware classification/reference/
  random trace generation. Hardened to reject NaN/Inf, enforce exactly zero FILL
  power, reject impossible uniqueness requests, and sample representable power
  values within configured bounds.
- `run_stage_b_simulations.py`: automation of the real HotSpot executable only.
- `test_dataset_preparation.py`: geometry, power, reproduction, integrity,
  command, parsing, conversion, dry-run, failure, locking, and resume tests.
- `test_generate_power_traces.py`: retains existing tests; real HotSpot
  integration now requires explicit `STAGE_B_RUN_HOTSPOT_TESTS=1` opt-in.
- `dataset_v1/floorplan_manifest.json`: all 77 layouts, parameters, seeds, typed
  core names, die dimensions, source provenance, input paths and SHA-256 hashes.
- `dataset_v1/workload_manifest.json`: all 1,540 input traces, sample IDs,
  reference flags, ranges, and file hashes.
- `dataset_v1/floorplans/b_fp_*/`: silicon `.flp`, full-die TIM `.flp`, `.lcf`,
  and `metadata.json` for each layout (308 files).
- `dataset_v1/workloads/b_fp_*/sample_001.ptrace` through `sample_020.ptrace`:
  1,540 independent single-row steady-state inputs (reference first).
- `DATASET_VALIDATION.md`: validation evidence and practical limits.

The generated dataset contains 1,850 files, including its two manifests.
No temperature output, label CSV, model, cache, test dependency, or temporary
reproduction directory is committed. Runtime results are ignored by Git.

## Geometry and workload choices

All new layouts contain two 4×4 mm Big cores and six 2×2 mm Little cores.
Total active area stays 56 mm². Added coordinate-rank spacing (0–800 µm),
die margin (0–1,600 µm), and accepted position jitter (up to 1,000 µm per move)
produce variants of the source templates. Core dimensions never scale.
The source-family counts are Style_1: 20, Style_2: 19, Style_3: 19, Style_4: 19.
Style records template provenance, not a guarantee about unknown original
style-generation rules. The layout seed is `20261108 + zero_based_layout_index`.

All eight core IDs and their FLP order are shuffled independently of their
positions and dimensions. The new dataset uses 24 different Big-core ID pairs.
Geometry is validated before writing and after reading back from disk. FILLs
cover exactly the complement of active cores with no overlaps or gaps. The
LCF material values and full-die TIM convention are inherited unchanged.

The original typed pairwise-center-distance descriptor plus die dimensions
must have RMS distance at least 150 µm from the 23 originals, five pilots,
and all previously accepted new layouts. The smallest accepted distance is
278.13 µm. Names, order, reflection, rotation, and translation cannot turn the
same arrangement into a distinct sample for this check. This is a conservative
heuristic; geometrically different layouts can share a descriptor and be rejected.
It does not establish a uniformly sampled design space.

Power types are inferred from actual dimensions, never IDs. Each trace header
matches the complete silicon FLP block order, including passive FILLs at 0 W.
Sample 001 is exactly 20 W/Big and 8 W/Little (88 W total). Samples 002–020
sample every core independently on a 0.01 W grid: Big 10–20 W, Little 5–10 W.
These reuse the project's practice power assumptions; they are not measured
workloads or a claim of validated physical power ranges. Configurable ranges
apply to random samples; the reference remains fixed even if outside them.
Duplicate eight-core power tuples are rejected within each layout.
Each layout's power RNG seed derives from SHA-256 of `seed:floorplan_id:powers`.
Both geometry and workload manifests are portable relative to their directory.
The original pilot generator's five-layout cap is unchanged.

## Validate on WSL Ubuntu without simulations

From an existing repository checkout:

```bash
git fetch origin
git switch stage-b-dataset-preparation
cd temperature_project
python3 --version                     # Python 3.10 or newer
python3 stage_b/prepare_stage_b_dataset.py --validate-only stage_b/dataset_v1/floorplan_manifest.json
PYTHONDONTWRITEBYTECODE=1 python3 -m unittest discover -s stage_b -v
```

The new Stage B scripts use only the Python standard library. For Stage A
regression tests, install the repository's existing dependencies in a venv:

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install -r requirements.txt
PYTHONDONTWRITEBYTECODE=1 python3 -m unittest discover -s tests -v
```

By default the real-HotSpot integration test is skipped. Do not run the legacy
`validate_pilot.py` as a preparation-only check: it invokes HotSpot. The
manifest validation command and default unittest command above do not simulate.

## Reproduce inputs in a new directory

From `temperature_project/`:

```bash
python3 stage_b/prepare_stage_b_dataset.py --count 77 --samples 20 --seed 20261108 \
  --big-range 10 20 --little-range 5 10 --output stage_b/dataset_reproduced
python3 stage_b/prepare_stage_b_dataset.py --validate-only stage_b/dataset_reproduced/floorplan_manifest.json
```

`dataset_v1` already exists and is protected against overwrite. Output must be
a new direct child of `stage_b`, outside `pilot` and `simulation_results`.
Generation builds in a temporary directory and publishes only after validation.
Failures clean up that temporary tree. Same parameters and sources reproduce
all 1,850 files byte-for-byte. Input paths, counts, manifests, and metadata are
included in that comparison. If parameter constraints are infeasible, preparation
fails rather than silently relaxing them.

## Later HotSpot commands — not executed during preparation

First inspect one command. `--dry-run` validates inputs and prints commands;
it never invokes HotSpot or creates simulation directories:

```bash
python3 stage_b/run_stage_b_simulations.py --hotspot-bin "$HOME/HotSpot/hotspot" \
  --floorplans b_fp_006 --samples sample_001 --dry-run
```

When you choose to start actual simulation, run one reference as a smoke test:

```bash
python3 stage_b/run_stage_b_simulations.py --hotspot-bin "$HOME/HotSpot/hotspot" \
  --config hotspot.config --floorplans b_fp_006 --samples sample_001 \
  --output-dir stage_b/simulation_results/dataset_v1
```

Inspect `stage_b/simulation_results/dataset_v1/b_fp_006/sample_001/result/`
and its sibling `attempts/` logs before starting the full batch. The command
below selects all 1,540 jobs, skipping that smoke test if its result still validates:

```bash
python3 stage_b/run_stage_b_simulations.py --hotspot-bin "$HOME/HotSpot/hotspot" \
  --config hotspot.config --output-dir stage_b/simulation_results/dataset_v1
```

After a smoke test or batch, check existing outputs without executing or writing
anything (use `--floorplans b_fp_006 --samples sample_001` to check only the smoke test):

```bash
python3 stage_b/run_stage_b_simulations.py --hotspot-bin "$HOME/HotSpot/hotspot" \
  --config hotspot.config --output-dir stage_b/simulation_results/dataset_v1 --validate-results
```

A zero exit code means every selected result passed the resume integrity checks;
missing or invalid results return nonzero. `--validate-results` and `--dry-run`
are mutually exclusive. The installed binary is required for its content hash,
but it is never executed in result-validation mode.

Repeat the simulation command to resume. To select a subset:

```bash
python3 stage_b/run_stage_b_simulations.py --floorplans b_fp_006 b_fp_007 \
  --samples sample_001 sample_002 --limit 4 --timeout 300 --dry-run
```

`--manifest`, `--hotspot-bin`, `--config`, `--output-dir`, `--floorplans`,
`--samples`, `--limit`, and per-job `--timeout` are configurable. Unknown IDs
fail rather than silently expanding selection. Default binary: `~/HotSpot/hotspot`.
Repository-local output is restricted to `stage_b/simulation_results/`;
external result directories are supported if they do not overlap protected inputs.

## Runner output and resume contract

For each floorplan/sample, every attempt gets its own directory with stdout,
stderr, command, return code (if available), elapsed time, input hashes, and
success/failure status. The LCF and its referenced layer FLPs are copied into an
isolated attempt workspace so relative paths resolve without solver writes in
source folders. HotSpot is invoked without a shell, with `-model_type grid`,
`-steady_file`, the selected ptrace, and the project's unchanged config.

After successful exit, exactly eight named core temperatures must be present in
the raw steady output. Both plain `Core_*` names and `layer_0_Core_*` names are
accepted; repeated, missing, negative-Kelvin, and nonfinite values fail. Temperature
order is irrelevant. FILL/other-layer temperatures are excluded from labels.
Celsius is Kelvin minus exactly 273.15. Only then is `result/` published with:

- `steady.temp`: original HotSpot block steady-state output.
- `labels.csv`: eight rows with `floorplan_id,sample_id,core_name,core_type,`
  `power_w,temperature_k,temperature_c`.
- `run.json`: successful metadata, hashes, command and input fingerprint.

A complete batch will produce 1,540 per-job label CSVs and 12,320 core label
rows. There is no aggregate training dataset or ML fitting step here.
Each invocation also produces a uniquely named batch summary JSON.

Resume verifies binary, config, runner source, LCF, silicon/TIM FLPs and trace
fingerprints, raw output/label hashes, and labels reconstructed from the raw
steady output. Valid results are skipped. Changed inputs or corrupted/incomplete
`result/` directories are errors and are never overwritten: investigate them and
use a new `--output-dir` for a new experiment. Failed attempts without `result/`
can be retried and remain available for inspection. A per-job exclusive lock
prevents duplicate execution; after a process crash, inspect any `.running.lock`
and confirm the old process has stopped before manually removing that lock.
Nonzero exit, timeout, or bad output records failure and creates no valid result.
Batch processing continues to other jobs and exits nonzero if any fail.

## Remaining limits and stopping point

The supplied config uses a 30 mm spreader, 60 mm sink and 64×64 grid. New die
widths range 6.717–19.869 mm and heights 6.207–20.025 mm, so both package sizes
exceed every die. Silicon/TIM materials, ambient temperature and cooling values
are unchanged. Narrow FILL strips remain possible; grid convergence and actual
HotSpot acceptance need the later real smoke test. No runtime thermal validation
or physical accuracy claim is made for the new layouts.

All runner tests use explicitly mocked solver calls and synthetic temperatures
in temporary test directories. Those values are not dataset labels. No full
simulation, ML training/tuning/evaluation, recalibration, or frontend integration
was performed. Dataset preparation ends here.
