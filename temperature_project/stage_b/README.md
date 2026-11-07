# Stage B dataset preparation

The expanded input dataset and WSL simulation runner are documented in
[DATASET_PREPARATION.md](DATASET_PREPARATION.md). The new dataset contains 77
additional layouts and 1,540 power configurations. No temperature labels or ML
models were produced during preparation.

The following is the historical five-floorplan pilot report; its stopping limit
applies to the original pilot generator. The new preparation script reuses the
geometry functions without regenerating that pilot.

# Stage B: five-floorplan geometry pilot

This directory contains only new files. Source: repository commit
`2b713196f4bd941ca25663d643f13a81beba69da`. No Stage A files were changed;
no HotSpot runs, recalibration, ML training, commits, or pushes were performed.

## What the source files actually show

All 23 silicon and 23 TIM floorplans and all 23 layer configuration files were
parsed and checked. Big cores are 4×4 mm; Little cores are 2×2 mm, with total
active area 56 mm² in every die. Core IDs do not reliably identify type:
Style_2 changes the Big-core IDs (3/7, 3/5, 6/7, 0/1, 6/7 respectively).
Style_3 also lists cores in a nonnumeric order. Preserve names and file ordering;
infer type from geometry. Do not attach an old matrix or trace to new geometry.

| Family | Actual arrangement | Dies, width×height (mm) | FILL count | Whitespace |
|---|---|---|---|---|
| Style_1 (3) | Two touching Big cores in a horizontal pair; six touching Little cores in a separate 3×2 or 2×3 cluster | 9×10, 9×11, 16×8 | 3 each | 37.78–56.25% |
| Style_2 (5) | Compact mixed rows; Big-core identities vary. Bigs touch one another, horizontally or vertically. Little cores occupy rows around them; fp_002 has a less regular stepped arrangement. | 10×8, 9×10, 12×6, 10×6 (twice) | 1–4 | 6.67–37.78% |
| Style_3 (5) | A Little core separates Big cores in the row/column sequence. fp_001/002 include explicit 1 mm gaps; fp_003–005 have touching blocks and no universal 1 mm spacing rule. | 7×14, 13×12, 14×6, 6×14, 14×6 | 3–12 | 33.33–64.10% |
| Style_4 (10) | Irregular, scattered placements on a fixed square die; some Little cores touch. These are not regular rows or a uniform-spacing grid. | 14×14 throughout | 10–17 | 71.43% throughout |

Source positions use 1 mm increments; source core dimensions are 2 or 4 mm.
Style_1 cluster separation is 2 mm, 3 mm, or 4 mm depending on the sample.
The style labels are observed families, not mathematically disjoint classes;
no original generator specification is present. A filename's style cannot be
reconstructed from a universal whitespace or gap threshold.

Every original silicon layer has a rectangular, fully tiled die: FILL is the
passive complement, not an additional core. TIM is one full-die `TIM_Top` block.
All LCFs use silicon: lateral Y, power Y, specific heat 1.75e6 J/m³K,
resistivity 0.01 mK/W, thickness 0.00015 m; TIM: Y, N, 4e6, 0.25,
2.0e-05 m. These values are reproduced without alteration.

## Files and algorithm

- `generate_floorplans.py`: dependency-free generator and geometry validation.
- `test_generate_floorplans.py`: seven tests, including independent cell coverage.
- `pilot/source_inventory.json`: dimensions, whitespace, FILL counts for all 23 sources.
- `pilot/manifest.json`: pilot metadata and validation summary.
- `pilot/b_fp_001` through `b_fp_005`: silicon FLP, TIM FLP, LCF,
  zero-power formatting fixture, and `metadata.json` per layout.
- `pilot/preview.svg`: visual inspection of the five generated layouts.

The generator cycles through the selected families and randomly chooses an
original template. It expands ordered x/y coordinate ranks by random added
spacing, adds die margin, then attempts bounded independent position jitter.
Moves that overlap or cross the die boundary are rejected. All calculations use
integer micrometres; FLP output has six decimal places in metres. Big and Little
core dimensions and original core names/order remain fixed. No rotations or
scaling of cores occur. Family membership records source provenance; perturbed
layouts are not claimed to satisfy an unknown original generation rule.

FILL is rebuilt by partitioning at all core edges, merging free horizontal runs,
and extending identical runs vertically. Positive dimensions, unique names,
bounds, non-overlap, exact total area, and 2 Big + 6 Little counts are checked
before writing and again after FLP round-trip. Given in-bounds disjoint rectangles,
equal summed area proves complete die coverage. Tests independently verify each
induced cell has exactly one owner, including original layouts.

Diversity uses sorted center-to-center distance lists grouped by pair type
(Big–Big, Big–Little, Little–Little), plus sorted die dimensions. RMS descriptor
distance must be at least 150 µm from all 23 originals and every previously
accepted pilot layout. This comparison ignores IDs, translations, reflections,
and rotations of the core arrangement. It is a conservative heuristic: distinct
geometries can share descriptors and be rejected; it does not guarantee an
optimal or exhaustive diversity distribution. Exact duplicate core layouts
cannot pass. No uniqueness claim is based on FILL segmentation.

Parameters support die dimensions (`--die-um WIDTH HEIGHT`), added spacing,
margin, jitter, minimum gap, selected styles, seed, and all eight explicit
core positions through `--positions-json` (JSON name -> [x,y] in µm; count=1).
Whitespace is controlled through die area: fraction = 1 - 56,000,000/(W*H),
with W/H in µm; spacing changes internal whitespace distribution. There is no
independent whitespace target switch because fixed core area and die area
already determine that fraction. Infeasible constraints fail without retaining
candidate output. CLI count is deliberately capped at five pending approval.
Output must be a new subdirectory of `stage_b`; existing paths are refused.

## Pilot results

| ID | Source family/template | Die (mm) | FILLs | Nearest descriptor RMS (µm) |
|---|---|---|---|---|
| b_fp_001 | Style_1/fp_003 | 17.466×8.423 | 22 | 934.01 |
| b_fp_002 | Style_2/fp_003 | 13.287×6.278 | 20 | 553.52 |
| b_fp_003 | Style_3/fp_001 | 8.031×16.422 | 25 | 891.11 |
| b_fp_004 | Style_4/fp_001 | 15.556×14.670 | 25 | 682.96 |
| b_fp_005 | Style_1/fp_001 | 10.076×10.700 | 25 | 841.01 |

All five pass bounds, no overlap, exact full-die coverage, fixed dimensions,
FLP formatting/round-trip, TIM coverage, LCF references, trace header ordering,
and diversity checks. Seven unittest tests pass. Repeating generation in a
temporary directory produces byte-identical FLP/LCF/ptrace files; temporary
copies are deleted. Only five generated floorplans are retained. Metadata output
paths naturally differ between reproduction directories.

## Reproduce

From repository root (Python 3.10+):

```bash
python3 temperature_project/stage_b/generate_floorplans.py --seed 20261107 --count 5 --output temperature_project/stage_b/pilot_reproduced
python3 -m unittest discover -s temperature_project/stage_b -v
```

The delivered `pilot/` already exists, so use a new output name. The generator
can also be copied with this Stage B directory into the same position in an
existing repository checkout. It locates the original dataset relative to itself.

## HotSpot compatibility status

Static conventions match the supplied dataset: five fields per FLP row, metres,
Core_* / FILL_* names, full rectangular coverage, original LCF materials, two
layers, relative filenames, and a full-die TIM layer. Run HotSpot from inside a
floorplan directory so LCF relative references resolve. Future powered ptraces
must include every silicon block in FLP order, with FILL power zero. The included
`*_zero.ptrace` is only a zero-power format fixture, not a workload or training set.

The supplied reference config has a 30 mm spreader, 60 mm sink, and 64×64 grid.
All pilot dies are smaller than the spreader; the generator rejects sides of
30 mm or more. Jitter can produce narrow FILL strips smaller than a grid cell;
geometry validation does not establish numerical grid convergence. FILLs inherit
silicon layer properties; zero power must be specified in a future powered trace.
No thermal/ambient assumptions were changed and no config was copied or modified.

`hotspot` is not installed on PATH in this environment. No simulator invocation
was made, so runtime acceptance and numerical behavior remain unverified.
No bulk simulation or ML work is authorized by this pilot. Stop here pending
approval before expanding the catalog or performing simulation work.
