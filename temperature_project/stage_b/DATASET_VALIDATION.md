# Dataset preparation validation

Base: `c8cc4e27a332af168b9e539aae80da615735eeca` (`stage-b-pilot`).
Validation date: 2026-11-08 (Asia/Calcutta).

## Verified inputs

- 77 new layouts, IDs `b_fp_006`–`b_fp_082`, across all four source families.
- Two 4×4 mm Big cores plus six 2×2 mm Little cores in every layout.
- Positive, in-bounds, nonoverlapping rectangles with complete die coverage.
- Independent cell-by-cell oracle verifies every induced cell has one owner.
- Full-die TIM, two-layer LCF references and original material assumptions.
- All layouts distinct from the 23 originals, five pilots, and each other under
  the original conservative typed-distance descriptor (minimum required 150 µm;
  smallest accepted distance 278.13 µm).
- 20 unique power configurations per new layout, 1,540 total: 77 references and
  1,463 independent random configurations. Every ptrace was read and validated
  for exact FLP header order, geometry-based Big/Little assignment, range, finite
  nonnegative values, exactly zero FILL power, and per-layout uniqueness.
- 24 distinct Big-core ID pairs; classification is independent of numbering.
- Complete 77-layout / 1,540-trace regeneration compared byte-for-byte:
  all 1,850 generated files identical. Temporary reproduction removed afterward.
- Both manifests include portable relative paths and SHA-256 input inventories.

## Automated checks

| Suite / check | Result |
|---|---|
| Stage B unittest discovery (existing and new) | 26 tests: 25 passed, 1 real-HotSpot integration test skipped |
| Stage A saved-model regressions | 5 tests passed |
| Stage A frontend regressions | 240 assertions passed, including all 23 layouts and response-order cases |
| Full manifest validation | 77 floorplans / 1,540 traces passed |
| Full runner dry run | 1,540 planned commands, no solver invocation or result directories |
| Full dataset reproduction | All 1,850 files byte-identical |
| Protected input audit | Original tracked files outside Stage B and all 28 pilot files unchanged |

Stage A regression sources were not edited to achieve these results. Python
and jsdom test dependencies were installed outside the checkout. Runtime/test
caches, logs, and temporary data are not included in the commit.

Mocked runner tests cover exact command construction; copied relative LCF
paths; shuffled/raw/prefixed core output names; missing, duplicate and nonfinite
temperatures; Kelvin-to-Celsius conversion; valid-result skip without solver
invocation; changed config/binary and corrupted-result rejection; timeout,
nonzero exit, missing output, retry with retained failure records, locks, and read-only result validation without solver calls or output writes.
Synthetic temperatures exist only inside temporary test fixtures and are never
committed as real labels. Default Stage B tests cannot invoke the opt-in real
HotSpot integration test.

## Boundaries

No real HotSpot simulation was run. New dataset runtime acceptance, thermal
accuracy, and grid convergence remain to be checked later on WSL. No new
calibrated matrices, temperature label CSVs, ML models, training, tuning,
evaluation, or frontend changes were produced. Narrow FILL strips can exist,
so the shipped 64×64 grid is not asserted to be numerically converged.

See [DATASET_PREPARATION.md](DATASET_PREPARATION.md) for exact WSL validation,
dry-run, smoke-test, batch execution, and resume commands. The batch execution
commands are provided for later use and were not executed for this delivery.
