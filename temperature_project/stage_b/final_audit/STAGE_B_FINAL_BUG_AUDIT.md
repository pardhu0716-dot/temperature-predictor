# Stage B final bug audit

Local audit on 20 November 2026; base `b9ec778e68605f78e717829266e16b38c7e284c6`, working branch `stage-b-web-integration`. No commit, push, deployment, model training or artifact replacement.

## Confirmed defects and fixes

| # | Defect / root cause | Minimal correction | Evidence / regression |
|---|---|---|---|
| 1 | v2 selected the first equidistant neighbor with `argmin`; different core indices changed temperatures for the same symmetric physical layout. Reproduced maximum delta 0.008604152714°C. | Both production and experimental feature builders average the powers of tied nearest neighbors, excluding self, with absolute 1e-12 mm tolerance and zero relative tolerance. | Symmetric permutations, reflection, exact nearest-feature value, production/experimental feature equality, and permutation checks on all 1,540 dataset records. |
| 2 | `independent_audit.py` and `evaluate_final.py` labeled calls as v1 but relied on the now-v2 default, including warmup calls. | Explicit `model_version='v1'` on all four affected calls. | AST regression requires both calls in each script to select v1. |
| 3 | Request size was checked after an unbounded read when content length was unavailable. | Read at most 100,001 bytes before applying the local 100 KB limit. Stage A limits unchanged. | Guarded unknown-length WSGI stream returns JSON 413 without an unbounded read. |
| 4 | Batch requests silently ignored root-level model selection and other settings. | Reject root keys other than `batch`; settings belong on each batch item. | Root model/ambient tests return useful 400 errors. Existing mixed/bad batch tests pass. |
| 5 | Very large JSON integers leaked NumPy ufunc errors during numeric validation. | Convert numeric scalars safely before finiteness and supported-range checks; reject bools/non-numbers. | Negative, zero, huge, NaN and infinity power cases fail safely; field-specific errors replace ufunc messages. |
| 6 | v2 loader accepted zero/negative scales and unchecked nonfinite scalar parameters; feature arrays lacked a strict inference contract. | Validate finite positive scales, scalar finiteness, metadata, shape/order/units, feature shape/finiteness, and output finiteness. | Missing/tampered artifacts, invalid scalers/scalars and malformed feature arrays fail closed. Frozen arrays unchanged. |
| 7 | Runtime model failures generated an HTML error instead of the API's JSON contract. | Log internal RuntimeError and return sanitized JSON 503; reject malformed/nonfinite model output. | Injected model error produces JSON 503 without private exception details. |
| 8 | A pending floorplan GET could finish after the user switched to custom mode and overwrite edits. | Lock mode, reset and editor controls while geometry is loading, then unlock. Existing request/revision guards remain. | Browser test reproduces old failure, verifies loading locks, unlocks and both model selections. |
| 9 | Two experimental report scripts depended on benchmark CSVs absent from the tracked checkout. | Strict named input parser and in-memory baseline reconstruction from tracked traces, steady files and matrices; explicit v1 inference. | Parser regressions plus full 460-workload reconstruction matched the new saved predictions; no historical output overwritten. |

## Preserved model semantics and provenance

No coefficient, intercept, fitted scaler, model metadata, schema, raw data or historical benchmark was edited. The nearest-neighbor fix intentionally changes inference feature semantics on ties. Five Stage B **training** layouts have ties (`b_fp_031`, `035`, `038`, `067`, `076`); none of the saved validation/test layouts do. Frozen scalers were originally fitted with the first-neighbor rule. That legacy rule is retained only as an explicit audit option, and training-only scaler statistics still match to 1e-12. Production uses the corrected rule. This is disclosed rather than presenting old metrics as newly measured results.

Frozen v1 uses identity-dependent features and separate output weights: arbitrary physical core renaming is not a model symmetry. Input row reordering is safely canonicalized by IDs. No attempt was made to disguise that model limitation by averaging predictions or retraining; the UI now explains it. Corrected v2 is physically core-relabel equivariant within floating-point tolerance.

## Verification boundaries

`test_final_audit.py` adds 16 regression/audit tests. Old-code reproduction logs, browser evidence and current test logs are in `evidence/`. Expected injected-error stack traces in unit logs are test fixtures, not unexplained runtime failures. The complete legacy report-writing mains were not executed because they overwrite historical outputs; their changed baseline path was exercised directly. The optional real HotSpot simulation integration test was skipped; existing simulation labels were checked and used without rerunning HotSpot.

See `STAGE_B_FINAL_VERIFICATION.md` for the complete gate and `STAGE_A_VS_STAGE_B_23_FLOORPLANS.md` for fresh numerical results.
