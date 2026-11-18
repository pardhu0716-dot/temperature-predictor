# Local integration verification

Verified on 19 November 2026 (Asia/Kolkata) on local branch
`stage-b-web-integration`, based on `5ea2a592af0b212030df292157382c27e88bf9c4`.
No commit, push, merge or deployment was performed.

## Measured outcomes

- **9 new API/integration tests passed.** All 77 real geometry records and default
  predictions load; geometry values match the existing FLP loader. All eight
  power inputs influence predictions. Invalid numbers, units, IDs, dimensions,
  overlaps, boundaries, powers, ambient/material overrides, malformed JSON,
  unknown layouts, batch bounds and body limits return expected errors.
- **240 saved validation cases reproduced in both existing and custom modes**,
  including reverse-ordered core inputs: 480 API responses / 3,840 temperatures
  match the previously measured Lasso predictions within 1e-10°C absolute
  tolerance. No fit is allowed during this test. These are reproduction checks,
  not new ML benchmark estimates.
- **All 23 Stage A floorplans** passed all-core and individual-core API checks
  against their existing `45 + R @ P` matrices, within output-rounding tolerance.
- **5 original Stage A backend tests passed.**
- **240 original frontend regression assertions passed.**
- **Chromium desktop 1440×1100 and mobile 390×844 checks passed**: all eight power
  controls, floorplan switching, existing/custom modes, die/coordinate editing,
  preview updates, reset, overlap errors, server-error clearing, delayed-response
  rejection, result summaries and shared Stage A/Stage B navigation. SVG heatmap
  and bar values exactly equal the API temperatures. Stage A's original all-core
  and individual-core buttons also work in the combined application.
- **No JavaScript errors or failed Stage B assets** were observed. Stage B has
  no page-level horizontal overflow at those viewports; mobile inputs use cards.
  Desktop and mobile screenshots were visually inspected.
- Removing the new navigation fragment and its link from the combined Stage A
  response reproduces the original rendered HTML exactly.

## Integrity and scope

All **3,306 files tracked by the base commit remain unchanged**. Only the 12 new
integration files below are created. The approved 11,197-byte Lasso and its SHA-256
`db1b49992c8a11b95c85a98be861b95502b00b0d3ec680709778236484d9ae80`
remain unchanged; so do fitted scalers, source CSV, geometry/simulation inputs,
Stage A, Stage C, experiment results and deployment configuration.

No held-out temperature data, test NPZ or final test prediction CSV was used for
these tests. Only historical metric constants are displayed as model information.
The validation reference CSV and original geometry/power inputs supply inference
checks. No model selection, recalibration, training, HotSpot or new ML accuracy
benchmark occurred. Model loading is once per combined Flask application/process.

## New files

- `temperature_project/combined_app.py`
- `temperature_project/stage_b/web/__init__.py`
- `temperature_project/stage_b/web/service.py`
- `temperature_project/stage_b/web/requirements.txt`
- `temperature_project/stage_b/web/README.md`
- `temperature_project/stage_b/web/VERIFICATION.md`
- `temperature_project/stage_b/web/templates/stage_navigation.html`
- `temperature_project/stage_b/web/templates/stage_b.html`
- `temperature_project/stage_b/web/static/stage_b.css`
- `temperature_project/stage_b/web/static/stage_b.js`
- `temperature_project/stage_b/web/tests/test_integration.py`
- `temperature_project/stage_b/web/tests/browser_checks.py`

## Remaining limitations

Only the supported 2D, two-Big/six-Little, original fixed-material/ambient setup is
accepted. Custom geometry extrapolates beyond measured layouts; neither the UI
nor historical held-out error figures guarantee arbitrary-layout accuracy.
Inference is not an online simulator, and model outputs are not clamped to hide
poor extrapolation. Timing is actual local inference, not an HTTP latency promise.
Original raw HotSpot provenance remains subject to the already documented limit.
Only Chromium desktop/mobile emulation was exercised; no physical-device or
Safari/Firefox verification is claimed. The combined entry point is local-only;
existing Render/Stage A deployment continues using its unchanged code/settings.
