# Local Stage A deployment validation — add-only preparation

Date: 2026-11-13 (Asia/Calcutta).
Local branch: `stage-a-render-additive`.
Unchanged base: `2b713196f4bd941ca25663d643f13a81beba69da` (`main`).

## Preservation

All 1,338 original tracked files were individually hashed using Git's blob hash
and compared with the original commit; every hash matches. No existing file is
modified, removed, moved, renamed or replaced. This includes the original Flask
app, prediction/calibration scripts, requirements, README, HTML, CSS, JavaScript,
23 calibrated floorplans, matrix CSVs, training/validation assets and metrics.
The local main reference is unchanged. Original Stage B branches and the earlier
deployment branch/draft PR were not changed by this preparation.

The working tree contains only ten untracked additions: `render.yaml` and nine
files inside `deployment/` (see its README). No files are staged. No new commit,
push, merge, GitHub change, Render service or public website was created during
this add-only preparation. Runtime/test dependencies, screenshots and logs are
outside the checkout. No HotSpot, recalibration or ML training was invoked.

## Tested production wrapper

Linux; Python 3.12.14; Flask 3.1.3; NumPy 2.3.5; Gunicorn 27.0.1. The server
ran the original Flask application through `deployment/wsgi.py`, using one
Gunicorn gthread worker and four threads. No source datasets or model files
were copied or regenerated. The application serves only the original Stage A
catalog, not Stage B/C or future predictors. Dedicated tests check this scope.

| Test | Result |
|---|---|
| Runtime asset check | 23 finite 8x8 saved models, geometry, templates/static assets and saved metrics passed |
| Original Stage A regression tests | 5 passed |
| New deployment tests | 6 passed |
| Original frontend regressions | 240 assertions passed |
| Production HTTP checks | 276 prediction requests; 2,316 assertions passed |
| Prediction equivalence | Matches original CLI/matrices within 1e-6 K/C for all 23 floorplans, default/midpoint/minimum/maximum powers, and all eight individual targets |
| Chromium controls | 23 floorplan dropdown selections; 184 individual-core buttons; presets, manual power entry, Predict All, result/coupling tables and SVG chart/heatmap passed |
| Browser JavaScript errors | None |
| Homepage/CSS/JS and API endpoints | Passed, including saved benchmark endpoints |
| Invalid inputs/missing floorplan | Expected 400/404 responses |
| Oversized JSON | 413 for more than 16 KiB |
| Arbitrary working directory | Runtime checks and prediction passed without simulator execution |
| Render configuration | YAML parsed; plan Free, repository-root build, health route and manual deploy setting checked |

## Original layout limitations — deliberately preserved

Screenshots and Chromium measurements confirm pre-existing page-level horizontal
overflow. Measured viewport/document widths in pixels:

| Viewport | Document |
|---:|---:|
| 320 | 680 |
| 390 | 680 |
| 700 | 700 |
| 960 | 1,003 |
| 1,440 | 1,483 |

Functional controls, temperature results and visualizations work, but users may
need horizontal scrolling at these widths. No CSS fix, content hiding or layout
rewrite is included, because existing website files must remain unchanged.
This is an existing display limitation, not an unreported passing responsive
layout check.

## Hosting status and approval boundary

Render Free is configured but not provisioned. There is no verified public URL.
The owner must approve committing/pushing these new files, connect Render, and
approve public service creation before those actions occur. No merge into main
is required. No paid plan is authorized. Free instances may sleep after inactivity
and have resource/account limits.

After an approved actual deploy, repeat `deployment/verify_deployment.py` and
`deployment/tests/browser_smoke.py` against the real HTTPS URL before claiming
successful public deployment. Full commands and exact additions are documented
in `deployment/README.md`.
