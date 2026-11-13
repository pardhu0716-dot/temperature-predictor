# Stage A production deployment validation

Tested on 2026-11-13 (Asia/Calcutta) from the dedicated
`stage-a-render-deployment` branch based on `main` commit
`2b713196f4bd941ca25663d643f13a81beba69da`.

## Local production results

Runtime: Linux, Python 3.12.14, Flask 3.1.3, NumPy 2.3.5, Gunicorn 27.0.1.
Gunicorn runs the production `wsgi:app` entry point, not Flask's development
server. The exact Render build/start commands are in the repository's
`DEPLOYMENT.md` and `render.yaml`.

| Check | Result |
|---|---|
| Original Stage A regression suite | 5 tests passed |
| Existing frontend regression suite | 240 assertions passed |
| New production-readiness suite | 5 tests passed |
| Read-only runtime checks | All 23 finite 8x8 saved matrices, geometry, static assets and metric CSVs available |
| Gunicorn startup/config | Passed, one gthread worker / four threads, debug disabled |
| Production HTTP comparison | 276 prediction requests; 2,316 assertions passed |
| Prediction equivalence | All 23 layouts, defaults/midpoints/minimum/maximum powers and all eight individual targets match unchanged CLI/matrices within 1e-6 K/C |
| Homepage, CSS, JS, catalog, floorplan/benchmark APIs | Passed |
| Invalid powers and nonexistent floorplan responses | Expected HTTP 400/404 |
| Request body size | Payload exceeding 16 KiB rejected with HTTP 413 |
| Arbitrary working directory | Models and frontend resolve correctly; tests prohibit simulator calls |
| Chromium interactions | All 23 dropdown selections and 184 individual-core buttons passed; presets, manual power entry, invalid-input feedback, SVGs and tables passed |
| Responsive rendering | No page-level horizontal overflow at widths 320, 390, 700, 960 and 1440 pixels; wide tables scroll locally |
| Browser JavaScript errors | None |
| Render YAML | Parsed and checked against documented service fields |

The responsive CSS correction retains the existing colors, controls, structure,
HTML, JavaScript and SVG prediction/rendering logic. Only grid/flex minimum
widths and wrapping were adjusted. Gunicorn's optional management socket is
disabled so it requires no home-directory management files.

Preservation audit: the original backend, prediction/calibration algorithms,
HTML and JavaScript, all dataset assets, saved matrices and metric files are
unchanged. The only modified existing files are the root README, runtime
requirements (adding Gunicorn), and CSS (responsive overflow correction).
Existing Stage B branches are not part of the Stage A deployment branch and
were not changed. No HotSpot process, recalibration or ML training was run.

## Public verification status

**Not yet deployed or verified on Render.** The owner must connect Render and
approve creation of the public Free web service first. Local checks do not
establish that a public hostname exists or that a deployment succeeded.
No public URL has been invented or added to the repository About/Live Demo.

Once Render has deployed the approved service, run `verify_deployment.py` and
`deployment_tests/browser_smoke.py` with its actual HTTPS URL. Inspect hosting
logs and the live page, record the deployment ID/URL/results, and then add the
verified URL to README/About. If Free is unavailable or Render requires account
verification, resolve it with the owner rather than selecting a paid plan.
