# Render deployment — existing Stage A application

Deployment branch: `stage-a-render-deployment`, based on the existing `main`
commit `2b713196f4bd941ca25663d643f13a81beba69da`.

## Publication status

The production code is prepared and tested locally under Gunicorn. A public
Render deployment has not yet been created or verified. Connecting the owner's
Render account and obtaining approval to make the website public are required
before creating the service. Do not describe a predicted onrender.com hostname
as a verified live demo. The live URL and repository About/README demo link
should be added only after the deployed site passes the checks below.

## Existing application and preserved scope

`temperature_project/app.py` is the original Flask dashboard/API. It serves its
HTML and local CSS/JavaScript directly from `templates/` and `static/`. The
frontend uses same-origin API requests and renders the floorplan, temperature
chart and coupling tables using SVG. There are no external frontend CDN assets.

Predictions load a saved 8x8 thermal matrix for the selected floorplan and compute
`T_K = 318.15 + R @ P`, then `T_C = T_K - 273.15`. Power validation and matrix
ordering are inherited unchanged. Only the 23 original calibrated Stage A 2D
layouts are listed: Style_1 (3), Style_2 (5), Style_3 (5), Style_4 (10). Online
prediction never invokes HotSpot, calibrates matrices, generates data, or trains
models. Stage B, transient prediction and proactive DTM remain future work.

The app already resolves templates, static files and datasets from its source
file location, not the shell working directory. The deployment branch keeps
`app.py`, `predict.py`, `stage_a_pipeline.py`, HTML/JavaScript, datasets, matrices,
and published metrics byte-for-byte unchanged. A small CSS correction allows
grid items and long paths to shrink/wrap without page-level horizontal overflow;
the existing design, colors, controls and visualization logic are preserved. All model CSVs and floorplan
assets remain in Git; there is no upload, external model download, or database
required to serve the application.

## Deployment files

- Root `render.yaml`: Render Free Python web service for this deployment branch,
  service root `temperature_project`, health check `/healthz`, automatic redeploy
  on commits to the linked branch. No disk, paid database, secret or paid plan.
- `temperature_project/static/css/style.css`: prevents viewport overflow without
  redesigning the dashboard; wide tables retain their own horizontal scrolling.
- `temperature_project/requirements.txt`: adds Gunicorn 27.x to existing Flask
  and NumPy dependencies.
- `temperature_project/gunicorn.conf.py`: listens on `0.0.0.0:$PORT` (default
  10000); one preloaded worker with four threads, 60-second timeout and stdout/
  stderr logging. No Flask development server or debugger is exposed.
- `temperature_project/wsgi.py`: imports the original app, verifies runtime
  readiness, adds `/healthz`, disables debug/testing, caps bodies at 16 KiB,
  and adds `nosniff` and referrer policy headers. Predictions are unmodified.
- `temperature_project/deployment_check.py`: read-only build/startup checks for
  all 23 finite 8x8 models, geometry, templates, JS/CSS and saved metric files.
  A missing or invalid required asset causes build/startup to fail explicitly.
- `temperature_project/verify_deployment.py`: checks a local or public origin
  against the unchanged CLI and checked-in model matrices.
- `temperature_project/deployment_tests/`: production readiness tests and an
  optional real Chromium interaction test. Playwright is not a runtime dependency.

## Run locally

From the repository root on Linux/WSL (Python 3.12 recommended):

```bash
cd temperature_project
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install -r requirements.txt
python deployment_check.py
```

The original development command remains:

```bash
python app.py
# http://localhost:5000
```

Run the same production server used by Render:

```bash
PORT=10000 gunicorn --config gunicorn.conf.py wsgi:app
# http://localhost:10000
```

The Gunicorn config also sets its working directory, so this command from the
repository root works:

```bash
PORT=10000 gunicorn --config temperature_project/gunicorn.conf.py wsgi:app
```

## Render service settings — publish only after owner approval

Use the connected owner's Render workspace and free web-service option if the
account is eligible. Link GitHub to Render if Render requests its own GitHub
repository authorization; the ChatGPT GitHub connection does not automatically
grant Render access. Do not add paid services or switch plans without approval.

| Setting | Value |
|---|---|
| Repository | `https://github.com/pardhu0716-dot/temperature-predictor` |
| Branch | `stage-a-render-deployment` |
| Runtime | Python |
| Root directory | `temperature_project` |
| Build | `pip install -r requirements.txt && python deployment_check.py` |
| Start | `gunicorn --config gunicorn.conf.py wsgi:app` |
| Health check | `/healthz` |
| Instance plan | Free |
| Python | `PYTHON_VERSION=3.12.14` |
| BLAS threads | `OPENBLAS_NUM_THREADS=1`, `OMP_NUM_THREADS=1` |
| Logging | `PYTHONUNBUFFERED=1` |
| Bytecode | `PYTHONDONTWRITEBYTECODE=1` |
| Auto deploy | On commit to the linked deployment branch |

Render supplies `PORT`; do not hardcode its external URL or bind to localhost.
No secret API key is needed by this read-only predictor. Render handles the
public HTTPS endpoint. Keep any hosting/account credentials in the hosting
connection or secret settings, never in Git or the frontend.

The Blueprint can be used instead of manually entering equivalent settings.
Creating the web service publishes it, so wait for the owner's publication
approval even if the Blueprint is already committed. No merge into `main` is
needed to deploy this branch.

## Functional verification

From `temperature_project/`, with dependencies installed:

```bash
PYTHONDONTWRITEBYTECODE=1 python -m unittest discover -s tests -v
PYTHONDONTWRITEBYTECODE=1 python -m unittest discover -s deployment_tests -v
python verify_deployment.py --url http://localhost:10000
```

The HTTP verifier performs 276 prediction requests across all 23 floorplans:
default, midpoint, minimum and maximum power vectors, plus all eight individual
core targets. It compares K/C outputs with the unchanged `predict.py` and local
matrix CSVs within 1e-6 degrees, and checks homepage/assets, models, benchmark
APIs, missing layouts and invalid input responses. Use the same source revision
locally as the deployed application.

After Render reports a healthy deploy, replace the URL with the actual service
URL and run the same verification:

```bash
python verify_deployment.py --url https://ACTUAL-SERVICE.onrender.com
```

For real browser tests (test dependencies only):

```bash
python -m pip install playwright
python -m playwright install chromium
python deployment_tests/browser_smoke.py --url http://localhost:10000
# After deployment, repeat with the actual public HTTPS URL.
```

Chromium selects every layout through the dropdowns, uses presets and manual
power inputs, exercises all 184 individual-core buttons, checks SVGs/tables and
invalid-power messaging, and checks desktop/mobile viewports. Optional
`--screenshots /tmp/temperature-browser-checks` saves screenshots outside Git.
The existing JSDOM frontend regression suite in `tests/README.md` also remains
available and unchanged.

Local production verification is recorded in `temperature_project/DEPLOYMENT_VALIDATION.md`.
Live verification must be recorded separately once an actual service exists.

## Updating and automatic redeployment

Render watches `stage-a-render-deployment`, not `main` or a Stage B branch.
After approval and initial service creation, pushes to this linked branch trigger
new builds, runtime asset checks, and Gunicorn startup/health checks. Track deploy
status and errors in Render. After each material update, rerun the public HTTP
and browser checks before presenting the new release as verified.

If work lands on `main`, it will not automatically update this service: explicitly
bring reviewed Stage A changes into the deployment branch. Changing the service
to deploy `main` requires an approved merge and a branch-setting update. Do not
merge unfinished Stage B code to publish it as a working predictor. Render's
service settings can disable auto deploy or select an earlier successful deploy
for rollback without changing calibrated data.

## Limits

The Free plan may sleep after inactivity, causing a cold-start delay, and has
account-level usage/resource limits. The live URL is not guaranteed before
Render creates the service. Free-plan availability and any account verification
requirements must be confirmed in the owner's workspace. No always-on SLA or
unlimited load is claimed. Stage A's stored benchmark timings are historical;
the API's live timing on shared hosting will vary. This remains a steady-state
academic demonstration for calibrated layouts and the accepted power ranges.
