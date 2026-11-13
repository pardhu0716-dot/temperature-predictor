# Add-only Render Free deployment: Stage A

Status: local preparation only. No commit, push, merge or publication is authorized
by this preparation. The local branch is `stage-a-render-additive`, based on
unchanged `main` commit `2b713196f4bd941ca25663d643f13a81beba69da`.

Every existing tracked project file is preserved byte-for-byte. This approach
adds a separate deployment wrapper and configuration; it does not edit, rename,
move or replace the original application, requirements, README, HTML, CSS, JS,
data, models, calibrated matrices or metrics. Earlier deployment branch/PR work
is separate and is not modified by this local preparation.

## Scope

The original `temperature_project/app.py` exposes the completed Stage A Flask
website and APIs for 23 calibrated 2D floorplans. It uses the original saved
8x8 thermal resistance CSVs with `T_K = 318.15 + R @ P` and
`T_C = T_K - 273.15`. Existing power ranges, geometry and matrix ordering are
unchanged. No HotSpot, calibration, model fitting or training occurs online.

This branch starts at Stage A's existing `main`, rather than a Stage B branch.
No Stage B/C or other unfinished predictor is added or exposed. The original
sample assets and documentation remain intact; their presence is not a claim
that future stages are complete. The production catalog is exactly the 23
original Stage A floorplans across Style_1–Style_4.

## Exact new files

| New file | Purpose |
|---|---|
| `render.yaml` | Render Free service configuration; automatic deployment initially off |
| `deployment/requirements.txt` | Includes original requirements without editing them; adds Gunicorn 27.x |
| `deployment/gunicorn.conf.py` | One process / four threads, stdout logging, dynamic PORT, source import paths |
| `deployment/wsgi.py` | Imports original app, checks readiness, adds health route and production settings |
| `deployment/check_assets.py` | Read-only verification of all 23 saved models and frontend/metric assets |
| `deployment/verify_deployment.py` | HTTP comparison against original CLI and saved matrices |
| `deployment/tests/test_runtime.py` | Production readiness, Stage A scope, paths and request-limit tests |
| `deployment/tests/browser_smoke.py` | Optional Chromium checks of controls, predictions and visualizations |
| `deployment/README.md` | Deployment and verification instructions |
| `deployment/VALIDATION.md` | Local results, integrity evidence and remaining limitations |

No existing file needs a modification. The WSGI wrapper sets debug/testing off,
limits request bodies to 16 KiB, adds basic response headers, and adds `/healthz`.
These are in-memory configuration changes during production startup; the
original source files and prediction calculation are untouched. Startup fails
if required models or assets are missing, invalid or nonfinite. Health reports
Stage A and 23 calibrated floorplans. Gunicorn's optional management socket is
disabled; no credentials or home-directory management files are required.

## Local build and production server

Run from repository root on Linux/WSL, Python 3.12 recommended:

```bash
python3 -m venv /tmp/temperature-render-venv
source /tmp/temperature-render-venv/bin/activate
python -m pip install -r deployment/requirements.txt
PYTHONDONTWRITEBYTECODE=1 python deployment/check_assets.py
PORT=10000 PYTHONDONTWRITEBYTECODE=1 gunicorn --config deployment/gunicorn.conf.py wsgi:app
```

Open `http://localhost:10000`. The original development command remains
`python temperature_project/app.py`; it does not need Gunicorn and was not edited.
The production config sets source/import paths itself. The app's original paths
already resolve from its source directory, so no dataset-path edits are needed.

## Tests

From repository root using the venv above:

```bash
PYTHONDONTWRITEBYTECODE=1 python -m unittest discover -s temperature_project/tests -v
PYTHONDONTWRITEBYTECODE=1 python -m unittest discover -s deployment/tests -v
PYTHONDONTWRITEBYTECODE=1 python deployment/verify_deployment.py --url http://localhost:10000
```

The HTTP verifier checks the homepage, CSS/JS, health, catalog, geometry/model and
saved benchmark endpoints, invalid requests, and 276 prediction requests across
all 23 floorplans. It compares temperatures against the unchanged `predict.py`
and original matrix CSVs to within 1e-6 K/C. It tests default/midpoint/minimum/
maximum powers plus all eight individual targets per floorplan.

For optional browser tests, install test-only dependencies outside project files:

```bash
python -m pip install playwright
python -m playwright install chromium
PYTHONDONTWRITEBYTECODE=1 python deployment/tests/browser_smoke.py --url http://localhost:10000
```

Chromium selects all 23 floorplans through their actual controls, uses presets,
manual input, all 184 individual-core prediction buttons, invalid-power feedback,
SVG heatmap/chart and result/coupling tables. It reports viewport measurements
without modifying or hiding the original CSS behavior. Optional
`--screenshots /tmp/temperature-browser-checks` stores images outside the repo.
The original frontend regression commands remain in `temperature_project/tests/README.md`.

## Render settings: only after approval

| Setting | Value |
|---|---|
| Service type/runtime | Web Service / Python |
| Repository | `https://github.com/pardhu0716-dot/temperature-predictor` |
| Branch | `stage-a-render-additive` (not pushed yet) |
| Root directory | Repository root (leave blank) |
| Plan | Free |
| Build | `pip install -r deployment/requirements.txt && python deployment/check_assets.py` |
| Start | `gunicorn --config deployment/gunicorn.conf.py wsgi:app` |
| Health | `/healthz` |
| Python | `PYTHON_VERSION=3.12.14` |
| Automatic deploy | Off until the owner explicitly enables it |

The Blueprint also sets `PYTHONUNBUFFERED=1`, `PYTHONDONTWRITEBYTECODE=1`,
`OPENBLAS_NUM_THREADS=1` and `OMP_NUM_THREADS=1`. Render provides PORT and HTTPS.
No secret is required by the predictor. Hosting authentication belongs in the
Render connection/settings, never in Git, chat, HTML or JavaScript.

Owner approval is required before committing/pushing these new files, and again
before creating the public service. Render must be connected to the owner's
account; Render may separately require GitHub repository authorization. Use Free
only. If the account cannot use Free, stop and ask rather than selecting a paid
plan. There is no need to merge into `main` to deploy this dedicated branch.

After publication is approved and Render reports a healthy deploy, use the actual
HTTPS URL to run both verification scripts again. Local tests cannot prove a
public deployment works. Do not invent a service URL or add an unverified Live
Demo link. A subsequent README/About update would require separate permission
because existing-file modifications are currently prohibited.

To enable future auto-deploys after initial approval, change the Render service's
automatic deployment setting to commits on the dedicated deployment branch.
Changes to `main` or Stage B branches would not deploy automatically. Keep
Stage A-only changes reviewed and retest the public app after updates.

## Limits

Render Free may sleep after inactivity and has resource/usage limits; a first
request may have a cold-start delay. Free availability/account verification
must be checked in the connected account. No always-on SLA is claimed.

Original CSS has horizontal overflow on narrow screens, especially in model
paths and grid/table layouts. Functional controls and plots are tested, but this
pre-existing display behavior is deliberately preserved. No CSS fix is included
because the owner explicitly prohibited edits to existing website files.

Only Stage A's calibrated layouts and accepted input ranges are supported.
Historical benchmark timings displayed by the existing site do not predict
Render's live latency. No new simulations or thermal calibration are needed.
