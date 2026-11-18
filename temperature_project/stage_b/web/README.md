# Local Stage A + Stage B integration

Stage B is a page of the **same existing Flask application**, not a second website.
This work is local to `stage-b-web-integration`, based on published commit
`5ea2a592af0b212030df292157382c27e88bf9c4`. Nothing is committed, pushed, merged
or deployed as part of this integration task.

`combined_app.py` imports the original `app.app`, registers a Stage B Blueprint
and adds a shared Stage A / Stage B navigation fragment to the Stage A homepage
response. Removing that fragment and its stylesheet link yields exactly the
original HTML. Original Stage A Python/templates/CSS/JS and prediction endpoints
remain unchanged; `python app.py` still runs the original Stage A application.
Stage B reuses its design tokens, engineering header, cards and buttons.

## Local setup

Python 3.11+ is required by the pinned numerical dependencies; tested on 3.12.14.
From this checkout's `temperature_project/` directory:

```bash
python3 -m venv /tmp/temperature-web-venv
source /tmp/temperature-web-venv/bin/activate
python -m pip install -r stage_b/web/requirements.txt
python combined_app.py
```

Open `http://127.0.0.1:5000/` for Stage A and use the shared navigation for
`http://127.0.0.1:5000/stage-b`. This is a local development server, not a deployment.
In the current Work workspace a verified instance also runs on port 5205.

Dependencies are existing Flask plus NumPy 2.3.5, scikit-learn 1.8.0 and joblib
1.5.3. No XGBoost, neural-network package, simulator or training CSV is needed
for inference. The model, preprocessing files and geometry manifest/FLPs must
remain available at their committed repository paths.

## Model and geometry path

`Predictor` loads the trusted, hash-pinned local Lasso bundle once when the
combined app registers its Blueprint. Startup fails if the model checksum,
scalers, alpha, core ordering or feature definitions do not match. Re-registering
on the same Flask app reuses the existing predictor. There is no online fitting.

The adapter calls the unchanged `stage_b.ml.features.build_features` and the
saved `ModelBundle.predict`: 252 engineered features, saved X scaling, eight
Lasso outputs, inverse Y scaling to °C. No duplicated ML implementation or new
scaler is introduced. Geometry comes from the original manifest and hashed FLPs;
its original loader validates bounds, core count, FILL coverage and LCF assumptions.
The Stage B catalogue contains the 77 original b_fp_006…082 geometries and does
not load temperature labels.

## API

| Method/path | Behavior |
|---|---|
| `GET /` | Original Stage A dashboard plus shared stage navigation |
| `GET /stage-b` | Stage B existing/custom input modes in the same app |
| `GET /api/stage-b/model` | Approved model, units, limitations and historical metadata |
| `GET /api/stage-b/floorplans` | All 77 real Stage B IDs and die dimensions |
| `GET /api/stage-b/floorplans/<id>` | Real die/core geometry and supported power ranges |
| `POST /api/stage-b/predict` | One prediction or a batch of 1–32 predictions |

All original `/api/floorplans`, `/api/floorplan/...`, `/api/predict` and benchmark
endpoints retain their existing behavior. Stage B request limits are local to
its route; no global Stage A limits/configuration were changed.

Existing-layout request:

```json
{
  "mode": "existing",
  "units": {"geometry": "mm", "power": "W", "temperature": "C"},
  "floorplan_id": "b_fp_006",
  "powers_w": {
    "Core_0_L0": 7.5, "Core_1_L0": 7.5, "Core_2_L0": 7.5, "Core_3_L0": 7.5,
    "Core_4_L0": 15, "Core_5_L0": 15, "Core_6_L0": 7.5, "Core_7_L0": 7.5
  }
}
```

For custom input, use `mode: "custom"`, the same units, an explicit
`die: {"width_mm": ..., "height_mm": ...}` and `cores`: eight objects containing
`id`, `width_mm`, `height_mm`, `x_mm`, `y_mm`, `power_w`. Rows may be in any order;
IDs are canonicalized before features and targets. IDs must be exactly
Core_0_L0 through Core_7_L0, each once. Die coordinates use a bottom-left origin.

Constraints: two 4×4 mm Big and six 2×2 mm Little cores; no overlap, negative
coordinates or out-of-bounds rectangles. Powers remain within the training
ranges (Big 10–20 W, Little 5–10 W). Booleans, strings and nonfinite numbers are
rejected. Custom die dimensions must be positive and at most 1000 mm as a
numerical/rendering guard, **not an accuracy-supported die range**. Ambient and
material/layer settings are fixed to training assumptions, including declared
45°C ambient; the API rejects requests attempting to add those parameters.

Batch format: `{"batch": [prediction_object, ...]}`. The 100 KB body cap and
32-case cap bound work per request. Every batch item validates before prediction;
invalid batches return an error rather than a partial successful response.
Errors return JSON with `error`, status 400 (invalid/malformed input), 404 (unknown
floorplan), 413 (oversized request) or 415 (non-JSON content type).

Results contain canonical `core_order`, `temperatures_c`, minimum/maximum,
hottest core, actual measured `latency_ms`, separately measured
`feature_engineering_ms`, units and model metadata. Inference latency includes
X scaling, Lasso and inverse Y scaling; excludes features, HTTP and rendering.
It is not a guarantee derived from old benchmark timing.

## Frontend behavior

Existing mode loads the real geometry, locks it and exposes eight power controls.
Custom mode starts from that same real layout, allowing IDs, die and all core
geometry to be edited. Reset restores the selected original layout and midpoint
powers while retaining the selected mode. Geometry previews update as fields
change. Displayed repository coordinates remove floating-point serialization
noise at six decimal mm places; original coordinates have micrometre precision.

Each heatmap rectangle uses its API temperature and actual core geometry. Passive
space stays neutral. The blue-to-red scale maps the current minimum/maximum;
there is no fabricated interpolation. SVG titles and the result table expose
numerical temperatures. Bars use actual temperatures with a zero baseline.

Every edit, reset, layout or mode change clears old results. Out-of-date async
responses cannot overwrite a newer input state. Server errors clear chart and
heatmap values. The mobile input table becomes eight compact editable cards;
Stage B has no page-level horizontal scrolling at the tested desktop/mobile
viewports. The existing Stage A layout is preserved, including any pre-existing
viewport limitations.

## Verification

From `temperature_project/`, with dependencies installed:

```bash
PYTHONDONTWRITEBYTECODE=1 python -m unittest discover -s stage_b/web/tests -v
PYTHONDONTWRITEBYTECODE=1 python -m unittest discover -s tests -v
```

Optional browser checks, with the combined app running separately:

```bash
python -m pip install playwright
python -m playwright install chromium
python stage_b/web/tests/browser_checks.py --url http://127.0.0.1:5000 --out /tmp/stage-b-web-checks
```

The unchanged Stage A Node suite additionally needs Node and jsdom available:
`node tests/frontend_regressions.cjs`. Its Python subprocess must use the same
virtual environment with Flask installed.

The new tests use only original **validation** references and original workload
powers/geometry. All 240 validation cases are replayed in existing and custom
modes, including reversed core order, to tolerance 1e-10°C without fitting. No
held-out temperature labels are read, no new model selection occurs, and no
historical test benchmark is rerun. See `VERIFICATION.md` for measured outcomes.

## Limits

Valid custom geometry does not guarantee accuracy. This remains an eight-core,
2D, fixed-material/ambient steady-state model; no transient, 3D, DTM or Stage C
functionality is added. Historical test metrics are labeled historical and are
not website guarantees. Extreme valid custom layouts may yield physically poor
extrapolations; the UI does not clamp or alter model outputs. Full original
HotSpot provenance is still not independently established. No simulation,
calibration, retraining, Render setting, website deployment or GitHub push occurs.
