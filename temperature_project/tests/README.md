# Saved-model and frontend regression tests

Run from `temperature_project/` after installing `requirements.txt`.

```bash
PYTHONDONTWRITEBYTECODE=1 python3 -m unittest discover -s tests -v
```

The five Python tests verify all 23 saved models, matrix-only readiness, missing/invalid model failures, rejection of overwrite arguments, and byte-for-byte preservation of dataset files and saved metric CSVs. They prohibit calibration and subprocess calls during batch verification.

For frontend tests, use Node.js compatible with jsdom 30.1.3 and install test-only dependencies outside the project:

```bash
npm install --prefix /tmp/temperature-ui-tests jsdom@30.1.3
NODE_PATH=/tmp/temperature-ui-tests/node_modules PYTHON=python3 node tests/frontend_regressions.cjs
```

This starts a temporary Flask server, exercises all 23 floorplans, and uses controlled response delays to test outdated prediction/floorplan responses, overlapping target requests, power edits during prediction, outdated errors, and prediction while a floorplan is loading. The server is stopped afterward. Python bytecode writes are disabled.

The frontend suite uses JSDOM with real HTTP API calls. It does not test visual browser layout. No HotSpot executable or raw training simulation archive is required.

`python3 run_all_stage_a.py` now verifies existing calibrated models without changing matrices, data, or summaries. Missing/invalid models return exit code 1; the runner does not attempt calibration. `--force-overwrite` is rejected. Existing `stage_a_pipeline.py` remains the separate calibration entry point and is not called by these tests or the batch verifier.
