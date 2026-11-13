"""Production entry point; prediction logic remains in the original app.py."""
from flask import jsonify

from app import app
from deployment_check import check_runtime_assets

# Fail the build/worker startup rather than serve a dashboard with missing models.
MODEL_COUNT = check_runtime_assets()
app.config.update(DEBUG=False, TESTING=False, MAX_CONTENT_LENGTH=16 * 1024)


@app.get('/healthz')
def healthz():
    return jsonify(status='ok', stage='A', calibrated_floorplans=MODEL_COUNT)


@app.after_request
def production_headers(response):
    response.headers['X-Content-Type-Options'] = 'nosniff'
    response.headers['Referrer-Policy'] = 'strict-origin-when-cross-origin'
    return response
