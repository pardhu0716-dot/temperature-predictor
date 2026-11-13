#!/usr/bin/env python3
"""Check local or deployed HTTP predictions against the unchanged CLI and matrices.

Run from the same repository revision deployed to the target URL. This is
read-only: no calibration, data writes, or HotSpot invocation.
"""
import argparse
import json
from pathlib import Path
from urllib.error import HTTPError
from urllib.request import Request, urlopen

from predict import discover_floorplans, locate_and_load_thermal_matrix, predict_temperatures

PROJECT = Path(__file__).resolve().parent


def verify(base_url):
    checks = 0
    prediction_requests = 0
    def check(condition, message):
        nonlocal checks
        if not condition: raise AssertionError(message)
        checks += 1
    def request(path, data=None, status=200):
        payload = None if data is None else json.dumps(data).encode()
        req = Request(base_url.rstrip('/') + path, data=payload,
                      headers={'Content-Type': 'application/json'} if payload is not None else {})
        try: response = urlopen(req, timeout=120)
        except HTTPError as exc: response = exc
        with response:
            body = response.read()
            check(response.status == status, f'{path}: expected HTTP {status}, received {response.status}: {body[:200]}')
            return body, response.headers
    def api(path, data=None, status=200):
        return json.loads(request(path, data, status)[0])

    page, _ = request('/')
    check(b'Lightweight Temperature Predictor' in page and b'STAGE A' in page, 'Stage A homepage missing')
    for path, content_type in [('/static/css/style.css', 'text/css'), ('/static/js/app.js', 'javascript')]:
        data, headers = request(path)
        check(content_type in headers.get('Content-Type', '') and len(data) > 1000, f'Broken asset: {path}')
    check(api('/healthz') == dict(status='ok', stage='A', calibrated_floorplans=23), 'Health check mismatch')
    local = discover_floorplans(str(PROJECT/'dataset/Sample_dataset/2D_floorplans'))
    catalog = api('/api/floorplans')
    check(len(local) == catalog['total_floorplans'] == 23, 'Expected exactly 23 Stage A floorplans')
    check({f['full_name'] for f in local} == {f['full_name'] for f in catalog['floorplans']}, 'Catalog differs from local')
    for floor in local:
        full = floor['full_name']
        bundle = api('/api/floorplan/' + full)
        _, names, matrix, _, _ = locate_and_load_thermal_matrix(floor['path'], floor['floorplan'])
        check(bundle['matrix_core_order'] == names, f'{full}: matrix ordering differs')
        core_map = {c['name']: c for c in bundle['cores']}
        for field in ['default_power_w', 'midpoint_power_w', 'min_power_w', 'max_power_w']:
            powers = [core_map[name][field] for name in names]
            _, expected_k, expected_c, _ = predict_temperatures(matrix, powers)
            result = api('/api/predict', dict(floorplan=full, powers=powers))
            prediction_requests += 1
            for i, name in enumerate(names):
                actual = result['predictions_by_core'][name]
                check(abs(actual['temp_c']-expected_c[i]) <= 1e-6, f'{full}/{name}: Celsius mismatch')
                check(abs(actual['temp_k']-expected_k[i]) <= 1e-6, f'{full}/{name}: Kelvin mismatch')
            check(abs(result['max_temp_c']-max(expected_c)) <= 1e-6, f'{full}: maximum mismatch')
            if field == 'default_power_w':
                for i, name in enumerate(names):
                    single = api('/api/predict', dict(floorplan=full, powers_by_core=dict(zip(names, powers)), target=name))
                    prediction_requests += 1
                    check(abs(single['selected_core']['temp_c']-expected_c[i]) <= 1e-6, f'{full}/{name}: individual mismatch')
                    check(len(single['selected_core']['coupling_contributions']) == 8, 'Missing coupling contributions')
        check(api('/api/benchmark/' + full)['available'], f'{full}: missing saved benchmarks')
    check(api('/api/benchmark/overall')['total_floorplans'] == 23, 'Missing overall benchmark')
    api('/api/predict', {}, 400)
    api('/api/floorplan/Style_99/fp_001', status=404)
    api('/api/predict', dict(floorplan='Style_1/fp_001', powers=[-1]*8), 400)
    print(json.dumps(dict(status='passed', base_url=base_url, floorplans=23,
                          prediction_requests=prediction_requests, assertions=checks,
                          comparison='unchanged predict.py + original saved matrices; tolerance 1e-6 K/C'), indent=2))


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--url', required=True, help='Local or live origin, e.g. https://YOUR-SERVICE.onrender.com')
    verify(p.parse_args().url)
