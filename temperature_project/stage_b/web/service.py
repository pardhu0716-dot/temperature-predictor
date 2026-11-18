"""Inference adapter: immutable approved model, existing geometry/features/scaling."""
import hashlib
import json
from pathlib import Path
from time import perf_counter_ns
import joblib
import numpy as np
from stage_b.ml.data import CORE_IDS, Dataset, load_geometry_manifest, validate_geometry
from stage_b.ml.features import build_features

STAGE_B = Path(__file__).resolve().parents[1]
MODEL_DIR = STAGE_B / 'models/final_lasso'
MODEL_SHA = 'db1b49992c8a11b95c85a98be861b95502b00b0d3ec680709778236484d9ae80'
UNITS = {'geometry': 'mm', 'power': 'W', 'temperature': 'C'}


def numeric(value, label):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not np.isfinite(value):
        raise ValueError(f'{label} must be a finite JSON number')
    return float(value)


class Predictor:
    def __init__(self):
        path = MODEL_DIR / 'model.joblib'
        if hashlib.sha256(path.read_bytes()).hexdigest() != MODEL_SHA:
            raise RuntimeError('Approved model integrity check failed')
        self.bundle = joblib.load(path)  # Local, hash-pinned, trusted artifact only.
        self.features = json.loads((MODEL_DIR / 'features.json').read_text())
        scaling = json.loads((MODEL_DIR / 'scaling.json').read_text())
        if self.bundle.scaling != scaling or self.bundle.core_order != list(CORE_IDS):
            raise RuntimeError('Saved preprocessing/core order mismatch')
        if self.bundle.estimator.alpha != .001 or type(self.bundle.estimator).__name__ != 'Lasso':
            raise RuntimeError('Expected approved Lasso alpha=0.001')
        self.layouts = load_geometry_manifest(STAGE_B / 'dataset_v1/floorplan_manifest.json')
        self.manifest = json.loads((STAGE_B / 'dataset_v1/floorplan_manifest.json').read_text())
        self.info = dict(name='Lasso Regression', alpha=.001, ambient_c=45., serialized_bundle_bytes=path.stat().st_size,
                         feature_count=252, model_sha256=MODEL_SHA, units=UNITS,
                         historical_test=dict(mae_c=1.070281, rmse_c=1.339064, max_absolute_error_c=4.974435,
                                              median_latency_ms=.0632965),
                         limitations='Historical test metrics are not website guarantees. Custom geometries may be outside training coverage; only the supported 2D eight-core setup is accepted.')
        # Verify the feature definition itself before any online request.
        fid = next(iter(self.layouts));g,die = self.layouts[fid]
        self._features(g, die, np.where(g[:, 0] == 4, 15., 7.5))

    def _features(self, geometry, die, powers):
        data = Dataset(np.array(['inference']), np.array(['request']), geometry[None], powers[None],
                       np.zeros((1, 8)), die[None], '', {})
        X, definitions = build_features(data)
        digest = hashlib.sha256(json.dumps(definitions, sort_keys=True).encode()).hexdigest()
        if definitions != self.features['definitions'] or digest != self.bundle.feature_sha256:
            raise RuntimeError('Engineered feature representation differs from approved model')
        return X

    def floorplan(self, fid):
        if fid not in self.layouts:
            raise KeyError('Unknown Stage B floorplan')
        g, die = self.layouts[fid]
        return dict(id=fid, die=dict(width_mm=float(die[0]), height_mm=float(die[1])), units=UNITS,
                    cores=[dict(id=name, width_mm=float(r[0]), height_mm=float(r[1]), x_mm=float(r[2]),
                                y_mm=float(r[3]), power_w=15. if r[0] == 4 else 7.5,
                                core_type='Big' if r[0] == 4 else 'Little',
                                min_power_w=10. if r[0] == 4 else 5., max_power_w=20. if r[0] == 4 else 10.)
                           for name, r in zip(CORE_IDS, g)])

    def parse(self, payload):
        if not isinstance(payload, dict):
            raise ValueError('Prediction must be a JSON object')
        if payload.get('units') != UNITS:
            raise ValueError('Declare units exactly: geometry=mm, power=W, temperature=C')
        mode = payload.get('mode')
        if mode not in ('existing', 'custom'):
            raise ValueError('mode must be existing or custom')
        allowed = {'mode', 'units', 'floorplan_id', 'powers_w'} if mode == 'existing' else {'mode', 'units', 'die', 'cores'}
        if set(payload) - allowed:
            raise ValueError('Unsupported prediction fields; ambient/material/layer settings are fixed')
        if mode == 'existing':
            fid = payload.get('floorplan_id')
            if not isinstance(fid, str) or fid not in self.layouts:
                raise KeyError('Unknown Stage B floorplan')
            g, die = self.layouts[fid]
            source = payload.get('powers_w')
            if not isinstance(source, dict) or set(source) != set(CORE_IDS):
                raise ValueError('powers_w must map exactly Core_0_L0 through Core_7_L0')
            powers = np.array([numeric(source[n], f'{n} power') for n in CORE_IDS])
        else:
            fid = None
            d = payload.get('die')
            if not isinstance(d, dict):
                raise ValueError('Custom floorplans require explicit die width_mm and height_mm')
            die = np.array([numeric(d.get(k), k) for k in ('width_mm', 'height_mm')])
            # Bounded physical display/input range prevents overflow; this is not an accuracy claim.
            if np.any(die <= 0) or np.any(die > 1000):
                raise ValueError('Die dimensions must be >0 and <=1000 mm')
            rows = payload.get('cores')
            if not isinstance(rows, list) or len(rows) != 8 or any(not isinstance(r, dict) for r in rows):
                raise ValueError('Provide exactly eight core objects')
            names = [r.get('id') for r in rows]
            if any(not isinstance(n, str) for n in names) or len(set(names)) != 8 or set(names) != set(CORE_IDS):
                raise ValueError('Core identifiers must be unique Core_0_L0 through Core_7_L0')
            mapped = {r['id']: r for r in rows}
            g = np.array([[numeric(mapped[n].get(k), f'{n} {k}') for k in ('width_mm', 'height_mm', 'x_mm', 'y_mm')] for n in CORE_IDS])
            powers = np.array([numeric(mapped[n].get('power_w'), f'{n} power') for n in CORE_IDS])
        big = validate_geometry(g, die)  # Reuse original topology/bounds/overlap validation.
        if np.any(powers < np.where(big, 10., 5.)) or np.any(powers > np.where(big, 20., 10.)):
            raise ValueError('Supported powers: Big 10–20 W and Little 5–10 W')
        return fid, g, die, powers

    def predict(self, payload):
        fid, g, die, powers = self.parse(payload)
        start = perf_counter_ns()
        X = self._features(g, die, powers)
        feature_ns = perf_counter_ns() - start
        inference_start = perf_counter_ns()
        temperatures = self.bundle.predict(X)[0]
        inference_ns = perf_counter_ns() - inference_start
        hot = int(np.argmax(temperatures))
        return dict(floorplan_id=fid, mode=payload['mode'], units=UNITS, core_order=list(CORE_IDS),
                    temperatures_c={n: float(t) for n, t in zip(CORE_IDS, temperatures)},
                    minimum_c=float(min(temperatures)), maximum_c=float(max(temperatures)), hottest_core=CORE_IDS[hot],
                    latency_ms=inference_ns / 1e6, feature_engineering_ms=feature_ns / 1e6,
                    latency_boundary='Input scaling + Lasso prediction + inverse target scaling; excludes feature engineering and HTTP',
                    model=dict(name='Lasso Regression', alpha=.001, serialized_bundle_bytes=11197),
                    warning='Custom geometry accuracy is not guaranteed.' if fid is None else None)
