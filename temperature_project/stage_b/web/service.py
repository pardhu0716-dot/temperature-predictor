"""Inference adapter: immutable production v2 (default) & v1 (baseline) models, no retraining."""
import hashlib
import json
from pathlib import Path
from time import perf_counter_ns
import joblib
import numpy as np
from stage_b.ml.data import CORE_IDS, Dataset, load_geometry_manifest, validate_geometry
from stage_b.ml.features import build_features
from stage_b.ml.v2_model import V2_MODEL_DIR, load_v2_production_model

STAGE_B = Path(__file__).resolve().parents[1]
MODEL_DIR = STAGE_B / 'models/final_lasso'
MODEL_SHA = 'db1b49992c8a11b95c85a98be861b95502b00b0d3ec680709778236484d9ae80'

V2_MODEL_PATH = V2_MODEL_DIR / 'model_v2.npz'
V2_MODEL_SHA = 'edbae799105b57134ab99cfe3a68db2e4afd9c7aeb0ae78b3ec49476899c2175'
DEFAULT_MODEL_VERSION = 'v2'
SUPPORTED_MODEL_VERSIONS = ('v2', 'v1')

UNITS = {'geometry': 'mm', 'power': 'W', 'temperature': 'C'}


def numeric(value, label):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not np.isfinite(value):
        raise ValueError(f'{label} must be a finite JSON number')
    return float(value)


class Predictor:
    def __init__(self):
        # 1. Load & verify immutable Stage B v1 baseline model (Lasso alpha=0.001)
        v1_path = MODEL_DIR / 'model.joblib'
        if hashlib.sha256(v1_path.read_bytes()).hexdigest() != MODEL_SHA:
            raise RuntimeError('Approved Stage B v1 model integrity check failed')
        self.bundle = joblib.load(v1_path)  # Local, hash-pinned, trusted artifact only.
        self.features = json.loads((MODEL_DIR / 'features.json').read_text())
        scaling = json.loads((MODEL_DIR / 'scaling.json').read_text())
        if self.bundle.scaling != scaling or self.bundle.core_order != list(CORE_IDS):
            raise RuntimeError('Saved v1 preprocessing/core order mismatch')
        if self.bundle.estimator.alpha != .001 or type(self.bundle.estimator).__name__ != 'Lasso':
            raise RuntimeError('Expected approved v1 Lasso alpha=0.001')

        # 2. Load & verify portable pickle-free Stage B v2 production model (Ridge alpha=0.01)
        self.v2_model = load_v2_production_model(V2_MODEL_DIR)
        if self.v2_model.artifact_sha256 != V2_MODEL_SHA:
            raise RuntimeError('Stage B v2 production artifact SHA-256 mismatch')
        if self.v2_model.metadata['hyperparameters']['alpha'] != .01 or len(self.v2_model.feature_names) != 63:
            raise RuntimeError('Unexpected Stage B v2 production metadata')

        self.layouts = load_geometry_manifest(STAGE_B / 'dataset_v1/floorplan_manifest.json')
        self.manifest = json.loads((STAGE_B / 'dataset_v1/floorplan_manifest.json').read_text())

        v1_info = dict(
            version='v1',
            name='Lasso Regression (Stage B v1)',
            alpha=.001,
            ambient_c=45.,
            serialized_bundle_bytes=v1_path.stat().st_size,
            feature_count=252,
            model_sha256=MODEL_SHA,
            artifact_format='joblib',
            units=UNITS,
            historical_test=dict(mae_c=1.070281, rmse_c=1.339064, max_absolute_error_c=4.974435,
                                 hottest_core_acc_pct=56.67, median_latency_ms=.0632965),
            limitations='Historical evaluation metrics are not website guarantees. Custom geometries may be outside training coverage; only the supported 2D eight-core setup is accepted.',
        )
        v2_info = dict(
            version='v2',
            name='Ridge Regression (Stage B v2 · Shared Physics)',
            alpha=.01,
            ambient_c=45.,
            serialized_bundle_bytes=self.v2_model.artifact_bytes,
            feature_count=len(self.v2_model.feature_names),
            model_sha256=self.v2_model.artifact_sha256,
            artifact_format='numpy_npz_no_pickle',
            units=UNITS,
            historical_test=dict(mae_c=0.350588, rmse_c=0.445882, max_absolute_error_c=1.512353,
                                 hottest_core_acc_pct=84.17, external_23fp_mae_c=0.460754),
            limitations='Historical evaluation metrics (including previously inspected held-out and external sets) are not website guarantees. Custom geometries may be outside training coverage; only the supported 2D eight-core setup is accepted.',
        )
        self.models_info = {'v2': v2_info, 'v1': v1_info}
        self.info = dict(v2_info, default_model_version=DEFAULT_MODEL_VERSION,
                         available_versions=list(SUPPORTED_MODEL_VERSIONS), models=self.models_info)

        # Verify both feature pipelines before any online request.
        fid = next(iter(self.layouts))
        g, die = self.layouts[fid]
        sample_p = np.where(g[:, 0] == 4, 15., 7.5)
        self._features_v1(g, die, sample_p)
        self.v2_model.build_features(g, sample_p, die)

    def _features_v1(self, geometry, die, powers):
        data = Dataset(np.array(['inference']), np.array(['request']), geometry[None], powers[None],
                       np.zeros((1, 8)), die[None], '', {})
        X, definitions = build_features(data)
        digest = hashlib.sha256(json.dumps(definitions, sort_keys=True).encode()).hexdigest()
        if definitions != self.features['definitions'] or digest != self.bundle.feature_sha256:
            raise RuntimeError('Engineered v1 feature representation differs from approved model')
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
        model_version = payload.get('model_version', DEFAULT_MODEL_VERSION)
        if not isinstance(model_version, str) or model_version not in SUPPORTED_MODEL_VERSIONS:
            raise ValueError(f'model_version must be one of {list(SUPPORTED_MODEL_VERSIONS)}')
        allowed = (
            {'mode', 'units', 'floorplan_id', 'powers_w', 'model_version'}
            if mode == 'existing'
            else {'mode', 'units', 'die', 'cores', 'model_version'}
        )
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
        big = validate_geometry(g, die)
        if np.any(powers < np.where(big, 10., 5.)) or np.any(powers > np.where(big, 20., 10.)):
            raise ValueError('Supported powers: Big 10–20 W and Little 5–10 W')
        return fid, g, die, powers, model_version

    def predict(self, payload):
        fid, g, die, powers, model_version = self.parse(payload)
        if model_version == 'v1':
            start = perf_counter_ns()
            X = self._features_v1(g, die, powers)
            feature_ns = perf_counter_ns() - start
            inference_start = perf_counter_ns()
            temperatures = self.bundle.predict(X)[0]
            inference_ns = perf_counter_ns() - inference_start
            model_meta = dict(
                version='v1', name='Lasso Regression (Stage B v1)', alpha=.001,
                feature_count=252, serialized_bundle_bytes=11197,
            )
            boundary = 'Input scaling + Lasso prediction + inverse target scaling; excludes feature engineering and HTTP'
        else:
            start = perf_counter_ns()
            X = self.v2_model.build_features(g, powers, die)
            feature_ns = perf_counter_ns() - start
            inference_start = perf_counter_ns()
            temperatures = self.v2_model.predict_from_features(X)[0]
            inference_ns = perf_counter_ns() - inference_start
            model_meta = dict(
                version='v2', name='Ridge Regression (Stage B v2 · Shared Physics)', alpha=.01,
                feature_count=len(self.v2_model.feature_names),
                serialized_bundle_bytes=self.v2_model.artifact_bytes,
            )
            boundary = 'Input scaling + shared per-core Ridge prediction + inverse target scaling; excludes feature engineering and HTTP'

        hot = int(np.argmax(temperatures))
        return dict(floorplan_id=fid, mode=payload['mode'], model_version=model_version, units=UNITS,
                    core_order=list(CORE_IDS),
                    temperatures_c={n: float(t) for n, t in zip(CORE_IDS, temperatures)},
                    minimum_c=float(min(temperatures)), maximum_c=float(max(temperatures)), hottest_core=CORE_IDS[hot],
                    latency_ms=inference_ns / 1e6, feature_engineering_ms=feature_ns / 1e6,
                    latency_boundary=boundary, model=model_meta,
                    warning='Custom geometry accuracy is not guaranteed.' if fid is None else None)
