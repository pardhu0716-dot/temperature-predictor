"""Future external benchmark gate. No Stage A prediction/matrix is a ground-truth label."""
import numpy as np
from .data import CORE_IDS,load_csv
from .features import build_features
from .evaluation import metrics

REQUIRED=('hotspot_config_sha256','material_layers','ambient_c','geometry_unit','temperature_unit','power_unit','core_order','label_source','feature_sha256')


def check_compatibility(training,external):
    errors=[]
    for key in REQUIRED:
        if key not in training or training[key] in (None,'',[]):errors.append(f'Training contract missing {key}')
        if key not in external or external[key] in (None,'',[]):errors.append(f'External contract missing {key}')
    if errors:raise ValueError('; '.join(errors))
    import re
    for contract in (training,external):
        if not isinstance(contract['hotspot_config_sha256'],str) or not re.fullmatch(r'[0-9a-f]{64}',contract['hotspot_config_sha256']):raise ValueError('Expected actual SHA-256 of HotSpot config')
        if not isinstance(contract['material_layers'],list) or not contract['material_layers']:raise ValueError('Explicit material-layer metadata required')
    if training['label_source']!='HotSpot' or external['label_source']!='HotSpot':raise ValueError('Only independently obtained HotSpot labels qualify')
    if training['core_order']!=list(CORE_IDS) or external['core_order']!=list(CORE_IDS):raise ValueError('Canonical core order required')
    for key in REQUIRED:
        if training[key]!=external[key]:raise ValueError(f'External compatibility mismatch: {key}')
    return dict(compatible=True,checks=list(REQUIRED),note='Contracts assert provenance; retain original simulator inputs/outputs for audit')


def evaluate_external(bundle,csv_path,schema,geometry_manifest,training_contract,external_contract):
    """Future API only; not called by train or evaluate-test and no CLI auto-evaluation."""
    if bundle.simulation_contract is None or bundle.simulation_contract!=training_contract:raise ValueError('Training artifact lacks the same verified simulation contract')
    check_compatibility(training_contract,external_contract)
    if training_contract['feature_sha256']!=bundle.feature_sha256:raise ValueError('Wrong feature representation')
    data=load_csv(csv_path,schema,geometry_manifest,expected_floorplans=None,samples_per_floorplan=None,external=True)
    if data.schema['ambient_c']!=external_contract['ambient_c']:raise ValueError('External CSV ambient differs from verified simulation contract')
    if any(str(fid).startswith('b_fp_') for fid in data.floorplan_ids):raise ValueError('External benchmark must not contain Stage B training floorplans')
    X,_=build_features(data)
    return metrics(data.temperatures_c,bundle.predict(X))
