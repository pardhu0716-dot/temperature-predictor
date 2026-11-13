#!/usr/bin/env python3
"""Read-only build/startup checks for the 23 saved Stage A models."""
import math
from pathlib import Path

from app import PROJECT_ROOT, FLOORPLANS_ROOT, _load_floorplan_bundle
from predict import discover_floorplans


def check_runtime_assets():
    catalog = discover_floorplans(FLOORPLANS_ROOT)
    if len(catalog) != 23:
        raise RuntimeError(f'Expected 23 calibrated Stage A floorplans; found {len(catalog)}')
    for item in catalog:
        bundle = _load_floorplan_bundle(item['style'], item['floorplan'])
        names = bundle['matrix_core_order']
        matrix = bundle['thermal_matrix_k_per_w']
        if len(names) != 8 or len(set(names)) != 8 or len(matrix) != 8:
            raise RuntimeError(f'Invalid saved model: {item["full_name"]}')
        if any(len(row) != 8 or any(not math.isfinite(v) for v in row) for row in matrix):
            raise RuntimeError(f'Invalid matrix values: {item["full_name"]}')
        if bundle['num_big_cores'] != 2 or bundle['num_little_cores'] != 6:
            raise RuntimeError(f'Invalid core geometry: {item["full_name"]}')
    for relative in ['templates/index.html', 'static/css/style.css', 'static/js/app.js',
                     'stage_a_all_floorplans_summary.csv', 'stage_a_benchmark_per_floorplan.csv',
                     'stage_a_benchmark_overall_summary.csv']:
        if not (Path(PROJECT_ROOT) / relative).is_file():
            raise RuntimeError(f'Missing runtime asset: {relative}')
    return len(catalog)


if __name__ == '__main__':
    print(f'Ready: {check_runtime_assets()} saved Stage A 8x8 models and frontend assets; no HotSpot required.')
