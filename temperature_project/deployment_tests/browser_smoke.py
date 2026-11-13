#!/usr/bin/env python3
"""Optional Chromium checks: pip install playwright; playwright install chromium."""
import argparse
import json
from pathlib import Path
from playwright.sync_api import sync_playwright


def verify(url, screenshots=None):
    errors = []
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page(viewport={'width': 1440, 'height': 1000})
        page.on('pageerror', lambda exc: errors.append(str(exc)))
        response = page.goto(url, wait_until='networkidle', timeout=120000)
        assert response.status == 200
        page.wait_for_function('AppState.lastPrediction !== null')
        assert page.evaluate('document.documentElement.scrollWidth <= innerWidth')
        catalog = page.evaluate('AppState.catalog')
        assert len(catalog) == 23
        for floor in catalog:
            if page.locator('#style-select').input_value() != floor['style']:
                page.locator('#style-select').select_option(floor['style'])
                page.wait_for_function('expected => !AppState.loadingFloorplan && AppState.currentStyle === expected', arg=floor['style'])
            if page.locator('#floorplan-select').input_value() != floor['floorplan']:
                page.locator('#floorplan-select').select_option(floor['floorplan'])
            page.wait_for_function('expected => AppState.currentFloorplan.full_name === expected && !AppState.loadingFloorplan', arg=floor['full_name'])
            assert page.locator('.power-input').count() == 8
            page.locator('#btn-preset-mid').click()
            with page.expect_response(lambda r: r.url.endswith('/api/predict') and r.request.method == 'POST') as prediction:
                page.locator('#btn-predict-all').click()
            assert prediction.value.status == 200
            page.wait_for_function('AppState.lastPrediction !== null && AppState.lastPrediction.mode === "all"')
            assert page.locator('#results-table-body tr').count() == 8
            assert page.locator('#floorplan-svg rect').count() >= 8
            assert page.locator('#core-temp-chart-svg').inner_html().strip()
            for name in page.evaluate('AppState.currentFloorplan.matrix_core_order'):
                page.locator('#btn-predict-' + name).click()
                page.wait_for_function('expected => AppState.lastPrediction !== null && AppState.lastPrediction.target === expected', arg=name)
                assert page.locator('#results-table-body tr').count() == 1
                assert page.locator('#coupling-table-body tr').count() == 8
        field = page.locator('.power-input').first
        field.fill('-1')
        assert page.locator('#stale-warning-banner').is_visible()
        page.locator('#btn-predict-all').click()
        assert page.locator('#validation-error-banner').is_visible()
        page.locator('#btn-preset-mid').click()
        # Enter a valid nonpreset value to exercise manual input and recalculation.
        minimum = float(field.get_attribute('min'))
        field.fill(str(minimum + 0.25))
        page.locator('#btn-predict-all').click()
        page.wait_for_function('AppState.lastPrediction !== null && AppState.lastPrediction.mode === "all"')
        assert page.locator('#results-table-body tr').count() == 8
        if screenshots:
            screenshots.mkdir(parents=True, exist_ok=True)
            page.screenshot(path=str(screenshots/'stage-a-desktop.png'), full_page=True)
        page.set_viewport_size({'width': 390, 'height': 844})
        assert page.locator('#style-select').is_visible()
        page.locator('#btn-predict-all').click()
        assert page.locator('#floorplan-svg').is_visible()
        assert page.evaluate('document.documentElement.scrollWidth <= innerWidth')
        if screenshots:
            page.screenshot(path=str(screenshots/'stage-a-mobile.png'), full_page=True)
        for width in [320, 700, 960]:
            page.set_viewport_size({'width': width, 'height': 900})
            assert page.evaluate('document.documentElement.scrollWidth <= innerWidth'), f'Page overflow at {width}px'
        assert not errors, errors
        print(json.dumps(dict(status='passed', url=url, floorplans=23, single_core_buttons=184,
                              viewport_widths=[1440, 390, 320, 700, 960], javascript_errors=errors), indent=2))
        browser.close()

if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--url', required=True)
    p.add_argument('--screenshots', type=Path)
    args = p.parse_args()
    verify(args.url, args.screenshots)
