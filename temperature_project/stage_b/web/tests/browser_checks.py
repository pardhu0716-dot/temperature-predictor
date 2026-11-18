"""Local Chromium user-flow checks. Requires a separately running combined_app."""
import argparse,json
from pathlib import Path
from playwright.sync_api import sync_playwright

def run(base,out):
    out.mkdir(parents=True,exist_ok=True);evidence={};errors=[];failed=[]
    with sync_playwright() as p:
        browser=p.chromium.launch(headless=True)
        for name,width,height in [('desktop',1440,1100),('mobile',390,844)]:
            page=browser.new_page(viewport={'width':width,'height':height})
            page.on('pageerror',lambda e:errors.append(str(e)))
            page.on('response',lambda r:failed.append(r.url) if r.status>=400 and '/stage-b/assets/' in r.url else None)
            assert page.goto(base+'/stage-b',wait_until='networkidle').status==200
            page.wait_for_function("document.querySelector('#b-inputs').rows.length===8 && !document.querySelector('#b-predict').disabled")
            assert page.locator('#b-floorplan option').count()==77
            assert page.locator('#b-inputs input[data-field="power_w"]').count()==8
            # All eight power controls work and invalidate stale results.
            for i in range(8):
                item=page.locator('#b-inputs input[data-field="power_w"]').nth(i)
                item.fill(str(float(item.input_value())+.1))
            with page.expect_response('**/api/stage-b/predict') as response:page.click('#b-predict')
            result=response.value.json();page.wait_for_function("document.querySelector('#b-results').rows.length===8")
            assert not page.locator('#b-error').is_visible()
            for core,temp in result['temperatures_c'].items():
                assert float(page.locator(f'#b-heatmap rect[data-core="{core}"]').get_attribute('data-temperature-c'))==temp
                assert float(page.locator(f'#b-bars rect[data-core="{core}"]').get_attribute('data-temperature-c'))==temp
            colors=page.locator('#b-heatmap rect[data-core]').evaluate_all('(els)=>new Set(els.map(e=>e.getAttribute("fill"))).size')
            assert colors>1
            existing=result
            page.screenshot(path=str(out/f'{name}-existing.png'),full_page=True)
            # Switching floorplans changes real die/coordinates and clears results.
            oldx=page.locator('#b-heatmap rect[data-core="Core_0_L0"]').get_attribute('x')
            page.select_option('#b-floorplan','b_fp_007');page.wait_for_function("document.querySelector('#b-floorplan-note').textContent.includes('b_fp_007') && !document.querySelector('#b-predict').disabled")
            assert page.locator('#b-results tr').count()==0
            assert page.locator('#b-heatmap rect[data-core="Core_0_L0"]').get_attribute('x')!=oldx
            page.select_option('#b-floorplan','b_fp_006');page.wait_for_function("document.querySelector('#b-floorplan-note').textContent.includes('b_fp_006') && !document.querySelector('#b-predict').disabled")
            page.select_option('#b-mode','custom')
            assert page.locator('#b-die-width').is_editable()
            dw=float(page.locator('#b-die-width').input_value());page.locator('#b-die-width').fill(str(dw+1))
            xinput=page.locator('#b-inputs input[data-field="x_mm"]').nth(7);newx=float(xinput.input_value())+.05;xinput.fill(str(newx))
            assert float(page.locator('#b-heatmap rect[data-core="Core_7_L0"]').get_attribute('x'))==newx
            with page.expect_response('**/api/stage-b/predict') as response:page.click('#b-predict')
            custom=response.value.json();page.wait_for_function("document.querySelector('#b-results').rows.length===8")
            assert custom['mode']=='custom' and custom['warning']
            assert custom['temperatures_c']!=existing['temperatures_c']
            page.screenshot(path=str(out/f'{name}-custom.png'),full_page=True)
            # Reset restores original geometry without leaving custom mode.
            page.click('#b-reset');assert float(page.locator('#b-die-width').input_value())==dw
            assert page.locator('#b-mode').input_value()=='custom'
            # Invalid overlap is caught in the preview and cannot produce temperatures.
            for field in ('x_mm','y_mm'):
                value=page.locator(f'#b-inputs input[data-field="{field}"]').nth(1).input_value()
                page.locator(f'#b-inputs input[data-field="{field}"]').nth(0).fill(value)
            page.click('#b-predict');assert 'overlap' in page.locator('#b-error').inner_text()
            assert page.locator('#b-results tr').count()==0
            page.click('#b-reset')
            # Server errors clear all previous values, including heatmap coloring.
            page.click('#b-predict');page.wait_for_function("document.querySelector('#b-results').rows.length===8")
            page.route('**/api/stage-b/predict',lambda route:route.fulfill(status=400,content_type='application/json',body=json.dumps({'error':'Controlled invalid-input response'})))
            page.click('#b-predict');page.wait_for_function("document.querySelector('#b-error').textContent.includes('Controlled')")
            assert page.locator('#b-heatmap rect[data-temperature-c]').count()==0
            page.unroute('**/api/stage-b/predict')
            # Delayed old prediction must not overwrite a newer edited state.
            held={}
            def hold(route):held['route']=route
            page.route('**/api/stage-b/predict',hold)
            page.click('#b-predict');page.wait_for_function("document.querySelector('#b-status').textContent==='Predicting…'")
            assert 'route' in held
            payload=held['route'].request.post_data_json
            old_result=page.request.post(base+'/api/stage-b/predict',data=payload).json()
            power=page.locator('#b-inputs input[data-field="power_w"]').first;power.fill(str(float(power.input_value())+.1))
            held['route'].fulfill(status=200,content_type='application/json',body=json.dumps(old_result));page.unroute('**/api/stage-b/predict')
            page.wait_for_function("!document.querySelector('#b-predict').disabled")
            assert page.locator('#b-results tr').count()==0
            assert page.locator('#b-heatmap rect[data-temperature-c]').count()==0
            assert page.evaluate('document.documentElement.scrollWidth<=window.innerWidth'), 'Stage B page overflows viewport'
            evidence[name]=dict(floorplans=77,power_inputs=8,existing_result=existing,custom_result=custom,heatmap_values_match_api=True,geometry_preview_updates=True,reset_and_overlap_checks=True,api_error_clears_results=True,stale_response_ignored=True,no_page_horizontal_overflow=True)
            page.close()
        # Same application navigates to and runs the original Stage A UI.
        page=browser.new_page(viewport={'width':1440,'height':1100});page.on('pageerror',lambda e:errors.append(str(e)))
        page.goto(base+'/',wait_until='networkidle');page.wait_for_function("typeof AppState !== 'undefined' && AppState.lastPrediction!==null")
        assert len(page.evaluate('AppState.catalog'))==23
        with page.expect_response('**/api/predict'):page.click('#btn-predict-all')
        button=page.locator('#individual-buttons-container button').first
        with page.expect_response('**/api/predict'):button.click()
        page.screenshot(path=str(out/'stage-a-unchanged.png'),full_page=True)
        page.get_by_role('link',name='Stage B · Geometry-aware predictor').click()
        page.wait_for_selector('#b-inputs tr')
        assert page.url==base+'/stage-b'
        evidence['stage_a']=dict(floorplans=23,all_core_prediction=True,individual_core_prediction=True,shared_navigation=True)
        browser.close()
    assert not errors,errors
    assert not failed,failed
    evidence['javascript_errors']=errors;evidence['failed_stage_b_assets']=failed
    (out/'browser_results.json').write_text(json.dumps(evidence,indent=2));print(json.dumps({k:'PASS' for k in evidence},indent=2))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--url',default='http://127.0.0.1:5000');p.add_argument('--out',type=Path,required=True);a=p.parse_args();run(a.url.rstrip('/'),a.out)
