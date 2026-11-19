"""Regression for pending geometry loads; preserve user edits by locking inputs."""
import argparse,json
from pathlib import Path
from playwright.sync_api import sync_playwright

def check(url,out):
 with sync_playwright() as p:
  b=p.chromium.launch(headless=True);page=b.new_page(viewport={'width':1440,'height':1050});errors=[];page.on('pageerror',lambda e:errors.append(str(e)))
  page.goto(url+'/stage-b',wait_until='networkidle');page.wait_for_function("document.querySelector('#b-inputs').rows.length===8 && !document.querySelector('#b-predict').disabled")
  held={}
  def hold(route):held['route']=route
  page.route('**/api/stage-b/floorplans/b_fp_007',hold);page.select_option('#b-floorplan','b_fp_007');page.wait_for_function("document.querySelector('#b-predict').disabled")
  assert page.locator('#b-mode').is_disabled(),'Mode change can race a pending geometry response'
  assert page.locator('#b-reset').is_disabled(),'Reset can race a pending geometry response'
  assert not page.locator('#b-inputs input[data-field="power_w"]').first.is_editable()
  data=page.request.get(url+'/api/stage-b/floorplans/b_fp_007').json();held['route'].fulfill(status=200,content_type='application/json',body=json.dumps(data));page.unroute('**/api/stage-b/floorplans/b_fp_007')
  page.wait_for_function("!document.querySelector('#b-predict').disabled")
  assert page.locator('#b-mode').is_enabled();assert page.locator('#b-reset').is_enabled()
  for ver in ('v1','v2'):
   page.select_option('#b-model',ver)
   with page.expect_response('**/api/stage-b/predict') as response:page.click('#b-predict')
   r=response.value.json();page.wait_for_function("document.querySelector('#b-results').rows.length===8");assert r['model_version']==ver
   for n,t in r['temperatures_c'].items():assert float(page.locator(f'#b-heatmap rect[data-core="{n}"]').get_attribute('data-temperature-c'))==t
  assert not errors
  out.mkdir(parents=True,exist_ok=True);page.screenshot(path=str(out/'stage-b-audit.png'),full_page=True);(out/'race_checks.json').write_text(json.dumps(dict(pending_geometry_controls_locked=True,model_switches_verified=['v1','v2'],javascript_errors=errors),indent=2));b.close()
 print('Geometry-load race and version/heatmap regressions: PASS')
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--url',required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args();check(a.url.rstrip('/'),a.out)
