/* Run from temperature_project with the Python dependencies installed:
 *   npm install --prefix /tmp/temperature-ui-tests jsdom@30.1.3
 *   NODE_PATH=/tmp/temperature-ui-tests/node_modules PYTHON=python3 node tests/frontend_regressions.cjs
 * Uses DOM emulation with the real Flask backend, not visual browser rendering.
 */
const {JSDOM, VirtualConsole} = require('jsdom');
const assert = require('node:assert/strict');
const {spawn} = require('node:child_process');
const net = require('node:net');
const path = require('node:path');
const root = path.resolve(__dirname, '..');
const sleep = ms => new Promise(resolve => setTimeout(resolve, ms));
function gate() {
  let release;
  const promise = new Promise(resolve => { release = resolve; });
  return {promise, release};
}
async function waitUntil(predicate) {
  for (let i = 0; i < 250; i++) {
    try { if (predicate()) return; } catch (_) { /* scripts may still be loading */ }
    await sleep(20);
  }
  throw Error('Timed out waiting for frontend initialization');
}
async function freePort() {
  const socket = net.createServer();
  await new Promise(resolve => socket.listen(0, '127.0.0.1', resolve));
  const port = socket.address().port;
  await new Promise(resolve => socket.close(resolve));
  return port;
}
(async () => {
  const port = await freePort();
  const base = `http://127.0.0.1:${port}`;
  const server = spawn(process.env.PYTHON || 'python3', [path.join(root, 'app.py')], {
    cwd: root, env: {...process.env, PORT: String(port), PYTHONDONTWRITEBYTECODE: '1'},
    stdio: ['ignore', 'ignore', 'pipe'],
  });
  let serverError = '';
  server.stderr.on('data', data => { serverError += data; });
  let dom;
  const consoleErrors = [];
  let passed = 0;
  function check(label, predicate) { assert.ok(predicate, label); passed++; }
  try {
    let ready = false;
    for (let i = 0; i < 200; i++) {
      try { ready = (await fetch(base)).ok; if (ready) break; } catch (_) {}
      if (server.exitCode !== null) throw Error(serverError);
      await sleep(25);
    }
    assert.ok(ready, `Flask startup failed: ${serverError}`);
    const virtualConsole = new VirtualConsole();
    virtualConsole.on('jsdomError', error => consoleErrors.push(error.message));
    dom = await JSDOM.fromURL(base, {
      runScripts: 'dangerously', resources: 'usable', pretendToBeVisual: true, virtualConsole,
      beforeParse(window) { window.fetch = (url, options) => fetch(new URL(url, base), options); },
    });
    const w = dom.window, d = w.document;
    const state = expression => w.eval(expression);
    await waitUntil(() => state('AppState.lastPrediction !== null'));
    const originalFetch = w.fetch;
    const catalog = await (await fetch(`${base}/api/floorplans`)).json();
    check('23-floorplan catalog', catalog.floorplans.length === 23);
    for (const fp of catalog.floorplans) {
      await w.loadFloorplan(fp.full_name, {autoPredict: false});
      check(`${fp.full_name}: eight inputs and targets`,
        d.querySelectorAll('.power-input').length === 8 && d.querySelectorAll('.btn-predict-core').length === 8);
      await w.runPrediction('all');
      const all = state('AppState.lastPrediction');
      check(`${fp.full_name}: displayed maximum and eight rows`,
        all.floorplan === fp.full_name && d.querySelectorAll('#results-table-body tr').length === 8 &&
        d.querySelector('#kpi-temp-c').textContent === `${all.max_temp_c.toFixed(2)} °C`);
      for (let i = 0; i < 8; i++) {
        const core = `Core_${i}_L0`;
        await w.runPrediction(core);
        const single = state('AppState.lastPrediction');
        check(`${fp.full_name}: ${core} matches all-core output`, single.target === core &&
          Math.abs(single.selected_core.temp_c - all.predictions_by_core[core].temp_c) < 1e-6 &&
          d.querySelectorAll('#results-table-body tr').length === 1 &&
          d.querySelectorAll('#coupling-table-body tr').length === 8);
      }
    }
    // A delayed prediction must not render after a newer floorplan selection.
    await w.loadFloorplan('Style_1/fp_001', {autoPredict: false});
    let held = gate();
    w.fetch = async (url, options) => { if (url === '/api/predict') await held.promise; return originalFetch(url, options); };
    let pending = w.runPrediction('all');
    await w.loadFloorplan('Style_2/fp_001', {autoPredict: false});
    held.release(); await pending;
    check('outdated prediction response ignored', state('AppState.lastPrediction === null') &&
      state('AppState.currentFloorplan.full_name') === 'Style_2/fp_001');

    // A delayed floorplan response must not undo a newer selection.
    held = gate();
    w.fetch = async (url, options) => {
      if (url === '/api/floorplan/Style_1/fp_001') await held.promise;
      return originalFetch(url, options);
    };
    pending = w.loadFloorplan('Style_1/fp_001', {autoPredict: false});
    await w.loadFloorplan('Style_4/fp_010', {autoPredict: false});
    held.release(); await pending;
    check('outdated floorplan-loading response ignored',
      state('AppState.currentFloorplan.full_name') === 'Style_4/fp_010' &&
      d.querySelector('#hdr-floorplan').textContent === 'Style_4/fp_010');

    // Even on the same floorplan, only the latest target request may render.
    held = gate(); let calls = 0;
    w.fetch = async (url, options) => {
      if (url === '/api/predict' && ++calls === 1) await held.promise;
      return originalFetch(url, options);
    };
    pending = w.runPrediction('Core_0_L0');
    await w.runPrediction('Core_1_L0');
    held.release(); await pending;
    check('latest prediction target wins', state('AppState.lastPrediction.target') === 'Core_1_L0');

    held = gate();
    w.fetch = async (url, options) => { if (url === '/api/predict') await held.promise; return originalFetch(url, options); };
    pending = w.runPrediction('all');
    const input = d.querySelector('.power-input');
    input.value = String(Number(input.value) + 0.01);
    input.dispatchEvent(new w.Event('input', {bubbles: true}));
    held.release(); await pending;
    check('power edits invalidate in-flight prediction', state('AppState.lastPrediction === null && AppState.isOutdated'));

    // Outdated errors must not show on the new floorplan either.
    held = gate();
    w.fetch = async (url, options) => {
      if (url === '/api/predict') { await held.promise; throw Error('obsolete prediction failure'); }
      return originalFetch(url, options);
    };
    pending = w.runPrediction('all');
    await w.loadFloorplan('Style_2/fp_001', {autoPredict: false});
    held.release(); await pending;
    check('outdated prediction error ignored', d.querySelector('#validation-error-banner').classList.contains('hidden'));

    held = gate();
    w.fetch = async (url, options) => {
      if (url === '/api/floorplan/Style_1/fp_001') {
        await held.promise;
        return {ok: false, json: async () => ({error: 'obsolete floorplan failure'})};
      }
      return originalFetch(url, options);
    };
    pending = w.loadFloorplan('Style_1/fp_001', {autoPredict: false});
    await w.loadFloorplan('Style_4/fp_010', {autoPredict: false});
    held.release(); await pending;
    check('outdated floorplan error ignored', d.querySelector('#validation-error-banner').classList.contains('hidden') &&
      state('AppState.currentFloorplan.full_name') === 'Style_4/fp_010');

    held = gate(); calls = 0;
    w.fetch = async (url, options) => {
      if (url.startsWith('/api/floorplan/')) await held.promise;
      if (url === '/api/predict') calls++;
      return originalFetch(url, options);
    };
    pending = w.loadFloorplan('Style_1/fp_001', {autoPredict: false});
    await w.runPrediction('all');
    check('prediction blocked while floorplan loads', calls === 0);
    held.release(); await pending;
    w.fetch = originalFetch;
    await w.runPrediction('all');
    check('prediction resumes after loading', state('AppState.lastPrediction.floorplan') === 'Style_1/fp_001');
    check('no JavaScript runtime errors', consoleErrors.length === 0);
    console.log(`PASS: ${passed} frontend assertions, including both original response-order regressions.`);
  } finally {
    if (dom) dom.window.close();
    server.kill();
  }
})().catch(error => { console.error(error); process.exitCode = 1; });
