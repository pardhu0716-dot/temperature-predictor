'use strict';
(() => {
  const $ = id => document.getElementById(id);
  const IDS = Array.from({length: 8}, (_, i) => `Core_${i}_L0`);
  const units = {geometry: 'mm', power: 'W', temperature: 'C'};
  const state = {floor: null, revision: 0, request: 0, loading: false, result: null, loadToken: 0};
  const ns = 'http://www.w3.org/2000/svg';
  function svg(tag, attrs, text) {
    const el = document.createElementNS(ns, tag);
    Object.entries(attrs).forEach(([k, v]) => el.setAttribute(k, String(v)));
    if (text !== undefined) el.textContent = text;
    return el;
  }
  function error(message) { $('b-error').textContent = message || ''; $('b-error').classList.toggle('hidden', !message); }
  function clearResults() {
    state.result = null; $('b-results').replaceChildren(); $('b-bars').replaceChildren();
    ['b-max', 'b-min', 'b-hot', 'b-latency'].forEach(id => $(id).textContent = '—');
    $('b-low').textContent = 'No prediction'; $('b-high').textContent = '';
    try { if ($('b-inputs').rows.length === 8) preview(snapshot(), null); else $('b-heatmap').replaceChildren(); } catch (_) { $('b-heatmap').replaceChildren(); }
  }
  function number(input, label) {
    if (input.value.trim() === '' || !Number.isFinite(Number(input.value))) throw new Error(`${label} must be finite.`);
    return Number(input.value);
  }
  function snapshot() {
    const die = {width_mm: number($('b-die-width'), 'Die width'), height_mm: number($('b-die-height'), 'Die height')};
    const cores = [...$('b-inputs').rows].map(row => {
      const r = {id: row.querySelector('select').value};
      ['width_mm', 'height_mm', 'x_mm', 'y_mm', 'power_w'].forEach(k => r[k] = number(row.querySelector(`[data-field="${k}"]`), `${r.id} ${k}`));
      return r;
    });
    return {die, cores};
  }
  function validate(s) {
    if (!(s.die.width_mm > 0 && s.die.height_mm > 0 && s.die.width_mm <= 1000 && s.die.height_mm <= 1000)) throw new Error('Die dimensions must be >0 and <=1000 mm.');
    if (new Set(s.cores.map(c => c.id)).size !== 8) throw new Error('Each core ID must appear exactly once.');
    let bigCount = 0;
    s.cores.forEach((c, i) => {
      const big = c.width_mm === 4 && c.height_mm === 4;
      if (!big && !(c.width_mm === 2 && c.height_mm === 2)) throw new Error('Use 4×4 mm Big or 2×2 mm Little cores.');
      bigCount += big ? 1 : 0;
      if (c.x_mm < 0 || c.y_mm < 0 || c.x_mm + c.width_mm > s.die.width_mm + 1e-8 || c.y_mm + c.height_mm > s.die.height_mm + 1e-8) throw new Error(`${c.id} extends outside the die.`);
      if (c.power_w < (big ? 10 : 5) || c.power_w > (big ? 20 : 10)) throw new Error(`${c.id}: Big powers must be 10–20 W and Little powers 5–10 W.`);
      s.cores.slice(0, i).forEach(b => {
        if (Math.min(c.x_mm + c.width_mm, b.x_mm + b.width_mm) - Math.max(c.x_mm, b.x_mm) > 1e-8 && Math.min(c.y_mm + c.height_mm, b.y_mm + b.height_mm) - Math.max(c.y_mm, b.y_mm) > 1e-8) throw new Error(`${c.id} overlaps ${b.id}.`);
      });
    });
    if (bigCount !== 2) throw new Error('Exactly two Big and six Little cores are required.');
  }
  function preview(s, result) {
    const el = $('b-heatmap'); el.replaceChildren();
    const w = s.die.width_mm, h = s.die.height_mm;
    if (!(w > 0 && h > 0 && w <= 1000 && h <= 1000)) return;
    const pad = Math.max(w, h) * .025;
    el.setAttribute('viewBox', `${-pad} ${-pad} ${w+2*pad} ${h+2*pad}`);
    el.append(svg('rect', {x: 0, y: 0, width: w, height: h, fill: '#172235', stroke: '#64748b', 'stroke-width': Math.max(w, h)*.003}));
    const lo = result ? result.minimum_c : 0, hi = result ? result.maximum_c : 1;
    s.cores.forEach(c => {
      const t = result ? result.temperatures_c[c.id] : null;
      const ratio = t === null ? 0 : (hi === lo ? .5 : (t-lo)/(hi-lo));
      const fill = t === null ? '#245476' : `hsl(${220*(1-ratio)} 75% 48%)`;
      const y = h-c.y_mm-c.height_mm;
      const r = svg('rect', {x:c.x_mm, y, width:c.width_mm, height:c.height_mm, fill, stroke:'#e2e8f0','stroke-width':Math.max(w,h)*.004,'data-core':c.id});
      if (t !== null) r.setAttribute('data-temperature-c', t);
      r.append(svg('title', {}, `${c.id}: ${c.width_mm}×${c.height_mm} mm at (${c.x_mm}, ${c.y_mm}); ${t === null ? 'no prediction' : t.toFixed(3)+' °C'}`)); el.append(r);
      const fs = Math.min(c.width_mm, c.height_mm)*.21;
      el.append(svg('text', {x:c.x_mm+c.width_mm/2,y:y+c.height_mm*.43,'text-anchor':'middle',fill:'white','font-size':fs,'font-weight':700}, c.id.replace('_L0','')));
      if (t !== null) el.append(svg('text', {x:c.x_mm+c.width_mm/2,y:y+c.height_mm*.65,'text-anchor':'middle',fill:'white','font-size':fs}, `${t.toFixed(2)}°C`));
    });
  }
  function dirty() {
    state.revision++; clearResults(); error(''); $('b-status').textContent = 'Inputs changed — predict to update temperatures.';
    try { const s = snapshot(); preview(s, null); validate(s); } catch(e) { error(e.message); }
  }
  function populate(floor) {
    $('b-die-width').value = floor.die.width_mm; $('b-die-height').value = floor.die.height_mm;
    $('b-inputs').replaceChildren();
    floor.cores.forEach(c => {
      const row = document.createElement('tr'), idCell = document.createElement('td'), select = document.createElement('select');
      select.setAttribute('aria-label', `${c.id} identifier`);
      IDS.forEach(id => { const o = new Option(id, id); select.add(o); }); select.value = c.id; select.addEventListener('change', dirty); idCell.append(select); row.append(idCell);
      ['width_mm','height_mm','x_mm','y_mm','power_w'].forEach(k => {
        const td = document.createElement('td'), input = document.createElement('input');
        td.dataset.label = {width_mm:'Width mm',height_mm:'Height mm',x_mm:'X mm',y_mm:'Y mm',power_w:'Power W'}[k];
        input.type = 'number'; input.step = k === 'power_w' ? '.1' : '.001'; input.min = '0'; input.value = Number(c[k].toFixed(6)); input.dataset.field = k; input.setAttribute('aria-label', `${c.id} ${k}`); input.addEventListener('input', dirty); td.append(input); row.append(td);
      }); $('b-inputs').append(row);
    });
    setMode(false); dirty(); $('b-status').textContent = 'Geometry loaded — adjust powers and predict.';
  }
  function setMode(reset) {
    const custom = $('b-mode').value === 'custom';
    $('b-floorplan').disabled = custom || state.loading;
    ['b-die-width','b-die-height'].forEach(id => $(id).readOnly = !custom);
    [...$('b-inputs').rows].forEach(r => {r.querySelector('select').disabled = !custom; r.querySelectorAll('input').forEach(i => i.readOnly = !custom && i.dataset.field !== 'power_w');});
    if (reset && state.floor) { populate(state.floor); return; }
    $('b-floorplan-note').textContent = custom ? 'Custom mode starts from the selected real layout. Edit IDs, geometry and die dimensions below.' : `${state.floor ? state.floor.id : ''} · repository geometry · 77 Stage B layouts`;
  }
  async function api(url, options) {
    const response = await fetch(url, options); const data = await response.json();
    if (!response.ok) throw new Error(data.error || `Request failed (${response.status})`);
    return data;
  }
  async function loadFloor() {
    const token = ++state.loadToken; state.revision++; state.loading = true; $('b-predict').disabled = true; clearResults(); error('');
    try {
      const data = await api(`/api/stage-b/floorplans/${encodeURIComponent($('b-floorplan').value)}`);
      if (token !== state.loadToken) return;
      state.floor = data; populate(data);
    } catch(e) { if (token === state.loadToken) {state.floor = null; error(e.message);} }
    finally { if (token === state.loadToken) {state.loading = false; $('b-predict').disabled = !state.floor; setMode(false);} }
  }
  const MODEL_COPY = {
    v2: {
      badge: 'Stage B v2 · Ridge (α = 0.01) · 3,028 B',
      summary: 'Stage B v2 · Ridge (α = 0.01)',
      scope: 'Shared per-core physics-informed Ridge model (Stage B v2, 63 features, 3,028 B portable .npz) for the supported 2D eight-core configuration. Fixed 45°C ambient and original training material settings. No per-floorplan calibration or online HotSpot run.',
      eval: 'Stage B v2 held-out test: MAE 0.350588°C · RMSE 0.445882°C · Max error 1.512353°C · Hottest-core 84.17%. External 23 Stage A layouts: MAE 0.460754°C. Note: these evaluation sets were reused across v1/v2 comparisons and are historical reference metrics, not guarantees for individual predictions.'
    },
    v1: {
      badge: 'Stage B v1 · Lasso (α = 0.001) · 11,197 B',
      summary: 'Stage B v1 · Lasso (α = 0.001)',
      scope: 'Baseline multi-output Lasso model (Stage B v1, 252 features, 11,197 B) preserved for direct comparison on identical inputs. Fixed 45°C ambient and original training material settings. No per-floorplan calibration or online HotSpot run.',
      eval: 'Stage B v1 held-out test: MAE 1.070281°C · RMSE 1.339064°C · Max error 4.974435°C · Hottest-core 56.67%. External 23 Stage A layouts: MAE 1.491469°C. Historical reference metrics, not guarantees for individual predictions.'
    }
  };
  function syncModelInfo() {
    const ver = $('b-model') ? $('b-model').value : 'v2';
    const copy = MODEL_COPY[ver] || MODEL_COPY.v2;
    if ($('b-model-badge')) $('b-model-badge').textContent = copy.badge;
    if ($('b-active-model')) $('b-active-model').textContent = copy.summary;
    if ($('b-model-scope')) $('b-model-scope').textContent = copy.scope;
    if ($('b-model-eval')) $('b-model-eval').textContent = copy.eval;
  }
  function show(s, result) {
    state.result = result; preview(s, result); $('b-results').replaceChildren();
    IDS.forEach(id => { const row = document.createElement('tr'); [id, `${result.temperatures_c[id].toFixed(3)} °C`].forEach(text => {const cell = document.createElement('td'); cell.textContent = text; row.append(cell);}); $('b-results').append(row); });
    if ($('b-active-model') && result.model) $('b-active-model').textContent = `${result.model.name} (α = ${result.model.alpha})`;
    $('b-max').textContent = `${result.maximum_c.toFixed(3)} °C`; $('b-min').textContent = `${result.minimum_c.toFixed(3)} °C`; $('b-hot').textContent = result.hottest_core; $('b-latency').textContent = `${result.latency_ms.toFixed(4)} ms`;
    $('b-low').textContent = `${result.minimum_c.toFixed(2)} °C`; $('b-high').textContent = `${result.maximum_c.toFixed(2)} °C`;
    const chart = $('b-bars'); chart.replaceChildren(); chart.setAttribute('viewBox','0 0 440 235');
    const base = 195, top = 35;
    const values = IDS.map(id => result.temperatures_c[id]);
    const lo = Math.min(0, ...values), hi = Math.max(0, ...values), span = Math.max(1,hi-lo)*1.12;
    const y = t => top+(hi+(span-(hi-lo))/2-t)/span*(base-top);
    const zero = y(0);
    chart.append(svg('text',{x:10,y:16,fill:'#94a3b8','font-size':12},'Temperature (°C) · zero baseline'));
    chart.append(svg('line',{x1:12,x2:433,y1:zero,y2:zero,stroke:'#64748b'}));
    IDS.forEach((id,i) => {const t = result.temperatures_c[id], yt=y(t), x=18+i*53;
      const r=svg('rect',{x,y:Math.min(zero,yt),width:34,height:Math.abs(yt-zero),fill:'#38bdf8','data-core':id,'data-temperature-c':t}); chart.append(r);
      chart.append(svg('text',{x:x+17,y:t>=0?yt-6:yt+12,'text-anchor':'middle',fill:'#e2e8f0','font-size':10},t.toFixed(1)));
      chart.append(svg('text',{x:x+17,y:224,'text-anchor':'middle',fill:'#94a3b8','font-size':10},`Core ${i}`));});
    $('b-status').textContent = `Prediction complete (${(result.model_version || 'v2').toUpperCase()}). Heatmap colors use this result’s minimum and maximum.`;
  }
  async function predict() {
    const revision = state.revision, token = ++state.request;
    try {
      error(''); const s = snapshot(); validate(s);
      const payload = {mode:$('b-mode').value,units,model_version:$('b-model') ? $('b-model').value : 'v2'};
      if (payload.mode === 'existing') {payload.floorplan_id = state.floor.id; payload.powers_w = Object.fromEntries(s.cores.map(c => [c.id,c.power_w]));}
      else {payload.die=s.die;payload.cores=s.cores;}
      $('b-predict').disabled = true; $('b-status').textContent = 'Predicting…';
      const result = await api('/api/stage-b/predict', {method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(payload)});
      if (revision !== state.revision || token !== state.request) return;
      show(s, result);
    } catch(e) { if (revision === state.revision && token === state.request) {clearResults(); error(e.message); $('b-status').textContent='Prediction unavailable — check inputs.';} }
    finally {if (token === state.request) $('b-predict').disabled = state.loading || !state.floor;}
  }
  if ($('b-model')) $('b-model').addEventListener('change', () => {syncModelInfo(); dirty(); $('b-status').textContent = 'Model switched — predict to compare temperatures on the same inputs.';});
  $('b-mode').addEventListener('change', () => {setMode(true);}); $('b-floorplan').addEventListener('change', loadFloor);
  ['b-die-width','b-die-height'].forEach(id => $(id).addEventListener('input',dirty));
  $('b-predict').addEventListener('click',predict); $('b-reset').addEventListener('click',() => {if(state.floor)populate(state.floor);});
  syncModelInfo();
  api('/api/stage-b/floorplans').then(data => {data.floorplans.forEach(f => $('b-floorplan').add(new Option(f.id,f.id)));return loadFloor();}).catch(e => error(e.message));
})();
