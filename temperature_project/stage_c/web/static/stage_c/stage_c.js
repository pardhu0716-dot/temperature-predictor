(() => {
  "use strict";

  const CORE_COLORS = [
    "#f43f5e", "#f97316", "#eab308", "#22c55e",
    "#06b6d4", "#3b82f6", "#a855f7", "#ec4899",
  ];

  const state = {
    allFloorplans: [],
    currentDetail: null,
    mode: "constant",
    hiddenCores: new Set(),
    lastPrediction: null,
  };

  const els = {
    styleFilter: document.getElementById("styleFilter"),
    floorplanSelect: document.getElementById("floorplanSelect"),
    coreInputsGrid: document.getElementById("coreInputsGrid"),
    presetSelect: document.getElementById("presetSelect"),
    presetDesc: document.getElementById("presetDesc"),
    scheduleCoreHeaders: document.getElementById("scheduleCoreHeaders"),
    scheduleBody: document.getElementById("scheduleBody"),
    traceTextarea: document.getElementById("traceTextarea"),
    btnPredict: document.getElementById("btnPredict"),
    btnWorkloadA: document.getElementById("btnWorkloadA"),
    btnMidCalibrated: document.getElementById("btnMidCalibrated"),
    btnAddSegment: document.getElementById("btnAddSegment"),
    btnFillTraceFromPreset: document.getElementById("btnFillTraceFromPreset"),
    errorBanner: document.getElementById("errorBanner"),
    domainBanner: document.getElementById("domainBanner"),
    domainBadge: document.getElementById("domainBadge"),
    domainHeadline: document.getElementById("domainHeadline"),
    extrapolationList: document.getElementById("extrapolationList"),
    tempChart: document.getElementById("tempChart"),
    chartTooltip: document.getElementById("chartTooltip"),
    chartLegend: document.getElementById("chartLegend"),
    chartSubtitle: document.getElementById("chartSubtitle"),
    peakSummaryInline: document.getElementById("peakSummaryInline"),
    accuracyMetrics: document.getElementById("accuracyMetrics"),
    footprintMetrics: document.getElementById("footprintMetrics"),
    coreSummaryBody: document.getElementById("coreSummaryBody"),
    shaHeaderBadge: document.getElementById("shaHeaderBadge"),
  };

  function showError(msg) {
    els.errorBanner.textContent = msg;
    els.errorBanner.classList.remove("sc-hidden");
  }

  function clearError() {
    els.errorBanner.textContent = "";
    els.errorBanner.classList.add("sc-hidden");
  }

  async function fetchJson(url, options = {}) {
    const resp = await fetch(url, options);
    const data = await resp.json().catch(() => ({}));
    if (!resp.ok) {
      throw new Error(data.error || `Request failed (${resp.status})`);
    }
    return data;
  }

  function populateFloorplanDropdown() {
    const filter = els.styleFilter.value;
    const filtered = state.allFloorplans.filter(
      (fp) => filter === "ALL" || fp.style === filter
    );
    const prev = els.floorplanSelect.value;
    els.floorplanSelect.innerHTML = "";
    filtered.forEach((fp) => {
      const opt = document.createElement("option");
      opt.value = fp.full_name;
      opt.textContent = `${fp.full_name} (RMSE ${fp.accuracy.two_speed_unseen_rmse_c.toFixed(2)} °C, -${fp.accuracy.rmse_reduction_pct.toFixed(1)}%)`;
      els.floorplanSelect.appendChild(opt);
    });
    if (filtered.some((fp) => fp.full_name === prev)) {
      els.floorplanSelect.value = prev;
    }
  }

  function renderCoreInputs(cores) {
    els.coreInputsGrid.innerHTML = "";
    cores.forEach((core) => {
      const card = document.createElement("div");
      card.className = "sc-core-card";
      card.dataset.coreName = core.name;
      const tagClass = core.core_type === "Big" ? "sc-type-big" : "sc-type-little";
      card.innerHTML = `
        <div class="sc-core-card-head">
          <span>${core.name}</span>
          <span class="sc-type-tag ${tagClass}">${core.core_type}</span>
        </div>
        <div class="sc-core-input-row">
          <input type="number" step="0.1" min="0" max="50"
                 data-core-input="${core.name}"
                 data-min-val="${core.validated_power_min_w}"
                 data-max-val="${core.validated_power_max_w}"
                 value="${core.default_power_w.toFixed(1)}" />
          <span>W</span>
        </div>
        <div class="sc-range-hint">Validated: ${core.validated_power_min_w.toFixed(0)}–${core.validated_power_max_w.toFixed(0)} W</div>
      `;
      const input = card.querySelector("input");
      input.addEventListener("input", () => updateConstantCardBorder(card, input));
      els.coreInputsGrid.appendChild(card);
    });
  }

  function updateConstantCardBorder(card, input) {
    const val = parseFloat(input.value);
    const lo = parseFloat(input.dataset.minVal);
    const hi = parseFloat(input.dataset.maxVal);
    if (!Number.isFinite(val) || val < lo || val > hi) {
      card.classList.add("out-of-domain");
    } else {
      card.classList.remove("out-of-domain");
    }
  }

  function setConstantPowers(bigW, littleW) {
    if (!state.currentDetail) return;
    state.currentDetail.cores.forEach((core) => {
      const input = els.coreInputsGrid.querySelector(`input[data-core-input="${core.name}"]`);
      if (input) {
        input.value = (core.core_type === "Big" ? bigW : littleW).toFixed(1);
        updateConstantCardBorder(input.closest(".sc-core-card"), input);
      }
    });
  }

  function renderScheduleHeaders(cores) {
    const cells = [
      "<th>Start–End</th>",
      "<th>Time (s)</th>",
      ...cores.map((c) => `<th>${c.name.replace("_L0", "")}<br><small>${c.core_type[0]}</small></th>`),
      "<th></th>",
    ];
    els.scheduleCoreHeaders.innerHTML = cells.join("");
  }

  function addScheduleRow(startStep, endStep, powers) {
    const tr = document.createElement("tr");
    const t0 = ((startStep - 1) * 0.01).toFixed(2);
    const t1 = (endStep * 0.01).toFixed(2);
    tr.innerHTML = `
      <td>
        <input type="number" class="seg-start" min="1" max="200" value="${startStep}" />
        –
        <input type="number" class="seg-end" min="1" max="200" value="${endStep}" />
      </td>
      <td class="seg-time">${t0}–${t1}</td>
      ${powers
        .map(
          (p) =>
            `<td><input type="number" class="seg-power" step="0.1" min="0" max="50" value="${Number(p).toFixed(1)}" /></td>`
        )
        .join("")}
      <td><button type="button" class="sc-btn sc-btn-secondary seg-del">×</button></td>
    `;
    const updateTime = () => {
      const s = parseInt(tr.querySelector(".seg-start").value, 10) || 1;
      const e = parseInt(tr.querySelector(".seg-end").value, 10) || 1;
      tr.querySelector(".seg-time").textContent = `${((s - 1) * 0.01).toFixed(2)}–${(e * 0.01).toFixed(2)}`;
    };
    tr.querySelector(".seg-start").addEventListener("input", updateTime);
    tr.querySelector(".seg-end").addEventListener("input", updateTime);
    tr.querySelector(".seg-del").addEventListener("click", () => {
      if (els.scheduleBody.children.length > 1) tr.remove();
    });
    els.scheduleBody.appendChild(tr);
  }

  function loadPresetIntoSchedule(preset) {
    els.presetDesc.textContent = preset.description;
    els.scheduleBody.innerHTML = "";
    preset.segments.forEach((seg) => {
      addScheduleRow(seg.start_step, seg.end_step, seg.powers_w);
    });
  }

  function populatePresets(presets) {
    els.presetSelect.innerHTML = "";
    presets.forEach((p) => {
      const opt = document.createElement("option");
      opt.value = p.id;
      opt.textContent = p.name;
      els.presetSelect.appendChild(opt);
    });
    if (presets.length > 0) {
      loadPresetIntoSchedule(presets[0]);
    }
  }

  function collectScheduleSegments() {
    const rows = Array.from(els.scheduleBody.querySelectorAll("tr"));
    return rows.map((tr) => ({
      start_step: parseInt(tr.querySelector(".seg-start").value, 10),
      end_step: parseInt(tr.querySelector(".seg-end").value, 10),
      powers_w: Array.from(tr.querySelectorAll(".seg-power")).map((inp) => parseFloat(inp.value)),
    }));
  }

  function parseCsvTrace(text) {
    const lines = text
      .split(/\r?\n/)
      .map((l) => l.trim())
      .filter((l) => l.length > 0);
    if (lines.length === 0) throw new Error("Trace textarea is empty.");
    const rows = [];
    for (let i = 0; i < lines.length; i++) {
      const parts = lines[i].split(/[,\t;\s]+/).filter(Boolean);
      if (i === 0 && parts.some((tok) => Number.isNaN(Number(tok)))) {
        continue; // skip header row
      }
      rows.push(parts.map((tok) => Number(tok)));
    }
    return rows;
  }

  function renderMetrics(accuracy, footprint) {
    els.accuracyMetrics.innerHTML = `
      <div class="sc-metric">
        <div class="sc-metric-label">Two-Speed Unseen RMSE</div>
        <div class="sc-metric-value">${accuracy.two_speed_unseen_rmse_c.toFixed(4)} °C</div>
        <div class="sc-metric-sub">vs One-Speed ${accuracy.one_speed_unseen_rmse_c.toFixed(4)} °C</div>
      </div>
      <div class="sc-metric">
        <div class="sc-metric-label">RMSE Reduction</div>
        <div class="sc-metric-value">${accuracy.rmse_reduction_pct.toFixed(2)}%</div>
        <div class="sc-metric-sub">${accuracy.rmse_improvement_factor.toFixed(2)}× improvement</div>
      </div>
      <div class="sc-metric">
        <div class="sc-metric-label">Two-Speed Unseen MAE</div>
        <div class="sc-metric-value">${accuracy.two_speed_unseen_mae_c.toFixed(4)} °C</div>
      </div>
      <div class="sc-metric">
        <div class="sc-metric-label">Two-Speed Max Error</div>
        <div class="sc-metric-value">${accuracy.two_speed_unseen_max_c.toFixed(4)} °C</div>
        <div class="sc-metric-sub">Peak instantaneous error</div>
      </div>
    `;

    const liveUs =
      footprint.live_inference_time_us !== undefined
        ? `${footprint.live_inference_time_us.toFixed(2)} µs`
        : "—";

    els.footprintMetrics.innerHTML = `
      <div class="sc-metric">
        <div class="sc-metric-label">Model Raw Size</div>
        <div class="sc-metric-value">${footprint.raw_float64_bytes} B</div>
        <div class="sc-metric-sub">${footprint.total_parameters} float64 (${footprint.steady_state_r_parameters} R + ${footprint.transient_parameters} transient)</div>
      </div>
      <div class="sc-metric">
        <div class="sc-metric-label">Live 200-Step Latency</div>
        <div class="sc-metric-value">${liveUs}</div>
        <div class="sc-metric-sub">Audited median: ${footprint.benchmark_median_200steps_us.toFixed(2)} µs</div>
      </div>
      <div class="sc-metric" style="grid-column: span 2;">
        <div class="sc-metric-label">Frozen Parameter SHA-256 (${footprint.sha256_verified ? "Verified" : "Unverified"})</div>
        <div class="sc-metric-sub">${footprint.param_sha256}</div>
      </div>
    `;
  }

  function renderDomainBanner(pred) {
    if (pred.domain_status === "validated") {
      els.domainBanner.className = "sc-domain-banner sc-domain-validated";
      els.domainBadge.className = "sc-pill sc-pill-ok";
      els.domainBadge.textContent = "VALIDATED DOMAIN";
      els.domainHeadline.textContent =
        "All 8 core power inputs lie within the calibrated workload domain (Big 10–20 W, Little 5–10 W, 200 steps @ 0.01 s).";
      els.extrapolationList.classList.add("sc-hidden");
      els.extrapolationList.innerHTML = "";
    } else {
      els.domainBanner.className = "sc-domain-banner sc-domain-extrapolated";
      els.domainBadge.className = "sc-pill sc-pill-warn";
      els.domainBadge.textContent = "EXTRAPOLATED";
      els.domainHeadline.textContent =
        "Inputs are physically valid but extend outside the calibrated workload domain; accuracy is not bounded by audited Workload C RMSE.";
      els.extrapolationList.innerHTML = pred.extrapolation_warnings
        .map((w) => `<li>${w.reason}</li>`)
        .join("");
      els.extrapolationList.classList.remove("sc-hidden");
    }
  }

  function renderCoreSummaryTable(rows) {
    els.coreSummaryBody.innerHTML = rows
      .map(
        (r) => `
      <tr>
        <td><strong>${r.core_name}</strong></td>
        <td><span class="sc-type-tag ${r.core_type === "Big" ? "sc-type-big" : "sc-type-little"}">${r.core_type}</span></td>
        <td>${r.min_power_w.toFixed(2)} – ${r.max_power_w.toFixed(2)} W</td>
        <td><strong>${r.peak_temp_c.toFixed(2)} °C</strong></td>
        <td>${r.peak_time_s.toFixed(2)} s</td>
        <td>${r.final_temp_c.toFixed(2)} °C</td>
        <td>+${r.delta_final_c.toFixed(2)} °C</td>
      </tr>`
      )
      .join("");
  }

  function renderChart(pred) {
    const svg = els.tempChart;
    const W = 860, H = 360;
    const pad = { l: 58, r: 20, t: 18, b: 42 };
    const plotW = W - pad.l - pad.r;
    const plotH = H - pad.t - pad.b;

    const times = pred.plot_times_s;
    const maxTime = times[times.length - 1] || 2.0;

    let yMin = 44.0;
    let yMax = 55.0;
    pred.core_names.forEach((name) => {
      if (state.hiddenCores.has(name)) return;
      const arr = pred.plot_temperatures_c[name];
      arr.forEach((v) => {
        if (v < yMin) yMin = v;
        if (v > yMax) yMax = v;
      });
    });
    const yPad = Math.max(2.0, (yMax - yMin) * 0.08);
    yMin = Math.floor(Math.min(45.0, yMin - yPad));
    yMax = Math.ceil(yMax + yPad);

    const xScale = (t) => pad.l + (t / maxTime) * plotW;
    const yScale = (v) => pad.t + plotH - ((v - yMin) / (yMax - yMin)) * plotH;

    let html = "";

    // Horizontal grid & Y labels
    const yTicks = 6;
    for (let i = 0; i <= yTicks; i++) {
      const val = yMin + ((yMax - yMin) * i) / yTicks;
      const y = yScale(val);
      html += `<line x1="${pad.l}" y1="${y}" x2="${W - pad.r}" y2="${y}" stroke="#1e293b" stroke-width="1" />`;
      html += `<text x="${pad.l - 8}" y="${y + 4}" text-anchor="end" fill="#94a3b8" font-size="11">${val.toFixed(1)}</text>`;
    }

    // Vertical grid & X labels
    const xTicks = 8;
    for (let i = 0; i <= xTicks; i++) {
      const t = (maxTime * i) / xTicks;
      const x = xScale(t);
      html += `<line x1="${x}" y1="${pad.t}" x2="${x}" y2="${H - pad.b}" stroke="#1e293b" stroke-width="1" />`;
      html += `<text x="${x}" y="${H - pad.b + 16}" text-anchor="middle" fill="#94a3b8" font-size="11">${t.toFixed(2)}s</text>`;
    }

    // Axis titles
    html += `<text x="${pad.l + plotW / 2}" y="${H - 6}" text-anchor="middle" fill="#cbd5e1" font-size="12">Time (seconds)</text>`;
    html += `<text x="15" y="${pad.t + plotH / 2}" text-anchor="middle" fill="#cbd5e1" font-size="12" transform="rotate(-90 15 ${pad.t + plotH / 2})">Temperature (°C)</text>`;

    // Lines
    pred.core_names.forEach((name, idx) => {
      if (state.hiddenCores.has(name)) return;
      const arr = pred.plot_temperatures_c[name];
      const color = CORE_COLORS[idx % CORE_COLORS.length];
      const isBig = pred.core_types[idx] === "Big";
      const pts = arr.map((v, k) => `${xScale(times[k]).toFixed(1)},${yScale(v).toFixed(1)}`).join(" ");
      html += `<polyline fill="none" stroke="${color}" stroke-width="${isBig ? 2.6 : 1.7}" ${isBig ? "" : 'stroke-dasharray="5,3"'} points="${pts}" />`;
    });

    // Crosshair line + interactive overlay
    html += `<line id="scCrosshair" x1="${pad.l}" y1="${pad.t}" x2="${pad.l}" y2="${H - pad.b}" stroke="#94a3b8" stroke-dasharray="3,3" stroke-width="1" visibility="hidden" />`;
    html += `<rect id="scHoverRect" x="${pad.l}" y="${pad.t}" width="${plotW}" height="${plotH}" fill="transparent" style="cursor:crosshair" />`;

    svg.innerHTML = html;

    const crosshair = svg.querySelector("#scCrosshair");
    const hoverRect = svg.querySelector("#scHoverRect");
    hoverRect.addEventListener("mousemove", (evt) => {
      const rect = svg.getBoundingClientRect();
      const svgX = ((evt.clientX - rect.left) / rect.width) * W;
      const frac = Math.max(0, Math.min(1, (svgX - pad.l) / plotW));
      const idx = Math.round(frac * (times.length - 1));
      const xPos = xScale(times[idx]);
      crosshair.setAttribute("x1", xPos);
      crosshair.setAttribute("x2", xPos);
      crosshair.setAttribute("visibility", "visible");

      const lines = pred.core_names.map((name, cIdx) => {
        const tVal = pred.plot_temperatures_c[name][idx];
        const pVal = idx === 0 ? 0 : pred.powers_w[name][idx - 1];
        const color = CORE_COLORS[cIdx % CORE_COLORS.length];
        return `<div style="display:flex;justify-content:space-between;gap:10px;"><span style="color:${color};font-weight:700">${name} (${pred.core_types[cIdx][0]})</span><span>${tVal.toFixed(2)} °C (${pVal.toFixed(1)}W)</span></div>`;
      });
      els.chartTooltip.innerHTML = `<div style="font-weight:700;margin-bottom:4px;border-bottom:1px solid #334155;padding-bottom:3px;">t = ${times[idx].toFixed(2)} s (step ${idx})</div>${lines.join("")}`;
      els.chartTooltip.classList.remove("sc-hidden");

      const leftPx = Math.min(rect.width - 210, Math.max(8, evt.clientX - rect.left + 14));
      els.chartTooltip.style.left = `${leftPx}px`;
      els.chartTooltip.style.top = "12px";
    });

    hoverRect.addEventListener("mouseleave", () => {
      crosshair.setAttribute("visibility", "hidden");
      els.chartTooltip.classList.add("sc-hidden");
    });

    // Legend
    els.chartLegend.innerHTML = "";
    pred.core_names.forEach((name, idx) => {
      const item = document.createElement("button");
      item.type = "button";
      item.className = `sc-legend-item ${state.hiddenCores.has(name) ? "muted" : ""}`;
      const color = CORE_COLORS[idx % CORE_COLORS.length];
      item.innerHTML = `<span class="sc-swatch" style="background:${color}"></span>${name} (${pred.core_types[idx]})`;
      item.addEventListener("click", () => {
        if (state.hiddenCores.has(name)) state.hiddenCores.delete(name);
        else state.hiddenCores.add(name);
        renderChart(pred);
      });
      els.chartLegend.appendChild(item);
    });
  }

  async function runPrediction() {
    clearError();
    if (!state.currentDetail) return;

    const body = {
      floorplan: state.currentDetail.full_name,
      mode: state.mode,
      num_steps: 200,
      dt_seconds: 0.01,
      init_temp_c: 45.0,
    };

    try {
      if (state.mode === "constant") {
        const powersByCore = {};
        state.currentDetail.cores.forEach((c) => {
          const inp = els.coreInputsGrid.querySelector(`input[data-core-input="${c.name}"]`);
          powersByCore[c.name] = parseFloat(inp.value);
        });
        body.powers_by_core = powersByCore;
      } else if (state.mode === "schedule") {
        body.segments = collectScheduleSegments();
      } else {
        const rows = parseCsvTrace(els.traceTextarea.value);
        body.num_steps = rows.length;
        body.power_trace_w = rows;
      }

      const pred = await fetchJson("/api/stage-c/predict", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(body),
      });

      state.lastPrediction = pred;
      els.chartSubtitle.textContent = `${pred.floorplan} • Mode: ${pred.mode} • ${pred.num_steps} steps (${pred.horizon_seconds.toFixed(2)} s @ ${pred.dt_seconds} s)`;
      els.peakSummaryInline.textContent = `Peak: ${pred.overall_summary.peak_temp_c.toFixed(2)} °C (${pred.overall_summary.hottest_core} @ ${pred.overall_summary.peak_time_s.toFixed(2)} s)`;
      renderDomainBanner(pred);
      renderMetrics(pred.accuracy, pred.model_footprint);
      renderCoreSummaryTable(pred.per_core_summary);
      renderChart(pred);
    } catch (err) {
      showError(err.message);
    }
  }

  async function loadFloorplanDetail(fullName) {
    clearError();
    const [style, fp] = fullName.split("/");
    const detail = await fetchJson(`/api/stage-c/floorplan/${encodeURIComponent(style)}/${encodeURIComponent(fp)}`);
    state.currentDetail = detail;
    state.hiddenCores.clear();
    renderCoreInputs(detail.cores);
    renderScheduleHeaders(detail.cores);
    populatePresets(detail.presets);
    renderMetrics(detail.accuracy, detail.model_footprint);
    await runPrediction();
  }

  function switchMode(newMode) {
    state.mode = newMode;
    document.querySelectorAll(".sc-tab").forEach((btn) => {
      btn.classList.toggle("active", btn.dataset.mode === newMode);
    });
    document.getElementById("modeConstant").classList.toggle("sc-hidden", newMode !== "constant");
    document.getElementById("modeSchedule").classList.toggle("sc-hidden", newMode !== "schedule");
    document.getElementById("modeTrace").classList.toggle("sc-hidden", newMode !== "trace");
  }

  async function init() {
    try {
      const data = await fetchJson("/api/stage-c/floorplans");
      state.allFloorplans = data.floorplans;
      if (data.summary && data.summary.all_sha256_verified) {
        els.shaHeaderBadge.textContent = `SHA-256 Verified (${data.summary.floorplans_verified}/23)`;
      }
      populateFloorplanDropdown();
      if (state.allFloorplans.length > 0) {
        await loadFloorplanDetail(state.allFloorplans[0].full_name);
      }
    } catch (err) {
      showError(`Failed to initialize Stage C registry: ${err.message}`);
    }
  }

  els.styleFilter.addEventListener("change", async () => {
    populateFloorplanDropdown();
    if (els.floorplanSelect.value) {
      await loadFloorplanDetail(els.floorplanSelect.value);
    }
  });

  els.floorplanSelect.addEventListener("change", async () => {
    if (els.floorplanSelect.value) {
      await loadFloorplanDetail(els.floorplanSelect.value);
    }
  });

  document.querySelectorAll(".sc-tab").forEach((btn) => {
    btn.addEventListener("click", () => switchMode(btn.dataset.mode));
  });

  els.btnWorkloadA.addEventListener("click", () => setConstantPowers(20.0, 8.0));
  els.btnMidCalibrated.addEventListener("click", () => setConstantPowers(15.0, 7.5));

  els.presetSelect.addEventListener("change", () => {
    if (!state.currentDetail) return;
    const preset = state.currentDetail.presets.find((p) => p.id === els.presetSelect.value);
    if (preset) loadPresetIntoSchedule(preset);
  });

  els.btnAddSegment.addEventListener("click", () => {
    if (!state.currentDetail) return;
    const defaults = state.currentDetail.cores.map((c) => c.default_power_w);
    addScheduleRow(1, 200, defaults);
  });

  els.btnFillTraceFromPreset.addEventListener("click", () => {
    if (!state.currentDetail || !state.currentDetail.presets.length) return;
    const preset = state.currentDetail.presets[0];
    const header = state.currentDetail.cores.map((c) => c.name).join(",");
    const lines = [header];
    preset.segments.forEach((seg) => {
      const rowStr = seg.powers_w.map((v) => Number(v).toFixed(2)).join(",");
      for (let s = seg.start_step; s <= seg.end_step; s++) {
        lines.push(rowStr);
      }
    });
    els.traceTextarea.value = lines.join("\n");
  });

  els.btnPredict.addEventListener("click", runPrediction);

  init();
})();
