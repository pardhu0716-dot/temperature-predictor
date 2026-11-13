/**
 * Stage A Temperature Predictor Website — Frontend Application Logic
 * Lightweight Temperature Predictor Project
 */

const AppState = {
  catalog: [],
  stylesMap: {},
  currentStyle: "Style_1",
  currentFloorplan: null, // Full bundle from GET /api/floorplan/<style>/<fp>
  currentMode: "all",     // "all" or exact core name e.g. "Core_0_L0"
  lastPrediction: null,   // Last response from POST /api/predict (null when cleared/outdated)
  isOutdated: false,
  floorplanRequestId: 0,
  predictionRequestId: 0,
  loadingFloorplan: false,
};

// DOM References
const dom = {
  hdrFloorplan: document.getElementById("hdr-floorplan"),
  hdrDieSize: document.getElementById("hdr-die-size"),
  hdrTopology: document.getElementById("hdr-topology"),
  hdrValRmse: document.getElementById("hdr-val-rmse"),
  styleSelect: document.getElementById("style-select"),
  floorplanSelect: document.getElementById("floorplan-select"),
  fpStatusLine: document.getElementById("fp-status-line"),
  powerInputsContainer: document.getElementById("power-inputs-container"),
  errorBanner: document.getElementById("validation-error-banner"),
  staleBanner: document.getElementById("stale-warning-banner"),
  btnPresetVal: document.getElementById("btn-preset-val"),
  btnPresetMid: document.getElementById("btn-preset-mid"),
  btnPresetMin: document.getElementById("btn-preset-min"),
  btnPresetMax: document.getElementById("btn-preset-max"),
  btnPredictAll: document.getElementById("btn-predict-all"),
  individualBtnsContainer: document.getElementById("individual-buttons-container"),
  heatmapModeBadge: document.getElementById("heatmap-mode-badge"),
  floorplanSvg: document.getElementById("floorplan-svg"),
  chipStageWrapper: document.getElementById("chip-stage-wrapper"),
  chipTooltip: document.getElementById("chip-tooltip"),
  legendTitle: document.getElementById("legend-title"),
  legendModeNote: document.getElementById("legend-mode-note"),
  legendSingleMarker: document.getElementById("legend-single-marker"),
  legendMinVal: document.getElementById("legend-min-val"),
  legendMid1Val: document.getElementById("legend-mid1-val"),
  legendMid2Val: document.getElementById("legend-mid2-val"),
  legendMaxVal: document.getElementById("legend-max-val"),
  chartSubtitle: document.getElementById("chart-subtitle"),
  chartStatusBadge: document.getElementById("chart-status-badge"),
  chartStageWrapper: document.getElementById("chart-stage-wrapper"),
  coreTempChartSvg: document.getElementById("core-temp-chart-svg"),
  chartTooltip: document.getElementById("chart-tooltip"),
  resultBannerTitle: document.getElementById("result-banner-title"),
  resultModePill: document.getElementById("result-mode-pill"),
  kpiTempLabel: document.getElementById("kpi-temp-label"),
  kpiTempC: document.getElementById("kpi-temp-c"),
  kpiTempK: document.getElementById("kpi-temp-k"),
  kpiCoreLabel: document.getElementById("kpi-core-label"),
  kpiCoreName: document.getElementById("kpi-core-name"),
  kpiCoreType: document.getElementById("kpi-core-type"),
  kpiTotalPower: document.getElementById("kpi-total-power"),
  kpiPredTime: document.getElementById("kpi-pred-time"),
  kpiRoundtripTime: document.getElementById("kpi-roundtrip-time"),
  kpiModelSize: document.getElementById("kpi-model-size"),
  kpiMatrixPath: document.getElementById("kpi-matrix-path"),
  tableCardTitle: document.getElementById("table-card-title"),
  tableRowCount: document.getElementById("table-row-count"),
  resultsTableBody: document.getElementById("results-table-body"),
  couplingSection: document.getElementById("coupling-breakdown-section"),
  couplingTitle: document.getElementById("coupling-title"),
  couplingTableBody: document.getElementById("coupling-table-body"),
  inspectorMatrixFile: document.getElementById("inspector-matrix-file"),
  matrixRThead: document.getElementById("matrix-r-thead"),
  matrixRTbody: document.getElementById("matrix-r-tbody"),
  evalCard: document.getElementById("eval-card"),
  evalSubtitle: document.getElementById("eval-subtitle"),
  evalStatusBadge: document.getElementById("eval-status-badge"),
  evalContentContainer: document.getElementById("eval-content-container"),
  evalValRmse: document.getElementById("eval-val-rmse"),
  evalValMae: document.getElementById("eval-val-mae"),
  evalValMax: document.getElementById("eval-val-max"),
  evalSpeedup: document.getElementById("eval-speedup"),
  evalSpeedupSub: document.getElementById("eval-speedup-sub"),
  evalNumpyTime: document.getElementById("eval-numpy-time"),
  evalHotspotTime: document.getElementById("eval-hotspot-time"),
  evalWebTime: document.getElementById("eval-web-time"),
  evalModelSize: document.getElementById("eval-model-size"),
  evalBarHotspot: document.getElementById("eval-bar-hotspot"),
  evalBarHotspotVal: document.getElementById("eval-bar-hotspot-val"),
  evalBarWeb: document.getElementById("eval-bar-web"),
  evalBarWebVal: document.getElementById("eval-bar-web-val"),
  evalBarNumpy: document.getElementById("eval-bar-numpy"),
  evalBarNumpyVal: document.getElementById("eval-bar-numpy-val"),
  evalBarStoragePredictor: document.getElementById("eval-bar-storage-predictor"),
  evalBarStoragePredictorVal: document.getElementById("eval-bar-storage-predictor-val"),
  evalBarStoragePredictorMeta: document.getElementById("eval-bar-storage-predictor-meta"),
  evalBarStorageTotal: document.getElementById("eval-bar-storage-total"),
  evalBarStorageTotalVal: document.getElementById("eval-bar-storage-total-val"),
  evalBarStorageTotalMeta: document.getElementById("eval-bar-storage-total-meta"),
  evalUnavailableBanner: document.getElementById("eval-unavailable-banner"),
};

/**
 * Interpolate a normalized value u in [0, 1] along a 5-stop thermal heatmap palette:
 *   0.00 -> Blue   (#2563eb)
 *   0.28 -> Cyan   (#0891b2)
 *   0.55 -> Yellow (#eab308)
 *   0.78 -> Orange (#f97316)
 *   1.00 -> Red    (#dc2626)
 */
function interpolateHeatmapColor(u) {
  const clamped = Math.max(0, Math.min(1, u));
  const stops = [
    { pos: 0.0, rgb: [37, 99, 235] },
    { pos: 0.28, rgb: [8, 145, 178] },
    { pos: 0.55, rgb: [234, 179, 8] },
    { pos: 0.78, rgb: [249, 115, 22] },
    { pos: 1.0, rgb: [220, 38, 38] },
  ];
  for (let i = 0; i < stops.length - 1; i++) {
    const s0 = stops[i];
    const s1 = stops[i + 1];
    if (clamped >= s0.pos && clamped <= s1.pos) {
      const t = (clamped - s0.pos) / (s1.pos - s0.pos || 1);
      const r = Math.round(s0.rgb[0] + t * (s1.rgb[0] - s0.rgb[0]));
      const g = Math.round(s0.rgb[1] + t * (s1.rgb[1] - s0.rgb[1]));
      const b = Math.round(s0.rgb[2] + t * (s1.rgb[2] - s0.rgb[2]));
      return `rgb(${r}, ${g}, ${b})`;
    }
  }
  return "rgb(220, 38, 38)";
}

/**
 * Initialize dashboard: load catalog of 23 floorplans from backend metadata
 * and select Style_1 / fp_001 by default.
 */
async function initDashboard() {
  try {
    const res = await fetch("/api/floorplans");
    if (!res.ok) throw new Error(`Failed to fetch floorplans (${res.status})`);
    const data = await res.json();
    AppState.catalog = data.floorplans || [];
    AppState.stylesMap = buildStylesMapFromCatalog(AppState.catalog, data.styles);

    const styleNames = Object.keys(AppState.stylesMap);
    const initialStyle = styleNames.includes("Style_1") ? "Style_1" : styleNames[0];
    populateStyleDropdown(initialStyle);
    populateFloorplanDropdown(initialStyle, "fp_001");
    bindEventListeners();

    const initialFp = dom.floorplanSelect.value;
    if (initialStyle && initialFp) {
      await loadFloorplan(`${initialStyle}/${initialFp}`, { autoPredict: true });
    }
  } catch (err) {
    showValidationError(`Initialization error: ${err.message}`);
  }
}

/**
 * Group backend floorplans by style using backend floorplan metadata as source of truth.
 */
function buildStylesMapFromCatalog(catalog, backendStyles) {
  const byStyle = {};
  for (const item of catalog) {
    if (!byStyle[item.style]) byStyle[item.style] = [];
    byStyle[item.style].push(item);
  }
  if (Object.keys(byStyle).length === 0 && backendStyles) {
    for (const [st, info] of Object.entries(backendStyles)) {
      byStyle[st] = info.floorplans || [];
    }
  }
  return byStyle;
}

/**
 * Dropdown 1: Populate "Floorplan Style" options dynamically from backend metadata:
 *   Style_1 (3 floorplans), Style_2 (5 floorplans), Style_3 (5 floorplans), Style_4 (10 floorplans)
 */
function populateStyleDropdown(selectedStyle) {
  dom.styleSelect.innerHTML = "";
  const styleEntries = Object.entries(AppState.stylesMap);
  for (const [styleName, fps] of styleEntries) {
    const opt = document.createElement("option");
    opt.value = styleName;
    opt.textContent = `${styleName} (${fps.length} floorplans)`;
    dom.styleSelect.appendChild(opt);
  }
  if (selectedStyle && AppState.stylesMap[selectedStyle]) {
    dom.styleSelect.value = selectedStyle;
    AppState.currentStyle = selectedStyle;
  } else if (styleEntries.length > 0) {
    dom.styleSelect.value = styleEntries[0][0];
    AppState.currentStyle = styleEntries[0][0];
  }
}

/**
 * Dropdown 2: Populate "Floorplan" options dynamically for the selected style only.
 * Displays simple labels such as fp_001, fp_002, fp_003, etc.
 */
function populateFloorplanDropdown(styleName, selectedFpId) {
  dom.floorplanSelect.innerHTML = "";
  const fps = AppState.stylesMap[styleName] || AppState.catalog.filter((f) => f.style === styleName);

  for (const fp of fps) {
    const opt = document.createElement("option");
    opt.value = fp.floorplan;
    opt.dataset.fullName = fp.full_name;
    opt.textContent = fp.floorplan;
    dom.floorplanSelect.appendChild(opt);
  }

  const normalizedTarget = selectedFpId && selectedFpId.includes("/")
    ? selectedFpId.split("/")[1]
    : selectedFpId;

  if (normalizedTarget && fps.some((f) => f.floorplan === normalizedTarget)) {
    dom.floorplanSelect.value = normalizedTarget;
  } else if (fps.length > 0) {
    dom.floorplanSelect.value = fps[0].floorplan;
  }
}

/**
 * Load a specific floorplan's geometry, core order, power ranges, and matrix metadata.
 * Clears stale predictions, graph points, temperature table, and selected-core state on floorplan changes.
 */
async function loadFloorplan(fullName, options = { autoPredict: false }) {
  const requestId = ++AppState.floorplanRequestId;
  ++AppState.predictionRequestId;
  AppState.loadingFloorplan = true;
  if (AppState.currentFloorplan) {
    clearStalePredictionDisplay(AppState.currentFloorplan, "Loading floorplan…");
  }
  clearValidationErrors();
  dom.staleBanner.classList.add("hidden");

  // Immediately clear stale prediction state before fetching new floorplan
  AppState.lastPrediction = null;
  AppState.isOutdated = false;
  AppState.currentMode = "all";

  let style = AppState.currentStyle || dom.styleSelect.value;
  let fpId = fullName;
  if (fullName && fullName.includes("/")) {
    const parts = fullName.split("/");
    style = parts[0];
    fpId = parts[1];
  }

  try {
    const res = await fetch(`/api/floorplan/${encodeURIComponent(style)}/${encodeURIComponent(fpId)}`);
    const bundle = await res.json();
    if (requestId !== AppState.floorplanRequestId) return;
    if (!res.ok) {
      throw new Error(bundle.error || `Failed to load ${style}/${fpId}`);
    }

    AppState.currentStyle = bundle.style;
    AppState.currentFloorplan = bundle;
    AppState.loadingFloorplan = false;

    // Keep both side-by-side dropdowns synchronized with the loaded floorplan
    if (dom.styleSelect.value !== bundle.style) {
      dom.styleSelect.value = bundle.style;
    }
    populateFloorplanDropdown(bundle.style, bundle.floorplan);

    // Update Header & Short Muted Status Line below dropdowns
    dom.hdrFloorplan.textContent = bundle.full_name;
    dom.hdrDieSize.textContent = `${bundle.die_width_mm.toFixed(1)} × ${bundle.die_height_mm.toFixed(1)} mm (${bundle.die_area_mm2.toFixed(0)} mm²)`;
    dom.hdrTopology.textContent = `${bundle.num_big_cores} Big, ${bundle.num_little_cores} Little, ${bundle.num_fill_blocks} FILL`;
    if (bundle.validation_metrics && bundle.validation_metrics.val_rmse_k !== null) {
      dom.hdrValRmse.textContent = `${bundle.validation_metrics.val_rmse_k.toFixed(6)} K`;
    } else {
      dom.hdrValRmse.textContent = "Verified";
    }
    dom.fpStatusLine.textContent = `${bundle.style} / ${bundle.floorplan} · Calibrated model loaded`;

    // Build compact 8 power inputs in exact matrix column ordering
    renderPowerInputFields(bundle);

    // Build concise 8 individual core prediction buttons mapped to actual core names
    renderIndividualCoreButtons(bundle);

    // Render 8x8 Thermal Resistance Matrix R table
    renderThermalMatrixTable(bundle);

    // Render Stage A Performance & Evaluation Benchmark section
    renderBenchmarkSection(bundle);

    // Clear stale results, reset floorplan SVG to neutral unpredicted state, and reset graph
    clearStalePredictionDisplay(
      bundle,
      `Loaded ${bundle.full_name} — click Predict All Cores or an individual core button.`
    );

    if (options && options.autoPredict) {
      await runPrediction("all");
    }
  } catch (err) {
    if (requestId !== AppState.floorplanRequestId) return;
    AppState.currentFloorplan = null;
    showValidationError(err.message);
  } finally {
    if (requestId === AppState.floorplanRequestId) AppState.loadingFloorplan = false;
  }
}

/**
 * 1. Build the 8 compact power input rows in exact calibrated matrix column order.
 * Shows only: Core k · Big/Little · Allowed range [input] W (no coordinates or Col indices).
 */
function renderPowerInputFields(bundle) {
  dom.powerInputsContainer.innerHTML = "";
  bundle.cores.forEach((core) => {
    const row = document.createElement("div");
    row.className = "power-row";
    row.id = `power-row-${core.name}`;
    row.dataset.coreName = core.name;

    const isBig = core.short_type === "Big";
    const badgeClass = isBig ? "type-badge type-big" : "type-badge type-little";
    const rangeStr = `${core.min_power_w.toFixed(0)}–${core.max_power_w.toFixed(0)} W`;

    row.innerHTML = `
      <div class="power-core-inline" title="${core.name} (${core.core_type})">
        <span class="core-name-label">Core ${core.core_number}</span>
        <span class="dot-sep">&middot;</span>
        <span class="${badgeClass}">${core.short_type}</span>
        <span class="dot-sep">&middot;</span>
        <span class="power-range-compact">${rangeStr}</span>
      </div>
      <div class="power-input-wrap">
        <input
          type="number"
          step="0.01"
          min="${core.min_power_w}"
          max="${core.max_power_w}"
          value="${core.default_power_w.toFixed(2)}"
          class="power-input"
          id="input-power-${core.name}"
          data-core-name="${core.name}"
          data-core-num="${core.core_number}"
          data-min="${core.min_power_w}"
          data-max="${core.max_power_w}"
          data-type="${core.short_type}"
          aria-label="Power in Watts for Core ${core.core_number} (${core.name})"
        />
        <span class="unit-w">W</span>
      </div>
      <div class="power-field-error hidden" id="err-power-${core.name}"></div>
    `;

    const inputEl = row.querySelector("input");
    inputEl.addEventListener("input", () => {
      validateSingleInputElement(inputEl);
      markPredictionOutdated();
    });
    inputEl.addEventListener("keydown", (e) => {
      if (e.key === "Enter") {
        e.preventDefault();
        runPrediction(AppState.currentMode || "all");
      }
    });

    dom.powerInputsContainer.appendChild(row);
  });
}

/**
 * 2. Build the 8 concise individual core prediction buttons:
 * Exactly "Predict Core 0" .. "Predict Core 7", mapped by actual core name.
 */
function renderIndividualCoreButtons(bundle) {
  dom.individualBtnsContainer.innerHTML = "";
  const sortedCores = bundle.cores_by_number || bundle.cores;
  sortedCores.forEach((core) => {
    const btn = document.createElement("button");
    btn.type = "button";
    btn.className = "btn-predict-core";
    btn.id = `btn-predict-${core.name}`;
    btn.dataset.coreName = core.name;
    btn.dataset.coreNum = String(core.core_number);
    btn.title = `Predict ${core.name} (${core.core_type})`;
    btn.textContent = `Predict Core ${core.core_number}`;
    btn.addEventListener("click", () => {
      runPrediction(core.name);
    });
    dom.individualBtnsContainer.appendChild(btn);
  });
}

/**
 * Mark current prediction as outdated when any power input changes:
 * Clears stale chart points, table rows, and heatmap colors until the user predicts again.
 */
function markPredictionOutdated() {
  ++AppState.predictionRequestId;
  const bundle = AppState.currentFloorplan;
  if (!bundle) return;

  AppState.lastPrediction = null;
  AppState.isOutdated = true;
  dom.staleBanner.classList.remove("hidden");

  clearStalePredictionDisplay(
    bundle,
    "Power inputs modified — click Predict All Cores or Predict Core 0–7 to update."
  );
}

/**
 * Clear stale KPI values, temperature table rows, heatmap coloring, and graph points.
 */
function clearStalePredictionDisplay(bundle, statusMessage) {
  updateActivePredictionButtons(null);

  dom.heatmapModeBadge.textContent = "Awaiting Prediction";
  dom.resultBannerTitle.textContent = `Prediction Summary — ${bundle.full_name}`;
  dom.resultModePill.textContent = AppState.isOutdated ? "OUTDATED" : "READY";
  dom.kpiTempLabel.textContent = "Maximum Predicted Temp";
  dom.kpiTempC.textContent = "-- °C";
  dom.kpiTempK.textContent = "-- K";
  dom.kpiCoreLabel.textContent = "Hottest / Target Core";
  dom.kpiCoreName.textContent = "--";
  dom.kpiCoreType.textContent = `${bundle.num_big_cores} Big, ${bundle.num_little_cores} Little`;
  dom.kpiTotalPower.textContent = "-- W";
  dom.kpiPredTime.textContent = "-- µs";
  dom.kpiRoundtripTime.textContent = AppState.isOutdated ? "Stale (inputs modified)" : "Awaiting prediction";
  const perFpParams = bundle.model_parameters_per_floorplan || 64;
  const totalParams = bundle.total_model_parameters_all_floorplans || 1472;
  const totalRawBytes = bundle.total_raw_float64_bytes_all_floorplans || 11776;
  const totalCsvBytes = bundle.total_csv_size_bytes_all_floorplans || 17733;
  dom.kpiModelSize.textContent =
    `Per floorplan: ${perFpParams} params, ${bundle.model_raw_float64_bytes} B raw float64 (${bundle.model_csv_size_bytes} B CSV) · ` +
    `All 23 floorplans: ${totalParams.toLocaleString()} params, ${totalRawBytes.toLocaleString()} B raw (${totalCsvBytes.toLocaleString()} B CSV)`;
  dom.kpiMatrixPath.textContent = bundle.matrix_file;

  dom.tableCardTitle.textContent = "Predicted Core Temperatures";
  dom.tableRowCount.textContent = "No active prediction";
  dom.couplingSection.classList.add("hidden");

  dom.resultsTableBody.innerHTML = `
    <tr class="empty-table-row">
      <td colspan="6">${statusMessage}</td>
    </tr>
  `;

  renderFloorplanSvg(bundle, null);
  renderCoreTempChart(bundle, null);
}

/**
 * Validate a single power input element and update inline error styling.
 */
function validateSingleInputElement(inputEl) {
  const coreName = inputEl.dataset.coreName;
  const coreNum = inputEl.dataset.coreNum;
  const minW = parseFloat(inputEl.dataset.min);
  const maxW = parseFloat(inputEl.dataset.max);
  const coreType = inputEl.dataset.type;
  const rawVal = inputEl.value.trim();
  const rowEl = document.getElementById(`power-row-${coreName}`);
  const errEl = document.getElementById(`err-power-${coreName}`);

  if (rawVal === "") {
    const msg = `Core ${coreNum} (${coreType}): Enter a power value between ${minW.toFixed(0)} and ${maxW.toFixed(0)} W.`;
    rowEl.classList.add("row-invalid");
    errEl.textContent = msg;
    errEl.classList.remove("hidden");
    return { valid: false, error: msg };
  }

  const val = Number(rawVal);
  if (Number.isNaN(val) || !Number.isFinite(val)) {
    const msg = `Core ${coreNum} (${coreType}): Invalid number '${rawVal}'. Allowed: ${minW.toFixed(0)}–${maxW.toFixed(0)} W.`;
    rowEl.classList.add("row-invalid");
    errEl.textContent = msg;
    errEl.classList.remove("hidden");
    return { valid: false, error: msg };
  }

  if (val < minW - 1e-9 || val > maxW + 1e-9) {
    const msg = `Core ${coreNum} (${coreType}): ${val} W is outside ${minW.toFixed(0)}–${maxW.toFixed(0)} W.`;
    rowEl.classList.add("row-invalid");
    errEl.textContent = msg;
    errEl.classList.remove("hidden");
    return { valid: false, error: msg };
  }

  rowEl.classList.remove("row-invalid");
  errEl.textContent = "";
  errEl.classList.add("hidden");
  return { valid: true, value: val };
}

/**
 * Collect and validate all 8 core powers in exact matrix column ordering.
 */
function collectAndValidatePowers() {
  const bundle = AppState.currentFloorplan;
  if (!bundle) return { valid: false, errors: ["No floorplan loaded."] };

  const powersInMatrixOrder = [];
  const powersByCore = {};
  const errors = [];

  for (const core of bundle.cores) {
    const inputEl = document.getElementById(`input-power-${core.name}`);
    const check = validateSingleInputElement(inputEl);
    if (!check.valid) {
      errors.push(check.error);
    } else {
      powersInMatrixOrder.push(check.value);
      powersByCore[core.name] = check.value;
    }
  }

  if (errors.length > 0) {
    showValidationError(errors[0]);
    return { valid: false, errors };
  }

  clearValidationErrors();
  return { valid: true, powers: powersInMatrixOrder, powersByCore };
}

function showValidationError(msg) {
  dom.errorBanner.textContent = msg;
  dom.errorBanner.classList.remove("hidden");
}

function clearValidationErrors() {
  dom.errorBanner.textContent = "";
  dom.errorBanner.classList.add("hidden");
  document.querySelectorAll(".power-row.row-invalid").forEach((el) => el.classList.remove("row-invalid"));
  document.querySelectorAll(".power-field-error").forEach((el) => {
    el.textContent = "";
    el.classList.add("hidden");
  });
}

/**
 * Execute thermal prediction via POST /api/predict for either "all" or a specific core name.
 */
async function runPrediction(target) {
  const requestId = ++AppState.predictionRequestId;
  const floorplanRequestId = AppState.floorplanRequestId;
  const bundle = AppState.currentFloorplan;
  if (!bundle || AppState.loadingFloorplan) return;
  const isCurrentRequest = () => (
    requestId === AppState.predictionRequestId &&
    floorplanRequestId === AppState.floorplanRequestId &&
    bundle === AppState.currentFloorplan &&
    !AppState.loadingFloorplan
  );

  const collected = collectAndValidatePowers();
  if (!collected.valid) return;

  AppState.currentMode = target;
  AppState.isOutdated = false;
  dom.staleBanner.classList.add("hidden");
  updateActivePredictionButtons(target);

  const tClientStart = performance.now();
  try {
    const res = await fetch("/api/predict", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        floorplan: bundle.full_name,
        powers: collected.powers,
        powers_by_core: collected.powersByCore,
        target: target,
      }),
    });
    const data = await res.json();
    if (!isCurrentRequest()) return;
    const roundtripMs = performance.now() - tClientStart;

    if (!res.ok) {
      if (data.field_errors) {
        for (const [cName, errMsg] of Object.entries(data.field_errors)) {
          const rowEl = document.getElementById(`power-row-${cName}`);
          const errEl = document.getElementById(`err-power-${cName}`);
          if (rowEl) rowEl.classList.add("row-invalid");
          if (errEl) {
            errEl.textContent = errMsg;
            errEl.classList.remove("hidden");
          }
        }
      }
      showValidationError(data.error || "Prediction request failed.");
      return;
    }

    AppState.lastPrediction = data;
    renderPredictionResults(bundle, data, roundtripMs);
    renderFloorplanSvg(bundle, data);
    renderCoreTempChart(bundle, data);
  } catch (err) {
    if (!isCurrentRequest()) return;
    showValidationError(`Network error calling /api/predict: ${err.message}`);
  }
}

/**
 * Highlight the active prediction button ("Predict All Cores" or "Predict Core k")
 * and highlight the corresponding power row when in single-core mode.
 */
function updateActivePredictionButtons(target) {
  if (target === "all") {
    dom.btnPredictAll.classList.add("active-mode");
  } else {
    dom.btnPredictAll.classList.remove("active-mode");
  }

  document.querySelectorAll(".btn-predict-core").forEach((btn) => {
    if (target && btn.dataset.coreName === target) {
      btn.classList.add("active-mode");
    } else {
      btn.classList.remove("active-mode");
    }
  });

  document.querySelectorAll(".power-row").forEach((row) => {
    if (target && row.dataset.coreName === target) {
      row.classList.add("row-selected");
    } else {
      row.classList.remove("row-selected");
    }
  });
}

/**
 * 3. Render the geometrically proportional 2D silicon floorplan SVG with zero text overlap.
 *
 * Inside each active core rectangle, shows ONLY:
 *   - Short core label: "Core k"
 *   - Predicted temperature when available: e.g. "136.3°C"
 * All detailed info (full core name, Big/Little type, dimensions, X/Y coordinates, input power,
 * and 6-decimal temperature) is displayed in the hover tooltip.
 */
function renderFloorplanSvg(bundle, predictionData) {
  const svg = dom.floorplanSvg;
  svg.innerHTML = "";

  const svgW = 720;
  const svgH = 410;
  svg.setAttribute("viewBox", `0 0 ${svgW} ${svgH}`);

  const padLeft = 58;
  const padRight = 30;
  const padTop = 26;
  const padBottom = 44;

  const availW = svgW - padLeft - padRight;
  const availH = svgH - padTop - padBottom;

  const dieW = bundle.die_width_mm;
  const dieH = bundle.die_height_mm;

  // Single uniform scale (pixels per millimeter) for BOTH X and Y axes
  const pxPerMm = Math.min(availW / dieW, availH / dieH);
  const drawnDieW = dieW * pxPerMm;
  const drawnDieH = dieH * pxPerMm;

  const originX = padLeft + (availW - drawnDieW) / 2;
  const originY = padTop + (availH - drawnDieH) / 2;

  const ns = "http://www.w3.org/2000/svg";

  // 1. SVG Defs: hatch pattern for passive FILL blocks + per-core clipPaths to guarantee zero text overflow
  const defs = document.createElementNS(ns, "defs");
  let defsHtml = `
    <pattern id="fillHatchPattern" width="10" height="10" patternTransform="rotate(45 0 0)" patternUnits="userSpaceOnUse">
      <rect width="10" height="10" fill="#111827"></rect>
      <line x1="0" y1="0" x2="0" y2="10" stroke="#334155" stroke-width="2"></line>
    </pattern>
  `;

  const toSvgRect = (leftX, bottomY, widthMm, heightMm) => ({
    x: originX + leftX * pxPerMm,
    y: originY + (dieH - (bottomY + heightMm)) * pxPerMm,
    w: widthMm * pxPerMm,
    h: heightMm * pxPerMm,
  });

  bundle.cores.forEach((core) => {
    const r = toSvgRect(core.left_x_mm, core.bottom_y_mm, core.width_mm, core.height_mm);
    defsHtml += `
      <clipPath id="clip-core-${core.core_number}">
        <rect x="${r.x + 2}" y="${r.y + 2}" width="${Math.max(2, r.w - 4)}" height="${Math.max(2, r.h - 4)}" rx="3"></rect>
      </clipPath>
    `;
  });
  defs.innerHTML = defsHtml;
  svg.appendChild(defs);

  // 2. Draw X and Y millimeter rulers
  const axesGroup = document.createElementNS(ns, "g");
  const xStep = dieW > 12 ? 2 : 1;
  for (let xm = 0; xm <= dieW + 1e-6; xm += xStep) {
    const sx = originX + xm * pxPerMm;
    const tick = document.createElementNS(ns, "line");
    tick.setAttribute("x1", sx);
    tick.setAttribute("y1", originY + drawnDieH);
    tick.setAttribute("x2", sx);
    tick.setAttribute("y2", originY + drawnDieH + 5);
    tick.setAttribute("stroke", "#64748b");
    tick.setAttribute("stroke-width", "1");
    axesGroup.appendChild(tick);

    const lbl = document.createElementNS(ns, "text");
    lbl.setAttribute("x", sx);
    lbl.setAttribute("y", originY + drawnDieH + 17);
    lbl.setAttribute("text-anchor", "middle");
    lbl.setAttribute("fill", "#94a3b8");
    lbl.setAttribute("font-size", "10.5");
    lbl.setAttribute("font-family", "monospace");
    lbl.textContent = `${xm}`;
    axesGroup.appendChild(lbl);
  }

  const yStep = dieH > 12 ? 2 : 1;
  for (let ym = 0; ym <= dieH + 1e-6; ym += yStep) {
    const sy = originY + (dieH - ym) * pxPerMm;
    const tick = document.createElementNS(ns, "line");
    tick.setAttribute("x1", originX - 5);
    tick.setAttribute("y1", sy);
    tick.setAttribute("x2", originX);
    tick.setAttribute("y2", sy);
    tick.setAttribute("stroke", "#64748b");
    tick.setAttribute("stroke-width", "1");
    axesGroup.appendChild(tick);

    const lbl = document.createElementNS(ns, "text");
    lbl.setAttribute("x", originX - 9);
    lbl.setAttribute("y", sy + 3.5);
    lbl.setAttribute("text-anchor", "end");
    lbl.setAttribute("fill", "#94a3b8");
    lbl.setAttribute("font-size", "10.5");
    lbl.setAttribute("font-family", "monospace");
    lbl.textContent = `${ym}`;
    axesGroup.appendChild(lbl);
  }

  const xAxisTitle = document.createElementNS(ns, "text");
  xAxisTitle.setAttribute("x", originX + drawnDieW / 2);
  xAxisTitle.setAttribute("y", originY + drawnDieH + 34);
  xAxisTitle.setAttribute("text-anchor", "middle");
  xAxisTitle.setAttribute("fill", "#cbd5e1");
  xAxisTitle.setAttribute("font-size", "11.5");
  xAxisTitle.setAttribute("font-weight", "600");
  xAxisTitle.textContent = `X (mm) — Die: ${dieW.toFixed(1)} × ${dieH.toFixed(1)} mm`;
  axesGroup.appendChild(xAxisTitle);

  const yAxisTitle = document.createElementNS(ns, "text");
  const yCenter = originY + drawnDieH / 2;
  yAxisTitle.setAttribute("x", originX - 34);
  yAxisTitle.setAttribute("y", yCenter);
  yAxisTitle.setAttribute("text-anchor", "middle");
  yAxisTitle.setAttribute("fill", "#cbd5e1");
  yAxisTitle.setAttribute("font-size", "11.5");
  yAxisTitle.setAttribute("font-weight", "600");
  yAxisTitle.setAttribute("transform", `rotate(-90, ${originX - 34}, ${yCenter})`);
  yAxisTitle.textContent = `Y (mm)`;
  axesGroup.appendChild(yAxisTitle);

  svg.appendChild(axesGroup);

  // 3. Outer Silicon Die Boundary
  const dieOutline = document.createElementNS(ns, "rect");
  dieOutline.setAttribute("x", originX);
  dieOutline.setAttribute("y", originY);
  dieOutline.setAttribute("width", drawnDieW);
  dieOutline.setAttribute("height", drawnDieH);
  dieOutline.setAttribute("fill", "#0b1120");
  dieOutline.setAttribute("stroke", "#94a3b8");
  dieOutline.setAttribute("stroke-width", "2");
  svg.appendChild(dieOutline);

  // 4. Render Passive FILL Blocks (clean neutral substrate regions without cluttered text)
  bundle.fill_blocks.forEach((fillBlk) => {
    const r = toSvgRect(fillBlk.left_x_mm, fillBlk.bottom_y_mm, fillBlk.width_mm, fillBlk.height_mm);
    const g = document.createElementNS(ns, "g");

    const rect = document.createElementNS(ns, "rect");
    rect.setAttribute("x", r.x);
    rect.setAttribute("y", r.y);
    rect.setAttribute("width", r.w);
    rect.setAttribute("height", r.h);
    rect.setAttribute("fill", "url(#fillHatchPattern)");
    rect.setAttribute("stroke", "#475569");
    rect.setAttribute("stroke-width", "1");
    rect.setAttribute("stroke-dasharray", "4 2");
    g.appendChild(rect);

    g.addEventListener("mousemove", (ev) => {
      showFloatingTooltip(ev, dom.chipStageWrapper, dom.chipTooltip, {
        title: `${fillBlk.name} (Passive FILL)`,
        rows: [
          ["Type", "Passive Silicon Substrate"],
          ["Size", `${fillBlk.width_mm.toFixed(1)} × ${fillBlk.height_mm.toFixed(1)} mm (${fillBlk.area_mm2.toFixed(1)} mm²)`],
          ["Position (X, Y)", `(${fillBlk.left_x_mm.toFixed(1)}, ${fillBlk.bottom_y_mm.toFixed(1)}) mm`],
          ["Power", "0.00 W"],
        ],
      });
    });
    g.addEventListener("mouseleave", () => hideFloatingTooltip(dom.chipTooltip));

    svg.appendChild(g);
  });

  // 5. Determine Heatmap Scale for Active Cores
  let minScaleC = 105.0;
  let maxScaleC = 175.0;
  const isAllMode = predictionData && predictionData.mode === "all";
  const isSingleMode = predictionData && predictionData.mode === "single";

  if (isAllMode) {
    minScaleC = predictionData.min_temp_c;
    maxScaleC = predictionData.max_temp_c;
    if (Math.abs(maxScaleC - minScaleC) < 0.5) {
      minScaleC -= 1.0;
      maxScaleC += 1.0;
    }
  } else if (isSingleMode && predictionData.selected_core) {
    const selT = predictionData.selected_core.temp_c;
    minScaleC = Math.min(105.0, Math.floor(selT - 10));
    maxScaleC = Math.max(175.0, Math.ceil(selT + 10));
  }

  updateHeatmapLegend(minScaleC, maxScaleC, predictionData);

  // Determine which core is hottest across all 8 cores even when in single-core mode
  let singleModeHottestCore = null;
  if (
    isSingleMode &&
    Array.isArray(bundle.thermal_matrix_k_per_w) &&
    Array.isArray(predictionData.powers_w) &&
    predictionData.powers_w.length === bundle.cores.length
  ) {
    let bestT = -Infinity;
    bundle.thermal_matrix_k_per_w.forEach((rowVals, rIdx) => {
      const dt = rowVals.reduce((acc, r_ij, cIdx) => acc + r_ij * predictionData.powers_w[cIdx], 0);
      const tC = 45.0 + dt;
      if (tC > bestT) {
        bestT = tC;
        singleModeHottestCore = bundle.matrix_core_order[rIdx];
      }
    });
  }

  // 6. Render Active Cores (Big 4x4 mm and Little 2x2 mm)
  bundle.cores.forEach((core) => {
    const r = toSvgRect(core.left_x_mm, core.bottom_y_mm, core.width_mm, core.height_mm);
    const predEntry =
      predictionData && predictionData.predictions_by_core
        ? predictionData.predictions_by_core[core.name]
        : null;

    const inputEl = document.getElementById(`input-power-${core.name}`);
    const currentPowerW =
      predictionData && predictionData.powers_by_core_w && predictionData.powers_by_core_w[core.name] !== undefined
        ? predictionData.powers_by_core_w[core.name]
        : inputEl
        ? parseFloat(inputEl.value) || core.default_power_w
        : core.default_power_w;

    const isSelectedSingle = Boolean(isSingleMode && predEntry && predEntry.is_selected);
    const isHottest = Boolean(
      (isAllMode && predEntry && predEntry.is_hottest) ||
        (isSingleMode && predEntry && singleModeHottestCore === core.name)
    );

    // Categorical core-type colors (Big Core = orange/yellow, Little Core = cyan/blue)
    const isBig = core.short_type === "Big";
    const catFill = isBig ? "rgba(245, 158, 11, 0.34)" : "rgba(56, 189, 248, 0.30)";
    const catStroke = isBig ? "#fbbf24" : "#38bdf8";

    let fillColor = catFill;
    let fillOpacity = isSingleMode && !predEntry ? "0.45" : "0.85";
    let strokeColor = catStroke;
    let strokeWidth = isSingleMode && !predEntry ? "1.4" : "1.8";

    if (predEntry) {
      const u = (predEntry.temp_c - minScaleC) / (maxScaleC - minScaleC || 1);
      fillColor = interpolateHeatmapColor(u);
      fillOpacity = "0.90";
      if (isHottest) {
        strokeColor = "#facc15";
        strokeWidth = "3.0";
      } else if (isSelectedSingle) {
        strokeColor = "#22d3ee";
        strokeWidth = "2.8";
      } else {
        strokeColor = catStroke;
        strokeWidth = "1.8";
      }
    }

    const g = document.createElementNS(ns, "g");
    g.style.cursor = "pointer";

    const rect = document.createElementNS(ns, "rect");
    rect.setAttribute("x", r.x + 1);
    rect.setAttribute("y", r.y + 1);
    rect.setAttribute("width", Math.max(2, r.w - 2));
    rect.setAttribute("height", Math.max(2, r.h - 2));
    rect.setAttribute("rx", "4");
    rect.setAttribute("fill", fillColor);
    rect.setAttribute("fill-opacity", fillOpacity);
    rect.setAttribute("stroke", strokeColor);
    rect.setAttribute("stroke-width", strokeWidth);
    g.appendChild(rect);

    // Distinct Cyan/White outer selection outline when core is Selected (works alongside Gold Hottest border if both)
    if (isSelectedSingle) {
      const selRing = document.createElementNS(ns, "rect");
      selRing.setAttribute("class", "core-outline-selected");
      selRing.setAttribute("x", r.x - 1);
      selRing.setAttribute("y", r.y - 1);
      selRing.setAttribute("width", Math.max(4, r.w + 2));
      selRing.setAttribute("height", Math.max(4, r.h + 2));
      selRing.setAttribute("rx", "5");
      selRing.setAttribute("fill", "none");
      selRing.setAttribute("stroke", isHottest ? "#22d3ee" : "#ffffff");
      selRing.setAttribute("stroke-width", isHottest ? "2.0" : "1.4");
      g.appendChild(selRing);
    }

    // Clipped interior group for labels & corner dots so elements NEVER overflow the core rectangle
    const textGroup = document.createElementNS(ns, "g");
    textGroup.setAttribute("clip-path", `url(#clip-core-${core.core_number})`);

    // Selected Core visual indicator: top-left cyan/white corner dot
    if (isSelectedSingle) {
      const selDot = document.createElementNS(ns, "circle");
      selDot.setAttribute("class", "core-marker-selected");
      selDot.setAttribute("cx", r.x + 8);
      selDot.setAttribute("cy", r.y + 8);
      selDot.setAttribute("r", "3.3");
      selDot.setAttribute("fill", "#22d3ee");
      selDot.setAttribute("stroke", "#ffffff");
      selDot.setAttribute("stroke-width", "1.1");
      textGroup.appendChild(selDot);
    }

    // Hottest Core visual indicator: top-right gold/yellow corner marker dot
    if (isHottest) {
      const hotDot = document.createElementNS(ns, "circle");
      hotDot.setAttribute("class", "core-marker-hottest");
      hotDot.setAttribute("cx", r.x + r.w - 8);
      hotDot.setAttribute("cy", r.y + 8);
      hotDot.setAttribute("r", "3.3");
      hotDot.setAttribute("fill", "#facc15");
      hotDot.setAttribute("stroke", "#713f12");
      hotDot.setAttribute("stroke-width", "1.0");
      textGroup.appendChild(hotDot);
    }

    const cx = r.x + r.w / 2;
    const cy = r.y + r.h / 2;
    const minDim = Math.min(r.w, r.h);

    // Responsive font size scaled to rectangle size (works cleanly from 44px Little cores up to 130px Big cores)
    const labelFontSize = Math.max(9.5, Math.min(13.5, minDim * 0.22));
    const tempFontSize = Math.max(9.0, Math.min(13.0, minDim * 0.21));

    const shortLabel = `Core ${core.core_number}`;

    if (predEntry) {
      // Show ONLY short core label ("Core k") and predicted temperature ("136.3°C")
      const nameEl = document.createElementNS(ns, "text");
      nameEl.setAttribute("x", cx);
      nameEl.setAttribute("y", cy - tempFontSize * 0.35);
      nameEl.setAttribute("text-anchor", "middle");
      nameEl.setAttribute("fill", "#ffffff");
      nameEl.setAttribute("font-size", labelFontSize.toFixed(1));
      nameEl.setAttribute("font-weight", "800");
      nameEl.setAttribute("font-family", "sans-serif");
      nameEl.textContent = shortLabel;
      textGroup.appendChild(nameEl);

      const tempEl = document.createElementNS(ns, "text");
      tempEl.setAttribute("x", cx);
      tempEl.setAttribute("y", cy + tempFontSize * 0.82);
      tempEl.setAttribute("text-anchor", "middle");
      tempEl.setAttribute("fill", "#fef08a");
      tempEl.setAttribute("font-size", tempFontSize.toFixed(1));
      tempEl.setAttribute("font-weight", "700");
      tempEl.setAttribute("font-family", "monospace");
      const decimals = minDim < 58 ? 1 : 2;
      tempEl.textContent = `${predEntry.temp_c.toFixed(decimals)}°C`;
      textGroup.appendChild(tempEl);
    } else {
      // Unpredicted core: show ONLY its short label centered
      const nameEl = document.createElementNS(ns, "text");
      nameEl.setAttribute("x", cx);
      nameEl.setAttribute("y", cy + labelFontSize * 0.35);
      nameEl.setAttribute("text-anchor", "middle");
      nameEl.setAttribute("fill", "#f8fafc");
      nameEl.setAttribute("font-size", labelFontSize.toFixed(1));
      nameEl.setAttribute("font-weight", "700");
      nameEl.setAttribute("font-family", "sans-serif");
      nameEl.textContent = shortLabel;
      textGroup.appendChild(nameEl);
    }

    g.appendChild(textGroup);

    // Detailed Hover Tooltip
    g.addEventListener("mousemove", (ev) => {
      const statusBadges = [];
      if (isSelectedSingle) statusBadges.push("Selected");
      if (isHottest) statusBadges.push("Hottest");
      const statusSuffix = statusBadges.length > 0 ? ` — ${statusBadges.join(" & ")}` : "";

      const rows = [
        ["Full Core Name", core.name],
        ["Core Type", core.core_type],
        ["Dimensions", `${core.width_mm.toFixed(1)} × ${core.height_mm.toFixed(1)} mm (${core.area_mm2.toFixed(1)} mm²)`],
        ["Coordinates (X, Y)", `(${core.left_x_mm.toFixed(1)}, ${core.bottom_y_mm.toFixed(1)}) mm`],
        ["Input Power", `${currentPowerW.toFixed(2)} W (Allowed: ${core.min_power_w}–${core.max_power_w} W)`],
      ];
      if (predEntry) {
        rows.push(["Predicted Temp (°C)", `${predEntry.temp_c.toFixed(2)} °C (${predEntry.temp_c.toFixed(6)} °C)`]);
        rows.push(["Predicted Temp (K)", `${predEntry.temp_k.toFixed(2)} K (${predEntry.temp_k.toFixed(6)} K)`]);
        rows.push(["Temp Rise (ΔT)", `+${predEntry.delta_t_c.toFixed(2)} °C above 45°C`]);
      } else {
        rows.push(["Predicted Temp", "Not calculated yet (showing categorical core type color)"]);
      }
      showFloatingTooltip(ev, dom.chipStageWrapper, dom.chipTooltip, {
        title: `Core ${core.core_number} (${core.name})${statusSuffix}`,
        rows,
      });
    });
    g.addEventListener("mouseleave", () => hideFloatingTooltip(dom.chipTooltip));

    g.addEventListener("click", () => {
      runPrediction(core.name);
    });

    svg.appendChild(g);
  });
}

/**
 * 4. Render the "Core vs Predicted Temperature" Graph below the floorplan.
 *
 * - X-axis ALWAYS shows all 8 labels: Core 0, Core 1, Core 2, Core 3, Core 4, Core 5, Core 6, Core 7.
 * - Y-axis shows Predicted Temperature (°C).
 * - Predict All: plots all 8 points at Core 0..Core 7, connects them with a line, highlights hottest point.
 * - Predict Core k: plots ONLY Core k at X-axis slot k, hides the other 7 points & hides the connecting line.
 * - Cleared/Outdated: shows axes with Core 0..Core 7 and a centered prompt without stale points.
 */
function renderCoreTempChart(bundle, predictionData) {
  const svg = dom.coreTempChartSvg;
  svg.innerHTML = "";

  const svgW = 720;
  const svgH = 230;
  svg.setAttribute("viewBox", `0 0 ${svgW} ${svgH}`);

  const padLeft = 64;
  const padRight = 32;
  const padTop = 28;
  const padBottom = 42;

  const plotW = svgW - padLeft - padRight;
  const plotH = svgH - padTop - padBottom;

  const ns = "http://www.w3.org/2000/svg";
  const xLabels = ["Core 0", "Core 1", "Core 2", "Core 3", "Core 4", "Core 5", "Core 6", "Core 7"];
  const xForIndex = (idx) => padLeft + (idx / 7) * plotW;

  // Build 8-element data array indexed by core_number (0..7)
  const pointsByCoreNum = new Array(8).fill(null);
  const isAllMode = Boolean(predictionData && predictionData.mode === "all");
  const isSingleMode = Boolean(predictionData && predictionData.mode === "single");

  if (isAllMode && Array.isArray(predictionData.predictions)) {
    predictionData.predictions.forEach((pred) => {
      if (pred.core_number >= 0 && pred.core_number < 8) {
        pointsByCoreNum[pred.core_number] = pred;
      }
    });
    dom.chartStatusBadge.textContent = "All 8 Cores Plotted";
    dom.chartSubtitle.textContent = `Steady-state temperature profile across Core 0–Core 7 (${bundle.full_name})`;
  } else if (isSingleMode && predictionData.selected_core) {
    const sel = predictionData.selected_core;
    if (sel.core_number >= 0 && sel.core_number < 8) {
      pointsByCoreNum[sel.core_number] = sel;
    }
    dom.chartStatusBadge.textContent = `Core ${sel.core_number} Only`;
    dom.chartSubtitle.textContent = `Single-core steady-state prediction for Core ${sel.core_number} (${sel.core_name})`;
  } else {
    dom.chartStatusBadge.textContent = AppState.isOutdated ? "Outdated — Click Predict" : "Ready";
    dom.chartSubtitle.textContent = "Run Predict All Cores or Predict Core 0–7 to plot temperatures";
  }

  // Determine Y-axis domain (°C) from actual backend values
  const activeVals = pointsByCoreNum.filter((p) => p !== null).map((p) => p.temp_c);
  let yMin = 100;
  let yMax = 160;
  if (activeVals.length > 1) {
    const minV = Math.min(...activeVals);
    const maxV = Math.max(...activeVals);
    const span = Math.max(4, maxV - minV);
    yMin = Math.floor((minV - span * 0.22) / 2) * 2;
    yMax = Math.ceil((maxV + span * 0.28) / 2) * 2;
  } else if (activeVals.length === 1) {
    const v = activeVals[0];
    yMin = Math.floor((v - 10) / 5) * 5;
    yMax = Math.ceil((v + 10) / 5) * 5;
  }

  const yForTemp = (tempC) => padTop + (1 - (tempC - yMin) / (yMax - yMin || 1)) * plotH;

  // 1. Horizontal Y gridlines and tick labels
  const gridGroup = document.createElementNS(ns, "g");
  const numYTicks = 4;
  for (let i = 0; i <= numYTicks; i++) {
    const val = yMin + (i / numYTicks) * (yMax - yMin);
    const py = yForTemp(val);

    const line = document.createElementNS(ns, "line");
    line.setAttribute("x1", padLeft);
    line.setAttribute("y1", py);
    line.setAttribute("x2", padLeft + plotW);
    line.setAttribute("y2", py);
    line.setAttribute("stroke", i === 0 ? "#475569" : "#1e293b");
    line.setAttribute("stroke-width", i === 0 ? "1.5" : "1");
    gridGroup.appendChild(line);

    const lbl = document.createElementNS(ns, "text");
    lbl.setAttribute("x", padLeft - 8);
    lbl.setAttribute("y", py + 4);
    lbl.setAttribute("text-anchor", "end");
    lbl.setAttribute("fill", "#94a3b8");
    lbl.setAttribute("font-size", "10.5");
    lbl.setAttribute("font-family", "monospace");
    lbl.textContent = `${val.toFixed(1)}°`;
    gridGroup.appendChild(lbl);
  }

  // Y-axis title
  const yTitle = document.createElementNS(ns, "text");
  const yMid = padTop + plotH / 2;
  yTitle.setAttribute("x", 16);
  yTitle.setAttribute("y", yMid);
  yTitle.setAttribute("text-anchor", "middle");
  yTitle.setAttribute("fill", "#cbd5e1");
  yTitle.setAttribute("font-size", "11");
  yTitle.setAttribute("font-weight", "600");
  yTitle.setAttribute("transform", `rotate(-90, 16, ${yMid})`);
  yTitle.textContent = "Temp (°C)";
  gridGroup.appendChild(yTitle);

  // 2. X-axis ticks and all 8 labels (Core 0 .. Core 7 always visible)
  xLabels.forEach((labelText, idx) => {
    const px = xForIndex(idx);

    const vLine = document.createElementNS(ns, "line");
    vLine.setAttribute("x1", px);
    vLine.setAttribute("y1", padTop);
    vLine.setAttribute("x2", px);
    vLine.setAttribute("y2", padTop + plotH);
    vLine.setAttribute("stroke", "#1e293b");
    vLine.setAttribute("stroke-width", "1");
    vLine.setAttribute("stroke-dasharray", "3 3");
    gridGroup.appendChild(vLine);

    const tick = document.createElementNS(ns, "line");
    tick.setAttribute("x1", px);
    tick.setAttribute("y1", padTop + plotH);
    tick.setAttribute("x2", px);
    tick.setAttribute("y2", padTop + plotH + 5);
    tick.setAttribute("stroke", "#64748b");
    tick.setAttribute("stroke-width", "1.2");
    gridGroup.appendChild(tick);

    const hasPoint = pointsByCoreNum[idx] !== null;
    const xLbl = document.createElementNS(ns, "text");
    xLbl.setAttribute("x", px);
    xLbl.setAttribute("y", padTop + plotH + 20);
    xLbl.setAttribute("text-anchor", "middle");
    xLbl.setAttribute("fill", hasPoint ? "#f8fafc" : "#94a3b8");
    xLbl.setAttribute("font-size", "11");
    xLbl.setAttribute("font-weight", hasPoint ? "700" : "500");
    xLbl.setAttribute("font-family", "sans-serif");
    xLbl.textContent = labelText;
    gridGroup.appendChild(xLbl);
  });

  svg.appendChild(gridGroup);

  // If no prediction is active (cleared on floorplan switch or outdated after power edit), show prompt
  if (activeVals.length === 0) {
    const msg = document.createElementNS(ns, "text");
    msg.setAttribute("x", padLeft + plotW / 2);
    msg.setAttribute("y", padTop + plotH / 2 + 4);
    msg.setAttribute("text-anchor", "middle");
    msg.setAttribute("fill", "#64748b");
    msg.setAttribute("font-size", "12.5");
    msg.textContent = AppState.isOutdated
      ? "Power inputs changed — click Predict to update graph"
      : "Click Predict All Cores or Predict Core 0–7 to plot temperatures";
    svg.appendChild(msg);
    return;
  }

  // 3. In "Predict All" mode, connect the 8 points with a line
  if (isAllMode) {
    const coords = pointsByCoreNum.map((pt, idx) => `${xForIndex(idx)},${yForTemp(pt.temp_c)}`);
    const areaCoords = [
      `${xForIndex(0)},${padTop + plotH}`,
      ...coords,
      `${xForIndex(7)},${padTop + plotH}`,
    ].join(" ");

    const areaPoly = document.createElementNS(ns, "polygon");
    areaPoly.setAttribute("points", areaCoords);
    areaPoly.setAttribute("fill", "rgba(56, 189, 248, 0.10)");
    svg.appendChild(areaPoly);

    const polyline = document.createElementNS(ns, "polyline");
    polyline.setAttribute("points", coords.join(" "));
    polyline.setAttribute("fill", "none");
    polyline.setAttribute("stroke", "#38bdf8");
    polyline.setAttribute("stroke-width", "2.4");
    svg.appendChild(polyline);
  }

  // 4. Draw plotted data point(s) (8 points in Predict All; 1 point at exact X slot in Single mode)
  pointsByCoreNum.forEach((pt, idx) => {
    if (!pt) return;
    const px = xForIndex(idx);
    const py = yForTemp(pt.temp_c);
    const isHottest = Boolean(isAllMode && pt.is_hottest);

    const ptGroup = document.createElementNS(ns, "g");
    ptGroup.style.cursor = "pointer";

    // Outer highlight ring for hottest or single selected core
    if (isHottest || isSingleMode) {
      const ring = document.createElementNS(ns, "circle");
      ring.setAttribute("cx", px);
      ring.setAttribute("cy", py);
      ring.setAttribute("r", "9");
      ring.setAttribute("fill", isHottest ? "rgba(239, 68, 68, 0.25)" : "rgba(56, 189, 248, 0.25)");
      ring.setAttribute("stroke", isHottest ? "#fef08a" : "#38bdf8");
      ring.setAttribute("stroke-width", "1.5");
      ptGroup.appendChild(ring);
    }

    const circle = document.createElementNS(ns, "circle");
    circle.setAttribute("cx", px);
    circle.setAttribute("cy", py);
    circle.setAttribute("r", isHottest || isSingleMode ? "5.5" : "4.5");
    circle.setAttribute("fill", isHottest ? "#ef4444" : "#38bdf8");
    circle.setAttribute("stroke", "#ffffff");
    circle.setAttribute("stroke-width", "1.8");
    ptGroup.appendChild(circle);

    // Value label directly above the point
    const valLbl = document.createElementNS(ns, "text");
    valLbl.setAttribute("x", px);
    valLbl.setAttribute("y", Math.max(14, py - 11));
    valLbl.setAttribute("text-anchor", "middle");
    valLbl.setAttribute("fill", isHottest ? "#fef08a" : "#e2e8f0");
    valLbl.setAttribute("font-size", "10.5");
    valLbl.setAttribute("font-weight", "700");
    valLbl.setAttribute("font-family", "monospace");
    valLbl.textContent = `${pt.temp_c.toFixed(2)}°C`;
    ptGroup.appendChild(valLbl);

    ptGroup.addEventListener("mousemove", (ev) => {
      showFloatingTooltip(ev, dom.chartStageWrapper, dom.chartTooltip, {
        title: `Core ${pt.core_number} (${pt.core_name})${isHottest ? " — Hottest Core" : ""}`,
        rows: [
          ["Core Type", pt.core_type],
          ["Input Power", `${pt.power_w.toFixed(2)} W`],
          ["Temp Rise (ΔT)", `+${pt.delta_t_c.toFixed(2)} °C`],
          ["Predicted Temp (°C)", `${pt.temp_c.toFixed(2)} °C (${pt.temp_c.toFixed(6)} °C)`],
          ["Predicted Temp (K)", `${pt.temp_k.toFixed(2)} K (${pt.temp_k.toFixed(6)} K)`],
        ],
      });
    });
    ptGroup.addEventListener("mouseleave", () => hideFloatingTooltip(dom.chartTooltip));

    svg.appendChild(ptGroup);
  });
}

/**
 * Update the temperature legend bar ticks, single-core marker, and active color-mode note.
 */
function updateHeatmapLegend(minC, maxC, predictionData) {
  if (!predictionData) {
    dom.legendMinVal.textContent = "-- °C";
    dom.legendMid1Val.textContent = "-- °C";
    dom.legendMid2Val.textContent = "-- °C";
    dom.legendMaxVal.textContent = "-- °C";
    dom.legendSingleMarker.classList.add("hidden");
    dom.legendTitle.textContent = "Predicted Temperature Heatmap Scale (°C)";
    if (dom.legendModeNote) {
      dom.legendModeNote.textContent =
        "Current view: Unpredicted — active core fills show Categorical Core Type (Big = Orange/Yellow, Little = Cyan/Blue).";
    }
    return;
  }

  const span = maxC - minC;
  const mid1 = minC + span * 0.35;
  const mid2 = minC + span * 0.7;

  dom.legendMinVal.textContent = `${minC.toFixed(1)} °C`;
  dom.legendMid1Val.textContent = `${mid1.toFixed(1)} °C`;
  dom.legendMid2Val.textContent = `${mid2.toFixed(1)} °C`;
  dom.legendMaxVal.textContent = `${maxC.toFixed(1)} °C`;

  if (predictionData.mode === "single" && predictionData.selected_core) {
    const selT = predictionData.selected_core.temp_c;
    const pct = Math.max(0, Math.min(100, ((selT - minC) / (span || 1)) * 100));
    dom.legendSingleMarker.style.left = `${pct}%`;
    dom.legendSingleMarker.classList.remove("hidden");
    dom.legendTitle.textContent = `Selected Core Temperature — Core ${predictionData.selected_core.core_number}: ${selT.toFixed(2)} °C`;
    if (dom.legendModeNote) {
      dom.legendModeNote.textContent =
        "Current view: Selected core fill shows Predicted Temperature (°C); other cores show Categorical Core Type.";
    }
  } else {
    dom.legendSingleMarker.classList.add("hidden");
    dom.legendTitle.textContent = "Predicted Temperature Heatmap Scale (°C)";
    if (dom.legendModeNote) {
      dom.legendModeNote.textContent =
        "Current view: Active core fills show Predicted Temperature (°C) along this thermal scale; borders & sizes reflect Core Type.";
    }
  }
}

/**
 * Generic floating tooltip helper for both the floorplan SVG and the chart SVG.
 */
function showFloatingTooltip(mouseEvent, wrapperEl, tooltipEl, content) {
  const wrapperRect = wrapperEl.getBoundingClientRect();
  let html = `<div class="tt-title">${content.title}</div>`;
  for (const [k, v] of content.rows) {
    html += `<div class="tt-row"><span class="tt-key">${k}:</span><span class="tt-val">${v}</span></div>`;
  }
  tooltipEl.innerHTML = html;
  tooltipEl.classList.remove("hidden");

  const ttW = tooltipEl.offsetWidth || 220;
  const ttH = tooltipEl.offsetHeight || 120;
  let left = mouseEvent.clientX - wrapperRect.left + 14;
  let top = mouseEvent.clientY - wrapperRect.top + 14;

  if (left + ttW > wrapperRect.width - 8) {
    left = mouseEvent.clientX - wrapperRect.left - ttW - 14;
  }
  if (top + ttH > wrapperRect.height - 8) {
    top = mouseEvent.clientY - wrapperRect.top - ttH - 14;
  }
  tooltipEl.style.left = `${Math.max(6, left)}px`;
  tooltipEl.style.top = `${Math.max(6, top)}px`;
}

function hideFloatingTooltip(tooltipEl) {
  tooltipEl.classList.add("hidden");
}

/**
 * 5. Update the Right Panel KPIs, Clean Temperature Table, and Coupling Breakdown.
 */
function renderPredictionResults(bundle, data, roundtripMs) {
  const liveUs = data.live_prediction_time_us !== undefined ? data.live_prediction_time_us : data.prediction_time_us;
  dom.kpiTotalPower.textContent = `${data.total_power_w.toFixed(2)} W`;
  dom.kpiPredTime.textContent = `${liveUs.toFixed(3)} µs`;
  const benchMedianUs = bundle.benchmark && bundle.benchmark.numpy_median_time_us
    ? `${bundle.benchmark.numpy_median_time_us.toFixed(3)} µs`
    : null;
  if (benchMedianUs) {
    dom.kpiRoundtripTime.textContent = `Live single-call (Benchmark median: ${benchMedianUs})`;
  } else {
    dom.kpiRoundtripTime.textContent = `Live single-call (HTTP: ${roundtripMs.toFixed(1)} ms)`;
  }
  const perFpParams = data.model_parameters_per_floorplan || bundle.model_parameters_per_floorplan || 64;
  const totalParams = data.total_model_parameters_all_floorplans || bundle.total_model_parameters_all_floorplans || 1472;
  const totalRawBytes = data.total_raw_float64_bytes_all_floorplans || bundle.total_raw_float64_bytes_all_floorplans || 11776;
  const totalCsvBytes = data.total_csv_size_bytes_all_floorplans || bundle.total_csv_size_bytes_all_floorplans || 17733;
  dom.kpiModelSize.textContent =
    `Per floorplan: ${perFpParams} params, ${data.model_raw_float64_bytes} B raw float64 (${data.model_csv_size_bytes} B CSV) · ` +
    `All 23 floorplans: ${totalParams.toLocaleString()} params, ${totalRawBytes.toLocaleString()} B raw (${totalCsvBytes.toLocaleString()} B CSV)`;
  dom.kpiMatrixPath.textContent = bundle.matrix_file;

  if (data.mode === "all") {
    dom.heatmapModeBadge.textContent = `Predict All (${bundle.full_name})`;
    dom.resultBannerTitle.textContent = `Prediction Summary — ${bundle.full_name}`;
    dom.resultModePill.textContent = "PREDICT ALL";
    dom.kpiTempLabel.textContent = "Maximum Predicted Temp";
    dom.kpiTempC.textContent = `${data.max_temp_c.toFixed(2)} °C`;
    dom.kpiTempK.textContent = `${data.max_temp_k.toFixed(2)} K (Exact: ${data.max_temp_c.toFixed(6)} °C)`;
    dom.kpiCoreLabel.textContent = "Hottest Core";
    dom.kpiCoreName.textContent = data.hottest_core;
    dom.kpiCoreType.textContent = data.hottest_core_type;

    dom.tableCardTitle.textContent = "Predicted Core Temperatures";
    dom.tableRowCount.textContent = "All 8 Active Cores";
    dom.couplingSection.classList.add("hidden");

    dom.resultsTableBody.innerHTML = "";
    data.predictions.forEach((pred) => {
      const tr = document.createElement("tr");
      if (pred.is_hottest) tr.className = "row-hottest";
      const hotBadge = pred.is_hottest ? `<span class="hottest-tag">HOTTEST</span>` : "";
      tr.innerHTML = `
        <td title="${pred.core_name}">Core ${pred.core_number} (${pred.core_name})${hotBadge}</td>
        <td>${pred.short_type}</td>
        <td class="num">${pred.power_w.toFixed(2)}</td>
        <td class="num">+${pred.delta_t_c.toFixed(2)}</td>
        <td class="num" title="Exact: ${pred.temp_c.toFixed(6)} °C"><strong>${pred.temp_c.toFixed(2)}</strong></td>
        <td class="num" title="Exact: ${pred.temp_k.toFixed(6)} K">${pred.temp_k.toFixed(2)}</td>
      `;
      dom.resultsTableBody.appendChild(tr);
    });
  } else {
    // Individual core prediction mode: show ONLY the selected core's row
    const sel = data.selected_core;
    dom.heatmapModeBadge.textContent = `Core ${sel.core_number} (${sel.core_name})`;
    dom.resultBannerTitle.textContent = `Individual Prediction — Core ${sel.core_number}`;
    dom.resultModePill.textContent = `CORE ${sel.core_number} ONLY`;
    dom.kpiTempLabel.textContent = `Predicted Temp (Core ${sel.core_number})`;
    dom.kpiTempC.textContent = `${sel.temp_c.toFixed(2)} °C`;
    dom.kpiTempK.textContent = `${sel.temp_k.toFixed(2)} K (Exact: ${sel.temp_c.toFixed(6)} °C)`;
    dom.kpiCoreLabel.textContent = "Selected Core";
    dom.kpiCoreName.textContent = `Core ${sel.core_number} (${sel.core_name})`;
    dom.kpiCoreType.textContent = `${sel.core_type} · Power: ${sel.power_w.toFixed(2)} W`;

    dom.tableCardTitle.textContent = `Selected Core Temperature Result`;
    dom.tableRowCount.textContent = `Core ${sel.core_number} Only`;

    dom.resultsTableBody.innerHTML = `
      <tr class="row-selected-single">
        <td><strong>Core ${sel.core_number} (${sel.core_name})</strong></td>
        <td>${sel.short_type}</td>
        <td class="num">${sel.power_w.toFixed(2)}</td>
        <td class="num">+${sel.delta_t_c.toFixed(2)}</td>
        <td class="num" title="Exact: ${sel.temp_c.toFixed(6)} °C"><strong>${sel.temp_c.toFixed(2)}</strong></td>
        <td class="num" title="Exact: ${sel.temp_k.toFixed(6)} K">${sel.temp_k.toFixed(2)}</td>
      </tr>
    `;

    // Populate expandable thermal coupling breakdown
    dom.couplingSection.classList.remove("hidden");
    dom.couplingTitle.textContent = `View 8-core thermal coupling breakdown for Core ${sel.core_number} (Self: +${sel.self_heating_c.toFixed(2)}°C, Coupled: +${sel.cross_coupling_c.toFixed(2)}°C)`;
    dom.couplingTableBody.innerHTML = "";
    (sel.coupling_contributions || []).forEach((contrib) => {
      const tr = document.createElement("tr");
      if (contrib.is_self) tr.className = "row-selected-single";
      const tag = contrib.is_self ? " (Self)" : "";
      tr.innerHTML = `
        <td>${contrib.source_core}${tag}</td>
        <td>${contrib.source_type.split(" ")[0]}</td>
        <td class="num">${contrib.r_ij_k_per_w.toFixed(4)}</td>
        <td class="num">${contrib.power_w.toFixed(2)}</td>
        <td class="num">+${contrib.temp_rise_c.toFixed(2)}</td>
      `;
      dom.couplingTableBody.appendChild(tr);
    });
  }
}

/**
 * Render the loaded 8x8 Thermal Resistance Matrix R (K/W) inside the collapsible inspector.
 */
function renderThermalMatrixTable(bundle) {
  dom.inspectorMatrixFile.textContent = bundle.matrix_file;
  const cols = bundle.matrix_core_order;
  const R = bundle.thermal_matrix_k_per_w;

  let headHtml = `<tr><th>Target \\ Source</th>`;
  cols.forEach((c) => {
    headHtml += `<th>${c.replace("_L0", "")}</th>`;
  });
  headHtml += `</tr>`;
  dom.matrixRThead.innerHTML = headHtml;

  let bodyHtml = "";
  R.forEach((rowVals, rIdx) => {
    bodyHtml += `<tr><th>${cols[rIdx]}</th>`;
    rowVals.forEach((val, cIdx) => {
      const cls = rIdx === cIdx ? 'class="diag-cell"' : "";
      bodyHtml += `<td ${cls}>${val.toFixed(4)}</td>`;
    });
    bodyHtml += `</tr>`;
  });
  dom.matrixRTbody.innerHTML = bodyHtml;
}

/**
 * Render precomputed Stage A evaluation benchmark comparisons:
 * A. Execution Time Comparison (Logarithmic 1 µs – 10 s horizontal bars)
 * B. Model Size (Parameter Storage: 64 params / 512 B raw / 771 B CSV per floorplan vs.
 *    1,472 params / 11,776 B raw / 17,733 B CSV across all 23 floorplans)
 * Automatically called whenever a floorplan is loaded.
 */
function renderBenchmarkSection(bundle) {
  if (!dom.evalContentContainer || !dom.evalUnavailableBanner) return;

  const bench = bundle.benchmark;
  if (!bench || !bench.available) {
    dom.evalContentContainer.classList.add("hidden");
    dom.evalUnavailableBanner.classList.remove("hidden");
    if (dom.evalStatusBadge) dom.evalStatusBadge.textContent = "Benchmark Unavailable";
    return;
  }

  dom.evalContentContainer.classList.remove("hidden");
  dom.evalUnavailableBanner.classList.add("hidden");
  if (dom.evalStatusBadge) {
    dom.evalStatusBadge.textContent = `${bundle.full_name} Benchmark`;
  }

  // Execution Time Comparison
  const npTimeUs = bench.numpy_median_time_us || 0;
  const hsTimeMs = bench.hotspot_median_time_ms || 0;
  const webTimeMs = bench.web_api_median_time_ms !== null && bench.web_api_median_time_ms !== undefined
    ? bench.web_api_median_time_ms
    : 1.66; // Fallback to overall median if not set per fp

  // Calculate Logarithmic Bar Widths:
  // Scale range: 1 µs (10^0 µs) to 10 s (10^7 µs), i.e., log10 span of [0, 7].
  const hsUs = Math.max(1, hsTimeMs * 1000);
  const webUs = Math.max(1, webTimeMs * 1000);
  const npUs = Math.max(1, npTimeUs);

  function logPct(us) {
    const logVal = Math.log10(Math.max(1, us));
    const pct = (logVal / 7.0) * 100;
    return Math.max(2.5, Math.min(98.5, pct));
  }

  const hsPct = logPct(hsUs);
  const webPct = logPct(webUs);
  const npPct = logPct(npUs);

  if (dom.evalBarHotspot) dom.evalBarHotspot.style.width = `${hsPct.toFixed(1)}%`;
  if (dom.evalBarHotspotVal) {
    dom.evalBarHotspotVal.innerHTML = `${hsTimeMs.toFixed(2)} ms <span class="eval-bar-unit-sub">(${(hsTimeMs / 1000).toFixed(2)} s)</span>`;
  }

  if (dom.evalBarWeb) dom.evalBarWeb.style.width = `${webPct.toFixed(1)}%`;
  if (dom.evalBarWebVal) {
    dom.evalBarWebVal.innerHTML = `${webTimeMs.toFixed(2)} ms <span class="eval-bar-unit-sub">(${Math.round(webTimeMs * 1000)} &mu;s)</span>`;
  }

  if (dom.evalBarNumpy) dom.evalBarNumpy.style.width = `${npPct.toFixed(1)}%`;
  if (dom.evalBarNumpyVal) {
    dom.evalBarNumpyVal.innerHTML = `${npTimeUs.toFixed(3)} &mu;s <span class="eval-bar-unit-sub">(${((npTimeUs / 1000).toFixed(4))} ms)</span>`;
  }

  // Model Size (Parameter Storage — Not Runtime RAM)
  const perFpParams = bench.num_parameters || bundle.model_parameters_per_floorplan || 64;
  const rawBytes = bench.raw_float64_bytes || bundle.model_raw_float64_bytes || 512;
  const csvBytes = bench.serialized_csv_bytes || bundle.model_csv_size_bytes || 771;

  const totalParams = bench.total_parameters_all_floorplans || bundle.total_model_parameters_all_floorplans || 1472;
  const totalRawBytes = bench.total_raw_float64_bytes_all_floorplans || bundle.total_raw_float64_bytes_all_floorplans || 11776;
  const totalCsvBytes = bench.total_csv_size_bytes_all_floorplans || bundle.total_csv_size_bytes_all_floorplans || 17733;

  if (dom.evalBarStoragePredictor) {
    dom.evalBarStoragePredictor.style.width = "30%";
  }
  if (dom.evalBarStoragePredictorMeta) {
    dom.evalBarStoragePredictorMeta.textContent = `1 floorplan (${bundle.full_name}) • 8×8 matrix (${perFpParams} float64 params)`;
  }
  if (dom.evalBarStoragePredictorVal) {
    dom.evalBarStoragePredictorVal.innerHTML = `${rawBytes} B raw (${perFpParams} params) <span class="eval-bar-unit-sub">&middot; ${csvBytes} B CSV</span>`;
  }

  if (dom.evalBarStorageTotal) {
    dom.evalBarStorageTotal.style.width = "68%";
  }
  if (dom.evalBarStorageTotalMeta) {
    dom.evalBarStorageTotalMeta.textContent = `All 23 floorplans • 23×8×8 (${totalParams.toLocaleString()} float64 params)`;
  }
  if (dom.evalBarStorageTotalVal) {
    dom.evalBarStorageTotalVal.innerHTML = `${totalRawBytes.toLocaleString()} B raw (${totalParams.toLocaleString()} params) <span class="eval-bar-unit-sub">&middot; ${totalCsvBytes.toLocaleString()} B CSV</span>`;
  }
}

/**
 * Bind UI event listeners for side-by-side Style & Floorplan dropdowns, presets, and Predict All.
 */
function bindEventListeners() {
  // Dropdown 1: Floorplan Style
  dom.styleSelect.addEventListener("change", () => {
    const selectedStyle = dom.styleSelect.value;
    AppState.currentStyle = selectedStyle;
    populateFloorplanDropdown(selectedStyle, null);
    const firstFp = dom.floorplanSelect.value;
    if (selectedStyle && firstFp) {
      loadFloorplan(`${selectedStyle}/${firstFp}`, { autoPredict: false });
    }
  });

  // Dropdown 2: Floorplan within selected Style
  dom.floorplanSelect.addEventListener("change", () => {
    const selectedStyle = dom.styleSelect.value || AppState.currentStyle;
    const selectedFp = dom.floorplanSelect.value;
    if (selectedStyle && selectedFp) {
      const fullName = selectedFp.includes("/") ? selectedFp : `${selectedStyle}/${selectedFp}`;
      loadFloorplan(fullName, { autoPredict: false });
    }
  });

  // Preset power buttons
  dom.btnPresetVal.addEventListener("click", () => applyPowerPreset("default"));
  dom.btnPresetMid.addEventListener("click", () => applyPowerPreset("midpoint"));
  dom.btnPresetMin.addEventListener("click", () => applyPowerPreset("min"));
  dom.btnPresetMax.addEventListener("click", () => applyPowerPreset("max"));

  // Predict All Cores button
  dom.btnPredictAll.addEventListener("click", () => {
    runPrediction("all");
  });
}

function applyPowerPreset(mode) {
  const bundle = AppState.currentFloorplan;
  if (!bundle) return;
  clearValidationErrors();

  bundle.cores.forEach((core) => {
    const inputEl = document.getElementById(`input-power-${core.name}`);
    if (!inputEl) return;
    if (mode === "default") {
      inputEl.value = core.default_power_w.toFixed(2);
    } else if (mode === "midpoint") {
      inputEl.value = core.midpoint_power_w.toFixed(2);
    } else if (mode === "min") {
      inputEl.value = core.min_power_w.toFixed(2);
    } else if (mode === "max") {
      inputEl.value = core.max_power_w.toFixed(2);
    }
  });

  markPredictionOutdated();
}

document.addEventListener("DOMContentLoaded", initDashboard);
