const state = {
  datasets: [],
  selectedDataset: null,
  rangeCount: 0,
  roiPresets: [],
  algorithmPresets: [],
  algorithmDefaults: {},
  preview: null,
  previewImage: null,
  roiCorners: null,
  dragCorner: null,
  activeJobId: null,
  pollTimer: null,
  runProgressSamples: [],
  sharedStateReady: false,
  applyingSharedState: false,
  sharedStateSaveTimer: null,
  currentRunId: null,
  currentMetrics: [],
  defaultMetrics: [],
  availability: {
    available_dates: [],
    day_counts: {},
    first_timestamp: null,
    last_timestamp: null,
  },
  availableTimesByDate: {},
  selectedMetrics: [],
  resultRows: [],
  plotZoom: null,
  gpuStatus: null,
  workflow: {
    dataset: false,
    roi: false,
    algorithm: false,
  },
};

const cornerOrder = ["top_left", "top_right", "bottom_right", "bottom_left"];
const algorithmTypeOptions = [
  ["difference", "Difference metrics"],
  ["tile_statistics", "Tile mean/median summary"],
];
const algorithmFields = [
  {
    name: "algorithm_type",
    type: "select",
    label: "Algorithm",
    options: algorithmTypeOptions,
    algorithms: ["difference", "tile_statistics"],
  },
  {
    name: "reference_window_size_images",
    type: "number",
    label: "Reference window images",
    min: 1,
    step: 1,
    algorithms: ["difference"],
  },
  {
    name: "reference_gap_images",
    type: "number",
    label: "Reference gap images",
    min: 0,
    step: 1,
    algorithms: ["difference"],
  },
  {
    name: "live_average_size_images",
    type: "number",
    label: "Live average images",
    min: 1,
    step: 1,
    algorithms: ["difference", "tile_statistics"],
  },
  {
    name: "processing_stride_images",
    type: "number",
    label: "Processing stride images",
    min: 1,
    step: 1,
    algorithms: ["difference", "tile_statistics"],
  },
  {
    name: "difference_threshold_abs",
    type: "number",
    label: "Difference threshold abs",
    min: 0,
    step: 0.1,
    algorithms: ["difference"],
  },
  {
    name: "smoothing_window_images",
    type: "number",
    label: "Smoothing window images",
    min: 1,
    step: 1,
    algorithms: ["difference", "tile_statistics"],
  },
  {
    name: "image_downscale_factor",
    type: "number",
    label: "Image downscale factor",
    min: 0.001,
    step: 0.05,
    algorithms: ["difference"],
  },
  {
    name: "use_median_reference",
    type: "checkbox",
    label: "Use median reference",
    algorithms: ["difference"],
  },
  {
    name: "reference_refresh_interval_minutes",
    type: "number",
    label: "Reference refresh minutes",
    min: 0,
    step: 1,
    algorithms: ["difference"],
  },
  {
    name: "image_cache_size_images",
    type: "number",
    label: "Image cache size images",
    min: 0,
    step: 1,
    algorithms: ["difference"],
  },
  {
    name: "grid_size",
    type: "number",
    label: "Grid size",
    min: 1,
    step: 1,
    algorithms: ["difference"],
  },
  {
    name: "output_directory",
    type: "text",
    label: "Output directory",
    algorithms: ["difference"],
  },
  {
    name: "save_preview_images",
    type: "checkbox",
    label: "Save preview images",
    algorithms: ["difference"],
  },
  {
    name: "preview_image_count",
    type: "number",
    label: "Preview image count",
    min: 0,
    step: 1,
    algorithms: ["difference"],
  },
  {
    name: "run_name",
    type: "text",
    label: "Run name",
    algorithms: ["difference"],
  },
];

const algorithmHelp = {
  algorithm_type:
    "Difference metrics computes the existing reference-vs-live metrics. Tile mean/median summary only scans the selected images and returns one mean and one median over time for each ROI tile.",
  reference_window_size_images:
    "Number of previous images used to compute the reference image.",
  reference_gap_images:
    "Number of images between the current live image and the end of the reference window.",
  live_average_size_images:
    "Number of consecutive images averaged before processing the current image. Use larger values to reduce noise and flicker.",
  processing_stride_images:
    "Process every nth image. A value of 1 processes every image; larger values reduce runtime and output density.",
  difference_threshold_abs:
    "Absolute difference threshold used for area-ratio metrics and affected-cell detection.",
  smoothing_window_images:
    "Rolling smoothing window applied to numeric output values. A value of 1 disables smoothing.",
  image_downscale_factor:
    "Scale factor applied while loading images. Keep 1.0 for identical full-resolution analysis; values below 1.0 downscale images and change results.",
  use_median_reference:
    "Use a pixelwise median for the reference image instead of a mean. Median is more robust to short transient changes.",
  reference_refresh_interval_minutes:
    "How long a computed reference image is reused before it is rebuilt. 0 recomputes the reference for every processed image; 60 refreshes roughly hourly.",
  image_cache_size_images:
    "Maximum number of decoded images kept in RAM for CPU runs or VRAM for GPU runs. Larger values reduce repeated TIFF reads for overlapping windows but use more memory. 0 disables caching.",
  grid_size:
    "Splits the quadrilateral ROI into grid_size x grid_size cells that follow the ROI geometry.",
  output_directory:
    "Directory where timestamped run folders, CSV results, configs, logs, plots, and preview images are written.",
  save_preview_images:
    "Save example reference, live, diff, and ROI-grid overlay images into the run folder.",
  preview_image_count:
    "Number of processed examples for which preview images are saved. Use 0 to disable preview image output.",
  run_name:
    "Optional name appended to the timestamped output folder. Unsafe filename characters are replaced.",
};

function activeAlgorithmType(values = null) {
  const candidate = values?.algorithm_type || document.querySelector("[data-algorithm-field='algorithm_type']")?.value;
  return candidate === "tile_statistics" ? "tile_statistics" : "difference";
}

function visibleAlgorithmFields(algorithmType) {
  return algorithmFields.filter((field) => field.algorithms.includes(algorithmType) || field.name === "algorithm_type");
}

function readVisibleAlgorithmInputs() {
  const config = {};
  for (const input of document.querySelectorAll("[data-algorithm-field]")) {
    const name = input.dataset.algorithmField;
    if (input.type === "checkbox") {
      config[name] = input.checked;
    } else if (input.type === "number") {
      config[name] = Number(input.value);
    } else {
      config[name] = input.value;
    }
  }
  return config;
}

function $(id) {
  return document.getElementById(id);
}

async function fetchJson(url, options = {}) {
  const response = await fetch(url, {
    headers: { "Content-Type": "application/json" },
    ...options,
  });
  if (!response.ok) {
    const contentType = response.headers.get("Content-Type") || "";
    if (contentType.includes("application/json")) {
      const payload = await response.json();
      throw new Error(payload.error || payload.message || `${response.status} ${response.statusText}`);
    }
    const text = await response.text();
    const cleanText = text.replace(/<[^>]+>/g, " ").replace(/\s+/g, " ").trim();
    throw new Error(cleanText || `${response.status} ${response.statusText}`);
  }
  return response.json();
}

function setStatus(id, message, kind = "") {
  const el = $(id);
  el.textContent = message || "";
  el.classList.remove("warning", "error", "success");
  if (kind) el.classList.add(kind);
}

function renderDetails(id, pairs) {
  const dl = $(id);
  dl.replaceChildren();
  for (const [label, value] of pairs) {
    const dt = document.createElement("dt");
    dt.textContent = label;
    const dd = document.createElement("dd");
    if (Array.isArray(value)) {
      dd.textContent = value.join(", ");
    } else {
      dd.textContent = value ?? "";
    }
    dl.append(dt, dd);
  }
}

function collectSharedState() {
  let algorithmConfig = {};
  try {
    algorithmConfig = getAlgorithmConfig();
  } catch (_error) {
    algorithmConfig = state.algorithmDefaults || {};
  }
  return {
    selected_dataset_name: state.selectedDataset?.name || null,
    time_mode: document.querySelector("input[name='time-mode']:checked")?.value || "complete",
    range_start: $("range-start").value || "",
    range_end: $("range-end").value || "",
    workflow: { ...state.workflow },
    roi_corners: state.roiCorners ? JSON.parse(JSON.stringify(state.roiCorners)) : null,
    grid_size: getGridSize(),
    roi_preset_name: $("roi-preset-select")?.value || "",
    algorithm_config: algorithmConfig,
    algorithm_preset_name: $("algorithm-preset-select")?.value || "",
    compute_backend: $("compute-backend")?.value || algorithmConfig.compute_backend || "gpu",
  };
}

function scheduleSharedStateSave(extraPatch = null) {
  if (!state.sharedStateReady || state.applyingSharedState) return;
  clearTimeout(state.sharedStateSaveTimer);
  state.sharedStateSaveTimer = setTimeout(() => {
    void saveSharedState(extraPatch);
  }, 250);
}

async function saveSharedState(extraPatch = null) {
  if (state.applyingSharedState) return;
  const payload = extraPatch ? { ...collectSharedState(), ...extraPatch } : collectSharedState();
  try {
    await fetchJson("/api/app-state", {
      method: "PATCH",
      body: JSON.stringify(payload),
    });
  } catch (error) {
    console.warn("Shared state save failed", error);
  }
}

async function loadSharedState() {
  try {
    const payload = await fetchJson("/api/app-state");
    await applySharedState(payload);
  } catch (error) {
    setStatus("range-summary", `Could not load shared app state: ${error.message}`, "warning");
    state.sharedStateReady = true;
  }
}

async function applySharedState(payload) {
  const ui = payload.ui_state || {};
  state.applyingSharedState = true;
  try {
    if (ui.selected_dataset_name && state.datasets.some((dataset) => dataset.name === ui.selected_dataset_name)) {
      $("dataset-select").value = ui.selected_dataset_name;
      setSelectedDataset(ui.selected_dataset_name, { resetDependent: false });
    }

    const mode = ui.time_mode === "custom" ? "custom" : "complete";
    const radio = document.querySelector(`input[name='time-mode'][value='${mode}']`);
    if (radio) radio.checked = true;
    if (ui.range_start) setRangeValue("start", ui.range_start);
    if (ui.range_end) setRangeValue("end", ui.range_end);

    if (ui.algorithm_config) {
      renderAlgorithmForm({ ...state.algorithmDefaults, ...ui.algorithm_config });
      $("compute-backend").value = ui.compute_backend || ui.algorithm_config.compute_backend || "gpu";
    }
    if (ui.grid_size) $("grid-size").value = ui.grid_size;
    if (ui.roi_corners) state.roiCorners = JSON.parse(JSON.stringify(ui.roi_corners));
    if (ui.algorithm_preset_name && $("algorithm-preset-select")) {
      $("algorithm-preset-select").value = ui.algorithm_preset_name;
    }

    if (ui.workflow) {
      state.workflow = {
        dataset: Boolean(ui.workflow.dataset),
        roi: Boolean(ui.workflow.roi),
        algorithm: Boolean(ui.workflow.algorithm),
      };
    }

    syncRangeVisibility();
    await updateRangeCount();
    await loadRoiPresets();
    if (ui.roi_preset_name && $("roi-preset-select")) {
      $("roi-preset-select").value = ui.roi_preset_name;
    }
    if (state.workflow.dataset && state.selectedDataset?.indexed && state.roiCorners && !payload.active_job) {
      await loadPreview({ preserveWorkflow: true });
    }
    updateWorkflowStatus();
  } finally {
    state.applyingSharedState = false;
    state.sharedStateReady = true;
  }

  if (payload.active_job && ["running", "cancelling"].includes(payload.active_job.state)) {
    attachRunJob(payload.active_job);
  } else if (payload.latest_job) {
    renderProgress(payload.latest_job);
  }
}

function setWorkflowStatus(section, complete) {
  state.workflow[section] = Boolean(complete);
  updateWorkflowStatus();
}

function resetDownstreamStatus(fromSection) {
  if (fromSection === "dataset") {
    state.workflow.dataset = false;
    state.workflow.roi = false;
    state.workflow.algorithm = false;
  } else if (fromSection === "roi") {
    state.workflow.roi = false;
    state.workflow.algorithm = false;
  } else if (fromSection === "algorithm") {
    state.workflow.algorithm = false;
  }
  updateWorkflowStatus();
}

function updateWorkflowStatus() {
  const statusByStep = {
    "dataset-step": state.workflow.dataset,
    "roi-step": state.workflow.roi,
    "algorithm-step": state.workflow.algorithm,
    "execute-step": state.workflow.dataset && state.workflow.roi && state.workflow.algorithm,
  };
  for (const [stepId, complete] of Object.entries(statusByStep)) {
    const button = document.querySelector(`.run-step-button[data-step="${stepId}"]`);
    if (!button) continue;
    button.classList.toggle("step-complete", complete);
    button.setAttribute("aria-label", `${button.textContent.trim()} ${complete ? "complete" : "incomplete"}`);
  }
  $("index-dataset").disabled = !state.selectedDataset;
  $("confirm-dataset").disabled = !(state.selectedDataset?.indexed && state.rangeCount > 0);
  $("load-preview").disabled = !state.workflow.dataset;
  refreshRunButton();
  scheduleSharedStateSave();
}

function resetPreviewAndRoi() {
  state.preview = null;
  state.previewImage = null;
  state.roiCorners = null;
  state.dragCorner = null;
  $("roi-preset-select").value = "";
  $("roi-preset-name").value = "";
  $("roi-comment").value = "";
  setStatus("roi-status", "");
  drawRoiCanvas();
}

function resetAlgorithmConfirmation() {
  state.workflow.algorithm = false;
  updateWorkflowStatus();
}

function markDatasetSelectionDirty() {
  state.workflow.dataset = false;
  state.workflow.roi = false;
  state.workflow.algorithm = false;
  resetPreviewAndRoi();
  updateWorkflowStatus();
}

function markRoiDirty() {
  state.workflow.roi = false;
  state.workflow.algorithm = false;
  updateWorkflowStatus();
}

function markAlgorithmDirty() {
  state.workflow.algorithm = false;
  updateWorkflowStatus();
}

function createInfoTip(text) {
  const tip = document.createElement("span");
  tip.className = "info-tip";
  tip.tabIndex = 0;
  tip.dataset.tooltip = text;
  tip.title = text;
  tip.textContent = "i";
  return tip;
}

function createLabelHeading(text, helpText) {
  const heading = document.createElement("span");
  heading.className = "label-heading";
  heading.append(document.createTextNode(text));
  if (helpText) heading.append(createInfoTip(helpText));
  return heading;
}

function toDatetimeLocalValue(value) {
  if (!value) return "";
  const match = String(value).trim().match(/^(\d{4}-\d{2}-\d{2})[ T](\d{2}:\d{2})(?::(\d{2}))?/);
  return match ? `${match[1]}T${match[2]}:${match[3] || "00"}` : "";
}

function setDefaultTimeRangeFromMetadata() {
  if (!state.selectedDataset) return;
  setRangeValue("start", toDatetimeLocalValue(state.selectedDataset.start_time));
  setRangeValue("end", toDatetimeLocalValue(state.selectedDataset.end_time));
  enforceRangeOrder();
}

function syncRangeVisibility() {
  const mode = document.querySelector("input[name='time-mode']:checked")?.value || "complete";
  const customFields = $("custom-range-fields");
  const hidden = mode !== "custom";
  customFields.hidden = hidden;
  $("range-start-date").disabled = hidden;
  $("range-end-date").disabled = hidden;
  $("range-start-time").disabled = hidden;
  $("range-end-time").disabled = hidden;
  enforceRangeOrder();
  void renderTimestampPicker();
}

function storeAvailability(availability, renderPicker = true) {
  if (!availability) return;
  state.availability = {
    available_dates: availability.available_dates || [],
    day_counts: availability.day_counts || {},
    first_timestamp: availability.first_timestamp || null,
    last_timestamp: availability.last_timestamp || null,
  };
  state.availableTimesByDate = {};
  if (renderPicker) void renderTimestampPicker();
}

function enforceRangeOrder() {
  const startInput = $("range-start");
  const endInput = $("range-end");
  if (startInput.value && endInput.value && endInput.value < startInput.value) {
    setRangeValue("end", startInput.value);
  }
  syncRangeDisplay("start");
  syncRangeDisplay("end");
}

function datePart(datetimeLocal) {
  return datetimeLocal ? datetimeLocal.slice(0, 10) : "";
}

function setRangeValue(which, value) {
  const hidden = which === "start" ? $("range-start") : $("range-end");
  const display = which === "start" ? $("range-start-display") : $("range-end-display");
  hidden.value = value || "";
  if (display) display.value = value ? value.replace("T", " ") : "";
}

function syncRangeDisplay(which) {
  const hidden = which === "start" ? $("range-start") : $("range-end");
  const display = which === "start" ? $("range-start-display") : $("range-end-display");
  if (display) display.value = hidden.value ? hidden.value.replace("T", " ") : "";
}

function resetAvailability() {
  state.availability = { available_dates: [], day_counts: {}, first_timestamp: null, last_timestamp: null };
  state.availableTimesByDate = {};
}

function setSelectPlaceholder(select, text) {
  select.replaceChildren();
  const option = document.createElement("option");
  option.value = "";
  option.textContent = text;
  select.append(option);
  select.value = "";
}

function populateDateSelect(select, dates, selectedDate, isSelectable) {
  select.replaceChildren();
  for (const dateText of dates) {
    const option = document.createElement("option");
    option.value = dateText;
    const count = state.availability.day_counts?.[dateText];
    option.textContent = count ? `${dateText} (${count})` : dateText;
    option.disabled = isSelectable ? !isSelectable(dateText) : false;
    select.append(option);
  }
  select.value = selectedDate;
}

function timeValueForSelection(item, which) {
  if (which === "end") return item.end_timestamp || item.timestamp;
  return item.start_timestamp || item.timestamp;
}

function timeOptionLabel(item) {
  if (item.count && item.count > 1) return `${item.time} (${item.count})`;
  return item.time;
}

function populateTimeSelect(select, times, selectedTimestamp, isSelectable, which = "start") {
  select.replaceChildren();
  for (const item of times) {
    const timestamp = timeValueForSelection(item, which);
    const option = document.createElement("option");
    option.value = timestamp;
    option.textContent = timeOptionLabel(item);
    option.disabled = isSelectable ? !isSelectable(timestamp) : false;
    select.append(option);
  }
  select.value = selectedTimestamp;
}

function pickDate(dates, preferredDate, isSelectable, preferLast = false) {
  const validDates = dates.filter((dateText) => (isSelectable ? isSelectable(dateText) : true));
  if (!validDates.length) return "";
  if (validDates.includes(preferredDate)) return preferredDate;
  return validDates[preferLast ? validDates.length - 1 : 0];
}

function pickTimestamp(times, preferredTimestamp, isSelectable, preferLast = false, which = "start") {
  const validTimes = times.filter((item) => {
    const timestamp = timeValueForSelection(item, which);
    return isSelectable ? isSelectable(timestamp) : true;
  });
  if (!validTimes.length) return null;
  const exact = validTimes.find((item) => timeValueForSelection(item, which) === preferredTimestamp);
  const selected = exact || validTimes[preferLast ? validTimes.length - 1 : 0];
  return { ...selected, selected_timestamp: timeValueForSelection(selected, which) };
}

async function loadTimesForDate(dateText) {
  if (!state.selectedDataset || !dateText) return [];
  const cacheKey = `${dateText}|minute`;
  if (state.availableTimesByDate[cacheKey]) return state.availableTimesByDate[cacheKey];
  const data = await fetchJson(
    `/api/datasets/${encodeURIComponent(state.selectedDataset.name)}/available-times?date=${encodeURIComponent(dateText)}&granularity=minute`
  );
  state.availableTimesByDate[cacheKey] = data.times || [];
  return state.availableTimesByDate[cacheKey];
}

let timestampPickerRenderId = 0;

async function renderTimestampPicker() {
  const renderId = ++timestampPickerRenderId;
  const dates = state.availability.available_dates || [];
  const dateSelects = [$("range-start-date"), $("range-end-date")];
  const timeSelects = [$("range-start-time"), $("range-end-time")];
  const mode = document.querySelector("input[name='time-mode']:checked")?.value || "complete";
  const controlsHidden = mode !== "custom";
  const help = $("timestamp-picker-help");

  if (!dates.length) {
    for (const select of dateSelects) setSelectPlaceholder(select, "No indexed dates");
    for (const select of timeSelects) setSelectPlaceholder(select, "No indexed times");
    for (const select of [...dateSelects, ...timeSelects]) select.disabled = true;
    if (help) help.textContent = "Index the dataset first. Custom range choices are built from parsed image timestamps.";
    return;
  }

  let startValue = $("range-start").value || toDatetimeLocalValue(state.availability.first_timestamp) || toDatetimeLocalValue(state.selectedDataset?.start_time);
  let endValue = $("range-end").value || toDatetimeLocalValue(state.availability.last_timestamp) || toDatetimeLocalValue(state.selectedDataset?.end_time);
  let startDate = pickDate(dates, datePart(startValue), null);
  let endDate = pickDate(dates, datePart(endValue), (dateText) => !startDate || dateText >= startDate, true);
  startDate = pickDate(dates, startDate, (dateText) => !endDate || dateText <= endDate);
  if (!startDate || !endDate) return;

  const [startTimes, endTimes] = await Promise.all([loadTimesForDate(startDate), loadTimesForDate(endDate)]);
  if (renderId !== timestampPickerRenderId) return;

  let startChoice = pickTimestamp(
    startTimes,
    startValue,
    (timestamp) => startDate !== endDate || !endValue || timestamp <= endValue,
    false,
    "start"
  );
  if (!startChoice) startChoice = pickTimestamp(startTimes, startValue, null, false, "start");
  if (startChoice) {
    startValue = startChoice.selected_timestamp;
    setRangeValue("start", startValue);
  }

  let endChoice = pickTimestamp(
    endTimes,
    endValue,
    (timestamp) => endDate !== startDate || !startValue || timestamp >= startValue,
    true,
    "end"
  );
  if (!endChoice) endChoice = pickTimestamp(endTimes, endValue, null, true, "end");
  if (endChoice) {
    endValue = endChoice.selected_timestamp;
    setRangeValue("end", endValue);
  }

  enforceRangeOrder();
  startDate = datePart($("range-start").value);
  endDate = datePart($("range-end").value);
  populateDateSelect($("range-start-date"), dates, startDate, (dateText) => !endDate || dateText <= endDate);
  populateDateSelect($("range-end-date"), dates, endDate, (dateText) => !startDate || dateText >= startDate);
  populateTimeSelect(
    $("range-start-time"),
    startTimes,
    $("range-start").value,
    (timestamp) => startDate !== endDate || timestamp <= $("range-end").value,
    "start"
  );
  populateTimeSelect(
    $("range-end-time"),
    endTimes,
    $("range-end").value,
    (timestamp) => startDate !== endDate || timestamp >= $("range-start").value,
    "end"
  );
  for (const select of [...dateSelects, ...timeSelects]) select.disabled = controlsHidden;
  if (help) {
    const first = state.availability.first_timestamp || "";
    const last = state.availability.last_timestamp || "";
    help.textContent = `Only dates and minute groups with parsed images are selectable. End-minute selections include all images through that minute. Indexed timestamp range: ${first} to ${last}.`;
  }
}

async function onTimestampDateChange(which) {
  const startDateSelect = $("range-start-date");
  const endDateSelect = $("range-end-date");
  const dateText = which === "start" ? startDateSelect.value : endDateSelect.value;
  const times = await loadTimesForDate(dateText);
  const preferred = which === "start" ? $("range-start").value : $("range-end").value;
  const other = which === "start" ? $("range-end").value : $("range-start").value;
  const choice = pickTimestamp(
    times,
    preferred,
    (timestamp) => {
      if (!other || datePart(other) !== dateText) return true;
      return which === "start" ? timestamp <= other : timestamp >= other;
    },
    which === "end",
    which
  );
  if (choice) setRangeValue(which, choice.selected_timestamp);
  enforceRangeOrder();
  await renderTimestampPicker();
  markDatasetSelectionDirty();
  updateRangeCount();
}

async function onTimestampTimeChange(which) {
  const select = which === "start" ? $("range-start-time") : $("range-end-time");
  if (select.value) setRangeValue(which, select.value);
  enforceRangeOrder();
  await renderTimestampPicker();
  markDatasetSelectionDirty();
  updateRangeCount();
}

function resetWorkflow() {
  if (state.activeJobId) {
    switchRunStep("execute-step");
    setStatus("gpu-status", "A run is active. Cancel or wait for it to finish before resetting the workflow.", "warning");
    return;
  }
  state.workflow.dataset = false;
  state.workflow.roi = false;
  state.workflow.algorithm = false;
  state.rangeCount = 0;

  const completeRadio = document.querySelector("input[name='time-mode'][value='complete']");
  if (completeRadio) completeRadio.checked = true;
  syncRangeVisibility();

  if (state.datasets.length) {
    const firstDataset = state.datasets[0];
    $("dataset-select").value = firstDataset.name;
    setSelectedDataset(firstDataset.name, { resetDependent: true });
  }

  if (state.algorithmDefaults && Object.keys(state.algorithmDefaults).length) {
    renderAlgorithmForm(state.algorithmDefaults);
  }
  $("algorithm-preset-select").value = "";
  $("algorithm-preset-name").value = "";
  $("algorithm-comment").value = "";
  $("roi-scope").value = "dataset";
  $("compute-backend").value = "gpu";
  state.gpuStatus = null;
  setStatus("gpu-status", "");
  setStatus("plot-status", "");
  $("progress-fill").style.width = "0%";
  renderDetails("progress-details", []);
  renderRunLog([]);
  state.activeJobId = null;
  state.runProgressSamples = [];
  $("cancel-run").disabled = true;

  switchTab("run-tab");
  switchRunStep("dataset-step");
  resetPreviewAndRoi();
  updateWorkflowStatus();
  drawRoiCanvas();
  refreshRunButton();
}

function switchTab(tabId) {
  document.querySelectorAll(".tab-button").forEach((button) => {
    button.classList.toggle("active", button.dataset.tab === tabId);
  });
  document.querySelectorAll(".tab-panel").forEach((panel) => {
    panel.classList.toggle("active", panel.id === tabId);
  });
  if (tabId === "run-tab") drawRoiCanvas();
}

function switchRunStep(stepId) {
  document.querySelectorAll(".run-step-button").forEach((button) => {
    button.classList.toggle("active", button.dataset.step === stepId);
  });
  document.querySelectorAll(".run-step-panel").forEach((panel) => {
    panel.classList.toggle("active", panel.id === stepId);
  });
  if (stepId === "roi-step") drawRoiCanvas();
}

async function loadDatasets() {
  const data = await fetchJson("/api/datasets");
  state.datasets = data.datasets;
  const select = $("dataset-select");
  select.replaceChildren();
  for (const dataset of state.datasets) {
    const option = document.createElement("option");
    option.value = dataset.name;
    option.textContent = dataset.name;
    select.append(option);
  }
  if (state.datasets.length) {
    select.value = state.selectedDataset?.name || state.datasets[0].name;
    setSelectedDataset(select.value);
  } else {
    renderDetails("dataset-details", [["Status", "No datasets found in configs/datasets.yaml"]]);
  }
}

function setSelectedDataset(name, options = {}) {
  const previousDatasetName = state.selectedDataset?.name || "";
  state.selectedDataset = state.datasets.find((dataset) => dataset.name === name) || null;
  if (!state.selectedDataset) return;
  if (options.resetDependent || (previousDatasetName && previousDatasetName !== state.selectedDataset.name)) {
    state.rangeCount = 0;
    resetAvailability();
    state.workflow.dataset = false;
    state.workflow.roi = false;
    state.workflow.algorithm = false;
    resetPreviewAndRoi();
  }
  setDefaultTimeRangeFromMetadata();
  syncRangeVisibility();
  renderDatasetDetails();
  loadRoiPresets();
  updateRangeCount();
  updateWorkflowStatus();
}

function renderDatasetDetails() {
  const dataset = state.selectedDataset;
  if (!dataset) return;
  renderDetails("dataset-details", [
    ["Name", dataset.name],
    ["Folders", dataset.folders || []],
    ["Metadata start", dataset.start_time],
    ["Metadata end", dataset.end_time],
    ["Description", dataset.description],
    ["Indexed files", dataset.file_count ?? "not indexed"],
    ["Missing timestamps", dataset.missing_timestamp_count ?? ""],
    ["First parsed timestamp", dataset.first_timestamp ?? ""],
    ["Last parsed timestamp", dataset.last_timestamp ?? ""],
  ]);
}

async function indexSelectedDataset() {
  if (!state.selectedDataset) return;
  setStatus("range-summary", "Indexing dataset images...");
  const data = await fetchJson(`/api/datasets/${encodeURIComponent(state.selectedDataset.name)}/index`, {
    method: "POST",
  });
  storeAvailability(data.availability, false);
  Object.assign(state.selectedDataset, data.summary, { indexed: true });
  renderDatasetDetails();
  state.workflow.dataset = false;
  state.workflow.roi = false;
  state.workflow.algorithm = false;
  resetPreviewAndRoi();
  updateWorkflowStatus();
  await renderTimestampPicker();
  await updateRangeCount();
}

async function confirmSelectedDataset() {
  if (!state.selectedDataset) return;
  if (!state.selectedDataset.indexed) {
    setStatus("range-summary", "Index the selected dataset before confirming the dataset and time range.", "warning");
    return;
  }
  await updateRangeCount();
  if (state.rangeCount > 0) {
    state.workflow.dataset = true;
    state.workflow.roi = false;
    state.workflow.algorithm = false;
    resetPreviewAndRoi();
    updateWorkflowStatus();
    setStatus("range-summary", `Dataset and time range confirmed. Images in selected range: ${state.rangeCount}.`, "success");
    switchRunStep("roi-step");
    await loadPreview();
  } else {
    setStatus("range-summary", "The selected time range contains no images. Adjust the range before confirming.", "warning");
    state.workflow.dataset = false;
    updateWorkflowStatus();
  }
}

function rangeParams() {
  const params = new URLSearchParams();
  const mode = document.querySelector("input[name='time-mode']:checked").value;
  params.set("mode", mode);
  if (mode === "custom") {
    params.set("start", $("range-start").value);
    params.set("end", $("range-end").value);
  }
  return params;
}

async function updateRangeCount() {
  if (!state.selectedDataset) return;
  enforceRangeOrder();
  if (!state.selectedDataset.indexed) {
    state.rangeCount = 0;
    resetAvailability();
    void renderTimestampPicker();
    setStatus("range-summary", "Index the selected dataset to count images in the processing range.");
    updateWorkflowStatus();
    refreshRunButton();
    return;
  }
  try {
    const data = await fetchJson(
      `/api/datasets/${encodeURIComponent(state.selectedDataset.name)}/range-count?${rangeParams()}`
    );
    storeAvailability(data.availability);
    state.rangeCount = data.count;
    const message = data.warning
      ? `${data.warning} Indexed files: ${data.file_count}.`
      : `Images in selected range: ${data.count}. Actual parsed range: ${data.first_timestamp || ""} to ${data.last_timestamp || ""}.`;
    setStatus("range-summary", message, data.warning ? "warning" : "");
  } catch (error) {
    state.rangeCount = 0;
    setStatus("range-summary", error.message, "error");
  }
  updateWorkflowStatus();
  refreshRunButton();
}

async function loadPreview(options = {}) {
  if (!state.selectedDataset) return;
  if (!state.selectedDataset.indexed) {
    setStatus("roi-status", "Confirm the selected dataset before loading the first image preview.", "warning");
    return;
  }
  const datasetName = state.selectedDataset.name;
  setStatus("roi-status", "Loading preview image...");
  const data = await fetchJson(
    `/api/datasets/${encodeURIComponent(datasetName)}/preview-info?${rangeParams()}`
  );
  if (!state.selectedDataset || state.selectedDataset.name !== datasetName) return;
  state.preview = data;
  state.previewImage = new Image();
  state.previewImage.onload = () => {
    if (!state.selectedDataset || state.selectedDataset.name !== datasetName) return;
    if (!state.roiCorners) {
      state.roiCorners = defaultCorners(data.original_shape);
    }
    if (options.preserveWorkflow) {
      updateWorkflowStatus();
    } else {
      markRoiDirty();
    }
    drawRoiCanvas();
    setStatus(
      "roi-status",
      `Preview loaded: ${data.image_path}. Original shape: ${data.original_shape[0]} x ${data.original_shape[1]}.`
    );
    refreshRunButton();
  };
  state.previewImage.src = data.image_url;
}

function defaultCorners(shape) {
  const height = shape[0];
  const width = shape[1];
  const marginX = width * 0.15;
  const marginY = height * 0.15;
  return {
    top_left: [marginX, marginY],
    top_right: [width - marginX, marginY],
    bottom_right: [width - marginX, height - marginY],
    bottom_left: [marginX, height - marginY],
  };
}

function previewScale() {
  if (!state.preview) return 1;
  return state.preview.scale || state.preview.preview_shape[1] / state.preview.original_shape[1];
}

function toPreviewPoint(point) {
  const scale = previewScale();
  return [point[0] * scale, point[1] * scale];
}

function bilinear(corners, u, v) {
  const tl = corners.top_left;
  const tr = corners.top_right;
  const br = corners.bottom_right;
  const bl = corners.bottom_left;
  return [
    (1 - u) * (1 - v) * tl[0] + u * (1 - v) * tr[0] + u * v * br[0] + (1 - u) * v * bl[0],
    (1 - u) * (1 - v) * tl[1] + u * (1 - v) * tr[1] + u * v * br[1] + (1 - u) * v * bl[1],
  ];
}

function drawRoiCanvas() {
  const canvas = $("roi-canvas");
  const ctx = canvas.getContext("2d");
  if (!state.preview || !state.previewImage) {
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    ctx.fillStyle = "#cbd5e1";
    ctx.fillText("Load a dataset preview to define an ROI.", 20, 30);
    return;
  }
  const [previewHeight, previewWidth] = state.preview.preview_shape;
  canvas.width = previewWidth;
  canvas.height = previewHeight;
  ctx.drawImage(state.previewImage, 0, 0, previewWidth, previewHeight);
  if (!state.roiCorners) return;

  const points = cornerOrder.map((name) => toPreviewPoint(state.roiCorners[name]));
  ctx.beginPath();
  ctx.moveTo(points[0][0], points[0][1]);
  for (let i = 1; i < points.length; i += 1) ctx.lineTo(points[i][0], points[i][1]);
  ctx.closePath();
  ctx.fillStyle = "rgba(255, 0, 0, 0.22)";
  ctx.fill();
  ctx.strokeStyle = "rgba(255, 50, 50, 0.95)";
  ctx.lineWidth = 2;
  ctx.stroke();

  const gridSize = getGridSize();
  ctx.strokeStyle = "rgba(255, 220, 80, 0.9)";
  ctx.lineWidth = gridSize > 7 ? 1 : 1.5;
  for (let i = 0; i <= gridSize; i += 1) {
    const t = i / gridSize;
    drawLine(ctx, toPreviewPoint(bilinear(state.roiCorners, t, 0)), toPreviewPoint(bilinear(state.roiCorners, t, 1)));
    drawLine(ctx, toPreviewPoint(bilinear(state.roiCorners, 0, t)), toPreviewPoint(bilinear(state.roiCorners, 1, t)));
  }

  for (const [index, name] of cornerOrder.entries()) {
    const [x, y] = points[index];
    ctx.beginPath();
    ctx.arc(x, y, 7, 0, Math.PI * 2);
    ctx.fillStyle = "#ffffff";
    ctx.fill();
    ctx.strokeStyle = "#dc2626";
    ctx.lineWidth = 2;
    ctx.stroke();
    ctx.fillStyle = "#111827";
    ctx.font = "12px system-ui";
    ctx.fillText(String(index + 1), x + 9, y - 9);
  }
}

function drawLine(ctx, start, end) {
  ctx.beginPath();
  ctx.moveTo(start[0], start[1]);
  ctx.lineTo(end[0], end[1]);
  ctx.stroke();
}

function getGridSize() {
  const value = Math.max(1, parseInt($("grid-size").value || "3", 10));
  $("grid-size").value = value;
  const field = document.querySelector("[data-algorithm-field='grid_size']");
  if (field && field.value !== String(value)) field.value = String(value);
  return value;
}

function canvasPointFromEvent(event) {
  const canvas = $("roi-canvas");
  const rect = canvas.getBoundingClientRect();
  return [
    ((event.clientX - rect.left) * canvas.width) / rect.width,
    ((event.clientY - rect.top) * canvas.height) / rect.height,
  ];
}

function nearestCorner(canvasPoint) {
  if (!state.roiCorners) return null;
  let bestName = null;
  let bestDistance = Number.POSITIVE_INFINITY;
  for (const name of cornerOrder) {
    const point = toPreviewPoint(state.roiCorners[name]);
    const distance = Math.hypot(point[0] - canvasPoint[0], point[1] - canvasPoint[1]);
    if (distance < bestDistance) {
      bestDistance = distance;
      bestName = name;
    }
  }
  return bestDistance <= 18 ? bestName : null;
}

function handleCanvasDown(event) {
  const corner = nearestCorner(canvasPointFromEvent(event));
  if (corner) {
    state.dragCorner = corner;
    event.preventDefault();
  }
}

function handleCanvasMove(event) {
  if (!state.dragCorner || !state.preview) return;
  const [x, y] = canvasPointFromEvent(event);
  const scale = previewScale();
  const width = state.preview.original_shape[1];
  const height = state.preview.original_shape[0];
  state.roiCorners[state.dragCorner] = [
    Math.max(0, Math.min(width - 1, x / scale)),
    Math.max(0, Math.min(height - 1, y / scale)),
  ];
  markRoiDirty();
  drawRoiCanvas();
}

function handleCanvasUp() {
  state.dragCorner = null;
}

async function loadRoiPresets() {
  if (!state.selectedDataset) return;
  const data = await fetchJson(`/api/roi-presets?dataset=${encodeURIComponent(state.selectedDataset.name)}`);
  state.roiPresets = data.presets;
  const select = $("roi-preset-select");
  select.replaceChildren();
  const empty = document.createElement("option");
  empty.value = "";
  empty.textContent = "Create new ROI";
  select.append(empty);
  for (const preset of state.roiPresets) {
    const option = document.createElement("option");
    option.value = preset.preset_name;
    option.textContent = `${preset.preset_name} (${preset.dataset_name})`;
    select.append(option);
  }
}

function selectedRoiPreset() {
  const name = $("roi-preset-select").value;
  return state.roiPresets.find((preset) => preset.preset_name === name) || null;
}

function applyRoiPreset() {
  const preset = selectedRoiPreset();
  if (!preset) return;
  state.roiCorners = JSON.parse(JSON.stringify(preset.corners));
  $("roi-preset-name").value = preset.preset_name;
  $("roi-comment").value = preset.comment || "";
  markRoiDirty();
  drawRoiCanvas();
  setStatus("roi-status", `Applied ROI preset ${preset.preset_name}.`);
  refreshRunButton();
}

async function saveRoiPreset(overwrite) {
  if (!state.preview || !state.roiCorners || !state.selectedDataset) {
    setStatus("roi-status", "Load a preview image and define an ROI first.", "warning");
    return;
  }
  const name = $("roi-preset-name").value.trim();
  if (!name) {
    setStatus("roi-status", "Enter an ROI preset name.", "warning");
    return;
  }
  const datasetName = $("roi-scope").value === "global" ? "global" : state.selectedDataset.name;
  const payload = {
    preset_name: name,
    dataset_name: datasetName,
    image_shape: state.preview.original_shape,
    corners: state.roiCorners,
    comment: $("roi-comment").value,
    overwrite,
  };
  try {
    await fetchJson("/api/roi-presets", {
      method: "POST",
      body: JSON.stringify(payload),
    });
    await loadRoiPresets();
    $("roi-preset-select").value = name;
    setStatus("roi-status", `Saved ROI preset ${name}.`);
  } catch (error) {
    setStatus("roi-status", error.message, "error");
  }
}

function confirmRoi() {
  if (!state.workflow.dataset) {
    setStatus("roi-status", "Confirm the dataset and time range first.", "warning");
    switchRunStep("dataset-step");
    return;
  }
  if (!state.preview || !state.roiCorners) {
    setStatus("roi-status", "Load the first image and define an ROI before confirming.", "warning");
    return;
  }
  setWorkflowStatus("roi", true);
  setStatus("roi-status", "ROI confirmed for this run.");
  switchRunStep("algorithm-step");
}

async function loadAlgorithmPresets() {
  const data = await fetchJson("/api/algorithm-presets");
  state.algorithmPresets = data.presets;
  state.algorithmDefaults = data.defaults;
  renderAlgorithmForm(state.algorithmDefaults);
  const select = $("algorithm-preset-select");
  select.replaceChildren();
  const empty = document.createElement("option");
  empty.value = "";
  empty.textContent = "Use current values";
  select.append(empty);
  for (const preset of state.algorithmPresets) {
    const option = document.createElement("option");
    option.value = preset.preset_name;
    option.textContent = preset.preset_name;
    select.append(option);
  }
}

function renderAlgorithmForm(values) {
  const form = $("algorithm-form");
  form.replaceChildren();
  const algorithmType = activeAlgorithmType(values);
  for (const field of visibleAlgorithmFields(algorithmType)) {
    const { name, type, label, min, step, options } = field;
    if (type === "checkbox") {
      const labelEl = document.createElement("label");
      labelEl.className = "checkbox-label";
      const input = document.createElement("input");
      input.type = "checkbox";
      input.dataset.algorithmField = name;
      input.checked = Boolean(values[name]);
      input.addEventListener("change", markAlgorithmDirty);
      labelEl.append(input, createLabelHeading(label, algorithmHelp[name]));
      form.append(labelEl);
    } else if (type === "select") {
      const labelEl = document.createElement("label");
      labelEl.append(createLabelHeading(label, algorithmHelp[name]));
      const select = document.createElement("select");
      select.dataset.algorithmField = name;
      for (const [optionValue, optionLabel] of options) {
        const option = document.createElement("option");
        option.value = optionValue;
        option.textContent = optionLabel;
        select.append(option);
      }
      select.value = values[name] ?? "difference";
      select.addEventListener("change", () => {
        const nextValues = { ...state.algorithmDefaults, ...readVisibleAlgorithmInputs(), algorithm_type: select.value };
        renderAlgorithmForm(nextValues);
        markAlgorithmDirty();
      });
      labelEl.append(select);
      form.append(labelEl);
    } else {
      const labelEl = document.createElement("label");
      labelEl.append(createLabelHeading(label, algorithmHelp[name]));
      const input = document.createElement("input");
      input.type = type;
      input.dataset.algorithmField = name;
      input.value = values[name] ?? "";
      if (min !== null) input.min = min;
      if (step !== null) input.step = step;
      if (name === "grid_size") {
        input.addEventListener("change", () => {
          $("grid-size").value = input.value;
          markRoiDirty();
          markAlgorithmDirty();
          drawRoiCanvas();
        });
      } else {
        input.addEventListener("change", markAlgorithmDirty);
      }
      labelEl.append(input);
      form.append(labelEl);
    }
  }
  $("grid-size").value = values.grid_size ?? 3;
  drawRoiCanvas();
}

function selectedAlgorithmPreset() {
  const name = $("algorithm-preset-select").value;
  return state.algorithmPresets.find((preset) => preset.preset_name === name) || null;
}

function applyAlgorithmPreset() {
  const preset = selectedAlgorithmPreset();
  if (!preset) return;
  renderAlgorithmForm({ ...state.algorithmDefaults, ...preset });
  if (preset.compute_backend && $("compute-backend")) {
    $("compute-backend").value = preset.compute_backend;
  }
  $("algorithm-preset-name").value = preset.preset_name;
  $("algorithm-comment").value = preset.comment || "";
  markAlgorithmDirty();
}

function getAlgorithmConfig() {
  const config = { ...state.algorithmDefaults, ...readVisibleAlgorithmInputs() };
  config.grid_size = getGridSize();
  if ($("compute-backend")) config.compute_backend = $("compute-backend").value || "gpu";
  return config;
}

async function saveAlgorithmPreset(overwrite) {
  const name = $("algorithm-preset-name").value.trim();
  if (!name) {
    alert("Enter an algorithm preset name.");
    return;
  }
  try {
    await fetchJson("/api/algorithm-presets", {
      method: "POST",
      body: JSON.stringify({
        preset_name: name,
        config: getAlgorithmConfig(),
        comment: $("algorithm-comment").value,
        overwrite,
      }),
    });
    await loadAlgorithmPresets();
    $("algorithm-preset-select").value = name;
  } catch (error) {
    alert(error.message);
  }
}

function confirmAlgorithm() {
  if (!state.workflow.dataset) {
    switchRunStep("dataset-step");
    setStatus("range-summary", "Confirm the dataset and time range before confirming the algorithm.", "warning");
    return;
  }
  if (!state.workflow.roi) {
    switchRunStep("roi-step");
    setStatus("roi-status", "Confirm the ROI before confirming the algorithm.", "warning");
    return;
  }
  try {
    getAlgorithmConfig();
    setWorkflowStatus("algorithm", true);
    switchRunStep("execute-step");
  } catch (error) {
    alert(error.message);
  }
}

function refreshRunButton() {
  const runActive = Boolean(state.activeJobId);
  $("start-run").disabled = !(
    !runActive
    && state.workflow.dataset
    && state.workflow.roi
    && state.workflow.algorithm
    && state.selectedDataset
    && state.rangeCount > 0
    && state.roiCorners
    && state.preview
  );
  $("cancel-run").disabled = !runActive;
}

async function checkGpuStatus() {
  setStatus("gpu-status", "Checking GPU availability...");
  try {
    const data = await fetchJson("/api/compute/gpu-status");
    state.gpuStatus = data;
    const parts = [
      data.message,
      `Expected: torch ${data.expected_torch_version}, torchvision ${data.expected_torchvision_version}, CUDA ${data.expected_cuda_version}.`,
    ];
    if (data.available && !data.version_matches) {
      parts.push("GPU is usable, but installed versions do not exactly match the expected CUDA 12.4 stack.");
    }
    setStatus("gpu-status", parts.filter(Boolean).join(" "), data.available ? (data.version_matches ? "success" : "warning") : "error");
  } catch (error) {
    setStatus("gpu-status", error.message, "error");
  }
}

async function startRun() {
  if ($("start-run").disabled || !state.selectedDataset || !state.preview || !state.roiCorners) return;
  switchRunStep("execute-step");
  const mode = document.querySelector("input[name='time-mode']:checked").value;
  const algorithmConfig = getAlgorithmConfig();
  algorithmConfig.compute_backend = $("compute-backend").value || "gpu";
  const roiPreset = selectedRoiPreset();
  const payload = {
    dataset_name: state.selectedDataset.name,
    time_mode: mode,
    range_start: $("range-start").value,
    range_end: $("range-end").value,
    algorithm_config: algorithmConfig,
    algorithm_preset_name: $("algorithm-preset-select").value,
    roi_preset_name: roiPreset ? roiPreset.preset_name : "",
    roi_config: {
      preset_name: roiPreset ? roiPreset.preset_name : "current_ui_roi",
      dataset_name: roiPreset ? roiPreset.dataset_name : state.selectedDataset.name,
      image_shape: state.preview.original_shape,
      corners: state.roiCorners,
      created_at: new Date().toISOString(),
      comment: $("roi-comment").value,
    },
  };
  $("start-run").disabled = true;
  $("cancel-run").disabled = true;
  state.activeJobId = null;
  state.runProgressSamples = [];
  renderRunLog(["Starting run..."]);
  try {
    const data = await fetchJson("/api/runs", {
      method: "POST",
      body: JSON.stringify(payload),
    });
    state.activeJobId = data.job_id;
    $("cancel-run").disabled = false;
    scheduleRunPoll(data.job_id, 0);
  } catch (error) {
    state.activeJobId = null;
    setStatus("gpu-status", error.message, "error");
    $("progress-fill").style.width = "0%";
    renderDetails("progress-details", [
      ["State", "failed"],
      ["Compute backend", algorithmConfig.compute_backend],
      ["Status", error.message],
    ]);
    renderRunLog([`Run failed before starting: ${error.message}`]);
    await loadSharedState();
    refreshRunButton();
  }
}

function attachRunJob(job) {
  state.activeJobId = job.job_id;
  state.runProgressSamples = [];
  switchTab("run-tab");
  switchRunStep("execute-step");
  renderProgress(job);
  refreshRunButton();
  if (job.state === "running" || job.state === "cancelling") {
    scheduleRunPoll(job.job_id, 1000);
  }
}

function scheduleRunPoll(jobId, delayMs = 1000) {
  clearTimeout(state.pollTimer);
  state.pollTimer = setTimeout(() => {
    void pollRun(jobId);
  }, delayMs);
}

async function pollRun(jobId) {
  let data;
  try {
    data = await fetchJson(`/api/runs/${jobId}/status`);
  } catch (error) {
    setStatus("gpu-status", `Run status connection interrupted; retrying. ${error.message}`, "warning");
    if (state.activeJobId === jobId) {
      scheduleRunPoll(jobId, 2000);
    }
    return;
  }
  renderProgress(data);
  if (data.state === "running" || data.state === "cancelling") {
    scheduleRunPoll(jobId, 1000);
  } else {
    clearTimeout(state.pollTimer);
    if (state.activeJobId === jobId) {
      state.activeJobId = null;
      $("cancel-run").disabled = true;
    }
    refreshRunButton();
    if (data.state === "finished" && data.result?.run_id) {
      await loadRuns();
      switchTab("results-tab");
      $("run-select").value = data.result.run_id;
      await loadRunDetails();
    } else if (data.state === "cancelled" && data.result?.run_id) {
      await loadRuns();
    }
  }
}

async function cancelRun() {
  if (!state.activeJobId) return;
  const jobId = state.activeJobId;
  $("cancel-run").disabled = true;
  try {
    const data = await fetchJson(`/api/runs/${encodeURIComponent(jobId)}/cancel`, {
      method: "POST",
    });
    renderProgress(data);
  } catch (error) {
    setStatus("gpu-status", error.message, "error");
    $("cancel-run").disabled = false;
  }
}

function renderProgress(job) {
  const progress = job.progress || {};
  const pct = Math.max(0, Math.min(100, Number(progress.percentage || 0)));
  const eta = estimateRemaining(job, pct);
  $("progress-fill").style.width = `${pct}%`;
  renderDetails("progress-details", [
    ["State", job.state],
    ["Dataset", progress.dataset_name || ""],
    ["Compute backend", progress.compute_backend || ""],
    ["Compute device", progress.compute_device || ""],
    ["Total images", progress.total_images ?? ""],
    ["Processed images", progress.processed_images ?? ""],
    ["Current image index", progress.current_image_index ?? ""],
    ["Current timestamp", progress.current_timestamp ?? ""],
    ["Progress", `${pct.toFixed(1)}%`],
    ["Estimated remaining", eta],
    ["Status", job.error || progress.status_message || ""],
    ["Run folder", job.result?.run_folder || ""],
  ]);
  renderRunLog(job.logs || []);
}

function renderRunLog(logs) {
  const pre = $("run-log");
  const count = $("run-log-count");
  const lines = logs || [];
  pre.textContent = lines.length ? lines.join("\n") : "No run log entries yet.";
  pre.scrollTop = pre.scrollHeight;
  count.textContent = `${lines.length} ${lines.length === 1 ? "entry" : "entries"}`;
}

function estimateRemaining(job, pct) {
  if (job.state === "cancelling") return "cancelling...";
  if (job.state !== "running") return "";
  if (!Number.isFinite(pct) || pct <= 0) return "calculating...";
  if (pct >= 100) return "done";

  const now = Date.now();
  const samples = state.runProgressSamples;
  const last = samples[samples.length - 1];
  if (!last || last.pct !== pct) {
    samples.push({ pct, time: now });
  }
  while (samples.length > 200) samples.shift();

  const threshold = Math.max(0, pct - 2);
  let baseline = null;
  for (let idx = samples.length - 1; idx >= 0; idx -= 1) {
    if (samples[idx].pct <= threshold) {
      baseline = samples[idx];
      break;
    }
  }
  if (!baseline) {
    baseline = samples.find((sample) => sample.pct < pct) || null;
  }
  if (!baseline) return "calculating...";

  const deltaPct = pct - baseline.pct;
  const deltaSeconds = (now - baseline.time) / 1000;
  if (deltaPct <= 0 || deltaSeconds <= 0) return "calculating...";
  const secondsRemaining = (deltaSeconds / deltaPct) * (100 - pct);
  if (!Number.isFinite(secondsRemaining) || secondsRemaining < 0) return "calculating...";
  return `~${formatDuration(secondsRemaining)}`;
}

function formatDuration(seconds) {
  const rounded = Math.max(0, Math.round(seconds));
  const hours = Math.floor(rounded / 3600);
  const minutes = Math.floor((rounded % 3600) / 60);
  const secs = rounded % 60;
  if (hours > 0) return `${hours}h ${minutes}m`;
  if (minutes > 0) return `${minutes}m ${secs}s`;
  return `${secs}s`;
}

async function loadRuns() {
  const data = await fetchJson("/api/results/runs");
  const select = $("run-select");
  const current = select.value;
  select.replaceChildren();
  for (const run of data.runs) {
    const option = document.createElement("option");
    option.value = run.run_id;
    option.textContent = run.run_id;
    select.append(option);
  }
  if (current) select.value = current;
}

async function loadRunDetails() {
  const runId = $("run-select").value;
  if (!runId) return;
  const data = await fetchJson(`/api/results/runs/${encodeURIComponent(runId)}`);
  state.currentRunId = runId;
  state.currentMetrics = data.metric_columns;
  state.defaultMetrics = data.default_metrics;
  state.selectedMetrics = [...data.default_metrics];
  state.plotZoom = null;
  state.resultRows = [];
  $("run-config-view").textContent = JSON.stringify(data.run_config, null, 2);
  $("dataset-config-view").textContent = JSON.stringify(data.dataset_config_used, null, 2);
  $("roi-config-view").textContent = JSON.stringify(data.roi_config, null, 2);
  renderMetricList();
  await loadInteractiveResultData();
  if (data.summary_plot_url) {
    $("result-plot").src = `${data.summary_plot_url}?t=${Date.now()}`;
    $("result-plot").style.display = "block";
  }
}

function renderMetricList() {
  const list = $("metric-list");
  const query = $("metric-search").value.toLowerCase();
  list.replaceChildren();
  for (const metric of state.currentMetrics) {
    if (query && !metric.toLowerCase().includes(query)) continue;
    const label = document.createElement("label");
    const input = document.createElement("input");
    input.type = "checkbox";
    input.value = metric;
    input.checked = state.selectedMetrics.includes(metric);
    input.addEventListener("change", () => updateMetricSelection(metric, input.checked));
    label.append(input, document.createTextNode(metric));
    list.append(label);
  }
}

async function updateMetricSelection(metric, selected) {
  const existing = new Set(state.selectedMetrics);
  if (selected) {
    existing.add(metric);
  } else {
    existing.delete(metric);
  }
  state.selectedMetrics = Array.from(existing).filter((name) => state.currentMetrics.includes(name));
  renderMetricList();
  await loadInteractiveResultData();
}

async function loadInteractiveResultData() {
  const container = $("interactive-plots");
  if (!state.currentRunId || !state.selectedMetrics.length) {
    state.resultRows = [];
    renderInteractivePlots();
    return;
  }
  const params = new URLSearchParams();
  for (const metric of state.selectedMetrics) params.append("metric", metric);
  const data = await fetchJson(`/api/results/runs/${encodeURIComponent(state.currentRunId)}/data?${params}`);
  state.resultRows = data.rows.map((row) => ({
    ...row,
    timestampMs: row.timestamp ? new Date(row.timestamp).getTime() : NaN,
  }));
  container.dataset.loaded = "true";
  renderInteractivePlots();
}

function renderInteractivePlots() {
  const container = $("interactive-plots");
  container.replaceChildren();
  if (!state.selectedMetrics.length) {
    const empty = document.createElement("div");
    empty.className = "plot-empty";
    empty.textContent = "Select one or more metrics to show interactive plots.";
    container.append(empty);
    return;
  }
  if (!state.resultRows.length) {
    const empty = document.createElement("div");
    empty.className = "plot-empty";
    empty.textContent = "No result rows loaded for the selected metrics.";
    container.append(empty);
    return;
  }
  for (const metric of state.selectedMetrics) {
    container.append(createPlotCard(metric));
  }
}

function createPlotCard(metric) {
  const card = document.createElement("div");
  card.className = "plot-card";
  const header = document.createElement("div");
  header.className = "plot-card-header";
  const title = document.createElement("div");
  title.className = "plot-card-title";
  title.textContent = metric;
  const remove = document.createElement("button");
  remove.type = "button";
  remove.className = "plot-remove";
  remove.title = `Remove ${metric}`;
  remove.textContent = "×";
  remove.addEventListener("click", () => updateMetricSelection(metric, false));
  header.append(title, remove);
  const svg = renderSvgPlot(metric);
  card.append(header, svg);
  return card;
}

function renderSvgPlot(metric) {
  const svg = document.createElementNS("http://www.w3.org/2000/svg", "svg");
  svg.setAttribute("class", "plot-svg");
  svg.setAttribute("viewBox", "0 0 900 240");
  const width = 900;
  const height = 240;
  const margin = { top: 18, right: 24, bottom: 42, left: 64 };
  const points = state.resultRows
    .map((row) => ({
      x: row.timestampMs,
      y: row[metric] == null ? Number.NaN : Number(row[metric]),
    }))
    .filter((point) => Number.isFinite(point.x) && Number.isFinite(point.y));
  if (!points.length) {
    const text = svgText(450, 120, "No numeric values for this metric", "middle");
    svg.append(text);
    return svg;
  }
  let xMin = Math.min(...points.map((point) => point.x));
  let xMax = Math.max(...points.map((point) => point.x));
  if (state.plotZoom) {
    xMin = state.plotZoom[0];
    xMax = state.plotZoom[1];
  }
  const visible = points.filter((point) => point.x >= xMin && point.x <= xMax);
  const yValues = visible.length ? visible.map((point) => point.y) : points.map((point) => point.y);
  let yMin = Math.min(...yValues);
  let yMax = Math.max(...yValues);
  if (yMin === yMax) {
    yMin -= 1;
    yMax += 1;
  }
  const plotWidth = width - margin.left - margin.right;
  const plotHeight = height - margin.top - margin.bottom;
  const xScale = (x) => margin.left + ((x - xMin) / Math.max(1, xMax - xMin)) * plotWidth;
  const yScale = (y) => margin.top + (1 - (y - yMin) / Math.max(1e-9, yMax - yMin)) * plotHeight;

  for (let i = 0; i <= 4; i += 1) {
    const y = margin.top + (plotHeight * i) / 4;
    svg.append(svgLine(margin.left, y, width - margin.right, y, "plot-gridline"));
    const value = yMax - ((yMax - yMin) * i) / 4;
    svg.append(svgText(margin.left - 8, y + 4, formatMetricTick(value), "end"));
  }
  svg.append(svgLine(margin.left, height - margin.bottom, width - margin.right, height - margin.bottom, "plot-axis"));
  svg.append(svgLine(margin.left, margin.top, margin.left, height - margin.bottom, "plot-axis"));

  const pathPoints = visible.length ? visible : points;
  const path = document.createElementNS("http://www.w3.org/2000/svg", "path");
  if (pathPoints.length === 1) {
    const point = pathPoints[0];
    const circle = document.createElementNS("http://www.w3.org/2000/svg", "circle");
    circle.setAttribute("cx", xScale(point.x).toFixed(2));
    circle.setAttribute("cy", yScale(point.y).toFixed(2));
    circle.setAttribute("r", "3");
    circle.setAttribute("class", "plot-line");
    circle.setAttribute("fill", "var(--accent)");
    svg.append(circle);
  } else {
    path.setAttribute(
      "d",
      pathPoints.map((point, index) => `${index === 0 ? "M" : "L"} ${xScale(point.x).toFixed(2)} ${yScale(point.y).toFixed(2)}`).join(" ")
    );
    path.setAttribute("class", "plot-line");
    svg.append(path);
  }

  const tickCount = 4;
  for (let i = 0; i <= tickCount; i += 1) {
    const x = margin.left + (plotWidth * i) / tickCount;
    const time = xMin + ((xMax - xMin) * i) / tickCount;
    svg.append(svgText(x, height - 16, formatTimeTick(time), "middle"));
  }

  let dragStart = null;
  let brush = null;
  svg.addEventListener("pointerdown", (event) => {
    const x = svgPointerX(svg, event);
    if (x < margin.left || x > width - margin.right) return;
    dragStart = x;
    brush = document.createElementNS("http://www.w3.org/2000/svg", "rect");
    brush.setAttribute("class", "plot-brush");
    brush.setAttribute("y", String(margin.top));
    brush.setAttribute("height", String(plotHeight));
    brush.setAttribute("x", String(x));
    brush.setAttribute("width", "0");
    svg.append(brush);
    svg.setPointerCapture(event.pointerId);
  });
  svg.addEventListener("pointermove", (event) => {
    if (dragStart === null || !brush) return;
    const x = Math.max(margin.left, Math.min(width - margin.right, svgPointerX(svg, event)));
    brush.setAttribute("x", String(Math.min(dragStart, x)));
    brush.setAttribute("width", String(Math.abs(x - dragStart)));
  });
  svg.addEventListener("pointerup", (event) => {
    if (dragStart === null || !brush) return;
    const endX = Math.max(margin.left, Math.min(width - margin.right, svgPointerX(svg, event)));
    const startX = dragStart;
    brush.remove();
    brush = null;
    dragStart = null;
    if (Math.abs(endX - startX) > 10) {
      const left = Math.min(startX, endX);
      const right = Math.max(startX, endX);
      const toTime = (x) => xMin + ((x - margin.left) / plotWidth) * (xMax - xMin);
      state.plotZoom = [toTime(left), toTime(right)];
      renderInteractivePlots();
    }
  });
  return svg;
}

function svgPointerX(svg, event) {
  const rect = svg.getBoundingClientRect();
  return ((event.clientX - rect.left) / rect.width) * 900;
}

function svgLine(x1, y1, x2, y2, className) {
  const line = document.createElementNS("http://www.w3.org/2000/svg", "line");
  line.setAttribute("x1", String(x1));
  line.setAttribute("y1", String(y1));
  line.setAttribute("x2", String(x2));
  line.setAttribute("y2", String(y2));
  line.setAttribute("class", className);
  return line;
}

function svgText(x, y, text, anchor) {
  const el = document.createElementNS("http://www.w3.org/2000/svg", "text");
  el.setAttribute("x", String(x));
  el.setAttribute("y", String(y));
  el.setAttribute("text-anchor", anchor);
  el.setAttribute("font-size", "11");
  el.setAttribute("fill", "#667085");
  el.textContent = text;
  return el;
}

function formatMetricTick(value) {
  if (Math.abs(value) >= 100) return value.toFixed(0);
  if (Math.abs(value) >= 10) return value.toFixed(1);
  return value.toFixed(3).replace(/0+$/, "").replace(/\.$/, "");
}

function formatTimeTick(ms) {
  return new Date(ms).toLocaleString(undefined, {
    month: "2-digit",
    day: "2-digit",
    hour: "2-digit",
    minute: "2-digit",
  });
}

async function plotSelectedMetrics() {
  if (!state.currentRunId) return;
  const metrics = state.selectedMetrics;
  if (!metrics.length) {
    setStatus("plot-status", "Select at least one metric.", "warning");
    return;
  }
  const data = await fetchJson(`/api/results/runs/${encodeURIComponent(state.currentRunId)}/plot`, {
    method: "POST",
    body: JSON.stringify({ metrics }),
  });
  $("result-plot").src = `${data.plot_url}?t=${Date.now()}`;
  $("result-plot").style.display = "block";
  setStatus("plot-status", `Saved plot with ${data.plotted_metrics.length} metric(s).`);
}

function bindEvents() {
  document.querySelectorAll(".tab-button").forEach((button) => {
    button.addEventListener("click", () => switchTab(button.dataset.tab));
  });
  $("reset-workflow").addEventListener("click", resetWorkflow);
  document.querySelectorAll(".run-step-button").forEach((button) => {
    button.addEventListener("click", () => switchRunStep(button.dataset.step));
  });
  $("refresh-datasets").addEventListener("click", loadDatasets);
  $("dataset-select").addEventListener("change", (event) => {
    setSelectedDataset(event.target.value, { resetDependent: true });
    switchRunStep("dataset-step");
  });
  $("index-dataset").addEventListener("click", indexSelectedDataset);
  $("confirm-dataset").addEventListener("click", confirmSelectedDataset);
  document.querySelectorAll("input[name='time-mode']").forEach((input) => {
    input.addEventListener("change", () => {
      markDatasetSelectionDirty();
      syncRangeVisibility();
      updateRangeCount();
    });
  });
  $("range-start-date").addEventListener("change", () => void onTimestampDateChange("start"));
  $("range-end-date").addEventListener("change", () => void onTimestampDateChange("end"));
  $("range-start-time").addEventListener("change", () => void onTimestampTimeChange("start"));
  $("range-end-time").addEventListener("change", () => void onTimestampTimeChange("end"));
  $("load-preview").addEventListener("click", loadPreview);
  $("grid-size").addEventListener("change", () => {
    getGridSize();
    markRoiDirty();
    markAlgorithmDirty();
    drawRoiCanvas();
  });
  $("apply-roi-preset").addEventListener("click", applyRoiPreset);
  $("confirm-roi").addEventListener("click", confirmRoi);
  $("save-roi-new").addEventListener("click", () => saveRoiPreset(false));
  $("save-roi-overwrite").addEventListener("click", () => saveRoiPreset(true));
  $("reload-algorithm-presets").addEventListener("click", loadAlgorithmPresets);
  $("apply-algorithm-preset").addEventListener("click", applyAlgorithmPreset);
  $("confirm-algorithm").addEventListener("click", confirmAlgorithm);
  $("save-algorithm-new").addEventListener("click", () => saveAlgorithmPreset(false));
  $("save-algorithm-overwrite").addEventListener("click", () => saveAlgorithmPreset(true));
  $("compute-backend").addEventListener("change", () => {
    const backend = $("compute-backend").value;
    setStatus(
      "gpu-status",
      backend === "gpu"
        ? "GPU selected. Use Check GPU to verify CUDA before running."
        : "CPU selected. The run will not require PyTorch/CUDA.",
      backend === "gpu" ? "warning" : ""
    );
    markAlgorithmDirty();
  });
  $("check-gpu").addEventListener("click", checkGpuStatus);
  $("start-run").addEventListener("click", startRun);
  $("cancel-run").addEventListener("click", cancelRun);
  $("refresh-runs").addEventListener("click", loadRuns);
  $("load-run").addEventListener("click", loadRunDetails);
  $("metric-search").addEventListener("input", renderMetricList);
  $("reset-plot-zoom").addEventListener("click", () => {
    state.plotZoom = null;
    renderInteractivePlots();
  });
  $("plot-selected").addEventListener("click", plotSelectedMetrics);

  const canvas = $("roi-canvas");
  canvas.addEventListener("mousedown", handleCanvasDown);
  canvas.addEventListener("mousemove", handleCanvasMove);
  window.addEventListener("mouseup", handleCanvasUp);
}

async function init() {
  bindEvents();
  syncRangeVisibility();
  await loadDatasets();
  await loadAlgorithmPresets();
  await loadRuns();
  await loadSharedState();
  drawRoiCanvas();
}

init().catch((error) => {
  console.error(error);
  setStatus("range-summary", error.message, "error");
});
