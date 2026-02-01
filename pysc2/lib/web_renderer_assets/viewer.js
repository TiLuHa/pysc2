const statusEl = document.getElementById("status");
const screenCanvas = document.getElementById("screen");
const minimapCanvas = document.getElementById("minimap");
const screenModeEl = document.getElementById("screenMode");
const minimapModeEl = document.getElementById("minimapMode");
const screenLayerEl = document.getElementById("screenLayer");
const minimapLayerEl = document.getElementById("minimapLayer");
const showEffectsEl = document.getElementById("showEffects");
const showUnitsEl = document.getElementById("showUnits");
const showLabelsEl = document.getElementById("showLabels");
const showCameraEl = document.getElementById("showCamera");
const inspectorDetailsEl = document.getElementById("inspectorDetails");
const legendPanelEl = document.getElementById("legendPanel");
const controlStatusEl = document.getElementById("controlStatus");
const startButtonEl = document.getElementById("startButton");
const stopButtonEl = document.getElementById("stopButton");
const stepButtonEl = document.getElementById("stepButton");
const hudPanelEl = document.getElementById("hudPanel");
const alertsPanelEl = document.getElementById("alertsPanel");
const selectionPanelEl = document.getElementById("selectionPanel");

const windowScale = 0.78;
let lastScreenTarget = null;
let palettes = {screen: {}, minimap: {}};
let layerNames = {screen: [], minimap: []};
let layerMeta = {screen: {}, minimap: {}};
let lastFrame = null;
let controlState = {running: true};

const state = {
  screenMode: "composite",
  minimapMode: "composite",
  screenLayer: "height_map",
  minimapLayer: "height_map",
  effects: true,
  units: true,
  labels: true,
  camera: true,
};

function b64ToBytes(b64) {
  const bin = atob(b64);
  const len = bin.length;
  const bytes = new Uint8Array(len);
  for (let i = 0; i < len; i++) bytes[i] = bin.charCodeAt(i);
  return bytes;
}

function decodeLayer(layer) {
  const bytes = b64ToBytes(layer.data);
  const buf = bytes.buffer.slice(bytes.byteOffset, bytes.byteOffset + bytes.byteLength);
  let arr;
  if (layer.dtype === "u16") arr = new Uint16Array(buf);
  else if (layer.dtype === "u32") arr = new Uint32Array(buf);
  else arr = new Uint8Array(buf);
  return {name: layer.name, w: layer.w, h: layer.h, arr: arr};
}

function decodeLayers(list) {
  const out = {};
  if (!list) return out;
  for (const layer of list) {
    out[layer.name] = decodeLayer(layer);
  }
  return out;
}

function snapScale(scale) {
  if (scale >= 1) {
    return Math.max(1, Math.floor(scale));
  }
  return Math.max(0.25, scale);
}

function computeTarget(frame, kind, canvas) {
  const cardWidth = Math.max(
      1,
      (canvas && canvas.parentElement && canvas.parentElement.clientWidth) ?
          canvas.parentElement.clientWidth - 2 :
          window.innerWidth * windowScale);
  if (kind === "screen") {
    const targetW = cardWidth;
    const targetH = window.innerHeight * windowScale;
    const scale = snapScale(Math.min(targetW / frame.w, targetH / frame.h));
    const w = Math.max(1, Math.floor(frame.w * scale));
    const h = Math.max(1, Math.floor(frame.h * scale));
    lastScreenTarget = [w, h];
    return lastScreenTarget;
  }
  if (kind === "minimap" && lastScreenTarget) {
    const targetW = Math.min(cardWidth, Math.max(180, lastScreenTarget[0] / 4));
    const targetH = Math.max(180, lastScreenTarget[1] / 4);
    const scale = snapScale(Math.min(targetW / frame.w, targetH / frame.h));
    return [Math.max(1, Math.floor(frame.w * scale)),
            Math.max(1, Math.floor(frame.h * scale))];
  }
  return null;
}

function drawImage(canvas, imageData, targetPx, kind) {
  if (!imageData) return null;
  let targetW = imageData.width;
  let targetH = imageData.height;
  if (targetPx && targetPx.length === 2) {
    const scale = snapScale(Math.min(
        targetPx[0] / imageData.width,
        targetPx[1] / imageData.height));
    targetW = Math.max(1, Math.floor(imageData.width * scale));
    targetH = Math.max(1, Math.floor(imageData.height * scale));
  } else {
    const autoTarget = computeTarget(
        {w: imageData.width, h: imageData.height}, kind, canvas);
    if (autoTarget) {
      targetW = autoTarget[0];
      targetH = autoTarget[1];
    }
  }
  const sourceCanvas = document.createElement("canvas");
  sourceCanvas.width = imageData.width;
  sourceCanvas.height = imageData.height;
  sourceCanvas.getContext("2d").putImageData(imageData, 0, 0);
  canvas.width = targetW;
  canvas.height = targetH;
  canvas.style.width = targetW + "px";
  canvas.style.height = targetH + "px";
  const ctx = canvas.getContext("2d");
  ctx.imageSmoothingEnabled = false;
  ctx.webkitImageSmoothingEnabled = false;
  ctx.mozImageSmoothingEnabled = false;
  ctx.msImageSmoothingEnabled = false;
  ctx.clearRect(0, 0, targetW, targetH);
  ctx.drawImage(sourceCanvas, 0, 0, targetW, targetH);
  ctx.setTransform(
      targetW / imageData.width, 0,
      0, targetH / imageData.height,
      0, 0);
  return ctx;
}

function frameToImage(frame) {
  if (!frame || !frame.data) return null;
  const bytes = b64ToBytes(frame.data);
  const img = new ImageData(frame.w, frame.h);
  for (let i = 0, j = 0; i < bytes.length; i += 3, j += 4) {
    img.data[j] = bytes[i];
    img.data[j + 1] = bytes[i + 1];
    img.data[j + 2] = bytes[i + 2];
    img.data[j + 3] = 255;
  }
  return img;
}

function renderLayer(layer, palette) {
  if (!layer || !palette) return null;
  const img = new ImageData(layer.w, layer.h);
  const data = img.data;
  const allZeroHeightMap =
      layer.name === "height_map" && !layer.arr.some((value) => value !== 0);
  for (let i = 0; i < layer.arr.length; i++) {
    const rawValue = allZeroHeightMap ? 100 : layer.arr[i];
    const value = rawValue * 3;
    const offset = i * 4;
    data[offset] = palette[value] || 0;
    data[offset + 1] = palette[value + 1] || 0;
    data[offset + 2] = palette[value + 2] || 0;
    data[offset + 3] = 255;
  }
  return img;
}

function compositeScreen(layers) {
  const hmap = layers.height_map;
  const creep = layers.creep;
  const power = layers.power;
  const vis = layers.visibility_map;
  if (!hmap || !creep || !power || !vis) return null;
  const palH = palettes.screen.height_map;
  const palC = palettes.screen.creep;
  const palP = palettes.screen.power;
  const fade = [0.5, 0.75, 1.0];
  const img = new ImageData(hmap.w, hmap.h);
  const allZeroHeightMap = !hmap.arr.some((value) => value !== 0);
  for (let i = 0; i < hmap.arr.length; i++) {
    const value = (allZeroHeightMap ? 100 : hmap.arr[i]) * 3;
    const offset = i * 4;
    let r = (palH[value] || 0) * 0.6;
    let g = (palH[value + 1] || 0) * 0.6;
    let b = (palH[value + 2] || 0) * 0.6;
    if (creep.arr[i] > 0) {
      const c = creep.arr[i] * 3;
      r = 0.4 * r + 0.6 * (palC[c] || 0);
      g = 0.4 * g + 0.6 * (palC[c + 1] || 0);
      b = 0.4 * b + 0.6 * (palC[c + 2] || 0);
    }
    if (power.arr[i] > 0) {
      const p = power.arr[i] * 3;
      r = 0.7 * r + 0.3 * (palP[p] || 0);
      g = 0.7 * g + 0.3 * (palP[p + 1] || 0);
      b = 0.7 * b + 0.3 * (palP[p + 2] || 0);
    }
    const mult = fade[vis.arr[i]] || 1;
    img.data[offset] = r * mult;
    img.data[offset + 1] = g * mult;
    img.data[offset + 2] = b * mult;
    img.data[offset + 3] = 255;
  }
  return img;
}

function compositeMinimap(layers) {
  const hmap = layers.height_map;
  const creep = layers.creep;
  const vis = layers.visibility_map;
  if (!hmap || !creep || !vis) return null;
  const palH = palettes.minimap.height_map;
  const palC = palettes.minimap.creep;
  const fade = [0.5, 0.75, 1.0];
  const img = new ImageData(hmap.w, hmap.h);
  const allZeroHeightMap = !hmap.arr.some((value) => value !== 0);
  for (let i = 0; i < hmap.arr.length; i++) {
    const value = (allZeroHeightMap ? 100 : hmap.arr[i]) * 3;
    const offset = i * 4;
    let r = (palH[value] || 0) * 0.6;
    let g = (palH[value + 1] || 0) * 0.6;
    let b = (palH[value + 2] || 0) * 0.6;
    if (creep.arr[i] > 0) {
      const c = creep.arr[i] * 3;
      r = 0.4 * r + 0.6 * (palC[c] || 0);
      g = 0.4 * g + 0.6 * (palC[c + 1] || 0);
      b = 0.4 * b + 0.6 * (palC[c + 2] || 0);
    }
    const mult = fade[vis.arr[i]] || 1;
    img.data[offset] = r * mult;
    img.data[offset + 1] = g * mult;
    img.data[offset + 2] = b * mult;
    img.data[offset + 3] = 255;
  }
  return img;
}

function drawEffects(ctx, effects) {
  if (!ctx || !effects) return;
  ctx.save();
  const scale = Math.max(1, ctx.getTransform().a || 1);
  ctx.lineWidth = 1 / scale;
  for (const effect of effects) {
    ctx.strokeStyle = `rgba(${effect.color[0]}, ${effect.color[1]}, ${effect.color[2]}, 0.55)`;
    ctx.beginPath();
    ctx.arc(
        effect.x, effect.y,
        Math.max(0.75 / scale, effect.radius),
        0, Math.PI * 2);
    ctx.stroke();
  }
  ctx.restore();
}

function drawUnits(ctx, units) {
  if (!ctx || !units) return;
  ctx.save();
  const scale = Math.max(1, ctx.getTransform().a || 1);
  for (const unit of units) {
    ctx.fillStyle = `rgba(${unit.color[0]}, ${unit.color[1]}, ${unit.color[2]}, 0.82)`;
    ctx.strokeStyle = `rgb(${unit.color[0]}, ${unit.color[1]}, ${unit.color[2]})`;
    ctx.lineWidth = (unit.selected ? 2 : 1) / scale;
    ctx.beginPath();
    ctx.arc(unit.x, unit.y, Math.max(1, unit.radius), 0, Math.PI * 2);
    ctx.fill();
    ctx.stroke();
  }
  ctx.restore();

  if (!state.labels) return;

  ctx.save();
  ctx.setTransform(1, 0, 0, 1, 0, 0);
  ctx.font = "8px IBM Plex Sans, sans-serif";
  ctx.textAlign = "center";
  ctx.textBaseline = "top";
  for (const unit of units) {
    if (!unit.name) continue;
    const x = unit.x * scale;
    const y = unit.y * scale;
    const radius = Math.max(1, unit.radius * scale);
    ctx.fillStyle = "rgba(255,255,255,0.95)";
    ctx.fillText(unit.name, x, y + radius);
    if (unit.detail) {
      ctx.fillStyle = "rgba(180,210,255,0.92)";
      ctx.fillText(unit.detail, x, y + radius + 8);
    }
  }
  ctx.restore();
}

function drawCamera(ctx, camera) {
  if (!ctx || !camera) return;
  const [x, y, w, h] = camera;
  ctx.save();
  ctx.strokeStyle = "rgba(255,255,255,0.82)";
  ctx.lineWidth = 1;
  ctx.strokeRect(x, y, w, h);
  ctx.restore();
}

function drawActionMarkers(ctx, markers) {
  if (!ctx || !markers) return;
  ctx.save();
  const scale = Math.max(1, ctx.getTransform().a || 1);
  for (const marker of markers) {
    ctx.fillStyle = "rgba(255,255,255,0.78)";
    ctx.strokeStyle = "rgba(255,255,255,0.92)";
    ctx.lineWidth = 1 / scale;
    if (marker.kind === "circle") {
      ctx.beginPath();
      ctx.arc(
          marker.x, marker.y,
          Math.max(1.1 / scale, marker.radius / scale),
          0, Math.PI * 2);
      ctx.fill();
      ctx.stroke();
    } else if (marker.kind === "rect") {
      ctx.strokeRect(marker.x, marker.y, marker.w, marker.h);
    }
  }
  ctx.restore();
}

function drawUnitPaths(ctx, paths) {
  if (!ctx || !paths) return;
  ctx.save();
  const scale = Math.max(1, ctx.getTransform().a || 1);
  ctx.lineWidth = 1.25 / scale;
  for (const path of paths) {
    ctx.strokeStyle = "rgba(255,255,255,0.88)";
    for (const segment of path.segments || []) {
      ctx.beginPath();
      ctx.moveTo(segment.x1, segment.y1);
      ctx.lineTo(segment.x2, segment.y2);
      ctx.stroke();
    }
  }
  ctx.restore();
}

function rgbCss(color) {
  return `rgb(${color[0] || 0}, ${color[1] || 0}, ${color[2] || 0})`;
}

function swatchForValue(palette, value) {
  const idx = Math.max(0, value * 3);
  return [
    palette[idx] || 0,
    palette[idx + 1] || 0,
    palette[idx + 2] || 0,
  ];
}

function gradientForPalette(palette, scale) {
  const maxIndex = Math.max(1, scale - 1);
  const stops = [];
  for (const ratio of [0, 0.25, 0.5, 0.75, 1]) {
    const value = Math.min(maxIndex, Math.round(maxIndex * ratio));
    stops.push(`${rgbCss(swatchForValue(palette, value))} ${Math.round(ratio * 100)}%`);
  }
  return `linear-gradient(90deg, ${stops.join(", ")})`;
}

function renderLegendGroup(kind, layerName) {
  const meta = layerMeta[kind]?.[layerName];
  const palette = palettes[kind]?.[layerName];
  if (!meta || !palette) return null;

  const group = document.createElement("div");
  group.className = "legend-group";

  const title = document.createElement("div");
  title.className = "legend-title";
  title.textContent = kind === "screen" ? "Screen Layer" : "Minimap Layer";
  group.appendChild(title);

  const subtitle = document.createElement("div");
  subtitle.className = "legend-subtitle";
  subtitle.textContent = layerName;
  group.appendChild(subtitle);

  if (meta.type === "scalar") {
    const gradient = document.createElement("div");
    gradient.className = "legend-gradient";
    gradient.style.background = gradientForPalette(palette, meta.scale || 1);
    group.appendChild(gradient);

    const scale = document.createElement("div");
    scale.className = "legend-scale";
    scale.innerHTML = `<span>0</span><span>${Math.max(0, (meta.scale || 1) - 1)}</span>`;
    group.appendChild(scale);
    return group;
  }

  const items = document.createElement("div");
  items.className = "legend-items";
  const labels = meta.legend_labels || [];
  if (labels.length > 0 && labels.length <= 16) {
    for (const entry of labels) {
      const item = document.createElement("div");
      item.className = "legend-item";
      const swatch = document.createElement("div");
      swatch.className = "legend-swatch";
      swatch.style.background = rgbCss(swatchForValue(palette, entry.value));
      const text = document.createElement("div");
      text.textContent = `${entry.value}: ${entry.label}`;
      item.appendChild(swatch);
      item.appendChild(text);
      items.appendChild(item);
    }
    group.appendChild(items);
    return group;
  }

  const summary = document.createElement("div");
  summary.className = "empty";
  summary.textContent =
      `Categorical layer with values 0..${Math.max(0, (meta.scale || 1) - 1)}.`;
  group.appendChild(summary);
  return group;
}

function renderLegend() {
  if (!legendPanelEl) return;
  legendPanelEl.innerHTML = "";

  if (!lastFrame || lastFrame.mode !== "feature") {
    legendPanelEl.innerHTML = `<div class="empty">Legend is available for feature layers.</div>`;
    return;
  }

  const groups = [];
  if (state.screenMode === "layer") {
    groups.push(renderLegendGroup("screen", state.screenLayer));
  }
  if (state.minimapMode === "layer") {
    groups.push(renderLegendGroup("minimap", state.minimapLayer));
  }

  const visibleGroups = groups.filter(Boolean);
  if (visibleGroups.length === 0) {
    legendPanelEl.innerHTML = `<div class="empty">Switch a view to single layer to see its legend.</div>`;
    return;
  }
  for (const group of visibleGroups) {
    legendPanelEl.appendChild(group);
  }
}

function updateStatus(msg) {
  const parts = ["Connected"];
  if (msg.frame_id !== undefined) parts.push("frame: " + msg.frame_id);
  if (msg.game_loop !== undefined) parts.push("game_loop: " + msg.game_loop);
  if (msg.mode) parts.push("mode: " + msg.mode);
  statusEl.textContent = parts.join(" | ");
}

function buildSelect(el, names, selected) {
  el.innerHTML = "";
  for (const name of names) {
    const opt = document.createElement("option");
    opt.value = name;
    opt.textContent = name;
    el.appendChild(opt);
  }
  if (selected && names.includes(selected)) {
    el.value = selected;
  } else if (names.length > 0) {
    el.value = names[0];
  }
}

function applyDefaults(view) {
  if (!view) return;
  state.screenMode = view.screen_mode || state.screenMode;
  state.minimapMode = view.minimap_mode || state.minimapMode;
  state.screenLayer = view.screen_layer || state.screenLayer;
  state.minimapLayer = view.minimap_layer || state.minimapLayer;
  state.effects = view.effects !== undefined ? view.effects : state.effects;
  state.units = view.units !== undefined ? view.units : state.units;
  state.labels = view.labels !== undefined ? view.labels : state.labels;
  state.camera = view.camera !== undefined ? view.camera : state.camera;
  if (!state.units) {
    state.labels = false;
  }
}

function updateControls() {
  screenModeEl.value = state.screenMode;
  minimapModeEl.value = state.minimapMode;
  screenLayerEl.value = state.screenLayer;
  minimapLayerEl.value = state.minimapLayer;
  showEffectsEl.checked = state.effects;
  showUnitsEl.checked = state.units;
  showLabelsEl.checked = state.labels;
  showCameraEl.checked = state.camera;
  showLabelsEl.disabled = !state.units;
}

function syncInspectorForMode() {
  const isFeature = !lastFrame || lastFrame.mode === "feature";
  screenModeEl.disabled = !isFeature;
  minimapModeEl.disabled = !isFeature;
  screenLayerEl.disabled = !isFeature || state.screenMode !== "layer";
  minimapLayerEl.disabled = !isFeature || state.minimapMode !== "layer";
  showEffectsEl.disabled = false;
  if (!isFeature && inspectorDetailsEl && !inspectorDetailsEl.open) {
    return;
  }
}

function neededScreenLayers() {
  if (state.screenMode === "composite") {
    return ["height_map", "creep", "power", "visibility_map"];
  }
  return [state.screenLayer];
}

function neededMinimapLayers() {
  if (state.minimapMode === "composite") {
    return ["height_map", "creep", "visibility_map"];
  }
  return [state.minimapLayer];
}

function sendConfig() {
  const screenLayers = neededScreenLayers().filter((name) => layerNames.screen.includes(name));
  const minimapLayers = neededMinimapLayers().filter((name) => layerNames.minimap.includes(name));
  const cfg = {
    type: "config",
    screen_layers: screenLayers,
    minimap_layers: minimapLayers,
    effects: state.effects,
    units: state.units,
    labels: state.units ? state.labels : false,
    camera: state.camera,
  };
  if (ws.readyState === 1) {
    ws.send(JSON.stringify(cfg));
  }
}

function renderHud(hud) {
  const stats = [
    ["Minerals", hud?.minerals ?? "-"],
    ["Vespene", hud?.vespene ?? "-"],
    ["Food", hud ? `${hud.food_used} / ${hud.food_cap}` : "-"],
    ["Score", hud?.score ?? "-"],
    ["Game Time", hud?.time ?? "-"],
    ["Step", hud?.step ?? "-"],
    ["Game Rate", hud ? `${hud.game_rate}/s` : "-"],
    ["Observed FPS", hud?.observed_fps ?? "-"],
    ["Render FPS", hud?.render_fps ?? "-"],
    ["APM", hud?.apm ?? "-"],
    ["EPM", hud?.epm ?? "-"],
  ];
  hudPanelEl.innerHTML = "";
  for (const [label, value] of stats) {
    const el = document.createElement("div");
    el.className = "stat";
    el.innerHTML = `<span class="label">${label}</span><span class="value">${value}</span>`;
    hudPanelEl.appendChild(el);
  }
}

function renderAlerts(alerts) {
  alertsPanelEl.innerHTML = "";
  if (!alerts || alerts.length === 0) {
    alertsPanelEl.innerHTML = `<div class="empty">No active alerts.</div>`;
    return;
  }
  const list = document.createElement("div");
  list.className = "list";
  for (const alert of alerts) {
    const el = document.createElement("div");
    el.className = "list-item alert";
    el.textContent = alert;
    list.appendChild(el);
  }
  alertsPanelEl.appendChild(list);
}

function renderSections(container, sections, emptyText) {
  container.innerHTML = "";
  if (!sections || sections.length === 0) {
    container.innerHTML = `<div class="empty">${emptyText}</div>`;
    return;
  }
  for (const section of sections) {
    const sectionEl = document.createElement("div");
    sectionEl.className = "section";
    if (section.title) {
      const title = document.createElement("div");
      title.className = "section-title";
      title.textContent = section.title;
      sectionEl.appendChild(title);
    }
    if (!section.entries || section.entries.length === 0) {
      const empty = document.createElement("div");
      empty.className = "empty";
      empty.textContent = "No entries.";
      sectionEl.appendChild(empty);
    } else {
      const list = document.createElement("div");
      list.className = "list";
      for (const entry of section.entries) {
        const item = document.createElement("div");
        item.className = "list-item";
        item.textContent = entry;
        list.appendChild(item);
      }
      sectionEl.appendChild(list);
    }
    container.appendChild(sectionEl);
  }
}

function renderPanels(frame) {
  renderHud(frame.hud || null);
  renderAlerts(frame.alerts || []);
  renderSections(selectionPanelEl, frame.selectionPanel || [], "Nothing selected.");
  renderLegend();
  renderControls(frame.controlState || controlState);
}

function renderControls(nextState) {
  if (nextState) {
    controlState = {running: !!nextState.running};
  }
  if (!controlStatusEl) return;
  controlStatusEl.textContent = controlState.running ? "Running" : "Paused";
  if (startButtonEl) startButtonEl.disabled = controlState.running;
  if (stopButtonEl) stopButtonEl.disabled = !controlState.running;
}

function renderFrame() {
  if (!lastFrame) return;
  syncInspectorForMode();
  if (lastFrame.mode === "rgb") {
    const screenImg = frameToImage(lastFrame.screen);
    const minimapImg = frameToImage(lastFrame.minimap);
    const screenCtx = drawImage(screenCanvas, screenImg, lastFrame.screen_px, "screen");
    const minimapCtx = drawImage(minimapCanvas, minimapImg, lastFrame.minimap_px, "minimap");
    if (state.effects) {
      drawEffects(screenCtx, lastFrame.effects);
      drawUnitPaths(screenCtx, lastFrame.unitPaths);
      drawActionMarkers(screenCtx, lastFrame.actionMarkers?.screen);
      drawActionMarkers(minimapCtx, lastFrame.actionMarkers?.minimap);
    }
    if (state.units) {
      drawUnits(screenCtx, lastFrame.units);
    }
    if (state.camera) drawCamera(minimapCtx, lastFrame.camera);
    renderPanels(lastFrame);
    updateStatus(lastFrame);
    return;
  }

  const screenLayers = lastFrame.screenLayers;
  const minimapLayers = lastFrame.minimapLayers;
  let screenImage = null;
  let minimapImage = null;
  if (state.screenMode === "composite") {
    screenImage = compositeScreen(screenLayers);
  } else {
    screenImage = renderLayer(screenLayers[state.screenLayer], palettes.screen[state.screenLayer]);
  }
  if (state.minimapMode === "composite") {
    minimapImage = compositeMinimap(minimapLayers);
  } else {
    minimapImage = renderLayer(minimapLayers[state.minimapLayer], palettes.minimap[state.minimapLayer]);
  }
  const screenCtx = drawImage(screenCanvas, screenImage, lastFrame.screen_px, "screen");
  const minimapCtx = drawImage(minimapCanvas, minimapImage, lastFrame.minimap_px, "minimap");
  if (state.effects) {
    drawEffects(screenCtx, lastFrame.effects);
    drawUnitPaths(screenCtx, lastFrame.unitPaths);
    drawActionMarkers(screenCtx, lastFrame.actionMarkers?.screen);
    drawActionMarkers(minimapCtx, lastFrame.actionMarkers?.minimap);
  }
  if (state.units) {
    drawUnits(screenCtx, lastFrame.units);
  }
  if (state.camera) drawCamera(minimapCtx, lastFrame.camera);
  renderPanels(lastFrame);
  updateStatus(lastFrame);
}

function attachHandlers() {
  screenModeEl.onchange = () => {
    state.screenMode = screenModeEl.value;
    updateControls();
    syncInspectorForMode();
    sendConfig();
    renderFrame();
  };
  minimapModeEl.onchange = () => {
    state.minimapMode = minimapModeEl.value;
    updateControls();
    syncInspectorForMode();
    sendConfig();
    renderFrame();
  };
  screenLayerEl.onchange = () => { state.screenLayer = screenLayerEl.value; sendConfig(); renderFrame(); };
  minimapLayerEl.onchange = () => { state.minimapLayer = minimapLayerEl.value; sendConfig(); renderFrame(); };
  showEffectsEl.onchange = () => { state.effects = showEffectsEl.checked; sendConfig(); renderFrame(); };
  showUnitsEl.onchange = () => {
    state.units = showUnitsEl.checked;
    if (!state.units) {
      state.labels = false;
    }
    updateControls();
    sendConfig();
    renderFrame();
  };
  showLabelsEl.onchange = () => { state.labels = showLabelsEl.checked; sendConfig(); renderFrame(); };
  showCameraEl.onchange = () => { state.camera = showCameraEl.checked; sendConfig(); renderFrame(); };
  if (startButtonEl) {
    startButtonEl.onclick = () => {
      controlState.running = true;
      renderControls(controlState);
      if (ws.readyState === 1) {
        ws.send(JSON.stringify({type: "control", command: "start"}));
      }
    };
  }
  if (stopButtonEl) {
    stopButtonEl.onclick = () => {
      controlState.running = false;
      renderControls(controlState);
      if (ws.readyState === 1) {
        ws.send(JSON.stringify({type: "control", command: "stop"}));
      }
    };
  }
  if (stepButtonEl) {
    stepButtonEl.onclick = () => {
      controlState.running = false;
      renderControls(controlState);
      if (ws.readyState === 1) {
        ws.send(JSON.stringify({type: "control", command: "step"}));
      }
    };
  }
  window.onresize = () => { renderFrame(); };
}

const wsProto = location.protocol === "https:" ? "wss://" : "ws://";
const ws = new WebSocket(wsProto + location.host + "/ws");

ws.onopen = () => {
  statusEl.textContent = "Connected";
};

ws.onclose = () => {
  statusEl.textContent = "Disconnected";
};

ws.onerror = () => {
  statusEl.textContent = "Error";
};

ws.onmessage = (evt) => {
  const msg = JSON.parse(evt.data);
  if (msg.type === "hello") {
    statusEl.textContent = "Connected | hello";
    return;
  }
  if (msg.type === "static") {
    layerNames.screen = msg.screen_layers.map((layer) => layer.name);
    layerNames.minimap = msg.minimap_layers.map((layer) => layer.name);
    for (const layer of msg.screen_layers) {
      palettes.screen[layer.name] = new Uint8Array(layer.palette);
      layerMeta.screen[layer.name] = layer;
    }
    for (const layer of msg.minimap_layers) {
      palettes.minimap[layer.name] = new Uint8Array(layer.palette);
      layerMeta.minimap[layer.name] = layer;
    }
    applyDefaults(msg.default_view);
    buildSelect(screenLayerEl, layerNames.screen, state.screenLayer);
    buildSelect(minimapLayerEl, layerNames.minimap, state.minimapLayer);
    updateControls();
    syncInspectorForMode();
    renderLegend();
    renderControls(controlState);
    attachHandlers();
    sendConfig();
    return;
  }
  if (msg.type !== "frame") return;

  if (msg.mode === "rgb") {
    lastFrame = {
      mode: "rgb",
      screen: msg.screen,
      minimap: msg.minimap,
      screen_px: msg.screen_px,
      minimap_px: msg.minimap_px,
      units: msg.units || [],
      effects: msg.effects || [],
      camera: msg.camera,
      hud: msg.hud,
      alerts: msg.alerts || [],
      selectionPanel: msg.selection_panel || [],
      unitPaths: msg.unit_paths || [],
      actionMarkers: msg.action_markers || {screen: [], minimap: []},
      controlState: msg.control_state || controlState,
      frame_id: msg.frame_id,
      game_loop: msg.game_loop,
    };
    renderFrame();
    return;
  }

  lastFrame = {
    mode: "feature",
    screenLayers: decodeLayers(msg.screen_layers),
    minimapLayers: decodeLayers(msg.minimap_layers),
    screen_px: msg.screen_px,
    minimap_px: msg.minimap_px,
    units: msg.units || [],
    effects: msg.effects || [],
    camera: msg.camera,
    hud: msg.hud,
    alerts: msg.alerts || [],
    selectionPanel: msg.selection_panel || [],
    unitPaths: msg.unit_paths || [],
    actionMarkers: msg.action_markers || {screen: [], minimap: []},
    controlState: msg.control_state || controlState,
    frame_id: msg.frame_id,
    game_loop: msg.game_loop,
  };
  renderFrame();
};
