import init, { WasmFluidApp, WasmFluid3DApp } from "./pkg/fluid_wasm.js";

const canvas = document.getElementById("fluid-canvas");
const toggleButton = document.getElementById("toggle");
const resetButton = document.getElementById("reset");
const modeSelect = document.getElementById("mode-select");
const presetSelect = document.getElementById("preset-select");
const presetLabel = document.getElementById("preset");
const particleLabel = document.getElementById("particles");
const fpsLabel = document.getElementById("fps");
const simCpuLabel = document.getElementById("sim-cpu");
const renderCpuLabel = document.getElementById("render-cpu");
const rendererLabel = document.getElementById("renderer");
const gridPeakLabel = document.getElementById("grid-peak");
const gridDropsLabel = document.getElementById("grid-drops");
const hintLabel = document.getElementById("mode-hint");

const MODE_2D = "2d";
const MODE_3D = "3d";

let app;
let activeMode = MODE_2D;
let paused = false;
let pointerState = {
  active: false,
  button: 0,
  worldX: 0,
  worldY: 0,
  lastClientX: 0,
  lastClientY: 0,
};
let lastFrameTime = 0;
let fpsWindow = [];
let simCpuWindow = [];
let renderCpuWindow = [];
let diagnosticsInFlight = false;
let lastDiagnosticsSample = 0;

function is3dMode() {
  return activeMode === MODE_3D;
}

function resetFrameStats() {
  lastFrameTime = 0;
  fpsWindow = [];
  simCpuWindow = [];
  renderCpuWindow = [];
  lastDiagnosticsSample = 0;
}

function updateAverage(window, sample, maxSamples = 20) {
  window.push(sample);
  if (window.length > maxSamples) {
    window.shift();
  }

  return window.reduce((sum, value) => sum + value, 0) / window.length;
}

function resizeCanvas() {
  const dpr = window.devicePixelRatio || 1;
  const rect = canvas.getBoundingClientRect();
  canvas.width = Math.round(rect.width * dpr);
  canvas.height = Math.round(rect.height * dpr);
  app?.resize(canvas.width, canvas.height);
}

function canvasToWorld(clientX, clientY) {
  const rect = canvas.getBoundingClientRect();
  const dpr = window.devicePixelRatio || 1;
  const px = (clientX - rect.left) * dpr;
  const py = (clientY - rect.top) * dpr;

  const bw = app.boundsWidth();
  const bh = app.boundsHeight();
  const simAspect = bw / bh;
  const canvasAspect = canvas.width / canvas.height;

  const scaleX = (canvasAspect > simAspect)
    ? canvas.height / bh
    : canvas.width / bw;
  const scaleY = scaleX;

  return {
    x: (px - canvas.width * 0.5) / scaleX,
    y: (canvas.height * 0.5 - py) / scaleY,
  };
}

function syncInteraction() {
  if (is3dMode()) {
    return;
  }

  if (!pointerState.active) {
    app.clearInteraction();
    return;
  }

  const baseStrength = app.interactionStrength();
  const signedStrength = pointerState.button === 2 ? -baseStrength : baseStrength;
  app.setInteraction(
    pointerState.worldX,
    pointerState.worldY,
    signedStrength,
    true,
  );
}

function updateFps(frameTime) {
  const average = updateAverage(fpsWindow, frameTime);
  fpsLabel.textContent = average > 0 ? Math.round(1 / average).toString() : "0";
}

function updateCpuMetric(label, window, duration) {
  const average = updateAverage(window, duration);
  label.textContent = `${average.toFixed(2)} ms`;
}

async function pollDiagnostics(force = false) {
  if (!app || diagnosticsInFlight) {
    return;
  }

  const now = performance.now();
  if (!force && now - lastDiagnosticsSample < 250) {
    return;
  }

  diagnosticsInFlight = true;
  lastDiagnosticsSample = now;

  try {
    const [peak, dropped, overflowedCells, capacity] = (
      await app.readDiagnostics()
    ).map((value) => Number(value));
    gridPeakLabel.textContent = `${peak} / ${capacity}`;
    gridDropsLabel.textContent =
      dropped > 0 ? `${dropped} (${overflowedCells} cells)` : "0";
  } catch (error) {
    console.error("Failed to sample GPU diagnostics", error);
  } finally {
    diagnosticsInFlight = false;
  }
}

function animate(timestamp) {
  if (!lastFrameTime) {
    lastFrameTime = timestamp;
  }

  const frameTime = Math.min((timestamp - lastFrameTime) / 1000, 0.05);
  lastFrameTime = timestamp;
  updateFps(frameTime);

  const simStart = performance.now();
  if (!paused) {
    app.stepFrame(frameTime);
  }
  updateCpuMetric(simCpuLabel, simCpuWindow, performance.now() - simStart);

  const renderStart = performance.now();
  app.render();
  updateCpuMetric(renderCpuLabel, renderCpuWindow, performance.now() - renderStart);
  void pollDiagnostics();
  requestAnimationFrame(animate);
}

function updatePointer(event) {
  if (is3dMode()) {
    if (pointerState.active) {
      const dx = event.clientX - pointerState.lastClientX;
      const dy = event.clientY - pointerState.lastClientY;
      app.orbitCamera(dx, dy);
    }
    pointerState.lastClientX = event.clientX;
    pointerState.lastClientY = event.clientY;
    return;
  }

  const world = canvasToWorld(event.clientX, event.clientY);
  pointerState.worldX = world.x;
  pointerState.worldY = world.y;
  syncInteraction();
}

function populateModeOptions() {
  modeSelect.innerHTML = "";

  const modes = [
    { value: MODE_2D, label: "2D Solver" },
    { value: MODE_3D, label: "3D Preview" },
  ];

  for (const mode of modes) {
    const option = document.createElement("option");
    option.value = mode.value;
    option.textContent = mode.label;
    modeSelect.append(option);
  }
}

function populatePresetOptions() {
  presetSelect.innerHTML = "";
  const count = app.presetCount();

  for (let index = 0; index < count; index += 1) {
    const option = document.createElement("option");
    option.value = String(index);
    option.textContent = app.presetLabel(index);
    presetSelect.append(option);
  }
}

function syncUi() {
  presetLabel.textContent = app.presetName();
  particleLabel.textContent = new Intl.NumberFormat().format(
    app.particleCount(),
  );
  presetSelect.value = String(app.activePreset());
  modeSelect.value = activeMode;
  rendererLabel.textContent = app.rendererName();

  if (is3dMode()) {
    gridPeakLabel.textContent = `0 / ${app.maxParticlesPerCell()}`;
    gridDropsLabel.textContent = "0";
    hintLabel.innerHTML =
      "Drag to orbit the camera and scroll to zoom. This 3D mode now runs a reduced-density GPU SPH solver with simple billboard rendering so the browser path stays focused on functional simulation rather than polished fluid shading. Watch <code>Grid Peak</code> and <code>Dropped</code> while switching presets.";
  } else {
    gridPeakLabel.textContent = `0 / ${app.maxParticlesPerCell()}`;
    gridDropsLabel.textContent = "0";
    hintLabel.innerHTML =
      "Left click attracts. Right click repels. Drag inside the canvas to perturb the flow. <code>Oil &amp; Water</code> demonstrates immiscible multi-fluid simulation with two distinct fluid types that resist mixing. <code>Test C</code> is the heaviest single-fluid scenario. All presets run on the GPU compute solver.";
  }
}

async function createApp(mode) {
  activeMode = mode;
  app = mode === MODE_3D
    ? await WasmFluid3DApp.create(canvas)
    : await WasmFluidApp.create(canvas);
  pointerState.active = false;
  paused = false;
  diagnosticsInFlight = false;
  toggleButton.textContent = "Pause";
  populatePresetOptions();
  syncUi();
  resizeCanvas();
  void pollDiagnostics(true);
}

canvas.addEventListener("contextmenu", (event) => event.preventDefault());
canvas.addEventListener("pointerdown", (event) => {
  pointerState.active = true;
  pointerState.button = event.button;
  pointerState.lastClientX = event.clientX;
  pointerState.lastClientY = event.clientY;
  updatePointer(event);
});
canvas.addEventListener("pointermove", updatePointer);
canvas.addEventListener(
  "wheel",
  (event) => {
    if (!is3dMode()) {
      return;
    }
    event.preventDefault();
    app.zoomCamera(event.deltaY);
  },
  { passive: false },
);
window.addEventListener("pointerup", () => {
  pointerState.active = false;
  if (!is3dMode()) {
    app?.clearInteraction();
  }
});
window.addEventListener("resize", resizeCanvas);

toggleButton.addEventListener("click", () => {
  paused = !paused;
  toggleButton.textContent = paused ? "Resume" : "Pause";
});

resetButton.addEventListener("click", () => {
  app.reset();
  pointerState.active = false;
  resetFrameStats();
  paused = false;
  toggleButton.textContent = "Pause";
  syncUi();
  void pollDiagnostics(true);
});

modeSelect.addEventListener("change", async (event) => {
  resetFrameStats();
  await createApp(event.target.value);
});

presetSelect.addEventListener("change", (event) => {
  const presetIndex = Number(event.target.value);
  if (!app.loadPreset(presetIndex)) {
    return;
  }

  pointerState.active = false;
  if (!is3dMode()) {
    app.clearInteraction();
  }
  paused = false;
  resetFrameStats();
  toggleButton.textContent = "Pause";
  syncUi();
  resizeCanvas();
  void pollDiagnostics(true);
});

await init();
populateModeOptions();
await createApp(MODE_2D);
requestAnimationFrame(animate);
