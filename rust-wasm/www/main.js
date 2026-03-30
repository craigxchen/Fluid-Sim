import init, { WasmFluidApp } from "./pkg/fluid_wasm.js";

const canvas = document.getElementById("fluid-canvas");
const toggleButton = document.getElementById("toggle");
const resetButton = document.getElementById("reset");
const presetSelect = document.getElementById("preset-select");
const presetLabel = document.getElementById("preset");
const particleLabel = document.getElementById("particles");
const fpsLabel = document.getElementById("fps");
const simCpuLabel = document.getElementById("sim-cpu");
const renderCpuLabel = document.getElementById("render-cpu");
const rendererLabel = document.getElementById("renderer");
const gridPeakLabel = document.getElementById("grid-peak");
const gridDropsLabel = document.getElementById("grid-drops");

let app;
let paused = false;
let pointerState = {
  active: false,
  button: 0,
  worldX: 0,
  worldY: 0,
};
let lastFrameTime = 0;
let fpsWindow = [];
let simCpuWindow = [];
let renderCpuWindow = [];
let diagnosticsInFlight = false;
let lastDiagnosticsSample = 0;

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
  const scale = Math.min(
    canvas.width / app.boundsWidth(),
    canvas.height / app.boundsHeight(),
  ) * 0.9;

  return {
    x: (px - canvas.width * 0.5) / scale,
    y: (canvas.height * 0.5 - py) / scale,
  };
}

function syncInteraction() {
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
  const world = canvasToWorld(event.clientX, event.clientY);
  pointerState.worldX = world.x;
  pointerState.worldY = world.y;
  syncInteraction();
}

function populatePresetOptions() {
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
  gridPeakLabel.textContent = `0 / ${app.maxParticlesPerCell()}`;
  gridDropsLabel.textContent = "0";
}

canvas.addEventListener("contextmenu", (event) => event.preventDefault());
canvas.addEventListener("pointerdown", (event) => {
  pointerState.active = true;
  pointerState.button = event.button;
  updatePointer(event);
});
canvas.addEventListener("pointermove", updatePointer);
window.addEventListener("pointerup", () => {
  pointerState.active = false;
  app?.clearInteraction();
});
window.addEventListener("resize", resizeCanvas);

toggleButton.addEventListener("click", () => {
  paused = !paused;
  toggleButton.textContent = paused ? "Resume" : "Pause";
});

resetButton.addEventListener("click", () => {
  app.reset();
  paused = false;
  lastFrameTime = 0;
  fpsWindow = [];
  simCpuWindow = [];
  renderCpuWindow = [];
  lastDiagnosticsSample = 0;
  toggleButton.textContent = "Pause";
  void pollDiagnostics(true);
});

presetSelect.addEventListener("change", (event) => {
  const presetIndex = Number(event.target.value);
  if (!app.loadPreset(presetIndex)) {
    return;
  }

  pointerState.active = false;
  app.clearInteraction();
  paused = false;
  lastFrameTime = 0;
  fpsWindow = [];
  simCpuWindow = [];
  renderCpuWindow = [];
  lastDiagnosticsSample = 0;
  toggleButton.textContent = "Pause";
  syncUi();
  resizeCanvas();
  void pollDiagnostics(true);
});

await init();
app = await WasmFluidApp.create(canvas);
populatePresetOptions();
syncUi();
rendererLabel.textContent = "wgpu compute + render";
resizeCanvas();
void pollDiagnostics(true);
requestAnimationFrame(animate);
