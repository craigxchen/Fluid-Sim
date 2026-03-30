import init, { WasmFluidApp } from "./pkg/fluid_wasm.js";

const canvas = document.getElementById("fluid-canvas");
const toggleButton = document.getElementById("toggle");
const resetButton = document.getElementById("reset");
const presetSelect = document.getElementById("preset-select");
const presetLabel = document.getElementById("preset");
const particleLabel = document.getElementById("particles");
const fpsLabel = document.getElementById("fps");
const rendererLabel = document.getElementById("renderer");

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
  fpsWindow.push(frameTime);
  if (fpsWindow.length > 20) {
    fpsWindow.shift();
  }

  const average = fpsWindow.reduce((sum, value) => sum + value, 0) / fpsWindow.length;
  fpsLabel.textContent = average > 0 ? Math.round(1 / average).toString() : "0";
}

function animate(timestamp) {
  if (!lastFrameTime) {
    lastFrameTime = timestamp;
  }

  const frameTime = Math.min((timestamp - lastFrameTime) / 1000, 0.05);
  lastFrameTime = timestamp;
  updateFps(frameTime);

  if (!paused) {
    app.stepFrame(frameTime);
  }

  app.render();
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
  toggleButton.textContent = "Pause";
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
  toggleButton.textContent = "Pause";
  syncUi();
  resizeCanvas();
});

await init();
app = await WasmFluidApp.create(canvas);
populatePresetOptions();
syncUi();
rendererLabel.textContent = "wgpu compute + render";
resizeCanvas();
requestAnimationFrame(animate);
