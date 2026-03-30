import init, {
  WasmFluidSimulation,
  wasmMemory,
} from "./pkg/fluid_wasm.js";

const canvas = document.getElementById("fluid-canvas");
const toggleButton = document.getElementById("toggle");
const resetButton = document.getElementById("reset");
const presetSelect = document.getElementById("preset-select");
const presetLabel = document.getElementById("preset");
const particleLabel = document.getElementById("particles");
const fpsLabel = document.getElementById("fps");
const rendererLabel = document.getElementById("renderer");

const gl = canvas.getContext("webgl2", { alpha: true, antialias: true });
if (!gl) {
  throw new Error("WebGL2 is required for this viewer.");
}

const PARTICLE_VERTEX_SHADER = `#version 300 es
layout(location = 0) in vec2 a_position;
layout(location = 1) in float a_speed;
layout(location = 2) in float a_density;

uniform vec2 u_bounds;
uniform float u_point_size;

out float v_speed;
out float v_density;

void main() {
  vec2 clip = vec2(
    a_position.x / (u_bounds.x * 0.5),
    a_position.y / (u_bounds.y * 0.5)
  );
  gl_Position = vec4(clip, 0.0, 1.0);
  gl_PointSize = u_point_size;
  v_speed = a_speed;
  v_density = a_density;
}
`;

const PARTICLE_FRAGMENT_SHADER = `#version 300 es
precision highp float;

in float v_speed;
in float v_density;

out vec4 outColor;

vec3 palette(float speed, float density) {
  float speedT = clamp(speed / 8.0, 0.0, 1.0);
  float densityT = clamp(density, 0.0, 1.5) / 1.5;

  vec3 deep = vec3(0.08, 0.24, 0.56);
  vec3 mid = vec3(0.11, 0.66, 0.73);
  vec3 foam = vec3(0.96, 0.97, 0.89);
  vec3 hot = vec3(0.94, 0.42, 0.17);

  vec3 base = mix(deep, mid, densityT);
  vec3 energy = mix(base, foam, speedT * 0.65);
  return mix(energy, hot, speedT * densityT * 0.55);
}

void main() {
  vec2 local = gl_PointCoord * 2.0 - 1.0;
  float radial = dot(local, local);
  if (radial > 1.0) {
    discard;
  }

  float edge = smoothstep(1.0, 0.08, radial);
  float specular = smoothstep(0.85, 0.0, distance(gl_PointCoord, vec2(0.32, 0.3)));
  vec3 color = palette(v_speed, v_density) + specular * 0.08;
  outColor = vec4(color, edge * 0.92);
}
`;

const SHAPE_VERTEX_SHADER = `#version 300 es
layout(location = 0) in vec2 a_clip_position;

void main() {
  gl_Position = vec4(a_clip_position, 0.0, 1.0);
}
`;

const SHAPE_FRAGMENT_SHADER = `#version 300 es
precision highp float;

uniform vec4 u_color;

out vec4 outColor;

void main() {
  outColor = u_color;
}
`;

let simulation;
let memory;
let paused = false;
let pointerState = {
  active: false,
  button: 0,
  worldX: 0,
  worldY: 0,
};
let lastFrameTime = 0;
let fpsWindow = [];
let particleBufferByteLength = 0;

const particleProgram = createProgram(
  gl,
  PARTICLE_VERTEX_SHADER,
  PARTICLE_FRAGMENT_SHADER,
);
const shapeProgram = createProgram(gl, SHAPE_VERTEX_SHADER, SHAPE_FRAGMENT_SHADER);
const particleState = createParticleState(gl, particleProgram);
const shapeState = createShapeState(gl, shapeProgram);

function resizeCanvas() {
  const dpr = window.devicePixelRatio || 1;
  const rect = canvas.getBoundingClientRect();
  canvas.width = Math.round(rect.width * dpr);
  canvas.height = Math.round(rect.height * dpr);
  gl.viewport(0, 0, canvas.width, canvas.height);
}

function worldScale() {
  return Math.min(
    canvas.width / simulation.boundsWidth(),
    canvas.height / simulation.boundsHeight(),
  ) * 0.9;
}

function worldToClip(x, y) {
  return {
    x: x / (simulation.boundsWidth() * 0.5),
    y: y / (simulation.boundsHeight() * 0.5),
  };
}

function canvasToWorld(clientX, clientY) {
  const rect = canvas.getBoundingClientRect();
  const dpr = window.devicePixelRatio || 1;
  const px = (clientX - rect.left) * dpr;
  const py = (clientY - rect.top) * dpr;
  const scale = worldScale();

  return {
    x: (px - canvas.width * 0.5) / scale,
    y: (canvas.height * 0.5 - py) / scale,
  };
}

function syncInteraction() {
  if (!pointerState.active) {
    simulation.clearInteraction();
    return;
  }

  const baseStrength = simulation.interactionStrength();
  const signedStrength = pointerState.button === 2 ? -baseStrength : baseStrength;
  simulation.setInteraction(
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

function createShader(glContext, type, source) {
  const shader = glContext.createShader(type);
  glContext.shaderSource(shader, source);
  glContext.compileShader(shader);

  if (!glContext.getShaderParameter(shader, glContext.COMPILE_STATUS)) {
    const error = glContext.getShaderInfoLog(shader);
    glContext.deleteShader(shader);
    throw new Error(error);
  }

  return shader;
}

function createProgram(glContext, vertexSource, fragmentSource) {
  const vertexShader = createShader(glContext, glContext.VERTEX_SHADER, vertexSource);
  const fragmentShader = createShader(
    glContext,
    glContext.FRAGMENT_SHADER,
    fragmentSource,
  );

  const program = glContext.createProgram();
  glContext.attachShader(program, vertexShader);
  glContext.attachShader(program, fragmentShader);
  glContext.linkProgram(program);

  glContext.deleteShader(vertexShader);
  glContext.deleteShader(fragmentShader);

  if (!glContext.getProgramParameter(program, glContext.LINK_STATUS)) {
    const error = glContext.getProgramInfoLog(program);
    glContext.deleteProgram(program);
    throw new Error(error);
  }

  return program;
}

function createParticleState(glContext, program) {
  const vao = glContext.createVertexArray();
  const buffer = glContext.createBuffer();

  glContext.bindVertexArray(vao);
  glContext.bindBuffer(glContext.ARRAY_BUFFER, buffer);
  glContext.enableVertexAttribArray(0);
  glContext.vertexAttribPointer(0, 2, glContext.FLOAT, false, 16, 0);
  glContext.enableVertexAttribArray(1);
  glContext.vertexAttribPointer(1, 1, glContext.FLOAT, false, 16, 8);
  glContext.enableVertexAttribArray(2);
  glContext.vertexAttribPointer(2, 1, glContext.FLOAT, false, 16, 12);
  glContext.bindVertexArray(null);

  return {
    vao,
    buffer,
    boundsLocation: glContext.getUniformLocation(program, "u_bounds"),
    pointSizeLocation: glContext.getUniformLocation(program, "u_point_size"),
  };
}

function createShapeState(glContext, program) {
  const vao = glContext.createVertexArray();
  const buffer = glContext.createBuffer();

  glContext.bindVertexArray(vao);
  glContext.bindBuffer(glContext.ARRAY_BUFFER, buffer);
  glContext.enableVertexAttribArray(0);
  glContext.vertexAttribPointer(0, 2, glContext.FLOAT, false, 8, 0);
  glContext.bindVertexArray(null);

  return {
    vao,
    buffer,
    colorLocation: glContext.getUniformLocation(program, "u_color"),
  };
}

function particleDataView() {
  const ptr = simulation.particleDataPtr();
  const len = simulation.particleDataLen();
  return new Float32Array(memory.buffer, ptr, len);
}

function uploadParticleData(data) {
  gl.bindBuffer(gl.ARRAY_BUFFER, particleState.buffer);

  if (particleBufferByteLength !== data.byteLength) {
    particleBufferByteLength = data.byteLength;
    gl.bufferData(gl.ARRAY_BUFFER, data, gl.DYNAMIC_DRAW);
    return;
  }

  gl.bufferSubData(gl.ARRAY_BUFFER, 0, data);
}

function drawShape(vertices, mode, color) {
  gl.useProgram(shapeProgram);
  gl.bindVertexArray(shapeState.vao);
  gl.bindBuffer(gl.ARRAY_BUFFER, shapeState.buffer);
  gl.bufferData(gl.ARRAY_BUFFER, vertices, gl.DYNAMIC_DRAW);
  gl.uniform4fv(shapeState.colorLocation, color);
  gl.drawArrays(mode, 0, vertices.length / 2);
}

function drawBounds() {
  const vertices = new Float32Array([
    -1, 1,
    1, 1,
    1, -1,
    -1, -1,
  ]);

  drawShape(vertices, gl.LINE_LOOP, [0.08, 0.2, 0.28, 0.3]);
}

function drawObstacle() {
  const [cx, cy, width, height] = simulation.obstacleData();
  if (width <= 0 || height <= 0) {
    return;
  }

  const topLeft = worldToClip(cx - width * 0.5, cy + height * 0.5);
  const topRight = worldToClip(cx + width * 0.5, cy + height * 0.5);
  const bottomLeft = worldToClip(cx - width * 0.5, cy - height * 0.5);
  const bottomRight = worldToClip(cx + width * 0.5, cy - height * 0.5);

  drawShape(
    new Float32Array([
      topLeft.x, topLeft.y,
      topRight.x, topRight.y,
      bottomLeft.x, bottomLeft.y,
      bottomRight.x, bottomRight.y,
    ]),
    gl.TRIANGLE_STRIP,
    [0.05, 0.2, 0.28, 0.14],
  );

  drawShape(
    new Float32Array([
      topLeft.x, topLeft.y,
      topRight.x, topRight.y,
      bottomRight.x, bottomRight.y,
      bottomLeft.x, bottomLeft.y,
    ]),
    gl.LINE_LOOP,
    [0.05, 0.2, 0.28, 0.32],
  );
}

function drawInteractionRing() {
  if (!pointerState.active) {
    return;
  }

  const segments = 48;
  const radius = simulation.interactionRadius();
  const vertices = new Float32Array(segments * 2);

  for (let index = 0; index < segments; index += 1) {
    const angle = (index / segments) * Math.PI * 2;
    const point = worldToClip(
      pointerState.worldX + Math.cos(angle) * radius,
      pointerState.worldY + Math.sin(angle) * radius,
    );
    vertices[index * 2] = point.x;
    vertices[index * 2 + 1] = point.y;
  }

  drawShape(vertices, gl.LINE_LOOP, [0.87, 0.42, 0.15, 0.45]);
}

function drawParticles() {
  const particleData = particleDataView();
  uploadParticleData(particleData);

  gl.useProgram(particleProgram);
  gl.bindVertexArray(particleState.vao);
  gl.uniform2f(
    particleState.boundsLocation,
    simulation.boundsWidth(),
    simulation.boundsHeight(),
  );
  gl.uniform1f(
    particleState.pointSizeLocation,
    Math.max(1.5, simulation.particleRadius() * worldScale() * 2.0),
  );
  gl.drawArrays(gl.POINTS, 0, simulation.particleCount());
}

function render() {
  gl.clearColor(0, 0, 0, 0);
  gl.clear(gl.COLOR_BUFFER_BIT);

  drawObstacle();
  drawBounds();
  drawInteractionRing();
  drawParticles();
}

function animate(timestamp) {
  if (!lastFrameTime) {
    lastFrameTime = timestamp;
  }

  const frameTime = Math.min((timestamp - lastFrameTime) / 1000, 0.05);
  lastFrameTime = timestamp;
  updateFps(frameTime);

  if (!paused) {
    simulation.stepFrame(frameTime);
  }

  render();
  requestAnimationFrame(animate);
}

function updatePointer(event) {
  const world = canvasToWorld(event.clientX, event.clientY);
  pointerState.worldX = world.x;
  pointerState.worldY = world.y;
  syncInteraction();
}

function populatePresetOptions() {
  const count = simulation.presetCount();

  for (let index = 0; index < count; index += 1) {
    const option = document.createElement("option");
    option.value = String(index);
    option.textContent = simulation.presetLabel(index);
    presetSelect.append(option);
  }
}

function syncUi() {
  presetLabel.textContent = simulation.presetName();
  particleLabel.textContent = new Intl.NumberFormat().format(
    simulation.particleCount(),
  );
  presetSelect.value = String(simulation.activePreset());
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
  simulation?.clearInteraction();
});
window.addEventListener("resize", resizeCanvas);

toggleButton.addEventListener("click", () => {
  paused = !paused;
  toggleButton.textContent = paused ? "Resume" : "Pause";
});

resetButton.addEventListener("click", () => {
  simulation.reset();
  paused = false;
  lastFrameTime = 0;
  toggleButton.textContent = "Pause";
});

presetSelect.addEventListener("change", (event) => {
  const presetIndex = Number(event.target.value);
  if (!simulation.loadPreset(presetIndex)) {
    return;
  }

  pointerState.active = false;
  simulation.clearInteraction();
  paused = false;
  lastFrameTime = 0;
  fpsWindow = [];
  particleBufferByteLength = 0;
  toggleButton.textContent = "Pause";
  syncUi();
  resizeCanvas();
});

await init();
memory = wasmMemory();
simulation = new WasmFluidSimulation();
populatePresetOptions();
syncUi();
rendererLabel.textContent = "WebGL2";
gl.enable(gl.BLEND);
gl.blendFunc(gl.SRC_ALPHA, gl.ONE_MINUS_SRC_ALPHA);
gl.disable(gl.DEPTH_TEST);
resizeCanvas();
requestAnimationFrame(animate);
