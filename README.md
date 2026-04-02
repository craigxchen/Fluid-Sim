# Fluid-Sim

This is a rust port of Sebastian Lague's fluid simulator with additional functionality added. For example, this simulation supports mixing of different fluid types.

## Rust + WASM port

This repository now also contains a browser-first Rust/WASM implementation in [`rust-wasm`](./rust-wasm). The current port replaces the Unity runtime for all three 2D Unity test scenes with:

* a Rust `wgpu` SPH solver that mirrors the Unity 2D compute pipeline on the GPU
* scenario switching for `Test A`, `Test B`, and `Test C`
* a `wgpu` particle renderer that draws directly from GPU simulation buffers
* mouse attraction/repulsion controls matching the Unity demo interaction model

The browser app now also includes a first 3D port slice:

* Unity-derived 3D scene presets for the particle, raymarch, and marching-cubes scenes
* a reduced-density Rust `wgpu` 3D SPH solver suitable for the browser bring-up phase
* a simple `wgpu` billboard renderer with orbit camera controls for validating the live 3D simulation
* shared browser UI that lets you switch between the completed 2D solver path and the new functional 3D runtime

### Run the browser version

From the repo root:

```bash
cd rust-wasm
wasm-pack build --target web --out-dir www/pkg
cd www
python3 -m http.server 8080
```

Then open `http://localhost:8080`.

### Migration status

The browser port fully covers the 2D simulation path. The 3D port now has a functional reduced-density GPU simulation with simple billboard rendering, but the higher-end Unity 3D features such as raymarching, foam particles, screen-space fluid rendering, and marching cubes still need to be ported separately.
