# Fluid-Sim

Development videos: [Simulation](https://youtu.be/rSKMYc1CQHE?si=KNw_i1sN2_CWEmzA) and [Rendering](https://youtu.be/kOkfC5fLfgE?si=1hXtw9nIiHllA6gn).
</br>Project created in Unity 2022.3

![Fluid Simulation](https://raw.githubusercontent.com/SebLague/Images/master/Fluid%20vid%20thumb.jpg)
![Fluid Rendering](https://raw.githubusercontent.com/SebLague/Images/refs/heads/master/FluidRendering.jpg)

With thanks to the following papers:
* Simulation:
* https://matthias-research.github.io/pages/publications/sca03.pdf
* https://web.archive.org/web/20250106201614/http://www.ligum.umontreal.ca/Clavet-2005-PVFS/pvfs.pdf
* https://sph-tutorial.physics-simulation.org/pdf/SPH_Tutorial.pdf
* https://web.archive.org/web/20140725014123/https://docs.nvidia.com/cuda/samples/5_Simulations/particles/doc/particles.pdf
* Rendering:
* https://developer.download.nvidia.com/presentations/2010/gdc/Direct3D_Effects.pdf
* https://cg.informatik.uni-freiburg.de/publications/2012_CGI_sprayFoamBubbles.pdf

## Rust + WASM port

This repository now also contains a browser-first Rust/WASM implementation in [`rust-wasm`](./rust-wasm). The current port replaces the Unity runtime for all three 2D Unity test scenes with:

* a Rust `wgpu` SPH solver that mirrors the Unity 2D compute pipeline on the GPU
* scenario switching for `Test A`, `Test B`, and `Test C`
* a `wgpu` particle renderer that draws directly from GPU simulation buffers
* mouse attraction/repulsion controls matching the Unity demo interaction model

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

The browser port currently covers the 2D simulation path only. The Unity 3D features such as raymarching, foam particles, screen-space fluid rendering, and marching cubes still need to be ported separately.
