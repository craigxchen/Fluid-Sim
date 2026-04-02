use std::borrow::Cow;
use std::f32::consts::PI;
use std::future::poll_fn;
use std::mem::size_of;
use std::sync::{Arc, Mutex};

use bytemuck::{Pod, Zeroable};

use crate::EPSILON;

const NUM_THREADS: u32 = 64;
const DIAGNOSTICS_WORDS: usize = 4;

const COMPUTE_SHADER_3D: &str = r#"
const EPSILON: f32 = 1e-6;
const PREDICTION_FACTOR: f32 = 1.0 / 120.0;
const NEIGHBOUR_OFFSETS: array<vec3<i32>, 27> = array<vec3<i32>, 27>(
    vec3<i32>(-1, -1, -1),
    vec3<i32>(0, -1, -1),
    vec3<i32>(1, -1, -1),
    vec3<i32>(-1, 0, -1),
    vec3<i32>(0, 0, -1),
    vec3<i32>(1, 0, -1),
    vec3<i32>(-1, 1, -1),
    vec3<i32>(0, 1, -1),
    vec3<i32>(1, 1, -1),
    vec3<i32>(-1, -1, 0),
    vec3<i32>(0, -1, 0),
    vec3<i32>(1, -1, 0),
    vec3<i32>(-1, 0, 0),
    vec3<i32>(0, 0, 0),
    vec3<i32>(1, 0, 0),
    vec3<i32>(-1, 1, 0),
    vec3<i32>(0, 1, 0),
    vec3<i32>(1, 1, 0),
    vec3<i32>(-1, -1, 1),
    vec3<i32>(0, -1, 1),
    vec3<i32>(1, -1, 1),
    vec3<i32>(-1, 0, 1),
    vec3<i32>(0, 0, 1),
    vec3<i32>(1, 0, 1),
    vec3<i32>(-1, 1, 1),
    vec3<i32>(0, 1, 1),
    vec3<i32>(1, 1, 1),
);

struct SimulationUniforms3D {
    counts0: vec4<u32>,
    counts1: vec4<u32>,
    step0: vec4<f32>,
    step1: vec4<f32>,
    bounds0: vec4<f32>,
    bounds1: vec4<f32>,
    kernels0: vec4<f32>,
    kernels1: vec4<f32>,
};

@group(0) @binding(0)
var<uniform> uniforms: SimulationUniforms3D;

@group(0) @binding(1)
var<storage, read_write> positions: array<vec4<f32>>;
@group(0) @binding(2)
var<storage, read_write> predicted_positions: array<vec4<f32>>;
@group(0) @binding(3)
var<storage, read_write> velocities: array<vec4<f32>>;
@group(0) @binding(4)
var<storage, read_write> velocities_scratch: array<vec4<f32>>;
@group(0) @binding(5)
var<storage, read_write> densities: array<vec2<f32>>;
@group(0) @binding(6)
var<storage, read_write> grid_state: array<atomic<u32>>;
@group(0) @binding(7)
var<storage, read_write> cell_particles: array<u32>;
@group(0) @binding(8)
var<storage, read_write> render_data: array<vec4<f32>>;

fn num_particles() -> u32 {
    return uniforms.counts0.x;
}

fn num_cells() -> u32 {
    return uniforms.counts0.y;
}

fn grid_width() -> u32 {
    return uniforms.counts0.z;
}

fn grid_height() -> u32 {
    return uniforms.counts0.w;
}

fn grid_depth() -> u32 {
    return uniforms.counts1.x;
}

fn max_particles_per_cell() -> u32 {
    return uniforms.counts1.y;
}

fn smoothing_radius() -> f32 {
    return uniforms.step0.w;
}

fn target_density() -> f32 {
    return uniforms.step1.x;
}

fn grid_origin() -> vec3<f32> {
    return uniforms.bounds1.xyz;
}

fn grid_cell_count_index(cell_index: u32) -> u32 {
    return 4u + cell_index;
}

fn get_cell(position: vec3<f32>) -> vec3<i32> {
    let radius = smoothing_radius();
    return vec3<i32>(floor((position - grid_origin()) / vec3<f32>(radius)));
}

fn flat_cell_index(cell: vec3<i32>) -> u32 {
    return (u32(cell.z) * grid_height() + u32(cell.y)) * grid_width() + u32(cell.x);
}

fn cell_is_valid(cell: vec3<i32>) -> bool {
    return cell.x >= 0
        && cell.y >= 0
        && cell.z >= 0
        && u32(cell.x) < grid_width()
        && u32(cell.y) < grid_height()
        && u32(cell.z) < grid_depth();
}

fn smoothing_kernel_poly6(distance: f32, radius: f32) -> f32 {
    if (distance < radius) {
        let value = radius * radius - distance * distance;
        return value * value * value * uniforms.kernels0.x;
    }
    return 0.0;
}

fn spiky_kernel_pow3(distance: f32, radius: f32) -> f32 {
    if (distance < radius) {
        let value = radius - distance;
        return value * value * value * uniforms.kernels0.y;
    }
    return 0.0;
}

fn spiky_kernel_pow2(distance: f32, radius: f32) -> f32 {
    if (distance < radius) {
        let value = radius - distance;
        return value * value * uniforms.kernels0.z;
    }
    return 0.0;
}

fn derivative_spiky_pow3(distance: f32, radius: f32) -> f32 {
    if (distance <= radius) {
        let value = radius - distance;
        return -value * value * uniforms.kernels0.w;
    }
    return 0.0;
}

fn derivative_spiky_pow2(distance: f32, radius: f32) -> f32 {
    if (distance <= radius) {
        let value = radius - distance;
        return -value * uniforms.kernels1.x;
    }
    return 0.0;
}

fn pressure_from_density(density: f32) -> f32 {
    return (density - uniforms.step1.x) * uniforms.step1.y;
}

fn near_pressure_from_density(near_density: f32) -> f32 {
    return uniforms.step1.z * near_density;
}

fn calculate_density(position: vec3<f32>) -> vec2<f32> {
    let origin = get_cell(position);
    let radius = smoothing_radius();
    let radius_sq = radius * radius;
    var density = 0.0;
    var near_density = 0.0;

    for (var offset_index = 0u; offset_index < 27u; offset_index += 1u) {
        let neighbour_cell = origin + NEIGHBOUR_OFFSETS[offset_index];
        if (!cell_is_valid(neighbour_cell)) {
            continue;
        }

        let cell_index = flat_cell_index(neighbour_cell);
        let count = min(
            atomicLoad(&grid_state[grid_cell_count_index(cell_index)]),
            max_particles_per_cell(),
        );
        for (var slot = 0u; slot < count; slot += 1u) {
            let neighbour_index = cell_particles[cell_index * max_particles_per_cell() + slot];
            let neighbour_position = predicted_positions[neighbour_index].xyz;
            let offset_to_neighbour = neighbour_position - position;
            let sqr_dst = dot(offset_to_neighbour, offset_to_neighbour);
            if (sqr_dst > radius_sq) {
                continue;
            }

            let distance = sqrt(sqr_dst);
            density += spiky_kernel_pow2(distance, radius);
            near_density += spiky_kernel_pow3(distance, radius);
        }
    }

    return vec2<f32>(density, near_density);
}

fn handle_collisions(index: u32) {
    var position = positions[index].xyz;
    var velocity = velocities[index].xyz;

    let half_bounds = uniforms.bounds0.xyz * 0.5;
    let edge_distance = half_bounds - abs(position);
    if (edge_distance.x <= 0.0) {
        position.x = half_bounds.x * sign(position.x);
        velocity.x *= -uniforms.step0.z;
    }
    if (edge_distance.y <= 0.0) {
        position.y = half_bounds.y * sign(position.y);
        velocity.y *= -uniforms.step0.z;
    }
    if (edge_distance.z <= 0.0) {
        position.z = half_bounds.z * sign(position.z);
        velocity.z *= -uniforms.step0.z;
    }

    positions[index] = vec4<f32>(position, 0.0);
    velocities[index] = vec4<f32>(velocity, 0.0);
}

@compute @workgroup_size(64)
fn clear_grid(@builtin(global_invocation_id) id: vec3<u32>) {
    let cell_index = id.x;
    if (cell_index < 4u) {
        atomicStore(&grid_state[cell_index], 0u);
    }
    if (cell_index >= num_cells()) {
        return;
    }

    atomicStore(&grid_state[grid_cell_count_index(cell_index)], 0u);
}

@compute @workgroup_size(64)
fn external_forces(@builtin(global_invocation_id) id: vec3<u32>) {
    let index = id.x;
    if (index >= num_particles()) {
        return;
    }

    let next_velocity = velocities[index].xyz + vec3<f32>(0.0, uniforms.step0.x, 0.0) * uniforms.step0.y;
    let next_position = positions[index].xyz + next_velocity * PREDICTION_FACTOR;
    velocities[index] = vec4<f32>(next_velocity, 0.0);
    predicted_positions[index] = vec4<f32>(next_position, 0.0);
}

@compute @workgroup_size(64)
fn build_grid(@builtin(global_invocation_id) id: vec3<u32>) {
    let index = id.x;
    if (index >= num_particles()) {
        return;
    }

    let cell = get_cell(predicted_positions[index].xyz);
    if (!cell_is_valid(cell)) {
        return;
    }

    let cell_index = flat_cell_index(cell);
    let slot = atomicAdd(&grid_state[grid_cell_count_index(cell_index)], 1u);
    atomicMax(&grid_state[0], slot + 1u);
    if (slot < max_particles_per_cell()) {
        cell_particles[cell_index * max_particles_per_cell() + slot] = index;
    } else {
        atomicAdd(&grid_state[1], 1u);
        if (slot == max_particles_per_cell()) {
            atomicAdd(&grid_state[2], 1u);
        }
    }
}

@compute @workgroup_size(64)
fn calculate_densities(@builtin(global_invocation_id) id: vec3<u32>) {
    let index = id.x;
    if (index >= num_particles()) {
        return;
    }

    densities[index] = calculate_density(predicted_positions[index].xyz);
}

@compute @workgroup_size(64)
fn calculate_pressure(@builtin(global_invocation_id) id: vec3<u32>) {
    let index = id.x;
    if (index >= num_particles()) {
        return;
    }

    let density = max(densities[index].x, EPSILON);
    let near_density = max(densities[index].y, EPSILON);
    let pressure = pressure_from_density(density);
    let near_pressure = near_pressure_from_density(near_density);
    let position = predicted_positions[index].xyz;
    let origin = get_cell(position);
    let radius = smoothing_radius();
    let radius_sq = radius * radius;
    var pressure_force = vec3<f32>(0.0);

    for (var offset_index = 0u; offset_index < 27u; offset_index += 1u) {
        let neighbour_cell = origin + NEIGHBOUR_OFFSETS[offset_index];
        if (!cell_is_valid(neighbour_cell)) {
            continue;
        }

        let cell_index = flat_cell_index(neighbour_cell);
        let count = min(
            atomicLoad(&grid_state[grid_cell_count_index(cell_index)]),
            max_particles_per_cell(),
        );
        for (var slot = 0u; slot < count; slot += 1u) {
            let neighbour_index = cell_particles[cell_index * max_particles_per_cell() + slot];
            if (neighbour_index == index) {
                continue;
            }

            let neighbour_position = predicted_positions[neighbour_index].xyz;
            let offset_to_neighbour = neighbour_position - position;
            let sqr_dst = dot(offset_to_neighbour, offset_to_neighbour);
            if (sqr_dst > radius_sq) {
                continue;
            }

            let distance = sqrt(sqr_dst);
            let direction = select(
                vec3<f32>(0.0, 1.0, 0.0),
                offset_to_neighbour / max(distance, EPSILON),
                distance > EPSILON,
            );
            let neighbour_density = max(densities[neighbour_index].x, EPSILON);
            let neighbour_near_density = max(densities[neighbour_index].y, EPSILON);
            let neighbour_pressure = pressure_from_density(neighbour_density);
            let neighbour_near_pressure = near_pressure_from_density(neighbour_near_density);
            let shared_pressure = (pressure + neighbour_pressure) * 0.5;
            let shared_near_pressure = (near_pressure + neighbour_near_pressure) * 0.5;

            pressure_force += direction
                * derivative_spiky_pow2(distance, radius)
                * shared_pressure
                / neighbour_density;
            pressure_force += direction
                * derivative_spiky_pow3(distance, radius)
                * shared_near_pressure
                / neighbour_near_density;
        }
    }

    let next_velocity = velocities[index].xyz + pressure_force / density * uniforms.step0.y;
    velocities_scratch[index] = vec4<f32>(next_velocity, 0.0);
}

@compute @workgroup_size(64)
fn calculate_viscosity(@builtin(global_invocation_id) id: vec3<u32>) {
    let index = id.x;
    if (index >= num_particles()) {
        return;
    }

    let position = predicted_positions[index].xyz;
    let origin = get_cell(position);
    let radius = smoothing_radius();
    let radius_sq = radius * radius;
    let velocity = velocities_scratch[index].xyz;
    var viscosity_force = vec3<f32>(0.0);

    for (var offset_index = 0u; offset_index < 27u; offset_index += 1u) {
        let neighbour_cell = origin + NEIGHBOUR_OFFSETS[offset_index];
        if (!cell_is_valid(neighbour_cell)) {
            continue;
        }

        let cell_index = flat_cell_index(neighbour_cell);
        let count = min(
            atomicLoad(&grid_state[grid_cell_count_index(cell_index)]),
            max_particles_per_cell(),
        );
        for (var slot = 0u; slot < count; slot += 1u) {
            let neighbour_index = cell_particles[cell_index * max_particles_per_cell() + slot];
            if (neighbour_index == index) {
                continue;
            }

            let neighbour_position = predicted_positions[neighbour_index].xyz;
            let offset_to_neighbour = neighbour_position - position;
            let sqr_dst = dot(offset_to_neighbour, offset_to_neighbour);
            if (sqr_dst > radius_sq) {
                continue;
            }

            let distance = sqrt(sqr_dst);
            viscosity_force +=
                (velocities_scratch[neighbour_index].xyz - velocity) * smoothing_kernel_poly6(distance, radius);
        }
    }

    velocities[index] = vec4<f32>(velocity + viscosity_force * uniforms.step1.w * uniforms.step0.y, 0.0);
}

@compute @workgroup_size(64)
fn update_positions(@builtin(global_invocation_id) id: vec3<u32>) {
    let index = id.x;
    if (index >= num_particles()) {
        return;
    }

    positions[index] = vec4<f32>(positions[index].xyz + velocities[index].xyz * uniforms.step0.y, 0.0);
    handle_collisions(index);
}

@compute @workgroup_size(64)
fn prepare_render_data(@builtin(global_invocation_id) id: vec3<u32>) {
    let index = id.x;
    if (index >= num_particles()) {
        return;
    }

    render_data[index] = vec4<f32>(
        positions[index].x,
        positions[index].y,
        positions[index].z,
        densities[index].x / max(target_density(), EPSILON),
    );
}
"#;

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct SimulationUniforms3D {
    counts0: [u32; 4],
    counts1: [u32; 4],
    step0: [f32; 4],
    step1: [f32; 4],
    bounds0: [f32; 4],
    bounds1: [f32; 4],
    kernels0: [f32; 4],
    kernels1: [f32; 4],
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable, Default)]
pub(crate) struct SimulationDiagnostics3D {
    pub(crate) peak_cell_occupancy: u32,
    pub(crate) dropped_particles: u32,
    pub(crate) overflowed_cells: u32,
    _reserved: u32,
}

pub(crate) struct GpuFluidSimulation3D {
    settings: SimulationSettings3D,
    initial_positions: Vec<[f32; 4]>,
    initial_velocities: Vec<[f32; 4]>,
    num_particles: usize,
    num_cells: u32,
    grid_width: u32,
    grid_height: u32,
    grid_depth: u32,
    max_particles_per_cell: u32,
    uniform_buffer: wgpu::Buffer,
    bind_group: wgpu::BindGroup,
    positions: wgpu::Buffer,
    predicted_positions: wgpu::Buffer,
    velocities: wgpu::Buffer,
    velocities_scratch: wgpu::Buffer,
    _densities: wgpu::Buffer,
    grid_state: wgpu::Buffer,
    _cell_particles: wgpu::Buffer,
    render_data: wgpu::Buffer,
    clear_grid_pipeline: wgpu::ComputePipeline,
    external_forces_pipeline: wgpu::ComputePipeline,
    build_grid_pipeline: wgpu::ComputePipeline,
    calculate_densities_pipeline: wgpu::ComputePipeline,
    calculate_pressure_pipeline: wgpu::ComputePipeline,
    calculate_viscosity_pipeline: wgpu::ComputePipeline,
    update_positions_pipeline: wgpu::ComputePipeline,
    prepare_render_data_pipeline: wgpu::ComputePipeline,
}

impl GpuFluidSimulation3D {
    pub(crate) fn new(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        settings: SimulationSettings3D,
    ) -> Self {
        let spawn = build_spawn_points_3d(&settings.spawn);
        let initial_positions: Vec<[f32; 4]> = spawn
            .positions
            .into_iter()
            .map(|value| [value.x, value.y, value.z, 0.0])
            .collect();
        let initial_velocities: Vec<[f32; 4]> = initial_positions
            .iter()
            .map(|_| {
                [
                    settings.initial_velocity.x,
                    settings.initial_velocity.y,
                    settings.initial_velocity.z,
                    0.0,
                ]
            })
            .collect();
        let num_particles = initial_positions.len();
        let grid_width = ((settings.bounds_size.x / settings.smoothing_radius).ceil() as u32) + 3;
        let grid_height = ((settings.bounds_size.y / settings.smoothing_radius).ceil() as u32) + 3;
        let grid_depth = ((settings.bounds_size.z / settings.smoothing_radius).ceil() as u32) + 3;
        let num_cells = grid_width
            .saturating_mul(grid_height)
            .saturating_mul(grid_depth);
        let max_particles_per_cell = estimate_max_particles_per_cell(
            &settings,
            &initial_positions,
            grid_width,
            grid_height,
            grid_depth,
        );

        let uniform_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("simulation 3d uniforms"),
            size: size_of::<SimulationUniforms3D>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let positions = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("simulation 3d positions"),
            size: buffer_size::<[f32; 4]>(num_particles),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let predicted_positions = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("simulation 3d predicted positions"),
            size: buffer_size::<[f32; 4]>(num_particles),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let velocities = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("simulation 3d velocities"),
            size: buffer_size::<[f32; 4]>(num_particles),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let velocities_scratch = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("simulation 3d velocities scratch"),
            size: buffer_size::<[f32; 4]>(num_particles),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let densities = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("simulation 3d densities"),
            size: buffer_size::<[f32; 2]>(num_particles),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let grid_state = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("simulation 3d grid state"),
            size: buffer_size::<u32>(num_cells as usize + DIAGNOSTICS_WORDS),
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let cell_particles = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("simulation 3d cell particles"),
            size: buffer_size::<u32>((num_cells * max_particles_per_cell) as usize),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let render_data = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("simulation 3d render data"),
            size: buffer_size::<[f32; 4]>(num_particles),
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::VERTEX
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("simulation 3d bind group layout"),
            entries: &[
                uniform_entry(0),
                storage_entry(1),
                storage_entry(2),
                storage_entry(3),
                storage_entry(4),
                storage_entry(5),
                storage_entry(6),
                storage_entry(7),
                storage_entry(8),
            ],
        });
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("simulation 3d bind group"),
            layout: &bind_group_layout,
            entries: &[
                buffer_entry(0, &uniform_buffer),
                buffer_entry(1, &positions),
                buffer_entry(2, &predicted_positions),
                buffer_entry(3, &velocities),
                buffer_entry(4, &velocities_scratch),
                buffer_entry(5, &densities),
                buffer_entry(6, &grid_state),
                buffer_entry(7, &cell_particles),
                buffer_entry(8, &render_data),
            ],
        });

        let shader_module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("simulation 3d compute shader"),
            source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(COMPUTE_SHADER_3D)),
        });
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("simulation 3d pipeline layout"),
            bind_group_layouts: &[Some(&bind_group_layout)],
            immediate_size: 0,
        });

        let mut simulation = Self {
            settings,
            initial_positions,
            initial_velocities,
            num_particles,
            num_cells,
            grid_width,
            grid_height,
            grid_depth,
            max_particles_per_cell,
            uniform_buffer,
            bind_group,
            positions,
            predicted_positions,
            velocities,
            velocities_scratch,
            _densities: densities,
            grid_state,
            _cell_particles: cell_particles,
            render_data,
            clear_grid_pipeline: compute_pipeline(
                device,
                &pipeline_layout,
                &shader_module,
                "clear_grid",
            ),
            external_forces_pipeline: compute_pipeline(
                device,
                &pipeline_layout,
                &shader_module,
                "external_forces",
            ),
            build_grid_pipeline: compute_pipeline(
                device,
                &pipeline_layout,
                &shader_module,
                "build_grid",
            ),
            calculate_densities_pipeline: compute_pipeline(
                device,
                &pipeline_layout,
                &shader_module,
                "calculate_densities",
            ),
            calculate_pressure_pipeline: compute_pipeline(
                device,
                &pipeline_layout,
                &shader_module,
                "calculate_pressure",
            ),
            calculate_viscosity_pipeline: compute_pipeline(
                device,
                &pipeline_layout,
                &shader_module,
                "calculate_viscosity",
            ),
            update_positions_pipeline: compute_pipeline(
                device,
                &pipeline_layout,
                &shader_module,
                "update_positions",
            ),
            prepare_render_data_pipeline: compute_pipeline(
                device,
                &pipeline_layout,
                &shader_module,
                "prepare_render_data",
            ),
        };

        simulation.write_initial_state(queue);
        simulation.synchronize(queue, device);
        simulation
    }

    pub(crate) fn settings(&self) -> &SimulationSettings3D {
        &self.settings
    }

    pub(crate) fn reset(&mut self, queue: &wgpu::Queue, device: &wgpu::Device) {
        self.write_initial_state(queue);
        self.synchronize(queue, device);
    }

    pub(crate) fn particle_count(&self) -> usize {
        self.num_particles
    }

    pub(crate) fn max_particles_per_cell(&self) -> u32 {
        self.max_particles_per_cell
    }

    pub(crate) fn render_buffer(&self) -> &wgpu::Buffer {
        &self.render_data
    }

    pub(crate) fn diagnostics_buffer(&self) -> wgpu::Buffer {
        self.grid_state.clone()
    }

    pub(crate) async fn read_diagnostics_buffer(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        diagnostics: &wgpu::Buffer,
    ) -> Result<SimulationDiagnostics3D, String> {
        readback_value(device, queue, diagnostics).await
    }

    pub(crate) fn step_frame(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        frame_time: f32,
    ) {
        if self.num_particles == 0 {
            return;
        }

        let max_delta = if self.settings.max_timestep_fps > 0.0 {
            1.0 / self.settings.max_timestep_fps
        } else {
            f32::INFINITY
        };
        let frame_delta = (frame_time * self.settings.time_scale).min(max_delta);
        let iterations = self.settings.iterations_per_frame.max(1) as f32;
        let step_delta = frame_delta / iterations;

        self.write_uniforms(queue, step_delta);
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("simulation 3d step encoder"),
        });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("simulation 3d step pass"),
                timestamp_writes: None,
            });
            pass.set_bind_group(0, &self.bind_group, &[]);

            for _ in 0..self.settings.iterations_per_frame.max(1) {
                self.encode_clear_and_build_grid(&mut pass);
                self.encode_pipeline(
                    &mut pass,
                    &self.calculate_densities_pipeline,
                    self.particle_workgroups(),
                );
                self.encode_pipeline(
                    &mut pass,
                    &self.calculate_pressure_pipeline,
                    self.particle_workgroups(),
                );
                self.encode_pipeline(
                    &mut pass,
                    &self.calculate_viscosity_pipeline,
                    self.particle_workgroups(),
                );
                self.encode_pipeline(
                    &mut pass,
                    &self.update_positions_pipeline,
                    self.particle_workgroups(),
                );
            }

            self.encode_pipeline(
                &mut pass,
                &self.prepare_render_data_pipeline,
                self.particle_workgroups(),
            );
        }
        queue.submit(Some(encoder.finish()));
    }

    fn synchronize(&mut self, queue: &wgpu::Queue, device: &wgpu::Device) {
        self.write_uniforms(queue, 0.0);
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("simulation 3d sync encoder"),
        });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("simulation 3d sync pass"),
                timestamp_writes: None,
            });
            pass.set_bind_group(0, &self.bind_group, &[]);
            self.encode_clear_and_build_grid(&mut pass);
            self.encode_pipeline(
                &mut pass,
                &self.calculate_densities_pipeline,
                self.particle_workgroups(),
            );
            self.encode_pipeline(
                &mut pass,
                &self.prepare_render_data_pipeline,
                self.particle_workgroups(),
            );
        }
        queue.submit(Some(encoder.finish()));
    }

    fn encode_clear_and_build_grid(&self, pass: &mut wgpu::ComputePass<'_>) {
        self.encode_pipeline(pass, &self.clear_grid_pipeline, self.cell_workgroups());
        self.encode_pipeline(
            pass,
            &self.external_forces_pipeline,
            self.particle_workgroups(),
        );
        self.encode_pipeline(pass, &self.build_grid_pipeline, self.particle_workgroups());
    }

    fn encode_pipeline(
        &self,
        pass: &mut wgpu::ComputePass<'_>,
        pipeline: &wgpu::ComputePipeline,
        workgroups: u32,
    ) {
        if workgroups == 0 {
            return;
        }
        pass.set_pipeline(pipeline);
        pass.dispatch_workgroups(workgroups, 1, 1);
    }

    fn write_initial_state(&self, queue: &wgpu::Queue) {
        if self.num_particles == 0 {
            return;
        }

        queue.write_buffer(
            &self.positions,
            0,
            bytemuck::cast_slice(&self.initial_positions),
        );
        queue.write_buffer(
            &self.predicted_positions,
            0,
            bytemuck::cast_slice(&self.initial_positions),
        );
        queue.write_buffer(
            &self.velocities,
            0,
            bytemuck::cast_slice(&self.initial_velocities),
        );
        queue.write_buffer(
            &self.velocities_scratch,
            0,
            bytemuck::cast_slice(&self.initial_velocities),
        );
    }

    fn write_uniforms(&self, queue: &wgpu::Queue, delta_time: f32) {
        let radius = self.settings.smoothing_radius.max(EPSILON);
        let uniform = SimulationUniforms3D {
            counts0: [
                self.num_particles as u32,
                self.num_cells,
                self.grid_width,
                self.grid_height,
            ],
            counts1: [self.grid_depth, self.max_particles_per_cell, 0, 0],
            step0: [
                self.settings.gravity,
                delta_time,
                self.settings.collision_damping,
                radius,
            ],
            step1: [
                self.settings.target_density,
                self.settings.pressure_multiplier,
                self.settings.near_pressure_multiplier,
                self.settings.viscosity_strength,
            ],
            bounds0: [
                self.settings.bounds_size.x,
                self.settings.bounds_size.y,
                self.settings.bounds_size.z,
                0.0,
            ],
            bounds1: [
                -self.settings.bounds_size.x * 0.5 - radius,
                -self.settings.bounds_size.y * 0.5 - radius,
                -self.settings.bounds_size.z * 0.5 - radius,
                0.0,
            ],
            kernels0: [
                315.0 / (64.0 * PI * radius.powi(9)),
                15.0 / (PI * radius.powi(6)),
                15.0 / (2.0 * PI * radius.powi(5)),
                45.0 / (PI * radius.powi(6)),
            ],
            kernels1: [15.0 / (PI * radius.powi(5)), 0.0, 0.0, 0.0],
        };

        queue.write_buffer(&self.uniform_buffer, 0, bytemuck::bytes_of(&uniform));
    }

    fn particle_workgroups(&self) -> u32 {
        workgroups_for(self.num_particles as u32)
    }

    fn cell_workgroups(&self) -> u32 {
        workgroups_for(self.num_cells.max(DIAGNOSTICS_WORDS as u32))
    }
}

fn compute_pipeline(
    device: &wgpu::Device,
    layout: &wgpu::PipelineLayout,
    shader_module: &wgpu::ShaderModule,
    entry_point: &'static str,
) -> wgpu::ComputePipeline {
    device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some(entry_point),
        layout: Some(layout),
        module: shader_module,
        entry_point: Some(entry_point),
        compilation_options: wgpu::PipelineCompilationOptions::default(),
        cache: None,
    })
}

fn storage_entry(binding: u32) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Storage { read_only: false },
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    }
}

fn uniform_entry(binding: u32) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Uniform,
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    }
}

fn buffer_entry<'a>(binding: u32, buffer: &'a wgpu::Buffer) -> wgpu::BindGroupEntry<'a> {
    wgpu::BindGroupEntry {
        binding,
        resource: buffer.as_entire_binding(),
    }
}

fn workgroups_for(count: u32) -> u32 {
    count.div_ceil(NUM_THREADS)
}

fn buffer_size<T>(count: usize) -> u64 {
    (count.max(1) * size_of::<T>()) as u64
}

fn estimate_max_particles_per_cell(
    settings: &SimulationSettings3D,
    initial_positions: &[[f32; 4]],
    grid_width: u32,
    grid_height: u32,
    grid_depth: u32,
) -> u32 {
    let num_cells = grid_width
        .saturating_mul(grid_height)
        .saturating_mul(grid_depth);
    if initial_positions.is_empty() || num_cells == 0 {
        return 1;
    }

    let radius = settings.smoothing_radius.max(EPSILON);
    let grid_origin_x = -settings.bounds_size.x * 0.5 - radius;
    let grid_origin_y = -settings.bounds_size.y * 0.5 - radius;
    let grid_origin_z = -settings.bounds_size.z * 0.5 - radius;
    let mut occupancy = vec![0u32; num_cells as usize];
    let mut initial_peak = 0u32;

    for position in initial_positions {
        let cell_x = ((position[0] - grid_origin_x) / radius).floor() as i32;
        let cell_y = ((position[1] - grid_origin_y) / radius).floor() as i32;
        let cell_z = ((position[2] - grid_origin_z) / radius).floor() as i32;
        if cell_x < 0
            || cell_y < 0
            || cell_z < 0
            || cell_x as u32 >= grid_width
            || cell_y as u32 >= grid_height
            || cell_z as u32 >= grid_depth
        {
            continue;
        }

        let flat_index = ((cell_z as usize * grid_height as usize) + cell_y as usize)
            * grid_width as usize
            + cell_x as usize;
        occupancy[flat_index] += 1;
        initial_peak = initial_peak.max(occupancy[flat_index]);
    }

    let average_occupancy = (initial_positions.len() as u32).div_ceil(num_cells);
    let density_estimate =
        (settings.spawn.spawn_density as f32 * settings.smoothing_radius.powi(3)).ceil() as u32;
    let baseline = initial_peak
        .max(average_occupancy)
        .max(density_estimate)
        .max(1);

    baseline
        .saturating_mul(16)
        .next_power_of_two()
        .min(initial_positions.len() as u32)
        .max(baseline)
}

async fn readback_value<T: Pod>(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    source: &wgpu::Buffer,
) -> Result<T, String> {
    let size = size_of::<T>() as u64;
    let download = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("simulation 3d readback"),
        size,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });
    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("simulation 3d readback encoder"),
    });
    encoder.copy_buffer_to_buffer(source, 0, &download, 0, size);
    queue.submit(Some(encoder.finish()));

    let state = Arc::new(Mutex::new(MapState::default()));
    let state_for_callback = Arc::clone(&state);
    download
        .slice(..)
        .map_async(wgpu::MapMode::Read, move |result| {
            let mut state = state_for_callback
                .lock()
                .expect("map readback state lock poisoned");
            state.result = Some(result.map_err(|error| error.to_string()));
            if let Some(waker) = state.waker.take() {
                waker.wake();
            }
        });

    poll_fn(|cx| {
        let mut state = state.lock().expect("map readback state lock poisoned");
        if let Some(result) = state.result.take() {
            return std::task::Poll::Ready(result);
        }
        state.waker = Some(cx.waker().clone());
        std::task::Poll::Pending
    })
    .await?;

    let mapped = download.slice(..).get_mapped_range();
    let value = *bytemuck::from_bytes::<T>(&mapped);
    drop(mapped);
    download.unmap();
    Ok(value)
}

#[derive(Default)]
struct MapState {
    result: Option<Result<(), String>>,
    waker: Option<std::task::Waker>,
}

#[derive(Clone)]
pub(crate) struct SimulationSettings3D {
    pub(crate) preset_name: &'static str,
    pub(crate) time_scale: f32,
    pub(crate) max_timestep_fps: f32,
    pub(crate) iterations_per_frame: usize,
    pub(crate) gravity: f32,
    pub(crate) collision_damping: f32,
    pub(crate) smoothing_radius: f32,
    pub(crate) target_density: f32,
    pub(crate) pressure_multiplier: f32,
    pub(crate) near_pressure_multiplier: f32,
    pub(crate) viscosity_strength: f32,
    pub(crate) bounds_size: Vec3,
    pub(crate) render_radius: f32,
    pub(crate) initial_velocity: Vec3,
    spawn: SpawnSettings3D,
}

impl SimulationSettings3D {
    pub(crate) fn preset(index: usize) -> Option<Self> {
        match index {
            0 => Some(Self::particles()),
            1 => Some(Self::raymarch()),
            2 => Some(Self::marching_cubes()),
            _ => None,
        }
    }

    pub(crate) fn preset_count() -> usize {
        3
    }

    pub(crate) fn preset_name(index: usize) -> &'static str {
        match index {
            0 => "Unity 3D Particles",
            1 => "Unity 3D Raymarch",
            2 => "Unity 3D Marching Cubes",
            _ => "Unknown Preset",
        }
    }

    pub(crate) fn particles() -> Self {
        Self {
            preset_name: "Unity 3D Particles",
            time_scale: 1.0,
            max_timestep_fps: 60.0,
            iterations_per_frame: 3,
            gravity: -10.0,
            collision_damping: 0.95,
            smoothing_radius: 0.24,
            target_density: 430.0,
            pressure_multiplier: 230.0,
            near_pressure_multiplier: 2.0,
            viscosity_strength: 0.004,
            bounds_size: Vec3::new(24.0, 10.0, 15.0),
            render_radius: 0.13,
            initial_velocity: Vec3::ZERO,
            spawn: SpawnSettings3D {
                spawn_density: 80,
                jitter_strength: 0.03,
                regions: vec![
                    SpawnRegion3D {
                        centre: Vec3::new(-8.3, -1.3, 3.65),
                        size: 7.0,
                    },
                    SpawnRegion3D {
                        centre: Vec3::new(-8.3, -1.3, -3.65),
                        size: 7.0,
                    },
                ],
            },
        }
    }

    pub(crate) fn raymarch() -> Self {
        Self {
            preset_name: "Unity 3D Raymarch",
            time_scale: 1.0,
            max_timestep_fps: 60.0,
            iterations_per_frame: 3,
            gravity: -10.0,
            collision_damping: 0.95,
            smoothing_radius: 0.24,
            target_density: 430.0,
            pressure_multiplier: 230.0,
            near_pressure_multiplier: 2.0,
            viscosity_strength: 0.004,
            bounds_size: Vec3::new(15.0, 10.0, 8.0),
            render_radius: 0.12,
            initial_velocity: Vec3::ZERO,
            spawn: SpawnSettings3D {
                spawn_density: 72,
                jitter_strength: 0.03,
                regions: vec![SpawnRegion3D {
                    centre: Vec3::new(-4.0, -1.42, 0.0),
                    size: 7.0,
                }],
            },
        }
    }

    pub(crate) fn marching_cubes() -> Self {
        Self {
            preset_name: "Unity 3D Marching Cubes",
            time_scale: 1.0,
            max_timestep_fps: 60.0,
            iterations_per_frame: 3,
            gravity: -10.0,
            collision_damping: 0.95,
            smoothing_radius: 0.24,
            target_density: 430.0,
            pressure_multiplier: 230.0,
            near_pressure_multiplier: 2.0,
            viscosity_strength: 0.004,
            bounds_size: Vec3::new(16.0, 12.0, 8.0),
            render_radius: 0.12,
            initial_velocity: Vec3::ZERO,
            spawn: SpawnSettings3D {
                spawn_density: 72,
                jitter_strength: 0.03,
                regions: vec![SpawnRegion3D {
                    centre: Vec3::new(3.92, -1.94, 0.0),
                    size: 7.0,
                }],
            },
        }
    }
}

#[derive(Clone)]
struct SpawnSettings3D {
    spawn_density: usize,
    jitter_strength: f32,
    regions: Vec<SpawnRegion3D>,
}

#[derive(Clone, Copy)]
struct SpawnRegion3D {
    centre: Vec3,
    size: f32,
}

struct SpawnData3D {
    positions: Vec<Vec3>,
}

fn build_spawn_points_3d(settings: &SpawnSettings3D) -> SpawnData3D {
    let mut rng = SimpleRng::new(7);
    let mut positions = Vec::new();

    for region in &settings.regions {
        let particles_per_axis =
            calculate_spawn_count_per_axis_3d(region.size, settings.spawn_density);
        for z in 0..particles_per_axis {
            for y in 0..particles_per_axis {
                for x in 0..particles_per_axis {
                    let tx = coordinate_t(x, particles_per_axis);
                    let ty = coordinate_t(y, particles_per_axis);
                    let tz = coordinate_t(z, particles_per_axis);
                    let position = Vec3::new(
                        (tx - 0.5) * region.size + region.centre.x,
                        (ty - 0.5) * region.size + region.centre.y,
                        (tz - 0.5) * region.size + region.centre.z,
                    );
                    let jitter = random_in_unit_sphere(&mut rng) * settings.jitter_strength;
                    positions.push(position + jitter);
                }
            }
        }
    }

    SpawnData3D { positions }
}

fn calculate_spawn_count_per_axis_3d(size: f32, spawn_density: usize) -> usize {
    let target_particle_count = (size * size * size * spawn_density as f32).max(1.0);
    target_particle_count.cbrt().floor().max(1.0) as usize
}

fn coordinate_t(index: usize, count: usize) -> f32 {
    if count > 1 {
        index as f32 / (count - 1) as f32
    } else {
        0.5
    }
}

fn random_in_unit_sphere(rng: &mut SimpleRng) -> Vec3 {
    loop {
        let candidate = Vec3::new(
            rng.next_f32() * 2.0 - 1.0,
            rng.next_f32() * 2.0 - 1.0,
            rng.next_f32() * 2.0 - 1.0,
        );
        if candidate.length_squared() <= 1.0 {
            return candidate;
        }
    }
}

#[derive(Clone, Copy, Default)]
pub(crate) struct Vec3 {
    pub(crate) x: f32,
    pub(crate) y: f32,
    pub(crate) z: f32,
}

impl Vec3 {
    pub(crate) const ZERO: Self = Self {
        x: 0.0,
        y: 0.0,
        z: 0.0,
    };

    pub(crate) const fn new(x: f32, y: f32, z: f32) -> Self {
        Self { x, y, z }
    }

    fn length_squared(self) -> f32 {
        self.x * self.x + self.y * self.y + self.z * self.z
    }
}

impl std::ops::Add for Vec3 {
    type Output = Self;

    fn add(self, rhs: Self) -> Self::Output {
        Self::new(self.x + rhs.x, self.y + rhs.y, self.z + rhs.z)
    }
}

impl std::ops::Mul<f32> for Vec3 {
    type Output = Self;

    fn mul(self, rhs: f32) -> Self::Output {
        Self::new(self.x * rhs, self.y * rhs, self.z * rhs)
    }
}

struct SimpleRng {
    state: u64,
}

impl SimpleRng {
    const fn new(seed: u64) -> Self {
        Self { state: seed }
    }

    fn next_f32(&mut self) -> f32 {
        self.state = self
            .state
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1);
        let bits = (self.state >> 40) as u32;
        bits as f32 / ((1 << 24) - 1) as f32
    }
}
