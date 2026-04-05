use std::borrow::Cow;
#[cfg(target_arch = "wasm32")]
use std::future::poll_fn;
use std::f32::consts::PI;
use std::mem::size_of;
#[cfg(target_arch = "wasm32")]
use std::sync::{Arc, Mutex};
#[cfg(test)]
use std::sync::mpsc;

use bytemuck::{Pod, Zeroable};

use crate::{
    EPSILON, InteractionState, SimulationSettings, Vec2, build_spawn_points,
};
#[cfg(test)]
use crate::FluidSimulation;

const NUM_THREADS: u32 = 64;
const DIAGNOSTICS_WORDS: usize = 4;

const COMPUTE_SHADER: &str = r#"
const EPSILON: f32 = 1e-6;
const PREDICTION_FACTOR: f32 = 1.0 / 120.0;
const NEIGHBOUR_OFFSETS: array<vec2<i32>, 9> = array<vec2<i32>, 9>(
    vec2<i32>(-1, 1),
    vec2<i32>(0, 1),
    vec2<i32>(1, 1),
    vec2<i32>(-1, 0),
    vec2<i32>(0, 0),
    vec2<i32>(1, 0),
    vec2<i32>(-1, -1),
    vec2<i32>(0, -1),
    vec2<i32>(1, -1),
);

struct SimulationUniforms {
    counts0: vec4<u32>,
    counts1: vec4<u32>,
    step0: vec4<f32>,
    step1: vec4<f32>,
    bounds: vec4<f32>,
    interaction: vec4<f32>,
    obstacle: vec4<f32>,
    kernels0: vec4<f32>,
    kernels1: vec4<f32>,
    fluid_type0: vec4<f32>,
    fluid_type1: vec4<f32>,
    multi_fluid: vec4<f32>,
};

@group(0) @binding(0)
var<uniform> uniforms: SimulationUniforms;

@group(0) @binding(1)
var<storage, read_write> positions: array<vec2<f32>>;
@group(0) @binding(2)
var<storage, read_write> predicted_positions: array<vec2<f32>>;
@group(0) @binding(3)
var<storage, read_write> velocities: array<vec2<f32>>;
@group(0) @binding(4)
var<storage, read_write> velocities_scratch: array<vec2<f32>>;
@group(0) @binding(5)
var<storage, read_write> densities: array<vec4<f32>>;
@group(0) @binding(6)
var<storage, read_write> grid_state: array<atomic<u32>>;
@group(0) @binding(7)
var<storage, read_write> cell_particles: array<u32>;
@group(0) @binding(8)
var<storage, read_write> render_data: array<vec4<f32>>;

fn smoothing_radius() -> f32 {
    return uniforms.step0.w;
}

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

fn max_particles_per_cell() -> u32 {
    return uniforms.counts1.x;
}

fn grid_cell_count_index(cell_index: u32) -> u32 {
    return 4u + cell_index;
}

fn target_density() -> f32 {
    return uniforms.step1.x;
}

fn get_cell(position: vec2<f32>) -> vec2<i32> {
    let grid_origin = uniforms.bounds.zw;
    let radius = smoothing_radius();
    return vec2<i32>(floor((position - grid_origin) / vec2<f32>(radius, radius)));
}

fn flat_cell_index(cell: vec2<i32>) -> u32 {
    return u32(cell.y) * grid_width() + u32(cell.x);
}

fn cell_is_valid(cell: vec2<i32>) -> bool {
    return cell.x >= 0
        && cell.y >= 0
        && u32(cell.x) < grid_width()
        && u32(cell.y) < grid_height();
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

fn get_fluid_type(index: u32) -> u32 {
    return u32(densities[index].z);
}

fn fluid_target_density(ft: u32) -> f32 {
    return select(uniforms.fluid_type1.x, uniforms.fluid_type0.x, ft == 0u);
}

fn fluid_pressure_mult(ft: u32) -> f32 {
    return select(uniforms.fluid_type1.y, uniforms.fluid_type0.y, ft == 0u);
}

fn fluid_near_pressure_mult(ft: u32) -> f32 {
    return select(uniforms.fluid_type1.z, uniforms.fluid_type0.z, ft == 0u);
}

fn fluid_viscosity(ft: u32) -> f32 {
    return select(uniforms.fluid_type1.w, uniforms.fluid_type0.w, ft == 0u);
}

fn fluid_mass(ft: u32) -> f32 {
    return fluid_target_density(ft) / max(uniforms.fluid_type0.x, EPSILON);
}

fn immiscibility_strength() -> f32 {
    return uniforms.multi_fluid.x;
}

fn surface_tension_gamma(ft_a: u32, ft_b: u32) -> f32 {
    if (ft_a == ft_b) {
        return select(uniforms.multi_fluid.z, uniforms.multi_fluid.y, ft_a == 0u);
    }
    return uniforms.multi_fluid.w;
}

fn akinci_cohesion_kernel_2d(distance: f32, radius: f32) -> f32 {
    if (distance >= radius || distance <= 0.0) {
        return 0.0;
    }
    let coeff = 32.0 / (3.141592653589793 * pow(radius, 9.0));
    let half_r = radius * 0.5;
    if (distance > half_r) {
        let d = radius - distance;
        return coeff * d * d * d * distance * distance * distance;
    }
    let d = radius - distance;
    return coeff * (2.0 * d * d * d * distance * distance * distance - pow(radius, 6.0) / 64.0);
}

fn external_force(position: vec2<f32>, velocity: vec2<f32>) -> vec2<f32> {
    let gravity = vec2<f32>(0.0, uniforms.step0.x);
    let interaction_strength = uniforms.interaction.z;
    if (abs(interaction_strength) <= EPSILON) {
        return gravity;
    }

    let offset = uniforms.interaction.xy - position;
    let sqr_dst = dot(offset, offset);
    let radius = uniforms.interaction.w;
    let radius_sq = radius * radius;
    if (sqr_dst >= radius_sq) {
        return gravity;
    }

    let distance = sqrt(sqr_dst);
    let edge_t = distance / max(radius, EPSILON);
    let centre_t = 1.0 - edge_t;
    let direction = select(vec2<f32>(0.0, 0.0), offset / max(distance, EPSILON), distance > EPSILON);
    let gravity_weight = 1.0 - centre_t * clamp(interaction_strength / 10.0, 0.0, 1.0);
    return gravity * gravity_weight + direction * centre_t * interaction_strength - velocity * centre_t;
}

fn calculate_density(position: vec2<f32>) -> vec2<f32> {
    let origin = get_cell(position);
    let radius = smoothing_radius();
    let radius_sq = radius * radius;
    var density = 0.0;
    var near_density = 0.0;

    for (var offset_index = 0u; offset_index < 9u; offset_index += 1u) {
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
            let neighbour_position = predicted_positions[neighbour_index];
            let offset_to_neighbour = neighbour_position - position;
            let sqr_dst = dot(offset_to_neighbour, offset_to_neighbour);
            if (sqr_dst > radius_sq) {
                continue;
            }

            let distance = sqrt(sqr_dst);
            let mass = fluid_mass(get_fluid_type(neighbour_index));
            density += mass * spiky_kernel_pow2(distance, radius);
            near_density += mass * spiky_kernel_pow3(distance, radius);
        }
    }

    return vec2<f32>(density, near_density);
}

const FLUID_TYPE_PACK: f32 = 1024.0;

fn handle_collisions(index: u32) {
    var position = positions[index];
    var velocity = velocities[index];

    let half_bounds = uniforms.bounds.xy * 0.5;
    let edge_distance = half_bounds - abs(position);
    if (edge_distance.x <= 0.0) {
        position.x = half_bounds.x * sign(position.x);
        velocity.x *= -uniforms.step0.z;
    }
    if (edge_distance.y <= 0.0) {
        position.y = half_bounds.y * sign(position.y);
        velocity.y *= -uniforms.step0.z;
    }

    let obstacle_half = uniforms.obstacle.xy * 0.5;
    let obstacle_delta = position - uniforms.obstacle.zw;
    let obstacle_edge_distance = obstacle_half - abs(obstacle_delta);
    if (obstacle_edge_distance.x >= 0.0 && obstacle_edge_distance.y >= 0.0) {
        if (obstacle_edge_distance.x < obstacle_edge_distance.y) {
            position.x = obstacle_half.x * sign(obstacle_delta.x) + uniforms.obstacle.z;
            velocity.x *= -uniforms.step0.z;
        } else {
            position.y = obstacle_half.y * sign(obstacle_delta.y) + uniforms.obstacle.w;
            velocity.y *= -uniforms.step0.z;
        }
    }

    positions[index] = position;
    velocities[index] = velocity;
}

@compute @workgroup_size(64)
fn clear_grid(@builtin(global_invocation_id) id: vec3<u32>) {
    let cell_index = id.x;
    if (cell_index == 0u) {
        atomicStore(&grid_state[0], 0u);
        atomicStore(&grid_state[1], 0u);
        atomicStore(&grid_state[2], 0u);
        atomicStore(&grid_state[3], 0u);
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

    let next_velocity = velocities[index] + external_force(positions[index], velocities[index]) * uniforms.step0.y;
    velocities[index] = next_velocity;
    predicted_positions[index] = positions[index] + next_velocity * PREDICTION_FACTOR;
}

@compute @workgroup_size(64)
fn build_grid(@builtin(global_invocation_id) id: vec3<u32>) {
    let index = id.x;
    if (index >= num_particles()) {
        return;
    }

    let cell = get_cell(predicted_positions[index]);
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

    let ft = densities[index].z;
    let d = calculate_density(predicted_positions[index]);
    densities[index] = vec4<f32>(d.x, d.y, ft, 0.0);
}

@compute @workgroup_size(64)
fn calculate_pressure(@builtin(global_invocation_id) id: vec3<u32>) {
    let index = id.x;
    if (index >= num_particles()) {
        return;
    }

    let my_type = get_fluid_type(index);
    let my_density = max(densities[index].x, EPSILON);
    let my_near_density = densities[index].y;
    let my_target = fluid_target_density(my_type);
    let my_mass = fluid_mass(my_type);
    let pressure = (my_density - my_target) * fluid_pressure_mult(my_type);
    let near_pressure = fluid_near_pressure_mult(my_type) * my_near_density;

    let position = predicted_positions[index];
    let origin = get_cell(position);
    let radius = smoothing_radius();
    let radius_sq = radius * radius;
    var pressure_force = vec2<f32>(0.0, 0.0);

    for (var offset_index = 0u; offset_index < 9u; offset_index += 1u) {
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

            let neighbour_position = predicted_positions[neighbour_index];
            let offset_to_neighbour = neighbour_position - position;
            let sqr_dst = dot(offset_to_neighbour, offset_to_neighbour);
            if (sqr_dst > radius_sq) {
                continue;
            }

            let distance = sqrt(sqr_dst);
            let direction = select(vec2<f32>(0.0, 1.0), offset_to_neighbour / max(distance, EPSILON), distance > EPSILON);
            let neighbour_type = get_fluid_type(neighbour_index);
            let mass = fluid_mass(neighbour_type);
            let neighbour_density = max(densities[neighbour_index].x, EPSILON);
            let neighbour_near_density = densities[neighbour_index].y;
            let neighbour_pressure = (neighbour_density - fluid_target_density(neighbour_type)) * fluid_pressure_mult(neighbour_type);
            let neighbour_near_pressure = fluid_near_pressure_mult(neighbour_type) * neighbour_near_density;
            let density_ratio = fluid_target_density(neighbour_type) / max(my_target, EPSILON);

            pressure_force += 0.5 * mass
                * (pressure / (my_density * my_density) + density_ratio * neighbour_pressure / (neighbour_density * neighbour_density))
                * derivative_spiky_pow2(distance, radius) * direction;

            pressure_force += 0.5 * mass
                * (near_pressure / (my_density * my_density) + density_ratio * neighbour_near_pressure / (neighbour_density * neighbour_density))
                * derivative_spiky_pow3(distance, radius) * direction;
        }
    }

    velocities_scratch[index] = velocities[index] + pressure_force * uniforms.step0.y;
}

@compute @workgroup_size(64)
fn calculate_viscosity(@builtin(global_invocation_id) id: vec3<u32>) {
    let index = id.x;
    if (index >= num_particles()) {
        return;
    }

    let my_type = get_fluid_type(index);
    let my_viscosity = fluid_viscosity(my_type);
    let position = predicted_positions[index];
    let origin = get_cell(position);
    let radius = smoothing_radius();
    let radius_sq = radius * radius;
    let velocity = velocities_scratch[index];
    var viscosity_force = vec2<f32>(0.0, 0.0);

    for (var offset_index = 0u; offset_index < 9u; offset_index += 1u) {
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

            let neighbour_position = predicted_positions[neighbour_index];
            let offset_to_neighbour = neighbour_position - position;
            let sqr_dst = dot(offset_to_neighbour, offset_to_neighbour);
            if (sqr_dst > radius_sq) {
                continue;
            }

            let distance = sqrt(sqr_dst);
            let neighbour_type = get_fluid_type(neighbour_index);
            let viscosity = select(my_viscosity * 0.1, my_viscosity, my_type == neighbour_type);
            let mass = fluid_mass(neighbour_type);
            viscosity_force +=
                (velocities_scratch[neighbour_index] - velocity)
                * smoothing_kernel_poly6(distance, radius) * viscosity * mass;
        }
    }

    velocities[index] = velocity + viscosity_force * uniforms.step0.y;
}

@compute @workgroup_size(64)
fn calculate_surface_tension(@builtin(global_invocation_id) id: vec3<u32>) {
    let index = id.x;
    if (index >= num_particles()) {
        return;
    }

    let my_type = get_fluid_type(index);
    let my_mass = fluid_mass(my_type);
    let my_density = max(densities[index].x, EPSILON);
    let position = predicted_positions[index];
    let origin = get_cell(position);
    let radius = smoothing_radius();
    let radius_sq = radius * radius;
    var cohesion_force = vec2<f32>(0.0, 0.0);
    var normal = vec2<f32>(0.0, 0.0);

    for (var offset_index = 0u; offset_index < 9u; offset_index += 1u) {
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

            let neighbour_position = predicted_positions[neighbour_index];
            let offset_to_neighbour = neighbour_position - position;
            let sqr_dst = dot(offset_to_neighbour, offset_to_neighbour);
            if (sqr_dst > radius_sq) {
                continue;
            }

            let distance = sqrt(sqr_dst);
            let neighbour_type = get_fluid_type(neighbour_index);
            let neighbour_mass = fluid_mass(neighbour_type);
            let neighbour_density = max(densities[neighbour_index].x, EPSILON);
            let gamma = surface_tension_gamma(my_type, neighbour_type);

            if (gamma <= 0.0) {
                continue;
            }

            let K = 2.0 * target_density() / max(my_density + neighbour_density, EPSILON);
            let direction = select(
                vec2<f32>(0.0, 1.0),
                offset_to_neighbour / max(distance, EPSILON),
                distance > EPSILON,
            );

            cohesion_force -= gamma * my_mass * neighbour_mass
                * akinci_cohesion_kernel_2d(distance, radius) * direction * K;

            normal += radius * (neighbour_mass / neighbour_density)
                * derivative_spiky_pow2(distance, radius) * direction;
        }
    }

    var curvature_force = vec2<f32>(0.0, 0.0);
    let normal_len = length(normal);
    if (normal_len > EPSILON) {
        curvature_force = -surface_tension_gamma(my_type, my_type) * normal;
    }

    velocities[index] = velocities[index] + (cohesion_force + curvature_force) * uniforms.step0.y;
}

@compute @workgroup_size(64)
fn update_positions(@builtin(global_invocation_id) id: vec3<u32>) {
    let index = id.x;
    if (index >= num_particles()) {
        return;
    }

    positions[index] = positions[index] + velocities[index] * uniforms.step0.y;
    handle_collisions(index);
    let speed_sq = dot(velocities[index], velocities[index]);
    atomicMax(&grid_state[3], bitcast<u32>(speed_sq));
}

@compute @workgroup_size(64)
fn prepare_render_data(@builtin(global_invocation_id) id: vec3<u32>) {
    let index = id.x;
    if (index >= num_particles()) {
        return;
    }

    let ft = get_fluid_type(index);
    let density_norm = densities[index].x / max(fluid_target_density(ft), EPSILON);
    render_data[index] = vec4<f32>(
        positions[index].x,
        positions[index].y,
        length(velocities[index]),
        density_norm + f32(ft) * FLUID_TYPE_PACK,
    );
}
"#;

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct SimulationUniforms {
    counts0: [u32; 4],
    counts1: [u32; 4],
    step0: [f32; 4],
    step1: [f32; 4],
    bounds: [f32; 4],
    interaction: [f32; 4],
    obstacle: [f32; 4],
    kernels0: [f32; 4],
    kernels1: [f32; 4],
    fluid_type0: [f32; 4],
    fluid_type1: [f32; 4],
    multi_fluid: [f32; 4],
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable, Default)]
pub(crate) struct SimulationDiagnostics {
    pub(crate) peak_cell_occupancy: u32,
    pub(crate) dropped_particles: u32,
    pub(crate) overflowed_cells: u32,
    pub(crate) max_speed_sq_bits: u32,
}

pub(crate) struct GpuFluidSimulation {
    settings: SimulationSettings,
    max_speed_sq: f32,
    initial_positions: Vec<[f32; 2]>,
    initial_velocities: Vec<[f32; 2]>,
    initial_fluid_types: Vec<f32>,
    interaction: InteractionState,
    num_particles: usize,
    num_cells: u32,
    grid_width: u32,
    grid_height: u32,
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
    surface_tension_pipeline: wgpu::ComputePipeline,
    update_positions_pipeline: wgpu::ComputePipeline,
    prepare_render_data_pipeline: wgpu::ComputePipeline,
}

#[cfg_attr(test, allow(dead_code))]
impl GpuFluidSimulation {
    pub(crate) fn new(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        settings: SimulationSettings,
    ) -> Self {
        let spawn = build_spawn_points(&settings.spawn);
        let initial_positions: Vec<[f32; 2]> = spawn
            .positions
            .into_iter()
            .map(|value| [value.x, value.y])
            .collect();
        let initial_velocities: Vec<[f32; 2]> = spawn
            .velocities
            .into_iter()
            .map(|value| [value.x, value.y])
            .collect();
        let initial_fluid_types: Vec<f32> = spawn
            .fluid_types
            .into_iter()
            .map(|ft| ft as f32)
            .collect();
        let num_particles = initial_positions.len();
        let grid_width = ((settings.bounds_size.x / settings.smoothing_radius).ceil() as u32) + 3;
        let grid_height = ((settings.bounds_size.y / settings.smoothing_radius).ceil() as u32) + 3;
        let num_cells = grid_width.saturating_mul(grid_height);
        let max_particles_per_cell =
            estimate_max_particles_per_cell(&settings, &initial_positions, grid_width, grid_height);

        let uniform_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("simulation uniforms"),
            size: size_of::<SimulationUniforms>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let positions = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("simulation positions"),
            size: buffer_size::<[f32; 2]>(num_particles),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let predicted_positions = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("simulation predicted positions"),
            size: buffer_size::<[f32; 2]>(num_particles),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let velocities = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("simulation velocities"),
            size: buffer_size::<[f32; 2]>(num_particles),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let velocities_scratch = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("simulation velocities scratch"),
            size: buffer_size::<[f32; 2]>(num_particles),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let densities = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("simulation densities"),
            size: buffer_size::<[f32; 4]>(num_particles),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let grid_state = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("simulation grid state"),
            size: buffer_size::<u32>(num_cells as usize + DIAGNOSTICS_WORDS),
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let cell_particles = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("simulation cell particles"),
            size: buffer_size::<u32>((num_cells * max_particles_per_cell) as usize),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let render_data = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("simulation render data"),
            size: buffer_size::<[f32; 4]>(num_particles),
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::VERTEX
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("simulation bind group layout"),
            entries: &[
                uniform_entry(0, true),
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
            label: Some("simulation bind group"),
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
            label: Some("simulation compute shader"),
            source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(COMPUTE_SHADER)),
        });
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("simulation pipeline layout"),
            bind_group_layouts: &[Some(&bind_group_layout)],
            immediate_size: 0,
        });

        let mut simulation = Self {
            settings,
            max_speed_sq: 0.0,
            initial_positions,
            initial_velocities,
            initial_fluid_types,
            interaction: InteractionState::default(),
            num_particles,
            num_cells,
            grid_width,
            grid_height,
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
            surface_tension_pipeline: compute_pipeline(
                device,
                &pipeline_layout,
                &shader_module,
                "calculate_surface_tension",
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

    pub(crate) fn settings(&self) -> &SimulationSettings {
        &self.settings
    }

    pub(crate) fn interaction(&self) -> InteractionState {
        self.interaction
    }

    pub(crate) fn set_interaction(&mut self, x: f32, y: f32, strength: f32, active: bool) {
        self.interaction = InteractionState {
            point: Vec2::new(x, y),
            strength,
            active,
            radius: self.settings.interaction_radius,
        };
    }

    pub(crate) fn clear_interaction(&mut self) {
        self.interaction.active = false;
        self.interaction.strength = 0.0;
    }

    pub(crate) fn reset(&mut self, queue: &wgpu::Queue, device: &wgpu::Device) {
        self.clear_interaction();
        self.write_initial_state(queue);
        self.synchronize(queue, device);
    }

    pub(crate) fn particle_count(&self) -> usize {
        self.num_particles
    }

    pub(crate) fn max_particles_per_cell(&self) -> u32 {
        self.max_particles_per_cell
    }

    pub(crate) fn update_max_speed(&mut self, diagnostics: &SimulationDiagnostics) {
        let speed_sq = f32::from_bits(diagnostics.max_speed_sq_bits);
        if speed_sq.is_finite() {
            self.max_speed_sq = speed_sq;
        }
    }

    pub(crate) fn render_buffer(&self) -> &wgpu::Buffer {
        &self.render_data
    }


    #[cfg(target_arch = "wasm32")]
    pub(crate) fn diagnostics_buffer(&self) -> wgpu::Buffer {
        self.grid_state.clone()
    }

    #[cfg(target_arch = "wasm32")]
    pub(crate) async fn read_diagnostics_buffer(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        diagnostics: &wgpu::Buffer,
    ) -> Result<SimulationDiagnostics, String> {
        readback_value(device, queue, diagnostics).await
    }

    #[cfg(test)]
    fn read_diagnostics_blocking(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
    ) -> Result<SimulationDiagnostics, String> {
        readback_value_blocking(device, queue, &self.grid_state)
    }

    #[cfg(test)]
    fn read_render_data_blocking(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
    ) -> Result<Vec<[f32; 4]>, String> {
        readback_vec_blocking(device, queue, &self.render_data, self.num_particles)
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
            label: Some("simulation step encoder"),
        });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("simulation step pass"),
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
                    &self.surface_tension_pipeline,
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
            label: Some("simulation sync encoder"),
        });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("simulation sync pass"),
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

        let initial_densities: Vec<[f32; 4]> = self
            .initial_fluid_types
            .iter()
            .map(|&ft| [0.0, 0.0, ft, 0.0])
            .collect();
        queue.write_buffer(
            &self._densities,
            0,
            bytemuck::cast_slice(&initial_densities),
        );
    }

    fn write_uniforms(&self, queue: &wgpu::Queue, delta_time: f32) {
        let radius = self.settings.smoothing_radius;
        let uniform = SimulationUniforms {
            counts0: [
                self.num_particles as u32,
                self.num_cells,
                self.grid_width,
                self.grid_height,
            ],
            counts1: [self.max_particles_per_cell, 0, 0, 0],
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
            bounds: [
                self.settings.bounds_size.x,
                self.settings.bounds_size.y,
                -self.settings.bounds_size.x * 0.5 - radius,
                -self.settings.bounds_size.y * 0.5 - radius,
            ],
            interaction: [
                self.interaction.point.x,
                self.interaction.point.y,
                if self.interaction.active {
                    self.interaction.strength
                } else {
                    0.0
                },
                self.settings.interaction_radius,
            ],
            obstacle: [
                self.settings.obstacle.size.x,
                self.settings.obstacle.size.y,
                self.settings.obstacle.centre.x,
                self.settings.obstacle.centre.y,
            ],
            kernels0: [
                4.0 / (PI * radius.powi(8)),
                10.0 / (PI * radius.powi(5)),
                6.0 / (PI * radius.powi(4)),
                30.0 / (PI * radius.powi(5)),
            ],
            kernels1: [12.0 / (PI * radius.powi(4)), 0.0, 0.0, 0.0],
            fluid_type0: fluid_type_uniform(&self.settings, 0),
            fluid_type1: fluid_type_uniform(&self.settings, 1),
            multi_fluid: [
                self.settings.immiscibility_strength,
                self.settings.surface_tension_same[0],
                self.settings.surface_tension_same[1],
                self.settings.surface_tension_cross,
            ],
        };

        queue.write_buffer(&self.uniform_buffer, 0, bytemuck::bytes_of(&uniform));
    }

    fn particle_workgroups(&self) -> u32 {
        workgroups_for(self.num_particles as u32)
    }

    fn cell_workgroups(&self) -> u32 {
        workgroups_for(self.num_cells)
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

fn uniform_entry(binding: u32, _read_only: bool) -> wgpu::BindGroupLayoutEntry {
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

fn fluid_type_uniform(settings: &SimulationSettings, type_index: u8) -> [f32; 4] {
    let props = settings.fluid_type_props(type_index);
    [
        props.target_density,
        props.pressure_multiplier,
        props.near_pressure_multiplier,
        props.viscosity_strength,
    ]
}

fn estimate_max_particles_per_cell(
    settings: &SimulationSettings,
    initial_positions: &[[f32; 2]],
    grid_width: u32,
    grid_height: u32,
) -> u32 {
    let num_cells = grid_width.saturating_mul(grid_height);
    if initial_positions.is_empty() || num_cells == 0 {
        return 1;
    }

    let radius = settings.smoothing_radius.max(EPSILON);
    let grid_origin_x = -settings.bounds_size.x * 0.5 - radius;
    let grid_origin_y = -settings.bounds_size.y * 0.5 - radius;
    let mut occupancy = vec![0u32; num_cells as usize];
    let mut initial_peak = 0u32;

    for position in initial_positions {
        let cell_x = ((position[0] - grid_origin_x) / radius).floor() as i32;
        let cell_y = ((position[1] - grid_origin_y) / radius).floor() as i32;
        if cell_x < 0 || cell_y < 0 || cell_x as u32 >= grid_width || cell_y as u32 >= grid_height {
            continue;
        }

        let flat_index = cell_y as usize * grid_width as usize + cell_x as usize;
        occupancy[flat_index] += 1;
        initial_peak = initial_peak.max(occupancy[flat_index]);
    }

    let average_occupancy = (initial_positions.len() as u32).div_ceil(num_cells);
    let density_estimate =
        (settings.spawn.spawn_density * settings.smoothing_radius * settings.smoothing_radius)
            .ceil() as u32;
    let baseline = initial_peak.max(average_occupancy).max(density_estimate).max(1);

    baseline
        .saturating_mul(8)
        .next_power_of_two()
        .min(initial_positions.len() as u32)
        .max(baseline)
}

#[cfg(target_arch = "wasm32")]
async fn readback_value<T: Pod>(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    source: &wgpu::Buffer,
) -> Result<T, String> {
    let size = size_of::<T>() as u64;
    let download = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("simulation readback"),
        size,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });
    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("simulation readback encoder"),
    });
    encoder.copy_buffer_to_buffer(source, 0, &download, 0, size);
    queue.submit(Some(encoder.finish()));

    let state = Arc::new(Mutex::new(MapState::default()));
    let state_for_callback = Arc::clone(&state);
    download.slice(..).map_async(wgpu::MapMode::Read, move |result| {
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

#[cfg(target_arch = "wasm32")]
#[derive(Default)]
struct MapState {
    result: Option<Result<(), String>>,
    waker: Option<std::task::Waker>,
}

#[cfg(test)]
fn readback_value_blocking<T: Pod>(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    source: &wgpu::Buffer,
) -> Result<T, String> {
    let size = size_of::<T>() as u64;
    let download = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("simulation readback"),
        size,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });
    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("simulation readback encoder"),
    });
    encoder.copy_buffer_to_buffer(source, 0, &download, 0, size);
    queue.submit(Some(encoder.finish()));

    let (sender, receiver) = mpsc::channel();
    download.slice(..).map_async(wgpu::MapMode::Read, move |result| {
        let _ = sender.send(result.map_err(|error| error.to_string()));
    });
    device
        .poll(wgpu::PollType::wait_indefinitely())
        .map_err(|error| error.to_string())?;
    receiver
        .recv()
        .map_err(|error| error.to_string())??;

    let mapped = download.slice(..).get_mapped_range();
    let value = *bytemuck::from_bytes::<T>(&mapped);
    drop(mapped);
    download.unmap();
    Ok(value)
}

#[cfg(test)]
fn readback_vec_blocking<T: Pod>(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    source: &wgpu::Buffer,
    count: usize,
) -> Result<Vec<T>, String> {
    let size = buffer_size::<T>(count);
    let download = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("simulation vector readback"),
        size,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });
    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("simulation vector readback encoder"),
    });
    encoder.copy_buffer_to_buffer(source, 0, &download, 0, size);
    queue.submit(Some(encoder.finish()));

    let (sender, receiver) = mpsc::channel();
    download.slice(..).map_async(wgpu::MapMode::Read, move |result| {
        let _ = sender.send(result.map_err(|error| error.to_string()));
    });
    device
        .poll(wgpu::PollType::wait_indefinitely())
        .map_err(|error| error.to_string())?;
    receiver
        .recv()
        .map_err(|error| error.to_string())??;

    let mapped = download.slice(..).get_mapped_range();
    let values = bytemuck::cast_slice(&mapped).to_vec();
    drop(mapped);
    download.unmap();
    Ok(values)
}

#[cfg(test)]
mod tests {
    use std::cmp::Ordering;
    use std::future::Future;
    use std::pin::pin;
    use std::sync::Arc;
    use std::task::{Context, Poll, Wake, Waker};
    use std::thread;
    use std::time::Duration;

    use super::*;

    #[test]
    fn gpu_matches_cpu_reference_for_test_a() {
        let Some((device, queue)) = create_test_device() else {
            eprintln!("Skipping GPU parity test because no headless wgpu adapter is available.");
            return;
        };
        let settings = SimulationSettings::test_a();
        let mut cpu = FluidSimulation::from_settings(settings.clone());
        let mut gpu = GpuFluidSimulation::new(&device, &queue, settings);
        let frame_times = [1.0 / 60.0, 1.0 / 55.0, 1.0 / 72.0];

        for frame_time in frame_times {
            cpu.step_frame(frame_time);
            gpu.step_frame(&device, &queue, frame_time);
        }

        let mut cpu_particles = cpu
            .render_data
            .chunks_exact(4)
            .map(|chunk| [chunk[0], chunk[1], chunk[2], chunk[3]])
            .collect::<Vec<_>>();
        let mut gpu_particles = gpu
            .read_render_data_blocking(&device, &queue)
            .expect("read GPU render data");

        sort_particles(&mut cpu_particles);
        sort_particles(&mut gpu_particles);
        assert_eq!(cpu_particles.len(), gpu_particles.len());

        let max_error = cpu_particles
            .iter()
            .zip(&gpu_particles)
            .flat_map(|(cpu_particle, gpu_particle)| {
                cpu_particle
                    .iter()
                    .zip(gpu_particle.iter())
                    .map(|(cpu_value, gpu_value)| (cpu_value - gpu_value).abs())
            })
            .fold(0.0_f32, f32::max);

        assert!(
            max_error < 1.0e-3,
            "GPU parity drifted too far from CPU reference: max error {max_error}"
        );
    }

    #[test]
    fn presets_do_not_overflow_grid_in_steady_state() {
        let Some((device, queue)) = create_test_device() else {
            eprintln!("Skipping GPU overflow test because no headless wgpu adapter is available.");
            return;
        };

        for settings in [
            SimulationSettings::test_a(),
            SimulationSettings::test_b(),
            SimulationSettings::test_c(),
            SimulationSettings::oil_and_water(),
        ] {
            let preset_name = settings.preset_name;
            let mut gpu = GpuFluidSimulation::new(&device, &queue, settings);
            for _ in 0..12 {
                gpu.step_frame(&device, &queue, 1.0 / 60.0);
            }

            let diagnostics = gpu
                .read_diagnostics_blocking(&device, &queue)
                .expect("read diagnostics");
            assert_eq!(
                diagnostics.dropped_particles, 0,
                "{preset_name} dropped particles during grid build"
            );
            assert_eq!(
                diagnostics.overflowed_cells, 0,
                "{preset_name} overflowed grid cells during grid build"
            );
            assert!(
                diagnostics.peak_cell_occupancy <= gpu.max_particles_per_cell(),
                "{preset_name} exceeded configured cell capacity"
            );
        }
    }

    fn create_test_device() -> Option<(wgpu::Device, wgpu::Queue)> {
        let instance = wgpu::Instance::default();
        let adapter = block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::LowPower,
            force_fallback_adapter: false,
            compatible_surface: None,
        }))
        .ok()?;

        block_on(adapter.request_device(&wgpu::DeviceDescriptor {
            label: Some("fluid-wasm test device"),
            required_features: wgpu::Features::empty(),
            required_limits: wgpu::Limits::downlevel_defaults(),
            memory_hints: wgpu::MemoryHints::Performance,
            trace: wgpu::Trace::default(),
            experimental_features: wgpu::ExperimentalFeatures::disabled(),
        }))
        .ok()
    }

    fn block_on<F: Future>(future: F) -> F::Output {
        let parker = Arc::new(ThreadWaker {
            thread: thread::current(),
        });
        let waker = Waker::from(parker);
        let mut context = Context::from_waker(&waker);
        let mut future = pin!(future);

        loop {
            match future.as_mut().poll(&mut context) {
                Poll::Ready(value) => return value,
                Poll::Pending => thread::park_timeout(Duration::from_millis(10)),
            }
        }
    }

    fn sort_particles(particles: &mut [[f32; 4]]) {
        particles.sort_by(|left, right| compare_particle(left, right));
    }

    fn compare_particle(left: &[f32; 4], right: &[f32; 4]) -> Ordering {
        for (lhs, rhs) in left.iter().zip(right.iter()) {
            let ordering = lhs.total_cmp(rhs);
            if ordering != Ordering::Equal {
                return ordering;
            }
        }
        Ordering::Equal
    }

    struct ThreadWaker {
        thread: thread::Thread,
    }

    impl Wake for ThreadWaker {
        fn wake(self: Arc<Self>) {
            self.thread.unpark();
        }

        fn wake_by_ref(self: &Arc<Self>) {
            self.thread.unpark();
        }
    }
}
