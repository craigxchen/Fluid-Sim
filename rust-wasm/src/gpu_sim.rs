use std::borrow::Cow;
use std::f32::consts::PI;
use std::mem::size_of;

use bytemuck::{Pod, Zeroable};

use crate::{EPSILON, InteractionState, SimulationSettings, Vec2, build_spawn_points};

const NUM_THREADS: u32 = 64;

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
var<storage, read_write> densities: array<vec2<f32>>;
@group(0) @binding(6)
var<storage, read_write> cell_counts: array<atomic<u32>>;
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

fn pressure_from_density(density: f32) -> f32 {
    return (density - uniforms.step1.x) * uniforms.step1.y;
}

fn near_pressure_from_density(near_density: f32) -> f32 {
    return uniforms.step1.z * near_density;
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
        let count = min(atomicLoad(&cell_counts[cell_index]), max_particles_per_cell());
        for (var slot = 0u; slot < count; slot += 1u) {
            let neighbour_index = cell_particles[cell_index * max_particles_per_cell() + slot];
            let neighbour_position = predicted_positions[neighbour_index];
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
    if (cell_index >= num_cells()) {
        return;
    }

    atomicStore(&cell_counts[cell_index], 0u);
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
    let slot = atomicAdd(&cell_counts[cell_index], 1u);
    if (slot < max_particles_per_cell()) {
        cell_particles[cell_index * max_particles_per_cell() + slot] = index;
    }
}

@compute @workgroup_size(64)
fn calculate_densities(@builtin(global_invocation_id) id: vec3<u32>) {
    let index = id.x;
    if (index >= num_particles()) {
        return;
    }

    densities[index] = calculate_density(predicted_positions[index]);
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
        let count = min(atomicLoad(&cell_counts[cell_index]), max_particles_per_cell());
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

    velocities_scratch[index] = velocities[index] + pressure_force / density * uniforms.step0.y;
}

@compute @workgroup_size(64)
fn calculate_viscosity(@builtin(global_invocation_id) id: vec3<u32>) {
    let index = id.x;
    if (index >= num_particles()) {
        return;
    }

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
        let count = min(atomicLoad(&cell_counts[cell_index]), max_particles_per_cell());
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
            viscosity_force +=
                (velocities_scratch[neighbour_index] - velocity) * smoothing_kernel_poly6(distance, radius);
        }
    }

    velocities[index] = velocity + viscosity_force * uniforms.step1.w * uniforms.step0.y;
}

@compute @workgroup_size(64)
fn update_positions(@builtin(global_invocation_id) id: vec3<u32>) {
    let index = id.x;
    if (index >= num_particles()) {
        return;
    }

    positions[index] = positions[index] + velocities[index] * uniforms.step0.y;
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
        length(velocities[index]),
        densities[index].x / max(target_density(), EPSILON),
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
}

pub(crate) struct GpuFluidSimulation {
    settings: SimulationSettings,
    initial_positions: Vec<[f32; 2]>,
    initial_velocities: Vec<[f32; 2]>,
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
    _cell_counts: wgpu::Buffer,
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
            size: buffer_size::<[f32; 2]>(num_particles),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let cell_counts = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("simulation cell counts"),
            size: buffer_size::<u32>(num_cells as usize),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
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
                buffer_entry(6, &cell_counts),
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
            initial_positions,
            initial_velocities,
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
            _cell_counts: cell_counts,
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

    pub(crate) fn render_buffer(&self) -> &wgpu::Buffer {
        &self.render_data
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
