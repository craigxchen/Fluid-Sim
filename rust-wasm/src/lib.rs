use std::f32::consts::PI;

use wasm_bindgen::prelude::*;

#[cfg(target_arch = "wasm32")]
mod gpu_sim;
#[cfg(target_arch = "wasm32")]
mod renderer;
#[cfg(target_arch = "wasm32")]
pub use renderer::WasmFluidApp;

const PREDICTION_FACTOR: f32 = 1.0 / 120.0;
const EPSILON: f32 = 1.0e-6;
const PARTICLE_STRIDE: usize = 4;
const PRESET_COUNT: usize = 3;
const NEIGHBOUR_OFFSETS: [(i32, i32); 9] = [
    (-1, 1),
    (0, 1),
    (1, 1),
    (-1, 0),
    (0, 0),
    (1, 0),
    (-1, -1),
    (0, -1),
    (1, -1),
];

#[wasm_bindgen(js_name = wasmMemory)]
pub fn wasm_memory() -> JsValue {
    wasm_bindgen::memory()
}

#[wasm_bindgen]
pub struct WasmFluidSimulation {
    simulation: FluidSimulation,
    active_preset: usize,
}

#[wasm_bindgen]
impl WasmFluidSimulation {
    #[wasm_bindgen(constructor)]
    pub fn new() -> Self {
        console_error_panic_hook::set_once();
        Self::with_preset(0)
    }

    #[wasm_bindgen(js_name = loadPreset)]
    pub fn load_preset(&mut self, preset_index: usize) -> bool {
        match FluidSimulation::from_preset(preset_index) {
            Some(simulation) => {
                self.simulation = simulation;
                self.active_preset = preset_index;
                true
            }
            None => false,
        }
    }

    #[wasm_bindgen(js_name = presetCount)]
    pub fn preset_count(&self) -> usize {
        PRESET_COUNT
    }

    #[wasm_bindgen(js_name = presetLabel)]
    pub fn preset_label(&self, preset_index: usize) -> String {
        SimulationSettings::preset_name(preset_index).to_owned()
    }

    #[wasm_bindgen(js_name = activePreset)]
    pub fn active_preset(&self) -> usize {
        self.active_preset
    }

    pub fn reset(&mut self) {
        self.simulation.reset();
    }

    #[wasm_bindgen(js_name = stepFrame)]
    pub fn step_frame(&mut self, frame_time: f32) {
        self.simulation.step_frame(frame_time);
    }

    #[wasm_bindgen(js_name = setInteraction)]
    pub fn set_interaction(&mut self, x: f32, y: f32, strength: f32, active: bool) {
        self.simulation.interaction = InteractionState {
            point: Vec2::new(x, y),
            strength,
            active,
            radius: self.simulation.settings.interaction_radius,
        };
    }

    #[wasm_bindgen(js_name = clearInteraction)]
    pub fn clear_interaction(&mut self) {
        self.simulation.interaction.active = false;
        self.simulation.interaction.strength = 0.0;
    }

    #[wasm_bindgen(js_name = particleDataPtr)]
    pub fn particle_data_ptr(&self) -> usize {
        self.simulation.render_data.as_ptr() as usize
    }

    #[wasm_bindgen(js_name = particleDataLen)]
    pub fn particle_data_len(&self) -> usize {
        self.simulation.render_data.len()
    }

    #[wasm_bindgen(js_name = particleStride)]
    pub fn particle_stride(&self) -> usize {
        PARTICLE_STRIDE
    }

    #[wasm_bindgen(js_name = particleCount)]
    pub fn particle_count(&self) -> usize {
        self.simulation.positions.len()
    }

    #[wasm_bindgen(js_name = boundsWidth)]
    pub fn bounds_width(&self) -> f32 {
        self.simulation.settings.bounds_size.x
    }

    #[wasm_bindgen(js_name = boundsHeight)]
    pub fn bounds_height(&self) -> f32 {
        self.simulation.settings.bounds_size.y
    }

    #[wasm_bindgen(js_name = obstacleData)]
    pub fn obstacle_data(&self) -> Vec<f32> {
        let obstacle = self.simulation.settings.obstacle;
        vec![
            obstacle.centre.x,
            obstacle.centre.y,
            obstacle.size.x,
            obstacle.size.y,
        ]
    }

    #[wasm_bindgen(js_name = particleRadius)]
    pub fn particle_radius(&self) -> f32 {
        self.simulation.settings.render_radius
    }

    #[wasm_bindgen(js_name = interactionRadius)]
    pub fn interaction_radius(&self) -> f32 {
        self.simulation.settings.interaction_radius
    }

    #[wasm_bindgen(js_name = interactionStrength)]
    pub fn interaction_strength(&self) -> f32 {
        self.simulation.settings.interaction_strength
    }

    #[wasm_bindgen(js_name = presetName)]
    pub fn preset_name(&self) -> String {
        self.simulation.settings.preset_name.to_owned()
    }

    fn with_preset(preset_index: usize) -> Self {
        let simulation = FluidSimulation::from_preset(preset_index)
            .unwrap_or_else(|| FluidSimulation::from_settings(SimulationSettings::test_a()));

        Self {
            simulation,
            active_preset: preset_index.min(PRESET_COUNT - 1),
        }
    }
}

#[derive(Clone)]
struct FluidSimulation {
    settings: SimulationSettings,
    initial_positions: Vec<Vec2>,
    initial_velocities: Vec<Vec2>,
    positions: Vec<Vec2>,
    predicted_positions: Vec<Vec2>,
    velocities: Vec<Vec2>,
    velocity_scratch: Vec<Vec2>,
    densities: Vec<Vec2>,
    spatial_keys: Vec<u32>,
    spatial_offsets: Vec<usize>,
    render_data: Vec<f32>,
    interaction: InteractionState,
}

impl FluidSimulation {
    fn from_preset(preset_index: usize) -> Option<Self> {
        SimulationSettings::preset(preset_index).map(Self::from_settings)
    }

    fn from_settings(settings: SimulationSettings) -> Self {
        let spawn = build_spawn_points(&settings.spawn);
        let particle_count = spawn.positions.len();
        let mut simulation = Self {
            settings,
            initial_positions: spawn.positions.clone(),
            initial_velocities: spawn.velocities.clone(),
            positions: spawn.positions.clone(),
            predicted_positions: spawn.positions,
            velocities: spawn.velocities.clone(),
            velocity_scratch: vec![Vec2::ZERO; particle_count],
            densities: vec![Vec2::ZERO; particle_count],
            spatial_keys: vec![0; particle_count],
            spatial_offsets: vec![usize::MAX; particle_count.max(1)],
            render_data: vec![0.0; particle_count * PARTICLE_STRIDE],
            interaction: InteractionState::default(),
        };
        simulation.synchronize_render_state();
        simulation
    }

    fn reset(&mut self) {
        self.positions.clone_from(&self.initial_positions);
        self.predicted_positions.clone_from(&self.initial_positions);
        self.velocities.clone_from(&self.initial_velocities);
        self.densities.fill(Vec2::ZERO);
        self.interaction.active = false;
        self.interaction.strength = 0.0;
        self.synchronize_render_state();
    }

    fn step_frame(&mut self, frame_time: f32) {
        if self.positions.is_empty() {
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

        for _ in 0..self.settings.iterations_per_frame.max(1) {
            self.run_step(step_delta);
        }

        self.refresh_render_data();
    }

    fn run_step(&mut self, delta_time: f32) {
        self.apply_external_forces(delta_time);
        self.update_spatial_lookup();
        self.calculate_densities();
        self.apply_pressure_forces(delta_time);
        self.apply_viscosity(delta_time);
        self.update_positions(delta_time);
    }

    fn synchronize_render_state(&mut self) {
        self.update_spatial_lookup();
        self.calculate_densities();
        self.refresh_render_data();
    }

    fn apply_external_forces(&mut self, delta_time: f32) {
        for index in 0..self.positions.len() {
            let acceleration = self.external_force(self.positions[index], self.velocities[index]);
            self.velocities[index] += acceleration * delta_time;
            self.predicted_positions[index] =
                self.positions[index] + self.velocities[index] * PREDICTION_FACTOR;
        }
    }

    fn external_force(&self, position: Vec2, velocity: Vec2) -> Vec2 {
        let gravity = Vec2::new(0.0, self.settings.gravity);
        if !self.interaction.active || self.interaction.strength.abs() <= EPSILON {
            return gravity;
        }

        let offset = self.interaction.point - position;
        let sqr_dst = offset.length_squared();
        let radius_sq = self.interaction.radius * self.interaction.radius;
        if sqr_dst >= radius_sq {
            return gravity;
        }

        let distance = sqr_dst.sqrt();
        let edge_t = distance / self.interaction.radius.max(EPSILON);
        let centre_t = 1.0 - edge_t;
        let direction = if distance > EPSILON {
            offset / distance
        } else {
            Vec2::ZERO
        };

        let gravity_weight = 1.0 - centre_t * (self.interaction.strength / 10.0).clamp(0.0, 1.0);
        gravity * gravity_weight + direction * centre_t * self.interaction.strength
            - velocity * centre_t
    }

    fn update_spatial_lookup(&mut self) {
        let particle_count = self.positions.len();
        let table_size = particle_count.max(1);
        let mut entries = Vec::with_capacity(particle_count);

        for index in 0..particle_count {
            let cell = get_cell(
                self.predicted_positions[index],
                self.settings.smoothing_radius,
            );
            let hash = hash_cell(cell);
            let key = hash % table_size as u32;
            entries.push((key, index));
        }

        entries.sort_unstable_by_key(|(key, _)| *key);
        self.spatial_offsets.clear();
        self.spatial_offsets.resize(table_size, usize::MAX);

        let mut sorted_positions = Vec::with_capacity(particle_count);
        let mut sorted_predicted = Vec::with_capacity(particle_count);
        let mut sorted_velocities = Vec::with_capacity(particle_count);
        let mut sorted_keys = Vec::with_capacity(particle_count);

        for (sorted_index, (key, original_index)) in entries.into_iter().enumerate() {
            if self.spatial_offsets[key as usize] == usize::MAX {
                self.spatial_offsets[key as usize] = sorted_index;
            }

            sorted_positions.push(self.positions[original_index]);
            sorted_predicted.push(self.predicted_positions[original_index]);
            sorted_velocities.push(self.velocities[original_index]);
            sorted_keys.push(key);
        }

        self.positions = sorted_positions;
        self.predicted_positions = sorted_predicted;
        self.velocities = sorted_velocities;
        self.spatial_keys = sorted_keys;
    }

    fn calculate_densities(&mut self) {
        for index in 0..self.positions.len() {
            self.densities[index] = self.calculate_density(self.predicted_positions[index]);
        }
    }

    fn calculate_density(&self, position: Vec2) -> Vec2 {
        let origin = get_cell(position, self.settings.smoothing_radius);
        let radius_sq = self.settings.smoothing_radius * self.settings.smoothing_radius;
        let mut density = 0.0;
        let mut near_density = 0.0;

        for offset in NEIGHBOUR_OFFSETS {
            let hash = hash_cell((origin.0 + offset.0, origin.1 + offset.1));
            let key = hash % self.positions.len().max(1) as u32;
            let mut current = self.spatial_offsets[key as usize];
            if current == usize::MAX {
                continue;
            }

            while current < self.positions.len() {
                if self.spatial_keys[current] != key {
                    break;
                }

                let neighbour_pos = self.predicted_positions[current];
                let to_neighbour = neighbour_pos - position;
                let sqr_dst = to_neighbour.length_squared();
                if sqr_dst <= radius_sq {
                    let dst = sqr_dst.sqrt();
                    density += spiky_kernel_pow2(dst, self.settings.smoothing_radius);
                    near_density += spiky_kernel_pow3(dst, self.settings.smoothing_radius);
                }
                current += 1;
            }
        }

        Vec2::new(density, near_density)
    }

    fn apply_pressure_forces(&mut self, delta_time: f32) {
        let particle_count = self.positions.len();
        self.velocity_scratch.clone_from(&self.velocities);

        for index in 0..particle_count {
            let density = self.densities[index].x.max(EPSILON);
            let near_density = self.densities[index].y.max(EPSILON);
            let pressure = self.pressure_from_density(density);
            let near_pressure = self.near_pressure_from_density(near_density);
            let position = self.predicted_positions[index];
            let origin = get_cell(position, self.settings.smoothing_radius);
            let radius_sq = self.settings.smoothing_radius * self.settings.smoothing_radius;
            let mut pressure_force = Vec2::ZERO;

            for offset in NEIGHBOUR_OFFSETS {
                let hash = hash_cell((origin.0 + offset.0, origin.1 + offset.1));
                let key = hash % particle_count.max(1) as u32;
                let mut current = self.spatial_offsets[key as usize];
                if current == usize::MAX {
                    continue;
                }

                while current < particle_count {
                    if self.spatial_keys[current] != key {
                        break;
                    }
                    if current == index {
                        current += 1;
                        continue;
                    }

                    let neighbour_pos = self.predicted_positions[current];
                    let to_neighbour = neighbour_pos - position;
                    let sqr_dst = to_neighbour.length_squared();
                    if sqr_dst > radius_sq {
                        current += 1;
                        continue;
                    }

                    let dst = sqr_dst.sqrt();
                    let direction = if dst > EPSILON {
                        to_neighbour / dst
                    } else {
                        Vec2::new(0.0, 1.0)
                    };

                    let neighbour_density = self.densities[current].x.max(EPSILON);
                    let neighbour_near_density = self.densities[current].y.max(EPSILON);
                    let neighbour_pressure = self.pressure_from_density(neighbour_density);
                    let neighbour_near_pressure =
                        self.near_pressure_from_density(neighbour_near_density);

                    let shared_pressure = (pressure + neighbour_pressure) * 0.5;
                    let shared_near_pressure = (near_pressure + neighbour_near_pressure) * 0.5;

                    pressure_force += direction
                        * derivative_spiky_pow2(dst, self.settings.smoothing_radius)
                        * shared_pressure
                        / neighbour_density;
                    pressure_force += direction
                        * derivative_spiky_pow3(dst, self.settings.smoothing_radius)
                        * shared_near_pressure
                        / neighbour_near_density;

                    current += 1;
                }
            }

            self.velocity_scratch[index] += pressure_force / density * delta_time;
        }

        std::mem::swap(&mut self.velocities, &mut self.velocity_scratch);
    }

    fn apply_viscosity(&mut self, delta_time: f32) {
        let particle_count = self.positions.len();
        self.velocity_scratch.clone_from(&self.velocities);

        for index in 0..particle_count {
            let position = self.predicted_positions[index];
            let origin = get_cell(position, self.settings.smoothing_radius);
            let radius_sq = self.settings.smoothing_radius * self.settings.smoothing_radius;
            let velocity = self.velocities[index];
            let mut viscosity_force = Vec2::ZERO;

            for offset in NEIGHBOUR_OFFSETS {
                let hash = hash_cell((origin.0 + offset.0, origin.1 + offset.1));
                let key = hash % particle_count.max(1) as u32;
                let mut current = self.spatial_offsets[key as usize];
                if current == usize::MAX {
                    continue;
                }

                while current < particle_count {
                    if self.spatial_keys[current] != key {
                        break;
                    }
                    if current == index {
                        current += 1;
                        continue;
                    }

                    let neighbour_pos = self.predicted_positions[current];
                    let to_neighbour = neighbour_pos - position;
                    let sqr_dst = to_neighbour.length_squared();
                    if sqr_dst <= radius_sq {
                        let dst = sqr_dst.sqrt();
                        viscosity_force += (self.velocities[current] - velocity)
                            * smoothing_kernel_poly6(dst, self.settings.smoothing_radius);
                    }

                    current += 1;
                }
            }

            self.velocity_scratch[index] +=
                viscosity_force * self.settings.viscosity_strength * delta_time;
        }

        std::mem::swap(&mut self.velocities, &mut self.velocity_scratch);
    }

    fn update_positions(&mut self, delta_time: f32) {
        for index in 0..self.positions.len() {
            self.positions[index] += self.velocities[index] * delta_time;
            self.handle_collisions(index);
        }
    }

    fn handle_collisions(&mut self, index: usize) {
        let half_bounds = self.settings.bounds_size * 0.5;
        let obstacle_half = self.settings.obstacle.size * 0.5;
        let obstacle_delta = self.positions[index] - self.settings.obstacle.centre;

        let mut position = self.positions[index];
        let mut velocity = self.velocities[index];

        let edge_distance = half_bounds - position.abs();
        if edge_distance.x <= 0.0 {
            position.x = half_bounds.x * signed_unit(position.x);
            velocity.x *= -self.settings.collision_damping;
        }
        if edge_distance.y <= 0.0 {
            position.y = half_bounds.y * signed_unit(position.y);
            velocity.y *= -self.settings.collision_damping;
        }

        let obstacle_edge_distance = obstacle_half - obstacle_delta.abs();
        if obstacle_edge_distance.x >= 0.0 && obstacle_edge_distance.y >= 0.0 {
            if obstacle_edge_distance.x < obstacle_edge_distance.y {
                position.x = obstacle_half.x * signed_unit(obstacle_delta.x)
                    + self.settings.obstacle.centre.x;
                velocity.x *= -self.settings.collision_damping;
            } else {
                position.y = obstacle_half.y * signed_unit(obstacle_delta.y)
                    + self.settings.obstacle.centre.y;
                velocity.y *= -self.settings.collision_damping;
            }
        }

        self.positions[index] = position;
        self.velocities[index] = velocity;
    }

    fn refresh_render_data(&mut self) {
        for index in 0..self.positions.len() {
            let write_index = index * PARTICLE_STRIDE;
            self.render_data[write_index] = self.positions[index].x;
            self.render_data[write_index + 1] = self.positions[index].y;
            self.render_data[write_index + 2] = self.velocities[index].length();
            self.render_data[write_index + 3] =
                self.densities[index].x / self.settings.target_density.max(EPSILON);
        }
    }

    fn pressure_from_density(&self, density: f32) -> f32 {
        (density - self.settings.target_density) * self.settings.pressure_multiplier
    }

    fn near_pressure_from_density(&self, density: f32) -> f32 {
        self.settings.near_pressure_multiplier * density
    }
}

#[derive(Clone)]
struct SimulationSettings {
    preset_name: &'static str,
    time_scale: f32,
    max_timestep_fps: f32,
    iterations_per_frame: usize,
    gravity: f32,
    collision_damping: f32,
    smoothing_radius: f32,
    target_density: f32,
    pressure_multiplier: f32,
    near_pressure_multiplier: f32,
    viscosity_strength: f32,
    bounds_size: Vec2,
    obstacle: Obstacle,
    interaction_radius: f32,
    interaction_strength: f32,
    render_radius: f32,
    spawn: SpawnSettings,
}

impl SimulationSettings {
    fn preset(index: usize) -> Option<Self> {
        match index {
            0 => Some(Self::test_a()),
            1 => Some(Self::test_b()),
            2 => Some(Self::test_c()),
            _ => None,
        }
    }

    fn preset_name(index: usize) -> &'static str {
        match index {
            0 => "Unity Test A",
            1 => "Unity Test B",
            2 => "Unity Test C",
            _ => "Unknown Preset",
        }
    }

    fn test_a() -> Self {
        Self {
            preset_name: "Unity Test A",
            time_scale: 1.0,
            max_timestep_fps: 60.0,
            iterations_per_frame: 3,
            gravity: -12.0,
            collision_damping: 0.95,
            smoothing_radius: 0.35,
            target_density: 55.0,
            pressure_multiplier: 500.0,
            near_pressure_multiplier: 5.0,
            viscosity_strength: 0.03,
            bounds_size: Vec2::new(17.1, 9.3),
            obstacle: Obstacle {
                centre: Vec2::ZERO,
                size: Vec2::ZERO,
            },
            interaction_radius: 2.0,
            interaction_strength: 90.0,
            render_radius: 0.08,
            spawn: SpawnSettings {
                spawn_density: 159.0,
                initial_velocity: Vec2::ZERO,
                jitter_strength: 0.03,
                regions: vec![SpawnRegion {
                    position: Vec2::new(0.0, 0.66),
                    size: Vec2::new(6.42, 4.39),
                }],
            },
        }
    }

    fn test_b() -> Self {
        Self {
            preset_name: "Unity Test B",
            time_scale: 1.0,
            max_timestep_fps: 60.0,
            iterations_per_frame: 3,
            gravity: -13.0,
            collision_damping: 0.5,
            smoothing_radius: 0.2,
            target_density: 234.0,
            pressure_multiplier: 225.0,
            near_pressure_multiplier: 18.0,
            viscosity_strength: 0.03,
            bounds_size: Vec2::new(17.1, 9.3),
            obstacle: Obstacle {
                centre: Vec2::new(0.03, -1.0),
                size: Vec2::new(2.75, 5.0),
            },
            interaction_radius: 2.5,
            interaction_strength: 75.0,
            render_radius: 0.045,
            spawn: SpawnSettings {
                spawn_density: 500.0,
                initial_velocity: Vec2::ZERO,
                jitter_strength: 0.03,
                regions: vec![
                    SpawnRegion {
                        position: Vec2::new(-5.25, 2.35),
                        size: Vec2::new(4.71, 3.4),
                    },
                    SpawnRegion {
                        position: Vec2::new(5.53, 2.35),
                        size: Vec2::new(4.71, 3.4),
                    },
                ],
            },
        }
    }

    fn test_c() -> Self {
        Self {
            preset_name: "Unity Test C",
            time_scale: 1.0,
            max_timestep_fps: 60.0,
            iterations_per_frame: 3,
            gravity: -13.0,
            collision_damping: 0.5,
            smoothing_radius: 0.175,
            target_density: 234.0,
            pressure_multiplier: 225.0,
            near_pressure_multiplier: 5.0,
            viscosity_strength: 0.0,
            bounds_size: Vec2::new(43.1, 23.86),
            obstacle: Obstacle {
                centre: Vec2::new(7.5, -8.16),
                size: Vec2::new(20.4, 1.69),
            },
            interaction_radius: 2.5,
            interaction_strength: 75.0,
            render_radius: 0.07,
            spawn: SpawnSettings {
                spawn_density: 500.0,
                initial_velocity: Vec2::ZERO,
                jitter_strength: 0.03,
                regions: vec![SpawnRegion {
                    position: Vec2::new(0.0, 2.95),
                    size: Vec2::new(34.0, 15.0),
                }],
            },
        }
    }
}

#[derive(Clone)]
struct SpawnSettings {
    spawn_density: f32,
    initial_velocity: Vec2,
    jitter_strength: f32,
    regions: Vec<SpawnRegion>,
}

#[derive(Clone, Copy)]
struct SpawnRegion {
    position: Vec2,
    size: Vec2,
}

struct SpawnData {
    positions: Vec<Vec2>,
    velocities: Vec<Vec2>,
}

fn build_spawn_points(settings: &SpawnSettings) -> SpawnData {
    let mut rng = SimpleRng::new(42);
    let mut positions = Vec::new();
    let mut velocities = Vec::new();

    for region in &settings.regions {
        let counts = calculate_spawn_count_per_axis(region.size, settings.spawn_density);
        for y in 0..counts.1 {
            for x in 0..counts.0 {
                let tx = if counts.0 > 1 {
                    x as f32 / (counts.0 - 1) as f32
                } else {
                    0.5
                };
                let ty = if counts.1 > 1 {
                    y as f32 / (counts.1 - 1) as f32
                } else {
                    0.5
                };

                let position = Vec2::new(
                    (tx - 0.5) * region.size.x + region.position.x,
                    (ty - 0.5) * region.size.y + region.position.y,
                );

                let angle = rng.next_f32() * PI * 2.0;
                let direction = Vec2::new(angle.cos(), angle.sin());
                let jitter = direction * settings.jitter_strength * (rng.next_f32() - 0.5);
                positions.push(position + jitter);
                velocities.push(settings.initial_velocity);
            }
        }
    }

    SpawnData {
        positions,
        velocities,
    }
}

fn calculate_spawn_count_per_axis(size: Vec2, spawn_density: f32) -> (usize, usize) {
    let area = size.x * size.y;
    let target_total = (area * spawn_density).ceil().max(1.0);
    let length_sum = (size.x + size.y).max(EPSILON);
    let t = size / length_sum;
    let scale = (target_total / (t.x * t.y).max(EPSILON)).sqrt();
    let nx = (t.x * scale).ceil().max(1.0) as usize;
    let ny = (t.y * scale).ceil().max(1.0) as usize;
    (nx, ny)
}

#[derive(Clone, Copy)]
struct Obstacle {
    centre: Vec2,
    size: Vec2,
}

#[derive(Clone, Copy)]
struct InteractionState {
    point: Vec2,
    strength: f32,
    radius: f32,
    active: bool,
}

impl Default for InteractionState {
    fn default() -> Self {
        Self {
            point: Vec2::ZERO,
            strength: 0.0,
            radius: 0.0,
            active: false,
        }
    }
}

#[derive(Clone, Copy, Default)]
struct Vec2 {
    x: f32,
    y: f32,
}

impl Vec2 {
    const ZERO: Self = Self { x: 0.0, y: 0.0 };

    const fn new(x: f32, y: f32) -> Self {
        Self { x, y }
    }

    fn abs(self) -> Self {
        Self::new(self.x.abs(), self.y.abs())
    }

    fn length(self) -> f32 {
        self.length_squared().sqrt()
    }

    fn length_squared(self) -> f32 {
        self.x * self.x + self.y * self.y
    }
}

impl std::ops::Add for Vec2 {
    type Output = Self;

    fn add(self, rhs: Self) -> Self::Output {
        Self::new(self.x + rhs.x, self.y + rhs.y)
    }
}

impl std::ops::AddAssign for Vec2 {
    fn add_assign(&mut self, rhs: Self) {
        self.x += rhs.x;
        self.y += rhs.y;
    }
}

impl std::ops::Sub for Vec2 {
    type Output = Self;

    fn sub(self, rhs: Self) -> Self::Output {
        Self::new(self.x - rhs.x, self.y - rhs.y)
    }
}

impl std::ops::Mul<f32> for Vec2 {
    type Output = Self;

    fn mul(self, rhs: f32) -> Self::Output {
        Self::new(self.x * rhs, self.y * rhs)
    }
}

impl std::ops::Div<f32> for Vec2 {
    type Output = Self;

    fn div(self, rhs: f32) -> Self::Output {
        if rhs.abs() <= EPSILON {
            Self::ZERO
        } else {
            Self::new(self.x / rhs, self.y / rhs)
        }
    }
}

fn smoothing_kernel_poly6(distance: f32, radius: f32) -> f32 {
    if distance < radius {
        let value = radius * radius - distance * distance;
        value * value * value * (4.0 / (PI * radius.powi(8)))
    } else {
        0.0
    }
}

fn spiky_kernel_pow3(distance: f32, radius: f32) -> f32 {
    if distance < radius {
        let value = radius - distance;
        value * value * value * (10.0 / (PI * radius.powi(5)))
    } else {
        0.0
    }
}

fn spiky_kernel_pow2(distance: f32, radius: f32) -> f32 {
    if distance < radius {
        let value = radius - distance;
        value * value * (6.0 / (PI * radius.powi(4)))
    } else {
        0.0
    }
}

fn derivative_spiky_pow3(distance: f32, radius: f32) -> f32 {
    if distance <= radius {
        let value = radius - distance;
        -value * value * (30.0 / (PI * radius.powi(5)))
    } else {
        0.0
    }
}

fn derivative_spiky_pow2(distance: f32, radius: f32) -> f32 {
    if distance <= radius {
        let value = radius - distance;
        -value * (12.0 / (PI * radius.powi(4)))
    } else {
        0.0
    }
}

fn get_cell(position: Vec2, radius: f32) -> (i32, i32) {
    (
        (position.x / radius).floor() as i32,
        (position.y / radius).floor() as i32,
    )
}

fn hash_cell(cell: (i32, i32)) -> u32 {
    let x = cell.0 as u32;
    let y = cell.1 as u32;
    x.wrapping_mul(15_823)
        .wrapping_add(y.wrapping_mul(9_737_333))
}

fn signed_unit(value: f32) -> f32 {
    if value > 0.0 {
        1.0
    } else if value < 0.0 {
        -1.0
    } else {
        0.0
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
