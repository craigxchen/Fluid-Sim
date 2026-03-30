use std::borrow::Cow;
use std::mem::size_of;

use bytemuck::{Pod, Zeroable};
use js_sys::{Array, Promise};
use wasm_bindgen::prelude::*;
use web_sys::HtmlCanvasElement;

use crate::{
    EPSILON, InteractionState, Obstacle, SimulationSettings, Vec2, gpu_sim::GpuFluidSimulation,
};

const INTERACTION_RING_SEGMENTS: usize = 48;
const MIN_BUFFER_BYTES: usize = 16;

const PARTICLE_SHADER: &str = r#"
struct ParticleUniforms {
    bounds: vec2<f32>,
    radius: f32,
    padding: f32,
};

@group(0) @binding(0)
var<uniform> uniforms: ParticleUniforms;

struct ParticleVertexInput {
    @location(0) local: vec2<f32>,
    @location(1) world_position: vec2<f32>,
    @location(2) speed: f32,
    @location(3) density: f32,
};

struct ParticleVertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) local: vec2<f32>,
    @location(1) speed: f32,
    @location(2) density: f32,
};

@vertex
fn vs_main(input: ParticleVertexInput) -> ParticleVertexOutput {
    let world = input.world_position + input.local * uniforms.radius;
    let clip = vec2<f32>(
        world.x / (uniforms.bounds.x * 0.5),
        world.y / (uniforms.bounds.y * 0.5),
    );

    var output: ParticleVertexOutput;
    output.clip_position = vec4<f32>(clip, 0.0, 1.0);
    output.local = input.local;
    output.speed = input.speed;
    output.density = input.density;
    return output;
}

fn palette(speed: f32, density: f32) -> vec3<f32> {
    let speed_t = clamp(speed / 8.0, 0.0, 1.0);
    let density_t = clamp(density, 0.0, 1.5) / 1.5;

    let deep = vec3<f32>(0.08, 0.24, 0.56);
    let mid = vec3<f32>(0.11, 0.66, 0.73);
    let foam = vec3<f32>(0.96, 0.97, 0.89);
    let hot = vec3<f32>(0.94, 0.42, 0.17);

    let base = mix(deep, mid, density_t);
    let energy = mix(base, foam, speed_t * 0.65);
    return mix(energy, hot, speed_t * density_t * 0.55);
}

@fragment
fn fs_main(input: ParticleVertexOutput) -> @location(0) vec4<f32> {
    let radial = dot(input.local, input.local);
    if (radial > 1.0) {
        discard;
    }

    let point_coord = input.local * 0.5 + vec2<f32>(0.5, 0.5);
    let edge = smoothstep(1.0, 0.08, radial);
    let specular = smoothstep(0.85, 0.0, distance(point_coord, vec2<f32>(0.32, 0.3)));
    let color = palette(input.speed, input.density) + vec3<f32>(specular * 0.08);
    return vec4<f32>(color, edge * 0.92);
}
"#;

const SHAPE_SHADER: &str = r#"
struct ShapeVertexInput {
    @location(0) clip_position: vec2<f32>,
    @location(1) color: vec4<f32>,
};

struct ShapeVertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) color: vec4<f32>,
};

@vertex
fn vs_main(input: ShapeVertexInput) -> ShapeVertexOutput {
    var output: ShapeVertexOutput;
    output.clip_position = vec4<f32>(input.clip_position, 0.0, 1.0);
    output.color = input.color;
    return output;
}

@fragment
fn fs_main(input: ShapeVertexOutput) -> @location(0) vec4<f32> {
    return input.color;
}
"#;

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct QuadVertex {
    local: [f32; 2],
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct ParticleUniforms {
    bounds: [f32; 2],
    radius: f32,
    padding: f32,
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct ShapeVertex {
    clip_position: [f32; 2],
    color: [f32; 4],
}

enum ShapePipelineKind {
    Fill,
    Line,
}

struct ShapeDrawCommand {
    pipeline: ShapePipelineKind,
    start_vertex: u32,
    vertex_count: u32,
}

#[wasm_bindgen]
pub struct WasmFluidApp {
    simulation: GpuFluidSimulation,
    active_preset: usize,
    renderer: Renderer,
}

#[wasm_bindgen]
impl WasmFluidApp {
    #[wasm_bindgen(js_name = create)]
    pub async fn create(canvas: HtmlCanvasElement) -> Result<Self, JsValue> {
        console_error_panic_hook::set_once();

        let renderer = Renderer::new(canvas).await?;
        let simulation = GpuFluidSimulation::new(
            renderer.device(),
            renderer.queue(),
            SimulationSettings::test_a(),
        );

        Ok(Self {
            simulation,
            active_preset: 0,
            renderer,
        })
    }

    #[wasm_bindgen(js_name = loadPreset)]
    pub fn load_preset(&mut self, preset_index: usize) -> bool {
        match SimulationSettings::preset(preset_index) {
            Some(settings) => {
                self.simulation = GpuFluidSimulation::new(
                    self.renderer.device(),
                    self.renderer.queue(),
                    settings,
                );
                self.active_preset = preset_index;
                true
            }
            None => false,
        }
    }

    #[wasm_bindgen(js_name = presetCount)]
    pub fn preset_count(&self) -> usize {
        crate::PRESET_COUNT
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
        self.simulation
            .reset(self.renderer.queue(), self.renderer.device());
    }

    #[wasm_bindgen(js_name = stepFrame)]
    pub fn step_frame(&mut self, frame_time: f32) {
        self.simulation
            .step_frame(self.renderer.device(), self.renderer.queue(), frame_time);
    }

    #[wasm_bindgen(js_name = setInteraction)]
    pub fn set_interaction(&mut self, x: f32, y: f32, strength: f32, active: bool) {
        self.simulation.set_interaction(x, y, strength, active);
    }

    #[wasm_bindgen(js_name = clearInteraction)]
    pub fn clear_interaction(&mut self) {
        self.simulation.clear_interaction();
    }

    #[wasm_bindgen(js_name = resize)]
    pub fn resize(&mut self, width: u32, height: u32) {
        self.renderer.resize(width, height);
    }

    pub fn render(&mut self) -> Result<(), JsValue> {
        self.renderer.render(&self.simulation)
    }

    #[wasm_bindgen(js_name = particleCount)]
    pub fn particle_count(&self) -> usize {
        self.simulation.particle_count()
    }

    #[wasm_bindgen(js_name = maxParticlesPerCell)]
    pub fn max_particles_per_cell(&self) -> u32 {
        self.simulation.max_particles_per_cell()
    }

    #[wasm_bindgen(js_name = readDiagnostics)]
    pub fn read_diagnostics(&self) -> Promise {
        let device = self.renderer.device().clone();
        let queue = self.renderer.queue().clone();
        let diagnostics_buffer = self.simulation.diagnostics_buffer();
        let capacity = self.simulation.max_particles_per_cell();

        wasm_bindgen_futures::future_to_promise(async move {
            let diagnostics =
                GpuFluidSimulation::read_diagnostics_buffer(&device, &queue, &diagnostics_buffer)
                    .await
                    .map_err(|error| JsValue::from_str(&error))?;

            Ok(Array::of4(
                &JsValue::from_f64(diagnostics.peak_cell_occupancy as f64),
                &JsValue::from_f64(diagnostics.dropped_particles as f64),
                &JsValue::from_f64(diagnostics.overflowed_cells as f64),
                &JsValue::from_f64(capacity as f64),
            )
            .into())
        })
    }

    #[wasm_bindgen(js_name = boundsWidth)]
    pub fn bounds_width(&self) -> f32 {
        self.simulation.settings().bounds_size.x
    }

    #[wasm_bindgen(js_name = boundsHeight)]
    pub fn bounds_height(&self) -> f32 {
        self.simulation.settings().bounds_size.y
    }

    #[wasm_bindgen(js_name = particleRadius)]
    pub fn particle_radius(&self) -> f32 {
        self.simulation.settings().render_radius
    }

    #[wasm_bindgen(js_name = interactionRadius)]
    pub fn interaction_radius(&self) -> f32 {
        self.simulation.settings().interaction_radius
    }

    #[wasm_bindgen(js_name = interactionStrength)]
    pub fn interaction_strength(&self) -> f32 {
        self.simulation.settings().interaction_strength
    }

    #[wasm_bindgen(js_name = presetName)]
    pub fn preset_name(&self) -> String {
        self.simulation.settings().preset_name.to_owned()
    }
}

struct Renderer {
    surface: wgpu::Surface<'static>,
    device: wgpu::Device,
    queue: wgpu::Queue,
    config: wgpu::SurfaceConfiguration,
    particle_pipeline: wgpu::RenderPipeline,
    shape_fill_pipeline: wgpu::RenderPipeline,
    shape_line_pipeline: wgpu::RenderPipeline,
    particle_uniform_buffer: wgpu::Buffer,
    particle_bind_group: wgpu::BindGroup,
    quad_vertex_buffer: wgpu::Buffer,
    shape_buffer: wgpu::Buffer,
    shape_capacity: usize,
}

impl Renderer {
    async fn new(canvas: HtmlCanvasElement) -> Result<Self, JsValue> {
        let width = canvas.width().max(1);
        let height = canvas.height().max(1);

        let instance = wgpu::Instance::default();
        let surface = instance
            .create_surface(wgpu::SurfaceTarget::Canvas(canvas))
            .map_err(js_error)?;
        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                force_fallback_adapter: false,
                compatible_surface: Some(&surface),
            })
            .await
            .map_err(js_error)?;

        let (device, queue) = adapter
            .request_device(&wgpu::DeviceDescriptor {
                label: Some("fluid-wasm device"),
                required_features: wgpu::Features::empty(),
                required_limits: wgpu::Limits::default(),
                memory_hints: wgpu::MemoryHints::Performance,
                trace: wgpu::Trace::default(),
                experimental_features: wgpu::ExperimentalFeatures::disabled(),
            })
            .await
            .map_err(js_error)?;

        let mut config = surface
            .get_default_config(&adapter, width, height)
            .ok_or_else(|| {
                JsValue::from_str("The current browser adapter cannot present to this canvas.")
            })?;
        config.alpha_mode = wgpu::CompositeAlphaMode::PreMultiplied;
        surface.configure(&device, &config);

        let particle_uniform_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("particle uniforms"),
            size: size_of::<ParticleUniforms>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let particle_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("particle bind group layout"),
                entries: &[wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::VERTEX,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                }],
            });
        let particle_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("particle bind group"),
            layout: &particle_bind_group_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: particle_uniform_buffer.as_entire_binding(),
            }],
        });

        let particle_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("particle shader"),
            source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(PARTICLE_SHADER)),
        });
        let shape_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("shape shader"),
            source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(SHAPE_SHADER)),
        });

        let particle_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("particle pipeline layout"),
                bind_group_layouts: &[Some(&particle_bind_group_layout)],
                immediate_size: 0,
            });
        let shape_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("shape pipeline layout"),
                bind_group_layouts: &[],
                immediate_size: 0,
            });

        let particle_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("particle pipeline"),
            layout: Some(&particle_pipeline_layout),
            vertex: wgpu::VertexState {
                module: &particle_shader,
                entry_point: Some("vs_main"),
                compilation_options: wgpu::PipelineCompilationOptions::default(),
                buffers: &[
                    wgpu::VertexBufferLayout {
                        array_stride: size_of::<QuadVertex>() as u64,
                        step_mode: wgpu::VertexStepMode::Vertex,
                        attributes: &[wgpu::VertexAttribute {
                            offset: 0,
                            shader_location: 0,
                            format: wgpu::VertexFormat::Float32x2,
                        }],
                    },
                    wgpu::VertexBufferLayout {
                        array_stride: (size_of::<f32>() * 4) as u64,
                        step_mode: wgpu::VertexStepMode::Instance,
                        attributes: &[
                            wgpu::VertexAttribute {
                                offset: 0,
                                shader_location: 1,
                                format: wgpu::VertexFormat::Float32x2,
                            },
                            wgpu::VertexAttribute {
                                offset: 8,
                                shader_location: 2,
                                format: wgpu::VertexFormat::Float32,
                            },
                            wgpu::VertexAttribute {
                                offset: 12,
                                shader_location: 3,
                                format: wgpu::VertexFormat::Float32,
                            },
                        ],
                    },
                ],
            },
            fragment: Some(wgpu::FragmentState {
                module: &particle_shader,
                entry_point: Some("fs_main"),
                compilation_options: wgpu::PipelineCompilationOptions::default(),
                targets: &[Some(wgpu::ColorTargetState {
                    format: config.format,
                    blend: Some(wgpu::BlendState::ALPHA_BLENDING),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                strip_index_format: None,
                front_face: wgpu::FrontFace::Ccw,
                cull_mode: None,
                unclipped_depth: false,
                polygon_mode: wgpu::PolygonMode::Fill,
                conservative: false,
            },
            depth_stencil: None,
            multisample: wgpu::MultisampleState::default(),
            multiview_mask: None,
            cache: None,
        });

        let shape_vertex_layout = wgpu::VertexBufferLayout {
            array_stride: size_of::<ShapeVertex>() as u64,
            step_mode: wgpu::VertexStepMode::Vertex,
            attributes: &[
                wgpu::VertexAttribute {
                    offset: 0,
                    shader_location: 0,
                    format: wgpu::VertexFormat::Float32x2,
                },
                wgpu::VertexAttribute {
                    offset: 8,
                    shader_location: 1,
                    format: wgpu::VertexFormat::Float32x4,
                },
            ],
        };

        let shape_fill_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("shape fill pipeline"),
            layout: Some(&shape_pipeline_layout),
            vertex: wgpu::VertexState {
                module: &shape_shader,
                entry_point: Some("vs_main"),
                compilation_options: wgpu::PipelineCompilationOptions::default(),
                buffers: std::slice::from_ref(&shape_vertex_layout),
            },
            fragment: Some(wgpu::FragmentState {
                module: &shape_shader,
                entry_point: Some("fs_main"),
                compilation_options: wgpu::PipelineCompilationOptions::default(),
                targets: &[Some(wgpu::ColorTargetState {
                    format: config.format,
                    blend: Some(wgpu::BlendState::ALPHA_BLENDING),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleStrip,
                strip_index_format: None,
                front_face: wgpu::FrontFace::Ccw,
                cull_mode: None,
                unclipped_depth: false,
                polygon_mode: wgpu::PolygonMode::Fill,
                conservative: false,
            },
            depth_stencil: None,
            multisample: wgpu::MultisampleState::default(),
            multiview_mask: None,
            cache: None,
        });

        let shape_line_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("shape line pipeline"),
            layout: Some(&shape_pipeline_layout),
            vertex: wgpu::VertexState {
                module: &shape_shader,
                entry_point: Some("vs_main"),
                compilation_options: wgpu::PipelineCompilationOptions::default(),
                buffers: &[shape_vertex_layout],
            },
            fragment: Some(wgpu::FragmentState {
                module: &shape_shader,
                entry_point: Some("fs_main"),
                compilation_options: wgpu::PipelineCompilationOptions::default(),
                targets: &[Some(wgpu::ColorTargetState {
                    format: config.format,
                    blend: Some(wgpu::BlendState::ALPHA_BLENDING),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::LineStrip,
                strip_index_format: None,
                front_face: wgpu::FrontFace::Ccw,
                cull_mode: None,
                unclipped_depth: false,
                polygon_mode: wgpu::PolygonMode::Fill,
                conservative: false,
            },
            depth_stencil: None,
            multisample: wgpu::MultisampleState::default(),
            multiview_mask: None,
            cache: None,
        });

        let quad_vertices = [
            QuadVertex {
                local: [-1.0, -1.0],
            },
            QuadVertex { local: [1.0, -1.0] },
            QuadVertex { local: [1.0, 1.0] },
            QuadVertex {
                local: [-1.0, -1.0],
            },
            QuadVertex { local: [1.0, 1.0] },
            QuadVertex { local: [-1.0, 1.0] },
        ];
        let quad_vertex_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("quad vertex buffer"),
            size: (quad_vertices.len() * size_of::<QuadVertex>()) as u64,
            usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        queue.write_buffer(&quad_vertex_buffer, 0, bytemuck::cast_slice(&quad_vertices));

        let shape_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("shape vertex buffer"),
            size: MIN_BUFFER_BYTES as u64,
            usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        Ok(Self {
            surface,
            device,
            queue,
            config,
            particle_pipeline,
            shape_fill_pipeline,
            shape_line_pipeline,
            particle_uniform_buffer,
            particle_bind_group,
            quad_vertex_buffer,
            shape_buffer,
            shape_capacity: MIN_BUFFER_BYTES,
        })
    }

    fn device(&self) -> &wgpu::Device {
        &self.device
    }

    fn queue(&self) -> &wgpu::Queue {
        &self.queue
    }

    fn resize(&mut self, width: u32, height: u32) {
        if width == 0 || height == 0 {
            return;
        }

        self.config.width = width;
        self.config.height = height;
        self.surface.configure(&self.device, &self.config);
    }

    fn render(&mut self, simulation: &GpuFluidSimulation) -> Result<(), JsValue> {
        if self.config.width == 0 || self.config.height == 0 {
            return Ok(());
        }

        self.update_particle_uniforms(simulation.settings());
        let (shape_vertices, shape_commands) = self.build_shape_batch(simulation);
        if !shape_vertices.is_empty() {
            self.ensure_shape_capacity(shape_vertices.len() * size_of::<ShapeVertex>());
            self.queue
                .write_buffer(&self.shape_buffer, 0, bytemuck::cast_slice(&shape_vertices));
        }

        let frame = match self.surface.get_current_texture() {
            wgpu::CurrentSurfaceTexture::Success(frame)
            | wgpu::CurrentSurfaceTexture::Suboptimal(frame) => frame,
            wgpu::CurrentSurfaceTexture::Timeout | wgpu::CurrentSurfaceTexture::Occluded => {
                return Ok(());
            }
            wgpu::CurrentSurfaceTexture::Outdated => {
                self.surface.configure(&self.device, &self.config);
                return Ok(());
            }
            wgpu::CurrentSurfaceTexture::Lost | wgpu::CurrentSurfaceTexture::Validation => {
                return Err(JsValue::from_str(
                    "wgpu could not acquire the current surface texture.",
                ));
            }
        };

        let view = frame
            .texture
            .create_view(&wgpu::TextureViewDescriptor::default());
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("fluid render encoder"),
            });

        {
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("fluid render pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &view,
                    resolve_target: None,
                    depth_slice: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
                multiview_mask: None,
            });

            if !shape_vertices.is_empty() {
                pass.set_vertex_buffer(0, self.shape_buffer.slice(..));
                for command in &shape_commands {
                    match command.pipeline {
                        ShapePipelineKind::Fill => pass.set_pipeline(&self.shape_fill_pipeline),
                        ShapePipelineKind::Line => pass.set_pipeline(&self.shape_line_pipeline),
                    }
                    pass.draw(
                        command.start_vertex..command.start_vertex + command.vertex_count,
                        0..1,
                    );
                }
            }

            if simulation.particle_count() > 0 {
                pass.set_pipeline(&self.particle_pipeline);
                pass.set_bind_group(0, &self.particle_bind_group, &[]);
                pass.set_vertex_buffer(0, self.quad_vertex_buffer.slice(..));
                pass.set_vertex_buffer(1, simulation.render_buffer().slice(..));
                pass.draw(0..6, 0..simulation.particle_count() as u32);
            }
        }

        self.queue.submit(Some(encoder.finish()));
        frame.present();
        Ok(())
    }

    fn update_particle_uniforms(&self, settings: &SimulationSettings) {
        let uniforms = ParticleUniforms {
            bounds: [
                settings.bounds_size.x.max(EPSILON),
                settings.bounds_size.y.max(EPSILON),
            ],
            radius: settings.render_radius,
            padding: 0.0,
        };
        self.queue.write_buffer(
            &self.particle_uniform_buffer,
            0,
            bytemuck::bytes_of(&uniforms),
        );
    }

    fn ensure_shape_capacity(&mut self, required_bytes: usize) {
        if required_bytes <= self.shape_capacity {
            return;
        }

        let new_capacity = required_bytes.next_power_of_two().max(MIN_BUFFER_BYTES) as u64;
        self.shape_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("shape vertex buffer"),
            size: new_capacity,
            usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        self.shape_capacity = new_capacity as usize;
    }

    fn build_shape_batch(
        &self,
        simulation: &GpuFluidSimulation,
    ) -> (Vec<ShapeVertex>, Vec<ShapeDrawCommand>) {
        let settings = simulation.settings();
        let mut vertices = Vec::with_capacity(96);
        let mut commands = Vec::new();

        self.push_shape(
            &mut vertices,
            &mut commands,
            ShapePipelineKind::Line,
            &bounds_outline(settings.bounds_size),
            [0.08, 0.20, 0.28, 0.30],
        );

        let obstacle = settings.obstacle;
        if obstacle.size.x > 0.0 && obstacle.size.y > 0.0 {
            self.push_shape(
                &mut vertices,
                &mut commands,
                ShapePipelineKind::Fill,
                &obstacle_fill(settings.bounds_size, obstacle),
                [0.05, 0.20, 0.28, 0.14],
            );
            self.push_shape(
                &mut vertices,
                &mut commands,
                ShapePipelineKind::Line,
                &obstacle_outline(settings.bounds_size, obstacle),
                [0.05, 0.20, 0.28, 0.32],
            );
        }

        let interaction = simulation.interaction();
        if interaction.active {
            self.push_shape(
                &mut vertices,
                &mut commands,
                ShapePipelineKind::Line,
                &interaction_ring(
                    settings.bounds_size,
                    interaction,
                    settings.interaction_radius,
                ),
                [0.87, 0.42, 0.15, 0.45],
            );
        }

        (vertices, commands)
    }

    fn push_shape(
        &self,
        vertices: &mut Vec<ShapeVertex>,
        commands: &mut Vec<ShapeDrawCommand>,
        pipeline: ShapePipelineKind,
        clip_positions: &[[f32; 2]],
        color: [f32; 4],
    ) {
        if clip_positions.is_empty() {
            return;
        }

        let start_vertex = vertices.len() as u32;
        vertices.extend(clip_positions.iter().map(|clip_position| ShapeVertex {
            clip_position: *clip_position,
            color,
        }));
        commands.push(ShapeDrawCommand {
            pipeline,
            start_vertex,
            vertex_count: clip_positions.len() as u32,
        });
    }
}

fn bounds_outline(bounds: Vec2) -> [[f32; 2]; 5] {
    [
        world_to_clip(bounds, Vec2::new(-bounds.x * 0.5, bounds.y * 0.5)),
        world_to_clip(bounds, Vec2::new(bounds.x * 0.5, bounds.y * 0.5)),
        world_to_clip(bounds, Vec2::new(bounds.x * 0.5, -bounds.y * 0.5)),
        world_to_clip(bounds, Vec2::new(-bounds.x * 0.5, -bounds.y * 0.5)),
        world_to_clip(bounds, Vec2::new(-bounds.x * 0.5, bounds.y * 0.5)),
    ]
}

fn obstacle_fill(bounds: Vec2, obstacle: Obstacle) -> [[f32; 2]; 4] {
    [
        world_to_clip(
            bounds,
            Vec2::new(
                obstacle.centre.x - obstacle.size.x * 0.5,
                obstacle.centre.y + obstacle.size.y * 0.5,
            ),
        ),
        world_to_clip(
            bounds,
            Vec2::new(
                obstacle.centre.x + obstacle.size.x * 0.5,
                obstacle.centre.y + obstacle.size.y * 0.5,
            ),
        ),
        world_to_clip(
            bounds,
            Vec2::new(
                obstacle.centre.x - obstacle.size.x * 0.5,
                obstacle.centre.y - obstacle.size.y * 0.5,
            ),
        ),
        world_to_clip(
            bounds,
            Vec2::new(
                obstacle.centre.x + obstacle.size.x * 0.5,
                obstacle.centre.y - obstacle.size.y * 0.5,
            ),
        ),
    ]
}

fn obstacle_outline(bounds: Vec2, obstacle: Obstacle) -> [[f32; 2]; 5] {
    [
        world_to_clip(
            bounds,
            Vec2::new(
                obstacle.centre.x - obstacle.size.x * 0.5,
                obstacle.centre.y + obstacle.size.y * 0.5,
            ),
        ),
        world_to_clip(
            bounds,
            Vec2::new(
                obstacle.centre.x + obstacle.size.x * 0.5,
                obstacle.centre.y + obstacle.size.y * 0.5,
            ),
        ),
        world_to_clip(
            bounds,
            Vec2::new(
                obstacle.centre.x + obstacle.size.x * 0.5,
                obstacle.centre.y - obstacle.size.y * 0.5,
            ),
        ),
        world_to_clip(
            bounds,
            Vec2::new(
                obstacle.centre.x - obstacle.size.x * 0.5,
                obstacle.centre.y - obstacle.size.y * 0.5,
            ),
        ),
        world_to_clip(
            bounds,
            Vec2::new(
                obstacle.centre.x - obstacle.size.x * 0.5,
                obstacle.centre.y + obstacle.size.y * 0.5,
            ),
        ),
    ]
}

fn interaction_ring(bounds: Vec2, interaction: InteractionState, radius: f32) -> Vec<[f32; 2]> {
    let mut vertices = Vec::with_capacity(INTERACTION_RING_SEGMENTS + 1);
    for index in 0..=INTERACTION_RING_SEGMENTS {
        let angle = index as f32 / INTERACTION_RING_SEGMENTS as f32 * std::f32::consts::TAU;
        let world = Vec2::new(
            interaction.point.x + angle.cos() * radius,
            interaction.point.y + angle.sin() * radius,
        );
        vertices.push(world_to_clip(bounds, world));
    }
    vertices
}

fn world_to_clip(bounds: Vec2, world: Vec2) -> [f32; 2] {
    [
        world.x / (bounds.x.max(EPSILON) * 0.5),
        world.y / (bounds.y.max(EPSILON) * 0.5),
    ]
}

fn js_error(error: impl ToString) -> JsValue {
    JsValue::from_str(&error.to_string())
}
