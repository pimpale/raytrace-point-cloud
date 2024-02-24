use vulkano::{acceleration_structure::AabbPositions, buffer::BufferContents, pipeline::graphics::vertex_input::Vertex};

#[derive(Clone, Copy, Debug, BufferContents, Vertex, Default)]
#[repr(C)]
pub struct Vertex3D {
    #[format(R32G32B32_SFLOAT)]
    pub position: [f32; 3],
    #[format(R32G32B32_SFLOAT)]
    pub tuv: [f32; 3],
}

impl Vertex3D {
    pub fn new(position: [f32; 3], tuv: [f32; 3]) -> Self {
        Self { position, tuv }
    }

    pub fn new2(position: [f32; 3], t: u32, uv: [f32; 2]) -> Self {
        Self {
            position,
            tuv: [t as f32, uv[0], uv[1]],
        }
    }
}

#[derive(Clone, Copy, Debug, BufferContents)]
#[repr(C)]
pub struct InstanceData {
    pub vertex_buffer_addr: u64,
    pub transform: [[f32; 3]; 4],
}

#[derive(Clone, Copy, Debug, BufferContents, Default)]
#[repr(C)]
pub struct GaussianSplat {
    pub aabb: AabbPositions,
}

impl GaussianSplat {
    pub fn new(min: [f32; 3], max: [f32; 3]) -> Self {
        Self { aabb: AabbPositions { min, max } }
    }
}
