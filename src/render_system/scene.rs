use std::{
    collections::{BTreeMap, VecDeque},
    fmt::Debug,
    sync::Arc,
};

use nalgebra::{Isometry3, Matrix3x4, Matrix4};
use vulkano::{
    acceleration_structure::{
        AabbPositions, AccelerationStructure, AccelerationStructureBuildGeometryInfo,
        AccelerationStructureBuildRangeInfo, AccelerationStructureBuildSizesInfo,
        AccelerationStructureBuildType, AccelerationStructureCreateInfo,
        AccelerationStructureGeometries, AccelerationStructureGeometryAabbsData,
        AccelerationStructureGeometryInstancesData, AccelerationStructureGeometryInstancesDataType,
        AccelerationStructureInstance, AccelerationStructureType, BuildAccelerationStructureFlags,
        BuildAccelerationStructureMode, GeometryFlags,
    },
    buffer::{Buffer, BufferContents, BufferCreateInfo, BufferUsage, Subbuffer},
    command_buffer::{
        allocator::StandardCommandBufferAllocator, AutoCommandBufferBuilder, CommandBufferUsage,
        CopyBufferInfo, PrimaryAutoCommandBuffer, PrimaryCommandBufferAbstract,
    },
    device::Queue,
    memory::allocator::{AllocationCreateInfo, MemoryAllocator, MemoryTypeFilter},
    sync::GpuFuture,
    DeviceSize, Packed24_8,
};

use super::vertex::{GaussianSplat, InstanceData};

pub struct Object {
    isometry: Isometry3<f32>,
    gsplat_buffer: Subbuffer<[GaussianSplat]>,
    vertex_buffer: Subbuffer<[AabbPositions]>,
    blas: Arc<AccelerationStructure>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum TopLevelAccelerationStructureState {
    UpToDate,
    NeedsRebuild,
}

/// Corresponds to a TLAS
pub struct Scene<K> {
    general_queue: Arc<Queue>,
    transfer_queue: Arc<Queue>,
    command_buffer_allocator: Arc<StandardCommandBufferAllocator>,
    memory_allocator: Arc<dyn MemoryAllocator>,
    objects: BTreeMap<K, Option<Object>>,
    // we have to keep around old objects for n_swapchain_images frames to ensure that the TLAS is not in use
    old_objects: VecDeque<Vec<Object>>,
    n_swapchain_images: usize,
    // cached data from the last frame
    cached_tlas: Option<Arc<AccelerationStructure>>,
    cached_instance_data: Option<Subbuffer<[InstanceData]>>,
    // last frame state
    cached_tlas_state: TopLevelAccelerationStructureState,
    // command buffer all building commands are submitted to
    blas_command_buffer: AutoCommandBufferBuilder<PrimaryAutoCommandBuffer>,
}

#[allow(dead_code)]
impl<K> Scene<K>
where
    K: Ord + Clone + std::cmp::Eq + std::hash::Hash,
{
    pub fn new(
        general_queue: Arc<Queue>,
        transfer_queue: Arc<Queue>,
        memory_allocator: Arc<dyn MemoryAllocator>,
        command_buffer_allocator: Arc<StandardCommandBufferAllocator>,
        n_swapchain_images: usize,
    ) -> Scene<K> {
        let command_buffer = AutoCommandBufferBuilder::primary(
            command_buffer_allocator.as_ref(),
            general_queue.queue_family_index(),
            CommandBufferUsage::OneTimeSubmit,
        )
        .unwrap();

        Scene {
            general_queue,
            transfer_queue,
            command_buffer_allocator,
            memory_allocator,
            objects: BTreeMap::new(),
            old_objects: VecDeque::from([vec![]]),
            n_swapchain_images,
            cached_tlas: None,
            cached_instance_data: None,
            cached_tlas_state: TopLevelAccelerationStructureState::NeedsRebuild,
            blas_command_buffer: command_buffer,
        }
    }

    // adds a new object to the scene with the given isometry
    pub fn add_object(
        &mut self,
        key: K,
        object: &Vec<(AabbPositions, GaussianSplat)>,
        isometry: Isometry3<f32>,
    ) {
        if object.len() == 0 {
            self.objects.insert(key, None);
            return;
        }

        let (vertex_buffer, gsplat_buffer) = blas_vertex_buffer(
            self.command_buffer_allocator.clone(),
            self.transfer_queue.clone(),
            self.memory_allocator.clone(),
            object,
        );
        let blas = create_bottom_level_acceleration_structure_aabb(
            &mut self.blas_command_buffer,
            self.memory_allocator.clone(),
            &vertex_buffer,
        );

        self.objects.insert(
            key,
            Some(Object {
                isometry,
                gsplat_buffer,
                vertex_buffer,
                blas,
            }),
        );
        self.cached_tlas_state = TopLevelAccelerationStructureState::NeedsRebuild;
    }

    // updates the isometry of the object with the given key
    pub fn update_object(&mut self, key: K, isometry: Isometry3<f32>) {
        match self.objects.get_mut(&key) {
            Some(Some(object)) => {
                object.isometry = isometry;
                if self.cached_tlas_state == TopLevelAccelerationStructureState::UpToDate {
                    self.cached_tlas_state = TopLevelAccelerationStructureState::NeedsRebuild;
                }
            }
            Some(None) => {}
            None => panic!("object with key does not exist"),
        }
    }

    pub fn remove_object(&mut self, key: K) {
        let removed = self.objects.remove(&key);
        if let Some(removed) = removed.flatten() {
            self.old_objects[0].push(removed);
            self.cached_tlas_state = TopLevelAccelerationStructureState::NeedsRebuild;
        }
    }

    // SAFETY: after calling this function, any TLAS previously returned by get_tlas() is invalid, and must not in use
    pub unsafe fn dispose_old_objects(&mut self) {
        self.old_objects.push_front(vec![]);
        while self.old_objects.len() > self.n_swapchain_images + 10 {
            self.old_objects.pop_back();
        }
    }

    // the returned TLAS may only be used after the returned future has been waited on
    pub fn get_tlas(
        &mut self,
    ) -> (
        Arc<AccelerationStructure>,
        Subbuffer<[InstanceData]>,
        Box<dyn GpuFuture>,
    ) {
        // rebuild the instance transforms buffer if any object was moved, added, or removed
        if self.cached_tlas_state != TopLevelAccelerationStructureState::UpToDate {
            let instance_data = Buffer::from_iter(
                self.memory_allocator.clone(),
                BufferCreateInfo {
                    usage: BufferUsage::STORAGE_BUFFER,
                    ..Default::default()
                },
                AllocationCreateInfo {
                    memory_type_filter: MemoryTypeFilter::PREFER_DEVICE
                        | MemoryTypeFilter::HOST_SEQUENTIAL_WRITE,
                    ..Default::default()
                },
                self.objects
                    .values()
                    .flatten()
                    .map(
                        |Object {
                             isometry,
                             vertex_buffer,
                             gsplat_buffer,
                             ..
                         }| InstanceData {
                            transform: {
                                let mat4: Matrix4<f32> = isometry.clone().into();
                                let mat3x4: Matrix3x4<f32> = mat4.fixed_view::<3, 4>(0, 0).into();
                                mat3x4.into()
                            },
                            vertex_buffer_addr: vertex_buffer.device_address().unwrap().get(),
                            gsplat_buffer_addr: gsplat_buffer.device_address().unwrap().get(),
                        },
                    )
                    .collect::<Vec<_>>(),
            )
            .unwrap();

            self.cached_instance_data = Some(instance_data);
        }

        let future = match self.cached_tlas_state {
            TopLevelAccelerationStructureState::UpToDate => {
                vulkano::sync::now(self.general_queue.device().clone()).boxed()
            }
            _ => {
                // swap command buffers
                let blas_command_buffer = std::mem::replace(
                    &mut self.blas_command_buffer,
                    AutoCommandBufferBuilder::primary(
                        self.command_buffer_allocator.as_ref(),
                        self.general_queue.queue_family_index(),
                        CommandBufferUsage::OneTimeSubmit,
                    )
                    .unwrap(),
                );

                let blas_build_future = blas_command_buffer
                    .build()
                    .unwrap()
                    .execute(self.general_queue.clone())
                    .unwrap();

                let mut tlas_command_buffer = AutoCommandBufferBuilder::primary(
                    self.command_buffer_allocator.as_ref(),
                    self.general_queue.queue_family_index(),
                    CommandBufferUsage::OneTimeSubmit,
                )
                .unwrap();

                // initialize tlas build
                let tlas = create_top_level_acceleration_structure(
                    &mut tlas_command_buffer,
                    self.memory_allocator.clone(),
                    &self
                        .objects
                        .values()
                        .flatten()
                        .map(|Object { blas, isometry, .. }| {
                            (blas as &AccelerationStructure, isometry)
                        })
                        .collect::<Vec<_>>(),
                );

                // actually submit acceleration structure build future
                let tlas_build_future = tlas_command_buffer
                    .build()
                    .unwrap()
                    .execute_after(blas_build_future, self.general_queue.clone())
                    .unwrap();

                // update state
                self.cached_tlas = Some(tlas);

                // return the future
                tlas_build_future.boxed()
            }
        };

        // at this point the tlas is up to date
        self.cached_tlas_state = TopLevelAccelerationStructureState::UpToDate;

        // return the tlas
        return (
            self.cached_tlas.clone().unwrap(),
            self.cached_instance_data.clone().unwrap(),
            future,
        );
    }
}

fn blas_vertex_buffer(
    command_buffer_allocator: Arc<StandardCommandBufferAllocator>,
    transfer_queue: Arc<Queue>,
    memory_allocator: Arc<dyn MemoryAllocator>,
    objects: &[(AabbPositions, GaussianSplat)],
) -> (Subbuffer<[AabbPositions]>, Subbuffer<[GaussianSplat]>) {
    let n_vertexes = objects.len() as u64;
    let (vertexes, gsplats): (Vec<_>, Vec<_>) = objects.iter().cloned().unzip();

    let vertex_buffer_tmp = Buffer::from_iter(
        memory_allocator.clone(),
        BufferCreateInfo {
            usage: BufferUsage::TRANSFER_SRC,
            ..Default::default()
        },
        AllocationCreateInfo {
            memory_type_filter: MemoryTypeFilter::PREFER_DEVICE
                | MemoryTypeFilter::HOST_SEQUENTIAL_WRITE,
            ..Default::default()
        },
        vertexes,
    )
    .unwrap();

    let gsplat_buffer_tmp = Buffer::from_iter(
        memory_allocator.clone(),
        BufferCreateInfo {
            usage: BufferUsage::TRANSFER_SRC,
            ..Default::default()
        },
        AllocationCreateInfo {
            memory_type_filter: MemoryTypeFilter::PREFER_DEVICE
                | MemoryTypeFilter::HOST_SEQUENTIAL_WRITE,
            ..Default::default()
        },
        gsplats,
    )
    .unwrap();

    // command buffers
    let mut transfer_command_buffer = AutoCommandBufferBuilder::primary(
        command_buffer_allocator.as_ref(),
        transfer_queue.queue_family_index(),
        CommandBufferUsage::OneTimeSubmit,
    )
    .unwrap();

    let vertex_buffer = Buffer::new_slice(
        memory_allocator.clone(),
        BufferCreateInfo {
            usage: BufferUsage::ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY
                | BufferUsage::SHADER_DEVICE_ADDRESS
                | BufferUsage::TRANSFER_DST,
            ..Default::default()
        },
        AllocationCreateInfo {
            memory_type_filter: MemoryTypeFilter::PREFER_DEVICE,
            ..Default::default()
        },
        n_vertexes,
    )
    .unwrap();

    let gsplat_buffer = Buffer::new_slice(
        memory_allocator.clone(),
        BufferCreateInfo {
            usage: BufferUsage::SHADER_DEVICE_ADDRESS | BufferUsage::TRANSFER_DST,
            ..Default::default()
        },
        AllocationCreateInfo {
            memory_type_filter: MemoryTypeFilter::PREFER_DEVICE,
            ..Default::default()
        },
        n_vertexes,
    )
    .unwrap();

    transfer_command_buffer
        .copy_buffer(CopyBufferInfo::buffers(
            vertex_buffer_tmp.clone(),
            vertex_buffer.clone(),
        ))
        .unwrap()
        .copy_buffer(CopyBufferInfo::buffers(
            gsplat_buffer_tmp.clone(),
            gsplat_buffer.clone(),
        ))
        .unwrap();

    transfer_command_buffer
        .build()
        .unwrap()
        .execute(transfer_queue.clone())
        .unwrap()
        .then_signal_fence()
        .wait(None)
        .unwrap();

    (vertex_buffer, gsplat_buffer)
}

fn create_top_level_acceleration_structure(
    builder: &mut AutoCommandBufferBuilder<PrimaryAutoCommandBuffer>,
    memory_allocator: Arc<dyn MemoryAllocator>,
    bottom_level_acceleration_structures: &[(&AccelerationStructure, &Isometry3<f32>)],
) -> Arc<AccelerationStructure> {
    let instances = bottom_level_acceleration_structures
        .iter()
        .map(
            |(bottom_level_acceleration_structure, &isometry)| AccelerationStructureInstance {
                instance_shader_binding_table_record_offset_and_flags: Packed24_8::new(0, 0),
                acceleration_structure_reference: bottom_level_acceleration_structure
                    .device_address()
                    .get(),
                transform: {
                    let isometry_matrix: [[f32; 4]; 4] = Matrix4::from(isometry).transpose().into();
                    [isometry_matrix[0], isometry_matrix[1], isometry_matrix[2]]
                },
                ..Default::default()
            },
        )
        .collect::<Vec<_>>();

    let values = Buffer::from_iter(
        memory_allocator.clone(),
        BufferCreateInfo {
            usage: BufferUsage::ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY
                | BufferUsage::SHADER_DEVICE_ADDRESS,
            ..Default::default()
        },
        AllocationCreateInfo {
            memory_type_filter: MemoryTypeFilter::PREFER_DEVICE
                | MemoryTypeFilter::HOST_SEQUENTIAL_WRITE,
            ..Default::default()
        },
        instances,
    )
    .unwrap();

    let geometries =
        AccelerationStructureGeometries::Instances(AccelerationStructureGeometryInstancesData {
            flags: GeometryFlags::OPAQUE,
            ..AccelerationStructureGeometryInstancesData::new(
                AccelerationStructureGeometryInstancesDataType::Values(Some(values)),
            )
        });

    let build_info = AccelerationStructureBuildGeometryInfo {
        flags: BuildAccelerationStructureFlags::PREFER_FAST_TRACE,
        mode: BuildAccelerationStructureMode::Build,
        ..AccelerationStructureBuildGeometryInfo::new(geometries)
    };

    let build_range_infos = [AccelerationStructureBuildRangeInfo {
        primitive_count: bottom_level_acceleration_structures.len() as _,
        primitive_offset: 0,
        first_vertex: 0,
        transform_offset: 0,
    }];

    build_acceleration_structure(
        builder,
        memory_allocator,
        AccelerationStructureType::TopLevel,
        build_info,
        &[bottom_level_acceleration_structures.len() as u32],
        build_range_infos,
    )
}

fn create_bottom_level_acceleration_structure_aabb(
    builder: &mut AutoCommandBufferBuilder<PrimaryAutoCommandBuffer>,
    memory_allocator: Arc<dyn MemoryAllocator>,
    aabb_buffer: &Subbuffer<[AabbPositions]>,
) -> Arc<AccelerationStructure> {
    let primitive_count = aabb_buffer.len() as u32;
    let aabbs = AccelerationStructureGeometryAabbsData {
        flags: GeometryFlags::OPAQUE,
        data: Some(aabb_buffer.clone().into_bytes()),
        stride: std::mem::size_of::<AabbPositions>() as u32,
        ..AccelerationStructureGeometryAabbsData::default()
    };
    let build_range_info = AccelerationStructureBuildRangeInfo {
        primitive_count,
        primitive_offset: 0,
        first_vertex: 0,
        transform_offset: 0,
    };

    let build_info = AccelerationStructureBuildGeometryInfo {
        flags: BuildAccelerationStructureFlags::PREFER_FAST_TRACE,
        mode: BuildAccelerationStructureMode::Build,
        ..AccelerationStructureBuildGeometryInfo::new(AccelerationStructureGeometries::Aabbs(vec![
            aabbs,
        ]))
    };

    build_acceleration_structure(
        builder,
        memory_allocator,
        AccelerationStructureType::BottomLevel,
        build_info,
        &[primitive_count],
        [build_range_info],
    )
}

fn create_acceleration_structure(
    memory_allocator: Arc<dyn MemoryAllocator>,
    ty: AccelerationStructureType,
    size: DeviceSize,
) -> Arc<AccelerationStructure> {
    let buffer = Buffer::new_slice::<u8>(
        memory_allocator.clone(),
        BufferCreateInfo {
            usage: BufferUsage::ACCELERATION_STRUCTURE_STORAGE | BufferUsage::SHADER_DEVICE_ADDRESS,
            ..Default::default()
        },
        AllocationCreateInfo {
            memory_type_filter: MemoryTypeFilter::PREFER_DEVICE,
            ..Default::default()
        },
        size,
    )
    .unwrap();

    unsafe {
        AccelerationStructure::new(
            memory_allocator.device().clone(),
            AccelerationStructureCreateInfo {
                ty,
                ..AccelerationStructureCreateInfo::new(buffer)
            },
        )
        .unwrap()
    }
}

fn create_scratch_buffer(
    memory_allocator: Arc<dyn MemoryAllocator>,
    size: DeviceSize,
) -> Subbuffer<[u8]> {
    let alignment_requirement = memory_allocator
        .device()
        .physical_device()
        .properties()
        .min_acceleration_structure_scratch_offset_alignment
        .unwrap() as DeviceSize;

    let subbuffer = Buffer::new_slice::<u8>(
        memory_allocator,
        BufferCreateInfo {
            usage: BufferUsage::STORAGE_BUFFER | BufferUsage::SHADER_DEVICE_ADDRESS,
            ..Default::default()
        },
        AllocationCreateInfo {
            memory_type_filter: MemoryTypeFilter::PREFER_DEVICE,
            ..Default::default()
        },
        size + alignment_requirement,
    )
    .unwrap();

    // get the next aligned offset
    let subbuffer_address: DeviceSize = subbuffer.device_address().unwrap().into();
    let aligned_offset = alignment_requirement - (subbuffer_address % alignment_requirement);

    // slice the buffer to the aligned offset
    let subbuffer2 = subbuffer.slice(aligned_offset..(aligned_offset + size));
    assert!(u64::from(subbuffer2.device_address().unwrap()) % alignment_requirement == 0);
    assert!(subbuffer2.size() == size);

    return subbuffer2;
}

// SAFETY: If build_info.geometries is AccelerationStructureGeometries::Triangles, then the data in
// build_info.geometries.triangles.vertex_data must be valid for the duration of the use of the returned
// acceleration structure.
fn build_acceleration_structure(
    builder: &mut AutoCommandBufferBuilder<PrimaryAutoCommandBuffer>,
    memory_allocator: Arc<dyn MemoryAllocator>,
    ty: AccelerationStructureType,
    mut build_info: AccelerationStructureBuildGeometryInfo,
    max_primitive_counts: &[u32],
    build_range_infos: impl IntoIterator<Item = AccelerationStructureBuildRangeInfo>,
) -> Arc<AccelerationStructure> {
    let device = memory_allocator.device();

    let AccelerationStructureBuildSizesInfo {
        acceleration_structure_size,
        build_scratch_size,
        ..
    } = device
        .acceleration_structure_build_sizes(
            AccelerationStructureBuildType::Device,
            &build_info,
            max_primitive_counts,
        )
        .unwrap();

    let acceleration_structure =
        create_acceleration_structure(memory_allocator.clone(), ty, acceleration_structure_size);
    let scratch_buffer = create_scratch_buffer(memory_allocator.clone(), build_scratch_size);

    build_info.dst_acceleration_structure = Some(acceleration_structure.clone());
    build_info.scratch_data = Some(scratch_buffer);

    unsafe {
        builder
            .build_acceleration_structure(build_info, build_range_infos.into_iter().collect())
            .unwrap();
    }

    acceleration_structure
}
