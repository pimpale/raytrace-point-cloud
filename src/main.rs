use std::path;
use std::sync::Arc;

use game_system::game_world::{EntityCreationData, GameWorld};
use nalgebra::{Isometry3, Matrix3, Point3, UnitQuaternion, Vector3};

use statrs::distribution::{ContinuousCDF, Normal};
use utils::PointCloudPoint;
use vulkano::acceleration_structure::AabbPositions;
use vulkano::command_buffer::allocator::StandardCommandBufferAllocator;
use vulkano::descriptor_set::allocator::StandardDescriptorSetAllocator;
use vulkano::swapchain::Surface;

use vulkano::instance::{Instance, InstanceCreateFlags, InstanceCreateInfo};
use vulkano::memory::allocator::StandardMemoryAllocator;

use vulkano::VulkanLibrary;
use winit::event_loop::{ControlFlow, EventLoop};

use winit::event::{Event, WindowEvent};
use winit::window::WindowBuilder;

use ply_rs;
use rayon::prelude::*;

use crate::render_system::vertex::GaussianSplat;

mod camera;
mod game_system;
mod handle_user_input;
mod render_system;
mod utils;

fn build_scene(
    general_queue: Arc<vulkano::device::Queue>,
    transfer_queue: Arc<vulkano::device::Queue>,
    command_buffer_allocator: Arc<StandardCommandBufferAllocator>,
    memory_allocator: Arc<StandardMemoryAllocator>,
    descriptor_set_allocator: Arc<StandardDescriptorSetAllocator>,
    surface: Arc<Surface>,
) -> GameWorld {
    let rd: Vec<Point3<f32>> = vec![
        [0.0, 0.0, 0.0].into(),
        [1.0, 0.0, 0.0].into(),
        [2.0, 0.0, 0.0].into(),
        [3.0, 0.0, 0.0].into(),
        [4.0, 0.0, 0.0].into(),
        [5.0, 0.0, 0.0].into(),
        [6.0, 0.0, 0.0].into(),
        [7.0, 0.0, 0.0].into(),
        [8.0, 0.0, 0.0].into(),
        [9.0, 0.0, 0.0].into(),
        [10.0, 0.0, 0.0].into(),
        [11.0, 0.0, 0.0].into(),
        [12.0, 0.0, 0.0].into(),
        [13.0, 0.0, 0.0].into(),
        [14.0, 0.0, 0.0].into(),
        [15.0, 0.0, 0.0].into(),
        [15.0, 0.0, 1.0].into(),
        [15.0, 0.0, 2.0].into(),
        [15.0, 0.0, 3.0].into(),
        [15.0, 0.0, 4.0].into(),
        [15.0, 0.0, 5.0].into(),
        [15.0, 0.0, 6.0].into(),
        [15.0, 0.0, 7.0].into(),
        [15.0, 0.0, 8.0].into(),
        [15.0, 0.0, 9.0].into(),
        [15.0, 0.0, 10.0].into(),
        [15.0, 0.0, 11.0].into(),
        [15.0, 0.0, 12.0].into(),
        [15.0, 0.0, 13.0].into(),
        [15.0, 0.0, 14.0].into(),
        [15.0, 0.0, 15.0].into(),
    ];

    let g: Vec<Point3<f32>> = vec![[0.0, -0.1, -50.0].into(), [0.0, -0.1, 50.0].into()];

    let mut world = GameWorld::new(
        general_queue,
        transfer_queue,
        command_buffer_allocator,
        memory_allocator,
        descriptor_set_allocator,
        0,
        surface,
        Box::new(camera::SphericalCamera::new()),
    );

    // add ego agent
    let ego_mesh = utils::unitcube();
    world.add_entity(
        0,
        EntityCreationData {
            mesh: vec![(
                AabbPositions {
                    min: [-0.5, -0.5, -0.5],
                    max: [0.5, 0.5, 0.5],
                },
                GaussianSplat::new(
                    UnitQuaternion::identity().coords.into(),
                    [1.0, 1.0, 1.0],
                    [0.5, 0.5, 0.5],
                    1.0,
                ),
            )],
            isometry: Isometry3::translation(0.0, 0.0, 0.0),
        },
    );

    // // add road
    // world.add_entity(
    //     1,
    //     EntityCreationData {
    //         mesh: utils::flat_polyline(rd.clone(), 1.0, [0.5, 0.5, 0.5]),
    //         isometry: Isometry3::identity(),
    //     },
    // );

    // // add road yellow line
    // world.add_entity(
    //     2,
    //     EntityCreationData {
    //         mesh: utils::flat_polyline(
    //             rd.iter().map(|v| v + Vector3::new(0.0, 0.1, 0.0)).collect(),
    //             0.1,
    //             [1.0, 1.0, 0.0],
    //         ),
    //         isometry: Isometry3::identity(),
    //     },
    // );

    // // add ground
    // let ground_mesh = utils::flat_polyline(g.clone(), 50.0, [0.5, 1.0, 0.5]);
    // world.add_entity(
    //     3,
    //     EntityCreationData {
    //         mesh: ground_mesh,
    //         isometry: Isometry3::identity(),
    //     },
    // );

    // add teapot
    let path = path::Path::new("./assets/point_cloud.ply");
    let parser = ply_rs::parser::Parser::<utils::PointCloudPoint>::new();
    let point_cloud = parser
        .read_ply(&mut std::fs::File::open(path).unwrap())
        .unwrap();

    let vertexes = point_cloud.payload["vertex"]
        .par_iter()
        .cloned()
        .map(
            |PointCloudPoint {
                 scale,
                 position,
                 rot,
                 color,
                 opacity,
             }| {
                let r: Matrix3<f32> = UnitQuaternion::new_normalize(rot).into();
                let s = Matrix3::from_diagonal(&scale);

                let sigma = r * s * s.transpose() * r.transpose();

                let threshold = 0.1;

                let mut min = [0.0, 0.0, 0.0];
                let mut max = [1.0, 1.0, 1.0];
                for dim in 0..3 {
                    min[dim] = Normal::new(position[dim] as f64, sigma[(dim, dim)].sqrt() as f64)
                        .unwrap()
                        .inverse_cdf(threshold) as f32;
                    max[dim] = Normal::new(position[dim] as f64, sigma[(dim, dim)].sqrt() as f64)
                        .unwrap()
                        .inverse_cdf(1.0 - threshold) as f32;
                }

                (
                    AabbPositions { min, max },
                    GaussianSplat::new(rot.coords.into(), scale.into(), color, opacity),
                )
            },
        )
        .collect::<Vec<_>>();

    dbg!(vertexes.len());

    world.add_entity(
        4,
        EntityCreationData {
            mesh: vertexes,
            isometry: Isometry3::identity(),
        },
    );

    world
}

fn main() {
    let library = VulkanLibrary::new().unwrap();
    let event_loop = EventLoop::new();
    let required_extensions = Surface::required_extensions(&event_loop);

    let instance = Instance::new(
        library,
        InstanceCreateInfo {
            flags: InstanceCreateFlags::ENUMERATE_PORTABILITY,
            enabled_extensions: required_extensions,
            ..Default::default()
        },
    )
    .unwrap();

    let window = Arc::new(WindowBuilder::new().build(&event_loop).unwrap());

    let surface = Surface::from_window(instance.clone(), window).unwrap();

    let (device, general_queue, transfer_queue) =
        render_system::interactive_rendering::get_device_for_rendering_on(
            instance.clone(),
            surface.clone(),
        );

    //Print some info about the device currently being used
    println!(
        "Using device: {} (type: {:?})",
        device.physical_device().properties().device_name,
        device.physical_device().properties().device_type
    );

    let memory_allocator = Arc::new(StandardMemoryAllocator::new_default(device.clone()));
    let command_buffer_allocator = Arc::new(StandardCommandBufferAllocator::new(
        device.clone(),
        Default::default(),
    ));
    let descriptor_set_allocator = Arc::new(StandardDescriptorSetAllocator::new(
        device.clone(),
        Default::default(),
    ));

    let mut start_time = std::time::Instant::now();
    let mut frame_count = 0;

    let mut world = build_scene(
        general_queue.clone(),
        transfer_queue.clone(),
        command_buffer_allocator.clone(),
        memory_allocator.clone(),
        descriptor_set_allocator.clone(),
        surface.clone(),
    );

    event_loop.run(move |event, _, control_flow| match event {
        Event::WindowEvent {
            event: WindowEvent::CloseRequested,
            ..
        } => {
            *control_flow = ControlFlow::Exit;
        }
        Event::WindowEvent { event, .. } => {
            world.handle_window_event(event);
        }
        Event::RedrawEventsCleared => {
            // print fps
            frame_count += 1;
            let elapsed = start_time.elapsed();
            if elapsed.as_secs() >= 1 {
                println!("fps: {}", frame_count);
                frame_count = 0;
                start_time = std::time::Instant::now();
            }

            // game step and render
            world.step();
        }
        _ => (),
    });
}
