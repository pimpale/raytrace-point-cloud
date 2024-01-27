use std::{cell::RefCell, rc::Rc};

use nalgebra::Vector3;

use crate::{
    camera::{InteractiveCamera, RenderingPreferences},
    game_system::game_world::WorldChange,
    handle_user_input::UserInputState,
};

use super::manager::{Manager, UpdateData};

pub struct EgoControlsManager {
    camera: Rc<RefCell<Box<dyn InteractiveCamera>>>,
    user_input_state: UserInputState,
}

impl EgoControlsManager {
    pub fn new(camera: Rc<RefCell<Box<dyn InteractiveCamera>>>) -> Self {
        Self {
            camera,
            user_input_state: UserInputState::new(),
        }
    }
}

impl Manager for EgoControlsManager {
    fn update<'a>(&mut self, data: UpdateData<'a>) -> Vec<WorldChange> {
        let UpdateData {
            ego_entity_id,
            entities,
            extent,
            window_events,
            ..
        } = data;

        // update user input state
        self.user_input_state.handle_input(window_events);

        let ego = entities.get(&ego_entity_id).unwrap();

        // update camera
        let mut camera = self.camera.borrow_mut();
        camera.set_root_position(ego.isometry.translation.vector.clone().cast().into());
        camera.set_root_rotation(ego.isometry.rotation.clone().cast().into());
        camera.handle_event(extent, window_events);
        if UserInputState::key_pressed(window_events, winit::event::VirtualKeyCode::R) {
            let current_prefs = camera.rendering_preferences();
            let new_samples = match current_prefs.samples {
                1 => 2,
                2 => 4,
                4 => 8,
                8 => 16,
                16 => 32,
                32 => 64,
                _ => 1,
            };
            camera.set_rendering_preferences(RenderingPreferences {
                samples: new_samples,
            });
        }

        let mut changes = vec![];

        let move_magnitude: f32 = 0.1;
        let rotate_magnitude: f32 = 0.1;
        let jump_magnitude: f32 = 0.1;

        let mut target_linvel = Vector3::zeros();
        let mut target_angvel = Vector3::zeros();

        if self.user_input_state.current.w {
            target_linvel += move_magnitude * Vector3::new(1.0, 0.0, 0.0);
        }
        if self.user_input_state.current.s {
            target_linvel += move_magnitude * Vector3::new(-1.0, 0.0, 0.0);
        }

        if self.user_input_state.current.space {
            target_linvel += jump_magnitude * Vector3::new(0.0, 1.0, 0.0);
        };
        if self.user_input_state.current.shift {
            target_linvel += jump_magnitude * Vector3::new(0.0, -1.0, 0.0);
        };

        if self.user_input_state.current.a {
            target_angvel += rotate_magnitude * Vector3::new(0.0, -1.0, 0.0);
        }
        if self.user_input_state.current.d {
            target_angvel += rotate_magnitude * Vector3::new(0.0, 1.0, 0.0);
        }

        let mut new_isometry = ego.isometry.clone();
        new_isometry.translation.vector += target_linvel;
        new_isometry.rotation *=
            nalgebra::UnitQuaternion::from_axis_angle(&nalgebra::Vector3::y_axis(), target_angvel[1]);

        // if kinematic we can directly set the velocity
        changes.push(WorldChange::GlobalEntityUpdateIsometry(ego_entity_id, new_isometry));

        changes
    }
}
