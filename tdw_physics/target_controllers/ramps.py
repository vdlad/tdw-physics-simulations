from typing import List
from tdw.tdw_utils import TDWUtils
from tdw.librarian import ModelLibrarian
from tdw_physics.rigidbodies_dataset import RigidbodiesDataset
import sys

class Ramp(RigidbodiesDataset):
    def __init__(self, port: int = 1071):
        super().__init__(port=port)
        self.ball_record = ModelLibrarian("models_flex.json").get_record("sphere")
        if sys.argv[2] == '30':
            self.ramp_record = ModelLibrarian("models_full.json").get_record("ramp_with_platform_30")  
        elif sys.argv[2] == '60':
            self.ramp_record = ModelLibrarian("models_full.json").get_record("ramp_with_platform_60")
        self.pin_record = ModelLibrarian("models_flex.json").get_record("cylinder")
        self.ball_material = "marble_white"
        self.start()

    def _write_static_data(self, static_group) -> None:
        pass

    def get_trial_initialization_commands(self) -> List[dict]:
        ball_id = self.get_unique_id()
        commands = []
        # Add the ramp.
        ramp_id = self.get_unique_id()
        commands.extend(self.add_physics_object(record=self.ramp_record,
                                                o_id=ramp_id,
                                                position=TDWUtils.VECTOR3_ZERO,
                                                rotation=TDWUtils.VECTOR3_ZERO,
                                                mass=1000,
                                                bounciness=0.1,
                                                # static_friction=0.1,
                                                # dynamic_friction=0.1
                                                # mass=5,
                                                # bounciness=0,
                                                static_friction=0.1,
                                                dynamic_friction=0.5
                                                ))
        # Make the ramp kinematic.
        commands.append({"$type": "set_kinematic_state",
                         "id": ramp_id,
                         "is_kinematic": True,
                         "use_gravity": False})
        # Add the pin.
        pin_id = self.get_unique_id()
        commands.extend(self.add_physics_object(record=self.pin_record,
                                                position={"x": -1.69, "y": 0, "z": 0},
                                                rotation=TDWUtils.VECTOR3_ZERO,
                                                mass=1,
                                                dynamic_friction=0.5,
                                                static_friction=0.3,
                                                bounciness=0.8,
                                                o_id=pin_id))
        commands.append({"$type": "scale_object",
                          "id": pin_id,
                          "scale_factor": {"x": 0.2, "y": 0.7, "z": 0.2}})
        # Add the ball.
        ball_x = self.ramp_record.bounds['right']['x']
        ball_y = self.ramp_record.bounds['top']['y']
        commands.extend(self.add_physics_object(record=self.ball_record,
                                                # position={"x": 1.29, "y": 1.001, "z": 0},
                                                position={"x": ball_x, "y": ball_y, "z": 0.0},
                                                rotation=TDWUtils.VECTOR3_ZERO,
                                                mass=1,
                                                dynamic_friction=0.1,
                                                static_friction=0.1,
                                                bounciness=0.6,
                                                # dynamic_friction=0.01,
                                                # static_friction=0.01,
                                                # bounciness=0,
                                                o_id=ball_id))
        # Make the ball smaller. Set the ball's visual material. Apply a force. Move the camera.
        commands.extend([{"$type": "scale_object",
                          "id": ball_id,
                          "scale_factor": {"x": 0.3, "y": 0.3, "z": 0.3}},
                         self.get_add_material(material_name=self.ball_material, library="materials_low.json"),
                         {"$type": "set_visual_material",
                          "id": ball_id,
                          "material_name": self.ball_material,
                          "object_name": "sphere",
                          "material_index": 0},
                         {"$type": "set_mass",
                          "mass": 10,
                          "id": ball_id},
                         {"$type": "apply_force_to_object",
                          "id": ball_id,
                          "force": {"x": -3, "y": -1, "z": 0}},
                         {"$type": "teleport_avatar_to",
                          "position": {"x": -4.438, "y": 1.926, "z": -4.244}},
                         {"$type": "look_at_position",
                          "position": TDWUtils.VECTOR3_ZERO}])
        return commands

    def get_per_frame_commands(self, resp: List[bytes], frame: int) -> List[dict]:
        return []

    def is_done(self, resp: List[bytes], frame: int) -> bool:
        return frame > 500

    def get_field_of_view(self) -> float:
        return 55

    def get_scene_initialization_commands(self) -> List[dict]:
        return [{"$type": "load_scene",
                 "scene_name": "ProcGenScene"},
                TDWUtils.create_empty_room(12, 12)]


if __name__ == "__main__":
    import sys
    c = Ramp()
    c.run(num=int(sys.argv[1]), output_dir="/Users/dbear/neuroailab/physics_benchmarking/stimuli/ramp_test0", temp_path="D:/temp.hdf5", width=512, height=512,
          write_passes=["_img"], save_passes=["_img"], save_movies=True)
    c.communicate({"$type": "terminate"})        
