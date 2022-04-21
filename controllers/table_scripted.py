from abc import ABC
from typing import List, Dict
from operator import add
import random
from tdw.controller import Controller
from tdw.tdw_utils import TDWUtils
from tdw_physics.rigidbodies_dataset import RigidbodiesDataset
from tdw_physics.util import MODEL_LIBRARIES


class _TableScripted(RigidbodiesDataset, ABC):
    """
    A scene with near-photorealism and a pre-scripted dining table setup.
    """

    # Pre-calculated height of the table.
    _TABLE_HEIGHT = 1.9
    # Pre-calculated height of the floor.
    _FLOOR_HEIGHT = 0.98
    # Valid camera positions.
    _CAMERA_POSITIONS = [{"x": -8.1, "y": 2.25, "z": -4.0},
                         {"x": -8.175, "y": 2.25, "z": -6.35},
                         {"x": -12.0, "y": 3, "z": -4.5},
                         {"x": -12.75, "y": 2.25, "z": -6.6},
                         {"x": -8.1, "y": 2.5, "z": -6.0},
                         {"x": -11.0, "y": 3.65, "z": -5.8}]
    _TABLE_POSITION = {"x": -10.8, "y": 1.0, "z": -5.5}

    def __init__(self, port: int = 1071):
        super().__init__(port=port)

        self._table_id = 0

    def get_scene_initialization_commands(self) -> List[dict]:
        return [self.get_add_scene("archviz_house"),
                {"$type": "set_aperture",
                 "aperture": 2.6},
                {"$type": "set_focus_distance",
                 "focus_distance": 2.25},
                {"$type": "set_post_exposure",
                 "post_exposure": 0.4},
                {"$type": "set_ambient_occlusion_intensity",
                 "intensity": 0.175},
                {"$type": "set_ambient_occlusion_thickness_modifier",
                 "thickness": 3.5},
                {"$type": "set_shadow_strength",
                 "strength": 1.0}]

    def get_field_of_view(self) -> float:
        return 68

    def get_trial_initialization_commands(self) -> List[dict]:
        self._table_id = Controller.get_unique_id()
        # Teleport the avatar.
        commands = [{"$type": "teleport_avatar_to",
                     "position": random.choice(_TableScripted._CAMERA_POSITIONS)},
                    {"$type": "look_at_position",
                     "position": {"x": -10.8, "y": _TableScripted._TABLE_HEIGHT, "z": -5.5}}]
        # Add the table.
        commands.extend(self.add_physics_object_default(name="quatre_dining_table",
                                                        position=self._TABLE_POSITION,
                                                        rotation={"x": 0, "y": -90, "z": 0},
                                                        o_id=self._table_id))
        x0 = -9.35
        x1 = -9.9
        x2 = -10.15
        x3 = -10.8
        x4 = -11.55
        x5 = -11.8
        x6 = -12.275

        z0 = -4.5
        z1 = -5.05
        z2 = -5.45
        z3 = -5.95
        z4 = -6.475

        # Add chairs.
        for x, z, rot in zip([x2, x3, x4, x6, x4, x3, x2, x0],
                             [z0, z0, z0, z2, z4, z4, z4, z2],
                             [180, 180, 180, 90, 0, 0, 0, -90]):
            commands.extend(self.add_physics_object_default(name="brown_leather_dining_chair",
                                                            position={"x": x,
                                                                      "y": _TableScripted._FLOOR_HEIGHT,
                                                                      "z": z},
                                                            rotation={"x": 0, "y": rot, "z": 0}))

        # Add plates.
        plates_x = [x2, x3, x4, x5, x4, x3, x2, x1]
        plates_z = [z1, z1, z1, z2, z3, z3, z3, z2]
        for x, z in zip(plates_x, plates_z):
            commands.extend(self.add_physics_object_default(name="plate05",
                                                            position={"x": x,
                                                                      "y": _TableScripted._TABLE_HEIGHT,
                                                                      "z": z},
                                                            rotation=TDWUtils.VECTOR3_ZERO))
        # Cutlery offset from plate (can be + or -).
        cutlery_off = 0.1
        cutlery_rots = [180, 180, 180, 90, 0, 0, 0, -90]

        # Forks.
        fork_x_offsets = [cutlery_off, cutlery_off, cutlery_off, 0, -cutlery_off, -cutlery_off, -cutlery_off, 0]
        fork_z_offsets = [0, 0, 0, cutlery_off, 0, 0, 0, -cutlery_off]
        forks_x = list(map(add, plates_x, fork_x_offsets))
        forks_z = list(map(add, plates_z, fork_z_offsets))
        for x, z, rot in zip(forks_x, forks_z, cutlery_rots):
            commands.extend(self.add_physics_object_default(name="vk0010_dinner_fork_subd0",
                                                            position={"x": x,
                                                                      "y": _TableScripted._TABLE_HEIGHT,
                                                                      "z": z},
                                                            rotation={"x": 0, "y": rot, "z": 0}))
        # Knives.
        knife_x_offsets = [-cutlery_off, -cutlery_off, -cutlery_off, 0, cutlery_off, cutlery_off, cutlery_off, 0]
        knife_z_offsets = [0, 0, 0, -cutlery_off, 0, 0, 0, cutlery_off]
        knives_x = list(map(add, plates_x, knife_x_offsets))
        knives_z = list(map(add, plates_z, knife_z_offsets))
        for x, z, rot in zip(knives_x, knives_z, cutlery_rots):
            commands.extend(self.add_physics_object_default(name="vk0007_steak_knife",
                                                            position={"x": x,
                                                                      "y": _TableScripted._TABLE_HEIGHT,
                                                                      "z": z},
                                                            rotation={"x": 0, "y": rot, "z": 0}))

        incidental_names = ["moet_chandon_bottle_vray", "peppermill", "salt_shaker", "coffeemug", "coffeecup004",
                            "bowl_wood_a_01", "glass1", "glass2", "glass3"]
        incidental_positions = [{"x": -10.35, "y": _TableScripted._TABLE_HEIGHT, "z": -5.325},
                                {"x": -10.175, "y": _TableScripted._TABLE_HEIGHT, "z": -5.635},
                                {"x": -11.15, "y": _TableScripted._TABLE_HEIGHT, "z": -5.85},
                                {"x": -11.525, "y": _TableScripted._TABLE_HEIGHT, "z": -5.625},
                                {"x": -11.25, "y": _TableScripted._TABLE_HEIGHT, "z": -5.185},
                                {"x": -10.5, "y": _TableScripted._TABLE_HEIGHT, "z": -5.05}]
        random.shuffle(incidental_names)
        random.shuffle(incidental_positions)
        # Select 4 incidental objects.
        for i in range(4):
            name = incidental_names.pop(0)
            o_id = Controller.get_unique_id()
            commands.extend(self.add_physics_object_default(name=name,
                                                            position=incidental_positions.pop(0),
                                                            rotation=TDWUtils.VECTOR3_ZERO,
                                                            o_id=o_id))
            # These objects need further scaling.
            if name in ["salt_shaker", "peppermill"]:
                commands.append({"$type": "scale_object",
                                 "id": o_id,
                                 "scale_factor": {"x": 0.254, "y": 0.254, "z": 0.254}})

        # Select 2 bread objects.
        bread_names = ["bread", "bread_01", "bread_02", "bread_03"]
        bread_positions = [{"x": x2, "y": _TableScripted._TABLE_HEIGHT, "z": z2},
                           {"x": x4, "y": _TableScripted._TABLE_HEIGHT, "z": z2},
                           {"x": x3, "y": _TableScripted._TABLE_HEIGHT, "z": z2}]
        random.shuffle(bread_names)
        random.shuffle(bread_positions)
        for i in range(2):
            commands.extend(self.add_physics_object_default(name=bread_names.pop(0),
                                                            position=bread_positions.pop(0),
                                                            rotation=TDWUtils.VECTOR3_ZERO))
        return commands


class TableScriptedTilt(_TableScripted):
    """
    Tilt a table in a pre-scripted room.
    """

    def __init__(self, port: int = 1071):
        super().__init__(port=port)

        self._tip_table_frames = 0
        self._tip_table_force = 0

        table_record = MODEL_LIBRARIES["models_full.json"].get_record("quatre_dining_table")

        self._tip_positions = [table_record.bounds["front"],
                               table_record.bounds["back"],
                               table_record.bounds["left"],
                               table_record.bounds["right"]]
        # Set the y value of each tip position to the floor height and offset the x,z values by the table position.
        for tip_position in self._tip_positions:
            tip_position["y"] = self._FLOOR_HEIGHT
            tip_position["x"] += self._TABLE_POSITION["x"]
            tip_position["z"] += self._TABLE_POSITION["z"]
        self._tip_pos: Dict[str, float] = {}

    def is_done(self, resp: List[bytes], frame: int) -> bool:
        return frame >= 300

    def get_trial_initialization_commands(self) -> List[dict]:
        # Set the tip force per frame and how long the table will be tipped.
        self._tip_pos = random.choice(self._tip_positions)
        if self._tip_pos == self._tip_positions[2] or self._tip_pos == self._tip_positions[3]:
            self._tip_table_frames = 37
            self._tip_table_force = random.uniform(30, 35)
        else:
            self._tip_table_frames = 27
            self._tip_table_force = random.uniform(23, 25)

        return super().get_trial_initialization_commands()

    def get_per_frame_commands(self, resp: List[bytes], frame: int) -> List[dict]:
        # Tip the table up.
        if frame < self._tip_table_frames:
            return [{"$type": "apply_force_at_position",
                     "id": self._table_id,
                     "position": self._tip_pos,
                     "force": {"x": 0, "y": self._tip_table_force, "z": 0}}]
        # Make the table kinematic to allow it to hang in the air.
        # Set the detection mode to continuous speculative in order to continue to detect collisions.
        elif frame == self._tip_table_frames:
            return [{"$type": "set_object_collision_detection_mode",
                     "id": self._table_id,
                     "mode": "continuous_speculative"},
                    {"$type": "set_kinematic_state",
                     "use_gravity": False,
                     "is_kinematic": True,
                     "id": self._table_id}]
        else:
            return []


class TableScriptedFalling(_TableScripted):
    """
    Small objects fly up and then fall down.
    """

    def __init__(self, port: int = 1071):
        super().__init__(port=port)
        # Commands to be sent per-frame.
        self._per_frame_commands: List[List[dict]] = []
        self._num_per_frame_commands = 0

    def get_trial_initialization_commands(self) -> List[dict]:
        commands = super().get_trial_initialization_commands()

        del self._per_frame_commands[:]
        self._per_frame_commands = self.get_falling_commands()
        self._num_per_frame_commands = len(self._per_frame_commands)
        return commands

    def get_per_frame_commands(self, resp: List[bytes], frame: int) -> List[dict]:
        if len(self._per_frame_commands) > 0:
            return self._per_frame_commands.pop(0)
        else:
            return []

    def is_done(self, resp: List[bytes], frame: int) -> bool:
        return frame > self._num_per_frame_commands + 300


if __name__ == "__main__":
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument("--dir", type=str, default="D:/table_scripted_tilt", help="Root output directory.")
    parser.add_argument("--num", type=int, default=1500, help="The number of trials in the dataset.")
    parser.add_argument("--temp", type=str, default="D:/temp.hdf5", help="Temp path for incomplete files.")
    parser.add_argument("--scenario", type=str, choices=["tilt", "fall"], default="tilt", help="The type of scenario")
    parser.add_argument("--width", type=int, default=256, help="Screen width in pixels.")
    parser.add_argument("--height", type=int, default=256, help="Screen width in pixels.")

    args = parser.parse_args()
    if args.scenario == "tilt":
        c = TableScriptedTilt()
    elif args.scenario == "fall":
        c = TableScriptedFalling()
    else:
        raise Exception(f"Scenario not defined: {args.scenario}")
    c.run(num=args.num, output_dir=args.dir, temp_path=args.temp, width=args.width, height=args.height)
