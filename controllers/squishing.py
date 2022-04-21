import numpy as np
import random
from pathlib import Path
from json import loads
from typing import List, Dict, Tuple
from tdw.tdw_utils import TDWUtils
from tdw.librarian import ModelLibrarian
from tdw_physics.flex_dataset import FlexDataset
from tdw_physics.util import get_args


class Squishing(FlexDataset):
    """
    "Squish" Flex cloth-body objects. All objects are Flex primitives. Each trial is one of the following scenarios:

    1. An object is dropped on the floor.
    2. An solid object is dropped onto a squishable object.
    3. An object is thrown at a wall.
    4. An object is pushed along the floor into another object.
    """

    def __init__(self, port: int = 1071):
        super().__init__(port=port)
        self.model_librarian = ModelLibrarian("models_flex.json")

        # A list of functions that will return commands to initialize a trial.
        self.scenarios = [self.drop_onto_object]
        #self.scenarios = [self.drop_onto_floor, self.drop_onto_object, self.throw_into_wall, self.push_into_other]

        # Size of the room bounds (used for placing objects near walls).
        self.env_bounds = {"x": 4, "y": 0, "z": 2}

        # Cached pressure parameters per model.
        self.pressures = loads(Path("squish_pressures.json").read_text())
        # Only use records in the pressures dictionary.
        self.records = [r for r in self.model_librarian.records if r.name in self.pressures]
        self.zone_id = 0
        self.target_id = 0
        self.material_types = ['Fabric', 'Glass', 'Leather', 'Metal'] #, 'Rubber']
    def get_scene_initialization_commands(self) -> List[dict]:
        return [self.get_add_scene(scene_name="box_room_2018"),
                {"$type": "set_aperture",
                 "aperture": 4.8},
                {"$type": "set_post_exposure",
                 "post_exposure": 0.4},
                {"$type": "set_ambient_occlusion_intensity",
                 "intensity": 0.175},
                {"$type": "set_ambient_occlusion_thickness_modifier",
                 "thickness": 3.5},
                {"$type": "set_shadow_strength",
                 "strength": 1.0},
                {"$type": "create_flex_container",
                 "collision_distance": 0.075,
                 "static_friction": 0.1,
                 "dynamic_friction": 0.1,
                 "particle_friction": 0.1,
                 "iteration_count": 5,
                 "substep_count": 2,
                 "radius": 0.225,
                 "damping": 0,
                 "solid_rest": 0.15,
                 "fluid_rest": 0.1425,
                 "surface_tension": 0.01,
                 "drag": 0}]

    def get_trial_initialization_commands(self) -> List[dict]:
        super().get_trial_initialization_commands()
        # Select a random scenario.
        return random.choice(self.scenarios)()

    def get_field_of_view(self) -> float:
        return 68

    def get_per_frame_commands(self, resp: List[bytes], frame: int) -> List[dict]:
        return []

    def is_done(self, resp: List[bytes], frame: int) -> bool:
        return frame >= 200

    def drop_onto_floor(self) -> List[dict]:
        """
        :return: A list of commands to drop a single object onto the floor.
        """

        # Add the object.
        o_pos = {"x": random.uniform(-0.2, 0.2),
                 "y": random.uniform(1.5, 3.5),
                 "z": random.uniform(-0.2, 0.2)}
        commands, soft_id = self._get_squishable(position=o_pos,
                                                 rotation={"x": random.uniform(0, 360),
                                                           "y": random.uniform(0, 360),
                                                           "z": random.uniform(0, 360)})
        commands.extend(self._get_drop_camera(o_pos))
        commands.append(self._get_drop_force(soft_id))
        return commands

    def drop_onto_object(self) -> List[dict]:
        """
        :return: A list of commands to drop one object onto another.
        """

        # Add a solid object.
        o_pos = {"x": random.uniform(-0.2, 0.2),
                 "y": random.uniform(1.5, 3.5),
                 "z": random.uniform(-0.2, 0.2)}
        solid_id = self.get_unique_id()
        # Add the soft-body (squishable) object.
        import ipdb; ipdb.set_trace()
        solid_record = random.choice([f for f in self.records if f.name=="cube"])
        commands = self.add_solid_object(record=solid_record,
                                         o_id=solid_id,
                                         position=o_pos,
                                         rotation={"x": random.uniform(0, 360),
                                                   "y": random.uniform(0, 360),
                                                   "z": random.uniform(0, 360)},
                                         mass_scale=random.uniform(4, 8))

        # Set the color.
        # Add a small downward force.
        commands.append({"$type": "set_color",
                         "color": {"r": random.random(), "g": random.random(), "b": random.random(), "a": 1.0},
                         "id": solid_id})
        commands.append(self._get_drop_force(solid_id))
        commands.extend(self._get_drop_camera(o_pos))

        # Add a second object on the floor.
        second_object_commands, s_id = self._get_squishable(position={"x": o_pos["x"] + random.uniform(-0.125, 0.125),
                                                                      "y": 0.1,
                                                                      "z": o_pos["z"] + random.uniform(-0.125, 0.125)},
                                                            rotation={"x": 0,
                                                                      "y": random.uniform(0, 360),
                                                                      "z": 0})
        commands.extend(second_object_commands)
        return commands

    def throw_into_wall(self) -> List[dict]:
        """
        :return: A list of commands to push an object into a wall.
        """

        # Pick a random position close to a wall and a random target position.
        if random.random() < 0.5:
            if random.random() < 0.5:
                x = self.env_bounds["x"]
                tx = x + 1
            else:
                x = -self.env_bounds["x"]
                tx = x - 1
            z = random.uniform(-self.env_bounds["z"], self.env_bounds["z"])
            tz = 0
        else:
            if random.random() < 0.5:
                z = self.env_bounds["z"]
                tz = z + 1
            else:
                z = -self.env_bounds["z"]
                tz = z - 1
            x = random.uniform(-self.env_bounds["x"], self.env_bounds["x"])
            tx = 0
        x += random.uniform(-0.5, 0.5)
        z += random.uniform(-0.5, 0.5)
        tx += random.uniform(-0.125, 0.125)
        tz += random.uniform(-0.125, 0.125)

        # Set the y values above the floor so that the object is "thrown" rather than "shoved"
        p0 = {"x": x, "y": 0, "z": z}
        p1 = {"x": tx, "y": random.uniform(0.4, 0.6), "z": tz}

        # Push the object.
        commands = self._push(position=p0, target=p1, force_mag=random.uniform(2000, 3000))

        # Set the avatar.
        commands.extend(self._set_avatar(a_pos=self.get_random_avatar_position(radius_min=0.1,
                                                                               radius_max=0.5,
                                                                               y_min=1,
                                                                               y_max=1.5,
                                                                               center=TDWUtils.VECTOR3_ZERO),
                                         cam_aim=p1))
        return commands

    def push_into_other(self) -> List[dict]:
        """
        :return: A list of commands to push one object into another object.
        """

        # Get two points some distance from each other.
        p0 = np.array([0, 0, 0])
        p1 = np.array([0, 0, 0])
        center = np.array([0, 0, 0])
        count = 0
        while count < 1000 and np.linalg.norm(np.abs(p1 - p0)) < 1.5:
            p0 = TDWUtils.get_random_point_in_circle(center=center, radius=2)
            p1 = TDWUtils.get_random_point_in_circle(center=center, radius=2)
            count += 1
        p0 = TDWUtils.array_to_vector3(p0)
        p0["y"] = 0.2
        p1 = TDWUtils.array_to_vector3(p1)
        p1["y"] = 0.1
        # Get commands to create the first object and push it.
        commands = self._push(position=p0, target=p1, force_mag=random.uniform(1000, 2000))
        # Add commands for the second object.
        second_object_commands, s_id = self._get_squishable(position=p1,
                                                            rotation={"x": 0,
                                                                      "y": random.uniform(0, 360),
                                                                      "z": 0})
        commands.extend(second_object_commands)



        # Set the avatar.
        # Get the midpoint between the objects and use this to place the avatar.
        p_med = TDWUtils.array_to_vector3(np.array([(p0["x"] + p1["x"]) / 2, 0, (p0["z"] + p1["z"]) / 2]))
        commands.extend(self._set_avatar(a_pos=self.get_random_avatar_position(radius_min=1.8,
                                                                               radius_max=2.3,
                                                                               y_min=1,
                                                                               y_max=1.75,
                                                                               center=p_med),
                                         cam_aim=p1))
        return commands

    def _get_drop_camera(self, o_pos: Dict[str, float]) -> List[dict]:
        """
        :param o_pos: The position of the object being dropped.

        :return: A list of commands to set the position of a camera when dropping an object onto the floor.
        """

        # Teleport to the avatar to a random position using o_pos as a centerpoint.
        return self._set_avatar(a_pos=self.get_random_avatar_position(radius_min=1.8,
                                                                      radius_max=2.3,
                                                                      y_min=1.5,
                                                                      y_max=2,
                                                                      center={"x": o_pos["x"],
                                                                              "y": 0,
                                                                              "z": o_pos["z"]}),
                                cam_aim={"x": 0, "y": 0.125, "z": 0})

    @staticmethod
    def _get_drop_force(o_id: int) -> dict:
        """
        Get a command for applying a small force to an object being dropped on the floor.

        :param o_id: The object ID.

        :return: An apply_force_to_flex_object command.
        """

        # Add a small downward force.
        return {"$type": "apply_force_to_flex_object",
                "force": {"x": random.uniform(-100, 100),
                          "y": random.uniform(0, -500),
                          "z": random.uniform(-100, 100)},
                "id": o_id}

    def _push(self, position: Dict[str, float], target: Dict[str, float], force_mag: float) -> List[dict]:
        """
        :param position: The initial position.
        :param target: The target position (used for the force vector).
        :param force_mag: The force magnitutde.

        :return: A list of commands to push the object along the floor.
        """

        commands, soft_id = self._get_squishable(position=position,
                                                 rotation={"x": 0,
                                                           "y": random.uniform(0, 360),
                                                           "z": 0})
        # Get a force vector towards the target.
        p0 = TDWUtils.vector3_to_array(position)
        p1 = TDWUtils.vector3_to_array(target)
        force = ((p1 - p0) / (np.linalg.norm(p1 - p0))) * force_mag
        # Add the force.
        commands.append({"$type": "apply_force_to_flex_object",
                         "force": TDWUtils.array_to_vector3(force),
                         "id": soft_id})

        return commands

    def _get_squishable(self, position: Dict[str, float], rotation: [str, float]) -> \
            Tuple[List[dict], int]:
        """
        Get a "squishable" soft-body object. The object is always a random Flex primitive with a random color.

        :param position: The initial position of the object.
        :param rotation: The initial rotation of the object.

        :return: A list of commands to create the object, and the object ID.
        """

        commands = []
        soft_id = self.get_unique_id()
        # Add the soft-body (squishable) object.
        record = random.choice(self.records)
        pressures = self.pressures[record.name]
        commands.extend(self.add_cloth_object(record=record,
                                              o_id=soft_id,
                                              position=position,
                                              rotation=rotation,
                                              scale={"x": 1, "y": 1, "z": 1},
<<<<<<< HEAD
                                              stretch_stiffness=1,
                                              bend_stiffness=1,
                                              draw_particles = False,
                                              pressure=random.uniform(pressures[0], pressures[1])))

        # set material and color
        # ['Ceramic', 'Concrete', 'Fabric', 'Glass', 'Leather', 'Metal', 'Organic', 'Paper', 'Plastic, 'Rubber','Stone', 'Wood']
        material = self.get_material_name(None, print_info=True)
        commands.extend(
            self.get_object_material_commands(
                record, soft_id, self.get_material_name(None)))
=======
                                              stretch_stiffness=0.5,
                                              bend_stiffness=0.5,
                                              pressure=random.uniform(pressures[0], pressures[1])))

        # commands.extend(self.add_soft_object(record=record,
        #                                       o_id=soft_id,
        #                                       position=position,
        #                                       rotation=rotation,
        #                                       cluster_stiffness = 0.5,
        #                                       scale={"x": 1, "y": 1, "z": 1},
        #                                       link_stiffness=0.5,
        #                                       particle_spacing = 0.035,
        #                                       #bend_stiffness=1,
        #                                       #pressure=random.uniform(pressures[0], pressures[1])
        #                                       ))

        # def add_soft_object(self, record: ModelRecord, position: Dict[str, float], rotation: Dict[str, float],
        #                     scale: Dict[str, float] = None, volume_sampling: float = 2, surface_sampling: float = 0,
        #                     cluster_spacing: float = 0.2, cluster_radius: float = 0.2, cluster_stiffness: float = 0.2,
        #                     link_radius: float = 0.1, link_stiffness: float = 0.5, particle_spacing: float = 0.02,
        #                     mass_scale: float = 1, o_id: Optional[int] = None) -> List[dict]:

        #     commands.extend(self.add_flex_soft_object(
        #                           record = self.objrec1,
        #                           position = self.objrec1_position,
        #                           rotation = self.objrec1_rotation,
        #                           mesh_expansion = 0.0,
        #                           particle_spacing = 0.035,
        #                           mass = self.objrec1_mass,
        #                           scale = self.objrec1_scale,
        #                           o_id = self.objrec1_id,
        #                           ))
>>>>>>> 4d6bd0b4e3cbef3ec30ab283464792c2eeb295dd

        commands.append({"$type": "set_color",
                         "color": {"r": random.random(), "g": random.random(), "b": random.random(), "a": 1.0},
                         "id": soft_id})
        return commands, soft_id

    @staticmethod
    def _set_avatar(a_pos: Dict[str, float], cam_aim: Dict[str, float]) -> List[dict]:
        """
        :param a_pos: The avatar position.
        :param cam_aim: The camera aim point.

        :return: A list of commands to teleport the avatar and rotate the sensor container.
        """

        return [{"$type": "teleport_avatar_to",
                 "position": a_pos},
                {"$type": "look_at_position",
                 "position": cam_aim},
                {"$type": "set_focus_distance",
                 "focus_distance": TDWUtils.get_distance(a_pos, cam_aim)},
                {"$type": "rotate_sensor_container_by",
                 "axis": "pitch",
                 "angle": random.uniform(-3, 3)},
                {"$type": "rotate_sensor_container_by",
                 "axis": "yaw",
                 "angle": random.uniform(-3, 3)}]


if __name__ == "__main__":
    args = get_args("squishing")
    Squishing().run(num=args.num, output_dir=args.dir, temp_path=args.temp, width=args.width, height=args.height)
