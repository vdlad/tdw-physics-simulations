import sys, os, copy
from typing import List, Dict, Tuple, Optional
from pathlib import Path

import random
import numpy as np

from tdw.librarian import ModelRecord, MaterialLibrarian, ModelLibrarian
from tdw.tdw_utils import TDWUtils
from tdw_physics.target_controllers.dominoes import Dominoes, get_args, ArgumentParser
from tdw_physics.flex_dataset import FlexDataset
from tdw_physics.util import MODEL_LIBRARIES, get_parser, none_or_str
from tdw_physics.rigidbodies_dataset import get_random_xyz_transform

# fluid
from tdw.flex.fluid_types import FluidTypes

def get_flex_args(dataset_dir: str, parse=True):

    common = get_parser(dataset_dir, get_help=False)
    domino, domino_postproc = get_args(dataset_dir, parse=False)
    parser = ArgumentParser(parents=[common, domino], conflict_handler='resolve', fromfile_prefix_chars='@')

    parser.add_argument("--all_flex_objects",
                        type=int,
                        default=1,
                        help="Whether all rigid objects should be FLEX")
    parser.add_argument("--step_physics",
                        type=int,
                        default=100,
                        help="How many physics steps to run forward after adding a solid FLEX object")
    parser.add_argument("--cloth",
                        action="store_true",
                        help="Demo: whether to drop a cloth")
    parser.add_argument("--squishy",
                        action="store_true",
                        help="Demo: whether to drop a squishy ball")
    parser.add_argument("--fluid",
                        action="store_true",
                        help="Demo: whether to drop fluid")
    parser.add_argument("--fwait",
                        type=none_or_str,
                        default="30",
                        help="How many frames to wait before applying the force")


    def postprocess(args):

        args = domino_postproc(args)
        args.all_flex_objects = bool(int(args.all_flex_objects))

        return args

    if not parse:
        return (parser, postproccess)

    args = parser.parse_args()
    args = postprocess(args)

    return args


class FlexDominoes(Dominoes, FlexDataset):

    FLEX_RECORDS = ModelLibrarian(str(Path("flex.json").resolve())).records
    CLOTH_RECORD = MODEL_LIBRARIES["models_special.json"].get_record("cloth_square")
    SOFT_RECORD = MODEL_LIBRARIES["models_flex.json"].get_record("sphere")
    RECEPTACLE_RECORD = MODEL_LIBRARIES["models_special.json"].get_record("fluid_receptacle1x1")
    FLUID_TYPES = FluidTypes()

    def __init__(self, port: int = 1071,
                 all_flex_objects=True,
                 use_cloth=False,
                 use_squishy=False,
                 use_fluid=False,
                 step_physics=False,
                 middle_scale_range=0.5,
                 **kwargs):

        Dominoes.__init__(self, port=port, **kwargs)
        self._clear_flex_data()

        self.all_flex_objects = all_flex_objects
        self._set_add_physics_object()

        self.step_physics = step_physics
        self.use_cloth = use_cloth
        self.use_squishy = use_squishy
        self.use_fluid = use_fluid

        self.middle_scale_range = middle_scale_range
        print("MIDDLE SCALE RANGE", self.middle_scale_range)

        if self.use_fluid:
            self.ft_selection = random.choice(self.FLUID_TYPES.fluid_type_names)

    def _set_add_physics_object(self):
        if self.all_flex_objects:
            self.add_physics_object = self.add_flex_solid_object
            self.add_primitive = self.add_flex_solid_object
        else:
            self.add_physics_object = self.add_rigid_physics_object


    def get_scene_initialization_commands(self) -> List[dict]:

        commands = Dominoes.get_scene_initialization_commands(self)
        commands[0].update({'convexify': True})
        create_container = {
            "$type": "create_flex_container",
            # "collision_distance": 0.001,
            "collision_distance": 0.025,
            # "collision_distance": 0.1,
            "static_friction": 1.0,
            "dynamic_friction": 1.0,
            "radius": 0.1875,
            'max_particles': 50000}
            # 'max_particles': 250000}

        if self.use_fluid:
            create_container.update({
                'viscosity': self.FLUID_TYPES.fluid_types[self.ft_selection].viscosity,
                'adhesion': self.FLUID_TYPES.fluid_types[self.ft_selection].adhesion,
                'cohesion': self.FLUID_TYPES.fluid_types[self.ft_selection].cohesion,
                'fluid_rest': 0.05,
                'damping': 0.01,
                'subsetp_count': 5,
                'iteration_count': 8,
                'buoyancy': 1.0})

        commands.append(create_container)

        if self.use_fluid:
            commands.append({"$type": "set_time_step", "time_step": 0.005})

        return commands

    def get_trial_initialization_commands(self) -> List[dict]:

        # clear the flex data
        FlexDataset.get_trial_initialization_commands(self)
        return Dominoes.get_trial_initialization_commands(self)

    def _get_send_data_commands(self) -> List[dict]:
        commands = Dominoes._get_send_data_commands(self)
        commands.extend(FlexDataset._get_send_data_commands(self))
        return commands

    def add_rigid_physics_object(self, *args, **kwargs):
        """
        Make sure controller knows to treat probe, zone, target, etc. as non-flex objects
        """

        o_id = kwargs.get('o_id', None)
        if o_id is None:
            o_id: int = self.get_unique_id()
            kwargs['o_id'] = o_id

        commands = Dominoes.add_physics_object(self, *args, **kwargs)
        self.non_flex_objects.append(o_id)

        print("Add rigid physics object", o_id)

        return commands

    def add_flex_solid_object(self,
                              record: ModelRecord,
                              position: Dict[str, float],
                              rotation: Dict[str, float],
                              mesh_expansion: float = 0,
                              particle_spacing: float = 0.035,
                              mass: float = 1,
                              scale: Optional[Dict[str, float]] = {"x": 0.1, "y": 0.5, "z": 0.25},
                              material: Optional[str] = None,
                              color: Optional[list] = None,
                              exclude_color: Optional[list] = None,
                              o_id: Optional[int] = None,
                              add_data: Optional[bool] = True,
                              **kwargs) -> List[dict]:

        # so objects don't get stuck in each other -- an unfortunate feature of FLEX
        position = {'x': position['x'], 'y': position['y'] + 0.1, 'z': position['z']}

        commands = FlexDataset.add_solid_object(
            self,
            record = record,
            position = position,
            rotation = rotation,
            scale = scale,
            mesh_expansion = mesh_expansion,
            particle_spacing = particle_spacing,
            mass_scale = 1,
            o_id = o_id)

        # set mass
        commands.append({"$type": "set_flex_object_mass",
                         "mass": mass,
                         "id": o_id})

        # set material and color
        commands.extend(
            self.get_object_material_commands(
                record, o_id, self.get_material_name(material)))

        color = color if color is not None else self.random_color(exclude=exclude_color)
        commands.append(
            {"$type": "set_color",
             "color": {"r": color[0], "g": color[1], "b": color[2], "a": 1.},
             "id": o_id})

        # step physics
        if bool(self.step_physics):
            print("stepping physics forward", self.step_physics)
            commands.append({"$type": "step_physics",
                             "frames": self.step_physics})

        # add data
        print("Add FLEX physics object", o_id)
        if add_data:
            self._add_name_scale_color(record, {'color': color, 'scale': scale, 'id': o_id})
            self.masses = np.append(self.masses, mass)

        return commands

    # def _place_and_push_probe_object(self):
    #     return []

    def _get_push_cmd(self, o_id, position_or_particle=None):
        if not self.all_flex_objects:
            return Dominoes._get_push_cmd(self, o_id, position_or_particle)
        cmd = {"$type": "apply_force_to_flex_object",
               "force": self.push_force,
               "id": o_id,
               "particle": -1}
        print("PUSH CMD FLEX")
        print(cmd)
        return cmd

    def drop_cloth(self) -> List[dict]:

        self.cloth = self.CLOTH_RECORD
        self.cloth_id = self._get_next_object_id()
        self.cloth_position = copy.deepcopy({'x':1.0, 'y':1.5,'z':0.0})
        self.cloth_color = [0.8,0.5,1.0]
        self.cloth_scale = {'x': 1.0, 'y': 1.0, 'z': 1.0}
        self.cloth_mass = 0.5

        commands = self.add_cloth_object(
            record = self.cloth,
            position = self.cloth_position,
            rotation = {k:0 for k in ['x','y','z']},
            scale=self.cloth_scale,
            mass_scale = 1,
            mesh_tesselation = 1,
            tether_stiffness = 1.,
            bend_stiffness = 1.,
            stretch_stiffness = 1.,
            o_id = self.cloth_id)

        # set mass
        commands.append({"$type": "set_flex_object_mass",
                         "mass": self.cloth_mass,
                         "id": self.cloth_id})

        # color cloth
        commands.append(
            {"$type": "set_color",
             "color": {"r": self.cloth_color[0], "g": self.cloth_color[1], "b": self.cloth_color[2], "a": 1.},
             "id": self.cloth_id})

        self._add_name_scale_color(
            self.cloth, {'color': self.cloth_color, 'scale': self.cloth_scale, 'id': self.cloth_id})
        self.masses = np.append(self.masses, self.cloth_mass)

        return commands

    def drop_squishy(self) -> List[dict]:

        self.squishy = self.SOFT_RECORD
        self.squishy_id = self._get_next_object_id()
        self.squishy_position = {'x': 0., 'y': 1.0, 'z': 0.}
        rotation = {k:0 for k in ['x','y','z']}

        self.squishy_color = [0.0,0.8,1.0]
        self.squishy_scale = get_random_xyz_transform(self.middle_scale_range)
        self.squishy_mass = 2.0

        commands = self.add_soft_object(
            record = self.squishy,
            position = self.squishy_position,
            rotation = rotation,
            scale=self.squishy_scale,
            o_id = self.squishy_id)

        # set mass
        commands.append({"$type": "set_flex_object_mass",
                         "mass": self.squishy_mass,
                         "id": self.squishy_id})

        commands.append(
            {"$type": "set_color",
             "color": {"r": self.squishy_color[0], "g": self.squishy_color[1], "b": self.squishy_color[2], "a": 1.},
             "id": self.squishy_id})

        self._add_name_scale_color(
            self.squishy, {'color': self.squishy_color, 'scale': self.squishy_scale, 'id': self.squishy_id})
        self.masses = np.append(self.masses, self.squishy_mass)

        return commands

    def drop_fluid(self) -> List[dict]:

        commands = []

        # create a pool for the fluid
        self.pool_id = self._get_next_object_id()
        print("POOL ID", self.pool_id)
        self.non_flex_objects.append(self.pool_id)
        commands.append(self.add_transforms_object(record=self.RECEPTACLE_RECORD,
                                                   position=TDWUtils.VECTOR3_ZERO,
                                                   rotation=TDWUtils.VECTOR3_ZERO,
                                                   o_id=self.pool_id,
                                                   add_data=True))
        commands.append({"$type": "set_kinematic_state",
                         "id": self.pool_id,
                         "is_kinematic": True,
                         "use_gravity": False})

        # add the fluid; this will also step physics forward 500 times
        self.fluid_id = self._get_next_object_id()
        print("FLUID ID", self.fluid_id)
        commands.extend(self.add_fluid_object(
            position={"x": 0.0, "y": 1.0, "z": 0.0},
            rotation=TDWUtils.VECTOR3_ZERO,
            o_id=self.fluid_id,
            fluid_type=self.ft_selection))
        self.fluid_object_ids.append(self.fluid_id)

        # restore usual time step
        commands.append({"$type": "set_time_step", "time_step": 0.01})

        return commands

    def _place_ramp_under_probe(self) -> List[dict]:

        cmds = Dominoes._place_ramp_under_probe(self)
        self.non_flex_objects.append(self.ramp_id)
        if self.ramp_base_height >= 0.01:
            self.non_flex_objects.append(self.ramp_base_id)
        return cmds

    def _build_intermediate_structure(self) -> List[dict]:

        commands = []
        # step physics
        # if self.all_flex_objects:
        #     commands.append({"$type": "step_physics",
        #                      "frames": 50})

        commands.extend(self.drop_fluid() if self.use_fluid else [])
        commands.extend(self.drop_cloth() if self.use_cloth else [])
        commands.extend(self.drop_squishy() if self.use_squishy else [])

        return commands

if __name__ == '__main__':
    import platform, os

    args = get_flex_args("flex_dominoes")

    if platform.system() == 'Linux':
        if args.gpu is not None:
            os.environ["DISPLAY"] = ":0." + str(args.gpu)
        else:
            os.environ["DISPLAY"] = ":0"

        launch_build = False
    else:
        launch_build = True

    if platform.system() != 'Windows' and args.fluid:
        print("WARNING: Flex fluids are only supported in Windows")

    C = FlexDominoes(
        launch_build=launch_build,
        all_flex_objects=args.all_flex_objects,
        use_cloth=args.cloth,
        use_squishy=args.squishy,
        use_fluid=args.fluid,
        step_physics=args.step_physics,
        room=args.room,
        num_middle_objects=args.num_middle_objects,
        randomize=args.random,
        seed=args.seed,
        target_zone=args.zone,
        zone_location=args.zlocation,
        zone_scale_range=args.zscale,
        zone_color=args.zcolor,
        zone_material=args.zmaterial,
        zone_friction=args.zfriction,
        target_objects=args.target,
        probe_objects=args.probe,
        middle_objects=args.middle,
        target_scale_range=args.tscale,
        target_rotation_range=args.trot,
        probe_rotation_range=args.prot,
        probe_scale_range=args.pscale,
        probe_mass_range=args.pmass,
        target_color=args.color,
        probe_color=args.pcolor,
        middle_color=args.mcolor,
        collision_axis_length=args.collision_axis_length,
        force_scale_range=args.fscale,
        force_angle_range=args.frot,
        force_offset=args.foffset,
        force_offset_jitter=args.fjitter,
        force_wait=args.fwait,
        spacing_jitter=args.spacing_jitter,
        lateral_jitter=args.lateral_jitter,
        middle_scale_range=args.mscale,
        middle_rotation_range=args.mrot,
        middle_mass_range=args.mmass,
        horizontal=args.horizontal,
        remove_target=bool(args.remove_target),
        remove_zone=bool(args.remove_zone),
        ## not scenario-specific
        camera_radius=args.camera_distance,
        camera_min_angle=args.camera_min_angle,
        camera_max_angle=args.camera_max_angle,
        camera_min_height=args.camera_min_height,
        camera_max_height=args.camera_max_height,
        monochrome=args.monochrome,
        material_types=args.material_types,
        target_material=args.tmaterial,
        probe_material=args.pmaterial,
        middle_material=args.mmaterial,
        distractor_types=args.distractor,
        distractor_categories=args.distractor_categories,
        num_distractors=args.num_distractors,
        occluder_types=args.occluder,
        occluder_categories=args.occluder_categories,
        num_occluders=args.num_occluders,
        occlusion_scale=args.occlusion_scale,
        remove_middle=args.remove_middle,
        use_ramp=bool(args.ramp),
        ramp_color=args.rcolor,
        flex_only=args.only_use_flex_objects,
        no_moving_distractors=args.no_moving_distractors        
    )

    if bool(args.run):
        C.run(num=args.num,
             output_dir=args.dir,
             temp_path=args.temp,
             width=args.width,
             height=args.height,
             write_passes=args.write_passes.split(','),
             save_passes=args.save_passes.split(','),
             save_movies=args.save_movies,
             save_labels=args.save_labels,
             args_dict=vars(args))
    else:
        end = C.communicate({"$type": "terminate"})
        print([OutputData.get_data_type_id(r) for r in end])
