import sys, os
from argparse import ArgumentParser
import h5py
import json
import copy
import importlib
import numpy as np
from enum import Enum
import random
from typing import List, Dict, Tuple
from weighted_collection import WeightedCollection
from tdw.tdw_utils import TDWUtils
from tdw.librarian import ModelRecord, MaterialLibrarian
from tdw.output_data import OutputData, Transforms
from tdw_physics.rigidbodies_dataset import (RigidbodiesDataset,
                                             get_random_xyz_transform,
                                             get_range,
                                             handle_random_transform_args)
from tdw_physics.util import (MODEL_LIBRARIES,
                              get_parser,
                              xyz_to_arr, arr_to_xyz, str_to_xyz,
                              none_or_str, none_or_int, int_or_bool)

from tdw_physics.target_controllers.dominoes import Dominoes, MultiDominoes, get_args
from tdw_physics.postprocessing.labels import is_trial_valid

MODEL_NAMES = [r.name for r in MODEL_LIBRARIES['models_flex.json'].records]
M = MaterialLibrarian()
MATERIAL_TYPES = M.get_material_types()
MATERIAL_NAMES = {mtype: [m.name for m in M.get_all_materials_of_type(mtype)] \
                  for mtype in MATERIAL_TYPES}

def get_gravity_args(dataset_dir: str, parse=True):

    common = get_parser(dataset_dir, get_help=False)
    domino, domino_postproc = get_args(dataset_dir, parse=False)
    parser = ArgumentParser(parents=[common, domino], conflict_handler='resolve', fromfile_prefix_chars='@')

    # what type of scenario to make it
    parser.add_argument("--middle",
                        type=str,
                        default="ramp",
                        help="comma-separated list of possible middle objects/scenario types")
    parser.add_argument("--num_middle_objects",
                        type=int,
                        default=2,
                        help="The number of middle objects to place")    
    parser.add_argument("--mfriction",
                        type=float,
                        default=0.1,
                        help="Static and dynamic friction on middle objects")        
    parser.add_argument("--ramp",
                        type=int,
                        default=1,
                        help="Whether to place the probe object on the top of a ramp")
    parser.add_argument("--rheight",
                        type=none_or_str,
                        default=0.5,
                        help="Height of the ramp base")
    parser.add_argument("--rscale",
                        type=none_or_str,
                        default=None,
                        help="The xyz scale of the ramp")
    parser.add_argument("--rcolor",
                        type=none_or_str,
                        default=None,
                        help="The rgb color of the ramp")        
    parser.add_argument("--probe",
                        type=str,
                        default="sphere",
                        help="comma-separated list of possible probe objects")
    parser.add_argument("--pscale",
                        type=str,
                        default="0.2",
                        help="scale of probe objects")
    parser.add_argument("--pmass",
                        type=str,
                        default="1.0",
                        help="scale of probe objects")
    parser.add_argument("--foffset",
                        type=str,
                        default="0.0,0.5,0.0",
                        help="offset from probe centroid from which to apply force, relative to probe scale")    

    parser.add_argument("--collision_axis_length",
                        type=float,
                        default=2.5,
                        help="How far to put the probe and target")    

    # camera
    parser.add_argument("--camera_distance",
                        type=float,
                        default=3.0,
                        help="radial distance from camera to centerpoint")
    parser.add_argument("--camera_min_angle",
                        type=float,
                        default=0,
                        help="minimum angle of camera rotation around centerpoint")
    parser.add_argument("--camera_max_angle",
                        type=float,
                        default=90,
                        help="maximum angle of camera rotation around centerpoint")
    parser.add_argument("--camera_min_height",
                        type=float,
                        default=2.0,
                         help="min height of camera")
    parser.add_argument("--camera_max_height",
                        type=float,
                        default=3.0,
                        help="max height of camera")
        


    def postprocess(args):

        args = domino_postproc(args)
        args.rheight = handle_random_transform_args(args.rheight)

        return args

    if not parse:
        return (parser, postprocess)

    args = parser.parse_args()
    args = postprocess(args)

    return args
    
class Gravity(Dominoes):
    
    def __init__(self,
                 port: int = 1071,
                 middle_scale_range=1.0,
                 middle_color=None,
                 middle_material=None,
                 middle_friction=0.1,
                 remove_middle = False,
                 **kwargs):

        Dominoes.__init__(self, port=port, **kwargs)

        # always use a ramp for probe
        self.use_ramp = True
        
        # middle
        self.middle_color = middle_color
        self.middle_material = middle_material
        self.middle_friction = middle_friction
        self.middle_objects = []
        self.remove_middle = remove_middle

    def _write_static_data(self, static_group: h5py.Group) -> None:
        Dominoes._write_static_data(self, static_group)
        self.middle_type = type(self).__name__
        static_group.create_dataset("middle_type", data=self.middle_type)        

    @staticmethod
    def get_controller_label_funcs(classname = 'Gravity'):

        funcs = super(Gravity, Gravity).get_controller_label_funcs(classname)

        return funcs

    def _build_intermediate_structure(self) -> None:

        # append scales and colors and names
        for (record, data) in self.middle_objects:
            self.model_names.append(record.name)
            self.scales.append(copy.deepcopy(data['scale']))        
            self.colors = np.concatenate([self.colors, np.array(data['color']).reshape((1,3))], axis=0)

        # aim camera
        camera_y_aim = max(0.5, self.ramp_base_height)
        self.camera_aim = arr_to_xyz([0., camera_y_aim, 0.])

        return []

class Ramp(Gravity):

    RAMPS = [r for r in MODEL_LIBRARIES['models_full.json'].records if 'ramp' in r.name]

    def __init__(self,
                 port: int = 1071,
                 middle_objects='ramp',
                 middle_scale_range=1.0,
                 spacing_jitter=0.0,
                 **kwargs):
        
        super().__init__(port=port, **kwargs)
        
        self._middle_types = self.RAMPS
        self.middle_scale_range = middle_scale_range

    def _write_static_data(self, static_group: h5py.Group) -> None:
        super()._write_static_data(static_group)


        static_group.create_dataset("middle_material", data=self.middle_material)                
        static_group.create_dataset("middle_id", data=self.middle_id)

    def _build_intermediate_structure(self) -> List[dict]:

        ramp_pos = TDWUtils.VECTOR3_ZERO
        ramp_rot = TDWUtils.VECTOR3_ZERO

        self.middle = random.choice(self._middle_types)
        self.middle_id = self._get_next_object_id()
        self.middle_scale = get_random_xyz_transform(self.middle_scale_range)
        rgb = self.middle_color or self.random_color(exclude=self.target_color)

        ramp_data = {
            'name': self.middle.name,
            'id': self.middle_id,
            'scale': self.middle_scale,
            'color': rgb}        
        
        commands = self.add_ramp(
            record = self.middle,
            position = ramp_pos,
            rotation = ramp_rot,
            scale = self.middle_scale,
            o_id = self.middle_id,
            color=rgb,
            material=self.middle_material,
            mass = 500,
            static_friction=self.middle_friction,
            dynamic_friction=self.middle_friction,
            bounciness = 0,            
            add_data = False)

        # append data        
        self.middle_objects.append(
            (self.middle, ramp_data))
        commands.extend(super()._build_intermediate_structure())        

        return commands

class Pit(Gravity, MultiDominoes):

    def __init__(self,
                 port=1071,
                 num_middle_objects=2,
                 middle_scale_range={'x': 0.5, 'y': 1.0, 'z': 0.5},
                 spacing_jitter = 0.25,
                 lateral_jitter = 0.0,
                 middle_friction = 0.1,
                 **kwargs):

        # middle config
        Gravity.__init__(self, port=port, middle_friction=middle_friction, **kwargs)        
        MultiDominoes.__init__(self, launch_build=False,
                               middle_objects='cube',
                               num_middle_objects=num_middle_objects,
                               middle_scale_range=middle_scale_range,
                               spacing_jitter=spacing_jitter,
                               lateral_jitter=lateral_jitter,
                               **kwargs)

        # raise the ramp
        if hasattr(self.middle_scale_range, 'keys'):
            self.pit_max_height = get_range(self.middle_scale_range['y'])[1]
        elif hasattr(self.middle_scale_range, '__len__'):
            self.pit_max_height = self.middle_scale_range[1]
        else:
            self.pit_max_height = self.middle_scale_range

        self.ramp_base_height_range = [r + self.pit_max_height
                                       for r in get_range(self.ramp_base_height_range)]

    def _write_static_data(self, static_group: h5py.Group) -> None:
        Gravity._write_static_data(self, static_group)
        static_group.create_dataset("num_middle_objects", data=self.num_middle_objects)
        static_group.create_dataset("pit_widths", data=self.pit_widths)

    def _build_intermediate_structure(self) -> List[dict]:
        
        commands = []

        print("THIS IS A PIT!")
        print(self.num_middle_objects)

        # get the scale of the total pit object
        scale = get_random_xyz_transform(self.middle_scale_range)
        self.pit_mass = random.uniform(*get_range(self.middle_mass_range))

        # get color and texture
        self.pit_color = self.middle_color or self.random_color(exclude=self.target_color)
        self.pit_material = self.middle_material

        # how wide are the pits?
        self.pit_widths = [
            random.uniform(0.0, self.spacing_jitter) * scale['x']
            for _ in range(self.num_middle_objects - 1)]

        # make M cubes and scale in x accordingly
        x_remaining = scale['x'] - self.pit_widths[0]
        x_filled = 0.0
        
        print("PIT WIDTHS", self.pit_widths)
        
        for m in range(self.num_middle_objects):
            print("x_filled, remaining", x_filled, x_remaining)

            m_rec = random.choice(self._middle_types)

            x_scale = random.uniform(0.0, x_remaining)
                
            x_len,_,_ = self.get_record_dimensions(m_rec)
            x_len *= x_scale
            x_pos = self.ramp_end_x + x_filled + (0.5 * x_len)
            z_pos = random.uniform(-self.lateral_jitter, self.lateral_jitter)

            print(m)
            print("ramp_end", self.ramp_end_x)
            print("x_len", x_len)
            print("x_scale", x_scale)
            print("x_pos", x_pos)
            print("z_pos", z_pos)

            m_scale = arr_to_xyz([x_scale, scale['y'], scale['z']])

            commands.extend(
                self.add_primitive(
                    record = m_rec,
                    position=arr_to_xyz([x_pos, 0., z_pos]),
                    rotation=TDWUtils.VECTOR3_ZERO,
                    scale = m_scale,
                    color = self.pit_color,
                    exclude_color = self.target_color,
                    material = self.pit_material,
                    mass = self.pit_mass,
                    dynamic_friction = self.middle_friction,
                    static_friction = self.middle_friction,
                    scale_mass = True,
                    make_kinematic = True,
                    add_data = True,
                    obj_list = self.middle_objects))

            if m < len(self.pit_widths):
                x_filled += self.pit_widths[m] + x_len
                x_remaining -= (self.pit_widths[m] + x_len)

        commands.extend(Gravity._build_intermediate_structure(self))

        print("INTERMEDIATE")
        print(commands)

        return commands

class Pendulum(Gravity):

    def _build_intermediate_structure(self) -> List[dict]:
        
        commands = []

        commands.extend(super()._build_intermediate_structure())
        
        return commands

if __name__ == '__main__':

    args = get_gravity_args("gravity")
    
    classes = [Ramp, Pit, Pendulum]
    for c in classes:
        if args.middle[0].capitalize() == c.__name__:
            classtype = c
            break
        else:
            classtype = Gravity
    
    GC = classtype(

        # gravity specific
        num_middle_objects=args.num_middle_objects,
        middle_scale_range=args.mscale,
        middle_friction=args.mfriction,
        ramp_base_height_range=args.rheight,
        ramp_scale=args.rscale,
        ramp_has_friction=args.rfriction,
        probe_has_friction=args.pfriction,
        spacing_jitter=args.spacing_jitter,
        lateral_jitter=args.lateral_jitter,
        
        # domino specific
        target_zone=args.zone,
        zone_location=args.zlocation,
        zone_scale_range=args.zscale,
        zone_color=args.zcolor,
        zone_friction=args.zfriction,        
        target_objects=args.target,
        probe_objects=args.probe,
        target_scale_range=args.tscale,
        target_rotation_range=args.trot,
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
        remove_target=bool(args.remove_target),
        remove_zone=bool(args.remove_zone),
        
        ## not scenario-specific
        room=args.room,
        randomize=args.random,
        seed=args.seed,
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
        zone_material=args.zmaterial,
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
        ramp_material=args.rmaterial,
        flex_only=args.only_use_flex_objects,
        no_moving_distractors=args.no_moving_distractors        
    )


    if bool(args.run):
        GC.run(num=args.num,
               output_dir=args.dir,
               temp_path=args.temp,
               width=args.width,
               height=args.height,
               write_passes=args.write_passes.split(','),               
               save_passes=args.save_passes.split(','),
               save_movies=args.save_movies,
               save_labels=args.save_labels,               
               args_dict=vars(args)
        )
    else:
        GC.communicate({"$type": "terminate"})    
