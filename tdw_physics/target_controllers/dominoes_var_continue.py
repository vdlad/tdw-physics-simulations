from argparse import ArgumentParser
import sys
import subprocess
import h5py
import json
import copy
import importlib
from PIL import Image
import io
from tqdm import tqdm
import numpy as np
from pathlib import Path
from enum import Enum
import random
from typing import List, Dict, Tuple
from collections import OrderedDict
from weighted_collection import WeightedCollection
from tdw.tdw_utils import TDWUtils
from tdw.librarian import ModelRecord, MaterialLibrarian
from tdw.output_data import OutputData, Transforms, Images, CameraMatrices
from tdw_physics.rigidbodies_dataset import (RigidbodiesDataset,
                                             get_random_xyz_transform,
                                             get_range,
                                             handle_random_transform_args)
from tdw_physics.util import (MODEL_LIBRARIES, FLEX_MODELS, MODEL_CATEGORIES,
                              MATERIAL_TYPES, MATERIAL_NAMES,
                              get_parser,
                              xyz_to_arr, arr_to_xyz, str_to_xyz,
                              none_or_str, none_or_int, int_or_bool)
from tdw_physics.postprocessing.stimuli import pngs_to_mp4
from tdw_physics.postprocessing.labels import (get_all_label_funcs,
                                               get_labels_from)
from tdw_physics.util_geom import save_obj
PRIMITIVE_NAMES = [r.name for r in MODEL_LIBRARIES['models_flex.json'].records if not r.do_not_use]
FULL_NAMES = [r.name for r in MODEL_LIBRARIES['models_full.json'].records if not r.do_not_use]
PASSES = ["_img", "_depth", "_normals", "_flow", "_id"]
ZONE_COLOR = [255,255,0]
TARGET_COLOR = [255,0,0]


def get_args(dataset_dir: str, parse=True):
    """
    Combine Domino-specific arguments with controller-common arguments
    """
    common = get_parser(dataset_dir, get_help=False)
    parser = ArgumentParser(parents=[common], add_help=parse, fromfile_prefix_chars='@')

    parser.add_argument("--num_middle_objects",
                        type=int,
                        default=3,
                        help="The number of middle objects to place")
    parser.add_argument("--num_distinct_objects",
                        type=int,
                        default=3,
                        help="The number of middle objects to place")
    parser.add_argument("--zone",
                        type=str,
                        default="cube",
                        help="comma-separated list of possible target zone shapes")
    parser.add_argument("--target",
                        type=str,
                        default="cube",
                        help="comma-separated list of possible target objects")
    parser.add_argument("--probe",
                        type=str,
                        default="cube",
                        help="comma-separated list of possible probe objects")
    parser.add_argument("--middle",
                        type=str,
                        default=None,
                        help="comma-separated list of possible middle objects; default to same as target")
    parser.add_argument("--ramp",
                        type=int,
                        default=0,
                        help="Whether to place the probe object on the top of a ramp")
    parser.add_argument("--rscale",
                        type=none_or_str,
                        default=None,
                        help="The xyz scale of the ramp")
    parser.add_argument("--rfriction",
                        action="store_true",
                        help="Whether the ramp has friction")

    parser.add_argument("--zscale",
                        type=str,
                        default="0.5,0.01,2.0",
                        help="scale of target zone")
    parser.add_argument("--zlocation",
                        type=none_or_str,
                        default=None,
                        help="Where to place the target zone. None will default to a scenario-specific place.")
    parser.add_argument("--zfriction",
                        type=float,
                        default=0.1,
                        help="Static and dynamic friction on the target zone.")
    parser.add_argument("--tscale",
                        type=str,
                        default="0.1,0.5,0.25",
                        help="scale of target objects")
    parser.add_argument("--trot",
                        type=str,
                        default="[0,0]",
                        help="comma separated list of initial target rotation values")
    parser.add_argument("--mrot",
                        type=str,
                        default="[-30,30]",
                        help="comma separated list of initial middle object rotation values")
    parser.add_argument("--prot",
                        type=str,
                        default="[0,0]",
                        help="comma separated list of initial probe rotation values")
    parser.add_argument("--phorizontal",
                        type=int_or_bool,
                        default=0,
                        help="whether the probe is horizontal")
    parser.add_argument("--mscale",
                        type=str,
                        default="0.1,0.5,0.25",
                        help="Scale or scale range for middle objects")
    parser.add_argument("--mmass",
                        type=str,
                        default="2.0",
                        help="Scale or scale range for middle objects")
    parser.add_argument("--horizontal",
                        type=int_or_bool,
                        default=0,
                        help="Whether to rotate middle objects horizontally")
    parser.add_argument("--pscale",
                        type=str,
                        default="0.1,0.5,0.25",
                        help="scale of probe objects")
    parser.add_argument("--pmass",
                        type=str,
                        default="2.0",
                        help="scale of probe objects")
    parser.add_argument("--fscale",
                        type=str,
                        default="2.0",
                        help="range of scales to apply to push force")
    parser.add_argument("--frot",
                        type=str,
                        default="[0,0]",
                        help="range of angles in xz plane to apply push force")
    parser.add_argument("--foffset",
                        type=str,
                        default="0.0,0.8,0.0",
                        help="offset from probe centroid from which to apply force, relative to probe scale")
    parser.add_argument("--fjitter",
                        type=float,
                        default=0.0,
                        help="jitter around object centroid to apply force")
    parser.add_argument("--fwait",
                        type=none_or_str,
                        default="[0,0]",
                        help="How many frames to wait before applying the force")
    parser.add_argument("--tcolor",
                        type=none_or_str,
                        default="1.0,0.0,0.0",
                        help="comma-separated R,G,B values for the target object color. None to random.")
    parser.add_argument("--zcolor",
                        type=none_or_str,
                        default="1.0,1.0,0.0",
                        help="comma-separated R,G,B values for the target zone color. None is random")
    parser.add_argument("--rcolor",
                        type=none_or_str,
                        default="0.75,0.75,1.0",
                        help="comma-separated R,G,B values for the target zone color. None is random")
    parser.add_argument("--pcolor",
                        type=none_or_str,
                        default="0.0,1.0,1.0",
                        help="comma-separated R,G,B values for the probe object color. None is random.")
    parser.add_argument("--mcolor",
                        type=none_or_str,
                        default=None,
                        help="comma-separated R,G,B values for the middle object color. None is random.")
    parser.add_argument("--collision_axis_length",
                        type=float,
                        default=2.0,
                        help="Length of spacing between probe and target objects at initialization.")
    parser.add_argument("--spacing_jitter",
                        type=float,
                        default=0.2,
                        help="jitter in how to space middle objects, as a fraction of uniform spacing")
    parser.add_argument("--lateral_jitter",
                        type=float,
                        default=0.2,
                        help="lateral jitter in how to space middle objects, as a fraction of object width")
    parser.add_argument("--remove_target",
                        type=int_or_bool,
                        default=0,
                        help="Don't actually put the target object in the scene.")
    parser.add_argument("--remove_zone",
                        type=int_or_bool,
                        default=0,
                        help="Don't actually put the target zone in the scene.")
    parser.add_argument("--camera_distance",
                        type=none_or_str,
                        default="1.75",
                        help="radial distance from camera to centerpoint")
    parser.add_argument("--camera_min_height",
                        type=float,
                        default=0.75,
                         help="min height of camera")
    parser.add_argument("--camera_max_height",
                        type=float,
                        default=2.0,
                        help="max height of camera")
    parser.add_argument("--camera_min_angle",
                        type=float,
                        default=45,
                        help="minimum angle of camera rotation around centerpoint")
    parser.add_argument("--camera_max_angle",
                        type=float,
                        default=225,
                        help="maximum angle of camera rotation around centerpoint")
    parser.add_argument("--camera_left_right_reflections",
                        action="store_true",
                        help="Whether camera angle range includes reflections along the collision axis")
    parser.add_argument("--material_types",
                        type=none_or_str,
                        default="Wood,Metal,Plastic",
                        help="Which class of materials to sample material names from")
    parser.add_argument("--tmaterial",
                        type=none_or_str,
                        default="parquet_wood_red_cedar",
                        help="Material name for target. If None, samples from material_type")
    parser.add_argument("--zmaterial",
                        type=none_or_str,
                        default="wood_european_ash",
                        help="Material name for target. If None, samples from material_type")
    parser.add_argument("--rmaterial",
                        type=none_or_str,
                        default=None,
                        help="Material name for ramp. If None, same as zone material")
    parser.add_argument("--pmaterial",
                        type=none_or_str,
                        default="parquet_wood_red_cedar",
                        help="Material name for probe. If None, samples from material_type")
    parser.add_argument("--pfriction",
                        action="store_true",
                        help="Whether the probe object has friction")
    parser.add_argument("--mmaterial",
                        type=none_or_str,
                        default="parquet_wood_red_cedar",
                        help="Material name for middle objects. If None, samples from material_type")
    parser.add_argument("--distractor",
                        type=none_or_str,
                        default="core",
                        help="The names or library of distractor objects to use")
    parser.add_argument("--distractor_categories",
                        type=none_or_str,
                        help="The categories of distractors to choose from (comma-separated)")
    parser.add_argument("--num_distractors",
                        type=int,
                        default=0,
                        help="The number of background distractor objects to place")
    parser.add_argument("--distractor_aspect_ratio",
                        type=none_or_str,
                        default=None,
                        help="The range of valid distractor aspect ratios")
    parser.add_argument("--occluder",
                        type=none_or_str,
                        default="core",
                        help="The names or library of occluder objects to use")
    parser.add_argument("--occluder_categories",
                        type=none_or_str,
                        help="The categories of occluders to choose from (comma-separated)")
    parser.add_argument("--num_occluders",
                        type=int,
                        default=0,
                        help="The number of foreground occluder objects to place")
    parser.add_argument("--occlusion_scale",
                        type=float,
                        default=0.75,
                        help="The height of the occluders as a proportion of camera height")
    parser.add_argument("--occluder_aspect_ratio",
                        type=none_or_str,
                        default=None,
                        help="The range of valid occluder aspect ratios")
    parser.add_argument("--no_moving_distractors",
                        action="store_true",
                        help="Prevent all distractors (and occluders) from moving by making them 'kinematic' objects")


    parser.add_argument("--remove_middle",
                        action="store_true",
                        help="Remove one of the middle dominoes scene.")

    # which models are allowed
    parser.add_argument("--model_libraries",
                        type=none_or_str,
                        default=','.join(list(MODEL_LIBRARIES.keys())),
                        help="Which model libraries can be drawn from")
    parser.add_argument("--only_use_flex_objects",
                        action="store_true",
                        help="Only use models that are FLEX models (and have readable meshes)")

    # for generating training data without zones, targets, caps, and at lower resolution
    parser.add_argument("--training_data_mode",
                        action="store_true",
                        help="Overwrite some parameters to generate training data without target objects, zones, etc.")
    parser.add_argument("--readout_data_mode",
                        action="store_true",
                        help="Overwrite some parameters to generate training data without target objects, zones, etc.")
    parser.add_argument("--testing_data_mode",
                        action="store_true",
                        help="Overwrite some parameters to generate training data without target objects, zones, etc.")
    parser.add_argument("--match_probe_and_target_color",
                        action="store_true",
                        help="Probe and target will have the same color.")

    def postprocess(args):

        # testing set data drew from a different set of models; needs to be preserved
        # for correct occluder/distractor sampling
        PRIMITIVE_NAMES = [r.name for r in MODEL_LIBRARIES['models_flex.json'].records if not r.do_not_use]
        FULL_NAMES = [r.name for r in MODEL_LIBRARIES['models_full.json'].records if not r.do_not_use]

        if not (args.training_data_mode or args.readout_data_mode):
            PRIMITIVE_NAMES = [r.name for r in MODEL_LIBRARIES['models_flex.json'].records]
            FULL_NAMES = [r.name for r in MODEL_LIBRARIES['models_full.json'].records]

        # choose a valid room
        assert args.room in ['box', 'tdw', 'house'], args.room

        # parse the model libraries
        if args.model_libraries is not None:
            if not isinstance(args.model_libraries, list):
                args.model_libraries = args.model_libraries.split(',')
            libs = []
            for lib in args.model_libraries:
                if 'models_' not in lib:
                    libs.append('models_' + lib)
                else:
                    libs.append(lib)
            args.model_libraries = libs

        # whether to set all objects same color
        args.monochrome = bool(args.monochrome)

        # camera distance
        args.camera_distance = handle_random_transform_args(args.camera_distance)

        # scaling and rotating of objects
        args.rscale = handle_random_transform_args(args.rscale)
        args.zscale = handle_random_transform_args(args.zscale)
        args.zlocation = handle_random_transform_args(args.zlocation)
        args.tscale = handle_random_transform_args(args.tscale)
        args.trot = handle_random_transform_args(args.trot)
        args.pscale = handle_random_transform_args(args.pscale)
        args.pmass = handle_random_transform_args(args.pmass)
        args.prot = handle_random_transform_args(args.prot)
        args.mscale = handle_random_transform_args(args.mscale)
        args.mrot = handle_random_transform_args(args.mrot)
        args.mmass = handle_random_transform_args(args.mmass)

        # the push force scale and direction
        args.fscale = handle_random_transform_args(args.fscale)
        args.frot = handle_random_transform_args(args.frot)
        args.foffset = handle_random_transform_args(args.foffset)
        args.fwait = handle_random_transform_args(args.fwait)

        args.horizontal = bool(args.horizontal)

        # occluders and distrators
        args.occluder_aspect_ratio = handle_random_transform_args(args.occluder_aspect_ratio)
        args.distractor_aspect_ratio = handle_random_transform_args(args.distractor_aspect_ratio)

        if args.zone is not None:
            zone_list = args.zone.split(',')
            assert all([t in PRIMITIVE_NAMES for t in zone_list]), \
                "All target object names must be elements of %s" % PRIMITIVE_NAMES
            args.zone = zone_list
        else:
            args.zone = PRIMITIVE_NAMES

        if args.target is not None:
            targ_list = args.target.split(',')
            assert all([t in PRIMITIVE_NAMES for t in targ_list]), \
                "All target object names must be elements of %s" % PRIMITIVE_NAMES
            args.target = targ_list
        else:
            args.target = PRIMITIVE_NAMES

        if args.probe is not None:
            probe_list = args.probe.split(',')
            assert all([t in PRIMITIVE_NAMES for t in probe_list]), \
                "All target object names must be elements of %s" % PRIMITIVE_NAMES
            args.probe = probe_list
        else:
            args.probe = PRIMITIVE_NAMES

        if args.middle is not None:
            middle_list = args.middle.split(',')
            args.middle = middle_list

        if args.tcolor is not None:
            rgb = [float(c) for c in args.tcolor.split(',')]
            assert len(rgb) == 3, rgb
            args.tcolor = args.color = rgb
        else:
            args.tcolor = args.color = None

        if args.zcolor is not None:
            rgb = [float(c) for c in args.zcolor.split(',')]
            assert len(rgb) == 3, rgb
            args.zcolor = rgb

        if args.rcolor is not None:
            rgb = [float(c) for c in args.rcolor.split(',')]
            assert len(rgb) == 3, rgb
            args.rcolor = rgb

        if args.pcolor is not None:
            rgb = [float(c) for c in args.pcolor.split(',')]
            assert len(rgb) == 3, rgb
            args.pcolor = rgb

        if args.mcolor is not None:
            rgb = [float(c) for c in args.mcolor.split(',')]
            assert len(rgb) == 3, rgb
            args.mcolor = rgb


        if args.material_types is None:
            args.material_types = MATERIAL_TYPES
        else:
            matlist = args.material_types.split(',')
            assert all ([m in MATERIAL_TYPES for m in matlist]), \
                "All material types must be elements of %s" % MATERIAL_TYPES
            args.material_types = matlist

        if args.distractor is None or args.distractor == 'full':
            args.distractor = FULL_NAMES
        elif args.distractor == 'core':
            args.distractor = [r.name for r in MODEL_LIBRARIES['models_core.json'].records]
        elif args.distractor in ['flex', 'primitives']:
            args.distractor = PRIMITIVE_NAMES
        else:
            d_names = args.distractor.split(',')
            args.distractor = [r for r in FULL_NAMES if any((nm in r for nm in d_names))]

        if args.occluder is None or args.occluder == 'full':
            args.occluder = FULL_NAMES
        elif args.occluder == 'core':
            args.occluder = [r.name for r in MODEL_LIBRARIES['models_core.json'].records]
        elif args.occluder in ['flex', 'primitives']:
            args.occluder = PRIMITIVE_NAMES
        else:
            o_names = args.occluder.split(',')
            args.occluder = [r for r in FULL_NAMES if any((nm in r for nm in o_names))]

        # produce training data
        if args.training_data_mode:

            # multiply the number of trials by a factor
            args.num = int(float(args.num) * args.num_multiplier)

            # change the random seed in a deterministic way
            args.random = 0
            args.seed = (args.seed * 1000) % 997

            # randomize colors and wood textures
            args.match_probe_and_target_color = False
            args.color = args.tcolor = args.zcolor = args.pcolor = args.mcolor = args.rcolor = None

            # only use the flex objects and make sure the distractors don't move
            args.only_use_flex_objects = args.no_moving_distractors = True

            # only save out the RGB images and the segmentation masks
            args.write_passes = "_img,_id,_depth"
            args.save_passes = ""
            args.save_movies = False
            args.save_meshes = True
            args.use_test_mode_colors = False

        # produce "readout" training data with red target and yellow zone,
        # but seed is still different from whatever it was in the commandline_args.txt config
        elif args.readout_data_mode:

            # multiply the number of trials by a factor
            args.num = int(float(args.num) * args.num_multiplier)

            # change the random seed in a deterministic way
            args.random = 0
            args.seed = (args.seed * 3000) % 1999

            # target is red, zone is yellow, others are random
            args.use_test_mode_colors = False
            args.color = args.tcolor = [1.0, 0.0, 0.0]
            args.zcolor = [1.0, 1.0, 0.0]
            args.pcolor = args.mcolor = args.rcolor = None

            # only use the flex objects and make sure the distractors don't move
            args.only_use_flex_objects = args.no_moving_distractors = True

            # only save out the RGB images and the segmentation masks
            args.write_passes = "_img,_id"
            args.save_passes = ""
            args.save_movies = False
            args.save_meshes = True

        # produce the same trials as the testing trials, but with red / yellow;
        # seed MUST be pulled from a config.
        elif args.testing_data_mode:

            assert args.random == 0, "You can't regenerate the testing data without --random 0"
            assert args.seed != -1, "Seed has to be specified but is instead the default"
            assert all((('seed' not in a) for a in sys.argv[1:])), "You can't pass a new seed argument for generating the testing data; use the one in the commandline_args.txt config!"

            # red and yellow target and zone
            args.use_test_mode_colors = True

            args.write_passes = "_img,_id,_depth,_normals,_flow"
            args.save_passes = "_img,_id"
            args.save_movies = True
            args.save_meshes = True
        else:
            args.use_test_mode_colors = False

        return args

    if not parse:
        return (parser, postprocess)

    args = parser.parse_args()
    args = postprocess(args)

    return args

class Dominoes(RigidbodiesDataset):
    """
    Drop a random Flex primitive object on another random Flex primitive object
    """

    MAX_TRIALS = 1000
    DEFAULT_RAMPS = [r for r in MODEL_LIBRARIES['models_full.json'].records if 'ramp_with_platform_30' in r.name]
    CUBE = [r for r in MODEL_LIBRARIES['models_flex.json'].records if 'cube' in r.name][0]
    PRINT = False

    def __init__(self,
                 port: int = None,
                 room='box',
                 target_zone=['cube'],
                 zone_color=[1.0,1.0,0.0], #yellow is the default color for target zones
                 zone_location=None,
                 zone_scale_range=[0.5,0.01,0.5],
                 zone_friction=0.1,
                 probe_objects=PRIMITIVE_NAMES,
                 target_objects=PRIMITIVE_NAMES,
                 probe_scale_range=[0.2, 0.3],
                 probe_mass_range=[2.,7.],
                 probe_color=None,
                 probe_rotation_range=[0,0],
                 target_scale_range=[0.2, 0.3],
                 target_rotation_range=None,
                 target_color=None,
                 target_motion_thresh=0.01,
                 collision_axis_length=1.,
                 force_scale_range=[0.,8.],
                 force_angle_range=[-60,60],
                 force_offset={"x":0.,"y":0.5,"z":0.0},
                 force_offset_jitter=0.1,
                 force_wait=None,
                 remove_target=False,
                 remove_zone=False,
                 camera_radius=2.0,
                 camera_min_angle=0,
                 camera_max_angle=360,
                 camera_left_right_reflections=False,
                 camera_min_height=1./3,
                 camera_max_height=2./3,
                 material_types=MATERIAL_TYPES,
                 target_material=None,
                 probe_material=None,
                 probe_has_friction=False,
                 ramp_material=None,
                 zone_material=None,
                 model_libraries=MODEL_LIBRARIES.keys(),
                 distractor_types=PRIMITIVE_NAMES,
                 distractor_categories=None,
                 num_distractors=0,
                 distractor_aspect_ratio=None,
                 occluder_types=PRIMITIVE_NAMES,
                 occluder_categories=None,
                 num_occluders=0,
                 occlusion_scale=0.6,
                 occluder_aspect_ratio=None,
                 use_ramp=False,
                 ramp_has_friction=False,
                 ramp_scale=None,
                 ramp_color=[0.75,0.75,1.0],
                 ramp_base_height_range=0,
                 flex_only=False,
                 no_moving_distractors=False,
                 match_probe_and_target_color=False,
                 use_test_mode_colors=False,
                 **kwargs):

        ## get random port unless one is specified
        if port is None:
            port = np.random.randint(1000,4000)
            print("random port",port,"chosen. If communication with tdw build fails, set port to 1071 or update your tdw installation.")

        ## initializes static data and RNG
        super().__init__(port=port, **kwargs)

        ## which room to use
        self.room = room

        ## which model libraries can be sampled from
        self.model_libraries = model_libraries

        ## whether only flex objects are allowed
        self.flex_only = flex_only

        ## whether the occluders and distractors can move
        self.no_moving_distractors = no_moving_distractors

        ## color randomization
        self._random_target_color = (target_color is None)
        self._random_zone_color = (zone_color is None)
        self._random_probe_color = (probe_color is None)

        ## target zone
        self.set_zone_types(target_zone)
        self.zone_location = zone_location
        self.zone_color = zone_color
        self.zone_scale_range = zone_scale_range
        self.zone_material = zone_material
        self.zone_friction = zone_friction
        self.remove_zone = remove_zone

        ## allowable object types
        self.set_probe_types(probe_objects)
        self.set_target_types(target_objects)
        self.material_types = material_types
        self.remove_target = remove_target

        # whether to use a ramp
        self.use_ramp = use_ramp
        self.ramp_color = ramp_color
        self.ramp_material = ramp_material or self.zone_material
        if ramp_scale is not None:
            self.ramp_scale = get_random_xyz_transform(ramp_scale)
        else:
            self.ramp_scale = None
        self.ramp_base_height_range = ramp_base_height_range
        self.ramp_physics_info = {}
        if ramp_has_friction:
            self.ramp_physics_info.update({
                'mass': 1000,
                'static_friction': 0.1,
                'dynamic_friction': 0.1,
                'bounciness': 0.1})
        self.probe_has_friction = probe_has_friction

        ## object generation properties
        self.target_scale_range = target_scale_range
        self.target_color = target_color
        self.target_rotation_range = target_rotation_range
        self.target_material = target_material
        self.target_motion_thresh = target_motion_thresh

        self.probe_color = probe_color
        self.probe_scale_range = probe_scale_range
        self.probe_rotation_range = probe_rotation_range
        self.probe_mass_range = get_range(probe_mass_range)
        self.probe_material = probe_material
        self.match_probe_and_target_color = match_probe_and_target_color

        self.middle_scale_range = target_scale_range

        ## Scenario config properties
        self.collision_axis_length = collision_axis_length
        self.force_scale_range = force_scale_range
        self.force_angle_range = force_angle_range
        self.force_offset = get_random_xyz_transform(force_offset)
        self.force_offset_jitter = force_offset_jitter
        self.force_wait_range = force_wait or [0,0]

        ## camera properties
        self.camera_radius_range = get_range(camera_radius)
        self.camera_min_angle = camera_min_angle
        self.camera_max_angle = camera_max_angle
        self.camera_left_right_reflections = camera_left_right_reflections
        self.camera_min_height = camera_min_height
        self.camera_max_height = camera_max_height
        self.camera_aim = {"x": 0., "y": 0.5, "z": 0.} # fixed aim

        ## distractors and occluders
        self.num_distractors = num_distractors
        self.distractor_aspect_ratio = get_range(distractor_aspect_ratio)
        self.distractor_types = self.get_types(
            distractor_types,
            libraries=self.model_libraries,
            categories=distractor_categories,
            flex_only=self.flex_only,
            aspect_ratio_min=self.distractor_aspect_ratio[0],
            aspect_ratio_max=self.distractor_aspect_ratio[1]
        )

        self.num_occluders = num_occluders
        self.occlusion_scale = occlusion_scale
        self.occluder_aspect_ratio = get_range(occluder_aspect_ratio)
        self.occluder_types = self.get_types(
            occluder_types,
            libraries=self.model_libraries,
            categories=occluder_categories,
            flex_only=self.flex_only,
            aspect_ratio_min=self.occluder_aspect_ratio[0],
            aspect_ratio_max=self.occluder_aspect_ratio[1],
        )

        self.use_test_mode_colors = use_test_mode_colors

    def trial_loop(self,
                   num: int,
                   output_dir: str,
                   temp_path: str) -> None:


        output_dir = Path(output_dir)
        if not output_dir.exists():
            output_dir.mkdir(parents=True)
        temp_path = Path(temp_path)
        if not temp_path.parent.exists():
            temp_path.parent.mkdir(parents=True)
        # Remove an incomplete temp path.
        if temp_path.exists():
            temp_path.unlink()

        pbar = tqdm(total=num)
        # Skip trials that aren't on the disk, and presumably have been uploaded; jump to the highest number.
        exists_up_to = 0
        for f in output_dir.glob("*.hdf5"):
            if int(f.stem) > exists_up_to:
                exists_up_to = int(f.stem)

        if exists_up_to > 0:
            print('Trials up to %d already exist, skipping those' % exists_up_to)
        self.num_interactions = 10
        pbar.update(exists_up_to)
        for i in range(exists_up_to, num):
            object_info = None
            # see if the last iteraction is there, otherwise re-generate the trials
            filepath = output_dir.joinpath(TDWUtils.zero_padding(i, 4) + "_" + TDWUtils.zero_padding(self.num_interactions - 1, 3) + ".hdf5")
            if not filepath.exists():
                #for interact_id in range(self.num_interactions):
                filepath = output_dir.joinpath(TDWUtils.zero_padding(i, 4) + ".hdf5")
                self.stimulus_name = '_'.join([filepath.parent.name, str(Path(filepath.name).with_suffix(''))])

                # Save out images
                self.png_dir = None
                if any([pa in PASSES for pa in self.save_passes]):
                    self.png_dir = output_dir.joinpath("pngs_" + TDWUtils.zero_padding(i, 4))
                    if not self.png_dir.exists():
                        self.png_dir.mkdir(parents=True)

                # Do the trial.
                object_info = self.trial(filepath=filepath,
                           temp_path=temp_path,
                           trial_num=i,
                           object_info=object_info)

                # Save an MP4 of the stimulus
                if self.save_movies:

                    for pass_mask in self.save_passes:
                        mp4_filename = str(filepath).split('.hdf5')[0] + pass_mask
                        cmd, stdout, stderr = pngs_to_mp4(
                            filename=mp4_filename,
                            image_stem=pass_mask[1:]+'_',
                            png_dir=self.png_dir,
                            size=[self._height, self._width],
                            overwrite=True,
                            remove_pngs=True,
                            use_parent_dir=False)

                    rm = subprocess.run('rm -rf ' + str(self.png_dir), shell=True)


                if self.save_meshes:
                    for o_id in self.object_ids:
                        obj_filename = str(filepath).split('.hdf5')[0] + f"_obj{o_id}.obj"
                        vertices, faces = self.object_meshes[o_id]
                        save_obj(vertices, faces, obj_filename)
            pbar.update(1)
        pbar.close()

    def trial(self,
              filepath: Path,
              temp_path: Path,
              object_info,
              trial_num: int) -> None:
        # generate a batch of trials instead of one
        """
        Run a trial. Write static and per-frame data to disk until the trial is done.

        :param filepath: The path to this trial's hdf5 file.
        :param temp_path: The path to the temporary file.
        :param trial_num: The number of the current trial.
        """
        # Clear the object IDs and other static data
        from tdw_physics.rigidbodies_dataset import get_random_xyz_transform
        if object_info == None:
            self.repeat_trial = False
            # sample distinct objects
            self.candidate_dict = dict()
            colors = [[0.01844594, 0.77508636, 0.12749255],#pink
                      [0.17443318, 0.22064707, 0.39867442],#black
                      [0.9774568,  0.87879388, 0.40082996],#orange
                      [0.75136046, 0.06584012, 0.22674323],#red
                      [0.47, 0.38,   0.901],#purple
                       ]
            distinct_masses = [0.1, 2.0, 10.0]

            random.shuffle(colors)
            random.shuffle(distinct_masses)
            for distinct_id in range(self.num_distinct_objects):
                self.candidate_dict[distinct_id] = dict()
                self.candidate_dict[distinct_id]["type"] = random.choice(self._middle_types)
                self.candidate_dict[distinct_id]["scale"] = get_random_xyz_transform(self.middle_scale_range)
                self.candidate_dict[distinct_id]["color"] = colors[distinct_id] #np.array(self.random_color(None, 0.25))
                self.candidate_dict[distinct_id]["mass"] = distinct_masses[distinct_id]

        else:
            self.repeat_trial = True

        self.trial_num_middle_objects = random.choice(range(self.num_middle_objects + 1))
        self.trial_collision_axis_length = self.spacing * (self.trial_num_middle_objects + 1)


        self.clear_static_data()
        self._trial_num = trial_num

        # Create the .hdf5 file.
        f = h5py.File(str(temp_path.resolve()), "a")

        commands = []
        # Remove asset bundles (to prevent a memory leak).
        if trial_num % 100 == 0:
            commands.append({"$type": "unload_asset_bundles"})
        frame = 0
        for interact_id in range(self.num_interactions):
            print("interact_id", interact_id)
            # Add commands to start the trial.
            commands.extend(self.get_trial_initialization_commands(interact_id))
            # Add commands to request output data.
            commands.extend(self._get_send_data_commands())

            # Send the commands and start the trial.
            r_types = ['']
            count = 0


            if interact_id == 0:
                resp = self.communicate(commands)
            self._set_segmentation_colors(resp)

            self._get_object_meshes(resp)

            # Write static data to disk.
            if interact_id == 0:
                static_group = f.create_group("static")
                self._write_static_data(static_group)

                # Add the first frame.

                frames_grp = f.create_group("frames")
                frame_grp, _, _, _ = self._write_frame(frames_grp=frames_grp, resp=resp, frame_num=frame)
                self._write_frame_labels(frame_grp, resp, -1, False)
            done = False
            # Continue the trial. Send commands, and parse output data.
            if interact_id > 0:
                for t in range(100):
                    resp = self.communicate(commands)
                import ipdb; ipdb.set_trace()
            while not done:
                frame += 1
                # print('frame %d' % frame)
                resp = self.communicate(self.get_per_frame_commands(resp, frame))
                r_ids = [OutputData.get_data_type_id(r) for r in resp[:-1]]

                # Sometimes the build freezes and has to reopen the socket.
                # This prevents such errors from throwing off the frame numbering
                if ('imag' not in r_ids) or ('tran' not in r_ids):
                    print("retrying frame %d, response only had %s" % (frame, r_ids))
                    frame -= 1
                    continue

                frame_grp, objs_grp, tr_dict, done = self._write_frame(frames_grp=frames_grp, resp=resp, frame_num=frame)

                # Write whether this frame completed the trial and any other trial-level data
                labels_grp, _, _, done = self._write_frame_labels(frame_grp, resp, frame, done)
            commands = []

        # Cleanup.
        commands = []
        for o_id in self.object_ids:
            commands.append({"$type": self._get_destroy_object_command_name(o_id),
                             "id": int(o_id)})
        self.communicate(commands)

        # Compute the trial-level metadata. Save it per trial in case of failure mid-trial loop
        if self.save_labels:
            meta = OrderedDict()
            meta = get_labels_from(f, label_funcs=self.get_controller_label_funcs(type(self).__name__), res=meta)
            self.trial_metadata.append(meta)

            # Save the trial-level metadata
            json_str =json.dumps(self.trial_metadata, indent=4)
            self.meta_file.write_text(json_str, encoding='utf-8')
            print("TRIAL %d LABELS" % self._trial_num)
            print(json.dumps(self.trial_metadata[-1], indent=4))

        # Save out the target/zone segmentation mask
        _id = f['frames']['0000']['images']['_id']
        #get PIL image
        _id_map = np.array(Image.open(io.BytesIO(np.array(_id))))
        #get colors
        zone_idx = [i for i,o_id in enumerate(self.object_ids) if o_id == self.zone_id]
        zone_color = self.object_segmentation_colors[zone_idx[0] if len(zone_idx) else 0]
        target_idx = [i for i,o_id in enumerate(self.object_ids) if o_id == self.target_id]
        target_color = self.object_segmentation_colors[target_idx[0] if len(target_idx) else 1]
        #get individual maps
        zone_map = (_id_map == zone_color).min(axis=-1, keepdims=True)
        target_map = (_id_map == target_color).min(axis=-1, keepdims=True)
        #colorize
        zone_map = zone_map * ZONE_COLOR
        target_map = target_map * TARGET_COLOR
        joint_map = zone_map + target_map
        # add alpha
        alpha = ((target_map.sum(axis=2) | zone_map.sum(axis=2)) != 0) * 255
        joint_map = np.dstack((joint_map, alpha))
        #as image
        map_img = Image.fromarray(np.uint8(joint_map))
        #save image
        map_img.save(filepath.parent.joinpath(filepath.stem+"_map.png"))

        # Close the file.
        f.close()
        # Move the file.
        try:
            temp_path.replace(filepath)
        except OSError:
            shutil.move(temp_path, filepath)


    def get_types(self,
                  objlist,
                  libraries=["models_flex.json"],
                  categories=None,
                  flex_only=True,
                  aspect_ratio_min=None,
                  aspect_ratio_max=None):

        if isinstance(objlist, str):
            objlist = [objlist]
        recs = []
        for lib in libraries:
            recs.extend(MODEL_LIBRARIES[lib].records)
        tlist = [r for r in recs if r.name in objlist]
        if categories is not None:
            if not isinstance(categories, list):
                categories = categories.split(',')
            tlist = [r for r in tlist if r.wcategory in categories]

        if flex_only:
            tlist = [r for r in tlist if r.flex == True]

        if aspect_ratio_min:
            tlist = [r for r in tlist if self.aspect_ratios(r)[0] > aspect_ratio_min]
        if aspect_ratio_max:
            tlist = [r for r in tlist if self.aspect_ratios(r)[1] < aspect_ratio_max]

        assert len(tlist), "You're trying to choose objects from an empty list"
        return tlist

    def set_probe_types(self, olist):
        tlist = self.get_types(olist, flex_only=self.flex_only)
        self._probe_types = tlist

    def set_target_types(self, olist):
        tlist = self.get_types(olist, flex_only=self.flex_only)
        self._target_types = tlist

    def set_zone_types(self, olist):
        tlist = self.get_types(olist, flex_only=self.flex_only)
        self._zone_types = tlist


    def clear_static_data(self) -> None:
        super().clear_static_data()

        ## set randomize colors
        if not self.repeat_trial:
            # don't change the target
            if self._random_zone_color:
                self.zone_color = None
        if self._random_target_color:
            self.target_color = None
        if self._random_probe_color:
            self.probe_color = None

        ## scenario-specific metadata: object types and drop position
        self.target_type = None
        self.target_rotation = None
        self.target_position = None
        self.target_delta_position = None
        self.replace_target = False

        self.probe_type = None
        self.probe_mass = None
        self.push_force = None
        self.push_position = None
        self.force_wait = None

    @staticmethod
    def get_controller_label_funcs(classname = 'Dominoes'):

        funcs = super(Dominoes, Dominoes).get_controller_label_funcs(classname)
        funcs += get_all_label_funcs()

        def room(f):
            return str(np.array(f['static']['room']))
        def trial_seed(f):
            return int(np.array(f['static']['trial_seed']))
        def num_distractors(f):
            try:
                return int(len(f['static']['distractors']))
            except KeyError:
                return int(0)
        def num_occluders(f):
            try:
                return int(len(f['static']['occluders']))
            except KeyError:
                return int(0)
        def push_time(f):
            try:
                return int(np.array(f['static']['push_time']))
            except KeyError:
                return int(0)
        funcs += [room, trial_seed, push_time, num_distractors, num_occluders]

        return funcs

    def get_field_of_view(self) -> float:
        return 55

    def get_scene_initialization_commands(self) -> List[dict]:
        if self.room == 'box':
            add_scene = self.get_add_scene(scene_name="box_room_2018")
        elif self.room == 'tdw':
            add_scene = self.get_add_scene(scene_name="tdw_room")
        elif self.room == 'house':
            add_scene = self.get_add_scene(scene_name='archviz_house')
        return [add_scene,
                {"$type": "set_aperture",
                 "aperture": 8.0},
                {"$type": "set_post_exposure",
                 "post_exposure": 0.4},
                {"$type": "set_ambient_occlusion_intensity",
                 "intensity": 0.175},
                {"$type": "set_ambient_occlusion_thickness_modifier",
                 "thickness": 3.5}]

    def get_trial_initialization_commands(self, interact_id=0) -> List[dict]:
        commands = []

        # randomization across trials
        if not(self.randomize):
            self.trial_seed = (self.MAX_TRIALS * self.seed) + self._trial_num
            random.seed(self.trial_seed)
        else:
            self.trial_seed = -1 # not used
        if interact_id == 0:
            # Choose and place the target zone.
            commands.extend(self._place_target_zone())

            # Choose and place a target object
            commands.extend(self._sample_all_objects())

            from magnebot import Magnebot, Arm

            m = Magnebot()
            import ipdb; ipdb.set_trace()


            # magnebot_position = {
            # 'x': 0,
            # 'y': 1,
            # 'z': 0
            # }
            # commands.extend([{"$type": "add_magnebot",
            #                   "position": magnebot_position if magnebot_position is not None else TDWUtils.VECTOR3_ZERO},
            #                  {"$type": "send_magnebots",
            #                   "frequency": "always"}])

        else:
            self.shuffle_id = [1,2,0]



        commands.extend(self._place_target_object(interact_id))

        #self.model_names.append(record.name)
        #self.scales.append(data['scale'])
        #self.colors

        # Set the probe color
        if self.probe_color is None:
            self.probe_color = self.target_color if (self.monochrome and self.match_probe_and_target_color) else None

        # Choose, place, and push a probe object.
        commands.extend(self._place_and_push_probe_object(interact_id))

        # Build the intermediate structure that captures some aspect of "intuitive physics."
        commands.extend(self._build_intermediate_structure(interact_id))

        if interact_id == 0:
            # Teleport the avatar to a reasonable position based on the drop height.
            a_pos = self.get_random_avatar_position(radius_min=self.camera_radius_range[0],
                                                    radius_max=self.camera_radius_range[1],
                                                    angle_min=self.camera_min_angle,
                                                    angle_max=self.camera_max_angle,
                                                    y_min=self.camera_min_height,
                                                    y_max=self.camera_max_height,
                                                    center=TDWUtils.VECTOR3_ZERO,
                                                    reflections=self.camera_left_right_reflections)

            # Set the camera parameters
            self._set_avatar_attributes(a_pos)

            commands.extend([
                {"$type": "teleport_avatar_to",
                 "position": self.camera_position},
                {"$type": "look_at_position",
                 "position": self.camera_aim},
                {"$type": "set_focus_distance",
                 "focus_distance": TDWUtils.get_distance(a_pos, self.camera_aim)}
            ])


            # Place distractor objects in the background
            commands.extend(self._place_background_distractors())

            # Place occluder objects in the background
            commands.extend(self._place_occluders())

            # test mode colors
            if self.use_test_mode_colors:
                self._set_test_mode_colors(commands)

        return commands

    def get_per_frame_commands(self, resp: List[bytes], frame: int) -> List[dict]:

        if (self.force_wait != 0) and frame == self.force_wait:
            if self.PRINT:
                print("applied %s at time step %d" % (self.push_cmd, frame))
            return [self.push_cmd]
        else:
            return []

    def _write_static_data(self, static_group: h5py.Group) -> None:
        super()._write_static_data(static_group)

        # randomization
        try:
            static_group.create_dataset("room", data=self.room)
        except (AttributeError,TypeError):
            pass
        try:
            static_group.create_dataset("seed", data=self.seed)
        except (AttributeError,TypeError):
            pass
        try:
            static_group.create_dataset("randomize", data=self.randomize)
        except (AttributeError,TypeError):
            pass
        try:
            static_group.create_dataset("trial_seed", data=self.trial_seed)
        except (AttributeError,TypeError):
            pass
        try:
            static_group.create_dataset("trial_num", data=self._trial_num)
        except (AttributeError,TypeError):
            pass

        ## which objects are the zone, target, and probe
        try:
            static_group.create_dataset("zone_id", data=self.zone_id)
        except (AttributeError,TypeError):
            pass
        try:
            static_group.create_dataset("target_id", data=self.target_id)
        except (AttributeError,TypeError):
            pass
        try:
            static_group.create_dataset("probe_id", data=self.probe_id)
        except (AttributeError,TypeError):
            pass

        if self.use_ramp:
            static_group.create_dataset("ramp_id", data=self.ramp_id)
            if self.ramp_base_height > 0.0:
                static_group.create_dataset("ramp_base_height", data=float(self.ramp_base_height))
                static_group.create_dataset("ramp_base_id", data=self.ramp_base_id)

        ## color and scales of primitive objects
        try:
            static_group.create_dataset("target_type", data=self.target_type)
        except (AttributeError,TypeError):
            pass
        try:
            static_group.create_dataset("target_rotation", data=xyz_to_arr(self.target_rotation))
        except (AttributeError,TypeError):
            pass
        try:
            static_group.create_dataset("probe_type", data=self.probe_type)
        except (AttributeError,TypeError):
            pass
        try:
            static_group.create_dataset("probe_mass", data=self.probe_mass)
        except (AttributeError,TypeError):
            pass
        try:
            static_group.create_dataset("push_force", data=xyz_to_arr(self.push_force))
        except (AttributeError,TypeError):
            pass
        try:
            static_group.create_dataset("push_position", data=xyz_to_arr(self.push_position))
        except (AttributeError,TypeError):
            pass
        try:
            static_group.create_dataset("push_time", data=int(self.force_wait))
        except (AttributeError,TypeError):
            pass

        # distractors and occluders
        try:
            static_group.create_dataset("distractors", data=[r.name.encode('utf8') for r in self.distractors.values()])
        except (AttributeError,TypeError):
            pass
        try:
            static_group.create_dataset("occluders", data=[r.name.encode('utf8') for r in self.occluders.values()])
        except (AttributeError,TypeError):
            pass

    def _write_frame(self,
                     frames_grp: h5py.Group,
                     resp: List[bytes],
                     frame_num: int) -> \
            Tuple[h5py.Group, h5py.Group, dict, bool]:
        frame, objs, tr, sleeping = super()._write_frame(frames_grp=frames_grp,
                                                         resp=resp,
                                                         frame_num=frame_num)
        # If this is a stable structure, disregard whether anything is actually moving.
        return frame, objs, tr, sleeping and not (frame_num < 150)

    def _update_target_position(self, resp: List[bytes], frame_num: int) -> None:
        if frame_num <= 0:
            self.target_delta_position = xyz_to_arr(TDWUtils.VECTOR3_ZERO)
        elif 'tran' in [OutputData.get_data_type_id(r) for r in resp[:-1]]:
            target_position_new = self.get_object_position(self.target_id, resp) or self.target_position
            try:
                self.target_delta_position += (target_position_new - xyz_to_arr(self.target_position))
                self.target_position = arr_to_xyz(target_position_new)
            except TypeError:
                print("Failed to get a new object position, %s" % target_position_new)

    def _write_frame_labels(self,
                            frame_grp: h5py.Group,
                            resp: List[bytes],
                            frame_num: int,
                            sleeping: bool) -> Tuple[h5py.Group, List[bytes], int, bool]:

        labels, resp, frame_num, done = super()._write_frame_labels(frame_grp, resp, frame_num, sleeping)

        # Whether this trial has a target or zone to track
        has_target = (not self.remove_target) or self.replace_target
        has_zone = not self.remove_zone
        labels.create_dataset("has_target", data=has_target)
        labels.create_dataset("has_zone", data=has_zone)
        if not (has_target or has_zone):
            return labels, resp, frame_num, done

        # Whether target moved from its initial position, and how much
        if has_target:
            self._update_target_position(resp, frame_num)
            has_moved = np.sqrt((self.target_delta_position**2).sum()) > self.target_motion_thresh
            labels.create_dataset("target_delta_position", data=self.target_delta_position)
            labels.create_dataset("target_has_moved", data=has_moved)

            # Whether target has fallen to the ground
            c_points, c_normals = self.get_object_environment_collision(
                self.target_id, resp)

            if frame_num <= 0:
                self.target_on_ground = False
                self.target_ground_contacts = c_points
            elif len(c_points) == 0:
                self.target_on_ground = False
            elif len(c_points) != len(self.target_ground_contacts):
                self.target_on_ground = True
            elif any([np.sqrt(((c_points[i] - self.target_ground_contacts[i])**2).sum()) > self.target_motion_thresh \
                      for i in range(min(len(c_points), len(self.target_ground_contacts)))]):
                self.target_on_ground = True

            labels.create_dataset("target_on_ground", data=self.target_on_ground)

        # Whether target has hit the zone
        if has_target and has_zone:
            c_points, c_normals = self.get_object_target_collision(
                self.target_id, self.zone_id, resp)
            target_zone_contact = bool(len(c_points))
            labels.create_dataset("target_contacting_zone", data=target_zone_contact)

        return labels, resp, frame_num, done

    def is_done(self, resp: List[bytes], frame: int) -> bool:
        return frame > 300

    def get_rotation(self, rot_range):
        if rot_range is None:
            return {"x": 0,
                    "y": random.uniform(0, 360),
                    "z": 0.}
        else:
            return get_random_xyz_transform(rot_range)

    def get_y_rotation(self, rot_range):
        if rot_range is None:
            return self.get_rotation(rot_range)
        else:
            return {"x": 0.,
                    "y": random.uniform(*get_range(rot_range)),
                    "z": 0.}

    def get_push_force(self, scale_range, angle_range, yforce = [0,0]):
        #sample y force component
        yforce = random.uniform(*yforce)
        # rotate a unit vector initially pointing in positive-x direction
        theta = np.radians(random.uniform(*get_range(angle_range)))
        push = np.array([np.cos(theta), yforce, np.sin(theta)])

        # scale it
        push *= random.uniform(*get_range(scale_range))

        # convert to xyz
        return arr_to_xyz(push)

    def _get_push_cmd(self, o_id, position_or_particle=None):
        if position_or_particle is None:
            cmd = {
                "$type": "apply_force_to_object",
                "force": self.push_force,
                "id": o_id}
        else:
            cmd = {
                "$type": "apply_force_at_position",
                "force": self.push_force,
                "position": position_or_particle,
                "id": o_id}
        return cmd

    def _get_zone_location(self, scale):
        return {
            "x": 0.5 * self.trial_collision_axis_length + scale["x"] + 0.1,
            "y": 0.0 if not self.remove_zone else 10.0,
            "z": 0.0 if not self.remove_zone else 10.0
        }

    def element_wise_equal(self, x, y):
        for ele in x:
            if x[ele] != y[ele]:
                return False
        return True
    def _place_target_zone(self) -> List[dict]:

        # create a target zone (usually flat, with same texture as room)
        if not self.repeat_trial: # sample from scratch

            record, data = self.random_primitive(self._zone_types,
                                                 scale=self.zone_scale_range,
                                                 color=self.zone_color,
                                                 add_data=False
            )
            o_id, scale, rgb = [data[k] for k in ["id", "scale", "color"]]
            self.zone = record
            self.zone_type = data["name"]
            self.zone_color = rgb
            self.zone_id = o_id
            self.zone_scale = scale
        else:
            # dry pass to get the obj id counter correct
            record, data = self.random_primitive([self.zone],
                                                 scale=self.zone_scale,
                                                 color=self.zone_color,
                                                 add_data=False
            )
            o_id, scale, rgb = [data[k] for k in ["id", "scale", "color"]]
            assert(record == self.zone)
            assert(o_id == self.zone_id)
            assert(self.element_wise_equal(scale, self.zone_scale))
            assert(self.element_wise_equal(scale, self.zone_scale))
            assert(np.array_equal(rgb, self.zone_color))


        if any((s <= 0 for s in scale.values())):
            self.remove_zone = True
            self.scales = self.scales[:-1]
            self.colors = self.colors[:-1]
            self.model_names = self.model_names[:-1]
        self.distinct_ids = np.append(self.distinct_ids, -1)
        # place it just beyond the target object with an effectively immovable mass and high friction
        commands = []
        commands.extend(
            self.add_primitive(
                record=record,
                position=(self.zone_location or self._get_zone_location(scale)),
                rotation=TDWUtils.VECTOR3_ZERO,
                scale=scale,
                material=self.zone_material,
                color=rgb,
                mass=500,
                scale_mass=False,
                dynamic_friction=self.zone_friction,
                static_friction=(10.0 * self.zone_friction),
                bounciness=0,
                o_id=o_id,
                add_data=(not self.remove_zone),
                make_kinematic=True # zone shouldn't move
            ))

        # get rid of it if not using a target object
        if self.remove_zone:
            commands.append(
                {"$type": self._get_destroy_object_command_name(o_id),
                 "id": int(o_id)})
            self.object_ids = self.object_ids[:-1]

        return commands


    def _sample_all_objects(self) -> List[dict]:

        total_num_objects_to_sample = 2 + self.num_middle_objects
        self.type_scale_mass_color = []
        self.oid_scale_rgb_record_name = []
        self.positions = []
        commands = []
        for i in range(total_num_objects_to_sample):
            distinct_id = random.choice(range(self.num_distinct_objects))
            self.distinct_ids = np.append(self.distinct_ids, distinct_id)

            type_ = self.candidate_dict[distinct_id]["type"]
            scale_ = self.candidate_dict[distinct_id]["scale"]
            mass_ = self.candidate_dict[distinct_id]["mass"]
            color_ = self.candidate_dict[distinct_id]["color"]

            record, data = self.random_primitive([type_],
                                                 scale=scale_,
                                                 color=color_,
                                                 add_data=False
            )
            o_id = data["id"]
            scale = data["scale"]
            rgb = data["color"]
            self.oid_scale_rgb_record_name.append((o_id, scale, rgb, record, data["name"]))
            self.type_scale_mass_color.append((type_, scale_, mass_, color_))

            position = {
                "x": 0.5 * self.trial_collision_axis_length - i * self.spacing,
                "y": 0.,
                "z": 0.
            }
            rotation = {"x": 0,
                    "y": 0.,
                    "z": 0.}
            self.positions.append(position)
            commands.extend(
                self.add_primitive(
                    record=record,
                    position=position,
                    rotation=rotation,
                    scale=scale,
                    material=self.target_material,
                    color=rgb,
                    mass=mass_, #2.0,
                    scale_mass=False,
                    dynamic_friction=0.5,
                    static_friction=0.5,
                    bounciness=0.0,
                    o_id=o_id,
                    add_data=1,
                    make_kinematic=False
                ))
        return commands
    def _place_target_object(self, interact_id) -> List[dict]:
        """
        Place a primitive object at one end of the collision axis.
        """

        # create a target object
        #if not self.repeat_trial: # sample from scratch
        target_type, target_scale, target_mass, target_color = self.type_scale_mass_color[0]
        o_id, scale, rgb, record, name = self.oid_scale_rgb_record_name[0]
        #o_id, scale, rgb = [data[k] for k in ["id", "scale", "color"]]

        self.target = record
        self.target_type = name
        self.target_color = rgb
        self.target_scale = self.middle_scale = scale
        self.target_id = o_id


        if any((s <= 0 for s in scale.values())):
            self.remove_target = True

        # Where to put the target
        if self.target_rotation is None:
            self.target_rotation = self.get_rotation(self.target_rotation_range)

        if self.target_position is None:
            self.target_position = self.positions[0]

        # Commands for adding hte object


        # If this scene won't have a target
        commands = []
        if self.remove_target:
            commands.append(
                {"$type": self._get_destroy_object_command_name(o_id),
                 "id": int(o_id)})
            self.object_ids = self.object_ids[:-1]
        if interact_id > 0: # drag target object to the right position
            current_id = self.shuffle_id[0]
            o_id, scale, rgb, record, name = self.oid_scale_rgb_record_name[current_id]
            # commands.extend([{
            #     "$type": "apply_force_to_object",
            #     "force":  {"x": 0, "y": 0, "z": 1},
            #     "id": o_id}])
            self.communicate([{
                "$type": "teleport_object",
                "positions":  {"x": 0, "y": 0, "z": 2},
                "id": o_id,
                "physics": False}])


        return commands

    def _place_and_push_probe_object(self, interact_id=0) -> List[dict]:
        """
        Place a probe object at the other end of the collision axis, then apply a force to push it.
        """
        # create a target object
        #if not self.repeat_trial: # sample from scratch
        probe_type, probe_scale, probe_mass, probe_color = self.type_scale_mass_color[-1]
        o_id, scale, rgb, record, name = self.oid_scale_rgb_record_name[-1]


        self.probe = record
        self.probe_type = name
        self.probe_scale = scale
        self.probe_id = o_id
        self.probe_mass = probe_mass

        # Add the object with random physics values
        commands = []

        ### better sampling of random physics values
        #self.probe_mass = random.uniform(self.probe_mass_range[0], self.probe_mass_range[1])
        self.probe_initial_position = self.positions[-1]
        rot = self.get_y_rotation(self.probe_rotation_range)

        if self.use_ramp:
            commands.extend(self._place_ramp_under_probe())

        if self.probe_has_friction:
            probe_physics_info = {'dynamic_friction': 0.1, 'static_friction': 0.1, 'bounciness': 0.6}
        else:
            probe_physics_info = {'dynamic_friction': 0.5, 'static_friction': 0.5, 'bounciness': 0}

            #probe_physics_info = {'dynamic_friction': 0.01, 'static_friction': 0.01, 'bounciness': 0}

        # Set its collision mode
        commands.extend([
            {"$type": "set_object_drag",
             "id": o_id,
             "drag": 0, "angular_drag": 0}])


        # Apply a force to the probe object
        self.push_force = self.get_push_force(
            scale_range=self.probe_mass * np.array(self.force_scale_range),
            angle_range=self.force_angle_range)
        self.push_force = self.rotate_vector_parallel_to_floor(
            self.push_force, -rot['y'], degrees=True)

        self.push_position = self.probe_initial_position

        if self.PRINT:
            print("PROBE MASS", self.probe_mass)
            print("PUSH FORCE", self.push_force)
        if self.use_ramp:
            self.push_cmd = self._get_push_cmd(o_id, None)
        else:
            self.push_position = {
                k:v+self.force_offset[k]*self.rotate_vector_parallel_to_floor(
                    self.probe_scale, rot['y'])[k]
                for k,v in self.push_position.items()}
            self.push_position = {
                k:v+random.uniform(-self.force_offset_jitter, self.force_offset_jitter)
                for k,v in self.push_position.items()}

            self.push_cmd = self._get_push_cmd(o_id, self.push_position)

        # decide when to apply the force
        self.force_wait = int(random.uniform(*get_range(self.force_wait_range)))
        if self.PRINT:
            print("force wait", self.force_wait)

        if self.force_wait == 0:
            commands.append(self.push_cmd)

        return commands

    def _place_ramp_under_probe(self) -> List[dict]:

        cmds = []

        # ramp params
        self.ramp = random.choice(self.DEFAULT_RAMPS)
        rgb = self.ramp_color or self.random_color(exclude=self.target_color)
        ramp_pos = copy.deepcopy(self.probe_initial_position)
        ramp_pos['y'] = self.zone_scale['y'] if not self.remove_zone else 0.0 # don't intersect w zone
        ramp_rot = self.get_y_rotation([180,180])
        ramp_id = self._get_next_object_id()

        self.ramp_pos = ramp_pos
        self.ramp_rot = ramp_rot
        self.ramp_id = ramp_id

        # figure out scale
        r_len, r_height, r_dep = self.get_record_dimensions(self.ramp)
        scale_x = (0.75 * self.collision_axis_length) / r_len
        if self.ramp_scale is None:
            self.ramp_scale = arr_to_xyz([scale_x, self.scale_to(r_height, 1.5), 0.75 * scale_x])
        self.ramp_end_x = self.ramp_pos['x'] + self.ramp_scale['x'] * r_len * 0.5

        # optionally add base
        cmds.extend(self._add_ramp_base_to_ramp(color=rgb))

        # add the ramp
        cmds.extend(
            self.add_ramp(
                record = self.ramp,
                position=self.ramp_pos,
                rotation=self.ramp_rot,
                scale=self.ramp_scale,
                material=self.ramp_material,
                color=rgb,
                o_id=self.ramp_id,
                add_data=True,
                **self.ramp_physics_info
            ))

        # need to adjust probe height as a result of ramp placement
        self.probe_initial_position['x'] -= 0.5 * self.ramp_scale['x'] * r_len - 0.15
        self.probe_initial_position['y'] = self.ramp_scale['y'] * r_height + self.ramp_base_height + self.probe_initial_position['y']

        return cmds

    def _add_ramp_base_to_ramp(self, color=None) -> None:

        cmds = []

        if color is None:
            color = self.random_color(exclude=self.target_color)

        self.ramp_base_height = random.uniform(*get_range(self.ramp_base_height_range))
        if self.ramp_base_height < 0.01:
            self.ramp_base_scale = copy.deepcopy(self.ramp_scale)
            return []

        self.ramp_base = self.CUBE
        r_len, r_height, r_dep = self.get_record_dimensions(self.ramp)
        self.ramp_base_scale = arr_to_xyz([
            float(self.ramp_scale['x'] * r_len),
            float(self.ramp_base_height),
            float(self.ramp_scale['z'] * r_dep)])
        self.ramp_base_id = self._get_next_object_id()

        # add the base
        ramp_base_physics_info = {
            'mass': 500,
            'dynamic_friction': 0.01,
            'static_friction': 0.01,
            'bounciness': 0}
        if self.ramp_physics_info.get('dynamic_friction', None) is not None:
            ramp_base_physics_info.update(self.ramp_physics_info)
        cmds.extend(
            RigidbodiesDataset.add_physics_object(
                self,
                record=self.ramp_base,
                position=copy.deepcopy(self.ramp_pos),
                rotation=TDWUtils.VECTOR3_ZERO,
                o_id=self.ramp_base_id,
                add_data=True,
                **ramp_base_physics_info))

        # scale it, color it, fix it
        cmds.extend(
            self.get_object_material_commands(
                self.ramp_base, self.ramp_base_id, self.get_material_name(self.ramp_material)))
        cmds.extend([
            {"$type": "scale_object",
             "scale_factor": self.ramp_base_scale,
             "id": self.ramp_base_id},
            {"$type": "set_color",
             "color": {"r": color[0], "g": color[1], "b": color[2], "a": 1.},
             "id": self.ramp_base_id},
            {"$type": "set_object_collision_detection_mode",
             "mode": "continuous_speculative",
             "id": self.ramp_base_id},
            {"$type": "set_kinematic_state",
             "id": self.ramp_base_id,
             "is_kinematic": True,
             "use_gravity": True}])

        # add data
        self.model_names.append(self.ramp_base.name)
        self.scales.append(self.ramp_base_scale)
        self.colors = np.concatenate([self.colors, np.array(color).reshape((1,3))], axis=0)

        # raise the ramp
        self.ramp_pos['y'] += self.ramp_base_scale['y']

        return cmds


    def _replace_target_with_object(self, record, data):
        self.target = record
        self.target_type = data["name"]
        self.target_color = data["color"]
        self.target_scale = data["scale"]
        self.target_id = data["id"]

        self.replace_target = True

    def _set_test_mode_colors(self, commands) -> None:

        tcolor = {'r': 1.0, 'g': 0.0, 'b': 0.0, 'a': 1.0}
        zcolor = {'r': 1.0, 'g': 1.0, 'b': 0.0, 'a': 1.0}
        exclude = {'r': 1.0, 'g': 0.0, 'b': 0.0}
        exclude_range = 0.25

        for c in commands:
            if "set_color" in c.values():
                o_id = c['id']
                if o_id == self.target_id:
                    c['color'] = tcolor
                elif o_id == self.zone_id:
                    c['color'] = zcolor
                elif any((np.abs(exclude[k] - c['color'][k]) < exclude_range for k in exclude.keys())):
                    rgb = self.random_color_from_rng(exclude=[exclude[k] for k in ['r','g','b']],
                                                     exclude_range=exclude_range,
                                                     seed=self.trial_seed)
                    c['color'] = {'r': rgb[0], 'g': rgb[1], 'b': rgb[2], 'a': 1.0}

    def _build_intermediate_structure(self) -> List[dict]:
        """
        Abstract method for building a physically interesting intermediate structure between the probe and the target.
        """
        commands = []
        return commands

    def _set_distractor_objects(self) -> None:

        self.distractors = OrderedDict()
        for i in range(self.num_distractors):
            record, data = self.random_model(self.distractor_types, add_data=True)
            self.distractors[data['id']] = record

    def _set_occluder_objects(self) -> None:
        self.occluders = OrderedDict()
        for i in range(self.num_occluders):
            record, data = self.random_model(self.occluder_types, add_data=True)
            self.occluders[data['id']] = record

    @staticmethod
    def get_record_dimensions(record: ModelRecord) -> List[float]:
        length = np.abs(record.bounds['left']['x'] - record.bounds['right']['x'])
        height = np.abs(record.bounds['top']['y'] - record.bounds['bottom']['y'])
        depth = np.abs(record.bounds['front']['z'] - record.bounds['back']['z'])
        return (length, height, depth)

    @staticmethod
    def aspect_ratios(record: ModelRecord) -> List[float]:
        l,h,d = Dominoes.get_record_dimensions(record)
        a1 = float(h) / l
        a2 = float(h) / d
        min_ar = min(a1, a2)
        max_ar = max(a1, a2)
        return (min_ar, max_ar)

    @staticmethod
    def scale_to(current_scale : float, target_scale : float) -> float:

        return target_scale / current_scale

    def _set_avatar_attributes(self, avatar_position) -> None:

        a_pos = avatar_position

        self.camera_position = a_pos
        self.camera_rotation = np.degrees(np.arctan2(a_pos['z'], a_pos['x']))
        dist = TDWUtils.get_distance(a_pos, self.camera_aim)
        self.camera_altitude = np.degrees(np.arcsin((a_pos['y'] - self.camera_aim['y'])/dist))
        camera_ray = np.array([self.camera_position['x'], 0., self.camera_position['z']])
        self.camera_radius = np.linalg.norm(camera_ray)
        camera_ray /= np.linalg.norm(camera_ray)
        self.camera_ray = arr_to_xyz(camera_ray)

        if self.PRINT:
            print("camera distance", self.camera_radius)
            print("camera ray", self.camera_ray)
            print("camera angle", self.camera_rotation)
            print("camera altitude", self.camera_altitude)
            print("camera position", self.camera_position)


    def _set_occlusion_attributes(self) -> None:

        self.occluder_angular_spacing = 10
        self.occlusion_distance_fraction = [0.6, 0.8]
        self.occluder_rotation_jitter = 30.
        self.occluder_min_z = self.middle_scale['z'] + 0.25
        self.occluder_min_size = 0.25
        self.occluder_max_size = 1.5
        self.rescale_occluder_height = True

    def _get_occluder_position_pose_scale(self, record, unit_position_vector):
        """
        Given a unit vector direction in world coordinates, adjust in a Controller-specific
        manner to avoid interactions with the physically relevant objects.
        """

        o_len, o_height, o_dep = self.get_record_dimensions(record)

        ## get the occluder pose
        ang = self.camera_rotation
        rot = self.get_y_rotation(
            [ang - self.occluder_rotation_jitter, ang + self.occluder_rotation_jitter])
        bounds = {'x': o_len, 'y': o_height, 'z': o_dep}
        bounds_rot = self.rotate_vector_parallel_to_floor(bounds, rot['y'])
        bounds = {k:np.maximum(np.abs(v), bounds[k]) for k,v in bounds_rot.items()}

        # make sure it's in a reasonable size range
        size = max(list(bounds.values()))
        size = np.minimum(np.maximum(size, self.occluder_min_size), self.occluder_max_size)
        scale = size / max(list(bounds.values()))
        bounds = self.scale_vector(bounds, scale)

        ## choose the initial position of the object, before adjustment
        occ_distance = random.uniform(*get_range(self.occlusion_distance_fraction))
        pos = self.scale_vector(
            unit_position_vector, occ_distance * self.camera_radius)

        ## reposition and rescale it so it's not too close to the "physical dynamics" axis (z)
        if np.abs(pos['z']) < (self.occluder_min_z + self.occluder_min_size):
            pos.update({'z' : np.sign(pos['z']) * (self.occluder_min_z + self.occluder_min_size)})

        reach_z = np.abs(pos['z']) - 0.5 * bounds['z']
        if reach_z < self.occluder_min_z: # scale down
            scale_z = (np.abs(pos['z']) - self.occluder_min_z) / (0.5 * bounds['z'])
        else:
            scale_z = 1.0
        bounds = self.scale_vector(bounds, scale_z)
        scale *= scale_z

        ## reposition and rescale it so it's not too close to other occluders
        if self.num_occluders > 1 and len(self.occluder_positions):
            last_pos_x = self.occluder_positions[-1]['x']
            last_bounds_x = self.occluder_dimensions[-1]['x']

            if (pos['x'] + self.occluder_min_size) > (last_pos_x - 0.5 * last_bounds_x):
                pos.update({'x': (last_pos_x - 0.5 * last_bounds_x) - self.occluder_min_size})

            reach_x = pos['x'] + 0.5 * bounds['x']
            if reach_x > (last_pos_x - 0.5 * last_bounds_x): # scale down
                scale_x = (last_pos_x - 0.5 * last_bounds_x - pos['x']) / (0.5 * bounds['x'])
            else:
                scale_x = 1.0

            bounds = self.scale_vector(bounds, scale_x)
            scale *= scale_x

        # do some trigonometry to figure out the scale of the occluder
        if self.rescale_occluder_height:
            occ_dist = np.sqrt(pos['x']**2 + pos['z']**2)
            occ_target_height = self.camera_aim['y'] + occ_dist * np.tan(np.radians(self.camera_altitude))
            occ_target_height *= self.occlusion_scale
            occ_target_height = np.minimum(occ_target_height, self.occluder_max_size)
            scale_y = occ_target_height / bounds['y']
            scale_y = np.minimum(
                scale_y, (np.abs(pos['z']) - self.occluder_min_z) / (0.5 * bounds['z']))

            bounds = self.scale_vector(bounds, scale_y)
            scale *= scale_y
        scale = arr_to_xyz([scale] * 3)

        self.occluder_positions.append(pos)
        self.occluder_dimensions.append(bounds)

        return (pos, rot, scale)

    def _set_distractor_attributes(self) -> None:

        self.distractor_angular_spacing = 15
        self.distractor_distance_fraction = [0.4,1.0]
        self.distractor_rotation_jitter = 30
        self.distractor_min_z = self.middle_scale['z'] + 0.25
        self.distractor_min_size = 0.5
        self.distractor_max_size = 1.5

    def _get_distractor_position_pose_scale(self, record, unit_position_vector):

        d_len, d_height, d_dep = self.get_record_dimensions(record)

        ## get distractor pose and initial bounds
        ang = 0 if (self.camera_rotation > 0) else 180
        rot = self.get_y_rotation(
            [ang - self.distractor_rotation_jitter, ang + self.distractor_rotation_jitter])
        bounds = {'x': d_len, 'y': d_height, 'z': d_dep}
        bounds_rot = self.rotate_vector_parallel_to_floor(bounds, rot['y'])
        bounds = {k:np.maximum(np.abs(v), bounds[k]) for k,v in bounds_rot.items()}

        ## make sure it's in a reasonable size range
        size = max(list(bounds.values()))
        size = np.minimum(np.maximum(size, self.distractor_min_size), self.distractor_max_size)
        scale = size / max(list(bounds.values()))
        bounds = self.scale_vector(bounds, scale)

        ## choose the initial position of the object
        distract_distance = random.uniform(*get_range(self.distractor_distance_fraction))
        pos = self.scale_vector(
            unit_position_vector, distract_distance * self.camera_radius)

        ## reposition and rescale it away from the "physical dynamics axis"
        if np.abs(pos['z']) < (self.distractor_min_z + self.distractor_min_size):
            pos.update({'z': np.sign(pos['z']) * (self.distractor_min_z + self.distractor_min_size)})

        reach_z = np.abs(pos['z']) - 0.5 * bounds['z']
        if reach_z < self.distractor_min_z: # scale down
            scale_z = (np.abs(pos['z']) - self.distractor_min_z) / (0.5 * bounds['z'])
        else:
            scale_z = 1.0
        bounds = self.scale_vector(bounds, scale_z)
        scale *= scale_z

        ## reposition and rescale it so it's not too close to other distractors
        if self.num_distractors > 1 and len(self.distractor_positions):
            last_pos_x = self.distractor_positions[-1]['x']
            last_bounds_x = self.distractor_dimensions[-1]['x']

            if (pos['x'] + self.distractor_min_size) > (last_pos_x - 0.5 * last_bounds_x):
                pos.update({'x': (last_pos_x - 0.5 * last_bounds_x) - self.distractor_min_size})

            reach_x = pos['x'] + 0.5 * bounds['x']
            if reach_x > (last_pos_x - 0.5 * last_bounds_x): # scale down
                scale_x = (last_pos_x - 0.5 * last_bounds_x - pos['x']) / (0.5 * bounds['x'])
            else:
                scale_x = 1.0

            bounds = self.scale_vector(bounds, scale_x)
            scale *= scale_x

        scale = arr_to_xyz([scale] * 3)


        self.distractor_positions.append(pos)
        self.distractor_dimensions.append(bounds)

        return (pos, rot, scale)

    def _place_background_distractors(self,z_pos_scale = 4.) -> List[dict]:
        """
        Put one or more objects in the background of the scene; they will not interfere with trial dynamics
        """

        commands = []

        # randomly sample distractors and give them obj_ids
        self._set_distractor_objects()

        # set the distractor attributes
        self._set_distractor_attributes()
        self.distractor_positions = self.distractor_dimensions = []

        # distractors will be placed opposite camera
        opposite = np.array([-self.camera_position['x'], 0., -self.camera_position['z']])
        opposite /= np.linalg.norm(opposite)
        opposite = arr_to_xyz(opposite)

        max_theta = self.distractor_angular_spacing * (self.num_distractors - 1) * np.sign(opposite['z'])
        thetas = np.linspace(-max_theta, max_theta, self.num_distractors)
        for i, o_id in enumerate(self.distractors.keys()):
            record = self.distractors[o_id]


            theta = thetas[i]
            pos_unit = self.rotate_vector_parallel_to_floor(opposite, theta)

            pos, rot, scale = self._get_distractor_position_pose_scale(record, pos_unit)

            # add the object
            commands.append(
                self.add_transforms_object(
                    record=record,
                    position=pos,
                    rotation=rot,
                    o_id=o_id,
                    add_data=True))

            # give it a color and texture if it's a primitive
            # make sure it doesn't have the same color as the target object
            rgb = self.random_color(exclude=(self.target_color if not self._random_target_color else None), exclude_range=0.5)
            if record.name in PRIMITIVE_NAMES:
                commands.extend(
                    self.get_object_material_commands(
                        record, o_id, self.get_material_name(self.target_material)))
                commands.append(
                    {"$type": "set_color",
                     "color": {"r": rgb[0], "g": rgb[1], "b": rgb[2], "a": 1.},
                     "id": o_id})

            # scale = arr_to_xyz([1.,1.,1.])
            commands.extend([
                {"$type": "scale_object",
                 "scale_factor": scale,
                 "id": o_id}
            ])

            if self.no_moving_distractors:
                commands.extend([
                    {"$type": "set_object_collision_detection_mode",
                     "mode": "discrete",
                     "id": o_id},
                    {"$type": "set_kinematic_state",
                     "id": o_id,
                     "is_kinematic": True,
                     "use_gravity": True}])

            # add the metadata
            self.colors = np.concatenate([self.colors, np.array(rgb).reshape((1,3))], axis=0)
            self.scales.append(scale)

            if self.PRINT:
                print("distractor record", record.name)
                print("distractor category", record.wcategory)
                print("distractor position", pos)
                print("distractor scale", scale)

        return commands

    def _place_occluders(self, z_pos_scale = 4.0) -> List[dict]:
        """
        Put one or more objects in the foreground to occlude the intermediate part of the scene
        """

        commands = []

        # randomly sample occluders and give them obj_ids
        self._set_occluder_objects()

        # set the attributes that determine position, pose, scale for this controller
        self._set_occlusion_attributes()

        # path to camera
        max_theta = self.occluder_angular_spacing * (self.num_occluders - 1)
        thetas = np.linspace(-max_theta, max_theta, self.num_occluders)
        self.occluder_positions = self.occluder_dimensions = []
        for i, o_id in enumerate(self.occluders.keys()):
            record = self.occluders[o_id]

            # set a position
            theta = thetas[i]
            pos_unit = self.rotate_vector_parallel_to_floor(self.camera_ray, theta)

            pos, rot, scale = self._get_occluder_position_pose_scale(record, pos_unit)

            # add the occluder
            commands.append(
                self.add_transforms_object(
                    record=record,
                    position=pos,
                    rotation=rot,
                    o_id=o_id,
                    add_data=True))

            # give it a texture if it's a primitive
            # make sure it doesn't have the same color as the target object
            rgb = self.random_color(exclude=(self.target_color if not self._random_target_color else None), exclude_range=0.5)
            if record.name in PRIMITIVE_NAMES:
                commands.extend(
                    self.get_object_material_commands(
                        record, o_id, self.get_material_name(self.target_material)))
                commands.append(
                    {"$type": "set_color",
                     "color": {"r": rgb[0], "g": rgb[1], "b": rgb[2], "a": 1.},
                     "id": o_id})


            commands.extend([
                {"$type": "scale_object",
                 "scale_factor": scale,
                 "id": o_id}
            ])

            if self.no_moving_distractors:
                commands.extend([
                    {"$type": "set_object_collision_detection_mode",
                     "mode": "discrete",
                     "id": o_id},
                    {"$type": "set_kinematic_state",
                     "id": o_id,
                     "is_kinematic": True,
                     "use_gravity": True}])


            if self.PRINT:
                print("occluder name", record.name)
                print("occluder category", record.wcategory)
                print("occluder position", pos)
                print("occluder pose", rot)
                print("occluder scale", scale)

            # add the metadata
            self.colors = np.concatenate([self.colors, np.array(rgb).reshape((1,3))], axis=0)
            self.scales.append(scale)

        return commands


class MultiDominoes(Dominoes):

    def __init__(self,
                 port: int = None,
                 middle_objects=None,
                 num_middle_objects=1,
                 num_distinct_objects=3,
                 middle_color=None,
                 middle_scale_range=None,
                 middle_rotation_range=None,
                 middle_mass_range=[2.,7.],
                 horizontal=False,
                 spacing_jitter=0.2,
                 lateral_jitter=0.2,
                 middle_material=None,
                 remove_middle=False,
                 **kwargs):

        super().__init__(port=port, **kwargs)

        # Default to same type as target
        self.set_middle_types(middle_objects)

        # Appearance of middle objects
        self.middle_scale_range = middle_scale_range or self.target_scale_range
        self.middle_mass_range = middle_mass_range
        self.middle_rotation_range = middle_rotation_range
        self.middle_color = middle_color
        self.randomize_colors_across_trials = False if (middle_color is not None) else True
        self.middle_material = self.get_material_name(middle_material)
        self.horizontal = horizontal
        self.remove_middle = remove_middle
        self.num_distinct_objects = num_distinct_objects

        # How many middle objects and their spacing
        self.num_middle_objects = num_middle_objects
        self.spacing = self.collision_axis_length / (self.num_middle_objects + 1.)
        self.spacing_jitter = spacing_jitter
        self.lateral_jitter = lateral_jitter

    def set_middle_types(self, olist):
        if isinstance(olist, str):
            olist = [olist]

        if olist is None:
            self._middle_types = self._target_types
        else:
            tlist = self.get_types(olist, flex_only=self.flex_only)
            self._middle_types = tlist

    def clear_static_data(self) -> None:
        super().clear_static_data()

        self.distinct_ids = np.empty(dtype=np.int32, shape=0)
        self.middle_type = None
        self.distractors = OrderedDict()
        self.occluders = OrderedDict()

        if self.randomize_colors_across_trials:
            self.middle_color = None

    def _write_static_data(self, static_group: h5py.Group) -> None:
        super()._write_static_data(static_group)
        static_group.create_dataset("distinct_ids", data=self.distinct_ids)

        # about the distinct_objects
        distinct_group = static_group.create_group("distinct_objects")
        colors = np.stack([self.candidate_dict[id_]["color"] for id_ in range(self.num_distinct_objects)], axis=0)

        distinct_scales = [self.candidate_dict[id_]["scale"] for id_ in range(self.num_distinct_objects)]
        scales = np.stack([xyz_to_arr(_s) for _s in distinct_scales])
        distinct_group.create_dataset("scales", data=scales)
        distinct_group.create_dataset("colors", data=colors)
        distinct_group.create_dataset("masses", data=[self.candidate_dict[id_]["mass"] for id_ in range(self.num_distinct_objects)])


        static_group.create_dataset("remove_middle", data=self.remove_middle)
        if self.middle_type is not None:
            static_group.create_dataset("middle_objects", data=[self.middle_type.encode('utf8') for _ in range(self.trial_num_middle_objects)])
            static_group.create_dataset("middle_type", data=self.middle_type)

    @staticmethod
    def get_controller_label_funcs(classname = 'MultiDominoes'):
        funcs = super(MultiDominoes, MultiDominoes).get_controller_label_funcs(classname)

        def num_middle_objects(f):
            try:
                return int(len(f['static']['middle_objects']))
            except KeyError:
                return int(len(f['static']['mass']) - 3)
        def remove_middle(f):
            try:
                return bool(np.array(f['static']['remove_middle']))
            except KeyError:
                return bool(False)

        funcs += [num_middle_objects, remove_middle]

        return funcs

    def _build_intermediate_structure(self, interact_id=0) -> List[dict]:
        # set the middle object color
        if self.monochrome:
            self.middle_color = self.random_color(exclude=self.target_color)

        return self._place_middle_objects(interact_id=interact_id) if bool(self.trial_num_middle_objects) else []

    def _place_middle_objects(self, interact_id=0) -> List[dict]:

        offset = -0.5 * self.trial_collision_axis_length
        min_offset = offset + self.target_scale["x"]
        max_offset = 0.5 * self.trial_collision_axis_length - self.target_scale["x"]

        commands = []

        if self.remove_middle:
            rm_idx = random.choice(range(self.num_middle_objects))
        else:
            rm_idx = -1



        for m in range(self.trial_num_middle_objects):
            offset += self.spacing * random.uniform(1.-self.spacing_jitter, 1.+self.spacing_jitter)
            offset = np.minimum(np.maximum(offset, min_offset), max_offset)
            if offset >= max_offset:
                print("couldn't place middle object %s" % str(m+1))
                print("offset now", offset)
                break

            if m == rm_idx:
                continue

            middle_type, middle_scale, middle_mass, middle_color = self.type_scale_mass_color[1+m]
            o_id, scale, rgb, record, name = self.oid_scale_rgb_record_name[1+m]

            # create a target object
            #if not self.repeat_trial: # sample from scratch


            zpos = scale["z"] * random.uniform(-self.lateral_jitter, self.lateral_jitter)
            pos = arr_to_xyz([offset, 0., zpos])
            rot = self.get_y_rotation(self.middle_rotation_range)
            if self.horizontal:
                rot["z"] = 90
                pos["z"] += -np.sin(np.radians(rot["y"])) * scale["y"] * 0.5
                pos["x"] += np.cos(np.radians(rot["y"])) * scale["y"] * 0.5
            self.middle_type = name
            self.middle_scale = {k:max([scale[k], self.middle_scale[k]]) for k in scale.keys()}

            # Set the middle object material

        return commands



if __name__ == "__main__":
    import platform, os

    args = get_args("dominoes")

    # if platform.system() == 'Linux':
    #     if args.gpu is not None:
    #         os.environ["DISPLAY"] = ":0." + str(args.gpu)
    #     else:
    #         os.environ["DISPLAY"] = ":0"

    DomC = MultiDominoes(
        port=args.port,
        room=args.room,
        model_libraries=args.model_libraries,
        num_middle_objects=args.num_middle_objects,
        num_distinct_objects=args.num_distinct_objects,
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
        target_color=args.tcolor,
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
        camera_left_right_reflections=args.camera_left_right_reflections,
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
        occluder_aspect_ratio=args.occluder_aspect_ratio,
        distractor_aspect_ratio=args.distractor_aspect_ratio,
        remove_middle=args.remove_middle,
        use_ramp=bool(args.ramp),
        ramp_color=args.rcolor,
        flex_only=args.only_use_flex_objects,
        no_moving_distractors=args.no_moving_distractors,
        match_probe_and_target_color=args.match_probe_and_target_color,
        use_test_mode_colors=args.use_test_mode_colors
    )

    if bool(args.run):
        DomC.run(num=args.num,
                 output_dir=args.dir,
                 temp_path=args.temp,
                 width=args.width,
                 height=args.height,
                 framerate=args.framerate,
                 write_passes=args.write_passes.split(','),
                 save_passes=args.save_passes.split(','),
                 save_movies=args.save_movies,
                 save_labels=args.save_labels,
                 save_meshes=args.save_meshes,
                 args_dict=vars(args))
    else:
        end = DomC.communicate({"$type": "terminate"})
        print([OutputData.get_data_type_id(r) for r in end])
