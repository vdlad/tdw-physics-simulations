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
from tdw_physics.util import MODEL_LIBRARIES, get_parser, xyz_to_arr, arr_to_xyz, str_to_xyz

from tdw_physics.target_controllers.dominoes import Dominoes, MultiDominoes, get_args
from tdw_physics.postprocessing.labels import is_trial_valid

MODEL_NAMES = [r.name for r in MODEL_LIBRARIES['models_flex.json'].records]
M = MaterialLibrarian()
MATERIAL_TYPES = M.get_material_types()
MATERIAL_NAMES = {mtype: [m.name for m in M.get_all_materials_of_type(mtype)] \
                  for mtype in MATERIAL_TYPES}

def none_or_str(value):
    if value == 'None':
        return None
    else:
        return value


def get_barrier_args(dataset_dir: str, parse=True):

    common = get_parser(dataset_dir, get_help=False)
    domino, domino_postproc = get_args(dataset_dir, parse=False)
    parser = ArgumentParser(parents=[common, domino], conflict_handler='resolve', fromfile_prefix_chars='@')

    parser.add_argument("--middle",
                        type=str,
                        default='platonic',
                        help="middle object type")
    parser.add_argument("--bridge_height",
                        type=float,
                        default=1.0,
                        help="How high to make the bridge")

    def postprocess(args):
        return args

    args = parser.parse_args()
    args = domino_postproc(args)
    args = postprocess(args)

    return args

class Barrier(MultiDominoes):

    def __init__(self,
                 port: int = None,
                 bridge_height=1.0,
                 **kwargs):
        # initialize everything in common w / Multidominoes
        super().__init__(port=port, **kwargs)

        # do some Barrier-specific stuff
        self.bridge_height = bridge_height

    def _build_intermediate_structure(self) -> List[dict]:

        print("middle color", self.middle_color)
        if self.randomize_colors_across_trials:
            self.middle_color = self.random_color(exclude=self.target_color) if self.monochrome else None

        commands = []

        # Go nuts
        commands.extend(self._place_barrier_foundation())
        commands.extend(self._build_bridge())

        return commands

    def _place_barrier_foundation(self):
        return []

    def _build_bridge(self):
        return []

    def clear_static_data(self) --> None:
        Dominoes.clear_static_data(self)

        # clear some other stuff

    def _write_static_data(self, static_group: h5py.Group) -> None:
        Dominoes._write_static_data(self, static_group)

        static_group.create_dataset("bridge_height", data=self.bridge_height)


    @staticmethod
    def get_controller_label_funcs(classname = "Barrier"):

        funcs = Dominoes.get_controller_label_funcs(classname)

        return funcs
    

if __name__ == "__main__":

    args = get_barrier_args("barriers")

    BC = Barrier(
        middle_objects=args.middle,
        bridge_height=args.bridge_height
    )

    if bool(args.run):
        BC.run(num=args.num,
               output_dir=args.dir,
               temp_path=args.temp,
               width=args.width,
               height=args.height,
               save_passes=args.save_passes.split(','),
               save_movies=args.save_movies,
               save_labels=args.save_labels,               
               args_dict=vars(args)
        )
    else:
        BC.communicate({"$type": "terminate"})
