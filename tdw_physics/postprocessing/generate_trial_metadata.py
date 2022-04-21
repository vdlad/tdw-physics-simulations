import os, sys, glob, subprocess
from pkgutil import iter_modules
import importlib
from pathlib import Path
from collections import OrderedDict
from typing import List,Dict,Tuple
import h5py, json
import numpy as np


from tdw_physics.postprocessing.labels import get_labels_from
import tdw_physics.target_controllers as controllers

def list_controllers():
    cs = []
    for c in iter_modules(controllers.__path__):
        cs.append(c.name)
    return cs

def get_controller_label_funcs_by_class(cls: str = 'MultiDominoes'):

    cs = list_controllers()
    Class = None
    for c in cs:
        module = importlib.import_module("tdw_physics.target_controllers." + c)
        if cls in module.__dict__.keys():
            Class = getattr(module, cls)
            funcs = Class.get_controller_label_funcs(cls)
            return funcs

def compute_metadata_from_stimuli(
        stimulus_dir : str,
        file_pattern : str = "*.hdf5",
        controller_class: str = None,
        label_funcs: List[type(list_controllers)] = [],
        overwrite: bool = False,
        outfile: str = 'metadata') -> None:

    # get the hdf5s in the directory
    stims = sorted(glob.glob(stimulus_dir + file_pattern))

    # try to infer the controller class
    if controller_class is None:
        meta_file = Path(stimulus_dir).joinpath('metadata.json')
        if meta_file.exists():
            trial_meta = json.loads(meta_file.read_text())[0]
            controller_class = str(trial_meta['controller_name'])
        else:
            raise ValueError("Controller classname could not be read from existing metadata.json")

    # add the label funcs
    label_funcs += get_controller_label_funcs_by_class(controller_class)

    # iterate over stims
    metadata = []
    for stimpath in stims:
        f = h5py.File(stimpath, 'r')
        trial_meta = OrderedDict()
        trial_meta = get_labels_from(f, label_funcs, res=trial_meta)
        metadata.append(trial_meta)
        f.close()

    # write out new metadata
    json_str = json.dumps(metadata, indent=4)    
    meta_file = Path(stimulus_dir).joinpath(outfile + ('' if overwrite else '_post') + '.json')
    meta_file.write_text(json_str, encoding='utf-8')
    print("Wrote new metadata: %s\nfor %d trials" % (str(meta_file), len(metadata)))
    return
        
if __name__ == '__main__':
    # print(get_controller_label_funcs_by_class())

    stim_dir = sys.argv[1]
    compute_metadata_from_stimuli(stim_dir)
