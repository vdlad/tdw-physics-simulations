import sys, os, copy, subprocess, glob
import platform
from typing import List, Dict, Tuple
from abc import ABC, abstractmethod
from pathlib import Path
from tqdm import tqdm
import stopit
from PIL import Image
import io
import h5py, json
from collections import OrderedDict
import numpy as np
import random
from tdw.controller import Controller
from tdw.tdw_utils import TDWUtils
from tdw.output_data import OutputData, SegmentationColors, Meshes
from tdw.librarian import ModelRecord, MaterialLibrarian

from tdw_physics.postprocessing.stimuli import pngs_to_mp4
from tdw_physics.postprocessing.labels import (get_labels_from,
                                               get_all_label_funcs,
                                               get_across_trial_stats_from)
from tdw_physics.util_geom import save_obj
import shutil

PASSES = ["_img", "_depth", "_normals", "_flow", "_id"]
M = MaterialLibrarian()
MATERIAL_TYPES = M.get_material_types()
MATERIAL_NAMES = {mtype: [m.name for m in M.get_all_materials_of_type(mtype)] \
                  for mtype in MATERIAL_TYPES}

# colors for the target/zone overlay
ZONE_COLOR = [255,255,0]
TARGET_COLOR = [255,0,0]

class Dataset(Controller, ABC):
    """
    Abstract class for a physics dataset.

    1. Create a dataset .hdf5 file.
    2. Send commands to initialize the scene.
    3. Run a series of trials. Per trial, do the following:
        1. Get commands to initialize the trial. Write "static" data (which doesn't change between trials).
        2. Run the trial until it is "done" (defined by output from the writer). Write per-frame data to disk,.
        3. Clean up the scene and start a new trial.
    """
    def __init__(self,
                 port: int = 1071,
                 check_version: bool=False,
                 launch_build: bool=True,
                 randomize: int=0,
                 seed: int=0,
                 save_args=True,
                 **kwargs
    ):
        # save the command-line args
        self.save_args = save_args
        self._trial_num = None
        self.command_log = None

        super().__init__(port=port,
                         check_version=check_version,
                         launch_build=launch_build)

        # set random state
        self.randomize = randomize
        self.seed = seed
        if not bool(self.randomize):
            random.seed(self.seed)
            print("SET RANDOM SEED: %d" % self.seed)

        # fluid actors need to be handled separately
        self.fluid_object_ids = []

    def communicate(self, commands) -> list:
        '''
        Save a log of the commands so that they can be rerun
        '''
        if self.command_log is not None:
            with open(str(self.command_log), "at") as f:
                f.write(json.dumps(commands) + (" trial %s" % self._trial_num) + "\n")
        return super().communicate(commands)

    def clear_static_data(self) -> None:
        self.object_ids = np.empty(dtype=int, shape=0)
        self.model_names = []
        self._initialize_object_counter()

    @staticmethod
    def get_controller_label_funcs(classname = 'Dataset'):
        """
        A list of funcs with signature func(f: h5py.File) -> JSON-serializeable data
        """
        def stimulus_name(f):
            try:
                stim_name = str(np.array(f['static']['stimulus_name'], dtype=str))
            except TypeError:
                # happens if we have an empty stimulus name
                stim_name = "None"
            return stim_name
        def controller_name(f):
            return classname
        def git_commit(f):
            try:
                return str(np.array(f['static']['git_commit'], dtype=str))
            except TypeError:
                # happens when no git commit
                return "None"

        return [stimulus_name, controller_name, git_commit]

    def save_command_line_args(self, output_dir: str) -> None:
        if not self.save_args:
            return

        # save all the args, including defaults
        self._save_all_args(output_dir)

        # save just the commandline args
        output_dir = Path(output_dir)
        filepath = output_dir.joinpath("commandline_args.txt")
        if not filepath.exists():
            with open(filepath, 'w') as f:
                f.write('\n'.join(sys.argv[1:]))

        return

    def _save_all_args(self, output_dir: str) -> None:
        writelist = []
        for k,v in self.args_dict.items():
            writelist.extend(["--"+str(k),str(v)])

        self._script_args = writelist

        output_dir = Path(output_dir)
        filepath = output_dir.joinpath("args.txt")
        if not filepath.exists():
            with open(filepath, 'w') as f:
                f.write('\n'.join(writelist))
        return

    def get_initialization_commands(self,
                                    width: int,
                                    height: int) -> None:
        # Global commands for all physics datasets.
        commands = [{"$type": "set_screen_size",
                     "width": width,
                     "height": height},
                    {"$type": "set_render_quality",
                     "render_quality": 5},
                    {"$type": "set_physics_solver_iterations",
                     "iterations": 32},
                    {"$type": "set_vignette",
                     "enabled": False},
                    {"$type": "set_shadow_strength",
                     "strength": 1.0},
                    {"$type": "set_sleep_threshold",
                     "sleep_threshold": 0.01}]

        commands.extend(self.get_scene_initialization_commands())
        # Add the avatar.
        commands.extend([{"$type": "create_avatar",
                          "type": "A_Img_Caps_Kinematic",
                          "id": "a"},
                         {"$type": "set_target_framerate",
                          "framerate": self._framerate},
                         {"$type": "set_pass_masks",
                          "pass_masks": self.write_passes},
                         {"$type": "set_field_of_view",
                          "field_of_view": self.get_field_of_view()},
                         {"$type": "send_images",
                          "frequency": "always"}])
        return commands

    def run(self,
            num: int,
            output_dir: str,
            temp_path: str,
            width: int,
            height: int,
            framerate: int = 30,
            write_passes: List[str] = PASSES,
            save_passes: List[str] = [],
            save_movies: bool = False,
            save_labels: bool = False,
            save_meshes: bool = False,
            args_dict: dict={}) -> None:
        """
        Create the dataset.

        :param num: The number of trials in the dataset.
        :param output_dir: The root output directory.
        :param temp_path: Temporary path to a file being written.
        :param width: Screen width in pixels.
        :param height: Screen height in pixels.
        :param save_passes: a list of which passes to save out as PNGs (or convert to MP4)
        :param save_movies: whether to save out a movie of each trial
        :param save_labels: whether to save out JSON labels for the full trial set.
        """

        # If no temp_path given, place in local folder to prevent conflicts with other builds
        if temp_path == "NONE": temp_path = output_dir + "/temp.hdf5"

        self._height, self._width, self._framerate = height, width, framerate
        print("height: %d, width: %d, fps: %d" % (self._height, self._width, self._framerate))

        # the dir where files and metadata will go
        if not Path(output_dir).exists():
            Path(output_dir).mkdir(parents=True)

        # save a log of the commands send to TDW build
        self.command_log = Path(output_dir).joinpath('tdw_commands.json')

        # which passes to write to the HDF5
        self.write_passes = write_passes
        if isinstance(self.write_passes, str):
            self.write_passes = self.write_passes.split(',')
        self.write_passes = [p for p in self.write_passes if (p in PASSES)]

        # which passes to save as an MP4
        self.save_passes = save_passes
        if isinstance(self.save_passes, str):
            self.save_passes = self.save_passes.split(',')
        self.save_passes = [p for p in self.save_passes if (p in self.write_passes)]
        self.save_movies = save_movies

        # whether to send and save meshes
        self.save_meshes = save_meshes

        print("write passes", self.write_passes)
        print("save passes", self.save_passes)
        print("save movies", self.save_movies)
        print("save meshes", self.save_meshes)

        if self.save_movies:
            assert len(self.save_passes),\
                "You need to pass \'--save_passes [PASSES]\' to save out movies, where [PASSES] is a comma-separated list of items from %s" % PASSES

        # whether to save a JSON of trial-level labels
        self.save_labels = save_labels
        if self.save_labels:
            self.meta_file = Path(output_dir).joinpath('metadata.json')
            if self.meta_file.exists():
                self.trial_metadata = json.loads(self.meta_file.read_text())
            else:
                self.trial_metadata = []

        initialization_commands = self.get_initialization_commands(width=width, height=height)
        # Initialize the scene.
        self.communicate(initialization_commands)

        # Run trials
        self.trial_loop(num, output_dir, temp_path)

        # Terminate TDW
        # Windows doesn't know signal timeout
        if platform.system() == 'Windows': end = self.communicate({"$type": "terminate"})
        else: #Unix systems can use signal to timeout
            with stopit.SignalTimeout(5) as to_ctx_mgr: #since TDW sometimes doesn't acknowledge being stopped we only *try* to close it
                assert to_ctx_mgr.state == to_ctx_mgr.EXECUTING
                end = self.communicate({"$type": "terminate"})
            if to_ctx_mgr.state == to_ctx_mgr.EXECUTED:
                print("tdw closed successfully")
            elif to_ctx_mgr.state == to_ctx_mgr.TIMED_OUT:
                print("tdw failed to acknowledge being closed. tdw window might need to be manually closed")

        # Save the command line args
        if self.save_args:
            self.args_dict = copy.deepcopy(args_dict)
        self.save_command_line_args(output_dir)

        # Save the across-trial stats
        if self.save_labels:
            hdf5_paths = glob.glob(str(output_dir) + '/*.hdf5')
            stats = get_across_trial_stats_from(
                hdf5_paths, funcs=self.get_controller_label_funcs(classname=type(self).__name__))
            stats["num_trials"] = int(len(hdf5_paths))
            stats_str = json.dumps(stats, indent=4)
            stats_file = Path(output_dir).joinpath('trial_stats.json')
            stats_file.write_text(stats_str, encoding='utf-8')
            print("ACROSS TRIAL STATS")
            print(stats_str)



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

        pbar.update(exists_up_to)
        for i in range(exists_up_to, num):
            filepath = output_dir.joinpath(TDWUtils.zero_padding(i, 4) + ".hdf5")
            self.stimulus_name = '_'.join([filepath.parent.name, str(Path(filepath.name).with_suffix(''))])

            if not filepath.exists():

                # Save out images
                self.png_dir = None
                if any([pa in PASSES for pa in self.save_passes]):
                    self.png_dir = output_dir.joinpath("pngs_" + TDWUtils.zero_padding(i, 4))
                    if not self.png_dir.exists():
                        self.png_dir.mkdir(parents=True)

                # Do the trial.
                self.trial(filepath=filepath,
                           temp_path=temp_path,
                           trial_num=i)

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
              trial_num: int) -> None:
        """
        Run a trial. Write static and per-frame data to disk until the trial is done.

        :param filepath: The path to this trial's hdf5 file.
        :param temp_path: The path to the temporary file.
        :param trial_num: The number of the current trial.
        """

        # Clear the object IDs and other static data
        self.clear_static_data()
        self._trial_num = trial_num

        # Create the .hdf5 file.
        f = h5py.File(str(temp_path.resolve()), "a")

        commands = []
        # Remove asset bundles (to prevent a memory leak).
        if trial_num % 100 == 0:
            commands.append({"$type": "unload_asset_bundles"})

        # Add commands to start the trial.
        commands.extend(self.get_trial_initialization_commands())
        # Add commands to request output data.
        commands.extend(self._get_send_data_commands())

        # Send the commands and start the trial.
        r_types = ['']
        count = 0
        resp = self.communicate(commands)

        self._set_segmentation_colors(resp)

        self._get_object_meshes(resp)
        frame = 0
        # Write static data to disk.
        static_group = f.create_group("static")
        self._write_static_data(static_group)

        # Add the first frame.
        done = False
        frames_grp = f.create_group("frames")
        frame_grp, _, _, _ = self._write_frame(frames_grp=frames_grp, resp=resp, frame_num=frame)
        self._write_frame_labels(frame_grp, resp, -1, False)

        # Continue the trial. Send commands, and parse output data.
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

    @staticmethod
    def rotate_vector_parallel_to_floor(
            vector: Dict[str, float],
            theta: float,
            degrees: bool = True) -> Dict[str, float]:

        v_x = vector['x']
        v_z = vector['z']
        if degrees:
            theta = np.radians(theta)

        v_x_new = np.cos(theta) * v_x - np.sin(theta) * v_z
        v_z_new = np.sin(theta) * v_x + np.cos(theta) * v_z

        return {'x': v_x_new, 'y': vector['y'], 'z': v_z_new}

    @staticmethod
    def scale_vector(
            vector: Dict[str, float],
            scale: float) -> Dict[str, float]:
        return {k:vector[k] * scale for k in ['x','y','z']}

    @staticmethod
    def get_random_avatar_position(radius_min: float,
                                   radius_max: float,
                                   y_min: float,
                                   y_max: float,
                                   center: Dict[str, float],
                                   angle_min: float = 0,
                                   angle_max: float = 360,
                                   reflections: bool = False,
                                   ) -> Dict[str, float]:
        """
        :param radius_min: The minimum distance from the center.
        :param radius_max: The maximum distance from the center.
        :param y_min: The minimum y positional coordinate.
        :param y_max: The maximum y positional coordinate.
        :param center: The centerpoint.
        :param angle_min: The minimum angle of rotation around the centerpoint.
        :param angle_max: The maximum angle of rotation around the centerpoint.

        :return: A random position for the avatar around a centerpoint.
        """

        a_r = random.uniform(radius_min, radius_max)
        a_x = center["x"] + a_r
        a_z = center["z"] + a_r
        theta = np.radians(random.uniform(angle_min, angle_max))
        if reflections:
            theta2 = random.uniform(angle_min+180, angle_max+180)
            theta = random.choice([theta, theta2])
        a_y = random.uniform(y_min, y_max)
        a_x_new = np.cos(theta) * (a_x - center["x"]) - np.sin(theta) * (a_z - center["z"]) + center["x"]
        a_z_new = np.sin(theta) * (a_x - center["x"]) + np.cos(theta) * (a_z - center["z"]) + center["z"]
        a_x = a_x_new
        a_z = a_z_new

        return {"x": a_x, "y": a_y, "z": a_z}

    def is_done(self, resp: List[bytes], frame: int) -> bool:
        """
        Override this command for special logic to end the trial.

        :param resp: The output data response.
        :param frame: The frame number.

        :return: True if the trial is done.
        """

        return False

    @abstractmethod
    def get_scene_initialization_commands(self) -> List[dict]:
        """
        :return: Commands to initialize the scene ONLY for the first time (not per-trial).
        """

        raise Exception()

    @abstractmethod
    def get_trial_initialization_commands(self) -> List[dict]:
        """
        :return: Commands to initialize each trial.
        """

        raise Exception()

    @abstractmethod
    def _get_send_data_commands(self) -> List[dict]:
        """
        :return: A list of commands to request per-frame output data. Appended to the trial initialization commands.
        """

        raise Exception()

    def _write_static_data(self, static_group: h5py.Group) -> None:
        """
        Write static data to disk after assembling the trial initialization commands.

        :param static_group: The static data group.
        """
        # git commit and args
        #res = subprocess.run('git rev-parse HEAD', shell=True, capture_output=True, text=True)
        self.commit = "" #res.stdout.strip()
        static_group.create_dataset("git_commit", data=self.commit)

        # stimulus name
        static_group.create_dataset("stimulus_name", data=self.stimulus_name)
        static_group.create_dataset("object_ids", data=self.object_ids)
        static_group.create_dataset("model_names", data=[s.encode('utf8') for s in self.model_names])

        if self.object_segmentation_colors is not None:
            static_group.create_dataset("object_segmentation_colors", data=self.object_segmentation_colors)

    @abstractmethod
    def _write_frame(self, frames_grp: h5py.Group, resp: List[bytes], frame_num: int) -> \
            Tuple[h5py.Group, h5py.Group, dict, bool]:
        """
        Write a frame to the hdf5 file.

        :param frames_grp: The frames hdf5 group.
        :param resp: The response from the build.
        :param frame_num: The frame number.

        :return: Tuple: (The frame group, the objects group, a dictionary of Transforms, True if the trial is "done")
        """

        raise Exception()

    def _write_frame_labels(self,
                            frame_grp: h5py.Group,
                            resp: List[bytes],
                            frame_num: int,
                            sleeping: bool) -> Tuple[h5py.Group, bool]:
        """
        Writes the trial-level data for this frame.

        :param frame_grp: The hdf5 group for a single frame.
        :param resp: The response from the build.
        :param frame_num: The frame number.
        :param sleeping: Whether this trial timed out due to objects falling asleep.

        :return: Tuple(h5py.Group labels, bool done): the labels data and whether this is the last frame of the trial.
        """
        labels = frame_grp.create_group("labels")
        if frame_num > 0:
            complete = self.is_done(resp, frame_num)
        else:
            complete = False

        # If the trial is over, one way or another
        done = sleeping or complete

        # Write labels indicate whether and why the trial is over
        labels.create_dataset("trial_end", data=done)
        labels.create_dataset("trial_timeout", data=(sleeping and not complete))
        labels.create_dataset("trial_complete", data=(complete and not sleeping))

        # if done:
        #     print("Trial Ended: timeout? %s, completed? %s" % \
        #           ("YES" if sleeping and not complete else "NO",\
        #            "YES" if complete and not sleeping else "NO"))

        return labels, resp, frame_num, done

    def _get_destroy_object_command_name(self, o_id: int) -> str:
        """
        :param o_id: The object ID.

        :return: The name of the command used to destroy an object.
        """

        return "destroy_object"

    @abstractmethod
    def get_per_frame_commands(self, resp: List[bytes], frame: int) -> List[dict]:
        """
        :param resp: The output data response.
        :param frame: The frame number

        :return: Commands to send per frame.
        """
        raise Exception()

    @abstractmethod
    def get_field_of_view(self) -> float:
        """
        :return: The camera field of view.
        """

        raise Exception()

    def add_object(self, model_name: str, position={"x": 0, "y": 0, "z": 0}, rotation={"x": 0, "y": 0, "z": 0},
                   library: str = "") -> int:
        raise Exception("Don't use this function; see README for functions that supersede it.")

    def get_add_object(self, model_name: str, object_id: int, position={"x": 0, "y": 0, "z": 0},
                       rotation={"x": 0, "y": 0, "z": 0}, library: str = "") -> dict:
        raise Exception("Don't use this function; see README for functions that supersede it.")

    def _initialize_object_counter(self) -> None:
        self._object_id_counter = int(0)
        self._object_id_increment = int(1)

    def _increment_object_id(self) -> None:
        self._object_id_counter = int(self._object_id_counter + self._object_id_increment)

    def _get_next_object_id(self) -> int:
        self._increment_object_id()
        return int(self._object_id_counter)

    def get_material_name(self, material, print_info=False):

        if material is not None:
            if material in MATERIAL_TYPES:
                mat = random.choice(MATERIAL_NAMES[material])
            else:
                assert any((material in MATERIAL_NAMES[mtype] for mtype in self.material_types)), \
                    (material, self.material_types)
                mat = material
        else:
            mtype = random.choice(self.material_types)
            mat = random.choice(MATERIAL_NAMES[mtype])
            if print_info:
                print(mtype, mat)

        return mat

    def get_object_material_commands(self, record, object_id, material):
        commands = TDWUtils.set_visual_material(
            self, record.substructure, object_id, material, quality="high")
        return commands


    def _set_segmentation_colors(self, resp: List[bytes]) -> None:

        self.object_segmentation_colors = None
        for r in resp:
            if OutputData.get_data_type_id(r) == 'segm':
                seg = SegmentationColors(r)
                colors = {}
                for i in range(seg.get_num()):
                    try:
                        colors[seg.get_object_id(i)] = seg.get_object_color(i)
                    except:
                        print("No object id found for seg", i)

                self.object_segmentation_colors = []
                for o_id in self.object_ids:
                    if o_id in colors.keys():
                        self.object_segmentation_colors.append(
                            np.array(colors[o_id], dtype=np.uint8).reshape(1,3))
                    else:
                        self.object_segmentation_colors.append(
                            np.array([0,0,0], dtype=np.uint8).reshape(1,3))

                self.object_segmentation_colors = np.concatenate(self.object_segmentation_colors, 0)
    def _get_object_meshes(self, resp: List[bytes]) -> None:

        self.object_meshes = dict()
        # {object_id: (vertices, faces)}
        for r in resp:
            if OutputData.get_data_type_id(r) == 'mesh':
                meshes = Meshes(r)
                nmeshes = meshes.get_num()

                assert(len(self.object_ids) == nmeshes)
                for index in range(nmeshes):
                    o_id = meshes.get_object_id(index)
                    vertices = meshes.get_vertices(index)
                    faces = meshes.get_triangles(index)
                    self.object_meshes[o_id] = (vertices, faces)
