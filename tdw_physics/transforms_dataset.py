from typing import List, Tuple, Dict, Optional
from abc import ABC
import h5py
import numpy as np
import random
from tdw.tdw_utils import TDWUtils
from tdw.output_data import OutputData, Transforms, Images, CameraMatrices, Bounds
from tdw.controller import Controller
from tdw.librarian import ModelRecord
from tdw_physics.dataset import Dataset
from tdw_physics.util import xyz_to_arr, arr_to_xyz, MODEL_LIBRARIES

from PIL import Image

class TransformsDataset(Dataset, ABC):
    """
    A dataset creator that receives and writes per frame: `Transforms`, `Images`, `CameraMatrices`.
    See README for more info.
    """

    def clear_static_data(self) -> None:
        super().clear_static_data()

        self.initial_positions = []
        self.initial_rotations = []

    def _write_static_data(self, static_group: h5py.Group) -> None:
        super()._write_static_data(static_group)

        # positions and rotations of objects
        static_group.create_dataset("initial_position",
                                    data=np.stack([xyz_to_arr(p) for p in self.initial_positions], 0))
        static_group.create_dataset("initial_rotation",
                                    data=np.stack([xyz_to_arr(r) for r in self.initial_rotations], 0))

    def random_model(self,
                     object_types: List[ModelRecord],
                     random_obj_id: bool = False,
                     add_data: bool = True) -> dict:
        obj_record = random.choice(object_types)
        obj_data = {
            "id": self.get_unique_id() if random_obj_id else self._get_next_object_id(),
            "name": obj_record.name
        }

        if add_data:
            self.model_names.append(obj_data["name"])

        return obj_record, obj_data

    def add_transforms_object(self,
                              record: ModelRecord,
                              position: Dict[str, float],
                              rotation: Dict[str, float],
                              o_id: Optional[int] = None,
                              add_data: Optional[bool] = True
    ) -> dict:
        """
        This is a wrapper for `Controller.get_add_object()` and the `add_object` command.
        This caches the ID of the object so that it can be easily cleaned up later.

        :param record: The model record.
        :param position: The initial position of the object.
        :param rotation: The initial rotation of the object, in Euler angles.
        :param o_id: The unique ID of the object. If None, a random ID is generated.
        :param add_data: whether to add the chosen data to the hdf5

        :return: An `add_object` command.
        """

        if o_id is None:
            o_id: int = Controller.get_unique_id()

        # Log the static data.
        self.object_ids = np.append(self.object_ids, o_id)

        if add_data:
            self.initial_positions = np.append(self.initial_positions, position)
            self.initial_rotations = np.append(self.initial_rotations, rotation)

        return {"$type": "add_object",
                "name": record.name,
                "url": record.get_url(),
                "scale_factor": record.scale_factor,
                "position": position,
                "rotation": rotation,
                "category": record.wcategory,
                "id": o_id}


    def _get_send_data_commands(self) -> List[dict]:
        commands = [{"$type": "send_transforms",
                     "frequency": "always"},
                    {"$type": "send_camera_matrices",
                     "frequency": "always"},
                    {"$type": "send_bounds",
                     "frequency": "always"},
                    {"$type": "send_segmentation_colors",
                     "ids": [int(oid) for oid in self.object_ids],
                     "frequency": "once"}]

        return commands

    def _write_frame(self, frames_grp: h5py.Group, resp: List[bytes], frame_num: int) -> \
            Tuple[h5py.Group, h5py.Group, dict, bool]:
        num_objects = len(self.object_ids)

        # Create a group for this frame.
        frame = frames_grp.create_group(TDWUtils.zero_padding(frame_num, 4))

        # Create a group for images.
        images = frame.create_group("images")

        # Transforms data.
        positions = np.empty(dtype=np.float32, shape=(num_objects, 3))
        forwards = np.empty(dtype=np.float32, shape=(num_objects, 3))
        rotations = np.empty(dtype=np.float32, shape=(num_objects, 4))

        # Bounds data.
        bounds = dict()
        for bound_type in ['front', 'back', 'left', 'right', 'top', 'bottom', 'center']:
            bounds[bound_type] = np.empty(dtype=np.float32, shape=(num_objects, 3))

        camera_matrices = frame.create_group("camera_matrices")

        # Parse the data in an ordered manner so that it can be mapped back to the object IDs.
        tr_dict = dict()

        # r_types = [OutputData.get_data_type_id(r) for r in resp[:-1]]
        # print(frame_num, r_types)

        for r in resp[:-1]:
            r_id = OutputData.get_data_type_id(r)
            if r_id == "tran":
                tr = Transforms(r)
                for i in range(tr.get_num()):
                    pos = tr.get_position(i)
                    tr_dict.update({tr.get_id(i): {"pos": pos,
                                                   "for": tr.get_forward(i),
                                                   "rot": tr.get_rotation(i)}})
                # Add the Transforms data.
                for o_id, i in zip(self.object_ids, range(num_objects)):
                    if o_id not in tr_dict:
                        continue
                    positions[i] = tr_dict[o_id]["pos"]
                    forwards[i] = tr_dict[o_id]["for"]
                    rotations[i] = tr_dict[o_id]["rot"]
            elif r_id == "imag":
                im = Images(r)
                # Add each image.
                for i in range(im.get_num_passes()):
                    pass_mask = im.get_pass_mask(i)
                    # Reshape the depth pass array.
                    if pass_mask == "_depth":
                        image_data = TDWUtils.get_shaped_depth_pass(images=im, index=i)
                    else:
                        image_data = im.get_image(i)
                    images.create_dataset(pass_mask, data=image_data, compression="gzip")

                    # Save PNGs
                    if pass_mask in self.save_passes:
                        filename = pass_mask[1:] + "_" + TDWUtils.zero_padding(frame_num, 4) + "." + im.get_extension(i)
                        path = self.png_dir.joinpath(filename)
                        if pass_mask in ["_depth", "_depth_simple"]:
                            Image.fromarray(TDWUtils.get_shaped_depth_pass(images=im, index=i)).save(path)
                        else:
                            with open(path, "wb") as f:
                                f.write(im.get_image(i))
            elif r_id == "boun":
                bo = Bounds(r)
                bo_dict = dict()
                for i in range(bo.get_num()):
                    bo_dict.update({bo.get_id(i): {"front": bo.get_front(i),
                                                   "back": bo.get_back(i),
                                                   "left": bo.get_left(i),
                                                   "right": bo.get_right(i),
                                                   "top": bo.get_top(i),
                                                   "bottom": bo.get_bottom(i),
                                                   "center": bo.get_center(i)}})
                for o_id, i in zip(self.object_ids, range(num_objects)):
                    for bound_type in bounds.keys():
                        try:
                            bounds[bound_type][i] = bo_dict[o_id][bound_type]
                        except KeyError:
                            print("couldn't store bound data for object %d" % o_id)


            # Add the camera matrices.
            elif OutputData.get_data_type_id(r) == "cama":
                matrices = CameraMatrices(r)
                camera_matrices.create_dataset("projection_matrix", data=matrices.get_projection_matrix())
                camera_matrices.create_dataset("camera_matrix", data=matrices.get_camera_matrix())


    

        objs = frame.create_group("objects")
        objs.create_dataset("positions", data=positions.reshape(num_objects, 3), compression="gzip")
        objs.create_dataset("forwards", data=forwards.reshape(num_objects, 3), compression="gzip")
        objs.create_dataset("rotations", data=rotations.reshape(num_objects, 4), compression="gzip")
        for bound_type in bounds.keys():
            objs.create_dataset(bound_type, data=bounds[bound_type], compression="gzip")

        return frame, objs, tr_dict, False

    def get_object_position(self, obj_id: int, resp: List[bytes]) -> None:
        position = None
        for r in resp:
            r_id = OutputData.get_data_type_id(r)
            if r_id == "tran":
                tr = Transforms(r)
                for i in range(tr.get_num()):
                    if tr.get_id(i) == obj_id:
                        position = tr.get_position(i)

        return position
