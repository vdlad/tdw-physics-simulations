from tdw.controller import Controller
from tdw.tdw_utils import TDWUtils
from tdw.output_data import OutputData, Images, Meshes
from pathlib import Path
from typing import List, Dict, Tuple

from tdw.librarian import ModelLibrarian
from tdw.tdw_utils import TDWUtils

# Every model library, sorted by name.
MODEL_LIBRARIES: Dict[str, ModelLibrarian] = {}
for filename in ModelLibrarian.get_library_filenames():
    MODEL_LIBRARIES.update({filename: ModelLibrarian(filename)})

import numpy as np
import ipdb
st= ipdb.set_trace
MODEL_NAMES = [r.name for r in MODEL_LIBRARIES['models_flex.json'].records]
MODEL_NAMES_FULL = [r.name for r in MODEL_LIBRARIES['models_full.json'].records]
MODEL_NAMES_SPECIAL = [r.name for r in MODEL_LIBRARIES['models_special.json'].records]



# lib = ModelLibrarian("models_full.json")
# for record in lib.records:
#     if record.flex:
#         print(record.name)
#st()



#['bowl', 'cone', 'cube', 'cylinder', 'dumbbell', 'octahedron', 'pentagon', 'pipe', 'platonic', 'pyramid', 'sphere', 'torus', 'triangular_prism']
def save_obj(vertices: np.ndarray, faces: np.ndarray, filepath: str):
    with open(filepath, 'w') as f:
        f.write("# OBJ file\n")
        for v in vertices:
            f.write("v %.4f %.4f %.4f\n" % (v[0],v[1],v[2]))
        for face in faces:
            f.write("f")
            for vertex in face:
                f.write(" %d" % (vertex + 1))
            f.write("\n")

def get_types(objlist, libraries=["models_flex.json"], categories=None):
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
    return tlist


def _get_object_meshes(resp: List[bytes]):

    object_meshes = dict()
    # {object_id: (vertices, faces)}
    for r in resp:
        if OutputData.get_data_type_id(r) == 'mesh':
            meshes = Meshes(r)
            nmeshes = meshes.get_num()

            #assert(len(self.object_ids) == nmeshes)
            for index in range(nmeshes):
                o_id = meshes.get_object_id(index)
                vertices = meshes.get_vertices(index)
                faces = meshes.get_triangles(index)
                object_meshes[o_id] = (vertices, faces)
    return object_meshes


#model_flex.json: ['bowl', 'cone', 'cube', 'cylinder', 'dumbbell', 'octahedron', 'pentagon', 'pipe', 'platonic', 'pyramid', 'sphere', 'torus', 'triangular_prism']
# "models_full.json": ['red_lounger_chair', 'green_side_chair','white_lounger_chair', 'potted_plant_wide' ]
# "models_special.json":

# "chairs"

chairs = ['emeco_navy_chair', 'lapalma_stil_chair', 'linbrazil_diz_armchair', 'naughtone_pinch_stool_chair', '648972_chair_poliform_harmony', 'adirondack_chair', 'chair_thonet_marshall', 'emeco_navy_chair', 'lapalma_stil_chair', 'linbrazil_diz_armchair', 'naughtone_pinch_stool_chair']

# o'emeco_navy_chair', 'lapalma_stil_chair', 'linbrazil_diz_armchair', 'naughtone_pinch_stool_chair',  'adirondack_chair', 'chair_thonet_marshall', 'emeco_navy_chair', 'lapalma_stil_chair', 'linbrazil_diz_armchair', 'naughtone_pinch_stool_chair']
# x '648972_chair_poliform_harmony',
c = Controller()
for chair in chairs:
    olist = [chair]
    print("object name:", olist[0])
    print("object is in model_flex.json :", olist[0] in MODEL_NAMES)
    print("object is in model_full.json :", olist[0] in MODEL_NAMES_FULL)
    print("object is in model_special.json :", olist[0] in MODEL_NAMES_SPECIAL)

    tlist = get_types(olist, libraries=["models_flex.json", "models_full.json", "models_special.json"])

    assert(len(tlist) > 0), f"cannot find the object named {olist[0]}"

    obj_id = 0
    record = tlist[0]

    # load the target object

    resp = c.communicate([{"$type": "load_scene",
                           "scene_name": "ProcGenScene"},
                          TDWUtils.create_empty_room(12, 12),
                          {"$type": "add_object",
                            "name": record.name,
                            "url": record.get_url(),
                            "scale_factor": record.scale_factor,
                            "position": {"x": 0, "y": 0, "z": 0},
                            "rotation": {"x": 0, "y": 0, "z": 0},
                            "category": record.wcategory,
                            "id": obj_id},
                          # {"$type": "send_bounds",
                          #  "ids": [obj_id],
                          #  "frequency": "once"}
                             ])





    # draw coordinate system xyz, x is red, y is green, z is blue
    resp = c.communicate([  {"$type": "add_object",
                            "name": record.name,
                            "url": record.get_url(),
                            "position": {"x": 1, "y": 0, "z": 0},
                            "rotation": {"x": 0, "y": 0, "z": 0},
                            "category": record.wcategory,
                            "id": obj_id + 1},
                            {"$type": "set_color",
                                "color": {"r": 1, "g": 0, "b": 0, "a": 1.},
                            "id": obj_id + 1},
                            {"$type": "scale_object",
                             "scale_factor": {"x": 0.1, "y": 0.1, "z": 0.1},
                            "id": obj_id + 1}
                        ])

    resp = c.communicate([  {"$type": "add_object",
                            "name": record.name,
                            "url": record.get_url(),
                            "position": {"x": 0, "y": 0, "z": 1},
                            "rotation": {"x": 0, "y": 0, "z": 0},
                            "category": record.wcategory,
                            "id": obj_id + 2},
                            {"$type": "set_color",
                                "color": {"r": 0, "g": 0, "b": 1, "a": 1.},
                            "id": obj_id + 2},
                            {"$type": "scale_object",
                             "scale_factor": {"x": 0.1, "y": 0.1, "z": 0.1},
                            "id": obj_id + 2}
                        ])

    resp = c.communicate([  {"$type": "add_object",
                            "name": record.name,
                            "url": record.get_url(),
                            "position": {"x": 0, "y": 1, "z": 0},
                            "rotation": {"x": 0, "y": 0, "z": 0},
                            "category": record.wcategory,
                            "id": obj_id + 3},
                            {"$type": "set_color",
                                "color": {"r": 0, "g": 1, "b": 0, "a": 1.},
                            "id": obj_id + 3},
                            {"$type": "scale_object",
                             "scale_factor": {"x": 0.1, "y": 0.1, "z": 0.1},
                            "id": obj_id + 3}
                        ])


    # load image
    avatar_id = "a"
    resp = c.communicate([{"$type": "create_avatar",
                           "type": "A_Img_Caps_Kinematic",
                           "avatar_id": avatar_id},
                          {"$type": "teleport_avatar_to",
                           "position": {"x": 1, "y": 5.5, "z": 5},
                           "avatar_id": avatar_id},
                          {"$type": "look_at",
                           "avatar_id": avatar_id,
                           "object_id": obj_id},
                          {"$type": "set_pass_masks",
                           "avatar_id": avatar_id,
                           "pass_masks": ["_img"]},
                          {"$type": "send_images",
                           "frequency": "once",
                           "avatar_id": avatar_id}])


    # Get the image.
    # image from tdw is lef-right flipped!
    import imageio
    for r in resp[:-1]:
        r_id = OutputData.get_data_type_id(r)
        # Find the image data.
        if r_id == "imag":
            img = Images(r)

            # Usually, you'll want to use one of these functions, but not both of them:

            # Use this to save a .jpg
            TDWUtils.save_images(img, filename="test_img")

            print(f"Image saved to: {Path('dist/test_img.jpg').resolve()}")
            img_array = imageio.imread("dist/img_test_img.png")
            imageio.imwrite("dist/img_test_img_fliplr.png", np.fliplr(img_array))
            # Use this to convert the image to a PIL image, which can be processed by a ML system at runtime.
            # The index is 0 because we know that there is only one pass ("_img").
            pil_img = TDWUtils.get_pil_image(img, index=0)

    # get the mesh

    resp = c.communicate({"$type": "send_meshes"})
    print([OutputData.get_data_type_id(r) for r in resp])
    meshes = _get_object_meshes(resp)

    c.communicate({"$type": "terminate"})
    vertices, faces = meshes[0]
    if vertices.shape[0] == 0:
        print("zero_vertex: ", olist)
        import ipdb; ipdb.set_trace()


    save_obj(vertices, faces, "dist/tmp.obj")

    import ipdb; ipdb.set_trace()
    print("end")


