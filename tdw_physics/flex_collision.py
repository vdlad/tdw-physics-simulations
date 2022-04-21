from tdw.controller import Controller
from tdw.tdw_utils import TDWUtils
from tdw.output_data import OutputData, FlexParticles, EnvironmentCollision, Collision

class FlexCollisions(Controller):
    def run(self):
        self.start()
        object_id = self.get_unique_id()
        resp = self.communicate([TDWUtils.create_empty_room(12, 12),
                                 {"$type": "create_flex_container",
                                  "particle_size": 0.1,
                                  "collision_distance": 0.025,
                                  "solid_rest": 0.1},
                                 # {"$type": "create_flex_container",
                                 # "collision_distance": 0.001,
                                 # "static_friction": 1.0,
                                 # "dynamic_friction": 1.0,
                                 # "radius": 0.1875,
                                 #  "max_particles": 200000},
                                 self.get_add_object(model_name="linbrazil_diz_armchair",
                                                     position={"x": 0.0, "y": 2.0, "z": 0.0},
                                                     rotation={"x": 25.0, "y": 45.0, "z": -40.0},
                                                     object_id=object_id),
                                 {"$type": "set_kinematic_state",
                                  "id": object_id},
                                 {"$type": "set_flex_solid_actor",
                                  "id": object_id,
                                  "mesh_expansion": 0,
                                  "particle_spacing": 0.025,
                                  "mass_scale": 1},
                                 # {"$type": "set_flex_soft_actor",
                                 #  "id": object_id,
                                 #  "skinning_falloff": 0.5,
                                 #  "volume_sampling": 1.0,
                                 #  "mass_scale": 1.0,
                                 #  "cluster_stiffness": 0.2,
                                 #  "cluster_spacing": 0.2,
                                 #  "cluster_radius": 0.2,
                                 #  "link_radius": 0,
                                 #  "link_stiffness": 1.0,
                                 #  "particle_spacing": 0.025},
                                 {"$type": "assign_flex_container",
                                  "id": object_id,
                                  "container_id": 0},
                                 {'$type': 'load_primitive_from_resources',
                                  'id': 2,
                                  'primitive_type': 'Cube',
                                  'position': {'x': 0, 'y': 0.5, 'z': 0},
                                  'orientation': {'x': 0, 'y': 0, 'z': 0}},
                                 {"$type": "scale_object",
                                  "id": 2,
                                  "scale_factor": {"x": 0.5, "y": 0.5, "z": 0.5}},
                                 {'$type': 'set_kinematic_state',
                                  'id': 2,
                                  'is_kinematic': False},
                                 {"$type": "set_flex_solid_actor",
                                  "id": 2,
                                  "mesh_expansion": 0,
                                  "particle_spacing": 0.025,
                                  "mass_scale": 1},
                                 # {'$type': 'set_flex_soft_actor',
                                 #  'id': 2,
                                 #  'draw_particles': False,
                                 #  'particle_spacing': 0.125,
                                 #  'cluster_stiffness': 0.22055267521432875},
                                 {'$type': 'assign_flex_container',
                                  'id': 2,
                                  'container_id': 0},
                                 {'$type': 'set_flex_particles_mass',
                                  'id': 2,
                                  'mass': 15.625},
                                 {"$type": "send_flex_particles",
                                  "frequency": "always"},
                                 {"$type": "send_collisions",
                                  "enter": True,
                                  "stay": True,
                                  "exit": True,
                                  "collision_types": ["obj", "env"]}])
        for i in range(100):
            for j in range(len(resp) - 1):
                r_id = OutputData.get_data_type_id(resp[j])
                if r_id == "flex":
                    flex = FlexParticles(resp[j])
                    print(i, "flex", flex.get_id(0))
                elif r_id == "enco":
                    enco = EnvironmentCollision(resp[j])
                    print(i, "enco", enco.get_state())
                if r_id == "coll":
                    coll = Collision(resp[j])
                    print(i, "coll", coll.get_state())
                    if coll.get_state() == 'enter':
                        print("BAM!")
            resp = self.communicate([])


if __name__ == "__main__":
    C = FlexCollisions(launch_build=False)
    C.run()
    end = C.communicate({"$type": "terminate"})
