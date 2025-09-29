from .simple_objects import SolidObjectSet
from .slab import Slab
from .wall import Wall
import numpy as np
from collections import defaultdict
import pandas as pd
from json import load
from .IFC_objects import IfcObject


class IfcObjectSet(SolidObjectSet):
    def __init__(self):
        super().__init__()
        self._seen_ids = set()

    def add_objects(self, obj_list, only_geometry=False, verbose=False):
        if not isinstance(obj_list, list):
            obj_list = [obj_list]
        for obj in obj_list:
            self.add_object(obj, only_geometry=False, verbose=False)
        return self.objects

    def add_object(self, obj, only_geometry=False, verbose=False):
        gid = obj.GlobalId
        if gid in self._seen_ids:
            print(f"Duplicate ID: {gid}")
            return
        self._seen_ids.add(gid)
        if obj.is_a("IfcBuildingElement"):
            if obj.is_a("IfcWall"):
                self.objects.append(Wall(obj))
            elif obj.is_a("IfcSlab"):
                self.objects.append(Slab(obj, create_partitions=not only_geometry))
                if verbose:
                    print(f"added {obj}")
            else:
                self.objects.append(IfcObject(obj))

    ################################################################################
    #            Basic methods to add/filter objects
    ################################################################################

    def filter_objects_by_type(self, ifc_type, objs=None):
        if objs is None:
            objs = self.objects
        filtered_objects = [obj for obj in objs if obj.ifc_obj.is_a(ifc_type)]
        return filtered_objects

    def filter_walls_by_floor(self, floor, objs=None):
        if objs is None:
            objs = self.objects
        walls = self.filter_objects_by_type("IfcWall", objs=objs)
        filtered_objects = [W_i for W_i in walls if W_i.floor == floor]
        return filtered_objects

    def filter_objects_by_floor(self, floor, objs=None):
        if objs is None:
            objs = self.objects
        filtered_objects = [obj for obj in objs if obj.floor == floor]
        return filtered_objects

    def find_object_by_id(self, id):
        for obj in self.objects:
            if obj.ifc_obj.id() == id:
                return obj
        return None

    ################################################################################
    #            methods to generate mesh
    ################################################################################

    def get_all_meshes_highlighting_elements_with_id(self, elem_id=[], objects=None):
        meshes = []
        if not isinstance(elem_id, list):
            elem_id = [elem_id]
        for id_i in elem_id:
            main_obj = self.find_object_by_id(id_i)
            meshes.append(main_obj.get_mesh(color="orange", opacity=1))
        # Other (not highlighted) elements
        if objects is None:
            objects = self.objects
        for obj in objects:
            if obj.id not in elem_id:
                meshes.append(obj.get_mesh(color="grey", opacity=0.3))
        return meshes

    def get_mesh_by_type(self, obj_type, opacity=0.8):
        filtered_objects = self.filter_objects_by_type(obj_type)
        meshes = []
        for obj in filtered_objects:
            meshes.append(obj.get_mesh(color=self.random_pastel(), opacity=opacity))
        return meshes

    def get_mesh_of_wall_partitions(self, mode="with_loading_elements", opacity=0.8):
        WP_meshes = []
        walls = self.filter_objects_by_type("IfcWall")
        for W in walls:
            WPs = W.create_wall_partitions()
            WP_meshes.extend(WPs.get_meshes(mode=mode, opacity=opacity))
        return WP_meshes

    def get_mesh_by_floor(self, floor, color=None, opacity=0.8):
        filtered_objects = self.filter_objects_by_floor(floor)
        meshes = self.get_meshes(objects=filtered_objects, color=color, opacity=opacity)
        return meshes

    def get_meshes(self, objects=None, color=None, opacity=0.8):
        if objects is None:
            objects = self.objects
        meshes = []
        for obj in objects:
            meshes.append(obj.get_mesh(color=color, opacity=0.8))
        return meshes


class Building(IfcObjectSet):
    def __init__(self, ifc_file, only_geometry=False, verbose=False):
        super().__init__()
        self.ifc_file = ifc_file
        self._seen_ids = set()
        if ifc_file is not None:
            self.collect_ifc_objects(only_geometry=only_geometry, verbose=verbose)
        if not only_geometry:
            if verbose:
                print("Grouping wall by floor, getting floor heights")
            self.floor_wall_dict, self.floor_heights = (
                self.group_walls_by_floor_get_floor_heights()
            )
            if verbose:
                print("Setting floor levels of slabs")
            self.set_floor_of_slabs()

            self.slab_wall_links_df = None
            self.wall_wall_links_df = None
            if verbose:
                print("Linking slabs with supporting walls")
            self.slab_wall_links_df = self.get_slab_wall_links()
            if verbose:
                print("Linking walls with supporting walls")
            self.wall_wall_links_df = self.get_wall_below_wall_links()
            if verbose:
                print("Setting transmitting distributed loads")
            self.set_transmitting_distributed_loads()
            if verbose:
                print("Creating wall partitions")
            self.create_wall_partitions()
            if verbose:
                print("Setting floor ids of wall partitions")
            self.set_floor_ids_of_wall_partitions()

    def collect_ifc_objects(
        self, ifc_type="IfcProduct", only_geometry=False, verbose=False
    ):
        for obj in self.ifc_file.by_type(ifc_type):
            self.add_object(obj, only_geometry=only_geometry, verbose=verbose)
        return self.objects

    ################################################################################
    #        Methods related to interconnection between elements
    ################################################################################

    def set_floor_ids_of_wall_partitions(self):
        WP_counter = 0
        for floor in self.floor_wall_dict:
            walls = self.filter_walls_by_floor(floor)
            for W in walls:
                WPs = W.get_wall_partitions()
                for i, WP_i in enumerate(WPs.objects):
                    if floor == 0:
                        WP_counter += 1
                        WP_i.set_floor_id(WP_counter)
                    else:
                        WP_i.set_floor_id(WP_i.supporting_elements[0].floor_id)
                    # with this numbering it is not resolved when WPs are supported
                    # by several WPs. If this is the case, it can happen that two WPs get the same numbering

    def set_transmitting_distributed_loads(self):
        # Loop over floors from top to bottom
        for floor in reversed([floor for floor in self.floor_wall_dict]):
            walls = self.filter_walls_by_floor(floor)
            for W_i in walls:
                W_i.set_transmitting_distributed_loads()

    def get_slab_wall_links(self):
        if self.slab_wall_links_df is not None:
            return self.slab_wall_links_df

        slabs = self.filter_objects_by_type("IfcSlab")
        walls = self.filter_objects_by_type("IfcWall")
        S_W_links = []
        for S_k in slabs:
            for SP_i in S_k.slab_partitions.objects:
                e_points_i = SP_i.e_points
                for W_j in walls:
                    # Check if any edge point of the partition is close to the wall's top face
                    tol_z = S_k.get_dimensions()["Height"] + 0.02
                    is_linked, contact_info = W_j.is_edge_on_wall_top(
                        e_points_i, tol_z=tol_z
                    )

                    if is_linked:
                        entry = {}
                        entry["SlabId"] = S_k.id
                        entry["SlabPartitionId"] = SP_i.id
                        entry["ContactLength"] = contact_info["ContactLength"]
                        entry["ContactStart"] = contact_info["ContactStart"]
                        entry["ContactEnd"] = contact_info["ContactEnd"]
                        entry["WallLength"] = (W_j.get_dimensions()["Length"],)
                        entry["SlabEdgeLength"] = np.linalg.norm(
                            e_points_i[0] - e_points_i[1]
                        )
                        entry["SlabHeight"] = S_k.get_dimensions()["Height"]
                        entry["LoadedWallId"] = W_j.id
                        W_j.add_loading_slab_partition(
                            {
                                "Object": SP_i,
                                "ContactLength": contact_info["ContactLength"],
                            }
                        )
                        SP_i.add_supporting_elements([W_j])
                        S_W_links.append(entry)
        return pd.DataFrame(S_W_links)

    def get_wall_slab_links(self):
        df = self.slab_wall_links_df
        wall_slab_links = df.groupby("LoadedWallId").agg(
            {
                "SlabId": lambda x: sorted(list(set(x))),
                "SlabPartitionId": lambda x: sorted(list(set(x))),
            }
        )
        return wall_slab_links

    def get_wall_below_wall_links(self):
        if self.wall_wall_links_df is not None:
            return self.wall_wall_links_df
        W_W_links = []
        # loop over floors from the top to the bottom
        for floor in reversed([floor for floor in self.floor_wall_dict if floor != 0]):
            walls_floor = self.filter_walls_by_floor(floor)
            walls_bellow_floor = self.filter_walls_by_floor(floor - 1)
            for W_i in walls_floor:  # loop over walls within floor
                axis_points = W_i.get_lower_face_axis_points()
                supported = False
                for W_b_j in walls_bellow_floor:
                    is_linked, contact_info = W_b_j.is_edge_on_wall_top(
                        axis_points, tol_z=0.3
                    )
                    if is_linked:  # np.allclose(axis_points, axis_points_bellow):
                        supported = True
                        entry = {}
                        entry["LoadingWallId"] = W_i.id
                        entry["SupportingWallId"] = W_b_j.id
                        entry["ContactLength"] = contact_info["ContactLength"]
                        entry["WallLength"] = W_i.get_dimensions()["Length"]

                        W_i.add_supporting_wall(
                            {
                                "Object": W_b_j,
                                "ContactLength": contact_info["ContactLength"],
                                "ContactStart": contact_info["ContactStart"],
                                "ContactEnd": contact_info["ContactEnd"],
                            }
                        )
                        entry["ContactStart"] = contact_info["ContactStart"]
                        entry["ContactEnd"] = contact_info["ContactEnd"]
                        W_b_j.add_loading_wall(
                            {
                                "Object": W_i,
                                "ContactLength": contact_info["ContactLength"],
                                "ContactStart": contact_info["ContactStart"],
                                "ContactEnd": contact_info["ContactEnd"],
                            }
                        )
                        W_W_links.append(entry)
            if not supported:
                W_W_links.append(
                    {"loading_wall_id": W_i.id, "supporting_wall_id": None}
                )

        return pd.DataFrame(W_W_links)

    def get_WP_WP_links(self):
        records = []
        for floor in reversed([floor for floor in self.floor_wall_dict if floor != 0]):
            walls = self.filter_walls_by_floor(floor)
            for W_i in walls:
                for WP_j in W_i.wall_partitions.objects:
                    record = {}
                    record["WallIfcId"] = W_i.id
                    record["PartitionNumber"] = WP_j.id
                    record["FloorId"] = WP_j.floor_id
                    record["SupportingElement"] = [
                        wp.id for wp in WP_j.supporting_elements
                    ]
                    record["SupportingElementFloorId"] = [
                        wp.floor_id for wp in WP_j.supporting_elements
                    ]
                    # Add record to list
                    records.append(record)
        df_WP_WP = pd.DataFrame(records)
        df_WP_WP.set_index("FloorId")
        return df_WP_WP

    ################################################################################
    #        Generate wall partition, get its properties
    ################################################################################

    def create_wall_partitions(self):
        walls = self.filter_objects_by_type("IfcWall")

        # loop over floors from the bottom
        for floor, walls_floor in self.floor_wall_dict.items():
            for W_i in walls_floor:
                # Create wall partitions
                W_i.set_wall_partitions()

    def get_wall_partition_properties(self, detailed=False):
        # initiate list of dataframes for the different floors
        prop_dfs = {}
        # Loop over floors from top to bottom
        for floor in self.floor_wall_dict:
            walls = self.filter_walls_by_floor(floor)
            records = []
            for W_j in walls:
                WPs = W_j.get_wall_partitions()
                if WPs is None or WPs.objects == []:
                    continue
                for i, WP_i in enumerate(WPs.objects):  # loop over wall partitions
                    # set floor id, to have the same id for walls that have the same x,y position
                    record = {}
                    points = WP_i.points
                    record["WallIfcId"] = W_j.id
                    record["PartitionNumber"] = WP_i.id
                    # record['PartitionFloorId'] = WP_i.floor_id
                    record["Wall"] = WP_i.floor_id
                    dim_ji = W_j.get_dimensions(points=points)
                    L_ji = dim_ji["Length"]
                    w_ji = dim_ji["Width"]
                    h_ji = dim_ji["Height"]
                    # record.update(dim_ji)
                    record["L [m]"] = L_ji
                    record["w [m]"] = w_ji
                    record["H [m]"] = h_ji
                    record["Cx [m]"] = dim_ji["CenterPointX"]
                    record["Cy [m]"] = dim_ji["CenterPointY"]
                    # Angle of wall (degree between global x axis and local wall axis)
                    record["α"] = W_j.get_angle_of_axis()
                    # wall density
                    # record["Density [kg/m3]"] = W_j.properties.density
                    record["γ [kN/m3]"] = W_j.properties.density * 9.81 / 1e3
                    # normal stresses on the wall bottom
                    sigma_SLU = W_j.get_limit_state_stress(limit_state="SLU")
                    sigma_SLE = W_j.get_limit_state_stress(limit_state="SLE")
                    record["σ {SLU} [N/mm2]"] = (
                        sigma_SLU / 1e6 if sigma_SLU is not None else None
                    )
                    record["s [N/mm2]"] = (
                        sigma_SLE / 1e6 if sigma_SLE is not None else None
                    )
                    # Wall elastic moduli
                    record["E [N/mm2]"] = (
                        W_j.properties.E / 1e6 if W_j.properties.E is not None else None
                    )
                    # wall's ShearModulus
                    record["G [N/mm2]"] = (
                        W_j.properties.G / 1e6 if W_j.properties.G is not None else None
                    )
                    # wall's TensileStrength
                    record["s [N/mm2]"] = (
                        W_j.properties.f_u / 1e6
                        if W_j.properties.f_u is not None
                        else None
                    )
                    # wall's ShearStrength
                    record["t [N/mm2]"] = (
                        W_j.properties.f_u / 1.5e6
                        if W_j.properties.f_u is not None
                        else None
                    )
                    # wall's compressive strength
                    record["fm [N/mm2]"] = (
                        W_j.properties.f_c / 1e6
                        if W_j.properties.f_c is not None
                        else None
                    )
                    # wall's PoissonRatio
                    record["m"] = W_j.properties.nu
                    # Add record to list
                    records.append(record)
            df = pd.DataFrame(records)
            if not detailed:
                df = df[
                    [
                        "Wall",
                        "L [m]",
                        "w [m]",
                        "H [m]",
                        "Cx [m]",
                        "Cy [m]",
                        "α",
                        "s [N/mm2]",
                        "t [N/mm2]",
                        "fm [N/mm2]",
                        "γ [kN/m3]",
                        "E [N/mm2]",
                        "G [N/mm2]",
                        "m",
                    ]
                ]
            prop_dfs[floor] = df

        return prop_dfs

    ################################################################################
    #  Methods to identify floors, define its heights, categorize elements by floor
    ################################################################################

    def group_walls_by_floor_get_floor_heights(self):
        walls = super().filter_objects_by_type("IfcWall")
        # Create array of [bottom_height, top_height]
        hs = []
        for W in walls:
            # coordinate of wall bottom and top
            z_b = np.round(W.get_bottom_height(), decimals=1)
            z_t = np.round(W.get_top_height(), decimals=1)
            hs.append([z_b, z_t])
        hs = np.array(hs)

        # Find unique height pairs and inverse indices
        floor_hs, inv_ind = np.unique(hs, axis=0, return_inverse=True)

        # Sort by bottom height and remap to floor numbers
        sort_ind = np.argsort(floor_hs[:, 0])  # sort by bottom height
        # floor_hs_sorted = [tuple(floor_hs[i]) for i in sort_ind]
        floor_hs_dict = {}
        for i, ind_i in enumerate(sort_ind):
            floor_hs_dict[i] = {
                "BottomHeight": floor_hs[i, 0],
                "TopHeight": floor_hs[i, 1],
            }

        floor_no_map = {tuple(floor_hs[i]): idx for idx, i in enumerate(sort_ind)}

        # Group walls by those unique indices
        floor_wall_dict = defaultdict(list)
        for idx, W in zip(inv_ind, walls):
            key = tuple(floor_hs[idx])
            floor_no = floor_no_map[key]
            W.set_floor(floor_no)
            floor_wall_dict[floor_no].append(W)

        return floor_wall_dict, floor_hs_dict

    def set_floor_of_slabs(self):
        slabs = super().filter_objects_by_type("IfcSlab")
        h_ground = self.floor_heights[0]["BottomHeight"]
        for S in slabs:
            h_top = S.get_top_height()
            h_bottom = S.get_bottom_height()
            if np.isclose(h_top, h_ground) or np.isclose(h_bottom, h_ground):
                S.set_floor(0)
                continue
            for floor, props in self.floor_heights.items():
                h_floor = props["TopHeight"]
                if np.isclose(h_top, h_floor, atol=0.2) or np.isclose(
                    h_bottom, h_floor, atol=0.2
                ):
                    S.set_floor(floor + 1)
                    break

    ################################################################################
    #            compute floor weights
    ################################################################################
    def get_floor_weights(self):
        slabs = self.filter_objects_by_type("IfcSlab")
        floor_weight_dict = {}
        for floor in self.floor_wall_dict:
            w = 0
            # get slab weight
            slabs_i = super().filter_objects_by_floor(floor + 1, objs=slabs)
            for S_ji in slabs_i:
                w += S_ji.get_weight()
            # get lower wall top half weight
            walls_b_i = super().filter_walls_by_floor(floor)
            for W_b_ji in walls_b_i:
                w += W_b_ji.get_upper_half_weight()
            # get upper wall top half weight
            if floor != max(self.floor_wall_dict):
                walls_t_i = super().filter_walls_by_floor(floor + 1)
                for W_t_ji in walls_t_i:
                    w += W_t_ji.get_lower_half_weight()
            # For the last top wall
            floor_weight_dict[floor] = w

        return floor_weight_dict

    def get_floor_properties(self):
        floor_props = {}
        floor_heights = self.floor_heights
        floor_weights = self.get_floor_weights()
        for floor in floor_heights:
            floor_props[f"Floor{floor+1}"] = {
                "H [m]": floor_heights[floor]["TopHeight"]
                - floor_heights[floor]["BottomHeight"],
                "W [kN]": floor_weights[floor] / 1000,
            }
        return pd.DataFrame(floor_props).T.reset_index().rename(columns={"index": "Floor"})

    def __repr__(self):
        return f"<IfcObjectSet with {len(self.objects)} objects>"
