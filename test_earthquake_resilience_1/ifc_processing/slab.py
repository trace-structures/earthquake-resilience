
# Define class of wall created from the IFC product
from IFC_objects import IfcObject
from structural_partitions import StructuralUnit, StructuralUnitSet, SlabPartition
import numpy as np


class Slab(IfcObject):
    def __init__(self, ifc_object, create_partitions=True):
        super().__init__(ifc_object)
        # load transmitting edges
        self.load_principal_direction, self.load_transfer_type = self.get_load_transfer_direction()   #A vector of the principle direction #OneWay /TwoWay
        self.slab_partitions = StructuralUnitSet()
        if create_partitions:
          self.set_slab_partitions()

    def get_load_transfer_direction(self):
        # Here we would need to check whether the direction is given in the IFC object
        # TODO
        # If direction is not given in the IFC object, decide by dimensions
        if self.properties.load_direction is not None:
          # angle of direction (direction and global x axis)
          angle_g = self.properties.load_direction
          # angle of direction (direction and local x axis)
          angle_l = self.get_angle_of_axis()
          angle = angle_g-angle_l
          if angle == 0:
            load_principal_direction = "x_axis"
          elif angle == 90:
            load_principal_direction = "y_axis"
          else:
            raise Exception("principle load direction needs to be parallel or orthogonal to the local coordinate x")

        if self.properties.load_distribution is not None:
          load_transfer_type = self.properties.load_distribution

        if self.properties.load_direction is None and self.properties.load_distribution is None:
          dims = self.get_dimensions()
          if dims["Length"] > dims["Width"]:
            load_principal_direction = "y_axis"
            if dims["Length"] > 1.5*dims["Height"]:
              load_transfer_type = "OneWay"
            else:
              load_transfer_type = "TwoWay"
          else:
            load_principal_direction = "x_axis"
            if dims["Width"] > 1.5 * dims["Height"]:
              load_transfer_type = "OneWay"
            else:
              load_transfer_type = "TwoWay"
        return load_principal_direction, load_transfer_type

    def get_weight(self):
        p = 0
        categories = ["A", "B", "C", "D", "E", "F", "G", "H"]
        if self.properties.g1 is not None:
                p += self.properties.g1 
        if self.properties.g2 is not None:
                p += self.properties.g2
        for cat_i in categories:
          q_i = "q_"+ cat_i
          
          if hasattr(self, q_i) and getattr(self, q_i) is not None:
            val_i = getattr(self, q_i)
            p += self.get_psi_2_factor(cat_i) * val_i
        A = self.get_upper_face_area()
        return p*A

    def create_slab_partition_from_local_props(self, start_points_l, end_points_l, e_points_l, A, id, linked_elem_points=None):
        cube_points= self.transform_to_global(np.concatenate((start_points_l, end_points_l), axis=0))
        e_points = self.transform_to_global(e_points_l)
        # rho = self.properties.density
        g1 = self.properties.g1
        g2 = self.properties.g2
        q = self.properties.get_q_dict()
        return SlabPartition(cube_points, id, e_points, A,
                             g1=g1, g2=g2, q=q,
                             linked_struct_points=linked_elem_points)

    def set_slab_partitions(self):
        """
        Divides the slab into parts according to the walls it loads.
        Returns:
            HierarchicalStructuresSet: Collection of Hierarchical solid objects.
        """

        # Check whether there are more then 8 points of the slab
        if self.points.shape[0] != 8:
            # then choose the bounding
            ind_corner = self.get_bounding_box_corner_indices(tol=0.5)
            self.set_points(self.points[ind_corner])
            self.set_faces(self.get_faces_from_point_coords(self.points))

        # Sort local coordinates
        local_points = self.transform_to_local(self.points)
        ind_sort = self.sort_points_by_coordinates(points=local_points, sort_by_axis="xyz")

        # Get dictionary with local coordinates of partitions, of edge points where the loads are transmitted to the wall, and the surface areas
        if self.load_transfer_type == "OneWay":
            P = self.get_partition_properties_by_halving(local_points, ind_sort)

        elif self.load_transfer_type in ["TwoWay", 'TwoWays']:
            P = self.get_partition_properties_by_four_way_distribution(local_points, ind_sort)

        # Fianally, add partitions by the computed partition coordinates
        for key, coords in P.items():
          id_i = f"{self.id}_{key}"
          self.slab_partitions.add_object(
              self.create_slab_partition_from_local_props(
                  P[key]["Start"], P[key]["End"], P[key]["EPoints"],
                  P[key]["A"], id_i)
              )

        return self.slab_partitions

    def get_slab_partitions(self):
        return self.slab_partitions

    def get_partition_properties_by_halving(self, points_l, ind_sort):
        """
        Divides the slab into parts according to the walls it loads.
        Returns:
            local_coordinates of wall partitions.
        """
        # Initialize partition dictionary, list of partition areas and load transmitting faces
        P = {1: {"Start": None, "End": None, "EPoints": None, "A": None},
             2: {"Start": None, "End": None, "EPoints": None, "A": None}}

        if self.load_principal_direction == "x_axis":
            # Get face points of load bearing boundary faces
            P[1]["Start"] = points_l[ind_sort[:4]]
            P[2]["Start"] = points_l[ind_sort[-4:]]

             # Calculate end points by halving x coordinates
            P[1]["End"] = P[2]["Start"].copy()
            dx = (P[2]["Start"][:, 0] - P[1]["Start"][:, 0]) / 2
            P[1]["End"][:, 0] = P[1]["Start"][:, 0] + dx
            P[2]["End"] = P[1]["End"]

        else:
            # Get face points of load bearing boundary faces
            P[1]["Start"] = points_l[ind_sort[[0,1,4,5]]]  # Starting points for P1
            P[2]["Start"] = points_l[ind_sort[[2,3,6,7]]]  # Starting points for P2

            # Calculate end points by halving y coordinates
            P[1]["End"] = P[2]["Start"].copy()
            dy = (P[2]["Start"][:, 1] - P[1]["Start"][:, 1]) / 2
            P[1]["End"][:, 1] =  P[1]["Start"][:, 1] + dy
            P[2]["End"] = P[1]["End"]

        # define points of slab edge (bottom edge) where loads are transmitted to the wall
        P[1]["EPoints"] = P[1]["Start"][[0,2],:]
        P[2]["EPoints"] = P[2]["Start"][[0,2],:]

        # get area of floor
        dim = self.get_dimensions()
        # compute area of wall partitions
        A = dim["Length"]*dim["Width"]/2
        P[1]["A"] = A
        P[2]["A"] = A

        return P

    def get_partition_properties_by_four_way_distribution(self, points_l, ind_sort):
        """
        Divides the slab into four parts to the walls it loads
            Returns:
            local_coordinates of wall partitions.
        """
       # Initialize partition dictionary, list of partition areas and load transmitting faces
        P = {i: {"Start": None, "End": None} for i in range(1, 5)}

        # get area of floor
        dim = self.get_dimensions()

        # Boundary faces of the slab
        P[1]["Start"] = points_l[ind_sort[:4]]         # x-direction start face
        P[2]["Start"] = points_l[ind_sort[-4:]]         # x-direction end face
        P[3]["Start"] = points_l[ind_sort[[0,1,4,5]]]  # y-direction start face
        P[4]["Start"] = points_l[ind_sort[[2,3,6,7]]]  # y-direction end face

        for i in range(1, 5):
            P[i]["EPoints"] = P[i]["Start"][[0,2],:]

        if self.load_principal_direction == "x_axis":

            # Define trapezoidal elements
            P[1]["End"] =  P[2]["Start"].copy()
            # change x coordinates by halving
            dx = ( P[2]["Start"] [:, 0] - P[1]["Start"][:, 0]) / 2
            P[1]["End"][:, 0] = dx + P[1]["Start"][:, 0]
            # change y coordinate by adding/subtracting delta x/2 to y coordinates)
            P[1]["End"][[0,1], 1] = dx[[0,1]] + P[1]["Start"][[0,1], 1]
            P[1]["End"][[2,3], 1] = -dx[[2,3]] + P[1]["Start"][[2,3], 1]

            # Set remaining end points
            P[2]["End"] = P[1]["End"]    #trapezoidal element inner points
            P[3]["End"] = P[1]["End"][[0,1]]  # triangular element inner points
            P[4]["End"] = P[1]["End"][[2,3]]   # triangular element inner points

            # Compute partition surface areas
            # first the trapezoids
            a = dim["Width"] # dy trapezoid basis longer
            c = dim["Width"] - dim["Length"]   # dy trapezoid basis shorter
            m =  dim["Length"]/2 #dx height of trapezoid
            A_1 = (a+c)*m/2
            A_2 = A_1
            # then the triangles
            A_3 = dim["Length"]*dim["Length"]/4
            A_4 = A_3

        elif self.load_principal_direction == "y_axis":

            # Define trapezoidal elements
            P[3]["End"] =  P[4]["Start"].copy()
            # change y coordinates by halving
            dy = ( P[4]["Start"] [:, 1] - P[3]["Start"][:, 1]) / 2
            P[3]["End"][:, 1] = dy + P[3]["Start"][:, 1]
            # change x coordinate by adding/subtracting delta y/2 to x coordinates)
            P[3]["End"][[0,1], 0] = dy[[0,1]] + P[3]["Start"][[0,1], 0]
            P[3]["End"][[2,3], 0] = -dy[[2,3]] + P[3]["Start"][[2,3], 0]

            # Set remaining end points
            P[4]["End"] = P[3]["End"] #trapezoidal element inner points
            P[1]["End"] = P[3]["End"][[0,1]] # triangular element inner points
            P[2]["End"] = P[3]["End"][[2,3]]  # triangular element inner points

            # Compute partition surface areas
            # first the trapezoids
            a = dim["Length"] # dx trapezoid basis longer
            c = dim["Length"] - dim["Width"]   # dy trapezoid basis shorter
            m =  dim["Width"]/2 #dx height of trapezoid
            A_3 = (a+c)*m/2
             # then the triangles
            A_4 = A_3
            A_1 = dim["Width"]*dim["Width"]/4
            A_2 = A_1

        P[1]["A"] = A_1
        P[2]["A"] = A_2
        P[3]["A"] = A_3
        P[4]["A"] = A_4

        return P

    def find_slab_partition_by_id(self, id):
        for SP_i in self.slab_partitions.objects:
            if SP_i.id == id:
                return SP_i
        return None

    def __repr__(self):
        return f"<Slab #{self.ifc_obj.id()}>"

