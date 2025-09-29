
from simple_objects import SolidObject, SolidObjectSet
import numpy as np

# Class to define wall partitions and the
# belonging  structural segments (loading cuboids) that load this part
class StructuralUnit(SolidObject):
    def __init__(self, coords, id, linked_struct_points=None, struct_type='LoadBearing', supporting_elements=None):
        super().__init__(coords)
        self.id = id
        self.floor_id = None
        self.supporting_elements = []
        if supporting_elements is not None:
          self.add_supporting_elements(supporting_elements)
        self.struct_type = struct_type
        if linked_struct_points is None:
            self.linked_struct = []
        else:
            self.set_linked_struct(linked_struct_points)

    def set_floor_id(self, floor_id):
        self.floor_id = floor_id

    def set_linked_struct(self, linked_struct_points):
        if isinstance(linked_struct_points, list):
            self.linked_struct = [StructuralUnit(points_i, f"{self.id}_{i}", struct_type='Loading')
            for i, points_i in enumerate(linked_struct_points)]
        else:
            self.linked_struct = [StructuralUnit(linked_struct_points, f"{self.id}_1", struct_type='Loading')]

    def add_linked_struct(self, linked_struct_points):
        l = len(self.linked_struct)
        self.linked_struct.append(StructuralUnit(linked_struct_points, f"{self.id}_{l+1}", struct_type='Loading'))

    def add_supporting_elements(self, supporting_elements):
        self.supporting_elements += supporting_elements

    def get_mesh_with_linked_structs(self, base_name="HierarchicalStructure", color='blue'):
        meshes = [self.get_mesh(name=f'{base_name} - Main',
                  color=color)]
        if hasattr(self, 'linked_struct'):
            for i, part in enumerate(self.linked_struct):
                meshes.append(part.get_mesh(name=part.id,
                                            color=color, opacity=0.6,
                                            ).update(
                            hovertext=part.id,
                            hoverinfo='text'))
        return meshes
        
        
class StructuralUnitSet(SolidObjectSet):
    def __init__(self):
        super().__init__()

    def add_linked_structure(self, h_structure):
        if isinstance(h_structure, StructuralUnit):
            self.objects.append(h_structure)
        else:
            raise TypeError("Only HierarchicalStructures objects can be added.")

    def get_meshes(self, mode="only_main", opacity=0.8):
      meshes = []
      if mode == "only_main":
        for i, obj in enumerate(self.objects):
          meshes.append(obj.get_mesh(name=f"Part {i}",
                                      color= self.random_pastel(), opacity=opacity))
      else:
        for i, obj in enumerate(self.objects):
          color = self.random_pastel()
          meshes.extend(obj.get_mesh_with_linked_structs(base_name=obj.id, color=color))
      return meshes

      def __repr__(self):
        return f"<HierarchicalStructuresSet with {len(self.objects)} objects>"


class SlabPartition(StructuralUnit):
    def __init__(self, coords, id, e_points, A, g1=0, g2=0, q=0, linked_struct_points=None):
      """
      rho is the density
      g1, g2 is the structural and non-structural uniform pernament surface load on the structure
      q is a dictionary with keys corresponding to different categories (A, B, ..H)
      and values of the vertical unifrom surface load on the slab partition
      """
      super().__init__(coords, id, linked_struct_points)
      # Load transmitting edge points
      self.e_points = e_points
      # Area of partitions
      self.A = A
      # Total vertical load at the transmitting edge [N] and distributed force [N/m]
      self.G1, self.p_g1 = self.get_edge_load(g1)
      self.G2, self.p_g2 = self.get_edge_load(g2)
      self.Q, self.p_q = self.get_edge_load_from_q(q)

    def get_edge_load(self, p):
      if p is None:
        return None, None
      F_p = p * self.A
      L = distance = np.linalg.norm(self.e_points[1] - self.e_points[0])
      f_p  = F_p/L
      return F_p, f_p

    def get_edge_load_from_q(self, q):
      F_q = {cat_i: self.get_edge_load(val_i)[0] for cat_i, val_i in q.items()}
      f_q = {cat_i: self.get_edge_load(val_i)[1] for cat_i, val_i in q.items()}
      return F_q, f_q
        