
import numpy as np
import plotly.graph_objects as go
from scipy.spatial import ConvexHull
import random

class SolidObject:
    def __init__(self, coords):
        self.points = coords
        self.faces = self.get_faces_from_point_coords()
       
    def get_bottom_height(self):
        return np.min(self.points[:, 2])

    def get_top_height(self):
        return np.max(self.points[:, 2])

    def get_faces_from_point_coords(self, points=None):
        if points is None:
            points = self.points
        # Calculate the hull of the points
        hull = ConvexHull(points)
        faces = hull.simplices  # These are the triangle indices
        return faces

    def get_plotly_scatterpoints(self, points=None, name=None, mode='markers', color='green', size=2):
        if points is None:
          points = self.points
        plotly_points = go.Scatter3d(
          x=points[:,0], y=points[:,1], z=points[:,2],
            mode='markers',
            marker=dict(size=size, color=color),
          name=name
        )
        return plotly_points

    def get_mesh(self, name=None, color='blue', opacity=1):
        x, y, z = self.points[:, 0], self.points[:, 1], self.points[:, 2]
        i, j, k = self.faces[:, 0], self.faces[:, 1], self.faces[:, 2]
        mesh = go.Mesh3d(
            x=x, y=y, z=z,
            i=i, j=j, k=k,
            color=color,
            opacity=opacity,
            name=name
        )
        return mesh

    def __repr__(self):
        return f"<SolidObject with {len(self.points)} points and {len(self.faces)} faces>"

class SolidObjectSet:
    def __init__(self):
        self.objects = []
        self.points = []
        self.meshes = []

    def add_object(self, obj):
        if isinstance(obj, SolidObject):
            self.objects.append(obj)
        else:
            print(type(obj))
            print(obj)
            raise TypeError("Only SolidObject objects can be added.")
        return self.objects

    def remove_object_by_index(self, ind):
        del self.objects[ind]

    def get_all_points(self):
        for obj in self.objects:
            self.points.extend(obj.points)
        self.points = np.array(self.points)
        return self.points

    def random_pastel(self):
        r = lambda: random.randint(100, 255)
        return f'rgb({r()},{r()},{r()})'

    def get_plotly_scatterpoints(self, points=None, mode='markers', color='green', size=2):
        if points is None:
          points = self.get_all_points()
        plotly_points = go.Scatter3d(
          x=points[:,0], y=points[:,1], z=points[:,2],
            mode='markers',
            marker=dict(size=size, color=color)
        )
        return plotly_points

    def get_all_meshes(self, opacity=0.8):
        for obj in self.objects:
            self.meshes.append(obj.get_mesh(color=self.random_pastel(), opacity=opacity))
        return self.meshes
