import plotly.graph_objects as go
from shapely.geometry import LineString, Polygon   # for matching wall below wall connections
import numpy as np

def plot_by_plotly(data, title, showlegend=True):
    fig = go.Figure(data=data)
    fig.update_layout(
        title = title,
        showlegend=showlegend,
        margin=dict(l=0, r=0, b=0, t=30),
        scene=dict(
          xaxis_title='X',
          yaxis_title='Y',
          zaxis_title='Z',
          aspectmode='data'
          )
        )
    return fig


def get_line_polygon_intersection_and_gaps(line_points, poly_points):
    # Create polygon
    poly = Polygon([tuple(pt) for pt in poly_points])
    if not poly.is_valid:
        poly = poly.buffer(0)
        if not poly.is_valid:
            raise ValueError("Invalid polygon that cannot be fixed")

    # Create line
    line = LineString([tuple(pt) for pt in line_points])
    if len(line.coords) != 2:
        raise ValueError("Line must have exactly two distinct points")

    # Inside part
    inter = line.intersection(poly)
    contact_segments = None
    if inter.is_empty:
        contact_segments = None
    elif inter.geom_type == "LineString":
        contact_segments = [np.array(inter.coords)]
    elif inter.geom_type == "MultiLineString":
        contact_segments = [np.array(seg.coords) for seg in inter.geoms]

    # Outside part
    diff = line.difference(poly)
    unsupported_segments = []
    if diff.is_empty:
        unsupported_segments = []
    elif diff.geom_type == "LineString":
        unsupported_segments = [np.array(diff.coords)]
    elif diff.geom_type == "MultiLineString":
        unsupported_segments = [np.array(seg.coords) for seg in diff.geoms]

    return contact_segments, unsupported_segments


def segments_overlap(seg1, seg2, tol=1e-6):
    """
    Return the overlap (start, end) of two 1D segments if they intersect.

    Returns:
      (start, end) of the overlapping section, or None if no overlap.

    """
    start1, end1 = sorted([seg1[0], seg1[1]])
    start2, end2 = sorted([seg2[0], seg2[1]])
    overlap_start = max(start1, start2)
    overlap_end = min(end1, end2)
    if overlap_start > overlap_end:
      return None
    else:
      return (overlap_start, overlap_end)