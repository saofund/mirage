"""Differential test: the C++ op-log (mirage::Program) vs the Python op-log
(meshlang.MeshProgram).

The whole point of the dual-operator architecture is ONE op-log dialect with TWO
engines (a human's native GUI and an AI's Python/MCP). This test is the proof:
the SAME JSON op-log, replayed by each engine, must yield the SAME mesh. The
Python meshlang is the truth oracle; the native Program must match it exactly, so
the C++ replay (selection-as-query, durable tags, region operators) can be
refactored freely without drift. It also exercises the native selection engine
directly (selector_count) and the JSON round-trip (the bridge itself).

Skipped if the pybind11 module hasn't been built (configure core/ with
-Dpybind11_DIR=... and build the _mirage_core target).
"""
import json
import os
import sys

import pytest

_PYD = os.path.join(os.path.dirname(__file__), "..", "core", "build", "Release")
if os.path.isdir(_PYD):
    sys.path.insert(0, _PYD)

cpp = pytest.importorskip("_mirage_core")

from mirage.meshlang import MeshProgram, resolve, Sel  # noqa: E402
from mirage.kernel import make_cube, make_cylinder_ngon, make_uv_sphere  # noqa: E402


def _s(stats):
    return (stats["verts"], stats["edges"], stats["faces"], stats["euler"], stats["closed_manifold"])


def _cutter(mesh, dx=0.0, dy=0.0, dz=0.0):
    """Bake a built mesh into (verts, faces) for a boolean op's operand B, optionally
    translated — so the differential cutters are defined once and stay in sync."""
    verts = [[v.co[0] + dx, v.co[1] + dy, v.co[2] + dz] for v in mesh.verts]
    faces = [[lp.vert.id for lp in mesh.face_loops(f)] for f in mesh.faces]
    return verts, faces


_CUBE_CUT = _cutter(make_cube(1.0), 0.5, 0.5, 0.5)           # a corner-overlapping cube
_DRILL = _cutter(make_cylinder_ngon(16, 0.3, 2.0))          # a thin tall cylinder (a drill bit)
_BALL = _cutter(make_uv_sphere(16, 12, 0.62))               # a sphere to round a cube


# Op-logs expressed in the shared dialect (no "near" — that selector is the
# GUI/click recording, C++-only; everything here is common to both engines).
OPLOGS = {
    "cube": [
        {"op": "cube", "size": 1.0},
    ],
    "cylinder": [
        {"op": "cylinder", "sides": 8, "radius": 0.5, "height": 1.0},
    ],
    "boss_normal_last": [
        {"op": "cube", "size": 1.0},
        {"op": "inset", "on": {"by": "normal", "axis": "z"}, "thickness": 0.3},
        {"op": "extrude", "on": {"by": "last_created"}, "distance": 0.6},
    ],
    "boss_extreme": [
        {"op": "cube", "size": 1.0},
        {"op": "inset", "on": {"by": "extreme", "axis": "z", "which": "max"}, "thickness": 0.25},
        {"op": "extrude", "on": {"by": "last_created"}, "distance": 0.5},
    ],
    "tag_then_select": [
        {"op": "cube", "size": 2.0},
        {"op": "tag", "on": {"by": "normal", "axis": "z", "sign": 1.0}, "name": "lid"},
        {"op": "extrude", "on": {"by": "tag", "name": "lid"}, "distance": 0.4},
    ],
    "side_extrude": [
        {"op": "cube", "size": 1.0},
        {"op": "extrude", "on": {"by": "side", "axis": "x", "sign": 1.0}, "distance": 0.7},
    ],
    "inset_all": [
        {"op": "cube", "size": 1.0},
        {"op": "inset", "on": {"by": "all"}, "thickness": 0.2},
    ],
    "subdivide": [
        {"op": "cube", "size": 1.0},
        {"op": "subdivide", "levels": 1},
    ],
    "subdivide_then_op": [
        {"op": "cylinder", "sides": 6, "radius": 0.5, "height": 1.0},
        {"op": "subdivide", "levels": 1},
        {"op": "inset", "on": {"by": "normal", "axis": "z", "sign": 1.0, "tol": 0.3}, "thickness": 0.2},
    ],
    "and_not_combo": [
        {"op": "cube", "size": 1.0},
        {"op": "extrude",
         "on": {"and": [{"by": "normal", "axis": "z", "sign": 1.0},
                        {"not": {"by": "side", "axis": "x", "sign": 1.0}}]},
         "distance": 0.5},
    ],
    "translate_scale": [
        {"op": "cube", "size": 1.0},
        {"op": "scale", "on": {"by": "normal", "axis": "z"}, "by": [1.5, 1.5, 1.0]},
        {"op": "translate", "on": {"by": "extreme", "axis": "z", "which": "max"}, "by": [0.0, 0.0, 0.3]},
    ],
    "dir_normal": [
        {"op": "cube", "size": 1.0},
        {"op": "inset", "on": {"by": "normal", "dir": [0.0, 0.0, 1.0], "tol": 0.4}, "thickness": 0.3},
    ],
    "marked_handle": [
        {"op": "cube", "size": 1.0},
        {"op": "inset", "on": {"by": "normal", "axis": "z"}, "thickness": 0.3, "mark": "panel"},
        {"op": "extrude", "on": {"by": "tag", "name": "panel"}, "distance": 0.4},
    ],
    "bevel_top": [
        {"op": "cube", "size": 1.0},
        {"op": "bevel", "on": {"by": "normal", "axis": "z"}, "width": 0.25, "depth": 0.2},
    ],
    "bevel_then_extrude": [
        {"op": "cube", "size": 1.2},
        {"op": "bevel", "on": {"by": "normal", "axis": "z"}, "width": 0.2, "depth": 0.15},
        {"op": "extrude", "on": {"by": "last_created"}, "distance": 0.5},
    ],
    "bevel_all_negative_depth": [
        {"op": "cube", "size": 1.0},
        {"op": "bevel", "on": {"by": "all"}, "width": 0.15, "depth": -0.1},
    ],
    "loop_cut_cube": [
        {"op": "cube", "size": 1.0},
        {"op": "loop_cut", "on": {"by": "normal", "axis": "y", "sign": -1.0}, "axis": "z"},
    ],
    "loop_cut_cylinder": [
        {"op": "cylinder", "sides": 10, "radius": 0.5, "height": 1.2},
        {"op": "loop_cut", "on": {"by": "side", "axis": "x", "sign": 1.0}, "axis": "z"},
    ],
    "loop_cut_then_extrude": [
        {"op": "cube", "size": 1.0},
        {"op": "loop_cut", "on": {"by": "normal", "axis": "x", "sign": 1.0}, "axis": "z", "mark": "band"},
        {"op": "extrude", "on": {"by": "tag", "name": "band"}, "distance": 0.2},
    ],
    "edge_bevel_cube_all": [
        {"op": "cube", "size": 1.0},
        {"op": "edge_bevel", "on": {"by": "all"}, "width": 0.2},
    ],
    "edge_bevel_cube_sharp": [
        {"op": "cube", "size": 1.2},
        {"op": "edge_bevel", "on": {"by": "sharp", "angle": 30}, "width": 0.15},
    ],
    "edge_bevel_cylinder": [
        {"op": "cylinder", "sides": 8, "radius": 0.5, "height": 1.0},
        {"op": "edge_bevel", "on": {"by": "sharp", "angle": 20}, "width": 0.12},
    ],
    "edge_bevel_top_loop_mixed": [
        {"op": "cube", "size": 1.0},
        # only the 4 top edges (a closed loop) -> rounded top rim, sharp sides
        {"op": "edge_bevel", "on": {"by": "on_face", "face": {"by": "normal", "axis": "z", "sign": 1.0}}, "width": 0.2},
    ],
    "edge_bevel_cyl_rim_mixed": [
        {"op": "cylinder", "sides": 10, "radius": 0.5, "height": 1.2},
        {"op": "edge_bevel", "on": {"by": "on_face", "face": {"by": "normal", "axis": "z", "sign": 1.0}}, "width": 0.08},
    ],
    "edge_bevel_lonely_noop": [
        {"op": "cube", "size": 1.0},
        # the 4 vertical edges are pairwise disjoint -> lone cuts -> pruned -> no-op
        {"op": "edge_bevel", "on": {"by": "axis", "axis": "z"}, "width": 0.2},
    ],
    # regression: subdivided mesh has >=3-sector vertices (was manifold-lost before the corner-face fix)
    "edge_bevel_subdiv_sharp": [
        {"op": "cube", "size": 1.0},
        {"op": "subdivide", "levels": 1},
        {"op": "edge_bevel", "on": {"by": "sharp", "angle": 30}, "width": 0.08},
    ],
    "edge_bevel_subdiv_loop": [
        {"op": "cube", "size": 1.0},
        {"op": "subdivide", "levels": 1},
        {"op": "edge_bevel", "on": {"by": "on_face", "face": {"by": "normal", "axis": "z", "sign": 1.0}}, "width": 0.06},
    ],
    "edge_bevel_inset_rim": [   # inset makes inner-rim loops -> >=3-sector verts
        {"op": "cube", "size": 1.0},
        {"op": "inset", "on": {"by": "normal", "axis": "z", "sign": 1.0}, "thickness": 0.3},
        {"op": "edge_bevel", "on": {"by": "sharp", "angle": 30}, "width": 0.1},
    ],
    "edge_bevel_open_box": [    # open mesh: boundary verts pruned -> no crash, valid
        {"op": "cube", "size": 1.0},
        {"op": "delete", "on": {"by": "normal", "axis": "z", "sign": 1.0}},
        {"op": "edge_bevel", "on": {"by": "all"}, "width": 0.1},
    ],
    "edge_bevel_then_extrude": [
        {"op": "cube", "size": 1.0},
        {"op": "edge_bevel", "on": {"by": "all"}, "width": 0.18, "mark": "rounded"},
        {"op": "extrude", "on": {"by": "normal", "axis": "z", "sign": 1.0, "tol": 0.3}, "distance": 0.3},
    ],
    "plane": [
        {"op": "plane", "size_x": 1.5, "size_y": 1.0},
    ],
    "uv_sphere": [
        {"op": "uv_sphere", "segments": 12, "rings": 8, "radius": 0.6},
    ],
    "uv_sphere_min": [   # rings=2 -> just two triangle fans (a bipyramid), no quad bands
        {"op": "uv_sphere", "segments": 5, "rings": 2, "radius": 0.5},
    ],
    "cone": [
        {"op": "cone", "sides": 16, "radius": 0.5, "height": 1.2},
    ],
    "torus": [   # genus-1: euler 0, still a closed manifold
        {"op": "torus", "major_segments": 16, "minor_segments": 10, "major_radius": 0.6, "minor_radius": 0.22},
    ],
    "grid": [
        {"op": "grid", "size_x": 2.0, "size_y": 1.0, "x_div": 6, "y_div": 4},
    ],
    # primitives feed the operators identically in both engines (order-sensitive ops
    # would diverge instantly if the vertex/face ordering differed):
    "sphere_inset_extrude": [
        {"op": "uv_sphere", "segments": 10, "rings": 6, "radius": 0.5},
        {"op": "inset", "on": {"by": "normal", "axis": "z", "sign": 1.0}, "thickness": 0.3},
        {"op": "extrude", "on": {"by": "last_created"}, "distance": 0.4},
    ],
    "cone_edge_bevel_sharp": [
        {"op": "cone", "sides": 8, "radius": 0.5, "height": 1.0},
        {"op": "edge_bevel", "on": {"by": "sharp", "angle": 20}, "width": 0.05},
    ],
    "torus_loop_cut": [
        {"op": "torus", "major_segments": 12, "minor_segments": 8, "major_radius": 0.6, "minor_radius": 0.2},
        {"op": "subdivide", "levels": 1},
    ],
    "grid_extrude": [
        {"op": "grid", "size_x": 2.0, "size_y": 2.0, "x_div": 4, "y_div": 4},
        {"op": "extrude", "on": {"by": "all"}, "distance": 0.2},
    ],
    # inline geometry (the glTF-import seam): both engines from_pydata it identically
    "mesh_quad": [
        {"op": "mesh", "verts": [[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]], "faces": [[0, 1, 2, 3]]},
    ],
    "mesh_tetra": [
        {"op": "mesh",
         "verts": [[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]],
         "faces": [[0, 2, 1], [0, 1, 3], [0, 3, 2], [1, 2, 3]]},   # a closed tetrahedron
    ],
    "mesh_then_extrude": [
        {"op": "mesh", "verts": [[-1, -1, 0], [1, -1, 0], [1, 1, 0], [-1, 1, 0]], "faces": [[0, 1, 2, 3]]},
        {"op": "extrude", "on": {"by": "all"}, "distance": 0.5},     # operators stack on imported geometry
    ],
    # whole-mesh operators: solidify / mirror / array
    "plane_solidify": [
        {"op": "plane", "size_x": 1.0},
        {"op": "solidify", "thickness": 0.2},                        # an open quad -> a closed box
    ],
    "grid_solidify": [
        {"op": "grid", "size_x": 1.0, "x_div": 3, "y_div": 3},
        {"op": "solidify", "thickness": 0.15},
    ],
    "openbox_solidify": [
        {"op": "cube", "size": 1.0},
        {"op": "delete", "on": {"by": "normal", "axis": "z", "sign": 1.0}},
        {"op": "solidify", "thickness": 0.1},                        # shell with walls round the opening
    ],
    "cube_array": [
        {"op": "cube", "size": 1.0},
        {"op": "array", "count": 3, "offset": [1.2, 0.0, 0.0]},
    ],
    "mirror_plane": [
        {"op": "plane", "size_x": 1.0},
        {"op": "translate", "on": {"by": "all"}, "by": [0.5, 0.0, 0.0]},   # push a left edge onto x=0
        {"op": "mirror", "axis": "x"},                              # weld the seam -> a 2-wide sheet
    ],
    "mirror_halfcyl": [
        {"op": "cylinder", "sides": 8, "radius": 0.5, "height": 1.0},
        {"op": "delete", "on": {"by": "side", "axis": "x", "sign": -1.0}},
        {"op": "mirror", "axis": "x"},
    ],
    "solidify_then_bevel": [
        {"op": "grid", "size_x": 1.0, "x_div": 2, "y_div": 2},
        {"op": "solidify", "thickness": 0.2},
        {"op": "bevel", "on": {"by": "normal", "axis": "z", "sign": 1.0}, "width": 0.1, "depth": 0.05},
    ],
    # bisect: plane cut (Sutherland-Hodgman clip + shared edge intersections)
    "bisect_cube_open": [
        {"op": "cube", "size": 1.0},
        {"op": "bisect", "point": [0, 0, 0], "normal": [0, 0, 1]},        # keep the bottom half (open)
    ],
    "bisect_cube_fill": [
        {"op": "cube", "size": 1.0},
        {"op": "bisect", "point": [0, 0, 0], "normal": [0, 0, 1], "fill": True},   # capped -> closed
    ],
    "bisect_diagonal": [
        {"op": "cube", "size": 1.0},
        {"op": "bisect", "point": [0, 0, 0], "normal": [1, 1, 1], "fill": True},
    ],
    "bisect_sphere_dome": [
        {"op": "uv_sphere", "segments": 12, "rings": 8, "radius": 0.6},
        {"op": "bisect", "point": [0, 0, 0], "normal": [0, 0, 1], "fill": True},
    ],
    "bisect_corner": [   # two cuts -> a clipped corner
        {"op": "cube", "size": 1.0},
        {"op": "bisect", "point": [0.2, 0, 0], "normal": [1, 0, 0], "fill": True},
        {"op": "bisect", "point": [0, 0.2, 0], "normal": [0, 1, 0], "fill": True},
    ],
    "bisect_then_extrude": [
        {"op": "cube", "size": 1.0},
        {"op": "bisect", "point": [0, 0, 0], "normal": [0, 0, 1], "fill": True},
        {"op": "extrude", "on": {"by": "normal", "axis": "z", "sign": 1.0}, "distance": 0.3},
    ],
    # spin (lathe): revolve an open profile's boundary edges around an axis
    "spin_rect_tube": [   # a rectangle cross-section -> a hollow tube (genus-1)
        {"op": "mesh", "verts": [[0.4, 0, -0.5], [0.6, 0, -0.5], [0.6, 0, 0.5], [0.4, 0, 0.5]], "faces": [[0, 1, 2, 3]]},
        {"op": "spin", "axis": "z", "steps": 16, "angle": 360},
    ],
    "spin_partial_open": [
        {"op": "mesh", "verts": [[0.4, 0, -0.5], [0.6, 0, -0.5], [0.6, 0, 0.5], [0.4, 0, 0.5]], "faces": [[0, 1, 2, 3]]},
        {"op": "spin", "axis": "z", "steps": 8, "angle": 90},
    ],
    "spin_profile_to_axis": [   # profile touching the axis -> a closed solid of revolution
        {"op": "mesh", "verts": [[0.0, 0, -0.5], [0.5, 0, -0.3], [0.5, 0, 0.3], [0.0, 0, 0.5]], "faces": [[0, 1, 2, 3]]},
        {"op": "spin", "axis": "z", "steps": 16, "angle": 360},
    ],
    "spin_then_solidify": [
        {"op": "mesh", "verts": [[0.4, 0, -0.5], [0.6, 0, -0.5], [0.6, 0, 0.5], [0.4, 0, 0.5]], "faces": [[0, 1, 2, 3]]},
        {"op": "spin", "axis": "z", "steps": 12, "angle": 360},
        {"op": "solidify", "thickness": 0.05},
    ],
    "spin_y_axis": [   # a plane pushed off the y-axis, revolved around it
        {"op": "plane", "size_x": 0.3},
        {"op": "translate", "on": {"by": "all"}, "by": [0.6, 0.0, 0.0]},
        {"op": "spin", "axis": "y", "steps": 16, "angle": 360},
    ],
    # profile (a first-class 2D generatrix / wire) -> the real lathe input
    "profile_vase_single_wall": [   # an OPEN curve revolved 360 -> a single-walled open vase
        {"op": "profile", "points": [[0.0, -0.5], [0.4, -0.45], [0.25, 0.0], [0.45, 0.45], [0.3, 0.5]],
         "plane": "xz"},
        {"op": "spin", "axis": "z", "steps": 24, "angle": 360},
    ],
    "profile_open_partial": [       # the open curve swept a partial angle -> an open sheet
        {"op": "profile", "points": [[0.3, -0.4], [0.5, 0.0], [0.3, 0.4]], "plane": "xz"},
        {"op": "spin", "axis": "z", "steps": 12, "angle": 140},
    ],
    "profile_closed_ring_torus": [  # a CLOSED profile (a small loop) revolved -> a torus surface
        {"op": "profile", "points": [[0.4, -0.1], [0.6, -0.1], [0.6, 0.1], [0.4, 0.1]],
         "plane": "xz", "closed": True},
        {"op": "spin", "axis": "z", "steps": 16, "angle": 360},
    ],
    "profile_screw_thread": [       # an open profile climbed into a single-walled helix
        {"op": "profile", "points": [[0.4, -0.05], [0.55, 0.0], [0.4, 0.05]], "plane": "xz"},
        {"op": "screw", "axis": "z", "steps": 16, "turns": 2, "height": 0.3, "angle": 360},
    ],
    "profile_vase_then_solidify": [ # give the single wall thickness -> a watertight vase
        {"op": "profile", "points": [[0.05, -0.5], [0.4, -0.4], [0.3, 0.3], [0.45, 0.5]], "plane": "xz"},
        {"op": "spin", "axis": "z", "steps": 20, "angle": 360},
        {"op": "solidify", "thickness": 0.03},
    ],
    # boolean (real BSP mesh-mesh CSG): A = current mesh, B = inline cutter geometry
    "bool_union_cubes": [
        {"op": "cube", "size": 1.0},
        {"op": "boolean", "mode": "union", "verts": _CUBE_CUT[0], "faces": _CUBE_CUT[1]},
    ],
    "bool_difference_cubes": [
        {"op": "cube", "size": 1.0},
        {"op": "boolean", "mode": "difference", "verts": _CUBE_CUT[0], "faces": _CUBE_CUT[1]},
    ],
    "bool_intersection_cubes": [   # the overlap region -> a clean little box (closed, euler 2)
        {"op": "cube", "size": 1.0},
        {"op": "boolean", "mode": "intersection", "verts": _CUBE_CUT[0], "faces": _CUBE_CUT[1]},
    ],
    "bool_drill_through_cube": [   # subtract a cylinder -> a cube with a bore
        {"op": "cube", "size": 1.0},
        {"op": "boolean", "mode": "difference", "verts": _DRILL[0], "faces": _DRILL[1]},
    ],
    "bool_sphere_intersect_cube": [  # a cube rounded by a sphere intersection
        {"op": "cube", "size": 1.0},
        {"op": "boolean", "mode": "intersection", "verts": _BALL[0], "faces": _BALL[1]},
    ],
    "bool_then_bisect": [          # compose: drill, then plane-cut the result
        {"op": "cube", "size": 1.0},
        {"op": "boolean", "mode": "difference", "verts": _DRILL[0], "faces": _DRILL[1]},
        {"op": "bisect", "point": [0, 0, 0.2], "normal": [0, 0, 1]},
    ],
    # screw (helical sweep): like spin but the profile climbs along the axis
    "screw_single_turn": [   # a square cross-section -> one helical coil
        {"op": "mesh", "verts": [[0.4, 0, -0.05], [0.5, 0, -0.05], [0.5, 0, 0.05], [0.4, 0, 0.05]], "faces": [[0, 1, 2, 3]]},
        {"op": "screw", "axis": "z", "steps": 16, "turns": 1, "height": 0.4, "angle": 360},
    ],
    "screw_spring_3turns": [   # 3 coils -> a spring / auger
        {"op": "mesh", "verts": [[0.4, 0, -0.05], [0.5, 0, -0.05], [0.5, 0, 0.05], [0.4, 0, 0.05]], "faces": [[0, 1, 2, 3]]},
        {"op": "screw", "axis": "z", "steps": 12, "turns": 3, "height": 0.3, "angle": 360},
    ],
    "screw_axis_touching": [   # profile edge meeting the axis -> a climbing fan, no poles
        {"op": "mesh", "verts": [[0.0, 0, 0.0], [0.5, 0, -0.1], [0.5, 0, 0.1]], "faces": [[0, 1, 2]]},
        {"op": "screw", "axis": "z", "steps": 16, "turns": 1, "height": 0.5, "angle": 360},
    ],
    "screw_then_solidify": [   # give the helical sheet thickness -> a solid thread
        {"op": "mesh", "verts": [[0.4, 0, -0.05], [0.6, 0, -0.05], [0.6, 0, 0.05], [0.4, 0, 0.05]], "faces": [[0, 1, 2, 3]]},
        {"op": "screw", "axis": "z", "steps": 12, "turns": 2, "height": 0.25, "angle": 360},
        {"op": "solidify", "thickness": 0.03},
    ],
    "screw_y_axis": [   # climb along y instead of z
        {"op": "plane", "size_x": 0.2},
        {"op": "translate", "on": {"by": "all"}, "by": [0.5, 0.0, 0.0]},
        {"op": "screw", "axis": "y", "steps": 16, "turns": 2, "height": 0.4, "angle": 360},
    ],
    "cube_open_top": [
        {"op": "cube", "size": 1.0},
        {"op": "delete", "on": {"by": "normal", "axis": "z", "sign": 1.0}},   # an open box
    ],
    "delete_then_refill": [
        {"op": "cube", "size": 1.0},
        {"op": "delete", "on": {"by": "normal", "axis": "z", "sign": 1.0}},
        {"op": "fill"},                                                       # re-cap -> closed again
    ],
    "open_box_from_bridge": [
        {"op": "cube", "size": 1.0},
        {"op": "delete", "on": {"not": {"or": [{"by": "normal", "axis": "z", "sign": 1.0},
                                               {"by": "normal", "axis": "z", "sign": -1.0}]}}},
        {"op": "bridge", "on": {"by": "all"}},   # tunnel the 2 disjoint quads -> tube
        {"op": "fill"},                          # cap -> closed box
    ],
    # place (scene composition): disjoint-union a sub-object at a transform, so the
    # op-log is natively multi-object. Both engines must merge it identically.
    "place_starts_model": [   # place can be the FIRST op (it starts the mesh)
        {"op": "place", "program": [{"op": "cube", "size": 1.0}], "translate": [2, 0, 0]},
    ],
    "place_two_cubes": [      # two disjoint cubes -> verts/edges/faces sum, euler 4
        {"op": "place", "program": [{"op": "cube", "size": 1.0}]},
        {"op": "place", "program": [{"op": "cube", "size": 1.0}], "translate": [2, 0, 0]},
    ],
    "place_rotated_scaled": [
        {"op": "cube", "size": 1.0},
        {"op": "place", "program": [{"op": "cylinder", "sides": 12, "radius": 0.3, "height": 1.0}],
         "translate": [2, 0, 0], "rotate": [0, 0, 45], "scale": [1, 1, 2]},
    ],
    "place_inline_geometry": [   # inline verts/faces (a tetra) placed at a transform
        {"op": "place", "verts": [[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]],
         "faces": [[0, 2, 1], [0, 1, 3], [0, 3, 2], [1, 2, 3]], "translate": [0, 0, 0.5]},
    ],
    "place_lathe_and_solid": [   # a single-walled lathe object + a solid, unioned
        {"op": "place", "program": [
            {"op": "profile", "points": [[0.1, -0.4], [0.3, 0.0], [0.1, 0.4]], "plane": "xz"},
            {"op": "spin", "axis": "z", "steps": 16, "angle": 360}]},
        {"op": "place", "program": [{"op": "cube", "size": 0.4}], "translate": [1.2, 0, 0]},
    ],
    "place_then_edit": [         # place, then operate on it via last_created
        {"op": "place", "program": [{"op": "cube", "size": 1.0}]},
        {"op": "inset", "on": {"by": "normal", "axis": "z", "sign": 1.0}, "thickness": 0.3},
        {"op": "extrude", "on": {"by": "last_created"}, "distance": 0.4},
    ],
    "place_nested": [            # a placed object whose program itself places another
        {"op": "place", "program": [
            {"op": "cube", "size": 1.0},
            {"op": "place", "program": [{"op": "cube", "size": 0.5}], "translate": [1.5, 0, 0]}],
         "translate": [0, 0, 0.5]},
    ],
    # --- parametric op-log: params + expressions + repeat (resolved before build) ----
    "param_expr_primitive": [    # numeric fields as expressions of params
        {"op": "params", "set": {"w": 0.5, "n": 12}},
        {"op": "cylinder", "sides": "n", "radius": "w*0.8", "height": "w*2 + 0.1"},
    ],
    "param_vase_lathe": [        # a profile whose points are expressions, spun on the lathe
        {"op": "params", "set": {"belly": 0.3, "height": 1.0, "neck": 0.18}},
        {"op": "profile", "points": [["belly*0.6", 0.0], ["belly", "height*0.4"],
                                     ["neck", "height"]], "plane": "xz"},
        {"op": "spin", "axis": "z", "steps": "floor(48/2)", "angle": 360.0},
    ],
    "param_repeat_tower": [      # a repeat loop placing cubes by expressions of the index i
        {"op": "params", "set": {"floors": 7, "twist": 12.0}},
        {"op": "repeat", "count": "floors", "index": "i", "body": [
            {"op": "place", "program": [{"op": "cube", "size": 1.0}],
             "translate": [0, 0, "i*0.5"], "rotate": [0, 0, "twist*i"],
             "scale": ["0.95^i", "0.95^i", 0.3]}]},
    ],
    "param_repeat_nested": [     # nested repeat (floors x columns), trig placement
        {"op": "params", "set": {"floors": 3, "cols": 5, "r": 0.8}},
        {"op": "repeat", "count": "floors", "index": "i", "body": [
            {"op": "repeat", "count": "cols", "index": "j", "body": [
                {"op": "place", "program": [{"op": "cube", "size": 0.2}],
                 "translate": ["r*cos(tau*j/cols)", "r*sin(tau*j/cols)", "i*0.4"]}]}]},
    ],
}


@pytest.mark.parametrize("name", list(OPLOGS))
def test_oplog_replays_identically_in_both_engines(name):
    ops = OPLOGS[name]
    ops_json = json.dumps(ops)

    py_mesh = MeshProgram(ops).build()
    cpp_mesh = cpp.replay_json(ops_json)
    cpp_mesh.validate()

    assert _s(cpp_mesh.stats()) == _s(py_mesh.stats()), f"op-log '{name}' diverged"


def test_program_json_roundtrip_native():
    ops = OPLOGS["boss_normal_last"]
    p = cpp.Program.from_json(json.dumps(ops))
    # to_json must reparse to the same op list (order-insensitive per-op).
    reparsed = json.loads(p.to_json(0))
    assert [o["op"] for o in reparsed] == [o["op"] for o in ops]
    # and rebuilding from that JSON yields the same mesh as the python oracle
    assert _s(p.build().stats()) == _s(MeshProgram(ops).build().stats())


# ----- the native selection-as-query engine, checked against the Python one ---
SELECTORS = [
    {"by": "all"},
    {"by": "normal", "axis": "z", "sign": 1.0},
    {"by": "normal", "axis": "x", "sign": -1.0},
    {"by": "normal", "dir": [0, 0, 1], "tol": 0.4},
    {"by": "extreme", "axis": "z", "which": "max"},
    {"by": "extreme", "axis": "z", "which": "min"},
    {"by": "side", "axis": "x", "sign": 1.0},
    {"by": "side", "axis": "y", "sign": -1.0},
    {"or": [{"by": "normal", "axis": "z", "sign": 1.0}, {"by": "normal", "axis": "z", "sign": -1.0}]},
    {"and": [{"by": "all"}, {"not": {"by": "normal", "axis": "z", "sign": 1.0}}]},
    {"by": "box", "min": [0.0, -1.0, -1.0], "max": [1.0, 1.0, 1.0]},   # half-space x>=0
    {"by": "box", "min": [-2.0, -2.0, 0.0], "max": [2.0, 2.0, 2.0]},   # upper slab
    {"by": "area", "which": "largest"},
    {"by": "area", "which": "smallest"},
    {"by": "area", "min": 0.5, "max": 2.0},
    {"by": "curvature", "min": 45.0},                                  # every cube face (dihedral 90)
]


@pytest.mark.parametrize("selector", SELECTORS)
def test_selector_count_matches_python(selector):
    py_n = len(resolve(make_cube(1.0), selector))
    cpp_n = cpp.selector_count(cpp.make_cube(1.0), json.dumps(selector))
    assert cpp_n == py_n, f"selector {selector} matched {cpp_n} faces (C++) vs {py_n} (Python)"


# material + connected need a richer mesh than a bare cube (a materialed array).
# materials are applied AFTER array (a final-mesh assignment doesn't survive a rebuild).
RICH_OPLOG = [
    {"op": "cube", "size": 1.0},
    {"op": "array", "count": 3, "offset": [1.5, 0.0, 0.0]},
    {"op": "material", "on": {"by": "normal", "axis": "z", "sign": 1.0}, "color": [1.0, 0.0, 0.0]},
    {"op": "material", "on": {"by": "normal", "axis": "z", "sign": -1.0}, "color": [0.0, 1.0, 0.0]},
]
RICH_SELECTORS = [
    {"by": "material"},
    {"by": "material", "color": [1.0, 0.0, 0.0]},
    {"by": "material", "color": [0.0, 1.0, 0.0]},
    {"by": "connected", "which": "largest"},
    {"by": "connected", "which": "smallest"},
    {"by": "connected", "seed": {"by": "extreme", "axis": "x", "which": "max"}},
    {"and": [{"by": "connected", "seed": {"by": "extreme", "axis": "x", "which": "max"}}, {"by": "material"}]},
    {"by": "box", "min": [-1.0, -2.0, -2.0], "max": [0.6, 2.0, 2.0]},   # spatial: the first cube
    {"by": "area", "which": "largest"},                                  # one face out of 18
    {"by": "area", "min": 0.5, "max": 2.0},                              # all 18 (unit cubes)
    {"and": [{"by": "box", "min": [-1.0, -2.0, -2.0], "max": [0.6, 2.0, 2.0]}, {"by": "material"}]},
]


@pytest.mark.parametrize("selector", RICH_SELECTORS)
def test_material_connected_selectors_match_python(selector):
    py_mesh = MeshProgram(RICH_OPLOG).build()
    cpp_mesh = cpp.replay_json(json.dumps(RICH_OPLOG))
    py_n = len(resolve(py_mesh, selector))
    cpp_n = cpp.selector_count(cpp_mesh, json.dumps(selector))
    assert cpp_n == py_n, f"selector {selector} matched {cpp_n} (C++) vs {py_n} (Python)"


from mirage.meshlang import resolve_edges  # noqa: E402

EDGE_SELECTORS = [
    {"by": "all"},
    {"by": "sharp", "angle": 30},
    {"by": "sharp", "angle": 89},
    {"by": "axis", "axis": "z"},
    {"by": "axis", "axis": "x", "tol": 0.2},
    {"by": "on_face", "face": {"by": "normal", "axis": "z", "sign": 1.0}},
    {"or": [{"by": "axis", "axis": "x"}, {"by": "axis", "axis": "y"}]},
    {"and": [{"by": "all"}, {"not": {"by": "axis", "axis": "z"}}]},
]


@pytest.mark.parametrize("esel", EDGE_SELECTORS)
def test_edge_selector_count_matches_python(esel):
    py_n = len(resolve_edges(make_cube(1.0), esel))
    cpp_n = cpp.edge_selector_count(cpp.make_cube(1.0), json.dumps(esel))
    assert cpp_n == py_n, f"edge selector {esel} matched {cpp_n} (C++) vs {py_n} (Python)"


def test_near_selector_is_native_only():
    # The "near" selector records a GUI click; it has no Python twin, but the
    # native engine resolves it to exactly one face (closest centroid).
    n = cpp.selector_count(cpp.make_cube(1.0), json.dumps({"by": "near", "point": [0.5, 0.0, 0.0]}))
    assert n == 1


def test_empty_selector_raises():
    # A selector that matches nothing must raise (localised), not silently no-op.
    with pytest.raises(Exception):
        cpp.selector_count(cpp.make_cube(1.0), json.dumps({"by": "tag", "name": "nonexistent"}))


def test_op_before_primitive_raises():
    with pytest.raises(Exception):
        cpp.replay_json(json.dumps([{"op": "inset", "on": {"by": "all"}, "thickness": 0.3}]))


# ----- per-face material (the `material` op) checked against the Python engine ---
MATERIAL_OPLOGS = {
    "cube_top_gold": [
        {"op": "cube", "size": 1.0},
        {"op": "material", "on": {"by": "normal", "axis": "z", "sign": 1.0},
         "color": [1.0, 0.78, 0.34], "metallic": 1.0, "roughness": 0.2},
    ],
    "boss_two_materials": [
        {"op": "cube", "size": 1.2},
        {"op": "inset", "on": {"by": "normal", "axis": "z"}, "thickness": 0.3},
        {"op": "extrude", "on": {"by": "last_created"}, "distance": 0.5},
        {"op": "material", "on": {"by": "last_created"}, "color": [0.85, 0.1, 0.1], "metallic": 0.0, "roughness": 0.4},
        {"op": "material", "on": {"by": "normal", "axis": "z", "sign": -1.0}, "color": [0.1, 0.2, 0.9]},
    ],
    # inline per-face materials on a mesh op (the glTF-import lowering) must match too
    "mesh_inline_materials": [
        {"op": "mesh",
         "verts": [[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0], [0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1]],
         "faces": [[0, 1, 2, 3], [4, 5, 6, 7]],
         "face_materials": [{"color": [1.0, 0.2, 0.1], "metallic": 0.0, "roughness": 0.6},
                            {"color": [0.2, 0.9, 0.3], "metallic": 1.0, "roughness": 0.15}]},
    ],
    # place with a per-object material: the FIRST object's colour must survive the
    # SECOND place (the merge preserves already-placed materials), in both engines.
    "place_two_colored": [
        {"op": "place", "program": [{"op": "cube", "size": 1.0}], "material": {"color": [1.0, 0.2, 0.1]}},
        {"op": "place", "program": [{"op": "cube", "size": 1.0}], "translate": [2, 0, 0],
         "material": {"color": [0.1, 0.3, 0.9], "metallic": 1.0, "roughness": 0.2}},
    ],
}


def _py_face_materials(ops):
    out = []
    for f in MeshProgram(ops).build().faces:
        mat = f.attrs.get("material")
        if mat:
            out.append((mat["color"][0], mat["color"][1], mat["color"][2],
                        mat["metallic"], mat["roughness"], True))
        else:
            out.append((0.8, 0.8, 0.8, 0.0, 0.5, False))  # the C++ default Material
    return out


@pytest.mark.parametrize("name", list(MATERIAL_OPLOGS))
def test_material_matches_python(name):
    ops = MATERIAL_OPLOGS[name]
    py = _py_face_materials(ops)
    cp = [tuple(t) for t in cpp.replay_json(json.dumps(ops)).face_materials()]
    assert cp == py, f"material '{name}' diverged: C++ {cp} vs Python {py}"


# ----- the native lint pass, checked against the Python one -------------------
from mirage.repair import lint_program  # noqa: E402

LINT_CASES = [
    # each builds cleanly but trips a silent-trap warning
    [{"op": "cube"}, {"op": "extrude", "on": {"by": "normal", "axis": "z"}, "distance": 0.0}],     # extrude_noop
    [{"op": "cube"}, {"op": "extrude", "on": {"by": "all"}, "distance": 0.5}],                     # extrude_all
    [{"op": "cube"}, {"op": "inset", "on": {"by": "normal", "axis": "z"}, "thickness": 1.5}],      # inset_clamped
    [{"op": "cube"}, {"op": "bevel", "on": {"by": "normal", "axis": "z"}, "width": 2.0, "depth": 0.1}],  # bevel_width_clamped
    [{"op": "cube"}, {"op": "bevel", "on": {"by": "normal", "axis": "z"}, "width": 0.2, "depth": 0.0}],  # bevel_flat
    [{"op": "cube"}, {"op": "subdivide", "levels": 0}],                                            # subdivide_noop
    [{"op": "cube"}, {"op": "subdivide", "levels": 9}],                                            # subdivide_explosive
    [{"op": "cube"}, {"op": "inset", "on": {"by": "extreme", "axis": "z", "which": "highest"}, "thickness": 0.3}],  # extreme_which
    [{"op": "cube"}, {"op": "inset", "on": {"by": "last_created"}, "thickness": 0.3}],             # last_created_broad
    [{"op": "cube"}, {"op": "extrude", "on": {"by": "side", "axis": "x", "sign": 0}, "distance": 0.5}],  # sign_zero
    [{"op": "cube"},  # nested trap inside and/not — must be caught at depth
     {"op": "inset", "on": {"and": [{"by": "all"}, {"not": {"by": "extreme", "axis": "z", "which": "top"}}]}, "thickness": 0.3}],
    [{"op": "cube"}, {"op": "extrude", "on": {"by": "normal", "axis": "z"}, "distance": 0.5}],     # clean: no warnings
]


@pytest.mark.parametrize("ops", LINT_CASES)
def test_lint_matches_python(ops):
    py = sorted((w["op_index"], w["code"]) for w in lint_program(ops))
    cp = sorted((w["op_index"], w["code"]) for w in cpp.lint_json(json.dumps(ops)))
    assert cp == py, f"lint diverged for {ops}: C++ {cp} vs Python {py}"
