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
from mirage.kernel import make_cube  # noqa: E402


def _s(stats):
    return (stats["verts"], stats["edges"], stats["faces"], stats["euler"], stats["closed_manifold"])


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
]


@pytest.mark.parametrize("selector", SELECTORS)
def test_selector_count_matches_python(selector):
    py_n = len(resolve(make_cube(1.0), selector))
    cpp_n = cpp.selector_count(cpp.make_cube(1.0), json.dumps(selector))
    assert cpp_n == py_n, f"selector {selector} matched {cpp_n} faces (C++) vs {py_n} (Python)"


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
