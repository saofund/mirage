"""Tests for repair.py — grounded in the empirical failure-mode sweep.

Each test names the failure class it pins: AUTO repairs (high-confidence,
intent-preserving) must be applied; SUGGEST repairs (intent-changing) must be
offered but never silently applied; lint catches the silent (no-exception)
class; repair_mesh cleans geometry defects.
"""
import pytest

from mirage.kernel import Mesh, make_cube
from mirage.meshlang import MeshProgram
from mirage.repair import (
    diagnose, repair_program, lint_program, repair_mesh, Diagnostic,
)


def _ops(*ops):
    return list(ops)


# --------------------------------------------------------------------------- #
# Classification + idempotence
# --------------------------------------------------------------------------- #
def test_valid_program_is_noop():
    r = repair_program(_ops({"op": "cube"}, {"op": "extrude", "on": {"by": "extreme", "axis": "z"}, "distance": 0.3}))
    assert r.ok and not r.repaired
    assert diagnose(_ops({"op": "cube"})) is None


def test_classify_selector_empty_has_diagnostics():
    d = diagnose(_ops({"op": "cube"}, {"op": "extrude", "on": {"by": "tag", "name": "nope"}, "distance": 0.3}))
    assert d.kind == "selector_empty"
    assert "normal_histogram" in d.selector_diagnostics


# --------------------------------------------------------------------------- #
# AUTO repairs — applied silently because intent is preserved
# --------------------------------------------------------------------------- #
def test_auto_fuzzy_tag_typo():
    ops = _ops({"op": "cube"},
               {"op": "tag", "on": {"by": "normal", "axis": "z", "sign": 1}, "name": "top"},
               {"op": "extrude", "on": {"by": "tag", "name": "tpo"}, "distance": 0.3})
    r = repair_program(ops)
    assert r.ok and r.repaired
    assert r.program[2]["on"] == {"by": "tag", "name": "top"}
    MeshProgram(r.program).build()   # the repaired program builds


def test_auto_normal_tol_relax():
    ops = _ops({"op": "cube"},
               {"op": "extrude", "on": {"by": "normal", "axis": "z", "sign": 1, "tol": 1.5}, "distance": 0.3})
    r = repair_program(ops)
    assert r.ok and r.repaired
    assert r.program[1]["on"]["tol"] < 1.0 and r.program[1]["on"]["axis"] == "z"


def test_auto_scalar_scale_broadcast():
    ops = _ops({"op": "cube"}, {"op": "scale", "on": {"by": "all"}, "by": 2})
    r = repair_program(ops)
    assert r.ok and r.repaired
    assert r.program[1]["by"] == [2, 2, 2]


def test_auto_numeric_string_coerce():
    ops = _ops({"op": "cube"}, {"op": "extrude", "on": {"by": "all"}, "distance": "0.5"})
    r = repair_program(ops)
    assert r.ok and r.repaired
    assert r.program[1]["distance"] == 0.5


def test_auto_op_name_typo_with_param_signature():
    ops = _ops({"op": "cube"}, {"op": "extrod", "on": {"by": "all"}, "distance": 0.5})
    r = repair_program(ops)
    assert r.ok and r.repaired
    assert r.program[1]["op"] == "extrude"


def test_auto_selector_by_typo():
    ops = _ops({"op": "cube"}, {"op": "extrude", "on": {"by": "normalz", "axis": "z", "sign": 1}, "distance": 0.3})
    r = repair_program(ops)
    assert r.ok and r.repaired
    assert r.program[1]["on"]["by"] == "normal"


def test_auto_axis_casing_normalize():
    ops = _ops({"op": "cube"}, {"op": "extrude", "on": {"by": "extreme", "axis": "Z", "which": "max"}, "distance": 0.2})
    r = repair_program(ops)
    assert r.ok and r.repaired
    assert r.program[1]["on"]["axis"] == "z"


def test_auto_signed_axis_no_explicit_sign():
    # '-x' with no explicit sign: deriving sign=-1 is intent-preserving -> auto
    ops = _ops({"op": "cube"}, {"op": "extrude", "on": {"by": "normal", "axis": "-x"}, "distance": 0.2})
    r = repair_program(ops)
    assert r.ok and r.repaired
    assert r.program[1]["on"]["axis"] == "x" and r.program[1]["on"]["sign"] == -1.0


def test_auto_int_axis_recovers():
    # regression: int axis (2) raised TypeError 'must be str' and yielded no candidate
    ops = _ops({"op": "cube"}, {"op": "extrude", "on": {"by": "normal", "axis": 2}, "distance": 0.2})
    r = repair_program(ops)
    assert r.ok and r.repaired
    assert r.program[1]["on"]["axis"] == "z"


# --------------------------------------------------------------------------- #
# SUGGEST repairs — never applied silently (the safety invariant)
# --------------------------------------------------------------------------- #
def test_suggest_tag_absent_no_auto():
    ops = _ops({"op": "cube"}, {"op": "extrude", "on": {"by": "tag", "name": "rim"}, "distance": 0.3})
    r = repair_program(ops)
    assert not r.ok and not r.repaired           # no intent-preserving fix exists
    assert r.suggestions                          # but options are offered
    assert all(s["apply_mode"] == "suggest" for s in r.suggestions)


def test_suggest_scale_to_zero_invalid_mesh():
    ops = _ops({"op": "cube"}, {"op": "scale", "on": {"by": "all"}, "by": [0, 0, 0]})
    r = repair_program(ops)
    assert not r.repaired                          # magnitude is a guess -> never auto
    assert any(s["validated"] for s in r.suggestions)   # at least one offered fix builds


def test_assert_failure_never_auto():
    ops = _ops({"op": "cube"}, {"op": "assert", "euler": 10})
    r = repair_program(ops)
    assert not r.repaired
    assert r.diagnostic["kind"] == "assert_failed"
    assert any("euler" in str(s["op"]) for s in r.suggestions if s["op"])


def test_structural_op_before_primitive_suggests_insert():
    ops = _ops({"op": "extrude", "on": {"by": "all"}, "distance": 0.5})
    r = repair_program(ops)
    assert not r.repaired
    assert any(s["action"] == "insert_before" for s in r.suggestions)


def test_and_contradiction_suggests_or():
    ops = _ops({"op": "cube"},
               {"op": "extrude", "on": {"and": [{"by": "normal", "axis": "z", "sign": 1},
                                                {"by": "normal", "axis": "z", "sign": -1}]}, "distance": 0.3})
    r = repair_program(ops)
    assert not r.repaired
    assert any(s["op"] and "or" in s["op"].get("on", {}) for s in r.suggestions)


def test_signed_axis_conflicting_sign_never_auto():
    # red-team: '-x' with explicit sign:1.0 is contradictory -> must NOT silently flip the face
    ops = _ops({"op": "cube"}, {"op": "extrude", "on": {"by": "normal", "axis": "-x", "sign": 1.0, "tol": 0.3}, "distance": 0.2})
    r = repair_program(ops)
    assert not r.repaired
    labels = [s["label"] for s in r.suggestions]
    assert any("keep sign" in s for s in labels) and any("sign -1" in s for s in labels)


def test_translate_scalar_is_suggest_not_auto():
    # red-team: scalar broadcast is unambiguous for SCALE only; translate invents a direction
    ops = _ops({"op": "cube"}, {"op": "translate", "on": {"by": "all"}, "by": 0.5})
    r = repair_program(ops)
    assert not r.repaired
    assert any(s["op"] and s["op"].get("by") == [0.5, 0.5, 0.5] for s in r.suggestions)


def test_cylinder_sides_is_suggest_not_auto():
    # red-team: clamping sides<3 to 3 is a magnitude pick, not a tolerance relax
    r = repair_program(_ops({"op": "cylinder", "sides": 2, "radius": 0.5, "height": 1}))
    assert not r.repaired
    assert any("sides 2 ->" in s["label"] for s in r.suggestions)


# --------------------------------------------------------------------------- #
# Termination / bounded
# --------------------------------------------------------------------------- #
def test_bounded_attempts():
    ops = _ops({"op": "cube"}, {"op": "extrude", "on": {"by": "tag", "name": "nope"}, "distance": 0.3})
    r = repair_program(ops, max_attempts=3)
    assert len(r.attempts) <= 3


def test_not_all_is_terminal():
    ops = _ops({"op": "cube"}, {"op": "extrude", "on": {"not": {"by": "all"}}, "distance": 0.3})
    r = repair_program(ops)        # structurally empty for any mesh — must terminate, not loop
    assert not r.repaired


def test_subdivide_cost_guard_no_hang():
    # red-team: a runaway subdivide must be refused statically, never built (~4^levels)
    import time
    t = time.perf_counter()
    r = repair_program(_ops({"op": "cube"}, {"op": "subdivide", "levels": 50}))
    assert time.perf_counter() - t < 1.0          # never builds the oversized mesh
    assert not r.repaired and r.diagnostic["kind"] == "cost"
    # and a broken op AFTER a big subdivide is also caught cheaply
    t = time.perf_counter()
    repair_program(_ops({"op": "cube"}, {"op": "subdivide", "levels": 9},
                        {"op": "extrude", "on": {"by": "tag", "name": "NOPE"}, "distance": 0.4}))
    assert time.perf_counter() - t < 1.0


def test_non_dict_op_no_crash():
    # red-team: a non-dict op must not crash either entry point
    d = diagnose(_ops({"op": "cube"}, "garbage"))
    assert d is not None and d.kind == "malformed_op"
    r = repair_program(_ops({"op": "cube"}, "garbage"))   # must return, not raise
    assert not r.ok and not r.repaired


# --------------------------------------------------------------------------- #
# Lint — the silent (no-exception) class
# --------------------------------------------------------------------------- #
def test_lint_zero_distance_extrude():
    w = lint_program(_ops({"op": "cube"}, {"op": "extrude", "on": {"by": "all"}, "distance": 0, "mark": "x"}))
    assert any(x["code"] == "extrude_noop" for x in w)


def test_lint_inset_clamp():
    w = lint_program(_ops({"op": "cube"}, {"op": "inset", "on": {"by": "extreme", "axis": "z"}, "thickness": 5}))
    assert any(x["code"] == "inset_clamped" for x in w)


def test_lint_extreme_which_silent_min():
    w = lint_program(_ops({"op": "cube"}, {"op": "extrude", "on": {"by": "extreme", "axis": "z", "which": "highest"}, "distance": 0.2}))
    assert any(x["code"] == "extreme_which" for x in w)


def test_lint_last_created_after_primitive():
    w = lint_program(_ops({"op": "cube"}, {"op": "extrude", "on": {"by": "last_created"}, "distance": 0.2}))
    assert any(x["code"] == "last_created_broad" for x in w)


def test_lint_clean_program_has_no_warnings():
    ops = _ops({"op": "cube"}, {"op": "extrude", "on": {"by": "extreme", "axis": "z", "which": "max"}, "distance": 0.3})
    assert lint_program(ops) == []


def test_lint_inset_boundary_not_flagged():
    # red-team: 1e-3 and 0.999 are the inclusive clamp endpoints — kernel honors them verbatim
    for t in (1e-3, 0.999, 0.5):
        ops = _ops({"op": "cube"}, {"op": "inset", "on": {"by": "all"}, "thickness": t})
        assert not any(w["code"] == "inset_clamped" for w in lint_program(ops)), t
    ops = _ops({"op": "cube"}, {"op": "inset", "on": {"by": "all"}, "thickness": 1.5})
    assert any(w["code"] == "inset_clamped" for w in lint_program(ops))


def test_lint_recurses_into_nested_selectors():
    # red-team: a trap buried inside and/or/not must still be flagged
    ops = _ops({"op": "cube"},
               {"op": "tag", "on": {"and": [{"by": "extreme", "axis": "z", "which": "top"}, {"by": "all"}]}, "name": "lid"})
    assert any(w["code"] == "extreme_which" for w in lint_program(ops))


# --------------------------------------------------------------------------- #
# repair_mesh — geometry-level cleanup
# --------------------------------------------------------------------------- #
def test_repair_mesh_drops_zero_area_face():
    m = Mesh.from_pydata([(0, 0, 0), (1, 0, 0), (2, 0, 0)], [[0, 1, 2]])   # collinear -> zero area
    cleaned, report = repair_mesh(m)
    assert report["dropped_degenerate"] == 1
    assert report["faces_after"] == 0
    cleaned.validate()


def test_repair_mesh_drops_orphan_vert():
    m = make_cube(1.0)
    m.add_vert((9, 9, 9))                       # loose vertex
    with pytest.raises(AssertionError):
        m.validate()
    cleaned, report = repair_mesh(m)
    assert report["dropped_orphans"] == 1
    cleaned.validate()
    assert cleaned.is_closed_manifold()


def test_repair_mesh_welds_seam():
    # two triangles sharing a diagonal, but duplicated coincident verts split the seam
    m = Mesh.from_pydata([(0, 0, 0), (1, 0, 0), (0, 1, 0), (1, 0, 0), (0, 1, 0), (1, 1, 0)],
                         [[0, 1, 2], [3, 4, 5]])
    assert not m.is_closed_manifold()
    cleaned, report = repair_mesh(m)
    assert report["welded_verts"] == 2
    assert len(cleaned.verts) == 4
    cleaned.validate()


def test_repair_mesh_detects_nonmanifold_without_deleting():
    # edge (0,1) shared by 3 triangles — validate() passes, but it's non-manifold
    m = Mesh.from_pydata([(0, 0, 0), (1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1)],
                         [[0, 1, 2], [0, 1, 3], [0, 1, 4]])
    cleaned, report = repair_mesh(m)
    assert report["nonmanifold_edges"] == 1
    assert report["apply_mode"] == "suggest"      # ambiguous -> not auto
    assert report["faces_after"] == 3             # nothing deleted


def test_repair_mesh_pentagon_weld_no_crash():
    # red-team: a weld making NON-consecutive verts of an n-gon coincide must not crash
    # from_pydata ('face has repeated vertices'); the self-touching face is dropped.
    pos = [[0, 0, 0], [1, 0, 0], [0.001, 0.001, 0], [0, 1, 0], [-1, 1, 0]]
    m = Mesh.from_pydata(pos, [[0, 1, 2, 3, 4]])
    cleaned, report = repair_mesh(m, eps=0.01)    # v0 and v2 weld -> cycle self-touches
    assert report["dropped_degenerate"] == 1
    cleaned.validate()


def test_repair_mesh_large_eps_collapse_is_suggest():
    # red-team: an eps that welds away a REAL face must downgrade to 'suggest', not stay 'auto'
    pos = [[0, 0, 0], [0.05, 0, 0], [0, 1, 0], [2, 0, 0], [3, 0, 0], [2, 1, 0]]
    m = Mesh.from_pydata(pos, [[0, 1, 2], [3, 4, 5]])
    _c, report = repair_mesh(m, eps=0.1)          # the 0.05 edge is shorter than eps
    assert report["weld_collapsed"] == 1
    assert report["apply_mode"] == "suggest"


def test_repair_mesh_duplicate_faces_is_suggest():
    # red-team: a weld merging two distinct faces into coincident duplicates -> suggest, deduped
    pos = [[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0],
           [0, 0, 0.2], [1, 0, 0.2], [1, 1, 0.2], [0, 1, 0.2]]
    m = Mesh.from_pydata(pos, [[0, 1, 2, 3], [4, 5, 6, 7]])
    cleaned, report = repair_mesh(m, eps=0.5)     # the 0.2 z-gap is welded away
    assert report["duplicate_faces"] == 1
    assert report["apply_mode"] == "suggest"
    assert report["faces_after"] == 1
    cleaned.validate()


def test_repair_mesh_nan_eps_no_crash():
    # red-team: a non-finite eps (MCP-controllable) must not crash round()
    cleaned, report = repair_mesh(make_cube(1.0), eps=float("nan"))
    assert report["welded_verts"] == 0
    cleaned.validate()
