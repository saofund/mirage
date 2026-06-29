import pytest

from mirage.meshlang import MeshProgram, Sel, SelectorEmpty, resolve
from mirage.kernel import make_cube


def test_build_cube_program():
    m = MeshProgram().cube(1.0).build()
    m.validate()
    assert m.stats()["faces"] == 6 and m.is_closed_manifold()


def test_inset_extrude_via_tags():
    m = (MeshProgram().cube(1.0)
         .inset(on=Sel.normal("z", 1), thickness=0.3, mark="ring")
         .extrude(on=Sel.tag("ring"), distance=0.5, mark="boss")
         .assert_(closed_manifold=True, euler=2)).build()
    m.validate()
    assert m.euler() == 2 and m.is_closed_manifold()


def test_last_created_chaining():
    p = MeshProgram().cube(1.6)
    p.inset(on=Sel.normal("z", 1), thickness=0.26)
    p.extrude(on=Sel.last(), distance=0.45)        # last_created = the inset inner face
    for _ in range(3):
        p.inset(on=Sel.last(), thickness=0.26)
        p.extrude(on=Sel.last(), distance=0.45)
    m = p.build()
    m.validate()
    assert m.euler() == 2 and m.is_closed_manifold()


def test_json_roundtrip_is_deterministic():
    p = (MeshProgram().cube(1.0)
         .inset(on=Sel.normal("z", 1), thickness=0.3, mark="r")
         .extrude(on=Sel.tag("r"), distance=0.4))
    assert MeshProgram.from_json(p.to_json()).build().stats() == p.build().stats()


def test_selector_empty_raises_with_diagnostics():
    p = MeshProgram().cube(1.0).extrude(on=Sel.tag("does_not_exist"))
    with pytest.raises(SelectorEmpty) as exc:
        p.build()
    assert "bbox" in exc.value.diagnostics


def test_parametric_edit_changes_geometry():
    base = MeshProgram().cylinder(sides=24, radius=0.4, height=0.9)
    edited = MeshProgram.from_json(base.to_json())
    edited.ops[0]["sides"] = 8
    assert edited.build().stats()["faces"] != base.build().stats()["faces"]
    assert edited.build().is_closed_manifold()


def test_tags_survive_subdivision():
    m = (MeshProgram().cube(1.0)
         .tag(on=Sel.normal("z", 1), name="lid")
         .subdivide(levels=1)).build()
    from mirage.meshlang import _tags
    assert any("lid" in _tags(f) for f in m.faces)   # tag propagated to child quads


def test_selector_extreme_picks_top_face():
    assert len(resolve(make_cube(1.0), Sel.extreme("z", "max"))) == 1


def test_new_primitives_are_valid():
    sphere = MeshProgram().uv_sphere(12, 8, 0.6).build()
    sphere.validate()
    assert sphere.is_closed_manifold() and sphere.euler() == 2

    cone = MeshProgram().cone(16, 0.5, 1.0).build()
    cone.validate()
    assert cone.is_closed_manifold() and cone.euler() == 2

    torus = MeshProgram().torus(16, 10, 0.6, 0.22).build()
    torus.validate()
    assert torus.is_closed_manifold() and torus.euler() == 0   # genus-1

    grid = MeshProgram().grid(2.0, 1.0, 6, 4).build()
    grid.validate()
    assert not grid.is_closed_manifold() and grid.stats()["faces"] == 24   # open, 6x4 quads


def test_uv_sphere_min_rings_is_a_bipyramid():
    # rings=2 -> two triangle fans, no quad bands; verts = 2 poles + 1 ring
    m = MeshProgram().uv_sphere(6, 2, 0.5).build()
    m.validate()
    assert m.stats()["verts"] == 8 and m.stats()["faces"] == 12 and m.is_closed_manifold()


def test_primitive_parametric_edit():
    base = MeshProgram().uv_sphere(8, 6, 0.5)
    edited = MeshProgram.from_json(base.to_json())
    edited.ops[0]["segments"] = 16
    assert edited.build().stats()["faces"] != base.build().stats()["faces"]
    assert edited.build().is_closed_manifold()


def test_solidify_turns_open_surface_into_solid():
    # an open plane (1 quad) shelled -> a watertight box (2 caps + 4 walls)
    m = MeshProgram().plane(1.0).solidify(0.2).build()
    m.validate()
    assert m.is_closed_manifold() and m.euler() == 2 and m.stats()["faces"] == 6


def test_mirror_welds_the_seam():
    # a plane pushed so a left edge sits on x=0, mirrored: the 2 seam verts are shared
    m = (MeshProgram().plane(1.0)
         .translate(Sel.all(), [0.5, 0.0, 0.0])
         .mirror("x")).build()
    m.validate()
    assert m.stats()["verts"] == 6 and m.stats()["faces"] == 2     # 4+4 - 2 welded


def test_array_makes_disjoint_copies():
    m = MeshProgram().cube(1.0).array(4, (1.3, 0.0, 0.0)).build()
    m.validate()
    s = m.stats()
    assert s["verts"] == 32 and s["faces"] == 24 and s["euler"] == 8   # 4 closed cubes


def test_material_selector_picks_materialed_faces():
    m = (MeshProgram().cube(1.0)
         .material(Sel.normal("z", 1), color=[1, 0, 0])
         .material(Sel.normal("z", -1), color=[0, 1, 0])).build()
    assert len(resolve(m, Sel.material())) == 2           # both materialed faces
    assert len(resolve(m, Sel.material([1, 0, 0]))) == 1  # red only


def test_connected_selector_isolates_one_component():
    m = MeshProgram().cube(1.0).array(3, (1.5, 0.0, 0.0)).build()
    assert len(resolve(m, Sel.connected("largest"))) == 6     # one cube out of three
    # the component containing the +x-most face is exactly one cube
    assert len(resolve(m, Sel.component_of(Sel.extreme("x", "max")))) == 6


def test_bisect_cuts_and_optionally_fills():
    # cut a cube at z=0: the bottom half is an open 5-face box ...
    openm = MeshProgram().cube(1.0).bisect((0, 0, 0), (0, 0, 1)).build()
    openm.validate()
    assert not openm.is_closed_manifold() and openm.stats()["faces"] == 5
    # ... and with fill it is capped back into a closed manifold
    closed = MeshProgram().cube(1.0).bisect((0, 0, 0), (0, 0, 1), fill=True).build()
    closed.validate()
    assert closed.is_closed_manifold() and closed.euler() == 2


def test_spin_revolves_a_profile_into_a_solid():
    # a profile touching the axis, revolved 360 -> a watertight surface of revolution
    profile = [[0.0, 0, -0.5], [0.5, 0, -0.3], [0.5, 0, 0.3], [0.0, 0, 0.5]]
    m = MeshProgram().mesh(profile, [[0, 1, 2, 3]]).spin("z", 16, 360).build()
    m.validate()
    assert m.is_closed_manifold() and m.euler() == 2


def test_spin_partial_angle_is_open():
    profile = [[0.4, 0, -0.5], [0.6, 0, -0.5], [0.6, 0, 0.5], [0.4, 0, 0.5]]
    m = MeshProgram().mesh(profile, [[0, 1, 2, 3]]).spin("z", 8, 120).build()
    m.validate()
    assert not m.is_closed_manifold()      # a partial sweep leaves an open sheet


def test_box_selector_picks_one_cube_of_an_array():
    m = MeshProgram().cube(1.0).array(3, (1.5, 0.0, 0.0)).build()
    # the array climbs +x from -0.5; an AABB clipping x<=0.6 catches only the first cube
    sel = resolve(m, Sel.box([-1.0, -2.0, -2.0], [0.6, 2.0, 2.0]))
    assert len(sel) == 6


def test_area_selector_finds_the_biggest_face():
    # a cube with one face inset leaves a smaller inner quad + a big untouched bottom
    m = (MeshProgram().cube(1.0)
         .inset(on=Sel.normal("z", 1), thickness=0.3)).build()
    biggest = resolve(m, Sel.area("largest"))
    smallest = resolve(m, Sel.area("smallest"))
    from mirage.kernel import face_area
    assert len(biggest) == 1 and len(smallest) == 1
    assert face_area(m, biggest[0]) >= face_area(m, smallest[0])
    assert face_area(m, smallest[0]) < 1.0          # the inset inner face is the small one


def test_curvature_selector_separates_flat_from_creased():
    # a smooth-ish UV sphere: no face's neighbourhood is a hard 90 crease ...
    sphere = MeshProgram().uv_sphere(16, 12, 0.6).build()
    assert len(resolve(sphere, Sel.curvature(0.0, 40.0))) == sphere.stats()["faces"]
    # ... whereas every face of a cube sits on 90-degree creases
    cube = make_cube(1.0)
    assert len(resolve(cube, Sel.curvature(80.0, 100.0))) == 6


def test_boolean_intersection_is_the_overlap_solid():
    # two unit cubes overlapping at a corner: their intersection is the small box
    # they share -> a clean closed manifold (the BSP boolean is exact here)
    cutter = make_cube(1.0)
    for v in cutter.verts:
        v.co = (v.co[0] + 0.5, v.co[1] + 0.5, v.co[2] + 0.5)
    m = MeshProgram().cube(1.0).boolean("intersection", cutter).build()
    m.validate()
    assert m.is_closed_manifold() and m.euler() == 2 and m.stats()["faces"] == 6
    # the overlap box spans [0, 0.5] on each axis
    lo = [min(v.co[k] for v in m.verts) for k in range(3)]
    hi = [max(v.co[k] for v in m.verts) for k in range(3)]
    assert lo == pytest.approx([0, 0, 0]) and hi == pytest.approx([0.5, 0.5, 0.5])


def test_boolean_difference_removes_material():
    # cube minus an overlapping cube -> fewer than the 8 corners of a full cube remain
    # solid; the carved notch makes it non-convex (validate still passes)
    cutter = make_cube(1.0)
    for v in cutter.verts:
        v.co = (v.co[0] + 0.5, v.co[1] + 0.5, v.co[2] + 0.5)
    full = MeshProgram().cube(1.0).build()
    carved = MeshProgram().cube(1.0).boolean("difference", cutter).build()
    carved.validate()
    assert carved.stats()["faces"] > full.stats()["faces"]   # the notch adds faces


def test_boolean_drill_makes_a_bore():
    # subtract a tall thin cylinder from a cube -> a cube with a hole punched through
    from mirage.kernel import make_cylinder_ngon
    drill = make_cylinder_ngon(16, 0.25, 2.0)
    m = MeshProgram().cube(1.0).boolean("difference", drill).build()
    m.validate()
    # the bore opens the top and bottom faces, so the result is no longer convex
    assert m.stats()["faces"] > 6


def test_profile_lathe_makes_a_single_wall():
    # an OPEN profile curve (a wire) revolved 360 -> a single-walled open vase.
    # A filled cross-section would instead give a double-walled (closed) tube; the
    # profile is the difference between the two — a real generatrix.
    pts = [[0.3, -0.5], [0.5, 0.0], [0.3, 0.5]]
    vase = MeshProgram().profile(pts, plane="xz").spin("z", 16, 360).build()
    vase.validate()
    assert not vase.is_closed_manifold()          # single wall: open top + open bottom rim
    # the same outline as a FILLED quad strip revolves into a closed double wall
    strip = [[0.3, -0.5], [0.5, 0.0], [0.3, 0.5], [0.28, 0.0]]
    tube = MeshProgram().mesh([[p[0], 0, p[1]] for p in strip], [[0, 1, 2, 3]]).spin("z", 16, 360).build()
    tube.validate()
    assert tube.is_closed_manifold()              # double wall: watertight


def test_profile_closed_loop_revolves_to_a_torus():
    # a CLOSED profile (a small rectangle loop offset from the axis) -> a torus surface
    pts = [[0.4, -0.1], [0.6, -0.1], [0.6, 0.1], [0.4, 0.1]]
    m = MeshProgram().profile(pts, plane="xz", closed=True).spin("z", 16, 360).build()
    m.validate()
    assert m.is_closed_manifold() and m.euler() == 0     # genus-1, like the torus primitive


def test_profile_is_a_wire_with_no_faces():
    from mirage.kernel import make_profile
    w = make_profile([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0]], plane="xz")
    assert w.stats()["faces"] == 0 and w.stats()["edges"] == 2 and w.stats()["verts"] == 3
    w.validate()                                          # a wire is valid (edge-referenced verts)


def test_screw_climbs_into_a_helix():
    # a small square cross-section swept 2 turns -> an open helical band (a spring)
    profile = [[0.4, 0, -0.05], [0.5, 0, -0.05], [0.5, 0, 0.05], [0.4, 0, 0.05]]
    m = MeshProgram().mesh(profile, [[0, 1, 2, 3]]).screw("z", 12, turns=2, height=0.3).build()
    m.validate()
    assert not m.is_closed_manifold()      # a screw always climbs -> never closes
    # the top of the band sits `height*turns` above the bottom along the axis
    zs = [v.co[2] for v in m.verts]
    assert max(zs) - min(zs) == pytest.approx(0.3 * 2 + 0.1)   # climb + profile thickness


def test_describe_reports_materials_and_components():
    from mirage.meshlang import describe
    m = (MeshProgram().cube(1.0).array(3, (1.5, 0.0, 0.0))
         .material(Sel.normal("z", 1), color=[1.0, 0.0, 0.0])).build()
    d = describe(m)
    assert d["components"] == [6, 6, 6]          # three separate cubes
    assert d["materials"].get("rgb(1.00,0.00,0.00)") == 3   # the painted top of each


def test_material_assigns_to_final_faces():
    m = (MeshProgram().cube(1.0)
         .material(Sel.normal("z", 1), color=[1.0, 0.0, 0.0], metallic=0.5)).build()
    mats = [f.attrs.get("material") for f in m.faces if f.attrs.get("material")]
    assert len(mats) == 1 and mats[0]["color"] == [1.0, 0.0, 0.0] and mats[0]["metallic"] == 0.5


def test_material_does_not_propagate_through_rebuilds():
    # unlike tags, material is a final-mesh assignment (matches the C++ engine):
    # a geometry op after `material` rebuilds the mesh and drops it.
    m = (MeshProgram().cube(1.0)
         .material(Sel.all(), color=[1.0, 0.0, 0.0])
         .subdivide(levels=1)).build()
    assert all(f.attrs.get("material") is None for f in m.faces)


def test_last_created_after_scale():
    # regression: scale/translate/tag set last_tag but never stamped it onto the
    # faces, so `last_created` on the NEXT op matched 0 (goblet-flare repro).
    m = (MeshProgram().cylinder(sides=8, radius=0.5, height=0.4)
         .extrude(on=Sel.extreme("z", "max"), distance=0.2)
         .scale(on=Sel.last(), by=[2, 2, 1])
         .extrude(on=Sel.last(), distance=0.2)        # was SelectorEmpty before the fix
         .assert_(closed_manifold=True, euler=2)).build()
    m.validate()
    assert m.is_closed_manifold() and m.euler() == 2
