"""Geometric invariants for case 27's part kit — the checks a render cannot make for you.

Every fidelity error this case has shipped had the same shape: the part was individually
plausible, the cloud metrics stayed green, and nobody looked at the geometry at a
magnification where it could be seen. Two of them were not subtle once stated:

* `trim_ring` was not a ring. Its lathe section began on the axis, so it swept a solid
  plate over the whole pocket, five millimetres above the cap, and every frame that drew
  one had no visible cap at all. It was in the kit for weeks.
* `pressed_dish` put its floor on the cap's own face plane, so the floor covered the cap.

Both are one arithmetic assertion away from impossible, and neither is expressible as a
statistic over a point cloud, which is where all this case's other checks live. So they go
here: cheap, exact statements about where the surfaces are.
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "examples" / "cases"))

fuelcap = pytest.importorskip("fuelcap")
from fuelcap import parts as P                                  # noqa: E402
from fuelcap import scene as S                                  # noqa: E402


def verts(prog):
    return np.array([v.co for v in prog.build().verts], float)


def radii(co):
    return np.hypot(co[:, 0], co[:, 1])


# --------------------------------------------------------------------------- #
# the cap
# --------------------------------------------------------------------------- #
def test_cap_is_a_cylinder_not_a_disc():
    d, flange = 0.078, 0.013
    co = verts(P.cap(d=d, flange=flange, printing=False))
    r = radii(co)
    # the flutes are grooves cut INTO the wall, so the lugs are the diameter and nothing
    # may exceed it — but a discretely sampled plan need not land exactly on a lug centre
    assert d / 2 * 0.997 < r.max() <= d / 2 + 1e-9
    # a visible side wall: material at full radius spread down the skirt (the last couple
    # of millimetres are the underside chamfer, which pulls back in)
    # The wall starts just under the rim bevel and runs most of the way down the skirt; the
    # last couple of millimetres are the underside chamfer, which pulls back in.
    wall = co[r > d / 2 * 0.985, 2]
    assert wall.min() <= -flange * 0.75
    assert -0.0030 <= wall.max() <= 0.0


def test_flutes_groove_the_wall_and_leave_the_face_round():
    d = 0.078
    plain = radii(verts(P.cap(d=d, flutes=0, printing=False)))
    ridged = radii(verts(P.cap(d=d, flutes=12, flute_depth=0.065, printing=False)))
    assert abs(plain.max() - ridged.max()) < d / 2 * 0.002   # lugs ARE the diameter
    # the grooves are real, and about as deep as asked
    outer = ridged[ridged > d / 2 * 0.90]
    assert (outer.max() - outer.min()) > d / 2 * 0.05


def test_the_handle_stays_inside_the_caps_own_silhouette():
    # The bar's end section leans (the path dips there to bury the open ring), so its corner
    # reaches past the last station. Left unclamped it sliced two chords off the cap.
    for d in (0.068, 0.078, 0.086):
        for rib_len in (d * 0.84, d * 0.96, d * 1.4):
            co = verts(P.cap(d=d, rib_len=rib_len, printing=False))
            assert radii(co).max() <= d / 2 + 1e-6, f"d={d} rib_len={rib_len}"


def _band(co, lo, hi):
    """Vertices whose |x| falls in a band. Slicing by an exact x does not work here: the
    section tilts wherever the path does, so a 'station' is not a plane of constant x."""
    a = np.abs(co[:, 0])
    return co[(a >= lo) & (a <= hi)]


def test_the_handle_is_waisted_and_troughed():
    co = verts(P.handle(0.070, 0.030, 0.021, 0.0066))
    ends = np.abs(_band(co, 0.028, 0.034)[:, 1]).max()
    mid = np.abs(_band(co, 0.000, 0.004)[:, 1]).max()
    assert mid < ends * 0.90, f"the middle must be pinched: {mid:.4f} vs {ends:.4f}"
    # The trough that matters is the one ACROSS the bar — the thumb groove. (Not along it:
    # the path's end dip lowers the flanks faster than the length sag lowers the middle, so
    # the highest point on the whole bar really is on its centre station, and asserting
    # otherwise was asserting an intention rather than the shape.)
    band = _band(co, 0.000, 0.006)
    spine = band[np.abs(band[:, 1]) < 0.0015, 2].max()
    shoulder = band[np.abs(band[:, 1]) > 0.0045, 2].max()
    assert spine < shoulder - 0.0005, f"spine {spine:.4f} shoulder {shoulder:.4f}"


def test_the_handle_ends_are_buried_below_the_face():
    # What makes the visible end rounded is that the open sweep ring finishes inside solid
    # material — so the very tips must sit below z = 0, or the bar ends in a hole. That
    # needs `ends_down` GREATER than the height, which is not obvious and was wrong: at
    # 0.55 of the height the end rings stood 3 mm proud with their interiors showing.
    height = 0.0066
    co = verts(P.handle(0.070, 0.030, 0.021, height, ends_down=height * 1.15))
    tips = _band(co, 0.036, 1.0)
    assert len(tips) and tips[:, 2].max() < 0.0


# --------------------------------------------------------------------------- #
# the pocket
# --------------------------------------------------------------------------- #
def test_trim_ring_has_a_hole_in_it():
    r_in = 0.062
    co = verts(P.trim_ring(r_in=r_in, r_out=0.076, screws=0))
    assert radii(co).min() > r_in * 0.98, "a trim ring that reaches the axis is a plate"


def test_pressed_dish_floor_sits_below_the_cap_face():
    cap_r, sink = 0.039, 0.015
    co = verts(P.pressed_dish(cap_r=cap_r, gap=0.006, sink=sink, wall_z=0.011))
    r = radii(co)
    # everything in the annulus just outside the cap has to be at least `sink` down, or the
    # dish is lying across the very thing the dataset is about
    near = co[(r > cap_r * 1.15) & (r < cap_r * 1.55), 2]
    assert len(near) and near.max() < -sink * 0.75


def test_pocket_hardware_keeps_off_the_cap():
    cap_r = 0.039
    co = verts(P.well_details(0.062, 0.092, 0.033, ribs=5, screws=3, catches=2,
                              grommets=2, keep_out=cap_r, seed=7))
    # Only what could stand IN FRONT of the cap counts. The pocket floor is a disc that
    # reaches the axis by construction and sits three centimetres down, behind the cap and
    # invisible; a bracket at the cap's own height is the failure.
    near = co[co[:, 2] > -0.020]
    assert len(near) and radii(near).min() > cap_r, \
        "a screw lying on the cap still measures as a cap"


# --------------------------------------------------------------------------- #
# the assembly
# --------------------------------------------------------------------------- #
def test_the_opening_is_a_rounded_rectangle_not_a_circle():
    # `superellipse_plan` has no aspect ratio, so everything built with it came out as wide
    # as it is tall. Every reference opening is about 170 by 150 with a 25 mm corner radius.
    co = verts(P.filler_box(open_w=0.170, open_h=0.152, sq=4.4, drain=False))
    top = co[co[:, 2] > -0.002]
    wx = top[:, 0].max() - top[:, 0].min()
    hy = top[:, 1].max() - top[:, 1].min()
    assert 1.06 < wx / hy < 1.20, f"aspect {wx / hy:.3f}"
    # and it is SQUARE-ish: a circle's corner radius equals its radius, a superellipse's is
    # much smaller, so the diagonal reaches further than the axis does
    r_ax = 0.5 * wx
    diag = max(math.hypot(x, y) for x, y, z in top)
    assert diag > r_ax * 1.10, "the corners must reach past the sides — this is a rectangle"


def test_the_filler_boss_can_sit_off_centre():
    # A deliberately small boss, so its points and the drafted wall's cannot overlap in
    # radius — the wall is at 73 mm here and the boss reaches 50.
    off = (0.012, -0.020)
    co = verts(P.filler_box(boss_off=off, neck_r=0.030, neck_h=0.010, drain=False))
    floor = co[:, 2].min()
    boss = co[(co[:, 2] > floor + 0.002) & (radii(co) < 0.055)]
    cx = 0.5 * (boss[:, 0].min() + boss[:, 0].max())
    cy = 0.5 * (boss[:, 1].min() + boss[:, 1].max())
    assert abs(cx - off[0]) < 0.004 and abs(cy - off[1]) < 0.004, f"boss at {cx:.3f},{cy:.3f}"


@pytest.mark.parametrize("open_deg", [120.0, 150.0, 175.0])
def test_an_open_door_does_not_lie_across_its_own_opening(open_deg):
    """A door swung back must be BESIDE the pocket, not over it.

    The old part hinged along its own bottom edge and stood out at ninety degrees; whether
    it cleared the opening depended on numbers nobody was checking. The camera looks down
    the pocket's axis, so a door over the opening hides the entire subject."""
    w, h, hinge_r = 0.196, 0.176, 0.104
    co = verts(P.fuel_door(w=w, h=h, open_deg=open_deg, az=180.0, hinge_r=hinge_r))
    over = co[(np.abs(co[:, 0]) < w * 0.42) & (np.abs(co[:, 1]) < h * 0.42)]
    assert len(over) == 0, f"{len(over)} door vertices over the opening at {open_deg} deg"


@pytest.mark.parametrize("az", [0.0, 90.0, 180.0, 270.0])
def test_the_door_hinges_on_the_side_it_was_told_to(az):
    co = verts(P.fuel_door(open_deg=150.0, az=az, hinge_r=0.100))
    c = co.mean(0)
    want = np.array([math.cos(math.radians(az)), math.sin(math.radians(az))])
    got = c[:2] / (np.linalg.norm(c[:2]) + 1e-12)
    assert float(got @ want) > 0.9, f"az={az}: door centroid at {c[:2]}"


@pytest.mark.parametrize("style", ["liner", "dish", "box"])
def test_nothing_is_placed_over_the_cap_face(style):
    """No surface may sit above the cap's own face inside the cap's radius.

    This is the one statement that would have caught the trim plate, the flush dish floor
    AND the screws on the cap in a single line — every occlusion failure this case has had
    is 'something ended up in front of the subject', and the subject is at the origin
    looking up +z by construction."""
    rng = np.random.default_rng(3)
    for _ in range(14):
        v = S.sample(rng, domain="prod")
        v["pocket_style"] = style
        prog, gt = S.build(v)
        co = np.array([q.co for q in prog.build().verts], float)
        # the assembly is tilted, so measure in the cap's own frame
        R = S._rot_xyz(v["tilt_x"], v["tilt_y"], 0.0)
        local = co @ R
        r = np.hypot(local[:, 0], local[:, 1])
        # the grip's own ceiling: the handle's height, or the slot family's raised land
        top = max(v["rib_h"], 0.0035) + max(v["dome"], 0.0) + 2e-4
        over = local[(r < v["d_cap"] / 2 * 0.92) & (local[:, 2] > top + 1e-4)]
        assert len(over) == 0, (
            f"{style}: {len(over)} vertices above the cap face, highest "
            f"{over[:, 2].max() * 1000:.1f} mm vs handle top {top * 1000:.1f} mm")


def test_both_pocket_families_are_actually_drawn():
    rng = np.random.default_rng(0)
    seen = {S.sample(rng)["pocket_style"] for _ in range(60)}
    assert seen == {"box", "dish"}


def test_every_variant_builds():
    rng = np.random.default_rng(17)
    for _ in range(12):
        S.build(S.sample(rng))[0].build().validate()
