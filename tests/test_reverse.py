"""mirage.reverse — measured geometry in, op-log out.

The tests are round trips, because that is the only thing worth asserting here: build a
shape with known dimensions, render it into a depth map (or sample it directly), recover a
section, spin the section, and check the result is the shape that was started with. A test
that only checks "returns an array of the right length" would pass on every sign error and
every convention flip this module could possibly make.
"""
import math

import numpy as np
import pytest

from mirage.kernel import spin
from mirage.meshlang import MeshProgram
from mirage.reverse import (
    cloud_from_depth, plan_from_cloud, principal_axis, section_from_cloud,
    section_to_profile, superellipse_plan,
)


def _cone_cloud(r_max=0.05, slope=0.4, n=40000, seed=0):
    """A cone opening upward: z = slope * r. Section is a straight line of known gradient."""
    rng = np.random.default_rng(seed)
    r = r_max * np.sqrt(rng.random(n))
    a = rng.random(n) * 2 * math.pi
    return np.stack([r * np.cos(a), r * np.sin(a), slope * r], 1)


def test_section_recovers_a_known_gradient():
    P = _cone_cloud(slope=0.4)
    sec = section_from_cloud(P, origin=(0, 0, 0), axis=(0, 0, -1), step=0.002)
    assert len(sec) > 15
    # height is measured along `axis`, which points at -z here, so the recovered gradient
    # is -0.4; what matters is that it is a straight line of the right magnitude
    g = np.polyfit(sec[:, 0], sec[:, 1], 1)[0]
    assert abs(abs(g) - 0.4) < 0.02, f"recovered gradient {g}"


def test_section_is_robust_to_flying_pixels():
    """A median reduction is the point: outliers on every edge are what real depth has."""
    P = _cone_cloud(slope=0.4, n=20000)
    rng = np.random.default_rng(1)
    bad = rng.random(len(P)) < 0.08
    P[bad, 2] += rng.normal(0, 0.05, bad.sum())        # 8% wild outliers
    sec = section_from_cloud(P, origin=(0, 0, 0), axis=(0, 0, -1), step=0.002)
    g = np.polyfit(sec[:, 0], sec[:, 1], 1)[0]
    assert abs(abs(g) - 0.4) < 0.03, f"outliers moved the gradient to {g}"


def test_principal_axis_points_at_the_viewer():
    """Sign convention: a mirrored axis mirrors every section built through it."""
    rng = np.random.default_rng(2)
    P = np.stack([rng.normal(0, 0.05, 3000), rng.normal(0, 0.05, 3000),
                  np.full(3000, 0.5)], 1)
    _, n = principal_axis(P)
    assert n[2] < 0, "axis must point back toward the camera"


def test_cloud_from_depth_unprojects_a_plane_flat():
    h, w = 60, 80
    K = np.array([[90.0, 0, w / 2], [0, 90.0, h / 2], [0, 0, 1]])
    depth = np.full((h, w), 1.5)
    P = cloud_from_depth(depth, K)
    assert len(P) == h * w
    assert abs(P[:, 2].mean() - 1.5) < 1e-9
    assert P[:, 2].std() < 1e-9                        # a fronto-parallel plane is flat
    # the recovered extent must match the field of view: w/fx * z
    assert abs(np.ptp(P[:, 0]) - (w - 1) / 90.0 * 1.5) < 1e-6


def test_cloud_from_depth_skips_holes():
    h, w = 20, 20
    K = np.array([[40.0, 0, 10.0], [0, 40.0, 10.0], [0, 0, 1]])
    depth = np.full((h, w), 2.0)
    depth[::2, :] = 0.0                                # every other row has no answer
    P = cloud_from_depth(depth, K)
    assert len(P) == h * w // 2
    assert (P[:, 2] > 0).all()


def test_section_to_profile_closes_on_the_axis():
    """spin() needs a polyline starting and ending at radius 0, or it makes an open tube."""
    sec = np.array([[0.02, 0.0], [0.03, -0.01], [0.04, -0.005], [0.05, 0.004]])
    prof = section_to_profile(sec)
    assert abs(prof[0][0]) < 1e-12 and abs(prof[-1][0]) < 1e-12
    m = MeshProgram().profile(prof, plane="xz").spin(steps=24).build()
    assert m.stats()["closed_manifold"], "the profile did not close into a solid"


def test_round_trip_cone_section_to_solid():
    """Measure a cone, rebuild it, and get a solid whose size matches the measurement."""
    P = _cone_cloud(r_max=0.05, slope=0.4)
    sec = section_from_cloud(P, origin=(0, 0, 0), axis=(0, 0, -1), step=0.002)
    prog = MeshProgram().profile(section_to_profile(sec), plane="xz").spin(steps=32)
    st = prog.get_state()
    assert st["stats"]["closed_manifold"]
    assert abs(st["size"][0] - 2 * 0.05 * 1.04) < 0.006   # profile pads the outer radius 4%


# --------------------------------------------------------------------------- #
# the generalised lathe
# --------------------------------------------------------------------------- #
def test_plan_none_is_exactly_the_classical_lathe():
    """The default must not move a single vertex, or every existing model shifts."""
    # a fresh program each time: MeshProgram.add appends to self and returns self, so
    # reusing one here would spin the second call's mesh twice
    prof = [(0, 0.02), (0.03, 0.02), (0.03, 0), (0, 0)]
    a = MeshProgram().profile(prof, plane="xz").spin(steps=16).build()
    b = MeshProgram().profile(prof, plane="xz").spin(steps=16, plan=None).build()
    pa = np.array([list(v.co) for v in a.verts])
    pb = np.array([list(v.co) for v in b.verts])
    assert pa.shape == pb.shape
    assert np.abs(pa - pb).max() == 0.0


def test_superellipse_plan_is_normalised():
    """Default normalisation preserves AREA, not perimeter.

    A ring's contribution to anything measured over a surface goes as r^2, so a plan whose
    arithmetic mean is 1 still enlarges the swept area — and a section binned by physical
    radius then samples further in wherever the plan pushed out, which reads as the outer
    surface sitting low. `by="mean"` is still available for callers that want the mean."""
    ks = superellipse_plan(3.6, 64)
    rms = (sum(k * k for k in ks) / len(ks)) ** 0.5
    assert abs(rms - 1.0) < 1e-12, "the default must preserve area"
    assert max(ks) > 1.05 and min(ks) < 0.98, "n=3.6 should be visibly non-circular"
    km = superellipse_plan(3.6, 64, by="mean")
    assert abs(sum(km) / len(km) - 1.0) < 1e-12, "by='mean' must preserve the mean"


def test_plan_squares_the_outline_without_changing_the_section():
    """The whole point: same heights, different plan."""
    prof = [(0, 0.0), (0.04, 0.0), (0.04, -0.02), (0, -0.02)]
    ks = superellipse_plan(4.0, 48)
    a = MeshProgram().profile(prof, plane="xz").spin(steps=48).build()
    b = MeshProgram().profile(prof, plane="xz").spin(steps=48, plan=ks).build()
    pa = np.array([list(v.co) for v in a.verts])
    pb = np.array([list(v.co) for v in b.verts])
    # heights untouched
    assert abs(np.sort(pa[:, 2]) - np.sort(pb[:, 2])).max() < 1e-12
    # the plan is squarer: the corner reaches further than the circle did
    ra = np.hypot(pa[:, 0], pa[:, 1])
    rb = np.hypot(pb[:, 0], pb[:, 1])
    assert rb.max() > ra.max() * 1.03
    assert abs(rb.mean() - ra.mean()) < ra.mean() * 0.02   # same size, different shape


def test_plan_from_fades_in_so_a_part_can_be_round_inside_and_pressed_outside():
    prof = [(0, 0.0), (0.02, 0.0), (0.05, -0.01), (0.05, -0.03), (0, -0.03)]
    ks = superellipse_plan(5.0, 48)
    m = MeshProgram().profile(prof, plane="xz").spin(steps=48, plan=ks,
                                                     plan_from=0.035).build()
    P = np.array([list(v.co) for v in m.verts])
    r = np.hypot(P[:, 0], P[:, 1])
    inner = P[(r > 0.015) & (r < 0.025)]
    outer = P[r > 0.045]
    ri = np.hypot(inner[:, 0], inner[:, 1])
    ro = np.hypot(outer[:, 0], outer[:, 1])
    assert np.ptp(ri) < 0.0005, "inside plan_from the part must stay round"
    assert np.ptp(ro) > 0.002, "outside it the plan must take effect"


def test_plan_from_cloud_recovers_a_squarish_outline():
    """Measure a plan back off a cloud of the thing the plan built."""
    ks = superellipse_plan(4.0, 64)
    prof = [(0, 0.0), (0.05, 0.0), (0.05, -0.01), (0, -0.01)]
    m = MeshProgram().profile(prof, plane="xz").spin(steps=64, plan=ks).build()
    P = np.array([list(v.co) for v in m.verts])
    P = P[P[:, 2] > -0.005]                        # the top face only
    got = plan_from_cloud(P, origin=(0, 0, 0), axis=(0, 0, -1), steps=16,
                          r_band=(0.03, 0.08))
    want = superellipse_plan(4.0, 16)
    # both normalised, so compare shape: the diagonals must exceed the axes in both
    assert np.corrcoef(got, want)[0, 1] > 0.85, f"recovered plan {got}"
    assert max(got) > 1.03 and min(got) < 0.97, "recovered plan is not squarish"


@pytest.mark.parametrize("n", [2.0, 3.6, 8.0])
def test_plan_keeps_the_mesh_valid(n):
    prof = [(0, 0.0), (0.04, 0.0), (0.04, -0.02), (0, -0.02)]
    st = (MeshProgram().profile(prof, plane="xz")
          .spin(steps=40, plan=superellipse_plan(n, 40)).get_state())
    assert st["stats"]["closed_manifold"]
