"""Camera and space probes.

The load-bearing test is the round trip: project a known world point to a pixel, render
the AOVs, then unproject that pixel through its depth and land back on the point. That is
the only check that the Python camera and the tracer's camera are the same camera -- a
tool with its own slightly different projection produces confident, wrong millimetres.
"""

import os
import shutil
import subprocess

import numpy as np
import pytest

from mirage.meshlang import MeshProgram
from mirage.probe import (Probe, basis, face_screen_pos, project, ray, to_pixels,
                          unproject)
from mirage.visibility import face_ids, read_pfm

POSE = {"eye": (0.0, 0.0, 3.0), "target": (0.0, 0.0, 0.0), "up": (0.0, 1.0, 0.0),
        "fov": 0.7}
RENDER = shutil.which("mirage_render") or "core/build/Release/mirage_render"
needs_render = pytest.mark.skipif(not os.path.exists(RENDER),
                                  reason="mirage_render not built")


def test_basis_is_orthonormal_and_right_handed():
    eye, fwd, right, up2 = basis(POSE)
    for v in (fwd, right, up2):
        assert np.linalg.norm(v) == pytest.approx(1.0)
    assert abs(fwd @ right) < 1e-12 and abs(fwd @ up2) < 1e-12 and abs(right @ up2) < 1e-12
    assert np.allclose(np.cross(right, fwd), up2)


def test_image_y_points_down():
    """A point ABOVE the target must land in the UPPER half of the frame."""
    px = to_pixels([[0.0, 0.5, 0.0]], POSE, 200, 200)
    assert px[0, 1] < 100.0
    assert to_pixels([[0.0, -0.5, 0.0]], POSE, 200, 200)[0, 1] > 100.0


def test_centre_pixel_maps_to_the_target():
    px = to_pixels([[0.0, 0.0, 0.0]], POSE, 200, 160)
    assert px[0] == pytest.approx([100.0, 80.0])


def test_ray_and_pixel_are_inverse():
    for x, y in ((0, 0), (37, 12), (99, 79), (199, 159)):
        eye, d = ray(POSE, x, y, 200, 160)
        back = to_pixels([eye + d * 2.0], POSE, 200, 160)[0]
        assert back == pytest.approx([x + 0.5, y + 0.5], abs=1e-9)


def test_unproject_uses_view_axis_depth_not_ray_length():
    """A plane at z = 0 has ONE depth everywhere; unprojecting must land on the plane.

    Treating the AOV as distance along the ray instead bends the plane toward the camera
    at the corners -- the failure this test exists to prevent, worth 5% of the depth at
    the edge of a 0.7 rad frame.
    """
    w, h = 120, 90
    for x, y in ((5, 5), (60, 45), (115, 85)):
        p = unproject(POSE, x, y, 3.0, w, h)      # the plane z=0 is 3.0 in front of the eye
        assert p[2] == pytest.approx(0.0, abs=1e-9)


def test_probe_rejects_aovs_from_different_frames():
    with pytest.raises(ValueError):
        Probe(POSE, depth=np.zeros((4, 4)), face_ids=np.zeros((5, 5), int))
    with pytest.raises(ValueError):
        Probe(POSE)


def test_probe_reports_background_and_bounds():
    p = Probe(POSE, depth=np.zeros((4, 4)), face_ids=np.full((4, 4), -1))
    assert p.raycast(1, 1).hit is False
    with pytest.raises(IndexError):
        p.raycast(4, 0)


def test_pick_ranks_faces_by_coverage():
    fids = np.array([[1, 1, 2, -1], [1, 3, 2, 2], [1, 1, 2, 2]])
    p = Probe(POSE, face_ids=fids)
    assert list(p.pick((0, 0, 4, 3))) == [1, 2, 3]
    assert p.pick((0, 0, 4, 3)) == {1: 5, 2: 5, 3: 1} or p.pick((0, 0, 4, 3))[3] == 1
    assert p.pick((1, 1, 2, 2)) == {3: 1}
    assert p.pick((3, 0, 4, 1)) == {}
    assert p.coverage(2) == 5


def test_pick_accepts_a_mask():
    fids = np.array([[1, 1], [2, 2]])
    m = np.array([[False, False], [True, True]])
    assert Probe(POSE, face_ids=fids).pick(m) == {2: 2}
    with pytest.raises(ValueError):
        Probe(POSE, face_ids=fids).pick(np.zeros((3, 3), bool))


@needs_render
def test_round_trip_through_the_renderers_own_camera(tmp_path):
    """Project -> render -> unproject must return the point it started from."""
    prog = MeshProgram().plane(size_x=1.6, size_y=1.6)
    log = tmp_path / "p.json"
    log.write_text(prog.to_json())
    w, h = 200, 160
    subprocess.run([RENDER, "--oplog", str(log), "--out", str(tmp_path / "p.ppm"),
                    "--w", str(w), "--h", str(h), "--spp", "1", "--no-ground",
                    "--depth", str(tmp_path / "d.pfm"),
                    "--normal", str(tmp_path / "n.pfm"),
                    "--face-ids", str(tmp_path / "f.pfm"),
                    "--cam-eye", "0", "0", "3.0", "--cam-target", "0", "0", "0",
                    "--cam-up", "0", "1", "0", "--cam-fov", "0.7"],
                   check=True, stdout=subprocess.DEVNULL)
    pr = Probe(POSE, depth=read_pfm(tmp_path / "d.pfm"),
               normal=read_pfm(tmp_path / "n.pfm"),
               face_ids=face_ids(tmp_path / "f.pfm"))

    for world in ([0.0, 0.0, 0.0], [0.35, -0.22, 0.0], [-0.4, 0.3, 0.0]):
        px = to_pixels([world], POSE, w, h)[0]
        s = pr.raycast(int(px[0]), int(px[1]))
        assert s.hit, f"{world} projected to {px} and hit nothing"
        # within half a pixel of the point we started from
        half_px = 2.0 * math_tan_half() * 3.0 / h
        assert np.allclose(s.world, world, atol=half_px), f"{s.world} vs {world}"
        assert abs(abs(s.normal[2]) - 1.0) < 1e-6, "a z-plane should face z"
        assert s.depth == pytest.approx(3.0, abs=1e-6)


def math_tan_half():
    import math
    return math.tan(0.7 * 0.5)


@needs_render
def test_face_screen_pos_agrees_with_the_face_id_aov(tmp_path):
    """Where the model thinks a face is must be where the renderer put it."""
    prog = MeshProgram().cube(size=1.0).subdivide(levels=1)
    log = tmp_path / "c.json"
    log.write_text(prog.to_json())
    w, h = 200, 160
    subprocess.run([RENDER, "--oplog", str(log), "--out", str(tmp_path / "c.ppm"),
                    "--w", str(w), "--h", str(h), "--spp", "1", "--no-ground",
                    "--face-ids", str(tmp_path / "f.pfm"),
                    "--cam-eye", "0", "0", "3.0", "--cam-target", "0", "0", "0",
                    "--cam-up", "0", "1", "0", "--cam-fov", "0.7"],
                   check=True, stdout=subprocess.DEVNULL)
    fids = face_ids(tmp_path / "f.pfm")
    rows = face_screen_pos(prog.build(), POSE, w, h, face_ids=fids)
    pr = Probe(POSE, face_ids=fids)

    visible = [r for r in rows if r["visible_px"] > 12]
    assert visible, "nothing was visible at all"
    for r in visible:
        x, y = int(r["pixel"][0]), int(r["pixel"][1])
        under = pr.pick((x - 1, y - 1, x + 2, y + 2))
        assert r["face"] in under, (
            f"face {r['face']} says it is at ({x}, {y}) but the AOV has {under} there")
