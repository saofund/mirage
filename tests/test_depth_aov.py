"""The renderer's metric depth AOV — `mirage_render --depth`.

The AOV exists so a render can become a point cloud, which means the only thing worth
testing about it is whether unprojecting it through the matching intrinsics reconstructs
the geometry that was rendered. A "does it write a file" test would have passed on the
first draft of every bug this catches.

The one that matters: depth is distance along the VIEW AXIS, not along the ray. The two
differ by 1/cos(angle off axis) — 8 % at the corners of a 40-degree field — and a cloud
built from ray distance comes out domed, so every plane fitted to it comes out curved.
A flat surface rendered head-on and unprojected is exactly the test that separates them.
"""
import os
import shutil
import subprocess
import sys

import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

RENDER = os.path.join(ROOT, "core", "build", "Release", "mirage_render.exe")
if not os.path.exists(RENDER):
    RENDER = os.path.join(ROOT, "core", "build", "mirage_render")
pytestmark = pytest.mark.skipif(not os.path.exists(RENDER),
                                reason="mirage_render not built")

from mirage.meshlang import MeshProgram  # noqa: E402

FOV = 0.9
W = H = 160


def read_pfm(path):
    with open(path, "rb") as f:
        assert f.readline().strip() == b"Pf", "not a single-channel PFM"
        w, h = (int(x) for x in f.readline().split())
        scale = float(f.readline())
        d = np.frombuffer(f.read(w * h * 4), "<f4" if scale < 0 else ">f4").reshape(h, w)
        return d[::-1].copy()                       # PFM rows are stored bottom-up


def render(tmp_path, prog, eye, target, ground=False, up=(0, 1, 0), extra=()):
    """Note the non-default `up`. These cameras look straight down -z, and the renderer's
    default up IS +z: leave it and the view basis is degenerate (right = fwd x up = 0).
    The first draft of this file did exactly that and got a frame where every pixel
    reported the same depth, which looks a lot like a correct answer."""
    js = tmp_path / "m.json"
    js.write_text(prog.to_json(), encoding="utf-8")
    out = tmp_path / "d.pfm"
    cmd = [RENDER, "--oplog", str(js), "--out", str(tmp_path / "c.ppm"),
           "--depth", str(out), "--w", str(W), "--h", str(H), "--spp", "1",
           "--cam-eye", *map(str, eye), "--cam-target", *map(str, target),
           "--cam-up", *map(str, up),
           "--cam-fov", str(FOV), *([] if ground else ["--no-ground"]), *extra]
    r = subprocess.run(cmd, capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    return read_pfm(out)


def test_depth_is_metric_and_along_the_view_axis(tmp_path):
    """A plate 2 m in front of the camera reads 2.0 everywhere — not 2.0/cos(theta)."""
    plate = (MeshProgram().cube(size=1.0).scale({"by": "all"}, [4.0, 4.0, 0.02]))
    d = render(tmp_path, plate, eye=(0, 0, 2.0), target=(0, 0, 0))
    hit = d > 0
    assert hit.mean() > 0.5, "the plate should fill most of the frame"
    # 2.0 m eye, plate half-thickness 0.01 -> front face at z = 0.01
    assert abs(float(np.median(d[hit])) - 1.99) < 0.02
    # The decisive one: a plane parallel to the sensor is CONSTANT in a view-axis depth
    # map. Ray distance would grow toward the corners by 1/cos, which over this field is
    # several percent — an order of magnitude more than the spread allowed here.
    assert float(d[hit].std()) < 0.004, "depth varies across a flat plate: ray distance?"


def test_depth_unprojects_to_the_modelled_geometry(tmp_path):
    """Unproject with the matching K and the recovered box is the box that was built."""
    sx, sy, sz = 0.30, 0.22, 0.10
    box = MeshProgram().cube(size=1.0).scale({"by": "all"}, [sx, sy, sz])
    d = render(tmp_path, box, eye=(0, 0, 1.2), target=(0, 0, 0))
    fy = (H / 2.0) / np.tan(FOV / 2.0)
    fx, cx, cy = fy, W / 2.0, H / 2.0
    yy, xx = np.mgrid[0:H, 0:W]
    hit = d > 0
    X = (xx + 0.5 - cx) / fx * d
    Y = (yy + 0.5 - cy) / fy * d
    top = hit & (d < np.median(d[hit]) + 0.005)      # the face pointing at the camera
    assert top.sum() > 500
    # the top face sits at z = sz/2, i.e. 1.2 - 0.05 = 1.15 from the eye
    assert abs(float(np.median(d[top])) - 1.15) < 0.01
    # ...and it is sx by sy across. Half a pixel of footprint is ~0.004 m here.
    assert abs(float(np.ptp(X[top])) - sx) < 0.012
    assert abs(float(np.ptp(Y[top])) - sy) < 0.012


def test_depth_is_zero_where_nothing_was_hit(tmp_path):
    """Misses must be 0, not the far plane: a caller filters on ``depth > 0``."""
    small = MeshProgram().cube(size=1.0).scale({"by": "all"}, [0.05, 0.05, 0.05])
    d = render(tmp_path, small, eye=(0, 0, 2.0), target=(0, 0, 0))
    assert (d == 0).mean() > 0.8, "most of this frame is empty and must read exactly 0"
    assert float(d.max()) < 10.0


def test_depth_and_ids_agree_pixel_for_pixel(tmp_path):
    """Both AOVs ride the same centre ray, so a tagged pixel always has a depth."""
    prog = (MeshProgram().cube(size=1.0).scale({"by": "all"}, [0.4, 0.4, 0.1])
            .tag({"by": "all"}, "slab"))
    js = tmp_path / "m.json"
    js.write_text(prog.to_json(), encoding="utf-8")
    cmd = [RENDER, "--oplog", str(js), "--out", str(tmp_path / "c.ppm"),
           "--depth", str(tmp_path / "d.pfm"), "--ids", str(tmp_path / "i.pgm"),
           "--id-tags", "slab", "--w", str(W), "--h", str(H), "--spp", "1",
           "--cam-eye", "0", "0", "1.5", "--cam-target", "0", "0", "0",
           "--cam-up", "0", "1", "0", "--cam-fov", str(FOV), "--no-ground"]
    r = subprocess.run(cmd, capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    d = read_pfm(tmp_path / "d.pfm")
    with open(tmp_path / "i.pgm", "rb") as f:
        assert f.readline().strip() == b"P5"
        w, h = (int(x) for x in f.readline().split())
        f.readline()
        ids = np.frombuffer(f.read(w * h * 2), ">u2").reshape(h, w)
    tagged = ids > 0
    assert tagged.sum() > 500
    assert (d[tagged] > 0).all(), "a pixel with an object id must carry a depth"


def test_no_ground_removes_the_implicit_floor(tmp_path):
    """--no-ground: a close-up of a part must not have a floor plane through it."""
    part = MeshProgram().cube(size=1.0).scale({"by": "all"}, [0.05, 0.05, 0.05])
    with_ground = render(tmp_path, part, eye=(0.4, -0.4, 0.3), target=(0, 0, 0), ground=True)
    tmp2 = tmp_path / "b"
    tmp2.mkdir()
    without = render(tmp2, part, eye=(0.4, -0.4, 0.3), target=(0, 0, 0), ground=False)
    # the floor fills most of a frame aimed down at a 50 mm cube; without it, only the cube
    assert (with_ground > 0).mean() > 0.5
    assert (without > 0).mean() < 0.15
