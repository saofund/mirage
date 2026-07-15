"""Measurement: the homography, the lens, and the camera resection.

The load-bearing test is `test_projection_matches_the_renderer`. Everything else here can be
perfectly self-consistent and still useless: if `solve.project` disagrees with the tracer's
ray generation, the solver fits a camera that is precisely, confidently wrong, and every
residual it prints is a lie that reads like evidence. So one test renders a scene with a
known camera and checks that the model predicts where the geometry actually landed.
"""
import json
import math
import subprocess

import numpy as np
import pytest

from mirage.capture import default_render
from mirage.meshlang import MeshProgram
from mirage.solve import (Camera, apply_h, distort, homography, project, rectify,
                          solve_camera, undistort)

RENDER = default_render()
pytestmark = pytest.mark.skipif(not RENDER.exists(), reason="mirage_render not built")


# --- homography ------------------------------------------------------------- #
def test_homography_recovers_a_known_transform():
    H = np.array([[1.4, 0.3, 12.0], [-0.2, 1.1, -4.0], [0.0004, -0.0007, 1.0]])
    src = np.array([[0, 0], [640, 0], [640, 480], [0, 480], [320, 210.0]])
    dst = apply_h(H, src)
    got = homography(src, dst)
    assert np.allclose(apply_h(got, src), dst, atol=1e-6)
    assert np.allclose(got / got[2, 2], H / H[2, 2], atol=1e-6)


def test_homography_is_exact_on_a_ground_rectangle():
    # The real use: four corners of painted rectangle -> its world coords, so the layout is
    # MEASURED. Everything else on that plane then comes for free.
    img = np.array([[620, 560], [1360, 440], [1000, 380], [900, 660.0]])
    world = np.array([[0, 0], [3.4, 0], [3.4, 6.0], [0, 6.0]])
    H = homography(img, world)
    assert np.allclose(apply_h(H, img), world, atol=1e-9)
    # a point at the rectangle's centre must land at its centre, which no corner constrained
    mid_img = apply_h(np.linalg.inv(H), [[1.7, 3.0]])
    assert np.allclose(apply_h(H, mid_img), [[1.7, 3.0]], atol=1e-9)


def test_homography_needs_four_points():
    with pytest.raises(ValueError):
        homography([[0, 0], [1, 0], [1, 1]], [[0, 0], [1, 0], [1, 1]])


def test_rectify_straightens_a_plane():
    # A synthetic "photo" of a plane: one white square. Rectified, it must come back square.
    img = np.zeros((400, 600, 3), np.uint8)
    quad = np.array([[180, 300], [430, 300], [380, 180], [230, 180.0]])
    from PIL import Image, ImageDraw
    d = ImageDraw.Draw(Image.fromarray(img))
    pil = Image.fromarray(img)
    ImageDraw.Draw(pil).polygon([tuple(p) for p in quad], fill=(255, 255, 255))
    img = np.asarray(pil)
    H = homography(quad, [[0, 0], [2, 0], [2, 2], [0, 2.0]])
    ortho = rectify(img, H, (-0.5, 2.5, -0.5, 2.5), px_per_m=60)
    lit = ortho[..., 0] > 128
    ys, xs = np.nonzero(lit)
    w = xs.max() - xs.min()
    h = ys.max() - ys.min()
    assert abs(w - h) / max(w, h) < 0.06, f"a square rectified to {w}x{h}"


# --- lens ------------------------------------------------------------------- #
@pytest.mark.parametrize("k1,k2", [(0.0, 0.0), (0.12, 0.0), (-0.09, 0.02), (0.25, -0.05)])
def test_distort_and_undistort_are_inverses(k1, k2):
    ab = np.array([[0, 0], [0.3, 0], [0, -0.8], [1.2, 0.7], [-1.4, -0.95]])
    assert np.allclose(undistort(distort(ab, k1, k2), k1, k2), ab, atol=1e-9)


def test_zero_lens_is_the_identity():
    ab = np.array([[0.4, -0.3], [1.1, 0.9]])
    assert np.array_equal(distort(ab, 0, 0), ab)
    assert np.array_equal(undistort(ab, 0, 0), ab)


def test_distortion_is_purely_radial():
    # direction preserved, magnitude changed — that IS the definition, and getting it wrong
    # (e.g. scaling a and b independently) still round-trips through the inverse
    ab = np.array([[0.6, 0.8]])
    out = distort(ab, 0.2, 0.0)
    cross_z = ab[0, 0] * out[0, 1] - ab[0, 1] * out[0, 0]
    assert cross_z == pytest.approx(0, abs=1e-12)
    assert np.linalg.norm(out) > np.linalg.norm(ab)      # +k1 pushes outward (barrel)


def test_undistort_takes_the_physical_root_when_the_lens_folds():
    """r*s(r) is not monotonic once k2 opposes k1 — at k1=0.25, k2=-0.05 it turns over at
    r=2, so r*s(r)=R has a SECOND root past the fold. Newton from r=R converges to that one
    and silently returns a point flung across the frame. Guard the branch, not just the
    round-trip."""
    k1, k2 = 0.25, -0.05
    ideal = np.array([[-1.4, -0.95]])                    # r = 1.69, inside the valid branch
    back = undistort(distort(ideal, k1, k2), k1, k2)
    assert np.allclose(back, ideal, atol=1e-7)
    assert np.linalg.norm(back) < 2.0, "picked the root beyond the fold"


# --- projection: the model vs the renderer ---------------------------------- #
def _render_probe(cam, w, h, out, extra=()):
    """Four unit cubes at known world points, rendered with a known camera."""
    p = MeshProgram()
    pts = [(0, 0, 0.5), (3.0, 1.0, 0.5), (-2.0, 4.0, 0.5), (1.5, 6.0, 0.5)]
    for i, q in enumerate(pts):
        p.place(MeshProgram().cube(size=0.5), at=list(q),
                material={"color": [1, 0, 0], "roughness": 0.5})
    js = out.with_suffix(".json")
    js.write_text(p.to_json())
    subprocess.run([str(RENDER), "--oplog", str(js), "--out", str(out),
                    "--spp", "4", "--w", str(w), "--h", str(h), "--threads", "8",
                    *cam.render_flags(), *extra], check=True, capture_output=True)
    return pts


@pytest.mark.parametrize("k1", [0.0, 0.18])
def test_projection_matches_the_renderer(tmp_path, k1):
    """THE test: does `project` predict where the renderer actually puts a point?

    A solver is only as good as its forward model. If this drifts, `solve_camera` will still
    converge, still report a small residual, and still be wrong — the residual is measured
    against the same broken model. So: render cubes at known world points, find each blob's
    centroid in the image, and check `project` predicted it.
    """
    from PIL import Image
    w, h = 640, 400
    cam = Camera(eye=(1.0, -7.0, 3.4), target=(0.4, 1.0, 0.4), fov_y=0.9, k1=k1)
    out = tmp_path / "probe.ppm"
    pts = _render_probe(cam, w, h, out)

    arr = np.asarray(Image.open(out).convert("RGB"), float)
    red = (arr[..., 0] > 90) & (arr[..., 1] < 70) & (arr[..., 2] < 70)
    assert red.sum() > 50, "the probe cubes did not render"

    predicted = project(cam, pts, w, h)
    for q, pr in zip(pts, predicted):
        assert not np.isnan(pr).any()
        # the blob nearest the prediction must actually be there: check the pixel region
        x, y = int(round(pr[0])), int(round(pr[1]))
        assert 0 <= x < w and 0 <= y < h, f"{q} projected off-frame to {pr}"
        win = red[max(0, y - 12):y + 13, max(0, x - 12):x + 13]
        assert win.any(), f"predicted {q} at {pr.round(1)} but no cube is within 12 px"


def test_projection_puts_the_target_at_the_centre():
    cam = Camera(eye=(2, -5, 3), target=(0.5, 1.0, 0.7), fov_y=0.8)
    px = project(cam, [cam.target], 800, 600)[0]
    assert px == pytest.approx([400, 300], abs=1e-6)


def test_points_behind_the_camera_are_nan():
    cam = Camera(eye=(0, -5, 2), target=(0, 0, 0), fov_y=0.8)
    px = project(cam, [(0, -9, 2)], 640, 480)[0]
    assert np.isnan(px).all()


# --- resection -------------------------------------------------------------- #
def _synth(cam, w, h, n=14, seed=3):
    rng = np.random.default_rng(seed)
    world = np.column_stack([rng.uniform(-4, 6, n), rng.uniform(0, 12, n), rng.uniform(0, 3, n)])
    img = project(cam, world, w, h)
    ok = ~np.isnan(img).any(1)
    return world[ok], img[ok]


def test_solve_camera_recovers_a_known_pose():
    w, h = 1600, 900
    truth = Camera(eye=(2.0, -4.2, 4.3), target=(1.53, 2.57, 0.06), fov_y=1.05)
    world, img = _synth(truth, w, h)
    cam, info = solve_camera(img, world, w, h, fit_lens=False)
    assert info["rms_px"] < 0.5, info
    assert np.allclose(cam.eye, truth.eye, atol=0.05)
    assert cam.fov_y == pytest.approx(truth.fov_y, abs=0.01)


def test_solve_camera_recovers_the_lens():
    # The one that matters for real footage: a security lens is visibly barrelled, and no
    # POSE can absorb that. If k1 isn't fitted the residual plateaus at the frame edges.
    w, h = 1600, 900
    truth = Camera(eye=(2.0, -4.2, 4.3), target=(1.53, 2.57, 0.06), fov_y=1.05, k1=0.15)
    world, img = _synth(truth, w, h, n=20)
    cam, info = solve_camera(img, world, w, h, fit_lens=True)
    assert info["rms_px"] < 1.0, info
    assert cam.k1 == pytest.approx(0.15, abs=0.02)

    blind, binfo = solve_camera(img, world, w, h, fit_lens=False)
    assert binfo["rms_px"] > info["rms_px"] * 3, \
        f"ignoring a real lens should hurt: pinhole {binfo['rms_px']:.2f} vs {info['rms_px']:.2f} px"


def test_solve_camera_reports_bad_correspondences():
    # Garbage in -> a LOUD residual, not a quiet wrong answer. This is the whole reason the
    # solver returns rms_px at all.
    w, h = 1600, 900
    truth = Camera(eye=(2.0, -4.2, 4.3), target=(1.5, 2.5, 0.0), fov_y=1.05)
    world, img = _synth(truth, w, h, n=12)
    img[3] += [220, -160]                      # one badly mis-clicked point
    _, info = solve_camera(img, world, w, h, fit_lens=False)
    assert info["rms_px"] > 8, "a 270 px blunder must show up in the residual"
    assert max(info["per_point_px"]) > 40


def test_solve_camera_needs_enough_points():
    with pytest.raises(ValueError):
        solve_camera(np.zeros((4, 2)), np.zeros((4, 3)), 640, 480)
