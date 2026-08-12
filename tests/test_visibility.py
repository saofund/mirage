"""The face-id AOV and the occlusion diagnostics built on it.

The end-to-end case is the one that motivated the module: put a disc in front of a box,
render, and check the tools say out loud that the box is hidden -- because every score
this repo computes reads the picture, and a hidden part and a wrong part look the same to
all of them.
"""

import os
import shutil
import subprocess

import numpy as np
import pytest

from mirage.meshlang import MeshProgram
from mirage.visibility import (face_ids, face_visibility, hidden_faces, occluders,
                               occlusion_report, read_pfm)

RENDER = shutil.which("mirage_render") or "core/build/Release/mirage_render"
needs_render = pytest.mark.skipif(not os.path.exists(RENDER),
                                  reason="mirage_render not built")


def _render(prog, out, extra=(), w=180, h=140):
    log = str(out) + ".json"
    with open(log, "w") as fh:
        fh.write(prog.to_json())
    cmd = [RENDER, "--oplog", log, "--out", str(out) + ".ppm", "--w", str(w), "--h", str(h),
           "--spp", "1", "--no-ground",
           "--cam-eye", "0", "0", "3.0", "--cam-target", "0", "0", "0",
           "--cam-up", "0", "1", "0", "--cam-fov", "0.7", *extra]
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL)


def test_read_pfm_roundtrip(tmp_path):
    p = tmp_path / "one.pfm"
    with open(p, "wb") as fh:
        fh.write(b"Pf\n2 2\n-1.0\n")
        # PFM is bottom-up, so the first row written is the LAST row of the image
        fh.write(np.array([3, 4, 1, 2], dtype="<f4").tobytes())
    assert read_pfm(p).tolist() == [[1.0, 2.0], [3.0, 4.0]]


def test_face_visibility_ignores_background():
    fids = np.array([[-1, -1, 7], [7, 7, 3]])
    assert face_visibility(fids) == {3: 1, 7: 3}
    assert face_visibility(np.full((4, 4), -1)) == {}


def test_hidden_faces_names_what_never_rendered():
    fids = np.array([[5, 5], [-1, 2]])
    assert hidden_faces(fids, expected=[2, 5, 9, 11]) == [9, 11]
    assert hidden_faces(fids, expected=[2, 5]) == []


def test_occluders_only_counts_what_is_in_front():
    fids = np.array([[1, 2], [3, 4]])
    depth = np.array([[0.5, 2.0], [0.9, 0.0]])
    region = np.ones((2, 2), bool)
    # faces 1 and 3 are nearer than 1.0; face 2 is behind it and face 4 was never hit
    assert occluders(fids, depth, region, ahead_of=1.0) == {1: 1, 3: 1}


def test_report_flags_a_fully_hidden_edit():
    before = np.zeros((8, 8), int)
    after = np.zeros((8, 8), int)
    rep = occlusion_report(before, after, changed_faces=[10, 11, 12])
    assert rep.verdict == "hidden"
    assert rep.hidden == [10, 11, 12]


def test_report_flags_a_big_edit_that_moved_nothing():
    before = np.zeros((100, 100), int)
    after = before.copy()
    after[0, 0] = 5                                    # one pixel out of ten thousand
    changed = list(range(20))
    changed[0] = 5                                     # one of them is visible
    rep = occlusion_report(before, after, changed_faces=changed)
    assert rep.verdict in ("mostly hidden", "suspect")
    assert rep.pixels_changed == pytest.approx(1e-4)


def test_report_is_quiet_when_the_edit_landed():
    before = np.zeros((40, 40), int)
    after = before.copy()
    after[5:25, 5:25] = 7
    rep = occlusion_report(before, after, changed_faces=[7])
    assert rep.verdict == "ok"
    assert rep.pixels_changed > 0.2


def test_report_rejects_mismatched_frames():
    with pytest.raises(ValueError):
        occlusion_report(np.zeros((4, 4), int), np.zeros((5, 5), int))


@needs_render
def test_face_id_aov_matches_the_mesh(tmp_path):
    prog = MeshProgram().cube(size=1.0)
    out = tmp_path / "cube"
    _render(prog, out, ["--face-ids", str(out) + ".fids.pfm"])
    fids = face_ids(str(out) + ".fids.pfm")
    seen = face_visibility(fids)
    ids = {f.id for f in prog.build().faces}
    assert seen, "the cube filled no pixels at all"
    assert set(seen) <= ids, "the AOV named a face the mesh does not have"
    # A cube square to the camera shows exactly one face.
    assert len(seen) == 1


@needs_render
def test_a_disc_in_front_makes_the_box_invisible(tmp_path):
    """The failure this module exists for, end to end.

    A box, then the same box with a plate parked in front of it. Every pixel-based score
    would call the second one 'barely changed'; the point is that the tools call it
    hidden instead.
    """
    box = MeshProgram().cube(size=1.0, mark="box")
    box_ids = sorted(f.id for f in box.build().faces)

    covered = (MeshProgram().cube(size=1.0, mark="box")
               .place(obj=MeshProgram().cube(size=1.0, mark="plate")
                      .scale({"by": "all"}, [2.4, 2.4, 0.05]),
                      at=(0.0, 0.0, 1.1)))

    a, b = tmp_path / "open", tmp_path / "covered"
    _render(box, a, ["--face-ids", str(a) + ".fids.pfm"])
    _render(covered, b, ["--face-ids", str(b) + ".fids.pfm"])
    fa, fb = face_ids(str(a) + ".fids.pfm"), face_ids(str(b) + ".fids.pfm")

    assert set(face_visibility(fa)) & set(box_ids), "the bare box was not visible"
    rep = occlusion_report(fa, fb, changed_faces=box_ids)
    assert rep.verdict in ("hidden", "mostly hidden"), str(rep)
    assert len(rep.hidden) >= len(box_ids) - 1


@needs_render
def test_aov_only_matches_the_traced_geometry(tmp_path):
    """The fast preview must describe the SAME geometry as a full render.

    `--aov-only` skips light transport, so its picture is different by design. Its AOVs
    are not allowed to be: they come from the same primary ray, and a preview that
    disagreed with the render about which face is where would be worse than no preview.
    """
    prog = MeshProgram().cube(size=1.0).subdivide(levels=1)
    slow, fast = tmp_path / "slow", tmp_path / "fast"
    _render(prog, slow, ["--face-ids", str(slow) + ".pfm", "--depth", str(slow) + ".d"])
    _render(prog, fast, ["--face-ids", str(fast) + ".pfm", "--depth", str(fast) + ".d",
                         "--aov-only"])
    a, b = face_ids(str(slow) + ".pfm"), face_ids(str(fast) + ".pfm")
    assert np.array_equal(a, b), "the preview and the render disagree about the faces"
    da, db = read_pfm(str(slow) + ".d"), read_pfm(str(fast) + ".d")
    assert np.allclose(da, db, atol=1e-9)
