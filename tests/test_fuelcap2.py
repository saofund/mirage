"""Exact geometric checks for the photograph-matched Polo filler region."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "examples" / "cases"))

from fuelcap2 import polo  # noqa: E402


def test_polo_region_builds_and_replays_identically():
    a = polo.build().build()
    b = polo.build().build()
    a.validate()
    b.validate()
    assert [v.co for v in a.verts] == [v.co for v in b.verts]
    fa = [[loop.vert.id for loop in a.face_loops(face)] for face in a.faces]
    fb = [[loop.vert.id for loop in b.face_loops(face)] for face in b.faces]
    assert fa == fb


def test_polo_scene_retains_the_measured_circular_aperture():
    # The top edge of the liner is authored in the panel plane. Check that the radius is
    # circular before the camera foreshortens it into the photographed vertical ellipse.
    mesh = polo.P.liner([1.0] * 96, polo.OPEN_R, depth=polo.POCKET_DEPTH,
                        flange_w=13 * polo.MM, flange_z=3 * polo.MM,
                        steps=96, material=polo.LINER).build()
    co = np.asarray([v.co for v in mesh.verts], float)
    rim = co[co[:, 2] > -4.0 * polo.MM]
    radii = np.hypot(rim[:, 0], rim[:, 1])
    assert radii.max() - radii.min() < 15 * polo.MM
    assert abs(radii.max() - polo.OPEN_R) < 2 * polo.MM
