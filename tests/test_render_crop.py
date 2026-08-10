"""A crop must be the same pixels as that region of the full frame — bit for bit.

The point of `--crop` is iteration cost: reverse-modelling a part means rendering one frame
a hundred times to look at one corner, and paying for the sky each time is most of the wall
clock. That is only worth anything if the crop is trustworthy, which means two things and
not one:

* the RAYS must be the rays the full frame would have cast for those pixels, so a crop is a
  sub-image and not a zoom;
* the per-sample RNG must be seeded from the FULL-FRAME pixel, so the crop carries the same
  noise as well as the same geometry.

Get the second wrong and a crop still looks right on its own and cannot be diffed against
anything, which is a subtle enough trap to be worth a test.
"""
from __future__ import annotations

import os
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import pytest

from mirage.capture import default_render


def _bin():
    try:
        return default_render()
    except Exception as exc:                      # remote-only checkout, or not built
        pytest.skip(f"no local renderer: {exc}")


def _read_ppm(path):
    with open(path, "rb") as f:
        assert f.readline().strip() == b"P6"
        line = f.readline()
        while line.startswith(b"#"):
            line = f.readline()
        w, h = (int(v) for v in line.split())
        f.readline()
        return np.frombuffer(f.read(), np.uint8).reshape(h, w, 3)


def test_crop_is_bit_identical_to_that_region_of_the_full_frame():
    exe = _bin()
    w, h = 320, 240
    x, y, cw, ch = 96, 64, 128, 96
    with tempfile.TemporaryDirectory() as td:
        full = Path(td) / "full.ppm"
        crop = Path(td) / "crop.ppm"
        common = [str(exe), "--w", str(w), "--h", str(h), "--spp", "8", "--threads", "4"]
        subprocess.run(common + ["--out", str(full)], check=True,
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        subprocess.run(common + ["--crop", str(x), str(y), str(cw), str(ch),
                                 "--out", str(crop)], check=True,
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        a = _read_ppm(full)[y:y + ch, x:x + cw]
        b = _read_ppm(crop)
    if b.shape == (h, w, 3):
        # The binary ignored --crop, which means it predates it. That is the stale-local-
        # build trap this repo hits often enough to be worth naming rather than failing on:
        # rebuild `core/build` or run on the box, where every trip rebuilds from source.
        pytest.skip("this mirage_render has no --crop; rebuild core/build")
    assert b.shape == (ch, cw, 3)
    assert np.array_equal(a, b), (
        "a crop differs from the same region of the full frame; if the geometry lines up "
        "but the pixels do not, the per-sample RNG is being seeded from the crop-local "
        "pixel instead of the full-frame one")
