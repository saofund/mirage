"""Case 23 — scale: a million-triangle terrain in one op, and where the ceilings are.

Two different limits, measured (see docs/scene-scaling.md):

* **Objects** — the legible ``place`` composition is O(N^2) (each op disjoint-unions onto the
  running mesh by rebuilding it), so *many separate objects* tops out around 1-2k before the
  build gets slow. That's the composition seam, not the renderer.
* **Faces** — a single ``mesh`` op sidesteps that: build is O(faces) once, and the tracer's
  median-split BVH is O(log n) per ray, so it renders **millions** of triangles. Measured on
  the 152-core box: 7.2M tris in ~42s (24spp, 640x400), ~1 GB RAM per million.

So this case builds a big displaced-noise terrain as ONE mesh op (height-coloured per face)
and path-traces it under a low sun. ``N`` is the grid resolution; ``2*(N-1)^2`` triangles.

    uv run python examples/cases/23_terrain.py            # N=800 -> ~1.28M tris -> docs/gallery
    uv run python examples/cases/23_terrain.py --n 1200   # ~2.9M tris

Needs mirage_render (--sun-dir / --denoise) and Pillow. Big N is happiest on a many-core box.
"""
import sys
import json
import math
import time
import subprocess
from pathlib import Path

from mirage.meshlang import MeshProgram
from mirage.capture import default_render

ROOT = Path(__file__).resolve().parents[2]
RENDER = default_render()
OUT = Path(__file__).resolve().parent / "outputs" / "23_terrain"
GALLERY = ROOT / "docs" / "gallery"
SIZE = 14.0


def _h(i, j):
    n = (i * 374761393 + j * 668265263) & 0xffffffff
    n = ((n ^ (n >> 13)) * 1274126177) & 0xffffffff
    return (n & 0xffff) / 0xffff


def _vnoise(x, y):
    xi, yi = math.floor(x), math.floor(y)
    xf, yf = x - xi, y - yi
    u = xf * xf * (3 - 2 * xf); v = yf * yf * (3 - 2 * yf)
    a, b = _h(xi, yi), _h(xi + 1, yi)
    c, d = _h(xi, yi + 1), _h(xi + 1, yi + 1)
    return (a * (1 - u) + b * u) * (1 - v) + (c * (1 - u) + d * u) * v


def _height(x, y):
    h, amp, freq = 0.0, 1.0, 0.28
    for _ in range(6):                                   # fractal Brownian motion (6 octaves)
        h += amp * _vnoise(x * freq + 5, y * freq + 9)
        amp *= 0.5; freq *= 2.03
    ridge = 1.0 - abs(_vnoise(x * 0.5, y * 0.5) - 0.5) * 2.0
    return (h - 0.95) * 3.6 + ridge * 2.2 - 1.2


ROCK, GRASS = [0.42, 0.36, 0.30], [0.33, 0.40, 0.24]
SNOW, SAND = [0.92, 0.93, 0.96], [0.62, 0.57, 0.44]


def _mat(z):
    c = SNOW if z > 3.4 else ROCK if z > 1.2 else GRASS if z > -0.2 else SAND
    return {"color": c, "metallic": 0.0, "roughness": 0.85}


def terrain(n=800):
    """An n x n displaced grid as ONE mesh op — 2*(n-1)^2 triangles, coloured by height."""
    verts, zs = [], []
    for j in range(n):
        for i in range(n):
            x = (i / (n - 1) * 2 - 1) * SIZE
            y = (j / (n - 1) * 2 - 1) * SIZE
            z = _height(x, y)
            verts.append([round(x, 4), round(y, 4), round(z, 4)]); zs.append(z)
    faces, fmats = [], []
    for j in range(n - 1):
        for i in range(n - 1):
            a = j * n + i; d = a + n + 1
            faces.append([a, a + 1, d]); faces.append([a, d, a + n])
            m = _mat((zs[a] + zs[d]) * 0.5)
            fmats.append(m); fmats.append(m)
    return MeshProgram().mesh(verts, faces, face_materials=fmats)


def trace(prog, png, w=1400, h=820, spp=200, denoise=5, threads=16):
    OUT.mkdir(parents=True, exist_ok=True)
    jp = OUT / (png.stem + ".json"); jp.write_text(prog.to_json(indent=None))
    ppm = OUT / (png.stem + ".ppm")
    subprocess.run([str(RENDER), "--oplog", str(jp), "--out", str(ppm), "--w", str(w), "--h", str(h),
                    "--spp", str(spp), "--bounce", "7", "--threads", str(threads), "--denoise", str(denoise),
                    "--sun", "1.3", "--env", "1.0", "--exposure", "1.0", "--sun-dir", "0.72", "0.4", "0.3",
                    "--cam-eye", "18", "-22", "12.5", "--cam-target", "0", "0", "0.2", "--cam-fov", "0.70"],
                   check=True)
    from PIL import Image
    png.parent.mkdir(parents=True, exist_ok=True)
    Image.open(ppm).save(png)


def main():
    n = 800
    for i, a in enumerate(sys.argv):
        if a == "--n" and i + 1 < len(sys.argv):
            n = int(sys.argv[i + 1])
    t0 = time.perf_counter()
    p = terrain(n)
    print(f"  terrain N={n}: {2 * (n - 1) ** 2} triangles, generated in {time.perf_counter()-t0:.1f}s")
    trace(p, GALLERY / "terrain.png", threads=16)
    print(f"  wrote {GALLERY / 'terrain.png'}")


if __name__ == "__main__":
    main()
