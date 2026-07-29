"""Audit and FIT the forecourt's layout against the photograph. No render involved.

    uv run python -m forecourt.place            # draw the overlay, report per-object chamfer
    uv run python -m forecourt.place --fit van suv hump   # search x/y/yaw for those objects

The overlay is the thing to look at after moving anything: every object's ground footprint
and bounding box drawn on the reference frame, in milliseconds, with no lighting or
materials in the way to argue about. The fit is the same question asked as a search.

The output embeds the reference photograph, so like the side-by-side it stays in outputs/
and is never published.
"""
import sys
from pathlib import Path

import numpy as np

from mirage.layout import Camera, fit_ground, overlay, reference_field, score, silhouette
from mirage.meshlang import MeshProgram

from . import parts as P

CASE = Path(__file__).resolve().parents[1] / "26_forecourt.py"
OUT = Path(__file__).resolve().parents[1] / "outputs" / "26_forecourt"
W, H = 1600, 900


def _case():
    """Load the case module without running it (it owns the camera and the layout)."""
    import importlib.util
    spec = importlib.util.spec_from_file_location("case26", CASE)
    m = importlib.util.module_from_spec(spec)
    sys.modules["case26"] = m
    spec.loader.exec_module(m)
    return m


def _mesh(prog):
    m = prog.build()
    verts = np.array([v.co for v in m.verts], float)
    idx = {v.id: i for i, v in enumerate(m.verts)}
    faces = [[idx[lp.vert.id] for lp in m.face_loops(f)] for f in m.faces]
    return verts, faces


# The placeable objects, each as a builder plus where the layout currently puts it. Kept
# here rather than parsed out of the case so that a fit reports an offset the case author
# applies by hand — an optimiser that edits the scene behind you is not an audit.
def ground_rects(c):
    """The painted bays as flat world rectangles — the scene's dominant graphic, and the one
    thing whose placement is fully determined by the camera solve, so it is the check that
    tells you whether everything ELSE is drifting or the camera is."""
    LW = 0.11
    out = []
    for name, (x0, x1, y0, y1) in [("bay_blue", (0, 3.47, 0, 6)),
                                   ("bay_r", (3.62, 7.5, -3.4, 6.0)),
                                   ("bay_top", (0, 3.47, 6.2, 9.7))]:
        q = np.array([[x0 - LW, y0 - LW, 0.0], [x1 + LW, y0 - LW, 0.0],
                      [x1 + LW, y1 + LW, 0.0], [x0 - LW, y1 + LW, 0.0]])
        out.append((name, q))
    return out


def placements(c):
    """Straight from the case. Never restate a layout in the tool that audits it."""
    return c.PLACEMENTS()


_CACHE = {}


def world(build, at, yaw, dx=0.0, dy=0.0, dyaw=0.0, key=None):
    """The object's world geometry at an offset from where the layout puts it.

    The part is built ONCE and its vertices are then transformed in numpy. Rebuilding a
    5,700-face island through the op-log for each of 375 search trials is ten minutes of
    kernel work to answer a question about two translations and an angle, and a fit that
    takes ten minutes is a fit nobody runs."""
    if key not in _CACHE:
        _CACHE[key] = _mesh(build())
    v0, faces = _CACHE[key]
    a = np.radians(yaw + dyaw)
    ca, sa = np.cos(a), np.sin(a)
    v = np.empty_like(v0)
    v[:, 0] = v0[:, 0] * ca - v0[:, 1] * sa + at[0] + dx
    v[:, 1] = v0[:, 0] * sa + v0[:, 1] * ca + at[1] + dy
    v[:, 2] = v0[:, 2] + at[2]
    return v, faces


def main():
    c = _case()
    cam = Camera(c.CAM_EYE, c.CAM_TGT, fov_y=c.CAM_FOV)
    if not c.REF.exists():
        raise SystemExit(f"no reference at {c.REF} — set MIRAGE_REF")
    from PIL import Image
    ref = np.asarray(Image.open(c.REF).convert("RGB").resize((W, H), Image.LANCZOS), float) / 255
    field = reference_field(ref)
    items = placements(c)

    want = [a for a in sys.argv[1:] if not a.startswith("-")]
    OUT.mkdir(parents=True, exist_ok=True)

    if "--fit" in sys.argv:
        for name in (want or list(items)):
            build, at, yaw = items[name]
            print(f"{name}:")
            best, s, s0 = fit_ground(
                lambda a, b, g, _b=build, _at=at, _y=yaw, _k=name: world(_b, _at, _y, a, b, g, _k),
                cam, field, W, H, log=lambda t: print(t))
            gain = s0 - s
            print(f"  {s0:6.2f} -> {s:6.2f} px  ({gain:+.2f})   "
                  f"at=[{at[0] + best[0]:.2f}, {at[1] + best[1]:.2f}]  yaw={yaw + best[2]:.1f}"
                  + ("   <- apply this" if gain > 0.4 else "   (no better than where it is)"))
        return

    draw, rows = list(ground_rects(c)), []
    for name, (build, at, yaw) in items.items():
        v, f = world(build, at, yaw, key=name)
        draw.append((name, v))
        s, n = score(field, silhouette(v, f, cam, W, H))
        rows.append((s, n, name))
    p = overlay(cam, draw, ref, OUT / "layout.png", W, H)
    print(f"{'object':<10}{'chamfer':>9}{'outline px':>12}")
    for s, n, name in sorted(rows, reverse=True):
        print(f"{name:<10}{s:9.2f}{n:12,d}")
    print("wrote", p)


if __name__ == "__main__":
    main()
