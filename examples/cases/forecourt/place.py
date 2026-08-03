"""Audit and FIT the forecourt's layout against the photograph. No render involved.

    uv run python -m forecourt.place            # draw the overlay, report per-object chamfer
    uv run python -m forecourt.place --fit van suv hump   # search x/y/yaw for those objects
    uv run python -m forecourt.place --contacts van      # fit x/y/yaw/scale to ground contacts
    uv run python -m forecourt.place --measure           # measure the photograph itself

The overlay is the thing to look at after moving anything: every object's ground footprint
and bounding box drawn on the reference frame, in milliseconds, with no lighting or
materials in the way to argue about. The fit is the same question asked as a search.

The output embeds the reference photograph, so like the side-by-side it stays in outputs/
and is never published.
"""
import sys
from pathlib import Path

import numpy as np

from mirage.layout import (Camera, fit_contacts, fit_ground, ground_line, overlay,
                           reference_field, score, silhouette)
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

    if "--measure" in sys.argv:
        from PIL import Image as _I
        ren = np.asarray(_I.open(OUT / "hero.png").convert("RGB"), float) / 255             if (OUT / "hero.png").exists() else None
        print("== painted regions ==");    measure_paint(c, cam, ref)
        print("== the building line =="); measure_yard_line(c, cam, ref)
        if ren is not None:
            print("== ground luminance, photo vs render =="); measure_profile(c, cam, ref, ren)
        return

    if "--contacts" in sys.argv:
        for name in (want or list(CONTACTS)):
            fit_by_contacts(c, cam, items, name)
        return

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




# --------------------------------------------------------------------------- #
# fitting to ground contacts
# --------------------------------------------------------------------------- #
# What the photograph says about a vehicle, read off it once and written down: the columns
# its body spans, and the row where the dark band under it meets the ground, column by
# column. Nothing here is a guess about height, which is the whole point -- tyres touch z=0
# and nothing else in the picture is that certain. `layout.fit_contacts` does the rest.
#
# The van is why this exists. It sat 3.5 m too close to the camera through a dozen rounds of
# material and tone work while the scorecard called its chamfer the best of any vehicle in
# the scene, because it stands against a wall of roller-shutter slats and an object in
# clutter is near SOMETHING wherever you put it.
CONTACTS = {
    "van": dict(
        columns=(762.0, 1120.0),
        rows={760: 109, 780: 110, 800: 109, 820: 109, 840: 107, 860: 106, 880: 105,
              920: 101, 940: 100, 960: 99, 980: 98, 1000: 97, 1020: 97, 1060: 95, 1080: 95,
              1100: 94},
        top_row=0.0,                 # the frame cuts its roof off, which bounds its height
    ),
}


def fit_by_contacts(c, cam, items, name):
    spec = CONTACTS.get(name)
    if spec is None:
        print(f"{name}: no measured contacts -- add them to CONTACTS")
        return
    build, at, yaw = items[name]
    verts, _ = _mesh(build())
    line = ground_line(spec["rows"])
    x, y, g, sc, rms = fit_contacts(verts, cam, W, H, columns=spec["columns"], line=line,
                                    start=(at[0], at[1], yaw, 1.0),
                                    top_row=spec.get("top_row"))
    print(f"{name}: ground line row = {line[0]:+.5f} * col {line[1]:+.2f}")
    print(f"  was  at=[{at[0]:.2f}, {at[1]:.2f}]  yaw={yaw:.2f}")
    print(f"  fit  at=[{x:.2f}, {y:.2f}]  yaw={g:.2f}  scale={sc:.4f}   rms={rms:.2f}px")
    if abs(sc - 1.0) > 0.04:
        print(f"  NOTE scale {sc:.3f} -- the PART is the wrong size, not the placement; "
              f"rebuild it and re-fit rather than scaling it here")


# --------------------------------------------------------------------------- #
# measuring the photograph
# --------------------------------------------------------------------------- #
# Everything in this scene that is RIGHT is right because it was measured here rather than
# adjusted until it looked better. These three ran as throwaway scripts and found, in one
# afternoon: a painted bay in the wrong lane, a drain three metres out into the road, an
# 8.5-degree error in the whole yard's direction, and a van pushed four metres too far away
# by unprojecting its sill as though it lay on the ground. That is too good a hit rate to
# leave in a shell history.
def measure_paint(c, cam, ref):
    """Segment the painted regions by colour and unproject their corners. Answers 'where is
    the paint', which is the question the bays, the strips and the lines all turned on."""
    R, B, L = ref[..., 0], ref[..., 2], ref.mean(-1)
    for name, m in (("ORANGE", (R - B > 0.10) & (R > 0.18) & (L > 0.10)),
                    ("BLUE", (B - R > 0.035) & (L > 0.10) & (L < 0.62))):
        for i, pix in enumerate(_blobs(m)[:3]):
            ys, xs = pix[:, 0], pix[:, 1]
            s, d = xs + ys, xs - ys
            print(f"  {name} blob {i}  {len(pix):7,d} px")
            for px, py in [(xs[s.argmin()], ys[s.argmin()]), (xs[d.argmax()], ys[d.argmax()]),
                           (xs[s.argmax()], ys[s.argmax()]), (xs[d.argmin()], ys[d.argmin()])]:
                w = _gp(cam, px, py)
                print(f"      px({px:4d},{py:4d}) -> world ({w[0]:7.2f}, {w[1]:7.2f})")


def measure_yard_line(c, cam, ref):
    """Fit the BUILDING line off the bollard row. Two directions run through this scene —
    the forecourt's paint and the shop's frontage — and they are not parallel."""
    R, B, L = ref[..., 0], ref[..., 2], ref.mean(-1)
    m = (B - R > 0.055) & (L < 0.42) & (B > 0.13)
    band = np.zeros_like(m); band[30:130, 200:1600] = True
    ys, xs = np.nonzero(m & band)
    order = np.argsort(xs); xs, ys = xs[order], ys[order]
    groups, cur = [], [0]
    for i in range(1, len(xs)):
        if xs[i] - xs[i - 1] > 12:
            groups.append(cur); cur = []
        cur.append(i)
    groups.append(cur)
    pts = []
    for g in groups:
        if len(g) < 40:
            continue
        w = _gp(cam, float(xs[g].mean()), float(ys[g].max()))   # lowest pixel = ground contact
        if 8 < w[1] < 45:
            pts.append(w)
            print(f"  bollard -> world ({w[0]:6.2f}, {w[1]:6.2f})")
    if len(pts) >= 3:
        P = np.array(pts); A = np.polyfit(P[:, 0], P[:, 1], 1)
        print(f"  fit y = {A[0]:.4f}x + {A[1]:.2f}   YAW {np.degrees(np.arctan(A[0])):+.2f} deg "
              f"(case YARD_ANG = {c.YARD_ANG:+.1f})")


def measure_profile(c, cam, ref, ren, x=6.0):
    """Ground luminance into the distance, photo against render. Catches whole materials
    that are the wrong brightness — a tarmac band that is not there, a lighter pour that is.

    KNOW WHERE TO STOP. Past about y=13 the reference sits at 0.68..0.97 where the render
    sits at 0.27..0.44, and that gap is NOT a material. It is four metres from a five-metre
    wall: an overcast dome is most of this scene's light, and a wall that close takes fifty
    degrees of it away. The render is doing the right thing. The reference is a security
    camera with wide-dynamic-range processing, which lifts exactly those shadows on purpose,
    and matching it there would mean lying about the geometry to imitate somebody's tone
    curve. Read this profile for the NEAR ground, where both images are describing the same
    physics, and read the far end as a reminder that the reference is not a light meter."""
    from mirage.solve import project
    ys = np.arange(4.0, 18.1, 1.0)
    pts = project(cam, np.array([[x, y, 0.0] for y in ys]), W, H)
    tot = n = 0
    for y, (px, py) in zip(ys, pts):
        if not np.isfinite([px, py]).all():
            continue
        a, b = int(round(px)), int(round(py))
        if not (0 <= a < W and 0 <= b < H):
            continue
        u, v = float(ref[b, a].mean()), float(ren[b, a].mean())
        tot += abs(u - v); n += 1
        print(f"  y={y:5.1f}  photo {u:.3f}   render {v:.3f}   {v - u:+.3f}")
    print(f"  mean |delta| = {tot / max(n, 1):.3f}")


def _gp(cam, px, py, z=0.0):
    from mirage.solve import ground_point
    return np.asarray(ground_point(cam, [float(px), float(py)], W, H, z=z), float).ravel()[:2]


def _blobs(mask, min_px=6000):
    seen = np.zeros(mask.shape, bool); out = []
    for y0, x0 in zip(*np.nonzero(mask)):
        if seen[y0, x0]:
            continue
        st = [(y0, x0)]; seen[y0, x0] = True; pix = []
        while st:
            y, x = st.pop(); pix.append((y, x))
            for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                ny, nx = y + dy, x + dx
                if (0 <= ny < mask.shape[0] and 0 <= nx < mask.shape[1]
                        and mask[ny, nx] and not seen[ny, nx]):
                    seen[ny, nx] = True; st.append((ny, nx))
        if len(pix) >= min_px:
            out.append(np.array(pix))
    return sorted(out, key=len, reverse=True)

if __name__ == "__main__":
    main()
