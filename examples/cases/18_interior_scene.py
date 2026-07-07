"""Case 18 — a whole INTERIOR, every object native, composed by the engine.

Not a skyline of extruded boxes, and not Python glue: a furnished room where each
thing is modelled from Mirage's **own operators** — the lathe (``profile`` +
``spin``) turns the vase and the lampshade, ``bevel`` softens the sofa and table,
``array`` stacks the shelves and books, ``boolean`` cuts the window, ``mirror``
pairs the sofa arms — and the room is assembled by the **native ``place`` op**:
the op-log is a legible list of ``place`` ops, each carrying its object's operators
and a transform, so the *engine* composes the scene (not a merge in Python). That
one op-log builds byte-identically in the Python kernel and the C++ core, and the
path tracer (``mirage_render``) shoots it from a camera standing *inside* the room.

    uv run python examples/cases/18_interior_scene.py           # preview to outputs/
    uv run python examples/cases/18_interior_scene.py --hero    # docs/gallery/interior.png
    uv run python examples/cases/18_interior_scene.py --film    # the making-of (mp4 + gif)
    uv run python examples/cases/18_interior_scene.py --oplog   # write the scene op-log JSON

The still needs ``mirage_render.exe`` (``cmake --build core/build --config
Release``; ``--cam-*`` / ``--threads``) and Pillow. ``--film`` needs
``mirage_viewer.exe`` (``-DMIRAGE_BUILD_VIEWER=ON``) and ffmpeg.
"""
import sys
import time
import math
import subprocess
from pathlib import Path

from mirage.meshlang import MeshProgram, Sel

ROOT = Path(__file__).resolve().parents[2]
RENDER = ROOT / "core" / "build" / "Release" / "mirage_render.exe"
OUT = Path(__file__).resolve().parent / "outputs" / "18_interior_scene"


# ---- materials (physically-based; the tracer reads color/metallic/roughness) ---- #
def mat(color, metallic=0.0, roughness=0.5):
    return {"color": list(color), "metallic": metallic, "roughness": roughness}

FLOOR   = mat((0.47, 0.31, 0.17), 0.0, 0.35)   # warm oak boards
RUG     = mat((0.56, 0.32, 0.26), 0.0, 0.92)   # terracotta rug
PLASTER = mat((0.87, 0.84, 0.79), 0.0, 0.95)   # warm white wall
WALNUT  = mat((0.36, 0.22, 0.12), 0.0, 0.42)   # dark wood
OAK     = mat((0.62, 0.45, 0.26), 0.0, 0.48)   # light wood
SOFA    = mat((0.30, 0.39, 0.49), 0.0, 0.85)   # blue-grey upholstery
CHAIR   = mat((0.68, 0.35, 0.24), 0.0, 0.85)   # rust upholstery
CUSHION = mat((0.78, 0.70, 0.55), 0.0, 0.80)   # cream cushions
CERAMIC = mat((0.92, 0.89, 0.84), 0.0, 0.12)   # glossy off-white
TEAL    = mat((0.28, 0.47, 0.52), 0.0, 0.14)   # glazed teal vase
BRASS   = mat((0.82, 0.61, 0.29), 1.0, 0.28)
STEEL   = mat((0.70, 0.71, 0.74), 1.0, 0.25)
SHADE   = mat((0.98, 0.92, 0.78), 0.0, 0.60)   # bright warm shade (reads as lit)
LEAF    = mat((0.26, 0.43, 0.22), 0.0, 0.60)
POT     = mat((0.58, 0.31, 0.20), 0.0, 0.70)   # terracotta
CANVAS  = mat((0.40, 0.50, 0.58), 0.0, 0.70)   # a soft painting
BOOKS   = [mat((0.62, 0.24, 0.20), 0.0, 0.7), mat((0.24, 0.36, 0.46), 0.0, 0.7),
           mat((0.30, 0.42, 0.28), 0.0, 0.7), mat((0.74, 0.60, 0.30), 0.0, 0.7),
           mat((0.46, 0.30, 0.44), 0.0, 0.7), mat((0.80, 0.72, 0.58), 0.0, 0.7)]
FRUIT   = [mat((0.78, 0.24, 0.18), 0.0, 0.5), mat((0.86, 0.52, 0.16), 0.0, 0.5),
           mat((0.52, 0.62, 0.20), 0.0, 0.5)]


def rnd(i, a=0.0, b=1.0):
    x = math.sin((i + 1) * 12.9898) * 43758.5453
    return a + (b - a) * (x - math.floor(x))


# ---- geometry helpers --------------------------------------------------------- #
def box(cx, cy, cz, sx, sy, sz):
    """A box centered at (cx,cy,cz) as (verts, faces) — inline geometry for slabs and
    for the local frame of objects built from raw boxes (walls, books, mullions)."""
    hx, hy, hz = sx / 2, sy / 2, sz / 2
    v = [(cx-hx,cy-hy,cz-hz),(cx+hx,cy-hy,cz-hz),(cx+hx,cy+hy,cz-hz),(cx-hx,cy+hy,cz-hz),
         (cx-hx,cy-hy,cz+hz),(cx+hx,cy-hy,cz+hz),(cx+hx,cy+hy,cz+hz),(cx-hx,cy+hy,cz+hz)]
    f = [(0,3,2,1),(4,5,6,7),(0,1,5,4),(1,2,6,5),(2,3,7,6),(3,0,4,7)]  # outward-wound
    return v, f


def rounded(sx, sy, sz, w=0.05):
    """A box with softened edges — the upholstery / table-top primitive (bevel)."""
    return MeshProgram().mesh(*box(0, 0, 0, sx, sy, sz)).bevel(Sel.all(), width=w, depth=w * 0.8)


# ---- object builders (each a MeshProgram in its own local frame) --------------- #
def lathe_vase(angle=360.0):
    """The hero of the room: an open silhouette revolved on the lathe (profile+spin).
    A partial `angle` catches the lathe mid-sweep (used by the making-of)."""
    pts = [(0.145, 0.0), (0.155, 0.02), (0.10, 0.13), (0.085, 0.28),
           (0.135, 0.44), (0.165, 0.56), (0.14, 0.65), (0.095, 0.70)]
    return MeshProgram().profile(pts, plane="xz", closed=False).spin(axis="z", steps=56, angle=angle)


def lathe_shade():
    """A lampshade — a trapezoid profile turned into a truncated cone (the lathe)."""
    return MeshProgram().profile([(0.085, 0.0), (0.185, -0.24)], plane="xz", closed=False).spin(axis="z", steps=44)


def turned_bowl():
    """A shallow fruit bowl — an arc revolved."""
    pts = [(0.0, 0.055), (0.09, 0.012), (0.17, 0.02), (0.215, 0.095)]
    return MeshProgram().profile(pts, plane="xz", closed=False).spin(axis="z", steps=44)


def window_wall(width, thick, height, hole_w, hole_h, hole_cx, hole_cz):
    """The back wall with a real opening punched by a boolean difference."""
    p = MeshProgram().mesh(*box(0, 0, height / 2, width, thick, height))
    return p.boolean("difference", box(hole_cx, 0, hole_cz, hole_w, thick * 3, hole_h))


def framed_picture():
    """A wall picture: a slab whose front face is inset to make a frame border."""
    return (MeshProgram().mesh(*box(0, 0, 0, 0.05, 0.86, 0.62))
            .inset(on=Sel.side("x", 1), thickness=0.07)
            .extrude(on=Sel.last(), distance=-0.015))


# ---- assembly: a legible op-log of `place` ops (the ENGINE composes the scene) -- #
def slab(p, cx, cy, cz, sx, sy, sz, material):
    """Place a box slab (inline geometry) — floor, walls, rug, canvas."""
    v, f = box(cx, cy, cz, sx, sy, sz)
    return p.place(verts=v, faces=f, material=material)


def _shell(p):
    """Floor + two far walls (roofless dollhouse; sun floods the open sides)."""
    slab(p, 0.0, 0.0, -0.06, 4.8, 4.8, 0.12, FLOOR)             # floor (top at z=0)
    slab(p, -2.26, 0.0, 1.20, 0.12, 4.6, 2.40, PLASTER)         # left wall (x = -2.2)
    p.place(obj=window_wall(4.6, 0.12, 2.40, 1.25, 1.05, -0.7, 1.28), at=(0, 2.26, 0), material=PLASTER)
    p.place(obj=MeshProgram().mesh(*box(-0.7, 0, 1.28, 0.03, 0.06, 1.05)).array(count=3, offset=(0.42, 0, 0)),
            at=(0, 2.20, 0), material=OAK)                       # vertical mullions (array)
    p.place(obj=MeshProgram().mesh(*box(-0.7, 0, 1.28, 1.29, 0.06, 0.03)).array(count=2, offset=(0, 0, 0.52)),
            at=(0, 2.20, 0), material=OAK)                       # horizontal mullions (array)


def _table(p):
    """Rug, beveled coffee table + turned legs, and the lathe vase / fruit bowl."""
    slab(p, -0.15, 0.05, 0.012, 2.4, 1.7, 0.024, RUG)
    p.place(obj=rounded(1.5, 0.82, 0.10, w=0.03), at=(-0.15, 0.05, 0.44), material=WALNUT)   # beveled top
    for lx in (-0.63, 0.63):
        for ly in (-0.30, 0.30):
            p.place(obj=MeshProgram().cylinder(sides=16, radius=0.045, height=0.40),
                    at=(-0.15 + lx, 0.05 + ly, 0.20), material=WALNUT)                        # legs
    p.place(obj=lathe_vase(), at=(-0.55, 0.10, 0.49), material=TEAL)                          # the lathe
    p.place(obj=turned_bowl(), at=(0.35, 0.02, 0.49), material=CERAMIC)
    for k, (fx, fy) in enumerate([(0.30, 0.02), (0.40, 0.06), (0.35, -0.05)]):
        p.place(obj=MeshProgram().uv_sphere(segments=18, rings=12, radius=0.052),
                at=(fx, fy, 0.58 + 0.01 * k), material=FRUIT[k % len(FRUIT)])


def _sofa(p):
    """A sofa against the left wall (bevel + a mirrored arm pair)."""
    sx0, sy0 = -1.58, 0.10
    p.place(obj=rounded(0.92, 2.30, 0.34, w=0.05), at=(sx0, sy0, 0.30), material=SOFA)         # seat base
    p.place(obj=rounded(0.28, 2.30, 0.72, w=0.05), at=(sx0 - 0.32, sy0, 0.60), material=SOFA)  # backrest
    arms = MeshProgram().mesh(*box(0, 1.14, 0.44, 0.92, 0.30, 0.52)).mirror("y").bevel(Sel.all(), width=0.05, depth=0.04)
    p.place(obj=arms, at=(sx0, sy0, 0), material=SOFA)                                          # mirrored arm pair
    for cy in (-0.72, 0.02, 0.76):                                                             # seat cushions
        p.place(obj=rounded(0.74, 0.66, 0.20, w=0.06), at=(sx0 + 0.04, sy0 + cy, 0.52), material=CUSHION)
    for k, cy in enumerate((-0.86, 0.86)):                                                     # throw pillows
        p.place(obj=rounded(0.34, 0.34, 0.14, w=0.06), at=(sx0 + 0.16, sy0 + cy, 0.60),
                rotate=(0, 0, 20 - 40 * k), material=BOOKS[k])


def _armchair(p):
    """An armchair on the right, turned to face the table (bevel)."""
    ax, ay, arot = 1.05, -0.70, -118
    p.place(obj=rounded(0.86, 0.86, 0.30, w=0.05), at=(ax, ay, 0.28), rotate=(0, 0, arot), material=CHAIR)
    p.place(obj=rounded(0.24, 0.86, 0.62, w=0.05),
            at=(ax + 0.31 * math.cos(math.radians(arot)), ay + 0.31 * math.sin(math.radians(arot)), 0.56),
            rotate=(0, 0, arot), material=CHAIR)
    p.place(obj=rounded(0.70, 0.70, 0.18, w=0.06), at=(ax, ay, 0.46), rotate=(0, 0, arot), material=CUSHION)


def _bookshelf(p):
    """A bookshelf against the back wall (boolean carcass, array shelves, books)."""
    bx, by = 1.28, 1.98
    p.place(obj=MeshProgram().mesh(*box(0, 0, 0.85, 1.5, 0.34, 1.70)).boolean(
        "difference", box(0, 0.10, 0.86, 1.36, 0.34, 1.52)), at=(bx, by, 0), material=OAK)     # hollow carcass
    p.place(obj=MeshProgram().mesh(*box(0, 0, 0.22, 1.36, 0.30, 0.04)).array(count=4, offset=(0, 0, 0.42)),
            at=(bx, by, 0), material=OAK)                                                       # shelves (array)
    for shelf in range(4):
        z0 = 0.24 + shelf * 0.42
        x = -0.60
        while x < 0.60:
            i = shelf * 20 + int((x + 1) * 13)
            h = 0.24 + 0.06 * rnd(i)
            w = 0.035 + 0.02 * rnd(i + 3)
            tilt = 14 if rnd(i + 7) > 0.86 else 0
            p.place(obj=MeshProgram().mesh(*box(0, 0, 0, w, 0.24, h)),
                    at=(bx + x, by - 0.01, z0 + h / 2), rotate=(0, tilt, 0), material=BOOKS[i % len(BOOKS)])
            x += w + 0.008


def _decor(p):
    """Floor lamp (turned shade), potted plant (uv_sphere), wall art + clock (torus)."""
    lx, ly = -1.75, 1.65                                          # floor lamp
    p.place(obj=MeshProgram().cylinder(sides=28, radius=0.17, height=0.05), at=(lx, ly, 0.025), material=STEEL)
    p.place(obj=MeshProgram().cylinder(sides=16, radius=0.028, height=1.45), at=(lx, ly, 0.75), material=STEEL)
    p.place(obj=lathe_shade(), at=(lx, ly, 1.52), material=SHADE)
    px, py = 1.92, 1.12                                           # potted plant
    p.place(obj=MeshProgram().cylinder(sides=24, radius=0.16, height=0.30), at=(px, py, 0.15), material=POT)
    for k in range(6):
        a = k * 1.05
        p.place(obj=MeshProgram().uv_sphere(segments=16, rings=10, radius=0.13 + 0.03 * rnd(k)),
                at=(px + 0.10 * math.cos(a), py + 0.10 * math.sin(a), 0.40 + 0.09 * rnd(k + 2)), material=LEAF)
    p.place(obj=framed_picture(), at=(-2.18, 0.55, 1.55), material=WALNUT)   # picture on the left wall
    slab(p, -2.15, 0.55, 1.55, 0.02, 0.66, 0.44, CANVAS)
    p.place(obj=MeshProgram().torus(major_segments=36, minor_segments=16, major_radius=0.22, minor_radius=0.025),
            at=(-1.5, 2.13, 1.62), rotate=(90, 0, 0), material=BRASS)         # clock rim (torus)
    p.place(obj=MeshProgram().cylinder(sides=36, radius=0.205, height=0.03),
            at=(-1.5, 2.14, 1.62), rotate=(90, 0, 0), material=CERAMIC)
    for hr, ln in ((90, 0.14), (20, 0.10)):                       # two hands
        p.place(obj=MeshProgram().mesh(*box(0, ln / 2 - 0.01, 0, 0.02, ln, 0.008)),
                at=(-1.5, 2.12, 1.62), rotate=(90, 0, hr), material=WALNUT)


GROUPS = [
    ("room shell  ·  boolean-cut window", _shell),
    ("table + vase  ·  bevel, the lathe", _table),
    ("sofa  ·  bevel + mirror", _sofa),
    ("armchair  ·  bevel", _armchair),
    ("bookshelf  ·  array + books", _bookshelf),
    ("lamp, plant, art  ·  spin, uv_sphere, torus", _decor),
]


def build_room():
    """The whole interior as ONE legible op-log — a list of `place` ops the engine composes."""
    p = MeshProgram()
    for _, fn in GROUPS:
        fn(p)
    return p


def film_stages():
    """Curated making-of: feature the *shaping* of a few hero objects in place — the
    lathe sweeping the vase, `boolean` punching the window, `bevel` rounding the sofa —
    then populate the rest, ending at the full room. Each stage is an independent
    op-log the viewer renders; the sequence tells the build story. Because the scene
    is a legible place-op-log, a hero's partial operator sequence builds a partial
    object *at its final spot*, so the shaping reads in context."""
    stages, caps = [], []
    p = MeshProgram()

    def snap(cap):
        stages.append(MeshProgram(p.ops)); caps.append(cap)

    # 1. shell — floor, left wall, a SOLID back wall (the window is cut next)
    slab(p, 0.0, 0.0, -0.06, 4.8, 4.8, 0.12, FLOOR)
    slab(p, -2.26, 0.0, 1.20, 0.12, 4.6, 2.40, PLASTER)
    p.place(obj=MeshProgram().mesh(*box(0, 0, 1.20, 4.6, 0.12, 2.40)), at=(0, 2.26, 0), material=PLASTER)
    snap("room shell  ·  walls up")

    # 2. the window: a boolean punches the opening, then array mullions
    p.ops.pop()   # drop the solid back wall; re-place it windowed
    p.place(obj=window_wall(4.6, 0.12, 2.40, 1.25, 1.05, -0.7, 1.28), at=(0, 2.26, 0), material=PLASTER)
    p.place(obj=MeshProgram().mesh(*box(-0.7, 0, 1.28, 0.03, 0.06, 1.05)).array(count=3, offset=(0.42, 0, 0)),
            at=(0, 2.20, 0), material=OAK)
    p.place(obj=MeshProgram().mesh(*box(-0.7, 0, 1.28, 1.29, 0.06, 0.03)).array(count=2, offset=(0, 0, 0.52)),
            at=(0, 2.20, 0), material=OAK)
    snap("window  ·  boolean cut + array mullions")

    # 3. coffee table + rug
    slab(p, -0.15, 0.05, 0.012, 2.4, 1.7, 0.024, RUG)
    p.place(obj=rounded(1.5, 0.82, 0.10, w=0.03), at=(-0.15, 0.05, 0.44), material=WALNUT)
    for lx in (-0.63, 0.63):
        for ly in (-0.30, 0.30):
            p.place(obj=MeshProgram().cylinder(sides=16, radius=0.045, height=0.40),
                    at=(-0.15 + lx, 0.05 + ly, 0.20), material=WALNUT)
    snap("coffee table  ·  bevel + turned legs")

    # 4-5. the vase on the lathe: an open profile revolved, sweeping partial -> full
    p.place(obj=lathe_vase(130), at=(-0.55, 0.10, 0.49), material=TEAL)
    snap("vase  ·  the lathe sweeps (profile + spin)")
    p.ops.pop()
    p.place(obj=lathe_vase(360), at=(-0.55, 0.10, 0.49), material=TEAL)
    p.place(obj=turned_bowl(), at=(0.35, 0.02, 0.49), material=CERAMIC)
    for k, (fx, fy) in enumerate([(0.30, 0.02), (0.40, 0.06), (0.35, -0.05)]):
        p.place(obj=MeshProgram().uv_sphere(segments=18, rings=12, radius=0.052),
                at=(fx, fy, 0.58 + 0.01 * k), material=FRUIT[k % len(FRUIT)])
    snap("vase  ·  a full surface of revolution")

    # 6. the sofa, placed whole (it sits against the far-left wall, behind the panel)
    _sofa(p)
    snap("sofa  ·  bevel + mirror")

    # 7-8. the armchair in the FOREGROUND: blocked out sharp, then bevel rounds it
    ax, ay, arot = 1.05, -0.70, -118
    ac, asn = math.cos(math.radians(arot)), math.sin(math.radians(arot))
    p.place(obj=MeshProgram().mesh(*box(0, 0, 0, 0.86, 0.86, 0.30)), at=(ax, ay, 0.28), rotate=(0, 0, arot), material=CHAIR)
    p.place(obj=MeshProgram().mesh(*box(0, 0, 0, 0.24, 0.86, 0.62)),
            at=(ax + 0.31 * ac, ay + 0.31 * asn, 0.56), rotate=(0, 0, arot), material=CHAIR)
    snap("armchair  ·  blocked out (sharp boxes)")
    p.ops.pop(); p.ops.pop()
    _armchair(p)
    snap("armchair  ·  bevel rounds it")

    # 9-10. bookshelf + decor, placed whole
    _bookshelf(p); snap("bookshelf  ·  array shelves + books")
    _decor(p); snap(f"complete  ·  one op-log, {len(p.ops)} place ops")
    return stages, caps


# ---- render (still) ----------------------------------------------------------- #
def trace_prog(prog, png, spp=200, w=1200, h=800, eye=(3.62, -3.46, 2.78),
               target=(-0.12, 0.16, 0.80), fov=0.70, threads=None, **knobs):
    if not RENDER.exists():
        print(f"  ! mirage_render not built — expected {RENDER}\n"
              f"    build it:  cmake --build {ROOT/'core'/'build'} --config Release")
        return False
    OUT.mkdir(parents=True, exist_ok=True)
    jp = OUT / (png.stem + ".json")
    jp.write_text(prog.to_json(indent=None))     # the legible scene op-log the tracer reads
    ppm = OUT / (png.stem + ".ppm")
    cmd = [str(RENDER), "--oplog", str(jp), "--out", str(ppm),
           "--spp", str(spp), "--w", str(w), "--h", str(h), "--bounce", "8",
           "--cam-eye", *map(str, eye), "--cam-target", *map(str, target), "--cam-fov", str(fov)]
    if threads:
        cmd += ["--threads", str(threads)]
    for k, val in knobs.items():
        cmd += [f"--{k}", str(val)]
    t0 = time.perf_counter()
    subprocess.run(cmd, check=True)
    try:
        from PIL import Image
    except ImportError:
        print(f"  wrote {ppm} (install pillow to get a .png)")
        return True
    png.parent.mkdir(parents=True, exist_ok=True)
    Image.open(ppm).save(png)
    print(f"  wrote {png}  [{time.perf_counter()-t0:.1f}s]")
    return True


def render_interior(png, spp=200, w=1200, h=800, threads=None):
    prog = build_room()
    m = prog.build()   # engine-composed mesh (also a parity check against the tracer)
    print(f"  interior: {len(prog.ops)} place ops  {len(m.faces)} faces  "
          f"op-log {len(prog.to_json(indent=None))/1e6:.2f}MB (a legible scene, engine-composed)")
    trace_prog(prog, png, spp=spp, w=w, h=h, threads=threads, sun=1.2, env=1.15, exposure=1.1)


# ---- making-of (the room assembling in the real native viewer) ---------------- #
def film():
    """Film the room building group-by-group in mirage_viewer -> interior_build.mp4/.gif."""
    import os
    from mirage.capture import record_build
    stages, captions = film_stages()
    quick = os.environ.get("ANIM_QUICK") == "1"
    # a moving camera (dolly + orbit + a closing focus-pull) through the open +x,-y
    # corner: establish high & wide, push in to the closest beat as the vase turns on
    # the lathe, ease back and swing to a frontal read, then drive all the way IN to an
    # intimate close-up of the vase / table (aim pulling onto it) so the materials read,
    # and hold there. (t, yaw, pitch, dist, tx, ty, tz); last two keys equal -> static dwell.
    moves = [
        (0.00, 0.66, 0.54, 6.8, -0.10, 0.50, 0.95),   # establishing — high & wide, room centre
        (0.15, 0.82, 0.46, 5.7, -0.10, 0.50, 0.95),   # dolly in as the shell + table go up
        (0.30, 0.96, 0.40, 4.9, -0.22, 0.30, 0.82),   # closest on the vase beat, aim drifting in
        (0.48, 0.86, 0.44, 5.5, -0.10, 0.50, 0.95),   # ease back, recentre on the room
        (0.64, 0.64, 0.46, 5.7, -0.10, 0.50, 0.95),   # swing to a frontal angle (sofa / armchair)
        (0.78, 0.84, 0.40, 4.6, -0.16, 0.40, 0.86),   # come back in
        (0.90, 0.91, 0.31, 2.95, -0.31, 0.15, 0.71),  # focus-pull close on the vase / table (材质)
        (1.00, 0.91, 0.31, 2.95, -0.31, 0.15, 0.71),  # hold on the close-up (flat -> cached dwell)
    ]
    record_build(
        stages, "interior_build", captions=captions, automode=True, keyframes=moves,
        size=(854, 480) if quick else (1280, 720),
        fps=24, per=8 if quick else 12, hold=12 if quick else 26,
        gif_w=480, gif_fps=10,   # a moving clip can't dedupe frames -> keep the .gif lean
        tmp=Path(os.environ["ANIM_TMP"]) if os.environ.get("ANIM_TMP") else None,
    )


def main():
    threads = None
    for a in sys.argv:
        if a.startswith("--threads="):
            threads = int(a.split("=", 1)[1])
    if "--hero" in sys.argv:
        render_interior(ROOT / "docs" / "gallery" / "interior.png", spp=400, w=1280, h=860, threads=threads)
    elif "--film" in sys.argv:
        film()
    elif "--oplog" in sys.argv:
        OUT.mkdir(parents=True, exist_ok=True)
        (OUT / "interior.json").write_text(build_room().to_json())
        print(f"  wrote {OUT/'interior.json'}  (the scene as a legible place-op-log)")
    elif "--preview" in sys.argv:
        render_interior(OUT / "interior_preview.png", spp=24, w=520, h=350, threads=threads)
    else:
        render_interior(OUT / "interior.png", spp=160, threads=threads)
        print("  (a whole interior: every object native-modelled, engine-composed via place)")


if __name__ == "__main__":
    main()
