"""Case 26 — reproducing a real scene from one photograph.

The reference is a single CCTV still of a petrol-station forecourt (Camera 01, wet
morning). Nothing about it was authored for a renderer: a wide-angle lens near the ceiling,
a graphic painted floor, a dispenser island covered in small hardware, and a working yard
behind it. The job is to read the photo and rebuild the scene as an op-log — geometry,
palette, layout and camera — then render it from the same viewpoint and put the two side by
side.

What it exercises: `spin` (the lathe) for the hose loops and the fire bucket, box-selector
`material` painting for the rail's hazard bands, `place` for the layout, and the tracer's
smooth shading + creases on the hardware.

    uv run python examples/cases/26_forecourt.py            # hero -> docs/gallery
    uv run python examples/cases/26_forecourt.py --preview  # fast low-spp look
    uv run python examples/cases/26_forecourt.py --compare  # reference | render

Needs mirage_render + Pillow.
"""
import math
import os
import subprocess
import sys
from pathlib import Path

from mirage.capture import default_render
from mirage.meshlang import MeshProgram

ROOT = Path(__file__).resolve().parents[2]
RENDER = default_render()
OUT = Path(__file__).resolve().parent / "outputs" / "26_forecourt"
GALLERY = ROOT / "docs" / "gallery"
# The reference lives outside the repo (it is somebody's CCTV frame, not an asset). Override
# with MIRAGE_REF; without it the render still runs, only the side-by-side is skipped.
REF = Path(os.environ.get("MIRAGE_REF", "D:/dRepo_26/frame_2026-5-9_09-28-55.png"))
# A path-traced forecourt at 300spp is minutes on a laptop and seconds on the 152-core box.
# Don't hardcode the machine you happen to be sitting at.
THREADS = os.environ.get("MIRAGE_THREADS", "14")

BIG = 500.0


# ---- palette ------------------------------------------------------------------- #
# Sampled from the reference and converted sRGB -> linear, then lifted for the fact that a
# photo's pixel is albedo x light x tonemap, not albedo.
def mat(c, metallic=0.0, roughness=0.5, emission=None):
    m = {"color": list(c), "metallic": metallic, "roughness": roughness}
    if emission:
        m["emission"] = list(emission)
    return m


CONCRETE  = mat((0.30, 0.305, 0.295), 0.0, 0.72)    # the forecourt slab, damp
WET       = mat((0.16, 0.17, 0.185), 0.0, 0.18)     # standing water / wet patches
ORANGE    = mat((0.46, 0.165, 0.085), 0.0, 0.62)    # terracotta bay
ORANGE_W  = mat((0.60, 0.32, 0.235), 0.0, 0.68)     # its worn/faded half
SLATE     = mat((0.15, 0.185, 0.275), 0.0, 0.42)    # the blue bay
NAVY      = mat((0.115, 0.14, 0.20), 0.0, 0.55)
LINE      = mat((0.74, 0.745, 0.72), 0.0, 0.70)     # painted lines
ASPHALT   = mat((0.055, 0.058, 0.060), 0.0, 0.55)   # the road beyond
YELLOWP   = mat((0.62, 0.44, 0.05), 0.0, 0.70)      # yellow road paint

COL_GREY  = mat((0.56, 0.56, 0.55), 0.0, 0.42)      # canopy column cladding
PANEL_BL  = mat((0.045, 0.085, 0.40), 0.0, 0.35)    # the ad panel / dispenser blue
PANEL_WH  = mat((0.80, 0.80, 0.79), 0.0, 0.40)
RED       = mat((0.50, 0.045, 0.030), 0.0, 0.45)    # fire box / the panel's red banner
ORANGE_S  = mat((0.72, 0.26, 0.03), 0.0, 0.45)      # the dispenser's orange stripe
YELLOW    = mat((0.74, 0.55, 0.03), 0.0, 0.45)      # rail bands / the big arrow
BLACK     = mat((0.028, 0.028, 0.030), 0.0, 0.55)
HOSE      = mat((0.022, 0.022, 0.024), 0.0, 0.62)
STEEL     = mat((0.52, 0.52, 0.53), 1.0, 0.38)      # the bucket, checker plate
CHROME    = mat((0.62, 0.63, 0.64), 1.0, 0.22)
WHITE     = mat((0.78, 0.78, 0.77), 0.0, 0.45)
GLASS     = mat((0.06, 0.07, 0.08), 0.0, 0.10)
TYRE      = mat((0.020, 0.020, 0.022), 0.0, 0.80)
WALL_TILE = mat((0.62, 0.62, 0.60), 0.0, 0.35)
SHUTTER   = mat((0.34, 0.35, 0.36), 0.3, 0.45)
BANNER_BL = mat((0.03, 0.10, 0.42), 0.0, 0.55)
CONE_BL   = mat((0.05, 0.09, 0.34), 0.0, 0.50)


def box(sx, sy, sz):
    """An axis-aligned box of the given SIZE (the cube primitive is unit, centred)."""
    return MeshProgram().cube(size=1.0).scale({"by": "all"}, [sx, sy, sz])


def slab(p, x0, x1, y0, y1, z, t, material):
    """A flat painted patch lying ON the ground, spanning [z, z+t] — not straddling z, which
    buries half of it in the slab and z-fights the rest."""
    p.place(box(x1 - x0, y1 - y0, t), at=[(x0 + x1) / 2, (y0 + y1) / 2, z + t / 2],
            material=material)


# ---- the painted forecourt ------------------------------------------------------ #
# The dominant graphic. Bays run away from the camera; the island is along their left.
# Read off the reference: a terracotta bay, a slate-blue bay, another terracotta, each
# ~3.4 x 6 m, separated by ~0.14 m white lines, on a damp concrete slab.
LINE_W = 0.14
BAYS = [  # (x0, x1, y0, y1, material)
    (-0.05, 0.70, -1.2, 5.4, NAVY),       # the narrow dark strip beside the island
    (0.84, 4.25, -0.4, 6.1, SLATE),       # the big blue bay
    (0.84, 3.55, 6.25, 9.6, ORANGE_W),    # the faded terracotta beyond it
    (4.39, 8.10, -2.6, 4.6, ORANGE),      # the big near terracotta bay
]


def forecourt():
    p = MeshProgram()
    p.place(box(46, 54, 0.4), at=[8, 12, -0.2], material=CONCRETE)          # the slab
    p.place(box(46, 26, 0.42), at=[8, 30, -0.19], material=ASPHALT)         # the road beyond
    for x0, x1, y0, y1, m in BAYS:                                           # painted bays
        slab(p, x0, x1, y0, y1, 0.004, 0.006, m)
        for lx in (x0 - LINE_W, x1):                                         # bordering lines
            slab(p, lx, lx + LINE_W, y0 - LINE_W, y1 + LINE_W, 0.006, 0.006, LINE)
        for ly in (y0 - LINE_W, y1):
            slab(p, x0 - LINE_W, x1 + LINE_W, ly, ly + LINE_W, 0.006, 0.006, LINE)
    # standing water: the wet sheen the whole shot lives on
    for x0, x1, y0, y1 in [(2.6, 5.4, 2.2, 5.6), (3.0, 4.6, 0.2, 2.0),
                           (1.6, 3.4, 4.4, 6.0), (6.4, 9.2, 6.0, 12.0),
                           (10.0, 20.0, 8.0, 18.0), (0.5, 6.0, 12.0, 20.0)]:
        slab(p, x0, x1, y0, y1, 0.008, 0.004, WET)
    # the white circle painted around the island, and a yellow lane arrow
    p.place(MeshProgram().cylinder(sides=64, radius=2.30, height=0.006)
            .place(MeshProgram().cylinder(sides=64, radius=2.16, height=0.02),
                   at=[0, 0, 0], material=CONCRETE),
            at=[-1.75, -2.6, 0.004], material=LINE)
    p.place(box(0.16, 2.2, 0.006), at=[1.6, 13.6, 0.004], material=YELLOWP)
    for s in (-1, 1):
        p.place(box(0.16, 1.0, 0.006), at=[1.6 + s * 0.30, 12.8, 0.004],
                rotate=[0, 0, s * 38], material=YELLOWP)
    return p


# ---- the dispenser island ------------------------------------------------------- #
ISL_X, ISL_Y, ISL_Z = 1.05, 5.2, 0.17     # plinth footprint and height


def hose_loop(radius=0.34, tube=0.021, angle=250.0):
    """A fuel hose hanging in a loop — the lathe, used as a bender.

    `profile` lays a small circle as a wire, `spin` revolves it about the z axis. Revolved
    only part-way it is not a torus but an ARC of tube: exactly a hose drooping from its
    holster. Six of these, rotated and placed, are the black tangle either side.
    """
    pts = [(radius + tube * math.cos(a * math.pi / 8), tube * math.sin(a * math.pi / 8))
           for a in range(16)]
    p = MeshProgram()
    p.profile(points=[list(q) for q in pts], plane="xz", closed=True)
    p.spin(axis="z", steps=40, angle=angle)
    return p


def dispenser():
    """The blue pump body, its hoses, and the ad panel above — all hung off the column."""
    p = MeshProgram()
    # pump body: a blue box with a silver band and an orange stripe
    p.place(box(0.86, 0.62, 1.28), at=[0, 0, 0.81], material=PANEL_BL)
    p.place(box(0.88, 0.64, 0.075), at=[0, 0, 1.30], material=ORANGE_S)
    p.place(box(0.885, 0.645, 0.05), at=[0, 0, 1.03], material=PANEL_WH)
    for s in (-1, 1):                                   # the white corner pilasters
        p.place(box(0.045, 0.645, 1.28), at=[s * 0.425, 0, 0.81], material=PANEL_WH)
    p.place(box(0.20, 0.06, 0.16), at=[0.16, -0.33, 1.16], material=PANEL_WH)  # keypad plate

    # hoses: loops drooping either side, plus the nozzles in their holsters
    for s in (-1, 1):
        for k, (r, dz, tilt) in enumerate([(0.40, 0.30, 8), (0.34, 0.52, -6), (0.29, 0.74, 4)]):
            p.place(hose_loop(radius=r, angle=252 - 14 * k),
                    at=[s * 0.50, 0.02 + 0.10 * k, dz + 0.34],
                    rotate=[90, tilt, 90 + s * 8], material=HOSE)
        p.place(box(0.07, 0.10, 0.30), at=[s * 0.52, -0.20, 1.10],
                rotate=[16, 0, 0], material=BLACK)      # the nozzle in its holster
    return p


def ad_panel():
    """The lightbox on the column: 油卡支付 超划算, a red banner, and a big yellow arrow."""
    p = MeshProgram()
    p.place(box(0.94, 0.09, 2.05), at=[0, 0, 0], material=PANEL_WH)       # the white frame
    p.place(box(0.86, 0.11, 1.95), at=[0, 0, 0], material=PANEL_BL)       # the blue face
    p.place(box(0.64, 0.13, 0.16), at=[0, -0.015, 0.16], material=RED)    # 每周二…特惠日
    # the down arrow: a shaft plus a rotated square for the head
    p.place(box(0.30, 0.13, 0.36), at=[0, -0.02, -0.19], material=YELLOW)
    p.place(box(0.34, 0.13, 0.34), at=[0, -0.02, -0.46], rotate=[0, 45, 0], material=YELLOW)
    p.place(box(0.60, 0.13, 0.10), at=[0, -0.02, 0.86], material=PANEL_WH)  # 油卡支付
    p.place(box(0.60, 0.13, 0.10), at=[0, -0.02, 0.66], material=PANEL_WH)  # 超划算
    p.place(box(0.52, 0.13, 0.05), at=[0, -0.02, -0.84], material=PANEL_WH)  # 其他时段…
    return p


def fire_box():
    """灭火器箱 — the red cabinet, two doors and an orange stripe."""
    p = MeshProgram()
    p.place(box(0.44, 0.30, 0.86), at=[0, 0, 0.43], material=RED)
    p.place(box(0.455, 0.31, 0.045), at=[0, 0, 0.72], material=ORANGE_S)
    p.place(box(0.40, 0.02, 0.012), at=[0, -0.155, 0.44], material=BLACK)   # the door split
    p.place(box(0.13, 0.10, 0.16), at=[0.02, 0.0, 0.93], material=WHITE)    # the glove box on top
    return p


def bucket():
    """消防桶 — a tapered fire bucket, turned on the lathe from its own section."""
    p = MeshProgram()
    p.profile(points=[[0.005, 0.0], [0.115, 0.0], [0.152, 0.30], [0.158, 0.315],
                      [0.150, 0.318], [0.145, 0.31], [0.108, 0.012], [0.005, 0.012]],
              plane="xz", closed=True)
    p.spin(axis="z", steps=44, angle=360.0)
    return p


def hazard_rail():
    """The guard rail: three legs and a top rail, banded black and yellow.

    Built from SEGMENTS rather than painted by a box query, and the reason is a nice bit of
    kernel reality: a cylinder's side faces are full-length quads whose centroids all sit at
    its middle, so a box selector asking for "the faces in this 0.15 m slice" correctly
    matches nothing — there is no geometry there to select. Segment the tube and the bands
    are the segments, which is also what the real thing is: separate paint.
    """
    p = MeshProgram()
    LEN, H, R, band = 3.05, 0.62, 0.043, 0.152
    n = max(2, int(round(LEN / band)))
    for i in range(n):                                              # the top rail
        p.place(MeshProgram().cylinder(sides=18, radius=R, height=LEN / n + 0.003),
                at=[-LEN / 2 + (i + 0.5) * LEN / n, 0, H], rotate=[0, 90, 0],
                material=BLACK if i % 2 else YELLOW)
    for lx in (-LEN / 2 + R, 0.02, LEN / 2 - R):                    # three legs
        m = max(2, int(round(H / band)))
        for k in range(m):
            p.place(MeshProgram().cylinder(sides=18, radius=R, height=H / m + 0.003),
                    at=[lx, 0, (k + 0.5) * H / m],
                    material=BLACK if k % 2 else YELLOW)
        p.place(MeshProgram().uv_sphere(segments=18, rings=10, radius=R),   # the elbow
                at=[lx, 0, H], material=YELLOW)
    return p


def island():
    p = MeshProgram()
    # the plinth: a concrete kerb capped with checker plate
    p.place(box(2 * ISL_X, ISL_Y, ISL_Z), at=[0, 0, ISL_Z / 2], material=CONCRETE)
    p.place(box(2 * ISL_X - 0.10, ISL_Y - 0.10, 0.02), at=[0, 0, ISL_Z], material=STEEL)
    # the canopy column, rising out of frame
    p.place(box(0.46, 0.46, 5.4), at=[0, 0.75, ISL_Z + 2.7], material=COL_GREY)
    p.place(ad_panel(), at=[0, 0.52, ISL_Z + 2.62], material=None)
    p.place(box(0.40, 0.02, 0.10), at=[0, 0.51, ISL_Z + 3.78], material=WHITE)  # the Gulf plate
    p.place(dispenser(), at=[0, 0.30, ISL_Z])
    p.place(fire_box(), at=[0.10, -0.44, ISL_Z])
    p.place(bucket(), at=[0.58, -0.42, ISL_Z], material=STEEL)
    for i, (dx, h, c) in enumerate([(-0.30, 0.115, (0.34, 0.20, 0.05)),
                                    (-0.14, 0.105, (0.72, 0.72, 0.70))]):
        p.place(MeshProgram().cylinder(sides=16, radius=0.032, height=h),
                at=[dx, -0.50, ISL_Z + h / 2], material=mat(c, 0.0, 0.30))   # the little bottles
    return p


# ---- the yard behind ------------------------------------------------------------ #
def van():
    """The white van: a body, a cab step-down, a glass band and four wheels."""
    p = MeshProgram()
    p.place(box(5.30, 2.02, 1.30), at=[0, 0, 1.30], material=WHITE)          # box body
    p.place(box(1.55, 1.96, 0.72), at=[2.55, 0, 0.86], material=WHITE)       # bonnet
    p.place(box(5.34, 2.04, 0.34), at=[0, 0, 0.62], material=mat((0.30, 0.31, 0.32), 0.0, 0.5))
    p.place(box(3.30, 2.06, 0.52), at=[-0.35, 0, 1.72], material=GLASS)      # side glass
    p.place(box(0.10, 1.90, 0.62), at=[3.28, 0, 1.55], rotate=[0, 22, 0], material=GLASS)
    for x, y in [(1.95, 1.02), (1.95, -1.02), (-1.85, 1.02), (-1.85, -1.02)]:
        p.place(MeshProgram().cylinder(sides=24, radius=0.37, height=0.24),
                at=[x, y, 0.37], rotate=[90, 0, 0], material=TYRE)
        p.place(MeshProgram().cylinder(sides=20, radius=0.21, height=0.26),
                at=[x, y, 0.37], rotate=[90, 0, 0], material=CHROME)
    p.place(box(0.30, 0.16, 0.10), at=[2.66, -0.86, 0.60], material=YELLOW)  # plate
    return p


def suv():
    p = MeshProgram()
    p.place(box(4.55, 1.86, 0.80), at=[0, 0, 0.72], material=WHITE)
    p.place(box(2.70, 1.80, 0.62), at=[-0.25, 0, 1.42], material=WHITE)
    p.place(box(2.40, 1.84, 0.44), at=[-0.25, 0, 1.44], material=GLASS)
    for x, y in [(1.45, 0.92), (1.45, -0.92), (-1.45, 0.92), (-1.45, -0.92)]:
        p.place(MeshProgram().cylinder(sides=24, radius=0.34, height=0.22),
                at=[x, y, 0.34], rotate=[90, 0, 0], material=TYRE)
    p.place(box(0.24, 0.14, 0.09), at=[2.30, -0.60, 0.52], material=mat((0.10, 0.35, 0.75), 0, 0.4))
    return p


def cone(h=0.80, r=0.14):
    """A blue/white reflective bollard — stacked frusta, same reason as the rail: the bands
    have to BE geometry before a colour can land on only some of them."""
    p = MeshProgram()
    n = 8
    for k in range(n):
        t0, t1 = k / n, (k + 1) / n
        rr = r * (1.0 - 0.62 * (t0 + t1) / 2)
        p.place(MeshProgram().cylinder(sides=20, radius=rr, height=h / n + 0.004),
                at=[0, 0, 0.03 + (k + 0.5) * h / n],
                material=WHITE if k % 2 else CONE_BL)
    p.place(box(2 * r + 0.12, 2 * r + 0.12, 0.03), at=[0, 0, 0.015], material=CONE_BL)
    return p


def yard():
    """The building line across the back: tiled wall, roller shutters, and its clutter."""
    p = MeshProgram()
    p.place(box(40, 0.4, 5.0), at=[10, 27.5, 2.5], material=WALL_TILE)       # the facade
    for i in range(6):                                                        # roller shutters
        p.place(box(3.10, 0.10, 3.20), at=[-2.0 + i * 4.2, 27.26, 1.75], material=SHUTTER)
        p.place(box(3.30, 0.14, 0.16), at=[-2.0 + i * 4.2, 27.24, 3.42], material=BLACK)
    p.place(box(2.6, 0.12, 2.9), at=[-6.6, 27.26, 1.55], material=BLACK)      # an open doorway
    for x in (-4.4, 8.6, 14.0):                                               # hanging banners
        p.place(box(0.42, 0.06, 1.9), at=[x, 27.1, 2.6], material=ORANGE_S)
    p.place(box(0.10, 0.10, 5.4), at=[19.5, 27.0, 2.7], material=BLACK)

    p.place(van(), at=[9.0, 24.4, 0], rotate=[0, 0, 180])
    p.place(suv(), at=[-4.35, 6.60, 0], rotate=[0, 0, 92])   # far left, mostly out of frame
    for x, y in [(-3.0, 22.6), (-1.9, 22.6), (12.4, 24.0), (16.2, 23.2), (18.4, 20.0)]:
        p.place(cone(), at=[x, y, 0], material=None)
    for s in (0, 1):                                                          # 小心地滑 A-frames
        p.place(box(0.42, 0.32, 0.66), at=[-5.6 + s * 0.62, 25.9, 0.33],
                rotate=[0, 0, 6 * s], material=YELLOW)
    # the 加油站洗车机 banner at the left edge
    p.place(box(0.06, 2.4, 2.5), at=[-6.2, 17.0, 1.6], rotate=[0, 0, -8], material=BANNER_BL)
    p.place(box(0.06, 3.0, 2.9), at=[-7.4, 8.0, 1.9], rotate=[0, 0, -4], material=BANNER_BL)
    # a black/white speed hump at the right
    for i in range(9):
        p.place(box(0.34, 0.30, 0.07), at=[13.0 + i * 0.34, 9.6, 0.035],
                material=WHITE if i % 2 else BLACK)
    return p


ISLAND_X = -1.60   # the island sits left of the bays; the camera looks past it down the lane


def scene():
    p = MeshProgram()
    p.place(forecourt())
    p.place(island(), at=[ISLAND_X, 0, 0])
    p.place(hazard_rail(), at=[ISLAND_X + 0.05, -2.05, 0], material=None)
    p.place(yard())
    return p


# ---- render --------------------------------------------------------------------- #
# SOLVED from the reference, not nudged. Constraints read off the photo:
#   ~100 deg horizontal at 16:9        -> half-vFOV 33.8, so fov_y = 1.181 (not the 1.02 a
#                                         guess gives -- a security lens is wider than it looks)
#   the yard sits on the TOP edge      -> axis pitch -32 deg
#   the column lands ~20% from left    -> ~30 deg left of the axis
#   the column fills the frame height  -> ~5 m from the eye
#   the island RECEDES UP-AND-RIGHT    -> the eye is to its RIGHT, looking down its length.
#     This one is easy to get backwards and it inverts the whole composition. The tell is in
#     the photo: the hoses hang to the LEFT and RIGHT of the pump and the platform runs
#     across the frame, so the camera cannot be looking along the island from its end.
# The solution checks out: frame centre lands on ground ~6.9 m out (the blue bay, where the
# reference has it), the bottom edge at ~1.9 m (the island), the yard at 99% to the top.
CAM_EYE, CAM_TGT, CAM_FOV = [2.00, -4.20, 4.30], [1.53, 2.57, 0.06], 1.181


def render(prog, out, spp, w, h, extra=()):
    OUT.mkdir(parents=True, exist_ok=True)
    js = OUT / (out + ".json")
    js.write_text(prog.to_json())
    ppm = OUT / (out + ".ppm")
    subprocess.run([str(RENDER), "--oplog", str(js), "--out", str(ppm),
                    "--spp", str(spp), "--w", str(w), "--h", str(h), "--threads", THREADS,
                    "--cam-eye", *[str(v) for v in CAM_EYE],
                    "--cam-target", *[str(v) for v in CAM_TGT],
                    "--cam-fov", str(CAM_FOV), *extra], check=True)
    from PIL import Image
    png = OUT / (out + ".png")
    Image.open(ppm).save(png)
    return png


def compare(png):
    """Reference above, render below — the only honest way to report a reproduction."""
    from PIL import Image, ImageDraw
    a = Image.open(REF).convert("RGB")
    b = Image.open(png).convert("RGB")
    w = 1280
    a = a.resize((w, int(a.height * w / a.width)), Image.LANCZOS)
    b = b.resize((w, int(b.height * w / b.width)), Image.LANCZOS)
    out = Image.new("RGB", (w, a.height + b.height + 6), (18, 18, 20))
    out.paste(a, (0, 0))
    out.paste(b, (0, a.height + 6))
    d = ImageDraw.Draw(out)
    for y, t in [(0, "reference  (CCTV still)"), (a.height + 6, "mirage  (op-log, path-traced)")]:
        d.rectangle([0, y, 250, y + 18], fill=(12, 12, 14))
        d.text((6, 4 + y), t, fill=(238, 238, 240))
    p = OUT / "compare.png"
    out.save(p)
    return p


def main():
    preview = "--preview" in sys.argv
    p = scene()
    m = p.build()
    print(f"forecourt: {len(m.verts):,} verts  {len(m.faces):,} faces  ({len(p.ops)} top-level ops)")
    spp = 44 if preview else 300
    # An overcast wet morning: soft high sun, sky fill, no hard key.
    #
    # The exposure is SOLVED, not chosen: mirage.photomatch reports the render's median
    # luma against the reference's in stops, and a secant on that converges in two steps
    # (0.38 -> -2.36 stops, 1.406 -> +0.13, 1.353 -> +0.07). By eye I had swung from a stop
    # too bright to two and a third too dark across half a dozen renders and called each of
    # them about right. Do not hand-tune what you can measure.
    png = render(p, "hero", spp, 1600, 900,
                 extra=["--sun", "0.18", "--env", "0.42", "--exposure", "1.353",
                        "--sun-dir", "0.30", "0.62", "0.72", "--denoise", "4"])
    print("wrote", png)
    have_ref = REF.exists()
    if have_ref:
        print("wrote", compare(png))
    else:
        print(f"(no reference at {REF} — skipping the side-by-side; set MIRAGE_REF)")
    if not preview:
        GALLERY.mkdir(parents=True, exist_ok=True)
        from PIL import Image
        Image.open(png).save(GALLERY / "forecourt.png")
        if have_ref:
            Image.open(OUT / "compare.png").save(GALLERY / "forecourt_compare.png")
        print("wrote", GALLERY / "forecourt.png")


if __name__ == "__main__":
    main()
