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


CONCRETE  = mat((0.30, 0.305, 0.295), 0.0, 0.72)    # dry plinth kerb / circle infill
YELLOWP   = mat((0.62, 0.44, 0.05), 0.0, 0.70)      # yellow road paint
# (the wet ground palette lives just below, after the object materials)

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

# Wet-overcast ground. The whole forecourt is a mirror of a bright sky, so the wet surfaces
# are near-BLACK diffuse with a low-roughness specular that carries the reflection: the
# contrast IS the wetness. A light matte grey (what "damp concrete" naively wants to be) kills
# it. Sampled darker than the photo's pixels, because a photo pixel is albedo x light, and the
# reflection supplies the light. See the render notes at the foot of the file.
WET_CONC  = mat((0.058, 0.061, 0.066), 0.0, 0.14)    # damp concrete slab
DRY_CONC  = mat((0.255, 0.255, 0.247), 0.0, 0.66)    # the drier apron patches (matte)
WET_STRK  = mat((0.030, 0.032, 0.037), 0.0, 0.10)    # darker wet streaks
TYRE_MK   = mat((0.045, 0.045, 0.047), 0.0, 0.55)    # dry tyre-scuff, matte
ASPH_W    = mat((0.018, 0.020, 0.024), 0.0, 0.10)    # wet asphalt road, near-black mirror
# The bays are soaked-MATTE, not mirrors: a broad specular lobe (roughness ~0.3-0.45) reflects
# the sky and column as a soft sheen the denoiser can clean, where a sharp lobe left a firefly
# triangle it could not (the reflected column is not in its albedo/normal guide buffer).
SLATE_W   = mat((0.034, 0.044, 0.072), 0.0, 0.18)    # the blue bay, soaked
SHEET_W   = mat((0.050, 0.055, 0.066), 0.0, 0.30)    # the blue bay's wettest sheet
TERRA_D   = mat((0.360, 0.135, 0.072), 0.0, 0.44)    # terracotta, drier
TERRA_F   = mat((0.260, 0.125, 0.085), 0.0, 0.42)    # faded terracotta beyond
NAVY_W    = mat((0.028, 0.038, 0.062), 0.0, 0.16)    # the dark strip by the island
PUDDLE    = mat((0.022, 0.028, 0.040), 0.0, 0.26)    # standing water, broad soft reflection
LINE_W    = mat((0.60, 0.605, 0.59), 0.0, 0.58)      # painted lines, damp

# The scene's forward axis (the bay's long edge, world +y) is not square to the building line
# behind it; the yard and road sit at this yaw.
ANG = -15.0


def box(sx, sy, sz):
    """An axis-aligned box of the given SIZE (the cube primitive is unit, centred)."""
    return MeshProgram().cube(size=1.0).scale({"by": "all"}, [sx, sy, sz])


def slab(p, x0, x1, y0, y1, z, t, material):
    """A flat painted patch lying ON the ground, spanning [z, z+t] — not straddling z, which
    buries half of it in the slab and z-fights the rest."""
    p.place(box(x1 - x0, y1 - y0, t), at=[(x0 + x1) / 2, (y0 + y1) / 2, z + t / 2],
            material=material)


# ---- the painted forecourt ------------------------------------------------------ #
# The dominant graphic, rebuilt in the SOLVED camera's frame by unprojecting the photo. The
# blue bay's four measured corners are a 3.47 x 6.0 m rectangle at world x[0,3.47] y[0,6]; the
# terracotta bays, the road and the building line were unprojected the same way with
# solve.ground_point. Nothing here is eyeballed against a render -- the camera put every edge
# where it saw it, and a projection overlay on the photo confirmed the fit before any render.
def forecourt():
    p = MeshProgram()
    p.place(box(44, 48, 0.4), at=[6, 8, -0.2], material=WET_CONC)            # the damp slab
    for cx, cy, sx, sy in [(9, 10.5, 9, 6), (13, 6, 8, 8), (-4, 3, 5, 9)]:   # drier patches
        p.place(box(sx, sy, 0.01), at=[cx, cy, 0.006], material=DRY_CONC)
    p.place(box(40, 9.5, 0.03), at=[7, 16.2, 0.012], rotate=[0, 0, ANG], material=ASPH_W)
    LW = 0.12
    bays = [(0, 3.47, 0, 6, SLATE_W), (3.62, 7.5, -3.4, 6.0, TERRA_D),
            (0, 3.47, 6.2, 9.7, TERRA_F), (-1.30, -0.14, -1.6, 5.7, NAVY_W)]
    for x0, x1, y0, y1, m in bays:
        slab(p, x0, x1, y0, y1, 0.004, 0.006, m)
        for lx in (x0 - LW, x1):
            slab(p, lx, lx + LW, y0 - LW, y1 + LW, 0.006, 0.006, LINE_W)
        for ly in (y0 - LW, y1):
            slab(p, x0 - LW, x1 + LW, ly, ly + LW, 0.006, 0.006, LINE_W)
    # the soaked centre of the blue bay + wet sheets bridging into the terracotta. DISTINCT
    # heights on purpose: overlapping slabs that share a top face z-fight, and the denoiser
    # smears that flicker into a grainy triangular patch that no spp or roughness removes.
    for x0, x1, y0, y1, z, m in [(0.25, 3.1, 0.4, 4.9, 0.014, SHEET_W),
                                 (0.7, 2.5, 1.1, 3.6, 0.018, PUDDLE),
                                 (3.7, 6.9, -1.2, 3.2, 0.016, PUDDLE)]:
        slab(p, x0, x1, y0, y1, z, 0.003, m)
    # tyre tracks and wet streaks that break the flat sheet
    for x, y0, y1, ww, m in [(4.9, 7, 15, 0.24, WET_STRK), (5.5, 7, 15, 0.24, WET_STRK),
                             (8.6, 8, 16, 0.24, TYRE_MK), (9.2, 8, 16, 0.24, TYRE_MK),
                             (12.0, 3, 12, 0.8, WET_STRK), (6.4, -3, 5, 0.6, WET_STRK)]:
        slab(p, x - ww / 2, x + ww / 2, y0, y1, 0.007, 0.004, m)
    # the white circle painted around the island (centre + radius measured), the yellow arrow
    p.place(MeshProgram().cylinder(sides=72, radius=1.03, height=0.006)
            .place(MeshProgram().cylinder(sides=72, radius=0.91, height=0.02),
                   at=[0, 0, 0], material=WET_CONC),
            at=[-3.20, -2.44, 0.004], material=LINE_W)
    p.place(box(0.15, 2.0, 0.006), at=[9.0, 9.4, 0.004], rotate=[0, 0, ANG], material=YELLOWP)
    for s in (-1, 1):
        p.place(box(0.15, 0.95, 0.006), at=[9.0 + s * 0.28, 8.55, 0.004],
                rotate=[0, 0, ANG + s * 40], material=YELLOWP)
    p.place(box(14, 0.14, 0.006), at=[8, 12.6, 0.004], rotate=[0, 0, ANG], material=YELLOWP)
    return p


# ---- the dispenser island ------------------------------------------------------- #


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
    """The dispenser island, built for THIS shot: a tall clad column that leaves the frame,
    the blue ad lightbox on its face, the pump and hoses at its foot, fire box and bucket.
    Placed by unprojecting the pump footprint, the fire box and the painted circle; the sign
    height was set against a projection overlay so 油卡支付 / 降 1.1 land on the real sign."""
    p = MeshProgram()
    Z = 0.14
    p.place(box(1.9, 1.35, Z), at=[0, 0.10, Z / 2], material=CONCRETE)          # grate plinth
    p.place(box(1.78, 1.24, 0.03), at=[0, 0.10, Z], material=STEEL)
    p.place(box(0.72, 0.60, 8.0), at=[0.04, 0.42, Z + 3.9], material=COL_GREY, mark="column")
    p.place(box(0.44, 0.02, 0.11), at=[0.04, 0.13, Z + 3.28], material=WHITE)   # the Gulf plate
    p.place(ad_panel(), at=[0.04, 0.13, Z + 2.07], material=None, mark="sign")
    p.place(dispenser(), at=[0, 0.02, Z], mark="dispenser")
    p.place(fire_box(), at=[0.06, -0.40, Z], mark="firebox")
    p.place(bucket(), at=[0.54, -0.40, Z], material=STEEL)
    for i, (dx, h, col) in enumerate([(-0.30, 0.115, (0.34, 0.20, 0.05)),
                                      (-0.14, 0.105, (0.72, 0.72, 0.70))]):
        p.place(MeshProgram().cylinder(sides=16, radius=0.032, height=h),
                at=[dx, -0.46, Z + h / 2], material=mat(col, 0.0, 0.30))        # the little bottles
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
    """The building line across the back and its clutter -- facade, roller shutters, the white
    van, bollards, the speed hump, the wash-machine banner -- placed by unprojecting the
    building base line (world y ~ 20, yawed by ANG) and the road front."""
    p = MeshProgram()
    p.place(box(24, 0.4, 5.2), at=[8.5, 20.2, 2.6], rotate=[0, 0, ANG], material=WALL_TILE,
            mark="facade")
    for i in range(6):                                                        # roller shutters
        dx = -6 + i * 3.9
        cxy = [8.5 + dx * math.cos(math.radians(ANG)), 20.2 + dx * math.sin(math.radians(ANG))]
        p.place(box(3.0, 0.10, 3.1), at=[cxy[0], cxy[1] - 0.18, 1.75], rotate=[0, 0, ANG],
                material=SHUTTER)
    p.place(box(2.4, 0.12, 2.8), at=[15.5, 19.0, 1.5], rotate=[0, 0, ANG], material=BLACK)  # doorway
    for dx in (-7.5, 5.5, 10.0):                                              # hanging banners
        cxy = [8.5 + dx * math.cos(math.radians(ANG)), 20.2 + dx * math.sin(math.radians(ANG))]
        p.place(box(0.42, 0.06, 1.8), at=[cxy[0], cxy[1] - 0.25, 2.7], rotate=[0, 0, ANG],
                material=ORANGE_S)
    p.place(van(), at=[9.9, 17.3, 0], rotate=[0, 0, -11], mark="van")
    for cx, cy in [(9.6, 18.3), (11.1, 18.2), (12.6, 18.1), (14.0, 17.9), (7.9, 18.4),
                   (15.4, 17.7)]:                                             # blue-white bollards
        p.place(cone(), at=[cx, cy, 0], material=None)
    for i in range(9):                                                        # the speed hump
        dx = -1.6 + i * 0.4
        cxy = [11.0 + dx * math.cos(math.radians(26)), 5.6 + dx * math.sin(math.radians(26))]
        p.place(box(0.36, 0.30, 0.08), at=[cxy[0], cxy[1], 0.04], rotate=[0, 0, 26],
                material=WHITE if i % 2 else BLACK)
    p.place(box(0.06, 2.6, 2.7), at=[-5.6, 11.0, 1.5], rotate=[0, 0, -6], material=BANNER_BL)
    for s in (0, 1):                                                          # 小心地滑 A-frame
        p.place(box(0.40, 0.30, 0.64), at=[4.1 + s * 0.55, 19.3, 0.32],
                rotate=[0, 0, 6 * s], material=YELLOW)
    p.place(suv(), at=[-5.3, 4.4, 0], rotate=[0, 0, 96])   # far left, mostly out of frame
    return p


ISLAND_AT = [-3.12, -1.05]   # where the pump island stands, unprojected from its footprint


def scene():
    p = MeshProgram()
    # Every top-level object is MARKED, which makes the scene measurable rather than just
    # renderable: mirage_render --ids turns these tags into a per-pixel object id, so
    # photomatch.chamfer_per_object can score each against the photo separately. That per-object
    # score is the loss that will POLISH this scene; measurement -- the solved camera and the
    # unprojected layout below -- is what landed it in the basin first, which no loss could do:
    # eleven cameras once all scored 14-15 px because nothing downhill led to the true camera,
    # 22 deg of yaw and half the fov away.
    p.place(forecourt(), mark="forecourt")
    p.place(island(), at=[ISLAND_AT[0], ISLAND_AT[1], 0], rotate=[0, 0, ANG], mark="island")
    p.place(hazard_rail().scale({"by": "all"}, [0.60, 1.0, 1.0]),
            at=[-3.16, -2.20, 0], rotate=[0, 0, -16], material=None, mark="rail")
    p.place(yard(), mark="yard")
    return p


# ---- render --------------------------------------------------------------------- #
# THE CAMERA, SOLVED. The 1189 px falsification (git log: "let a camera be tested against the
# photograph, and fail") retired the asserted camera; this is what replaced it, and how:
#
#   - orientation + fov from the bay's strip vanishing point (272,-412) FUSED with "the bay is
#     a rectangle": the single fov that unprojects the four measured corners to right angles.
#     That pins fov to 0.505 with no fragile vertical trace, and the strip VP reproduces exactly.
#   - the corners then unproject to 91/89/89/91 deg (the asserted camera gave 77/93/81/108) and
#     aspect 1.73 -- a 3.47 x 6.0 m bay. An independent check that the orientation+fov are right.
#   - eye by linear least squares once orientation+fov are fixed and the gauge is chosen (bay
#     length 6 m): the four corners reproject within 4.2 px. Eye lands 6.1 m up, a CCTV on a post.
#
# The recovered camera differs from the asserted one by exactly what the falsification predicted:
# yaw pulled back ~22 deg, fov roughly halved (1.181 -> 0.505). Every object in the scene above
# was then unprojected through THIS camera with solve.ground_point and checked by projecting the
# layout back onto the photo -- not by eyeballing a render. Measurement lands you in the basin;
# the per-object chamfer loss is what polishes inside it. (solve.camera_from_vanishing_points.)
CAM_EYE, CAM_TGT, CAM_FOV = [-3.1349, -12.0379, 6.0852], [-2.8408, -11.1592, 5.7095], 0.5054


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
    # An overcast wet morning: soft high sun, a bright sky fill for the wet surfaces to mirror,
    # no hard key. Env is turned up (0.86) because on this shot the reflected sky IS the light on
    # the floor -- the wet materials are near-black diffuse and read only through what they
    # reflect. Exposure 1.35 sits the concrete where the reference's is; --clamp 2 stops a hot
    # specular sample on the near-mirror puddles from leaving a firefly the denoiser can't fix.
    png = render(p, "hero", spp, 1600, 900,
                 extra=["--sun", "0.12", "--env", "0.86", "--exposure", "1.35", "--clamp", "1.5",
                        "--sun-dir", "0.25", "0.55", "0.80", "--denoise", "4"])
    print("wrote", png)
    have_ref = REF.exists()
    if have_ref:
        print("wrote", compare(png))
    else:
        print(f"(no reference at {REF} — skipping the side-by-side; set MIRAGE_REF)")
    if not preview:
        GALLERY.mkdir(parents=True, exist_ok=True)
        from PIL import Image
        Image.open(png).save(GALLERY / "forecourt.png")   # the render alone: a pure synthetic image
        # The side-by-side is NOT copied to the gallery, and must never be: it embeds the
        # reference CCTV frame, which is somebody's security footage, not ours to publish. It
        # lives only in outputs/ (gitignored). Only the render ships.
        print("wrote", GALLERY / "forecourt.png")


if __name__ == "__main__":
    main()
