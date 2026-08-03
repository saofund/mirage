"""Case 26 — reproducing a real scene from one photograph.

The reference is a single CCTV still of a petrol-station forecourt (Camera 01, wet
morning). Nothing about it was authored for a renderer: a wide-angle lens near the ceiling,
a graphic painted floor, a dispenser island covered in small hardware, and a working yard
behind it. The job is to read the photo and rebuild the scene as an op-log — geometry,
palette, layout and camera — then render it from the same viewpoint and put the two side by
side.

The scene is built in three separable pieces, which is the point of the case:

* the **camera**, solved from the photo (`mirage.solve`) — see the block above CAM_EYE;
* the **parts**, each modelled and reviewed on its own (`forecourt/parts.py`, and
  `python -m forecourt.sheet` to look at them one at a time);
* the **layout** below, which is nothing but `place` ops at unprojected positions.

Keeping those apart is what makes the thing improvable: a part can be rebuilt without
touching the camera, and the layout can be re-solved without touching the parts.

    uv run python examples/cases/26_forecourt.py            # hero -> docs/gallery
    uv run python examples/cases/26_forecourt.py --preview  # fast low-spp look

Needs mirage_render + Pillow.
"""
import math
import os
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))   # so `forecourt` imports as a kit

from forecourt import parts as P                                          # noqa: E402
from forecourt.materials import (APRON, APRON_LT, BAY_BLUE, BAY_ORNG, BAY_SLATE,  # noqa: E402
                                 DAMP, DAMP2, PUDDLE, PUDDLE_L,
                                 BLACK, CONCRETE, LINE_W, PROMO_F, REPAIR_F, ROAD, SHUTTER_D,
                                 WASH_F, WHITE, YELLOW, YELLOWP, mat)
from mirage.capture import default_render                                 # noqa: E402
from mirage.meshlang import MeshProgram                                   # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent / "outputs" / "26_forecourt"
GALLERY = ROOT / "docs" / "gallery"
# The reference lives outside the repo (it is somebody's CCTV frame, not an asset). Override
# with MIRAGE_REF; without it the render still runs, only the side-by-side is skipped.
REF = Path(os.environ.get("MIRAGE_REF", "D:/dRepo_26/frame_2026-5-9_09-28-55.png"))
# A path-traced forecourt at 300spp is minutes on a laptop and seconds on the 152-core box.
# Don't hardcode the machine you happen to be sitting at.
THREADS = os.environ.get("MIRAGE_THREADS", "14")

# TWO directions, and conflating them was an 8.5-degree error across the whole back of the
# scene — the thing that reads as "the layout is facing the wrong way" without any single
# object looking wrong. Both are now measured off the photograph rather than asserted:
#
#   ANG      the forecourt's own painted edge. Segmenting the yellow road line and
#            unprojecting 6,525 pixels fits y = -0.3635x + 16.61, i.e. -19.98 deg.
#   YARD_ANG the BUILDING line. Two independent fits agree and both disagree with the
#            above: the bollard row (eight bollards, segmented by their blue, contact
#            point = the lowest pixel of each) gives -6.36 deg through y = 18.09 - 0.111x,
#            and the shutter bases give -6.72. A forecourt's paint does not have to be
#            parallel to the shop it is painted in front of, and here it is not.
ANG = -20.0
YARD_ANG = -6.4
YARD_BASE = (8.5, 19.13)
# WHERE THINGS STAND. One dict, read by `scene()` and by `forecourt.place` alike — the audit
# tool used to restate these numbers and therefore audited a scene that did not exist: the
# speed hump had already been moved and the fit was still searching around its old spot,
# reporting "no better than where it is" about a place it no longer was.
#   name: (build, [x, y, z], yaw)
def PLACEMENTS():
    return {
        # Back at the position unprojected from its FOOTPRINT. fit_ground offered
        # [-4.62, -0.15] and halved the chamfer getting there (8.87 -> 3.37) -- and the
        # render shows it pushed half out of frame. Same failure as the rail, one object
        # later: an outline in clutter finds support wherever you put it. Two for two, so
        # the rule is now explicit -- fit_ground is a HINT, and the overlay and the render
        # are the verdict. It earns its keep on isolated objects (the speed hump, alone on
        # bare concrete, went 10.78 -> 2.47 and was right).
        "island":  (P.island, [-3.12, -1.05, 0], ANG),
        # RIGIDLY tied to the island: in the photograph the hoop stands 1.25 m in front of
        # the plinth and always will. fit_ground moved it 2.5 m the other way and improved
        # its chamfer doing it, because it landed in the hose tangle -- exactly the failure
        # photomatch.chamfer_per_object documents ("an object sitting in clutter is near
        # SOMETHING no matter where you put it"). The overlay caught it; the number did not.
        "rail":    (lambda: P.hazard_rail(1.88, 0.74), [-3.16, -2.30, 0], ANG - 1),
        # Solved by `layout.fit_contacts` (see `place.py --measure`) against two things the
        # photograph states outright: the line its tyres' contacts trace, row 109 at column
        # 760 falling to 94 at 1100, and the columns its body spans, 762..1120. 1.2 px rms.
        # It had been three and a half metres too close to the camera all along, which is why
        # it loomed over the frame and covered shutter bases the reference shows — and the
        # scorecard reported its chamfer as 5.3 px, the best of any vehicle in the scene.
        "van":     (P.van, [9.81, 18.15, 0], -4.07),
        "suv":     (P.suv, [-5.3, 4.4, 0], 96),
        "hump":    (lambda: P.speed_hump(3.6, 0.52), [12.6, 5.0, 0], 26),
        "facade":  (lambda: P.facade(24.0, 5.2), [8.5, 19.13, 0], YARD_ANG),
    }


def at(dx, dy, ang=YARD_ANG, base=YARD_BASE):
    """A point `dx` along the building line and `dy` off it — the yard's own coordinates.
    The facade runs at ANG, so laying its clutter out in world x/y by hand is a way to get
    everything subtly off the wall."""
    c, s = math.cos(math.radians(ang)), math.sin(math.radians(ang))
    return [base[0] + dx * c - dy * s, base[1] + dx * s + dy * c]


def slab(p, x0, x1, y0, y1, z, t, material):
    """A flat painted patch lying ON the ground, spanning [z, z+t] — not straddling z, which
    buries half of it in the slab and z-fights the rest."""
    p.place(P.box(x1 - x0, y1 - y0, t), at=[(x0 + x1) / 2, (y0 + y1) / 2, z + t / 2],
            material=material)


# ---- the painted forecourt ------------------------------------------------------ #
# The dominant graphic, rebuilt in the SOLVED camera's frame by unprojecting the photo. The
# blue bay's four measured corners are a 3.47 x 6.0 m rectangle at world x[0,3.47] y[0,6];
# the terracotta bays, the road and the building line were unprojected the same way with
# solve.ground_point. Nothing here is eyeballed against a render -- the camera put every
# edge where it saw it, and a projection overlay on the photo confirmed the fit.
def forecourt():
    p = MeshProgram()
    # the damp concrete apron: ONE textured slab. Aggregate, hairline cracks and dark
    # oil/water staining all live in the map; the wet sheen is the map's low-roughness
    # pools mirroring the sky, not a flat near-black fill.
    p.place(P.box(44, 48, 0.4), at=[6, 8, -0.2], material=APRON)
    # There is no asphalt band. A luminance profile along world y (photo against render, at
    # x=6) settles it: the photograph runs 0.60, 0.68, 0.81, 0.66 out to y=12 and 0.67, 0.62,
    # 0.76, 0.73 beyond it — bright wet CONCRETE the whole way. The only dark things are two
    # thin lines, the drain at y=8.0 (0.025) and a gutter at y=12.8 (0.227). A five-metre
    # slab of tarmac had been laid across the middle of it, which is why the render's
    # distance went dark exactly where the photograph's goes bright.
    # the shop frontage: a separate, lighter pour, starting just beyond the gutter
    p.place(P.box(42, 8.0, 0.02), at=[8.5, 16.6, 0.007], rotate=[0, 0, YARD_ANG],
            material=APRON_LT)
    p.place(P.box(40, 0.55, 0.02), at=[7, 12.9, 0.011], rotate=[0, 0, ANG], material=ROAD)
    # Saw-cut joints, but ONE way only. A poured apron is cast in STRIPS and the joints run
    # with them; a scan across the bare concrete right of the bays found no periodic
    # structure crossing it at all — the dips there are stains, not joints — and the full
    # grid that used to be here was reading as floor tiles. Laid below the bays (z
    # 0.001..0.005) so the paint covers them, as it does.
    JOINT = mat((0.128, 0.131, 0.133), 0.0, 0.58)
    for k in range(-2, 7):
        p.place(P.box(0.020, 46, 0.004), at=[-7.6 + k * 5.4, 8, 0.003], material=JOINT)
    # MEASURED, not placed. The painted regions were segmented out of the photograph by
    # colour and their corners unprojected through the solved camera (forecourt/place.py),
    # which found the layout wrong in a way no render was going to reveal: the big
    # terracotta bay is not to the RIGHT of the blue one, it is IN FRONT of it, in the same
    # lane, filling the bottom of the frame. What sat to the right was bare concrete.
    #
    #   blue     x[0.11, 3.36]  y[0.09, 5.89]   <- the solve's own bay, reproduced exactly
    #   near     x[0.13, 3.42]  y[<-5.5, -0.04] <- clipped by the frame; runs to about -6.6
    #   near-r   x[3.59, 4.05+] y[<-5.4, -0.11] <- its neighbour, mostly off-frame right
    #   far      x[0, 3.47]     y[6.2, 7.9]     <- scanned up a column: orange 6.47..7.63
    #   slate    x[-1.16,-0.11] y[-2.28, 3.29]  <- a strip left of the blue bay, unmodelled
    #   left-t   x[-1.15,-0.10] y[-4.4, -2.4]   <- the same strip, terracotta nearer in
    bays = [(0, 3.47, 0, 6, BAY_BLUE),
            (0, 3.47, -6.6, -0.12, BAY_ORNG),
            (3.62, 7.5, -6.6, -0.12, BAY_ORNG),
            (0, 3.47, 6.2, 7.9, BAY_ORNG),
            (-1.20, -0.12, -2.4, 3.35, BAY_SLATE),
            (-1.20, -0.12, -4.6, -2.45, BAY_ORNG)]
    LW = 0.18
    for x0, x1, y0, y1, m in bays:
        slab(p, x0, x1, y0, y1, 0.004, 0.006, m)
    # each boundary once: (x, y0, y1) for the lines running away from the camera, and
    # (y, x0, x1) for the ones running across it
    for x, y0, y1 in [(-1.20, -4.6, 3.35), (-0.06, -6.6, 7.9), (3.55, -6.6, 7.9),
                      (7.50, -6.6, -0.12)]:
        slab(p, x - LW / 2, x + LW / 2, y0, y1, 0.006, 0.006, LINE_W)
    for y, x0, x1 in [(0.015, -0.06, 7.5), (6.05, -0.06, 3.55), (7.90, -0.06, 3.55),
                      (-2.42, -1.20, -0.06), (-4.60, -1.20, -0.06)]:
        slab(p, x0, x1, y - LW / 2, y + LW / 2, 0.006, 0.006, LINE_W)
    # The white RING painted around the island. Both discs are painted INSIDE the
    # sub-program: a material on the outer `place` would repaint every face it carries,
    # including the concrete infill, and the ring would come out as a solid white pancake —
    # which is exactly what it had been doing.
    ring = MeshProgram()
    ring.place(MeshProgram().cylinder(sides=80, radius=1.42, height=0.006), material=LINE_W)
    ring.place(MeshProgram().cylinder(sides=80, radius=1.24, height=0.014), material=APRON)
    # the ring is painted around the island, so it moves with it
    p.place(ring, at=[PLACEMENTS()['island'][1][0] - 0.08,
                      PLACEMENTS()['island'][1][1] - 1.39, 0.004])
    p.place(P.box(0.15, 2.0, 0.006), at=[9.0, 9.4, 0.004], rotate=[0, 0, ANG], material=YELLOWP)
    for s in (-1, 1):
        p.place(P.box(0.15, 0.95, 0.006), at=[9.0 + s * 0.28, 8.55, 0.004],
                rotate=[0, 0, ANG + s * 40], material=YELLOWP)
    p.place(P.box(20, 0.14, 0.006), at=[8.0, 13.70, 0.004], rotate=[0, 0, ANG],
            material=YELLOWP)
    # Scanning a column up the photograph put the drain at world y = 8.0 (black at 8.04,
    # concrete again by 8.47). It had been sitting at 11.3, three metres into the road.
    # WHERE THE WATER IS. Mapping wetness on a two-metre world grid, photograph against
    # render: the far band by the road reads 0.79 against 0.42 here and the blue bay reads
    # 0.17 against 0.55 — this scene was wet in the wrong places by as much as it was wet at
    # all. A texture can only make a surface uniformly damp; water collects where the slab
    # falls away, so it belongs in the LAYOUT.
    for i, (x0, y0, w, d, rot, m) in enumerate([(9.0, 13.2, 15.0, 3.6, ANG, PUDDLE),
                                               (2.0, 13.0, 9.0, 2.8, ANG, PUDDLE),
                                               (11.0, 7.2, 5.4, 3.2, ANG + 8, PUDDLE_L),
                                               (7.2, 3.0, 3.8, 2.8, ANG - 6, PUDDLE_L),
                                               (-1.0, 5.2, 2.6, 3.2, 4, PUDDLE_L)]):
        # TWO TIERS. A puddle does not end at a line: outside the standing water there is a
        # margin of damp concrete where the film got too thin to hold, and that margin is
        # what stops the edge reading as a cut-out. Damp halo first, water inside it.
        for scale, mm, zz in ((1.65, DAMP2, 0.006), (1.28, DAMP, 0.009)):
            p.place(P.puddle(w * scale, d * scale, seed=i * 13 + 41 + int(scale * 10), lobes=4),
                    at=[x0, y0, zz], rotate=[0, 0, rot], material=mm)
        p.place(P.puddle(w, d, seed=i * 13 + 3, lobes=4), at=[x0, y0, 0.013],
                rotate=[0, 0, rot], material=m)
    p.place(P.drain_channel(15.0), at=[6.0, 8.05, 0], rotate=[0, 0, ANG])
    return p


# ---- the yard behind ------------------------------------------------------------ #
def yard():
    """The building line across the back and its clutter, placed by unprojecting the
    building base line (world y ~ 20, yawed by ANG) and the road front."""
    p = MeshProgram()
    fb, fpos, fyaw = PLACEMENTS()["facade"]
    p.place(fb(), at=fpos, rotate=[0, 0, fyaw], mark="facade")
    for dx in (-8.4, -4.3, -0.2, 4.6, 8.7):                       # the workshop bays
        c = at(dx, -0.22)
        p.place(P.roller_shutter(3.4, 3.2), at=[c[0], c[1], 0], rotate=[0, 0, YARD_ANG],
                mark="shutters")
    for dx in (-6.35, 2.2, 6.65):                                 # the piers between them
        c = at(dx, -0.24)
        p.place(P.tiled_pilaster(0.62, 4.6), at=[c[0], c[1], 0], rotate=[0, 0, YARD_ANG],
                mark="piers")
    c = at(11.6, -0.30)                                           # an open doorway
    # An open doorway is a HOLE into an unlit room, and it has to be modelled as one: a dark
    # PANEL on a wall still faces the sky and still comes back at 0.25. A recess three
    # metres deep occludes itself and lands where the photograph's does, near black. The
    # photograph puts 4.7% of its pixels below 0.10; this render had 0.1%, and the missing
    # blacks are all openings like this one.
    for dxx, hh in ((11.6, 2.6), (5.6, 2.2)):
        cc = at(dxx, 1.30)
        p.place(P.box(2.1, 3.0, hh), at=[cc[0], cc[1], hh / 2], rotate=[0, 0, YARD_ANG],
                material=mat((0.012, 0.013, 0.014), 0.0, 0.8))
    c = at(-2.4, -0.55)
    p.place(P.hanging_banner(1.30, 1.76, WASH_F), at=[c[0], c[1], 3.15], rotate=[0, 0, YARD_ANG],
            mark="banners")
    c = at(3.9, -0.55)
    p.place(P.hanging_banner(0.55, 2.75, PROMO_F), at=[c[0], c[1], 3.30], rotate=[0, 0, YARD_ANG],
            mark="banners")
    # the vehicles and the ground clutter along the wall
    L = PLACEMENTS()
    for name in ("van", "hump", "suv"):
        build, pos, yaw = L[name]          # not `at`: that is this module's yard-coord helper
        p.place(build(), at=pos, rotate=[0, 0, yaw], mark=name)
    # measured: eight bollards from world x 0.3 to 22.2, all about 1.5 m off the wall
    for dx, dy in [(-8.0, -2.05), (-4.7, -1.85), (-1.6, -1.55), (4.5, -1.35),
                   (6.6, -1.25), (9.4, -1.15), (10.3, -1.10), (13.9, -1.07)]:
        c = at(dx, dy)
        p.place(P.bollard(), at=[c[0], c[1], 0], mark="bollards")
    # The clutter a working yard accumulates. Placed in the facade's own coordinates (dx
    # along the wall, dy off it) so it sits ON the wall rather than near it, and grouped
    # around the open bays the way it is in the photograph — around the doors people
    # actually use, not spread evenly along the frontage.
    for dx, dy, rot in [(9.6, -1.30, 12), (10.3, -1.15, -40), (10.1, -1.75, 70),
                        (3.1, -1.20, 25), (12.2, -1.35, -8), (9.2, -1.05, -22),
                        (11.4, -1.20, 48), (11.8, -1.60, -15), (2.6, -1.55, -34),
                        (13.0, -1.10, 20)]:
        q = at(dx, dy)
        p.place(P.carton(0.52 + 0.10 * (dx % 3), 0.40, 0.34 + 0.05 * (dx % 2)),
                at=[q[0], q[1], 0], rotate=[0, 0, YARD_ANG + rot])
    for dx, dy in [(10.0, -1.55), (9.7, -2.05)]:
        q = at(dx, dy)
        p.place(P.carton(0.46, 0.36, 0.30), at=[q[0], q[1], 0.34], rotate=[0, 0, YARD_ANG - 18])
    q = at(6.4, -1.55)
    p.place(P.scooter(), at=[q[0], q[1], 0], rotate=[0, 0, YARD_ANG + 96])
    for dx, dy, n in [(12.9, -1.15, 3), (13.6, -1.45, 2), (6.9, -1.20, 4)]:
        q = at(dx, dy)
        p.place(P.tyre_stack(n), at=[q[0], q[1], 0])
    for dx, dy in [(1.4, -1.35), (7.8, -1.10)]:                # more bins along the wall
        q = at(dx, dy)
        p.place(P.bin_(0.46, 0.40, 0.66), at=[q[0], q[1], 0], rotate=[0, 0, YARD_ANG + 14])
    # the hazard tape strung along the bollard line
    tape = [at(dx, dy) for dx, dy in [(-8.0, -2.05), (-4.7, -1.85), (-1.6, -1.55),
                                      (4.5, -1.35), (6.6, -1.25), (9.4, -1.15)]]
    p.place(P.tape_run([[a0, b0, 0.62] for a0, b0 in tape]))
    for dx, dy, kind in [(-7.6, -0.95, "bin"), (-7.0, -0.9, "broom"), (-6.8, -0.85, "broom")]:
        c = at(dx, dy)
        if kind == "bin":
            p.place(P.bin_(), at=[c[0], c[1], 0], rotate=[0, 0, YARD_ANG])
        else:
            p.place(P.broom(1.4), at=[c[0], c[1], 0], rotate=[0, 9, YARD_ANG])
    for dx, dy, top in [(-5.2, -1.15, (0.13, 0.115, 0.105)), (0.9, -1.05, (0.32, 0.30, 0.27)),
                        (1.5, -1.15, (0.22, 0.17, 0.12))]:
        c = at(dx, dy)
        p.place(P.person(1.70, top), at=[c[0], c[1], 0], rotate=[0, 0, YARD_ANG + 150],
                mark="people")
    for k, (dx, dy) in enumerate([(-9.6, -1.05), (-9.0, -1.15)]):  # the 小心地滑 A-frames
        c = at(dx, dy)
        p.place(P.wet_floor_sign(), at=[c[0], c[1], 0], rotate=[0, 0, YARD_ANG + 8 * k],
                mark="wetsign")
    p.place(P.bollard(0.86), at=[13.6, 8.6, 0])
    # BLUE, not orange. The scorecard put the banners at cast +0.340, the largest colour
    # error on the board by a factor of three, and it was not a colour error at all: the
    # photograph has a big blue promotional banner at this corner and this scene had the
    # workshop's orange 修保养 sign there. The orange one belongs on a pier, where
    # tiled_pilaster already carries it.
    p.place(P.hanging_banner(0.62, 2.70, PROMO_F), at=[-5.6, 11.0, 1.75], rotate=[0, 0, -6],
            mark="banners")
    return p


def scene():
    p = MeshProgram()
    # Every top-level object is MARKED, which makes the scene measurable rather than just
    # renderable: mirage_render --ids turns these tags into a per-pixel object id, so
    # photomatch.chamfer_per_object can score each against the photo separately. That
    # per-object score is the loss that will POLISH this scene; measurement -- the solved
    # camera and the unprojected layout -- is what landed it in the basin first, which no
    # loss could do: eleven cameras once all scored 14-15 px because nothing downhill led
    # to the true camera, 22 deg of yaw and half the fov away.
    L = PLACEMENTS()
    p.place(forecourt(), mark="forecourt")
    for name in ("island", "rail"):
        build, pos, yaw = L[name]
        p.place(build(), at=pos, rotate=[0, 0, yaw], mark=name)
    p.place(P.jerrycan(), at=[L["island"][1][0] - 0.80, L["island"][1][1] - 1.95, 0])
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

# The scene's parts, FINEST FIRST — a face takes the first of these tags it carries, so
# listing "sign" before "island" scores the sign on its own instead of dissolving it into
# the island. This list is the scorecard's vocabulary: anything not named here lands in the
# coarse bucket and cannot be blamed for anything.
ID_TAGS = ["sign", "dispenser", "hoses", "firebox", "bucket", "plinth", "column", "rail",
           "van", "suv", "shutters", "bollards", "hump", "banners", "piers", "people",
           "wetsign", "facade", "island", "yard", "forecourt"]


def render(prog, out, spp, w, h, extra=()):
    OUT.mkdir(parents=True, exist_ok=True)
    js = OUT / (out + ".json")
    js.write_text(prog.to_json())
    ppm = OUT / (out + ".ppm")
    # Resolved HERE, not at import: scoring a render (compare/critique) needs this module
    # but no renderer, and a remote-only checkout must not refuse to let you read a
    # scorecard just because it will not let you make a picture.
    subprocess.run([str(default_render()), "--oplog", str(js), "--out", str(ppm),
                    "--spp", str(spp), "--w", str(w), "--h", str(h), "--threads", THREADS,
                    "--cam-eye", *[str(v) for v in CAM_EYE],
                    "--cam-target", *[str(v) for v in CAM_TGT],
                    "--cam-fov", str(CAM_FOV), *extra], check=True)
    from PIL import Image
    import numpy as np
    from mirage import sensor
    png = OUT / (out + ".png")
    # THE CAMERA THAT TOOK THE PICTURE. A path tracer hands back an image with no sensor in
    # front of it, and measured against the reference that shows up twice: the render's
    # noise floor is 0.011 against the photograph's 0.029 (too clean), and its CHROMA
    # high-frequency is three times the photograph's, because a path tracer's sampling noise
    # is coloured and no camera's is — every camera resolves colour far more coarsely than
    # luma. Both numbers were solved with sensor.match against this reference (see
    # forecourt/place.py --measure) and are pinned here so the render is reproducible
    # without it. Grain is calibrated on a MEDIAN ABSOLUTE DEVIATION, not a standard
    # deviation: on a textured photograph the detail dominates the sd, and matching that
    # buries the image in film grain, which the first attempt duly did.
    raw = np.asarray(Image.open(ppm).convert("RGB"), float) / 255.0
    # THE CAMERA'S TRANSFER CURVE, fitted to the reference with sensor.fit_tone: a black
    # point, a white point and a gamma, three numbers, no lookup table. The render had no
    # blacks and no highlights (p1 0.171 and p99 0.780 against the photograph's 0.038 and
    # 0.980; 0.1% of pixels at each end against 4.7% and 3.4%) and NONE of that is a scene
    # property. Raising the firefly clamp helped a little, and nothing else in the renderer
    # could: brightening the sky or opening the exposure scales diffuse and specular
    # together and leaves their RATIO, which is what sets the spread, exactly where it was.
    # The reference is a security camera whose auto-exposure is set for its shadows — it
    # lifts the low end and lets the high end clip. Fitted: p1 0.039, p50 0.468, p99 0.966.
    raw = sensor.tone_curve(raw, black=0.1331, white=0.7880, gamma=1.1918)
    img = sensor.apply(raw,
                       # re-solved after the hard-edged stains went in: the maps now
                       # supply detail the grain used to be standing in for, so it
                       # comes down (floor was overshooting 0.0356 against 0.0289),
                       # and the chroma blur was over-correcting past the photograph.
                       # grain solved AFTER the whole chain, not before it: sensor.match
                       # measures the render as handed to it, and the saturation gain and
                       # the chroma blur both move the noise floor afterwards. Solved
                       # 0.0208, which landed 0.0366 against the photograph's 0.0289;
                       # 0.0134 lands it. Saturation 1.275: every consumer pipeline
                       # pushes colour, and the frame was a flat 22% short of the
                       # reference (0.109 against 0.140) with every material measured
                       # right, which is a missing gain and not N wrong materials.
                       grain=0.0134, chroma_blur=1.0, saturation=1.275)
    Image.fromarray((img * 255 + 0.5).astype(np.uint8)).save(png)
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


def critique(png):
    """Score the render against the photograph, per object, worst first.

    This is the answer to a question that kept being asked by a human: "which bit is wrong?"
    A render is re-rendered dozens of times and the eye adapts to it after the third — every
    fault this scene has shipped with (an apron twice as light as the photo, a yard paved
    black, a van that was the wrong kind of van, ten objects that were painted boxes) was
    visible in the picture the whole time and still needed somebody to say so. The
    scorecard says so instead, in a table sorted by how much it matters.
    """
    from mirage import critique as C
    from mirage.photomatch import read_ids
    ids_path = OUT / "hero_ids.pgm"
    if not ids_path.exists():
        print("(no ids AOV — run without --preview, or with --critique, to write one)")
        return
    ren, ref = C.load_pair(png, REF)
    rows = C.scorecard(ren, ref, read_ids(ids_path), names=ID_TAGS)
    print()
    print(C.report(rows))
    print("wrote", C.plate(ren, read_ids(ids_path), rows, OUT / "scorecard.png", names=ID_TAGS))
    return rows


def main():
    preview = "--preview" in sys.argv
    p = scene()
    m = p.build()
    print(f"forecourt: {len(m.verts):,} verts  {len(m.faces):,} faces  ({len(p.ops)} top-level ops)")
    spp = 40 if preview else int(os.environ.get("MIRAGE_SPP", "260"))
    # An overcast wet morning: soft high sun, a bright sky fill for the wet surfaces to mirror,
    # no hard key. Env is turned up (0.86) because on this shot the reflected sky IS the light on
    # the floor -- the wet materials are near-black diffuse and read only through what they
    # reflect. Exposure 1.35 sits the concrete where the reference's is; --clamp stops a hot
    # specular sample on the near-mirror puddles from leaving a firefly the denoiser can't fix.
    # --sky-tint warms the sky fill. The scorecard is what found this: EVERY object came
    # back with a cool cast, all seventeen, which is not seventeen colour errors but one —
    # the sky is the only thing lighting most of this scene, and it was blue.
    extra = ["--sun", "0.12", "--env", "0.74", "--exposure", "1.35",
             # --clamp 1.5 was capping the picture. It exists to stop a rare hot
             # specular sample leaving a firefly, but on a WET scene the bright
             # speculars are not fireflies, they are the subject: the photograph puts
             # 3.4% of its pixels above 0.90 and this render put 0.0% there, because
             # every one of them was clipped on the way out. At 260 spp with the
             # denoiser on, 6.0 is enough guard.
             "--clamp", "6.0",
             "--sun-dir", "0.25", "0.55", "0.80",
             # An OVERCAST dome, not a clear-day gradient. --sky-flat was added for this
             # scene: the residual colour error after tinting sat entirely on the METALS
             # -- plinth, bucket, bollards, all at cast -0.25 while the diffuse surfaces
             # were neutral -- because a mirror sees the zenith, and the built-in zenith
             # is the darkest, bluest part of a clear sky. Flattening the dome is what a
             # tint cannot do.
             # SOLVED, not nudged: the flattened dome is (0.255, 0.330, 0.480) as authored,
             # chromaticity r-b = -0.211, and every mirror in the scene was returning
             # exactly that. The tint below takes it to (0.425, 0.412, 0.398) -- overcast
             # white, a touch warm -- and --env drops by the same luminance ratio so the
             # exposure that was already measured right does not move.
             # --sky-flat 0.94, not 0.80. The diffuse surfaces have been neutral for several
             # rounds while the METALS -- plinth, bucket, chrome -- keep coming back cool,
             # and that is the 20% of clear-day gradient left in the dome: a mirror sees
             # the zenith specifically, where the gradient is bluest, while a diffuse
             # surface integrates the whole thing and never notices.
             "--sky-tint", "1.560", "1.240", "0.900", "--sky-flat", "0.94",
             # Aerial perspective, and a hypothesis that was HALF WRONG. The table flagged
             # every distant DARK object as too dark while distant BRIGHT ones were fine,
             # which looks exactly like missing haze -- so haze went in at 130 m, and total
             # severity got WORSE (2.29 -> 2.59): the shutters, the piers and the apron all
             # went from right to too light. The disproof is in the same table that
             # suggested it: the facade and the shutters sit at the SAME distance and wanted
             # +0.15 and +0.015, which no function of depth can deliver. So those four dark
             # objects were four dark albedos after all. Haze stays, at a value small enough
             # to be the real thing rather than a fudge for something else.
             # 320 m, and this number is MEASURED, not reasoned. The argument for deleting
             # haze entirely was good: at 320 the shutters still drifted 0.361 -> 0.455, and
             # the four dark objects had turned out to be dark albedos rather than distance.
             # The A/B says otherwise -- haze off scores 2.678, haze at 320 scores 2.337 --
             # because the albedo lift and the haze were fitted together and removing one
             # leaves the other stranded. Twice now the table has overruled the reasoning:
             # once killing haze at 130, once refusing to let it go at 320. Both times the
             # render was cheaper than the argument.
             "--haze", "320",
             # denoise 4 at 260 spp was eating the apron's grain: the scorecard read detail
             # 0.18 and barely moved when the map got twice the staining, because the filter
             # was removing exactly what the map was adding. 2 keeps it.
             # MIRAGE_DENOISE overrides so the filter can be A/B'd without editing the case.
             # It is a real suspect for the ground's missing contrast: an edge-avoiding filter
             # keeps EDGES, and low-amplitude texture is exactly what it cannot tell from noise.
             # --smooth-angle 55, not the 30 default. A lofted vehicle flank is one curved
             # surface whose ear-clipped triangles meet at more than 30 degrees, so the
             # tracer was flat-shading it and the body came out banded in diagonal facets —
             # a shading choice, not geometry. The bodywork is smooth; say so.
             "--smooth-angle", "55",
             "--denoise", os.environ.get("MIRAGE_DENOISE", "2")]
    # The id AOV comes out of the SAME trace as the beauty pass, so the scorecard's masks
    # are the pixels it is scoring and not a re-render that drifted.
    extra += ["--ids", str(OUT / "hero_ids.pgm"), "--id-tags", ",".join(ID_TAGS)]
    png = render(p, "hero", spp, 1600, 900, extra=extra)
    print("wrote", png)
    if REF.exists():
        print("wrote", compare(png))
        critique(png)
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
