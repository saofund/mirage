"""The forecourt's parts, modelled one at a time.

Every object in this scene used to be a box. A box is a fine *placeholder* — it gets the
silhouette and the layout right, which is what the camera solve needed — but it is not a
thing. What separates a real object from a box, at the scale this camera sees, is a short
and very specific list:

* **a chamfer on every edge.** Real manufactured edges are broken. Under a bright overcast
  sky a chamfer is a thin bright line along every edge, and its absence is most of what
  "CG" means on a painted-metal object. Hence :func:`cbox`, used almost everywhere.
* **printed graphics.** A red cabinet is a red box; a red cabinet with 灭火器箱 on it is a
  fire cabinet. Carried by decals (see ``materials.face_decal``).
* **the small hardware.** Hinges, latches, pilasters, bolt heads, the rim of a grate. Each
  is tiny; together they are the difference between 4 faces and a machine.
* **curves that are actually curved.** The hoses are swept along drape curves
  (:func:`drape` + the ``sweep`` operator), not assembled from circular arcs — a hose
  hangs under its own weight and a torus section does not.

Conventions, so the layout can place these without guessing:

* metres; **z = 0 is the ground**; a part's origin is its footprint centre.
* a part's **front faces -y** — it is the face you read — so the layout aims it with a yaw.
  Vehicles are the exception: they run along **+x** (nose at +x), the way a car is driven.
"""
import math

from mirage.meshlang import MeshProgram

from .materials import (
    BLACK, BODY_WH, BUCKET_F, CHROME, CLAD, CONCRETE, FIREBOX_F, GALV, GLASS, HOSE, KERB,
    LAMP, NAVY, ORANGE_S, PANEL_WH, PLATE_F, PROMO_F, PUMP_BL, PUMP_BL_D, RED, RED_D,
    WET_STEEL,
    REPAIR_F, RUBBER, SEAM, SHUTTER, SHUTTER_D, SIGN_FACE, STEEL, TAIL_RED, WALL_TILE,
    WASH_F, WETSIGN_F, WHITE, YELLOW, mat,
)

# --------------------------------------------------------------------------- #
# primitives
# --------------------------------------------------------------------------- #
def box(sx, sy, sz):
    """An axis-aligned box of the given SIZE (the cube primitive is unit, centred)."""
    return MeshProgram().cube(size=1.0).scale({"by": "all"}, [sx, sy, sz])


_CBOX_CACHE: dict = {}


def cbox(sx, sy, sz, c=0.010):
    """A CHAMFERED box — the workhorse of this kit.

    The chamfer is clamped to the part's smallest dimension, and then VERIFIED: on a small
    box with an awkward aspect the edge bevel can fold a face onto itself, and the kernel
    correctly refuses to hand back an invalid mesh. Rather than let one 26 mm latch take a
    whole scene down, the width steps down a short ladder and, if none of it survives, the
    part gets a plain box. A missing chamfer on a latch is a rounding error; a scene that
    will not build is not."""
    mn = min(abs(sx), abs(sy), abs(sz))
    key = (round(sx, 5), round(sy, 5), round(sz, 5), round(c, 5))
    if key in _CBOX_CACHE:
        w = _CBOX_CACHE[key]
        return box(sx, sy, sz).edge_bevel({"by": "all"}, width=w) if w else box(sx, sy, sz)
    for w in (min(c, 0.30 * mn), 0.24 * mn, 0.15 * mn, 0.0):
        if not w:
            break
        try:
            box(sx, sy, sz).edge_bevel({"by": "all"}, width=w).build()
            _CBOX_CACHE[key] = w
            return box(sx, sy, sz).edge_bevel({"by": "all"}, width=w)
        except Exception:
            continue
    _CBOX_CACHE[key] = 0.0
    return box(sx, sy, sz)


def cyl(r, h, sides=20):
    return MeshProgram().cylinder(sides=sides, radius=r, height=h)


def prism(poly, y0, y1):
    """A 2-D outline in the xz plane, extruded between two y — a vehicle's side
    silhouette, a kerb section, an A-frame's leg. Windings are made consistent (and the
    polygon flipped if it was given clockwise) so the solid is manifold."""
    poly = [list(p) for p in poly]
    area = sum((poly[(i + 1) % len(poly)][0] - poly[i][0]) *
               (poly[(i + 1) % len(poly)][1] + poly[i][1]) for i in range(len(poly)))
    if area < 0:
        poly = poly[::-1]
    n = len(poly)
    verts = [[x, y0, z] for x, z in poly] + [[x, y1, z] for x, z in poly]
    faces = [list(reversed(range(n))), [n + i for i in range(n)]]
    faces += [[i, (i + 1) % n, n + (i + 1) % n, n + i] for i in range(n)]
    return MeshProgram().mesh(verts=verts, faces=faces)


def pane(p0, p1, thick, y0, y1):
    """A thin sheet lying exactly ON a body's slope, between two points of its side
    profile. Glazing has to be built this way on a vehicle: a rotated box placed by eye
    ends up hanging off the nose, and at thirty metres a floating windscreen is the one
    thing that reads instantly as wrong."""
    dx, dz = p1[0] - p0[0], p1[1] - p0[1]
    L = math.hypot(dx, dz)
    nx, nz = dz / L, -dx / L                          # outward normal of the sloped face
    return prism([[p0[0], p0[1]], [p1[0], p1[1]],
                  [p1[0] + nx * thick, p1[1] + nz * thick],
                  [p0[0] + nx * thick, p0[1] + nz * thick]], y0, y1)


def circle(r, n=12):
    return [[r * math.cos(2 * math.pi * k / n), r * math.sin(2 * math.pi * k / n)]
            for k in range(n)]


def tube_along(path, r, sides=10, closed=False):
    """A round tube following a 3-D path — the `sweep` operator with a circular profile.
    This is how every hose, handle and bent rail in the scene is made."""
    return (MeshProgram().profile(points=circle(r, sides), plane="xy", closed=True)
            .sweep(path=path, closed=closed))


def lathe(points, steps=32, angle=360.0, closed=True):
    """Turn a closed section about z (the lathe): buckets, bollards, bottles."""
    return MeshProgram().profile(points=points, plane="xz", closed=closed).spin(
        axis="z", steps=steps, angle=angle)


def drape(a, b, sag=0.5, bulge=(0.0, 0.0, 0.0), n=18):
    """A hanging curve from a to b: the straight chord pulled DOWN by `sag` and pushed out
    by the `bulge` vector, both peaking at mid-span.

    The bulge is a full 3-vector and not just a sideways nudge, because a fuel hose does
    not hang in the vertical plane through its two ends — it is stiff, so it bows OUT from
    the pump as well as down, and both ends land back on the same cabinet. Without the
    outward bow the five hoses collapse into one vertical bundle, which is exactly what the
    first version of this part looked like."""
    out = []
    for k in range(n + 1):
        t = k / n
        w = 4.0 * t * (1.0 - t)                       # 0 at the ends, 1 in the middle
        out.append([a[i] + (b[i] - a[i]) * t + bulge[i] * w - (sag * w if i == 2 else 0.0)
                    for i in range(3)])
    return out


def catmull(ctrl, n=6):
    """Resample control points into a smooth polyline (Catmull-Rom). The curve passes
    THROUGH its controls, which is what makes it usable for shaping a hose by hand."""
    pts = [ctrl[0]] + [list(c) for c in ctrl] + [ctrl[-1]]
    out = []
    for i in range(len(pts) - 3):
        p0, p1, p2, p3 = pts[i:i + 4]
        for k in range(n):
            t = k / n
            t2, t3 = t * t, t * t * t
            out.append([0.5 * (2 * p1[c] + (-p0[c] + p2[c]) * t
                               + (2 * p0[c] - 5 * p1[c] + 4 * p2[c] - p3[c]) * t2
                               + (-p0[c] + 3 * p1[c] - 3 * p2[c] + p3[c]) * t3)
                        for c in range(3)])
    out.append(list(ctrl[-1]))
    return out


def hose_path(a, b, side, out=0.55, drop=0.95, lean=0.14):
    """One hose's route: out and down from the swivel, round the bottom, and back up
    INBOARD to its holster.

    The two limbs have to take DIFFERENT routes. A symmetric sag between two points that
    are 20 cm apart folds the loop into a hairpin whose limbs lie on top of each other —
    which path-traces into a fat black BAR, not a hose, and no amount of extra sag fixes it
    because sag moves both limbs together. Sending the return leg up inboard of the outward
    one is what opens the loop."""
    return catmull([
        a,
        [a[0] + side * out * 0.62, a[1] - lean * 0.50, a[2] - drop * 0.42],
        [a[0] + side * out, a[1] - lean, a[2] - drop * 0.82],
        [a[0] + side * out * 0.66, a[1] - lean * 0.60, min(a[2], b[2]) - drop],
        [b[0] + side * out * 0.16, b[1] + lean * 0.25, b[2] - drop * 0.55],
        b,
    ], n=6)


def arc(centre, r, a0, a1, plane_u=(1, 0, 0), plane_v=(0, 0, 1), n=12):
    """A circular arc as a path (for bail handles, tube elbows swept as one run)."""
    out = []
    for k in range(n + 1):
        a = math.radians(a0 + (a1 - a0) * k / n)
        out.append([centre[i] + r * (math.cos(a) * plane_u[i] + math.sin(a) * plane_v[i])
                    for i in range(3)])
    return out


def resample(path, step):
    """Re-space a polyline at a fixed arc length, so a path built from a couple of long
    straights and a few short bend samples can be cut into even pieces."""
    out = [list(path[0])]
    carry = 0.0
    for i in range(len(path) - 1):
        a, b = path[i], path[i + 1]
        d = math.dist(a, b)
        if d < 1e-12:
            continue
        t = step - carry
        while t <= d:
            out.append([a[k] + (b[k] - a[k]) * (t / d) for k in range(3)])
            t += step
        carry = (carry + d) % step
    if math.dist(out[-1], path[-1]) > 1e-9:
        out.append(list(path[-1]))
    return out


def banded_tube(path, r, mats, band=0.16, sides=12):
    """Sweep a tube along `path` in alternating painted BANDS.

    The bands have to be separate sweeps because paint is not geometry: a selector asking
    for "the faces in this 15 cm slice" of a single swept tube finds quads whose centroids
    all sit elsewhere. Splitting the path is also what the real thing is — the rail was
    masked and sprayed in sections. Runs overlap by one station so the bands butt without
    a gap where the tube bends."""
    fine = resample(path, band / 5.0)
    p = MeshProgram()
    k = 0
    for i in range(0, len(fine) - 1, 5):
        run = fine[i:i + 6]
        if len(run) >= 2:
            p.place(tube_along(run, r, sides=sides), material=mats[k % len(mats)])
        k += 1
    return p


# --------------------------------------------------------------------------- #
# the dispenser island
# --------------------------------------------------------------------------- #
def grate_plinth(w=1.90, d=1.35, h=0.14):
    """The island's raised base: a concrete kerb carrying a steel grating.

    In the photo this is one of the brightest things in the foreground — the wet grating
    mirrors the sky between its bars — so it is built as bars, not as a grey lid."""
    p = MeshProgram()
    p.place(cbox(w, d, h, 0.018), at=[0, 0, h / 2], material=KERB)
    p.place(cbox(w - 0.05, d - 0.05, 0.02, 0.006), at=[0, 0, h - 0.012], material=CONCRETE)
    for sx, sy, dx, dy in [(w - 0.02, 0.045, 0, (d - 0.045) / 2), (w - 0.02, 0.045, 0, -(d - 0.045) / 2),
                           (0.045, d - 0.02, (w - 0.045) / 2, 0), (0.045, d - 0.02, -(w - 0.045) / 2, 0)]:
        p.place(cbox(sx, sy, 0.030, 0.006), at=[dx, dy, h + 0.005], material=WET_STEEL)  # rim
    n = int((w - 0.13) / 0.048)                                       # the grating's bars
    for i in range(n):
        x = -(w - 0.13) / 2 + (i + 0.5) * (w - 0.13) / n
        p.place(box(0.010, d - 0.10, 0.026), at=[x, 0, h + 0.003], material=WET_STEEL)
    for j in range(7):
        y = -(d - 0.10) / 2 + (j + 0.5) * (d - 0.10) / 7
        p.place(box(w - 0.13, 0.008, 0.014), at=[0, y, h - 0.004], material=WET_STEEL)
    return p


def nozzle():
    """A fuel nozzle: body, spout, trigger guard. Six of them ride in the pump's holsters."""
    p = MeshProgram()
    p.place(cbox(0.062, 0.085, 0.17, 0.008), at=[0, 0, 0], material=BLACK)
    p.place(cbox(0.048, 0.062, 0.05, 0.006), at=[0, 0, 0.10], material=RED_D)
    p.place(cyl(0.013, 0.16, 12), at=[0, -0.005, -0.16], rotate=[10, 0, 0], material=CHROME)
    p.place(cyl(0.019, 0.03, 12), at=[0, -0.003, -0.09], material=STEEL)
    p.place(tube_along(arc([0, -0.045, -0.02], 0.045, -70, 70, (0, 1, 0), (0, 0, 1), 6),
                       0.006, 6), material=BLACK)                     # the trigger guard
    return p


def dispenser():
    """The blue pump: body, pilasters, cap, painted bands, display, keypad, holsters.

    The face the camera sees is deliberately PLAIN — in the reference it is flat blue with
    one orange band and a small white plate, and nothing else. The display, the grade
    buttons and the keypad live on the flank, where the photo puts them; crowding them onto
    the front because they are pleasant to model is how a part stops matching its subject."""
    W, D, H = 1.00, 0.66, 1.46
    p = MeshProgram()
    p.place(cbox(W, D, H, 0.016), at=[0, 0, H / 2], material=PUMP_BL)
    p.place(cbox(W + 0.03, D + 0.03, 0.05, 0.010), at=[0, 0, 0.04], material=PUMP_BL_D)
    p.place(cbox(W + 0.03, D + 0.03, 0.035, 0.010), at=[0, 0, H - 0.01], material=PANEL_WH)
    for s in (-1, 1):                                                 # corner pilasters
        for t in (-1, 1):
            p.place(cbox(0.058, 0.058, H - 0.05, 0.012),
                    at=[s * (W / 2 - 0.012), t * (D / 2 - 0.012), H / 2 - 0.01],
                    material=PANEL_WH)
    p.place(cbox(W + 0.016, D + 0.016, 0.075, 0.010), at=[0, 0, 0.50], material=ORANGE_S)
    for t in (-1, 1):
        p.place(cbox(0.17, 0.012, 0.115, 0.004), at=[0.27, t * (D / 2 + 0.012), 0.50],
                material=PANEL_WH)
    # the working side: a recessed display, the grade buttons, a tilted keypad
    p.place(cbox(0.03, 0.44, 0.40, 0.008), at=[W / 2 - 0.008, 0.02, 1.02], material=PUMP_BL_D)
    p.place(cbox(0.02, 0.38, 0.28, 0.006), at=[W / 2 + 0.004, 0.02, 1.06], material=BLACK)
    p.place(cbox(0.014, 0.33, 0.095, 0.004), at=[W / 2 + 0.012, 0.02, 1.12],
            material=mat((0.06, 0.10, 0.07), 0.0, 0.14))
    for k in range(3):
        p.place(cbox(0.016, 0.085, 0.05, 0.004), at=[W / 2 + 0.012, -0.12 + k * 0.14, 0.90],
                material=[PANEL_WH, YELLOW, RED][k])
    p.place(cbox(0.05, 0.15, 0.12, 0.008), at=[W / 2 + 0.02, 0.16, 0.74],
            rotate=[0, -18, 0], material=PANEL_WH)
    for s in (-1, 1):                                                 # holsters and swivels
        p.place(cbox(0.05, 0.32, 0.10, 0.008), at=[s * (W / 2 + 0.016), -0.03, 1.26],
                material=PUMP_BL_D)
        for k in range(3):
            p.place(cyl(0.026, 0.10, 12), at=[s * (W / 2 + 0.02), -0.11 + k * 0.09, 1.31],
                    rotate=[0, 90, 0], material=STEEL)
        for k in range(2):
            p.place(nozzle(), at=[s * (W / 2 + 0.055), -0.11 + k * 0.16, 1.12],
                    rotate=[14, 0, s * 6])
    return p


def hose_tangle(side=1):
    """The drooping hoses on one flank — the scene's signature foreground detail.

    Five hoses, each on its own open loop (see :func:`hose_path`), with the reach, the drop
    and the anchor spacing walked apart so the loops CROSS instead of nesting. The crossing
    is the whole read: five identical loops look like a wiring diagram, five different ones
    look like a working pump."""
    p = MeshProgram()
    # Walked apart with a JITTER, not a ramp: five loops whose reach grows monotonically
    # nest like a rainbow. Real ones cross, because nobody re-racks a fuel hose tidily.
    jit = [0.00, 0.13, -0.07, 0.17, 0.04]
    for k in range(5):
        t = k / 4.0
        a = [side * 0.54, -0.17 + 0.066 * k, 1.33]
        b = [side * 0.50, 0.02 + 0.034 * k, 1.12 - 0.035 * k]
        p.place(tube_along(hose_path(a, b, side, out=0.44 + 0.16 * t + jit[k],
                                     drop=0.78 + 0.20 * t - jit[k] * 0.5,
                                     lean=0.07 + 0.19 * t), 0.0215, 8), material=HOSE)
        p.place(cyl(0.030, 0.055, 10), at=a, rotate=[0, 90, 0], material=STEEL)
    for k in range(2):                                # the rigid risers behind the loops
        p.place(tube_along([[side * 0.56, 0.26 + 0.08 * k, 0.06],
                            [side * 0.56, 0.26 + 0.08 * k, 1.24]], 0.024, 8), material=HOSE)
    return p


def fire_cabinet():
    """灭火器箱 — the extinguisher cabinet: two doors, a lid, hinges, a latch, and the
    printed front that makes it a fire cabinet rather than a red box."""
    W, D, H = 0.40, 0.28, 0.74
    p = MeshProgram()
    p.place(cbox(W, D, H, 0.010), at=[0, 0, H / 2], material=RED)
    p.place(cbox(W - 0.02, 0.012, H - 0.04, 0.004), at=[0, -D / 2 - 0.004, H / 2],
            material=FIREBOX_F)                                        # the printed doors
    p.place(cbox(W + 0.02, D + 0.02, 0.05, 0.010), at=[0, 0, H - 0.02], material=ORANGE_S)
    p.place(box(W - 0.02, 0.014, 0.010), at=[0, -D / 2 - 0.008, H * 0.46], material=RED_D)
    for z in (H * 0.22, H * 0.70):                                     # hinges and the latch
        for s in (-1, 1):
            p.place(cbox(0.020, 0.03, 0.055, 0.004), at=[s * (W / 2 - 0.012), -D / 2 - 0.010, z],
                    material=STEEL)
    p.place(cbox(0.026, 0.02, 0.07, 0.004), at=[0, -D / 2 - 0.014, H * 0.46], material=STEEL)
    p.place(cbox(0.14, 0.11, 0.15, 0.008), at=[0.02, -0.01, H + 0.075], material=WHITE)
    p.place(cbox(0.10, 0.075, 0.03, 0.006), at=[0.02, -0.01, H + 0.16],
            material=mat((0.55, 0.56, 0.58), 0.0, 0.30))               # the glove box on top
    return p


def fire_bucket():
    """消防桶 — a tapered galvanised bucket, turned on the lathe, with a rolled rim, a bail
    handle swept over the top, and its stencil."""
    p = MeshProgram()
    # The stencil goes on the BUCKET, not on a card in front of it: the decal is projected,
    # so it wraps the turned body and foreshortens round the sides the way paint does.
    p.place(lathe([[0.005, 0.0], [0.115, 0.0], [0.150, 0.29], [0.160, 0.315],
                   [0.150, 0.325], [0.142, 0.312], [0.108, 0.014], [0.005, 0.014]], steps=40),
            material=BUCKET_F)
    p.place(lathe([[0.150, 0.305], [0.163, 0.312], [0.157, 0.325], [0.145, 0.320]], steps=40),
            material=GALV)                                             # the rolled rim
    p.place(tube_along(arc([0, 0, 0.30], 0.155, 6, 174, (1, 0, 0), (0, 0, 1), 12), 0.005, 6),
            material=STEEL)                                            # the bail handle
    return p


def bottle(h=0.115, r=0.032, col=(0.34, 0.20, 0.05)):
    """A small bottle left at the island's foot — the kind of litter that only a real place
    has, and whose absence reads as 'set dressing was skipped'."""
    return (lathe([[0.001, 0.0], [r, 0.006], [r, h * 0.72], [r * 0.55, h * 0.84],
                   [r * 0.42, h], [0.001, h]], steps=16)
            .material({"by": "all"}, color=list(col), metallic=0.0, roughness=0.28))


def jerrycan(h=0.34, r=0.135):
    """The white plastic pail parked at the island's foot. Small, and one of the few things
    in the foreground with a soft silhouette — worth the fifty faces."""
    p = MeshProgram()
    p.place(lathe([[0.001, 0.0], [r * 0.80, 0.0], [r, 0.055], [r, h - 0.06],
                   [r * 0.94, h - 0.02], [r * 0.55, h], [0.001, h]], steps=24),
            material=mat((0.66, 0.665, 0.655), 0.0, 0.30))
    p.place(lathe([[0.001, h - 0.012], [r * 0.98, h - 0.012], [r * 0.92, h + 0.014],
                   [0.001, h + 0.014]], steps=24), material=mat((0.20, 0.21, 0.21), 0.0, 0.35))
    return p


def hazard_rail(width=1.83, height=0.72, r=0.045):
    """The black-and-yellow guard hoop in front of the island.

    A real one is ONE bent tube: up a leg, round a radius, across, round, down. Modelled
    that way — a single path through the bends, swept and banded — instead of straight
    cylinders capped with spheres, which is what made the old one read as plumbing."""
    b = 0.13                                                           # the bend radius
    hw = width / 2
    path = ([[-hw, 0, 0.0], [-hw, 0, height - b]]
            + arc([-hw + b, 0, height - b], b, 180, 90, (1, 0, 0), (0, 0, 1), 5)[1:]
            + arc([hw - b, 0, height - b], b, 90, 0, (1, 0, 0), (0, 0, 1), 5)
            + [[hw, 0, 0.0]])
    p = banded_tube(path, r, [YELLOW, BLACK], band=0.155, sides=12)
    p.place(tube_along([[-hw + 0.02, 0, height * 0.52], [hw - 0.02, 0, height * 0.52]], r * 0.62, 10),
            material=BLACK)                                            # the mid rail
    p.place(tube_along([[0.02, 0, 0.0], [0.02, 0, height * 0.55]], r * 0.72, 10), material=YELLOW)
    for s in (-1, 1):                                                  # the floor flanges
        p.place(cyl(0.070, 0.022, 14), at=[s * hw, 0, 0.011], material=STEEL)
    p.place(cyl(0.060, 0.020, 14), at=[0.02, 0, 0.010], material=STEEL)
    return p


def clad_column(w=1.16, d=0.88, h=8.0):
    """The canopy column: clad panels with a shadow-line seam every 1.2 m and broken
    corners, so the biggest flat object in frame stops being a flat object."""
    p = MeshProgram()
    p.place(cbox(w, d, h, 0.022), at=[0, 0, h / 2], material=CLAD)
    for k in range(1, int(h / 1.2) + 1):                               # panel joints
        p.place(box(w + 0.006, d + 0.006, 0.012), at=[0, 0, k * 1.2], material=SEAM)
    p.place(cbox(w + 0.05, d + 0.05, 0.09, 0.014), at=[0, 0, 0.045], material=SEAM)  # base
    return p


def pump_sign():
    """The lightbox on the column: an aluminium frame, a lit face, and the artwork."""
    W, H, D = 1.00, 2.16, 0.10
    p = MeshProgram()
    p.place(cbox(W, D, H, 0.012), at=[0, 0, 0], material=PANEL_WH)
    p.place(cbox(W - 0.02, 0.02, H - 0.02, 0.006), at=[0, -D / 2 - 0.006, 0], material=SIGN_FACE)
    for s in (-1, 1):                                                  # the frame's extrusions
        p.place(cbox(0.045, D + 0.012, H + 0.01, 0.008), at=[s * (W / 2 - 0.018), 0, 0],
                material=PANEL_WH)
    for s in (-1, 1):
        p.place(cbox(W + 0.01, D + 0.012, 0.045, 0.008), at=[0, 0, s * (H / 2 - 0.018)],
                material=PANEL_WH)
    return p


def island():
    """The dispenser island assembled: plinth, column, sign, pump, hoses, cabinet, bucket.

    Dimensions and positions are the ones the camera solve produced by unprojecting the
    pump footprint, the fire box and the painted circle — only the parts got deeper."""
    Z = 0.14
    p = MeshProgram()
    p.place(grate_plinth(2.06, 1.40, 0.14), mark="plinth")
    p.place(clad_column(), at=[0.02, 0.46, Z], mark="column")
    p.place(pump_sign(), at=[0.02, 0.02, Z + 2.34], mark="sign")
    p.place(dispenser(), at=[0, 0.06, Z], mark="dispenser")
    for s in (-1, 1):
        p.place(hose_tangle(s), at=[0, 0.06, Z])
    p.place(fire_cabinet(), at=[0.02, -0.44, Z], mark="firebox")
    p.place(fire_bucket(), at=[0.52, -0.46, Z])
    p.place(bottle(0.115, 0.032, (0.34, 0.20, 0.05)), at=[-0.34, -0.48, Z])
    p.place(bottle(0.105, 0.030, (0.72, 0.72, 0.70)), at=[-0.20, -0.49, Z])
    p.place(bottle(0.098, 0.028, (0.20, 0.42, 0.16)), at=[-0.27, -0.42, Z])
    return p


# --------------------------------------------------------------------------- #
# the yard
# --------------------------------------------------------------------------- #
def roller_shutter(w=3.0, h=3.1):
    """A ribbed roller shutter: real slats, side guides, a bottom rail.

    Slats are geometry rather than a texture because the light is the point — under an
    overcast sky each slat's upper face catches the sky and its lower face goes dark, and
    that ladder of alternating tone IS what a shutter looks like from across a yard."""
    p = MeshProgram()
    n = int(h / 0.105)
    p.place(box(w + 0.04, 0.05, h), at=[0, 0.03, h / 2], material=SHUTTER_D)
    for k in range(n):
        z = (k + 0.5) * h / n
        p.place(cbox(w, 0.045, h / n * 0.80, 0.007), at=[0, 0, z], material=SHUTTER)
    for s in (-1, 1):                                                  # the guide channels
        p.place(cbox(0.055, 0.09, h, 0.008), at=[s * (w / 2 + 0.03), 0.01, h / 2],
                material=SHUTTER_D)
    p.place(cbox(w + 0.06, 0.075, 0.10, 0.010), at=[0, 0, 0.05], material=SHUTTER_D)
    p.place(cbox(w + 0.10, 0.13, 0.20, 0.014), at=[0, 0.02, h + 0.09], material=SHUTTER_D)
    return p


def facade(length=24.0, h=5.2):
    """The workshop frontage: a tiled wall, a plinth, a lintel band and a parapet."""
    p = MeshProgram()
    p.place(cbox(length, 0.40, h, 0.03), at=[0, 0, h / 2], material=WALL_TILE)
    p.place(cbox(length + 0.10, 0.52, 0.30, 0.02), at=[0, -0.04, 0.15], material=SEAM)
    p.place(cbox(length + 0.06, 0.50, 0.22, 0.02), at=[0, -0.05, 3.55], material=SEAM)
    p.place(cbox(length + 0.14, 0.56, 0.34, 0.03), at=[0, -0.06, h + 0.10], material=WALL_TILE)
    return p


def tiled_pilaster(w=0.55, h=4.4):
    """A tiled pier between two shop bays, carrying the workshop's vertical sign."""
    p = MeshProgram()
    p.place(cbox(w, 0.34, h, 0.02), at=[0, 0, h / 2], material=WALL_TILE)
    p.place(cbox(w * 0.72, 0.06, 1.80, 0.010), at=[0, -0.20, h * 0.62], material=REPAIR_F)
    return p


def bollard(h=0.82, r=0.150):
    """A blue-and-white reflective bollard.

    The bands are turned as FRUSTA that follow the body's taper, not as straight cylinders
    laid over it: a cylinder around a cone pokes through unevenly and comes out as a
    sawtooth ring, which is precisely what the first pass of this part did."""
    def radius(z):                                     # the body's profile, so bands fit it
        t = max(0.0, min(1.0, (z - 0.06) / (h - 0.10)))
        return r * (0.90 - 0.56 * t)

    p = MeshProgram()
    p.place(lathe([[0.001, 0.0], [r, 0.0], [r, 0.045], [r * 0.90, 0.06],
                   [r * 0.34, h - 0.09], [r * 0.26, h - 0.02], [r * 0.13, h],
                   [0.001, h]], steps=24), material=NAVY)
    for z0 in (0.20, 0.40, 0.60):                      # reflective bands
        z1 = z0 + 0.062
        p.place(lathe([[radius(z0) + 0.004, z0], [radius(z1) + 0.004, z1],
                       [0.001, z1], [0.001, z0]], steps=24), material=WHITE)
    p.place(lathe([[0.001, 0.0], [r * 1.20, 0.0], [r * 1.20, 0.030],
                   [r * 0.98, 0.044], [0.001, 0.044]], steps=24), material=BLACK)
    return p


def wet_floor_sign(h=0.62):
    """The folding 小心地滑 A-frame — two hinged panels leaning together, not one yellow
    box. The panels are built centred on their own origin so the printed face rides with
    them through the lean."""
    p = MeshProgram()
    for s in (-1, 1):
        p.place(prism([[-0.17, -h / 2], [0.17, -h / 2], [0.115, h / 2], [-0.115, h / 2]],
                      -0.012, 0.012).edge_bevel({"by": "all"}, width=0.006),
                at=[0, s * 0.10, h / 2], rotate=[s * 9, 0, 0], material=WETSIGN_F)
    p.place(cyl(0.010, 0.14, 8), at=[0, 0, h - 0.015], rotate=[90, 0, 0], material=BLACK)
    return p


def hanging_banner(w, h, material):
    """A printed banner hung off the facade: a panel between a top and a bottom batten.
    Origin at the banner's CENTRE, which is where its artwork is anchored."""
    p = MeshProgram()
    p.place(cbox(w, 0.02, h, 0.005), material=material)
    for s in (-1, 1):
        p.place(cyl(0.014, w + 0.05, 10), at=[0, 0, s * h / 2], rotate=[0, 90, 0],
                material=STEEL)
    return p


def speed_hump(length=3.6, w=0.50, h=0.075):
    """A modular rubber speed hump: chamfered sections with the checker painted on top."""
    p = MeshProgram()
    for k in range(3):
        x = (k - 1) * (length / 3 + 0.01)
        p.place(prism([[-length / 6, 0.0], [length / 6, 0.0], [length / 6 - 0.02, h],
                       [-length / 6 + 0.02, h]], -w / 2, w / 2),
                at=[x, 0, 0], material=BLACK)
    for k in range(int(length / 0.09)):                                # the checker, as paint
        for j in range(int(w / 0.09)):
            if (k + j) % 2:
                continue
            p.place(box(0.088, 0.088, 0.004),
                    at=[-length / 2 + 0.045 + k * 0.09, -w / 2 + 0.045 + j * 0.09, h - 0.002],
                    material=WHITE)
    return p


def drain_channel(length=13.0, w=0.28):
    """The forecourt's drainage run: two kerbs with a slotted cover between them. The kerbs
    have to be a FRAME rather than a slab, or they roof over the grating they contain."""
    p = MeshProgram()
    for s in (-1, 1):
        p.place(cbox(length, 0.055, 0.045, 0.008), at=[0, s * (w / 2 + 0.028), 0.020],
                material=CONCRETE)
    p.place(box(length, w, 0.03), at=[0, 0, 0.006], material=BLACK)
    for k in range(int(length / 0.075)):
        p.place(box(0.048, w, 0.020), at=[-length / 2 + 0.037 + k * 0.075, 0, 0.028],
                material=STEEL)
    return p


def bin_(w=0.52, d=0.42, h=0.72):
    p = MeshProgram()
    p.place(cbox(w, d, h, 0.012), at=[0, 0, h / 2], material=mat((0.16, 0.17, 0.17), 0.0, 0.5))
    p.place(cbox(w + 0.03, d + 0.03, 0.05, 0.010), at=[0, 0, h], material=mat((0.10, 0.11, 0.11), 0.0, 0.5))
    return p


def broom(h=1.35):
    p = MeshProgram()
    p.place(cyl(0.014, h, 8), at=[0, 0, h / 2], material=mat((0.36, 0.26, 0.12), 0.0, 0.6))
    p.place(cbox(0.30, 0.05, 0.10, 0.008), at=[0, 0, 0.05], material=mat((0.30, 0.22, 0.10), 0.0, 0.7))
    return p


def person(h=1.70, top=(0.10, 0.11, 0.13), leg=(0.06, 0.06, 0.07)):
    """A worker across the yard. At forty pixels tall a figure is a silhouette — but an
    empty forecourt at half past nine on a Saturday is its own kind of wrong."""
    p = MeshProgram()
    for s in (-1, 1):
        p.place(cyl(0.058, h * 0.47, 8), at=[s * 0.075, 0, h * 0.235], material=mat(leg, 0.0, 0.6))
    p.place(cbox(0.34, 0.20, h * 0.32, 0.03), at=[0, 0, h * 0.63], material=mat(top, 0.0, 0.62))
    for s in (-1, 1):
        p.place(cyl(0.043, h * 0.30, 8), at=[s * 0.20, 0.01, h * 0.63], rotate=[6, 0, 0],
                material=mat(top, 0.0, 0.62))
    p.place(MeshProgram().uv_sphere(segments=14, rings=9, radius=0.098), at=[0, 0, h * 0.87],
            material=mat((0.22, 0.16, 0.13), 0.0, 0.55))
    return p


# --------------------------------------------------------------------------- #
# vehicles  (these run along +x — nose at +x — not front-to--y)
# --------------------------------------------------------------------------- #
def wheel(r=0.37, width=0.24, hub=(0.55, 0.55, 0.56)):
    """A wheel with a shouldered tyre and a real hub: the flat disc that used to stand in
    for one is the single most obvious tell on a parked vehicle."""
    p = MeshProgram()
    p.place(cyl(r, width, 26), rotate=[90, 0, 0], material=RUBBER)
    for s in (-1, 1):                                                  # the tyre's shoulders
        p.place(cyl(r - 0.035, 0.03, 26), at=[0, s * width / 2, 0], rotate=[90, 0, 0],
                material=RUBBER)
    p.place(cyl(r * 0.60, width * 0.55, 22), at=[0, -width * 0.30, 0], rotate=[90, 0, 0],
            material=mat(hub, 0.6, 0.35))
    p.place(cyl(r * 0.24, width * 0.62, 16), at=[0, -width * 0.36, 0], rotate=[90, 0, 0],
            material=CHROME)
    for k in range(6):                                                 # the hub's lightening holes
        a = 2 * math.pi * k / 6
        p.place(cyl(r * 0.115, 0.02, 10),
                at=[r * 0.40 * math.cos(a), -width * 0.42, r * 0.40 * math.sin(a)],
                rotate=[90, 0, 0], material=BLACK)
    return p


def van():
    """The white light van across the yard: a silhouette extruded from its side section,
    then glazed, bumpered, lit and plated."""
    side = [[-2.65, 0.42], [2.02, 0.42], [2.40, 0.50], [2.62, 0.74], [2.65, 1.10],
            [2.28, 1.22], [2.00, 1.36], [1.52, 2.10], [1.22, 2.26], [-2.56, 2.28],
            [-2.65, 2.06]]
    p = MeshProgram()
    p.place(prism(side, -0.98, 0.98).edge_bevel({"by": "all"}, width=0.035), material=BODY_WH)
    p.place(cbox(4.9, 2.01, 0.14, 0.02), at=[-0.2, 0, 0.50],
            material=mat((0.28, 0.29, 0.30), 0.0, 0.45))               # the rocker moulding
    p.place(cbox(4.6, 2.015, 0.05, 0.012), at=[-0.3, 0, 1.24],
            material=mat((0.60, 0.605, 0.61), 0.0, 0.3))               # the side crease
    # Glass sits just PROUD of the body (3 mm), not bolted on as a slab standing 30 mm off
    # it — at this distance a slab reads as a sticker, which is what the first pass looked like.
    for s in (-1, 1):
        p.place(cbox(1.02, 0.03, 0.58, 0.010), at=[1.30, s * 0.968, 1.72], material=GLASS)
        p.place(cbox(0.70, 0.03, 0.38, 0.010), at=[-2.02, s * 0.55, 1.88], material=GLASS)
    p.place(pane([1.99, 1.40], [1.56, 2.06], 0.02, -0.88, 0.88), material=GLASS)  # windscreen
    p.place(cbox(0.03, 1.60, 0.46, 0.010), at=[-2.665, 0, 1.86], material=GLASS)  # rear doors
    p.place(cbox(0.05, 1.86, 0.06, 0.01), at=[-2.66, 0, 1.30],
            material=mat((0.30, 0.31, 0.32), 0.0, 0.5))                # the door seam
    p.place(cbox(0.16, 1.92, 0.26, 0.02), at=[2.66, 0, 0.72],
            material=mat((0.30, 0.31, 0.32), 0.0, 0.45))
    p.place(cbox(0.14, 1.94, 0.22, 0.02), at=[-2.70, 0, 0.62],
            material=mat((0.30, 0.31, 0.32), 0.0, 0.45))
    for s in (-1, 1):                                                  # lamps
        p.place(cbox(0.10, 0.34, 0.16, 0.014), at=[2.66, s * 0.72, 1.02], material=LAMP)
        p.place(cbox(0.06, 0.13, 0.50, 0.012), at=[-2.70, s * 0.78, 1.12], material=TAIL_RED)
        p.place(cbox(0.13, 0.07, 0.09, 0.010), at=[1.96, s * 1.05, 1.76], material=BLACK)
    p.place(cbox(0.03, 0.44, 0.15, 0.006), at=[-2.73, 0.16, 0.86], material=PLATE_F)
    for x, s in [(1.78, 1), (1.78, -1), (-1.62, 1), (-1.62, -1)]:
        p.place(wheel(), at=[x, s * 0.86, 0.37])
    return p


def suv():
    """A crossover parked at the frame's left edge, built the same way as the van."""
    side = [[-2.30, 0.40], [2.18, 0.40], [2.34, 0.56], [2.30, 0.86], [1.62, 1.02],
            [1.02, 1.52], [-0.55, 1.68], [-1.70, 1.60], [-2.16, 1.20], [-2.30, 0.86]]
    p = MeshProgram()
    p.place(prism(side, -0.93, 0.93).edge_bevel({"by": "all"}, width=0.045), material=BODY_WH)
    for s in (-1, 1):
        for x, w in ((0.30, 0.80), (-0.62, 0.78), (-1.52, 0.52)):      # door glass, per door
            p.place(cbox(w, 0.03, 0.36, 0.010), at=[x, s * 0.918, 1.44], material=GLASS)
        p.place(cbox(0.05, 0.05, 0.30, 0.010), at=[-0.16, s * 0.925, 1.44], material=BODY_WH)
    p.place(pane([1.60, 1.05], [1.06, 1.49], 0.02, -0.83, 0.83), material=GLASS)
    p.place(pane([-1.72, 1.58], [-2.12, 1.24], 0.02, -0.80, 0.80), material=GLASS)
    p.place(cbox(0.16, 1.80, 0.24, 0.02), at=[2.30, 0, 0.62], material=mat((0.26, 0.27, 0.28), 0.0, 0.45))
    for s in (-1, 1):
        p.place(cbox(0.10, 0.30, 0.13, 0.012), at=[2.32, s * 0.66, 0.92], material=LAMP)
    p.place(cbox(0.03, 0.44, 0.15, 0.006), at=[2.38, -0.10, 0.62], material=PLATE_F)
    for x, s in [(1.48, 1), (1.48, -1), (-1.42, 1), (-1.42, -1)]:
        p.place(wheel(0.34, 0.22), at=[x, s * 0.82, 0.34])
    return p
