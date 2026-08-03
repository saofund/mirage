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
    HUMP, LAMP, NAVY, ORANGE_S, PANEL_WH, PLATE_F, PROMO_F, PUMP_BL, PUMP_BL_D, RED, RED_D,
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


def _ear_clip(poly):
    """Triangulate a simple polygon (CCW) by ear clipping.

    The caps of an extruded silhouette CANNOT be left as one n-gon once the outline goes
    concave: the tracer fans an n-gon from its first vertex, and on a concave outline that
    fan lays triangles across the notches — so a wheel arch cut into a van's side comes back
    filled in, with the fan's stray triangles floating over the tyre. Ear clipping is what
    makes a concave silhouette usable, and a wheel arch IS a concave silhouette."""
    idx = list(range(len(poly)))
    out = []
    guard = 0
    while len(idx) > 3 and guard < 4 * len(poly):
        guard += 1
        for k in range(len(idx)):
            a, b, c = idx[k - 1], idx[k], idx[(k + 1) % len(idx)]
            pa, pb, pc = poly[a], poly[b], poly[c]
            cross = ((pb[0] - pa[0]) * (pc[1] - pa[1]) - (pb[1] - pa[1]) * (pc[0] - pa[0]))
            if cross <= 1e-12:                       # reflex or collinear: not an ear
                continue
            bad = False
            for j in idx:            # no REFLEX vertex strictly inside the candidate ear
                if j in (a, b, c):
                    continue
                q0, q1, q2 = poly[idx[idx.index(j) - 1]], poly[j], poly[(idx + idx)[idx.index(j) + 1]]
                if ((q1[0] - q0[0]) * (q2[1] - q0[1]) - (q1[1] - q0[1]) * (q2[0] - q0[0])) > 0:
                    continue                         # convex vertex can't block an ear
                p = poly[j]
                d1 = (pb[0] - pa[0]) * (p[1] - pa[1]) - (pb[1] - pa[1]) * (p[0] - pa[0])
                d2 = (pc[0] - pb[0]) * (p[1] - pb[1]) - (pc[1] - pb[1]) * (p[0] - pb[0])
                d3 = (pa[0] - pc[0]) * (p[1] - pc[1]) - (pa[1] - pc[1]) * (p[0] - pc[0])
                if d1 > 1e-12 and d2 > 1e-12 and d3 > 1e-12:
                    bad = True
                    break
            if not bad:
                out.append([a, b, c])
                idx.pop(k)
                break
        else:
            break                                     # degenerate: stop with what we have
    if len(idx) == 3:
        out.append(list(idx))
    return out


def prism(poly, y0, y1):
    """A 2-D outline in the xz plane, extruded between two y — a vehicle's side
    silhouette, a kerb section, an A-frame's leg.

    Windings are made consistent (and the polygon flipped if it was given clockwise) so the
    solid is manifold, and the two caps are ear-clipped, so the outline may be CONCAVE —
    which is what it takes to cut wheel arches into a body rather than bolt them on."""
    poly = [list(p) for p in poly]
    # The shoelace sum is POSITIVE for a clockwise outline; ear clipping wants CCW, so
    # flip a clockwise one. (This test used to flip the other way. It did not matter while
    # the caps were single n-gons — winding only had to be self-consistent — and it mattered
    # very much the moment they had to be triangulated.)
    area = sum((poly[(i + 1) % len(poly)][0] - poly[i][0]) *
               (poly[(i + 1) % len(poly)][1] + poly[i][1]) for i in range(len(poly)))
    if area > 0:
        poly = poly[::-1]
    n = len(poly)
    verts = [[x, y0, z] for x, z in poly] + [[x, y1, z] for x, z in poly]
    tris = _ear_clip(poly)
    faces = [list(reversed(t)) for t in tris] + [[n + i for i in t] for t in tris]
    faces += [[i, (i + 1) % n, n + (i + 1) % n, n + i] for i in range(n)]
    return MeshProgram().mesh(verts=verts, faces=faces)


def loft(poly, half_width, n_smooth=0, crease_angle=20.0):
    """A silhouette extruded with a VARYING half-width — the prism a vehicle actually needs.

    `half_width(x, z)` returns the body's half-width at each point of the outline, so the
    shell can be narrower at the roof and at the sill and fullest through the middle. That
    single change is most of what separates a car from a slab with wheels: a real body has
    tumblehome, and a constant-width extrusion cannot have any.

    `n_smooth` rounds it with Catmull-Clark, holding edges sharper than `crease_angle`. The
    threshold has to be LOW — 20 degrees, not 50. A vehicle's silhouette is mostly gentle
    turns, and at 50 only the few sharpest corners were held: the sill, the nose and the
    wheel-arch openings all melted into the body and it came out looking poured rather than
    pressed. What should round is the cross-section, not the profile.

    `n_smooth` DEFAULTS OFF, and the vehicles use it that way. The side caps are ear-clipped
    TRIANGLES and Catmull-Clark wants quads: subdividing them spreads the triangulation's
    unevenness into diagonal banding straight across a flat flank, at one level as much as
    at two. The loft's tumblehome plus the tracer's --smooth-angle shading gives the rounded
    READ without inventing geometry that is not there, which is the honest trade until the
    cage is rebuilt as an all-quad tube."""
    poly = [list(q) for q in poly]
    area = sum((poly[(i + 1) % len(poly)][0] - poly[i][0]) *
               (poly[(i + 1) % len(poly)][1] + poly[i][1]) for i in range(len(poly)))
    if area > 0:
        poly = poly[::-1]
    n = len(poly)
    hw = [float(half_width(x, z)) for x, z in poly]
    verts = ([[x, -w, z] for (x, z), w in zip(poly, hw)]
             + [[x, w, z] for (x, z), w in zip(poly, hw)])
    tris = _ear_clip(poly)
    faces = [list(reversed(t)) for t in tris] + [[n + i for i in t] for t in tris]
    faces += [[i, (i + 1) % n, n + (i + 1) % n, n + i] for i in range(n)]
    p = MeshProgram().mesh(verts=verts, faces=faces)
    if n_smooth:
        p.crease({"by": "sharp", "angle": crease_angle}, weight=float(n_smooth))
        p.subdivide(levels=int(n_smooth))
    return p


def _interp(pts, x):
    """Piecewise-linear lookup over an x-monotonic control list."""
    if x <= pts[0][0]:
        return pts[0][1]
    for (x0, z0), (x1, z1) in zip(pts, pts[1:]):
        if x <= x1:
            t = 0.0 if x1 <= x0 else (x - x0) / (x1 - x0)
            return z0 + (z1 - z0) * t
    return pts[-1][1]


def loft_grid(lower, upper, hw, nx=90, nz=11, n_smooth=1, crease_angle=15.0):
    """A body as an ALL-QUAD grid: for each x, the shell spans `lower(x)` to `upper(x)`.

    This is the cage a lofted, subdivided body needs. The obvious construction — extrude a
    silhouette polygon and cap the ends — gives caps that must be TRIANGULATED, and
    Catmull-Clark on long thin triangles spreads their unevenness into diagonal facet bands
    right across a flat flank. Four attempts at fixing that from the outside (two subdivision
    levels, one, none, a wider smooth-shading angle) each softened it and none worked,
    because the fault was the topology.

    Parameterising by x instead removes the problem: a vehicle's silhouette is single-valued
    in x — one bottom edge and one top edge at every station, wheel arches included — so the
    surface is a regular (nx x nz) grid on each flank plus a rim of quads joining them, and
    every face is a quad. Subdivision then does what it is for.

    `nx` and `crease_angle` are set for WHEEL ARCHES, which are the one place a car body
    turns hard. At 44 stations over six metres the arch is sampled every 13 cm and its
    corners fall between samples; at a 26-degree crease the opening then rounds away and the
    body hangs over the tyre like a skirt. 90 stations and 15 degrees hold it."""
    xs = [lower[0][0] + (lower[-1][0] - lower[0][0]) * i / (nx - 1) for i in range(nx)]
    verts, idx = [], {}
    for side in (0, 1):
        for i, x in enumerate(xs):
            zb, zt = _interp(lower, x), _interp(upper, x)
            for j in range(nz):
                z = zb + (zt - zb) * j / (nz - 1)
                w = float(hw(x, z))
                idx[(side, i, j)] = len(verts)
                verts.append([x, (w if side else -w), z])
    faces = []
    for i in range(nx - 1):
        for j in range(nz - 1):
            a0, b0 = idx[(0, i, j)], idx[(0, i + 1, j)]
            c0, d0 = idx[(0, i + 1, j + 1)], idx[(0, i, j + 1)]
            faces.append([a0, b0, c0, d0])
            a1, b1 = idx[(1, i, j)], idx[(1, i + 1, j)]
            c1, d1 = idx[(1, i + 1, j + 1)], idx[(1, i, j + 1)]
            faces.append([d1, c1, b1, a1])
    for i in range(nx - 1):                       # the bottom and top rims
        faces.append([idx[(0, i, 0)], idx[(1, i, 0)], idx[(1, i + 1, 0)], idx[(0, i + 1, 0)]])
        faces.append([idx[(0, i + 1, nz - 1)], idx[(1, i + 1, nz - 1)],
                      idx[(1, i, nz - 1)], idx[(0, i, nz - 1)]])
    for j in range(nz - 1):                       # the nose and tail ends
        faces.append([idx[(0, 0, j + 1)], idx[(1, 0, j + 1)], idx[(1, 0, j)], idx[(0, 0, j)]])
        faces.append([idx[(0, nx - 1, j)], idx[(1, nx - 1, j)],
                      idx[(1, nx - 1, j + 1)], idx[(0, nx - 1, j + 1)]])
    p = MeshProgram().mesh(verts=verts, faces=faces)
    if n_smooth:
        p.crease({"by": "sharp", "angle": crease_angle}, weight=float(n_smooth))
        p.subdivide(levels=int(n_smooth))
    return p


def arch(cx, r, z0, n=9):
    """The wheel-arch notch in a body's side outline: an arc from (cx-r, z0) up over the
    wheel and back down to (cx+r, z0), in the direction a CCW bottom edge runs."""
    out = []
    for k in range(n + 1):
        a = math.pi - math.pi * k / n
        out.append([cx + r * math.cos(a), z0 + r * math.sin(a)])
    return out


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
    # 6.5 cm pitch, not 4.8. At the finer spacing the grating rendered 1.65x the
    # photograph's detail — busier than the real thing, which is its own kind of wrong.
    n = int((w - 0.13) / 0.065)                                       # the grating's bars
    for i in range(n):
        x = -(w - 0.13) / 2 + (i + 0.5) * (w - 0.13) / n
        p.place(box(0.014, d - 0.10, 0.024), at=[x, 0, h + 0.003], material=WET_STEEL)
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
    # EIGHT, not five. Count them in the photograph: the flank of a multi-product dispenser
    # carries one hose per grade per side and they hang over each other. Five reads as a
    # tidy diagram of a pump; eight reads as a pump somebody uses.
    jit = [0.00, 0.13, -0.07, 0.17, 0.04, -0.11, 0.09, 0.20]
    for k in range(8):
        t = k / 7.0
        a = [side * 0.54, -0.19 + 0.048 * k, 1.33]
        b = [side * 0.50, 0.00 + 0.026 * k, 1.14 - 0.028 * k]
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
    """The canopy column — the largest single object in the frame, and the last one that
    was still a painted box.

    Cladding is PANELS, and panels have joints in both directions: a recessed shadow line
    every 1.2 m up and every 0.55 m across, plus a corner reveal down each arris and a
    bolted access hatch at working height. The scorecard put it at 0.20 of the photograph's
    structure while every hand-detailed part in the scene sat near 0.9 — which is the number
    saying "this is a big smooth rectangle", and it was."""
    p = MeshProgram()
    p.place(cbox(w, d, h, 0.022), at=[0, 0, h / 2], material=CLAD)
    for k in range(1, int(h / 1.2) + 1):                               # horizontal joints
        p.place(box(w + 0.008, d + 0.008, 0.016), at=[0, 0, k * 1.2], material=SEAM)
        p.place(box(w + 0.014, d + 0.014, 0.006), at=[0, 0, k * 1.2 + 0.012], material=CLAD)
    for t in (-1, 1):                                                  # vertical joints
        for u in (-1, 0, 1):
            p.place(box(0.012, d + 0.008, h), at=[u * w * 0.30, 0, h / 2], material=SEAM)
            p.place(box(w + 0.008, 0.012, h), at=[0, u * d * 0.30, h / 2], material=SEAM)
        p.place(cbox(0.030, 0.030, h - 0.1, 0.008), at=[t * (w / 2 - 0.004), 0, h / 2],
                material=SEAM)                                          # corner reveals
        p.place(cbox(0.030, 0.030, h - 0.1, 0.008), at=[0, t * (d / 2 - 0.004), h / 2],
                material=SEAM)
    p.place(cbox(0.34, 0.02, 0.44, 0.008), at=[-0.24, -d / 2 - 0.006, 1.05], material=SEAM)
    p.place(cbox(0.30, 0.02, 0.40, 0.006), at=[-0.24, -d / 2 - 0.012, 1.05], material=CLAD)
    for z in (0.88, 1.22):                                             # the hatch's fixings
        for x in (-0.36, -0.12):
            p.place(cyl(0.011, 0.014, 8), at=[x, -d / 2 - 0.016, z], rotate=[90, 0, 0],
                    material=STEEL)
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
    # 2.9 x 1.95, not 2.06 x 1.40: the grating segments out of the photograph at roughly
    # 3.1 x 2.5 m, and an island that small was most of why the whole assembly read as a
    # model of a petrol pump rather than a petrol pump.
    p.place(grate_plinth(2.90, 1.95, 0.15), mark="plinth")
    p.place(clad_column(), at=[0.02, 0.46, Z], mark="column")
    p.place(pump_sign(), at=[0.02, 0.02, Z + 2.34], mark="sign")
    p.place(dispenser(), at=[0, 0.06, Z], mark="dispenser")
    for s in (-1, 1):
        p.place(hose_tangle(s), at=[0, 0.06, Z], mark="hoses")
    p.place(fire_cabinet(), at=[0.02, -0.44, Z], mark="firebox")
    p.place(fire_bucket(), at=[0.52, -0.46, Z], mark="bucket")
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
    """The workshop frontage.

    Was a tiled slab with three bands on it and scored 0.17 of the photograph's structure.
    A real shop front carries the things nobody models and everybody sees: a lintel with a
    shadow gap over the openings, split air-conditioners bracketed above them, downpipes
    running the full height into the kerb, and a fascia the signage hangs off."""
    p = MeshProgram()
    p.place(cbox(length, 0.40, h, 0.03), at=[0, 0, h / 2], material=WALL_TILE)
    p.place(cbox(length + 0.10, 0.52, 0.30, 0.02), at=[0, -0.04, 0.15], material=SEAM)
    p.place(cbox(length + 0.06, 0.54, 0.26, 0.02), at=[0, -0.07, 3.42], material=SEAM)
    p.place(cbox(length + 0.02, 0.46, 0.09, 0.012), at=[0, -0.03, 3.60], material=WALL_TILE)
    p.place(cbox(length + 0.14, 0.56, 0.34, 0.03), at=[0, -0.06, h + 0.10], material=WALL_TILE)
    for dx in (-9.6, -3.4, 3.0, 8.4):                                  # split air-con units
        p.place(cbox(0.86, 0.34, 0.58, 0.014), at=[dx, -0.34, 4.10],
                material=mat((0.58, 0.585, 0.58), 0.0, 0.45))
        p.place(cbox(0.74, 0.03, 0.44, 0.008), at=[dx, -0.52, 4.10],
                material=mat((0.30, 0.305, 0.30), 0.0, 0.55))          # the grille
        p.place(cbox(0.94, 0.30, 0.05, 0.010), at=[dx, -0.32, 3.78], material=SEAM)
    for dx in (-11.2, -5.0, 1.4, 7.2, 11.4):                           # downpipes
        p.place(cyl(0.058, h - 0.1, 12), at=[dx, -0.24, (h - 0.1) / 2],
                material=mat((0.42, 0.425, 0.42), 0.0, 0.5))
        for z in (1.1, 2.6, 4.1):
            p.place(cyl(0.072, 0.05, 12), at=[dx, -0.24, z],
                    material=mat((0.34, 0.345, 0.34), 0.0, 0.5))
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
                at=[x, 0, 0], material=HUMP)
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
    # A drain is a HOLE. The photograph reads 0.025 there — near black, the darkest thing
    # in the frame — and this was rendering 0.32 because a grey trough with steel bars over
    # it is not dark, it is a grey trough with steel bars over it.
    p.place(box(length, w, 0.24), at=[0, 0, -0.10], material=mat((0.008, 0.008, 0.009), 0.0, 0.9))
    for k in range(int(length / 0.075)):
        p.place(box(0.040, w, 0.018), at=[-length / 2 + 0.037 + k * 0.075, 0, 0.026],
                material=mat((0.055, 0.056, 0.058), 1.0, 0.42))
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
def wheel(r=0.36, width=0.24, hub=(0.56, 0.56, 0.57)):
    """A wheel with a shouldered tyre and a dished six-spoke face. The flat disc that used
    to stand in for one is the most obvious tell on a parked vehicle."""
    p = MeshProgram()
    p.place(cyl(r, width, 26), rotate=[90, 0, 0], material=RUBBER)
    for s in (-1, 1):                                                  # tyre shoulders
        p.place(cyl(r - 0.035, 0.03, 26), at=[0, s * width / 2, 0], rotate=[90, 0, 0],
                material=RUBBER)
    p.place(cyl(r * 0.66, width * 0.50, 22), at=[0, -width * 0.32, 0], rotate=[90, 0, 0],
            material=mat(hub, 0.7, 0.32))                              # the rim face
    p.place(cyl(r * 0.26, width * 0.58, 16), at=[0, -width * 0.38, 0], rotate=[90, 0, 0],
            material=CHROME)                                           # the hub cap
    for k in range(6):                                                 # the spoke windows
        a = 2 * math.pi * k / 6 + 0.3
        p.place(cyl(r * 0.125, 0.02, 10),
                at=[r * 0.43 * math.cos(a), -width * 0.44, r * 0.43 * math.sin(a)],
                rotate=[90, 0, 0], material=BLACK)
    return p


def van():
    """The white passenger van across the yard.

    MEASURED, after three rounds of adding faces to a body whose proportions were wrong.
    Unprojecting the sill at its own height gives 7.40 m of van, and solving the roofline
    against the van's own vertical plane gives a roof at 1.89 m — against the 5.81 x 2.28
    it had been built at. Length over height 3.9, not 2.6. A long-wheelbase minibus, which
    is what the four side windows in the photograph were saying all along, and no quantity
    of subdivision was ever going to fix a body that was a metre and a half too short and
    forty centimetres too tall.

    The shell is a `loft_grid` — an all-quad cage carrying real tumblehome, narrower at the
    roof and the sill — and the hardware on it is INFERRED, which at thirty metres is the
    honest call: the photograph cannot show a door handle, and a van without one is a toy."""
    FA, RA, WR = 2.45, -2.05, 0.36
    Y = 0.99
    ROOF = 1.95
    lower = ([[-3.70, 0.62], [-3.64, 0.50], [-3.00, 0.47]]
             + arch(RA, 0.50, 0.47)
             + [[-0.30, 0.47]]
             + arch(FA, 0.50, 0.47)
             + [[3.24, 0.48], [3.46, 0.56], [3.58, 0.78]])
    upper = [[-3.70, 1.78], [-3.54, ROOF], [1.55, ROOF - 0.01], [1.88, ROOF - 0.07],
             [2.50, 1.32], [2.82, 1.20], [3.30, 1.06], [3.58, 0.98]]

    def hw(x, z):
        t = max(0.0, min(1.0, (z - 0.47) / (ROOF - 0.47)))
        w = Y - 0.46 * (t - 0.42) ** 2 / 0.34 - 0.15 * max(0.0, t - 0.72) / 0.28
        return w * (1.0 - 0.36 * max(0.0, (x - 2.60) / 1.00) ** 2)

    p = MeshProgram()
    p.place(loft_grid(lower, upper, hw, n_smooth=2), material=BODY_WH)
    # the rocker in the photograph is nearly black against the white body, not grey
    body = mat((0.105, 0.108, 0.115), 0.0, 0.45)
    trim = mat((0.50, 0.505, 0.51), 0.0, 0.35)

    def flank(x, z, out=0.0):
        return hw(x, z) + out

    GZ, GH = 1.52, 0.42                      # the window band: centre and height
    for s in (-1, 1):
        p.place(cbox(4.70, 0.03, GH, 0.012), at=[-1.10, s * flank(-1.10, GZ, -0.012), GZ],
                material=GLASS)
        p.place(cbox(1.02, 0.03, GH + 0.04, 0.012), at=[1.34, s * flank(1.34, GZ, -0.012), GZ],
                material=GLASS)
        for x in (0.70, -0.35, -1.42, -2.50):                       # B/C/D/E pillars
            p.place(cbox(0.10, 0.05, GH + 0.03, 0.012), at=[x, s * flank(x, GZ, -0.004), GZ],
                    material=BODY_WH)
        p.place(cbox(4.80, 0.02, 0.030, 0.006), at=[-1.05, s * flank(-1.05, GZ + GH / 2, 0.004),
                                                    GZ + GH / 2], material=trim)
        p.place(cbox(4.80, 0.02, 0.030, 0.006), at=[-1.05, s * flank(-1.05, GZ - GH / 2, 0.004),
                                                    GZ - GH / 2], material=trim)
        # the heavy dark rocker: in the photograph it is a third of the body's height
        p.place(cbox(6.60, 0.05, 0.30, 0.012), at=[-0.20, s * flank(-0.20, 0.70, -0.004), 0.70],
                material=body)
        p.place(cbox(5.60, 0.03, 0.020, 0.008), at=[-0.30, s * flank(-0.30, 1.02, 0.006), 1.02],
                material=mat((0.60, 0.605, 0.61), 0.0, 0.30))
        for x in (0.72, -2.52):                                     # door shut lines
            p.place(cbox(0.04, 0.06, 0.70, 0.010), at=[x, s * flank(x, 1.05, 0.002), 1.05],
                    material=trim)
        for x in (0.44, -2.24):                                     # door handles
            p.place(cbox(0.17, 0.05, 0.045, 0.010), at=[x, s * flank(x, 1.20, 0.020), 1.20],
                    material=mat((0.62, 0.625, 0.63), 0.5, 0.30))
        p.place(cyl(0.022, 0.17, 8), at=[2.30, s * flank(2.30, 1.48, 0.07), 1.48],
                rotate=[90, 0, 0], material=BLACK)
        p.place(cbox(0.13, 0.07, 0.20, 0.016), at=[2.30, s * flank(2.30, 1.50, 0.16), 1.50],
                material=BLACK)
        p.place(cbox(0.02, 0.05, 0.16, 0.006), at=[2.26, s * flank(2.30, 1.50, 0.17), 1.50],
                material=GLASS)
        p.place(cbox(0.13, 0.30, 0.10, 0.012), at=[-3.71, s * 0.72, 1.10], material=TAIL_RED)
        p.place(cbox(0.10, 0.13, 0.48, 0.012), at=[-3.71, s * 0.70, 1.36], material=TAIL_RED)
        p.place(cbox(0.12, 0.34, 0.15, 0.014), at=[3.58, s * 0.64, 0.90], material=LAMP)
        p.place(cbox(0.06, 0.15, 0.09, 0.010), at=[3.60, s * 0.78, 0.72],
                material=mat((0.62, 0.42, 0.05), 0.0, 0.25))
    p.place(pane([2.50, 1.34], [1.90, 1.86], 0.02, -0.86, 0.86), material=GLASS)
    p.place(cbox(0.03, 1.40, 0.40, 0.010), at=[-3.705, 0, 1.56], material=GLASS)
    p.place(cbox(0.05, 0.05, 1.20, 0.012), at=[-3.70, 0, 1.30], material=trim)
    p.place(cbox(0.16, 1.84, 0.22, 0.02), at=[-3.73, 0, 0.66], material=BODY_WH)
    p.place(cbox(0.10, 1.28, 0.10, 0.014), at=[-3.77, 0, 0.52], material=body)
    p.place(cbox(0.14, 1.28, 0.28, 0.02), at=[3.62, 0, 0.62],
            material=mat((0.26, 0.27, 0.28), 0.0, 0.40))
    for k in range(4):
        p.place(cbox(0.04, 1.00, 0.026, 0.006), at=[3.60, 0, 1.02 + k * 0.042],
                material=mat((0.08, 0.085, 0.09), 0.0, 0.35))
    p.place(cbox(0.03, 0.44, 0.15, 0.006), at=[-3.84, 0.26, 0.86], material=PLATE_F)
    p.place(cyl(0.030, 0.16, 10), at=[-3.10, -Y + 0.12, 0.28], rotate=[0, 90, 0],
            material=mat((0.30, 0.30, 0.31), 0.6, 0.5))
    for x in (FA, RA):
        for s in (-1, 1):
            p.place(wheel(WR, 0.26), at=[x, s * flank(x, WR + 0.30, -0.14), WR])
    return p


def suv():
    """The crossover at the frame's left edge, lofted and smoothed the same way.

    The camera sees its BACK, so that is where the detail went: wrap-around tail lights
    climbing off the tailgate onto the quarter panel, a tailgate window under a small roof
    spoiler, roof rails, a wiper, and the blue plate low in the middle."""
    FA, RA, WR = 1.40, -1.42, 0.34
    Y = 0.93
    lower = ([[-2.26, 0.56], [-2.20, 0.40], [-1.98, 0.36]]
             + arch(RA, 0.46, 0.36)
             + [[-0.20, 0.36]]
             + arch(FA, 0.46, 0.36)
             # The nose keeps a little HEIGHT: where the lower and upper profiles meet
             # exactly, the end rim collapses to zero-area quads and the kernel rejects
             # the mesh, correctly.
             + [[2.08, 0.38], [2.26, 0.50], [2.32, 0.62]])
    upper = [[-2.26, 0.92], [-2.02, 1.30], [-1.70, 1.64], [-0.90, 1.72], [0.30, 1.64],
             [1.10, 1.26], [1.52, 1.14], [2.18, 1.02], [2.32, 0.86]]

    def hw(x, z):
        t = max(0.0, min(1.0, (z - 0.36) / 1.36))
        w = Y - 0.40 * (t - 0.38) ** 2 / 0.30 - 0.19 * max(0.0, t - 0.66) / 0.34
        return w * (1.0 - 0.30 * max(0.0, (x - 1.5) / 0.9) ** 2
                    - 0.22 * max(0.0, (-1.4 - x) / 0.9) ** 2)
    p = MeshProgram()
    p.place(loft_grid(lower, upper, hw, n_smooth=2), material=BODY_WH)
    trim = mat((0.45, 0.455, 0.46), 0.0, 0.35)

    def flank(x, z, out=0.0):
        return hw(x, z) + out
    for s in (-1, 1):
        for x, w in ((0.34, 0.82), (-0.60, 0.80), (-1.44, 0.44)):
            p.place(cbox(w, 0.03, 0.34, 0.010), at=[x, s * flank(x, 1.46, -0.010), 1.46],
                    material=GLASS)
        p.place(cbox(0.07, 0.05, 0.30, 0.010), at=[-0.14, s * flank(-0.14, 1.46, -0.002), 1.46],
                material=BODY_WH)
        p.place(cbox(1.34, 0.07, 0.035, 0.010), at=[-0.55, s * flank(-0.55, 1.70, -0.10), 1.715],
                material=mat((0.38, 0.385, 0.39), 0.7, 0.30))
        p.place(cbox(0.05, 0.06, 0.50, 0.010), at=[-0.14, s * flank(-0.14, 1.02, 0.004), 1.02],
                material=trim)
        for x in (0.10, -0.84):
            p.place(cbox(0.16, 0.05, 0.045, 0.008), at=[x, s * flank(x, 1.16, 0.018), 1.16],
                    material=mat((0.62, 0.625, 0.63), 0.5, 0.30))
        p.place(cyl(0.020, 0.13, 8), at=[0.98, s * flank(0.98, 1.32, 0.06), 1.32], rotate=[90, 0, 0],
                material=mat((0.16, 0.165, 0.17), 0.0, 0.35))
        p.place(cbox(0.19, 0.08, 0.12, 0.014), at=[0.98, s * flank(0.98, 1.33, 0.13), 1.33],
                material=mat((0.16, 0.165, 0.17), 0.0, 0.35))
        p.place(cbox(0.06, 0.30, 0.32, 0.010), at=[-2.24, s * 0.58, 1.04], material=TAIL_RED)
        p.place(cbox(0.30, 0.05, 0.24, 0.010), at=[-2.03, s * flank(-2.03, 1.08, -0.010), 1.08],
                material=TAIL_RED)
        p.place(cbox(0.10, 0.30, 0.12, 0.012), at=[2.28, s * 0.62, 0.92], material=LAMP)
    p.place(pane([1.10, 1.28], [0.34, 1.62], 0.02, -0.82, 0.82), material=GLASS)
    p.place(pane([-1.72, 1.62], [-2.04, 1.30], 0.02, -0.78, 0.78), material=GLASS)
    p.place(cbox(0.24, 1.54, 0.09, 0.016), at=[-1.76, 0, 1.74], material=BODY_WH)
    p.place(cyl(0.012, 0.44, 8), at=[-1.92, 0.10, 1.42], rotate=[0, 62, 0], material=BLACK)
    p.place(cbox(0.16, 1.74, 0.30, 0.02), at=[-2.28, 0, 0.62],
            material=mat((0.30, 0.305, 0.31), 0.0, 0.42))
    p.place(cbox(0.03, 0.44, 0.15, 0.006), at=[-2.36, -0.06, 0.86], material=PLATE_F)
    p.place(cbox(0.16, 1.66, 0.24, 0.02), at=[2.30, 0, 0.58],
            material=mat((0.30, 0.305, 0.31), 0.0, 0.42))
    for x in (FA, RA):
        for s in (-1, 1):
            p.place(wheel(WR, 0.24), at=[x, s * flank(x, WR + 0.28, -0.13), WR])
            p.place(tube_along(arc([x, s * flank(x, WR + 0.32, -0.02), 0.36], 0.46,
                                   8, 172, (1, 0, 0), (0, 0, 1), 10), 0.028, 7),
                    material=mat((0.18, 0.185, 0.19), 0.0, 0.55))
    return p


# --------------------------------------------------------------------------- #
# yard clutter — the things a working forecourt accumulates
# --------------------------------------------------------------------------- #
def carton(w=0.52, d=0.40, h=0.36):
    """A cardboard box, taped. Nobody models these and every yard is full of them."""
    p = MeshProgram()
    p.place(cbox(w, d, h, 0.012), at=[0, 0, h / 2], material=mat((0.34, 0.25, 0.15), 0.0, 0.72))
    p.place(cbox(0.06, d + 0.004, 0.004, 0.001), at=[0, 0, h + 0.001],
            material=mat((0.55, 0.52, 0.46), 0.0, 0.55))          # the tape down the middle
    p.place(cbox(w + 0.004, 0.05, 0.004, 0.001), at=[0, 0, h * 0.55],
            material=mat((0.28, 0.20, 0.12), 0.0, 0.72))          # the flap seam
    return p


def tyre_stack(n=3, r=0.33, t=0.20):
    p = MeshProgram()
    for k in range(n):
        p.place(MeshProgram().torus(major_segments=20, minor_segments=8,
                                    major_radius=r * 0.72, minor_radius=t * 0.48),
                at=[0, 0, t * (k + 0.5)], material=RUBBER)
    return p


def scooter(length=1.72):
    """A delivery scooter: two wheels, a deck, a column and a seat. Small, and the yard
    reads as abandoned without one."""
    p = MeshProgram()
    body = mat((0.10, 0.11, 0.13), 0.0, 0.40)
    for x, r in ((length * 0.42, 0.21), (-length * 0.40, 0.21)):
        p.place(cyl(r, 0.09, 20), at=[x, 0, r], rotate=[90, 0, 0], material=RUBBER)
        p.place(cyl(r * 0.42, 0.10, 14), at=[x, 0, r], rotate=[90, 0, 0], material=CHROME)
    p.place(cbox(length * 0.62, 0.30, 0.16, 0.02), at=[0, 0, 0.34], material=body)
    p.place(cbox(0.42, 0.26, 0.14, 0.02), at=[-length * 0.20, 0, 0.56], material=body)  # seat
    p.place(cbox(0.30, 0.24, 0.34, 0.02), at=[length * 0.34, 0, 0.52], rotate=[0, -14, 0],
            material=body)                                        # the front apron
    p.place(cyl(0.030, 0.52, 10), at=[length * 0.36, 0, 0.78], rotate=[0, -14, 0],
            material=CHROME)                                      # the steering column
    p.place(cyl(0.022, 0.56, 10), at=[length * 0.30, 0, 1.02], rotate=[0, 90, 0],
            material=BLACK)                                       # handlebars
    p.place(cbox(0.34, 0.36, 0.28, 0.02), at=[-length * 0.46, 0, 0.66],
            material=mat((0.42, 0.10, 0.08), 0.0, 0.45))          # the delivery box
    return p


def tape_run(points, r=0.018, sag=0.09):
    """Red-and-white hazard tape strung between posts.

    Banded, like the rail, because the bands have to BE geometry before a colour can land
    on only some of them — and SAGGING, because tape strung between two posts is the one
    thing in a yard that cannot be straight. A dead-level run reads as a laser."""
    out = []
    for a0, b0 in zip(points, points[1:]):
        out.extend(drape(a0, b0, sag=sag, n=6)[:-1])
    out.append(list(points[-1]))
    return banded_tube(out, r, [mat((0.52, 0.06, 0.05), 0.0, 0.55), WHITE], band=0.22,
                       sides=6)
