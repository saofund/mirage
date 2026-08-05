"""The fuel-filler pocket, modelled one part at a time.

Conventions, so the assembly can place these without guessing:

* **metres**, and a part's origin is the centre of the feature it is named for.
* the **cap's own frame**: its sealing face is the z = 0 plane, its outward normal is
  **+z**, and its grip rib runs along **+x**. That plane and that normal are the label —
  everything this case generates is ground truth *about* them — so no part is allowed to
  quietly redefine them.
* the **pocket frame**: the body panel is the z = 0 plane with the car's outside at +z, and
  the cap sits at z = -(recess depth). A cap is therefore *never* flush with the panel;
  every real one is at the bottom of a hole, and half of what a depth sensor sees at this
  distance is the hole rather than the cap.

The shapes come from measurement, not from taste. `fit.py` reads 400 real frames and
returns a canonical height field of the cap face; the defaults below are that field's
dimensions, and `examples/cases/fuelcap/sheet.py --compare` puts the two side by side.
"""
from __future__ import annotations

import math

import numpy as np

from mirage.meshlang import MeshProgram

from .materials import (
    CAP_ALU, CAP_ALU_D, CAP_BLACK, CAP_CHROME, CATCH_STEEL, GRIME, GROMMET, NECK_STEEL,
    SCREW_STEEL, SEAL_RED, SEAL_RUBBER, TETHER, WELL_METAL, WELL_PLASTIC, mat,
)

TAU = 2.0 * math.pi


# --------------------------------------------------------------------------- #
# primitives
# --------------------------------------------------------------------------- #
def lathe(section, steps=48, mark=None):
    """A solid of revolution about z from a (radius, z) SECTION.

    The section is an open polyline that must start and end **on the axis** (radius 0):
    spun through 360 degrees that closes into a watertight solid, which is what every
    round part in this pocket is. Passing a section that misses the axis gives a tube with
    two open rims — silently, and the mesh only fails much later, so it is checked here."""
    if abs(section[0][0]) > 1e-9 or abs(section[-1][0]) > 1e-9:
        raise ValueError("lathe section must begin and end on the axis (radius 0)")
    return MeshProgram().profile([(r, z) for r, z in section], plane="xz",
                                 closed=False).spin(axis="z", steps=steps, angle=360.0, mark=mark)


def _ear_clip(poly):
    """Triangulate a simple CCW polygon. The tracer fans an n-gon from its first vertex,
    which lays triangles across the notches of anything concave — and the grip rib's plan,
    once it grows a slot down the middle, is concave."""
    idx = list(range(len(poly)))
    out, guard = [], 0
    while len(idx) > 3 and guard < 6 * len(poly):
        guard += 1
        for k in range(len(idx)):
            a, b, c = idx[k - 1], idx[k], idx[(k + 1) % len(idx)]
            pa, pb, pc = poly[a], poly[b], poly[c]
            if ((pb[0] - pa[0]) * (pc[1] - pa[1]) - (pb[1] - pa[1]) * (pc[0] - pa[0])) <= 1e-14:
                continue
            bad = False
            for j in idx:
                if j in (a, b, c):
                    continue
                p = poly[j]
                d1 = (pb[0] - pa[0]) * (p[1] - pa[1]) - (pb[1] - pa[1]) * (p[0] - pa[0])
                d2 = (pc[0] - pb[0]) * (p[1] - pb[1]) - (pc[1] - pb[1]) * (p[0] - pb[0])
                d3 = (pa[0] - pc[0]) * (p[1] - pc[1]) - (pa[1] - pc[1]) * (p[0] - pc[0])
                if d1 > 1e-14 and d2 > 1e-14 and d3 > 1e-14:
                    bad = True
                    break
            if not bad:
                out.append((a, b, c))
                idx.pop(k)
                break
        else:
            break
    if len(idx) == 3:
        out.append(tuple(idx))
    return out


def prism(poly, z0, z1, mark=None):
    """A closed prism between two z planes from a CCW plan polygon."""
    n = len(poly)
    verts = [(x, y, z0) for x, y in poly] + [(x, y, z1) for x, y in poly]
    tris = _ear_clip(poly)
    faces = [list(reversed(t)) for t in tris] + [[n + i for i in t] for t in tris]
    faces += [[i, (i + 1) % n, n + (i + 1) % n, n + i] for i in range(n)]
    return MeshProgram().mesh(verts=verts, faces=faces, mark=mark)


def frustum(poly, z0, z1, shrink, mark=None):
    """A prism whose top plan is `shrink` times the bottom — the rib's sloped flanks.

    A rib with vertical sides is a step, and a step is what a naive model of this part
    always is. Every real grip is drafted so it can leave the mould, and that draft is
    what the highlight runs along."""
    n = len(poly)
    cx = sum(p[0] for p in poly) / n
    cy = sum(p[1] for p in poly) / n
    top = [(cx + (x - cx) * shrink, cy + (y - cy) * shrink) for x, y in poly]
    verts = [(x, y, z0) for x, y in poly] + [(x, y, z1) for x, y in top]
    tris = _ear_clip(poly)
    faces = [list(reversed(t)) for t in tris] + [[n + i for i in t] for t in tris]
    faces += [[i, (i + 1) % n, n + (i + 1) % n, n + i] for i in range(n)]
    return MeshProgram().mesh(verts=verts, faces=faces, mark=mark)


def resample_by_angle(poly, n):
    """Re-sample a closed plan outline at `n` EVENLY SPACED ANGLES about its centroid.

    Needed whenever two outlines have to be joined ring-to-ring. A stadium is parameterised
    by arc length, not by angle, so pairing its points with a circle's by index — or by
    rounding an angle to an index — pairs the wrong ones and the quads between them
    self-intersect. On a cap that showed up as a grip well whose mouth was sealed over by
    its own surrounding face."""
    cx = sum(x for x, _ in poly) / len(poly)
    cy = sum(y for _, y in poly) / len(poly)
    # A real ray-segment intersection. Interpolating a (angle, radius) table looks simpler
    # and is wrong on any outline with a STRAIGHT edge: a stadium's flat side runs from
    # 45 to 135 degrees at a constant y, so interpolating its endpoint radii gives 21 mm
    # at 90 degrees where the true answer is 15. Every grip slot in this kit has flat
    # sides.
    #
    #   t*d = A + u*(B-A)  ->  t = cross(A, B-A) / cross(d, B-A),  u = cross(A, d) / same
    cr = lambda ax, ay, bx, by: ax * by - ay * bx
    m = len(poly)
    out = []
    for k in range(n):
        a = TAU * k / n
        dx, dy = math.cos(a), math.sin(a)
        best = None
        for i in range(m):
            ax, ay = poly[i][0] - cx, poly[i][1] - cy
            bx, by = poly[(i + 1) % m][0] - cx, poly[(i + 1) % m][1] - cy
            ex, ey = bx - ax, by - ay
            den = cr(dx, dy, ex, ey)
            if abs(den) < 1e-15:
                continue
            t = cr(ax, ay, ex, ey) / den
            u = cr(ax, ay, dx, dy) / den
            if t > 1e-12 and -1e-9 <= u <= 1 + 1e-9 and (best is None or t < best):
                best = t
        if best is None:
            # No hit: float noise on a ray that grazes a vertex, or a centroid that is not
            # strictly inside a very flat outline. Falling back to zero puts the sample ON
            # the centroid, which collapses a quad into a degenerate face and kills the
            # whole build — so fall back to the nearest vertex's radius instead, which is
            # wrong by a fraction of a segment and always valid.
            best = min(math.hypot(px - cx, py - cy) for px, py in poly)
        out.append((cx + best * dx, cy + best * dy))
    return out


def flute_plan(lobes, depth, steps, phase=0.0):
    """A `spin` plan for the COARSE VERTICAL FLUTES around a cap's skirt.

    Not the same shape as `lobe_plan` and not normalised the same way, and both differences
    matter. A fuel cap's grip is a ring of broad flats separated by narrow grooves — you can
    count ten to fourteen of them on the reference photographs — so the wave is squared off
    with a tanh rather than left as a cosine, which would give a scalloped edge that reads as
    a gear. And the lugs are the cap's actual outside diameter, so the plan PEAKS at 1
    instead of averaging 1: normalising it, as `lobe_plan` does, would quietly grow the cap
    by half the flute depth every time the flute count changed."""
    ks = []
    for j in range(steps):
        s = math.cos(lobes * (TAU * j / steps) + phase)
        ks.append(1.0 - depth * (0.5 - 0.5 * math.tanh(2.6 * s)))
    return ks


def lobe_plan(lobes, depth, steps, phase=0.0):
    """A `spin` plan with `lobes` rounded bumps round it — the petal-edged cap.

    A handful of the reference cars have a cap whose rim is not a circle but a ring of
    shallow lobes you grip with your fingertips. That is a plan, not a section, so it is
    exactly what the generalised lathe is for: same profile, different plan, one line.
    Normalised so adding lobes does not also change the cap's diameter."""
    ks = [1.0 + depth * math.cos(lobes * (TAU * j / steps) + phase) for j in range(steps)]
    m = sum(ks) / len(ks)
    return [k / m for k in ks]


def handle(length, w_end, w_mid, height, dish=0.34, sag=0.10, ends_down=0.006,
           base=0.008, stations=30, mark="cap_rib"):
    """The moulded grip across a fuel cap: a WAISTED bar with a TROUGH along its top.

    This is the part the old kit got most wrong, and it got it wrong by being a primitive it
    already had rather than the shape in the photograph. A `frustum` of a stadium is a
    straight-sided block with a flat top; every cap in the reference set has a bar that is
    wide at both ends, pinched in the middle, and hollowed along the top so a thumb sits in
    it. Measured off the head-on frames: ends 0.38 of the cap diameter, middle 0.27 — a waist
    ratio of about 0.71 — with the top some 0.085 of the diameter proud of the face.

    Three things vary along the run and each is a different mechanism:

    * the WAIST is `scale`'s x — the section gets narrower and comes back;
    * the TROUGH along the length is `scale`'s y, which lowers the whole section (its base is
      buried, so only the top moves visibly);
    * the TROUGH across the width is in the profile itself, a concave top between two
      shoulders.

    The ends are not capped and do not need to be: the path DIPS below the cap's face at both
    ends, so the two open rings finish inside solid material. That also produces the rounded
    end the photographs show, for free and for the right reason — what you see is the curve
    where the bar's flank crosses the face, not an end cap somebody drew.
    """
    hw = w_end / 2.0
    prof = [
        (-hw, -base), (hw, -base),                 # buried base, well below the face
        (hw, height * 0.34), (hw * 0.90, height * 0.86), (hw * 0.66, height),
        (hw * 0.30, height * (1.0 - dish * 0.55)),
        (0.0, height * (1.0 - dish)),              # the trough, across the bar
        (-hw * 0.30, height * (1.0 - dish * 0.55)),
        (-hw * 0.66, height), (-hw * 0.90, height * 0.86), (-hw, height * 0.34),
    ]
    waist = max(0.0, 1.0 - w_mid / max(w_end, 1e-6))
    path, scale = [], []
    for j in range(stations):
        u = j / (stations - 1.0)
        s = math.sin(math.pi * u)
        # ends below the face, middle at it: `ends_down` is what buries the open rings
        path.append((length * (u - 0.5), 0.0, -ends_down * (1.0 - s ** 0.55)))
        scale.append((1.0 - waist * s ** 1.2, 1.0 - sag * s))
    return (MeshProgram().profile(prof, plane="xy", closed=True)
            .sweep(path, scale=scale, mark=mark))


def slotted_face(r, slot, z_top, z_floor, draft=0.8, ring=44, mark="cap_body"):
    """A disc with a WELL sunk into it — an annular top, drafted walls, and a floor.

    `place` composes by UNION, so a well cannot be made by placing a block where the
    material should be missing: that just adds a block, which is why the first recessed
    grip rendered as a raised pad. Without reaching for a boolean (slow, and it has to
    succeed on every frame of a dataset) the answer is to build the face WITH the hole in
    it: a ring of quads from the outer circle in to the slot outline, the slot outline
    swept down to the floor, and the floor capped."""
    slot = resample_by_angle(slot, ring)         # same count, same angles as the circle
    n = ring
    verts, faces = [], []
    for i in range(ring):                                   # 0: outer circle, at z_top
        a = TAU * i / ring
        verts.append((r * math.cos(a), r * math.sin(a), z_top))
    for x, y in slot:                                       # ring: slot mouth, at z_top
        verts.append((x, y, z_top))
    cx = sum(x for x, _ in slot) / n
    cy = sum(y for _, y in slot) / n
    for x, y in slot:                                       # ring+n: slot floor, drafted in
        verts.append((cx + (x - cx) * draft, cy + (y - cy) * draft, z_floor))
    verts.append((cx, cy, z_floor))
    mid = len(verts) - 1
    for i in range(ring):                                   # the annular top
        j = (i + 1) % ring
        faces.append([i, j, ring + j, ring + i])
    for i in range(n):
        # The slot WALL, wound so its normal faces INTO the well. A well is seen from
        # inside it, so a wall wound the other way presents its back face to the only
        # camera that will ever look at it — which shades black and reads as no well at
        # all, on top of a cap that then looks like a flat disc with a pad on it.
        j = (i + 1) % n
        faces.append([ring + i, ring + j, ring + n + j, ring + n + i])
    for i in range(n):                                      # the floor
        j = (i + 1) % n
        faces.append([ring + n + i, ring + n + j, mid])
    return MeshProgram().mesh(verts=verts, faces=faces, mark=mark)


def stadium(length, width, arc=10):
    """A rounded-rectangle plan (CCW): the grip rib seen from above.

    `length` is clamped above `width`, because a stadium shorter than it is wide is not a
    shape — its two end arcs land on top of each other and every point is duplicated,
    which the kernel correctly rejects as a degenerate face. Callers derive both from
    ratios of the cap, so the combination does arise; better to round it to the circle it
    is trying to be than to fail a whole frame."""
    length = max(length, width * 1.02)
    a, r = (length - width) / 2.0, width / 2.0
    pts = []
    for i in range(arc + 1):
        t = -math.pi / 2 + math.pi * i / arc
        pts.append((a + r * math.cos(t), r * math.sin(t)))
    for i in range(arc + 1):
        t = math.pi / 2 + math.pi * i / arc
        pts.append((-a + r * math.cos(t), r * math.sin(t)))
    return pts


# --------------------------------------------------------------------------- #
# the cap
# --------------------------------------------------------------------------- #
def cap(d=0.078, flange=0.013, rib_len=None, rib_w=None, rib_h=None, rib_draft=0.66,
        rib_slot=0.0, dome=0.0008, chamfer=0.0025, flutes=12, flute_depth=0.030,
        skirt=0.020, neck_d=0.048, bevel=0.055, spin=0.0, printing=True, grip="rib",
        lobes=0, lobe_depth=0.06, waist=0.71, material=None, rib_material=None, steps=64):
    """The inner fuel cap: a fluted cylinder with a waisted handle moulded across its face.

    Rebuilt against the photographs, which say something different from the clouds. A cloud
    binned by radius sees this part as a disc with a bump; ninety head-on colour frames see
    a short **cylinder** — 10 to 14 mm of visible side wall, with ten to fourteen broad
    vertical FLUTES round it that are what you actually grip — carrying a top face that is
    inset behind a narrow bevel, with a **waisted handle** across it and two arcs of faint
    printing either side. The old model had none of those four things: it was a flat disc
    with a straight-sided pad in the middle, and at the magnification a reference photograph
    is at, that is not the same object.

    Proportions, measured off the head-on frames as fractions of the diameter:

        visible skirt      0.16          handle length     0.98 (ends buried, 0.86 visible)
        rim bevel          0.055         handle end width  0.38
        flutes             10-14         handle waist      0.27  (ratio 0.71)
        printing height    0.05          handle height     0.085

    `rib_len` / `rib_w` / `rib_h` default to those fractions and are kept as overrides so a
    caller that measured its own cap can say so. `flutes=0` gives the plain cylinder a
    minority of cars have.

    The flutes are a `spin` PLAN, not a ring of placed blocks. That is not just tidier: the
    blocks version stood a millimetre proud of a 39 mm radius and read as a cog, because a
    tooth added on top of a cylinder is a different shape from a groove cut into one. A plan
    faded in from the bevel radius grooves the wall and leaves the face alone, which is the
    real moulding.

    `spin` — where the screw thread happened to stop, in degrees about the cap's OWN axis.
    It belongs here and not in the caller's `place` rotation, and that is not a stylistic
    preference. `place` composes its angles as Rz @ Ry @ Rx, so a z-rotation passed to it
    is applied LAST, outside the tilt — which swings the cap's axis around the panel normal
    instead of turning the cap about itself. The pose label and the geometry then disagree
    by up to twice the tilt angle, and every frame in the set is quietly mislabelled. That
    is exactly what this case shipped with until the label was checked against a plane fit
    of the very points it describes: 8 degrees of median error, invisible in every render."""
    material = material or CAP_BLACK
    rib_material = rib_material or material
    r = d / 2.0
    rib_len = d * 0.98 if rib_len is None else rib_len
    rib_w = d * 0.38 if rib_w is None else rib_w
    rib_h = d * 0.085 if rib_h is None else rib_h
    if printing:
        # The warning text round the annulus. Four 460 px reference photographs all have
        # it and none of the synthetic caps did — it is the most recognisable single
        # feature of this part in a colour image, and it is free: the tracer pins the
        # artwork to a rectangle on the cap's own +z face.
        from mirage.decals import ensure_decals
        from .materials import with_decal
        art = ensure_decals(["fuelcap_face"])["fuelcap_face"]
        material = with_decal(material, art, d * 1.02, d * 1.02, dome + 1e-4)
    rn = min(neck_d / 2.0, r - 0.004)
    c = min(chamfer, flange * 0.45, r * 0.08)
    rf = r * (1.0 - bevel)                 # where the flat top face stops and the bevel starts

    # Section, axis outward. z = 0 is the sealing face — the plane this whole case labels —
    # and it is the flat top the printing sits on, so the bevel and the whole skirt hang
    # BELOW it. `flange` is the visible height of that skirt.
    section = [
        (0.0, dome),                       # a barely-domed centre; real caps are not flat
        (rf * 0.55, dome * 0.80),
        (rf, 0.0),                         # the flat face, out to the bevel
        (r, -r * bevel * 0.55),            # the narrow rim bevel
        (r, -flange + c),                  # the fluted outer wall
        (r - c, -flange),                  # underside chamfer
        (rn, -flange),
        (rn, -flange - skirt),             # the skirt that goes down the filler neck
        (rn * 0.72, -flange - skirt),
        (0.0, -flange - skirt),
    ]
    # Two families of plan, and they are not interchangeable. Lobes reshape the WHOLE rim
    # (a petal-edged cap, a minority fitment); flutes groove only the wall, so they are
    # faded in from the bevel radius and leave the printed face perfectly round.
    if lobes:
        plan, plan_from = lobe_plan(lobes, lobe_depth, steps, math.radians(spin)), r * 0.72
    elif flutes:
        plan, plan_from = flute_plan(flutes, flute_depth, steps, math.radians(spin)), rf
    else:
        plan, plan_from = None, 0.0
    p = (MeshProgram().profile([(rr, zz) for rr, zz in section], plane="xz", closed=False)
         .spin(axis="z", steps=steps, plan=plan, plan_from=plan_from, mark="cap_body")
         .material({"by": "tag", "name": "cap_body"}, **material))

    # The grip, turned about the cap's own axis by `spin`. TWO families, both common:
    #
    #   grip="rib"   the waisted handle across the face — three quarters of the sample
    #   grip="slot"  a rectangular WELL sunk into the face, with walls and a small pad in
    #                its floor — about a quarter of the reference cars, and the shape is
    #                not a handle at all. Building it as one and hoping was the same mistake
    #                as everywhere else on this part: it is a different surface, not a
    #                parameter of the one I already had.
    sa, ca = math.sin(math.radians(spin)), math.cos(math.radians(spin))
    turn = lambda pl: [(x * ca - y * sa, x * sa + y * ca) for x, y in pl]

    if grip == "slot":
        # The land has to be TALL and the well can only be as deep as the land, because
        # what is underneath is the cap body's own solid top face. Sinking the floor past
        # it just buries the well: the body's face is what the camera then sees, which is
        # why this rendered as a flat disc with a pad on it through three attempts.
        land = max(rib_h * 0.62, 0.008)
        depth = land * 0.90
        p = p.place(obj=slotted_face(rf, turn(stadium(rib_len * 0.86, rib_w * 1.05, arc=22)),
                                     dome + land, dome + land - depth,
                                     draft=max(rib_draft, 0.74), mark="cap_slot"),
                    at=(0.0, 0.0, 0.0), material=material)
        # the pad in the floor of the well — every photographed slot has one
        p = p.place(obj=frustum(turn(stadium(rib_len * 0.40, rib_w * 0.42)),
                                dome + land - depth, dome + land - depth * 0.45, 0.86,
                                mark="cap_rib"),
                    at=(0.0, 0.0, 0.0), material=rib_material)
        return p

    p = p.place(obj=handle(rib_len, rib_w, rib_w * waist, rib_h,
                           dish=0.34, sag=0.10, ends_down=rib_h * 0.85,
                           base=rib_h * 1.1, mark="cap_rib"),
                at=(0.0, 0.0, dome), rotate=(0.0, 0.0, spin), material=rib_material)
    if rib_slot > 0:
        # The groove down the spine of a raised bar — the double-decker variant. Sunk into
        # the handle's top, narrower than the waist so it survives the pinch.
        sl = rib_len * 0.62
        sw = rib_w * waist * 0.34
        d2 = min(rib_slot, rib_h * 0.40)
        p = p.place(obj=frustum(turn(stadium(sl, sw)), dome + rib_h + 1e-4,
                                dome + rib_h - d2, 0.88, mark="cap_slot"),
                    at=(0.0, 0.0, 0.0),
                    material=mat([c * 0.68 for c in material["color"]],
                                 material["metallic"], min(0.95, material["roughness"] * 1.15)))
    return p


def seal(d=0.070, thick=0.0035, material=None):
    """The gasket ring under the flange — the orange-red band visible on a removed cap."""
    r = d / 2.0
    return lathe([(0.0, 0.0), (r, 0.0), (r, -thick), (r - 0.004, -thick), (0.0, -thick)],
                 steps=40, mark="seal").material({"by": "tag", "name": "seal"},
                                                 **(material or SEAL_RED))


# --------------------------------------------------------------------------- #
# the pocket
# --------------------------------------------------------------------------- #
def superellipse(rx, ry, n=4.0, seg=48):
    """A rounded-rectangle outline — the aperture shape most of these pockets actually have.

    `n` = 2 is an ellipse and `n` -> infinity is a rectangle; real filler apertures sit
    around 3-5. Modelling them as circles was wrong in a way that only the reference photos
    showed: over half the by-car sample has a squarish opening with a corner radius, and a
    perfect circle is the one shape none of them is."""
    pts = []
    for i in range(seg):
        t = TAU * i / seg
        c, s = math.cos(t), math.sin(t)
        k = (abs(c) ** n + abs(s) ** n) ** (-1.0 / n)
        pts.append((rx * c * k, ry * s * k))
    return pts


def screw(r=0.0035, head_h=0.0016, slot=True, material=None):
    """A pan-head screw. Two or three of them go through the liner on most of the
    reference cars, and at 7 mm across they are a fifth the size of the cap — small, but
    they are metal in a matt black hole, so they are among the brightest things in it."""
    p = lathe([(0.0, head_h), (r * 0.72, head_h), (r, head_h * 0.35), (r, 0.0),
               (r * 0.45, -0.002), (0.0, -0.002)], steps=16, mark="well")
    p = p.material({"by": "tag", "name": "well"}, **(material or SCREW_STEEL))
    if slot:
        p = p.place(obj=prism([(-r * 0.78, -r * 0.13), (r * 0.78, -r * 0.13),
                               (r * 0.78, r * 0.13), (-r * 0.78, r * 0.13)],
                              head_h * 0.45, head_h + 1e-4, mark="well"),
                    at=(0.0, 0.0, 0.0), material=mat([c * 0.35 for c in
                                                      (material or SCREW_STEEL)["color"]],
                                                     0.7, 0.5))
    return p


def catch(w=0.016, h=0.026, t=0.005, material=None):
    """The sprung steel catch / striker the fuel door latches onto — the bright L-shaped
    bracket standing off the pocket wall in several of the reference frames."""
    base = prism([(-w / 2, -t / 2), (w / 2, -t / 2), (w / 2, t / 2), (-w / 2, t / 2)],
                 0.0, h, mark="well")
    arm = prism([(-w / 2, -t / 2), (w / 2, -t / 2), (w / 2, t / 2), (-w / 2, t / 2)],
                0.0, t * 1.6, mark="well")
    p = base.place(obj=arm, at=(0.0, h * 0.30, h - t * 0.8), rotate=(90.0, 0.0, 0.0))
    return p.material({"by": "all"}, **(material or CATCH_STEEL))


def grommet(r=0.006, h=0.004, material=None):
    """A rubber grommet — the soft black plug where a drain tube or a cable leaves the
    pocket. Matt where everything around it is at least slightly glossy."""
    return lathe([(0.0, h), (r * 0.55, h), (r, h * 0.45), (r, 0.0), (0.0, 0.0)],
                 steps=18, mark="well").material({"by": "tag", "name": "well"},
                                                 **(material or GROMMET))


def well_details(rim_r, floor_d, depth, rng=None, ribs=4, drain=True, screws=2,
                 catches=1, grommets=1, seed=0, material=None, keep_out=0.0):
    """The furniture down the recess: stiffening ribs, a step, a drain notch.

    None of this is decoration. `fit.complexity` measures the ratio of plane residual at
    24 mm to residual at 6 mm — how much structure a surface has at the scale between
    those — and the real pockets read 2.13 against a smooth cone's 1.2. That gap is these
    parts: a real recess is a moulding with draft ribs up its wall, a step where the liner
    meets the neck, and a drain notch at its low point. A sensor model cannot put them
    back; only modelling them can."""
    material = material or WELL_PLASTIC
    rf = floor_d / 2.0
    p = MeshProgram()
    blade = prism([(-0.0038, -0.0024), (0.0038, -0.0024), (0.0032, 0.0024),
                   (-0.0032, 0.0024)], 0.0, depth * 0.55, mark="well")
    for i in range(ribs):
        a = TAU * (i + 0.5) / ribs
        r = max(rim_r - 0.005, keep_out * 1.12)
        p = p.place(obj=blade, at=(r * math.cos(a), r * math.sin(a), -depth + 0.002),
                    rotate=(0.0, 0.0, math.degrees(a)), material=material)
    # the step where the moulded liner meets the neck flange
    p = p.place(obj=lathe([(0.0, 0.0), (rf * 0.98, 0.0), (rf * 0.98, -0.0045),
                           (rf * 0.86, -0.0045), (0.0, -0.0045)], steps=40, mark="well"),
                at=(0.0, 0.0, -depth + 0.0055), material=material)
    if drain:
        # the notch at the low point, where water is meant to leave
        p = p.place(obj=prism(superellipse(0.008, 0.0045, 2.6, 14), 0.0, 0.011, mark="well"),
                    at=(0.0, -(rf - 0.005), -depth - 0.001), material=GRIME)
    # The hardware. Scattered on a deterministic pseudo-random ring rather than at fixed
    # angles: on the reference cars these sit wherever the moulding allowed, and a fixed
    # arrangement would be one more constant for a network to learn instead of the pose.
    #
    # `keep_out` is the cap's own radius, and everything here has to stay OUTSIDE it. The
    # radii used to be fractions of `rim_r` alone, and with the values the assembly passed
    # that put every screw and catch at 37-43 mm — which is precisely the cap's rim. So the
    # renders had bright steel rectangles lying on top of the black cap, in a frame whose
    # whole subject is that cap, in every variant that drew any hardware at all. It is the
    # loudest error in the whole set and no cloud metric could ever have reported it,
    # because a screw sitting on the cap still measures as a cap-shaped surface.
    import random
    rr = random.Random(seed)
    lo = max(keep_out * 1.10, rim_r * 0.50)
    hi = max(lo + 0.004, rim_r)
    ring = lambda f0, f1: lo + (hi - lo) * rr.uniform(f0, f1)
    for i in range(screws):
        a = rr.uniform(0, TAU)
        rad = ring(0.45, 0.95)
        p = p.place(obj=screw(r=rr.uniform(0.0028, 0.0042)),
                    at=(rad * math.cos(a), rad * math.sin(a), -depth * rr.uniform(0.35, 0.85)))
    for i in range(catches):
        a = rr.uniform(0, TAU)
        rad = ring(0.72, 1.0)
        p = p.place(obj=catch(w=rr.uniform(0.013, 0.020), h=rr.uniform(0.020, 0.032)),
                    at=(rad * math.cos(a), rad * math.sin(a), -depth * 0.75),
                    rotate=(0.0, 0.0, math.degrees(a) + 90.0))
    for i in range(grommets):
        a = rr.uniform(0, TAU)
        rad = ring(0.10, 0.70)
        p = p.place(obj=grommet(r=rr.uniform(0.0045, 0.0075)),
                    at=(rad * math.cos(a), rad * math.sin(a), -depth + 0.001))
    return p


def well(rim_r=0.062, floor_d=0.098, depth=0.052, neck_d=0.052, neck_len=0.055, lip=0.006,
         wall_taper=0.86, neck=True, material=None, steps=48):
    """The recess the cap sits in: a tapered cup with the filler neck down its middle.

    Modelled as a SURFACE OF REVOLUTION with real thickness rather than a hole cut out of a
    panel. A boolean through a large panel to make a 120 mm aperture is the obvious
    construction and the wrong one — it is by far the slowest op in the kit and it has to
    succeed on every one of thousands of frames. A cup placed *into* an annular panel gives
    the same silhouette, the same self-shadowing and the same depth map, and cannot fail.

    The section is one polyline from the axis to the axis, and it matters which way round it
    goes: down the neck, out across the floor, up the tapered wall to the rim, back under
    the flange and down the outside. Get the return leg wrong and it runs straight across
    the opening — which reads, in a render, as a pocket with a lid on it and no cap in
    sight. That is what the first version of this function did.

    `depth` is measured from the panel plane to the FLOOR, so the cap's own flange decides
    how proud of the floor its face ends up."""
    material = material or WELL_PLASTIC
    rn, rf = neck_d / 2.0, floor_d / 2.0
    rb = min(rf, rim_r * wall_taper)
    zf, zn = -depth, -depth - neck_len
    section = [
        (0.0, zn - 0.004),                 # the blind bottom of the neck, on the axis
        (rn, zn - 0.004),
        (rn, zf),                          # up the neck bore to the floor
        (rf, zf),                          # the floor annulus the cap sits on
        (rb, zf + lip),                    # fillet up onto the wall
        (rim_r, -0.001),                   # the tapered wall, out to the aperture
        (rim_r + 0.014, -0.011),           # the flange, tucked under the body panel
        (rim_r + 0.014, -0.016),
        (rim_r + 0.003, -0.016),
        (rim_r + 0.003, zn - 0.008),       # back down the outside
        (0.0, zn - 0.008),                 # and closed on the axis
    ]
    p = lathe(section, steps=steps, mark="well").material({"by": "tag", "name": "well"},
                                                          **material)
    if neck:
        # the steel filler-neck ring, just inside the bore — the bright ring you can see
        # around the cap's skirt on a real pocket
        p = p.place(obj=lathe([(0.0, 0.0), (rn * 1.02, 0.0), (rn * 1.02, -0.012),
                               (rn * 0.90, -0.012), (0.0, -0.012)], steps=36, mark="neck"),
                    at=(0.0, 0.0, zf + 0.001), material=NECK_STEEL)
    return p


_KS_CACHE = {}


def _KS(n, steps):
    key = (round(float(n), 4), int(steps))
    if key not in _KS_CACHE:
        _KS_CACHE[key] = _norm_superellipse(n, steps)
    return _KS_CACHE[key]


def _norm_superellipse(n, steps):
    """Superellipse radii for `steps` directions, scaled so their MEAN is 1.

    Shared by `panel` and `pocket_shaped` so the two meet edge to edge. They used the raw
    form and the normalised one respectively, which differ by 8% at n=3.6 — so the panel's
    hole and the pocket's outer ring were the same nominal size and different shapes, and
    the gap between them let the pocket's outer skirt show through. In the depth map that
    is a wall where the body should be, and it dragged the measured recess depth from +8 mm
    to -0.2."""
    if n <= 1.0:
        return [1.0] * steps
    ks = []
    for j in range(steps):
        a = TAU * j / steps
        c, s_ = math.cos(a), math.sin(a)
        ks.append((abs(c) ** n + abs(s_) ** n) ** (-1.0 / n))
    m = sum(ks) / len(ks)
    return [k / m for k in ks]


def panel(size=0.44, hole_d=0.135, thick=0.010, material=None, steps=48, ring=48,
          squareness=1.0, crown=0.0, crown_ax=0.0, hole_stretch=1.0, hole_ax=0.0):
    """The body panel around the pocket: an annular plate, not a plate with a hole in it.

    Same reasoning as `well` — the aperture is built, not subtracted. A ring of quads from
    the hole edge out to a square border is a dozen lines and never fails; the boolean it
    replaces is the single most expensive thing that could be in this loop."""
    material = material or {"color": [0.3, 0.3, 0.3], "metallic": 0.6, "roughness": 0.15}
    rh, s = hole_d / 2.0, size / 2.0

    # A car's flank is a shallow cylinder, not a plane, and at 0.4 m across a 0.5 m panel
    # that is 10-20 mm of sag — an order of magnitude more relief than anything inside the
    # recess. `crown` is that sag at the panel edge, `crown_ax` the direction it bends
    # about. Zero gives the flat plate this kit started with, which reads in a depth map as
    # a machined surface plate with a car's fuel cap sitting in it.
    ca, sa = math.cos(crown_ax), math.sin(crown_ax)

    def z_of(x, y, z0):
        if not crown:
            return z0
        t = (x * ca + y * sa) / max(s, 1e-9)
        return z0 - crown * t * t

    verts, faces = [], []
    for z in (0.0, -thick):
        for i in range(ring):
            a = TAU * i / ring
            c, sn = math.cos(a), math.sin(a)
            # `squareness` 1 = a circle, 4-5 = the rounded rectangle over half the reference
            # cars actually have. A perfect circle is the one aperture shape none of them is.
            k = _KS(squareness, ring)[i]
            px, py = rh * c * k, rh * sn * k
            verts.append((px, py, z_of(px, py, z)))
        for i in range(ring):
            # The border walks the SQUARE's perimeter by arc length, not by angle. Mapping
            # a ring of angles onto a square only lands on the corners if a sample happens
            # to fall at 45 degrees, and with any ring count that is not a multiple of 8 it
            # does not — which is why the first version of this panel was a rounded octagon
            # sitting behind every frame in the set.
            t = 8.0 * i / ring                         # 0..8, one unit per half-side
            side, u = int(t) % 8, t % 1.0
            cx_, cy_ = [(1, 0), (1, 1), (0, 1), (-1, 1), (-1, 0), (-1, -1), (0, -1), (1, -1)][side]
            nx_, ny_ = [(1, 1), (0, 1), (-1, 1), (-1, 0), (-1, -1), (0, -1), (1, -1), (1, 0)][side]
            px = s * (cx_ + (nx_ - cx_) * u)
            py = s * (cy_ + (ny_ - cy_) * u)
            verts.append((px, py, z_of(px, py, z)))
    N = 2 * ring
    for lay, flip in ((0, False), (N, True)):          # top face, bottom face
        for i in range(ring):
            j = (i + 1) % ring
            q = [lay + i, lay + j, lay + ring + j, lay + ring + i]
            faces.append(q[::-1] if flip else q)
    for i in range(ring):                              # the hole's inner wall
        j = (i + 1) % ring
        faces.append([i, N + i, N + j, j])
    for i in range(ring):                              # the outer edge
        j = (i + 1) % ring
        faces.append([ring + i, ring + j, N + ring + j, N + ring + i])
    return MeshProgram().mesh(verts=verts, faces=faces, mark="panel").material(
        {"by": "tag", "name": "panel"}, **material)


# The pocket's section, MEASURED — median height above the cap face against radius, over
# 376 real frames (`fit.radial_profile`). Millimetres.
#
# Read it before changing anything here. Outside the cap's own rim the surface drops to
# -33 mm at r = 44, climbs back through -16 at r = 56, crosses the cap's own plane at
# r = 63, and levels off on the body at +10. That is a deep annular trench around the cap,
# and it is most of what a depth sensor at this range actually sees.
#
# The hand-built version of this part had that whole span between -1 and -6 mm: a shallow
# saucer. Every feature in it was individually defensible and the shape between the
# features was flat, which is why the renders had a cap sitting alone on a coloured plate
# with nothing around it. A lathe IS a radius-to-height table and the clouds contain that
# table directly, so there was never a reason for it to come out of anyone's head.
MEASURED_SECTION_MM = [
    (34.0, -12.0), (36.0, -15.6), (40.0, -21.9), (44.0, -32.6), (48.0, -28.8),
    (52.0, -25.5), (56.0, -16.0), (60.0, -5.9), (64.0, 0.9), (68.0, 4.1),
    (72.0, 5.0), (76.0, 7.1), (80.0, 7.9), (84.0, 9.5), (88.0, 9.1),
    (92.0, 10.5), (96.0, 11.0), (100.0, 10.5),
]


def pocket_shaped(cap_r=0.037, r_gain=1.0, z_gain=1.0, neck_r=0.026, neck_len=0.045,
                  squareness=3.6, blend_from=0.085, steps=56, material=None, mark="well"):
    """The measured section, spun with a plan — three lines of engine, not a hand-built mesh.

    This used to assemble its own vertex grid here in the case. It does not need to: the
    section is what `spin` takes, and the round-inside/pressed-outside plan is what
    `spin(plan=..., plan_from=...)` does. `mirage.reverse` supplies both, and the C++ kernel
    builds it identically, which a hand-rolled vertex list in a case never would.
    """
    from mirage.reverse import section_to_profile, superellipse_plan

    sec = [(r / 1000.0 * r_gain, z / 1000.0 * z_gain) for r, z in MEASURED_SECTION_MM]
    sec = [(r, z) for r, z in sec if r > cap_r * 0.92]
    if len(sec) < 4:
        raise ValueError("cap covers the whole measured section")
    prof = section_to_profile(sec, floor=min(z for _, z in sec) - neck_len,
                              outer=sec[-1][0] * 1.02)
    p = (MeshProgram().profile(prof, plane="xz")
         .spin(axis="z", steps=steps, plan=superellipse_plan(squareness, steps),
               plan_from=blend_from, mark=mark))
    return p.material({"by": "tag", "name": mark}, **(material or WELL_PLASTIC))


def pocket(cap_r=0.037, r_gain=1.0, z_gain=1.0, neck_r=0.026, neck_len=0.045,
           out_r=0.115, steps=56, material=None, mark="well"):
    """The whole pocket as ONE lathe, straight off the measured section.

    Replaces the well / dish / aperture stack, which tried to reach this shape by adding
    three hand-built parts and could not: the trench is a single continuous surface and
    was being approximated by a saucer, a step and a flat panel.

    `blend_from` has to sit where the SECTION IS ALREADY FLAT, and that is not a detail.
    A plan makes physical radius r sample the profile at r/k, and h(r) through the trench
    wall is strongly concave — so averaging h(r/k) over directions lands BELOW h(r) by
    Jensen, systematically, however the plan is normalised. Starting the blend at 55 mm,
    in the middle of that wall, put the whole outer surface 8 mm low and cost seven
    attempts at a metric that was reporting it correctly the entire time.

    `r_gain` and `z_gain` are the generalisation — they scale the measured section
    radially and in depth, so one real vehicle's profile becomes a family. `cap_r` slides
    where the section starts, since a bigger cap covers more of the trench's inner wall.
    Imitate first, then generalise; the defaults imitate.
    """
    sec = [(r / 1000.0 * r_gain, z / 1000.0 * z_gain) for r, z in MEASURED_SECTION_MM]
    sec = [(r, z) for r, z in sec if r > cap_r * 0.92]      # the cap covers the rest
    if len(sec) < 4:
        raise ValueError("cap covers the whole measured section")
    z_deep = min(z for _, z in sec) - 0.006
    outer = max(out_r, sec[-1][0] + 0.004)
    # One polyline, axis to axis: out along the underside, up the outer edge, IN along the
    # measured surface, then down the neck bore and back to the axis under it.
    section = [(0.0, z_deep - 0.010), (outer, z_deep - 0.010), (outer, sec[-1][1])]
    section += [(r, z) for r, z in reversed(sec)]
    section += [(max(neck_r, 0.004), sec[0][1] - 0.004),
                (max(neck_r, 0.004), -neck_len),
                (0.0, -neck_len)]
    p = lathe(section, steps=steps, mark=mark)
    return p.material({"by": "tag", "name": mark}, **(material or WELL_PLASTIC))


def pressed_dish(cap_r=0.039, gap=0.006, throat=0.030, wall_z=0.030, out_r=0.105,
                 squareness=3.4, blend_from=0.070, steps=56, rim=0.004,
                 material=None, mark="panel"):
    """The OTHER pocket: a shallow dish pressed into painted sheet metal, cap in its floor.

    Half the reference cars do not have a moulded black liner at all. They have the body's
    own panel drawn into a shallow squarish dish — silver, white, yellow, whatever the car
    is — with a round hole in its floor that the cap very nearly fills, a short vertical
    throat down from that hole, and a couple of black rubber bumpers for the door to close
    onto. The whole kit only had the black-liner family, so every synthetic frame was the
    same half of the world, and it was the darker half: painting this dish body colour
    changes the exposure of the entire scene, because the biggest surface near the cap stops
    being a light trap and starts being the brightest thing in the frame.

    Built as one lathe with a squarish plan, same machinery as `pocket_shaped`. The section
    goes: down the throat, out across the narrow floor, straight UP the throat wall (it is a
    drawn edge, near vertical — this is the detail a radius-binned cloud smears into a ramp
    and a photograph does not), then out along the dish floor and up onto the body.
    """
    from mirage.reverse import section_to_profile, superellipse_plan
    rh = cap_r + gap                                   # the hole the cap sits in
    sec = [
        (rh, -throat),                                 # the bottom of the drawn throat
        (rh * 1.02, -throat * 0.55),
        (rh * 1.06, -0.002),                           # up to the dish floor, near vertical
        (rh * 1.35, 0.0),
        (rh * 1.35 + (out_r - rh * 1.35) * 0.55, wall_z * 0.62),   # out and up the dish wall
        (out_r - rim, wall_z),
        (out_r, wall_z + rim * 0.25),                  # the lip, level with the paint
    ]
    prof = section_to_profile(sec, floor=-throat - 0.030, outer=out_r * 1.02)
    p = (MeshProgram().profile(prof, plane="xz")
         .spin(axis="z", steps=steps, plan=superellipse_plan(squareness, steps),
               plan_from=blend_from, mark=mark))
    return p.material({"by": "tag", "name": mark}, **(material or WELL_PLASTIC))


def bumper(r=0.007, h=0.006, material=None):
    """The rubber stop the fuel door closes onto — two or three round a pressed dish.

    Small, black, matt, and sitting on a body-coloured panel, so it is one of the few
    high-contrast marks in that half of the reference set and it lands well inside the ROI."""
    return lathe([(0.0, h), (r * 0.5, h), (r, h * 0.55), (r, 0.0), (0.0, 0.0)],
                 steps=16, mark="well").material({"by": "tag", "name": "well"},
                                                 **(material or GROMMET))


def door_pan(outer=0.150, depth=0.011, hole_r=0.048, squareness=4.0, wall=0.014,
             ring=48, material=None):
    """The shallow squarish dish the fuel DOOR sits in, with the filler aperture in its floor.

    A car has two recesses here, not one, and this kit only had the inner one. The outer is
    the pressing the door lies flush in when it is shut — roughly 150 mm across, 10 mm deep,
    a rounded rectangle — and the round filler aperture is a hole in ITS floor.

    It is the most important thing that was missing, because of where it is: the ROI is the
    cap's bounding box times 3.9, about 270 mm across, and this dish fills the middle of it.
    The body structure added before it — crown, shut line — is all further out than that and
    never appears in a single cropped frame. Structure only counts if it lands inside the
    crop that ships."""
    material = material or {"color": [0.3, 0.3, 0.3], "metallic": 0.6, "roughness": 0.15}
    R = outer / 2.0
    verts, faces = [], []
    lips = superellipse(R, R * 0.94, squareness, ring)            # the rim, at z = 0
    fl = superellipse(R - wall, R * 0.94 - wall, squareness, ring)  # the floor outline
    for x, y in lips:
        verts.append((x, y, 0.0))
    for x, y in fl:
        verts.append((x, y, -depth))
    for i in range(ring):                                          # the drafted side wall
        j = (i + 1) % ring
        faces.append([i, j, ring + j, ring + i])
    for i in range(ring):                                          # the floor, out to the hole
        j = (i + 1) % ring
        a = TAU * i / ring
        b = TAU * j / ring
        verts.append((hole_r * math.cos(a), hole_r * math.sin(a), -depth))
    base = 2 * ring
    for i in range(ring):
        j = (i + 1) % ring
        faces.append([ring + i, ring + j, base + j, base + i])
    return MeshProgram().mesh(verts=verts, faces=faces, mark="panel").material(
        {"by": "tag", "name": "panel"}, **material)


def shutline(panel_size, seam_x, gap=0.005, step=0.0035, side=1.0, material=None):
    """The car's own panel gap — one sheet lapped over another with a slot between them.

    Built as a second skin standing `step` proud of the body panel and stopping `gap/2`
    short of the seam, so the slot is a REAL slot: rays go down it and the depth map gets a
    3-4 mm trench, exactly as they do on the reference frames. Painting a dark stripe on a
    flat panel would look the same in RGB and would be invisible in the cloud, which is the
    half that matters here.

    This is the single largest piece of what `fit.complexity` says is missing. The recess's
    own furniture is small; the ROI is mostly *body*, and a real body is not one plane."""
    s = panel_size / 2.0
    # The skin lies on ONE side of the seam — the side away from the pocket. It used to
    # always run from the seam toward +x, so with the seam placed at negative x the sheet
    # covered the pocket instead of sitting beside it: a 3 mm slab straight over the
    # recess, which in a render is a crescent of well peeping out from under a panel and
    # in an id map looks like the pocket itself is broken. It is not; it is buried.
    if side >= 0:
        poly = [(seam_x + gap / 2, -s), (s, -s), (s, s), (seam_x + gap / 2, s)]
    else:
        poly = [(-s, -s), (seam_x - gap / 2, -s), (seam_x - gap / 2, s), (-s, s)]
    return (prism(poly, 0.0, step, mark="panel")
            .material({"by": "all"}, **(material or {"color": [0.3, 0.3, 0.3],
                                                     "metallic": 0.6, "roughness": 0.15})))


def hinge_arm(length=0.075, w=0.020, t=0.0025, holes=2, material=None):
    """The flat steel strap that carries the fuel door.

    This is the most conspicuous thing about an open filler pocket after the cap itself,
    and the model had two little blocks where it goes. In the reference photographs it is
    a wide, thin strap — 15 to 25 mm across, 2 to 3 mm thick — running from the door's edge
    back to the body, usually with a couple of lightening holes or a pressed rib along it,
    and it is bare or lightly painted steel in a pocket that is otherwise matt black, so it
    catches the light and reads as bright.

    Built as two rails either side of each hole rather than as a plate with holes cut in
    it: same silhouette, and no boolean in a loop that has to run ten thousand times."""
    material = material or CATCH_STEEL
    p = MeshProgram()
    if holes <= 0:
        p = p.place(obj=prism([(0.0, -w / 2), (length, -w / 2), (length, w / 2),
                               (0.0, w / 2)], 0.0, t, mark="door"), at=(0, 0, 0),
                    material=material)
        return p
    # a rail down each side, plus a bridge at each end and between holes
    rail = w * 0.26
    for sgn in (-1, 1):
        p = p.place(obj=prism([(0.0, sgn * (w / 2 - rail)), (length, sgn * (w / 2 - rail)),
                               (length, sgn * w / 2), (0.0, sgn * w / 2)][::int(sgn)],
                              0.0, t, mark="door"), at=(0, 0, 0), material=material)
    for k in range(holes + 1):
        x = length * k / holes
        bw = length / (holes * 3.2)
        p = p.place(obj=prism([(x - bw / 2, -w / 2), (x + bw / 2, -w / 2),
                               (x + bw / 2, w / 2), (x - bw / 2, w / 2)], 0.0, t,
                              mark="door"), at=(0, 0, 0), material=material)
    return p


def trim_ring(r_in=0.062, r_out=0.076, thick=0.003, screws=6, steps=44, material=None):
    """The chromed trim ring some cars put round the filler aperture, with its screws.

    A minority fitment but an unmistakable one: a bright annulus in a scene whose whole
    subject is matt black, which changes what the ROI looks like far more than its size
    suggests."""
    material = material or CAP_CHROME
    p = lathe([(0.0, 0.0), (r_out, 0.0), (r_out, -thick), (r_in, -thick * 0.55),
               (r_in, 0.0), (0.0, 0.0)], steps=steps, mark="panel")
    p = p.material({"by": "tag", "name": "panel"}, **material)
    rm = (r_in + r_out) / 2
    for i in range(screws):
        a = TAU * (i + 0.5) / screws
        p = p.place(obj=screw(r=0.0026, head_h=0.0012),
                    at=(rm * math.cos(a), rm * math.sin(a), 0.0005))
    return p


def door(w=0.175, h=0.165, thick=0.008, open_deg=95.0, hinge_x=-0.10, skin=None, liner=None,
         rim=0.012, ribs=3, hinge=True, arm_len=0.075, arm_w=0.020, arm_holes=2,
         label=False):
    """The fuel door, hinged open. Two materials: body paint outside, dark liner inside.

    It is in this scene because it is in every real frame, and because it is the largest
    single thing bouncing light back INTO the pocket — take it away and the cap's own
    shading changes, which is a geometry error that would show up as a normal error."""
    skin = skin or {"color": [0.3, 0.3, 0.3], "metallic": 0.6, "roughness": 0.15}
    liner = liner or WELL_PLASTIC
    # The plan runs 0..w in x, NOT -w/2..w/2: the hinge is the door's own edge, so the
    # rotation has to be about that edge. Centring it instead swings half the door back
    # THROUGH the body panel and leaves the visible half sticking out like a shelf, which
    # is what the first version did in every frame.
    body = prism([(0.0, -h / 2), (w, -h / 2), (w, h / 2), (0.0, h / 2)],
                 0.0, thick, mark="door")
    # Paint every face first, THEN the two big ones. Selecting only +z and -z leaves the
    # four edge faces on the renderer's default albedo, which is a bright warm grey — a
    # 5 mm white rim all the way round a black fuel door, and the brightest thing in the
    # frame in a scene whose whole subject is dark.
    body = body.material({"by": "all"}, **liner)
    body = body.material({"by": "normal", "axis": "z", "sign": 1}, **skin)
    body = body.material({"by": "normal", "axis": "z", "sign": -1}, **liner)
    # A real fuel door is a pressing with a MOULDED LINER clipped into it: an outer skin
    # in body colour, a return flange round the edge, and one dark plastic panel covering
    # most of the inside. Not a set of exposed ribs — that reads as a louvre or a tray, and
    # is what the first version rendered in every frame. The liner sits a few millimetres
    # inside the flange, so its edge shows as a thin dark step rather than as slats.
    #
    # This face matters more than the outside: with the door open it is what points back at
    # the camera, so it is the side that lands in the cloud.
    if rim > 0:
        for dx, dy, lx, ly in ((0, -h / 2, w, 0.0), (0, h / 2, w, 0.0),
                               (0.0, 0, 0.0, h), (w, 0, 0.0, h)):
            bx = max(lx, 0.004) / 2.0
            by = max(ly, 0.004) / 2.0
            body = body.place(obj=prism([(-bx, -by), (bx, -by), (bx, by), (-bx, by)],
                                        0.0, -rim, mark="door"),
                              at=(dx, dy, 0.0), material=liner)
        inset = 0.010
        body = body.place(obj=prism([(inset, -h / 2 + inset), (w - inset, -h / 2 + inset),
                                     (w - inset, h / 2 - inset), (inset, h / 2 - inset)],
                                    -rim * 0.30, -rim * 0.75, mark="door"),
                          at=(0.0, 0.0, 0.0), material=liner)
    # hinged along the door's own -x edge, swung out about the panel's y axis. The door
    # opens AWAY from the pocket, so `hinge_x` is negative and `open_deg` positive.
    p = MeshProgram().place(obj=body, at=(hinge_x, 0.0, 0.0), rotate=(0.0, -open_deg, 0.0))
    if hinge:
        # The strap runs from the door's hinged edge back onto the body, in the plane of
        # the body rather than the plane of the door — it is what holds the door out.
        p = p.place(obj=hinge_arm(length=arm_len, w=arm_w, holes=arm_holes),
                    at=(hinge_x - arm_len + 0.004, 0.0, -0.004))
    return p


def tether(start, end, sag=0.030, r=0.0022, n=90, kinks=2, seed=0, material=None):
    """The cord from the cap to the pocket — swept along a path that actually hangs.

    Rebuilt against the photographs. The old one was a neat stretched helix, which reads as
    a spring in a catalogue; a real tether is a short moulded cord that has been bent the
    same way ten thousand times, so it hangs with two or three lazy kinks in it and no
    periodicity at all. It is also thicker than it looks — 2 to 3 mm — and it is one of the
    few BRIGHT-edged things in a pocket full of matt black, so its silhouette matters.

    Built as a Catmull-Rom spline through a few jittered control points rather than as a
    formula, because a formula is exactly what made the old one look manufactured."""
    import random
    rng = random.Random(seed)
    s_, e = np.asarray(start, float) if False else list(start), list(end)
    ctrl = [list(s_)]
    for i in range(1, kinks + 2):
        t = i / (kinks + 2)
        base = [s_[k] + (e[k] - s_[k]) * t for k in range(3)]
        amp = math.sin(math.pi * t)
        base[0] += rng.uniform(-0.014, 0.014) * amp
        base[1] += rng.uniform(-0.016, 0.016) * amp
        base[2] -= sag * amp * rng.uniform(0.75, 1.25)
        ctrl.append(base)
    ctrl.append(list(e))
    # Catmull-Rom through the control points
    pad = [ctrl[0]] + ctrl + [ctrl[-1]]
    path = []
    for i in range(len(pad) - 3):
        p0, p1, p2, p3 = (np.array(pad[i + k], float) for k in range(4))
        for j in range(n // (len(pad) - 3)):
            t = j / max(1, n // (len(pad) - 3))
            t2, t3 = t * t, t * t * t
            q = 0.5 * ((2 * p1) + (-p0 + p2) * t + (2 * p0 - 5 * p1 + 4 * p2 - p3) * t2
                       + (-p0 + 3 * p1 - 3 * p2 + p3) * t3)
            path.append([float(x) for x in q])
    path.append(list(e))
    # Drop points the spline put on top of each other. A Catmull-Rom through jittered
    # controls can double back within a fraction of a millimetre, and `sweep` turns two
    # coincident path points into a zero-area quad — which fails the whole frame for a
    # cord that is three pixels wide.
    clean = [path[0]]
    for q in path[1:]:
        d = sum((q[k] - clean[-1][k]) ** 2 for k in range(3)) ** 0.5
        if d > r * 0.9:
            clean.append(q)
    # A sweep also degenerates where the path turns back on itself sharply: the ring at
    # the corner sweeps through zero area. Drop points whose turn exceeds ~100 degrees.
    out2 = clean[:1]
    for k in range(1, len(clean) - 1):
        a = [clean[k][j] - out2[-1][j] for j in range(3)]
        b = [clean[k + 1][j] - clean[k][j] for j in range(3)]
        na = sum(x * x for x in a) ** 0.5
        nb = sum(x * x for x in b) ** 0.5
        if na < 1e-9 or nb < 1e-9:
            continue
        cosang = sum(a[j] * b[j] for j in range(3)) / (na * nb)
        if cosang > -0.17:
            out2.append(clean[k])
    out2.append(clean[-1])
    path = out2 if len(out2) >= 4 else (clean if len(clean) >= 4 else path[:4])
    ring = [(r * math.cos(TAU * k / 8), r * math.sin(TAU * k / 8)) for k in range(8)]
    return (MeshProgram().profile(ring, plane="xy", closed=True).sweep(path, mark="tether")
            .material({"by": "tag", "name": "tether"}, **(material or TETHER)))


def cap_boss(r_cap, spin=0.0, material=None):
    """The little lug on the cap's rim that the tether is moulded onto.

    Small, and present on every cap in the reference set that has a cord. Without it the
    cord appears to grow out of a smooth disc, which is the sort of detail that is invisible
    until it is missing."""
    a = math.radians(spin)
    return (prism([(-0.006, -0.004), (0.006, -0.004), (0.005, 0.004), (-0.005, 0.004)],
                  0.0, 0.005, mark="cap_body")
            .material({"by": "all"}, **(material or CAP_BLACK)))
