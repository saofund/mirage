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


def lobe_plan(lobes, depth, steps, phase=0.0):
    """A `spin` plan with `lobes` rounded bumps round it — the petal-edged cap.

    A handful of the reference cars have a cap whose rim is not a circle but a ring of
    shallow lobes you grip with your fingertips. That is a plan, not a section, so it is
    exactly what the generalised lathe is for: same profile, different plan, one line.
    Normalised so adding lobes does not also change the cap's diameter."""
    ks = [1.0 + depth * math.cos(lobes * (TAU * j / steps) + phase) for j in range(steps)]
    m = sum(ks) / len(ks)
    return [k / m for k in ks]


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
def cap(d=0.078, flange=0.009, rib_len=0.052, rib_w=0.024, rib_h=0.011, rib_draft=0.66,
        rib_slot=0.0, dome=0.0008, chamfer=0.0025, teeth=0, skirt=0.020, neck_d=0.048,
        spin=0.0, printing=True, grip="rib", lobes=0, lobe_depth=0.06,
        material=None, rib_material=None, steps=48):
    """The inner fuel cap: a lathed disc with a raised grip rib across it.

    Everything about this part that a 6D-pose network can see is in three numbers — the
    disc diameter, the rib's footprint and the rib's height — and all three were measured
    off the real clouds rather than chosen: disc 70–78 mm, rib 43 x 18 mm at its top face,
    rib 3.4 mm proud of the annulus in the production set's grown patch and 10–13 mm proud
    of the disc rim once the whole disc is included.

    `rib_slot` cuts the groove that runs down the middle of most of these grips; 0 leaves
    it solid. `teeth` adds the ratchet ring around the skirt (the clicking torque limiter)
    — invisible head-on, plainly visible at the oblique angles this camera actually works
    at, which is exactly the kind of detail that only matters off-axis.

    The rib is `place`d onto the disc rather than booleaned into it. They interpenetrate by
    design: a BSP union of a 48-segment lathe with a drafted stadium is slow and fragile,
    and at 200 px across the seam it would buy is smaller than a pixel. What would be
    visible is a missing rib on one frame in fifty because a boolean failed, so this trades
    a fillet nobody can resolve for a mesh that always builds.

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

    # Section, axis outward. z = 0 is the sealing face — the plane this whole case labels.
    section = [
        (0.0, dome),                       # a barely-domed centre; real caps are not flat
        (r * 0.45, dome * 0.75),
        (r - c, 0.0),                      # the annulus: the plane the normal belongs to
        (r, -c),                           # rim chamfer
        (r, -flange + c),                  # outer wall
        (r - c, -flange),                  # underside chamfer
        (rn, -flange),
        (rn, -flange - skirt),             # the skirt that goes down the filler neck
        (rn * 0.72, -flange - skirt),
        (0.0, -flange - skirt),
    ]
    plan = lobe_plan(lobes, lobe_depth, steps, math.radians(spin)) if lobes else None
    p = (MeshProgram().profile([(rr, zz) for rr, zz in section], plane="xz", closed=False)
         .spin(axis="z", steps=steps, plan=plan, plan_from=r * 0.72, mark="cap_body")
         .material({"by": "tag", "name": "cap_body"}, **material))

    if teeth:
        # The ratchet ring — the clicking torque limiter. On the OUTER RIM, where a camera
        # can see it: this used to put the teeth on the skirt, which is down inside the
        # filler neck and hidden by the cap itself in every frame ever rendered. In the
        # reference photographs the teeth are a fine scalloped edge right round the
        # silhouette, and on an obliquely-viewed cap they are most of what says "this is a
        # moulding you twist" rather than "this is a disc".
        tw = TAU * r / teeth * 0.55
        for i in range(teeth):
            a = TAU * i / teeth
            # shallow: the teeth are a scalloped edge a millimetre proud, not a gear. The
            # first version stood 2 mm out on a 39 mm radius and turned the cap into a
            # cog — recognisable, but not this part.
            p = p.place(obj=prism([(-tw / 2, -0.0010), (tw / 2, -0.0010),
                                   (tw / 2 * 0.66, 0.0010), (-tw / 2 * 0.66, 0.0010)],
                                  0.0, flange * 0.72, mark="cap_teeth"),
                        at=(r * 0.9975 * math.cos(a), r * 0.9975 * math.sin(a), -flange * 0.74),
                        rotate=(0.0, 0.0, math.degrees(a)), material=material)

    # The grip, turned about the cap's own axis by `spin`. TWO families, both common:
    #
    #   grip="rib"   a raised bar across the face — what this kit had
    #   grip="slot"  a rectangular WELL sunk into the face, with walls and a small pad in
    #                its floor — about a quarter of the reference cars, and the shape is
    #                not a rib at all. Building it as a rib and hoping was the same mistake
    #                as everywhere else on this part: it is a different surface, not a
    #                parameter of the one I already had.
    sa, ca = math.sin(math.radians(spin)), math.cos(math.radians(spin))
    turn = lambda pl: [(x * ca - y * sa, x * sa + y * ca) for x, y in pl]

    if grip == "slot":
        # The land has to be TALL and the well can only be as deep as the land, because
        # what is underneath is the cap body's own solid top face. Sinking the floor past
        # it just buries the well: the body's face is what the camera then sees, which is
        # why this rendered as a flat disc with a pad on it through three attempts.
        land = max(rib_h * 0.95, 0.008)
        depth = land * 0.90
        p = p.place(obj=slotted_face(r - chamfer * 0.8, turn(stadium(rib_len, rib_w, arc=22)),
                                     dome + land, dome + land - depth,
                                     draft=max(rib_draft, 0.74), mark="cap_slot"),
                    at=(0.0, 0.0, 0.0), material=material)
        # the pad in the floor of the well — every photographed slot has one
        p = p.place(obj=frustum(turn(stadium(rib_len * 0.44, rib_w * 0.40)),
                                dome + land - depth, dome + land - depth * 0.45, 0.86,
                                mark="cap_rib"),
                    at=(0.0, 0.0, 0.0), material=rib_material)
        return p

    plan = turn(stadium(rib_len, rib_w))
    p = p.place(obj=frustum(plan, dome * 0.5, rib_h, rib_draft, mark="cap_rib"),
                at=(0.0, 0.0, 0.0), material=rib_material)
    if rib_slot > 0:
        # The groove down the spine of a raised bar — the double-decker variant, and very
        # common. Sunk into the bar's TOP, narrower than the bar and stopping short of its
        # ends, so the bar reads as two rails rather than as one block.
        sl = rib_len * rib_draft * 0.80
        sw = rib_w * rib_draft * 0.42
        d2 = min(rib_slot, rib_h * 0.45)
        p = p.place(obj=frustum(turn(stadium(sl, sw)), rib_h + 1e-4, rib_h - d2, 0.88,
                                mark="cap_slot"),
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
                 catches=1, grommets=1, seed=0, material=None):
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
        r = rim_r - 0.005
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
    import random
    rr = random.Random(seed)
    for i in range(screws):
        a = rr.uniform(0, TAU)
        rad = rim_r * rr.uniform(0.80, 0.94)
        p = p.place(obj=screw(r=rr.uniform(0.0028, 0.0042)),
                    at=(rad * math.cos(a), rad * math.sin(a), -depth * rr.uniform(0.05, 0.35)))
    for i in range(catches):
        a = rr.uniform(0, TAU)
        rad = rim_r * 0.92
        p = p.place(obj=catch(w=rr.uniform(0.013, 0.020), h=rr.uniform(0.020, 0.032)),
                    at=(rad * math.cos(a), rad * math.sin(a), -depth * 0.55),
                    rotate=(0.0, 0.0, math.degrees(a) + 90.0))
    for i in range(grommets):
        a = rr.uniform(0, TAU)
        rad = rim_r * rr.uniform(0.55, 0.80)
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
                  squareness=3.6, blend_from=0.055, steps=56, material=None, mark="well"):
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


def shutline(panel_size, seam_x, gap=0.005, step=0.0035, material=None):
    """The car's own panel gap — one sheet lapped over another with a slot between them.

    Built as a second skin standing `step` proud of the body panel and stopping `gap/2`
    short of the seam, so the slot is a REAL slot: rays go down it and the depth map gets a
    3-4 mm trench, exactly as they do on the reference frames. Painting a dark stripe on a
    flat panel would look the same in RGB and would be invisible in the cloud, which is the
    half that matters here.

    This is the single largest piece of what `fit.complexity` says is missing. The recess's
    own furniture is small; the ROI is mostly *body*, and a real body is not one plane."""
    s = panel_size / 2.0
    return (prism([(seam_x + gap / 2, -s), (s, -s), (s, s), (seam_x + gap / 2, s)],
                  0.0, step, mark="panel")
            .material({"by": "all"}, **(material or {"color": [0.3, 0.3, 0.3],
                                                     "metallic": 0.6, "roughness": 0.15})))


def door(w=0.175, h=0.165, thick=0.008, open_deg=95.0, hinge_x=-0.10, skin=None, liner=None,
         rim=0.012, ribs=3, hinge=True):
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
        for sy in (-1, 1):
            p = p.place(obj=prism([(-0.004, -0.0035), (0.030, -0.0035),
                                   (0.030, 0.0035), (-0.004, 0.0035)], 0.0, 0.010,
                                  mark="door"),
                        at=(hinge_x - 0.004, sy * h * 0.32, -0.004), material=liner)
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
