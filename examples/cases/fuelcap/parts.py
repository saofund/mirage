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


def handle(length, w_end, w_mid, height, dish=0.20, sag=0.10, ends_down=0.004,
           base=0.003, tip=1.0, shoulder=0.80, stations=34, mark="cap_rib"):
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
    # The scoop is WIDE and SHALLOW — the centre is a fifth down, not a third. Narrow and
    # deep it reads as a groove milled down the bar rather than as a place to put a thumb,
    # and with a smoothed normal it puts two bright rails and a black valley on a part the
    # photographs show as one soft surface.
    #
    # `shoulder` is where the flat top starts, as a fraction of the half width, and `dish`
    # may be NEGATIVE for a crowned ridge instead of a scooped one. Both matter more than
    # they look: at 0.80 with a positive dish the bar is a flat plateau with near-vertical
    # flanks covering two thirds of the cap, the wings shrink to slivers and the printing
    # ends up on the rim. Photographed caps split roughly half and half between a scooped
    # bar with a wide top and a crowned ridge whose top is barely half its base, and the
    # difference is the whole silhouette of the part.
    sh = max(0.05, min(0.97, shoulder))
    # The point between the shoulder and the centre is a fraction of the SHOULDER's own
    # position, not of a value interpolated toward the rim. Written the other way it lands
    # exactly on the shoulder once `shoulder` drops below about 0.5 — two profile points at
    # the same x and different heights, which is a vertical wall, and it renders as a hard
    # crease running the length of the bar.
    mid = sh * 0.52
    prof = [
        (-hw, -base), (hw, -base),                 # buried base, well below the face
        (hw, height * 0.34), (hw * (1.0 - (1.0 - sh) * 0.30), height * 0.88), (hw * sh, height),
        (hw * mid, height * (1.0 - dish * 0.40)),
        (0.0, height * (1.0 - dish)),              # the scoop (or crown) across the bar
        (-hw * mid, height * (1.0 - dish * 0.40)),
        (-hw * sh, height), (-hw * (1.0 - (1.0 - sh) * 0.30), height * 0.88), (-hw, height * 0.34),
    ]
    waist = max(0.0, 1.0 - w_mid / max(w_end, 1e-6))
    path, scale = [], []
    for j in range(stations):
        u = j / (stations - 1.0)
        s = math.sin(math.pi * u)
        # `tip` narrows the plan over the last tenth at each end, and DEFAULTS TO OFF.
        #
        # It went in to stop the bar's corner poking past the cap's rim, and it is the
        # wrong tool for that. Overlaying the lit region of a real bar back onto its
        # photograph shows the ends are its WIDEST part and finish in a blunt, near-vertical
        # wall: a squared-off hourglass, not a leaf. Every value below one produced a
        # pointed end plus a visible kink where the ramp stopped, and both are in the
        # renders that led to this measurement. Keeping the bar inside the rim is the length
        # clamp's job, which measures the built bar and simply shortens it.
        t = 0.5 - 0.5 * math.cos(math.pi * min(1.0, min(u, 1.0 - u) / 0.11))
        # ends below the face, middle at it: `ends_down` is what buries the open rings
        path.append((length * (u - 0.5), 0.0, -ends_down * (1.0 - s ** 0.55)))
        scale.append(((tip + (1.0 - tip) * t) * (1.0 - waist * s ** 1.2),
                      1.0 - sag * s))
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
        rib_slot=0.0, dome=0.0008, chamfer=0.0025, flutes=12, flute_depth=0.042,
        skirt=0.020, neck_d=0.048, bevel=0.055, spin=0.0, printing=True, grip="rib",
        lobes=0, lobe_depth=0.06, waist=0.71, rib_dish=0.20, rib_shoulder=0.80,
        groove_r=None, groove_w=0.0016, groove_d=0.0,
        decal="fuelcap_face",
        material=None, rib_material=None, steps=64):
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
    r = d / 2.0
    rib_len = d * 0.90 if rib_len is None else rib_len
    rib_w = d * 0.36 if rib_w is None else rib_w
    rib_h = d * 0.070 if rib_h is None else rib_h
    if printing:
        # The warning text round the annulus. Four 460 px reference photographs all have
        # it and none of the synthetic caps did — it is the most recognisable single
        # feature of this part in a colour image, and it is free: the tracer pins the
        # artwork to a rectangle on the cap's own +z face.
        from mirage.decals import ensure_decals
        from .materials import with_decal
        # `decal` names the artwork. The generic one carries a plausible cap's markings, and
        # plausible is the wrong target when the render is going next to the photograph it
        # is copying — so a reproduction hands in that cap's own transcribed printing.
        art = ensure_decals([decal])[decal]
        material = with_decal(material, art, d * 1.02, d * 1.02, max(dome, 0.0) + 1e-4)
    # The handle takes the SAME material as the body, decal and all, and it has to. An
    # albedo map REPLACES the flat colour where it hits (raytrace.cpp `alb = tmp`), so a
    # printed body renders at the artwork's background — 0.0196 — while a handle carrying
    # the material's own jittered colour renders at whatever that instance drew, up to four
    # times brighter. Every printed cap therefore had a pale handle stuck on a black cap,
    # which looks like a lighting artefact and is really two different albedos.
    #
    # It maps cleanly because the decal's text arcs are laid out to leave the sides clear,
    # which is where the handle's ends are; the middle of the artwork is bare background.
    rib_material = rib_material or material
    rn = min(neck_d / 2.0, r - 0.004)
    c = min(chamfer, flange * 0.45, r * 0.08)
    rf = r * (1.0 - bevel)                 # where the flat top face stops and the bevel starts

    # Section, axis outward. z = 0 is the sealing face — the plane this whole case labels —
    # and it is the flat top the printing sits on, so the bevel and the whole skirt hang
    # BELOW it. `flange` is the visible height of that skirt.
    # `dome` is usually NEGATIVE here, and that is the shape the photographs have: the face
    # is very slightly DISHED inside a raised ring at `rf`, not crowned. Head-on that ring is
    # the bright line just inside the silhouette on every reference cap, and it is what makes
    # the face read as let into the moulding rather than as the top of a cylinder.
    # THE GROOVE. Every cap in the reference set has a narrow turned groove a few
    # millimetres inside its rim -- it is where the moulding's two halves meet, and it is
    # the one feature that makes a cap read as a turned part rather than as a disc. It costs
    # four points in the section and it is the difference between "a grey circle" and "a
    # fuel cap" at the magnification anybody actually judges these at.
    gr = groove_r if groove_r else rf * 0.86
    gw = max(groove_w, 1e-4)
    section = [
        (0.0, dome),
        (rf * 0.55, dome * 0.80),
    ] + ([
        (gr - gw, dome * 0.35),
        (gr - gw * 0.35, -groove_d),       # down into the groove
        (gr + gw * 0.35, -groove_d),
        (gr + gw, dome * 0.30),
    ] if groove_d > 0 else []) + [
        (rf, 0.0),                         # the raised ring, and the plane the label means
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
        # The land is a nuisance, not a feature: on the photographed slot caps the well is
        # sunk straight into the face and there is no raised platform at all. It exists here
        # only because `place` unions, so a floor cannot go below the cap body's own top
        # face — the well has to be lifted just far enough to have somewhere to sink into.
        # At 8 mm that lift stood taller than the handle on the other grip family and put a
        # tower in the middle of the cap; 3.5 mm reads as a sunk grip.
        land = max(rib_h * 0.45, 0.0035)
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

    # Keep the bar inside the cap's silhouette, MEASURED rather than estimated. The path
    # dips steeply at the ends to bury the open rings, so the section there leans by nearly
    # forty degrees and its top corner reaches several millimetres past the last station —
    # diagonally, so what matters is a radius, not an x. Two closed-form guesses at that
    # overhang were both wrong (one by 4 mm, one so conservative it cost a fifth of the
    # bar's length), and the mesh is 370 vertices, so it is cheaper to ask it.
    bar = lambda L: handle(L, rib_w, rib_w * waist, rib_h, dish=rib_dish, sag=0.10,
                           ends_down=rib_h * 1.15, base=0.003,
                           shoulder=rib_shoulder, mark="cap_rib")
    lim, L = r * 0.995, rib_len
    for _ in range(4):
        co = [v.co for v in bar(L).build().verts]
        reach = max(math.hypot(c[0], c[1]) for c in co)
        if reach <= lim:
            break
        L -= 2.0 * (reach - lim)
    p = p.place(obj=bar(L), at=(0.0, 0.0, min(dome, 0.0)), rotate=(0.0, 0.0, spin),
                material=rib_material)
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
                    at=(0.0, -max(rf - 0.005, keep_out + 0.011), -depth - 0.001),
                    material=GRIME)
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
    # `lo` clears the cap by the widest HALF-EXTENT anything here has (a catch is 20 mm
    # across), not merely by its centre. Clearing centres is what let a 16 mm bracket
    # centred 4 mm outside the cap still lie half on top of it.
    lo = max(keep_out + 0.013, rim_r * 0.50)
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
          squareness=1.0, crown=0.0, crown_ax=0.0, crown_cross=0.0, crown_at=(0.0, 0.0),
          hole_stretch=1.0, hole_ax=0.0, hole_plan=None):
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
    # ...and a flank is not a cylinder either, which is what `crown` alone makes it. A
    # cylinder is straight in one whole direction, so half of its surface directions have
    # zero curvature and the normal never leaves a single great circle. Measured against
    # the reference that is the largest single error left in this case: over the bodywork
    # the photograph runs from 92 to 226 grey levels, a spread of 134, and a cylindrical
    # panel gives 157 to 185 -- a spread of 28, one fifth as much, at any albedo, any
    # roughness and any environment. It is not the shading. There is nothing there to
    # shade.
    #
    # `crown_cross` is the sag about the PERPENDICULAR axis. A real rear quarter bends
    # both ways at once -- around the car, and along it into the wheel arch -- and a
    # doubly curved surface sweeps its normal across a patch instead of an arc, which is
    # what carries part of the panel through the grazing angles where a clearcoat starts
    # returning the sky.
    # `crown_at` moves the crown's APEX off the aperture, and without it neither crown can
    # produce what the reference shows. A parabola centred on the hole is symmetric: the
    # panel above the filler and the panel below it curve away by exactly the same amount,
    # so they take the same light and the body renders as one flat tone whatever the
    # curvature is. Sweeping `crown_cross` from 0 to 300 mm moved the body's tonal spread
    # from 72 to 73 -- the parameter was working, the shape was wrong.
    #
    # A car's filler is not in the middle of its flank. It sits low and aft, so the surface
    # above it rolls toward the roof and the surface below it toward the sill, and the
    # gradient that produces is the largest tonal feature on the bodywork: the photograph
    # falls about 100 grey levels from the top of the frame to the bottom, where a
    # symmetric panel falls 13.
    ca, sa = math.cos(crown_ax), math.sin(crown_ax)
    t0, u0 = crown_at

    def z_of(x, y, z0):
        if not crown and not crown_cross:
            return z0
        t = (x * ca + y * sa) / max(s, 1e-9) - t0
        u = (-x * sa + y * ca) / max(s, 1e-9) - u0
        return z0 - crown * t * t - crown_cross * u * u

    verts, faces = [], []
    for z in (0.0, -thick):
        for i in range(ring):
            a = TAU * i / ring
            c, sn = math.cos(a), math.sin(a)
            # `hole_plan` gives the aperture's radius per direction outright — that is how
            # the hole is made to be the same ROUNDED RECTANGLE as the pocket it surrounds
            # rather than merely the same nominal size. `squareness` is the older circular
            # form, kept for the parts that still use it.
            if hole_plan is not None:
                px, py = hole_plan[i % len(hole_plan)] * c, hole_plan[i % len(hole_plan)] * sn
            else:
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


def rrect_plan(w, h, n, steps, ref=None):
    """A `spin` plan for a ROUNDED RECTANGLE of given width, height and corner sharpness.

    `superellipse_plan` can only square a circle — it has no aspect ratio, so everything
    built with it comes out as wide as it is tall. A filler opening is not: it is 170 by
    150, and a fuel door more so. Returns the radius at each of `steps` directions divided
    by `ref`, so a profile authored in units of `ref` sweeps out this outline.

    `n` = 2 is an ellipse, 4-6 the rounded rectangles a body press actually makes."""
    ref = ref or max(w, h) / 2.0
    a, b = w / 2.0, h / 2.0
    out = []
    for j in range(steps):
        t = TAU * j / steps
        c, s = math.cos(t), math.sin(t)
        # the superellipse |x/a|^n + |y/b|^n = 1, solved along the ray
        k = ((abs(c) / a) ** n + (abs(s) / b) ** n) ** (-1.0 / n)
        out.append(k / ref)
    return out


def filler_box(open_w=0.170, open_h=0.152, depth=0.062, draft=0.86, sq=4.4,
               neck_r=0.048, neck_h=0.010, boss_off=(0.0, -0.012), lip=0.010,
               rim_step=0.004, drain=True, steps=72, material=None, mark="well",
               floor_material=None):
    """The filler pocket as what it is: a rectangular BOX pressed into the body.

    Every version of this part until now was a lathe — a round trench with a squareness plan
    that only began at 85 mm radius, which is outside the entire cavity. So the whole pocket,
    the part a photograph is mostly OF, was a smooth circular funnel, and no amount of work
    on the cap could fix the fact that the thing it sits in was the wrong shape.

    What the reference photographs show, on car after car: a rounded-rectangle opening about
    170 x 150 mm with a pressed lip and a seal bead round it; four near-vertical drafted
    walls; a flat floor 50-75 mm down; and on that floor a round raised boss carrying the
    filler neck, sitting OFF-CENTRE — usually low and toward the hinge. Not concentric with
    the opening, which is what a lathe forces it to be.

    Built with the same generalised-lathe machinery, so it is still three ops: the section is
    the box's own profile (floor, wall, lip) and the plan is the rounded rectangle. The boss
    is a separate small lathe placed at its real offset, which is the whole reason it can be
    off-centre at all."""
    material = material or WELL_PLASTIC
    ref = max(open_w, open_h) / 2.0
    # Section in units where r = ref is the opening. Axis outward, floor first.
    section = [
        (0.0, -depth),
        (ref * draft, -depth),                    # the floor, in by the wall's draft
        (ref * draft + (ref - ref * draft) * 0.25, -depth + lip),
        (ref, -rim_step),                         # up the drafted wall to the opening
        (ref + lip, 0.0),                         # the pressed lip the door seals on
        (ref + lip * 1.8, 0.0),
        (ref + lip * 1.8, -0.008),                # and back down the outside
        (ref + lip * 0.4, -0.010),
        (ref + lip * 0.4, -depth - 0.008),
        (0.0, -depth - 0.008),
    ]
    p = (MeshProgram().profile([(r, z) for r, z in section], plane="xz", closed=False)
         .spin(axis="z", steps=steps, plan=rrect_plan(open_w, open_h, sq, steps, ref),
               plan_from=0.0, mark=mark)
         .material({"by": "tag", "name": mark}, **material))
    # the boss the filler neck stands on — off-centre, which is the point
    if neck_r > 0:
        boss = lathe([(0.0, neck_h), (neck_r * 0.86, neck_h), (neck_r, neck_h * 0.35),
                      (neck_r, 0.0), (0.0, 0.0)], steps=44, mark=mark)
        boss = boss.material({"by": "tag", "name": mark},
                             **(floor_material or material))
        p = p.place(obj=boss, at=(boss_off[0], boss_off[1], -depth))
    if drain:
        # the drain, at the low corner of the floor rather than on the axis
        p = p.place(obj=lathe([(0.0, 0.0), (0.006, 0.0), (0.006, -0.010), (0.0, -0.010)],
                              steps=14, mark=mark),
                    at=(open_w * 0.26, -open_h * 0.30, -depth + 0.0005), material=GRIME)
    return p


def box_details(open_w, open_h, depth, draft=0.86, sq=4.4, keep_out=0.045, screws=2,
                catches=1, grommets=1, seed=0, material=None):
    """The hardware down a box pocket — on the WALLS, which is where a box has walls.

    `well_details` scatters its parts on a ring, because the pocket it was written for was a
    solid of revolution. A box has four flat sides and a floor, and everything real in these
    photographs is bolted to one of them: the striker on the wall nearest the latch, screws
    through the liner, a grommet where the breather leaves, the tether's anchor.

    Empty, a box reads as a render. It is also the only structure inside the ROI, so it is
    what `fit.complexity` is measuring when it asks how much relief there is between 6 and
    24 mm."""
    import random
    rr = random.Random(seed)
    material = material or WELL_PLASTIC
    ref = max(open_w, open_h) / 2.0
    p = MeshProgram()

    def on_wall(frac_depth, jitter=0.10):
        """A point on the box's wall at a given depth, and the outward normal's azimuth."""
        a = rr.uniform(0, TAU)
        k = rrect_plan(open_w, open_h, sq, 96, ref)[int(a / TAU * 96) % 96]
        t = frac_depth
        r = ref * k * (1.0 - (1.0 - draft) * t) * (1.0 - jitter * rr.random())
        return r * math.cos(a), r * math.sin(a), -depth * t, math.degrees(a)

    for _ in range(screws):
        x, y, z, a = on_wall(rr.uniform(0.30, 0.85))
        p = p.place(obj=screw(r=rr.uniform(0.0028, 0.0040)), at=(x, y, z),
                    rotate=(0.0, 0.0, a))
    for _ in range(catches):
        x, y, z, a = on_wall(rr.uniform(0.16, 0.40), jitter=0.02)
        p = p.place(obj=catch(w=rr.uniform(0.012, 0.019), h=rr.uniform(0.016, 0.026)),
                    at=(x, y, z), rotate=(90.0, 0.0, a + 90.0))
    for _ in range(grommets):
        x, y, z, a = on_wall(rr.uniform(0.70, 0.95))
        p = p.place(obj=grommet(r=rr.uniform(0.0045, 0.0070)), at=(x, y, z),
                    rotate=(0.0, 0.0, a))
    return p


def opening_seal(open_w=0.170, open_h=0.152, sq=4.4, bead=0.0045, lip=0.010, steps=72,
                 material=None):
    """The rubber weatherstrip round the opening — the black band in every reference frame.

    Small, and it does something out of proportion to its size: it is the only matt black
    line separating body paint from pocket, so it draws the opening's outline in a scene
    where paint and shadow otherwise meet with no edge at all."""
    ref = max(open_w, open_h) / 2.0 + lip * 0.6
    sec = [(ref - bead, 0.0), (ref, bead * 0.75), (ref + bead * 0.6, 0.0),
           (ref, -bead * 0.4)]
    return (MeshProgram().profile(sec, plane="xz", closed=True)
            .spin(axis="z", steps=steps,
                  plan=rrect_plan(open_w + 2 * lip * 0.6, open_h + 2 * lip * 0.6, sq, steps, ref),
                  plan_from=0.0, mark="well")
            .material({"by": "tag", "name": "well"}, **(material or SEAL_RUBBER)))


def pressed_dish(cap_r=0.039, gap=0.006, sink=0.015, throat=0.030, wall_z=0.030, out_r=0.105,
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
    # `sink` is the whole point and the first version did not have it: the dish's FLOOR is
    # a good centimetre BELOW the cap's face, because the cap stands up out of its hole with
    # its whole fluted wall showing. Built flush, the floor is coplanar with the face and
    # simply covers the subject — three of eight frames came back as a flat sheet of body
    # colour with a handle lying on it and no cap at all.
    z0 = -abs(sink)
    sec = [
        (rh, z0 - throat),                             # the bottom of the drawn throat
        (rh * 1.02, z0 - throat * 0.55),
        (rh * 1.06, z0 - 0.002),                       # up to the dish floor, near vertical
        (rh * 1.35, z0),
        (rh * 1.35 + (out_r - rh * 1.35) * 0.55, z0 + (wall_z - z0) * 0.62),
        (out_r - rim, wall_z),                         # out and up the dish wall
        (out_r, wall_z + rim * 0.25),                  # the lip, level with the paint
    ]
    prof = section_to_profile(sec, floor=z0 - throat - 0.030, outer=out_r * 1.02)
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
    # A CLOSED section spun 360 degrees — an annulus with a hole, not a disc.
    #
    # `lathe` insists its section start and end ON the axis, because that is what closes an
    # open polyline into a solid. This part is the one shape in the kit that must NOT: it
    # has a hole in the middle. Written as a lathe it began (0, 0) -> (r_out, 0), which
    # sweeps a solid plate from the axis outwards, five millimetres ABOVE the cap face —
    # so on every frame that drew a trim ring the whole pocket was capped by a chrome plate
    # with the cap invisible underneath and only the handle poking through. Three of eight
    # closeup frames, and it had been in the kit unnoticed since it was written, because at
    # the sixty-pixel scale everything was judged at, a plate over the pocket and a pocket
    # are the same grey blob.
    p = (MeshProgram()
         .profile([(r_in, 0.0), (r_out, 0.0), (r_out, -thick), (r_in, -thick * 0.55)],
                  plane="xz", closed=True)
         .spin(axis="z", steps=steps, angle=360.0, mark="panel")
         .material({"by": "tag", "name": "panel"}, **material))
    rm = (r_in + r_out) / 2
    for i in range(screws):
        a = TAU * (i + 0.5) / screws
        p = p.place(obj=screw(r=0.0026, head_h=0.0012),
                    at=(rm * math.cos(a), rm * math.sin(a), 0.0005))
    return p


def fuel_door(w=0.186, h=0.166, sq=4.2, flange=0.014, face=0.007, rim=0.013,
              open_deg=150.0, az=180.0, hinge_r=0.098, gap=0.004, steps=64,
              skin=None, liner=None, strap=True, arm_w=0.020, arm_t=0.0025,
              latch=True, plan=None, inside_material=None, inner_details=False,
              inner_parts=None, mark="door"):
    """The fuel door: a shallow ROUNDED-RECTANGLE pressing, hinged at one side.

    The old part was a flat slab whose plan was a plain rectangle, hinged below the pocket
    and standing out at ninety degrees like a shelf. Against the reference photographs that
    is wrong in every one of its four decisions.

    * **Shape.** It is the opening's own outline, a little larger — a rounded rectangle with
      a 25 mm corner radius, not a rectangle and not a square. Same `spin` + plan machinery
      as the pocket, so the two match by construction rather than by two sets of numbers.
    * **Section.** It is a pressing: an outer skin in body colour, an edge that turns down,
      a return flange all round, and a liner panel set a few millimetres inside it. Seen
      from the pocket — which is where the camera is — that flange is the shape you read.
    * **Hinge.** At a SIDE, on a cranked steel strap, not along the bottom.
    * **Angle.** Real ones are usually swung right back, 140 to 175 degrees, so the door
      lies nearly flat against the wing and presents its edge. At 95 it fills the frame.

    Returns the door already swung and placed in the POCKET's frame, so the caller only has
    to put the pocket where the pocket goes. `az` is which side the hinge is on, measured
    round the panel normal."""
    skin = skin or {"color": [0.3, 0.3, 0.3], "metallic": 0.6, "roughness": 0.15}
    liner = liner or WELL_PLASTIC
    ref = max(w, h) / 2.0
    section = [
        (0.0, 0.0), (ref * 0.965, 0.0),            # the outer skin
        (ref, -0.0028),                            # the edge radius
        (ref, -flange),                            # the return flange, turning into the car
        (ref - rim, -flange),
        (ref - rim, -face),                        # the liner face, set inside the flange
        (0.0, -face),
    ]
    # `plan` lets a caller hand in a MEASURED outline instead of a described one. A door is
    # the opening's own pressing a few millimetres larger, so when the opening came off a
    # photograph the door has to come off the same photograph or the two stop matching at
    # exactly the place — the shut line — where a mismatch is most visible.
    d = (MeshProgram().profile(section, plane="xz", closed=False)
         .spin(axis="z", steps=steps,
               plan=plan if plan is not None else rrect_plan(w, h, sq, steps, ref),
               plan_from=0.0, mark=mark))
    # paint every face first, THEN the two big ones — selecting only +z and -z leaves the
    # turned edge on the renderer's default albedo, a bright rim right round a dark door
    # The painted outer skin wraps around the turned edge. Painting that edge as liner
    # removes the white return flange visible around an open white door and leaves a black
    # slab. The inset -z face and its return are the separate black moulding.
    d = d.material({"by": "all"}, **skin)
    d = d.material({"by": "normal", "axis": "z", "sign": 1}, **skin)
    d = d.material({"by": "normal", "axis": "z", "sign": -1},
                   **(inside_material or liner))
    if inner_details:
        # Shallow bosses moulded into the inner liner. They catch broad highlights and make
        # the door read as a stamped automotive assembly instead of a featureless plate.
        z0 = -face - 0.0002
        # Never reuse a decal material on separately placed details: its map frame belongs
        # to each little boss, so the full door label gets repeated and enlarged on them.
        detail_mat = liner
        for cx, cy, sx, sy, dz in (
                (-ref * 0.20,  ref * 0.22, ref * 0.32, ref * 0.13, 0.0017),
                ( ref * 0.20, -ref * 0.24, ref * 0.18, ref * 0.10, 0.0012),
                ( ref * 0.28,  ref * 0.25, ref * 0.08, ref * 0.20, 0.0015)):
            d = d.place(obj=prism([(-sx, -sy), (sx, -sy), (sx, sy), (-sx, sy)],
                                  z0, z0 - dz, mark=mark),
                        at=(cx, cy, 0.0), material=detail_mat)
    if inner_parts is not None:
        # Geometry mounted on the inner face, in the door's OWN frame, before any of the
        # swing is applied. A caller cannot do this from outside: `fuel_door` composes a
        # translation to the hinge line, a rotation by `open_deg` and a further rotation by
        # `az`, and reproducing that by hand puts the parts somewhere else entirely — a set
        # of stamped ribs meant for the door's face ended up as a star inside the pocket.
        d = d.place(obj=inner_parts, at=(0.0, 0.0, -face))
    if latch:
        # the striker on the free edge, opposite the hinge
        d = d.place(obj=prism([(-0.008, -0.005), (0.008, -0.005), (0.008, 0.005),
                               (-0.008, 0.005)], -flange, -flange - 0.006, mark=mark),
                    at=(-(ref - rim * 1.4), 0.0, 0.0), material=CATCH_STEEL)

    # hinge at the door's own -x edge: shift so that edge sits on the origin, then swing
    edge = ref + gap
    p = MeshProgram().place(obj=d, at=(edge, 0.0, 0.0), rotate=(0.0, 0.0, 0.0))
    if strap:
        # The cranked strap, on the door and swinging with it. Its far end reaches back to
        # the body when the door is open, which is the pose it is in for every frame this
        # case renders — a hinge modelled as two rigid halves would need the second half to
        # move, and at 150 degrees this is the half you can see.
        p = p.place(obj=hinge_arm(length=0.052, w=arm_w, t=arm_t, holes=2),
                    at=(0.004, -h * 0.22, -flange - arm_t), rotate=(0.0, 0.0, 0.0))
    p = MeshProgram().place(obj=p, at=(0.0, 0.0, 0.0), rotate=(0.0, -open_deg, 0.0))
    # out to the hinge line, then round to whichever side the hinge is on
    p = MeshProgram().place(obj=p, at=(-hinge_r, 0.0, 0.0))
    # `az` names the side the HINGE is on, which is the way anybody describes a fuel door.
    # The construction above swings the door back over -x, so the extra half turn is what
    # makes the parameter mean what it says instead of its opposite.
    return MeshProgram().place(obj=p, at=(0.0, 0.0, 0.0), rotate=(0.0, 0.0, az + 180.0))


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
    until it is missing.

    `spin` is applied HERE, to the lug's own plan, and not by the caller's `place`. Same
    reason as for the cap: `place` composes Rz @ Ry @ Rx, so a z-rotation handed to it lands
    OUTSIDE the assembly's tilt and becomes a rotation about a tilted axis. On a 12 mm lug
    at 15 degrees of tilt that lifted a corner clear of the cap's own face — which is how it
    was found, by the invariant that says nothing may sit above that face."""
    a = math.radians(spin)
    c, s = math.cos(a), math.sin(a)
    plan = [(-0.006, -0.004), (0.006, -0.004), (0.005, 0.004), (-0.005, 0.004)]
    return (prism([(x * c - y * s, x * s + y * c) for x, y in plan], 0.0, 0.005,
                  mark="cap_body")
            .material({"by": "all"}, **(material or CAP_BLACK)))


# --------------------------------------------------------------------------- #
# parts that only appeared once ONE pocket was measured instead of described
# --------------------------------------------------------------------------- #
def measured_plan(table, steps, phase=0.0):
    """A `spin` plan resampled from a MEASURED radius-vs-angle table.

    `rrect_plan` describes an opening with three numbers, which is the right thing to do
    when the opening is being *drawn*. It is the wrong thing when the opening has been
    *photographed*. A body press has unequal corner radii, one edge straighter than the
    other, and a local flat where a bracket passes; a superellipse fitted to that outline
    leaves 5 mm of residual with the sign of the error changing from corner to corner —
    a third of the flange's width, on the one line in the picture that separates paint
    from shadow. So take the outline itself.

    `table` is radius divided by the profile's own reference radius, evenly spaced over a
    full turn starting at +x. Resampled with a PERIODIC Catmull-Rom, because a 36-entry
    measurement driven through a 96-segment lathe by linear interpolation is a 36-sided
    polygon with rounded corners: the radius is continuous and its slope is not, and a
    smooth-shaded lathe shows exactly that as 36 faint vertical bands.
    """
    n = len(table)
    out = []
    for j in range(steps):
        t = ((TAU * j / steps + phase) % TAU) / TAU * n
        i = int(math.floor(t))
        f = t - i
        p0, p1, p2, p3 = (table[(i - 1) % n], table[i % n],
                          table[(i + 1) % n], table[(i + 2) % n])
        out.append(0.5 * (2 * p1 + (-p0 + p2) * f + (2 * p0 - 5 * p1 + 4 * p2 - p3) * f * f
                          + (-p0 + 3 * p1 - 3 * p2 + p3) * f * f * f))
    return out


def coil_cord(start, end, coils=6.0, coil_r=0.011, wire_r=0.0022, taper=0.22,
              per_coil=18, sides=8, up=(0.0, 0.0, 1.0), material=None, mark="tether"):
    """The tether as the HELIX it usually is, not as a hanging cord.

    `tether` builds a cord that droops, and that shape is right for maybe a third of the
    reference cars. The rest — and most of the ones whose cord is clearly visible — have a
    coiled lead exactly like a telephone handset's: six or seven tight turns of about 20 mm
    diameter between two straight ends. Seen from the side that reads as a run of even
    loops, and it is one of the very few bright, high-frequency things in a pocket that is
    otherwise a black box, so it carries far more of the picture than its 2 mm section
    suggests.

    The coil radius is faded in and out by `taper` rather than butted onto straight leads.
    That is not cosmetic: a straight segment meeting a helix at full radius turns through
    ninety degrees in a single path step, and `sweep` collapses the ring at a corner that
    sharp. Fading the radius to zero gives the straight ends for free and keeps the path
    smooth.
    """
    s = [float(x) for x in start]
    e = [float(x) for x in end]
    v = [e[k] - s[k] for k in range(3)]
    L = math.sqrt(sum(x * x for x in v))
    if L < 1e-6:
        raise ValueError("coil_cord needs two distinct endpoints")
    ax = [x / L for x in v]
    u0 = [float(x) for x in up]
    d = sum(u0[k] * ax[k] for k in range(3))
    b1 = [u0[k] - d * ax[k] for k in range(3)]
    if sum(x * x for x in b1) < 1e-8:                 # the cord runs along `up`
        u0 = [1.0, 0.0, 0.0]
        d = ax[0]
        b1 = [u0[k] - d * ax[k] for k in range(3)]
    n1 = math.sqrt(sum(x * x for x in b1))
    b1 = [x / n1 for x in b1]
    b2 = [ax[1] * b1[2] - ax[2] * b1[1], ax[2] * b1[0] - ax[0] * b1[2],
          ax[0] * b1[1] - ax[1] * b1[0]]

    n = max(28, int(per_coil * coils) + 2)
    path = []
    for k in range(n):
        t = k / (n - 1.0)
        g = min(t, 1.0 - t) / max(taper, 1e-6)
        env = 0.5 - 0.5 * math.cos(math.pi * min(1.0, max(0.0, g)))
        a = TAU * coils * t
        r = coil_r * env
        path.append([s[j] + ax[j] * (L * t) + r * (math.cos(a) * b1[j] + math.sin(a) * b2[j])
                     for j in range(3)])
    ring = [(wire_r * math.cos(TAU * k / sides), wire_r * math.sin(TAU * k / sides))
            for k in range(sides)]
    return (MeshProgram().profile(ring, plane="xy", closed=True).sweep(path, mark=mark)
            .material({"by": "tag", "name": mark}, **(material or TETHER)))


def pip(r=0.0028, h=0.0013, steps=12, material=None, mark="well"):
    """One of the little round moulding pips on a liner's flange.

    Six or eight of them go round the rim of an injection-moulded pocket — ejector-pin
    witnesses and locating bosses. Individually they are a millimetre high and pointless;
    together they are the only thing that says the flange is a moulding rather than a flat
    black band drawn round the opening, which is what it renders as without them.
    """
    return lathe([(0.0, h), (r * 0.55, h), (r, 0.0), (0.0, 0.0)], steps=steps,
                 mark=mark).material({"by": "tag", "name": mark},
                                     **(material or WELL_PLASTIC))


def bump_stop(r=0.0065, h=0.011, material=None, mark="well"):
    """The rubber stop the closed door rests on, standing off the pocket wall.

    Built along +z and placed by the caller, like everything else here. Stepped rather than
    plain: the real ones are a soft cap on a hard stem, and the step is what catches a
    highlight and stops it reading as a smudge on the wall.
    """
    return lathe([(0.0, h), (r * 0.62, h), (r * 0.80, h * 0.86), (r * 0.80, h * 0.46),
                  (r * 0.52, h * 0.38), (r * 0.52, 0.0), (0.0, 0.0)],
                 steps=20, mark=mark).material({"by": "tag", "name": mark},
                                               **(material or SEAL_RUBBER))


def neck_stack(cap_r, floor_z, top_z, flare=1.55, material=None, mark="neck"):
    """The filler neck standing off the pocket floor, carrying the cap's landing.

    The reason this part has to exist: measured against the body panel, the cap's face is
    only about 10 mm down, while the pocket floor around it is 45. A model that stands the
    cap on the floor therefore has to make the pocket 10 mm deep to keep the cap where the
    measurement puts it — and a 10 mm pocket is a saucer. The cap is near the top of a deep
    box because it is standing on this.
    """
    r = cap_r
    h = top_z - floor_z
    return lathe([(0.0, top_z), (r * 1.02, top_z), (r * 1.10, top_z - h * 0.16),
                  (r * flare * 0.86, floor_z + h * 0.22), (r * flare, floor_z),
                  (0.0, floor_z)], steps=44,
                 mark=mark).material({"by": "tag", "name": mark},
                                     **(material or WELL_PLASTIC))


def liner(plan, ref, depth=0.045, flange_w=0.011, flange_z=0.002, fold=0.005,
          wall_k=0.745, ledge=0.008, floor_k=0.66, back=0.010, steps=96,
          material=None, mark="well"):
    """The moulded pocket liner, as the five distinct surfaces a photograph of one shows.

    `filler_box` has three: a floor, one drafted wall, and a lip. Reading inwards from the
    paint, the reference actually has

        1. a flat FLANGE about 11 mm wide, parallel to the body and just below it,
        2. a crisp FOLD where that flange turns down — the brightest line in the pocket,
        3. a steep drafted WALL, 40-odd mm of it,
        4. a LEDGE part way down where the moulding steps in,
        5. the FLOOR.

    Between them those five make four tone steps, and the tone steps are what read as depth.
    One drafted wall makes a single smooth gradient, which is why a box pocket renders as a
    dark trapezoid however the light is set: there is nothing in it for the light to break
    on.

    `plan` is a per-direction radius multiplier — pass `measured_plan(...)` to sweep a
    photographed outline or `rrect_plan(...)` for a described one. `ref` is the radius the
    section is authored against, so `ref` is the flange's OUTER edge, and the body panel's
    hole has to be cut to the same plan at exactly that radius.
    """
    material = material or WELL_PLASTIC
    fi = (ref - flange_w) / ref                      # flange inner edge, as a fraction
    section = [
        (0.0, -depth),
        (ref * floor_k, -depth),                              # 5. the floor
        (ref * (floor_k + 0.045), -depth + ledge * 0.75),     # 4. up onto the ledge
        (ref * wall_k, -depth + ledge),                       #    the ledge itself
        (ref * (fi - 0.020), -flange_z - fold * 1.6),         # 3. the drafted wall
        (ref * (fi - 0.004), -flange_z - fold * 0.45),        # 2. the fold
        (ref * fi, -flange_z - fold * 0.10),
        (ref * 0.996, -flange_z),                             # 1. the flange
        (ref, -flange_z + 0.0007),                            #    tipping up to meet the paint
        (ref, -flange_z - 0.007),                             # over the edge, down the back
        (ref * (fi + 0.02), -flange_z - 0.009),
        (ref * (fi + 0.02), -depth - back),
        (0.0, -depth - back),
    ]
    return (MeshProgram().profile([(r, z) for r, z in section], plane="xz", closed=False)
            .spin(axis="z", steps=steps, plan=plan, plan_from=0.0, mark=mark)
            .material({"by": "tag", "name": mark}, **material))


def door_strap(length=0.078, w=0.021, t=0.0032, bend_at=0.42, bend_deg=34.0,
               stations=18, material=None, mark="door"):
    """The stamped arm between the door and its hinge — a flat bar with one bend in it.

    Visible in most of the reference frames and in every one where the door is open, because
    it crosses the opening: whatever else the pocket contains, a wide matt-black bar cuts
    over the top of it. Modelled as a swept section rather than a plate so the bend is a real
    bend; a straight plate rotated into place leaves the strap either buried in the pocket
    wall or floating clear of the door.
    """
    hw, ht = w / 2.0, t / 2.0
    k = min(hw, ht) * 0.9
    prof = [(-hw + k, -ht), (hw - k, -ht), (hw, -ht + k), (hw, ht - k),
            (hw - k, ht), (-hw + k, ht), (-hw, ht - k), (-hw, -ht + k)]
    a = math.radians(bend_deg)
    path = []
    for j in range(stations):
        u = j / (stations - 1.0)
        if u <= bend_at:
            path.append([length * u, 0.0, 0.0])
        else:
            d = length * (u - bend_at)
            path.append([length * bend_at + d * math.cos(a), 0.0, -d * math.sin(a)])
    return (MeshProgram().profile(prof, plane="xy", closed=True).sweep(path, mark=mark)
            .material({"by": "tag", "name": mark},
                      **(material or mat((0.018, 0.018, 0.020), 0.0, 0.62))))


def hinge_bracket(w=0.056, h=0.042, t=0.0028, lip=0.010, lip_deg=68.0, bolts=2,
                  bolt_r=0.0042, material=None, bolt_material=None, mark="well"):
    """The stamped plate the door's hinge is bolted to, standing on the pocket wall.

    The reason this exists: an id map of the finished pocket showed one continuous `well`
    surface covering sixty per cent of the aperture, with nothing on it. A photographed
    filler pocket is not a moulded shell — it is a shell with an assembly bolted into one
    corner of it, and that assembly is the largest thing inside the opening after the cap.
    Modelled as a shell alone the pocket renders as a smooth dark bowl however carefully the
    shell's own section is measured, because a bowl is what it is.

    A plate, a folded lip along its far edge, and the bolt heads. Built lying in its own
    xy plane with +z its outward normal, so the caller places it against a wall with the
    same `rotate` it would use for any other wall furniture.
    """
    material = material or mat((0.021, 0.021, 0.023), 0.0, 0.58)
    hw, hh = w / 2.0, h / 2.0
    r = min(hw, hh) * 0.22
    plate = [(-hw + r, -hh), (hw - r, -hh), (hw, -hh + r), (hw, hh - r),
             (hw - r, hh), (-hw + r, hh), (-hw, hh - r), (-hw, -hh + r)]
    p = prism(plate, 0.0, t, mark=mark).material({"by": "all"}, **material)

    # The folded lip. A stamping is stiffened by a return along its free edge, and that
    # return is what catches the one hard highlight the bracket shows in the photograph —
    # a flat plate has nowhere for that line to come from.
    if lip > 0.0:
        # A prism turned about x, not a sweep. Sweeping a section authored in xy along a
        # path in xz and then rotating the result puts the transported frame somewhere
        # nobody predicted — the first version reached 23 mm through the wall behind it.
        # A lip is a flat tab; build it flat.
        a = math.radians(lip_deg)
        tab = [(-hw * 0.86, 0.0), (hw * 0.86, 0.0), (hw * 0.86, lip), (-hw * 0.86, lip)]
        p = p.place(obj=prism(tab, 0.0, t, mark=mark),
                    at=(0.0, hh - t * 0.5, t * 0.5), rotate=(-(90.0 - lip_deg + 90.0), 0.0, 0.0),
                    material=material)
    for k in range(bolts):
        x = (k + 0.5) / max(bolts, 1) * w - hw
        p = p.place(obj=lathe([(0.0, bolt_r * 0.62), (bolt_r * 0.70, bolt_r * 0.62),
                               (bolt_r, bolt_r * 0.18), (bolt_r, 0.0), (0.0, 0.0)],
                              steps=16, mark=mark),
                    at=(x, -hh * 0.42, t),
                    # NOT SCREW_STEEL. A catalogue steel head is the brightest thing in
                    # a frame whose subject is black plastic in shadow, and this kit has
                    # already been round that loop once with the liner's screws. These
                    # are dark fasteners in a pocket that fills with road dirt.
                    material=bolt_material or mat((0.055, 0.055, 0.058), 0.55, 0.62))
    return p


def emboss(prog, centre, half, depth, inset=0.35, mark=None):
    """Raise or sink a patch of an EXISTING surface, in place.

    This is the operator a moulding actually needs and the one this case kept not using.
    A loft can only be given features by rewriting the loft; `inset` + `extrude` puts a boss
    or a recess on a surface that is already there, wherever a selector can reach — which is
    how every boss, pad, drain shelf and screw land in a real pocket is made.

    `centre`/`half` are a world-space box that picks the faces; `depth` is positive out of
    the surface and negative into it. `inset` is how much of the patch becomes the drafted
    flank rather than the flat top — 0 gives a straight-sided step, which no tool can make.
    """
    lo = [centre[k] - half[k] for k in range(3)]
    hi = [centre[k] + half[k] for k in range(3)]
    # region=True. The per-face default insets every quad in the patch separately, so
    # a pad on a few hundred lofted quads becomes a few hundred islands and renders as
    # a field of stipple. That is what the first pass of this did.
    p = prog.inset({"by": "box", "min": lo, "max": hi}, thickness=inset, region=True)
    p = p.extrude({"by": "last_created"}, distance=depth)
    return p.tag({"by": "last_created"}, mark) if mark else p


def stamped_strap(path, width, thick=0.0026, flange=0.0, beads=(), bead_r=0.0030,
                  bead_h=0.0022, bead_span=(0.0, 0.42),
                  bosses=(), boss_r=0.006, boss_h=0.0011, steps=14,
                  material=None, mark="door"):
    """A wide sheet-metal band that cranks along a path, with turned edges and beads.

    The part this replaces was three thin bars floating in the air beside the pocket.  In
    the reference the thing between the pocket and the open door is not a bar at all: it is
    a **band**, about as tall as the aperture's radius and only a little longer than it is
    tall, with its long edges turned toward the camera, three half-round stiffening beads
    pressed across it, and a couple of raised moulding marks on its face.  Measured off the
    photograph its top edge runs from 126 mm to 185 mm out from the aperture centre while
    dropping 13 mm -- a band, tilted, not a stick.

    Geometry.  `path` is the crank, given as (x, z) stations in the plane perpendicular to
    the band's width; the band is that path swept along y.  Doing it this way means the
    crank is a real crank -- a flat plate rotated into position cannot both meet the pocket
    rim and meet the door, which is why the earlier arms either buried themselves in the
    wall or floated clear of the door.

    `flange` turns both long edges up out of the band's face by that height, following the
    same path.  That lip is what puts the bright line along the top and bottom edges; with
    the band alone the part reads as a strip of paper.

    `beads` are positions ACROSS the width, as fractions of it, where a rib runs ALONG the
    band; `bead_span` bounds how much of the length they cover.  `bosses` are
    (fraction-along, y) positions for the small raised moulding marks.
    """
    material = material or {"color": [0.16, 0.40, 0.56], "metallic": 0.32,
                            "roughness": 0.14}
    pts = [(float(x), float(z)) for x, z in path]
    if len(pts) < 2:
        raise ValueError("stamped_strap needs at least two path stations")

    def normals():
        # Outward normal of the band at every station, from the neighbouring segments.
        out = []
        for i in range(len(pts)):
            a = pts[max(0, i - 1)]
            b = pts[min(len(pts) - 1, i + 1)]
            dx, dz = b[0] - a[0], b[1] - a[1]
            L = math.hypot(dx, dz) or 1.0
            out.append((-dz / L, dx / L))
        return out

    nrm = normals()
    front = [(x + nx * thick, z + nz * thick) for (x, z), (nx, nz) in zip(pts, nrm)]
    # Everything below is assembled in a LOCAL frame, then turned into the panel's frame
    # in one step at the end.  `prism` lays its polygon in xy and extrudes along z, so the
    # local axes are (x, out-of-panel, width) and the final Rx(90) sends local y to the
    # panel's +z and local z to its -y -- hence the negated boss coordinate, so a caller
    # still gives boss positions in the panel's own +y.  Building in the panel frame and
    # hoping
    # each sub-part lands right is how the bead cylinders ended up running out of the
    # panel instead of across the band.
    poly = [(x, z) for x, z in pts] + [(x, z) for x, z in reversed(front)]
    p = MeshProgram().place(obj=prism(poly, -width / 2.0, width / 2.0, mark=mark))
    if flange > 0.0:
        # A lip standing off the band's face along both long edges. Same path, so the lip
        # follows the crank instead of cutting across it.
        lip = [(x + nx * flange, z + nz * flange) for (x, z), (nx, nz) in zip(front, nrm)]
        wall = [(x, z) for x, z in front] + [(x, z) for x, z in reversed(lip)]
        for s in (-1.0, 1.0):
            e = s * width / 2.0
            p = p.place(obj=prism(wall, e - s * thick, e, mark=mark))
    # Beads run ALONG the band, spaced across its width -- which is the way round the
    # reference has them, and the opposite of the first attempt. Getting this backwards is
    # not a detail: ribs across a band read as a corrugation or a hinge knuckle, ribs along
    # it read as what they are, a stiffener stopping the band folding about its long axis.
    # In the photograph they occupy only the first 40 per cent of the length, dying out
    # before the band reaches the door, so `bead_span` bounds them.
    if beads:
        i0 = max(0, int(round(bead_span[0] * (len(pts) - 1))))
        i1 = min(len(pts) - 1, int(round(bead_span[1] * (len(pts) - 1))))
        if i1 <= i0:
            i1 = min(len(pts) - 1, i0 + 1)
        # HALF-ROUND, not a flat-topped slab. Rendered alone at part scale the prism
        # version read as four rectangular plates laid on the band; a pressed bead is a
        # tube half sunk in the sheet, and it is the round crest that carries the bright
        # line down each one in the reference. A straight cylinder is a fair stand-in over
        # a rib this short even though the path under it curves.
        (ax, az), (bx, bz) = front[i0], front[i1]
        ang = math.degrees(math.atan2(bz - az, bx - ax))
        run_len = math.hypot(bx - ax, bz - az)
        # Run the ribs a bead's width PAST the band's free end so they break its
        # silhouette. That scalloped edge is not decoration -- it is the feature that gives
        # up the count and the pitch when the band's face is too foreshortened to read, and
        # it is how the four were counted in the first place.
        # A bead's own height of overhang was far too much: a rib standing 4 mm proud and
        # poking 6 mm past the edge projects as a pipe on the end of the band, not as a
        # pressing. In the reference the break is barely two millimetres.
        ux, uz = (bx - ax) / (run_len or 1.0), (bz - az) / (run_len or 1.0)
        ax, az = ax - ux * bead_h * 0.45, az - uz * bead_h * 0.45
        run_len += bead_h * 0.45
        mx, mz = 0.5 * (ax + bx), 0.5 * (az + bz)
        nx, nz = nrm[(i0 + i1) // 2]
        for f in beads:
            rib = (MeshProgram().cylinder(radius=bead_h, height=run_len, sides=steps,
                                          mark=mark)
                   .material({"by": "tag", "name": mark}, **material))
            p = p.place(obj=rib,
                        at=(mx - nx * bead_h * 0.60, mz - nz * bead_h * 0.60,
                            float(f) * width),
                        rotate=(0.0, 90.0, ang))
    for f, y in bosses:
        i = min(len(pts) - 1, max(0, int(round(f * (len(pts) - 1)))))
        (x, z), (nx, nz) = front[i], nrm[i]
        # `pip` stands on its own +z; Ry(90) lays that along local +x and Rz then swings
        # it round to the band's normal, which is the composition `place` applies.
        p = p.place(obj=pip(r=boss_r, h=boss_h, steps=10, material=material, mark=mark),
                    at=(x + nx * boss_h, z + nz * boss_h, -float(y)),
                    rotate=(0.0, 90.0, math.degrees(math.atan2(nz, nx))))
    p = MeshProgram().place(obj=p, at=(0.0, 0.0, 0.0), rotate=(90.0, 0.0, 0.0))
    return p.material({"by": "tag", "name": mark}, **material)


def slotted_bracket(w=0.066, h=0.060, t=0.0026, window=(0.30, 0.18, 0.52, 0.60),
                    hook=0.014, hook_t=0.0030, material=None, mark="door"):
    """A stamped plate with a rectangular window cut through it and a hook along one edge.

    Below the strap the reference has a second, separate pressing: a plate roughly 66 mm
    wide and 60 mm tall with a big rectangular hole through it and a tab turned off its
    lower edge.  It reads as a hole because it is a hole -- the body behind shows through
    it -- so it is built as four rails round the opening rather than as a plate with a dark
    rectangle painted on, which is what a texture would have given.

    `window` is (x0, y0, x1, y1) as fractions of the plate, measured from its lower left.
    """
    material = material or {"color": [0.16, 0.40, 0.56], "metallic": 0.32,
                            "roughness": 0.14}
    x0, y0, x1, y1 = window
    ax, bx = -w / 2 + x0 * w, -w / 2 + x1 * w
    ay, by = -h / 2 + y0 * h, -h / 2 + y1 * h
    rails = [
        [(-w / 2, -h / 2), (w / 2, -h / 2), (w / 2, ay), (-w / 2, ay)],       # below
        [(-w / 2, by), (w / 2, by), (w / 2, h / 2), (-w / 2, h / 2)],         # above
        [(-w / 2, ay), (ax, ay), (ax, by), (-w / 2, by)],                     # left
        [(bx, ay), (w / 2, ay), (w / 2, by), (bx, by)],                       # right
    ]
    p = MeshProgram()
    for r in rails:
        p = p.place(obj=prism(r, 0.0, t, mark=mark))
    if hook > 0.0:
        # The tab turned off the bottom edge, standing proud toward the camera.
        p = p.place(obj=prism([(-w * 0.22, 0.0), (w * 0.22, 0.0),
                               (w * 0.22, hook_t), (-w * 0.22, hook_t)],
                              0.0, hook, mark=mark),
                    at=(0.0, -h / 2, 0.0))
    return p.material({"by": "tag", "name": mark}, **material)


def loft_sections(sections, angles, steps=96, plan=None, ref=1.0, mark=None,
                  centre=None):
    """Loft a closed shell from SECTIONS THAT DIFFER, interpolated around the axis.

    Why this exists
    ---------------
    Everything else in this kit that makes a hollow -- `spin`, `spin(plan=)`, the
    hand-written ring loft in case 28 -- builds one section and varies its RADIUS with
    angle. That family is closed under "bowl". At every angle the radius falls and the
    depth falls, both monotonically, so the contour never reverses and no part of the
    surface can ever face down and inward. A moulded filler pocket is not in that family:
    over the top it is a canopy whose underside faces the floor, and at the bottom it is a
    flat shelf. Case 28 spent four rounds adding analytic correction terms to a circle,
    and its own comment said "scoop" while its arithmetic said "bowl".

    The failure has a signature worth naming, because it is not confined to this part:
    when the generator cannot make the form, the modeller decorates the wrong form. The
    canopy in the reference is one continuous volume; the model grew a separate embossed
    crescent rib in roughly the right place, which reads as an eyebrow stuck on a bowl.

    So: no formula. Hand in the sections. An overhang is a section whose radius INCREASES
    with depth -- nothing more -- and the comment and the geometry cannot disagree because
    there is no longer anything to disagree with.

    Parameters
    ----------
    sections : list of polylines, each [(r, z), ...] with the SAME number of stations, so
        station j of every section becomes ring j. `r` is a fraction of the local radius.
    angles : the angle in radians each section is measured at, ascending; wrapped as a
        closed periodic set so the shell has no seam.
    plan : optional per-angle outer radius, exactly as the rest of the kit uses it, so a
        measured aperture outline still drives the mouth.
    centre : optional callable (u) -> (cx, cy) offsetting each ring, for a throat that
        does not sit under the middle of the opening.

    Interpolation between sections is cosine in angle -- C1 at the sections themselves,
    which matters because a linear blend leaves a visible crease along every section line
    and those creases catch light exactly where a moulding has none.
    """
    n = len(sections)
    if n < 2:
        raise ValueError("loft_sections needs at least two sections")
    m = len(sections[0])
    if any(len(s) != m for s in sections):
        raise ValueError("every section needs the same number of stations")
    ang = [float(a) % TAU for a in angles]

    def blend(a):
        # Locate `a` between two sections and blend them with a smoothstep in angle.
        for k in range(n):
            a0, a1 = ang[k], ang[(k + 1) % n]
            span = (a1 - a0) % TAU
            off = (a - a0) % TAU
            if off <= span or span == 0.0:
                t = 0.0 if span == 0.0 else off / span
                w = t * t * (3.0 - 2.0 * t)
                s0, s1 = sections[k], sections[(k + 1) % n]
                return [(s0[j][0] * (1 - w) + s1[j][0] * w,
                         s0[j][1] * (1 - w) + s1[j][1] * w) for j in range(m)]
        return list(sections[0])

    cols = []
    for i in range(steps):
        a = TAU * i / steps
        cols.append(blend(a))

    verts = []
    for j in range(m):
        for i in range(steps):
            a = TAU * i / steps
            r_frac, z = cols[i][j]
            rr = (plan[i % len(plan)] if plan is not None else ref) * r_frac
            cx, cy = centre(j / max(1, m - 1)) if centre else (0.0, 0.0)
            verts.append((cx + rr * math.cos(a), cy + rr * math.sin(a), z))
    faces = []
    for j in range(m - 1):
        a0, b0 = j * steps, (j + 1) * steps
        for i in range(steps):
            k = (i + 1) % steps
            faces.append([a0 + i, a0 + k, b0 + k, b0 + i])
    return MeshProgram().mesh(verts=verts, faces=faces, mark=mark)
