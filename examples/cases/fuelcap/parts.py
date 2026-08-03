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

from mirage.meshlang import MeshProgram

from .materials import (
    CAP_ALU, CAP_BLACK, CAP_CHROME, NECK_STEEL, SEAL_RED, SEAL_RUBBER, TETHER,
    WELL_METAL, WELL_PLASTIC, mat,
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


def stadium(length, width, arc=10):
    """A rounded-rectangle plan (CCW): the grip rib seen from above."""
    a, r = max(1e-6, (length - width) / 2.0), width / 2.0
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
        spin=0.0, material=None, rib_material=None, steps=48):
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
    p = lathe(section, steps=steps, mark="cap_body").material({"by": "tag", "name": "cap_body"}, **material)

    if teeth:
        # ratchet teeth: little wedges around the skirt, arrayed by placing them.
        tw, th = TAU * rn / teeth * 0.55, 0.0022
        for i in range(teeth):
            a = TAU * i / teeth
            p = p.place(obj=prism([(-tw / 2, -th), (tw / 2, -th), (tw / 2, th), (-tw / 2, th)],
                                  0.0, 0.006, mark="cap_teeth"),
                        at=(rn * math.cos(a), rn * math.sin(a), -flange - skirt * 0.55),
                        rotate=(0.0, 0.0, math.degrees(a)), material=material)

    # the grip rib, turned about the cap's own axis by `spin`
    sa, ca = math.sin(math.radians(spin)), math.cos(math.radians(spin))
    turn = lambda pl: [(x * ca - y * sa, x * sa + y * ca) for x, y in pl]
    plan = turn(stadium(rib_len, rib_w))
    if rib_slot > 0:
        # the groove down the rib's spine: an inner stadium walked back the other way makes
        # the plan concave, which is exactly why prism() ear-clips.
        pass
    p = p.place(obj=frustum(plan, dome * 0.5, rib_h, rib_draft, mark="cap_rib"),
                at=(0.0, 0.0, 0.0), material=rib_material)
    if rib_slot > 0:
        sl, sw = rib_len * 0.72, rib_w * rib_draft * 0.34
        p = p.place(obj=frustum(turn(stadium(sl, sw)), rib_h - min(rib_slot, rib_h * 0.55),
                                rib_h + 1e-4, 1.25, mark="cap_slot"),
                    at=(0.0, 0.0, 0.0), material=mat([c * 0.55 for c in material["color"]],
                                                     material["metallic"], material["roughness"]))
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


def panel(size=0.44, hole_d=0.135, thick=0.010, material=None, steps=48, ring=28):
    """The body panel around the pocket: an annular plate, not a plate with a hole in it.

    Same reasoning as `well` — the aperture is built, not subtracted. A ring of quads from
    the hole edge out to a square border is a dozen lines and never fails; the boolean it
    replaces is the single most expensive thing that could be in this loop."""
    material = material or {"color": [0.3, 0.3, 0.3], "metallic": 0.6, "roughness": 0.15}
    rh, s = hole_d / 2.0, size / 2.0
    verts, faces = [], []
    for z in (0.0, -thick):
        for i in range(ring):
            a = TAU * i / ring
            verts.append((rh * math.cos(a), rh * math.sin(a), z))
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
            verts.append((s * (cx_ + (nx_ - cx_) * u), s * (cy_ + (ny_ - cy_) * u), z))
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


def door(w=0.175, h=0.165, thick=0.008, open_deg=95.0, hinge_x=-0.10, skin=None, liner=None):
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
    # hinged along the door's own -x edge, swung out about the panel's y axis. The door
    # opens AWAY from the pocket, so `hinge_x` is negative and `open_deg` positive.
    return MeshProgram().place(obj=body, at=(hinge_x, 0.0, 0.0), rotate=(0.0, -open_deg, 0.0))


def tether(start, end, sag=0.030, coils=3.0, r=0.0022, n=64, material=None):
    """The curly cord from the cap to the pocket. Swept, not assembled from arcs.

    A helix that hangs — the coil is stretched along the chord and pulled down by gravity,
    which is what makes it read as a cord rather than as a spring in a catalogue."""
    sx, sy, sz = start
    ex, ey, ez = end
    path = []
    for i in range(n + 1):
        t = i / n
        a = TAU * coils * t
        px = sx + (ex - sx) * t
        py = sy + (ey - sy) * t + 0.010 * math.sin(a) * math.sin(math.pi * t)
        pz = sz + (ez - sz) * t - sag * math.sin(math.pi * t) + 0.006 * (math.cos(a) - 1) * math.sin(math.pi * t)
        path.append((px, py, pz))
    ring = [(r * math.cos(TAU * k / 8), r * math.sin(TAU * k / 8)) for k in range(8)]
    return (MeshProgram().profile(ring, plane="xy", closed=True).sweep(path, mark="tether")
            .material({"by": "tag", "name": "tether"}, **(material or TETHER)))
