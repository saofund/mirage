"""One-to-one reconstruction of a 2007 Volkswagen Polo filler region.

The earlier fuelcap hero reproduces a modern white SUV with a rectangular pocket. This
scene deliberately starts from another photograph and another construction: a round
pressed opening in metallic blue bodywork, a tilted wet cap, a compact latch at the left,
and a nearly edge-on circular door at the right. Dimensions are ratios measured from:

    fuelcap/_ref/bycar/Polo/粗筛done2_Polo_2007款劲情14自动风尚版_38.png

The model frame is the body panel: +x is right, +y is up, +z is out of the paint.
"""
from __future__ import annotations

import math

from mirage.meshlang import MeshProgram
from mirage.textures import ensure_textures

from fuelcap import materials as FM
from fuelcap import parts as P

MM = 1e-3
TAU = 2.0 * math.pi

OPEN_RX = 91.0 * MM
OPEN_RY = 128.0 * MM
OPEN_R = max(OPEN_RX, OPEN_RY)
CAP_D = 113.0 * MM
POCKET_DEPTH = 48.0 * MM
CAP_TILT = 24.0
CAP_SPIN = 84.0

TEX = ensure_textures(["fuelcap_polo_blue_paint", "fuelcap_plastic",
                       "fuelcap_polo_liner", "fuelcap_polo_cap"])


def mat(color, metallic=0.0, roughness=0.5, maps=None, uv_scale=1.0):
    out = {"color": list(color), "metallic": metallic, "roughness": roughness}
    if maps:
        out.update(albedo_map=str(maps["albedo"]), roughness_map=str(maps["rough"]),
                   normal_map=str(maps["normal"]), uv_scale=uv_scale)
    return out


PAINT = mat((0.080, 0.285, 0.470), 0.32, 0.16,
            maps=TEX["fuelcap_polo_blue_paint"], uv_scale=0.060)
LINER = mat((0.038, 0.040, 0.043), 0.0, 0.70,
            maps=TEX["fuelcap_polo_liner"], uv_scale=0.014)
CAP = mat((0.072, 0.074, 0.077), 0.0, 0.50,
          maps=TEX["fuelcap_polo_cap"], uv_scale=0.010)
RUBBER = mat((0.008, 0.008, 0.009), 0.0, 0.84)
STEEL = mat((0.038, 0.040, 0.043), 0.58, 0.52)
WATER = mat((0.34, 0.38, 0.42), 0.0, 0.055)
LAMP_RED = mat((0.31, 0.006, 0.008), 0.18, 0.090)
LAMP_DARK = mat((0.045, 0.002, 0.003), 0.10, 0.18)
LAMP_CLEAR = mat((0.65, 0.66, 0.62), 0.05, 0.08)
LAMP_RIB = mat((0.14, 0.003, 0.004), 0.12, 0.14)


def _ellipse_plan(rx, ry, steps):
    out = []
    for i in range(steps):
        a = TAU * i / steps
        c, s = math.cos(a), math.sin(a)
        out.append(1.0 / math.sqrt((c / rx) ** 2 + (s / ry) ** 2))
    return out


def _ring_solid(r_outer, r_inner, z_front, z_back, steps=96, mark="well", material=None,
                plan=None):
    section = [(r_inner, z_front), (r_outer, z_front), (r_outer, z_back),
               (r_inner, z_back)]
    return (MeshProgram().profile(section, plane="xz", closed=True)
            .spin(axis="z", steps=steps, plan=plan, plan_from=0.0, mark=mark)
            .material({"by": "tag", "name": mark}, **(material or LINER)))


def _smooth_liner(plan, ref, depth=POCKET_DEPTH, steps=96):
    """Continuous pressed bowl: one fold and one broad wall, not concentric terraces."""
    section = [(0.0, -depth - 10 * MM), (ref, -depth - 10 * MM),
               (ref, -3.0 * MM), (ref * 0.955, -4.0 * MM),
               (ref * 0.885, -12.0 * MM), (ref * 0.785, -31.0 * MM),
               (ref * 0.685, -depth), (0.0, -depth)]
    return (MeshProgram().profile(section, plane="xz", closed=False)
            .spin(axis="z", steps=steps, plan=plan, plan_from=0.0, mark="well")
            .material({"by": "tag", "name": "well"}, **LINER))


def _disc(radius, height, mark, material, sides=72):
    return (MeshProgram().cylinder(sides=sides, radius=radius, height=height, mark=mark)
            .material({"by": "tag", "name": mark}, **material))


def _box(size, mark, material):
    return (MeshProgram().cube(size=1.0, mark=mark)
            .scale({"by": "all"}, list(size))
            .material({"by": "tag", "name": mark}, **material))


def _tube(path, radius, material, mark="panel", sides=10):
    profile = [(radius * math.cos(TAU * i / sides), radius * math.sin(TAU * i / sides))
               for i in range(sides)]
    return (MeshProgram().profile(profile, plane="xy", closed=True)
            .sweep(path, mark=mark)
            .material({"by": "tag", "name": mark}, **material))


def _lofted_lamp(outline, depths, scales, material, mark="tail_lamp"):
    """Closed multi-ring lens with a rounded-looking bevel, authored as one mesh."""
    n = len(outline)
    cx = sum(x for x, _ in outline) / n
    cy = sum(y for _, y in outline) / n
    verts = []
    for z, scale in zip(depths, scales):
        verts.extend((cx + (x - cx) * scale, cy + (y - cy) * scale, z)
                     for x, y in outline)
    faces = []
    rings = len(depths)
    for k in range(rings - 1):
        a, b = k * n, (k + 1) * n
        for i in range(n):
            j = (i + 1) % n
            faces.append([a + i, a + j, b + j, b + i])
    faces.append(list(reversed(range(n))))
    faces.append([(rings - 1) * n + i for i in range(n)])
    return (MeshProgram().mesh(verts=verts, faces=faces, mark=mark)
            .material({"by": "tag", "name": mark}, **material))


def _panel_details(prog):
    # The Polo photograph is readable as a car because the tall rear lamp and the body
    # seam remain in frame. Omitting them makes even accurate filler geometry float on a
    # blue material test card.
    seam_path = []
    for i in range(22):
        u = i / 21.0
        y = 0.315 - 0.650 * u
        x = -0.298 + 0.017 * math.sin((u - 0.20) * math.pi)
        seam_path.append((x, y, 4.2 * MM))
    prog = prog.place(obj=_tube(seam_path, 1.45 * MM, RUBBER, "body_seam"))

    outline = [(-0.447, 0.300), (-0.360, 0.286), (-0.318, 0.210),
               (-0.313, 0.070), (-0.326, -0.115), (-0.350, -0.295),
               (-0.432, -0.330), (-0.468, -0.230), (-0.475, 0.180)]
    backing = _lofted_lamp(outline, (2.0 * MM, 7.0 * MM), (1.04, 1.0), LAMP_DARK)
    lens = _lofted_lamp(outline, (7.2 * MM, 12.5 * MM, 18.0 * MM),
                        (1.0, 0.975, 0.92), LAMP_RED)
    prog = prog.place(obj=backing).place(obj=lens)

    # Clear reversing-lamp insert and the repeated horizontal reflector bands.
    clear = [(-0.454, -0.192), (-0.350, -0.164), (-0.357, -0.292),
             (-0.431, -0.319), (-0.461, -0.280)]
    prog = prog.place(obj=_lofted_lamp(clear, (18.1 * MM, 20.0 * MM),
                                       (1.0, 0.94), LAMP_CLEAR))
    for y in (0.205, 0.155, 0.105, 0.055, 0.005, -0.045, -0.095):
        prog = prog.place(obj=_box((0.105, 2.2 * MM, 2.0 * MM), "tail_lamp", LAMP_RIB),
                          at=(-0.390, y, 18.3 * MM), rotate=(0.0, 0.0, -4.0))

    # Lower bumper joint and the subtle shoulder crease above the filler.
    bumper = [(x, -0.305 + 0.012 * math.cos((x + 0.05) * 4.5), 2.0 * MM)
              for x in [(-0.28 + 0.04 * i) for i in range(20)]]
    shoulder = [(x, 0.235 + 0.008 * math.sin((x + 0.18) * 5.0), 3.0 * MM)
                for x in [(-0.27 + 0.04 * i) for i in range(19)]]
    prog = prog.place(obj=_tube(bumper, 1.2 * MM, mat((0.010, 0.012, 0.014), 0, 0.7),
                                "body_seam"))
    return prog.place(obj=_tube(shoulder, 0.7 * MM,
                                mat((0.20, 0.35, 0.48), 0.25, 0.22), "panel"))


def _latch(prog):
    # A square pocket and spring-loaded circular plunger on the left wall.
    plate = _box((34 * MM, 39 * MM, 4 * MM), "well", RUBBER)
    prog = prog.place(obj=plate, at=(-82 * MM, -3 * MM, -22 * MM))
    prog = prog.place(obj=_disc(9.2 * MM, 7.0 * MM, "well", STEEL, 32),
                      at=(-82 * MM, -3 * MM, -19 * MM))
    prog = prog.place(obj=_disc(5.7 * MM, 8.0 * MM, "well", LINER, 32),
                      at=(-82 * MM, -3 * MM, -12 * MM))
    for x, y in ((-94, 9), (-70, 10), (-94, -17), (-69, -16)):
        prog = prog.place(obj=P.screw(r=2.1 * MM, head_h=0.8 * MM, material=STEEL),
                          at=(x * MM, y * MM, -17 * MM))
    # The small exposed latch tongue that reaches toward the door.
    tongue = _box((11 * MM, 7 * MM, 3.0 * MM), "well", STEEL)
    return prog.place(obj=tongue, at=(-67 * MM, -3 * MM, -13 * MM),
                      rotate=(0.0, 0.0, 8.0))


def _cap(prog):
    cap_obj = P.cap(d=CAP_D, flange=12 * MM, rib_len=CAP_D * 0.92,
                    rib_w=CAP_D * 0.42, rib_h=CAP_D * 0.095,
                    rib_draft=0.82, dome=-0.6 * MM, chamfer=2.1 * MM,
                    flutes=18, flute_depth=0.010, skirt=18 * MM,
                    neck_d=CAP_D * 0.72, bevel=0.040, spin=CAP_SPIN,
                    printing=False, grip="rib", waist=0.90, rib_dish=-0.04,
                    rib_shoulder=0.66, material=CAP, steps=96)
    # Unlike the old hero, this cap is not parallel to the body. Its face ellipse and the
    # visible lower skirt in the source require a separately tilted filler-neck axis.
    prog = prog.place(obj=cap_obj, at=(4 * MM, -38 * MM, -18 * MM),
                      rotate=(CAP_TILT, -5.0, 0.0))

    # Opaque glossy beads are a better approximation than a flat decal in Mirage's current
    # material model: each one produces the tiny white specular point visible in the photo.
    droplets = [(-31, 20, 0.8), (-20, 28, 1.1), (-9, 25, 0.7), (3, 31, 1.0),
                (17, 24, 0.8), (27, 17, 1.2), (-35, 5, 0.7), (-24, -5, 1.0),
                (-12, 7, 0.8), (8, 10, 1.3), (22, 2, 0.7), (31, -10, 1.0),
                (-28, -22, 1.2), (-8, -26, 0.8), (15, -21, 1.0), (29, -27, 0.7)]
    # Droplets are kept close to the face. The small positive z survives the cap tilt and
    # prevents coplanar flicker without making the beads float visibly.
    rx, ry = math.radians(CAP_TILT), math.radians(-5.0)
    cx, sx = math.cos(rx), math.sin(rx)
    cy, sy = math.cos(ry), math.sin(ry)

    def world_on_cap(x, y, z):
        # Same Ry @ Rx composition as `place`; Rz is zero for this assembly.
        qx = cy * x + sy * (sx * y + cx * z)
        qy = cx * y - sx * z
        qz = -sy * x + cy * (sx * y + cx * z)
        return (4 * MM + qx, -38 * MM + qy, -18 * MM + qz)

    for x, y, r in droplets:
        bead = (MeshProgram().uv_sphere(segments=12, rings=7, radius=r * MM, mark="water")
                .scale({"by": "all"}, [1.0, 1.0, 0.42])
                .material({"by": "tag", "name": "water"}, **WATER))
        prog = prog.place(obj=bead, at=world_on_cap(x * MM, y * MM, 1.6 * MM),
                          rotate=(CAP_TILT, -5.0, 0.0))
    return prog


def _door(prog):
    door_w, door_h = 194 * MM, 231 * MM
    door_ref = max(door_w, door_h) / 2.0
    door_plan = [r / door_ref for r in _ellipse_plan(door_w / 2, door_h / 2, 96)]
    door = P.fuel_door(w=door_w, h=door_h, flange=9 * MM, face=5 * MM,
                       rim=8 * MM, open_deg=118.0, az=0.0, hinge_r=102 * MM,
                       gap=3 * MM, steps=96, skin=PAINT, liner=PAINT, strap=False,
                       latch=False, plan=door_plan, inside_material=PAINT,
                       inner_details=True)
    prog = prog.place(obj=door, at=(0.0, -65 * MM, 2.0 * MM))

    # The broad stamped hinge plate is the largest blue shape inside the opening's right
    # edge. Three raised ribs sit on it; isolated bars read as an exploded assembly.
    hinge_plate = P.prism([(82 * MM, 48 * MM), (166 * MM, 39 * MM),
                           (171 * MM, -50 * MM), (88 * MM, -31 * MM)],
                          12 * MM, 16 * MM, mark="door").material(
                              {"by": "tag", "name": "door"}, **PAINT)
    prog = prog.place(obj=hinge_plate)
    for y in (-42, -8, 27):
        prog = prog.place(obj=_box((63 * MM, 7 * MM, 4 * MM), "door", PAINT),
                          at=(126 * MM, y * MM, 17 * MM), rotate=(0.0, 0.0, -7.0))
    pin = _disc(5.0 * MM, 72 * MM, "door", STEEL, 28)
    prog = prog.place(obj=pin, at=(111 * MM, -43 * MM, -1 * MM),
                      rotate=(90.0, 0.0, 0.0))

    # A few beads on the inner door face; sparse enough to remain details, not a pattern.
    for y, z, r in ((42, 6, 1.0), (15, 10, 0.8), (-18, 7, 1.1), (-47, 4, 0.7)):
        bead = (MeshProgram().uv_sphere(segments=12, rings=7, radius=r * MM, mark="water")
                .scale({"by": "all"}, [1.0, 1.0, 0.45])
                .material({"by": "tag", "name": "water"}, **WATER))
        prog = prog.place(obj=bead, at=(212 * MM, y * MM, (65 + z) * MM),
                          rotate=(0.0, -118.0, 0.0))
    return prog


def build():
    steps = 96
    radii = _ellipse_plan(OPEN_RX, OPEN_RY, steps)
    plan = [r / OPEN_R for r in radii]
    prog = P.panel(size=0.92, hole_d=OPEN_R * 2, thick=10 * MM, ring=steps,
                   crown=34 * MM, crown_ax=math.radians(6.0),
                   hole_plan=radii, material=PAINT)
    prog = _panel_details(prog)

    # Rubber weather bead and the five-level circular liner. The shallow first flange and
    # steep wall reproduce the strong black ring plus broad soft inner bowl of the Polo.
    # A painted rolled lip catches the thin cyan highlight in the photograph; the rubber
    # weather bead begins inside it instead of replacing the whole edge with black.
    prog = prog.place(obj=_ring_solid(OPEN_R + 4 * MM, OPEN_R + 0.4 * MM,
                                      1.6 * MM, -2.5 * MM, material=PAINT, plan=plan))
    prog = prog.place(obj=_ring_solid(OPEN_R + 0.2 * MM, OPEN_R - 4.0 * MM,
                                      0.5 * MM, -5 * MM, material=RUBBER, plan=plan))
    prog = prog.place(obj=_smooth_liner(plan, OPEN_R, steps=steps))
    # The real neck housing is not rotationally symmetric: a broad moulded hood rises
    # behind the cap and fills the upper third of the bowl.
    shroud = (MeshProgram().uv_sphere(segments=48, rings=28, radius=1.0, mark="well")
              .scale({"by": "all"}, [76 * MM, 91 * MM, 24 * MM])
              .material({"by": "tag", "name": "well"}, **LINER))
    prog = prog.place(obj=shroud, at=(3 * MM, 38 * MM, -44 * MM),
                      rotate=(8.0, 0.0, 0.0))
    # Fasteners and moulding pips break the perfect radial symmetry of a generated bowl.
    for x, y, r in ((63, 86, 3.0), (-57, 84, 2.2), (67, -76, 2.0), (-62, -74, 1.8)):
        prog = prog.place(obj=P.screw(r=r * MM, head_h=0.9 * MM,
                                      slot=(r > 2.5), material=STEEL),
                          at=(x * MM, y * MM, -6.0 * MM))
    prog = prog.place(obj=P.neck_stack(CAP_D * 0.44, -POCKET_DEPTH,
                                       -24 * MM, flare=1.34, material=LINER),
                      at=(4 * MM, -18 * MM, 0.0), rotate=(CAP_TILT, -5.0, 0.0))
    prog = _latch(prog)
    prog = _cap(prog)
    return _door(prog)


def pose(distance=0.74):
    # The circular opening is 0.83 as wide as it is high in the photograph, fixing the
    # horizontal camera obliquity. Looking slightly from the right also exposes the door's
    # blue inner face instead of reducing it to a line.
    eye = (0.245, -0.020, distance)
    target = (-0.018, -0.030, -0.012)
    return {"eye": eye, "target": target, "up": (0.0, 1.0, 0.02), "fov": 0.570}
