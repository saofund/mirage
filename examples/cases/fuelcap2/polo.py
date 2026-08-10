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

# The stamped aperture is almost circular in the photograph.  The old 91 x 128 mm
# ellipse tried to encode camera perspective in the part itself and produced the tall
# black oval that dominated every comparison.
# Measured off the photograph the same way case 27's was: the aperture spans 560 px along
# the direction of steepest foreshortening against a cap whose ellipse is 257 px on its
# major axis, so that axis of the opening is 2.46 cap diameters. It was drawn at 2.2, which
# is the same error case 27 had and the one a viewer reads first.
#
# The ASPECT is left where the earlier measurement put it. The probe above reads one
# direction, so it says how big the opening is and nothing about how oval; scaling both axes
# by 1.115 keeps the 0.906 that was measured separately. Changing the aspect on this
# evidence would have been inventing a number, and the case's own test said so.
OPEN_RX = 129.3 * MM
OPEN_RY = 142.7 * MM
OPEN_R = max(OPEN_RX, OPEN_RY)
CAP_D = 116.0 * MM
POCKET_DEPTH = 48.0 * MM
CAP_TILT = 12.0
# The door's open angle, as a module constant so the depth-residual loop can sweep it.
# It was two literals in two places -- the door and the ribs that ride on it -- which is
# why a sweep had to rewrite the source to move it, and why the two could silently
# disagree. A parameter the loop is meant to estimate has to be reachable.
# 101, and the story of this number is the most useful thing in this file.
#
# Sweeping it against the depth residual in the door's image region gave a clean minimum:
# 6.6 mm at 70 against 70.5 at 100 and 78.6 at 115, an order of magnitude. It is also
# completely wrong -- at 70 the door lies ACROSS the pocket and hides it. The metric was
# measuring a crop whose CONTENT changes with the parameter, so a door swung over the
# opening puts a flat surface at body depth across the whole region, which matches the
# prior's smooth body far better than an open pocket does. The residual rewarded occlusion.
#
# A depth residual alone cannot estimate a parameter that changes what is visible. It needs
# a silhouette term -- the photographed door covers a particular area with a particular
# outline, and 70 gets both wrong -- which `mirage.compare.silhouette` provides and this
# loop does not yet use. Until it does, this stays where the picture puts it.
DOOR_OPEN_DEG = 101.0
CAP_SPIN = 84.0

TEX = ensure_textures(["fuelcap_polo_blue_paint", "fuelcap_plastic",
                       "fuelcap_polo_liner", "fuelcap_polo_cap",
                       "fuelcap_wet_liner", "fuelcap_wet_cap"])


def mat(color, metallic=0.0, roughness=0.5, maps=None, uv_scale=1.0):
    out = {"color": list(color), "metallic": metallic, "roughness": roughness}
    if maps:
        out.update(albedo_map=str(maps["albedo"]), roughness_map=str(maps["rough"]),
                   normal_map=str(maps["normal"]), uv_scale=uv_scale)
    return out


PAINT = mat((0.065, 0.245, 0.405), 0.32, 0.16,
            maps=TEX["fuelcap_polo_blue_paint"], uv_scale=0.060)
# The procedural roughness map was authored for generic metallic paint and creates a
# concentrated white lobe on this strongly crowned panel.  The photographed older paint
# has a broader clear-coat response, so keep its colour/normal variation but fix roughness.
PAINT.pop("roughness_map", None)
PAINT["roughness"] = 0.29
PAINT["metallic"] = 0.18
LINER = mat((0.038, 0.040, 0.043), 0.0, 0.70,
            maps=TEX["fuelcap_polo_liner"], uv_scale=0.014)
CAP = mat((0.072, 0.074, 0.077), 0.0, 0.50,
          maps=TEX["fuelcap_polo_cap"], uv_scale=0.010)
CAP.pop("roughness_map", None)
CAP["roughness"] = 0.64
CAP_GRIP = mat((0.043, 0.045, 0.048), 0.0, 0.58)
RUBBER = mat((0.008, 0.008, 0.009), 0.0, 0.84)
STEEL = mat((0.060, 0.063, 0.068), 0.52, 0.50)
WATER = mat((0.34, 0.38, 0.42), 0.0, 0.055)
LAMP_RED = mat((0.31, 0.006, 0.008), 0.18, 0.090)
LAMP_DARK = mat((0.045, 0.002, 0.003), 0.10, 0.18)
LAMP_CLEAR = mat((0.65, 0.66, 0.62), 0.05, 0.08)
LAMP_RIB = mat((0.14, 0.003, 0.004), 0.12, 0.14)
DOOR_INNER = mat((0.105, 0.330, 0.500), 0.18, 0.28)
HINGE_BLUE = mat((0.155, 0.405, 0.570), 0.14, 0.32)


# The wet moulding and the wet cap. The Polo was photographed in the rain, and the beading
# is most of that picture's surface character — a few hundred tiny near-mirrors on a matt
# black shell, each returning a hard white point where the substrate around it returns
# almost nothing. It is a ROUGHNESS and NORMAL effect, not an albedo one: painting light
# dots into a colour map reads as dirt. `textures._beaded_water` does it properly.
LINER_WET = mat((0.026, 0.027, 0.029), 0.0, 0.70,
                maps=TEX["fuelcap_wet_liner"], uv_scale=0.075)
CAP_WET = mat((0.062, 0.064, 0.068), 0.0, 0.58,
              maps=TEX["fuelcap_wet_cap"], uv_scale=0.045)


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
    """Continuous asymmetric pressing whose throat follows the lowered filler neck.

    A lathed profile necessarily draws concentric bands.  This Polo moulding does not:
    its throat is about 25 mm below the aperture centre, leaving a tall hood above the
    cap and a short drain shelf below it.  Interpolated rings preserve that measured
    silhouette while giving the renderer enough strips to shade it as one surface.
    """
    # THE FLANGE, which this part did not have. Between the paint's rolled edge and the
    # bowl's mouth the photograph has a wide, nearly flat black annulus — about a quarter of
    # the aperture radius, carrying the two moulding bosses and catching the one soft
    # highlight in the whole pocket. Without it the liner runs straight from the paint into
    # the bowl, and the entire aperture renders as one flat dark disc with a cap in it,
    # which is what it did.
    #
    # `flange` is that annulus as a fraction of the radius; the bowl proper starts inside it.
    flange = 0.24
    rings = 26
    verts = []
    for j in range(rings):
        u = j / (rings - 1)
        if u <= 0.30:                       # the flange: barely dropping, barely narrowing
            t = u / 0.30
            radial = 1.0 - flange * t
            z = -3.0 * MM - 5.0 * MM * t * t
            centre_y = 0.0
            hood_k = 0.0
        else:                               # the bowl
            v = (u - 0.30) / 0.70
            ease = v * v * (3.0 - 2.0 * v)
            radial = (1.0 - flange) * (1.0 - 0.46 * ease)
            z = -8.0 * MM - depth * ease
            centre_y = -25.0 * MM * ease
            hood_k = ease
        for i in range(steps):
            a = TAU * i / steps
            directional_r = ref * plan[i]
            # The upper hood rolls farther inward than the drain shelf at the bottom.
            hood = max(math.sin(a), 0.0) * 4.0 * MM * hood_k
            r = directional_r * radial - hood
            verts.append((r * math.cos(a), centre_y + r * math.sin(a), z))
    faces = []
    for j in range(rings - 1):
        a0, b0 = j * steps, (j + 1) * steps
        for i in range(steps):
            k = (i + 1) % steps
            faces.append([a0 + i, a0 + k, b0 + k, b0 + i])
    # RADIUS THE CREASES. Every edge on an injection moulding has a radius on it — the
    # tool cannot make a zero-radius corner and the part could not leave it if it could —
    # and a zero-radius corner is a large part of why a render reads as CAD. This loft's
    # flange/bowl junction is a 96-edge ring, so `edge_bevel` can round it.
    #
    # `interior` matters: without it `sharp` also returns the loft's two open rims, which
    # cannot be bevelled, and the prune cascades until nothing is left. The angle has to be
    # low enough to catch the WHOLE ring — at 25 degrees only 54 of the 96 qualify, the
    # ring is broken, and every one of them is pruned as a lone cut.
    return (MeshProgram().mesh(verts=verts, faces=faces, mark="well")
            .edge_bevel({"by": "sharp", "angle": 6.0, "interior": True}, width=0.20)
            .material({"by": "tag", "name": "well"}, **LINER_WET))



def _liner_features(prog):
    """Moulded features pressed INTO the liner surface, not placed on top of it.

    Everything this case had inside the aperture was a separate solid dropped in front of a
    smooth loft, and a loft can only be given features by rewriting the loft. `inset` +
    `extrude` puts a boss or a recess on a surface that already exists, which is how every
    pad, shelf and land in a real pocket is made — and it is the operator these fuelcap
    cases had never once used.

    Boxes are in the liner's own frame; the loft runs from the flange at z = -3 mm out at
    r = 97..143, down the wall, to a throat ring at z = -55 spanning x -57..57, y -86..36.
    """
    # Boxes taken FROM the surface, not guessed at it. The loft is 27 rings of 96; walking
    # a ring band over an angular span and taking its bounds gives a box that is guaranteed
    # to select faces. Guessed boxes miss — the first three did, and `resolve` says so with
    # a face count and a bbox, which is how they got fixed.
    #
    # the drain shelf across the bottom of the bowl, sunk
    prog = P.emboss(prog, (0.0, -0.0934, -0.0330), (0.0302, 0.0076, 0.0081),
                    -0.0035, inset=0.30)
    # the broad shallow rib up the hood, raised — it is what breaks the hood's single
    # smooth gradient into the two tones the photograph has
    prog = P.emboss(prog, (0.0, 0.0720, -0.0270), (0.0400, 0.0210, 0.0099),
                    0.0022, inset=0.42)
    # the land the latch bracket bolts to, on the left wall
    prog = P.emboss(prog, (-0.0818, -0.0079, -0.0232), (0.0119, 0.0300, 0.0095),
                    0.0028, inset=0.28)
    # two moulding pads on the flange, upper right and upper left
    prog = P.emboss(prog, (0.066, 0.088, -0.005), (0.013, 0.012, 0.005), 0.0018, inset=0.34)
    prog = P.emboss(prog, (-0.058, 0.092, -0.005), (0.011, 0.010, 0.005), 0.0015, inset=0.34)
    # PAINT LAST. `inset` and `extrude` create faces, and a created face carries the
    # renderer's default albedo, not the material the surface it grew out of was given.
    # Embossing after the material call turned the whole liner from black plastic into a
    # white plate -- the same class of mistake as selecting only +z and -z on the door and
    # leaving its turned edge default, which this kit has already made once.
    return prog.material({"by": "all"}, **LINER_WET)


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
        x = -0.278 + 0.017 * math.sin((u - 0.20) * math.pi)
        seam_path.append((x, y, 4.2 * MM))
    prog = prog.place(obj=_tube(seam_path, 1.45 * MM, RUBBER, "body_seam"))

    outline = [(-0.427, 0.300), (-0.340, 0.286), (-0.298, 0.210),
               (-0.293, 0.070), (-0.306, -0.115), (-0.330, -0.295),
               (-0.412, -0.330), (-0.448, -0.230), (-0.455, 0.180)]
    backing = _lofted_lamp(outline, (2.0 * MM, 7.0 * MM), (1.04, 1.0), LAMP_DARK)
    lens = _lofted_lamp(outline, (7.2 * MM, 12.5 * MM, 18.0 * MM),
                        (1.0, 0.975, 0.92), LAMP_RED)
    prog = prog.place(obj=backing).place(obj=lens)

    # Clear reversing-lamp insert and the repeated horizontal reflector bands.
    clear = [(-0.434, -0.192), (-0.330, -0.164), (-0.337, -0.292),
             (-0.411, -0.319), (-0.441, -0.280)]
    prog = prog.place(obj=_lofted_lamp(clear, (18.1 * MM, 20.0 * MM),
                                       (1.0, 0.94), LAMP_CLEAR))
    for y in (0.205, 0.155, 0.105, 0.055, 0.005, -0.045, -0.095):
        prog = prog.place(obj=_box((0.105, 2.2 * MM, 2.0 * MM), "tail_lamp", LAMP_RIB),
                          at=(-0.370, y, 18.3 * MM), rotate=(0.0, 0.0, -4.0))

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
    """The stamped latch housing on the left wall — a pressing, not a plate.

    In the reference this is the second-largest object inside the aperture after the cap: a
    squarish stamped housing let into the wall, with a deep recess in its face, a stepped
    barrel standing out of it, a hook reaching toward the door and four fasteners. Built as
    a flat box with two discs on it, which is what it was, it reads as a smudge.

    The recess is `inset(region=True)` + `extrude` inward — the operator that makes a
    pocket in a face that already exists, rather than a second box placed in front of one.
    """
    # the housing, let into the wall and drafted
    plate = (_box((52 * MM, 60 * MM, 5 * MM), "well", LINER)
             .inset({"by": "normal", "axis": "x", "sign": 1.0}, thickness=0.22, region=True)
             .extrude({"by": "last_created"}, distance=-3.2 * MM)
             .material({"by": "all"}, **LINER))
    prog = prog.place(obj=plate, at=(-88 * MM, -3 * MM, -22 * MM))

    # the stepped barrel: a wide collar, a shoulder, then the plunger
    barrel = (MeshProgram()
              .profile([(0.0, 0.0), (15.0 * MM, 0.0), (15.0 * MM, 5.0 * MM),
                        (11.5 * MM, 6.2 * MM), (11.5 * MM, 11.0 * MM),
                        (7.6 * MM, 12.4 * MM), (7.6 * MM, 17.5 * MM),
                        (5.2 * MM, 18.6 * MM), (0.0, 18.6 * MM)], plane="xz", closed=False)
              .spin(axis="z", steps=40, mark="well")
              .material({"by": "tag", "name": "well"}, **STEEL))
    prog = prog.place(obj=barrel, at=(-86 * MM, -1 * MM, -21 * MM), rotate=(0.0, 84.0, 0.0))

    # the hook that reaches toward the door, and its return
    hook = _box((17 * MM, 6 * MM, 3.4 * MM), "well", STEEL)
    prog = prog.place(obj=hook, at=(-66 * MM, 3 * MM, -12 * MM), rotate=(0.0, 0.0, -12.0))
    prog = prog.place(obj=_box((4 * MM, 6 * MM, 7 * MM), "well", STEEL),
                      at=(-58 * MM, 3 * MM, -13 * MM))

    # the tab at the far edge, folded out of the same pressing
    prog = prog.place(obj=_box((6 * MM, 22 * MM, 3 * MM), "well", LINER),
                      at=(-110 * MM, -6 * MM, -17 * MM), rotate=(0.0, 22.0, 0.0))

    for x, y in ((-107, 17), (-72, 18), (-107, -23), (-72, -22)):
        prog = prog.place(obj=P.screw(r=2.3 * MM, head_h=0.9 * MM, material=STEEL),
                          at=(x * MM, y * MM, -17.5 * MM))
    return prog


def _cap(prog):
    cap_obj = P.cap(d=CAP_D, flange=7 * MM, rib_len=CAP_D * 0.92,
                    rib_w=CAP_D * 0.42, rib_h=CAP_D * 0.095,
                    rib_draft=0.82, dome=-0.6 * MM, chamfer=2.1 * MM,
                    flutes=18, flute_depth=0.010, skirt=9 * MM,
                    neck_d=CAP_D * 0.72, bevel=0.040, spin=CAP_SPIN,
                    printing=False, grip="rib", waist=0.90, rib_dish=-0.04,
                    rib_shoulder=0.66, material=CAP,
                    rib_material=CAP_GRIP, steps=96)
    # Project the photographed wear only onto camera-facing local top surfaces. Applying it
    # to the fluted wall and drafted grip sides wraps unrelated image regions round every
    # edge and turns one cap into a collage.
    cap_obj = cap_obj.material({"by": "normal", "axis": "z", "sign": 1, "tol": 0.22},
                               **CAP_WET)
    # Unlike the old hero, this cap is not parallel to the body. Its face ellipse and the
    # visible lower skirt in the source require a separately tilted filler-neck axis.
    prog = prog.place(obj=cap_obj, at=(-15 * MM, -27 * MM, -18 * MM),
                      rotate=(CAP_TILT, -5.0, 0.0))

    # Opaque glossy beads are a better approximation than a flat decal in Mirage's current
    # material model: each one produces the tiny white specular point visible in the photo.
    droplets = [(-31, 20, 0.8), (-20, 28, 1.1), (-9, 25, 0.7), (3, 31, 1.0),
                (17, 24, 0.8), (27, 17, 1.2), (-35, 5, 0.7), (-24, -5, 1.0),
                (-12, 7, 0.8), (8, 10, 1.3), (22, 2, 0.7), (31, -10, 1.0),
                (-28, -22, 1.2), (-8, -26, 0.8), (15, -21, 1.0), (29, -27, 0.7),
                (-34, 13, 0.7), (-18, 15, 0.8), (-2, 18, 0.6), (13, 17, 0.7),
                (34, 8, 0.8), (-18, -14, 0.7), (1, -10, 0.9), (18, -8, 0.6)]
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
        return (-15 * MM + qx, -27 * MM + qy, -18 * MM + qz)

    for x, y, r in droplets:
        bead = (MeshProgram().uv_sphere(segments=12, rings=7, radius=r * MM * 1.45,
                                        mark="water")
                .scale({"by": "all"}, [1.0, 1.0, 0.42])
                .material({"by": "tag", "name": "water"}, **WATER))
        prog = prog.place(obj=bead, at=world_on_cap(x * MM, y * MM, 1.6 * MM),
                          rotate=(CAP_TILT, -5.0, 0.0))
    return prog


def _door(prog):
    door_w, door_h = 270 * MM, 243 * MM
    door_ref = max(door_w, door_h) / 2.0
    door_plan = [r / door_ref for r in _ellipse_plan(door_w / 2, door_h / 2, 96)]
    # The stamped RIBS on the inner face. A fuel door is a pressing, and the reference's
    # inner face is crossed by a raised outer ring and three radial webs — which is most of
    # what tells you it is sheet steel and not a disc. Placed on the door before it swings,
    # so they ride with it.
    ribs = MeshProgram()
    for a in (0.0, 118.0, 242.0):
        ribs = ribs.place(obj=_box((0.088, 0.016, 0.004), "door", HINGE_BLUE),
                          at=(0.0, 0.0, 0.0), rotate=(0.0, 0.0, a))
    ring = (MeshProgram()
            .profile([(0.086, 0.0), (0.098, 0.0), (0.098, 0.005), (0.086, 0.005)],
                     plane="xz", closed=True)
            .spin(axis="z", steps=64, mark="door")
            .material({"by": "tag", "name": "door"}, **HINGE_BLUE))
    inner = ribs.place(obj=ring)

    # OPEN ANGLE, from the depth residual rather than by eye. At 101 degrees the door
    # presents its face to the camera and renders as a large lens over what the photograph
    # shows as flat bodywork: the residual against the monocular prior put 83 mm mean and
    # 265 mm p95 in that region against 26 mm inside the pocket, and the map is a single
    # bright blob exactly the door's shape. The reference door is nearly edge-on.
    door = P.fuel_door(w=door_w, h=door_h, flange=9 * MM, face=5 * MM,
                       rim=8 * MM, open_deg=DOOR_OPEN_DEG, az=0.0, hinge_r=82 * MM,
                       gap=3 * MM, steps=96, skin=HINGE_BLUE, liner=DOOR_INNER, strap=False,
                       latch=False, plan=door_plan, inside_material=DOOR_INNER,
                       inner_details=False, inner_parts=inner)
    prog = prog.place(obj=door, at=(14 * MM, -48 * MM, 2.0 * MM))


    # A few beads on the inner door face; sparse enough to remain details, not a pattern.
    for y, z, r in ((42, 6, 1.0), (15, 10, 0.8), (-18, 7, 1.1), (-47, 4, 0.7)):
        bead = (MeshProgram().uv_sphere(segments=12, rings=7, radius=r * MM, mark="water")
                .scale({"by": "all"}, [1.0, 1.0, 0.45])
                .material({"by": "tag", "name": "water"}, **WATER))
        prog = prog.place(obj=bead, at=(214 * MM, (y - 48) * MM, (65 + z) * MM),
                          rotate=(0.0, -DOOR_OPEN_DEG, 0.0))
    return prog


def build():
    steps = 96
    radii = _ellipse_plan(OPEN_RX, OPEN_RY, steps)
    plan = [r / OPEN_R for r in radii]
    prog = P.panel(size=2.20, hole_d=OPEN_R * 2, thick=10 * MM, ring=steps,
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
    prog = prog.place(obj=_liner_features(_smooth_liner(plan, OPEN_R, steps=steps)))
    # Fasteners and moulding pips break the perfect radial symmetry of a generated bowl.
    for x, y, r in ((63, 86, 3.0), (-57, 84, 2.2), (67, -76, 2.0), (-62, -74, 1.8)):
        prog = prog.place(obj=P.screw(r=r * MM, head_h=0.9 * MM,
                                      slot=(r > 2.5), material=STEEL),
                          at=(x * MM, y * MM, -6.0 * MM))
    # Moulding pips and drip noses around the flange, as seen across the upper arc.
    for x, y, r in ((-34, 108, 2.0), (0, 118, 1.7), (35, 108, 2.1),
                    (-73, 77, 1.5), (72, 70, 1.8), (-69, -79, 1.4), (61, -91, 1.5)):
        prog = prog.place(obj=P.pip(r=r * MM, h=1.3 * MM, material=LINER),
                          at=(x * MM, y * MM, -2.0 * MM))
    # Close the off-centre throat behind the cap.  Without this backing the narrow gap on
    # the right sees the bright environment and becomes a false pale crescent.
    prog = prog.place(obj=_disc(82 * MM, 2 * MM, "well", RUBBER, 96),
                      at=(-15 * MM, -27 * MM, -55 * MM))
    prog = _latch(prog)
    prog = _cap(prog)
    return _door(prog)


def pose(distance=0.78):
    # Perspective belongs to the camera, not the aperture.  The source is only mildly
    # oblique: the circular lip stays circular while the open door is nearly edge-on.
    eye = (0.105, -0.020, distance)
    target = (-0.036, -0.030, -0.012)
    return {"eye": eye, "target": target, "up": (0.0, 1.0, 0.02), "fov": 0.570}
