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
# The door's ANGLE was never the problem, and neither was the side. `hinge_r` was 82 mm
# while the aperture's own radius is 142.7 -- the hinge was INSIDE the hole. A door pivoting
# from inside its own opening sweeps across it at every angle, which is exactly what every
# render showed and what three separate parameter sweeps (depth residual, silhouette, and
# solving n.d against the camera) each measured carefully within the wrong family.
#
# A parameter search cannot find a fault outside its parameter. Each sweep returned a
# confident number; none of them could express "the hinge is in the wrong place". Looking at
# the picture found it.
# 105, measured. With the hinge finally outside the aperture the angle means what it should,
# and a sweep against the photograph gives a clean minimum: door-region |diff| of 96 at 85,
# 70 at 95, 56.9 at 105, 62.7 at 115, rising steadily to 79 at 145.
#
# I derived 80 from n.d = 0 against the view direction and it is the WORST of the set. That
# arithmetic assumed the door's normal is (-sin, 0, cos), which is what `fuel_door`'s
# composition looked like on paper; the render says otherwise. Three derivations of this
# angle have now been wrong and one sweep has been right. When a part owns a layered
# transform, measure the result -- do not model the transform.
#
# ...and that sweep is now void, because of WHAT IT WAS COMPARING. It ran with no hinge
# assembly in the scene at all, so the gap between pocket and door was bare bodywork. The
# door at 105 degrees is a large lens lying across exactly that gap: it won the pixel
# comparison by covering an absence. Add the band and the plate that really occupy the
# gap and the same score stops preferring it. A whole-frame |diff| will happily buy one
# part's error with another part's, which is what it did here for four rounds.
#
# The replacement is not another sweep. It is two measurements of the door itself, taken
# off the photograph and checked by projecting the built part through the renderer's own
# camera (`tools/project.py`): the outer edge sits at 1.40 aperture radii and the
# silhouette is 1.97 radii tall, and its NEAR edge -- the one the hinge band disappears
# behind -- is at 1.19. Co-fitting the angle with the hinge stand-off against all three
# lands at 88 degrees and 1.12 aperture radii: 1.18, 1.38 and 1.96 against 1.19, 1.40 and
# 1.97, every one inside 3 mm on an aperture 272 mm across. 105 degrees gives 2.23 and
# 2.44, and the stand-off matters because at 1.06 the door's near edge creeps 10 mm left
# and eats the part of the band the photograph shows clear of it.
DOOR_OPEN_DEG = 88.0
DOOR_HINGE_R = 1.12
CAP_SPIN = 84.0

TEX = ensure_textures(["fuelcap_polo_blue_paint", "fuelcap_plastic",
                       "fuelcap_polo_liner", "fuelcap_polo_cap",
                       "fuelcap_wet_liner", "fuelcap_wet_cap",
                       "fuelcap_cast_cap"])


def mat(color, metallic=0.0, roughness=0.5, maps=None, uv_scale=1.0):
    out = {"color": list(color), "metallic": metallic, "roughness": roughness}
    if maps:
        out.update(albedo_map=str(maps["albedo"]), roughness_map=str(maps["rough"]),
                   normal_map=str(maps["normal"]), uv_scale=uv_scale)
    return out


PAINT = mat((0.065, 0.245, 0.405), 0.32, 0.16,
            maps=TEX["fuelcap_polo_blue_paint"], uv_scale=0.060)
# Clearcoat roughness, swept with the environment map in place: frame |diff| 51.8 at 0.05
# rising monotonically to 54.2 at 0.29. A car's clearcoat is 0.05-0.10, not 0.29.
#
# A correction to my own reasoning, kept because it was confidently wrong. I expected the
# environment map plus a smooth clearcoat to give the body the photograph's tonal spread of
# 53 grey levels, on the grounds that "paint is mostly the world reflected in it". It does
# not: at NORMAL incidence a dielectric returns about 4% specular, so a panel facing the
# camera shows mostly base coat, and the render's spread went the wrong way -- 38.6 at 0.29
# down to 35.0 at 0.05. The mirror-like reading only appears at GRAZING angles, which is
# why the photograph's bright sweep sits where the flank turns away. The environment map is
# necessary and it is not sufficient; the missing spread is curvature carrying the surface
# through grazing, not reflectivity.
PAINT.pop("roughness_map", None)
PAINT["roughness"] = 0.05
PAINT["metallic"] = 0.18
LINER = mat((0.026, 0.027, 0.030), 0.0, 0.74,
            maps=TEX["fuelcap_polo_liner"], uv_scale=0.014)
CAP = mat((0.072, 0.074, 0.077), 0.0, 0.50,
          maps=TEX["fuelcap_polo_cap"], uv_scale=0.010)
CAP.pop("roughness_map", None)
CAP["roughness"] = 0.64
CAP_GRIP = mat((0.043, 0.045, 0.048), 0.0, 0.58)
RUBBER = mat((0.008, 0.008, 0.009), 0.0, 0.84)
# The latch is the darkest metal in the pocket, not a catalogue steel. At 0.060 it
# rendered 36 luma ABOVE the photograph once the lighting was corrected -- the dark
# scene had been hiding it.
STEEL = mat((0.026, 0.027, 0.030), 0.48, 0.58)
# REAL WATER, now that the tracer has transmission. A bead is a lens: it refracts what is
# under it and returns a specular pinpoint off its crown, which is why a photographed one
# reads as a dark distorted disc with a bright speck. Two rounds were spent faking that with
# an opaque material -- 0.34 albedo gave fifty white blobs, 0.045 gave fifty invisible
# smudges -- because no opaque BSDF has the shape. `transmission` is what it needed.
WATER = mat((0.86, 0.90, 0.93), 0.0, 0.030)
WATER["transmission"] = 1.0
WATER["ior"] = 1.33
# The lamp lens is a COLOURED lens over a reflector, not red paint: its colour comes
# from absorption on the way through. As an opaque material it read as a flat red
# cartoon, which is what it has looked like in every render of this case.
LAMP_RED = mat((0.52, 0.020, 0.026), 0.0, 0.055)
LAMP_RED["transmission"] = 0.82
LAMP_RED["ior"] = 1.55
LAMP_DARK = mat((0.045, 0.002, 0.003), 0.10, 0.18)
LAMP_CLEAR = mat((0.65, 0.66, 0.62), 0.05, 0.08)
LAMP_RIB = mat((0.14, 0.003, 0.004), 0.12, 0.14)
DOOR_INNER = mat((0.105, 0.330, 0.500), 0.18, 0.28)
# The hinge band and its bracket are BODY-COLOURED pressings -- they are painted on the
# line with the rest of the car, and in the photograph they read as the same blue as the
# wing. They were 2.4x the paint's albedo, which is why the gap between pocket and door
# came out 34 luma brighter than the photograph's. What makes them look lighter in the
# reference is that they face more sky than the wing does, and that is the renderer's job,
# not the albedo's.
HINGE_BLUE = mat((0.075, 0.255, 0.415), 0.18, 0.12)


# The wet moulding and the wet cap. The Polo was photographed in the rain, and the beading
# is most of that picture's surface character — a few hundred tiny near-mirrors on a matt
# black shell, each returning a hard white point where the substrate around it returns
# almost nothing. It is a ROUGHNESS and NORMAL effect, not an albedo one: painting light
# dots into a colour map reads as dirt. `textures._beaded_water` does it properly.
LINER_WET = mat((0.026, 0.027, 0.029), 0.0, 0.70,
                maps=TEX["fuelcap_wet_liner"], uv_scale=0.190)
CAP_WET = mat((0.062, 0.064, 0.068), 0.0, 0.58,
              maps=TEX["fuelcap_wet_cap"], uv_scale=0.160)
# The cap's own surface, under the water. A photographed cap is granular at this
# magnification; a perfectly smooth disc is the single thing that most makes one read as a
# render, and no amount of correct silhouette fixes it.
# uv_scale is WORLD UNITS PER TILE, and at 0.030 a 1024-texel map puts every feature
# it has at 0.03 mm -- far below a pixel at any magnification this is judged at, so
# the whole map renders as dither. The casting grain should read at about 2 mm, and
# the generator's grain has a period of a seventysecond of a tile, so the tile has
# to be ~150 mm.
CAP_CAST = mat((0.058, 0.059, 0.062), 0.12, 0.50,
               maps=TEX["fuelcap_cast_cap"], uv_scale=0.150)


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
            # NEARLY STRAIGHT-SIDED. At a 0.46 shrink the throat closes to the cap's own
            # diameter and the cap fills the bowl with no black annulus round it; the
            # photograph has the cap at about 55% of the bowl's width all the way down,
            # which is a deep cylinder with a little draft, not a funnel.
            radial = (1.0 - flange) * (1.0 - 0.10 * ease)
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



def _ring_box(mesh, j0, j1, a0, a1, steps=96, pad=0.002):
    """A world box covering rings j0..j1 of the loft over the angular span a0..a1 degrees.

    Derived from the mesh, not written down. The first version of the feature pass had its
    five boxes as literals read off one build, and the moment the bowl's throat changed
    shape three of them selected nothing and the case stopped building. A box that has to be
    re-measured by hand every time the surface moves is not a way to put a feature on a
    surface; the loft emits rings of `steps`, so a band of rings over a span of angles is a
    description that survives the surface changing under it.
    """
    import numpy as np
    co = np.array([v.co for v in mesh.verts], float)
    idx = []
    for j in range(j0, j1 + 1):
        for i in range(steps):
            a = 360.0 * i / steps
            if a0 <= a <= a1 and j * steps + i < len(co):
                idx.append(j * steps + i)
    if not idx:
        return None
    p = co[idx]
    c = p.mean(0)
    h = (p.max(0) - p.min(0)) / 2.0 + pad
    return tuple(c), tuple(h)


def _liner_features(prog):
    """Moulded features pressed INTO the liner surface, not placed on top of it.

    `inset(region=True)` + `extrude` puts a boss or a recess on a surface that already
    exists, which is how every pad, shelf and land in a real pocket is made -- and a loft
    can only be given features by rewriting the loft.
    """
    mesh = prog.build()
    for name, (j0, j1, a0, a1), depth, ins in (
            ("drain shelf",  (14, 17, 250, 290), -0.0035, 0.30),
            ("hood rib",     (12, 16,  60, 120),  0.0022, 0.42),
            ("latch land",   (11, 15, 155, 205),  0.0028, 0.28),
            ("flange pad R", ( 1,  3,  40,  70),  0.0018, 0.34),
            ("flange pad L", ( 1,  3, 110, 140),  0.0015, 0.34)):
        box = _ring_box(mesh, j0, j1, a0, a1)
        if box is None:
            print(f"  liner feature '{name}': no rings in that band, skipped")
            continue
        prog = P.emboss(prog, box[0], box[1], depth, inset=ins)
    # PAINT LAST. `inset` and `extrude` CREATE faces, and a created face carries the
    # renderer's default albedo, not the material of the surface it grew out of.
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
                    rib_w=CAP_D * 0.40, rib_h=CAP_D * 0.068,
                    rib_draft=0.82, dome=-0.6 * MM, chamfer=2.1 * MM,
                    flutes=18, flute_depth=0.010, skirt=9 * MM,
                    neck_d=CAP_D * 0.72, bevel=0.040, spin=CAP_SPIN,
                    # the turned groove, a few mm inside the rim
                    groove_r=CAP_D * 0.40, groove_w=2.2 * MM, groove_d=1.3 * MM,
                    printing=False, grip="rib", waist=0.90, rib_dish=-0.04,
                    rib_shoulder=0.66, material=CAP_CAST,
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
    # BEADS, and there have to be a lot of them. The photograph's cap carries fifty-odd
    # droplets of widely varying size; two dozen even ones read as dust. Scattered from a
    # fixed seed so the scene stays reproducible, rejected outside the cap's radius.
    import random as _r
    _rng = _r.Random(2007)
    droplets = []
    while len(droplets) < 58:
        x = _rng.uniform(-1.0, 1.0) * CAP_D * 0.47 * 1e3
        y = _rng.uniform(-1.0, 1.0) * CAP_D * 0.47 * 1e3
        if math.hypot(x, y) > CAP_D * 0.455 * 1e3:
            continue
        droplets.append((x, y, _rng.uniform(0.45, 1.45)))
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


def _hinge_assembly(prog):
    """The two stampings between the pocket and the open door.

    What was here before: three thin bars, 88 x 10 mm, laid across the door's inner face
    in a three-pointed star, plus nothing at all in the gap between pocket and door. In
    the photograph that gap is not empty and it is not spanned by bars. It holds

      * a BAND about as tall as the aperture's radius, cranking out of the pocket to the
        door, with its long edges turned toward the camera, three half-round beads pressed
        across its inner half, and two small raised moulding marks on its face;
      * below it a second, separate plate with a rectangular hole right through it -- the
        bodywork shows through that hole, which is why it is built as four rails and not
        as a dark rectangle painted on a plate.

    Both are placed on the BODY, not on the door: they emerge from inside the pocket, so
    they stay put while the door swings.

    Positions come from Hough segments fitted to the photograph, converted at the
    aperture's own scale (2.010 px/mm, from an ellipse fit to the opening). Measured, in
    millimetres from the aperture centre with +y downward in the image:

        band top edge   (126, -26) -> (185, -13),  sloping 14.9 deg
        band bottom     around y = +86
        plate           x 120..186,  y 80..140,  right edge vertical, base at 9.3 deg

    The image is rolled 8.5 degrees (the aperture fits as an ellipse at that angle), which
    is most of the measured slope: in the panel's own frame the band is very nearly level,
    and the roll is applied here rather than baked into the numbers.
    """
    # Sizes and placements are FITTED to those four numbers per part, by projecting the
    # built part through the renderer's camera and minimising the squared error against
    # them -- not chosen and hoped for. A part standing 20 mm out of the panel projects
    # wider than its own length, so its size cannot be read off the image directly.
    #     band   u 0.84..1.32  v -0.19..0.61   against  0.84..1.33  -0.19..0.60
    #     plate  u 0.82..1.29  v  0.56..1.00   against  0.84..1.30   0.56..0.98
    # which is a mean deviation of 2 mm, at an aperture 272 mm across.
    #
    # The crank, in the band's own frame: out of the pocket floor, up over the rim, and
    # away to meet the door. z is out of the panel, and the rise is quadratic in the
    # station so the band leaves the pocket floor flat and turns as it goes.
    # z0 is POSITIVE: the band's free end lies in FRONT of the pocket's flange, not down
    # inside it. In the photograph the beads are visible curving over the rim, so the band
    # passes across the opening rather than emerging from it, and at z0 = -14 mm the whole
    # left third of the part was correctly placed in projection and completely hidden
    # behind the rim. A silhouette fit cannot see that -- u and v were right either way;
    # only the depth was wrong, and only the picture showed it.
    rise, length, z0 = 34 * MM, 44 * MM, 6 * MM
    path = [(length * t, z0 + rise * t * t) for t in
            (0.0, 0.14, 0.28, 0.42, 0.55, 0.68, 0.80, 0.90, 1.0)]
    # Three beads spaced across the width, running along the first 45 per cent of the
    # length. In the photograph they sit in the band's upper half and stop well short of
    # the door; ribs the other way round -- across the band, spaced along it -- read as a
    # corrugation, which is what the first version of this part rendered.
    # FOUR beads at 28 mm centres, not three at 12-18. Counted and measured on a five-times
    # magnification of the band's left edge, where the silhouette is visibly scalloped --
    # the undulation is what gives the count and the pitch, and it varies ALONG the edge,
    # which is what says the ribs run lengthwise. They are 9 mm across and stand 4 mm proud;
    # at 6.8 x 3.4 they disappeared into the surface at this scale.
    band = P.stamped_strap(path, width=100 * MM, thick=2.6 * MM, flange=5.0 * MM,
                           beads=(-0.42, -0.14, 0.14, 0.42),
                           bead_r=4.5 * MM, bead_h=3.0 * MM,
                           bead_span=(0.0, 0.38),
                           bosses=((0.55, 24 * MM), (0.70, -6 * MM)),
                           boss_r=3.5 * MM, boss_h=0.8 * MM, material=HINGE_BLUE)
    prog = prog.place(obj=band, at=(128 * MM, -24 * MM, 0.0), rotate=(0.0, 0.0, -8.5))
    plate = P.slotted_bracket(w=56 * MM, h=50 * MM, t=2.6 * MM,
                              window=(0.44, 0.20, 0.96, 0.74), hook=14 * MM,
                              hook_t=3.0 * MM, material=HINGE_BLUE)
    prog = prog.place(obj=plate, at=(148 * MM, -106 * MM, 6 * MM),
                      rotate=(0.0, 0.0, -8.5))
    # The band is as wet as the cap is -- the reference has three dozen beads over its
    # face, and they are the only thing on it at the scale between the beads pressed into
    # it and the paint. Seeded on the crank itself so they sit ON the surface rather than
    # at some remembered height above a surface that has since moved.
    import random as _r
    rng = _r.Random(20071)
    for _ in range(26):
        t = rng.uniform(0.06, 0.96)
        wy = rng.uniform(-0.44, 0.44) * 100 * MM
        r = rng.uniform(0.5, 1.5) * MM
        bx = length * t
        bz = z0 + rise * t * t
        # Outward normal of the crank at t, so a bead on the far end does not float.
        slope = 2.0 * rise * t / max(1e-6, length)
        n = math.hypot(1.0, slope)
        nx, nz = -slope / n, 1.0 / n
        drop = (MeshProgram().uv_sphere(segments=12, rings=7, radius=r, mark="water")
                .scale({"by": "all"}, [1.0, 1.0, 0.46])
                .material({"by": "tag", "name": "water"}, **WATER))
        px = 128 * MM + bx + nx * (2.6 * MM + r * 0.35)
        pz = bz + nz * (2.6 * MM + r * 0.35)
        a = math.radians(-8.5)
        prog = prog.place(obj=drop,
                          at=(px * math.cos(a) - (wy - 24 * MM) * math.sin(a),
                              px * math.sin(a) + (wy - 24 * MM) * math.cos(a), pz),
                          rotate=(0.0, math.degrees(math.atan2(-nx, nz)), -8.5))
    return prog


def _door(prog):
    # SIZE AND ANGLE, both read off the photograph and both checked by projecting the
    # built part through the renderer's own camera (`tools/project.py`) rather than by
    # reasoning about it.
    #
    # The reasoning is what kept going wrong. The door's centroid stands 135 mm clear of
    # the panel, so the camera sees it along a direction whose x component has the
    # OPPOSITE SIGN to the one it sees the panel along; computing the view vector at the
    # panel made 105 degrees look 4.9 degrees off edge-on when in fact it was 23, and 23
    # degrees turns a 5 mm bright line into a 106 mm blue lens sitting over the bodywork.
    # Parallax is not a correction for a part this far out, it is the whole effect.
    #
    # Two independent measurements pin it down, and at 90 degrees both land at once:
    #   the door's outer edge is a near-vertical line at 1.40 aperture radii  (model 1.38)
    #   its silhouette is 1.97 aperture radii tall                            (model 1.97)
    door_w, door_h = 232 * MM, 209 * MM
    door_ref = max(door_w, door_h) / 2.0
    door_plan = [r / door_ref for r in _ellipse_plan(door_w / 2, door_h / 2, 96)]
    ring = (MeshProgram()
            .profile([(0.074, 0.0), (0.086, 0.0), (0.086, 0.0022), (0.074, 0.0022)],
                     plane="xz", closed=True)
            .spin(axis="z", steps=64, mark="door")
            .material({"by": "tag", "name": "door"}, **HINGE_BLUE))

    door = P.fuel_door(w=door_w, h=door_h, flange=9 * MM, face=5 * MM,
                       rim=8 * MM, open_deg=DOOR_OPEN_DEG, az=0.0,
                       hinge_r=OPEN_R * DOOR_HINGE_R,
                       gap=3 * MM, steps=96, skin=HINGE_BLUE, liner=DOOR_INNER, strap=False,
                       latch=False, plan=door_plan, inside_material=DOOR_INNER,
                       inner_details=False, inner_parts=ring)
    prog = prog.place(obj=door, at=(14 * MM, -48 * MM, 2.0 * MM))
    prog = _hinge_assembly(prog)


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
                   # A car's flank curves, and the curvature is what makes its lower
                   # half darker than its upper half once the environment has a
                   # ground. At 34 mm over a 1.1 m half-panel this was flat, so the
                   # render had no vertical falloff at all where the photograph has
                   # threefold. Sweeping it: 34 mm gives a lower-body |diff| of 91,
                   # 280 gives 78, and it keeps falling to 66 at 620 -- but 620 mm of
                   # sag is half a barrel, not a car. The metric goes on improving for
                   # a reason that stops being physical, so this stops at a curvature
                   # a Polo actually has and takes the 3/4 of the gain that is real.
                   crown=280 * MM, crown_ax=math.radians(6.0),
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
    # Sized FROM the throat, not written down. The backing has to be wider than whatever
    # the bowl narrows to, and when the throat was widened this disc stayed at 82 mm --
    # so the bowl became wider than its own floor and the render showed a bright crescent
    # of open sky through the gap. The id map named it in one run: those pixels were
    # `<none>`, which is the tracer saying nothing was hit.
    throat = OPEN_R * (1.0 - 0.24) * (1.0 - 0.10) + 14 * MM
    prog = prog.place(obj=_disc(throat, 2 * MM, "well", RUBBER, 96),
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
