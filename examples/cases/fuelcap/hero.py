"""ONE filler region, copied from ONE photograph, to see how close the kit can actually get.

`scene.py` draws a randomised pocket from measured *distributions*, and that is the right
shape for a dataset. It is the wrong shape for answering "does this look like the thing?",
because a distribution has no answer to compare against — every sample is defensible and
none of them is the picture. So this module drops the randomisation entirely and rebuilds a
single named car, with every number in it taken off that car's own photograph:

    _ref/bycar/博越L/粗筛done2_博越L_2023款15T豪华型_29.png

**How the numbers were got.** The cap is a circle of known diameter lying in the body panel,
so under weak perspective it images as an ellipse whose major axis is unforeshortened and
whose minor axis is that same diameter times cos(tilt). Fitting that ellipse to the cap's
own annotation polygon fixes the panel's orientation (32.6 degrees off the camera) and the
pixel-per-millimetre scale, with no camera calibration and no assumptions beyond the cap
being round. After that the photograph can be *rectified into the panel plane* and every
distance in it read off in millimetres — which is where the outline, the flange width, the
bar's width and the coil's pitch below all come from. `_out/_ref_analysis/` holds the
rectified frame with its millimetre grid.

**What the exercise found, before any of it was modelled.** The opening is 186 x 166 mm and
the cap is 57 mm, so the pocket is **3.3 x 2.9 cap diameters**. `scene.py` draws 2.25 x 2.0.
Checked across the 26 reference cars whose aperture can be segmented from the body, the real
figure is a median of 2.9 *by height alone* — the direction that cannot be inflated by an
open door leaking into the measurement — against the 2.0 the case has always drawn. Every
frame this case has ever produced has a cap around 40% too big for the pocket it sits in,
and that single ratio is most of what makes the synthetic row not look like the real one:
the eye reads a knob filling a socket where the photograph has a small cap at the bottom of
a large box.

**The second finding is the depth.** The cap's centre sits 6 mm off the opening's centre in
the photograph. That is not an asymmetric pocket — it is parallax: a cap recessed by `d`
below the panel appears displaced by `d * tan(32.6 deg)` toward the camera's azimuth, so
6 mm of displacement means the cap's face is 9-10 mm below the paint. The pocket floor is
another 35 mm below *that*. So the cap is not sitting on the floor of a shallow dish, and it
is not sitting on the floor of a deep box either: it is standing near the top of a deep box
on the filler neck. Hence `parts.neck_stack`.

Run it with `python -m fuelcap.sheet hero`.
"""
from __future__ import annotations

import math

from mirage.meshlang import MeshProgram

from . import materials as M
from . import parts as P

MM = 1e-3

# --------------------------------------------------------------------------- #
# what the photograph says, in millimetres
# --------------------------------------------------------------------------- #
CAP_D = 57.0 * MM              # the assumed scale; everything else is a ratio to it
TILT_DEG = 32.6                # camera off the panel normal, from the cap's ellipse
# Which SIDE of the tilt axis the camera is on. A circle projects to the same ellipse from
# either, so this is the one number the photograph cannot supply and it was settled by
# rendering both and looking: at 235.8 the cap's printing arrives upside down against the
# photograph's, at 55.8 it reads the same way round.
AZIMUTH_DEG = 55.8
# The cap's width as a fraction of the frame, so the render and the photograph crop can be
# framed on the same feature. The cap, not the opening: it is the one thing in both pictures
# whose extent is unambiguous, since the aperture's edge is a threshold and the cap's is an
# annotation.
CAP_FILL = 0.192
# Camera roll about the view axis, and it is worth saying how this number was got, because
# the obvious way is wrong twice over. `up = +y` leaves the pocket 60-odd degrees rotated
# against the photograph. Projecting the model's outline and solving for the roll gets the
# SHAPE right — the projected elongation is 1.16 against the photographed contour's 1.22 —
# but lands 35 degrees out, because the renderer's image basis is not the one that
# projection assumed. So the last step is a calibration rather than a derivation: render at
# two rolls, measure the aperture's principal axis in each, and interpolate. -1.21 degrees
# of image per degree of roll, and the photograph wants +2.4.
ROLL_DEG = 58.0
RECESS = 9.4 * MM              # cap face below the paint, from the parallax above
DEPTH = 45.0 * MM              # paint to pocket floor
OPENING_REF = 96.5 * MM        # the radius the opening's plan is authored against

# The opening's outline, measured: radius divided by OPENING_REF every 10 degrees counter-
# clockwise from +x (+x is toward the hinge, +y is up on the car). Taken off the rectified
# photograph by thresholding the aperture out of the body paint. Between 10 and 70 degrees
# the aperture's dark region runs straight out along the equally dark hinge strap, so the
# contour there is the strap's silhouette and not the opening's; that sector alone is a
# smooth blend across its neighbours, and it is the only part of this table that is not
# measured.
OPENING_PLAN = (
    0.9512, 0.9466, 0.9419, 0.9373, 0.9326, 0.9280,
    0.9234, 0.9187, 0.9141, 0.8488, 0.8119, 0.7979,
    0.8059, 0.8379, 0.8921, 0.9523, 0.9921, 1.0000,
    0.9786, 0.9445, 0.9098, 0.8866, 0.8849, 0.9053,
    0.9465, 0.9826, 0.9497, 0.8731, 0.8089, 0.7686,
    0.7535, 0.7589, 0.7857, 0.8344, 0.8918, 0.9343,
)

# The cap, read off the rectified close-up. The bar is the number that matters: it is HALF
# the cap's width, where the kit's default is a third, and a bar that width changes the
# whole silhouette — at this scale the cap reads as a bar with two crescents beside it, not
# as a disc with a rib on it.
BAR_W = 0.49                   # bar width / cap diameter, at its base
BAR_H = 0.105                  # bar height / cap diameter
BAR_SHOULDER = 0.56            # where its flat top starts, as a fraction of the half width
BAR_AZ = -19.0                 # bar angle in the panel, degrees, from the photograph
FLUTES = 14
# Shallow. The flutes on this cap do reach the silhouette — you can count the scallops on
# the lower right of the photograph — but at 1.5% of the radius, not at 3%. At 3% the rim
# stops reading as a circle at all and the cap renders as a flower.
FLUTE_DEPTH = 0.016

# The coil. Six and a half turns of roughly 20 mm diameter between the cap's lug and an
# anchor high on the far wall.
COIL_TURNS = 6.5
COIL_R = 10.0 * MM
COIL_WIRE = 2.1 * MM


def _plan(steps):
    return P.measured_plan(OPENING_PLAN, steps)


def build(paint=None, cap_material=None, printing=True):
    """The assembly, in the panel frame: paint at z = 0, outward is +z, cap at the origin."""
    paint = paint or M.PAINTS["white"]
    cap_mat = cap_material or M.mat((0.021, 0.021, 0.023), 0.0, 0.44)
    liner_mat = M.mat((0.0165, 0.0165, 0.018), 0.0, 0.62)

    steps = 96
    plan = _plan(steps)

    # 1. the body panel, its aperture cut to the measured outline
    # Big enough to fill the frame at this obliquity, and crowned like a car's flank rather
    # than flat. `crown` is the sag at the panel's own EDGE, so it scales with the square of
    # the panel size for a fixed curvature — a value carried over from a smaller panel is a
    # different car.
    prog = P.panel(size=0.75, thick=0.010, ring=steps, material=paint,
                   crown=0.053, crown_ax=math.radians(78.0),
                   hole_plan=[OPENING_REF * k for k in plan])

    # 2. the liner: flange, fold, wall, ledge, floor
    prog = prog.place(obj=P.liner(plan, OPENING_REF, depth=DEPTH, flange_w=11.0 * MM,
                                  flange_z=2.0 * MM, fold=5.0 * MM, wall_k=0.745,
                                  ledge=8.0 * MM, floor_k=0.66, steps=steps,
                                  material=liner_mat),
                      at=(0.0, 0.0, 0.0))

    # 3. the neck the cap stands on, from the floor up to just under the cap's skirt
    flange = 13.0 * MM
    prog = prog.place(obj=P.neck_stack(CAP_D * 0.46, -DEPTH, -RECESS - flange + 1.0 * MM,
                                       flare=1.62, material=liner_mat),
                      at=(0.0, 0.0, 0.0))

    # 4. the cap
    prog = prog.place(obj=cap(cap_mat, printing=printing), at=(0.0, 0.0, -RECESS))

    # 5. the coiled tether. It leaves the cap at about four o'clock and runs out to an anchor
    # on the +x wall — read straight off the rectified photograph, where the coil sits
    # between x = +30 and +85 mm and rises about 15 mm across that run.
    lug_a = math.radians(-26.0)
    lug = (CAP_D * 0.50 * math.cos(lug_a), CAP_D * 0.50 * math.sin(lug_a), -RECESS - 5.0 * MM)
    anchor = (OPENING_REF * 0.86, -1.0 * MM, -23.0 * MM)
    prog = prog.place(obj=P.coil_cord(lug, anchor, coils=COIL_TURNS, coil_r=COIL_R,
                                      wire_r=COIL_WIRE, up=(0.0, 0.0, 1.0),
                                      material=M.mat((0.024, 0.024, 0.026), 0.0, 0.55)),
                      at=(0.0, 0.0, 0.0))
    prog = prog.place(obj=P.cap_boss(CAP_D / 2.0, spin=-26.0, material=cap_mat),
                      at=(CAP_D * 0.46 * math.cos(lug_a), CAP_D * 0.46 * math.sin(lug_a),
                          -RECESS - 5.5 * MM))

    # 6. the door bumper, on the wall opposite the hinge
    prog = prog.place(obj=P.bump_stop(r=6.5 * MM, h=12.0 * MM),
                      at=(-OPENING_REF * 0.80, -2.0 * MM, -14.0 * MM),
                      rotate=(0.0, 74.0, 0.0))

    # 7. the moulding pips round the flange
    for a in (24.0, 96.0, 152.0, 208.0, 262.0, 318.0):
        k = P.measured_plan(OPENING_PLAN, 360)[int(a) % 360]
        r = OPENING_REF * k - 5.5 * MM
        prog = prog.place(obj=P.pip(r=2.2 * MM, h=0.9 * MM, material=liner_mat),
                          at=(r * math.cos(math.radians(a)), r * math.sin(math.radians(a)),
                              -2.0 * MM))

    # 8. the door, swung open on the +x side, and the strap that carries it
    prog = door(prog, plan, liner_mat, paint)
    return prog


def cap(material, printing=True):
    """The photographed cap: 57 mm, fourteen flutes, and a bar half its own width across it."""
    d = CAP_D
    return P.cap(d=d, flange=13.0 * MM, rib_len=d * 1.00, rib_w=d * BAR_W,
                 rib_h=d * BAR_H, rib_draft=0.80, dome=-0.4 * MM, chamfer=2.0 * MM,
                 flutes=FLUTES, flute_depth=FLUTE_DEPTH, skirt=18.0 * MM,
                 neck_d=d * 0.72, bevel=0.048, spin=BAR_AZ, printing=printing,
                 grip="rib", waist=0.86, rib_dish=-0.06, rib_shoulder=BAR_SHOULDER,
                 material=material, steps=72, decal="fuelcap_boyue")


def door(prog, plan, liner_mat, paint):
    """The door, swung open about the +x edge, and the stamped strap between the two.

    Swung 112 degrees rather than flat back: on the photograph the door's inner face is
    turned toward the camera, which is what puts its dark liner — and not its painted outer
    skin — across the right-hand third of the picture. Its outline is the OPENING's measured
    plan scaled up, not a separately described rounded rectangle, so the two agree at the
    shut line by construction.
    """
    # `plan` is a multiplier on the door's own reference radius, so the door is grown by
    # growing w and h — multiplying the plan as well applies the growth twice, which is a
    # door 11% oversize and was the first thing the render showed.
    grow = 1.055
    prog = prog.place(obj=P.fuel_door(w=OPENING_REF * 2 * grow, h=OPENING_REF * 2 * grow,
                                      flange=11.0 * MM, face=6.0 * MM, rim=13.0 * MM,
                                      open_deg=120.0, az=0.0,
                                      hinge_r=OPENING_REF * 1.06, gap=3.0 * MM,
                                      steps=96, skin=paint, liner=liner_mat,
                                      strap=False, plan=list(plan)),
                      at=(0.0, 0.0, 2.0 * MM))
    prog = prog.place(obj=P.door_strap(length=90.0 * MM, w=22.0 * MM, t=3.4 * MM,
                                       bend_at=0.38, bend_deg=26.0, material=liner_mat),
                      at=(OPENING_REF * 0.16, 30.0 * MM, -7.0 * MM),
                      rotate=(0.0, 0.0, -6.0))
    return prog


def pose(dist=0.52, azimuth_deg=None, tilt_deg=TILT_DEG, roll_deg=None):
    """Camera at the photograph's own obliquity.

    `azimuth_deg` is where the camera sits round the panel normal. The ellipse fit gives the
    TILT AXIS but not which side of it the camera is on — a circle projects to the same
    ellipse from either — so the azimuth is the one number here that a single photograph
    cannot supply, and it is resolved by rendering the four candidates and looking.
    """
    az = math.radians(AZIMUTH_DEG if azimuth_deg is None else azimuth_deg)
    t = math.radians(tilt_deg)
    e = (math.sin(t) * math.cos(az), math.sin(t) * math.sin(az), math.cos(t))
    eye = tuple(dist * x for x in e)
    target = (0.0, 0.0, -RECESS * 0.5)
    # ROLL, and it is not a detail. `up = +y` leaves the pocket in frame 25 degrees rotated
    # against the photograph — the largest single difference in the first side-by-side, and
    # one that no amount of work on the geometry would have touched.
    #
    # It is not the plan's phase either, which was the first guess: projecting the model's
    # own outline through this camera and turning it shows the projected principal axis
    # never reaches the photograph's for ANY phase, because the foreshortening direction is
    # fixed by the camera and not by the part. So the camera is what has to turn.
    #
    # `up` is +y rotated about the view axis by ROLL_DEG, solved by projecting the outline
    # rather than by rendering and eyeballing: elongation 1.16 against the photographed
    # contour's 1.22, principal axis matched to under a degree.
    d = [target[k] - eye[k] for k in range(3)]
    n = math.sqrt(sum(x * x for x in d))
    d = [x / n for x in d]
    a = math.radians(ROLL_DEG if roll_deg is None else roll_deg)
    u = [0.0, 1.0, 0.0]
    ca, sa = math.cos(a), math.sin(a)
    cr = [d[1] * u[2] - d[2] * u[1], d[2] * u[0] - d[0] * u[2], d[0] * u[1] - d[1] * u[0]]
    dot = sum(d[k] * u[k] for k in range(3))
    up = [u[k] * ca + cr[k] * sa + d[k] * dot * (1.0 - ca) for k in range(3)]
    return dict(eye=eye, target=target, up=tuple(up))
