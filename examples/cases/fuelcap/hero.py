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
# The model frame is the OPENING'S OWN AXES: +x along its long axis (toward the hinge),
# +y up on the car, +z out of the paint, origin at the cap's centre. Defining it that way
# rather than off the rectification's arbitrary axes is what finally made the numbers below
# agree with each other — see the note on RECESS.
CAP_D = 57.0 * MM              # the assumed scale; everything else is a ratio to it
TILT_DEG = 32.6                # camera off the panel normal, from the cap's ellipse

# The opening. 3.30 by 2.66 cap diameters, and that ratio is the headline of this whole
# exercise: `scene.py` draws 2.25 by 2.0, so every frame the case has ever produced has a
# cap around 40% too big for its pocket.
#
# Three numbers rather than the 36-entry measured table this started with. The table was
# taken about the bounding box of a contour that leaks along the hinge strap, in a frame
# whose rotation came from a superellipse fit whose two axes had been read the wrong way
# round — so it described an egg, and it rendered as one. Re-fitting on the clean sectors
# gives a proper stamped rounded rectangle: exponent 2.90, residual 4.5 mm rms, which is
# 2.4% of the radius and is dominated by the segmentation rather than by any real
# lumpiness. `parts.measured_plan` stays in the kit for outlines that genuinely are not
# superelliptic; this one is.
OPEN_W = 188.1 * MM
OPEN_H = 151.8 * MM
OPEN_N = 2.90
OPENING_REF = max(OPEN_W, OPEN_H) / 2.0

# Which SIDE of the tilt axis the camera is on. A circle projects to the same ellipse from
# either, so the cap's own ellipse cannot say — but the PARALLAX can, see RECESS.
AZIMUTH_DEG = 40.5
# The cap's width as a fraction of the frame, so the render and the photograph crop can be
# framed on the same feature. The cap, not the opening: it is the one thing in both pictures
# whose extent is unambiguous, since the aperture's edge is a threshold and the cap's is an
# annotation.
CAP_FILL = 0.192
# Camera roll about the view axis. `up = +y` leaves the pocket badly rotated against the
# photograph, and that was the largest single difference in the first side-by-side — one
# that no amount of work on the geometry would have touched.
#
# Solved, not tuned: project the model's own outline through this camera and turn it until
# its principal axis matches the photographed contour's. The projection is the renderer's
# own — fwd, right = cross(fwd, up), up2 = cross(right, fwd), image y downward — read off
# `core/src/raytrace.cpp`, so this is arithmetic and not a guess at a convention.
#
# It was briefly "calibrated" instead, by rendering at two rolls and measuring the
# aperture's principal axis in each. That measurement is worthless: the aperture is a
# threshold and it merges with the equally black hinge strap and door, which drags its
# principal axis by a constant 37 degrees. Both render measurements were off by that same
# amount, which is what gave it away.
# Solved against the CAP's projected ellipse, not against the aperture's outline. The
# aperture in the photograph is a threshold that runs out along the equally black hinge
# strap, so its principal axis is biased; the cap's ellipse comes from an annotation polygon
# and is the same measurement on both sides. At this roll the projected cap's major axis is
# 34.22 degrees against the photograph's 34.2, and its minor/major is 0.851 against 0.842 —
# which is also an independent confirmation of the 32.6-degree tilt, since nothing was
# fitted to that ratio.
ROLL_DEG = 19.90

# The cap's face below the paint, and this one is worth the paragraph.
#
# The cap's centre sits 10.1 mm from the opening's in the rectified photograph. Read as
# parallax that gives the recess directly: a feature `d` below the panel appears displaced
# by `d * tan(tilt)` toward the camera's azimuth, so 10.1 mm at 32.6 degrees is 15.8 mm.
#
# In the first, broken frame that same displacement came out along the tilt AXIS rather than
# along the camera's azimuth — which parallax cannot do — and it was written up as an
# off-centre neck boss instead, complete with the observation that the reference bosses do
# sit off-centre. It was a wrong conclusion drawn confidently from a frame that was 90
# degrees out. In the corrected frame the displacement is at 57 degrees against the camera's
# 40.5, so it is parallax, the boss is central, and the DIRECTION of the displacement also
# settles which side of the tilt axis the camera is on — the number the ellipse alone could
# not supply.
RECESS = 15.8 * MM
# Paint to pocket floor, MEASURED rather than argued about, and the argument is worth
# recording because it produced a wrong number twice.
#
# Walk a line out of the cap's centre along the direction of steepest foreshortening (the
# cap ellipse's minor axis, 124.2 degrees in the photograph) and read the luma: paint at
# 226, a step down to the flange at 14-24, a fold, then 61 px of flat 8 — the wall, in
# total darkness — then a slow climb as the floor picks up light. A near-vertical wall of
# depth d projects to d*sin(tilt), and there are 3.002 px/mm along the tilt axis, so 61 px
# is 61 / (3.002 * sin 32.6) = 38 mm.
#
# It was 45, then "a pocket is dark because it is DEEP" pushed it to 64 to stop the floor
# rendering as a lit grey ramp. That was fixing a lighting fault with geometry: the floor
# was bright because the scene was lit by a sun and a blue sky, and the photograph is
# overcast. Once the lighting was corrected the 64 mm remained, and it forces the cap to
# stand on a 36 mm cone — which is the "bowl" the pocket now reads as.
DEPTH = 38.0 * MM

# The cap, read off the rectified close-up. The bar is the number that matters: it is HALF
# the cap's width, where the kit's default is a third, and a bar that width changes the
# whole silhouette — at this scale the cap reads as a bar with two crescents beside it, not
# as a disc with a rib on it.
BAR_W = 0.40                   # bar width / cap diameter, at its base
BAR_H = 0.105                  # bar height / cap diameter
BAR_SHOULDER = 0.48            # where its flat top starts, as a fraction of the half width
BAR_AZ = -19.0                 # bar angle in the panel, degrees, from the photograph
FLUTES = 14
# Shallow. The flutes on this cap do reach the silhouette — you can count the scallops on
# the lower right of the photograph — but at 1.5% of the radius, not at 3%. At 3% the rim
# stops reading as a circle at all and the cap renders as a flower.
FLUTE_DEPTH = 0.016

# The coil. Six and a half turns of roughly 20 mm diameter between the cap's lug and an
# anchor high on the far wall.
COIL_TURNS = 6.5
COIL_R = 9.0 * MM
COIL_WIRE = 2.1 * MM


def _plan(steps):
    return P.rrect_plan(OPEN_W, OPEN_H, OPEN_N, steps, OPENING_REF)


def build(paint=None, cap_material=None, printing=True):
    """The assembly, in the panel frame: paint at z = 0, outward is +z, cap at the origin."""
    paint = paint or M.PAINT_WHITE_TEXTURED
    # ROUGHNESS is what was making the cap read pale. A fuel cap is moulded
    # polypropylene with a matt grain, and at 0.44 the bar's whole flat top works as
    # one broad mirror onto the sky — 150 sRGB where the photograph has 60. The albedo
    # was never the problem; the specular lobe was.
    cap_mat = cap_material or M.mat((0.021, 0.021, 0.023), 0.0, 0.62)
    liner_mat = M.WELL_TEXTURED

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
    prog = exterior_details(prog, paint)

    # 2. the liner: flange, fold, wall, ledge, floor
    prog = prog.place(obj=P.liner(plan, OPENING_REF, depth=DEPTH, flange_w=17.0 * MM,
                                  # STEEP. At wall_k 0.745 over a 66% floor the pocket is a
                                  # funnel, and a funnel is not a light trap: its walls face
                                  # the sky enough to render as a bright gradient where the
                                  # photograph has near-black. A moulded liner is a box with
                                  # just enough draft to leave the tool.
                                  flange_z=2.0 * MM, fold=5.0 * MM, wall_k=0.875,
                                  ledge=7.0 * MM, floor_k=0.78, steps=steps,
                                  material=liner_mat),
                      at=(0.0, 0.0, 0.0))

    # 3. the neck the cap stands on, from the floor up to just under the cap's skirt
    flange = 13.0 * MM
    # A COLLAR, not a pedestal. With the floor at 64 mm this part was a 36 mm cone 68 mm
    # across standing in the middle of the pocket, and it is most of what made the interior
    # read as a smooth bowl instead of a tray. In the photograph the cap stands on a short
    # circular step barely wider than itself.
    prog = prog.place(obj=P.neck_stack(CAP_D * 0.46, -DEPTH, -RECESS - flange + 1.0 * MM,
                                       flare=1.12, material=liner_mat),
                      at=(0.0, 0.0, 0.0))

    # 4. the cap
    prog = prog.place(obj=cap(cap_mat, printing=printing),
                      at=(0.0, 0.0, -RECESS))

    # 5. the coiled tether. It leaves the cap at about four o'clock and runs out to an anchor
    # on the +x wall — read straight off the rectified photograph, where the coil sits
    # between x = +30 and +85 mm and rises about 15 mm across that run.
    lug_a = math.radians(-26.0)
    lug = (CAP_D * 0.50 * math.cos(lug_a), CAP_D * 0.50 * math.sin(lug_a),
           -RECESS - 5.0 * MM)
    anchor = (OPENING_REF * 0.80, 2.0 * MM, -25.0 * MM)
    prog = prog.place(obj=P.coil_cord(lug, anchor, coils=COIL_TURNS, coil_r=COIL_R,
                                      wire_r=COIL_WIRE, up=(0.0, 0.0, 1.0),
                                      material=M.mat((0.022, 0.022, 0.024), 0.0, 0.74)),
                      at=(0.0, 0.0, 0.0))
    prog = prog.place(obj=P.cap_boss(CAP_D / 2.0, spin=-26.0, material=cap_mat),
                      at=(CAP_D * 0.46 * math.cos(lug_a),
                          CAP_D * 0.46 * math.sin(lug_a), -RECESS - 5.5 * MM))

    # 6. the door bumper, on the wall opposite the hinge
    prog = prog.place(obj=P.bump_stop(r=6.5 * MM, h=12.0 * MM),
                      at=(-OPENING_REF * 0.80, -2.0 * MM, -14.0 * MM),
                      rotate=(0.0, 74.0, 0.0))

    # 7. the moulding pips round the flange
    for a in (24.0, 96.0, 152.0, 208.0, 262.0, 318.0):
        k = P.rrect_plan(OPEN_W, OPEN_H, OPEN_N, 360, OPENING_REF)[int(a) % 360]
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


def exterior_details(prog, paint):
    """The two exterior cues that remain inside the photograph-matched crop.

    Without the rear wheel opening and the rear-door shut line the pocket floats on an
    infinite white plate. Both are outside the perception ROI, but they dominate a beauty
    comparison and fix the physical scale of the car around the 188 mm aperture.
    """
    rubber = M.mat((0.007, 0.007, 0.008), 0.0, 0.82)
    wheel_c = (30.0 * MM, -300.0 * MM)
    wheel_r = 150.0 * MM
    disc = (MeshProgram().cylinder(sides=96, radius=wheel_r, height=4.0 * MM,
                                   mark="wheel_opening")
            .material({"by": "tag", "name": "wheel_opening"}, **rubber))
    # The visible upper arc lies on the crowned panel at about z=-7 mm. Keeping the disc
    # there removes the detached blue shadow cast by the first, front-floating version.
    prog = prog.place(obj=disc, at=(wheel_c[0], wheel_c[1], -7.0 * MM))
    # A narrow painted return around the wheel aperture, just enough to catch the same soft
    # highlight as the photographed fender lip.
    ring = (MeshProgram()
            .profile([(wheel_r, 0.0), (wheel_r + 2.5 * MM, 0.0),
                      (wheel_r + 2.5 * MM, -2.0 * MM), (wheel_r, -2.0 * MM)],
                     plane="xz", closed=True)
            .spin(axis="z", steps=96, mark="wheel_lip")
            .material({"by": "tag", "name": "wheel_lip"}, **paint))
    prog = prog.place(obj=ring, at=(wheel_c[0], wheel_c[1], -4.8 * MM))

    # The rear-door seam follows the crowned panel instead of hovering over it as a flat
    # rectangle. It is mostly occluded by the open fuel door, exactly as in the source.
    ca, sa = math.cos(math.radians(78.0)), math.sin(math.radians(78.0))
    def z_of(x, y):
        t = (x * ca + y * sa) / 0.375
        return -0.053 * t * t + 0.9 * MM
    xy = [(160.0, 100.0), (153.0, 86.0), (146.0, 72.0), (139.0, 58.0),
          (132.0, 44.0), (125.0, 30.0), (118.0, 16.0)]
    path = [(x * MM, y * MM, z_of(x * MM, y * MM)) for x, y in xy]
    seam_r = 1.25 * MM
    prof = [(seam_r * math.cos(2 * math.pi * k / 10),
             seam_r * math.sin(2 * math.pi * k / 10)) for k in range(10)]
    seam = (MeshProgram().profile(prof, plane="xy", closed=True)
            .sweep(path, mark="body_seam")
            .material({"by": "tag", "name": "body_seam"},
                      **M.mat((0.004, 0.004, 0.005), 0.0, 0.75)))
    return prog.place(obj=seam, at=(0.0, 0.0, 0.0))


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
    from mirage.decals import ensure_decals
    door_art = ensure_decals(["fuelcap_boyue_door"])["fuelcap_boyue_door"]
    door_inside = M.with_face_decal(
        M.mat((0.016, 0.016, 0.018), 0.0, 0.70), door_art,
        OPENING_REF * 2 * grow, OPENING_REF * 2 * grow,
        -6.0 * MM - 1e-4)
    prog = prog.place(obj=P.fuel_door(w=OPENING_REF * 2 * grow, h=OPENING_REF * 2 * grow,
                                      flange=11.0 * MM, face=6.0 * MM, rim=13.0 * MM,
                                      open_deg=150.0, az=0.0,
                                      hinge_r=OPENING_REF * 1.06, gap=3.0 * MM,
                                      steps=96, skin=paint, liner=liner_mat,
                                      strap=False, plan=list(plan),
                                      inside_material=door_inside, inner_details=True),
                      at=(0.0, 0.0, 2.0 * MM))
    prog = prog.place(obj=P.door_strap(length=125.0 * MM, w=24.0 * MM, t=3.4 * MM,
                                       bend_at=0.38, bend_deg=26.0,
                                       material=M.mat((0.019, 0.019, 0.021), 0.0, 0.72)),
                      # Clear of the cap. It was starting at x = 13 mm and running 90 mm,
                      # so it lay straight across the cap's face and hid the printing; in
                      # the photograph it enters at the pocket's right-hand edge and never
                      # crosses the cap at all.
                      at=(50.0 * MM, 24.0 * MM, -8.0 * MM),
                      rotate=(0.0, 0.0, -4.0))
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
