"""One randomised fuel-filler pocket, plus the ground truth about it.

`sample()` draws a variant; `build()` turns it into an op-log and the labels that go with
it. The two are separate on purpose: the labels are computed from the *parameters*, in
closed form, never read back off the render. A pose label recovered from a depth map is a
measurement of the renderer, and a network trained on it learns the renderer's error.

The randomisation is aimed at the distributions `fit.py` measures off the real capture,
not at "lots of variety". Three sets were measured:

    set        distance m     obliquity deg    cap px     depth noise mm
    prod       0.43 – 0.46     3 – 13          57 –  74   0.32
    orbbec     0.29 – 0.62     4 – 39          44 – 124   0.44
    depthcap   1.12 – 1.29    29 – 53          41 –  51   0.42

`prod` is the deployment domain — a robot arm that comes in square-on at 45 cm — and it is
the narrowest of the three. Training only on it would be the obvious move and the wrong
one: the model has to survive the pose it is *handed*, not the pose it was promised, so the
default camera spans the union of all three and then some.
"""
from __future__ import annotations

import math

import numpy as np

from mirage.meshlang import MeshProgram

from . import materials as M
from . import parts as P

# where the measured section puts the paint, relative to the cap face
MEASURED_BODY_MM = 10.5

# The two real cameras in this dataset are both Orbbec Gemini 330: 640x480 at fx=fy=367.4
# and 1280x800 at fx=fy=611.5, principal point centred to within half a pixel in both. Same
# lens, so the same 66.4-degree vertical field — which is the only intrinsic a pinhole
# renderer needs, and why nothing here has to fight a shifted principal point.
CAMERAS = {
    "orbbec640": dict(w=640, h=480, fx=367.4, fy=367.4, cx=321.9, cy=241.1),
    "prod1280": dict(w=1280, h=800, fx=611.4, fy=611.5, cx=644.8, cy=403.0),
}

# Face tags that ARE the cap, in the order the id AOV numbers them. Ids 1-2 are the cap
# body and its grip rib; the rib's own id is what gives the knob OBB for free, which is
# otherwise a separate detection model in the real pipeline.
CAP_TAGS = ("cap_body", "cap_rib", "cap_teeth", "cap_slot")
ID_TAGS = CAP_TAGS + ("seal", "well", "neck", "panel", "door", "tether")
RIB_ID = ID_TAGS.index("cap_rib") + 1


def _lerp(rng, lo, hi):
    return float(rng.uniform(lo, hi))


def sample(rng, camera="orbbec640", domain="wide"):
    """Draw one pocket. Returns a plain dict — the whole variant, nothing hidden."""
    d_cap = _lerp(rng, 0.068, 0.086)
    # The rib scales with the cap rather than varying freely: across ninety cars the grip
    # is always most of the diameter and about half of that across, and sampling the two
    # independently makes caps no manufacturer has ever made.
    #
    # These ratios are not guesses and were not right the first time. Measured per-frame
    # through `fit.py --compare`, the real grip is 59.5 x 46.7 mm on a 73.6 mm disc — it
    # covers most of the cap face. The first draft made it 53 x 30: a narrow bar across a
    # wide disc, which is what this part looks like in memory and not what it looks like in
    # 148 measured frames.
    #     rib_len / disc  0.81      rib_wid / rib_len  0.79      rib height  17.7 mm
    rib_len = d_cap * _lerp(rng, 0.78, 0.95)
    rib_w = rib_len * _lerp(rng, 0.55, 0.78)
    v = dict(
        camera=camera,
        d_cap=d_cap, flange=_lerp(rng, 0.007, 0.013),
        rib_len=rib_len, rib_w=rib_w,
        rib_h=_lerp(rng, 0.011, 0.019), rib_draft=_lerp(rng, 0.68, 0.90),
        rib_slot=float(rng.random() < 0.5) * _lerp(rng, 0.002, 0.006),
        dome=_lerp(rng, 0.0, 0.0018), chamfer=_lerp(rng, 0.0015, 0.0035),
        # Grip family, from the reference sheet: about a quarter of the ninety cars have a
        # sunk rectangular well rather than a raised bar, and many raised bars carry a
        # groove down the spine. Teeth are on now by default and on the OUTER rim, where
        # they are actually visible.
        grip=str(rng.choice(["rib", "rib", "rib", "slot"])),
        teeth=int(rng.choice([0, 30, 36, 42, 48])), skirt=_lerp(rng, 0.014, 0.028),
        # The pocket. The aperture is only a little wider than the cap — on every one of
        # the ninety reference cars the cap very nearly fills its hole. Allowing it to be
        # 48 mm wider, as the first draft did, turns the recess into a funnel whose sloped
        # wall faces the sky and blows out to near-white, which is the opposite of the
        # light trap a real pocket is.

        # Depth from the DISH FLOOR to the cap face. Tiny, and it has to be: measured on
        # 349 real frames, the body surface stands a median of 8 mm proud of the cap face
        # (p10 -13, p90 17) — the whole pocket, dish and recess together, is about a
        # centimetre deep. This was 22-65 mm, which with the dish on top put the model at
        # 42 mm, five times too deep, and is most of why the renders read as a box hung on
        # a wall rather than a filler let into a wing. See fit.recess_depth.

        neck=bool(rng.random() < 0.75),
        well_metal=bool(rng.random() < 0.25),
        # the furniture down the recess, and the shape of the aperture — both are what
        # `fit.complexity` says is missing between 6 and 24 mm
        well_ribs=int(rng.choice([0, 3, 4, 4, 5, 6])),
        well_drain=bool(rng.random() < 0.7),
        # pocket hardware — screws, the door catch, a rubber grommet. All of it is in the
        # reference photographs and none of it was in the model.
        n_screws=int(rng.choice([0, 1, 2, 2, 3])),
        n_catches=int(rng.choice([0, 1, 1, 2])),
        n_grommets=int(rng.choice([0, 0, 1, 1, 2])),
        squareness=float(rng.choice([1.0, 1.0, 2.6, 3.4, 4.2, 5.0])),
        # the BODY, which is most of the ROI and was a flat rectangle
        crown=_lerp(rng, 0.006, 0.030), crown_ax=_lerp(rng, 0.0, math.pi),
        # the outer dish the door lies in — the structure that actually falls inside the ROI
        # The generalisation of the measured section: scale it radially and in depth.
        # Imitation is r_gain = z_gain = 1.
        r_gain=_lerp(rng, 0.88, 1.14), z_gain=_lerp(rng, 0.75, 1.30),
        # The filler axis really is about 14 degrees off the body normal (measured over 197
        # frames), and this is deliberately NOT that — it is 0 to 5.
        #
        # The reason is a geometric incompatibility, not a preference. `pocket` is a lathe
        # about the filler axis, so its mouth is a FLAT ring perpendicular to that axis.
        # Tilting the body panel relative to it opens a wedge between the two: at 13.5
        # degrees and a 90 mm radius the panel stands 20 mm proud of the pocket rim on one
        # side and 20 mm below it on the other, and the camera sees straight through the
        # gap. Measured, that dragged recess_depth from +8 mm to about 0 and cost several
        # attempts at other explanations.
        #
        # Getting the full 14 degrees needs the pocket's mouth to be cut OBLIQUELY — a
        # per-direction z offset on the lathe, which `spin` cannot express (its plan
        # scales radius only). Until then the honest choice is a small tilt that keeps the
        # surfaces closed, and this comment rather than a number that looks right.
        body_tilt=_lerp(rng, 0.0, 5.0), body_tilt_az=_lerp(rng, 0.0, 360.0),
        # The measured section is binned by physical radius, so it ALREADY averages over
        # whatever squareness the real pocket has. Re-applying a strong plan on top of it
        # applies that squareness twice and drags the outer surface down. Kept mild.
        pan_sq=float(rng.choice([1.0, 1.6, 2.0, 2.4, 3.0])),
        seam=bool(rng.random() < 0.75), seam_gap=_lerp(rng, 0.0035, 0.0075),
        seam_step=_lerp(rng, 0.002, 0.005), seam_side=float(rng.choice([-1.0, 1.0])),
        door_rim=_lerp(rng, 0.004, 0.009), door_ribs=int(rng.choice([0, 2, 3, 3, 4])),
        # The hinge strap — the brightest thing in an open pocket after the cap, and it
        # was two little blocks. Plus the chromed trim ring a minority of cars have.
        arm_len=_lerp(rng, 0.055, 0.095), arm_w=_lerp(rng, 0.014, 0.026),
        arm_holes=int(rng.choice([0, 1, 2, 2, 3])),
        trim=bool(rng.random() < 0.18), trim_w=_lerp(rng, 0.010, 0.020),
        # the filler neck is not square to the body panel on any real car
        tilt_x=_lerp(rng, -14.0, 14.0), tilt_y=_lerp(rng, -16.0, 10.0),
        cap_spin=_lerp(rng, 0.0, 360.0),        # a screw cap stops wherever it stops
        # the door and the paint
        # A fuel door is only a little bigger than the hole it covers — it is not a hatch.
        # Sized at 180-205 mm it stands out 200 mm from a panel the camera is 400 mm from
        # and takes over the frame, which is what the first draft did.
        # Opened wide, and THIN. A fuel door is a pressing a few millimetres deep, not a
        # box: at an 18 mm return flange it renders as a tray held out toward the camera.
        # And a real one is usually swung well back toward the wing rather than standing
        # square, so it presents its edge, not its face.
        door_open=_lerp(rng, 95.0, 165.0),
        door_w=_lerp(rng, 0.115, 0.160), door_h=_lerp(rng, 0.115, 0.165),
        paint=str(rng.choice(M.PAINT_NAMES)),
        tether=bool(rng.random() < 0.7), tether_az=_lerp(rng, 0.0, 360.0),
        tether_r=_lerp(rng, 0.0016, 0.0028), tether_kinks=int(rng.choice([1, 2, 2, 3])),
        # About one in ten of the ninety reference cars has a metal cap, and a handful
        # have a petal-edged rim rather than a round one — a plan, not a section, so the
        # generalised lathe does it in one line.
        alu=bool(rng.random() < 0.11),
        lobes=int(rng.choice([0, 0, 0, 0, 0, 6, 8, 10, 12])),
        lobe_depth=_lerp(rng, 0.035, 0.085),
        # Lighting. Bounded well below what the tracer will happily accept: the subject is
        # a black object inside a shadowed hole, and the exposure that makes IT readable is
        # one that blows the surrounding paint. Real frames of this scene are exactly that
        # — a correctly exposed pocket and a hot panel — so the range is set by the pocket.
        sun=_lerp(rng, 0.4, 2.6), env=_lerp(rng, 0.30, 1.20),
        sun_az=_lerp(rng, 0.0, 360.0), sun_el=_lerp(rng, 12.0, 82.0),
        sky_flat=float(rng.random() < 0.55),
        exposure=float(math.exp(rng.normal(0.0, 0.22))),
        sky_tint=[_lerp(rng, 0.92, 1.20), _lerp(rng, 0.95, 1.08), _lerp(rng, 0.82, 1.12)],
    )
    # camera placement, in the CAP's frame: obliquity off its normal, azimuth around it.
    # Obliquity is drawn from a SKEWED distribution, not a uniform one. Whoever is holding
    # the camera is trying to look at the cap, so square-on is the mode and the tail is
    # long: the real sets have a median of 8 degrees (prod) and 17 (orbbec) against ranges
    # reaching 13 and 39. Sampling uniformly over the range puts the median at the middle
    # of it and over-represents the awkward angles by a third.
    skew = float(rng.beta(1.6, 2.4))
    if domain == "prod":
        v.update(dist=_lerp(rng, 0.41, 0.48), obliq=skew * 16.0)
    elif domain == "orbbec":
        v.update(dist=_lerp(rng, 0.28, 0.64), obliq=skew * 45.0)
    else:                                        # the union, widened at both ends
        v.update(dist=float(np.exp(rng.uniform(math.log(0.26), math.log(1.35)))),
                 obliq=skew * 60.0)
    v.update(azim=_lerp(rng, 0.0, 360.0), roll=_lerp(rng, -14.0, 14.0),
             # the cap is not centred in a real frame; the arm aims at the pocket, not the cap
             aim_off=[_lerp(rng, -0.022, 0.022), _lerp(rng, -0.022, 0.022)])
    return v


# --------------------------------------------------------------------------- #
# geometry helpers
# --------------------------------------------------------------------------- #
def _compose(tilt, extra_deg, azim_deg):
    """`tilt` followed by an extra rotation of `extra_deg` about an in-plane axis at
    `azim_deg` — returned as the XYZ euler triple `place` takes.

    Needed because the body is tilted relative to the filler, not relative to the world:
    composing the two as separate euler angles would tilt about a world axis and slide the
    panel sideways as well."""
    R = _rot_xyz(*tilt)
    a = math.radians(azim_deg)
    k = R @ np.array([math.cos(a), math.sin(a), 0.0])       # the hinge, in the tilted plane
    t = math.radians(extra_deg)
    K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    Rx = np.eye(3) + math.sin(t) * K + (1 - math.cos(t)) * (K @ K)
    return _euler_xyz(Rx @ R)


def _euler_xyz(R):
    """Rz@Ry@Rx back to (rx, ry, rz) degrees — the convention `place` composes in."""
    sy = -R[2, 0]
    if abs(sy) < 0.999999:
        ry = math.asin(sy)
        rx = math.atan2(R[2, 1], R[2, 2])
        rz = math.atan2(R[1, 0], R[0, 0])
    else:
        ry = math.copysign(math.pi / 2, sy)
        rx = math.atan2(-R[1, 2], R[1, 1])
        rz = 0.0
    return (math.degrees(rx), math.degrees(ry), math.degrees(rz))


def _rot_xyz(rx, ry, rz):
    """The renderer's own place() convention: degrees, applied X then Y then Z."""
    cx, sx = math.cos(math.radians(rx)), math.sin(math.radians(rx))
    cy, sy = math.cos(math.radians(ry)), math.sin(math.radians(ry))
    cz, sz = math.cos(math.radians(rz)), math.sin(math.radians(rz))
    Rx = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]])
    Ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
    Rz = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]])
    return Rz @ Ry @ Rx


def build(v):
    """The op-log for one variant, plus everything true about it.

    Returns ``(program, gt)``. ``gt`` carries the cap's frame in WORLD coordinates and the
    camera that looks at it; `dataset.py` turns that pair into the camera-frame labels the
    real annotation files use."""
    rng = np.random.default_rng(abs(hash((v["paint"], round(v["d_cap"], 6)))) % (2 ** 31))
    paint = M.jitter(M.PAINTS[v["paint"]], rng, dc=0.10, dr=0.08)
    cap_mat = (M.jitter(M.CAP_METAL_FAMILY[int(rng.integers(0, len(M.CAP_METAL_FAMILY)))],
                        rng, dc=0.12, dr=0.14) if v["alu"]
               else M.jitter(M.CAP_FAMILY[int(rng.integers(0, len(M.CAP_FAMILY)))], rng))
    well_mat = M.jitter(M.WELL_METAL if v["well_metal"] else M.WELL_PLASTIC, rng, dc=0.22)

    # The CAP FACE is the world origin, and everything is placed relative to it. That is
    # not a convenience: the measured section is a height-above-the-cap-face table, so
    # putting the cap anywhere else means converting on every use. It also makes the pose
    # label trivially exact — the thing being labelled is at the origin by construction.
    #
    # `tilt` then orients the whole assembly, pocket and body together. The measured
    # section already averages over whatever angle the real body makes with the filler
    # axis, so it is self-consistent to treat them as square to each other here.
    tilt = (v["tilt_x"], v["tilt_y"], 0.0)
    R = _rot_xyz(*tilt)
    axis = R @ np.array([0.0, 0.0, 1.0])
    cap_c = np.zeros(3)

    prog = MeshProgram()
    # Panel first: it is the biggest thing and it is what everything else is a hole in.
    # Its size follows the camera distance so it always overfills the frame — a fixed
    # panel leaves the car floating in a square of sky, and every pixel of that sky is a
    # pixel of background a real frame would have had car in.
    panel_size = max(0.55, 3.0 * v["dist"])
    # ONE lathe from the measured section carries the whole pocket: the trench around the
    # cap, its outer wall, the dish the door lies in, and the run out onto the body. The
    # three hand-built parts this replaces could not reach that shape between them — see
    # parts.MEASURED_SECTION_MM.
    pocket_r = 0.100 * v["r_gain"]
    prog = prog.place(obj=P.pocket_shaped(cap_r=v["d_cap"] / 2.0, r_gain=v["r_gain"],
                                          z_gain=v["z_gain"], neck_r=v["d_cap"] * 0.34,
                                          neck_len=0.050, squareness=v["pan_sq"],
                                          blend_from=0.085 * v["r_gain"],
                                          material=well_mat),
                      at=tuple(cap_c), rotate=tilt)
    # The body panel meets the pocket at its outer radius, and sits where the measured
    # section says the paint is: 10.5 mm above the cap face, scaled with z_gain.
    body_z = MEASURED_BODY_MM * 1e-3 * v["z_gain"]
    bt, baz = math.radians(v["body_tilt"]), math.radians(v["body_tilt_az"])
    # the body is tilted about an axis in its own plane, at azimuth `baz`
    body_rot = _compose(tilt, v["body_tilt"], v["body_tilt_az"])
    prog = prog.place(obj=P.panel(size=panel_size, hole_d=2 * pocket_r * 0.995,
                                  thick=0.009, material=paint,
                                  squareness=v["pan_sq"], crown=v["crown"],
                                  crown_ax=v["crown_ax"],
                                  hole_stretch=1.0 / max(math.cos(bt), 0.35),
                                  # The panel hinges ABOUT the axis at `baz`, so the
                                  # foreshortening — and therefore the ellipse — runs
                                  # PERPENDICULAR to it. Stretching along `baz` instead
                                  # leaves the hole misaligned with the pocket by the full
                                  # tilt, and the panel then slices across the recess: in
                                  # an id map, the well comes out as a crescent.
                                  hole_ax=baz + math.pi / 2),
                      at=tuple(cap_c + axis * body_z), rotate=body_rot)
    if v["trim"]:
        prog = prog.place(obj=P.trim_ring(r_in=pocket_r * 0.62,
                                          r_out=pocket_r * 0.62 + v["trim_w"],
                                          screws=int(rng.integers(4, 8))),
                          at=tuple(cap_c + axis * (MEASURED_BODY_MM * 1e-3 * v["z_gain"]
                                                   * 0.55)), rotate=tilt)
    if v["seam"]:
        sx = v["seam_side"] * (pocket_r + v["door_w"] + 0.040)
        sx = max(-panel_size / 2 + 0.03, min(sx, panel_size / 2 - 0.03))
        prog = prog.place(obj=P.shutline(panel_size, sx, gap=v["seam_gap"],
                                         step=v["seam_step"], side=v["seam_side"],
                                         material=paint),
                          at=tuple(cap_c + axis * body_z), rotate=body_rot)
    if v["well_ribs"] or v["well_drain"] or v["n_screws"] or v["n_catches"]:
        # the ribs and drain live down in the trench, whose floor the section puts at
        # roughly -33 mm scaled
        trench = 0.0326 * v["z_gain"]
        prog = prog.place(obj=P.well_details(pocket_r * 0.46, v["d_cap"] + 0.014, trench,
                                             ribs=v["well_ribs"], drain=v["well_drain"],
                                             screws=v["n_screws"], catches=v["n_catches"],
                                             grommets=v["n_grommets"],
                                             seed=int(v["cap_spin"] * 13) % 9999,
                                             material=well_mat),
                          at=tuple(cap_c), rotate=tilt)
    cap_prog = P.cap(d=v["d_cap"], flange=v["flange"], rib_len=v["rib_len"], rib_w=v["rib_w"],
                     rib_h=v["rib_h"], rib_draft=v["rib_draft"], rib_slot=v["rib_slot"],
                     dome=v["dome"], chamfer=v["chamfer"], teeth=v["teeth"],
                     skirt=v["skirt"], neck_d=v["d_cap"] * 0.62, spin=v["cap_spin"],
                     grip=v["grip"], lobes=v["lobes"], lobe_depth=v["lobe_depth"],
                     printing=not v["alu"], material=cap_mat)
    # rotate=tilt ONLY. The spin is already baked into the cap about its own axis; passing
    # it here as well would swing the axis instead of turning the cap (see parts.cap).
    prog = prog.place(obj=cap_prog, at=tuple(cap_c), rotate=tilt)
    prog = prog.place(obj=P.seal(d=v["d_cap"] * 0.90, material=M.SEAL_RED if rng.random() < 0.4
                                 else M.SEAL_RUBBER),
                      at=tuple(cap_c - axis * v["flange"]),
                      rotate=(v["tilt_x"], v["tilt_y"], 0.0))
    prog = prog.place(obj=P.door(w=v["door_w"], h=v["door_h"], open_deg=v["door_open"],
                                 hinge_x=-(pocket_r + 0.004), skin=paint,
                                 liner=well_mat, rim=v["door_rim"],
                                 ribs=v["door_ribs"], arm_len=v["arm_len"],
                                 arm_w=v["arm_w"], arm_holes=v["arm_holes"]),
                      at=tuple(cap_c + axis * (body_z + 0.004)), rotate=body_rot)
    if v["tether"]:
        a = math.radians(v["cap_spin"] + v["tether_az"])
        prog = prog.place(obj=P.cap_boss(v["d_cap"] / 2, material=cap_mat),
                          at=tuple(cap_c + R @ np.array([0.46 * v["d_cap"] * math.cos(a),
                                                         0.46 * v["d_cap"] * math.sin(a),
                                                         -0.002])),
                          rotate=(v["tilt_x"], v["tilt_y"], v["cap_spin"] + v["tether_az"]))
        s = cap_c + R @ np.array([0.47 * v["d_cap"] * math.cos(a),
                                  0.47 * v["d_cap"] * math.sin(a), 0.001])
        e = np.array([pocket_r * 0.55, 0.010, -0.020 * v["z_gain"]])
        prog = prog.place(obj=P.tether(tuple(s), tuple(e), sag=0.008 + 0.016 * v["z_gain"],
                                       r=v["tether_r"], kinks=v["tether_kinks"],
                                       seed=int(v["cap_spin"] * 7) % 9999))

    cam = _camera(v, cap_c, R)
    gt = dict(variant=v, cap_centre=cap_c.tolist(), cap_normal=axis.tolist(),
              cap_x=(R @ _rot_xyz(0, 0, v["cap_spin"]) @ np.array([1.0, 0, 0])).tolist(),
              **cam)
    return prog, gt


def _camera(v, cap_c, R):
    """Place the eye on a cone of half-angle `obliq` about the cap's normal."""
    n = R @ np.array([0.0, 0.0, 1.0])
    t = np.array([0.0, 0.0, 1.0]) if abs(n[2]) < 0.9 else np.array([1.0, 0.0, 0.0])
    e1 = np.cross(t, n); e1 /= np.linalg.norm(e1)
    e2 = np.cross(n, e1)
    th, ph = math.radians(v["obliq"]), math.radians(v["azim"])
    d = n * math.cos(th) + (e1 * math.cos(ph) + e2 * math.sin(ph)) * math.sin(th)
    eye = cap_c + d * v["dist"]
    # aim slightly off the cap so it is not pinned to the principal point in every frame
    target = cap_c + e1 * v["aim_off"][0] + e2 * v["aim_off"][1]
    # roll the camera about its own view axis
    up0 = e2 if abs(float(n @ np.array([0, 0, 1.0]))) > 0.9 else np.array([0.0, 0.0, 1.0])
    f = target - eye; f /= np.linalg.norm(f)
    rt = np.cross(f, up0); rt /= np.linalg.norm(rt)
    up = np.cross(rt, f)
    a = math.radians(v["roll"])
    up = up * math.cos(a) + rt * math.sin(a)
    K = CAMERAS[v["camera"]]
    return dict(eye=eye.tolist(), target=target.tolist(), up=up.tolist(),
                fov_y=2.0 * math.atan((K["h"] / 2.0) / K["fy"]), K=K)
