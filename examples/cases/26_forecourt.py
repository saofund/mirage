"""Case 26 — reproducing a real scene from one photograph.

The reference is a single CCTV still of a petrol-station forecourt (Camera 01, wet
morning). Nothing about it was authored for a renderer: a wide-angle lens near the ceiling,
a graphic painted floor, a dispenser island covered in small hardware, and a working yard
behind it. The job is to read the photo and rebuild the scene as an op-log — geometry,
palette, layout and camera — then render it from the same viewpoint and put the two side by
side.

The scene is built in three separable pieces, which is the point of the case:

* the **camera**, solved from the photo (`mirage.solve`) — see the block above CAM_EYE;
* the **parts**, each modelled and reviewed on its own (`forecourt/parts.py`, and
  `python -m forecourt.sheet` to look at them one at a time);
* the **layout** below, which is nothing but `place` ops at unprojected positions.

Keeping those apart is what makes the thing improvable: a part can be rebuilt without
touching the camera, and the layout can be re-solved without touching the parts.

    uv run python examples/cases/26_forecourt.py            # hero -> docs/gallery
    uv run python examples/cases/26_forecourt.py --preview  # fast low-spp look

Needs mirage_render + Pillow.
"""
import math
import os
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))   # so `forecourt` imports as a kit

from forecourt import parts as P                                          # noqa: E402
from forecourt.materials import (APRON, BAY_BLUE, BAY_ORNG, BLACK, CONCRETE,  # noqa: E402
                                 LINE_W, PROMO_F, REPAIR_F, ROAD, SHUTTER_D,
                                 WASH_F, WHITE, YELLOW, YELLOWP, mat)
from mirage.capture import default_render                                 # noqa: E402
from mirage.meshlang import MeshProgram                                   # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
RENDER = default_render()
OUT = Path(__file__).resolve().parent / "outputs" / "26_forecourt"
GALLERY = ROOT / "docs" / "gallery"
# The reference lives outside the repo (it is somebody's CCTV frame, not an asset). Override
# with MIRAGE_REF; without it the render still runs, only the side-by-side is skipped.
REF = Path(os.environ.get("MIRAGE_REF", "D:/dRepo_26/frame_2026-5-9_09-28-55.png"))
# A path-traced forecourt at 300spp is minutes on a laptop and seconds on the 152-core box.
# Don't hardcode the machine you happen to be sitting at.
THREADS = os.environ.get("MIRAGE_THREADS", "14")

# The scene's forward axis (the bay's long edge, world +y) is not square to the building
# line behind it; the yard and road sit at this yaw.
ANG = -15.0
ISLAND_AT = [-3.12, -1.05]   # where the pump island stands, unprojected from its footprint


def at(dx, dy, ang=ANG, base=(8.5, 20.2)):
    """A point `dx` along the building line and `dy` off it — the yard's own coordinates.
    The facade runs at ANG, so laying its clutter out in world x/y by hand is a way to get
    everything subtly off the wall."""
    c, s = math.cos(math.radians(ang)), math.sin(math.radians(ang))
    return [base[0] + dx * c - dy * s, base[1] + dx * s + dy * c]


def slab(p, x0, x1, y0, y1, z, t, material):
    """A flat painted patch lying ON the ground, spanning [z, z+t] — not straddling z, which
    buries half of it in the slab and z-fights the rest."""
    p.place(P.box(x1 - x0, y1 - y0, t), at=[(x0 + x1) / 2, (y0 + y1) / 2, z + t / 2],
            material=material)


# ---- the painted forecourt ------------------------------------------------------ #
# The dominant graphic, rebuilt in the SOLVED camera's frame by unprojecting the photo. The
# blue bay's four measured corners are a 3.47 x 6.0 m rectangle at world x[0,3.47] y[0,6];
# the terracotta bays, the road and the building line were unprojected the same way with
# solve.ground_point. Nothing here is eyeballed against a render -- the camera put every
# edge where it saw it, and a projection overlay on the photo confirmed the fit.
def forecourt():
    p = MeshProgram()
    # the damp concrete apron: ONE textured slab. Aggregate, hairline cracks and dark
    # oil/water staining all live in the map; the wet sheen is the map's low-roughness
    # pools mirroring the sky, not a flat near-black fill.
    p.place(P.box(44, 48, 0.4), at=[6, 8, -0.2], material=APRON)
    # The wet asphalt is a BAND, not the whole back of the yard: in the reference the shop
    # apron in front of the shutters is light concrete, and paving it black is what made
    # the whole yard read as a night scene.
    p.place(P.box(40, 5.0, 0.03), at=[7, 15.0, 0.012], rotate=[0, 0, ANG], material=ROAD)
    # The saw-cut construction joints. A poured apron is a GRID of slabs, and those long
    # straight lines are most of what tells the eye it is looking at concrete rather than at
    # a grey plane — the texture's random cracks cannot supply them, because they are not
    # random. Laid below the bays (z 0.001..0.005) so the paint covers them, as it does.
    JOINT = mat((0.115, 0.118, 0.120), 0.0, 0.58)
    for k in range(-2, 7):
        p.place(P.box(0.022, 46, 0.004), at=[-7.6 + k * 5.4, 8, 0.003], material=JOINT)
    for k in range(-2, 6):
        p.place(P.box(42, 0.022, 0.004), at=[6, -8.1 + k * 5.4, 0.003], material=JOINT)
    LW = 0.11
    # the painted bays, each ONE textured slab: the worn paint, the fine cracks and the
    # organic standing water are baked into the bay maps, so there are no stacked overlay
    # rectangles (which read as a dartboard) and no overlapping top faces to z-fight.
    bays = [(0, 3.47, 0, 6, BAY_BLUE), (3.62, 7.5, -3.4, 6.0, BAY_ORNG),
            (0, 3.47, 6.2, 9.7, BAY_ORNG)]
    for x0, x1, y0, y1, m in bays:
        slab(p, x0, x1, y0, y1, 0.004, 0.006, m)
        for lx in (x0 - LW, x1):
            slab(p, lx, lx + LW, y0 - LW, y1 + LW, 0.006, 0.006, LINE_W)
        for ly in (y0 - LW, y1):
            slab(p, x0 - LW, x1 + LW, ly, ly + LW, 0.006, 0.006, LINE_W)
    # The white RING painted around the island. Both discs are painted INSIDE the
    # sub-program: a material on the outer `place` would repaint every face it carries,
    # including the concrete infill, and the ring would come out as a solid white pancake —
    # which is exactly what it had been doing.
    ring = MeshProgram()
    ring.place(MeshProgram().cylinder(sides=72, radius=1.03, height=0.006), material=LINE_W)
    ring.place(MeshProgram().cylinder(sides=72, radius=0.90, height=0.014), material=APRON)
    p.place(ring, at=[-3.20, -2.44, 0.004])
    p.place(P.box(0.15, 2.0, 0.006), at=[9.0, 9.4, 0.004], rotate=[0, 0, ANG], material=YELLOWP)
    for s in (-1, 1):
        p.place(P.box(0.15, 0.95, 0.006), at=[9.0 + s * 0.28, 8.55, 0.004],
                rotate=[0, 0, ANG + s * 40], material=YELLOWP)
    p.place(P.box(14, 0.14, 0.006), at=[8, 12.6, 0.004], rotate=[0, 0, ANG], material=YELLOWP)
    p.place(P.drain_channel(13.0), at=[7.0, 11.3, 0], rotate=[0, 0, ANG])
    return p


# ---- the yard behind ------------------------------------------------------------ #
def yard():
    """The building line across the back and its clutter, placed by unprojecting the
    building base line (world y ~ 20, yawed by ANG) and the road front."""
    p = MeshProgram()
    p.place(P.facade(24.0, 5.2), at=[8.5, 20.2, 0], rotate=[0, 0, ANG], mark="facade")
    for dx in (-8.4, -4.3, -0.2, 4.6, 8.7):                       # the workshop bays
        c = at(dx, -0.22)
        p.place(P.roller_shutter(3.4, 3.2), at=[c[0], c[1], 0], rotate=[0, 0, ANG],
                mark="shutters")
    for dx in (-6.35, 2.2, 6.65):                                 # the piers between them
        c = at(dx, -0.24)
        p.place(P.tiled_pilaster(0.62, 4.6), at=[c[0], c[1], 0], rotate=[0, 0, ANG])
    c = at(11.6, -0.30)                                           # an open doorway
    p.place(P.box(2.1, 0.12, 2.6), at=[c[0], c[1], 1.35], rotate=[0, 0, ANG],
            material=mat((0.045, 0.048, 0.05), 0.0, 0.6))
    c = at(-2.4, -0.55)
    p.place(P.hanging_banner(1.30, 1.76, WASH_F), at=[c[0], c[1], 3.15], rotate=[0, 0, ANG])
    c = at(3.9, -0.55)
    p.place(P.hanging_banner(0.55, 2.75, PROMO_F), at=[c[0], c[1], 3.30], rotate=[0, 0, ANG])
    # the vehicles and the ground clutter along the wall
    p.place(P.van(), at=[9.9, 17.4, 0], rotate=[0, 0, ANG + 4], mark="van")
    for dx, dy in [(-9.0, -1.5), (-6.1, -1.6), (-3.2, -1.7), (-0.4, -1.8), (2.4, -1.9),
                   (5.2, -2.0)]:
        c = at(dx, dy)
        p.place(P.bollard(), at=[c[0], c[1], 0], mark="bollards")
    for dx, dy, kind in [(-7.6, -0.95, "bin"), (-7.0, -0.9, "broom"), (-6.8, -0.85, "broom")]:
        c = at(dx, dy)
        if kind == "bin":
            p.place(P.bin_(), at=[c[0], c[1], 0], rotate=[0, 0, ANG])
        else:
            p.place(P.broom(1.4), at=[c[0], c[1], 0], rotate=[0, 9, ANG])
    for dx, dy, top in [(-5.2, -1.15, (0.10, 0.11, 0.13)), (0.9, -1.05, (0.30, 0.30, 0.31)),
                        (1.5, -1.15, (0.14, 0.15, 0.30))]:
        c = at(dx, dy)
        p.place(P.person(1.70, top), at=[c[0], c[1], 0], rotate=[0, 0, ANG + 150])
    for k, (dx, dy) in enumerate([(-9.6, -1.05), (-9.0, -1.15)]):  # the 小心地滑 A-frames
        c = at(dx, dy)
        p.place(P.wet_floor_sign(), at=[c[0], c[1], 0], rotate=[0, 0, ANG + 8 * k])
    p.place(P.speed_hump(3.6, 0.52), at=[11.0, 5.6, 0], rotate=[0, 0, 26], mark="hump")
    p.place(P.bollard(0.86), at=[13.6, 8.6, 0])
    p.place(P.hanging_banner(0.60, 2.60, REPAIR_F), at=[-5.6, 11.0, 1.75], rotate=[0, 0, -6])
    p.place(P.suv(), at=[-5.3, 4.4, 0], rotate=[0, 0, 96], mark="suv")   # half out of frame
    return p


def scene():
    p = MeshProgram()
    # Every top-level object is MARKED, which makes the scene measurable rather than just
    # renderable: mirage_render --ids turns these tags into a per-pixel object id, so
    # photomatch.chamfer_per_object can score each against the photo separately. That
    # per-object score is the loss that will POLISH this scene; measurement -- the solved
    # camera and the unprojected layout -- is what landed it in the basin first, which no
    # loss could do: eleven cameras once all scored 14-15 px because nothing downhill led
    # to the true camera, 22 deg of yaw and half the fov away.
    p.place(forecourt(), mark="forecourt")
    p.place(P.island(), at=[ISLAND_AT[0], ISLAND_AT[1], 0], rotate=[0, 0, ANG], mark="island")
    p.place(P.hazard_rail(1.88, 0.74), at=[-3.16, -2.30, 0], rotate=[0, 0, ANG - 1],
            mark="rail")
    p.place(P.jerrycan(), at=[-3.92, -3.00, 0])
    p.place(yard(), mark="yard")
    return p


# ---- render --------------------------------------------------------------------- #
# THE CAMERA, SOLVED. The 1189 px falsification (git log: "let a camera be tested against the
# photograph, and fail") retired the asserted camera; this is what replaced it, and how:
#
#   - orientation + fov from the bay's strip vanishing point (272,-412) FUSED with "the bay is
#     a rectangle": the single fov that unprojects the four measured corners to right angles.
#     That pins fov to 0.505 with no fragile vertical trace, and the strip VP reproduces exactly.
#   - the corners then unproject to 91/89/89/91 deg (the asserted camera gave 77/93/81/108) and
#     aspect 1.73 -- a 3.47 x 6.0 m bay. An independent check that the orientation+fov are right.
#   - eye by linear least squares once orientation+fov are fixed and the gauge is chosen (bay
#     length 6 m): the four corners reproject within 4.2 px. Eye lands 6.1 m up, a CCTV on a post.
#
# The recovered camera differs from the asserted one by exactly what the falsification predicted:
# yaw pulled back ~22 deg, fov roughly halved (1.181 -> 0.505). Every object in the scene above
# was then unprojected through THIS camera with solve.ground_point and checked by projecting the
# layout back onto the photo -- not by eyeballing a render. Measurement lands you in the basin;
# the per-object chamfer loss is what polishes inside it. (solve.camera_from_vanishing_points.)
CAM_EYE, CAM_TGT, CAM_FOV = [-3.1349, -12.0379, 6.0852], [-2.8408, -11.1592, 5.7095], 0.5054

# The scene's parts, FINEST FIRST — a face takes the first of these tags it carries, so
# listing "sign" before "island" scores the sign on its own instead of dissolving it into
# the island. This list is the scorecard's vocabulary: anything not named here lands in the
# coarse bucket and cannot be blamed for anything.
ID_TAGS = ["sign", "dispenser", "hoses", "firebox", "bucket", "plinth", "column", "rail",
           "van", "suv", "shutters", "bollards", "hump", "facade", "island", "yard",
           "forecourt"]


def render(prog, out, spp, w, h, extra=()):
    OUT.mkdir(parents=True, exist_ok=True)
    js = OUT / (out + ".json")
    js.write_text(prog.to_json())
    ppm = OUT / (out + ".ppm")
    subprocess.run([str(RENDER), "--oplog", str(js), "--out", str(ppm),
                    "--spp", str(spp), "--w", str(w), "--h", str(h), "--threads", THREADS,
                    "--cam-eye", *[str(v) for v in CAM_EYE],
                    "--cam-target", *[str(v) for v in CAM_TGT],
                    "--cam-fov", str(CAM_FOV), *extra], check=True)
    from PIL import Image
    png = OUT / (out + ".png")
    Image.open(ppm).save(png)
    return png


def compare(png):
    """Reference above, render below — the only honest way to report a reproduction."""
    from PIL import Image, ImageDraw
    a = Image.open(REF).convert("RGB")
    b = Image.open(png).convert("RGB")
    w = 1280
    a = a.resize((w, int(a.height * w / a.width)), Image.LANCZOS)
    b = b.resize((w, int(b.height * w / b.width)), Image.LANCZOS)
    out = Image.new("RGB", (w, a.height + b.height + 6), (18, 18, 20))
    out.paste(a, (0, 0))
    out.paste(b, (0, a.height + 6))
    d = ImageDraw.Draw(out)
    for y, t in [(0, "reference  (CCTV still)"), (a.height + 6, "mirage  (op-log, path-traced)")]:
        d.rectangle([0, y, 250, y + 18], fill=(12, 12, 14))
        d.text((6, 4 + y), t, fill=(238, 238, 240))
    p = OUT / "compare.png"
    out.save(p)
    return p


def critique(png):
    """Score the render against the photograph, per object, worst first.

    This is the answer to a question that kept being asked by a human: "which bit is wrong?"
    A render is re-rendered dozens of times and the eye adapts to it after the third — every
    fault this scene has shipped with (an apron twice as light as the photo, a yard paved
    black, a van that was the wrong kind of van, ten objects that were painted boxes) was
    visible in the picture the whole time and still needed somebody to say so. The
    scorecard says so instead, in a table sorted by how much it matters.
    """
    from mirage import critique as C
    from mirage.photomatch import read_ids
    ids_path = OUT / "hero_ids.pgm"
    if not ids_path.exists():
        print("(no ids AOV — run without --preview, or with --critique, to write one)")
        return
    ren, ref = C.load_pair(png, REF)
    rows = C.scorecard(ren, ref, read_ids(ids_path), names=ID_TAGS)
    print()
    print(C.report(rows))
    print("wrote", C.plate(ren, read_ids(ids_path), rows, OUT / "scorecard.png", names=ID_TAGS))
    return rows


def main():
    preview = "--preview" in sys.argv
    p = scene()
    m = p.build()
    print(f"forecourt: {len(m.verts):,} verts  {len(m.faces):,} faces  ({len(p.ops)} top-level ops)")
    spp = 40 if preview else 260
    # An overcast wet morning: soft high sun, a bright sky fill for the wet surfaces to mirror,
    # no hard key. Env is turned up (0.86) because on this shot the reflected sky IS the light on
    # the floor -- the wet materials are near-black diffuse and read only through what they
    # reflect. Exposure 1.35 sits the concrete where the reference's is; --clamp stops a hot
    # specular sample on the near-mirror puddles from leaving a firefly the denoiser can't fix.
    # --sky-tint warms the sky fill. The scorecard is what found this: EVERY object came
    # back with a cool cast, all seventeen, which is not seventeen colour errors but one —
    # the sky is the only thing lighting most of this scene, and it was blue.
    extra = ["--sun", "0.12", "--env", "0.86", "--exposure", "1.35", "--clamp", "1.5",
             "--sun-dir", "0.25", "0.55", "0.80", "--sky-tint", "1.26", "1.07", "0.86",
             # denoise 4 at 260 spp was eating the apron's grain: the scorecard read detail
             # 0.18 and barely moved when the map got twice the staining, because the filter
             # was removing exactly what the map was adding. 2 keeps it.
             "--denoise", "2"]
    # The id AOV comes out of the SAME trace as the beauty pass, so the scorecard's masks
    # are the pixels it is scoring and not a re-render that drifted.
    extra += ["--ids", str(OUT / "hero_ids.pgm"), "--id-tags", ",".join(ID_TAGS)]
    png = render(p, "hero", spp, 1600, 900, extra=extra)
    print("wrote", png)
    if REF.exists():
        print("wrote", compare(png))
        critique(png)
    else:
        print(f"(no reference at {REF} — skipping the side-by-side; set MIRAGE_REF)")
    if not preview:
        GALLERY.mkdir(parents=True, exist_ok=True)
        from PIL import Image
        Image.open(png).save(GALLERY / "forecourt.png")   # the render alone: a pure synthetic image
        # The side-by-side is NOT copied to the gallery, and must never be: it embeds the
        # reference CCTV frame, which is somebody's security footage, not ours to publish. It
        # lives only in outputs/ (gitignored). Only the render ships.
        print("wrote", GALLERY / "forecourt.png")


if __name__ == "__main__":
    main()
