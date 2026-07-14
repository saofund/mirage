"""Case 25 — the Eames Lounge Chair (670 & 671), sculpted from reference photographs.

Not a box-modelling exercise and not a procedural one: an icon, modelled the way a
sub-d modeller actually works — a coarse control cage, pushed into shape, held at the
rims by semi-sharp creases, then subdivided to its limit surface. Every curve here was
read off photographs of the real chair and its published dimensions (32.75" square,
31.5" high, 15" seat, 19" arm); nothing is imported, scanned or generated.

The interesting constraint: Mirage's op-log has NO vertex-addressing grammar, by design.
There is no "move vertex 47". So the cage is sculpted entirely through re-evaluable
QUERIES — `translate{on: {"by":"box", ...}}` — using the fact that nested box selections
accumulate into a cumulative sum, which `bend_surface` inverts to land any cage vertex on
any z(x,y) you can write down. The shape is a function, and the op-log stays legible.

    uv run python examples/cases/25_eames_lounge.py            # hero -> docs/gallery
    uv run python examples/cases/25_eames_lounge.py --preview  # fast low-spp look
    uv run python examples/cases/25_eames_lounge.py --parts    # each part, exploded

Needs mirage_render + Pillow.
"""
import math
import subprocess
import sys
from pathlib import Path

from mirage.capture import default_render
from mirage.meshlang import MeshProgram
from mirage.textures import ensure_textures

ROOT = Path(__file__).resolve().parents[2]
RENDER = default_render()
OUT = Path(__file__).resolve().parent / "outputs" / "25_eames_lounge"
GALLERY = ROOT / "docs" / "gallery"

TEX = ensure_textures(["wood_veneer", "leather"])

BIG = 100.0


# ---- materials ---------------------------------------------------------------- #
def mat(c, metallic=0.0, roughness=0.5, maps=None, uv_scale=1.0):
    m = {"color": list(c), "metallic": metallic, "roughness": roughness}
    if maps:
        m["albedo_map"] = str(maps["albedo"]); m["roughness_map"] = str(maps["rough"])
        m["normal_map"] = str(maps["normal"]); m["uv_scale"] = uv_scale
    return m


# uv_scale is metres per tile: the veneer tiles once across ~1.3 m, so the grain runs the
# length of the shell instead of repeating into stripes.
WALNUT  = mat((0.20, 0.10, 0.05), 0.0, 0.30, maps=TEX["wood_veneer"], uv_scale=1.30)
# The cut edge of the laminate: raw end-grain, pale and unlacquered against the dark face
# veneer. It is the chair's loudest signature — the cream line tracing every rim.
PLY     = mat((0.50, 0.38, 0.24), 0.0, 0.62)
LEATHER = mat((0.045, 0.042, 0.044), 0.0, 0.36, maps=TEX["leather"], uv_scale=0.16)


def paint(p, on, m):
    """The `material` op with the full PBR map set — the fluent builder only carries
    colour/metallic/roughness, and these surfaces need albedo/roughness/normal maps."""
    return p.add(op="material", on=on, **m)
ALU     = mat((0.62, 0.63, 0.65), 1.0, 0.25)          # polished aluminium column
BLACKM  = mat((0.045, 0.045, 0.048), 0.5, 0.42)       # the black base / brackets
GLIDE   = mat((0.10, 0.10, 0.10), 0.0, 0.75)


# ---- sculpting a cage through queries ------------------------------------------ #
def bend1(p, coords, axis, f, along=2):
    """Land the cage vertex at coords[m] on `along` += f(coords[m]) - f(coords[0]).

    Each box selects everything at or beyond coords[j] — a strict SUBSET of the last —
    so vertex m accumulates every band j <= m. That is a discrete integral, and choosing
    dz_j = f(c_j) - f(c_{j-1}) telescopes it exactly onto f.

    The band at coords[-1] would need faces beyond the final vertex and there are none,
    so the outermost cell always inherits its neighbour and stays parallel. That is a
    property of having only face selectors, not a bug; keep the cage fine enough that it
    lands in the rim, where solidify and subdivide round it away.
    """
    k = "xyz".index(axis)
    for j in range(1, len(coords) - 1):
        d = f(coords[j]) - f(coords[j - 1])
        if abs(d) < 1e-9:
            continue
        lo = [-BIG, -BIG, -BIG]
        lo[k] = coords[j]
        by = [0.0, 0.0, 0.0]
        by[along] = d
        p.translate({"by": "box", "min": lo, "max": [BIG, BIG, BIG]}, by)


def bend_surface(p, xs, ys, g):
    """Land cage vertex (xs[m], ys[n]) on z += g(x, y) EXACTLY — any surface you can write.

    A quadrant box (x >= xs[i] AND y >= ys[j]) moves exactly the vertices with m >= i and
    n >= j, so a grid of them accumulates into a 2-D cumulative sum that the second finite
    difference inverts. But that sum telescopes to g(m,n) - g(0,n) - g(m,0) + g(0,0): no
    quadrant can reach the x = xs[0] or y = ys[0] edges alone. So g is split into its two
    edge profiles — 1-D bands, which do reach them — plus the mixed remainder C, which
    vanishes on both edges by construction. The three parts sum back to exactly g.

    All three are DIFFERENCES, so they land the cage on g minus g at the first corner. That
    constant is added back at the end: without it the whole part silently floats by
    g(xs[0], ys[0]) — for the seat shell that is the front lip's own 0.125 drop, which lifted
    the arms to 0.58 and the chair to 0.91 against a 0.80 spec. Absolute is the only sane
    contract, because every caller places against measured heights.
    """
    bend1(p, xs, "x", lambda x: g(x, ys[0]))
    bend1(p, ys, "y", lambda y: g(xs[0], y))

    def C(x, y):
        return g(x, y) - g(x, ys[0]) - g(xs[0], y) + g(xs[0], ys[0])

    for i in range(1, len(xs) - 1):
        for j in range(1, len(ys) - 1):
            d = (C(xs[i], ys[j]) - C(xs[i - 1], ys[j])
                 - C(xs[i], ys[j - 1]) + C(xs[i - 1], ys[j - 1]))
            if abs(d) < 1e-9:
                continue
            p.translate({"by": "box", "min": [xs[i], ys[j], -BIG], "max": [BIG, BIG, BIG]},
                        [0, 0, d])

    corner = g(xs[0], ys[0])
    if abs(corner) > 1e-9:
        p.translate({"by": "all"}, [0, 0, corner])


def cage(x0, x1, nx, y0, y1, ny):
    xs = [x0 + i * (x1 - x0) / nx for i in range(nx + 1)]
    ys = [y0 + j * (y1 - y0) / ny for j in range(ny + 1)]
    return xs, ys


def smooth(t):
    t = min(max(t, 0.0), 1.0)
    return t * t * (3 - 2 * t)


# ---- the three plywood shells -------------------------------------------------- #
# Published dimensions: 32.75" (0.83) square, 31.5" (0.80) high, 15" (0.38) seat,
# 19" (0.48) arm. The pan sits at 0.30 so a 0.08 cushion tops out at the 0.38 seat
# height; the arm wing rises 0.15 above it so a 0.03 pad tops out at 0.48.
# The chair is 0.83 across the ARMS; the pan between them is ~0.55. So the wing has only
# ~0.14 of run to climb its 0.16 — it is a wall, not a ramp. Getting that ratio wrong is
# what makes a lounge chair read as a plank.
HALF_W, S_DEPTH, PAN_Z = 0.415, 0.62, 0.30
PAN_HALF = 0.275                     # where the pan ends and the wing starts
SHELL_T = 0.014                      # 7-ply, ~14 mm


def seat_z(x, y):
    lip = -0.125 * smooth((-0.09 - y) / 0.21) ** 1.25         # front lip curls down & under
    w = smooth((x - PAN_HALF) / (HALF_W - PAN_HALF)) ** 0.85  # nothing across the pan ...
    tall = 0.075 + 0.115 * smooth((y + 0.14) / 0.28)          # ... a wing, taller at the back
    crest = 1.0 - 0.22 * smooth((y - 0.18) / 0.13)            # top rounds over at the back
    # A lounge seat is a BUCKET: the pan has to hollow, or the cushion reads as a mattress
    # dropped on a plank rather than nestled into a shell.
    dish = -0.058 * smooth((PAN_HALF + 0.03 - abs(x)) / 0.26) * smooth((0.30 - abs(y + 0.01)) / 0.34)
    return lip + w * tall * crest + dish


def seat_shell():
    xs, ys = cage(0.0, HALF_W, 10, -S_DEPTH / 2, S_DEPTH / 2, 10)
    p = MeshProgram()
    p.grid(size_x=HALF_W, size_y=S_DEPTH, x_div=10, y_div=10)
    p.translate({"by": "all"}, [HALF_W / 2, 0.0, PAN_Z])       # build the +x half only
    bend_surface(p, xs, ys, seat_z)
    p.mirror(axis="x")                                         # weld the halves: exact symmetry
    p.solidify(thickness=SHELL_T, rim_mark="ply")              # tag the cut edge while it exists
    p.crease({"by": "sharp", "angle": 30.0}, weight=3.0)       # the plywood rim stays crisp
    p.subdivide(levels=3)
    paint(p, {"by": "all"}, WALNUT)
    paint(p, {"by": "tag", "name": "ply"}, PLY)                # tags survived the subdivision
    return p


def panel(half_w, h, curve, nx=6, ny=6, thickness=SHELL_T, levels=3):
    """A curved plywood panel: flat cage in xy, wrapped in +z by `curve(|x|)`.

    Built lying down and stood up by the caller's `place` rotation, so the same helper
    makes the backrest and the headrest. +z becomes 'toward the sitter' once rotated.
    """
    xs, ys = cage(0.0, half_w, nx, 0.0, h, ny)
    p = MeshProgram()
    p.grid(size_x=half_w, size_y=h, x_div=nx, y_div=ny)
    p.translate({"by": "all"}, [half_w / 2, h / 2, 0.0])
    bend1(p, xs, "x", curve)
    p.mirror(axis="x")
    p.solidify(thickness=thickness, rim_mark="ply")
    p.crease({"by": "sharp", "angle": 30.0}, weight=3.0)
    p.subdivide(levels=levels)
    paint(p, {"by": "all"}, WALNUT)
    paint(p, {"by": "tag", "name": "ply"}, PLY)
    return p


BACK_HW, BACK_H = 0.29, 0.30
HEAD_HW, HEAD_H = 0.255, 0.20


def back_shell():
    return panel(BACK_HW, BACK_H, lambda x: 0.085 * (x / BACK_HW) ** 2)


def head_shell():
    return panel(HEAD_HW, HEAD_H, lambda x: 0.070 * (x / HEAD_HW) ** 2)


# ---- cushions ------------------------------------------------------------------ #
def cushion(half_w, half_d, thick, buttons=(), nx=8, ny=8, levels=3):
    """A buttoned leather cushion: a dimpled cage, given depth by solidify, welted at the
    rim by a light crease, then subdivided into a pillow. The buttons are dimples in the
    cage — small boxes pulled down — so the tufting is real geometry, not a texture."""
    xs, ys = cage(-half_w, half_w, nx, -half_d, half_d, ny)

    def z(x, y):
        s = 0.0
        for bx, by, depth, r in buttons:
            d = math.hypot(x - bx, y - by)
            s -= depth * smooth((r - d) / r)               # a soft dimple at each button
        crown = 0.034 * smooth((half_w - abs(x)) / 0.19) * smooth((half_d - abs(y)) / 0.19)
        return s + crown                                    # the cushion crowns in the middle

    p = MeshProgram()
    p.grid(size_x=2 * half_w, size_y=2 * half_d, x_div=nx, y_div=ny)
    bend_surface(p, xs, ys, z)
    p.solidify(thickness=thick)
    # A fractional crease is the whole point here: weight 1 held the rim hard for a full
    # level and left a foam block with square corners. 0.35 only leans toward the sharp
    # rule, which reads as the stitched welt of a leather cushion rather than an edge.
    p.crease({"by": "sharp", "angle": 40.0}, weight=0.35)
    p.subdivide(levels=levels)
    return p


# ---- the aluminium base -------------------------------------------------------- #
def star_leg(length=0.33):
    """One flat tapered blade of the star base, lying along +x."""
    xs, _ = cage(0.0, length, 6, 0.0, 1.0, 1)
    p = MeshProgram()
    p.cube(size=1.0)
    p.scale({"by": "all"}, [length, 0.052, 0.030])
    p.translate({"by": "all"}, [length / 2, 0, 0])
    # taper: the blade narrows and thins toward the foot
    for j in range(1, len(xs) - 1):
        t = xs[j] / length
        p.scale({"by": "box", "min": [xs[j], -BIG, -BIG], "max": [BIG, BIG, BIG]},
                [1.0, 1.0 - 0.16 * t, 1.0 - 0.10 * t])
    p.crease({"by": "sharp", "angle": 40.0}, weight=2.0)
    p.subdivide(levels=2)
    return p


def base(n_arms, length=0.33):
    p = MeshProgram()
    p.place(star_leg(length), at=[0, 0, 0.028], material=BLACKM)
    for k in range(1, n_arms):
        p.place(star_leg(length), at=[0, 0, 0.028],
                rotate=[0, 0, 360.0 * k / n_arms], material=BLACKM)
    p.place(MeshProgram().cylinder(sides=28, radius=0.052, height=0.055),
            at=[0, 0, 0.042], material=BLACKM)                       # hub
    for k in range(n_arms):                                                 # glides
        a = math.radians(360.0 * k / n_arms)
        p.place(MeshProgram().cylinder(sides=16, radius=0.017, height=0.028),
                at=[length * 0.94 * math.cos(a), length * 0.94 * math.sin(a), 0.014],
                material=GLIDE)
    return p


# ---- the chair ----------------------------------------------------------------- #
BACK_TILT, HEAD_TILT = 71.0, 57.0        # degrees from horizontal (so 19 / 33 off vertical)


def facing(tilt_deg):
    """The 'toward the sitter' direction of a panel raked back by `tilt_deg`.

    A panel is built lying in z=0 and stood up by rotate=[tilt,0,0], so its local +z — the
    side solidify did NOT eat into — ends up here. Every cushion also occupies local
    [-thick, 0], so offsetting a cushion's origin by thick along this vector is what lands
    its back exactly on the panel's face instead of buried inside it.
    """
    a = math.radians(tilt_deg)
    return (0.0, -math.sin(a), math.cos(a))


def offset(origin, direction, d):
    return [origin[0] + direction[0] * d, origin[1] + direction[1] * d, origin[2] + direction[2] * d]


def chair():
    p = MeshProgram()
    p.place(base(5), at=[0, 0, 0])
    p.place(MeshProgram().cylinder(sides=28, radius=0.030, height=0.235),
            at=[0, 0, 0.175], material=ALU)                          # swivel column
    # No `material=` on the shells: they paint themselves walnut + laminate rim, and a
    # place material would flatten both back to one colour.
    p.place(seat_shell())

    # backrest: stood up and raked back, sitting just behind the seat shell's rim
    back_at = (0.0, 0.315, 0.475)
    p.place(back_shell(), at=list(back_at), rotate=[BACK_TILT, 0, 0])
    p.place(cushion(0.255, 0.135, 0.085,
                    buttons=[(-0.11, 0.06, 0.016, 0.09), (0.11, 0.06, 0.016, 0.09),
                             (0.0, -0.07, 0.016, 0.09)]),
            at=offset(back_at, facing(BACK_TILT), 0.085),
            rotate=[BACK_TILT, 0, 0], material=LEATHER)

    # headrest: raked back further again — the angle break is the chair's signature
    head_at = (0.0, 0.408, 0.715)
    p.place(head_shell(), at=list(head_at), rotate=[HEAD_TILT, 0, 0])
    p.place(cushion(0.222, 0.088, 0.075,
                    buttons=[(-0.09, 0.0, 0.013, 0.075), (0.09, 0.0, 0.013, 0.075)]),
            at=offset(head_at, facing(HEAD_TILT), 0.075),
            rotate=[HEAD_TILT, 0, 0], material=LEATHER)

    # seat cushion, filling the dished pan (whose top is PAN_Z + seat_z)
    p.place(cushion(0.263, 0.248, 0.105,
                    buttons=[(-0.105, 0.105, 0.016, 0.105), (0.105, 0.105, 0.016, 0.105),
                             (0.0, -0.065, 0.016, 0.105)]),
            at=[0, 0.015, PAN_Z + seat_z(0.0, 0.015) + 0.098], material=LEATHER)

    # arm pads, sitting on top of the wings
    for sx in (-1, 1):
        p.place(cushion(0.052, 0.135, 0.042),
                at=[sx * 0.372, 0.06, PAN_Z + seat_z(0.372, 0.06) + 0.042],
                material=LEATHER)

    # the black steel spine that carries the back and head shells off the seat wings
    for sx in (-1, 1):
        p.place(MeshProgram().cube(size=1.0),
                at=[sx * 0.30, 0.335, 0.44], scale=[0.020, 0.075, 0.16],
                rotate=[BACK_TILT - 90, 0, 0], material=BLACKM)
    for sx in (-1, 1):
        p.place(MeshProgram().cube(size=1.0),
                at=[sx * 0.115, 0.40, 0.655], scale=[0.016, 0.014, 0.10],
                rotate=[HEAD_TILT - 90, 0, 0], material=BLACKM)
    return p


def ottoman():
    p = MeshProgram()
    p.place(base(4, length=0.255), at=[0, 0, 0])
    p.place(MeshProgram().cylinder(sides=28, radius=0.026, height=0.16),
            at=[0, 0, 0.135], material=ALU)
    xs, ys = cage(0.0, 0.33, 6, -0.275, 0.275, 8)
    s = MeshProgram()
    s.grid(size_x=0.33, size_y=0.55, x_div=6, y_div=8)
    s.translate({"by": "all"}, [0.165, 0, 0.245])
    bend_surface(s, xs, ys, lambda x, y: 0.075 * smooth((x - 0.13) / 0.20)
                 - 0.030 * smooth((abs(y) - 0.15) / 0.13))
    s.mirror(axis="x")
    s.solidify(thickness=SHELL_T, rim_mark="ply")
    s.crease({"by": "sharp", "angle": 30.0}, weight=3.0)
    s.subdivide(levels=3)
    paint(s, {"by": "all"}, WALNUT)
    paint(s, {"by": "tag", "name": "ply"}, PLY)
    p.place(s)
    p.place(cushion(0.235, 0.20, 0.09,
                    buttons=[(-0.09, 0.0, 0.014, 0.09), (0.09, 0.0, 0.014, 0.09)]),
            at=[0, 0, 0.245 + 0.09], material=LEATHER)
    return p


def scene():
    p = MeshProgram()
    p.place(chair())
    p.place(ottoman(), at=[0, -0.80, 0])
    return p


# ---- render -------------------------------------------------------------------- #
def render(prog, out, spp, w, h, eye, target, fov=0.62, extra=()):
    OUT.mkdir(parents=True, exist_ok=True)
    js = OUT / (out + ".json")
    js.write_text(prog.to_json())
    ppm = OUT / (out + ".ppm")
    cmd = [str(RENDER), "--oplog", str(js), "--out", str(ppm),
           "--spp", str(spp), "--w", str(w), "--h", str(h), "--threads", "14",
           "--cam-eye", *[str(v) for v in eye], "--cam-target", *[str(v) for v in target],
           "--cam-fov", str(fov), *extra]
    subprocess.run(cmd, check=True)
    from PIL import Image
    png = OUT / (out + ".png")
    Image.open(ppm).save(png)
    return png


def main():
    preview = "--preview" in sys.argv
    p = scene()
    m = p.build()
    print(f"Eames lounge + ottoman: {len(m.verts):,} verts  {len(m.faces):,} faces  "
          f"({len(p.ops)} top-level ops)")
    spp = 40 if preview else 300
    # dark walnut and near-black leather blow out fast: keep the key soft and the exposure
    # honest rather than lighting it like a white studio product shot
    png = render(p, "hero", spp, 1280, 800, [1.32, -1.52, 0.74], [-0.06, -0.20, 0.40],
                 fov=0.62, extra=["--sun", "0.45", "--env", "0.34", "--exposure", "0.95",
                                  "--sun-dir", "0.45", "0.55", "0.70", "--denoise", "4"])
    print("wrote", png)
    if not preview:
        GALLERY.mkdir(parents=True, exist_ok=True)
        from PIL import Image
        Image.open(png).save(GALLERY / "eames_lounge.png")
        print("wrote", GALLERY / "eames_lounge.png")


if __name__ == "__main__":
    main()
