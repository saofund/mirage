"""Case 24 — a grand living room, HAND-COMPOSED (no procedural generation).

Not a loop, not noise: a large, carefully arranged interior where every piece is modelled
from Mirage's operators (the lathe turns the vases and lamp bases, `bevel` softens every
cushion and tabletop, `boolean` cuts the window, `inset`+`extrude` makes the picture frames)
and every piece is *placed by hand* into a considered layout — a sofa grouping on a rug, a
reading nook, a sideboard under a mirror, plants, art, warm light through a big window. The
whole room is one legible op-log of `place` ops, path-traced by the first-party renderer.

    uv run python examples/cases/24_grand_interior.py            # hero -> docs/gallery
    uv run python examples/cases/24_grand_interior.py --preview  # fast low-spp look

Needs mirage_render + Pillow.
"""
import sys
import math
import subprocess
from pathlib import Path

from mirage.meshlang import MeshProgram, Sel
from mirage.capture import default_render
from mirage.textures import ensure_textures

ROOT = Path(__file__).resolve().parents[2]
RENDER = default_render()
OUT = Path(__file__).resolve().parent / "outputs" / "24_grand_interior"
GALLERY = ROOT / "docs" / "gallery"

# real PBR map sets (albedo / roughness / normal), generated on demand -> assets/textures/
TEX = ensure_textures(["wood_floor", "wood_walnut", "wood_oak",
                       "fabric_sofa", "fabric_cush", "fabric_rug", "plaster", "marble"])


# ---- a cohesive, warm mid-century palette ------------------------------------ #
def mat(c, metallic=0.0, roughness=0.5, emission=None, tex=None, tex_scale=4.0, tex2=None,
        maps=None, uv_scale=1.0):
    m = {"color": list(c), "metallic": metallic, "roughness": roughness}
    if emission:
        m["emission"] = list(emission)     # a light source (radiance)
    if tex:                                # procedural texture fallback: "wood"/"fabric"/"stone"
        m["tex"] = tex; m["tex_scale"] = tex_scale
        if tex2:
            m["tex2"] = list(tex2)
    if maps:                               # real image maps (triplanar); uv_scale = m per tile
        m["albedo_map"] = str(maps["albedo"]); m["roughness_map"] = str(maps["rough"])
        m["normal_map"] = str(maps["normal"]); m["uv_scale"] = uv_scale
    return m

FLOOR   = mat((0.46, 0.32, 0.17), 0.0, 0.30, maps=TEX["wood_floor"],  uv_scale=1.5)
WALL    = mat((0.85, 0.81, 0.74), 0.0, 0.95, maps=TEX["plaster"],     uv_scale=2.6)
RUG     = mat((0.50, 0.28, 0.23), 0.0, 0.92, maps=TEX["fabric_rug"],  uv_scale=0.7)
RUG2    = mat((0.66, 0.58, 0.46), 0.0, 0.92)   # cream border
SOFA    = mat((0.40, 0.48, 0.39), 0.0, 0.88, maps=TEX["fabric_sofa"], uv_scale=0.45)
CHAIR   = mat((0.50, 0.27, 0.13), 0.0, 0.55)   # cognac leather (smooth)
CUSH_A  = mat((0.80, 0.60, 0.26), 0.0, 0.85, maps=TEX["fabric_cush"], uv_scale=0.35)
CUSH_B  = mat((0.40, 0.28, 0.30), 0.0, 0.85, maps=TEX["fabric_cush"], uv_scale=0.35)
CUSH_C  = mat((0.85, 0.80, 0.70), 0.0, 0.85, maps=TEX["fabric_cush"], uv_scale=0.35)
WALNUT  = mat((0.30, 0.19, 0.12), 0.0, 0.38, maps=TEX["wood_walnut"], uv_scale=0.6)
OAK     = mat((0.60, 0.44, 0.26), 0.0, 0.45, maps=TEX["wood_oak"],    uv_scale=0.7)
BRASS   = mat((0.82, 0.62, 0.30), 1.0, 0.28)
BLACKM  = mat((0.10, 0.10, 0.11), 0.6, 0.4)    # black metal
CERAMIC = mat((0.92, 0.90, 0.85), 0.0, 0.14)
TEAL    = mat((0.24, 0.46, 0.50), 0.0, 0.12)
TERRA   = mat((0.70, 0.40, 0.27), 0.0, 0.55)   # terracotta pot
SHADE   = mat((0.98, 0.93, 0.80), 0.0, 0.6, emission=(7.0, 4.9, 2.8))   # a glowing warm lampshade
WINGLOW = mat((0.96, 0.90, 0.78), 0.0, 0.9, emission=(3.0, 2.5, 1.7))   # warm golden-hour sky behind the window
LEAF    = mat((0.27, 0.43, 0.25), 0.0, 0.6)
LEAF2   = mat((0.33, 0.49, 0.29), 0.0, 0.6)
CANVAS  = mat((0.62, 0.55, 0.46), 0.0, 0.7)
BOOKS   = [mat((0.55, 0.24, 0.20)), mat((0.24, 0.34, 0.42)), mat((0.30, 0.40, 0.28)),
           mat((0.70, 0.58, 0.32)), mat((0.44, 0.30, 0.42)), mat((0.78, 0.72, 0.60))]


# ---- geometry helpers -------------------------------------------------------- #
def box(cx, cy, cz, sx, sy, sz):
    hx, hy, hz = sx / 2, sy / 2, sz / 2
    v = [(cx-hx, cy-hy, cz-hz), (cx+hx, cy-hy, cz-hz), (cx+hx, cy+hy, cz-hz), (cx-hx, cy+hy, cz-hz),
         (cx-hx, cy-hy, cz+hz), (cx+hx, cy-hy, cz+hz), (cx+hx, cy+hy, cz+hz), (cx-hx, cy+hy, cz+hz)]
    f = [(0, 3, 2, 1), (4, 5, 6, 7), (0, 1, 5, 4), (1, 2, 6, 5), (2, 3, 7, 6), (3, 0, 4, 7)]
    return v, f


def rounded(sx, sy, sz, w=0.05):
    """A soft-edged box (the upholstery / tabletop primitive) via bevel."""
    return MeshProgram().mesh(*box(0, 0, 0, sx, sy, sz)).bevel(Sel.all(), width=w, depth=w * 0.8)


def slab(p, cx, cy, cz, sx, sy, sz, m):
    v, f = box(cx, cy, cz, sx, sy, sz)
    return p.place(verts=v, faces=f, material=m)


def cyl(sides, r, h):
    return MeshProgram().cylinder(sides=sides, radius=r, height=h)


def lathe(points, steps=48):
    return MeshProgram().profile(points, plane="xz", closed=False).spin("z", steps=steps)


def vase(kind="tall"):
    if kind == "tall":
        pts = [(0.075, 0), (0.09, 0.03), (0.06, 0.16), (0.075, 0.34), (0.10, 0.5), (0.08, 0.58), (0.055, 0.62)]
    else:
        pts = [(0.09, 0), (0.13, 0.05), (0.15, 0.18), (0.10, 0.30), (0.075, 0.34)]
    return lathe(pts, 44)


def bowl():
    return lathe([(0.0, 0.04), (0.08, 0.01), (0.15, 0.02), (0.185, 0.085)], 40)


def lamp_shade(r0=0.09, r1=0.17, h=0.2):
    return MeshProgram().profile([(r0, 0.0), (r1, -h)], plane="xz", closed=False).spin("z", steps=40)


def plant(pot_r=0.16, pot_h=0.30, spread=0.18, blobs=7, base_z=0.0, leaf=LEAF):
    """A potted plant: a terracotta pot + a cluster of foliage blobs (uv_spheres)."""
    g = MeshProgram()
    g.place(obj=lathe([(pot_r*0.8, 0), (pot_r, 0.04), (pot_r*0.95, pot_h*0.9), (pot_r*1.02, pot_h)], 28),
            at=(0, 0, base_z), material=TERRA)
    for k in range(blobs):
        a = k * 2.399
        rr = spread * (0.5 + 0.5 * ((k * 7) % 5) / 4)
        g.place(obj=MeshProgram().uv_sphere(segments=16, rings=10, radius=0.10 + 0.03 * ((k * 3) % 4) / 3),
                at=(rr*math.cos(a), rr*math.sin(a), base_z + pot_h + 0.12 + 0.09*((k*5) % 4)/3),
                material=leaf if k % 2 else LEAF2)
    return g


def framed(w, h, thick=0.05):
    return (MeshProgram().mesh(*box(0, 0, 0, thick, w, h))
            .inset(on=Sel.side("x", 1), thickness=0.06)
            .extrude(on=Sel.last(), distance=-0.02))


def window_wall(width, thick, height, hw, hh, hcx, hcz):
    p = MeshProgram().mesh(*box(0, 0, height/2, width, thick, height))
    return p.boolean("difference", box(hcx, 0, hcz, hw, thick*3, hh))


# ---- the room: every piece placed BY HAND, grouped so the making-of can reuse it #
CONX, CONY = -0.3, 3.2   # sideboard anchor (shared by console + its plant + mirror)


def _shell(p, windowed=True):
    """Floor, left wall, and the back wall — solid, or with the window boolean cut."""
    slab(p, 0.0, 0.4, -0.06, 7.2, 6.0, 0.12, FLOOR)                       # floor
    slab(p, -3.5, 0.4, 1.35, 0.14, 6.0, 2.9, WALL)                        # left wall (x = -3.5)
    if windowed:
        _window(p)
    else:
        p.place(obj=MeshProgram().mesh(*box(0, 0, 1.45, 7.2, 0.14, 2.9)), at=(0, 3.42, 0), material=WALL)  # solid back wall


def _window(p):
    """The back wall with the window punched out, its mullions, and the glowing sky pane."""
    p.place(obj=window_wall(7.2, 0.14, 2.9, 2.6, 1.5, 0.4, 1.45), at=(0, 3.42, 0), material=WALL)  # back wall + window
    p.place(obj=MeshProgram().mesh(*box(-0.86, 0, 1.45, 0.04, 0.08, 1.5)).array(count=4, offset=(0.55, 0, 0)),
            at=(0, 3.34, 0), material=OAK)                                # window mullions (vertical)
    p.place(obj=MeshProgram().mesh(*box(-0.86, 0, 1.45, 2.65, 0.08, 0.04)).array(count=2, offset=(0, 0, 0.72)),
            at=(0, 3.34, 0), material=OAK)                                # window mullions (horizontal)
    slab(p, 0.4, 3.55, 1.45, 2.5, 0.04, 1.42, WINGLOW)                    # glowing golden-hour sky behind the window


def _rug(p):
    slab(p, -0.9, 0.7, 0.012, 3.6, 2.8, 0.02, RUG)
    slab(p, -0.9, 0.7, 0.024, 3.2, 2.4, 0.014, RUG2)


def _sofa(p):
    """Main sofa against the LEFT wall, facing +x."""
    sx0, sy0 = -2.85, 0.7
    p.place(obj=rounded(0.98, 2.5, 0.34, 0.05), at=(sx0, sy0, 0.32), material=SOFA)        # seat base
    p.place(obj=rounded(0.30, 2.5, 0.78, 0.05), at=(sx0 - 0.34, sy0, 0.62), material=SOFA) # backrest
    arms = MeshProgram().mesh(*box(0, 1.24, 0.46, 0.98, 0.30, 0.56)).mirror("y").bevel(Sel.all(), 0.05, 0.04)
    p.place(obj=arms, at=(sx0, sy0, 0), material=SOFA)                                     # mirrored arms
    for k, cy in enumerate((-0.78, 0.02, 0.82)):
        p.place(obj=rounded(0.80, 0.72, 0.20, 0.06), at=(sx0 + 0.06, sy0 + cy, 0.54), material=CUSH_C)  # seat cushions
    for k, cy in enumerate((-0.92, -0.1, 0.9)):
        p.place(obj=rounded(0.40, 0.40, 0.16, 0.07), at=(sx0 + 0.18, sy0 + cy, 0.62),
                rotate=(0, 0, 18 - 16 * k), material=[CUSH_A, CUSH_B, CUSH_A][k])          # throw pillows
    p.place(obj=rounded(0.9, 1.0, 0.05, 0.03), at=(sx0 + 0.28, sy0 + 0.5, 0.56), rotate=(0, 0, 8), material=CUSH_B)  # throw


def _coffee(p):
    """Coffee table on the rug, with a still life (the lathe turns the vase and bowl)."""
    ctx, cty = -0.95, 0.7
    p.place(obj=rounded(1.5, 0.86, 0.10, 0.03), at=(ctx, cty, 0.40), material=WALNUT)      # beveled top
    for lx in (-0.62, 0.62):
        for ly in (-0.32, 0.32):
            p.place(obj=cyl(14, 0.045, 0.36), at=(ctx + lx, cty + ly, 0.20), material=WALNUT)  # legs
    p.place(obj=vase("tall"), at=(ctx - 0.45, cty + 0.1, 0.45), material=TEAL)
    p.place(obj=bowl(), at=(ctx + 0.2, cty - 0.08, 0.45), material=CERAMIC)
    for k, (fx, fy) in enumerate([(0.16, -0.06), (0.24, -0.02), (0.2, -0.12)]):
        p.place(obj=MeshProgram().uv_sphere(segments=16, rings=10, radius=0.05),
                at=(ctx + fx, cty + fy, 0.52 + 0.01*k), material=[mat((0.78,0.28,0.2)), mat((0.86,0.54,0.18)), mat((0.5,0.6,0.22))][k])
    p.place(obj=MeshProgram().mesh(*box(0, 0, 0, 0.42, 0.30, 0.05)).bevel(Sel.all(), 0.01),  # stacked books
            at=(ctx + 0.5, cty + 0.28, 0.44), rotate=(0, 0, -12), material=BOOKS[1])
    p.place(obj=MeshProgram().mesh(*box(0, 0, 0, 0.40, 0.28, 0.045)).bevel(Sel.all(), 0.01),
            at=(ctx + 0.52, cty + 0.26, 0.49), rotate=(0, 0, -6), material=BOOKS[3])


def armchair(p, ax, ay, arot, blocked=False):
    """A leather armchair, turned to the sofa. blocked=True leaves it as sharp boxes
    (the making-of shows `bevel` rounding it)."""
    c, s = math.cos(math.radians(arot)), math.sin(math.radians(arot))
    if blocked:
        p.place(obj=MeshProgram().mesh(*box(0, 0, 0, 0.80, 0.80, 0.30)), at=(ax, ay, 0.30), rotate=(0, 0, arot), material=CHAIR)
        p.place(obj=MeshProgram().mesh(*box(0, 0, 0, 0.22, 0.80, 0.60)), at=(ax + 0.29*c, ay + 0.29*s, 0.57), rotate=(0, 0, arot), material=CHAIR)
        return
    p.place(obj=rounded(0.80, 0.80, 0.30, 0.06), at=(ax, ay, 0.30), rotate=(0, 0, arot), material=CHAIR)
    p.place(obj=rounded(0.22, 0.80, 0.60, 0.06), at=(ax + 0.29*c, ay + 0.29*s, 0.57), rotate=(0, 0, arot), material=CHAIR)
    p.place(obj=MeshProgram().mesh(*box(0, 0.98, 0.42, 0.80, 0.22, 0.46)).mirror("y").bevel(Sel.all(), 0.05, 0.04),
            at=(ax, ay, 0), rotate=(0, 0, arot), material=CHAIR)                           # arms
    p.place(obj=rounded(0.66, 0.66, 0.15, 0.06), at=(ax, ay, 0.46), rotate=(0, 0, arot), material=CUSH_C)


def _nook(p):
    """The reading armchair (foreground) + a knitted pouf."""
    armchair(p, 0.55, -0.45, 150)
    p.place(obj=rounded(0.52, 0.52, 0.26, 0.09), at=(-0.5, -0.7, 0.15), material=CUSH_A)     # a knitted pouf


def _bookshelf(p):
    """Bookshelf against the back wall (right of the window)."""
    bx, by = 2.5, 3.16
    p.place(obj=MeshProgram().mesh(*box(0, 0, 1.05, 1.7, 0.36, 2.1)).boolean("difference", box(0, 0.12, 1.06, 1.54, 0.36, 1.9)),
            at=(bx, by, 0), material=OAK)                                                   # carcass
    p.place(obj=MeshProgram().mesh(*box(0, 0, 0.28, 1.54, 0.32, 0.04)).array(count=5, offset=(0, 0, 0.42)),
            at=(bx, by, 0), material=OAK)                                                   # shelves
    for shelf in range(5):
        z0 = 0.30 + shelf * 0.42
        x = -0.66
        while x < 0.62:
            i = shelf * 17 + int((x + 1) * 11)
            h = 0.26 + 0.05 * ((i * 3) % 5) / 4
            w = 0.04 + 0.02 * ((i * 7) % 3) / 2
            tilt = 12 if (i % 9) == 0 else 0
            p.place(obj=MeshProgram().mesh(*box(0, 0, 0, w, 0.24, h)),
                    at=(bx + x, by - 0.02, z0 + h/2), rotate=(0, tilt, 0), material=BOOKS[i % len(BOOKS)])
            x += w + 0.012
        if shelf in (1, 3):                                                                # a decorative object per couple shelves
            p.place(obj=vase("squat") if shelf == 1 else bowl(),
                    at=(bx + 0.45, by - 0.02, z0 + 0.02), material=[CERAMIC, TEAL][shelf == 3])


def _console(p):
    """Sideboard under the window, with vases, a dish, and a glowing table lamp."""
    conx, cony = CONX, CONY
    p.place(obj=rounded(2.0, 0.5, 0.7, 0.03), at=(conx, cony, 0.42), material=WALNUT)       # console body
    for lx in (-0.9, 0.9):
        p.place(obj=cyl(10, 0.03, 0.36), at=(conx + lx, cony - 0.02, 0.18), rotate=(0, 0, 0), material=BRASS)  # legs
    p.place(obj=vase("tall"), at=(conx - 0.7, cony, 0.77), material=CERAMIC)
    p.place(obj=lathe([(0.0, 0.05), (0.09, 0.01), (0.16, 0.02), (0.2, 0.09)], 36), at=(conx + 0.6, cony, 0.77), material=TEAL)  # a dish
    p.place(obj=lathe([(0.05, 0), (0.09, 0.02), (0.05, 0.10), (0.04, 0.24)], 32), at=(conx + 0.0, cony + 0.02, 0.77), material=BRASS)  # lamp base
    p.place(obj=lamp_shade(0.10, 0.15, 0.18), at=(conx + 0.0, cony + 0.02, 1.12), material=SHADE)  # glowing shade


def _floor_lamp(p):
    """Floor lamp in the back-right reading corner (by the bookshelf)."""
    lx, ly = 2.95, 2.35
    p.place(obj=cyl(24, 0.14, 0.05), at=(lx, ly, 0.025), material=BLACKM)
    p.place(obj=cyl(14, 0.022, 1.55), at=(lx, ly, 0.80), material=BRASS)
    p.place(obj=lamp_shade(0.13, 0.20, 0.24), at=(lx, ly, 1.58), material=SHADE)


def _greenery_and_art(p):
    """Plants in the corner + on the console, framed art on the left wall, a round mirror."""
    conx, cony = CONX, CONY
    p.place(obj=plant(pot_r=0.24, pot_h=0.5, spread=0.34, blobs=11), at=(-3.05, 2.7, 0.0))
    p.place(obj=plant(pot_r=0.10, pot_h=0.16, spread=0.13, blobs=6, leaf=LEAF2), at=(conx + 0.85, cony, 0.77))
    p.place(obj=framed(1.0, 0.72), at=(-3.42, 0.2, 1.7), material=WALNUT)                  # large picture on left wall
    slab(p, -3.39, 0.2, 1.7, 0.02, 0.86, 0.58, CANVAS)
    p.place(obj=framed(0.62, 0.5), at=(-3.42, 1.6, 1.55), material=BRASS)                  # smaller picture
    slab(p, -3.39, 1.6, 1.55, 0.02, 0.5, 0.38, mat((0.5, 0.56, 0.6), 0.0, 0.6))
    p.place(obj=MeshProgram().torus(major_segments=40, minor_segments=16, major_radius=0.34, minor_radius=0.03),
            at=(conx, 3.38, 1.9), rotate=(90, 0, 0), material=BRASS)                        # round mirror frame over console
    p.place(obj=cyl(40, 0.33, 0.02), at=(conx, 3.39, 1.9), rotate=(90, 0, 0), material=mat((0.7, 0.75, 0.78), 0.3, 0.15))


def build_room():
    """The whole room, groups placed in the same order the making-of reveals them."""
    p = MeshProgram()
    _shell(p)
    _rug(p)
    _sofa(p)
    _coffee(p)
    _nook(p)
    _bookshelf(p)
    _console(p)
    _floor_lamp(p)
    _greenery_and_art(p)
    return p


# ---- making-of: the room assembling in the real native viewer ----------------- #
def film_stages():
    """The build story, group by group, with a couple of hero-operator beats featured in
    place — `boolean` punches the window, `bevel` rounds the blocked-out armchair. Each stage
    is an independent op-log the viewer renders; the last stage equals build_room() exactly,
    so the path-traced money-shot hold is the same geometry as the hero still."""
    stages, caps = [], []
    p = MeshProgram()

    def snap(cap):
        stages.append(MeshProgram(p.ops)); caps.append(cap)

    _shell(p, windowed=False)                       # floor, left wall, a SOLID back wall
    snap("room shell  ·  floor + walls up")

    p.ops.pop()                                     # drop the solid wall; re-place it windowed
    _window(p)
    snap("window  ·  boolean cut + array mullions")

    _rug(p);    snap("rug  ·  anchoring the living grouping")
    _sofa(p);   snap("sofa  ·  bevel + mirrored arms")
    _coffee(p); snap("coffee table  ·  the lathe turns the vase")

    armchair(p, 0.55, -0.45, 150, blocked=True)     # sharp boxes, then bevel rounds them
    snap("armchair  ·  blocked out (sharp boxes)")
    p.ops.pop(); p.ops.pop()
    _nook(p)
    snap("armchair  ·  bevel rounds it, + a pouf")

    _bookshelf(p);        snap("bookshelf  ·  array shelves + books")
    _console(p);          snap("sideboard  ·  a turned lamp, vases")
    _floor_lamp(p);       snap("floor lamp  ·  the reading corner")
    _greenery_and_art(p); snap(f"complete  ·  {len(p.ops)} place ops, one op-log")
    return stages, caps


def film():
    """Film the room building group-by-group in mirage_viewer -> grand_interior_build.mp4/.gif,
    settling onto a path-traced golden-hour close-up (the money shot)."""
    import os
    from mirage.capture import record_build
    stages, captions = film_stages()
    quick = os.environ.get("ANIM_QUICK") == "1"
    # a moving camera through the open +x,-y corner: establish high & wide, descend and push in
    # as the shell/window go up, swing to the window & sofa reads with some 运镜, then drive IN
    # and settle to a tight golden-hour framing of the whole grouping (materials read close).
    # (t, yaw, pitch, dist, tx, ty, tz); last two keys equal -> a static, path-traced dwell.
    moves = [
        (0.00, 0.62, 0.50, 8.2, -0.30, 0.70, 0.95),   # establishing — high & wide over the corner
        (0.14, 0.80, 0.40, 6.9, -0.35, 0.70, 0.92),   # descend + dolly in as walls/window go up
        (0.30, 0.96, 0.30, 5.7, -0.10, 0.95, 0.92),   # swing to the window / console, lower
        (0.46, 0.74, 0.34, 6.2, -0.55, 0.75, 0.90),   # ease back, recentre on the sofa grouping
        (0.60, 0.57, 0.29, 5.6, -0.72, 0.55, 0.86),   # swing to a frontal read of sofa + armchair
        (0.74, 0.82, 0.23, 4.9, -0.30, 0.80, 0.85),   # come back in, low — the materials read
        (0.90, 0.85, 0.18, 5.6, -0.52, 0.98, 0.92),   # settle to the hero framing at money-shot dist
        (1.00, 0.85, 0.18, 5.6, -0.52, 0.98, 0.92),   # hold (flat -> the path-traced money shot)
    ]
    tmp = Path(os.environ["ANIM_TMP"]) if os.environ.get("ANIM_TMP") else None
    knobs = {"sun": 0.66, "env": 0.15, "exposure": 1.08, "sun-dir": (0.42, 0.62, 0.33),
             "bloom": 0.15, "aperture": 0.08}  # golden-hour hero light + photographic post
    if os.environ.get("ANIM_RAYTRACE") == "1":
        # a fully PATH-TRACED promo reel -> its own asset (grand_interior_raytrace.*): every
        # frame via mirage_render (GI, soft shadows, emissive lamps), low spp + denoise = clean.
        record_build(
            stages, "grand_interior_raytrace", captions=captions, automode=False, keyframes=moves,
            renderer="raytrace", trace_spp=128, trace_threads=int(os.environ.get("ANIM_THREADS", "128")),
            trace_denoise=5, trace_knobs=knobs, cam_fov=0.82,
            size=(960, 540), fps=24, per=6, hold=16, gif_w=520, gif_fps=10, tmp=tmp,
        )
    else:
        # default: fast smooth AA'd viewport for the build, one path-traced frame for the hold.
        record_build(
            stages, "grand_interior_build", captions=captions, automode=True, keyframes=moves,
            smooth=True, trace_hold=True, trace_spp=420, trace_threads=12, trace_denoise=5, trace_knobs=knobs, cam_fov=0.82,
            size=(854, 480) if quick else (1280, 720),
            fps=24, per=8 if quick else 12, hold=12 if quick else 28,
            gif_w=520, gif_fps=10, tmp=tmp,
        )


# ---- render ------------------------------------------------------------------ #
def trace(prog, png, w=1400, h=900, spp=240, denoise=5, threads=16):
    OUT.mkdir(parents=True, exist_ok=True)
    jp = OUT / (png.stem + ".json"); jp.write_text(prog.to_json(indent=None))
    ppm = OUT / (png.stem + ".ppm")
    subprocess.run([str(RENDER), "--oplog", str(jp), "--out", str(ppm), "--w", str(w), "--h", str(h),
                    "--spp", str(spp), "--bounce", "8", "--threads", str(threads), "--denoise", str(denoise),
                    "--sun", "0.66", "--env", "0.15", "--exposure", "1.08", "--sun-dir", "0.42", "0.62", "0.33",
                    "--bloom", "0.15", "--aperture", "0.08",   # photographic glow + a gentle depth of field
                    "--cam-eye", "4.6", "-3.7", "2.05", "--cam-target", "-0.6", "1.02", "0.9", "--cam-fov", "0.82"],
                   check=True)
    from PIL import Image
    png.parent.mkdir(parents=True, exist_ok=True)
    Image.open(ppm).save(png)


def main():
    p = build_room()
    m = p.build()
    print(f"  grand interior: {len(p.ops)} place ops, {len(m.faces)} faces")
    if "--film" in sys.argv:
        film()
        return
    if "--preview" in sys.argv:
        trace(p, OUT / "preview.png", w=640, h=430, spp=40, denoise=4)
        print(f"  wrote {OUT / 'preview.png'}")
    else:
        trace(p, GALLERY / "grand_interior.png", w=1500, h=980, spp=420, threads=140)
        print(f"  wrote {GALLERY / 'grand_interior.png'}")


if __name__ == "__main__":
    main()
