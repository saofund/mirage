"""Case 21 — a self-refining render loop: the agent sees its own model and fixes it.

A puppet-an-app MCP is blind: it fires commands and can't tell what came out. Mirage's
agent can **read** the op-log, **render** it (first-party tracer + denoiser), **look** at the
result, and **edit** the op-log to fix what it sees — a closed perception→action loop on its
own creation. This case starts from a scene with deliberate, render-only flaws (a floating
vase, a book clipping it, a muddy bowl, an over-exposed frame) and refines it round by round;
each round's critique + edit is exactly what the agent saw in the previous render.

    uv run python examples/cases/21_self_refine.py     # render every round -> docs/gallery

Needs mirage_render + Pillow.
"""
import sys
import copy
import json
import subprocess
from pathlib import Path

from mirage.meshlang import MeshProgram
from mirage.capture import default_render

ROOT = Path(__file__).resolve().parents[2]
RENDER = default_render()
OUT = Path(__file__).resolve().parent / "outputs" / "21_self_refine"
GALLERY = ROOT / "docs" / "gallery"


def mat(c, rough=0.45):
    return {"color": list(c), "metallic": 0.0, "roughness": rough}

FLOOR = mat((0.66, 0.62, 0.55)); TEAL = mat((0.28, 0.47, 0.52), 0.15)
MUD = mat((0.22, 0.24, 0.12)); CREAM = mat((0.90, 0.88, 0.83), 0.2)
RED = mat((0.62, 0.24, 0.20)); AMBER = mat((0.86, 0.55, 0.16))


def _box():
    return MeshProgram().cube(1.0)


def _vase():
    return MeshProgram().profile([[0.12, 0], [0.14, 0.03], [0.09, 0.14], [0.12, 0.32],
                                  [0.14, 0.46], [0.10, 0.54]], plane="xz").spin("z", steps=44)


def _bowl():
    return MeshProgram().profile([[0.0, 0.05], [0.09, 0.01], [0.16, 0.02], [0.20, 0.09]],
                                 plane="xz").spin("z", steps=40)


def by_mark(ops, m):
    return next(o for o in ops if o.get("mark") == m)


def flawed():
    """The starting state: op-log + render settings, with four deliberate flaws."""
    p = MeshProgram()
    p.place(obj=_box(), at=(0.0, 0.0, -0.05), scale=(1.9, 1.25, 0.06), material=FLOOR, mark="floor")
    p.place(obj=_vase(), at=(-0.28, 0.02, 0.34), material=TEAL, mark="vase")   # FLAW: floats (z=0.34)
    p.place(obj=_bowl(), at=(0.30, -0.08, 0.0), material=MUD, mark="bowl")     # FLAW: muddy colour
    p.place(obj=_box(), at=(-0.22, 0.06, 0.42), scale=(0.34, 0.24, 0.09),
            rotate=(0, 0, -8), material=RED, mark="book")                      # FLAW: clips the vase
    return {"ops": p.ops, "exposure": 1.7}                                     # FLAW: over-exposed


def trace(state, png, w=520, h=420, spp=120, denoise=5):
    OUT.mkdir(parents=True, exist_ok=True)
    jp = OUT / (png.stem + ".json"); jp.write_text(json.dumps(state["ops"]))
    ppm = OUT / (png.stem + ".ppm")
    subprocess.run([str(RENDER), "--oplog", str(jp), "--out", str(ppm), "--w", str(w), "--h", str(h),
                    "--spp", str(spp), "--bounce", "8", "--threads", "12", "--denoise", str(denoise),
                    "--sun", "1.15", "--env", "1.05", "--exposure", str(state["exposure"]),
                    "--cam-eye", "1.55", "-1.75", "1.10", "--cam-target", "0.0", "-0.02", "0.14",
                    "--cam-fov", "0.82"], check=True)
    from PIL import Image
    Image.open(ppm).save(png)


# ---- the refinement rounds (each derived from LOOKING at the previous render) ---- #
def round1(s):
    """Grounding & exposure. Seen in round 0: the vase and the book hover above the
    table (a shadow floats below them), the book is stuck through the vase, the bowl is a
    muddy green, and the whole frame is blown out."""
    s = copy.deepcopy(s)
    by_mark(s["ops"], "vase")["translate"] = [-0.30, 0.04, 0.0]          # drop the vase to the table
    by_mark(s["ops"], "book")["translate"] = [0.46, 0.30, 0.05]          # book onto the table, clear of the vase
    by_mark(s["ops"], "book")["rotate"] = [0, 0, -14]
    by_mark(s["ops"], "bowl")["material"]["color"] = list(CREAM["color"])  # muddy -> cream
    s["exposure"] = 1.05                                                 # fix the wash-out
    return s


def round2(s):
    """Composition & life. Round 1 is grounded and correctly lit, but the three objects are
    spread loosely and the bowl sits empty. Tighten the group and add a warm accent."""
    s = copy.deepcopy(s)
    by_mark(s["ops"], "vase")["translate"] = [-0.18, 0.08, 0.0]          # pull the vase toward centre
    by_mark(s["ops"], "bowl")["translate"] = [0.20, -0.06, 0.0]
    by_mark(s["ops"], "book")["translate"] = [0.50, 0.26, 0.05]
    fruit = MeshProgram().place(obj=MeshProgram().uv_sphere(segments=18, rings=12, radius=0.05),
                                at=(0.20, -0.06, 0.075), material=AMBER, mark="fruit").ops[0]
    s["ops"].append(fruit)                                               # a piece of fruit in the bowl
    return s


# each step: (edit fn or None for the start, label, the critique that DROVE this edit) --
STEPS = [
    (None,   "round 0  ·  flawed",   "vase & book float above the table; the book clips the vase; the bowl is muddy; the frame is blown out"),
    (round1, "round 1  ·  grounded", "dropped both to the table, moved the book clear of the vase, cream bowl, fixed the exposure"),
    (round2, "round 2  ·  composed", "tightened the group and set a warm fruit in the bowl"),
]


def _label(img, title, sub):
    from PIL import ImageDraw, ImageFont
    def font(sz):
        for n in ("seguisb.ttf", "segoeui.ttf", "arial.ttf"):
            p = Path("C:/Windows/Fonts") / n
            if p.exists():
                return ImageFont.truetype(str(p), sz)
        return ImageFont.load_default()
    d = ImageDraw.Draw(img, "RGBA")
    bar = 42
    d.rectangle([0, img.height - bar, img.width, img.height], fill=(16, 18, 22, 215))
    d.text((10, img.height - bar + 5), title, font=font(17), fill=(70, 200, 214, 255))
    d.text((10, img.height - bar + 24), sub, font=font(12), fill=(200, 205, 214, 255))
    return img


def main():
    from PIL import Image
    state = flawed()
    panels, log = [], []
    for i, (edit, title, crit) in enumerate(STEPS):
        if edit is not None:
            state = edit(state)
        png = OUT / f"round{i}.png"
        trace(state, png)
        panels.append(_label(Image.open(png).convert("RGB"), title, crit))
        log.append(f"  {title}: {crit}")
        print(f"rendered {title}")
    print("\n=== self-refinement log (each edit derived from the previous render) ===")
    print("\n".join(log))
    if "--no-strip" in sys.argv:
        return
    W, H = panels[0].size
    strip = Image.new("RGB", (W * len(panels) + 6 * (len(panels) - 1), H), (12, 13, 16))
    for i, p in enumerate(panels):
        strip.paste(p, (i * (W + 6), 0))
    GALLERY.mkdir(parents=True, exist_ok=True)
    strip.save(GALLERY / "self_refine.png")
    print(f"\nwrote {GALLERY / 'self_refine.png'}")


if __name__ == "__main__":
    main()
