"""Case 20 — diff & 3-way merge of op-logs: git-for-3D.

The model is a legible op-log, so two people (or a human and an AI) can edit it on
separate branches and reconcile like source code — impossible with an opaque scene file.
Here one base scene gets two *disjoint* edits: a "human" recolours the vase and nudges the
bowl; an "AI" adds a second book and repaints the floor. :func:`merge3` combines both with
no conflict; the render proves every edit landed.

    uv run python examples/cases/20_diff_merge.py            # print the diffs + render the 2x2

Prints the structured diff of each branch and a conflict example, and path-traces
base | human | AI | merged into docs/gallery/diff_merge.png. Needs mirage_render + Pillow.
"""
import sys
import copy
import json
import subprocess
from pathlib import Path

from mirage.meshlang import MeshProgram
from mirage.capture import default_render
from mirage.oplog_diff import diff_by_key, format_key_diff, merge_by_key

ROOT = Path(__file__).resolve().parents[2]
RENDER = default_render()
OUT = Path(__file__).resolve().parent / "outputs" / "20_diff_merge"
GALLERY = ROOT / "docs" / "gallery"


def mat(c):
    return {"color": list(c), "metallic": 0.0, "roughness": 0.45}

FLOOR = mat((0.50, 0.48, 0.45)); OAK = mat((0.60, 0.44, 0.25))
TEAL = mat((0.28, 0.47, 0.52)); AMBER = mat((0.86, 0.55, 0.16))
CREAM = mat((0.90, 0.88, 0.83)); RED = mat((0.62, 0.24, 0.20)); BLUE = mat((0.24, 0.36, 0.46))


def _box():
    return MeshProgram().cube(1.0)


def _vase():
    return MeshProgram().profile([[0.12, 0], [0.14, 0.03], [0.09, 0.14], [0.12, 0.32],
                                  [0.14, 0.46], [0.10, 0.54]], plane="xz").spin("z", steps=40)


def _bowl():
    return MeshProgram().profile([[0.0, 0.05], [0.09, 0.01], [0.16, 0.02], [0.20, 0.09]],
                                 plane="xz").spin("z", steps=36)


def base_scene():
    """The common ancestor: a floor + vase + bowl + one book, each a marked place op."""
    p = MeshProgram()
    p.place(obj=_box(), at=(0.0, 0.0, -0.05), scale=(1.9, 1.25, 0.06), material=FLOOR, mark="floor")
    p.place(obj=_vase(), at=(-0.42, 0.02, 0.0), material=TEAL, mark="vase")
    p.place(obj=_bowl(), at=(0.30, -0.10, 0.0), material=CREAM, mark="bowl")
    p.place(obj=_box(), at=(0.44, 0.30, 0.05), scale=(0.34, 0.24, 0.09), rotate=(0, 0, -8),
            material=RED, mark="book")
    return p.ops


def _by_mark(ops, mark):
    return next(o for o in ops if o.get("mark") == mark)


def human_edit(base):
    """A human at the GUI: recolour the vase (teal -> amber) and nudge the bowl."""
    ops = copy.deepcopy(base)
    _by_mark(ops, "vase")["material"]["color"] = list(AMBER["color"])
    _by_mark(ops, "bowl")["translate"] = [0.34, -0.24, 0.0]
    return ops


def ai_edit(base):
    """An AI over MCP: repaint the floor (-> oak) and add a second book."""
    ops = copy.deepcopy(base)
    _by_mark(ops, "floor")["material"]["color"] = list(OAK["color"])
    book2 = MeshProgram().place(obj=_box(), at=(0.52, 0.36, 0.05), scale=(0.32, 0.22, 0.10),
                                rotate=(0, 0, 7), material=BLUE, mark="book2").ops[0]
    ops.append(book2)
    return ops


def trace(ops, png, w=470, h=380, spp=110, denoise=5):
    OUT.mkdir(parents=True, exist_ok=True)
    jp = OUT / (png.stem + ".json"); jp.write_text(json.dumps(ops))
    ppm = OUT / (png.stem + ".ppm")
    subprocess.run([str(RENDER), "--oplog", str(jp), "--out", str(ppm), "--w", str(w), "--h", str(h),
                    "--spp", str(spp), "--bounce", "8", "--threads", "12", "--denoise", str(denoise),
                    "--sun", "1.15", "--env", "1.05", "--exposure", "1.0",
                    "--cam-eye", "1.55", "-1.75", "1.12", "--cam-target", "0.0", "-0.02", "0.12",
                    "--cam-fov", "0.82"], check=True)
    from PIL import Image
    Image.open(ppm).save(png)


def _panel(img, title, sub):
    from PIL import ImageDraw, ImageFont
    d = ImageDraw.Draw(img, "RGBA")
    def font(sz):
        for n in ("seguisb.ttf", "segoeui.ttf", "arial.ttf"):
            p = Path("C:/Windows/Fonts") / n
            if p.exists():
                return ImageFont.truetype(str(p), sz)
        return ImageFont.load_default()
    d.rectangle([0, 0, img.width, 30], fill=(18, 20, 24, 205))
    d.text((10, 6), title, font=font(19), fill=(70, 200, 214, 255))
    d.text((10 + int(d.textlength(title, font=font(19))) + 12, 8), sub, font=font(15), fill=(210, 214, 220, 255))
    return img


def main():
    base = base_scene()
    ours = human_edit(base)
    theirs = ai_edit(base)
    merged, conflicts = merge_by_key(base, ours, theirs)   # per-object (keyed on `mark`)

    print("=== diff  base -> human (GUI) ===")
    print(format_key_diff(diff_by_key(base, ours)))
    print("\n=== diff  base -> AI (MCP) ===")
    print(format_key_diff(diff_by_key(base, theirs)))
    print(f"\n=== 3-way merge ===  {len(merged)} objects, {len(conflicts)} conflict(s) "
          f"-- disjoint edits to different objects both applied automatically")

    # a conflict example: both branches recolour the SAME vase differently
    c_ours = copy.deepcopy(base); _by_mark(c_ours, "vase")["material"]["color"] = [0.8, 0.1, 0.1]
    c_theirs = copy.deepcopy(base); _by_mark(c_theirs, "vase")["material"]["color"] = [0.1, 0.7, 0.2]
    _, cc = merge_by_key(base, c_ours, c_theirs)
    print(f"=== conflict demo ===  both recolour the vase -> {len(cc)} conflict flagged (not silently lost)")

    if "--no-render" in sys.argv or not RENDER.exists():
        return
    from PIL import Image
    panels = [(base, "base", ""), (ours, "human", "recolour vase · move bowl"),
              (theirs, "AI", "add book · repaint floor"), (merged, "merged", "auto · both branches")]
    imgs = []
    for i, (ops, title, sub) in enumerate(panels):
        png = OUT / f"panel_{i}.png"; trace(ops, png)
        imgs.append(_panel(Image.open(png).convert("RGB"), title, sub))
    W, H = imgs[0].size
    sheet = Image.new("RGB", (W * 2 + 6, H * 2 + 6), (12, 13, 16))
    for i, im in enumerate(imgs):
        sheet.paste(im, ((i % 2) * (W + 6), (i // 2) * (H + 6)))
    GALLERY.mkdir(parents=True, exist_ok=True)
    sheet.save(GALLERY / "diff_merge.png")
    print(f"\nwrote {GALLERY / 'diff_merge.png'}")
    from mirage.capture import crossfade_clip                 # base -> human -> AI -> merged, as a clip
    crossfade_clip(imgs, "diff_merge", hold=26, fade=16, gif_w=W, tmp=OUT / "_clip")


if __name__ == "__main__":
    main()
