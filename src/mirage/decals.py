"""mirage.decals — printed artwork for parts.

A modelled object is a shape. A real one is a shape with PRINTING on it: a price board,
119 stencilled on a fire bucket, a number plate, a wet-floor pictogram, a shop's banner.
Strip the printing and even a well-modelled part reads as a prop — which is most of what
"looks CG" means on man-made objects.

This module draws that artwork with Pillow and writes it as a PPM the tracer pins to a
rectangle (a *decal*: `Material.decal_origin/du/dv`), so a graphic lands exactly where it
was authored, at the size it was authored, and stops at the panel's edge — unlike a tiling
triplanar map, which has no anchor and would repeat the sign across the whole column.

    from mirage.decals import ensure_decals
    D = ensure_decals(["pump_sign"])        # {name: {"albedo": Path}}

Colours here are LINEAR (the renderer's albedo space), not sRGB — so the constants match
the ones a case file passes to `mat()`, and antialiasing blends in the space the tracer
integrates in. A decal PNG therefore looks washed out if opened directly; that is correct.

Same cache discipline as `mirage.textures`: the recipe digest hashes the drawing code, so
editing a decal regenerates it instead of silently leaving the old art on disk.

Needs Pillow.
"""
from __future__ import annotations

import hashlib
import math
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[2]
DECAL_DIR = ROOT / "assets" / "decals"

# A heavy CJK sans is what shop signage actually uses. Fall back through the usual
# Windows/Linux names; a missing font degrades to Pillow's bitmap default rather than
# raising, because a case that renders slightly-wrong text still renders.
_FONT_CANDIDATES = [
    "C:/Windows/Fonts/simhei.ttf", "C:/Windows/Fonts/msyhbd.ttc", "C:/Windows/Fonts/msyh.ttc",
    "/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc",
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc",
    "/usr/share/fonts/truetype/arphic/uming.ttc",
]
_FONT_CACHE: dict = {}


def font(px: int):
    key = int(px)
    if key not in _FONT_CACHE:
        f = None
        for cand in _FONT_CANDIDATES:
            if Path(cand).exists():
                try:
                    f = ImageFont.truetype(cand, key)
                    break
                except OSError:
                    continue
        _FONT_CACHE[key] = f or ImageFont.load_default()
    return _FONT_CACHE[key]


def _text(d, xy, s, px, fill, anchor="mm", spacing=0.0):
    """Centred text with optional letter-spacing (signage is usually tracked out)."""
    f = font(px)
    if not spacing:
        d.text(xy, s, font=f, fill=fill, anchor=anchor)
        return
    widths = [d.textlength(ch, font=f) for ch in s]
    total = sum(widths) + spacing * (len(s) - 1)
    x = xy[0] - total / 2 if anchor[0] == "m" else xy[0]
    for ch, w in zip(s, widths):
        d.text((x + w / 2, xy[1]), ch, font=f, fill=fill, anchor="m" + anchor[1])
        x += w + spacing


# Linear-space palette (see the module docstring on why these look dark as sRGB).
NAVY = (86, 118, 214)     # banners: 0.230, 0.335, 0.470, against the photograph's 0.554
# The LIGHTBOX gets its own, darker. It shared NAVY with the banners, so three rounds of
# lifting a banner that measured too dark also lifted a sign that was already too light —
# 0.578 against 0.396. Two objects with different targets cannot share one constant.
SIGN_NAVY = (48, 76, 170)   # 0.578 too light, then 0.285 too dark, target 0.396
SIGN_NAVY_D = (32, 52, 126)
NAVY_D = (40, 62, 150)
RED = (152, 22, 6)       # scarlet. The cabinet's FACE is this decal, not the RED material,
                         # so lightening the material alone left it brown on screen.
RED_D = (104, 14, 6)
ORANGE = (172, 80, 34)
YELLOW = (215, 165, 12)
WHITE = (232, 232, 228)
OFFWHITE = (206, 206, 200)
GREY = (120, 120, 118)
BLACK = (14, 14, 15)


# --------------------------------------------------------------------------- #
# the artwork
# --------------------------------------------------------------------------- #
def _pump_sign(W, H):
    """The lightbox on the column: 油卡支付 超划算 over a red price arrow.

    This one panel is the largest readable graphic in the frame, so it carries most of
    the shot's "this is a petrol station in China" information. Built as flat colour
    blocks plus type, which is what a printed lightbox face IS."""
    im = Image.new("RGB", (W, H), OFFWHITE)
    d = ImageDraw.Draw(im)
    m = int(W * 0.055)                                  # the aluminium frame's reveal
    d.rectangle([m, m, W - m, H - m], fill=WHITE)
    hdr = int(H * 0.088)                                # white header: logo + station name
    top = m + int(H * 0.004)
    d.rectangle([m + 4, top, W - m - 4, top + hdr], fill=WHITE)
    # the oil-company roundel
    r = int(hdr * 0.42)
    cx, cy = m + int(W * 0.13), top + hdr // 2
    d.ellipse([cx - r, cy - r, cx + r, cy + r], fill=(140, 26, 18))
    _text(d, (cx, cy), "Gulf", int(r * 0.85), WHITE)
    _text(d, ((cx + r + W - m) / 2 + 4, cy), "海湾石油 · 联河加油站", int(hdr * 0.30), (40, 40, 44))
    # the blue field
    fy0, fy1 = top + hdr + int(H * 0.008), H - m - 4
    d.rectangle([m + 4, fy0, W - m - 4, fy1], fill=SIGN_NAVY)
    for k in range(9):                                  # a faint vertical gradient/sheen
        d.rectangle([m + 4, fy0 + int((fy1 - fy0) * k / 9), W - m - 4,
                     fy0 + int((fy1 - fy0) * (k + 1) / 9)],
                    fill=tuple(int(c * (1.0 + 0.13 * (4 - k) / 4)) for c in SIGN_NAVY))
    big = int(W * 0.180)   # sized to FIT: 4 chars + tracking must clear the blue field
    _text(d, (W / 2, fy0 + (fy1 - fy0) * 0.115), "油卡支付", big, WHITE, spacing=W * 0.010)
    _text(d, (W / 2, fy0 + (fy1 - fy0) * 0.225), "超划算", big, WHITE, spacing=W * 0.010)

    # the price arrow: a broad down-pointing pentagon
    ax0, ax1 = W * 0.095, W * 0.905
    ay0, ay1 = fy0 + (fy1 - fy0) * 0.325, fy0 + (fy1 - fy0) * 0.825
    sh = (ay1 - ay0) * 0.66                             # where the shoulders are
    inx = (ax1 - ax0) * 0.17
    d.polygon([(ax0, ay0), (ax1, ay0), (ax1, ay0 + sh), (ax1 - inx, ay0 + sh),
               ((ax0 + ax1) / 2, ay1), (ax0 + inx, ay0 + sh), (ax0, ay0 + sh)], fill=RED)
    d.line([(ax0, ay0), (ax1, ay0), (ax1, ay0 + sh), (ax1 - inx, ay0 + sh),
            ((ax0 + ax1) / 2, ay1), (ax0 + inx, ay0 + sh), (ax0, ay0 + sh), (ax0, ay0)],
           fill=(196, 44, 24), width=max(2, W // 180))
    _text(d, (W / 2, ay0 + (ay1 - ay0) * 0.085), "每周二油卡特惠日", int(W * 0.082), WHITE)
    _text(d, (W / 2, ay0 + (ay1 - ay0) * 0.215), "降", int(W * 0.115), WHITE)
    _text(d, (W / 2, ay0 + (ay1 - ay0) * 0.435), "1.1", int(W * 0.40), YELLOW)
    _text(d, (W / 2, ay0 + (ay1 - ay0) * 0.655), "元/升", int(W * 0.088), YELLOW)
    _text(d, (W / 2, fy0 + (fy1 - fy0) * 0.90), "其他时段0.8元/升", int(W * 0.082), WHITE)
    return im


def _fire_cabinet(W, H):
    """灭火器箱 — the red extinguisher cabinet's printed front."""
    im = Image.new("RGB", (W, H), RED)
    d = ImageDraw.Draw(im)
    d.rectangle([0, 0, W, int(H * 0.055)], fill=RED_D)             # the lid: RED in the
                                                                  # reference, not orange
    d.rectangle([0, int(H * 0.46), W, int(H * 0.475)], fill=RED_D)  # the two-door split
    ey = int(H * 0.16)                                              # extinguisher pictogram
    d.rounded_rectangle([int(W * 0.10), ey, int(W * 0.19), ey + int(H * 0.11)],
                        radius=int(W * 0.02), fill=WHITE)
    _text(d, (W * 0.58, ey + H * 0.055), "灭火器箱", int(W * 0.175), WHITE, spacing=W * 0.008)
    _text(d, (W * 0.5, H * 0.36), "火警电话 119", int(W * 0.115), WHITE)
    for k in (0, 1):                                                # the two instruction panels
        x0 = W * (0.10 + 0.42 * k)
        d.rectangle([x0, H * 0.60, x0 + W * 0.38, H * 0.86], fill=OFFWHITE)
        d.rectangle([x0 + W * 0.03, H * 0.63, x0 + W * 0.35, H * 0.79], fill=(150, 150, 146))
        _text(d, (x0 + W * 0.19, H * 0.825), "使用方法", int(W * 0.075), (30, 30, 32))
    _text(d, (W * 0.5, H * 0.545), "灭火器使用方法", int(W * 0.085), WHITE)
    return im


def _fire_bucket(W, H):
    """消防桶 119 — white stencil on galvanised steel.

    The background is the BUCKET'S OWN colour, not black or transparent: a decal has no
    alpha, so whatever is behind the letters is painted onto the part. Get this wrong and
    the bucket wears a black card."""
    # Galvanised steel photographs BRIGHT -- the reference's bucket is one of the lighter
    # things on the island. At 117/120/122 it read as a dark cylinder in the corner.
    im = Image.new("RGB", (W, H), (186, 189, 191))
    d = ImageDraw.Draw(im)
    _text(d, (W * 0.5, H * 0.40), "消防桶", int(W * 0.30), WHITE, spacing=W * 0.01)
    _text(d, (W * 0.5, H * 0.68), "119", int(W * 0.20), WHITE)
    return im


def _wet_floor(W, H):
    """The folding 小心地滑 caution sign's face."""
    im = Image.new("RGB", (W, H), (198, 158, 12))
    d = ImageDraw.Draw(im)
    d.rectangle([int(W * 0.05), int(H * 0.04), int(W * 0.95), int(H * 0.96)], outline=BLACK,
                width=max(2, W // 90))
    px, py = W * 0.5, H * 0.34                                     # the slipping figure
    d.ellipse([px - W * 0.07, py - H * 0.14, px + W * 0.07, py - H * 0.045], fill=BLACK)
    d.line([(px - W * 0.02, py - H * 0.04), (px + W * 0.16, py + H * 0.10)], fill=BLACK,
           width=int(W * 0.075))
    d.line([(px - W * 0.02, py - H * 0.02), (px - W * 0.22, py + H * 0.06)], fill=BLACK,
           width=int(W * 0.06))
    d.line([(px + W * 0.10, py + H * 0.10), (px + W * 0.26, py + H * 0.02)], fill=BLACK,
           width=int(W * 0.055))
    for k in range(3):
        d.arc([px - W * (0.30 - 0.06 * k), py + H * (0.13 + 0.02 * k),
               px + W * (0.30 - 0.06 * k), py + H * (0.25 + 0.02 * k)], 0, 180, fill=BLACK,
              width=max(2, W // 70))
    _text(d, (W * 0.5, H * 0.72), "小心地滑", int(W * 0.185), BLACK, spacing=W * 0.008)
    _text(d, (W * 0.5, H * 0.86), "CAUTION", int(W * 0.11), BLACK, spacing=W * 0.02)
    return im


def _wash_banner(W, H):
    """加油站洗车机 右转100米 — the hanging banner over the yard."""
    im = Image.new("RGB", (W, H), NAVY)
    d = ImageDraw.Draw(im)
    d.rectangle([0, 0, W, int(H * 0.10)], fill=NAVY_D)
    _text(d, (W * 0.5, H * 0.20), "加油站洗车机", int(W * 0.145), WHITE, spacing=W * 0.006)
    r = W * 0.085
    cx, cy = W * 0.17, H * 0.46
    d.ellipse([cx - r, cy - r, cx + r, cy + r], fill=WHITE)
    d.polygon([(cx - r * 0.45, cy - r * 0.35), (cx + r * 0.5, cy), (cx - r * 0.45, cy + r * 0.35)],
              fill=NAVY)
    _text(d, (W * 0.60, cy), "右转100米", int(W * 0.155), WHITE)
    d.rectangle([W * 0.14, H * 0.66, W * 0.86, H * 0.80], fill=(150, 150, 146))
    _text(d, (W * 0.5, H * 0.73), "加油积分免费洗车", int(W * 0.085), NAVY_D)
    return im


def _promo_banner(W, H):
    """The big blue promotional banner at the yard's left edge."""
    im = Image.new("RGB", (W, H), NAVY)
    d = ImageDraw.Draw(im)
    for k, s in enumerate(["高", "牌", "油", "品", "质", "优"]):
        _text(d, (W * 0.5, H * (0.085 + 0.132 * k)), s, int(W * 0.52), YELLOW)
    d.rectangle([W * 0.08, H * 0.87, W * 0.92, H * 0.97], fill=WHITE)
    _text(d, (W * 0.5, H * 0.92), "海湾石油", int(W * 0.21), NAVY_D)
    return im


def _repair_sign(W, H):
    """修 · 保养 — the workshop's vertical sign on the tiled pilaster."""
    im = Image.new("RGB", (W, H), (188, 92, 8))
    d = ImageDraw.Draw(im)
    d.rectangle([W * 0.06, H * 0.02, W * 0.94, H * 0.98], outline=YELLOW, width=max(2, W // 30))
    for k, s in enumerate(["维", "修", "保", "养"]):
        _text(d, (W * 0.5, H * (0.14 + 0.24 * k)), s, int(W * 0.66), WHITE)
    return im


def _plate(W, H):
    """A blue Chinese number plate — small, but a van without one reads as a toy."""
    im = Image.new("RGB", (W, H), (16, 46, 120))
    d = ImageDraw.Draw(im)
    d.rectangle([2, 2, W - 3, H - 3], outline=WHITE, width=max(1, W // 90))
    _text(d, (W * 0.5, H * 0.52), "粤BFJ676", int(H * 0.62), WHITE, spacing=W * 0.012)
    return im


def _shutter_slat(W, H):
    """One roller-shutter panel: the horizontal slat corrugation plus its grime, as a
    tiling map rather than a decal (a shutter is the same slat 40 times)."""
    im = Image.new("RGB", (W, H), (88, 90, 92))
    d = ImageDraw.Draw(im)
    n = 14
    for k in range(n):
        y0 = H * k / n
        y1 = H * (k + 0.62) / n
        d.rectangle([0, y0, W, y1], fill=(104, 106, 108))
        d.rectangle([0, y1, W, H * (k + 1) / n], fill=(58, 60, 62))
    return im


# name -> (pixel width, pixel height, painter). The pixel aspect should match the panel's
# real-world aspect or the artwork arrives stretched.
def _fuelcap_face(w, h):
    """The printing on an inner fuel cap, set on a curve around its annulus.

    Straight off four 460 px reference photographs: warning text runs ROUND the rim, not
    across it, in a light grey that is nearly worn out, with a maker's part number set
    smaller on the opposite side. It is the single most recognisable thing about this part
    in a colour image and the synthetic caps had none of it.

    The background is the CAP'S OWN COLOUR, not black. A decal here is an albedo map, not
    an overlay: whatever it contains replaces the material inside its rectangle, and the
    rectangle covers the whole disc. Painted on black, it therefore drove the albedo of
    every unlettered part of the cap — including the inside of a recessed grip — to zero,
    and a well whose walls return no light at all is indistinguishable from no well. That
    cost four rounds of debugging geometry that was correct the whole time.
    And the values are LINEAR, as everything in this module is — 0.02 albedo is byte 5,
    not byte 38. Writing the sRGB number made the cap eight times too bright and turned a
    black moulding into a white one, which is the same class of error as painting it black
    and just as easy to make while thinking about something else."""
    im = Image.new("RGB", (w, h), (5, 5, 6))
    d = ImageDraw.Draw(im)
    cx, cy = w / 2, h / 2
    # Worn light-grey printing: linear 0.28, not the paper-white it would be in sRGB. On
    # the reference caps the ink is half rubbed away and reads well below the highlights.
    ink = (72, 72, 68)
    # SIZE, measured off the head-on frames rather than chosen: a glyph is about 0.037 of
    # the cap's diameter and the arcs run three quarters of the way round, forty-odd
    # characters between them. At 0.050 with six degrees a character — what this drew first
    # — twenty letters filled the annulus edge to edge and the printing became the loudest
    # feature of the part instead of a faint marking near the rim. It is the same error as
    # a too-big grip: at the sixty-pixel scale the sensor works at nobody could see it.
    #
    # Four arcs on two radii, top and bottom, with the sides left clear — that is the layout
    # every photographed cap in the set uses, and the clear sides are where the handle's
    # ends land. A single arc long enough to carry all the text wraps past 230 degrees and
    # collides with the inner line, which is what the first attempt at this did.
    for text, radius, start, size, step in (
            ("ATTENZIONE  RIMUOVERE PIANO", 0.425, -152.0, 0.037, 4.5),
            ("WARNING  REMOVE SLOWLY", 0.425, 30.0, 0.037, 4.5),
            ("请拧紧至听到三次响声", 0.322, -126.0, 0.030, 5.6),
            ("SGMW 8450557", 0.322, 54.0, 0.031, 5.0)):
        f = font(int(size * h))
        a = start
        for ch in text:
            r = radius * min(w, h)
            th = math.radians(a)
            gx, gy = cx + r * math.cos(th), cy + r * math.sin(th)
            g = Image.new("RGB", (int(size * h * 1.9), int(size * h * 1.9)), (0, 0, 0))
            ImageDraw.Draw(g).text((g.width / 2, g.height / 2), ch, font=f, fill=ink,
                                   anchor="mm")
            g = g.rotate(-(a + 90.0), resample=Image.BICUBIC)
            im.paste(g, (int(gx - g.width / 2), int(gy - g.height / 2)), g.convert("L"))
            a += step
    return im


def _arc_text(im, spec, ink=(72, 72, 68)):
    """Set each line of `spec` — (text, radius, centre angle, glyph size, degrees per glyph)
    — around a circle centred in `im`, glyphs turned to stand on the arc."""
    w, h = im.size
    cx, cy = w / 2, h / 2
    for text, radius, centre, size, step in spec:
        f = font(int(size * h))
        a = centre - step * (len(text) - 1) / 2.0
        for ch in text:
            r = radius * min(w, h)
            th = math.radians(a)
            gx, gy = cx + r * math.cos(th), cy + r * math.sin(th)
            g = Image.new("RGB", (int(size * h * 1.9), int(size * h * 1.9)), (0, 0, 0))
            ImageDraw.Draw(g).text((g.width / 2, g.height / 2), ch, font=f, fill=ink,
                                   anchor="mm")
            g = g.rotate(-(a + 90.0), resample=Image.BICUBIC)
            im.paste(g, (int(gx - g.width / 2), int(gy - g.height / 2)), g.convert("L"))
            a += step
    return im


def _fuelcap_boyue(w, h):
    """The printing on ONE photographed cap, transcribed rather than invented.

    `_fuelcap_face` carries a plausible cap's markings — an Italian warning and a maker's
    part number — and plausible is the wrong target when the render is going next to the
    photograph it is copying. This is what that cap actually says, in the layout it says it
    in: two concentric arcs on each wing, English outside and Chinese inside, top and bottom,
    with the middle band left clear because the handle bar crosses there.

    Both wings run the same way round the circle, so the lower one reads upside down from a
    fixed viewpoint. That is not a bug to fix — it is what the photograph shows, because the
    text is set to be read while the cap is being turned.
    """
    im = Image.new("RGB", (w, h), (5, 5, 6))
    return _arc_text(im, (
        # Sizes are glyph height as a fraction of the artwork, steps are degrees per glyph;
        # both measured off the photograph. Pillow's face is wider than the condensed one
        # a cap is tooled with, so the size is set from the text's HEIGHT and the step from
        # its arc length, and the two are not derived from each other.
        # Wing angles are in the artwork's own frame and map straight through: -90 is the
        # image's top row, and `with_decal` runs v along the part's +y, so this arc arrives
        # at the top of the cap. (Checked against the photograph at magnification after a
        # round spent "fixing" it the other way on a small composite.)
        ("TIGHTEN UNTIL THREE CLICKS", 0.430, -90.0, 0.046, 5.6),
        ("请拧紧至听到三声响", 0.337, -90.0, 0.040, 8.4),
        ("否则发动机故障灯可能点亮", 0.337, 90.0, 0.038, 7.2),
        ("OR SERVICE ENGINE LIGHT MAY TURN ON", 0.430, 90.0, 0.042, 4.4),
    ))


def _fuelcap_boyue_door(w, h):
    """Inner-door service label and restrained moulding wear from the matched BoYue L."""
    im = Image.new("RGB", (w, h), (4, 4, 5))
    d = ImageDraw.Draw(im)
    # The photographed label is a small white fuel icon and a large 92 octane callout on a
    # black pad. Keep it small: it is a scale cue, not a signboard.
    # Authored in the opposite quadrant because the finished image is rotated for the
    # camera-facing -z side below. After that turn the label sits beside the hinge, which
    # is the part of the open door that remains inside the photograph-matched crop.
    x0, y0, x1, y1 = int(w * 0.56), int(h * 0.26), int(w * 0.77), int(h * 0.44)
    d.rounded_rectangle((x0, y0, x1, y1), radius=max(2, w // 90), fill=(6, 6, 7))
    ink = (152, 152, 145)
    lw = max(2, w // 180)
    bx, by = int(w * 0.585), int(h * 0.288)
    d.rounded_rectangle((bx, by, bx + int(w * 0.043), by + int(h * 0.071)),
                        radius=lw, outline=ink, width=lw)
    d.line((bx + int(w * 0.034), by + int(h * 0.017), bx + int(w * 0.055),
            by + int(h * 0.038), bx + int(w * 0.055), by + int(h * 0.067)),
           fill=ink, width=lw)
    _text(d, (w * 0.695, h * 0.325), ">92", int(h * 0.078), ink)
    _text(d, (w * 0.670, h * 0.395), "RON", int(h * 0.024), (105, 105, 101))
    # A few faint rub marks stop the full-face decal background reading as perfectly flat.
    for k in range(7):
        yy = int(h * (0.18 + 0.095 * k))
        d.line((int(w * 0.06), yy, int(w * (0.10 + 0.035 * (k % 3))), yy + 1),
               fill=(7, 7, 8), width=1)
    # The map is viewed from the local -z side of the door. That reverses both image axes
    # relative to the +z cap decals, so author upright and turn the finished artwork once.
    return im.transpose(Image.Transpose.ROTATE_180)


_LIBRARY = {
    "fuelcap_face": (512, 512, _fuelcap_face),
    "fuelcap_boyue": (640, 640, _fuelcap_boyue),
    "fuelcap_boyue_door": (760, 620, _fuelcap_boyue_door),
    "pump_sign":    (512, 1116, _pump_sign),
    "fire_cabinet": (420, 820, _fire_cabinet),
    "fire_bucket":  (360, 300, _fire_bucket),
    "wet_floor":    (420, 520, _wet_floor),
    "wash_banner":  (560, 760, _wash_banner),
    "promo_banner": (300, 1500, _promo_banner),
    "repair_sign":  (200, 900, _repair_sign),
    "plate":        (440, 150, _plate),
    "shutter_slat": (256, 256, _shutter_slat),
}


def _path(name: str, decal_dir: Path) -> Path:
    return decal_dir / f"{name}_albedo.ppm"


def generate(name: str, decal_dir: Path = DECAL_DIR) -> dict:
    if name not in _LIBRARY:
        raise KeyError(f"unknown decal '{name}'; have {sorted(_LIBRARY)}")
    decal_dir.mkdir(parents=True, exist_ok=True)
    w, h, fn = _LIBRARY[name]
    im = fn(w, h)
    p = _path(name, decal_dir)
    im.save(p, format="PPM")
    return {"albedo": p}


def _recipe_id(name: str) -> str:
    w, h, fn = _LIBRARY[name]
    parts = [name, str(w), str(h), hashlib.sha1(fn.__code__.co_code).hexdigest()[:10]]
    parts += [repr(c) for c in (fn.__code__.co_consts or ()) if c is not None]
    for helper in (_text, font):
        parts.append(hashlib.sha1(helper.__code__.co_code).hexdigest()[:8])
    parts += [repr(c) for c in (NAVY, SIGN_NAVY, RED, ORANGE, YELLOW, WHITE, OFFWHITE, BLACK)]
    return hashlib.sha1("|".join(parts).encode()).hexdigest()[:16]


def ensure_decals(names, decal_dir: Path = DECAL_DIR) -> dict:
    """Draw any of `names` whose artwork is missing or stale; return {name: {'albedo': path}}."""
    out = {}
    for name in names:
        p = _path(name, decal_dir)
        stamp = decal_dir / f"{name}.recipe"
        want = _recipe_id(name)
        if not (p.exists() and stamp.exists() and stamp.read_text().strip() == want):
            generate(name, decal_dir)
            decal_dir.mkdir(parents=True, exist_ok=True)
            stamp.write_text(want)
        out[name] = {"albedo": p}
    return out


def main():
    for name in _LIBRARY:
        p = generate(name)
        print(f"  {name:14s} -> {p['albedo'].name}")


if __name__ == "__main__":
    main()
