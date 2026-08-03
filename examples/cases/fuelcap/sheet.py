"""Look at the parts ALONE, and at the assembly's id map — the two things a beauty render
will not tell you.

A composed render is the worst place to debug a part. Everything in this pocket is black
plastic inside a black hole, so a part that is the wrong size, inside-out, or simply
missing looks very much like a part that is right: the frame is dark either way. Rendering
each part on its own against a neutral background, at a known scale, is what makes a
missing rib or an inverted cup obvious in one look.

    python -m fuelcap.sheet parts          # every part alone, labelled, to scale
    python -m fuelcap.sheet scenes -n 6    # six random assemblies
    python -m fuelcap.sheet ids            # the assembly, coloured by object id
"""
from __future__ import annotations

import argparse, math, os, subprocess, sys, tempfile
from pathlib import Path

import numpy as np

from mirage.meshlang import MeshProgram

from . import materials as M
from . import parts as P
from . import scene as S
from .dataset import RENDER, _read_pgm16, _read_pfm, _read_ppm, render_frame

OUT = Path(__file__).resolve().parent / "_out"


def _render(prog, out, eye, target, w=340, h=340, spp=64, up=(0, 0, 1), fov=0.7, ids=None):
    out.parent.mkdir(parents=True, exist_ok=True)
    js = out.with_suffix(".json")
    js.write_text(prog.to_json(), encoding="utf-8")
    args = [str(RENDER), "--oplog", str(js), "--out", str(out.with_suffix(".ppm")),
            "--w", str(w), "--h", str(h), "--spp", str(spp), "--denoise", "4",
            "--cam-eye", *[f"{x:.5f}" for x in eye],
            "--cam-target", *[f"{x:.5f}" for x in target],
            "--cam-up", *[f"{x:.5f}" for x in up],
            "--cam-fov", f"{fov:.4f}", "--no-ground", "--env", "1.1", "--sun", "2.2",
            "--sun-dir", "0.4", "-0.5", "0.75", "--sky-flat", "0.4", "--smooth-angle", "24"]
    if ids:
        args += ["--ids", str(out.with_suffix(".pgm")), "--id-tags", ",".join(ids)]
    r = subprocess.run(args, capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(r.stderr[:300])
    return _read_ppm(out.with_suffix(".ppm"))


def _grid(tiles, cols, out, cell=340):
    import cv2
    rows = (len(tiles) + cols - 1) // cols
    sheet = np.zeros((rows * cell, cols * cell, 3), np.uint8)
    for i, (img, label) in enumerate(tiles):
        r, c = divmod(i, cols)
        t = img.copy()
        cv2.putText(t, label, (8, 22), 0, 0.5, (255, 255, 40), 1, cv2.LINE_AA)
        sheet[r * cell:(r + 1) * cell, c * cell:(c + 1) * cell] = t[:, :, ::-1]
    cv2.imwrite(str(out), sheet)
    print(out, sheet.shape)


def parts_sheet():
    """Every part alone. The camera distance is fixed per part so the SCALE is readable:
    two parts side by side are at the same magnification only if you make them so."""
    tmp = OUT / "_parts"
    items = [
        ("cap  d78 plain", P.cap(), 0.16),
        ("cap  teeth+slot", P.cap(teeth=28, rib_slot=0.004, rib_draft=0.58), 0.16),
        ("cap  wide rib", P.cap(d=0.088, rib_len=0.068, rib_w=0.034, rib_h=0.015), 0.16),
        ("cap  alu", P.cap(material=M.CAP_ALU, rib_material=M.CAP_ALU), 0.16),
        ("well  d124 x 52", P.well(), 0.30),
        ("well  shallow", P.well(depth=0.030, neck=False), 0.30),
        ("panel  aperture", P.panel(size=0.30, hole_d=0.124), 0.45),
        ("seal", P.seal(), 0.14),
        ("door  open 95", P.door(), 0.45),
        ("tether", P.tether((0.0, 0.0, 0.0), (0.075, 0.012, -0.02)), 0.20),
    ]
    tiles = []
    for name, prog, dist in items:
        img = _render(prog, tmp / name.split()[0], eye=(dist * 0.55, -dist * 0.72, dist * 0.60),
                      target=(0, 0, -0.01), fov=0.75)
        tiles.append((img, name))
    _grid(tiles, 5, OUT / "parts.png")


def scenes_sheet(n=6, seed=3, domain="orbbec"):
    tmp = OUT / "_scenes"
    tmp.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    tiles = []
    for i in range(n):
        v = S.sample(rng, domain=domain)
        prog, gt = S.build(v)
        (tmp / "scene.json").write_text(prog.to_json(), encoding="utf-8")
        rgb, ids, depth = render_frame(v, gt, tmp, spp=64)
        import cv2
        tiles.append((cv2.resize(rgb, (340, 255)),
                      f"d={v['dist']:.2f} obl={v['obliq']:.0f} {v['paint']}"))
    import cv2
    rows = (len(tiles) + 2) // 3
    sheet = np.zeros((rows * 255, 3 * 340, 3), np.uint8)
    for i, (img, label) in enumerate(tiles):
        r, c = divmod(i, 3)
        t = img.copy()
        cv2.putText(t, label, (6, 18), 0, 0.45, (255, 255, 40), 1, cv2.LINE_AA)
        sheet[r * 255:(r + 1) * 255, c * 340:(c + 1) * 340] = t[:, :, ::-1]
    cv2.imwrite(str(OUT / "scenes.png"), sheet)
    print(OUT / "scenes.png", sheet.shape)


def ids_sheet(seed=0, n=3, domain="orbbec"):
    """The assembly coloured by object id — which pixel belongs to which part.

    This is the sheet that finds the errors a beauty render hides: a cap that is really the
    neck ring, a rib that never got placed, a well the camera is seeing the *outside* of."""
    import cv2
    tmp = OUT / "_ids"
    tmp.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    palette = np.array([[24, 24, 28]] + [
        [255, 90, 90], [255, 210, 40], [120, 255, 120], [90, 200, 255],
        [200, 120, 255], [255, 150, 60], [60, 255, 210], [160, 160, 160],
        [90, 130, 255], [255, 255, 255]], np.uint8)
    tiles = []
    for i in range(n):
        v = S.sample(rng, domain=domain)
        prog, gt = S.build(v)
        (tmp / "scene.json").write_text(prog.to_json(), encoding="utf-8")
        rgb, ids, depth = render_frame(v, gt, tmp, spp=24)
        col = palette[np.clip(ids, 0, len(palette) - 1)]
        tiles.append((cv2.resize(rgb, (340, 255)), f"rgb  d={v['dist']:.2f}"))
        tiles.append((cv2.resize(col, (340, 255), interpolation=cv2.INTER_NEAREST), "ids"))
        dv = depth.copy()
        m = dv > 0
        if m.any():
            dv = np.clip((dv - dv[m].min()) / max(1e-6, np.ptp(dv[m])), 0, 1)
        dv = cv2.applyColorMap((dv * 255).astype(np.uint8), cv2.COLORMAP_TURBO)
        dv[~m] = 0
        tiles.append((cv2.resize(dv[:, :, ::-1], (340, 255), interpolation=cv2.INTER_NEAREST),
                      "depth"))
    rows = len(tiles) // 3
    sheet = np.zeros((rows * 255, 3 * 340, 3), np.uint8)
    for i, (img, label) in enumerate(tiles):
        r, c = divmod(i, 3)
        t = img.copy()
        cv2.putText(t, label, (6, 18), 0, 0.45, (255, 255, 40), 1, cv2.LINE_AA)
        sheet[r * 255:(r + 1) * 255, c * 340:(c + 1) * 340] = t[:, :, ::-1]
    cv2.imwrite(str(OUT / "ids.png"), sheet)
    print(OUT / "ids.png", sheet.shape)
    print("id legend:", {k + 1: t for k, t in enumerate(S.ID_TAGS)})


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("what", choices=("parts", "scenes", "ids"))
    ap.add_argument("-n", type=int, default=6)
    ap.add_argument("--seed", type=int, default=3)
    ap.add_argument("--domain", default="orbbec")
    a = ap.parse_args(argv)
    OUT.mkdir(parents=True, exist_ok=True)
    if a.what == "parts":
        parts_sheet()
    elif a.what == "scenes":
        scenes_sheet(a.n, a.seed, a.domain)
    else:
        ids_sheet(a.seed, max(1, a.n // 3), a.domain)


if __name__ == "__main__":
    main()
