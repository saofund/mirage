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
from .dataset import _render_bin, _read_pgm16, _read_pfm, _read_ppm, render_frame

OUT = Path(__file__).resolve().parent / "_out"


def _render(prog, out, eye, target, w=340, h=340, spp=64, up=(0, 0, 1), fov=0.7, ids=None):
    out.parent.mkdir(parents=True, exist_ok=True)
    js = out.with_suffix(".json")
    js.write_text(prog.to_json(), encoding="utf-8")
    args = [str(_render_bin()), "--oplog", str(js), "--out", str(out.with_suffix(".ppm")),
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
        ("cap  d78 12 flutes", P.cap(), 0.16),
        ("cap  no flutes", P.cap(flutes=0), 0.16),
        ("cap  slot grip", P.cap(grip="slot"), 0.16),
        ("cap  alu", P.cap(material=M.CAP_ALU, rib_material=M.CAP_ALU, printing=False), 0.16),
        ("handle alone", P.handle(0.076, 0.030, 0.021, 0.0066), 0.09),
        ("pressed dish", P.pressed_dish(), 0.30),
        ("well  d124 x 52", P.well(), 0.30),
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


def roi_sheet(synth_dir, n=8, seed=0):
    """The PRODUCT, next to the reference: synthetic ROI crops over real ROI crops.

    Not the full frame. The full frame is a render and it is easy to be pleased by; the ROI
    crop is the only part of it that reaches the dataset, and it is the only part worth
    judging. Real crops are rebuilt from the reference clouds through the same projection,
    so the two rows are the same operation applied to different worlds."""
    import cv2
    from .fit import REF

    def cloud_to_img(npz, size=150):
        d = np.load(npz)
        xyz, rgb, lab = d["xyz"], d["rgb"], d["label"]
        K, w, h = d["K_norm"], int(d["w"]), int(d["h"])
        fx, fy, cx, cy = K[0, 0] * w, K[1, 1] * h, K[0, 2] * w, K[1, 2] * h
        u = xyz[:, 0] / xyz[:, 2] * fx + cx
        v = xyz[:, 1] / xyz[:, 2] * fy + cy
        u0, v0 = u.min(), v.min()
        s = size / max(float(np.ptp(u)), float(np.ptp(v)), 1e-6)
        img = np.zeros((size + 1, size + 1, 3), np.uint8)
        px, py = ((u - u0) * s).astype(int), ((v - v0) * s).astype(int)
        ok = (px >= 0) & (px <= size) & (py >= 0) & (py <= size)
        img[py[ok], px[ok]] = rgb[ok][:, ::-1]
        m = lab.astype(bool) & ok
        edge = np.zeros(img.shape[:2], np.uint8)
        edge[py[m], px[m]] = 255
        cnt, _ = cv2.findContours(cv2.morphologyEx(edge, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8)),
                                  cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(img, cnt, -1, (0, 255, 0), 1)
        return img

    real = sorted((Path(REF) / "orbbec_clouds").glob("[!_]*.npz"))
    synth = sorted(Path(synth_dir).glob("[!_]*.npz"))
    rng = np.random.default_rng(seed)
    rows = []
    for src in (real, synth):
        if not src:
            continue
        pick = [src[i] for i in rng.choice(len(src), min(n, len(src)), replace=False)]
        rows.append(np.hstack([cloud_to_img(p) for p in pick]))
    wmin = min(r.shape[1] for r in rows)
    sheet = np.vstack([r[:, :wmin] for r in rows])
    cv2.putText(sheet, "REAL", (6, 18), 0, 0.5, (60, 255, 255), 1)
    cv2.putText(sheet, "SYNTH", (6, 169), 0, 0.5, (60, 255, 255), 1)
    OUT.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(OUT / "roi.png"), sheet)
    print(OUT / "roi.png", sheet.shape)


def depth_sheet(synth_dir, n=6, seed=0, span=(-45.0, 20.0)):
    """Real depth against synthetic depth, same colour scale — the honest side-by-side.

    Statistics can agree while the shapes disagree; that has happened twice on this case.
    A depth map cannot hide it. Both rows are rendered the same way: rotate each cloud into
    its own cap's frame, colour by height above the cap face in millimetres, and hold the
    scale fixed so a colour means the same depth in both rows.

    Height above the CAP, not raw camera distance: raw distance is dominated by how far
    away the camera happened to be, so two pictures of the same pocket at 0.35 and 0.55 m
    look nothing alike and two pictures of different pockets at the same range look
    identical. It is the wrong variable to compare on."""
    import cv2
    from .fit import REF

    def height_map(npz, size=190):
        d = np.load(npz)
        P = d["xyz"].astype(np.float64)
        cap = P[d["label"] == 1]
        if len(cap) < 400:
            return None
        c = cap.mean(0)
        _, _, V = np.linalg.svd(cap - c, full_matrices=False)
        nrm = V[2]
        if nrm[2] > 0:
            nrm = -nrm
        K, w, h = d["K_norm"], int(d["w"]), int(d["h"])
        fx, fy = K[0, 0] * w, K[1, 1] * h
        cx, cy = K[0, 2] * w, K[1, 2] * h
        uf = P[:, 0] / P[:, 2] * fx + cx
        vf = P[:, 1] / P[:, 2] * fy + cy
        half = float(np.median(np.abs(np.mod(uf, 1.0) - 0.5))) < 0.25   # see fit.complexity
        u = np.floor(uf if half else uf + 0.5).astype(int)
        v = np.floor(vf if half else vf + 0.5).astype(int)
        z = (P - c) @ nrm * 1000.0
        u -= u.min(); v -= v.min()
        H, W = v.max() + 1, u.max() + 1
        if H < 8 or W < 8 or H * W > 4_000_000:
            return None
        img = np.full((H, W), np.nan, np.float32)
        img[v, u] = z
        t = np.clip((img - span[0]) / (span[1] - span[0]), 0, 1)
        col = cv2.applyColorMap((np.nan_to_num(t) * 255).astype(np.uint8), cv2.COLORMAP_TURBO)
        col[np.isnan(img)] = 0
        s = size / max(H, W)
        col = cv2.resize(col, (max(1, int(W * s)), max(1, int(H * s))),
                         interpolation=cv2.INTER_NEAREST)
        tile = np.zeros((size, size, 3), np.uint8)
        tile[:col.shape[0], :col.shape[1]] = col
        return tile

    rng = np.random.default_rng(seed)
    rows = []
    for src, lab in (((Path(REF) / "orbbec_clouds"), "REAL"), (Path(synth_dir), "SYNTH")):
        fs = sorted(src.glob("[!_]*.npz"))
        if not fs:
            continue
        pick = [fs[i] for i in rng.choice(len(fs), min(n * 3, len(fs)), replace=False)]
        tiles = [t for t in (height_map(p) for p in pick) if t is not None][:n]
        if tiles:
            row = np.hstack(tiles)
            cv2.putText(row, lab, (6, 18), 0, 0.5, (255, 255, 255), 1)
            rows.append(row)
    if not rows:
        print("nothing to draw")
        return
    wmin = min(r.shape[1] for r in rows)
    sheet = np.vstack([r[:, :wmin] for r in rows])
    # a colour key, so the picture is readable without the source
    bar = np.zeros((26, wmin, 3), np.uint8)
    g = np.linspace(0, 255, wmin).astype(np.uint8)
    bar[:] = cv2.applyColorMap(np.tile(g, (26, 1)), cv2.COLORMAP_TURBO)
    for frac, mm in ((0.02, span[0]), (0.5, (span[0] + span[1]) / 2), (0.95, span[1])):
        cv2.putText(bar, f"{mm:+.0f}mm", (int(frac * wmin), 18), 0, 0.45, (0, 0, 0), 1)
    OUT.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(OUT / "depth.png"), np.vstack([sheet, bar]))
    print(OUT / "depth.png", sheet.shape)


def _real_crops(n=8, size=300, cls=1, pad=0.30, skip=0):
    """Big crops of the annotated inner cap, one per car, straight off the reference set.

    The row this returns is the only thing worth scoring a render against. Every metric in
    `fit.py` measures the CLOUD, and a cloud cannot tell you that a cap's grip is a plain
    bar where the real one is a waisted handle with a dished top, or that the skirt has
    flutes. Those are the differences somebody looking at the two pictures sees first."""
    import glob
    import cv2
    from .fit import REF

    def imread(p):                     # cv2.imread cannot open the CJK paths in bycar/
        return cv2.imdecode(np.fromfile(p, np.uint8), cv2.IMREAD_COLOR)

    out, seen = [], set()
    for png in sorted(glob.glob(os.path.join(REF, "bycar", "*", "*.png"))):
        car = os.path.basename(os.path.dirname(png))
        txt = png[:-4] + ".txt"
        if car in seen or not os.path.exists(txt):
            continue
        poly = None
        for line in open(txt):
            p = line.split()
            if p and int(p[0]) == cls:
                poly = np.array([float(x) for x in p[1:]], np.float32).reshape(-1, 2)
                break
        im = imread(png) if poly is not None else None
        if im is None:
            continue
        H, W = im.shape[:2]
        x, y, w, h = cv2.boundingRect((poly * [W, H]).astype(np.int32))
        d = int(pad * max(w, h))
        c = im[max(0, y - d):min(H, y + h + d), max(0, x - d):min(W, x + w + d)]
        if c.size == 0 or min(c.shape[:2]) < 24:
            continue
        seen.add(car)
        out.append(cv2.resize(c, (size, size)))
    return out[skip:skip + n]


def closeup_sheet(n=8, seed=11, size=300, real=True, skip=0):
    """Synthetic pockets framed the way a reference photograph is framed, over the real ones.

    The scene sheets render a whole frame at the distance the sensor works at, where the cap
    is sixty pixels across and everything looks approximately right. Nothing in this case
    was ever judged at the magnification the *reference photographs* are at, which is the
    magnification the person looking at them judges at — so every detail finer than the
    silhouette went unchecked for as long as this case has existed.

    The camera is placed so the cap subtends the same fraction of the frame as the crop
    above it: same framing, same apparent size, so the two rows differ only by what is
    actually being photographed."""
    import cv2
    tmp = OUT / "_closeup"
    tmp.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    tiles = []
    while len(tiles) < n:
        v = S.sample(rng, domain="prod")
        # Frame ON the cap, not on the pocket: the crop above is the cap's own bounding box
        # padded by 30%, so the cap has to fill 1/1.6 of the frame here too.
        v["dist"], v["aim_off"] = 0.40, [0.0, 0.0]
        v["obliq"] = float(rng.uniform(0.0, 22.0))
        prog, gt = S.build(v)
        js = tmp / "scene.json"
        js.write_text(prog.to_json(), encoding="utf-8")
        fov = 2.0 * math.atan(v["d_cap"] * 0.80 / v["dist"])
        img = _render(prog, tmp / f"c{len(tiles)}", eye=gt["eye"], target=gt["target"],
                      w=size, h=size, spp=96, up=gt["up"], fov=fov)
        tiles.append(img)
    row = np.hstack([t[:, :, ::-1] for t in tiles])
    cv2.putText(row, "SYNTH", (8, 24), 0, 0.6, (60, 255, 255), 2)
    # The synth row is written on its own because the reference photographs are gitignored
    # and therefore are NOT on the build box: the render happens there, the comparison
    # happens here. `compose` does the second half.
    cv2.imwrite(str(OUT / "closeup_synth.png"), row)
    print(OUT / "closeup_synth.png")
    if real:
        compose_sheet(size, skip)


def compose_sheet(size=300, skip=0):
    """Stack the (possibly remotely rendered) synth row under a row of real crops."""
    import cv2
    row = cv2.imread(str(OUT / "closeup_synth.png"))
    if row is None:
        print("no closeup_synth.png yet")
        return
    rr = _real_crops(row.shape[1] // size, size, skip=skip)
    if not rr:
        print("no reference crops")
        return
    top = np.hstack(rr)
    cv2.putText(top, "REAL", (8, 24), 0, 0.6, (60, 255, 255), 2)
    w = min(top.shape[1], row.shape[1])
    cv2.imwrite(str(OUT / "closeup.png"), np.vstack([top[:, :w], row[:, :w]]))
    print(OUT / "closeup.png")


def cap_sheet(size=380):
    """The cap alone, big, from the three angles the reference photographs are taken from.

    Rendered against a mid grey rather than in its pocket. The pocket is a light trap by
    design, so a part inspected inside it is inspected in the dark — which is exactly how a
    grip that is the wrong shape survives being looked at."""
    tmp = OUT / "_cap"
    variants = [
        ("12 flutes", dict()),
        ("plain wall", dict(flutes=0)),
        ("slot grip", dict(grip="slot")),
        ("alu", dict(material=M.CAP_ALU, rib_material=M.CAP_ALU, printing=False)),
    ]
    tiles = []
    for name, kw in variants:
        prog = P.cap(**kw)
        for ang, tag in ((0.10, "top"), (0.62, "oblique")):
            eye = (0.16 * math.sin(ang) * 0.6, -0.16 * math.sin(ang),
                   0.16 * math.cos(ang))
            img = _render(prog, tmp / f"{name.split()[0]}_{tag}", eye=eye,
                          target=(0, 0, 0.002), w=size, h=size, spp=120, fov=0.62)
            tiles.append((img, f"{name}  {tag}"))
    _grid(tiles, 4, OUT / "cap.png", cell=size)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("what", choices=("parts", "scenes", "ids", "roi", "depth",
                                     "closeup", "cap", "compose"))
    ap.add_argument("--synth", default=None, help="a generated dataset dir (for roi)")
    ap.add_argument("-n", type=int, default=6)
    ap.add_argument("--seed", type=int, default=3)
    ap.add_argument("--domain", default="orbbec")
    ap.add_argument("--skip", type=int, default=0, help="offset into the real-crop row")
    a = ap.parse_args(argv)
    OUT.mkdir(parents=True, exist_ok=True)
    if a.what == "closeup":
        closeup_sheet(a.n, a.seed, skip=a.skip)
    elif a.what == "compose":
        compose_sheet(skip=a.skip)
    elif a.what == "cap":
        cap_sheet()
    elif a.what == "parts":
        parts_sheet()
    elif a.what == "scenes":
        scenes_sheet(a.n, a.seed, a.domain)
    elif a.what == "roi":
        roi_sheet(a.synth, a.n, a.seed)
    elif a.what == "depth":
        depth_sheet(a.synth, a.n, a.seed)
    else:
        ids_sheet(a.seed, max(1, a.n // 3), a.domain)


if __name__ == "__main__":
    main()
