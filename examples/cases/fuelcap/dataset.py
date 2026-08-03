"""Render randomised pockets into the npz a real capture becomes — infinitely.

The output is deliberately not "images plus a json". It is the *same file* the real
pipeline already consumes: one ``.npz`` per frame carrying ``xyz, rgb, label, K_norm,
w, h`` (plus ``anchor`` and the pose labels), cropped to the cap's ROI the way
``build_clouds.py --roi-class 1 --crop-pad 2.0`` crops it. Drop the directory into
``datasets/``, add a line to ``registry.json``, and the annotation server and the training
loader both open it without knowing it was never photographed.

The part that decides whether any of this transfers is the **sensor model**, not the
renderer. A path tracer hands back a perfect depth map; a real Orbbec hands back a surface
plus 0.4 mm of axial scatter, a third of its pixels missing, and a fringe of flying pixels
along every depth discontinuity. Those three are measured off the real clouds in `fit.py`
and reproduced here:

    axial scatter   0.32 mm (prod) / 0.44 mm (orbbec), roughly constant over 0.3-1.3 m
    ROI fill        0.68 of ROI pixels carry depth in the orbbec set
    ROI extent      3.9x the cap's own bounding box

Leave any of the three out and a classifier can separate real from synthetic in one line
of numpy, which means a pose network can too — and will, because it is the easier signal.
"""
from __future__ import annotations

import argparse, json, math, os, shutil, subprocess, sys, time
from pathlib import Path

import numpy as np

from . import scene as S

ROOT = Path(__file__).resolve().parents[3]
RENDER = ROOT / "core" / "build" / "Release" / "mirage_render.exe"
if not RENDER.exists():
    RENDER = ROOT / "core" / "build" / "mirage_render"          # posix build


# --------------------------------------------------------------------------- #
# image io — the renderer's three AOVs
# --------------------------------------------------------------------------- #
def _read_ppm(p):
    with open(p, "rb") as f:
        assert f.readline().strip() == b"P6"
        w, h = (int(x) for x in f.readline().split())
        f.readline()
        return np.frombuffer(f.read(w * h * 3), np.uint8).reshape(h, w, 3)


def _read_pgm16(p):
    with open(p, "rb") as f:
        assert f.readline().strip() == b"P5"
        w, h = (int(x) for x in f.readline().split())
        f.readline()
        return np.frombuffer(f.read(w * h * 2), ">u2").reshape(h, w).astype(np.int32)


def _read_pfm(p):
    with open(p, "rb") as f:
        assert f.readline().strip() == b"Pf"
        w, h = (int(x) for x in f.readline().split())
        scale = float(f.readline())
        d = np.frombuffer(f.read(w * h * 4), "<f4" if scale < 0 else ">f4").reshape(h, w)
        return d[::-1].copy()                    # PFM is stored bottom-up


# --------------------------------------------------------------------------- #
# the sensor
# --------------------------------------------------------------------------- #
def degrade_depth(depth, rng, sigma_mm=0.42, jump_mm=12.0, flying=0.35, edge_px=1.6,
                  quant_mm=0.0):
    """Turn a perfect depth map into one a stereo camera would have produced.

    Four effects, in the order they happen in a real pipeline:

    * **axial scatter** — gaussian along the view axis. Measured at 0.32-0.44 mm on both
      real sets and, notably, *flat* with range over 0.3-1.3 m, so it is applied flat here
      rather than as the textbook ``k*z^2``. Fitting the textbook form to this data gives
      three different k's for three sets, which is the fit telling you it is the wrong
      model.
    * **flying pixels** — at a depth discontinuity a correlation window straddles both
      surfaces and returns something between them. This is the single most recognisable
      signature of real stereo depth, and the reason a synthetic cloud's edges look
      surgically clean.
    * **edge erosion** — the same window is also simply wrong near the jump, and most
      pipelines throw those pixels away.
    * **dropout** — everything else that fails: dark surfaces returning no texture, specular
      glare, shadowed pockets. Biased toward the dark, because that is where it happens.

    ``quant_mm`` is off by default. Disparity quantisation is the *physically* right model
    for the axial term, but it needs a baseline this dataset does not record, and the
    measured noise is flat with range where quantisation would grow quadratically."""
    d = depth.astype(np.float32).copy()
    valid = d > 0

    gx = np.zeros_like(d); gy = np.zeros_like(d)
    gx[:, 1:] = np.abs(d[:, 1:] - d[:, :-1])
    gy[1:, :] = np.abs(d[1:, :] - d[:-1, :])
    gx[~valid] = 0; gy[~valid] = 0
    grad = np.maximum(gx, gy)
    # What counts as a discontinuity has to be set against THIS object's own relief, not
    # against a generic "a few millimetres". At 0.4 m the grip rib's drafted flank falls
    # about 4 mm per pixel — so a 4 mm threshold flags the rib's own sides as occlusion
    # boundaries and the erosion below then eats the rib, which cost the cap half its
    # points. 12 mm keeps only the true occlusion edges: the cap's rim against the well
    # floor, and the well against the panel.
    jump = grad > jump_mm * 1e-3

    if flying > 0 and jump.any():
        # pull edge pixels toward their neighbour across the jump
        shifted = np.roll(d, 1, axis=1)
        pick = jump & valid & (np.roll(valid, 1, axis=1)) & (rng.random(d.shape) < flying)
        t = rng.random(d.shape).astype(np.float32)
        d[pick] = d[pick] * (1 - t[pick]) + shifted[pick] * t[pick]

    if edge_px > 0:
        k = int(max(1, round(edge_px)))
        fat = jump.copy()
        for _ in range(k):
            fat[:, 1:] |= fat[:, :-1]; fat[:, :-1] |= fat[:, 1:]
            fat[1:, :] |= fat[:-1, :]; fat[:-1, :] |= fat[1:, :]
        valid &= ~(fat & (rng.random(d.shape) < 0.55))

    if quant_mm > 0:
        d = np.round(d / (quant_mm * 1e-3)) * (quant_mm * 1e-3)
    d += rng.normal(0.0, sigma_mm * 1e-3, d.shape).astype(np.float32)

    return d, valid


# Survival rate against LOCAL TEXTURE, measured on 80 real orbbec frames by projecting
# each cloud back onto its own source image (`fit.dropout_vs_texture`). The bins are the
# local standard deviation of luminance in a 5x5 window — the quantity a block matcher's
# correlation score actually depends on.
#
# Two earlier models were wrong here and the measurement caught both. "Dark surfaces
# return no signal" is backwards: the matt black plastic down the pocket is the easiest
# surface in the frame. Brightness alone is better but still a proxy, and it predicts the
# cap should drop out MORE than the paint because the cap is darker. Texture is the thing:
# a glossy body panel is a smooth ramp with nothing to match, and a moulded cap covered in
# text and scuffs and shading detail survives at 0.68 where the panel survives at 0.53.
TEXTURE_EDGES = np.array([0.0, 0.004, 0.008, 0.015, 0.028, 0.050, 1e9])
SURVIVAL = np.array([0.529, 0.570, 0.586, 0.587, 0.575, 0.679])


def _box_mean(a, win):
    """Box mean over a win x win window, edges clamped. numpy only.

    Deliberately not cv2.blur, even though the calibration curve in `fit.py` is measured
    with cv2.blur: generating data is the product and reviewing it is a tool, and the
    product should not need OpenCV installed. Checked against it — bit-identical over the
    interior, differing only on the 2-pixel border, where cv2 reflects and this clamps."""
    p = win // 2
    b = np.pad(a, p, mode="edge").astype(np.float64)
    c = b.cumsum(0).cumsum(1)
    c = np.pad(c, ((1, 0), (1, 0)))
    s = c[win:, win:] - c[:-win, win:] - c[win:, :-win] + c[:-win, :-win]
    return (s / (win * win)).astype(np.float32)


def _texture(rgb, win=5):
    """Local luminance standard deviation, the same way `fit.dropout_vs_texture` measures
    it on the real frames — the two have to agree or the calibration means nothing."""
    lum = rgb.astype(np.float32).mean(-1) / 255.0
    mu = _box_mean(lum, win)
    return np.sqrt(np.maximum(_box_mean(lum * lum, win) - mu * mu, 0.0))


def _dropout(valid, rgb, rng, fill=0.60, blob=0.35):
    """Thin the valid mask to the measured ROI fill, following the measured curve."""
    if fill >= 1.0:
        return valid
    if not valid.any():
        return valid
    # Real dropouts arrive in patches, not as salt and pepper: a correlation window that
    # fails, fails for its whole neighbourhood. Eroding a random field twice gives patches
    # a few pixels across, which is the scale the reference frames show. The patches are
    # drawn FIRST so the per-pixel curve can be rescaled to land on `fill` including them —
    # rescaling first and then eroding overshoots the dropout by whatever the patches cost.
    lost = np.zeros(valid.shape, bool)
    if blob > 0:
        lost = rng.random(valid.shape) < blob
        for _ in range(2):
            lost[:, 1:] &= lost[:, :-1]
            lost[1:, :] &= lost[:-1, :]
    sd = _texture(rgb)
    curve = SURVIVAL[np.clip(np.digitize(sd, TEXTURE_EDGES) - 1, 0, len(SURVIVAL) - 1)]
    live = valid & ~lost
    got = float(curve[live].mean() * live.sum() / max(valid.sum(), 1)) if live.any() else 1.0
    curve = np.clip(curve * (fill / max(got, 1e-6)), 0.0, 1.0)
    return live & (rng.random(valid.shape) < curve)


# --------------------------------------------------------------------------- #
# one frame
# --------------------------------------------------------------------------- #
def render_frame(v, gt, work: Path, spp=48, threads=0, denoise=4):
    """Run the tracer for one variant. Returns (rgb, ids, depth)."""
    prog, _ = None, None
    oplog = work / "scene.json"
    cam = gt
    args = [str(RENDER), "--oplog", str(oplog), "--out", str(work / "rgb.ppm"),
            "--depth", str(work / "d.pfm"), "--ids", str(work / "ids.pgm"),
            "--id-tags", ",".join(S.ID_TAGS),
            "--w", str(v_w(v)), "--h", str(v_h(v)), "--spp", str(spp),
            "--cam-eye", *[f"{x:.6f}" for x in cam["eye"]],
            "--cam-target", *[f"{x:.6f}" for x in cam["target"]],
            "--cam-up", *[f"{x:.6f}" for x in cam["up"]],
            "--cam-fov", f"{cam['fov_y']:.6f}",
            "--no-ground", "--bounce", "5",
            "--sun", f"{v['sun']:.3f}", "--env", f"{v['env']:.3f}",
            "--sky-flat", f"{v['sky_flat']:.2f}",
            "--sky-tint", *[f"{c:.3f}" for c in v["sky_tint"]],
            "--exposure", f"{v['exposure']:.3f}", "--smooth-angle", "24"]
    el, az = math.radians(v["sun_el"]), math.radians(v["sun_az"])
    args += ["--sun-dir", f"{math.cos(el)*math.cos(az):.4f}",
             f"{math.cos(el)*math.sin(az):.4f}", f"{math.sin(el):.4f}"]
    if denoise:
        args += ["--denoise", str(denoise)]
    if threads:
        args += ["--threads", str(threads)]
    r = subprocess.run(args, capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(f"render failed: {r.stderr[:400]}")
    return (_read_ppm(work / "rgb.ppm"), _read_pgm16(work / "ids.pgm"),
            _read_pfm(work / "d.pfm"))


def v_w(v):
    return S.CAMERAS[v["camera"]]["w"]


def v_h(v):
    return S.CAMERAS[v["camera"]]["h"]


def world_to_cam(gt):
    """The camera's rotation, world -> OpenCV camera (x right, y down, z forward).

    OpenCV and not the renderer's own basis, because that is the frame the real clouds are
    in — their y really does point down, which is why every measured cap normal in the
    production set has a negative z."""
    eye = np.array(gt["eye"]); tgt = np.array(gt["target"]); up = np.array(gt["up"])
    f = tgt - eye; f /= np.linalg.norm(f)
    right = np.cross(f, up); right /= np.linalg.norm(right)
    up2 = np.cross(right, f)
    return np.stack([right, -up2, f]), eye        # rows: x_cv, y_cv, z_cv


def make_cloud(rgb, ids, depth, gt, rng, cfg):
    """Unproject to a labelled ROI cloud in camera coordinates — the real npz's contents."""
    h, w = depth.shape
    K = gt["K"]
    # The renderer is a centred pinhole, so the intrinsics that describe THIS image are
    # fx = fy = (h/2)/tan(fov/2) with the principal point at the exact centre — 1.2 px from
    # the real camera's. Emitting the real cx/cy instead would be a lie about this image:
    # the cloud would be unprojected with one K and labelled with another.
    fy = (h / 2.0) / math.tan(gt["fov_y"] / 2.0)
    fx, cx, cy = fy, w / 2.0, h / 2.0

    cap = np.isin(ids, np.arange(1, len(S.CAP_TAGS) + 1))
    if cap.sum() < 60:
        return None
    ys, xs = np.nonzero(cap)
    # ROI: the cap's bbox blown up to the 3.9x the real build_clouds crop produces
    cxm, cym = 0.5 * (xs.min() + xs.max()), 0.5 * (ys.min() + ys.max())
    half = 0.5 * max(xs.max() - xs.min(), ys.max() - ys.min()) * cfg["roi_scale"]
    x0, x1 = int(max(0, cxm - half)), int(min(w, cxm + half + 1))
    y0, y1 = int(max(0, cym - half)), int(min(h, cym + half + 1))
    if x1 - x0 < 24 or y1 - y0 < 24:
        return None

    d, valid = degrade_depth(depth, rng, sigma_mm=cfg["sigma_mm"], jump_mm=cfg["jump_mm"],
                             flying=cfg["flying"], edge_px=cfg["edge_px"],
                             quant_mm=cfg["quant_mm"])
    valid = _dropout(valid, rgb, rng, fill=cfg["fill"])
    d, valid = d[y0:y1, x0:x1], valid[y0:y1, x0:x1]
    sub_rgb, sub_ids = rgb[y0:y1, x0:x1], ids[y0:y1, x0:x1]

    yy, xx = np.mgrid[y0:y1, x0:x1]
    Z = d
    X = (xx + 0.5 - cx) / fx * Z
    Y = (yy + 0.5 - cy) / fy * Z
    m = valid & (Z > 0.05)
    if m.sum() < 200:
        return None
    xyz = np.stack([X[m], Y[m], Z[m]], 1).astype(np.float32)
    col = sub_rgb[m].astype(np.uint8)
    lab = np.isin(sub_ids[m], np.arange(1, len(S.CAP_TAGS) + 1)).astype(np.uint8)

    Rwc, eye = world_to_cam(gt)
    n_cam = Rwc @ np.array(gt["cap_normal"])
    if n_cam[2] > 0:                              # face the camera, as every real label does
        n_cam = -n_cam
    c_cam = Rwc @ (np.array(gt["cap_centre"]) - eye)
    x_cam = Rwc @ np.array(gt["cap_x"])

    # the knob OBB, straight off the rib's own id — free ground truth for what is a whole
    # separate detection model in the real pipeline
    rib = np.nonzero(sub_ids == S.RIB_ID)
    obb = None
    if len(rib[0]) > 20:
        pr = np.stack([rib[1] + x0, rib[0] + y0], 1).astype(np.float32)
        c = pr.mean(0)
        u, s, vt = np.linalg.svd(pr - c, full_matrices=False)
        e = (pr - c) @ vt.T
        half_e = np.array([np.abs(e[:, 0]).max(), np.abs(e[:, 1]).max()])
        corners = np.array([[-1, -1], [1, -1], [1, 1], [-1, 1]], np.float32) * half_e
        obb = (corners @ vt + c).astype(np.int32)

    K_norm = np.array([[fx / w, 0, cx / w], [0, fy / h, cy / h], [0, 0, 1]], np.float64)
    out = dict(xyz=xyz, rgb=col, label=lab, K_norm=K_norm,
               w=np.int64(w), h=np.int64(h), scale=np.float32(1.0),
               anchor=c_cam.astype(np.float32),
               normal=n_cam.astype(np.float32),          # THE label: cap face normal
               cap_x=x_cam.astype(np.float32),           # + the in-plane axis => full 6D
               roi=np.array([x0, y0, x1, y1], np.int32))
    if obb is not None:
        out["obb_px"] = obb
    return out, (x0, y0, x1, y1)


# --------------------------------------------------------------------------- #
# the loop
# --------------------------------------------------------------------------- #
DEFAULT_CFG = dict(roi_scale=3.9, fill=0.60, sigma_mm=0.42, jump_mm=12.0, flying=0.35,
                   edge_px=1.6, quant_mm=0.0,
                   rgb_grain=0.012, rgb_chroma_blur=1.2, rgb_saturation=1.12)


def generate(out_dir, n=32, seed=0, camera="orbbec640", domain="wide", spp=48, threads=0,
             cfg=None, keep_png=0, quiet=False):
    cfg = {**DEFAULT_CFG, **(cfg or {})}
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    work = out / "_work"
    work.mkdir(exist_ok=True)
    rng = np.random.default_rng(seed)
    from mirage import sensor

    files, normals, anchors, meta, src_map = [], [], [], [], {}
    t0 = time.time()
    made = 0
    for i in range(n):
        v = S.sample(rng, camera=camera, domain=domain)
        prog, gt = S.build(v)
        (work / "scene.json").write_text(prog.to_json(), encoding="utf-8")
        try:
            rgb, ids, depth = render_frame(v, gt, work, spp=spp, threads=threads)
        except Exception as e:
            if not quiet:
                print(f"  [{i}] render failed: {e}")
            continue
        # the imaging chain: a path tracer's output has never been through a camera
        rgb = (sensor.apply(rgb, grain=cfg["rgb_grain"], chroma_blur=cfg["rgb_chroma_blur"],
                            saturation=cfg["rgb_saturation"], seed=int(rng.integers(1 << 30)))
               * 255 + 0.5).astype(np.uint8)
        got = make_cloud(rgb, ids, depth, gt, rng, cfg)
        if got is None:
            if not quiet:
                print(f"  [{i}] cap not visible enough, skipped")
            continue
        rec, roi = got
        name = f"synth__{seed:04d}_{i:05d}.npz"
        np.savez_compressed(out / name, **rec)
        files.append(name)
        normals.append(rec["normal"])
        anchors.append(rec["anchor"])
        meta.append({"file": name, "variant": v, "roi": list(roi),
                     "n_pts": int(len(rec["xyz"]))})
        if made < keep_png:
            _save_png(out / name.replace(".npz", ".png"), rgb)
            src_map[name] = str((out / name.replace(".npz", ".png")).resolve())
        made += 1
        if not quiet and (made % 10 == 0 or made == 1):
            print(f"  {made}/{n}  {(time.time()-t0)/made:.2f}s/frame")

    if files:
        np.savez(out / "_labels.npz", files=np.array(files),
                 normal=np.array(normals, np.float32), anchor=np.array(anchors, np.float32))
        (out / "_meta.json").write_text(json.dumps(meta, indent=1), encoding="utf-8")
        if src_map:
            (out / "_src_map.json").write_text(json.dumps(src_map), encoding="utf-8")
    shutil.rmtree(work, ignore_errors=True)
    return {"frames": len(files), "out_dir": str(out), "seconds": round(time.time() - t0, 1)}


def _save_png(path, rgb):
    try:
        from PIL import Image
        Image.fromarray(rgb).save(path)
    except Exception:
        pass


def main(argv=None):
    ap = argparse.ArgumentParser(description="synthesise labelled 内盖 frames")
    ap.add_argument("out")
    ap.add_argument("-n", type=int, default=32)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--camera", default="orbbec640", choices=sorted(S.CAMERAS))
    ap.add_argument("--domain", default="wide", choices=("wide", "prod", "orbbec"))
    ap.add_argument("--spp", type=int, default=48)
    ap.add_argument("--threads", type=int, default=0)
    ap.add_argument("--keep-png", type=int, default=8, help="also write this many RGBs")
    a = ap.parse_args(argv)
    print(json.dumps(generate(a.out, n=a.n, seed=a.seed, camera=a.camera, domain=a.domain,
                              spp=a.spp, threads=a.threads, keep_png=a.keep_png), indent=2))


if __name__ == "__main__":
    main()
