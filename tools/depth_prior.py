"""A monocular depth prior for reverse modelling, and the comparison that makes it useful.

Reverse-modelling a part from one photograph has been done here by probing: walk a ray
across the picture, read the tone steps, convert a pixel run into millimetres. That works
and it is slow, and it only ever measures the handful of places somebody thought to probe.
A monocular depth network gives a DENSE estimate of the same thing in one pass.

The catch, and the reason this is a `prior` and not a measurement: monocular depth is
**scale and shift ambiguous**. The network cannot know how big the car is. So the raw map
is worth very little on its own, and worth a lot the moment there is a render to compare
it against: fit one global scale and shift that best maps the predicted inverse depth onto
the render's own depth AOV, then look at what is LEFT. The residual is where the model's
shape disagrees with the photograph's, densely, in millimetres, without ever needing the
network to be metrically right.

    # on the box, where the GPU and the model live
    ~/depth_env/bin/python tools/depth_prior.py predict photo.png --out photo_depth.npy
    # anywhere
    python tools/depth_prior.py residual photo_depth.npy render_depth.pfm --mask ids.pgm

`predict` needs torch + transformers; `residual` needs numpy alone, so the analysis runs
on the laptop against a map computed once on the box.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

# Tried in order. V3 if it is published under one of these names, else V2 Large, which is
# the strongest thing that is definitely on the hub today. The choice is printed, because
# "which model produced this prior" is part of the measurement.
CANDIDATES = [
    "depth-anything/Depth-Anything-V3-Large-hf",
    "depth-anything/Depth-Anything-V3-Base-hf",
    "depth-anything/Depth-Anything-V2-Large-hf",
    "depth-anything/Depth-Anything-V2-Base-hf",
]


def predict(image_path: Path, out: Path, model: str | None = None):
    import torch
    from PIL import Image
    from transformers import pipeline

    names = [model] if model else CANDIDATES
    pipe, used = None, None
    for name in names:
        try:
            pipe = pipeline("depth-estimation", model=name,
                            device=0 if torch.cuda.is_available() else -1)
            used = name
            break
        except Exception as exc:                      # not published / not cached / no net
            print(f"  {name}: {type(exc).__name__}: {str(exc)[:90]}")
    if pipe is None:
        raise SystemExit("no depth model could be loaded; see the attempts above")
    print(f"using {used} on {'cuda' if torch.cuda.is_available() else 'cpu'}")

    img = Image.open(image_path).convert("RGB")
    res = pipe(img)
    d = np.asarray(res["predicted_depth"] if "predicted_depth" in res else res["depth"],
                   dtype=np.float32)
    if d.ndim == 3:
        d = d[0]
    out.parent.mkdir(parents=True, exist_ok=True)
    np.save(out, d)
    print(f"wrote {out}  {d.shape}  range {d.min():.3f}..{d.max():.3f}  model={used}")
    return d


def _read_pfm(path: Path):
    with open(path, "rb") as f:
        hdr = f.readline().strip()
        if hdr not in (b"Pf", b"PF"):
            raise ValueError(f"{path}: not a PFM ({hdr!r})")
        ch = 1 if hdr == b"Pf" else 3
        w, h = (int(x) for x in f.readline().split())
        scale = float(f.readline())
        data = np.frombuffer(f.read(w * h * ch * 4), dtype="<f4" if scale < 0 else ">f4")
    a = data.reshape(h, w, ch) if ch == 3 else data.reshape(h, w)
    return a[::-1].copy()                              # PFM rows are bottom-up


def residual(prior_path: Path, render_depth_path: Path, mask_path: Path | None = None,
             out: Path | None = None):
    """Fit scale+shift from the prior onto the render's depth, then report what is left.

    The fit is the whole point. A monocular prior is defined up to an affine transform of
    INVERSE depth, so the only honest comparison is: take the render's depth as truth,
    solve `a * prior + b` in inverse-depth space by least squares over the valid pixels,
    and read the residual. A model that is the right shape but the wrong size gives a large
    scale and a small residual; a model that is the wrong shape gives a large residual
    wherever it is wrong, and that map is the useful output.
    """
    prior = np.load(prior_path).astype(np.float64)
    dep = _read_pfm(render_depth_path).astype(np.float64)
    if dep.ndim == 3:
        dep = dep[..., 0]
    if prior.shape != dep.shape:
        # nearest resample; the prior is smooth so this loses nothing that matters
        yi = (np.linspace(0, prior.shape[0] - 1, dep.shape[0])).astype(int)
        xi = (np.linspace(0, prior.shape[1] - 1, dep.shape[1])).astype(int)
        prior = prior[yi][:, xi]

    valid = dep > 1e-6
    if mask_path is not None:
        m = _read_pgm(Path(mask_path))
        if m.shape == dep.shape:
            valid &= m > 0
    if valid.sum() < 50:
        raise SystemExit("fewer than 50 valid pixels to fit against")

    inv_render = 1.0 / dep[valid]
    p = prior[valid]
    A = np.stack([p, np.ones_like(p)], 1)
    (a, b), *_ = np.linalg.lstsq(A, inv_render, rcond=None)
    fitted = np.full(dep.shape, np.nan)
    fitted[valid] = 1.0 / np.maximum(a * prior[valid] + b, 1e-9)
    res = np.full(dep.shape, np.nan)
    res[valid] = fitted[valid] - dep[valid]
    r = res[valid]
    print(f"scale {a:.6g}  shift {b:.6g}   valid {int(valid.sum())} px")
    print(f"residual mm:  mean {1e3*np.mean(np.abs(r)):.2f}   p50 {1e3*np.median(np.abs(r)):.2f}"
          f"   p95 {1e3*np.percentile(np.abs(r), 95):.2f}   max {1e3*np.max(np.abs(r)):.2f}")
    # where, not just how much
    h, w = res.shape
    ny, nx = 6, 6
    print("worst cells (row, col, mm):")
    cells = []
    for j in range(ny):
        for i in range(nx):
            s = res[j*h//ny:(j+1)*h//ny, i*w//nx:(i+1)*w//nx]
            s = s[np.isfinite(s)]
            if s.size:
                cells.append((float(np.mean(np.abs(s))), j, i))
    for v, j, i in sorted(cells, reverse=True)[:4]:
        print(f"   ({j},{i})  {1e3*v:6.2f} mm")
    if out:
        np.save(out, res)
        print(f"wrote {out}")
    return res


def _read_pgm(path: Path):
    with open(path, "rb") as f:
        magic = f.readline().strip()
        line = f.readline()
        while line.startswith(b"#"):
            line = f.readline()
        w, h = (int(x) for x in line.split())
        maxv = int(f.readline())
        dt = ">u2" if maxv > 255 else "u1"
        return np.frombuffer(f.read(), dtype=dt).reshape(h, w)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("predict")
    p.add_argument("image")
    p.add_argument("--out", default="depth_prior.npy")
    p.add_argument("--model", default=None)
    r = sub.add_parser("residual")
    r.add_argument("prior")
    r.add_argument("render_depth")
    r.add_argument("--mask", default=None)
    r.add_argument("--out", default=None)
    a = ap.parse_args(argv)
    if a.cmd == "predict":
        predict(Path(a.image), Path(a.out), a.model)
    else:
        residual(Path(a.prior), Path(a.render_depth), a.mask, Path(a.out) if a.out else None)


if __name__ == "__main__":
    sys.exit(main())
