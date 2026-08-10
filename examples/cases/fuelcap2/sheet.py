"""Render and compare the photograph-matched Polo fuel-filler scene."""
from __future__ import annotations

import argparse
import os
import subprocess
from pathlib import Path

import numpy as np

from fuelcap.dataset import _read_ppm, _render_bin

from . import polo

HERE = Path(__file__).resolve().parent
OUT = HERE / "_out"
REF = (HERE.parent / "fuelcap" / "_ref" / "bycar" / "Polo" /
       "粗筛done2_Polo_2007款劲情14自动风尚版_38.png")


def _render_view(oplog, out_ppm, pose, size, spp, extra=()):
    """One frame at a given pose, with this case's lighting held fixed.

    Held fixed on purpose: a validation view that also changes the light is not a
    validation of anything.
    """
    width, height = size, round(size * 0.75)
    args = [str(_render_bin()), "--oplog", str(oplog), "--out", str(out_ppm),
            "--w", str(width), "--h", str(height), "--spp", str(spp), "--denoise", "4",
            "--threads", os.environ.get("MIRAGE_THREADS", "0"),
            "--cam-eye", *[f"{v:.7f}" for v in pose["eye"]],
            "--cam-target", *[f"{v:.7f}" for v in pose["target"]],
            "--cam-up", *[f"{v:.7f}" for v in pose["up"]],
            "--cam-fov", f"{pose.get('fov', polo.pose()['fov']):.7f}",
            # MEASURED, not chosen. `tools/look.py` said the render was 20-62 luma below
            # the photograph in every region -- the largest fault in the whole
            # reproduction and invisible on a 900 px compare sheet. Sweeping env/sun/
            # exposure over a --crop of the pocket (a twelfth of a frame, so five
            # renders cost less than one) lands the pocket mean on the photograph's 83.
            "--env", "1.40", "--sun", "0.58",
            "--sun-dir", "0.18", "-0.12", "0.976",
            "--sky-tint", "1.02", "1.08", "1.12", "--sky-flat", "0.72",
            "--exposure", "1.13", "--smooth-angle", "34", "--bounce", "7", "--no-ground",
            *extra]
    subprocess.run(args, check=True)
    return _read_ppm(out_ppm)


def verify(size=420, spp=64, degrees=(-5.0, 5.0)):
    """Render the matched view and a few degrees either side, and measure the parallax.

    This is the check that would have caught this case's projected albedo maps in one
    render: artwork projected from the reference camera is perfect from that camera and
    smears from every other, and flat geometry painted to look modelled does not move at
    all. A LOW parallax after a real camera move is the warning, not a pass.

    Small and cheap on purpose — 420 px at 64 spp — because its job is to be run often.
    """
    from mirage import validate as V
    OUT.mkdir(parents=True, exist_ok=True)
    oplog = OUT / "polo.json"
    oplog.write_text(polo.build().to_json(), encoding="utf-8")
    base = polo.pose()
    ref = _render_view(oplog, OUT / "verify_base.ppm", base, size, spp)
    rows = []
    for v in V.orbit(base["eye"], base["target"], base["up"], degrees=degrees):
        pose = dict(v)
        pose["fov"] = base["fov"]
        img = _render_view(oplog, OUT / f"verify_{v['label']}.ppm", pose, size, spp)
        rows.append((v["label"], V.parallax(ref, img)))
    print(f"{'view':12s} {'mean|d|':>8s} {'p95':>7s} {'moved':>7s}")
    for lab, p in rows:
        print(f"{lab:12s} {p['mean_abs']:8.2f} {p['p95']:7.1f} {p['moved_frac']:7.2%}")
    w = V.flatness_warning(rows, threshold=2.0)
    print("suspect views:", w["suspect"] or "none", "|", w["note"])
    return rows


def render(size=1000, spp=160):
    OUT.mkdir(parents=True, exist_ok=True)
    oplog = OUT / "polo.json"
    ppm = OUT / "polo_synth.ppm"
    png = OUT / "polo_synth.png"
    oplog.write_text(polo.build().to_json(), encoding="utf-8")
    p = polo.pose()
    width, height = size, round(size * 0.75)
    args = [str(_render_bin()), "--oplog", str(oplog), "--out", str(ppm),
            "--w", str(width), "--h", str(height), "--spp", str(spp), "--denoise", "4",
            "--threads", os.environ.get("MIRAGE_THREADS", "0"),
            "--cam-eye", *[f"{v:.7f}" for v in p["eye"]],
            "--cam-target", *[f"{v:.7f}" for v in p["target"]],
            "--cam-up", *[f"{v:.7f}" for v in p["up"]],
            "--cam-fov", f"{p['fov']:.7f}", "--env", "1.40", "--sun", "0.58",
            "--sun-dir", "0.18", "-0.12", "0.976",
            "--sky-tint", "1.02", "1.08", "1.12", "--sky-flat", "0.72",
            "--exposure", "1.13", "--smooth-angle", "34", "--bounce", "7", "--no-ground"]
    subprocess.run(args, check=True)
    import cv2
    rgb = _read_ppm(ppm).astype(np.float32)
    # The source is a compact-camera frame, not a noiseless float buffer. Preserve the
    # path-traced edges while adding the low-amplitude luma/chroma grain visible across its
    # blue paint. This is deterministic so a visual diff still means something.
    rng = np.random.default_rng(2007)
    luma = rng.normal(0.0, 1.25, rgb.shape[:2])[..., None]
    chroma = rng.normal(0.0, 0.55, rgb.shape)
    rgb = np.clip(rgb + luma + chroma, 0, 255).astype(np.uint8)
    cv2.imwrite(str(png), rgb[:, :, ::-1])
    print(png)
    return png


def compose(size=1000):
    import cv2
    if not REF.exists():
        raise SystemExit(f"missing local reference: {REF}")
    raw = cv2.imdecode(np.fromfile(REF, np.uint8), cv2.IMREAD_COLOR)
    synth = cv2.imread(str(OUT / "polo_synth.png"))
    if synth is None:
        raise SystemExit("render polo_synth.png first")

    # The reconstruction covers the same 4:3 view as the source. Keep the full left lamp,
    # open door and surrounding body instead of flattering the model with a tight cap crop.
    real = cv2.resize(raw, (size, round(size * 0.75)), interpolation=cv2.INTER_AREA)
    synth = cv2.resize(synth, real.shape[1::-1], interpolation=cv2.INTER_AREA)
    cv2.putText(real, "PHOTOGRAPH", (18, 38), cv2.FONT_HERSHEY_SIMPLEX,
                0.85, (35, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(synth, "MIRAGE", (18, 38), cv2.FONT_HERSHEY_SIMPLEX,
                0.85, (35, 255, 255), 2, cv2.LINE_AA)
    out = OUT / "polo_compare.png"
    cv2.imwrite(str(out), np.hstack([real, synth]))
    print(out)
    return out


def compose_closeup(size=900):
    """Square inspection crop: opening, cap, latch, hinge and door, without the tail lamp."""
    import cv2
    raw = cv2.imdecode(np.fromfile(REF, np.uint8), cv2.IMREAD_COLOR)
    synth = cv2.imread(str(OUT / "polo_synth.png"))
    if raw is None or synth is None:
        raise SystemExit("reference and polo_synth.png are required")

    # Measured once on the fixed source framing. The synthetic crop uses the same subject
    # bounds, not the same raw pixel indices, because its renderer output may be any size.
    real = raw[35:1035, 315:1315]
    h, w = synth.shape[:2]
    side = min(h, int(w * 0.76))
    cx, cy = int(w * 0.57), int(h * 0.47)
    x0 = max(0, min(w - side, cx - side // 2))
    y0 = max(0, min(h - side, cy - side // 2))
    syn = synth[y0:y0 + side, x0:x0 + side]
    real = cv2.resize(real, (size, size), interpolation=cv2.INTER_AREA)
    syn = cv2.resize(syn, (size, size), interpolation=cv2.INTER_AREA)
    cv2.putText(real, "PHOTOGRAPH", (18, 38), cv2.FONT_HERSHEY_SIMPLEX,
                0.85, (35, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(syn, "MIRAGE", (18, 38), cv2.FONT_HERSHEY_SIMPLEX,
                0.85, (35, 255, 255), 2, cv2.LINE_AA)
    out = OUT / "polo_closeup.png"
    cv2.imwrite(str(out), np.hstack([real, syn]))
    print(out)
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("what", choices=("render", "compose", "closeup", "all", "verify"),
                    default="all", nargs="?")
    ap.add_argument("--size", type=int, default=1000)
    ap.add_argument("--spp", type=int, default=160)
    a = ap.parse_args(argv)
    if a.what == "verify":
        verify(a.size if a.size < 700 else 420, min(a.spp, 80))
        return
    if a.what in ("render", "all"):
        render(a.size, a.spp)
    if a.what in ("compose", "all"):
        compose(a.size)
        compose_closeup(a.size)
    elif a.what == "closeup":
        compose_closeup(a.size)


if __name__ == "__main__":
    main()
