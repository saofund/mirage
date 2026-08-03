"""What the real capture actually looks like — the numbers the synthetic set is aimed at.

Three distributions decide whether a synthetic 6D-pose set transfers, and none of them is
about how pretty the render is:

1. **where the camera is** — distance to the cap, and the angle between the cap's normal
   and the view axis. Sample the wrong cone and the network never sees the poses it will
   be asked about.
2. **how big the cap is on the sensor** — set by (1) and the intrinsics, but worth checking
   directly, because it is the one number a downstream crop depends on.
3. **how noisy the depth is** — a real stereo/ToF cloud is not a surface, it is a surface
   plus a milimetre of axial scatter that grows with range, plus dropouts, plus fattened
   edges. A synthetic cloud without that is trivially separable from a real one, and a
   network trained on it learns the difference instead of the shape.

Run it against the pulled reference set:

    python -m fuelcap.fit                    # all three, printed
    python -m fuelcap.fit --json out.json    # the same, for scene.py to read
"""
from __future__ import annotations

import argparse, glob, json, os
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REF = os.path.join(HERE, "_ref")


def _plane(P):
    c = P.mean(0)
    _, _, V = np.linalg.svd(P - c, full_matrices=False)
    n = V[2]
    return c, (-n if n[2] > 0 else n)          # toward the camera, which looks down +z


def view_geometry(files):
    """Distance, obliquity and apparent size, per frame."""
    out = []
    for f in files:
        d = np.load(f)
        P = d["xyz"][d["label"] == 1]
        if len(P) < 300:
            continue
        z = P[:, 2]
        hist, edges = np.histogram(z, 60)
        P = P[np.abs(z - 0.5 * (edges[np.argmax(hist)] + edges[np.argmax(hist) + 1])) < 0.030]
        if len(P) < 300:
            continue
        c, n = _plane(P)
        # obliquity: angle between the cap normal and the ray to the cap centre. 0 = the
        # camera is square-on to the cap. Measured against the RAY, not against the optical
        # axis: a cap 40 degrees off centre in the frame is seen obliquely even if the
        # sensor is pointing straight at the panel.
        ray = c / max(np.linalg.norm(c), 1e-9)
        obl = math_deg(np.arccos(np.clip(abs(float(n @ ray)), 0, 1)))
        K = d["K_norm"].copy()
        fx = float(K[0, 0]) * int(d["w"])
        r = np.percentile(np.linalg.norm(P - c, axis=1), 95)
        out.append(dict(dist=float(np.linalg.norm(c)), obliquity=obl,
                        px=float(2 * r * fx / max(float(c[2]), 1e-6)),
                        normal=[float(v) for v in n], centre=[float(v) for v in c]))
    return out


def math_deg(x):
    return float(np.degrees(x))


def depth_noise(files, patch=0.008):
    """Axial scatter of a real cloud about its own local plane, versus range.

    Fitted on small patches of the cap face, which is genuinely flat, so whatever is left
    over is the sensor. Reported as sigma in millimetres and as the coefficient k in
    sigma = k * z^2, which is the standard form for a stereo rig (disparity quantisation
    turns into range error quadratically)."""
    sig, rng = [], []
    for f in files:
        d = np.load(f)
        P = d["xyz"][d["label"] == 1]
        if len(P) < 400:
            continue
        c, n = _plane(P)
        keep = np.abs((P - c) @ n) < 0.020
        P = P[keep]
        if len(P) < 400:
            continue
        # local planes: bin the cap face into patches and take the residual within each
        u = np.eye(3)[np.argmin(np.abs(n))]
        e1 = np.cross(n, u); e1 /= np.linalg.norm(e1)
        e2 = np.cross(n, e1)
        a, b = (P - c) @ e1, (P - c) @ e2
        gi = np.floor(a / patch).astype(int) * 1000 + np.floor(b / patch).astype(int)
        res = []
        for g in np.unique(gi):
            Q = P[gi == g]
            if len(Q) < 12:
                continue
            qc, qn = _plane(Q)
            res.append((Q - qc) @ qn)
        if res:
            r = np.concatenate(res)
            sig.append(1.4826 * np.median(np.abs(r - np.median(r))))
            rng.append(float(np.median(P[:, 2])))
    if not sig:
        return {}
    sig, rng = np.array(sig), np.array(rng)
    k = float(np.median(sig / rng ** 2))
    return {"sigma_mm_med": float(np.median(sig) * 1000),
            "sigma_mm_p10": float(np.percentile(sig, 10) * 1000),
            "sigma_mm_p90": float(np.percentile(sig, 90) * 1000),
            "range_m_med": float(np.median(rng)), "k_quadratic": k,
            "n_frames": int(len(sig))}


def dropouts(files):
    """What fraction of the ROI a real depth map simply has no answer for."""
    fr = []
    for f in files:
        d = np.load(f)
        xyz = d["xyz"]
        w, h = int(d["w"]), int(d["h"])
        if len(xyz) == 0:
            continue
        # the cloud is already the kept pixels; against a full ROI it is the survival rate
        fr.append(float(len(xyz)))
    return {"pts_med": float(np.median(fr)) if fr else 0.0}


def summarise(tag, dirn):
    files = sorted(glob.glob(os.path.join(REF, dirn, "[!_]*.npz")))
    if not files:
        return None
    vg = view_geometry(files)
    if not vg:
        return None
    A = lambda k: np.array([v[k] for v in vg])
    out = {"tag": tag, "frames": len(vg)}
    for k in ("dist", "obliquity", "px"):
        a = A(k)
        out[k] = {"p05": float(np.percentile(a, 5)), "med": float(np.median(a)),
                  "p95": float(np.percentile(a, 95))}
    out["noise"] = depth_noise(files)
    return out


# --------------------------------------------------------------------------- #
# the shape itself
# --------------------------------------------------------------------------- #
GRID, HALF = 121, 0.050          # a 100 mm window across the cap, 0.83 mm cells


def height_field(files, limit=None):
    """Average many frames into one canonical height map of the cap face.

    Each frame is rotated into the cap's own frame — normal up, grip rib along +u — and
    accumulated into a shared grid; the median over frames is the shape. This is the metric
    the whole kit is aimed at, and it is deliberately the SAME function for real clouds and
    for synthetic ones. A metric that measures the two differently cannot tell you they
    differ."""
    acc = []
    for f in (files[:limit] if limit else files):
        d = np.load(f)
        P = d["xyz"][d["label"] == 1]
        if len(P) < 300:
            continue
        z = P[:, 2]
        hist, edges = np.histogram(z, 60)
        P = P[np.abs(z - 0.5 * (edges[np.argmax(hist)] + edges[np.argmax(hist) + 1])) < 0.030]
        if len(P) < 300:
            continue
        c, n = _plane(P)
        t = np.array([0.0, 0.0, 1.0]) if abs(n[2]) < 0.9 else np.array([1.0, 0.0, 0.0])
        u = np.cross(t, n); u /= np.linalg.norm(u)
        v = np.cross(n, u)
        L = np.stack([(P - c) @ u, (P - c) @ v, (P - c) @ n], 1)
        L[:, :2] -= np.median(L[:, :2], 0)
        top = L[L[:, 2] > np.percentile(L[:, 2], 75)]
        if len(top) < 50:
            continue
        _, _, W = np.linalg.svd(top[:, :2] - top[:, :2].mean(0), full_matrices=False)
        a = W[0]
        L[:, :2] = L[:, :2] @ np.array([[a[0], -a[1]], [a[1], a[0]]])
        L[:, 2] -= np.percentile(L[:, 2], 5)
        gx = ((L[:, 0] + HALF) / (2 * HALF) * GRID).astype(np.int32)
        gy = ((L[:, 1] + HALF) / (2 * HALF) * GRID).astype(np.int32)
        m = (gx >= 0) & (gx < GRID) & (gy >= 0) & (gy < GRID)
        H = np.full((GRID, GRID), -np.inf, np.float32)
        np.maximum.at(H, (gy[m], gx[m]), L[m, 2])       # NaN would poison np.maximum
        H[np.isinf(H)] = np.nan
        acc.append(H)
    if not acc:
        return None
    A = np.array(acc)
    with np.errstate(all="ignore"):
        med = np.nanmedian(A, 0)
        cnt = np.sum(~np.isnan(A), 0)
    med[cnt < max(3, 0.15 * len(A))] = np.nan
    return med


def field_stats(med):
    px = 2 * HALF / GRID
    filled = ~np.isnan(med)
    if filled.sum() < 20:
        return {}
    ys, xs = np.nonzero(filled)
    hmax = float(np.nanpercentile(med, 99))
    rib = filled & (med > 0.55 * hmax)
    out = {"disc_u_mm": (xs.max() - xs.min() + 1) * px * 1000,
           "disc_v_mm": (ys.max() - ys.min() + 1) * px * 1000,
           "rib_h_mm": hmax * 1000}
    if rib.sum() > 4:
        ys, xs = np.nonzero(rib)
        out["rib_len_mm"] = (xs.max() - xs.min() + 1) * px * 1000
        out["rib_wid_mm"] = (ys.max() - ys.min() + 1) * px * 1000
    return out


def compare(real_dir, synth_dir, png=None):
    """Real vs synthetic, through the identical measurement."""
    rf = sorted(glob.glob(os.path.join(REF, real_dir, "[!_]*.npz")))
    sf = sorted(glob.glob(os.path.join(synth_dir, "[!_]*.npz")))
    if not rf or not sf:
        print(f"need both: {len(rf)} real, {len(sf)} synthetic")
        return
    a, b = height_field(rf), height_field(sf)
    sa, sb = field_stats(a), field_stats(b)
    print(f"\n{'':16s} {'real':>10s} {'synth':>10s} {'delta':>10s}")
    for k in sorted(set(sa) | set(sb)):
        va, vb = sa.get(k), sb.get(k)
        d = f"{vb-va:+10.1f}" if (va is not None and vb is not None) else " " * 10
        print(f"{k:16s} {va if va is None else f'{va:10.1f}'} "
              f"{vb if vb is None else f'{vb:10.1f}'} {d}")
    if png:
        import cv2
        tiles = []
        hm = max(float(np.nanpercentile(a, 99)), float(np.nanpercentile(b, 99)), 1e-6)
        for f, lab in ((a, "real"), (b, "synth")):
            vis = np.nan_to_num(f / hm, nan=-0.2)
            im = cv2.applyColorMap(np.clip((vis + 0.2) / 1.2 * 255, 0, 255).astype(np.uint8),
                                   cv2.COLORMAP_TURBO)
            im[np.isnan(f)] = 0
            im = cv2.resize(im, (420, 420), interpolation=cv2.INTER_NEAREST)
            cv2.putText(im, lab, (10, 26), 0, 0.7, (255, 255, 255), 2)
            bl = int(0.020 / (2 * HALF) * 420)
            cv2.line(im, (12, 408), (12 + bl, 408), (255, 255, 255), 2)
            cv2.putText(im, "20mm", (12, 400), 0, 0.45, (255, 255, 255), 1)
            tiles.append(im)
        cv2.imwrite(png, np.hstack(tiles))
        print("wrote", png)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", default=None)
    ap.add_argument("--compare", nargs=2, metavar=("REAL_DIR", "SYNTH_DIR"),
                    help="e.g. --compare orbbec_clouds out/synth")
    ap.add_argument("--png", default=None)
    a = ap.parse_args()
    if a.compare:
        compare(a.compare[0], a.compare[1], a.png)
        return
    res = [r for r in (summarise("prod", "prod_open_inside_clouds"),
                       summarise("orbbec", "orbbec_clouds"),
                       summarise("depthcap", "prod_depthcap_clouds")) if r]
    for r in res:
        print(f"\n=== {r['tag']}  ({r['frames']} frames) ===")
        for k, lbl, sc in (("dist", "camera distance   m", 1), ("obliquity", "obliquity      deg", 1),
                           ("px", "cap width      px", 1)):
            v = r[k]
            print(f"  {lbl:22s} p05={v['p05']*sc:7.3f}  med={v['med']*sc:7.3f}  p95={v['p95']*sc:7.3f}")
        n = r["noise"]
        if n:
            print(f"  depth noise      mm   p10={n['sigma_mm_p10']:5.2f}  med={n['sigma_mm_med']:5.2f}"
                  f"  p90={n['sigma_mm_p90']:5.2f}   (k in s=k*z^2: {n['k_quadratic']:.5f}, "
                  f"at z={n['range_m_med']:.2f} m)")
    if a.json:
        with open(a.json, "w", encoding="utf-8") as f:
            json.dump(res, f, indent=2)
        print("\nwrote", a.json)


if __name__ == "__main__":
    main()
