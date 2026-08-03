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


TEXTURE_EDGES = [0.0, 0.004, 0.008, 0.015, 0.028, 0.050, 1.0]


def dropout_vs_texture(cloud_dir="orbbec_clouds", raw_dir="orbbec_raw", limit=60, win=5):
    """Survival against LOCAL TEXTURE — the thing a stereo matcher actually needs.

    Brightness (below) was only ever a proxy, and a poor one: it says the cap should drop
    out more than the paint because the cap is darker, and the measurement says the
    opposite. Texture explains it. A block matcher can only find a correspondence where
    there is local structure to correspond, so a glossy body panel — a smooth brightness
    ramp with nothing in it — is the hardest surface in the frame, while matt moulded
    plastic covered in text, scuffs and shading detail is one of the easiest.

    Texture is measured as the local standard deviation of luminance in a `win`x`win`
    window, which is exactly the quantity a matcher's correlation score depends on."""
    cd = os.path.join(REF, cloud_dir)
    rd = os.path.join(REF, raw_dir)
    try:
        import cv2
    except ImportError:
        return {}
    p = os.path.join(cd, "_src_map.json")
    if not os.path.exists(p):
        return {}
    src = {k: os.path.join(rd, os.path.basename(v)) for k, v in json.load(open(p)).items()}
    nb = len(TEXTURE_EDGES) - 1
    tot, kept = np.zeros(nb), np.zeros(nb)
    n = 0
    for f in sorted(glob.glob(os.path.join(cd, "[!_]*.npz")))[:limit]:
        img_p = src.get(os.path.basename(f))
        if not img_p or not os.path.exists(img_p):
            continue
        im = cv2.imdecode(np.fromfile(img_p, np.uint8), cv2.IMREAD_COLOR)
        if im is None:
            continue
        d = np.load(f)
        xyz, K, w, h = d["xyz"], d["K_norm"], int(d["w"]), int(d["h"])
        if im.shape[1] != w:
            im = cv2.resize(im, (w, h))
        fx, fy, cx, cy = K[0, 0] * w, K[1, 1] * h, K[0, 2] * w, K[1, 2] * h
        u = np.round(xyz[:, 0] / xyz[:, 2] * fx + cx).astype(int)
        v = np.round(xyz[:, 1] / xyz[:, 2] * fy + cy).astype(int)
        ok = (u >= 0) & (u < w) & (v >= 0) & (v < h)
        u, v = u[ok], v[ok]
        if len(u) < 200:
            continue
        x0, x1, y0, y1 = u.min(), u.max() + 1, v.min(), v.max() + 1
        lum = im[y0:y1, x0:x1].astype(np.float32).mean(-1) / 255.0
        mu = cv2.blur(lum, (win, win))
        sd = np.sqrt(np.maximum(cv2.blur(lum * lum, (win, win)) - mu * mu, 0))
        idx = np.clip(np.digitize(sd, TEXTURE_EDGES) - 1, 0, nb - 1)
        tot += np.bincount(idx.ravel(), minlength=nb)
        hit = np.zeros(lum.shape, bool)
        hit[v - y0, u - x0] = True
        kept += np.bincount(idx[hit], minlength=nb)
        n += 1
    if not n:
        return {}
    rate = np.divide(kept, tot, out=np.zeros(nb), where=tot > 0)
    return {"frames": n, "edges": TEXTURE_EDGES,
            "survival": [round(float(r), 3) for r in rate],
            "share": [round(float(t / max(tot.sum(), 1)), 3) for t in tot],
            "overall": round(float(kept.sum() / max(tot.sum(), 1)), 3)}


def dropout_vs_brightness(cloud_dir="orbbec_clouds", raw_dir="orbbec_raw", bins=6, limit=60):
    """WHICH pixels a real depth camera fails on, measured against the source image.

    The cloud only contains the pixels that survived, so on its own it cannot say what was
    lost. With the source frame it can: project the cloud back to pixels, and compare the
    brightness histogram of the surviving pixels against the histogram of the whole ROI.
    The ratio is the survival rate as a function of brightness.

    This measurement overturned the model. Reasoning from "dark surfaces return no signal"
    gives a dropout that is worst in the shadows, which is what the first sensor model did
    — and it is backwards for this scene. What actually fails here is the CAR PAINT: a
    glossy, textureless, brightly-lit panel gives a stereo matcher nothing to match, while
    the matt black plastic down the pocket is full of texture and survives. Getting this
    the wrong way round produces clouds that are dense exactly where the real ones are
    empty."""
    cd = os.path.join(REF, cloud_dir)
    rd = os.path.join(REF, raw_dir)
    try:
        import cv2
    except ImportError:
        return {}
    src = {}
    p = os.path.join(cd, "_src_map.json")
    if os.path.exists(p):
        src = {k: os.path.join(rd, os.path.basename(v)) for k, v in json.load(open(p)).items()}
    tot = np.zeros(bins)
    kept = np.zeros(bins)
    n = 0
    for f in sorted(glob.glob(os.path.join(cd, "[!_]*.npz")))[:limit]:
        img_p = src.get(os.path.basename(f))
        if not img_p or not os.path.exists(img_p):
            continue
        im = cv2.imdecode(np.fromfile(img_p, np.uint8), cv2.IMREAD_COLOR)
        if im is None:
            continue
        d = np.load(f)
        xyz, K, w, h = d["xyz"], d["K_norm"], int(d["w"]), int(d["h"])
        if im.shape[1] != w:
            im = cv2.resize(im, (w, h))
        fx, fy, cx, cy = K[0, 0] * w, K[1, 1] * h, K[0, 2] * w, K[1, 2] * h
        u = np.round(xyz[:, 0] / xyz[:, 2] * fx + cx).astype(int)
        v = np.round(xyz[:, 1] / xyz[:, 2] * fy + cy).astype(int)
        ok = (u >= 0) & (u < w) & (v >= 0) & (v < h)
        u, v = u[ok], v[ok]
        if len(u) < 200:
            continue
        x0, x1, y0, y1 = u.min(), u.max() + 1, v.min(), v.max() + 1
        lum = im[y0:y1, x0:x1].astype(np.float32).mean(-1) / 255.0
        idx = np.clip((lum * bins).astype(int), 0, bins - 1)
        tot += np.bincount(idx.ravel(), minlength=bins)
        hit = np.zeros(lum.shape, bool)
        hit[v - y0, u - x0] = True
        kept += np.bincount(idx[hit], minlength=bins)
        n += 1
    if not n:
        return {}
    rate = np.divide(kept, tot, out=np.zeros(bins), where=tot > 0)
    return {"frames": n, "bin_edges": [round(i / bins, 2) for i in range(bins + 1)],
            "survival": [round(float(r), 3) for r in rate],
            "overall": round(float(kept.sum() / max(tot.sum(), 1)), 3)}


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
    # Require a cell to be seen by a decent share of the frames before trusting it — but
    # never require more frames than there are, or the single-frame case (which is how you
    # tell an averaging artefact from a real shape) comes back entirely empty.
    med[cnt < max(1, min(len(A), int(0.15 * len(A)) + 1))] = np.nan
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


def per_frame_stats(files, limit=200):
    """The shape measured on each frame ALONE, then reduced across frames.

    This is the primary comparison, and the averaged field below is the secondary one —
    which is the opposite of how it started. Averaging first looked like the cleaner
    measurement, and it is not a fair one: the real frames only stack after a plane fit
    that is itself a few degrees uncertain, so the average is smeared by its own alignment
    error, while synthetic frames stack on exact poses and are not smeared at all. The
    averaged numbers therefore compare a blurred real cap against a sharp synthetic one,
    and they said the model's rib was too WIDE at the same moment the per-frame numbers
    said it was much too NARROW. The per-frame numbers were right — both sides carry one
    frame's worth of sensor noise and nothing else."""
    S = []
    for f in files[:limit]:
        m = height_field([f])
        if m is None:
            continue
        s = field_stats(m)
        if "rib_len_mm" in s and s["disc_u_mm"] > 40:
            S.append(s)
    if not S:
        return {}
    return {k: float(np.median([s[k] for s in S if k in s]))
            for k in ("disc_u_mm", "disc_v_mm", "rib_h_mm", "rib_len_mm", "rib_wid_mm")}


def compare(real_dir, synth_dir, png=None):
    """Real vs synthetic, through the identical measurement, both ways."""
    rf = sorted(glob.glob(os.path.join(REF, real_dir, "[!_]*.npz")))
    sf = sorted(glob.glob(os.path.join(synth_dir, "[!_]*.npz")))
    if not rf or not sf:
        print(f"need both: {len(rf)} real, {len(sf)} synthetic")
        return
    pa, pb = per_frame_stats(rf), per_frame_stats(sf)
    print(f"\nPER-FRAME (primary)   {len(rf)} real / {len(sf)} synth")
    print(f"{'':16s} {'real':>10s} {'synth':>10s} {'delta':>10s}")
    for k in sorted(set(pa) | set(pb)):
        va, vb = pa.get(k), pb.get(k)
        d = f"{vb-va:+10.1f}" if (va is not None and vb is not None) else " " * 10
        print(f"{k:16s} {'':>10s}" if va is None else f"{k:16s} {va:10.1f}", end="")
        print(f" {vb:10.1f} {d}" if vb is not None else "")

    a, b = height_field(rf), height_field(sf)
    sa, sb = field_stats(a), field_stats(b)
    print("\nAVERAGED FIELD (secondary — real side is blurred by its own pose error)")
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


def audit(synth_dir, real_dir="orbbec_clouds"):
    """Every measured statistic, real against synthetic, in one table.

    This is the scorecard the case is steered by, and it is deliberately the whole list
    rather than one summary number: a synthetic set can match the cap's shape perfectly and
    still be trivially separable from real data on point count, or noise, or how full its
    ROI is. Each row is a way the two could differ that a network could learn instead of
    the pose."""
    rf = sorted(glob.glob(os.path.join(REF, real_dir, "[!_]*.npz")))
    sf = sorted(glob.glob(os.path.join(synth_dir, "[!_]*.npz")))
    if not rf or not sf:
        print(f"need both: {len(rf)} real, {len(sf)} synthetic")
        return

    def block(files):
        vg = view_geometry(files[:200])
        out = per_frame_stats(files)
        if vg:
            out["dist_m"] = float(np.median([v["dist"] for v in vg]))
            out["obliquity_deg"] = float(np.median([v["obliquity"] for v in vg]))
            out["cap_px"] = float(np.median([v["px"] for v in vg]))
        n = depth_noise(files[:80])
        if n:
            out["noise_mm"] = n["sigma_mm_med"]
        out["pts_per_frame"] = float(np.median([len(np.load(f)["xyz"]) for f in files[:80]]))
        out["cap_frac"] = float(np.median([
            float((np.load(f)["label"] == 1).mean()) for f in files[:80]]))
        return out

    A, B = block(rf), block(sf)
    print(f"\n{'':18s} {'real':>10s} {'synth':>10s} {'ratio':>8s}")
    for k in ("disc_u_mm", "disc_v_mm", "rib_h_mm", "rib_len_mm", "rib_wid_mm",
              "dist_m", "obliquity_deg", "cap_px", "noise_mm", "pts_per_frame", "cap_frac"):
        va, vb = A.get(k), B.get(k)
        if va is None or vb is None:
            continue
        print(f"{k:18s} {va:10.2f} {vb:10.2f} {vb/max(va,1e-9):8.2f}")


def check_labels(synth_dir, limit=200):
    """Does the stored pose label actually describe the points it is stored with?

    The one test that has to pass before any of the rest matters. A synthetic set's whole
    claim is that its labels are exact, and that claim is easy to break in a way no render
    will ever show: an axis convention flipped, a rotation composed in the wrong order, a
    normal left in world coordinates. The check is to ignore the label, fit a plane to the
    cap's own flat annulus, and see whether the two agree.

    It has already earned its place. The first version of this case composed the cap's
    screw-stop angle OUTSIDE its tilt, so spinning the cap swung its axis: every frame was
    mislabelled by a median of 8 degrees, every render looked perfect, and every downstream
    number in the audit table was unaffected."""
    files = sorted(glob.glob(os.path.join(synth_dir, "[!_]*.npz")))[:limit]
    ang, dcm = [], []
    for f in files:
        d = np.load(f)
        if "normal" not in d.files:
            continue
        P = d["xyz"][d["label"] == 1]
        if len(P) < 300:
            continue
        n_gt = np.asarray(d["normal"], float)
        h = (P - P.mean(0)) @ n_gt
        # the FLAT annulus only — fitting a plane through the raised grip rib as well
        # measures the rib, not the face the label is about
        A = P[h < np.percentile(h, 55)]
        if len(A) < 150:
            continue
        c = A.mean(0)
        _, _, V = np.linalg.svd(A - c, full_matrices=False)
        n = V[2]
        if n @ n_gt < 0:
            n = -n
        ang.append(np.degrees(np.arccos(np.clip(float(n @ n_gt), -1, 1))))
        dcm.append(1000 * float(np.linalg.norm(np.asarray(d["anchor"], float) - c)))
    if not ang:
        print("no labelled synthetic frames found")
        return
    ang, dcm = np.array(ang), np.array(dcm)
    print(f"\nLABEL SELF-CONSISTENCY   ({len(ang)} frames)")
    print(f"  normal vs plane fit of its own cap   med={np.median(ang):5.2f} deg  "
          f"p90={np.percentile(ang, 90):5.2f}  max={ang.max():5.2f}")
    print(f"  anchor vs centroid of cap points     med={np.median(dcm):5.1f} mm   "
          f"p90={np.percentile(dcm, 90):5.1f}")
    print("  (a few degrees is the sensor model and the partly-occluded annulus; "
          "8+ is a bug)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", default=None)
    ap.add_argument("--check-labels", default=None, metavar="SYNTH_DIR")
    ap.add_argument("--compare", nargs=2, metavar=("REAL_DIR", "SYNTH_DIR"),
                    help="e.g. --compare orbbec_clouds out/synth")
    ap.add_argument("--audit", default=None, metavar="SYNTH_DIR",
                    help="the full real-vs-synthetic scorecard")
    ap.add_argument("--png", default=None)
    a = ap.parse_args()
    if a.compare:
        compare(a.compare[0], a.compare[1], a.png)
        return
    if a.audit:
        audit(a.audit)
        check_labels(a.audit)
        return
    if a.check_labels:
        check_labels(a.check_labels)
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
    do = dropout_vs_brightness()
    if do:
        print(f"\n=== depth survival vs brightness  ({do['frames']} frames, overall "
              f"{do['overall']}) ===")
        print("  " + "  ".join(f"{a:.2f}-{b:.2f}:{s:.2f}" for a, b, s in
                               zip(do["bin_edges"], do["bin_edges"][1:], do["survival"])))
    if a.json:
        with open(a.json, "w", encoding="utf-8") as f:
            json.dump(res, f, indent=2)
        print("\nwrote", a.json)


if __name__ == "__main__":
    main()
