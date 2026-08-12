"""What the camera can actually SEE, and what is in the way.

Why this module exists
----------------------
This repo's feedback loop had a hole in exactly the place where it was most confident.
Every score it computed -- frame difference, region difference, tonal spread -- reads the
picture. A part that renders zero pixels contributes nothing to any of them, so a part
that has been built wrong and a part that has been hidden are indistinguishable to every
metric, and the hidden one looks *slightly better* because whatever is in front of it is
usually closer to the reference than a mistake would be.

That is not hypothetical. A pocket in one of the cases was rebuilt from scratch onto a
new construction -- a different generator, a canopy where there had been a bowl, a
contour that reversed where none had before -- and the region's mean absolute difference
moved from 46.7 to 46.3. The rebuild was invisible: a backing disc 223 mm across sat at
z = -55 while the new canopy ran down to z = -76 and was 194 mm across, so the disc stood
in front of all of it. The signal was right there in the numbers. Nobody read it, because
"a big change that barely moved the picture" is not something the tools said out loud.

So they say it now:

    occlusion_report(before_fids, after_fids, changed_faces=..., pixels_changed=...)

`face_visibility` turns a face-id AOV into pixel counts per face -- which is also what
answers "how big does this part appear", "what fraction of it is covered", and "what is
in front of it", none of which the object-id AOV can do because one tag covers thousands
of faces.

The renderer writes the AOV with `--face-ids out.pfm` (one-channel float PFM, -1 where a
ray hit nothing, ids being the mesh's own -- the same ones selectors and the marked
render use, so a pixel maps back to an op-log face).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


def read_pfm(path):
    """One- or three-channel PFM as a float array, rows top-down."""
    with open(path, "rb") as fh:
        kind = fh.readline().strip()
        if kind not in (b"Pf", b"PF"):
            raise ValueError(f"{path}: not a PFM (header {kind!r})")
        w, h = (int(v) for v in fh.readline().split())
        scale = float(fh.readline())
        n = 1 if kind == b"Pf" else 3
        data = np.frombuffer(fh.read(w * h * n * 4), dtype="<f4" if scale < 0 else ">f4")
    img = data.reshape(h, w) if n == 1 else data.reshape(h, w, 3)
    return img[::-1].copy()          # PFM stores rows bottom-up


def face_ids(path):
    """The face-id AOV as an int array; -1 where nothing was hit."""
    return np.rint(read_pfm(path)).astype(np.int64)


def face_visibility(fids):
    """{face id: pixels} for every face the centre ray reached. Excludes the background."""
    flat = np.asarray(fids).ravel()
    flat = flat[flat >= 0]
    if flat.size == 0:
        return {}
    ids, counts = np.unique(flat, return_counts=True)
    return {int(i): int(c) for i, c in zip(ids, counts)}


def hidden_faces(fids, expected):
    """Of `expected`, which render not one pixel.

    The question to ask after any edit that creates geometry. `expected` is whatever the
    op-log says was made -- the ids `mark`ed by the op, or the whole mesh for a full
    audit.
    """
    seen = face_visibility(fids)
    return sorted(int(f) for f in expected if seen.get(int(f), 0) == 0)


def occluders(fids, depth, region, ahead_of):
    """Which faces cover `region`, given they are nearer than `ahead_of` metres.

    `region` is a boolean mask over the frame. Returns {face id: pixels} for the faces the
    camera sees there, so "what is in front of my canopy" is one call rather than an
    afternoon.
    """
    fid = np.asarray(fids)
    d = np.asarray(depth)
    m = np.asarray(region, bool) & (fid >= 0) & (d > 0) & (d < ahead_of)
    return face_visibility(np.where(m, fid, -1))


@dataclass
class OcclusionReport:
    changed_faces: int = 0
    hidden: list = field(default_factory=list)
    pixels_changed: float = 0.0
    verdict: str = "ok"
    note: str = ""

    def __str__(self):
        return (f"{self.verdict}: {self.changed_faces} faces changed, "
                f"{len(self.hidden)} of them invisible, "
                f"{self.pixels_changed * 100:.2f}% of pixels moved. {self.note}").strip()


def occlusion_report(before_fids, after_fids, changed_faces=(), rgb_before=None,
                     rgb_after=None, tol=2.0, suspicious_below=0.002):
    """Did a geometric edit reach the picture, or is something standing in front of it?

    `changed_faces` are the ids the edit created or moved. The two failure modes this
    catches are the ones that cost the most time, because neither raises anything:

      * every new face is invisible -- built inside something, or behind it;
      * a large edit moves almost no pixels, which is the same fault seen from the other
        side and the one that fools every difference metric.

    `suspicious_below` is a fraction of the frame, not an absolute count, so it means the
    same thing at any resolution. 0.2% is deliberately low: the point is to catch "this
    did essentially nothing", not to second-guess small deliberate edits.
    """
    rep = OcclusionReport(changed_faces=len(changed_faces))
    rep.hidden = hidden_faces(after_fids, changed_faces) if len(changed_faces) else []

    if rgb_before is not None and rgb_after is not None:
        a = np.asarray(rgb_before, float)
        b = np.asarray(rgb_after, float)
        if a.shape != b.shape:
            raise ValueError(f"frames differ in shape: {a.shape} vs {b.shape}")
        moved = np.abs(a - b).max(axis=-1) > tol if a.ndim == 3 else np.abs(a - b) > tol
        rep.pixels_changed = float(moved.mean())
    else:
        bf, af = np.asarray(before_fids), np.asarray(after_fids)
        if bf.shape != af.shape:
            raise ValueError(f"AOVs differ in shape: {bf.shape} vs {af.shape}")
        rep.pixels_changed = float((bf != af).mean())

    if len(changed_faces) and len(rep.hidden) == len(changed_faces):
        rep.verdict = "hidden"
        rep.note = ("Every face this edit touched renders zero pixels. It is inside or "
                    "behind something -- check what `occluders` finds in front of it.")
    elif len(changed_faces) >= 8 and len(rep.hidden) > 0.6 * len(changed_faces):
        rep.verdict = "mostly hidden"
        rep.note = (f"{len(rep.hidden)} of {len(changed_faces)} new faces are invisible. "
                    "Any score you read from this frame is scoring what is in front.")
    elif len(changed_faces) >= 8 and rep.pixels_changed < suspicious_below:
        rep.verdict = "suspect"
        rep.note = ("A large geometric edit moved almost no pixels. That has nearly "
                    "always meant it was occluded, not that it was subtle.")
    return rep
