"""Case 27 — an infinite labelled dataset of a real part: the inner fuel cap (内盖).

Case 26 rebuilt one photograph. This one goes the other way: it takes a real perception
problem — estimate the 6D pose of the cap inside a car's fuel-filler pocket, so a refuelling
robot can grip and turn it — and replaces its data. The customer's set is ~30k hand-labelled
frames across ninety cars and three cameras. This case generates frames that drop into that
same pipeline, with exact labels, at 2.5 s each, without limit.

The interesting part is not the renderer. It is what the frames have to match to be worth
anything, and how each of those was found:

* **the shape.** Measured, not styled. `fit.py` reduces 376 real clouds to a canonical
  height field of the cap face; `parts.cap` reproduces it. Steered by
  `python -m fuelcap.fit --audit`, which puts eleven statistics of the real set beside the
  synthetic one. The rib was 30 mm wide against the real 44 until that table said so.
* **the sensor.** A path tracer returns a perfect depth map and a real Orbbec does not: it
  returns a surface plus 0.4 mm of scatter, 40 % of its pixels missing, and flying pixels
  along every occlusion edge. All three are measured off the real clouds. The dropout model
  was wrong twice — "dark surfaces fail" is backwards here, and brightness is only a proxy
  for the thing that actually matters, which is local texture.
* **the label.** Verified, not assumed. `--check-labels` ignores the stored pose, fits a
  plane to the cap's own annulus, and compares. It found that the cap's screw-stop angle was
  being composed outside its tilt, mislabelling every frame in the set by a median of 8
  degrees while every render looked perfect.

Two engine features exist because this case needed them: a metric **depth AOV**
(`mirage_render --depth`, a float PFM of view-axis distance) and `--no-ground`. With the
existing object-id AOV, that is everything required to turn a render into a labelled cloud.

    uv run python examples/cases/27_fuelcap.py                  # a small set + the audit
    uv run python examples/cases/27_fuelcap.py -n 2000          # a training-sized set
    uv run python examples/cases/27_fuelcap.py --sheets         # look at the parts alone
    python -m fuelcap.fit --audit  <dir>                        # real vs synthetic, 11 rows
    python -m fuelcap.fit --check-labels <dir>                  # are the labels true?

Output is one ``.npz`` per frame carrying ``xyz, rgb, label, K_norm, w, h, anchor, normal,
cap_x, obb_px`` — the format ``pipeline/build_clouds.py`` produces, so the annotation server
and the training loader both open it unchanged. See `fuelcap/README.md` for the registry
entry.

Needs mirage_render + numpy + opencv (Pillow for the preview PNGs).
"""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from fuelcap import dataset, fit, sheet   # noqa: E402

OUT = Path(__file__).resolve().parent / "outputs" / "27_fuelcap"


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("-n", type=int, default=48, help="frames to generate")
    ap.add_argument("--out", default=str(OUT))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--domain", default="wide", choices=("wide", "prod", "orbbec"),
                    help="camera envelope: prod = the robot arm, orbbec = handheld, "
                         "wide = the union of both, widened")
    ap.add_argument("--camera", default="orbbec640", help="orbbec640 | prod1280")
    ap.add_argument("--spp", type=int, default=32)
    ap.add_argument("--threads", type=int, default=0)
    ap.add_argument("--sheets", action="store_true", help="also render the review sheets")
    ap.add_argument("--no-audit", action="store_true")
    a = ap.parse_args()

    print(f"generating {a.n} frames -> {a.out}")
    summary = dataset.generate(a.out, n=a.n, seed=a.seed, camera=a.camera, domain=a.domain,
                               spp=a.spp, threads=a.threads, keep_png=min(8, a.n))
    print(json.dumps(summary, indent=2))

    if a.sheets:
        sheet.OUT.mkdir(parents=True, exist_ok=True)
        sheet.parts_sheet()
        sheet.scenes_sheet(6, a.seed, a.domain)
        sheet.ids_sheet(a.seed, 2, a.domain)
        try:
            sheet.roi_sheet(a.out, 8, a.seed)
        except Exception as e:                       # needs the pulled reference set
            print(f"(roi sheet skipped: {e})")

    if not a.no_audit and summary["frames"]:
        try:
            fit.audit(a.out)
        except Exception as e:
            print(f"(audit needs the reference clouds in fuelcap/_ref: {e})")
        fit.check_labels(a.out)


if __name__ == "__main__":
    main()
