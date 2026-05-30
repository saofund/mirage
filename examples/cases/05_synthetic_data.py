"""Case 05 — Synthetic data via domain randomization (Replicator-style).

Thin wrapper over ``mirage.synthetic.generate_dataset`` — the pipeline is now a
first-class API. Generates labeled RGB samples (+ box overlays) and a COCO
``annotations.json``, then assembles a 3x3 contact sheet.

    uv run python examples/cases/05_synthetic_data.py
"""
from mirage.synthetic import generate_dataset
from _util import outdir

N = 9


def main() -> None:
    out = outdir("05_synthetic_data")
    summary = generate_dataset(out, n=N, seed=0, width=640, height=480)

    from PIL import Image
    thumbs = [Image.open(out / f"boxes_{i:02d}.png").resize((320, 240)) for i in range(N)]
    sheet = Image.new("RGB", (320 * 3, 240 * 3), (15, 15, 18))
    for i, t in enumerate(thumbs):
        sheet.paste(t, ((i % 3) * 320, (i // 3) * 240))
    sheet.save(out / "contact_sheet.png")

    print(f"generated {summary['samples']} samples, {summary['boxes']} labeled boxes "
          f"across {len(summary['classes'])} classes {summary['classes']}")
    print(f"wrote {out/'annotations.json'}, contact_sheet.png, rgb_*.png, boxes_*.png")


if __name__ == "__main__":
    main()
