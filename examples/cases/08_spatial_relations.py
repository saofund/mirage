"""Case 08 — The AI-native spatial layer: relation graph + Set-of-Mark grounding.

Author a desk scene, lift it into an explicit **spatial relation graph** (what is
on / in / next-to / left-of / aligned-with what, in the viewer frame), render a
**Set-of-Mark** image (every object tagged by id), then issue **intent-level edits**
(stack / place_on / place_beside — no coordinates) and show the relations + image
update. This is the representation an LLM reasons over instead of vertex soup.

    uv run python examples/cases/08_spatial_relations.py
"""
import json

from mirage import Session
from mirage.relations import relation_sentences
from mirage.imaging import save_png
from _util import outdir

VIEW = {"lookat": [0, 0, 0.55], "distance": 2.4, "azimuth": 90, "elevation": -25}


def build(s: Session) -> None:
    s.add_box("table", position=[0, 0, 0.25], size=[1.4, 0.9, 0.5], color=[0.45, 0.3, 0.2], dynamic=False)
    s.add_box("book_red", position=[-0.45, 0.10, 0.525], size=[0.30, 0.22, 0.05], color=[0.8, 0.2, 0.2])
    s.add_box("book_blue", position=[-0.10, 0.10, 0.525], size=[0.30, 0.22, 0.05], color=[0.2, 0.4, 0.85])
    s.add_cylinder("mug", position=[0.30, 0.15, 0.560], radius=0.06, height=0.12, color=[0.9, 0.85, 0.85])
    s.add_cylinder("lamp", position=[0.50, -0.15, 0.750], radius=0.05, height=0.50, color=[0.95, 0.8, 0.2])
    s.add_box("tray", position=[-0.40, -0.20, 0.530], size=[0.40, 0.30, 0.06], color=[0.3, 0.3, 0.32])
    s.add_cylinder("coin", position=[0, 0, 2.0], radius=0.04, height=0.02, color=[0.9, 0.75, 0.2])
    s.place_inside("coin", "tray")


def snapshot(s: Session, out, tag: str) -> dict:
    g = s.relations(view=VIEW)
    (out / f"relations_{tag}.json").write_text(json.dumps(g, indent=2), encoding="utf-8")
    img, _ = s.set_of_mark(view=VIEW)
    save_png(img, out / f"som_{tag}.png")
    return g


def main() -> None:
    out = outdir("08_spatial_relations")
    s = Session(name="desk")
    build(s)

    g = snapshot(s, out, "before")
    print(f"== spatial scene graph: {len(g['relations'])} relations ==")
    for line in relation_sentences(g):
        print("  -", line)

    print("\n== intent-level edits (no coordinates) ==")
    s.stack(["book_blue"], "book_red"); print("  stack book_blue on book_red")
    s.place_on("mug", "book_blue");     print("  place mug on book_blue")
    s.place_beside("tray", "lamp", side="left", gap=0.06); print("  tray to the left of lamp")

    g2 = snapshot(s, out, "after")
    print(f"\n== after: {len(g2['relations'])} relations ==")
    for line in relation_sentences(g2):
        print("  -", line)

    print(f"\nwrote {out}\\relations_before.json, relations_after.json, som_before.png, som_after.png")
    print(f"every edit was intent-level and logged — replayable in {len(s.get_log())} ops")


if __name__ == "__main__":
    main()
