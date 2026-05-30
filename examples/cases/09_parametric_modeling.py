"""Case 09 — Parametric hard-surface modeling: geometry as an editable op-graph.

Builds mechanical parts (an L-bracket, a flanged pipe, a perforated plate) from
feature programs (primitive + boolean CSG + patterns) via ``mirage.modeling``,
renders them, shows a *parametric edit* (change one parameter -> rebuild), and
assembles parts into a scene with spatial relations. Each part is a serializable
program an agent edits by feature/parameter — not by moving vertices.

    uv run python examples/cases/09_parametric_modeling.py
"""
from mirage import Session
from mirage.modeling import l_bracket, flanged_pipe, perforated_plate
from mirage.mujoco_backend import MujocoSim
from mirage.imaging import save_png
from mirage.relations import relation_sentences
from _util import outdir


def render_part(name, part, view, out, color=(0.62, 0.64, 0.7)):
    s = Session(name=name)
    s.add_plane("ground", size=[3, 3])
    s.add_part(name, part, position=[0, 0, 0], color=list(color), dynamic=False)
    sim = MujocoSim.from_scene(s.scene)
    save_png(sim.render(720, 540, **view)["rgb"], out / f"{name}.png")
    return part.stats()


def main() -> None:
    out = outdir("09_parametric_modeling")
    lib = [
        ("l_bracket", l_bracket(holes=3), dict(lookat=[0.2, 0, 0.2], distance=1.5, azimuth=125, elevation=-20)),
        ("flanged_pipe", flanged_pipe(bolts=8), dict(lookat=[0, 0, 0.3], distance=1.7, azimuth=120, elevation=-15)),
        ("perforated_plate", perforated_plate(nx=6, ny=4), dict(lookat=[0, 0, 0.06], distance=1.6, azimuth=120, elevation=-35)),
    ]
    print("== part library (each is a feature-graph program) ==")
    for name, part, view in lib:
        st = render_part(name, part, view, out)
        print(f"  {name}: {st['features']} features -> {st['faces']} faces, "
              f"watertight={st['watertight']}, vol={st['volume']}")

    print("\n== parametric edit: same program, change one parameter ==")
    plate_view = dict(lookat=[0, 0, 0.06], distance=1.6, azimuth=120, elevation=-35)
    render_part("plate_5x3", perforated_plate(nx=5, ny=3), plate_view, out)
    render_part("plate_9x5", perforated_plate(nx=9, ny=5), plate_view, out)
    print("  wrote plate_5x3.png and plate_9x5.png (one parameter changed, mesh rebuilt)")

    print("\n== assemble parts into a scene via intent ops (modeling + spatial layer) ==")
    s = Session(name="assembly")
    s.add_plane("ground", size=[3, 3])
    s.add_part("plate", perforated_plate(nx=5, ny=3), position=[0, 0, 0], color=[0.55, 0.57, 0.62])
    s.add_part("bracket", l_bracket(holes=2), position=[0, 0, 0], color=[0.7, 0.55, 0.3])
    s.add_part("pipe", flanged_pipe(), position=[0, 0, 0], color=[0.5, 0.6, 0.7])
    s.place_on("bracket", "plate")
    s.place_beside("pipe", "bracket", side="right", gap=0.12)
    view = dict(lookat=[0.1, 0, 0.15], distance=1.9, azimuth=125, elevation=-22)
    img, _ = s.set_of_mark(view=view, width=720, height=540)
    save_png(img, out / "assembly_som.png")
    for line in relation_sentences(s.relations(view=view)):
        print("  -", line)

    print(f"\nwrote {out}\\*.png (parts + parametric edit + labeled assembly)")
    print(f"every part is a program; the assembly was authored by intent, replayable in {len(s.get_log())} ops")


if __name__ == "__main__":
    main()
