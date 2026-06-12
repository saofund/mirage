"""Case 15 — one op-log, two operators: the AI and the human share the SAME model.

This is the dual-operator thesis made concrete. The op-log is the single source
of truth, serialized as a JSON dialect that BOTH engines speak:

  * the Python meshlang.MeshProgram (what an AI drives over MCP), and
  * the native C++ mirage::Program (what the GUI, mirage_viewer, replays).

Here the "AI" authors a stepped pedestal and saves it. The native viewer can
then open that exact file (`mirage_viewer --oplog mirage_oplog.json`) and the
human keeps modeling — no export/translation step, because it was never a
different model. The reverse is identical: whatever the human saves, this script
loads back. The C++/Python differential test (tests/test_cpp_program.py) proves
the two engines replay any such op-log to the identical mesh.

    uv run python examples/cases/15_shared_oplog.py
"""
import json
import os

from mirage.meshlang import MeshProgram, Sel

SHARED = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "mirage_oplog.json")


def author_as_ai() -> MeshProgram:
    """The AI builds a stepped pedestal with selection-as-query (no indices)."""
    p = MeshProgram()
    p.cube(1.2)
    p.inset(Sel.normal("z"), 0.25)       # the top, by its normal
    p.extrude(Sel.last(), 0.5)           # lift the inner face -> tier 1
    p.inset(Sel.last(), 0.3)             # inset what we just made
    p.extrude(Sel.last(), 0.4)           # -> tier 2
    p.tag(Sel.last(), "lid")             # a durable handle for later
    return p


def main():
    p = author_as_ai()
    mesh = p.build()
    print("AI authored a stepped pedestal:")
    print("  ", mesh.stats())

    with open(SHARED, "w", encoding="utf-8") as f:
        f.write(p.to_json())
    print(f"  saved op-log ({len(p.ops)} ops) -> {SHARED}")
    print("  open it in the GUI:  mirage_viewer --oplog mirage_oplog.json")
    print("  for LIVE co-editing: tick 'live sync' in the GUI — then every save_mesh_program")
    print("  the AI does auto-reloads in the viewport (and the human's edits auto-save back).")

    # ... the human edits in the GUI and saves. Loading it back is symmetric: it's
    # the same op-log, no translation. (Here we just round-trip our own file.)
    with open(SHARED, "r", encoding="utf-8") as f:
        reloaded = MeshProgram(json.load(f))
    assert reloaded.build().stats() == mesh.stats(), "round-trip changed the model"
    print(f"  reloaded op-log -> same mesh {reloaded.build().stats()['faces']} faces  [ok]")
    print("\none model. a human and an AI both editing it.")


if __name__ == "__main__":
    main()
