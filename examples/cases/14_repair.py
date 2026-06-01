"""Case 14 — repair: how the AI's mistakes get caught and fixed.

An LLM driving the kernel makes four kinds of mistake. This shows what repair.py
does with each:

  1. a TYPO / too-tight tol / scalar scale / numeric string  -> AUTO-repaired
     (high-confidence, intent-preserving — applied silently, reported).
  2. a reference to something that doesn't exist (a tag never created) -> a
     structured DIAGNOSTIC + ranked SUGGESTIONS for the agent to choose (never a
     silent guess about which faces were meant).
  3. a SILENT trap that builds cleanly but loses intent (extrude distance 0,
     which='highest' meaning min) -> caught by the LINT pass (build() can't see it).
  4. a geometry defect in a built/imported mesh (zero-area faces, orphan verts,
     an unwelded seam) -> repair_mesh cleans it, reporting exactly what changed.

    uv run python examples/cases/14_repair.py
"""
from mirage.meshlang import MeshProgram
from mirage.kernel import Mesh, make_cube
from mirage.repair import repair_program, lint_program, repair_mesh


def show_repair(label, ops):
    r = repair_program(ops)
    if r.repaired:
        print(f"  [{label}] AUTO -> {r.applied['label']}   ({r.applied['rationale']})")
    elif r.ok:
        print(f"  [{label}] already valid")
    else:
        kinds = ", ".join(f"{s['label']}" for s in r.suggestions[:3])
        print(f"  [{label}] no safe auto-fix ({r.diagnostic['kind']}); {len(r.suggestions)} suggestions: {kinds} ...")
    return r


def main():
    base = [{"op": "cube"}, {"op": "tag", "on": {"by": "normal", "axis": "z", "sign": 1}, "name": "top"}]

    print("--- 1. mistakes that auto-repair (intent preserved) ---")
    show_repair("tag typo",   base + [{"op": "extrude", "on": {"by": "tag", "name": "tpo"}, "distance": 0.4}])
    show_repair("tol=1.5",    [{"op": "cube"}, {"op": "extrude", "on": {"by": "normal", "axis": "z", "sign": 1, "tol": 1.5}, "distance": 0.3}])
    show_repair("scalar scale", [{"op": "cube"}, {"op": "scale", "on": {"by": "all"}, "by": 0.5}])
    show_repair("distance '0.5'", [{"op": "cube"}, {"op": "extrude", "on": {"by": "all"}, "distance": "0.5"}])
    show_repair("op 'extrod'", [{"op": "cube"}, {"op": "extrod", "on": {"by": "all"}, "distance": 0.5}])

    print("\n--- 2. mistakes that need the agent to choose (suggested, never silent) ---")
    show_repair("sides=2",     [{"op": "cylinder", "sides": 2, "radius": 0.5, "height": 1}])   # magnitude pick
    show_repair("missing tag 'rim'", [{"op": "cube"}, {"op": "extrude", "on": {"by": "tag", "name": "rim"}, "distance": 0.3}])
    show_repair("scale by 0",   [{"op": "cube"}, {"op": "scale", "on": {"by": "all"}, "by": [0, 0, 0]}])
    show_repair("assert euler=9", [{"op": "cube"}, {"op": "assert", "euler": 9}])
    show_repair("top AND bottom", [{"op": "cube"}, {"op": "extrude", "on": {"and": [
        {"by": "normal", "axis": "z", "sign": 1}, {"by": "normal", "axis": "z", "sign": -1}]}, "distance": 0.3}])

    print("\n--- 3. silent traps the LINT pass catches (these build fine!) ---")
    trap = [{"op": "cube"},
            {"op": "extrude", "on": {"by": "all"}, "distance": 0, "mark": "x"},
            {"op": "inset", "on": {"by": "extreme", "axis": "z"}, "thickness": 5},
            {"op": "extrude", "on": {"by": "extreme", "axis": "z", "which": "highest"}, "distance": 0.2}]
    for w in lint_program(trap):
        print(f"  op#{w['op_index']} [{w['code']}] {w['message']}")

    print("\n--- 4. geometry cleanup of a defective mesh ---")
    m = make_cube(1.0)
    m.add_vert((9, 9, 9))                                  # an orphan/loose vertex
    cleaned, report = repair_mesh(m)
    print(f"  cube + loose vert -> welded={report['welded_verts']} "
          f"dropped_orphans={report['dropped_orphans']} "
          f"faces {report['faces_before']}->{report['faces_after']} (manifold={cleaned.is_closed_manifold()})")
    seam = Mesh.from_pydata([(0, 0, 0), (1, 0, 0), (0, 1, 0), (1, 0, 0), (0, 1, 0), (1, 1, 0)],
                            [[0, 1, 2], [3, 4, 5]])         # two tris with a duplicated, unwelded seam
    _c, rep = repair_mesh(seam)
    print(f"  split seam -> welded={rep['welded_verts']} verts {rep['verts_before']}->{rep['verts_after']} "
          f"(the two tris now share an edge instead of a split seam)")


if __name__ == "__main__":
    main()
