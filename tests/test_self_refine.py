"""Guard the self-refinement demo (case 21): the rounds must actually fix the flaws and
converge to a valid scene, so the narrative in the gallery strip stays true."""
import importlib.util
from pathlib import Path

from mirage.meshlang import MeshProgram


def _case():
    p = Path(__file__).resolve().parents[1] / "examples" / "cases" / "21_self_refine.py"
    spec = importlib.util.spec_from_file_location("case21", p)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def test_refinement_grounds_and_converges():
    m = _case()
    s = m.flawed()
    # the deliberate flaws are present at the start
    assert m.by_mark(s["ops"], "vase")["translate"][2] > 0.2          # vase floats
    assert m.by_mark(s["ops"], "book")["translate"][2] > 0.2          # book floats
    assert s["exposure"] > 1.3                                        # over-exposed
    for edit, _title, _crit in m.STEPS:
        if edit is not None:
            s = edit(s)
    # converged: objects grounded, exposure fixed, a warm accent added, scene still valid
    assert m.by_mark(s["ops"], "vase")["translate"][2] == 0.0
    assert m.by_mark(s["ops"], "book")["translate"][2] < 0.1
    assert s["exposure"] <= 1.1
    assert any(o.get("mark") == "fruit" for o in s["ops"])
    MeshProgram(s["ops"]).build().validate()                         # the refined op-log builds
