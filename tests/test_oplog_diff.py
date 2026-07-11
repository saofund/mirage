import copy

from mirage.oplog_diff import (diff, format_diff, merge3, diff_by_key, format_key_diff,
                               merge_by_key)

CUBE = {"op": "cube", "size": 1.0}


def place(mark, color):
    return {"op": "place", "program": [{"op": "cube", "size": 1.0}],
            "translate": [0, 0, 0], "material": {"color": list(color)}, "mark": mark}


# ---- positional diff / merge3 (for linear op histories) ------------------- #
def test_positional_diff_add_del():
    a = [CUBE, {"op": "inset", "on": {"by": "all"}, "thickness": 0.3}]
    b = a + [{"op": "extrude", "on": {"by": "last_created"}, "distance": 0.5}]
    assert [h[0] for h in diff(a, b)] == ["same", "same", "add"]
    assert [h[0] for h in diff(b, a)] == ["same", "same", "del"]
    assert "+ " in format_diff(diff(a, b))


def test_positional_diff_mod_needs_same_mark():
    a = [place("vase", [0, 0, 1])]
    b = [place("vase", [1, 0, 0])]                       # same marked op, recoloured
    h = diff(a, b)
    assert h[0][0] == "mod" and "material" in h[0][3]
    # two DIFFERENT marked ops must NOT coalesce into a bogus modify
    a2 = [place("vase", [0, 0, 1])]
    b2 = [place("lamp", [1, 0, 0])]
    assert [x[0] for x in diff(a2, b2)] == ["del", "add"]


def test_merge3_disjoint_and_conflict():
    base = [CUBE, place("vase", [0, 0, 1]), place("lamp", [1, 1, 1])]
    ours = [CUBE, place("vase", [1, 0.5, 0]), place("lamp", [1, 1, 1])]   # recolour vase
    theirs = base + [place("book", [0.5, 0.5, 0.5])]                      # add book (separated by lamp)
    merged, conflicts = merge3(base, ours, theirs)
    assert conflicts == [] and len(merged) == 4
    assert place("vase", [1, 0.5, 0]) in merged and place("book", [0.5, 0.5, 0.5]) in merged
    # same op changed by both -> conflict
    ot = [CUBE, place("vase", [0, 1, 0])]
    _, cc = merge3([CUBE, place("vase", [0, 0, 1])], [CUBE, place("vase", [1, 0, 0])], ot)
    assert len(cc) == 1


# ---- key-based diff / merge (for scenes: place ops with a `mark`) ---------- #
def test_diff_by_key_per_object():
    a = [CUBE, place("vase", [0, 0, 1]), place("lamp", [1, 1, 1])]
    b = [CUBE, place("vase", [1, 0.5, 0]), place("book", [0.5, 0.5, 0.5])]
    byk = {h[1]: h[0] for h in diff_by_key(a, b)}
    assert byk == {"vase": "mod", "lamp": "del", "book": "add"}
    mod = next(h for h in diff_by_key(a, b) if h[1] == "vase")
    assert "material" in mod[4]
    assert "book" in format_key_diff(diff_by_key(a, b))


def test_merge_by_key_disjoint_no_conflict():
    base = [place("floor", [.5, .5, .5]), place("vase", [0, 0, 1]), place("book", [1, 0, 0])]
    ours = copy.deepcopy(base); ours[1]["material"]["color"] = [1, .5, 0]     # human recolours vase
    theirs = copy.deepcopy(base); theirs[0]["material"]["color"] = [.6, .4, .2]  # AI repaints floor
    theirs.append(place("book2", [0, 0, 1]))                                  # AI adds a book
    merged, conflicts = merge_by_key(base, ours, theirs)
    assert conflicts == []
    cols = {o["mark"]: o["material"]["color"] for o in merged}
    assert cols["vase"] == [1, .5, 0] and cols["floor"] == [.6, .4, .2] and "book2" in cols
    assert len(merged) == 4                              # both branches' edits applied


def test_merge_by_key_conflict_and_delete():
    base = [place("vase", [0, 0, 1])]
    ours = [place("vase", [1, 0, 0])]                    # human -> red
    theirs = [place("vase", [0, 1, 0])]                  # AI -> green
    merged, conflicts = merge_by_key(base, ours, theirs)
    assert len(conflicts) == 1 and conflicts[0]["key"] == "vase"
    assert merged[0]["material"]["color"] == [1, 0, 0]   # ours kept
    # a delete by one side, untouched by the other, wins
    m2, c2 = merge_by_key(base, [], base)
    assert m2 == [] and c2 == []
