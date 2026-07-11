"""mirage.oplog_diff — diff and 3-way merge for op-logs (git-for-3D).

The model is a **legible op-log** (an ordered list of op dicts), so — unlike an opaque
binary scene — two versions can be *diffed* and *merged* like source code. This is what
lets a human (GUI) and an AI (MCP / file) edit the same model on separate branches and
reconcile: disjoint edits merge automatically, overlapping ones surface as conflicts.

* :func:`diff` — a structured diff of two op-logs (added / removed / modified ops, with a
  per-field delta on a modify). :func:`format_diff` renders it git-style.
* :func:`merge3` — a 3-way merge (the classic diff3 over the op sequences): non-conflicting
  edits from both sides are taken; a region both sides changed differently is a conflict.

Both operate on plain lists of op dicts (``MeshProgram.ops`` / a loaded op-log JSON).
"""
from __future__ import annotations


def _lcs_pairs(a, b):
    """Index pairs ``(i, j)`` of a longest common subsequence of ``a`` and ``b`` (by ==)."""
    n, m = len(a), len(b)
    dp = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n - 1, -1, -1):
        for j in range(m - 1, -1, -1):
            dp[i][j] = dp[i + 1][j + 1] + 1 if a[i] == b[j] else max(dp[i + 1][j], dp[i][j + 1])
    pairs, i, j = [], 0, 0
    while i < n and j < m:
        if a[i] == b[j]:
            pairs.append((i, j)); i += 1; j += 1
        elif dp[i + 1][j] >= dp[i][j + 1]:
            i += 1
        else:
            j += 1
    return pairs


def _field_delta(old, new):
    """Per-field changes between two op dicts (ignoring 'op'): {field: (old, new)} where a
    missing side is None."""
    delta = {}
    for k in sorted(set(old) | set(new)):
        if k == "op":
            continue
        if old.get(k) != new.get(k):
            delta[k] = (old.get(k), new.get(k))
    return delta


def diff(a, b):
    """Structured diff of op-logs ``a`` -> ``b``. Returns a list of hunks:

    * ``("same", op)`` — unchanged
    * ``("del",  op)`` — removed from ``a``
    * ``("add",  op)`` — added in ``b``
    * ``("mod",  old, new, delta)`` — same op type, changed fields (``delta`` from :func:`_field_delta`)

    A remove immediately followed by an add of the *same op type* is coalesced into a modify.
    """
    pairs = _lcs_pairs(a, b)
    raw, i, j = [], 0, 0
    for (mi, mj) in pairs + [(len(a), len(b))]:
        while i < mi:
            raw.append(("del", a[i])); i += 1
        while j < mj:
            raw.append(("add", b[j])); j += 1
        if mi < len(a):
            raw.append(("same", a[mi])); i, j = mi + 1, mj + 1
    out, k = [], 0
    while k < len(raw):
        # coalesce a del+add of the SAME identified op (same op type and same `mark`) into a
        # modify. Requiring a shared mark avoids pairing two unrelated same-type ops (which
        # LCS can leave adjacent) into a bogus modify; unmarked ops stay as del + add.
        if (k + 1 < len(raw) and raw[k][0] == "del" and raw[k + 1][0] == "add"
                and raw[k][1].get("op") == raw[k + 1][1].get("op")
                and raw[k][1].get("mark") is not None
                and raw[k][1].get("mark") == raw[k + 1][1].get("mark")):
            old, new = raw[k][1], raw[k + 1][1]
            out.append(("mod", old, new, _field_delta(old, new)))
            k += 2
        else:
            out.append(raw[k]); k += 1
    return out


def _label(op):
    """A short, human tag for an op — type plus a salient hint (colour / position / size)."""
    t = op.get("op", "?")
    hint = ""
    m = op.get("material")
    if isinstance(m, dict) and "color" in m:
        c = m["color"]
        hint = f"  rgb({c[0]:.2f},{c[1]:.2f},{c[2]:.2f})" if all(isinstance(x, (int, float)) for x in c) else ""
    elif "translate" in op:
        hint = f"  @{tuple(op['translate'])}"
    elif "size" in op:
        hint = f"  size={op['size']}"
    return f"{t}{hint}"


def format_diff(hunks):
    """Render :func:`diff` output git-style (``+`` add, ``-`` del, ``~`` mod)."""
    lines = []
    for h in hunks:
        if h[0] == "same":
            lines.append(f"   {_label(h[1])}")
        elif h[0] == "add":
            lines.append(f"  + {_label(h[1])}")
        elif h[0] == "del":
            lines.append(f"  - {_label(h[1])}")
        elif h[0] == "mod":
            lines.append(f"  ~ {_label(h[1])}   [changed {', '.join(h[3])}]")
    return "\n".join(lines)


def merge3(base, ours, theirs):
    """3-way merge (diff3) of op-logs. ``base`` is the common ancestor; ``ours`` and
    ``theirs`` are the two branches. Returns ``(merged, conflicts)``: non-conflicting edits
    from both branches are combined; a region both branches changed *differently* becomes a
    conflict (``{"base", "ours", "theirs"}``) and ``ours`` is kept in ``merged`` there."""
    ma = dict(_lcs_pairs(base, ours))     # base index -> ours index
    mb = dict(_lcs_pairs(base, theirs))   # base index -> theirs index
    anchors, la, lb = [], -1, -1          # base indices stable in BOTH branches, aligned
    for i in range(len(base)):
        if i in ma and i in mb and ma[i] > la and mb[i] > lb:
            anchors.append(i); la, lb = ma[i], mb[i]

    merged, conflicts = [], []
    bi = oi = ti = 0
    for anchor in anchors + [None]:
        be, oe, te = (len(base), len(ours), len(theirs)) if anchor is None else (anchor, ma[anchor], mb[anchor])
        base_s, ours_s, theirs_s = base[bi:be], ours[oi:oe], theirs[ti:te]
        if ours_s == base_s:
            merged.extend(theirs_s)                 # only theirs touched this region
        elif theirs_s == base_s:
            merged.extend(ours_s)                   # only ours touched it
        elif ours_s == theirs_s:
            merged.extend(ours_s)                   # both made the identical edit
        else:
            conflicts.append({"base": base_s, "ours": ours_s, "theirs": theirs_s})
            merged.extend(ours_s)                   # keep ours; the caller resolves conflicts
        if anchor is not None:
            merged.append(base[anchor])             # the stable anchor op itself
            bi, oi, ti = be + 1, oe + 1, te + 1
    return merged, conflicts


# --------------------------------------------------------------------------- #
# Key-based diff / merge — for SCENE op-logs, where each object is a `place` op with a
# stable identity (its `mark`). Reconciling per-object (not per-position) means two edits
# to *different* objects never spuriously conflict, even when the ops sit next to each
# other — the failure mode of a purely positional diff3 on adjacent edits.
# --------------------------------------------------------------------------- #
def _default_key(op):
    return op.get("mark")


def diff_by_key(a, b, key=_default_key):
    """Per-object diff of two scene op-logs, keyed by ``key`` (default: the op's ``mark``).
    Returns hunks: ``("same"|"add"|"del", k, op)`` or ``("mod", k, old, new, delta)``.
    Ops without a key are ignored (use :func:`diff` for unkeyed / linear op-logs)."""
    ka = {key(o): o for o in a if key(o) is not None}
    kb = {key(o): o for o in b if key(o) is not None}
    order = list(ka) + [k for k in kb if k not in ka]
    hunks = []
    for k in order:
        if k in ka and k in kb:
            if ka[k] == kb[k]:
                hunks.append(("same", k, ka[k]))
            else:
                hunks.append(("mod", k, ka[k], kb[k], _field_delta(ka[k], kb[k])))
        elif k in ka:
            hunks.append(("del", k, ka[k]))
        else:
            hunks.append(("add", k, kb[k]))
    return hunks


def format_key_diff(hunks):
    """Render :func:`diff_by_key` output, one object per line."""
    lines = []
    for h in hunks:
        if h[0] == "same":
            lines.append(f"   {h[1]}: {_label(h[2])}")
        elif h[0] == "add":
            lines.append(f"  + {h[1]}: {_label(h[2])}")
        elif h[0] == "del":
            lines.append(f"  - {h[1]}: {_label(h[2])}")
        elif h[0] == "mod":
            lines.append(f"  ~ {h[1]}: changed {', '.join(h[4])}")
    return "\n".join(lines)


def merge_by_key(base, ours, theirs, key=_default_key):
    """Per-object 3-way merge of scene op-logs (keyed by ``mark``). Each object is
    reconciled on its own: if only one branch changed it, take that; if both changed it the
    same way, take it; if both changed it *differently*, it's a conflict (``{"key","base",
    "ours","theirs"}``) and ours is kept. Adds/removes are handled per key. Returns
    ``(merged_ops, conflicts)``; object order follows base, then ours' then theirs' new keys."""
    kb = {key(o): o for o in base if key(o) is not None}
    ko = {key(o): o for o in ours if key(o) is not None}
    kt = {key(o): o for o in theirs if key(o) is not None}
    base_keys = [key(o) for o in base if key(o) is not None]
    order = base_keys + [k for k in ([key(o) for o in ours if key(o) is not None] +
                                     [key(o) for o in theirs if key(o) is not None])
                         if k not in kb]
    seen, ordered_keys = set(), []
    for k in order:
        if k not in seen:
            seen.add(k); ordered_keys.append(k)

    merged, conflicts = [], []
    for k in ordered_keys:
        b, o, t = kb.get(k), ko.get(k), kt.get(k)
        if o == b:            # ours left it alone -> theirs' verdict (change / add / delete)
            chosen = t
        elif t == b:          # theirs left it alone -> ours
            chosen = o
        elif o == t:          # both made the identical change
            chosen = o
        else:
            conflicts.append({"key": k, "base": b, "ours": o, "theirs": t})
            chosen = o        # keep ours; the caller resolves
        if chosen is not None:
            merged.append(chosen)
    # carry any unkeyed base ops (kept positionally at the front)
    unkeyed = [o for o in base if key(o) is None]
    return unkeyed + merged, conflicts
