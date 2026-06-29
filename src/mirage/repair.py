"""repair — structured diagnostics + bounded auto-repair for meshlang programs.

When an LLM drives the modeling kernel it *will* emit broken ops: a mistyped
tag, a tol so tight nothing matches, a scale-to-zero that collapses a face, a
number sent as a JSON string. ``meshlang.build()`` already fails loudly and
localises the error to its op; this module turns that failure into something an
agent (or the MCP server) can act on:

* :func:`diagnose` — classify a failure into a structured :class:`Diagnostic`
  (kind, the offending op, the live mesh summary, a *ranked* candidate list).
* :func:`repair_program` — **bounded-retry** auto-repair: try the ranked
  candidates, rebuilding + validating the whole program for each, and APPLY the
  first high-confidence, intent-preserving fix. Anything that would change
  *which* faces are targeted (selector-kind swaps, ``{by:all}``, dropping an
  assert, picking a magnitude) is never applied silently — it is returned as a
  *suggestion* for the agent. This is the safety invariant.
* :func:`lint_program` — a proactive pass for the dangerous class that raises
  **no exception**: a zero-distance extrude (a silent no-op that still stamps its
  mark), an inset thickness that gets silently clamped, ``which != 'max'`` which
  silently means *min*, ``last_created`` after a primitive/subdivide resolving to
  the whole surface. ``build()`` can't catch these; only reading the op-log can.
* :func:`repair_mesh` — geometry-level cleanup of a built/imported ``Mesh``:
  weld coincident verts, drop zero-area faces + orphan verts, and *detect but
  never silently delete* non-manifold edges.

The taxonomy, the rank order, and every "auto vs suggest" call are grounded in
an empirical sweep of the kernel's real failure behaviour; ``tests/test_repair.py``
carries the reproducing cases.
"""
from __future__ import annotations

import difflib
import json
import re
from dataclasses import dataclass, field

from .kernel import Mesh, face_normal, _compact, _copy_attrs
from .meshlang import MeshProgram, SelectorEmpty, describe, resolve, _tags

KNOWN_OPS = ["cube", "cylinder", "plane", "uv_sphere", "cone", "torus", "grid", "mesh",
             "extrude", "inset", "bevel", "loop_cut", "edge_bevel",
             "solidify", "mirror", "array", "bisect", "spin", "screw",
             "delete", "bridge", "fill", "subdivide", "tag", "material", "translate", "scale", "assert"]
KNOWN_BY = ["all", "normal", "tag", "extreme", "side", "last_created", "near", "material", "connected"]
_PARAM_SIG = {  # a param key -> the op it most likely belongs to (for op-name inference)
    "distance": "extrude", "thickness": "inset", "width": "bevel", "depth": "bevel",
    "levels": "subdivide", "size": "cube",
    "sides": "cylinder", "radius": "cylinder", "height": "cylinder",
    "segments": "uv_sphere", "rings": "uv_sphere",
    "major_segments": "torus", "minor_segments": "torus", "major_radius": "torus", "minor_radius": "torus",
    "x_div": "grid", "y_div": "grid",
    "count": "array", "offset": "array",
    "angle": "spin", "turns": "screw",
}
PRIMITIVE_OPS = ("cube", "cylinder", "plane", "uv_sphere", "cone", "torus", "grid", "mesh")
MAX_SUBDIVIDE = 6
_DEGENERATE = 1e-9


# --------------------------------------------------------------------------- #
# Structured results
# --------------------------------------------------------------------------- #
@dataclass
class Candidate:
    """One proposed fix for a failing op. ``op`` is the replacement (or, for
    ``action='insert_before'``, the op to insert). ``apply_mode`` is the whole
    point: ``auto`` may be applied silently, ``suggest`` must go to the agent."""
    label: str
    apply_mode: str            # "auto" | "suggest"
    confidence: str            # "high" | "low"
    rationale: str
    op: dict | None = None
    action: str = "replace"    # "replace" | "insert_before" | "drop"
    validated: bool | None = None   # filled in by the probe loop (did it build?)

    def to_dict(self) -> dict:
        return {"label": self.label, "action": self.action, "op": self.op,
                "apply_mode": self.apply_mode, "confidence": self.confidence,
                "rationale": self.rationale, "validated": self.validated}


@dataclass
class Diagnostic:
    kind: str
    message: str
    op_index: int | None
    op: dict | None = None
    mesh_summary: dict = field(default_factory=dict)
    selector_diagnostics: dict = field(default_factory=dict)
    inner: dict = field(default_factory=dict)         # {type, detail} for wrapped kernel errors
    candidates: list = field(default_factory=list)    # list[Candidate], ranked
    notes: list = field(default_factory=list)
    prefix: object = None                             # the live prefix Mesh (not serialized)

    def to_dict(self) -> dict:
        return {"kind": self.kind, "message": self.message, "op_index": self.op_index,
                "op": self.op, "mesh_summary": self.mesh_summary,
                "selector_diagnostics": self.selector_diagnostics, "inner": self.inner,
                "candidates": [c.to_dict() for c in self.candidates], "notes": self.notes}


@dataclass
class RepairResult:
    ok: bool                 # program builds (already valid, or after an auto fix)
    repaired: bool           # an auto fix was applied
    program: list            # the (possibly repaired) op list
    applied: dict | None = None
    diagnostic: dict | None = None
    attempts: list = field(default_factory=list)
    suggestions: list = field(default_factory=list)

    def to_dict(self) -> dict:
        return {"ok": self.ok, "repaired": self.repaired, "program": self.program,
                "applied": self.applied, "diagnostic": self.diagnostic,
                "attempts": self.attempts, "suggestions": self.suggestions}


# --------------------------------------------------------------------------- #
# Build probing (re-derive the failing op robustly — SelectorEmpty carries no index)
# --------------------------------------------------------------------------- #
def _builds(ops) -> tuple[bool, Exception | None]:
    try:
        MeshProgram([dict(o) for o in ops]).build()
        return True, None
    except Exception as exc:   # noqa: BLE001 — we classify it downstream
        return False, exc


def _first_failing(ops) -> tuple[int | None, Exception | None, Mesh | None]:
    """(index, exception, prefix_mesh) for the first op whose prefix build fails. Works
    for every error kind because it never trusts the message for the index, and returns
    the build of ops[:idx] (the live mesh that op saw) for free, so callers never rebuild
    the prefix — important when a slow subdivide sits earlier in the program."""
    prev = None
    for i in range(len(ops)):
        try:
            prev = MeshProgram([dict(o) for o in ops[: i + 1]]).build()
        except Exception as exc:   # noqa: BLE001
            return i, exc, prev    # prev is the build of ops[:i]
    return None, None, prev


# --------------------------------------------------------------------------- #
# Classification — two message grammars + the structured SelectorEmpty
# --------------------------------------------------------------------------- #
_INVALID_RE = re.compile(r"^op #(\d+) '([^']*)' produced an invalid mesh: (.*)$", re.S)
_WRAP_RE = re.compile(r"^op #(\d+) '([^']*)': (\w+): (.*)$", re.S)
_ASSERT_EULER_RE = re.compile(r"assert euler=(-?\d+) failed \(got (-?\d+)\)")


def _coerce_levels(v):
    try:
        return int(round(float(v)))
    except (ValueError, TypeError):
        return None


def _as_ops(program) -> list:
    return list(program.ops if isinstance(program, MeshProgram) else program)


def _precheck(ops) -> Diagnostic | None:
    """Cheap *static* guards that must NOT build — they prevent crashes (non-dict ops)
    and hangs (a runaway subdivide whose ~4^levels build would never return). Returns a
    short-circuit Diagnostic, or None to proceed to the normal build-based path."""
    if not ops:
        return Diagnostic("structural", "empty program", None, None,
                          candidates=[Candidate("seed {op:cube}", "suggest", "low",
                                                "an empty program has no geometry", op={"op": "cube", "size": 1.0},
                                                action="insert_before")])
    for i, op in enumerate(ops):
        if not isinstance(op, dict):
            return Diagnostic("malformed_op", f"op #{i} is not a dict: {op!r}", i, None,
                              notes=["each op must be a dict like {'op':'extrude', ...}"])
    for i, op in enumerate(ops):
        if op.get("op") == "subdivide":
            lv = _coerce_levels(op.get("levels", 1))
            if lv is None or lv > MAX_SUBDIVIDE:
                d = Diagnostic("cost", f"subdivide levels={op.get('levels')!r} is too expensive to build "
                               f"(faces grow ~4^levels); refusing to build", i, op)
                d.candidates = [Candidate(f"levels -> {MAX_SUBDIVIDE}", "suggest", "low",
                                          f"cap subdivision at {MAX_SUBDIVIDE} to stay tractable",
                                          op=_set(op, levels=MAX_SUBDIVIDE))]
                return d
    return None


def _classify(ops, idx, exc, mesh) -> Diagnostic:
    op = ops[idx] if (idx is not None and 0 <= idx < len(ops)) else None
    summary = describe(mesh) if mesh is not None else {}

    if isinstance(exc, SelectorEmpty):
        d = Diagnostic("selector_empty", str(exc), idx, op, summary,
                       selector_diagnostics=dict(exc.diagnostics))
    else:
        msg = str(exc)
        if _INVALID_RE.match(msg):
            d = Diagnostic("invalid_mesh", msg, idx, op, summary,
                           notes=[f"validate() rejected: {_INVALID_RE.match(msg).group(3)}"])
        elif msg.startswith("assert "):
            d = Diagnostic("assert_failed", msg, idx, op, summary)
        elif _WRAP_RE.match(msg):
            m = _WRAP_RE.match(msg)
            d = Diagnostic("kernel_error", msg, idx, op, summary,
                           inner={"type": m.group(3), "detail": m.group(4)})
        elif "before any primitive" in msg or msg.startswith("unknown op") or msg == "empty program":
            d = Diagnostic("structural", msg, idx, op, summary)
        elif "selector must be a dict" in msg or msg.startswith("unknown selector"):
            d = Diagnostic("malformed_selector", msg, idx, op, summary)
        else:
            d = Diagnostic("unknown", msg, idx, op, summary)
    d.prefix = mesh
    return d


# --------------------------------------------------------------------------- #
# Candidate generators (ranked; auto = high-confidence & intent-preserving only)
# --------------------------------------------------------------------------- #
def _with_on(op, sel) -> dict:
    out = dict(op); out["on"] = sel; return out


def _set(base, **changes) -> dict:
    out = dict(base); out.update(changes); return out


def _hist_count(diags, axis, sign) -> int:
    key = ("+" if sign >= 0 else "-") + axis
    return diags.get("normal_histogram", {}).get(key, 0)


def _best_tag(name, tags):
    """(match, is_unique_high): a single difflib match at cutoff>=0.6 with a clear
    lead over the runner-up — the only case safe to auto-apply."""
    ranked = sorted(tags, key=lambda t: difflib.SequenceMatcher(None, name, t).ratio(), reverse=True)
    if not ranked:
        return None, False
    top = ranked[0]
    r0 = difflib.SequenceMatcher(None, name, top).ratio()
    r1 = difflib.SequenceMatcher(None, name, ranked[1]).ratio() if len(ranked) > 1 else 0.0
    return top, (r0 >= 0.6 and r0 - r1 > 1e-6)


def _kind_substitutions(op, diags, low_first=False):
    """Selector-kind swaps derived from the histogram — ALWAYS suggest (they change
    which faces are targeted)."""
    cands, groups = [], diags.get("normal_histogram", {})
    for key, _cnt in sorted(groups.items(), key=lambda kv: -kv[1]):
        sign, axis = (1 if key[0] == "+" else -1), key[1]
        cands.append(Candidate(
            f"normal {key} -> extreme {axis} {'max' if sign > 0 else 'min'}", "suggest", "low",
            "selector-kind swap (axis-alignment -> argmax of centroid): changes which face is picked",
            op=_with_on(op, {"by": "extreme", "axis": axis, "which": "max" if sign > 0 else "min"})))
    cands.append(Candidate("fallback -> {by:all}", "suggest", "low",
                           "edits EVERY face — only if you truly mean the whole mesh",
                           op=_with_on(op, {"by": "all"})))
    return cands


def _gen_selector_empty(op, diag):
    sel = op.get("on", {})
    diags = diag.selector_diagnostics
    cands = []
    if not isinstance(sel, dict):
        return cands
    if any(k in sel for k in ("and", "or", "not")):
        return _gen_composite(op, sel, diag)
    by = sel.get("by")
    if by == "tag":
        name = sel.get("name", "")
        tags = diags.get("tags", [])
        match, unique = _best_tag(name, tags)
        if match is not None:
            cands.append(Candidate(
                f"tag '{name}' -> '{match}'", "auto" if unique else "suggest",
                "high" if unique else "low", "fuzzy tag match (same selector kind)",
                op=_with_on(op, {"by": "tag", "name": match})))
        cands += _kind_substitutions(op, diags)
    elif by == "normal":
        cands += _gen_normal_relax(op, sel, diags)
        cands += _kind_substitutions(op, diags)
    elif by == "extreme":
        cands += _gen_extreme_relax(op, sel)
    elif by == "side":
        for s in (1.0, -1.0):
            if s != sel.get("sign", 1.0):
                cands.append(Candidate(
                    f"side sign -> {s:+.0f}", "suggest", "low",
                    "flips which half is selected — pick the side you meant",
                    op=_with_on(op, _set(sel, sign=s))))
    elif by == "last_created":
        cands.append(Candidate(
            "last_created -> extreme z max", "suggest", "low",
            "no creating op yet (or stale after subdivide); pick an explicit selector",
            op=_with_on(op, {"by": "extreme", "axis": "z", "which": "max"})))
    return cands


def _gen_normal_relax(op, sel, diags):
    """Loosen a too-tight tol along the SAME axis/sign — auto only when the histogram
    proves that direction actually has faces."""
    cands = []
    if "dir" in sel:   # arbitrary direction: which face it 'should' hit is ambiguous
        for tol in (0.5, 0.7, 0.9):
            if tol > sel.get("tol", 0.5):
                cands.append(Candidate(
                    f"normal dir tol -> {tol}", "suggest", "low",
                    "widening tol on a free direction may grab the wrong / several faces",
                    op=_with_on(op, _set(sel, tol=tol))))
        return cands
    axis, sign = sel.get("axis", "z"), sel.get("sign", 1.0)
    has = _hist_count(diags, axis, sign) > 0   # NB: histogram uses a fixed 0.5 threshold
    for tol in (0.9, 0.7, 0.5, 0.3, 0.1):      # descending: take the tightest that still matches
        if tol < sel.get("tol", 0.5):
            cands.append(Candidate(
                f"normal tol -> {tol}", "auto" if has else "suggest", "high" if has else "low",
                "relax cone half-angle, same axis/sign" if has
                else f"no face points {('+' if sign >= 0 else '-')}{axis}; relaxing may grab another face",
                op=_with_on(op, _set(sel, tol=tol))))
    return cands


def _gen_extreme_relax(op, sel):
    cands = []
    for tol in (0.1, 0.25):
        cands.append(Candidate(
            f"extreme tol -> {tol}", "auto", "high", "widen the extreme band (same axis/which)",
            op=_with_on(op, _set(sel, tol=tol))))
    return cands


def _gen_composite(op, sel, diag):
    """and/or/not that resolved empty — report per-branch, never silently broaden."""
    mesh = diag.prefix
    notes = []
    if "and" in sel:
        if not sel["and"]:
            notes.append("empty 'and' clause selects nothing")
        else:
            counts = [_safe_count(mesh, b) for b in sel["and"]]
            notes.append(f"AND branch face-counts: {counts}")
            if all(c and c > 0 for c in counts):
                return [Candidate("and -> or", "suggest", "low",
                                  "every branch matches alone but the intersection is empty; "
                                  "you likely meant the union (or)",
                                  op=_with_on(op, {"or": sel["and"]}))] + _note_only(notes)
        return _note_only(notes + ["fix the empty branch or rebuild the selector"])
    if "or" in sel:
        return _note_only(["all OR branches matched 0 faces — fix each branch (e.g. tag typos)"])
    if "not" in sel:
        return _note_only(["NOT(inner) is empty -> inner matched every face; "
                           "narrow the inner selector or drop the NOT"])
    return []


def _gen_kernel_error(op, diag):
    inner = diag.inner
    detail, itype = inner.get("detail", ""), inner.get("type", "")
    cands = []
    opname = op.get("op")
    # `by` given a scalar where a 3-vector is required. A uniform SCALE is unambiguous
    # (auto); a scalar TRANSLATE invents a diagonal direction (suggest only).
    if itype == "TypeError" and "subscriptable" in detail and isinstance(op.get("by"), (int, float)):
        by = op["by"]
        if opname == "scale":
            cands.append(Candidate(
                f"scale by {by} -> [{by},{by},{by}]", "auto", "high",
                "a scalar scale unambiguously means uniform scale", op=_set(op, by=[by, by, by])))
        else:
            cands.append(Candidate(
                f"{opname} by {by} -> [{by},{by},{by}]", "suggest", "low",
                f"a scalar {opname} has no clear direction; this moves diagonally by {by} — confirm",
                op=_set(op, by=[by, by, by])))
    # numeric params sent as strings / wrong number type
    for key in ("distance", "thickness", "size", "radius", "height"):
        if key in op and isinstance(op[key], str):
            try:
                cands.append(Candidate(
                    f"{key} '{op[key]}' -> {float(op[key])}", "auto", "high",
                    "coerce numeric string (exact magnitude preserved)",
                    op=_set(op, **{key: float(op[key])})))
            except ValueError:
                pass
    if "levels" in op and not isinstance(op["levels"], int):
        try:
            lv = max(0, min(MAX_SUBDIVIDE, int(round(float(op["levels"])))))
            cands.append(Candidate(f"levels -> {lv}", "auto", "high",
                                   "subdivide levels must be a small int", op=_set(op, levels=lv)))
        except (ValueError, TypeError):
            pass
    # bad axis: a str-prefix axis ('-x') raises ValueError 'substring not found'; an int
    # axis (2) raises TypeError 'must be str' — both are recoverable by _canon_axis.
    if (itype == "ValueError" and "substring" in detail) or (itype == "TypeError" and "must be str" in detail):
        cands += _gen_axis_fix(op)
    # selector missing required subkey (KeyError mis-attributed to the consuming op)
    if itype == "KeyError" and "name" in detail:
        cands.append(Candidate(
            "tag selector missing 'name'", "suggest", "low",
            "a {by:tag} selector needs a 'name'; supply the tag you meant",
            op=op))
    # cylinder with < 3 sides — the floor (3) is objective, but the intended polygon count
    # can't be recovered from an out-of-range value, so this is a magnitude pick -> suggest.
    if itype == "ValueError" and ">= 3 verts" in detail and opname == "cylinder":
        for n in (3, 6, 24):
            cands.append(Candidate(
                f"sides {op.get('sides')} -> {n}", "suggest", "low",
                f"an n-gon prism needs >= 3 sides; the intended count can't be inferred from {op.get('sides')!r}",
                op=_set(op, sides=n)))
    return cands


def _gen_axis_fix(op):
    sel = op.get("on")
    if not isinstance(sel, dict) or "axis" not in sel:
        return []
    raw = sel["axis"]
    canon = _canon_axis(raw)
    if not canon:
        return [Candidate(f"axis '{raw}' invalid", "suggest", "low",
                          "axis must be one of x, y, z — pick one (ambiguous which)", op=op)]
    axis, sign = canon
    has_sign = "sign" in sel
    if sign is None or not has_sign or sel.get("sign") == sign:
        # pure casing/index normalization, or a spelled sign that AGREES with (or supplies)
        # the explicit one -> intent-preserving, safe to auto.
        new_sel = _set(sel, axis=axis)
        if sign is not None and not has_sign:
            new_sel["sign"] = sign
        return [Candidate(f"axis '{raw}' -> '{axis}'", "auto", "high",
                          "axis casing/index/sign normalization", op=_with_on(op, new_sel))]
    # the spelled sign on the axis (e.g. '-x') CONTRADICTS an explicit `sign` — which face
    # is meant is ambiguous, so never auto. Offer both readings for the agent to pick.
    return [
        Candidate(f"axis '{raw}' -> '{axis}', keep sign {sel.get('sign')}", "suggest", "low",
                  f"axis '{raw}' and explicit sign={sel.get('sign')} disagree; honoring your explicit sign",
                  op=_with_on(op, _set(sel, axis=axis))),
        Candidate(f"axis '{raw}' -> '{axis}', sign {sign}", "suggest", "low",
                  f"axis '{raw}' and explicit sign={sel.get('sign')} disagree; honoring the spelled sign",
                  op=_with_on(op, _set(sel, axis=axis, sign=sign))),
    ]


def _canon_axis(raw):
    """Return (axis, sign|None) for recoverable axis spellings, else None."""
    if isinstance(raw, int) and raw in (0, 1, 2):
        return "xyz"[raw], None
    if isinstance(raw, str):
        s = raw.strip().lower()
        sign = None
        if s.startswith("-"):
            sign, s = -1.0, s[1:]
        elif s.startswith("+"):
            sign, s = 1.0, s[1:]
        if s in ("x", "y", "z"):
            return s, sign
    return None


def _gen_invalid_mesh(op, diag):
    """A param numerically collapsed a face. We can only offer *validating* candidates;
    the magnitude is a guess, so these are always suggestions."""
    opname = op.get("op")
    cands = []
    if opname in ("scale",) and isinstance(op.get("by"), (list, tuple)):
        by = list(op["by"])
        for mult in (0.1, 0.5, 1.0):
            fixed = [mult if abs(c) < _DEGENERATE else c for c in by]
            if fixed != by:
                cands.append(Candidate(
                    f"scale nudge zeros -> {fixed}", "suggest", "low",
                    "a 0 factor collapsed a face; magnitude is a guess", op=_set(op, by=fixed)))
    if opname == "cylinder":
        if op.get("radius", 0.5) == 0:
            for r in (0.1, 0.25, 0.5):
                cands.append(Candidate(f"radius 0 -> {r}", "suggest", "low",
                                       "a zero radius collapses the cap", op=_set(op, radius=r)))
        if op.get("height", 1.0) == 0:
            for h in (0.1, 0.5, 1.0):
                cands.append(Candidate(f"height 0 -> {h}", "suggest", "low",
                                       "a zero height collapses the side walls", op=_set(op, height=h)))
    cands.append(Candidate("drop this op", "suggest", "low",
                           "remove the op that produced the invalid mesh", action="drop", op=None))
    return cands


def _gen_assert(op, diag):
    m = _ASSERT_EULER_RE.search(diag.message)
    cands = []
    if m:
        actual = int(m.group(2))
        cands.append(Candidate(
            f"assert euler -> {actual} (observed)", "suggest", "low",
            "an assert is YOUR invariant; correcting it to reality hides that an intended "
            "topology-changing op may be missing", op=_set(op, euler=actual)))
    cands.append(Candidate("drop the assert", "suggest", "low",
                           "dropping a guardrail can ship a broken mesh — confirm intent",
                           action="drop", op=None))
    return cands


def _gen_structural(op, diag):
    msg = diag.message
    cands = []
    if "before any primitive" in msg or msg == "empty program":
        cands.append(Candidate("insert {op:cube} first", "suggest", "low",
                               "no mesh exists yet; which base primitive (and size) is your choice",
                               op={"op": "cube", "size": 1.0}, action="insert_before"))
        cands.append(Candidate("insert {op:cylinder} first", "suggest", "low",
                               "alternative base primitive", op={"op": "cylinder"}, action="insert_before"))
    if msg.startswith("unknown op"):
        bad = (op or {}).get("op")
        if bad is None:   # dict with no 'op' key — infer from params
            inferred = {_PARAM_SIG[k] for k in (op or {}) if k in _PARAM_SIG}
            if len(inferred) == 1:
                cands.append(Candidate(f"infer op -> '{next(iter(inferred))}'", "suggest", "low",
                                       "no 'op' key; inferred from the param signature",
                                       op=_set(op, op=next(iter(inferred)))))
        else:
            match, unique = _best_choice(str(bad), KNOWN_OPS)
            sig_ok = unique and any(k in op for k in _PARAM_SIG if _PARAM_SIG[k] == match)
            if match:
                cands.append(Candidate(
                    f"op '{bad}' -> '{match}'", "auto" if (unique and sig_ok) else "suggest",
                    "high" if (unique and sig_ok) else "low",
                    "fuzzy op-name match" + (" corroborated by params" if sig_ok else ""),
                    op=_set(op, op=match)))
    return cands


def _gen_malformed_selector(op, diag):
    msg = diag.message
    sel = op.get("on")
    cands = []
    if "selector must be a dict" in msg and isinstance(sel, str):
        words = {"top": ("z", "max"), "bottom": ("z", "min"), "up": ("z", "max"), "down": ("z", "min"),
                 "right": ("x", "max"), "left": ("x", "min"), "front": ("y", "min"), "back": ("y", "max")}
        if sel.lower() in words:
            axis, which = words[sel.lower()]
            cands.append(Candidate(
                f"on '{sel}' -> extreme {axis} {which}", "suggest", "high",
                "a direction word maps to an extreme face (confirm vs a tag of the same name)",
                op=_with_on(op, {"by": "extreme", "axis": axis, "which": which})))
        cands.append(Candidate(f"on '{sel}' -> tag '{sel}'", "suggest", "low",
                               "treat the bare string as a tag name",
                               op=_with_on(op, {"by": "tag", "name": sel})))
    if msg.startswith("unknown selector") and isinstance(sel, dict):
        by = sel.get("by")
        if by is not None:
            match, unique = _best_choice(str(by), KNOWN_BY)
            if match:
                cands.append(Candidate(
                    f"selector by '{by}' -> '{match}'", "auto" if unique else "suggest",
                    "high" if unique else "low", "fuzzy selector-kind match",
                    op=_with_on(op, _set(sel, by=match))))
        else:   # no 'by' — infer from keys
            if {"axis", "which"} <= sel.keys():
                cands.append(Candidate("infer by -> extreme", "suggest", "low",
                                       "{axis,which} matches the extreme selector",
                                       op=_with_on(op, _set(sel, by="extreme"))))
            elif "name" in sel:
                cands.append(Candidate("infer by -> tag", "suggest", "low",
                                       "{name} matches the tag selector",
                                       op=_with_on(op, _set(sel, by="tag"))))
            elif {"axis", "sign"} <= sel.keys() or "dir" in sel:
                cands.append(Candidate("infer by -> normal", "suggest", "low",
                                       "{axis,sign}/{dir} matches normal (or side — ambiguous)",
                                       op=_with_on(op, _set(sel, by="normal"))))
    return cands


def _best_choice(word, vocab):
    """(match, unique): closest vocab word and whether it's an unambiguous winner."""
    ranked = sorted(vocab, key=lambda v: difflib.SequenceMatcher(None, word, v).ratio(), reverse=True)
    r0 = difflib.SequenceMatcher(None, word, ranked[0]).ratio()
    r1 = difflib.SequenceMatcher(None, word, ranked[1]).ratio() if len(ranked) > 1 else 0.0
    return (ranked[0], r0 >= 0.6 and r0 - r1 > 1e-6) if r0 >= 0.5 else (None, False)


def _note_only(notes):
    return [Candidate("; ".join(notes), "suggest", "low", "diagnostic only", op=None, action="none")]


def _safe_count(mesh, sel):
    if mesh is None:
        return None
    try:
        return len(resolve(mesh, sel))
    except Exception:   # noqa: BLE001
        return 0


def _candidates(ops, diag) -> list:
    op = diag.op or {}
    if diag.kind == "selector_empty":
        return _gen_selector_empty(op, diag)
    if diag.kind == "kernel_error":
        return _gen_kernel_error(op, diag)
    if diag.kind == "invalid_mesh":
        return _gen_invalid_mesh(op, diag)
    if diag.kind == "assert_failed":
        return _gen_assert(op, diag)
    if diag.kind == "structural":
        return _gen_structural(op, diag)
    if diag.kind == "malformed_selector":
        return _gen_malformed_selector(op, diag)
    return []


# --------------------------------------------------------------------------- #
# Public: diagnose + bounded auto-repair
# --------------------------------------------------------------------------- #
def diagnose(program) -> Diagnostic | None:
    """Classify the current build failure of ``program`` (a MeshProgram or op list).
    Returns ``None`` if the program already builds."""
    ops = _as_ops(program)
    pre = _precheck(ops)        # non-dict / empty / runaway-subdivide guards (never builds)
    if pre is not None:
        return pre
    idx, exc, prefix = _first_failing(ops)
    if idx is None:
        return None
    diag = _classify(ops, idx, exc, prefix)
    diag.candidates = _candidates(ops, diag)
    return diag


def _apply_candidate(ops, idx, cand) -> list:
    if cand.action == "insert_before":
        return ops[:idx] + [cand.op] + ops[idx:]
    if cand.action == "drop":
        return ops[:idx] + ops[idx + 1:]
    return ops[:idx] + [cand.op] + ops[idx + 1:]   # replace


def repair_program(program, max_attempts: int = 6) -> RepairResult:
    """Bounded-retry auto-repair. Builds candidates for the first failing op, probes
    each (full-program rebuild + validate), and APPLIES the first high-confidence,
    intent-preserving (`auto`) fix. Validating `suggest` candidates are returned for
    the agent to choose; nothing intent-changing is applied silently. Terminating:
    the candidate set is finite and each is tried at most once, hard-capped at
    ``max_attempts``."""
    ops = _as_ops(program)
    pre = _precheck(ops)        # non-dict / empty / runaway-subdivide guards (never builds)
    if pre is not None:
        return RepairResult(False, False, ops, diagnostic=pre.to_dict(),
                            suggestions=[c.to_dict() for c in pre.candidates])
    idx, exc, prefix = _first_failing(ops)
    if idx is None:   # idempotence: an already-valid program needs no repair
        return RepairResult(True, False, ops)
    diag = _classify(ops, idx, exc, prefix)
    diag.candidates = _candidates(ops, diag)

    tried, attempts, suggestions, applied = set(), [], [], None
    for cand in diag.candidates:
        if len(attempts) >= max_attempts:
            break
        key = (cand.action, json.dumps(cand.op, sort_keys=True, default=str) if cand.op else cand.label)
        if key in tried:
            continue
        tried.add(key)
        if cand.op is None and cand.action != "drop":   # pure note, nothing to probe
            suggestions.append(cand.to_dict())
            continue
        probe = _apply_candidate(ops, idx, cand)
        built, perr = _builds(probe)
        cand.validated = built
        attempts.append({"label": cand.label, "apply_mode": cand.apply_mode,
                         "built": built, "error": None if built else str(perr)})
        if built and cand.apply_mode == "auto" and applied is None:
            applied = (cand, probe)
            break
        if built:
            suggestions.append(cand.to_dict())
    # carry along un-probed suggestions (budget exhausted) so the agent still sees options
    probed_labels = {a["label"] for a in attempts} | {s["label"] for s in suggestions}
    for cand in diag.candidates:
        if cand.label not in probed_labels and cand.apply_mode == "suggest":
            suggestions.append(cand.to_dict())

    if applied is not None:
        cand, probe = applied
        return RepairResult(True, True, probe, applied=cand.to_dict(),
                            diagnostic=diag.to_dict(), attempts=attempts, suggestions=suggestions)
    return RepairResult(False, False, ops, diagnostic=diag.to_dict(),
                        attempts=attempts, suggestions=suggestions)


# --------------------------------------------------------------------------- #
# Public: lint — the silent class build() cannot catch
# --------------------------------------------------------------------------- #
def lint_program(program) -> list:
    """Static pass over the op-log for traps that raise NO exception (so a repair
    loop keyed on failures never sees them). Returns a list of warning dicts."""
    ops = _as_ops(program)
    warns = []

    def warn(i, code, message, suggestion=None):
        warns.append({"op_index": i, "code": code, "message": message, "suggestion": suggestion})

    def lint_selector(i, sel):   # recurse so traps at ANY nesting depth are caught
        if not isinstance(sel, dict):
            return
        if "and" in sel or "or" in sel:
            for s in list(sel.get("and", [])) + list(sel.get("or", [])):
                lint_selector(i, s)
            return
        if "not" in sel:
            lint_selector(i, sel["not"])
            return
        if sel.get("by") == "extreme" and sel.get("which", "max") not in ("max", "min"):
            warn(i, "extreme_which", f"which='{sel.get('which')}' is not 'max'/'min'; the kernel "
                 "silently treats anything != 'max' as MIN (selects the opposite face)",
                 "use which: 'max' or 'min'")
        if sel.get("by") in ("side", "normal") and sel.get("sign", 1.0) == 0:
            warn(i, "sign_zero", "sign=0 selects nothing meaningful", "use sign +1 or -1")
        if sel.get("by") == "normal" and sel.get("dir") == [0, 0, 0]:
            warn(i, "dir_zero", "dir=[0,0,0] has no direction", "give a real direction or use axis/sign")

    prev_op = None
    for i, op in enumerate(ops):
        if not isinstance(op, dict):
            warn(i, "malformed_op", f"op #{i} is not a dict", "each op must be a dict")
            continue
        name, sel = op.get("op"), op.get("on")
        if name == "extrude":
            d = op.get("distance", 0.5)
            if isinstance(d, (int, float)) and abs(d) < _DEGENERATE:
                warn(i, "extrude_noop", "distance ~0 is a silent no-op; its mark is stamped on the "
                     "un-extruded face, so a later selector picks the wrong geometry",
                     "use a non-zero distance")
            if isinstance(sel, dict) and sel.get("by") == "all":
                warn(i, "extrude_all", "extrude on {by:all} has no boundary edges -> no side walls (no-op)",
                     "select a face region, not the whole mesh")
        if name == "inset":
            t = op.get("thickness", 0.3)
            if isinstance(t, (int, float)) and not (1e-3 <= t <= 0.999):  # inclusive: kernel honors the endpoints
                warn(i, "inset_clamped", f"thickness {t} is silently clamped to [1e-3, 0.999] — the built "
                     "geometry will NOT match the requested value", "pick a thickness in (0, 1)")
        if name == "bevel":
            w = op.get("width", 0.2)
            if isinstance(w, (int, float)) and not (1e-3 <= w <= 0.999):
                warn(i, "bevel_width_clamped", f"width {w} is silently clamped to [1e-3, 0.999] — the built "
                     "geometry will NOT match the requested value", "pick a width in (0, 1)")
            if op.get("depth", 0.1) == 0:
                warn(i, "bevel_flat", "depth=0 makes bevel a plain inset (no chamfer)", "use a non-zero depth")
        if name == "subdivide":
            lv = op.get("levels", 1)
            if isinstance(lv, (int, float)) and lv <= 0:
                warn(i, "subdivide_noop", f"levels={lv} subdivides nothing", "use levels >= 1")
            elif isinstance(lv, (int, float)) and lv > MAX_SUBDIVIDE:
                warn(i, "subdivide_explosive", f"levels={lv} grows faces ~4^levels (very large/slow)",
                     f"keep levels <= {MAX_SUBDIVIDE}")
        lint_selector(i, sel)
        if isinstance(sel, dict) and sel.get("by") == "last_created" and prev_op in (None, "subdivide", *PRIMITIVE_OPS):
            warn(i, "last_created_broad", "last_created right after a primitive/subdivide resolves to the "
                 "WHOLE surface (every face inherits the step tag)", "use an explicit selector")
        prev_op = name
    return warns


# --------------------------------------------------------------------------- #
# Public: mesh-geometry cleanup (weld / drop degenerate / drop orphans / detect NM)
# --------------------------------------------------------------------------- #
def _dedupe_cycle(idxs):
    """Drop consecutive + wraparound duplicate indices (a welded face may collapse)."""
    out = []
    for v in idxs:
        if not out or out[-1] != v:
            out.append(v)
    while len(out) > 1 and out[0] == out[-1]:
        out.pop()
    return out


def _zero_area(coords) -> bool:
    nx = ny = nz = 0.0
    n = len(coords)
    for i in range(n):
        a, b = coords[i], coords[(i + 1) % n]
        nx += (a[1] - b[1]) * (a[2] + b[2])
        ny += (a[2] - b[2]) * (a[0] + b[0])
        nz += (a[0] - b[0]) * (a[1] + b[1])
    return (nx * nx + ny * ny + nz * nz) <= 1e-16


def _nonmanifold_edges(faces):
    counts = {}
    for f in faces:
        n = len(f)
        for i in range(n):
            a, b = f[i], f[(i + 1) % n]
            key = (a, b) if a < b else (b, a)
            counts[key] = counts.get(key, 0) + 1
    return [e for e, c in counts.items() if c > 2]


def _face_key(cycle):
    """Rotation/reflection-invariant key — two faces with the same key are the same
    (possibly oppositely-wound) face, so welding made them coincident duplicates."""
    n = len(cycle)
    rots = [tuple(cycle[i:] + cycle[:i]) for i in range(n)]
    rev = cycle[::-1]
    rots += [tuple(rev[i:] + rev[:i]) for i in range(n)]
    return min(rots)


def repair_mesh(mesh: Mesh, eps: float = 1e-6) -> tuple[Mesh, dict]:
    """Clean a built/imported mesh on its *extracted* representation (never mutating the
    live topology). Single deterministic pass — weld coincident verts, drop zero-area /
    self-touching faces, drop weld-coincident duplicate faces, detect (never delete)
    non-manifold edges, compact orphans last — then rebuild + validate. Any *intent-
    changing* weld effect (an over-large eps that collapses or merges real faces, or
    that creates a non-manifold edge) downgrades ``report['apply_mode']`` to 'suggest'.
    Returns ``(cleaned_mesh, report)`` for ANY float eps (never raises)."""
    import math
    pos = [list(v.co) for v in mesh.verts]
    faces = [[lp.vert.id for lp in mesh.face_loops(f)] for f in mesh.faces]
    attrs = [_copy_attrs(f.attrs) for f in mesh.faces]
    report = {"verts_before": len(pos), "faces_before": len(faces), "welded_verts": 0,
              "dropped_degenerate": 0, "weld_collapsed": 0, "duplicate_faces": 0,
              "dropped_orphans": 0, "nonmanifold_edges": 0, "notes": [], "apply_mode": "auto"}

    # 1. weld coincident verts (grid bucket; representative = lowest original index).
    #    A non-finite or non-positive eps means "no weld" (so NaN never reaches round()).
    weld = math.isfinite(eps) and eps > 0
    rep, bucket = list(range(len(pos))), {}
    if weld:
        for i, c in enumerate(pos):
            key = tuple(round(c[k] / eps) for k in range(3))
            if key in bucket:
                rep[i] = bucket[key]
            else:
                bucket[key] = i
    report["welded_verts"] = sum(1 for i in range(len(pos)) if rep[i] != i)

    nm_before = len(_nonmanifold_edges(faces))   # on the ORIGINAL faces (pre-weld)

    # 2. remap + collapse + drop degenerate (incl. faces a weld made self-touching).
    kept, kept_attrs, dropped_tags = [], [], set()
    for fi, f in enumerate(faces):
        collapsed = _dedupe_cycle([rep[v] for v in f])
        degenerate = (len(collapsed) < 3 or len(set(collapsed)) != len(collapsed)
                      or _zero_area([pos[v] for v in collapsed]))
        if degenerate:
            report["dropped_degenerate"] += 1
            orig = [pos[v] for v in f]                       # was the ORIGINAL face real?
            if len(set(f)) >= 3 and not _zero_area(orig):    # then welding destroyed it
                report["weld_collapsed"] += 1
            dropped_tags |= {t for t in attrs[fi].get("tags", []) if not t.startswith("__")}
            continue
        kept.append(collapsed)
        kept_attrs.append(attrs[fi])

    # 3. drop duplicate faces a weld made coincident (rotation/reflection invariant)
    seen, new_faces, new_attrs = set(), [], []
    for fc, at in zip(kept, kept_attrs):
        k = _face_key(fc)
        if k in seen:
            report["duplicate_faces"] += 1
            dropped_tags |= {t for t in at.get("tags", []) if not t.startswith("__")}
            continue
        seen.add(k)
        new_faces.append(fc)
        new_attrs.append(at)

    # 4. detect non-manifold (report only — which face is the intruder is ambiguous)
    nm = _nonmanifold_edges(new_faces)
    report["nonmanifold_edges"] = len(nm)
    if report["weld_collapsed"]:
        report["apply_mode"] = "suggest"
        report["notes"].append(f"welding (eps={eps:g}) collapsed {report['weld_collapsed']} "
                               "previously-valid face(s); review eps")
    if report["duplicate_faces"]:
        report["apply_mode"] = "suggest"
        report["notes"].append(f"welding merged {report['duplicate_faces']} distinct face(s) into "
                               "coincident duplicates; review eps")
    if len(nm) > nm_before:
        report["apply_mode"] = "suggest"
        report["notes"].append(f"welding CREATED {len(nm) - nm_before} non-manifold edge(s); review eps")
    elif nm:
        report["apply_mode"] = "suggest"
        report["notes"].append(f"{len(nm)} non-manifold edge(s) detected (not modified)")

    # 5. true orphans (own representative, referenced by no surviving face)
    referenced = {v for f in new_faces for v in f}
    report["dropped_orphans"] = sum(1 for i in range(len(pos)) if rep[i] == i and i not in referenced)

    # 6. compact (drops welded-away + orphan verts) and rebuild
    newpos, compfaces = _compact(pos, new_faces) if new_faces else ([], [])
    cleaned = Mesh.from_pydata(newpos, compfaces, new_attrs)
    cleaned.validate()

    report["verts_after"] = len(cleaned.verts)
    report["faces_after"] = len(cleaned.faces)
    if dropped_tags:
        report["notes"].append(f"tags lost with dropped faces: {sorted(dropped_tags)}")
    if report["faces_after"] == 0:
        report["notes"].append("every face was degenerate — result is the empty mesh")
    return cleaned, report
