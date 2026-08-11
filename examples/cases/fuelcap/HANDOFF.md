# Handoff: operating Mirage

Written for the next agent picking up work in this repo. `AGENTS.md` at the root tells you
what Mirage *is* and where things live; `skills/mirage/SKILL.md` tells you how to drive the
op-log. This file is the layer neither of those covers: **the things that will cost you an
hour each if nobody warns you**. Every item below is something that actually went wrong.

Ordered by how early it will bite you.

---

## 1. Renders run on the build box, not here

There is a `.render-remote-only` marker in the repo root (gitignored). `capture.default_render()`
refuses to run with it present — `src/mirage/capture.py:79`. That is deliberate and it is a
guard, not an inconvenience:

```
tools/render_on_box.py --cwd examples/cases -- -m yourcase.sheet whatever
```

is the whole round trip — dirty-tree check, push, remote pull + cmake build, run, scp the
outputs back.

* **Run it from PowerShell, not Git Bash.** MSYS rewrites anything that looks like a Unix
  path in an argument, so a remote target arrives at `scp` as `C:/Program Files/Git/home/...`.
  The tool sets `MSYS_NO_PATHCONV` for its own subprocesses but cannot fix its own argv.
* It needs `MIRAGE_BOX`, `MIRAGE_BOX_DIR`, `MIRAGE_BOX_ENV`, `MIRAGE_BOX_PULL` in the
  environment. Values are not in the repo and should not be put there.
* It **refuses if tracked files are uncommitted**, on purpose: otherwise the box renders
  something other than what is in front of you, and you report a picture that no commit
  produces. Commit first. `--allow-dirty` exists and you should not want it.
* `--no-build` skips the remote cmake when you are only changing Python. Use it; the build
  is most of the wall clock.
* **If a box step fails, stop.** A failed pull leaves the *previous* image on disk and the
  next step will happily score it and report it as a new result. Check the pull actually
  said `[box] pulled <your file>` for the file you care about, not just for some file.

`MIRAGE_ALLOW_LOCAL=1` overrides the guard. It is for a 64×64 correctness render or the test
suite (`tests/conftest.py` sets it), never for a loop and never for anything you will show
anyone. The test is "how many frames", not "is this a deliverable".

Write new cases with `MIRAGE_THREADS` rather than a hardcoded core count, and put any
out-of-repo input path behind an env var — otherwise the script dies on the box for a reason
that has nothing to do with rendering.

## 2. `place` composes Rz @ Ry @ Rx, and this is the repo's most-repeated bug

`_place_xform` in `src/mirage/meshlang.py:63` — scale, then **Rz @ Ry @ Rx**, then translate.

So a z-rotation handed to `place` is applied **last, outside any tilt**. If you are placing a
part onto a tilted surface and you also want the part turned about its own axis, passing that
turn to `place` swings the part's axis around the *world* z instead of turning it about
*itself*. The pose label and the geometry then disagree by up to twice the tilt, and every
frame in the batch is quietly mislabelled while every render looks fine.

**Bake the part's own rotation into the part.** A lathe takes a `plan` — turn the plan. A
prism takes a polygon — turn the polygon. This case has `cap(spin=)` and `cap_boss(spin=)`
for exactly this reason, and both existed only after the bug was found twice.

## 3. Colours are LINEAR albedo. Black plastic is 0.02, not 0.12

The renderer's albedo space is linear. A "black" moulding is sRGB ~0.12, which is linear
~0.014. Typing the sRGB number makes a black part render mid-grey, and that single mistake
is most of the distance between a synthetic object and a photographed one. Same in
`src/mirage/decals.py` — its palette constants are linear, which is why a decal PNG looks
washed out if you open it directly. That is correct.

## 4. An albedo map REPLACES the flat colour where it lands

`core/src/raytrace.cpp:471` — `if (sample_map(...)) alb = tmp;`. It does not modulate.

Two consequences, both of which have cost time here:

* the artwork's **background must be the part's own colour**, not black — paint it black and
  every unlettered part of the surface, including the inside of a recessed grip, goes to zero
  albedo, and a well whose walls return no light is indistinguishable from no well;
* **any other part sharing that surface must take the same (decalled) material.** A printed
  cap renders at the artwork's background while a separately-placed handle carrying the
  material's own jittered colour renders up to four times brighter, which looks like a
  lighting artefact and is really two different albedos.

`with_decal` (in this case's `materials.py`) pins artwork to a +z-facing rectangle in the
part's own frame; the angles you author in the artwork map straight through. Check text
orientation **at magnification**, not on a contact sheet — a cap 90 px across will let you
talk yourself into a flip that is not there, and then "fixing" it puts a real one in.

## 5. Read the camera basis, do not guess it

`core/src/raytrace.cpp:785-788`:

```
fwd   = normalize(target - eye)
right = normalize(cross(fwd, up))
up2   = cross(right, fwd)          // image y increases DOWNWARD from here
```

If you need to know where a model direction lands in frame — to match a reference photograph,
to solve a roll, to place a label — project it yourself with those three lines. It is exact
and it takes no renders. Two rounds went into "calibrating" a camera roll from measurements
taken off rendered images, and the measurements were biased; the arithmetic was right the
first time.

Also: **`up` parallel to the view direction is a degenerate basis.** A near head-on look down
+z with `up=+z` gives an arbitrary roll that changes with elevation, and the part appears
rotated by a random angle. Use `up=(0,1,0)` for axis-on views.

## 6. The two mesh engines must stay byte-identical, and a stale local build lies to you

Any op or parameter you add to `src/mirage/kernel.py` + `meshlang.py` must also go into
`core/src/mesh.cpp` + `program.cpp`, with a fixture in `tests/test_cpp_program.py`.

The trap: if you change the C++ and do not rebuild locally, `tests/test_cpp_program.py` fails
in a way that reads exactly like a regression you just introduced. Check the timestamps —
`core/build/Release/_mirage_core*.pyd` against `core/src/mesh.cpp` — before you go hunting.
The box rebuilds from source every trip, so remote runs are unaffected.

## 7. Op-specific edges that are not obvious

* **`lathe` sections must start and end on the axis** (radius 0). A section that misses gives
  a tube with two open rims — silently, and the mesh only fails much later. The helper in this
  case checks it; if you write your own, check it.
* **`spin(plan=, plan_from=)`**: `plan_from` is a **radius**. Inside it the plan is ignored,
  outside it the multiplier fades to full strength at the profile's greatest radius. There is
  no way to fade by *z*, so you cannot flute a cylinder's wall and leave its rim round with a
  plan alone — either accept the scalloped rim or build the fluted skirt as a second solid.
* **`sweep` degenerates** where two path points coincide (zero-area quad) or where the path
  turns more than about 100° in one step. If you are generating a path from a spline or from
  jittered control points, cull both cases before sweeping. Fading a helix's radius to zero at
  its ends is a cleaner way to get straight leads than butting a straight segment onto a coil.
* **The tracer fans an n-gon from its first vertex**, so a concave plan gets triangles laid
  across its notches. Ear-clip concave polygons yourself.
* **A plan does not have to average 1**, and if it does not it scales the part as well as
  shaping it. Decide deliberately whether yours peaks at 1 or averages 1 — changing the lobe
  count of a normalised plan silently changes the part's diameter.

## 8. Verify at the scale of what you CHANGED, not at the scale of the picture

The single most expensive habit in this work, stated as a rule because stating it as an
intention did not work:

> **After changing a part, render THAT PART alone, at a magnification where its smallest
> feature is at least ~20 px, and look — before putting it back in the assembly.**

A worked example of the cost. Fifty water beads were added to a cap. They rendered as solid
white blobs — fifty specks of paint — because the water material was 0.34 albedo when a bead
over near-black plastic is dark with one pinpoint highlight. That is a one-number error,
visible instantly in a view where a 1 mm bead is 40 px. It survived several rounds because
every check was made on a 900 px compare sheet where a 1 mm bead is THREE PIXELS, judged by
a regional mean. No amount of care fixes that: the verification did not contain the
information.

The same round also produced two competing bead systems — one in the texture map, one as
placed geometry — because neither was ever looked at alone.

`sheet.py parts` already exists for this and its docstring already says why. It went unused
for an entire day of part work. A tool nobody reaches for is a process gap, not a tooling
gap; the rule above is the fix.

Corollaries:

* a change you cannot see in your verification view is a change you cannot evaluate — either
  change the view or do not claim the change is an improvement;
* when a metric moves and another does not, report the one that did NOT move. Regional mean
  luma converged nicely here while the local-difference figure sat at 43-47 and never
  budged, and it is the second one that tracks "does this look like the photograph".

## 9. Debug parts alone and with the id AOV, never in the beauty render

A composed render is the worst place to find a modelling error. If everything in your scene
is dark plastic in a dark cavity, a part that is the wrong size, inside-out, or **entirely
missing** looks very much like a part that is right — the frame is dark either way.

* render each part alone against a neutral background at a **known scale** (`sheet parts`
  here) — two parts are comparable only if you make them so;
* render the id AOV (`--ids` + `--id-tags`) — that is what finds a cap that is really the
  neck ring, a rib that never got placed, or a well you are seeing the outside of;
* `--depth` writes a PFM, `--no-ground` removes the default floor.

## 10. Measure the picture; do not argue with it

The habit that produced every real finding in this case:

* **put a number on it before changing anything.** "The render looks too bright" was wrong —
  measured, the tile means matched to within 1 part in 150 and the fault was specular
  roughness, not exposure. Two rounds would have gone into the wrong knob.
* **before blaming the geometry, check that the reference statistic measures what you think.**
  A discrepancy in this case turned out to be in the *annotation* (a hand-drawn polygon that
  enclosed the part plus its shadow), and another in a radially-binned median of a shape that
  is not round. Both had survived rounds of confident geometry work.
* **an automatic measurement on a dark scene will merge things.** A threshold that finds an
  aperture will also find the black strap and the black door touching it, and the resulting
  principal axis was biased by a constant 37° — enough to send a solved parameter off by 35°.
  Prefer a feature nobody disputes (here: an annotated circle) over one you have to segment.
* **frame both sides on the same feature.** Match on the wrong one and an object that is the
  wrong size relative to its surroundings looks right in every frame.

## 11. Repo hygiene in this tree

* **Other agents commit here concurrently. Stage by explicit path — never `git add -A`.**
  It once swept hundreds of megabytes of a customer's RGB-D capture set into a commit that
  reached a public remote. Read `git diff --cached --stat` before every commit and confirm
  the file count and that no `.npz/.png/.jpg` data is in it.
* Commits rotate a self-owned alias from `git_identities.json`; do not reuse the previous
  commit's identity.
* `_ref/` (reference photographs) and `_out/` (renders) are **gitignored**, so reference data
  is *not on the box*. Anything that needs it — composing a comparison, cropping a photo —
  has to happen locally, and only the render goes remote.
* `assets/decals/` is gitignored but generated artwork must match on both machines;
  `render_on_box.py` scp's that directory each trip. Regenerate locally *before* pushing if
  you changed a recipe. Textures are pure numpy and regenerate identically, so they need no
  care.
* Renders are 99.998% identical across gcc and MSVC, not byte-identical — last-ULP float
  differences, worst delta 1/255. Fine for media; do not assert bit-exactness across the two.

## 12. Windows-local snags

* The conda `python` on PATH has no pytest. Use `.venv/Scripts/python.exe -m pytest`.
* Running a case module from `examples/cases` needs `PYTHONPATH` pointing at `src`.
* **OpenCV cannot open CJK paths on Windows.** `cv2.imread` returns `None` with no error.
  Use `cv2.imdecode(np.fromfile(p, np.uint8), cv2.IMREAD_COLOR)` and
  `cv2.imencode(ext, img)[1].tofile(out)`.
* NumPy 2 removed `ndarray.ptp()`. Use `np.ptp(a)`.
* Heredocs: `python - << 'EOF'` through the Bash tool works, but a long one occasionally
  fails to parse. Writing the script to a file and running it is more reliable and leaves
  something you can re-run.

---

## Where the fuelcap case's own pieces are

Only so you can find them, not because you need them: `parts.py` is the part kit (each part
alone, documented with what was measured and what went wrong), `scene.py` samples a
randomised variant, `dataset.py` renders and degrades frames, `fit.py` measures the real
capture, `sheet.py` is every review render, `hero.py` is a 1:1 reproduction of one
photographed car and `_out/README.md` says which picture answers which question.

If you are here to make renders look more like photographs: **`hero.py`'s comparison is the
one that finds things.** A sheet of N random variants beside N real ones invites the eye to
match them up loosely and it will flatter you; one variant that is supposed to be *the same
object* as the photograph beside it cannot be talked out of anything.
