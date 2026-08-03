# 内盖 (inner fuel cap) — synthetic 6D-pose data

Generates labelled frames of a car's fuel-filler pocket, in the file format
`fuelcap_6dpose/pipeline/build_clouds.py` already produces, so they drop into that
pipeline's `datasets/` directory without a converter.

```bash
uv run python examples/cases/27_fuelcap.py -n 2000 --out datasets/synth_fuelcap
python -m fuelcap.fit --audit datasets/synth_fuelcap        # real vs synthetic, 11 rows
python -m fuelcap.fit --check-labels datasets/synth_fuelcap # are the labels true?
python -m fuelcap.sheet parts | scenes | ids | roi          # look at it
```

~2.5 s/frame single-machine at `--spp 32`, and it is embarrassingly parallel — different
`--seed` values never collide, so N shards is N times faster.

## What comes out

One `.npz` per frame:

| key | | |
|---|---|---|
| `xyz` | `(N,3) float32` | camera frame, **OpenCV** (x right, y **down**, z forward), metres |
| `rgb` | `(N,3) uint8` | |
| `label` | `(N,) uint8` | `1` = the cap (body + grip rib + teeth), `0` = everything else |
| `K_norm` | `(3,3)` | row 0 ÷ w, row 1 ÷ h — the same convention as the real npz |
| `w`, `h` | | the full frame the ROI was cut from |
| `anchor` | `(3,)` | cap face centre, camera frame |
| **`normal`** | `(3,)` | **the label** — cap face normal, camera frame, pointing at the camera |
| `cap_x` | `(3,)` | the cap's in-plane axis (along the grip rib). With `normal` this is the full 6D pose, not just the normal |
| `obb_px` | `(4,2) int32` | the knob OBB in full-frame pixels, straight off the rib's object id |
| `roi` | `(4,)` | `x0,y0,x1,y1` of the crop in the full frame |

Plus `_labels.npz` (`files`, `normal`, `anchor`), `_meta.json` (every sampled parameter per
frame, so any frame can be rebuilt exactly), and `_src_map.json` for whatever RGB previews
were kept.

`cap_x` is the one field the real set does not have. The real annotation is a normal only,
because a human labelling a point cloud can see which way the cap faces but not reliably
where its thread stopped. A renderer knows, so it is emitted — free, and it is what a
gripper actually needs.

## Registry entry

```json
{
  "slug": "synth_fuelcap",
  "name": "内盖·合成(Mirage 路径追踪 + 标定过的深度传感器模型)",
  "cloud_dir": "datasets/synth_fuelcap",
  "labels_npz": "datasets/synth_fuelcap/_labels.npz",
  "knob_centers": null,
  "predictions": {},
  "audit": null, "tiers": false, "full_cloud_worker": false,
  "labels_store": "datasets/synth_fuelcap/manual_normals.json",
  "assignee": null
}
```

`knob_centers: null` per the datasets README — the annotator will fall back to the npz
`anchor`, which is exact here. Nothing needs hand-labelling, so the label store starts
empty and stays that way; the point of the group in the annotator is spot-checking.

## What it was calibrated against, and how

Everything below is measured by `fit.py` against clouds pulled from
`192.168.111.3:/data/.../fuelcap_6dpose` (read-only; nothing on that host was touched).
Reference clouds live in `_ref/` and are gitignored — re-pull them to re-run the audit.

| | real | synth | |
|---|---|---|---|
| cap disc | 73.6 × 69.4 mm | 74.4 × 71.9 | ±4% |
| grip rib | 60.3 × 43.8 × 17.7 mm | 57.9 × 42.2 × 17.8 | ±4% |
| camera distance | 0.41 m | 0.44 | |
| obliquity | 16.8° | 15.8° | |
| cap width | 64.9 px | 61.8 | |
| depth noise | 0.44 mm | 0.48 | |
| ROI fill | 0.595 | 0.60 | |

Three sets were measured, and they disagree usefully:

| set | distance | obliquity | cap px | depth noise |
|---|---|---|---|---|
| `prod` (robot arm, the deployment domain) | 0.43–0.46 m | 3–13° | 57–74 | 0.32 mm |
| `orbbec` (handheld) | 0.29–0.62 m | 4–39° | 44–124 | 0.44 mm |
| `prod_depthcap` | 1.12–1.29 m | 29–53° | 41–51 | 0.42 mm |

`--domain prod` samples only the first. The default `wide` spans the union and then some,
on the argument that the model has to survive the pose it is handed rather than the one it
was promised.

### Three things that were wrong, and what caught them

Worth reading before extending this — all three were invisible in the renders.

1. **The label was wrong by 8°.** The cap's screw-stop angle was passed to `place` as a
   z-rotation, and `place` composes `Rz @ Ry @ Rx` — so the spin was applied *outside* the
   filler-neck tilt and swung the cap's axis instead of turning the cap about it. Every
   frame was mislabelled and every render looked perfect. Caught by `--check-labels`, which
   throws the label away and fits a plane to the cap's own annulus. Now 1.1° median, and
   0.29° on a flat annulus with the sensor model off, which is the measurement's floor
   rather than the label's error.

2. **The rib was a third too narrow.** The averaged-height-field comparison said it was too
   *wide*; the per-frame comparison said much too narrow. The averaged one is not a fair
   test — real frames only stack after a plane fit that is itself uncertain, so the real
   average is blurred by its own alignment error while the synthetic average is not.
   Per-frame is primary for that reason.

3. **The dropout model was backwards, twice.** "Dark surfaces return no depth" predicts the
   pocket empties out; the measurement says the pocket is the *densest* part of a real
   cloud. Brightness is a better proxy and still wrong in the same direction. The driver is
   local texture: a glossy textureless body panel is the hardest surface in the frame for a
   block matcher (survival 0.53), and matt moulded plastic covered in text and scuffs is
   one of the easiest (0.68). Fixing this took `pts_per_frame` from 0.73 of real to 0.97.

## Engine changes this case needed

* `mirage_render --depth FILE` — metric depth AOV, 32-bit float PFM, distance along the
  **view axis** (not along the ray), 0 where nothing was hit. Tests in
  `tests/test_depth_aov.py`.
* `mirage_render --no-ground` — drop the implicit floor plane.

With the existing `--ids`/`--id-tags` AOV that is everything needed to turn any op-log into
a labelled point cloud, so this is not fuel-cap-specific machinery.

## Known gaps

* **No printed text on the cap face.** Most real caps carry white warning text around the
  annulus, and it is a strong RGB cue. The decal mechanism (`mirage.decals`,
  `materials.with_decal`) is wired up but no artwork is generated yet.
* **The body panel is flat and untextured.** Real ones are curved, have reflections, panel
  gaps and dirt. This matters for an RGB model and not much for a point-cloud one.
* **One pocket archetype.** The capless filler (Toyota RAV4 style) and the square-aperture
  pocket both appear in the by-car reference and are not modelled.
* **`cap_frac` sits at 0.75 of real** — the cap is a slightly smaller share of the cloud
  than in the real set. The remaining gap is cap survival, not cap size (`cap_px` is 0.95),
  and chasing it further risks tuning the sensor model to one reference vehicle.
