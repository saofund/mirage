// mirage path tracer — the offline ("Cycles-class") render pillar.
//
// A from-scratch, physically-based Monte-Carlo path tracer over the kernel mesh:
// cosine-importance-sampled diffuse global illumination lit by a sky + sun
// environment, with a ground plane for contact shadow and colour bleeding,
// Russian-roulette path termination, multi-threaded over scanlines, ACES
// tonemapped. The realtime viewport (mirage_viewer) is the Eevee-class preview;
// this is the ground-truth render of the SAME kernel mesh.
#pragma once

#include <array>
#include <vector>

#include "mirage/mesh.hpp"

namespace mirage {

struct Camera {
    std::array<double, 3> eye{3.2, -3.2, 2.2};
    std::array<double, 3> target{0, 0, 0.3};
    std::array<double, 3> up{0, 0, 1};
    double fov_y = 0.7;  // vertical field of view, radians
};

struct RenderSettings {
    int width = 640;
    int height = 480;
    int spp = 64;             // samples per pixel
    int max_bounce = 6;
    unsigned threads = 0;     // 0 => hardware_concurrency
    std::array<double, 3> albedo{0.82, 0.80, 0.74};  // surface base colour
    double metallic = 0.0;    // 0 = dielectric, 1 = metal (albedo tints the specular)
    double roughness = 0.5;   // microfacet roughness (GGX)
    bool ground = true;       // an implicit diffuse floor under the model

    // Environment & post. The sky is a gradient image-based light: it fills shadows
    // and bounces colour (the sun is added separately by NEE). env_intensity scales
    // that ambient fill; sun_intensity scales the directional key.
    double env_intensity = 1.0;
    double sun_intensity = 1.0;
    std::array<double, 3> sun_dir{0.4, 0.5, 0.8};  // sun direction (art-directable; low = long raking shadows)
    // Sky TINT (multiplies the horizon->zenith gradient). The built-in sky is a cool
    // blue, which is right for a clear day and wrong for the flat white overcast a lot
    // of reference photography is shot under -- and a cool sky tints EVERY surface at
    // once, so it reads as "my renders look CG" rather than as a colour error. 1,1,1
    // keeps the original; warming it toward {1.15, 1.05, 0.92} gives an overcast white.
    std::array<double, 3> sky_tint{1.0, 1.0, 1.0};
    // Sky FLATNESS. The built-in gradient is a clear day: blue, and DARKER toward the
    // zenith. Overcast is the opposite -- a near-uniform bright dome -- and the
    // difference shows up first on metal, which mirrors the zenith and comes back blue
    // under a sky no tint can fix, because a tint scales the gradient without flattening
    // it. 0 = the gradient as authored, 1 = a uniform sky of its average. ~0.75 = overcast.
    double sky_flat = 0.0;
    // How dark the LOWER hemisphere of the environment is (0 = a sky all round, 1 = a
    // ground below). Separate from `ground`, which is the floor GEOMETRY: dropping a
    // floor plane out of a close-up is normal, making the world glow from underneath
    // is not, and the two used to be the same switch.
    double env_ground = 0.0;
    // An equirectangular environment map (PPM). A clearcoat or a polished metal is
    // close to a mirror: what a viewer reads as its "paint" is mostly the world
    // reflected in it, and with an analytic two-colour sky there is no world to
    // reflect. Empty = the analytic sky.
    std::string env_map;
    double env_rot = 0.0;   // turn it about the vertical axis, radians
    // AERIAL PERSPECTIVE. Air is not empty: over tens of metres it scatters sky light into
    // the view and washes distance out, which is why a photograph's far objects are lower
    // in contrast and LIFTED in the blacks while its far whites barely move. Without it a
    // render's distance is as crisp and as black as its foreground, and that particular
    // wrongness is most of "this looks like CG" on any outdoor shot deeper than a room.
    // `haze_dist` is the distance at which roughly 63% of a surface is replaced by air
    // (0 = off); the air's colour is the sky the camera is already using.
    double haze_dist = 0.0;
    double exposure = 1.0;    // linear stops applied before the ACES tonemap
    // Firefly clamp: cap the luminance of INDIRECT (bounce>=1) contributions so a
    // rare high-variance specular bounce can't leave a white speckle. 0 = off. The
    // first hit's direct light is never clamped, so highlights stay crisp.
    double clamp_indirect = 12.0;
    // Denoise: N iterations of an edge-avoiding a-trous wavelet filter (guided by the
    // primary hit's albedo / normal / depth) applied to the HDR image before tonemap.
    // 0 = off. Lets a low-spp render (or a path-traced animation) come out clean.
    int denoise = 0;

    // Object-id AOV. Each entry is a face TAG — what `place(mark=...)` and every
    // primitive already write and what survives a rebuild — and a pixel gets the
    // 1-based index of the FIRST of these its centre ray lands on. Empty = no AOV.
    //
    // The caller supplies the order, which is the whole point: ids assigned by walking
    // the tags encountered while building would number objects by heap address, and
    // that exact bug already shipped here once — loop_cut and edge_bevel numbered
    // vertices by pointer and three identical runs disagreed 13/15/18 times while 328
    // tests stayed green. An explicit order cannot do that.
    std::vector<std::string> id_tags;

    // Depth AOV. Rides the same centre ray as the id AOV and the denoiser's G-buffer.
    // What it records is the hit's distance ALONG THE VIEW AXIS, not along the ray —
    // that is what a depth camera reports and what unprojecting with an intrinsic K
    // expects, and the two differ by 1/cos(angle from the axis), which at this
    // renderer's default 40 degree field is 8% at the corners. A dataset built from
    // ray distance would come out with a subtly domed world, and every plane fitted
    // to it would come out curved.
    bool want_depth = false;
    bool want_normal = false;
    // Per-pixel FACE id, which the object-id AOV cannot give: `ids` answers "which tagged
    // object is here", and a tag covers thousands of faces. Face ids answer "which faces
    // does this camera actually SEE", and therefore the questions that matter after an
    // edit -- did the faces I just made render a single pixel, and if not, what is in
    // front of them. A large geometric change that barely moves any pixels has almost
    // always been occluded, and without this the only way to find that out is to notice.
    bool want_face_ids = false;

    // CROP: render only this sub-rectangle of the frame, at the same camera. Not a zoom --
    // the rays are the rays the full frame would have cast, so a crop is a true sub-image
    // and can be compared against the same region of a full render pixel for pixel.
    //
    // This exists for iteration cost. Reverse-modelling a part means rendering the same
    // frame a hundred times to look at one corner of it, and paying for the sky each time
    // is most of the wall clock. crop_w = 0 means the whole frame.
    int crop_x = 0, crop_y = 0, crop_w = 0, crop_h = 0;

    // Depth of field: a thin-lens camera. aperture = lens radius in world units (0 = a
    // pinhole, everything sharp); focus_dist = distance to the sharp plane (0 = auto, the
    // distance from the eye to the camera target). Larger aperture -> shallower focus, more
    // background blur — converges with spp like any other camera jitter.
    double aperture = 0.0;
    double focus_dist = 0.0;

    // Bloom: bright regions (luminance above bloom_threshold) bleed a soft glow, added in
    // linear HDR before tonemapping — the photographic look of highlights and light sources.
    // 0 = off; typical strength 0.05–0.3.
    double bloom = 0.0;
    double bloom_threshold = 1.0;

    // Radial lens distortion — a real lens, not a pinhole. Applied to the normalised image
    // coordinates (a, b) before the primary ray is formed, where b spans [-1,1] top to
    // bottom and a spans [-aspect, aspect], so r = 1 on the top and bottom edges:
    //
    //     s = 1 + k1*r^2 + k2*r^4        a *= s;  b *= s
    //
    // Positive k1 = barrel (straight lines bow outward), negative = pincushion. 0 = pinhole,
    // bit-for-bit the old path.
    //
    // This exists because matching a real photograph is otherwise impossible, not as a
    // stylistic knob: a security camera, a phone, a dashcam are all noticeably distorted, and
    // no camera POSE can absorb that — get the pose perfect and the frame edges still refuse
    // to line up. It is a missing term in the model. `mirage.solve.solve_camera` fits k1
    // alongside the pose and reports the residual in pixels.
    //
    // The renderer runs the cheap direction (pixel -> ray, no inversion); mirage.solve owns
    // the inverse, where iterating costs nothing.
    double lens_k1 = 0.0;
    double lens_k2 = 0.0;

    // Smooth shading, by angle. Each face CORNER gets a shading normal averaged from the
    // faces around that vertex whose normal is within `smooth_angle` degrees of its own,
    // and the tracer interpolates that across the triangle. So a subdivided surface shades
    // as the smooth surface it approximates, while a crisp feature (a 90-degree rim, a
    // chamfer) keeps its hard edge — with nothing to author. This is Blender's "smooth by
    // angle"; below the threshold the geometry is treated as a sampling of a curved
    // surface, above it as a real crease.
    //
    // Without this, subdivision only buys silhouette smoothness — the shading stays
    // faceted, which is most of what reads as "CAD" instead of "organic".
    // 0 = off (flat/geometric normals, the pre-smooth-shading behaviour).
    double smooth_angle = 30.0;
};

struct Image {
    int w = 0, h = 0;
    std::vector<unsigned char> rgb;  // tonemapped, gamma-encoded, row-major, 3 bytes/px
    // Object id per pixel from the CENTRE ray — 0 where nothing tagged was hit, else
    // 1-based into RenderSettings::id_tags. Empty unless id_tags was set. This is what
    // lets a loss ask "is THIS object right" instead of "is the picture right": one
    // number per placed object rather than one for the frame.
    std::vector<int> ids;
    // Metric depth per pixel from the CENTRE ray: distance along the view axis in world
    // units, 0 where nothing was hit. Empty unless RenderSettings::want_depth was set.
    std::vector<float> depth;
    // Mesh face id per pixel from the CENTRE ray, -1 where nothing was hit. Empty unless
    // RenderSettings::want_face_ids was set. Ids are the mesh's own, the same ones
    // selectors and the marked render use, so a pixel maps back to an op-log face.
    std::vector<int> face_ids;
    // World-space shading normal per pixel from the CENTRE ray, xyz interleaved, all zero
    // where nothing was hit. Empty unless RenderSettings::want_normal was set. The G-buffer
    // already computes this to steer the denoiser; writing it out costs nothing and is what
    // lets a reverse-modelling loop ask "is this surface FACING the way the reference's is"
    // rather than only "is it in the same place".
    std::vector<float> normal;
};

// Path-trace the mesh. Deterministic for a given (mesh, camera, settings): each
// sample is seeded from its pixel + sample index, so renders are reproducible.
Image path_trace(const Mesh& mesh, const Camera& cam, const RenderSettings& settings);

// Write an Image to a binary PPM (P6).
void write_ppm(const Image& img, const std::string& path);

}  // namespace mirage
