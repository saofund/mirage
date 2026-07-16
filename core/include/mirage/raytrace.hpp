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
};

// Path-trace the mesh. Deterministic for a given (mesh, camera, settings): each
// sample is seeded from its pixel + sample index, so renders are reproducible.
Image path_trace(const Mesh& mesh, const Camera& cam, const RenderSettings& settings);

// Write an Image to a binary PPM (P6).
void write_ppm(const Image& img, const std::string& path);

}  // namespace mirage
