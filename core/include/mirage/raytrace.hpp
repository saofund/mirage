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
    double exposure = 1.0;    // linear stops applied before the ACES tonemap
    // Firefly clamp: cap the luminance of INDIRECT (bounce>=1) contributions so a
    // rare high-variance specular bounce can't leave a white speckle. 0 = off. The
    // first hit's direct light is never clamped, so highlights stay crisp.
    double clamp_indirect = 12.0;
    // Denoise: N iterations of an edge-avoiding a-trous wavelet filter (guided by the
    // primary hit's albedo / normal / depth) applied to the HDR image before tonemap.
    // 0 = off. Lets a low-spp render (or a path-traced animation) come out clean.
    int denoise = 0;
};

struct Image {
    int w = 0, h = 0;
    std::vector<unsigned char> rgb;  // tonemapped, gamma-encoded, row-major, 3 bytes/px
};

// Path-trace the mesh. Deterministic for a given (mesh, camera, settings): each
// sample is seeded from its pixel + sample index, so renders are reproducible.
Image path_trace(const Mesh& mesh, const Camera& cam, const RenderSettings& settings);

// Write an Image to a binary PPM (P6).
void write_ppm(const Image& img, const std::string& path);

}  // namespace mirage
