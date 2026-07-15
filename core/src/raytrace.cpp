#include "mirage/raytrace.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <string>
#include <thread>
#include <unordered_map>

namespace mirage {

namespace {

constexpr double PI = 3.14159265358979323846;
using V3 = std::array<double, 3>;

V3 operator+(const V3& a, const V3& b) { return {a[0] + b[0], a[1] + b[1], a[2] + b[2]}; }
V3 operator-(const V3& a, const V3& b) { return {a[0] - b[0], a[1] - b[1], a[2] - b[2]}; }
V3 operator*(const V3& a, double s) { return {a[0] * s, a[1] * s, a[2] * s}; }
V3 mulv(const V3& a, const V3& b) { return {a[0] * b[0], a[1] * b[1], a[2] * b[2]}; }
double dot(const V3& a, const V3& b) { return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]; }
V3 cross(const V3& a, const V3& b) {
    return {a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]};
}
double len(const V3& a) { return std::sqrt(dot(a, a)); }
V3 norm(const V3& a) { double l = len(a); return l > 0 ? a * (1.0 / l) : a; }

// Triangle soup with the face's PBR material baked in (the `material` op assigns it;
// default = scene material). The mesh is triangulated by fanning each face.
//   n        — the geometric normal (one per triangle): ray offsets, the ground, backface
//              flipping, and anything that must follow the real surface.
//   na/nb/nc — the SHADING normals at the three corners, interpolated per hit. Equal to n
//              when the corner is flat-shaded, so flat shading is the zero-cost default
//              path through the same code (see RenderSettings::smooth_angle).
struct Tri { V3 a, b, c, n; V3 na, nb, nc; V3 albedo; double metallic; double rough; V3 emission{0, 0, 0};
             int tex = 0; double tex_scale = 4.0; V3 tex2{0, 0, 0};
             int alb_tex = -1, rgh_tex = -1, nrm_tex = -1; double uv_scale = 1.0; };  // image-map indices

// Axis-aligned bounding box + a binary BVH so intersection is O(log n), not O(n)
// per ray — the difference between seconds and minutes on a subdivided mesh.
struct AABB {
    V3 lo{1e30, 1e30, 1e30}, hi{-1e30, -1e30, -1e30};
    void expand(const V3& p) { for (int k = 0; k < 3; ++k) { lo[k] = std::min(lo[k], p[k]); hi[k] = std::max(hi[k], p[k]); } }
    void expand(const AABB& b) { for (int k = 0; k < 3; ++k) { lo[k] = std::min(lo[k], b.lo[k]); hi[k] = std::max(hi[k], b.hi[k]); } }
};
struct BVHNode {
    AABB box;
    int left = -1, right = -1;  // children (internal nodes)
    int start = 0, count = 0;   // triangle range into `order` (leaf: count > 0)
};

// Ray/AABB slab test (invd may be +/-inf for axis-aligned rays; IEEE handles it).
bool hit_aabb(const AABB& b, const V3& o, const V3& invd, double tmax) {
    double t0 = 1e-5, t1 = tmax;
    for (int k = 0; k < 3; ++k) {
        double ta = (b.lo[k] - o[k]) * invd[k], tb = (b.hi[k] - o[k]) * invd[k];
        if (ta > tb) std::swap(ta, tb);
        t0 = std::max(t0, ta); t1 = std::min(t1, tb);
        if (t0 > t1) return false;
    }
    return true;
}

// Möller-Trumbore; returns t (>eps) on hit, else -1. On a hit, out_u/out_v (when given)
// receive the barycentric coordinates of b and c — what the shading normal interpolates on.
double ray_tri(const V3& o, const V3& d, const Tri& t, double* out_u = nullptr, double* out_v = nullptr) {
    V3 e1 = t.b - t.a, e2 = t.c - t.a, p = cross(d, e2);
    double det = dot(e1, p);
    if (std::fabs(det) < 1e-12) return -1;
    double inv = 1.0 / det;
    V3 tv = o - t.a;
    double u = dot(tv, p) * inv;
    if (u < 0 || u > 1) return -1;
    V3 q = cross(tv, e1);
    double v = dot(d, q) * inv;
    if (v < 0 || u + v > 1) return -1;
    double tt = dot(e2, q) * inv;
    if (tt <= 1e-5) return -1;
    if (out_u) *out_u = u;
    if (out_v) *out_v = v;
    return tt;
}

struct Hit {
    double t = 1e30;
    V3 n{0, 0, 1};    // geometric normal (ray offsets, ground, backface side)
    V3 ns{0, 0, 1};   // shading normal (interpolated; == n when flat-shaded)
    V3 albedo{0, 0, 0};
    double metallic = 0.0, rough = 0.5;
    V3 emission{0, 0, 0};
    int tex = 0; double tex_scale = 4.0; V3 tex2{0, 0, 0};
    int alb_tex = -1, rgh_tex = -1, nrm_tex = -1; double uv_scale = 1.0;
    bool is_ground = false;
};

// The sun — a warm directional key light, sampled explicitly (next-event estimation)
// for crisp shadows. Its DIRECTION is art-directable (RenderSettings::sun_dir, carried on
// the Scene); a low sun rakes long shadows across the ground, which the tracer resolves
// beautifully. The default matches the viewport's key light so preview and render agree.
const V3 SUN_E{6.5, 6.0, 5.0};      // sun irradiance (warm white)
constexpr double SUN_SOFT = 0.025;  // angular jitter -> soft penumbra

// Sky-only environment (the sun is added via NEE, never via the sky, so it can't
// be double-counted or spawn fireflies). A cool horizon->zenith gradient fill.
V3 sky(const V3& d) {
    const double up = std::clamp(d[2] * 0.5 + 0.5, 0.0, 1.0);
    return (V3{0.52, 0.60, 0.76} * (1.0 - up) + V3{0.16, 0.28, 0.52} * up) * 0.75;
}

// Deterministic per-sample RNG (xorshift32) — reproducible renders, no global state.
struct Rng {
    std::uint32_t s;
    explicit Rng(std::uint32_t seed) : s(seed ? seed : 0x9e3779b9u) {}
    double next() {
        s ^= s << 13; s ^= s >> 17; s ^= s << 5;
        return (s & 0xffffff) / double(0x1000000);  // [0,1)
    }
};

// Cosine-weighted hemisphere sample around n (pdf = cos/pi, which cancels the
// Lambertian cos and 1/pi so throughput just multiplies by albedo).
V3 cosine_sample(const V3& n, Rng& rng) {
    const double r1 = rng.next(), r2 = rng.next();
    const double phi = 2 * PI * r1, r = std::sqrt(r2);
    const V3 w = n;
    V3 a = std::fabs(w[0]) > 0.9 ? V3{0, 1, 0} : V3{1, 0, 0};
    const V3 u = norm(cross(a, w));
    const V3 v = cross(w, u);
    return norm(u * (r * std::cos(phi)) + v * (r * std::sin(phi)) + w * std::sqrt(1.0 - r2));
}

// --- Cook-Torrance microfacet (GGX) — same model as the realtime viewport ---
double luminance(const V3& c) { return 0.2126 * c[0] + 0.7152 * c[1] + 0.0722 * c[2]; }
V3 reflectv(const V3& v, const V3& n) { return n * (2.0 * dot(v, n)) - v; }  // mirror v about n
double D_ggx(double NoH, double a) { double a2 = a * a; double d = NoH * NoH * (a2 - 1) + 1; return a2 / (PI * d * d + 1e-12); }
double G_smith(double NoV, double NoL, double a) {
    double k = a * 0.5;
    return (NoV / (NoV * (1 - k) + k)) * (NoL / (NoL * (1 - k) + k));
}
V3 fresnel(double VoH, const V3& f0) { double p = std::pow(1.0 - VoH, 5.0); return f0 + (V3{1, 1, 1} - f0) * p; }

// Importance-sample a GGX half-vector around n.
V3 sample_ggx(const V3& n, double a, Rng& rng) {
    const double u1 = rng.next(), u2 = rng.next();
    const double ct = std::sqrt((1.0 - u1) / (1.0 + (a * a - 1.0) * u1));
    const double st = std::sqrt(std::max(0.0, 1.0 - ct * ct)), phi = 2 * PI * u2;
    const V3 t = std::fabs(n[0]) > 0.9 ? V3{0, 1, 0} : V3{1, 0, 0};
    const V3 tx = norm(cross(t, n)), ty = cross(n, tx);
    return norm(tx * (st * std::cos(phi)) + ty * (st * std::sin(phi)) + n * ct);
}

// --- image textures: real map files (albedo / roughness / normal), sampled TRIPLANAR (no UVs) ---
struct Texture {
    int w = 0, h = 0, ch = 0;
    std::vector<unsigned char> data;
    bool ok() const { return w > 0 && h > 0 && !data.empty(); }
    V3 texel(int x, int y) const {
        x = ((x % w) + w) % w; y = ((y % h) + h) % h;              // wrap
        const unsigned char* px = &data[(std::size_t(y) * w + x) * ch];
        if (ch == 1) { const double g = px[0] / 255.0; return {g, g, g}; }
        return {px[0] / 255.0, px[1] / 255.0, px[2] / 255.0};
    }
    V3 sample(double u, double v) const {                          // bilinear
        const double fx = u * w - 0.5, fy = v * h - 0.5;
        const int x0 = int(std::floor(fx)), y0 = int(std::floor(fy));
        const double tx = fx - x0, ty = fy - y0;
        const V3 a = texel(x0, y0), b = texel(x0 + 1, y0), c = texel(x0, y0 + 1), d = texel(x0 + 1, y0 + 1);
        return (a * (1 - tx) + b * tx) * (1 - ty) + (c * (1 - tx) + d * tx) * ty;
    }
};

// Minimal binary PPM reader (P5 gray / P6 rgb) — no image-decoder dependency; a real CC0 PBR
// set exported as PPM drops in unchanged. Empty texture on any failure (renderer falls back).
Texture load_ppm(const std::string& path) {
    Texture t;
    std::ifstream f(path, std::ios::binary);
    if (!f) return t;
    std::string magic; f >> magic;
    if (magic != "P5" && magic != "P6") return t;
    int maxv = 255;
    f >> t.w >> t.h >> maxv;
    f.get();  // consume the single whitespace before the binary block
    if (t.w <= 0 || t.h <= 0) { t.w = t.h = 0; return t; }
    t.ch = (magic == "P6") ? 3 : 1;
    t.data.resize(std::size_t(t.w) * t.h * t.ch);
    f.read(reinterpret_cast<char*>(t.data.data()), std::streamsize(t.data.size()));
    if (!f) { t.w = t.h = 0; t.data.clear(); }
    return t;
}

// Triplanar blend: sample a texture at world position p, weighting the three axis-plane
// projections by the (squared) geometric normal, so surfaces need no UVs and show no seams.
// uv_scale = world units per texture tile.
V3 triplanar(const Texture& tex, const V3& p, const V3& n, double uv_scale) {
    const double s = uv_scale > 1e-6 ? 1.0 / uv_scale : 1.0;
    V3 w{n[0] * n[0], n[1] * n[1], n[2] * n[2]};
    const double ws = w[0] + w[1] + w[2] + 1e-9;
    w = w * (1.0 / ws);
    const V3 cx = tex.sample(p[1] * s, p[2] * s);
    const V3 cy = tex.sample(p[0] * s, p[2] * s);
    const V3 cz = tex.sample(p[0] * s, p[1] * s);
    return cx * w[0] + cy * w[1] + cz * w[2];
}

// Perturb the shading normal by a tangent-space normal-map value tn in [-1,1]^3 using a
// consistent frame from N (exact for the axis-aligned faces that dominate hard-surface scenes).
V3 perturb_normal(const V3& N, const V3& tn, double strength) {
    const V3 aRef = std::fabs(N[0]) > 0.9 ? V3{0, 1, 0} : V3{1, 0, 0};
    const V3 T = norm(cross(aRef, N)), B = cross(N, T);
    return norm(T * (tn[0] * strength) + B * (tn[1] * strength) + N * std::max(tn[2], 0.05));
}

struct Scene {
    std::vector<Tri> tris;
    std::vector<AABB> tbox;     // per-triangle box
    std::vector<V3> tcen;       // per-triangle centroid
    std::vector<int> order;     // triangle indices, reordered by the BVH build
    std::vector<BVHNode> nodes;
    bool ground = true;
    double ground_z = 0.0;
    double ground_r2 = 1e9;     // ground disk radius squared (around the model center)
    V3 ground_center{0, 0, 0};
    V3 ground_albedo{0.40, 0.42, 0.46};
    V3 albedo{0.8, 0.8, 0.8};
    double metallic = 0.0, roughness = 0.5;
    double env_intensity = 1.0;     // scales the sky image-based fill
    double sun_intensity = 1.0;     // scales the NEE directional key
    V3 sun_dir{0.4, 0.5, 0.8};      // the sun's direction (normalized in path_trace)
    double clamp_indirect = 12.0;   // firefly cap on indirect contributions (0 = off)
    std::vector<int> emitter_idx;   // triangle indices that emit (area lights, for NEE)
    std::vector<Texture> textures;                    // loaded image maps (shared by index)
    std::unordered_map<std::string, int> tex_cache;   // path -> index (load each map once)
};

// Recursive median-split BVH build. Returns the node index of the built subtree.
int build_bvh(Scene& sc, int start, int count) {
    const int idx = int(sc.nodes.size());
    sc.nodes.push_back({});
    AABB box;
    for (int i = start; i < start + count; ++i) box.expand(sc.tbox[sc.order[i]]);
    if (count <= 4) {  // leaf
        sc.nodes[idx].box = box; sc.nodes[idx].start = start; sc.nodes[idx].count = count;
        return idx;
    }
    AABB cb;  // split on the longest axis of the centroid bounds
    for (int i = start; i < start + count; ++i) cb.expand(sc.tcen[sc.order[i]]);
    int axis = 0;
    double e0 = cb.hi[0] - cb.lo[0], e1 = cb.hi[1] - cb.lo[1], e2 = cb.hi[2] - cb.lo[2];
    if (e1 > e0 && e1 >= e2) axis = 1; else if (e2 > e0 && e2 > e1) axis = 2;
    const double mid = 0.5 * (cb.lo[axis] + cb.hi[axis]);
    int m = int(std::partition(sc.order.begin() + start, sc.order.begin() + start + count,
                               [&](int ti) { return sc.tcen[ti][axis] < mid; }) - sc.order.begin());
    if (m == start || m == start + count) m = start + count / 2;  // degenerate split -> median count
    const int l = build_bvh(sc, start, m - start);
    const int r = build_bvh(sc, m, start + count - m);
    sc.nodes[idx].box = box; sc.nodes[idx].left = l; sc.nodes[idx].right = r; sc.nodes[idx].count = 0;
    return idx;
}

void hit_ground(const Scene& sc, const V3& o, const V3& d, Hit& h) {
    if (!sc.ground || std::fabs(d[2]) <= 1e-9) return;
    double tt = (sc.ground_z - o[2]) / d[2];
    if (tt > 1e-5 && tt < h.t) {
        V3 p = o + d * tt;
        double dx = p[0] - sc.ground_center[0], dy = p[1] - sc.ground_center[1];
        if (dx * dx + dy * dy < sc.ground_r2) {
            h.t = tt; h.n = {0, 0, 1}; h.ns = {0, 0, 1}; h.albedo = sc.ground_albedo;
            h.metallic = 0.0; h.rough = 0.92; h.is_ground = true;  // matte floor
        }
    }
}

Hit intersect(const Scene& sc, const V3& o, const V3& d) {
    Hit h;
    const V3 invd{1.0 / d[0], 1.0 / d[1], 1.0 / d[2]};
    int stack[64], sp = 0;
    if (!sc.nodes.empty()) stack[sp++] = 0;
    while (sp) {
        const BVHNode& n = sc.nodes[stack[--sp]];
        if (!hit_aabb(n.box, o, invd, h.t)) continue;
        if (n.count > 0) {
            for (int i = n.start; i < n.start + n.count; ++i) {
                const Tri& t = sc.tris[sc.order[i]];
                double bu = 0, bv = 0;
                double tt = ray_tri(o, d, t, &bu, &bv);
                if (tt > 0 && tt < h.t) {
                    h.t = tt;
                    V3 nn = t.n;
                    // Shading normal: barycentric blend of the corner normals. Both normals are
                    // built from the same outward winding, so a backface hit flips them together
                    // and they stay on the same side of the surface.
                    V3 sn = norm(t.na * (1.0 - bu - bv) + t.nb * bu + t.nc * bv);
                    if (dot(nn, d) > 0) { nn = nn * -1.0; sn = sn * -1.0; }  // two-sided
                    h.n = nn; h.ns = sn;
                    h.albedo = t.albedo; h.metallic = t.metallic; h.rough = t.rough;
                    h.emission = t.emission; h.tex = t.tex; h.tex_scale = t.tex_scale; h.tex2 = t.tex2;
                    h.alb_tex = t.alb_tex; h.rgh_tex = t.rgh_tex; h.nrm_tex = t.nrm_tex; h.uv_scale = t.uv_scale;
                    h.is_ground = false;
                }
            }
        } else { stack[sp++] = n.left; stack[sp++] = n.right; }
    }
    hit_ground(sc, o, d, h);
    return h;
}

// Any-hit shadow test up to tmax (infinite for the sun; the distance to an area light
// otherwise, so geometry BEHIND the light doesn't cast a false shadow).
bool occluded(const Scene& sc, const V3& o, const V3& d, double tmax = 1e30) {
    const V3 invd{1.0 / d[0], 1.0 / d[1], 1.0 / d[2]};
    int stack[64], sp = 0;
    if (!sc.nodes.empty()) stack[sp++] = 0;
    while (sp) {
        const BVHNode& n = sc.nodes[stack[--sp]];
        if (!hit_aabb(n.box, o, invd, tmax)) continue;
        if (n.count > 0) {
            for (int i = n.start; i < n.start + n.count; ++i) {
                double tt = ray_tri(o, d, sc.tris[sc.order[i]]);
                if (tt > 0 && tt < tmax) return true;
            }
        } else { stack[sp++] = n.left; stack[sp++] = n.right; }
    }
    if (sc.ground && std::fabs(d[2]) > 1e-9) {
        double tt = (sc.ground_z - o[2]) / d[2];
        if (tt > 1e-5 && tt < tmax) {
            V3 p = o + d * tt;
            double dx = p[0] - sc.ground_center[0], dy = p[1] - sc.ground_center[1];
            if (dx * dx + dy * dy < sc.ground_r2) return true;
        }
    }
    return false;
}

// A sun direction jittered within its small cone (soft shadows).
V3 jittered_sun(const V3& sun, Rng& rng) {
    V3 a = std::fabs(sun[0]) > 0.9 ? V3{0, 1, 0} : V3{1, 0, 0};
    const V3 u = norm(cross(a, sun)), v = cross(sun, u);
    const double r = SUN_SOFT * std::sqrt(rng.next()), phi = 2 * PI * rng.next();
    return norm(sun + u * (r * std::cos(phi)) + v * (r * std::sin(phi)));
}

// Area-light next-event estimation: sample a point on a uniformly-chosen emitter triangle,
// so lamps / windows actually ILLUMINATE the scene (not just be visible). Two-sided.
struct LightSample { V3 pos, n, Le; double pdf_area = 0.0; };
LightSample sample_emitter(const Scene& sc, Rng& rng) {
    LightSample ls;
    if (sc.emitter_idx.empty()) return ls;
    const int m = int(sc.emitter_idx.size());
    const Tri& t = sc.tris[sc.emitter_idx[std::min(int(rng.next() * m), m - 1)]];
    double u = rng.next(), v = rng.next();
    if (u + v > 1.0) { u = 1.0 - u; v = 1.0 - v; }
    const V3 e1 = t.b - t.a, e2 = t.c - t.a;
    ls.pos = t.a + e1 * u + e2 * v;
    ls.n = t.n; ls.Le = t.emission;
    const double area = 0.5 * len(cross(e1, e2));
    ls.pdf_area = area > 1e-12 ? 1.0 / (area * m) : 0.0;   // pick a triangle then a point on it
    return ls;
}

// --- procedural object-space textures (no UVs): modulate albedo by a pattern of the world
// hit position, so wood grain / fabric weave / stone break up flat colour. Value-noise fBm. ---
double thash(int i, int j, int k) {
    unsigned n = (unsigned)(i * 374761393 + j * 668265263 + k * 1442695040);
    n = (n ^ (n >> 13)) * 1274126177u;
    return (n & 0xffff) / 65535.0;
}
double vnoise3(const V3& p) {
    const int xi = int(std::floor(p[0])), yi = int(std::floor(p[1])), zi = int(std::floor(p[2]));
    const double xf = p[0] - xi, yf = p[1] - yi, zf = p[2] - zi;
    auto sm = [](double t) { return t * t * (3 - 2 * t); };
    const double u = sm(xf), v = sm(yf), w = sm(zf);
    auto layer = [&](int dz) {
        const double a = thash(xi, yi, zi + dz), b = thash(xi + 1, yi, zi + dz);
        const double c = thash(xi, yi + 1, zi + dz), d = thash(xi + 1, yi + 1, zi + dz);
        return (a * (1 - u) + b * u) * (1 - v) + (c * (1 - u) + d * u) * v;
    };
    return layer(0) * (1 - w) + layer(1) * w;
}
double tfbm(const V3& p, int oct) {
    double s = 0, amp = 0.5, f = 1.0;
    for (int o = 0; o < oct; ++o) { s += amp * vnoise3(p * f); amp *= 0.5; f *= 2.0; }
    return s;
}
V3 apply_tex(const V3& base, int tex, double scale, const V3& c2, const V3& pos) {
    const V3 q = pos * scale;
    double t = 0.0;
    if (tex == 1) {            // wood: grain lines along y, gently warped by noise
        const double grain = std::sin((q[1] + tfbm(q * 0.6, 3) * 1.3) * 6.28318);
        t = 0.12 + 0.42 * (0.5 + 0.5 * grain) * (0.5 + 0.5 * grain);
    } else if (tex == 2) {     // fabric: a fine weave plus a soft mottle
        const double weave = std::sin(q[0] * 12.0) * std::sin(q[1] * 12.0);
        t = std::clamp(0.5 + 0.26 * weave + 0.24 * (tfbm(q * 1.5, 2) - 0.5), 0.0, 1.0) * 0.5;
    } else if (tex == 3) {     // stone / plaster: low-frequency veined mottle
        t = std::clamp(tfbm(q * 0.6, 4), 0.0, 1.0) * 0.5;
    }
    return base * (1.0 - t) + c2 * t;
}

// One path: a Cook-Torrance microfacet surface (diffuse + GGX specular) gathered
// by lobe-importance-sampled BSDF bounces (sky on miss), with the sun added by
// next-event estimation at each hit (crisp shadows + sharp speculars, low noise).
V3 radiance(const Scene& sc, V3 o, V3 d, int max_bounce, Rng& rng) {
    V3 L{0, 0, 0}, beta{1, 1, 1};
    // Add a light contribution, clamping its luminance on INDIRECT bounces so a rare
    // hot specular sample can't leave a firefly (direct, bounce 0, stays crisp).
    auto add = [&](V3 c, int bounce) {
        if (sc.clamp_indirect > 0 && bounce >= 1) {
            const double lm = luminance(c);
            if (lm > sc.clamp_indirect) c = c * (sc.clamp_indirect / lm);
        }
        L = L + c;
    };
    bool spec_bounce = true;   // camera ray + specular bounces may see an emitter directly;
                               // after a diffuse bounce the emitter is covered by NEE (no double-count)
    for (int bounce = 0; bounce < max_bounce; ++bounce) {
        Hit h = intersect(sc, o, d);
        if (h.t > 1e29) { add(mulv(beta, sky(d) * sc.env_intensity), bounce); break; }  // escaped -> sky fill
        if (luminance(h.emission) > 0.0 && (bounce == 0 || spec_bounce))
            add(mulv(beta, h.emission), bounce);                                          // a lamp seen directly
        const V3 wp = o + d * h.t;   // world hit position (texture lookups + offsets)
        // Resolve the surface: image maps (albedo / roughness / normal) win over the procedural
        // tex, which wins over the flat colour. Triplanar-projected off the SHADING normal, so
        // the three projections blend smoothly across a curved surface instead of stepping at
        // every facet boundary.
        V3 N = h.ns;
        if (h.nrm_tex >= 0) {
            const V3 tn = triplanar(sc.textures[h.nrm_tex], wp, h.ns, h.uv_scale) * 2.0 - V3{1, 1, 1};
            N = perturb_normal(h.ns, tn, 1.0);
        }
        V3 alb;
        if (h.alb_tex >= 0)  alb = triplanar(sc.textures[h.alb_tex], wp, h.ns, h.uv_scale);
        else if (h.tex)      alb = apply_tex(h.albedo, h.tex, h.tex_scale, h.tex2, wp);
        else                 alb = h.albedo;
        double rough = h.rough;
        if (h.rgh_tex >= 0)  rough = std::clamp(triplanar(sc.textures[h.rgh_tex], wp, h.ns, h.uv_scale)[0], 0.03, 1.0);
        const V3 V = d * -1.0;
        const double NoV = std::max(dot(N, V), 1e-4);
        const double a = std::max(rough * rough, 1e-3);
        const V3 f0 = V3{0.04, 0.04, 0.04} * (1.0 - h.metallic) + alb * h.metallic;
        const V3 diff_alb = alb * (1.0 - h.metallic);
        const V3 p = wp + h.n * 1e-4;   // offset along the geometric normal (stable shadow origin)

        // NEE: direct sun (diffuse + specular through the microfacet BRDF)
        const V3 sdir = jittered_sun(sc.sun_dir, rng);
        const double NoL = dot(N, sdir);
        if (NoL > 0 && !occluded(sc, p, sdir)) {
            const V3 H = norm(V + sdir);
            const double NoH = std::max(dot(N, H), 0.0), VoH = std::max(dot(V, H), 0.0);
            const V3 F = fresnel(VoH, f0);
            const V3 spec = F * (D_ggx(NoH, a) * G_smith(NoV, NoL, a) / (4 * NoV * NoL + 1e-6));
            const V3 fr = diff_alb * (1.0 / PI) + spec;
            add(mulv(mulv(beta, fr), SUN_E * sc.sun_intensity) * NoL, bounce);
        }

        // NEE: area lights (lamps / emissive surfaces) actually lighting the room
        if (!sc.emitter_idx.empty()) {
            const LightSample ls = sample_emitter(sc, rng);
            if (ls.pdf_area > 0.0) {
                const V3 to = ls.pos - p;
                const double dist2 = std::max(dot(to, to), 1e-8), dist = std::sqrt(dist2);
                const V3 wl = to * (1.0 / dist);
                const double lNoL = dot(N, wl), cosl = std::fabs(dot(ls.n, wl));   // two-sided emitter
                if (lNoL > 0 && cosl > 1e-4 && !occluded(sc, p, wl, dist - 1e-3)) {
                    const V3 H = norm(V + wl);
                    const double NoH = std::max(dot(N, H), 0.0), VoH = std::max(dot(V, H), 0.0);
                    const V3 F = fresnel(VoH, f0);
                    const V3 spec = F * (D_ggx(NoH, a) * G_smith(NoV, lNoL, a) / (4 * NoV * lNoL + 1e-6));
                    const V3 fr = diff_alb * (1.0 / PI) + spec;
                    const double G = lNoL * cosl / dist2;                          // geometry term
                    add(mulv(mulv(beta, fr), ls.Le) * (G / ls.pdf_area), bounce);
                }
            }
        }

        // indirect: stochastically pick the diffuse or specular lobe
        const double lf = luminance(f0), ld = luminance(diff_alb);
        const double pSpec = std::clamp(lf / (lf + ld + 1e-4), 0.1, 0.9);
        o = p;
        if (rng.next() < pSpec) {                    // GGX specular bounce
            const V3 Hh = sample_ggx(N, a, rng);
            const V3 Ld = reflectv(V, Hh);
            const double nl = dot(N, Ld);
            if (nl <= 0) break;
            const double VoH = std::max(dot(V, Hh), 1e-4), NoH = std::max(dot(N, Hh), 1e-4);
            const V3 F = fresnel(VoH, f0);
            beta = mulv(beta, F) * (G_smith(NoV, nl, a) * VoH / (NoV * NoH) / pSpec);
            d = Ld; spec_bounce = true;
        } else {                                     // Lambertian diffuse bounce
            d = cosine_sample(N, rng);
            beta = mulv(beta, diff_alb) * (1.0 / (1.0 - pSpec));
            spec_bounce = false;
        }
        if (bounce >= 3) {                           // Russian roulette
            double q = std::max({beta[0], beta[1], beta[2]});
            if (rng.next() > q) break;
            beta = beta * (1.0 / std::max(q, 1e-4));
        }
    }
    return L;
}

V3 aces(const V3& x) {
    auto f = [](double v) {
        return std::clamp((v * (2.51 * v + 0.03)) / (v * (2.43 * v + 0.59) + 0.14), 0.0, 1.0);
    };
    return {f(x[0]), f(x[1]), f(x[2])};
}

// Bloom: isolate bright pixels (soft knee above `threshold`), blur them wide with a few
// separable-Gaussian passes at widening tap spacing, and add the glow back into the linear
// HDR image before tonemapping — the photographic bleed of light sources and hot highlights.
void bloom_hdr(std::vector<V3>& hdr, int w, int h, double threshold, double strength) {
    const std::size_t N = std::size_t(w) * h;
    std::vector<V3> bright(N), tmp(N);
    for (std::size_t i = 0; i < N; ++i) {
        const double l = luminance(hdr[i]);
        const double k = l > threshold ? (l - threshold) / std::max(l, 1e-6) : 0.0;  // soft knee
        bright[i] = hdr[i] * k;
    }
    const double g[3] = {0.375, 0.25, 0.0625};   // energy-preserving 5-tap row (sums to 1)
    for (int it = 0; it < 4; ++it) {
        const int step = 1 << it;                // 1,2,4,8 -> a broad glow in O(passes) taps
        for (int y = 0; y < h; ++y) for (int x = 0; x < w; ++x) {   // horizontal
            V3 s = bright[std::size_t(y) * w + x] * g[0];
            for (int t = 1; t <= 2; ++t) {
                const int xl = std::max(0, x - t * step), xr = std::min(w - 1, x + t * step);
                s = s + (bright[std::size_t(y) * w + xl] + bright[std::size_t(y) * w + xr]) * g[t];
            }
            tmp[std::size_t(y) * w + x] = s;
        }
        for (int y = 0; y < h; ++y) for (int x = 0; x < w; ++x) {   // vertical
            V3 s = tmp[std::size_t(y) * w + x] * g[0];
            for (int t = 1; t <= 2; ++t) {
                const int yl = std::max(0, y - t * step), yr = std::min(h - 1, y + t * step);
                s = s + (tmp[std::size_t(yl) * w + x] + tmp[std::size_t(yr) * w + x]) * g[t];
            }
            bright[std::size_t(y) * w + x] = s;
        }
    }
    for (std::size_t i = 0; i < N; ++i) hdr[i] = hdr[i] + bright[i] * strength;
}

// Edge-avoiding a-trous wavelet denoise (Dammertz et al. 2010): blur the HDR
// illumination to kill Monte-Carlo grain, but weight each tap by how well the
// neighbour matches the pixel's primary normal / depth / brightness, so real edges
// (silhouettes, face boundaries, shadow terminators) survive. Albedo is demodulated
// first — we denoise the *lighting* and re-apply the texture — so material detail
// isn't smeared. Guided by the noise-free primary G-buffer (centre-ray hit). Five
// passes with doubling tap spacing cover a wide radius at O(25) taps per pixel.
void denoise_atrous(std::vector<V3>& color, const std::vector<V3>& gAlb,
                    const std::vector<V3>& gNrm, const std::vector<double>& gDep,
                    const std::vector<char>& gMask, int w, int h, int iters, double scale) {
    const std::size_t N = std::size_t(w) * h;
    std::vector<V3> irr(N), tmp(N);
    for (std::size_t i = 0; i < N; ++i) {              // demodulate albedo -> lighting only
        const V3 a = gAlb[i];
        irr[i] = {color[i][0] / std::max(a[0], 0.03),
                  color[i][1] / std::max(a[1], 0.03),
                  color[i][2] / std::max(a[2], 0.03)};
    }
    const double kern[5] = {0.0625, 0.25, 0.375, 0.25, 0.0625};   // B3-spline row
    const double sn = 64.0;                            // normal sharpness (rejects across edges)
    const double sz = std::max(scale * 0.06, 1e-4);    // depth sigma (scene-relative)
    double sl = 6.0;                                   // luminance sigma (loosened per pass)
    for (int it = 0; it < iters; ++it) {
        const int step = 1 << it;                      // 1,2,4,8,16 — the "a-trous" holes
        for (int y = 0; y < h; ++y) for (int x = 0; x < w; ++x) {
            const std::size_t p = std::size_t(y) * w + x;
            if (!gMask[p]) { tmp[p] = irr[p]; continue; }   // sky/miss: leave untouched
            const V3 np = gNrm[p], cp = irr[p];
            const double zp = gDep[p], lp = luminance(cp);
            V3 sum{0, 0, 0}; double wsum = 0.0;
            for (int dy = -2; dy <= 2; ++dy) for (int dx = -2; dx <= 2; ++dx) {
                const int xx = x + dx * step, yy = y + dy * step;
                if (xx < 0 || xx >= w || yy < 0 || yy >= h) continue;
                const std::size_t q = std::size_t(yy) * w + xx;
                if (!gMask[q]) continue;
                const double wn = std::pow(std::max(0.0, dot(np, gNrm[q])), sn);
                const double wz = std::exp(-std::fabs(zp - gDep[q]) / sz);
                const double wl = std::exp(-std::fabs(lp - luminance(irr[q])) / sl);
                const double wk = kern[dx + 2] * kern[dy + 2] * wn * wz * wl;
                sum = sum + irr[q] * wk; wsum += wk;
            }
            tmp[p] = wsum > 1e-8 ? sum * (1.0 / wsum) : irr[p];
        }
        std::swap(irr, tmp);
        sl *= 2.0;                                     // grain averaged out -> loosen the colour test
    }
    for (std::size_t i = 0; i < N; ++i) color[i] = mulv(irr[i], gAlb[i]);   // remodulate texture
}

}  // namespace

Image path_trace(const Mesh& mesh, const Camera& cam, const RenderSettings& settings) {
    // Build the triangle soup + ground from the mesh bounds.
    Scene sc;
    sc.albedo = settings.albedo;
    sc.metallic = settings.metallic;
    sc.roughness = settings.roughness;
    sc.ground = settings.ground;
    sc.env_intensity = settings.env_intensity;
    sc.sun_intensity = settings.sun_intensity;
    sc.sun_dir = norm({settings.sun_dir[0], settings.sun_dir[1], settings.sun_dir[2]});
    sc.clamp_indirect = settings.clamp_indirect;
    auto load_tex = [&](const std::string& path) -> int {   // load each map file once, keyed by path
        if (path.empty()) return -1;
        auto it = sc.tex_cache.find(path);
        if (it != sc.tex_cache.end()) return it->second;
        Texture tx = load_ppm(path);
        const int idx = tx.ok() ? int(sc.textures.size()) : -1;
        if (tx.ok()) sc.textures.push_back(std::move(tx));
        sc.tex_cache[path] = idx;
        return idx;
    };
    // --- Smooth shading, by angle (RenderSettings::smooth_angle) ---
    // A face corner's shading normal is the area-weighted average of the normals of the faces
    // meeting at that vertex whose own normal is within `smooth_angle` of this face's. Faces
    // across a sharper edge are excluded, so the SAME vertex can be smooth on one face and hard
    // on the next (a cylinder's side is round while its cap rim stays crisp) — no authoring, no
    // per-face flags. Adjacency is stored CSR (one contiguous array rather than a vector per
    // vertex), so this costs one extra pass over the faces and stays cheap at millions of them.
    const double smooth_deg = std::clamp(settings.smooth_angle, 0.0, 180.0);
    const bool want_smooth = smooth_deg > 0.0;
    const double cos_smooth = std::cos(smooth_deg * PI / 180.0);
    std::vector<V3> fnorm;
    std::vector<double> farea;
    std::vector<int> voff, vadj;   // faces around vertex v = vadj[voff[v] .. voff[v+1])
    if (want_smooth) {
        const std::size_t NF = mesh.num_faces(), NV = mesh.num_verts();
        fnorm.resize(NF);
        farea.resize(NF);
        voff.assign(NV + 1, 0);
        std::size_t fi = 0;
        for (const auto& f : mesh.faces()) {
            const auto a = face_normal(mesh, f.get());
            fnorm[fi] = {a[0], a[1], a[2]};
            farea[fi] = face_area(mesh, f.get());
            for (Loop* lp : mesh.face_loops(f.get())) ++voff[lp->vert->id + 1];
            ++fi;
        }
        for (std::size_t i = 1; i <= NV; ++i) voff[i] += voff[i - 1];
        vadj.resize(std::size_t(voff[NV]));
        std::vector<int> cur(voff.begin(), voff.end() - 1);
        fi = 0;
        for (const auto& f : mesh.faces()) {
            for (Loop* lp : mesh.face_loops(f.get())) vadj[std::size_t(cur[lp->vert->id]++)] = int(fi);
            ++fi;
        }
    }
    // The shading normal at face `fi`'s corner on vertex `v` (fn = that face's normal).
    auto corner_normal = [&](const Vert* v, const V3& fn) -> V3 {
        if (!want_smooth) return fn;
        V3 s{0, 0, 0};
        for (int k = voff[v->id]; k < voff[v->id + 1]; ++k) {
            const int g = vadj[std::size_t(k)];
            if (dot(fnorm[std::size_t(g)], fn) >= cos_smooth) s = s + fnorm[std::size_t(g)] * farea[std::size_t(g)];
        }
        const double l = len(s);
        return l > 1e-12 ? s * (1.0 / l) : fn;   // degenerate fan -> stay flat
    };

    V3 lo{1e30, 1e30, 1e30}, hi{-1e30, -1e30, -1e30};
    std::vector<V3> cn;   // per-corner shading normals, buffer reused across faces
    std::size_t fidx = 0;
    for (const auto& f : mesh.faces()) {
        std::vector<Vert*> vs = mesh.face_verts(f.get());
        V3 n = want_smooth ? fnorm[fidx]
                           : [&] { auto a = face_normal(mesh, f.get()); return V3{a[0], a[1], a[2]}; }();
        cn.resize(vs.size());
        for (std::size_t k = 0; k < vs.size(); ++k) cn[k] = corner_normal(vs[k], n);
        const Material& fm = f->material;  // per-face material, or the scene default
        const V3 alb = fm.set ? V3{fm.color[0], fm.color[1], fm.color[2]} : sc.albedo;
        const double met = fm.set ? fm.metallic : sc.metallic;
        const double rgh = fm.set ? fm.roughness : sc.roughness;
        const V3 emis = fm.set ? V3{fm.emission[0], fm.emission[1], fm.emission[2]} : V3{0, 0, 0};
        const int tex = fm.set ? fm.tex : 0;
        const V3 tc2{fm.tex_color2[0], fm.tex_color2[1], fm.tex_color2[2]};
        const int at = fm.set ? load_tex(fm.albedo_map) : -1;      // image maps -> texture indices
        const int rt = fm.set ? load_tex(fm.roughness_map) : -1;
        const int nt = fm.set ? load_tex(fm.normal_map) : -1;
        const double uvs = fm.set ? fm.uv_scale : 1.0;
        for (std::size_t i = 1; i + 1 < vs.size(); ++i) {
            Tri t;
            t.a = {vs[0]->co[0], vs[0]->co[1], vs[0]->co[2]};
            t.b = {vs[i]->co[0], vs[i]->co[1], vs[i]->co[2]};
            t.c = {vs[i + 1]->co[0], vs[i + 1]->co[1], vs[i + 1]->co[2]};
            t.n = n;
            t.na = cn[0]; t.nb = cn[i]; t.nc = cn[i + 1];   // fan: corner 0 is shared by every tri
            t.albedo = alb; t.metallic = met; t.rough = rgh; t.emission = emis;
            t.tex = tex; t.tex_scale = fm.tex_scale; t.tex2 = tc2;
            t.alb_tex = at; t.rgh_tex = rt; t.nrm_tex = nt; t.uv_scale = uvs;
            sc.tris.push_back(t);
        }
        ++fidx;
    }
    for (int i = 0; i < int(sc.tris.size()); ++i)                     // collect the area lights
        if (luminance(sc.tris[i].emission) > 0.0) sc.emitter_idx.push_back(i);
    for (const auto& v : mesh.verts())
        for (int k = 0; k < 3; ++k) { lo[k] = std::min(lo[k], v->co[k]); hi[k] = std::max(hi[k], v->co[k]); }
    if (mesh.num_verts()) {
        sc.ground_z = lo[2] - 1e-3 * (hi[2] - lo[2] + 1.0);
        sc.ground_center = {(lo[0] + hi[0]) * 0.5, (lo[1] + hi[1]) * 0.5, sc.ground_z};
        double rad = len(hi - lo) * 0.5;
        sc.ground_r2 = (rad * 9.0) * (rad * 9.0);
    }
    const double scene_diag = mesh.num_verts() ? len(hi - lo) : 1.0;  // depth-weight scale for the denoiser

    // BVH over the triangles.
    sc.tbox.resize(sc.tris.size());
    sc.tcen.resize(sc.tris.size());
    sc.order.resize(sc.tris.size());
    for (std::size_t i = 0; i < sc.tris.size(); ++i) {
        AABB b; b.expand(sc.tris[i].a); b.expand(sc.tris[i].b); b.expand(sc.tris[i].c);
        sc.tbox[i] = b;
        sc.tcen[i] = (sc.tris[i].a + sc.tris[i].b + sc.tris[i].c) * (1.0 / 3.0);
        sc.order[i] = int(i);
    }
    if (!sc.tris.empty()) { sc.nodes.reserve(sc.tris.size() * 2); build_bvh(sc, 0, int(sc.tris.size())); }

    // Camera basis.
    const V3 fwd = norm(V3{cam.target[0], cam.target[1], cam.target[2]} -
                        V3{cam.eye[0], cam.eye[1], cam.eye[2]});
    const V3 right = norm(cross(fwd, cam.up));
    const V3 up2 = cross(right, fwd);
    const V3 eye{cam.eye[0], cam.eye[1], cam.eye[2]};
    const double th = std::tan(cam.fov_y * 0.5);
    const double aspect = double(settings.width) / double(settings.height);
    // thin-lens focus distance: explicit, or auto = eye->target distance
    const double focus_dist = settings.focus_dist > 0.0
        ? settings.focus_dist
        : len(V3{cam.target[0], cam.target[1], cam.target[2]} - eye);

    Image img;
    img.w = settings.width;
    img.h = settings.height;
    img.rgb.assign(std::size_t(img.w) * img.h * 3, 0);
    const std::size_t NP = std::size_t(img.w) * img.h;

    // HDR accumulation buffer, plus (only if denoising) a noise-free primary G-buffer:
    // the centre-ray hit's albedo / normal / depth, which guides the edge-avoiding filter.
    std::vector<V3> hdr(NP, V3{0, 0, 0});
    const bool want_gbuf = settings.denoise > 0;
    std::vector<V3> gAlb(want_gbuf ? NP : 0, V3{1, 1, 1});
    std::vector<V3> gNrm(want_gbuf ? NP : 0, V3{0, 0, 1});
    std::vector<double> gDep(want_gbuf ? NP : 0, 1e30);
    std::vector<char> gMask(want_gbuf ? NP : 0, 0);

    unsigned nthreads = settings.threads ? settings.threads : std::thread::hardware_concurrency();
    if (nthreads == 0) nthreads = 4;

    // Pixel -> primary ray direction, through the lens. The normalised coords (a, b) have
    // b in [-1,1] and a in [-aspect, aspect], so r = 1 on the top and bottom edges; the
    // radial term bends the ray a real lens would bend. k1 = k2 = 0 short-circuits to the
    // exact pinhole path. This is the CHEAP direction — no inversion — which is why
    // mirage.solve owns the inverse instead.
    const double lk1 = settings.lens_k1, lk2 = settings.lens_k2;
    const bool has_lens = (lk1 != 0.0 || lk2 != 0.0);
    auto primary = [&](double px, double py) {
        double a = (2.0 * px / img.w - 1.0) * aspect;
        double b = (1.0 - 2.0 * py / img.h);
        if (has_lens) {
            const double r2 = a * a + b * b;
            const double s = 1.0 + lk1 * r2 + lk2 * r2 * r2;
            a *= s; b *= s;
        }
        return norm(fwd + right * (a * th) + up2 * (b * th));
    };

    auto render_rows = [&](int y0, int y1) {
        for (int y = y0; y < y1; ++y) {
            for (int x = 0; x < img.w; ++x) {
                const std::size_t p = std::size_t(y) * img.w + x;
                if (want_gbuf) {  // centre-ray primary hit: albedo/normal/depth (noise-free)
                    const V3 gd = primary(x + 0.5, y + 0.5);
                    const Hit gh = intersect(sc, eye, gd);
                    if (gh.t < 1e29) {
                        // Guide the denoiser with the SHADING normal: the flat normal breaks at every
                        // facet, which makes the edge-avoiding weight reject taps across a smooth
                        // surface and leaves the grain it was meant to remove. Albedo is sampled the
                        // same way radiance() does it, so demodulate/remodulate stays exact.
                        gMask[p] = 1; gNrm[p] = gh.ns; gDep[p] = gh.t;
                        const V3 gwp = eye + gd * gh.t;
                        gAlb[p] = gh.alb_tex >= 0 ? triplanar(sc.textures[gh.alb_tex], gwp, gh.ns, gh.uv_scale)
                                : gh.tex          ? apply_tex(gh.albedo, gh.tex, gh.tex_scale, gh.tex2, gwp)
                                                  : gh.albedo;
                    }
                }
                V3 acc{0, 0, 0};
                for (int s = 0; s < settings.spp; ++s) {
                    Rng rng(std::uint32_t((x * 1973u) ^ (y * 9277u) ^ (s * 26699u)) | 1u);
                    const double jx = rng.next(), jy = rng.next();
                    V3 ro = eye, rd = primary(x + jx, y + jy);
                    if (settings.aperture > 0.0) {   // thin lens: sample the aperture, aim at the focal plane
                        const V3 focal = eye + rd * (focus_dist / std::max(dot(rd, fwd), 1e-6));
                        const double lr = settings.aperture * std::sqrt(rng.next()), la = 2 * PI * rng.next();
                        ro = eye + right * (lr * std::cos(la)) + up2 * (lr * std::sin(la));
                        rd = norm(focal - ro);
                    }
                    acc = acc + radiance(sc, ro, rd, settings.max_bounce, rng);
                }
                hdr[p] = acc * (1.0 / settings.spp);
            }
        }
    };

    std::vector<std::thread> pool;
    int rows = (img.h + int(nthreads) - 1) / int(nthreads);
    for (unsigned i = 0; i < nthreads; ++i) {
        int y0 = int(i) * rows, y1 = std::min(img.h, y0 + rows);
        if (y0 < y1) pool.emplace_back(render_rows, y0, y1);
    }
    for (auto& t : pool) t.join();

    if (settings.denoise > 0)  // edge-avoiding a-trous wavelet, guided by the G-buffer
        denoise_atrous(hdr, gAlb, gNrm, gDep, gMask, img.w, img.h, settings.denoise, scene_diag);

    if (settings.bloom > 0.0)  // photographic glow on bright regions (in linear HDR, pre-tonemap)
        bloom_hdr(hdr, img.w, img.h, settings.bloom_threshold, settings.bloom);

    for (std::size_t p = 0; p < NP; ++p) {  // exposure -> ACES -> gamma -> 8-bit sRGB
        const V3 c = aces(hdr[p] * settings.exposure);
        for (int k = 0; k < 3; ++k)
            img.rgb[p * 3 + k] = (unsigned char)(std::pow(std::clamp(c[k], 0.0, 1.0), 1.0 / 2.2) * 255.0 + 0.5);
    }
    return img;
}

void write_ppm(const Image& img, const std::string& path) {
    std::ofstream f(path, std::ios::binary);
    f << "P6\n" << img.w << " " << img.h << "\n255\n";
    f.write(reinterpret_cast<const char*>(img.rgb.data()), std::streamsize(img.rgb.size()));
}

}  // namespace mirage
