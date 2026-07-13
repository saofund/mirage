#include "mirage/raytrace.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <thread>

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

// Triangle soup with per-triangle geometric normal (the mesh is triangulated by
// fanning each face; flat normals match the hard-surface kernel) and the face's
// PBR material baked in (the `material` op assigns it; default = scene material).
struct Tri { V3 a, b, c, n; V3 albedo; double metallic; double rough; };

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

// Möller-Trumbore; returns t (>eps) on hit, else -1.
double ray_tri(const V3& o, const V3& d, const Tri& t) {
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
    return tt > 1e-5 ? tt : -1;
}

struct Hit {
    double t = 1e30;
    V3 n{0, 0, 1};
    V3 albedo{0, 0, 0};
    double metallic = 0.0, rough = 0.5;
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
            h.t = tt; h.n = {0, 0, 1}; h.albedo = sc.ground_albedo;
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
                double tt = ray_tri(o, d, t);
                if (tt > 0 && tt < h.t) {
                    h.t = tt;
                    V3 nn = t.n; if (dot(nn, d) > 0) nn = nn * -1.0;  // two-sided
                    h.n = nn; h.albedo = t.albedo; h.metallic = t.metallic; h.rough = t.rough;
                    h.is_ground = false;
                }
            }
        } else { stack[sp++] = n.left; stack[sp++] = n.right; }
    }
    hit_ground(sc, o, d, h);
    return h;
}

// Any-hit shadow test (the sun is at infinity, so any intersection occludes it).
bool occluded(const Scene& sc, const V3& o, const V3& d) {
    const V3 invd{1.0 / d[0], 1.0 / d[1], 1.0 / d[2]};
    int stack[64], sp = 0;
    if (!sc.nodes.empty()) stack[sp++] = 0;
    while (sp) {
        const BVHNode& n = sc.nodes[stack[--sp]];
        if (!hit_aabb(n.box, o, invd, 1e30)) continue;
        if (n.count > 0) {
            for (int i = n.start; i < n.start + n.count; ++i)
                if (ray_tri(o, d, sc.tris[sc.order[i]]) > 0) return true;
        } else { stack[sp++] = n.left; stack[sp++] = n.right; }
    }
    if (sc.ground && std::fabs(d[2]) > 1e-9) {
        double tt = (sc.ground_z - o[2]) / d[2];
        if (tt > 1e-5) {
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
    for (int bounce = 0; bounce < max_bounce; ++bounce) {
        Hit h = intersect(sc, o, d);
        if (h.t > 1e29) { add(mulv(beta, sky(d) * sc.env_intensity), bounce); break; }  // escaped -> sky fill
        const V3 N = h.n, V = d * -1.0;
        const double NoV = std::max(dot(N, V), 1e-4);
        const double a = std::max(h.rough * h.rough, 1e-3);
        const V3 f0 = V3{0.04, 0.04, 0.04} * (1.0 - h.metallic) + h.albedo * h.metallic;
        const V3 diff_alb = h.albedo * (1.0 - h.metallic);
        const V3 p = o + d * h.t + N * 1e-4;

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
            d = Ld;
        } else {                                     // Lambertian diffuse bounce
            d = cosine_sample(N, rng);
            beta = mulv(beta, diff_alb) * (1.0 / (1.0 - pSpec));
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
    V3 lo{1e30, 1e30, 1e30}, hi{-1e30, -1e30, -1e30};
    for (const auto& f : mesh.faces()) {
        std::vector<Vert*> vs = mesh.face_verts(f.get());
        V3 n = [&] { auto a = face_normal(mesh, f.get()); return V3{a[0], a[1], a[2]}; }();
        const Material& fm = f->material;  // per-face material, or the scene default
        const V3 alb = fm.set ? V3{fm.color[0], fm.color[1], fm.color[2]} : sc.albedo;
        const double met = fm.set ? fm.metallic : sc.metallic;
        const double rgh = fm.set ? fm.roughness : sc.roughness;
        for (std::size_t i = 1; i + 1 < vs.size(); ++i) {
            Tri t;
            t.a = {vs[0]->co[0], vs[0]->co[1], vs[0]->co[2]};
            t.b = {vs[i]->co[0], vs[i]->co[1], vs[i]->co[2]};
            t.c = {vs[i + 1]->co[0], vs[i + 1]->co[1], vs[i + 1]->co[2]};
            t.n = n;
            t.albedo = alb; t.metallic = met; t.rough = rgh;
            sc.tris.push_back(t);
        }
    }
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

    auto render_rows = [&](int y0, int y1) {
        for (int y = y0; y < y1; ++y) {
            for (int x = 0; x < img.w; ++x) {
                const std::size_t p = std::size_t(y) * img.w + x;
                if (want_gbuf) {  // centre-ray primary hit: albedo/normal/depth (noise-free)
                    const double uc = (2.0 * (x + 0.5) / img.w - 1.0) * aspect * th;
                    const double vc = (1.0 - 2.0 * (y + 0.5) / img.h) * th;
                    const Hit gh = intersect(sc, eye, norm(fwd + right * uc + up2 * vc));
                    if (gh.t < 1e29) { gMask[p] = 1; gAlb[p] = gh.albedo; gNrm[p] = gh.n; gDep[p] = gh.t; }
                }
                V3 acc{0, 0, 0};
                for (int s = 0; s < settings.spp; ++s) {
                    Rng rng(std::uint32_t((x * 1973u) ^ (y * 9277u) ^ (s * 26699u)) | 1u);
                    const double jx = rng.next(), jy = rng.next();
                    const double u = (2.0 * (x + jx) / img.w - 1.0) * aspect * th;
                    const double v = (1.0 - 2.0 * (y + jy) / img.h) * th;
                    const V3 d = norm(fwd + right * u + up2 * v);
                    acc = acc + radiance(sc, eye, d, settings.max_bounce, rng);
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
