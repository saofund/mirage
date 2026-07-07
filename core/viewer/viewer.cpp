// mirage_viewer — the native modeling GUI (.exe): a GL 3.3 viewport over the
// live mirage_core kernel mesh, driven by an op-log (mirage::Program) through a
// Dear ImGui tool panel. Click a tool -> append an op -> the mesh rebuilds and
// re-uploads live. The op-log/history is shown as the model. `--screenshot
// out.ppm` renders one frame headless (for verification).
#include <glad/gl.h>
//
#include <GLFW/glfw3.h>

#include "imgui.h"
#include "imgui_impl_glfw.h"
#include "imgui_impl_opengl3.h"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <map>
#include <fstream>
#include <iterator>
#include <string>
#include <system_error>
#include <thread>
#include <unordered_map>
#include <vector>

#include "mirage/mesh.hpp"
#include "mirage/program.hpp"
#include "mirage/raytrace.hpp"

using namespace mirage;
using Mat4 = std::array<float, 16>;  // column-major (OpenGL)
using V3 = std::array<float, 3>;

// --- tiny matrix/vector math (no GLM dependency) ---------------------------
static Mat4 identity() { Mat4 m{}; m[0] = m[5] = m[10] = m[15] = 1; return m; }
static Mat4 mul(const Mat4& a, const Mat4& b) {
    Mat4 r{};
    for (int c = 0; c < 4; ++c)
        for (int row = 0; row < 4; ++row) {
            float s = 0;
            for (int k = 0; k < 4; ++k) s += a[k * 4 + row] * b[c * 4 + k];
            r[c * 4 + row] = s;
        }
    return r;
}
static Mat4 perspective(float fovy, float asp, float n, float f) {
    const float t = 1.0f / std::tan(fovy * 0.5f);
    Mat4 m{};
    m[0] = t / asp; m[5] = t; m[10] = (f + n) / (n - f); m[11] = -1; m[14] = (2 * f * n) / (n - f);
    return m;
}
static Mat4 ortho(float l, float r, float b, float t, float n, float f) {  // for the shadow light
    Mat4 m{};
    m[0] = 2 / (r - l); m[5] = 2 / (t - b); m[10] = -2 / (f - n); m[15] = 1;
    m[12] = -(r + l) / (r - l); m[13] = -(t + b) / (t - b); m[14] = -(f + n) / (f - n);
    return m;
}
static V3 sub(V3 a, V3 b) { return {a[0] - b[0], a[1] - b[1], a[2] - b[2]}; }
static V3 cross(V3 a, V3 b) { return {a[1]*b[2]-a[2]*b[1], a[2]*b[0]-a[0]*b[2], a[0]*b[1]-a[1]*b[0]}; }
static float dot(V3 a, V3 b) { return a[0]*b[0] + a[1]*b[1] + a[2]*b[2]; }
static V3 norm(V3 a) { float l = std::sqrt(dot(a, a)); if (l == 0) l = 1; return {a[0]/l, a[1]/l, a[2]/l}; }
static Mat4 look_at(V3 eye, V3 c, V3 up) {
    V3 f = norm(sub(c, eye)), s = norm(cross(f, up)), u = cross(s, f);
    Mat4 m = identity();
    m[0]=s[0]; m[4]=s[1]; m[8]=s[2];  m[1]=u[0]; m[5]=u[1]; m[9]=u[2];
    m[2]=-f[0]; m[6]=-f[1]; m[10]=-f[2];
    m[12]=-dot(s, eye); m[13]=-dot(u, eye); m[14]=dot(f, eye);
    return m;
}

// --- GPU geometry (smooth-normal triangulation of a kernel mesh) ------------
struct Gpu { std::vector<float> data; int verts = 0; V3 center{0,0,0}; float radius = 1; };

static Gpu build_gpu(const Mesh& m) {
    std::unordered_map<const Vert*, V3> vn;
    for (const auto& v : m.verts()) vn[v.get()] = {0, 0, 0};
    for (const auto& f : m.faces()) {
        auto fnv = face_normal(m, f.get());
        V3 fn{(float)fnv[0], (float)fnv[1], (float)fnv[2]};
        for (Vert* v : m.face_verts(f.get())) { auto& a = vn[v]; a[0]+=fn[0]; a[1]+=fn[1]; a[2]+=fn[2]; }
    }
    for (auto& kv : vn) kv.second = norm(kv.second);

    V3 lo{1e9f,1e9f,1e9f}, hi{-1e9f,-1e9f,-1e9f};
    for (const auto& v : m.verts())
        for (int k = 0; k < 3; ++k) { float c = (float)v->co[k]; lo[k] = std::min(lo[k], c); hi[k] = std::max(hi[k], c); }
    Gpu g;
    if (m.num_verts() == 0) return g;
    g.center = {(lo[0]+hi[0])*0.5f, (lo[1]+hi[1])*0.5f, (lo[2]+hi[2])*0.5f};
    g.radius = 0.5f * std::sqrt((hi[0]-lo[0])*(hi[0]-lo[0]) + (hi[1]-lo[1])*(hi[1]-lo[1]) + (hi[2]-lo[2])*(hi[2]-lo[2]));
    if (g.radius < 1e-3f) g.radius = 1;
    for (const auto& f : m.faces()) {
        auto vs = m.face_verts(f.get());
        const Material& fm = f->material;  // bake the face's material into its verts (loc 2,3)
        const float ar = fm.set ? (float)fm.color[0] : -1.0f;  // r<0 -> "no material, use the slider"
        const float ag = (float)fm.color[1], ab = (float)fm.color[2];
        const float met = (float)fm.metallic, rgh = (float)fm.roughness;
        for (size_t i = 1; i + 1 < vs.size(); ++i) {
            Vert* tri[3] = {vs[0], vs[i], vs[i + 1]};
            for (Vert* v : tri) {
                V3 n = vn[v];
                g.data.insert(g.data.end(), {(float)v->co[0], (float)v->co[1], (float)v->co[2], n[0], n[1], n[2],
                                             ar, ag, ab, met, rgh});
                g.verts++;
            }
        }
    }
    return g;
}

// --- GL helpers ------------------------------------------------------------
static GLuint compile(GLenum type, const char* src) {
    GLuint s = glCreateShader(type);
    glShaderSource(s, 1, &src, nullptr);
    glCompileShader(s);
    GLint ok = 0; glGetShaderiv(s, GL_COMPILE_STATUS, &ok);
    if (!ok) { char log[1024]; glGetShaderInfoLog(s, 1024, nullptr, log); std::fprintf(stderr, "shader: %s\n", log); }
    return s;
}
static GLuint make_program(const char* vs, const char* fs) {
    GLuint p = glCreateProgram();
    GLuint v = compile(GL_VERTEX_SHADER, vs), f = compile(GL_FRAGMENT_SHADER, fs);
    glAttachShader(p, v); glAttachShader(p, f); glLinkProgram(p);
    GLint ok = 0; glGetProgramiv(p, GL_LINK_STATUS, &ok);
    if (!ok) { char log[1024]; glGetProgramInfoLog(p, 1024, nullptr, log); std::fprintf(stderr, "link: %s\n", log); }
    glDeleteShader(v); glDeleteShader(f);
    return p;
}
// The key light also casts the shadow — keep this in sync with the C++ light rig.
#define KEY_LIGHT "normalize(vec3(0.4,0.5,0.8))"

static const char* VERT = R"(#version 330 core
layout(location=0) in vec3 aPos;
layout(location=1) in vec3 aNormal;
layout(location=2) in vec3 aAlbedo;     // per-face material (r<0 -> use the slider uniforms)
layout(location=3) in vec2 aMetRough;
uniform mat4 uMVP;
uniform mat4 uLightVP;
out vec3 vN;
out vec3 vWorld;
out vec4 vLightPos;
out vec3 vMatAlbedo;
out vec2 vMatMR;
void main(){ vWorld=aPos; vN=aNormal; vMatAlbedo=aAlbedo; vMatMR=aMetRough;
  vLightPos=uLightVP*vec4(aPos,1.0); gl_Position=uMVP*vec4(aPos,1.0); }
)";
// Physically-based viewport: Cook-Torrance microfacet specular (GGX/Trowbridge-
// Reitz NDF, Smith height-correlated geometry, Schlick Fresnel) + Lambert
// diffuse, lit by a studio 3-point rig and a hemispherical ambient, with a
// shadow-mapped key light, then ACES tonemapped. uFlat reconstructs per-face
// normals from screen-space derivatives (true faceting, no geometry change).
static const char* FRAG = R"(#version 330 core
in vec3 vN; in vec3 vWorld; in vec4 vLightPos;
in vec3 vMatAlbedo; in vec2 vMatMR;
out vec4 frag;
uniform vec3 uEye;
uniform vec3 uAlbedo;
uniform float uMetallic;
uniform float uRough;
uniform int uFlat;
uniform int uHighlight;
uniform sampler2D uShadow;
const float PI = 3.14159265359;

float D_GGX(float NoH, float a){ float a2=a*a; float d=NoH*NoH*(a2-1.0)+1.0; return a2/(PI*d*d); }
float G_Smith(float NoV, float NoL, float a){ float k=a*0.5;
  float gv=NoV/(NoV*(1.0-k)+k); float gl=NoL/(NoL*(1.0-k)+k); return gv*gl; }
vec3 F_Schlick(float VoH, vec3 f0){ return f0 + (1.0-f0)*pow(1.0-VoH,5.0); }

vec3 brdf(vec3 N, vec3 V, vec3 L, vec3 radiance, vec3 albedo, float metallic, float rough){
  vec3 H = normalize(V+L);
  float NoL = max(dot(N,L),0.0);
  float NoV = max(dot(N,V),1e-4);
  float NoH = max(dot(N,H),0.0);
  float VoH = max(dot(V,H),0.0);
  float a = max(rough*rough,1e-3);
  vec3 f0 = mix(vec3(0.04), albedo, metallic);
  float D = D_GGX(NoH,a);
  float G = G_Smith(NoV,NoL,a);
  vec3 F = F_Schlick(VoH,f0);
  vec3 spec = (D*G*F)/max(4.0*NoV*NoL,1e-4);
  vec3 kd = (vec3(1.0)-F)*(1.0-metallic);
  return (kd*albedo/PI + spec)*radiance*NoL;
}
// 3x3 PCF shadow visibility (1 = lit, 0 = fully shadowed).
float shadow_vis(vec4 lp, float NoL){
  vec3 p = lp.xyz/lp.w * 0.5 + 0.5;
  if(p.z > 1.0) return 1.0;
  float bias = max(0.003*(1.0-NoL), 0.0010);
  vec2 tx = 1.0/vec2(textureSize(uShadow,0));
  float s = 0.0;
  for(int x=-1;x<=1;x++) for(int y=-1;y<=1;y++)
    s += (p.z - bias > texture(uShadow, p.xy + vec2(x,y)*tx).r) ? 0.0 : 1.0;
  return s/9.0;
}
vec3 aces(vec3 x){ return clamp((x*(2.51*x+0.03))/(x*(2.43*x+0.59)+0.14),0.0,1.0); }

void main(){
  if(uHighlight==1){ frag = vec4(pow(vec3(1.0,0.55,0.12),vec3(0.4545)),1.0); return; }
  if(uHighlight==2){ frag = vec4(pow(vec3(0.60,0.80,0.86),vec3(0.4545)),1.0); return; }  // wireframe lines
  vec3 N = (uFlat==1) ? normalize(cross(dFdx(vWorld), dFdy(vWorld))) : normalize(vN);
  vec3 V = normalize(uEye - vWorld);
  if(dot(N,V) < 0.0) N = -N;                       // two-sided shading
  // per-face material if assigned (albedo.r >= 0), else the global slider
  bool hasMat = vMatAlbedo.r >= 0.0;
  vec3 albedo   = hasMat ? vMatAlbedo : uAlbedo;
  float metallic= hasMat ? vMatMR.x   : uMetallic;
  float rough   = hasMat ? vMatMR.y   : uRough;
  vec3 key = )" KEY_LIGHT R"(;
  float vis = shadow_vis(vLightPos, max(dot(N,key),0.0));
  vec3 col = vec3(0.0);
  col += vis * brdf(N,V, key,                       vec3(3.0,2.9,2.7), albedo,metallic,rough); // key (shadowed)
  col += brdf(N,V, normalize(vec3(-0.6, 0.2, 0.3)), vec3(0.5,0.6,0.8), albedo,metallic,rough); // fill (cool)
  col += brdf(N,V, normalize(vec3( 0.1,-0.7, 0.4)), vec3(0.5,0.45,0.4), albedo,metallic,rough); // rim (warm)
  float hemi = 0.5+0.5*N.z;                          // sky/ground hemispherical ambient
  col += mix(vec3(0.10,0.10,0.12), vec3(0.32,0.35,0.40), hemi) * albedo * (1.0-metallic*0.7);
  frag = vec4(pow(aces(col), vec3(0.4545)), 1.0);
}
)";

// Shadow caster — depth-only render from the light's point of view.
static const char* DEPTH_VERT = R"(#version 330 core
layout(location=0) in vec3 aPos;
uniform mat4 uLightVP;
void main(){ gl_Position = uLightVP * vec4(aPos,1.0); }
)";
static const char* DEPTH_FRAG = R"(#version 330 core
void main(){}
)";

// Studio floor — an anti-aliased world grid that receives the key-light shadow
// and fades to the background at the rim (so the model sits in a scene, not void).
static const char* GRID_VERT = R"(#version 330 core
layout(location=0) in vec3 aPos;
uniform mat4 uMVP;
uniform mat4 uLightVP;
out vec3 vWorld;
out vec4 vLightPos;
void main(){ vWorld=aPos; vLightPos=uLightVP*vec4(aPos,1.0); gl_Position=uMVP*vec4(aPos,1.0); }
)";
static const char* GRID_FRAG = R"(#version 330 core
in vec3 vWorld; in vec4 vLightPos;
out vec4 frag;
uniform sampler2D uShadow;
uniform vec2 uCenter;
uniform float uFade;
float shadow_vis(vec4 lp){
  vec3 p = lp.xyz/lp.w * 0.5 + 0.5;
  if(p.z > 1.0) return 1.0;
  vec2 tx = 1.0/vec2(textureSize(uShadow,0));
  float s = 0.0;
  for(int x=-1;x<=1;x++) for(int y=-1;y<=1;y++)
    s += (p.z - 0.0015 > texture(uShadow, p.xy + vec2(x,y)*tx).r) ? 0.0 : 1.0;
  return s/9.0;
}
float gridline(vec2 p, float scale){
  vec2 c = abs(fract(p/scale - 0.5) - 0.5) / fwidth(p/scale);
  return 1.0 - min(min(c.x,c.y),1.0);
}
void main(){
  float vis = shadow_vis(vLightPos);
  vec3 base = vec3(0.165,0.175,0.205);
  float major = gridline(vWorld.xy, 1.0);
  float minor = gridline(vWorld.xy, 0.25)*0.35;
  vec3 col = mix(base, vec3(0.34,0.36,0.42), max(major,minor));
  if(abs(vWorld.y) < 0.012) col = vec3(0.55,0.27,0.27);  // X axis (red)
  if(abs(vWorld.x) < 0.012) col = vec3(0.30,0.52,0.32);  // Y axis (green)
  col *= (0.32 + 0.68*vis);                              // the model's shadow on the floor
  float a = clamp(1.0 - length(vWorld.xy - uCenter)/uFade, 0.0, 1.0);
  frag = vec4(pow(col, vec3(0.4545)), a*a*0.97);
}
)";

// Studio backdrop: a soft radial gradient (a lit cyclorama) filling the viewport
// behind everything, so the frame reads as a studio rather than a flat grey fill.
// Drawn as a fullscreen triangle (gl_VertexID, no vertex buffer), depth write off.
static const char* GRAD_VERT = R"(#version 330 core
void main(){ vec2 p = vec2((gl_VertexID<<1)&2, gl_VertexID&2); gl_Position = vec4(p*2.0-1.0, 0.0, 1.0); }
)";
static const char* GRAD_FRAG = R"(#version 330 core
out vec4 frag;
uniform vec2 uRes;
void main(){
  vec2 uv = gl_FragCoord.xy / uRes;
  vec2 d = uv - vec2(0.5, 0.62);          // light pool sits a little above centre
  d.x *= uRes.x / uRes.y;                  // aspect-correct so it stays circular
  float r = length(d);
  float t = smoothstep(0.05, 0.98, r);
  vec3 core = vec3(0.170, 0.188, 0.223);   // lit centre
  vec3 edge = vec3(0.064, 0.072, 0.088);   // shaded rim
  frag = vec4(mix(core, edge, t), 1.0);
}
)";

// --- glTF (.glb) import: parse baked geometry into one replayable `mesh` op ----
// The native parity for the AI's import_gltf MCP tool: a human at the GUI can load
// a .glb directly. We weld the triangle soup back into shared-vertex topology and
// invert the Y-up axis swap, lowering to the same `mesh` op both engines replay.
static Program program_from_glb(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    if (!f) throw std::runtime_error("cannot open " + path);
    std::vector<unsigned char> blob((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
    auto u32 = [&](std::size_t o) -> std::uint32_t {
        return blob[o] | (blob[o + 1] << 8) | (blob[o + 2] << 16) | (std::uint32_t(blob[o + 3]) << 24);
    };
    if (blob.size() < 20 || u32(0) != 0x46546C67u) throw std::runtime_error("not a .glb file");
    const std::uint32_t jlen = u32(12);
    json g = json::parse(std::string(reinterpret_cast<char*>(&blob[20]), jlen));
    const unsigned char* bin = blob.data() + 20 + jlen + 8;  // skip JSON + BIN chunk headers

    auto comp_size = [](int ct) { return ct == 5120 || ct == 5121 ? 1 : ct == 5122 || ct == 5123 ? 2 : 4; };
    auto acc_base = [&](const json& acc) {
        const json& bv = g["bufferViews"][acc.at("bufferView").get<int>()];
        return std::size_t(bv.value("byteOffset", 0)) + std::size_t(acc.value("byteOffset", 0));
    };
    auto read_index = [&](int ct, const unsigned char* p) -> std::uint32_t {
        if (ct == 5121) return *p;
        if (ct == 5123) { std::uint16_t v; std::memcpy(&v, p, 2); return v; }
        std::uint32_t v; std::memcpy(&v, p, 4); return v;
    };

    std::vector<std::array<double, 3>> verts;
    std::map<std::array<long long, 3>, int> weld;
    json faces = json::array(), fmats = json::array();
    bool any_mat = false;
    auto weld_vertex = [&](double X, double Y, double Z) {  // glTF Y-up -> Mirage Z-up
        std::array<double, 3> p{X, -Z, Y};
        std::array<long long, 3> key{std::llround(p[0] * 1e5), std::llround(p[1] * 1e5), std::llround(p[2] * 1e5)};
        auto it = weld.find(key);
        if (it != weld.end()) return it->second;
        int id = int(verts.size());
        weld[key] = id; verts.push_back(p);
        return id;
    };

    for (const auto& m : g.value("meshes", json::array())) {
        for (const auto& prim : m.value("primitives", json::array())) {
            if (prim.value("mode", 4) != 4) continue;  // TRIANGLES only
            const json& pacc = g["accessors"][prim["attributes"]["POSITION"].get<int>()];
            const std::size_t pbase = acc_base(pacc);
            const int pcount = pacc.at("count").get<int>();
            std::vector<std::array<double, 3>> pos(pcount);
            for (int i = 0; i < pcount; ++i) {
                float xyz[3];
                std::memcpy(xyz, bin + pbase + std::size_t(i) * 12, 12);
                pos[i] = {xyz[0], xyz[1], xyz[2]};
            }
            std::vector<std::uint32_t> idx;
            if (prim.contains("indices")) {
                const json& iacc = g["accessors"][prim["indices"].get<int>()];
                const int ct = iacc.at("componentType").get<int>(), cs = comp_size(ct);
                const std::size_t ibase = acc_base(iacc);
                const int icount = iacc.at("count").get<int>();
                for (int i = 0; i < icount; ++i) idx.push_back(read_index(ct, bin + ibase + std::size_t(i) * cs));
            } else {
                for (int i = 0; i < pcount; ++i) idx.push_back(i);
            }
            json mat;  // pull this primitive's material (if any)
            if (prim.contains("material")) {
                const json& pbr = g["materials"][prim["material"].get<int>()].value("pbrMetallicRoughness", json::object());
                auto col = pbr.value("baseColorFactor", std::vector<double>{0.8, 0.8, 0.8, 1.0});
                mat = json{{"color", {col[0], col[1], col[2]}}, {"metallic", pbr.value("metallicFactor", 1.0)},
                           {"roughness", pbr.value("roughnessFactor", 1.0)}};
                any_mat = true;
            }
            for (std::size_t t = 0; t + 2 < idx.size(); t += 3) {
                int a = weld_vertex(pos[idx[t]][0], pos[idx[t]][1], pos[idx[t]][2]);
                int b = weld_vertex(pos[idx[t + 1]][0], pos[idx[t + 1]][1], pos[idx[t + 1]][2]);
                int c = weld_vertex(pos[idx[t + 2]][0], pos[idx[t + 2]][1], pos[idx[t + 2]][2]);
                if (a == b || b == c || a == c) continue;  // degenerate after welding
                faces.push_back({a, b, c}); fmats.push_back(mat);
            }
        }
    }
    if (faces.empty()) throw std::runtime_error("glTF had no triangle geometry");
    json vj = json::array();
    for (const auto& v : verts) vj.push_back({v[0], v[1], v[2]});
    json op{{"op", "mesh"}, {"verts", vj}, {"faces", faces}};
    if (any_mat) op["face_materials"] = fmats;
    return Program::from_json(json::array({op}).dump());
}

// camera / interaction state
static bool g_imgui = false;
static float g_yaw = 2.3f, g_pitch = 0.35f, g_dist = 3.0f;
static double g_lx = 0, g_ly = 0;
static bool g_drag = false, g_moved = false;
static double g_press_x = 0, g_press_y = 0;
static bool g_pick_request = false; static double g_px = 0, g_py = 0;  // a click to resolve into a face
static V3 g_pan{0, 0, 0};                       // orbit-target pan offset (right-drag)
static bool g_panning = false;
static V3 g_cam_right{1, 0, 0}, g_cam_up{0, 0, 1};  // camera basis, refreshed each frame for panning
// What the next operator targets, as a re-evaluable selector (never a stored
// index — TNP-safe):
//   NONE  -> the top face(s)          sel::normal z+
//   PICK  -> the face nearest a click sel::near(point)
//   STACK -> whatever the last op made sel::last
// A pick promotes to STACK after one op, so a picked region keeps stacking
// (inset -> extrude -> inset ...) on the geometry each step actually produced.
enum SelMode { SEL_NONE, SEL_PICK, SEL_STACK };
static SelMode g_sel_mode = SEL_NONE;
static std::array<double, 3> g_sel{0, 0, 0};  // the picked point (PICK mode)

// shared op-log file (the dual-operator bridge): Save/Load round-trips the same
// JSON the AI (MCP save_mesh_program/load_mesh_program) reads and writes. With
// live sync on, the viewer watches the file's mtime and auto-reloads the AI's
// edits, and auto-saves the human's — real-time co-editing of one op-log.
static char g_oplog_path[256] = "mirage_oplog.json";
static char g_glb_path[256] = "mirage_export.glb";
static char g_io_status[256] = "";
static bool g_live_sync = false;
static double g_last_poll = 0.0;
static long long g_last_mtime = 0;  // mtime we last reloaded/wrote (to ignore our own writes)

// AI "AUTO" mode. When an operator that isn't the human is driving the op-log —
// the AI streaming edits into the shared file under live sync — the wall of tool
// buttons gives way to a slim top-left status HUD, and the viewport shows the
// model building itself. It's the editor saying "hands off, I've got this": the
// panel returns the instant the human clicks (take control) or the AI goes quiet.
static bool g_auto_mode = false;          // currently showing the AUTO HUD (vs the tool panel)
static bool g_auto_force = false;         // --automode: pin the HUD (headless promo / making-of)
static double g_auto_last_edit = 0.0;     // when the last external (AI) edit landed
static double g_auto_suppress_until = 0.0;// after a human takes control, don't re-arm until this time
static char g_auto_msg[192] = "";         // "what the AI is editing" line (op delta / label, or --autocap)

// viewport material (PBR) — a warm off-white dielectric by default; flat shading
// reads more truthfully for hard-surface models.
static float g_albedo[3] = {0.82f, 0.80f, 0.74f};
static float g_metallic = 0.0f, g_rough = 0.45f;
static bool g_flat = true;
static bool g_wire = false;  // wireframe overlay (a viewport shading toggle)

static json current_on() {
    if (g_sel_mode == SEL_PICK) return sel::near(g_sel);
    if (g_sel_mode == SEL_STACK) return sel::last();
    return sel::normal("z", 1.0);  // default: the top face
}

static V3 orbit_eye(V3 c) {
    return {c[0] + g_dist * std::cos(g_pitch) * std::sin(g_yaw),
            c[1] - g_dist * std::cos(g_pitch) * std::cos(g_yaw),
            c[2] + g_dist * std::sin(g_pitch)};
}
static bool ray_tri(V3 o, V3 d, V3 a, V3 b, V3 c, float& t) {  // Moller-Trumbore
    V3 e1 = sub(b, a), e2 = sub(c, a), p = cross(d, e2);
    float det = dot(e1, p);
    if (std::fabs(det) < 1e-8f) return false;
    float inv = 1.0f / det;
    V3 tv = sub(o, a);
    float u = dot(tv, p) * inv; if (u < 0 || u > 1) return false;
    V3 q = cross(tv, e1);
    float v = dot(d, q) * inv; if (v < 0 || u + v > 1) return false;
    t = dot(e2, q) * inv;
    return t > 1e-4f;
}

static bool ui_wants_mouse() { return g_imgui && ImGui::GetIO().WantCaptureMouse; }
static void on_mouse(GLFWwindow* w, int button, int action, int) {
    if (action == GLFW_PRESS) {  // any click is the human taking control back from AUTO
        g_auto_mode = false; g_auto_suppress_until = glfwGetTime() + 3.0;
    }
    if (button == GLFW_MOUSE_BUTTON_RIGHT) {  // right-drag pans the orbit target
        if (action == GLFW_PRESS && !ui_wants_mouse()) {
            g_panning = true; glfwGetCursorPos(w, &g_lx, &g_ly);
        } else if (action == GLFW_RELEASE) g_panning = false;
        return;
    }
    if (button != GLFW_MOUSE_BUTTON_LEFT) return;
    if (action == GLFW_PRESS) {
        if (ui_wants_mouse()) { g_drag = false; return; }
        g_drag = true; g_moved = false;
        glfwGetCursorPos(w, &g_press_x, &g_press_y); g_lx = g_press_x; g_ly = g_press_y;
    } else {  // release: a press+release that didn't drag is a pick
        if (g_drag && !g_moved) { glfwGetCursorPos(w, &g_px, &g_py); g_pick_request = true; }
        g_drag = false;
    }
}
static void on_cursor(GLFWwindow*, double x, double y) {
    if (g_panning) {  // move the orbit target in the camera's screen plane
        const float k = g_dist * 0.0018f;
        for (int i = 0; i < 3; ++i)
            g_pan[i] += -g_cam_right[i] * float(x - g_lx) * k + g_cam_up[i] * float(y - g_ly) * k;
    } else if (g_drag && !ui_wants_mouse()) {
        if (std::fabs(x - g_press_x) + std::fabs(y - g_press_y) > 4.0) g_moved = true;
        if (g_moved) {  // drag past a small threshold -> orbit
            // Drag right -> the model turns to follow the cursor (yaw decreases as
            // x grows). Sign pinned by the headless --drag regression (test_viewer_orbit).
            g_yaw -= float(x - g_lx) * 0.01f; g_pitch += float(y - g_ly) * 0.01f;
            if (g_pitch > 1.5f) g_pitch = 1.5f; if (g_pitch < -1.5f) g_pitch = -1.5f;
        }
    }
    g_lx = x; g_ly = y;
}
static void on_scroll(GLFWwindow*, double, double dy) {
    if (ui_wants_mouse()) return;
    g_dist *= (1.0f - 0.12f * float(dy)); if (g_dist < 0.2f) g_dist = 0.2f;
}

// File modification time as a comparable integer (0 if the file is absent).
static long long file_mtime(const char* path) {
    std::error_code ec;
    auto t = std::filesystem::last_write_time(path, ec);
    return ec ? 0 : static_cast<long long>(t.time_since_epoch().count());
}

static void write_ppm(const std::string& path, int W, int H) {
    std::vector<unsigned char> px(size_t(W) * H * 3);
    glPixelStorei(GL_PACK_ALIGNMENT, 1);  // tight rows — else GL pads to 4 bytes and overruns px
    glReadPixels(0, 0, W, H, GL_RGB, GL_UNSIGNED_BYTE, px.data());
    std::ofstream f(path, std::ios::binary);
    f << "P6\n" << W << " " << H << "\n255\n";
    for (int y = H - 1; y >= 0; --y) f.write(reinterpret_cast<char*>(&px[size_t(y) * W * 3]), W * 3);
}

// ---- Mirage UI theme -------------------------------------------------------
// A deliberate dark palette (graphite + one teal accent, coral for destructive)
// so the panel reads as a product, not a debug overlay. One accent, used for
// active state / selection / the primary action; everything else stays neutral.
namespace ui {
static const ImVec4 bg        = ImVec4(0.086f, 0.094f, 0.110f, 1.00f);  // window
static const ImVec4 console   = ImVec4(0.063f, 0.071f, 0.086f, 1.00f);  // op-log child
static const ImVec4 frame     = ImVec4(0.149f, 0.169f, 0.200f, 1.00f);  // button / input
static const ImVec4 frame_h   = ImVec4(0.192f, 0.216f, 0.259f, 1.00f);
static const ImVec4 frame_a   = ImVec4(0.231f, 0.267f, 0.325f, 1.00f);
static const ImVec4 accent    = ImVec4(0.243f, 0.604f, 0.651f, 1.00f);  // teal
static const ImVec4 accent_br = ImVec4(0.322f, 0.745f, 0.800f, 1.00f);
static const ImVec4 header    = ImVec4(0.498f, 0.769f, 0.808f, 1.00f);  // section titles
static const ImVec4 danger    = ImVec4(0.694f, 0.290f, 0.251f, 1.00f);  // coral
static const ImVec4 danger_h  = ImVec4(0.784f, 0.353f, 0.310f, 1.00f);
static const ImVec4 text      = ImVec4(0.863f, 0.871f, 0.886f, 1.00f);
static const ImVec4 text_dim  = ImVec4(0.459f, 0.490f, 0.533f, 1.00f);
static const ImVec4 border    = ImVec4(0.020f, 0.030f, 0.040f, 0.85f);
static const ImVec4 ok        = ImVec4(0.400f, 0.749f, 0.447f, 1.00f);
static const ImVec4 warn      = ImVec4(0.918f, 0.678f, 0.278f, 1.00f);
}  // namespace ui

static void apply_mirage_style() {
    ImGuiStyle& s = ImGui::GetStyle();
    s.WindowRounding = 8.0f;  s.ChildRounding = 6.0f;  s.FrameRounding = 5.0f;
    s.PopupRounding = 6.0f;   s.GrabRounding = 4.0f;   s.ScrollbarRounding = 6.0f;
    s.WindowBorderSize = 1.0f;  s.FrameBorderSize = 0.0f;  s.ChildBorderSize = 1.0f;
    s.WindowPadding = ImVec2(16, 14);  s.FramePadding = ImVec2(10, 7);
    s.ItemSpacing = ImVec2(8, 8);      s.ItemInnerSpacing = ImVec2(8, 6);
    s.IndentSpacing = 18;  s.ScrollbarSize = 12;  s.GrabMinSize = 10;
    s.WindowMenuButtonPosition = ImGuiDir_None;

    ImVec4* c = s.Colors;
    c[ImGuiCol_Text]                 = ui::text;
    c[ImGuiCol_TextDisabled]         = ui::text_dim;
    c[ImGuiCol_WindowBg]             = ui::bg;
    c[ImGuiCol_ChildBg]              = ui::console;
    c[ImGuiCol_PopupBg]              = ui::bg;
    c[ImGuiCol_Border]               = ui::border;
    c[ImGuiCol_FrameBg]              = ui::frame;
    c[ImGuiCol_FrameBgHovered]       = ui::frame_h;
    c[ImGuiCol_FrameBgActive]        = ui::frame_a;
    c[ImGuiCol_TitleBg]              = ui::bg;
    c[ImGuiCol_TitleBgActive]        = ui::bg;
    c[ImGuiCol_Button]               = ui::frame;
    c[ImGuiCol_ButtonHovered]        = ui::frame_h;
    c[ImGuiCol_ButtonActive]         = ui::frame_a;
    c[ImGuiCol_Header]               = ui::frame_h;
    c[ImGuiCol_HeaderHovered]        = ui::frame_a;
    c[ImGuiCol_HeaderActive]         = ui::accent;
    c[ImGuiCol_CheckMark]            = ui::accent_br;
    c[ImGuiCol_SliderGrab]           = ui::accent;
    c[ImGuiCol_SliderGrabActive]     = ui::accent_br;
    c[ImGuiCol_Separator]            = ui::border;
    c[ImGuiCol_SeparatorHovered]     = ui::accent;
    c[ImGuiCol_ScrollbarBg]          = ImVec4(0, 0, 0, 0.16f);
    c[ImGuiCol_ScrollbarGrab]        = ui::frame_h;
    c[ImGuiCol_ScrollbarGrabHovered] = ui::accent;
    c[ImGuiCol_ScrollbarGrabActive]  = ui::accent_br;
    c[ImGuiCol_ResizeGrip]           = ui::frame_h;
    c[ImGuiCol_ResizeGripHovered]    = ui::accent;
    c[ImGuiCol_TextSelectedBg]       = ImVec4(ui::accent.x, ui::accent.y, ui::accent.z, 0.35f);
}

int main(int argc, char** argv) {
    std::string shot, load_path, glb_path;
    double watch_secs = 0.0;  // --watch N: headless live-sync proof (poll the file for N s)
    // Headless UI-regression hooks. `--cam` pins a known viewpoint; `--drag`
    // synthesises a mouse drag through the REAL on_mouse/on_cursor handlers, so a
    // scripted run exercises the exact input->camera mapping a human triggers and
    // dumps the resulting frame — the GUI's camera controls become testable and
    // observable without a visible window or a human at the mouse.
    bool cam_set = false, drag_set = false;
    int win_w = 0, win_h = 0;  // --winsize override (0 = default), for responsive UI checks
    float panscroll = 0.0f;    // --panscroll N: scroll the tool panel (screenshot lower sections)
    double cam_yaw = 0, cam_pitch = 0, cam_dist = 0;
    char drag_btn = 'L';
    double drag_dx = 0, drag_dy = 0;
    // Headless-animation hooks: a scripted assembly clip feeds a DIFFERENT (growing)
    // op-log per frame, so the model's bbox centre and base move between frames.
    // `--target` pins the orbit centre and `--floorz` pins the studio floor to fixed
    // world points, so neither the framing nor the ground drifts across the clip;
    // `--nohighlight` drops the selection overlay for clean beauty frames.
    bool tgt_set = false, floorz_set = false, nohl = false;
    double tgt_x = 0, tgt_y = 0, tgt_z = 0;
    float floorz_val = 0.0f;
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        if (a == "--screenshot" && i + 1 < argc) shot = argv[++i];
        else if (a == "--oplog" && i + 1 < argc) { load_path = argv[++i]; std::snprintf(g_oplog_path, sizeof(g_oplog_path), "%s", load_path.c_str()); }
        else if (a == "--import-glb" && i + 1 < argc) { glb_path = argv[++i]; std::snprintf(g_glb_path, sizeof(g_glb_path), "%s", glb_path.c_str()); }
        else if (a == "--watch" && i + 1 < argc) watch_secs = std::atof(argv[++i]);
        else if (a == "--cam" && i + 3 < argc) { cam_set = true; cam_yaw = std::atof(argv[++i]); cam_pitch = std::atof(argv[++i]); cam_dist = std::atof(argv[++i]); }
        else if (a == "--drag" && i + 3 < argc) { drag_set = true; drag_btn = argv[++i][0]; drag_dx = std::atof(argv[++i]); drag_dy = std::atof(argv[++i]); }
        else if (a == "--winsize" && i + 2 < argc) { win_w = std::atoi(argv[++i]); win_h = std::atoi(argv[++i]); }
        else if (a == "--panscroll" && i + 1 < argc) { panscroll = float(std::atof(argv[++i])); }
        else if (a == "--target" && i + 3 < argc) { tgt_set = true; tgt_x = std::atof(argv[++i]); tgt_y = std::atof(argv[++i]); tgt_z = std::atof(argv[++i]); }
        else if (a == "--floorz" && i + 1 < argc) { floorz_set = true; floorz_val = float(std::atof(argv[++i])); }
        else if (a == "--nohighlight") nohl = true;
        else if (a == "--automode") g_auto_force = true;  // force the AI "AUTO" HUD (hide the panel)
        else if (a == "--wire") g_wire = true;            // force the wireframe overlay (promo / verification)
        else if (a == "--autocap" && i + 1 < argc) std::snprintf(g_auto_msg, sizeof(g_auto_msg), "%s", argv[++i]);
        else if (a == "--autocap-file" && i + 1 < argc) {  // read the HUD line as UTF-8 from a file
            std::ifstream cf(argv[++i], std::ios::binary);  // (Windows argv is ANSI-mangled; a file is byte-exact)
            if (cf) {
                std::string s((std::istreambuf_iterator<char>(cf)), std::istreambuf_iterator<char>());
                while (!s.empty() && (s.back() == '\n' || s.back() == '\r')) s.pop_back();
                std::snprintf(g_auto_msg, sizeof(g_auto_msg), "%.*s", int(sizeof(g_auto_msg) - 1), s.c_str());
            }
        }
    }

    Program prog;
    bool loaded_ok = false;
    if (!glb_path.empty()) {  // import a .glb straight into a `mesh` op
        try { prog = program_from_glb(glb_path); prog.build(); loaded_ok = true;
              std::snprintf(g_io_status, sizeof(g_io_status), "imported %s -> 1 mesh op", glb_path.c_str()); }
        catch (const std::exception& e) { std::fprintf(stderr, "could not import %s: %s\n", glb_path.c_str(), e.what()); }
    } else if (!load_path.empty()) {  // open straight onto a shared op-log (e.g. one an AI just saved)
        std::ifstream f(load_path);
        if (f) {
            std::string s((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
            try { prog = Program::from_json(s); prog.build(); loaded_ok = true;
                  std::snprintf(g_io_status, sizeof(g_io_status), "loaded %zu ops <- %s", prog.size(), load_path.c_str()); }
            catch (const std::exception& e) { std::fprintf(stderr, "could not load %s: %s\n", load_path.c_str(), e.what()); }
        } else std::fprintf(stderr, "could not open %s\n", load_path.c_str());
    }
    if (!loaded_ok) {
        prog.cube(1.0);
        if (!shot.empty()) {  // a faceted default for the headless image (a boss on top)
            prog.inset(sel::normal("z"), 0.3); prog.extrude(sel::last(), 0.6);
        }
    }

    if (!glfwInit()) { std::fprintf(stderr, "glfwInit failed\n"); return 1; }
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
    if (!shot.empty()) glfwWindowHint(GLFW_VISIBLE, GLFW_FALSE);
    int W = 1100, H = 760;
    if (win_w > 0 && win_h > 0) { W = win_w; H = win_h; }
    GLFWwindow* win = glfwCreateWindow(W, H, "Mirage — native modeling viewport", nullptr, nullptr);
    if (!win) { std::fprintf(stderr, "window/context creation failed\n"); glfwTerminate(); return 1; }
    glfwMakeContextCurrent(win);
    if (!gladLoadGL(reinterpret_cast<GLADloadfunc>(glfwGetProcAddress))) {
        std::fprintf(stderr, "glad load failed\n"); return 1;
    }
    if (shot.empty()) {
        glfwSwapInterval(1);  // vsync — don't spin the GPU at thousands of fps
        glfwSetMouseButtonCallback(win, on_mouse);
        glfwSetCursorPosCallback(win, on_cursor);
        glfwSetScrollCallback(win, on_scroll);
    }
    glEnable(GL_DEPTH_TEST);

    GLuint prog_gl = make_program(VERT, FRAG);
    const GLint locMVP = glGetUniformLocation(prog_gl, "uMVP");
    const GLint locLightVP = glGetUniformLocation(prog_gl, "uLightVP");
    const GLint locEye = glGetUniformLocation(prog_gl, "uEye");
    const GLint locAlbedo = glGetUniformLocation(prog_gl, "uAlbedo");
    const GLint locMetallic = glGetUniformLocation(prog_gl, "uMetallic");
    const GLint locRough = glGetUniformLocation(prog_gl, "uRough");
    const GLint locFlat = glGetUniformLocation(prog_gl, "uFlat");
    const GLint locHighlight = glGetUniformLocation(prog_gl, "uHighlight");
    const GLint locShadow = glGetUniformLocation(prog_gl, "uShadow");

    GLuint depth_gl = make_program(DEPTH_VERT, DEPTH_FRAG);  // shadow caster
    const GLint locDLightVP = glGetUniformLocation(depth_gl, "uLightVP");

    GLuint grid_gl = make_program(GRID_VERT, GRID_FRAG);     // studio floor
    const GLint locGMVP = glGetUniformLocation(grid_gl, "uMVP");
    const GLint locGLightVP = glGetUniformLocation(grid_gl, "uLightVP");
    const GLint locGShadow = glGetUniformLocation(grid_gl, "uShadow");
    const GLint locGCenter = glGetUniformLocation(grid_gl, "uCenter");
    const GLint locGFade = glGetUniformLocation(grid_gl, "uFade");

    GLuint grad_gl = make_program(GRAD_VERT, GRAD_FRAG);     // studio backdrop gradient
    const GLint locGradRes = glGetUniformLocation(grad_gl, "uRes");
    GLuint bgvao; glGenVertexArrays(1, &bgvao);              // attribute-less (fullscreen triangle)

    GLuint vao, vbo;
    glGenVertexArrays(1, &vao); glGenBuffers(1, &vbo);
    glBindVertexArray(vao);
    glBindBuffer(GL_ARRAY_BUFFER, vbo);  // 11 floats: pos3 normal3 albedo3 metallic1 roughness1
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 11 * sizeof(float), (void*)0);
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 11 * sizeof(float), (void*)(3 * sizeof(float)));
    glEnableVertexAttribArray(1);
    glVertexAttribPointer(2, 3, GL_FLOAT, GL_FALSE, 11 * sizeof(float), (void*)(6 * sizeof(float)));  // albedo (r<0 = use slider)
    glEnableVertexAttribArray(2);
    glVertexAttribPointer(3, 2, GL_FLOAT, GL_FALSE, 11 * sizeof(float), (void*)(9 * sizeof(float)));  // metallic, roughness
    glEnableVertexAttribArray(3);

    GLuint hvao, hvbo; int hl_verts = 0;  // selection highlight geometry (pos3 normal3)
    glGenVertexArrays(1, &hvao); glGenBuffers(1, &hvbo);
    glBindVertexArray(hvao);
    glBindBuffer(GL_ARRAY_BUFFER, hvbo);
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 6 * sizeof(float), (void*)0);
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 6 * sizeof(float), (void*)(3 * sizeof(float)));
    glEnableVertexAttribArray(1);

    GLuint gvao, gvbo;  // studio floor quad (position-only)
    glGenVertexArrays(1, &gvao); glGenBuffers(1, &gvbo);
    glBindVertexArray(gvao);
    glBindBuffer(GL_ARRAY_BUFFER, gvbo);
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(float), (void*)0);
    glEnableVertexAttribArray(0);

    // shadow map: a depth texture rendered from the key light's POV
    const int SHADOW_SZ = 2048;
    GLuint depthFBO, depthTex;
    glGenFramebuffers(1, &depthFBO);
    glGenTextures(1, &depthTex);
    glBindTexture(GL_TEXTURE_2D, depthTex);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_DEPTH_COMPONENT24, SHADOW_SZ, SHADOW_SZ, 0,
                 GL_DEPTH_COMPONENT, GL_FLOAT, nullptr);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_BORDER);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_BORDER);
    const float border[4] = {1, 1, 1, 1};  // outside the light frustum = fully lit
    glTexParameterfv(GL_TEXTURE_2D, GL_TEXTURE_BORDER_COLOR, border);
    glBindFramebuffer(GL_FRAMEBUFFER, depthFBO);
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_TEXTURE_2D, depthTex, 0);
    glDrawBuffer(GL_NONE); glReadBuffer(GL_NONE);
    glBindFramebuffer(GL_FRAMEBUFFER, 0);

    std::string last_tag;            // the build's most recent __out tag (for sel::last)
    Mesh model = prog.build(&last_tag);
    Gpu g = build_gpu(model);
    g_dist = g.radius * 3.0f;
    g_auto_mode = g_auto_force;  // headless promo / a forced demo opens straight into AUTO
    auto upload = [&]() {
        glBindBuffer(GL_ARRAY_BUFFER, vbo);
        glBufferData(GL_ARRAY_BUFFER, GLsizeiptr(g.data.size() * sizeof(float)),
                     g.data.empty() ? nullptr : g.data.data(), GL_DYNAMIC_DRAW);
    };
    upload();

    // The studio floor: a large quad just below the model, rebuilt when the model
    // changes so the grid + shadow always sit at the model's base.
    float ground_z = 0.0f, ground_S = 10.0f;
    auto build_ground = [&]() {
        float minz = 1e9f;
        for (const auto& v : model.verts()) minz = std::min(minz, (float)v->co[2]);
        ground_z = floorz_set ? floorz_val
                              : (model.num_verts() ? minz : 0.0f) - 0.002f * g.radius;
        ground_S = std::max(g.radius * 9.0f, 5.0f);
        const float cx = g.center[0], cy = g.center[1], z = ground_z, S = ground_S;
        const float q[18] = {cx-S, cy-S, z,  cx+S, cy-S, z,  cx+S, cy+S, z,
                             cx-S, cy-S, z,  cx+S, cy+S, z,  cx-S, cy+S, z};
        glBindBuffer(GL_ARRAY_BUFFER, gvbo);
        glBufferData(GL_ARRAY_BUFFER, sizeof(q), q, GL_DYNAMIC_DRAW);
    };
    build_ground();

    // Highlight = exactly what the next op will hit: resolve the current selector
    // against the live mesh (so the highlight and the action can never disagree).
    auto rebuild_highlight = [&]() {
        hl_verts = 0;
        if (model.num_faces() == 0) return;
        std::vector<const Face*> tgt;
        try { tgt = resolve(model, current_on(), last_tag); }
        catch (const std::exception&) { return; }  // SelectorEmpty -> nothing to show
        std::vector<float> hd;
        for (const Face* f : tgt) {
            auto fnv = face_normal(model, f);
            V3 n{(float)fnv[0], (float)fnv[1], (float)fnv[2]};
            auto vs = model.face_verts(f);
            for (size_t i = 1; i + 1 < vs.size(); ++i) {
                Vert* tri[3] = {vs[0], vs[i], vs[i + 1]};
                for (Vert* v : tri) {
                    hd.insert(hd.end(), {(float)v->co[0], (float)v->co[1], (float)v->co[2], n[0], n[1], n[2]});
                    hl_verts++;
                }
            }
        }
        glBindBuffer(GL_ARRAY_BUFFER, hvbo);
        glBufferData(GL_ARRAY_BUFFER, GLsizeiptr(hd.size() * sizeof(float)),
                     hd.empty() ? nullptr : hd.data(), GL_DYNAMIC_DRAW);
    };

    // cast a ray through the cursor and select the nearest hit face
    auto do_pick = [&](double px, double py) {
        // cursor is in window (screen) coords; aspect is from the framebuffer —
        // these differ on HiDPI, so normalize each by its own size.
        int ww, wh; glfwGetWindowSize(win, &ww, &wh);
        int fw, fh; glfwGetFramebufferSize(win, &fw, &fh);
        float ndcx = 2.0f * float(px) / float(ww ? ww : 1) - 1.0f;
        float ndcy = 1.0f - 2.0f * float(py) / float(wh ? wh : 1);
        V3 c = {g.center[0] + g_pan[0], g.center[1] + g_pan[1], g.center[2] + g_pan[2]};
        V3 eye = orbit_eye(c);
        V3 fwd = norm(sub(c, eye)), s = norm(cross(fwd, {0, 0, 1})), u = cross(s, fwd);
        float fovy = 0.9f, asp = float(fw) / float(fh ? fh : 1), tt = std::tan(fovy * 0.5f);
        V3 dir = norm({fwd[0] + ndcx*tt*asp*s[0] + ndcy*tt*u[0],
                       fwd[1] + ndcx*tt*asp*s[1] + ndcy*tt*u[1],
                       fwd[2] + ndcx*tt*asp*s[2] + ndcy*tt*u[2]});
        float best = 1e30f; const Face* hit = nullptr;
        for (const auto& fc : model.faces()) {
            auto vs = model.face_verts(fc.get());
            for (size_t i = 1; i + 1 < vs.size(); ++i) {
                V3 a{(float)vs[0]->co[0], (float)vs[0]->co[1], (float)vs[0]->co[2]};
                V3 b{(float)vs[i]->co[0], (float)vs[i]->co[1], (float)vs[i]->co[2]};
                V3 cc{(float)vs[i+1]->co[0], (float)vs[i+1]->co[1], (float)vs[i+1]->co[2]};
                float t;
                if (ray_tri(eye, dir, a, b, cc, t) && t < best) { best = t; hit = fc.get(); }
            }
        }
        if (hit) { g_sel_mode = SEL_PICK; g_sel = face_centroid(model, hit); rebuild_highlight(); }
    };

    // Load the shared op-log file and rebuild everything (validate before adopting).
    // Used by the Load button AND the live-sync poll, so the two can't diverge.
    auto reload_oplog = [&]() -> bool {
        std::ifstream f(g_oplog_path);
        if (!f) { std::snprintf(g_io_status, sizeof(g_io_status), "load failed: %s", g_oplog_path); return false; }
        std::string s((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
        try {
            Program loaded = Program::from_json(s);
            loaded.build();  // validate before adopting
            prog = std::move(loaded);
            g_sel_mode = SEL_NONE;
            model = prog.build(&last_tag); g = build_gpu(model); upload(); build_ground(); rebuild_highlight();
            g_last_mtime = file_mtime(g_oplog_path);
            std::snprintf(g_io_status, sizeof(g_io_status), "loaded %zu ops <- %s", prog.size(), g_oplog_path);
            return true;
        } catch (const std::exception& e) {
            std::snprintf(g_io_status, sizeof(g_io_status), "bad op-log: %.180s", e.what());
            return false;
        }
    };
    auto save_oplog = [&]() {
        std::ofstream f(g_oplog_path);
        if (!f) { std::snprintf(g_io_status, sizeof(g_io_status), "save failed: %s", g_oplog_path); return; }
        f << prog.to_json(2); f.close();
        g_last_mtime = file_mtime(g_oplog_path);  // remember our own write so the poll won't echo it
        std::snprintf(g_io_status, sizeof(g_io_status), "saved %zu ops -> %s", prog.size(), g_oplog_path);
    };

    auto draw = [&]() {
        // The key light's view-projection (must agree with KEY_LIGHT in the shader):
        // an orthographic camera looking down the key direction, fitted to the model.
        V3 Ld = norm({0.4f, 0.5f, 0.8f});
        const float R = g.radius * 2.4f + 0.3f;
        V3 lc = {g.center[0], g.center[1], ground_z + g.radius * 0.5f};
        V3 le = {lc[0] + Ld[0] * R * 3, lc[1] + Ld[1] * R * 3, lc[2] + Ld[2] * R * 3};
        Mat4 lightVP = mul(ortho(-R, R, -R, R, 0.05f, R * 8), look_at(le, lc, {0, 0, 1}));

        // --- shadow pass: render the model's depth from the light ---
        if (g.verts > 0) {
            glViewport(0, 0, SHADOW_SZ, SHADOW_SZ);
            glBindFramebuffer(GL_FRAMEBUFFER, depthFBO);
            glClear(GL_DEPTH_BUFFER_BIT);
            glUseProgram(depth_gl);
            glUniformMatrix4fv(locDLightVP, 1, GL_FALSE, lightVP.data());
            glEnable(GL_POLYGON_OFFSET_FILL); glPolygonOffset(2.5f, 4.0f);  // tame shadow acne
            glBindVertexArray(vao);
            glDrawArrays(GL_TRIANGLES, 0, g.verts);
            glDisable(GL_POLYGON_OFFSET_FILL);
            glBindFramebuffer(GL_FRAMEBUFFER, 0);
        }

        // --- main pass ---
        int fw, fh; glfwGetFramebufferSize(win, &fw, &fh);
        glViewport(0, 0, fw, fh);
        glClearColor(0.13f, 0.14f, 0.16f, 1.0f);  // fallback fill (overdrawn by the backdrop)
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
        // studio backdrop: fill the frame with the radial gradient before anything
        // else (depth write off so it never occludes the scene).
        glDepthMask(GL_FALSE); glDisable(GL_DEPTH_TEST);
        glUseProgram(grad_gl);
        glUniform2f(locGradRes, float(fw), float(fh));
        glBindVertexArray(bgvao);
        glDrawArrays(GL_TRIANGLES, 0, 3);
        glEnable(GL_DEPTH_TEST); glDepthMask(GL_TRUE);
        V3 c = {g.center[0] + g_pan[0], g.center[1] + g_pan[1], g.center[2] + g_pan[2]};
        V3 eye = orbit_eye(c);
        Mat4 mvp = mul(perspective(0.9f, float(fw) / float(fh ? fh : 1), 0.05f, 100.0f),
                       look_at(eye, c, {0, 0, 1}));
        V3 fwd = norm(sub(c, eye));                 // refresh the camera basis (for panning)
        g_cam_right = norm(cross(fwd, {0, 0, 1}));
        g_cam_up = cross(g_cam_right, fwd);
        glActiveTexture(GL_TEXTURE0);
        glBindTexture(GL_TEXTURE_2D, depthTex);

        // studio floor (grid + received shadow), alpha-blended to fade at the rim
        glUseProgram(grid_gl);
        glUniformMatrix4fv(locGMVP, 1, GL_FALSE, mvp.data());
        glUniformMatrix4fv(locGLightVP, 1, GL_FALSE, lightVP.data());
        glUniform1i(locGShadow, 0);
        glUniform2f(locGCenter, c[0], c[1]);
        glUniform1f(locGFade, ground_S);
        glEnable(GL_BLEND); glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
        glBindVertexArray(gvao);
        glDrawArrays(GL_TRIANGLES, 0, 6);
        glDisable(GL_BLEND);

        if (g.verts == 0) return;
        // the model
        glUseProgram(prog_gl);
        glUniformMatrix4fv(locMVP, 1, GL_FALSE, mvp.data());
        glUniformMatrix4fv(locLightVP, 1, GL_FALSE, lightVP.data());
        glUniform3f(locEye, eye[0], eye[1], eye[2]);
        glUniform3f(locAlbedo, g_albedo[0], g_albedo[1], g_albedo[2]);
        glUniform1f(locMetallic, g_metallic);
        glUniform1f(locRough, g_rough);
        glUniform1i(locFlat, g_flat ? 1 : 0);
        glUniform1i(locHighlight, 0);
        glUniform1i(locShadow, 0);
        glBindVertexArray(vao);
        glDrawArrays(GL_TRIANGLES, 0, g.verts);
        if (hl_verts > 0) {  // selected face, pulled slightly forward to avoid z-fighting
            glEnable(GL_POLYGON_OFFSET_FILL); glPolygonOffset(-1.0f, -1.0f);
            glUniform1i(locHighlight, 1);  // flat orange overlay
            glBindVertexArray(hvao);
            glDrawArrays(GL_TRIANGLES, 0, hl_verts);
            glDisable(GL_POLYGON_OFFSET_FILL);
        }
        if (g_wire) {  // wireframe overlay: the mesh edges as flat teal lines over the solid
            glUseProgram(prog_gl);
            glUniform1i(locHighlight, 2);
            glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);
            glEnable(GL_POLYGON_OFFSET_LINE); glPolygonOffset(-1.0f, -1.0f);
            glBindVertexArray(vao);
            glDrawArrays(GL_TRIANGLES, 0, g.verts);
            glDisable(GL_POLYGON_OFFSET_LINE);
            glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
            glUniform1i(locHighlight, 0);
        }
    };

    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    ImGui_ImplGlfw_InitForOpenGL(win, shot.empty());  // install input callbacks only when interactive
    ImGui_ImplOpenGL3_Init("#version 330");
    g_imgui = shot.empty();

    // Typography: a real native font (Segoe UI) with a Semibold face for titles
    // and a monospace face (Consolas) for the op-log. The default ImGui bitmap
    // font reads as a debug overlay; a proper font is the single biggest step
    // toward product feel. Falls back to the built-in font if the files are absent.
    ImGuiIO& io = ImGui::GetIO();
    if (!shot.empty()) io.IniFilename = nullptr;  // headless: no persisted layout -> deterministic frames
    ImFont* font_body = nullptr, *font_h = nullptr, *font_mono = nullptr, *font_title = nullptr;
    {
        ImFontConfig cfg; cfg.OversampleH = 3; cfg.OversampleV = 2;
        // Glyph atlas = Latin-1 default plus the few punctuation/shape glyphs the UI
        // uses beyond it (em-dash, filled/empty circles) — else they render as tofu.
        static ImVector<ImWchar> ranges;
        ImFontGlyphRangesBuilder gb;
        gb.AddRanges(io.Fonts->GetGlyphRangesDefault());
        gb.AddText("\xe2\x80\x94\xe2\x97\x8f\xe2\x97\x8b\xe2\x96\xb8");  // U+2014 —, U+25CF ●, U+25CB ○, U+25B8 ▸
        gb.BuildRanges(&ranges);
        const ImWchar* gr = ranges.Data;
        const char* body_ttf = "C:\\Windows\\Fonts\\segoeui.ttf";
        const char* semi_ttf = "C:\\Windows\\Fonts\\seguisb.ttf";
        const char* mono_ttf = "C:\\Windows\\Fonts\\consola.ttf";
        // Microsoft YaHei carries the CJK glyphs Segoe UI lacks; merge it in so the AUTO
        // HUD / captions can read Chinese (e.g. "AI 正在编辑:沙发") while Latin stays Segoe UI.
        const char* cjk_ttf = "C:\\Windows\\Fonts\\msyh.ttc";
        const bool has_cjk = std::filesystem::exists(cjk_ttf);
        const ImWchar* cjk_gr = io.Fonts->GetGlyphRangesChineseSimplifiedCommon();
        ImFontConfig mcfg; mcfg.MergeMode = true; mcfg.OversampleH = 2; mcfg.OversampleV = 1;
        auto add_font = [&](const char* ttf, float sz, bool cjk) -> ImFont* {
            if (!std::filesystem::exists(ttf)) return nullptr;
            ImFont* f = io.Fonts->AddFontFromFileTTF(ttf, sz, &cfg, gr);
            if (cjk && has_cjk) io.Fonts->AddFontFromFileTTF(cjk_ttf, sz, &mcfg, cjk_gr);  // merge CJK into f
            return f;
        };
        font_body  = add_font(body_ttf, 17.5f, true);
        font_title = add_font(semi_ttf, 22.0f, false);   // "AUTO" / "MIRAGE" — Latin only
        font_h     = add_font(semi_ttf, 15.5f, true);
        font_mono  = add_font(mono_ttf, 15.0f, true);
        if (!font_body)  font_body  = io.Fonts->AddFontDefault();
        if (!font_title) font_title = font_body;
        if (!font_h)     font_h     = font_body;
        if (!font_mono)  font_mono  = font_body;
        io.FontDefault = font_body;
    }
    apply_mirage_style();

    auto panel = [&]() -> bool {  // the tool panel; returns true if the op-log changed
        bool dirty = false;
        // -- layout helpers: an equal-width button grid + styled section headers --
        const ImGuiStyle& st = ImGui::GetStyle();
        int col = 0, percol = 1; float bwidth = 0.0f;
        auto newrow = [&](int n) {  // begin a row of n equal columns (flush to width)
            col = 0; percol = n;
            bwidth = (ImGui::GetContentRegionAvail().x - st.ItemSpacing.x * (n - 1)) / n;
        };
        auto cell = [&]() { if (col % percol) ImGui::SameLine(); ++col; };
        auto B = [&](const char* label) { cell(); return ImGui::Button(label, ImVec2(bwidth, 0)); };
        auto DANGER = [&](const char* label, float w) {  // coral destructive button
            ImGui::PushStyleColor(ImGuiCol_Button, ui::danger);
            ImGui::PushStyleColor(ImGuiCol_ButtonHovered, ui::danger_h);
            ImGui::PushStyleColor(ImGuiCol_ButtonActive, ui::danger_h);
            bool c = ImGui::Button(label, ImVec2(w, 0));
            ImGui::PopStyleColor(3);
            return c;
        };
        auto section = [&](const char* label) {  // teal uppercase section title
            ImGui::Dummy(ImVec2(0, 3));
            ImGui::PushFont(font_h);
            ImGui::PushStyleColor(ImGuiCol_Text, ui::header);
            ImGui::TextUnformatted(label);
            ImGui::PopStyleColor();
            ImGui::PopFont();
            ImGui::Spacing();
        };

        ImGui::SetNextWindowPos(ImVec2(16, 16), ImGuiCond_FirstUseEver);
        ImGui::SetNextWindowSize(ImVec2(376, float(H) - 32), ImGuiCond_FirstUseEver);
        ImGui::Begin("Mirage", nullptr, ImGuiWindowFlags_NoTitleBar | ImGuiWindowFlags_NoCollapse);
        if (panscroll > 0.0f) ImGui::SetScrollY(panscroll);  // headless: reveal lower sections for a shot

        // -- title ---------------------------------------------------------------
        ImGui::PushFont(font_title);
        ImGui::PushStyleColor(ImGuiCol_Text, ui::accent_br);
        ImGui::TextUnformatted("MIRAGE");
        ImGui::PopStyleColor();
        ImGui::PopFont();
        ImGui::PushStyleColor(ImGuiCol_Text, ui::text_dim);
        ImGui::TextUnformatted("native modeling  \xc2\xb7  the op-log is the model");
        ImGui::PopStyleColor();
        ImGui::Spacing();
        ImGui::Separator();

        // -- primitives ----------------------------------------------------------
        section("PRIMITIVES");
        newrow(3);
        if (B("Cube"))     { prog.clear(); prog.cube(1.0); g_sel_mode = SEL_NONE; dirty = true; }
        if (B("Cylinder")) { prog.clear(); prog.cylinder(24, 0.5, 1.0); g_sel_mode = SEL_NONE; dirty = true; }
        if (B("Sphere"))   { prog.clear(); prog.uv_sphere(24, 16, 0.6); g_sel_mode = SEL_NONE; dirty = true; }
        if (B("Cone"))     { prog.clear(); prog.cone(24, 0.5, 1.0); g_sel_mode = SEL_NONE; dirty = true; }
        if (B("Torus"))    { prog.clear(); prog.torus(24, 12, 0.6, 0.22); g_sel_mode = SEL_NONE; dirty = true; }
        if (B("Plane"))    { prog.clear(); prog.plane(1.0); g_sel_mode = SEL_NONE; dirty = true; }
        if (B("Grid"))     { prog.clear(); prog.grid(1.0, 1.0, 10, 10); g_sel_mode = SEL_NONE; dirty = true; }

        // -- face tools (act on the highlighted target) --------------------------
        section("FACE TOOLS");
        ImGui::PushStyleColor(ImGuiCol_Text, ui::text_dim);
        ImGui::TextUnformatted("act on the orange selection; a pick then stacks");
        ImGui::PopStyleColor();
        ImGui::Spacing();
        newrow(3);
        if (B("Inset"))     { prog.inset(current_on(), 0.3);   if (g_sel_mode == SEL_PICK) g_sel_mode = SEL_STACK; dirty = true; }
        if (B("Extrude"))   { prog.extrude(current_on(), 0.5); if (g_sel_mode == SEL_PICK) g_sel_mode = SEL_STACK; dirty = true; }
        if (B("Bevel"))     { prog.bevel(current_on(), 0.2, 0.15); if (g_sel_mode == SEL_PICK) g_sel_mode = SEL_STACK; dirty = true; }
        if (B("Loop Cut"))  { prog.loop_cut(current_on(), "z"); g_sel_mode = SEL_NONE; dirty = true; }
        if (B("Subdivide")) { prog.subdivide(1); dirty = true; }
        if (B("Fill"))      { prog.fill(); g_sel_mode = SEL_NONE; dirty = true; }  // cap holes
        ImGui::Spacing();
        if (DANGER("Delete selected faces", -1)) { prog.del(current_on()); g_sel_mode = SEL_NONE; dirty = true; }

        // -- form: revolve / sweep / boolean -------------------------------------
        section("FORM");
        newrow(2);
        if (B("Spin \xc2\xb7 lathe"))   { prog.spin("z", 32, 360.0); g_sel_mode = SEL_NONE; dirty = true; }
        if (B("Screw \xc2\xb7 helix"))  { prog.screw("z", 24, 3, 0.4, 360.0); g_sel_mode = SEL_NONE; dirty = true; }
        if (B("Vase \xc2\xb7 profile")) {  // an open profile curve spun into a single-walled vase
            prog.profile({{0.05, -0.5}, {0.42, -0.42}, {0.30, -0.05}, {0.46, 0.30}, {0.34, 0.5}}, "xz", false);
            prog.spin("z", 48, 360.0);
            g_sel_mode = SEL_NONE; dirty = true;
        }
        if (B("Drill \xc2\xb7 boolean")) {  // real BSP boolean: subtract a cylinder bore
            mirage::Mesh bit = mirage::make_cylinder(24, 0.25, 3.0);
            std::vector<std::array<double, 3>> bv;
            for (const auto& v : bit.verts()) bv.push_back(v->co);
            std::vector<std::vector<int>> bf;
            for (const auto& f : bit.faces()) {
                std::vector<int> fi;
                for (Vert* vv : bit.face_verts(f.get())) fi.push_back(vv->id);
                bf.push_back(std::move(fi));
            }
            prog.boolean_op("difference", bv, bf);
            g_sel_mode = SEL_NONE; dirty = true;
        }

        // -- whole-mesh operators ------------------------------------------------
        section("MESH");
        newrow(3);
        if (B("Solidify"))   { prog.solidify(0.1); g_sel_mode = SEL_NONE; dirty = true; }      // shell open surfaces
        if (B("Mirror X"))   { prog.mirror("x"); g_sel_mode = SEL_NONE; dirty = true; }        // reflect + weld seam
        if (B("Array x3"))   { prog.array(3, {1.2, 0.0, 0.0}); g_sel_mode = SEL_NONE; dirty = true; }
        if (B("Bisect Z"))   { prog.bisect({0, 0, 0}, {0, 0, 1}, true); g_sel_mode = SEL_NONE; dirty = true; }
        if (B("Edge Bevel")) { prog.edge_bevel(json{{"by", "sharp"}, {"angle", 30}}, 0.12); g_sel_mode = SEL_NONE; dirty = true; }

        // -- history -------------------------------------------------------------
        section("HISTORY");
        newrow(4);
        if (B("Undo")) { prog.undo(); dirty = true; }
        const bool redoable = prog.can_redo();
        if (!redoable) ImGui::BeginDisabled();
        if (B("Redo")) { prog.redo(); dirty = true; }
        if (!redoable) ImGui::EndDisabled();
        cell();
        if (DANGER("Reset", bwidth)) { prog.clear(); prog.cube(1.0); g_sel_mode = SEL_NONE; dirty = true; }
        if (B("Frame")) { g_yaw = 2.3f; g_pitch = 0.35f; g_dist = g.radius * 3.0f; g_pan = {0, 0, 0}; }  // reset the view
        ImGui::Spacing();
        // -- shared op-log file (the human/AI bridge) ----------------------------
        // Save writes the JSON an AI (MCP) can Load, and vice-versa — one model, a
        // human and an AI both editing it. Live sync makes it continuous.
        section("OP-LOG FILE");
        ImGui::PushStyleColor(ImGuiCol_Text, ui::text_dim);
        ImGui::TextUnformatted("the same JSON an AI reads and writes");
        ImGui::PopStyleColor();
        ImGui::Spacing();
        ImGui::SetNextItemWidth(-1);
        ImGui::InputText("##path", g_oplog_path, sizeof(g_oplog_path));
        newrow(2);
        if (B("Save")) save_oplog();
        if (B("Load")) reload_oplog();  // rebuilds itself; no dirty needed
        if (ImGui::Checkbox("live sync (co-edit with the AI)", &g_live_sync)) {
            g_last_mtime = file_mtime(g_oplog_path);  // baseline: watch from now (don't clobber on enable)
            std::snprintf(g_io_status, sizeof(g_io_status), g_live_sync ? "live sync ON — co-editing %s" : "live sync OFF", g_oplog_path);
        }
        ImGui::Spacing();
        // glTF import: bring a baked .glb in as a replayable `mesh` op (human parity
        // with the AI's import_gltf). Replaces the current op-log with the import.
        ImGui::SetNextItemWidth(-1);
        ImGui::InputText("##glb", g_glb_path, sizeof(g_glb_path));
        if (ImGui::Button("Import .glb", ImVec2(-1, 0))) {
            try {
                prog = program_from_glb(g_glb_path);
                g_sel_mode = SEL_NONE; dirty = true;
                std::snprintf(g_io_status, sizeof(g_io_status), "imported %s -> 1 mesh op", g_glb_path);
            } catch (const std::exception& e) {
                std::snprintf(g_io_status, sizeof(g_io_status), "import failed: %.180s", e.what());
            }
        }
        if (g_io_status[0]) {
            ImGui::Spacing();
            ImGui::PushStyleColor(ImGuiCol_Text, ui::header);
            ImGui::TextWrapped("%s", g_io_status);
            ImGui::PopStyleColor();
        }

        // -- selection -----------------------------------------------------------
        section("SELECTION");
        const char* mode_txt = g_sel_mode == SEL_PICK  ? "picked face"
                             : g_sel_mode == SEL_STACK ? "last result (stacking)"
                                                       : "top face (default) \xe2\x80\x94 click any face to retarget";
        ImGui::TextWrapped("target: %s", mode_txt);
        if (g_sel_mode != SEL_NONE) {
            if (ImGui::SmallButton("reset to top face")) { g_sel_mode = SEL_NONE; rebuild_highlight(); }
        }

        // -- material ------------------------------------------------------------
        section("MATERIAL");
        ImGui::SetNextItemWidth(-74); ImGui::ColorEdit3("albedo", g_albedo);
        ImGui::SetNextItemWidth(-74); ImGui::SliderFloat("metallic", &g_metallic, 0.0f, 1.0f, "%.2f");
        ImGui::SetNextItemWidth(-74); ImGui::SliderFloat("roughness", &g_rough, 0.04f, 1.0f, "%.2f");
        ImGui::Checkbox("flat shading (faceted)", &g_flat);
        ImGui::SameLine(); ImGui::Checkbox("wireframe", &g_wire);
        ImGui::Spacing();
        // Bake these as a PER-FACE `material` op on the current selection — it writes
        // to the op-log SoT, so the same assignment shows in the viewport, the path
        // tracer, and the glTF export. Unassigned faces keep the sliders as fallback.
        if (ImGui::Button("Assign to selection", ImVec2(-1, 0))) {
            prog.material(current_on(), {g_albedo[0], g_albedo[1], g_albedo[2]}, g_metallic, g_rough);
            if (g_sel_mode == SEL_PICK) g_sel_mode = SEL_STACK;  // keep the highlight on the just-painted face
            dirty = true;
        }

        // -- render (the primary action) -----------------------------------------
        section("RENDER");
        ImGui::PushStyleColor(ImGuiCol_Button, ui::accent);
        ImGui::PushStyleColor(ImGuiCol_ButtonHovered, ui::accent_br);
        ImGui::PushStyleColor(ImGuiCol_ButtonActive, ui::accent_br);
        ImGui::PushStyleColor(ImGuiCol_Text, ImVec4(1, 1, 1, 1));
        const bool do_render = ImGui::Button("Path-trace this view", ImVec2(-1, 34));
        ImGui::PopStyleColor(4);
        ImGui::PushStyleColor(ImGuiCol_Text, ui::text_dim);
        ImGui::TextUnformatted("ground truth \xe2\x80\x94 the Cycles-class render");
        ImGui::PopStyleColor();
        if (do_render) {  // path-trace the current model from this camera
            V3 c = g.center, e = orbit_eye(c);
            Camera cam;
            cam.eye = {e[0], e[1], e[2]};
            cam.target = {c[0], c[1], c[2]};
            cam.fov_y = 0.9;
            RenderSettings rs;
            rs.width = 800; rs.height = 600; rs.spp = 64;
            rs.albedo = {g_albedo[0], g_albedo[1], g_albedo[2]};
            rs.metallic = g_metallic; rs.roughness = g_rough;  // same material as the preview
            Image img = path_trace(model, cam, rs);
            write_ppm(img, "mirage_render.ppm");
            std::snprintf(g_io_status, sizeof(g_io_status), "rendered %dx%d @ %dspp -> mirage_render.ppm", rs.width, rs.height, rs.spp);
        }

        // -- model stats + op-log console ----------------------------------------
        section("MODEL");
        ImGui::Text("verts %zu     edges %zu     faces %zu", model.num_verts(), model.num_edges(), model.num_faces());
        ImGui::Text("euler %d", model.euler());
        ImGui::SameLine();
        const bool mani = model.is_closed_manifold();
        ImGui::TextColored(mani ? ui::ok : ui::warn, mani ? "\xe2\x97\x8f closed manifold" : "\xe2\x97\x8b open surface");
        ImGui::Spacing();
        ImGui::PushStyleColor(ImGuiCol_Text, ui::text_dim);
        ImGui::TextUnformatted("op-log (the model itself)");
        ImGui::PopStyleColor();
        ImGui::BeginChild("oplog", ImVec2(0, 172), true);
        ImGui::PushFont(font_mono);
        int i = 0;
        for (const auto& op : prog.ops()) {
            ImGui::TextColored(ui::accent_br, "%2d", i++);
            ImGui::SameLine();
            ImGui::TextUnformatted(Program::label(op).c_str());
        }
        // Lint: silent traps that build but lose intent (same checks the AI gets).
        auto warns = prog.lint();
        if (!warns.empty()) {
            ImGui::Spacing();
            for (const auto& w : warns)
                ImGui::TextColored(ui::warn, "! op %d  %s", w.op_index, w.code.c_str());
        }
        ImGui::PopFont();
        ImGui::EndChild();
        ImGui::End();
        return dirty;
    };

    // The AUTO HUD: what the panel collapses to while the AI drives. A pulsing
    // status dot, the AUTO mark, and the one line that matters — what's being
    // edited right now (an op delta under live sync, or the caller's --autocap).
    // Deliberately tiny, top-left, so the viewport is all model.
    auto auto_hud = [&]() {
        ImGui::SetNextWindowPos(ImVec2(24, 24), ImGuiCond_Always);
        ImGui::SetNextWindowBgAlpha(0.92f);
        ImGui::Begin("##auto", nullptr,
                     ImGuiWindowFlags_NoTitleBar | ImGuiWindowFlags_NoResize | ImGuiWindowFlags_NoMove |
                     ImGuiWindowFlags_AlwaysAutoResize | ImGuiWindowFlags_NoNav |
                     ImGuiWindowFlags_NoScrollbar | ImGuiWindowFlags_NoFocusOnAppearing);
        ImDrawList* dl = ImGui::GetWindowDrawList();
        // a live pulse when interactive; a steady dot for deterministic headless frames
        const float t = float(glfwGetTime());
        const float pulse = shot.empty() ? (0.5f + 0.5f * std::sin(t * 5.2f)) : 1.0f;
        ImGui::PushFont(font_title);
        const float lh = ImGui::GetTextLineHeight();
        const ImVec2 dp = ImGui::GetCursorScreenPos();
        const ImVec2 dc(dp.x + 8.0f, dp.y + lh * 0.5f);
        const ImU32 acc  = ImGui::GetColorU32(ui::accent_br);
        const ImU32 glow = ImGui::GetColorU32(ImVec4(ui::accent_br.x, ui::accent_br.y, ui::accent_br.z, 0.30f * pulse));
        dl->AddCircleFilled(dc, 6.0f + 5.0f * pulse, glow);
        dl->AddCircleFilled(dc, 6.0f, acc);
        ImGui::Dummy(ImVec2(24.0f, lh)); ImGui::SameLine();
        ImGui::PushStyleColor(ImGuiCol_Text, ui::accent_br);
        ImGui::TextUnformatted("AUTO");
        ImGui::PopStyleColor();
        ImGui::PopFont();
        ImGui::PushStyleColor(ImGuiCol_Text, ui::text_dim);
        ImGui::TextUnformatted("Mirage is modeling  \xc2\xb7  the op-log is the model");
        ImGui::PopStyleColor();
        if (g_auto_msg[0]) {  // the headline: what's being edited right now
            ImGui::Dummy(ImVec2(0, 3));
            ImGui::PushFont(font_h);
            ImGui::PushStyleColor(ImGuiCol_Text, ui::text);
            ImGui::TextUnformatted(g_auto_msg);
            ImGui::PopStyleColor();
            ImGui::PopFont();
        }
        // a streaming tail of the op-log — the last few ops as they land (newest marked)
        ImGui::Dummy(ImVec2(0, 3));
        ImGui::PushFont(font_mono);
        const size_t nops = prog.size();
        for (size_t k = (nops > 3 ? nops - 3 : 0); k < nops; ++k) {
            const bool newest = (k + 1 == nops);
            ImGui::PushStyleColor(ImGuiCol_Text, newest ? ui::accent_br : ui::text_dim);
            ImGui::Text("%s %s", newest ? "\xe2\x96\xb8" : "  ", Program::label(prog.ops()[k]).c_str());  // U+25B8 marks newest
            ImGui::PopStyleColor();
        }
        ImGui::PushStyleColor(ImGuiCol_Text, ui::text_dim);
        ImGui::Text("%zu ops  \xc2\xb7  %zu faces", prog.size(), model.num_faces());
        ImGui::PopStyleColor();
        ImGui::PopFont();
        if (shot.empty()) {  // interactive: tell the human how to reclaim the tools
            ImGui::Dummy(ImVec2(0, 1));
            ImGui::PushStyleColor(ImGuiCol_Text, ui::text_dim);
            ImGui::TextUnformatted("click anywhere to take control");
            ImGui::PopStyleColor();
        }
        // a thin accent spine down the card's left edge — reads as a live/product chip
        const ImVec2 wp = ImGui::GetWindowPos();
        dl->AddRectFilled(wp, ImVec2(wp.x + 3.0f, wp.y + ImGui::GetWindowHeight()), acc);
        ImGui::End();
    };

    // A small orientation gizmo in the bottom-right corner: the world axes projected
    // by the live camera basis, so you always know which way is up (a human navigation
    // aid — hidden while the AI drives, to keep AUTO frames clean). ±X/Y/Z as coloured
    // tips, positive ends labelled, drawn far-to-near so the nearest overlays.
    auto axis_gizmo = [&]() {
        ImDrawList* dl = ImGui::GetForegroundDrawList();
        const ImVec2 ds = ImGui::GetIO().DisplaySize;
        const ImVec2 o(ds.x - 58.0f, ds.y - 58.0f);
        const float R = 25.0f;
        const V3 fwd = cross(g_cam_up, g_cam_right);  // camera forward (eye -> scene)
        struct Tip { float depth; ImVec2 p; ImU32 col; char n; bool pos; };
        std::array<Tip, 6> tips;
        const V3 dir[3] = {{1,0,0}, {0,1,0}, {0,0,1}};
        const ImU32 col[3] = {IM_COL32(216,98,98,255), IM_COL32(122,198,114,255), IM_COL32(114,154,234,255)};
        const char nm[3] = {'X', 'Y', 'Z'};
        int ti = 0;
        for (int a = 0; a < 3; ++a)
            for (int s = 1; s >= -1; s -= 2) {
                const V3 d{dir[a][0]*s, dir[a][1]*s, dir[a][2]*s};
                const float sx = dot(d, g_cam_right), sy = -dot(d, g_cam_up);
                tips[ti++] = {dot(d, fwd), ImVec2(o.x + sx * R, o.y + sy * R), col[a], nm[a], s > 0};
            }
        std::sort(tips.begin(), tips.end(), [](const Tip& x, const Tip& y) { return x.depth > y.depth; });
        for (const auto& tp : tips) {
            dl->AddLine(o, tp.p, IM_COL32(205, 210, 220, 80), 1.6f);
            if (tp.pos) {
                dl->AddCircleFilled(tp.p, 8.5f, tp.col);
                char lbl[2] = {tp.n, 0};
                const ImVec2 ts = ImGui::CalcTextSize(lbl);
                dl->AddText(ImVec2(tp.p.x - ts.x * 0.5f, tp.p.y - ts.y * 0.5f), IM_COL32(18, 20, 24, 255), lbl);
            } else {
                dl->AddCircleFilled(tp.p, 6.0f, IM_COL32(38, 42, 50, 255));
                dl->AddCircle(tp.p, 6.0f, tp.col, 0, 1.6f);
            }
        }
    };

    auto frame = [&]() {
        ImGui_ImplOpenGL3_NewFrame();
        ImGui_ImplGlfw_NewFrame();
        ImGui::NewFrame();
        if (g_auto_mode) {
            auto_hud();  // the AI is driving; the panel steps aside for the status HUD
        } else if (panel()) {
            model = prog.build(&last_tag); g = build_gpu(model); upload(); build_ground(); rebuild_highlight();
            if (g_live_sync) save_oplog();  // push the human's edit to the shared op-log
        }
        if (!g_auto_mode && g.verts > 0) axis_gizmo();  // orientation aid (human mode only)
        draw();
        ImGui::Render();
        ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());
    };

    if (!shot.empty()) {  // headless verification: render a couple frames (mesh + GUI) -> PPM
        if (cam_set) { g_yaw = float(cam_yaw); g_pitch = float(cam_pitch); g_dist = float(cam_dist); }
        // pin the orbit centre to a fixed world point (see --target above): each
        // partial op-log has its own bbox centre, so compensate via the pan offset
        // that the camera already honours, keeping the view rock-steady frame to frame.
        if (tgt_set) g_pan = {float(tgt_x) - g.center[0], float(tgt_y) - g.center[1], float(tgt_z) - g.center[2]};
        const Face* sf = nohl ? nullptr : nearest_face(model, {0.5, 0.0, 0.2});  // pre-pick a side face to show the highlight
        if (sf) { g_sel_mode = SEL_PICK; g_sel = face_centroid(model, sf); rebuild_highlight(); }
        frame();  // warm-up (ImGui font atlas + first-frame auto-sizing)
        frame();
        if (drag_set) {
            // Drive the REAL input handlers with a synthetic drag from the viewport
            // centre, then re-render. This is the exact path a human's mouse takes
            // (on_mouse press -> on_cursor moves -> release), so the resulting camera
            // is a true readout of the input->camera mapping under test.
            const double x0 = W * 0.5, y0 = H * 0.5;
            const float yaw0 = g_yaw, pitch0 = g_pitch;
            if (drag_btn == 'R' || drag_btn == 'r') {
                g_panning = true; g_lx = x0; g_ly = y0;
                on_cursor(win, x0 + drag_dx, y0 + drag_dy);
                g_panning = false;
            } else {
                g_drag = true; g_moved = false; g_press_x = x0; g_press_y = y0; g_lx = x0; g_ly = y0;
                on_cursor(win, x0 + drag_dx * 0.5, y0 + drag_dy * 0.5);  // trip the move threshold
                on_cursor(win, x0 + drag_dx, y0 + drag_dy);              // apply the full delta
                g_drag = false;
            }
            std::printf("drag %c dx=%.0f dy=%.0f : yaw %.4f -> %.4f (d=%+.4f)  pitch %.4f -> %.4f (d=%+.4f)\n",
                        drag_btn, drag_dx, drag_dy, yaw0, g_yaw, g_yaw - yaw0, pitch0, g_pitch, g_pitch - pitch0);
            frame();  // render the post-drag view
        }
        if (watch_secs > 0.0) {  // headless live-sync: poll the shared op-log and reload external edits
            g_live_sync = true;
            g_last_mtime = file_mtime(g_oplog_path);
            std::printf("watching %s for %.1fs (live sync)...\n", g_oplog_path, watch_secs);
            const double t0 = glfwGetTime();
            while (glfwGetTime() - t0 < watch_secs) {
                long long m = file_mtime(g_oplog_path);
                if (m != 0 && m != g_last_mtime) {
                    g_last_mtime = m;
                    if (reload_oplog()) std::printf("  reloaded -> %zu ops, %zu faces\n", prog.size(), model.num_faces());
                }
                frame();
                std::this_thread::sleep_for(std::chrono::milliseconds(100));
            }
        }
        glFinish();
        int fw, fh; glfwGetFramebufferSize(win, &fw, &fh);
        write_ppm(shot, fw, fh);
        std::printf("wrote %s (%dx%d, %d tris)\n", shot.c_str(), fw, fh, g.verts / 3);
    } else {
        std::printf("Mirage native viewport — left-drag orbit, right-drag pan, scroll zoom, click to pick, Esc quit.\n");
        while (!glfwWindowShouldClose(win)) {
            glfwPollEvents();
            if (glfwGetKey(win, GLFW_KEY_ESCAPE) == GLFW_PRESS) break;
            if (g_pick_request) { g_pick_request = false; do_pick(g_px, g_py); }
            const double now = glfwGetTime();
            if (g_live_sync) {  // watch the shared op-log; reload the AI's edits (~4 Hz)
                if (now - g_last_poll > 0.25) {
                    g_last_poll = now;
                    long long m = file_mtime(g_oplog_path);
                    if (m != 0 && m != g_last_mtime) {
                        g_last_mtime = m;
                        const int before = int(prog.size());
                        if (reload_oplog()) {  // an AI edit landed -> arm AUTO and name it
                            const int delta = int(prog.size()) - before;
                            const std::string last = prog.size() ? Program::label(prog.ops().back()) : std::string("idle");
                            if (delta > 0) std::snprintf(g_auto_msg, sizeof(g_auto_msg), "+%d  \xc2\xb7  %.120s", delta, last.c_str());
                            else           std::snprintf(g_auto_msg, sizeof(g_auto_msg), "%.150s", last.c_str());
                            if (now > g_auto_suppress_until) g_auto_mode = true;  // unless the human just took control
                            g_auto_last_edit = now;
                        }
                    }
                }
            }
            if (g_auto_mode && !g_auto_force && now - g_auto_last_edit > 2.2)
                g_auto_mode = false;  // the AI went quiet — hand the tools back
            frame();
            glfwSwapBuffers(win);
        }
    }

    ImGui_ImplOpenGL3_Shutdown();
    ImGui_ImplGlfw_Shutdown();
    ImGui::DestroyContext();
    glfwTerminate();
    return 0;
}
