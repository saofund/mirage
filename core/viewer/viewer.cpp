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

#include <array>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <iterator>
#include <string>
#include <unordered_map>
#include <vector>

#include "mirage/mesh.hpp"
#include "mirage/program.hpp"

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
        for (size_t i = 1; i + 1 < vs.size(); ++i) {
            Vert* tri[3] = {vs[0], vs[i], vs[i + 1]};
            for (Vert* v : tri) {
                V3 n = vn[v];
                g.data.insert(g.data.end(), {(float)v->co[0], (float)v->co[1], (float)v->co[2], n[0], n[1], n[2]});
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
static const char* VERT = R"(#version 330 core
layout(location=0) in vec3 aPos;
layout(location=1) in vec3 aNormal;
uniform mat4 uMVP;
out vec3 vN;
void main(){ gl_Position = uMVP * vec4(aPos,1.0); vN = aNormal; }
)";
static const char* FRAG = R"(#version 330 core
in vec3 vN; out vec4 frag;
uniform vec3 uColor;
void main(){
  vec3 n = normalize(vN);
  vec3 L = normalize(vec3(0.35,0.5,0.8));
  float d = max(dot(n,L),0.0);
  vec3 c = uColor*(0.28 + 0.72*d);
  frag = vec4(pow(c, vec3(0.4545)), 1.0);
}
)";

// camera / interaction state
static bool g_imgui = false;
static float g_yaw = 2.3f, g_pitch = 0.35f, g_dist = 3.0f;
static double g_lx = 0, g_ly = 0;
static bool g_drag = false, g_moved = false;
static double g_press_x = 0, g_press_y = 0;
static bool g_pick_request = false; static double g_px = 0, g_py = 0;  // a click to resolve into a face
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
// JSON the AI (MCP save_mesh_program/load_mesh_program) reads and writes.
static char g_oplog_path[256] = "mirage_oplog.json";
static char g_io_status[256] = "";

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
    if (g_drag && !ui_wants_mouse()) {
        if (std::fabs(x - g_press_x) + std::fabs(y - g_press_y) > 4.0) g_moved = true;
        if (g_moved) {  // drag past a small threshold -> orbit
            g_yaw += float(x - g_lx) * 0.01f; g_pitch += float(y - g_ly) * 0.01f;
            if (g_pitch > 1.5f) g_pitch = 1.5f; if (g_pitch < -1.5f) g_pitch = -1.5f;
        }
    }
    g_lx = x; g_ly = y;
}
static void on_scroll(GLFWwindow*, double, double dy) {
    if (ui_wants_mouse()) return;
    g_dist *= (1.0f - 0.12f * float(dy)); if (g_dist < 0.2f) g_dist = 0.2f;
}

static void write_ppm(const std::string& path, int W, int H) {
    std::vector<unsigned char> px(size_t(W) * H * 3);
    glReadPixels(0, 0, W, H, GL_RGB, GL_UNSIGNED_BYTE, px.data());
    std::ofstream f(path, std::ios::binary);
    f << "P6\n" << W << " " << H << "\n255\n";
    for (int y = H - 1; y >= 0; --y) f.write(reinterpret_cast<char*>(&px[size_t(y) * W * 3]), W * 3);
}

int main(int argc, char** argv) {
    std::string shot, load_path;
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        if (a == "--screenshot" && i + 1 < argc) shot = argv[++i];
        else if (a == "--oplog" && i + 1 < argc) { load_path = argv[++i]; std::snprintf(g_oplog_path, sizeof(g_oplog_path), "%s", load_path.c_str()); }
    }

    Program prog;
    bool loaded_ok = false;
    if (!load_path.empty()) {  // open straight onto a shared op-log (e.g. one an AI just saved)
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
    GLFWwindow* win = glfwCreateWindow(W, H, "Mirage — native modeling viewport", nullptr, nullptr);
    if (!win) { std::fprintf(stderr, "window/context creation failed\n"); glfwTerminate(); return 1; }
    glfwMakeContextCurrent(win);
    if (!gladLoadGL(reinterpret_cast<GLADloadfunc>(glfwGetProcAddress))) {
        std::fprintf(stderr, "glad load failed\n"); return 1;
    }
    if (shot.empty()) {
        glfwSetMouseButtonCallback(win, on_mouse);
        glfwSetCursorPosCallback(win, on_cursor);
        glfwSetScrollCallback(win, on_scroll);
    }
    glEnable(GL_DEPTH_TEST);

    GLuint prog_gl = glCreateProgram();
    glAttachShader(prog_gl, compile(GL_VERTEX_SHADER, VERT));
    glAttachShader(prog_gl, compile(GL_FRAGMENT_SHADER, FRAG));
    glLinkProgram(prog_gl);
    const GLint locMVP = glGetUniformLocation(prog_gl, "uMVP");
    const GLint locColor = glGetUniformLocation(prog_gl, "uColor");

    GLuint vao, vbo;
    glGenVertexArrays(1, &vao); glGenBuffers(1, &vbo);
    glBindVertexArray(vao);
    glBindBuffer(GL_ARRAY_BUFFER, vbo);
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 6 * sizeof(float), (void*)0);
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 6 * sizeof(float), (void*)(3 * sizeof(float)));
    glEnableVertexAttribArray(1);

    GLuint hvao, hvbo; int hl_verts = 0;  // selection highlight geometry
    glGenVertexArrays(1, &hvao); glGenBuffers(1, &hvbo);
    glBindVertexArray(hvao);
    glBindBuffer(GL_ARRAY_BUFFER, hvbo);
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 6 * sizeof(float), (void*)0);
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 6 * sizeof(float), (void*)(3 * sizeof(float)));
    glEnableVertexAttribArray(1);

    std::string last_tag;            // the build's most recent __out tag (for sel::last)
    Mesh model = prog.build(&last_tag);
    Gpu g = build_gpu(model);
    g_dist = g.radius * 3.0f;
    auto upload = [&]() {
        glBindBuffer(GL_ARRAY_BUFFER, vbo);
        glBufferData(GL_ARRAY_BUFFER, GLsizeiptr(g.data.size() * sizeof(float)),
                     g.data.empty() ? nullptr : g.data.data(), GL_DYNAMIC_DRAW);
    };
    upload();

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
        int fw, fh; glfwGetFramebufferSize(win, &fw, &fh);
        float ndcx = 2.0f * float(px) / float(fw) - 1.0f;
        float ndcy = 1.0f - 2.0f * float(py) / float(fh);
        V3 c = g.center, eye = orbit_eye(c);
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

    auto draw = [&]() {
        int fw, fh; glfwGetFramebufferSize(win, &fw, &fh);
        glViewport(0, 0, fw, fh);
        glClearColor(0.10f, 0.11f, 0.13f, 1.0f);
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
        if (g.verts == 0) return;
        V3 c = g.center, eye = orbit_eye(c);
        Mat4 mvp = mul(perspective(0.9f, float(fw) / float(fh ? fh : 1), 0.05f, 100.0f),
                       look_at(eye, c, {0, 0, 1}));
        glUseProgram(prog_gl);
        glUniformMatrix4fv(locMVP, 1, GL_FALSE, mvp.data());
        glUniform3f(locColor, 0.82f, 0.80f, 0.74f);
        glBindVertexArray(vao);
        glDrawArrays(GL_TRIANGLES, 0, g.verts);
        if (hl_verts > 0) {  // selected face, pulled slightly forward to avoid z-fighting
            glEnable(GL_POLYGON_OFFSET_FILL); glPolygonOffset(-1.0f, -1.0f);
            glUniform3f(locColor, 1.0f, 0.55f, 0.12f);
            glBindVertexArray(hvao);
            glDrawArrays(GL_TRIANGLES, 0, hl_verts);
            glDisable(GL_POLYGON_OFFSET_FILL);
        }
    };

    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    ImGui::StyleColorsDark();
    ImGui_ImplGlfw_InitForOpenGL(win, shot.empty());  // install input callbacks only when interactive
    ImGui_ImplOpenGL3_Init("#version 330");
    g_imgui = shot.empty();

    auto panel = [&]() -> bool {  // the tool panel; returns true if the op-log changed
        bool dirty = false;
        ImGui::SetNextWindowPos(ImVec2(14, 14), ImGuiCond_FirstUseEver);
        ImGui::Begin("Mirage  -  modeling", nullptr, ImGuiWindowFlags_AlwaysAutoResize);
        ImGui::TextDisabled("primitives (start fresh)");
        if (ImGui::Button("New Cube"))     { prog.clear(); prog.cube(1.0); g_sel_mode = SEL_NONE; dirty = true; }
        ImGui::SameLine();
        if (ImGui::Button("New Cylinder")) { prog.clear(); prog.cylinder(24, 0.5, 1.0); g_sel_mode = SEL_NONE; dirty = true; }
        ImGui::Spacing();
        ImGui::TextDisabled("operators (on the highlighted target)");
        // act on the current selector; a pick then stacks on what the op produced
        if (ImGui::Button("Inset"))   { prog.inset(current_on(), 0.3);   if (g_sel_mode == SEL_PICK) g_sel_mode = SEL_STACK; dirty = true; }
        ImGui::SameLine();
        if (ImGui::Button("Extrude")) { prog.extrude(current_on(), 0.5); if (g_sel_mode == SEL_PICK) g_sel_mode = SEL_STACK; dirty = true; }
        ImGui::SameLine();
        if (ImGui::Button("Subdivide")) { prog.subdivide(1); dirty = true; }
        ImGui::Spacing();
        if (ImGui::Button("Undo"))  { prog.undo(); dirty = true; }
        ImGui::SameLine();
        if (ImGui::Button("Reset")) { prog.clear(); prog.cube(1.0); g_sel_mode = SEL_NONE; dirty = true; }
        ImGui::Spacing();
        // The op-log is the shared SoT: Save writes the JSON an AI (MCP) can Load,
        // and vice-versa — one model, a human and an AI both editing it.
        ImGui::TextDisabled("shared op-log (same JSON the AI reads/writes)");
        ImGui::SetNextItemWidth(220);
        ImGui::InputText("##path", g_oplog_path, sizeof(g_oplog_path));
        if (ImGui::Button("Save")) {
            std::ofstream f(g_oplog_path);
            if (f) { f << prog.to_json(2); std::snprintf(g_io_status, sizeof(g_io_status), "saved %zu ops -> %s", prog.size(), g_oplog_path); }
            else std::snprintf(g_io_status, sizeof(g_io_status), "save failed: %s", g_oplog_path);
        }
        ImGui::SameLine();
        if (ImGui::Button("Load")) {
            std::ifstream f(g_oplog_path);
            if (!f) std::snprintf(g_io_status, sizeof(g_io_status), "load failed: %s", g_oplog_path);
            else {
                std::string s((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
                try {
                    Program loaded = Program::from_json(s);
                    loaded.build();  // validate before adopting
                    prog = std::move(loaded);
                    g_sel_mode = SEL_NONE;
                    dirty = true;
                    std::snprintf(g_io_status, sizeof(g_io_status), "loaded %zu ops <- %s", prog.size(), g_oplog_path);
                } catch (const std::exception& e) {
                    std::snprintf(g_io_status, sizeof(g_io_status), "bad op-log: %.180s", e.what());
                }
            }
        }
        if (g_io_status[0]) { ImGui::SameLine(); ImGui::TextDisabled("%s", g_io_status); }
        ImGui::Spacing();
        ImGui::TextDisabled("selection (the next op's target, highlighted orange)");
        const char* mode_txt = g_sel_mode == SEL_PICK  ? "picked face"
                             : g_sel_mode == SEL_STACK ? "last result (stacking)"
                                                       : "top face (default) — click any face to retarget";
        ImGui::Text("target: %s", mode_txt);
        if (g_sel_mode != SEL_NONE) {
            ImGui::SameLine();
            if (ImGui::SmallButton("reset to top")) { g_sel_mode = SEL_NONE; rebuild_highlight(); }
        }
        ImGui::Separator();
        ImGui::Text("verts %zu   edges %zu   faces %zu", model.num_verts(), model.num_edges(), model.num_faces());
        ImGui::Text("euler %d   manifold %s", model.euler(), model.is_closed_manifold() ? "yes" : "no");
        ImGui::Separator();
        ImGui::TextDisabled("op-log (the model)");
        int i = 0;
        for (const auto& op : prog.ops()) ImGui::Text("%2d  %s", i++, Program::label(op).c_str());
        ImGui::End();
        return dirty;
    };

    auto frame = [&]() {
        ImGui_ImplOpenGL3_NewFrame();
        ImGui_ImplGlfw_NewFrame();
        ImGui::NewFrame();
        if (panel()) { model = prog.build(&last_tag); g = build_gpu(model); upload(); rebuild_highlight(); }
        draw();
        ImGui::Render();
        ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());
    };

    if (!shot.empty()) {  // headless verification: render a couple frames (mesh + GUI) -> PPM
        const Face* sf = nearest_face(model, {0.5, 0.0, 0.2});  // pre-pick a side face to show the highlight
        if (sf) { g_sel_mode = SEL_PICK; g_sel = face_centroid(model, sf); rebuild_highlight(); }
        frame();  // warm-up (ImGui font atlas + first-frame auto-sizing)
        frame();
        glFinish();
        int fw, fh; glfwGetFramebufferSize(win, &fw, &fh);
        write_ppm(shot, fw, fh);
        std::printf("wrote %s (%dx%d, %d tris)\n", shot.c_str(), fw, fh, g.verts / 3);
    } else {
        std::printf("Mirage native viewport — drag to orbit, scroll to zoom, click tools to model, Esc to quit.\n");
        while (!glfwWindowShouldClose(win)) {
            glfwPollEvents();
            if (glfwGetKey(win, GLFW_KEY_ESCAPE) == GLFW_PRESS) break;
            if (g_pick_request) { g_pick_request = false; do_pick(g_px, g_py); }
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
