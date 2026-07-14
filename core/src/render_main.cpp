// mirage_render — the offline path-traced renderer (CLI).
//
//   mirage_render [--oplog FILE] [--out IMG.ppm] [--spp N] [--w N --h N] [--threads N]
//                 [--cam-eye X Y Z] [--cam-target X Y Z] [--cam-up X Y Z] [--cam-fov RAD]
//                 [--denoise [N]] [--smooth-angle DEG | --flat]
//
// --denoise runs an edge-avoiding a-trous wavelet filter (N passes, default 5) so a
// low-spp render comes out clean — the difference between a grainy path-traced clip
// and a usable one.
//
// --smooth-angle shades a face corner smooth when the faces meeting there are within DEG
// of each other (default 30), so a subdivided surface renders as the curved surface it
// approximates while real creases stay hard. --flat restores faceted geometric normals.
//
// --threads caps the worker count (default 0 = every logical core, which pins the
// CPU for the duration of a high-spp render); set it below your core count to
// leave the machine responsive.
//
// Builds an op-log (from FILE, or a default beveled boss) and path-traces it to a
// PPM with global illumination, soft sky+sun lighting and a ground plane. This is
// the ground-truth render of the same mirage::Program a human/AI authored. The
// camera defaults to a 3/4 exterior view; the --cam-* flags place it anywhere
// (e.g. inside a room), which is what interior scenes need.
#include <array>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <string>

#include "mirage/program.hpp"
#include "mirage/raytrace.hpp"

using namespace mirage;

static std::string read_file(const std::string& path) {
    std::ifstream f(path);
    if (!f) return "";
    return std::string((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
}

int main(int argc, char** argv) {
    std::string oplog, out = "render.ppm";
    RenderSettings s;
    Camera cam;  // default 3/4 exterior view; any field overridable via --cam-* below
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        auto next = [&](double dflt) { return i + 1 < argc ? std::atof(argv[++i]) : dflt; };
        auto next3 = [&](std::array<double, 3> dflt) {  // read up to 3 floats into a vector
            for (int k = 0; k < 3 && i + 1 < argc; ++k) dflt[k] = std::atof(argv[++i]);
            return dflt;
        };
        if (a == "--oplog" && i + 1 < argc) oplog = argv[++i];
        else if (a == "--out" && i + 1 < argc) out = argv[++i];
        else if (a == "--spp") s.spp = int(next(s.spp));
        else if (a == "--w") s.width = int(next(s.width));
        else if (a == "--h") s.height = int(next(s.height));
        else if (a == "--bounce") s.max_bounce = int(next(s.max_bounce));
        else if (a == "--metallic") s.metallic = next(s.metallic);
        else if (a == "--rough") s.roughness = next(s.roughness);
        else if (a == "--env") s.env_intensity = next(s.env_intensity);
        else if (a == "--sun") s.sun_intensity = next(s.sun_intensity);
        else if (a == "--sun-dir") s.sun_dir = next3(s.sun_dir);
        else if (a == "--exposure") s.exposure = next(s.exposure);
        else if (a == "--clamp") s.clamp_indirect = next(s.clamp_indirect);
        else if (a == "--threads") s.threads = unsigned(next(double(s.threads)));
        else if (a == "--denoise") {  // edge-avoiding a-trous denoise; optional iteration count
            if (i + 1 < argc && argv[i + 1][0] >= '0' && argv[i + 1][0] <= '9') s.denoise = std::atoi(argv[++i]);
            else s.denoise = 5;
        }
        else if (a == "--cam-eye") cam.eye = next3(cam.eye);
        else if (a == "--cam-target") cam.target = next3(cam.target);
        else if (a == "--cam-up") cam.up = next3(cam.up);
        else if (a == "--cam-fov") cam.fov_y = next(cam.fov_y);
        else if (a == "--aperture") s.aperture = next(s.aperture);        // thin-lens DOF radius
        else if (a == "--focus-dist") s.focus_dist = next(s.focus_dist);  // sharp-plane distance (0 = auto)
        else if (a == "--bloom") {   // photographic glow; optional strength (default 0.12)
            if (i + 1 < argc && argv[i + 1][0] >= '0' && argv[i + 1][0] <= '9') s.bloom = std::atof(argv[++i]);
            else s.bloom = 0.12;
        }
        else if (a == "--bloom-threshold") s.bloom_threshold = next(s.bloom_threshold);
        else if (a == "--smooth-angle") s.smooth_angle = next(s.smooth_angle);  // shade smooth below DEG
        else if (a == "--flat") s.smooth_angle = 0.0;                           // faceted (geometric normals)
    }

    Program prog;
    if (!oplog.empty()) {
        const std::string js = read_file(oplog);
        if (js.empty()) { std::fprintf(stderr, "could not read %s\n", oplog.c_str()); return 1; }
        try { prog = Program::from_json(js); }
        catch (const std::exception& e) { std::fprintf(stderr, "bad op-log: %s\n", e.what()); return 1; }
    } else {  // a default scene: a chamfered boss
        prog.cube(1.2).bevel(sel::normal("z"), 0.28, 0.22).extrude(sel::last(), 0.5);
    }

    Mesh mesh;
    try { mesh = prog.build(); }
    catch (const std::exception& e) { std::fprintf(stderr, "build failed: %s\n", e.what()); return 1; }

    std::printf("path-tracing %zu faces  %dx%d  spp=%d  bounce=%d%s ...\n",
                mesh.num_faces(), s.width, s.height, s.spp, s.max_bounce,
                s.denoise ? "  +denoise" : "");
    Image img = path_trace(mesh, cam, s);
    write_ppm(img, out);
    std::printf("wrote %s (%dx%d)\n", out.c_str(), img.w, img.h);
    return 0;
}
