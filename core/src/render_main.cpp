// mirage_render — the offline path-traced renderer (CLI).
//
//   mirage_render [--oplog FILE] [--out IMG.ppm] [--spp N] [--w N --h N]
//
// Builds an op-log (from FILE, or a default beveled boss) and path-traces it to a
// PPM with global illumination, soft sky+sun lighting and a ground plane. This is
// the ground-truth render of the same mirage::Program a human/AI authored.
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
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        auto next = [&](double dflt) { return i + 1 < argc ? std::atof(argv[++i]) : dflt; };
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
        else if (a == "--exposure") s.exposure = next(s.exposure);
        else if (a == "--clamp") s.clamp_indirect = next(s.clamp_indirect);
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

    Camera cam;  // a 3/4 view that frames the model on the floor
    std::printf("path-tracing %zu faces  %dx%d  spp=%d  bounce=%d ...\n",
                mesh.num_faces(), s.width, s.height, s.spp, s.max_bounce);
    Image img = path_trace(mesh, cam, s);
    write_ppm(img, out);
    std::printf("wrote %s (%dx%d)\n", out.c_str(), img.w, img.h);
    return 0;
}
