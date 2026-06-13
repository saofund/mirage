#include "mirage/lint.hpp"

#include <cmath>
#include <cstdio>

namespace mirage {

namespace {

constexpr double DEGENERATE = 1e-9;
constexpr int MAX_SUBDIVIDE = 6;

std::string num(double v) {
    char buf[32];
    std::snprintf(buf, sizeof(buf), "%g", v);
    return buf;
}

// A numeric op param honoring the same defaults as the kernel/lint (absent or
// non-numeric => the default, so the clamp checks match the built geometry).
bool num_param(const json& op, const char* key, double& out) {
    if (!op.contains(key) || !op[key].is_number()) return false;
    out = op[key].get<double>();
    return true;
}

void add(std::vector<LintWarning>& w, int i, std::string code, std::string msg, std::string sug) {
    w.push_back({i, std::move(code), std::move(msg), std::move(sug)});
}

// Recurse so traps at ANY nesting depth (inside and/or/not) are caught.
void lint_selector(int i, const json& sel, std::vector<LintWarning>& w) {
    if (!sel.is_object()) return;
    if (sel.contains("and") || sel.contains("or")) {
        if (sel.contains("and")) for (const auto& s : sel["and"]) lint_selector(i, s, w);
        if (sel.contains("or")) for (const auto& s : sel["or"]) lint_selector(i, s, w);
        return;
    }
    if (sel.contains("not")) { lint_selector(i, sel["not"], w); return; }
    const std::string by = sel.value("by", "");
    if (by == "extreme") {
        const std::string which = sel.value("which", "max");
        if (which != "max" && which != "min")
            add(w, i, "extreme_which",
                "which='" + which + "' is not 'max'/'min'; the kernel silently treats anything != 'max' "
                "as MIN (selects the opposite face)", "use which: 'max' or 'min'");
    }
    if ((by == "side" || by == "normal") && sel.contains("sign") && sel["sign"].is_number() &&
        sel["sign"].get<double>() == 0.0)
        add(w, i, "sign_zero", "sign=0 selects nothing meaningful", "use sign +1 or -1");
    if (by == "normal" && sel.contains("dir") && sel["dir"] == json::array({0, 0, 0}))
        add(w, i, "dir_zero", "dir=[0,0,0] has no direction", "give a real direction or use axis/sign");
}

}  // namespace

std::vector<LintWarning> lint_program(const std::vector<json>& ops) {
    std::vector<LintWarning> w;
    std::string prev_op;
    bool have_prev = false;
    for (int i = 0; i < static_cast<int>(ops.size()); ++i) {
        const json& op = ops[i];
        if (!op.is_object()) {
            add(w, i, "malformed_op", "op #" + std::to_string(i) + " is not a dict", "each op must be a dict");
            continue;
        }
        const std::string name = op.value("op", "");
        const json sel = op.contains("on") ? op["on"] : json(nullptr);

        if (name == "extrude") {
            double d;
            if (num_param(op, "distance", d) && std::fabs(d) < DEGENERATE)
                add(w, i, "extrude_noop",
                    "distance ~0 is a silent no-op; its mark is stamped on the un-extruded face, so a "
                    "later selector picks the wrong geometry", "use a non-zero distance");
            if (sel.is_object() && sel.value("by", "") == "all")
                add(w, i, "extrude_all", "extrude on {by:all} has no boundary edges -> no side walls (no-op)",
                    "select a face region, not the whole mesh");
        }
        if (name == "inset") {
            double t;
            if (num_param(op, "thickness", t) && !(1e-3 <= t && t <= 0.999))
                add(w, i, "inset_clamped", "thickness " + num(t) + " is silently clamped to [1e-3, 0.999] — "
                    "the built geometry will NOT match the requested value", "pick a thickness in (0, 1)");
        }
        if (name == "bevel") {
            double bw;
            if (num_param(op, "width", bw) && !(1e-3 <= bw && bw <= 0.999))
                add(w, i, "bevel_width_clamped", "width " + num(bw) + " is silently clamped to [1e-3, 0.999] — "
                    "the built geometry will NOT match the requested value", "pick a width in (0, 1)");
            double depth = 0.1;
            num_param(op, "depth", depth);
            if (depth == 0.0)
                add(w, i, "bevel_flat", "depth=0 makes bevel a plain inset (no chamfer)", "use a non-zero depth");
        }
        if (name == "subdivide") {
            double lv = 1;
            if (num_param(op, "levels", lv)) {
                if (lv <= 0)
                    add(w, i, "subdivide_noop", "levels=" + num(lv) + " subdivides nothing", "use levels >= 1");
                else if (lv > MAX_SUBDIVIDE)
                    add(w, i, "subdivide_explosive", "levels=" + num(lv) + " grows faces ~4^levels (very large/slow)",
                        "keep levels <= " + std::to_string(MAX_SUBDIVIDE));
            }
        }
        if (sel.is_object()) lint_selector(i, sel, w);
        if (sel.is_object() && sel.value("by", "") == "last_created" &&
            (!have_prev || prev_op == "cube" || prev_op == "cylinder" || prev_op == "subdivide"))
            add(w, i, "last_created_broad",
                "last_created right after a primitive/subdivide resolves to the WHOLE surface (every face "
                "inherits the step tag)", "use an explicit selector");
        prev_op = name;
        have_prev = true;
    }
    return w;
}

}  // namespace mirage
