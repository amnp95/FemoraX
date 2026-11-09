#include "io/support.hpp"

#include <algorithm>
#include <cstdint>
#include <string>
#include <variant>
#include <vector>

#include "io/h5.hpp"

namespace femorax::io {

namespace {

constexpr const char* kOpenSeesHint = " Run this model with the OpenSees backend.";

[[noreturn]] void unsupported(const std::string& msg) {
  throw UnsupportedComponent("unsupported component: " + msg + "." + kOpenSeesHint);
}

}  // namespace

void check_supported(const Model& m, const Capabilities& caps) {
  for (const auto& s : m.time_series) {
    if (!s.dt && (s.use_last || s.prepend_zero)) {
      unsupported("Path time series " + std::to_string(s.tag) + " with a time array and " +
                  (s.use_last ? "use_last" : "prepend_zero") + ": OpenSees (PathTimeSeries) silently " +
                  "ignores that flag for -time/-fileTime series, so the two programs would differ. Remove " +
                  "the flag, or give the series a uniform dt");
    }
  }
  for (const auto& r : m.recorders) {
    if (const auto* n = std::get_if<NodeRecorder>(&r.data); n && !n->time_series.empty()) {
      unsupported("Node recorder " + r.name + " with -timeSeries (absolute response) is not available yet");
    }
  }
  for (const auto& mat : m.materials) {
    const bool pdmy = std::holds_alternative<PressureDependMultiYieldData>(mat.data);
    const bool pimy = std::holds_alternative<PressureIndependMultiYieldData>(mat.data);
    if ((pdmy && !caps.pressure_depend_multi_yield) || (pimy && !caps.pressure_independ_multi_yield)) {
      unsupported(std::string("nD material ") + material_type_name(mat) + " (/materials/nd/" +
                  std::to_string(mat.tag) + ") is not available yet (milestone 2)");
    }
  }
  // PressureIndependMultiYield: nothing that reads its initial tangent after the switch to
  // the plastic stage, which OpenSees scales per point (docs/deviations.md): betaKinit
  // damping on its elements and ModifiedNewton -initial. (The schema keeps the number of
  // surfaces below 40, within FemoraX's buckets.)
  std::vector<std::int32_t> pimy_tags;
  for (const auto& mat : m.materials)
    if (std::holds_alternative<PressureIndependMultiYieldData>(mat.data)) pimy_tags.push_back(mat.tag);
  if (!pimy_tags.empty()) {
    std::vector<std::int32_t> pimy_elements;
    for (const auto& g : m.elements)
      if (std::find(pimy_tags.begin(), pimy_tags.end(), g.material) != pimy_tags.end())
        pimy_elements.insert(pimy_elements.end(), g.tags.begin(), g.tags.end());
    std::sort(pimy_elements.begin(), pimy_elements.end());
    for (const auto& r : m.rayleigh) {
      if (r.betaKinit == 0.0) continue;
      for (const auto e : r.elements)
        if (std::binary_search(pimy_elements.begin(), pimy_elements.end(), e)) {
          unsupported("betaKinit damping on PressureIndependMultiYield elements (element " + std::to_string(e) +
                      "): OpenSees scales that material's initial tangent per point at the switch to the "
                      "plastic stage, and FemoraX does not keep it");
        }
    }
    for (const auto& s : m.process) {
      const auto* a = std::get_if<AnalysisStep>(&s.step);
      if (a && a->algorithm == Algorithm::ModifiedNewton && a->modified_newton_initial) {
        unsupported("ModifiedNewton -initial (/process/" + s.id +
                    ") with PressureIndependMultiYield: OpenSees scales that material's initial tangent per "
                    "point at the switch to the plastic stage, and FemoraX does not keep it");
      }
    }
  }
  for (const auto& r : m.recorders) {
    if (const auto* v = std::get_if<VtkhdfRecorder>(&r.data); v && !caps.vtkhdf_element_responses) {
      for (const auto& resp : v->responses)
        if (resp == "stress3D6" || resp == "strain3D6") {
          unsupported("VTKHDF response \"" + resp + "\" (/recorders/" + r.name + ") is not available yet (milestone 2)");
        }
    }
  }
  for (const auto& g : m.elements) {
    if (std::find(caps.elements.begin(), caps.elements.end(), g.type) == caps.elements.end()) {
      unsupported(std::string("element type \"") + to_string(g.type) + "\" (/elements/" + g.id +
                  ") is not available in this build");
    }
  }

  for (const auto& s : m.process) {
    const std::string where = "/process/" + s.id;
    if (std::holds_alternative<UpdateMaterialStageStep>(s.step) && !caps.material_stages) {
      unsupported("update_material_stage (" + where + ") is not available yet (milestone 2)");
    }
    if (std::holds_alternative<LoadConstStep>(s.step) && !caps.implicit) {
      unsupported("load_const (" + where + ") is available from milestone 1b");
    }
    const auto* a = std::get_if<AnalysisStep>(&s.step);
    if (!a) continue;

    if (!is_explicit(a->integrator)) {
      if (!caps.implicit) {
        unsupported(std::string(to_string(a->integrator)) + " (" + where +
                    ") is available from milestone 1b");
      }
      continue;
    }

    // Explicit stages.
    if (a->algorithm != Algorithm::Linear) {
      throw ModelError(where + ": " + to_string(a->integrator) +
                       " requires the Linear algorithm, as in OpenSees");
    }
    for (const auto& g : m.elements) {
      if (!g.lumped) {
        unsupported(std::string(to_string(a->integrator)) + " (" + where + ") with element group /elements/" +
                    g.id + ", which uses a consistent mass matrix: explicit stages need lumped "
                    "mass (stdBrick -lumped, or SSPbrick); OpenSees would solve with the full "
                    "mass matrix");
      }
    }
    if (a->integrator == Integrator::CentralDifference) {
      for (const auto& r : m.rayleigh) {
        if (r.betaK != 0.0 || r.betaKinit != 0.0 || r.betaKcomm != 0.0) {
          unsupported("CentralDifference (" + where + ") with stiffness-proportional damping in "
                      "/damping/rayleigh/" + r.id + ": OpenSees solves a non-diagonal system "
                      "here, which FemoraX does not solve yet; use "
                      "ExplicitDifference, or mass-proportional damping only");
        }
      }
    }
  }
}

}  // namespace femorax::io
