#include "io/model_reader.hpp"

#include <algorithm>
#include <charconv>
#include <cmath>

#include <sys/stat.h>
#include <map>
#include <set>
#include <sstream>
#include <unordered_map>
#include <unordered_set>

namespace femorax::io {

// ---------------------------------------------------------------------------
// Enum names
// ---------------------------------------------------------------------------
const char* to_string(ElementType t) {
  switch (t) {
    case ElementType::StdBrick: return "stdBrick";
    case ElementType::SSPbrick: return "SSPbrick";
  }
  return "?";
}

const char* material_type_name(const NDMaterial& m) {
  if (std::holds_alternative<ElasticIsotropicData>(m.data)) return "ElasticIsotropic";
  if (std::holds_alternative<PressureDependMultiYieldData>(m.data)) return "PressureDependMultiYield";
  return "PressureIndependMultiYield";
}

bool has_material_stage(const NDMaterial& m) {
  return std::holds_alternative<PressureDependMultiYieldData>(m.data) ||
         std::holds_alternative<PressureIndependMultiYieldData>(m.data);
}

int node_count(ElementType t) {
  switch (t) {
    case ElementType::StdBrick:
    case ElementType::SSPbrick: return 8;
  }
  return 0;
}

const char* to_string(Integrator i) {
  switch (i) {
    case Integrator::ExplicitDifference: return "ExplicitDifference";
    case Integrator::CentralDifference: return "CentralDifference";
    case Integrator::Newmark: return "Newmark";
    case Integrator::LoadControl: return "LoadControl";
  }
  return "?";
}

bool is_explicit(Integrator i) {
  return i == Integrator::ExplicitDifference || i == Integrator::CentralDifference;
}

std::size_t Model::element_count() const {
  std::size_t n = 0;
  for (const auto& g : elements) n += g.size();
  return n;
}

namespace {

using h5::Group;

constexpr const char* kOpenSeesHint = " Run this model with the OpenSees backend.";

[[noreturn]] void unsupported(const std::string& what, const std::string& supported) {
  throw UnsupportedComponent("unsupported component: " + what + ". FemoraX supports " +
                             supported + "." + kOpenSeesHint);
}

[[noreturn]] void invalid(const std::string& msg) { throw ModelError(msg); }

std::string quoted(const std::string& s) { return "\"" + s + "\""; }

// A group name that is a tag: a canonical positive decimal int32 ("12", not "012" or "+1").
std::int32_t parse_tag(const std::string& name, const std::string& where) {
  std::int32_t v = 0;
  const char* end = name.data() + name.size();
  const auto [p, ec] = std::from_chars(name.data(), end, v);
  if (ec != std::errc() || p != end || v <= 0 || std::to_string(v) != name) {
    invalid(where + "/" + name + ": group name must be a positive integer tag");
  }
  return v;
}

// Children of an ordered container: all-digit names of one width, returned in order.
std::vector<std::string> ordered_children(const Group& g) {
  auto names = g.children();
  for (const auto& n : names) {
    if (n.empty() || !std::all_of(n.begin(), n.end(), [](char c) { return c >= '0' && c <= '9'; }) ||
        n.size() != names.front().size()) {
      invalid(g.item(n) + ": children of " + g.path() +
              " must be zero-padded indices of equal width (000000, 000001, ...)");
    }
  }
  return names;  // already sorted; equal width makes this numeric order
}

// An item a later minor version added: reading it from an older file is an error.
void require_minor(const Model& m, int minor, const std::string& what) {
  if (m.version.minor < minor) {
    invalid(what + " was added in schema 0." + std::to_string(minor) + ", but the file declares " +
            std::to_string(m.version.major) + "." + std::to_string(m.version.minor));
  }
}

Metadata read_meta(Group& g) {
  Metadata m;
  m.femora_tag = g.opt_attr_int("femora_tag");
  m.femora_name = g.opt_attr_string("femora_name");
  return m;
}

void require_finite(double v, const std::string& what) {
  if (!std::isfinite(v)) invalid(what + " must be finite");
}

void require_finite(const std::vector<double>& v, const std::string& what) {
  for (double x : v) require_finite(x, what);
}

// ---------------------------------------------------------------------------
// Sections of the file
// ---------------------------------------------------------------------------
void read_root_attrs(Group& root, Model& m) {
  if (!root.has_attr("schema_version")) invalid("missing / attribute 'schema_version'");
  const auto v = root.attr_int_array("schema_version", 2);
  m.version = {v[0], v[1]};
  const auto r = reader_schema_version;
  if (m.version.major != r.major) {
    invalid("schema version " + std::to_string(m.version.major) + "." +
            std::to_string(m.version.minor) + " is not supported: this reader reads " +
            std::to_string(r.major) + ".x files");
  }
  if (m.version.minor > r.minor || m.version.minor < 0) {
    invalid("schema version " + std::to_string(m.version.major) + "." +
            std::to_string(m.version.minor) + " is newer than this reader (" +
            std::to_string(r.major) + "." + std::to_string(r.minor) +
            "); update FemoraX");
  }
  m.ndm = root.attr_int("ndm");
  if (m.ndm != 3) unsupported("ndm = " + std::to_string(m.ndm), "ndm = 3");
  m.generator = root.attr_string("generator");
  m.model_name = root.attr_string("model_name");
  m.units = root.opt_attr_string("units").value_or("");
}

void read_nodes(Group& root, Model& m) {
  Group g = root.group("nodes");
  auto tags = g.dataset_int("tags");
  auto coords = g.dataset_double("coords", m.ndm);
  auto ndf = g.dataset_int("ndf");
  g.finish();
  if (coords.rows() != tags.rows() || ndf.rows() != tags.rows()) {
    invalid("/nodes: tags, coords and ndf must have the same number of rows (" +
            std::to_string(tags.rows()) + ", " + std::to_string(coords.rows()) + ", " +
            std::to_string(ndf.rows()) + ")");
  }
  if (tags.rows() == 0) invalid("/nodes: the model has no nodes");
  require_finite(coords.data, "/nodes/coords");
  for (std::size_t i = 0; i < ndf.data.size(); ++i) {
    if (ndf.data[i] <= 0) {
      invalid("/nodes/ndf: node " + std::to_string(tags.data[i]) + " has ndf " +
              std::to_string(ndf.data[i]) + "; ndf must be positive");
    }
  }
  m.nodes.tags = std::move(tags.data);
  m.nodes.coords = std::move(coords.data);
  m.nodes.ndf = std::move(ndf.data);
}

// --- Multi-yield soil materials (schema 0.4) -------------------------------------------
// Rules: Femora's checks (femora/components/material/nd/pressure_*_multi_yield.py) and the
// conditions OpenSees' constructors treat as fatal; values OpenSees silently resets are
// rejected too, except where Femora allows them (handled by the material, docs/deviations.md).
double finite_attr(Group& g, const char* name) {
  const double v = g.attr_double(name);
  require_finite(v, g.attr_item(name));
  return v;
}

double finite_attr_or(Group& g, const char* name, double fallback) {
  const auto v = g.opt_attr_double(name);
  if (v) require_finite(*v, g.attr_item(name));
  return v.value_or(fallback);
}

void require(bool ok, Group& g, const char* name, const std::string& rule) {
  if (!ok) invalid(g.attr_item(name) + " " + rule);
}

std::int32_t read_nd(Group& g) {
  const std::int32_t nd = g.attr_int("nd");
  if (nd == 2) unsupported(g.attr_item("nd") + " = 2 (plane strain) in " + g.path(), "nd = 3");
  if (nd != 3) invalid(g.attr_item("nd") + " must be 2 or 3, found " + std::to_string(nd));
  return nd;
}

// noYieldSurf and the optional pairs dataset: positive N, automatic surfaces and no pairs;
// negative N, |N| rows of (gamma, G/Gmax) with gamma > 0 and 0 < G/Gmax <= 1.
void read_surfaces(Group& g, std::int32_t& n, std::vector<double>& pairs, std::int32_t fallback) {
  n = g.opt_attr_int("noYieldSurf").value_or(fallback);
  if (n == 0 || n <= -40 || n >= 40) {
    invalid(g.attr_item("noYieldSurf") + " must be non-zero and less than 40 in magnitude, found " +
            std::to_string(n));
  }
  if (n > 0) {
    if (g.has_child("pairs")) invalid(g.item("pairs") + " is allowed only with a negative noYieldSurf");
    return;
  }
  if (!g.has_child("pairs")) invalid(g.path() + ": noYieldSurf < 0 needs the dataset 'pairs'");
  auto a = g.dataset_double("pairs", 2);
  if (a.rows() != static_cast<std::int64_t>(-n)) {
    invalid(g.item("pairs") + " must have " + std::to_string(-n) + " rows (|noYieldSurf|), found " +
            std::to_string(a.rows()));
  }
  require_finite(a.data, g.item("pairs"));
  for (std::size_t i = 0; i < a.data.size() / 2; ++i) {
    if (!(a.data[2 * i] > 0.0)) invalid(g.item("pairs") + ": every gamma must be positive");
    if (!(a.data[2 * i + 1] > 0.0 && a.data[2 * i + 1] <= 1.0)) {
      invalid(g.item("pairs") + ": every G/Gmax must be in (0, 1]");
    }
  }
  pairs = std::move(a.data);
}

PressureDependMultiYieldData read_pdmy(Group& g) {
  PressureDependMultiYieldData d;
  d.nd = read_nd(g);
  d.rho = finite_attr(g, "rho");
  d.refShearModul = finite_attr(g, "refShearModul");
  d.refBulkModul = finite_attr(g, "refBulkModul");
  d.frictionAng = finite_attr(g, "frictionAng");
  d.peakShearStra = finite_attr(g, "peakShearStra");
  d.refPress = finite_attr(g, "refPress");
  d.pressDependCoe = finite_attr(g, "pressDependCoe");
  d.PTAng = finite_attr(g, "PTAng");
  d.contrac = finite_attr(g, "contrac");
  d.dilat1 = finite_attr(g, "dilat1");
  d.dilat2 = finite_attr(g, "dilat2");
  d.liquefac1 = finite_attr(g, "liquefac1");
  d.liquefac2 = finite_attr(g, "liquefac2");
  d.liquefac3 = finite_attr(g, "liquefac3");
  read_surfaces(g, d.noYieldSurf, d.pairs, 20);
  d.e = finite_attr_or(g, "e", 0.6);
  d.cs1 = finite_attr_or(g, "cs1", 0.9);
  d.cs2 = finite_attr_or(g, "cs2", 0.02);
  d.cs3 = finite_attr_or(g, "cs3", 0.7);
  d.pa = finite_attr_or(g, "pa", 101.0);
  d.c = finite_attr_or(g, "c", 0.3);
  require(d.rho >= 0.0, g, "rho", "must not be negative");
  require(d.refShearModul > 0.0, g, "refShearModul", "must be positive");
  require(d.refBulkModul > 0.0, g, "refBulkModul", "must be positive");
  require(d.frictionAng > 0.0 && d.frictionAng < 90.0, g, "frictionAng", "must be in (0, 90) degrees");
  require(d.PTAng > 0.0 && d.PTAng <= 90.0, g, "PTAng", "must be in (0, 90] degrees");
  require(d.peakShearStra > 0.0, g, "peakShearStra", "must be positive");
  require(d.refPress > 0.0, g, "refPress", "must be positive");
  require(d.pressDependCoe >= 0.0, g, "pressDependCoe", "must not be negative");
  require(d.e >= 0.0, g, "e", "must not be negative");
  require(d.cs1 >= 0.0, g, "cs1", "must not be negative");
  require(d.cs2 >= 0.0, g, "cs2", "must not be negative");
  require(d.cs3 >= 0.0, g, "cs3", "must not be negative");
  require(d.pa > 0.0, g, "pa", "must be positive");
  require(d.c >= 0.0, g, "c", "must not be negative");
  return d;
}

PressureIndependMultiYieldData read_pimy(Group& g) {
  PressureIndependMultiYieldData d;
  d.nd = read_nd(g);
  d.rho = finite_attr(g, "rho");
  d.refShearModul = finite_attr(g, "refShearModul");
  d.refBulkModul = finite_attr(g, "refBulkModul");
  d.cohesi = finite_attr(g, "cohesi");
  d.peakShearStra = finite_attr(g, "peakShearStra");
  d.frictionAng = finite_attr(g, "frictionAng");
  d.refPress = finite_attr(g, "refPress");
  d.pressDependCoe = finite_attr(g, "pressDependCoe");
  if (!g.has_attr("noYieldSurf")) invalid(g.path() + ": missing attribute 'noYieldSurf'");
  read_surfaces(g, d.noYieldSurf, d.pairs, 20);
  require(d.rho >= 0.0, g, "rho", "must not be negative");
  require(d.refShearModul > 0.0, g, "refShearModul", "must be positive");
  require(d.refBulkModul > 0.0, g, "refBulkModul", "must be positive");
  require(d.cohesi >= 0.0, g, "cohesi", "must not be negative");
  require(d.peakShearStra > 0.0, g, "peakShearStra", "must be positive");
  require(d.frictionAng >= 0.0 && d.frictionAng < 90.0, g, "frictionAng", "must be in [0, 90) degrees");
  require(d.frictionAng > 0.0 || d.cohesi > 0.0, g, "cohesi",
          "must be positive when frictionAng is 0 (no strength otherwise)");
  require(d.refPress > 0.0, g, "refPress", "must be positive");
  require(d.pressDependCoe >= 0.0, g, "pressDependCoe", "must not be negative");
  return d;
}

void read_materials(Group& root, Model& m) {
  Group mats = root.group("materials");
  for (const auto& kind : mats.children()) {
    if (kind != "nd") unsupported(mats.item(kind) + " (material kind " + quoted(kind) + ")",
                                  "nD materials (/materials/nd)");
  }
  Group nd = mats.group("nd");
  for (const auto& name : nd.children()) {
    NDMaterial mat;
    mat.tag = parse_tag(name, nd.path());
    Group g = nd.group(name);
    const std::string type = g.attr_string("type");
    if (type == "ElasticIsotropic") {
      ElasticIsotropicData d;
      d.E = g.attr_double("E");
      d.nu = g.attr_double("nu");
      d.rho = g.attr_double("rho");
      require_finite(d.E, g.attr_item("E"));
      require_finite(d.nu, g.attr_item("nu"));
      require_finite(d.rho, g.attr_item("rho"));
      mat.data = d;
    } else if (type == "PressureDependMultiYield") {
      require_minor(m, 4, g.path() + " (type \"PressureDependMultiYield\")");
      const auto d = read_pdmy(g);
      if (d.noYieldSurf > 0 && d.PTAng > d.frictionAng) {
        m.warnings.push_back(g.path() + ": PTAng " + std::to_string(d.PTAng) + " > frictionAng " +
                             std::to_string(d.frictionAng) + "; the phase transformation angle is set to "
                             "the friction angle, as OpenSees does");
      }
      mat.data = d;
    } else if (type == "PressureIndependMultiYield") {
      require_minor(m, 4, g.path() + " (type \"PressureIndependMultiYield\")");
      mat.data = read_pimy(g);
    } else {
      unsupported("nD material type " + quoted(type) + " in " + g.path(),
                  "ElasticIsotropic, PressureDependMultiYield, PressureIndependMultiYield");
    }
    mat.meta = read_meta(g);
    g.finish();
    m.materials.push_back(std::move(mat));
  }
  nd.finish();
  mats.finish();
}

void read_elements(Group& root, Model& m) {
  Group els = root.group("elements");
  for (const auto& id : els.children()) {
    Group g = els.group(id);
    ElementGroup eg;
    eg.id = id;
    const std::string type = g.attr_string("type");
    if (type == "stdBrick") eg.type = ElementType::StdBrick;
    else if (type == "SSPbrick") eg.type = ElementType::SSPbrick;
    else unsupported("element type " + quoted(type) + " in " + g.path(), "stdBrick, SSPbrick");

    eg.material = g.attr_int("material");
    const auto b = g.attr_double_array("body_force", 3);
    require_finite(b, g.attr_item("body_force"));
    eg.body_force = {b[0], b[1], b[2]};
    if (eg.type == ElementType::StdBrick) {
      eg.lumped = g.opt_attr_bool("lumped").value_or(false);
    } else {
      if (g.has_attr("lumped")) {
        invalid(g.attr_item("lumped") + ": SSPbrick always lumps its mass; omit the attribute");
      }
      eg.lumped = true;
    }
    const int nn = node_count(eg.type);
    auto tags = g.dataset_int("tags");
    auto conn = g.dataset_int("connectivity", nn);
    if (conn.rows() != tags.rows()) {
      invalid(g.path() + ": tags and connectivity must have the same number of rows");
    }
    eg.tags = std::move(tags.data);
    eg.connectivity = std::move(conn.data);
    eg.meta = read_meta(g);
    g.finish();
    m.elements.push_back(std::move(eg));
  }
  els.finish();
  if (m.elements.empty()) invalid("/elements: the model has no element groups");
}

void read_mass(Group& root, Model& m) {
  auto g = root.optional_group("mass");
  if (!g) return;
  auto node = g->dataset_int("node");
  auto dof = g->dataset_int("dof");
  auto value = g->dataset_double("value");
  g->finish();
  if (dof.rows() != node.rows() || value.rows() != node.rows()) {
    invalid("/mass: node, dof and value must have the same number of rows");
  }
  require_finite(value.data, "/mass/value");
  m.mass = {std::move(node.data), std::move(dof.data), std::move(value.data)};
}

void read_constraints(Group& root, Model& m) {
  auto c = root.optional_group("constraints");
  if (!c) return;
  for (const auto& kind : c->children()) {
    if (kind != "sp" && kind != "mp") invalid("unknown item " + c->item(kind));
  }
  if (auto sp = c->optional_group("sp")) {
    for (const auto& t : sp->children()) {
      if (t != "fix") unsupported("SP constraint " + quoted(t) + " (" + sp->item(t) + ")", "fix");
    }
    if (auto fix = sp->optional_group("fix")) {
      auto node = fix->dataset_int("node");
      auto dof = fix->dataset_int("dof");
      fix->finish();
      if (dof.rows() != node.rows()) invalid("/constraints/sp/fix: node and dof differ in length");
      m.fix = {std::move(node.data), std::move(dof.data)};
    }
    sp->finish();
  }
  if (auto mp = c->optional_group("mp")) {
    for (const auto& t : mp->children()) {
      if (t != "equal_dof") {
        unsupported("MP constraint " + quoted(t) + " (" + mp->item(t) + ")",
                    "equal_dof (equalDOF and the laminar boundary)");
      }
    }
    if (auto eq = mp->optional_group("equal_dof")) {
      auto retained = eq->dataset_int("retained");
      auto constrained = eq->dataset_int("constrained");
      auto dof = eq->dataset_int("dof");
      std::vector<std::int32_t> command;
      if (eq->has_child("command")) {
        require_minor(m, 2, eq->item("command"));
        command = eq->dataset_int("command").data;
      }
      eq->finish();
      if (constrained.rows() != retained.rows() || dof.rows() != retained.rows()) {
        invalid("/constraints/mp/equal_dof: retained, constrained and dof differ in length");
      }
      if (!command.empty() && static_cast<std::int64_t>(command.size()) != retained.rows()) {
        invalid("/constraints/mp/equal_dof: command must have one entry per row");
      }
      m.equal_dof = {std::move(retained.data), std::move(constrained.data), std::move(dof.data),
                     std::move(command)};
    }
    mp->finish();
  }
  c->finish();
}

void read_damping(Group& root, Model& m) {
  auto d = root.optional_group("damping");
  if (!d) return;
  for (const auto& t : d->children()) {
    if (t != "rayleigh" && t != "uniform") {
      unsupported("damping type " + quoted(t) + " (" + d->item(t) + ")",
                  "per-region Rayleigh damping (/damping/rayleigh) and Uniform damping (/damping/uniform)");
    }
  }
  if (auto r = d->optional_group("rayleigh")) {
    for (const auto& id : ordered_children(*r)) {
      Group g = r->group(id);
      RayleighRegion reg;
      reg.id = id;
      reg.alphaM = g.attr_double("alphaM");
      reg.betaK = g.attr_double("betaK");
      reg.betaKinit = g.attr_double("betaKinit");
      reg.betaKcomm = g.attr_double("betaKcomm");
      for (double v : {reg.alphaM, reg.betaK, reg.betaKinit, reg.betaKcomm}) {
        require_finite(v, g.path() + " coefficients");
      }
      reg.elements = g.dataset_int("elements").data;
      reg.meta = read_meta(g);
      g.finish();
      m.rayleigh.push_back(std::move(reg));
    }
    r->finish();
  }
  if (auto u = d->optional_group("uniform")) {
    require_minor(m, 7, u->path() + " (Uniform damping)");
    for (const auto& id : ordered_children(*u)) {
      Group g = u->group(id);
      UniformDampingRegion reg;
      reg.id = id;
      reg.ratio = g.attr_double("ratio");
      reg.freq1 = g.attr_double("freq1");
      reg.freq2 = g.attr_double("freq2");
      if (auto v = g.opt_attr_double("activate_time")) reg.activate_time = *v;
      if (auto v = g.opt_attr_double("deactivate_time")) reg.deactivate_time = *v;
      reg.factor_series = g.opt_attr_int("factor_series");
      if (auto v = g.opt_attr_string("fit")) reg.fit = *v;
      const std::string where = g.path();
      for (double v : {reg.ratio, reg.freq1, reg.freq2, reg.activate_time, reg.deactivate_time}) {
        require_finite(v, where + " attributes");
      }
      if (!(reg.ratio > 0.0)) invalid(where + ": attribute 'ratio' must be positive");
      if (!(reg.freq1 > 0.0 && reg.freq2 > reg.freq1)) {
        invalid(where + ": the band needs 0 < freq1 < freq2, found " + std::to_string(reg.freq1) + " and " +
                std::to_string(reg.freq2));
      }
      if (!(reg.activate_time < reg.deactivate_time)) {
        invalid(where + ": activate_time must be before deactivate_time");
      }
      if (reg.fit != "femorax" && reg.fit != "opensees") {
        invalid(where + ": attribute 'fit' must be \"femorax\" or \"opensees\", found " + quoted(reg.fit));
      }
      reg.elements = g.dataset_int("elements").data;
      reg.meta = read_meta(g);
      g.finish();
      m.uniform.push_back(std::move(reg));
    }
    u->finish();
  }
  d->finish();
}

void read_time_series(Group& root, Model& m) {
  auto ts = root.optional_group("time_series");
  if (!ts) return;
  for (const auto& name : ts->children()) {
    const std::int32_t tag = parse_tag(name, ts->path());
    Group g = ts->group(name);
    const std::string type = g.attr_string("type");
    if (type != "Path") unsupported("time series type " + quoted(type) + " in " + g.path(), "Path");
    PathSeries p;
    p.tag = tag;
    p.factor = g.opt_attr_double("factor").value_or(1.0);
    p.dt = g.opt_attr_double("dt");
    auto time = g.opt_dataset_double("time");
    p.values = g.dataset_double("values").data;
    p.use_last = g.opt_attr_bool("use_last").value_or(false);
    p.prepend_zero = g.opt_attr_bool("prepend_zero").value_or(false);
    p.meta = read_meta(g);
    g.finish();

    if (p.dt.has_value() == time.has_value()) {
      invalid(g.path() + ": give exactly one of attribute 'dt' and dataset 'time'");
    }
    if (p.values.empty()) invalid(g.path() + "/values: the series is empty");
    require_finite(p.factor, g.attr_item("factor"));
    require_finite(p.values, g.item("values"));
    if (p.dt) {
      if (!(*p.dt > 0.0) || !std::isfinite(*p.dt)) invalid(g.attr_item("dt") + " must be positive");
    } else {
      p.time = std::move(time->data);
      if (p.time.size() != p.values.size()) {
        invalid(g.path() + ": time and values differ in length (" +
                std::to_string(p.time.size()) + " and " + std::to_string(p.values.size()) + ")");
      }
      require_finite(p.time, g.item("time"));
      for (std::size_t i = 1; i < p.time.size(); ++i) {
        if (p.time[i] < p.time[i - 1]) {
          invalid(g.item("time") + " must not decrease (index " + std::to_string(i) + ")");
        }
      }
    }
    m.time_series.push_back(std::move(p));
  }
  ts->finish();
}

H5DrmPattern read_h5drm(Group& g, std::int32_t tag, const Model& m, const std::string& model_file) {
  require_minor(m, 6, g.path() + " (type \"H5DRM\")");
  H5DrmPattern d;
  d.tag = tag;
  d.file = g.attr_string("file");
  if (d.file.empty()) invalid(g.attr_item("file") + " is empty");
  // (std::filesystem would bring std::quoted into this file's overload sets.)
  if (d.file.front() == '/') {
    d.path = d.file;
  } else {
    const auto slash = model_file.find_last_of('/');
    d.path = slash == std::string::npos ? d.file : model_file.substr(0, slash + 1) + d.file;
  }
  struct stat st {};
  if (::stat(d.path.c_str(), &st) != 0 || !S_ISREG(st.st_mode)) {
    invalid(g.attr_item("file") + ": the DRM file " + quoted(d.path) + " does not exist");
  }
  d.factor = g.attr_double("factor");
  d.crd_scale = g.attr_double("crd_scale");
  d.distance_tolerance = g.attr_double("distance_tolerance");
  const std::int32_t t = g.attr_int("do_coordinate_transformation");
  require_finite(d.factor, g.attr_item("factor"));
  if (!(std::isfinite(d.crd_scale) && d.crd_scale != 0.0)) invalid(g.attr_item("crd_scale") + " must be finite and non-zero");
  if (!(d.distance_tolerance > 0.0 && std::isfinite(d.distance_tolerance))) {
    invalid(g.attr_item("distance_tolerance") + " must be positive");
  }
  if (t != 0 && t != 1) invalid(g.attr_item("do_coordinate_transformation") + " must be 0 or 1");
  d.do_coordinate_transformation = t == 1;
  if (g.has_child("transform") || t == 1) {
    auto a = g.dataset_double("transform", 3);
    if (a.rows() != 3) invalid(g.item("transform") + " must be 3 x 3");
    require_finite(a.data, g.item("transform"));
    for (int i = 0; i < 3; ++i)
      for (int j = 0; j < 3; ++j) d.T[i][j] = a.data[static_cast<std::size_t>(3 * i + j)];
  }
  if (g.has_child("origin") || t == 1) {
    auto a = g.dataset_double("origin");
    if (a.data.size() != 3) invalid(g.item("origin") + " must have 3 values");
    require_finite(a.data, g.item("origin"));
    for (int i = 0; i < 3; ++i) d.x0[i] = a.data[static_cast<std::size_t>(i)];
  }
  d.meta = read_meta(g);
  return d;
}

void read_patterns(Group& root, Model& m, const std::string& model_file) {
  auto ps = root.optional_group("patterns");
  if (!ps) return;
  for (const auto& name : ps->children()) {
    const std::int32_t tag = parse_tag(name, ps->path());
    Group g = ps->group(name);
    const std::string type = g.attr_string("type");
    if (type == "H5DRM") {
      m.drm_patterns.push_back(read_h5drm(g, tag, m, model_file));
      g.finish();
      continue;
    }
    if (type != "UniformExcitation") {
      unsupported("pattern type " + quoted(type) + " in " + g.path(), "UniformExcitation, H5DRM");
    }
    UniformExcitation u;
    u.tag = tag;
    u.dof = g.attr_int("dof");
    u.accel = g.attr_int("accel");
    u.vel0 = g.opt_attr_double("vel0").value_or(0.0);
    u.factor = g.opt_attr_double("factor").value_or(1.0);
    require_finite(u.vel0, g.attr_item("vel0"));
    require_finite(u.factor, g.attr_item("factor"));
    u.meta = read_meta(g);
    g.finish();
    m.patterns.push_back(std::move(u));
  }
  ps->finish();
}

NodeResponse parse_node_response(const std::string& s, const std::string& where) {
  if (s == "disp") return NodeResponse::Disp;
  if (s == "vel") return NodeResponse::Vel;
  if (s == "accel") return NodeResponse::Accel;
  unsupported("Node recorder response " + quoted(s) + " in " + where, "disp, vel, accel");
}

void read_recorders(Group& root, Model& m) {
  auto rs = root.optional_group("recorders");
  if (!rs) return;
  for (const auto& name : rs->children()) {
    Group g = rs->group(name);
    Recorder rec;
    rec.name = name;
    const std::string type = g.attr_string("type");
    if (type == "Node") {
      NodeRecorder r;
      r.file = g.attr_string("file");
      r.response = parse_node_response(g.attr_string("response"), g.path());
      r.time = g.opt_attr_bool("time").value_or(false);
      r.dT = g.opt_attr_double("dT");
      r.precision = g.opt_attr_int("precision").value_or(6);
      r.close_on_write = g.opt_attr_bool("close_on_write").value_or(false);
      r.nodes = g.dataset_int("nodes").data;
      r.dofs = g.dataset_int("dofs").data;
      if (auto ts = g.opt_dataset_int("time_series")) r.time_series = std::move(ts->data);
      if (r.file.empty()) invalid(g.attr_item("file") + " is empty");
      if (r.dT && (!(*r.dT >= 0.0) || !std::isfinite(*r.dT))) {
        invalid(g.attr_item("dT") + " must be non-negative");
      }
      if (r.precision <= 0 || r.precision > 17) invalid(g.attr_item("precision") + " must be 1 to 17");
      if (r.nodes.empty()) invalid(g.item("nodes") + " is empty");
      if (r.dofs.empty()) invalid(g.item("dofs") + " is empty");
      if (!r.time_series.empty() && r.time_series.size() != r.dofs.size()) {
        invalid(g.item("time_series") + " needs one tag per DOF (" +
                std::to_string(r.dofs.size()) + ")");
      }
      rec.data = std::move(r);
    } else if (type == "VTKHDF") {
      VtkhdfRecorder r;
      r.file = g.attr_string("file");
      r.dT = g.opt_attr_double("dT");
      r.rTolDt = g.opt_attr_double("rTolDt");
      std::istringstream words(g.attr_string("responses"));
      for (std::string w; words >> w;) {
        const bool element_response = w == "stress3D6" || w == "strain3D6";
        if (w != "disp" && w != "vel" && w != "accel" && !element_response) {
          unsupported("VTKHDF response " + quoted(w) + " in " + g.path(),
                      "disp, vel, accel, stress3D6, strain3D6");
        }
        if (element_response) require_minor(m, 4, g.attr_item("responses") + " " + quoted(w));
        if (std::find(r.responses.begin(), r.responses.end(), w) != r.responses.end()) {
          invalid(g.attr_item("responses") + " lists " + quoted(w) + " twice");
        }
        r.responses.push_back(w);
      }
      r.region = g.opt_attr_int("region");
      if (g.has_child("elements")) {
        require_minor(m, 1, g.item("elements"));
        r.elements = g.dataset_int("elements").data;
        if (r.elements.empty()) invalid(g.item("elements") + " is empty; omit it to record the whole model");
      }
      if (r.file.empty()) invalid(g.attr_item("file") + " is empty");
      if (r.responses.empty()) invalid(g.attr_item("responses") + " is empty");
      if (r.region) {
        unsupported("VTKHDF recorder -region (" + g.path() + ")",
                    "VTKHDF recorders of the whole model or of an 'elements' list (schema 0.1)");
      }
      rec.data = std::move(r);
    } else {
      unsupported("recorder type " + quoted(type) + " in " + g.path(), "Node, VTKHDF");
    }
    rec.meta = read_meta(g);
    g.finish();
    m.recorders.push_back(std::move(rec));
  }
  rs->finish();
}

AnalysisStep read_analysis(Group& g, const Model& m) {
  AnalysisStep a;
  a.name = g.attr_string("name");
  const std::string type = g.attr_string("analysis_type");
  if (type == "Transient") {
    a.type = AnalysisType::Transient;
  } else if (type == "Static") {
    require_minor(m, 3, g.attr_item("analysis_type") + " \"Static\"");
    a.type = AnalysisType::Static;
  } else {
    unsupported("analysis type " + quoted(type) + " in " + g.path(), "Transient, Static");
  }
  const bool is_static = a.type == AnalysisType::Static;
  if (is_static) {
    if (g.has_attr("dt")) invalid(g.attr_item("dt") + " applies only to Transient analyses");
    if (g.has_attr("final_time")) invalid(g.attr_item("final_time") + " applies only to Transient analyses");
    a.num_steps = g.attr_int("num_steps");
    if (*a.num_steps < 0) invalid(g.attr_item("num_steps") + " must not be negative");
  } else {
    a.dt = g.attr_double("dt");
    if (!(a.dt > 0.0) || !std::isfinite(a.dt)) invalid(g.attr_item("dt") + " must be positive");
    a.num_steps = g.opt_attr_int("num_steps");
    a.final_time = g.opt_attr_double("final_time");
    if (a.num_steps.has_value() == a.final_time.has_value()) {
      invalid(g.path() + ": give exactly one of 'num_steps' and 'final_time'");
    }
    if (a.num_steps && *a.num_steps < 0) invalid(g.attr_item("num_steps") + " must not be negative");
    if (a.final_time) require_finite(*a.final_time, g.attr_item("final_time"));
  }

  const std::string integ = g.attr_string("integrator");
  if (integ == "ExplicitDifference") a.integrator = Integrator::ExplicitDifference;
  else if (integ == "CentralDifference") a.integrator = Integrator::CentralDifference;
  else if (integ == "Newmark") a.integrator = Integrator::Newmark;
  else if (integ == "LoadControl") a.integrator = Integrator::LoadControl;
  else unsupported("integrator " + quoted(integ) + " in " + g.path(),
                   "ExplicitDifference, CentralDifference, Newmark, LoadControl");
  if (is_static != (a.integrator == Integrator::LoadControl)) {
    invalid(g.path() + ": " + (is_static ? "a Static analysis needs the LoadControl integrator"
                                         : "the LoadControl integrator needs a Static analysis"));
  }
  if (a.integrator == Integrator::Newmark) {
    a.gamma = g.attr_double("gamma");
    a.beta = g.attr_double("beta");
    a.newmark_form = g.opt_attr_string("newmark_form").value_or("D");
    if (a.newmark_form != "D" && a.newmark_form != "V" && a.newmark_form != "A") {
      invalid(g.attr_item("newmark_form") + " must be \"D\", \"V\" or \"A\"");
    }
  } else {
    for (const char* k : {"gamma", "beta", "newmark_form"}) {
      if (g.has_attr(k)) invalid(g.attr_item(k) + " applies only to the Newmark integrator");
    }
  }
  if (a.integrator == Integrator::LoadControl) {
    a.load_increment = g.attr_double("load_increment");
    require_finite(a.load_increment, g.attr_item("load_increment"));
    a.load_num_iter = g.opt_attr_int("num_iter").value_or(1);
    if (a.load_num_iter < 1) invalid(g.attr_item("num_iter") + " must be at least 1");
    a.load_min_increment = g.opt_attr_double("min_increment");
    a.load_max_increment = g.opt_attr_double("max_increment");
    if (a.load_min_increment) require_finite(*a.load_min_increment, g.attr_item("min_increment"));
    if (a.load_max_increment) require_finite(*a.load_max_increment, g.attr_item("max_increment"));
  } else {
    for (const char* k : {"load_increment", "num_iter", "min_increment", "max_increment"}) {
      if (g.has_attr(k)) invalid(g.attr_item(k) + " applies only to the LoadControl integrator");
    }
  }

  const std::string alg = g.attr_string("algorithm");
  if (alg == "Linear") a.algorithm = Algorithm::Linear;
  else if (alg == "Newton") a.algorithm = Algorithm::Newton;
  else if (alg == "ModifiedNewton") {
    require_minor(m, 3, g.attr_item("algorithm") + " \"ModifiedNewton\"");
    a.algorithm = Algorithm::ModifiedNewton;
  } else {
    unsupported("algorithm " + quoted(alg) + " in " + g.path(), "Linear, Newton, ModifiedNewton");
  }
  if (a.algorithm == Algorithm::ModifiedNewton) {
    a.modified_newton_initial = g.opt_attr_bool("modified_newton_initial").value_or(false);
    a.modified_newton_factor_once = g.opt_attr_bool("modified_newton_factor_once").value_or(false);
  } else {
    for (const char* k : {"modified_newton_initial", "modified_newton_factor_once"}) {
      if (g.has_attr(k)) invalid(g.attr_item(k) + " applies only to the ModifiedNewton algorithm");
    }
  }

  const std::string handler = g.attr_string("constraint_handler");
  if (handler == "Transformation") a.constraint_handler = ConstraintHandler::Transformation;
  else if (handler == "Plain") a.constraint_handler = ConstraintHandler::Plain;
  else unsupported("constraint handler " + quoted(handler) + " in " + g.path(), "Transformation, Plain");

  a.numberer = g.attr_string("numberer");
  a.system = g.attr_string("system");

  if (auto t = g.opt_attr_string("test_type")) {
    if (*t == "NormUnbalance") a.test_type = TestType::NormUnbalance;
    else if (*t == "NormDispIncr") a.test_type = TestType::NormDispIncr;
    else if (*t == "EnergyIncr") a.test_type = TestType::EnergyIncr;
    else unsupported("convergence test " + quoted(*t) + " in " + g.path(),
                     "NormUnbalance, NormDispIncr, EnergyIncr");
  }
  a.test_tol = g.opt_attr_double("test_tol");
  a.test_max_iter = g.opt_attr_int("test_max_iter");
  a.max_retries = g.opt_attr_int("max_retries");
  a.num_sublevels = g.opt_attr_int("num_sublevels");
  a.num_substeps = g.opt_attr_int("num_substeps");
  // Femora's recovery (defined in 0.5; the attributes exist since 0.0): Femora's own checks.
  if (a.max_retries && *a.max_retries < 0) invalid(g.attr_item("max_retries") + " must be non-negative");
  if (a.num_sublevels.has_value() != a.num_substeps.has_value()) {
    invalid(g.path() + ": num_sublevels and num_substeps must be given together");
  }
  if (a.num_sublevels && *a.num_sublevels < 1) invalid(g.attr_item("num_sublevels") + " must be at least 1");
  if (a.num_substeps && *a.num_substeps < 2) invalid(g.attr_item("num_substeps") + " must be at least 2");
  if (a.num_sublevels && a.type == AnalysisType::Static) {
    invalid(g.path() + ": num_sublevels and num_substeps apply only to Transient analyses");
  }
  a.initialize = g.opt_attr_bool("initialize").value_or(false);
  a.allow_unstable_dt = g.opt_attr_bool("allow_unstable_dt").value_or(false);

  // FemoraX solver options (0.3).
  for (const char* k : {"solver_tolerance", "solver_max_iterations", "preconditioner"}) {
    if (g.has_attr(k)) require_minor(m, 3, g.attr_item(k));
  }
  a.solver_tolerance = g.opt_attr_double("solver_tolerance");
  if (a.solver_tolerance && !(*a.solver_tolerance > 0.0 && *a.solver_tolerance < 1.0)) {
    invalid(g.attr_item("solver_tolerance") + " must be in (0, 1)");
  }
  a.solver_max_iterations = g.opt_attr_int("solver_max_iterations");
  if (a.solver_max_iterations && *a.solver_max_iterations < 1) {
    invalid(g.attr_item("solver_max_iterations") + " must be at least 1");
  }
  if (auto pc = g.opt_attr_string("preconditioner")) {
    if (*pc == "auto") a.preconditioner = Preconditioner::Auto;
    else if (*pc == "jacobi") a.preconditioner = Preconditioner::Jacobi;
    else unsupported("preconditioner " + quoted(*pc) + " in " + g.path(), "auto, jacobi");
  }
  return a;
}

void read_process(Group& root, Model& m) {
  Group p = root.group("process");
  for (const auto& id : ordered_children(p)) {
    Group g = p.group(id);
    ProcessStep s;
    s.id = id;
    s.label = g.opt_attr_string("label").value_or("");
    const std::string kind = g.attr_string("kind");
    if (kind == "analysis") s.step = read_analysis(g, m);
    else if (kind == "pattern") s.step = PatternStep{g.attr_int("tag")};
    else if (kind == "recorder") s.step = RecorderStep{g.attr_string("name")};
    else if (kind == "set_time") {
      const double t = g.attr_double("time");
      require_finite(t, g.attr_item("time"));
      s.step = SetTimeStep{t};
    } else if (kind == "load_const") s.step = LoadConstStep{g.opt_attr_double("time")};
    else if (kind == "wipe_analysis") s.step = WipeAnalysisStep{};
    else if (kind == "update_material_stage") {
      require_minor(m, 4, g.path() + " (kind \"update_material_stage\")");
      UpdateMaterialStageStep u;
      u.stage = g.attr_int("stage");
      if (u.stage != 0 && u.stage != 1) {
        invalid(g.attr_item("stage") + " must be 0 (elastic) or 1 (plastic), found " + std::to_string(u.stage));
      }
      u.materials = g.dataset_int("materials").data;
      s.step = std::move(u);
    } else unsupported("process step " + quoted(kind) + " in " + g.path(),
                     "analysis, pattern, recorder, set_time, load_const, wipe_analysis, update_material_stage");
    g.finish();
    m.process.push_back(std::move(s));
  }
  p.finish();
}

// ---------------------------------------------------------------------------
// Cross-reference checks
// ---------------------------------------------------------------------------
template <class Key>
std::string dup_message(const std::string& what, Key k) {
  std::ostringstream o;
  o << what << " " << k << " appears more than once";
  return o.str();
}

void validate(Model& m) {
  // Nodes: unique positive tags.
  std::unordered_map<std::int32_t, std::int32_t> node_ndf;
  node_ndf.reserve(m.nodes.size());
  for (std::size_t i = 0; i < m.nodes.size(); ++i) {
    const auto tag = m.nodes.tags[i];
    if (tag <= 0) invalid("/nodes/tags: node tag " + std::to_string(tag) + " is not positive");
    if (!node_ndf.emplace(tag, m.nodes.ndf[i]).second) invalid(dup_message("/nodes/tags: node", tag));
  }
  auto ndf_of = [&](std::int32_t node, const std::string& where) {
    const auto it = node_ndf.find(node);
    if (it == node_ndf.end()) invalid(where + ": node " + std::to_string(node) + " does not exist");
    return it->second;
  };
  auto check_dof = [&](std::int32_t node, std::int32_t dof, const std::string& where) {
    const int ndf = ndf_of(node, where);
    if (dof < 1 || dof > ndf) {
      invalid(where + ": DOF " + std::to_string(dof) + " is out of range for node " +
              std::to_string(node) + " (ndf " + std::to_string(ndf) + ")");
    }
  };

  // Materials.
  std::unordered_set<std::int32_t> materials;
  for (const auto& mat : m.materials) materials.insert(mat.tag);

  // Elements: material exists, nodes exist with the element's DOFs, unique tags.
  std::unordered_set<std::int32_t> element_tags;
  for (const auto& g : m.elements) {
    const std::string where = "/elements/" + g.id;
    if (!materials.count(g.material)) {
      invalid(where + ": material " + std::to_string(g.material) + " does not exist");
    }
    const int nn = node_count(g.type);
    for (std::size_t e = 0; e < g.size(); ++e) {
      const auto tag = g.tags[e];
      if (tag <= 0) invalid(where + "/tags: element tag " + std::to_string(tag) + " is not positive");
      if (!element_tags.insert(tag).second) invalid(dup_message(where + ": element", tag));
      for (int k = 0; k < nn; ++k) {
        const auto node = g.connectivity[e * nn + k];
        const int ndf = ndf_of(node, where + "/connectivity (element " + std::to_string(tag) + ")");
        if (ndf != 3) {
          invalid(where + ": element " + std::to_string(tag) + " (" + to_string(g.type) +
                  ") needs 3 DOFs at node " + std::to_string(node) + ", which has ndf " +
                  std::to_string(ndf));
        }
      }
      // A node repeated within one element is a degenerate element.
      std::set<std::int32_t> seen(g.connectivity.begin() + e * nn, g.connectivity.begin() + (e + 1) * nn);
      if (static_cast<int>(seen.size()) != nn) {
        invalid(where + ": element " + std::to_string(tag) + " uses a node more than once");
      }
    }
  }

  // Nodal mass, fix, equalDOF: one row per DOF, no duplicates.
  {
    std::set<std::pair<std::int32_t, std::int32_t>> seen;
    for (std::size_t i = 0; i < m.mass.node.size(); ++i) {
      check_dof(m.mass.node[i], m.mass.dof[i], "/mass");
      if (!seen.emplace(m.mass.node[i], m.mass.dof[i]).second) {
        invalid("/mass: node " + std::to_string(m.mass.node[i]) + " DOF " +
                std::to_string(m.mass.dof[i]) + " appears more than once");
      }
    }
  }
  {
    std::set<std::pair<std::int32_t, std::int32_t>> seen;
    for (std::size_t i = 0; i < m.fix.node.size(); ++i) {
      check_dof(m.fix.node[i], m.fix.dof[i], "/constraints/sp/fix");
      if (!seen.emplace(m.fix.node[i], m.fix.dof[i]).second) {
        invalid("/constraints/sp/fix: node " + std::to_string(m.fix.node[i]) + " DOF " +
                std::to_string(m.fix.dof[i]) + " appears more than once");
      }
    }
  }
  {
    const std::string where = "/constraints/mp/equal_dof";
    std::set<std::pair<std::int32_t, std::int32_t>> constrained;
    std::unordered_map<std::int32_t, std::pair<std::int32_t, std::int32_t>> nodes_of_command;
    for (std::size_t i = 0; i < m.equal_dof.command.size(); ++i) {
      const auto cmd = m.equal_dof.command[i];
      if (cmd < 0) invalid(where + "/command: command index " + std::to_string(cmd) + " is negative");
      const auto pair = std::make_pair(m.equal_dof.retained[i], m.equal_dof.constrained[i]);
      const auto [it, fresh] = nodes_of_command.emplace(cmd, pair);
      if (!fresh && it->second != pair) {
        invalid(where + "/command: command " + std::to_string(cmd) + " has rows with different " +
                "retained or constrained nodes; one equalDOF command ties one node pair");
      }
    }
    for (std::size_t i = 0; i < m.equal_dof.retained.size(); ++i) {
      const auto r = m.equal_dof.retained[i], c = m.equal_dof.constrained[i], d = m.equal_dof.dof[i];
      check_dof(r, d, where);
      check_dof(c, d, where);
      if (r == c) invalid(where + ": node " + std::to_string(r) + " is tied to itself");
      // OpenSees allows one MP constraint per constrained DOF.
      if (!constrained.emplace(c, d).second) {
        invalid(where + ": node " + std::to_string(c) + " DOF " + std::to_string(d) +
                " is constrained more than once");
      }
    }
  }

  // Rayleigh regions: elements exist; overlaps are allowed (last wins) with a warning.
  {
    std::unordered_map<std::int32_t, std::size_t> owner;
    std::map<std::pair<std::size_t, std::size_t>, std::size_t> overlaps;
    for (std::size_t r = 0; r < m.rayleigh.size(); ++r) {
      const std::string where = "/damping/rayleigh/" + m.rayleigh[r].id;
      std::unordered_set<std::int32_t> in_region;
      for (const auto e : m.rayleigh[r].elements) {
        if (!element_tags.count(e)) {
          invalid(where + "/elements: element " + std::to_string(e) + " does not exist");
        }
        if (!in_region.insert(e).second) invalid(dup_message(where + "/elements: element", e));
        const auto [it, fresh] = owner.emplace(e, r);
        if (!fresh) {
          ++overlaps[{it->second, r}];
          it->second = r;
        }
      }
    }
    for (const auto& [pair, count] : overlaps) {
      m.warnings.push_back("damping regions /damping/rayleigh/" + m.rayleigh[pair.first].id +
                           " and /damping/rayleigh/" + m.rayleigh[pair.second].id + " share " +
                           std::to_string(count) + " element(s); the later region's " +
                           "coefficients apply, as in OpenSees");
    }
  }

  // Uniform regions: elements exist, the factor's series exists; overlaps as for Rayleigh.
  {
    std::unordered_map<std::int32_t, std::size_t> owner;
    std::map<std::pair<std::size_t, std::size_t>, std::size_t> overlaps;
    std::unordered_set<std::int32_t> known_series;
    for (const auto& s : m.time_series) known_series.insert(s.tag);
    for (std::size_t r = 0; r < m.uniform.size(); ++r) {
      const std::string where = "/damping/uniform/" + m.uniform[r].id;
      if (m.uniform[r].factor_series && !known_series.count(*m.uniform[r].factor_series)) {
        invalid(where + ": factor_series: time series " + std::to_string(*m.uniform[r].factor_series) +
                " does not exist");
      }
      std::unordered_set<std::int32_t> in_region;
      for (const auto e : m.uniform[r].elements) {
        if (!element_tags.count(e)) {
          invalid(where + "/elements: element " + std::to_string(e) + " does not exist");
        }
        if (!in_region.insert(e).second) invalid(dup_message(where + "/elements: element", e));
        const auto [it, fresh] = owner.emplace(e, r);
        if (!fresh) {
          ++overlaps[{it->second, r}];
          it->second = r;
        }
      }
    }
    for (const auto& [pair, count] : overlaps) {
      m.warnings.push_back("damping regions /damping/uniform/" + m.uniform[pair.first].id +
                           " and /damping/uniform/" + m.uniform[pair.second].id + " share " +
                           std::to_string(count) + " element(s); the later region's damping " +
                           "applies, as in OpenSees");
    }
  }

  // Time series, patterns.
  std::unordered_set<std::int32_t> series;
  for (const auto& s : m.time_series) series.insert(s.tag);
  int max_ndf = 0;
  for (auto v : m.nodes.ndf) max_ndf = std::max(max_ndf, static_cast<int>(v));
  std::unordered_set<std::int32_t> patterns;
  for (const auto& p : m.drm_patterns) patterns.insert(p.tag);
  for (const auto& p : m.patterns) {
    const std::string where = "/patterns/" + std::to_string(p.tag);
    patterns.insert(p.tag);
    if (!series.count(p.accel)) {
      invalid(where + ": time series " + std::to_string(p.accel) + " does not exist");
    }
    if (p.dof < 1 || p.dof > max_ndf) {
      invalid(where + ": dof " + std::to_string(p.dof) + " is out of range (1 to " +
              std::to_string(max_ndf) + ")");
    }
  }

  // Recorders.
  std::unordered_set<std::string> recorders;
  for (const auto& rec : m.recorders) {
    recorders.insert(rec.name);
    if (const auto* r = std::get_if<NodeRecorder>(&rec.data)) {
      const std::string where = "/recorders/" + rec.name;
      for (const auto n : r->nodes)
        for (const auto d : r->dofs) check_dof(n, d, where);
      for (const auto t : r->time_series) {
        if (!series.count(t)) invalid(where + ": time series " + std::to_string(t) + " does not exist");
      }
    } else if (const auto* v = std::get_if<VtkhdfRecorder>(&rec.data)) {
      const std::string where = "/recorders/" + rec.name + "/elements";
      std::unordered_set<std::int32_t> seen;
      for (const auto e : v->elements) {
        if (!element_tags.count(e)) invalid(where + ": element " + std::to_string(e) + " does not exist");
        if (!seen.insert(e).second) invalid(dup_message(where + ": element", e));
      }
    }
  }

  // Process: references, and each pattern and recorder activated at most once.
  std::unordered_set<std::int32_t> active_patterns;
  std::unordered_set<std::string> active_recorders;
  for (const auto& s : m.process) {
    const std::string where = "/process/" + s.id;
    if (const auto* p = std::get_if<PatternStep>(&s.step)) {
      if (!patterns.count(p->tag)) invalid(where + ": pattern " + std::to_string(p->tag) + " does not exist");
      if (!active_patterns.insert(p->tag).second) {
        invalid(where + ": pattern " + std::to_string(p->tag) + " is activated twice");
      }
    } else if (const auto* r = std::get_if<RecorderStep>(&s.step)) {
      if (!recorders.count(r->name)) invalid(where + ": recorder " + quoted(r->name) + " does not exist");
      if (!active_recorders.insert(r->name).second) {
        invalid(where + ": recorder " + quoted(r->name) + " is activated twice");
      }
    } else if (const auto* u = std::get_if<UpdateMaterialStageStep>(&s.step)) {
      std::unordered_set<std::int32_t> seen;
      for (const auto tag : u->materials) {
        const auto it = std::find_if(m.materials.begin(), m.materials.end(),
                                     [&](const NDMaterial& x) { return x.tag == tag; });
        if (it == m.materials.end()) invalid(where + "/materials: material " + std::to_string(tag) + " does not exist");
        if (!has_material_stage(*it)) {
          invalid(where + "/materials: material " + std::to_string(tag) + " (" + material_type_name(*it) +
                  ") has no material stage (updateMaterialStage applies to the multi-yield soil materials)");
        }
        if (!seen.insert(tag).second) invalid(dup_message(where + "/materials: material", tag));
      }
    }
  }
}

}  // namespace

Model read_model(const std::string& filename) {
  h5::File file(filename);
  Group root = file.root();
  Model m;
  read_root_attrs(root, m);

  // Root groups that a later schema may define: report them as unsupported, not as typos.
  static const std::set<std::string> known = {
      "nodes", "materials", "elements", "mass", "constraints", "damping",
      "time_series", "patterns", "recorders", "process"};
  for (const auto& c : root.children()) {
    if (!known.count(c)) unsupported("/" + c, "the groups of schema v0 (schema/v0/README.md)");
  }

  read_nodes(root, m);
  read_materials(root, m);
  read_elements(root, m);
  read_mass(root, m);
  read_constraints(root, m);
  read_damping(root, m);
  read_time_series(root, m);
  read_patterns(root, m, filename);
  read_recorders(root, m);
  read_process(root, m);
  root.finish();

  validate(m);
  return m;
}

}  // namespace femorax::io
