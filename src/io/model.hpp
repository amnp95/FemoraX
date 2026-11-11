#pragma once

// Host-side model, as read from a schema v0 file (schema/v0/README.md). Arrays keep the
// file's layout and OpenSees tags; the compile step renumbers and converts to the
// structure-of-arrays layout the kernels use.

#include <array>
#include <cstdint>
#include <optional>
#include <string>
#include <variant>
#include <vector>

namespace femorax::io {

struct SchemaVersion {
  int major = 0;
  int minor = 0;
};

// The schema version this reader implements. It reads files of the same major version
// with a minor version up to this one.
inline constexpr SchemaVersion reader_schema_version{0, 6};

// Femora's own identifiers, kept for messages only (never used to resolve references).
struct Metadata {
  std::optional<std::int32_t> femora_tag;
  std::optional<std::string> femora_name;
};

struct Nodes {
  std::vector<std::int32_t> tags;   // [N]
  std::vector<double> coords;       // [N * ndm], row-major as in the file
  std::vector<std::int32_t> ndf;    // [N]
  std::size_t size() const { return tags.size(); }
};

// ---------------------------------------------------------------------------
struct ElasticIsotropicData {
  double E = 0.0, nu = 0.0, rho = 0.0;
};

// nDMaterial PressureDependMultiYield (schema 0.4), Femora's / OpenSees' Tcl names. Defaults
// of the optional items are those of OpenSees' Tcl parser (OPS_PressureDependMultiYield).
struct PressureDependMultiYieldData {
  std::int32_t nd = 3;
  double rho = 0.0, refShearModul = 0.0, refBulkModul = 0.0, frictionAng = 0.0,
         peakShearStra = 0.0, refPress = 0.0, pressDependCoe = 0.0, PTAng = 0.0, contrac = 0.0,
         dilat1 = 0.0, dilat2 = 0.0, liquefac1 = 0.0, liquefac2 = 0.0, liquefac3 = 0.0;
  std::int32_t noYieldSurf = 20;  // < 0: |noYieldSurf| user (gamma, G/Gmax) pairs
  std::vector<double> pairs;      // [|noYieldSurf| * 2], row-major (gamma, G/Gmax)
  double e = 0.6, cs1 = 0.9, cs2 = 0.02, cs3 = 0.7, pa = 101.0, c = 0.3;
};

// nDMaterial PressureIndependMultiYield (schema 0.4). OpenSees' parser requires every item
// but the pairs.
struct PressureIndependMultiYieldData {
  std::int32_t nd = 3;
  double rho = 0.0, refShearModul = 0.0, refBulkModul = 0.0, cohesi = 0.0, peakShearStra = 0.0,
         frictionAng = 0.0, refPress = 100.0, pressDependCoe = 0.0;
  std::int32_t noYieldSurf = 20;
  std::vector<double> pairs;
};

struct NDMaterial {
  std::int32_t tag = 0;
  std::variant<ElasticIsotropicData, PressureDependMultiYieldData, PressureIndependMultiYieldData> data;
  Metadata meta;
};

const char* material_type_name(const NDMaterial& m);
// Whether the material has stages (updateMaterialStage): the multi-yield soil materials.
bool has_material_stage(const NDMaterial& m);

// ---------------------------------------------------------------------------
enum class ElementType { StdBrick, SSPbrick };

const char* to_string(ElementType t);
int node_count(ElementType t);

struct ElementGroup {
  std::string id;                       // group name in /elements
  ElementType type = ElementType::StdBrick;
  std::int32_t material = 0;            // tag of an NDMaterial
  std::array<double, 3> body_force{};   // b1 b2 b3
  bool lumped = false;                  // stdBrick -lumped; SSPbrick always lumps
  std::vector<std::int32_t> tags;       // [M]
  std::vector<std::int32_t> connectivity;  // [M * node_count(type)], node tags
  Metadata meta;
  std::size_t size() const { return tags.size(); }
};

// One row per DOF (1-based DOF numbers, as in OpenSees).
struct NodalMass {
  std::vector<std::int32_t> node, dof;
  std::vector<double> value;
};

struct Fix {
  std::vector<std::int32_t> node, dof;
};

struct EqualDof {
  std::vector<std::int32_t> retained, constrained, dof;
  // equalDOF command of each row (schema 0.2), or empty: each (retained, constrained) pair
  // is then one command.
  std::vector<std::int32_t> command;
};

struct RayleighRegion {
  std::string id;  // group name; regions apply in file order, the last one wins
  double alphaM = 0.0, betaK = 0.0, betaKinit = 0.0, betaKcomm = 0.0;
  std::vector<std::int32_t> elements;
  Metadata meta;
};

// Uniform damping per element region (docs/design.md, "Frequency-independent damping"):
// Femora's `damping Uniform` attached with `region ... -damp`. Built by tests until the
// schema 0.7 reader (step 4) reads /damping/uniform.
struct UniformDampingRegion {
  std::string id;
  double ratio = 0.0;           // the damping ratio over the band
  double freq1 = 0.0, freq2 = 0.0;
  double activate_time = 0.0, deactivate_time = 1e20;
  std::string fit = "femorax";  // or "opensees"
  std::vector<std::int32_t> elements;
};

// ---------------------------------------------------------------------------
struct PathSeries {
  std::int32_t tag = 0;
  double factor = 1.0;
  std::optional<double> dt;    // uniform spacing (OpenSees PathSeries) ...
  std::vector<double> time;    // ... or explicit times (OpenSees PathTimeSeries)
  std::vector<double> values;
  bool use_last = false;
  bool prepend_zero = false;
  Metadata meta;
};

struct UniformExcitation {
  std::int32_t tag = 0;
  std::int32_t dof = 1;     // 1-based direction
  std::int32_t accel = 0;   // time series tag
  double vel0 = 0.0;
  double factor = 1.0;
  Metadata meta;
};

// The Domain Reduction Method input motion (schema 0.6): OpenSees' H5DRM pattern, Femora's
// model.pattern.h5drm. The DRM data (station coordinates, flags, displacement and
// acceleration histories) stays in its own HDF5 file.
struct H5DrmPattern {
  std::int32_t tag = 0;
  std::string file;  // as written in the model file
  std::string path;  // resolved: relative paths are relative to the model file's folder
  double factor = 1.0, crd_scale = 1.0, distance_tolerance = 1e-6;
  bool do_coordinate_transformation = false;
  double T[3][3] = {{1, 0, 0}, {0, 1, 0}, {0, 0, 1}};
  double x0[3] = {0, 0, 0};
  Metadata meta;
};

// ---------------------------------------------------------------------------
enum class NodeResponse { Disp, Vel, Accel };

struct NodeRecorder {
  std::string file;
  NodeResponse response = NodeResponse::Disp;
  bool time = false;
  std::optional<double> dT;
  std::int32_t precision = 6;
  bool close_on_write = false;
  std::vector<std::int32_t> nodes;
  std::vector<std::int32_t> dofs;
  std::vector<std::int32_t> time_series;  // empty, or one tag per DOF
};

struct VtkhdfRecorder {
  std::string file;
  std::optional<double> dT;
  std::optional<double> rTolDt;
  // subset of disp vel accel (node data) and, since 0.4, stress3D6 strain3D6 (element data:
  // the average over the element's integration points), in file order
  std::vector<std::string> responses;
  std::optional<std::int32_t> region;  // always rejected (schema 0.1)
  std::vector<std::int32_t> elements;  // since 0.1; empty = whole model
};

struct Recorder {
  std::string name;
  std::variant<NodeRecorder, VtkhdfRecorder> data;
  Metadata meta;
};

// ---------------------------------------------------------------------------
enum class AnalysisType { Transient, Static };
enum class Integrator { ExplicitDifference, CentralDifference, Newmark, LoadControl };
enum class Algorithm { Linear, Newton, ModifiedNewton };
enum class Preconditioner { Auto, Jacobi };  // Auto: FemoraX chooses (Jacobi today)
enum class ConstraintHandler { Transformation, Plain };
enum class TestType { NormUnbalance, NormDispIncr, EnergyIncr };

const char* to_string(Integrator i);
bool is_explicit(Integrator i);

struct AnalysisStep {
  std::string name;
  AnalysisType type = AnalysisType::Transient;
  double dt = 0.0;                         // Transient only
  std::optional<std::int32_t> num_steps;  // exactly one of num_steps, final_time
  std::optional<double> final_time;
  Integrator integrator = Integrator::ExplicitDifference;
  double gamma = 0.0, beta = 0.0;          // Newmark
  std::string newmark_form = "D";
  // LoadControl (Static, schema 0.3): integrator LoadControl incr numIter minIncr maxIncr.
  double load_increment = 0.0;
  std::int32_t load_num_iter = 1;
  std::optional<double> load_min_increment, load_max_increment;
  Algorithm algorithm = Algorithm::Linear;
  // ModifiedNewton options (schema 0.3): -initial, -factoronce.
  bool modified_newton_initial = false, modified_newton_factor_once = false;
  // FemoraX solver options (schema 0.3, all optional; absent: FemoraX chooses).
  std::optional<double> solver_tolerance;
  std::optional<std::int32_t> solver_max_iterations;
  Preconditioner preconditioner = Preconditioner::Auto;
  ConstraintHandler constraint_handler = ConstraintHandler::Transformation;
  std::string numberer;                    // recorded, not used
  std::string system;                      // recorded, not used
  std::optional<TestType> test_type;
  std::optional<double> test_tol;
  std::optional<std::int32_t> test_max_iter;
  std::optional<std::int32_t> max_retries, num_sublevels, num_substeps;
  bool initialize = false;
  bool allow_unstable_dt = false;
};

struct PatternStep { std::int32_t tag = 0; };
struct RecorderStep { std::string name; };
struct SetTimeStep { double time = 0.0; };
struct LoadConstStep { std::optional<double> time; };
struct WipeAnalysisStep {};
// updateMaterialStage -material <tag> -stage <stage>, for each listed material (schema 0.4;
// Femora's update_material_stage_to_elastic / _to_plastic actions).
struct UpdateMaterialStageStep {
  std::vector<std::int32_t> materials;
  std::int32_t stage = 0;  // 0 elastic, 1 plastic
};

struct ProcessStep {
  std::string id;     // group name in /process (order)
  std::string label;  // Femora's description, may be empty
  std::variant<AnalysisStep, PatternStep, RecorderStep, SetTimeStep, LoadConstStep,
               WipeAnalysisStep, UpdateMaterialStageStep>
      step;
};

// ---------------------------------------------------------------------------
struct Model {
  SchemaVersion version;
  int ndm = 3;
  std::string generator;
  std::string model_name;
  std::string units;

  Nodes nodes;
  std::vector<NDMaterial> materials;
  std::vector<ElementGroup> elements;
  NodalMass mass;
  Fix fix;
  EqualDof equal_dof;
  std::vector<RayleighRegion> rayleigh;
  std::vector<UniformDampingRegion> uniform;  // the last region of an element wins
  std::vector<PathSeries> time_series;
  std::vector<UniformExcitation> patterns;
  std::vector<H5DrmPattern> drm_patterns;  // schema 0.6
  std::vector<Recorder> recorders;
  std::vector<ProcessStep> process;

  // Non-fatal findings (for example overlapping damping regions), for the caller to print.
  std::vector<std::string> warnings;

  std::size_t element_count() const;
};

}  // namespace femorax::io
