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
inline constexpr SchemaVersion reader_schema_version{0, 3};

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

struct NDMaterial {
  std::int32_t tag = 0;
  std::variant<ElasticIsotropicData> data;
  Metadata meta;
};

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
  std::vector<std::string> responses;  // subset of disp vel accel, in file order
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

struct ProcessStep {
  std::string id;     // group name in /process (order)
  std::string label;  // Femora's description, may be empty
  std::variant<AnalysisStep, PatternStep, RecorderStep, SetTimeStep, LoadConstStep,
               WipeAnalysisStep>
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
  std::vector<PathSeries> time_series;
  std::vector<UniformExcitation> patterns;
  std::vector<Recorder> recorders;
  std::vector<ProcessStep> process;

  // Non-fatal findings (for example overlapping damping regions), for the caller to print.
  std::vector<std::string> warnings;

  std::size_t element_count() const;
};

}  // namespace femorax::io
