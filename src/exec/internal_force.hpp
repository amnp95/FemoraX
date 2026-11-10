#pragma once

// Internal plus Rayleigh damping force of all element batches:
//   f = sum over elements of fe(u, v), assembled into equation space.
// Every registered (element, material) pair runs through the same composable kernels
// (exec/element_kernels.hpp) and assembly machinery (exec/assembly.hpp). Choices, all
// giving bit-identical element forces on a given backend:
//   Layout::Flat        one thread per element
//   Layout::Team        one thread per Gauss point, elements_per_team elements per team
//   Scatter::Gather     elements write their own slots in locality order, then each
//                       equation adds its slots in a fixed order (deterministic; default)
//   Scatter::Atomic     atomic adds (order varies between runs; developer option)
//   Scatter::Colored    one launch per color (deterministic; needs the compile step's
//                       coloring; developer option)
//   Geometry::Recompute / Geometry::Cached
//   MaterialPath::Fused      the material update inside the element kernel
//   MaterialPath::Compacted  for split materials (PressureDependMultiYield): predictor pass,
//                            the plastic points compacted, corrector on those only, then
//                            the element kernel (exec/compacted_kernels.hpp); same bits on
//                            host backends, rounding on GPUs
// Users never choose: exec/force_settings.hpp picks the settings. All memory is allocated
// in the constructor; compute() allocates nothing.

#include <Kokkos_Core.hpp>

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "compile/batches.hpp"
#include "exec/assembly.hpp"
#include "physics/operators/element_operator.hpp"

namespace femorax::exec {

enum class Layout { Flat, Team };
enum class Scatter { Gather, Atomic, Colored };
enum class Geometry { Recompute, Cached };
enum class MaterialPath { Fused, Compacted };

const char* to_string(Layout l);
const char* to_string(Scatter s);
const char* to_string(Geometry g);
const char* to_string(MaterialPath m);

struct ForceSettings {
  Layout layout = Layout::Flat;
  Scatter scatter = Scatter::Gather;
  int elements_per_team = 0;  // team layout; 0 = the backend's default
  Geometry geometry = Geometry::Recompute;
  MaterialPath material = MaterialPath::Fused;  // batches without a split material: Fused
  bool deterministic() const { return scatter != Scatter::Atomic; }
};

// "flat_gather_cached", "team_gather_ept4", "flat_gather_cached_compacted", ...;
// parse_settings reads the same form.
std::string to_string(const ForceSettings& s);
bool parse_settings(const std::string& text, ForceSettings& out);

// The matrix-free operator (docs/design.md, "The matrix-free operator"): where the
// Gauss-point tangent comes from. Auto: Constant when every material in the model has a
// constant tangent, Stored otherwise. Constant falls back to Stored for a material whose
// tangent is not constant; Stored needs enable_stored_tangents().
enum class TangentSource { Auto, Constant, Stored, Recomputed };
const char* to_string(TangentSource t);

struct OperatorSettings {
  Scatter scatter = Scatter::Gather;
  Geometry geometry = Geometry::Cached;  // falls back to Recompute without a cache
  TangentSource tangent = TangentSource::Auto;
};
std::string to_string(const OperatorSettings& s);  // "gather_cached_constant", ...

// Whether the geometry cache is built: On, Off, or Auto (built when the model, the cache
// and the work vectors fit in 80% of the device's free memory; not built if the free
// memory is unknown).
enum class GeometryCache { Auto, On, Off };

class InternalForce {
 public:
  explicit InternalForce(const compile::Compiled& c, GeometryCache cache = GeometryCache::Auto);
  ~InternalForce();
  InternalForce(const InternalForce&) = delete;
  InternalForce& operator=(const InternalForce&) = delete;

  // f = internal + damping force at (u, v). Cached geometry without a cache falls back to
  // recomputing (same bits). Allocates nothing.
  void compute(const ForceSettings& settings);

  // The settings compute() runs (cache fallback, elements-per-team clamp).
  ForceSettings resolve(const ForceSettings& settings) const;

  // Equation-space vectors, device-resident. Set u and v before compute().
  Vec u, v, f;
  void fill(double u0, double v0);

  std::int32_t num_equations() const { return neq_; }
  std::int64_t num_elements() const;
  bool has_geometry_cache() const { return has_cache_; }
  // Whether a batch's material splits into predictor and corrector (MaterialPath::Compacted
  // changes something), and how many points took the corrector in the last compute.
  bool has_split_material() const;
  std::int64_t last_plastic_points() const { return last_plastic_points_; }
  bool has_coloring() const { return has_coloring_; }
  std::int64_t geometry_cache_bytes() const { return cache_bytes_; }
  std::int64_t gather_bytes() const { return gather_.bytes(); }
  // Device bytes held by this object.
  std::int64_t device_bytes() const;

  // --- Matrix-free operator: K_eff = k.stiffness K_t + k.damping C + k.mass M (with the
  // nodal masses), applied element by element through the force kernels' scatter. All
  // vectors in equation space, on the device; nothing is allocated in these calls.
  // u: the displacement whose tangent is used (Recomputed, store_tangents).
  OperatorSettings resolve(const OperatorSettings& s) const;
  void apply_operator(const OperatorCoefficients& k, const OperatorSettings& s, const Vec& u,
                      const Kokkos::View<const double*, Space>& x, const Vec& y);
  void operator_diagonal(const OperatorCoefficients& k, const OperatorSettings& s, const Vec& u,
                         const Vec& d);
  // y = M r_d for the rigid unit translation r_d in direction d (0-based) of every element
  // DOF, the moving base's fixed DOFs included, plus the nodal masses: the UniformExcitation
  // load is -ag M r_d (Brick::addInertiaLoadToUnbalance). With a consistent mass the free
  // DOFs next to the base are loaded through their coupling to it.
  void mass_times_ground(int direction, const OperatorSettings& s, const Vec& u, const Vec& y);
  // Allocates the per-point tangent storage (setup; packed for declared-symmetric tangents).
  void enable_stored_tangents();
  bool has_stored_tangents() const { return stored_bytes_ > 0; }
  std::int64_t stored_tangent_bytes() const { return stored_bytes_; }
  // State determination for TangentSource::Stored: D at every point at u.
  void store_tangents(const Vec& u, const OperatorSettings& s);
  // The weakest declared tangent symmetry over the batches (TangentSymmetry).
  TangentSymmetry tangent_symmetry() const;
  bool constant_tangents() const;  // every material has CONSTANT_TANGENT

  // Material history state (committed and trial buffers per batch). The force kernels
  // read the committed state and write the trial one; commit_state makes the trial state
  // committed (a swap: the update writes every trial value). Nothing to revert: the next
  // force evaluation starts from the committed state again.
  bool has_state() const;
  void commit_state();
  // updateMaterialStage: sets the stage of model material `material` (index into
  // model.materials) in every parameter set that uses it. Returns false if none does.
  bool set_material_stage(std::int32_t material, int stage);

  // Element output for recorders (VTKHDF stress3D6 / strain3D6): select_elements fixes the
  // elements (by tag) and their output rows, once; element_output writes, for the current
  // u and the committed state, each element's 6 values (which 0: stress, 1: strain) to
  // out(offset + row * 6 + c), averaged over its points as the element does.
  int select_elements(const std::vector<std::int32_t>& tags);
  void element_output(int selection, int which, const Vec& out, std::int64_t offset);

  // H5DRM effective forces of the DRM elements of batch `batch` (exec/drm_load.hpp,
  // kernels::launch_drm_forces) into slots(base + a * n + i).
  void drm_forces(int batch, const Kokkos::View<std::int32_t*, Space>& elems,
                  const Kokkos::View<std::uint32_t*, Space>& masks, const Kokkos::View<std::int32_t*, Space>& node_slot,
                  const Vec& du, const Vec& da, const Vec& slots, std::int64_t base);

  // Smallest stable explicit step over all elements (docs/design.md, "Stable time step").
  double min_critical_dt(ExplicitScheme scheme) const;

  static int default_elements_per_team();
  int max_elements_per_team() const;  // smallest over the batches' scratch needs

  struct Batch;  // one per registered pair present (internal_force.cpp)

 private:
  std::int32_t neq_ = 0, nnodes_ = 0;
  bool has_cache_ = false, has_coloring_ = true;
  std::int64_t cache_bytes_ = 0, stored_bytes_ = 0, last_plastic_points_ = 0;
  Vec coords_;
  Vec nodal_mass_;
  Kokkos::View<std::int8_t*, Space> eq_dof_;
  SlotGather gather_;
  std::vector<std::unique_ptr<Batch>> batches_;
  std::vector<std::vector<std::int32_t>> batch_tags_;  // element tags per batch (-1: padding)
  struct Selection {
    std::vector<Kokkos::View<std::int32_t*, Space>> elems, rows;  // per batch
  };
  std::vector<Selection> selections_;
};

}  // namespace femorax::exec
