#pragma once

// Compile-time contracts for elements and materials (docs/design.md, "Element and material
// contract"). Every registered element and material must satisfy these concepts; the
// registries static_assert them, so a missing or inconsistent trait is a compile error.

#include <array>
#include <concepts>
#include <type_traits>

#include "physics/common/inner.hpp"
#include "physics/common/strided.hpp"
#include "physics/macros.hpp"

namespace femorax {

enum class MassType : int { Consistent = 0, Lumped = 1, Selectable = 2 };

enum class MaterialKind : int {
  NDMaterial3D = 0,           // 6 strains: 11 22 33 12 23 31 (engineering shear)
  NDMaterialPlaneStrain = 1,  // 3 strains: 11 22 12
  Uniaxial = 2,               // 1 strain
};

enum class ExplicitScheme : int { CentralDifference = 0, ExplicitDifference = 1 };

// What a tangent is known to be (docs/design.md, "Symmetry of the tangent, and the solver
// choice"). Materials declare it for D, elements for their stiffness (their own terms plus
// the material's). The implicit solver follows from the weakest declaration in a stage:
// CG for SymmetricPositiveDefinite, GMRES otherwise. Tangent storage is packed (21 values
// for 6 strains) only when the tangent is declared symmetric.
// SymmetricPositiveSemidefinite: symmetric, never indefinite, but possibly singular (an
// elastoplastic tangent at a surface with zero plastic modulus): the operator is positive
// definite when it has a mass term (implicit dynamics), otherwise treated as Symmetric.
enum class TangentSymmetry : int {
  SymmetricPositiveDefinite = 0,
  SymmetricPositiveSemidefinite = 1,
  Symmetric = 2,
  General = 3
};

FEMORAX_HD constexpr TangentSymmetry weakest(TangentSymmetry a, TangentSymmetry b) {
  return static_cast<int>(a) > static_cast<int>(b) ? a : b;
}

FEMORAX_HD constexpr bool is_symmetric(TangentSymmetry t) { return t != TangentSymmetry::General; }

// Rayleigh damping coefficients of one element (Femora assigns them per region):
//   C = alphaM*M + betaK*K_current + betaKinit*K_initial + betaKcomm*K_committed
struct Rayleigh {
  double alphaM = 0.0;
  double betaK = 0.0;
  double betaKinit = 0.0;
  double betaKcomm = 0.0;
};

namespace detail {

// The trait is a const int (static constexpr). Checked before any value comparison, so a
// non-constant member fails the concept instead of breaking the build.
template <class T>
inline constexpr bool is_constexpr_int = std::is_same_v<T, const int>;

template <std::size_t N>
constexpr int sum_ndf(const std::array<int, N>& ndf) {
  int s = 0;
  for (int v : ndf) s += v;
  return s;
}

template <std::size_t N>
constexpr bool all_positive(const std::array<int, N>& ndf) {
  for (int v : ndf)
    if (v <= 0) return false;
  return true;
}

}  // namespace detail

// ---------------------------------------------------------------------------
// Material contract
// ---------------------------------------------------------------------------
template <class M>
concept Material = requires {
  typename M::Params;
  requires std::is_trivially_copyable_v<typename M::Params>;
  requires std::is_same_v<decltype(M::KIND), const MaterialKind>;
  requires detail::is_constexpr_int<decltype(M::NSTRAIN)>;
  requires detail::is_constexpr_int<decltype(M::NSTATE)>;
  requires M::NSTRAIN > 0 && M::NSTATE >= 0;
  // The tangent D: its symmetry, and whether it is constant (independent of strain and
  // state, as for elastic materials, so one copy per parameter set serves every point).
  requires std::is_same_v<decltype(M::TANGENT), const TangentSymmetry>;
  requires std::is_same_v<decltype(M::CONSTANT_TANGENT), const bool>;
} && requires(const typename M::Params& p, Strided<const double> committed,
              Strided<double> trial, const double* eps, double* sig,
              double (*D)[M::NSTRAIN]) {
  { M::update(p, committed, trial, eps, sig, D) } -> std::same_as<void>;
  { M::initial_tangent(p, D) } -> std::same_as<void>;
  // Tangent at the committed state (betaKcomm damping, stable time step). A material may
  // recompute it from its committed state or keep a copy in its state.
  { M::committed_tangent(p, committed, D) } -> std::same_as<void>;
  { M::density(p) } -> std::same_as<double>;
};

// ---------------------------------------------------------------------------
// Element contract
// ---------------------------------------------------------------------------
template <class E>
concept Element = requires {
  typename E::Params;
  requires std::is_trivially_copyable_v<typename E::Params>;
  requires detail::is_constexpr_int<decltype(E::NDM)>;
  requires detail::is_constexpr_int<decltype(E::NNODE)>;
  requires detail::is_constexpr_int<decltype(E::NEXTRA)>;
  requires detail::is_constexpr_int<decltype(E::NDOF)>;
  requires detail::is_constexpr_int<decltype(E::NGP)>;
  requires detail::is_constexpr_int<decltype(E::NSTATE)>;
  requires std::is_same_v<decltype(E::NODE_NDF),
                          const std::array<int, static_cast<std::size_t>(E::NNODE)>>;
  requires std::is_same_v<decltype(E::MASS), const MassType>;
  requires std::is_same_v<decltype(E::EXPLICIT), const bool>;
  // Whether geometry (shape-function derivatives, det J) may be computed once and reused:
  // true only for small-strain elements. Large-deformation or corotational elements must
  // recompute or update it (docs/design.md, "Element kernels").
  requires std::is_same_v<decltype(E::GEOMETRY_CACHEABLE), const bool>;
  requires std::is_same_v<decltype(E::MATERIAL_KIND), const MaterialKind>;
  // The stiffness's symmetry: the element's own terms combined with its material's D.
  requires std::is_same_v<decltype(E::TANGENT), const TangentSymmetry>;
  // Consistency of the sizes.
  requires E::NDM >= 1 && E::NDM <= 3;
  requires E::NNODE >= 1 && E::NEXTRA >= 0 && E::NGP >= 0 && E::NSTATE >= 0;
  requires detail::all_positive(E::NODE_NDF);
  requires E::NDOF == detail::sum_ndf(E::NODE_NDF) + E::NEXTRA;
} && requires(const typename E::Params& p, const Rayleigh& damp,
              const double (*xe)[E::NDM], const double* ue, const double* ve,
              PointArray<const double> committed, PointArray<double> trial, double* fe,
              double (*Ke)[E::NDOF], double* Me, ExplicitScheme scheme) {
  { E::template compute<true>(p, damp, xe, ue, ve, committed, trial, fe, Ke) }
      -> std::same_as<void>;
  { E::template compute<false>(p, damp, xe, ue, ve, committed, trial, fe, Ke) }
      -> std::same_as<void>;
  { E::mass(p, xe, Me) } -> std::same_as<void>;
  { E::max_omega_sq(p, xe, committed) } -> std::same_as<double>;
  { E::critical_dt(p, damp, scheme, xe, committed) } -> std::same_as<double>;
};

// An element that may appear in an explicit stage must be able to lump its mass, and gives
// the lumped (diagonal) mass per node.
template <class E>
concept ExplicitElement =
    Element<E> && E::EXPLICIT && (E::MASS == MassType::Lumped || E::MASS == MassType::Selectable) &&
    requires(const typename E::Params& p, const double (*xe)[E::NDM], double* m) {
      { E::lumped_mass(p, xe, m) } -> std::same_as<void>;
    };

// An element whose work splits into integration points (docs/design.md, "Element
// kernels"): the launch layer can then run one thread per point and reduce in shared
// memory, or compute the tangent as node-pair blocks. Elements that do not split this way
// (for example force-based beams) only satisfy Element.
template <class E>
concept PointwiseElement =
    Element<E> && (E::NGP > 0) && requires { typename E::PointShape; } &&
    requires(const typename E::Params& p, const double (*xl)[8], int gp,
             typename E::PointShape& s, const typename E::PointShape& cs, const double* u,
             double* strain, const double* stress, double* fI,
             const double (*dd)[6], double (*KIJ)[3]) {
      { E::point_geometry(xl, gp, s) } -> std::same_as<void>;
      { E::point_strain(cs, u, strain) } -> std::same_as<void>;
      { E::point_node_force(cs, stress, gp, fI) } -> std::same_as<void>;
      { E::point_body_force(p, cs, gp, gp) } -> std::same_as<double>;
      { E::point_block_tangent(cs, dd, gp, gp, KIJ) } -> std::same_as<void>;
    };

// Geometry sources of a MatrixFreeElement (RecomputedGeometry, CachedGeometry), made by
// the element itself so the launch layer never needs their layout:
//   recomputed_geometry(xl, p)   from the local coordinates xl[3][8] and the parameters
//   cached_geometry(g, c)        from the geometry cache (per point) and the element
//                                constants (per element; empty for elements without them)
// and geometry.shape(gp, s) gives point gp's shape functions.
template <class E>
concept GeometryFactories = requires(const typename E::Params& p, const double (*xl)[8],
                                     PointArray<const double> g, Strided<const double> c) {
  { E::recomputed_geometry(xl, p) } -> std::same_as<typename E::RecomputedGeometry>;
  { E::cached_geometry(g, c) } -> std::same_as<typename E::CachedGeometry>;
};

// A pointwise element that also gives its mass per point, so the matrix-free operator
// (physics/operators/element_operator.hpp) and its diagonal apply M without forming it:
//   point_mass_apply:    (M v) at node j from point s, all components (out[ndf of j])
//   point_mass_entry:    the entry of M between nodes j and k from point s, for the same
//                        component (M couples no components)
template <class E>
concept MatrixFreeElement =
    PointwiseElement<E> &&
    requires {
      typename E::RecomputedGeometry;  // shape(gp, s) from local coordinates
      typename E::CachedGeometry;      // shape(gp, s) from the geometry cache
    } &&
    GeometryFactories<E> &&
    requires(const typename E::Params& p, const typename E::PointShape& cs, const double* v,
             int j, double* out) {
      { E::point_mass_apply(p, cs, v, j, out) } -> std::same_as<void>;
      { E::point_mass_entry(p, cs, j, j) } -> std::same_as<double>;
    };

// An element with a stiffness term of its own besides the point terms B^T D B (docs/design.md,
// "Element-level terms"): SSPbrick's hourglass stabilization Kstab, later beams' and
// interfaces' element matrices. The term K_el is built from per-element constants:
//   NCONST                        doubles per element, built once from the coordinates and
//                                 the parameters by element_constants(p, xl, c) (stored in
//                                 the element-constants array, value k at [k * nElements + e],
//                                 or recomputed in each kernel: the same function either way;
//                                 geometry.constants() gives them to the element's code)
//   element_term_reduce(p, c, v, z)        z[NTERM]: the element-wide part of K_el v
//   element_term_node(p, c, z, j, out)     (K_el v) at node j from z
//   element_term_entry(p, c, a, b)         K_el(a, b), for diagonals and assembled references
// so that K_el v is applied without forming the NDOF x NDOF matrix in one thread
// (CLAUDE.md, performance rule 3a), and the reduction runs once per element while the node
// expansion splits across lanes (inner.hpp). K_el is constant and part of the current,
// initial and committed stiffness alike: every stiffness-proportional damping term
// includes it.
template <class E>
concept ElementTermElement =
    detail::is_constexpr_int<decltype(E::NCONST)> && detail::is_constexpr_int<decltype(E::NTERM)> &&
    (E::NCONST > 0) && (E::NTERM > 0) &&
    requires(const typename E::Params& p, const double (*xl)[8], Strided<double> cw,
             Strided<const double> c, const double* v, double* z, int j, double* out) {
      { E::element_constants(p, xl, cw) } -> std::same_as<void>;
      { E::element_term_reduce(p, c, v, z) } -> std::same_as<void>;
      { E::element_term_node(p, c, z, j, out) } -> std::same_as<void>;
      { E::element_term_entry(p, c, j, j) } -> std::same_as<double>;
    };

// Whether the element's "initial stiffness" (betaKinit damping) is its current tangent, as
// SSPbrick::getInitialStiff returns getTangentStiff. Optional trait; false when absent.
template <class E>
FEMORAX_HD constexpr bool initial_stiffness_is_tangent() {
  if constexpr (requires { E::INITIAL_STIFFNESS_IS_TANGENT; }) return E::INITIAL_STIFFNESS_IS_TANGENT;
  else return false;
}

// Doubles of element constants per element (0 without an element term).
template <class E>
FEMORAX_HD constexpr int element_constant_count() {
  if constexpr (ElementTermElement<E>) return E::NCONST;
  else return 0;
}

// A material whose update splits into an elastic predictor and a plastic corrector
// (PressureDependMultiYield): predict(ctx, p, c, t, eps, sig, D) finishes a point that stays
// elastic and returns false, or returns true for a point that needs correct(...), which
// finishes it; together they give update's bits. The launch layer's compacted path runs the
// corrector only on the points that need it.
template <class M>
concept SplitMaterial = Material<M> && requires(const typename M::Params& p, Strided<const double> c,
                                                Strided<double> t, const double* eps, double* sig,
                                                double (*D)[M::NSTRAIN]) {
  { M::predict(SerialInner{}, p, c, t, eps, sig, D) } -> std::same_as<bool>;
  { M::correct(SerialInner{}, p, c, t, eps, sig, D) } -> std::same_as<void>;
};

// An element whose material can be swapped (Elem::WithMaterial<M>), with the pointwise
// interface the compacted path's predictor pass uses.
template <class E>
concept RebindableElement = requires { typename E::template WithMaterial<typename E::Material>; };

// The tangent the stable time step reads: the stiffest response the material can show from
// its committed state. Optional material function stability_tangent(p, committed, D); without
// it, the committed tangent. Elastoplastic materials provide it, because their committed
// tangent is softer than the elastic unloading stiffness that bounds the next step (for
// PressureDependMultiYield, the elastic moduli at the current confinement).
template <class M>
FEMORAX_HD void stability_tangent(const typename M::Params& p, Strided<const double> committed,
                                  double (*D)[M::NSTRAIN]) {
  if constexpr (requires { M::stability_tangent(p, committed, D); }) M::stability_tangent(p, committed, D);
  else M::committed_tangent(p, committed, D);
}

// An (element, material) pair that can be instantiated as a batch.
template <class E, class M>
concept CompatiblePair = Element<E> && Material<M> && (E::MATERIAL_KIND == M::KIND);

}  // namespace femorax
