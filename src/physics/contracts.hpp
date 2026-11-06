#pragma once

// Compile-time contracts for elements and materials (docs/design.md, "Element and material
// contract"). Every registered element and material must satisfy these concepts; the
// registries static_assert them, so a missing or inconsistent trait is a compile error.

#include <array>
#include <concepts>
#include <type_traits>

#include "physics/macros.hpp"

namespace femorax {

enum class MassType : int { Consistent = 0, Lumped = 1, Selectable = 2 };

enum class MaterialKind : int {
  NDMaterial3D = 0,           // 6 strains: 11 22 33 12 23 31 (engineering shear)
  NDMaterialPlaneStrain = 1,  // 3 strains: 11 22 12
  Uniaxial = 2,               // 1 strain
};

enum class ExplicitScheme : int { CentralDifference = 0, ExplicitDifference = 1 };

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
} && requires(const typename M::Params& p, const double* committed, double* trial,
              const double* eps, double* sig, double (*D)[M::NSTRAIN]) {
  { M::update(p, committed, trial, eps, sig, D) } -> std::same_as<void>;
  { M::initial_tangent(p, D) } -> std::same_as<void>;
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
  requires std::is_same_v<decltype(E::MATERIAL_KIND), const MaterialKind>;
  // Consistency of the sizes.
  requires E::NDM >= 1 && E::NDM <= 3;
  requires E::NNODE >= 1 && E::NEXTRA >= 0 && E::NGP >= 0 && E::NSTATE >= 0;
  requires detail::all_positive(E::NODE_NDF);
  requires E::NDOF == detail::sum_ndf(E::NODE_NDF) + E::NEXTRA;
} && requires(const typename E::Params& p, const Rayleigh& damp,
              const double (*xe)[E::NDM], const double* ue, const double* ve,
              const double* committed, double* trial, double* fe, double (*Ke)[E::NDOF],
              double* Me, ExplicitScheme scheme) {
  { E::template compute<true>(p, damp, xe, ue, ve, committed, trial, fe, Ke) }
      -> std::same_as<void>;
  { E::template compute<false>(p, damp, xe, ue, ve, committed, trial, fe, Ke) }
      -> std::same_as<void>;
  { E::mass(p, xe, Me) } -> std::same_as<void>;
  { E::max_omega_sq(p, xe, committed) } -> std::same_as<double>;
  { E::critical_dt(p, damp, scheme, xe, committed) } -> std::same_as<double>;
};

// An element that may appear in an explicit stage must be able to lump its mass.
template <class E>
concept ExplicitElement =
    Element<E> && E::EXPLICIT && (E::MASS == MassType::Lumped || E::MASS == MassType::Selectable);

// An (element, material) pair that can be instantiated as a batch.
template <class E, class M>
concept CompatiblePair = Element<E> && Material<M> && (E::MATERIAL_KIND == M::KIND);

}  // namespace femorax
