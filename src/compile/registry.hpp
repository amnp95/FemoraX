#pragma once

// Element and material registry (docs/design.md, "Registration"): the one place that maps
// model types onto physics types. Adding an element or material is its physics header plus
// one entry here (a small adapter from the model's data to its Params) and one line in the
// type list. Everything else (batches, lumped mass, kernels in every layout x scatter x
// geometry, benchmarks) is generic over the entries.
//
// Host code (it reads io::Model data); the kernels use the physics types it names.

#include <cstdint>
#include <cstring>
#include <type_traits>
#include <variant>
#include <vector>

#include "io/model.hpp"
#include "physics/contracts.hpp"
#include "physics/elements/ssp_brick.hpp"
#include "physics/elements/std_brick.hpp"
#include "physics/materials/elastic_isotropic.hpp"

namespace femorax::registry {

template <class... T>
struct TypeList {};

// --- Materials ---------------------------------------------------------------------------
struct ElasticIsotropicEntry {
  using Material = ElasticIsotropic;
  static constexpr const char* name = "ElasticIsotropic";
  static bool matches(const io::NDMaterial& m) {
    return std::holds_alternative<io::ElasticIsotropicData>(m.data);
  }
  static Material::Params params(const io::NDMaterial& m) {
    const auto& d = std::get<io::ElasticIsotropicData>(m.data);
    return {d.E, d.nu, d.rho};
  }
};

// --- Elements ------------------------------------------------------------------------------
struct StdBrickEntry {
  template <class Mat>
  using Element = StdBrick<Mat>;
  static constexpr io::ElementType type = io::ElementType::StdBrick;
  static constexpr const char* name = "stdBrick";
  template <class Mat>
  static typename Element<Mat>::Params params(const io::ElementGroup& g,
                                              const typename Mat::Params& mp) {
    return {mp, {g.body_force[0], g.body_force[1], g.body_force[2]}, g.lumped ? 1 : 0};
  }
};

struct SSPbrickEntry {
  template <class Mat>
  using Element = SSPbrick<Mat>;
  static constexpr io::ElementType type = io::ElementType::SSPbrick;
  static constexpr const char* name = "SSPbrick";
  template <class Mat>
  static typename Element<Mat>::Params params(const io::ElementGroup& g,
                                              const typename Mat::Params& mp) {
    return {mp, {g.body_force[0], g.body_force[1], g.body_force[2]}};
  }
};

// The registered entries: one line each.
using Materials = TypeList<ElasticIsotropicEntry>;
using Elements = TypeList<StdBrickEntry, SSPbrickEntry>;

// --- Dispatch ------------------------------------------------------------------------------
// Pair index: element entry index * number of materials + material entry index.
template <class L>
struct Size;
template <class... T>
struct Size<TypeList<T...>> {
  static constexpr int value = sizeof...(T);
};
inline constexpr int num_materials = Size<Materials>::value;
inline constexpr int num_elements = Size<Elements>::value;

// Calls f.template operator()<ElementEntry, MaterialEntry, pair index>() for every
// registered pair whose material kinds match.
template <class F, class E, class... Ms, std::size_t... I>
void for_materials(F& f, int e_index, TypeList<Ms...>, std::index_sequence<I...>) {
  (
      [&] {
        using M = Ms;
        if constexpr (CompatiblePair<typename E::template Element<typename M::Material>,
                                     typename M::Material>)
          f.template operator()<E, M>(e_index * num_materials + static_cast<int>(I));
      }(),
      ...);
}
template <class F, class... Es, std::size_t... I>
void for_elements(F& f, TypeList<Es...>, std::index_sequence<I...>) {
  (for_materials<F, Es>(f, static_cast<int>(I), Materials{},
                        std::make_index_sequence<num_materials>{}),
   ...);
}
template <class F>
void for_each_pair(F&& f) {
  for_elements(f, Elements{}, std::make_index_sequence<num_elements>{});
}

// The pair index of (element type, material), or -1 if no registered pair matches.
inline int pair_index(io::ElementType type, const io::NDMaterial& m) {
  int found = -1;
  for_each_pair([&]<class E, class M>(int index) {
    if (E::type == type && M::matches(m)) found = index;
  });
  return found;
}

// Typed element parameters stored as bytes in a parameter set (Params are trivially
// copyable by contract), so the compile step's batches stay generic.
struct ParamBlob {
  std::vector<unsigned char> bytes;
  template <class P>
  static ParamBlob of(const P& p) {
    static_assert(std::is_trivially_copyable_v<P>);
    ParamBlob b;
    b.bytes.resize(sizeof(P));
    std::memcpy(b.bytes.data(), &p, sizeof(P));
    return b;
  }
  template <class P>
  P as() const {
    static_assert(std::is_trivially_copyable_v<P>);
    P p;
    std::memcpy(&p, bytes.data(), sizeof(P));
    return p;
  }
};

}  // namespace femorax::registry
