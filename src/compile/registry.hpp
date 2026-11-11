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
#include <string>
#include <type_traits>
#include <variant>
#include <vector>

#include "io/h5.hpp"
#include "io/model.hpp"
#include "compile/uniform_damping_fit.hpp"
#include "physics/contracts.hpp"
#include "physics/damping/uniform_damped.hpp"
#include "physics/elements/ssp_brick.hpp"
#include "physics/elements/std_brick.hpp"
#include "physics/materials/elastic_isotropic.hpp"
#include "physics/materials/pressure_depend_multi_yield.hpp"
#include "physics/materials/pressure_independ_multi_yield.hpp"

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

// PressureDependMultiYield in surface buckets: the batch of a material is the smallest
// bucket that holds its surfaces. Another bucket is one more line in Materials below.
template <int NS>
struct PressureDependMultiYieldEntry {
  using Material = PressureDependMultiYield<NS>;
  static constexpr const char* name = NS == 20 ? "PressureDependMultiYield20" : "PressureDependMultiYield40";
  static int surfaces(const io::NDMaterial& m) {
    const auto* d = std::get_if<io::PressureDependMultiYieldData>(&m.data);
    if (!d) return -1;
    return d->noYieldSurf < 0 ? -d->noYieldSurf : d->noYieldSurf;
  }
  static bool matches(const io::NDMaterial& m);  // defined below the type list
  static typename Material::Params params(const io::NDMaterial& m) {
    const auto& d = std::get<io::PressureDependMultiYieldData>(m.data);
    if (d.nd != 3) throw io::UnsupportedComponent("unsupported component: PressureDependMultiYield nd = 2");
    PressureDependMultiYieldInput in{};
    in.rho = d.rho;
    in.refShearModul = d.refShearModul;
    in.refBulkModul = d.refBulkModul;
    in.frictionAng = d.frictionAng;
    in.peakShearStra = d.peakShearStra;
    in.refPress = d.refPress;
    in.pressDependCoe = d.pressDependCoe;
    in.phaseTransformAng = d.PTAng;
    in.contractionParam1 = d.contrac;
    in.dilationParam1 = d.dilat1;
    in.dilationParam2 = d.dilat2;
    in.liquefactionParam1 = d.liquefac1;
    in.liquefactionParam2 = d.liquefac2;
    in.liquefactionParam4 = d.liquefac3;
    in.numberOfYieldSurf = surfaces(m);
    in.gredu = d.noYieldSurf < 0 ? d.pairs.data() : nullptr;
    in.e = d.e;
    in.volLimit1 = d.cs1;
    in.volLimit2 = d.cs2;
    in.volLimit3 = d.cs3;
    in.atm = d.pa;
    in.cohesi = d.c;
    in.hv = 0.0;  // Femora never sets hv, pv (OpenSees' defaults)
    in.pv = 1.0;
    typename Material::Params p{};
    const int r = Material::setup(in, p);
    const std::string where = "/materials/nd/" + std::to_string(m.tag);
    switch (r) {
      case Material::Ok: return p;
      case Material::InvalidFrictionAngle:
        throw io::ModelError(where + ": the G/Gmax curve and refPress give sin(friction angle) outside [0, 1] "
                             "(OpenSees: \"Invalid friction angle, please modify ref. pressure or G/Gmax curve\")");
      case Material::NegativeModulus:
        throw io::ModelError(where + ": the G/Gmax curve gives a yield surface with a plastic modulus <= 0 "
                             "(OpenSees: \"has plastic modulus < 0. Please modify G/Gmax curve\")");
      case Material::TooFewPairs:
        throw io::ModelError(where + ": a user-defined backbone needs at least two (gamma, G/Gmax) pairs");
      default:
        throw io::ModelError(where + ": invalid PressureDependMultiYield parameters");
    }
  }
};

// PressureIndependMultiYield in the same surface buckets.
template <int NS>
struct PressureIndependMultiYieldEntry {
  using Material = PressureIndependMultiYield<NS>;
  static constexpr const char* name = NS == 20 ? "PressureIndependMultiYield20" : "PressureIndependMultiYield40";
  static int surfaces(const io::NDMaterial& m) {
    const auto* d = std::get_if<io::PressureIndependMultiYieldData>(&m.data);
    if (!d) return -1;
    return d->noYieldSurf < 0 ? -d->noYieldSurf : d->noYieldSurf;
  }
  static bool matches(const io::NDMaterial& m);  // defined below the type list
  static typename Material::Params params(const io::NDMaterial& m) {
    const auto& d = std::get<io::PressureIndependMultiYieldData>(m.data);
    if (d.nd != 3) throw io::UnsupportedComponent("unsupported component: PressureIndependMultiYield nd = 2");
    PressureIndependMultiYieldInput in{};
    in.rho = d.rho;
    in.refShearModul = d.refShearModul;
    in.refBulkModul = d.refBulkModul;
    in.cohesi = d.cohesi;
    in.peakShearStra = d.peakShearStra;
    in.frictionAng = d.frictionAng;
    in.refPress = d.refPress;
    in.pressDependCoe = d.pressDependCoe;
    in.numberOfYieldSurf = surfaces(m);
    in.gredu = d.noYieldSurf < 0 ? d.pairs.data() : nullptr;
    typename Material::Params p{};
    const int r = Material::setup(in, p);
    const std::string where = "/materials/nd/" + std::to_string(m.tag);
    switch (r) {
      case Material::Ok: return p;
      case Material::InvalidFrictionAngle:
        throw io::ModelError(where + ": the G/Gmax curve and refPress give sin(friction angle) outside [0, 1] "
                             "(OpenSees: \"Invalid friction angle, please modify ref. pressure or G/Gmax curve\")");
      case Material::NegativeModulus:
        throw io::ModelError(where + ": the G/Gmax curve gives a yield surface with a plastic modulus <= 0 "
                             "(OpenSees: \"has plastic modulus < 0. Please modify G/Gmax curve\")");
      case Material::TooFewPairs:
        throw io::ModelError(where + ": a user-defined backbone needs at least two (gamma, G/Gmax) pairs");
      default:
        throw io::ModelError(where + ": invalid PressureIndependMultiYield parameters");
    }
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
// Uniform damping (docs/design.md): the material of an element in a Uniform damping region is
// wrapped, UniformDamped<Base, NF>, in the smallest filter bucket NF that holds the region's
// fit. Another material or bucket is one more line in Materials below.
template <class BaseEntry, int NF>
struct UniformDampedEntry {
  using Material = UniformDamped<typename BaseEntry::Material, NF>;
  static constexpr int filters = NF;
  static constexpr const char* name = BaseEntry::name;
  static bool matches(const io::NDMaterial& m) { return BaseEntry::matches(m); }
  static typename Material::Params params(const io::NDMaterial& m, const io::UniformDampingRegion& r,
                                          const UniformDampingFilters& fl) {
    typename Material::Params p{};
    p.mat = BaseEntry::params(m);
    p.fl = fl;
    p.eta = uniform_damping_eta(r.ratio);
    p.ta = r.activate_time;
    p.td = r.deactivate_time;
    return p;
  }
};
template <class M>
inline constexpr bool is_damped_entry = requires { M::filters; };

using Materials = TypeList<ElasticIsotropicEntry, PressureDependMultiYieldEntry<20>, PressureDependMultiYieldEntry<40>,
                           PressureIndependMultiYieldEntry<20>, PressureIndependMultiYieldEntry<40>,
                           UniformDampedEntry<ElasticIsotropicEntry, 8>, UniformDampedEntry<ElasticIsotropicEntry, 32>>;

// The smallest bucket that holds the material's surfaces.
template <int NS>
bool PressureDependMultiYieldEntry<NS>::matches(const io::NDMaterial& m) {
  const int n = surfaces(m);
  if (n < 0) return false;
  if constexpr (NS == 20) return n <= 20;
  else return n > 20 && n <= NS;
}
template <int NS>
bool PressureIndependMultiYieldEntry<NS>::matches(const io::NDMaterial& m) {
  const int n = surfaces(m);
  if (n < 0) return false;
  if constexpr (NS == 20) return n <= 20;
  else return n > 20 && n <= NS;
}
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

// The pair index of (element type, material), or -1 if no registered pair matches. filters:
// 0 for an element without Uniform damping; else the region's fitted filter count, which
// picks the smallest damped bucket that holds it.
inline int pair_index(io::ElementType type, const io::NDMaterial& m, int filters = 0) {
  int found = -1, bucket = 1 << 30;
  for_each_pair([&]<class E, class M>(int index) {
    if (E::type != type || !M::matches(m)) return;
    if constexpr (is_damped_entry<M>) {
      if (filters > 0 && filters <= M::filters && M::filters < bucket) {
        found = index;
        bucket = M::filters;
      }
    } else {
      if (filters == 0) found = index;
    }
  });
  return found;
}

// The filters of a Uniform damping region: FemoraX's fit (the default) or OpenSees'.
inline UniformDampingFilters uniform_damping_filters(const io::UniformDampingRegion& r) {
  UniformDampingFilters fl;
  const bool ok = r.fit == "opensees" ? uniform_damping_fit(r.freq1, r.freq2, 0.05, fl)
                                      : uniform_damping_fit_femorax(r.ratio, r.freq1, r.freq2, 0.05, fl);
  if (!ok) {
    throw io::ModelError("Uniform damping region \"" + r.id + "\": no filter fit within " +
                         std::to_string(kUniformDampingMaxFilters) + " filters for the band " + std::to_string(r.freq1) +
                         "-" + std::to_string(r.freq2) + " Hz");
  }
  return fl;
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
