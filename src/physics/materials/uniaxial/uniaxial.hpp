#pragma once

// The uniaxial materials of a fiber, chosen per fiber at run time (docs/design.md, "Fiber
// sections"): one section kernel serves every mix of materials, instead of one kernel
// instantiation per combination. A fiber's material is a type and up to kMaxParam parameters;
// every fiber has kUniaxialState doubles of state per buffer (the largest of the registered
// materials; a material uses the first NSTATE of them).
//
// Adding a material: its header (the UniaxialMaterial concept of contracts.hpp), one enumerator,
// one case in uniaxial_update and uniaxial_initial_tangent, and its strain-path reference test.

#include "physics/contracts.hpp"
#include "physics/macros.hpp"
#include "physics/materials/uniaxial/concrete02.hpp"
#include "physics/materials/uniaxial/elastic_uniaxial.hpp"
#include "physics/materials/uniaxial/steel01.hpp"

namespace femorax {

enum class UniaxialType : int { Elastic = 0, Steel01 = 1, Concrete02 = 2 };

inline constexpr int kUniaxialMaxParam = 7;
inline constexpr int kUniaxialState = 8;

static_assert(ElasticUniaxial::NPARAM <= kUniaxialMaxParam && ElasticUniaxial::NSTATE <= kUniaxialState);
static_assert(Steel01::NPARAM <= kUniaxialMaxParam && Steel01::NSTATE <= kUniaxialState);
static_assert(Concrete02::NPARAM <= kUniaxialMaxParam && Concrete02::NSTATE <= kUniaxialState);

struct UniaxialParams {
  int type = 0;  // UniaxialType
  double v[kUniaxialMaxParam] = {};
};

template <UniaxialMaterial M>
FEMORAX_HD UniaxialParams uniaxial_params(UniaxialType type, const typename M::Params& p) {
  UniaxialParams u;
  u.type = static_cast<int>(type);
  M::pack(p, u.v);
  return u;
}

FEMORAX_HD inline void uniaxial_update(const UniaxialParams& u, Strided<const double> committed,
                                       Strided<double> trial, double strain, double& stress, double& tangent) {
  switch (static_cast<UniaxialType>(u.type)) {
    case UniaxialType::Steel01:
      Steel01::update(Steel01::unpack(u.v), committed, trial, strain, stress, tangent);
      break;
    case UniaxialType::Concrete02:
      Concrete02::update(Concrete02::unpack(u.v), committed, trial, strain, stress, tangent);
      break;
    case UniaxialType::Elastic:
    default:
      ElasticUniaxial::update(ElasticUniaxial::unpack(u.v), committed, trial, strain, stress, tangent);
      break;
  }
}

FEMORAX_HD inline double uniaxial_initial_tangent(const UniaxialParams& u) {
  switch (static_cast<UniaxialType>(u.type)) {
    case UniaxialType::Steel01:
      return Steel01::initial_tangent(Steel01::unpack(u.v));
    case UniaxialType::Concrete02:
      return Concrete02::initial_tangent(Concrete02::unpack(u.v));
    case UniaxialType::Elastic:
    default:
      return ElasticUniaxial::initial_tangent(ElasticUniaxial::unpack(u.v));
  }
}

}  // namespace femorax
