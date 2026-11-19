#pragma once

// Local time stepping levels (docs/design.md, "Local time stepping", step 2): from every
// element's own stable step, each element's level, each equation's level and the elements each
// level's substeps evaluate. Host code; computed at the start of an explicit stage (the stable
// steps depend on the scheme, the damping and the material state). Nothing uses the levels yet.
//
// Level l steps with dt / ratio^l. An element takes the coarsest level whose step is at most
// `margin` times its own stable step; an equation takes the finest level of the elements that
// touch it; an element is evaluated in the substeps of every level up to the finest level of
// its equations (the stabilized scheme applies the stiffness to the fine DOFs' values: the fine
// elements and the ring of coarser elements touching a fine DOF).

#include <cstdint>
#include <vector>

#include "compile/batches.hpp"

namespace femorax::compile {

struct LtsOptions {
  int ratio = 2;         // step ratio between neighbouring levels
  double margin = 0.9;   // an element's step is at most margin * its stable step
  int max_levels = 8;    // levels 0 .. max_levels - 1; finer needs are reported, not met
};

struct LtsLevels {
  int levels = 1;  // levels used (1: everything at dt)
  // Per batch, per element (batch order, real elements only):
  std::vector<std::vector<std::int8_t>> element_level;  // from the element's own stable step
  std::vector<std::vector<std::int8_t>> touched_level;  // the finest level of its equations
  std::vector<std::int8_t> equation_level;              // [num_equations]
  std::vector<std::int64_t> elements_at_level;          // by element_level
  std::vector<std::int64_t> equations_at_level;         // by equation_level
  std::int64_t elements = 0;
  // Elements whose stable step needs a level finer than max_levels allows (they would be
  // unstable at the finest level's step), and the smallest stable step over all elements.
  std::int64_t too_fine = 0;
  double min_element_dt = 0.0;
  // Element evaluations per coarse step: with a global step at the finest level's step every
  // element runs ratio^(levels-1) times; with LTS each runs ratio^touched_level times.
  double global_work = 0.0, lts_work = 0.0;
  double speedup() const { return lts_work > 0.0 ? global_work / lts_work : 1.0; }
};

// element_dt[b][e]: the stable step of element e of batch b (c.batches order, real elements).
// dt: the coarse (level 0) step.
LtsLevels assign_lts_levels(const Compiled& c, const std::vector<std::vector<double>>& element_dt, double dt,
                            const LtsOptions& opt = {});

}  // namespace femorax::compile
