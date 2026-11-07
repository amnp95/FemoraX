#pragma once

// A time as OpenSees' Tcl `getTime` returns it: printed with "%f", then read back by Tcl.
// Femora ends a stage with `while {[getTime] < %f} { analyze 1 dt }`, so the loop compares
// both times rounded to six decimals; matching that gives the same number of steps.

#include <cstdio>
#include <cstdlib>

namespace femorax::exec {

inline double as_tcl_time(double t) {
  char buf[64];
  std::snprintf(buf, sizeof(buf), "%f", t);
  return std::strtod(buf, nullptr);
}

}  // namespace femorax::exec
