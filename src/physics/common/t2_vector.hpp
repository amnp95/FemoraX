#pragma once

// Second-order tensors in vector form and OpenSees' Vector arithmetic, for the multi-yield
// soil materials.
//
// Port of OpenSees SRC/material/nD/soil/T2Vector.cpp and the parts of
// SRC/matrix/Vector.cpp those materials use, OpenSees commit
// 955ceb0a55f7b5764856ea82fc9a9cc7f07a1eea. Each function performs the same operations in
// the same order as the original, so the materials built on them reproduce OpenSees bit for
// bit on host compilers that do not contract multiply-adds.
//
// A T2 keeps what an OpenSees T2Vector keeps: the 6-component vector (11 22 33 12 23 31,
// tensor shear), its deviator and its volume (mean). The two setData forms derive them in
// different orders, so all 13 values are kept rather than recomputed.

#include <cfloat>
#include <cmath>
#include <limits>

#include "physics/common/strided.hpp"
#include "physics/macros.hpp"

namespace femorax::soil {

inline constexpr double LOW_LIMIT = 20. * DBL_EPSILON;         // T2Vector.h
inline constexpr double UP_LIMIT = 1.0e+30;                    // T2Vector.h
inline constexpr double LOCK_VALUE = 1.0e+30;                  // MultiYieldSurface.h
inline constexpr double VECTOR_VERY_LARGE_VALUE = 1.0e200;     // Vector.h
inline constexpr double pi = 3.14159265358979;                 // PressureDependMultiYield.cpp

// --- Vector (size 6) ----------------------------------------------------------------------
FEMORAX_HD inline void vcopy(double* a, const double* b) {
  for (int i = 0; i < 6; i++) a[i] = b[i];
}
FEMORAX_HD inline void vzero(double* a) {
  for (int i = 0; i < 6; i++) a[i] = 0.0;
}

// Vector::addVector(thisFact, other, otherFact), with its special cases.
FEMORAX_HD inline void add_vector(double* a, double thisFact, const double* other, double otherFact) {
  if (otherFact == 0.0) {
    if (thisFact == 1.0) return;
    if (thisFact == 0.0) {
      vzero(a);
      return;
    }
    for (int i = 0; i < 6; i++) a[i] *= thisFact;
    return;
  }
  if (thisFact == 1.0) {
    if (otherFact == 1.0) {
      for (int i = 0; i < 6; i++) a[i] += other[i];
    } else if (otherFact == -1.0) {
      for (int i = 0; i < 6; i++) a[i] -= other[i];
    } else {
      for (int i = 0; i < 6; i++) a[i] += other[i] * otherFact;
    }
  } else if (thisFact == 0.0) {
    if (otherFact == 1.0) {
      for (int i = 0; i < 6; i++) a[i] = other[i];
    } else if (otherFact == -1.0) {
      for (int i = 0; i < 6; i++) a[i] = -(other[i]);
    } else {
      for (int i = 0; i < 6; i++) a[i] = other[i] * otherFact;
    }
  } else {
    if (otherFact == 1.0) {
      for (int i = 0; i < 6; i++) a[i] = a[i] * thisFact + other[i];
    } else if (otherFact == -1.0) {
      for (int i = 0; i < 6; i++) a[i] = a[i] * thisFact - other[i];
    } else {
      for (int i = 0; i < 6; i++) a[i] = a[i] * thisFact + other[i] * otherFact;
    }
  }
}

FEMORAX_HD inline void vsub_eq(double* a, const double* b) {  // operator-=
  for (int i = 0; i < 6; i++) a[i] -= b[i];
}
FEMORAX_HD inline void vadd_eq(double* a, const double* b) {  // operator+=
  for (int i = 0; i < 6; i++) a[i] += b[i];
}
FEMORAX_HD inline void vmul_eq(double* a, double f) {  // operator*=
  for (int i = 0; i < 6; i++) a[i] *= f;
}
FEMORAX_HD inline void vdiv_eq(double* a, double f) {  // operator/= (a zero divisor gives 1e200)
  if (f == 0.0) {
    for (int i = 0; i < 6; i++) a[i] = VECTOR_VERY_LARGE_VALUE;
  } else {
    for (int i = 0; i < 6; i++) a[i] /= f;
  }
}

// T2Vector.cpp operator&&: the double-dot product of two tensors in vector form.
FEMORAX_HD inline double ddot(const double* a, const double* b) {
  double result = 0.;
  for (int i = 0; i < 3; i++) result += a[i] * b[i] + 2 * a[i + 3] * b[i + 3];
  return result;
}

// --- T2Vector -------------------------------------------------------------------------------
struct T2 {
  double t2[6];
  double dev[6];
  double vol;

  // setData(const Vector& init, int isEngrgStrain = 0)
  FEMORAX_HD void set(const double* init, int isEngrgStrain = 0) {
    vcopy(t2, init);
    vol = (t2[0] + t2[1] + t2[2]) / 3.0;
    for (int i = 0; i < 3; i++) {
      dev[i] = t2[i] - vol;
      dev[i + 3] = t2[i + 3];
      if (isEngrgStrain == 1) {
        dev[i + 3] /= 2.;
        t2[i + 3] /= 2.;
      }
    }
  }
  // setData(const Vector& deviat, double volume)
  FEMORAX_HD void set_dev(const double* deviat, double volume) {
    vol = volume;
    const double devolum = (deviat[0] + deviat[1] + deviat[2]) / 3.;
    for (int i = 0; i < 3; i++) {
      dev[i] = deviat[i] - devolum;
      dev[i + 3] = deviat[i + 3];
      t2[i] = dev[i] + vol;
      t2[i + 3] = dev[i + 3];
    }
  }
  FEMORAX_HD void zero() {
    vzero(t2);
    vzero(dev);
    vol = 0.0;
  }
  // t2Vector(1), deviator(1): engineering shear.
  FEMORAX_HD void t2_engr(double* out) const {
    vcopy(out, t2);
    for (int i = 0; i < 3; i++) out[i + 3] *= 2.;
  }
  FEMORAX_HD double volume() const { return vol; }
  FEMORAX_HD double t2_length() const { return std::sqrt(ddot(t2, t2)); }
  FEMORAX_HD double deviator_length() const { return std::sqrt(ddot(dev, dev)); }
  FEMORAX_HD double octahedral_shear(int isEngrgStrain = 0) const {
    if (isEngrgStrain) return 2. * std::sqrt(1. / 3.) * deviator_length();
    return std::sqrt(1. / 3.) * deviator_length();
  }
  // deviatorRatio (OpenSees exits when |volume| + |residualPress| <= LOW_LIMIT; here the
  // division proceeds and gives inf or nan).
  FEMORAX_HD double deviator_ratio(double residualPress = 0.) const {
    return std::sqrt(3. / 2. * ddot(dev, dev)) / (std::fabs(vol) + std::fabs(residualPress));
  }
  FEMORAX_HD void unit_t2(double* out) const {
    vcopy(out, t2);
    const double length = t2_length();
    if (length <= LOW_LIMIT) vdiv_eq(out, LOW_LIMIT);
    else vdiv_eq(out, length);
  }
  FEMORAX_HD bool is_zero() const {
    for (int i = 0; i < 6; i++)
      if (t2[i] != 0.0) return false;
    return true;
  }

  // 13 values in a strided state array at offset k.
  static constexpr int SIZE = 13;
  FEMORAX_HD void load(Strided<const double> s, int k) {
    for (int i = 0; i < 6; i++) t2[i] = s[k + i];
    for (int i = 0; i < 6; i++) dev[i] = s[k + 6 + i];
    vol = s[k + 12];
  }
  FEMORAX_HD void store(Strided<double> s, int k) const {
    for (int i = 0; i < 6; i++) s[k + i] = t2[i];
    for (int i = 0; i < 6; i++) s[k + 6 + i] = dev[i];
    s[k + 12] = vol;
  }
};

// MultiYieldSurface.cpp secondOrderEqn. OpenSees exits on A == 0, complex or negative
// roots; here `ok` is set false and the result is NaN, so the step fails instead.
FEMORAX_HD inline double second_order_eqn(double A, double B, double C, int i, bool& ok) {
  (void)i;
  const double nan = std::numeric_limits<double>::quiet_NaN();
  if (A == 0) {
    ok = false;
    return nan;
  }
  if (C == 0) return 0;
  if (B == 0) {
    if (C / A > 0) {
      ok = false;
      return nan;
    }
    return std::sqrt(-C / A);
  }
  double determ, val1, val2, val;
  determ = B * B - 4. * A * C;
  if (determ < 0) {
    ok = false;
    return nan;
  }
  if (B > 0) val1 = (-B - std::sqrt(determ)) / (2. * A);
  else val1 = (-B + std::sqrt(determ)) / (2. * A);
  val2 = C / (A * val1);
  if (val1 < 0 && val2 < 0) {
    if (std::fabs(val1) < LOW_LIMIT) val1 = 0.;
    else if (std::fabs(val2) < LOW_LIMIT) val2 = 0.;
  }
  if (val1 < 0 && val2 < 0) {
    ok = false;
    return nan;
  }
  if (val1 < 0) return val2;
  else if (val2 < 0) return val1;
  else {
    val = val1;
    if (val > val2) val = val2;
    return val;
  }
}

}  // namespace femorax::soil
