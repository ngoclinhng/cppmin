// Copyright 2018
//
// Author: mail2ngoclinh@gmail.com
//
// Quadratic and Cubic interpolations

#ifndef CPPMIN_INTERNAL_INTERPOLATION_H_
#define CPPMIN_INTERNAL_INTERPOLATION_H_

namespace cppmin {
namespace internal {

// Quadratic interpolation
//
// Returns the value of x that minimizes this quadratic function:
//
//  f(x) = a * x^2 + b * x + c
//
// In which a, b, c are interpolated so that:
//
//  f(0) = f0
//  f'(0) = g0
//  f(x) = fx.
inline
double QuadraticInterpolate(const double f0, const double g0,
                            const double x, const double fx) {
  // Given that f(x) is the quadratic function:
  //
  //  f(x)  = a * x^2 + b * x + c
  //
  // and
  //
  //  f(0) = f0
  //  f'(0) = g0
  //  f(x) = fx
  //
  // We have:
  //
  //  c = f0
  //  b = g0
  //  a = (fx - b * x - c) / x^2 = (fx - g0 * x - f0) / x^2.
  //
  // This quadratic function has the minimum at
  //
  //  x* = -b / 2a (where the gradient vanishes)
  //              -g0 * x^2
  //     = -----------------------
  //       2.0 * (fx - g0 *x - f0)
  return g0 * x * x / (2.0 * (g0 * x + f0 - fx));
}

// Cubic interpolation
//
// We want to find the value of x that minimizes this quadratic function:
//
//  f(x) = a * x^3 + b * x^2 + c * x + d.
//
// In which a, b, c, d are coefficients such that:
//
//  f(0)  = f0
//  f'(0) = g0
//  f(x1) = f1
//  f(x2) = f2
inline
double CubicInterpolate(const double f0, const double g0,
                        const double f1, const double x1,
                        const double f2, const double x2) {
  // Denote:
  //
  //   x1s = x1 * x1
  //   x2s = x2 * x2;
  //   x1c = x1 * x1 * x1
  //   x2c = x2 * x2 * x2
  //
  // We evaluate:
  //
  // | a |            1             | x2s  -x1s |   | f1 - f0 - x1 * g0 |
  // |   | = -------------------- * |           | * |                   |
  // | b |   x1s * x2s * (x1 - x2)  | -x2c  x1c |   | f2 - f0 - x2 * g0 |
  const double x1s = x1 * x1;
  const double x2s = x2 * x2;
  const double x1c = x1s * x1;
  const double x2c = x2s * x2;
  const double tmp1 = f1 - f0 - x1 * g0;
  const double tmp2 = f2 - f0 - x2 * g0;
  const double tmp = 1.0 / (x1s * x2s * (x1 - x2));

  double a, b, d;

  a = tmp * (x2s * tmp1 - x1s * tmp2);
  b = tmp * (x1c * tmp2 - x2c * tmp1);

  if (a == 0.0) {  // cubic is quadratic
    return -g0 / (2.0 * b);
  } else {
    d = b * b - 3.0 * a * g0;   // discriminant
    return (std::sqrt(d) - b)/(3.0 * a);
  }
}

// Contract step size between [lo, hi]
inline double ContractStep(const double step_size,
                           const double lo,
                           const double hi) {
  if (step_size < lo) {
    return lo;
  } else if (step_size > hi) {
    return hi;
  } else {
    return step_size;
  }
}
}  // namespace internal
}  // namespace cppmin
#endif  // CPPMIN_INTERNAL_INTERPOLATION_H_
