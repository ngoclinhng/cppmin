// Copyright 2018
//
// Author: mail2ngoclinh@gmail.com
//
// BLAS routines used by cppmin

#ifndef CPPMIN_CPPMIN_BLAS_H_
#define CPPMIN_CPPMIN_BLAS_H_

#ifdef CPPMIN_USE_ACCELERATE
#include <Accelerate/Accelerate.h>
#else
extern "C" {
  #include <cblas.h>
}
#endif

namespace cppmin {

// Y <- alpha * X + Y
inline
void cppmin_axpy(const int N, const float alpha, const double* X,
                 double* Y) {
  cblas_daxpy(N, alpha, X, 1, Y, 1);
}

// X <- alpha * X
inline
void cppmin_scal(const int N, const double alpha, double* X) {
  cblas_dscal(N, alpha, X, 1);
}

// Compute dot product of two vectors X & Y
inline
double cppmin_dot(const int N, const double* X, const double* Y) {
  return cblas_ddot(N, X, 1, Y, 1);
}

// L2-norm of a vector X
inline
double cppmin_nrm2(const int N, const double* X) {
  return cblas_dnrm2(N, X, 1);
}
}  // namespace cppmin
#endif  // CPPMIN_CPPMIN_BLAS_H_

