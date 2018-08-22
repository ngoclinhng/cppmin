// Copyright 2018
//
// Author: mail2ngoclinh@gmail.com

#ifndef CPPMIN_TYPES_H_
#define CPPMIN_TYPES_H_

namespace cppmin {

enum LineSearchDirectionType {
  // Search direction is the negative of gradient, i.e
  //
  //  p_k = -f'(x_k)
  STEEPEST_DESCENT,

  // Search direction is initialized to the negative of gradient at the
  // starting point, i.e:
  //
  //  p_0 = -f'(x_0)
  //
  // Subsequently, it is updated as follows:
  //
  //  p_k = -f'(x_k) + beta * p_{k-1}
  //
  // in which beta is the Fletcher-Reeves parameter:
  //
  //          |f'(x_k)|^2     ----> Squares L2-norm of gradient at x_k
  //  beta = ---------------
  //         |f'(x_{k-1})|^2  ----> squares L2-norm of gradient at x_{k-1}
  FLETCHER_REEVES_CONJUGATE_GRADIENT,

  // Search direction is initialized to the negative of gradient at the
  // starting point, i.e:
  //
  //  p_0 = -f'(x_0)
  //
  // Subsequently, it is updated as follows:
  //
  //  p_k = -f'(x_k) + beta * p_{k-1}
  //
  // in which beta is the Polak-Ribiere parameter and is computed as:
  //
  //         f'(x_k) . [f'(x_k) - f'(x_{k-1})]
  //  beta = ---------------------------------
  //               |f'(x_{k-1})|^2
  //
  // Even though there is not yet a sound convergence theory to support
  // Polak-Ribiere conjugate gradients, it is the perferred method whenever
  // it comes to nonlinear conjugate gradients.
  POLAK_RIBIERE_CONJUGATE_GRADIENT,

  // Search direction at iteration k is:
  //
  //  p_k = -H_k . f'(x_k)
  //
  // in which H_k is the inverse Hessian (f''(x_k)) approximation and
  // is updated using the limited memory BFGS formula
  // See Jorge Nocedal, Stephen J. Wright Numerical Optimization (2rd edition)
  LBFGS
};

enum LineSearchType {
  // Armijo line search,
  ARMIJO,

  // Strong Wolfe line search
  WOLFE
};

// Interface for precoditioner
class Preconditioner {
 public:
  // No matter what preconditioner type we choose (e.g. Jacobi,
  // incomplete Cholesky ..), the first one thing that we need to be able to
  // do quicky is to compute the dot product between preconditioner
  // and a vector, i.e to solve this linear system as quickly as possible:
  //
  //  M.x = b <=> x = M`.b
  virtual void InverseDot(const size_t n, const double* b,
                          double* result) const = 0;

  // And the second thing is to update the preconditioner at a given point
  // x. Since this method might potentially change the internal data, it
  // cannot be const!
  virtual void Update(const size_t n, const double* x) = 0;
};
}  // namespace cppmin

#endif  // CPPMIN_TYPES_H_
