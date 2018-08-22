// Copyright 2018
//
// Author: mail2ngoclinh@gmail.com
//
// LineSearchMinimizer solves the unconstrained problem
//
//  min f(x) in which f : R^n -> R is (at least) a continuously
//  differentiable Lipschitz function
//
// using the following scheme:
//
//  1. Setup:
//
//     - Starting point x
//     - Initialize a search direction search_direction
//
//  2. Repeat until some stopping criteria are met:
//
//     - Use line search (ARMIJO or WOLFE) to compute the step size along
//       the search direction search_direction.
//
//     - Update solution:
//
//       x <- x + step_size * search_direction
//
//     - Update search_direction based on whether it is a steepest descent,
//      a nonlinear conjugate gradients, or lbfgs search direction.
//
// Example:
//
// Estimating the global minimizer of Rosenbrock function:
//
//  f(x1, x2) = 100(x2 - x1^2)^2 + (x1 - 1)^2
//
// Create Rosenbrock function:
//
// struct Rosenbrock {
//  int n_variables() const { return 2; }
//
//  double operator()(const double* x) const {
//   const double x1 = x[0];
//   const double x2 = x[1];
//   const double tmp1 = x2 - x1 * x1;
//   const double tmp2 = x1 - 1.0;
//   return 100.0 * tmp1 * tmp1 + tmp2 * tmp2
//  }
//
//  void Gradient(double* gradient, const double *x) const {
//   const double x1 = x[0];
//   const double x2 = x[1];
//   const double tmp = x2 - x1 * x1
//   gradient[0] = 200.0 * tmp * (-2.0 * x1) + 2.0 * (x1 - 1.0);
//   gradient[1] = 200.0 * tmp;
//  }
// };
//
// LineSearchMinimizer  minimizer;
// Rosenbrock rosen;
//
// double solution[2] = {0.0, 0.0};  // starting point
// minimizer.Minimize(rosen, solution)

#ifndef CPPMIN_LINE_SEARCH_MINIMIZER_H_
#define CPPMIN_LINE_SEARCH_MINIMIZER_H_

#include <cstddef>
#include <cstring>
#include <iostream>

#include "cppmin_blas.h"  // NOLINT
#include "types.h"  // NOLINT

namespace cppmin {

class LineSearchMinimizer {
 public:
  struct Options {
    // Line search type to compute step_size at each iteration
    // By default strong Wolfe line search will be used.
    LineSearchType line_search_type;  //  = WOLFE;

    // By default Polack and Ribie're Nonlinear conjugate gradient
    // will be used
    LineSearchDirectionType line_search_direction_type;

    // Preconditioner for preconditioned nonlinear conjugate gradient descent
    // By default it is set to nullptr meaning that the (unconditioned)
    // nonlinear conjugate gradient descent will be used (not recommended)
    //
    // To get the most out of nonlinear conjugate gradients method, it is
    // recommended that the matrix should be conditioned, i.e to find
    // a symmetric, positive-definite 'matrix' M such that:
    //
    //  - M is a good approximation for Hessian matrix f''(x).
    //
    //  - M` (inverse of matrix M) is easy to compute or it is easy
    //    to solve this linear equation: Mx = b.
    //
    // Most commonly used preconditioners are:
    //
    //  - Jacobi: M = Diag(f''(x_k))
    //
    //  - M is the result of incomplete Cholesky factorization of Hessian
    //    matrix f''(x_k).
    Preconditioner* preconditioner;  //  = nullptr;

    // LineSearchMinimizer terminates if:
    //
    //  |current_residual| <= tolerance * |initial_residual|
    double tolerance;  //  = 1e-4;

    // LineSearchMinimizer minimizes an objective function f by generating
    // a sequence of points {x_k} such that
    //  f(x_{k+1}) <= f(x_k), for all k >=0.
    //
    // This parameter controls the maximum number of points generated.
    size_t max_num_iterations;  //  = 50;

    // line search specific parameters ------------------------------------

    // Armijo and Wolfe line search parameters.

    // Initial step_size. It is recommeded that the initial trial should
    // be unit step, i.e try step_size = 1.0 first and foremost.
    double initial_step_size;  //  = 1.0;

    // We want to find a step_size which results in sufficient decrease of
    // the objective function f along the search direction p_k. More
    // precisely, we are looking for a step size s.t
    //
    // f(x_k + step_size * p_k) <= f(x_k) +
    //                   sufficient_decrease * step_size * f'(x_k) * p_k
    // Or equivalently:
    //
    // phi(step_size) <= phi(0) + sufficient_decrease * step_size * phi'(0)
    double sufficient_decrease;  //  = 1e-4;

    // Say, at the current iteration of Armijo / Wolfe line search we found
    // a step_size that satifies either the sufficient decrease condition
    // (Armijo) or the sufficient decrease condition and the curvature
    // condition (Wolfe), then the next_step_size is determined as follows:
    //
    // if step_size <= max_step_contraction * previous_step_size:
    //    next_step_size = max_step_contraction * previous_step_size
    // else if step_size >= min_step_contraction * previous_step_size:
    //    next_step_size = min_step_contraction * previous_step_size
    // else
    //    next_step_size = step_size.
    //
    // Note that:
    //  0 < max_step_contraction < min_step_contraction < 1
    double max_step_contraction;  //  = 0.1;
    double min_step_contraction;  //  = 0.5;

    // If during the line search, the step_size falls below this value,
    // it is set to this value and the line search terminates.
    double min_step_size;  //  = 1e-9;

    // Maximum number of trial step size iterations during each line search,
    // If a step size satisfying the search coditions connot be found
    // within this number of trials, the line search will terminate.
    size_t max_num_step_size_trials;  //  = 20;

    // Wolfe-specific line search parameters.

    // The Wolfe conditions consist of the Armijo sufficient decrease
    // condition, and an additional requirement that the step_size be chosen
    // s.t:
    //
    //  phi'(step_size) >= sufficient_curvature_decrease * phi'(0)
    //
    // Note that: We only implement the Wolfe conditions NOT the strong
    // Wolfe conditions.
    double sufficient_curvature_decrease;  // = 0.9;

    // The Wolfe line search algorithm is similar to that of the Armijo
    // line search algorithm until it found a step size armijo_step_size
    // satisfying the sufficient decrease condition. At this point the
    // Armijo line search terminates while the Wolfe line search continues
    // the search in the interval [armijo_step_size, max_step_size]
    // (the ZOOM stage) until it found a step_size which satifies the Wolfe
    // condition.
    //
    // Note that, according to [1], the interval
    // [armijo_step_size, max_step_size] contains a step_size satisfying
    // the Wolfe condition.
    //
    // [1] Nocedal J., Wright S., Numerical Optimization, 2nd Ed., Springer, 1999.  // NOLINT
    double max_step_size;  // = 4.0;

    // At each iteration in the ZOOM stage of the Wolfe line search, we
    // enlarge the current step_size by multiplying it with
    // max_step_expansion, so that we have
    //
    //  next_step_size = step_size * max_step_expansion
    //
    // If this next_step_size violates the sufficient decrease condition
    // we go to the REFINE stage (see below), if it meets the Wolfe
    // condition we return it, otherwise keep expanding step size.
    double max_step_expansion;  // = 2.0;

    // We only reach the REFINE stage if in the ZOOM stage the step
    // expansion causes the next_step_size to violates the sufficient
    // decrease condition, once we enter this stage we have this interval:
    //
    //  [lo, hi]
    //
    // in which lo satisfies the sufficient decrease condition wile
    // the hi doesn't.
    //
    // At each iteration we'll use quadratic interpolation to generate
    // our next trial step_size (within this interval). If this step_size
    // statifies the Wolfe condition we're done, othersie we update
    // the search interval (by replacing either left endpoint or
    // or right endpoint) until its length <=
    // min_step_size_search_interval_length
    double min_step_size_search_interval_length;  // = 1e-3;

    // Default options
    // This is odd but it only happens once, and since we don't want you to
    // get in-class initialization of non-static data member warning.
    Options()
        : line_search_type(WOLFE),
          line_search_direction_type(POLAK_RIBIERE_CONJUGATE_GRADIENT),
          preconditioner(nullptr),
          tolerance(1e-4),
          max_num_iterations(50),
          initial_step_size(1.0),
          sufficient_decrease(1e-4),
          max_step_contraction(0.1),
          min_step_contraction(0.5),
          min_step_size(1e-9),
          max_num_step_size_trials(20),
          sufficient_curvature_decrease(0.9),
          max_step_size(4.0),
          max_step_expansion(2.0),
          min_step_size_search_interval_length(1e-3) {}
  };

  struct Summary {
    double function_value_at_starting_point;
    double function_value_at_estimated_solution;
    size_t total_num_iterations;

    Summary()
        : function_value_at_starting_point(0.0),
          function_value_at_estimated_solution(0.0),
          total_num_iterations(0) {}

    friend std::ostream& operator<<(std::ostream& os,
                                    const Summary& summary) {
      os << "Function value at starting point: "
         << summary.function_value_at_starting_point << "\n";
      os << "Function value at estimated solution: "
         << summary.function_value_at_estimated_solution << "\n";
      os << "Total number of iterations: "
         << summary.total_num_iterations;
      return os;
    }
  };

  // Construct a LineSearch object with default options
  LineSearchMinimizer() : options_(LineSearchMinimizer::Options()) {}

  // Construct a LineSearch object with custom options
  explicit LineSearchMinimizer(const LineSearchMinimizer::Options& options)
      : options_(options) {}

  // Given an objective function of type Function, minimizes it, and
  // stores the result in solution.
  template<typename Function>
  void Minimize(const Function& function, double* solution,
                LineSearchMinimizer::Summary* summary = nullptr) const;

 private:
  LineSearchMinimizer::Options options_;
};
}  // namespace cppmin

#include "internal/interpolation.h"
#include "internal/phi_function.h"
#include "internal/line_search.h"
#include "internal/line_search_impl.h"
#include "internal/line_search_algorithms.h"
#include "internal/line_search_minimizer_impl.h"

#endif  // CPPMIN_LINE_SEARCH_MINIMIZER_H_
