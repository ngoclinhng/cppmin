// Copyright 2018
//
// Author: mail2ngoclinh@gmail.com

#ifndef CPPMIN_INTERNAL_LINE_SEARCH_ALGORITHMS_H_
#define CPPMIN_INTERNAL_LINE_SEARCH_ALGORITHMS_H_

namespace cppmin {
namespace internal {

template<LineSearchDirectionType direction_type> class LineSearchAlgorithm;

// Steepest descent -------------------------------------------------------

template<>
class LineSearchAlgorithm <STEEPEST_DESCENT>{
 public:
  const LineSearchMinimizer::Options& options;

  explicit LineSearchAlgorithm(const LineSearchMinimizer::Options& options)
      : options(options) {}

  template<typename Function>
  void Minimize(const Function& function, double* solution,
                LineSearchMinimizer::Summary* summary = nullptr) const {
    const size_t n = function.n_variables();

    if (summary != nullptr) {
      summary->function_value_at_starting_point =
          function(solution);
      summary->total_num_iterations = 0;
    }

    // Init gradient
    double* gradient = new double[n];
    function.Gradient(solution, gradient);
    double gradient_norm2 = cppmin_dot(n, gradient, gradient);

    // Init line search
    LineSearch<Function>* line_search =
        LineSearch<Function>::Create(options);
    double step_size;

    // The steepest descent algorithm terminates if
    //
    //  gradient_norm2 <= epsilon
    //
    // or if
    //
    //  iter >= options().max_num_iterations
    const double epsilon = options.tolerance * options.tolerance *
        gradient_norm2;
    size_t iter = 0;

    while (iter < options.max_num_iterations && gradient_norm2 > epsilon) {
      // Compte step size using -gradient as a search diretion, i.e
      //
      //  step_sise <- min function(solution - s * gradient), s > 0
      step_size = line_search->Search(function, solution, gradient, -1.0);

      // Update solution
      // solution = step_size * (-gradient) + solution
      cppmin_axpy(n, -step_size, gradient, solution);

      // Update gradient
      function.Gradient(solution, gradient);
      gradient_norm2 = cppmin_dot(n, gradient, gradient);

      ++iter;

      if (summary != nullptr) {
        ++summary->total_num_iterations;
      }
    }

    if (summary != nullptr) {
      summary->function_value_at_estimated_solution =
          function(solution);
    }

    delete[] gradient;
    delete line_search;
  }
};

// Preconditioned nonlinear conjugate gradient with Polak-Ribiere parameter
// See https://www.cs.cmu.edu/~quake-papers/painless-conjugate-gradient.pdf
// ------------------------------------------------------------------------

template<>
class LineSearchAlgorithm <POLAK_RIBIERE_CONJUGATE_GRADIENT>{
 public:
  const LineSearchMinimizer::Options& options;

  explicit LineSearchAlgorithm(const LineSearchMinimizer::Options& options)
      : options(options) {}

  template<typename Function>
  void Minimize(const Function& function, double* solution,
                LineSearchMinimizer::Summary* summary = nullptr) const {
  }
};

// Preconditioned nonlinear conjugate gradients with Fletcher-Reeves
// parameter
// See https://www.cs.cmu.edu/~quake-papers/painless-conjugate-gradient.pdf
// -------------------------------------------------------------------------

template<>
class LineSearchAlgorithm <FLETCHER_REEVES_CONJUGATE_GRADIENT>{
 public:
  const LineSearchMinimizer::Options& options;

  explicit LineSearchAlgorithm(const LineSearchMinimizer::Options& options)
      : options(options) {}

  template<typename Function>
  void Minimize(const Function& function, double* solution,
                LineSearchMinimizer::Summary* summary = nullptr) const {
  }
};

// LBFGS --------------------------------------------------------------------

template<>
class LineSearchAlgorithm <LBFGS>{
 public:
  const LineSearchMinimizer::Options& options;

  explicit LineSearchAlgorithm(const LineSearchMinimizer::Options& options)
      : options(options) {}

  template<typename Function>
  void Minimize(const Function& function, double* solution,
                LineSearchMinimizer::Summary* summary = nullptr) const {
  }
};

}  // namespace internal
}  // namespace cppmin
#endif  // CPPMIN_INTERNAL_LINE_SEARCH_ALGORITHMS_H_
