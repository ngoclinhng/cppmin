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
class LineSearchAlgorithm<STEEPEST_DESCENT>{
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
class LineSearchAlgorithm<POLAK_RIBIERE_CONJUGATE_GRADIENT>{
 public:
  const LineSearchMinimizer::Options& options;

  explicit LineSearchAlgorithm(const LineSearchMinimizer::Options& options)
      : options(options) {}

  template<typename Function>
  void Minimize(const Function& function, double* solution,
                LineSearchMinimizer::Summary* summary = nullptr) const {
    const size_t n = function.n_variables();

    if (summary != nullptr) {
      summary->function_value_at_starting_point = function(solution);
      summary->total_num_iterations = 0;
    }

    // Init residual to -gradient
    double* residual = new double[n];
    function.Gradient(solution, residual);
    cppmin_scal(n, -1.0, residual);

    // Init preconditioner (M)
    if (options.preconditioner != nullptr) {
      options.preconditioner->Update(n, solution);
    }

    // Cache the term M`r
    double* M_inv_r = new double[n];
    if (options.preconditioner != nullptr) {
      options.preconditioner->InverseDot(n, residual, M_inv_r);
    } else {
      // M is the identity matrix, so that M_inv_r = r
      std::memcpy(M_inv_r, residual, n * sizeof(double));
    }

    // Init search direction
    // We have:
    //  d_hat = E'.d & r_hat = E`.r
    // So that:
    //  E'.d = E`.r <=> d = (E')`.E`.r = (EE')`.r = M`.r
    double* search_direction = new double[n];
    std::memcpy(search_direction, M_inv_r, n * sizeof(double));

    // preconditioned residual norm
    // We have:
    //  r_hat = E`.r => |r_hat|^2 = r_hat'.r_hat
    //                            = (E`.r)'.(E`.r)
    //                            = r'.(EE')`.r
    //                            = r'.M`.r
    double preconditioned_residual_norm2 = cppmin_dot(n, residual, M_inv_r);
    double previous_preconditioned_residual_norm2;
    double mid_preconditioned_residual_norm2;

    // Polak-Ribiere parameter
    double polak_ribiere_beta;

    // Init line search
    LineSearch<Function>* line_search =
        LineSearch<Function>::Create(options);
    double step_size;

    // The algorithm terminates if
    //  precondition_residual_norm2 <= epsilon
    // or if
    //  iter >= options().max_num_iterations
    const double epsilon = options.tolerance * options.tolerance *
        preconditioned_residual_norm2;
    size_t iter = 0;

    while (iter < options.max_num_iterations &&
           preconditioned_residual_norm2 > epsilon) {
      // Compute step size
      step_size = line_search->Search(function, solution, search_direction);

      // Update solution
      //  solution = step_size * search_direction + solution
      cppmin_axpy(n, step_size, search_direction, solution);

      // Update residual to current -gradient
      function.Gradient(solution, residual);
      cppmin_scal(n, -1.0, residual);

      // Compute Polak-Ribiere parameter

      previous_preconditioned_residual_norm2 = preconditioned_residual_norm2;
      mid_preconditioned_residual_norm2 = cppmin_dot(n, residual, M_inv_r);

      // Update preconditioner
      if (options.preconditioner != nullptr) {
        options.preconditioner->Update(n, solution);
        options.preconditioner->InverseDot(n, residual, M_inv_r);
      } else {
        std::memcpy(M_inv_r, residual, n * sizeof(double));
      }

      preconditioned_residual_norm2 = cppmin_dot(n, residual, M_inv_r);
      polak_ribiere_beta =
          (preconditioned_residual_norm2 - mid_preconditioned_residual_norm2) /
          previous_preconditioned_residual_norm2;

      // Update search direction
      cppmin_scal(n, polak_ribiere_beta, search_direction);
      cppmin_axpy(n, 1.0, M_inv_r, search_direction);

      // Since nonlinear conjugate gradients with Polak-Ribiere parameter
      // doesn't not guarentee to provide a descent search direction, we
      // need to restart it whenever the new computed search_direction is
      // not a descent direction.
      if (cppmin_dot(n, search_direction, M_inv_r) <= 0) {
        std::memcpy(search_direction, M_inv_r, n * sizeof(double));
      }

      ++iter;

      if (summary != nullptr) {
        ++summary->total_num_iterations;
      }
    }

    if (summary != nullptr) {
      summary->function_value_at_estimated_solution = function(solution);
    }

    delete[] residual;
    delete[] search_direction;
    delete[] M_inv_r;
    delete line_search;
  }
};

// Preconditioned nonlinear conjugate gradients with Fletcher-Reeves
// parameter
// See https://www.cs.cmu.edu/~quake-papers/painless-conjugate-gradient.pdf
// -------------------------------------------------------------------------

template<>
class LineSearchAlgorithm<FLETCHER_REEVES_CONJUGATE_GRADIENT>{
 public:
  const LineSearchMinimizer::Options& options;

  explicit LineSearchAlgorithm(const LineSearchMinimizer::Options& options)
      : options(options) {}

  template<typename Function>
  void Minimize(const Function& function, double* solution,
                LineSearchMinimizer::Summary* summary = nullptr) const {
    const size_t n = function.n_variables();

    if (summary != nullptr) {
      summary->function_value_at_starting_point = function(solution);
      summary->total_num_iterations = 0;
    }

    // Init residual to -gradient
    double* residual = new double[n];
    function.Gradient(solution, residual);
    cppmin_scal(n, -1.0, residual);

    // Cache the term M`r
    double* M_inv_r = new double[n];
    if (options.preconditioner != nullptr) {
      options.preconditioner->Update(n, solution);
      options.preconditioner->InverseDot(n, residual, M_inv_r);
    } else {
      // M is identity, so that M`r = r
      std::memcpy(M_inv_r, residual, n * sizeof(double));
    }

    // Init search direction
    double* search_direction = new double[n];
    std::memcpy(search_direction, M_inv_r, n * sizeof(double));

    // Preconditioned residual squares norm
    // We have
    //  r_hat = E`.r => |r_hat|^2 = r_hat'.r_hat
    //                            = r'.(EE')`.r
    //                            = r'.M`.r = r`.M_inv_r
    double preconditioned_residual_norm2;
    double previous_preconditioned_residual_norm2;
    preconditioned_residual_norm2 = cppmin_dot(n, residual, M_inv_r);

    // Init line search
    LineSearch<Function>* line_search =
        LineSearch<Function>::Create(options);
    double step_size;

    // This algorithm terminates if
    //  preconditioned_residual_norm2 <= epsilon
    // or if
    //  iter >= options().max_num_iterations
    const double epsilon = options.tolerance * options.tolerance *
        preconditioned_residual_norm2;
    size_t iter = 0;

    // Fletcher-Reeves parameter
    double fletcher_reeves_beta;

    while (iter < options.max_num_iterations &&
           preconditioned_residual_norm2 > epsilon) {
      // Compute step size
      step_size = line_search->Search(function, solution, search_direction);

      // Update solution & residual
      //  solution <- step_size * search_direction + solution
      //  residual <- -gradient(solution)
      cppmin_axpy(n, step_size, search_direction, solution);
      function.Gradient(solution, residual);
      cppmin_scal(n, -1.0, residual);

      previous_preconditioned_residual_norm2 = preconditioned_residual_norm2;

      // Update the term M`r
      if (options.preconditioner != nullptr) {
        options.preconditioner->Update(n, solution);
        options.preconditioner->InverseDot(n, residual, M_inv_r);
      } else {
        std::memcpy(M_inv_r, residual, n * sizeof(double));
      }

      // Compute Fletcher-Reeves parameter
      preconditioned_residual_norm2 = cppmin_dot(n, residual, M_inv_r);
      fletcher_reeves_beta = preconditioned_residual_norm2 /
          previous_preconditioned_residual_norm2;

      // Update search direction
      //  search_direction <- M_inv_r + beta * search_direction
      cppmin_scal(n, fletcher_reeves_beta, search_direction);
      cppmin_axpy(n, 1.0, M_inv_r, search_direction);

      // We need to restart CG if the new computed search_direction is not
      // a descent direction
      if (cppmin_dot(n, search_direction, M_inv_r) <= 0) {
        std::memcpy(search_direction, M_inv_r, n * sizeof(double));
      }

      ++iter;

      if (summary != nullptr) {
        ++summary->total_num_iterations;
      }
    }

    if (summary != nullptr) {
      summary->function_value_at_estimated_solution = function(solution);
    }

    delete[] residual;
    delete[] M_inv_r;
    delete[] search_direction;
    delete line_search;
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
