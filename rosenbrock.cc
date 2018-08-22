// Copyright 2018
//
// Author: mail2ngoclinh@gmail.com
//
// An example of using LineSearchMinimizer to estimate the global minimizer
// of N-dimensional Rosenbrock function
// See https://en.wikipedia.org/wiki/Test_functions_for_optimization

#include <iostream>

#include "cppmin/line_search_minimizer.h"

// N-dimensional Rosenbrock function
//
//  f(x) = sum 100 * (x_{i+1} - x_i^2)^2 + (1 - x_i)^2
//  (sum taken from i = 1 upto N - 1)
//
// N-dimensional Rosenbrock has a unique global minimizer at (1, 1, .., 1)'
// and the minimum is 0.0
struct Rosenbrock {
  // Default is 3-D Rosenbrock
  Rosenbrock() : N_(2) {}
  explicit Rosenbrock(const int N) : N_(N) {}

  // Returns number of variables (i.e dimension)
  int n_variables() const { return N_; }

  // Evaluates the value of Rosenbrock at a given position
  //  position = [x1, x2, ..., xn]
  double operator()(const double* x) const {
    double ret = 0.0;
    double xi;
    double xi1;

    for (int i = 0; i < N_ - 1; ++i) {
      xi = x[i];
      xi1 = x[i+1];

      ret += 100.0 * (xi1 - xi * xi) * (xi1 - xi * xi) +
          (1.0 - xi) * (1.0 - xi);
    }

    return ret;
  }

  // Evaluates the gradient of Rosenbrock at a given position
  //  position = [x1, x2, ..., xn]
  //
  //  f(x) = sum [100(x_{i+1} - x_i*x_i)^2 + (1 - x_i)^2]
  //       = sum t_i for i = 1 upto n-1
  //
  // We have
  //
  //  dt_i
  //  ---- = -400 * xi * (xi1 - xi^2) - 2 * (1 - xi)
  //  dxi
  //
  //  dt_i
  //  ---- = 200 * (xi1 - xi^2)
  //  dxi1
  void Gradient(const double* position, double* gradient) const {
    double xi;
    double xi1;  // x_{i+1}
    double xi2;  // x_{i+2}

    double ti_xi = 0.0;  // gradient of the term t_i w.r.t x_i
    double ti_xi1 = 0.0;  // gradient of the term t_i w.r.t x_{i+1}

    double previous = 0.0;

    int i;
    for (i = 0; i < N_ - 1; ++i) {
      xi = position[i];
      xi1 = position[i+1];

      ti_xi = -400.0 * xi* (xi1 - xi * xi) - 2.0 * (1 - xi);
      ti_xi1 = 200.0 * (xi1 - xi * xi);

      gradient[i] = previous + ti_xi;
      previous = ti_xi1;
    }
    gradient[i] = previous;
  }

 private:
  int N_;  // dimension
};

int main(int argc, char** argv) {
  const int N = 2;  // 3-D Rosenbrock, i.e f(x1, x2, x3)
  cppmin::LineSearchMinimizer::Summary summary;
  cppmin::LineSearchMinimizer::Options options;
  options.line_search_direction_type = cppmin::STEEPEST_DESCENT;
  options.max_num_iterations = 1000;
  cppmin::LineSearchMinimizer minimizer(options);
  Rosenbrock rosen(N);
  double solution[N];

  // starting point is (0, 0, ..,0)
  for (int i = 0; i < N; ++i) {
    solution[i] = 0.0;
  }

  minimizer.Minimize(rosen, solution, &summary);
  std::cout << summary << std::endl;

  for (int i = 0; i < N; ++i) {
    std::cout << solution[i] << "\n";
  }
  return 0;
}
