// Copyright 2018
//
// Author: mail2ngoclinh@gmail.com
//
// An example of using LineSearchMinimizer to estimate the global minimizer
// of N-dimensional Rosenbrock function
// See https://en.wikipedia.org/wiki/Test_functions_for_optimization

#include <iostream>

#include "cppmin/line_search_minimizer.h"

// Create a Rosenbrock function
//
//  f(x, y) = 100 * (y - x^2)^2 + (1 - x)^2
//
// This function has a unique global minimizer at (1, 1) and f(1, 1) = 0
struct Rosenbrock {
  // Returns number of variables (which is 2 in our case)
  int n_variables() const { return 2; }

  // Evaluates the value of this function at a given position = (x, y)
  double operator()(const double* position) const {
    const double x = position[0];
    const double y = position[1];
    double t1, t2;
    t1 = y - x * x;
    t2 = 1 - x;
    return 100.0 * t1 * t1 + t2 * t2;
  }

  // Evaluates the gradient of this function at a given position = (x, y)
  void Gradient(const double* position, double* gradient) const {
    const double x = position[0];
    const double y = position[1];
    double t1, t2;
    t1 = y - x * x;
    t2 = 1 - x;

    // partial derivative of f w.r.t x
    gradient[0] = 200.0 * (-2.0 * x) * t1 - 2.0 * t2;

    // partial derivative of f w.r.t y
    gradient[1] = 200.0 * t1;
  }
};

int main(int argc, char** argv) {
  // Use default LineSearchMinimizer
  cppmin::LineSearchMinimizer::Summary summary;
  cppmin::LineSearchMinimizer minimizer;

  Rosenbrock rosen;
  double solution[2] = {0.0, 0.0};  // starting point is (0, 0)

  minimizer.Minimize(rosen, solution, &summary);
  std::cout << summary << std::endl;

  std::cout << "Solution: " << std::endl;
  std::cout << "x = " << solution[0] << std::endl;
  std::cout << "y = " << solution[1] << std::endl;

  return 0;
}
