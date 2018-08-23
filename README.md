cppmin
======

__cppmin__ is a header-only C++ library for solving the unconstrained
minimization problem `min f(x)`, in which `f: R^n -> R` is (at least)
continously differentiable with [Lipschitz](https://en.wikipedia.org/wiki/Lipschitz_continuity) gradient. __cppmin__ solves the aforementioned problem using
two different approaches [Line search](https://en.wikipedia.org/wiki/Line_search) and trust region.

Before diving into the nitty-gritty of each algorithms and the implementation
details, let first take a look at an example of using __cppmin__ to estimate
the global minimizer of a simple [Rosenbrock function](https://en.wikipedia.org/wiki/Rosenbrock_function)

```cpp
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

// Use default LineSearchMinimizer
cppmin::LineSearchMinimizer::Summary summary;
cppmin::LineSearchMinimizer minimizer;

Rosenbrock rosen;
double solution[2] = {0.0, 0.0}  // starting point is (0, 0)

minimizer.Minimize(rosen, solution, &summary);
std::cout << summary << std::endl;

std::cout << "Solution: " << std::endl;
std::cout << "x = " << solution[0] << std::endl;
std::cout << "y = " << solution[1] << std::endl;

return 0;

```