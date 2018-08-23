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
// rosenbrock.cc

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

```
The only dependency that you need in order to compile and run the above code
is to link it against a BLAS libary ([OpenBLAS](https://www.openblas.net/), [MKL](https://software.intel.com/en-us/mkl), [Apple Accelerate](https://developer.apple.com/documentation/accelerate/blas?language=objc), etc..). For instance, if you are a Mac user, there is a built-int BLAS library inside the Accelerate framework and you could compile the above code as follows:

```shell
g++ -o rosenbrock rosenbrock.cc -DCPPMIN_USE_ACCELERATE -framework Accelerate
```

Or if you has MKL:

```shell
g++ -o rosenbrock rosenbrock.cc -DCPPMIN_USE_MKL -mkl
```

Or if you has OpenBLAS:

```shell
g++ -I/path/to/OpenBLAS/include -L/path/to/OpenBLAS/lib -o rosenbrock rosenbrock.cc -lopenblas
```

LineSearchMinimizer
===================

The idea of [line search minimization](https://en.wikipedia.org/wiki/Line_search) is rather simple:

Given the objective function `f`, a starting point `x`, `LineSearchMinimizer`
estimates the global minimizer of `f` by building a sequence `{x_k}` such that `f(x_{k+1}) < f(x_k)`, and `|f'(x_k)| -> 0` as `k -> infinity`. It does that using the following algorithm:

1. Setup: initialize a `search_direction` (usually the steepest descent direction at the starting point `-f'(x)`).

2. Repeat until some stopping criteria are met (e.g `|f'(x_k)| < threshold`):
  * Use some line search algorithm ([Armijo](https://en.wikipedia.org/wiki/Backtracking_line_search), [Wolfe](https://en.wikipedia.org/wiki/Wolfe_conditions), etc...) to compute a suitable `step_size` along the `search_direction`
  so that:
  
    `f(x_k + step_size * search_direction) < f(x_k)`
  
  * Update solution: `x <- x + step_size * search_direction`

  * Update `search_direction`.

### Armijo and Wolfe Line Search

Let's say we're now at the current position `x_k` with a `search_direction`
in which to take the next step. The question is how big a step should we
take?

Denote `phi(s) = f(x_k + s * search_direction), s >= 0`.

Ideally we would want to find a `step_size > 0` such that it minimizes the
`phi` function above. In practice, this is hard, computational expensive, and
often not worthwhile. Instead we use some cheap, iterative scheme to find
a `step_size` that is 'good enough'. `LineSearchMinimizer` lets you freely determine how good a step size is by customizing the `line_search_type` option
and a bunch of other line search related parameters (see header file).

  ```cpp
  cppmin::LineSearchMinimizer::Options options;

  // Use Armijo line search:
  options.line_search_type = cppmin::ARMIJO;

  // Use strong Wolfe line search:
  options.line_search_type = cppmin::WOLFE;
  ```
`ARMIJO` produces a `step_size` which satisfies the so-called sufficient
decrease condition:

   `f(x_k + step_size * search_direction) <=
   
    f(x_k) + step_size * sufficient_decrease * f'(x_k) * search_direction`
    
in which `sufficient_decrease` is some constant in the interval `(0, 1)`.
By default is is set to `0.0001`. You can change it by adjusting the
   `options.sufficient_decrease` option:
   ```cpp
   cppmin::LineSearchMinimizer::Options options;
   options.line_search_type = cppmin::ARMIJO;
   options.sufficient_decrease = 0.001;
   ```
whereas, (strong) `WOLFE` produces a `step_size` which satisfies the above
sufficient decrease condition and an additional condition named curvature
condition:

   `|f'(x_k + step_size * search_direction) * search_direction|
    <= sufficient_curvature_decrease * |f'(x_k) * search_direction|`
    
in which `sufficient_curvature_decrease` is a some constant such that
`0 < sufficient_decrease < sufficient_curvature_decrease < 1`. As with
`sufficient_decrease`, you can change this paramater by adjusting the
`options.sufficient_curvature_decrease` option. By default it is set to `0.9`
  ```cpp
  cppmin::LineSearchMinimizer::Options options;
  options.line_search_type = cppmin::WOLFE;
  options.sufficient_decrease = 0.003;
  options.sufficient_curvature_decrease = 0.8
  ```
  


