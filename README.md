cppmin
======

__cppmin__ is a header-only C++ library for solving the unconstrained
minimization problem `min f(x)`, in which `f: R^n -> R` is (at least)
continously differentiable with [Lipschitz](https://en.wikipedia.org/wiki/Lipschitz_continuity) gradient. __cppmin__ solves the aforementioned problem using
two different approaches:
  1. [Line search] (https://en.wikipedia.org/wiki/Line_search)
  2. Trust region