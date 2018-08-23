> __cppmin__ is a header-only C++ library for solving the unconstrained
> optimization problem `min f(x)`, in which `f : R^n -> R` is (at least)
> a continously differentiable function with Lipschitz gradient, i.e, there
> exists some constant `c > 0` sucht that:
>  `|f'(x) - f'(y)| <= c * |x - y|`