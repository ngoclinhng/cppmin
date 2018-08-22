// Copyright 2018
//
// Author: mail2ngoclinh@gmail.com
//
// Armijo and Wolfe line search algorithms

#ifndef CPPMIN_INTERNAL_LINE_SEARCH_H_
#define CPPMIN_INTERNAL_LINE_SEARCH_H_

namespace cppmin {
namespace internal {

// Base line search
template<typename Function> class LineSearch {
 public:
  struct Options {
    // Armijo and Wolfe line search parameters.

    // Initial step_size.
    double initial_step_size;

    // We want to find a step_size which results in sufficient decrease of
    // the objective function f along the search direction p_k. More
    // precisely, we are looking for a step size s.t
    //
    // phi_function(step_size) <= phi_function(0) +
    //                   sufficient_decrease * step_size * phi_function'(0)
    double sufficient_decrease;

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
    double max_step_contraction;
    double min_step_contraction;

    // If during the line search, the step_size falls below this value,
    // it is set to this value and the line search terminates.
    double min_step_size;

    // Maximum number of trial step size iterations during each line search,
    // If a step size satisfying the search coditions connot be found
    // within this number of trials, the line search will terminate.
    int max_iter;

    // Wolfe-specific line search parameters.

    // The strong Wolfe conditions consist of the Armijo sufficient decrease
    // condition, and an additional requirement that the step_size be chosen
    // s.t:
    //
    //  |phi_function'(step_size)| <= sufficient_curvature_decrease *
    //                              |phi_function'(0)|
    double sufficient_curvature_decrease;

    // The Wolfe line search algorithm is similar to that of the Armijo
    // line search algorithm until it found a step size sa satisfying the
    // sufficient decrease condition. At this point the Armijo line search
    // terminates while the Wolfe line search continues the search
    // in the interval [sa, max_step_size] (the zoom stage) until it found a
    // point which satifies the Wolfe condition.
    //
    // Note that, according to [1], the interval [sa, max_step_size]
    // contains a step size satisfying the Wolfe condition.
    //
    // [1] Nocedal J., Wright S., Numerical Optimization, 2nd Ed., Springer, 1999.  // NOLINT
    double max_step_size;

    // At each iteration in the zoom stage of the Wolfe line search, we
    // enlarge the current step_size by multiplying it with
    // max_step_expansion, so that we have
    //
    //  next_step_size = step_size * max_step_expansion
    //
    // If this next_step_size violates the sufficient decrease condition
    // we go to the refine stage (see below), if it meets the Wolfe
    // condition we return it, otherwise keep expanding step size.
    double max_step_expansion;

    // We only reach the refine stage if the step expansion causes the
    // next_step_size to violates the sufficient decrease condition,
    // once we enter this stage we have this interval:
    //
    //  [lo, hi]
    //
    // in which lo satisfies the sufficient decrease condition wile
    // the hi doesn't.
    //
    // At each iteration we'll use quadratic interpolation to generate
    // our next trial step_size (within this interval). If this step_size
    // statifies the Wolfe condition we're done, othersie we replace lo
    // by step_size continues until delta  = hi - lo <= epsilon.
    double epsilon;

    // Default options
    Options()
        : initial_step_size(1.0),
          sufficient_decrease(1e-4),
          max_step_contraction(1e-3),
          min_step_contraction(0.9),
          min_step_size(1e-9),
          max_iter(20),
          sufficient_curvature_decrease(0.9),
          max_step_size(4.0),
          max_step_expansion(2.0),
          epsilon(1e-3) {}
  };

  // Line search summary
  struct Summary {
    bool search_failed;
    Summary() : search_failed(false) {}
  };

  LineSearch() : options_(LineSearch<Function>::Options()) {}

  explicit LineSearch(const LineSearch<Function>::Options& options)
      : options_(options) {}

  virtual ~LineSearch() {}

  static LineSearch<Function>*
  Create(const LineSearchMinimizer::Options& options);

  // Perform the line search.
  // Given the first order (differentiable) function func, a position, and
  // a direction , returns a step_size.
  //
  // The existance of direction_scale here is to allow client to use
  // direction_scale * direction as as search direction.
  //
  // Note that, it is the caller's resposibility to make sure that the size
  // of position as well as direction is the same as func->n_variables().
  virtual
  double Search(const Function& function,
                const double* position,
                const double* direction,
                const double direction_scale = 1.0,
                LineSearch<Function>::Summary* summary = nullptr) const = 0;

 protected:
  const LineSearch<Function>::Options& options() const { return options_; }

 private:
  LineSearch<Function>::Options options_;
};

// Armijo line search ------------------------------------------------------
template<typename Function>
class ArmijoLineSearch : public LineSearch<Function> {
 public:
  ArmijoLineSearch() : LineSearch<Function>() {}
  explicit ArmijoLineSearch(const typename LineSearch<Function>::Options&
                            options)
      : LineSearch<Function>(options) {}

  // Perform the line search.
  // Given the first order (differentiable) function func, a position, and
  // a direction , returns a step_size.
  //
  // The existance of direction_scale here is to allow client to use
  // direction_scale * direction as as search direction.
  //
  // Note that, it is the caller's resposibility to make sure that the size
  // of position as well as direction is the same as func->n_variables().
  virtual double
  Search(const Function& function,
         const double* position,
         const double* direction,
         const double direction_scale = 1.0,
         typename LineSearch<Function>::Summary* summary = nullptr) const;
};

// Strong Wolfe line search ------------------------------------------------
template<typename Function> class PhiFunction;

template<typename Function>
class WolfeLineSearch : public LineSearch<Function> {
 public:
  WolfeLineSearch() : LineSearch<Function>() {}
  explicit WolfeLineSearch(const typename LineSearch<Function>::Options&
                           options)
      : LineSearch<Function>(options) {}

  // Perform the line search.
  // Given the first order (differentiable) function func, a position, and
  // a direction , returns a step_size.
  //
  // The existance of direction_scale here is to allow client to use
  // direction_scale * direction as as search direction.
  //
  // Note that, it is the caller's resposibility to make sure that the size
  // of position as well as direction is the same as func->n_variables().
  virtual double
  Search(const Function& function,
         const double* position,
         const double* direction,
         const double direction_scale = 1.0,
         typename LineSearch<Function>::Summary* summary = nullptr) const;

 private:
  double
  Refine(PhiFunction<Function>* phi_function,
         double lo, double phi_lo,
         double hi, double phi_hi,
         typename LineSearch<Function>::Summary* summary = nullptr) const;
};

}  // namespace internal
}  // namespace cppmin

#endif  // CPPMIN_INTERNAL_LINE_SEARCH_H_
