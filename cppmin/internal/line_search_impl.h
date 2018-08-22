// Copyright 2018
//
// Author: mail2ngoclinh@gmail.com
//
// Implemetation of LineSearch

#ifndef CPPMIN_INTERNAL_LINE_SEARCH_IMPL_H_
#define CPPMIN_INTERNAL_LINE_SEARCH_IMPL_H_

namespace cppmin {
namespace internal {


// Line search factory
template<typename Function>
LineSearch<Function>*
LineSearch<Function>::Create(const LineSearchMinimizer::Options& options) {
  typename LineSearch<Function>::Options ls_options;
  ls_options.initial_step_size = options.initial_step_size;
  ls_options.sufficient_decrease = options.sufficient_decrease;
  ls_options.max_step_contraction = options.max_step_contraction;
  ls_options.min_step_contraction = options.min_step_contraction;
  ls_options.min_step_size = options.min_step_size;
  ls_options.max_iter = options.max_num_step_size_trials;
  ls_options.sufficient_curvature_decrease =
      options.sufficient_curvature_decrease;
  ls_options.max_step_size = options.max_step_size;
  ls_options.max_step_expansion = options.max_step_expansion;
  ls_options.epsilon = options.min_step_size_search_interval_length;

  switch (options.line_search_type) {
    case cppmin::ARMIJO:
      return new ArmijoLineSearch<Function>(ls_options);
    case cppmin::WOLFE:
      return new WolfeLineSearch<Function>(ls_options);
    default:
      return nullptr;
  }
}

// Armijo line Search
//
// Perform the line search.
// Given the first order (differentiable) function func, a position, and
// a direction , returns a step_size.
//
// The existance of direction_scale here is to allow client to use
// direction_scale * direction as as search direction.
//
// Note that, it is the caller's resposibility to make sure that the size
// of position as well as direction is the same as func->n_variables().
template<typename Function>
double ArmijoLineSearch<Function>::
Search(const Function& func,
       const double* position,
       const double* direction,
       const double direction_scale,
       typename LineSearch<Function>::Summary* summary) const {
  // TODO(Linh): CHECK users inputs

  // Construct Phi function
  PhiFunction<Function> phi_function(func, position, direction,
                                     direction_scale);

  double previous_step_size = 0.0;
  double current_step_size = this->options().initial_step_size;
  double interpolated_step_size;

  double previous_phi;
  double current_phi;

  double decrease;

  const int max_iter = this->options().max_iter;
  int iter = 0;

  while (iter < max_iter && current_step_size >
         this->options().min_step_size) {
    current_phi = phi_function(current_step_size);
    decrease = phi_function.phi0 + this->options().sufficient_decrease *
        current_step_size * phi_function.gradient0;

    if (current_phi <= decrease) {  // sufficient decrease condition met
      if (summary != nullptr) {
        summary->search_failed = false;
      }
      return current_step_size;  // success
    } else if (iter == 0) {
      // Use Quadratic interpolation to guess next trial
      interpolated_step_size =
          QuadraticInterpolate(phi_function.phi0,
                               phi_function.gradient0,
                               current_step_size,
                               current_phi);
    } else {
      // Use Cubic interpolation to guess next trial
      interpolated_step_size =
          CubicInterpolate(phi_function.phi0,
                           phi_function.gradient0,
                           current_phi,
                           current_step_size,
                           previous_phi,
                           previous_step_size);
    }

    // Store the previous values for interpolation.
    previous_step_size = current_step_size;
    previous_phi = current_phi;

    // On on hand, we want our next trial step size to be less than the
    // previous one (so that we have a monotonically decreasing sequence
    // of step sizes). On the other hand, we don't want our next trial step
    // size to be too far less than the previous one. So we need to
    // contract our interpolated_step_size so that it always lies between
    // max_step_contraction * previous_step_size and
    // min_step_contraction * previous_step_size.
    current_step_size =
        ContractStep(interpolated_step_size,
                     this->options().max_step_contraction *
                     previous_step_size,
                     this->options().min_step_contraction *
                     previous_step_size);
    iter++;
  }

  if (summary != nullptr) {
    summary->search_failed = true;
  }
  return this->options().min_step_size;
}

// Strong Wolfe line search ------------------------------------------------

template<typename Function>
double WolfeLineSearch<Function>::
Search(const Function& func,
       const double* position,
       const double* direction,
       const double direction_scale,
       typename LineSearch<Function>::Summary* summary) const {
  // TODO(Linh): CHECK users inputs

  // The Wolfe line search algorithm is similar to that of the Armijo
  // line search until it found a step size armijo_step_size satisfying
  // the sufficient decrease condition. At this point the Armijo terminates
  // while the Wolfe continues the search within the interval
  //  [armijo_step_size, max_step_size] until it finds a step size satisfying
  // the Wolfe condition.

  // First stage: find armijo_step_size

  ArmijoLineSearch<Function> armijo(this->options());
  double armijo_step_size;
  armijo_step_size = armijo.Search(func, position, direction,
                                   direction_scale, summary);

  // If Armijo stage failed, we return immediately
  if (summary != nullptr && summary->search_failed) {
    return armijo_step_size;
  }

  if (armijo_step_size >= this->options().max_step_size) {
    if (summary != nullptr) {
      summary->search_failed = true;
    }
    return this->options().max_step_size;
  }

  // Zoom stage

  // construct Phi function
  PhiFunction<Function> phi_function(func, position, direction,
                                     direction_scale);

  const double sufficient_curvature =
      this->options().sufficient_curvature_decrease *
      std::fabs(phi_function.gradient0);
  double phi_gradient = phi_function.Derivative(armijo_step_size);

  if (std::fabs(phi_gradient) <= sufficient_curvature) {
    if (summary != nullptr) {
      summary->search_failed = false;
    }
    return armijo_step_size;
  }

  double previous_step_size;
  double current_step_size = armijo_step_size;

  double previous_phi;
  double current_phi = phi_function(current_step_size);

  double sufficient_decrease;

  while (current_step_size < this->options().max_step_size) {
    // Save values
    previous_step_size = current_step_size;
    previous_phi = current_phi;

    // Enlarge step size
    current_step_size = this->options().max_step_expansion *
        current_step_size;
    current_phi = phi_function(current_step_size);
    sufficient_decrease = phi_function.phi0 +
        this->options().sufficient_decrease * current_step_size *
        phi_function.gradient0;

    if (current_phi > sufficient_decrease) {
      // The current_step_size violates the sufficient decrease condition.
      // So we know from [1] that the solution must lie in the interval
      // [previous_step_size, current_step_size].
      return Refine(&phi_function,
                    previous_step_size,
                    previous_phi,
                    current_step_size,
                    current_phi,
                    summary);
    }

    phi_gradient = phi_function.Derivative(current_step_size);

    if (std::fabs(phi_gradient) <= sufficient_curvature) {
      // Found Wolfe point.
      if (summary != nullptr) {
        summary->search_failed = false;
      }
      return current_step_size;
    }
  }

  // Search failed.
  // Returns max_step_size as a fallback.
  if (summary != nullptr) {
    summary->search_failed = true;
  }
  return this->options().max_step_size;
}

template<typename Function>
double
WolfeLineSearch<Function>::
Refine(PhiFunction<Function>* phi_function,
       double lo, double phi_lo,
       double hi, double phi_hi,
       typename LineSearch<Function>::Summary* summary) const {
  double phi_gradient_lo = phi_function->Derivative(lo);

  double current_step_size = lo;
  double current_phi = phi_lo;
  double phi_gradient = phi_gradient_lo;

  double delta = hi - lo;
  double delta_step_size;

  const double sufficient_curvature =
      this->options().sufficient_curvature_decrease *
      std::fabs(phi_function->gradient0);
  double sufficient_decrease;

  while (delta > this->options().epsilon) {
    // Compute the delta step size.
    delta_step_size = (delta * delta * phi_gradient_lo) /
        (2.0 * (phi_lo + delta * phi_gradient_lo - phi_hi));
    delta_step_size = ContractStep(delta_step_size,
                                   0.2 * delta,
                                   0.8 * delta);

    // Update
    current_step_size = lo + delta_step_size;
    current_phi = (*phi_function)(current_step_size);
    sufficient_decrease = phi_function->phi0 +
        this->options().sufficient_decrease * current_step_size *
        phi_function->gradient0;

    if (current_phi <= sufficient_decrease) {
      phi_gradient = phi_function->Derivative(current_step_size);

      if (std::fabs(phi_gradient) <= sufficient_curvature) {
        // Found Wolfe point
        if (summary != nullptr) {
          summary->search_failed = false;
        }
        return current_step_size;
      }

      // Replace lo endpoint
      lo = current_step_size;
      phi_lo = current_phi;
      phi_gradient_lo = phi_gradient;
      delta -= delta_step_size;
    } else {
      // current_step_size violates the sufficient decrease condition,
      // we replace hi endpoint by current_step_size.
      hi = current_step_size;
      phi_hi = current_phi;
      delta = delta_step_size;
    }
  }

  // Search failed
  // Return whaever the current value of current_step_size as a fallback.
  // TODO(Linh): Is it a good idea to return current_step_size here?
  if (summary != nullptr) {
    summary->search_failed = true;
  }
  return current_step_size;
}
}  // namespace internal
}  // namespace cppmin
#endif  // CPPMIN_INTERNAL_LINE_SEARCH_IMPL_H_
