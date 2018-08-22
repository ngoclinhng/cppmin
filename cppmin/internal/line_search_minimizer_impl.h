// Copyright 2018
//
// Author: mail2ngoclinh@gmail.com
//
// Implementation LineSearchMinimizer

#ifndef CPPMIN_INTERNAL_LINE_SEARCH_MINIMIZER_IMPL_H_
#define CPPMIN_INTERNAL_LINE_SEARCH_MINIMIZER_IMPL_H_

namespace cppmin {

template<typename Function>
void
LineSearchMinimizer::Minimize(const Function& function,
                              double* solution,
                              LineSearchMinimizer::Summary* summary) const {
  // We delegate work to proper algorithm based on
  // options_.line_search_direction_type
  // TODO(Linh): This if-else thing is odd!
  const LineSearchDirectionType direction =
      options_.line_search_direction_type;
  if (direction == STEEPEST_DESCENT) {
    internal::LineSearchAlgorithm<STEEPEST_DESCENT> alg(options_);
    alg.Minimize(function, solution, summary);
  } else if (direction == FLETCHER_REEVES_CONJUGATE_GRADIENT) {
    internal::LineSearchAlgorithm<FLETCHER_REEVES_CONJUGATE_GRADIENT>
        fletcher_reeves(options_);
    fletcher_reeves.Minimize(function, solution, summary);
  } else if (direction == POLAK_RIBIERE_CONJUGATE_GRADIENT) {
    internal::LineSearchAlgorithm<POLAK_RIBIERE_CONJUGATE_GRADIENT>
        polak_ribiere(options_);
    polak_ribiere.Minimize(function, solution, summary);
  } else if (direction == LBFGS) {
    internal::LineSearchAlgorithm<LBFGS> lbfgs(options_);
    lbfgs.Minimize(function, solution, summary);
  }
}
}  // namespace cppmin

#endif  // CPPMIN_INTERNAL_LINE_SEARCH_MINIMIZER_IMPL_H_

