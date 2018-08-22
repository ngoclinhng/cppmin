// Copyright 2018
//
// Author: mail2ngoclinh@gmail.com
//
// Given function f (in high dimension), a position and a search_direction
// PhiFunction is the one-variable function defined as follows:
//
//  PhiFunction(s) = f(position) + s * search_direction
//
// This is the objective function of Armijo & Wolfe line searchs

#ifndef CPPMIN_INTERNAL_PHI_FUNCTION_H_
#define CPPMIN_INTERNAL_PHI_FUNCTION_H_

namespace cppmin {
namespace internal {

template<typename Function>
class PhiFunction {
 public:
  // The value of Phi at step_size = 0.0.
  const double phi0;

  // The direvative of phi at step_size = 0.0.
  const double gradient0;

  // Construct Phi function from a Function func, a position,
  // and a direction.
  //
  // Note that, it is caller's responsibility to make sure that the size of
  // position as well as direction is the same as func->n_variables()
  explicit PhiFunction(const Function& func,
                       const double* position,
                       const double* direction,
                       const double direction_scale = 1.0);
  ~PhiFunction();

  // Evaluate the value of this function at a given step_size.
  //
  // Note that, this methods update the current_step_size_ and
  // current_position_.
  double operator()(const double step_size);

  // Returns the derivative of Phi at step_size.
  // This method updates current_step_size_ and current_position_
  double Derivative(const double step_size);

 private:
  const Function& func_;
  const double* direction_;
  double direction_scale_;

  double current_step_size_;

  double* current_position_;
  double* current_func_gradient_;

  // We explicitly delete ctor and assignment operator
  PhiFunction(const PhiFunction& src);
  PhiFunction& operator=(const PhiFunction& rhs);
};


// Implemenatation ---------------------------------------------------------

template<typename Function>
PhiFunction<Function>::PhiFunction(const Function& func,
                                   const double* position,
                                   const double* direction,
                                   const double direction_scale)
    : func_(func),
      direction_scale_(direction_scale),
      direction_(direction),
      current_step_size_(0.0),
      phi0(func(position)),
      gradient0(0.0) {
  // Current position. We need to keep tract of this value as well as
  // current step size in order to quickly compute the value and
  // the derivative of phi at a given step_size
  current_position_ = new double[func.n_variables()];
  std::memcpy(current_position_, position,
              func.n_variables() * sizeof(double));

  // Current gradient of func_.
  current_func_gradient_ = new double[func.n_variables()];
  func.Gradient(position, current_func_gradient_);

  // The derivative of Phi at step_size = 0.0
  const_cast<double&>(gradient0) = direction_scale *
      cppmin_dot(func.n_variables(), current_func_gradient_, direction);
}

template<typename Function>
PhiFunction<Function>::~PhiFunction() {
  if (current_position_ != nullptr) {
    delete[] current_position_;
  }

  if (current_func_gradient_ != nullptr) {
    delete[] current_func_gradient_;
  }

  current_position_ = nullptr;
  current_func_gradient_ = nullptr;
}

// Evaluate the value of Phi at step_size
// This method updates current_step_size_ and current_positon_
template<typename Function>
double
PhiFunction<Function>::operator()(const double step_size) {
  // Update the current_position_
  cppmin_axpy(func_.n_variables(),
              direction_scale_ * (step_size - current_step_size_),
              direction_,
              current_position_);
  current_step_size_ = step_size;
  return func_(current_position_);
}

// Returns the derivative of Phi at step_size.
// This methods update current_step_size_ and current_position_
template<typename Function>
double
PhiFunction<Function>::Derivative(const double step_size) {
  // Update the current_position_
  cppmin_axpy(func_.n_variables(),
              direction_scale_ * (step_size - current_step_size_),
              direction_,
              current_position_);

  // And current_step_size_
  current_step_size_ = step_size;

  // Evaluate the gradient of func_ at the current_position_
  func_.Gradient(current_position_, current_func_gradient_);

  // Then the derivative of Phi at step_size is simply the dot product
  // of current_func_gradient_ and direction_
  return direction_scale_ * cppmin_dot(func_.n_variables(),
                                       current_func_gradient_,
                                       direction_);
}

}  // namespace internal
}  // namespace cppmin
#endif  // CPPMIN_INTERNAL_PHI_FUNCTION_H_
