#include "autodiff/arithmetic/IvyMathBaseArithmetic.h"
#include "autodiff/basic_nodes/IvyTensor.h"

#include <cmath>

/**
 * @file gradient_ergonomics_demo.cc
 * @brief Demonstrate user-facing gradient ergonomics for variable/tensor/function inputs.
 */
/**
 * @brief Run gradient ergonomics smoke checks for scalar, tensor, and chained functions.
 * @return 0 when all checks pass; non-zero otherwise.
 */
int main(){
  using namespace std_ivy;
  using namespace IvyMath;

  constexpr double tol = 1e-9;
  IvyGPUStream* stream = IvyStreamUtils::make_global_gpu_stream();
  bool ok = true;

  auto x = Scalar<double>(IvyMemoryType::Host, stream, 1.0);
  auto f = Sin(Exp(x));
  auto grad_x = f->gradient(x);
  double const expected_dx = std::cos(std::exp(1.0))*std::exp(1.0);
  double const got_dx = grad_x->value().value();
  __PRINT_INFO__("grad_x: got=%.12f expected=%.12f\n", got_dx, expected_dx);
  ok = ok && std::abs(got_dx-expected_dx) < tol;

  IvyTensorShape shape({ 2, 1 });
  auto t = Tensor<IvyScalarPtr_t<double>>(IvyMemoryType::Host, stream, shape, x);
  auto tf = Exp(t);
  auto grad_t = tf->gradient(x);
  for (IvyTensorDim_t i = 0; i < grad_t->value().num_elements(); ++i){
    double const got_t = grad_t->value()[i]->value();
    __PRINT_INFO__("grad_t[%llu]: got=%.12f expected=%.12f\n", i, got_t, std::exp(1.0));
    ok = ok && std::abs(got_t-std::exp(1.0)) < 1e-6;
  }

  auto u = Exp(x);
  auto h = Sin(u);
  auto grad_chain_x = h->gradient(x);
  __PRINT_INFO__("grad_chain_x: got=%.12f expected=%.12f\n", grad_chain_x->value().value(), expected_dx);
  ok = ok && std::abs(grad_chain_x->value().value()-expected_dx) < tol;

  IvyStreamUtils::destroy_stream(stream);
  return ok ? 0 : 1;
}
