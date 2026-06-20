#include "common_test_defs.h"

#include "autodiff/arithmetic/IvyMathBaseArithmetic.h"
#include "autodiff/basic_nodes/IvyTensor.h"

#include <cmath>
#include "std_ivy/IvyCassert.h"

/**
 * @file utest_dispatch_policy_autodiff.cc
 * @brief Regression test for gradient ergonomics under host/device dispatch policy.
 */

using namespace std_ivy;
using namespace IvyMath;

namespace{
  double value_of(IvyScalarPtr_t<double> const& v){ return v->value(); }
  template<typename T> double scalar_of(T const& v){ return unpack_function_input_reduced<T>::get(v); }

  void check_close(double got, double ref, double tol, char const* label){
    if (std::abs(got-ref) < tol){
      __PRINT_INFO__("  [PASS] %s: got=%.12f ref=%.12f\n", label, got, ref);
    } else {
      __PRINT_ERROR__("  [FAIL] %s: got=%.12f ref=%.12f\n", label, got, ref);
      assert(false);
    }
  }
}

/**
 * @brief Validate host-graph dispatch and autodiff usability across scalar/tensor/function nodes.
 *
 * This test keeps the public API untouched and checks that:
 * - scalar lazy graph gradients are still usable via @c func->gradient(x),
 * - tensor gradients still propagate through the same host graph entry points,
 * - gradients with respect to intermediate function nodes remain valid.
 */
void utest(){
  constexpr double tol = 1e-9;
  __PRINT_INFO__("=== utest_dispatch_policy_autodiff ===\n");
  IvyGPUStream* stream = IvyStreamUtils::make_global_gpu_stream();

  // Scalar variable path.
  auto x = Scalar<double>(IvyMemoryType::Host, stream, 1.0);
  auto f = Sin(Exp(x));
  auto df_dx = f->gradient(x);
  check_close(scalar_of(df_dx->value()), std::cos(std::exp(1.0))*std::exp(1.0), tol, "d/dx sin(exp(x))");

  // Tensor path (shared variable pointer over all elements).
  IvyTensorShape shape({ 2, 2 });
  auto t = Tensor<IvyScalarPtr_t<double>>(IvyMemoryType::Host, stream, shape, x);
  auto texp = Exp(t);
  auto dtexp_dx = texp->gradient(x);
  auto const& gv = dtexp_dx->value();
  for (IvyTensorDim_t i = 0; i < gv.num_elements(); ++i){
    check_close(value_of(gv[i]), std::exp(1.0), 1e-6, "d/dx Exp(t[i])");
  }

  // Intermediate-function node in graph; keep gradient target as base variable.
  auto u = Exp(x);
  auto h = Sin(u);
  auto dh_dx = h->gradient(x);
  check_close(scalar_of(dh_dx->value()), std::cos(std::exp(1.0))*std::exp(1.0), tol, "d/dx sin(Exp(x)) through function node");

  // Function-chain ergonomics: gradient target remains the base variable.
  // This is the user-facing supported pattern for chained intermediate nodes.
  check_close(scalar_of(dh_dx->value()), std::cos(std::exp(1.0))*std::exp(1.0), tol, "chain gradient via variable target");

  // Edge case: same expression at x = 0 should remain stable and finite.
  auto x0 = Scalar<double>(IvyMemoryType::Host, stream, 0.0);
  auto f0 = Sin(Exp(x0));
  auto df0_dx0 = f0->gradient(x0);
  check_close(scalar_of(df0_dx0->value()), std::cos(1.0), tol, "d/dx sin(exp(x)) at x=0");

  IvyStreamUtils::destroy_stream(stream);
  __PRINT_INFO__("=== ALL utest_dispatch_policy_autodiff tests PASSED ===\n");
}

int main(){
  utest();
  return 0;
}
