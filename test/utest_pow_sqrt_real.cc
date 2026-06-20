/**
 * @file utest_pow_sqrt_real.cc
 * @brief Regression test for real-domain scalar Pow / Sqrt graph nodes.
 *
 * Before the IvyPow domain fix, building a real-domain @c Pow(leaf, leaf) or
 * @c Sqrt(leaf) graph node failed to COMPILE: the IvyPow alias derived its node
 * domain from the reduced (bare-arithmetic) value type, so PowFcnal's gradient
 * recursively built real⊗arithmetic sub-nodes whose gradient-less evaluator was
 * force-instantiated through the vtable. Complex Pow was unaffected.
 *
 * This test exercises (real domain):
 *  - Pow(x,y)   value and d/dx, d/dy
 *  - Sqrt(x)    value and d/dx  (also guards the previously-latent missing 1/2
 *               coefficient in SqrtFcnal::gradient)
 * and confirms complex Pow still works (no regression).
 */

#include "common_test_defs.h"

#include "autodiff/arithmetic/IvyMathBaseArithmetic.h"
#include "autodiff/basic_nodes/IvyComplex.h"

#include <cmath>
#include "std_ivy/IvyCassert.h"


using namespace std_ivy;
using namespace IvyMath;


static void check(bool cond, char const* label){
  if (cond){
    __PRINT_INFO__("  [PASS] %s\n", label);
  } else {
    __PRINT_INFO__("  [FAIL] %s\n", label);
    assert(false);
  }
}

static bool close(double a, double b, double tol = 1e-9){ return std::abs(a - b) < tol; }

// Host-only readers of a node's value(): 2D op nodes (Pow, Multiply, ...) yield a bare
// arithmetic value; 1D op nodes (Sqrt, Exp, ...) yield an IvyScalar wrapper. Reading
// node->value() directly avoids instantiating the __HOST_DEVICE__ unpack helpers under nvcc.
static double val_of(double v){ return v; }
template<typename T> static double val_of(IvyScalar<T> const& v){ return v.value(); }
template<typename P> static double rd(P const& p){ return val_of(p->value()); }


static void test_pow(){
  __PRINT_INFO__("--- real Pow(x,y) ---\n");
  constexpr double x0 = 2.0, y0 = 3.0;
  auto x = Scalar<double>(IvyMemoryType::Host, nullptr, x0);
  auto y = Scalar<double>(IvyMemoryType::Host, nullptr, y0);

  auto p = Pow(x, y);                       // x^y
  check(close(rd(p), std::pow(x0, y0)), "Pow(2,3) == 8");

  // d/dx x^y = y x^(y-1)
  check(close(rd(p->gradient(x)), y0 * std::pow(x0, y0 - 1.0)), "dPow/dx == y*x^(y-1) == 12");
  // d/dy x^y = x^y ln(x)
  check(close(rd(p->gradient(y)), std::pow(x0, y0) * std::log(x0)), "dPow/dy == x^y ln(x)");
  // Independent leaf => 0
  auto z = Scalar<double>(IvyMemoryType::Host, nullptr, 5.0);
  check(close(rd(p->gradient(z)), 0.0), "dPow/dz == 0 (independent)");
}


static void test_sqrt(){
  __PRINT_INFO__("--- real Sqrt(x) ---\n");
  constexpr double x0 = 2.0;
  auto x = Scalar<double>(IvyMemoryType::Host, nullptr, x0);

  auto s = Sqrt(x);
  check(close(rd(s), std::sqrt(x0)), "Sqrt(2) == sqrt(2)");
  // d/dx sqrt(x) = 1/(2 sqrt(x))
  check(close(rd(s->gradient(x)), 0.5 / std::sqrt(x0)), "dSqrt/dx == 1/(2 sqrt(x))");
}


static void test_complex_pow_regression(){
  __PRINT_INFO__("--- complex Pow (no regression) ---\n");
  auto z = Complex<double>(IvyMemoryType::Host, nullptr, 2.0, 1.0);
  auto w = Complex<double>(IvyMemoryType::Host, nullptr, 3.0, 0.5);
  auto p = Pow(z, w);
  // value = exp(w log z); just sanity-check it is finite and buildable + gradient builds.
  auto g = p->gradient(z);
  check(std::isfinite(p->value().norm()) && g.get() != nullptr, "complex Pow value+gradient build");
}


void utest(){
  __PRINT_INFO__("=== utest_pow_sqrt_real ===\n");
  test_pow();
  test_sqrt();
  test_complex_pow_regression();
  __PRINT_INFO__("=== ALL utest_pow_sqrt_real tests PASSED ===\n");
}


int main(){
  utest();
  return 0;
}
