/**
 * @file utest_ad_tensor.cc
 * @brief Unit tests for IvyADTensor — the SoA reverse-mode autodiff tensor.
 *
 * Verifies forward values and reverse-mode gradients against closed-form results for:
 *  - unary ops (Negate, Exp, Log, Sin, Cos, Sqrt),
 *  - tensor-scalar ops (AddScalar, MultiplyScalar),
 *  - binary ops (Add, Subtract, Multiply, Divide),
 *  - the Sum reduction,
 *  - a multi-op chain (chain rule), and
 *  - multi-occurrence accumulation (f = x*x -> df/dx = 2x), zero_grad re-use.
 */

#include "common_test_defs.h"

#include "autodiff/tensor_soa/IvyADTensorOps.h"

#include <cmath>
#include "std_ivy/IvyCassert.h"


using namespace IvyAD;
using IvyMath::IvyTensorShape;
using IvyMath::IvyTensorDim_t;


static void check(bool cond, char const* label){
  if (cond){ __PRINT_INFO__("  [PASS] %s\n", label); }
  else { __PRINT_INFO__("  [FAIL] %s\n", label); assert(false); }
}

static bool close(double a, double b, double tol = 1e-9){ return std::abs(a - b) < tol; }

/** @brief Build a length-n leaf tensor from a small host array. */
static IvyADTensorPtr<double> leaf(std::initializer_list<double> vals){
  IvyTensorShape shape({ static_cast<IvyTensorDim_t>(vals.size()) });
  double tmp[16]; IvyTensorDim_t i = 0;
  for (double v : vals){ tmp[i++] = v; }
  return ADTensor<double>(shape, tmp, /*requires_grad*/true);
}


static void test_unary(){
  __PRINT_INFO__("--- unary ops ---\n");
  double const xv[3] = { 0.5, 1.0, 2.0 };

  // Exp: y=exp(x), dy/dx=exp(x)
  {
    auto x = leaf({ 0.5, 1.0, 2.0 });
    auto y = Sum(Exp(x));
    backward(y);
    bool vok = true, gok = true;
    double s = 0;
    for (int i = 0; i < 3; ++i){ s += std::exp(xv[i]); gok &= close(x->grad_at(i), std::exp(xv[i])); }
    vok = close(y->value_at(0), s);
    check(vok, "Sum(Exp(x)) value"); check(gok, "d Sum(Exp(x))/dx == exp(x)");
  }
  // Log
  {
    auto x = leaf({ 0.5, 1.0, 2.0 });
    auto y = Sum(Log(x));
    backward(y);
    bool gok = true; for (int i = 0; i < 3; ++i) gok &= close(x->grad_at(i), 1.0 / xv[i]);
    check(gok, "d Sum(Log(x))/dx == 1/x");
  }
  // Sin / Cos
  {
    auto x = leaf({ 0.5, 1.0, 2.0 });
    auto y = Sum(Sin(x));
    backward(y);
    bool gok = true; for (int i = 0; i < 3; ++i) gok &= close(x->grad_at(i), std::cos(xv[i]));
    check(gok, "d Sum(Sin(x))/dx == cos(x)");

    auto x2 = leaf({ 0.5, 1.0, 2.0 });
    auto y2 = Sum(Cos(x2));
    backward(y2);
    bool gok2 = true; for (int i = 0; i < 3; ++i) gok2 &= close(x2->grad_at(i), -std::sin(xv[i]));
    check(gok2, "d Sum(Cos(x))/dx == -sin(x)");
  }
  // Sqrt
  {
    auto x = leaf({ 0.5, 1.0, 2.0 });
    auto y = Sum(Sqrt(x));
    backward(y);
    bool gok = true; for (int i = 0; i < 3; ++i) gok &= close(x->grad_at(i), 0.5 / std::sqrt(xv[i]));
    check(gok, "d Sum(Sqrt(x))/dx == 1/(2 sqrt(x))");
  }
  // Negate
  {
    auto x = leaf({ 0.5, 1.0, 2.0 });
    auto y = Sum(Negate(x));
    backward(y);
    bool gok = true; for (int i = 0; i < 3; ++i) gok &= close(x->grad_at(i), -1.0);
    check(gok, "d Sum(-x)/dx == -1");
  }
}


static void test_scalar(){
  __PRINT_INFO__("--- tensor-scalar ops ---\n");
  auto x = leaf({ 1.0, 2.0, 3.0 });
  auto y = Sum(AddScalar(MultiplyScalar(x, 3.0), 1.0));  // sum(3x + 1)
  backward(y);
  check(close(y->value_at(0), (3.0 * 6.0) + 3.0), "Sum(3x+1) value");
  bool gok = true; for (int i = 0; i < 3; ++i) gok &= close(x->grad_at(i), 3.0);
  check(gok, "d Sum(3x+1)/dx == 3");
}


static void test_binary(){
  __PRINT_INFO__("--- binary ops ---\n");
  double const a[3] = { 1.0, 2.0, 3.0 };
  double const b[3] = { 4.0, 5.0, 6.0 };

  // Multiply: f = sum(x*y); df/dx = y, df/dy = x
  {
    auto x = leaf({ 1.0, 2.0, 3.0 });
    auto y = leaf({ 4.0, 5.0, 6.0 });
    auto f = Sum(Multiply(x, y));
    backward(f);
    bool gx = true, gy = true;
    for (int i = 0; i < 3; ++i){ gx &= close(x->grad_at(i), b[i]); gy &= close(y->grad_at(i), a[i]); }
    check(gx, "d Sum(x*y)/dx == y"); check(gy, "d Sum(x*y)/dy == x");
  }
  // Divide: f = sum(x/y); df/dx = 1/y, df/dy = -x/y^2
  {
    auto x = leaf({ 1.0, 2.0, 3.0 });
    auto y = leaf({ 4.0, 5.0, 6.0 });
    auto f = Sum(Divide(x, y));
    backward(f);
    bool gx = true, gy = true;
    for (int i = 0; i < 3; ++i){ gx &= close(x->grad_at(i), 1.0 / b[i]); gy &= close(y->grad_at(i), -a[i] / (b[i] * b[i])); }
    check(gx, "d Sum(x/y)/dx == 1/y"); check(gy, "d Sum(x/y)/dy == -x/y^2");
  }
  // Add / Subtract
  {
    auto x = leaf({ 1.0, 2.0, 3.0 });
    auto y = leaf({ 4.0, 5.0, 6.0 });
    auto f = Sum(Subtract(Add(x, y), y));   // (x+y)-y == x ; df/dx==1, df/dy==0
    backward(f);
    bool gx = true, gy = true;
    for (int i = 0; i < 3; ++i){ gx &= close(x->grad_at(i), 1.0); gy &= close(y->grad_at(i), 0.0); }
    check(gx, "d Sum((x+y)-y)/dx == 1"); check(gy, "d Sum((x+y)-y)/dy == 0");
  }
}


static void test_chain_and_multiuse(){
  __PRINT_INFO__("--- chain rule and multi-occurrence ---\n");
  double const xv[4] = { 0.1, 0.2, 0.3, 0.4 };

  // f = sum( exp(x) * x ) ; df/dx_i = exp(x_i)*(x_i + 1)
  {
    auto x = leaf({ 0.1, 0.2, 0.3, 0.4 });
    auto f = Sum(Multiply(Exp(x), x));
    backward(f);
    bool gok = true; for (int i = 0; i < 4; ++i) gok &= close(x->grad_at(i), std::exp(xv[i]) * (xv[i] + 1.0));
    check(gok, "d Sum(exp(x)*x)/dx == exp(x)(x+1)");
  }
  // Multi-occurrence: f = sum(x*x) ; df/dx = 2x  (adjoint accumulates over both branches)
  {
    auto x = leaf({ 0.1, 0.2, 0.3, 0.4 });
    auto f = Sum(Multiply(x, x));
    backward(f);
    bool gok = true; for (int i = 0; i < 4; ++i) gok &= close(x->grad_at(i), 2.0 * xv[i]);
    check(gok, "d Sum(x*x)/dx == 2x (multi-occurrence accumulation)");

    // zero_grad and re-run reproduces the same gradient (no stale accumulation).
    zero_grad(f);
    backward(f);
    bool gok2 = true; for (int i = 0; i < 4; ++i) gok2 &= close(x->grad_at(i), 2.0 * xv[i]);
    check(gok2, "zero_grad + backward reproduces gradient");
  }
}


void utest(){
  __PRINT_INFO__("=== utest_ad_tensor ===\n");
  test_unary();
  test_scalar();
  test_binary();
  test_chain_and_multiuse();
  __PRINT_INFO__("=== ALL utest_ad_tensor tests PASSED ===\n");
}


int main(){
  utest();
  return 0;
}
