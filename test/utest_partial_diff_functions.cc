/**
 * @file utest_partial_diff_functions.cc
 * @brief Unit tests for partial differentiation with respect to function nodes.
 *
 * Verifies that an intermediate function node @c u can be used as a
 * differentiation target exactly like a leaf variable, i.e. the chain rule
 * applies the identity @f$\partial u/\partial u = 1@f$.
 *
 * Exercises (real domain):
 *  - du/du == 1, du/dx == exp(x)            (1D function as its own target)
 *  - dh/du == cos(u), dh/dx == cos(u)exp(x) (function target one level up)
 *  - d(u*u)/du == 2u                        (2D node, multi-occurrence)
 *  - d(sin(u)+u)/du == cos(u)+1             (2D node, multi-occurrence)
 *
 * Exercises (complex domain):
 *  - dw/dw == (1, 0)
 *  - d sin(w)/dw == cos(w)
 *
 * These cases all returned 0 before the @c IvyConstantFunction identity fix.
 */

#include "common_test_defs.h"

#include "autodiff/arithmetic/IvyMathBaseArithmetic.h"
#include "autodiff/basic_nodes/IvyComplex.h"

#include <cmath>
#include <complex>
#include "std_ivy/IvyCassert.h"


using namespace std_ivy;
using namespace IvyMath;


/** @brief Print a PASS/FAIL result line and assert on failure. */
static void check(bool cond, char const* label){
  if (cond){
    __PRINT_INFO__("  [PASS] %s\n", label);
  } else {
    __PRINT_INFO__("  [FAIL] %s\n", label);
    assert(false);
  }
}

static bool close(double a, double b, double tol = 1e-9){ return std::abs(a - b) < tol; }


static void test_real(){
  __PRINT_INFO__("--- real-domain function targets ---\n");
  constexpr double x0 = 1.0;
  auto x = Scalar<double>(IvyMemoryType::Host, nullptr, x0);
  auto u = Exp(x);            // u = exp(x)
  auto h = Sin(u);           // h = sin(u) = sin(exp(x))

  double const uval = u->value().value();
  double const ex   = std::exp(x0);

  // Sanity: differentiation w.r.t. the leaf still works.
  check(close(u->gradient(x)->value().value(), ex), "du/dx == exp(x)");
  check(close(h->gradient(x)->value().value(), std::cos(uval) * ex), "dh/dx == cos(u)exp(x)");

  // The core fix: differentiation w.r.t. a function node.
  check(close(u->gradient(u)->value().value(), 1.0), "du/du == 1");
  check(close(h->gradient(u)->value().value(), std::cos(uval)), "dh/du == cos(u)");

  // 2D nodes with the target appearing on multiple branches.
  auto f = Multiply(u, u);                       // f = u^2
  check(close(f->gradient(u)->value().value(), 2.0 * uval), "d(u*u)/du == 2u");
  check(close(f->gradient(x)->value().value(), 2.0 * uval * ex), "d(u*u)/dx == 2u exp(x)");

  auto g = Add(Sin(u), u);                        // g = sin(u) + u
  check(close(g->gradient(u)->value().value(), std::cos(uval) + 1.0), "d(sin(u)+u)/du == cos(u)+1");

  // A function that does not depend on the target gives zero.
  auto y = Scalar<double>(IvyMemoryType::Host, nullptr, 2.0);
  auto w = Exp(y);
  check(close(h->gradient(w)->value().value(), 0.0), "dh/dw == 0 (independent function)");
}


static void test_complex(){
  __PRINT_INFO__("--- complex-domain function targets ---\n");
  std::complex<double> const z0(0.5, 0.3);
  auto z = Complex<double>(IvyMemoryType::Host, nullptr, z0.real(), z0.imag());
  auto w = Exp(z);            // w = exp(z)
  auto g = Sin(w);           // g = sin(w)

  auto const& wv = w->value();
  std::complex<double> const wc(wv.Re(), wv.Im());

  auto dw_dw = w->gradient(w)->value();
  check(close(dw_dw.Re(), 1.0) && close(dw_dw.Im(), 0.0), "dw/dw == (1,0)");

  std::complex<double> const cosw = std::cos(wc);
  auto dg_dw = g->gradient(w)->value();
  check(close(dg_dw.Re(), cosw.real(), 1e-9) && close(dg_dw.Im(), cosw.imag(), 1e-9), "d sin(w)/dw == cos(w)");
}


void utest(){
  __PRINT_INFO__("=== utest_partial_diff_functions ===\n");
  test_real();
  test_complex();
  __PRINT_INFO__("=== ALL utest_partial_diff_functions tests PASSED ===\n");
}


int main(){
  utest();
  return 0;
}
