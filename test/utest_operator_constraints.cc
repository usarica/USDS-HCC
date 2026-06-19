/**
 * @file utest_operator_constraints.cc
 * @brief Regression tests ensuring the global IvyMath operators do not hijack foreign types.
 *
 * The IvyMath::operator+/-/* / overloads are templated and live in namespace IvyMath. Before the
 * is_ivy_domain_v constraint, with `using namespace IvyMath` in scope these overloads were viable
 * candidates for unrelated types (e.g. std::chrono::time_point), producing hard compile errors deep
 * inside SubtractFcnal instead of selecting the type's own operators.
 *
 * This test compiles and runs foreign-type arithmetic under `using namespace IvyMath` and confirms:
 *  - std::chrono time_point subtraction and duration +/- resolve to the standard library, and
 *  - genuine Ivy-domain operators still build the autodiff graph and differentiate correctly.
 *
 * The main signal is that this translation unit compiles at all; the runtime checks guard values.
 */

#include "common_test_defs.h"

#include "autodiff/arithmetic/IvyMathBaseArithmetic.h"

#include <chrono>
#include <complex>
#include <cmath>
#include "std_ivy/IvyCassert.h"


using namespace std_ivy;
using namespace IvyMath;


static void check(bool cond, char const* label){
  if (cond){ __PRINT_INFO__("  [PASS] %s\n", label); }
  else { __PRINT_INFO__("  [FAIL] %s\n", label); assert(false); }
}

static bool close(double a, double b, double tol = 1e-9){ return std::abs(a - b) < tol; }


static void test_foreign_types_not_hijacked(){
  __PRINT_INFO__("--- foreign types use their own operators ---\n");

  // std::chrono::time_point subtraction must resolve to std::operator- (returns a duration),
  // NOT IvyMath::operator-. Before the fix this failed to compile.
  auto t0 = std::chrono::steady_clock::now();
  auto t1 = std::chrono::steady_clock::now();
  std::chrono::duration<double, std::milli> const dt = t1 - t0;
  check(dt.count() >= 0.0, "chrono time_point subtraction resolves to std::operator-");

  // std::chrono::duration +/- must also resolve to the standard library.
  std::chrono::duration<double> const d1(2.5);
  std::chrono::duration<double> const d2(1.0);
  check(close((d1 + d2).count(), 3.5), "chrono duration operator+ resolves to std");
  check(close((d1 - d2).count(), 1.5), "chrono duration operator- resolves to std");

  // std::complex (a foreign type that does define its own operators) must keep using them.
  std::complex<double> const c1(1.0, 2.0), c2(3.0, -1.0);
  std::complex<double> const csum = c1 + c2;
  std::complex<double> const cprod = c1 * c2;
  check(close(csum.real(), 4.0) && close(csum.imag(), 1.0), "std::complex operator+ resolves to std");
  check(close(cprod.real(), 5.0) && close(cprod.imag(), 5.0), "std::complex operator* resolves to std");
}


static void test_ivy_operators_still_work(){
  __PRINT_INFO__("--- Ivy-domain operators still build the graph ---\n");

  auto x = Variable<double>(IvyMemoryType::Host, nullptr, 2.0);
  auto y = Variable<double>(IvyMemoryType::Host, nullptr, 3.0);

  auto f = Add(x, Multiply(x, y));   // f = x + x*y
  check(close(f->value().value(), 8.0), "x + x*y == 8");
  check(close(f->gradient(x)->value().value(), 4.0), "d(x + x*y)/dx == 1 + y == 4");
  check(close(f->gradient(y)->value().value(), 2.0), "d(x + x*y)/dy == x == 2");

  auto g = Subtract(x, y);           // g = x - y
  check(close(g->value().value(), -1.0), "x - y == -1");

  auto h = Divide(x, y);             // h = x / y
  check(close(h->value().value(), 2.0 / 3.0), "x / y == 2/3");
}


void utest(){
  __PRINT_INFO__("=== utest_operator_constraints ===\n");
  test_foreign_types_not_hijacked();
  test_ivy_operators_still_work();
  __PRINT_INFO__("=== ALL utest_operator_constraints tests PASSED ===\n");
}


int main(){
  utest();
  return 0;
}
