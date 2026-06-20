#include "autodiff/arithmetic/IvyMathBaseArithmetic.h"

/**
 * @file shared_library_smoke.cc
 * @brief Minimal warning-free shared-library linkage smoke test.
 *
 * This executable is intentionally small and uses only scalar autodiff
 * operations so that the documented shared-library build/link/include command
 * can be validated without warning suppression flags.
 */
int main(){
  using namespace std_ivy;
  using namespace IvyMath;

  auto x = Scalar<double>(IvyMemoryType::Host, nullptr, 1.0);
  auto f = Sin(Exp(x));
  auto g = f->gradient(x);

  constexpr double min_expected = -10.0;
  constexpr double max_expected = 10.0;
  double const v = g->value().value();
  return (v>min_expected && v<max_expected ? 0 : 1);
}
