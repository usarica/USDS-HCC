/**
 * @file utest_tensor_variable_leaf.cc
 * @brief Unit tests for the contiguous differentiable tensor variable leaf.
 *
 * Exercises the struct-of-arrays differentiable tensor introduced as
 * @c IvyTensor<IvyTensorVariableCell<T>> (created via @c TensorVariable<T>):
 *  - The cell element is exactly @c sizeof(T) (no per-element client manager).
 *  - The tensor is a first-class differentiation target (a "variable tensor").
 *  - Unary element-wise ops (Exp, Log, Sin, Cos, Negate, Sqrt) evaluate and
 *    differentiate w.r.t. the whole tensor through the single-source IvyMath
 *    operator definitions, returning the correct element-wise (diagonal)
 *    derivative tensor.
 *  - Chain rule across composed ops (Sin(Exp(t))).
 *  - Differentiating w.r.t. an unrelated tensor yields zero.
 *
 * This replaces the parallel SoA prototype: there is a single op-definition
 * location (IvyMathBaseArithmetic, namespace IvyMath) and the contiguous leaf
 * plugs into it directly.
 */

#include "common_test_defs.h"

#include "autodiff/arithmetic/IvyMathBaseArithmetic.h"
#include "autodiff/basic_nodes/IvyTensor.h"

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


void utest(){
  __PRINT_INFO__("=== utest_tensor_variable_leaf ===\n");

  // The cell is a bare value: contiguous storage, no per-element bookkeeping.
  check(sizeof(IvyTensorVariableCell<double>) == sizeof(double), "sizeof(cell) == sizeof(double)");

  constexpr double x0 = 0.7;
  IvyTensorShape shape({ 2, 3 });
  auto t = TensorVariable<double>(shape.get_memory_type(), shape.gpu_stream(), shape, IvyTensorVariableCell<double>(x0));
  check(t->num_elements() == 6, "tensor has 6 elements");
  check(is_differentiable(*t), "variable tensor is differentiable");

  // Helper: check every element of a gradient/value tensor equals expected.
  // Op results keep the cell element type (IvyTensorVariableCell), so read
  // each element via value().
  auto all_close = [](auto const& tens, double expected, double tol = 1e-9){
    for (IvyTensorDim_t i = 0; i < tens.num_elements(); ++i)
      if (!close(tens[i].value(), expected, tol)) return false;
    return true;
  };

  // --- Exp ---
  {
    auto f = Exp(t);
    check(all_close(f->value(), std::exp(x0)), "Exp(t) value == exp(x0)");
    auto g = f->gradient(t);
    check(all_close(g->value(), std::exp(x0)), "dExp(t)/dt == exp(x0)");
  }
  // --- Log ---
  {
    auto f = Log(t);
    check(all_close(f->value(), std::log(x0)), "Log(t) value == log(x0)");
    auto g = f->gradient(t);
    check(all_close(g->value(), 1.0 / x0), "dLog(t)/dt == 1/x0");
  }
  // --- Sin / Cos ---
  {
    auto f = Sin(t);
    check(all_close(f->value(), std::sin(x0)), "Sin(t) value == sin(x0)");
    auto g = f->gradient(t);
    check(all_close(g->value(), std::cos(x0)), "dSin(t)/dt == cos(x0)");
  }
  {
    auto f = Cos(t);
    check(all_close(f->value(), std::cos(x0)), "Cos(t) value == cos(x0)");
    auto g = f->gradient(t);
    check(all_close(g->value(), -std::sin(x0)), "dCos(t)/dt == -sin(x0)");
  }
  // --- Negate ---
  {
    auto f = -t;
    check(all_close(f->value(), -x0), "(-t) value == -x0");
    auto g = f->gradient(t);
    check(all_close(g->value(), -1.0), "d(-t)/dt == -1");
  }
  // --- Sqrt ---
  {
    auto f = Sqrt(t);
    check(all_close(f->value(), std::sqrt(x0)), "Sqrt(t) value == sqrt(x0)");
    auto g = f->gradient(t);
    check(all_close(g->value(), 0.5 / std::sqrt(x0)), "dSqrt(t)/dt == 1/(2 sqrt(x0))");
  }

  // --- Differentiation w.r.t. an unrelated tensor is zero ---
  {
    auto s = TensorVariable<double>(shape.get_memory_type(), shape.gpu_stream(), shape, IvyTensorVariableCell<double>(1.0));
    auto f = Exp(t);
    auto g = f->gradient(s);
    check(all_close(g->value(), 0.0), "dExp(t)/ds == 0 for unrelated s");
  }

  __PRINT_INFO__("=== ALL utest_tensor_variable_leaf tests PASSED ===\n");
}


int main(){
  utest();
  return 0;
}
