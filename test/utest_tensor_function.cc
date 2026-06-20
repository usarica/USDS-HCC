/**
 * @file utest_tensor_function.cc
 * @brief Unit tests for tensor-domain autodiff: element-wise functions and gradients.
 *
 * Exercises:
 *  - Construction of IvyTensor<IvyScalarPtr_t<double>> from a scalar variable.
 *  - Applying Exp, Log, Sin, Cos to a tensor pointer (IvyTensorPtr_t).
 *  - Evaluating the resulting tensor functions (value()).
 *  - Computing gradients via gradient() and checking the output tensor shape and values.
 *  - Verifying that the function_gradient tensor specialization produces the correct
 *    derivative tensors (all ones when all elements share the same variable).
 */

#include "common_test_defs.h"

// Include IvyMathBaseArithmetic.h first so that IvyComplex.h (and thus
// convert_to_complex_type's primary template) is defined before IvyTensor.h
// adds its partial specialisation of convert_to_complex_type.
#include "autodiff/arithmetic/IvyMathBaseArithmetic.h"
#include "autodiff/basic_nodes/IvyTensor.h"

#include <cmath>
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

/** @brief Return the scalar value stored by an IvyScalarPtr_t<double>. */
static double var_val(IvyMath::IvyScalarPtr_t<double> const& p){
  return p->value();
}


void utest(){
  constexpr double tol = 1e-9;

  __PRINT_INFO__("=== utest_tensor_function ===\n");

  //--------------------------------------------------------------------------
  // 1. Construct a scalar variable and a tensor filled with pointers to it.
  //--------------------------------------------------------------------------
  auto x = Scalar<double>(IvyMemoryType::Host, nullptr, 3.0);
  IvyTensorShape shape({ 2, 3 });

  auto t = Tensor<IvyScalarPtr_t<double>>(
    shape.get_memory_type(), shape.gpu_stream(), shape, x
  );
  check(t->num_elements() == 6, "Tensor has 6 elements");
  check(std::abs(var_val(t->at({0,0})) - 3.0) < tol, "t[0,0] == 3");
  check(t->at({0,0}).get() == x.get(), "t[0,0] shares pointer with x");

  __PRINT_INFO__("  Tensor construction: OK\n");

  //--------------------------------------------------------------------------
  // 2. Exp(t) — create the function, evaluate it.
  //--------------------------------------------------------------------------
  auto fcn_exp = Exp(t);
  check(fcn_exp != nullptr, "Exp(t) != nullptr");

  auto const& val_exp = fcn_exp->value();
  check(val_exp.num_elements() == 6, "Exp(t) output has 6 elements");

  double const expected_exp = std::exp(3.0);
  for (IvyTensorDim_t i = 0; i < val_exp.num_elements(); ++i){
    check(std::abs(var_val(val_exp[i]) - expected_exp) < tol, "Exp(t)[i] == exp(3)");
  }
  __PRINT_INFO__("  Exp(t) eval: OK  (exp(3) = %.10f)\n", expected_exp);

  //--------------------------------------------------------------------------
  // 3. Log(t)
  //--------------------------------------------------------------------------
  auto fcn_log = Log(t);
  auto const& val_log = fcn_log->value();
  double const expected_log = std::log(3.0);
  check(std::abs(var_val(val_log[0]) - expected_log) < tol, "Log(t)[0] == log(3)");
  __PRINT_INFO__("  Log(t) eval: OK  (log(3) = %.10f)\n", expected_log);

  //--------------------------------------------------------------------------
  // 4. Sin(t) and Cos(t)
  //--------------------------------------------------------------------------
  auto fcn_sin = Sin(t);
  auto const& val_sin = fcn_sin->value();
  check(std::abs(var_val(val_sin[0]) - std::sin(3.0)) < tol, "Sin(t)[0] == sin(3)");
  __PRINT_INFO__("  Sin(t) eval: OK\n");

  auto fcn_cos = Cos(t);
  auto const& val_cos = fcn_cos->value();
  check(std::abs(var_val(val_cos[0]) - std::cos(3.0)) < tol, "Cos(t)[0] == cos(3)");
  __PRINT_INFO__("  Cos(t) eval: OK\n");

  //--------------------------------------------------------------------------
  // 5. function_gradient for the tensor: ∂t[i]/∂x == 1 for all i.
  //--------------------------------------------------------------------------
  {
    using tensor_t = IvyTensor<IvyScalarPtr_t<double>>;
    IvyThreadSafePtr_t<IvyBaseNode> base_x(x);
    auto grad_t_ptr = function_gradient<tensor_t>::get(*t, base_x);
    check(grad_t_ptr != nullptr, "function_gradient<tensor> != nullptr");
    check(grad_t_ptr->num_elements() == 6, "gradient tensor has 6 elements");
    for (IvyTensorDim_t i = 0; i < grad_t_ptr->num_elements(); ++i){
      double gval = var_val((*grad_t_ptr)[i]);
      check(std::abs(gval - 1.0) < tol, "∂t[i]/∂x == 1");
    }
    __PRINT_INFO__("  function_gradient<tensor>: OK\n");
  }

  // ── Tensor gradient tests ────────────────────────────────────────────────
  auto fcn_exp2 = Exp(t);
  auto grad_exp = fcn_exp2->gradient(x);
  auto const& vexp = grad_exp->value();
  bool exp_ok = true;
  for (IvyTensorDim_t i = 0; i < vexp.num_elements(); ++i)
    if (std::abs(var_val(vexp[i]) - std::exp(3.0)) >= 1e-6) exp_ok = false;
  check(exp_ok, "partial Exp(t)[i] / partial x == exp(3)");

  auto fcn_log2 = Log(t);
  auto grad_log = fcn_log2->gradient(x);
  auto const& vlog = grad_log->value();
  bool log_ok = true;
  for (IvyTensorDim_t i = 0; i < vlog.num_elements(); ++i)
    if (std::abs(var_val(vlog[i]) - 1.0/3.0) >= 1e-6) log_ok = false;
  check(log_ok, "partial Log(t)[i] / partial x == 1/3");

  auto fcn_sin2 = Sin(t);
  auto grad_sin = fcn_sin2->gradient(x);
  auto const& vsin = grad_sin->value();
  check(std::abs(var_val(vsin[0]) - std::cos(3.0)) < 1e-6, "partial Sin(t)[0] / partial x == cos(3)");

  auto fcn_neg2 = -t;
  auto grad_neg = fcn_neg2->gradient(x);
  auto const& vneg = grad_neg->value();
  check(std::abs(var_val(vneg[0]) - (-1.0)) < 1e-6, "partial (-t)[0] / partial x == -1");

  __PRINT_INFO__("  Tensor gradient tests: OK\n");

  __PRINT_INFO__("=== ALL utest_tensor_function tests PASSED ===\n");
}


int main(){
  utest();
  return 0;
}
