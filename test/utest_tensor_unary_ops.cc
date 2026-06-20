/**
 * @file utest_tensor_unary_ops.cc
 * @brief Functional coverage for element-wise math ops on differentiable tensors.
 *
 * Complements utest_tensor_ops.cc (binary/scalar/reduction) by exercising the
 * full set of element-wise unary functionals exposed by IvyMath on BOTH
 * differentiable tensor representations:
 *   (A) contiguous cells   : IvyTensor<IvyTensorScalarCell<T>>  (TensorScalar)
 *   (B) array-of-pointers  : IvyTensor<IvyScalarPtr_t<T>>       (Tensor sharing a leaf)
 *
 * Real-output ops (value + element-wise gradient through the lazy graph node):
 *   Sin, Cos, Tan, Cot, Sqrt, Log, Erf, Erfc, ErfFast, ErfcFast, SinH, CosH.
 * Non-differentiable op (value only): Abs.
 * Complex-output ops (eager value only — output domain differs from input):
 *   Faddeeva, FaddeevaFast.
 * Binary power on tensors (eager value): Pow(t, scalar) and Pow(t, t).
 */

#include "common_test_defs.h"

#include "autodiff/arithmetic/IvyMathBaseArithmetic.h"
#include "autodiff/basic_nodes/IvyTensor.h"

#include <cmath>
#include "std_ivy/IvyCassert.h"


using namespace std_ivy;
using namespace IvyMath;


static void check(bool cond, char const* label){
  if (cond){ __PRINT_INFO__("  [PASS] %s\n", label); }
  else { __PRINT_INFO__("  [FAIL] %s\n", label); assert(false); }
}
static bool close(double a, double b, double tol = 1e-9){ return std::abs(a - b) < tol; }

// Read a tensor element as a double, regardless of representation.
template<typename E> static double rd(E const& e){
  if constexpr (is_pointer_v<E>) return (*e).value();
  else if constexpr (std_ttraits::is_arithmetic_v<E>) return __STATIC_CAST__(double, e);
  else return e.value();
}
template<typename Tens, typename Fn> static bool all_of(Tens const& t, Fn fn, double tol = 1e-9){
  for (IvyTensorDim_t i = 0; i < t.num_elements(); ++i) if (!close(rd(t[i]), fn(), tol)) return false;
  return true;
}
// Read Re() of a complex tensor element, regardless of representation.
template<typename E> static double re_of(E const& e){
  if constexpr (is_pointer_v<E>) return (*e).Re();
  else return e.Re();
}
template<typename Tens, typename Fn> static bool all_re(Tens const& t, Fn fn, double tol){
  for (IvyTensorDim_t i = 0; i < t.num_elements(); ++i) if (!close(re_of(t[i]), fn(), tol)) return false;
  return true;
}


// Exercises every op against a tensor leaf `t` whose elements all equal `a`,
// differentiated w.r.t. the seed node `seed`.
template<typename Tptr, typename Seed>
static void run_suite(char const* tag, Tptr const& t, Seed const& seed, double a){
  __PRINT_INFO__("-- %s --\n", tag);
  const double TOSP = 2.0 / std::sqrt(M_PI);   // 2/sqrt(pi)

  auto diff_op = [&](auto node, auto val_fn, auto grad_fn, char const* name){
    char lbl[128];
    std::snprintf(lbl, sizeof(lbl), "%s value", name);
    check(all_of(node->value(), val_fn), lbl);
    auto g = node->gradient(seed);
    std::snprintf(lbl, sizeof(lbl), "d %s / dx", name);
    check(all_of(g->value(), grad_fn), lbl);
  };

  diff_op(Sin(t),  [&]{ return std::sin(a); },  [&]{ return std::cos(a); },  "Sin");
  diff_op(Cos(t),  [&]{ return std::cos(a); },  [&]{ return -std::sin(a); }, "Cos");
  diff_op(Tan(t),  [&]{ return std::tan(a); },  [&]{ double c=std::cos(a); return 1.0/(c*c); }, "Tan");
  diff_op(Cot(t),  [&]{ return std::cos(a)/std::sin(a); }, [&]{ double s=std::sin(a); return -1.0/(s*s); }, "Cot");
  diff_op(Sqrt(t), [&]{ return std::sqrt(a); }, [&]{ return 0.5/std::sqrt(a); }, "Sqrt");
  diff_op(Log(t),  [&]{ return std::log(a); },  [&]{ return 1.0/a; },          "Log");
  diff_op(Erf(t),  [&]{ return std::erf(a); },  [&]{ return TOSP*std::exp(-a*a); },  "Erf");
  diff_op(Erfc(t), [&]{ return std::erfc(a); }, [&]{ return -TOSP*std::exp(-a*a); }, "Erfc");
  diff_op(ErfFast(t),  [&]{ return std::erf(a); },  [&]{ return TOSP*std::exp(-a*a); },  "ErfFast");
  diff_op(ErfcFast(t), [&]{ return std::erfc(a); }, [&]{ return -TOSP*std::exp(-a*a); }, "ErfcFast");
  diff_op(SinH(t), [&]{ return std::sinh(a); }, [&]{ return std::cosh(a); }, "SinH");
  diff_op(CosH(t), [&]{ return std::cosh(a); }, [&]{ return std::sinh(a); }, "CosH");

  // Composition through several element-wise ops.
  {
    auto f = Log(SinH(Sqrt(t)));
    check(all_of(f->value(), [&]{ return std::log(std::sinh(std::sqrt(a))); }), "Log(SinH(Sqrt(t))) value");
    auto g = f->gradient(seed);
    check(all_of(g->value(), [&]{
      double s = std::sqrt(a);
      return (std::cosh(s)/std::sinh(s)) * (0.5/s);   // d/dx log(sinh(sqrt(x)))
    }), "d Log(SinH(Sqrt(t)))/dx");
  }

  // Non-differentiable: Abs (eager value).
  check(all_of(Abs(*t), [&]{ return std::abs(a); }), "Abs value (eager)");

  // Complex-output (eager value only): Re w(x) = exp(-x^2) for real x.
  check(all_re(Faddeeva(*t),     [&]{ return std::exp(-a*a); }, 1e-6), "Faddeeva Re value (eager)");
  check(all_re(FaddeevaFast(*t), [&]{ return std::exp(-a*a); }, 1e-3), "FaddeevaFast Re value (eager)");

  // Power on tensors (eager value): t^scalar and t^t.
  check(all_of(Pow(*t, 3.0), [&]{ return std::pow(a, 3.0); }), "Pow(t, 3) value (eager)");
  check(all_of(Pow(*t, *t),  [&]{ return std::pow(a, a);   }), "Pow(t, t) value (eager)");
}


void utest(){
  __PRINT_INFO__("=== utest_tensor_unary_ops ===\n");

  IvyTensorShape shape({ 2, 3 });            // 6 elements
  auto mem = shape.get_memory_type();
  auto st  = shape.gpu_stream();
  constexpr double a = 0.7;

  // (A) contiguous-cell representation; differentiate w.r.t. the tensor node.
  {
    auto t = TensorScalar<double>(mem, st, shape, IvyTensorScalarCell<double>(a));
    IvyThreadSafePtr_t<IvyBaseNode> tn(t);
    run_suite("contiguous-cell representation", t, tn, a);
  }

  // (B) array-of-pointers representation; all elements share one scalar leaf x,
  //     so gradients are taken w.r.t. x.
  {
    auto x = Scalar<double>(mem, st, a);
    auto t = Tensor<IvyScalarPtr_t<double>>(mem, st, shape, x);
    IvyThreadSafePtr_t<IvyBaseNode> xn(x);
    run_suite("array-of-pointers representation", t, xn, a);
  }

  __PRINT_INFO__("=== ALL utest_tensor_unary_ops tests PASSED ===\n");
}


int main(){
  utest();
  return 0;
}
