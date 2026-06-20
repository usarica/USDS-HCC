/**
 * @file utest_tensor_ops.cc
 * @brief Unit tests for Phase 3: functions of tensors / tensors of functions.
 *
 * Exercises differentiable tensor-domain arithmetic through the single-source
 * IvyMath operator definitions (IvyMathBaseArithmetic, namespace IvyMath), on
 * BOTH differentiable tensor representations:
 *   (A) contiguous cells   : IvyTensor<IvyTensorScalarCell<T>>  (TensorScalar)
 *   (B) array-of-pointers  : IvyTensor<IvyScalarPtr_t<T>>       (Tensor sharing a leaf)
 *
 * Coverage:
 *   - Unary chains across composed tensor ops: Sin(Exp(t)), Log(Sin(Exp(t))).
 *   - Binary tensor (X) tensor: +, -, *, / value and element-wise gradients.
 *   - Tensor (X) scalar with broadcast AND differentiation w.r.t. the scalar
 *     (the scalar may be an IvyScalar leaf): d(s*t)/ds, d(t/s)/ds, ...
 *   - Up-casting: int-tensor (X) double-scalar -> double tensor; real-tensor (X)
 *     complex-scalar -> complex tensor (value + d/d tensor).
 *   - Differentiable Sum reduction (tensor -> scalar): Sum(t), Sum(s*t),
 *     Sum(Exp(t)), with d/d scalar and d/d tensor, on both representations.
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

// Read a tensor element as a double, regardless of representation: a flat reduced
// value (double), a contiguous cell (.value()), or an array-of-pointers leaf (*->value()).
template<typename E> static double rd(E const& e){
  if constexpr (is_pointer_v<E>) return (*e).value();
  else if constexpr (std_ttraits::is_arithmetic_v<E>) return __STATIC_CAST__(double, e);
  else return e.value();
}
// Every element of a value/gradient tensor (any representation) equals expected.
template<typename Tens, typename Fn> static bool all_of(Tens const& t, Fn fn, double tol = 1e-9){
  for (IvyTensorDim_t i = 0; i < t.num_elements(); ++i) if (!close(rd(t[i]), fn(i), tol)) return false;
  return true;
}
static double sval(IvyScalar<double> const& s){ return s.value(); }


void utest(){
  __PRINT_INFO__("=== utest_tensor_ops ===\n");

  IvyTensorShape shape({ 2, 3 });            // 6 elements
  auto mem = shape.get_memory_type();
  auto st  = shape.gpu_stream();
  constexpr IvyTensorDim_t N = 6;

  // ── (A) contiguous-cell representation ───────────────────────────────────
  {
    __PRINT_INFO__("-- contiguous-cell representation --\n");
    constexpr double a = 0.7;
    auto t = TensorScalar<double>(mem, st, shape, IvyTensorScalarCell<double>(a));
    IvyThreadSafePtr_t<IvyBaseNode> tn(t);

    // Unary chains
    {
      auto f = Sin(Exp(t));
      check(all_of(f->value(), [&](IvyTensorDim_t){ return std::sin(std::exp(a)); }), "Sin(Exp(t)) value");
      auto g = f->gradient(tn);
      check(all_of(g->value(), [&](IvyTensorDim_t){ return std::cos(std::exp(a))*std::exp(a); }), "d Sin(Exp(t))/dt");

      auto h = Log(Sin(Exp(t)));
      check(all_of(h->value(), [&](IvyTensorDim_t){ return std::log(std::sin(std::exp(a))); }), "Log(Sin(Exp(t))) value");
      auto gh = h->gradient(tn);
      double const e = std::exp(a);
      check(all_of(gh->value(), [&](IvyTensorDim_t){ return (std::cos(e)*e)/std::sin(e); }), "d Log(Sin(Exp(t)))/dt");
    }

    // Binary tensor (X) tensor — two independent leaves
    {
      auto u = TensorScalar<double>(mem, st, shape, IvyTensorScalarCell<double>(3.0));
      IvyThreadSafePtr_t<IvyBaseNode> un(u);
      auto prod = t * u;
      check(all_of(prod->value(), [&](IvyTensorDim_t){ return a*3.0; }), "t*u value");
      check(all_of(prod->gradient(tn)->value(), [&](IvyTensorDim_t){ return 3.0; }), "d(t*u)/dt = u");
      check(all_of(prod->gradient(un)->value(), [&](IvyTensorDim_t){ return a; }), "d(t*u)/du = t");
      auto quot = t / u;
      check(all_of(quot->value(), [&](IvyTensorDim_t){ return a/3.0; }), "t/u value");
      check(all_of(quot->gradient(tn)->value(), [&](IvyTensorDim_t){ return 1.0/3.0; }), "d(t/u)/dt = 1/u");
      check(all_of(quot->gradient(un)->value(), [&](IvyTensorDim_t){ return -a/9.0; }), "d(t/u)/du = -t/u^2");
    }

    // Tensor (X) scalar broadcast + d/d scalar
    {
      auto s = Scalar<double>(IvyMemoryType::Host, nullptr, 5.0);
      IvyThreadSafePtr_t<IvyBaseNode> sn(s);
      auto f = s * t;
      check(all_of(f->value(), [&](IvyTensorDim_t){ return 5.0*a; }), "s*t value (broadcast)");
      check(all_of(f->gradient(tn)->value(), [&](IvyTensorDim_t){ return 5.0; }), "d(s*t)/dt = s");
      check(all_of(f->gradient(sn)->value(), [&](IvyTensorDim_t){ return a; }), "d(s*t)/ds = t");
      auto q = t / s;
      check(all_of(q->gradient(sn)->value(), [&](IvyTensorDim_t){ return -a/25.0; }), "d(t/s)/ds = -t/s^2");
      auto d = s - t;
      check(all_of(d->gradient(sn)->value(), [&](IvyTensorDim_t){ return 1.0; }), "d(s-t)/ds = 1");
      check(all_of(d->gradient(tn)->value(), [&](IvyTensorDim_t){ return -1.0; }), "d(s-t)/dt = -1");
    }

    // Sum reduction
    {
      auto f = Sum(t);
      check(close(sval(f->value()), N*a), "Sum(t) value");
      check(close(sval(f->gradient(tn)->value()), N*1.0), "d Sum(t)/dt = N");
      auto s = Scalar<double>(IvyMemoryType::Host, nullptr, 5.0);
      IvyThreadSafePtr_t<IvyBaseNode> sn(s);
      auto g = Sum(s * t);
      check(close(sval(g->value()), N*5.0*a), "Sum(s*t) value");
      check(close(sval(g->gradient(sn)->value()), N*a), "d Sum(s*t)/ds = Sum(t)");
      check(close(sval(g->gradient(tn)->value()), N*5.0), "d Sum(s*t)/dt = Sum(s)");
      auto h = Sum(Exp(t));
      check(close(sval(h->value()), N*std::exp(a)), "Sum(Exp(t)) value");
      check(close(sval(h->gradient(tn)->value()), N*std::exp(a)), "d Sum(Exp(t))/dt");
    }
  }

  // ── (B) array-of-pointers representation ─────────────────────────────────
  {
    __PRINT_INFO__("-- array-of-pointers representation --\n");
    auto x = Scalar<double>(IvyMemoryType::Host, nullptr, 0.7);
    auto y = Scalar<double>(IvyMemoryType::Host, nullptr, 3.0);
    IvyThreadSafePtr_t<IvyBaseNode> xn(x), yn(y);
    auto tx = Tensor<IvyScalarPtr_t<double>>(mem, st, shape, x);   // every element shares x
    auto ty = Tensor<IvyScalarPtr_t<double>>(mem, st, shape, y);

    auto prod = tx * ty;
    check(all_of(prod->value(), [&](IvyTensorDim_t){ return 0.7*3.0; }), "AoP t*u value");
    check(all_of(prod->gradient(xn)->value(), [&](IvyTensorDim_t){ return 3.0; }), "AoP d(t*u)/dx = u");
    check(all_of(prod->gradient(yn)->value(), [&](IvyTensorDim_t){ return 0.7; }), "AoP d(t*u)/dy = t");

    auto chain = Sin(Exp(tx));
    check(all_of(chain->value(), [&](IvyTensorDim_t){ return std::sin(std::exp(0.7)); }), "AoP Sin(Exp(t)) value");
    check(all_of(chain->gradient(xn)->value(), [&](IvyTensorDim_t){ return std::cos(std::exp(0.7))*std::exp(0.7); }), "AoP d Sin(Exp(t))/dx");

    auto red = Sum(tx);
    check(close(sval(red->value()), N*0.7), "AoP Sum(t) value");
    check(close(sval(red->gradient(xn)->value()), N*1.0), "AoP d Sum(t)/dx = N");
  }

  // ── Up-casting ───────────────────────────────────────────────────────────
  {
    __PRINT_INFO__("-- up-casting --\n");
    IvyTensorShape sh2({ 2, 2 });
    auto m2 = sh2.get_memory_type();
    auto st2 = sh2.gpu_stream();

    // int-tensor (X) double-scalar -> double tensor value
    auto ti = TensorScalar<int>(m2, st2, sh2, IvyTensorScalarCell<int>(3));
    auto sd = Scalar<double>(IvyMemoryType::Host, nullptr, 2.5);
    auto fi = ti * sd;
    static_assert(std_ttraits::is_same_v<std_ttraits::remove_reference_t<decltype(fi->value())>::dtype_t, double>,
                  "int-tensor * double-scalar up-casts element type to double");
    check(all_of(fi->value(), [](IvyTensorDim_t){ return 7.5; }), "int-tensor * double-scalar value = 7.5");

    // real-tensor (X) complex-scalar -> complex tensor value + d/d tensor
    auto tr = TensorScalar<double>(m2, st2, sh2, IvyTensorScalarCell<double>(2.0));
    IvyThreadSafePtr_t<IvyBaseNode> trn(tr);
    auto c = Complex<double>(IvyMemoryType::Host, nullptr, 3.0, 4.0);
    auto fc = c * tr;
    static_assert(std_ttraits::is_same_v<std_ttraits::remove_reference_t<decltype(fc->value())>::dtype_t, IvyComplex<double>>,
                  "real-tensor * complex-scalar up-casts element type to IvyComplex");
    bool okv = true, okg = true;
    auto const& vf = fc->value();
    for (IvyTensorDim_t i = 0; i < vf.num_elements(); ++i){ okv &= close(vf[i].Re(), 6.0) && close(vf[i].Im(), 8.0); }
    check(okv, "complex-scalar * real-tensor value = 6+8i");
    auto gf = fc->gradient(trn);
    auto const& vg = gf->value();
    for (IvyTensorDim_t i = 0; i < vg.num_elements(); ++i){ okg &= close(vg[i].Re(), 3.0) && close(vg[i].Im(), 4.0); }
    check(okg, "d(c*t)/dt = c = 3+4i (complex broadcast)");
  }

  __PRINT_INFO__("=== ALL utest_tensor_ops tests PASSED ===\n");
}


int main(){
  utest();
  return 0;
}
