#ifndef IVYMATHBASEARITHMETIC_H
#define IVYMATHBASEARITHMETIC_H


#include "autodiff/arithmetic/IvyMathBaseArithmetic.hh"
#include "std_ivy/IvyCmath.h"
#include "autodiff/special_functions/cerf/IvyCerf.h"

/**
 * @file IvyMathBaseArithmetic.h
 * @brief Definitions for arithmetic autodiff functionals.
 *
 * Definitions corresponding to pointer-dependent gradients use
 * @c IVY_MATH_GRAPH_QUALIFIER to enforce host-only graph execution.
 * Pure numeric paths remain @c __HOST_DEVICE__ and are unaffected.
 */


namespace IvyMath{
  /**
   * @brief Apply a scalar callable element-wise over a tensor, preserving its representation.
   *
   * Single-source helper for tensor-domain element-wise unary functionals: it
   * loops over a tensor's elements, reads each element's canonical numeric value
   * (a bare arithmetic type for the real domain, an @c IvyComplex for the complex
   * domain), applies @p fcn, and writes the result back in the operand's
   * representation (an array-of-pointers leaf or a contiguous cell).  Both the
   * value path (f(x)) and the gradient path (f'(x)) of an element-wise functional
   * are expressed by passing the appropriate scalar callable, which keeps each new
   * op a two-liner instead of a duplicated loop.
   *
   * The callable is **domain-generic**: it receives whatever canonical value the
   * element holds, so its body must be written with the domain-aware @c IvyMath
   * free functions (e.g. @c IvyMath::Exp), never the real-only @c std_math
   * primitives.  This makes one definition serve both real and complex tensors.
   *
   * @tparam T    An IvyTensor type (real- or complex-domain element type).
   * @tparam Fcn  A scalar callable: canonical value -> canonical value.
   * @param src   The source tensor.
   * @param fcn   The scalar map applied to every element's canonical value.
   * @return      A tensor (same shape/representation as @p src) holding @c fcn(elem).
   */
  template<typename T, typename Fcn>
  __HOST__ T tensor_apply_elementwise(T const& src, Fcn const& fcn){
    constexpr std_ivy::IvyMemoryType def_mem_type = IvyMemoryHelpers::get_execution_default_memory();
    using dtype_t = typename T::dtype_t;
    T res(src);
    for (IvyTensorDim_t i = 0; i < src.num_elements(); ++i){
      if constexpr (is_pointer_v<dtype_t>){
        // Array-of-pointers leaf: unpack the pointee's canonical value
        // (arithmetic for a real IvyScalar, IvyComplex for a complex leaf).
        using inner_t = typename dtype_t::element_type;
        auto const val = fcn(unpack_function_input_reduced<inner_t>::get(*src[i]));
        res[i] = make_IvyThreadSafePtr<inner_t>(def_mem_type, nullptr, val);
      } else if constexpr (is_complex_v<dtype_t>){
        // Contiguous complex cell: read as an IvyComplex so the callable runs in
        // the complex domain, then store the (Re, Im) back into the cell (the
        // cell is not assignable from an IvyComplex directly).
        using rT = typename dtype_t::dtype_t;
        auto const val = fcn(IvyComplex<rT>(src[i].Re(), src[i].Im()));
        res[i] = dtype_t(val.Re(), val.Im());
      } else {
        // Contiguous real cell (or bare-arithmetic, non-differentiable tensor).
        res[i] = fcn(unpack_function_input_reduced<dtype_t>::get(src[i]));
      }
    }
    return res;
  }

  /****************/
  /* 1D FUNCTIONS */
  /****************/

  // Get real part of a variable
  template<typename T, typename domain_tag>
  __HOST_DEVICE__ RealFcnal<T, domain_tag>::value_t RealFcnal<T, domain_tag>::eval(T const& x){ return x; }
  template<typename T>
  __HOST_DEVICE__ RealFcnal<T, complex_domain_tag>::value_t RealFcnal<T, complex_domain_tag>::eval(T const& x){ return unpack_function_input_reduced<T>::get(x).Re(); }
  template<typename T, ENABLE_IF_BOOL_IMPL(!is_pointer_v<T>)> __HOST_DEVICE__ typename RealFcnal<T>::value_t Real(T const& x){ return RealFcnal<T>::eval(x); }

  // Get imaginary part of a variable
  template<typename T, typename domain_tag>
  __HOST_DEVICE__ constexpr ImagFcnal<T, domain_tag>::value_t ImagFcnal<T, domain_tag>::eval(T const& x){ return Zero<value_t>(); }
  template<typename T>
  __HOST_DEVICE__ ImagFcnal<T, complex_domain_tag>::value_t ImagFcnal<T, complex_domain_tag>::eval(T const& x){ return unpack_function_input_reduced<T>::get(x).Im(); }
  template<typename T, ENABLE_IF_BOOL_IMPL(!is_pointer_v<T>)> __HOST_DEVICE__ typename ImagFcnal<T>::value_t Imag(T const& x){ return ImagFcnal<T>::eval(x); }

  // Test to check whether value is an integer
  template<typename T, typename domain_tag>
  __HOST_DEVICE__ constexpr IsIntegerFcnal<T, domain_tag>::value_t IsIntegerFcnal<T, domain_tag>::eval(T const& x){
    if constexpr (std_ttraits::is_integral_v<T>) return true;
    using IU_t = convert_to_integral_precision_t<T>;
    return x==__STATIC_CAST__(T, __ENCAPSULATE__(__STATIC_CAST__(IU_t, x)));
  }
  template<typename T>
  __HOST_DEVICE__ IsIntegerFcnal<T, complex_domain_tag>::value_t IsIntegerFcnal<T, complex_domain_tag>::eval(T const& x){ return IsIntegerFcnal::eval(unpack_function_input_reduced<T>::get(x).Re()) && unpack_function_input_reduced<T>::get(x).Im()==Zero<T>(); }
  template<typename T, ENABLE_IF_BOOL_IMPL(!is_pointer_v<T>)> __HOST_DEVICE__ typename IsIntegerFcnal<T>::value_t IsInteger(T const& x){ return IsIntegerFcnal<T>::eval(x); }

  // Test to check whether value is real
  template<typename T, typename domain_tag>
  __HOST_DEVICE__ constexpr IsRealFcnal<T, domain_tag>::value_t IsRealFcnal<T, domain_tag>::eval(T const& x){ return true; }
  template<typename T>
  __HOST_DEVICE__ IsRealFcnal<T, complex_domain_tag>::value_t IsRealFcnal<T, complex_domain_tag>::eval(T const& x){ return unpack_function_input_reduced<T>::get(x).Im()==Zero<fundamental_data_t<T>>(); }
  template<typename T, ENABLE_IF_BOOL_IMPL(!is_pointer_v<T>)> __HOST_DEVICE__ typename IsRealFcnal<T>::value_t IsReal(T const& x){ return IsRealFcnal<T>::eval(x); }

  // Test to check whether value is imaginary
  template<typename T, typename domain_tag>
  __HOST_DEVICE__ constexpr IsImaginaryFcnal<T, domain_tag>::value_t IsImaginaryFcnal<T, domain_tag>::eval(T const& x){ return false; }
  template<typename T>
  __HOST_DEVICE__ IsImaginaryFcnal<T, complex_domain_tag>::value_t IsImaginaryFcnal<T, complex_domain_tag>::eval(T const& x){ return unpack_function_input_reduced<T>::get(x).Re()==Zero<T>() && unpack_function_input_reduced<T>::get(x).Im()!=Zero<T>(); }
  template<typename T, ENABLE_IF_BOOL_IMPL(!is_pointer_v<T>)> __HOST_DEVICE__ typename IsImaginaryFcnal<T>::value_t IsImaginary(T const& x){ return IsImaginaryFcnal<T>::eval(x); }

  // NEGATION
  template<typename T, typename domain_tag>
  __HOST_DEVICE__ constexpr NegateFcnal<T, domain_tag>::value_t NegateFcnal<T, domain_tag>::eval(T const& x){ return -x; }
  template<typename T>
  __HOST_DEVICE__ NegateFcnal<T, real_domain_tag>::value_t NegateFcnal<T, real_domain_tag>::eval(T const& x){ return value_t(-unpack_function_input_reduced<T>::get(x)); }
  template<typename T> template<typename X_t>
  IVY_MATH_GRAPH_QUALIFIER NegateFcnal<T, real_domain_tag>::grad_t NegateFcnal<T, real_domain_tag>::gradient(IvyThreadSafePtr_t<X_t> const& x){
    return make_IvyThreadSafePtr<typename grad_t::element_type>(x.get_memory_type(), x.gpu_stream(), MinusOne<fndtype_t>());
  }
  template<typename T>
  __HOST_DEVICE__ NegateFcnal<T, complex_domain_tag>::value_t NegateFcnal<T, complex_domain_tag>::eval(T const& x){
    auto const& xx = unpack_function_input_reduced<T>::get(x);
    return value_t(-xx.Re(), -xx.Im());
  }
  template<typename T> template<typename X_t>
  IVY_MATH_GRAPH_QUALIFIER NegateFcnal<T, complex_domain_tag>::grad_t NegateFcnal<T, complex_domain_tag>::gradient(IvyThreadSafePtr_t<X_t> const& x){
    return make_IvyThreadSafePtr<typename grad_t::element_type>(x.get_memory_type(), x.gpu_stream(), MinusOne<fndtype_t>());
  }
  template<typename T>
  __HOST_DEVICE__ NegateFcnal<T, quaternion_domain_tag>::value_t NegateFcnal<T, quaternion_domain_tag>::eval(T const& x){
    auto const& xx = unpack_function_input_reduced<T>::get(x);
    return value_t(-xx.W(), -xx.X(), -xx.Y(), -xx.Z());
  }
  template<typename T> template<typename X_t>
  IVY_MATH_GRAPH_QUALIFIER NegateFcnal<T, quaternion_domain_tag>::grad_t NegateFcnal<T, quaternion_domain_tag>::gradient(IvyThreadSafePtr_t<X_t> const& x){
    return make_IvyThreadSafePtr<typename grad_t::element_type>(x.get_memory_type(), x.gpu_stream(), MinusOne<fndtype_t>());
  }
  template<typename T>
  __HOST__ NegateFcnal<T, tensor_domain_tag>::value_t NegateFcnal<T, tensor_domain_tag>::eval(T const& x){
    constexpr std_ivy::IvyMemoryType def_mem_type = IvyMemoryHelpers::get_execution_default_memory();
    value_t res(x);
    for (IvyTensorDim_t i = 0; i < x.num_elements(); ++i){
      if constexpr (is_pointer_v<dtype_t>){
        using inner_t = typename dtype_t::element_type;
        auto const val = -unpack_function_input_reduced<inner_t>::get(*x[i]);
        res[i] = make_IvyThreadSafePtr<inner_t>(def_mem_type, nullptr, val);
      } else {
        res[i] = -unpack_function_input_reduced<dtype_t>::get(x[i]);
      }
    }
    return res;
  }
  template<typename T>
  __HOST__ IvyThreadSafePtr_t<T> NegateFcnal<T, tensor_domain_tag>::gradient(IvyThreadSafePtr_t<T> const& dep){
    constexpr std_ivy::IvyMemoryType def_mem_type = IvyMemoryHelpers::get_execution_default_memory();
    T res(*dep);
    for (IvyTensorDim_t i = 0; i < (*dep).num_elements(); ++i){
      if constexpr (is_pointer_v<dtype_t>){
        using inner_t = typename dtype_t::element_type;
        res[i] = make_IvyThreadSafePtr<inner_t>(def_mem_type, nullptr, typename inner_t::value_t(-1));
      } else {
        res[i] = dtype_t(-1);
      }
    }
    return make_IvyThreadSafePtr<T>(def_mem_type, nullptr, res);
  }
  template<typename T, ENABLE_IF_BOOL_IMPL(!is_pointer_v<T> && !is_tensor_v<T>)> __HOST_DEVICE__ typename NegateFcnal<T>::value_t Negate(T const& x){ return NegateFcnal<T>::eval(x); }
  template<typename T, ENABLE_IF_BOOL_IMPL(!is_pointer_v<T> && is_tensor_v<T>)> __HOST__ typename NegateFcnal<T>::value_t Negate(T const& x){ return NegateFcnal<T>::eval(x); }
  template<typename T, ENABLE_IF_BOOL_IMPL(!is_arithmetic_v<T> && !is_pointer_v<T> && !is_tensor_v<T> && is_ivy_domain_v<T>)> __HOST_DEVICE__ typename NegateFcnal<T>::value_t operator-(T const& x){ return Negate(x); }
  template<typename T, ENABLE_IF_BOOL_IMPL(!is_arithmetic_v<T> && !is_pointer_v<T> && is_tensor_v<T>)> __HOST__ typename NegateFcnal<T>::value_t operator-(T const& x){ return Negate(x); }
  /**
   * @brief Construct a lazy Negate function node for autodiff.
   * @note  Host-only: function-graph objects (IvyRegularFunction) use
   *        virtual dispatch and RAII, which are incompatible with device code.
   *        Direct numerical evaluation (the non-pointer overload) is __HOST_DEVICE__.
   */
  template<typename T, ENABLE_IF_BOOL_IMPL(is_pointer_v<T>)>
  __HOST__ IvyThreadSafePtr_t<typename IvyNegate<typename T::element_type>::base_t> Negate(T const& x){
    constexpr std_ivy::IvyMemoryType def_mem_type = IvyMemoryHelpers::get_execution_default_memory();
    auto res = make_IvyThreadSafePtr<IvyNegate<typename T::element_type>>(def_mem_type, nullptr, IvyNegate(x));
    add_fcn_to_clients(res, x);
    return res;
  }
  /**
   * @brief Construct a lazy function function node for autodiff.
   * @note  Host-only: function-graph objects (IvyRegularFunction) use
   *        virtual dispatch and RAII, which are incompatible with device code.
   *        Direct numerical evaluation (the non-pointer overload) is __HOST_DEVICE__.
   */
  template<typename T, ENABLE_IF_BOOL_IMPL(is_pointer_v<T>)>
  __HOST__ IvyThreadSafePtr_t<typename IvyNegate<typename T::element_type>::base_t> operator-(T const& x){
    return Negate(x);
  }

  // MULTIPLICATIVE INVERSE
  template<typename T, typename domain_tag>
  __HOST_DEVICE__ MultInverseFcnal<T, domain_tag>::value_t MultInverseFcnal<T, domain_tag>::eval(T const& x){ return One<fndtype_t>()/x; }
  template<typename T>
  __HOST_DEVICE__ MultInverseFcnal<T, real_domain_tag>::value_t MultInverseFcnal<T, real_domain_tag>::eval(T const& x){
    return value_t(MultInverse(unpack_function_input_reduced<T>::get(x)));
  }
  template<typename T> template<typename X_t>
  IVY_MATH_GRAPH_QUALIFIER MultInverseFcnal<T, real_domain_tag>::grad_t MultInverseFcnal<T, real_domain_tag>::gradient(IvyThreadSafePtr_t<X_t> const& x){
    return -MultInverse(x*x);
  }
  template<typename T>
  __HOST_DEVICE__ MultInverseFcnal<T, complex_domain_tag>::value_t MultInverseFcnal<T, complex_domain_tag>::eval(T const& x){
    auto const r = unpack_function_input_reduced<T>::get(x).norm();
    auto const phi = unpack_function_input_reduced<T>::get(x).phase();
    value_t res; res.set_absval_phase(MultInverse(r), -phi);
    return res;
  }
  template<typename T> template<typename X_t>
  IVY_MATH_GRAPH_QUALIFIER MultInverseFcnal<T, complex_domain_tag>::grad_t MultInverseFcnal<T, complex_domain_tag>::gradient(IvyThreadSafePtr_t<X_t> const& x){
    return -MultInverse(x*x);
  }
  template<typename T>
  __HOST_DEVICE__ MultInverseFcnal<T, quaternion_domain_tag>::value_t MultInverseFcnal<T, quaternion_domain_tag>::eval(T const& x){
    auto const& q = unpack_function_input_reduced<T>::get(x);
    fndtype_t const n2 = q.norm2();
    fndtype_t const inv = One<fndtype_t>()/n2;
    // q^{-1} = conj(q)/|q|^2
    return value_t(q.W()*inv, -q.X()*inv, -q.Y()*inv, -q.Z()*inv);
  }
  template<typename T> template<typename X_t, typename G_t>
  __HOST__ MultInverseFcnal<T, quaternion_domain_tag>::grad_t MultInverseFcnal<T, quaternion_domain_tag>::combine_gradient(IvyThreadSafePtr_t<X_t> const& dep, G_t const& grad_dep){
    // d(q^{-1}) = -q^{-1} (dq) q^{-1}  (two-sided, non-commutative).
    auto binv = MultInverse(dep);
    return Negate(binv * grad_dep * binv);
  }
  template<typename T>
  __HOST__ MultInverseFcnal<T, tensor_domain_tag>::value_t MultInverseFcnal<T, tensor_domain_tag>::eval(T const& x){
    constexpr std_ivy::IvyMemoryType def_mem_type = IvyMemoryHelpers::get_execution_default_memory();
    value_t res(x);
    for (IvyTensorDim_t i = 0; i < x.num_elements(); ++i){
      if constexpr (is_pointer_v<dtype_t>){
        using inner_t = typename dtype_t::element_type;
        auto const val = unpack_function_input_reduced<inner_t>::get(*x[i]);
        res[i] = make_IvyThreadSafePtr<inner_t>(def_mem_type, nullptr, One<decltype(val)>()/val);
      } else {
        auto const val = unpack_function_input_reduced<dtype_t>::get(x[i]);
        res[i] = One<decltype(val)>()/val;
      }
    }
    return res;
  }
  template<typename T>
  __HOST__ IvyThreadSafePtr_t<T> MultInverseFcnal<T, tensor_domain_tag>::gradient(IvyThreadSafePtr_t<T> const& dep){
    // f(x) = 1/x, f'(x) = -1/x^2
    constexpr std_ivy::IvyMemoryType def_mem_type = IvyMemoryHelpers::get_execution_default_memory();
    T res(*dep);
    for (IvyTensorDim_t i = 0; i < (*dep).num_elements(); ++i){
      if constexpr (is_pointer_v<dtype_t>){
        using inner_t = typename dtype_t::element_type;
        auto const val = unpack_function_input_reduced<inner_t>::get(*(*dep)[i]);
        res[i] = make_IvyThreadSafePtr<inner_t>(def_mem_type, nullptr, MinusOne<decltype(val)>()/(val*val));
      } else {
        auto const val = unpack_function_input_reduced<dtype_t>::get((*dep)[i]);
        res[i] = MinusOne<decltype(val)>()/(val*val);
      }
    }
    return make_IvyThreadSafePtr<T>(def_mem_type, nullptr, res);
  }
  template<typename T, ENABLE_IF_BOOL_IMPL(!is_pointer_v<T> && !is_tensor_v<T>)> __HOST_DEVICE__ typename MultInverseFcnal<T>::value_t MultInverse(T const& x){ return MultInverseFcnal<T>::eval(x); }
  template<typename T, ENABLE_IF_BOOL_IMPL(!is_pointer_v<T> && is_tensor_v<T>)> __HOST__ typename MultInverseFcnal<T>::value_t MultInverse(T const& x){ return MultInverseFcnal<T>::eval(x); }
  /**
   * @brief Construct a lazy MultInverse function node for autodiff.
   * @note  Host-only: function-graph objects (IvyRegularFunction) use
   *        virtual dispatch and RAII, which are incompatible with device code.
   *        Direct numerical evaluation (the non-pointer overload) is __HOST_DEVICE__.
   */
  template<typename T, ENABLE_IF_BOOL_IMPL(is_pointer_v<T>)>
  __HOST__ IvyThreadSafePtr_t<typename IvyMultInverse<typename T::element_type>::base_t> MultInverse(T const& x){
    constexpr std_ivy::IvyMemoryType def_mem_type = IvyMemoryHelpers::get_execution_default_memory();
    auto res = make_IvyThreadSafePtr<IvyMultInverse<typename T::element_type>>(def_mem_type, nullptr, IvyMultInverse(x));
    add_fcn_to_clients(res, x);
    return res;
  }

  // SQUARE ROOT
  template<typename T, typename domain_tag>
  __HOST_DEVICE__ SqrtFcnal<T, domain_tag>::value_t SqrtFcnal<T, domain_tag>::eval(T const& x){ return std_math::sqrt(x); }
  template<typename T>
  __HOST_DEVICE__ SqrtFcnal<T, real_domain_tag>::value_t SqrtFcnal<T, real_domain_tag>::eval(T const& x){ return value_t(SqrtFcnal<dtype_t>::eval(unpack_function_input_reduced<T>::get(x))); }
  template<typename T> template<typename X_t>
  IVY_MATH_GRAPH_QUALIFIER SqrtFcnal<T, real_domain_tag>::grad_t SqrtFcnal<T, real_domain_tag>::gradient(IvyThreadSafePtr_t<X_t> const& x){
    return Pow(x, Scalar<fndtype_t>(x.get_memory_type(), x.gpu_stream(), MinusOneHalf<fndtype_t>()));
  }
  template<typename T>
  __HOST_DEVICE__ SqrtFcnal<T, complex_domain_tag>::value_t SqrtFcnal<T, complex_domain_tag>::eval(T const& x){
    value_t res;
    auto const& re = unpack_function_input_reduced<T>::get(x).Re();
    auto const& im = unpack_function_input_reduced<T>::get(x).Im();
    // |sqrt(z)| = sqrt(|z|) = (re^2 + im^2)^(1/4); arg(sqrt(z)) = arg(z)/2.
    auto R = SqrtFcnal<dtype_t>::eval(SqrtFcnal<dtype_t>::eval(re*re + im*im));
    dtype_t phi = std_math::atan2(im, re);
    res.set_absval_phase(R, phi*OneHalf<dtype_t>());
    return res;
  }
  template<typename T> template<typename X_t>
  IVY_MATH_GRAPH_QUALIFIER SqrtFcnal<T, complex_domain_tag>::grad_t SqrtFcnal<T, complex_domain_tag>::gradient(IvyThreadSafePtr_t<X_t> const& x){
    return Pow(x, Scalar<fndtype_t>(x.get_memory_type(), x.gpu_stream(), MinusOneHalf<fndtype_t>()));
  }
  template<typename T>
  __HOST__ SqrtFcnal<T, tensor_domain_tag>::value_t SqrtFcnal<T, tensor_domain_tag>::eval(T const& x){
    return tensor_apply_elementwise(x, [](auto v){ return Sqrt(v); });
  }
  template<typename T>
  __HOST__ IvyThreadSafePtr_t<T> SqrtFcnal<T, tensor_domain_tag>::gradient(IvyThreadSafePtr_t<T> const& dep){
    // f(x) = sqrt(x), f'(x) = 1/(2*sqrt(x))
    constexpr std_ivy::IvyMemoryType def_mem_type = IvyMemoryHelpers::get_execution_default_memory();
    return make_IvyThreadSafePtr<T>(def_mem_type, nullptr, tensor_apply_elementwise(*dep, [](auto v){ using rt = fundamental_data_t<decltype(v)>; return OneHalf<rt>()/Sqrt(v); }));
  }
  template<typename T, ENABLE_IF_BOOL_IMPL(!is_pointer_v<T> && !is_tensor_v<T>)> __HOST_DEVICE__ typename SqrtFcnal<T>::value_t Sqrt(T const& x){ return SqrtFcnal<T>::eval(x); }
  template<typename T, ENABLE_IF_BOOL_IMPL(!is_pointer_v<T> && is_tensor_v<T>)> __HOST__ typename SqrtFcnal<T>::value_t Sqrt(T const& x){ return SqrtFcnal<T>::eval(x); }
  /**
   * @brief Construct a lazy Sqrt function node for autodiff.
   * @note  Host-only: function-graph objects (IvyRegularFunction) use
   *        virtual dispatch and RAII, which are incompatible with device code.
   *        Direct numerical evaluation (the non-pointer overload) is __HOST_DEVICE__.
   */
  template<typename T, ENABLE_IF_BOOL_IMPL(is_pointer_v<T>)>
  __HOST__ IvyThreadSafePtr_t<typename IvySqrt<typename T::element_type>::base_t> Sqrt(T const& x){
    constexpr std_ivy::IvyMemoryType def_mem_type = IvyMemoryHelpers::get_execution_default_memory();
    auto res = make_IvyThreadSafePtr<IvySqrt<typename T::element_type>>(def_mem_type, nullptr, IvySqrt(x));
    add_fcn_to_clients(res, x);
    return res;
  }

  // ABSOLUTE VALUE
  template<typename T, typename domain_tag>
  __HOST_DEVICE__ AbsFcnal<T, domain_tag>::value_t AbsFcnal<T, domain_tag>::eval(T const& x){ return std_math::abs(x); }
  template<typename T>
  __HOST_DEVICE__ AbsFcnal<T, real_domain_tag>::value_t AbsFcnal<T, real_domain_tag>::eval(T const& x){
    return value_t(Abs(unpack_function_input_reduced<T>::get(x)));
  }
  template<typename T>
  __HOST_DEVICE__ AbsFcnal<T, complex_domain_tag>::value_t AbsFcnal<T, complex_domain_tag>::eval(T const& x){
    return value_t(unpack_function_input_reduced<T>::get(x).norm());
  }
  template<typename T>
  __HOST_DEVICE__ AbsFcnal<T, quaternion_domain_tag>::value_t AbsFcnal<T, quaternion_domain_tag>::eval(T const& x){
    return value_t(unpack_function_input_reduced<T>::get(x).norm());
  }
  template<typename T>
  __HOST__ AbsFcnal<T, tensor_domain_tag>::value_t AbsFcnal<T, tensor_domain_tag>::eval(T const& x){
    return tensor_apply_elementwise(x, [](auto v){ return Abs(v); });
  }
  template<typename T, ENABLE_IF_BOOL_IMPL(!is_pointer_v<T> && !is_tensor_v<T>)> __HOST_DEVICE__ typename AbsFcnal<T>::value_t Abs(T const& x){ return AbsFcnal<T>::eval(x); }
  template<typename T, ENABLE_IF_BOOL_IMPL(!is_pointer_v<T> && is_tensor_v<T>)> __HOST__ typename AbsFcnal<T>::value_t Abs(T const& x){ return AbsFcnal<T>::eval(x); }

  // COMPLEX PHASE
  template<typename T, typename domain_tag>
  __HOST_DEVICE__ constexpr PhaseFcnal<T, domain_tag>::value_t PhaseFcnal<T, domain_tag>::eval(T const& x){
    return value_t(Zero<dtype_t>());
  }
  template<typename T>
  __HOST_DEVICE__ PhaseFcnal<T, complex_domain_tag>::value_t PhaseFcnal<T, complex_domain_tag>::eval(T const& x){
    return value_t(unpack_function_input_reduced<T>::get(x).phase());
  }
  template<typename T, ENABLE_IF_BOOL_IMPL(!is_pointer_v<T>)> __HOST_DEVICE__ typename PhaseFcnal<T>::value_t Phase(T const& x){
    return PhaseFcnal<T>::eval(x);
  }

  // CONJUGATION
  template<typename T, typename domain_tag>
  __HOST_DEVICE__ ConjugateFcnal<T, domain_tag>::value_t ConjugateFcnal<T, domain_tag>::eval(T const& x){
    value_t res(unpack_function_input_reduced<T>::get(x));
    conjugate(res);
    return res;
  }
  template<typename T, ENABLE_IF_BOOL_IMPL(!is_pointer_v<T>)>
  __HOST_DEVICE__ typename ConjugateFcnal<T>::value_t Conjugate(T const& x){
    return ConjugateFcnal<T>::eval(x);
  }

  // EXPONENTIAL
  template<typename T, typename domain_tag>
  __HOST_DEVICE__ ExpFcnal<T, domain_tag>::value_t ExpFcnal<T, domain_tag>::eval(T const& x){ return std_math::exp(x); }
  template<typename T>
  __HOST_DEVICE__ ExpFcnal<T, real_domain_tag>::value_t ExpFcnal<T, real_domain_tag>::eval(T const& x){ return value_t(Exp(unpack_function_input_reduced<T>::get(x))); }
  template<typename T> template<typename X_t>
  IVY_MATH_GRAPH_QUALIFIER ExpFcnal<T, real_domain_tag>::grad_t ExpFcnal<T, real_domain_tag>::gradient(IvyThreadSafePtr_t<X_t> const& x){
    return Exp(x);
  }
  template<typename T>
  __HOST_DEVICE__ ExpFcnal<T, complex_domain_tag>::value_t ExpFcnal<T, complex_domain_tag>::eval(T const& x){
    auto const r = Exp(unpack_function_input_reduced<T>::get(x).Re());
    auto const& im = unpack_function_input_reduced<T>::get(x).Im();
    return value_t(r*Cos(im), r*Sin(im));
  }
  template<typename T> template<typename X_t>
  IVY_MATH_GRAPH_QUALIFIER ExpFcnal<T, complex_domain_tag>::grad_t ExpFcnal<T, complex_domain_tag>::gradient(IvyThreadSafePtr_t<X_t> const& x){
    return Exp(x);
  }
  template<typename T>
  __HOST__ ExpFcnal<T, tensor_domain_tag>::value_t ExpFcnal<T, tensor_domain_tag>::eval(T const& x){
    return tensor_apply_elementwise(x, [](auto v){ return Exp(v); });
  }
  template<typename T>
  __HOST__ IvyThreadSafePtr_t<T> ExpFcnal<T, tensor_domain_tag>::gradient(IvyThreadSafePtr_t<T> const& dep){
    // f(x) = exp(x), f'(x) = exp(x)
    constexpr std_ivy::IvyMemoryType def_mem_type = IvyMemoryHelpers::get_execution_default_memory();
    return make_IvyThreadSafePtr<T>(def_mem_type, nullptr, tensor_apply_elementwise(*dep, [](auto v){ return Exp(v); }));
  }
  template<typename T, ENABLE_IF_BOOL_IMPL(!is_pointer_v<T> && !is_tensor_v<T>)> __HOST_DEVICE__ typename ExpFcnal<T>::value_t Exp(T const& x){ return ExpFcnal<T>::eval(x); }
  template<typename T, ENABLE_IF_BOOL_IMPL(!is_pointer_v<T> && is_tensor_v<T>)> __HOST__ typename ExpFcnal<T>::value_t Exp(T const& x){ return ExpFcnal<T>::eval(x); }
  /**
   * @brief Construct a lazy Exp function node for autodiff.
   * @note  Host-only: function-graph objects (IvyRegularFunction) use
   *        virtual dispatch and RAII, which are incompatible with device code.
   *        Direct numerical evaluation (the non-pointer overload) is __HOST_DEVICE__.
   */
  template<typename T, ENABLE_IF_BOOL_IMPL(is_pointer_v<T>)>
  __HOST__ IvyThreadSafePtr_t<typename IvyExp<typename T::element_type>::base_t> Exp(T const& x){
    constexpr std_ivy::IvyMemoryType def_mem_type = IvyMemoryHelpers::get_execution_default_memory();
    auto res = make_IvyThreadSafePtr<IvyExp<typename T::element_type>>(def_mem_type, nullptr, IvyExp(x));
    add_fcn_to_clients(res, x);
    return res;
  }

  // SUM (tensor -> scalar reduction)
  template<typename T>
  __HOST__ typename SumFcnal<T>::value_t SumFcnal<T>::eval(T const& x){
    dtype_t acc = dtype_t(0);
    for (IvyTensorDim_t i = 0; i < x.num_elements(); ++i){
      if constexpr (is_pointer_v<elem_t>) acc += unpack_function_input_reduced<typename elem_t::element_type>::get(*x[i]);
      else acc += unpack_function_input_reduced<elem_t>::get(x[i]);
    }
    return value_t(acc);
  }
  template<typename T, ENABLE_IF_BOOL_IMPL(!is_pointer_v<T> && is_tensor_v<T>)>
  __HOST__ typename SumFcnal<T>::value_t Sum(T const& x){ return SumFcnal<T>::eval(x); }
  template<typename T, ENABLE_IF_BOOL_IMPL(is_pointer_v<T>)>
  __HOST__ IvyThreadSafePtr_t<typename IvySum<typename T::element_type>::base_t> Sum(T const& x){
    constexpr std_ivy::IvyMemoryType def_mem_type = IvyMemoryHelpers::get_execution_default_memory();
    auto res = make_IvyThreadSafePtr<IvySum<typename T::element_type>>(def_mem_type, nullptr, IvySum(x));
    add_fcn_to_clients(res, x);
    return res;
  }

  // LOG (NATURAL LOG)
  template<typename T, typename domain_tag>
  __HOST_DEVICE__ LogFcnal<T, domain_tag>::value_t LogFcnal<T, domain_tag>::eval(T const& x){ return std_math::log(x); }
  template<typename T>
  __HOST_DEVICE__ LogFcnal<T, real_domain_tag>::value_t LogFcnal<T, real_domain_tag>::eval(T const& x){ return value_t(Log(unpack_function_input_reduced<T>::get(x))); }
  template<typename T> template<typename X_t>
  IVY_MATH_GRAPH_QUALIFIER LogFcnal<T, real_domain_tag>::grad_t LogFcnal<T, real_domain_tag>::gradient(IvyThreadSafePtr_t<X_t> const& x){
    return MultInverse(x);
  }
  template<typename T>
  __HOST_DEVICE__ LogFcnal<T, complex_domain_tag>::value_t LogFcnal<T, complex_domain_tag>::eval(T const& x){
    auto const r = unpack_function_input_reduced<T>::get(x).norm();
    auto const phi = unpack_function_input_reduced<T>::get(x).phase();
    return value_t(Log(r), phi);
  }
  template<typename T> template<typename X_t>
  IVY_MATH_GRAPH_QUALIFIER LogFcnal<T, complex_domain_tag>::grad_t LogFcnal<T, complex_domain_tag>::gradient(IvyThreadSafePtr_t<X_t> const& x){
    return MultInverse(x);
  }
  template<typename T>
  __HOST__ LogFcnal<T, tensor_domain_tag>::value_t LogFcnal<T, tensor_domain_tag>::eval(T const& x){
    return tensor_apply_elementwise(x, [](auto v){ return Log(v); });
  }
  template<typename T>
  __HOST__ IvyThreadSafePtr_t<T> LogFcnal<T, tensor_domain_tag>::gradient(IvyThreadSafePtr_t<T> const& dep){
    // f(x) = log(x), f'(x) = 1/x
    constexpr std_ivy::IvyMemoryType def_mem_type = IvyMemoryHelpers::get_execution_default_memory();
    return make_IvyThreadSafePtr<T>(def_mem_type, nullptr, tensor_apply_elementwise(*dep, [](auto v){ return MultInverse(v); }));
  }
  template<typename T, ENABLE_IF_BOOL_IMPL(!is_pointer_v<T> && !is_tensor_v<T>)> __HOST_DEVICE__ typename LogFcnal<T>::value_t Log(T const& x){ return LogFcnal<T>::eval(x); }
  template<typename T, ENABLE_IF_BOOL_IMPL(!is_pointer_v<T> && is_tensor_v<T>)> __HOST__ typename LogFcnal<T>::value_t Log(T const& x){ return LogFcnal<T>::eval(x); }
  /**
   * @brief Construct a lazy Log function node for autodiff.
   * @note  Host-only: function-graph objects (IvyRegularFunction) use
   *        virtual dispatch and RAII, which are incompatible with device code.
   *        Direct numerical evaluation (the non-pointer overload) is __HOST_DEVICE__.
   */
  template<typename T, ENABLE_IF_BOOL_IMPL(is_pointer_v<T>)>
  __HOST__ IvyThreadSafePtr_t<typename IvyLog<typename T::element_type>::base_t> Log(T const& x){
    constexpr std_ivy::IvyMemoryType def_mem_type = IvyMemoryHelpers::get_execution_default_memory();
    auto res = make_IvyThreadSafePtr<IvyLog<typename T::element_type>>(def_mem_type, nullptr, IvyLog(x));
    add_fcn_to_clients(res, x);
    return res;
  }

  // LOG10 (BASE=10 LOG)
  template<typename T, typename domain_tag>
  __HOST_DEVICE__ Log10Fcnal<T, domain_tag>::value_t Log10Fcnal<T, domain_tag>::eval(T const& x){ return Log(x)/LogTen<value_t>(); }
  template<typename T>
  __HOST_DEVICE__ Log10Fcnal<T, real_domain_tag>::value_t Log10Fcnal<T, real_domain_tag>::eval(T const& x){ return value_t(Log10(unpack_function_input_reduced<T>::get(x))); }
  template<typename T> template<typename X_t>
  IVY_MATH_GRAPH_QUALIFIER Log10Fcnal<T, real_domain_tag>::grad_t Log10Fcnal<T, real_domain_tag>::gradient(IvyThreadSafePtr_t<X_t> const& x){
    return MultInverse(x) / Scalar<fndtype_t>(x.get_memory_type(), x.gpu_stream(), LogTen<fndtype_t>());
  }
  template<typename T>
  __HOST_DEVICE__ Log10Fcnal<T, complex_domain_tag>::value_t Log10Fcnal<T, complex_domain_tag>::eval(T const& x){
    value_t res = Log(unpack_function_input_reduced<T>::get(x));
    return res/LogTen<dtype_t>();
  }
  template<typename T> template<typename X_t>
  IVY_MATH_GRAPH_QUALIFIER Log10Fcnal<T, complex_domain_tag>::grad_t Log10Fcnal<T, complex_domain_tag>::gradient(IvyThreadSafePtr_t<X_t> const& x){
    return MultInverse(x) / Scalar<fndtype_t>(x.get_memory_type(), x.gpu_stream(), LogTen<fndtype_t>());
  }
  template<typename T, ENABLE_IF_BOOL_IMPL(!is_pointer_v<T>)> __HOST_DEVICE__ typename Log10Fcnal<T>::value_t Log10(T const& x){ return Log10Fcnal<T>::eval(x); }
  /**
   * @brief Construct a lazy Log10 function node for autodiff.
   * @note  Host-only: function-graph objects (IvyRegularFunction) use
   *        virtual dispatch and RAII, which are incompatible with device code.
   *        Direct numerical evaluation (the non-pointer overload) is __HOST_DEVICE__.
   */
  template<typename T, ENABLE_IF_BOOL_IMPL(is_pointer_v<T>)>
  __HOST__ IvyThreadSafePtr_t<typename IvyLog10<typename T::element_type>::base_t> Log10(T const& x){
    constexpr std_ivy::IvyMemoryType def_mem_type = IvyMemoryHelpers::get_execution_default_memory();
    auto res = make_IvyThreadSafePtr<IvyLog10<typename T::element_type>>(def_mem_type, nullptr, IvyLog10(x));
    add_fcn_to_clients(res, x);
    return res;
  }

  // SINE
  template<typename T, typename domain_tag>
  __HOST_DEVICE__ SinFcnal<T, domain_tag>::value_t SinFcnal<T, domain_tag>::eval(T const& x){ return std_math::sin(x); }
  template<typename T>
  __HOST_DEVICE__ SinFcnal<T, real_domain_tag>::value_t SinFcnal<T, real_domain_tag>::eval(T const& x){ return value_t(Sin(unpack_function_input_reduced<T>::get(x))); }
  template<typename T> template<typename X_t>
  IVY_MATH_GRAPH_QUALIFIER SinFcnal<T, real_domain_tag>::grad_t SinFcnal<T, real_domain_tag>::gradient(IvyThreadSafePtr_t<X_t> const& x){ return Cos(x); }
  template<typename T>
  __HOST_DEVICE__ SinFcnal<T, complex_domain_tag>::value_t SinFcnal<T, complex_domain_tag>::eval(T const& x){
    auto const& a = unpack_function_input_reduced<T>::get(x).Re();
    auto const& b = unpack_function_input_reduced<T>::get(x).Im();
    value_t arg(-b, a);
    auto const exp_arg = Exp(arg);
    return (exp_arg - MultInverse(exp_arg))/value_t(Zero<fndtype_t>(), Two<fndtype_t>());
  }
  template<typename T> template<typename X_t>
  IVY_MATH_GRAPH_QUALIFIER SinFcnal<T, complex_domain_tag>::grad_t SinFcnal<T, complex_domain_tag>::gradient(IvyThreadSafePtr_t<X_t> const& x){ return Cos(x); }
  template<typename T>
  __HOST__ SinFcnal<T, tensor_domain_tag>::value_t SinFcnal<T, tensor_domain_tag>::eval(T const& x){
    return tensor_apply_elementwise(x, [](auto v){ return Sin(v); });
  }
  template<typename T>
  __HOST__ IvyThreadSafePtr_t<T> SinFcnal<T, tensor_domain_tag>::gradient(IvyThreadSafePtr_t<T> const& dep){
    // f(x) = sin(x), f'(x) = cos(x)
    constexpr std_ivy::IvyMemoryType def_mem_type = IvyMemoryHelpers::get_execution_default_memory();
    return make_IvyThreadSafePtr<T>(def_mem_type, nullptr, tensor_apply_elementwise(*dep, [](auto v){ return Cos(v); }));
  }
  template<typename T, ENABLE_IF_BOOL_IMPL(!is_pointer_v<T> && !is_tensor_v<T>)> __HOST_DEVICE__ typename SinFcnal<T>::value_t Sin(T const& x){ return SinFcnal<T>::eval(x); }
  template<typename T, ENABLE_IF_BOOL_IMPL(!is_pointer_v<T> && is_tensor_v<T>)> __HOST__ typename SinFcnal<T>::value_t Sin(T const& x){ return SinFcnal<T>::eval(x); }
  /**
   * @brief Construct a lazy Sin function node for autodiff.
   * @note  Host-only: function-graph objects (IvyRegularFunction) use
   *        virtual dispatch and RAII, which are incompatible with device code.
   *        Direct numerical evaluation (the non-pointer overload) is __HOST_DEVICE__.
   */
  template<typename T, ENABLE_IF_BOOL_IMPL(is_pointer_v<T>)>
  __HOST__ IvyThreadSafePtr_t<typename IvySin<typename T::element_type>::base_t> Sin(T const& x){
    constexpr std_ivy::IvyMemoryType def_mem_type = IvyMemoryHelpers::get_execution_default_memory();
    auto res = make_IvyThreadSafePtr<IvySin<typename T::element_type>>(def_mem_type, nullptr, IvySin(x));
    add_fcn_to_clients(res, x);
    return res;
  }

  // COSINE
  template<typename T, typename domain_tag>
  __HOST_DEVICE__ CosFcnal<T, domain_tag>::value_t CosFcnal<T, domain_tag>::eval(T const& x){ return std_math::cos(x); }
  template<typename T>
  __HOST_DEVICE__ CosFcnal<T, real_domain_tag>::value_t CosFcnal<T, real_domain_tag>::eval(T const& x){ return value_t(Cos(unpack_function_input_reduced<T>::get(x))); }
  template<typename T> template<typename X_t>
  IVY_MATH_GRAPH_QUALIFIER CosFcnal<T, real_domain_tag>::grad_t CosFcnal<T, real_domain_tag>::gradient(IvyThreadSafePtr_t<X_t> const& x){ return -Sin(x); }
  template<typename T>
  __HOST_DEVICE__ CosFcnal<T, complex_domain_tag>::value_t CosFcnal<T, complex_domain_tag>::eval(T const& x){
    auto const& a = unpack_function_input_reduced<T>::get(x).Re();
    auto const& b = unpack_function_input_reduced<T>::get(x).Im();
    value_t arg(-b, a);
    auto const exp_arg = Exp(arg);
    return (exp_arg + MultInverse(exp_arg))/Two<fndtype_t>();
  }
  template<typename T> template<typename X_t>
  IVY_MATH_GRAPH_QUALIFIER CosFcnal<T, complex_domain_tag>::grad_t CosFcnal<T, complex_domain_tag>::gradient(IvyThreadSafePtr_t<X_t> const& x){ return -Sin(x); }
  template<typename T>
  __HOST__ CosFcnal<T, tensor_domain_tag>::value_t CosFcnal<T, tensor_domain_tag>::eval(T const& x){
    return tensor_apply_elementwise(x, [](auto v){ return Cos(v); });
  }
  template<typename T>
  __HOST__ IvyThreadSafePtr_t<T> CosFcnal<T, tensor_domain_tag>::gradient(IvyThreadSafePtr_t<T> const& dep){
    // f(x) = cos(x), f'(x) = -sin(x)
    constexpr std_ivy::IvyMemoryType def_mem_type = IvyMemoryHelpers::get_execution_default_memory();
    return make_IvyThreadSafePtr<T>(def_mem_type, nullptr, tensor_apply_elementwise(*dep, [](auto v){ return -Sin(v); }));
  }
  template<typename T, ENABLE_IF_BOOL_IMPL(!is_pointer_v<T> && !is_tensor_v<T>)> __HOST_DEVICE__ typename CosFcnal<T>::value_t Cos(T const& x){ return CosFcnal<T>::eval(x); }
  template<typename T, ENABLE_IF_BOOL_IMPL(!is_pointer_v<T> && is_tensor_v<T>)> __HOST__ typename CosFcnal<T>::value_t Cos(T const& x){ return CosFcnal<T>::eval(x); }
  /**
   * @brief Construct a lazy Cos function node for autodiff.
   * @note  Host-only: function-graph objects (IvyRegularFunction) use
   *        virtual dispatch and RAII, which are incompatible with device code.
   *        Direct numerical evaluation (the non-pointer overload) is __HOST_DEVICE__.
   */
  template<typename T, ENABLE_IF_BOOL_IMPL(is_pointer_v<T>)>
  __HOST__ IvyThreadSafePtr_t<typename IvyCos<typename T::element_type>::base_t> Cos(T const& x){
    constexpr std_ivy::IvyMemoryType def_mem_type = IvyMemoryHelpers::get_execution_default_memory();
    auto res = make_IvyThreadSafePtr<IvyCos<typename T::element_type>>(def_mem_type, nullptr, IvyCos(x));
    add_fcn_to_clients(res, x);
    return res;
  }

  // TANGENT
  template<typename T, typename domain_tag>
  __HOST_DEVICE__ TanFcnal<T, domain_tag>::value_t TanFcnal<T, domain_tag>::eval(T const& x){ return Sin(x)/Cos(x); }
  template<typename T, typename domain_tag> template<typename X_t>
  IVY_MATH_GRAPH_QUALIFIER TanFcnal<T, domain_tag>::grad_t TanFcnal<T, domain_tag>::gradient(IvyThreadSafePtr_t<X_t> const& x){ auto r = Sec(x); return r*r; }
  template<typename T>
  __HOST__ TanFcnal<T, tensor_domain_tag>::value_t TanFcnal<T, tensor_domain_tag>::eval(T const& x){
    return tensor_apply_elementwise(x, [](auto v){ return Tan(v); });
  }
  template<typename T>
  __HOST__ IvyThreadSafePtr_t<T> TanFcnal<T, tensor_domain_tag>::gradient(IvyThreadSafePtr_t<T> const& dep){
    // f(x) = tan(x), f'(x) = sec^2(x) = 1/cos^2(x)
    constexpr std_ivy::IvyMemoryType def_mem_type = IvyMemoryHelpers::get_execution_default_memory();
    return make_IvyThreadSafePtr<T>(def_mem_type, nullptr, tensor_apply_elementwise(*dep, [](auto v){ auto const c = Cos(v); return MultInverse(c*c); }));
  }
  template<typename T, ENABLE_IF_BOOL_IMPL(!is_pointer_v<T> && !is_tensor_v<T>)> __HOST_DEVICE__ typename TanFcnal<T>::value_t Tan(T const& x){ return TanFcnal<T>::eval(x); }
  template<typename T, ENABLE_IF_BOOL_IMPL(!is_pointer_v<T> && is_tensor_v<T>)> __HOST__ typename TanFcnal<T>::value_t Tan(T const& x){ return TanFcnal<T>::eval(x); }
  /**
   * @brief Construct a lazy Tan function node for autodiff.
   * @note  Host-only: function-graph objects (IvyRegularFunction) use
   *        virtual dispatch and RAII, which are incompatible with device code.
   *        Direct numerical evaluation (the non-pointer overload) is __HOST_DEVICE__.
   */
  template<typename T, ENABLE_IF_BOOL_IMPL(is_pointer_v<T>)>
  __HOST__ IvyThreadSafePtr_t<typename IvyTan<typename T::element_type>::base_t> Tan(T const& x){
    constexpr std_ivy::IvyMemoryType def_mem_type = IvyMemoryHelpers::get_execution_default_memory();
    auto res = make_IvyThreadSafePtr<IvyTan<typename T::element_type>>(def_mem_type, nullptr, IvyTan(x));
    add_fcn_to_clients(res, x);
    return res;
  }

  // SECANT
  template<typename T, typename domain_tag>
  __HOST_DEVICE__ SecFcnal<T, domain_tag>::value_t SecFcnal<T, domain_tag>::eval(T const& x){ return MultInverse(Cos(x)); }
  template<typename T, typename domain_tag> template<typename X_t>
  IVY_MATH_GRAPH_QUALIFIER SecFcnal<T, domain_tag>::grad_t SecFcnal<T, domain_tag>::gradient(IvyThreadSafePtr_t<X_t> const& x){ return Sec(x)*Tan(x); }
  template<typename T, ENABLE_IF_BOOL_IMPL(!is_pointer_v<T>)> __HOST_DEVICE__ typename SecFcnal<T>::value_t Sec(T const& x){ return SecFcnal<T>::eval(x); }
  /**
   * @brief Construct a lazy Sec function node for autodiff.
   * @note  Host-only: function-graph objects (IvyRegularFunction) use
   *        virtual dispatch and RAII, which are incompatible with device code.
   *        Direct numerical evaluation (the non-pointer overload) is __HOST_DEVICE__.
   */
  template<typename T, ENABLE_IF_BOOL_IMPL(is_pointer_v<T>)>
  __HOST__ IvyThreadSafePtr_t<typename IvySec<typename T::element_type>::base_t> Sec(T const& x){
    constexpr std_ivy::IvyMemoryType def_mem_type = IvyMemoryHelpers::get_execution_default_memory();
    auto res = make_IvyThreadSafePtr<IvySec<typename T::element_type>>(def_mem_type, nullptr, IvySec(x));
    add_fcn_to_clients(res, x);
    return res;
  }

  // COSECANT
  template<typename T, typename domain_tag>
  __HOST_DEVICE__ CscFcnal<T, domain_tag>::value_t CscFcnal<T, domain_tag>::eval(T const& x){ return MultInverse(Sin(x)); }
  template<typename T, typename domain_tag> template<typename X_t>
  IVY_MATH_GRAPH_QUALIFIER CscFcnal<T, domain_tag>::grad_t CscFcnal<T, domain_tag>::gradient(IvyThreadSafePtr_t<X_t> const& x){ return -Csc(x)*Cot(x);; }
  template<typename T, ENABLE_IF_BOOL_IMPL(!is_pointer_v<T>)> __HOST_DEVICE__ typename CscFcnal<T>::value_t Csc(T const& x){ return CscFcnal<T>::eval(x); }
  /**
   * @brief Construct a lazy Csc function node for autodiff.
   * @note  Host-only: function-graph objects (IvyRegularFunction) use
   *        virtual dispatch and RAII, which are incompatible with device code.
   *        Direct numerical evaluation (the non-pointer overload) is __HOST_DEVICE__.
   */
  template<typename T, ENABLE_IF_BOOL_IMPL(is_pointer_v<T>)>
  __HOST__ IvyThreadSafePtr_t<typename IvyCsc<typename T::element_type>::base_t> Csc(T const& x){
    constexpr std_ivy::IvyMemoryType def_mem_type = IvyMemoryHelpers::get_execution_default_memory();
    auto res = make_IvyThreadSafePtr<IvyCsc<typename T::element_type>>(def_mem_type, nullptr, IvyCsc(x));
    add_fcn_to_clients(res, x);
    return res;
  }

  // COTANGENT
  template<typename T, typename domain_tag>
  __HOST_DEVICE__ CotFcnal<T, domain_tag>::value_t CotFcnal<T, domain_tag>::eval(T const& x){ return Cos(x)/Sin(x); }
  template<typename T, typename domain_tag> template<typename X_t>
  IVY_MATH_GRAPH_QUALIFIER CotFcnal<T, domain_tag>::grad_t CotFcnal<T, domain_tag>::gradient(IvyThreadSafePtr_t<X_t> const& x){ auto r = Csc(x); return -r*r; }
  template<typename T>
  __HOST__ CotFcnal<T, tensor_domain_tag>::value_t CotFcnal<T, tensor_domain_tag>::eval(T const& x){
    // f(x) = cot(x) = cos(x)/sin(x)
    return tensor_apply_elementwise(x, [](auto v){ return Cot(v); });
  }
  template<typename T>
  __HOST__ IvyThreadSafePtr_t<T> CotFcnal<T, tensor_domain_tag>::gradient(IvyThreadSafePtr_t<T> const& dep){
    // f'(x) = -csc^2(x) = -1/sin^2(x)
    constexpr std_ivy::IvyMemoryType def_mem_type = IvyMemoryHelpers::get_execution_default_memory();
    return make_IvyThreadSafePtr<T>(def_mem_type, nullptr, tensor_apply_elementwise(*dep, [](auto v){ auto const s = Sin(v); return -MultInverse(s*s); }));
  }
  template<typename T, ENABLE_IF_BOOL_IMPL(!is_pointer_v<T> && !is_tensor_v<T>)> __HOST_DEVICE__ typename CotFcnal<T>::value_t Cot(T const& x){ return CotFcnal<T>::eval(x); }
  template<typename T, ENABLE_IF_BOOL_IMPL(!is_pointer_v<T> && is_tensor_v<T>)> __HOST__ typename CotFcnal<T>::value_t Cot(T const& x){ return CotFcnal<T>::eval(x); }
  /**
   * @brief Construct a lazy Cot function node for autodiff.
   * @note  Host-only: function-graph objects (IvyRegularFunction) use
   *        virtual dispatch and RAII, which are incompatible with device code.
   *        Direct numerical evaluation (the non-pointer overload) is __HOST_DEVICE__.
   */
  template<typename T, ENABLE_IF_BOOL_IMPL(is_pointer_v<T>)>
  __HOST__ IvyThreadSafePtr_t<typename IvyCot<typename T::element_type>::base_t> Cot(T const& x){
    constexpr std_ivy::IvyMemoryType def_mem_type = IvyMemoryHelpers::get_execution_default_memory();
    auto res = make_IvyThreadSafePtr<IvyCot<typename T::element_type>>(def_mem_type, nullptr, IvyCot(x));
    add_fcn_to_clients(res, x);
    return res;
  }

  // SINH
  template<typename T, typename domain_tag>
  __HOST_DEVICE__ SinHFcnal<T, domain_tag>::value_t SinHFcnal<T, domain_tag>::eval(T const& x){
    auto ex = Exp(x);
    return (ex - MultInverse(ex))/Two<dtype_t>();
  }
  template<typename T>
  __HOST_DEVICE__ SinHFcnal<T, real_domain_tag>::value_t SinHFcnal<T, real_domain_tag>::eval(T const& x){ return value_t(SinH(unpack_function_input_reduced<T>::get(x))); }
  template<typename T> template<typename X_t>
  IVY_MATH_GRAPH_QUALIFIER SinHFcnal<T, real_domain_tag>::grad_t SinHFcnal<T, real_domain_tag>::gradient(IvyThreadSafePtr_t<X_t> const& x){ return CosH(x); }
  template<typename T>
  __HOST_DEVICE__ SinHFcnal<T, complex_domain_tag>::value_t SinHFcnal<T, complex_domain_tag>::eval(T const& x){
    auto const& a = unpack_function_input_reduced<T>::get(x).Re();
    auto const& b = unpack_function_input_reduced<T>::get(x).Im();
    auto sha = SinH(a);
    auto cb = Cos(b);
    auto cha = CosH(a);
    auto sb = Sin(b);
    return value_t(sha*cb, cha*sb);
  }
  template<typename T> template<typename X_t>
  IVY_MATH_GRAPH_QUALIFIER SinHFcnal<T, complex_domain_tag>::grad_t SinHFcnal<T, complex_domain_tag>::gradient(IvyThreadSafePtr_t<X_t> const& x){ return CosH(x); }
  template<typename T>
  __HOST__ SinHFcnal<T, tensor_domain_tag>::value_t SinHFcnal<T, tensor_domain_tag>::eval(T const& x){
    // f(x) = sinh(x)
    return tensor_apply_elementwise(x, [](auto v){ return SinH(v); });
  }
  template<typename T>
  __HOST__ IvyThreadSafePtr_t<T> SinHFcnal<T, tensor_domain_tag>::gradient(IvyThreadSafePtr_t<T> const& dep){
    // f'(x) = cosh(x)
    constexpr std_ivy::IvyMemoryType def_mem_type = IvyMemoryHelpers::get_execution_default_memory();
    return make_IvyThreadSafePtr<T>(def_mem_type, nullptr, tensor_apply_elementwise(*dep, [](auto v){ return CosH(v); }));
  }
  template<typename T, ENABLE_IF_BOOL_IMPL(!is_pointer_v<T> && !is_tensor_v<T>)> __HOST_DEVICE__ typename SinHFcnal<T>::value_t SinH(T const& x){ return SinHFcnal<T>::eval(x); }
  template<typename T, ENABLE_IF_BOOL_IMPL(!is_pointer_v<T> && is_tensor_v<T>)> __HOST__ typename SinHFcnal<T>::value_t SinH(T const& x){ return SinHFcnal<T>::eval(x); }
  /**
   * @brief Construct a lazy SinH function node for autodiff.
   * @note  Host-only: function-graph objects (IvyRegularFunction) use
   *        virtual dispatch and RAII, which are incompatible with device code.
   *        Direct numerical evaluation (the non-pointer overload) is __HOST_DEVICE__.
   */
  template<typename T, ENABLE_IF_BOOL_IMPL(is_pointer_v<T>)>
  __HOST__ IvyThreadSafePtr_t<typename IvySinH<typename T::element_type>::base_t> SinH(T const& x){
    constexpr std_ivy::IvyMemoryType def_mem_type = IvyMemoryHelpers::get_execution_default_memory();
    auto res = make_IvyThreadSafePtr<IvySinH<typename T::element_type>>(def_mem_type, nullptr, IvySinH(x));
    add_fcn_to_clients(res, x);
    return res;
  }

  // COSH
  template<typename T, typename domain_tag>
  __HOST_DEVICE__ CosHFcnal<T, domain_tag>::value_t CosHFcnal<T, domain_tag>::eval(T const& x){
    auto ex = Exp(x);
    return (ex + MultInverse(ex))/Two<dtype_t>();
  }
  template<typename T>
  __HOST_DEVICE__ CosHFcnal<T, real_domain_tag>::value_t CosHFcnal<T, real_domain_tag>::eval(T const& x){ return value_t(CosH(unpack_function_input_reduced<T>::get(x))); }
  template<typename T> template<typename X_t>
  IVY_MATH_GRAPH_QUALIFIER CosHFcnal<T, real_domain_tag>::grad_t CosHFcnal<T, real_domain_tag>::gradient(IvyThreadSafePtr_t<X_t> const& x){ return SinH(x); }
  template<typename T>
  __HOST_DEVICE__ CosHFcnal<T, complex_domain_tag>::value_t CosHFcnal<T, complex_domain_tag>::eval(T const& x){
    auto const& a = unpack_function_input_reduced<T>::get(x).Re();
    auto const& b = unpack_function_input_reduced<T>::get(x).Im();
    auto cha = CosH(a);
    auto cb = Cos(b);
    auto sha = SinH(a);
    auto sb = Sin(b);
    return value_t(cha*cb, sha*sb);
  }
  template<typename T> template<typename X_t>
  IVY_MATH_GRAPH_QUALIFIER CosHFcnal<T, complex_domain_tag>::grad_t CosHFcnal<T, complex_domain_tag>::gradient(IvyThreadSafePtr_t<X_t> const& x){
    return SinH(x);
  }
  template<typename T>
  __HOST__ CosHFcnal<T, tensor_domain_tag>::value_t CosHFcnal<T, tensor_domain_tag>::eval(T const& x){
    // f(x) = cosh(x)
    return tensor_apply_elementwise(x, [](auto v){ return CosH(v); });
  }
  template<typename T>
  __HOST__ IvyThreadSafePtr_t<T> CosHFcnal<T, tensor_domain_tag>::gradient(IvyThreadSafePtr_t<T> const& dep){
    // f'(x) = sinh(x)
    constexpr std_ivy::IvyMemoryType def_mem_type = IvyMemoryHelpers::get_execution_default_memory();
    return make_IvyThreadSafePtr<T>(def_mem_type, nullptr, tensor_apply_elementwise(*dep, [](auto v){ return SinH(v); }));
  }
  template<typename T, ENABLE_IF_BOOL_IMPL(!is_pointer_v<T> && !is_tensor_v<T>)> __HOST_DEVICE__ typename CosHFcnal<T>::value_t CosH(T const& x){ return CosHFcnal<T>::eval(x); }
  template<typename T, ENABLE_IF_BOOL_IMPL(!is_pointer_v<T> && is_tensor_v<T>)> __HOST__ typename CosHFcnal<T>::value_t CosH(T const& x){ return CosHFcnal<T>::eval(x); }
  /**
   * @brief Construct a lazy CosH function node for autodiff.
   * @note  Host-only: function-graph objects (IvyRegularFunction) use
   *        virtual dispatch and RAII, which are incompatible with device code.
   *        Direct numerical evaluation (the non-pointer overload) is __HOST_DEVICE__.
   */
  template<typename T, ENABLE_IF_BOOL_IMPL(is_pointer_v<T>)>
  __HOST__ IvyThreadSafePtr_t<typename IvyCosH<typename T::element_type>::base_t> CosH(T const& x){
    constexpr std_ivy::IvyMemoryType def_mem_type = IvyMemoryHelpers::get_execution_default_memory();
    auto res = make_IvyThreadSafePtr<IvyCosH<typename T::element_type>>(def_mem_type, nullptr, IvyCosH(x));
    add_fcn_to_clients(res, x);
    return res;
  }

  // ERF
  template<typename T, typename domain_tag>
  __HOST_DEVICE__ ErfFcnal<T, domain_tag>::value_t ErfFcnal<T, domain_tag>::eval(T const& x){
    return std_math::erf(x);
  }
  template<typename T>
  __HOST_DEVICE__ ErfFcnal<T, real_domain_tag>::value_t ErfFcnal<T, real_domain_tag>::eval(T const& x){
    return value_t(Erf(unpack_function_input_reduced<T>::get(x)));
  }
  template<typename T> template<typename X_t>
  IVY_MATH_GRAPH_QUALIFIER ErfFcnal<T, real_domain_tag>::grad_t ErfFcnal<T, real_domain_tag>::gradient(IvyThreadSafePtr_t<X_t> const& x){
    return Exp(-x*x)*Scalar<fndtype_t>(x.get_memory_type(), x.gpu_stream(), TwoOverSqrtPi<fndtype_t>());
  }
  template<typename T>
  __HOST_DEVICE__ ErfFcnal<T, complex_domain_tag>::value_t ErfFcnal<T, complex_domain_tag>::eval(T const& x){
    return IvyCerf::erf(unpack_function_input_reduced<T>::get(x));
  }
  template<typename T> template<typename X_t>
  IVY_MATH_GRAPH_QUALIFIER ErfFcnal<T, complex_domain_tag>::grad_t ErfFcnal<T, complex_domain_tag>::gradient(IvyThreadSafePtr_t<X_t> const& x){
    return Exp(-x*x)*Scalar<fndtype_t>(x.get_memory_type(), x.gpu_stream(), TwoOverSqrtPi<fndtype_t>());
  }
  template<typename T>
  __HOST__ ErfFcnal<T, tensor_domain_tag>::value_t ErfFcnal<T, tensor_domain_tag>::eval(T const& x){
    return tensor_apply_elementwise(x, [](auto v){ return Erf(v); });
  }
  template<typename T>
  __HOST__ IvyThreadSafePtr_t<T> ErfFcnal<T, tensor_domain_tag>::gradient(IvyThreadSafePtr_t<T> const& dep){
    // f(x) = erf(x), f'(x) = (2/sqrt(pi)) * exp(-x^2)
    constexpr std_ivy::IvyMemoryType def_mem_type = IvyMemoryHelpers::get_execution_default_memory();
    return make_IvyThreadSafePtr<T>(def_mem_type, nullptr, tensor_apply_elementwise(*dep, [](auto v){ using rt = fundamental_data_t<decltype(v)>; return TwoOverSqrtPi<rt>()*Exp(-v*v); }));
  }
  template<typename T, ENABLE_IF_BOOL_IMPL(!is_pointer_v<T> && !is_tensor_v<T>)> __HOST_DEVICE__ typename ErfFcnal<T>::value_t Erf(T const& x){ return ErfFcnal<T>::eval(x); }
  template<typename T, ENABLE_IF_BOOL_IMPL(!is_pointer_v<T> && is_tensor_v<T>)> __HOST__ typename ErfFcnal<T>::value_t Erf(T const& x){ return ErfFcnal<T>::eval(x); }
  /**
   * @brief Construct a lazy Erf function node for autodiff.
   * @note  Host-only: function-graph objects (IvyRegularFunction) use
   *        virtual dispatch and RAII, which are incompatible with device code.
   *        Direct numerical evaluation (the non-pointer overload) is __HOST_DEVICE__.
   */
  template<typename T, ENABLE_IF_BOOL_IMPL(is_pointer_v<T>)>
  __HOST__ IvyThreadSafePtr_t<typename IvyErf<typename T::element_type>::base_t> Erf(T const& x){
    constexpr std_ivy::IvyMemoryType def_mem_type = IvyMemoryHelpers::get_execution_default_memory();
    auto res = make_IvyThreadSafePtr<IvyErf<typename T::element_type>>(def_mem_type, nullptr, IvyErf(x));
    add_fcn_to_clients(res, x);
    return res;
  }

  // ERFC
  template<typename T, typename domain_tag>
  __HOST_DEVICE__ ErfcFcnal<T, domain_tag>::value_t ErfcFcnal<T, domain_tag>::eval(T const& x){
    return std_math::erfc(x);
  }
  template<typename T>
  __HOST_DEVICE__ ErfcFcnal<T, real_domain_tag>::value_t ErfcFcnal<T, real_domain_tag>::eval(T const& x){
    return value_t(Erfc(unpack_function_input_reduced<T>::get(x)));
  }
  template<typename T> template<typename X_t>
  IVY_MATH_GRAPH_QUALIFIER ErfcFcnal<T, real_domain_tag>::grad_t ErfcFcnal<T, real_domain_tag>::gradient(IvyThreadSafePtr_t<X_t> const& x){
    return -Exp(-x*x)*Scalar<fndtype_t>(x.get_memory_type(), x.gpu_stream(), TwoOverSqrtPi<fndtype_t>());
  }
  template<typename T>
  __HOST_DEVICE__ ErfcFcnal<T, complex_domain_tag>::value_t ErfcFcnal<T, complex_domain_tag>::eval(T const& x){
    return IvyCerf::erfc(unpack_function_input_reduced<T>::get(x));
  }
  template<typename T> template<typename X_t>
  IVY_MATH_GRAPH_QUALIFIER ErfcFcnal<T, complex_domain_tag>::grad_t ErfcFcnal<T, complex_domain_tag>::gradient(IvyThreadSafePtr_t<X_t> const& x){
    return -Exp(-x*x)*Scalar<fndtype_t>(x.get_memory_type(), x.gpu_stream(), TwoOverSqrtPi<fndtype_t>());
  }
  template<typename T>
  __HOST__ ErfcFcnal<T, tensor_domain_tag>::value_t ErfcFcnal<T, tensor_domain_tag>::eval(T const& x){
    // f(x) = erfc(x) = 1 - erf(x)
    return tensor_apply_elementwise(x, [](auto v){ return Erfc(v); });
  }
  template<typename T>
  __HOST__ IvyThreadSafePtr_t<T> ErfcFcnal<T, tensor_domain_tag>::gradient(IvyThreadSafePtr_t<T> const& dep){
    // f'(x) = -(2/sqrt(pi)) * exp(-x^2)
    constexpr std_ivy::IvyMemoryType def_mem_type = IvyMemoryHelpers::get_execution_default_memory();
    return make_IvyThreadSafePtr<T>(def_mem_type, nullptr, tensor_apply_elementwise(*dep, [](auto v){ using rt = fundamental_data_t<decltype(v)>; return -TwoOverSqrtPi<rt>()*Exp(-v*v); }));
  }
  template<typename T, ENABLE_IF_BOOL_IMPL(!is_pointer_v<T> && !is_tensor_v<T>)> __HOST_DEVICE__ typename ErfcFcnal<T>::value_t Erfc(T const& x){ return ErfcFcnal<T>::eval(x); }
  template<typename T, ENABLE_IF_BOOL_IMPL(!is_pointer_v<T> && is_tensor_v<T>)> __HOST__ typename ErfcFcnal<T>::value_t Erfc(T const& x){ return ErfcFcnal<T>::eval(x); }
  /**
   * @brief Construct a lazy Erfc function node for autodiff.
   * @note  Host-only: function-graph objects (IvyRegularFunction) use
   *        virtual dispatch and RAII, which are incompatible with device code.
   *        Direct numerical evaluation (the non-pointer overload) is __HOST_DEVICE__.
   */
  template<typename T, ENABLE_IF_BOOL_IMPL(is_pointer_v<T>)>
  __HOST__ IvyThreadSafePtr_t<typename IvyErfc<typename T::element_type>::base_t> Erfc(T const& x){
    constexpr std_ivy::IvyMemoryType def_mem_type = IvyMemoryHelpers::get_execution_default_memory();
    auto res = make_IvyThreadSafePtr<IvyErfc<typename T::element_type>>(def_mem_type, nullptr, IvyErfc(x));
    add_fcn_to_clients(res, x);
    return res;
  }

  // FADDEEVA
  template<typename T, typename domain_tag>
  __HOST_DEVICE__ FaddeevaFcnal<T, domain_tag>::value_t FaddeevaFcnal<T, domain_tag>::eval(T const& x){
    return IvyCerf::faddeeva(IvyComplex(x));
  }
  template<typename T>
  __HOST_DEVICE__ FaddeevaFcnal<T, real_domain_tag>::value_t FaddeevaFcnal<T, real_domain_tag>::eval(T const& x){
    return value_t(Faddeeva(IvyComplex(unpack_function_input_reduced<T>::get(x))));
  }
  template<typename T> template<typename X_t>
  IVY_MATH_GRAPH_QUALIFIER FaddeevaFcnal<T, real_domain_tag>::grad_t FaddeevaFcnal<T, real_domain_tag>::gradient(IvyThreadSafePtr_t<X_t> const& x){
    return
      Complex<fndtype_t>(x.get_memory_type(), x.gpu_stream(), Zero<fndtype_t>(), TwoOverSqrtPi<fndtype_t>())
      - Scalar<fndtype_t>(x.get_memory_type(), x.gpu_stream(), Two<fndtype_t>())*x*Faddeeva(x);
  }
  template<typename T>
  __HOST_DEVICE__ FaddeevaFcnal<T, complex_domain_tag>::value_t FaddeevaFcnal<T, complex_domain_tag>::eval(T const& x){
    return IvyCerf::faddeeva(unpack_function_input_reduced<T>::get(x));
  }
  template<typename T> template<typename X_t>
  IVY_MATH_GRAPH_QUALIFIER FaddeevaFcnal<T, complex_domain_tag>::grad_t FaddeevaFcnal<T, complex_domain_tag>::gradient(IvyThreadSafePtr_t<X_t> const& x){
    return
      Complex<fndtype_t>(x.get_memory_type(), x.gpu_stream(), Zero<fndtype_t>(), TwoOverSqrtPi<fndtype_t>())
      - Scalar<fndtype_t>(x.get_memory_type(), x.gpu_stream(), Two<fndtype_t>())*x*Faddeeva(x);
  }

  // FADDEEVA — tensor domain (element-wise, output is a complex-valued tensor)
  template<typename T>
  __HOST__ FaddeevaFcnal<T, tensor_domain_tag>::value_t FaddeevaFcnal<T, tensor_domain_tag>::eval(T const& x){
    constexpr std_ivy::IvyMemoryType def_mem_type = IvyMemoryHelpers::get_execution_default_memory();
    using cplx_dtype_t = convert_to_complex_t<dtype_t>;
    value_t res(x.shape());
    for (IvyTensorDim_t i = 0; i < x.num_elements(); ++i){
      if constexpr (is_pointer_v<dtype_t>){
        using inner_t = typename dtype_t::element_type;
        auto const rv = unpack_function_input_reduced<inner_t>::get(*x[i]);
        using num_t = std_ttraits::remove_const_t<decltype(rv)>;
        auto const cval = IvyCerf::faddeeva(IvyComplex<num_t>(rv, Zero<num_t>()));
        using cinner_t = typename cplx_dtype_t::element_type;
        res[i] = make_IvyThreadSafePtr<cinner_t>(def_mem_type, nullptr, cval);
      } else {
        auto const rv = unpack_function_input_reduced<dtype_t>::get(x[i]);
        using num_t = std_ttraits::remove_const_t<decltype(rv)>;
        auto const cval = IvyCerf::faddeeva(IvyComplex<num_t>(rv, Zero<num_t>()));
        res[i] = cplx_dtype_t(cval.Re(), cval.Im());
      }
    }
    return res;
  }
  template<typename T>
  __HOST__ auto FaddeevaFcnal<T, tensor_domain_tag>::gradient(IvyThreadSafePtr_t<T> const& dep)
  -> IvyThreadSafePtr_t<grad_value_t>{
    // dw/dz element-wise = (2i/√π) − 2·z·w(z)
    constexpr std_ivy::IvyMemoryType def_mem_type = IvyMemoryHelpers::get_execution_default_memory();
    using cplx_dtype_t = convert_to_complex_t<dtype_t>;
    grad_value_t res((*dep).shape());
    for (IvyTensorDim_t i = 0; i < (*dep).num_elements(); ++i){
      if constexpr (is_pointer_v<dtype_t>){
        using inner_t = typename dtype_t::element_type;
        auto const rv = unpack_function_input_reduced<inner_t>::get(*(*dep)[i]);
        using num_t = std_ttraits::remove_const_t<decltype(rv)>;
        using cval_t = IvyComplex<num_t>;
        cval_t const z(rv, Zero<num_t>());
        cval_t const wz = IvyCerf::faddeeva(z);
        cval_t const two_i_over_sqrtpi(Zero<num_t>(), TwoOverSqrtPi<num_t>());
        cval_t const grad_i = two_i_over_sqrtpi - cval_t(Two<num_t>(), Zero<num_t>()) * z * wz;
        using cinner_t = typename cplx_dtype_t::element_type;
        res[i] = make_IvyThreadSafePtr<cinner_t>(def_mem_type, nullptr, grad_i);
      } else {
        auto const rv = unpack_function_input_reduced<dtype_t>::get((*dep)[i]);
        using num_t = std_ttraits::remove_const_t<decltype(rv)>;
        using cval_t = IvyComplex<num_t>;
        cval_t const z(rv, Zero<num_t>());
        cval_t const wz = IvyCerf::faddeeva(z);
        cval_t const two_i_over_sqrtpi(Zero<num_t>(), TwoOverSqrtPi<num_t>());
        cval_t const grad_i = two_i_over_sqrtpi - cval_t(Two<num_t>(), Zero<num_t>()) * z * wz;
        res[i] = cplx_dtype_t(grad_i.Re(), grad_i.Im());
      }
    }
    return make_IvyThreadSafePtr<grad_value_t>(def_mem_type, nullptr, res);
  }
  template<typename T, ENABLE_IF_BOOL_IMPL(!is_pointer_v<T> && !is_tensor_v<T>)> __HOST_DEVICE__ typename FaddeevaFcnal<T>::value_t Faddeeva(T const& x){ return FaddeevaFcnal<T>::eval(x); }
  template<typename T, ENABLE_IF_BOOL_IMPL(!is_pointer_v<T> && is_tensor_v<T>)> __HOST__ typename FaddeevaFcnal<T>::value_t Faddeeva(T const& x){ return FaddeevaFcnal<T>::eval(x); }
  /**
   * @brief Construct a lazy Faddeeva function node for autodiff.
   * @note  Host-only: function-graph objects (IvyRegularFunction) use
   *        virtual dispatch and RAII, which are incompatible with device code.
   *        Direct numerical evaluation (the non-pointer overload) is __HOST_DEVICE__.
   */
  template<typename T, ENABLE_IF_BOOL_IMPL(is_pointer_v<T>)>
  __HOST__ IvyThreadSafePtr_t<typename IvyFaddeeva<typename T::element_type>::base_t> Faddeeva(T const& x){
    constexpr std_ivy::IvyMemoryType def_mem_type = IvyMemoryHelpers::get_execution_default_memory();
    auto res = make_IvyThreadSafePtr<IvyFaddeeva<typename T::element_type>>(def_mem_type, nullptr, IvyFaddeeva(x));
    add_fcn_to_clients(res, x);
    return res;
  }

  // ERF-FAST
  template<typename T, typename domain_tag>
  __HOST_DEVICE__ ErfFastFcnal<T, domain_tag>::value_t ErfFastFcnal<T, domain_tag>::eval(T const& x){
    return std_math::erf(x);
  }
  template<typename T>
  __HOST_DEVICE__ ErfFastFcnal<T, real_domain_tag>::value_t ErfFastFcnal<T, real_domain_tag>::eval(T const& x){
    return value_t(ErfFast(unpack_function_input_reduced<T>::get(x)));
  }
  template<typename T> template<typename X_t>
  IVY_MATH_GRAPH_QUALIFIER ErfFastFcnal<T, real_domain_tag>::grad_t ErfFastFcnal<T, real_domain_tag>::gradient(IvyThreadSafePtr_t<X_t> const& x){
    return Exp(-x*x)*Scalar<fndtype_t>(x.get_memory_type(), x.gpu_stream(), TwoOverSqrtPi<fndtype_t>());
  }
  template<typename T>
  __HOST_DEVICE__ ErfFastFcnal<T, complex_domain_tag>::value_t ErfFastFcnal<T, complex_domain_tag>::eval(T const& x){
    return IvyCerf::erf_fast(unpack_function_input_reduced<T>::get(x));
  }
  template<typename T> template<typename X_t>
  IVY_MATH_GRAPH_QUALIFIER ErfFastFcnal<T, complex_domain_tag>::grad_t ErfFastFcnal<T, complex_domain_tag>::gradient(IvyThreadSafePtr_t<X_t> const& x){
    return Exp(-x*x)*Scalar<fndtype_t>(x.get_memory_type(), x.gpu_stream(), TwoOverSqrtPi<fndtype_t>());
  }
  template<typename T>
  __HOST__ ErfFastFcnal<T, tensor_domain_tag>::value_t ErfFastFcnal<T, tensor_domain_tag>::eval(T const& x){
    // Real-domain ErfFast coincides with erf; the complex domain uses the fast cerf path.
    return tensor_apply_elementwise(x, [](auto v){ return ErfFast(v); });
  }
  template<typename T>
  __HOST__ IvyThreadSafePtr_t<T> ErfFastFcnal<T, tensor_domain_tag>::gradient(IvyThreadSafePtr_t<T> const& dep){
    // f'(x) = (2/sqrt(pi)) * exp(-x^2)
    constexpr std_ivy::IvyMemoryType def_mem_type = IvyMemoryHelpers::get_execution_default_memory();
    return make_IvyThreadSafePtr<T>(def_mem_type, nullptr, tensor_apply_elementwise(*dep, [](auto v){ using rt = fundamental_data_t<decltype(v)>; return TwoOverSqrtPi<rt>()*Exp(-v*v); }));
  }
  template<typename T, ENABLE_IF_BOOL_IMPL(!is_pointer_v<T> && !is_tensor_v<T>)> __HOST_DEVICE__ typename ErfFastFcnal<T>::value_t ErfFast(T const& x){ return ErfFastFcnal<T>::eval(x); }
  template<typename T, ENABLE_IF_BOOL_IMPL(!is_pointer_v<T> && is_tensor_v<T>)> __HOST__ typename ErfFastFcnal<T>::value_t ErfFast(T const& x){ return ErfFastFcnal<T>::eval(x); }
  /**
   * @brief Construct a lazy ErfFast function node for autodiff.
   * @note  Host-only: function-graph objects (IvyRegularFunction) use
   *        virtual dispatch and RAII, which are incompatible with device code.
   *        Direct numerical evaluation (the non-pointer overload) is __HOST_DEVICE__.
   */
  template<typename T, ENABLE_IF_BOOL_IMPL(is_pointer_v<T>)>
  __HOST__ IvyThreadSafePtr_t<typename IvyErfFast<typename T::element_type>::base_t> ErfFast(T const& x){
    constexpr std_ivy::IvyMemoryType def_mem_type = IvyMemoryHelpers::get_execution_default_memory();
    auto res = make_IvyThreadSafePtr<IvyErfFast<typename T::element_type>>(def_mem_type, nullptr, IvyErfFast(x));
    add_fcn_to_clients(res, x);
    return res;
  }

  // ERFC-FAST
  template<typename T, typename domain_tag>
  __HOST_DEVICE__ ErfcFastFcnal<T, domain_tag>::value_t ErfcFastFcnal<T, domain_tag>::eval(T const& x){
    return std_math::erfc(x);
  }
  template<typename T>
  __HOST_DEVICE__ ErfcFastFcnal<T, real_domain_tag>::value_t ErfcFastFcnal<T, real_domain_tag>::eval(T const& x){
    return value_t(ErfcFast(unpack_function_input_reduced<T>::get(x)));
  }
  template<typename T> template<typename X_t>
  IVY_MATH_GRAPH_QUALIFIER ErfcFastFcnal<T, real_domain_tag>::grad_t ErfcFastFcnal<T, real_domain_tag>::gradient(IvyThreadSafePtr_t<X_t> const& x){
    return -Exp(-x*x)*Scalar<fndtype_t>(x.get_memory_type(), x.gpu_stream(), TwoOverSqrtPi<fndtype_t>());
  }
  template<typename T>
  __HOST_DEVICE__ ErfcFastFcnal<T, complex_domain_tag>::value_t ErfcFastFcnal<T, complex_domain_tag>::eval(T const& x){
    return IvyCerf::erfc_fast(unpack_function_input_reduced<T>::get(x));
  }
  template<typename T> template<typename X_t>
  IVY_MATH_GRAPH_QUALIFIER ErfcFastFcnal<T, complex_domain_tag>::grad_t ErfcFastFcnal<T, complex_domain_tag>::gradient(IvyThreadSafePtr_t<X_t> const& x){
    return -Exp(-x*x)*Scalar<fndtype_t>(x.get_memory_type(), x.gpu_stream(), TwoOverSqrtPi<fndtype_t>());
  }
  template<typename T>
  __HOST__ ErfcFastFcnal<T, tensor_domain_tag>::value_t ErfcFastFcnal<T, tensor_domain_tag>::eval(T const& x){
    // Real-domain ErfcFast coincides with erfc; the complex domain uses the fast cerf path.
    return tensor_apply_elementwise(x, [](auto v){ return ErfcFast(v); });
  }
  template<typename T>
  __HOST__ IvyThreadSafePtr_t<T> ErfcFastFcnal<T, tensor_domain_tag>::gradient(IvyThreadSafePtr_t<T> const& dep){
    // f'(x) = -(2/sqrt(pi)) * exp(-x^2)
    constexpr std_ivy::IvyMemoryType def_mem_type = IvyMemoryHelpers::get_execution_default_memory();
    return make_IvyThreadSafePtr<T>(def_mem_type, nullptr, tensor_apply_elementwise(*dep, [](auto v){ using rt = fundamental_data_t<decltype(v)>; return -TwoOverSqrtPi<rt>()*Exp(-v*v); }));
  }
  template<typename T, ENABLE_IF_BOOL_IMPL(!is_pointer_v<T> && !is_tensor_v<T>)> __HOST_DEVICE__ typename ErfcFastFcnal<T>::value_t ErfcFast(T const& x){ return ErfcFastFcnal<T>::eval(x); }
  template<typename T, ENABLE_IF_BOOL_IMPL(!is_pointer_v<T> && is_tensor_v<T>)> __HOST__ typename ErfcFastFcnal<T>::value_t ErfcFast(T const& x){ return ErfcFastFcnal<T>::eval(x); }
  /**
   * @brief Construct a lazy ErfcFast function node for autodiff.
   * @note  Host-only: function-graph objects (IvyRegularFunction) use
   *        virtual dispatch and RAII, which are incompatible with device code.
   *        Direct numerical evaluation (the non-pointer overload) is __HOST_DEVICE__.
   */
  template<typename T, ENABLE_IF_BOOL_IMPL(is_pointer_v<T>)>
  __HOST__ IvyThreadSafePtr_t<typename IvyErfcFast<typename T::element_type>::base_t> ErfcFast(T const& x){
    constexpr std_ivy::IvyMemoryType def_mem_type = IvyMemoryHelpers::get_execution_default_memory();
    auto res = make_IvyThreadSafePtr<IvyErfcFast<typename T::element_type>>(def_mem_type, nullptr, IvyErfcFast(x));
    add_fcn_to_clients(res, x);
    return res;
  }

  // FADDEEVA-FAST
  template<typename T, typename domain_tag>
  __HOST_DEVICE__ FaddeevaFastFcnal<T, domain_tag>::value_t FaddeevaFastFcnal<T, domain_tag>::eval(T const& x){
    return IvyCerf::faddeeva_fast(IvyComplex(x));
  }
  template<typename T>
  __HOST_DEVICE__ FaddeevaFastFcnal<T, real_domain_tag>::value_t FaddeevaFastFcnal<T, real_domain_tag>::eval(T const& x){
    return value_t(FaddeevaFast(IvyComplex(unpack_function_input_reduced<T>::get(x))));
  }
  template<typename T> template<typename X_t>
  IVY_MATH_GRAPH_QUALIFIER FaddeevaFastFcnal<T, real_domain_tag>::grad_t FaddeevaFastFcnal<T, real_domain_tag>::gradient(IvyThreadSafePtr_t<X_t> const& x){
    return
      Complex<fndtype_t>(x.get_memory_type(), x.gpu_stream(), Zero<fndtype_t>(), TwoOverSqrtPi<fndtype_t>())
      - Scalar<fndtype_t>(x.get_memory_type(), x.gpu_stream(), Two<fndtype_t>())*x*FaddeevaFast(x);
  }
  template<typename T>
  __HOST_DEVICE__ FaddeevaFastFcnal<T, complex_domain_tag>::value_t FaddeevaFastFcnal<T, complex_domain_tag>::eval(T const& x){
    return IvyCerf::faddeeva_fast(unpack_function_input_reduced<T>::get(x));
  }
  template<typename T> template<typename X_t>
  IVY_MATH_GRAPH_QUALIFIER FaddeevaFastFcnal<T, complex_domain_tag>::grad_t FaddeevaFastFcnal<T, complex_domain_tag>::gradient(IvyThreadSafePtr_t<X_t> const& x){
    return
      Complex<fndtype_t>(x.get_memory_type(), x.gpu_stream(), Zero<fndtype_t>(), TwoOverSqrtPi<fndtype_t>())
      - Scalar<fndtype_t>(x.get_memory_type(), x.gpu_stream(), Two<fndtype_t>())*x*FaddeevaFast(x);
  }
  // FADDEEVA-FAST — tensor domain (element-wise, output is a complex-valued tensor)
  template<typename T>
  __HOST__ FaddeevaFastFcnal<T, tensor_domain_tag>::value_t FaddeevaFastFcnal<T, tensor_domain_tag>::eval(T const& x){
    constexpr std_ivy::IvyMemoryType def_mem_type = IvyMemoryHelpers::get_execution_default_memory();
    using cplx_dtype_t = convert_to_complex_t<dtype_t>;
    value_t res(x.shape());
    for (IvyTensorDim_t i = 0; i < x.num_elements(); ++i){
      if constexpr (is_pointer_v<dtype_t>){
        using inner_t = typename dtype_t::element_type;
        auto const rv = unpack_function_input_reduced<inner_t>::get(*x[i]);
        using num_t = std_ttraits::remove_const_t<decltype(rv)>;
        auto const cval = IvyCerf::faddeeva_fast(IvyComplex<num_t>(rv, Zero<num_t>()));
        using cinner_t = typename cplx_dtype_t::element_type;
        res[i] = make_IvyThreadSafePtr<cinner_t>(def_mem_type, nullptr, cval);
      } else {
        auto const rv = unpack_function_input_reduced<dtype_t>::get(x[i]);
        using num_t = std_ttraits::remove_const_t<decltype(rv)>;
        auto const cval = IvyCerf::faddeeva_fast(IvyComplex<num_t>(rv, Zero<num_t>()));
        res[i] = cplx_dtype_t(cval.Re(), cval.Im());
      }
    }
    return res;
  }
  template<typename T>
  __HOST__ IvyThreadSafePtr_t<typename FaddeevaFastFcnal<T, tensor_domain_tag>::grad_value_t> FaddeevaFastFcnal<T, tensor_domain_tag>::gradient(IvyThreadSafePtr_t<T> const& dep){
    // dw/dz element-wise = (2i/√π) − 2·z·w(z)
    constexpr std_ivy::IvyMemoryType def_mem_type = IvyMemoryHelpers::get_execution_default_memory();
    using cplx_dtype_t = convert_to_complex_t<dtype_t>;
    grad_value_t res((*dep).shape());
    for (IvyTensorDim_t i = 0; i < (*dep).num_elements(); ++i){
      if constexpr (is_pointer_v<dtype_t>){
        using inner_t = typename dtype_t::element_type;
        auto const rv = unpack_function_input_reduced<inner_t>::get(*(*dep)[i]);
        using num_t = std_ttraits::remove_const_t<decltype(rv)>;
        using cval_t = IvyComplex<num_t>;
        cval_t const z(rv, Zero<num_t>());
        cval_t const wz = IvyCerf::faddeeva_fast(z);
        cval_t const two_i_over_sqrtpi(Zero<num_t>(), TwoOverSqrtPi<num_t>());
        cval_t const grad_i = two_i_over_sqrtpi - cval_t(Two<num_t>(), Zero<num_t>()) * z * wz;
        using cinner_t = typename cplx_dtype_t::element_type;
        res[i] = make_IvyThreadSafePtr<cinner_t>(def_mem_type, nullptr, grad_i);
      } else {
        auto const rv = unpack_function_input_reduced<dtype_t>::get((*dep)[i]);
        using num_t = std_ttraits::remove_const_t<decltype(rv)>;
        using cval_t = IvyComplex<num_t>;
        cval_t const z(rv, Zero<num_t>());
        cval_t const wz = IvyCerf::faddeeva_fast(z);
        cval_t const two_i_over_sqrtpi(Zero<num_t>(), TwoOverSqrtPi<num_t>());
        cval_t const grad_i = two_i_over_sqrtpi - cval_t(Two<num_t>(), Zero<num_t>()) * z * wz;
        res[i] = cplx_dtype_t(grad_i.Re(), grad_i.Im());
      }
    }
    return make_IvyThreadSafePtr<grad_value_t>(def_mem_type, nullptr, res);
  }
  template<typename T, ENABLE_IF_BOOL_IMPL(!is_pointer_v<T> && !is_tensor_v<T>)> __HOST_DEVICE__ typename FaddeevaFastFcnal<T>::value_t FaddeevaFast(T const& x){ return FaddeevaFastFcnal<T>::eval(x); }
  template<typename T, ENABLE_IF_BOOL_IMPL(!is_pointer_v<T> && is_tensor_v<T>)> __HOST__ typename FaddeevaFastFcnal<T>::value_t FaddeevaFast(T const& x){ return FaddeevaFastFcnal<T>::eval(x); }
  /**
   * @brief Construct a lazy FaddeevaFast function node for autodiff.
   * @note  Host-only: function-graph objects (IvyRegularFunction) use
   *        virtual dispatch and RAII, which are incompatible with device code.
   *        Direct numerical evaluation (the non-pointer overload) is __HOST_DEVICE__.
   */
  template<typename T, ENABLE_IF_BOOL_IMPL(is_pointer_v<T>)>
  __HOST__ IvyThreadSafePtr_t<typename IvyFaddeevaFast<typename T::element_type>::base_t> FaddeevaFast(T const& x){
    constexpr std_ivy::IvyMemoryType def_mem_type = IvyMemoryHelpers::get_execution_default_memory();
    auto res = make_IvyThreadSafePtr<IvyFaddeevaFast<typename T::element_type>>(def_mem_type, nullptr, IvyFaddeevaFast(x));
    add_fcn_to_clients(res, x);
    return res;
  }


  /****************/
  /* 2D FUNCTIONS */
  /****************/

  // ADDITION
  template<typename T, typename U, typename domain_T, typename domain_U>
  __HOST_DEVICE__ AddFcnal<T, U, domain_T, domain_U>::value_t AddFcnal<T, U, domain_T, domain_U>::eval(T const& x, U const& y){ return x+y; }
  template<typename T, typename U>
  __HOST_DEVICE__ AddFcnal<T, U, real_domain_tag, real_domain_tag>::value_t AddFcnal<T, U, real_domain_tag, real_domain_tag>::eval(T const& x, U const& y){
    return value_t(unpack_function_input_reduced<T>::get(x)+unpack_function_input_reduced<U>::get(y));
  }
  template<typename T, typename U> template<typename X_t, typename Y_t>
  IVY_MATH_GRAPH_QUALIFIER AddFcnal<T, U, real_domain_tag, real_domain_tag>::grad_t AddFcnal<T, U, real_domain_tag, real_domain_tag>::gradient( unsigned char ivar, IvyThreadSafePtr_t<X_t> const& x, IvyThreadSafePtr_t<Y_t> const& y){
    auto mem_type = (ivar==0 ? x.get_memory_type() : y.get_memory_type());
    auto gpu_stream = (ivar==0 ? x.gpu_stream() : y.gpu_stream());
    return make_IvyThreadSafePtr<typename grad_t::element_type>(
      mem_type, gpu_stream,
      One<fndtype_t>()
    );
  }
  template<typename T, typename U>
  __HOST_DEVICE__ AddFcnal<T, U, complex_domain_tag, complex_domain_tag>::value_t AddFcnal<T, U, complex_domain_tag, complex_domain_tag>::eval(T const& x, U const& y){
    return value_t(unpack_function_input_reduced<T>::get(x).Re()+unpack_function_input_reduced<U>::get(y).Re(), unpack_function_input_reduced<T>::get(x).Im()+unpack_function_input_reduced<U>::get(y).Im());
  }
  template<typename T, typename U> template<typename X_t, typename Y_t>
  IVY_MATH_GRAPH_QUALIFIER AddFcnal<T, U, complex_domain_tag, complex_domain_tag>::grad_t AddFcnal<T, U, complex_domain_tag, complex_domain_tag>::gradient(unsigned char ivar, IvyThreadSafePtr_t<X_t> const& x, IvyThreadSafePtr_t<Y_t> const& y){
    auto mem_type = (ivar==0 ? x.get_memory_type() : y.get_memory_type());
    auto gpu_stream = (ivar==0 ? x.gpu_stream() : y.gpu_stream());
    return make_IvyThreadSafePtr<typename grad_t::element_type>(
      mem_type, gpu_stream,
      One<fndtype_t>()
    );
  }
  template<typename T, typename U>
  __HOST_DEVICE__ AddFcnal<T, U, quaternion_domain_tag, quaternion_domain_tag>::value_t AddFcnal<T, U, quaternion_domain_tag, quaternion_domain_tag>::eval(T const& x, U const& y){
    auto const& a = unpack_function_input_reduced<T>::get(x);
    auto const& b = unpack_function_input_reduced<U>::get(y);
    return value_t(a.W()+b.W(), a.X()+b.X(), a.Y()+b.Y(), a.Z()+b.Z());
  }
  template<typename T, typename U> template<typename X_t, typename Y_t>
  IVY_MATH_GRAPH_QUALIFIER AddFcnal<T, U, quaternion_domain_tag, quaternion_domain_tag>::grad_t AddFcnal<T, U, quaternion_domain_tag, quaternion_domain_tag>::gradient(unsigned char ivar, IvyThreadSafePtr_t<X_t> const& x, IvyThreadSafePtr_t<Y_t> const& y){
    auto mem_type = (ivar==0 ? x.get_memory_type() : y.get_memory_type());
    auto gpu_stream = (ivar==0 ? x.gpu_stream() : y.gpu_stream());
    return make_IvyThreadSafePtr<typename grad_t::element_type>(mem_type, gpu_stream, One<fndtype_t>());
  }
  template<typename T, typename U>
  __HOST_DEVICE__ AddFcnal<T, U, arithmetic_domain_tag, real_domain_tag>::value_t AddFcnal<T, U, arithmetic_domain_tag, real_domain_tag>::eval(T const& x, U const& y){
    return value_t(x+unpack_function_input_reduced<U>::get(y));
  }
  template<typename T, typename U>
  __HOST_DEVICE__ AddFcnal<T, U, real_domain_tag, arithmetic_domain_tag>::value_t AddFcnal<T, U, real_domain_tag, arithmetic_domain_tag>::eval(T const& x, U const& y){
    return value_t(unpack_function_input_reduced<T>::get(x)+y);
  }
  template<typename T, typename U>
  __HOST_DEVICE__ AddFcnal<T, U, arithmetic_domain_tag, complex_domain_tag>::value_t AddFcnal<T, U, arithmetic_domain_tag, complex_domain_tag>::eval(T const& x, U const& y){
    return value_t(x+unpack_function_input_reduced<U>::get(y).Re(), unpack_function_input_reduced<U>::get(y).Im());
  }
  template<typename T, typename U>
  __HOST_DEVICE__ AddFcnal<T, U, complex_domain_tag, arithmetic_domain_tag>::value_t AddFcnal<T, U, complex_domain_tag, arithmetic_domain_tag>::eval(T const& x, U const& y){
    return value_t(unpack_function_input_reduced<T>::get(x).Re()+y, unpack_function_input_reduced<T>::get(x).Im());
  }
  template<typename T, typename U>
  __HOST_DEVICE__ AddFcnal<T, U, real_domain_tag, complex_domain_tag>::value_t AddFcnal<T, U, real_domain_tag, complex_domain_tag>::eval(T const& x, U const& y){
    return value_t(unpack_function_input_reduced<T>::get(x)+unpack_function_input_reduced<U>::get(y).Re(), unpack_function_input_reduced<U>::get(y).Im());
  }
  template<typename T, typename U> template<typename X_t, typename Y_t>
  IVY_MATH_GRAPH_QUALIFIER AddFcnal<T, U, real_domain_tag, complex_domain_tag>::grad_t AddFcnal<T, U, real_domain_tag, complex_domain_tag>::gradient(unsigned char ivar, IvyThreadSafePtr_t<X_t> const& x, IvyThreadSafePtr_t<Y_t> const& y){
    auto mem_type = (ivar==0 ? x.get_memory_type() : y.get_memory_type());
    auto gpu_stream = (ivar==0 ? x.gpu_stream() : y.gpu_stream());
    return make_IvyThreadSafePtr<typename grad_t::element_type>(
      mem_type, gpu_stream,
      One<fndtype_t>()
    );
  }
  template<typename T, typename U>
  __HOST_DEVICE__ AddFcnal<T, U, complex_domain_tag, real_domain_tag>::value_t AddFcnal<T, U, complex_domain_tag, real_domain_tag>::eval(T const& x, U const& y){
    return value_t(unpack_function_input_reduced<T>::get(x).Re()+unpack_function_input_reduced<U>::get(y), unpack_function_input_reduced<T>::get(x).Im());
  }
  template<typename T, typename U> template<typename X_t, typename Y_t>
  IVY_MATH_GRAPH_QUALIFIER AddFcnal<T, U, complex_domain_tag, real_domain_tag>::grad_t AddFcnal<T, U, complex_domain_tag, real_domain_tag>::gradient(unsigned char ivar, IvyThreadSafePtr_t<X_t> const& x, IvyThreadSafePtr_t<Y_t> const& y){
    auto mem_type = (ivar==0 ? x.get_memory_type() : y.get_memory_type());
    auto gpu_stream = (ivar==0 ? x.gpu_stream() : y.gpu_stream());
    return make_IvyThreadSafePtr<typename grad_t::element_type>(
      mem_type, gpu_stream,
      One<fndtype_t>()
    );
  }
  template<typename T, typename U, ENABLE_IF_BOOL_IMPL(!is_pointer_v<T> && !is_pointer_v<U>)>
  __HOST_DEVICE__ typename AddFcnal<T, U>::value_t Add(T const& x, U const& y){ return AddFcnal<T, U>::eval(x, y); }
  template<typename T, typename U, ENABLE_IF_BOOL_IMPL(!(is_arithmetic_v<T> && is_arithmetic_v<U>) && !is_pointer_v<T> && !is_pointer_v<U> && (is_ivy_domain_v<T> || is_ivy_domain_v<U>))>
  __HOST_DEVICE__ typename AddFcnal<T, U>::value_t operator+(T const& x, U const& y){ return Add(x, y); }
  /**
   * @brief Construct a lazy Add function node for autodiff.
   * @note  Host-only: function-graph objects (IvyRegularFunction) use
   *        virtual dispatch and RAII, which are incompatible with device code.
   *        Direct numerical evaluation (the non-pointer overload) is __HOST_DEVICE__.
   */
  template<typename T, typename U, ENABLE_IF_BOOL_IMPL(is_pointer_v<T> && is_pointer_v<U>)>
  __HOST__ IvyThreadSafePtr_t<typename IvyAdd<typename T::element_type, typename U::element_type>::base_t> Add(T const& x, U const& y){
    constexpr std_ivy::IvyMemoryType def_mem_type = IvyMemoryHelpers::get_execution_default_memory();
    auto res = make_IvyThreadSafePtr<IvyAdd<typename T::element_type, typename U::element_type>>(def_mem_type, nullptr, IvyAdd(x, y));
    add_fcn_to_clients(res, x);
    add_fcn_to_clients(res, y);
    return res;
  }
  /**
   * @brief Construct a lazy function function node for autodiff.
   * @note  Host-only: function-graph objects (IvyRegularFunction) use
   *        virtual dispatch and RAII, which are incompatible with device code.
   *        Direct numerical evaluation (the non-pointer overload) is __HOST_DEVICE__.
   */
  template<typename T, typename U, ENABLE_IF_BOOL_IMPL(is_pointer_v<T> && is_pointer_v<U>)>
  __HOST__ IvyThreadSafePtr_t<typename IvyAdd<typename T::element_type, typename U::element_type>::base_t> operator+(T const& x, U const& y){
    return Add(x, y);
  }

  // SUBTRACTION
  template<typename T, typename U, typename domain_T, typename domain_U>
  __HOST_DEVICE__ SubtractFcnal<T, U, domain_T, domain_U>::value_t SubtractFcnal<T, U, domain_T, domain_U>::eval(T const& x, U const& y){ return x+y; }
  template<typename T, typename U>
  __HOST_DEVICE__ SubtractFcnal<T, U, real_domain_tag, real_domain_tag>::value_t SubtractFcnal<T, U, real_domain_tag, real_domain_tag>::eval(T const& x, U const& y){
    return value_t(unpack_function_input_reduced<T>::get(x)-unpack_function_input_reduced<U>::get(y));
  }
  template<typename T, typename U> template<typename X_t, typename Y_t>
  IVY_MATH_GRAPH_QUALIFIER SubtractFcnal<T, U, real_domain_tag, real_domain_tag>::grad_t SubtractFcnal<T, U, real_domain_tag, real_domain_tag>::gradient(unsigned char ivar, IvyThreadSafePtr_t<X_t> const& x, IvyThreadSafePtr_t<Y_t> const& y){
    auto mem_type = (ivar==0 ? x.get_memory_type() : y.get_memory_type());
    auto gpu_stream = (ivar==0 ? x.gpu_stream() : y.gpu_stream());
    return make_IvyThreadSafePtr<typename grad_t::element_type>(
      mem_type, gpu_stream,
      (ivar==0 ? One<fndtype_t>() : MinusOne<fndtype_t>())
    );
  }
  template<typename T, typename U>
  __HOST_DEVICE__ SubtractFcnal<T, U, complex_domain_tag, complex_domain_tag>::value_t SubtractFcnal<T, U, complex_domain_tag, complex_domain_tag>::eval(T const& x, U const& y){
    return value_t(unpack_function_input_reduced<T>::get(x).Re()-unpack_function_input_reduced<U>::get(y).Re(), unpack_function_input_reduced<T>::get(x).Im()-unpack_function_input_reduced<U>::get(y).Im());
  }
  template<typename T, typename U> template<typename X_t, typename Y_t>
  IVY_MATH_GRAPH_QUALIFIER SubtractFcnal<T, U, complex_domain_tag, complex_domain_tag>::grad_t SubtractFcnal<T, U, complex_domain_tag, complex_domain_tag>::gradient(unsigned char ivar, IvyThreadSafePtr_t<X_t> const& x, IvyThreadSafePtr_t<Y_t> const& y){
    auto mem_type = (ivar==0 ? x.get_memory_type() : y.get_memory_type());
    auto gpu_stream = (ivar==0 ? x.gpu_stream() : y.gpu_stream());
    return make_IvyThreadSafePtr<typename grad_t::element_type>(
      mem_type, gpu_stream,
      (ivar==0 ? One<fndtype_t>() : MinusOne<fndtype_t>())
    );
  }
  template<typename T, typename U>
  __HOST_DEVICE__ SubtractFcnal<T, U, quaternion_domain_tag, quaternion_domain_tag>::value_t SubtractFcnal<T, U, quaternion_domain_tag, quaternion_domain_tag>::eval(T const& x, U const& y){
    auto const& a = unpack_function_input_reduced<T>::get(x);
    auto const& b = unpack_function_input_reduced<U>::get(y);
    return value_t(a.W()-b.W(), a.X()-b.X(), a.Y()-b.Y(), a.Z()-b.Z());
  }
  template<typename T, typename U> template<typename X_t, typename Y_t>
  IVY_MATH_GRAPH_QUALIFIER SubtractFcnal<T, U, quaternion_domain_tag, quaternion_domain_tag>::grad_t SubtractFcnal<T, U, quaternion_domain_tag, quaternion_domain_tag>::gradient(unsigned char ivar, IvyThreadSafePtr_t<X_t> const& x, IvyThreadSafePtr_t<Y_t> const& y){
    auto mem_type = (ivar==0 ? x.get_memory_type() : y.get_memory_type());
    auto gpu_stream = (ivar==0 ? x.gpu_stream() : y.gpu_stream());
    return make_IvyThreadSafePtr<typename grad_t::element_type>(mem_type, gpu_stream, (ivar==0 ? One<fndtype_t>() : MinusOne<fndtype_t>()));
  }
  template<typename T, typename U>
  __HOST_DEVICE__ SubtractFcnal<T, U, arithmetic_domain_tag, real_domain_tag>::value_t SubtractFcnal<T, U, arithmetic_domain_tag, real_domain_tag>::eval(T const& x, U const& y){
    return value_t(x-unpack_function_input_reduced<U>::get(y));
  }
  template<typename T, typename U>
  __HOST_DEVICE__ SubtractFcnal<T, U, real_domain_tag, arithmetic_domain_tag>::value_t SubtractFcnal<T, U, real_domain_tag, arithmetic_domain_tag>::eval(T const& x, U const& y){
    return value_t(unpack_function_input_reduced<T>::get(x)-y);
  }
  template<typename T, typename U>
  __HOST_DEVICE__ SubtractFcnal<T, U, arithmetic_domain_tag, complex_domain_tag>::value_t SubtractFcnal<T, U, arithmetic_domain_tag, complex_domain_tag>::eval(T const& x, U const& y){
    return value_t(x-unpack_function_input_reduced<U>::get(y).Re(), -unpack_function_input_reduced<U>::get(y).Im());
  }
  template<typename T, typename U>
  __HOST_DEVICE__ SubtractFcnal<T, U, complex_domain_tag, arithmetic_domain_tag>::value_t SubtractFcnal<T, U, complex_domain_tag, arithmetic_domain_tag>::eval(T const& x, U const& y){
    return value_t(unpack_function_input_reduced<T>::get(x).Re()-y, unpack_function_input_reduced<T>::get(x).Im());
  }
  template<typename T, typename U>
  __HOST_DEVICE__ SubtractFcnal<T, U, real_domain_tag, complex_domain_tag>::value_t SubtractFcnal<T, U, real_domain_tag, complex_domain_tag>::eval(T const& x, U const& y){
    return value_t(unpack_function_input_reduced<T>::get(x)-unpack_function_input_reduced<U>::get(y).Re(), -unpack_function_input_reduced<U>::get(y).Im());
  }
  template<typename T, typename U> template<typename X_t, typename Y_t>
  IVY_MATH_GRAPH_QUALIFIER SubtractFcnal<T, U, real_domain_tag, complex_domain_tag>::grad_t SubtractFcnal<T, U, real_domain_tag, complex_domain_tag>::gradient(unsigned char ivar, IvyThreadSafePtr_t<X_t> const& x, IvyThreadSafePtr_t<Y_t> const& y){
    auto mem_type = (ivar==0 ? x.get_memory_type() : y.get_memory_type());
    auto gpu_stream = (ivar==0 ? x.gpu_stream() : y.gpu_stream());
    return make_IvyThreadSafePtr<typename grad_t::element_type>(
      mem_type, gpu_stream,
      (ivar==0 ? One<fndtype_t>() : MinusOne<fndtype_t>())
    );
  }
  template<typename T, typename U>
  __HOST_DEVICE__ SubtractFcnal<T, U, complex_domain_tag, real_domain_tag>::value_t SubtractFcnal<T, U, complex_domain_tag, real_domain_tag>::eval(T const& x, U const& y){
    return value_t(unpack_function_input_reduced<T>::get(x).Re()-unpack_function_input_reduced<U>::get(y), unpack_function_input_reduced<T>::get(x).Im());
  }
  template<typename T, typename U> template<typename X_t, typename Y_t>
  IVY_MATH_GRAPH_QUALIFIER SubtractFcnal<T, U, complex_domain_tag, real_domain_tag>::grad_t SubtractFcnal<T, U, complex_domain_tag, real_domain_tag>::gradient(unsigned char ivar, IvyThreadSafePtr_t<X_t> const& x, IvyThreadSafePtr_t<Y_t> const& y){
    auto mem_type = (ivar==0 ? x.get_memory_type() : y.get_memory_type());
    auto gpu_stream = (ivar==0 ? x.gpu_stream() : y.gpu_stream());
    return make_IvyThreadSafePtr<typename grad_t::element_type>(
      mem_type, gpu_stream,
      (ivar==0 ? One<fndtype_t>() : MinusOne<fndtype_t>())
    );
  }
  template<typename T, typename U, ENABLE_IF_BOOL_IMPL(!is_pointer_v<T> && !is_pointer_v<U>)>
  __HOST_DEVICE__ typename SubtractFcnal<T, U>::value_t Subtract(T const& x, U const& y){ return SubtractFcnal<T, U>::eval(x, y); }
  template<
    typename T, typename U,
    ENABLE_IF_BOOL_IMPL(
      !(is_arithmetic_v<T> && is_arithmetic_v<U>)
      &&
      !is_pointer_v<T> && !is_pointer_v<U>
      &&
      !std_iter::is_contiguous_iterator_v<T> && !std_iter::is_contiguous_iterator_v<U>
      &&
      (is_ivy_domain_v<T> || is_ivy_domain_v<U>)
    )
  > __HOST_DEVICE__ typename SubtractFcnal<T, U>::value_t operator-(T const& x, U const& y){ return Subtract(x, y); }
  /**
   * @brief Construct a lazy Subtract function node for autodiff.
   * @note  Host-only: function-graph objects (IvyRegularFunction) use
   *        virtual dispatch and RAII, which are incompatible with device code.
   *        Direct numerical evaluation (the non-pointer overload) is __HOST_DEVICE__.
   */
  template<typename T, typename U, ENABLE_IF_BOOL_IMPL(is_pointer_v<T> && is_pointer_v<U>)>
  __HOST__ IvyThreadSafePtr_t<typename IvySubtract<typename T::element_type, typename U::element_type>::base_t> Subtract(T const& x, U const& y){
    constexpr std_ivy::IvyMemoryType def_mem_type = IvyMemoryHelpers::get_execution_default_memory();
    auto res = make_IvyThreadSafePtr<IvySubtract<typename T::element_type, typename U::element_type>>(def_mem_type, nullptr, IvySubtract(x, y));
    add_fcn_to_clients(res, x);
    add_fcn_to_clients(res, y);
    return res;
  }
  /**
   * @brief Construct a lazy function function node for autodiff.
   * @note  Host-only: function-graph objects (IvyRegularFunction) use
   *        virtual dispatch and RAII, which are incompatible with device code.
   *        Direct numerical evaluation (the non-pointer overload) is __HOST_DEVICE__.
   */
  template<typename T, typename U, ENABLE_IF_BOOL_IMPL(is_pointer_v<T> && is_pointer_v<U>)>
  __HOST__ IvyThreadSafePtr_t<typename IvySubtract<typename T::element_type, typename U::element_type>::base_t> operator-(T const& x, U const& y){
    return Subtract(x, y);
  }

  // MULTIPLICATION
  template<typename T, typename U, typename domain_T, typename domain_U>
  __HOST_DEVICE__ MultiplyFcnal<T, U, domain_T, domain_U>::value_t MultiplyFcnal<T, U, domain_T, domain_U>::eval(T const& x, U const& y){ return x*y; }
  template<typename T, typename U>
  __HOST_DEVICE__ MultiplyFcnal<T, U, real_domain_tag, real_domain_tag>::value_t MultiplyFcnal<T, U, real_domain_tag, real_domain_tag>::eval(T const& x, U const& y){
    return value_t(unpack_function_input_reduced<T>::get(x)*unpack_function_input_reduced<U>::get(y));
  }
  template<typename T, typename U> template<typename X_t, typename Y_t>
  IVY_MATH_GRAPH_QUALIFIER MultiplyFcnal<T, U, real_domain_tag, real_domain_tag>::grad_t MultiplyFcnal<T, U, real_domain_tag, real_domain_tag>::gradient(unsigned char ivar, IvyThreadSafePtr_t<X_t> const& x, IvyThreadSafePtr_t<Y_t> const& y){
    using grad_type_T = IvyScalar<fndtype_t>;
    using grad_type_U = IvyScalar<fndtype_t>;
    switch (ivar){
    case 0:
      return make_IvyThreadSafePtr<grad_type_T>(x.get_memory_type(), x.gpu_stream(), One<fndtype_t>()) * y;
    default:
      return make_IvyThreadSafePtr<grad_type_U>(y.get_memory_type(), y.gpu_stream(), One<fndtype_t>()) * x;
    }
  }
  template<typename T, typename U>
  __HOST_DEVICE__ MultiplyFcnal<T, U, complex_domain_tag, complex_domain_tag>::value_t MultiplyFcnal<T, U, complex_domain_tag, complex_domain_tag>::eval(T const& x, U const& y){
    auto const xr = unpack_function_input_reduced<T>::get(x).Re();
    auto const xi = unpack_function_input_reduced<T>::get(x).Im();
    auto const yr = unpack_function_input_reduced<U>::get(y).Re();
    auto const yi = unpack_function_input_reduced<U>::get(y).Im();
    return value_t(xr*yr - xi*yi, xr*yi + xi*yr);
  }
  template<typename T, typename U> template<typename X_t, typename Y_t>
  IVY_MATH_GRAPH_QUALIFIER MultiplyFcnal<T, U, complex_domain_tag, complex_domain_tag>::grad_t MultiplyFcnal<T, U, complex_domain_tag, complex_domain_tag>::gradient(unsigned char ivar, IvyThreadSafePtr_t<X_t> const& x, IvyThreadSafePtr_t<Y_t> const& y){
    switch (ivar){
    case 0:
      return Scalar<fndtype_t>(x.get_memory_type(), x.gpu_stream(), One<fndtype_t>()) * y;
    default:
      return Scalar<fndtype_t>(y.get_memory_type(), y.gpu_stream(), One<fndtype_t>()) * x;
    }
  }
  template<typename T, typename U>
  __HOST_DEVICE__ MultiplyFcnal<T, U, arithmetic_domain_tag, real_domain_tag>::value_t MultiplyFcnal<T, U, arithmetic_domain_tag, real_domain_tag>::eval(T const& x, U const& y){
    return value_t(x*unpack_function_input_reduced<U>::get(y));
  }
  template<typename T, typename U>
  __HOST_DEVICE__ MultiplyFcnal<T, U, real_domain_tag, arithmetic_domain_tag>::value_t MultiplyFcnal<T, U, real_domain_tag, arithmetic_domain_tag>::eval(T const& x, U const& y){
    return value_t(unpack_function_input_reduced<T>::get(x)*y);
  }
  template<typename T, typename U>
  __HOST_DEVICE__ MultiplyFcnal<T, U, arithmetic_domain_tag, complex_domain_tag>::value_t MultiplyFcnal<T, U, arithmetic_domain_tag, complex_domain_tag>::eval(T const& x, U const& y){
    return value_t(x*unpack_function_input_reduced<U>::get(y).Re(), x*unpack_function_input_reduced<U>::get(y).Im());
  }
  template<typename T, typename U>
  __HOST_DEVICE__ MultiplyFcnal<T, U, complex_domain_tag, arithmetic_domain_tag>::value_t MultiplyFcnal<T, U, complex_domain_tag, arithmetic_domain_tag>::eval(T const& x, U const& y){
    return value_t(unpack_function_input_reduced<T>::get(x).Re()*y, unpack_function_input_reduced<T>::get(x).Im()*y);
  }
  template<typename T, typename U>
  __HOST_DEVICE__ MultiplyFcnal<T, U, real_domain_tag, complex_domain_tag>::value_t MultiplyFcnal<T, U, real_domain_tag, complex_domain_tag>::eval(T const& x, U const& y){
    return value_t(unpack_function_input_reduced<T>::get(x)*unpack_function_input_reduced<U>::get(y).Re(), unpack_function_input_reduced<T>::get(x)*unpack_function_input_reduced<U>::get(y).Im());
  }
  template<typename T, typename U> template<typename X_t, typename Y_t>
  IVY_MATH_GRAPH_QUALIFIER MultiplyFcnal<T, U, real_domain_tag, complex_domain_tag>::grad_t MultiplyFcnal<T, U, real_domain_tag, complex_domain_tag>::gradient(unsigned char ivar, IvyThreadSafePtr_t<X_t> const& x, IvyThreadSafePtr_t<Y_t> const& y){
    using grad_type_T = IvyScalar<fndtype_t>;
    switch (ivar){
    case 0:
      return make_IvyThreadSafePtr<grad_type_T>(x.get_memory_type(), x.gpu_stream(), One<fndtype_t>()) * y;
    default:
      return Complex<fndtype_t>(y.get_memory_type(), y.gpu_stream(), One<fndtype_t>()) * x;
    }
  }
  template<typename T, typename U>
  __HOST_DEVICE__ MultiplyFcnal<T, U, complex_domain_tag, real_domain_tag>::value_t MultiplyFcnal<T, U, complex_domain_tag, real_domain_tag>::eval(T const& x, U const& y){
    return value_t(unpack_function_input_reduced<T>::get(x).Re()*unpack_function_input_reduced<U>::get(y), unpack_function_input_reduced<T>::get(x).Im()*unpack_function_input_reduced<U>::get(y));
  }
  template<typename T, typename U> template<typename X_t, typename Y_t>
  IVY_MATH_GRAPH_QUALIFIER MultiplyFcnal<T, U, complex_domain_tag, real_domain_tag>::grad_t MultiplyFcnal<T, U, complex_domain_tag, real_domain_tag>::gradient(unsigned char ivar, IvyThreadSafePtr_t<X_t> const& x, IvyThreadSafePtr_t<Y_t> const& y){
    using grad_type_U = IvyScalar<fndtype_t>;
    switch (ivar){
    case 0:
      return Complex<fndtype_t>(x.get_memory_type(), x.gpu_stream(), One<fndtype_t>()) * y;
    default:
      return make_IvyThreadSafePtr<grad_type_U>(y.get_memory_type(), y.gpu_stream(), One<fndtype_t>()) * x;
    }
  }
  template<typename T, typename U>
  __HOST_DEVICE__ MultiplyFcnal<T, U, quaternion_domain_tag, quaternion_domain_tag>::value_t MultiplyFcnal<T, U, quaternion_domain_tag, quaternion_domain_tag>::eval(T const& x, U const& y){
    auto const& a = unpack_function_input_reduced<T>::get(x);
    auto const& b = unpack_function_input_reduced<U>::get(y);
    // Hamilton product (non-commutative): i*j = k, j*i = -k.
    fndtype_t const a0 = a.W(), a1 = a.X(), a2 = a.Y(), a3 = a.Z();
    fndtype_t const b0 = b.W(), b1 = b.X(), b2 = b.Y(), b3 = b.Z();
    return value_t(
      a0*b0 - a1*b1 - a2*b2 - a3*b3,
      a0*b1 + a1*b0 + a2*b3 - a3*b2,
      a0*b2 - a1*b3 + a2*b0 + a3*b1,
      a0*b3 + a1*b2 - a2*b1 + a3*b0
    );
  }
  template<typename T, typename U> template<typename X_t, typename Y_t, typename GX_t, typename GY_t>
  __HOST__ MultiplyFcnal<T, U, quaternion_domain_tag, quaternion_domain_tag>::grad_t MultiplyFcnal<T, U, quaternion_domain_tag, quaternion_domain_tag>::combine_gradient(
    IvyThreadSafePtr_t<X_t> const& x, IvyThreadSafePtr_t<Y_t> const& y, GX_t const& grad_x, GY_t const& grad_y
  ){
    // d(x*y) = (dx)*y + x*(dy): grad_x acts from the left (right-multiplied by y),
    // grad_y acts from the right (left-multiplied by x).
    return grad_x * y + x * grad_y;
  }
  // Tensor-domain binary element-wise ops. The function output is the *reduced*
  // value tensor (e.g. IvyTensor<double>) regardless of whether the operands are
  // array-of-pointers (IvyTensor<IvyScalarPtr_t>) or contiguous cells
  // (IvyTensor<IvyTensorScalarCell>), matching the function's precision_type =
  // more_precise_reduced_t<T,U>. Each operand element is unpacked to its
  // fundamental value and combined element-wise.
  // Element-wise binary tensor op shared implementation (tensor⊗tensor and tensor⊗scalar with
  // broadcast). The function output is the *reduced* value tensor (e.g. IvyTensor<double>, or
  // IvyTensor<IvyComplex<double>> under real⊗complex up-casting) regardless of whether the operands
  // are array-of-pointers (IvyTensor<IvyScalarPtr_t>), contiguous cells (IvyTensor<IvyTensorScalarCell>),
  // or a broadcast scalar leaf (real/arithmetic/complex). Each operand element is unpacked to its
  // reduced value and combined element-wise via OpTag::combine; a scalar operand is broadcast.
  template<typename OpTag, typename T, typename U>
  __HOST__ typename IvyTensorBinaryFcnal<OpTag, T, U>::value_t IvyTensorBinaryFcnal<OpTag, T, U>::eval(T const& x, U const& y){
    constexpr bool x_is_tensor = is_tensor_v<T>;
    auto read = [](auto const& operand, IvyTensorDim_t i){
      using P = std_ttraits::remove_cv_t<std_ttraits::remove_reference_t<decltype(operand)>>;
      if constexpr (is_tensor_v<P>){
        using elem = typename P::dtype_t;
        if constexpr (is_pointer_v<elem>) return unpack_function_input_reduced<typename elem::element_type>::get(*operand[i]);
        else return unpack_function_input_reduced<elem>::get(operand[i]);
      }
      else return unpack_function_input_reduced<P>::get(operand);
    };
    IvyTensorShape const& shape = [&]() -> IvyTensorShape const&{ if constexpr (x_is_tensor) return x.shape(); else return y.shape(); }();
    value_t res(shape);
    IvyTensorDim_t const n = res.num_elements();
    for (IvyTensorDim_t i = 0; i < n; ++i) res[i] = OpTag::combine(read(x, i), read(y, i));
    return res;
  }
  template<typename T, typename U, ENABLE_IF_BOOL_IMPL(!is_pointer_v<T> && !is_pointer_v<U> && !is_tensor_v<T> && !is_tensor_v<U>)>
  __HOST_DEVICE__ typename MultiplyFcnal<T, U>::value_t Multiply(T const& x, U const& y){ return MultiplyFcnal<T, U>::eval(x, y); }
  template<typename T, typename U, ENABLE_IF_BOOL_IMPL(!is_pointer_v<T> && !is_pointer_v<U> && (is_tensor_v<T> || is_tensor_v<U>))>
  __HOST__ typename MultiplyFcnal<T, U>::value_t Multiply(T const& x, U const& y){ return MultiplyFcnal<T, U>::eval(x, y); }
  template<typename T, typename U, ENABLE_IF_BOOL_IMPL(!(is_arithmetic_v<T> && is_arithmetic_v<U>) && !is_pointer_v<T> && !is_pointer_v<U> && !is_tensor_v<T> && !is_tensor_v<U> && (is_ivy_domain_v<T> || is_ivy_domain_v<U>))>
  __HOST_DEVICE__ typename MultiplyFcnal<T, U>::value_t operator*(T const& x, U const& y){ return Multiply(x, y); }
  template<typename T, typename U, ENABLE_IF_BOOL_IMPL(!(is_arithmetic_v<T> && is_arithmetic_v<U>) && !is_pointer_v<T> && !is_pointer_v<U> && (is_tensor_v<T> || is_tensor_v<U>))>
  __HOST__ typename MultiplyFcnal<T, U>::value_t operator*(T const& x, U const& y){ return Multiply(x, y); }
  /**
   * @brief Construct a lazy Multiply function node for autodiff.
   * @note  Host-only: function-graph objects (IvyRegularFunction) use
   *        virtual dispatch and RAII, which are incompatible with device code.
   *        Direct numerical evaluation (the non-pointer overload) is __HOST_DEVICE__.
   */
  template<typename T, typename U, ENABLE_IF_BOOL_IMPL(is_pointer_v<T> && is_pointer_v<U>)>
  __HOST__ IvyThreadSafePtr_t<typename IvyMultiply<typename T::element_type, typename U::element_type>::base_t> Multiply(T const& x, U const& y){
    constexpr std_ivy::IvyMemoryType def_mem_type = IvyMemoryHelpers::get_execution_default_memory();
    auto res = make_IvyThreadSafePtr<IvyMultiply<typename T::element_type, typename U::element_type>>(def_mem_type, nullptr, IvyMultiply(x, y));
    add_fcn_to_clients(res, x);
    add_fcn_to_clients(res, y);
    return res;
  }
  /**
   * @brief Construct a lazy function function node for autodiff.
   * @note  Host-only: function-graph objects (IvyRegularFunction) use
   *        virtual dispatch and RAII, which are incompatible with device code.
   *        Direct numerical evaluation (the non-pointer overload) is __HOST_DEVICE__.
   */
  template<typename T, typename U, ENABLE_IF_BOOL_IMPL(is_pointer_v<T> && is_pointer_v<U>)>
  __HOST__ IvyThreadSafePtr_t<typename IvyMultiply<typename T::element_type, typename U::element_type>::base_t> operator*(T const& x, U const& y){
    return Multiply(x, y);
  }

  // DIVISION
  template<typename T, typename U, typename domain_T, typename domain_U>
  __HOST_DEVICE__ DivideFcnal<T, U, domain_T, domain_U>::value_t DivideFcnal<T, U, domain_T, domain_U>::eval(T const& x, U const& y){ return x/y; }
  template<typename T, typename U>
  __HOST_DEVICE__ DivideFcnal<T, U, real_domain_tag, real_domain_tag>::value_t DivideFcnal<T, U, real_domain_tag, real_domain_tag>::eval(T const& x, U const& y){
    return value_t(unpack_function_input_reduced<T>::get(x)/unpack_function_input_reduced<U>::get(y));
  }
  template<typename T, typename U> template<typename X_t, typename Y_t>
  IVY_MATH_GRAPH_QUALIFIER DivideFcnal<T, U, real_domain_tag, real_domain_tag>::grad_t DivideFcnal<T, U, real_domain_tag, real_domain_tag>::gradient(unsigned char ivar, IvyThreadSafePtr_t<X_t> const& x, IvyThreadSafePtr_t<Y_t> const& y){
    using grad_type_T = IvyScalar<fndtype_t>;
    switch (ivar){
    case 0:
      return make_IvyThreadSafePtr<grad_type_T>(x.get_memory_type(), x.gpu_stream(), One<fndtype_t>()) / y;
    default:
      return -x/(y*y);
    }
  }
  template<typename T, typename U>
  __HOST_DEVICE__ DivideFcnal<T, U, complex_domain_tag, complex_domain_tag>::value_t DivideFcnal<T, U, complex_domain_tag, complex_domain_tag>::eval(T const& x, U const& y){
    return unpack_function_input_reduced<T>::get(x)*MultInverse(unpack_function_input_reduced<U>::get(y));
  }
  template<typename T, typename U> template<typename X_t, typename Y_t>
  IVY_MATH_GRAPH_QUALIFIER DivideFcnal<T, U, complex_domain_tag, complex_domain_tag>::grad_t DivideFcnal<T, U, complex_domain_tag, complex_domain_tag>::gradient(unsigned char ivar, IvyThreadSafePtr_t<X_t> const& x, IvyThreadSafePtr_t<Y_t> const& y){
    switch (ivar){
    case 0:
      return Scalar<fndtype_t>(x.get_memory_type(), x.gpu_stream(), One<fndtype_t>()) / y;
    default:
      return -x/(y*y);
    }
  }
  template<typename T, typename U>
  __HOST_DEVICE__ DivideFcnal<T, U, quaternion_domain_tag, quaternion_domain_tag>::value_t DivideFcnal<T, U, quaternion_domain_tag, quaternion_domain_tag>::eval(T const& x, U const& y){
    // Right division: x / y = x * y^{-1}.
    return unpack_function_input_reduced<T>::get(x)*MultInverse(unpack_function_input_reduced<U>::get(y));
  }
  template<typename T, typename U> template<typename X_t, typename Y_t, typename GX_t, typename GY_t>
  __HOST__ DivideFcnal<T, U, quaternion_domain_tag, quaternion_domain_tag>::grad_t DivideFcnal<T, U, quaternion_domain_tag, quaternion_domain_tag>::combine_gradient(
    IvyThreadSafePtr_t<X_t> const& x, IvyThreadSafePtr_t<Y_t> const& y, GX_t const& grad_x, GY_t const& grad_y
  ){
    // d(x*y^{-1}) = (dx)*y^{-1} - x*y^{-1}*(dy)*y^{-1}.
    auto binv = MultInverse(y);
    return grad_x * binv - x * binv * grad_y * binv;
  }
  template<typename T, typename U>
  __HOST_DEVICE__ DivideFcnal<T, U, arithmetic_domain_tag, real_domain_tag>::value_t DivideFcnal<T, U, arithmetic_domain_tag, real_domain_tag>::eval(T const& x, U const& y){
    return value_t(x/unpack_function_input_reduced<U>::get(y));
  }
  template<typename T, typename U>
  __HOST_DEVICE__ DivideFcnal<T, U, real_domain_tag, arithmetic_domain_tag>::value_t DivideFcnal<T, U, real_domain_tag, arithmetic_domain_tag>::eval(T const& x, U const& y){
    return value_t(unpack_function_input_reduced<T>::get(x)/y);
  }
  template<typename T, typename U>
  __HOST_DEVICE__ DivideFcnal<T, U, arithmetic_domain_tag, complex_domain_tag>::value_t DivideFcnal<T, U, arithmetic_domain_tag, complex_domain_tag>::eval(T const& x, U const& y){
    return x*MultInverse(y);
  }
  template<typename T, typename U>
  __HOST_DEVICE__ DivideFcnal<T, U, complex_domain_tag, arithmetic_domain_tag>::value_t DivideFcnal<T, U, complex_domain_tag, arithmetic_domain_tag>::eval(T const& x, U const& y){
    return value_t(unpack_function_input_reduced<T>::get(x).Re()/y, unpack_function_input_reduced<T>::get(x).Im()/y);
  }
  template<typename T, typename U>
  __HOST_DEVICE__ DivideFcnal<T, U, real_domain_tag, complex_domain_tag>::value_t DivideFcnal<T, U, real_domain_tag, complex_domain_tag>::eval(T const& x, U const& y){
    return unpack_function_input_reduced<T>::get(x)*MultInverse(y);
  }
  template<typename T, typename U> template<typename X_t, typename Y_t>
  IVY_MATH_GRAPH_QUALIFIER DivideFcnal<T, U, real_domain_tag, complex_domain_tag>::grad_t DivideFcnal<T, U, real_domain_tag, complex_domain_tag>::gradient(unsigned char ivar, IvyThreadSafePtr_t<X_t> const& x, IvyThreadSafePtr_t<Y_t> const& y){
    using grad_type_T = IvyScalar<fndtype_t>;
    switch (ivar){
    case 0:
      return Scalar<fndtype_t>(x.get_memory_type(), x.gpu_stream(), One<fndtype_t>()) / y;
    default:
      return -x/(y*y);
    }
  }
  template<typename T, typename U>
  __HOST_DEVICE__ DivideFcnal<T, U, complex_domain_tag, real_domain_tag>::value_t DivideFcnal<T, U, complex_domain_tag, real_domain_tag>::eval(T const& x, U const& y){
    return unpack_function_input_reduced<T>::get(x)*MultInverse(y);
  }
  template<typename T, typename U> template<typename X_t, typename Y_t>
  IVY_MATH_GRAPH_QUALIFIER DivideFcnal<T, U, complex_domain_tag, real_domain_tag>::grad_t DivideFcnal<T, U, complex_domain_tag, real_domain_tag>::gradient(unsigned char ivar, IvyThreadSafePtr_t<X_t> const& x, IvyThreadSafePtr_t<Y_t> const& y){
    switch (ivar){
    case 0:
      return Complex<fndtype_t>(x.get_memory_type(), x.gpu_stream(), One<fndtype_t>()) / y;
    default:
      return -x/(y*y);
    }
  }
  template<typename T, typename U, ENABLE_IF_BOOL_IMPL(!is_pointer_v<T> && !is_pointer_v<U>)>
  __HOST_DEVICE__ typename DivideFcnal<T, U>::value_t Divide(T const& x, U const& y){ return DivideFcnal<T, U>::eval(x, y); }
  template<typename T, typename U, ENABLE_IF_BOOL_IMPL(!(is_arithmetic_v<T> && is_arithmetic_v<U>) && !is_pointer_v<T> && !is_pointer_v<U> && (is_ivy_domain_v<T> || is_ivy_domain_v<U>))>
  __HOST_DEVICE__ typename DivideFcnal<T, U>::value_t operator/(T const& x, U const& y){ return Divide(x, y); }
  /**
   * @brief Construct a lazy Divide function node for autodiff.
   * @note  Host-only: function-graph objects (IvyRegularFunction) use
   *        virtual dispatch and RAII, which are incompatible with device code.
   *        Direct numerical evaluation (the non-pointer overload) is __HOST_DEVICE__.
   */
  template<typename T, typename U, ENABLE_IF_BOOL_IMPL(is_pointer_v<T> && is_pointer_v<U>)>
  __HOST__ IvyThreadSafePtr_t<typename IvyDivide<typename T::element_type, typename U::element_type>::base_t> Divide(T const& x, U const& y){
    constexpr std_ivy::IvyMemoryType def_mem_type = IvyMemoryHelpers::get_execution_default_memory();
    auto res = make_IvyThreadSafePtr<IvyDivide<typename T::element_type, typename U::element_type>>(def_mem_type, nullptr, IvyDivide(x, y));
    add_fcn_to_clients(res, x);
    add_fcn_to_clients(res, y);
    return res;
  }
  /**
   * @brief Construct a lazy function function node for autodiff.
   * @note  Host-only: function-graph objects (IvyRegularFunction) use
   *        virtual dispatch and RAII, which are incompatible with device code.
   *        Direct numerical evaluation (the non-pointer overload) is __HOST_DEVICE__.
   */
  template<typename T, typename U, ENABLE_IF_BOOL_IMPL(is_pointer_v<T> && is_pointer_v<U>)>
  __HOST__ IvyThreadSafePtr_t<typename IvyDivide<typename T::element_type, typename U::element_type>::base_t> operator/(T const& x, U const& y){
    return Divide(x, y);
  }

  // POWER
  template<typename T, typename U, typename domain_T, typename domain_U>
  __HOST_DEVICE__ PowFcnal<T, U, domain_T, domain_U>::value_t PowFcnal<T, U, domain_T, domain_U>::eval(T const& x, U const& y){ return std_math::pow(x, y); }
  template<typename T, typename U>
  __HOST_DEVICE__ PowFcnal<T, U, real_domain_tag, real_domain_tag>::value_t PowFcnal<T, U, real_domain_tag, real_domain_tag>::eval(T const& x, U const& y){
    return Pow(unpack_function_input_reduced<T>::get(x), unpack_function_input_reduced<U>::get(y));
  }
  template<typename T, typename U> template<typename X_t, typename Y_t>
  IVY_MATH_GRAPH_QUALIFIER PowFcnal<T, U, real_domain_tag, real_domain_tag>::grad_t PowFcnal<T, U, real_domain_tag, real_domain_tag>::gradient(unsigned char ivar, IvyThreadSafePtr_t<X_t> const& x, IvyThreadSafePtr_t<Y_t> const& y){
    using ctype = fundamental_data_t<U>;
    switch (ivar){
    case 0:
      return y*Pow(x, y-Scalar<ctype>(y.get_memory_type(), y.gpu_stream(), One<ctype>()));
    default:
      return Log(x)*Pow(x, y);
    }
  }
  template<typename T, typename U>
  __HOST_DEVICE__ PowFcnal<T, U, complex_domain_tag, complex_domain_tag>::value_t PowFcnal<T, U, complex_domain_tag, complex_domain_tag>::eval(T const& x, U const& y){
    auto const xn = unpack_function_input_reduced<T>::get(x).norm();
    auto const xp = unpack_function_input_reduced<T>::get(x).phase();
    auto const yr = unpack_function_input_reduced<U>::get(y).Re();
    auto const yi = unpack_function_input_reduced<U>::get(y).Im();
    auto const res_norm = Pow(xn, yr)*Exp(-xp*yi);
    auto const res_phase = xp*yr + yi*Log(xn);
    return value_t(res_norm*Cos(res_phase), res_norm*Sin(res_phase));
  }
  template<typename T, typename U> template<typename X_t, typename Y_t>
  IVY_MATH_GRAPH_QUALIFIER PowFcnal<T, U, complex_domain_tag, complex_domain_tag>::grad_t PowFcnal<T, U, complex_domain_tag, complex_domain_tag>::gradient(unsigned char ivar, IvyThreadSafePtr_t<X_t> const& x, IvyThreadSafePtr_t<Y_t> const& y){
    using ctype = fundamental_data_t<U>;
    switch (ivar){
    case 0:
      return y*Pow(x, y-Scalar<ctype>(y.get_memory_type(), y.gpu_stream(), One<ctype>()));
    default:
      return Log(x)*Pow(x, y);
    }
  }
  template<typename T, typename U>
  __HOST_DEVICE__ PowFcnal<T, U, arithmetic_domain_tag, real_domain_tag>::value_t PowFcnal<T, U, arithmetic_domain_tag, real_domain_tag>::eval(T const& x, U const& y){
    return value_t(Pow(x, unpack_function_input_reduced<U>::get(y)));
  }
  template<typename T, typename U>
  __HOST_DEVICE__ PowFcnal<T, U, real_domain_tag, arithmetic_domain_tag>::value_t PowFcnal<T, U, real_domain_tag, arithmetic_domain_tag>::eval(T const& x, U const& y){
    return value_t(Pow(unpack_function_input_reduced<T>::get(x), y));
  }
  template<typename T, typename U>
  __HOST_DEVICE__ PowFcnal<T, U, arithmetic_domain_tag, complex_domain_tag>::value_t PowFcnal<T, U, arithmetic_domain_tag, complex_domain_tag>::eval(T const& x, U const& y){
    auto const xn = Abs(x);
    auto const xp = x>=0 ? Zero<fndtype_t>() : Pi<fndtype_t>();
    auto const yr = unpack_function_input_reduced<U>::get(y).Re();
    auto const yi = unpack_function_input_reduced<U>::get(y).Im();
    auto const res_norm = Pow(xn, yr)*Exp(-xp*yi);
    auto const res_phase = xp*yr + yi*Log(xn);
    return value_t(res_norm*Cos(res_phase), res_norm*Sin(res_phase));
  }
  template<typename T, typename U>
  __HOST_DEVICE__ PowFcnal<T, U, complex_domain_tag, arithmetic_domain_tag>::value_t PowFcnal<T, U, complex_domain_tag, arithmetic_domain_tag>::eval(T const& x, U const& y){
    auto const xn = unpack_function_input_reduced<T>::get(x).norm();
    auto const xp = unpack_function_input_reduced<T>::get(x).phase();
    auto const& yr = y;
    auto const res_norm = Pow(xn, yr);
    auto const res_phase = xp*yr;
    return value_t(res_norm*Cos(res_phase), res_norm*Sin(res_phase));
  }
  template<typename T, typename U>
  __HOST_DEVICE__ PowFcnal<T, U, real_domain_tag, complex_domain_tag>::value_t PowFcnal<T, U, real_domain_tag, complex_domain_tag>::eval(T const& x, U const& y){
    auto const& xx = unpack_function_input_reduced<T>::get(x);
    auto const xn = Abs(xx);
    auto const xp = xx>=0 ? Zero<fndtype_t>() : Pi<fndtype_t>();
    auto const yr = unpack_function_input_reduced<U>::get(y).Re();
    auto const yi = unpack_function_input_reduced<U>::get(y).Im();
    auto const res_norm = Pow(xn, yr)*Exp(-xp*yi);
    auto const res_phase = xp*yr + yi*Log(xn);
    return value_t(res_norm*Cos(res_phase), res_norm*Sin(res_phase));
  }
  template<typename T, typename U> template<typename X_t, typename Y_t>
  IVY_MATH_GRAPH_QUALIFIER PowFcnal<T, U, real_domain_tag, complex_domain_tag>::grad_t PowFcnal<T, U, real_domain_tag, complex_domain_tag>::gradient(unsigned char ivar, IvyThreadSafePtr_t<X_t> const& x, IvyThreadSafePtr_t<Y_t> const& y){
    using ctype = fundamental_data_t<U>;
    switch (ivar){
    case 0:
      return y*Pow(x, y-Scalar<ctype>(y.get_memory_type(), y.gpu_stream(), One<ctype>()));
    default:
      return Log(x)*Pow(x, y);
    }
  }
  template<typename T, typename U>
  __HOST_DEVICE__ PowFcnal<T, U, complex_domain_tag, real_domain_tag>::value_t PowFcnal<T, U, complex_domain_tag, real_domain_tag>::eval(T const& x, U const& y){
    auto const xn = unpack_function_input_reduced<T>::get(x).norm();
    auto const xp = unpack_function_input_reduced<T>::get(x).phase();
    auto const yr = unpack_function_input_reduced<U>::get(y);
    auto const res_norm = Pow(xn, yr);
    auto const res_phase = xp*yr;
    return value_t(res_norm*Cos(res_phase), res_norm*Sin(res_phase));
  }
  template<typename T, typename U> template<typename X_t, typename Y_t>
  IVY_MATH_GRAPH_QUALIFIER PowFcnal<T, U, complex_domain_tag, real_domain_tag>::grad_t PowFcnal<T, U, complex_domain_tag, real_domain_tag>::gradient(unsigned char ivar, IvyThreadSafePtr_t<X_t> const& x, IvyThreadSafePtr_t<Y_t> const& y){
    using ctype = fundamental_data_t<U>;
    switch (ivar){
    case 0:
      return y*Pow(x, y-Scalar<ctype>(y.get_memory_type(), y.gpu_stream(), One<ctype>()));
    default:
      return Log(x)*Pow(x, y);
    }
  }
  template<typename T, typename U>
  __HOST__ PowFcnal<T, U, tensor_domain_tag, tensor_domain_tag>::value_t PowFcnal<T, U, tensor_domain_tag, tensor_domain_tag>::eval(T const& x, U const& y){
    constexpr std_ivy::IvyMemoryType def_mem_type = IvyMemoryHelpers::get_execution_default_memory();
    value_t res(x);
    for (IvyTensorDim_t i = 0; i < x.num_elements(); ++i){
      using x_elem_t = typename T::dtype_t;
      using y_elem_t = typename U::dtype_t;
      if constexpr (is_pointer_v<x_elem_t> && is_pointer_v<y_elem_t>){
        using inner_t = typename x_elem_t::element_type;
        auto const xv = unpack_function_input_reduced<inner_t>::get(*x[i]);
        auto const yv = unpack_function_input_reduced<typename y_elem_t::element_type>::get(*y[i]);
        res[i] = make_IvyThreadSafePtr<inner_t>(def_mem_type, nullptr, Pow(xv, yv));
      } else if constexpr (!is_pointer_v<x_elem_t> && !is_pointer_v<y_elem_t>){
        auto const xv = unpack_function_input_reduced<x_elem_t>::get(x[i]);
        auto const yv = unpack_function_input_reduced<y_elem_t>::get(y[i]);
        res[i] = Pow(xv, yv);
      }
    }
    return res;
  }
  template<typename T, typename U> template<typename X_t, typename Y_t>
  __HOST__ IvyThreadSafePtr_t<typename PowFcnal<T, U, tensor_domain_tag, tensor_domain_tag>::value_t> PowFcnal<T, U, tensor_domain_tag, tensor_domain_tag>::gradient(unsigned char ivar, IvyThreadSafePtr_t<X_t> const& x, IvyThreadSafePtr_t<Y_t> const& y){
    // ivar==0: d/dx x^y = y * x^(y-1)
    // ivar==1: d/dy x^y = log(x) * x^y
    constexpr std_ivy::IvyMemoryType def_mem_type = IvyMemoryHelpers::get_execution_default_memory();
    value_t res(*x);
    for (IvyTensorDim_t i = 0; i < (*x).num_elements(); ++i){
      using x_elem_t = typename X_t::dtype_t;
      using y_elem_t = typename Y_t::dtype_t;
      if constexpr (is_pointer_v<x_elem_t> && is_pointer_v<y_elem_t>){
        using inner_t = typename x_elem_t::element_type;
        auto const xv = unpack_function_input_reduced<inner_t>::get(*(*x)[i]);
        auto const yv = unpack_function_input_reduced<typename y_elem_t::element_type>::get(*(*y)[i]);
        std_ttraits::remove_const_t<decltype(xv)> gval{};
        if (ivar == 0){
          gval = yv * Pow(xv, yv - One<decltype(yv)>());
        } else {
          gval = Log(xv) * Pow(xv, yv);
        }
        res[i] = make_IvyThreadSafePtr<inner_t>(def_mem_type, nullptr, gval);
      } else if constexpr (!is_pointer_v<x_elem_t> && !is_pointer_v<y_elem_t>){
        auto const xv = unpack_function_input_reduced<x_elem_t>::get((*x)[i]);
        auto const yv = unpack_function_input_reduced<y_elem_t>::get((*y)[i]);
        if (ivar == 0) res[i] = yv * Pow(xv, yv - One<decltype(yv)>());
        else res[i] = Log(xv) * Pow(xv, yv);
      }
    }
    return make_IvyThreadSafePtr<value_t>(def_mem_type, nullptr, res);
  }
  template<typename T, typename U>
  __HOST__ PowFcnal<T, U, tensor_domain_tag, arithmetic_domain_tag>::value_t PowFcnal<T, U, tensor_domain_tag, arithmetic_domain_tag>::eval(T const& x, U const& y){
    constexpr std_ivy::IvyMemoryType def_mem_type = IvyMemoryHelpers::get_execution_default_memory();
    value_t res(x);
    for (IvyTensorDim_t i = 0; i < x.num_elements(); ++i){
      if constexpr (is_pointer_v<dtype_t>){
        using inner_t = typename dtype_t::element_type;
        auto const xv = unpack_function_input_reduced<inner_t>::get(*x[i]);
        res[i] = make_IvyThreadSafePtr<inner_t>(def_mem_type, nullptr, Pow(xv, static_cast<decltype(xv)>(y)));
      } else {
        auto const xv = unpack_function_input_reduced<dtype_t>::get(x[i]);
        res[i] = Pow(xv, static_cast<decltype(xv)>(y));
      }
    }
    return res;
  }
  template<typename T, typename U> template<typename X_t, typename Y_t>
  __HOST__ IvyThreadSafePtr_t<typename PowFcnal<T, U, tensor_domain_tag, arithmetic_domain_tag>::value_t> PowFcnal<T, U, tensor_domain_tag, arithmetic_domain_tag>::gradient(unsigned char ivar, IvyThreadSafePtr_t<X_t> const& x, IvyThreadSafePtr_t<Y_t> const& y){
    // gradient wrt tensor x: y * x^(y-1)
    // gradient wrt scalar y: undefined (no graph node for arithmetic)
    constexpr std_ivy::IvyMemoryType def_mem_type = IvyMemoryHelpers::get_execution_default_memory();
    value_t res(*x);
    if (ivar == 0){
      for (IvyTensorDim_t i = 0; i < (*x).num_elements(); ++i){
        if constexpr (is_pointer_v<dtype_t>){
          using inner_t = typename dtype_t::element_type;
          auto const xv = unpack_function_input_reduced<inner_t>::get(*(*x)[i]);
          auto const yv = static_cast<decltype(xv)>(*y);
          res[i] = make_IvyThreadSafePtr<inner_t>(def_mem_type, nullptr, yv * Pow(xv, yv - One<decltype(yv)>()));
        } else {
          auto const xv = unpack_function_input_reduced<dtype_t>::get((*x)[i]);
          auto const yv = static_cast<decltype(xv)>(*y);
          res[i] = yv * Pow(xv, yv - One<decltype(yv)>());
        }
      }
    }
    return make_IvyThreadSafePtr<value_t>(def_mem_type, nullptr, res);
  }
  template<typename T, typename U>
  __HOST__ PowFcnal<T, U, arithmetic_domain_tag, tensor_domain_tag>::value_t PowFcnal<T, U, arithmetic_domain_tag, tensor_domain_tag>::eval(T const& x, U const& y){
    constexpr std_ivy::IvyMemoryType def_mem_type = IvyMemoryHelpers::get_execution_default_memory();
    value_t res(y);
    for (IvyTensorDim_t i = 0; i < y.num_elements(); ++i){
      if constexpr (is_pointer_v<dtype_t>){
        using inner_t = typename dtype_t::element_type;
        auto const yv = unpack_function_input_reduced<inner_t>::get(*y[i]);
        res[i] = make_IvyThreadSafePtr<inner_t>(def_mem_type, nullptr, Pow(static_cast<decltype(yv)>(x), yv));
      } else {
        auto const yv = unpack_function_input_reduced<dtype_t>::get(y[i]);
        res[i] = Pow(static_cast<decltype(yv)>(x), yv);
      }
    }
    return res;
  }
  template<typename T, typename U> template<typename X_t, typename Y_t>
  __HOST__ IvyThreadSafePtr_t<typename PowFcnal<T, U, arithmetic_domain_tag, tensor_domain_tag>::value_t> PowFcnal<T, U, arithmetic_domain_tag, tensor_domain_tag>::gradient(unsigned char ivar, IvyThreadSafePtr_t<X_t> const& x, IvyThreadSafePtr_t<Y_t> const& y){
    // gradient wrt tensor y: log(x) * x^y
    constexpr std_ivy::IvyMemoryType def_mem_type = IvyMemoryHelpers::get_execution_default_memory();
    value_t res(*y);
    if (ivar == 1){
      for (IvyTensorDim_t i = 0; i < (*y).num_elements(); ++i){
        if constexpr (is_pointer_v<dtype_t>){
          using inner_t = typename dtype_t::element_type;
          auto const xv = static_cast<fundamental_data_t<typename inner_t::value_t>>(*x);
          auto const yv = unpack_function_input_reduced<inner_t>::get(*(*y)[i]);
          res[i] = make_IvyThreadSafePtr<inner_t>(def_mem_type, nullptr, Log(xv) * Pow(xv, yv));
        } else {
          auto const xv = static_cast<fundamental_data_t<dtype_t>>(*x);
          auto const yv = unpack_function_input_reduced<dtype_t>::get((*y)[i]);
          res[i] = Log(xv) * Pow(xv, yv);
        }
      }
    }
    return make_IvyThreadSafePtr<value_t>(def_mem_type, nullptr, res);
  }
  template<typename T, typename U, ENABLE_IF_BOOL_IMPL(!is_pointer_v<T> && !is_pointer_v<U> && !is_tensor_v<T> && !is_tensor_v<U>)>
  __HOST_DEVICE__ typename PowFcnal<T, U>::value_t Pow(T const& x, U const& y){ return PowFcnal<T, U>::eval(x, y); }
  template<typename T, typename U, ENABLE_IF_BOOL_IMPL(!is_pointer_v<T> && !is_pointer_v<U> && (is_tensor_v<T> || is_tensor_v<U>))>
  __HOST__ typename PowFcnal<T, U>::value_t Pow(T const& x, U const& y){ return PowFcnal<T, U>::eval(x, y); }
  /**
   * @brief Construct a lazy Pow function node for autodiff.
   * @note  Host-only: function-graph objects (IvyRegularFunction) use
   *        virtual dispatch and RAII, which are incompatible with device code.
   *        Direct numerical evaluation (the non-pointer overload) is __HOST_DEVICE__.
   */
  template<typename T, typename U, ENABLE_IF_BOOL_IMPL(is_pointer_v<T>&& is_pointer_v<U>)>
  __HOST__ IvyThreadSafePtr_t<typename IvyPow<typename T::element_type, typename U::element_type>::base_t> Pow(T const& x, U const& y){
    constexpr std_ivy::IvyMemoryType def_mem_type = IvyMemoryHelpers::get_execution_default_memory();
    auto res = make_IvyThreadSafePtr<IvyPow<typename T::element_type, typename U::element_type>>(def_mem_type, nullptr, IvyPow(x, y));
    add_fcn_to_clients(res, x);
    add_fcn_to_clients(res, y);
    return res;
  }


  /******************/
  /* 1D COMPARISONS */
  /******************/

  // NOT
  template<typename T, typename domain_tag>
  __HOST_DEVICE__ constexpr NotFcnal<T, domain_tag>::value_t NotFcnal<T, domain_tag>::eval(T const& x){ return !x; }
  template<typename T>
  __HOST_DEVICE__ NotFcnal<T, real_domain_tag>::value_t NotFcnal<T, real_domain_tag>::eval(T const& x){ return value_t(Not(unpack_function_input_reduced<T>::get(x))); }
  template<typename T, ENABLE_IF_BOOL_IMPL(!is_pointer_v<T>)> __HOST_DEVICE__ typename NotFcnal<T>::value_t Not(T const& x){ return NotFcnal<T>::eval(x); }
  /**
   * @brief Construct a lazy Not function node for autodiff.
   * @note  Host-only: function-graph objects (IvyRegularFunction) use
   *        virtual dispatch and RAII, which are incompatible with device code.
   *        Direct numerical evaluation (the non-pointer overload) is __HOST_DEVICE__.
   */
  template<typename T, ENABLE_IF_BOOL_IMPL(is_pointer_v<T>)>
  __HOST__ IvyThreadSafePtr_t<typename IvyNot<typename T::element_type>::base_t> Not(T const& x){
    constexpr std_ivy::IvyMemoryType def_mem_type = IvyMemoryHelpers::get_execution_default_memory();
    auto res = make_IvyThreadSafePtr<IvyNot<typename T::element_type>>(def_mem_type, nullptr, IvyNot(x));
    add_fcn_to_clients(res, x);
    return res;
  }


  /******************/
  /* 2D COMPARISONS */
  /******************/

  // EQUALITY
  template<typename T, typename U, typename domain_T, typename domain_U>
  __HOST_DEVICE__ EqualFcnal<T, U, domain_T, domain_U>::value_t EqualFcnal<T, U, domain_T, domain_U>::eval(T const& x, U const& y){ return x==y; }
  template<typename T, typename U>
  __HOST_DEVICE__ EqualFcnal<T, U, real_domain_tag, real_domain_tag>::value_t EqualFcnal<T, U, real_domain_tag, real_domain_tag>::eval(T const& x, U const& y){
    return Equal(unpack_function_input_reduced<T>::get(x), unpack_function_input_reduced<U>::get(y));
  }
  template<typename T, typename U>
  __HOST_DEVICE__ EqualFcnal<T, U, complex_domain_tag, complex_domain_tag>::value_t EqualFcnal<T, U, complex_domain_tag, complex_domain_tag>::eval(T const& x, U const& y){
    return value_t(Equal(unpack_function_input_reduced<T>::get(x).Re(), unpack_function_input_reduced<U>::get(y).Re()) && Equal(unpack_function_input_reduced<T>::get(x).Im(), unpack_function_input_reduced<U>::get(y).Im()));
  }
  template<typename T, typename U>
  __HOST_DEVICE__ EqualFcnal<T, U, arithmetic_domain_tag, real_domain_tag>::value_t EqualFcnal<T, U, arithmetic_domain_tag, real_domain_tag>::eval(T const& x, U const& y){
    return value_t(Equal(unpack_function_input_reduced<U>::get(y), x));
  }
  template<typename T, typename U>
  __HOST_DEVICE__ EqualFcnal<T, U, real_domain_tag, arithmetic_domain_tag>::value_t EqualFcnal<T, U, real_domain_tag, arithmetic_domain_tag>::eval(T const& x, U const& y){
    return value_t(Equal(unpack_function_input_reduced<T>::get(x), y));
  }
  template<typename T, typename U>
  __HOST_DEVICE__ EqualFcnal<T, U, arithmetic_domain_tag, complex_domain_tag>::value_t EqualFcnal<T, U, arithmetic_domain_tag, complex_domain_tag>::eval(T const& x, U const& y){
    return value_t(Equal(unpack_function_input_reduced<U>::get(y).Re(), x) && IsReal(unpack_function_input_reduced<U>::get(y)));
  }
  template<typename T, typename U>
  __HOST_DEVICE__ EqualFcnal<T, U, complex_domain_tag, arithmetic_domain_tag>::value_t EqualFcnal<T, U, complex_domain_tag, arithmetic_domain_tag>::eval(T const& x, U const& y){
    return value_t(Equal(unpack_function_input_reduced<T>::get(x).Re(), y) && IsReal(unpack_function_input_reduced<T>::get(x)));
  }
  template<typename T, typename U>
  __HOST_DEVICE__ EqualFcnal<T, U, real_domain_tag, complex_domain_tag>::value_t EqualFcnal<T, U, real_domain_tag, complex_domain_tag>::eval(T const& x, U const& y){
    return value_t(Equal(unpack_function_input_reduced<U>::get(y).Re(), unpack_function_input_reduced<T>::get(x)) && IsReal(unpack_function_input_reduced<U>::get(y)));
  }
  template<typename T, typename U>
  __HOST_DEVICE__ EqualFcnal<T, U, complex_domain_tag, real_domain_tag>::value_t EqualFcnal<T, U, complex_domain_tag, real_domain_tag>::eval(T const& x, U const& y){
    return value_t(Equal(unpack_function_input_reduced<T>::get(x).Re(), unpack_function_input_reduced<U>::get(y)) && IsReal(unpack_function_input_reduced<T>::get(x)));
  }
  template<typename T, typename U, ENABLE_IF_BOOL_IMPL(!is_pointer_v<T> && !is_pointer_v<U>)>
  __HOST_DEVICE__ typename EqualFcnal<T, U>::value_t Equal(T const& x, U const& y){ return EqualFcnal<T, U>::eval(x, y); }
  /**
   * @brief Construct a lazy Equal function node for autodiff.
   * @note  Host-only: function-graph objects (IvyRegularFunction) use
   *        virtual dispatch and RAII, which are incompatible with device code.
   *        Direct numerical evaluation (the non-pointer overload) is __HOST_DEVICE__.
   */
  template<typename T, typename U, ENABLE_IF_BOOL_IMPL(is_pointer_v<T> && is_pointer_v<U>)>
  __HOST__ IvyThreadSafePtr_t<typename IvyEqual<typename T::element_type, typename U::element_type>::base_t> Equal(T const& x, U const& y){
    constexpr std_ivy::IvyMemoryType def_mem_type = IvyMemoryHelpers::get_execution_default_memory();
    auto res = make_IvyThreadSafePtr<IvyEqual<typename T::element_type, typename U::element_type>>(def_mem_type, nullptr, IvyEqual(x, y));
    add_fcn_to_clients(res, x);
    add_fcn_to_clients(res, y);
    return res;
  }

  // OR
  template<typename T, typename U, typename domain_T, typename domain_U>
  __HOST_DEVICE__ OrFcnal<T, U, domain_T, domain_U>::value_t OrFcnal<T, U, domain_T, domain_U>::eval(T const& x, U const& y){ return x||y; }
  template<typename T, typename U>
  __HOST_DEVICE__ OrFcnal<T, U, real_domain_tag, real_domain_tag>::value_t OrFcnal<T, U, real_domain_tag, real_domain_tag>::eval(T const& x, U const& y){
    return Or(unpack_function_input_reduced<T>::get(x), unpack_function_input_reduced<U>::get(y));
  }
  template<typename T, typename U>
  __HOST_DEVICE__ OrFcnal<T, U, arithmetic_domain_tag, real_domain_tag>::value_t OrFcnal<T, U, arithmetic_domain_tag, real_domain_tag>::eval(T const& x, U const& y){
    return value_t(Or(unpack_function_input_reduced<U>::get(y), x));
  }
  template<typename T, typename U>
  __HOST_DEVICE__ OrFcnal<T, U, real_domain_tag, arithmetic_domain_tag>::value_t OrFcnal<T, U, real_domain_tag, arithmetic_domain_tag>::eval(T const& x, U const& y){
    return value_t(Or(unpack_function_input_reduced<T>::get(x), y));
  }
  template<typename T, typename U, ENABLE_IF_BOOL_IMPL(!is_pointer_v<T> && !is_pointer_v<U>)>
  __HOST_DEVICE__ typename OrFcnal<T, U>::value_t Or(T const& x, U const& y){ return OrFcnal<T, U>::eval(x, y); }
  /**
   * @brief Construct a lazy Or function node for autodiff.
   * @note  Host-only: function-graph objects (IvyRegularFunction) use
   *        virtual dispatch and RAII, which are incompatible with device code.
   *        Direct numerical evaluation (the non-pointer overload) is __HOST_DEVICE__.
   */
  template<typename T, typename U, ENABLE_IF_BOOL_IMPL(is_pointer_v<T> && is_pointer_v<U>)>
  __HOST__ IvyThreadSafePtr_t<typename IvyOr<typename T::element_type, typename U::element_type>::base_t> Or(T const& x, U const& y){
    constexpr std_ivy::IvyMemoryType def_mem_type = IvyMemoryHelpers::get_execution_default_memory();
    auto res = make_IvyThreadSafePtr<IvyOr<typename T::element_type, typename U::element_type>>(def_mem_type, nullptr, IvyOr(x, y));
    add_fcn_to_clients(res, x);
    add_fcn_to_clients(res, y);
    return res;
  }

  // XOR
  template<typename T, typename U, typename domain_T, typename domain_U>
  __HOST_DEVICE__ XorFcnal<T, U, domain_T, domain_U>::value_t XorFcnal<T, U, domain_T, domain_U>::eval(T const& x, U const& y){ return x^y; }
  template<typename T, typename U>
  __HOST_DEVICE__ XorFcnal<T, U, real_domain_tag, real_domain_tag>::value_t XorFcnal<T, U, real_domain_tag, real_domain_tag>::eval(T const& x, U const& y){
    return Xor(unpack_function_input_reduced<T>::get(x), unpack_function_input_reduced<U>::get(y));
  }
  template<typename T, typename U>
  __HOST_DEVICE__ XorFcnal<T, U, arithmetic_domain_tag, real_domain_tag>::value_t XorFcnal<T, U, arithmetic_domain_tag, real_domain_tag>::eval(T const& x, U const& y){
    return value_t(Xor(unpack_function_input_reduced<U>::get(y), x));
  }
  template<typename T, typename U>
  __HOST_DEVICE__ XorFcnal<T, U, real_domain_tag, arithmetic_domain_tag>::value_t XorFcnal<T, U, real_domain_tag, arithmetic_domain_tag>::eval(T const& x, U const& y){
    return value_t(Xor(unpack_function_input_reduced<T>::get(x), y));
  }
  template<typename T, typename U, ENABLE_IF_BOOL_IMPL(!is_pointer_v<T> && !is_pointer_v<U>)>
  __HOST_DEVICE__ typename XorFcnal<T, U>::value_t Xor(T const& x, U const& y){ return XorFcnal<T, U>::eval(x, y); }
  /**
   * @brief Construct a lazy Xor function node for autodiff.
   * @note  Host-only: function-graph objects (IvyRegularFunction) use
   *        virtual dispatch and RAII, which are incompatible with device code.
   *        Direct numerical evaluation (the non-pointer overload) is __HOST_DEVICE__.
   */
  template<typename T, typename U, ENABLE_IF_BOOL_IMPL(is_pointer_v<T> && is_pointer_v<U>)>
  __HOST__ IvyThreadSafePtr_t<typename IvyXor<typename T::element_type, typename U::element_type>::base_t> Xor(T const& x, U const& y){
    constexpr std_ivy::IvyMemoryType def_mem_type = IvyMemoryHelpers::get_execution_default_memory();
    auto res = make_IvyThreadSafePtr<IvyXor<typename T::element_type, typename U::element_type>>(def_mem_type, nullptr, IvyXor(x, y));
    add_fcn_to_clients(res, x);
    add_fcn_to_clients(res, y);
    return res;
  }
  template<typename T, typename U> __HOST_DEVICE__ auto XOr(T const& x, U const& y) -> decltype(Xor(x, y)){ return Xor(x, y); }

  // AND
  template<typename T, typename U, typename domain_T, typename domain_U>
  __HOST_DEVICE__ AndFcnal<T, U, domain_T, domain_U>::value_t AndFcnal<T, U, domain_T, domain_U>::eval(T const& x, U const& y){ return x&&y; }
  template<typename T, typename U>
  __HOST_DEVICE__ AndFcnal<T, U, real_domain_tag, real_domain_tag>::value_t AndFcnal<T, U, real_domain_tag, real_domain_tag>::eval(T const& x, U const& y){
    return And(unpack_function_input_reduced<T>::get(x), unpack_function_input_reduced<U>::get(y));
  }
  template<typename T, typename U>
  __HOST_DEVICE__ AndFcnal<T, U, arithmetic_domain_tag, real_domain_tag>::value_t AndFcnal<T, U, arithmetic_domain_tag, real_domain_tag>::eval(T const& x, U const& y){
    return value_t(And(unpack_function_input_reduced<U>::get(y), x));
  }
  template<typename T, typename U>
  __HOST_DEVICE__ AndFcnal<T, U, real_domain_tag, arithmetic_domain_tag>::value_t AndFcnal<T, U, real_domain_tag, arithmetic_domain_tag>::eval(T const& x, U const& y){
    return value_t(And(unpack_function_input_reduced<T>::get(x), y));
  }
  template<typename T, typename U, ENABLE_IF_BOOL_IMPL(!is_pointer_v<T> && !is_pointer_v<U>)>
  __HOST_DEVICE__ typename AndFcnal<T, U>::value_t And(T const& x, U const& y){ return AndFcnal<T, U>::eval(x, y); }
  /**
   * @brief Construct a lazy And function node for autodiff.
   * @note  Host-only: function-graph objects (IvyRegularFunction) use
   *        virtual dispatch and RAII, which are incompatible with device code.
   *        Direct numerical evaluation (the non-pointer overload) is __HOST_DEVICE__.
   */
  template<typename T, typename U, ENABLE_IF_BOOL_IMPL(is_pointer_v<T> && is_pointer_v<U>)>
  __HOST__ IvyThreadSafePtr_t<typename IvyAnd<typename T::element_type, typename U::element_type>::base_t> And(T const& x, U const& y){
    constexpr std_ivy::IvyMemoryType def_mem_type = IvyMemoryHelpers::get_execution_default_memory();
    auto res = make_IvyThreadSafePtr<IvyAnd<typename T::element_type, typename U::element_type>>(def_mem_type, nullptr, IvyAnd(x, y));
    add_fcn_to_clients(res, x);
    add_fcn_to_clients(res, y);
    return res;
  }

  // GREATER THAN
  template<typename T, typename U, typename domain_T, typename domain_U>
  __HOST_DEVICE__ GreaterThanFcnal<T, U, domain_T, domain_U>::value_t GreaterThanFcnal<T, U, domain_T, domain_U>::eval(T const& x, U const& y){ return x>y; }
  template<typename T, typename U>
  __HOST_DEVICE__ GreaterThanFcnal<T, U, real_domain_tag, real_domain_tag>::value_t GreaterThanFcnal<T, U, real_domain_tag, real_domain_tag>::eval(T const& x, U const& y){
    return GreaterThan(unpack_function_input_reduced<T>::get(x), unpack_function_input_reduced<U>::get(y));
  }
  template<typename T, typename U>
  __HOST_DEVICE__ GreaterThanFcnal<T, U, complex_domain_tag, complex_domain_tag>::value_t GreaterThanFcnal<T, U, complex_domain_tag, complex_domain_tag>::eval(T const& x, U const& y){
    return value_t(GreaterThan(unpack_function_input_reduced<T>::get(x).Re(), unpack_function_input_reduced<U>::get(y).Re()));
  }
  template<typename T, typename U>
  __HOST_DEVICE__ GreaterThanFcnal<T, U, arithmetic_domain_tag, real_domain_tag>::value_t GreaterThanFcnal<T, U, arithmetic_domain_tag, real_domain_tag>::eval(T const& x, U const& y){
    return value_t(GreaterThan(x, unpack_function_input_reduced<U>::get(y)));
  }
  template<typename T, typename U>
  __HOST_DEVICE__ GreaterThanFcnal<T, U, real_domain_tag, arithmetic_domain_tag>::value_t GreaterThanFcnal<T, U, real_domain_tag, arithmetic_domain_tag>::eval(T const& x, U const& y){
    return value_t(GreaterThan(unpack_function_input_reduced<T>::get(x), y));
  }
  template<typename T, typename U>
  __HOST_DEVICE__ GreaterThanFcnal<T, U, arithmetic_domain_tag, complex_domain_tag>::value_t GreaterThanFcnal<T, U, arithmetic_domain_tag, complex_domain_tag>::eval(T const& x, U const& y){
    return value_t(GreaterThan(x, unpack_function_input_reduced<U>::get(y).Re()));
  }
  template<typename T, typename U>
  __HOST_DEVICE__ GreaterThanFcnal<T, U, complex_domain_tag, arithmetic_domain_tag>::value_t GreaterThanFcnal<T, U, complex_domain_tag, arithmetic_domain_tag>::eval(T const& x, U const& y){
    return value_t(GreaterThan(unpack_function_input_reduced<T>::get(x).Re(), y));
  }
  template<typename T, typename U>
  __HOST_DEVICE__ GreaterThanFcnal<T, U, real_domain_tag, complex_domain_tag>::value_t GreaterThanFcnal<T, U, real_domain_tag, complex_domain_tag>::eval(T const& x, U const& y){
    return value_t(GreaterThan(unpack_function_input_reduced<T>::get(x), unpack_function_input_reduced<U>::get(y).Re()));
  }
  template<typename T, typename U>
  __HOST_DEVICE__ GreaterThanFcnal<T, U, complex_domain_tag, real_domain_tag>::value_t GreaterThanFcnal<T, U, complex_domain_tag, real_domain_tag>::eval(T const& x, U const& y){
    return value_t(GreaterThan(unpack_function_input_reduced<T>::get(x).Re(), unpack_function_input_reduced<U>::get(y)));
  }
  template<typename T, typename U, ENABLE_IF_BOOL_IMPL(!is_pointer_v<T> && !is_pointer_v<U>)>
  __HOST_DEVICE__ typename GreaterThanFcnal<T, U>::value_t GreaterThan(T const& x, U const& y){ return GreaterThanFcnal<T, U>::eval(x, y); }
  /**
   * @brief Construct a lazy GreaterThan function node for autodiff.
   * @note  Host-only: function-graph objects (IvyRegularFunction) use
   *        virtual dispatch and RAII, which are incompatible with device code.
   *        Direct numerical evaluation (the non-pointer overload) is __HOST_DEVICE__.
   */
  template<typename T, typename U, ENABLE_IF_BOOL_IMPL(is_pointer_v<T> && is_pointer_v<U>)>
  __HOST__ IvyThreadSafePtr_t<typename IvyGreaterThan<typename T::element_type, typename U::element_type>::base_t> GreaterThan(T const& x, U const& y){
    constexpr std_ivy::IvyMemoryType def_mem_type = IvyMemoryHelpers::get_execution_default_memory();
    auto res = make_IvyThreadSafePtr<IvyGreaterThan<typename T::element_type, typename U::element_type>>(def_mem_type, nullptr, IvyGreaterThan(x, y));
    add_fcn_to_clients(res, x);
    add_fcn_to_clients(res, y);
    return res;
  }
  template<typename T, typename U> __HOST_DEVICE__ auto GT(T const& x, U const& y) -> decltype(GreaterThan(x, y)){ return GreaterThan(x, y); }

  // LESS THAN
  template<typename T, typename U, typename domain_T, typename domain_U>
  __HOST_DEVICE__ LessThanFcnal<T, U, domain_T, domain_U>::value_t LessThanFcnal<T, U, domain_T, domain_U>::eval(T const& x, U const& y){ return x<y; }
  template<typename T, typename U>
  __HOST_DEVICE__ LessThanFcnal<T, U, real_domain_tag, real_domain_tag>::value_t LessThanFcnal<T, U, real_domain_tag, real_domain_tag>::eval(T const& x, U const& y){
    return LessThan(unpack_function_input_reduced<T>::get(x), unpack_function_input_reduced<U>::get(y));
  }
  template<typename T, typename U>
  __HOST_DEVICE__ LessThanFcnal<T, U, complex_domain_tag, complex_domain_tag>::value_t LessThanFcnal<T, U, complex_domain_tag, complex_domain_tag>::eval(T const& x, U const& y){
    return value_t(LessThan(unpack_function_input_reduced<T>::get(x).Re(), unpack_function_input_reduced<U>::get(y).Re()));
  }
  template<typename T, typename U>
  __HOST_DEVICE__ LessThanFcnal<T, U, arithmetic_domain_tag, real_domain_tag>::value_t LessThanFcnal<T, U, arithmetic_domain_tag, real_domain_tag>::eval(T const& x, U const& y){
    return value_t(LessThan(x, unpack_function_input_reduced<U>::get(y)));
  }
  template<typename T, typename U>
  __HOST_DEVICE__ LessThanFcnal<T, U, real_domain_tag, arithmetic_domain_tag>::value_t LessThanFcnal<T, U, real_domain_tag, arithmetic_domain_tag>::eval(T const& x, U const& y){
    return value_t(LessThan(unpack_function_input_reduced<T>::get(x), y));
  }
  template<typename T, typename U>
  __HOST_DEVICE__ LessThanFcnal<T, U, arithmetic_domain_tag, complex_domain_tag>::value_t LessThanFcnal<T, U, arithmetic_domain_tag, complex_domain_tag>::eval(T const& x, U const& y){
    return value_t(LessThan(x, unpack_function_input_reduced<U>::get(y).Re()));
  }
  template<typename T, typename U>
  __HOST_DEVICE__ LessThanFcnal<T, U, complex_domain_tag, arithmetic_domain_tag>::value_t LessThanFcnal<T, U, complex_domain_tag, arithmetic_domain_tag>::eval(T const& x, U const& y){
    return value_t(LessThan(unpack_function_input_reduced<T>::get(x).Re(), y));
  }
  template<typename T, typename U>
  __HOST_DEVICE__ LessThanFcnal<T, U, real_domain_tag, complex_domain_tag>::value_t LessThanFcnal<T, U, real_domain_tag, complex_domain_tag>::eval(T const& x, U const& y){
    return value_t(LessThan(unpack_function_input_reduced<T>::get(x), unpack_function_input_reduced<U>::get(y).Re()));
  }
  template<typename T, typename U>
  __HOST_DEVICE__ LessThanFcnal<T, U, complex_domain_tag, real_domain_tag>::value_t LessThanFcnal<T, U, complex_domain_tag, real_domain_tag>::eval(T const& x, U const& y){
    return value_t(LessThan(unpack_function_input_reduced<T>::get(x).Re(), unpack_function_input_reduced<U>::get(y)));
  }
  template<typename T, typename U, ENABLE_IF_BOOL_IMPL(!is_pointer_v<T> && !is_pointer_v<U>)>
  __HOST_DEVICE__ typename LessThanFcnal<T, U>::value_t LessThan(T const& x, U const& y){ return LessThanFcnal<T, U>::eval(x, y); }
  /**
   * @brief Construct a lazy LessThan function node for autodiff.
   * @note  Host-only: function-graph objects (IvyRegularFunction) use
   *        virtual dispatch and RAII, which are incompatible with device code.
   *        Direct numerical evaluation (the non-pointer overload) is __HOST_DEVICE__.
   */
  template<typename T, typename U, ENABLE_IF_BOOL_IMPL(is_pointer_v<T> && is_pointer_v<U>)>
  __HOST__ IvyThreadSafePtr_t<typename IvyLessThan<typename T::element_type, typename U::element_type>::base_t> LessThan(T const& x, U const& y){
    constexpr std_ivy::IvyMemoryType def_mem_type = IvyMemoryHelpers::get_execution_default_memory();
    auto res = make_IvyThreadSafePtr<IvyLessThan<typename T::element_type, typename U::element_type>>(def_mem_type, nullptr, IvyLessThan(x, y));
    add_fcn_to_clients(res, x);
    add_fcn_to_clients(res, y);
    return res;
  }
  template<typename T, typename U> __HOST_DEVICE__ auto LT(T const& x, U const& y) -> decltype(LessThan(x, y)){ return LessThan(x, y); }

  // GREATER THAN OR EQUAL TO
  template<typename T, typename U, typename domain_T, typename domain_U>
  __HOST_DEVICE__ GreaterOrEqualFcnal<T, U, domain_T, domain_U>::value_t GreaterOrEqualFcnal<T, U, domain_T, domain_U>::eval(T const& x, U const& y){ return x>=y; }
  template<typename T, typename U>
  __HOST_DEVICE__ GreaterOrEqualFcnal<T, U, real_domain_tag, real_domain_tag>::value_t GreaterOrEqualFcnal<T, U, real_domain_tag, real_domain_tag>::eval(T const& x, U const& y){
    return GreaterOrEqual(unpack_function_input_reduced<T>::get(x), unpack_function_input_reduced<U>::get(y));
  }
  template<typename T, typename U>
  __HOST_DEVICE__ GreaterOrEqualFcnal<T, U, complex_domain_tag, complex_domain_tag>::value_t GreaterOrEqualFcnal<T, U, complex_domain_tag, complex_domain_tag>::eval(T const& x, U const& y){
    return value_t(GreaterOrEqual(unpack_function_input_reduced<T>::get(x).Re(), unpack_function_input_reduced<U>::get(y).Re()));
  }
  template<typename T, typename U>
  __HOST_DEVICE__ GreaterOrEqualFcnal<T, U, arithmetic_domain_tag, real_domain_tag>::value_t GreaterOrEqualFcnal<T, U, arithmetic_domain_tag, real_domain_tag>::eval(T const& x, U const& y){
    return value_t(GreaterOrEqual(x, unpack_function_input_reduced<U>::get(y)));
  }
  template<typename T, typename U>
  __HOST_DEVICE__ GreaterOrEqualFcnal<T, U, real_domain_tag, arithmetic_domain_tag>::value_t GreaterOrEqualFcnal<T, U, real_domain_tag, arithmetic_domain_tag>::eval(T const& x, U const& y){
    return value_t(GreaterOrEqual(unpack_function_input_reduced<T>::get(x), y));
  }
  template<typename T, typename U>
  __HOST_DEVICE__ GreaterOrEqualFcnal<T, U, arithmetic_domain_tag, complex_domain_tag>::value_t GreaterOrEqualFcnal<T, U, arithmetic_domain_tag, complex_domain_tag>::eval(T const& x, U const& y){
    return value_t(GreaterOrEqual(x, unpack_function_input_reduced<U>::get(y).Re()));
  }
  template<typename T, typename U>
  __HOST_DEVICE__ GreaterOrEqualFcnal<T, U, complex_domain_tag, arithmetic_domain_tag>::value_t GreaterOrEqualFcnal<T, U, complex_domain_tag, arithmetic_domain_tag>::eval(T const& x, U const& y){
    return value_t(GreaterOrEqual(unpack_function_input_reduced<T>::get(x).Re(), y));
  }
  template<typename T, typename U>
  __HOST_DEVICE__ GreaterOrEqualFcnal<T, U, real_domain_tag, complex_domain_tag>::value_t GreaterOrEqualFcnal<T, U, real_domain_tag, complex_domain_tag>::eval(T const& x, U const& y){
    return value_t(GreaterOrEqual(unpack_function_input_reduced<T>::get(x), unpack_function_input_reduced<U>::get(y).Re()));
  }
  template<typename T, typename U>
  __HOST_DEVICE__ GreaterOrEqualFcnal<T, U, complex_domain_tag, real_domain_tag>::value_t GreaterOrEqualFcnal<T, U, complex_domain_tag, real_domain_tag>::eval(T const& x, U const& y){
    return value_t(GreaterOrEqual(unpack_function_input_reduced<T>::get(x).Re(), unpack_function_input_reduced<U>::get(y)));
  }
  template<typename T, typename U, ENABLE_IF_BOOL_IMPL(!is_pointer_v<T> && !is_pointer_v<U>)>
  __HOST_DEVICE__ typename GreaterOrEqualFcnal<T, U>::value_t GreaterOrEqual(T const& x, U const& y){ return GreaterOrEqualFcnal<T, U>::eval(x, y); }
  /**
   * @brief Construct a lazy GreaterOrEqual function node for autodiff.
   * @note  Host-only: function-graph objects (IvyRegularFunction) use
   *        virtual dispatch and RAII, which are incompatible with device code.
   *        Direct numerical evaluation (the non-pointer overload) is __HOST_DEVICE__.
   */
  template<typename T, typename U, ENABLE_IF_BOOL_IMPL(is_pointer_v<T> && is_pointer_v<U>)>
  __HOST__ IvyThreadSafePtr_t<typename IvyGreaterOrEqual<typename T::element_type, typename U::element_type>::base_t> GreaterOrEqual(T const& x, U const& y){
    constexpr std_ivy::IvyMemoryType def_mem_type = IvyMemoryHelpers::get_execution_default_memory();
    auto res = make_IvyThreadSafePtr<IvyGreaterOrEqual<typename T::element_type, typename U::element_type>>(def_mem_type, nullptr, IvyGreaterOrEqual(x, y));
    add_fcn_to_clients(res, x);
    add_fcn_to_clients(res, y);
    return res;
  }
  template<typename T, typename U> __HOST_DEVICE__ auto GE(T const& x, U const& y) -> decltype(GreaterOrEqual(x, y)){ return GreaterOrEqual(x, y); }
  template<typename T, typename U> __HOST_DEVICE__ auto GEQ(T const& x, U const& y) -> decltype(GreaterOrEqual(x, y)){ return GreaterOrEqual(x, y); }

  // LESS THAN OR EQUAL TO
  template<typename T, typename U, typename domain_T, typename domain_U>
  __HOST_DEVICE__ LessOrEqualFcnal<T, U, domain_T, domain_U>::value_t LessOrEqualFcnal<T, U, domain_T, domain_U>::eval(T const& x, U const& y){ return x<=y; }
  template<typename T, typename U>
  __HOST_DEVICE__ LessOrEqualFcnal<T, U, real_domain_tag, real_domain_tag>::value_t LessOrEqualFcnal<T, U, real_domain_tag, real_domain_tag>::eval(T const& x, U const& y){
    return LessOrEqual(unpack_function_input_reduced<T>::get(x), unpack_function_input_reduced<U>::get(y));
  }
  template<typename T, typename U>
  __HOST_DEVICE__ LessOrEqualFcnal<T, U, complex_domain_tag, complex_domain_tag>::value_t LessOrEqualFcnal<T, U, complex_domain_tag, complex_domain_tag>::eval(T const& x, U const& y){
    return value_t(LessOrEqual(unpack_function_input_reduced<T>::get(x).Re(), unpack_function_input_reduced<U>::get(y).Re()));
  }
  template<typename T, typename U>
  __HOST_DEVICE__ LessOrEqualFcnal<T, U, arithmetic_domain_tag, real_domain_tag>::value_t LessOrEqualFcnal<T, U, arithmetic_domain_tag, real_domain_tag>::eval(T const& x, U const& y){
    return value_t(LessOrEqual(x, unpack_function_input_reduced<U>::get(y)));
  }
  template<typename T, typename U>
  __HOST_DEVICE__ LessOrEqualFcnal<T, U, real_domain_tag, arithmetic_domain_tag>::value_t LessOrEqualFcnal<T, U, real_domain_tag, arithmetic_domain_tag>::eval(T const& x, U const& y){
    return value_t(LessOrEqual(unpack_function_input_reduced<T>::get(x), y));
  }
  template<typename T, typename U>
  __HOST_DEVICE__ LessOrEqualFcnal<T, U, arithmetic_domain_tag, complex_domain_tag>::value_t LessOrEqualFcnal<T, U, arithmetic_domain_tag, complex_domain_tag>::eval(T const& x, U const& y){
    return value_t(LessOrEqual(x, unpack_function_input_reduced<U>::get(y).Re()));
  }
  template<typename T, typename U>
  __HOST_DEVICE__ LessOrEqualFcnal<T, U, complex_domain_tag, arithmetic_domain_tag>::value_t LessOrEqualFcnal<T, U, complex_domain_tag, arithmetic_domain_tag>::eval(T const& x, U const& y){
    return value_t(LessOrEqual(unpack_function_input_reduced<T>::get(x).Re(), y));
  }
  template<typename T, typename U>
  __HOST_DEVICE__ LessOrEqualFcnal<T, U, real_domain_tag, complex_domain_tag>::value_t LessOrEqualFcnal<T, U, real_domain_tag, complex_domain_tag>::eval(T const& x, U const& y){
    return value_t(LessOrEqual(unpack_function_input_reduced<T>::get(x), unpack_function_input_reduced<U>::get(y).Re()));
  }
  template<typename T, typename U>
  __HOST_DEVICE__ LessOrEqualFcnal<T, U, complex_domain_tag, real_domain_tag>::value_t LessOrEqualFcnal<T, U, complex_domain_tag, real_domain_tag>::eval(T const& x, U const& y){
    return value_t(LessOrEqual(unpack_function_input_reduced<T>::get(x).Re(), unpack_function_input_reduced<U>::get(y)));
  }
  template<typename T, typename U, ENABLE_IF_BOOL_IMPL(!is_pointer_v<T> && !is_pointer_v<U>)>
  __HOST_DEVICE__ typename LessOrEqualFcnal<T, U>::value_t LessOrEqual(T const& x, U const& y){ return LessOrEqualFcnal<T, U>::eval(x, y); }
  /**
   * @brief Construct a lazy LessOrEqual function node for autodiff.
   * @note  Host-only: function-graph objects (IvyRegularFunction) use
   *        virtual dispatch and RAII, which are incompatible with device code.
   *        Direct numerical evaluation (the non-pointer overload) is __HOST_DEVICE__.
   */
  template<typename T, typename U, ENABLE_IF_BOOL_IMPL(is_pointer_v<T> && is_pointer_v<U>)>
  __HOST__ IvyThreadSafePtr_t<typename IvyLessOrEqual<typename T::element_type, typename U::element_type>::base_t> LessOrEqual(T const& x, U const& y){
    constexpr std_ivy::IvyMemoryType def_mem_type = IvyMemoryHelpers::get_execution_default_memory();
    auto res = make_IvyThreadSafePtr<IvyLessOrEqual<typename T::element_type, typename U::element_type>>(def_mem_type, nullptr, IvyLessOrEqual(x, y));
    add_fcn_to_clients(res, x);
    add_fcn_to_clients(res, y);
    return res;
  }
  template<typename T, typename U> __HOST_DEVICE__ auto LE(T const& x, U const& y) -> decltype(LessOrEqual(x, y)){ return LessOrEqual(x, y); }
  template<typename T, typename U> __HOST_DEVICE__ auto LEQ(T const& x, U const& y) -> decltype(LessOrEqual(x, y)){ return LessOrEqual(x, y); }

}


#endif
