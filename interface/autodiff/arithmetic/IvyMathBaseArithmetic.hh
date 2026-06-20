#ifndef IVYMATHBASEARITHMETIC_HH
#define IVYMATHBASEARITHMETIC_HH


#include "autodiff/basic_nodes/IvyScalar.h"
#include "autodiff/basic_nodes/IvyComplex.h"
#include "autodiff/basic_nodes/IvyQuaternion.h"
#include "autodiff/arithmetic/IvyMathConstOps.h"
#include "autodiff/arithmetic/IvyMathFunctionPrimitives.h"
#include "config/IvyAnnotationDispatchPolicy.h"

/**
 * @file IvyMathBaseArithmetic.hh
 * @brief Declarations for arithmetic autodiff functionals.
 *
 * Pointer-dependent gradient declarations in this header intentionally use
 * @c IVY_MATH_GRAPH_QUALIFIER, which maps to host-only execution. This keeps
 * lazy graph semantics available through @c func->gradient(x) while preventing
 * device-side instantiation of host-only graph code.
 */


namespace IvyMath{
  // Get real part of a variable
  template<typename T, typename domain_tag = get_domain_t<T>> struct RealFcnal{
    using value_t = convert_to_real_t<T>;
    static __INLINE_FCN_FORCE__ __HOST_DEVICE__ value_t eval(T const& x);
  };
  template<typename T> struct RealFcnal<T, complex_domain_tag>{
    using value_t = convert_to_real_t<T>;
    static __INLINE_FCN_FORCE__ __HOST_DEVICE__ value_t eval(T const& x);
  };
  template<typename T, ENABLE_IF_BOOL(!is_pointer_v<T>)> __INLINE_FCN_FORCE__ __HOST_DEVICE__ typename RealFcnal<T>::value_t Real(T const& x);

  // Get imaginary part of a variable
  template<typename T, typename domain_tag = get_domain_t<T>> struct ImagFcnal{
    using value_t = convert_to_real_t<T>;
    static __INLINE_FCN_FORCE__ __HOST_DEVICE__ constexpr value_t eval(T const& x);
  };
  template<typename T> struct ImagFcnal<T, complex_domain_tag>{
    using value_t = convert_to_real_t<T>;
    static __HOST_DEVICE__ value_t eval(T const& x);
  };
  template<typename T, ENABLE_IF_BOOL(!is_pointer_v<T>)> __INLINE_FCN_FORCE__ __HOST_DEVICE__ typename ImagFcnal<T>::value_t Imag(T const& x);

  // Test to check whether value is an integer
  template<typename T, typename domain_tag = get_domain_t<T>> struct IsIntegerFcnal{
    using value_t = bool;
    static __INLINE_FCN_FORCE__ __HOST_DEVICE__ constexpr value_t eval(T const& x);
  };
  template<typename T> struct IsIntegerFcnal<T, complex_domain_tag>{
    using value_t = bool;
    static __HOST_DEVICE__ value_t eval(T const& x);
  };
  template<typename T, ENABLE_IF_BOOL(!is_pointer_v<T>)> __INLINE_FCN_FORCE__ __HOST_DEVICE__ typename IsIntegerFcnal<T>::value_t IsInteger(T const& x);

  // Test to check whether value is real
  template<typename T, typename domain_tag = get_domain_t<T>> struct IsRealFcnal{
    using value_t = bool;
    static __INLINE_FCN_FORCE__ __HOST_DEVICE__ constexpr value_t eval(T const& x);
  };
  template<typename T> struct IsRealFcnal<T, complex_domain_tag>{
    using value_t = bool;
    static __HOST_DEVICE__ value_t eval(T const& x);
  };
  template<typename T, ENABLE_IF_BOOL(!is_pointer_v<T>)> __INLINE_FCN_FORCE__ __HOST_DEVICE__ typename IsRealFcnal<T>::value_t IsReal(T const& x);

  // Test to check whether value is imaginary
  template<typename T, typename domain_tag = get_domain_t<T>> struct IsImaginaryFcnal{
    using value_t = bool;
    static __INLINE_FCN_FORCE__ __HOST_DEVICE__ constexpr value_t eval(T const& x);
  };
  template<typename T> struct IsImaginaryFcnal<T, complex_domain_tag>{
    using value_t = bool;
    static __HOST_DEVICE__ value_t eval(T const& x);
  };
  template<typename T, ENABLE_IF_BOOL(!is_pointer_v<T>)> __INLINE_FCN_FORCE__ __HOST_DEVICE__ typename IsImaginaryFcnal<T>::value_t IsImaginary(T const& x);

  // NEGATION
  template<typename T, typename domain_tag = get_domain_t<T>> struct NegateFcnal{
    using value_t = unpacked_reduced_value_t<T>;
    using dtype_t = reduced_data_t<value_t>;
    static __INLINE_FCN_FORCE__ __HOST_DEVICE__ constexpr value_t eval(T const& x);
  };
  template<typename T> struct NegateFcnal<T, real_domain_tag>{
    using value_t = unpacked_reduced_value_t<T>;
    using dtype_t = reduced_data_t<value_t>;
    using fndtype_t = fundamental_data_t<value_t>;
    using grad_t = IvyScalarPtr_t<fndtype_t>;
    static __HOST_DEVICE__ value_t eval(T const& x);
    template<typename X_t>
    static __INLINE_FCN_FORCE__ IVY_MATH_GRAPH_QUALIFIER grad_t gradient(IvyThreadSafePtr_t<X_t> const& x);
  };
  template<typename T> struct NegateFcnal<T, complex_domain_tag>{
    using value_t = unpacked_reduced_value_t<T>;
    using dtype_t = reduced_data_t<value_t>;
    using fndtype_t = fundamental_data_t<value_t>;
    using grad_t = IvyComplexPtr_t<fndtype_t>;
    static __HOST_DEVICE__ value_t eval(T const& x);
    template<typename X_t>
    static __INLINE_FCN_FORCE__ IVY_MATH_GRAPH_QUALIFIER grad_t gradient(IvyThreadSafePtr_t<X_t> const& x);
  };
  /** @brief Quaternion negation. Linear (order-independent): d(-q) = -dq. */
  template<typename T> struct NegateFcnal<T, quaternion_domain_tag>{
    using value_t = unpacked_reduced_value_t<T>;
    using dtype_t = reduced_data_t<value_t>;
    using fndtype_t = fundamental_data_t<value_t>;
    using grad_t = IvyQuaternionPtr_t<fndtype_t>;
    static __HOST_DEVICE__ value_t eval(T const& x);
    template<typename X_t>
    static __INLINE_FCN_FORCE__ IVY_MATH_GRAPH_QUALIFIER grad_t gradient(IvyThreadSafePtr_t<X_t> const& x);
  };
  /** @brief Element-wise negation for tensor-domain inputs. */
  template<typename T> struct NegateFcnal<T, tensor_domain_tag>{
    using dtype_t = typename T::dtype_t;
    using value_t = T;
    using fndtype_t = fundamental_data_t<dtype_t>;
    static __HOST__ value_t eval(T const& x);
    static __HOST__ IvyThreadSafePtr_t<T> gradient(IvyThreadSafePtr_t<T> const& dep);
  };
  template<typename T> using IvyNegate = IvyRegularFunction_1D<T, NegateFcnal<unpack_if_function_t<T>>>;
  template<typename T, ENABLE_IF_BOOL(!is_pointer_v<T> && !is_tensor_v<T>)> __INLINE_FCN_FORCE__ __HOST_DEVICE__ typename NegateFcnal<T>::value_t Negate(T const& x);
  /// @brief Tensor-domain overload — @c __HOST__ only: tensor eval uses host-only STL constructs.
  template<typename T, ENABLE_IF_BOOL(!is_pointer_v<T> && is_tensor_v<T>)> __INLINE_FCN_FORCE__ __HOST__ typename NegateFcnal<T>::value_t Negate(T const& x);
  template<typename T, ENABLE_IF_BOOL(!is_arithmetic_v<T> && !is_pointer_v<T> && !is_tensor_v<T> && is_ivy_domain_v<T>)>
  __INLINE_FCN_RELAXED__ __HOST_DEVICE__ typename NegateFcnal<T>::value_t operator-(T const& x);
  /// @brief Tensor-domain unary minus — @c __HOST__ only: tensor eval uses host-only STL constructs.
  template<typename T, ENABLE_IF_BOOL(!is_arithmetic_v<T> && !is_pointer_v<T> && is_tensor_v<T>)>
  __INLINE_FCN_RELAXED__ __HOST__ typename NegateFcnal<T>::value_t operator-(T const& x);
  /**
   * @brief Construct a lazy Negate function node for autodiff.
   * @note  Host-only: function-graph objects (IvyRegularFunction) use
   *        virtual dispatch and RAII, which are incompatible with device code.
   *        Direct numerical evaluation (the non-pointer overload) is __HOST_DEVICE__.
   */
  template<typename T, ENABLE_IF_BOOL(is_pointer_v<T>)>
  __HOST__ IvyThreadSafePtr_t<typename IvyNegate<typename T::element_type>::base_t> Negate(T const& x);
  template<typename T, ENABLE_IF_BOOL(is_pointer_v<T>)>
  __HOST__ IvyThreadSafePtr_t<typename IvyNegate<typename T::element_type>::base_t> operator-(T const& x);

  // MULTIPLICATIVE INVERSE
  template<typename T, typename domain_tag = get_domain_t<T>> struct MultInverseFcnal{
    using value_t = unpacked_reduced_value_t<T>;
    using dtype_t = reduced_data_t<value_t>;
    using fndtype_t = fundamental_data_t<value_t>;
    static __INLINE_FCN_FORCE__ __HOST_DEVICE__ value_t eval(T const& x);
  };
  template<typename T> struct MultInverseFcnal<T, real_domain_tag>{
    using value_t = unpacked_reduced_value_t<T>;
    using dtype_t = reduced_data_t<value_t>;
    using fndtype_t = fundamental_data_t<value_t>;
    using grad_t = IvyThreadSafePtr_t<IvyFunction<value_t, real_domain_tag>>;
    static __HOST_DEVICE__ value_t eval(T const& x);
    template<typename X_t>
    static __INLINE_FCN_FORCE__ IVY_MATH_GRAPH_QUALIFIER grad_t gradient(IvyThreadSafePtr_t<X_t> const& x);
  };
  template<typename T> struct MultInverseFcnal<T, complex_domain_tag>{
    using value_t = unpacked_reduced_value_t<T>;
    using dtype_t = reduced_data_t<value_t>;
    using fndtype_t = fundamental_data_t<value_t>;
    using grad_t = IvyThreadSafePtr_t<IvyFunction<value_t, complex_domain_tag>>;
    static __HOST_DEVICE__ value_t eval(T const& x);
    template<typename X_t>
    static __INLINE_FCN_FORCE__ IVY_MATH_GRAPH_QUALIFIER grad_t gradient(IvyThreadSafePtr_t<X_t> const& x);
  };
  /**
   * @brief Quaternion multiplicative inverse, q^{-1} = conj(q)/|q|^2.
   *
   * NON-commutative and hence order-aware: the differential of the inverse is
   * d(q^{-1}) = -q^{-1} (dq) q^{-1}, a two-sided action that cannot be expressed
   * as a single left/right factor. It opts into the order-aware combiner.
   */
  template<typename T> struct MultInverseFcnal<T, quaternion_domain_tag>{
    using value_t = unpacked_reduced_value_t<T>;
    using dtype_t = reduced_data_t<value_t>;
    using fndtype_t = fundamental_data_t<value_t>;
    using grad_t = IvyThreadSafePtr_t<IvyFunction<value_t, quaternion_domain_tag>>;
    using order_aware_tag = void;
    static __HOST_DEVICE__ value_t eval(T const& x);
    template<typename X_t, typename G_t>
    static __HOST__ grad_t combine_gradient(IvyThreadSafePtr_t<X_t> const& dep, G_t const& grad_dep);
  };
  /** @brief Element-wise multiplicative inverse (1/x) for tensor inputs. */
  template<typename T> struct MultInverseFcnal<T, tensor_domain_tag>{
    using dtype_t = typename T::dtype_t;
    using value_t = T;
    using fndtype_t = fundamental_data_t<dtype_t>;
    static __HOST__ value_t eval(T const& x);
    static __HOST__ IvyThreadSafePtr_t<T> gradient(IvyThreadSafePtr_t<T> const& dep);
  };
  template<typename T> using IvyMultInverse = IvyRegularFunction_1D<T, MultInverseFcnal<unpack_if_function_t<T>>>;
  template<typename T, ENABLE_IF_BOOL(!is_pointer_v<T> && !is_tensor_v<T>)> __INLINE_FCN_FORCE__ __HOST_DEVICE__ typename MultInverseFcnal<T>::value_t MultInverse(T const& x);
  /// @brief Tensor-domain overload — @c __HOST__ only: tensor eval uses host-only STL constructs.
  template<typename T, ENABLE_IF_BOOL(!is_pointer_v<T> && is_tensor_v<T>)> __INLINE_FCN_FORCE__ __HOST__ typename MultInverseFcnal<T>::value_t MultInverse(T const& x);
  /**
   * @brief Construct a lazy MultInverse function node for autodiff.
   * @note  Host-only: function-graph objects (IvyRegularFunction) use
   *        virtual dispatch and RAII, which are incompatible with device code.
   *        Direct numerical evaluation (the non-pointer overload) is __HOST_DEVICE__.
   */
  template<typename T, ENABLE_IF_BOOL(is_pointer_v<T>)>
  __HOST__ IvyThreadSafePtr_t<typename IvyMultInverse<typename T::element_type>::base_t> MultInverse(T const& x);

  // SQUARE ROOT
  template<typename T, typename domain_tag = get_domain_t<T>> struct SqrtFcnal{
    using value_t = unpacked_reduced_value_t<T>;
    using dtype_t = reduced_data_t<value_t>;
    static __INLINE_FCN_FORCE__ __HOST_DEVICE__ value_t eval(T const& x);
  };
  template<typename T> struct SqrtFcnal<T, real_domain_tag>{
    using dtype_t = reduced_data_t<unpacked_reduced_value_t<T>>;
    using value_t = IvyScalar<dtype_t>;
    using fndtype_t = fundamental_data_t<value_t>;
    // The gradient is built from Pow(x, -1/2), a 2D function node whose precision_type is the
    // bare reduced data type (dtype_t), not the IvyScalar wrapper. grad_t must therefore name
    // IvyFunction<dtype_t, ...> to match the type Pow(...) actually returns.
    using grad_t = IvyThreadSafePtr_t<IvyFunction<dtype_t, real_domain_tag>>;
    static __HOST_DEVICE__ value_t eval(T const& x);
    template<typename X_t>
    static IVY_MATH_GRAPH_QUALIFIER grad_t gradient(IvyThreadSafePtr_t<X_t> const& x);
  };
  template<typename T> struct SqrtFcnal<T, complex_domain_tag>{
    using dtype_t = reduced_data_t<unpacked_reduced_value_t<T>>;
    using value_t = IvyComplex<dtype_t>;
    using fndtype_t = fundamental_data_t<value_t>;
    using grad_t = IvyThreadSafePtr_t<IvyFunction<value_t, complex_domain_tag>>;
    static __HOST_DEVICE__ value_t eval(T const& x);
    template<typename X_t>
    static IVY_MATH_GRAPH_QUALIFIER grad_t gradient(IvyThreadSafePtr_t<X_t> const& x);
  };
  /** @brief Element-wise square root for tensor inputs. */
  template<typename T> struct SqrtFcnal<T, tensor_domain_tag>{
    using dtype_t = typename T::dtype_t;
    using value_t = T;
    using fndtype_t = fundamental_data_t<dtype_t>;
    static __HOST__ value_t eval(T const& x);
    static __HOST__ IvyThreadSafePtr_t<T> gradient(IvyThreadSafePtr_t<T> const& dep);
  };
  template<typename T> using IvySqrt = IvyRegularFunction_1D<T, SqrtFcnal<unpack_if_function_t<T>>>;
  template<typename T, ENABLE_IF_BOOL(!is_pointer_v<T> && !is_tensor_v<T>)> __INLINE_FCN_FORCE__ __HOST_DEVICE__ typename SqrtFcnal<T>::value_t Sqrt(T const& x);
  /// @brief Tensor-domain overload — @c __HOST__ only: tensor eval uses host-only STL constructs.
  template<typename T, ENABLE_IF_BOOL(!is_pointer_v<T> && is_tensor_v<T>)> __INLINE_FCN_FORCE__ __HOST__ typename SqrtFcnal<T>::value_t Sqrt(T const& x);
  /**
   * @brief Construct a lazy Sqrt function node for autodiff.
   * @note  Host-only: function-graph objects (IvyRegularFunction) use
   *        virtual dispatch and RAII, which are incompatible with device code.
   *        Direct numerical evaluation (the non-pointer overload) is __HOST_DEVICE__.
   */
  template<typename T, ENABLE_IF_BOOL(is_pointer_v<T>)>
  __HOST__ IvyThreadSafePtr_t<typename IvySqrt<typename T::element_type>::base_t> Sqrt(T const& x);

  // ABSOLUTE VALUE
  template<typename T, typename domain_tag = get_domain_t<T>> struct AbsFcnal{
    using value_t = convert_to_real_t<T>;
    using dtype_t = reduced_data_t<value_t>;
    static __HOST_DEVICE__ value_t eval(T const& x);
  };
  template<typename T> struct AbsFcnal<T, real_domain_tag>{
    using value_t = convert_to_real_t<T>;
    using dtype_t = reduced_data_t<value_t>;
    static __HOST_DEVICE__ value_t eval(T const& x);
  };
  template<typename T> struct AbsFcnal<T, complex_domain_tag>{
    using value_t = convert_to_real_t<T>;
    using dtype_t = reduced_data_t<value_t>;
    static __HOST_DEVICE__ value_t eval(T const& x);
  };
  template<typename T> struct AbsFcnal<T, quaternion_domain_tag>{
    using value_t = convert_to_real_t<T>;
    using dtype_t = reduced_data_t<value_t>;
    static __HOST_DEVICE__ value_t eval(T const& x);
  };
  /**
   * @brief Element-wise absolute value for tensor inputs.
   * @note  The absolute value is not differentiable at zero; gradient is undefined.
   *        Declaring @c gradient_domain_tag = @c undefined_domain_tag prevents
   *        @c IvyRegularFunction_1D from instantiating the differentiable overload,
   *        which would fail to compile (AbsFcnal has no static gradient() member).
   */
  template<typename T> struct AbsFcnal<T, tensor_domain_tag>{
    using dtype_t = typename T::dtype_t;
    using value_t = T;
    using fndtype_t = fundamental_data_t<dtype_t>;
    using gradient_domain_tag = undefined_domain_tag;  ///< Abs is non-differentiable.
    static __HOST__ value_t eval(T const& x);
  };
  template<typename T, ENABLE_IF_BOOL(!is_pointer_v<T> && !is_tensor_v<T>)> __INLINE_FCN_FORCE__ __HOST_DEVICE__ typename AbsFcnal<T>::value_t Abs(T const& x);
  /// @brief Tensor-domain overload — @c __HOST__ only: tensor eval uses host-only STL constructs.
  template<typename T, ENABLE_IF_BOOL(!is_pointer_v<T> && is_tensor_v<T>)> __INLINE_FCN_FORCE__ __HOST__ typename AbsFcnal<T>::value_t Abs(T const& x);

  // COMPLEX PHASE
  template<typename T, typename domain_tag = get_domain_t<T>> struct PhaseFcnal{
    using value_t = convert_to_real_t<T>;
    using dtype_t = reduced_data_t<value_t>;
    static __HOST_DEVICE__ constexpr value_t eval(T const& x);
  };
  template<typename T> struct PhaseFcnal<T, complex_domain_tag>{
    using value_t = convert_to_real_t<T>;
    using dtype_t = reduced_data_t<value_t>;
    static __HOST_DEVICE__ value_t eval(T const& x);
  };
  template<typename T, ENABLE_IF_BOOL(!is_pointer_v<T>)> __INLINE_FCN_FORCE__ __HOST_DEVICE__ typename PhaseFcnal<T>::value_t Phase(T const& x);

  // CONJUGATION
  template<typename T, typename domain_tag = get_domain_t<T>> struct ConjugateFcnal{
    using value_t = unpacked_reduced_value_t<T>;
    using dtype_t = reduced_data_t<value_t>;
    static __INLINE_FCN_FORCE__ __HOST_DEVICE__ value_t eval(T const& x);
  };
  template<typename T, ENABLE_IF_BOOL(!is_pointer_v<T>)> __INLINE_FCN_FORCE__ __HOST_DEVICE__ typename ConjugateFcnal<T>::value_t Conjugate(T const& x);

  // EXPONENTIAL
  template<typename T, typename domain_tag = get_domain_t<T>> struct ExpFcnal{
    using value_t = unpacked_reduced_value_t<T>;
    using dtype_t = reduced_data_t<value_t>;
    static __INLINE_FCN_FORCE__ __HOST_DEVICE__ value_t eval(T const& x);
  };
  template<typename T> struct ExpFcnal<T, real_domain_tag>{
    using value_t = unpacked_reduced_value_t<T>;
    using dtype_t = reduced_data_t<value_t>;
    using fndtype_t = fundamental_data_t<value_t>;
    using grad_t = IvyThreadSafePtr_t<IvyFunction<value_t, real_domain_tag>>;
    static __HOST_DEVICE__ value_t eval(T const& x);
    template<typename X_t>
    static __INLINE_FCN_FORCE__ IVY_MATH_GRAPH_QUALIFIER grad_t gradient(IvyThreadSafePtr_t<X_t> const& x);
  };
  template<typename T> struct ExpFcnal<T, complex_domain_tag>{
    using value_t = unpacked_reduced_value_t<T>;
    using dtype_t = reduced_data_t<value_t>;
    using fndtype_t = fundamental_data_t<value_t>;
    using grad_t = IvyThreadSafePtr_t<IvyFunction<value_t, complex_domain_tag>>;
    static __HOST_DEVICE__ value_t eval(T const& x);
    template<typename X_t>
    static __INLINE_FCN_FORCE__ IVY_MATH_GRAPH_QUALIFIER grad_t gradient(IvyThreadSafePtr_t<X_t> const& x);
  };
  /** @brief Element-wise exponential for tensor-domain inputs. */
  template<typename T> struct ExpFcnal<T, tensor_domain_tag>{
    using dtype_t = typename T::dtype_t;
    using value_t = T;
    using fndtype_t = fundamental_data_t<dtype_t>;
    static __HOST__ value_t eval(T const& x);
    static __HOST__ IvyThreadSafePtr_t<T> gradient(IvyThreadSafePtr_t<T> const& dep);
  };
  template<typename T> using IvyExp = IvyRegularFunction_1D<T, ExpFcnal<unpack_if_function_t<T>>>;
  template<typename T, ENABLE_IF_BOOL(!is_pointer_v<T> && !is_tensor_v<T>)> __INLINE_FCN_FORCE__ __HOST_DEVICE__ typename ExpFcnal<T>::value_t Exp(T const& x);
  /// @brief Tensor-domain overload — @c __HOST__ only: tensor eval uses host-only STL constructs.
  template<typename T, ENABLE_IF_BOOL(!is_pointer_v<T> && is_tensor_v<T>)> __INLINE_FCN_FORCE__ __HOST__ typename ExpFcnal<T>::value_t Exp(T const& x);
  /**
   * @brief Construct a lazy Exp function node for autodiff.
   * @note  Host-only: function-graph objects (IvyRegularFunction) use
   *        virtual dispatch and RAII, which are incompatible with device code.
   *        Direct numerical evaluation (the non-pointer overload) is __HOST_DEVICE__.
   */
  template<typename T, ENABLE_IF_BOOL(is_pointer_v<T>)>
  __HOST__ IvyThreadSafePtr_t<typename IvyExp<typename T::element_type>::base_t> Exp(T const& x);

  // SUM (tensor -> scalar reduction)
  /**
   * @brief Differentiable sum reduction of a (real-domain) tensor to a scalar.
   *
   * value = Σ_i t_i ; the gradient is the same reduction applied to the operand
   * gradient: d(Sum t)/dvar = Σ_i ∂t_i/∂var (a scalar), which makes Sum a
   * first-class differentiation node wrt any scalar/function seed the tensor
   * depends on (and wrt the Sum node itself, giving 1). Supports both the
   * contiguous-cell (IvyTensor<IvyTensorScalarCell>) and array-of-pointers
   * (IvyTensor<IvyScalarPtr_t>) real representations. Complex-tensor reduction is
   * a future extension.
   */
  template<typename E> struct sum_elem_value{ using type = reduced_value_t<E>; };
  template<typename E, std_mem::IvyPointerType P> struct sum_elem_value<std_mem::IvyUnifiedPtr<E, P>>{ using type = reduced_value_t<E>; };
  template<typename T> struct SumFcnal{
    using elem_t = typename T::dtype_t;
    using elem_value_t = typename sum_elem_value<elem_t>::type;
    using fndtype_t = fundamental_data_t<elem_value_t>;
    using value_t = minimal_fcn_output_t<fndtype_t, get_domain_t<elem_value_t>, leaf_value_tag>;
    using dtype_t = reduced_data_t<value_t>;
    using reduction_tag = void; ///< marks this evaluator as a reduction (see evaluator_is_reduction)
    static __HOST__ value_t eval(T const& x);
  };
  template<typename T> using IvySum = IvyRegularFunction_1D<
    T, SumFcnal<unpack_if_function_t<T>>,
    reduced_data_t<typename SumFcnal<unpack_if_function_t<T>>::value_t>,
    get_domain_t<typename SumFcnal<unpack_if_function_t<T>>::value_t>,
    get_domain_t<typename SumFcnal<unpack_if_function_t<T>>::value_t>
  >;
  /// @brief Non-pointer (direct value) sum of a tensor. Host-only.
  template<typename T, ENABLE_IF_BOOL(!is_pointer_v<T> && is_tensor_v<T>)>
  __INLINE_FCN_FORCE__ __HOST__ typename SumFcnal<T>::value_t Sum(T const& x);
  /// @brief Construct a lazy Sum reduction node for autodiff. Host-only.
  template<typename T, ENABLE_IF_BOOL(is_pointer_v<T>)>
  __HOST__ IvyThreadSafePtr_t<typename IvySum<typename T::element_type>::base_t> Sum(T const& x);

  // LOG (NATURAL LOG)
  template<typename T, typename domain_tag = get_domain_t<T>> struct LogFcnal{
    using value_t = unpacked_reduced_value_t<T>;
    using dtype_t = reduced_data_t<value_t>;
    static __INLINE_FCN_FORCE__ __HOST_DEVICE__ value_t eval(T const& x);
  };
  template<typename T> struct LogFcnal<T, real_domain_tag>{
    using value_t = unpacked_reduced_value_t<T>;
    using dtype_t = reduced_data_t<value_t>;
    using fndtype_t = fundamental_data_t<value_t>;
    using grad_t = IvyThreadSafePtr_t<IvyFunction<value_t, real_domain_tag>>;
    static __HOST_DEVICE__ value_t eval(T const& x);
    template<typename X_t>
    static __INLINE_FCN_FORCE__ IVY_MATH_GRAPH_QUALIFIER grad_t gradient(IvyThreadSafePtr_t<X_t> const& x);
  };
  template<typename T> struct LogFcnal<T, complex_domain_tag>{
    using value_t = unpacked_reduced_value_t<T>;
    using dtype_t = reduced_data_t<value_t>;
    using fndtype_t = fundamental_data_t<value_t>;
    using grad_t = IvyThreadSafePtr_t<IvyFunction<value_t, complex_domain_tag>>;
    static __HOST_DEVICE__ value_t eval(T const& x);
    template<typename X_t>
    static __INLINE_FCN_FORCE__ IVY_MATH_GRAPH_QUALIFIER grad_t gradient(IvyThreadSafePtr_t<X_t> const& x);
  };
  /** @brief Element-wise logarithm for tensor-domain inputs. */
  template<typename T> struct LogFcnal<T, tensor_domain_tag>{
    using dtype_t = typename T::dtype_t;
    using value_t = T;
    using fndtype_t = fundamental_data_t<dtype_t>;
    static __HOST__ value_t eval(T const& x);
    static __HOST__ IvyThreadSafePtr_t<T> gradient(IvyThreadSafePtr_t<T> const& dep);
  };
  template<typename T> using IvyLog = IvyRegularFunction_1D<T, LogFcnal<unpack_if_function_t<T>>>;
  template<typename T, ENABLE_IF_BOOL(!is_pointer_v<T> && !is_tensor_v<T>)> __INLINE_FCN_FORCE__ __HOST_DEVICE__ typename LogFcnal<T>::value_t Log(T const& x);
  /// @brief Tensor-domain overload — @c __HOST__ only: tensor eval uses host-only STL constructs.
  template<typename T, ENABLE_IF_BOOL(!is_pointer_v<T> && is_tensor_v<T>)> __INLINE_FCN_FORCE__ __HOST__ typename LogFcnal<T>::value_t Log(T const& x);
  /**
   * @brief Construct a lazy Log function node for autodiff.
   * @note  Host-only: function-graph objects (IvyRegularFunction) use
   *        virtual dispatch and RAII, which are incompatible with device code.
   *        Direct numerical evaluation (the non-pointer overload) is __HOST_DEVICE__.
   */
  template<typename T, ENABLE_IF_BOOL(is_pointer_v<T>)>
  __HOST__ IvyThreadSafePtr_t<typename IvyLog<typename T::element_type>::base_t> Log(T const& x);

  // LOG10 (BASE=10 LOG)
  template<typename T, typename domain_tag = get_domain_t<T>> struct Log10Fcnal{
    using value_t = unpacked_reduced_value_t<T>;
    using dtype_t = reduced_data_t<value_t>;
    static __INLINE_FCN_FORCE__ __HOST_DEVICE__ value_t eval(T const& x);
  };
  template<typename T> struct Log10Fcnal<T, real_domain_tag>{
    using value_t = unpacked_reduced_value_t<T>;
    using dtype_t = reduced_data_t<value_t>;
    using fndtype_t = fundamental_data_t<value_t>;
    using grad_t = IvyThreadSafePtr_t<IvyFunction<value_t, real_domain_tag>>;
    static __INLINE_FCN_FORCE__ __HOST_DEVICE__ value_t eval(T const& x);
    template<typename X_t>
    static __INLINE_FCN_FORCE__ IVY_MATH_GRAPH_QUALIFIER grad_t gradient(IvyThreadSafePtr_t<X_t> const& x);
  };
  template<typename T> struct Log10Fcnal<T, complex_domain_tag>{
    using value_t = unpacked_reduced_value_t<T>;
    using dtype_t = reduced_data_t<value_t>;
    using fndtype_t = fundamental_data_t<value_t>;
    using grad_t = IvyThreadSafePtr_t<IvyFunction<value_t, complex_domain_tag>>;
    static __INLINE_FCN_FORCE__ __HOST_DEVICE__ value_t eval(T const& x);
    template<typename X_t>
    static __INLINE_FCN_FORCE__ IVY_MATH_GRAPH_QUALIFIER grad_t gradient(IvyThreadSafePtr_t<X_t> const& x);
  };
  template<typename T> using IvyLog10 = IvyRegularFunction_1D<T, Log10Fcnal<unpack_if_function_t<T>>>;
  template<typename T, ENABLE_IF_BOOL(!is_pointer_v<T>)> __INLINE_FCN_FORCE__ __HOST_DEVICE__ typename Log10Fcnal<T>::value_t Log10(T const& x);
  /**
   * @brief Construct a lazy Log10 function node for autodiff.
   * @note  Host-only: function-graph objects (IvyRegularFunction) use
   *        virtual dispatch and RAII, which are incompatible with device code.
   *        Direct numerical evaluation (the non-pointer overload) is __HOST_DEVICE__.
   */
  template<typename T, ENABLE_IF_BOOL(is_pointer_v<T>)>
  __HOST__ IvyThreadSafePtr_t<typename IvyLog10<typename T::element_type>::base_t> Log10(T const& x);

  // SINE
  template<typename T, typename domain_tag = get_domain_t<T>> struct SinFcnal{
    using value_t = unpacked_reduced_value_t<T>;
    using dtype_t = reduced_data_t<value_t>;
    static __INLINE_FCN_FORCE__ __HOST_DEVICE__ value_t eval(T const& x);
  };
  template<typename T> struct SinFcnal<T, real_domain_tag>{
    using value_t = unpacked_reduced_value_t<T>;
    using dtype_t = reduced_data_t<value_t>;
    using fndtype_t = fundamental_data_t<value_t>;
    using grad_t = IvyThreadSafePtr_t<IvyFunction<value_t, real_domain_tag>>;
    static __HOST_DEVICE__ value_t eval(T const& x);
    template<typename X_t>
    static __INLINE_FCN_FORCE__ IVY_MATH_GRAPH_QUALIFIER grad_t gradient(IvyThreadSafePtr_t<X_t> const& x);
  };
  template<typename T> struct SinFcnal<T, complex_domain_tag>{
    using value_t = unpacked_reduced_value_t<T>;
    using dtype_t = reduced_data_t<value_t>;
    using fndtype_t = fundamental_data_t<value_t>;
    using grad_t = IvyThreadSafePtr_t<IvyFunction<value_t, complex_domain_tag>>;
    static __HOST_DEVICE__ value_t eval(T const& x);
    template<typename X_t>
    static __INLINE_FCN_FORCE__ IVY_MATH_GRAPH_QUALIFIER grad_t gradient(IvyThreadSafePtr_t<X_t> const& x);
  };
  /** @brief Element-wise sine for tensor-domain inputs. */
  template<typename T> struct SinFcnal<T, tensor_domain_tag>{
    using dtype_t = typename T::dtype_t;
    using value_t = T;
    using fndtype_t = fundamental_data_t<dtype_t>;
    static __HOST__ value_t eval(T const& x);
    static __HOST__ IvyThreadSafePtr_t<T> gradient(IvyThreadSafePtr_t<T> const& dep);
  };
  template<typename T> using IvySin = IvyRegularFunction_1D<T, SinFcnal<unpack_if_function_t<T>>>;
  template<typename T, ENABLE_IF_BOOL(!is_pointer_v<T> && !is_tensor_v<T>)> __INLINE_FCN_FORCE__ __HOST_DEVICE__ typename SinFcnal<T>::value_t Sin(T const& x);
  /// @brief Tensor-domain overload — @c __HOST__ only: tensor eval uses host-only STL constructs.
  template<typename T, ENABLE_IF_BOOL(!is_pointer_v<T> && is_tensor_v<T>)> __INLINE_FCN_FORCE__ __HOST__ typename SinFcnal<T>::value_t Sin(T const& x);
  /**
   * @brief Construct a lazy Sin function node for autodiff.
   * @note  Host-only: function-graph objects (IvyRegularFunction) use
   *        virtual dispatch and RAII, which are incompatible with device code.
   *        Direct numerical evaluation (the non-pointer overload) is __HOST_DEVICE__.
   */
  template<typename T, ENABLE_IF_BOOL(is_pointer_v<T>)>
  __HOST__ IvyThreadSafePtr_t<typename IvySin<typename T::element_type>::base_t> Sin(T const& x);

  // COSINE
  template<typename T, typename domain_tag = get_domain_t<T>> struct CosFcnal{
    using value_t = unpacked_reduced_value_t<T>;
    using dtype_t = reduced_data_t<value_t>;
    static __INLINE_FCN_FORCE__ __HOST_DEVICE__ value_t eval(T const& x);
  };
  template<typename T> struct CosFcnal<T, real_domain_tag>{
    using value_t = unpacked_reduced_value_t<T>;
    using dtype_t = reduced_data_t<value_t>;
    using fndtype_t = fundamental_data_t<value_t>;
    using grad_t = IvyThreadSafePtr_t<IvyFunction<value_t, real_domain_tag>>;
    static __HOST_DEVICE__ value_t eval(T const& x);
    template<typename X_t>
    static __INLINE_FCN_FORCE__ IVY_MATH_GRAPH_QUALIFIER grad_t gradient(IvyThreadSafePtr_t<X_t> const& x);
  };
  template<typename T> struct CosFcnal<T, complex_domain_tag>{
    using value_t = unpacked_reduced_value_t<T>;
    using dtype_t = reduced_data_t<value_t>;
    using fndtype_t = fundamental_data_t<value_t>;
    using grad_t = IvyThreadSafePtr_t<IvyFunction<value_t, complex_domain_tag>>;
    static __HOST_DEVICE__ value_t eval(T const& x);
    template<typename X_t>
    static __INLINE_FCN_FORCE__ IVY_MATH_GRAPH_QUALIFIER grad_t gradient(IvyThreadSafePtr_t<X_t> const& x);
  };
  /** @brief Element-wise cosine for tensor-domain inputs. */
  template<typename T> struct CosFcnal<T, tensor_domain_tag>{
    using dtype_t = typename T::dtype_t;
    using value_t = T;
    using fndtype_t = fundamental_data_t<dtype_t>;
    static __HOST__ value_t eval(T const& x);
    static __HOST__ IvyThreadSafePtr_t<T> gradient(IvyThreadSafePtr_t<T> const& dep);
  };
  template<typename T> using IvyCos = IvyRegularFunction_1D<T, CosFcnal<unpack_if_function_t<T>>>;
  template<typename T, ENABLE_IF_BOOL(!is_pointer_v<T> && !is_tensor_v<T>)> __INLINE_FCN_FORCE__ __HOST_DEVICE__ typename CosFcnal<T>::value_t Cos(T const& x);
  /// @brief Tensor-domain overload — @c __HOST__ only: tensor eval uses host-only STL constructs.
  template<typename T, ENABLE_IF_BOOL(!is_pointer_v<T> && is_tensor_v<T>)> __INLINE_FCN_FORCE__ __HOST__ typename CosFcnal<T>::value_t Cos(T const& x);
  /**
   * @brief Construct a lazy Cos function node for autodiff.
   * @note  Host-only: function-graph objects (IvyRegularFunction) use
   *        virtual dispatch and RAII, which are incompatible with device code.
   *        Direct numerical evaluation (the non-pointer overload) is __HOST_DEVICE__.
   */
  template<typename T, ENABLE_IF_BOOL(is_pointer_v<T>)>
  __HOST__ IvyThreadSafePtr_t<typename IvyCos<typename T::element_type>::base_t> Cos(T const& x);

  // TANGENT
  template<typename T, typename domain_tag = get_domain_t<T>> struct TanFcnal{
    using value_t = unpacked_reduced_value_t<T>;
    using dtype_t = reduced_data_t<value_t>;
    using fndtype_t = fundamental_data_t<value_t>;
    using grad_t = IvyThreadSafePtr_t<IvyFunction<value_t, domain_tag>>;
    static __INLINE_FCN_FORCE__ __HOST_DEVICE__ value_t eval(T const& x);
    template<typename X_t>
    static __INLINE_FCN_FORCE__ IVY_MATH_GRAPH_QUALIFIER grad_t gradient(IvyThreadSafePtr_t<X_t> const& x);
  };
  /** @brief Element-wise tangent for tensor inputs. */
  template<typename T> struct TanFcnal<T, tensor_domain_tag>{
    using dtype_t = typename T::dtype_t;
    using value_t = T;
    using fndtype_t = fundamental_data_t<dtype_t>;
    static __HOST__ value_t eval(T const& x);
    static __HOST__ IvyThreadSafePtr_t<T> gradient(IvyThreadSafePtr_t<T> const& dep);
  };
  template<typename T> using IvyTan = IvyRegularFunction_1D<T, TanFcnal<unpack_if_function_t<T>>>;
  template<typename T, ENABLE_IF_BOOL(!is_pointer_v<T> && !is_tensor_v<T>)> __INLINE_FCN_FORCE__ __HOST_DEVICE__ typename TanFcnal<T>::value_t Tan(T const& x);
  /// @brief Tensor-domain overload — @c __HOST__ only: tensor eval uses host-only STL constructs.
  template<typename T, ENABLE_IF_BOOL(!is_pointer_v<T> && is_tensor_v<T>)> __INLINE_FCN_FORCE__ __HOST__ typename TanFcnal<T>::value_t Tan(T const& x);
  /**
   * @brief Construct a lazy Tan function node for autodiff.
   * @note  Host-only: function-graph objects (IvyRegularFunction) use
   *        virtual dispatch and RAII, which are incompatible with device code.
   *        Direct numerical evaluation (the non-pointer overload) is __HOST_DEVICE__.
   */
  template<typename T, ENABLE_IF_BOOL(is_pointer_v<T>)>
  __HOST__ IvyThreadSafePtr_t<typename IvyTan<typename T::element_type>::base_t> Tan(T const& x);

  // SECANT
  template<typename T, typename domain_tag = get_domain_t<T>> struct SecFcnal{
    using value_t = unpacked_reduced_value_t<T>;
    using dtype_t = reduced_data_t<value_t>;
    using fndtype_t = fundamental_data_t<value_t>;
    using grad_t = IvyThreadSafePtr_t<IvyFunction<value_t, domain_tag>>;
    static __INLINE_FCN_FORCE__ __HOST_DEVICE__ value_t eval(T const& x);
    template<typename X_t>
    static __INLINE_FCN_FORCE__ IVY_MATH_GRAPH_QUALIFIER grad_t gradient(IvyThreadSafePtr_t<X_t> const& x);
  };
  template<typename T> using IvySec = IvyRegularFunction_1D<T, SecFcnal<unpack_if_function_t<T>>>;
  template<typename T, ENABLE_IF_BOOL(!is_pointer_v<T>)> __INLINE_FCN_FORCE__ __HOST_DEVICE__ typename SecFcnal<T>::value_t Sec(T const& x);
  /**
   * @brief Construct a lazy Sec function node for autodiff.
   * @note  Host-only: function-graph objects (IvyRegularFunction) use
   *        virtual dispatch and RAII, which are incompatible with device code.
   *        Direct numerical evaluation (the non-pointer overload) is __HOST_DEVICE__.
   */
  template<typename T, ENABLE_IF_BOOL(is_pointer_v<T>)>
  __HOST__ IvyThreadSafePtr_t<typename IvySec<typename T::element_type>::base_t> Sec(T const& x);

  // COSECANT
  template<typename T, typename domain_tag = get_domain_t<T>> struct CscFcnal{
    using value_t = unpacked_reduced_value_t<T>;
    using dtype_t = reduced_data_t<value_t>;
    using fndtype_t = fundamental_data_t<value_t>;
    using grad_t = IvyThreadSafePtr_t<IvyFunction<value_t, domain_tag>>;
    static __INLINE_FCN_FORCE__ __HOST_DEVICE__ value_t eval(T const& x);
    template<typename X_t>
    static __INLINE_FCN_FORCE__ IVY_MATH_GRAPH_QUALIFIER grad_t gradient(IvyThreadSafePtr_t<X_t> const& x);
  };
  template<typename T> using IvyCsc = IvyRegularFunction_1D<T, CscFcnal<unpack_if_function_t<T>>>;
  template<typename T, ENABLE_IF_BOOL(!is_pointer_v<T>)> __INLINE_FCN_FORCE__ __HOST_DEVICE__ typename CscFcnal<T>::value_t Csc(T const& x);
  /**
   * @brief Construct a lazy Csc function node for autodiff.
   * @note  Host-only: function-graph objects (IvyRegularFunction) use
   *        virtual dispatch and RAII, which are incompatible with device code.
   *        Direct numerical evaluation (the non-pointer overload) is __HOST_DEVICE__.
   */
  template<typename T, ENABLE_IF_BOOL(is_pointer_v<T>)>
  __HOST__ IvyThreadSafePtr_t<typename IvyCsc<typename T::element_type>::base_t> Csc(T const& x);

  // COTANGENT
  template<typename T, typename domain_tag = get_domain_t<T>> struct CotFcnal{
    using value_t = unpacked_reduced_value_t<T>;
    using dtype_t = reduced_data_t<value_t>;
    using fndtype_t = fundamental_data_t<value_t>;
    using grad_t = IvyThreadSafePtr_t<IvyFunction<value_t, domain_tag>>;
    static __INLINE_FCN_FORCE__ __HOST_DEVICE__ value_t eval(T const& x);
    template<typename X_t>
    static __INLINE_FCN_FORCE__ IVY_MATH_GRAPH_QUALIFIER grad_t gradient(IvyThreadSafePtr_t<X_t> const& x);
  };
  /** @brief Element-wise cotangent for tensor inputs. */
  template<typename T> struct CotFcnal<T, tensor_domain_tag>{
    using dtype_t = typename T::dtype_t;
    using value_t = T;
    using fndtype_t = fundamental_data_t<dtype_t>;
    static __HOST__ value_t eval(T const& x);
    static __HOST__ IvyThreadSafePtr_t<T> gradient(IvyThreadSafePtr_t<T> const& dep);
  };
  template<typename T> using IvyCot = IvyRegularFunction_1D<T, CotFcnal<unpack_if_function_t<T>>>;
  template<typename T, ENABLE_IF_BOOL(!is_pointer_v<T> && !is_tensor_v<T>)> __INLINE_FCN_FORCE__ __HOST_DEVICE__ typename CotFcnal<T>::value_t Cot(T const& x);
  /// @brief Tensor-domain overload — @c __HOST__ only: tensor eval uses host-only STL constructs.
  template<typename T, ENABLE_IF_BOOL(!is_pointer_v<T> && is_tensor_v<T>)> __INLINE_FCN_FORCE__ __HOST__ typename CotFcnal<T>::value_t Cot(T const& x);
  /**
   * @brief Construct a lazy Cot function node for autodiff.
   * @note  Host-only: function-graph objects (IvyRegularFunction) use
   *        virtual dispatch and RAII, which are incompatible with device code.
   *        Direct numerical evaluation (the non-pointer overload) is __HOST_DEVICE__.
   */
  template<typename T, ENABLE_IF_BOOL(is_pointer_v<T>)>
  __HOST__ IvyThreadSafePtr_t<typename IvyCot<typename T::element_type>::base_t> Cot(T const& x);

  // SINH
  template<typename T, typename domain_tag = get_domain_t<T>> struct SinHFcnal{
    using value_t = unpacked_reduced_value_t<T>;
    using dtype_t = reduced_data_t<value_t>;
    static __INLINE_FCN_FORCE__ __HOST_DEVICE__ value_t eval(T const& x);
  };
  template<typename T> struct SinHFcnal<T, real_domain_tag>{
    using value_t = unpacked_reduced_value_t<T>;
    using dtype_t = reduced_data_t<value_t>;
    using fndtype_t = fundamental_data_t<value_t>;
    using grad_t = IvyThreadSafePtr_t<IvyFunction<value_t, real_domain_tag>>;
    static __HOST_DEVICE__ value_t eval(T const& x);
    template<typename X_t>
    static __INLINE_FCN_FORCE__ IVY_MATH_GRAPH_QUALIFIER grad_t gradient(IvyThreadSafePtr_t<X_t> const& x);
  };
  template<typename T> struct SinHFcnal<T, complex_domain_tag>{
    using value_t = unpacked_reduced_value_t<T>;
    using dtype_t = reduced_data_t<value_t>;
    using fndtype_t = fundamental_data_t<value_t>;
    using grad_t = IvyThreadSafePtr_t<IvyFunction<value_t, complex_domain_tag>>;
    static __HOST_DEVICE__ value_t eval(T const& x);
    template<typename X_t>
    static __INLINE_FCN_FORCE__ IVY_MATH_GRAPH_QUALIFIER grad_t gradient(IvyThreadSafePtr_t<X_t> const& x);
  };
  /** @brief Element-wise hyperbolic sine for tensor inputs. */
  template<typename T> struct SinHFcnal<T, tensor_domain_tag>{
    using dtype_t = typename T::dtype_t;
    using value_t = T;
    using fndtype_t = fundamental_data_t<dtype_t>;
    static __HOST__ value_t eval(T const& x);
    static __HOST__ IvyThreadSafePtr_t<T> gradient(IvyThreadSafePtr_t<T> const& dep);
  };
  template<typename T> using IvySinH = IvyRegularFunction_1D<T, SinHFcnal<unpack_if_function_t<T>>>;
  template<typename T, ENABLE_IF_BOOL(!is_pointer_v<T> && !is_tensor_v<T>)> __INLINE_FCN_FORCE__ __HOST_DEVICE__ typename SinHFcnal<T>::value_t SinH(T const& x);
  /// @brief Tensor-domain overload — @c __HOST__ only: tensor eval uses host-only STL constructs.
  template<typename T, ENABLE_IF_BOOL(!is_pointer_v<T> && is_tensor_v<T>)> __INLINE_FCN_FORCE__ __HOST__ typename SinHFcnal<T>::value_t SinH(T const& x);
  /**
   * @brief Construct a lazy SinH function node for autodiff.
   * @note  Host-only: function-graph objects (IvyRegularFunction) use
   *        virtual dispatch and RAII, which are incompatible with device code.
   *        Direct numerical evaluation (the non-pointer overload) is __HOST_DEVICE__.
   */
  template<typename T, ENABLE_IF_BOOL(is_pointer_v<T>)>
  __HOST__ IvyThreadSafePtr_t<typename IvySinH<typename T::element_type>::base_t> SinH(T const& x);

  // COSH
  template<typename T, typename domain_tag = get_domain_t<T>> struct CosHFcnal{
    using value_t = unpacked_reduced_value_t<T>;
    using dtype_t = reduced_data_t<value_t>;
    static __INLINE_FCN_FORCE__ __HOST_DEVICE__ value_t eval(T const& x);
  };
  template<typename T> struct CosHFcnal<T, real_domain_tag>{
    using value_t = unpacked_reduced_value_t<T>;
    using dtype_t = reduced_data_t<value_t>;
    using fndtype_t = fundamental_data_t<value_t>;
    using grad_t = IvyThreadSafePtr_t<IvyFunction<value_t, real_domain_tag>>;
    static __HOST_DEVICE__ value_t eval(T const& x);
    template<typename X_t>
    static __INLINE_FCN_FORCE__ IVY_MATH_GRAPH_QUALIFIER grad_t gradient(IvyThreadSafePtr_t<X_t> const& x);
  };
  template<typename T> struct CosHFcnal<T, complex_domain_tag>{
    using value_t = unpacked_reduced_value_t<T>;
    using dtype_t = reduced_data_t<value_t>;
    using fndtype_t = fundamental_data_t<value_t>;
    using grad_t = IvyThreadSafePtr_t<IvyFunction<value_t, complex_domain_tag>>;
    static __HOST_DEVICE__ value_t eval(T const& x);
    template<typename X_t>
    static __INLINE_FCN_FORCE__ IVY_MATH_GRAPH_QUALIFIER grad_t gradient(IvyThreadSafePtr_t<X_t> const& x);
  };
  /** @brief Element-wise hyperbolic cosine for tensor inputs. */
  template<typename T> struct CosHFcnal<T, tensor_domain_tag>{
    using dtype_t = typename T::dtype_t;
    using value_t = T;
    using fndtype_t = fundamental_data_t<dtype_t>;
    static __HOST__ value_t eval(T const& x);
    static __HOST__ IvyThreadSafePtr_t<T> gradient(IvyThreadSafePtr_t<T> const& dep);
  };
  template<typename T> using IvyCosH = IvyRegularFunction_1D<T, CosHFcnal<unpack_if_function_t<T>>>;
  template<typename T, ENABLE_IF_BOOL(!is_pointer_v<T> && !is_tensor_v<T>)> __INLINE_FCN_FORCE__ __HOST_DEVICE__ typename CosHFcnal<T>::value_t CosH(T const& x);
  /// @brief Tensor-domain overload — @c __HOST__ only: tensor eval uses host-only STL constructs.
  template<typename T, ENABLE_IF_BOOL(!is_pointer_v<T> && is_tensor_v<T>)> __INLINE_FCN_FORCE__ __HOST__ typename CosHFcnal<T>::value_t CosH(T const& x);
  /**
   * @brief Construct a lazy CosH function node for autodiff.
   * @note  Host-only: function-graph objects (IvyRegularFunction) use
   *        virtual dispatch and RAII, which are incompatible with device code.
   *        Direct numerical evaluation (the non-pointer overload) is __HOST_DEVICE__.
   */
  template<typename T, ENABLE_IF_BOOL(is_pointer_v<T>)>
  __HOST__ IvyThreadSafePtr_t<typename IvyCosH<typename T::element_type>::base_t> CosH(T const& x);

  // ERF
  template<typename T, typename domain_tag = get_domain_t<T>> struct ErfFcnal{
    using value_t = unpacked_reduced_value_t<T>;
    using dtype_t = reduced_data_t<value_t>;
    static __INLINE_FCN_FORCE__ __HOST_DEVICE__ value_t eval(T const& x);
  };
  template<typename T> struct ErfFcnal<T, real_domain_tag>{
    using value_t = unpacked_reduced_value_t<T>;
    using dtype_t = reduced_data_t<value_t>;
    using fndtype_t = fundamental_data_t<value_t>;
    using grad_t = IvyThreadSafePtr_t<IvyFunction<value_t, real_domain_tag>>;
    static __HOST_DEVICE__ value_t eval(T const& x);
    template<typename X_t>
    static __INLINE_FCN_FORCE__ IVY_MATH_GRAPH_QUALIFIER grad_t gradient(IvyThreadSafePtr_t<X_t> const& x);
  };
  template<typename T> struct ErfFcnal<T, complex_domain_tag>{
    using value_t = unpacked_reduced_value_t<T>;
    using dtype_t = reduced_data_t<value_t>;
    using fndtype_t = fundamental_data_t<value_t>;
    using grad_t = IvyThreadSafePtr_t<IvyFunction<value_t, complex_domain_tag>>;
    static __HOST_DEVICE__ value_t eval(T const& x);
    template<typename X_t>
    static __INLINE_FCN_FORCE__ IVY_MATH_GRAPH_QUALIFIER grad_t gradient(IvyThreadSafePtr_t<X_t> const& x);
  };
  /** @brief Element-wise error function for tensor inputs. */
  template<typename T> struct ErfFcnal<T, tensor_domain_tag>{
    using dtype_t = typename T::dtype_t;
    using value_t = T;
    using fndtype_t = fundamental_data_t<dtype_t>;
    static __HOST__ value_t eval(T const& x);
    static __HOST__ IvyThreadSafePtr_t<T> gradient(IvyThreadSafePtr_t<T> const& dep);
  };
  template<typename T> using IvyErf = IvyRegularFunction_1D<T, ErfFcnal<unpack_if_function_t<T>>>;
  template<typename T, ENABLE_IF_BOOL(!is_pointer_v<T> && !is_tensor_v<T>)> __INLINE_FCN_FORCE__ __HOST_DEVICE__ typename ErfFcnal<T>::value_t Erf(T const& x);
  /// @brief Tensor-domain overload — @c __HOST__ only: tensor eval uses host-only STL constructs.
  template<typename T, ENABLE_IF_BOOL(!is_pointer_v<T> && is_tensor_v<T>)> __INLINE_FCN_FORCE__ __HOST__ typename ErfFcnal<T>::value_t Erf(T const& x);
  /**
   * @brief Construct a lazy Erf function node for autodiff.
   * @note  Host-only: function-graph objects (IvyRegularFunction) use
   *        virtual dispatch and RAII, which are incompatible with device code.
   *        Direct numerical evaluation (the non-pointer overload) is __HOST_DEVICE__.
   */
  template<typename T, ENABLE_IF_BOOL(is_pointer_v<T>)>
  __HOST__ IvyThreadSafePtr_t<typename IvyErf<typename T::element_type>::base_t> Erf(T const& x);

  // ERFC
  template<typename T, typename domain_tag = get_domain_t<T>> struct ErfcFcnal {
    using value_t = unpacked_reduced_value_t<T>;
    using dtype_t = reduced_data_t<value_t>;
    static __INLINE_FCN_FORCE__ __HOST_DEVICE__ value_t eval(T const& x);
  };
  template<typename T> struct ErfcFcnal<T, real_domain_tag> {
    using value_t = unpacked_reduced_value_t<T>;
    using dtype_t = reduced_data_t<value_t>;
    using fndtype_t = fundamental_data_t<value_t>;
    using grad_t = IvyThreadSafePtr_t<IvyFunction<value_t, real_domain_tag>>;
    static __HOST_DEVICE__ value_t eval(T const& x);
    template<typename X_t>
    static __INLINE_FCN_FORCE__ IVY_MATH_GRAPH_QUALIFIER grad_t gradient(IvyThreadSafePtr_t<X_t> const& x);
  };
  template<typename T> struct ErfcFcnal<T, complex_domain_tag> {
    using value_t = unpacked_reduced_value_t<T>;
    using dtype_t = reduced_data_t<value_t>;
    using fndtype_t = fundamental_data_t<value_t>;
    using grad_t = IvyThreadSafePtr_t<IvyFunction<value_t, complex_domain_tag>>;
    static __HOST_DEVICE__ value_t eval(T const& x);
    template<typename X_t>
    static __INLINE_FCN_FORCE__ IVY_MATH_GRAPH_QUALIFIER grad_t gradient(IvyThreadSafePtr_t<X_t> const& x);
  };
  /** @brief Element-wise complementary error function for tensor inputs. */
  template<typename T> struct ErfcFcnal<T, tensor_domain_tag>{
    using dtype_t = typename T::dtype_t;
    using value_t = T;
    using fndtype_t = fundamental_data_t<dtype_t>;
    static __HOST__ value_t eval(T const& x);
    static __HOST__ IvyThreadSafePtr_t<T> gradient(IvyThreadSafePtr_t<T> const& dep);
  };
  template<typename T> using IvyErfc = IvyRegularFunction_1D<T, ErfcFcnal<unpack_if_function_t<T>>>;
  template<typename T, ENABLE_IF_BOOL(!is_pointer_v<T> && !is_tensor_v<T>)> __INLINE_FCN_FORCE__ __HOST_DEVICE__ typename ErfcFcnal<T>::value_t Erfc(T const& x);
  /// @brief Tensor-domain overload — @c __HOST__ only: tensor eval uses host-only STL constructs.
  template<typename T, ENABLE_IF_BOOL(!is_pointer_v<T> && is_tensor_v<T>)> __INLINE_FCN_FORCE__ __HOST__ typename ErfcFcnal<T>::value_t Erfc(T const& x);
  /**
   * @brief Construct a lazy Erfc function node for autodiff.
   * @note  Host-only: function-graph objects (IvyRegularFunction) use
   *        virtual dispatch and RAII, which are incompatible with device code.
   *        Direct numerical evaluation (the non-pointer overload) is __HOST_DEVICE__.
   */
  template<typename T, ENABLE_IF_BOOL(is_pointer_v<T>)>
  __HOST__ IvyThreadSafePtr_t<typename IvyErfc<typename T::element_type>::base_t> Erfc(T const& x);

  // FADDEEVA
  template<typename T, typename domain_tag = get_domain_t<T>> struct FaddeevaFcnal{
    using value_t = convert_to_complex_t<unpacked_reduced_value_t<T>>;
    using dtype_t = reduced_data_t<value_t>;
    static __INLINE_FCN_FORCE__ __HOST_DEVICE__ value_t eval(T const& x);
  };
  template<typename T> struct FaddeevaFcnal<T, real_domain_tag>{
    using value_t = convert_to_complex_t<unpacked_reduced_value_t<T>>;
    using dtype_t = reduced_data_t<value_t>;
    using fndtype_t = fundamental_data_t<value_t>;
    using grad_t = IvyThreadSafePtr_t<IvyFunction<value_t, complex_domain_tag>>;
    static __HOST_DEVICE__ value_t eval(T const& x);
    template<typename X_t>
    static __INLINE_FCN_FORCE__ IVY_MATH_GRAPH_QUALIFIER grad_t gradient(IvyThreadSafePtr_t<X_t> const& x);
  };
  template<typename T> struct FaddeevaFcnal<T, complex_domain_tag>{
    using value_t = unpacked_reduced_value_t<T>;
    using dtype_t = reduced_data_t<value_t>;
    using fndtype_t = fundamental_data_t<value_t>;
    using grad_t = IvyThreadSafePtr_t<IvyFunction<value_t, complex_domain_tag>>;
    static __HOST_DEVICE__ value_t eval(T const& x);
    template<typename X_t>
    static __INLINE_FCN_FORCE__ IVY_MATH_GRAPH_QUALIFIER grad_t gradient(IvyThreadSafePtr_t<X_t> const& x);
  };
  /**
   * @brief Element-wise Faddeeva function for tensor inputs.
   *
   * The Faddeeva function w(z) = exp(-z²)·erfc(-iz) is applied element-wise.
   * For a real-valued input tensor element x, the output is complex: w(x+0i).
   * The output @c value_t is therefore a tensor of complex variables, and the
   * gradient tensor is likewise complex-valued.
   *
   * @note Because the output domain (complex) differs from the input domain
   *       (tensor of real variables), this specialization cannot be used with
   *       the generic @c IvyFaddeeva graph-node alias.  Access it via the
   *       direct non-pointer eval path: @c Faddeeva(*tensor).
   * @tparam T  An IvyTensor type whose element type is real-domain.
   */
  template<typename T> struct FaddeevaFcnal<T, tensor_domain_tag>{
    using dtype_t = typename T::dtype_t;
    using fndtype_t = fundamental_data_t<dtype_t>;
    /// Output tensor holds complex elements.
    using value_t = IvyTensor<convert_to_complex_t<dtype_t>>;
    /// Gradient is also a complex-valued tensor.
    using grad_value_t = IvyTensor<convert_to_complex_t<dtype_t>>;
    using gradient_domain_tag = undefined_domain_tag;
    static __HOST__ value_t eval(T const& x);
    static __HOST__ IvyThreadSafePtr_t<grad_value_t> gradient(IvyThreadSafePtr_t<T> const& dep);
  };
  template<typename T> using IvyFaddeeva = IvyRegularFunction_1D<
    T,
    FaddeevaFcnal<unpack_if_function_t<T>>,
    unpacked_reduced_value_t< typename FaddeevaFcnal<unpack_if_function_t<T>>::value_t >,
    get_domain_t< typename FaddeevaFcnal<unpack_if_function_t<T>>::value_t >
  >;
  template<typename T, ENABLE_IF_BOOL(!is_pointer_v<T> && !is_tensor_v<T>)> __INLINE_FCN_FORCE__ __HOST_DEVICE__ typename FaddeevaFcnal<T>::value_t Faddeeva(T const& x);
  /// @brief Tensor-domain overload — @c __HOST__ only: tensor eval uses host-only STL constructs.
  template<typename T, ENABLE_IF_BOOL(!is_pointer_v<T> && is_tensor_v<T>)> __INLINE_FCN_FORCE__ __HOST__ typename FaddeevaFcnal<T>::value_t Faddeeva(T const& x);
  /**
   * @brief Construct a lazy Faddeeva function node for autodiff.
   * @note  Host-only: function-graph objects (IvyRegularFunction) use
   *        virtual dispatch and RAII, which are incompatible with device code.
   *        Direct numerical evaluation (the non-pointer overload) is __HOST_DEVICE__.
   */
  template<typename T, ENABLE_IF_BOOL(is_pointer_v<T>)>
  __HOST__ IvyThreadSafePtr_t<typename IvyFaddeeva<typename T::element_type>::base_t> Faddeeva(T const& x);

  // ERF-FAST
  template<typename T, typename domain_tag = get_domain_t<T>> struct ErfFastFcnal{
    using value_t = unpacked_reduced_value_t<T>;
    using dtype_t = reduced_data_t<value_t>;
    static __INLINE_FCN_FORCE__ __HOST_DEVICE__ value_t eval(T const& x);
  };
  template<typename T> struct ErfFastFcnal<T, real_domain_tag>{
    using value_t = unpacked_reduced_value_t<T>;
    using dtype_t = reduced_data_t<value_t>;
    using fndtype_t = fundamental_data_t<value_t>;
    using grad_t = IvyThreadSafePtr_t<IvyFunction<value_t, real_domain_tag>>;
    static __HOST_DEVICE__ value_t eval(T const& x);
    template<typename X_t>
    static __INLINE_FCN_FORCE__ IVY_MATH_GRAPH_QUALIFIER grad_t gradient(IvyThreadSafePtr_t<X_t> const& x);
  };
  template<typename T> struct ErfFastFcnal<T, complex_domain_tag>{
    using value_t = unpacked_reduced_value_t<T>;
    using dtype_t = reduced_data_t<value_t>;
    using fndtype_t = fundamental_data_t<value_t>;
    using grad_t = IvyThreadSafePtr_t<IvyFunction<value_t, complex_domain_tag>>;
    static __HOST_DEVICE__ value_t eval(T const& x);
    template<typename X_t>
    static __INLINE_FCN_FORCE__ IVY_MATH_GRAPH_QUALIFIER grad_t gradient(IvyThreadSafePtr_t<X_t> const& x);
  };
  /** @brief Element-wise fast error function for tensor inputs. */
  template<typename T> struct ErfFastFcnal<T, tensor_domain_tag>{
    using dtype_t = typename T::dtype_t;
    using value_t = T;
    using fndtype_t = fundamental_data_t<dtype_t>;
    static __HOST__ value_t eval(T const& x);
    static __HOST__ IvyThreadSafePtr_t<T> gradient(IvyThreadSafePtr_t<T> const& dep);
  };
  template<typename T> using IvyErfFast = IvyRegularFunction_1D<T, ErfFastFcnal<unpack_if_function_t<T>>>;
  template<typename T, ENABLE_IF_BOOL(!is_pointer_v<T> && !is_tensor_v<T>)> __INLINE_FCN_FORCE__ __HOST_DEVICE__ typename ErfFastFcnal<T>::value_t ErfFast(T const& x);
  /// @brief Tensor-domain overload — @c __HOST__ only: tensor eval uses host-only STL constructs.
  template<typename T, ENABLE_IF_BOOL(!is_pointer_v<T> && is_tensor_v<T>)> __INLINE_FCN_FORCE__ __HOST__ typename ErfFastFcnal<T>::value_t ErfFast(T const& x);
  /**
   * @brief Construct a lazy ErfFast function node for autodiff.
   * @note  Host-only: function-graph objects (IvyRegularFunction) use
   *        virtual dispatch and RAII, which are incompatible with device code.
   *        Direct numerical evaluation (the non-pointer overload) is __HOST_DEVICE__.
   */
  template<typename T, ENABLE_IF_BOOL(is_pointer_v<T>)>
  __HOST__ IvyThreadSafePtr_t<typename IvyErfFast<typename T::element_type>::base_t> ErfFast(T const& x);

  // ERFC-FAST
  template<typename T, typename domain_tag = get_domain_t<T>> struct ErfcFastFcnal{
    using value_t = unpacked_reduced_value_t<T>;
    using dtype_t = reduced_data_t<value_t>;
    static __INLINE_FCN_FORCE__ __HOST_DEVICE__ value_t eval(T const& x);
  };
  template<typename T> struct ErfcFastFcnal<T, real_domain_tag>{
    using value_t = unpacked_reduced_value_t<T>;
    using dtype_t = reduced_data_t<value_t>;
    using fndtype_t = fundamental_data_t<value_t>;
    using grad_t = IvyThreadSafePtr_t<IvyFunction<value_t, real_domain_tag>>;
    static __HOST_DEVICE__ value_t eval(T const& x);
    template<typename X_t>
    static __INLINE_FCN_FORCE__ IVY_MATH_GRAPH_QUALIFIER grad_t gradient(IvyThreadSafePtr_t<X_t> const& x);
  };
  template<typename T> struct ErfcFastFcnal<T, complex_domain_tag>{
    using value_t = unpacked_reduced_value_t<T>;
    using dtype_t = reduced_data_t<value_t>;
    using fndtype_t = fundamental_data_t<value_t>;
    using grad_t = IvyThreadSafePtr_t<IvyFunction<value_t, complex_domain_tag>>;
    static __HOST_DEVICE__ value_t eval(T const& x);
    template<typename X_t>
    static __INLINE_FCN_FORCE__ IVY_MATH_GRAPH_QUALIFIER grad_t gradient(IvyThreadSafePtr_t<X_t> const& x);
  };
  /** @brief Element-wise fast complementary error function for tensor inputs. */
  template<typename T> struct ErfcFastFcnal<T, tensor_domain_tag>{
    using dtype_t = typename T::dtype_t;
    using value_t = T;
    using fndtype_t = fundamental_data_t<dtype_t>;
    static __HOST__ value_t eval(T const& x);
    static __HOST__ IvyThreadSafePtr_t<T> gradient(IvyThreadSafePtr_t<T> const& dep);
  };
  template<typename T> using IvyErfcFast = IvyRegularFunction_1D<T, ErfcFastFcnal<unpack_if_function_t<T>>>;
  template<typename T, ENABLE_IF_BOOL(!is_pointer_v<T> && !is_tensor_v<T>)> __INLINE_FCN_FORCE__ __HOST_DEVICE__ typename ErfcFastFcnal<T>::value_t ErfcFast(T const& x);
  /// @brief Tensor-domain overload — @c __HOST__ only: tensor eval uses host-only STL constructs.
  template<typename T, ENABLE_IF_BOOL(!is_pointer_v<T> && is_tensor_v<T>)> __INLINE_FCN_FORCE__ __HOST__ typename ErfcFastFcnal<T>::value_t ErfcFast(T const& x);
  /**
   * @brief Construct a lazy ErfcFast function node for autodiff.
   * @note  Host-only: function-graph objects (IvyRegularFunction) use
   *        virtual dispatch and RAII, which are incompatible with device code.
   *        Direct numerical evaluation (the non-pointer overload) is __HOST_DEVICE__.
   */
  template<typename T, ENABLE_IF_BOOL(is_pointer_v<T>)>
  __HOST__ IvyThreadSafePtr_t<typename IvyErfcFast<typename T::element_type>::base_t> ErfcFast(T const& x);

  // FADDEEVA-FAST
  template<typename T, typename domain_tag = get_domain_t<T>> struct FaddeevaFastFcnal{
    using value_t = convert_to_complex_t<unpacked_reduced_value_t<T>>;
    using dtype_t = reduced_data_t<value_t>;
    static __INLINE_FCN_FORCE__ __HOST_DEVICE__ value_t eval(T const& x);
  };
  template<typename T> struct FaddeevaFastFcnal<T, real_domain_tag>{
    using value_t = convert_to_complex_t<unpacked_reduced_value_t<T>>;
    using dtype_t = reduced_data_t<value_t>;
    using fndtype_t = fundamental_data_t<value_t>;
    using grad_t = IvyThreadSafePtr_t<IvyFunction<value_t, complex_domain_tag>>;
    static __HOST_DEVICE__ value_t eval(T const& x);
    template<typename X_t>
    static __INLINE_FCN_FORCE__ IVY_MATH_GRAPH_QUALIFIER grad_t gradient(IvyThreadSafePtr_t<X_t> const& x);
  };
  template<typename T> struct FaddeevaFastFcnal<T, complex_domain_tag>{
    using value_t = unpacked_reduced_value_t<T>;
    using dtype_t = reduced_data_t<value_t>;
    using fndtype_t = fundamental_data_t<value_t>;
    using grad_t = IvyThreadSafePtr_t<IvyFunction<value_t, complex_domain_tag>>;
    static __HOST_DEVICE__ value_t eval(T const& x);
    template<typename X_t>
    static __INLINE_FCN_FORCE__ IVY_MATH_GRAPH_QUALIFIER grad_t gradient(IvyThreadSafePtr_t<X_t> const& x);
  };
  /**
   * @brief Element-wise fast Faddeeva function for tensor inputs (complex-valued output).
   * @see FaddeevaFcnal<T, tensor_domain_tag> — same complex-output semantics; access
   *      via the eager path @c FaddeevaFast(*tensor).
   */
  template<typename T> struct FaddeevaFastFcnal<T, tensor_domain_tag>{
    using dtype_t = typename T::dtype_t;
    using fndtype_t = fundamental_data_t<dtype_t>;
    using value_t = IvyTensor<convert_to_complex_t<dtype_t>>;
    using grad_value_t = IvyTensor<convert_to_complex_t<dtype_t>>;
    using gradient_domain_tag = undefined_domain_tag;
    static __HOST__ value_t eval(T const& x);
    static __HOST__ IvyThreadSafePtr_t<grad_value_t> gradient(IvyThreadSafePtr_t<T> const& dep);
  };
  template<typename T> using IvyFaddeevaFast = IvyRegularFunction_1D<
    T,
    FaddeevaFastFcnal<unpack_if_function_t<T>>,
    unpacked_reduced_value_t< typename FaddeevaFastFcnal<unpack_if_function_t<T>>::value_t >,
    get_domain_t< typename FaddeevaFastFcnal<unpack_if_function_t<T>>::value_t >
  >;
  template<typename T, ENABLE_IF_BOOL(!is_pointer_v<T> && !is_tensor_v<T>)> __INLINE_FCN_FORCE__ __HOST_DEVICE__ typename FaddeevaFastFcnal<T>::value_t FaddeevaFast(T const& x);
  /// @brief Tensor-domain overload — @c __HOST__ only: tensor eval uses host-only STL constructs.
  template<typename T, ENABLE_IF_BOOL(!is_pointer_v<T> && is_tensor_v<T>)> __INLINE_FCN_FORCE__ __HOST__ typename FaddeevaFastFcnal<T>::value_t FaddeevaFast(T const& x);
  /**
   * @brief Construct a lazy FaddeevaFast function node for autodiff.
   * @note  Host-only: function-graph objects (IvyRegularFunction) use
   *        virtual dispatch and RAII, which are incompatible with device code.
   *        Direct numerical evaluation (the non-pointer overload) is __HOST_DEVICE__.
   */
  template<typename T, ENABLE_IF_BOOL(is_pointer_v<T>)>
  __HOST__ IvyThreadSafePtr_t<typename IvyFaddeevaFast<typename T::element_type>::base_t> FaddeevaFast(T const& x);

  /****************/
  /* 2D FUNCTIONS */
  /****************/

  /**
   * @brief Operation tags for the shared element-wise binary tensor functional. Each tag supplies
   *        the value-level @c combine (used by the broadcast-aware @c eval) and the
   *        directional-derivative @c dcombine (used by the eager tensor chain rule in
   *        IvyRegularFunction_2D::gradient). Keeping these here means the per-op math lives in
   *        exactly one place and is shared across the tensor⊗tensor and tensor⊗scalar specializations.
   */
  struct ivy_tensor_add_op{
    template<typename A, typename B> static __INLINE_FCN_FORCE__ __HOST__ auto combine(A const& a, B const& b){ return a + b; }
    template<typename XV, typename YV, typename GXV, typename GYV> static __INLINE_FCN_FORCE__ __HOST__ auto dcombine(XV const&, YV const&, GXV const& gxv, GYV const& gyv){ return gxv + gyv; }
  };
  struct ivy_tensor_subtract_op{
    template<typename A, typename B> static __INLINE_FCN_FORCE__ __HOST__ auto combine(A const& a, B const& b){ return a - b; }
    template<typename XV, typename YV, typename GXV, typename GYV> static __INLINE_FCN_FORCE__ __HOST__ auto dcombine(XV const&, YV const&, GXV const& gxv, GYV const& gyv){ return gxv - gyv; }
  };
  struct ivy_tensor_multiply_op{
    template<typename A, typename B> static __INLINE_FCN_FORCE__ __HOST__ auto combine(A const& a, B const& b){ return a * b; }
    template<typename XV, typename YV, typename GXV, typename GYV> static __INLINE_FCN_FORCE__ __HOST__ auto dcombine(XV const& xv, YV const& yv, GXV const& gxv, GYV const& gyv){ return yv*gxv + xv*gyv; }
  };
  struct ivy_tensor_divide_op{
    template<typename A, typename B> static __INLINE_FCN_FORCE__ __HOST__ auto combine(A const& a, B const& b){ return a / b; }
    template<typename XV, typename YV, typename GXV, typename GYV> static __INLINE_FCN_FORCE__ __HOST__ auto dcombine(XV const& xv, YV const& yv, GXV const& gxv, GYV const& gyv){ return gxv/yv - (xv*gyv)/(yv*yv); }
  };
  /**
   * @brief Shared implementation of element-wise binary tensor operations covering both
   *        tensor⊗tensor and tensor⊗scalar (with broadcast). The function output is always the
   *        reduced value tensor @c more_precise_reduced_t<T,U> (e.g. @c IvyTensor<double> or, under
   *        real⊗complex up-casting, @c IvyTensor<IvyComplex<double>>) regardless of whether the
   *        operands are array-of-pointers, contiguous cells, or a broadcast scalar leaf. @c eval is
   *        defined in IvyMathBaseArithmetic.h; @c dcombine forwards to the op tag.
   */
  template<typename OpTag, typename T, typename U> struct IvyTensorBinaryFcnal{
    using value_t = more_precise_reduced_t<T, U>;
    using dtype_t = typename value_t::dtype_t;
    using fndtype_t = fundamental_data_t<dtype_t>;
    static __HOST__ value_t eval(T const& x, U const& y);
    template<typename XV, typename YV, typename GXV, typename GYV>
    static __INLINE_FCN_FORCE__ __HOST__ auto dcombine(XV const& xv, YV const& yv, GXV const& gxv, GYV const& gyv){ return OpTag::dcombine(xv, yv, gxv, gyv); }
  };
  /// @brief Generate the tensor⊗tensor and tensor⊗scalar (real/arithmetic/complex, both orders)
  ///        specializations of a binary functional, all delegating to the shared implementation.
#define IVY_DECL_TENSOR_BINOP_FCNAL(FCNAL, OPTAG) \
  template<typename T, typename U> struct FCNAL<T, U, tensor_domain_tag, tensor_domain_tag> : IvyTensorBinaryFcnal<OPTAG, T, U>{}; \
  template<typename T, typename U> struct FCNAL<T, U, real_domain_tag, tensor_domain_tag> : IvyTensorBinaryFcnal<OPTAG, T, U>{}; \
  template<typename T, typename U> struct FCNAL<T, U, tensor_domain_tag, real_domain_tag> : IvyTensorBinaryFcnal<OPTAG, T, U>{}; \
  template<typename T, typename U> struct FCNAL<T, U, arithmetic_domain_tag, tensor_domain_tag> : IvyTensorBinaryFcnal<OPTAG, T, U>{}; \
  template<typename T, typename U> struct FCNAL<T, U, tensor_domain_tag, arithmetic_domain_tag> : IvyTensorBinaryFcnal<OPTAG, T, U>{}; \
  template<typename T, typename U> struct FCNAL<T, U, complex_domain_tag, tensor_domain_tag> : IvyTensorBinaryFcnal<OPTAG, T, U>{}; \
  template<typename T, typename U> struct FCNAL<T, U, tensor_domain_tag, complex_domain_tag> : IvyTensorBinaryFcnal<OPTAG, T, U>{};

  // ADDITION
  template<typename T, typename U, typename domain_T = get_domain_t<T>, typename domain_U = get_domain_t<U>> struct AddFcnal{
    using value_t = more_precise_reduced_t<T, U>;
    using dtype_t = reduced_data_t<value_t>;
    using fndtype_t = fundamental_data_t<value_t>;
    static __INLINE_FCN_FORCE__ __HOST_DEVICE__ value_t eval(T const& x, U const& y);
  };
  template<typename T, typename U> struct AddFcnal<T, U, real_domain_tag, real_domain_tag>{
    using value_t = more_precise_reduced_t<T, U>;
    using dtype_t = reduced_data_t<value_t>;
    using fndtype_t = fundamental_data_t<value_t>;
    using grad_t = IvyScalarPtr_t<fndtype_t>;
    static __HOST_DEVICE__ value_t eval(T const& x, U const& y);
    template<typename X_t, typename Y_t>
    static __INLINE_FCN_FORCE__ IVY_MATH_GRAPH_QUALIFIER grad_t gradient(unsigned char ivar, IvyThreadSafePtr_t<X_t> const& x, IvyThreadSafePtr_t<Y_t> const& y);
  };
  template<typename T, typename U> struct AddFcnal<T, U, complex_domain_tag, complex_domain_tag>{
    using value_t = more_precise_reduced_t<T, U>;
    using dtype_t = reduced_data_t<value_t>;
    using fndtype_t = fundamental_data_t<value_t>;
    using grad_t = IvyComplexPtr_t<fndtype_t>;
    static __HOST_DEVICE__ value_t eval(T const& x, U const& y);
    template<typename X_t, typename Y_t>
    static __INLINE_FCN_FORCE__ IVY_MATH_GRAPH_QUALIFIER grad_t gradient(unsigned char ivar, IvyThreadSafePtr_t<X_t> const& x, IvyThreadSafePtr_t<Y_t> const& y);
  };
  /** @brief Quaternion addition. Linear: d(x+y) = dx + dy (order-independent). */
  template<typename T, typename U> struct AddFcnal<T, U, quaternion_domain_tag, quaternion_domain_tag>{
    using value_t = more_precise_reduced_t<T, U>;
    using dtype_t = reduced_data_t<value_t>;
    using fndtype_t = fundamental_data_t<value_t>;
    using grad_t = IvyQuaternionPtr_t<fndtype_t>;
    static __HOST_DEVICE__ value_t eval(T const& x, U const& y);
    template<typename X_t, typename Y_t>
    static __INLINE_FCN_FORCE__ IVY_MATH_GRAPH_QUALIFIER grad_t gradient(unsigned char ivar, IvyThreadSafePtr_t<X_t> const& x, IvyThreadSafePtr_t<Y_t> const& y);
  };
  template<typename T, typename U> struct AddFcnal<T, U, arithmetic_domain_tag, real_domain_tag>{
    using value_t = more_precise_reduced_t<T, U>;
    using dtype_t = reduced_data_t<value_t>;
    using fndtype_t = fundamental_data_t<value_t>;
    static __HOST_DEVICE__ value_t eval(T const& x, U const& y);
  };
  template<typename T, typename U> struct AddFcnal<T, U, real_domain_tag, arithmetic_domain_tag>{
    using value_t = more_precise_reduced_t<T, U>;
    using dtype_t = reduced_data_t<value_t>;
    using fndtype_t = fundamental_data_t<value_t>;
    static __HOST_DEVICE__ value_t eval(T const& x, U const& y);
  };
  template<typename T, typename U> struct AddFcnal<T, U, arithmetic_domain_tag, complex_domain_tag>{
    using value_t = more_precise_reduced_t<T, U>;
    using dtype_t = reduced_data_t<value_t>;
    using fndtype_t = fundamental_data_t<value_t>;
    static __HOST_DEVICE__ value_t eval(T const& x, U const& y);
  };
  template<typename T, typename U> struct AddFcnal<T, U, complex_domain_tag, arithmetic_domain_tag>{
    using value_t = more_precise_reduced_t<T, U>;
    using dtype_t = reduced_data_t<value_t>;
    using fndtype_t = fundamental_data_t<value_t>;
    static __HOST_DEVICE__ value_t eval(T const& x, U const& y);
  };
  template<typename T, typename U> struct AddFcnal<T, U, real_domain_tag, complex_domain_tag>{
    using value_t = more_precise_reduced_t<T, U>;
    using dtype_t = reduced_data_t<value_t>;
    using fndtype_t = fundamental_data_t<value_t>;
    using grad_t = IvyComplexPtr_t<fndtype_t>;
    static __HOST_DEVICE__ value_t eval(T const& x, U const& y);
    template<typename X_t, typename Y_t>
    static __INLINE_FCN_FORCE__ IVY_MATH_GRAPH_QUALIFIER grad_t gradient(unsigned char ivar, IvyThreadSafePtr_t<X_t> const& x, IvyThreadSafePtr_t<Y_t> const& y);
  };
  template<typename T, typename U> struct AddFcnal<T, U, complex_domain_tag, real_domain_tag>{
    using value_t = more_precise_reduced_t<T, U>;
    using dtype_t = reduced_data_t<value_t>;
    using fndtype_t = fundamental_data_t<value_t>;
    using grad_t = IvyComplexPtr_t<fndtype_t>;
    static __HOST_DEVICE__ value_t eval(T const& x, U const& y);
    template<typename X_t, typename Y_t>
    static __INLINE_FCN_FORCE__ IVY_MATH_GRAPH_QUALIFIER grad_t gradient(unsigned char ivar, IvyThreadSafePtr_t<X_t> const& x, IvyThreadSafePtr_t<Y_t> const& y);
  };
  IVY_DECL_TENSOR_BINOP_FCNAL(AddFcnal, ivy_tensor_add_op)
  template<typename T, typename U> using IvyAdd = IvyRegularFunction_2D<T, U, AddFcnal<unpack_if_function_t<T>, unpack_if_function_t<U>>>;
  template<typename T, typename U, ENABLE_IF_BOOL(!is_pointer_v<T> && !is_pointer_v<U>)>
  __INLINE_FCN_FORCE__ __HOST_DEVICE__ typename AddFcnal<T, U>::value_t Add(T const& x, U const& y);
  template<typename T, typename U, ENABLE_IF_BOOL(!(is_arithmetic_v<T> && is_arithmetic_v<U>) && !is_pointer_v<T> && !is_pointer_v<U> && (is_ivy_domain_v<T> || is_ivy_domain_v<U>))>
  __INLINE_FCN_FORCE__ __HOST_DEVICE__ typename AddFcnal<T, U>::value_t operator+(T const& x, U const& y);
  /**
   * @brief Construct a lazy Add function node for autodiff.
   * @note  Host-only: function-graph objects (IvyRegularFunction) use
   *        virtual dispatch and RAII, which are incompatible with device code.
   *        Direct numerical evaluation (the non-pointer overload) is __HOST_DEVICE__.
   */
  template<typename T, typename U, ENABLE_IF_BOOL(is_pointer_v<T> && is_pointer_v<U>)>
  __HOST__ IvyThreadSafePtr_t<typename IvyAdd<typename T::element_type, typename U::element_type>::base_t> Add(T const& x, U const& y);
  template<typename T, typename U, ENABLE_IF_BOOL(is_pointer_v<T> && is_pointer_v<U>)>
  __HOST__ IvyThreadSafePtr_t<typename IvyAdd<typename T::element_type, typename U::element_type>::base_t> operator+(T const& x, U const& y);

  // SUBTRACTION
  template<typename T, typename U, typename domain_T = get_domain_t<T>, typename domain_U = get_domain_t<U>> struct SubtractFcnal{
    using value_t = more_precise_reduced_t<T, U>;
    using dtype_t = reduced_data_t<value_t>;
    using fndtype_t = fundamental_data_t<value_t>;
    static __INLINE_FCN_FORCE__ __HOST_DEVICE__ value_t eval(T const& x, U const& y);
  };
  template<typename T, typename U> struct SubtractFcnal<T, U, real_domain_tag, real_domain_tag>{
    using value_t = more_precise_reduced_t<T, U>;
    using dtype_t = reduced_data_t<value_t>;
    using fndtype_t = fundamental_data_t<value_t>;
    using grad_t = IvyScalarPtr_t<fndtype_t>;
    static __HOST_DEVICE__ value_t eval(T const& x, U const& y);
    template<typename X_t, typename Y_t>
    static __INLINE_FCN_FORCE__ IVY_MATH_GRAPH_QUALIFIER grad_t gradient(unsigned char ivar, IvyThreadSafePtr_t<X_t> const& x, IvyThreadSafePtr_t<Y_t> const& y);
  };
  template<typename T, typename U> struct SubtractFcnal<T, U, complex_domain_tag, complex_domain_tag>{
    using value_t = more_precise_reduced_t<T, U>;
    using dtype_t = reduced_data_t<value_t>;
    using fndtype_t = fundamental_data_t<value_t>;
    using grad_t = IvyComplexPtr_t<fndtype_t>;
    static __HOST_DEVICE__ value_t eval(T const& x, U const& y);
    template<typename X_t, typename Y_t>
    static __INLINE_FCN_FORCE__ IVY_MATH_GRAPH_QUALIFIER grad_t gradient(unsigned char ivar, IvyThreadSafePtr_t<X_t> const& x, IvyThreadSafePtr_t<Y_t> const& y);
  };
  /** @brief Quaternion subtraction. Linear: d(x-y) = dx - dy (order-independent). */
  template<typename T, typename U> struct SubtractFcnal<T, U, quaternion_domain_tag, quaternion_domain_tag>{
    using value_t = more_precise_reduced_t<T, U>;
    using dtype_t = reduced_data_t<value_t>;
    using fndtype_t = fundamental_data_t<value_t>;
    using grad_t = IvyQuaternionPtr_t<fndtype_t>;
    static __HOST_DEVICE__ value_t eval(T const& x, U const& y);
    template<typename X_t, typename Y_t>
    static __INLINE_FCN_FORCE__ IVY_MATH_GRAPH_QUALIFIER grad_t gradient(unsigned char ivar, IvyThreadSafePtr_t<X_t> const& x, IvyThreadSafePtr_t<Y_t> const& y);
  };
  template<typename T, typename U> struct SubtractFcnal<T, U, arithmetic_domain_tag, real_domain_tag>{
    using value_t = more_precise_reduced_t<T, U>;
    using dtype_t = reduced_data_t<value_t>;
    using fndtype_t = fundamental_data_t<value_t>;
    static __HOST_DEVICE__ value_t eval(T const& x, U const& y);
  };
  template<typename T, typename U> struct SubtractFcnal<T, U, real_domain_tag, arithmetic_domain_tag>{
    using value_t = more_precise_reduced_t<T, U>;
    using dtype_t = reduced_data_t<value_t>;
    using fndtype_t = fundamental_data_t<value_t>;
    static __HOST_DEVICE__ value_t eval(T const& x, U const& y);
  };
  template<typename T, typename U> struct SubtractFcnal<T, U, arithmetic_domain_tag, complex_domain_tag>{
    using value_t = more_precise_reduced_t<T, U>;
    using dtype_t = reduced_data_t<value_t>;
    using fndtype_t = fundamental_data_t<value_t>;
    static __HOST_DEVICE__ value_t eval(T const& x, U const& y);
  };
  template<typename T, typename U> struct SubtractFcnal<T, U, complex_domain_tag, arithmetic_domain_tag>{
    using value_t = more_precise_reduced_t<T, U>;
    using dtype_t = reduced_data_t<value_t>;
    using fndtype_t = fundamental_data_t<value_t>;
    static __HOST_DEVICE__ value_t eval(T const& x, U const& y);
  };
  template<typename T, typename U> struct SubtractFcnal<T, U, real_domain_tag, complex_domain_tag>{
    using value_t = more_precise_reduced_t<T, U>;
    using dtype_t = reduced_data_t<value_t>;
    using fndtype_t = fundamental_data_t<value_t>;
    using grad_t = IvyComplexPtr_t<fndtype_t>;
    static __HOST_DEVICE__ value_t eval(T const& x, U const& y);
    template<typename X_t, typename Y_t>
    static __INLINE_FCN_FORCE__ IVY_MATH_GRAPH_QUALIFIER grad_t gradient(unsigned char ivar, IvyThreadSafePtr_t<X_t> const& x, IvyThreadSafePtr_t<Y_t> const& y);
  };
  template<typename T, typename U> struct SubtractFcnal<T, U, complex_domain_tag, real_domain_tag>{
    using value_t = more_precise_reduced_t<T, U>;
    using dtype_t = reduced_data_t<value_t>;
    using fndtype_t = fundamental_data_t<value_t>;
    using grad_t = IvyComplexPtr_t<fndtype_t>;
    static __HOST_DEVICE__ value_t eval(T const& x, U const& y);
    template<typename X_t, typename Y_t>
    static __INLINE_FCN_FORCE__ IVY_MATH_GRAPH_QUALIFIER grad_t gradient(unsigned char ivar, IvyThreadSafePtr_t<X_t> const& x, IvyThreadSafePtr_t<Y_t> const& y);
  };
  IVY_DECL_TENSOR_BINOP_FCNAL(SubtractFcnal, ivy_tensor_subtract_op)
  template<typename T, typename U> using IvySubtract = IvyRegularFunction_2D<T, U, SubtractFcnal<unpack_if_function_t<T>, unpack_if_function_t<U>>>;
  template<typename T, typename U, ENABLE_IF_BOOL(!is_pointer_v<T> && !is_pointer_v<U>)>
  __INLINE_FCN_FORCE__ __HOST_DEVICE__ typename SubtractFcnal<T, U>::value_t Subtract(T const& x, U const& y);
  template<
    typename T, typename U, ENABLE_IF_BOOL(
      !(is_arithmetic_v<T> && is_arithmetic_v<U>)
      &&
      !is_pointer_v<T> && !is_pointer_v<U>
      &&
      !std_iter::is_contiguous_iterator_v<T> && !std_iter::is_contiguous_iterator_v<U>
      &&
      (is_ivy_domain_v<T> || is_ivy_domain_v<U>)
    )
  >
  __INLINE_FCN_FORCE__ __HOST_DEVICE__ typename SubtractFcnal<T, U>::value_t operator-(T const& x, U const& y);
  /**
   * @brief Construct a lazy Subtract function node for autodiff.
   * @note  Host-only: function-graph objects (IvyRegularFunction) use
   *        virtual dispatch and RAII, which are incompatible with device code.
   *        Direct numerical evaluation (the non-pointer overload) is __HOST_DEVICE__.
   */
  template<typename T, typename U, ENABLE_IF_BOOL(is_pointer_v<T> && is_pointer_v<U>)>
  __HOST__ IvyThreadSafePtr_t<typename IvySubtract<typename T::element_type, typename U::element_type>::base_t> Subtract(T const& x, U const& y);
  template<typename T, typename U, ENABLE_IF_BOOL(is_pointer_v<T> && is_pointer_v<U>)>
  __HOST__ IvyThreadSafePtr_t<typename IvySubtract<typename T::element_type, typename U::element_type>::base_t> operator-(T const& x, U const& y);

  // MULTIPLICATION
  template<typename T, typename U, typename domain_T = get_domain_t<T>, typename domain_U = get_domain_t<U>> struct MultiplyFcnal{
    using value_t = more_precise_reduced_t<T, U>;
    using dtype_t = reduced_data_t<value_t>;
    using fndtype_t = fundamental_data_t<value_t>;
    static __INLINE_FCN_FORCE__ __HOST_DEVICE__ value_t eval(T const& x, U const& y);
  };
  template<typename T, typename U> struct MultiplyFcnal<T, U, real_domain_tag, real_domain_tag>{
    using value_t = more_precise_reduced_t<T, U>;
    using dtype_t = reduced_data_t<value_t>;
    using fndtype_t = fundamental_data_t<value_t>;
    using grad_t = IvyFunctionPtr_t<value_t, get_domain_t<more_precise_t<T, U>>>;
    static __HOST_DEVICE__ value_t eval(T const& x, U const& y);
    template<typename X_t, typename Y_t>
    static __INLINE_FCN_FORCE__ IVY_MATH_GRAPH_QUALIFIER grad_t gradient(unsigned char ivar, IvyThreadSafePtr_t<X_t> const& x, IvyThreadSafePtr_t<Y_t> const& y);
  };
  template<typename T, typename U> struct MultiplyFcnal<T, U, complex_domain_tag, complex_domain_tag>{
    using value_t = more_precise_reduced_t<T, U>;
    using dtype_t = reduced_data_t<value_t>;
    using fndtype_t = fundamental_data_t<value_t>;
    using grad_t = IvyFunctionPtr_t<value_t, get_domain_t<more_precise_t<T, U>>>;
    static __HOST_DEVICE__ value_t eval(T const& x, U const& y);
    template<typename X_t, typename Y_t>
    static __INLINE_FCN_FORCE__ IVY_MATH_GRAPH_QUALIFIER grad_t gradient(unsigned char ivar, IvyThreadSafePtr_t<X_t> const& x, IvyThreadSafePtr_t<Y_t> const& y);
  };
  template<typename T, typename U> struct MultiplyFcnal<T, U, arithmetic_domain_tag, real_domain_tag>{
    using value_t = more_precise_reduced_t<T, U>;
    using dtype_t = reduced_data_t<value_t>;
    using fndtype_t = fundamental_data_t<value_t>;
    static __HOST_DEVICE__ value_t eval(T const& x, U const& y);
  };
  template<typename T, typename U> struct MultiplyFcnal<T, U, real_domain_tag, arithmetic_domain_tag>{
    using value_t = more_precise_reduced_t<T, U>;
    using dtype_t = reduced_data_t<value_t>;
    using fndtype_t = fundamental_data_t<value_t>;
    static __HOST_DEVICE__ value_t eval(T const& x, U const& y);
  };
  template<typename T, typename U> struct MultiplyFcnal<T, U, arithmetic_domain_tag, complex_domain_tag>{
    using value_t = more_precise_reduced_t<T, U>;
    using dtype_t = reduced_data_t<value_t>;
    using fndtype_t = fundamental_data_t<value_t>;
    static __HOST_DEVICE__ value_t eval(T const& x, U const& y);
  };
  template<typename T, typename U> struct MultiplyFcnal<T, U, complex_domain_tag, arithmetic_domain_tag>{
    using value_t = more_precise_reduced_t<T, U>;
    using dtype_t = reduced_data_t<value_t>;
    using fndtype_t = fundamental_data_t<value_t>;
    static __HOST_DEVICE__ value_t eval(T const& x, U const& y);
  };
  template<typename T, typename U> struct MultiplyFcnal<T, U, real_domain_tag, complex_domain_tag>{
    using value_t = more_precise_reduced_t<T, U>;
    using dtype_t = reduced_data_t<value_t>;
    using fndtype_t = fundamental_data_t<value_t>;
    using grad_t = IvyFunctionPtr_t<value_t, get_domain_t<more_precise_t<T, U>>>;
    static __HOST_DEVICE__ value_t eval(T const& x, U const& y);
    template<typename X_t, typename Y_t>
    static __INLINE_FCN_FORCE__ IVY_MATH_GRAPH_QUALIFIER grad_t gradient(unsigned char ivar, IvyThreadSafePtr_t<X_t> const& x, IvyThreadSafePtr_t<Y_t> const& y);
  };
  template<typename T, typename U> struct MultiplyFcnal<T, U, complex_domain_tag, real_domain_tag>{
    using value_t = more_precise_reduced_t<T, U>;
    using dtype_t = reduced_data_t<value_t>;
    using fndtype_t = fundamental_data_t<value_t>;
    using grad_t = IvyFunctionPtr_t<value_t, get_domain_t<more_precise_t<T, U>>>;
    static __HOST_DEVICE__ value_t eval(T const& x, U const& y);
    template<typename X_t, typename Y_t>
    static __INLINE_FCN_FORCE__ IVY_MATH_GRAPH_QUALIFIER grad_t gradient(unsigned char ivar, IvyThreadSafePtr_t<X_t> const& x, IvyThreadSafePtr_t<Y_t> const& y);
  };
  /**
   * @brief Quaternion (Hamilton) product. NON-commutative and order-aware.
   *
   * The differential is d(x*y) = dx*y + x*dy, so the upstream gradient of x is
   * right-multiplied by y and that of y is left-multiplied by x. It opts into
   * the order-aware combiner instead of the commutative local-partial form.
   */
  template<typename T, typename U> struct MultiplyFcnal<T, U, quaternion_domain_tag, quaternion_domain_tag>{
    using value_t = more_precise_reduced_t<T, U>;
    using dtype_t = reduced_data_t<value_t>;
    using fndtype_t = fundamental_data_t<value_t>;
    using grad_t = IvyFunctionPtr_t<value_t, quaternion_domain_tag>;
    using order_aware_tag = void;
    static __HOST_DEVICE__ value_t eval(T const& x, U const& y);
    template<typename X_t, typename Y_t, typename GX_t, typename GY_t>
    static __HOST__ grad_t combine_gradient(IvyThreadSafePtr_t<X_t> const& x, IvyThreadSafePtr_t<Y_t> const& y, GX_t const& grad_x, GY_t const& grad_y);
  };
  /**
   * @brief Element-wise multiplication for two tensor-domain inputs.
   *
   * Computes the Hadamard (element-wise) product.  Each output element is the
   * scalar product of the corresponding elements in @p x and @p y.
   */
  IVY_DECL_TENSOR_BINOP_FCNAL(MultiplyFcnal, ivy_tensor_multiply_op)
  template<typename T, typename U> using IvyMultiply = IvyRegularFunction_2D<T, U, MultiplyFcnal<unpack_if_function_t<T>, unpack_if_function_t<U>>>;
  template<typename T, typename U, ENABLE_IF_BOOL(!is_pointer_v<T> && !is_pointer_v<U> && !is_tensor_v<T> && !is_tensor_v<U>)>
  __INLINE_FCN_FORCE__ __HOST_DEVICE__ typename MultiplyFcnal<T, U>::value_t Multiply(T const& x, U const& y);
  /// @brief Tensor-domain overload — @c __HOST__ only: tensor eval uses host-only STL constructs.
  template<typename T, typename U, ENABLE_IF_BOOL(!is_pointer_v<T> && !is_pointer_v<U> && (is_tensor_v<T> || is_tensor_v<U>))>
  __INLINE_FCN_FORCE__ __HOST__ typename MultiplyFcnal<T, U>::value_t Multiply(T const& x, U const& y);
  template<typename T, typename U, ENABLE_IF_BOOL(!(is_arithmetic_v<T> && is_arithmetic_v<U>) && !is_pointer_v<T> && !is_pointer_v<U> && !is_tensor_v<T> && !is_tensor_v<U> && (is_ivy_domain_v<T> || is_ivy_domain_v<U>))>
  __INLINE_FCN_FORCE__ __HOST_DEVICE__ typename MultiplyFcnal<T, U>::value_t operator*(T const& x, U const& y);
  /// @brief Tensor-domain @c operator* — @c __HOST__ only: tensor eval uses host-only STL constructs.
  template<typename T, typename U, ENABLE_IF_BOOL(!(is_arithmetic_v<T> && is_arithmetic_v<U>) && !is_pointer_v<T> && !is_pointer_v<U> && (is_tensor_v<T> || is_tensor_v<U>))>
  __INLINE_FCN_FORCE__ __HOST__ typename MultiplyFcnal<T, U>::value_t operator*(T const& x, U const& y);
  /**
   * @brief Construct a lazy Multiply function node for autodiff.
   * @note  Host-only: function-graph objects (IvyRegularFunction) use
   *        virtual dispatch and RAII, which are incompatible with device code.
   *        Direct numerical evaluation (the non-pointer overload) is __HOST_DEVICE__.
   */
  template<typename T, typename U, ENABLE_IF_BOOL(is_pointer_v<T> && is_pointer_v<U>)>
  __HOST__ IvyThreadSafePtr_t<typename IvyMultiply<typename T::element_type, typename U::element_type>::base_t> Multiply(T const& x, U const& y);
  template<typename T, typename U, ENABLE_IF_BOOL(is_pointer_v<T> && is_pointer_v<U>)>
  __HOST__ IvyThreadSafePtr_t<typename IvyMultiply<typename T::element_type, typename U::element_type>::base_t> operator*(T const& x, U const& y);

  // DIVISION
  template<typename T, typename U, typename domain_T = get_domain_t<T>, typename domain_U = get_domain_t<U>> struct DivideFcnal{
    using value_t = more_precise_reduced_t<T, U>;
    using dtype_t = reduced_data_t<value_t>;
    using fndtype_t = fundamental_data_t<value_t>;
    static __INLINE_FCN_FORCE__ __HOST_DEVICE__ value_t eval(T const& x, U const& y);
  };
  template<typename T, typename U> struct DivideFcnal<T, U, real_domain_tag, real_domain_tag>{
    using value_t = more_precise_reduced_t<T, U>;
    using dtype_t = reduced_data_t<value_t>;
    using fndtype_t = fundamental_data_t<value_t>;
    using grad_t = IvyFunctionPtr_t<value_t, get_domain_t<more_precise_t<T, U>>>;
    static __HOST_DEVICE__ value_t eval(T const& x, U const& y);
    template<typename X_t, typename Y_t>
    static __INLINE_FCN_FORCE__ IVY_MATH_GRAPH_QUALIFIER grad_t gradient(unsigned char ivar, IvyThreadSafePtr_t<X_t> const& x, IvyThreadSafePtr_t<Y_t> const& y);
  };
  template<typename T, typename U> struct DivideFcnal<T, U, complex_domain_tag, complex_domain_tag>{
    using value_t = more_precise_reduced_t<T, U>;
    using dtype_t = reduced_data_t<value_t>;
    using fndtype_t = fundamental_data_t<value_t>;
    using grad_t = IvyFunctionPtr_t<value_t, get_domain_t<more_precise_t<T, U>>>;
    static __HOST_DEVICE__ value_t eval(T const& x, U const& y);
    template<typename X_t, typename Y_t>
    static __INLINE_FCN_FORCE__ IVY_MATH_GRAPH_QUALIFIER grad_t gradient(unsigned char ivar, IvyThreadSafePtr_t<X_t> const& x, IvyThreadSafePtr_t<Y_t> const& y);
  };
  /**
   * @brief Quaternion right-division, x / y = x * y^{-1}. NON-commutative and order-aware.
   *
   * Per the agreed convention, @c operator/ is RIGHT division (multiply by the
   * inverse on the right). The differential follows from x*y^{-1}:
   *   d(x/y) = dx*y^{-1} - x*y^{-1}*dy*y^{-1}.
   */
  template<typename T, typename U> struct DivideFcnal<T, U, quaternion_domain_tag, quaternion_domain_tag>{
    using value_t = more_precise_reduced_t<T, U>;
    using dtype_t = reduced_data_t<value_t>;
    using fndtype_t = fundamental_data_t<value_t>;
    using grad_t = IvyFunctionPtr_t<value_t, quaternion_domain_tag>;
    using order_aware_tag = void;
    static __HOST_DEVICE__ value_t eval(T const& x, U const& y);
    template<typename X_t, typename Y_t, typename GX_t, typename GY_t>
    static __HOST__ grad_t combine_gradient(IvyThreadSafePtr_t<X_t> const& x, IvyThreadSafePtr_t<Y_t> const& y, GX_t const& grad_x, GY_t const& grad_y);
  };
  template<typename T, typename U> struct DivideFcnal<T, U, arithmetic_domain_tag, real_domain_tag>{
    using value_t = more_precise_reduced_t<T, U>;
    using dtype_t = reduced_data_t<value_t>;
    using fndtype_t = fundamental_data_t<value_t>;
    static __HOST_DEVICE__ value_t eval(T const& x, U const& y);
  };
  template<typename T, typename U> struct DivideFcnal<T, U, real_domain_tag, arithmetic_domain_tag>{
    using value_t = more_precise_reduced_t<T, U>;
    using dtype_t = reduced_data_t<value_t>;
    using fndtype_t = fundamental_data_t<value_t>;
    static __HOST_DEVICE__ value_t eval(T const& x, U const& y);
  };
  template<typename T, typename U> struct DivideFcnal<T, U, arithmetic_domain_tag, complex_domain_tag>{
    using value_t = more_precise_reduced_t<T, U>;
    using dtype_t = reduced_data_t<value_t>;
    using fndtype_t = fundamental_data_t<value_t>;
    static __HOST_DEVICE__ value_t eval(T const& x, U const& y);
  };
  template<typename T, typename U> struct DivideFcnal<T, U, complex_domain_tag, arithmetic_domain_tag>{
    using value_t = more_precise_reduced_t<T, U>;
    using dtype_t = reduced_data_t<value_t>;
    using fndtype_t = fundamental_data_t<value_t>;
    static __HOST_DEVICE__ value_t eval(T const& x, U const& y);
  };
  template<typename T, typename U> struct DivideFcnal<T, U, real_domain_tag, complex_domain_tag>{
    using value_t = more_precise_reduced_t<T, U>;
    using dtype_t = reduced_data_t<value_t>;
    using fndtype_t = fundamental_data_t<value_t>;
    using grad_t = IvyFunctionPtr_t<value_t, get_domain_t<more_precise_t<T, U>>>;
    static __HOST_DEVICE__ value_t eval(T const& x, U const& y);
    template<typename X_t, typename Y_t>
    static __INLINE_FCN_FORCE__ IVY_MATH_GRAPH_QUALIFIER grad_t gradient(unsigned char ivar, IvyThreadSafePtr_t<X_t> const& x, IvyThreadSafePtr_t<Y_t> const& y);
  };
  template<typename T, typename U> struct DivideFcnal<T, U, complex_domain_tag, real_domain_tag>{
    using value_t = more_precise_reduced_t<T, U>;
    using dtype_t = reduced_data_t<value_t>;
    using fndtype_t = fundamental_data_t<value_t>;
    using grad_t = IvyFunctionPtr_t<value_t, get_domain_t<more_precise_t<T, U>>>;
    static __HOST_DEVICE__ value_t eval(T const& x, U const& y);
    template<typename X_t, typename Y_t>
    static __INLINE_FCN_FORCE__ IVY_MATH_GRAPH_QUALIFIER grad_t gradient(unsigned char ivar, IvyThreadSafePtr_t<X_t> const& x, IvyThreadSafePtr_t<Y_t> const& y);
  };
  IVY_DECL_TENSOR_BINOP_FCNAL(DivideFcnal, ivy_tensor_divide_op)
  template<typename T, typename U> using IvyDivide = IvyRegularFunction_2D<T, U, DivideFcnal<unpack_if_function_t<T>, unpack_if_function_t<U>>>;
  template<typename T, typename U, ENABLE_IF_BOOL(!is_pointer_v<T> && !is_pointer_v<U>)>
  __INLINE_FCN_FORCE__ __HOST_DEVICE__ typename DivideFcnal<T, U>::value_t Divide(T const& x, U const& y);
  template<typename T, typename U, ENABLE_IF_BOOL(!(is_arithmetic_v<T> && is_arithmetic_v<U>) && !is_pointer_v<T> && !is_pointer_v<U> && (is_ivy_domain_v<T> || is_ivy_domain_v<U>))>
  __INLINE_FCN_FORCE__ __HOST_DEVICE__ typename DivideFcnal<T, U>::value_t operator/(T const& x, U const& y);
  /**
   * @brief Construct a lazy Divide function node for autodiff.
   * @note  Host-only: function-graph objects (IvyRegularFunction) use
   *        virtual dispatch and RAII, which are incompatible with device code.
   *        Direct numerical evaluation (the non-pointer overload) is __HOST_DEVICE__.
   */
  template<typename T, typename U, ENABLE_IF_BOOL(is_pointer_v<T> && is_pointer_v<U>)>
  __HOST__ IvyThreadSafePtr_t<typename IvyDivide<typename T::element_type, typename U::element_type>::base_t> Divide(T const& x, U const& y);
  template<typename T, typename U, ENABLE_IF_BOOL(is_pointer_v<T> && is_pointer_v<U>)>
  __HOST__ IvyThreadSafePtr_t<typename IvyDivide<typename T::element_type, typename U::element_type>::base_t> operator/(T const& x, U const& y);

  // POWER
  template<typename T, typename U, typename domain_T = get_domain_t<T>, typename domain_U = get_domain_t<U>> struct PowFcnal{
    using value_t = more_precise_reduced_t<T, U>;
    using dtype_t = reduced_data_t<value_t>;
    using fndtype_t = fundamental_data_t<value_t>;
    static __INLINE_FCN_FORCE__ __HOST_DEVICE__ value_t eval(T const& x, U const& y);
  };
  template<typename T, typename U> struct PowFcnal<T, U, real_domain_tag, real_domain_tag>{
    using value_t = more_precise_reduced_t<T, U>;
    using dtype_t = reduced_data_t<value_t>;
    using fndtype_t = fundamental_data_t<value_t>;
    using grad_t = IvyFunctionPtr_t<value_t, get_domain_t<more_precise_t<T, U>>>;
    static __HOST_DEVICE__ value_t eval(T const& x, U const& y);
    template<typename X_t, typename Y_t>
    static __INLINE_FCN_FORCE__ IVY_MATH_GRAPH_QUALIFIER grad_t gradient(unsigned char ivar, IvyThreadSafePtr_t<X_t> const& x, IvyThreadSafePtr_t<Y_t> const& y);
  };
  template<typename T, typename U> struct PowFcnal<T, U, complex_domain_tag, complex_domain_tag>{
    using value_t = more_precise_reduced_t<T, U>;
    using dtype_t = reduced_data_t<value_t>;
    using fndtype_t = fundamental_data_t<value_t>;
    using grad_t = IvyFunctionPtr_t<value_t, get_domain_t<more_precise_t<T, U>>>;
    static __HOST_DEVICE__ value_t eval(T const& x, U const& y);
    template<typename X_t, typename Y_t>
    static __INLINE_FCN_FORCE__ IVY_MATH_GRAPH_QUALIFIER grad_t gradient(unsigned char ivar, IvyThreadSafePtr_t<X_t> const& x, IvyThreadSafePtr_t<Y_t> const& y);
  };
  template<typename T, typename U> struct PowFcnal<T, U, arithmetic_domain_tag, real_domain_tag>{
    using value_t = more_precise_reduced_t<T, U>;
    using dtype_t = reduced_data_t<value_t>;
    using fndtype_t = fundamental_data_t<value_t>;
    static __HOST_DEVICE__ value_t eval(T const& x, U const& y);
  };
  template<typename T, typename U> struct PowFcnal<T, U, real_domain_tag, arithmetic_domain_tag>{
    using value_t = more_precise_reduced_t<T, U>;
    using dtype_t = reduced_data_t<value_t>;
    using fndtype_t = fundamental_data_t<value_t>;
    static __HOST_DEVICE__ value_t eval(T const& x, U const& y);
  };
  template<typename T, typename U> struct PowFcnal<T, U, arithmetic_domain_tag, complex_domain_tag>{
    using value_t = more_precise_reduced_t<T, U>;
    using dtype_t = reduced_data_t<value_t>;
    using fndtype_t = fundamental_data_t<value_t>;
    static __HOST_DEVICE__ value_t eval(T const& x, U const& y);
  };
  template<typename T, typename U> struct PowFcnal<T, U, complex_domain_tag, arithmetic_domain_tag>{
    using value_t = more_precise_reduced_t<T, U>;
    using dtype_t = reduced_data_t<value_t>;
    using fndtype_t = fundamental_data_t<value_t>;
    static __HOST_DEVICE__ value_t eval(T const& x, U const& y);
  };
  template<typename T, typename U> struct PowFcnal<T, U, real_domain_tag, complex_domain_tag>{
    using value_t = more_precise_reduced_t<T, U>;
    using dtype_t = reduced_data_t<value_t>;
    using fndtype_t = fundamental_data_t<value_t>;
    using grad_t = IvyFunctionPtr_t<value_t, get_domain_t<more_precise_t<T, U>>>;
    static __HOST_DEVICE__ value_t eval(T const& x, U const& y);
    template<typename X_t, typename Y_t>
    static __INLINE_FCN_FORCE__ IVY_MATH_GRAPH_QUALIFIER grad_t gradient(unsigned char ivar, IvyThreadSafePtr_t<X_t> const& x, IvyThreadSafePtr_t<Y_t> const& y);
  };
  template<typename T, typename U> struct PowFcnal<T, U, complex_domain_tag, real_domain_tag>{
    using value_t = more_precise_reduced_t<T, U>;
    using dtype_t = reduced_data_t<value_t>;
    using fndtype_t = fundamental_data_t<value_t>;
    using grad_t = IvyFunctionPtr_t<value_t, get_domain_t<more_precise_t<T, U>>>;
    static __HOST_DEVICE__ value_t eval(T const& x, U const& y);
    template<typename X_t, typename Y_t>
    static __INLINE_FCN_FORCE__ IVY_MATH_GRAPH_QUALIFIER grad_t gradient(unsigned char ivar, IvyThreadSafePtr_t<X_t> const& x, IvyThreadSafePtr_t<Y_t> const& y);
  };
  /** @brief Element-wise power for tensor inputs. Both tensors must have identical shapes. */
  template<typename T, typename U> struct PowFcnal<T, U, tensor_domain_tag, tensor_domain_tag>{
    using dtype_t = typename T::dtype_t;
    using value_t = T;
    using fndtype_t = fundamental_data_t<dtype_t>;
    using gradient_domain_tag = undefined_domain_tag;
    static __HOST__ value_t eval(T const& x, U const& y);
    template<typename X_t, typename Y_t>
    static __HOST__ IvyThreadSafePtr_t<value_t> gradient(unsigned char ivar, IvyThreadSafePtr_t<X_t> const& x, IvyThreadSafePtr_t<Y_t> const& y);
  };
  /** @brief Element-wise power for tensor base and arithmetic exponent. */
  template<typename T, typename U> struct PowFcnal<T, U, tensor_domain_tag, arithmetic_domain_tag>{
    using dtype_t = typename T::dtype_t;
    using value_t = T;
    using fndtype_t = fundamental_data_t<dtype_t>;
    static __HOST__ value_t eval(T const& x, U const& y);
    template<typename X_t, typename Y_t>
    static __HOST__ IvyThreadSafePtr_t<value_t> gradient(unsigned char ivar, IvyThreadSafePtr_t<X_t> const& x, IvyThreadSafePtr_t<Y_t> const& y);
  };
  /** @brief Element-wise power for arithmetic base and tensor exponent. */
  template<typename T, typename U> struct PowFcnal<T, U, arithmetic_domain_tag, tensor_domain_tag>{
    using dtype_t = typename U::dtype_t;
    using value_t = U;
    using fndtype_t = fundamental_data_t<dtype_t>;
    static __HOST__ value_t eval(T const& x, U const& y);
    template<typename X_t, typename Y_t>
    static __HOST__ IvyThreadSafePtr_t<value_t> gradient(unsigned char ivar, IvyThreadSafePtr_t<X_t> const& x, IvyThreadSafePtr_t<Y_t> const& y);
  };
  // NOTE: like IvyAdd/IvySubtract/IvyMultiply/IvyDivide, IvyPow relies on the default
  // precision_type/Domain of IvyRegularFunction_2D, i.e. Domain = get_domain_t<more_precise_t<T,U>>.
  // The earlier explicit override Domain = get_domain_t<PowFcnal::value_t> derived the domain from
  // the *reduced* value type, which for real⊗real collapses to a bare arithmetic type
  // (e.g. double -> arithmetic_domain_tag). That mis-tagged the Pow node and, because PowFcnal's
  // gradient recursively builds Pow(x, y-1) nodes, produced real⊗arithmetic sub-nodes whose
  // (gradient-less) evaluator is force-instantiated via the vtable -> the real-domain scalar
  // Pow/Sqrt graph node failed to compile. The default derivation keeps the algebra domain
  // (real stays real, complex stays complex) while still up-casting mixed operands.
  template<typename T, typename U> using IvyPow = IvyRegularFunction_2D<
    T, U,
    PowFcnal<unpack_if_function_t<T>, unpack_if_function_t<U>>
  >;
  template<typename T, typename U, ENABLE_IF_BOOL(!is_pointer_v<T> && !is_pointer_v<U> && !is_tensor_v<T> && !is_tensor_v<U>)>
  __INLINE_FCN_FORCE__ __HOST_DEVICE__ typename PowFcnal<T, U>::value_t Pow(T const& x, U const& y);
  /// @brief Tensor-domain overload — @c __HOST__ only: tensor eval uses host-only STL constructs.
  template<typename T, typename U, ENABLE_IF_BOOL(!is_pointer_v<T> && !is_pointer_v<U> && (is_tensor_v<T> || is_tensor_v<U>))>
  __INLINE_FCN_FORCE__ __HOST__ typename PowFcnal<T, U>::value_t Pow(T const& x, U const& y);
  /**
   * @brief Construct a lazy Pow function node for autodiff.
   * @note  Host-only: function-graph objects (IvyRegularFunction) use
   *        virtual dispatch and RAII, which are incompatible with device code.
   *        Direct numerical evaluation (the non-pointer overload) is __HOST_DEVICE__.
   */
  template<typename T, typename U, ENABLE_IF_BOOL(is_pointer_v<T>&& is_pointer_v<U>)>
  __HOST__ IvyThreadSafePtr_t<typename IvyPow<typename T::element_type, typename U::element_type>::base_t> Pow(T const& x, U const& y);


  /******************/
  /* 1D COMPARISONS */
  /******************/

  // NOT
  template<typename T, typename domain_tag = get_domain_t<T>> struct NotFcnal{
    using value_t = bool;
    using dtype_t = reduced_data_t<value_t>;
    static __INLINE_FCN_FORCE__ __HOST_DEVICE__ constexpr value_t eval(T const& x);
  };
  template<typename T> struct NotFcnal<T, real_domain_tag>{
    using value_t = bool;
    using dtype_t = reduced_data_t<value_t>;
    static __HOST_DEVICE__ value_t eval(T const& x);
  };
  template<typename T> using IvyNot = IvyConditionalFunction_1D<T, NotFcnal<unpack_if_function_t<T>>>;
  template<typename T, ENABLE_IF_BOOL(!is_pointer_v<T>)> __INLINE_FCN_FORCE__ __HOST_DEVICE__ typename NotFcnal<T>::value_t Not(T const& x);
  /**
   * @brief Construct a lazy Not function node for autodiff.
   * @note  Host-only: function-graph objects (IvyRegularFunction) use
   *        virtual dispatch and RAII, which are incompatible with device code.
   *        Direct numerical evaluation (the non-pointer overload) is __HOST_DEVICE__.
   */
  template<typename T, ENABLE_IF_BOOL(is_pointer_v<T>)>
  __HOST__ IvyThreadSafePtr_t<typename IvyNot<typename T::element_type>::base_t> Not(T const& x);


  /******************/
  /* 2D COMPARISONS */
  /******************/

  // EQUALITY
  template<typename T, typename U, typename domain_T = get_domain_t<T>, typename domain_U = get_domain_t<U>> struct EqualFcnal{
    using value_t = bool;
    using dtype_t = reduced_data_t<value_t>;
    using fndtype_t = fundamental_data_t<value_t>;
    static __INLINE_FCN_FORCE__ __HOST_DEVICE__ value_t eval(T const& x, U const& y);
  };
  template<typename T, typename U> struct EqualFcnal<T, U, real_domain_tag, real_domain_tag>{
    using value_t = bool;
    using dtype_t = reduced_data_t<value_t>;
    using fndtype_t = fundamental_data_t<value_t>;
    static __HOST_DEVICE__ value_t eval(T const& x, U const& y);
  };
  template<typename T, typename U> struct EqualFcnal<T, U, complex_domain_tag, complex_domain_tag>{
    using value_t = bool;
    using dtype_t = reduced_data_t<value_t>;
    using fndtype_t = fundamental_data_t<value_t>;
    static __HOST_DEVICE__ value_t eval(T const& x, U const& y);
  };
  template<typename T, typename U> struct EqualFcnal<T, U, arithmetic_domain_tag, real_domain_tag>{
    using value_t = bool;
    using dtype_t = reduced_data_t<value_t>;
    using fndtype_t = fundamental_data_t<value_t>;
    static __HOST_DEVICE__ value_t eval(T const& x, U const& y);
  };
  template<typename T, typename U> struct EqualFcnal<T, U, real_domain_tag, arithmetic_domain_tag>{
    using value_t = bool;
    using dtype_t = reduced_data_t<value_t>;
    using fndtype_t = fundamental_data_t<value_t>;
    static __HOST_DEVICE__ value_t eval(T const& x, U const& y);
  };
  template<typename T, typename U> struct EqualFcnal<T, U, arithmetic_domain_tag, complex_domain_tag>{
    using value_t = bool;
    using dtype_t = reduced_data_t<value_t>;
    using fndtype_t = fundamental_data_t<value_t>;
    static __HOST_DEVICE__ value_t eval(T const& x, U const& y);
  };
  template<typename T, typename U> struct EqualFcnal<T, U, complex_domain_tag, arithmetic_domain_tag>{
    using value_t = bool;
    using dtype_t = reduced_data_t<value_t>;
    using fndtype_t = fundamental_data_t<value_t>;
    static __HOST_DEVICE__ value_t eval(T const& x, U const& y);
  };
  template<typename T, typename U> struct EqualFcnal<T, U, real_domain_tag, complex_domain_tag>{
    using value_t = bool;
    using dtype_t = reduced_data_t<value_t>;
    using fndtype_t = fundamental_data_t<value_t>;
    static __HOST_DEVICE__ value_t eval(T const& x, U const& y);
  };
  template<typename T, typename U> struct EqualFcnal<T, U, complex_domain_tag, real_domain_tag>{
    using value_t = bool;
    using dtype_t = reduced_data_t<value_t>;
    using fndtype_t = fundamental_data_t<value_t>;
    static __HOST_DEVICE__ value_t eval(T const& x, U const& y);
  };
  template<typename T, typename U> using IvyEqual = IvyConditionalFunction_2D<
    T, U,
    EqualFcnal<unpack_if_function_t<T>, unpack_if_function_t<U>>
  >;
  template<typename T, typename U, ENABLE_IF_BOOL(!is_pointer_v<T> && !is_pointer_v<U>)>
  __INLINE_FCN_FORCE__ __HOST_DEVICE__ typename EqualFcnal<T, U>::value_t Equal(T const& x, U const& y);
  /**
   * @brief Construct a lazy Equal function node for autodiff.
   * @note  Host-only: function-graph objects (IvyRegularFunction) use
   *        virtual dispatch and RAII, which are incompatible with device code.
   *        Direct numerical evaluation (the non-pointer overload) is __HOST_DEVICE__.
   */
  template<typename T, typename U, ENABLE_IF_BOOL(is_pointer_v<T>&& is_pointer_v<U>)>
  __HOST__ IvyThreadSafePtr_t<typename IvyEqual<typename T::element_type, typename U::element_type>::base_t> Equal(T const& x, U const& y);

  // OR
  template<typename T, typename U, typename domain_T = get_domain_t<T>, typename domain_U = get_domain_t<U>> struct OrFcnal{
    using value_t = bool;
    using dtype_t = reduced_data_t<value_t>;
    using fndtype_t = fundamental_data_t<value_t>;
    static __INLINE_FCN_FORCE__ __HOST_DEVICE__ value_t eval(T const& x, U const& y);
  };
  template<typename T, typename U> struct OrFcnal<T, U, real_domain_tag, real_domain_tag>{
    using value_t = bool;
    using dtype_t = reduced_data_t<value_t>;
    using fndtype_t = fundamental_data_t<value_t>;
    static __HOST_DEVICE__ value_t eval(T const& x, U const& y);
  };
  template<typename T, typename U> struct OrFcnal<T, U, arithmetic_domain_tag, real_domain_tag>{
    using value_t = bool;
    using dtype_t = reduced_data_t<value_t>;
    using fndtype_t = fundamental_data_t<value_t>;
    static __HOST_DEVICE__ value_t eval(T const& x, U const& y);
  };
  template<typename T, typename U> struct OrFcnal<T, U, real_domain_tag, arithmetic_domain_tag>{
    using value_t = bool;
    using dtype_t = reduced_data_t<value_t>;
    using fndtype_t = fundamental_data_t<value_t>;
    static __HOST_DEVICE__ value_t eval(T const& x, U const& y);
  };
  template<typename T, typename U> struct OrFcnal<T, U, arithmetic_domain_tag, complex_domain_tag>{
    using value_t = bool;
    using dtype_t = reduced_data_t<value_t>;
    using fndtype_t = fundamental_data_t<value_t>;
    static __HOST_DEVICE__ value_t eval(T const& x, U const& y);
  };
  template<typename T, typename U> using IvyOr = IvyConditionalFunction_2D<
    T, U,
    OrFcnal<unpack_if_function_t<T>, unpack_if_function_t<U>>
  >;
  template<typename T, typename U, ENABLE_IF_BOOL(!is_pointer_v<T> && !is_pointer_v<U>)>
  __INLINE_FCN_FORCE__ __HOST_DEVICE__ typename OrFcnal<T, U>::value_t Or(T const& x, U const& y);
  /**
   * @brief Construct a lazy Or function node for autodiff.
   * @note  Host-only: function-graph objects (IvyRegularFunction) use
   *        virtual dispatch and RAII, which are incompatible with device code.
   *        Direct numerical evaluation (the non-pointer overload) is __HOST_DEVICE__.
   */
  template<typename T, typename U, ENABLE_IF_BOOL(is_pointer_v<T>&& is_pointer_v<U>)>
  __HOST__ IvyThreadSafePtr_t<typename IvyOr<typename T::element_type, typename U::element_type>::base_t> Or(T const& x, U const& y);

  // XOR
  template<typename T, typename U, typename domain_T = get_domain_t<T>, typename domain_U = get_domain_t<U>> struct XorFcnal{
    using value_t = bool;
    using dtype_t = reduced_data_t<value_t>;
    using fndtype_t = fundamental_data_t<value_t>;
    static __INLINE_FCN_FORCE__ __HOST_DEVICE__ value_t eval(T const& x, U const& y);
  };
  template<typename T, typename U> struct XorFcnal<T, U, real_domain_tag, real_domain_tag>{
    using value_t = bool;
    using dtype_t = reduced_data_t<value_t>;
    using fndtype_t = fundamental_data_t<value_t>;
    static __HOST_DEVICE__ value_t eval(T const& x, U const& y);
  };
  template<typename T, typename U> struct XorFcnal<T, U, arithmetic_domain_tag, real_domain_tag>{
    using value_t = bool;
    using dtype_t = reduced_data_t<value_t>;
    using fndtype_t = fundamental_data_t<value_t>;
    static __HOST_DEVICE__ value_t eval(T const& x, U const& y);
  };
  template<typename T, typename U> struct XorFcnal<T, U, real_domain_tag, arithmetic_domain_tag>{
    using value_t = bool;
    using dtype_t = reduced_data_t<value_t>;
    using fndtype_t = fundamental_data_t<value_t>;
    static __HOST_DEVICE__ value_t eval(T const& x, U const& y);
  };
  template<typename T, typename U> struct XorFcnal<T, U, arithmetic_domain_tag, complex_domain_tag>{
    using value_t = bool;
    using dtype_t = reduced_data_t<value_t>;
    using fndtype_t = fundamental_data_t<value_t>;
    static __HOST_DEVICE__ value_t eval(T const& x, U const& y);
  };
  template<typename T, typename U> using IvyXor = IvyConditionalFunction_2D<
    T, U,
    XorFcnal<unpack_if_function_t<T>, unpack_if_function_t<U>>
  >;
  template<typename T, typename U, ENABLE_IF_BOOL(!is_pointer_v<T> && !is_pointer_v<U>)>
  __INLINE_FCN_FORCE__ __HOST_DEVICE__ typename XorFcnal<T, U>::value_t Xor(T const& x, U const& y);
  /**
   * @brief Construct a lazy Xor function node for autodiff.
   * @note  Host-only: function-graph objects (IvyRegularFunction) use
   *        virtual dispatch and RAII, which are incompatible with device code.
   *        Direct numerical evaluation (the non-pointer overload) is __HOST_DEVICE__.
   */
  template<typename T, typename U, ENABLE_IF_BOOL(is_pointer_v<T>&& is_pointer_v<U>)>
  __HOST__ IvyThreadSafePtr_t<typename IvyXor<typename T::element_type, typename U::element_type>::base_t> Xor(T const& x, U const& y);
  // XOr is split (like Xor) into a __HOST_DEVICE__ non-pointer overload and a __HOST__-only
  // pointer overload, so NVCC never compiles the host-only graph builder in device context.
  template<typename T, typename U, ENABLE_IF_BOOL(!is_pointer_v<T> && !is_pointer_v<U>)>
  __HOST_DEVICE__ auto XOr(T const& x, U const& y) -> decltype(Xor(x, y));
  template<typename T, typename U, ENABLE_IF_BOOL(is_pointer_v<T>&& is_pointer_v<U>)>
  __HOST__ auto XOr(T const& x, U const& y) -> decltype(Xor(x, y));

  // AND
  template<typename T, typename U, typename domain_T = get_domain_t<T>, typename domain_U = get_domain_t<U>> struct AndFcnal{
    using value_t = bool;
    using dtype_t = reduced_data_t<value_t>;
    using fndtype_t = fundamental_data_t<value_t>;
    static __INLINE_FCN_FORCE__ __HOST_DEVICE__ value_t eval(T const& x, U const& y);
  };
  template<typename T, typename U> struct AndFcnal<T, U, real_domain_tag, real_domain_tag>{
    using value_t = bool;
    using dtype_t = reduced_data_t<value_t>;
    using fndtype_t = fundamental_data_t<value_t>;
    static __HOST_DEVICE__ value_t eval(T const& x, U const& y);
  };
  template<typename T, typename U> struct AndFcnal<T, U, arithmetic_domain_tag, real_domain_tag>{
    using value_t = bool;
    using dtype_t = reduced_data_t<value_t>;
    using fndtype_t = fundamental_data_t<value_t>;
    static __HOST_DEVICE__ value_t eval(T const& x, U const& y);
  };
  template<typename T, typename U> struct AndFcnal<T, U, real_domain_tag, arithmetic_domain_tag>{
    using value_t = bool;
    using dtype_t = reduced_data_t<value_t>;
    using fndtype_t = fundamental_data_t<value_t>;
    static __HOST_DEVICE__ value_t eval(T const& x, U const& y);
  };
  template<typename T, typename U> struct AndFcnal<T, U, arithmetic_domain_tag, complex_domain_tag>{
    using value_t = bool;
    using dtype_t = reduced_data_t<value_t>;
    using fndtype_t = fundamental_data_t<value_t>;
    static __HOST_DEVICE__ value_t eval(T const& x, U const& y);
  };
  template<typename T, typename U> using IvyAnd = IvyConditionalFunction_2D<
    T, U,
    AndFcnal<unpack_if_function_t<T>, unpack_if_function_t<U>>
  >;
  template<typename T, typename U, ENABLE_IF_BOOL(!is_pointer_v<T> && !is_pointer_v<U>)>
  __INLINE_FCN_FORCE__ __HOST_DEVICE__ typename AndFcnal<T, U>::value_t And(T const& x, U const& y);
  /**
   * @brief Construct a lazy And function node for autodiff.
   * @note  Host-only: function-graph objects (IvyRegularFunction) use
   *        virtual dispatch and RAII, which are incompatible with device code.
   *        Direct numerical evaluation (the non-pointer overload) is __HOST_DEVICE__.
   */
  template<typename T, typename U, ENABLE_IF_BOOL(is_pointer_v<T>&& is_pointer_v<U>)>
  __HOST__ IvyThreadSafePtr_t<typename IvyAnd<typename T::element_type, typename U::element_type>::base_t> And(T const& x, U const& y);

  // GREATER THAN
  template<typename T, typename U, typename domain_T = get_domain_t<T>, typename domain_U = get_domain_t<U>> struct GreaterThanFcnal{
    using value_t = bool;
    using dtype_t = reduced_data_t<value_t>;
    using fndtype_t = fundamental_data_t<value_t>;
    static __INLINE_FCN_FORCE__ __HOST_DEVICE__ value_t eval(T const& x, U const& y);
  };
  template<typename T, typename U> struct GreaterThanFcnal<T, U, real_domain_tag, real_domain_tag>{
    using value_t = bool;
    using dtype_t = reduced_data_t<value_t>;
    using fndtype_t = fundamental_data_t<value_t>;
    static __HOST_DEVICE__ value_t eval(T const& x, U const& y);
  };
  template<typename T, typename U> struct GreaterThanFcnal<T, U, complex_domain_tag, complex_domain_tag>{
    using value_t = bool;
    using dtype_t = reduced_data_t<value_t>;
    using fndtype_t = fundamental_data_t<value_t>;
    static __HOST_DEVICE__ value_t eval(T const& x, U const& y);
  };
  template<typename T, typename U> struct GreaterThanFcnal<T, U, arithmetic_domain_tag, real_domain_tag>{
    using value_t = bool;
    using dtype_t = reduced_data_t<value_t>;
    using fndtype_t = fundamental_data_t<value_t>;
    static __HOST_DEVICE__ value_t eval(T const& x, U const& y);
  };
  template<typename T, typename U> struct GreaterThanFcnal<T, U, real_domain_tag, arithmetic_domain_tag>{
    using value_t = bool;
    using dtype_t = reduced_data_t<value_t>;
    using fndtype_t = fundamental_data_t<value_t>;
    static __HOST_DEVICE__ value_t eval(T const& x, U const& y);
  };
  template<typename T, typename U> struct GreaterThanFcnal<T, U, arithmetic_domain_tag, complex_domain_tag>{
    using value_t = bool;
    using dtype_t = reduced_data_t<value_t>;
    using fndtype_t = fundamental_data_t<value_t>;
    static __HOST_DEVICE__ value_t eval(T const& x, U const& y);
  };
  template<typename T, typename U> struct GreaterThanFcnal<T, U, complex_domain_tag, arithmetic_domain_tag>{
    using value_t = bool;
    using dtype_t = reduced_data_t<value_t>;
    using fndtype_t = fundamental_data_t<value_t>;
    static __HOST_DEVICE__ value_t eval(T const& x, U const& y);
  };
  template<typename T, typename U> struct GreaterThanFcnal<T, U, real_domain_tag, complex_domain_tag>{
    using value_t = bool;
    using dtype_t = reduced_data_t<value_t>;
    using fndtype_t = fundamental_data_t<value_t>;
    static __HOST_DEVICE__ value_t eval(T const& x, U const& y);
  };
  template<typename T, typename U> struct GreaterThanFcnal<T, U, complex_domain_tag, real_domain_tag>{
    using value_t = bool;
    using dtype_t = reduced_data_t<value_t>;
    using fndtype_t = fundamental_data_t<value_t>;
    static __HOST_DEVICE__ value_t eval(T const& x, U const& y);
  };
  template<typename T, typename U> using IvyGreaterThan = IvyConditionalFunction_2D<
    T, U,
    GreaterThanFcnal<unpack_if_function_t<T>, unpack_if_function_t<U>>
  >;
  template<typename T, typename U, ENABLE_IF_BOOL(!is_pointer_v<T> && !is_pointer_v<U>)>
  __INLINE_FCN_FORCE__ __HOST_DEVICE__ typename GreaterThanFcnal<T, U>::value_t GreaterThan(T const& x, U const& y);
  /**
   * @brief Construct a lazy GreaterThan function node for autodiff.
   * @note  Host-only: function-graph objects (IvyRegularFunction) use
   *        virtual dispatch and RAII, which are incompatible with device code.
   *        Direct numerical evaluation (the non-pointer overload) is __HOST_DEVICE__.
   */
  template<typename T, typename U, ENABLE_IF_BOOL(is_pointer_v<T>&& is_pointer_v<U>)>
  __HOST__ IvyThreadSafePtr_t<typename IvyGreaterThan<typename T::element_type, typename U::element_type>::base_t> GreaterThan(T const& x, U const& y);
  template<typename T, typename U, ENABLE_IF_BOOL(!is_pointer_v<T> && !is_pointer_v<U>)>
  __HOST_DEVICE__ auto GT(T const& x, U const& y) -> decltype(GreaterThan(x, y));
  template<typename T, typename U, ENABLE_IF_BOOL(is_pointer_v<T>&& is_pointer_v<U>)>
  __HOST__ auto GT(T const& x, U const& y) -> decltype(GreaterThan(x, y));

  // LESS THAN
  template<typename T, typename U, typename domain_T = get_domain_t<T>, typename domain_U = get_domain_t<U>> struct LessThanFcnal{
    using value_t = bool;
    using dtype_t = reduced_data_t<value_t>;
    using fndtype_t = fundamental_data_t<value_t>;
    static __INLINE_FCN_FORCE__ __HOST_DEVICE__ value_t eval(T const& x, U const& y);
  };
  template<typename T, typename U> struct LessThanFcnal<T, U, real_domain_tag, real_domain_tag>{
    using value_t = bool;
    using dtype_t = reduced_data_t<value_t>;
    using fndtype_t = fundamental_data_t<value_t>;
    static __HOST_DEVICE__ value_t eval(T const& x, U const& y);
  };
  template<typename T, typename U> struct LessThanFcnal<T, U, complex_domain_tag, complex_domain_tag>{
    using value_t = bool;
    using dtype_t = reduced_data_t<value_t>;
    using fndtype_t = fundamental_data_t<value_t>;
    static __HOST_DEVICE__ value_t eval(T const& x, U const& y);
  };
  template<typename T, typename U> struct LessThanFcnal<T, U, arithmetic_domain_tag, real_domain_tag>{
    using value_t = bool;
    using dtype_t = reduced_data_t<value_t>;
    using fndtype_t = fundamental_data_t<value_t>;
    static __HOST_DEVICE__ value_t eval(T const& x, U const& y);
  };
  template<typename T, typename U> struct LessThanFcnal<T, U, real_domain_tag, arithmetic_domain_tag>{
    using value_t = bool;
    using dtype_t = reduced_data_t<value_t>;
    using fndtype_t = fundamental_data_t<value_t>;
    static __HOST_DEVICE__ value_t eval(T const& x, U const& y);
  };
  template<typename T, typename U> struct LessThanFcnal<T, U, arithmetic_domain_tag, complex_domain_tag>{
    using value_t = bool;
    using dtype_t = reduced_data_t<value_t>;
    using fndtype_t = fundamental_data_t<value_t>;
    static __HOST_DEVICE__ value_t eval(T const& x, U const& y);
  };
  template<typename T, typename U> struct LessThanFcnal<T, U, complex_domain_tag, arithmetic_domain_tag>{
    using value_t = bool;
    using dtype_t = reduced_data_t<value_t>;
    using fndtype_t = fundamental_data_t<value_t>;
    static __HOST_DEVICE__ value_t eval(T const& x, U const& y);
  };
  template<typename T, typename U> struct LessThanFcnal<T, U, real_domain_tag, complex_domain_tag>{
    using value_t = bool;
    using dtype_t = reduced_data_t<value_t>;
    using fndtype_t = fundamental_data_t<value_t>;
    static __HOST_DEVICE__ value_t eval(T const& x, U const& y);
  };
  template<typename T, typename U> struct LessThanFcnal<T, U, complex_domain_tag, real_domain_tag>{
    using value_t = bool;
    using dtype_t = reduced_data_t<value_t>;
    using fndtype_t = fundamental_data_t<value_t>;
    static __HOST_DEVICE__ value_t eval(T const& x, U const& y);
  };
  template<typename T, typename U> using IvyLessThan = IvyConditionalFunction_2D<
    T, U,
    LessThanFcnal<unpack_if_function_t<T>, unpack_if_function_t<U>>
  >;
  template<typename T, typename U, ENABLE_IF_BOOL(!is_pointer_v<T> && !is_pointer_v<U>)>
  __INLINE_FCN_FORCE__ __HOST_DEVICE__ typename LessThanFcnal<T, U>::value_t LessThan(T const& x, U const& y);
  /**
   * @brief Construct a lazy LessThan function node for autodiff.
   * @note  Host-only: function-graph objects (IvyRegularFunction) use
   *        virtual dispatch and RAII, which are incompatible with device code.
   *        Direct numerical evaluation (the non-pointer overload) is __HOST_DEVICE__.
   */
  template<typename T, typename U, ENABLE_IF_BOOL(is_pointer_v<T>&& is_pointer_v<U>)>
  __HOST__ IvyThreadSafePtr_t<typename IvyLessThan<typename T::element_type, typename U::element_type>::base_t> LessThan(T const& x, U const& y);
  template<typename T, typename U, ENABLE_IF_BOOL(!is_pointer_v<T> && !is_pointer_v<U>)>
  __HOST_DEVICE__ auto LT(T const& x, U const& y) -> decltype(LessThan(x, y));
  template<typename T, typename U, ENABLE_IF_BOOL(is_pointer_v<T>&& is_pointer_v<U>)>
  __HOST__ auto LT(T const& x, U const& y) -> decltype(LessThan(x, y));

  // GREATER THAN OR EQUAL TO
  template<typename T, typename U, typename domain_T = get_domain_t<T>, typename domain_U = get_domain_t<U>> struct GreaterOrEqualFcnal{
    using value_t = bool;
    using dtype_t = reduced_data_t<value_t>;
    using fndtype_t = fundamental_data_t<value_t>;
    static __INLINE_FCN_FORCE__ __HOST_DEVICE__ value_t eval(T const& x, U const& y);
  };
  template<typename T, typename U> struct GreaterOrEqualFcnal<T, U, real_domain_tag, real_domain_tag>{
    using value_t = bool;
    using dtype_t = reduced_data_t<value_t>;
    using fndtype_t = fundamental_data_t<value_t>;
    static __HOST_DEVICE__ value_t eval(T const& x, U const& y);
  };
  template<typename T, typename U> struct GreaterOrEqualFcnal<T, U, complex_domain_tag, complex_domain_tag>{
    using value_t = bool;
    using dtype_t = reduced_data_t<value_t>;
    using fndtype_t = fundamental_data_t<value_t>;
    static __HOST_DEVICE__ value_t eval(T const& x, U const& y);
  };
  template<typename T, typename U> struct GreaterOrEqualFcnal<T, U, arithmetic_domain_tag, real_domain_tag>{
    using value_t = bool;
    using dtype_t = reduced_data_t<value_t>;
    using fndtype_t = fundamental_data_t<value_t>;
    static __HOST_DEVICE__ value_t eval(T const& x, U const& y);
  };
  template<typename T, typename U> struct GreaterOrEqualFcnal<T, U, real_domain_tag, arithmetic_domain_tag>{
    using value_t = bool;
    using dtype_t = reduced_data_t<value_t>;
    using fndtype_t = fundamental_data_t<value_t>;
    static __HOST_DEVICE__ value_t eval(T const& x, U const& y);
  };
  template<typename T, typename U> struct GreaterOrEqualFcnal<T, U, arithmetic_domain_tag, complex_domain_tag>{
    using value_t = bool;
    using dtype_t = reduced_data_t<value_t>;
    using fndtype_t = fundamental_data_t<value_t>;
    static __HOST_DEVICE__ value_t eval(T const& x, U const& y);
  };
  template<typename T, typename U> struct GreaterOrEqualFcnal<T, U, complex_domain_tag, arithmetic_domain_tag>{
    using value_t = bool;
    using dtype_t = reduced_data_t<value_t>;
    using fndtype_t = fundamental_data_t<value_t>;
    static __HOST_DEVICE__ value_t eval(T const& x, U const& y);
  };
  template<typename T, typename U> struct GreaterOrEqualFcnal<T, U, real_domain_tag, complex_domain_tag>{
    using value_t = bool;
    using dtype_t = reduced_data_t<value_t>;
    using fndtype_t = fundamental_data_t<value_t>;
    static __HOST_DEVICE__ value_t eval(T const& x, U const& y);
  };
  template<typename T, typename U> struct GreaterOrEqualFcnal<T, U, complex_domain_tag, real_domain_tag>{
    using value_t = bool;
    using dtype_t = reduced_data_t<value_t>;
    using fndtype_t = fundamental_data_t<value_t>;
    static __HOST_DEVICE__ value_t eval(T const& x, U const& y);
  };
  template<typename T, typename U> using IvyGreaterOrEqual = IvyConditionalFunction_2D<
    T, U,
    GreaterOrEqualFcnal<unpack_if_function_t<T>, unpack_if_function_t<U>>
  >;
  template<typename T, typename U, ENABLE_IF_BOOL(!is_pointer_v<T> && !is_pointer_v<U>)>
  __INLINE_FCN_FORCE__ __HOST_DEVICE__ typename GreaterOrEqualFcnal<T, U>::value_t GreaterOrEqual(T const& x, U const& y);
  /**
   * @brief Construct a lazy GreaterOrEqual function node for autodiff.
   * @note  Host-only: function-graph objects (IvyRegularFunction) use
   *        virtual dispatch and RAII, which are incompatible with device code.
   *        Direct numerical evaluation (the non-pointer overload) is __HOST_DEVICE__.
   */
  template<typename T, typename U, ENABLE_IF_BOOL(is_pointer_v<T>&& is_pointer_v<U>)>
  __HOST__ IvyThreadSafePtr_t<typename IvyGreaterOrEqual<typename T::element_type, typename U::element_type>::base_t> GreaterOrEqual(T const& x, U const& y);
  template<typename T, typename U, ENABLE_IF_BOOL(!is_pointer_v<T> && !is_pointer_v<U>)>
  __HOST_DEVICE__ auto GE(T const& x, U const& y) -> decltype(GreaterOrEqual(x, y));
  template<typename T, typename U, ENABLE_IF_BOOL(is_pointer_v<T>&& is_pointer_v<U>)>
  __HOST__ auto GE(T const& x, U const& y) -> decltype(GreaterOrEqual(x, y));
  template<typename T, typename U, ENABLE_IF_BOOL(!is_pointer_v<T> && !is_pointer_v<U>)>
  __HOST_DEVICE__ auto GEQ(T const& x, U const& y) -> decltype(GreaterOrEqual(x, y));
  template<typename T, typename U, ENABLE_IF_BOOL(is_pointer_v<T>&& is_pointer_v<U>)>
  __HOST__ auto GEQ(T const& x, U const& y) -> decltype(GreaterOrEqual(x, y));

  // LESS THAN OR EQUAL TO
  template<typename T, typename U, typename domain_T = get_domain_t<T>, typename domain_U = get_domain_t<U>> struct LessOrEqualFcnal{
    using value_t = bool;
    using dtype_t = reduced_data_t<value_t>;
    using fndtype_t = fundamental_data_t<value_t>;
    static __INLINE_FCN_FORCE__ __HOST_DEVICE__ value_t eval(T const& x, U const& y);
  };
  template<typename T, typename U> struct LessOrEqualFcnal<T, U, real_domain_tag, real_domain_tag>{
    using value_t = bool;
    using dtype_t = reduced_data_t<value_t>;
    using fndtype_t = fundamental_data_t<value_t>;
    static __HOST_DEVICE__ value_t eval(T const& x, U const& y);
  };
  template<typename T, typename U> struct LessOrEqualFcnal<T, U, complex_domain_tag, complex_domain_tag>{
    using value_t = bool;
    using dtype_t = reduced_data_t<value_t>;
    using fndtype_t = fundamental_data_t<value_t>;
    static __HOST_DEVICE__ value_t eval(T const& x, U const& y);
  };
  template<typename T, typename U> struct LessOrEqualFcnal<T, U, arithmetic_domain_tag, real_domain_tag>{
    using value_t = bool;
    using dtype_t = reduced_data_t<value_t>;
    using fndtype_t = fundamental_data_t<value_t>;
    static __HOST_DEVICE__ value_t eval(T const& x, U const& y);
  };
  template<typename T, typename U> struct LessOrEqualFcnal<T, U, real_domain_tag, arithmetic_domain_tag>{
    using value_t = bool;
    using dtype_t = reduced_data_t<value_t>;
    using fndtype_t = fundamental_data_t<value_t>;
    static __HOST_DEVICE__ value_t eval(T const& x, U const& y);
  };
  template<typename T, typename U> struct LessOrEqualFcnal<T, U, arithmetic_domain_tag, complex_domain_tag>{
    using value_t = bool;
    using dtype_t = reduced_data_t<value_t>;
    using fndtype_t = fundamental_data_t<value_t>;
    static __HOST_DEVICE__ value_t eval(T const& x, U const& y);
  };
  template<typename T, typename U> struct LessOrEqualFcnal<T, U, complex_domain_tag, arithmetic_domain_tag>{
    using value_t = bool;
    using dtype_t = reduced_data_t<value_t>;
    using fndtype_t = fundamental_data_t<value_t>;
    static __HOST_DEVICE__ value_t eval(T const& x, U const& y);
  };
  template<typename T, typename U> struct LessOrEqualFcnal<T, U, real_domain_tag, complex_domain_tag>{
    using value_t = bool;
    using dtype_t = reduced_data_t<value_t>;
    using fndtype_t = fundamental_data_t<value_t>;
    static __HOST_DEVICE__ value_t eval(T const& x, U const& y);
  };
  template<typename T, typename U> struct LessOrEqualFcnal<T, U, complex_domain_tag, real_domain_tag>{
    using value_t = bool;
    using dtype_t = reduced_data_t<value_t>;
    using fndtype_t = fundamental_data_t<value_t>;
    static __HOST_DEVICE__ value_t eval(T const& x, U const& y);
  };
  template<typename T, typename U> using IvyLessOrEqual = IvyConditionalFunction_2D<
    T, U,
    LessOrEqualFcnal<unpack_if_function_t<T>, unpack_if_function_t<U>>
  >;
  template<typename T, typename U, ENABLE_IF_BOOL(!is_pointer_v<T> && !is_pointer_v<U>)>
  __INLINE_FCN_FORCE__ __HOST_DEVICE__ typename LessOrEqualFcnal<T, U>::value_t LessOrEqual(T const& x, U const& y);
  /**
   * @brief Construct a lazy LessOrEqual function node for autodiff.
   * @note  Host-only: function-graph objects (IvyRegularFunction) use
   *        virtual dispatch and RAII, which are incompatible with device code.
   *        Direct numerical evaluation (the non-pointer overload) is __HOST_DEVICE__.
   */
  template<typename T, typename U, ENABLE_IF_BOOL(is_pointer_v<T>&& is_pointer_v<U>)>
  __HOST__ IvyThreadSafePtr_t<typename IvyLessOrEqual<typename T::element_type, typename U::element_type>::base_t> LessOrEqual(T const& x, U const& y);
  template<typename T, typename U, ENABLE_IF_BOOL(!is_pointer_v<T> && !is_pointer_v<U>)>
  __HOST_DEVICE__ auto LE(T const& x, U const& y) -> decltype(LessOrEqual(x, y));
  template<typename T, typename U, ENABLE_IF_BOOL(is_pointer_v<T>&& is_pointer_v<U>)>
  __HOST__ auto LE(T const& x, U const& y) -> decltype(LessOrEqual(x, y));
  template<typename T, typename U, ENABLE_IF_BOOL(!is_pointer_v<T> && !is_pointer_v<U>)>
  __HOST_DEVICE__ auto LEQ(T const& x, U const& y) -> decltype(LessOrEqual(x, y));
  template<typename T, typename U, ENABLE_IF_BOOL(is_pointer_v<T>&& is_pointer_v<U>)>
  __HOST__ auto LEQ(T const& x, U const& y) -> decltype(LessOrEqual(x, y));

}


#endif
