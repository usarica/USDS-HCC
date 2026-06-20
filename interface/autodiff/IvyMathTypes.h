#ifndef IVYMATHTYPES_H
#define IVYMATHTYPES_H


#include "autodiff/IvyBaseMathTypes.h"


namespace IvyMath{
  template<typename T, typename U> struct more_precise{
    using ctype_T = unpack_if_function_t<T>;
    using ctype_U = unpack_if_function_t<U>;
    using fund_type = more_precise_fundamental_t<T, U>;

    static constexpr bool is_arithmetic_T = is_arithmetic_v<ctype_T>;
    static constexpr bool is_var_T = is_real_v<ctype_T>;
    static constexpr bool is_complex_T = is_complex_v<ctype_T>;
    static constexpr bool is_quaternion_T = is_quaternion_v<ctype_T>;
    static constexpr bool is_tensor_T = is_tensor_v<ctype_T>;

    static constexpr bool is_arithmetic_U = is_arithmetic_v<ctype_U>;
    static constexpr bool is_var_U = is_real_v<ctype_U>;
    static constexpr bool is_complex_U = is_complex_v<ctype_U>;
    static constexpr bool is_quaternion_U = is_quaternion_v<ctype_U>;
    static constexpr bool is_tensor_U = is_tensor_v<ctype_U>;

    // Precedence (highest first): tensor > quaternion > complex > real(scalar) > arithmetic.
    static constexpr bool swap_T_U = (
      is_tensor_U
      ||
      (!is_tensor_T && is_quaternion_U)
      ||
      (!is_tensor_T && !is_quaternion_T && is_complex_U)
      ||
      (!is_tensor_T && !is_quaternion_T && !is_complex_T && is_var_U)
      ||
      (!is_tensor_T && !is_quaternion_T && !is_complex_T && !is_var_T && is_arithmetic_U)
      );

    using left_type = std_ttraits::conditional_t<swap_T_U, ctype_U, ctype_T>;
    using right_type = std_ttraits::conditional_t<swap_T_U, ctype_T, ctype_U>;

    static constexpr bool is_arithmetic_left = (swap_T_U ? is_arithmetic_U : is_arithmetic_T);
    static constexpr bool is_var_left = (swap_T_U ? is_var_U : is_var_T);
    static constexpr bool is_complex_left = (swap_T_U ? is_complex_U : is_complex_T);
    static constexpr bool is_quaternion_left = (swap_T_U ? is_quaternion_U : is_quaternion_T);
    static constexpr bool is_tensor_left = (swap_T_U ? is_tensor_U : is_tensor_T);

    using type = std_ttraits::conditional_t<
      is_tensor_left,
      IvyTensor<fund_type>,
      std_ttraits::conditional_t<
        is_quaternion_left,
        IvyQuaternion<fund_type>,
        std_ttraits::conditional_t<
          is_complex_left,
          IvyComplex<fund_type>,
          std_ttraits::conditional_t<
            is_var_left,
            IvyScalar<fund_type>,
            fund_type
          >
        >
      >
    >;
  };
  template<typename T, typename U> using more_precise_t = typename more_precise<T, U>::type;

  // The precision-reduced form of more_precise is only needed for functions.
  template<typename T, typename U> struct more_precise_reduced{
    using ftype_T = unpack_if_function_t<T>;
    using ftype_U = unpack_if_function_t<U>;
    using ctype_T = minimal_fcn_output_t<fundamental_data_t<ftype_T>, get_domain_t<ftype_T>, get_operability_t<ftype_T>>;
    using ctype_U = minimal_fcn_output_t<fundamental_data_t<ftype_U>, get_domain_t<ftype_U>, get_operability_t<ftype_U>>;
    using vtype_T = reduced_value_t<ctype_T>;
    using vtype_U = reduced_value_t<ctype_U>;
    using type = reduced_value_t<more_precise_t<vtype_T, vtype_U>>;
  };
  template<typename T, typename U> using more_precise_reduced_t = typename more_precise_reduced<T, U>::type;

  // Structs to elevate pointers to functions
  template<typename T> struct elevateToRealFcnPtr_if_ptr{ using type = reduced_value_t<convert_to_real_t<T>>; };
  template<typename T> using elevateToRealFcnPtr_if_ptr_t = typename elevateToRealFcnPtr_if_ptr<T>::type;

  template<typename T> struct elevateToFloatFcnPtr_if_ptr{ using type = reduced_value_t<convert_to_floating_point_t<T>>; };
  template<typename T> using elevateToFloatFcnPtr_if_ptr_t = typename elevateToFloatFcnPtr_if_ptr<T>::type;

  template<typename T> struct elevateToFloatFcnPtr_if_ptrComplex{ using type = reduced_value_t<convert_to_floating_point_if_complex_t<T>>; };
  template<typename T> using elevateToFloatFcnPtr_if_ptrComplex_t = typename elevateToFloatFcnPtr_if_ptrComplex<T>::type;

  // Declarations of special cases for Ivy variables and pointers
  template<typename T> struct elevateToFcnPtr_if_ptr<IvyTensor<T>>;
  template<typename T> struct elevateToRealFcnPtr_if_ptr<IvyTensor<T>>;
  template<typename T> struct elevateToFloatFcnPtr_if_ptr<IvyTensor<T>>;
  template<typename T> struct elevateToFloatFcnPtr_if_ptrComplex<IvyTensor<T>>;

  // Definitions of these special cases
  template<typename T> struct elevateToFcnPtr_if_ptr<IvyTensor<T>>{ using type = IvyTensor< elevateToFcnPtr_if_ptr_t<T> >; };
  template<typename T> struct elevateToRealFcnPtr_if_ptr<IvyTensor<T>>{ using type = IvyTensor< elevateToRealFcnPtr_if_ptr_t<T> >; };
  template<typename T> struct elevateToFloatFcnPtr_if_ptr<IvyTensor<T>>{ using type = IvyTensor< elevateToFloatFcnPtr_if_ptr_t<T> >; };
  template<typename T> struct elevateToFloatFcnPtr_if_ptrComplex<IvyTensor<T>>{ using type = IvyTensor< elevateToFloatFcnPtr_if_ptrComplex_t<T> >; };


  // Declarations of special cases for tensors and pointers:
  // Tensor types can be anything, so if make sure that complex tensor values are handled correctly.
  // When dealing with pointers, we should always be dealing with pairs, and the type returned should be an IvyFunction
  // as this is the only realistic case to compare pointer types.
  // Definitions of special cases
  template<typename T, typename U> struct more_precise<T, IvyTensor<U>>{ using type = IvyTensor< more_precise_t<T, U> >; };
  template<typename T, typename U> struct more_precise<IvyTensor<T>, U>{ using type = IvyTensor< more_precise_t<T, U> >; };
  template<typename T, typename U> struct more_precise<IvyTensor<T>, IvyTensor<U>>{ using type = IvyTensor< more_precise_t<T, U> >; };
  template<typename T, typename U> struct more_precise_reduced<T, IvyTensor<U>>{ using type = IvyTensor< more_precise_reduced_t<T, U> >; };
  template<typename T, typename U> struct more_precise_reduced<IvyTensor<T>, U>{ using type = IvyTensor< more_precise_reduced_t<T, U> >; };
  template<typename T, typename U> struct more_precise_reduced<IvyTensor<T>, IvyTensor<U>>{ using type = IvyTensor< more_precise_reduced_t<T, U> >; };

}


#endif
