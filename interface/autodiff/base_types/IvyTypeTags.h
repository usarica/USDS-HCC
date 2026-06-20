#ifndef IVYTYPETAGS_H
#define IVYTYPETAGS_H


#include "std_ivy/IvyTypeTraits.h"
#include "autodiff/base_types/IvyThreadSafePtr.h"


namespace IvyMath{
  // Domains
  struct undefined_domain_tag{};
  struct arithmetic_domain_tag{};
  struct real_domain_tag{};
  struct complex_domain_tag{};
  struct tensor_domain_tag{};
  // Domain getter
  template<typename T> struct get_domain{
    using tag = std_ttraits::conditional_t<
      std_ttraits::is_base_of_v<real_domain_tag, T>,
      real_domain_tag, std_ttraits::conditional_t<
        std_ttraits::is_base_of_v<complex_domain_tag, T>,
        complex_domain_tag, std_ttraits::conditional_t<
          std_ttraits::is_base_of_v<tensor_domain_tag, T>,
          tensor_domain_tag, std_ttraits::conditional_t<
            std_ttraits::is_arithmetic_v<T>,
            arithmetic_domain_tag, undefined_domain_tag
          >
        >
      >
    >;
  };
  template<typename T> struct get_domain<T*>{ using tag = typename get_domain<T>::tag; };
  template<typename T> struct get_domain<IvyThreadSafePtr_t<T>>{ using tag = typename get_domain<T>::tag; };
  template<typename T> using get_domain_t = typename get_domain<T>::tag;
  // Convenience functions
  template<typename T> inline constexpr bool is_arithmetic_v = std_ttraits::is_arithmetic_v<T>;
  template<typename T> inline constexpr bool is_real_v = std_ttraits::is_same_v<real_domain_tag, get_domain_t<T>>;
  template<typename T> inline constexpr bool is_complex_v = std_ttraits::is_same_v<complex_domain_tag, get_domain_t<T>>;
  template<typename T> inline constexpr bool is_tensor_v = std_ttraits::is_same_v<tensor_domain_tag, get_domain_t<T>>;

  // True when T (after unwrapping pointers/IvyThreadSafePtr) carries an Ivy math domain
  // (real, complex, or tensor) — i.e. it is an Ivy node type, not a bare arithmetic value or an
  // unrelated foreign type. Used to constrain the global operators so they do not get selected
  // for foreign types (e.g. std::chrono), which previously caused hard errors via ADL.
  template<typename T> inline constexpr bool is_ivy_domain_v = is_real_v<T> || is_complex_v<T> || is_tensor_v<T>;

  // Operability properties
  // The constant/variable distinction was dropped: differentiation is fully
  // dynamic (by address identity), so every leaf is simply a differentiable
  // value. The operability axis is therefore { value-leaf, function }, with the
  // value-leaf being the default for any type that does not declare itself a
  // function.
  struct leaf_value_tag{};
  struct function_value_tag{};
  // Operability getter
  template<typename T> struct get_operability{
    using tag = std_ttraits::conditional_t<
      std_ttraits::is_base_of_v<function_value_tag, T>,
      function_value_tag, leaf_value_tag
    >;
  };
  template<typename T> struct get_operability<T*>{ using tag = typename get_operability<T>::tag; };
  template<typename T> struct get_operability<IvyThreadSafePtr_t<T>>{ using tag = typename get_operability<T>::tag; };
  template<typename T> using get_operability_t = typename get_operability<T>::tag;
  // Convenience functions
  template<typename T> inline constexpr bool is_leaf_v = std_ttraits::is_same_v<leaf_value_tag, get_operability_t<T>>;
  template<typename T> inline constexpr bool is_function_v = std_ttraits::is_same_v<function_value_tag, get_operability_t<T>>;

  /*
  minimal_domain_t:
  This is a helper struct to get the minimal class for given domain and operability tags, and a precision type.
  It is further specialized in IvyScalar, IvyComplex, IvyTensor, and IvyFunction
  for their own domain and operability tags.
  */
  template<typename precision_type, typename domain_tag, typename operability_tag, typename gradient_domain_tag=domain_tag>
  struct minimal_domain_type{ using type = std_ttraits::remove_cv_t<precision_type>; };
  template<typename precision_type, typename operability_tag, typename gradient_domain_tag>
  struct minimal_domain_type<precision_type, undefined_domain_tag, operability_tag, gradient_domain_tag>{};
  template<typename precision_type, typename domain_tag, typename operability_tag>
  using minimal_domain_t = typename minimal_domain_type<precision_type, domain_tag, operability_tag>::type;

  /*
  minimal_fcn_output_t:
  This is a helper struct to get the minimal class for function output.
  */
  template<typename precision_type, typename domain_tag, typename operability_tag>
  struct minimal_fcn_output_type{ using type = minimal_domain_t<precision_type, domain_tag, operability_tag>; };
  template<typename precision_type, typename operability_tag>
  struct minimal_fcn_output_type<precision_type, undefined_domain_tag, operability_tag>{};
  template<typename precision_type, typename domain_tag, typename operability_tag>
  using minimal_fcn_output_t = typename minimal_fcn_output_type<precision_type, domain_tag, operability_tag>::type;
}


#endif
