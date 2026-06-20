#ifndef IVYTENSORCELL_H
#define IVYTENSORCELL_H

/**
 * @file IvyTensorCell.h
 * @brief Lightweight value-type leaf cells used as the element type of a
 *        contiguous (struct-of-arrays) differentiable tensor leaf.
 *
 * The framework distinguishes a node's *domain* (real / complex) from its
 * *value* operability (variable / constant). The heavy leaf nodes cover three
 * of those combinations: @c IvyVariable (real, variable), @c IvyConstant
 * (real, constant) and @c IvyComplexVariable (complex, variable). Each carries
 * an @c IvyClientManager (96 bytes + a heap-allocated client vector), which is
 * what makes a tensor *of* such nodes expensive when stored element-by-element.
 *
 * The cells here are the value-type counterparts of those leaves, stripped of
 * the per-element @c IvyClientManager (the owning tensor node carries the one
 * client manager that tracks graph dependents, so per-element tracking is
 * redundant) and of the dead @c infinitesimal_ residue. They therefore occupy
 * exactly the storage of their payload:
 *   - @c IvyTensorRealCell<T, value_tag>     -> one T          (sizeof(T))
 *   - @c IvyTensorComplexCell<T, value_tag>  -> {re, im}       (2*sizeof(T))
 *
 * This is provided for *every* domain/value combination so that a contiguous
 * tensor leaf can be built for real variables, real constants, complex
 * variables and complex constants alike, each carrying the correct
 * @c domain_tag / @c value_tag so the existing Fcnal / operability machinery
 * treats it exactly like its heavy counterpart.
 *
 * Convenience aliases:
 *   - IvyTensorVariableCell<T>         = IvyTensorRealCell<T, variable_value_tag>
 *   - IvyTensorConstantCell<T>         = IvyTensorRealCell<T, constant_value_tag>
 *   - IvyTensorComplexVariableCell<T>  = IvyTensorComplexCell<T, variable_value_tag>
 *   - IvyTensorComplexConstantCell<T>  = IvyTensorComplexCell<T, constant_value_tag>
 */

#include "config/IvyCompilerConfig.h"
#include "std_ivy/IvyCmath.h"
#include "autodiff/IvyBaseMathTypes.h"
#include "autodiff/base_types/IvyNodeRelations.h"
#include "autodiff/basic_nodes/IvyComplexVariable.h"


namespace IvyMath{
  template<typename T, typename ValueTag, ENABLE_IF_ARITHMETIC(T)> class IvyTensorRealCell;
  template<typename T, typename ValueTag, ENABLE_IF_ARITHMETIC(T)> class IvyTensorComplexCell;
  template<typename T, typename ValueTag> struct IvyNodeSelfRelations<IvyTensorRealCell<T, ValueTag>>;
  template<typename T, typename ValueTag> struct IvyNodeSelfRelations<IvyTensorComplexCell<T, ValueTag>>;
}

//================================ REAL CELL ================================
namespace IvyMath{
  template<typename T, typename ValueTag, ENABLE_IF_ARITHMETIC_IMPL(T)> class IvyTensorRealCell final :
    public IvyBaseNode,
    public real_domain_tag,
    public ValueTag
  {
  public:
    using dtype_t = T;
    using value_t = T;
    using value_tag_t = ValueTag;
    static constexpr bool is_variable_cell = std_ttraits::is_same_v<ValueTag, variable_value_tag>;

  protected:
    value_t value_;

  public:
    __HOST_DEVICE__ IvyTensorRealCell() : value_(0){}
    template<typename U, ENABLE_IF_ARITHMETIC(U)> __HOST_DEVICE__ IvyTensorRealCell(U const& value) : value_(__STATIC_CAST__(T, value)){}
    __HOST_DEVICE__ IvyTensorRealCell(T const& value) : value_(value){}
    __HOST_DEVICE__ IvyTensorRealCell(T&& value) : value_(std_util::move(value)){}
    template<typename U> __HOST_DEVICE__ IvyTensorRealCell(IvyTensorRealCell<U, ValueTag> const& other) : value_(__STATIC_CAST__(T, other.value())){}
    __HOST_DEVICE__ IvyTensorRealCell(IvyTensorRealCell<T, ValueTag> const& other) : value_(other.value_){}
    __HOST_DEVICE__ IvyTensorRealCell(IvyTensorRealCell<T, ValueTag>&& other) : value_(std_util::move(other.value_)){}
    __HOST_DEVICE__ ~IvyTensorRealCell(){}

    template<typename U> __HOST_DEVICE__ IvyTensorRealCell<T, ValueTag>& operator=(IvyTensorRealCell<U, ValueTag> const& other){ this->value_ = __STATIC_CAST__(T, other.value()); return *this; }
    __HOST_DEVICE__ IvyTensorRealCell<T, ValueTag>& operator=(IvyTensorRealCell<T, ValueTag> const& other){ this->value_ = other.value_; return *this; }
    __HOST_DEVICE__ IvyTensorRealCell<T, ValueTag>& operator=(IvyTensorRealCell<T, ValueTag>&& other){ this->value_ = std_util::move(other.value_); return *this; }
    template<typename U, ENABLE_IF_ARITHMETIC(U)> __HOST_DEVICE__ IvyTensorRealCell<T, ValueTag>& operator=(U const& value){ this->value_ = __STATIC_CAST__(T, value); return *this; }
    __HOST_DEVICE__ IvyTensorRealCell<T, ValueTag>& operator=(T const& value){ this->value_ = value; return *this; }
    __HOST_DEVICE__ IvyTensorRealCell<T, ValueTag>& operator=(T&& value){ this->value_ = std_util::move(value); return *this; }

    __HOST_DEVICE__ void set_value(T const& value){ this->value_ = value; }
    __HOST_DEVICE__ value_t const& value() const{ return this->value_; }

    __HOST_DEVICE__ constexpr bool is_differentiable() __NOEXCEPT__ { return is_variable_cell; }

    friend struct IvyNodeSelfRelations<IvyTensorRealCell<T, ValueTag>>;
  };
}
namespace IvyTypes{
  template<typename T, typename ValueTag> struct convert_to_floating_point<IvyMath::IvyTensorRealCell<T, ValueTag>>{
    using type = IvyMath::IvyTensorRealCell<convert_to_floating_point_t<T>, ValueTag>;
  };
}
namespace IvyMath{
  template<typename T, typename ValueTag> struct IvyNodeSelfRelations<IvyTensorRealCell<T, ValueTag>>{
    static __HOST_DEVICE__ constexpr bool is_differentiable(IvyTensorRealCell<T, ValueTag> const& /*x*/) __NOEXCEPT__ {
      return std_ttraits::is_same_v<ValueTag, variable_value_tag>;
    }
    static __HOST_DEVICE__ void conjugate(IvyTensorRealCell<T, ValueTag>& /*x*/) __NOEXCEPT__ {}
    static constexpr bool is_conjugatable = false;
  };
  template<typename T, typename ValueTag> struct convert_to_real_type<IvyTensorRealCell<T, ValueTag>>{ using type = IvyTensorRealCell<T, ValueTag>; };
}

//============================== COMPLEX CELL ==============================
namespace IvyMath{
  template<typename T, typename ValueTag, ENABLE_IF_ARITHMETIC_IMPL(T)> class IvyTensorComplexCell final :
    public IvyBaseNode,
    public complex_domain_tag,
    public ValueTag
  {
  public:
    using dtype_t = T;
    using value_t = IvyTensorComplexCell<T, ValueTag>;
    using value_tag_t = ValueTag;
    static constexpr bool is_variable_cell = std_ttraits::is_same_v<ValueTag, variable_value_tag>;

  protected:
    dtype_t re;
    dtype_t im;

  public:
    __HOST_DEVICE__ IvyTensorComplexCell() : re(0), im(0){}
    template<typename U, ENABLE_IF_ARITHMETIC(U)> __HOST_DEVICE__ IvyTensorComplexCell(U const& re_) : re(__STATIC_CAST__(T, re_)), im(0){}
    __HOST_DEVICE__ IvyTensorComplexCell(T const& re_) : re(re_), im(0){}
    __HOST_DEVICE__ IvyTensorComplexCell(T const& re_, T const& im_) : re(re_), im(im_){}
    __HOST_DEVICE__ IvyTensorComplexCell(T&& re_) : re(std_util::move(re_)), im(0){}
    __HOST_DEVICE__ IvyTensorComplexCell(T&& re_, T&& im_) : re(std_util::move(re_)), im(std_util::move(im_)){}
    template<typename U> __HOST_DEVICE__ IvyTensorComplexCell(IvyTensorComplexCell<U, ValueTag> const& other) : re(__STATIC_CAST__(T, other.Re())), im(__STATIC_CAST__(T, other.Im())){}
    __HOST_DEVICE__ IvyTensorComplexCell(IvyTensorComplexCell<T, ValueTag> const& other) : re(other.re), im(other.im){}
    __HOST_DEVICE__ IvyTensorComplexCell(IvyTensorComplexCell<T, ValueTag>&& other) : re(std_util::move(other.re)), im(std_util::move(other.im)){}
    template<typename U> __HOST_DEVICE__ IvyTensorComplexCell(IvyTensorRealCell<U, ValueTag> const& other) : re(__STATIC_CAST__(T, other.value())), im(0){}
    __HOST_DEVICE__ ~IvyTensorComplexCell(){}

    template<typename U> __HOST_DEVICE__ IvyTensorComplexCell& operator=(IvyTensorComplexCell<U, ValueTag> const& other){ re = __STATIC_CAST__(T, other.Re()); im = __STATIC_CAST__(T, other.Im()); return *this; }
    __HOST_DEVICE__ IvyTensorComplexCell& operator=(IvyTensorComplexCell<T, ValueTag> const& other){ re = other.re; im = other.im; return *this; }
    __HOST_DEVICE__ IvyTensorComplexCell& operator=(IvyTensorComplexCell<T, ValueTag>&& other){ re = std_util::move(other.re); im = std_util::move(other.im); return *this; }
    template<typename U, ENABLE_IF_ARITHMETIC(U)> __HOST_DEVICE__ IvyTensorComplexCell& operator=(U const& re_){ re = __STATIC_CAST__(T, re_); im = T(0); return *this; }
    __HOST_DEVICE__ IvyTensorComplexCell& operator=(T const& re_){ re = re_; im = T(0); return *this; }
    __HOST_DEVICE__ IvyTensorComplexCell& operator=(T&& re_){ re = std_util::move(re_); im = T(0); return *this; }

    __HOST_DEVICE__ T const& Re() const{ return re; }
    __HOST_DEVICE__ T const& Im() const{ return im; }
    __HOST_DEVICE__ T norm() const{ return std_math::sqrt(re*re + im*im); }
    __HOST_DEVICE__ T phase() const{ return std_math::atan2(im, re); }

    __HOST_DEVICE__ value_t const& value() const{ return *this; }

    __HOST_DEVICE__ void set_real(T const& x){ re = x; }
    __HOST_DEVICE__ void set_imaginary(T const& x){ im = x; }
    __HOST_DEVICE__ void set_absval_phase(T const& v, T const& phi){ re = v*std_math::cos(phi); im = v*std_math::sin(phi); }

    friend struct IvyNodeSelfRelations<IvyTensorComplexCell<T, ValueTag>>;
  };
}
namespace IvyTypes{
  template<typename T, typename ValueTag> struct convert_to_floating_point<IvyMath::IvyTensorComplexCell<T, ValueTag>>{
    using type = IvyMath::IvyTensorComplexCell<convert_to_floating_point_t<T>, ValueTag>;
  };
}
namespace IvyMath{
  template<typename T, typename ValueTag> struct IvyNodeSelfRelations<IvyTensorComplexCell<T, ValueTag>>{
    // Mirror IvyComplexVariable: a complex leaf is not treated as a differentiation seed.
    static __HOST_DEVICE__ constexpr bool is_differentiable(IvyTensorComplexCell<T, ValueTag> const& /*x*/){ return false; }
    static __HOST_DEVICE__ void conjugate(IvyTensorComplexCell<T, ValueTag>& x){ x.im = -x.im; }
    static constexpr bool is_conjugatable = true;
  };
  template<typename T, typename ValueTag> struct convert_to_floating_point_if_complex<IvyTensorComplexCell<T, ValueTag>>{
    using type = IvyTensorComplexCell<convert_to_floating_point_t<T>, ValueTag>;
  };
  template<typename T, typename ValueTag> struct convert_to_real_type<IvyTensorComplexCell<T, ValueTag>>{
    using type = IvyTensorRealCell<T, ValueTag>;
  };
  template<typename T, typename ValueTag> struct convert_to_complex_type<IvyTensorRealCell<T, ValueTag>>{
    using type = IvyTensorComplexCell<T, ValueTag>;
  };
  template<typename T, typename ValueTag> struct convert_to_complex_type<IvyTensorComplexCell<T, ValueTag>>{
    using type = IvyTensorComplexCell<T, ValueTag>;
  };
}

//================================ ALIASES ================================
namespace IvyMath{
  template<typename T> using IvyTensorVariableCell        = IvyTensorRealCell<T, variable_value_tag>;
  template<typename T> using IvyTensorConstantCell        = IvyTensorRealCell<T, constant_value_tag>;
  template<typename T> using IvyTensorComplexVariableCell = IvyTensorComplexCell<T, variable_value_tag>;
  template<typename T> using IvyTensorComplexConstantCell = IvyTensorComplexCell<T, constant_value_tag>;
}

//============================== PRINTOUT ==============================
namespace std_ivy{
  template<typename T, typename ValueTag> struct value_printout<IvyMath::IvyTensorRealCell<T, ValueTag>>{
    static __HOST_DEVICE__ void print(IvyMath::IvyTensorRealCell<T, ValueTag> const& var){
      __PRINT_INFO__("Cell(");
      print_value(var.value(), false);
      __PRINT_INFO__(")");
    }
  };
  template<typename T, typename ValueTag> struct value_printout<IvyMath::IvyTensorComplexCell<T, ValueTag>>{
    static __HOST_DEVICE__ void print(IvyMath::IvyTensorComplexCell<T, ValueTag> const& var){
      __PRINT_INFO__("CellComplex(");
      print_value(var.Re(), false);
      if (var.Im()<T(0)){ __PRINT_INFO__(" - "); print_value(-var.Im(), false); __PRINT_INFO__("i"); }
      else{ __PRINT_INFO__(" + "); print_value(var.Im(), false); __PRINT_INFO__("i"); }
      __PRINT_INFO__(")");
    }
  };
}


#endif
