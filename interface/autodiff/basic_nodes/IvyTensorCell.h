#ifndef IVYTENSORCELL_H
#define IVYTENSORCELL_H

/**
 * @file IvyTensorCell.h
 * @brief Lightweight value-type leaf cells used as the element type of a
 *        contiguous (struct-of-arrays) differentiable tensor leaf.
 *
 * The framework distinguishes a node's *domain* (real / complex). With the
 * constant/variable operability distinction dropped, every leaf is simply a
 * differentiable *value*. The heavy leaf nodes are @c IvyScalar (real) and
 * @c IvyComplex (complex). Each carries an @c IvyClientManager (96 bytes + a
 * heap-allocated client vector), which is what makes a tensor *of* such nodes
 * expensive when stored element-by-element.
 *
 * The cells here are the value-type counterparts of those leaves, stripped of
 * the per-element @c IvyClientManager (the owning tensor node carries the one
 * client manager that tracks graph dependents, so per-element tracking is
 * redundant). They therefore occupy exactly the storage of their payload:
 *   - @c IvyTensorScalarCell<T>   -> one T          (sizeof(T))
 *   - @c IvyTensorComplexCell<T>  -> {re, im}       (2*sizeof(T))
 *
 * Each carries the correct @c real_domain_tag / @c complex_domain_tag plus the
 * single @c leaf_value_tag so the existing Fcnal / operability machinery treats
 * it exactly like its heavy counterpart.
 */

#include "config/IvyCompilerConfig.h"
#include "std_ivy/IvyCmath.h"
#include "autodiff/IvyBaseMathTypes.h"
#include "autodiff/base_types/IvyNodeRelations.h"
#include "autodiff/basic_nodes/IvyComplex.h"


namespace IvyMath{
  template<typename T, ENABLE_IF_ARITHMETIC(T)> class IvyTensorScalarCell;
  template<typename T, ENABLE_IF_ARITHMETIC(T)> class IvyTensorComplexCell;
  template<typename T> struct IvyNodeSelfRelations<IvyTensorScalarCell<T>>;
  template<typename T> struct IvyNodeSelfRelations<IvyTensorComplexCell<T>>;
}

//================================ REAL CELL ================================
namespace IvyMath{
  template<typename T, ENABLE_IF_ARITHMETIC_IMPL(T)> class IvyTensorScalarCell final :
    public IvyBaseNode,
    public real_domain_tag,
    public leaf_value_tag
  {
  public:
    using dtype_t = T;
    using value_t = T;

  protected:
    value_t value_;

  public:
    __HOST_DEVICE__ IvyTensorScalarCell() : value_(0){}
    template<typename U, ENABLE_IF_ARITHMETIC(U)> __HOST_DEVICE__ IvyTensorScalarCell(U const& value) : value_(__STATIC_CAST__(T, value)){}
    __HOST_DEVICE__ IvyTensorScalarCell(T const& value) : value_(value){}
    __HOST_DEVICE__ IvyTensorScalarCell(T&& value) : value_(std_util::move(value)){}
    template<typename U> __HOST_DEVICE__ IvyTensorScalarCell(IvyTensorScalarCell<U> const& other) : value_(__STATIC_CAST__(T, other.value())){}
    __HOST_DEVICE__ IvyTensorScalarCell(IvyTensorScalarCell<T> const& other) : value_(other.value_){}
    __HOST_DEVICE__ IvyTensorScalarCell(IvyTensorScalarCell<T>&& other) : value_(std_util::move(other.value_)){}
    __HOST_DEVICE__ ~IvyTensorScalarCell(){}

    template<typename U> __HOST_DEVICE__ IvyTensorScalarCell<T>& operator=(IvyTensorScalarCell<U> const& other){ this->value_ = __STATIC_CAST__(T, other.value()); return *this; }
    __HOST_DEVICE__ IvyTensorScalarCell<T>& operator=(IvyTensorScalarCell<T> const& other){ this->value_ = other.value_; return *this; }
    __HOST_DEVICE__ IvyTensorScalarCell<T>& operator=(IvyTensorScalarCell<T>&& other){ this->value_ = std_util::move(other.value_); return *this; }
    template<typename U, ENABLE_IF_ARITHMETIC(U)> __HOST_DEVICE__ IvyTensorScalarCell<T>& operator=(U const& value){ this->value_ = __STATIC_CAST__(T, value); return *this; }
    __HOST_DEVICE__ IvyTensorScalarCell<T>& operator=(T const& value){ this->value_ = value; return *this; }
    __HOST_DEVICE__ IvyTensorScalarCell<T>& operator=(T&& value){ this->value_ = std_util::move(value); return *this; }

    __HOST_DEVICE__ void set_value(T const& value){ this->value_ = value; }
    __HOST_DEVICE__ value_t const& value() const{ return this->value_; }

    __HOST_DEVICE__ constexpr bool is_differentiable() __NOEXCEPT__ { return true; }

    friend struct IvyNodeSelfRelations<IvyTensorScalarCell<T>>;
  };
}
namespace IvyTypes{
  template<typename T> struct convert_to_floating_point<IvyMath::IvyTensorScalarCell<T>>{
    using type = IvyMath::IvyTensorScalarCell<convert_to_floating_point_t<T>>;
  };
}
namespace IvyMath{
  template<typename T> struct IvyNodeSelfRelations<IvyTensorScalarCell<T>>{
    static __HOST_DEVICE__ constexpr bool is_differentiable(IvyTensorScalarCell<T> const& /*x*/) __NOEXCEPT__ { return true; }
    static __HOST_DEVICE__ void conjugate(IvyTensorScalarCell<T>& /*x*/) __NOEXCEPT__ {}
    static constexpr bool is_conjugatable = false;
  };
  template<typename T> struct convert_to_real_type<IvyTensorScalarCell<T>>{ using type = IvyTensorScalarCell<T>; };
}

//============================== COMPLEX CELL ==============================
namespace IvyMath{
  template<typename T, ENABLE_IF_ARITHMETIC_IMPL(T)> class IvyTensorComplexCell final :
    public IvyBaseNode,
    public complex_domain_tag,
    public leaf_value_tag
  {
  public:
    using dtype_t = T;
    using value_t = IvyTensorComplexCell<T>;

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
    template<typename U> __HOST_DEVICE__ IvyTensorComplexCell(IvyTensorComplexCell<U> const& other) : re(__STATIC_CAST__(T, other.Re())), im(__STATIC_CAST__(T, other.Im())){}
    __HOST_DEVICE__ IvyTensorComplexCell(IvyTensorComplexCell<T> const& other) : re(other.re), im(other.im){}
    __HOST_DEVICE__ IvyTensorComplexCell(IvyTensorComplexCell<T>&& other) : re(std_util::move(other.re)), im(std_util::move(other.im)){}
    template<typename U> __HOST_DEVICE__ IvyTensorComplexCell(IvyTensorScalarCell<U> const& other) : re(__STATIC_CAST__(T, other.value())), im(0){}
    __HOST_DEVICE__ ~IvyTensorComplexCell(){}

    template<typename U> __HOST_DEVICE__ IvyTensorComplexCell& operator=(IvyTensorComplexCell<U> const& other){ re = __STATIC_CAST__(T, other.Re()); im = __STATIC_CAST__(T, other.Im()); return *this; }
    __HOST_DEVICE__ IvyTensorComplexCell& operator=(IvyTensorComplexCell<T> const& other){ re = other.re; im = other.im; return *this; }
    __HOST_DEVICE__ IvyTensorComplexCell& operator=(IvyTensorComplexCell<T>&& other){ re = std_util::move(other.re); im = std_util::move(other.im); return *this; }
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

    friend struct IvyNodeSelfRelations<IvyTensorComplexCell<T>>;
  };
}
namespace IvyTypes{
  template<typename T> struct convert_to_floating_point<IvyMath::IvyTensorComplexCell<T>>{
    using type = IvyMath::IvyTensorComplexCell<convert_to_floating_point_t<T>>;
  };
}
namespace IvyMath{
  template<typename T> struct IvyNodeSelfRelations<IvyTensorComplexCell<T>>{
    // Mirror IvyComplex: a complex leaf is not treated as a differentiation seed.
    static __HOST_DEVICE__ constexpr bool is_differentiable(IvyTensorComplexCell<T> const& /*x*/){ return false; }
    static __HOST_DEVICE__ void conjugate(IvyTensorComplexCell<T>& x){ x.im = -x.im; }
    static constexpr bool is_conjugatable = true;
  };
  template<typename T> struct convert_to_floating_point_if_complex<IvyTensorComplexCell<T>>{
    using type = IvyTensorComplexCell<convert_to_floating_point_t<T>>;
  };
  template<typename T> struct convert_to_real_type<IvyTensorComplexCell<T>>{
    using type = IvyTensorScalarCell<T>;
  };
  template<typename T> struct convert_to_complex_type<IvyTensorScalarCell<T>>{
    using type = IvyTensorComplexCell<T>;
  };
  template<typename T> struct convert_to_complex_type<IvyTensorComplexCell<T>>{
    using type = IvyTensorComplexCell<T>;
  };
}

//============================== PRINTOUT ==============================
namespace std_ivy{
  template<typename T> struct value_printout<IvyMath::IvyTensorScalarCell<T>>{
    static __HOST_DEVICE__ void print(IvyMath::IvyTensorScalarCell<T> const& var){
      __PRINT_INFO__("Cell(");
      print_value(var.value(), false);
      __PRINT_INFO__(")");
    }
  };
  template<typename T> struct value_printout<IvyMath::IvyTensorComplexCell<T>>{
    static __HOST_DEVICE__ void print(IvyMath::IvyTensorComplexCell<T> const& var){
      __PRINT_INFO__("CellComplex(");
      print_value(var.Re(), false);
      if (var.Im()<T(0)){ __PRINT_INFO__(" - "); print_value(-var.Im(), false); __PRINT_INFO__("i"); }
      else{ __PRINT_INFO__(" + "); print_value(var.Im(), false); __PRINT_INFO__("i"); }
      __PRINT_INFO__(")");
    }
  };
}


#endif
