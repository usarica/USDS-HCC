#ifndef IVYTENSORVARIABLECELL_H
#define IVYTENSORVARIABLECELL_H

/**
 * @file IvyTensorVariableCell.h
 * @brief Lightweight value-type differentiable scalar used as the element type
 *        of a contiguous (struct-of-arrays) differentiable tensor leaf.
 *
 * @c IvyTensorVariableCell<T> is functionally an @c IvyVariable<T> stripped of
 * the per-element bookkeeping that makes a tensor of variables expensive:
 *   - no @c IvyClientManager (96 bytes + a heap-allocated client vector per
 *     element); the owning tensor node carries the single client manager that
 *     tracks graph dependents, so per-element tracking is redundant.
 *   - no dead @c infinitesimal_ residue field.
 *
 * The result is exactly @c sizeof(T) per element (the tag bases are empty and
 * fold away under empty-base optimization), so @c IvyTensor<IvyTensorVariableCell<T>>
 * stores a single contiguous @c T buffer while still being a differentiable
 * variable tensor that flows through every @c IvyMath arithmetic operator with
 * full domain integration. The cell carries @c real_domain_tag and
 * @c variable_value_tag so the existing Fcnal/operability machinery treats it
 * exactly like a real-valued variable.
 */

#include "config/IvyCompilerConfig.h"
#include "autodiff/IvyBaseMathTypes.h"
#include "autodiff/base_types/IvyNodeRelations.h"


namespace IvyMath{
  template<typename T, ENABLE_IF_ARITHMETIC(T)> class IvyTensorVariableCell;
  template<typename T> struct IvyNodeSelfRelations<IvyTensorVariableCell<T>>;
}
namespace IvyMath{
  template<typename T, ENABLE_IF_ARITHMETIC_IMPL(T)> class IvyTensorVariableCell final :
    public IvyBaseNode,
    public real_domain_tag,
    public variable_value_tag
  {
  public:
    using dtype_t = T;
    using value_t = T;

  protected:
    value_t value_;

  public:
    __HOST_DEVICE__ IvyTensorVariableCell() : value_(0){}
    template<typename U, ENABLE_IF_ARITHMETIC(U)> __HOST_DEVICE__ IvyTensorVariableCell(U const& value) : value_(__STATIC_CAST__(T, value)){}
    __HOST_DEVICE__ IvyTensorVariableCell(T const& value) : value_(value){}
    __HOST_DEVICE__ IvyTensorVariableCell(T&& value) : value_(std_util::move(value)){}
    template<typename U> __HOST_DEVICE__ IvyTensorVariableCell(IvyTensorVariableCell<U> const& other) : value_(__STATIC_CAST__(T, other.value())){}
    __HOST_DEVICE__ IvyTensorVariableCell(IvyTensorVariableCell<T> const& other) : value_(other.value_){}
    __HOST_DEVICE__ IvyTensorVariableCell(IvyTensorVariableCell<T>&& other) : value_(std_util::move(other.value_)){}
    __HOST_DEVICE__ ~IvyTensorVariableCell(){}

    template<typename U> __HOST_DEVICE__ IvyTensorVariableCell<T>& operator=(IvyTensorVariableCell<U> const& other){ this->value_ = __STATIC_CAST__(T, other.value()); return *this; }
    __HOST_DEVICE__ IvyTensorVariableCell<T>& operator=(IvyTensorVariableCell<T> const& other){ this->value_ = other.value_; return *this; }
    __HOST_DEVICE__ IvyTensorVariableCell<T>& operator=(IvyTensorVariableCell<T>&& other){ this->value_ = std_util::move(other.value_); return *this; }
    template<typename U, ENABLE_IF_ARITHMETIC(U)> __HOST_DEVICE__ IvyTensorVariableCell<T>& operator=(U const& value){ this->value_ = __STATIC_CAST__(T, value); return *this; }
    __HOST_DEVICE__ IvyTensorVariableCell<T>& operator=(T const& value){ this->value_ = value; return *this; }
    __HOST_DEVICE__ IvyTensorVariableCell<T>& operator=(T&& value){ this->value_ = std_util::move(value); return *this; }

    __HOST_DEVICE__ void set_value(T const& value){ this->value_ = value; }
    __HOST_DEVICE__ value_t const& value() const{ return this->value_; }

    __HOST_DEVICE__ constexpr bool is_differentiable() __NOEXCEPT__ { return true; }

    friend struct IvyNodeSelfRelations<IvyTensorVariableCell<T>>;
  };
}
namespace IvyTypes{
  template<typename T> struct convert_to_floating_point<IvyMath::IvyTensorVariableCell<T>>{
    using type = IvyMath::IvyTensorVariableCell<convert_to_floating_point_t<T>>;
  };
}
namespace IvyMath{
  template<typename T> struct IvyNodeSelfRelations<IvyTensorVariableCell<T>>{
    static __HOST_DEVICE__ constexpr bool is_differentiable(IvyTensorVariableCell<T> const& /*x*/) __NOEXCEPT__ { return true; }
    static __HOST_DEVICE__ void conjugate(IvyTensorVariableCell<T>& /*x*/) __NOEXCEPT__ {}
    static constexpr bool is_conjugatable = false;
  };
  template<typename T> struct convert_to_real_type<IvyTensorVariableCell<T>>{ using type = IvyTensorVariableCell<T>; };
}
namespace std_ivy{
  template<typename T> struct value_printout<IvyMath::IvyTensorVariableCell<T>>{
    static __HOST_DEVICE__ void print(IvyMath::IvyTensorVariableCell<T> const& var){
      __PRINT_INFO__("Cell(");
      print_value(var.value(), false);
      __PRINT_INFO__(")");
    }
  };
}


#endif
