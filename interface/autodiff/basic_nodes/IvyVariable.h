#ifndef IVYVARIABLE_H
#define IVYVARIABLE_H


#include "config/IvyCompilerConfig.h"
#include "stream/IvyStream.h"
#include "autodiff/IvyBaseMathTypes.h"
#include "autodiff/base_types/IvyNodeRelations.h"
#include "autodiff/basic_nodes/IvyConstant.h"


namespace IvyMath{
  template<typename T, ENABLE_IF_ARITHMETIC(T)> class IvyVariable;
  template<typename T> struct IvyNodeSelfRelations<IvyVariable<T>>;
}
namespace std_ivy{
  using namespace IvyMath;
  template<typename T> class transfer_memory_primitive<IvyVariable<T>> : public transfer_memory_primitive_with_internal_memory<IvyVariable<T>, IvyVariable<T>>{};
}
namespace IvyMath{
  template<typename T, ENABLE_IF_ARITHMETIC_IMPL(T)> class IvyVariable final :
    public IvyBaseNode,
    public IvyClientManager<IvyVariable<T>>,
    public real_domain_tag,
    public variable_value_tag
  {
  public:
    using clientmgr_t = IvyClientManager<IvyVariable<T>>;
    using dtype_t = T;
    using value_t = T;

    friend class std_mem::kernel_generic_transfer_internal_memory<IvyVariable<T>, IvyVariable<T>>;

  protected:
    value_t value_;

  public:
    // Empty default constructor
    __HOST_DEVICE__ IvyVariable() : clientmgr_t(), value_(0){}
    template<typename U, ENABLE_IF_ARITHMETIC(U)> __HOST_DEVICE__ IvyVariable(U const& value) : clientmgr_t(), value_(__STATIC_CAST__(T, value)){}
    __HOST_DEVICE__ IvyVariable(T const& value) : clientmgr_t(), value_(value){}
    __HOST_DEVICE__ IvyVariable(T&& value) : clientmgr_t(), value_(std_util::move(value)){}
    template<typename U> __HOST_DEVICE__ IvyVariable(IvyVariable<U> const& other) : clientmgr_t(), value_(__STATIC_CAST__(T, other.value())){}
    __HOST_DEVICE__ IvyVariable(IvyVariable<T> const& other) : clientmgr_t(), value_(other.value_){}
    __HOST_DEVICE__ IvyVariable(IvyVariable<T>&& other) : clientmgr_t(), value_(std_util::move(other.value_)){}
    template<typename U> __HOST_DEVICE__ IvyVariable(IvyConstant<U> const& value) : clientmgr_t(), value_(__STATIC_CAST__(T, value.value())){}
    __HOST_DEVICE__ IvyVariable(IvyConstant<T> const& value) : clientmgr_t(), value_(value.value()){}
    __HOST_DEVICE__ IvyVariable(IvyConstant<T>&& value) : clientmgr_t(), value_(value.value()){}
    __HOST_DEVICE__ ~IvyVariable(){}

    // Assignment operators
    template<typename U> __HOST_DEVICE__ IvyVariable<T>& operator=(IvyVariable<U> const& other){
      this->value_ = __STATIC_CAST__(T, other.value());
      this->update_clients_modified();
      return *this;
    }
    __HOST_DEVICE__ IvyVariable<T>& operator=(IvyVariable<T> const& other){
      this->value_ = other.value_;
      this->update_clients_modified();
      return *this;
    }
    __HOST_DEVICE__ IvyVariable<T>& operator=(IvyVariable<T>&& other){
      this->value_ = std_util::move(other.value_);
      this->update_clients_modified();
      return *this;
    }

    template<typename U, ENABLE_IF_ARITHMETIC(U)> __HOST_DEVICE__ IvyVariable<T>& operator=(U const& value){
      this->value_ = __STATIC_CAST__(T, value);
      this->update_clients_modified();
      return *this;
    }
    __HOST_DEVICE__ IvyVariable<T>& operator=(T const& value){
      this->value_ = value;
      this->update_clients_modified();
      return *this;
    }
    __HOST_DEVICE__ IvyVariable<T>& operator=(T&& value){
      this->value_ = std_util::move(value);
      this->update_clients_modified();
      return *this;
    }

    template<typename U> __HOST_DEVICE__ IvyVariable<T>& operator=(IvyConstant<U> const& value){
      this->value_ = __STATIC_CAST__(T, value.value());
      this->update_clients_modified();
      return *this;
    }
    __HOST_DEVICE__ IvyVariable<T>& operator=(IvyConstant<T> const& value){
      this->value_ = value.value();
      this->update_clients_modified();
      return *this;
    }
    __HOST_DEVICE__ IvyVariable<T>& operator=(IvyConstant<T>&& value){
      this->value_ = value.value();
      this->update_clients_modified();
      return *this;
    }

    // Set functions
    __HOST_DEVICE__ void set_value(T const& value){ this->value_ = value; this->update_clients_modified(); }

    // Get functions
    //__HOST_DEVICE__ value_t& value(){ return this->value_; }
    __HOST_DEVICE__ value_t const& value() const{ return this->value_; }

    // IvyVariables are differentiable objects.
    __HOST_DEVICE__ constexpr bool is_differentiable() __NOEXCEPT__ { return true; }

    friend struct IvyNodeSelfRelations<IvyVariable<T>>;
  };
}
namespace IvyTypes{
  template<typename T> struct convert_to_floating_point<IvyMath::IvyVariable<T>>{
    using type = IvyMath::IvyVariable<convert_to_floating_point_t<T>>;
  };
}
namespace IvyMath{
  template<typename T> struct IvyNodeSelfRelations<IvyVariable<T>>{
    static __HOST_DEVICE__ constexpr bool is_differentiable(IvyVariable<T> const& x) __NOEXCEPT__ { return true; }
    static __HOST_DEVICE__ void conjugate(IvyVariable<T>& x) __NOEXCEPT__ {}
    static constexpr bool is_conjugatable = false;
  };

  template<typename T> struct minimal_domain_type<T, real_domain_tag, variable_value_tag>{ using type = IvyVariable<std_ttraits::remove_cv_t<T>>; };

  template<typename T> using IvyVariablePtr_t = IvyThreadSafePtr_t< IvyVariable<T> >;

  template<typename T, typename... Args> __HOST_DEVICE__ IvyVariablePtr_t<T> Variable(Args&&... args){ return make_IvyThreadSafePtr< IvyVariable<T> >(args...); }
}
namespace std_ivy{
  template<typename T> struct value_printout<IvyMath::IvyVariable<T>>{
    static __HOST_DEVICE__ void print(IvyMath::IvyVariable<T> const& var){
      __PRINT_INFO__("Variable(");
      print_value(var.value(), false);
      __PRINT_INFO__(")");
    }
  };
}


#endif
