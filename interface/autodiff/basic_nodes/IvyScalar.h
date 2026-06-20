#ifndef IVYSCALAR_H
#define IVYSCALAR_H


#include "config/IvyCompilerConfig.h"
#include "stream/IvyStream.h"
#include "autodiff/IvyBaseMathTypes.h"
#include "autodiff/base_types/IvyNodeRelations.h"
#include "autodiff/base_types/IvyClientManager.h"


namespace IvyMath{
  template<typename T, ENABLE_IF_ARITHMETIC(T)> class IvyScalar;
  template<typename T> struct IvyNodeSelfRelations<IvyScalar<T>>;
}
namespace std_ivy{
  using namespace IvyMath;
  template<typename T> class transfer_memory_primitive<IvyScalar<T>> : public transfer_memory_primitive_with_internal_memory<IvyScalar<T>, IvyScalar<T>>{};
}
namespace IvyMath{
  template<typename T, ENABLE_IF_ARITHMETIC_IMPL(T)> class IvyScalar final :
    public IvyBaseNode,
    public IvyClientManager<IvyScalar<T>>,
    public real_domain_tag,
    public leaf_value_tag
  {
  public:
    using clientmgr_t = IvyClientManager<IvyScalar<T>>;
    using dtype_t = T;
    using value_t = T;

    friend class std_mem::kernel_generic_transfer_internal_memory<IvyScalar<T>, IvyScalar<T>>;

  protected:
    value_t value_;

  public:
    // Empty default constructor
    __HOST_DEVICE__ IvyScalar() : clientmgr_t(), value_(0){}
    template<typename U, ENABLE_IF_ARITHMETIC(U)> __HOST_DEVICE__ IvyScalar(U const& value) : clientmgr_t(), value_(__STATIC_CAST__(T, value)){}
    __HOST_DEVICE__ IvyScalar(T const& value) : clientmgr_t(), value_(value){}
    __HOST_DEVICE__ IvyScalar(T&& value) : clientmgr_t(), value_(std_util::move(value)){}
    template<typename U> __HOST_DEVICE__ IvyScalar(IvyScalar<U> const& other) : clientmgr_t(), value_(__STATIC_CAST__(T, other.value())){}
    __HOST_DEVICE__ IvyScalar(IvyScalar<T> const& other) : clientmgr_t(), value_(other.value_){}
    __HOST_DEVICE__ IvyScalar(IvyScalar<T>&& other) : clientmgr_t(), value_(std_util::move(other.value_)){}
    __HOST_DEVICE__ ~IvyScalar(){}

    // Assignment operators
    template<typename U> __HOST_DEVICE__ IvyScalar<T>& operator=(IvyScalar<U> const& other){
      this->value_ = __STATIC_CAST__(T, other.value());
      this->update_clients_modified();
      return *this;
    }
    __HOST_DEVICE__ IvyScalar<T>& operator=(IvyScalar<T> const& other){
      this->value_ = other.value_;
      this->update_clients_modified();
      return *this;
    }
    __HOST_DEVICE__ IvyScalar<T>& operator=(IvyScalar<T>&& other){
      this->value_ = std_util::move(other.value_);
      this->update_clients_modified();
      return *this;
    }

    template<typename U, ENABLE_IF_ARITHMETIC(U)> __HOST_DEVICE__ IvyScalar<T>& operator=(U const& value){
      this->value_ = __STATIC_CAST__(T, value);
      this->update_clients_modified();
      return *this;
    }
    __HOST_DEVICE__ IvyScalar<T>& operator=(T const& value){
      this->value_ = value;
      this->update_clients_modified();
      return *this;
    }
    __HOST_DEVICE__ IvyScalar<T>& operator=(T&& value){
      this->value_ = std_util::move(value);
      this->update_clients_modified();
      return *this;
    }

    // Set functions
    __HOST_DEVICE__ void set_value(T const& value){ this->value_ = value; this->update_clients_modified(); }

    // Get functions
    //__HOST_DEVICE__ value_t& value(){ return this->value_; }
    __HOST_DEVICE__ value_t const& value() const{ return this->value_; }

    // IvyScalars are differentiable objects.
    __HOST_DEVICE__ constexpr bool is_differentiable() __NOEXCEPT__ { return true; }

    friend struct IvyNodeSelfRelations<IvyScalar<T>>;
  };
}
namespace IvyTypes{
  template<typename T> struct convert_to_floating_point<IvyMath::IvyScalar<T>>{
    using type = IvyMath::IvyScalar<convert_to_floating_point_t<T>>;
  };
}
namespace IvyMath{
  template<typename T> struct IvyNodeSelfRelations<IvyScalar<T>>{
    static __HOST_DEVICE__ constexpr bool is_differentiable(IvyScalar<T> const& x) __NOEXCEPT__ { return true; }
    static __HOST_DEVICE__ void conjugate(IvyScalar<T>& x) __NOEXCEPT__ {}
    static constexpr bool is_conjugatable = false;
  };

  template<typename T> struct minimal_domain_type<T, real_domain_tag, leaf_value_tag>{ using type = IvyScalar<std_ttraits::remove_cv_t<T>>; };

  template<typename T> using IvyScalarPtr_t = IvyThreadSafePtr_t< IvyScalar<T> >;

  template<typename T, typename... Args> __HOST_DEVICE__ IvyScalarPtr_t<T> Scalar(Args&&... args){ return make_IvyThreadSafePtr< IvyScalar<T> >(args...); }
}
namespace std_ivy{
  template<typename T> struct value_printout<IvyMath::IvyScalar<T>>{
    static __HOST_DEVICE__ void print(IvyMath::IvyScalar<T> const& var){
      __PRINT_INFO__("Scalar(");
      print_value(var.value(), false);
      __PRINT_INFO__(")");
    }
  };
}


#endif
