#ifndef IVYCOMPLEX_H
#define IVYCOMPLEX_H


#include "config/IvyCompilerConfig.h"
#include "std_ivy/IvyCmath.h"
#include "stream/IvyStream.h"
#include "autodiff/IvyBaseMathTypes.h"
#include "autodiff/base_types/IvyNodeRelations.h"
#include "autodiff/base_types/IvyClientManager.h"
#include "autodiff/basic_nodes/IvyScalar.h"


namespace IvyMath{
  template<typename T, ENABLE_IF_ARITHMETIC(T)> class IvyComplex;
  template<typename T> struct IvyNodeSelfRelations<IvyComplex<T>>;
}
namespace std_ivy{
  using namespace IvyMath;
  template<typename T> class transfer_memory_primitive<IvyComplex<T>> : public transfer_memory_primitive_with_internal_memory<IvyComplex<T>, IvyComplex<T>>{};
}
namespace IvyMath{
  template<typename T, ENABLE_IF_ARITHMETIC_IMPL(T)> class IvyComplex final :
    public IvyBaseNode,
    public IvyClientManager<IvyComplex<T>>,
    public complex_domain_tag,
    public leaf_value_tag
  {
  public:
    using clientmgr_t = IvyClientManager<IvyComplex<T>>;
    using dtype_t = T;
    using value_t = IvyComplex<T>;

    friend class std_mem::kernel_generic_transfer_internal_memory<IvyComplex<T>, IvyComplex<T>>;

  protected:
    dtype_t re;
    dtype_t im;

  public:
    // Constructors
    __HOST_DEVICE__ IvyComplex() : clientmgr_t(), re(0), im(0){}
    template<typename U, ENABLE_IF_ARITHMETIC(U)> __HOST_DEVICE__ IvyComplex(U const& re_) : clientmgr_t(), re(__STATIC_CAST__(T, re_)), im(0){}
    __HOST_DEVICE__ IvyComplex(T const& re_) : clientmgr_t(), re(re_), im(0){}
    __HOST_DEVICE__ IvyComplex(T const& re_, T const& im_) : clientmgr_t(), re(re_), im(im_){}
    __HOST_DEVICE__ IvyComplex(T&& re_) : clientmgr_t(), re(std_util::move(re_)), im(0){}
    __HOST_DEVICE__ IvyComplex(T&& re_, T&& im_) : clientmgr_t(), re(std_util::move(re_)), im(std_util::move(im_)){}
    template<typename U> __HOST_DEVICE__ IvyComplex(IvyComplex<U> const& other) : clientmgr_t(), re(__STATIC_CAST__(T, other.re)), im(__STATIC_CAST__(T, other.im)){}
    __HOST_DEVICE__ IvyComplex(IvyComplex<T> const& other) : clientmgr_t(), re(other.re), im(other.im){}
    __HOST_DEVICE__ IvyComplex(IvyComplex<T>&& other) : clientmgr_t(), re(std_util::move(other.re)), im(std_util::move(other.im)){}
    template<typename U> __HOST_DEVICE__ IvyComplex(IvyScalar<U> const& other) : clientmgr_t(), re(__STATIC_CAST__(T, other.value())), im(0){}
    __HOST_DEVICE__ IvyComplex(IvyScalar<T> const& other) : clientmgr_t(), re(other.value()), im(0){}
    __HOST_DEVICE__ IvyComplex(IvyScalar<T>&& other) : clientmgr_t(), re(std_util::move(other.value())), im(0){}

    // Empty destructor
    __HOST_DEVICE__ ~IvyComplex(){}

    // Assignment operator
    template<typename U> __HOST_DEVICE__ IvyComplex& operator=(IvyComplex<U> const& other){
      re = __STATIC_CAST__(T, other.Re());
      im = __STATIC_CAST__(T, other.Im());
      this->update_clients_modified();
      return *this;
    }
    __HOST_DEVICE__ IvyComplex& operator=(IvyComplex<T> const& other){
      re = other.re;
      im = other.im;
      this->update_clients_modified();
      return *this;
    }
    __HOST_DEVICE__ IvyComplex& operator=(IvyComplex<T>&& other){
      re = std_util::move(other.re);
      im = std_util::move(other.im);
      this->update_clients_modified();
      return *this;
    }
    template<typename U, ENABLE_IF_ARITHMETIC(U)> __HOST_DEVICE__ IvyComplex& operator=(U const& re_){
      re = __STATIC_CAST__(T, re_);
      im = T(0);
      this->update_clients_modified();
      return *this;
    }
    __HOST_DEVICE__ IvyComplex& operator=(T const& re_){
      re = re_;
      im = T(0);
      this->update_clients_modified();
      return *this;
    }
    __HOST_DEVICE__ IvyComplex& operator=(T&& re_){
      re = std_util::move(re_);
      im = T(0);
      this->update_clients_modified();
      return *this;
    }
    template<typename U> __HOST_DEVICE__ IvyComplex& operator=(IvyScalar<U> const& other){
      re = __STATIC_CAST__(T, other.value());
      im = T(0);
      this->update_clients_modified();
      return *this;
    }
    __HOST_DEVICE__ IvyComplex& operator=(IvyScalar<T> const& other){
      re = other.value();
      im = T(0);
      this->update_clients_modified();
      return *this;
    }
    __HOST_DEVICE__ IvyComplex& operator=(IvyScalar<T>&& other){
      re = std_util::move(other.value());
      im = T(0);
      this->update_clients_modified();
      return *this;
    }

    // Accessors
    //__HOST_DEVICE__ T& Re(){ return re; }
    __HOST_DEVICE__ T const& Re() const{ return re; }

    //__HOST_DEVICE__ T& Im(){ return im; }
    __HOST_DEVICE__ T const& Im() const{ return im; }

    __HOST_DEVICE__ T norm() const{ return std_math::sqrt(re*re + im*im); }
    __HOST_DEVICE__ T phase() const{ return std_math::atan2(im, re); }

    // Ensure that a complex variable can operate in just the same way as a variable or a constant.
    //__HOST_DEVICE__ value_t& value(){ return *this; }
    __HOST_DEVICE__ value_t const& value() const{ return *this; }

    // Set functions
    __HOST_DEVICE__ void set_real(T const& x){ re=x; this->update_clients_modified(); }
    __HOST_DEVICE__ void set_imaginary(T const& x){ im=x; this->update_clients_modified(); }

    __HOST_DEVICE__ void set_absval_phase(T const& v, T const& phi){
      re = v*std_math::cos(phi);
      im = v*std_math::sin(phi);
      this->update_clients_modified();
    }

    friend struct IvyNodeSelfRelations<IvyComplex<T>>;
  };
}
namespace IvyTypes{
  template<typename T> struct convert_to_floating_point<IvyMath::IvyComplex<T>>{
    using type = IvyMath::IvyComplex<convert_to_floating_point_t<T>>;
  };
}
namespace IvyMath{
  template<typename T> struct IvyNodeSelfRelations<IvyComplex<T>>{
    static __HOST_DEVICE__ constexpr bool is_differentiable(IvyComplex<T> const& x){ return false; }
    static __HOST_DEVICE__ void conjugate(IvyComplex<T>& x){ x.im = -x.im; }
    static constexpr bool is_conjugatable = true;
  };

  template<typename T> struct convert_to_floating_point_if_complex<IvyComplex<T>>{
    using type = IvyComplex<convert_to_floating_point_t<T>>;
  };
  template<typename T> struct convert_to_real_type<IvyComplex<T>>{
    using type = IvyScalar<T>;
  };

  /*
  convert_to_complex_t:
  The type of convert_to_complex_t is supposed to be the corresponding complex value of a class.
  */
  template<typename T> struct convert_to_complex_type{
    using type = IvyComplex<T>;
  };
  template<typename T> using convert_to_complex_t = typename convert_to_complex_type<T>::type;
  template<typename T> struct convert_to_complex_type<IvyThreadSafePtr_t<T>>{
    using type = IvyThreadSafePtr_t<convert_to_complex_t<T>>;
  };
  template<typename T> struct convert_to_complex_type<T const>{
    using type = convert_to_complex_t<T> const;
  };
  template<typename T> struct convert_to_complex_type<IvyScalar<T>>{
    using type = IvyComplex<T>;
  };
  template<typename T> struct convert_to_complex_type<IvyComplex<T>>{
    using type = IvyComplex<T>;
  };

  template<typename T> struct minimal_domain_type<T, complex_domain_tag, leaf_value_tag>{ using type = IvyComplex<std_ttraits::remove_cv_t<T>>; };

  template<typename T> using IvyComplexPtr_t = IvyThreadSafePtr_t< IvyComplex<T> >;

  template<typename T, typename... Args> __HOST_DEVICE__ IvyComplexPtr_t<T> Complex(Args&&... args){ return make_IvyThreadSafePtr< IvyComplex<T> >(args...); }
}
namespace std_ivy{
  template<typename T> struct value_printout<IvyMath::IvyComplex<T>>{
    static __HOST_DEVICE__ void print(IvyMath::IvyComplex<T> const& var){
      __PRINT_INFO__("Complex(");
      print_value(var.Re(), false);
      if (var.Im()<T(0)){
        __PRINT_INFO__(" - ");
        print_value(-var.Im(), false);
        __PRINT_INFO__("i");
      }
      else{
        __PRINT_INFO__(" + ");
        print_value(var.Im(), false);
        __PRINT_INFO__("i");
      }
      __PRINT_INFO__(")");
    }
  };
}


#endif
