#ifndef IVYQUATERNION_H
#define IVYQUATERNION_H


#include "config/IvyCompilerConfig.h"
#include "std_ivy/IvyCmath.h"
#include "stream/IvyStream.h"
#include "autodiff/IvyBaseMathTypes.h"
#include "autodiff/base_types/IvyNodeRelations.h"
#include "autodiff/base_types/IvyClientManager.h"
#include "autodiff/basic_nodes/IvyScalar.h"
#include "autodiff/basic_nodes/IvyComplex.h"


namespace IvyMath{
  /*
  IvyQuaternion<T>: a leaf in the H (Hamilton quaternion) division algebra.
  Canonical storage is the four real Cartesian components (w, x, y, z) with
  w the scalar part and (x, y, z) the vector part. The Hamilton product is
  NON-commutative (i*j = k, j*i = -k), which is why the division-algebra
  traits in IvyTypeTags mark quaternion_domain_tag as is_commutative=false.
  Differentiation through quaternion-valued expressions is handled by the
  order-aware *Fcnal gradients in IvyMathBaseArithmetic.
  */
  template<typename T, ENABLE_IF_ARITHMETIC(T)> class IvyQuaternion;
  template<typename T> struct IvyNodeSelfRelations<IvyQuaternion<T>>;
}
namespace std_ivy{
  using namespace IvyMath;
  template<typename T> class transfer_memory_primitive<IvyQuaternion<T>> : public transfer_memory_primitive_with_internal_memory<IvyQuaternion<T>, IvyQuaternion<T>>{};
}
namespace IvyMath{
  template<typename T, ENABLE_IF_ARITHMETIC_IMPL(T)> class IvyQuaternion final :
    public IvyBaseNode,
    public IvyClientManager<IvyQuaternion<T>>,
    public quaternion_domain_tag,
    public leaf_value_tag
  {
  public:
    using clientmgr_t = IvyClientManager<IvyQuaternion<T>>;
    using dtype_t = T;
    using value_t = IvyQuaternion<T>;

    friend class std_mem::kernel_generic_transfer_internal_memory<IvyQuaternion<T>, IvyQuaternion<T>>;

  protected:
    dtype_t w_;
    dtype_t x_;
    dtype_t y_;
    dtype_t z_;

  public:
    // Constructors
    __HOST_DEVICE__ IvyQuaternion() : clientmgr_t(), w_(0), x_(0), y_(0), z_(0){}
    template<typename U, ENABLE_IF_ARITHMETIC(U)> __HOST_DEVICE__ IvyQuaternion(U const& w) : clientmgr_t(), w_(__STATIC_CAST__(T, w)), x_(0), y_(0), z_(0){}
    __HOST_DEVICE__ IvyQuaternion(T const& w) : clientmgr_t(), w_(w), x_(0), y_(0), z_(0){}
    __HOST_DEVICE__ IvyQuaternion(T const& w, T const& x, T const& y, T const& z) : clientmgr_t(), w_(w), x_(x), y_(y), z_(z){}
    template<typename U> __HOST_DEVICE__ IvyQuaternion(IvyQuaternion<U> const& other) :
      clientmgr_t(), w_(__STATIC_CAST__(T, other.W())), x_(__STATIC_CAST__(T, other.X())), y_(__STATIC_CAST__(T, other.Y())), z_(__STATIC_CAST__(T, other.Z())){}
    __HOST_DEVICE__ IvyQuaternion(IvyQuaternion<T> const& other) : clientmgr_t(), w_(other.w_), x_(other.x_), y_(other.y_), z_(other.z_){}
    __HOST_DEVICE__ IvyQuaternion(IvyQuaternion<T>&& other) :
      clientmgr_t(), w_(std_util::move(other.w_)), x_(std_util::move(other.x_)), y_(std_util::move(other.y_)), z_(std_util::move(other.z_)){}
    template<typename U> __HOST_DEVICE__ IvyQuaternion(IvyScalar<U> const& other) : clientmgr_t(), w_(__STATIC_CAST__(T, other.value())), x_(0), y_(0), z_(0){}
    __HOST_DEVICE__ IvyQuaternion(IvyScalar<T> const& other) : clientmgr_t(), w_(other.value()), x_(0), y_(0), z_(0){}
    template<typename U> __HOST_DEVICE__ IvyQuaternion(IvyComplex<U> const& other) : clientmgr_t(), w_(__STATIC_CAST__(T, other.Re())), x_(__STATIC_CAST__(T, other.Im())), y_(0), z_(0){}
    __HOST_DEVICE__ IvyQuaternion(IvyComplex<T> const& other) : clientmgr_t(), w_(other.Re()), x_(other.Im()), y_(0), z_(0){}

    // Empty destructor
    __HOST_DEVICE__ ~IvyQuaternion(){}

    // Assignment operators
    template<typename U> __HOST_DEVICE__ IvyQuaternion& operator=(IvyQuaternion<U> const& other){
      w_ = __STATIC_CAST__(T, other.W()); x_ = __STATIC_CAST__(T, other.X());
      y_ = __STATIC_CAST__(T, other.Y()); z_ = __STATIC_CAST__(T, other.Z());
      this->update_clients_modified();
      return *this;
    }
    __HOST_DEVICE__ IvyQuaternion& operator=(IvyQuaternion<T> const& other){
      w_ = other.w_; x_ = other.x_; y_ = other.y_; z_ = other.z_;
      this->update_clients_modified();
      return *this;
    }
    __HOST_DEVICE__ IvyQuaternion& operator=(IvyQuaternion<T>&& other){
      w_ = std_util::move(other.w_); x_ = std_util::move(other.x_);
      y_ = std_util::move(other.y_); z_ = std_util::move(other.z_);
      this->update_clients_modified();
      return *this;
    }
    template<typename U, ENABLE_IF_ARITHMETIC(U)> __HOST_DEVICE__ IvyQuaternion& operator=(U const& w){
      w_ = __STATIC_CAST__(T, w); x_ = T(0); y_ = T(0); z_ = T(0);
      this->update_clients_modified();
      return *this;
    }
    __HOST_DEVICE__ IvyQuaternion& operator=(T const& w){
      w_ = w; x_ = T(0); y_ = T(0); z_ = T(0);
      this->update_clients_modified();
      return *this;
    }

    // Accessors
    __HOST_DEVICE__ T const& W() const{ return w_; }
    __HOST_DEVICE__ T const& X() const{ return x_; }
    __HOST_DEVICE__ T const& Y() const{ return y_; }
    __HOST_DEVICE__ T const& Z() const{ return z_; }

    __HOST_DEVICE__ T norm2() const{ return w_*w_ + x_*x_ + y_*y_ + z_*z_; }
    __HOST_DEVICE__ T norm() const{ return std_math::sqrt(this->norm2()); }

    // value() lets a quaternion leaf operate like any other Ivy value node.
    __HOST_DEVICE__ value_t const& value() const{ return *this; }

    // Set functions
    __HOST_DEVICE__ void set_components(T const& w, T const& x, T const& y, T const& z){
      w_ = w; x_ = x; y_ = y; z_ = z; this->update_clients_modified();
    }
    __HOST_DEVICE__ void set_scalar(T const& w){ w_ = w; this->update_clients_modified(); }
    __HOST_DEVICE__ void set_vector(T const& x, T const& y, T const& z){ x_ = x; y_ = y; z_ = z; this->update_clients_modified(); }

    friend struct IvyNodeSelfRelations<IvyQuaternion<T>>;
  };
}
namespace IvyTypes{
  template<typename T> struct convert_to_floating_point<IvyMath::IvyQuaternion<T>>{
    using type = IvyMath::IvyQuaternion<convert_to_floating_point_t<T>>;
  };
}
namespace IvyMath{
  template<typename T> struct IvyNodeSelfRelations<IvyQuaternion<T>>{
    static __HOST_DEVICE__ constexpr bool is_differentiable(IvyQuaternion<T> const& x){ return false; }
    // Quaternion conjugation negates the vector part: conj(w,x,y,z) = (w,-x,-y,-z).
    static __HOST_DEVICE__ void conjugate(IvyQuaternion<T>& x){ x.x_ = -x.x_; x.y_ = -x.y_; x.z_ = -x.z_; }
    static constexpr bool is_conjugatable = true;
  };

  template<typename T> struct convert_to_floating_point_if_complex<IvyQuaternion<T>>{
    using type = IvyQuaternion<convert_to_floating_point_t<T>>;
  };
  template<typename T> struct convert_to_real_type<IvyQuaternion<T>>{
    using type = IvyScalar<T>;
  };

  template<typename T> struct minimal_domain_type<T, quaternion_domain_tag, leaf_value_tag>{ using type = IvyQuaternion<std_ttraits::remove_cv_t<T>>; };

  template<typename T> using IvyQuaternionPtr_t = IvyThreadSafePtr_t< IvyQuaternion<T> >;

  template<typename T, typename... Args> __HOST_DEVICE__ IvyQuaternionPtr_t<T> Quaternion(Args&&... args){ return make_IvyThreadSafePtr< IvyQuaternion<T> >(args...); }
}
namespace std_ivy{
  template<typename T> struct value_printout<IvyMath::IvyQuaternion<T>>{
    static __HOST_DEVICE__ void print(IvyMath::IvyQuaternion<T> const& var){
      __PRINT_INFO__("Quaternion(");
      print_value(var.W(), false); __PRINT_INFO__(", ");
      print_value(var.X(), false); __PRINT_INFO__("i, ");
      print_value(var.Y(), false); __PRINT_INFO__("j, ");
      print_value(var.Z(), false); __PRINT_INFO__("k)");
    }
  };
}


#endif
