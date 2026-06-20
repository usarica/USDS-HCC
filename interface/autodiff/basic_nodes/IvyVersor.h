#ifndef IVYVERSOR_H
#define IVYVERSOR_H


#include "config/IvyCompilerConfig.h"
#include "std_ivy/IvyCmath.h"
#include "stream/IvyStream.h"
#include "autodiff/IvyBaseMathTypes.h"
#include "autodiff/base_types/IvyNodeRelations.h"
#include "autodiff/base_types/IvyClientManager.h"
#include "autodiff/basic_nodes/IvyScalar.h"
#include "autodiff/basic_nodes/IvyQuaternion.h"


namespace IvyMath{
  /*
  Representation tag for the versor's native chart: the exponential map of the Lie
  algebra so(3) (the tangent space at the identity), which is R^3. A versor
  q = exp(omega/2) with omega the rotation vector (axis * angle). This chart is the
  natural differentiation basis for rotations: it is free of gimbal lock and of the
  unit-norm constraint, and the tangent space is a plain R^3.
  */
  struct versor_lie_chart{ using tag = real_domain_tag; };

  /*
  IvyVersor<T>: a constrained UNIT quaternion (a point on S^3) representing a rigid
  rotation in SO(3) (double cover S^3 -> SO(3)). Unlike the general IvyQuaternion,
  the versor stays normalized and its differentiation basis is the so(3) tangent
  (R^3), exposed through the exp/log maps. Group operations (compose, inverse) and
  vector rotation are provided with analytic tangent-space Jacobians.

  Storage is the canonical unit components (w, x, y, z). The native chart
  (versor_lie_chart) is the exponential map from the rotation vector omega in R^3.
  */
  template<typename T, ENABLE_IF_ARITHMETIC(T)> class IvyVersor;
  template<typename T> struct IvyNodeSelfRelations<IvyVersor<T>>;
}
namespace std_ivy{
  using namespace IvyMath;
  template<typename T> class transfer_memory_primitive<IvyVersor<T>> : public transfer_memory_primitive_with_internal_memory<IvyVersor<T>, IvyVersor<T>>{};
}
namespace IvyMath{
  template<typename T, ENABLE_IF_ARITHMETIC_IMPL(T)> class IvyVersor final :
    public IvyBaseNode,
    public IvyClientManager<IvyVersor<T>>,
    public leaf_value_tag
  {
  public:
    using clientmgr_t = IvyClientManager<IvyVersor<T>>;
    using dtype_t = T;
    using value_t = IvyVersor<T>;
    using representation_t = versor_lie_chart;

    friend class std_mem::kernel_generic_transfer_internal_memory<IvyVersor<T>, IvyVersor<T>>;

  protected:
    T w_, x_, y_, z_;

    __HOST_DEVICE__ void normalize(){
      T n = std_math::sqrt(w_*w_ + x_*x_ + y_*y_ + z_*z_);
      if (n > T(0)){ w_/=n; x_/=n; y_/=n; z_/=n; }
      else { w_=T(1); x_=T(0); y_=T(0); z_=T(0); }
    }

  public:
    // Constructors. Default = identity rotation.
    __HOST_DEVICE__ IvyVersor() : clientmgr_t(), w_(1), x_(0), y_(0), z_(0){}
    __HOST_DEVICE__ IvyVersor(T const& w, T const& x, T const& y, T const& z) : clientmgr_t(), w_(w), x_(x), y_(y), z_(z){ normalize(); }
    __HOST_DEVICE__ IvyVersor(IvyVersor<T> const& o) : clientmgr_t(), w_(o.w_), x_(o.x_), y_(o.y_), z_(o.z_){}
    __HOST_DEVICE__ IvyVersor(IvyVersor<T>&& o) : clientmgr_t(), w_(std_util::move(o.w_)), x_(std_util::move(o.x_)), y_(std_util::move(o.y_)), z_(std_util::move(o.z_)){}
    __HOST_DEVICE__ ~IvyVersor(){}

    __HOST_DEVICE__ IvyVersor& operator=(IvyVersor<T> const& o){ w_=o.w_; x_=o.x_; y_=o.y_; z_=o.z_; this->update_clients_modified(); return *this; }

    // Factories
    // Build from an axis (need not be unit) and an angle (radians).
    static __HOST_DEVICE__ IvyVersor from_axis_angle(T ax, T ay, T az, T angle){
      T an = std_math::sqrt(ax*ax + ay*ay + az*az);
      IvyVersor q;
      if (an > T(0)){ ax/=an; ay/=an; az/=an; }
      else { q.w_=T(1); q.x_=q.y_=q.z_=T(0); return q; }
      T h = angle/T(2);
      T s = std_math::sin(h);
      q.w_ = std_math::cos(h); q.x_ = ax*s; q.y_ = ay*s; q.z_ = az*s;
      return q;
    }
    // Exponential map from the rotation vector omega = axis*angle (so(3) -> S^3).
    static __HOST_DEVICE__ IvyVersor exp_map(T wx, T wy, T wz){
      T th = std_math::sqrt(wx*wx + wy*wy + wz*wz);
      IvyVersor q;
      T h = th/T(2);
      T s; // sin(h)/th  (so that vector part = omega * s)
      if (th > T(1e-8)) s = std_math::sin(h)/th;
      else s = T(0.5) - th*th/T(48); // (1/2)(1 - (th/2)^2/6 + ...)
      q.w_ = std_math::cos(h); q.x_ = wx*s; q.y_ = wy*s; q.z_ = wz*s;
      return q;
    }

    // Accessors
    __HOST_DEVICE__ T const& W() const{ return w_; }
    __HOST_DEVICE__ T const& X() const{ return x_; }
    __HOST_DEVICE__ T const& Y() const{ return y_; }
    __HOST_DEVICE__ T const& Z() const{ return z_; }
    __HOST_DEVICE__ T norm() const{ return std_math::sqrt(w_*w_ + x_*x_ + y_*y_ + z_*z_); }
    __HOST_DEVICE__ value_t const& value() const{ return *this; }

    __HOST_DEVICE__ T angle() const{
      T vn = std_math::sqrt(x_*x_ + y_*y_ + z_*z_);
      return T(2)*std_math::atan2(vn, w_);
    }
    // Logarithm map: rotation vector omega = axis*angle (S^3 -> so(3)).
    __HOST_DEVICE__ void log_map(T& wx, T& wy, T& wz) const{
      T vn = std_math::sqrt(x_*x_ + y_*y_ + z_*z_);
      if (vn > T(1e-12)){
        T th = T(2)*std_math::atan2(vn, w_); // full angle
        T f = th/vn;
        wx = x_*f; wy = y_*f; wz = z_*f;
      }
      else { wx = T(2)*x_; wy = T(2)*y_; wz = T(2)*z_; } // small-angle: omega ~ 2*vec
    }

    // Group operations
    __HOST_DEVICE__ IvyVersor inverse() const{ IvyVersor q; q.w_=w_; q.x_=-x_; q.y_=-y_; q.z_=-z_; return q; }
    // Hamilton product (this then other applied as composition this*other).
    __HOST_DEVICE__ IvyVersor compose(IvyVersor<T> const& b) const{
      IvyVersor q;
      q.w_ = w_*b.w_ - x_*b.x_ - y_*b.y_ - z_*b.z_;
      q.x_ = w_*b.x_ + x_*b.w_ + y_*b.z_ - z_*b.y_;
      q.y_ = w_*b.y_ - x_*b.z_ + y_*b.w_ + z_*b.x_;
      q.z_ = w_*b.z_ + x_*b.y_ - y_*b.x_ + z_*b.w_;
      q.normalize();
      return q;
    }

    // Rotate a 3-vector v -> R v (active rotation), R = the SO(3) matrix of this versor.
    __HOST_DEVICE__ void rotate(T vx, T vy, T vz, T& rx, T& ry, T& rz) const{
      // r = v + 2 w (u x v) + 2 (u x (u x v)), u = (x,y,z)
      T const ux=x_, uy=y_, uz=z_;
      T cx = uy*vz - uz*vy;
      T cy = uz*vx - ux*vz;
      T cz = ux*vy - uy*vx;
      T ccx = uy*cz - uz*cy;
      T ccy = uz*cx - ux*cz;
      T ccz = ux*cy - uy*cx;
      rx = vx + T(2)*w_*cx + T(2)*ccx;
      ry = vy + T(2)*w_*cy + T(2)*ccy;
      rz = vz + T(2)*w_*cz + T(2)*ccz;
    }
    // Tangent-space (spatial / left-perturbation) Jacobian of the rotated vector
    // wrt the rotation vector omega: d(Rv)/domega = -skew(Rv), i.e. for a small
    // spatial increment delta, R -> exp([delta]_x) R and (Rv) -> (Rv) + delta x (Rv).
    // Returns J (row-major 3x3), J[3*i + k] = d (Rv)_i / d omega_k.
    __HOST_DEVICE__ void rotate_jacobian(T vx, T vy, T vz, T (&J)[9]) const{
      T rx, ry, rz; rotate(vx, vy, vz, rx, ry, rz);
      // -skew(r): [ 0, rz, -ry ; -rz, 0, rx ; ry, -rx, 0 ]
      J[0]=T(0); J[1]= rz;  J[2]=-ry;
      J[3]=-rz;  J[4]=T(0); J[5]= rx;
      J[6]= ry;  J[7]=-rx;  J[8]=T(0);
    }

    friend struct IvyNodeSelfRelations<IvyVersor<T>>;
  };
}
namespace IvyTypes{
  template<typename T> struct convert_to_floating_point<IvyMath::IvyVersor<T>>{
    using type = IvyMath::IvyVersor<convert_to_floating_point_t<T>>;
  };
}
namespace IvyMath{
  template<typename T> struct IvyNodeSelfRelations<IvyVersor<T>>{
    static __HOST_DEVICE__ constexpr bool is_differentiable(IvyVersor<T> const& x){ return false; }
    static __HOST_DEVICE__ void conjugate(IvyVersor<T>& x){ x.x_ = -x.x_; x.y_ = -x.y_; x.z_ = -x.z_; }
    static constexpr bool is_conjugatable = true;
  };

  template<typename T> using IvyVersorPtr_t = IvyThreadSafePtr_t< IvyVersor<T> >;
  template<typename T, typename... Args> __HOST_DEVICE__ IvyVersorPtr_t<T> Versor(Args&&... args){ return make_IvyThreadSafePtr< IvyVersor<T> >(args...); }
}
namespace std_ivy{
  template<typename T> struct value_printout<IvyMath::IvyVersor<T>>{
    static __HOST_DEVICE__ void print(IvyMath::IvyVersor<T> const& var){
      __PRINT_INFO__("Versor(");
      print_value(var.W(), false); __PRINT_INFO__(", ");
      print_value(var.X(), false); __PRINT_INFO__("i, ");
      print_value(var.Y(), false); __PRINT_INFO__("j, ");
      print_value(var.Z(), false); __PRINT_INFO__("k)");
    }
  };
}


#endif
