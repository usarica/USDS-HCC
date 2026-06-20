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
  Quaternion representation charts.
  Canonical components are the four real Cartesian components (w, x, y, z) with
  w the scalar part and (x, y, z) the vector part. A chart fixes how those four
  canonical components are parametrized, and exposes the local 4x4 Jacobian
  d(w,x,y,z)/d(DOF) (row-major, j[4*i + k] = d comp_i / d dof_k). Every chart
  exposes:
    - storage_t<T>  : the stored DOFs
    - tag           : a representation tag
    - W/X/Y/Z       : canonical accessors computed from storage
    - from_components: build storage from canonical (w,x,y,z)
    - conjugate     : in-place quaternion conjugation in chart coordinates
    - jacobian      : the 4x4 matrix d(canonical)/d(DOF) at the current storage
  */
  struct quaternion_components_chart{
    using tag = canonical_chart_tag;
    template<typename T> struct storage_t{
      T w, x, y, z;
      __HOST_DEVICE__ storage_t() : w(0), x(0), y(0), z(0){}
      __HOST_DEVICE__ storage_t(T const& a, T const& b, T const& c, T const& d) : w(a), x(b), y(c), z(d){}
    };
    template<typename T> static __HOST_DEVICE__ T W(storage_t<T> const& s){ return s.w; }
    template<typename T> static __HOST_DEVICE__ T X(storage_t<T> const& s){ return s.x; }
    template<typename T> static __HOST_DEVICE__ T Y(storage_t<T> const& s){ return s.y; }
    template<typename T> static __HOST_DEVICE__ T Z(storage_t<T> const& s){ return s.z; }
    template<typename T> static __HOST_DEVICE__ storage_t<T> from_components(T const& w, T const& x, T const& y, T const& z){ return storage_t<T>(w, x, y, z); }
    template<typename T> static __HOST_DEVICE__ void conjugate(storage_t<T>& s){ s.x = -s.x; s.y = -s.y; s.z = -s.z; }
    template<typename T> static __HOST_DEVICE__ void jacobian(storage_t<T> const& /*s*/, T (&j)[16]){
      for (int i=0;i<16;++i) j[i]=T(0);
      j[0]=T(1); j[5]=T(1); j[10]=T(1); j[15]=T(1);
    }
  };

  /*
  Exponential / tangent chart for the full Hamilton algebra: the four stored DOFs
  (t0, t1, t2, t3) are a tangent vector whose quaternion exponential gives the
  canonical components. With a = t0 (scalar part), v = (t1, t2, t3), theta = |v|:
      exp(t) = e^a * ( cos(theta), sinc(theta)*v )    sinc(theta)=sin(theta)/theta.
  This is a smooth 4 -> 4 chart with a closed-form Jacobian (computed below). The
  scalar direction t0 scales the norm; the vector directions act in the Lie sense.
  */
  struct quaternion_exp_chart{
    using tag = real_domain_tag; // non-canonical structural marker
    template<typename T> struct storage_t{
      T t0, t1, t2, t3;
      __HOST_DEVICE__ storage_t() : t0(0), t1(0), t2(0), t3(0){}
      __HOST_DEVICE__ storage_t(T const& a, T const& b, T const& c, T const& d) : t0(a), t1(b), t2(c), t3(d){}
    };
    // exp evaluated to canonical (w,x,y,z)
    template<typename T> static __HOST_DEVICE__ void eval(storage_t<T> const& s, T& w, T& x, T& y, T& z){
      T const ea = std_math::exp(s.t0);
      T const th = std_math::sqrt(s.t1*s.t1 + s.t2*s.t2 + s.t3*s.t3);
      T sinc;
      if (th > T(1e-8)) sinc = std_math::sin(th)/th;
      else sinc = T(1) - th*th/T(6); // series for small theta
      w = ea*std_math::cos(th);
      x = ea*sinc*s.t1;
      y = ea*sinc*s.t2;
      z = ea*sinc*s.t3;
    }
    template<typename T> static __HOST_DEVICE__ T W(storage_t<T> const& s){ T w,x,y,z; eval(s,w,x,y,z); return w; }
    template<typename T> static __HOST_DEVICE__ T X(storage_t<T> const& s){ T w,x,y,z; eval(s,w,x,y,z); return x; }
    template<typename T> static __HOST_DEVICE__ T Y(storage_t<T> const& s){ T w,x,y,z; eval(s,w,x,y,z); return y; }
    template<typename T> static __HOST_DEVICE__ T Z(storage_t<T> const& s){ T w,x,y,z; eval(s,w,x,y,z); return z; }
    // log: inverse map from canonical (w,x,y,z) to tangent storage.
    template<typename T> static __HOST_DEVICE__ storage_t<T> from_components(T const& w, T const& x, T const& y, T const& z){
      T const rho = std_math::sqrt(x*x + y*y + z*z);
      T const n = std_math::sqrt(w*w + x*x + y*y + z*z);
      T const a = std_math::log(n);
      if (rho > T(1e-12)){
        T const th = std_math::atan2(rho, w);
        T const f = th/rho;
        return storage_t<T>(a, x*f, y*f, z*f);
      }
      return storage_t<T>(a, T(0), T(0), T(0));
    }
    // conj(exp(a,v)) = exp(a,-v): negate the vector tangent components.
    template<typename T> static __HOST_DEVICE__ void conjugate(storage_t<T>& s){ s.t1 = -s.t1; s.t2 = -s.t2; s.t3 = -s.t3; }
    // d(w,x,y,z)/d(t0,t1,t2,t3), closed form.
    template<typename T> static __HOST_DEVICE__ void jacobian(storage_t<T> const& s, T (&j)[16]){
      T const ea = std_math::exp(s.t0);
      T const tv[3] = { s.t1, s.t2, s.t3 };
      T const th2 = s.t1*s.t1 + s.t2*s.t2 + s.t3*s.t3;
      T const th = std_math::sqrt(th2);
      T sinc, dfac, cth;
      cth = std_math::cos(th);
      if (th > T(1e-6)){
        sinc = std_math::sin(th)/th;
        dfac = (th*cth - std_math::sin(th))/(th2*th); // (theta cos - sin)/theta^3
      }
      else{
        sinc = T(1) - th2/T(6);
        dfac = T(-1)/T(3) + th2/T(30);
      }
      // column 0 (d/dt0) = exp(t) itself
      T w,x,y,z; eval(s,w,x,y,z);
      j[0]=w; j[4]=x; j[8]=y; j[12]=z;
      // vector columns k=1..3 (index kk=0..2 over tv)
      for (int kk=0; kk<3; ++kk){
        int col = kk+1;
        // dw/dt_k = -e^a sinc t_k = -(vector comp k of q)
        j[0*4 + col] = -ea*sinc*tv[kk];
        // dx_i/dt_k = e^a [ dfac * t_i t_k + sinc * delta_ik ]
        for (int ii=0; ii<3; ++ii){
          T delta = (ii==kk) ? T(1) : T(0);
          j[(ii+1)*4 + col] = ea*(dfac*tv[ii]*tv[kk] + sinc*delta);
        }
      }
    }
  };

  /*
  IvyQuaternion<T, Chart>: a leaf in the H (Hamilton quaternion) division algebra.
  Canonical components are (w, x, y, z). The Hamilton product is NON-commutative
  (i*j = k, j*i = -k), which is why the division-algebra traits in IvyTypeTags
  mark quaternion_domain_tag as is_commutative=false. Differentiation through
  quaternion-valued expressions is handled by the order-aware *Fcnal gradients in
  IvyMathBaseArithmetic. The Chart policy fixes the parametrization and exposes
  the local Jacobian for chart-aware gradient projection at user leaves.
  */
  template<typename T, typename Chart = quaternion_components_chart, ENABLE_IF_ARITHMETIC(T)> class IvyQuaternion;
  template<typename T, typename Chart> struct IvyNodeSelfRelations<IvyQuaternion<T, Chart>>;
}
namespace std_ivy{
  using namespace IvyMath;
  template<typename T, typename Chart> class transfer_memory_primitive<IvyQuaternion<T, Chart>> : public transfer_memory_primitive_with_internal_memory<IvyQuaternion<T, Chart>, IvyQuaternion<T, Chart>>{};
}
namespace IvyMath{
  template<typename T, typename Chart, ENABLE_IF_ARITHMETIC_IMPL(T)> class IvyQuaternion final :
    public IvyBaseNode,
    public IvyClientManager<IvyQuaternion<T, Chart>>,
    public quaternion_domain_tag,
    public leaf_value_tag
  {
  public:
    using clientmgr_t = IvyClientManager<IvyQuaternion<T, Chart>>;
    using dtype_t = T;
    using value_t = IvyQuaternion<T, Chart>;
    using chart_t = Chart;
    using representation_t = Chart;
    using storage_t = typename Chart::template storage_t<T>;

    friend class std_mem::kernel_generic_transfer_internal_memory<IvyQuaternion<T, Chart>, IvyQuaternion<T, Chart>>;

  protected:
    storage_t data_;

    __HOST_DEVICE__ void set_canonical(T const& w, T const& x, T const& y, T const& z){ data_ = Chart::template from_components<T>(w, x, y, z); }

  public:
    // Constructors
    __HOST_DEVICE__ IvyQuaternion() : clientmgr_t(), data_(Chart::template from_components<T>(T(0), T(0), T(0), T(0))){}
    template<typename U, ENABLE_IF_ARITHMETIC(U)> __HOST_DEVICE__ IvyQuaternion(U const& w) : clientmgr_t(), data_(Chart::template from_components<T>(__STATIC_CAST__(T, w), T(0), T(0), T(0))){}
    __HOST_DEVICE__ IvyQuaternion(T const& w) : clientmgr_t(), data_(Chart::template from_components<T>(w, T(0), T(0), T(0))){}
    __HOST_DEVICE__ IvyQuaternion(T const& w, T const& x, T const& y, T const& z) : clientmgr_t(), data_(Chart::template from_components<T>(w, x, y, z)){}
    template<typename U, typename C2> __HOST_DEVICE__ IvyQuaternion(IvyQuaternion<U, C2> const& other) :
      clientmgr_t(), data_(Chart::template from_components<T>(__STATIC_CAST__(T, other.W()), __STATIC_CAST__(T, other.X()), __STATIC_CAST__(T, other.Y()), __STATIC_CAST__(T, other.Z()))){}
    __HOST_DEVICE__ IvyQuaternion(IvyQuaternion<T, Chart> const& other) : clientmgr_t(), data_(other.data_){}
    __HOST_DEVICE__ IvyQuaternion(IvyQuaternion<T, Chart>&& other) : clientmgr_t(), data_(std_util::move(other.data_)){}
    template<typename U> __HOST_DEVICE__ IvyQuaternion(IvyScalar<U> const& other) : clientmgr_t(), data_(Chart::template from_components<T>(__STATIC_CAST__(T, other.value()), T(0), T(0), T(0))){}
    __HOST_DEVICE__ IvyQuaternion(IvyScalar<T> const& other) : clientmgr_t(), data_(Chart::template from_components<T>(other.value(), T(0), T(0), T(0))){}
    template<typename U, typename C2> __HOST_DEVICE__ IvyQuaternion(IvyComplex<U, C2> const& other) : clientmgr_t(), data_(Chart::template from_components<T>(__STATIC_CAST__(T, other.Re()), __STATIC_CAST__(T, other.Im()), T(0), T(0))){}

    // Empty destructor
    __HOST_DEVICE__ ~IvyQuaternion(){}

    // Assignment operators
    template<typename U, typename C2> __HOST_DEVICE__ IvyQuaternion& operator=(IvyQuaternion<U, C2> const& other){
      set_canonical(__STATIC_CAST__(T, other.W()), __STATIC_CAST__(T, other.X()), __STATIC_CAST__(T, other.Y()), __STATIC_CAST__(T, other.Z()));
      this->update_clients_modified();
      return *this;
    }
    __HOST_DEVICE__ IvyQuaternion& operator=(IvyQuaternion<T, Chart> const& other){
      data_ = other.data_;
      this->update_clients_modified();
      return *this;
    }
    __HOST_DEVICE__ IvyQuaternion& operator=(IvyQuaternion<T, Chart>&& other){
      data_ = std_util::move(other.data_);
      this->update_clients_modified();
      return *this;
    }
    template<typename U, ENABLE_IF_ARITHMETIC(U)> __HOST_DEVICE__ IvyQuaternion& operator=(U const& w){
      set_canonical(__STATIC_CAST__(T, w), T(0), T(0), T(0));
      this->update_clients_modified();
      return *this;
    }
    __HOST_DEVICE__ IvyQuaternion& operator=(T const& w){
      set_canonical(w, T(0), T(0), T(0));
      this->update_clients_modified();
      return *this;
    }

    // Canonical (component) accessors
    __HOST_DEVICE__ T W() const{ return Chart::template W<T>(data_); }
    __HOST_DEVICE__ T X() const{ return Chart::template X<T>(data_); }
    __HOST_DEVICE__ T Y() const{ return Chart::template Y<T>(data_); }
    __HOST_DEVICE__ T Z() const{ return Chart::template Z<T>(data_); }

    __HOST_DEVICE__ T norm2() const{ T const w=W(), x=X(), y=Y(), z=Z(); return w*w + x*x + y*y + z*z; }
    __HOST_DEVICE__ T norm() const{ return std_math::sqrt(this->norm2()); }

    // Chart degrees of freedom and local Jacobian d(canonical)/d(DOF) (4x4 row-major).
    __HOST_DEVICE__ storage_t const& chart_dofs() const{ return data_; }
    __HOST_DEVICE__ void chart_jacobian(T (&j)[16]) const{ Chart::template jacobian<T>(data_, j); }

    // value() lets a quaternion leaf operate like any other Ivy value node.
    __HOST_DEVICE__ value_t const& value() const{ return *this; }

    // Set functions
    __HOST_DEVICE__ void set_components(T const& w, T const& x, T const& y, T const& z){
      set_canonical(w, x, y, z); this->update_clients_modified();
    }
    __HOST_DEVICE__ void set_scalar(T const& w){ set_canonical(w, X(), Y(), Z()); this->update_clients_modified(); }
    __HOST_DEVICE__ void set_vector(T const& x, T const& y, T const& z){ set_canonical(W(), x, y, z); this->update_clients_modified(); }

    friend struct IvyNodeSelfRelations<IvyQuaternion<T, Chart>>;
  };
}
namespace IvyTypes{
  template<typename T, typename Chart> struct convert_to_floating_point<IvyMath::IvyQuaternion<T, Chart>>{
    using type = IvyMath::IvyQuaternion<convert_to_floating_point_t<T>, Chart>;
  };
}
namespace IvyMath{
  template<typename T, typename Chart> struct IvyNodeSelfRelations<IvyQuaternion<T, Chart>>{
    static __HOST_DEVICE__ constexpr bool is_differentiable(IvyQuaternion<T, Chart> const& x){ return false; }
    // Quaternion conjugation negates the vector part: conj(w,x,y,z) = (w,-x,-y,-z).
    static __HOST_DEVICE__ void conjugate(IvyQuaternion<T, Chart>& x){ Chart::template conjugate<T>(x.data_); }
    static constexpr bool is_conjugatable = true;
  };

  template<typename T, typename Chart> struct convert_to_floating_point_if_complex<IvyQuaternion<T, Chart>>{
    using type = IvyQuaternion<convert_to_floating_point_t<T>, Chart>;
  };
  template<typename T, typename Chart> struct convert_to_real_type<IvyQuaternion<T, Chart>>{
    using type = IvyScalar<T>;
  };

  template<typename T> struct minimal_domain_type<T, quaternion_domain_tag, leaf_value_tag>{ using type = IvyQuaternion<std_ttraits::remove_cv_t<T>>; };

  template<typename T, typename Chart = quaternion_components_chart> using IvyQuaternionPtr_t = IvyThreadSafePtr_t< IvyQuaternion<T, Chart> >;

  template<typename T, typename... Args> __HOST_DEVICE__ IvyQuaternionPtr_t<T> Quaternion(Args&&... args){ return make_IvyThreadSafePtr< IvyQuaternion<T> >(args...); }
  template<typename T, typename Chart, typename... Args> __HOST_DEVICE__ IvyQuaternionPtr_t<T, Chart> QuaternionChart(Args&&... args){ return make_IvyThreadSafePtr< IvyQuaternion<T, Chart> >(args...); }
}
namespace std_ivy{
  template<typename T, typename Chart> struct value_printout<IvyMath::IvyQuaternion<T, Chart>>{
    static __HOST_DEVICE__ void print(IvyMath::IvyQuaternion<T, Chart> const& var){
      __PRINT_INFO__("Quaternion(");
      print_value(var.W(), false); __PRINT_INFO__(", ");
      print_value(var.X(), false); __PRINT_INFO__("i, ");
      print_value(var.Y(), false); __PRINT_INFO__("j, ");
      print_value(var.Z(), false); __PRINT_INFO__("k)");
    }
  };
}


#endif
