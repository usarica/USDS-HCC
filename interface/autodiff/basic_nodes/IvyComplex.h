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
  /*
  Complex representation charts.
  A chart is a compile-time policy that fixes how a complex value is parametrized
  (its stored degrees of freedom) and how those map to the CANONICAL Cartesian
  components (re, im). Function outputs are always canonical, so a non-canonical
  chart only ever lives at a user leaf and enters at the gradient seed via its
  local Jacobian d(re,im)/d(DOF). Every chart exposes:
    - storage_t<T>           : the stored DOFs
    - tag                    : a representation tag (for diagnostics/dispatch)
    - Re/Im                  : canonical accessors computed from storage
    - from_cartesian         : build storage from canonical (re, im)
    - conjugate              : in-place complex conjugation in chart coordinates
    - jacobian               : the 2x2 matrix [d re/d q0, d re/d q1; d im/...]
                               evaluated at the current storage (row-major j[4])
  */
  struct complex_cartesian_chart{
    using tag = canonical_chart_tag;
    template<typename T> struct storage_t{
      T re, im;
      __HOST_DEVICE__ storage_t() : re(0), im(0){}
      __HOST_DEVICE__ storage_t(T const& a, T const& b) : re(a), im(b){}
    };
    template<typename T> static __HOST_DEVICE__ T Re(storage_t<T> const& s){ return s.re; }
    template<typename T> static __HOST_DEVICE__ T Im(storage_t<T> const& s){ return s.im; }
    template<typename T> static __HOST_DEVICE__ storage_t<T> from_cartesian(T const& re, T const& im){ return storage_t<T>(re, im); }
    template<typename T> static __HOST_DEVICE__ void conjugate(storage_t<T>& s){ s.im = -s.im; }
    template<typename T> static __HOST_DEVICE__ void jacobian(storage_t<T> const& /*s*/, T (&j)[4]){
      j[0] = T(1); j[1] = T(0);
      j[2] = T(0); j[3] = T(1);
    }
  };
  struct complex_polar_chart{
    using tag = real_domain_tag; // non-canonical (placeholder structural tag)
    template<typename T> struct storage_t{
      T r, phi;
      __HOST_DEVICE__ storage_t() : r(0), phi(0){}
      __HOST_DEVICE__ storage_t(T const& a, T const& b) : r(a), phi(b){}
    };
    template<typename T> static __HOST_DEVICE__ T Re(storage_t<T> const& s){ return s.r*std_math::cos(s.phi); }
    template<typename T> static __HOST_DEVICE__ T Im(storage_t<T> const& s){ return s.r*std_math::sin(s.phi); }
    template<typename T> static __HOST_DEVICE__ storage_t<T> from_cartesian(T const& re, T const& im){
      return storage_t<T>(std_math::sqrt(re*re + im*im), std_math::atan2(im, re));
    }
    template<typename T> static __HOST_DEVICE__ void conjugate(storage_t<T>& s){ s.phi = -s.phi; }
    // d(re,im)/d(r,phi): [cosφ, -r sinφ ; sinφ, r cosφ].
    template<typename T> static __HOST_DEVICE__ void jacobian(storage_t<T> const& s, T (&j)[4]){
      T const c = std_math::cos(s.phi);
      T const sn = std_math::sin(s.phi);
      j[0] = c;  j[1] = -s.r*sn;
      j[2] = sn; j[3] =  s.r*c;
    }
  };

  template<typename T, typename Chart = complex_cartesian_chart, ENABLE_IF_ARITHMETIC(T)> class IvyComplex;
  template<typename T, typename Chart> struct IvyNodeSelfRelations<IvyComplex<T, Chart>>;
}
namespace std_ivy{
  using namespace IvyMath;
  template<typename T, typename Chart> class transfer_memory_primitive<IvyComplex<T, Chart>> : public transfer_memory_primitive_with_internal_memory<IvyComplex<T, Chart>, IvyComplex<T, Chart>>{};
}
namespace IvyMath{
  template<typename T, typename Chart, ENABLE_IF_ARITHMETIC_IMPL(T)> class IvyComplex final :
    public IvyBaseNode,
    public IvyClientManager<IvyComplex<T, Chart>>,
    public complex_domain_tag,
    public leaf_value_tag
  {
  public:
    using clientmgr_t = IvyClientManager<IvyComplex<T, Chart>>;
    using dtype_t = T;
    using value_t = IvyComplex<T, Chart>;
    using chart_t = Chart;
    using representation_t = Chart;
    using storage_t = typename Chart::template storage_t<T>;

    friend class std_mem::kernel_generic_transfer_internal_memory<IvyComplex<T, Chart>, IvyComplex<T, Chart>>;

  protected:
    storage_t data_;

    __HOST_DEVICE__ void set_canonical(T const& re_, T const& im_){ data_ = Chart::template from_cartesian<T>(re_, im_); }

  public:
    // Constructors
    __HOST_DEVICE__ IvyComplex() : clientmgr_t(), data_(Chart::template from_cartesian<T>(T(0), T(0))){}
    template<typename U, ENABLE_IF_ARITHMETIC(U)> __HOST_DEVICE__ IvyComplex(U const& re_) : clientmgr_t(), data_(Chart::template from_cartesian<T>(__STATIC_CAST__(T, re_), T(0))){}
    __HOST_DEVICE__ IvyComplex(T const& re_) : clientmgr_t(), data_(Chart::template from_cartesian<T>(re_, T(0))){}
    __HOST_DEVICE__ IvyComplex(T const& re_, T const& im_) : clientmgr_t(), data_(Chart::template from_cartesian<T>(re_, im_)){}
    template<typename U, typename C2> __HOST_DEVICE__ IvyComplex(IvyComplex<U, C2> const& other) : clientmgr_t(), data_(Chart::template from_cartesian<T>(__STATIC_CAST__(T, other.Re()), __STATIC_CAST__(T, other.Im()))){}
    __HOST_DEVICE__ IvyComplex(IvyComplex<T, Chart> const& other) : clientmgr_t(), data_(other.data_){}
    __HOST_DEVICE__ IvyComplex(IvyComplex<T, Chart>&& other) : clientmgr_t(), data_(std_util::move(other.data_)){}
    template<typename U> __HOST_DEVICE__ IvyComplex(IvyScalar<U> const& other) : clientmgr_t(), data_(Chart::template from_cartesian<T>(__STATIC_CAST__(T, other.value()), T(0))){}
    __HOST_DEVICE__ IvyComplex(IvyScalar<T> const& other) : clientmgr_t(), data_(Chart::template from_cartesian<T>(other.value(), T(0))){}

    // Empty destructor
    __HOST_DEVICE__ ~IvyComplex(){}

    // Assignment operator
    template<typename U, typename C2> __HOST_DEVICE__ IvyComplex& operator=(IvyComplex<U, C2> const& other){
      set_canonical(__STATIC_CAST__(T, other.Re()), __STATIC_CAST__(T, other.Im()));
      this->update_clients_modified();
      return *this;
    }
    __HOST_DEVICE__ IvyComplex& operator=(IvyComplex<T, Chart> const& other){
      data_ = other.data_;
      this->update_clients_modified();
      return *this;
    }
    __HOST_DEVICE__ IvyComplex& operator=(IvyComplex<T, Chart>&& other){
      data_ = std_util::move(other.data_);
      this->update_clients_modified();
      return *this;
    }
    template<typename U, ENABLE_IF_ARITHMETIC(U)> __HOST_DEVICE__ IvyComplex& operator=(U const& re_){
      set_canonical(__STATIC_CAST__(T, re_), T(0));
      this->update_clients_modified();
      return *this;
    }
    __HOST_DEVICE__ IvyComplex& operator=(T const& re_){
      set_canonical(re_, T(0));
      this->update_clients_modified();
      return *this;
    }
    template<typename U> __HOST_DEVICE__ IvyComplex& operator=(IvyScalar<U> const& other){
      set_canonical(__STATIC_CAST__(T, other.value()), T(0));
      this->update_clients_modified();
      return *this;
    }
    __HOST_DEVICE__ IvyComplex& operator=(IvyScalar<T> const& other){
      set_canonical(other.value(), T(0));
      this->update_clients_modified();
      return *this;
    }

    // Canonical (Cartesian) accessors
    __HOST_DEVICE__ T Re() const{ return Chart::template Re<T>(data_); }
    __HOST_DEVICE__ T Im() const{ return Chart::template Im<T>(data_); }

    __HOST_DEVICE__ T norm() const{ T const re = Re(); T const im = Im(); return std_math::sqrt(re*re + im*im); }
    __HOST_DEVICE__ T phase() const{ return std_math::atan2(Im(), Re()); }

    // Chart degrees of freedom and local Jacobian d(re,im)/d(DOF).
    __HOST_DEVICE__ storage_t const& chart_dofs() const{ return data_; }
    __HOST_DEVICE__ void chart_jacobian(T (&j)[4]) const{ Chart::template jacobian<T>(data_, j); }

    // Ensure that a complex value can operate in just the same way as a scalar leaf.
    __HOST_DEVICE__ value_t const& value() const{ return *this; }

    // Set functions
    __HOST_DEVICE__ void set_real(T const& x){ set_canonical(x, Im()); this->update_clients_modified(); }
    __HOST_DEVICE__ void set_imaginary(T const& x){ set_canonical(Re(), x); this->update_clients_modified(); }

    __HOST_DEVICE__ void set_absval_phase(T const& v, T const& phi){
      set_canonical(v*std_math::cos(phi), v*std_math::sin(phi));
      this->update_clients_modified();
    }

    friend struct IvyNodeSelfRelations<IvyComplex<T, Chart>>;
  };
}
namespace IvyTypes{
  template<typename T, typename Chart> struct convert_to_floating_point<IvyMath::IvyComplex<T, Chart>>{
    using type = IvyMath::IvyComplex<convert_to_floating_point_t<T>, Chart>;
  };
}
namespace IvyMath{
  template<typename T, typename Chart> struct IvyNodeSelfRelations<IvyComplex<T, Chart>>{
    static __HOST_DEVICE__ constexpr bool is_differentiable(IvyComplex<T, Chart> const& x){ return false; }
    static __HOST_DEVICE__ void conjugate(IvyComplex<T, Chart>& x){ Chart::template conjugate<T>(x.data_); }
    static constexpr bool is_conjugatable = true;
  };

  template<typename T, typename Chart> struct convert_to_floating_point_if_complex<IvyComplex<T, Chart>>{
    using type = IvyComplex<convert_to_floating_point_t<T>, Chart>;
  };
  template<typename T, typename Chart> struct convert_to_real_type<IvyComplex<T, Chart>>{
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
  template<typename T, typename Chart> struct convert_to_complex_type<IvyComplex<T, Chart>>{
    using type = IvyComplex<T, Chart>;
  };

  template<typename T> struct minimal_domain_type<T, complex_domain_tag, leaf_value_tag>{ using type = IvyComplex<std_ttraits::remove_cv_t<T>>; };

  template<typename T, typename Chart = complex_cartesian_chart> using IvyComplexPtr_t = IvyThreadSafePtr_t< IvyComplex<T, Chart> >;

  template<typename T, typename... Args> __HOST_DEVICE__ IvyComplexPtr_t<T> Complex(Args&&... args){ return make_IvyThreadSafePtr< IvyComplex<T> >(args...); }
  template<typename T, typename Chart, typename... Args> __HOST_DEVICE__ IvyComplexPtr_t<T, Chart> ComplexChart(Args&&... args){ return make_IvyThreadSafePtr< IvyComplex<T, Chart> >(args...); }
}
namespace std_ivy{
  template<typename T, typename Chart> struct value_printout<IvyMath::IvyComplex<T, Chart>>{
    static __HOST_DEVICE__ void print(IvyMath::IvyComplex<T, Chart> const& var){
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
