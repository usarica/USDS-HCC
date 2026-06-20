#ifndef IVYADTENSOROPS_H
#define IVYADTENSOROPS_H

/**
 * @file IvyADTensorOps.h
 * @brief Fused element-wise operations and reductions for IvyADTensor (SoA reverse-mode AD).
 *
 * @details
 * Each operation:
 *   1. allocates a result node sharing the input shape,
 *   2. computes the forward value with a single fused loop over the contiguous buffers, and
 *   3. records the parents and a backward closure implementing the vector-Jacobian product
 *      (VJP) that accumulates the output adjoint into the input adjoints.
 *
 * Backward closures capture the result and its inputs by raw pointer; ownership and reverse
 * traversal are provided by the result's `parents_` (`shared_ptr`) set via `set_tape`.
 *
 * @note Namespace usage: the SoA tensor lives in @c IvyAD (separate from @c IvyMath). The
 *       operator overloads (`+ - * /`) are robust even when @c IvyMath / @c std_ivy are in
 *       scope (the generic @c IvyMath operators are domain-constrained and SFINAE out). The
 *       named free functions (`Exp`, `Multiply`, …) share names with @c IvyMath; call them as
 *       @c IvyAD::Exp(...) or via `using namespace IvyAD` without also pulling @c IvyMath's
 *       math names unqualified.
 */

#include "autodiff/tensor_soa/IvyADTensor.h"


namespace IvyAD{
  using IvyMath::IvyTensorShape;
  using IvyMath::IvyTensorDim_t;

  namespace ad_detail{
    /** @brief True if the result of an op over these inputs should track gradients. */
    template<typename T> __INLINE_FCN_RELAXED__ __HOST__ bool any_requires_grad(IvyADTensorPtr<T> const& a){ return a->requires_grad(); }
    template<typename T> __INLINE_FCN_RELAXED__ __HOST__ bool any_requires_grad(IvyADTensorPtr<T> const& a, IvyADTensorPtr<T> const& b){ return a->requires_grad() || b->requires_grad(); }
  }

  // ===========================================================================
  // Unary element-wise operations
  // ===========================================================================

  /**
   * @brief Generic unary element-wise op.
   * @tparam FwdOp   f(x_i) -> value
   * @tparam BwdOp   (x_i, y_i) -> local derivative dy_i/dx_i (y is the forward output)
   */
  template<typename T, typename FwdOp, typename BwdOp>
  __HOST__ IvyADTensorPtr<T> ad_unary(IvyADTensorPtr<T> const& x, FwdOp fwd, BwdOp bwd){
    bool const rg = ad_detail::any_requires_grad(x);
    auto out = make_ADTensor<T>(x->shape(), rg);
    IvyTensorDim_t const n = out->num_elements();
    T const* xv = x->value_data();
    T* ov = out->value_data();
    ivy_ad_for(n, [ov, xv, fwd](IvyTensorDim_t i){ ov[i] = fwd(xv[i]); });
    if (rg){
      IvyADTensor<T>* self = out.get();
      IvyADTensor<T>* xp = x.get();
      out->set_tape({x}, [self, xp, n, bwd](){
        if (!xp->requires_grad()) return;
        T const* og = self->grad_data();
        T const* ov = self->value_data();
        T const* xv = xp->value_data();
        T* xg = xp->grad_data();
        ivy_ad_for(n, [xg, og, ov, xv, bwd](IvyTensorDim_t i){ xg[i] += og[i] * bwd(xv[i], ov[i]); });
      });
    }
    return out;
  }

  template<typename T> __HOST__ IvyADTensorPtr<T> Negate(IvyADTensorPtr<T> const& x){
    return ad_unary<T>(x, [](T v){ return -v; }, [](T, T){ return T(-1); });
  }
  template<typename T> __HOST__ IvyADTensorPtr<T> operator-(IvyADTensorPtr<T> const& x){ return Negate(x); }

  template<typename T> __HOST__ IvyADTensorPtr<T> Exp(IvyADTensorPtr<T> const& x){
    // y = exp(x), dy/dx = exp(x) = y
    return ad_unary<T>(x, [](T v){ return std_math::exp(v); }, [](T, T y){ return y; });
  }
  template<typename T> __HOST__ IvyADTensorPtr<T> Log(IvyADTensorPtr<T> const& x){
    // y = log(x), dy/dx = 1/x
    return ad_unary<T>(x, [](T v){ return std_math::log(v); }, [](T xv, T){ return T(1) / xv; });
  }
  template<typename T> __HOST__ IvyADTensorPtr<T> Sin(IvyADTensorPtr<T> const& x){
    return ad_unary<T>(x, [](T v){ return std_math::sin(v); }, [](T xv, T){ return std_math::cos(xv); });
  }
  template<typename T> __HOST__ IvyADTensorPtr<T> Cos(IvyADTensorPtr<T> const& x){
    return ad_unary<T>(x, [](T v){ return std_math::cos(v); }, [](T xv, T){ return -std_math::sin(xv); });
  }
  template<typename T> __HOST__ IvyADTensorPtr<T> Sqrt(IvyADTensorPtr<T> const& x){
    // y = sqrt(x), dy/dx = 1/(2 sqrt(x)) = 1/(2y)
    return ad_unary<T>(x, [](T v){ return std_math::sqrt(v); }, [](T, T y){ return T(1) / (T(2) * y); });
  }

  // ===========================================================================
  // Tensor-scalar operations (the scalar is a constant; no gradient flows to it)
  // ===========================================================================

  template<typename T> __HOST__ IvyADTensorPtr<T> AddScalar(IvyADTensorPtr<T> const& x, T c){
    return ad_unary<T>(x, [c](T v){ return v + c; }, [](T, T){ return T(1); });
  }
  template<typename T> __HOST__ IvyADTensorPtr<T> MultiplyScalar(IvyADTensorPtr<T> const& x, T c){
    return ad_unary<T>(x, [c](T v){ return v * c; }, [c](T, T){ return c; });
  }
  template<typename T> __HOST__ IvyADTensorPtr<T> operator+(IvyADTensorPtr<T> const& x, T c){ return AddScalar(x, c); }
  template<typename T> __HOST__ IvyADTensorPtr<T> operator*(IvyADTensorPtr<T> const& x, T c){ return MultiplyScalar(x, c); }
  template<typename T> __HOST__ IvyADTensorPtr<T> operator+(T c, IvyADTensorPtr<T> const& x){ return AddScalar(x, c); }
  template<typename T> __HOST__ IvyADTensorPtr<T> operator*(T c, IvyADTensorPtr<T> const& x){ return MultiplyScalar(x, c); }

  // ===========================================================================
  // Binary element-wise operations (same shape)
  // ===========================================================================

  template<typename T> __HOST__ IvyADTensorPtr<T> Add(IvyADTensorPtr<T> const& x, IvyADTensorPtr<T> const& y){
    bool const rg = ad_detail::any_requires_grad(x, y);
    auto out = make_ADTensor<T>(x->shape(), rg);
    IvyTensorDim_t const n = out->num_elements();
    T const* xv = x->value_data(); T const* yv = y->value_data(); T* ov = out->value_data();
    ivy_ad_for(n, [ov, xv, yv](IvyTensorDim_t i){ ov[i] = xv[i] + yv[i]; });
    if (rg){
      IvyADTensor<T>* self = out.get(); IvyADTensor<T>* xp = x.get(); IvyADTensor<T>* yp = y.get();
      out->set_tape({x, y}, [self, xp, yp, n](){
        T const* og = self->grad_data();
        if (xp->requires_grad()){ T* xg = xp->grad_data(); ivy_ad_for(n, [xg, og](IvyTensorDim_t i){ xg[i] += og[i]; }); }
        if (yp->requires_grad()){ T* yg = yp->grad_data(); ivy_ad_for(n, [yg, og](IvyTensorDim_t i){ yg[i] += og[i]; }); }
      });
    }
    return out;
  }

  template<typename T> __HOST__ IvyADTensorPtr<T> Subtract(IvyADTensorPtr<T> const& x, IvyADTensorPtr<T> const& y){
    bool const rg = ad_detail::any_requires_grad(x, y);
    auto out = make_ADTensor<T>(x->shape(), rg);
    IvyTensorDim_t const n = out->num_elements();
    T const* xv = x->value_data(); T const* yv = y->value_data(); T* ov = out->value_data();
    ivy_ad_for(n, [ov, xv, yv](IvyTensorDim_t i){ ov[i] = xv[i] - yv[i]; });
    if (rg){
      IvyADTensor<T>* self = out.get(); IvyADTensor<T>* xp = x.get(); IvyADTensor<T>* yp = y.get();
      out->set_tape({x, y}, [self, xp, yp, n](){
        T const* og = self->grad_data();
        if (xp->requires_grad()){ T* xg = xp->grad_data(); ivy_ad_for(n, [xg, og](IvyTensorDim_t i){ xg[i] += og[i]; }); }
        if (yp->requires_grad()){ T* yg = yp->grad_data(); ivy_ad_for(n, [yg, og](IvyTensorDim_t i){ yg[i] -= og[i]; }); }
      });
    }
    return out;
  }

  template<typename T> __HOST__ IvyADTensorPtr<T> Multiply(IvyADTensorPtr<T> const& x, IvyADTensorPtr<T> const& y){
    bool const rg = ad_detail::any_requires_grad(x, y);
    auto out = make_ADTensor<T>(x->shape(), rg);
    IvyTensorDim_t const n = out->num_elements();
    T const* xv = x->value_data(); T const* yv = y->value_data(); T* ov = out->value_data();
    ivy_ad_for(n, [ov, xv, yv](IvyTensorDim_t i){ ov[i] = xv[i] * yv[i]; });
    if (rg){
      IvyADTensor<T>* self = out.get(); IvyADTensor<T>* xp = x.get(); IvyADTensor<T>* yp = y.get();
      out->set_tape({x, y}, [self, xp, yp, n](){
        T const* og = self->grad_data();
        T const* xv = xp->value_data(); T const* yv = yp->value_data();
        if (xp->requires_grad()){ T* xg = xp->grad_data(); ivy_ad_for(n, [xg, og, yv](IvyTensorDim_t i){ xg[i] += og[i] * yv[i]; }); }
        if (yp->requires_grad()){ T* yg = yp->grad_data(); ivy_ad_for(n, [yg, og, xv](IvyTensorDim_t i){ yg[i] += og[i] * xv[i]; }); }
      });
    }
    return out;
  }

  template<typename T> __HOST__ IvyADTensorPtr<T> Divide(IvyADTensorPtr<T> const& x, IvyADTensorPtr<T> const& y){
    bool const rg = ad_detail::any_requires_grad(x, y);
    auto out = make_ADTensor<T>(x->shape(), rg);
    IvyTensorDim_t const n = out->num_elements();
    T const* xv = x->value_data(); T const* yv = y->value_data(); T* ov = out->value_data();
    ivy_ad_for(n, [ov, xv, yv](IvyTensorDim_t i){ ov[i] = xv[i] / yv[i]; });
    if (rg){
      IvyADTensor<T>* self = out.get(); IvyADTensor<T>* xp = x.get(); IvyADTensor<T>* yp = y.get();
      out->set_tape({x, y}, [self, xp, yp, n](){
        T const* og = self->grad_data();
        T const* xv = xp->value_data(); T const* yv = yp->value_data();
        // d(x/y)/dx = 1/y ; d(x/y)/dy = -x/y^2
        if (xp->requires_grad()){ T* xg = xp->grad_data(); ivy_ad_for(n, [xg, og, yv](IvyTensorDim_t i){ xg[i] += og[i] / yv[i]; }); }
        if (yp->requires_grad()){ T* yg = yp->grad_data(); ivy_ad_for(n, [yg, og, xv, yv](IvyTensorDim_t i){ yg[i] += -og[i] * xv[i] / (yv[i] * yv[i]); }); }
      });
    }
    return out;
  }

  template<typename T> __HOST__ IvyADTensorPtr<T> operator+(IvyADTensorPtr<T> const& x, IvyADTensorPtr<T> const& y){ return Add(x, y); }
  template<typename T> __HOST__ IvyADTensorPtr<T> operator-(IvyADTensorPtr<T> const& x, IvyADTensorPtr<T> const& y){ return Subtract(x, y); }
  template<typename T> __HOST__ IvyADTensorPtr<T> operator*(IvyADTensorPtr<T> const& x, IvyADTensorPtr<T> const& y){ return Multiply(x, y); }
  template<typename T> __HOST__ IvyADTensorPtr<T> operator/(IvyADTensorPtr<T> const& x, IvyADTensorPtr<T> const& y){ return Divide(x, y); }

  // ===========================================================================
  // Reductions
  // ===========================================================================

  /**
   * @brief Sum all elements into a scalar (1-element) tensor.
   * @details Backward broadcasts the scalar adjoint to every input element.
   */
  template<typename T> __HOST__ IvyADTensorPtr<T> Sum(IvyADTensorPtr<T> const& x){
    bool const rg = ad_detail::any_requires_grad(x);
    auto out = make_ADTensor<T>(IvyTensorShape(std_ilist::initializer_list<IvyTensorDim_t>{1}), rg);
    IvyTensorDim_t const n = x->num_elements();
    T const* xv = x->value_data();
    T s = T(0);
    for (IvyTensorDim_t i = 0; i < n; ++i) s += xv[i];   // serial reduction (deterministic)
    out->value_data()[0] = s;
    if (rg){
      IvyADTensor<T>* self = out.get(); IvyADTensor<T>* xp = x.get();
      out->set_tape({x}, [self, xp, n](){
        if (!xp->requires_grad()) return;
        T const g = self->grad_at(0);
        T* xg = xp->grad_data();
        ivy_ad_for(n, [xg, g](IvyTensorDim_t i){ xg[i] += g; });
      });
    }
    return out;
  }
}

#endif
