#ifndef IVYMATHFUNCTIONPRIMITIVES_H
#define IVYMATHFUNCTIONPRIMITIVES_H


#include "autodiff/arithmetic/IvyMathFunctionPrimitives.hh"


namespace IvyMath{
  // General 1D function implementation
  template<typename T, typename Evaluator, typename precision_type, typename Domain, typename GradientDomain>
  __HOST__ IvyRegularFunction_1D<T, Evaluator, precision_type, Domain, GradientDomain>::IvyRegularFunction_1D(IvyThreadSafePtr_t<T> const& dep) : base_t(), dep(dep){}
  template<typename T, typename Evaluator, typename precision_type, typename Domain, typename GradientDomain>
  __HOST__ IvyRegularFunction_1D<T, Evaluator, precision_type, Domain, GradientDomain>::IvyRegularFunction_1D(IvyRegularFunction_1D<T, Evaluator, precision_type, Domain, GradientDomain> const& other) :
    base_t(__DYNAMIC_CAST__(base_t const&, other)),
    dep(other.dep)
  {}
  template<typename T, typename Evaluator, typename precision_type, typename Domain, typename GradientDomain>
  __HOST__ IvyRegularFunction_1D<T, Evaluator, precision_type, Domain, GradientDomain>::IvyRegularFunction_1D(IvyRegularFunction_1D<T, Evaluator, precision_type, Domain, GradientDomain>&& other) :
    base_t(__DYNAMIC_CAST__(base_t&&, std_util::move(other))),
    dep(std_util::move(other.dep))
  {}
  template<typename T, typename Evaluator, typename precision_type, typename Domain, typename GradientDomain>
  __HOST__ void IvyRegularFunction_1D<T, Evaluator, precision_type, Domain, GradientDomain>::eval() const{
    if (this->is_modified()) *(this->output) = evaluator_t::eval(unpack_function_input<T>::get(*dep));
  }
  template<typename T, typename Evaluator, typename precision_type, typename Domain, typename GradientDomain>
  __HOST__ bool IvyRegularFunction_1D<T, Evaluator, precision_type, Domain, GradientDomain>::depends_on(IvyBaseNode const* node) const{
    return (base_t::depends_on(node) || IvyMath::depends_on(dep, node));
  }
  template<typename T, typename Evaluator, typename precision_type, typename Domain, typename GradientDomain>
  __HOST__ IvyThreadSafePtr_t<typename IvyRegularFunction_1D<T, Evaluator, precision_type, Domain, GradientDomain>::grad_t> IvyRegularFunction_1D<T, Evaluator, precision_type, Domain, GradientDomain>::gradient(
    IvyThreadSafePtr_t<IvyBaseNode> const& var
  ) const{
    if constexpr (std_ttraits::is_same_v<Domain, tensor_domain_tag>){
      // Tensor domain: compute the chain rule eagerly element-wise.
      // Using the lazy pointer × pointer multiply would instantiate
      // IvyMultiply<T,T> whose gradient() (pure-virtual in
      // IvyFunction<T,tensor,tensor>) cannot be satisfied without infinite
      // template recursion. Instead we compute f'(dep[i]) * ∂dep[i]/∂var
      // for each i directly and pack the results into an IvyTensorEagerFunction.
      //
      // VT is the *value tensor* type. When `dep` is a bare tensor leaf VT == T;
      // when `dep` is itself a function of a tensor (an inner node of a chain such
      // as Sin(Exp(t))), VT is dep's value tensor. Working in VT throughout makes
      // the single code path handle both the leaf and the chained case.
      using VT = unpack_if_function_t<T>;
      using dtype_t = typename VT::dtype_t;
      constexpr std_ivy::IvyMemoryType mem = IvyMemoryHelpers::get_execution_default_memory();
      // Local derivative f'(dep) evaluated at dep's current value tensor.
      auto dep_val_ptr = make_IvyThreadSafePtr<VT>(mem, nullptr, unpack_function_input<T>::get(*dep));
      auto f_prime  = evaluator_t::gradient(dep_val_ptr);   // IvyThreadSafePtr_t<VT>
      auto grad_dep = function_gradient<T>::get(*dep, var);  // IvyThreadSafePtr_t<VT>
      VT result(*f_prime);
      IvyTensorDim_t const n = result.num_elements();
      if constexpr (is_pointer_v<dtype_t>){
        using inner_t = typename dtype_t::element_type;
        for (IvyTensorDim_t i = 0; i < n; ++i){
          auto const fv = unpack_function_input_reduced<inner_t>::get(*(*f_prime)[i]);
          auto const gv = unpack_function_input_reduced<inner_t>::get(*(*grad_dep)[i]);
          result[i] = make_IvyThreadSafePtr<inner_t>(mem, nullptr, fv * gv);
        }
      } else if constexpr (is_complex_v<dtype_t>){
        // Complex cells are not closed under arithmetic (cell*cell -> IvyComplex):
        // read operands as IvyComplex, apply the (holomorphic) chain rule, and
        // write the canonical components back through the cell constructor.
        using rT = typename dtype_t::dtype_t;
        for (IvyTensorDim_t i = 0; i < n; ++i){
          IvyComplex<rT> const fv((*f_prime)[i].Re(), (*f_prime)[i].Im());
          IvyComplex<rT> const gv((*grad_dep)[i].Re(), (*grad_dep)[i].Im());
          auto const p = fv * gv;
          result[i] = dtype_t(p.Re(), p.Im());
        }
      } else {
        for (IvyTensorDim_t i = 0; i < n; ++i){
          auto const fv = unpack_function_input_reduced<dtype_t>::get((*f_prime)[i]);
          auto const gv = unpack_function_input_reduced<dtype_t>::get((*grad_dep)[i]);
          result[i] = fv * gv;
        }
      }
      return make_IvyThreadSafePtr<IvyTensorEagerFunction<VT>>(mem, nullptr, result);
    } else if constexpr (evaluator_is_reduction_v<evaluator_t>){
      // Reduction (tensor -> scalar), e.g. Sum: d(reduce(t))/dvar = reduce(dt/dvar).
      // Differentiate the operand to a value tensor of element partials ∂dep_i/∂var,
      // then contract it with the same reduction to a single scalar value node.
      // Self-identity: ∂(Sum t)/∂(Sum t) = 1.
      if (var && __STATIC_CAST__(IvyBaseNode const*, this) == var.get())
        return make_unit_function<precision_type, Domain>();
      using VT = unpack_if_function_t<T>;
      using E = typename VT::dtype_t;
      constexpr std_ivy::IvyMemoryType mem = IvyMemoryHelpers::get_execution_default_memory();
      auto grad_dep = function_gradient<T>::get(*dep, var); // IvyThreadSafePtr_t<VT>
      VT const& g = *grad_dep;
      IvyTensorDim_t const n = g.num_elements();
      dtype_t acc = dtype_t(0);
      for (IvyTensorDim_t i = 0; i < n; ++i){
        if constexpr (is_pointer_v<E>) acc += unpack_function_input_reduced<typename E::element_type>::get(*g[i]);
        else acc += unpack_function_input_reduced<E>::get(g[i]);
      }
      return make_IvyThreadSafePtr<IvyConstantFunction<precision_type, Domain>>(mem, nullptr, value_t(acc));
    } else {
      if (var && __STATIC_CAST__(IvyBaseNode const*, this) == var.get())
        return make_unit_function<precision_type, Domain>();
      auto grad_dep = function_gradient<T>::get(*dep, var);
      if constexpr (evaluator_is_order_aware_v<evaluator_t>)
        return evaluator_t::combine_gradient(dep, grad_dep);
      else
        return evaluator_t::gradient(dep)*grad_dep;
    }
  }

  // Special 1D case with no gradients
  template<typename T, typename Evaluator, typename precision_type, typename Domain>
  __HOST__ IvyRegularFunction_1D<T, Evaluator, precision_type, Domain, undefined_domain_tag>::IvyRegularFunction_1D(IvyThreadSafePtr_t<T> const& dep) : base_t(), dep(dep){}
  template<typename T, typename Evaluator, typename precision_type, typename Domain>
  __HOST__ IvyRegularFunction_1D<T, Evaluator, precision_type, Domain, undefined_domain_tag>::IvyRegularFunction_1D(IvyRegularFunction_1D<T, Evaluator, precision_type, Domain, undefined_domain_tag> const& other) :
    base_t(__DYNAMIC_CAST__(base_t const&, other)),
    dep(other.dep)
  {}
  template<typename T, typename Evaluator, typename precision_type, typename Domain>
  __HOST__ IvyRegularFunction_1D<T, Evaluator, precision_type, Domain, undefined_domain_tag>::IvyRegularFunction_1D(IvyRegularFunction_1D<T, Evaluator, precision_type, Domain, undefined_domain_tag>&& other) :
    base_t(__DYNAMIC_CAST__(base_t&&, std_util::move(other))),
    dep(std_util::move(other.dep))
  {}
  template<typename T, typename Evaluator, typename precision_type, typename Domain>
  __HOST__ void IvyRegularFunction_1D<T, Evaluator, precision_type, Domain, undefined_domain_tag>::eval() const{
    if (this->is_modified()) *(this->output) = evaluator_t::eval(unpack_function_input<T>::get(*dep));
  }
  template<typename T, typename Evaluator, typename precision_type, typename Domain>
  __HOST__ bool IvyRegularFunction_1D<T, Evaluator, precision_type, Domain, undefined_domain_tag>::depends_on(IvyBaseNode const* node) const{
    return (base_t::depends_on(node) || IvyMath::depends_on(dep, node));
  }

  // General 2D function implementation
  template<typename T, typename U, typename Evaluator, typename precision_type, typename Domain, typename GradientDomain>
  __HOST__ IvyRegularFunction_2D<T, U, Evaluator, precision_type, Domain, GradientDomain>::IvyRegularFunction_2D(IvyThreadSafePtr_t<T> const& x, IvyThreadSafePtr_t<U> const& y) : base_t(), x(x), y(y){}
  template<typename T, typename U, typename Evaluator, typename precision_type, typename Domain, typename GradientDomain>
  __HOST__ IvyRegularFunction_2D<T, U, Evaluator, precision_type, Domain, GradientDomain>::IvyRegularFunction_2D(IvyRegularFunction_2D<T, U, Evaluator, precision_type, Domain, GradientDomain> const& other) :
    base_t(__DYNAMIC_CAST__(base_t const&, other)),
    x(other.x),
    y(other.y)
  {}
  template<typename T, typename U, typename Evaluator, typename precision_type, typename Domain, typename GradientDomain>
  __HOST__ IvyRegularFunction_2D<T, U, Evaluator, precision_type, Domain, GradientDomain>::IvyRegularFunction_2D(IvyRegularFunction_2D<T, U, Evaluator, precision_type, Domain, GradientDomain>&& other) :
    base_t(__DYNAMIC_CAST__(base_t&&, std_util::move(other))),
    x(std_util::move(other.x)),
    y(std_util::move(other.y))
  {}
  template<typename T, typename U, typename Evaluator, typename precision_type, typename Domain, typename GradientDomain>
  __HOST__ void IvyRegularFunction_2D<T, U, Evaluator, precision_type, Domain, GradientDomain>::eval() const{
    if (this->is_modified()) *(this->output) = evaluator_t::eval(unpack_function_input<T>::get(*x), unpack_function_input<U>::get(*y));
  }
  template<typename T, typename U, typename Evaluator, typename precision_type, typename Domain, typename GradientDomain>
  __HOST__ bool IvyRegularFunction_2D<T, U, Evaluator, precision_type, Domain, GradientDomain>::depends_on(IvyBaseNode const* node) const{
    return (base_t::depends_on(node) || IvyMath::depends_on(x, node) || IvyMath::depends_on(y, node));
  }
  template<typename T, typename U, typename Evaluator, typename precision_type, typename Domain, typename GradientDomain>
  __HOST__ IvyThreadSafePtr_t<typename IvyRegularFunction_2D<T, U, Evaluator, precision_type, Domain, GradientDomain>::grad_t> IvyRegularFunction_2D<T, U, Evaluator, precision_type, Domain, GradientDomain>::gradient(
    IvyThreadSafePtr_t<IvyBaseNode> const& var
  ) const{
    if constexpr (std_ttraits::is_same_v<Domain, tensor_domain_tag>){
      // Tensor domain: eager element-wise binary chain rule. The lazy
      // evaluator_t::gradient(i,x,y)*grad path cannot be used for tensors (it
      // would instantiate IvyMultiply<tensor,tensor> whose gradient is
      // pure-virtual). Instead, for each element i we combine the operand values
      // x[i], y[i] with the recursed operand gradients ∂x[i]/∂var, ∂y[i]/∂var via
      // the op-specific local partials in evaluator_t::dcombine, and pack the
      // result into an IvyTensorEagerFunction. value_t is the (possibly up-cast)
      // result tensor type, so building `result` from this->value() yields the
      // correct shape and element type even for mixed/up-cast operands.
      //
      // A scalar operand (tensor⊗scalar) is supported transparently: its value and
      // its gradient ∂scalar/∂var are read once and broadcast across every element,
      // so differentiation wrt the scalar produces the correctly-broadcast gradient
      // tensor. The unified `read` lambda dispatches on whether the dereferenced
      // pointer is a tensor (per-element, pointer/cell) or a scalar leaf/function
      // (broadcast).
      using VT = value_t;
      using dtype_t = typename VT::dtype_t;
      constexpr std_ivy::IvyMemoryType mem = IvyMemoryHelpers::get_execution_default_memory();
      using XV = unpack_if_function_t<T>;
      using YV = unpack_if_function_t<U>;
      auto xval = make_IvyThreadSafePtr<XV>(mem, nullptr, unpack_function_input<T>::get(*x));
      auto yval = make_IvyThreadSafePtr<YV>(mem, nullptr, unpack_function_input<U>::get(*y));
      auto grad_x = function_gradient<T>::get(*x, var);
      auto grad_y = function_gradient<U>::get(*y, var);
      VT result(this->value());
      IvyTensorDim_t const n = result.num_elements();
      auto read = [](auto const& ptr, IvyTensorDim_t i){
        using P = std_ttraits::remove_cv_t<std_ttraits::remove_reference_t<decltype(*ptr)>>;
        if constexpr (is_tensor_v<P>){
          using elem = typename P::dtype_t;
          if constexpr (is_pointer_v<elem>) return unpack_function_input_reduced<typename elem::element_type>::get(*(*ptr)[i]);
          else return unpack_function_input_reduced<elem>::get((*ptr)[i]);
        }
        else return unpack_function_input_reduced<P>::get(*ptr);
      };
      for (IvyTensorDim_t i = 0; i < n; ++i){
        auto const rv = evaluator_t::dcombine(read(xval, i), read(yval, i), read(grad_x, i), read(grad_y, i));
        if constexpr (is_pointer_v<dtype_t>) result[i] = make_IvyThreadSafePtr<typename dtype_t::element_type>(mem, nullptr, rv);
        else result[i] = rv;
      }
      return make_IvyThreadSafePtr<IvyTensorEagerFunction<VT>>(mem, nullptr, result);
    } else {
      if (var && __STATIC_CAST__(IvyBaseNode const*, this) == var.get())
        return make_unit_function<precision_type, Domain>();
      auto grad_x = function_gradient<T>::get(*x, var);
      auto grad_y = function_gradient<U>::get(*y, var);
      if constexpr (evaluator_is_order_aware_v<evaluator_t>)
        return evaluator_t::combine_gradient(x, y, grad_x, grad_y);
      else
        return evaluator_t::gradient(0, x, y)*grad_x + evaluator_t::gradient(1, x, y)*grad_y;
    }
  }

  // Special 2D case with no gradients
  template<typename T, typename U, typename Evaluator, typename precision_type, typename Domain>
  __HOST__ IvyRegularFunction_2D<T, U, Evaluator, precision_type, Domain, undefined_domain_tag>::IvyRegularFunction_2D(IvyThreadSafePtr_t<T> const& x, IvyThreadSafePtr_t<U> const& y) : base_t(), x(x), y(y){}
  template<typename T, typename U, typename Evaluator, typename precision_type, typename Domain>
  __HOST__ IvyRegularFunction_2D<T, U, Evaluator, precision_type, Domain, undefined_domain_tag>::IvyRegularFunction_2D(IvyRegularFunction_2D<T, U, Evaluator, precision_type, Domain, undefined_domain_tag> const& other) :
    base_t(__DYNAMIC_CAST__(base_t const&, other)),
    x(other.x),
    y(other.y)
  {}
  template<typename T, typename U, typename Evaluator, typename precision_type, typename Domain>
  __HOST__ IvyRegularFunction_2D<T, U, Evaluator, precision_type, Domain, undefined_domain_tag>::IvyRegularFunction_2D(IvyRegularFunction_2D<T, U, Evaluator, precision_type, Domain, undefined_domain_tag>&& other) :
    base_t(__DYNAMIC_CAST__(base_t&&, std_util::move(other))),
    x(std_util::move(other.x)),
    y(std_util::move(other.y))
  {}
  template<typename T, typename U, typename Evaluator, typename precision_type, typename Domain>
  __HOST__ void IvyRegularFunction_2D<T, U, Evaluator, precision_type, Domain, undefined_domain_tag>::eval() const{
    if (this->is_modified()) *(this->output) = evaluator_t::eval(unpack_function_input<T>::get(*x), unpack_function_input<U>::get(*y));
  }
  template<typename T, typename U, typename Evaluator, typename precision_type, typename Domain>
  __HOST__ bool IvyRegularFunction_2D<T, U, Evaluator, precision_type, Domain, undefined_domain_tag>::depends_on(IvyBaseNode const* node) const{
    return (base_t::depends_on(node) || IvyMath::depends_on(x, node) || IvyMath::depends_on(y, node));
  }

}


#endif
