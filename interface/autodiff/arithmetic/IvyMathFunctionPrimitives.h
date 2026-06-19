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
      auto f_prime  = evaluator_t::gradient(dep);           // IvyThreadSafePtr_t<T>
      auto grad_dep = function_gradient<T>::get(*dep, var); // IvyThreadSafePtr_t<T>
      using dtype_t = typename T::dtype_t;
      constexpr std_ivy::IvyMemoryType mem = IvyMemoryHelpers::get_execution_default_memory();
      T result(*f_prime);
      IvyTensorDim_t const n = result.num_elements();
      if constexpr (is_pointer_v<dtype_t>){
        using inner_t = typename dtype_t::element_type;
        for (IvyTensorDim_t i = 0; i < n; ++i){
          auto const fv = unpack_function_input_reduced<inner_t>::get(*(*f_prime)[i]);
          auto const gv = unpack_function_input_reduced<inner_t>::get(*(*grad_dep)[i]);
          result[i] = make_IvyThreadSafePtr<inner_t>(mem, nullptr, fv * gv);
        }
      } else {
        for (IvyTensorDim_t i = 0; i < n; ++i){
          auto const fv = unpack_function_input_reduced<dtype_t>::get((*f_prime)[i]);
          auto const gv = unpack_function_input_reduced<dtype_t>::get((*grad_dep)[i]);
          result[i] = fv * gv;
        }
      }
      return make_IvyThreadSafePtr<IvyTensorEagerFunction<T>>(mem, nullptr, result);
    } else {
      if (var && __STATIC_CAST__(IvyBaseNode const*, this) == var.get())
        return make_unit_function<precision_type, Domain>();
      auto grad_dep = function_gradient<T>::get(*dep, var);
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
    if constexpr (!std_ttraits::is_same_v<Domain, tensor_domain_tag>){
      if (var && __STATIC_CAST__(IvyBaseNode const*, this) == var.get())
        return make_unit_function<precision_type, Domain>();
    }
    auto grad_x = function_gradient<T>::get(*x, var);
    auto grad_y = function_gradient<U>::get(*y, var);
    return evaluator_t::gradient(0, x, y)*grad_x + evaluator_t::gradient(1, x, y)*grad_y;
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
