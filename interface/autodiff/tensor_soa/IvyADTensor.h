#ifndef IVYADTENSOR_H
#define IVYADTENSOR_H

/**
 * @file IvyADTensor.h
 * @brief Structure-of-arrays (SoA) reverse-mode automatic-differentiation tensor.
 *
 * @details
 * `IvyADTensor<T>` is a performance-oriented alternative to the array-of-pointers
 * differentiable tensor (`IvyTensor<IvyVariablePtr_t<T>>`). Instead of storing one
 * heap-allocated autodiff node (plus control block and client list) per element, it
 * stores:
 *   - a single contiguous value buffer, and
 *   - a single contiguous gradient (adjoint) buffer (allocated lazily),
 * with the differentiation graph (the "tape") kept host-side as a small set of parent
 * links and a backward closure per node. Element-wise operations are evaluated by a
 * single fused loop over the contiguous buffers (OpenMP-parallel on the host), and the
 * reverse-mode `backward()` pass propagates adjoints with the same fused loops.
 *
 * Design notes:
 *   - The tape topology lives on the host (like the existing `IvyFunction` graph): nodes
 *     are owned through `std::shared_ptr` and a node's backward closure is a `std::function`.
 *   - Numeric buffers use the Ivy memory-domain-aware `unique_ptr` so the design can be
 *     extended to device-resident buffers later (plan Phase 5); for now they live in the
 *     host execution-default memory and ops run host-side, exactly like the existing
 *     tensor eval path.
 *   - Backward closures capture the owning node and its parents by raw pointer; ownership
 *     and reverse traversal are provided by `parents_` (`shared_ptr`), and no parent holds
 *     a strong reference back to its children, so no ownership cycle is formed.
 */

#include <vector>
#include <functional>
#include <memory>

#include "config/IvyCompilerConfig.h"
#include "config/IvyOpenMPConfig.h"
#include "std_ivy/IvyCmath.h"
#include "std_ivy/IvyMemory.h"
#include "autodiff/IvyBaseMathTypes.h"
#include "autodiff/basic_nodes/IvyTensorShape.h"
#include "IvyPrintout.h"


namespace IvyAD{
  using IvyMath::IvyTensorShape;
  using IvyMath::IvyTensorDim_t;

  template<typename T> class IvyADTensor;
  /** @brief Shared, host-side owning handle to an autodiff-tensor tape node. */
  template<typename T> using IvyADTensorPtr = std::shared_ptr<IvyADTensor<T>>;

  /**
   * @brief Run @p fcn(i) for i in [0,n), OpenMP-parallel above the CPU threshold.
   * @details Mirrors the host execution policy used by the existing tensor eval functions.
   */
  template<typename Fcn>
  __INLINE_FCN_RELAXED__ __HOST__ void ivy_ad_for(IvyTensorDim_t n, Fcn fcn){
#if defined(OPENMP_ENABLED)
    if (n >= NUM_CPU_THREADS_THRESHOLD){
      #pragma omp parallel for schedule(static)
      for (IvyTensorDim_t i = 0; i < n; ++i) fcn(i);
    }
    else
#endif
    {
      for (IvyTensorDim_t i = 0; i < n; ++i) fcn(i);
    }
  }

  /**
   * @brief SoA reverse-mode autodiff tensor node.
   * @tparam T Arithmetic scalar precision (e.g. @c double).
   */
  template<typename T> class IvyADTensor{
  public:
    using dtype_t = T;
    using buffer_t = std_mem::unique_ptr<T>;

  protected:
    IvyTensorShape shape_;                    //!< Logical shape (dimensions + element map).
    buffer_t value_;                          //!< Contiguous forward-value buffer (num_elements()).
    buffer_t grad_;                           //!< Contiguous adjoint buffer (lazily allocated, zeroed).
    bool requires_grad_;                      //!< Whether this node participates in differentiation.
    std::vector<IvyADTensorPtr<T>> parents_;  //!< Inputs this node was computed from (ownership + traversal).
    std::function<void()> backward_fcn_;      //!< Pushes this node's adjoint into its parents' adjoints.

    /** @brief Default memory domain for the numeric buffers (host execution default). */
    static __INLINE_FCN_RELAXED__ __HOST__ std_ivy::IvyMemoryType mem(){ return IvyMemoryHelpers::get_execution_default_memory(); }

  public:
    /** @brief Construct a leaf tensor of the given shape, filled with @p fill. */
    __HOST__ IvyADTensor(IvyTensorShape const& shape, T const& fill = T(0), bool requires_grad = false) :
      shape_(shape),
      value_(std_mem::make_unique<T>(shape.num_elements(), mem(), nullptr, fill)),
      requires_grad_(requires_grad)
    {}

    /** @brief Construct a tensor of the given shape, copying values from a host buffer. */
    __HOST__ IvyADTensor(IvyTensorShape const& shape, T const* src, bool requires_grad = false) :
      shape_(shape),
      value_(std_mem::make_unique<T>(shape.num_elements(), mem(), nullptr, T(0))),
      requires_grad_(requires_grad)
    {
      T* d = value_.get();
      IvyTensorDim_t const n = shape_.num_elements();
      for (IvyTensorDim_t i = 0; i < n; ++i) d[i] = src[i];
    }

    IvyADTensor(IvyADTensor const&) = delete;            //!< Nodes are identity objects; share via IvyADTensorPtr.
    IvyADTensor& operator=(IvyADTensor const&) = delete;
    __HOST__ ~IvyADTensor() = default;

    // --- Shape / metadata ---------------------------------------------------
    __HOST__ IvyTensorShape const& shape() const{ return shape_; }
    __HOST__ IvyTensorDim_t num_elements() const{ return shape_.num_elements(); }
    __HOST__ bool requires_grad() const{ return requires_grad_; }
    __HOST__ void set_requires_grad(bool v){ requires_grad_ = v; }

    // --- Raw buffer access (host-addressable) -------------------------------
    __HOST__ T* value_data(){ return value_.get(); }
    __HOST__ T const* value_data() const{ return value_.get(); }
    __HOST__ T value_at(IvyTensorDim_t i) const{ return value_.get()[i]; }

    /** @brief Lazily allocate (zeroed) and return the adjoint buffer. */
    __HOST__ T* grad_data(){
      if (!grad_) grad_ = std_mem::make_unique<T>(shape_.num_elements(), mem(), nullptr, T(0));
      return grad_.get();
    }
    __HOST__ bool has_grad() const{ return static_cast<bool>(grad_); }
    __HOST__ T grad_at(IvyTensorDim_t i) const{ return grad_ ? grad_.get()[i] : T(0); }

    /** @brief Set this node's parents and its backward closure (used by the op factories). */
    __HOST__ void set_tape(std::vector<IvyADTensorPtr<T>> parents, std::function<void()> backward_fcn){
      parents_ = std::move(parents);
      backward_fcn_ = std::move(backward_fcn);
    }
    __HOST__ std::vector<IvyADTensorPtr<T>> const& parents() const{ return parents_; }
    __HOST__ bool has_backward() const{ return static_cast<bool>(backward_fcn_); }
    __HOST__ void run_backward(){ if (backward_fcn_) backward_fcn_(); }

    /** @brief Reset this node's adjoint buffer to zero (does not touch parents). */
    __HOST__ void zero_grad(){
      if (!grad_) return;
      T* g = grad_.get();
      IvyTensorDim_t const n = shape_.num_elements();
      for (IvyTensorDim_t i = 0; i < n; ++i) g[i] = T(0);
    }
  };

  // --- Tape factory -----------------------------------------------------------
  /** @brief Allocate a fresh tape node sharing @p shape, computed (not a leaf). */
  template<typename T>
  __HOST__ IvyADTensorPtr<T> make_ADTensor(IvyTensorShape const& shape, bool requires_grad){
    return std::make_shared<IvyADTensor<T>>(shape, T(0), requires_grad);
  }
  /** @brief Build a leaf tensor (a differentiation input) filled with @p fill. */
  template<typename T>
  __HOST__ IvyADTensorPtr<T> ADTensor(IvyTensorShape const& shape, T const& fill = T(0), bool requires_grad = true){
    return std::make_shared<IvyADTensor<T>>(shape, fill, requires_grad);
  }
  /** @brief Build a leaf tensor (a differentiation input) from a host value buffer. */
  template<typename T>
  __HOST__ IvyADTensorPtr<T> ADTensor(IvyTensorShape const& shape, T const* src, bool requires_grad = true){
    return std::make_shared<IvyADTensor<T>>(shape, src, requires_grad);
  }

  // --- Reverse-mode backward --------------------------------------------------
  /**
   * @brief Collect the tape in reverse-topological order (root first), via post-order DFS.
   */
  template<typename T>
  __HOST__ void ivy_ad_topo(IvyADTensor<T>* node, std::vector<IvyADTensor<T>*>& order, std::vector<IvyADTensor<T>*>& visited){
    for (auto const& v : visited) if (v == node) return;
    visited.push_back(node);
    for (auto const& p : node->parents()) ivy_ad_topo(p.get(), order, visited);
    order.push_back(node); // post-order: parents appended before node
  }

  /**
   * @brief Run reverse-mode differentiation from @p root.
   * @param root The output node to differentiate.
   * @param seed Optional seed for @p root's adjoint (defaults to all-ones, i.e. d(sum(root))).
   *
   * After the call, each leaf with @c requires_grad() has its adjoint in @c grad_data().
   */
  template<typename T>
  __HOST__ void backward(IvyADTensorPtr<T> const& root, T const* seed = nullptr){
    std::vector<IvyADTensor<T>*> order, visited;
    ivy_ad_topo(root.get(), order, visited);

    // Seed the root adjoint (all-ones by default → gradient of the sum of the outputs).
    T* rg = root->grad_data();
    IvyTensorDim_t const nr = root->num_elements();
    if (seed) ivy_ad_for(nr, [rg, seed](IvyTensorDim_t i){ rg[i] = seed[i]; });
    else      ivy_ad_for(nr, [rg](IvyTensorDim_t i){ rg[i] = T(1); });

    // Reverse topological order: node before its parents.
    for (auto it = order.rbegin(); it != order.rend(); ++it) (*it)->run_backward();
  }

  /** @brief Zero the adjoint of every node reachable from @p root. */
  template<typename T>
  __HOST__ void zero_grad(IvyADTensorPtr<T> const& root){
    std::vector<IvyADTensor<T>*> order, visited;
    ivy_ad_topo(root.get(), order, visited);
    for (auto* n : order) n->zero_grad();
  }
}

#endif
