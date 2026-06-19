/**
 * @file IvyUnifiedPtr.h
 * @brief IvyUnifiedPtr method definitions and unified pointer lifecycle operations.
 */
#ifndef IVYUNIFIEDPTR_H
#define IVYUNIFIEDPTR_H


#include "std_ivy/IvyCassert.h"
#include "std_ivy/IvyCstdio.h"
#include "std_ivy/memory/IvyUnifiedPtr.hh"

// Reference counting uses atomic_ref over the heap-allocated counter so concurrent
// copies/destructions are race-free. IvyAtomic exposes the std_atomic alias, which selects
// cuda::std (CUDA) or std (host), giving consistent host/device semantics.
#include "std_ivy/IvyAtomic.h"
// std_atomic::atomic_flag backs the host-only spinlock pool that serializes
// non-addressable, cross-execution-space refcount updates; it is always host code.
// <thread> provides std::this_thread::yield() for spin-wait backoff in that pool.
#include <thread>


namespace std_ivy{
#if DEVICE_CODE == DEVICE_CODE_HOST
  namespace detail{
    /**
     * @brief Host-side striped spinlock pool guarding non-addressable refcount updates.
     *
     * When a control block lives in memory that is not directly addressable from the host
     * (e.g. device memory reached from a host thread), the counter cannot be mutated with a
     * single hardware atomic: it must be staged in through a transfer, modified, and written
     * back. That read-modify-write is not atomic on its own, so concurrent host-thread
     * ownership updates could lose increments/decrements. We serialize those sequences here
     * with a small pool of spinlocks keyed by the control-block address, which makes the
     * cross-execution-space refcount update atomic with respect to other host threads (the
     * only threads that can take this path; device code addresses its own memory directly and
     * uses the hardware-atomic branch). Striping keeps independent control blocks contention-free.
     *
     * This pool is host-only (it is compiled solely for host execution), and it uses the
     * backend-selecting std_atomic alias for atomic_flag so the namespace usage stays uniform.
     */
    __INLINE_FCN_RELAXED__ std_atomic::atomic_flag& nonaddressable_refcount_lock(void const* key){
      static constexpr unsigned int n_locks = 64;
      static std_atomic::atomic_flag locks[n_locks];
      auto const idx = (__REINTERPRET_CAST__(IvyTypes::size_t, key) / sizeof(void*)) % n_locks;
      return locks[idx];
    }
  }
#endif

  template<typename T, IvyPointerType IPT> __HOST_DEVICE__ IvyUnifiedPtr<T, IPT>::IvyUnifiedPtr(IvyGPUStream* stream) :
    ptr_(nullptr),
    cblock_(nullptr),
    stream_(stream),
    exec_mem_type_(IvyMemoryHelpers::get_execution_default_memory()),
    progenitor_mem_type_(IvyMemoryHelpers::get_execution_default_memory())
  {}
  template<typename T, IvyPointerType IPT> __HOST_DEVICE__ IvyUnifiedPtr<T, IPT>::IvyUnifiedPtr(std_cstddef::nullptr_t, IvyGPUStream* stream) :
    ptr_(nullptr),
    cblock_(nullptr),
    stream_(stream),
    exec_mem_type_(IvyMemoryHelpers::get_execution_default_memory()),
    progenitor_mem_type_(IvyMemoryHelpers::get_execution_default_memory())
  {}
  template<typename T, IvyPointerType IPT>
  __HOST_DEVICE__ IvyUnifiedPtr<T, IPT>::IvyUnifiedPtr(T* ptr, IvyMemoryType mem_type, IvyGPUStream* stream) :
    ptr_(ptr),
    cblock_(nullptr),
    stream_(stream),
    exec_mem_type_(IvyMemoryHelpers::get_execution_default_memory()),
    progenitor_mem_type_(IvyMemoryHelpers::get_execution_default_memory())
  {
    if (ptr_) this->init_members(mem_type, 1, 1);
  }
  template<typename T, IvyPointerType IPT>
  __HOST_DEVICE__ IvyUnifiedPtr<T, IPT>::IvyUnifiedPtr(T* ptr, size_type n, IvyMemoryType mem_type, IvyGPUStream* stream) :
    ptr_(ptr),
    cblock_(nullptr),
    stream_(stream),
    exec_mem_type_(IvyMemoryHelpers::get_execution_default_memory()),
    progenitor_mem_type_(IvyMemoryHelpers::get_execution_default_memory())
  {
    if (ptr_) this->init_members(mem_type, n, n);
  }
  template<typename T, IvyPointerType IPT>
  __HOST_DEVICE__ IvyUnifiedPtr<T, IPT>::IvyUnifiedPtr(T* ptr, size_type n_size, size_type n_capacity, IvyMemoryType mem_type, IvyGPUStream* stream) :
    ptr_(ptr),
    cblock_(nullptr),
    stream_(stream),
    exec_mem_type_(IvyMemoryHelpers::get_execution_default_memory()),
    progenitor_mem_type_(IvyMemoryHelpers::get_execution_default_memory())
  {
    if (ptr_) this->init_members(mem_type, n_size, n_capacity);
  }
  template<typename T, IvyPointerType IPT>
  template<typename U>
  __HOST_DEVICE__ IvyUnifiedPtr<T, IPT>::IvyUnifiedPtr(U* ptr, IvyMemoryType mem_type, IvyGPUStream* stream) :
    cblock_(nullptr),
    stream_(stream),
    exec_mem_type_(IvyMemoryHelpers::get_execution_default_memory()),
    progenitor_mem_type_(IvyMemoryHelpers::get_execution_default_memory())
  {
    ptr_ = __DYNAMIC_CAST__(pointer, ptr);
    if (ptr_) this->init_members(mem_type, 1, 1);
  }
  template<typename T, IvyPointerType IPT>
  template<typename U>
  __HOST_DEVICE__ IvyUnifiedPtr<T, IPT>::IvyUnifiedPtr(U* ptr, size_type n, IvyMemoryType mem_type, IvyGPUStream* stream) :
    cblock_(nullptr),
    stream_(stream),
    exec_mem_type_(IvyMemoryHelpers::get_execution_default_memory()),
    progenitor_mem_type_(IvyMemoryHelpers::get_execution_default_memory())
  {
    ptr_ = __DYNAMIC_CAST__(pointer, ptr);
    if (ptr_) this->init_members(mem_type, n, n);
  }
  template<typename T, IvyPointerType IPT>
  template<typename U, IvyPointerType IPU, ENABLE_IF_BOOL_IMPL((IPU==IPT || IPU==IvyPointerType::unique) && (IS_BASE_OF(T, U) || IS_BASE_OF(U, T)))>
  __HOST_DEVICE__ IvyUnifiedPtr<T, IPT>::IvyUnifiedPtr(IvyUnifiedPtr<U, IPU> const& other) :
    progenitor_mem_type_(other.get_progenitor_memory_type())
  {
    ptr_ = __DYNAMIC_CAST__(pointer, other.get());
    if (!ptr_ && other.get()){
      __PRINT_ERROR__("IvyUnifiedPtr copy constructor failed: Incompatible types\n");
      assert(false);
    }
    else{
      cblock_ = other.control_block();
      stream_ = other.gpu_stream();
      exec_mem_type_ = other.get_exec_memory_type();
      if (cblock_) this->inc_dec_counter(true);
    }
    // We should reset the other pointer if it is unique and its progenitor memory type is the same as that of the current context.
    if constexpr (IPU==IvyPointerType::unique){
      if (other.check_write_access()) __CONST_CAST__(__ENCAPSULATE__(IvyUnifiedPtr<U, IPU>&), other).reset();
    }
  }
  template<typename T, IvyPointerType IPT> __HOST_DEVICE__ IvyUnifiedPtr<T, IPT>::IvyUnifiedPtr(IvyUnifiedPtr<T, IPT> const& other) :
    ptr_(other.ptr_),
    cblock_(other.cblock_),
    stream_(other.stream_),
    exec_mem_type_(other.exec_mem_type_),
    progenitor_mem_type_(other.progenitor_mem_type_)
  {
    if (cblock_) this->inc_dec_counter(true);
    // We should reset the other pointer if it is unique and its progenitor memory type is the same as that of the current context.
    if constexpr (IPT==IvyPointerType::unique){
      if (other.check_write_access()) __CONST_CAST__(__ENCAPSULATE__(IvyUnifiedPtr<T, IPT>&), other).reset();
    }
  }
  template<typename T, IvyPointerType IPT> template<typename U, IvyPointerType IPU, ENABLE_IF_BOOL_IMPL((IPU==IPT || IPU==IvyPointerType::unique) && (IS_BASE_OF(T, U) || IS_BASE_OF(U, T)))>
  __HOST_DEVICE__ IvyUnifiedPtr<T, IPT>::IvyUnifiedPtr(IvyUnifiedPtr<U, IPU>&& other) :
    ptr_(std_util::move(other.get())),
    cblock_(std_util::move(other.control_block())),
    stream_(std_util::move(other.gpu_stream())),
    exec_mem_type_(std_util::move(other.get_exec_memory_type())),
    progenitor_mem_type_(std_util::move(other.get_progenitor_memory_type()))
  {
    other.check_write_access_or_die();
    IvySecrets::dump_helper::dump(other);
  }
  template<typename T, IvyPointerType IPT> __HOST_DEVICE__ IvyUnifiedPtr<T, IPT>::IvyUnifiedPtr(IvyUnifiedPtr&& other) :
    ptr_(std_util::move(other.ptr_)),
    cblock_(std_util::move(other.cblock_)),
    stream_(std_util::move(other.gpu_stream())),
    exec_mem_type_(std_util::move(other.exec_mem_type_)),
    progenitor_mem_type_(std_util::move(other.progenitor_mem_type_))
  {
    other.check_write_access_or_die();
    other.dump();
  }
  template<typename T, IvyPointerType IPT> __HOST_DEVICE__ IvyUnifiedPtr<T, IPT>::~IvyUnifiedPtr(){ this->reset(); }

  template<typename T, IvyPointerType IPT>
  template<typename U, IvyPointerType IPU, ENABLE_IF_BOOL_IMPL((IPU==IPT || IPU==IvyPointerType::unique) && (IS_BASE_OF(T, U) || IS_BASE_OF(U, T)))>
  __HOST_DEVICE__ IvyUnifiedPtr<T, IPT>& IvyUnifiedPtr<T, IPT>::operator=(IvyUnifiedPtr<U, IPU> const& other){
    if (*this != other){
      this->release();
      progenitor_mem_type_ = other.get_progenitor_memory_type();
      exec_mem_type_ = other.get_exec_memory_type();
      ptr_ = __DYNAMIC_CAST__(pointer, other.get());
      if (!ptr_ && other.get()){
        __PRINT_ERROR__("IvyUnifiedPtr copy assignment failed: Incompatible types\n");
        assert(false);
      }
      cblock_ = other.control_block();
      stream_ = other.gpu_stream();
      if (cblock_) this->inc_dec_counter(true);
      // We should reset the other pointer if it is unique and its progenitor memory type is the same as that of the current context.
      if constexpr (IPU==IvyPointerType::unique){
        if (other.check_write_access()) __CONST_CAST__(__ENCAPSULATE__(IvyUnifiedPtr<U, IPU>&), other).reset();
      }
    }
    return *this;
  }
  template<typename T, IvyPointerType IPT> __HOST_DEVICE__ IvyUnifiedPtr<T, IPT>& IvyUnifiedPtr<T, IPT>::operator=(IvyUnifiedPtr const& other){
    if (*this != other){
      this->release();
      progenitor_mem_type_ = other.progenitor_mem_type_;
      exec_mem_type_ = other.exec_mem_type_;
      ptr_ = other.ptr_;
      cblock_ = other.cblock_;
      stream_ = other.gpu_stream();
      if (cblock_) this->inc_dec_counter(true);
      // We should reset the other pointer if it is unique and its progenitor memory type is the same as that of the current context.
      if constexpr (IPT==IvyPointerType::unique){
        if (other.check_write_access()) __CONST_CAST__(__ENCAPSULATE__(IvyUnifiedPtr<T, IPT>&), other).reset();
      }
    }
    return *this;
  }
  template<typename T, IvyPointerType IPT>
  template<typename U, IvyPointerType IPU, ENABLE_IF_BOOL_IMPL((IPU==IPT || IPU==IvyPointerType::unique) && (IS_BASE_OF(T, U) || IS_BASE_OF(U, T)))>
  __HOST_DEVICE__ IvyUnifiedPtr<T, IPT>& IvyUnifiedPtr<T, IPT>::operator=(IvyUnifiedPtr<U, IPU>&& other){
    other.check_write_access_or_die();
    this->release();
    ptr_ = std_util::move(other.get());
    cblock_ = std_util::move(other.control_block());
    stream_ = std_util::move(other.gpu_stream());
    exec_mem_type_ = std_util::move(other.get_exec_memory_type());
    progenitor_mem_type_ = std_util::move(other.get_progenitor_memory_type());
    IvySecrets::dump_helper::dump(other);
    return *this;
  }
  template<typename T, IvyPointerType IPT> __HOST_DEVICE__ IvyUnifiedPtr<T, IPT>& IvyUnifiedPtr<T, IPT>::operator=(IvyUnifiedPtr&& other){
    other.check_write_access_or_die();
    this->release();
    ptr_ = std_util::move(other.ptr_);
    cblock_ = std_util::move(other.cblock_);
    stream_ = std_util::move(other.stream_);
    exec_mem_type_ = std_util::move(other.exec_mem_type_);
    progenitor_mem_type_ = std_util::move(other.progenitor_mem_type_);
    other.dump();
    return *this;
  }
  template<typename T, IvyPointerType IPT> __HOST_DEVICE__ IvyUnifiedPtr<T, IPT>& IvyUnifiedPtr<T, IPT>::operator=(std_cstddef::nullptr_t){ this->reset(nullptr); return *this; }

  template<typename T, IvyPointerType IPT> __HOST_DEVICE__ void IvyUnifiedPtr<T, IPT>::init_members(IvyMemoryType mem_type, size_type n_size, size_type n_capacity){
    assert(n_size<=n_capacity);
    this->check_write_access_or_die();
    operate_with_GPU_stream_from_pointer(
      stream_, ref_stream,
      __ENCAPSULATE__(
        cblock_ = control_block_allocator_traits::allocate(1, exec_mem_type_, ref_stream);
      )
    );
    if (!cblock_){
      __PRINT_ERROR__("IvyUnifiedPtr::init_members: Failed to allocate the control block at %p.\n", this);
      assert(false);
      return;
    }
    control_block_type cb{ __STATIC_CAST__(counter_type, 1), n_size, n_capacity, mem_type };
    this->store_control_block(cb);
  }
  template<typename T, IvyPointerType IPT> __HOST_DEVICE__ void IvyUnifiedPtr<T, IPT>::release(){
    if (cblock_){
      // Atomically decrement first. The thread that observes the transition to zero
      // (i.e. sees a previous value of 1) is the sole owner responsible for cleanup.
      // This avoids the check-then-act race of reading use_count() and then freeing.
      auto const prev_count = this->inc_dec_counter(false);
      if (prev_count==1){
        // We were the only owner, so we can clean up. We must have write access first.
        if (this->check_write_access()){
          // Sole owner: snapshot the whole control block once (no concurrent refcount activity).
          control_block_type const cb = this->load_control_block();
          operate_with_GPU_stream_from_pointer(
            stream_, ref_stream,
            __ENCAPSULATE__(
              if (ptr_){
                element_allocator_traits::destruct(ptr_, cb.size, cb.mem_type, ref_stream);
                element_allocator_traits::deallocate(ptr_, cb.capacity, cb.mem_type, ref_stream);
              }
              control_block_allocator_traits::destroy(cblock_, 1, exec_mem_type_, ref_stream);
            )
          );
        }
        else{
          __PRINT_ERROR__("IvyUnifiedPtr::release: No write access to clean object at %p.\n", this);
          assert(false);
        }
      }
    }
  }
  template<typename T, IvyPointerType IPT> __HOST_DEVICE__ void IvyUnifiedPtr<T, IPT>::dump(){
    cblock_ = nullptr;
    ptr_ = nullptr;
    stream_ = nullptr;
    progenitor_mem_type_ = exec_mem_type_ = IvyMemoryHelpers::get_execution_default_memory();
  }

  template<typename T, IvyPointerType IPT> __HOST_DEVICE__ bool IvyUnifiedPtr<T, IPT>::control_block_is_addressable() const{
    return IvyMemoryHelpers::is_addressable_from_execution(exec_mem_type_);
  }
  template<typename T, IvyPointerType IPT> template<typename U> __HOST_DEVICE__ U IvyUnifiedPtr<T, IPT>::read_meta_field(U* p_field) const{
    constexpr IvyMemoryType def_mem_type = IvyMemoryHelpers::get_execution_default_memory();
    // Directly addressable: read in place with no transfer at all.
    if (this->control_block_is_addressable()) return *p_field;
    // Otherwise transfer the single field into an on-stack temporary (no heap scratch buffer).
    U ret{};
    U* p_ret = &ret;
    bool transfer_success = false;
    operate_with_GPU_stream_from_pointer(
      stream_, ref_stream,
      __ENCAPSULATE__(
        transfer_success = std_ivy::allocator_traits<std_ivy::allocator<U>>::transfer(p_ret, p_field, 1, def_mem_type, exec_mem_type_, ref_stream);
      )
    );
    if (!transfer_success){
      __PRINT_ERROR__("IvyUnifiedPtr::read_meta_field: Failed to transfer the metadata field at %p.\n", p_field);
      assert(false);
    }
    return ret;
  }
  template<typename T, IvyPointerType IPT> template<typename U> __HOST_DEVICE__ void IvyUnifiedPtr<T, IPT>::write_meta_field(U* p_field, U const& val){
    constexpr IvyMemoryType def_mem_type = IvyMemoryHelpers::get_execution_default_memory();
    if (this->control_block_is_addressable()){ *p_field = val; return; }
    U tmp = val;
    U* p_tmp = &tmp;
    bool transfer_success = false;
    operate_with_GPU_stream_from_pointer(
      stream_, ref_stream,
      __ENCAPSULATE__(
        transfer_success = std_ivy::allocator_traits<std_ivy::allocator<U>>::transfer(p_field, p_tmp, 1, exec_mem_type_, def_mem_type, ref_stream);
      )
    );
    if (!transfer_success){
      __PRINT_ERROR__("IvyUnifiedPtr::write_meta_field: Failed to transfer the metadata field at %p.\n", p_field);
      assert(false);
    }
  }
  template<typename T, IvyPointerType IPT> __HOST_DEVICE__ IvyUnifiedPtr<T, IPT>::control_block_type IvyUnifiedPtr<T, IPT>::load_control_block() const{
    constexpr IvyMemoryType def_mem_type = IvyMemoryHelpers::get_execution_default_memory();
    control_block_type cb{ __STATIC_CAST__(counter_type, 0), __STATIC_CAST__(size_type, 0), __STATIC_CAST__(size_type, 0), def_mem_type };
    if (!cblock_) return cb;
    if (this->control_block_is_addressable()) return *cblock_;
    control_block_type* p_cb = &cb;
    bool transfer_success = false;
    operate_with_GPU_stream_from_pointer(
      stream_, ref_stream,
      __ENCAPSULATE__(
        transfer_success = control_block_allocator_traits::transfer(p_cb, cblock_, 1, def_mem_type, exec_mem_type_, ref_stream);
      )
    );
    if (!transfer_success){
      __PRINT_ERROR__("IvyUnifiedPtr::load_control_block: Failed to transfer the control block at %p.\n", cblock_);
      assert(false);
    }
    return cb;
  }
  template<typename T, IvyPointerType IPT> __HOST_DEVICE__ void IvyUnifiedPtr<T, IPT>::store_control_block(control_block_type const& cb){
    if (!cblock_) return;
    constexpr IvyMemoryType def_mem_type = IvyMemoryHelpers::get_execution_default_memory();
    if (this->control_block_is_addressable()){ *cblock_ = cb; return; }
    control_block_type tmp = cb;
    control_block_type* p_tmp = &tmp;
    bool transfer_success = false;
    operate_with_GPU_stream_from_pointer(
      stream_, ref_stream,
      __ENCAPSULATE__(
        transfer_success = control_block_allocator_traits::transfer(cblock_, p_tmp, 1, exec_mem_type_, def_mem_type, ref_stream);
      )
    );
    if (!transfer_success){
      __PRINT_ERROR__("IvyUnifiedPtr::store_control_block: Failed to transfer the control block at %p.\n", cblock_);
      assert(false);
    }
  }

  template<typename T, IvyPointerType IPT> __HOST_DEVICE__ IvyMemoryType const& IvyUnifiedPtr<T, IPT>::get_progenitor_memory_type() const __NOEXCEPT__{ return this->progenitor_mem_type_; }
  template<typename T, IvyPointerType IPT> __HOST_DEVICE__ IvyMemoryType const& IvyUnifiedPtr<T, IPT>::get_exec_memory_type() const __NOEXCEPT__{ return this->exec_mem_type_; }
  template<typename T, IvyPointerType IPT> __HOST_DEVICE__ IvyMemoryType* IvyUnifiedPtr<T, IPT>::get_memory_type_ptr() const __NOEXCEPT__{ return (this->cblock_ ? &this->cblock_->mem_type : nullptr); }
  template<typename T, IvyPointerType IPT> __HOST_DEVICE__ IvyGPUStream* IvyUnifiedPtr<T, IPT>::gpu_stream() const __NOEXCEPT__{ return this->stream_; }
  template<typename T, IvyPointerType IPT> __HOST_DEVICE__ IvyUnifiedPtr<T, IPT>::size_type* IvyUnifiedPtr<T, IPT>::size_ptr() const __NOEXCEPT__{ return (this->cblock_ ? &this->cblock_->size : nullptr); }
  template<typename T, IvyPointerType IPT> __HOST_DEVICE__ IvyUnifiedPtr<T, IPT>::size_type* IvyUnifiedPtr<T, IPT>::capacity_ptr() const __NOEXCEPT__{ return (this->cblock_ ? &this->cblock_->capacity : nullptr); }
  template<typename T, IvyPointerType IPT> __HOST_DEVICE__ IvyUnifiedPtr<T, IPT>::counter_type* IvyUnifiedPtr<T, IPT>::counter() const __NOEXCEPT__{ return (this->cblock_ ? &this->cblock_->ref_count : nullptr); }
  template<typename T, IvyPointerType IPT> __HOST_DEVICE__ IvyUnifiedPtr<T, IPT>::control_block_type* IvyUnifiedPtr<T, IPT>::control_block() const __NOEXCEPT__{ return this->cblock_; }
  template<typename T, IvyPointerType IPT> __HOST_DEVICE__ IvyUnifiedPtr<T, IPT>::pointer IvyUnifiedPtr<T, IPT>::get() const __NOEXCEPT__{ return this->ptr_; }

  template<typename T, IvyPointerType IPT> __HOST_DEVICE__ IvyMemoryType& IvyUnifiedPtr<T, IPT>::get_progenitor_memory_type() __NOEXCEPT__{ return this->progenitor_mem_type_; }
  template<typename T, IvyPointerType IPT> __HOST_DEVICE__ IvyMemoryType& IvyUnifiedPtr<T, IPT>::get_exec_memory_type() __NOEXCEPT__{ return this->exec_mem_type_; }
  template<typename T, IvyPointerType IPT> __HOST_DEVICE__ IvyGPUStream*& IvyUnifiedPtr<T, IPT>::gpu_stream() __NOEXCEPT__{ return this->stream_; }
  template<typename T, IvyPointerType IPT> __HOST_DEVICE__ IvyUnifiedPtr<T, IPT>::control_block_type*& IvyUnifiedPtr<T, IPT>::control_block() __NOEXCEPT__{ return this->cblock_; }
  template<typename T, IvyPointerType IPT> __HOST_DEVICE__ IvyUnifiedPtr<T, IPT>::pointer& IvyUnifiedPtr<T, IPT>::get() __NOEXCEPT__{ return this->ptr_; }

  template<typename T, IvyPointerType IPT> __HOST_DEVICE__ IvyUnifiedPtr<T, IPT>::reference IvyUnifiedPtr<T, IPT>::operator*() const __NOEXCEPT__{ return *(this->ptr_); }
  template<typename T, IvyPointerType IPT>
  __HOST_DEVICE__ IvyUnifiedPtr<T, IPT>::reference IvyUnifiedPtr<T, IPT>::operator[](size_type k) const{
    if (k >= this->size()){
      __PRINT_ERROR__("IvyUnifiedPtr::operator[] failed: Index out of bounds.\n");
      assert(false);
    }
    return *(this->ptr_ + k);
  }
  template<typename T, IvyPointerType IPT> __HOST_DEVICE__ IvyUnifiedPtr<T, IPT>::pointer IvyUnifiedPtr<T, IPT>::operator->() const __NOEXCEPT__{ return this->ptr_; }

  template<typename T, IvyPointerType IPT> __HOST_DEVICE__ bool IvyUnifiedPtr<T, IPT>::empty() const __NOEXCEPT__{ return this->size()==0; }

  template<typename T, IvyPointerType IPT> __HOST_DEVICE__ IvyUnifiedPtr<T, IPT>::size_type IvyUnifiedPtr<T, IPT>::size() const __NOEXCEPT__{
    if (!cblock_) return 0;
    return this->read_meta_field(&cblock_->size);
  }
  template<typename T, IvyPointerType IPT> __HOST_DEVICE__ IvyUnifiedPtr<T, IPT>::size_type IvyUnifiedPtr<T, IPT>::capacity() const __NOEXCEPT__{
    if (!cblock_) return 0;
    return this->read_meta_field(&cblock_->capacity);
  }
  template<typename T, IvyPointerType IPT> __HOST_DEVICE__ IvyMemoryType IvyUnifiedPtr<T, IPT>::get_memory_type() const __NOEXCEPT__{
    if (!cblock_) return IvyMemoryHelpers::get_execution_default_memory();
    return this->read_meta_field(&cblock_->mem_type);
  }

  template<typename T, IvyPointerType IPT> __HOST_DEVICE__ void IvyUnifiedPtr<T, IPT>::reset(){ this->release(); this->dump(); }
  template<typename T, IvyPointerType IPT> __HOST_DEVICE__ void IvyUnifiedPtr<T, IPT>::reset(std_cstddef::nullptr_t){ this->release(); this->dump(); }
  template<typename T, IvyPointerType IPT> template<typename U> __HOST_DEVICE__ void IvyUnifiedPtr<T, IPT>::reset(U* ptr, size_type n_size, size_type n_capacity, IvyMemoryType mem_type, IvyGPUStream* stream){
    bool const is_same = (ptr_ == ptr);
    if (!is_same){
      this->reset();
      stream_ = stream;
      ptr_ = __DYNAMIC_CAST__(pointer, ptr);
      if (!ptr_ && ptr){
        __PRINT_ERROR__("IvyUnifiedPtr::reset failed: Incompatible types\n");
        assert(false);
      }
      if (ptr_) this->init_members(mem_type, n_size, n_capacity);
    }
    else{
      if (stream && this->check_write_access()) stream_ = stream;
      if (cblock_ && this->get_memory_type() != mem_type){
        __PRINT_ERROR__("IvyUnifiedPtr::reset failed: Incompatible mem_type flags.\n");
        assert(false);
      }
    }
  }
  template<typename T, IvyPointerType IPT> template<typename U>
  __HOST_DEVICE__ void IvyUnifiedPtr<T, IPT>::reset(U* ptr, size_type n, IvyMemoryType mem_type, IvyGPUStream* stream){
    this->reset(ptr, n, n, mem_type, stream);
  }
  template<typename T, IvyPointerType IPT> template<typename U>
  __HOST_DEVICE__ void IvyUnifiedPtr<T, IPT>::reset(U* ptr, IvyMemoryType mem_type, IvyGPUStream* stream){
    this->reset(ptr, __STATIC_CAST__(size_type, 1), mem_type, stream);
  }
  template<typename T, IvyPointerType IPT> __HOST_DEVICE__ void IvyUnifiedPtr<T, IPT>::reset(T* ptr, size_type n_size, size_type n_capacity, IvyMemoryType mem_type, IvyGPUStream* stream){
    bool const is_same = (ptr_ == ptr);
    if (!is_same){
      this->reset();
      stream_ = stream;
      ptr_ = ptr;
      if (ptr_) this->init_members(mem_type, n_size, n_capacity);
    }
    else{
      if (stream && this->check_write_access()) stream_ = stream;
      if (cblock_ && this->get_memory_type() != mem_type){
        __PRINT_ERROR__("IvyUnifiedPtr::reset() failed: Incompatible mem_type flags.\n");
        assert(false);
      }
    }
  }
  template<typename T, IvyPointerType IPT>
  __HOST_DEVICE__ void IvyUnifiedPtr<T, IPT>::reset(T* ptr, size_type n, IvyMemoryType mem_type, IvyGPUStream* stream){
    this->reset(ptr, n, n, mem_type, stream);
  }
  template<typename T, IvyPointerType IPT>
  __HOST_DEVICE__ void IvyUnifiedPtr<T, IPT>::reset(T* ptr, IvyMemoryType mem_type, IvyGPUStream* stream){
    this->reset(ptr, __STATIC_CAST__(size_type, 1), mem_type, stream);
  }

  template<typename T, IvyPointerType IPT> __HOST_DEVICE__ bool IvyUnifiedPtr<T, IPT>::transfer_internal_memory(IvyMemoryType const& new_mem_type, bool release_old){
    return this->transfer_impl(new_mem_type, true, true, release_old);
  }

  template<typename T, IvyPointerType IPT> __HOST_DEVICE__ bool IvyUnifiedPtr<T, IPT>::transfer_impl(IvyMemoryType const& new_mem_type, bool transfer_all, bool copy_ptr, bool release_old){
    bool res = true;
    if (ptr_){
      auto const current_mem_type = this->get_memory_type();
      if (copy_ptr || current_mem_type != new_mem_type){
        IvyMemoryType misc_mem_type = (transfer_all ? new_mem_type : exec_mem_type_);
        pointer new_ptr_ = nullptr;
        control_block_type* new_cblock_ = nullptr;
        size_type const n_size = this->size();
        size_type const n_capacity = this->capacity();

        // The metadata is now a single coalesced control block, so a transfer allocates and
        // moves one object instead of four. Running this serially is faster than spawning
        // OpenMP sections and avoids the associated leak/race surface; OpenMP is reserved
        // for element-array loops.
        operate_with_GPU_stream_from_pointer(
          stream_, ref_stream,
          __ENCAPSULATE__(
            res &= element_allocator_traits::allocate(new_ptr_, n_capacity, new_mem_type, ref_stream);
            res &= element_allocator_traits::transfer(new_ptr_, ptr_, n_size, new_mem_type, current_mem_type, ref_stream);
            new_cblock_ = control_block_allocator_traits::allocate(1, misc_mem_type, ref_stream);
            res &= (new_cblock_ != nullptr);
          )
        );

        // The following line is correct.
        // We should never release when a copy is being made.
        // A copy is a direct memcpy operation from a read-only source.
        if (!copy_ptr || release_old) this->release();

        // If we were not the progenitor before, release() will abort the program, and execution will never reach beyond this point.
        // Otherwise, at this point, we have released the old memory successfully if we needed to.
        // For that reason, we can assume a new progenitor role now.
        progenitor_mem_type_ = IvyMemoryHelpers::get_execution_default_memory();

        exec_mem_type_ = misc_mem_type;
        ptr_ = new_ptr_;
        cblock_ = new_cblock_;
        // Populate the freshly allocated control block (refcount starts at 1 for the new owner).
        control_block_type const cb{ __STATIC_CAST__(counter_type, 1), n_size, n_capacity, new_mem_type };
        this->store_control_block(cb);
      }
    }
    return res;
  }
  template<typename T, IvyPointerType IPT> __HOST__ bool IvyUnifiedPtr<T, IPT>::transfer(IvyMemoryType const& new_mem_type, bool transfer_all, bool release_old){
    return this->transfer_impl(new_mem_type, transfer_all, false, release_old);
  }

  template<typename T, IvyPointerType IPT> __HOST_DEVICE__ bool IvyUnifiedPtr<T, IPT>::copy(T* ptr, size_type n, IvyMemoryType mem_type, IvyGPUStream* stream){
    this->reset();
    bool res = true;
    if (ptr && n>0){
      if (stream) stream_ = stream;
      // Serial: allocating a single coalesced control block is a trivial one-element op;
      // serial avoids the OpenMP-sections leak/race surface and is faster for this little work.
      operate_with_GPU_stream_from_pointer(
        stream_, ref_stream,
        __ENCAPSULATE__(
          res &= element_allocator_traits::allocate(ptr_, n, mem_type, ref_stream);
          res &= element_allocator_traits::transfer(ptr_, ptr, n, mem_type, mem_type, ref_stream);
          cblock_ = control_block_allocator_traits::allocate(1, exec_mem_type_, ref_stream);
          res &= (cblock_ != nullptr);
        )
      );
      control_block_type const cb{ __STATIC_CAST__(counter_type, 1), n, n, mem_type };
      this->store_control_block(cb);
    }
    return res;
  }

  template<typename T, IvyPointerType IPT> template<typename U> __HOST_DEVICE__ void IvyUnifiedPtr<T, IPT>::swap(IvyUnifiedPtr<U, IPT>& other) __NOEXCEPT__{
    if (
      !this->check_write_access(other.get_progenitor_memory_type())
      &&
      !this->empty() && !other.empty()
      ){
      __PRINT_ERROR__("IvyUnifiedPtr::swap() failed: No write access to swap pointers.\n");
      assert(false);
    }
    bool inull = (ptr_==nullptr), onull = (other.get()==nullptr);
    pointer tmp_ptr = ptr_;
    ptr_ = __DYNAMIC_CAST__(pointer, other.get());
    other.get() = __DYNAMIC_CAST__(__ENCAPSULATE__(typename IvyUnifiedPtr<U, IPT>::pointer), tmp_ptr);
    if ((inull != (other.ptr_==nullptr)) || (onull != (ptr_==nullptr))){
      __PRINT_ERROR__("IvyUnifiedPtr::swap() failed: Incompatible types\n");
      assert(false);
    }
    std_util::swap(cblock_, other.control_block());
    std_util::swap(exec_mem_type_, other.get_exec_memory_type());
    std_util::swap(progenitor_mem_type_, other.get_progenitor_memory_type());
    std_util::swap(stream_, other.gpu_stream());
  }
  template<typename T, IvyPointerType IPT> __HOST_DEVICE__ void IvyUnifiedPtr<T, IPT>::swap(IvyUnifiedPtr<T, IPT>& other) __NOEXCEPT__{
    if (
      !this->check_write_access(other.get_progenitor_memory_type())
      &&
      !this->empty() && !other.empty()
      ){
      __PRINT_ERROR__("IvyUnifiedPtr::swap() failed: No write access to swap pointers.\n");
      assert(false);
    }
    std_util::swap(ptr_, other.get());
    std_util::swap(cblock_, other.control_block());
    std_util::swap(exec_mem_type_, other.get_exec_memory_type());
    std_util::swap(progenitor_mem_type_, other.get_progenitor_memory_type());
    std_util::swap(stream_, other.gpu_stream());
  }

  template<typename T, IvyPointerType IPT> __HOST_DEVICE__ IvyUnifiedPtr<T, IPT>::counter_type IvyUnifiedPtr<T, IPT>::use_count() const{
    if (!cblock_) return 0;
    if (this->control_block_is_addressable()){
      return std_atomic::atomic_ref<counter_type>(cblock_->ref_count).load(std_atomic::memory_order_acquire);
    }
    else return this->read_meta_field(&cblock_->ref_count);
  }
  template<typename T, IvyPointerType IPT> __HOST_DEVICE__ IvyUnifiedPtr<T, IPT>::counter_type IvyUnifiedPtr<T, IPT>::inc_dec_counter(bool do_inc){
    if (!cblock_){
      __PRINT_ERROR__("IvyUnifiedPtr::inc_dec_counter() failed: cblock_ is null.\n");
      assert(false);
    }
    if (this->control_block_is_addressable()){
      // The control block lives in directly-addressable memory of the current execution space.
      // Mutate the counter atomically so concurrent copies/destructions of shared_ptr are race-free.
      // Increments may use relaxed ordering; the decrement must be acquire/release so that
      // the thread observing the transition to zero sees all prior writes before freeing.
      // std_atomic::atomic_ref maps to cuda::std::atomic_ref (CUDA) or std::atomic_ref (host),
      // so both execution spaces share the same ordering guarantees.
      std_atomic::atomic_ref<counter_type> a_ref_count(cblock_->ref_count);
      if (do_inc) return a_ref_count.fetch_add(__STATIC_CAST__(counter_type, 1), std_atomic::memory_order_relaxed);
      else return a_ref_count.fetch_sub(__STATIC_CAST__(counter_type, 1), std_atomic::memory_order_acq_rel);
    }
    else{
      // Non-addressable (e.g. device) memory reached from another execution space: the counter
      // cannot be mutated in place, so read just the ref_count field into an on-stack temporary,
      // modify it, and write it back. Touching only this field avoids clobbering concurrent
      // updates to the other control-block fields.
      // A host-side hardware atomic cannot operate on memory that is not directly addressable
      // from the host (the counter must be staged through a transfer), so we instead serialize
      // the staged read-modify-write with a host-side spinlock keyed by the control-block
      // address. This makes concurrent host-thread ownership updates race-free; only host threads
      // can reach this branch, since device code addresses its own memory directly and takes the
      // hardware-atomic path above.
#if DEVICE_CODE == DEVICE_CODE_HOST
      std_atomic::atomic_flag& rc_lock = detail::nonaddressable_refcount_lock(cblock_);
      while (rc_lock.test_and_set(std_atomic::memory_order_acquire)){ ::std::this_thread::yield(); /* spin with backoff until acquired */ }
#endif
      counter_type prev = this->read_meta_field(&cblock_->ref_count);
      counter_type next = prev;
      if (do_inc) ++next;
      else --next;
      this->write_meta_field(&cblock_->ref_count, next);
#if DEVICE_CODE == DEVICE_CODE_HOST
      rc_lock.clear(std_atomic::memory_order_release);
#endif
      return prev;
    }
  }
  template<typename T, IvyPointerType IPT> __HOST_DEVICE__ void IvyUnifiedPtr<T, IPT>::inc_dec_size(bool do_inc){
    if (!cblock_){
      __PRINT_ERROR__("IvyUnifiedPtr::inc_dec_size() failed: cblock_ is null.\n");
      assert(false);
    }
    if (this->control_block_is_addressable()){
      if (do_inc) ++(cblock_->size);
      else --(cblock_->size);
    }
    else{
      size_type s = this->read_meta_field(&cblock_->size);
      if (do_inc) ++s;
      else --s;
      this->write_meta_field(&cblock_->size, s);
    }
  }
  template<typename T, IvyPointerType IPT> __HOST_DEVICE__ void IvyUnifiedPtr<T, IPT>::inc_dec_capacity(bool do_inc, size_type inc){
    if (!cblock_){
      __PRINT_ERROR__("IvyUnifiedPtr::inc_dec_capacity() failed: cblock_ is null.\n");
      assert(false);
    }
    if (this->control_block_is_addressable()){
      if (do_inc) cblock_->capacity += inc;
      else cblock_->capacity -= inc;
    }
    else{
      size_type c = this->read_meta_field(&cblock_->capacity);
      if (do_inc) c += inc;
      else c -= inc;
      this->write_meta_field(&cblock_->capacity, c);
    }
  }

  template<typename T, IvyPointerType IPT> __HOST_DEVICE__ bool IvyUnifiedPtr<T, IPT>::unique() const{ return this->use_count() == __STATIC_CAST__(counter_type, 1); }
  template<typename T, IvyPointerType IPT> __HOST_DEVICE__ IvyUnifiedPtr<T, IPT>::operator bool() const __NOEXCEPT__{ return ptr_ != nullptr; }

  template<typename T, IvyPointerType IPT> __HOST_DEVICE__ void IvyUnifiedPtr<T, IPT>::reserve(size_type const& n){
    auto n_capacity = this->capacity();
    IvyMemoryType const mem_type = this->get_memory_type();
    if (n_capacity==0 && n>0){
      this->check_write_access_or_die();
      pointer new_ptr_ = nullptr;
      operate_with_GPU_stream_from_pointer(
        stream_, ref_stream,
        __ENCAPSULATE__(
          element_allocator_traits::allocate(new_ptr_, n, mem_type, ref_stream);
        )
      );
      this->reset(new_ptr_, 0, n, mem_type, stream_);
    }
    else if (n>n_capacity){
      this->check_write_access_or_die();
      pointer new_ptr_ = nullptr;
      operate_with_GPU_stream_from_pointer(
        stream_, ref_stream,
        __ENCAPSULATE__(
          element_allocator_traits::allocate(new_ptr_, n, mem_type, ref_stream);
          // Use barebones memory transfer; we do not want to call any transfer_internal_memory() function.
          IvyMemoryHelpers::transfer_memory(new_ptr_, ptr_, n_capacity, mem_type, mem_type, ref_stream);
          std_util::swap(new_ptr_, ptr_);
          element_allocator_traits::deallocate(new_ptr_, n_capacity, mem_type, ref_stream);
        )
      );
      this->inc_dec_capacity(true, n-n_capacity);
    }
  }
  template<typename T, IvyPointerType IPT> __HOST_DEVICE__ void IvyUnifiedPtr<T, IPT>::reserve(size_type const& n, IvyMemoryType new_mem_type, IvyGPUStream* stream){
    auto n_capacity = this->capacity();
    IvyMemoryType const mem_type = this->get_memory_type();
    if (stream) stream_ = stream;
    if (new_mem_type!=mem_type){
      this->check_write_access_or_die();
      pointer new_ptr_ = nullptr;
      operate_with_GPU_stream_from_pointer(
        stream_, ref_stream,
        __ENCAPSULATE__(
          element_allocator_traits::allocate(new_ptr_, n, new_mem_type, ref_stream);
        )
      );
      this->reset(new_ptr_, 0, n, new_mem_type, stream);
    }
    else this->reserve(n);
  }
  template<typename T, IvyPointerType IPT> __HOST_DEVICE__ void IvyUnifiedPtr<T, IPT>::shrink_to_fit(){
    auto n_capacity = this->capacity();
    auto n_size = this->size();
    if (n_size<n_capacity){
      this->check_write_access_or_die();
      IvyMemoryType const mem_type = this->get_memory_type();
      pointer new_ptr_ = nullptr;
      operate_with_GPU_stream_from_pointer(
        stream_, ref_stream,
        __ENCAPSULATE__(
          element_allocator_traits::allocate(new_ptr_, n_size, mem_type, ref_stream);
          // Use barebones memory transfer; we do not want to call any transfer_internal_memory() function.
          IvyMemoryHelpers::transfer_memory(new_ptr_, ptr_, n_size, mem_type, mem_type, ref_stream);
          std_util::swap(new_ptr_, ptr_);
          element_allocator_traits::deallocate(new_ptr_, n_capacity, mem_type, ref_stream);
        )
      );
      this->inc_dec_capacity(false, n_capacity-n_size);
    }
  }
  template<typename T, IvyPointerType IPT> template<typename... Args> __HOST_DEVICE__ void IvyUnifiedPtr<T, IPT>::emplace_back(Args&&... args){
    auto n_size = this->size();
    auto n_capacity = this->capacity();
    if (n_size==n_capacity) this->reserve(n_capacity+2);
    IvyMemoryType const current_mem_type = this->get_memory_type();
    pointer ptr_loc = (ptr_+n_size);
    operate_with_GPU_stream_from_pointer(
      stream_, ref_stream,
      __ENCAPSULATE__(
        element_allocator_traits::construct(ptr_loc, 1, current_mem_type, ref_stream, args...);
      )
    );
    this->inc_dec_size(true);
  }
  template<typename T, IvyPointerType IPT> template<typename... Args> __HOST_DEVICE__ void IvyUnifiedPtr<T, IPT>::push_back(Args&&... args){
    this->emplace_back(args...);
  }

  template<typename T, IvyPointerType IPT> template<typename... Args> __HOST_DEVICE__ void IvyUnifiedPtr<T, IPT>::insert(size_type const& i, Args&&... args){
    auto n_size = this->size();
    assert(i<n_size);
    auto n_capacity = this->capacity();
    IvyMemoryType const current_mem_type = this->get_memory_type();
    if (n_size==n_capacity) this->reserve(n_capacity+2);
    pointer ptr_here = (ptr_+i);
    pointer ptr_next = (ptr_here+1);
    pointer tmp_ptr_ = nullptr;
    size_type n_to_shift = n_size - i;
    operate_with_GPU_stream_from_pointer(
      stream_, ref_stream,
      __ENCAPSULATE__(
        // Shift objects via barebones memory transfer; we do not want to call any transfer_internal_memory() function.
        IvyMemoryHelpers::allocate_memory(tmp_ptr_, n_to_shift, current_mem_type, ref_stream);
        IvyMemoryHelpers::transfer_memory(tmp_ptr_, ptr_here, n_to_shift, current_mem_type, current_mem_type, ref_stream);
        IvyMemoryHelpers::transfer_memory(ptr_next, tmp_ptr_, n_to_shift, current_mem_type, current_mem_type, ref_stream);
        element_allocator_traits::construct(ptr_here, 1, current_mem_type, ref_stream, args...);
        IvyMemoryHelpers::free_memory(tmp_ptr_, n_to_shift, current_mem_type, ref_stream);
      )
    );
    this->inc_dec_size(true);
  }
  template<typename T, IvyPointerType IPT> __HOST_DEVICE__ void IvyUnifiedPtr<T, IPT>::pop_back(){
    auto n_size = this->size();
    if (n_size==0) return;
    IvyMemoryType const current_mem_type = this->get_memory_type();
    pointer ptr_loc = (ptr_+(n_size-1));
    operate_with_GPU_stream_from_pointer(
      stream_, ref_stream,
      __ENCAPSULATE__(
        element_allocator_traits::destruct(ptr_loc, 1, current_mem_type, ref_stream);
      )
    );
    this->inc_dec_size(false);
  }
  template<typename T, IvyPointerType IPT> __HOST_DEVICE__ void IvyUnifiedPtr<T, IPT>::erase(size_type const& i){
    auto n_size = this->size();
    if (n_size==0) return;
    assert(i<n_size);
    auto n_capacity = this->capacity();
    IvyMemoryType const current_mem_type = this->get_memory_type();
    pointer ptr_here = (ptr_+i);
    pointer ptr_next = (ptr_here+1);
    pointer tmp_ptr_ = nullptr;
    size_type n_to_shift = n_capacity - i - 1;
    operate_with_GPU_stream_from_pointer(
      stream_, ref_stream,
      __ENCAPSULATE__(
        element_allocator_traits::destruct(ptr_here, 1, current_mem_type, ref_stream);
        // Shift objects via barebones memory transfer; we do not want to call any transfer_internal_memory() function.
        IvyMemoryHelpers::allocate_memory(tmp_ptr_, n_to_shift, current_mem_type, ref_stream);
        IvyMemoryHelpers::transfer_memory(tmp_ptr_, ptr_next, n_to_shift, current_mem_type, current_mem_type, ref_stream);
        IvyMemoryHelpers::transfer_memory(ptr_here, tmp_ptr_, n_to_shift, current_mem_type, current_mem_type, ref_stream);
        IvyMemoryHelpers::free_memory(tmp_ptr_, n_to_shift, current_mem_type, ref_stream);
      )
    );
    this->inc_dec_size(false);
  }

  template<typename T, IvyPointerType IPT> __HOST_DEVICE__ bool IvyUnifiedPtr<T, IPT>::check_write_access() const{ return (this->progenitor_mem_type_==IvyMemoryHelpers::get_execution_default_memory()); }
  template<typename T, IvyPointerType IPT> __HOST_DEVICE__ bool IvyUnifiedPtr<T, IPT>::check_write_access(IvyMemoryType const& mem_type) const{ return (this->progenitor_mem_type_==mem_type); }
  template<typename T, IvyPointerType IPT> __HOST_DEVICE__ void IvyUnifiedPtr<T, IPT>::check_write_access_or_die() const{
    if (!this->check_write_access()){
      __PRINT_ERROR__("IvyUnifiedPtr: Write access denied.\n");
      assert(false);
    }
  }
  template<typename T, IvyPointerType IPT> __HOST_DEVICE__ void IvyUnifiedPtr<T, IPT>::check_write_access_or_die(IvyMemoryType const& mem_type) const{
    if (!this->check_write_access(mem_type)){
      __PRINT_ERROR__("IvyUnifiedPtr: Write access denied.\n");
      assert(false);
    }
  }

  // memview view
  template<typename T, IvyPointerType IPT>
  __HOST_DEVICE__ unifiedptr_view<T, IPT> view(IvyUnifiedPtr<T, IPT> const& ptr){
    using type_t = IvyUnifiedPtr<T, IPT>;
    auto def_mem_type = IvyMemoryHelpers::get_execution_default_memory();
    auto mem_type = ptr.get_memory_type();
    return unifiedptr_view<T, IPT>(
      __CONST_CAST__(type_t*, std_ivy::addressof(ptr)),
      def_mem_type, ptr.gpu_stream(), true,
      (mem_type!=def_mem_type)
    );
  }

  // Non-member helper functions
  template<typename T, typename U, IvyPointerType IPT, IvyPointerType IPU>
  __HOST_DEVICE__ bool operator==(IvyUnifiedPtr<T, IPT> const& a, IvyUnifiedPtr<U, IPU> const& b) __NOEXCEPT__{ return (a.get()==b.get()); }
  template<typename T, typename U, IvyPointerType IPT, IvyPointerType IPU>
  __HOST_DEVICE__ bool operator!=(IvyUnifiedPtr<T, IPT> const& a, IvyUnifiedPtr<U, IPU> const& b) __NOEXCEPT__{ return !(a==b); }

  template<typename T, IvyPointerType IPT> __HOST_DEVICE__ bool operator==(IvyUnifiedPtr<T, IPT> const& a, T* ptr) __NOEXCEPT__{ return (a.get()==ptr); }
  template<typename T, IvyPointerType IPT> __HOST_DEVICE__ bool operator!=(IvyUnifiedPtr<T, IPT> const& a, T* ptr) __NOEXCEPT__{ return !(a==ptr); }

  template<typename T, IvyPointerType IPT> __HOST_DEVICE__ bool operator==(T* ptr, IvyUnifiedPtr<T, IPT> const& a) __NOEXCEPT__{ return (ptr==a.get()); }
  template<typename T, IvyPointerType IPT> __HOST_DEVICE__ bool operator!=(T* ptr, IvyUnifiedPtr<T, IPT> const& a) __NOEXCEPT__{ return !(ptr==a); }

  template<typename T, IvyPointerType IPT> __HOST_DEVICE__ bool operator==(IvyUnifiedPtr<T, IPT> const& a, std_cstddef::nullptr_t) __NOEXCEPT__{ return (a.get()==nullptr); }
  template<typename T, IvyPointerType IPT> __HOST_DEVICE__ bool operator!=(IvyUnifiedPtr<T, IPT> const& a, std_cstddef::nullptr_t) __NOEXCEPT__{ return !(a==nullptr); }

  template<typename T, IvyPointerType IPT> __HOST_DEVICE__ bool operator==(std_cstddef::nullptr_t, IvyUnifiedPtr<T, IPT> const& a) __NOEXCEPT__{ return (nullptr==a.get()); }
  template<typename T, IvyPointerType IPT> __HOST_DEVICE__ bool operator!=(std_cstddef::nullptr_t, IvyUnifiedPtr<T, IPT> const& a) __NOEXCEPT__{ return !(nullptr==a); }

  template<typename T, IvyPointerType IPT, typename Allocator_t, typename... Args>
  __HOST_DEVICE__ IvyUnifiedPtr<T, IPT> build_unified(
    Allocator_t const& a,
    typename IvyUnifiedPtr<T, IPT>::size_type n_size, typename IvyUnifiedPtr<T, IPT>::size_type n_capacity,
    IvyMemoryType mem_type, IvyGPUStream* stream,
    Args&&... args
  ){
    if (n_capacity==0) return IvyUnifiedPtr<T, IPT>();
    operate_with_GPU_stream_from_pointer(
      stream, ref_stream,
      __ENCAPSULATE__(
        typename IvyUnifiedPtr<T, IPT>::pointer ptr = a.allocate(n_capacity, mem_type, ref_stream);
        a.construct(ptr, n_size, mem_type, ref_stream, args...);
      )
    );
    return IvyUnifiedPtr<T, IPT>(ptr, n_size, n_capacity, mem_type, stream);
  }
  template<typename T, typename Allocator_t, typename... Args>
  __HOST_DEVICE__ shared_ptr<T> build_shared(
    Allocator_t const& a,
    typename shared_ptr<T>::size_type n_size, typename shared_ptr<T>::size_type n_capacity,
    IvyMemoryType mem_type, IvyGPUStream* stream,
    Args&&... args
  ){ return build_unified<T, IvyPointerType::shared, Allocator_t>(a, n_size, n_capacity, mem_type, stream, args...); }
  template<typename T, typename Allocator_t, typename... Args>
  __HOST_DEVICE__ unique_ptr<T> build_unique(
    Allocator_t const& a,
    typename unique_ptr<T>::size_type n_size, typename unique_ptr<T>::size_type n_capacity,
    IvyMemoryType mem_type, IvyGPUStream* stream,
    Args&&... args
  ){ return build_unified<T, IvyPointerType::unique, Allocator_t>(a, n_size, n_capacity, mem_type, stream, args...); }

  template<typename T, IvyPointerType IPT, typename... Args>
  __HOST_DEVICE__ IvyUnifiedPtr<T, IPT> make_unified(
    typename IvyUnifiedPtr<T, IPT>::size_type n_size, typename IvyUnifiedPtr<T, IPT>::size_type n_capacity,
    IvyMemoryType mem_type, IvyGPUStream* stream,
    Args&&... args
  ){ return build_unified<T, IPT, std_ivy::allocator<T>>(std_ivy::allocator<T>(), n_size, n_capacity, mem_type, stream, args...); }
  template<typename T, typename... Args>
  __HOST_DEVICE__ shared_ptr<T> make_shared(
    typename shared_ptr<T>::size_type n_size, typename shared_ptr<T>::size_type n_capacity,
    IvyMemoryType mem_type, IvyGPUStream* stream,
    Args&&... args
  ){ return make_unified<T, IvyPointerType::shared>(n_size, n_capacity, mem_type, stream, args...); }
  template<typename T, typename... Args>
  __HOST_DEVICE__ unique_ptr<T> make_unique(
    typename unique_ptr<T>::size_type n_size, typename unique_ptr<T>::size_type n_capacity,
    IvyMemoryType mem_type, IvyGPUStream* stream,
    Args&&... args
  ){ return make_unified<T, IvyPointerType::unique>(n_size, n_capacity, mem_type, stream, args...); }


  template<typename T, IvyPointerType IPT, typename Allocator_t, typename... Args>
  __HOST_DEVICE__ IvyUnifiedPtr<T, IPT> build_unified(
    Allocator_t const& a,
    typename IvyUnifiedPtr<T, IPT>::size_type n,
    IvyMemoryType mem_type, IvyGPUStream* stream,
    Args&&... args
  ){
    if (n==0) return IvyUnifiedPtr<T, IPT>();
    operate_with_GPU_stream_from_pointer(
      stream, ref_stream,
      __ENCAPSULATE__(
        typename IvyUnifiedPtr<T, IPT>::pointer ptr = a.build(n, mem_type, ref_stream, args...);
      )
    );
    return IvyUnifiedPtr<T, IPT>(ptr, n, mem_type, stream);
  }
  template<typename T, typename Allocator_t, typename... Args>
  __HOST_DEVICE__ shared_ptr<T> build_shared(
    Allocator_t const& a,
    typename shared_ptr<T>::size_type n,
    IvyMemoryType mem_type, IvyGPUStream* stream,
    Args&&... args
  ){ return build_unified<T, IvyPointerType::shared, Allocator_t>(a, n, mem_type, stream, args...); }
  template<typename T, typename Allocator_t, typename... Args>
  __HOST_DEVICE__ unique_ptr<T> build_unique(
    Allocator_t const& a,
    typename unique_ptr<T>::size_type n,
    IvyMemoryType mem_type, IvyGPUStream* stream,
    Args&&... args
  ){ return build_unified<T, IvyPointerType::unique, Allocator_t>(a, n, mem_type, stream, args...); }

  template<typename T, IvyPointerType IPT, typename... Args>
  __HOST_DEVICE__ IvyUnifiedPtr<T, IPT> make_unified(
    typename IvyUnifiedPtr<T, IPT>::size_type n,
    IvyMemoryType mem_type, IvyGPUStream* stream,
    Args&&... args
  ){ return build_unified<T, IPT, std_ivy::allocator<T>>(std_ivy::allocator<T>(), n, mem_type, stream, args...); }
  template<typename T, typename... Args>
  __HOST_DEVICE__ shared_ptr<T> make_shared(
    typename shared_ptr<T>::size_type n,
    IvyMemoryType mem_type, IvyGPUStream* stream,
    Args&&... args
  ){ return make_unified<T, IvyPointerType::shared>(n, mem_type, stream, args...); }
  template<typename T, typename... Args>
  __HOST_DEVICE__ unique_ptr<T> make_unique(
    typename unique_ptr<T>::size_type n,
    IvyMemoryType mem_type, IvyGPUStream* stream,
    Args&&... args
  ){ return make_unified<T, IvyPointerType::unique>(n, mem_type, stream, args...); }


  template<typename T, IvyPointerType IPT, typename Allocator_t, typename... Args>
  __HOST_DEVICE__ IvyUnifiedPtr<T, IPT> build_unified(
    Allocator_t const& a, IvyMemoryType mem_type, IvyGPUStream* stream,
    Args&&... args
  ){
    return build_unified<T, IPT, Allocator_t>(a, __STATIC_CAST__(__ENCAPSULATE__(typename IvyUnifiedPtr<T, IPT>::size_type), 1), mem_type, stream, args...);
  }
  template<typename T, typename Allocator_t, typename... Args>
  __HOST_DEVICE__ shared_ptr<T> build_shared(
    Allocator_t const& a, IvyMemoryType mem_type, IvyGPUStream* stream,
    Args&&... args
  ){ return build_unified<T, IvyPointerType::shared, Allocator_t>(a, mem_type, stream, args...); }
  template<typename T, typename Allocator_t, typename... Args>
  __HOST_DEVICE__ unique_ptr<T> build_unique(
    Allocator_t const& a, IvyMemoryType mem_type, IvyGPUStream* stream,
    Args&&... args
  ){ return build_unified<T, IvyPointerType::unique, Allocator_t>(a, mem_type, stream, args...); }

  template<typename T, IvyPointerType IPT, typename... Args>
  __HOST_DEVICE__ IvyUnifiedPtr<T, IPT> make_unified(IvyMemoryType mem_type, IvyGPUStream* stream, Args&&... args){
    return build_unified<T, IPT, std_ivy::allocator<T>>(std_ivy::allocator<T>(), mem_type, stream, args...);
  }
  template<typename T, typename... Args>
  __HOST_DEVICE__ shared_ptr<T> make_shared(IvyMemoryType mem_type, IvyGPUStream* stream, Args&&... args){
    return make_unified<T, IvyPointerType::shared>(mem_type, stream, args...);
  }
  template<typename T, typename... Args>
  __HOST_DEVICE__ unique_ptr<T> make_unique(IvyMemoryType mem_type, IvyGPUStream* stream, Args&&... args){
    return make_unified<T, IvyPointerType::unique>(mem_type, stream, args...);
  }

  template<typename T, IvyPointerType IPT> struct value_printout<IvyUnifiedPtr<T, IPT>>{
    static __HOST_DEVICE__ void print(IvyUnifiedPtr<T, IPT> const& x){
      using element_allocator_type = typename IvyUnifiedPtr<T, IPT>::element_allocator_type;
      using size_type = typename IvyUnifiedPtr<T, IPT>::size_type;
      using element_type = typename IvyUnifiedPtr<T, IPT>::element_type;
      using pointer = typename IvyUnifiedPtr<T, IPT>::pointer;

      if (!x){
        __PRINT_INFO__("(null)");
        return;
      }

      auto const s = x.size();
      if (s==0){
        __PRINT_INFO__("(empty)");
        return;
      }

      if (s>1) __PRINT_INFO__("{ ");
      auto ptr = x.get();
      auto const mem_type = x.get_memory_type();
      auto stream = x.gpu_stream();
      memview<element_type, element_allocator_type> x_view(ptr, s, mem_type, stream, true);
      for (size_type i=0; i<s; ++i){
        print_value(x_view[i], false);
        if (i<s-1) __PRINT_INFO__(", ");
      }
      if (s>1) __PRINT_INFO__(" }");
    }
  };
}
namespace std_util{
  template<typename T, typename U, std_ivy::IvyPointerType IPT> __HOST_DEVICE__ void swap(std_ivy::IvyUnifiedPtr<T, IPT>& a, std_ivy::IvyUnifiedPtr<U, IPT>& b) __NOEXCEPT__{ a.swap(b); }
  template<typename T, std_ivy::IvyPointerType IPT> __HOST_DEVICE__ void swap(std_ivy::IvyUnifiedPtr<T, IPT>& a, std_ivy::IvyUnifiedPtr<T, IPT>& b) __NOEXCEPT__{ a.swap(b); }
}


#endif
