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
// copies/destructions are race-free. IvyAtomic selects cuda::atomic_ref on the CUDA
// build and std::atomic_ref otherwise, giving consistent host/device semantics.
#include "std_ivy/IvyAtomic.h"


namespace std_ivy{
  template<typename T, IvyPointerType IPT> __HOST_DEVICE__ IvyUnifiedPtr<T, IPT>::IvyUnifiedPtr(IvyGPUStream* stream) :
    ptr_(nullptr),
    mem_type_(nullptr),
    size_(nullptr),
    capacity_(nullptr),
    ref_count_(nullptr),
    stream_(stream),
    exec_mem_type_(IvyMemoryHelpers::get_execution_default_memory()),
    progenitor_mem_type_(IvyMemoryHelpers::get_execution_default_memory())
  {}
  template<typename T, IvyPointerType IPT> __HOST_DEVICE__ IvyUnifiedPtr<T, IPT>::IvyUnifiedPtr(std_cstddef::nullptr_t, IvyGPUStream* stream) :
    ptr_(nullptr),
    mem_type_(nullptr),
    size_(nullptr),
    capacity_(nullptr),
    ref_count_(nullptr),
    stream_(stream),
    exec_mem_type_(IvyMemoryHelpers::get_execution_default_memory()),
    progenitor_mem_type_(IvyMemoryHelpers::get_execution_default_memory())
  {}
  template<typename T, IvyPointerType IPT>
  __HOST_DEVICE__ IvyUnifiedPtr<T, IPT>::IvyUnifiedPtr(T* ptr, IvyMemoryType mem_type, IvyGPUStream* stream) :
    ptr_(ptr),
    stream_(stream),
    exec_mem_type_(IvyMemoryHelpers::get_execution_default_memory()),
    progenitor_mem_type_(IvyMemoryHelpers::get_execution_default_memory())
  {
    if (ptr_) this->init_members(mem_type, 1, 1);
  }
  template<typename T, IvyPointerType IPT>
  __HOST_DEVICE__ IvyUnifiedPtr<T, IPT>::IvyUnifiedPtr(T* ptr, size_type n, IvyMemoryType mem_type, IvyGPUStream* stream) :
    ptr_(ptr),
    stream_(stream),
    exec_mem_type_(IvyMemoryHelpers::get_execution_default_memory()),
    progenitor_mem_type_(IvyMemoryHelpers::get_execution_default_memory())
  {
    if (ptr_) this->init_members(mem_type, n, n);
  }
  template<typename T, IvyPointerType IPT>
  __HOST_DEVICE__ IvyUnifiedPtr<T, IPT>::IvyUnifiedPtr(T* ptr, size_type n_size, size_type n_capacity, IvyMemoryType mem_type, IvyGPUStream* stream) :
    ptr_(ptr),
    stream_(stream),
    exec_mem_type_(IvyMemoryHelpers::get_execution_default_memory()),
    progenitor_mem_type_(IvyMemoryHelpers::get_execution_default_memory())
  {
    if (ptr_) this->init_members(mem_type, n_size, n_capacity);
  }
  template<typename T, IvyPointerType IPT>
  template<typename U>
  __HOST_DEVICE__ IvyUnifiedPtr<T, IPT>::IvyUnifiedPtr(U* ptr, IvyMemoryType mem_type, IvyGPUStream* stream) :
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
      mem_type_ = other.get_memory_type_ptr();
      size_ = other.size_ptr();
      capacity_ = other.capacity_ptr();
      ref_count_ = other.counter();
      stream_ = other.gpu_stream();
      exec_mem_type_ = other.get_exec_memory_type();
      if (ref_count_) this->inc_dec_counter(true);
    }
    // We should reset the other pointer if it is unique and its progenitor memory type is the same as that of the current context.
    if constexpr (IPU==IvyPointerType::unique){
      if (other.check_write_access()) __CONST_CAST__(__ENCAPSULATE__(IvyUnifiedPtr<U, IPU>&), other).reset();
    }
  }
  template<typename T, IvyPointerType IPT> __HOST_DEVICE__ IvyUnifiedPtr<T, IPT>::IvyUnifiedPtr(IvyUnifiedPtr<T, IPT> const& other) :
    ptr_(other.ptr_),
    mem_type_(other.mem_type_),
    size_(other.size_),
    capacity_(other.capacity_),
    ref_count_(other.ref_count_),
    stream_(other.stream_),
    exec_mem_type_(other.exec_mem_type_),
    progenitor_mem_type_(other.progenitor_mem_type_)
  {
    if (ref_count_) this->inc_dec_counter(true);
    // We should reset the other pointer if it is unique and its progenitor memory type is the same as that of the current context.
    if constexpr (IPT==IvyPointerType::unique){
      if (other.check_write_access()) __CONST_CAST__(__ENCAPSULATE__(IvyUnifiedPtr<T, IPT>&), other).reset();
    }
  }
  template<typename T, IvyPointerType IPT> template<typename U, IvyPointerType IPU, ENABLE_IF_BOOL_IMPL((IPU==IPT || IPU==IvyPointerType::unique) && (IS_BASE_OF(T, U) || IS_BASE_OF(U, T)))>
  __HOST_DEVICE__ IvyUnifiedPtr<T, IPT>::IvyUnifiedPtr(IvyUnifiedPtr<U, IPU>&& other) :
    ptr_(std_util::move(other.get())),
    mem_type_(std_util::move(other.get_memory_type_ptr())),
    size_(std_util::move(other.size_ptr())),
    capacity_(std_util::move(other.capacity_ptr())),
    ref_count_(std_util::move(other.counter())),
    stream_(std_util::move(other.gpu_stream())),
    exec_mem_type_(std_util::move(other.get_exec_memory_type())),
    progenitor_mem_type_(std_util::move(other.get_progenitor_memory_type()))
  {
    other.check_write_access_or_die();
    IvySecrets::dump_helper::dump(other);
  }
  template<typename T, IvyPointerType IPT> __HOST_DEVICE__ IvyUnifiedPtr<T, IPT>::IvyUnifiedPtr(IvyUnifiedPtr&& other) :
    ptr_(std_util::move(other.ptr_)),
    mem_type_(std_util::move(other.mem_type_)),
    size_(std_util::move(other.size_)),
    capacity_(std_util::move(other.capacity_)),
    ref_count_(std_util::move(other.ref_count_)),
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
      mem_type_ = other.get_memory_type_ptr();
      ptr_ = __DYNAMIC_CAST__(pointer, other.get());
      if (!ptr_ && other.get()){
        __PRINT_ERROR__("IvyUnifiedPtr copy assignment failed: Incompatible types\n");
        assert(false);
      }
      size_ = other.size_ptr();
      capacity_ = other.capacity_ptr();
      ref_count_ = other.counter();
      stream_ = other.gpu_stream();
      if (ref_count_) this->inc_dec_counter(true);
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
      mem_type_ = other.mem_type_;
      ptr_ = other.ptr_;
      size_ = other.size_;
      capacity_ = other.capacity_;
      ref_count_ = other.ref_count_;
      stream_ = other.gpu_stream();
      if (ref_count_) this->inc_dec_counter(true);
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
    mem_type_ = std_util::move(other.get_memory_type_ptr());
    size_ = std_util::move(other.size_ptr());
    capacity_ = std_util::move(other.capacity_ptr());
    ref_count_ = std_util::move(other.counter());
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
    mem_type_ = std_util::move(other.mem_type_);
    size_ = std_util::move(other.size_);
    capacity_ = std_util::move(other.capacity_);
    ref_count_ = std_util::move(other.ref_count_);
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
        size_ = size_allocator_traits::build(1, exec_mem_type_, ref_stream, n_size);
        capacity_ = size_allocator_traits::build(1, exec_mem_type_, ref_stream, n_capacity);
        ref_count_ = counter_allocator_traits::build(1, exec_mem_type_, ref_stream, __STATIC_CAST__(counter_type, 1));
        mem_type_ = mem_type_allocator_traits::build(1, exec_mem_type_, ref_stream, mem_type);
      )
    );
  }
  template<typename T, IvyPointerType IPT> __HOST_DEVICE__ void IvyUnifiedPtr<T, IPT>::release(){
    if (ref_count_){
      // Atomically decrement first. The thread that observes the transition to zero
      // (i.e. sees a previous value of 1) is the sole owner responsible for cleanup.
      // This avoids the check-then-act race of reading use_count() and then freeing.
      auto const prev_count = this->inc_dec_counter(false);
      if (prev_count==1){
        // We were the only owner, so we can clean up. We must have write access first.
        if (this->check_write_access()){
          auto const current_mem_type = this->get_memory_type();
          operate_with_GPU_stream_from_pointer(
            stream_, ref_stream,
            __ENCAPSULATE__(
              if (ptr_){
                element_allocator_traits::destruct(ptr_, this->size(), current_mem_type, ref_stream);
                element_allocator_traits::deallocate(ptr_, this->capacity(), current_mem_type, ref_stream);
              }
              if (size_) size_allocator_traits::destroy(size_, 1, exec_mem_type_, ref_stream);
              if (capacity_) size_allocator_traits::destroy(capacity_, 1, exec_mem_type_, ref_stream);
              counter_allocator_traits::destroy(ref_count_, 1, exec_mem_type_, ref_stream);
              if (mem_type_) mem_type_allocator_traits::destroy(mem_type_, 1, exec_mem_type_, ref_stream);
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
    mem_type_ = nullptr;
    ptr_ = nullptr;
    size_ = nullptr;
    capacity_ = nullptr;
    ref_count_ = nullptr;
    stream_ = nullptr;
    progenitor_mem_type_ = exec_mem_type_ = IvyMemoryHelpers::get_execution_default_memory();
  }

  template<typename T, IvyPointerType IPT> __HOST_DEVICE__ IvyMemoryType const& IvyUnifiedPtr<T, IPT>::get_progenitor_memory_type() const __NOEXCEPT__{ return this->progenitor_mem_type_; }
  template<typename T, IvyPointerType IPT> __HOST_DEVICE__ IvyMemoryType const& IvyUnifiedPtr<T, IPT>::get_exec_memory_type() const __NOEXCEPT__{ return this->exec_mem_type_; }
  template<typename T, IvyPointerType IPT> __HOST_DEVICE__ IvyMemoryType* IvyUnifiedPtr<T, IPT>::get_memory_type_ptr() const __NOEXCEPT__{ return this->mem_type_; }
  template<typename T, IvyPointerType IPT> __HOST_DEVICE__ IvyGPUStream* IvyUnifiedPtr<T, IPT>::gpu_stream() const __NOEXCEPT__{ return this->stream_; }
  template<typename T, IvyPointerType IPT> __HOST_DEVICE__ IvyUnifiedPtr<T, IPT>::size_type* IvyUnifiedPtr<T, IPT>::size_ptr() const __NOEXCEPT__{ return this->size_; }
  template<typename T, IvyPointerType IPT> __HOST_DEVICE__ IvyUnifiedPtr<T, IPT>::size_type* IvyUnifiedPtr<T, IPT>::capacity_ptr() const __NOEXCEPT__{ return this->capacity_; }
  template<typename T, IvyPointerType IPT> __HOST_DEVICE__ IvyUnifiedPtr<T, IPT>::counter_type* IvyUnifiedPtr<T, IPT>::counter() const __NOEXCEPT__{ return this->ref_count_; }
  template<typename T, IvyPointerType IPT> __HOST_DEVICE__ IvyUnifiedPtr<T, IPT>::pointer IvyUnifiedPtr<T, IPT>::get() const __NOEXCEPT__{ return this->ptr_; }

  template<typename T, IvyPointerType IPT> __HOST_DEVICE__ IvyMemoryType& IvyUnifiedPtr<T, IPT>::get_progenitor_memory_type() __NOEXCEPT__{ return this->progenitor_mem_type_; }
  template<typename T, IvyPointerType IPT> __HOST_DEVICE__ IvyMemoryType& IvyUnifiedPtr<T, IPT>::get_exec_memory_type() __NOEXCEPT__{ return this->exec_mem_type_; }
  template<typename T, IvyPointerType IPT> __HOST_DEVICE__ IvyMemoryType*& IvyUnifiedPtr<T, IPT>::get_memory_type_ptr() __NOEXCEPT__{ return this->mem_type_; }
  template<typename T, IvyPointerType IPT> __HOST_DEVICE__ IvyGPUStream*& IvyUnifiedPtr<T, IPT>::gpu_stream() __NOEXCEPT__{ return this->stream_; }
  template<typename T, IvyPointerType IPT> __HOST_DEVICE__ IvyUnifiedPtr<T, IPT>::size_type*& IvyUnifiedPtr<T, IPT>::size_ptr() __NOEXCEPT__{ return this->size_; }
  template<typename T, IvyPointerType IPT> __HOST_DEVICE__ IvyUnifiedPtr<T, IPT>::size_type*& IvyUnifiedPtr<T, IPT>::capacity_ptr() __NOEXCEPT__{ return this->capacity_; }
  template<typename T, IvyPointerType IPT> __HOST_DEVICE__ IvyUnifiedPtr<T, IPT>::counter_type*& IvyUnifiedPtr<T, IPT>::counter() __NOEXCEPT__{ return this->ref_count_; }
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
    if (!size_) return 0;
    constexpr IvyMemoryType def_mem_type = IvyMemoryHelpers::get_execution_default_memory();
    if (exec_mem_type_ == def_mem_type) return *size_;
    else{
      size_type* p_size_ = nullptr;
      {
        operate_with_GPU_stream_from_pointer(
          stream_, ref_stream,
          __ENCAPSULATE__(
            p_size_ = size_allocator_traits::allocate(1, def_mem_type, ref_stream);
            size_allocator_traits::transfer(p_size_, size_, 1, def_mem_type, exec_mem_type_, ref_stream);
          )
        );
      }
      size_type ret = *p_size_;
      {
        operate_with_GPU_stream_from_pointer(
          stream_, ref_stream,
          __ENCAPSULATE__(
            size_allocator_traits::destroy(p_size_, 1, def_mem_type, ref_stream);
          )
        );
      }
      return ret;
    }
  }
  template<typename T, IvyPointerType IPT> __HOST_DEVICE__ IvyUnifiedPtr<T, IPT>::size_type IvyUnifiedPtr<T, IPT>::capacity() const __NOEXCEPT__{
    if (!capacity_) return 0;
    constexpr IvyMemoryType def_mem_type = IvyMemoryHelpers::get_execution_default_memory();
    if (exec_mem_type_ == def_mem_type) return *capacity_;
    else{
      size_type* p_capacity_ = nullptr;
      {
        operate_with_GPU_stream_from_pointer(
          stream_, ref_stream,
          __ENCAPSULATE__(
            p_capacity_ = size_allocator_traits::allocate(1, def_mem_type, ref_stream);
            size_allocator_traits::transfer(p_capacity_, capacity_, 1, def_mem_type, exec_mem_type_, ref_stream);
          )
        );
      }
      size_type ret = *p_capacity_;
      {
        operate_with_GPU_stream_from_pointer(
          stream_, ref_stream,
          __ENCAPSULATE__(
            size_allocator_traits::destroy(p_capacity_, 1, def_mem_type, ref_stream);
          )
        );
      }
      return ret;
    }
  }
  template<typename T, IvyPointerType IPT> __HOST_DEVICE__ IvyMemoryType IvyUnifiedPtr<T, IPT>::get_memory_type() const __NOEXCEPT__{
    constexpr IvyMemoryType def_mem_type = IvyMemoryHelpers::get_execution_default_memory();
    if (!mem_type_) return def_mem_type;
    if (exec_mem_type_ == def_mem_type) return *mem_type_;
    else{
      IvyMemoryType* p_mem_type_ = nullptr;
      {
        operate_with_GPU_stream_from_pointer(
          stream_, ref_stream,
          __ENCAPSULATE__(
            p_mem_type_ = mem_type_allocator_traits::allocate(1, def_mem_type, ref_stream);
            mem_type_allocator_traits::transfer(p_mem_type_, mem_type_, 1, def_mem_type, exec_mem_type_, ref_stream);
          )
        );
      }
      IvyMemoryType ret = *p_mem_type_;
      {
        operate_with_GPU_stream_from_pointer(
          stream_, ref_stream,
          __ENCAPSULATE__(
            mem_type_allocator_traits::destroy(p_mem_type_, 1, def_mem_type, ref_stream);
          )
        );
      }
      return ret;
    }
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
      if (mem_type_ && this->get_memory_type() != mem_type){
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
      if (mem_type_ && this->get_memory_type() != mem_type){
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
        size_type* new_size_ = nullptr;
        size_type* new_capacity_ = nullptr;
        counter_type* new_ref_count_ = nullptr;
        IvyMemoryType* new_mem_type_ = nullptr;
        size_type const n_size = this->size();
        size_type const n_capacity = this->capacity();

        // The per-field metadata allocations below are a handful of one-element ops.
        // Running them serially is faster than spawning OpenMP sections and avoids the
        // associated leak/race surface; OpenMP is reserved for element-array loops.
        operate_with_GPU_stream_from_pointer(
          stream_, ref_stream,
          __ENCAPSULATE__(
            res &= element_allocator_traits::allocate(new_ptr_, n_capacity, new_mem_type, ref_stream);
            res &= element_allocator_traits::transfer(new_ptr_, ptr_, n_size, new_mem_type, current_mem_type, ref_stream);
            res &= size_allocator_traits::build(new_size_, 1, misc_mem_type, ref_stream, n_size);
            res &= size_allocator_traits::build(new_capacity_, 1, misc_mem_type, ref_stream, n_capacity);
            res &= counter_allocator_traits::build(new_ref_count_, 1, misc_mem_type, ref_stream, 1);
            res &= mem_type_allocator_traits::build(new_mem_type_, 1, misc_mem_type, ref_stream, new_mem_type);
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
        size_ = new_size_;
        capacity_ = new_capacity_;
        ref_count_ = new_ref_count_;
        mem_type_ = new_mem_type_;
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
      // Serial: the metadata builds are trivial one-element ops; serial avoids the
      // OpenMP-sections leak/race surface and is faster for this little work.
      operate_with_GPU_stream_from_pointer(
        stream_, ref_stream,
        __ENCAPSULATE__(
          res &= element_allocator_traits::allocate(ptr_, n, mem_type, ref_stream);
          res &= element_allocator_traits::transfer(ptr_, ptr, n, mem_type, mem_type, ref_stream);
          res &= size_allocator_traits::build(size_, 1, exec_mem_type_, ref_stream, n);
          res &= size_allocator_traits::build(capacity_, 1, exec_mem_type_, ref_stream, n);
          res &= counter_allocator_traits::build(ref_count_, 1, exec_mem_type_, ref_stream, __STATIC_CAST__(counter_type, 1));
          res &= mem_type_allocator_traits::build(mem_type_, 1, exec_mem_type_, ref_stream, mem_type);
        )
      );
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
    std_util::swap(size_, other.size_ptr());
    std_util::swap(capacity_, other.capacity_ptr());
    std_util::swap(exec_mem_type_, other.get_exec_memory_type());
    std_util::swap(progenitor_mem_type_, other.get_progenitor_memory_type());
    std_util::swap(mem_type_, other.get_memory_type_ptr());
    std_util::swap(ref_count_, other.counter());
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
    std_util::swap(size_, other.size_ptr());
    std_util::swap(capacity_, other.capacity_ptr());
    std_util::swap(exec_mem_type_, other.get_exec_memory_type());
    std_util::swap(progenitor_mem_type_, other.get_progenitor_memory_type());
    std_util::swap(mem_type_, other.get_memory_type_ptr());
    std_util::swap(ref_count_, other.counter());
    std_util::swap(stream_, other.gpu_stream());
  }

  template<typename T, IvyPointerType IPT> __HOST_DEVICE__ IvyUnifiedPtr<T, IPT>::counter_type IvyUnifiedPtr<T, IPT>::use_count() const{
    if (!ref_count_) return 0;
    constexpr IvyMemoryType def_mem_type = IvyMemoryHelpers::get_execution_default_memory();
    if (exec_mem_type_ == def_mem_type){
      return std_atomic::atomic_ref<counter_type>(*ref_count_).load(std_atomic::memory_order_acquire);
    }
    else{
      counter_type* p_ref_count_ = nullptr;
      {
        operate_with_GPU_stream_from_pointer(
          stream_, ref_stream,
          __ENCAPSULATE__(
            p_ref_count_ = counter_allocator_traits::allocate(1, def_mem_type, ref_stream);
            counter_allocator_traits::transfer(p_ref_count_, ref_count_, 1, def_mem_type, exec_mem_type_, ref_stream);
          )
        );
      }
      counter_type ret = *p_ref_count_;
      {
        operate_with_GPU_stream_from_pointer(
          stream_, ref_stream,
          __ENCAPSULATE__(
            counter_allocator_traits::destroy(p_ref_count_, 1, def_mem_type, ref_stream);
          )
        );
      }
      return ret;
    }
  }
  template<typename T, IvyPointerType IPT> __HOST_DEVICE__ IvyUnifiedPtr<T, IPT>::counter_type IvyUnifiedPtr<T, IPT>::inc_dec_counter(bool do_inc){
    if (!ref_count_){
      __PRINT_ERROR__("IvyUnifiedPtr::inc_dec_counter() failed: ref_count_ is null.\n");
      assert(false);
    }
    constexpr IvyMemoryType def_mem_type = IvyMemoryHelpers::get_execution_default_memory();
    if (exec_mem_type_ == def_mem_type){
      // The counter lives in directly-addressable memory of the current execution space.
      // Mutate it atomically so concurrent copies/destructions of shared_ptr are race-free.
      // Increments may use relaxed ordering; the decrement must be acquire/release so that
      // the thread observing the transition to zero sees all prior writes before freeing.
      // std_atomic::atomic_ref maps to cuda::atomic_ref (CUDA) or std::atomic_ref (host),
      // so both execution spaces share the same ordering guarantees.
      std_atomic::atomic_ref<counter_type> a_ref_count(*ref_count_);
      if (do_inc) return a_ref_count.fetch_add(__STATIC_CAST__(counter_type, 1), std_atomic::memory_order_relaxed);
      else return a_ref_count.fetch_sub(__STATIC_CAST__(counter_type, 1), std_atomic::memory_order_acq_rel);
    }
    else{
      counter_type prev = 0;
      counter_type* p_ref_count_ = nullptr;
      operate_with_GPU_stream_from_pointer(
        stream_, ref_stream,
        __ENCAPSULATE__(
          p_ref_count_ = counter_allocator_traits::allocate(1, def_mem_type, ref_stream);
          counter_allocator_traits::transfer(p_ref_count_, ref_count_, 1, def_mem_type, exec_mem_type_, ref_stream);
          prev = *p_ref_count_;
          if (do_inc) ++(*p_ref_count_);
          else --(*p_ref_count_);
          counter_allocator_traits::transfer(ref_count_, p_ref_count_, 1, exec_mem_type_, def_mem_type, ref_stream);
          counter_allocator_traits::deallocate(p_ref_count_, 1, def_mem_type, ref_stream);
        )
      );
      return prev;
    }
  }
  template<typename T, IvyPointerType IPT> __HOST_DEVICE__ void IvyUnifiedPtr<T, IPT>::inc_dec_size(bool do_inc){
    if (!size_){
      __PRINT_ERROR__("IvyUnifiedPtr::inc_dec_size() failed: size_ is null.\n");
      assert(false);
    }
    constexpr IvyMemoryType def_mem_type = IvyMemoryHelpers::get_execution_default_memory();
    if (exec_mem_type_ == def_mem_type){
      if (do_inc) ++(*size_);
      else --(*size_);
    }
    else{
      size_type* p_size_ = nullptr;
      operate_with_GPU_stream_from_pointer(
        stream_, ref_stream,
        __ENCAPSULATE__(
          p_size_ = size_allocator_traits::allocate(1, def_mem_type, ref_stream);
          size_allocator_traits::transfer(p_size_, size_, 1, def_mem_type, exec_mem_type_, ref_stream);
          if (do_inc) ++(*p_size_);
          else --(*p_size_);
          size_allocator_traits::transfer(size_, p_size_, 1, exec_mem_type_, def_mem_type, ref_stream);
          size_allocator_traits::deallocate(p_size_, 1, def_mem_type, ref_stream);
        )
      );
    }
  }
  template<typename T, IvyPointerType IPT> __HOST_DEVICE__ void IvyUnifiedPtr<T, IPT>::inc_dec_capacity(bool do_inc, size_type inc){
    if (!capacity_){
      __PRINT_ERROR__("IvyUnifiedPtr::inc_dec_capacity() failed: capacity_ is null.\n");
      assert(false);
    }
    constexpr IvyMemoryType def_mem_type = IvyMemoryHelpers::get_execution_default_memory();
    if (exec_mem_type_ == def_mem_type){
      if (do_inc) ++(*capacity_);
      else --(*capacity_);
    }
    else{
      size_type* p_capacity_ = nullptr;
      operate_with_GPU_stream_from_pointer(
        stream_, ref_stream,
        __ENCAPSULATE__(
          p_capacity_ = size_allocator_traits::allocate(1, def_mem_type, ref_stream);
          size_allocator_traits::transfer(p_capacity_, capacity_, 1, def_mem_type, exec_mem_type_, ref_stream);
          if (do_inc) *p_capacity_ += inc;
          else *p_capacity_ -= inc;
          size_allocator_traits::transfer(capacity_, p_capacity_, 1, exec_mem_type_, def_mem_type, ref_stream);
          size_allocator_traits::deallocate(p_capacity_, 1, def_mem_type, ref_stream);
        )
      );
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
