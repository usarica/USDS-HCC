/**
 * @file IvyUnifiedPtr.hh
 * @brief IvyUnifiedPtr declarations, traits, and supporting types.
 */
#ifndef IVYUNIFIEDPTR_HH
#define IVYUNIFIEDPTR_HH


#include "config/IvyCompilerConfig.h"
#include "std_ivy/IvyCstddef.h"
#include "std_ivy/IvyTypeTraits.h"
#include "std_ivy/memory/IvyAllocator.h"
#include "std_ivy/memory/IvyPointerTraits.h"
//#include "std_ivy/IvyFunctional.h"
#include "IvySecrets.h"
#include "IvyPrintout.h"


namespace std_ivy{
  /**
   * @brief Ownership mode for IvyUnifiedPtr.
   */
  enum class IvyPointerType{
    /** @brief Shared ownership with reference counting. */
    shared,
    /** @brief Unique ownership with write-access transfer semantics. */
    unique
  };

  /**
   * @brief Coalesced control block holding all IvyUnifiedPtr metadata in a single allocation.
   *
   * Storing the reference count, size, capacity, and memory-domain descriptor contiguously
   * lets every "fetch metadata" operation move one object instead of four scattered ones,
   * shrinks the owner's footprint, and improves cache locality on the default-memory path.
   * The type is intentionally independent of the element type so that copies/moves between
   * compatible IvyUnifiedPtr specializations (different element types) share the same block.
   */
  struct IvyUnifiedPtrControlBlock{
    /** @brief Shared reference count. */
    IvyTypes::size_t ref_count;
    /** @brief Current logical element count. */
    IvyTypes::size_t size;
    /** @brief Current allocated capacity. */
    IvyTypes::size_t capacity;
    /** @brief Memory domain of the managed storage. */
    IvyMemoryType mem_type;
  };

  /** @brief Forward declaration of the unified smart pointer template. */
  template<typename T, IvyPointerType IPT> class IvyUnifiedPtr;
  /** @brief Transfer-memory primitive specialization for IvyUnifiedPtr types. */
  template<typename T, IvyPointerType IPT> class transfer_memory_primitive<IvyUnifiedPtr<T, IPT>> : public transfer_memory_primitive_with_internal_memory<IvyUnifiedPtr<T, IPT>>{};

  /**
   * @brief Unified host/device-aware pointer with shared or unique ownership and memory-domain metadata.
   * @tparam T Element type.
   * @tparam IPT Ownership policy.
   */
  template<typename T, IvyPointerType IPT> class IvyUnifiedPtr{
  public:
    /** @brief Element type. */
    typedef T element_type;
    /** @brief Raw pointer type. */
    typedef T* pointer;
    /** @brief Element reference type. */
    typedef T& reference;
    /** @brief Unsigned size/index type. */
    typedef IvyTypes::size_t size_type;
    /** @brief Reference-counter storage type. */
    typedef IvyTypes::size_t counter_type;
    /** @brief Pointer-difference type. */
    typedef size_type difference_type;
    /** @brief Allocator for element storage. */
    typedef std_ivy::allocator<element_type> element_allocator_type;
    /** @brief Allocator for size metadata. */
    typedef std_ivy::allocator<size_type> size_allocator_type;
    /** @brief Allocator for reference-counter metadata. */
    typedef std_ivy::allocator<counter_type> counter_allocator_type;
    /** @brief Allocator for memory-type metadata. */
    typedef std_ivy::allocator<IvyMemoryType> mem_type_allocator_type;
    /** @brief Traits for @ref element_allocator_type. */
    typedef std_ivy::allocator_traits<element_allocator_type> element_allocator_traits;
    /** @brief Traits for @ref size_allocator_type. */
    typedef std_ivy::allocator_traits<size_allocator_type> size_allocator_traits;
    /** @brief Traits for @ref counter_allocator_type. */
    typedef std_ivy::allocator_traits<counter_allocator_type> counter_allocator_traits;
    /** @brief Traits for @ref mem_type_allocator_type. */
    typedef std_ivy::allocator_traits<mem_type_allocator_type> mem_type_allocator_traits;

    /** @brief Coalesced metadata control-block type (element-type independent). */
    typedef IvyUnifiedPtrControlBlock control_block_type;
    /** @brief Allocator for the control block. */
    typedef std_ivy::allocator<control_block_type> control_block_allocator_type;
    /** @brief Traits for @ref control_block_allocator_type. */
    typedef std_ivy::allocator_traits<control_block_allocator_type> control_block_allocator_traits;

    /** @brief Rebind to a different element type while preserving ownership policy. */
    template<typename U> using rebind = IvyUnifiedPtr<U, IPT>;

    friend class kernel_generic_transfer_internal_memory<IvyUnifiedPtr<T, IPT>>;
    friend class IvySecrets::dump_helper;

  protected:
    /** @brief Raw managed pointer. */
    pointer ptr_;
    /** @brief Single coalesced control block holding ref count, size, capacity, and memory type. */
    control_block_type* cblock_;
    /** @brief Optional stream used for asynchronous memory operations. */
    IvyGPUStream* stream_;
    /** @brief Execution-context default memory domain. */
    IvyMemoryType exec_mem_type_;
    /** @brief Memory domain from which write access is considered valid. */
    IvyMemoryType progenitor_mem_type_;

    /** @brief Whether the control block lives in memory directly addressable from the current execution space. */
    __INLINE_FCN_RELAXED__ __HOST_DEVICE__ bool control_block_is_addressable() const;
    /** @brief Read a single trivially-copyable metadata field, using a stack temporary (no heap) and skipping the copy when directly addressable. */
    template<typename U> __HOST_DEVICE__ U read_meta_field(U* p_field) const;
    /** @brief Write a single trivially-copyable metadata field, using a stack temporary (no heap) and skipping the copy when directly addressable. */
    template<typename U> __HOST_DEVICE__ void write_meta_field(U* p_field, U const& val);
    /** @brief Snapshot the entire control block in one transfer into a stack copy (batched accessor). */
    __HOST_DEVICE__ control_block_type load_control_block() const;
    /** @brief Write the entire control block in one transfer from a stack copy (batched accessor). */
    __HOST_DEVICE__ void store_control_block(control_block_type const& cb);

    /** @brief Initialize metadata members for an already assigned pointer. */
    __HOST_DEVICE__ void init_members(IvyMemoryType mem_type, size_type n_size, size_type n_capacity);
    /** @brief Release ownership references and underlying resources as needed. */
    __HOST_DEVICE__ void release();
    /** @brief Reset this instance to an empty, detached state. */
    __INLINE_FCN_RELAXED__ __HOST_DEVICE__ void dump();

    /** @brief Atomically increment or decrement reference count; returns the previous value. */
    __HOST_DEVICE__ counter_type inc_dec_counter(bool do_inc);
    /** @brief Increment or decrement tracked size. */
    __HOST_DEVICE__ void inc_dec_size(bool do_inc);
    /** @brief Increment or decrement tracked capacity. */
    __HOST_DEVICE__ void inc_dec_capacity(bool do_inc, size_type inc=1);

    /** @brief Transfer internal memory owned by pointed-to elements. */
    __INLINE_FCN_RELAXED__ __HOST_DEVICE__ bool transfer_internal_memory(IvyMemoryType const& new_mem_type, bool release_old);

    /**
    transfer_impl: Implementation for transferring the memory type of the pointer to the new memory type.
    If transfer_all is true, pointers ref_count_ and mem_type_ are also transferred.
    Otherwise, these two pointers are created in the default memory location of the execution space.
    If copy_ptr is true, a new pointer is created.
    If release_old is true, the old pointers are released.
    */
    __HOST_DEVICE__ bool transfer_impl(IvyMemoryType const& new_mem_type, bool transfer_all, bool copy_ptr, bool release_old);

  public:
    /** @brief Default constructor producing an empty pointer wrapper. */
    __HOST_DEVICE__ IvyUnifiedPtr(IvyGPUStream* stream=nullptr);
    /** @brief Null constructor producing an empty pointer wrapper. */
    __HOST_DEVICE__ IvyUnifiedPtr(std_cstddef::nullptr_t, IvyGPUStream* stream=nullptr);
    /** @brief Construct from raw pointer with unit size/capacity metadata. */
    explicit __HOST_DEVICE__ IvyUnifiedPtr(T* ptr, IvyMemoryType mem_type, IvyGPUStream* stream);
    /** @brief Construct from raw pointer and explicit size metadata. */
    explicit __HOST_DEVICE__ IvyUnifiedPtr(T* ptr, size_type n, IvyMemoryType mem_type, IvyGPUStream* stream);
    /** @brief Construct from raw pointer and explicit size/capacity metadata. */
    explicit __HOST_DEVICE__ IvyUnifiedPtr(T* ptr, size_type n_size, size_type n_capacity, IvyMemoryType mem_type, IvyGPUStream* stream);
    template<typename U>
    explicit __HOST_DEVICE__ IvyUnifiedPtr(U* ptr, IvyMemoryType mem_type, IvyGPUStream* stream);
    template<typename U>
    explicit __HOST_DEVICE__ IvyUnifiedPtr(U* ptr, size_type n, IvyMemoryType mem_type, IvyGPUStream* stream);
    template<typename U>
    explicit __HOST_DEVICE__ IvyUnifiedPtr(U* ptr, size_type n_size, size_type n_capacity, IvyMemoryType mem_type, IvyGPUStream* stream);
    template<typename U, IvyPointerType IPU, ENABLE_IF_BOOL((IPU==IPT || IPU==IvyPointerType::unique) && (IS_BASE_OF(T, U) || IS_BASE_OF(U, T)))>
    /** @brief Copy-construct from a compatible IvyUnifiedPtr specialization. */
    __HOST_DEVICE__ IvyUnifiedPtr(IvyUnifiedPtr<U, IPU> const& other);
    /** @brief Copy-construct from the same IvyUnifiedPtr type. */
    __HOST_DEVICE__ IvyUnifiedPtr(IvyUnifiedPtr<T, IPT> const& other);
    template<typename U, IvyPointerType IPU, ENABLE_IF_BOOL((IPU==IPT || IPU==IvyPointerType::unique) && (IS_BASE_OF(T, U) || IS_BASE_OF(U, T)))>
    /** @brief Move-construct from a compatible IvyUnifiedPtr specialization. */
    __HOST_DEVICE__ IvyUnifiedPtr(IvyUnifiedPtr<U, IPU>&& other);
    /** @brief Move-construct from the same IvyUnifiedPtr type. */
    __HOST_DEVICE__ IvyUnifiedPtr(IvyUnifiedPtr&& other);
    /** @brief Destructor releasing managed resources according to ownership policy. */
    __HOST_DEVICE__ ~IvyUnifiedPtr();

    template<typename U, IvyPointerType IPU, ENABLE_IF_BOOL((IPU==IPT || IPU==IvyPointerType::unique) && (IS_BASE_OF(T, U) || IS_BASE_OF(U, T)))>
    /** @brief Assign from a compatible IvyUnifiedPtr specialization. */
    __HOST_DEVICE__ IvyUnifiedPtr<T, IPT>& operator=(IvyUnifiedPtr<U, IPU> const& other);
    /** @brief Assign from the same IvyUnifiedPtr type. */
    __HOST_DEVICE__ IvyUnifiedPtr<T, IPT>& operator=(IvyUnifiedPtr const& other);
    template<typename U, IvyPointerType IPU, ENABLE_IF_BOOL((IPU==IPT || IPU==IvyPointerType::unique) && (IS_BASE_OF(T, U) || IS_BASE_OF(U, T)))>
    /** @brief Move-assign from a compatible IvyUnifiedPtr specialization. */
    __HOST_DEVICE__ IvyUnifiedPtr<T, IPT>& operator=(IvyUnifiedPtr<U, IPU>&& other);
    /** @brief Move-assign from the same IvyUnifiedPtr type. */
    __HOST_DEVICE__ IvyUnifiedPtr<T, IPT>& operator=(IvyUnifiedPtr&& other);
    /** @brief Assign null and reset this pointer. */
    __HOST_DEVICE__ IvyUnifiedPtr<T, IPT>& operator=(std_cstddef::nullptr_t);

    /**
     * @brief Copy @p n values from an external pointer into this pointer via memory-transfer helpers.
     * @return True on success.
     */
    __HOST_DEVICE__ bool copy(T* ptr, size_type n, IvyMemoryType mem_type, IvyGPUStream* stream);

    /** @brief Get progenitor memory type (const). */
    __INLINE_FCN_RELAXED__ __HOST_DEVICE__ IvyMemoryType const& get_progenitor_memory_type() const __NOEXCEPT__;
    /** @brief Get execution memory type (const). */
    __INLINE_FCN_RELAXED__ __HOST_DEVICE__ IvyMemoryType const& get_exec_memory_type() const __NOEXCEPT__;
    /** @brief Get pointer to memory-type metadata (const, points into the control block). */
    __INLINE_FCN_RELAXED__ __HOST_DEVICE__ IvyMemoryType* get_memory_type_ptr() const __NOEXCEPT__;
    /** @brief Get associated stream pointer (const). */
    __INLINE_FCN_RELAXED__ __HOST_DEVICE__ IvyGPUStream* gpu_stream() const __NOEXCEPT__;
    /** @brief Get pointer to size metadata (const, points into the control block). */
    __INLINE_FCN_RELAXED__ __HOST_DEVICE__ size_type* size_ptr() const __NOEXCEPT__;
    /** @brief Get pointer to capacity metadata (const, points into the control block). */
    __INLINE_FCN_RELAXED__ __HOST_DEVICE__ size_type* capacity_ptr() const __NOEXCEPT__;
    /** @brief Get pointer to reference counter (const, points into the control block). */
    __INLINE_FCN_RELAXED__ __HOST_DEVICE__ counter_type* counter() const __NOEXCEPT__;
    /** @brief Get the control block pointer (const). */
    __INLINE_FCN_RELAXED__ __HOST_DEVICE__ control_block_type* control_block() const __NOEXCEPT__;
    /** @brief Get raw managed pointer (const). */
    __INLINE_FCN_RELAXED__ __HOST_DEVICE__ pointer get() const __NOEXCEPT__;

    /** @brief Get progenitor memory type (mutable). */
    __INLINE_FCN_RELAXED__ __HOST_DEVICE__ IvyMemoryType& get_progenitor_memory_type() __NOEXCEPT__;
    /** @brief Get execution memory type (mutable). */
    __INLINE_FCN_RELAXED__ __HOST_DEVICE__ IvyMemoryType& get_exec_memory_type() __NOEXCEPT__;
    /** @brief Get associated stream pointer (mutable). */
    __INLINE_FCN_RELAXED__ __HOST_DEVICE__ IvyGPUStream*& gpu_stream() __NOEXCEPT__;
    /** @brief Get the control block pointer (mutable). */
    __INLINE_FCN_RELAXED__ __HOST_DEVICE__ control_block_type*& control_block() __NOEXCEPT__;
    /** @brief Get raw managed pointer (mutable). */
    __INLINE_FCN_RELAXED__ __HOST_DEVICE__ pointer& get() __NOEXCEPT__;

    /** @brief Return current element count. */
    __HOST_DEVICE__ size_type size() const __NOEXCEPT__;
    /** @brief Return current capacity. */
    __HOST_DEVICE__ size_type capacity() const __NOEXCEPT__;
    /** @brief Return active memory type of managed storage. */
    __HOST_DEVICE__ IvyMemoryType get_memory_type() const __NOEXCEPT__;

    /** @brief Check whether managed storage is empty or null. */
    __INLINE_FCN_FORCE__ __HOST_DEVICE__ bool empty() const __NOEXCEPT__;

    /** @brief Dereference to first element. */
    __HOST_DEVICE__ reference operator*() const __NOEXCEPT__;
    /** @brief Indexed element access. */
    __HOST_DEVICE__ reference operator[](size_type k) const;
    /** @brief Member access to raw pointer. */
    __HOST_DEVICE__ pointer operator->() const __NOEXCEPT__;

    /** @brief Reset to an empty state and release ownership. */
    __HOST_DEVICE__ void reset();
    /** @brief Reset to null. */
    __HOST_DEVICE__ void reset(std_cstddef::nullptr_t);
    /** @brief Reset from a compatible raw pointer with unit size/capacity. */
    template<typename U> __INLINE_FCN_RELAXED__ __HOST_DEVICE__ void reset(U* ptr, IvyMemoryType mem_type, IvyGPUStream* stream);
    /** @brief Reset from a compatible raw pointer with explicit size. */
    template<typename U> __HOST_DEVICE__ void reset(U* ptr, size_type n, IvyMemoryType mem_type, IvyGPUStream* stream);
    /** @brief Reset from a compatible raw pointer with explicit size/capacity. */
    template<typename U> __HOST_DEVICE__ void reset(U* ptr, size_type n_size, size_type n_capacity, IvyMemoryType mem_type, IvyGPUStream* stream);
    /** @brief Reset from a raw pointer with unit size/capacity. */
    __INLINE_FCN_RELAXED__ __HOST_DEVICE__ void reset(T* ptr, IvyMemoryType mem_type, IvyGPUStream* stream);
    /** @brief Reset from a raw pointer with explicit size. */
    __HOST_DEVICE__ void reset(T* ptr, size_type n, IvyMemoryType mem_type, IvyGPUStream* stream);
    /** @brief Reset from a raw pointer with explicit size/capacity. */
    __HOST_DEVICE__ void reset(T* ptr, size_type n_size, size_type n_capacity, IvyMemoryType mem_type, IvyGPUStream* stream);

    /** @brief Swap state with another compatible IvyUnifiedPtr. */
    template<typename U> __HOST_DEVICE__ void swap(IvyUnifiedPtr<U, IPT>& other) __NOEXCEPT__;
    /** @brief Swap state with another IvyUnifiedPtr of the same type. */
    __HOST_DEVICE__ void swap(IvyUnifiedPtr<T, IPT>& other) __NOEXCEPT__;

    /** @brief Return current reference count. */
    __INLINE_FCN_RELAXED__ __HOST_DEVICE__ counter_type use_count() const;
    /** @brief Return true when this pointer is uniquely owned. */
    __INLINE_FCN_RELAXED__ __HOST_DEVICE__ bool unique() const;
    /** @brief Return true when holding a non-null pointer. */
    __INLINE_FCN_RELAXED__ __HOST_DEVICE__ explicit operator bool() const __NOEXCEPT__;

    /**
    transfer: Transfers the memory type of the pointer to the new memory type.
    If transfer_all is true, pointers ref_count_ and mem_type_ are also transferred.
    Otherwise, these two pointers are created in the default memory location of the execution space.

    See also transfer_impl for the internal implementation, allowing the user to also make a new copy of the pointer.
    */
    /**
     * @brief Transfer pointer storage and metadata to a new memory domain.
     * @param new_mem_type Destination memory domain.
     * @param transfer_all Whether metadata pointers also move to destination domain.
     * @param release_old Whether to release old allocations after transfer.
     * @return True on success.
     */
    __HOST__ bool transfer(IvyMemoryType const& new_mem_type, bool transfer_all, bool release_old);

    /** @brief Reserve storage capacity preserving current memory type and stream. */
    __HOST_DEVICE__ void reserve(size_type const& n);
    /** @brief Reserve storage capacity in a specified memory domain and stream context. */
    __HOST_DEVICE__ void reserve(size_type const& n, IvyMemoryType new_mem_type, IvyGPUStream* stream);
    /** @brief Construct and append an element at the end. */
    template<typename... Args> __INLINE_FCN_RELAXED__ __HOST_DEVICE__ void emplace_back(Args&&... args);
    /** @brief Append an element copy/move at the end. */
    template<typename... Args> __INLINE_FCN_RELAXED__ __HOST_DEVICE__ void push_back(Args&&... args);
    /** @brief Insert an element at position @p i. */
    template<typename... Args> __HOST_DEVICE__ void insert(size_type const& i, Args&&... args);
    /** @brief Remove the last element. */
    __HOST_DEVICE__ void pop_back();
    /** @brief Remove the element at position @p i. */
    __HOST_DEVICE__ void erase(size_type const& i);
    /** @brief Shrink capacity to match size. */
    __HOST_DEVICE__ void shrink_to_fit();

    /** @brief Non-terminating write-access check in current progenitor context. */
    __INLINE_FCN_FORCE__ __HOST_DEVICE__ bool check_write_access() const;
    /** @brief Non-terminating write-access check against a specific memory type. */
    __INLINE_FCN_FORCE__ __HOST_DEVICE__ bool check_write_access(IvyMemoryType const& mem_type) const;
    /** @brief Terminating write-access check in current progenitor context. */
    __INLINE_FCN_FORCE__ __HOST_DEVICE__ void check_write_access_or_die() const;
    /** @brief Terminating write-access check against a specific memory type. */
    __INLINE_FCN_FORCE__ __HOST_DEVICE__ void check_write_access_or_die(IvyMemoryType const& mem_type) const;
  };

  /** @brief Shared-ownership IvyUnifiedPtr alias. */
  template<typename T> using shared_ptr = IvyUnifiedPtr<T, IvyPointerType::shared>;
  /** @brief Unique-ownership IvyUnifiedPtr alias. */
  template<typename T> using unique_ptr = IvyUnifiedPtr<T, IvyPointerType::unique>;
  /** @brief Memory-view alias over IvyUnifiedPtr objects. */
  template<typename T, IvyPointerType IPT> using unifiedptr_view = std_ivy::memview<IvyUnifiedPtr<T, IPT>>;

  /** @brief Memory-view alias for shared_ptr. */
  template<typename T> using sharedptr_view = unifiedptr_view<T, IvyPointerType::shared>;
  /** @brief Memory-view alias for unique_ptr. */
  template<typename T> using uniqueptr_view = unifiedptr_view<T, IvyPointerType::unique>;
  /** @brief Create a memview over an IvyUnifiedPtr instance. */
  template<typename T, IvyPointerType IPT>
  __HOST_DEVICE__ unifiedptr_view<T, IPT> view(IvyUnifiedPtr<T, IPT> const& ptr);

  /** @brief Equality comparison between compatible IvyUnifiedPtr instances. */
  template<typename T, typename U, IvyPointerType IPT, IvyPointerType IPU> __HOST_DEVICE__ bool operator==(IvyUnifiedPtr<T, IPT> const& a, IvyUnifiedPtr<U, IPU> const& b) __NOEXCEPT__;
  /** @brief Inequality comparison between compatible IvyUnifiedPtr instances. */
  template<typename T, typename U, IvyPointerType IPT, IvyPointerType IPU> __HOST_DEVICE__ bool operator!=(IvyUnifiedPtr<T, IPT> const& a, IvyUnifiedPtr<U, IPU> const& b) __NOEXCEPT__;

  /** @brief Equality comparison between IvyUnifiedPtr and raw pointer. */
  template<typename T, IvyPointerType IPT> __HOST_DEVICE__ bool operator==(IvyUnifiedPtr<T, IPT> const& a, T* ptr) __NOEXCEPT__;
  /** @brief Inequality comparison between IvyUnifiedPtr and raw pointer. */
  template<typename T, IvyPointerType IPT> __HOST_DEVICE__ bool operator!=(IvyUnifiedPtr<T, IPT> const& a, T* ptr) __NOEXCEPT__;

  /** @brief Equality comparison between raw pointer and IvyUnifiedPtr. */
  template<typename T, IvyPointerType IPT> __HOST_DEVICE__ bool operator==(T* ptr, IvyUnifiedPtr<T, IPT> const& a) __NOEXCEPT__;
  /** @brief Inequality comparison between raw pointer and IvyUnifiedPtr. */
  template<typename T, IvyPointerType IPT> __HOST_DEVICE__ bool operator!=(T* ptr, IvyUnifiedPtr<T, IPT> const& a) __NOEXCEPT__;

  /** @brief Equality comparison between IvyUnifiedPtr and nullptr. */
  template<typename T, IvyPointerType IPT> __HOST_DEVICE__ bool operator==(IvyUnifiedPtr<T, IPT> const& a, std_cstddef::nullptr_t) __NOEXCEPT__;
  /** @brief Inequality comparison between IvyUnifiedPtr and nullptr. */
  template<typename T, IvyPointerType IPT> __HOST_DEVICE__ bool operator!=(IvyUnifiedPtr<T, IPT> const& a, std_cstddef::nullptr_t) __NOEXCEPT__;

  /** @brief Equality comparison between nullptr and IvyUnifiedPtr. */
  template<typename T, IvyPointerType IPT> __HOST_DEVICE__ bool operator==(std_cstddef::nullptr_t, IvyUnifiedPtr<T, IPT> const& a) __NOEXCEPT__;
  /** @brief Inequality comparison between nullptr and IvyUnifiedPtr. */
  template<typename T, IvyPointerType IPT> __HOST_DEVICE__ bool operator!=(std_cstddef::nullptr_t, IvyUnifiedPtr<T, IPT> const& a) __NOEXCEPT__;

  /** @brief Build IvyUnifiedPtr with explicit allocator and size/capacity. */
  template<typename T, IvyPointerType IPT, typename Allocator_t, typename... Args>
  __HOST_DEVICE__ IvyUnifiedPtr<T, IPT> build_unified(Allocator_t const& a, typename IvyUnifiedPtr<T, IPT>::size_type n_size, typename IvyUnifiedPtr<T, IPT>::size_type n_capacity, IvyMemoryType mem_type, IvyGPUStream* stream, Args&&... args);
  /** @brief Build shared_ptr with explicit allocator and size/capacity. */
  template<typename T, typename Allocator_t, typename... Args>
  __HOST_DEVICE__ shared_ptr<T> build_shared(Allocator_t const& a, typename shared_ptr<T>::size_type n_size, typename shared_ptr<T>::size_type n_capacity, IvyMemoryType mem_type, IvyGPUStream* stream, Args&&... args);
  /** @brief Build unique_ptr with explicit allocator and size/capacity. */
  template<typename T, typename Allocator_t, typename... Args>
  __HOST_DEVICE__ unique_ptr<T> build_unique(Allocator_t const& a, typename unique_ptr<T>::size_type n_size, typename unique_ptr<T>::size_type n_capacity, IvyMemoryType mem_type, IvyGPUStream* stream, Args&&... args);
  /** @brief Make IvyUnifiedPtr with default allocator and size/capacity. */
  template<typename T, IvyPointerType IPT, typename... Args>
  __HOST_DEVICE__ IvyUnifiedPtr<T, IPT> make_unified(typename IvyUnifiedPtr<T, IPT>::size_type n_size, typename IvyUnifiedPtr<T, IPT>::size_type n_capacity, IvyMemoryType mem_type, IvyGPUStream* stream, Args&&... args);
  /** @brief Make shared_ptr with default allocator and size/capacity. */
  template<typename T, typename... Args>
  __HOST_DEVICE__ shared_ptr<T> make_shared(typename shared_ptr<T>::size_type n_size, typename shared_ptr<T>::size_type n_capacity, IvyMemoryType mem_type, IvyGPUStream* stream, Args&&... args);
  /** @brief Make unique_ptr with default allocator and size/capacity. */
  template<typename T, typename... Args>
  __HOST_DEVICE__ unique_ptr<T> make_unique(typename unique_ptr<T>::size_type n_size, typename unique_ptr<T>::size_type n_capacity, IvyMemoryType mem_type, IvyGPUStream* stream, Args&&... args);

  /** @brief Build IvyUnifiedPtr with explicit allocator and size only. */
  template<typename T, IvyPointerType IPT, typename Allocator_t, typename... Args>
  __HOST_DEVICE__ IvyUnifiedPtr<T, IPT> build_unified(Allocator_t const& a, typename IvyUnifiedPtr<T, IPT>::size_type n, IvyMemoryType mem_type, IvyGPUStream* stream, Args&&... args);
  /** @brief Build shared_ptr with explicit allocator and size only. */
  template<typename T, typename Allocator_t, typename... Args>
  __HOST_DEVICE__ shared_ptr<T> build_shared(Allocator_t const& a, typename shared_ptr<T>::size_type n, IvyMemoryType mem_type, IvyGPUStream* stream, Args&&... args);
  /** @brief Build unique_ptr with explicit allocator and size only. */
  template<typename T, typename Allocator_t, typename... Args>
  __HOST_DEVICE__ unique_ptr<T> build_unique(Allocator_t const& a, typename unique_ptr<T>::size_type n, IvyMemoryType mem_type, IvyGPUStream* stream, Args&&... args);
  /** @brief Make IvyUnifiedPtr with default allocator and size only. */
  template<typename T, IvyPointerType IPT, typename... Args>
  __HOST_DEVICE__ IvyUnifiedPtr<T, IPT> make_unified(typename IvyUnifiedPtr<T, IPT>::size_type n, IvyMemoryType mem_type, IvyGPUStream* stream, Args&&... args);
  /** @brief Make shared_ptr with default allocator and size only. */
  template<typename T, typename... Args>
  __HOST_DEVICE__ shared_ptr<T> make_shared(typename shared_ptr<T>::size_type n, IvyMemoryType mem_type, IvyGPUStream* stream, Args&&... args);
  /** @brief Make unique_ptr with default allocator and size only. */
  template<typename T, typename... Args>
  __HOST_DEVICE__ unique_ptr<T> make_unique(typename unique_ptr<T>::size_type n, IvyMemoryType mem_type, IvyGPUStream* stream, Args&&... args);

  /** @brief Build IvyUnifiedPtr with explicit allocator and default size semantics. */
  template<typename T, IvyPointerType IPT, typename Allocator_t, typename... Args>
  __HOST_DEVICE__ IvyUnifiedPtr<T, IPT> build_unified(Allocator_t const& a, IvyMemoryType mem_type, IvyGPUStream* stream, Args&&... args);
  /** @brief Build shared_ptr with explicit allocator and default size semantics. */
  template<typename T, typename Allocator_t, typename... Args>
  __HOST_DEVICE__ shared_ptr<T> build_shared(Allocator_t const& a, IvyMemoryType mem_type, IvyGPUStream* stream, Args&&... args);
  /** @brief Build unique_ptr with explicit allocator and default size semantics. */
  template<typename T, typename Allocator_t, typename... Args>
  __HOST_DEVICE__ unique_ptr<T> build_unique(Allocator_t const& a, IvyMemoryType mem_type, IvyGPUStream* stream, Args&&... args);
  /** @brief Make IvyUnifiedPtr with default allocator and default size semantics. */
  template<typename T, IvyPointerType IPT, typename... Args>
  __HOST_DEVICE__ IvyUnifiedPtr<T, IPT> make_unified(IvyMemoryType mem_type, IvyGPUStream* stream, Args&&... args);
  /** @brief Make shared_ptr with default allocator and default size semantics. */
  template<typename T, typename... Args>
  __HOST_DEVICE__ shared_ptr<T> make_shared(IvyMemoryType mem_type, IvyGPUStream* stream, Args&&... args);
  /** @brief Make unique_ptr with default allocator and default size semantics. */
  template<typename T, typename... Args>
  __HOST_DEVICE__ unique_ptr<T> make_unique(IvyMemoryType mem_type, IvyGPUStream* stream, Args&&... args);

  /**
  Specialization of pointer_traits:
  There is no pointer_to for IvyUnifiedPtr since this class is for shared/unique pointers with ownership.
  */
  /**
   * @brief pointer_traits specialization for IvyUnifiedPtr.
   * @tparam T Element type.
   * @tparam IPT Ownership policy.
   */
  template<typename T, IvyPointerType IPT> class pointer_traits<IvyUnifiedPtr<T, IPT>>{
  public:
    /** @brief Ownership policy associated with this pointer type. */
    static constexpr IvyPointerType ivy_ptr_type = IPT;
    /** @brief Pointer type alias. */
    typedef IvyUnifiedPtr<T, IPT> pointer;
    /** @brief Element type alias. */
    typedef typename pointer::element_type element_type;
    /** @brief Difference type alias. */
    typedef typename pointer::difference_type difference_type;
    /** @brief Rebound pointer_traits alias. */
    template <typename U> using rebind = pointer_traits_rebind_t<pointer, U>;
  };

  /**
  Specialization of the value printout routine
  */
  template<typename T, IvyPointerType IPT> struct value_printout<IvyUnifiedPtr<T, IPT>>;
}
namespace IvyTypes{
  template<typename T, std_ivy::IvyPointerType IPT> struct convert_to_floating_point<std_ivy::IvyUnifiedPtr<T, IPT>>{
    using type = std_ivy::IvyUnifiedPtr<convert_to_floating_point_t<T>, IPT>;
  };
}

// Extension of std_fcnal::hash
// Current CUDA C++ library omits hashes, so we defer it as well.
/**
namespace std_fcnal{
  template<typename T, std_ivy::IvyPointerType IPT> struct hash<std_ivy::IvyUnifiedPtr<T, IPT>>{
    using argument_type = std_ivy::IvyUnifiedPtr<T, IPT>;
    using arg_ptr_t = typename std_ivy::IvyUnifiedPtr<T, IPT>::pointer;
    using result_type = typename hash<arg_ptr_t>::result_type;

    __HOST_DEVICE__ result_type operator()(argument_type const& arg) const{ return hash<arg_ptr_t>{}(arg.get()); }
  };
}
*/

/**
Extension of std_util::swap
*/
namespace std_util{
  /** @brief Swap two compatible IvyUnifiedPtr instances. */
  template<typename T, typename U, std_ivy::IvyPointerType IPT> __HOST_DEVICE__ void swap(std_ivy::IvyUnifiedPtr<T, IPT>& a, std_ivy::IvyUnifiedPtr<U, IPT>& b) __NOEXCEPT__;
  /** @brief Swap two IvyUnifiedPtr instances of identical type. */
  template<typename T, std_ivy::IvyPointerType IPT> __HOST_DEVICE__ void swap(std_ivy::IvyUnifiedPtr<T, IPT>& a, std_ivy::IvyUnifiedPtr<T, IPT>& b) __NOEXCEPT__;
}


#endif
