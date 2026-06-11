/**
 * @file IvyAllocator.h
 * @brief Allocation, deallocation, and memory-transfer primitives used by std_ivy containers.
 */
#ifndef IVYALLOCATOR_H
#define IVYALLOCATOR_H


#include "config/IvyCompilerConfig.h"
#include "IvyMemoryHelpers.h"
#include "std_ivy/IvyUtility.h"
#include "std_ivy/IvyLimits.h"
#include "std_ivy/memory/IvyAddressof.h"


namespace std_ivy{
  /**
  Base class of allocator primitives
  */
  template<typename T> class allocation_type_properties{
  public:
    /** @brief Value type handled by this allocation property specialization. */
    typedef T value_type;
    /** @brief Mutable pointer to value_type. */
    typedef T* pointer;
    /** @brief Const pointer to value_type. */
    typedef T const* const_pointer;
    /** @brief Mutable reference to value_type. */
    typedef T& reference;
    /** @brief Const reference to value_type. */
    typedef T const& const_reference;
    /** @brief Unsigned size/index type. */
    typedef IvyTypes::size_t size_type;
    /** @brief Signed pointer-difference type. */
    typedef IvyTypes::ptrdiff_t difference_type;

    /** @brief Get the address of a mutable reference. */
    static __INLINE_FCN_RELAXED__ __HOST_DEVICE__ pointer address(reference x){ return addressof(x); }
    /** @brief Get the address of a const reference. */
    static __INLINE_FCN_RELAXED__ __HOST_DEVICE__ const_pointer address(const_reference x){ return addressof(x); }
    /** @brief Return maximum representable element count for this allocator value type. */
    static __INLINE_FCN_RELAXED__ __HOST_DEVICE__ size_type max_size() noexcept{
      return std_limits::numeric_limits<size_type>::max() / sizeof(T);
    }
  };
  template<typename T> class allocation_type_properties<T const>{
  public:
    /** @brief Value type handled by this const allocation property specialization. */
    typedef T const value_type;
    /** @brief Const pointer to value_type. */
    typedef T const* pointer;
    /** @brief Const pointer to value_type. */
    typedef T const* const_pointer;
    /** @brief Const reference to value_type. */
    typedef T const& reference;
    /** @brief Const reference to value_type. */
    typedef T const& const_reference;
    /** @brief Unsigned size/index type. */
    typedef IvyTypes::size_t size_type;
    /** @brief Signed pointer-difference type. */
    typedef IvyTypes::ptrdiff_t difference_type;

    /** @brief Get the address of a const reference. */
    static __INLINE_FCN_RELAXED__ __HOST_DEVICE__ pointer address(reference x){ return addressof(x); }
    /** @brief Return maximum representable element count for this allocator value type. */
    static __INLINE_FCN_RELAXED__ __HOST_DEVICE__ size_type max_size() noexcept{
      return std_limits::numeric_limits<size_type>::max() / sizeof(T);
    }
  };

  /**
  allocator primitives
  */
  template<typename T> class allocator_primitive{
  public:
    /** @brief Base property type for allocator primitives. */
    using base_t = allocation_type_properties<T>;
    /** @brief Pointer type to allocator element values. */
    using pointer = typename base_t::pointer;
    /** @brief Unsigned size/index type. */
    using size_type = typename base_t::size_type;

    /**
     * @brief Allocate storage for `n` elements into `tgt`.
     * @param tgt Output pointer.
     * @param n Element count.
     * @param mem_type Target memory-domain.
     * @param stream Stream used by backend allocation path.
     * @return True on success.
     */
    static __INLINE_FCN_RELAXED__ __HOST_DEVICE__ bool allocate(
      pointer& tgt, size_type n, IvyMemoryType mem_type, IvyGPUStream& stream
    ){
      return IvyMemoryHelpers::allocate_memory(tgt, n, mem_type, stream);
    }
    /**
     * @brief Allocate storage and return pointer.
     * @param n Element count.
     * @param mem_type Target memory-domain.
     * @param stream Stream used by backend allocation path.
     * @return Allocated pointer or nullptr on failure path.
     */
    static __HOST_DEVICE__ pointer allocate(
      size_type n, IvyMemoryType mem_type, IvyGPUStream& stream
    ){
      pointer res = nullptr;
      IvyMemoryHelpers::allocate_memory(res, n, mem_type, stream);
      return res;
    }

    /** @brief Construct `n` elements in pre-allocated storage. */
    template<typename... Args> static __INLINE_FCN_RELAXED__ __HOST_DEVICE__ bool construct(
      pointer& tgt, size_type n, IvyMemoryType mem_type, IvyGPUStream& stream, Args&&... args
    ){
      return IvyMemoryHelpers::construct(tgt, n, mem_type, stream, args...);
    }

    /** @brief Allocate and construct values into `tgt`. */
    template<typename... Args> static __INLINE_FCN_RELAXED__ __HOST_DEVICE__ bool build(
      pointer& tgt, size_type n, IvyMemoryType mem_type, IvyGPUStream& stream, Args&&... args
    ){
      return IvyMemoryHelpers::build(tgt, n, mem_type, stream, args...);
    }
    /** @brief Allocate and construct values, returning the new pointer. */
    template<typename... Args> static __HOST_DEVICE__ pointer build(
      size_type n, IvyMemoryType mem_type, IvyGPUStream& stream, Args&&... args
    ){
      pointer res = nullptr;
      IvyMemoryHelpers::build(res, n, mem_type, stream, args...);
      return res;
    }
  };
  template<typename T> class deallocator_primitive{
  public:
    /** @brief Base property type for deallocator primitives. */
    using base_t = allocation_type_properties<T>;
    /** @brief Pointer type to allocator element values. */
    using pointer = typename base_t::pointer;
    /** @brief Unsigned size/index type. */
    using size_type = typename base_t::size_type;

    /** @brief Deallocate previously allocated storage. */
    static __INLINE_FCN_RELAXED__ __HOST_DEVICE__ bool deallocate(
      pointer& p, size_type n, IvyMemoryType mem_type, IvyGPUStream& stream
    ){
      return IvyMemoryHelpers::free_memory(p, n, mem_type, stream);
    }

    /** @brief Run destructors on stored elements without deallocating storage. */
    static __INLINE_FCN_RELAXED__ __HOST_DEVICE__ bool destruct(
      pointer& p, size_type n, IvyMemoryType mem_type, IvyGPUStream& stream
    ){
      return IvyMemoryHelpers::destruct(p, n, mem_type, stream);
    }

    /** @brief Destruct elements and deallocate storage. */
    static __INLINE_FCN_RELAXED__ __HOST_DEVICE__ bool destroy(
      pointer& p, size_type n, IvyMemoryType mem_type, IvyGPUStream& stream
    ){
      return IvyMemoryHelpers::destroy(p, n, mem_type, stream);
    }
  };

  /**
  We hold the convention to codfy all classes with internal data using a public or protected
  'bool transfer_internal_memory(IvyMemoryType const& new_mem_type)' member function.
  */
  template<typename T, typename = void> class kernel_generic_transfer_internal_memory;
  template<typename T> class transfer_memory_primitive_without_internal_memory;
  template<typename T, typename = void> class transfer_memory_primitive_with_internal_memory;
  template<typename T> class transfer_memory_primitive;
  /**
    transfer_memory_primitive<std_util::pair<T, U>>:
    std_util::pair is a container with internal memory, so we can use this object safely only when internal memory transfer is ensured.
    This specialization ensures that internal memory transfer routines are called for the first and second elements of the pair.
  */
  template<typename T, typename U> class transfer_memory_primitive<std_util::pair<T, U>>;

  template<typename T, typename Enabler> class kernel_generic_transfer_internal_memory final : public kernel_base_noprep_nofin{
  protected:
    /** @brief Dispatch element-level internal memory transfer. */
    static __INLINE_FCN_RELAXED__ __HOST_DEVICE__ bool transfer_internal_memory(T* const& ptr, IvyMemoryType const& mem_type, bool release_old){
      return ptr->transfer_internal_memory(mem_type, release_old);
    }

  public:
    /** @brief Kernel entry point for internal-memory transfer over an index range. */
    static __HOST_DEVICE__ void kernel(
      IvyTypes::size_t const& i, IvyTypes::size_t const& n, T* const& ptr,
      IvyMemoryType const& mem_type, bool const& release_old
    ){
      if (kernel_check_dims<kernel_generic_transfer_internal_memory<T>>::check_dims(i, n)) transfer_internal_memory(ptr+i, mem_type, release_old);
    }

    friend class transfer_memory_primitive_with_internal_memory<T, Enabler>;
  };

  template<typename T> class transfer_memory_primitive_without_internal_memory{
  public:
    /** @brief Base property type for transfer primitives. */
    using base_t = allocation_type_properties<T>;
    /** @brief Pointer type to transferred values. */
    using pointer = typename base_t::pointer;
    /** @brief Unsigned size/index type. */
    using size_type = typename base_t::size_type;

    /**
     * @brief No-op internal-memory transfer for trivially external objects.
     * @return Always true.
     */
    static __HOST_DEVICE__ constexpr bool transfer_internal_memory(pointer, IvyTypes::size_t const&, IvyMemoryType const&, IvyMemoryType const&, IvyGPUStream&, bool){
      return true;
    }
    /** @brief Transfer flat object bytes between memory domains. */
    static __INLINE_FCN_RELAXED__ __HOST_DEVICE__ bool transfer(
      pointer& tgt, pointer const& src, size_type n,
      IvyMemoryType type_tgt, IvyMemoryType type_src,
      IvyGPUStream& stream
    ){
      return IvyMemoryHelpers::transfer_memory(tgt, src, n, type_tgt, type_src, stream);
    }
  };
  template<typename T, typename Enabler> class transfer_memory_primitive_with_internal_memory{
  public:
    /** @brief Base property type for transfer primitives. */
    using base_t = allocation_type_properties<T>;
    /** @brief Value type handled by this transfer primitive. */
    using value_type = typename base_t::value_type;
    /** @brief Pointer type to transferred values. */
    using pointer = typename base_t::pointer;
    /** @brief Unsigned size/index type. */
    using size_type = typename base_t::size_type;
    /** @brief Kernel adapter used for internal-memory transfer. */
    using kernel_type = kernel_generic_transfer_internal_memory<value_type, Enabler>;

    /**
     * @brief Transfer nested/internal allocations for each element.
     * @note Uses accelerator kernels when host is orchestrating accelerator memory.
     */
    static __HOST_DEVICE__ bool transfer_internal_memory(pointer ptr, IvyTypes::size_t const& n, IvyMemoryType const& ptr_mem_type, IvyMemoryType const& mem_type, IvyGPUStream& stream, bool release_old){
      bool res = true;
      if (IvyMemoryHelpers::run_acc_on_host(ptr_mem_type)){
        if (!run_kernel<kernel_type>(0, stream).parallel_1D(n, ptr, mem_type, release_old)){
          __PRINT_ERROR__("transfer_memory_primitive::transfer_internal_memory: Unable to call the acc. hardware kernel...\n");
          res = false;
        }
      }
      else{
        // res is accumulated with &= across iterations; under OpenMP this must use a
        // reduction to avoid a data race on the shared accumulator (TSan-confirmed).
#if defined(OPENMP_ENABLED)
        if (n>=NUM_CPU_THREADS_THRESHOLD){
          #pragma omp parallel for schedule(static) reduction(&:res)
          for (size_type i=0; i<n; ++i) res &= kernel_type::transfer_internal_memory(ptr+i, mem_type, release_old);
        }
        else
#endif
        {
          for (size_type i=0; i<n; ++i) res &= kernel_type::transfer_internal_memory(ptr+i, mem_type, release_old);
        }
      }
      return res;
    }

    /**
     * @brief Transfer element storage and then reconcile nested/internal allocations.
     * @note Falls back through default execution memory when direct transfer path is unavailable.
     */
    static __INLINE_FCN_RELAXED__ __HOST_DEVICE__ bool transfer(
      pointer& tgt, pointer const& src, size_type n,
      IvyMemoryType type_tgt, IvyMemoryType type_src,
      IvyGPUStream& stream
    ){
      if (!src) return false;
      bool res = true;
      /**
#if DEVICE_CODE == DEVICE_CODE_HOST
      printf("transfer_memory_primitive_with_internal_memory::transfer: type = %s | n=%llu | src = %p (%d), tgt = %p (%d)\n", typeid(T).name(), n, src, int(type_src), tgt, int(type_tgt));
#endif
      */
      constexpr IvyMemoryType def_mem_type = IvyMemoryHelpers::get_execution_default_memory();
      constexpr bool release_old = false; // We do not release existing memory from internal memory transfers in order to preserve src pointer.
#if defined(__USE_CUDA__)
      if (def_mem_type==type_tgt && def_mem_type==type_src){
#endif
        res &= IvyMemoryHelpers::transfer_memory(tgt, src, n, type_tgt, type_src, stream);
        res &= transfer_internal_memory(tgt, n, type_tgt, type_tgt, stream, release_old);
#if defined(__USE_CUDA__)
      }
      else{
        pointer p_int = nullptr;
        res &= IvyMemoryHelpers::allocate_memory(p_int, n, def_mem_type, stream);
        res &= IvyMemoryHelpers::transfer_memory(p_int, src, n, def_mem_type, type_src, stream);
        res &= transfer_internal_memory(p_int, n, def_mem_type, type_tgt, stream, release_old);
        res &= IvyMemoryHelpers::transfer_memory(tgt, p_int, n, type_tgt, def_mem_type, stream);
        res &= IvyMemoryHelpers::free_memory(p_int, n, def_mem_type, stream);
      }
#endif
      return res;
    }
  };
  // By default, we assume that the class has no internal memory to transfer.
  template<typename T> class transfer_memory_primitive : public transfer_memory_primitive_without_internal_memory<T>{};
  /**
    transfer_memory_primitive<std_util::pair<T, U>>:
    std_util::pair is a container with internal memory, so we can use this object safely only when internal memory transfer is ensured.
    This specialization ensures that internal memory transfer routines are called for the first and second elements of the pair.
  */
  template<typename T, typename U> class transfer_memory_primitive<std_util::pair<T, U>>{
  public:
    /** @brief Base property type for pair transfer primitives. */
    using base_t = allocation_type_properties<std_util::pair<T, U>>;
    /** @brief Pair value type handled by this transfer primitive. */
    using value_type = typename base_t::value_type;
    /** @brief Pointer type to pair values. */
    using pointer = typename base_t::pointer;
    /** @brief Unsigned size/index type. */
    using size_type = typename base_t::size_type;

    using transfer_primitive_T = transfer_memory_primitive<T>;
    using transfer_primitive_U = transfer_memory_primitive<U>;

    /**
     * @brief Transfer internal memory of both pair members element-wise.
     * @note This specialization preserves correctness for pair-held nested state.
     */
    static __HOST_DEVICE__ bool transfer_internal_memory(pointer ptr, IvyTypes::size_t const& n, IvyMemoryType const& ptr_mem_type, IvyMemoryType const& mem_type, IvyGPUStream& stream, bool release_old){
      constexpr IvyMemoryType def_mem_type = IvyMemoryHelpers::get_execution_default_memory();
      bool res = true;
#if defined(__USE_CUDA__)
      if (ptr_mem_type==def_mem_type){
#endif
        for (size_type i=0; i<n; ++i){
          res &= (
            transfer_primitive_T::transfer_internal_memory(&(ptr[i].first), n, ptr_mem_type, mem_type, stream, release_old)
            &&
            transfer_primitive_U::transfer_internal_memory(&(ptr[i].second), n, ptr_mem_type, mem_type, stream, release_old)
            );
        }
#if defined(__USE_CUDA__)
      }
      else{
        pointer p_int = nullptr;
        res &= IvyMemoryHelpers::allocate_memory(p_int, n, def_mem_type, stream);
        res &= IvyMemoryHelpers::transfer_memory(p_int, ptr, n, def_mem_type, ptr_mem_type, stream);
        for (size_type i=0; i<n; ++i){
          res &= (
            transfer_primitive_T::transfer_internal_memory(&(p_int[i].first), n, def_mem_type, mem_type, stream, release_old)
            &&
            transfer_primitive_U::transfer_internal_memory(&(p_int[i].second), n, def_mem_type, mem_type, stream, release_old)
            );
        }
        res &= IvyMemoryHelpers::transfer_memory(ptr, p_int, n, ptr_mem_type, def_mem_type, stream);
        res &= IvyMemoryHelpers::free_memory(p_int, n, def_mem_type, stream);
      }
#endif
      return res;
    }

    /** @brief Transfer pair storage and then pair-member internal allocations. */
    static __INLINE_FCN_RELAXED__ __HOST_DEVICE__ bool transfer(
      pointer& tgt, pointer const& src, size_type n,
      IvyMemoryType type_tgt, IvyMemoryType type_src,
      IvyGPUStream& stream
    ){
      if (!src) return false;
      bool res = true;
      constexpr IvyMemoryType def_mem_type = IvyMemoryHelpers::get_execution_default_memory();
      constexpr bool release_old = false; // We do not release existing memory from internal memory transfers in order to preserve src pointer.
#if defined(__USE_CUDA__)
      if (def_mem_type==type_tgt && def_mem_type==type_src){
#endif
        res &= IvyMemoryHelpers::transfer_memory(tgt, src, n, type_tgt, type_src, stream);
        res &= transfer_internal_memory(tgt, n, type_tgt, type_tgt, stream, release_old);
#if defined(__USE_CUDA__)
      }
      else{
        pointer p_int = nullptr;
        res &= IvyMemoryHelpers::allocate_memory(p_int, n, def_mem_type, stream);
        res &= IvyMemoryHelpers::transfer_memory(p_int, src, n, def_mem_type, type_src, stream);
        res &= transfer_internal_memory(p_int, n, def_mem_type, type_tgt, stream, release_old);
        res &= IvyMemoryHelpers::transfer_memory(tgt, p_int, n, type_tgt, def_mem_type, stream);
        res &= IvyMemoryHelpers::free_memory(p_int, n, def_mem_type, stream);
      }
#endif
      return res;
    }
  };

  /**
  allocator
  */
  template<typename T> class allocator : public allocation_type_properties<T>, public allocator_primitive<T>, public deallocator_primitive<T>, public transfer_memory_primitive<T>{
  public:
    /** @brief Base allocation property type. */
    using base_t = allocation_type_properties<T>;
    /** @brief Element value type. */
    using value_type = typename base_t::value_type;
    /** @brief Mutable pointer type. */
    using pointer = typename base_t::pointer;
    /** @brief Const pointer type. */
    using const_pointer = typename base_t::const_pointer;
    /** @brief Mutable reference type. */
    using reference = typename base_t::reference;
    /** @brief Const reference type. */
    using const_reference = typename base_t::const_reference;
    /** @brief Unsigned size/index type. */
    using size_type = typename base_t::size_type;
    /** @brief Signed pointer-difference type. */
    using difference_type = typename base_t::difference_type;

    /** @brief Default constructor. */
    allocator() noexcept = default;
    /** @brief Copy constructor. */
    __HOST_DEVICE__ allocator(allocator const& other) noexcept{}
    /** @brief Converting copy constructor from compatible allocator type. */
    template<typename U> __HOST_DEVICE__ allocator(allocator<U> const& other) noexcept{}
    /** @brief Destructor. */
    /*__HOST_DEVICE__*/ ~allocator() noexcept = default;
  };
  /** @brief Equality comparison for allocator specializations. */
  template<typename T, typename U> bool operator==(std_ivy::allocator<T> const&, std_ivy::allocator<U> const&) noexcept{ return true; }
  /** @brief Inequality comparison for allocator specializations. */
  template<typename T, typename U> bool operator!=(std_ivy::allocator<T> const& a1, std_ivy::allocator<U> const& a2) noexcept{ return !(a1==a2); }

  /**
  allocator_traits
  */
  template<typename Allocator_t> class allocator_traits{
  public:
    /** @brief Underlying allocator type. */
    typedef Allocator_t allocator_type;
    /** @brief Element value type. */
    typedef typename allocator_type::value_type value_type;
    /** @brief Mutable reference type. */
    typedef typename allocator_type::reference reference;
    /** @brief Const reference type. */
    typedef typename allocator_type::const_reference const_reference;
    /** @brief Mutable pointer type. */
    typedef typename allocator_type::pointer pointer;
    /** @brief Const pointer type. */
    typedef typename allocator_type::const_pointer const_pointer;
    /** @brief Unsigned size/index type. */
    typedef typename allocator_type::size_type size_type;
    /** @brief Signed pointer-difference type. */
    typedef typename allocator_type::difference_type difference_type;

    /** @brief Allocate through allocator instance interface. */
    static __INLINE_FCN_RELAXED__ __HOST_DEVICE__ pointer allocate(allocator_type const& a, size_type n, IvyMemoryType mem_type, IvyGPUStream& stream){
      return a.allocate(n, mem_type, stream);
    }
    static __INLINE_FCN_RELAXED__ __HOST_DEVICE__ bool allocate(allocator_type const& a, pointer& ptr, size_type n, IvyMemoryType mem_type, IvyGPUStream& stream){
      return a.allocate(ptr, n, mem_type, stream);
    }
    template<typename... Args> static __INLINE_FCN_RELAXED__ __HOST_DEVICE__ bool construct(allocator_type const& a, pointer& ptr, size_type n, IvyMemoryType mem_type, IvyGPUStream& stream, Args&&... args){
      return a.construct(ptr, n, mem_type, stream, args...);
    }
    template<typename... Args> static __INLINE_FCN_RELAXED__ __HOST_DEVICE__ pointer build(allocator_type const& a, size_type n, IvyMemoryType mem_type, IvyGPUStream& stream, Args&&... args){
      return a.build(n, mem_type, stream, args...);
    }
    template<typename... Args> static __INLINE_FCN_RELAXED__ __HOST_DEVICE__ bool build(allocator_type const& a, pointer& ptr, size_type n, IvyMemoryType mem_type, IvyGPUStream& stream, Args&&... args){
      return a.build(ptr, n, mem_type, stream, args...);
    }
    static __INLINE_FCN_RELAXED__ __HOST_DEVICE__ bool deallocate(allocator_type const& a, pointer& p, size_type n, IvyMemoryType mem_type, IvyGPUStream& stream){
      return a.deallocate(p, n, mem_type, stream);
    }
    static __INLINE_FCN_RELAXED__ __HOST_DEVICE__ bool destruct(allocator_type const& a, pointer& p, size_type n, IvyMemoryType mem_type, IvyGPUStream& stream){
      return a.destruct(p, n, mem_type, stream);
    }
    static __INLINE_FCN_RELAXED__ __HOST_DEVICE__ bool destroy(allocator_type const& a, pointer& p, size_type n, IvyMemoryType mem_type, IvyGPUStream& stream){
      return a.destroy(p, n, mem_type, stream);
    }
    static __INLINE_FCN_RELAXED__ __HOST_DEVICE__ bool transfer(
      allocator_type const& a,
      pointer& tgt, pointer const& src, size_t n,
      IvyMemoryType type_tgt, IvyMemoryType type_src,
      IvyGPUStream& stream
    ){
      return a.transfer(tgt, src, n, type_tgt, type_src, stream);
    }
    static __INLINE_FCN_RELAXED__ __HOST_DEVICE__ bool transfer_internal_memory(
      allocator_type const& a,
      pointer ptr, IvyTypes::size_t const& n,
      IvyMemoryType const& ptr_mem_type, IvyMemoryType const& mem_type, IvyGPUStream& stream,
      bool release_old
    ){
      return a.transfer_internal_memory(ptr, n, ptr_mem_type, mem_type, stream, release_old);
    }
    static __INLINE_FCN_RELAXED__ __HOST_DEVICE__ size_type max_size(allocator_type const& a) noexcept{
      return a.max_size();
    }

    /** @brief Allocate through allocator static interface. */
    static __INLINE_FCN_RELAXED__ __HOST_DEVICE__ pointer allocate(size_type n, IvyMemoryType mem_type, IvyGPUStream& stream){
      return allocator_type::allocate(n, mem_type, stream);
    }
    static __INLINE_FCN_RELAXED__ __HOST_DEVICE__ bool allocate(pointer& ptr, size_type n, IvyMemoryType mem_type, IvyGPUStream& stream){
      return allocator_type::allocate(ptr, n, mem_type, stream);
    }
    template<typename... Args> static __INLINE_FCN_RELAXED__ __HOST_DEVICE__ bool construct(pointer& ptr, size_type n, IvyMemoryType mem_type, IvyGPUStream& stream, Args&&... args){
      return allocator_type::construct(ptr, n, mem_type, stream, args...);
    }
    template<typename... Args> static __INLINE_FCN_RELAXED__ __HOST_DEVICE__ pointer build(size_type n, IvyMemoryType mem_type, IvyGPUStream& stream, Args&&... args){
      return allocator_type::build(n, mem_type, stream, args...);
    }
    template<typename... Args> static __INLINE_FCN_RELAXED__ __HOST_DEVICE__ bool build(pointer& ptr, size_type n, IvyMemoryType mem_type, IvyGPUStream& stream, Args&&... args){
      return allocator_type::build(ptr, n, mem_type, stream, args...);
    }
    static __INLINE_FCN_RELAXED__ __HOST_DEVICE__ bool deallocate(pointer& p, size_type n, IvyMemoryType mem_type, IvyGPUStream& stream){
      return allocator_type::deallocate(p, n, mem_type, stream);
    }
    static __INLINE_FCN_RELAXED__ __HOST_DEVICE__ bool destruct(pointer& p, size_type n, IvyMemoryType mem_type, IvyGPUStream& stream){
      return allocator_type::destruct(p, n, mem_type, stream);
    }
    static __INLINE_FCN_RELAXED__ __HOST_DEVICE__ bool destroy(pointer& p, size_type n, IvyMemoryType mem_type, IvyGPUStream& stream){
      return allocator_type::destroy(p, n, mem_type, stream);
    }
    /** @brief Transfer element storage through allocator static interface. */
    static __INLINE_FCN_RELAXED__ __HOST_DEVICE__ bool transfer(
      pointer& tgt, pointer const& src, size_t n,
      IvyMemoryType type_tgt, IvyMemoryType type_src,
      IvyGPUStream& stream
    ){
      return allocator_type::transfer(tgt, src, n, type_tgt, type_src, stream);
    }
    /** @brief Transfer nested/internal allocations through allocator static interface. */
    static __INLINE_FCN_RELAXED__ __HOST_DEVICE__ bool transfer_internal_memory(
      pointer ptr, IvyTypes::size_t const& n,
      IvyMemoryType const& ptr_mem_type, IvyMemoryType const& mem_type, IvyGPUStream& stream,
      bool release_old
    ){
      return allocator_type::transfer_internal_memory(ptr, n, ptr_mem_type, mem_type, stream, release_old);
    }
    static __INLINE_FCN_RELAXED__ __HOST_DEVICE__ size_type max_size() noexcept{
      return allocator_type::max_size();
    }
  };

  /**
  allocator_arg_t
  */
  struct allocator_arg_t {
    /** @brief Tag constructor for allocator-aware overload resolution. */
    explicit /*__HOST_DEVICE__*/ allocator_arg_t() = default;
  };
  /** @brief Global tag instance for allocator-aware construction. */
  inline constexpr allocator_arg_t allocator_arg;

}


// IvyMemoryHelpers copy_data functionality should rely on allocators so that complex data structures can be copied correctly.
namespace IvyMemoryHelpers{
  /**
  copy_data: Copies data from a pointer of type U to a pointer of type T.
  - target: Pointer to the target data.
  - source: Pointer to the source data.
  - n_tgt_init: Number of elements in the target array before the copy. If the target array is not null, it is freed.
  - n_tgt: Number of elements in the target array after the copy.
  - n_src: Number of elements in the source array before the copy. It has to satisfy the constraint (n_src==n_tgt || n_src==1).
  - type_tgt: Location of the target data in memory.
  - type_src: Location of the source data in memory.
  - stream: CUDA stream to use for the copy.
    If stream is anything other than cudaStreamLegacy, the copy is asynchronous, even in device code.
  */
  template<typename T, typename U> struct copy_data_fcnal{
    /** @brief Copy source values into target storage with allocation/transfer semantics. */
    static __INLINE_FCN_RELAXED__ __HOST_DEVICE__ bool copy_data(
      T*& target, U* const& source,
      size_t n_tgt_init, size_t n_tgt, size_t n_src
      , IvyMemoryType type_tgt, IvyMemoryType type_src
      , IvyGPUStream& stream
    );
  };
  /** @brief Front-end wrapper dispatching to copy_data_fcnal specialization. */
  template<typename T, typename U> __INLINE_FCN_FORCE__ __HOST_DEVICE__ bool copy_data(
    T*& target, U* const& source,
    size_t n_tgt_init, size_t n_tgt, size_t n_src
    , IvyMemoryType type_tgt, IvyMemoryType type_src
    , IvyGPUStream& stream
  ){
    return copy_data_fcnal<T, U>::copy_data(
      target, source,
      n_tgt_init, n_tgt, n_src
      , type_tgt, type_src
      , stream
    );
  }

  /**
  Overload to allow passing raw cudaStream_t objects.
  */
#ifdef __USE_CUDA__
  /** @brief Overload of copy_data that accepts a raw CUDA stream handle. */
  template<typename T, typename U> __INLINE_FCN_RELAXED__ __HOST_DEVICE__ bool copy_data(
    T*& target, U* const& source,
    size_t n_tgt_init, size_t n_tgt, size_t n_src,
    IvyMemoryType type_tgt, IvyMemoryType type_src,
    cudaStream_t stream
  ){
    IvyGPUStream sr(stream, false);
    return copy_data(
      target, source,
      n_tgt_init, n_tgt, n_src,
      type_tgt, type_src,
      sr
    );
  }
#endif
}

namespace IvyMemoryHelpers{
    template<typename T, typename U> __HOST_DEVICE__ bool copy_data_fcnal<T, U>::copy_data(
    T*& target, U* const& source,
    size_t n_tgt_init, size_t n_tgt, size_t n_src
    , IvyMemoryType type_tgt, IvyMemoryType type_src
    , IvyGPUStream& stream
  ){
    bool res = true;
#if (DEVICE_CODE == DEVICE_CODE_HOST) && defined(__USE_CUDA__)
    bool const tgt_on_device = use_device_acc(type_tgt);
    bool const src_on_device = use_device_acc(type_src);
#else
    constexpr bool tgt_on_device = true;
    constexpr bool src_on_device = true;
#endif
    if (n_tgt==0 || n_src==0 || !source) return false;
    if (!(n_src==n_tgt || n_src==1)){
#if COMPILER == COMPILER_MSVC
      __PRINT_ERROR__("IvyMemoryHelpers::copy_data: Invalid values for n_tgt=%Iu, n_src=%Iu\n", n_tgt, n_src);
#else
      __PRINT_ERROR__("IvyMemoryHelpers::copy_data: Invalid values for n_tgt=%zu, n_src=%zu\n", n_tgt, n_src);
#endif
      assert(0);
    }
    if (n_tgt_init!=n_tgt){
      res &= std_ivy::deallocator_primitive<T>::destroy(target, n_tgt_init, type_tgt, stream);
      res &= std_ivy::allocator_primitive<T>::allocate(target, n_tgt, type_tgt, stream);
    }
    if (res){
#ifdef __USE_CUDA__
      if (!src_on_device && !tgt_on_device) res = false;
      else{
        U* d_source = (src_on_device ? source : nullptr);
        if (!src_on_device){
          res &= std_ivy::allocator_primitive<U>::allocate(d_source, n_src, IvyMemoryType::GPU, stream);
          res &= std_ivy::transfer_memory_primitive<U>::transfer(d_source, source, n_src, IvyMemoryType::GPU, type_src, stream);
        }
        T* d_target = nullptr;
        if (!tgt_on_device) res &= std_ivy::allocator_primitive<T>::allocate(d_target, n_tgt, IvyMemoryType::GPU, stream);
        else d_target = target;
        res &= run_kernel<copy_data_kernel<T, U>>(0, stream).parallel_1D(n_tgt, n_src, d_target, d_source);
        if (!tgt_on_device){
          res &= std_ivy::transfer_memory_primitive<T>::transfer(target, d_target, n_tgt, type_tgt, IvyMemoryType::GPU, stream);
          res &= std_ivy::deallocator_primitive<T>::deallocate(d_target, n_tgt, IvyMemoryType::GPU, stream);
        }
        if (!src_on_device) res &= std_ivy::deallocator_primitive<U>::deallocate(d_source, n_src, IvyMemoryType::GPU, stream);
      }
      if (!res){
        if (tgt_on_device!=src_on_device){
          __PRINT_ERROR__("IvyMemoryHelpers::copy_data: Failed to copy data between host and device.\n");
          assert(0);
        }
#else
      {
#endif
        #define _CMD \
        for (size_t i=0; i<n_tgt; ++i) target[i] = source[(n_src==1 ? 0 : i)];
#if defined(OPENMP_ENABLED)
        if (n_tgt>=NUM_CPU_THREADS_THRESHOLD){
          #pragma omp parallel for schedule(static)
          _CMD
        }
        else
#endif
        {
          _CMD
        }
        #undef _CMD
        res = true;
      }
    }
    return res;
  }
}


#endif
