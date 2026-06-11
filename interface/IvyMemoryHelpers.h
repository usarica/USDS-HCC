#ifndef IVYMEMORYHELPERS_H
#define IVYMEMORYHELPERS_H

/**
 * @file IvyMemoryHelpers.h
 * @brief Allocation, construction, destruction, and transfer helpers across memory domains.
 */


/**
IvyMemoryHelpers: A collection of functions for allocating, freeing, and copying memory.
The functions are overloaded for both host and device code when CUDA is enabled.
If CUDA is disabled, allocation and freeing are done with new and delete.
Otherwise, allocation and freeing conventions are as follows:
- Host code, native host memory: new/delete
- Host code, page-locked host memory: cudaMallocHost/cudaFreeHost
- Host code, device memory: cudaMallocAsync/cudaFreeAsync
- Host code, unified memory: cudaMallocManaged/cudaFreeAsync
  For unified memory, one can also specify a variation of the flag to enable prefetching the data.
  In that case, prefetching is done to both the CPU and GPU.
- GPU device code, device memory: new/delete
- GPU device code, host/page-locked host/unified memory: Disabled
Copy operations running on the device may call kernel functions to parallelize further.
For that reason, the -rdc=true flag is required when compiling device code.
*/


#include "IvyBasicTypes.h"
#include "IvyMemoryTypes.h"
#include "config/IvyConfig.h"
#include "config/IvyKernelRun.h"
#include "std_ivy/IvyCassert.h"
#include "std_ivy/IvyUtility.h"
#include "std_ivy/IvyCstdio.h"
#include "std_ivy/IvyTypeInfo.h"
#include "stream/IvyStream.h"


// Declarations and enum definitions
namespace IvyMemoryHelpers{
  using size_t = IvyTypes::size_t;
  using ptrdiff_t = IvyTypes::ptrdiff_t;

  /**
  allocate_memory: Allocates memory for an array of type T of size n. Constructors are called for the arguments args.
  - data: Pointer to the target data.
  - n: Number of elements.
  When using CUDA, the following arguments have nontrivial meaning:
  - type: In host code, this flag determines whether to allocate the data in device, host, unified, or page-locked host memory.
    In device code, this flag is ignored, and the memory is always allocated on the device.
  - stream: In host code, this is the CUDA stream to use for the allocation.
    In device code, any allocation and object construction operations are always synchronous with the running thread.
  */
  template<typename T, typename... Args> struct allocate_memory_fcnal{
    static __INLINE_FCN_RELAXED__ __HOST_DEVICE__ bool allocate_memory(
      T*& data,
      size_t n
      , IvyMemoryType type
      , IvyGPUStream& stream
    );
  };
  template<typename T, typename... Args> __INLINE_FCN_FORCE__ __HOST_DEVICE__ bool allocate_memory(
    T*& data,
    size_t n
    , IvyMemoryType type
    , IvyGPUStream& stream
  ){
    return allocate_memory_fcnal<T, Args...>::allocate_memory(
      data, n
      , type, stream
    );
  }

  /**
  construct: Constructs an array of n objects for the arguments args.
  - data: Pointer to the target data.
  - n: Number of elements.
  - args: Arguments for the constructors of the elements.
  When using CUDA, the following arguments have nontrivial meaning:
  - type: In host code, this flag determines whether to allocate the data in device, host, unified, or page-locked host memory.
    In device code, this flag is ignored, and the memory is always allocated on the device.
  - stream: In host code, this is the CUDA stream to use for the allocation.
    In device code, any allocation and object construction operations are always synchronous with the running thread.
  */
  template<typename T, typename... Args> struct construct_fcnal{
    static __INLINE_FCN_RELAXED__ __HOST_DEVICE__ bool construct(
      T*& data,
      size_t n
      , IvyMemoryType type
      , IvyGPUStream& stream
      , Args&&... args
    );
  };
  template<typename T, typename... Args> __INLINE_FCN_FORCE__ __HOST_DEVICE__ bool construct(
    T*& data,
    size_t n
    , IvyMemoryType type
    , IvyGPUStream& stream
    , Args&&... args
  ){
    return construct_fcnal<T, Args...>::construct(
      data, n
      , type, stream
      , args...
    );
  }

  /**
  build: Allocates memory for an array of type T of size n and builds the objects for the arguments args.
  Calling this function is a shorthand for allocate_memory + construct.
  - data: Pointer to the target data.
  - n: Number of elements.
  - args: Arguments for the buildors of the elements.
  When using CUDA, the following arguments have nontrivial meaning:
  - type: In host code, this flag determines whether to allocate the data in device, host, unified, or page-locked host memory.
    In device code, this flag is ignored, and the memory is always allocated on the device.
  - stream: In host code, this is the CUDA stream to use for the allocation.
    In device code, any allocation and object buildion operations are always synchronous with the running thread.
  */
  template<typename T, typename... Args> __INLINE_FCN_FORCE__ __HOST_DEVICE__ bool build(
    T*& data,
    size_t n
    , IvyMemoryType type
    , IvyGPUStream& stream
    , Args&&... args
  ){
    return
      allocate_memory<T, Args...>(
        data, n
        , type, stream
      )
      &&
      construct(
        data, n
        , type, stream
        , args...
      );
  }

  /**
  free_memory: Frees the memory storing an array of type T of size n in a way consistent with allocate_memory.
  - data: Pointer to the data.
  - n: Number of elements.
  When using CUDA, the following arguments have nontrivial meaning:
  - type: In host code, this flag determines whether the data resides in device, host, unified, or page-locked host memory.
    In device code, this flag is ignored, and the memory is always freed from the device.
  - stream: In host code, this is the CUDA stream to use for the deallocation.
    In device code, any deallocation operations are always synchronous with the running thread.
  */
  template<typename T> struct free_memory_fcnal{
    static __INLINE_FCN_RELAXED__ __HOST_DEVICE__ bool free_memory(
      T*& data,
      size_t n
      , IvyMemoryType type
      , IvyGPUStream& stream
    );
  };
  template<typename T> __INLINE_FCN_FORCE__ __HOST_DEVICE__ bool free_memory(
    T*& data,
    size_t n
    , IvyMemoryType type
    , IvyGPUStream& stream
  ){
    return free_memory_fcnal<T>::free_memory(
      data, n
      , type, stream
    );
  }

  /**
  destruct: Calls the destructors of each element of an array of type T of size n.
  - data: Pointer to the data.
  - n: Number of elements.
  When using CUDA, the following arguments have nontrivial meaning:
  - type: In host code, this flag determines whether the data resides in device, host, unified, or page-locked host memory.
    In device code, this flag is ignored, and the memory is always freed from the device.
  - stream: In host code, this is the CUDA stream to use for the deallocation.
    In device code, any deallocation operations are always synchronous with the running thread.
  */
  template<typename T> struct destruct_fcnal{
    static __INLINE_FCN_RELAXED__ __HOST_DEVICE__ bool destruct(
      T*& data,
      size_t n
      , IvyMemoryType type
      , IvyGPUStream& stream
    );
  };
  /*
  template<typename T> struct destruct_fcnal<T*>{
    static __INLINE_FCN_RELAXED__ __HOST_DEVICE__ bool destruct(
      T**& data,
      size_t n
      , IvyMemoryType type
      , IvyGPUStream& stream
    );
  };
  */
  template<typename T> __INLINE_FCN_FORCE__ __HOST_DEVICE__ bool destruct(
    T*& data,
    size_t n
    , IvyMemoryType type
    , IvyGPUStream& stream
  ){
    return destruct_fcnal<T>::destruct(
      data, n
      , type, stream
    );
  }

  /**
  destroy: Calls the destructors of each element of an array of type T of size n, and frees the memory of the pointer in a way consistent with allocate_memory.
  - data: Pointer to the data.
  - n: Number of elements.
  When using CUDA, the following arguments have nontrivial meaning:
  - type: In host code, this flag determines whether the data resides in device, host, unified, or page-locked host memory.
    In device code, this flag is ignored, and the memory is always freed from the device.
  - stream: In host code, this is the CUDA stream to use for the deallocation.
    In device code, any deallocation operations are always synchronous with the running thread.
  */
  template<typename T> __INLINE_FCN_FORCE__ __HOST_DEVICE__ bool destroy(
    T*& data,
    size_t n
    , IvyMemoryType type
    , IvyGPUStream& stream
  ){
    return
      destruct(
        data, n
        , type, stream
      )
      &&
      free_memory(
        data, n
        , type, stream
      );
  }

  /**
  construct_data_kernel: Kernel function for constructing object of type T from pointer.
  - n: Number of elements.
  - data: Pointer to the data array.
  - args: Arguments for the constructors of the elements.
  */
  template<typename T, typename... Args> struct construct_data_kernel : public kernel_base_noprep_nofin{
    /**
     * @brief Kernel entry point to construct element @p i in-place.
     * @param i Linear element index.
     * @param n Total number of elements.
     * @param data Target data pointer.
     * @param args Constructor arguments forwarded to each element.
     */
    static __HOST_DEVICE__ void kernel(size_t const& i, size_t const& n, T* data, Args&&... args);
  };

  /**
  destruct_data_kernel: Kernel function for destructing object of type T from pointer.
  - n: Number of elements.
  - data: Pointer to the data array.
  */
  template<typename T> struct destruct_data_kernel : public kernel_base_noprep_nofin{
    /**
     * @brief Kernel entry point to destruct element @p i.
     * @param i Linear element index.
     * @param n Total number of elements.
     * @param data Target data pointer.
     */
    static __HOST_DEVICE__ void kernel(size_t const& i, size_t const& n, T* data);
  };

  /**
  copy_data_kernel: Kernel function for copying data from a pointer of type U to a pointer of type T.
  - n_tgt: Number of elements in the target array.
  - n_src: Number of elements in the source array with the constraint (n_src==n_tgt || n_src==1).
  - target: Pointer to the target data.
  - source: Pointer to the source data.
  */
  template<typename T, typename U> struct copy_data_kernel : public kernel_base_noprep_nofin{
    /**
     * @brief Kernel entry point to copy source data into target element @p i.
     * @param i Linear target index.
     * @param n_tgt Total number of target elements.
     * @param n_src Number of source elements.
     * @param target Target pointer.
     * @param source Source pointer.
     */
    static __HOST_DEVICE__ void kernel(size_t const& i, size_t const& n_tgt, size_t const& n_src, T* target, U* source);
  };

#ifdef __USE_CUDA__
  /**
  get_cuda_transfer_direction: Translates the target and source memory locations to the corresponding cudaMemcpyKind transfer type.
  */
  /**
   * @brief Map Ivy memory domains to CUDA memcpy direction.
   * @param tgt Target memory type.
   * @param src Source memory type.
   * @return CUDA copy-kind enum corresponding to @p src -> @p tgt transfer.
   */
  __INLINE_FCN_RELAXED__ __HOST_DEVICE__ constexpr cudaMemcpyKind get_cuda_transfer_direction(IvyMemoryType tgt, IvyMemoryType src);
#endif

  /**
  transfer_memory: Runs the transfer operation between two pointers of type T.
  - tgt: Pointer to the target data.
  - src: Pointer to the source data.
  - n: Number of elements.
  - type_tgt: Location of the target data in memory.
  - type_src: Location of the source data in memory.
  - stream: CUDA stream to use for the transfer.
    In device code, since cudaMemcpy is not available, we always use cudaMemcpyAsync instead.
  */
  template<typename T> struct transfer_memory_fcnal{
    static __INLINE_FCN_RELAXED__ __HOST_DEVICE__ bool transfer_memory(
      T*& tgt, T* const& src, size_t n,
      IvyMemoryType type_tgt, IvyMemoryType type_src,
      IvyGPUStream& stream
    );
  };
  template<typename T> __INLINE_FCN_FORCE__ __HOST_DEVICE__ bool transfer_memory(
    T*& tgt, T* const& src, size_t n,
    IvyMemoryType type_tgt, IvyMemoryType type_src,
    IvyGPUStream& stream
  ){
    return transfer_memory_fcnal<T>::transfer_memory(
      tgt, src, n,
      type_tgt, type_src,
      stream
    );
  }

  /**
  Overloads to allow passing raw cudaStream_t objects.
  */
#ifdef __USE_CUDA__
  template<typename T, typename... Args> __INLINE_FCN_RELAXED__ __HOST_DEVICE__ bool allocate_memory(
    T*& data,
    size_t n
    , IvyMemoryType type
    , cudaStream_t stream
  ){
    IvyGPUStream sr(stream, false);
    return allocate_memory<T, Args...>(
      data, n
      , type
      , sr
    );
  }
  template<typename T, typename... Args> __INLINE_FCN_RELAXED__ __HOST_DEVICE__ bool construct(
    T*& data,
    size_t n
    , IvyMemoryType type
    , cudaStream_t stream
    , Args&&... args
  ){
    IvyGPUStream sr(stream, false);
    return construct(
      data, n
      , type
      , sr
      , args...
    );
  }
  template<typename T, typename... Args> __INLINE_FCN_RELAXED__ __HOST_DEVICE__ bool build(
    T*& data,
    size_t n
    , IvyMemoryType type
    , cudaStream_t stream
    , Args&&... args
  ){
    IvyGPUStream sr(stream, false);
    return build(
      data, n
      , type
      , sr
      , args...
    );
  }
  template<typename T, typename... Args> __INLINE_FCN_RELAXED__ __HOST_DEVICE__ bool free_memory(
    T*& data,
    size_t n
    , IvyMemoryType type
    , cudaStream_t stream
  ){
    IvyGPUStream sr(stream, false);
    return free_memory(
      data, n
      , type
      , sr
    );
  }
  template<typename T, typename... Args> __INLINE_FCN_RELAXED__ __HOST_DEVICE__ bool destruct(
    T*& data,
    size_t n
    , IvyMemoryType type
    , cudaStream_t stream
  ){
    IvyGPUStream sr(stream, false);
    return destruct(
      data, n
      , type
      , sr
    );
  }
  template<typename T, typename... Args> __INLINE_FCN_RELAXED__ __HOST_DEVICE__ bool destroy(
    T*& data,
    size_t n
    , IvyMemoryType type
    , cudaStream_t stream
  ){
    IvyGPUStream sr(stream, false);
    return destroy(
      data, n
      , type
      , sr
    );
  }
  template<typename T> __INLINE_FCN_RELAXED__ __HOST_DEVICE__ bool transfer_memory(
    T*& tgt, T* const& src, size_t n,
    IvyMemoryType type_tgt, IvyMemoryType type_src,
    cudaStream_t stream
  ){
    IvyGPUStream sr(stream, false);
    return transfer_memory(
      tgt, src, n,
      type_tgt, type_src,
      sr
    );
  }
#endif
}


// Definitions
namespace IvyMemoryHelpers{
  template<typename T, typename... Args> __HOST_DEVICE__ bool allocate_memory_fcnal<T, Args...>::allocate_memory(
    T*& data,
    size_t n
    , IvyMemoryType type
    , IvyGPUStream& stream
  ){
    if (n==0 || data) return false;
#ifdef __DEBUG_MEMORY__
    char const* name_T = __TYPE_NAME__(T);
    char const* name_mem_type = get_memory_type_name(type);
#endif
#if (DEVICE_CODE == DEVICE_CODE_HOST) && defined(__USE_CUDA__)
    static_assert(__STATIC_CAST__(unsigned char, IvyMemoryType::nMemoryTypes)==5);
    bool const is_pl = is_pagelocked(type);
    bool const is_gpu = is_gpu_memory(type);
    bool const is_uni = is_unified_memory(type);
    if (is_gpu || is_uni || is_pl){
      if (is_gpu){
        __CHECK_OR_EXIT_WITH_ERROR__(cudaMallocAsync((void**) &data, n*sizeof(T), stream));
#ifdef __DEBUG_MEMORY__
        __PRINT_DEBUG__("allocate_memory_fcnal::allocate_memory: Method = cudaMallocAsync, ptr = %p, n = %llu, mem_type = %s, type = %s\n", data, n, name_mem_type, name_T);
#endif
      }
      else if (is_pl){
        __CHECK_OR_EXIT_WITH_ERROR__(cudaHostAlloc((void**) &data, n*sizeof(T), cudaHostAllocDefault));
#ifdef __DEBUG_MEMORY__
        __PRINT_DEBUG__("allocate_memory_fcnal::allocate_memory: Method = cudaHostAlloc, ptr = %p, n = %llu, mem_type = %s, type = %s\n", data, n, name_mem_type, name_T);
#endif
      }
      else if (is_uni){
        __CHECK_OR_EXIT_WITH_ERROR__(cudaMallocManaged((void**) &data, n*sizeof(T), cudaMemAttachGlobal));
#ifdef __DEBUG_MEMORY__
        __PRINT_DEBUG__("allocate_memory_fcnal::allocate_memory: Method = cudaMallocManaged, ptr = %p, n = %llu, mem_type = %s, type = %s\n", data, n, name_mem_type, name_T);
#endif
        if (is_prefetched(type)){
          IvyCudaConfig::IvyDeviceNum_t dev_gpu = 0;
          int supports_prefetch = 0;
          if (
            __CHECK_SUCCESS__(cudaGetDevice(&dev_gpu))
            &&
            __CHECK_SUCCESS__(cudaDeviceGetAttribute(&supports_prefetch, cudaDevAttrConcurrentManagedAccess, dev_gpu))
            &&
            supports_prefetch==1
            ){
            __CHECK_OR_EXIT_WITH_ERROR__(cudaMemPrefetchAsync(data, n*sizeof(T), cudaCpuDeviceId, stream));
            __CHECK_OR_EXIT_WITH_ERROR__(cudaMemPrefetchAsync(data, n*sizeof(T), dev_gpu, stream));
          }
        }
      }
    }
    else
#endif
    {
      data = (T*) malloc(n*sizeof(T));
      if (!data){
        __PRINT_ERROR__("allocate_memory_fcnal::allocate_memory: malloc failed for n = %llu, type size = %llu.\n", __STATIC_CAST__(unsigned long long, n), __STATIC_CAST__(unsigned long long, sizeof(T)));
        return false;
      }
#ifdef __DEBUG_MEMORY__
      __PRINT_DEBUG__("allocate_memory_fcnal::allocate_memory: Method = malloc, ptr = %p, n = %llu, mem_type = %s, type = %s\n", data, n, name_mem_type, name_T);
#endif
    }
    return true;
  }


  template<typename T, typename... Args> __HOST_DEVICE__ bool construct_fcnal<T, Args...>::construct(
    T*& data,
    size_t n
    , IvyMemoryType type
    , IvyGPUStream& stream
    , Args&&... args
  ){
    if (!data) return false;
    if (n==0) return true;
    bool res = true;
#ifdef __DEBUG_CONSTRUCT_DESTRUCT__
    char const* name_T = __TYPE_NAME__(T);
    char const* name_mem_type = get_memory_type_name(type);
#endif
#if (DEVICE_CODE == DEVICE_CODE_HOST) && defined(__USE_CUDA__)
    static_assert(__STATIC_CAST__(unsigned char, IvyMemoryType::nMemoryTypes)==5);
    bool const is_pl = is_pagelocked(type);
    bool const is_acc = use_device_acc(type);
    if (is_acc || is_pl){
#ifdef __DEBUG_CONSTRUCT_DESTRUCT__
      __PRINT_DEBUG__("construct_fcnal::construct: Transfer sequence for ptr = %p, n = %llu, mem_type = %s, type = %s\n", data, n, name_mem_type, name_T);
#endif
      T* temp = nullptr;
      res &= build(temp, n, IvyMemoryType::Host, stream, args...);
      res &= transfer_memory(data, temp, n, type, IvyMemoryType::Host, stream);
      stream.synchronize();
      res &= free_memory(temp, n, IvyMemoryType::Host, stream); // Important! Calling the destructor can also invalidate the internal components of the data pointer on the device!
    }
    else
#endif
    {
#ifdef __DEBUG_CONSTRUCT_DESTRUCT__
      __PRINT_DEBUG__("construct_fcnal::construct: Direct sequence for ptr = %p, n = %llu, mem_type = %s, type = %s\n", data, n, name_mem_type, name_T);
#endif
      for (size_t i=0; i<n; ++i) new (data+i) T(std_util::forward<Args>(args)...);
    }
    return res;
  }

  template<typename T> __HOST_DEVICE__ bool free_memory_fcnal<T>::free_memory(
    T*& data,
    size_t n
    , IvyMemoryType type
    , IvyGPUStream& stream
  ){
    if (!data) return true;
    if (n==0) return false;
#ifdef __DEBUG_MEMORY__
    char const* name_T = __TYPE_NAME__(T);
    char const* name_mem_type = get_memory_type_name(type);
#endif
#if (DEVICE_CODE == DEVICE_CODE_HOST) && defined(__USE_CUDA__)
    static_assert(__STATIC_CAST__(unsigned char, IvyMemoryType::nMemoryTypes)==5);
    bool const is_pl = is_pagelocked(type);
    if (use_device_GPU(type) || is_pl){
      bool const is_uni = is_unified_memory(type);
      if (!is_pl && !is_uni){
#ifdef __DEBUG_MEMORY__
        __PRINT_DEBUG__("free_memory_fcnal::free_memory: Method = cudaFreeAsync, ptr = %p, n = %llu, mem_type = %s, type = %s\n", data, n, name_mem_type, name_T);
#endif
        __CHECK_OR_EXIT_WITH_ERROR__(cudaFreeAsync(data, stream));
      }
      else if (is_uni){
#ifdef __DEBUG_MEMORY__
        __PRINT_DEBUG__("free_memory_fcnal::free_memory: Method = cudaFree, ptr = %p, n = %llu, mem_type = %s, type = %s\n", data, n, name_mem_type, name_T);
#endif
        __CHECK_OR_EXIT_WITH_ERROR__(cudaFree(data));
      }
      else{
#ifdef __DEBUG_MEMORY__
        __PRINT_DEBUG__("free_memory_fcnal::free_memory: Method = cudaFreeHost, ptr = %p, n = %llu, mem_type = %s, type = %s\n", data, n, name_mem_type, name_T);
#endif
        __CHECK_OR_EXIT_WITH_ERROR__(cudaFreeHost(data));
      }
    }
    else
#endif
    {
#ifdef __DEBUG_MEMORY__
      __PRINT_DEBUG__("free_memory_fcnal::free_memory: Method = free, ptr = %p, n = %llu, mem_type = %s, type = %s\n", data, n, name_mem_type, name_T);
#endif
      free(data);
    }
    data = nullptr;
    return true;
  }

  template<typename T> __HOST_DEVICE__ bool destruct_fcnal<T>::destruct(
    T*& data,
    size_t n
    , IvyMemoryType type
    , IvyGPUStream& stream
  ){
    if (!data) return false;
    if (n==0) return true;
    bool res = true;
#ifdef __DEBUG_CONSTRUCT_DESTRUCT__
    char const* name_T = __TYPE_NAME__(T);
    char const* name_mem_type = get_memory_type_name(type);
#endif
#if (DEVICE_CODE == DEVICE_CODE_HOST) && defined(__USE_CUDA__)
    static_assert(__STATIC_CAST__(unsigned char, IvyMemoryType::nMemoryTypes)==5);
    bool const is_pl = is_pagelocked(type);
    bool const is_acc = use_device_acc(type);
    if (is_acc || is_pl){
#ifdef __DEBUG_CONSTRUCT_DESTRUCT__
      __PRINT_DEBUG__("destruct_fcnal::destruct: Transfer sequence for ptr = %p, n = %llu, mem_type = %s, type = %s\n", data, n, name_mem_type, name_T);
#endif
      T* temp = nullptr;
      res &= allocate_memory(temp, n, IvyMemoryType::Host, stream);
      res &= transfer_memory(temp, data, n, IvyMemoryType::Host, type, stream);
      stream.synchronize();
      {
        T* ptr = temp;
        for (size_t i=0; i<n; ++i){
          ptr->~T();
          ++ptr;
        }
      }
      res &= transfer_memory(data, temp, n, type, IvyMemoryType::Host, stream);
      res &= free_memory(temp, n, IvyMemoryType::Host, stream);
    }
    else
#endif
    {
#ifdef __DEBUG_CONSTRUCT_DESTRUCT__
      __PRINT_DEBUG__("destruct_fcnal::destruct: Direct sequence for ptr = %p, n = %llu, mem_type = %s, type = %s\n", data, n, name_mem_type, name_T);
#endif
      T* ptr = data;
      for (size_t i=0; i<n; ++i){
        ptr->~T();
        ++ptr;
      }
    }
    return res;
  }

  template<typename T, typename... Args> __HOST_DEVICE__ void construct_data_kernel<T, Args...>::kernel(size_t const& i, size_t const& n, T* data, Args&&... args){
    if (kernel_check_dims<construct_data_kernel<T, Args...>>::check_dims(i, n)){
      //__PRINT_DEBUG__("Called construct_data_kernel::kernel for i=%llu, n=%llu, data=%p\n", i, n, data);
      new (data+i) T(std_util::forward<Args>(args)...);
    }
  }

  template<typename T> __HOST_DEVICE__ void destruct_data_kernel<T>::kernel(size_t const& i, size_t const& n, T* data){
    if (kernel_check_dims<destruct_data_kernel<T>>::check_dims(i, n)) (data+i)->~T();
  }

  template<typename T, typename U> __HOST_DEVICE__ void copy_data_kernel<T, U>::kernel(size_t const& i, size_t const& n_tgt, size_t const& n_src, T* target, U* source){
    if (!(n_src==n_tgt || n_src==1)){
#if COMPILER == COMPILER_MSVC
      __PRINT_ERROR__("IvyMemoryHelpers::copy_data_kernel::kernel: Invalid values for n_tgt=%Iu, n_src=%Iu\n", n_tgt, n_src);
#else
      __PRINT_ERROR__("IvyMemoryHelpers::copy_data_kernel::kernel: Invalid values for n_tgt=%zu, n_src=%zu\n", n_tgt, n_src);
#endif
      assert(0);
    }
    if (kernel_check_dims<copy_data_kernel<T, U>>::check_dims(i, n_tgt)) *(target+i) = *(source + (n_src==1 ? 0 : i));
  }

#ifdef __USE_CUDA__
  __HOST_DEVICE__ constexpr cudaMemcpyKind get_cuda_transfer_direction(IvyMemoryType tgt, IvyMemoryType src){
#if (DEVICE_CODE == DEVICE_CODE_HOST)
    bool const tgt_on_device = is_gpu_memory(tgt);
    bool const tgt_on_host = is_host_memory(tgt);
    bool const tgt_unified = is_unified_memory(tgt);
    bool const src_on_device = is_gpu_memory(src);
    bool const src_on_host = is_host_memory(src);
    bool const src_unified = is_unified_memory(src);
    if (tgt_on_host && src_on_host) return cudaMemcpyHostToHost;
    else if (tgt_on_device && src_on_host) return cudaMemcpyHostToDevice;
    else if (tgt_on_host && src_on_device) return cudaMemcpyDeviceToHost;
    else if (tgt_on_device && src_on_device) return cudaMemcpyDeviceToDevice;
    else if (tgt_unified || src_unified) return cudaMemcpyDefault;
    else{
      __PRINT_ERROR__("IvyMemoryHelpers::get_cuda_transfer_direction: Unknown transfer direction from %d to %d.\n", __STATIC_CAST__(int, src), __STATIC_CAST__(int, tgt));
      assert(0);
      return cudaMemcpyDefault;
    }
#else
    return cudaMemcpyDeviceToDevice;
#endif
  }
#endif

  template<typename T> __HOST_DEVICE__ bool transfer_memory_fcnal<T>::transfer_memory(
    T*& tgt, T* const& src, size_t n,
    IvyMemoryType type_tgt, IvyMemoryType type_src,
    IvyGPUStream& stream
  ){
    if (!tgt || !src) return false;
    if (n==0 || tgt==src) return true;
#if DEVICE_CODE == DEVICE_CODE_HOST && defined(__USE_CUDA__)
    //__PRINT_DEBUG__("transfer_memory_fcnal::transfer_memory: Running on host code for type %s, %p -> %p\n", typeid(T).name(), src, tgt);
    auto dir = get_cuda_transfer_direction(type_tgt, type_src);
    //__PRINT_DEBUG__("Running transfer_memory_fcnal::transfer_memory: mem_type = %s, type = %s | size = %llu | %p (mem_type = %d) -> %p (mem_type = %d) | stream = %p\n", typeid(T).name(), n, src, int(type_src), tgt, int(type_tgt), stream.stream());
    __CHECK_OR_EXIT_WITH_ERROR__(cudaMemcpyAsync(tgt, src, n*sizeof(T), dir, stream));
#else
    //__PRINT_DEBUG__("transfer_memory_fcnal::transfer_memory: Running on device code, %p -> %p\n", src, tgt);
    memcpy(tgt, src, n*sizeof(T));
#endif
    return true;
  }
}


#endif
