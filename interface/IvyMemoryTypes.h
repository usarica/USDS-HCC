#ifndef IVYMEMORYTYPES_H
#define IVYMEMORYTYPES_H

/**
 * @file IvyMemoryTypes.h
 * @brief Memory-domain tags and constexpr query helpers.
 */


#include "config/IvyCompilerConfig.h"


namespace IvyMemoryHelpers{
  /** @brief Supported memory domains used by Ivy alloc/transfer helpers. */
  enum class IvyMemoryType : unsigned char{
    Host,
    GPU,
    PageLocked,
    Unified,
    UnifiedPrefetched,
    nMemoryTypes
  };

  /**
  get_mem_type_name: Returns the C-string name of the memory type.
  */
  /** @brief Return a printable memory-domain name for @p type. */
  __INLINE_FCN_RELAXED__ __HOST_DEVICE__ constexpr const char* get_memory_type_name(IvyMemoryType type);

  /**
  is_host_memory: Returns true if the memory type is host memory.
  */
  /** @brief Check if @p type resides in host-addressable CPU memory. */
  __INLINE_FCN_RELAXED__ __HOST_DEVICE__ constexpr bool is_host_memory(IvyMemoryType type);

  /**
  is_gpu_memory: Returns true if the memory type is GPU device memory.
  */
  /** @brief Check if @p type is GPU device memory. */
  __INLINE_FCN_RELAXED__ __HOST_DEVICE__ constexpr bool is_gpu_memory(IvyMemoryType type);

  /**
  is_unified_memory: Returns true if the memory type is unified memory.
  */
  /** @brief Check if @p type is any unified-memory variant. */
  __INLINE_FCN_RELAXED__ __HOST_DEVICE__ constexpr bool is_unified_memory(IvyMemoryType type);

  /**
  is_pagelocked: Returns true if the memory type is page-locked host memory.
  */
  /** @brief Check if @p type is page-locked host memory. */
  __INLINE_FCN_RELAXED__ __HOST_DEVICE__ constexpr bool is_pagelocked(IvyMemoryType type);

  /**
  is_prefetched: Returns true if the memory type is unified memory that has been prefetched.
  */
  /** @brief Check if @p type is prefetched unified memory. */
  __INLINE_FCN_RELAXED__ __HOST_DEVICE__ constexpr bool is_prefetched(IvyMemoryType type);

  /**
  use_device_GPU: Returns true if the memory type is associated with the GPU
  */
  /** @brief Check if operations on @p type should run on GPU execution resources. */
  __INLINE_FCN_RELAXED__ __HOST_DEVICE__ constexpr bool use_device_GPU(IvyMemoryType type);

  /**
  use_device_acc: Returns true if the memory type is associated with hardware accelerators.
  */
  /** @brief Check if operations on @p type should run on any accelerator backend. */
  __INLINE_FCN_RELAXED__ __HOST_DEVICE__ constexpr bool use_device_acc(IvyMemoryType type);

  /**
  use_device_host: Simply the same as !use_device_acc.
  */
  /** @brief Check if operations on @p type should run on host execution resources. */
  __INLINE_FCN_FORCE__ __HOST_DEVICE__ constexpr bool use_device_host(IvyMemoryType type);

  /**
  run_acc_on_host: Check if we are in the host context and the specified memory type needs to run hardware acceleration.
  */
  /** @brief Check if host-side code must dispatch accelerator kernels for @p type. */
  __INLINE_FCN_FORCE__ __HOST_DEVICE__ constexpr bool run_acc_on_host(IvyMemoryType type);

  /**
  get_execution_default_memory: Returns the default memory type for the current execution environment.
  For host code or if not using CUDA, this is IvyMemoryType::Host.
  Otherwise, for device code, this is IvyMemoryType::GPU.
  */
  /** @brief Return default memory domain associated with the current compilation/execution context. */
  __INLINE_FCN_FORCE__ __HOST_DEVICE__ constexpr IvyMemoryType get_execution_default_memory();
}

namespace IvyMemoryHelpers{
  __HOST_DEVICE__ constexpr const char* get_memory_type_name(IvyMemoryType type){
    static_assert(__STATIC_CAST__(unsigned char, IvyMemoryType::nMemoryTypes)==5);
    switch(type){
      case IvyMemoryType::Host: return "CPU";
      case IvyMemoryType::GPU: return "GPU";
      case IvyMemoryType::PageLocked: return "PageLocked";
      case IvyMemoryType::Unified: return "Unified";
      case IvyMemoryType::UnifiedPrefetched: return "UnifiedPrefetched";
      default: return "Unknown";
    }
  }

  __HOST_DEVICE__ constexpr bool is_host_memory(IvyMemoryType type){
    return type==IvyMemoryType::Host || type==IvyMemoryType::PageLocked;
  }

  __HOST_DEVICE__ constexpr bool is_gpu_memory(IvyMemoryType type){
    return type==IvyMemoryType::GPU;
  }

  __HOST_DEVICE__ constexpr bool is_unified_memory(IvyMemoryType type){
    return type==IvyMemoryType::Unified || type==IvyMemoryType::UnifiedPrefetched;
  }

  __HOST_DEVICE__ constexpr bool is_pagelocked(IvyMemoryType type){
    return type==IvyMemoryType::PageLocked;
  }

  __HOST_DEVICE__ constexpr bool is_prefetched(IvyMemoryType type){
    return type==IvyMemoryType::UnifiedPrefetched;
  }

  __HOST_DEVICE__ constexpr bool use_device_GPU(IvyMemoryType type){
    return
#if defined(__USE_CUDA__)
      true
#else
      false
#endif
      && (is_gpu_memory(type) || is_unified_memory(type));
  }

  __HOST_DEVICE__ constexpr bool use_device_host(IvyMemoryType type){ return !use_device_acc(type); }

  // For now, we can only test for GPUs. Once FPGAs or other devices are added, this function would need to be modified.
  __HOST_DEVICE__ constexpr bool use_device_acc(IvyMemoryType type){
    static_assert(__STATIC_CAST__(unsigned char, IvyMemoryType::nMemoryTypes)==5);
    return use_device_GPU(type);
  }

#if (DEVICE_CODE == DEVICE_CODE_GPU)
  __HOST_DEVICE__ constexpr IvyMemoryType get_execution_default_memory(){ return IvyMemoryType::GPU; }
#elif (DEVICE_CODE == DEVICE_CODE_HOST)
  __HOST_DEVICE__ constexpr IvyMemoryType get_execution_default_memory(){ return IvyMemoryType::Host; }
#else
  __HOST_DEVICE__ constexpr IvyMemoryType get_execution_default_memory(){
    static_assert(0, "IvyMemoryHelpers::get_execution_default_memory: Unknown device type.");
    return IvyMemoryType::nMemoryTypes;
  }
#endif

  __INLINE_FCN_FORCE__ __HOST_DEVICE__ constexpr bool run_acc_on_host(IvyMemoryType type){
    constexpr IvyMemoryType def_mem_type = get_execution_default_memory();
    return (use_device_host(def_mem_type) && use_device_acc(type));
  }

  /*
  is_addressable_from_execution: Returns true if memory of the given type can be dereferenced
  directly from the current execution space (no cross-domain copy needed).
  - In host execution, host memory (Host/PageLocked) and unified memory are directly addressable.
  - In device execution, GPU memory and unified memory are directly addressable.
  Unified memory is always addressable because it is mapped into both spaces.
  */
  __INLINE_FCN_FORCE__ __HOST_DEVICE__ constexpr bool is_addressable_from_execution(IvyMemoryType type){
    constexpr IvyMemoryType def_mem_type = get_execution_default_memory();
    return is_unified_memory(type) || (use_device_host(def_mem_type) ? is_host_memory(type) : is_gpu_memory(type));
  }
}


// Aliases for std_ivy namespace
namespace std_ivy{
  using IvyMemoryType = IvyMemoryHelpers::IvyMemoryType;
}


#endif
