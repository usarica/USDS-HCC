#ifndef IVYKERNELRUN_H
#define IVYKERNELRUN_H

/**
 * @file IvyKernelRun.h
 * @brief Unified kernel dispatch layer with GPU-first and CPU/OpenMP fallback execution.
 */


/**
 * @details
 * Kernel calls can be structured such that `Kernel_t::kernel` is either:
 * - A non-factorizable function, or
 * - A thin wrapper around `Kernel_t::kernel_unit_unified` with a guard like
 *   `if (i<n) kernel_unit_unified(i, n, args...);`.
 *
 * This dispatcher does not validate memory-placement compatibility of the arguments.
 * If argument memory locations are arranged properly and `kernel_unit_unified` calls are
 * factorizable, the `run_kernel` struct can execute the kernel in parallel (GPU) or in
 * loop/OpenMP fallback (CPU), depending on runtime GPU usability.
 *
 * Typical usage:
 * - Define a kernel type with static `kernel(...)`, or with `kernel_unit_unified(...)` and
 *   a guarded `kernel(...)` wrapper.
 * - Invoke via one of:
 *   - `run_kernel<example_kernel>(shared_mem_size, stream).parallel_1D(n, args...);`
 *   - `run_kernel<example_kernel>(shared_mem_size, stream).parallel_2D(nx, ny, args...);`
 *   - `run_kernel<example_kernel>(shared_mem_size, stream).parallel_3D(nx, ny, nz, args...);`
 */


#include "IvyBasicTypes.h"
#include "config/IvyConfig.h"
#include "std_ivy/IvyTypeTraits.h"
#include "stream/IvyStream.h"


struct run_kernel_base{
  /** @brief Dynamic shared-memory size passed to GPU kernels. */
  IvyTypes::size_t shared_mem_size;
  /** @brief Stream used for GPU-side launches and synchronization points. */
  IvyGPUStream& stream;

  /**
   * @brief Construct the common dispatch context.
   * @param shared_mem_size_ Dynamic shared-memory size for CUDA launches.
   * @param stream_ Stream used for kernel launches and related operations.
   */
  __HOST_DEVICE__ run_kernel_base(IvyTypes::size_t const& shared_mem_size_, IvyGPUStream& stream_) : shared_mem_size(shared_mem_size_), stream(stream_){}
};
struct kernel_base_noprep_nofin{
  /** @brief Default no-op pre-dispatch hook. */
  static __HOST_DEVICE__ void prepare(...){}
  /** @brief Default no-op post-dispatch hook. */
  static __HOST_DEVICE__ void finalize(...){}
};

/**
 * @brief Dimension-check policy for kernel dispatch.
 * @details
 * Users may specialize this policy for custom kernels to tailor boundary checks used by
 * `run_kernel` and `generic_kernel_[N]D` launch wrappers.
 *
 * Default behavior checks whether thread indices are within input bounds (e.g. `i<n`).
 * On GPU, this protects against out-of-range threads caused by launch-grid rounding.
 *
 * Example usage in a kernel implementation:
 * `if (kernel_check_dims<MyKernel>::check_dims(i, n)) { ... }`.
 */
template<typename Kernel_t> struct kernel_check_dims{
  /** @brief 1D bounds check helper. */
  static __INLINE_FCN_FORCE__ __HOST_DEVICE__ bool check_dims(IvyTypes::size_t const& i, IvyTypes::size_t const& n){
    return (i<n);
  }
  /** @brief 2D bounds check helper. */
  static __INLINE_FCN_FORCE__ __HOST_DEVICE__ bool check_dims(IvyTypes::size_t const& i, IvyTypes::size_t const& j, IvyTypes::size_t const& nx, IvyTypes::size_t const& ny){
    return (i<nx && j<ny);
  }
  /** @brief 3D bounds check helper. */
  static __INLINE_FCN_FORCE__ __HOST_DEVICE__ bool check_dims(IvyTypes::size_t const& i, IvyTypes::size_t const& j, IvyTypes::size_t const& k, IvyTypes::size_t const& nx, IvyTypes::size_t const& ny, IvyTypes::size_t const& nz){
    return (i<nx && j<ny && k<nz);
  }
};

#ifdef __USE_CUDA__

/**
 * @brief CUDA index-mapping policy from launch-space to logical kernel dimensions.
 * @details
 * This struct-based policy can be specialized/partially-specialized per kernel if needed.
 *
 * Mapping rules:
 * - 1D: fully flattened from `(x,y,z)` launch coordinates.
 * - 2D: `x` kept as-is, `z` folded into `y`.
 * - 3D: `(x,y,z)` kept directly.
 */
template<typename Kernel_t> struct kernel_call_dims{
  /** @brief Resolve flattened 1D logical index from CUDA launch indices. */
  static __INLINE_FCN_RELAXED__ __DEVICE__ void get_dims(IvyTypes::size_t& i){
    IvyTypes::size_t ix = blockIdx.x * blockDim.x + threadIdx.x;
    IvyTypes::size_t iy = blockIdx.y * blockDim.y + threadIdx.y;
    IvyTypes::size_t iz = blockIdx.z * blockDim.z + threadIdx.z;
    i = ix + iy * blockDim.x * gridDim.x + iz * blockDim.x * gridDim.x * blockDim.y * gridDim.y;
  }
  /** @brief Resolve 2D logical indices from CUDA launch indices. */
  static __INLINE_FCN_RELAXED__ __DEVICE__ void get_dims(IvyTypes::size_t& i, IvyTypes::size_t& j){
    i = blockIdx.x * blockDim.x + threadIdx.x;
    IvyTypes::size_t iy = blockIdx.y * blockDim.y + threadIdx.y;
    IvyTypes::size_t iz = blockIdx.z * blockDim.z + threadIdx.z;
    j = iy + iz * blockDim.y * gridDim.y;
  }
  /** @brief Resolve 3D logical indices from CUDA launch indices. */
  static __INLINE_FCN_RELAXED__ __DEVICE__ void get_dims(IvyTypes::size_t& i, IvyTypes::size_t& j, IvyTypes::size_t& k){
    i = blockIdx.x * blockDim.x + threadIdx.x;
    j = blockIdx.y * blockDim.y + threadIdx.y;
    k = blockIdx.z * blockDim.z + threadIdx.z;
  }
};


/**
 * @brief Generic 1D CUDA launcher wrapper that forwards logical index and arguments to `Kernel_t::kernel`.
 */
template<typename Kernel_t, typename... Args> __CUDA_GLOBAL__ void generic_kernel_1D(Args... args){
  IvyTypes::size_t i = 0;
  kernel_call_dims<Kernel_t>::get_dims(i);
  Kernel_t::kernel(i, args...);
}
/**
 * @brief Generic 2D CUDA launcher wrapper that forwards logical indices and arguments to `Kernel_t::kernel`.
 */
template<typename Kernel_t, typename... Args> __CUDA_GLOBAL__ void generic_kernel_2D(Args... args){
  IvyTypes::size_t i = 0, j = 0;
  kernel_call_dims<Kernel_t>::get_dims(i, j);
  Kernel_t::kernel(i, j, args...);
}
/**
 * @brief Generic 3D CUDA launcher wrapper that forwards logical indices and arguments to `Kernel_t::kernel`.
 */
template<typename Kernel_t, typename... Args> __CUDA_GLOBAL__ void generic_kernel_3D(Args... args){
  IvyTypes::size_t i = 0, j = 0, k = 0;
  kernel_call_dims<Kernel_t>::get_dims(i, j, k);
  Kernel_t::kernel(i, j, k, args...);
}

DEFINE_HAS_CALL(kernel_unit_unified);

template<typename Kernel_t, bool = has_call_kernel_unit_unified_v<Kernel_t>> struct run_kernel : run_kernel_base{
  /**
   * @brief Construct a dispatcher for `Kernel_t` with launch context.
   * @param shared_mem_size_ Dynamic shared-memory size for CUDA launches.
   * @param stream_ Stream used for launch and synchronization.
   */
  __HOST_DEVICE__ run_kernel(IvyTypes::size_t const& shared_mem_size_, IvyGPUStream& stream_) : run_kernel_base(shared_mem_size_, stream_){}

  /**
   * @brief Execute 1D kernel dispatch on GPU when usable.
   * @param n Number of logical elements.
   * @param args Kernel arguments.
   * @return `true` if GPU dispatch ran, `false` otherwise.
   */
  template<typename... Args> __HOST_DEVICE__ bool parallel_1D(IvyTypes::size_t n, Args... args){
    IvyBlockThreadDim_t nreq_blocks, nreq_threads_per_block;
    if (IvyCudaConfig::check_GPU_usable(nreq_blocks, nreq_threads_per_block, n)){
      Kernel_t::prepare(true, n, args...);
      __CHECK_KERNEL_AND_WARN_WITH_ERROR__(__ENCAPSULATE__(generic_kernel_1D<Kernel_t, IvyTypes::size_t, Args...><<<nreq_blocks, nreq_threads_per_block, shared_mem_size, stream>>>(n, args...)));
      Kernel_t::finalize(true, n, args...);
      return true;
    }
    else return false;
  }
  /**
   * @brief Execute 2D kernel dispatch on GPU when usable.
   * @param nx X dimension.
   * @param ny Y dimension.
   * @param args Kernel arguments.
   * @return `true` if GPU dispatch ran, `false` otherwise.
   */
  template<typename... Args> __HOST_DEVICE__ bool parallel_2D(IvyTypes::size_t nx, IvyTypes::size_t ny, Args... args){
    IvyBlockThreadDim_t nreq_blocks, nreq_threads_per_block;
    if (IvyCudaConfig::check_GPU_usable(nreq_blocks, nreq_threads_per_block, nx*ny)){
      Kernel_t::prepare(true, nx, ny, args...);
      __CHECK_KERNEL_AND_WARN_WITH_ERROR__(__ENCAPSULATE__(generic_kernel_2D<Kernel_t, IvyTypes::size_t, Args...><<<nreq_blocks, nreq_threads_per_block, shared_mem_size, stream>>>(nx, ny, args...)));
      Kernel_t::finalize(true, nx, ny, args...);
      return true;
    }
    else return false;
  }
  /**
   * @brief Execute 3D kernel dispatch on GPU when usable.
   * @param nx X dimension.
   * @param ny Y dimension.
   * @param nz Z dimension.
   * @param args Kernel arguments.
   * @return `true` if GPU dispatch ran, `false` otherwise.
   */
  template<typename... Args> __HOST_DEVICE__ bool parallel_3D(IvyTypes::size_t nx, IvyTypes::size_t ny, IvyTypes::size_t nz, Args... args){
    IvyBlockThreadDim_t nreq_blocks, nreq_threads_per_block;
    if (IvyCudaConfig::check_GPU_usable(nreq_blocks, nreq_threads_per_block, nx*ny*nz)){
      Kernel_t::prepare(true, nx, ny, nz, args...);
      __CHECK_KERNEL_AND_WARN_WITH_ERROR__(__ENCAPSULATE__(generic_kernel_3D<Kernel_t, IvyTypes::size_t, Args...><<<nreq_blocks, nreq_threads_per_block, shared_mem_size, stream>>>(nx, ny, nz, args...)));
      Kernel_t::finalize(true, nx, ny, nz, args...);
      return true;
    }
    else return false;
  }
};
template<typename Kernel_t> struct run_kernel<Kernel_t, true> : run_kernel_base{
  /**
   * @brief Construct a dispatcher for kernels that provide `kernel_unit_unified` fallback.
   * @param shared_mem_size_ Dynamic shared-memory size for CUDA launches.
   * @param stream_ Stream used for launch and synchronization.
   */
  __HOST_DEVICE__ run_kernel(IvyTypes::size_t const& shared_mem_size_, IvyGPUStream& stream_) : run_kernel_base(shared_mem_size_, stream_){}

  /**
   * @brief Execute 1D dispatch with GPU path when available, otherwise CPU/OpenMP fallback via `kernel_unit_unified`.
   * @param n Number of logical elements.
   * @param args Kernel arguments.
   * @return Always `true` after completing either path.
   */
  template<typename... Args> __HOST_DEVICE__ bool parallel_1D(IvyTypes::size_t n, Args... args){
    IvyBlockThreadDim_t nreq_blocks, nreq_threads_per_block;
    if (IvyCudaConfig::check_GPU_usable(nreq_blocks, nreq_threads_per_block, n)){
      Kernel_t::prepare(true, n, args...);
      __CHECK_KERNEL_AND_WARN_WITH_ERROR__(__ENCAPSULATE__(generic_kernel_1D<Kernel_t, IvyTypes::size_t, Args...><<<nreq_blocks, nreq_threads_per_block, shared_mem_size, stream>>>(n, args...)));
      Kernel_t::finalize(true, n, args...);
    }
    else{
      Kernel_t::prepare(false, n, args...);
      #define _KERNEL_UNIT_CMD for (IvyTypes::size_t i = 0; i < n; ++i) Kernel_t::kernel_unit_unified(i, n, args...);
#if defined(OPENMP_ENABLED)
      if (n>=NUM_CPU_THREADS_THRESHOLD){
        #pragma omp parallel for schedule(static)
        _KERNEL_UNIT_CMD
      }
      else
#endif
      {
        _KERNEL_UNIT_CMD
      }
      #undef _KERNEL_UNIT_CMD
      Kernel_t::finalize(false, n, args...);
    }
    return true;
  }
  /**
   * @brief Execute 2D dispatch with GPU path when available, otherwise CPU/OpenMP fallback via `kernel_unit_unified`.
   * @param nx X dimension.
   * @param ny Y dimension.
   * @param args Kernel arguments.
   * @return Always `true` after completing either path.
   */
  template<typename... Args> __HOST_DEVICE__ bool parallel_2D(IvyTypes::size_t nx, IvyTypes::size_t ny, Args... args){
    IvyBlockThreadDim_t nreq_blocks, nreq_threads_per_block;
    if (IvyCudaConfig::check_GPU_usable(nreq_blocks, nreq_threads_per_block, nx*ny)){
      Kernel_t::prepare(true, nx, ny, args...);
      __CHECK_KERNEL_AND_WARN_WITH_ERROR__(__ENCAPSULATE__(generic_kernel_2D<Kernel_t, IvyTypes::size_t, Args...><<<nreq_blocks, nreq_threads_per_block, shared_mem_size, stream>>>(nx, ny, args...)));
      Kernel_t::finalize(true, nx, ny, args...);
    }
    else{
      Kernel_t::prepare(false, nx, ny, args...);
      #define _KERNEL_UNIT_CMD \
      for (IvyTypes::size_t i = 0; i < nx; ++i){ \
        for (IvyTypes::size_t j = 0; j < ny; ++j) Kernel_t::kernel_unit_unified(i, j, nx, ny, args...); \
      }
#if defined(OPENMP_ENABLED)
      IvyTypes::size_t nxy = nx*ny;
      if (nxy>=NUM_CPU_THREADS_THRESHOLD_ND){
        #pragma omp parallel for collapse(2) schedule(static)
        _KERNEL_UNIT_CMD
      }
      else
#endif
      {
        _KERNEL_UNIT_CMD
      }
      #undef _KERNEL_UNIT_CMD
      Kernel_t::finalize(false, nx, ny, args...);
    }
    return true;
  }
  /**
   * @brief Execute 3D dispatch with GPU path when available, otherwise CPU/OpenMP fallback via `kernel_unit_unified`.
   * @param nx X dimension.
   * @param ny Y dimension.
   * @param nz Z dimension.
   * @param args Kernel arguments.
   * @return Always `true` after completing either path.
   */
  template<typename... Args> __HOST_DEVICE__ bool parallel_3D(IvyTypes::size_t nx, IvyTypes::size_t ny, IvyTypes::size_t nz, Args... args){
    IvyBlockThreadDim_t nreq_blocks, nreq_threads_per_block;
    if (IvyCudaConfig::check_GPU_usable(nreq_blocks, nreq_threads_per_block, nx*ny*nz)){
      Kernel_t::prepare(true, nx, ny, nz, args...);
      __CHECK_KERNEL_AND_WARN_WITH_ERROR__(__ENCAPSULATE__(generic_kernel_3D<Kernel_t, IvyTypes::size_t, Args...><<<nreq_blocks, nreq_threads_per_block, shared_mem_size, stream>>>(nx, ny, nz, args...)));
      Kernel_t::finalize(true, nx, ny, nz, args...);
    }
    else{
      Kernel_t::prepare(false, nx, ny, nz, args...);
      #define _KERNEL_UNIT_CMD \
      for (IvyTypes::size_t i = 0; i < nx; ++i){ \
        for (IvyTypes::size_t j = 0; j < ny; ++j){ \
          for (IvyTypes::size_t k = 0; k < nz; ++k) Kernel_t::kernel_unit_unified(i, j, k, nx, ny, nz, args...); \
        } \
      }
#if defined(OPENMP_ENABLED)
      IvyTypes::size_t nxyz = nx*ny*nz;
      if (nxyz>=NUM_CPU_THREADS_THRESHOLD_ND){
        #pragma omp parallel for collapse(3) schedule(static)
        _KERNEL_UNIT_CMD
      }
      else
#endif
      {
        _KERNEL_UNIT_CMD
      }
      #undef _KERNEL_UNIT_CMD
      Kernel_t::finalize(false, nx, ny, nz, args...);
    }
    return true;
  }
};

#else

template<typename Kernel_t> struct run_kernel : run_kernel_base{
  /**
   * @brief Construct a CPU/OpenMP dispatcher for `Kernel_t`.
   * @param shared_mem_size_ Retained for interface parity; unused in non-CUDA path.
   * @param stream_ Stream reference retained for interface parity.
   */
  __HOST_DEVICE__ run_kernel(IvyTypes::size_t const& shared_mem_size_, IvyGPUStream& stream_) : run_kernel_base(shared_mem_size_, stream_){}

  /**
   * @brief Execute 1D kernel loop on CPU/OpenMP.
   * @param n Number of logical elements.
   * @param args Kernel arguments.
   * @return Always `true` after execution.
   */
  template<typename... Args> __HOST_DEVICE__ bool parallel_1D(IvyTypes::size_t n, Args... args){
    Kernel_t::prepare(false, n, args...);
    #define _KERNEL_CMD for (IvyTypes::size_t i = 0; i < n; ++i) Kernel_t::kernel(i, n, args...);
#if defined(OPENMP_ENABLED)
    if (n>=NUM_CPU_THREADS_THRESHOLD){
      #pragma omp parallel for schedule(static)
      _KERNEL_CMD
    }
    else
#endif
    {
      _KERNEL_CMD
    }
    #undef _KERNEL_CMD
    Kernel_t::finalize(false, n, args...);
    return true;
  }
  /**
   * @brief Execute 2D kernel loop on CPU/OpenMP.
   * @param nx X dimension.
   * @param ny Y dimension.
   * @param args Kernel arguments.
   * @return Always `true` after execution.
   */
  template<typename... Args> __HOST_DEVICE__ bool parallel_2D(IvyTypes::size_t nx, IvyTypes::size_t ny, Args... args){
    Kernel_t::prepare(false, nx, ny, args...);
    #define _KERNEL_CMD \
    for (IvyTypes::size_t i = 0; i < nx; ++i){ \
      for (IvyTypes::size_t j = 0; j < ny; ++j) Kernel_t::kernel(i, j, nx, ny, args...); \
    }
#if defined(OPENMP_ENABLED)
    IvyTypes::size_t nxy = nx*ny;
    if (nxy>=NUM_CPU_THREADS_THRESHOLD_ND){
      #pragma omp parallel for collapse(2) schedule(static)
      _KERNEL_CMD
    }
    else
#endif
    {
      _KERNEL_CMD
    }
    #undef _KERNEL_CMD
    Kernel_t::finalize(false, nx, ny, args...);
    return true;
  }
  /**
   * @brief Execute 3D kernel loop on CPU/OpenMP.
   * @param nx X dimension.
   * @param ny Y dimension.
   * @param nz Z dimension.
   * @param args Kernel arguments.
   * @return Always `true` after execution.
   */
  template<typename... Args> __HOST_DEVICE__ bool parallel_3D(IvyTypes::size_t nx, IvyTypes::size_t ny, IvyTypes::size_t nz, Args... args){
    Kernel_t::prepare(false, nx, ny, nz, args...);
    #define _KERNEL_CMD \
    for (IvyTypes::size_t i = 0; i < nx; ++i){ \
      for (IvyTypes::size_t j = 0; j < ny; ++j){ \
        for (IvyTypes::size_t k = 0; k < nz; ++k) Kernel_t::kernel(i, j, k, nx, ny, nz, args...); \
      } \
    }
#if defined(OPENMP_ENABLED)
    IvyTypes::size_t nxyz = nx*ny*nz;
    if (nxyz>=NUM_CPU_THREADS_THRESHOLD_ND){
      #pragma omp parallel for collapse(3) schedule(static)
      _KERNEL_CMD
    }
    else
#endif
    {
      _KERNEL_CMD
    }
    #undef _KERNEL_CMD
    Kernel_t::finalize(false, nx, ny, nz, args...);
    return true;
  }
};

#endif


#endif
