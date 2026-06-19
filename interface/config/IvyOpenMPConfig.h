#ifndef IVYOPENMPCONFIG_H
#define IVYOPENMPCONFIG_H

/**
 * @file IvyOpenMPConfig.h
 * @brief OpenMP feature toggles for host-side parallel fallbacks.
 */


#include "IvyCudaFlags.h"


#if defined(_OPENMP) && !defined(__CUDA_DEVICE_CODE__)
  #include <omp.h>

  /** @brief Defined when host OpenMP support is available for this build. */
  #define OPENMP_ENABLED
  /**
   * @brief Minimum 1D loop size before enabling OpenMP parallel loops.
   * Override at build time with -DNUM_CPU_THREADS_THRESHOLD=<n>.
   */
  #ifndef NUM_CPU_THREADS_THRESHOLD
  #define NUM_CPU_THREADS_THRESHOLD 8
  #endif
  /**
   * @brief Minimum flattened size (nx*ny, nx*ny*nz) before enabling OpenMP for
   * multi-dimensional loops. Defaults to NUM_CPU_THREADS_THRESHOLD so the two
   * thresholds stay in sync; override at build time with
   * -DNUM_CPU_THREADS_THRESHOLD_ND=<n>.
   */
  #ifndef NUM_CPU_THREADS_THRESHOLD_ND
  #define NUM_CPU_THREADS_THRESHOLD_ND NUM_CPU_THREADS_THRESHOLD
  #endif
#endif


#endif
