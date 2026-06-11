/**
 * @file IvyAtomic.h
 * @brief Backend-selecting atomic wrapper that aliases CUDA or standard atomic facilities.
 */
#ifndef IVYATOMIC_H
#define IVYATOMIC_H


#ifdef __USE_CUDA__

#include "cuda_runtime.h"
#include <cuda/atomic>

// Map std_atomic onto cuda::std so the alias exposes the full standard atomic surface
// (atomic, atomic_ref, atomic_flag, memory_order, ...). These are the system-scoped
// heterogeneous atomics, matching the defaults of the cuda:: thread-scoped variants.
#ifndef std_atomic
#define std_atomic cuda::std
#endif

#else

#include <atomic>

#ifndef std_atomic
#define std_atomic std
#endif

#endif


#endif
