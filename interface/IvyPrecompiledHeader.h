/**
 * @file IvyPrecompiledHeader.h
 * @brief Precompiled header for g++ CPU builds only.
 *
 * When compiled with g++ (-x c++-header), produces IvyPrecompiledHeader.h.gch
 * which g++ uses automatically for any TU that includes this file first, provided
 * compiler flags are identical to those used when compiling the PCH.
 *
 * @note Has NO effect under nvcc. The CUDA toolchain does not support .gch.
 *       No PCHOUT, --pch-dir, or --pch-file variables exist in the CUDA makefile path.
 * @note Must be the FIRST include in every translation unit that uses it.
 */
#pragma once

#include "config/IvyCompilerConfig.h"
#include "IvyBasicTypes.h"
#include "IvyMemoryHelpers.h"
#include "IvyMemoryTypes.h"
#include "std_ivy/IvyCmath.h"
#include "std_ivy/IvyTypeTraits.h"
#include "std_ivy/IvyMemory.h"
#include "std_ivy/IvyVector.h"
#include "autodiff/base_types/IvyThreadSafePtr.h"
#include "autodiff/base_types/IvyBaseNode.h"
#include "autodiff/base_types/IvyClientManager.h"
#include "autodiff/basic_nodes/IvyScalar.h"
#include "autodiff/basic_nodes/IvyComplex.h"
#include "autodiff/basic_nodes/IvyTensor.h"
#include "autodiff/basic_nodes/IvyTensorShape.h"
#include "autodiff/arithmetic/IvyMathBaseArithmetic.h"
