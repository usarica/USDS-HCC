#ifndef IVYMATHCONSTOPS_HH
#define IVYMATHCONSTOPS_HH


#include "config/IvyCompilerConfig.h"
#include "autodiff/basic_nodes/IvyComplex.h"
#include "std_ivy/IvyTypeTraits.h"


namespace IvyMath{
  template<typename T, ENABLE_IF_ARITHMETIC(T)> __HOST_DEVICE__ constexpr T Zero();
  template<typename T, ENABLE_IF_ARITHMETIC(T)> __HOST_DEVICE__ constexpr T OneHalf();
  template<typename T, ENABLE_IF_ARITHMETIC(T)> __HOST_DEVICE__ constexpr T One();
  template<typename T, ENABLE_IF_ARITHMETIC(T)> __HOST_DEVICE__ constexpr T Two();
  template<typename T, ENABLE_IF_ARITHMETIC(T)> __HOST_DEVICE__ constexpr T MinusOneHalf();
  template<typename T, ENABLE_IF_ARITHMETIC(T)> __HOST_DEVICE__ constexpr T MinusOne();
  template<typename T, ENABLE_IF_ARITHMETIC(T)> __HOST_DEVICE__ constexpr T MinusTwo();
  template<typename T, ENABLE_IF_ARITHMETIC(T)> __HOST_DEVICE__ constexpr T LogTwo();
  template<typename T, ENABLE_IF_ARITHMETIC(T)> __HOST_DEVICE__ constexpr T LogTen();
  template<typename T, ENABLE_IF_ARITHMETIC(T)> __HOST_DEVICE__ constexpr T PiOverTwo();
  template<typename T, ENABLE_IF_ARITHMETIC(T)> __HOST_DEVICE__ constexpr T Pi();
  template<typename T, ENABLE_IF_ARITHMETIC(T)> __HOST_DEVICE__ constexpr T TwoPi();
  template<typename T, ENABLE_IF_ARITHMETIC(T)> __HOST_DEVICE__ constexpr T SqrtPi();
  template<typename T, ENABLE_IF_ARITHMETIC(T)> __HOST_DEVICE__ constexpr T SqrtTwoPi();
  template<typename T, ENABLE_IF_ARITHMETIC(T)> __HOST_DEVICE__ constexpr T SqrtTwoOverPi();
  template<typename T, ENABLE_IF_ARITHMETIC(T)> __HOST_DEVICE__ constexpr T TwoSqrtPi();
  template<typename T, ENABLE_IF_ARITHMETIC(T)> __HOST_DEVICE__ constexpr T TwoOverSqrtPi();
  template<typename T, ENABLE_IF_ARITHMETIC(T)> __INLINE_FCN_FORCE__ __HOST_DEVICE__ constexpr T NtimesPi(T const& n);
  template<typename T, ENABLE_IF_ARITHMETIC(T)> __INLINE_FCN_FORCE__ __HOST_DEVICE__ IvyComplex<T> UnitIm();
}


#endif
