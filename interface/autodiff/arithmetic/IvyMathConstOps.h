#ifndef IVYMATHCONSTOPS_H
#define IVYMATHCONSTOPS_H


#include "autodiff/arithmetic/IvyMathConstOps.hh"


namespace IvyMath{
  template<typename T, ENABLE_IF_ARITHMETIC_IMPL(T)> __HOST_DEVICE__ constexpr T Zero(){ return 0; }
  template<typename T, ENABLE_IF_ARITHMETIC_IMPL(T)> __HOST_DEVICE__ constexpr T OneHalf(){ return 0.5; }
  template<typename T, ENABLE_IF_ARITHMETIC_IMPL(T)> __HOST_DEVICE__ constexpr T One(){ return 1; }
  template<typename T, ENABLE_IF_ARITHMETIC_IMPL(T)> __HOST_DEVICE__ constexpr T Two() { return 2; }
  template<typename T, ENABLE_IF_ARITHMETIC_IMPL(T)> __HOST_DEVICE__ constexpr T MinusOneHalf(){ return -0.5; }
  template<typename T, ENABLE_IF_ARITHMETIC_IMPL(T)> __HOST_DEVICE__ constexpr T MinusOne(){ return -1; }
  template<typename T, ENABLE_IF_ARITHMETIC_IMPL(T)> __HOST_DEVICE__ constexpr T MinusTwo() { return -2; }
  template<typename T, ENABLE_IF_ARITHMETIC_IMPL(T)> __HOST_DEVICE__ constexpr T LogTwo(){ return 0.693147180559945309417232121458; }
  template<typename T, ENABLE_IF_ARITHMETIC_IMPL(T)> __HOST_DEVICE__ constexpr T LogTen(){ return 2.302585092994045684017991454684; }
  template<typename T, ENABLE_IF_ARITHMETIC_IMPL(T)> __HOST_DEVICE__ constexpr T PiOverTwo(){ return 1.57079632679489661923132169164; }
  template<typename T, ENABLE_IF_ARITHMETIC_IMPL(T)> __HOST_DEVICE__ constexpr T Pi(){ return 3.14159265358979323846264338328; }
  template<typename T, ENABLE_IF_ARITHMETIC_IMPL(T)> __HOST_DEVICE__ constexpr T TwoPi(){ return 6.28318530717958647692528676656; }
  template<typename T, ENABLE_IF_ARITHMETIC_IMPL(T)> __HOST_DEVICE__ constexpr T SqrtPi(){ return 1.77245385090551602729816748334; }
  template<typename T, ENABLE_IF_ARITHMETIC_IMPL(T)> __HOST_DEVICE__ constexpr T SqrtTwoPi(){ return 2.50662827463100050241576528481; }
  template<typename T, ENABLE_IF_ARITHMETIC_IMPL(T)> __HOST_DEVICE__ constexpr T SqrtTwoOverPi(){ return 0.797884560802865355879892119869; }
  template<typename T, ENABLE_IF_ARITHMETIC_IMPL(T)> __HOST_DEVICE__ constexpr T TwoSqrtPi(){ return 3.54490770181103205459633496668; }
  template<typename T, ENABLE_IF_ARITHMETIC_IMPL(T)> __HOST_DEVICE__ constexpr T TwoOverSqrtPi(){ return 1.12837916709551257389615890312; }
  template<typename T, ENABLE_IF_ARITHMETIC_IMPL(T)> __HOST_DEVICE__ constexpr T NtimesPi(T const& n){ return T(n)*Pi<T>(); }
  template<typename T, ENABLE_IF_ARITHMETIC_IMPL(T)> __HOST_DEVICE__ IvyComplex<T> UnitIm(){ return IvyComplex<T>(0, 1); }
}


#endif
