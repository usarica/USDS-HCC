#ifndef IVYCERF_HH
#define IVYCERF_HH


/*
File modified from https://arxiv.org/abs/1407.0748
to use CUDA cmath and IvyComplex.
This file is meant to be inserted, not to be used alone.
*/


#include "config/IvyCompilerConfig.h"
#include "autodiff/basic_nodes/IvyComplex.h"


namespace IvyCerf{
  using namespace IvyMath;

  /** @brief evaluate Faddeeva function for complex argument
  *
  * @author Manuel Schiller <manuel.schiller@nikhef.nl>
  * @date 2013-02-21
  *
  * Calculate the value of the Faddeeva function @f$w(z) = \exp(-z^2)
  * \mathrm{erfc}(-i z)@f$.
  *
  * The method described in
  *
  * S.M. Abrarov, B.M. Quine: "Efficient algorithmic implementation of
  * Voigt/complex error function based on exponential series approximation"
  * published in Applied Mathematics and Computation 218 (2011) 1894-1902
  * doi:10.1016/j.amc.2011.06.072
  *
  * is used. At the heart of the method (equation (14) of the paper) is the
  * following Fourier series based approximation:
  *
  * @f[ w(z) \approx \frac{i}{2\sqrt{\pi}}\left(
  * \sum^N_{n=0} a_n \tau_m\left(
  * \frac{1-e^{i(n\pi+\tau_m z)}}{n\pi + \tau_m z} -
  * \frac{1-e^{i(-n\pi+\tau_m z)}}{n\pi - \tau_m z}
  * \right) - a_0 \frac{1-e^{i \tau_m z}}{z}
  * \right) @f]
  *
  * The coefficients @f$a_b@f$ are given by:
  *
  * @f[ a_n=\frac{2\sqrt{\pi}}{\tau_m}
  * \exp\left(-\frac{n^2\pi^2}{\tau_m^2}\right) @f]
  *
  * To achieve machine accuracy in T precision floating point arithmetic
  * for most of the upper half of the complex plane, chose @f$t_m=12@f$ and
  * @f$N=23@f$ as is done in the paper.
  *
  * There are two complications: For Im(z) negative, the exponent in the
  * equation above becomes so large that the roundoff in the rest of the
  * calculation is amplified enough that the result cannot be trusted.
  * Therefore, for Im(z) < 0, the symmetry of the erfc function under the
  * transformation z --> -z is used to avoid accuracy issues for Im(z) < 0 by
  * formulating the problem such that the calculation can be done for Im(z) > 0
  * where the accuracy of the method is fine, and some postprocessing then
  * yields the desired final result.
  *
  * Second, the denominators in the equation above become singular at
  * @f$z = n * pi / 12@f$ (for 0 <= n < 24). In a tiny disc around these
  * points, Taylor expansions are used to overcome that difficulty.
  *
  * This routine precomputes everything it can, and tries to write out complex
  * operations to minimise subroutine calls, e.g. for the multiplication of
  * complex numbers.
  *
  * In the square -8 <= Re(z) <= 8, -8 <= Im(z) <= 8, the routine is accurate
  * to better than 4e-13 relative, the average relative error is better than
  * 7e-16. On a modern x86_64 machine, the routine is roughly three times as
  * fast than the old CERNLIB implementation and offers better accuracy.
  */
  template<typename T> __HOST_DEVICE__ IvyComplex<T> faddeeva(IvyComplex<T> const& z);
  /** @brief evaluate Faddeeva function for complex argument (fast version)
  *
  * @author Manuel Schiller <manuel.schiller@nikhef.nl>
  * @date 2013-02-21
  *
  * Calculate the value of the Faddeeva function @f$w(z) = \exp(-z^2)
  * \mathrm{erfc}(-i z)@f$.
  *
  * This is the "fast" version of the faddeeva routine above. Fast means that
  * is takes roughly half the amount of CPU of the slow version of the
  * routine, but is a little less accurate.
  *
  * To be fast, chose @f$t_m=8@f$ and @f$N=11@f$ which should give accuracies
  * around 1e-7.
  *
  * In the square -8 <= Re(z) <= 8, -8 <= Im(z) <= 8, the routine is accurate
  * to better than 4e-7 relative, the average relative error is better than
  * 5e-9. On a modern x86_64 machine, the routine is roughly five times as
  * fast than the old CERNLIB implementation, or about 30% faster than the
  * interpolation/lookup table based fast method used previously in RooFit,
  * and offers better accuracy than the latter (the relative error is roughly
  * a factor 280 smaller than the old interpolation/table lookup routine).
  */
  template<typename T> __HOST_DEVICE__ IvyComplex<T> faddeeva_fast(IvyComplex<T> const& z);

  /** @brief complex erf function
  *
  * @author Manuel Schiller <manuel.schiller@nikhef.nl>
  * @date 2013-02-21
  *
  * Calculate erf(z) for complex z.
  */
  template<typename T> __HOST_DEVICE__ IvyComplex<T> erf(IvyComplex<T> const& z);

  /** @brief complex erf function (fast version)
  *
  * @author Manuel Schiller <manuel.schiller@nikhef.nl>
  * @date 2013-02-21
  *
  * Calculate erf(z) for complex z. Use the code in faddeeva_fast to save some time.
  */
  template<typename T> __HOST_DEVICE__ IvyComplex<T> erf_fast(IvyComplex<T> const& z);
  /** @brief complex erfc function
  *
  * @author Manuel Schiller <manuel.schiller@nikhef.nl>
  * @date 2013-02-21
  *
  * Calculate erfc(z) for complex z.
  */
  template<typename T> __HOST_DEVICE__ IvyComplex<T> erfc(IvyComplex<T> const& z);
  /** @brief complex erfc function (fast version)
  *
  * @author Manuel Schiller <manuel.schiller@nikhef.nl>
  * @date 2013-02-21
  *
  * Calculate erfc(z) for complex z. Use the code in faddeeva_fast to save some time.
  */
  template<typename T> __HOST_DEVICE__ IvyComplex<T> erfc_fast(IvyComplex<T> const& z);
}

#endif
