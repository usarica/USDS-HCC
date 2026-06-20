#ifndef IVYQUATERNIONMATRIX_H
#define IVYQUATERNIONMATRIX_H


/**
 * @file IvyQuaternionMatrix.h
 * @brief Matrix views of a Hamilton quaternion in the two representations that
 *        recur in physics: the 2x2 complex (SU(2)/spin) form and the 4x4 real
 *        (left-multiplication / rotation) form.
 *
 * These are VIEWS, not charts: they do not change how a quaternion is stored or
 * differentiated, they only re-express the canonical components (w, x, y, z) as
 * an IvyTensor in a chosen matrix algebra. A physics layer built on top of this
 * autodiff library can use the 4x4 form for rotational logic, or the 2x2 form
 * for spin/SU(2) computations.
 *
 * Conventions (q = a + b i + c j + d k, with a = W, b = X, c = Y, d = Z):
 *
 *  2x2 complex:   q = [[ a + b i,  c + d i],
 *                      [-c + d i,  a - b i]]
 *    - norm(q)            = sqrt(det)
 *    - scalar part (a)    = (1/2) trace
 *    - conjugate(q)       <-> conjugate transpose
 *    - unit quaternions   <-> SU(2)  (the 3-sphere)
 *
 *  4x4 real:      q = [[ a, -b, -c, -d],
 *                      [ b,  a, -d,  c],
 *                      [ c,  d,  a, -b],
 *                      [ d, -c,  b,  a]]
 *    - norm(q)^4          = det
 *    - scalar part (a)    = (1/4) trace
 *    - conjugate(q)       <-> transpose
 *    - acts on (w,x,y,z)  as left Hamilton multiplication: M(p) * vec(q) = vec(p*q)
 */

#include "config/IvyCompilerConfig.h"
#include "autodiff/basic_nodes/IvyComplex.h"
#include "autodiff/basic_nodes/IvyQuaternion.h"
#include "autodiff/basic_nodes/IvyTensor.h"


namespace IvyMath{
  // Return-type aliases for the two quaternion matrix views.
  template<typename T> using IvyQuaternionComplexMatrix_t = IvyTensorPtr_t< IvyComplex<T> >;
  template<typename T> using IvyQuaternionRealMatrix_t    = IvyTensorPtr_t< T >;

  /**
   * @brief 2x2 complex-matrix view of a quaternion (SU(2)/spin representation).
   * @return A host IvyTensor of shape {2,2} with IvyComplex<T> entries.
   */
  template<typename T, typename Chart>
  __HOST__ IvyQuaternionComplexMatrix_t<T> to_complex_matrix(
    std_ivy::IvyMemoryType mem_type, IvyGPUStream* stream, IvyQuaternion<T, Chart> const& q
  ){
    T const a = q.W(), b = q.X(), c = q.Y(), d = q.Z();
    IvyTensorShape shape({ 2, 2 }, mem_type, stream);
    auto M = Tensor< IvyComplex<T> >(mem_type, stream, shape, IvyComplex<T>(T(0), T(0)));
    (*M)[{0, 0}] = IvyComplex<T>( a,  b);
    (*M)[{0, 1}] = IvyComplex<T>( c,  d);
    (*M)[{1, 0}] = IvyComplex<T>(-c,  d);
    (*M)[{1, 1}] = IvyComplex<T>( a, -b);
    return M;
  }
  /// @brief Pointer overload of the 2x2 complex-matrix view.
  template<typename T, typename Chart>
  __HOST__ IvyQuaternionComplexMatrix_t<T> to_complex_matrix(
    std_ivy::IvyMemoryType mem_type, IvyGPUStream* stream, IvyQuaternionPtr_t<T, Chart> const& q
  ){ return to_complex_matrix(mem_type, stream, *q); }

  /**
   * @brief 4x4 real-matrix view of a quaternion (left-multiplication / rotation form).
   * @return A host IvyTensor of shape {4,4} with real (T) entries.
   */
  template<typename T, typename Chart>
  __HOST__ IvyQuaternionRealMatrix_t<T> to_real_matrix(
    std_ivy::IvyMemoryType mem_type, IvyGPUStream* stream, IvyQuaternion<T, Chart> const& q
  ){
    T const a = q.W(), b = q.X(), c = q.Y(), d = q.Z();
    IvyTensorShape shape({ 4, 4 }, mem_type, stream);
    auto M = Tensor< T >(mem_type, stream, shape, T(0));
    (*M)[{0, 0}] = a; (*M)[{0, 1}] = -b; (*M)[{0, 2}] = -c; (*M)[{0, 3}] = -d;
    (*M)[{1, 0}] = b; (*M)[{1, 1}] =  a; (*M)[{1, 2}] = -d; (*M)[{1, 3}] =  c;
    (*M)[{2, 0}] = c; (*M)[{2, 1}] =  d; (*M)[{2, 2}] =  a; (*M)[{2, 3}] = -b;
    (*M)[{3, 0}] = d; (*M)[{3, 1}] = -c; (*M)[{3, 2}] =  b; (*M)[{3, 3}] =  a;
    return M;
  }
  /// @brief Pointer overload of the 4x4 real-matrix view.
  template<typename T, typename Chart>
  __HOST__ IvyQuaternionRealMatrix_t<T> to_real_matrix(
    std_ivy::IvyMemoryType mem_type, IvyGPUStream* stream, IvyQuaternionPtr_t<T, Chart> const& q
  ){ return to_real_matrix(mem_type, stream, *q); }
}


#endif
