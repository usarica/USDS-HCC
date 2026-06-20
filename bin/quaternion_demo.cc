/**
 * @file quaternion_demo.cc
 * @brief Demonstrate the Hamilton quaternion algebra, its order-aware autodiff,
 *        the representation (chart) axis, and the two physics matrix views
 *        (2x2 complex / SU(2) and 4x4 real / left-multiplication). Rotations
 *        themselves belong to a physics layer built on top of this autodiff
 *        library and are intentionally out of scope here. Returns 0 when all
 *        checks pass.
 */

#include "IvyHCC.h"

#include <cmath>


int main(){
  using namespace std_ivy;
  using namespace IvyMath;

  constexpr auto MEM = IvyMemoryHelpers::get_execution_default_memory();
  IvyTensorShape probe_shape({ 2, 2 });
  auto stream = probe_shape.gpu_stream();
  bool ok = true;

  // --- 1) Quaternion algebra: non-commutative Hamilton product ---------------
  IvyQuaternion<double> i(0,1,0,0), j(0,0,1,0);
  auto ij = i*j;   // == k
  auto ji = j*i;   // == -k
  __PRINT_INFO__("i*j = (%.1f,%.1f,%.1f,%.1f), j*i = (%.1f,%.1f,%.1f,%.1f)\n",
                 ij.W(),ij.X(),ij.Y(),ij.Z(), ji.W(),ji.X(),ji.Y(),ji.Z());
  ok = ok && ij.Z()==1.0 && ji.Z()==-1.0;

  // --- 2) Order-aware autodiff: d(q*q)/dq = 2q ; non-commutative chain --------
  auto q = Quaternion<double>(MEM, nullptr, 1.0,2.0,3.0,4.0);
  IvyThreadSafePtr_t<IvyBaseNode> qv(q);
  auto g = (q*q)->gradient(qv)->value();
  __PRINT_INFO__("d(q*q)/dq = (%.1f,%.1f,%.1f,%.1f) expect (2,4,6,8)\n", g.W(),g.X(),g.Y(),g.Z());
  ok = ok && g.W()==2 && g.X()==4 && g.Y()==6 && g.Z()==8;

  // --- 3) Representation chart: exp/tangent quaternion with Jacobian ----------
  IvyQuaternion<double, quaternion_exp_chart> qe;
  qe.set_components(0.7, 0.1, -0.2, 0.3);
  __PRINT_INFO__("exp-chart canonical = (%.4f,%.4f,%.4f,%.4f), |q| = %.4f\n",
                 qe.W(),qe.X(),qe.Y(),qe.Z(), qe.norm());

  // --- 4) Matrix views: 2x2 complex (SU(2)/spin) and 4x4 real (rotation) ------
  IvyQuaternion<double> p(1.0, 2.0, 3.0, 4.0);   // a + b i + c j + d k
  double const nrm2 = 1.0+4.0+9.0+16.0;          // |p|^2 = 30

  // 2x2 complex view: [[a+bi, c+di], [-c+di, a-bi]]
  auto M2 = to_complex_matrix(MEM, stream, p);
  IvyComplex<double> det2 = (*M2)[{0,0}]*(*M2)[{1,1}] - (*M2)[{0,1}]*(*M2)[{1,0}];
  double scal2 = ((*M2)[{0,0}].Re() + (*M2)[{1,1}].Re())/2.0;   // (1/2) trace
  __PRINT_INFO__("M2: det = %.4f (= |p|^2 = %.4f), 0.5*tr = %.4f (= scalar %.1f)\n",
                 det2.Re(), nrm2, scal2, p.W());
  ok = ok && std::fabs(det2.Re()-nrm2)<1e-9 && std::fabs(scal2-p.W())<1e-9;

  // 4x4 real view: left Hamilton multiplication M4(p)*vec(r) == vec(p*r)
  auto M4 = to_real_matrix(MEM, stream, p);
  double tr4 = (*M4)[{0,0}] + (*M4)[{1,1}] + (*M4)[{2,2}] + (*M4)[{3,3}];
  IvyQuaternion<double> r(0.5,-1.0,2.0,-0.5);
  double rv[4] = { r.W(), r.X(), r.Y(), r.Z() }, out[4] = {0,0,0,0};
  for (int a=0;a<4;++a) for (int b=0;b<4;++b) out[a] += (*M4)[{a,b}]*rv[b];
  IvyQuaternion<double> pr = p*r;
  __PRINT_INFO__("M4: 0.25*tr = %.4f (= scalar %.1f); M4*vec(r) = (%.3f,%.3f,%.3f,%.3f) == p*r (%.3f,%.3f,%.3f,%.3f)\n",
                 tr4/4.0, p.W(), out[0],out[1],out[2],out[3], pr.W(),pr.X(),pr.Y(),pr.Z());
  ok = ok && std::fabs(tr4/4.0-p.W())<1e-9
          && std::fabs(out[0]-pr.W())<1e-9 && std::fabs(out[1]-pr.X())<1e-9
          && std::fabs(out[2]-pr.Y())<1e-9 && std::fabs(out[3]-pr.Z())<1e-9;

  __PRINT_INFO__(ok ? "quaternion_demo: ALL CHECKS PASSED\n" : "quaternion_demo: FAILED\n");
  return ok ? 0 : 1;
}
