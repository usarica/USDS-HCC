/**
 * @file quaternion_versor_demo.cc
 * @brief Demonstrate the Hamilton quaternion algebra, its order-aware autodiff,
 *        the representation (chart) axis, and versor-based rotations with
 *        tangent-space (Lie) derivatives. Returns 0 when all checks pass.
 */

#include "IvyHCC.h"

#include <cmath>


int main(){
  using namespace std_ivy;
  using namespace IvyMath;

  constexpr auto MEM = IvyMemoryHelpers::get_execution_default_memory();
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

  // --- 4) Versor rotation + Lie tangent derivative ---------------------------
  auto Rz = IvyVersor<double>::from_axis_angle(0,0,1, M_PI/2);
  double rx,ry,rz; Rz.rotate(1,0,0, rx,ry,rz);
  __PRINT_INFO__("Rz(90) * x_hat = (%.4f,%.4f,%.4f) expect (0,1,0)\n", rx,ry,rz);
  ok = ok && std::fabs(rx)<1e-9 && std::fabs(ry-1)<1e-9 && std::fabs(rz)<1e-9;

  auto R = IvyVersor<double>::exp_map(0.3,-0.7,0.5);
  double J[9]; R.rotate(0.6,-0.2,0.9, rx,ry,rz); R.rotate_jacobian(0.6,-0.2,0.9, J);
  __PRINT_INFO__("R v = (%.4f,%.4f,%.4f); d(Rv)/domega row0 = (%.4f,%.4f,%.4f)\n",
                 rx,ry,rz, J[0],J[1],J[2]);

  __PRINT_INFO__(ok ? "quaternion_versor_demo: ALL CHECKS PASSED\n" : "quaternion_versor_demo: FAILED\n");
  return ok ? 0 : 1;
}
