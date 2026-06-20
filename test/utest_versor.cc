/**
 * @file utest_versor.cc
 * @brief Unit tests for IvyVersor — a constrained unit quaternion (S^3 -> SO(3))
 *        with the so(3) Lie/exp-tangent chart as its native differentiation basis.
 *
 * Exercises:
 *  - Axis-angle construction and active vector rotation (Rz(90) x = y, etc.).
 *  - Exp/log round-trip on the rotation vector (so(3) <-> S^3) and angle().
 *  - Analytic tangent-space Jacobian d(Rv)/domega vs finite differences using a
 *    spatial (left) perturbation R -> exp([delta]_x) R.
 *  - Group composition and inverse (Rz(90) twice == Rz(180); inverse undoes it).
 *  - representation_t is the versor_lie_chart, and the versor stays normalized.
 */

#include "common_test_defs.h"

#include "IvyHCC.h"

#include <cmath>
#include "std_ivy/IvyCassert.h"


using namespace std_ivy;
using namespace IvyMath;


static void check(bool cond, char const* label){
  if (cond){ __PRINT_INFO__("  [PASS] %s\n", label); }
  else { __PRINT_INFO__("  [FAIL] %s\n", label); assert(false); }
}
static bool close(double a, double b, double tol = 1e-9){ return std::abs(a - b) < tol; }


static void test_rotation(){
  __PRINT_INFO__("--- versor rotation ---\n");
  auto qz = IvyVersor<double>::from_axis_angle(0,0,1, M_PI/2);
  double rx,ry,rz; qz.rotate(1,0,0, rx,ry,rz);
  check(close(rx,0,1e-12) && close(ry,1,1e-12) && close(rz,0,1e-12), "Rz(90) * x_hat == y_hat");

  check(close(qz.norm(), 1.0, 1e-12), "versor stays unit-norm");
  static_assert(std_ttraits::is_same_v<get_representation_t<IvyVersor<double>>, versor_lie_chart>, "lie chart");
}

static void test_exp_log(){
  __PRINT_INFO__("--- versor exp/log (so(3) chart) ---\n");
  double ox=0.3, oy=-0.7, oz=0.5;
  auto q = IvyVersor<double>::exp_map(ox,oy,oz);
  double lx,ly,lz; q.log_map(lx,ly,lz);
  check(close(lx,ox) && close(ly,oy) && close(lz,oz), "log(exp(omega)) == omega");
  check(close(q.angle(), std::sqrt(ox*ox+oy*oy+oz*oz)), "angle == |omega|");
}

static void test_tangent_jacobian(){
  __PRINT_INFO__("--- versor tangent-space Jacobian ---\n");
  auto q = IvyVersor<double>::exp_map(0.3,-0.7,0.5);
  double vx=0.6, vy=-0.2, vz=0.9;
  double J[9]; q.rotate_jacobian(vx,vy,vz, J);
  double r0x,r0y,r0z; q.rotate(vx,vy,vz, r0x,r0y,r0z);
  double h=1e-6, maxerr=0;
  for (int k=0;k<3;++k){
    double d[3]={0,0,0}; d[k]=h;
    auto dq = IvyVersor<double>::exp_map(d[0],d[1],d[2]);
    auto qp = dq.compose(q); // spatial (left) perturbation
    double r1x,r1y,r1z; qp.rotate(vx,vy,vz, r1x,r1y,r1z);
    double fd[3]={(r1x-r0x)/h,(r1y-r0y)/h,(r1z-r0z)/h};
    for (int i=0;i<3;++i){ double e=std::fabs(fd[i]-J[i*3+k]); if(e>maxerr)maxerr=e; }
  }
  check(maxerr < 1e-4, "d(Rv)/domega matches finite differences");
}

static void test_group(){
  __PRINT_INFO__("--- versor group composition/inverse ---\n");
  auto qz = IvyVersor<double>::from_axis_angle(0,0,1, M_PI/2);
  auto q180 = qz.compose(qz);
  double rx,ry,rz; q180.rotate(1,0,0, rx,ry,rz);
  check(close(rx,-1,1e-12) && close(ry,0,1e-12) && close(rz,0,1e-12), "Rz(90) o Rz(90) == Rz(180)");

  double tx,ty,tz; qz.rotate(1,0,0, tx,ty,tz);
  auto qi = qz.inverse(); qi.rotate(tx,ty,tz, rx,ry,rz);
  check(close(rx,1,1e-12) && close(ry,0,1e-12) && close(rz,0,1e-12), "inverse undoes rotation");
}


void utest(){
  __PRINT_INFO__("=== utest_versor ===\n");
  test_rotation();
  test_exp_log();
  test_tangent_jacobian();
  test_group();
  __PRINT_INFO__("=== ALL utest_versor tests PASSED ===\n");
}


int main(){
  utest();
  return 0;
}
