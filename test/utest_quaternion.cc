/**
 * @file utest_quaternion.cc
 * @brief Unit tests for the Hamilton quaternion leaf (IvyQuaternion), its
 *        order-aware autodiff, the Abs functional, and the representation charts.
 *
 * Exercises:
 *  - Value-level algebra: Hamilton product (non-commutative i*j=k, j*i=-k),
 *    add/sub/negate, conjugate, multiplicative inverse, right division.
 *  - Abs(quaternion) == norm (real scalar).
 *  - Order-aware gradients (pointer/function graph): linear ops, q*q -> 2q,
 *    and finite-difference checks for non-commutative products, inverse, and
 *    division (the order-aware combine_gradient path).
 *  - Charts: components (canonical) and the 4-DOF exponential chart, validating
 *    the closed-form Jacobian against finite differences and exp/log round-trip.
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


static void test_value_algebra(){
  __PRINT_INFO__("--- quaternion value algebra ---\n");
  IvyQuaternion<double> a(1,2,3,4), b(5,6,7,8);

  auto p = a*b; // Hamilton product
  check(close(p.W(),-60) && close(p.X(),12) && close(p.Y(),30) && close(p.Z(),24), "a*b Hamilton product");

  IvyQuaternion<double> i(0,1,0,0), j(0,0,1,0);
  auto k = i*j; auto k2 = j*i;
  check(close(k.Z(),1) && close(k.W(),0), "i*j == k");
  check(close(k2.Z(),-1), "j*i == -k (non-commutative)");

  auto s = a+b; auto d = a-b; auto ng = -a;
  check(close(s.W(),6) && close(s.Z(),12), "a+b");
  check(close(d.X(),-4), "a-b");
  check(close(ng.Y(),-3), "-a");

  auto c = Conjugate(a);
  check(close(c.W(),1) && close(c.X(),-2) && close(c.Y(),-3) && close(c.Z(),-4), "conjugate");

  auto one = a*MultInverse(a);
  check(close(one.W(),1) && close(one.X(),0) && close(one.Y(),0) && close(one.Z(),0), "a * a^-1 == 1");

  auto back = (a/b)*b;
  check(close(back.W(),1) && close(back.X(),2) && close(back.Y(),3) && close(back.Z(),4), "(a/b)*b == a");

  auto an = Abs(a);
  check(close(an.value(), std::sqrt(30.0)), "Abs(a) == norm");
}


static int g_fails = 0;
template<typename Q> static void cmpq(char const* n, Q const& got, double w, double x, double y, double z){
  bool ok = close(got.W(),w,1e-5) && close(got.X(),x,1e-5) && close(got.Y(),y,1e-5) && close(got.Z(),z,1e-5);
  check(ok, n);
}

constexpr auto MEM = IvyMemoryHelpers::get_execution_default_memory();

template<typename Build>
static IvyQuaternion<double> fd_wrt_w(IvyQuaternionPtr_t<double> const& q, Build build){
  double w0 = q->W();
  double h = 1e-6;
  q->set_scalar(w0+h); auto fp = build(); IvyQuaternion<double> vp(fp->value());
  q->set_scalar(w0-h); auto fm = build(); IvyQuaternion<double> vm(fm->value());
  q->set_scalar(w0);
  return IvyQuaternion<double>((vp.W()-vm.W())/(2*h),(vp.X()-vm.X())/(2*h),(vp.Y()-vm.Y())/(2*h),(vp.Z()-vm.Z())/(2*h));
}

static void test_autodiff(){
  __PRINT_INFO__("--- quaternion order-aware autodiff ---\n");
  auto q = Quaternion<double>(MEM, nullptr, 1.0,2.0,3.0,4.0);
  auto c = Quaternion<double>(MEM, nullptr, 0.5,-1.0,2.0,0.3);
  IvyThreadSafePtr_t<IvyBaseNode> qv(q);

  { auto g = (q*c)->gradient(qv)->value(); cmpq("d(q*c)/dq == c", g, c->W(),c->X(),c->Y(),c->Z()); }
  { auto g = (c*q)->gradient(qv)->value(); cmpq("d(c*q)/dq == c", g, c->W(),c->X(),c->Y(),c->Z()); }
  { auto g = (q*q)->gradient(qv)->value(); cmpq("d(q*q)/dq == 2q", g, 2,4,6,8); }

  { auto an=(q*q*q)->gradient(qv)->value(); auto num=fd_wrt_w(q,[&](){return q*q*q;}); cmpq("d(q^3) vs FD",an,num.W(),num.X(),num.Y(),num.Z()); }
  { auto an=MultInverse(q)->gradient(qv)->value(); auto num=fd_wrt_w(q,[&](){return MultInverse(q);}); cmpq("d(q^-1) vs FD",an,num.W(),num.X(),num.Y(),num.Z()); }
  { auto an=(q/c)->gradient(qv)->value(); auto num=fd_wrt_w(q,[&](){return q/c;}); cmpq("d(q/c) vs FD",an,num.W(),num.X(),num.Y(),num.Z()); }
  { auto an=(c/q)->gradient(qv)->value(); auto num=fd_wrt_w(q,[&](){return c/q;}); cmpq("d(c/q) vs FD",an,num.W(),num.X(),num.Y(),num.Z()); }
  { auto an=((q*c)*q)->gradient(qv)->value(); auto num=fd_wrt_w(q,[&](){return (q*c)*q;}); cmpq("d((q*c)*q) vs FD (order matters)",an,num.W(),num.X(),num.Y(),num.Z()); }

  { auto an=(q+c)->gradient(qv)->value(); cmpq("d(q+c)/dq == 1",an,1,0,0,0); }
  { auto an=(q-c)->gradient(qv)->value(); cmpq("d(q-c)/dq == 1",an,1,0,0,0); }
  { auto an=Negate(q)->gradient(qv)->value(); cmpq("d(-q)/dq == -1",an,-1,0,0,0); }
}


static void test_charts(){
  __PRINT_INFO__("--- quaternion representation charts ---\n");
  // components chart is the default and the canonical representation.
  static_assert(std_ttraits::is_same_v<get_representation_t<IvyQuaternion<double>>, quaternion_components_chart>, "default chart");
  static_assert(std_ttraits::is_same_v<get_representation_t<IvyQuaternion<double, quaternion_exp_chart>>, quaternion_exp_chart>, "exp chart");

  // exp/log round-trip through the exponential chart.
  IvyQuaternion<double, quaternion_exp_chart> q;
  q.set_components(1.0, 0.2, -0.3, 0.4);
  check(close(q.W(),1.0) && close(q.X(),0.2) && close(q.Y(),-0.3) && close(q.Z(),0.4), "exp-chart exp(log(.)) round-trip");

  // closed-form Jacobian vs finite difference.
  quaternion_exp_chart::storage_t<double> s; s.t0=0.2; s.t1=0.3; s.t2=-0.5; s.t3=0.7;
  double w,x,y,z; quaternion_exp_chart::eval(s,w,x,y,z);
  double J[16]; quaternion_exp_chart::jacobian(s,J);
  double h=1e-6, maxerr=0;
  for (int kk=0; kk<4; ++kk){
    quaternion_exp_chart::storage_t<double> sp=s;
    double* p = (kk==0)?&sp.t0:(kk==1)?&sp.t1:(kk==2)?&sp.t2:&sp.t3;
    *p += h;
    double w2,x2,y2,z2; quaternion_exp_chart::eval(sp,w2,x2,y2,z2);
    double fd[4]={(w2-w)/h,(x2-x)/h,(y2-y)/h,(z2-z)/h};
    for (int ii=0; ii<4; ++ii){ double e=std::fabs(fd[ii]-J[ii*4+kk]); if(e>maxerr)maxerr=e; }
  }
  check(maxerr < 1e-4, "exp-chart Jacobian matches finite differences");
}


void utest(){
  __PRINT_INFO__("=== utest_quaternion ===\n");
  test_value_algebra();
  test_autodiff();
  test_charts();
  __PRINT_INFO__("=== ALL utest_quaternion tests PASSED ===\n");
}


int main(){
  utest();
  return 0;
}
