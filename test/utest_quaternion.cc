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


static void test_matrix_views(){
  __PRINT_INFO__("--- quaternion matrix views (2x2 complex, 4x4 real) ---\n");
  IvyTensorShape probe_shape({ 2, 2 });
  auto stream = probe_shape.gpu_stream();
  auto mem = probe_shape.get_memory_type();

  // q = a + b i + c j + d k
  double const a = 1.0, b = 2.0, c = 3.0, d = 4.0;
  IvyQuaternion<double> q(a, b, c, d);
  double const nrm2 = a*a + b*b + c*c + d*d; // |q|^2 = 30

  // 2x2 complex SU(2)-style view: [[a+bi, c+di], [-c+di, a-bi]]
  auto M2 = to_complex_matrix(mem, stream, q);
  check(close((*M2)[{0,0}].Re(),a) && close((*M2)[{0,0}].Im(),b), "M2[0,0] = a + b i");
  check(close((*M2)[{0,1}].Re(),c) && close((*M2)[{0,1}].Im(),d), "M2[0,1] = c + d i");
  check(close((*M2)[{1,0}].Re(),-c) && close((*M2)[{1,0}].Im(),d), "M2[1,0] = -c + d i");
  check(close((*M2)[{1,1}].Re(),a) && close((*M2)[{1,1}].Im(),-b), "M2[1,1] = a - b i");
  // scalar part = (1/2) trace
  check(close(((*M2)[{0,0}].Re() + (*M2)[{1,1}].Re())/2.0, a), "0.5 * tr(M2) = scalar part");
  // det(M2) = |q|^2
  IvyComplex<double> det2 = (*M2)[{0,0}]*(*M2)[{1,1}] - (*M2)[{0,1}]*(*M2)[{1,0}];
  check(close(det2.Re(), nrm2) && close(det2.Im(), 0.0), "det(M2) = |q|^2");
  // conjugate quaternion <-> conjugate transpose of M2
  IvyQuaternion<double> qc = Conjugate(q);
  auto M2c = to_complex_matrix(mem, stream, qc);
  bool ct_ok = true;
  for (int i = 0; i < 2; ++i) for (int j = 0; j < 2; ++j){
    IvyComplex<double> lhs = (*M2c)[{i,j}];
    IvyComplex<double> rhs = (*M2)[{j,i}]; // conjugate-transpose: compare to conj of transposed entry
    ct_ok = ct_ok && close(lhs.Re(),rhs.Re()) && close(lhs.Im(),-rhs.Im());
  }
  check(ct_ok, "conjugate(q) <-> conjugate transpose of M2");

  // 4x4 real left-multiplication view.
  auto M4 = to_real_matrix(mem, stream, q);
  check(close((*M4)[{0,0}],a) && close((*M4)[{0,1}],-b) && close((*M4)[{0,2}],-c) && close((*M4)[{0,3}],-d), "M4 row 0");
  check(close((*M4)[{1,0}],b) && close((*M4)[{1,1}],a) && close((*M4)[{1,2}],-d) && close((*M4)[{1,3}],c), "M4 row 1");
  check(close((*M4)[{2,0}],c) && close((*M4)[{2,1}],d) && close((*M4)[{2,2}],a) && close((*M4)[{2,3}],-b), "M4 row 2");
  check(close((*M4)[{3,0}],d) && close((*M4)[{3,1}],-c) && close((*M4)[{3,2}],b) && close((*M4)[{3,3}],a), "M4 row 3");
  // scalar part = (1/4) trace
  double tr4 = (*M4)[{0,0}] + (*M4)[{1,1}] + (*M4)[{2,2}] + (*M4)[{3,3}];
  check(close(tr4/4.0, a), "0.25 * tr(M4) = scalar part");
  // M4(q) * vec(r) = vec(q*r)  (left Hamilton multiplication)
  IvyQuaternion<double> r(0.5, -1.0, 2.0, -0.5);
  double rv[4] = { r.W(), r.X(), r.Y(), r.Z() };
  double out[4] = {0,0,0,0};
  for (int i = 0; i < 4; ++i) for (int j = 0; j < 4; ++j) out[i] += (*M4)[{i,j}] * rv[j];
  IvyQuaternion<double> qr = q * r;
  check(close(out[0],qr.W()) && close(out[1],qr.X()) && close(out[2],qr.Y()) && close(out[3],qr.Z()),
        "M4(q) * vec(r) = vec(q*r) (left Hamilton product)");
  // conjugate quaternion <-> transpose of M4
  auto M4c = to_real_matrix(mem, stream, qc);
  bool t_ok = true;
  for (int i = 0; i < 4; ++i) for (int j = 0; j < 4; ++j) t_ok = t_ok && close((*M4c)[{i,j}], (*M4)[{j,i}]);
  check(t_ok, "conjugate(q) <-> transpose of M4");
}


void utest(){
  __PRINT_INFO__("=== utest_quaternion ===\n");
  test_value_algebra();
  test_autodiff();
  test_charts();
  test_matrix_views();
  __PRINT_INFO__("=== ALL utest_quaternion tests PASSED ===\n");
}


int main(){
  utest();
  return 0;
}
