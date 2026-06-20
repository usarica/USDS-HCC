#include "autodiff/basic_nodes/IvyConstant.h"
#include "autodiff/basic_nodes/IvyVariable.h"
#include "autodiff/basic_nodes/IvyComplexVariable.h"
#include "autodiff/basic_nodes/IvyTensor.h"
#include "autodiff/arithmetic/IvyMathBaseArithmetic.h"


int main(){
  using namespace std_ivy;
  using namespace IvyMath;

  auto cplx = Complex<double>(IvyMemoryType::Host, nullptr, 1, 2);
  auto rvar = Variable<double>(IvyMemoryType::Host, nullptr, 3);
  auto rconst = Constant<double>(IvyMemoryType::Host, nullptr, 5);

  __PRINT_INFO__("cplx = "); print_value(cplx);
  __PRINT_INFO__("-cplx = "); print_value(-(*cplx));
  __PRINT_INFO__("exp(cplx) = "); print_value(Exp(*cplx));
  __PRINT_INFO__("ln(cplx) = "); print_value(Log(*cplx));
  __PRINT_INFO__("cplx* = "); print_value(Conjugate(*cplx));
  __PRINT_INFO__("Re(cplx) = "); print_value(Real(*cplx));
  __PRINT_INFO__("Im(cplx) = "); print_value(Imag(*cplx));

  IvyTensorShape tshape({2, 3, 4});
  tshape.print();
  auto tshape_slice1 = tshape.get_slice_shape(1);
  tshape_slice1.print();
  auto absi = tshape.get_abs_index({0, 1, 2});
  __PRINT_INFO__("Absolute index of tensor coordinates {0, 1, 2} = %llu, size = %llu\n", absi, tshape.num_elements());

  auto t3d = Tensor<double>(tshape.get_memory_type(), tshape.gpu_stream(), tshape, 5.);
  t3d->at({ 0, 1, 2 }) = 3.14;
  print_value(t3d);

  using NegateEvaluator = NegateFcnal<std_ttraits::remove_reference_t<decltype(*cplx)>>;
  auto grad_neg = NegateEvaluator::gradient(cplx);
  __PRINT_INFO__("grad_neg(%s) = ", typeid(grad_neg).name());
  print_value(grad_neg);

  auto grad_cplx_rvar = function_gradient<decltype(cplx)>::get(cplx, rvar);
  __PRINT_INFO__("grad_cplx_rvar(%s) = ", typeid(grad_cplx_rvar).name());
  print_value(grad_cplx_rvar);

  auto grad_cplx_cplx = function_gradient<decltype(cplx)>::get(cplx, cplx);
  __PRINT_INFO__("grad_cplx_cplx(%s) = ", typeid(grad_cplx_cplx).name());
  print_value(grad_cplx_cplx);

  auto fcn_negate_cplx = -cplx;
  __PRINT_INFO__("fcn_negate_cplx = ");
  print_value(fcn_negate_cplx);
  auto grad_fcn_negate_cplx = fcn_negate_cplx->gradient(rvar);
  __PRINT_INFO__("grad_fcn_negate_cplx(%s) = ", typeid(grad_fcn_negate_cplx).name());
  print_value(grad_fcn_negate_cplx);

#define FCN_TEST_COMMAND(FCN, VAR, EVAL_RES, GRAD_RES) \
  __PRINT_INFO__("*** Testing "#FCN"["); \
  print_value(VAR, false); \
  __PRINT_INFO__("] ***\n"); \
  { \
    auto fcn = FCN(VAR); \
    __PRINT_INFO__("fcn: "); \
    print_value(fcn, false); \
    __PRINT_INFO__(" ?= "); \
    print_value(EVAL_RES); \
    auto grad = fcn->gradient(VAR); \
    __PRINT_INFO__("grad: "); \
    print_value(grad, false); \
    __PRINT_INFO__(" ?= "); \
    print_value(GRAD_RES); \
    __PRINT_INFO__("Type of fcn: %s\n", typeid(fcn).name()); \
    __PRINT_INFO__("Type of fcn value: %s\n", typeid(fcn->value()).name()); \
    __PRINT_INFO__("Type of grad: %s\n", typeid(grad).name()); \
  } \
  __PRINT_INFO__("*****************\n");

  FCN_TEST_COMMAND(Exp, rvar, Constant<double>(IvyMemoryType::Host, nullptr, 20.0855), Constant<double>(IvyMemoryType::Host, nullptr, 20.0855));
  FCN_TEST_COMMAND(Log, rvar, Constant<double>(IvyMemoryType::Host, nullptr, 1.098612), Constant<double>(IvyMemoryType::Host, nullptr, 0.333333));
  FCN_TEST_COMMAND(Sin, rvar, Constant<double>(IvyMemoryType::Host, nullptr, 0.14112), Constant<double>(IvyMemoryType::Host, nullptr, -0.98999));
  FCN_TEST_COMMAND(Cos, rvar, Constant<double>(IvyMemoryType::Host, nullptr, -0.98999), Constant<double>(IvyMemoryType::Host, nullptr, -0.14112));
  FCN_TEST_COMMAND(Tan, rvar, Constant<double>(IvyMemoryType::Host, nullptr, -0.14255), Constant<double>(IvyMemoryType::Host, nullptr, 1.02032));
  FCN_TEST_COMMAND(Cot, rvar, Constant<double>(IvyMemoryType::Host, nullptr, -7.01525), Constant<double>(IvyMemoryType::Host, nullptr, -50.2138));
  FCN_TEST_COMMAND(Sec, rvar, Constant<double>(IvyMemoryType::Host, nullptr, -1.01011), Constant<double>(IvyMemoryType::Host, nullptr, 0.143987));
  FCN_TEST_COMMAND(Csc, rvar, Constant<double>(IvyMemoryType::Host, nullptr, 7.08617), Constant<double>(IvyMemoryType::Host, nullptr, 49.7113));
  FCN_TEST_COMMAND(SinH, rvar, Constant<double>(IvyMemoryType::Host, nullptr, 10.0179), Constant<double>(IvyMemoryType::Host, nullptr, 10.0677));
  FCN_TEST_COMMAND(CosH, rvar, Constant<double>(IvyMemoryType::Host, nullptr, 10.0677), Constant<double>(IvyMemoryType::Host, nullptr, 10.0179));
  FCN_TEST_COMMAND(Erf, rvar, Constant<double>(IvyMemoryType::Host, nullptr, 0.999978), Constant<double>(IvyMemoryType::Host, nullptr, 0.000139253));
  FCN_TEST_COMMAND(Erfc, rvar, Constant<double>(IvyMemoryType::Host, nullptr, 0.0000220905), Constant<double>(IvyMemoryType::Host, nullptr, -0.000139253));
  FCN_TEST_COMMAND(Faddeeva, rvar, Complex<double>(IvyMemoryType::Host, nullptr, 0.000123410, 0.201157), Complex<double>(IvyMemoryType::Host, nullptr, -0.000740459, -0.0785647));
  FCN_TEST_COMMAND(FaddeevaFast, rvar, Complex<double>(IvyMemoryType::Host, nullptr, 0.000123410, 0.201157), Complex<double>(IvyMemoryType::Host, nullptr, -0.000740459, -0.0785647));

  FCN_TEST_COMMAND(Exp, cplx, Complex<double>(IvyMemoryType::Host, nullptr, -1.1312, 2.47173), Complex<double>(IvyMemoryType::Host, nullptr, -1.1312, 2.47173));
  FCN_TEST_COMMAND(Log, cplx, Complex<double>(IvyMemoryType::Host, nullptr, 0.804719, 1.10715), Complex<double>(IvyMemoryType::Host, nullptr, 0.2, -0.4));
  FCN_TEST_COMMAND(Sin, cplx, Complex<double>(IvyMemoryType::Host, nullptr, 3.16578, 1.9596), Complex<double>(IvyMemoryType::Host, nullptr, 2.03272, -3.0519));
  FCN_TEST_COMMAND(Cos, cplx, Complex<double>(IvyMemoryType::Host, nullptr, 2.03272, -3.0519), Complex<double>(IvyMemoryType::Host, nullptr, -3.16578, -1.9596));
  FCN_TEST_COMMAND(Tan, cplx, Complex<double>(IvyMemoryType::Host, nullptr, 0.0338128, 1.01479), Complex<double>(IvyMemoryType::Host, nullptr, -0.0286628, 0.0686261));
  FCN_TEST_COMMAND(Cot, cplx, Complex<double>(IvyMemoryType::Host, nullptr, 0.0327978, -0.984329), Complex<double>(IvyMemoryType::Host, nullptr, -0.0321717, 0.0645676));
  FCN_TEST_COMMAND(Sec, cplx, Complex<double>(IvyMemoryType::Host, nullptr, 0.151176, 0.226974), Complex<double>(IvyMemoryType::Host, nullptr, -0.22522, 0.161087));
  FCN_TEST_COMMAND(Csc, cplx, Complex<double>(IvyMemoryType::Host, nullptr, 0.228375, -0.141363), Complex<double>(IvyMemoryType::Host, nullptr, 0.131658, 0.229433));
  FCN_TEST_COMMAND(SinH, cplx, Complex<double>(IvyMemoryType::Host, nullptr, -0.489056, 1.40312), Complex<double>(IvyMemoryType::Host, nullptr, -0.642148, 1.06861));
  FCN_TEST_COMMAND(CosH, cplx, Complex<double>(IvyMemoryType::Host, nullptr, -0.642148, 1.06861), Complex<double>(IvyMemoryType::Host, nullptr, -0.489056, 1.40312));
  FCN_TEST_COMMAND(Erf, cplx, Complex<double>(IvyMemoryType::Host, nullptr, -0.536644, -5.04914), Complex<double>(IvyMemoryType::Host, nullptr, -14.8142, 17.1522));
  FCN_TEST_COMMAND(Erfc, cplx, Complex<double>(IvyMemoryType::Host, nullptr, 1.536644, 5.04914), Complex<double>(IvyMemoryType::Host, nullptr, 14.8142, -17.1522));
  FCN_TEST_COMMAND(Faddeeva, cplx, Complex<double>(IvyMemoryType::Host, nullptr, 0.218492, 0.0929978), Complex<double>(IvyMemoryType::Host, nullptr, -0.0649940, 0.0684131));
  FCN_TEST_COMMAND(ErfFast, cplx, Complex<double>(IvyMemoryType::Host, nullptr, -0.536644, -5.04914), Complex<double>(IvyMemoryType::Host, nullptr, -14.8142, 17.1522));
  FCN_TEST_COMMAND(ErfcFast, cplx, Complex<double>(IvyMemoryType::Host, nullptr, 1.536644, 5.04914), Complex<double>(IvyMemoryType::Host, nullptr, 14.8142, -17.1522));
  FCN_TEST_COMMAND(FaddeevaFast, cplx, Complex<double>(IvyMemoryType::Host, nullptr, 0.218492, 0.0929978), Complex<double>(IvyMemoryType::Host, nullptr, -0.0649940, 0.0684131));
#undef FCN_TEST_COMMAND

  auto fcn_equal_cplx_rvar = Equal(cplx, rvar);
  __PRINT_INFO__("fcn_equal_cplx_rvar = ");
  print_value(fcn_equal_cplx_rvar);

  auto fcn_not_equal_cplx_rvar = Not(fcn_equal_cplx_rvar);
  __PRINT_INFO__("fcn_not_equal_cplx_rvar = ");
  print_value(fcn_not_equal_cplx_rvar);

  auto fcn_or = Or(fcn_not_equal_cplx_rvar, fcn_equal_cplx_rvar);
  __PRINT_INFO__("fcn_or = ");
  print_value(fcn_or);

  auto fcn_xor = XOr(fcn_not_equal_cplx_rvar, fcn_equal_cplx_rvar);
  __PRINT_INFO__("fcn_xor = ");
  print_value(fcn_xor);

  auto fcn_and = And(fcn_not_equal_cplx_rvar, fcn_equal_cplx_rvar);
  __PRINT_INFO__("fcn_and = ");
  print_value(fcn_and);

  auto fcn_gt_cplx_rvar = GT(cplx, rvar);
  __PRINT_INFO__("fcn_gt_cplx_rvar = ");
  print_value(fcn_gt_cplx_rvar);

  auto fcn_lt_cplx_rvar = LT(cplx, rvar);
  __PRINT_INFO__("fcn_lt_cplx_rvar = ");
  print_value(fcn_lt_cplx_rvar);

  auto fcn_geq_cplx_rvar = GEQ(cplx, rvar);
  __PRINT_INFO__("fcn_geq_cplx_rvar = ");
  print_value(fcn_geq_cplx_rvar);

  auto fcn_leq_cplx_rvar = LEQ(cplx, rvar);
  __PRINT_INFO__("fcn_leq_cplx_rvar = ");
  print_value(fcn_leq_cplx_rvar);

  auto fcn_add_cplx_rvar = cplx + rvar;
  __PRINT_INFO__("fcn_add_cplx_rvar = ");
  print_value(fcn_add_cplx_rvar);
  auto grad0_fcn_add_cplx_rvar = fcn_add_cplx_rvar->gradient(cplx);
  __PRINT_INFO__("grad0_fcn_add_cplx_rvar(%s) = ", typeid(grad0_fcn_add_cplx_rvar).name());
  print_value(grad0_fcn_add_cplx_rvar->value());
  auto grad1_fcn_add_cplx_rvar = fcn_add_cplx_rvar->gradient(rvar);
  __PRINT_INFO__("grad1_fcn_add_cplx_rvar(%s) = ", typeid(grad1_fcn_add_cplx_rvar).name());
  print_value(grad1_fcn_add_cplx_rvar->value());

  auto fcn_prod_cplx_rvar = cplx * rvar;
  __PRINT_INFO__("fcn_prod_cplx_rvar = ");
  print_value(fcn_prod_cplx_rvar);
  auto grad0_fcn_prod_cplx_rvar = fcn_prod_cplx_rvar->gradient(cplx);
  __PRINT_INFO__("grad0_fcn_prod_cplx_rvar(%s) = ", typeid(grad0_fcn_prod_cplx_rvar).name());
  print_value(grad0_fcn_prod_cplx_rvar->value());
  auto grad1_fcn_prod_cplx_rvar = fcn_prod_cplx_rvar->gradient(rvar);
  __PRINT_INFO__("grad1_fcn_prod_cplx_rvar(%s) = ", typeid(grad1_fcn_prod_cplx_rvar).name());
  print_value(grad1_fcn_prod_cplx_rvar->value());


  using MultiplyEvaluator = MultiplyFcnal<std_ttraits::remove_reference_t<decltype(*cplx)>, std_ttraits::remove_reference_t<decltype(*cplx)>>;
  auto prod_cplx_cplx = MultiplyEvaluator::eval(*cplx, *cplx);
  __PRINT_INFO__("prod_cplx_cplx = ");
  print_value(prod_cplx_cplx);

  auto grad0_prod_cplx_cplx = MultiplyEvaluator::gradient(0, cplx, cplx);
  __PRINT_INFO__("grad0_prod_cplx_cplx(%s) = ", typeid(grad0_prod_cplx_cplx).name());
  print_value(grad0_prod_cplx_cplx);
  auto grad1_prod_cplx_cplx = MultiplyEvaluator::gradient(1, cplx, cplx);
  __PRINT_INFO__("grad1_prod_cplx_cplx(%s) = ", typeid(grad1_prod_cplx_cplx).name());
  print_value(grad1_prod_cplx_cplx);

  auto fcn_manual_cubed = cplx*cplx*cplx;
  __PRINT_INFO__("fcn_manual_cubed = ");
  print_value(fcn_manual_cubed);
  auto grad_fcn_manual_cubed = fcn_manual_cubed->gradient(cplx);
  __PRINT_INFO__("grad_fcn_manual_cubed(%s) = ", typeid(grad_fcn_manual_cubed).name());
  print_value(grad_fcn_manual_cubed->value());

  auto fcn_pow_cubed = Pow(cplx, Constant<int>(IvyMemoryType::Host, nullptr, 3));
  __PRINT_INFO__("fcn_pow_cubed = ");
  print_value(fcn_pow_cubed);
  __PRINT_INFO__("fcn_pow_cubed addr. = %p\n", fcn_pow_cubed.get());
  auto grad_fcn_pow_cubed = fcn_pow_cubed->gradient(cplx);
  __PRINT_INFO__("grad_fcn_pow_cubed(%s) = ", typeid(grad_fcn_pow_cubed).name());
  print_value(grad_fcn_pow_cubed->value());
  // Client back-references are now weak (IvyWeakPtr); a weak client matches a shared owner
  // when they share the same control block. Search the client list by control-block identity.
  auto cplx_clients_contain = [&cplx](auto const& sp) -> bool{
    auto const& clients = cplx->get_clients();
    for (auto const& w : clients){ if (w.control_block() == sp.control_block()) return true; }
    return false;
  };
  bool found_cli = cplx_clients_contain(fcn_pow_cubed);
  __PRINT_INFO__("Found fcn_pow_cubed in cplx clients: %s ?= true\n", (found_cli ? "true" : "false"));
  found_cli = cplx_clients_contain(grad_fcn_pow_cubed);
  __PRINT_INFO__("Found grad_fcn_pow_cubed in cplx clients: %s ?= false\n", (found_cli ? "true" : "false"));

  auto pow_cplx_cplx = Pow(cplx, cplx);
  __PRINT_INFO__("pow_cplx_cplx = ");
  print_value(pow_cplx_cplx);
  auto grad_pow_cplx_cplx = pow_cplx_cplx->gradient(cplx);
  __PRINT_INFO__("grad_pow_cplx_cplx(%s) = ", typeid(grad_pow_cplx_cplx).name());
  print_value(grad_pow_cplx_cplx);


}