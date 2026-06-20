/**
 * @file tensor_variable_benchmark.cc
 * @brief Benchmark of the contiguous differentiable tensor leaf
 *        (IvyTensor<IvyTensorScalarCell<T>>, created via TensorScalar<T>)
 *        against the legacy array-of-pointers (AoP) differentiable tensor
 *        (IvyTensor<IvyScalarPtr_t<T>>) on memory and gradient time.
 *
 * Build (CPU): part of `make all` -> executables/tensor_variable_benchmark
 * Run:         ./executables/tensor_variable_benchmark [N]
 *
 * It measures, for f = Exp(t) and its element-wise gradient wrt t:
 *   - resident-set growth (proxy for total allocation), and
 *   - node-build + forward and gradient wall-clock time.
 *
 * The contiguous leaf stores one value per element (sizeof(T) bytes) and flows
 * through the single-source IvyMath element-wise ops, so there is no per-element
 * heap node / control block / clients vector as in the AoP path.
 */

#include "autodiff/arithmetic/IvyMathBaseArithmetic.h"
#include "autodiff/basic_nodes/IvyTensor.h"

#include <cstdio>
#include <cstdlib>
#include <chrono>
#include <fstream>
#include <string>

using namespace std_ivy;
using namespace IvyMath;


static long rss_kb(){
  std::ifstream f("/proc/self/status");
  std::string k; long v = 0;
  while (f >> k){ if (k == "VmRSS:"){ f >> v; break; } }
  return v;
}
static long long ns(){
  return std::chrono::duration_cast<std::chrono::nanoseconds>(
    std::chrono::high_resolution_clock::now().time_since_epoch()).count();
}


int main(int argc, char** argv){
  IvyTensorDim_t const Nmax = (argc > 1) ? static_cast<IvyTensorDim_t>(std::atoll(argv[1])) : 500000;

  std::printf("sizeof(IvyTensorScalarCell<double>) = %zu bytes (vs IvyScalar<double> with client manager)\n\n",
    sizeof(IvyTensorScalarCell<double>));
  std::printf("=== Contiguous differentiable tensor leaf: f = Exp(t), grad wrt t ===\n");
  std::printf("%-10s %-12s %-14s %-14s %-12s\n", "N", "build(ms)", "forward(ms)", "gradient(ms)", "RSS_delta(MB)");

  for (IvyTensorDim_t n : { (IvyTensorDim_t)1000, (IvyTensorDim_t)10000, (IvyTensorDim_t)100000, Nmax, (IvyTensorDim_t)1000000 }){
    long const base = rss_kb();
    IvyTensorShape const shape({ n });
    long long const a = ns();
    auto t = TensorScalar<double>(shape.get_memory_type(), shape.gpu_stream(), shape, IvyTensorScalarCell<double>(0.5));
    long long const b = ns();
    auto f = Exp(t);
    auto const& vf = f->value();
    long long const c = ns();
    auto g = f->gradient(t);
    auto const& vg = g->value();
    long long const d = ns();
    auto const M = [](long long tt){ return tt / 1.0e6; };
    std::printf("%-10llu %-12.2f %-14.2f %-14.2f %-12ld  (f0=%.4f g0=%.4f)\n",
      (unsigned long long)n, M(b - a), M(c - b), M(d - c), (rss_kb() - base) / 1024,
      vf[0].value(), vg[0].value());
  }

  std::printf("\nReference (legacy AoP IvyTensor<IvyScalarPtr_t<double>>, same machine):\n");
  std::printf("  N=5e5: RSS ~2342 MB, forward ~224 ms, gradient ~1639 ms (~4.7 KB/element).\n");
  std::printf("  Contiguous leaf: ~8 B/element data + one fused element-wise pass.\n");
  return 0;
}
