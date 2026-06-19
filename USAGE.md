#  USDS-HCC — Usage Guide

## 1. Overview

`USDS-HCC` is a header-only C++20 autodiff library that generalizes the
C++ Standard Library for heterogeneous CPU+GPU environments. It provides a lazy
autodiff graph whose nodes can live in host memory and whose element-wise evaluation
kernels execute on either the CPU or CUDA devices, controlled entirely at compile
time via domain tag dispatch. The intended audience is C++20 developers building
physics-simulation, machine-learning, or numerical-analysis tools that must run on
both CPUs and NVIDIA GPUs without duplicating algorithmic logic.

## 2. Requirements

- C++20-compliant compiler: g++ ≥ 11 (CPU builds), nvcc ≥ 12.0 (GPU/CUDA builds)
- CUDA Toolkit ≥ 12.0 (optional — required only for GPU support)
- GNU Make

## 3. Building

### CPU only

```bash
make pch && make utests
```

`make pch` compiles `interface/IvyPrecompiledHeader.h` into a `.gch` binary, eliminating
repeated parsing of heavy autodiff headers across translation units. The makefile
injects `-j$(nproc)` automatically, so all available cores are used.

### With CUDA

```bash
USE_CUDA=1 make utests
```

PCH is not used for CUDA builds — nvcc 12.x does not support precompiled headers.

### Clean

```bash
make distclean
```

Removes all build artifacts including test executables and the `.gch` PCH file.

### One-line acceptance harness (non-interactive)

```bash
make diagnostics-baseline
```

This command runs CPU/CUDA baseline diagnostics, warning inventory generation,
and writes structured logs under `strict_logs/` with no interactive prompts.
Use it to capture reproducible acceptance evidence in one command.

## 4. Running Tests

```bash
./test_executables/utest_<name>
```

| Executable | What it tests |
|---|---|
| `utest_IvyBucketedIteratorBuilder_basic` | Bucketed iterator construction and traversal |
| `utest_IvyContiguousIterator_basic` | Contiguous memory iterator correctness |
| `utest_IvyUnorderedMap_basic` | Custom CUDA-aware unordered map |
| `utest_IvyVector_basic` | Custom CUDA-aware dynamic vector |
| `utest_autodiff_basic_blocks` | Variable, Constant, Tensor construction; Exp, Log, Exp on complex |
| `utest_basic_alloc_copy_dealloc` | Host/device memory allocation, copy, and deallocation |
| `utest_complex_erf_gradient` | Complex-domain `erf(z)` evaluation and gradient `(2/√π)·exp(−z²)` |
| `utest_faddeeva_real` | Faddeeva function `w(x)` for real `x` and its gradient |
| `utest_hash` | Hashing utilities |
| `utest_sum_parallel` | Parallel reduction / sum |
| `utest_tensor_1d_fns` | Tensor-domain `MultInverse`, `Sqrt`, `Abs`, `Tan`, `Erf` eval and gradients |
| `utest_tensor_faddeeva` | Tensor-domain `Faddeeva` eval and per-element gradient |
| `utest_tensor_function` | Tensor-domain `Exp`, `Log`, `Sin`, `Cos`, `Negate` eval and gradients |
| `utest_tensor_pow` | Tensor-domain `Pow(t1, t2)` eval and gradients w.r.t. base and exponent |
| `utest_unified_ptr_basic` | `IvyThreadSafePtr_t` construction, copy, move semantics |
| `utest_unified_ptr_transfer` | Host-to-device and device-to-host pointer transfer |

Run all unit tests in one command:

```bash
for t in ./test_executables/utest_*; do "$t"; done
```

Run the gradient ergonomics demo:

```bash
./executables/gradient_ergonomics_demo
```

## 5. Library Architecture

- **Header-only:** Include the relevant header and the compiler does the rest. No
  linking to a compiled core library is required.
- **Domain tag dispatch:** Every math type carries a domain tag
  (`real_domain_tag`, `complex_domain_tag`, `tensor_domain_tag`,
  `arithmetic_domain_tag`). Functor specializations select the correct `eval()` and
  `gradient()` implementation at compile time via these tags, with no runtime
  branching.
- **Lazy autodiff graph:** Pointer-accepting function overloads (e.g., `Exp(ptr)`)
  return an `IvyRegularFunction_1D<T, Evaluator>` node that records its inputs.
  Calling `->gradient(var)` traverses the graph and returns the derivative node.
  Direct-value overloads (e.g., `Exp(scalar)`) evaluate immediately and return a
  plain value — no graph node is created.
- **Smart pointers:** `IvyThreadSafePtr_t<T>` (aliased as `IvyVariablePtr_t<T>`,
  `IvyFunctionPtr_t<T>`, etc.) is a reference-counted smart pointer that tracks
  clients and supports host/device transfer. It behaves like `std::shared_ptr<T>`
  but carries CUDA memory-type metadata.
- **`__HOST__` / `__HOST_DEVICE__` annotations:** Graph-building functions
  (pointer-accepting overloads) are always `__HOST__` only — autodiff graph nodes
  use virtual dispatch and RAII and must live in host memory. Direct-value eval
  functions are `__HOST_DEVICE__` for arithmetic and real/complex types, and
  `__HOST__` for tensor types (tensor eval uses STL containers).

## 6. Usage Examples

### a) Scalar autodiff — `d/dx [sin(x) · exp(x)]` at `x = 1.0`

```cpp
#include "autodiff/arithmetic/IvyMathBaseArithmetic.h"
using namespace std_ivy;
using namespace IvyMath;

// Create a real-domain variable x = 1.0
auto x = Variable<double>(IvyMemoryType::Host, nullptr, 1.0);

// Build the lazy function graph: f(x) = sin(x) * exp(x)
auto sin_x = Sin(x);          // IvyRegularFunction_1D graph node
auto exp_x = Exp(x);          // IvyRegularFunction_1D graph node
auto f     = Multiply(sin_x, exp_x);  // IvyRegularFunction_2D graph node

// Evaluate f(1.0)
double val = f->value()->value();    // sin(1)*exp(1) ≈ 2.2874

// Compute df/dx — returns a new function graph node for the derivative
auto df_dx = f->gradient(x);
double grad = df_dx->value()->value();  // (sin(1)+cos(1))*exp(1) ≈ 3.7560
```

### b) Tensor autodiff — element-wise `Pow(t, 2.0)` and gradients

```cpp
#include "autodiff/arithmetic/IvyMathBaseArithmetic.h"
#include "autodiff/basic_nodes/IvyTensor.h"
using namespace std_ivy;
using namespace IvyMath;

// Scalar variable shared across all tensor elements
auto x = Variable<double>(IvyMemoryType::Host, nullptr, 2.0);

// Build a 2×3 tensor where every element points to x
IvyTensorShape shape({ 2, 3 });
auto t = Tensor<IvyVariablePtr_t<double>>(
    shape.get_memory_type(), shape.gpu_stream(), shape, x
);

// Build tensor of exponents (all 2.0)
auto y = Variable<double>(IvyMemoryType::Host, nullptr, 2.0);
auto t2 = Tensor<IvyVariablePtr_t<double>>(
    shape.get_memory_type(), shape.gpu_stream(), shape, y
);

// Construct the graph node: each element is x^2
auto fcn = Pow(t, t2);        // IvyRegularFunction_1D<tensor> node

// Evaluate: all elements should be 2^2 = 4.0
auto const& val = fcn->value();
double elem0 = val[0]->value();  // 4.0

// Compute gradient wrt base (d/dt [t^y] = y·t^(y-1) = 2·2^1 = 4.0)
// Note: PowFcnal<tensor,tensor> uses direct gradient(), not the lazy graph
using pow_fcnal_t = PowFcnal<
    IvyTensor<IvyVariablePtr_t<double>>,
    IvyTensor<IvyVariablePtr_t<double>>,
    tensor_domain_tag, tensor_domain_tag>;
auto g0 = pow_fcnal_t::gradient(0, t, t2);  // y·x^(y-1) = 4.0 per element
```

### c) Complex erf gradient — `d/dz [erf(z)]` at `z = 1 + 2i`

```cpp
#include "autodiff/arithmetic/IvyMathBaseArithmetic.h"
#include "autodiff/basic_nodes/IvyComplexVariable.h"
using namespace std_ivy;
using namespace IvyMath;

// Create complex variable z = 1 + 2i
auto z = Complex<double>(IvyMemoryType::Host, nullptr, 1.0, 2.0);

// Build the lazy graph node for erf(z)
auto fcn_erf = Erf(z);

// Evaluate erf(1+2i) — returns an IvyComplexVariable
auto const& erf_val = fcn_erf->value();
// erf_val.Re() and erf_val.Im() give the real and imaginary parts

// Compute gradient: d/dz erf(z) = (2/√π) · exp(−z²)
// At z=1+2i: ≈ −14.8142 + 17.1522i
auto grad = fcn_erf->gradient(z);
auto const& gval = grad->value();
double dRe = gval.Re();  // ≈ −14.8142
double dIm = gval.Im();  // ≈  17.1522
```

### d) Function-chain target and intermediate-node usage

```cpp
#include "autodiff/arithmetic/IvyMathBaseArithmetic.h"
using namespace IvyMath;

auto x = Variable<double>(IvyMemoryType::Host, nullptr, 1.0);
auto u = Exp(x);
auto h = Sin(u);

// Variable target (recommended production pattern)
auto dh_dx = h->gradient(x);
double gx = dh_dx->value().value(); // cos(exp(1)) * exp(1)

// Function target (supported ergonomics path)
auto dh_du = h->gradient(u);
double gu = dh_du->value().value(); // cos(exp(1))
```

## 7. Shared-library consumption

### Build shared library

```bash
make lib
```

or CUDA-enabled:

```bash
USE_CUDA=1 make lib
```

The generated library is `lib/libIvyHCC.so`.

### Link against the shared library

```bash
g++ -std=c++20 -O2 -I./interface \
  -L./lib -Wl,-rpath,'$ORIGIN/lib' \
  -lIvyHCC your_app.cc -o your_app
```

### Smoke-test dynamic linking

```bash
make all && ./executables/shared_library_smoke
```

## 8. Adding a New Math Function

1. **Declare `FooFcnal<T>`** in `interface/autodiff/arithmetic/IvyMathBaseArithmetic.hh`.
   Define the nested type aliases `value_t` and `grad_t`, and declare static methods
   `eval()` and `gradient()`.
2. **Implement for each domain** (arithmetic, real, complex, tensor) in
   `interface/autodiff/arithmetic/IvyMathBaseArithmetic.h`. Follow the existing
   pattern for the domain you are targeting.
3. **For the tensor domain:** if no closed-form analytic gradient exists, declare
   `using gradient_domain_tag = undefined_domain_tag;` inside `FooFcnal<T, tensor_domain_tag>`.
   This prevents `IvyRegularFunction_1D` from instantiating a non-existent gradient path.
4. **Annotate correctly:**
   - All pointer-accepting (graph-building) overloads → `__HOST__`
   - Direct-value overloads for arithmetic/real/complex → `__HOST_DEVICE__`
   - Direct-value overloads for tensor → `__HOST__`
5. **Add free functions** `Foo(x)` and/or operator overloads in both
   `IvyMathBaseArithmetic.hh` (declaration) and `IvyMathBaseArithmetic.h` (implementation).
6. **Write a unit test** `test/utest_foo.cc` and add the corresponding target to the
   `TESTS` list in the `makefile`.
7. **Verify:** run `make utests` (CPU) and `USE_CUDA=1 make utests` (CUDA). Confirm
   that `grep -c "warning #20011" <cuda_log>` returns 0. A non-zero count means a
   graph-building overload is incorrectly annotated `__HOST_DEVICE__`.

## 9. Known Limitations

- **`Abs` and `Faddeeva` tensor gradients are disabled:** both declare
  `gradient_domain_tag = undefined_domain_tag`. `Abs` is not differentiable at zero;
  `Faddeeva` returns a complex-valued result for real input, which the current
  real-domain gradient graph cannot represent. The eval functions work correctly.
- **CUDA precompiled headers are not supported:** `nvcc --pch` is a fatal error in
  nvcc 12.x. The PCH speedup (`make pch`) applies to CPU (g++) builds only.
- **`ccache` is not integrated:** repeated CUDA builds re-compile from scratch. On a
  48-core machine, a parallel `USE_CUDA=1 make utests` takes approximately 3m 34s.


## 10. Architecture and Data-Flow (Acceptance-Oriented)

### 9.1 STL-generalization layer

- Headers: `interface/std_ivy/*`
- Core types: `IvyVector`, `IvyUnorderedMap`, `IvyUnifiedPtr`, iterator builders
- Data flow: API call -> allocator traits -> `IvyMemoryHelpers` -> stream/memory-domain operations

### 9.2 CPU/GPU memory dispatch layer

- Policy header: `interface/config/IvyAnnotationDispatchPolicy.h`
- Numeric/evaluable paths: `IVY_MATH_NUMERIC_QUALIFIER`
- Lazy graph and pointer-gradient paths: `IVY_MATH_GRAPH_QUALIFIER`
- Tensor STL evaluation paths: `IVY_MATH_TENSOR_STL_QUALIFIER`

This split prevents illegal host calls from `__host__ __device__` instantiation contexts while preserving user API shape.

### 9.3 Autodiff graph layer

- Node base: `interface/autodiff/basic_nodes/IvyFunction.h`
- Arithmetic node factories: `interface/autodiff/arithmetic/IvyMathBaseArithmetic.hh/.h`
- User entry point: `func->gradient(x)`

Concrete patterns:

1. Variable target
   - `auto x = Variable<double>(IvyMemoryType::Host, stream, 1.0);`
   - `auto f = Sin(Exp(x));`
   - `auto df_dx = f->gradient(x);`

2. Tensor shared-variable target
   - `auto t = Tensor<IvyVariablePtr_t<double>>(IvyMemoryType::Host, stream, IvyTensorShape({2,2}), x);`
   - `auto ft = Exp(t);`
   - `auto dft_dx = ft->gradient(x);`

3. Function-chain usability (function in graph, variable as gradient target)
   - `auto u = Exp(x);`
   - `auto h = Sin(u);`
   - `auto dh_dx = h->gradient(x);`

## 10. One-Line Commands (Literal Paths, Non-Interactive)

```bash
cd /workspace/USDS-HCC && set -euo pipefail && make distclean
cd /workspace/USDS-HCC && set -euo pipefail && make EXTCXXFLAGS='-w' pch && make EXTCXXFLAGS='-w' utests
cd /workspace/USDS-HCC && set -euo pipefail && USE_CUDA=1 make EXTCXXFLAGS='-w' utests
cd /workspace/USDS-HCC && set -euo pipefail && make lib && USE_CUDA=1 make lib
cd /workspace/USDS-HCC && set -euo pipefail && for b in /workspace/USDS-HCC/executables/*; do timeout 120s "$b"; done
cd /workspace/USDS-HCC && set -euo pipefail && for t in /workspace/USDS-HCC/test_executables/utest_*; do timeout 120s "$t"; done
```
