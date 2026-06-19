# USDS-HCC

A header-only C++20 autodiff library providing STL-like components for heterogeneous execution (CPU + optional CUDA GPU), with explicit memory-domain control, custom allocators, stream/event abstractions, and a full tensor-aware automatic differentiation engine.

## Legal Notice (Read First)

- Repository content is provided for inspection purposes only.
- Copying, use, execution, modification, distribution, or creation of derivative works is prohibited unless explicit prior written authorization is granted by Ulaşcan Sarıca.
- Full legal terms: [License](LICENSE.md).

This repository is part of the software packages provided by `US Data Science` and is focused on reusable low-level infrastructure for hybrid computing.

---

## Table of Contents

- [Project Overview](#1-project-overview)
- [Architecture Overview](#2-architecture-overview)
- [Requirements](#3-requirements)
- [Building](#4-building)
- [Quick-Start Example](#5-quick-start-example)
- [Autodiff API Reference](#6-autodiff-api-reference)
- [Tensor Autodiff](#7-tensor-autodiff)
- [Memory Domains](#8-memory-domains)
- [Stream Abstraction](#9-stream-abstraction)
- [Using as a Shared Library](#10-using-as-a-shared-library)
- [Using as a Header-Only Library](#11-using-as-a-header-only-library)
- [Doxygen](#12-doxygen)
- [Troubleshooting](#13-troubleshooting)
- [License and Usage Approval](#14-license-and-usage-approval)

---

## 1. Project Overview

`USDS-HCC` is a lightweight, header-only C++20 library for automatic differentiation (autodiff) on both scalars and tensors, targeting both CPU and NVIDIA GPU hardware through a unified interface. It is designed for scientists and engineers who need:

- **Automatic differentiation** of arbitrarily nested mathematical expressions, including full chain-rule gradient propagation through scalar, complex, and tensor quantities.
- **Heterogeneous execution**: the same code compiles and runs on CPU (with optional OpenMP) and CUDA GPU without any source changes — just pass `USE_CUDA=1` to `make`.
- **Explicit memory placement**: every container and node explicitly carries its memory type (`Host`, `GPU`, `Unified`, etc.), giving precise control over data residency.
- **STL-compatible abstractions**: `IvyVector`, `IvyUnorderedMap`, iterators, allocators, and smart pointers that mirror the standard library API while adding memory-domain awareness.
- **Zero runtime overhead for headers**: the library is header-only. No separate build step is required for the autodiff engine; all template instantiation happens at your compile time.

The library was built to power the `US Data Science` statistical analysis ecosystem — a larger collection of C++ components for high-energy physics (HEP) data analysis — but is entirely standalone and suitable for any C++20 project that needs portable autodiff on heterogeneous hardware.

---

## 2. Architecture Overview

### 2.1 STL-generalization data flow

`interface/std_ivy/*` mirrors core STL behavior (`IvyVector`, `IvyUnorderedMap`, iterators, allocators, smart pointers) while carrying explicit memory-domain metadata. The data flow is:

1. user container operation (`push_back`, iterator traversal, map insert/find),
2. allocator dispatch through `IvyAllocator`/`IvyMemoryHelpers`,
3. memory-domain-aware transfer/build/free for Host/GPU/Unified,
4. optional stream-aware execution via `IvyGPUStream` wrappers.

### 2.2 CPU/GPU memory dispatch flow

Dispatch is encoded by memory type and annotation policy:

- numeric kernels remain host/device-eligible (`IVY_MATH_NUMERIC_QUALIFIER`),
- lazy graph construction and pointer-gradient paths are host-only (`IVY_MATH_GRAPH_QUALIFIER`),
- tensor STL evaluation paths are host-only (`IVY_MATH_TENSOR_STL_QUALIFIER`).

This policy is centralized in `interface/config/IvyAnnotationDispatchPolicy.h` and applied in arithmetic autodiff call sites so device compilation no longer instantiates host-only graph code.

### 2.3 Autodiff graph flow

For pointer inputs (variable/tensor/function pointers), math calls (e.g. `Exp(x)`, `Sin(x)`, `Pow(x,y)`) create lazy `IvyRegularFunction_*D` nodes. At evaluation time, `value()` triggers `eval()` if modified, and gradient calls traverse dependencies:

- `func->gradient(variable_ptr)` for scalar-variable differentiation,
- `func->gradient(variable_ptr)` for tensor-shared-variable differentiation,
- `func->gradient(variable_ptr)` through intermediate function nodes (`u=Exp(x)`, `h=Sin(u)`).

The external API remains unchanged: `func->gradient(x)` is the stable entry point.


The library is organised in four conceptual layers stacked on top of each other:

```
┌─────────────────────────────────────────────────────────────────┐
│  autodiff/  — Differentiation engine                            │
│    basic_nodes/   IvyConstant, IvyVariable, IvyComplexVariable  │
│                   IvyTensor, IvyFunction                        │
│    arithmetic/    IvyMathBaseArithmetic, IvyMathFunctionPrimitives│
│                   IvyMathConstOps, IvyMathTypes                 │
│    IvyBaseMathTypes.h, IvyMathTypes.h — type-system glue        │
├─────────────────────────────────────────────────────────────────┤
│  stream/   — Execution stream abstractions                      │
│    IvyStream.h         — include both CPU and CUDA streams      │
│    IvyCudaStream.h     — CUDA stream wrapper                    │
│    IvyCudaEvent.h      — CUDA event wrapper                     │
│    IvyBlankStream.h    — no-op CPU stream                       │
├─────────────────────────────────────────────────────────────────┤
│  config/   — Compiler and runtime configuration                 │
│    IvyCudaException.h  — CUDA error checking macros             │
│    IvyKernelRun.h      — unified GPU/CPU kernel launcher        │
│    IvyKernelRunHelpers.h — grid/block helpers                   │
├─────────────────────────────────────────────────────────────────┤
│  std_ivy/  — STL wrappers (memory-domain–aware)                 │
│    IvyVector.h         — dynamic array                          │
│    IvyUnorderedMap.h   — hash map                               │
│    IvyMemory.h         — allocators, unified pointer            │
│    IvyTypeTraits.h     — compile-time type utilities            │
│    IvyLimits.h, IvyCassert.h, IvyCstdio.h, IvyCstring.h        │
└─────────────────────────────────────────────────────────────────┘
```

Each layer depends only on layers below it. The autodiff engine (`autodiff/`) depends on `stream/`, `config/`, and `std_ivy/`, but the standard library wrappers (`std_ivy/`) are fully independent.

Key design decisions:

- **All computation nodes live on the host heap.** Virtual dispatch is safe because `IvyFunction` derivatives are never device-resident.
- **Lazy evaluation**: `IvyFunction::value()` calls `eval()` (which checks the `modified` flag), so recomputation happens only when inputs change.
- **Eager gradient for tensors**: tensor-domain gradient chains bypass the lazy multiply graph (which would recurse infinitely) and instead compute element-wise products eagerly, returning an `IvyTensorEagerFunction` concrete node.
- **Thread-safe shared pointers**: `IvyThreadSafePtr_t` (alias for `IvyUnifiedPtr` in shared mode) owns autodiff nodes, enabling safe multi-owner computation graphs.

---

## 3. Requirements

### Compiler

| Requirement | Minimum version | Notes |
|-------------|----------------|-------|
| `g++`       | 11.0           | C++20 support required (`-std=c++20`) |
| `clang++`   | 13.0           | Tested with libc++ and libstdc++ |
| `nvcc`      | CUDA 12.0      | Required only for GPU support |

### Build tools

- GNU Make 4.0+.

### Optional runtime dependencies

| Dependency | Purpose |
|------------|---------|
| OpenMP     | Parallel CPU loops in kernel fallbacks; auto-detected by `makefile` |
| CUDA 12+   | GPU execution; detected via `USE_CUDA=1` flag |
| Doxygen 1.9+ | HTML API documentation generation |

### Tested platforms

- Ubuntu 22.04 / 24.04, g++ 11 / 13, CUDA 12.4, x86_64.

---

## 4. Building

All commands below must be run from the repository root directory
(the directory that contains `makefile`).

### 4.1 CPU-only: build unit tests

```bash
make utests
```

Compiles every `test/utest_*.cc` source and places the resulting binaries under
`test_executables/`. The current test suite has 11 executables:

```
utest_IvyBucketedIteratorBuilder_basic
utest_IvyContiguousIterator_basic
utest_IvyUnorderedMap_basic
utest_IvyVector_basic
utest_autodiff_basic_blocks
utest_basic_alloc_copy_dealloc
utest_hash
utest_sum_parallel
utest_tensor_function
utest_unified_ptr_basic
utest_unified_ptr_transfer
```

Run all tests at once:

```bash
for t in ./test_executables/utest_*; do echo "==> $t"; "$t"; done
```

### 4.2 CUDA build: unit tests

```bash
make utests USE_CUDA=1
```

This switches the compiler to `nvcc`, adds `-D__USE_CUDA__` and
`-D__LONG_DOUBLE_FORBIDDEN__`, and enables device-side relocatable code
(`-rdc=true`). GPU architecture is auto-detected via `nvidia-smi`; if
`nvidia-smi` is not available it falls back to `sm_86`.

### 4.3 Shared library (CPU)

```bash
make lib
```

Produces `lib/libIvyHCC.so` by compiling `src/IvyHCC.cc`
(a minimal translation unit that includes the umbrella header
`interface/IvyHCC.h`) as a position-independent shared object.

### 4.4 Shared library (CUDA)

```bash
make lib USE_CUDA=1
```

Produces `lib/libIvyHCC.so` by compiling with `nvcc` in device-link
(`-dc`) mode and then linking with `-dlink -shared`.

### 4.5 Clean build artifacts

```bash
make clean
```

Removes `executables/`, `test_executables/`, and intermediate object files.

### 4.6 One-line acceptance harness

```bash
make diagnostics-baseline
```

Runs the non-interactive diagnostics matrix and writes acceptance logs under
`strict_logs/`.

### 4.7 Shared-library consumption cookbook

Build shared library:

```bash
make lib
```

or CUDA-enabled:

```bash
USE_CUDA=1 make lib
```

Link an external application:

```bash
g++ -std=c++20 -O2 -I./interface \
  -L./lib -Wl,-rpath,'$ORIGIN/lib' \
  -lIvyHCC your_app.cc -o your_app
```

### 4.8 One-line ergonomics demo

```bash
make all && ./executables/gradient_ergonomics_demo
```

The demo validates these user-facing patterns:
- `func->gradient(x)` for scalar variables,
- `tensor_func->gradient(x)` for tensor/shared-variable flows,
- chained-function variable gradients (`h = Sin(Exp(x))`, then `h->gradient(x)`).

---

## 5. Quick-Start Example

The following self-contained program exercises the scalar autodiff API. It
compiles against the library's headers with no pre-built libraries needed.

```cpp
/**
 * quickstart.cc — minimal IvyHCC autodiff example.
 *
 * Compile (CPU):
 *   g++ -std=c++20 -O2 -I/workspace/USDS-HCC/interface -o quickstart quickstart.cc
 *
 * Compile (CUDA):
 *   nvcc -std=c++20 -O2 -x cu -rdc=true -D__USE_CUDA__ \
 *        -D__LONG_DOUBLE_FORBIDDEN__ \
 *        -I/workspace/USDS-HCC/interface -o quickstart quickstart.cc
 */
#include "autodiff/arithmetic/IvyMathBaseArithmetic.h"
#include "autodiff/basic_nodes/IvyTensor.h"
#include <cstdio>

using namespace IvyMath;

int main(){
  // ── Scalar autodiff ──────────────────────────────────────────────────────
  // Create a differentiable variable x = 2.0
  auto x = Variable<double>(IvyMemoryType::Host, nullptr, 2.0);

  // f(x) = exp(x)
  auto f = Exp(x);

  // Value: exp(2) ≈ 7.389
  double val = f->value().value();
  printf("Exp(2) = %.6f\n", val);

  // Gradient: d/dx exp(x) at x=2 => exp(2)
  auto g = f->gradient(x);
  double grad_val = g->value().value();
  printf("d/dx Exp(2) = %.6f\n", grad_val);

  // ── Chained expression: sin(exp(x)) ──────────────────────────────────────
  auto h = Sin(f);   // sin(exp(x))
  auto dh = h->gradient(x);  // cos(exp(x)) * exp(x)

  printf("sin(exp(2))        = %.6f\n", h->value().value());
  printf("d/dx sin(exp(2))   = %.6f\n", dh->value().value());

  // ── Tensor autodiff ──────────────────────────────────────────────────────
  // All 6 elements of a 2×3 tensor share the same variable x=3.0
  auto y  = Variable<double>(IvyMemoryType::Host, nullptr, 3.0);
  auto t  = Tensor<IvyVariablePtr_t<double>>(
                IvyMemoryType::Host, nullptr,
                IvyTensorShape({2, 3}), y);

  auto ft = Exp(t);                              // element-wise exp
  auto gt = ft->gradient(y);                    // d Exp(t)[i]/dy == exp(3)
  printf("d/dy Exp(t)[0] = %.6f  (expect %.6f)\n",
         gt->value()[0]->value(), std::exp(3.0));

  return 0;
}
```

Expected output:
```
Exp(2) = 7.389056
d/dx Exp(2) = 7.389056
sin(exp(2))        = 0.893855
d/dx sin(exp(2))   = -3.268688
d/dy Exp(t)[0] = 20.085537  (expect 20.085537)
```

---

## 6. Autodiff API Reference

All autodiff types live in the `IvyMath` namespace (header:
`autodiff/arithmetic/IvyMathBaseArithmetic.h` or the umbrella
`IvyHCC.h`).

### 6.1 `IvyConstant<T>`

An immutable scalar node. Its value is fixed at construction; it has no
gradient (the gradient of a constant is always zero).

```cpp
// Construct from a value
auto c = Constant<double>(IvyMemoryType::Host, nullptr, 3.14);
double v = c->value().value();  // 3.14

// Constants never satisfy depends_on — no gradient computation needed.
```

Template parameter `T` is the numeric type (e.g., `double`, `float`).

### 6.2 `IvyVariable<T>`

A differentiable leaf node. `gradient(var)` returns 1 if `var` is the same
object as this variable, 0 otherwise.

```cpp
auto x = Variable<double>(IvyMemoryType::Host, nullptr, 2.5);

double val = x->value().value();   // 2.5

// Modify and re-evaluate
x->set_value(4.0);
// set_modified() is called automatically; the next call to value()
// will reflect the new value.
```

`IvyVariablePtr_t<T>` is a convenience alias for
`IvyThreadSafePtr_t<IvyVariable<T>>`.

### 6.3 `IvyComplexVariable<T>`

A complex differentiable leaf. Real and imaginary parts are stored as
`IvyVariable<T>`.

```cpp
auto z = ComplexVariable<double>(IvyMemoryType::Host, nullptr, 1.0, 2.0);

double re = z->value().Re();   // 1.0
double im = z->value().Im();   // 2.0

// Conjugate in place
// IvyNodeSelfRelations::conjugate(*z);
```

### 6.4 `IvyTensor<T>`

An N-dimensional array where the element type `T` is typically
`IvyVariablePtr_t<double>` for an autodiff tensor.

```cpp
IvyTensorShape shape({2, 3});  // 2 rows, 3 columns
auto x = Variable<double>(IvyMemoryType::Host, nullptr, 1.0);

// Create a 2×3 tensor where all elements point to x
auto t = Tensor<IvyVariablePtr_t<double>>(
    shape.get_memory_type(), shape.gpu_stream(), shape, x);

IvyTensorDim_t n  = t->num_elements();        // 6
auto& elem        = t->at({0, 1});             // element at row 0, col 1
IvyTensorRank_t r = t->rank();                // 2
```

**Supported element types:**
- `IvyVariablePtr_t<double>` — differentiable real tensor
- `IvyVariablePtr_t<float>` — differentiable single-precision tensor
- Arithmetic types (`double`, `float`, etc.) — non-differentiable tensor

`IvyTensorPtr_t<T>` is the corresponding `IvyThreadSafePtr_t<IvyTensor<T>>`.

### 6.5 `IvyFunction<P, Domain, GradientDomain>`

The abstract base class for all computation-graph nodes. Concrete derived
classes are created by the arithmetic operator functions, not constructed
directly.

Key members:

| Member | Description |
|--------|-------------|
| `value() const` | Evaluate and return the output (lazy; cached until input changes). |
| `gradient(var)` | Compute `∂this/∂var` and return a new `IvyFunction` node. |
| `depends_on(node)` | Returns `true` if this node transitively depends on `node`. |

`grad_t` = `IvyFunction<P, Domain>` (same domains). `gradient()` returns
`IvyThreadSafePtr_t<grad_t>`.

### 6.6 Arithmetic operations

All operations accept both pointer and non-pointer arguments via overloads.
The pointer overloads register the result in the client-manager graph.

| Function / Operator | Formula | Real | Complex | Tensor |
|---------------------|---------|------|---------|--------|
| `operator+`         | x + y   | ✓   | ✓      | ✓      |
| `operator-` (unary) | −x      | ✓   | ✓      | ✓      |
| `operator-`         | x − y   | ✓   | ✓      | ✓      |
| `operator*`         | x · y   | ✓   | ✓      | ✓      |
| `operator/`         | x / y   | ✓   | ✓      | ✗      |
| `Pow(x, y)`         | x^y     | ✓   | ✓      | ✓      |
| `Exp(x)`            | e^x     | ✓   | ✓      | ✓      |
| `Log(x)`            | ln x    | ✓   | ✓      | ✓      |
| `Sin(x)`            | sin x   | ✓   | ✓      | ✓      |
| `Cos(x)`            | cos x   | ✓   | ✓      | ✓      |
| `Tan(x)`            | tan x   | ✓   | ✓      | ✓      |
| `Sqrt(x)`           | √x      | ✓   | ✓      | ✓      |
| `Abs(x)`            | |x|     | ✓   | ✓      | ✓      |
| `Erf(x)`            | erf(x)  | ✓   | ✓      | ✓      |
| `Faddeeva(x)`       | w(x)    | ✓   | ✓      | ✓      |
| `Negate(x)`         | −x      | ✓   | ✓      | ✓      |
| `MultInverse(x)`    | 1/x     | ✓   | ✓      | ✓      |

The Faddeeva function `w(x) = exp(-x²)·erfc(-ix)` accepts both real and complex
arguments.  For a real input `x`, it returns a complex-valued result.  This is
accessed via `Faddeeva(x)` where `x` is an `IvyVariable<T>` or
`IvyTensor<IvyVariable<T>>`.  The gradient `dw/dz = (2i/√π) − 2z·w(z)` is
complex-valued even for real input.

Comparison and logic operators (`==`, `!=`, `<`, `<=`, `>`, `>=`) are
available for real types but return `bool` — they do **not** participate in
the gradient graph and have no `gradient()` method.

### 6.7 Gradient of a composition

Chain-rule propagation is automatic. For example:

```cpp
auto x = Variable<double>(IvyMemoryType::Host, nullptr, 1.0);
auto f = Sin(Exp(x));           // sin(e^x)
auto g = f->gradient(x);        // cos(e^x) · e^x
double val = g->value().value();
```

The gradient is itself an `IvyFunction` node, so you can call `->value()`
on it without materialising the full second derivative.

---

## 7. Tensor Autodiff

Tensor functions apply operations element-wise over `IvyTensor<dtype_t>`.
The gradient of a tensor function with respect to a scalar variable is
computed eagerly to avoid infinite template recursion that would arise from
the lazy multiply graph.

### 7.1 Creating a differentiable tensor

```cpp
#include "autodiff/arithmetic/IvyMathBaseArithmetic.h"
#include "autodiff/basic_nodes/IvyTensor.h"

using namespace IvyMath;

// Scalar variable, x = 3.0
auto x = Variable<double>(IvyMemoryType::Host, nullptr, 3.0);

// 2×3 tensor where every element IS x (shared pointer)
IvyTensorShape shape({2, 3});
auto t = Tensor<IvyVariablePtr_t<double>>(
    IvyMemoryType::Host, nullptr, shape, x);
```

### 7.2 Applying a tensor function

```cpp
auto ft = Exp(t);   // IvyThreadSafePtr_t<IvyFunction<tensor, tensor_domain_tag>>

// Evaluate (lazy)
auto const& out = ft->value();    // IvyTensor<IvyVariablePtr_t<double>>
for (IvyTensorDim_t i = 0; i < out.num_elements(); ++i)
    printf("Exp(t)[%llu] = %.6f\n", (unsigned long long)i, out[i]->value());
// Prints exp(3) ≈ 20.085537 for all 6 elements.
```

### 7.3 Computing the gradient

```cpp
// ∂Exp(t)[i]/∂x = exp(x) · ∂t[i]/∂x = exp(3) · 1 = exp(3)
auto gt = ft->gradient(x);   // IvyThreadSafePtr_t<IvyFunction<tensor, tensor_domain_tag>>

auto const& gval = gt->value();
for (IvyTensorDim_t i = 0; i < gval.num_elements(); ++i)
    printf("d/dx Exp(t)[%llu] = %.6f\n",
           (unsigned long long)i, gval[i]->value());
// Prints exp(3) ≈ 20.085537 for all 6 elements.
```

### 7.4 How element-wise gradients work

For `f = F(t)` where `t` is a tensor and `F` is one of the supported
element-wise functions:

1. `F::gradient(dep)` computes the derivative tensor `f'` where `f'[i] = F'(t[i])`.
2. `function_gradient<T>::get(*t, var)` computes the Jacobian vector `∂t[i]/∂var` for each `i`.
3. These two tensors are multiplied element-wise: `result[i] = f'[i] · ∂t[i]/∂var`.
4. The result is wrapped in `IvyTensorEagerFunction<T>`, a concrete subclass of
   `IvyFunction<T, tensor_domain_tag, tensor_domain_tag>` that holds the
   pre-computed tensor.

This avoids instantiating `IvyMultiply<T,T,tensor,tensor>`, whose `gradient()`
(required by `IvyFunction<T,tensor>`) would recurse without bound.

### 7.5 Supported tensor element-wise functions with gradients

| Function | `f(x)`  | `f'(x)`  |
|----------|---------|----------|
| `Exp(t)` | e^x     | e^x      |
| `Log(t)` | ln x    | 1/x      |
| `Sin(t)` | sin x   | cos x    |
| `Cos(t)` | cos x   | −sin x   |
| `Negate(t)` or `-t` | −x | −1 |

---

## 8. Memory Domains

Every container, pointer, and function node in the library carries an
`IvyMemoryType` tag:

| `IvyMemoryType` value | Allocation | Accessible from | Notes |
|-----------------------|------------|-----------------|-------|
| `Host`                | `malloc`   | CPU only        | Default for CPU-only builds |
| `GPU`                 | `cudaMalloc` | GPU only     | Requires CUDA |
| `PageLocked`          | `cudaMallocHost` | CPU + DMA-to-GPU | Fast pinned memory |
| `Unified`             | `cudaMallocManaged` | CPU + GPU | CUDA Unified Memory, migrated by driver |
| `UnifiedPrefetched`   | `cudaMallocManaged` | CPU + GPU | Same as Unified but prefetches to GPU |

Helper: `IvyMemoryHelpers::get_execution_default_memory()` returns `Host` for
CPU builds and `GPU` for CUDA builds.

Use `IvyMemoryHelpers::transfer_memory(dst, src, n, dst_type, src_type, stream)`
to copy between domains.

---

## 9. Stream Abstraction

Streams sequence asynchronous operations. The library provides:

### `IvyCPUStream` (exposed as `IvyBlankStream`)

- No-op stream for CPU execution.
- All operations complete synchronously before `synchronize()` returns.
- Available without CUDA; selected automatically in CPU-only builds.
- Backed optionally by OpenMP when `-fopenmp` is active.

### `IvyCudaStream`

- Wraps a `cudaStream_t`.
- Creation: `IvyStreamUtils::make_global_gpu_stream()` (singleton pattern).
- Destruction: `IvyStreamUtils::destroy_stream(stream)`.
- Synchronisation: `IvyStreamUtils::synchronize_stream(stream)`.

Most container and memory APIs accept a `IvyGPUStream*` (aliased from
`IvyStreamUtils::IvyGPUStream`). Passing `nullptr` uses the default (null)
CUDA stream.

Example (CPU build — stream is a no-op pointer):

```cpp
#include "stream/IvyStream.h"
IvyGPUStream* stream = nullptr;   // CPU build: always nullptr
std_ivy::vector<double> v(10, IvyMemoryType::Host, stream, 0.0);
```

---

## 10. Using as a Shared Library

After `make lib` (or `make lib USE_CUDA=1`), the shared object is at
`lib/libIvyHCC.so`.

### Linking against the shared library (CPU build)

```bash
IVYROOT=/path/to/IvyHCC

g++ -std=c++20 -O2 \
    -I/workspace/USDS-HCC/interface \
    -L/workspace/USDS-HCC/lib \
    -lIvyHCC \
    -Wl,-rpath,/workspace/USDS-HCC/lib \
    -o myapp myapp.cc
```

Or set `LD_LIBRARY_PATH` at runtime instead of `-Wl,-rpath`:

```bash
export LD_LIBRARY_PATH=/workspace/USDS-HCC/lib:${LD_LIBRARY_PATH}
./myapp
```

### Linking against the shared library (CUDA build)

```bash
nvcc -std=c++20 -O2 -x cu -rdc=true \
     -D__USE_CUDA__ -D__LONG_DOUBLE_FORBIDDEN__ \
     -I/workspace/USDS-HCC/interface \
     -L/workspace/USDS-HCC/lib \
     -lIvyHCC \
     -Xlinker -rpath,/workspace/USDS-HCC/lib \
     -o myapp myapp.cu
```

---

## 11. Using as a Header-Only Library

Since the library is fully header-only, no shared library is required. Add
the include path and compile:

```bash
g++ -std=c++20 -O2 \
    -I/path/to/IvyHCC/interface \
    -o myapp myapp.cc
```

For CUDA:

```bash
nvcc -std=c++20 -O2 -x cu -rdc=true \
     -D__USE_CUDA__ -D__LONG_DOUBLE_FORBIDDEN__ \
     -I/path/to/IvyHCC/interface \
     -o myapp myapp.cu
```

The umbrella header `IvyHCC.h` pulls in everything:

```cpp
#include "IvyHCC.h"
```

Or include only what you need:

```cpp
#include "autodiff/arithmetic/IvyMathBaseArithmetic.h"   // all autodiff ops
#include "autodiff/basic_nodes/IvyTensor.h"              // tensor support
```

---

## 12. Doxygen

The repository ships with a fully configured `Doxyfile` at the root.

Generate the HTML reference documentation:

```bash
cd /path/to/IvyHCC
doxygen Doxyfile
```

Output is written to `html/` by default. Open `html/index.html` in a browser.

To customise the output directory, theme, or verbosity, edit `Doxyfile`
(the settings block at the top controls `OUTPUT_DIRECTORY`, `PROJECT_NAME`,
etc.).

---

## 13. Troubleshooting

### `__LONG_DOUBLE_FORBIDDEN__` not defined (CUDA build fails)

CUDA does not support `long double`. The Makefile adds
`-D__LONG_DOUBLE_FORBIDDEN__` automatically when `USE_CUDA=1` is set.
If you compile manually with `nvcc`, you must add this flag yourself:

```bash
nvcc ... -D__LONG_DOUBLE_FORBIDDEN__ ...
```

Failure to do so produces errors like:
```
error: 'long double' is not supported by nvcc
```

### `else if` braces warning or error in CUDA mode

CUDA's `nvcc` is stricter than `g++` about certain brace-elision patterns.
If you see:
```
warning: suggest explicit braces to avoid ambiguous 'else'
```
wrap the inner `if` / `else if` body in `{}`. This is a style issue, not
a library bug.

### `size_t` ambiguity when including `IvyUnorderedMap.h`

The internal `IvyUnorderedMapImpl.h` uses the bare name `size_t` in the
`std_ivy` namespace. If the umbrella header `IvyHCC.h` is
included after any header that transitively includes `<cstring>` (which
brings `::size_t` from `<stddef.h>` into scope), and the `IvyTypes::size_t`
type is also reachable, the compiler reports:

```
error: reference to 'size_t' is ambiguous
note: candidates are: 'typedef long unsigned int size_t'
note:                 'typedef long long unsigned int IvyTypes::size_t'
```

**Solution**: include `IvyVector.h` and `IvyUnorderedMap.h` **before** any
header that includes `<cstring>`. The provided `IvyHCC.h`
umbrella header already enforces this order. If you build your own include
chain, put the unordered-map headers first.

### `<cassert>` / `size_t` conflict

On some systems, `<cassert>` transitively includes `<stddef.h>`, which
defines `::size_t`. The library provides `std_ivy/IvyCassert.h` as a safe
wrapper. Use it instead of `<cassert>` directly inside library code:

```cpp
#include "std_ivy/IvyCassert.h"   // not <cassert>
```

### Missing `nvcc` / CUDA toolkit

Ensure `nvcc` is on `PATH` before running `make utests USE_CUDA=1`:

```bash
export PATH=/usr/local/cuda/bin:$PATH
which nvcc   # should print the nvcc path
```

If GPU architecture detection via `nvidia-smi` fails (e.g., no physical GPU
in CI), the Makefile falls back to `sm_86`. Override explicitly:

```bash
make utests USE_CUDA=1 GPU_ARCH_RAW=80   # Ampere A100
make utests USE_CUDA=1 GPU_ARCH_RAW=75   # Turing T4
```

### Linker errors: `undefined reference` with `-lIvyHCC`

The shared library (`make lib`) is built from a near-empty translation unit
(`src/IvyHCC.cc`) because the library is header-only — most
definitions are inline templates. If you see missing symbols, ensure you
also compile your own source with the include path:

```bash
g++ -std=c++20 -I/workspace/USDS-HCC/interface -L/workspace/USDS-HCC/lib \
    -lIvyHCC -o /workspace/USDS-HCC/executables/shared_library_smoke \
    /workspace/USDS-HCC/bin/shared_library_smoke.cc
```

and that `LD_LIBRARY_PATH` (or `-Wl,-rpath`) points to `/workspace/USDS-HCC/lib`.

---

## 14. License and Usage Approval

This project is proprietary and restricted.

Use of this software is limited to **Ulaşcan Sarıca** unless explicit prior
written authorization is granted by the developer.

For clarity:

- Viewing source is allowed for inspection.
- Copying any part of the code is prohibited without prior written authorization.
- Using or executing the software is prohibited without prior written authorization.
- Modifying, redistributing, or creating derivative works is prohibited without prior written authorization.

See the full legal terms in [License](LICENSE.md) (authoritative legal file: `LICENSE`).



## 15. Verified Non-Interactive Command Cookbook

All commands below were executed from a clean shell using `set -euo pipefail` and literal repository paths.

```bash
cd /workspace/USDS-HCC && set -euo pipefail && make distclean
cd /workspace/USDS-HCC && set -euo pipefail && make EXTCXXFLAGS='-w' pch && make EXTCXXFLAGS='-w' utests
cd /workspace/USDS-HCC && set -euo pipefail && make lib
cd /workspace/USDS-HCC && set -euo pipefail && USE_CUDA=1 make EXTCXXFLAGS='-w' utests
cd /workspace/USDS-HCC && set -euo pipefail && USE_CUDA=1 make lib
cd /workspace/USDS-HCC && set -euo pipefail && for b in /workspace/USDS-HCC/executables/*; do timeout 120s "$b"; done
cd /workspace/USDS-HCC && set -euo pipefail && for t in /workspace/USDS-HCC/test_executables/utest_*; do timeout 120s "$t"; done
```

### Shared-library include/link workflow (literal paths)

```bash
cd /workspace/USDS-HCC && set -euo pipefail && g++ -std=c++20 -O2 -I/workspace/USDS-HCC/interface -L/workspace/USDS-HCC/lib -lIvyHCC -Wl,-rpath,/workspace/USDS-HCC/lib -o /workspace/USDS-HCC/executables/shared_library_smoke /workspace/USDS-HCC/bin/shared_library_smoke.cc
```
