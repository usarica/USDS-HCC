# USDS-HCC

A modified, header-first C++20 core library that provides STL-like components for heterogeneous execution (CPU + optional CUDA GPU), with explicit memory-domain control, custom allocators, stream/event abstractions, and autodiff-oriented math/data structures.

## Legal Notice (Read First)

- Repository content is provided for inspection purposes only.
- Copying, use, execution, modification, distribution, or creation of derivative works is prohibited unless explicit prior written authorization is granted by Ulaşcan Sarıca.
- Full legal terms: [License](LICENSE.md).

This repository is part of `IvyFramework` and is focused on reusable low-level infrastructure for hybrid computing.

---

## Table of Contents

- [What This Project Is](#what-this-project-is)
- [Key Capabilities](#key-capabilities)
- [Repository Layout](#repository-layout)
- [Requirements](#requirements)
- [Build and Test](#build-and-test)
- [How to Use in Your Code](#how-to-use-in-your-code)
- [Core Concepts](#core-concepts)
- [Autodiff Module](#autodiff-module)
- [Documentation (Doxygen)](#documentation-doxygen)
- [Troubleshooting](#troubleshooting)
- [License and Usage Approval](#license-and-usage-approval)

---

## What This Project Is

`IvyHeterogeneousCore` provides a custom standard-library-like layer under `interface/std_ivy`, along with supporting config, memory, stream, and autodiff components under `interface/`.

Design goals:

- Keep APIs familiar to C++ users (`vector`, `unordered_map`, iterators, allocators, traits).
- Make memory placement explicit (`Host`, `GPU`, `PageLocked`, `Unified`, `UnifiedPrefetched`).
- Support both CPU and CUDA execution paths from shared interfaces.
- Enable higher-level math/autodiff components to run on heterogeneous backends.

---

## Key Capabilities

### 1) Heterogeneous containers and memory-aware STL-like layer

From `interface/std_ivy/`:

- `IvyVector` (`std_ivy/vector`) and aliasing via umbrella headers.
- `IvyUnorderedMap` (`std_ivy/unordered_map`) with custom key/hash-equality evaluators.
- Iterator stack (contiguous, reverse, bucketed builders, traits/primitives).
- Allocator/memory primitives (`IvyAllocator`, `IvyMemoryView`, `IvyPointerTraits`, `IvyUnifiedPtr`).

### 2) Unified dispatch model for kernels

From `interface/config/IvyKernelRun.h`:

- `run_kernel<Kernel_t>(shared_mem, stream).parallel_1D/2D/3D(...)`
- GPU-first execution when CUDA is usable.
- CPU/OpenMP fallback (with specialization support via `kernel_unit_unified`).

### 3) Stream and event abstraction

From `interface/stream/`:

- Common base abstractions: `IvyBaseStream`, `IvyBaseStreamEvent`.
- CUDA implementations: `IvyCudaStream`, `IvyCudaEvent`.
- CPU/no-op fallback implementations: `IvyBlankStream`, `IvyBlankStreamEvent`.

### 4) Autodiff and math-related foundations

From `interface/autodiff/`:

- Base math types and arithmetic/function primitives.
- Node/function/tensor/variable components.
- Specialized functions (e.g. `cerf`).

---

## Repository Layout

Top-level structure:

- `interface/` — public headers and core implementation headers.
- `bin/` — executable entry source files (`*.cc`) built by `make all`.
- `executables/` — generated binaries from `bin/`.
- `test/` — unit/integration-style test sources (`utest_*.cc`).
- `test_executables/` — generated test binaries from `make utests`.
- `makefile` — primary build entry.
- `Doxyfile` — Doxygen configuration (`PROJECT_NAME = IvyHeterogeneousCore`, version `v0.1.0`).
- `html/` — generated documentation output (if present after Doxygen run).

---

## Requirements

### Baseline (Linux)

- `g++` or `clang++` with C++20 support.
- GNU Make.

### Optional CPU parallelism

- OpenMP-capable compiler/runtime (`-fopenmp` is auto-detected in the Makefile).

### Optional CUDA support

- NVIDIA CUDA toolkit with `nvcc`.
- The current Makefile CUDA path targets `-arch=sm_86`.
  - If your GPU architecture differs, update the `CXXFLAGS` CUDA arch in `makefile`.

### Optional docs

- Doxygen 1.9+ for API docs generation.

---

## Build and Test

All commands below assume you are in the repository root.

### 1) Clean

```bash
make clean
```

### 2) Build binaries from `bin/*.cc`

```bash
make all
```

Outputs are placed in `executables/`.

### 3) Build tests from `test/*.cc`

```bash
make utests
```

Outputs are placed in `test_executables/`.

### 4) Build with CUDA enabled

```bash
make USE_CUDA=1 all
make USE_CUDA=1 utests
```

Notes:

- In CUDA mode the Makefile switches to `nvcc` and enables `-D__USE_CUDA__` and `-rdc=true`.
- OpenMP flags are disabled in the CUDA branch of the current Makefile.

### 5) Run tests

Each source in `test/` produces a counterpart binary in `test_executables/`.

Example:

```bash
./test_executables/utest_IvyVector_basic
./test_executables/utest_IvyUnorderedMap_basic
./test_executables/utest_unified_ptr_basic
```

A convenient way to run all test executables:

```bash
for t in ./test_executables/utest_*; do echo "==> $t"; "$t"; done
```

---

## How to Use in Your Code

This library is consumed as headers from `interface/`.

### Include path

Add:

```bash
-I/path/to/IvyHeterogeneousCore/interface
```

### Example: use `std_ivy::vector`

```cpp
#include "std_ivy/IvyVector.h"
#include "stream/IvyStream.h"

int main(){
  IvyGPUStream* stream = IvyStreamUtils::make_global_gpu_stream();

  std_ivy::vector<double> vec(5, IvyMemoryHelpers::IvyMemoryType::Host, stream, 1.0);
  vec.push_back(IvyMemoryHelpers::IvyMemoryType::Host, stream, 2.0);

  IvyStreamUtils::destroy_stream(stream);
  return 0;
}
```

### Example: run a kernel via `run_kernel`

```cpp
#include "config/IvyKernelRun.h"

struct saxpy_kernel{
  static __HOST_DEVICE__ void kernel(IvyTypes::size_t i, IvyTypes::size_t n, float* y, float const* x, float a){
    if (kernel_check_dims<saxpy_kernel>::check_dims(i, n)) y[i] += a * x[i];
  }
};

// run_kernel<saxpy_kernel>(0, stream).parallel_1D(n, y, x, a);
```

---

## Core Concepts

### Memory domains

Defined in `IvyMemoryHelpers::IvyMemoryType`:

- `Host`
- `GPU`
- `PageLocked`
- `Unified`
- `UnifiedPrefetched`

The helpers in `IvyMemoryHelpers.h` provide allocation/construct/destruct/transfer APIs across these domains.

### Unified pointer model

`IvyUnifiedPtr` supports pointer semantics with memory-type metadata, capacity/size tracking, transfer operations, and shared/unique pointer-type behavior via `IvyPointerType`.

### Stream-aware operations

Most memory/container operations accept `IvyGPUStream*` (or stream refs in helper internals), allowing backend-aware execution ordering.

### CPU/GPU execution strategy

The dispatch layer in `IvyKernelRun.h` allows a single kernel interface to run:

- On GPU when available.
- On CPU/OpenMP fallback otherwise.

---

## Autodiff Module

`interface/autodiff/` contains math and computation-graph components, including:

- Base and arithmetic primitives.
- Node/function/type tags and client-management utilities.
- Basic tensor/function/variable nodes.
- Specialized mathematical functions.

For concrete examples, inspect `test/utest_autodiff_basic_blocks.cc`.

---

## Documentation (Doxygen)

This repository includes a full `Doxyfile`.

Generate docs:

```bash
doxygen Doxyfile
```

Typical output is generated under `html/` (depending on your Doxygen settings and current output configuration).

---

## Troubleshooting

### Build fails in CUDA mode

- Check CUDA toolkit and `nvcc` availability.
- Verify GPU arch compatibility with `-arch=sm_86` in `makefile`.
- Ensure your toolchain supports relocatable device code (`-rdc=true`).

### OpenMP not being used

- OpenMP is auto-detected for non-CUDA builds; ensure compiler/runtime support.
- In CUDA mode, current Makefile disables OpenMP flags.

### Missing symbols or include errors

- Confirm `-I.../interface` is on your include path.
- Compile sources as C++20 (`-std=c++20`).

---

## License and Usage Approval

This project is proprietary and restricted.

Use of this software is limited to **Ulaşcan Sarıca** unless explicit prior written authorization is granted by the developer.

For clarity:

- Viewing source is allowed for inspection.
- Copying any part of the code is prohibited without prior written authorization.
- Using or executing the software is prohibited without prior written authorization.
- Modifying, redistributing, or creating derivative works is prohibited without prior written authorization.

See the full legal terms in [License](LICENSE.md) (authoritative legal file: `LICENSE`).

