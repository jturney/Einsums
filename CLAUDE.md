# OpenWolf

@.wolf/OPENWOLF.md

This project uses OpenWolf for context management. Read and follow .wolf/OPENWOLF.md every session. Check .wolf/cerebrum.md before generating code. Check .wolf/anatomy.md before reading files.


# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Einsums is a C++20 tensor algebra library that provides compile-time contraction pattern analysis to select optimal operations (BLAS calls, generic algorithms, or MLIR JIT). It uses a modular CMake build system with optional GPU (CUDA/HIP) and MLIR JIT backends.

## Build Commands

```bash
# Configure (from build directory, out-of-source required)
cmake -S . -B build -GNinja -DCMAKE_BUILD_TYPE=RelWithDebInfo

# With MLIR backend (requires MLIR >= 19 from conda-forge)
cmake -S . -B build -GNinja -DEINSUMS_WITH_MLIR=ON -DMLIR_DIR=$CONDA_PREFIX/lib/cmake/mlir

# With MPI distributed computing (requires Open MPI or MPICH)
cmake -S . -B build -GNinja -DEINSUMS_WITH_MPI=ON

# Build everything
cmake --build build

# Build a specific test target
cmake --build build --target Einsum1_test

# Run all tests
ctest --test-dir build

# Run a single test (Catch2-based, uses ctest discovery)
ctest --test-dir build -R "Einsum1"

# Run a specific Catch2 test case directly
./build/libs/Einsums/TensorAlgebra/tests/unit/Einsum1_test "[test-case-name]"

# Build a specific module's tests
cmake --build build --target Tests.Unit.Modules.TensorAlgebra
```

### Conda Development Environment

```bash
conda env create -f devtools/conda-envs/conda.yml
conda activate einsums-dev
```

When looking for header files outside of Einsums look inside the `einsums-dev` conda environment.

## Code Formatting

Pre-commit hooks enforce formatting. Install with `pre-commit install`.

- **C++**: clang-format v16 with LLVM-based style, 4-space indent, 140-column limit (`.clang-format`)
- **CMake**: cmake-format v0.6.10 (`.cmake-format.py`)
- **License headers**: Auto-inserted in `.cpp`, `.hpp`, `.cmake`, `.py` files

## Architecture

### Module System

All modules live under `libs/Einsums/<ModuleName>/` with the standard layout:
```
libs/Einsums/<Name>/
  CMakeLists.txt              # calls einsums_add_module()
  include/Einsums/<Name>/     # public headers
  src/                        # implementation files
  tests/unit/                 # unit tests
  tests/performance/          # benchmarks
```

Module registration uses `einsums_add_module(Einsums <Name> SOURCES ... HEADERS ... MODULE_DEPENDENCIES ...)` in CMake. The module list lives in `libs/Einsums/CMakeLists.txt`. New modules are created via `libs/create_module_skeleton.py`.

### Key Module Dependency Chain

`TensorAlgebra` -> `LinearAlgebra` -> `BLAS` -> `BLASBase` -> `BLASVendor`

`TensorAlgebra` is the main entry point. Its `einsum()` function dispatches through `Backends/Dispatch.hpp` which selects: BLAS specializations > MLIR JIT (if enabled) > generic algorithm.

### Tensor Types

- `Tensor<T, Rank>` - dense tensor (column-major; the `EINSUMS_ROW_MAJOR_DEFAULT` CMake option existed historically but never controlled the stride initializer — tensors are always column-major at construction)
- `TensorView<T, Rank>` - non-owning view into a tensor
- `BlockTensor`, `TiledTensor` - structured variants
- `T.data()` for raw pointer, `T.dim(i)` for size, `T.stride(i)` for stride

### Index System

Indices are compile-time structs created with `MAKE_INDEX(x)` -> struct with `static constexpr const char *letter = "x"`. Index letters drive compile-time contraction analysis.

### Test System

Tests use Catch2 v3. CMake helpers:
```cmake
# Create test executable (not installed)
einsums_add_executable(MyTest_test INTERNAL_FLAGS SOURCES MyTest.cpp NOINSTALL)

# Register as unit test (creates ctest entry under Tests.Unit.Modules.<Module>.MyTest)
einsums_add_unit_test("Modules.ModuleName" MyTest)

# Performance test
einsums_add_performance_test("Modules.ModuleName" MyTest)
```

### CMake Options

Key options (`-D<option>=ON/OFF`):
- `EINSUMS_WITH_TESTS` / `EINSUMS_WITH_TESTS_UNIT` / `EINSUMS_WITH_TESTS_BENCHMARKS` - test categories
- `EINSUMS_WITH_MPI` - MPI distributed computing (Comm module, ProcessGrid, distributed passes)
- `EINSUMS_WITH_CUDA` / `EINSUMS_WITH_HIP` - GPU backends
- `EINSUMS_BUILD_PYTHON` - Python bindings via pybind11
- `EINSUMS_WITH_PROFILER` - built-in profiling support

### Config Defines

Build options map to preprocessor defines: `EINSUMS_WITH_FOO=ON` -> `einsums_add_config_define(EINSUMS_HAVE_FOO)` -> `#if defined(EINSUMS_HAVE_FOO)` in code. The generated config header is `<Einsums/Config.hpp>`.

### External Dependencies

Each dependency has a `cmake/Einsums_Setup<Dep>.cmake` file included from the root CMakeLists.txt. Required: BLAS/LAPACK, HDF5, OpenMP. Auto-fetched if missing: fmt, Catch2, spdlog. Optional: MPI (Open MPI or MPICH).

### Distributed Computing (MPI)

The `Comm` module (`libs/Einsums/Comm/`) provides MPI communication with a mock backend for serial builds. Key components:

- **ProcessGrid** (`Comm/ProcessGrid.hpp`): 2D Pr×Pc process grid with row/col sub-communicators. Auto-computes near-square factorization.
- **DistributionDescriptor** (`Comm/DistributionDescriptor.hpp`): Per-dimension grid axis assignment (None/Row/Col) with balanced blocking.
- **Collectives** (`Comm/Collectives.hpp`): Allreduce, broadcast, scatter, allgather — blocking and non-blocking (iallreduce).

ComputeGraph passes for automatic distribution:
- `DistributionPlanning`: Classifies einsum indices (target_a→Row, target_b→Col, shared→balanced, link→None). Detects chain conflicts.
- `Materialization`: Resizes deferred tensors to local partitions using `DistributionDescriptor::local_dims_for()`.
- `InputSlicing`: Creates temporary views of pre-allocated inputs for each rank's slice. Handles cross-axis permute.
- `SUMMAExpansion`: Replaces einsum with broadcast+GEMM loop on square grids.
- `CommunicationInsertion`: Inserts allreduce for replicated outputs from distributed inputs.
- `CommunicationScheduling`: Splits allreduce into async iallreduce + wait for overlap.

```bash
# Run MPI tests
mpirun -np 4 ./build/libs/Einsums/ComputeGraph/tests/unit/DistributedIntegration_test

# Test on non-square grid
mpirun -np 6 ./build/libs/Einsums/ComputeGraph/tests/unit/DistributedIntegration_test
```

**Important**: `einsums_add_executable` already links `libEinsums` — do NOT add extra `target_link_libraries` for Einsums modules in test CMakeLists (causes duplicate symbol issues with MPI).
