⏺ Unfortunately, Metal is limited for scientific computing:

What Metal offers:
- 2.9 TFLOPS FP32 GEMM via MPS on M4 — fast for single precision
- Metal-cpp provides C++ bindings
- Custom compute shaders in Metal Shading Language (C++14 derivative)

What it can't do:
- No FP64 support at all — Metal doesn't support double precision on the GPU
- MPS matrix multiply is FP32/FP16 only
- MPS API is primarily Objective-C/Swift (Metal-cpp wraps it but it's awkward)
- Matrix size limit of 65535 in any dimension

So for Einsums, Metal is not viable for the general case — quantum chemistry needs double precision. The 2.9 TFLOPS FP32 on the GPU is actually the
same as what Accelerate gives you on the CPU via SME (which we're already using), so there's no speedup to gain even for FP32.

What you could prototype on your laptop for eventual CUDA:

1. The dispatch infrastructure — the DeviceTensor type, host-device transfer logic, GPU memory management already exist in Einsums (the GPUMemory,
   GPUStreams modules). You could work on the architecture without needing a GPU.
2. The ComputeGraph Ozaki pass — the pass logic (graph walking, node rewriting) is pure CPU code. You could implement and test it with a mock "GPU
   GEMM" that just calls CPU float GEMM, then swap in real cuBLAS when you get to CUDA hardware.
3. cuTENSOR/cuBLAS integration design — plan the CMake detection, the BLAS vendor abstraction for GPU, the tensor contraction dispatch for GPU
   tensors.

Would any of these interest you, or would you rather focus on something else?

Sources:
- https://developer.apple.com/documentation/metalperformanceshaders/mpsmatrixmultiplication
- https://github.com/bkvogel/metal_performance_testing
- https://arxiv.org/pdf/2502.05317
- https://developer.apple.com/metal/Metal-Shading-Language-Specification.pdf


⏺ Here's my honest assessment:

GPU Code Assessment

What exists (~11,600 lines)

GPUMemory module (~1,140 lines)
- GPUPointer — well-designed fancy pointer wrapping HIP device pointers. Handles complex type mapping (std::complex → hipComplex), iterator
  semantics, host↔device memcpy via std::memcpy overloads. This is solid C++20 code with concepts.
- GPUAllocator — standard allocator interface for GPU memory (hipMalloc/hipFree). Clean.

GPUStreams module (~440 lines)
- Stream management, hipBLAS/hipSolver handle lifecycle. Global state with thread-local stream selection.

hipBLAS/hipBLASVendor (~4,600 lines)
- Mirror of the CPU BLASVendor module but for hipBLAS/hipSolver. Covers gemm, gemv, dot, ger, gesv, gesvd, getrf, getri, syev, heev, scal, axpy, etc.
- Includes custom HIP kernels (lassq, scal, direct_prod) for operations not in hipBLAS.

DeviceTensor (~2,615 lines)
- Full tensor type for GPU-resident data. Parallel structure to CPU Tensor<T, Rank>.

GPULinearAlgebra (~622 lines)
- GPU versions of gesv, gemm, symm_gemm, direct_product, dot, etc.

GPUTensorAlgebra (~346 lines)
- GPU einsum dispatch paths.

The honest assessment

The good:
- It's a substantial, working GPU backend — this is not a skeleton
- The architecture mirrors the CPU path cleanly (same API, same concepts, GPU-specific implementations)
- GPUPointer is well-designed with proper C++20 concepts
- Error handling via hip_catch() is consistent
- 532 HIP API calls in production code — comprehensive BLAS/LAPACK coverage

The concerns:

1. HIP-only, no CUDA directly. HIP is AMD's portability layer that can target both AMD (ROCm) and NVIDIA (via hipcc translating to CUDA). In theory
   this gives you both vendors. In practice, HIP-on-CUDA has rough edges and lags behind native CUDA. There's no direct cuBLAS/cuSOLVER path.
2. No abstraction layer between Einsums and HIP. Every GPU file directly includes hip/hip_runtime.h, hipblas/hipblas.h, etc. If you wanted to add a
   Metal, SYCL, or native CUDA backend, you'd need to touch every file. There's no GPU abstraction equivalent to what we built with the SIMD module.
3. DeviceTensor is a near-duplicate of Tensor. 2,615 lines that largely parallel the CPU Tensor but with hipMalloc/hipMemcpy instead of std::vector.
   This is a maintenance burden — any Tensor API change needs to be mirrored.
4. Global stream state. GPUStreams uses global/thread-local hipBLAS handles. This works but doesn't compose well with the ComputeGraph model where
   you'd want per-graph or per-node stream management.
5. No async pipeline. The current code is mostly synchronous — hipMemcpy (not hipMemcpyAsync), no overlap of compute and transfer. For large-scale
   tensor contractions, overlapping data movement with computation is where the real GPU speedup comes from.
6. The F-01 bug we found — the TensorAlgebra CMakeLists.txt typo (EXTRA_TENSOR_ALGEBA_DEPENDENCIES) means GPU deps were silently dropped. This
   suggests the GPU path may not have been tested recently.

If you were adding CUDA support today

Rather than extending the HIP backend, I'd suggest:

1. Create a GPU abstraction layer (like we did with SIMD) — einsums::gpu::malloc(), einsums::gpu::memcpy(), einsums::gpu::gemm() that dispatch to HIP
   or CUDA at compile time
2. Unify DeviceTensor with Tensor — use a memory policy template parameter rather than a separate class
3. Integrate with ComputeGraph — GPU operations as graph nodes, stream management as part of graph execution, Ozaki pass as an optimizer


Plan: GPU Abstraction Layer for Einsums

Context

Einsums currently has a GPU backend (~11,600 lines) that directly uses HIP APIs throughout:
- GPUMemory module: GPUPointer, GPUAllocator — wrapping hipMalloc/hipFree/hipMemcpy
- GPUStreams module: stream/handle management — wrapping hipblasHandle_t, hipsolverHandle_t, hipStream_t
- hipBLAS/hipBLASVendor modules: BLAS/LAPACK on GPU — wrapping hipblasSgemm, hipsolverDnDsyev, etc.
- DeviceTensor<T, Rank>: GPU-resident tensor — near-duplicate of Tensor<T, Rank>

Every GPU file directly includes hip/hip_runtime.h, hipblas/hipblas.h, etc. Adding CUDA support would require touching every file, and there's no
path to other backends.

Goal: Create a new GPU module providing a thin abstraction layer over GPU vendor APIs, similar to what SIMD does for CPU intrinsics. Both native CUDA
(cuBLAS/cuSOLVER) and native HIP (hipBLAS/hipSOLVER) backends, selected at compile time. The existing HIP code continues to work during migration.

Module Structure

libs/Einsums/GPU/
CMakeLists.txt
include/Einsums/GPU/
Platform.hpp              # GPU backend detection (CUDA vs HIP), constexpr flags
Runtime.hpp               # malloc/free/memcpy/memset — wraps hipMalloc/cudaMalloc
Stream.hpp                # Stream and handle management — wraps hipStream/cudaStream
BLAS.hpp                  # GPU GEMM/GEMV/dot etc. — wraps hipblas/cublas
Solver.hpp                # GPU eigenvalue/SVD/factorization — wraps hipsolver/cusolver
Error.hpp                 # Error checking — wraps hip_catch/cuda_catch
Types.hpp                 # Complex type mapping, operation enums
src/
Runtime.cpp (or .hip/.cu) # Implementation of runtime functions
Stream.cpp                # Stream management implementation
BLAS.cpp                  # BLAS dispatch
Solver.cpp                # Solver dispatch
tests/
unit/
Runtime.cpp             # malloc/free/memcpy round-trip tests
BLAS.cpp                # GPU GEMM correctness tests

Design Details

Platform.hpp — Backend Detection

namespace einsums::gpu {

inline constexpr bool has_cuda =
#if defined(EINSUMS_HAVE_CUDA)
true;
#else
false;
#endif

inline constexpr bool has_hip =
#if defined(EINSUMS_HAVE_HIP)
true;
#else
false;
#endif

inline constexpr bool has_gpu = has_cuda || has_hip;

} // namespace einsums::gpu

Runtime.hpp — Memory Operations

Thin wrappers with Einsums-specific naming. One function, two implementations selected by #if:

namespace einsums::gpu {

void *device_malloc(size_t bytes);
void  device_free(void *ptr);
void  memcpy_host_to_device(void *dst, void const *src, size_t bytes);
void  memcpy_device_to_host(void *dst, void const *src, size_t bytes);
void  memcpy_device_to_device(void *dst, void const *src, size_t bytes);
void  device_memset(void *ptr, int value, size_t bytes);
void  device_synchronize();

// Async variants
void  memcpy_host_to_device_async(void *dst, void const *src, size_t bytes, stream_t stream);
void  memcpy_device_to_host_async(void *dst, void const *src, size_t bytes, stream_t stream);

} // namespace einsums::gpu

Implementation (in .cpp):
void *device_malloc(size_t bytes) {
void *ptr;
#if defined(EINSUMS_HAVE_CUDA)
gpu_catch(cudaMalloc(&ptr, bytes));
#elif defined(EINSUMS_HAVE_HIP)
gpu_catch(hipMalloc(&ptr, bytes));
#endif
return ptr;
}

Stream.hpp — Handle and Event Management

Type aliases that resolve to the vendor type, plus event-based synchronization
for ComputeGraph integration:

namespace einsums::gpu {

#if defined(EINSUMS_HAVE_CUDA)
using stream_t        = cudaStream_t;
using blas_handle_t   = cublasHandle_t;
using solver_handle_t = cusolverDnHandle_t;
using event_t         = cudaEvent_t;
#elif defined(EINSUMS_HAVE_HIP)
using stream_t        = hipStream_t;
using blas_handle_t   = hipblasHandle_t;
using solver_handle_t = hipsolverHandle_t;
using event_t         = hipEvent_t;
#else  // Mock
using stream_t        = int;
using blas_handle_t   = int;
using solver_handle_t = int;
using event_t         = int;
#endif

// --- Thread-local stream management (existing pattern) ---
stream_t get_stream();
stream_t get_stream(int thread_id);
blas_handle_t get_blas_handle();
solver_handle_t get_solver_handle();
void stream_wait(stream_t stream);
void stream_wait(bool may_skip = false);
void all_stream_wait();
void device_synchronize();

// --- Stream creation/destruction (for ComputeGraph node-level streams) ---
stream_t create_stream();
void     destroy_stream(stream_t stream);

// --- Events for dependency tracking between graph nodes ---
// A graph node records an event after launching work on its stream.
// Dependent nodes wait on the event before starting, enabling overlap.
event_t create_event();
void    destroy_event(event_t event);
void    record_event(event_t event, stream_t stream);    // Mark current point in stream
void    stream_wait_event(stream_t stream, event_t event); // Stream waits until event completes
bool    event_completed(event_t event);                    // Non-blocking query

} // namespace einsums::gpu

The event API enables ComputeGraph to implement fine-grained dependency tracking:
Node A (stream 1): gpu::blas::gemm(...); gpu::record_event(event_A, stream_1);
Node B (stream 2): gpu::stream_wait_event(stream_2, event_A); gpu::blas::gemm(...);
This allows overlapping compute and data movement without full device synchronization.
Mock backend: events are no-ops (everything is synchronous on CPU).

BLAS.hpp — GPU BLAS Operations

Same interface as CPU BLAS but operating on device pointers, plus reduced-precision
GEMM for the Ozaki optimization pass:

namespace einsums::gpu::blas {

// --- Standard precision GEMM (matches CPU BLAS signatures) ---
void sgemm(char transa, char transb, int64_t m, int64_t n, int64_t k,
float alpha, float const *a, int64_t lda,
float const *b, int64_t ldb, float beta, float *c, int64_t ldc);

void dgemm(char transa, char transb, int64_t m, int64_t n, int64_t k,
double alpha, double const *a, int64_t lda,
double const *b, int64_t ldb, double beta, double *c, int64_t ldc);

// ... cgemm, zgemm

// Template wrapper:
template <typename T>
void gemm(char transa, char transb, int64_t m, int64_t n, int64_t k,
T alpha, T const *a, int64_t lda, T const *b, int64_t ldb,
T beta, T *c, int64_t ldc);

// --- Reduced-precision GEMM for Ozaki pass ---
// These use tensor cores (FP16/FP8 compute) with FP32 accumulation.
// Input: FP16 (half) or FP8 (E4M3) matrices
// Accumulation: FP32
// Output: FP32

// FP16 GEMM: C(fp32) = alpha * A(fp16) * B(fp16) + beta * C(fp32)
// Uses cublasGemmEx / hipblasGemmEx with CUBLAS_COMPUTE_32F
void hgemm(char transa, char transb, int64_t m, int64_t n, int64_t k,
float alpha, __half const *a, int64_t lda,
__half const *b, int64_t ldb, float beta, float *c, int64_t ldc);

// FP8 GEMM (Blackwell/Hopper): C(fp32) = alpha * A(fp8) * B(fp8) + beta * C(fp32)
// Uses cublasLtMatmul with FP8 E4M3 format
void fp8gemm(char transa, char transb, int64_t m, int64_t n, int64_t k,
float alpha, void const *a, int64_t lda,  // E4M3 data
void const *b, int64_t ldb,                // E4M3 data
float beta, float *c, int64_t ldc);

} // namespace einsums::gpu::blas

The reduced-precision GEMM functions are the key enablers for the Ozaki pass:
- hgemm: FP16 input, FP32 accumulation — works on all NVIDIA GPUs with tensor cores (Volta+)
- fp8gemm: FP8 E4M3 input, FP32 accumulation — Hopper (H100) and Blackwell (B200, RTX 5000)
- Mock backend: both delegate to CPU sgemm (input data would be pre-converted to float)

The Ozaki pass would:
1. Split FP64 matrices into FP16/FP8 slices
2. Call hgemm/fp8gemm for each pair of slices (using tensor cores)
3. Accumulate partial products in FP64 on the host or in a separate kernel

Error.hpp — Unified Error Checking

namespace einsums::gpu {

// Unified error check macro — works for both CUDA and HIP
#if defined(EINSUMS_HAVE_CUDA)
#define gpu_catch(call) /* check cudaError_t */
#define gpu_blas_catch(call) /* check cublasStatus_t */
#define gpu_solver_catch(call) /* check cusolverStatus_t */
#elif defined(EINSUMS_HAVE_HIP)
#define gpu_catch(call) /* check hipError_t */
#define gpu_blas_catch(call) /* check hipblasStatus_t */
#define gpu_solver_catch(call) /* check hipsolverStatus_t */
#endif

} // namespace einsums::gpu

Types.hpp — Type Mapping

namespace einsums::gpu {

// Map std::complex to vendor complex types for BLAS calls
template <typename T> struct DeviceComplexType { using type = T; };

#if defined(EINSUMS_HAVE_CUDA)
template <> struct DeviceComplexType<std::complex<float>>  { using type = cuFloatComplex; };
template <> struct DeviceComplexType<std::complex<double>> { using type = cuDoubleComplex; };
#elif defined(EINSUMS_HAVE_HIP)
template <> struct DeviceComplexType<std::complex<float>>  { using type = hipFloatComplex; };
template <> struct DeviceComplexType<std::complex<double>> { using type = hipDoubleComplex; };
#endif

template <typename T>
using device_complex_t = typename DeviceComplexType<T>::type;

// BLAS operation enum
enum class Operation { None, Transpose, ConjTranspose };

// Reduced-precision type aliases for Ozaki pass
#if defined(EINSUMS_HAVE_CUDA)
using half_t = __half;            // CUDA FP16
using fp8_t  = __nv_fp8_e4m3;    // CUDA FP8 E4M3 (Hopper+)
#elif defined(EINSUMS_HAVE_HIP)
using half_t = _Float16;          // HIP FP16
using fp8_t  = uint8_t;           // HIP FP8 (placeholder, vendor-specific)
#else  // Mock
using half_t = uint16_t;          // Mock: raw 16-bit storage
using fp8_t  = uint8_t;           // Mock: raw 8-bit storage
#endif

// Feature detection for reduced precision
inline constexpr bool has_fp16_gemm = has_gpu;  // All tensor-core GPUs support FP16
inline constexpr bool has_fp8_gemm =
#if defined(EINSUMS_HAVE_CUDA) && defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 890
true;   // Hopper (sm_89+) and Blackwell
#else
false;
#endif

} // namespace einsums::gpu

Mock Backend — Testable Without GPU Hardware

A third backend alongside CUDA and HIP that uses the existing CPU infrastructure:

void *device_malloc(size_t bytes) {
#if defined(EINSUMS_HAVE_CUDA)
void *ptr; cudaMalloc(&ptr, bytes); return ptr;
#elif defined(EINSUMS_HAVE_HIP)
void *ptr; hipMalloc(&ptr, bytes); return ptr;
#else  // Mock backend
return std::malloc(bytes);    // Just host memory
#endif
}

void memcpy_host_to_device(void *dst, void const *src, size_t bytes) {
#if defined(EINSUMS_HAVE_CUDA)
cudaMemcpy(dst, src, bytes, cudaMemcpyHostToDevice);
#elif defined(EINSUMS_HAVE_HIP)
hipMemcpy(dst, src, bytes, hipMemcpyHostToDevice);
#else
std::memcpy(dst, src, bytes);  // Same address space
#endif
}

For BLAS, the mock backend delegates to the existing CPU BLAS vendor layer:

// gpu::blas::sgemm mock → einsums::blas::vendor::sgemm
// gpu::blas::dgemm mock → einsums::blas::vendor::dgemm
// etc.

For streams/handles, the mock backend uses no-ops or trivial placeholders:

#if defined(EINSUMS_HAVE_CUDA)
using stream_t = cudaStream_t;
#elif defined(EINSUMS_HAVE_HIP)
using stream_t = hipStream_t;
#else
using stream_t = int;  // Mock: unused placeholder
#endif

void stream_wait(bool) { /* mock: no-op, already synchronous */ }
void device_synchronize() { /* mock: no-op */ }

This means the entire GPU module compiles, links, and tests on any machine — M4 Mac, Linux CI without GPU, Windows. When a real GPU backend is
enabled, only the #if branches change.

CMake Integration

# libs/Einsums/GPU/CMakeLists.txt
# Always builds — mock backend when no GPU, real backend when CUDA/HIP enabled

set(GPUHeaders
Einsums/GPU/Platform.hpp
Einsums/GPU/Runtime.hpp
Einsums/GPU/Stream.hpp
Einsums/GPU/BLAS.hpp
Einsums/GPU/Solver.hpp
Einsums/GPU/Error.hpp
Einsums/GPU/Types.hpp
)

set(GPUSources Runtime.cpp Stream.cpp BLAS.cpp Solver.cpp)

if(EINSUMS_WITH_CUDA)
set(GPU_VENDOR_DEPS CUDA::cublas CUDA::cusolver)
elseif(EINSUMS_WITH_HIP)
set(GPU_VENDOR_DEPS hip::host hip::device roc::hipblas roc::hipsolver)
else()
# Mock backend: depends on CPU BLAS
set(GPU_VENDOR_DEPS)
endif()

einsums_add_module(
Einsums GPU
SOURCES ${GPUSources}
HEADERS ${GPUHeaders}
DEPENDENCIES ${GPU_VENDOR_DEPS}
MODULE_DEPENDENCIES Einsums_Config Einsums_Errors Einsums_Logging Einsums_BLAS Einsums_BLASVendor
CMAKE_SUBDIRS tests
)

Module always builds (no return() guard). Registered in libs/Einsums/CMakeLists.txt before GPUMemory and GPUStreams.

Implementation Steps

Step 1: Create the module skeleton

- Directory structure, CMakeLists.txt, register in module list
- Platform.hpp with backend detection constants
- Error.hpp with unified error checking macros (no-ops for mock)
- Types.hpp with complex type mapping (identity for mock)

Step 2: Runtime.hpp — Memory operations

- device_malloc/free/memcpy/memset with CUDA, HIP, and mock backends
- Mock: std::malloc/std::free/std::memcpy
- Async variants (mock: same as sync)
- Unit tests for malloc/free/memcpy round-trip — run on your M4 Mac

Step 3: Stream.hpp — Handle management

- Type aliases for stream/handle types (mock: int placeholders)
- get_stream/get_blas_handle/get_solver_handle (mock: return dummy values)
- Stream wait and synchronization primitives (mock: no-ops)

Step 4: BLAS.hpp — GPU BLAS

- gemm, gemv, dot, ger, axpy, scal for all 4 types
- Mock: delegate to einsums::blas::vendor::sgemm etc.
- Unit tests comparing gpu::blas::gemm results against direct CPU BLAS — run on your M4 Mac

Step 5: Solver.hpp — GPU LAPACK

- syev/heev, gesv, getrf/getri, gesvd
- Mock: delegate to CPU LAPACK

Step 6: Migration (future, not in initial scope)

- GPUMemory → uses gpu::device_malloc/free instead of hipMalloc/hipFree
- GPUStreams → uses gpu::stream_t instead of hipStream_t
- hipBLAS → uses gpu::blas::gemm instead of hipblasSgemm
- DeviceTensor → uses gpu:: abstraction throughout
- Automatic dispatch in LinearAlgebra: linear_algebra::gesv, linear_algebra::syev, etc. would use
  if constexpr (DeviceTensorConcept<AType>) to route to gpu::solver::gesv / gpu::blas::gemm
  automatically. The user writes one call (linear_algebra::gesv(&A, &B)) and it works for both
  CPU and GPU tensors. This eliminates the separate GPULinearAlgebra.hpp entirely.
  The gpu::blas and gpu::solver interfaces are designed to match CPU BLAS/LAPACK signatures
  so this dispatch is a clean swap.
- ComputeGraph GPU execution: Graph executor assigns gpu::stream_t per node (or group of
  independent nodes). Dependencies between nodes use gpu::event_t — node A records an event
  after launching GPU work, node B waits on that event before starting. This enables overlapping
  compute and data movement. The Ozaki pass (future) would rewrite GEMM nodes to use
  gpu::blas::sgemm (float) instead of gpu::blas::dgemm (double), with split/accumulate
  nodes around them.

Key Design Decisions

┌─────────────────┬──────────────────────────────┬───────────────────────────────────────────────┐
│    Decision     │            Choice            │                   Rationale                   │
├─────────────────┼──────────────────────────────┼───────────────────────────────────────────────┤
│ Both CUDA + HIP │ Native backends              │ Best performance on each vendor               │
├─────────────────┼──────────────────────────────┼───────────────────────────────────────────────┤
│ Mock backend    │ CPU fallback                 │ Full test coverage without GPU hardware       │
├─────────────────┼──────────────────────────────┼───────────────────────────────────────────────┤
│ New module      │ Migrate later                │ No breakage during transition                 │
├─────────────────┼──────────────────────────────┼───────────────────────────────────────────────┤
│ API level       │ CUDA/HIP focused             │ Familiar semantics, easy to map               │
├─────────────────┼──────────────────────────────┼───────────────────────────────────────────────┤
│ Namespace       │ einsums::gpu                 │ Matches existing convention                   │
├─────────────────┼──────────────────────────────┼───────────────────────────────────────────────┤
│ Dispatch        │ Compile-time #if             │ Same pattern as SIMD module                   │
├─────────────────┼──────────────────────────────┼───────────────────────────────────────────────┤
│ Source files    │ .cpp with #if (not .hip/.cu) │ Simpler build, vendor headers handle the rest │
└─────────────────┴──────────────────────────────┴───────────────────────────────────────────────┘

Critical Files

- New: libs/Einsums/GPU/ (entire module)
- Modified: libs/Einsums/CMakeLists.txt (add GPU to module list)
- Reference: libs/Einsums/GPUStreams/ (existing patterns to wrap)
- Reference: libs/Einsums/GPUMemory/ (existing patterns to wrap)
- Reference: libs/Einsums/hipBLASVendor/src/gemm.hip (BLAS call patterns)
- Reference: libs/Einsums/SIMD/ (module design pattern to follow)

Verification

1. Build: Module compiles with EINSUMS_WITH_HIP=ON (your existing setup) and would compile with EINSUMS_WITH_CUDA=ON on NVIDIA hardware
2. Unit tests: malloc/free round-trip, memcpy host↔device, GEMM correctness vs CPU reference
3. Existing GPU tests: Continue to pass (existing HIP code not modified)
4. No-GPU build: Module skipped cleanly when EINSUMS_WITH_GPU_SUPPORT=OFF

Not Yet Implemented

High Priority

1. Ozaki FP64 emulation pass — Designed (plan approved), not coded. Needs: splitting kernels, mock hgemm, OzakiSplit pass, tests
2. GPU PackedGemm — Device-side packing/transpose/unpacking kernels so arbitrary-rank contractions run on GPU. Prerequisite for Ozaki.
3. ContractionPlanning graph restructuring — The pass analyzes and reports optimal orderings but doesn't actually rewrite the graph. Blocked on correct
   leaf-tensor decomposition for the DP tree reconstruction.

Medium Priority

4. CUDA/HIP hgemm implementation — Currently a stub. Needs cublasGemmEx with CUDA_R_16F / CUBLAS_COMPUTE_32F
5. CUDA/HIP fp8gemm implementation — Stub. Needs cublasLtMatmul (Hopper+ only)
6. Calibration tool improvements — --database mode, --device gpu, rectangular GEMM shapes, argc/argv forwarding
7. --einsums:hardware-profile config option — Runtime override to load custom JSON profile
8. Shipped hardware_profiles.json file — Install rule for the database (currently compiled into load_defaults())
9. Mixed distributed/local einsum — The overloads for distributed C + regular A/B hit template resolution issues. Needs concept refinement.
10. gpu::is_device_pointer() — Needed for NCCL dispatch in collectives. Not yet implemented.

Lower Priority / Future

11. CRT-based Ozaki Scheme II — Reduces s² GEMMs to ~s GEMMs
12. FP8 precision option — Hopper+ only, more splits needed
13. CommunicationInsertion — Currently placeholder. Needs explicit collective nodes when CommunicationScheduling can overlap them
14. CommunicationScheduling — Currently detects existing async nodes. Needs to split synchronous collectives into async phases
15. Multi-GPU support — Multiple GPUs per node, NCCL for inter-GPU comm
16. Distributed Pipeline stages — Pipeline with per-stage distribution planning
17. Shell-pair parallel_for blueprint — Production ERI computation with round-robin shell pairs

Quick wins (complete half-built passes):
  1. PermuteFusion rewrite step — the analysis already finds Permute→Einsum chains fusible via
  transpose flags. Wiring the actual graph rewrite eliminates an entire tensor copy per match.
  Completes an existing pass instead of building new.
  2. GEMMBatching rewrite step — same shape: detection exists, rewrite doesn't. Batched-GEMM dispatch
  amortizes kernel-launch overhead; huge win on GPU for many small contractions (common in CCSD,
  quantum chemistry).

  Domain-specific optimizations (high-value for a tensor library):
  3. Full ElementWiseFusion — current pass only merges Scale→Scale. Generalizing to any chain of
  pointwise ops (add/mul/abs/sqrt/exp/...) into one fused kernel is a standard ML-compiler win.
  Moderate effort, big bandwidth savings.
  4. Algebraic-identity simplifier — A + 0, A * 1, A * 0, transpose(transpose(A)) → A,
  reshape(reshape(...)), A^{-1} * A → I. ConstantFolding doesn't cover these because the operands
  aren't literal constants. Cheap to write; surprisingly common in generated/templated code.
  5. Rematerialization (gradient-checkpointing style) — when peak memory exceeds a budget, recompute
  cheap intermediates instead of storing them. Unlocks problem sizes that currently OOM. Pairs well
  with MemoryPlanning (which already measures peak).

  Novel structural passes:
  6. Ozaki GEMM pass — already in the plan (memory mentions it). Split FP64 into multiple FP32/FP16
  products to use Tensor Cores on NVIDIA. The analysis (when it's numerically safe) and rewrite (emit
  the split + accumulate) sits naturally as a ComputeGraph pass next to ContractionPlanning.
  7. Low-rank approximation detection — spot when a tensor's effective rank is << full size (common in
   post-HF methods, kernel approximations) and auto-substitute a truncated SVD representation.
  Extremely domain-specific; very high leverage when it hits.
  8. Reuse-pool allocator via graph coloring — MemoryPlanning knows liveness; FreeInsertion adds Free
  nodes. A pass between them could run graph coloring to assign the same underlying buffer to
  non-overlapping tensors, reducing allocator pressure and fragmentation.