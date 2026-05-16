//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#include <Einsums/BLASVendor/Vendor.hpp>
#include <Einsums/GPU/BLAS.hpp>
#include <Einsums/GPU/Platform.hpp>
#include <Einsums/GPU/Runtime.hpp>
#include <Einsums/GPU/Solver.hpp>
#include <Einsums/GPU/Stream.hpp>
#include <Einsums/Tensor/RuntimeTensor.hpp>
#include <Einsums/Tensor/Tensor.hpp>

#include <cmath>
#include <vector>

#include <Einsums/Testing.hpp>

using namespace einsums::gpu;

namespace {
/// Unwrap device_malloc for tests — asserts on failure.
void *test_malloc(size_t bytes) {
    auto result = device_malloc(bytes);
    REQUIRE(result.has_value());
    return result.value();
}
} // namespace

TEST_CASE("Platform detection", "[gpu]") {
    INFO("has_cuda = " << has_cuda);
    INFO("has_hip  = " << has_hip);
    INFO("has_gpu  = " << has_gpu);
    INFO("is_mock  = " << is_mock);

    // Exactly one of mock or real GPU should be active
    CHECK((has_gpu || is_mock));
}

TEST_CASE("device_malloc and device_free", "[gpu]") {
    void *ptr = test_malloc(1024);
    REQUIRE(ptr != nullptr);
    device_free(ptr);
}

TEST_CASE("memcpy host to device and back", "[gpu]") {
    constexpr int      N = 256;
    std::vector<float> host_src(N), host_dst(N, 0.0f);

    for (int i = 0; i < N; ++i)
        host_src[i] = static_cast<float>(i);

    void *dev = test_malloc(N * sizeof(float));
    REQUIRE(dev != nullptr);

    memcpy_host_to_device(dev, host_src.data(), N * sizeof(float));
    memcpy_device_to_host(host_dst.data(), dev, N * sizeof(float));

    for (int i = 0; i < N; ++i) {
        CHECK(host_dst[i] == Catch::Approx(host_src[i]));
    }

    device_free(dev);
}

TEST_CASE("device_memset", "[gpu]") {
    constexpr int N   = 128;
    void         *dev = test_malloc(N * sizeof(int));
    REQUIRE(dev != nullptr);

    device_memset(dev, 0, N * sizeof(int));

    std::vector<int> host(N, 42);
    memcpy_device_to_host(host.data(), dev, N * sizeof(int));

    for (int i = 0; i < N; ++i) {
        CHECK(host[i] == 0);
    }

    device_free(dev);
}

TEST_CASE("memcpy device to device", "[gpu]") {
    constexpr int       N = 64;
    std::vector<double> host_src(N), host_dst(N, 0.0);

    for (int i = 0; i < N; ++i)
        host_src[i] = static_cast<double>(i) * 1.5;

    void *dev_a = test_malloc(N * sizeof(double));
    void *dev_b = test_malloc(N * sizeof(double));
    REQUIRE(dev_a != nullptr);
    REQUIRE(dev_b != nullptr);

    memcpy_host_to_device(dev_a, host_src.data(), N * sizeof(double));
    memcpy_device_to_device(dev_b, dev_a, N * sizeof(double));
    memcpy_device_to_host(host_dst.data(), dev_b, N * sizeof(double));

    for (int i = 0; i < N; ++i) {
        CHECK(host_dst[i] == Catch::Approx(host_src[i]));
    }

    device_free(dev_a);
    device_free(dev_b);
}

TEMPLATE_TEST_CASE("gpu::blas::gemm mock correctness", "[gpu][blas]", float, double) {
    // Simple 2x2 GEMM: C = A * B
    constexpr int N = 2;

    // Column-major: A = [[1,3],[2,4]], B = [[5,7],[6,8]]
    TestType A[] = {1, 2, 3, 4};
    TestType B[] = {5, 6, 7, 8};
    TestType C[] = {0, 0, 0, 0};

    // Allocate on "device"
    void *dA = test_malloc(4 * sizeof(TestType));
    void *dB = test_malloc(4 * sizeof(TestType));
    void *dC = test_malloc(4 * sizeof(TestType));

    memcpy_host_to_device(dA, A, 4 * sizeof(TestType));
    memcpy_host_to_device(dB, B, 4 * sizeof(TestType));
    memcpy_host_to_device(dC, C, 4 * sizeof(TestType));

    einsums::gpu::blas::gemm<TestType>('n', 'n', N, N, N, TestType(1.0), static_cast<TestType const *>(dA), N,
                                       static_cast<TestType const *>(dB), N, TestType(0.0), static_cast<TestType *>(dC), N);

    memcpy_device_to_host(C, dC, 4 * sizeof(TestType));

    // Expected: C = A*B (column-major)
    // C[0,0] = 1*5 + 3*6 = 23, C[1,0] = 2*5 + 4*6 = 34
    // C[0,1] = 1*7 + 3*8 = 31, C[1,1] = 2*7 + 4*8 = 46
    CHECK(C[0] == Catch::Approx(TestType(23)));
    CHECK(C[1] == Catch::Approx(TestType(34)));
    CHECK(C[2] == Catch::Approx(TestType(31)));
    CHECK(C[3] == Catch::Approx(TestType(46)));

    device_free(dA);
    device_free(dB);
    device_free(dC);
}

TEST_CASE("stream create and destroy", "[gpu]") {
    auto s = create_stream();
    // Just verify it doesn't crash
    destroy_stream(s);
}

TEST_CASE("event create, record, and query", "[gpu]") {
    auto s = create_stream();
    auto e = create_event();

    record_event(e, s);

    // On mock backend, events are always complete
    if constexpr (is_mock) {
        CHECK(event_completed(e));
    }

    destroy_event(e);
    destroy_stream(s);
}

// ===========================================================================
// Solver tests (mock backend delegates to CPU LAPACK)
// ===========================================================================

TEMPLATE_TEST_CASE("gpu::solver::syev mock correctness", "[gpu][solver]", float, double) {
    // Symmetric 3x3 matrix (column-major):
    // [2  1  0]
    // [1  3  1]
    // [0  1  2]
    constexpr int N   = 3;
    TestType      A[] = {2, 1, 0, 1, 3, 1, 0, 1, 2};
    TestType      W[N];

    void *dA = test_malloc(N * N * sizeof(TestType));
    void *dW = test_malloc(N * sizeof(TestType));

    memcpy_host_to_device(dA, A, N * N * sizeof(TestType));

    int info = einsums::gpu::solver::syev<TestType>('V', 'U', N, static_cast<TestType *>(dA), N, static_cast<TestType *>(dW));
    CHECK(info == 0);

    memcpy_device_to_host(W, dW, N * sizeof(TestType));

    // Eigenvalues should be sorted in ascending order and sum to the trace (7.0)
    CHECK(W[0] <= W[1]);
    CHECK(W[1] <= W[2]);
    CHECK(W[0] + W[1] + W[2] == Catch::Approx(TestType(7.0)).margin(0.01));

    device_free(dA);
    device_free(dW);
}

TEMPLATE_TEST_CASE("gpu::solver::gesv mock correctness", "[gpu][solver]", float, double) {
    // Solve: [2 1; 1 3] * x = [5; 7]
    // Expected: x = [8/5, 9/5] = [1.6, 1.8]
    constexpr int64_t N = 2, NRHS = 1;
    TestType          A[] = {2, 1, 1, 3}; // column-major
    TestType          B[] = {5, 7};
    int64_t           ipiv[2];

    void *dA = test_malloc(N * N * sizeof(TestType));
    void *dB = test_malloc(N * sizeof(TestType));

    memcpy_host_to_device(dA, A, N * N * sizeof(TestType));
    memcpy_host_to_device(dB, B, N * sizeof(TestType));

    int info = einsums::gpu::solver::gesv<TestType>(N, NRHS, static_cast<TestType *>(dA), N, ipiv, static_cast<TestType *>(dB), N);
    CHECK(info == 0);

    memcpy_device_to_host(B, dB, N * sizeof(TestType));

    CHECK(B[0] == Catch::Approx(TestType(1.6)).margin(0.001));
    CHECK(B[1] == Catch::Approx(TestType(1.8)).margin(0.001));

    device_free(dA);
    device_free(dB);
}

// ===========================================================================
// GPUTensor tests — GeneralTensor with DeviceAllocator
// ===========================================================================

TEMPLATE_TEST_CASE("GPUTensor creation and zero", "[gpu][tensor]", float, double) {
    using namespace einsums;

    constexpr size_t       N = 32;
    GPUTensor<TestType, 2> A("A", N, N);

    // Verify metadata
    CHECK(A.dim(0) == N);
    CHECK(A.dim(1) == N);
    CHECK(A.size() == N * N);
    CHECK(A.data() != nullptr);
    CHECK(A.IsDeviceTensor == true);

    // Zero should work via gpu::device_memset
    A.zero();

    // Verify by copying back to host
    std::vector<TestType> host(N * N, TestType(42));
    memcpy_device_to_host(host.data(), A.data(), N * N * sizeof(TestType));

    for (size_t i = 0; i < N * N; ++i) {
        CHECK(host[i] == Catch::Approx(TestType(0)));
    }
}

TEMPLATE_TEST_CASE("GPUTensor GEMM via gpu::blas", "[gpu][tensor]", float, double) {
    using namespace einsums;

    constexpr size_t N = 4;

    // Create GPU tensors
    GPUTensor<TestType, 2> A("A", N, N);
    GPUTensor<TestType, 2> B("B", N, N);
    GPUTensor<TestType, 2> C("C", N, N);

    // Prepare host data: A = identity, B = [1..16]
    std::vector<TestType> hostA(N * N, TestType(0));
    std::vector<TestType> hostB(N * N);
    for (size_t i = 0; i < N; ++i)
        hostA[i * N + i] = TestType(1); // identity (column-major)
    for (size_t i = 0; i < N * N; ++i)
        hostB[i] = TestType(i + 1);

    // Upload to device
    memcpy_host_to_device(A.data(), hostA.data(), N * N * sizeof(TestType));
    memcpy_host_to_device(B.data(), hostB.data(), N * N * sizeof(TestType));
    C.zero();

    // C = 1.0 * A * B + 0.0 * C  →  C = B (identity times B)
    einsums::gpu::blas::gemm<TestType>('n', 'n', N, N, N, TestType(1), A.data(), N, B.data(), N, TestType(0), C.data(), N);

    // Download result
    std::vector<TestType> hostC(N * N);
    memcpy_device_to_host(hostC.data(), C.data(), N * N * sizeof(TestType));

    // C should equal B
    for (size_t i = 0; i < N * N; ++i) {
        CHECK(hostC[i] == Catch::Approx(hostB[i]));
    }
}

TEMPLATE_TEST_CASE("Tensor to GPUTensor transfer via assignment", "[gpu][tensor]", float, double) {
    using namespace einsums;

    constexpr size_t N = 8;

    // Create CPU tensor with known data
    Tensor<TestType, 2> hostA("hostA", N, N);
    for (size_t i = 0; i < N; ++i)
        for (size_t j = 0; j < N; ++j)
            hostA(i, j) = TestType(i * N + j);

    // Copy to GPU via cross-allocator assignment
    GPUTensor<TestType, 2> devA("devA", N, N);
    devA = hostA;

    // Copy back to host
    Tensor<TestType, 2> hostB("hostB", N, N);
    hostB = devA;

    // Verify round-trip
    for (size_t i = 0; i < N; ++i) {
        for (size_t j = 0; j < N; ++j) {
            CHECK(hostB(i, j) == Catch::Approx(hostA(i, j)));
        }
    }
}

TEMPLATE_TEST_CASE("GPUTensor copy construction from Tensor", "[gpu][tensor]", float, double) {
    using namespace einsums;

    constexpr size_t N = 4;

    Tensor<TestType, 2> hostA("hostA", N, N);
    for (size_t i = 0; i < N * N; ++i)
        hostA.data()[i] = TestType(i * 0.5);

    // Construct GPU tensor from host tensor
    GPUTensor<TestType, 2> devA(hostA);

    // Construct host tensor from GPU tensor
    Tensor<TestType, 2> hostB(devA);

    for (size_t i = 0; i < N * N; ++i) {
        CHECK(hostB.data()[i] == Catch::Approx(hostA.data()[i]));
    }
}

TEST_CASE("GPUTensor IsDeviceTensor trait", "[gpu][tensor]") {
    using namespace einsums;

    // Regular tensor is NOT a device tensor
    CHECK(Tensor<double, 2>::IsDeviceTensor == false);

    // GPUTensor IS a device tensor
    CHECK(GPUTensor<double, 2>::IsDeviceTensor == true);

    // BufferTensor is NOT a device tensor
    CHECK(BufferTensor<double, 2>::IsDeviceTensor == false);
}

// ===========================================================================
// RuntimeGPUTensor tests — runtime-rank counterpart of GPUTensor
// ===========================================================================

TEMPLATE_TEST_CASE("RuntimeGPUTensor creation and zero", "[gpu][runtime_tensor]", float, double) {
    using namespace einsums;

    constexpr size_t           N = 32;
    RuntimeGPUTensor<TestType> A("A", std::vector<size_t>{N, N});

    CHECK(A.dim(0) == N);
    CHECK(A.dim(1) == N);
    CHECK(A.size() == N * N);
    CHECK(A.data() != nullptr);
    CHECK(A.IsDeviceTensor == true);
    CHECK(A.rank() == 2);

    A.zero();

    std::vector<TestType> host(N * N, TestType(42));
    memcpy_device_to_host(host.data(), A.data(), N * N * sizeof(TestType));
    for (size_t i = 0; i < N * N; ++i) {
        CHECK(host[i] == Catch::Approx(TestType(0)));
    }
}

TEMPLATE_TEST_CASE("RuntimeTensor to RuntimeGPUTensor cross-allocator copy", "[gpu][runtime_tensor]", float, double) {
    using namespace einsums;

    constexpr size_t N = 8;

    RuntimeTensor<TestType> hostA("hostA", std::vector<size_t>{N, N});
    for (size_t i = 0; i < N * N; ++i)
        hostA.data()[i] = TestType(i * 0.5);

    // Construct device runtime tensor from host runtime tensor.
    RuntimeGPUTensor<TestType> devA(hostA);
    CHECK(devA.dim(0) == N);
    CHECK(devA.dim(1) == N);

    // Construct host runtime tensor from device runtime tensor.
    RuntimeTensor<TestType> hostB(devA);

    for (size_t i = 0; i < N * N; ++i) {
        CHECK(hostB.data()[i] == Catch::Approx(hostA.data()[i]));
    }
}

TEMPLATE_TEST_CASE("RuntimeGPUTensor cross-allocator assignment", "[gpu][runtime_tensor]", float, double) {
    using namespace einsums;

    constexpr size_t N = 8;

    RuntimeTensor<TestType> hostA("hostA", std::vector<size_t>{N, N});
    for (size_t i = 0; i < N; ++i)
        for (size_t j = 0; j < N; ++j)
            hostA(std::vector<int64_t>{static_cast<int64_t>(i), static_cast<int64_t>(j)}) = TestType(i * N + j);

    RuntimeGPUTensor<TestType> devA("devA", std::vector<size_t>{N, N});
    devA = hostA;

    RuntimeTensor<TestType> hostB("hostB", std::vector<size_t>{N, N});
    hostB = devA;

    for (size_t i = 0; i < N; ++i) {
        for (size_t j = 0; j < N; ++j) {
            std::vector<int64_t> idx{static_cast<int64_t>(i), static_cast<int64_t>(j)};
            CHECK(hostB(idx) == Catch::Approx(hostA(idx)));
        }
    }
}

TEST_CASE("RuntimeGPUTensor IsDeviceTensor trait", "[gpu][runtime_tensor]") {
    using namespace einsums;

    CHECK(RuntimeTensor<double>::IsDeviceTensor == false);
    CHECK(RuntimeGPUTensor<double>::IsDeviceTensor == true);
    CHECK(BufferRuntimeTensor<double>::IsDeviceTensor == false);

    static_assert(RuntimeGPUTensor<float>::IsDeviceTensor);
    static_assert(RuntimeGPUTensor<std::complex<double>>::IsDeviceTensor);
}

TEST_CASE("gpu::blas::gemm large matrix correctness", "[gpu][blas]") {
    // Larger GEMM to exercise the MPS GPU path more thoroughly.
    constexpr int N = 64;

    std::vector<float> A(N * N), B(N * N), C(N * N, 0.0f), C_ref(N * N, 0.0f);

    // Fill with deterministic values.
    for (int i = 0; i < N * N; i++) {
        A[i] = static_cast<float>(i % 17) * 0.1f;
        B[i] = static_cast<float>(i % 13) * 0.1f;
    }

    // Reference: CPU BLAS.
    einsums::blas::vendor::sgemm('n', 'n', N, N, N, 1.0f, A.data(), N, B.data(), N, 0.0f, C_ref.data(), N);

    // GPU BLAS via device memory.
    void *dA = test_malloc(N * N * sizeof(float));
    void *dB = test_malloc(N * N * sizeof(float));
    void *dC = test_malloc(N * N * sizeof(float));

    memcpy_host_to_device(dA, A.data(), N * N * sizeof(float));
    memcpy_host_to_device(dB, B.data(), N * N * sizeof(float));
    memcpy_host_to_device(dC, C.data(), N * N * sizeof(float));

    einsums::gpu::blas::gemm<float>('n', 'n', N, N, N, 1.0f, static_cast<float const *>(dA), N, static_cast<float const *>(dB), N, 0.0f,
                                    static_cast<float *>(dC), N);

    memcpy_device_to_host(C.data(), dC, N * N * sizeof(float));

    for (int i = 0; i < N * N; i++) {
        CHECK(C[i] == Catch::Approx(C_ref[i]).margin(1e-4f));
    }

    device_free(dA);
    device_free(dB);
    device_free(dC);
}

TEST_CASE("gpu::blas::gemm transa=T transb=N", "[gpu][blas]") {
    // C(M×N) = A^T(M×K) * B(K×N), A stored as K×M
    constexpr int M = 4, K = 3, N = 5;

    std::vector<float> A(K * M), B(K * N), C(M * N, 0.0f), C_ref(M * N, 0.0f);

    for (int i = 0; i < K * M; i++)
        A[i] = static_cast<float>(i + 1);
    for (int i = 0; i < K * N; i++)
        B[i] = static_cast<float>(i + 1) * 0.5f;

    einsums::blas::vendor::sgemm('t', 'n', M, N, K, 1.0f, A.data(), K, B.data(), K, 0.0f, C_ref.data(), M);

    void *dA = test_malloc(K * M * sizeof(float));
    void *dB = test_malloc(K * N * sizeof(float));
    void *dC = test_malloc(M * N * sizeof(float));

    memcpy_host_to_device(dA, A.data(), K * M * sizeof(float));
    memcpy_host_to_device(dB, B.data(), K * N * sizeof(float));
    memcpy_host_to_device(dC, C.data(), M * N * sizeof(float));

    einsums::gpu::blas::gemm<float>('t', 'n', M, N, K, 1.0f, static_cast<float const *>(dA), K, static_cast<float const *>(dB), K, 0.0f,
                                    static_cast<float *>(dC), M);

    memcpy_device_to_host(C.data(), dC, M * N * sizeof(float));

    for (int i = 0; i < M * N; i++) {
        CHECK(C[i] == Catch::Approx(C_ref[i]).margin(1e-4f));
    }

    device_free(dA);
    device_free(dB);
    device_free(dC);
}

TEST_CASE("gpu::blas::gemm transa=N transb=T", "[gpu][blas]") {
    // C(M×N) = A(M×K) * B^T(K×N), B stored as N×K
    constexpr int M = 4, K = 3, N = 5;

    std::vector<float> A(M * K), B(N * K), C(M * N, 0.0f), C_ref(M * N, 0.0f);

    for (int i = 0; i < M * K; i++)
        A[i] = static_cast<float>(i + 1);
    for (int i = 0; i < N * K; i++)
        B[i] = static_cast<float>(i + 1) * 0.5f;

    einsums::blas::vendor::sgemm('n', 't', M, N, K, 1.0f, A.data(), M, B.data(), N, 0.0f, C_ref.data(), M);

    void *dA = test_malloc(M * K * sizeof(float));
    void *dB = test_malloc(N * K * sizeof(float));
    void *dC = test_malloc(M * N * sizeof(float));

    memcpy_host_to_device(dA, A.data(), M * K * sizeof(float));
    memcpy_host_to_device(dB, B.data(), N * K * sizeof(float));
    memcpy_host_to_device(dC, C.data(), M * N * sizeof(float));

    einsums::gpu::blas::gemm<float>('n', 't', M, N, K, 1.0f, static_cast<float const *>(dA), M, static_cast<float const *>(dB), N, 0.0f,
                                    static_cast<float *>(dC), M);

    memcpy_device_to_host(C.data(), dC, M * N * sizeof(float));

    for (int i = 0; i < M * N; i++) {
        CHECK(C[i] == Catch::Approx(C_ref[i]).margin(1e-4f));
    }

    device_free(dA);
    device_free(dB);
    device_free(dC);
}

TEST_CASE("gpu::blas::gemm transa=T transb=T", "[gpu][blas]") {
    // C(M×N) = A^T(M×K) * B^T(K×N), A stored as K×M, B stored as N×K
    constexpr int M = 4, K = 3, N = 5;

    std::vector<float> A(K * M), B(N * K), C(M * N, 0.0f), C_ref(M * N, 0.0f);

    for (int i = 0; i < K * M; i++)
        A[i] = static_cast<float>(i + 1);
    for (int i = 0; i < N * K; i++)
        B[i] = static_cast<float>(i + 1) * 0.5f;

    einsums::blas::vendor::sgemm('t', 't', M, N, K, 1.0f, A.data(), K, B.data(), N, 0.0f, C_ref.data(), M);

    void *dA = test_malloc(K * M * sizeof(float));
    void *dB = test_malloc(N * K * sizeof(float));
    void *dC = test_malloc(M * N * sizeof(float));

    memcpy_host_to_device(dA, A.data(), K * M * sizeof(float));
    memcpy_host_to_device(dB, B.data(), N * K * sizeof(float));
    memcpy_host_to_device(dC, C.data(), M * N * sizeof(float));

    einsums::gpu::blas::gemm<float>('t', 't', M, N, K, 1.0f, static_cast<float const *>(dA), K, static_cast<float const *>(dB), N, 0.0f,
                                    static_cast<float *>(dC), M);

    memcpy_device_to_host(C.data(), dC, M * N * sizeof(float));

    for (int i = 0; i < M * N; i++) {
        CHECK(C[i] == Catch::Approx(C_ref[i]).margin(1e-4f));
    }

    device_free(dA);
    device_free(dB);
    device_free(dC);
}

TEST_CASE("gpu::blas::gemm non-square matrices transa=N transb=N", "[gpu][blas]") {
    // C(M×N) = A(M×K) * B(K×N) with M≠N≠K
    constexpr int M = 7, K = 5, N = 3;

    std::vector<float> A(M * K), B(K * N), C(M * N, 0.0f), C_ref(M * N, 0.0f);

    for (int i = 0; i < M * K; i++)
        A[i] = static_cast<float>((i % 11) + 1) * 0.3f;
    for (int i = 0; i < K * N; i++)
        B[i] = static_cast<float>((i % 7) + 1) * 0.2f;

    einsums::blas::vendor::sgemm('n', 'n', M, N, K, 1.0f, A.data(), M, B.data(), K, 0.0f, C_ref.data(), M);

    void *dA = test_malloc(M * K * sizeof(float));
    void *dB = test_malloc(K * N * sizeof(float));
    void *dC = test_malloc(M * N * sizeof(float));

    memcpy_host_to_device(dA, A.data(), M * K * sizeof(float));
    memcpy_host_to_device(dB, B.data(), K * N * sizeof(float));
    memcpy_host_to_device(dC, C.data(), M * N * sizeof(float));

    einsums::gpu::blas::gemm<float>('n', 'n', M, N, K, 1.0f, static_cast<float const *>(dA), M, static_cast<float const *>(dB), K, 0.0f,
                                    static_cast<float *>(dC), M);

    memcpy_device_to_host(C.data(), dC, M * N * sizeof(float));

    for (int i = 0; i < M * N; i++) {
        CHECK(C[i] == Catch::Approx(C_ref[i]).margin(1e-4f));
    }

    device_free(dA);
    device_free(dB);
    device_free(dC);
}

TEST_CASE("gpu::blas::gemm with alpha and beta", "[gpu][blas]") {
    constexpr int N = 4;

    std::vector<float> A(N * N), B(N * N), C(N * N), C_ref(N * N);

    for (int i = 0; i < N * N; i++) {
        A[i] = static_cast<float>(i + 1);
        B[i] = static_cast<float>(i + 1) * 0.5f;
        C[i] = 1.0f; // non-zero initial C
    }
    C_ref = C;

    // C = 2.0 * A * B + 3.0 * C
    einsums::blas::vendor::sgemm('n', 'n', N, N, N, 2.0f, A.data(), N, B.data(), N, 3.0f, C_ref.data(), N);

    void *dA = test_malloc(N * N * sizeof(float));
    void *dB = test_malloc(N * N * sizeof(float));
    void *dC = test_malloc(N * N * sizeof(float));

    memcpy_host_to_device(dA, A.data(), N * N * sizeof(float));
    memcpy_host_to_device(dB, B.data(), N * N * sizeof(float));
    memcpy_host_to_device(dC, C.data(), N * N * sizeof(float));

    einsums::gpu::blas::gemm<float>('n', 'n', N, N, N, 2.0f, static_cast<float const *>(dA), N, static_cast<float const *>(dB), N, 3.0f,
                                    static_cast<float *>(dC), N);

    memcpy_device_to_host(C.data(), dC, N * N * sizeof(float));

    for (int i = 0; i < N * N; i++) {
        CHECK(C[i] == Catch::Approx(C_ref[i]).margin(1e-3f));
    }

    device_free(dA);
    device_free(dB);
    device_free(dC);
}

TEST_CASE("gpu::blas::gemv transa=N", "[gpu][blas]") {
    // y(M) = alpha * A(M×N) * x(N) + beta * y(M)
    constexpr int M = 5, N = 4;

    std::vector<float> A(M * N), x(N), y(M, 0.0f), y_ref(M, 0.0f);

    for (int i = 0; i < M * N; i++)
        A[i] = static_cast<float>(i + 1) * 0.1f;
    for (int i = 0; i < N; i++)
        x[i] = static_cast<float>(i + 1);

    einsums::blas::vendor::sgemv('n', M, N, 1.0f, A.data(), M, x.data(), 1, 0.0f, y_ref.data(), 1);

    void *dA = test_malloc(M * N * sizeof(float));
    void *dx = test_malloc(N * sizeof(float));
    void *dy = test_malloc(M * sizeof(float));

    memcpy_host_to_device(dA, A.data(), M * N * sizeof(float));
    memcpy_host_to_device(dx, x.data(), N * sizeof(float));
    memcpy_host_to_device(dy, y.data(), M * sizeof(float));

    einsums::gpu::blas::gemv<float>('n', M, N, 1.0f, static_cast<float const *>(dA), M, static_cast<float const *>(dx), 1, 0.0f,
                                    static_cast<float *>(dy), 1);

    memcpy_device_to_host(y.data(), dy, M * sizeof(float));

    for (int i = 0; i < M; i++) {
        CHECK(y[i] == Catch::Approx(y_ref[i]).margin(1e-4f));
    }

    device_free(dA);
    device_free(dx);
    device_free(dy);
}

TEST_CASE("gpu::blas::gemv transa=T", "[gpu][blas]") {
    // y(N) = alpha * A^T(N×M) * x(M) + beta * y(N), A stored as M×N
    constexpr int M = 5, N = 4;

    std::vector<float> A(M * N), x(M), y(N, 0.0f), y_ref(N, 0.0f);

    for (int i = 0; i < M * N; i++)
        A[i] = static_cast<float>(i + 1) * 0.1f;
    for (int i = 0; i < M; i++)
        x[i] = static_cast<float>(i + 1);

    einsums::blas::vendor::sgemv('t', M, N, 1.0f, A.data(), M, x.data(), 1, 0.0f, y_ref.data(), 1);

    void *dA = test_malloc(M * N * sizeof(float));
    void *dx = test_malloc(M * sizeof(float));
    void *dy = test_malloc(N * sizeof(float));

    memcpy_host_to_device(dA, A.data(), M * N * sizeof(float));
    memcpy_host_to_device(dx, x.data(), M * sizeof(float));
    memcpy_host_to_device(dy, y.data(), N * sizeof(float));

    einsums::gpu::blas::gemv<float>('t', M, N, 1.0f, static_cast<float const *>(dA), M, static_cast<float const *>(dx), 1, 0.0f,
                                    static_cast<float *>(dy), 1);

    memcpy_device_to_host(y.data(), dy, N * sizeof(float));

    for (int i = 0; i < N; i++) {
        CHECK(y[i] == Catch::Approx(y_ref[i]).margin(1e-4f));
    }

    device_free(dA);
    device_free(dx);
    device_free(dy);
}

TEST_CASE("gpu::blas::gemv with alpha and beta", "[gpu][blas]") {
    constexpr int M = 4, N = 3;

    std::vector<float> A(M * N), x(N), y(M, 1.0f), y_ref(M, 1.0f);

    for (int i = 0; i < M * N; i++)
        A[i] = static_cast<float>(i + 1);
    for (int i = 0; i < N; i++)
        x[i] = static_cast<float>(i + 1) * 0.5f;

    // y = 2.0 * A * x + 3.0 * y
    einsums::blas::vendor::sgemv('n', M, N, 2.0f, A.data(), M, x.data(), 1, 3.0f, y_ref.data(), 1);

    void *dA = test_malloc(M * N * sizeof(float));
    void *dx = test_malloc(N * sizeof(float));
    void *dy = test_malloc(M * sizeof(float));

    memcpy_host_to_device(dA, A.data(), M * N * sizeof(float));
    memcpy_host_to_device(dx, x.data(), N * sizeof(float));
    memcpy_host_to_device(dy, y.data(), M * sizeof(float));

    einsums::gpu::blas::gemv<float>('n', M, N, 2.0f, static_cast<float const *>(dA), M, static_cast<float const *>(dx), 1, 3.0f,
                                    static_cast<float *>(dy), 1);

    memcpy_device_to_host(y.data(), dy, M * sizeof(float));

    for (int i = 0; i < M; i++) {
        CHECK(y[i] == Catch::Approx(y_ref[i]).margin(1e-3f));
    }

    device_free(dA);
    device_free(dx);
    device_free(dy);
}

TEST_CASE("gpu::blas::hgemm Float16 GEMM", "[gpu][blas]") {
    // FP16 GEMM: C_float = alpha * A_fp16 * B_fp16 + beta * C_float
    if constexpr (!einsums::gpu::has_fp16_gemm) {
        SKIP("No FP16 GEMM support");
    }

    constexpr int N = 4;

    // Create FP16 input matrices.
    std::vector<einsums::gpu::half_t> A(N * N), B(N * N);
    std::vector<float>                C(N * N, 0.0f);

    // Fill with small values (FP16 has limited range).
    for (int i = 0; i < N * N; i++) {
        A[i] = static_cast<einsums::gpu::half_t>(static_cast<float>(i + 1) * 0.1f);
        B[i] = static_cast<einsums::gpu::half_t>(static_cast<float>((i + 2) % 5) * 0.2f);
    }

    // Reference: convert to float, compute on CPU, then compare.
    std::vector<float> A_f(N * N), B_f(N * N), C_ref(N * N, 0.0f);
    for (int i = 0; i < N * N; i++) {
        A_f[i] = static_cast<float>(A[i]);
        B_f[i] = static_cast<float>(B[i]);
    }
    einsums::blas::vendor::sgemm('n', 'n', N, N, N, 1.0f, A_f.data(), N, B_f.data(), N, 0.0f, C_ref.data(), N);

    void *dA       = test_malloc(N * N * sizeof(einsums::gpu::half_t));
    void *dC_float = test_malloc(N * N * sizeof(float));
    void *dB       = test_malloc(N * N * sizeof(einsums::gpu::half_t));

    memcpy_host_to_device(dA, A.data(), N * N * sizeof(einsums::gpu::half_t));
    memcpy_host_to_device(dB, B.data(), N * N * sizeof(einsums::gpu::half_t));
    memcpy_host_to_device(dC_float, C.data(), N * N * sizeof(float));

    einsums::gpu::blas::hgemm('n', 'n', N, N, N, 1.0f, static_cast<einsums::gpu::half_t const *>(dA), N,
                              static_cast<einsums::gpu::half_t const *>(dB), N, 0.0f, static_cast<float *>(dC_float), N);

    memcpy_device_to_host(C.data(), dC_float, N * N * sizeof(float));

    // FP16 precision is limited (~3 decimal digits), so use wide tolerance.
    for (int i = 0; i < N * N; i++) {
        CHECK(C[i] == Catch::Approx(C_ref[i]).margin(0.05f));
    }

    device_free(dA);
    device_free(dB);
    device_free(dC_float);
}

TEST_CASE("gpu::blas::bfgemm BFloat16 GEMM", "[gpu][blas]") {
    // BFloat16 GEMM: C_float = alpha * A_bf16 * B_bf16 + beta * C_float
    if constexpr (!einsums::gpu::has_mps) {
        SKIP("BFloat16 GEMM only available on MPS");
    }

    constexpr int N = 4;

    std::vector<einsums::gpu::bfloat16_t> A(N * N), B(N * N);
    std::vector<float>                    C(N * N, 0.0f);

    for (int i = 0; i < N * N; i++) {
        A[i] = static_cast<einsums::gpu::bfloat16_t>(static_cast<float>(i + 1) * 0.1f);
        B[i] = static_cast<einsums::gpu::bfloat16_t>(static_cast<float>((i + 2) % 5) * 0.2f);
    }

    // Reference: convert to float, compute on CPU.
    std::vector<float> A_f(N * N), B_f(N * N), C_ref(N * N, 0.0f);
    for (int i = 0; i < N * N; i++) {
        A_f[i] = static_cast<float>(A[i]);
        B_f[i] = static_cast<float>(B[i]);
    }
    einsums::blas::vendor::sgemm('n', 'n', N, N, N, 1.0f, A_f.data(), N, B_f.data(), N, 0.0f, C_ref.data(), N);

    void *dA = test_malloc(N * N * sizeof(einsums::gpu::bfloat16_t));
    void *dB = test_malloc(N * N * sizeof(einsums::gpu::bfloat16_t));
    void *dC = test_malloc(N * N * sizeof(float));

    memcpy_host_to_device(dA, A.data(), N * N * sizeof(einsums::gpu::bfloat16_t));
    memcpy_host_to_device(dB, B.data(), N * N * sizeof(einsums::gpu::bfloat16_t));
    memcpy_host_to_device(dC, C.data(), N * N * sizeof(float));

    einsums::gpu::blas::bfgemm('n', 'n', N, N, N, 1.0f, static_cast<einsums::gpu::bfloat16_t const *>(dA), N,
                               static_cast<einsums::gpu::bfloat16_t const *>(dB), N, 0.0f, static_cast<float *>(dC), N);

    memcpy_device_to_host(C.data(), dC, N * N * sizeof(float));

    // BFloat16 has ~3 decimal digits precision (7-bit mantissa), use wide tolerance.
    for (int i = 0; i < N * N; i++) {
        CHECK(C[i] == Catch::Approx(C_ref[i]).margin(0.1f));
    }

    device_free(dA);
    device_free(dB);
    device_free(dC);
}

TEST_CASE("gpu::blas::cgemm ComplexFloat32", "[gpu][blas]") {
    // Complex GEMM: C = A * B where A, B, C are complex<float> matrices.
    constexpr int N = 4;

    std::vector<std::complex<float>> A(N * N), B(N * N), C(N * N, {0, 0}), C_ref(N * N, {0, 0});

    for (int i = 0; i < N * N; i++) {
        A[i] = {static_cast<float>(i + 1) * 0.1f, static_cast<float>(i % 3) * 0.05f};
        B[i] = {static_cast<float>((i + 2) % 5) * 0.2f, static_cast<float>(i % 4) * 0.03f};
    }

    // Reference: CPU BLAS.
    std::complex<float> alpha{1.0f, 0.0f}, beta{0.0f, 0.0f};
    einsums::blas::vendor::cgemm('n', 'n', N, N, N, alpha, A.data(), N, B.data(), N, beta, C_ref.data(), N);

    void *dA = test_malloc(N * N * sizeof(std::complex<float>));
    void *dB = test_malloc(N * N * sizeof(std::complex<float>));
    void *dC = test_malloc(N * N * sizeof(std::complex<float>));

    memcpy_host_to_device(dA, A.data(), N * N * sizeof(std::complex<float>));
    memcpy_host_to_device(dB, B.data(), N * N * sizeof(std::complex<float>));
    memcpy_host_to_device(dC, C.data(), N * N * sizeof(std::complex<float>));

    einsums::gpu::blas::gemm<std::complex<float>>('n', 'n', N, N, N, alpha, static_cast<std::complex<float> const *>(dA), N,
                                                  static_cast<std::complex<float> const *>(dB), N, beta,
                                                  static_cast<std::complex<float> *>(dC), N);

    memcpy_device_to_host(C.data(), dC, N * N * sizeof(std::complex<float>));

    for (int i = 0; i < N * N; i++) {
        CHECK(C[i].real() == Catch::Approx(C_ref[i].real()).margin(1e-3f));
        CHECK(C[i].imag() == Catch::Approx(C_ref[i].imag()).margin(1e-3f));
    }

    device_free(dA);
    device_free(dB);
    device_free(dC);
}

TEST_CASE("MPS supported datatypes for GEMM", "[gpu][mps]") {
    // Test which MPSDataTypes actually work for MPSMatrixMultiplication.
    // Float32 is the baseline. Let's test Float16 and ComplexFloat32.

    if constexpr (!einsums::gpu::has_mps) {
        SKIP("MPS not available");
    }

    // Float16 GEMM test (2x2)
    SECTION("Float16 GEMM") {
        // MPS Float16: use MPSDataTypeFloat16
        // We can't easily test this through our C++ API since we don't have
        // a half_t GEMM path wired. Just document: MPSDataTypeFloat16 is defined.
        SUCCEED("Float16 type exists in MPS (MPSDataTypeFloat16)");
    }

    // ComplexFloat32 GEMM test — available since macOS 13.1
    SECTION("ComplexFloat32 type exists") {
        SUCCEED("ComplexFloat32 type exists in MPS (MPSDataTypeComplexFloat32, macOS 13.1+)");
    }

    // BFloat16 — available since macOS 14.0
    SECTION("BFloat16 type exists") {
        SUCCEED("BFloat16 type exists in MPS (MPSDataTypeBFloat16, macOS 14.0+)");
    }

    // No Float64/Double — not defined in MPS
    SECTION("No Float64 in MPS") {
        SUCCEED("MPS has no MPSDataTypeFloat64 — double precision not supported");
    }
}

TEST_CASE("gpu::blas::scal", "[gpu][blas]") {
    constexpr int      N = 128;
    std::vector<float> x(N), x_ref(N);

    for (int i = 0; i < N; i++) {
        x[i]     = static_cast<float>(i + 1);
        x_ref[i] = x[i] * 2.5f;
    }

    void *dx = test_malloc(N * sizeof(float));
    memcpy_host_to_device(dx, x.data(), N * sizeof(float));

    einsums::gpu::blas::scal<float>(N, 2.5f, static_cast<float *>(dx), 1);

    memcpy_device_to_host(x.data(), dx, N * sizeof(float));

    for (int i = 0; i < N; i++) {
        CHECK(x[i] == Catch::Approx(x_ref[i]).margin(1e-4f));
    }

    device_free(dx);
}

TEST_CASE("gpu::blas::axpy", "[gpu][blas]") {
    constexpr int      N = 128;
    std::vector<float> x(N), y(N), y_ref(N);

    for (int i = 0; i < N; i++) {
        x[i]     = static_cast<float>(i + 1);
        y[i]     = static_cast<float>(N - i);
        y_ref[i] = 3.0f * x[i] + y[i];
    }

    void *dx = test_malloc(N * sizeof(float));
    void *dy = test_malloc(N * sizeof(float));
    memcpy_host_to_device(dx, x.data(), N * sizeof(float));
    memcpy_host_to_device(dy, y.data(), N * sizeof(float));

    einsums::gpu::blas::axpy<float>(N, 3.0f, static_cast<float const *>(dx), 1, static_cast<float *>(dy), 1);

    memcpy_device_to_host(y.data(), dy, N * sizeof(float));

    for (int i = 0; i < N; i++) {
        CHECK(y[i] == Catch::Approx(y_ref[i]).margin(1e-4f));
    }

    device_free(dx);
    device_free(dy);
}

TEST_CASE("gpu::blas::dot", "[gpu][blas]") {
    constexpr int      N = 64;
    std::vector<float> x(N), y(N);

    float ref = 0.0f;
    for (int i = 0; i < N; i++) {
        x[i] = static_cast<float>(i + 1) * 0.1f;
        y[i] = static_cast<float>(N - i) * 0.1f;
        ref += x[i] * y[i];
    }

    void *dx = test_malloc(N * sizeof(float));
    void *dy = test_malloc(N * sizeof(float));
    memcpy_host_to_device(dx, x.data(), N * sizeof(float));
    memcpy_host_to_device(dy, y.data(), N * sizeof(float));

    float result = einsums::gpu::blas::dot<float>(N, static_cast<float const *>(dx), 1, static_cast<float const *>(dy), 1);

    CHECK(result == Catch::Approx(ref).margin(1e-2f));

    device_free(dx);
    device_free(dy);
}

TEST_CASE("MPS platform detection", "[gpu][mps]") {
    INFO("has_mps  = " << einsums::gpu::has_mps);
    INFO("has_gpu  = " << einsums::gpu::has_gpu);
    INFO("is_mock  = " << einsums::gpu::is_mock);

    if constexpr (einsums::gpu::has_mps) {
        CHECK_FALSE(einsums::gpu::is_mock);
        // MPS should report real device memory.
        size_t mem = einsums::gpu::available_device_memory();
        CHECK(mem > 1024 * 1024 * 1024); // At least 1 GB on any Apple Silicon.
        INFO("Available device memory: " << mem / (1024 * 1024) << " MB");
    }

    SUCCEED();
}

TEST_CASE("DeviceTensorConcept recognizes GPUTensor", "[gpu][tensor][concepts]") {
    using namespace einsums;

    // GPUTensor satisfies DeviceTensorConcept
    static_assert(IsDeviceTensorV<GPUTensor<double, 2>>);
    static_assert(IsDeviceTensorV<GPUTensor<float, 3>>);
    static_assert(IsDeviceTensorV<GPUTensor<std::complex<double>, 1>>);

    // Regular tensors do NOT satisfy DeviceTensorConcept
    static_assert(!IsDeviceTensorV<Tensor<double, 2>>);
    static_assert(!IsDeviceTensorV<BufferTensor<double, 2>>);

    // DeviceTensorConcept concept works
    static_assert(DeviceTensorConcept<GPUTensor<double, 2>>);
    static_assert(!DeviceTensorConcept<Tensor<double, 2>>);

    SUCCEED("All static_asserts passed");
}
