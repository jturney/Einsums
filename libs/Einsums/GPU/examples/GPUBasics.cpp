//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

/// @file GPUBasics.cpp
/// @brief Demonstrates the GPU abstraction layer basics.
///
/// Shows how to:
///   - Detect the GPU backend at compile time
///   - Allocate and free device memory
///   - Transfer data between host and device
///   - Run a GEMM on the GPU via gpu::blas::gemm
///   - Query available device memory

#include <Einsums/BLASVendor/Vendor.hpp>
#include <Einsums/GPU/BLAS.hpp>
#include <Einsums/GPU/Platform.hpp>
#include <Einsums/GPU/Runtime.hpp>
#include <Einsums/Runtime.hpp>

#include <cmath>
#include <iostream>
#include <vector>

namespace gpu = einsums::gpu;

int einsums_main() {
    // ── 1. Platform detection ───────────────────────────────────────────────
    std::cout << "GPU Backend Detection:\n";
    std::cout << "  has_cuda:           " << gpu::has_cuda << "\n";
    std::cout << "  has_hip:            " << gpu::has_hip << "\n";
    std::cout << "  has_mps:            " << gpu::has_mps << "\n";
    std::cout << "  has_gpu:            " << gpu::has_gpu << "\n";
    std::cout << "  is_mock:            " << gpu::is_mock << "\n";
    std::cout << "  has_unified_memory: " << gpu::has_unified_memory << "\n";
    std::cout << "  has_fp16_gemm:      " << gpu::has_fp16_gemm << "\n";

    size_t mem = gpu::available_device_memory();
    std::cout << "  Device memory:      " << mem / (1024 * 1024) << " MB\n\n";

    // ── 2. Device memory operations ─────────────────────────────────────────
    constexpr int      N = 64;
    std::vector<float> hostA(N * N), hostB(N * N), hostC(N * N, 0.0f);

    // Fill with test data
    for (int i = 0; i < N * N; i++) {
        hostA[i] = static_cast<float>(i % 17) * 0.1f;
        hostB[i] = static_cast<float>(i % 13) * 0.1f;
    }

    // Allocate on device
    void *devA = gpu::device_malloc(N * N * sizeof(float)).value();
    void *devB = gpu::device_malloc(N * N * sizeof(float)).value();
    void *devC = gpu::device_malloc(N * N * sizeof(float)).value();

    // Transfer host -> device
    gpu::memcpy_host_to_device(devA, hostA.data(), N * N * sizeof(float));
    gpu::memcpy_host_to_device(devB, hostB.data(), N * N * sizeof(float));
    gpu::memcpy_host_to_device(devC, hostC.data(), N * N * sizeof(float));

    // ── 3. GPU GEMM ────────────────────────────────────────────────────────
    // C = 1.0 * A * B + 0.0 * C
    gpu::blas::gemm<float>('n', 'n', N, N, N, 1.0f, static_cast<float const *>(devA), N, static_cast<float const *>(devB), N, 0.0f,
                           static_cast<float *>(devC), N);

    // Transfer result back
    gpu::memcpy_device_to_host(hostC.data(), devC, N * N * sizeof(float));

    // ── 4. Verify against CPU reference ─────────────────────────────────────
    std::vector<float> refC(N * N, 0.0f);
    einsums::blas::vendor::sgemm('n', 'n', N, N, N, 1.0f, hostA.data(), N, hostB.data(), N, 0.0f, refC.data(), N);

    float max_err = 0.0f;
    for (int i = 0; i < N * N; i++) {
        max_err = std::max(max_err, std::abs(hostC[i] - refC[i]));
    }
    std::cout << "GEMM Verification:\n";
    std::cout << "  Matrix size:  " << N << "x" << N << "\n";
    std::cout << "  Max error:    " << max_err << "\n";
    std::cout << "  Status:       " << (max_err < 1e-4f ? "PASSED" : "FAILED") << "\n\n";

    // ── 5. Cleanup ──────────────────────────────────────────────────────────
    gpu::device_free(devA);
    gpu::device_free(devB);
    gpu::device_free(devC);

    return 0;
}

int main(int argc, char **argv) {
    return einsums::start(einsums_main, argc, argv);
}
