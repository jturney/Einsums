//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

/// @file ConcurrentPermute.cpp
/// @brief Two permutes on DISJOINT tensors must not interfere.
///
/// Every parallel graph executor schedules independent nodes concurrently, so
/// this is the invariant those executors rest on: a permute whose operands are
/// touched by nobody else must produce the same values whether it ran alone or
/// alongside another one. No graph dependency can substitute for it - the
/// tensors here genuinely do not overlap, so there is no edge to add.
///
/// This is a regression guard for a real failure mode rather than a
/// hypothetical. HPTT plan clones share their ``_masterPlan`` through a
/// shared_ptr, so whether two concurrent permutes are safe depends on what
/// that shared tree does during execute. Handing HPTT a single-thread plan
/// (which is what a size gate on the plan cache's thread count does, to skip
/// the fork on small operands) made this fail roughly half the time, while the
/// default multi-thread plans pass - the nested parallel region was hiding it.
/// So a change to how plans are built or shared can flip this, and the failure
/// is silent: wrong values, no crash, only under concurrency.

#include <Einsums/TensorAlgebra.hpp>
#include <Einsums/TensorUtilities/CreateRandomTensor.hpp>
#include <Einsums/TensorUtilities/CreateZeroTensor.hpp>

#include <cmath>

#include <Einsums/Testing.hpp>

using namespace einsums;
using namespace einsums::index;

TEST_CASE("permute - concurrent permutes on disjoint tensors agree with serial", "[TensorAlgebra][permute][concurrency]") {
    constexpr size_t N     = 3; // small on purpose: the per-call overhead, not the copy, is what varies
    constexpr int    kReps = 500;

    auto A0 = create_random_tensor<double>(std::string("A0"), N, N);
    auto A1 = create_random_tensor<double>(std::string("A1"), N, N);

    // Serial references. The second uses the prefactor form, which is the
    // accumulating shape a graph emits after control flow.
    auto R0 = create_zero_tensor<double>(std::string("R0"), N, N);
    auto R1 = create_zero_tensor<double>(std::string("R1"), N, N);
    tensor_algebra::permute(Indices{j, i}, &R0, Indices{i, j}, A0);
    tensor_algebra::permute(0.0, Indices{j, i}, &R1, 1.0, Indices{i, j}, A1);

    int mismatches = 0;
    for (int r = 0; r < kReps; r++) {
        auto B0 = create_zero_tensor<double>(std::string("B0"), N, N);
        auto B1 = create_zero_tensor<double>(std::string("B1"), N, N);

#pragma omp parallel for schedule(dynamic)
        for (int t = 0; t < 2; t++) {
            if (t == 0) {
                tensor_algebra::permute(Indices{j, i}, &B0, Indices{i, j}, A0);
            } else {
                tensor_algebra::permute(0.0, Indices{j, i}, &B1, 1.0, Indices{i, j}, A1);
            }
        }

        for (size_t a = 0; a < N; a++) {
            for (size_t b = 0; b < N; b++) {
                if (std::abs(B0(a, b) - R0(a, b)) > 1e-12 || std::abs(B1(a, b) - R1(a, b)) > 1e-12) {
                    mismatches++;
                    a = N; // stop scanning this rep
                    break;
                }
            }
        }
    }

    INFO("concurrent permutes disagreed with their serial results in " << mismatches << " of " << kReps << " repetitions");
    REQUIRE(mismatches == 0);
}
