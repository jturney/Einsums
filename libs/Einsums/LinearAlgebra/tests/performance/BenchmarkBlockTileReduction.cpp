//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

/// @file BenchmarkBlockTileReduction.cpp
/// @brief Does a reduction over blocks or tiles pay for the team it launches?
///
/// The elementwise kernels consult @ref einsums::hardware::omp_min_parallel_elements
/// before opening a parallel region, because an unconditional one costs far
/// more than a short loop. The reductions over a block tensor's blocks and a
/// tiled tensor's tiles do not consult it: they open a region per call whatever
/// the structure holds. A block tensor with a handful of small blocks therefore
/// forks a team to add up a few hundred flops, which is the same shape of
/// mistake the gate was written for.
///
/// This measures whether that is true and where the break-even is, so the gate
/// can be applied on evidence rather than by analogy.
///
/// Reading this: the interesting column is the ratio against one thread, which
/// the harness cannot vary from inside a process - run the case twice,
///
///     OMP_NUM_THREADS=1  ./BenchmarkBlockTileReduction_test
///     OMP_NUM_THREADS=10 ./BenchmarkBlockTileReduction_test
///
/// and compare. A row where the threaded time is LARGER is a region that should
/// have been declined; the block count and block size at which the two cross is
/// where the gate belongs. A row whose time barely moves with the block size is
/// reporting fork cost rather than work.

#include <Einsums/LinearAlgebra.hpp>
#include <Einsums/Performance.hpp>
#include <Einsums/Tensor/BlockTensor.hpp>
#include <Einsums/Tensor/TiledTensor.hpp>
#include <Einsums/TensorUtilities/CreateRandomTensor.hpp>

#include <fmt/format.h>

#include <string>
#include <vector>

#include <Einsums/Testing.hpp>

using namespace einsums;
using namespace einsums::performance;

namespace {

constexpr int kReps = 500;

void row(std::string const &op, size_t blocks, size_t extent, TimingStats const &t) {
    // Elements actually touched, which is what the work is proportional to and
    // what a size gate would be phrased in.
    size_t const elements = blocks * extent * extent;
    fmt::println("[BlockTileReduction] {:<18} blocks={:4d} extent={:4d} elements={:8d}  {:9.3f} us  ({:8.2f} ns/element)", op, blocks,
                 extent, elements, t.avg, 1000.0 * t.avg / static_cast<double>(elements));
    std::string const key = fmt::format("BlockTileReduction {} b{} e{}", op, blocks, extent);
    publish_benchmark_result(key.c_str(), "t_call", static_cast<int>(elements), t);
}

} // namespace

EINSUMS_TEST_CASE("Bench BlockTileReduction: block dot", "[LinearAlgebra][BlockTileReduction][benchmark]") {
    LabeledSection0();

    // Few small blocks is the case the gate exists for; many large blocks is
    // the case that genuinely wants the team. The break-even is between them.
    for (auto [blocks, extent] :
         std::vector<std::pair<size_t, size_t>>{{2, 8}, {4, 8}, {8, 8}, {32, 8}, {4, 32}, {32, 32}, {4, 128}, {32, 128}}) {
        BlockTensor<double, 2> A("A", std::vector<size_t>(blocks, extent));
        BlockTensor<double, 2> B("B", std::vector<size_t>(blocks, extent));
        for (size_t i = 0; i < blocks; i++) {
            A[i] = create_random_tensor<double>(std::string("a"), extent, extent);
            B[i] = create_random_tensor<double>(std::string("b"), extent, extent);
        }

        row("block dot", blocks, extent,
            time_us(
                "dot", [&]() { [[maybe_unused]] auto out = linear_algebra::dot(A, B); }, kReps));
    }
}

EINSUMS_TEST_CASE("Bench BlockTileReduction: tiled dot", "[LinearAlgebra][BlockTileReduction][benchmark]") {
    LabeledSection0();

    // A tiled tensor's reduction walks the whole grid, so the interesting axis
    // is the grid size rather than the tile size: a fine grid of small tiles is
    // exactly a per-iteration workload's amplitude container.
    for (auto [grid, extent] : std::vector<std::pair<size_t, size_t>>{{2, 8}, {4, 8}, {8, 8}, {4, 32}, {8, 32}, {4, 128}}) {
        std::vector<size_t> const tiling(grid, extent);
        TiledTensor<double, 2>    A("A", tiling, tiling);
        TiledTensor<double, 2>    B("B", tiling, tiling);
        for (size_t i = 0; i < grid; i++) {
            for (size_t j = 0; j < grid; j++) {
                A.tile(i, j) = create_random_tensor<double>(std::string("a"), extent, extent);
                B.tile(i, j) = create_random_tensor<double>(std::string("b"), extent, extent);
            }
        }

        row("tiled dot", grid * grid, extent,
            time_us(
                "dot", [&]() { [[maybe_unused]] auto out = linear_algebra::dot(A, B); }, kReps));
    }
}
