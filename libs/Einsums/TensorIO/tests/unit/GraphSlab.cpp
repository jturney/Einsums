//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

/// @file GraphSlab.cpp
/// @brief Unit tests for tensor_io::Slab + read_slice_etn / write_slice_etn.
///
/// Exercises the graph-aware slab APIs end-to-end: capture a graph that
/// reads a slab, transforms it, writes it back, then run the graph and
/// inspect the file. Covers both the static-rank and runtime-rank
/// dispatcher branches, plus the read-transform-write-back loop pattern
/// that motivated the design, where a single graph node drives multiple
/// slabs by mutating Slab::ranges between executor invocations.

#include <Einsums/ComputeGraph/CaptureContext.hpp>
#include <Einsums/ComputeGraph/Executor.hpp>
#include <Einsums/ComputeGraph/Graph.hpp>
#include <Einsums/Tensor/RuntimeTensor.hpp>
#include <Einsums/Tensor/Tensor.hpp>
#include <Einsums/TensorIO/GraphIO.hpp>
#include <Einsums/TensorIO/TensorFile.hpp>
#include <Einsums/TensorUtilities/CreateZeroTensor.hpp>

#include <cstdio>
#include <filesystem>

#include <Einsums/Testing.hpp>

using namespace einsums;
using namespace einsums::tensor_io;

namespace {
struct TempFile {
    std::string path;
    explicit TempFile(std::string name)
        : path((std::filesystem::temp_directory_path() / ("einsums_graph_slab_" + std::move(name))).string()) {}
    ~TempFile() { std::remove(path.c_str()); }
};

/// A value of element type @p T built from @p x; complex types also get an imaginary part, so a
/// round trip that dropped or swapped halves of an element would show.
template <typename T>
T value(double x) {
    if constexpr (IsComplexV<T>) {
        using R = RemoveComplexT<T>;
        return T{static_cast<R>(x), static_cast<R>(-0.5 * x)};
    } else {
        return static_cast<T>(x);
    }
}
} // namespace

// ── Static-rank: read_slice_etn / write_slice_etn round-trip ────────────

TEMPLATE_LIST_TEST_CASE("read_slice_etn pulls a slab into a static-rank tensor", "[TensorIO][GraphIO][slice]", testing::AllScalarTypes) {
    TempFile const tmp("static_read.etn");

    // Seed the file: 4×4 with a unique pattern (i*10 + j).
    auto base = create_zero_tensor<TestType>("A", 4, 4);
    for (size_t i = 0; i < 4; ++i) {
        for (size_t j = 0; j < 4; ++j) {
            base(i, j) = value<TestType>(10.0 * i + j);
        }
    }
    {
        TensorFile out(tmp.path, TensorFile::Mode::Write);
        out.write("A", base);
    }

    Tensor<TestType, 2> block("blk", 2, 2);
    Slab const          slab{{{1, 3}, {1, 3}}};

    compute_graph::Graph g;
    {
        compute_graph::CaptureGuard const guard(g);
        read_slice_etn(tmp.path, "A", slab, &block);
    }

    // graph.execute() drives the default sequential executor.
    g.execute();

    REQUIRE(block(0, 0) == value<TestType>(11.0));
    REQUIRE(block(0, 1) == value<TestType>(12.0));
    REQUIRE(block(1, 0) == value<TestType>(21.0));
    REQUIRE(block(1, 1) == value<TestType>(22.0));
}

TEMPLATE_LIST_TEST_CASE("write_slice_etn patches a slab via the graph", "[TensorIO][GraphIO][slice]", testing::AllScalarTypes) {
    TempFile const tmp("static_write.etn");

    // Reserve a 4×4 zero tensor.
    auto base = create_zero_tensor<TestType>("A", 4, 4);
    {
        TensorFile out(tmp.path, TensorFile::Mode::Write);
        out.write("A", base);
    }

    // Build a 2×2 patch of 7's.
    Tensor<TestType, 2> patch("p", 2, 2);
    for (size_t i = 0; i < 2; ++i) {
        for (size_t j = 0; j < 2; ++j) {
            patch(i, j) = value<TestType>(7.0);
        }
    }

    Slab const slab{{{1, 3}, {1, 3}}};

    compute_graph::Graph g;
    {
        compute_graph::CaptureGuard const guard(g);
        write_slice_etn(tmp.path, "A", slab, &patch);
    }

    // graph.execute() drives the default sequential executor.
    g.execute();

    Tensor<TestType, 2> rt("rt", 4, 4);
    {
        TensorFile in(tmp.path, TensorFile::Mode::Read);
        in.read("A", rt);
    }
    for (size_t i = 0; i < 4; ++i) {
        for (size_t j = 0; j < 4; ++j) {
            TestType const expected = (i >= 1 && i < 3 && j >= 1 && j < 3) ? value<TestType>(7.0) : TestType{0};
            REQUIRE(rt(i, j) == expected);
        }
    }
}

// ── The motivating pattern: read-transform-write-back loop ──────────────

TEMPLATE_LIST_TEST_CASE("Slab loop: read→transform→write-back tiles via the graph", "[TensorIO][GraphIO][slice][loop]",
                        testing::AllScalarTypes) {
    TempFile tmp("loop.etn");

    // Seed file with arange(16) reshaped to 4×4.
    Tensor<TestType, 2> seed("A", 4, 4);
    for (size_t i = 0; i < 4; ++i) {
        for (size_t j = 0; j < 4; ++j) {
            seed(i, j) = value<TestType>(4.0 * i + j);
        }
    }
    {
        TensorFile out(tmp.path, TensorFile::Mode::Write);
        out.write("A", seed);
    }

    Tensor<TestType, 2> block("blk", 2, 2);
    Slab                slab{{{0, 2}, {0, 2}}}; // mutated each iteration

    // Build the graph once: read this slab, the user transforms in
    // place, then write back. The driver loop mutates `slab.ranges` and
    // re-runs the graph for every 2x2 tile.
    compute_graph::Graph g;
    {
        compute_graph::CaptureGuard const guard(g);
        read_slice_etn(tmp.path, "A", slab, &block);
        // Compute step is implicit between executor runs; see loop body.
        write_slice_etn(tmp.path, "A", slab, &block);
    }

    // graph.execute() drives the default sequential executor.

    auto run_tile = [&](size_t bi, size_t bj) {
        slab.ranges = {{bi * 2, bi * 2 + 2}, {bj * 2, bj * 2 + 2}};
        g.execute();
        // After read_slice_etn finishes and before write_slice_etn fires
        // we would want a compute step, but SequentialExecutor runs nodes
        // in topo order in one pass, so we transform between graph runs
        // instead. The test exercises the surface; in practice the user
        // would record a compute node between the read and the write, or
        // use Graph::add_loop in Step 4. Here we just multiply by 10 to
        // prove the round-trip works.
        for (size_t i = 0; i < 2; ++i) {
            for (size_t j = 0; j < 2; ++j) {
                block(i, j) *= TestType{10};
            }
        }
        // Re-run only the write half. Easiest: a dedicated mini-graph.
        compute_graph::Graph g_write;
        {
            compute_graph::CaptureGuard const guard(g_write);
            write_slice_etn(tmp.path, "A", slab, &block);
        }
        g_write.execute();
    };

    for (size_t bi = 0; bi < 2; ++bi) {
        for (size_t bj = 0; bj < 2; ++bj) {
            run_tile(bi, bj);
        }
    }

    // Verify every tile got multiplied by 10.
    Tensor<TestType, 2> rt("rt", 4, 4);
    {
        TensorFile in(tmp.path, TensorFile::Mode::Read);
        in.read("A", rt);
    }
    for (size_t i = 0; i < 4; ++i) {
        for (size_t j = 0; j < 4; ++j) {
            REQUIRE(rt(i, j) == value<TestType>(4.0 * i + j) * TestType{10});
        }
    }
}

// ── Runtime-rank dispatcher branch ──────────────────────────────────────

TEMPLATE_LIST_TEST_CASE("read_slice_etn / write_slice_etn dispatch on RuntimeTensor", "[TensorIO][GraphIO][slice][runtime]",
                        testing::AllScalarTypes) {
    TempFile const tmp("rt_slab.etn");

    // Seed file.
    GeneralRuntimeTensor<TestType, std::allocator<TestType>> seed("A", std::vector<size_t>{4, 4});
    for (size_t i = 0; i < seed.size(); ++i) {
        seed.data()[i] = value<TestType>(static_cast<double>(i));
    }
    {
        TensorFile out(tmp.path, TensorFile::Mode::Write);
        out.write("A", seed);
    }

    GeneralRuntimeTensor<TestType, std::allocator<TestType>> block("blk", std::vector<size_t>{2, 2});
    Slab const                                               slab{{{1, 3}, {1, 3}}};

    compute_graph::Graph g_read;
    {
        compute_graph::CaptureGuard const guard(g_read);
        read_slice_etn(tmp.path, "A", slab, &block);
    }
    // graph.execute() drives the default sequential executor.
    g_read.execute();

    // Mutate and write back.
    block.set_all(value<TestType>(99.0));
    compute_graph::Graph g_write;
    {
        compute_graph::CaptureGuard const guard(g_write);
        write_slice_etn(tmp.path, "A", slab, &block);
    }
    g_write.execute();

    // Verify.
    GeneralRuntimeTensor<TestType, std::allocator<TestType>> rt("rt", std::vector<size_t>{4, 4});
    {
        TensorFile in(tmp.path, TensorFile::Mode::Read);
        in.read("A", rt);
    }
    for (size_t i = 1; i < 3; ++i) {
        for (size_t j = 1; j < 3; ++j) {
            REQUIRE(rt.data()[j * 4 + i] == value<TestType>(99.0)); // column-major
        }
    }
}

// ── Element type mismatch ──────────────────────────────────────────────

// The slab executors call the same checked TensorFile methods, so a block of the wrong element
// type is refused at replay instead of reading bytes of another width into it.
TEST_CASE("read_slice_etn / write_slice_etn reject a block of another dtype", "[TensorIO][GraphIO][slice][Error]") {
    TempFile const tmp("dtype.etn");

    auto base = create_zero_tensor<std::complex<double>>("A", 4, 4);
    {
        TensorFile out(tmp.path, TensorFile::Mode::Write);
        out.write("A", base);
    }

    Slab const slab{{{1, 3}, {1, 3}}};

    Tensor<float, 2> block("blk", 2, 2);
    block.zero();
    GeneralRuntimeTensor<double, std::allocator<double>> rt_block("rt_blk", std::vector<size_t>{2, 2});
    rt_block.zero();

    // Outside capture the call runs at once.
    REQUIRE_THROWS_AS(read_slice_etn(tmp.path, "A", slab, &block), std::invalid_argument);
    REQUIRE_THROWS_AS(read_slice_etn(tmp.path, "A", slab, &rt_block), std::invalid_argument);
    REQUIRE_THROWS_AS(write_slice_etn(tmp.path, "A", slab, &block), std::invalid_argument);
    REQUIRE_THROWS_AS(write_slice_etn(tmp.path, "A", slab, &rt_block), std::invalid_argument);

    // Captured, the node's executor throws at replay.
    compute_graph::Graph g_read;
    {
        compute_graph::CaptureGuard const guard(g_read);
        read_slice_etn(tmp.path, "A", slab, &block);
    }
    REQUIRE_THROWS_AS(g_read.execute(), std::invalid_argument);

    compute_graph::Graph g_write;
    {
        compute_graph::CaptureGuard const guard(g_write);
        write_slice_etn(tmp.path, "A", slab, &rt_block);
    }
    REQUIRE_THROWS_AS(g_write.execute(), std::invalid_argument);

    for (size_t i = 0; i < block.size(); ++i) {
        CHECK(block.data()[i] == 0.0F);
    }

    // The stored entry is unchanged.
    auto back = create_zero_tensor<std::complex<double>>("back", 4, 4);
    {
        TensorFile in(tmp.path, TensorFile::Mode::Read);
        in.read("A", back);
    }
    for (size_t i = 0; i < back.size(); ++i) {
        CHECK(back.data()[i] == std::complex<double>{0.0, 0.0});
    }
}
