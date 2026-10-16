//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

/// @file SlabStreaming.cpp
/// @brief A backed tensor read a slab at a time, and never loaded whole.
///
/// The backing here serves slabs out of a tensor held in memory and records every range it is
/// asked for, so a case can assert what was READ, not only what came out: a stream that quietly
/// fell back to the whole load would produce the right numbers and fail here. Expected values come
/// from reference_einsum over the same tensor.

#include <Einsums/ComputeGraph.hpp>
// Not reached through the umbrella header, which does not carry these passes.
#include <Einsums/ComputeGraph/Passes/AxisTiling.hpp>
#include <Einsums/ComputeGraph/Passes/Materialization.hpp>
#include <Einsums/ComputeGraph/Passes/SlabStreaming.hpp>
#include <Einsums/ComputeGraph/TensorBacking.hpp>
#include <Einsums/Options/Get.hpp>
#include <Einsums/Tensor/RuntimeTensor.hpp>
#include <Einsums/TensorBase/Options.hpp>
#include <Einsums/Testing/ReferenceEinsum.hpp>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include <Einsums/Testing.hpp>

using namespace einsums;
namespace cg = einsums::compute_graph;

namespace {

RuntimeTensor<double> filled(std::string name, std::vector<std::size_t> dims, unsigned seed) {
    RuntimeTensor<double> out(std::move(name), std::move(dims));
    for (std::size_t i = 0; i < out.size(); ++i) {
        seed          = (seed * 1103515245U) + 12345U;
        out.data()[i] = (static_cast<double>((seed >> 8U) % 2048U) / 1024.0) - 1.0;
    }
    return out;
}

using Ranges = std::vector<std::pair<std::size_t, std::size_t>>;

/// A backing over a tensor in memory that keeps a log of every slab it served.
struct MemoryStore {
    RuntimeTensor<double> source;
    std::vector<Ranges>   reads;

    explicit MemoryStore(RuntimeTensor<double> tensor) : source(std::move(tensor)) {}

    cg::TensorBacking backing() {
        cg::TensorBacking out;
        out.kind      = "slab_test.memory";
        out.location  = {{"name", source.name()}};
        out.dims      = source.dims();
        out.dtype     = packed_gemm::ScalarType::Float64;
        out.read_slab = [this](void *destination, Ranges const &ranges) {
            reads.push_back(ranges);
            auto                    *dst  = static_cast<double *>(destination);
            std::size_t const        rank = ranges.size();
            std::vector<std::size_t> index(rank);
            for (std::size_t d = 0; d < rank; ++d) {
                index[d] = ranges[d].first;
            }
            std::size_t count = 1;
            for (auto const &[lo, hi] : ranges) {
                count *= hi - lo;
            }
            // Column-major over the slab, which is the order a backing promises.
            for (std::size_t n = 0; n < count; ++n) {
                // By the source's own strides, which follow whatever layout it was built in.
                std::size_t offset = 0;
                for (std::size_t d = 0; d < rank; ++d) {
                    offset += index[d] * source.stride(static_cast<int>(d));
                }
                dst[n] = source.data()[offset];
                for (std::size_t d = 0; d < rank; ++d) {
                    if (++index[d] < ranges[d].second) {
                        break;
                    }
                    index[d] = ranges[d].first;
                }
            }
        };
        return out;
    }

    /// Whether some read asked for the whole tensor at once.
    [[nodiscard]] bool read_whole() const {
        return std::ranges::any_of(reads, [&](Ranges const &ranges) {
            for (std::size_t d = 0; d < ranges.size(); ++d) {
                if (ranges[d].first != 0 || ranges[d].second != source.dim(static_cast<int>(d))) {
                    return false;
                }
            }
            return true;
        });
    }
};

/// The pass alone, at a slab budget of @p bytes, then Materialization so the graph runs.
std::shared_ptr<cg::passes::SlabStreaming> stream(cg::Graph &graph, std::int64_t bytes) {
    auto pass = std::make_shared<cg::passes::SlabStreaming>();
    pass->set_slab_budget(bytes);
    cg::PassManager manager;
    manager.add(pass);
    manager.add(std::make_shared<cg::passes::Materialization>());
    graph.apply(manager);
    return pass;
}

bool declined_because(cg::passes::SlabStreaming const &pass, std::string_view fragment) {
    for (auto const &[reason, count] : pass.skip_reasons()) {
        if (reason.find(fragment) != std::string::npos) {
            return count > 0;
        }
    }
    return false;
}

std::vector<std::byte> bytes_of(RuntimeTensor<double> const &tensor) {
    std::vector<std::byte> out(tensor.size() * sizeof(double));
    std::memcpy(out.data(), tensor.data(), out.size());
    return out;
}

void check_close(RuntimeTensor<double> const &got, RuntimeTensor<double> const &want, double terms) {
    double scale = 0.0;
    for (std::size_t n = 0; n < want.size(); ++n) {
        scale = std::max(scale, std::abs(want.data()[n]));
    }
    for (std::size_t n = 0; n < want.size(); ++n) {
        INFO("element " << n);
        CHECK(std::abs(got.data()[n] - want.data()[n]) <= 256.0 * std::numeric_limits<double>::epsilon() * terms * scale);
    }
}

} // namespace

TEST_CASE("SlabStreaming reads a backed operand whose last letter the output keeps, a slab per block",
          "[ComputeGraph][Pass][SlabStreaming]") {
    // C[i,j] = A[i,k] B[k,j]: B's last axis is j, which C keeps, so each block writes its own
    // columns of C and the contraction is the captured one, column block by column block.
    constexpr std::size_t ni = 5;
    constexpr std::size_t nk = 4;
    constexpr std::size_t nj = 7;
    MemoryStore           store(filled("B", {nk, nj}, 11U));
    auto const            a  = filled("A", {ni, nk}, 3U);
    auto                  c  = filled("C", {ni, nj}, 5U);
    auto const            c0 = c;

    cg::Graph graph("free");
    auto     &b = graph.declare_runtime_tensor<double>("B", {nk, nj});
    graph.annotate_backing(b, store.backing());
    {
        cg::CaptureGuard const guard(graph);
        cg::einsum("i,j <- i,k ; k,j", 0.5, &c, 2.0, a, b);
    }

    // Two columns of B fit: blocks of two, and a remainder of one.
    auto const pass = stream(graph, static_cast<std::int64_t>(2 * nk * sizeof(double)));
    REQUIRE(pass->num_streamed() == 1);
    CHECK(pass->blocks() == std::vector<std::int64_t>{2});

    graph.execute();
    CHECK_FALSE(b.is_materialized());
    CHECK_FALSE(store.read_whole());
    std::vector<std::pair<std::size_t, std::size_t>> columns;
    for (auto const &ranges : store.reads) {
        CHECK(ranges[0] == std::pair<std::size_t, std::size_t>{0, nk});
        columns.push_back(ranges[1]);
    }
    CHECK(columns == std::vector<std::pair<std::size_t, std::size_t>>{{0, 2}, {2, 4}, {4, 6}, {6, 7}});

    auto expected = c0;
    testing::reference_einsum("i,j <- i,k ; k,j", 0.5, &expected, 2.0, a, store.source);
    check_close(c, expected, static_cast<double>(nk));

    // A block of one index is still a block. The walk once made it a point, which drops the axis,
    // so the loop's views had no end bound and kept the axis whole ("index 'j' spans 1 elements in
    // operand B but 7 in operand C"); the fuzzer found it at a budget of one index.
    SECTION("and a block of one index is a block, not a point") {
        cg::Graph single("single");
        auto     &again = single.declare_runtime_tensor<double>("B", {nk, nj});
        single.annotate_backing(again, store.backing());
        auto d = c0;
        {
            cg::CaptureGuard const guard(single);
            cg::einsum("i,j <- i,k ; k,j", 0.5, &d, 2.0, a, again);
        }
        store.reads.clear();
        auto const one = stream(single, static_cast<std::int64_t>(nk * sizeof(double)));
        REQUIRE(one->blocks() == std::vector<std::int64_t>{1});
        single.execute();
        CHECK(store.reads.size() == nj);
        check_close(d, expected, static_cast<double>(nk));
    }
    SECTION("and a replay reads the same blocks again, from the start") {
        store.reads.clear();
        c = c0;
        graph.execute();
        CHECK(store.reads.size() == 4);
        check_close(c, expected, static_cast<double>(nk));
    }
}

TEST_CASE("SlabStreaming adds the blocks of a summed last letter, the destination scaled once", "[ComputeGraph][Pass][SlabStreaming]") {
    // C[i,j] = 0.5 C + A[i,k] B[j,k]: B's last axis is k, which is summed. The first block applies
    // the 0.5 and the rest add, so a destination scaled once per block would show.
    constexpr std::size_t ni = 3;
    constexpr std::size_t nj = 4;
    constexpr std::size_t nk = 9;
    MemoryStore           store(filled("B", {nj, nk}, 13U));
    auto const            a  = filled("A", {ni, nk}, 7U);
    auto                  c  = filled("C", {ni, nj}, 17U);
    auto const            c0 = c;

    cg::Graph graph("summed");
    auto     &b = graph.declare_runtime_tensor<double>("B", {nj, nk});
    graph.annotate_backing(b, store.backing());
    {
        cg::CaptureGuard const guard(graph);
        cg::einsum("i,j <- i,k ; j,k", 0.5, &c, 1.0, a, b);
    }

    auto const pass = stream(graph, static_cast<std::int64_t>(2 * nj * sizeof(double)));
    REQUIRE(pass->num_streamed() == 1);

    graph.execute();
    CHECK_FALSE(store.read_whole());
    CHECK(store.reads.size() == 5); // [0,2) [2,4) [4,6) [6,8) [8,9)

    auto expected = c0;
    testing::reference_einsum("i,j <- i,k ; j,k", 0.5, &expected, 1.0, a, store.source);
    check_close(c, expected, static_cast<double>(nk));

    SECTION("an overwriting contraction stays an overwrite") {
        cg::Graph overwrite("overwrite");
        auto     &again = overwrite.declare_runtime_tensor<double>("B", {nj, nk});
        overwrite.annotate_backing(again, store.backing());
        auto d      = filled("D", {ni, nj}, 23U);
        d.data()[0] = std::numeric_limits<double>::quiet_NaN();
        {
            cg::CaptureGuard const guard(overwrite);
            cg::einsum("i,j <- i,k ; j,k", 0.0, &d, 1.0, a, again);
        }
        stream(overwrite, static_cast<std::int64_t>(2 * nj * sizeof(double)));
        overwrite.execute();
        auto want = RuntimeTensor<double>("W", std::vector<std::size_t>{ni, nj});
        want.zero();
        testing::reference_einsum("i,j <- i,k ; j,k", 0.0, &want, 1.0, a, store.source);
        check_close(d, want, static_cast<double>(nk));
    }
}

// A backing writes column-major and a slab buffer the library declares follows the row-major
// option Python turns on, so the slab read scatters into the buffer's own layout.
TEST_CASE("SlabStreaming reads slabs into row-major buffers in their own layout", "[ComputeGraph][Pass][SlabStreaming]") {
    bool const previous = config::get(option::RowMajor);
    config::set(option::RowMajor, true);
    struct Restore {
        bool previous;
        ~Restore() { config::set(option::RowMajor, previous); }
    } const restore{previous};

    constexpr std::size_t ni = 3;
    constexpr std::size_t nk = 4;
    constexpr std::size_t nj = 5;
    MemoryStore           store(filled("B", {nk, nj}, 11U));
    auto const            a = filled("A", {ni, nk}, 3U);
    auto                  c = RuntimeTensor<double>("C", std::vector<std::size_t>{ni, nj});
    c.zero();

    cg::Graph graph("row_major");
    auto     &b = graph.declare_runtime_tensor<double>("B", {nk, nj});
    graph.annotate_backing(b, store.backing());
    {
        cg::CaptureGuard const guard(graph);
        cg::einsum("i,j <- i,k ; k,j", &c, a, b);
    }
    auto const pass = stream(graph, static_cast<std::int64_t>(2 * nk * sizeof(double)));
    REQUIRE(pass->num_streamed() == 1);
    graph.execute();
    CHECK_FALSE(store.read_whole());

    auto expected = RuntimeTensor<double>("E", std::vector<std::size_t>{ni, nj});
    expected.zero();
    testing::reference_einsum("i,j <- i,k ; k,j", &expected, a, store.source);
    for (std::size_t i = 0; i < ni; ++i) {
        for (std::size_t j = 0; j < nj; ++j) {
            CHECK_THAT(c(i, j), Catch::Matchers::WithinAbs(expected(i, j), 1e-12));
        }
    }
}

TEST_CASE("SlabStreaming leaves a backed tensor that fits its share to the whole load", "[ComputeGraph][Pass][SlabStreaming]") {
    MemoryStore store(filled("B", {4, 3}, 11U));
    auto const  a = filled("A", {5, 4}, 3U);
    auto        c = RuntimeTensor<double>("C", std::vector<std::size_t>{5, 3});
    c.zero();

    cg::Graph graph("fits");
    auto     &b = graph.declare_runtime_tensor<double>("B", {4, 3});
    graph.annotate_backing(b, store.backing());
    {
        cg::CaptureGuard const guard(graph);
        cg::einsum("i,j <- i,k ; k,j", &c, a, b);
    }

    SECTION("under a budget it fits") {
        auto const pass = stream(graph, static_cast<std::int64_t>(4 * 3 * sizeof(double)));
        CHECK(pass->num_streamed() == 0);
        CHECK(declined_because(*pass, "fits its share of the budget"));
    }
    SECTION("and with the pass off") {
        auto const pass = stream(graph, 0);
        CHECK(pass->num_streamed() == 0);
        CHECK(declined_because(*pass, "no slab budget"));
    }
    graph.execute();
    CHECK(store.read_whole());
    CHECK(store.reads.size() == 1);
}

TEST_CASE("SlabStreaming declines what a block of the last axis cannot express, and says why", "[ComputeGraph][Pass][SlabStreaming]") {
    MemoryStore store(filled("B", {4, 4}, 11U));
    MemoryStore other(filled("X", {4, 4}, 29U));
    auto const  a = filled("A", {4, 4}, 3U);
    auto        c = RuntimeTensor<double>("C", std::vector<std::size_t>{4, 4});
    c.zero();
    std::int64_t const small = static_cast<std::int64_t>(4 * sizeof(double));

    SECTION("both operands backed") {
        cg::Graph graph("both");
        auto     &b = graph.declare_runtime_tensor<double>("B", {4, 4});
        auto     &x = graph.declare_runtime_tensor<double>("X", {4, 4});
        graph.annotate_backing(b, store.backing());
        graph.annotate_backing(x, other.backing());
        {
            cg::CaptureGuard const guard(graph);
            cg::einsum("i,j <- i,k ; k,j", &c, x, b);
        }
        auto const pass = stream(graph, small);
        CHECK(pass->num_streamed() == 0);
        CHECK(declined_because(*pass, "both operands of the contraction are backed"));
    }
    SECTION("the last letter repeated") {
        cg::Graph graph("repeat");
        auto     &b = graph.declare_runtime_tensor<double>("B", {4, 4});
        graph.annotate_backing(b, store.backing());
        auto d = RuntimeTensor<double>("D", std::vector<std::size_t>{4});
        d.zero();
        {
            cg::CaptureGuard const guard(graph);
            cg::einsum("i <- i,k ; k,k", &d, a, b);
        }
        auto const pass = stream(graph, small);
        CHECK(pass->num_streamed() == 0);
        CHECK(declined_because(*pass, "repeats"));
    }
}

// An empty destination was once rewritten anyway, and the loop's range view of it kept the full
// extent of the streamed letter, so capture refused the shapes ("index 'j' spans 1 elements in
// operand B but 2 in operand C"). The fuzzer drew it as i = 0 in "i,j <- i,k ; k,j".
TEST_CASE("SlabStreaming leaves a contraction with an empty operand as captured", "[ComputeGraph][Pass][SlabStreaming]") {
    MemoryStore store(filled("B", {1, 2}, 11U));
    auto const  a = RuntimeTensor<double>("A", std::vector<std::size_t>{0, 1});
    auto        c = RuntimeTensor<double>("C", std::vector<std::size_t>{0, 2});

    cg::Graph graph("empty");
    auto     &b = graph.declare_runtime_tensor<double>("B", {1, 2});
    graph.annotate_backing(b, store.backing());
    {
        cg::CaptureGuard const guard(graph);
        cg::einsum("i,j <- i,k ; k,j", &c, a, b);
    }
    auto const pass = stream(graph, static_cast<std::int64_t>(sizeof(double)));
    CHECK(pass->num_streamed() == 0);
    CHECK(declined_because(*pass, "has no elements"));
    graph.execute();
}

TEST_CASE("SlabStreaming streams the shared integrals inside AxisTiling's block loop", "[ComputeGraph][Pass][SlabStreaming]") {
    // The density-fitted particle ladder, written over every index:
    //   H[Q,a,f,i,j]  = sum_e Bvv[Q,a,e] Tau[i,j,e,f]
    //   r2[i,j,a,b]  += sum_Qf H[Q,a,f,i,j] Bvv[Q,b,f]
    // AxisTiling streams H in blocks of an occupied index, and every block reads ALL of Bvv, the
    // panel the blocking reuses. With Bvv backed and over its share, each block's two
    // contractions read it a slab at a time, so it is never held whole.
    constexpr std::size_t o    = 4;
    constexpr std::size_t v    = 5;
    constexpr std::size_t naux = 6;
    MemoryStore           store(filled("Bvv", {naux, v, v}, 11U));
    auto const            tau = filled("Tau", {o, o, v, v}, 13U);
    auto                  r2  = filled("r2", {o, o, v, v}, 17U);
    auto const            r20 = r2;

    cg::Graph graph("ladder");
    {
        auto &bvv = graph.declare_runtime_tensor<double>("Bvv", {naux, v, v});
        graph.annotate_backing(bvv, store.backing());
        auto                  &half = graph.scratch_runtime<double>("H", std::vector<std::size_t>{naux, v, v, o, o});
        cg::CaptureGuard const guard(graph);
        cg::einsum("Q,a,f,i,j <- Q,a,e ; i,j,e,f", &half, bvv, tau);
        cg::einsum("i,j,a,b <- Q,a,f,i,j ; Q,b,f", 1.0, &r2, 1.0, half, bvv);
    }

    auto tiling = std::make_shared<cg::passes::AxisTiling>();
    tiling->set_memory_cap(static_cast<std::int64_t>(2 * naux * v * v * sizeof(double)));
    auto slabs = std::make_shared<cg::passes::SlabStreaming>();
    slabs->set_slab_budget(static_cast<std::int64_t>(naux * v * 2 * sizeof(double)));
    auto            materialize = std::make_shared<cg::passes::Materialization>();
    cg::PassManager manager;
    manager.add(tiling);
    manager.add(slabs);
    manager.add(materialize);
    graph.apply(manager);

    REQUIRE(tiling->num_tiled() == 1);
    // Both contractions of every block loop AxisTiling emitted.
    CHECK(slabs->num_streamed() >= 2);
    CHECK(std::ranges::all_of(slabs->streamed(), [](std::string const &name) { return name == "Bvv"; }));
    CHECK(materialize->num_loaded() == 0);

    graph.execute();
    CHECK_FALSE(store.read_whole());
    CHECK_FALSE(store.reads.empty());

    auto expected = r20;
    {
        RuntimeTensor<double> half("Href", std::vector<std::size_t>{naux, v, v, o, o});
        half.zero();
        testing::reference_einsum("Q,a,f,i,j <- Q,a,e ; i,j,e,f", &half, store.source, tau);
        testing::reference_einsum("i,j,a,b <- Q,a,f,i,j ; Q,b,f", 1.0, &expected, 1.0, half, store.source);
    }
    check_close(r2, expected, static_cast<double>(naux * v * v));
}
