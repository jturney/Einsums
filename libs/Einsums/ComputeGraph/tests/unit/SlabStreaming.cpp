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
#include <limits>
#include <memory>
#include <optional>
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

/// A write backing over column-major storage in memory that logs every slab it reads and writes.
/// The storage takes its shape from the tensor the backing is annotated on, as a file entry does.
struct WriteStore {
    std::string              name;
    std::vector<std::size_t> dims;
    std::vector<double>      storage;
    std::vector<Ranges>      reads;
    std::vector<Ranges>      writes;

    explicit WriteStore(std::string tensor) : name(std::move(tensor)) {}

    /// Visit every element of @p ranges in column-major order with its offset in the storage.
    template <typename F>
    void walk(Ranges const &ranges, F &&visit) const {
        std::size_t const        rank = ranges.size();
        std::vector<std::size_t> index(rank);
        std::size_t              count = 1;
        for (std::size_t d = 0; d < rank; ++d) {
            index[d] = ranges[d].first;
            count *= ranges[d].second - ranges[d].first;
        }
        for (std::size_t n = 0; n < count; ++n) {
            std::size_t offset = 0;
            std::size_t stride = 1;
            for (std::size_t d = 0; d < rank; ++d) {
                offset += index[d] * stride;
                stride *= dims[d];
            }
            visit(n, offset);
            for (std::size_t d = 0; d < rank; ++d) {
                if (++index[d] < ranges[d].second) {
                    break;
                }
                index[d] = ranges[d].first;
            }
        }
    }

    cg::TensorBacking backing() {
        cg::TensorBacking out;
        out.kind     = "slab_test.memory";
        out.location = {{"name", name}};
        out.write    = true;
        out.prepare  = [this](std::vector<std::size_t> const &shape, packed_gemm::ScalarType dtype) {
            REQUIRE(dtype == packed_gemm::ScalarType::Float64);
            dims              = shape;
            std::size_t count = 1;
            for (std::size_t const extent : shape) {
                count *= extent;
            }
            // NaN until written, so an element the graph never stores cannot pass for a result.
            storage.assign(count, std::numeric_limits<double>::quiet_NaN());
            cg::TensorBacking ready;
            ready.kind      = "slab_test.memory";
            ready.location  = {{"name", name}};
            ready.dims      = shape;
            ready.dtype     = dtype;
            ready.write     = true;
            ready.read_slab = [this](void *destination, Ranges const &ranges) {
                reads.push_back(ranges);
                auto *dst = static_cast<double *>(destination);
                walk(ranges, [&](std::size_t n, std::size_t offset) { dst[n] = storage[offset]; });
            };
            ready.write_slab = [this](void const *source, Ranges const &ranges) {
                writes.push_back(ranges);
                auto const *src = static_cast<double const *>(source);
                walk(ranges, [&](std::size_t n, std::size_t offset) { storage[offset] = src[n]; });
            };
            return ready;
        };
        return out;
    }

    /// The stored tensor, as a column-major tensor of the same shape.
    [[nodiscard]] RuntimeTensor<double> stored() const {
        RuntimeTensor<double> out(name, dims);
        Ranges                whole;
        for (std::size_t const extent : dims) {
            whole.emplace_back(0, extent);
        }
        walk(whole, [&](std::size_t, std::size_t offset) {
            std::vector<std::size_t> index(dims.size());
            std::size_t              rest = offset;
            for (std::size_t d = 0; d < dims.size(); ++d) {
                index[d] = rest % dims[d];
                rest /= dims[d];
            }
            std::size_t at = 0;
            for (std::size_t d = 0; d < dims.size(); ++d) {
                at += index[d] * out.stride(static_cast<int>(d));
            }
            out.data()[at] = storage[offset];
        });
        return out;
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

TEST_CASE("SlabStreaming reads the box AxisTiling's view names, at every iteration of its loop", "[ComputeGraph][Pass][SlabStreaming]") {
    //   H[Q,a,f] = sum_e Bvv[Q,a,e] X[e,f]
    //   out[a]   = sum_Qf H[Q,a,f] Y[Q,f]
    // AxisTiling loops over an axis of H, and every iteration views the box of Bvv that block
    // needs: Bvv[:, a-block, :] over a, or Bvv[Q-block, :, :] over Q, which the second contraction
    // sums into out. That view used to have Bvv loaded whole before the loop; the loop has already
    // chosen the box, so each iteration now reads exactly that box, whichever axis it is on.
    constexpr std::size_t naux = 6;
    constexpr std::size_t v    = 5;
    constexpr std::size_t w    = 3;
    MemoryStore           store(filled("Bvv", {naux, v, v}, 11U));
    auto const            x   = filled("X", {v, w}, 13U);
    auto const            y   = filled("Y", {naux, w}, 7U);
    auto                  out = filled("out", {v}, 17U);

    cg::Graph graph("box");
    {
        auto &bvv = graph.declare_runtime_tensor<double>("Bvv", {naux, v, v});
        graph.annotate_backing(bvv, store.backing());
        auto &half = graph.declare_runtime_tensor<double>("H", std::vector<std::size_t>{naux, v, w}, /*intermediate=*/true);
        cg::CaptureGuard const guard(graph);
        cg::einsum("Q,a,f <- Q,a,e ; e,f", &half, bvv, x);
        cg::einsum("a <- Q,a,f ; Q,f", &out, half, y);
    }

    auto tiling = std::make_shared<cg::passes::AxisTiling>();
    tiling->set_memory_cap(static_cast<std::int64_t>(2 * naux * w * sizeof(double)));
    auto slabs = std::make_shared<cg::passes::SlabStreaming>();
    slabs->set_slab_budget(static_cast<std::int64_t>(naux * v * 2 * sizeof(double)));
    auto            materialize = std::make_shared<cg::passes::Materialization>();
    cg::PassManager manager;
    manager.add(tiling);
    manager.add(slabs);
    manager.add(materialize);
    graph.apply(manager);

    REQUIRE(tiling->num_tiled() == 1);
    CHECK(slabs->num_boxes() >= 1);
    CHECK(materialize->num_loaded() == 0);

    graph.execute();
    CHECK_FALSE(store.read_whole());
    // Every read is a box, and together they cover every element exactly once.
    std::vector<int> covered(naux * v * v, 0);
    for (auto const &ranges : store.reads) {
        REQUIRE(ranges.size() == 3);
        for (std::size_t q = ranges[0].first; q < ranges[0].second; ++q) {
            for (std::size_t a = ranges[1].first; a < ranges[1].second; ++a) {
                for (std::size_t e = ranges[2].first; e < ranges[2].second; ++e) {
                    ++covered[q + (a * naux) + (e * naux * v)];
                }
            }
        }
    }
    CHECK(std::ranges::all_of(covered, [](int count) { return count == 1; }));

    RuntimeTensor<double> half("Href", std::vector<std::size_t>{naux, v, w});
    half.zero();
    testing::reference_einsum("Q,a,f <- Q,a,e ; e,f", &half, store.source, x);
    RuntimeTensor<double> expected("out_ref", std::vector<std::size_t>{v});
    expected.zero();
    testing::reference_einsum("a <- Q,a,f ; Q,f", &expected, half, y);
    check_close(out, expected, static_cast<double>(naux * v * w));
}

TEST_CASE("SlabStreaming writes the box AxisTiling's view names, at every iteration of its loop", "[ComputeGraph][Pass][SlabStreaming]") {
    //   H[Q,a,n] = sum_m P[Q,m,n] C[m,a]
    //   W[Q,a,b] = sum_n H[Q,a,n] C[n,b]
    // H is an intermediate over the cap, so AxisTiling loops over one of its axes and the second
    // contraction, H's reader, joins the loop and writes W through a view that slices it. That view
    // used to leave W computed whole in memory and stored after the loop; now each iteration writes
    // the box it computed, and W is never held whole.
    constexpr std::size_t naux = 6;
    constexpr std::size_t nbf  = 4;
    constexpr std::size_t v    = 5;
    WriteStore            store("W");
    auto const            p = filled("P", {naux, nbf, nbf}, 3U);
    auto const            c = filled("C", {nbf, v}, 7U);

    cg::Graph graph("box write");
    {
        auto &w = graph.declare_runtime_tensor<double>("W", {naux, v, v});
        graph.annotate_backing(w, store.backing());
        auto &half = graph.declare_runtime_tensor<double>("H", std::vector<std::size_t>{naux, v, nbf}, /*intermediate=*/true);
        cg::CaptureGuard const guard(graph);
        cg::einsum("Q,a,n <- Q,m,n ; m,a", &half, p, c);
        cg::einsum("Q,a,b <- Q,a,n ; n,b", &w, half, c);
    }

    auto tiling = std::make_shared<cg::passes::AxisTiling>();
    tiling->set_memory_cap(static_cast<std::int64_t>(2 * naux * nbf * sizeof(double)));
    auto slabs = std::make_shared<cg::passes::SlabStreaming>();
    slabs->set_slab_budget(static_cast<std::int64_t>(naux * v * 2 * sizeof(double)));
    auto            materialize = std::make_shared<cg::passes::Materialization>();
    cg::PassManager manager;
    manager.add(tiling);
    manager.add(slabs);
    manager.add(materialize);
    graph.apply(manager);

    REQUIRE(tiling->num_tiled() == 1);
    CHECK(slabs->num_boxes() >= 1);
    CHECK(graph.verify().empty());

    graph.execute();
    // No write covers the whole tensor, and together the writes cover every element exactly once.
    std::vector<int> covered(naux * v * v, 0);
    for (auto const &ranges : store.writes) {
        REQUIRE(ranges.size() == 3);
        CHECK_FALSE((ranges[0] == std::pair<std::size_t, std::size_t>{0, naux} && ranges[1] == std::pair<std::size_t, std::size_t>{0, v} &&
                     ranges[2] == std::pair<std::size_t, std::size_t>{0, v}));
        for (std::size_t q = ranges[0].first; q < ranges[0].second; ++q) {
            for (std::size_t a = ranges[1].first; a < ranges[1].second; ++a) {
                for (std::size_t b = ranges[2].first; b < ranges[2].second; ++b) {
                    ++covered[q + (a * naux) + (b * naux * v)];
                }
            }
        }
    }
    CHECK(std::ranges::all_of(covered, [](int count) { return count == 1; }));

    RuntimeTensor<double> half("Href", std::vector<std::size_t>{naux, v, nbf});
    half.zero();
    testing::reference_einsum("Q,a,n <- Q,m,n ; m,a", &half, p, c);
    RuntimeTensor<double> want("W_ref", std::vector<std::size_t>{naux, v, v});
    want.zero();
    testing::reference_einsum("Q,a,b <- Q,a,n ; n,b", &want, half, c);
    check_close(store.stored(), want, static_cast<double>(nbf * nbf));
}

TEST_CASE("SlabStreaming produces a write-backed tensor a slab per block, straight into its storage",
          "[ComputeGraph][Pass][SlabStreaming]") {
    // T[i,j] = A[i,k] B[k,j] overwrites T, and nothing else writes it: each block computes its
    // columns of T into a slab and writes the slab, so T is never held whole. The reader after it,
    // D[i,l] = T[i,j] E[j,l], then reads T back from the storage a block of j at a time.
    constexpr std::size_t ni = 5;
    constexpr std::size_t nk = 3;
    constexpr std::size_t nj = 7;
    constexpr std::size_t nl = 2;
    WriteStore            store("T");
    auto const            a = filled("A", {ni, nk}, 3U);
    auto const            b = filled("B", {nk, nj}, 7U);
    auto const            e = filled("E", {nj, nl}, 9U);
    auto                  d = filled("D", {ni, nl}, 5U);

    cg::Graph graph("produce");
    auto     &t = graph.declare_runtime_tensor<double>("T", {ni, nj});
    graph.annotate_backing(t, store.backing());
    {
        cg::CaptureGuard const guard(graph);
        cg::einsum("i,j <- i,k ; k,j", 0.0, &t, 2.0, a, b);
        cg::einsum("i,l <- i,j ; j,l", &d, t, e);
    }
    CHECK(graph.verify().empty());

    // Two columns of T fit: blocks of two, and a remainder of one.
    auto const pass = stream(graph, static_cast<std::int64_t>(2 * ni * sizeof(double)));
    REQUIRE(pass->num_streamed() == 2);
    CHECK(pass->num_written() == 1);
    CHECK(pass->streamed() == std::vector<std::string>{"T", "T"});
    CHECK(graph.verify().empty());

    graph.execute();
    CHECK_FALSE(t.is_materialized());
    std::vector<std::pair<std::size_t, std::size_t>> written;
    for (auto const &ranges : store.writes) {
        CHECK(ranges[0] == std::pair<std::size_t, std::size_t>{0, ni});
        written.push_back(ranges[1]);
    }
    CHECK(written == std::vector<std::pair<std::size_t, std::size_t>>{{0, 2}, {2, 4}, {4, 6}, {6, 7}});
    CHECK(store.reads.size() == 4);

    auto want_t = RuntimeTensor<double>("T", std::vector<std::size_t>{ni, nj});
    want_t.zero();
    testing::reference_einsum("i,j <- i,k ; k,j", 0.0, &want_t, 2.0, a, b);
    check_close(store.stored(), want_t, static_cast<double>(nk));

    auto want_d = RuntimeTensor<double>("D", std::vector<std::size_t>{ni, nl});
    want_d.zero();
    testing::reference_einsum("i,l <- i,j ; j,l", &want_d, want_t, e);
    check_close(d, want_d, static_cast<double>(nk * nj));

    SECTION("and a replay writes every block again") {
        store.writes.clear();
        store.reads.clear();
        std::ranges::fill(store.storage, std::numeric_limits<double>::quiet_NaN());
        graph.execute();
        CHECK(store.writes.size() == 4);
        check_close(store.stored(), want_t, static_cast<double>(nk));
    }
}

TEST_CASE("SlabStreaming builds its loops over the graph's operands, not the caller's", "[ComputeGraph][Pass][SlabStreaming]") {
    // Capture adopts each operand's storage so the caller may drop its own tensor before the graph
    // runs. The pass used to view the caller's tensor in the loops it built, and a program that
    // captured in a function and returned only the graph read freed memory in apply(). Here every
    // operand but the result is replaced, at the same address, by a tensor of another shape, which
    // a loop built over the caller's tensor would read, and both the slabs written and the result
    // read back from them are still the captured ones.
    constexpr std::size_t                ni = 5;
    constexpr std::size_t                nk = 3;
    constexpr std::size_t                nj = 7;
    constexpr std::size_t                nl = 2;
    WriteStore                           store("T");
    std::optional<RuntimeTensor<double>> a{filled("A", {ni, nk}, 3U)};
    std::optional<RuntimeTensor<double>> b{filled("B", {nk, nj}, 7U)};
    std::optional<RuntimeTensor<double>> e{filled("E", {nj, nl}, 9U)};
    auto                                 d  = filled("D", {ni, nl}, 5U);
    auto const                           d0 = d;

    auto want_t = RuntimeTensor<double>("T", std::vector<std::size_t>{ni, nj});
    want_t.zero();
    testing::reference_einsum("i,j <- i,k ; k,j", 0.0, &want_t, 2.0, *a, *b);
    auto want_d = d0;
    testing::reference_einsum("i,l <- i,j ; j,l", &want_d, want_t, *e);

    cg::Graph graph("dropped operands");
    auto     &t = graph.declare_runtime_tensor<double>("T", {ni, nj});
    graph.annotate_backing(t, store.backing());
    {
        cg::CaptureGuard const guard(graph);
        cg::einsum("i,j <- i,k ; k,j", 0.0, &t, 2.0, *a, *b);
        cg::einsum("i,l <- i,j ; j,l", &d, t, *e);
    }
    for (auto *operand : {&a, &b, &e}) {
        operand->reset();
        operand->emplace(filled("other", {2, 2}, 13U));
    }

    auto const pass = stream(graph, static_cast<std::int64_t>(2 * ni * sizeof(double)));
    REQUIRE(pass->num_streamed() == 2);
    REQUIRE(pass->num_written() == 1);
    CHECK(graph.verify().empty());

    graph.execute();
    CHECK(store.writes.size() == 4);
    CHECK(store.reads.size() == 4);
    check_close(store.stored(), want_t, static_cast<double>(nk));
    check_close(d, want_d, static_cast<double>(nk * nj));
}

TEST_CASE("SlabStreaming loads a streamed-out tensor whole for a reader it cannot stream, after the writes",
          "[ComputeGraph][Pass][SlabStreaming]") {
    // D[i,l] = T[i,j] T[l,j] reads T twice, which this pass declines, so T is filled whole from its
    // storage. The fill carries the storage's key, so it runs after the block loops that wrote it.
    constexpr std::size_t ni = 4;
    constexpr std::size_t nk = 3;
    constexpr std::size_t nj = 5;
    WriteStore            store("T");
    auto const            a = filled("A", {ni, nk}, 3U);
    auto const            b = filled("B", {nk, nj}, 7U);
    auto                  d = RuntimeTensor<double>("D", std::vector<std::size_t>{ni, ni});
    d.zero();

    cg::Graph graph("whole_reader");
    auto     &t = graph.declare_runtime_tensor<double>("T", {ni, nj});
    graph.annotate_backing(t, store.backing());
    {
        cg::CaptureGuard const guard(graph);
        cg::einsum("i,j <- i,k ; k,j", &t, a, b);
        cg::einsum("i,l <- i,j ; l,j", &d, t, t);
    }
    auto const pass = stream(graph, static_cast<std::int64_t>(2 * ni * sizeof(double)));
    REQUIRE(pass->num_written() == 1);
    CHECK(pass->num_streamed() == 1);
    CHECK(declined_because(*pass, "both operands of the contraction are backed"));

    // Ordered by the storage's key, not by where the nodes happen to sit: the fill reads T's
    // storage and each block loop writes it, so a scheduler that runs independent nodes together
    // must still put the fill after every loop. Sequential replay alone would not show a missing
    // edge, since the fill is placed after the loops anyway.
    graph.topological_sort();
    std::vector<std::size_t> level_of(graph.nodes().size(), 0);
    for (std::size_t l = 0; l < graph.dependencies().levels.size(); ++l) {
        for (std::size_t const n : graph.dependencies().levels[l]) {
            level_of[n] = l;
        }
    }
    std::optional<std::size_t> fill;
    std::vector<std::size_t>   loops;
    for (std::size_t n = 0; n < graph.nodes().size(); ++n) {
        if (graph.nodes()[n].kind == cg::OpKind::Initialize) {
            fill = n;
        } else if (graph.nodes()[n].kind == cg::OpKind::Loop) {
            loops.push_back(n);
        }
    }
    REQUIRE(fill.has_value());
    REQUIRE(loops.size() == 2);
    for (std::size_t const loop : loops) {
        CHECK(level_of[*fill] > level_of[loop]);
    }

    graph.execute();
    REQUIRE(store.reads.size() == 1);
    CHECK(store.reads[0] == Ranges{{0, ni}, {0, nj}});

    auto want_t = RuntimeTensor<double>("T", std::vector<std::size_t>{ni, nj});
    want_t.zero();
    testing::reference_einsum("i,j <- i,k ; k,j", &want_t, a, b);
    auto want_d = RuntimeTensor<double>("D", std::vector<std::size_t>{ni, ni});
    want_d.zero();
    testing::reference_einsum("i,l <- i,j ; l,j", &want_d, want_t, want_t);
    check_close(d, want_d, static_cast<double>(nk * nj));
}

TEST_CASE("A write-backed tensor computed in memory is stored after its last writer", "[ComputeGraph][Pass][SlabStreaming]") {
    // T = A B, then T += A B2: the second contraction adds, so the pass cannot produce T in
    // blocks. T is computed in memory, read from there, and stored once, after the last writer.
    // F = T E is a reader the pass would stream from a stored tensor; here it reads memory.
    constexpr std::size_t ni = 3;
    constexpr std::size_t nk = 2;
    constexpr std::size_t nj = 4;
    auto const            a  = filled("A", {ni, nk}, 3U);
    auto const            b  = filled("B", {nk, nj}, 7U);
    auto const            b2 = filled("B2", {nk, nj}, 13U);
    auto const            e  = filled("E", {nj, 2}, 9U);

    auto want = RuntimeTensor<double>("T", std::vector<std::size_t>{ni, nj});
    want.zero();
    testing::reference_einsum("i,j <- i,k ; k,j", &want, a, b);
    testing::reference_einsum("i,j <- i,k ; k,j", 1.0, &want, 1.0, a, b2);

    auto const capture = [&](cg::Graph &graph, WriteStore &store, RuntimeTensor<double> &d, RuntimeTensor<double> &f) {
        auto &t = graph.declare_runtime_tensor<double>("T", {ni, nj});
        graph.annotate_backing(t, store.backing());
        cg::CaptureGuard const guard(graph);
        cg::einsum("i,j <- i,k ; k,j", &t, a, b);
        cg::einsum("i,j <- i,k ; k,j", 1.0, &t, 1.0, a, b2);
        cg::einsum("i,l <- i,j ; l,j", &d, t, t);
        cg::einsum("i,m <- i,j ; j,m", &f, t, e);
    };
    auto want_d = RuntimeTensor<double>("D", std::vector<std::size_t>{ni, ni});
    want_d.zero();
    testing::reference_einsum("i,l <- i,j ; l,j", &want_d, want, want);
    auto want_f = RuntimeTensor<double>("F", std::vector<std::size_t>{ni, 2});
    want_f.zero();
    testing::reference_einsum("i,m <- i,j ; j,m", &want_f, want, e);

    SECTION("under a budget the producers do not fit") {
        WriteStore store("T");
        auto       d = RuntimeTensor<double>("D", std::vector<std::size_t>{ni, ni});
        d.zero();
        auto f = RuntimeTensor<double>("F", std::vector<std::size_t>{ni, 2});
        f.zero();
        cg::Graph graph("accumulate");
        capture(graph, store, d, f);
        auto const pass = stream(graph, static_cast<std::int64_t>(ni * sizeof(double)));
        CHECK(pass->num_streamed() == 0);
        CHECK(declined_because(*pass, "adds into the write-backed tensor"));
        CHECK(declined_because(*pass, "another operation writes the write-backed tensor in memory"));
        CHECK(declined_because(*pass, "the tensor is produced in memory, so its readers read it there"));
        graph.execute();
        CHECK(store.reads.empty());
        CHECK(store.writes == std::vector<Ranges>{Ranges{{0, ni}, {0, nj}}});
        check_close(store.stored(), want, static_cast<double>(2 * nk));
        check_close(d, want_d, static_cast<double>(4 * nk * nj));
        check_close(f, want_f, static_cast<double>(2 * nk * nj));
    }
    SECTION("with the pass off") {
        WriteStore store("T");
        auto       d = RuntimeTensor<double>("D", std::vector<std::size_t>{ni, ni});
        d.zero();
        auto f = RuntimeTensor<double>("F", std::vector<std::size_t>{ni, 2});
        f.zero();
        cg::Graph graph("pass_off");
        capture(graph, store, d, f);
        auto const pass = stream(graph, 0);
        CHECK(pass->num_streamed() == 0);
        graph.execute();
        CHECK(store.writes.size() == 1);
        check_close(store.stored(), want, static_cast<double>(2 * nk));
        check_close(d, want_d, static_cast<double>(4 * nk * nj));
        check_close(f, want_f, static_cast<double>(2 * nk * nj));
    }
}

TEST_CASE("SlabStreaming writes slabs out of row-major buffers in the storage's own order", "[ComputeGraph][Pass][SlabStreaming]") {
    bool const previous = config::get(option::RowMajor);
    config::set(option::RowMajor, true);
    struct Restore {
        bool previous;
        ~Restore() { config::set(option::RowMajor, previous); }
    } const restore{previous};

    constexpr std::size_t ni = 3;
    constexpr std::size_t nk = 4;
    constexpr std::size_t nj = 5;
    WriteStore            store("T");
    auto const            a = filled("A", {ni, nk}, 3U);
    auto const            b = filled("B", {nk, nj}, 11U);

    cg::Graph graph("row_major_write");
    auto     &t = graph.declare_runtime_tensor<double>("T", {ni, nj});
    graph.annotate_backing(t, store.backing());
    {
        cg::CaptureGuard const guard(graph);
        cg::einsum("i,j <- i,k ; k,j", &t, a, b);
    }
    auto const pass = stream(graph, static_cast<std::int64_t>(2 * ni * sizeof(double)));
    REQUIRE(pass->num_written() == 1);
    graph.execute();

    auto want = RuntimeTensor<double>("E", std::vector<std::size_t>{ni, nj});
    want.zero();
    testing::reference_einsum("i,j <- i,k ; k,j", &want, a, b);
    // The storage is column-major whatever the tensors are: element (i, j) at i + j * ni.
    for (std::size_t i = 0; i < ni; ++i) {
        for (std::size_t j = 0; j < nj; ++j) {
            CHECK_THAT(store.storage[i + (j * ni)], Catch::Matchers::WithinAbs(want(i, j), 1e-12));
        }
    }
}

TEST_CASE("verify refuses a write backing nothing writes", "[ComputeGraph][Pass][SlabStreaming]") {
    WriteStore store("T");
    cg::Graph  graph("unwritten");
    auto      &t = graph.declare_runtime_tensor<double>("T", {2, 3});
    graph.annotate_backing(t, store.backing());
    auto const problems = graph.verify();
    REQUIRE(problems.size() == 1);
    CHECK_THAT(problems[0],
               Catch::Matchers::ContainsSubstring("has a write backing") && Catch::Matchers::ContainsSubstring("no node writes it"));

    // Written, through either route, the declaration holds.
    auto const a = filled("A", {2, 2}, 3U);
    auto const b = filled("B", {2, 3}, 7U);
    {
        cg::CaptureGuard const guard(graph);
        cg::einsum("i,j <- i,k ; k,j", &t, a, b);
    }
    CHECK(graph.verify().empty());
    stream(graph, static_cast<std::int64_t>(2 * sizeof(double)));
    CHECK(graph.verify().empty());
}
