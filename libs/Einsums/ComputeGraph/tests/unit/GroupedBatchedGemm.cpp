//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

/// @file GroupedBatchedGemm.cpp
/// @brief Tests for `cg::grouped_batched_gemm`, the node that runs independent
///        GEMMs of DIFFERING shape under one OpenMP region.
///
/// The properties worth pinning are that it agrees with issuing the members one
/// at a time, that a single-shape call is exactly `cg::batched_gemm` rather than
/// merely close to it, that the grouping it derives is visible in the
/// descriptor, and that it survives a rebind - which is the one way a node that
/// baked its pointers in would pass every other test here and still be wrong.

#include <Einsums/ComputeGraph.hpp>
#include <Einsums/LinearAlgebra.hpp>
#include <Einsums/Tensor/RuntimeTensor.hpp>
#include <Einsums/Tensor/Tensor.hpp>
#include <Einsums/TensorUtilities/CreateRandomTensor.hpp>
#include <Einsums/TensorUtilities/CreateZeroTensor.hpp>

#include <Einsums/Testing.hpp>

using namespace einsums;
namespace cg = einsums::compute_graph;

namespace {

/// A member's three operands, owned so the caller can keep a batch in a vector.
template <typename T>
struct Member {
    Tensor<T, 2> a, b, c;
};

/// One member sized so that op(A) is m x k, op(B) is k x n, C is m x n.
template <typename T>
Member<T> make_member(size_t m, size_t n, size_t k, bool trans_a, bool trans_b, int seed) {
    auto const a_rows = trans_a ? k : m;
    auto const a_cols = trans_a ? m : k;
    auto const b_rows = trans_b ? n : k;
    auto const b_cols = trans_b ? k : n;
    return Member<T>{.a = create_random_tensor<T>(fmt::format("A{}", seed), a_rows, a_cols),
                     .b = create_random_tensor<T>(fmt::format("B{}", seed), b_rows, b_cols),
                     .c = create_random_tensor<T>(fmt::format("C{}", seed), m, n)};
}

/// The oracle: every member through the plain eager GEMM, one at a time.
template <typename T>
std::vector<Tensor<T, 2>> reference(std::vector<Member<T>> const &batch, T alpha, T beta, bool trans_a, bool trans_b) {
    std::vector<Tensor<T, 2>> out;
    out.reserve(batch.size());
    for (auto const &mem : batch) {
        Tensor<T, 2> c = mem.c;
        // Chars, not bools: the eager gemm takes BLAS transpose letters, and a
        // bool converts to char 0, which is not 'N' and so reads as transposed.
        linear_algebra::gemm(trans_a ? 'T' : 'N', trans_b ? 'T' : 'N', alpha, mem.a, mem.b, beta, &c);
        out.push_back(std::move(c));
    }
    return out;
}

template <typename T>
void require_close(Tensor<T, 2> const &got, Tensor<T, 2> const &ref) {
    auto const tol = static_cast<RemoveComplexT<T>>(std::is_same_v<RemoveComplexT<T>, float> ? 1.0e-4 : 1.0e-11);
    REQUIRE(got.size() == ref.size());
    for (size_t i = 0; i < got.size(); i++) {
        REQUIRE(std::abs(got.data()[i] - ref.data()[i]) < tol);
    }
}

/// Pointer lists over a batch, in the order the caller wrote it.
template <typename T>
struct Lists {
    std::vector<Tensor<T, 2> const *> a, b;
    std::vector<Tensor<T, 2> *>       c;
};

template <typename T>
Lists<T> lists_of(std::vector<Member<T>> &batch) {
    Lists<T> l;
    for (auto &mem : batch) {
        l.a.push_back(&mem.a);
        l.b.push_back(&mem.b);
        l.c.push_back(&mem.c);
    }
    return l;
}

/// A batch whose members deliberately do not agree on shape: three distinct
/// shape classes, one of them a singleton, interleaved so that grouping has to
/// reorder rather than merely partition a sorted list.
template <typename T>
std::vector<Member<T>> mixed_batch(bool trans_a, bool trans_b) {
    std::vector<Member<T>> batch;
    batch.push_back(make_member<T>(4, 5, 6, trans_a, trans_b, 0));
    batch.push_back(make_member<T>(7, 3, 9, trans_a, trans_b, 1));
    batch.push_back(make_member<T>(4, 5, 6, trans_a, trans_b, 2));
    batch.push_back(make_member<T>(20, 17, 24, trans_a, trans_b, 3));
    batch.push_back(make_member<T>(7, 3, 9, trans_a, trans_b, 4));
    batch.push_back(make_member<T>(4, 5, 6, trans_a, trans_b, 5));
    return batch;
}

} // namespace

TEMPLATE_TEST_CASE("grouped_batched_gemm: eager matches member-by-member", "[ComputeGraph][GroupedBatchedGemm]", float, double,
                   std::complex<float>, std::complex<double>) {
    using T = TestType;

    auto const trans_a = GENERATE(false, true);
    auto const trans_b = GENERATE(false, true);
    // Neither prefactor is 1 or 0, so an ignored alpha or beta cannot pass.
    T const alpha{2}, beta{-3};

    auto batch = mixed_batch<T>(trans_a, trans_b);
    auto ref   = reference<T>(batch, alpha, beta, trans_a, trans_b);

    auto l = lists_of(batch);
    cg::grouped_batched_gemm(static_cast<double>(std::real(alpha)), l.a, l.b, static_cast<double>(std::real(beta)), l.c, trans_a, trans_b);

    for (size_t i = 0; i < batch.size(); i++) {
        INFO("member " << i);
        require_close(batch[i].c, ref[i]);
    }
}

TEST_CASE("grouped_batched_gemm: capture and replay match eager", "[ComputeGraph][GroupedBatchedGemm]") {
    auto batch = mixed_batch<double>(true, false);
    auto ref   = reference<double>(batch, 1.5, 0.5, true, false);

    auto      l = lists_of(batch);
    cg::Graph graph("grouped");
    {
        cg::CaptureGuard const guard(graph);
        cg::grouped_batched_gemm(1.5, l.a, l.b, 0.5, l.c, true, false);
    }
    REQUIRE(graph.num_nodes() == 1);
    graph.execute();

    for (size_t i = 0; i < batch.size(); i++) {
        INFO("member " << i);
        require_close(batch[i].c, ref[i]);
    }
}

TEST_CASE("grouped_batched_gemm: the derived grouping is visible in the descriptor", "[ComputeGraph][GroupedBatchedGemm]") {
    // Collapsing many nodes into one must not cost the ability to see what is
    // inside it, so the descriptor names its groups and says how big they are.
    auto batch = mixed_batch<double>(false, false);
    auto l     = lists_of(batch);

    cg::Graph graph("grouped_desc");
    {
        cg::CaptureGuard const guard(graph);
        cg::grouped_batched_gemm(1.0, l.a, l.b, 0.0, l.c);
    }

    REQUIRE(graph.num_nodes() == 1);
    REQUIRE(graph.nodes()[0].kind == cg::OpKind::GroupedBatchedGemm);

    auto const *d = std::get_if<cg::GroupedBatchedGemmDescriptor>(&graph.nodes()[0].op_data);
    REQUIRE(d != nullptr);
    REQUIRE(d->total == 6);
    REQUIRE(d->groups.size() == 3);
    REQUIRE(d->labels.size() == 3);
    REQUIRE(d->scalar == cg::BlasScalar::Double);

    // Groups in order of first appearance: (4,5,6) x3, (7,3,9) x2, (20,17,24) x1.
    REQUIRE(d->groups[0].m == 4);
    REQUIRE(d->groups[0].count == 3);
    REQUIRE(d->groups[0].first == 0);
    REQUIRE(d->groups[1].m == 7);
    REQUIRE(d->groups[1].count == 2);
    REQUIRE(d->groups[1].first == 3);
    REQUIRE(d->groups[2].m == 20);
    REQUIRE(d->groups[2].count == 1);
    REQUIRE(d->groups[2].first == 5);

    // The offsets have to index the node's own operand lists, not just the
    // executor's, or a pass reading the node cannot follow a group to its
    // operands.
    REQUIRE(graph.nodes()[0].outputs.size() == 6);
    REQUIRE(graph.nodes()[0].inputs.size() == 12);
}

TEST_CASE("grouped_batched_gemm: one shape is bit-for-bit cg::batched_gemm", "[ComputeGraph][GroupedBatchedGemm]") {
    // The uniform form is the special case, so it has to be the SAME
    // computation and not merely an equivalent one. Anything less means two
    // routes through the same work that can drift.
    std::vector<Member<double>> batch;
    batch.reserve(5);
    for (int i = 0; i < 5; i++) {
        batch.push_back(make_member<double>(9, 7, 11, true, false, i));
    }
    auto uniform = batch;

    {
        auto l = lists_of(uniform);
        cg::batched_gemm(1.25, l.a, l.b, -0.75, l.c, true, false);
    }
    {
        auto l = lists_of(batch);
        cg::grouped_batched_gemm(1.25, l.a, l.b, -0.75, l.c, true, false);
    }

    for (size_t i = 0; i < batch.size(); i++) {
        INFO("member " << i);
        for (size_t e = 0; e < batch[i].c.size(); e++) {
            REQUIRE(batch[i].c.data()[e] == uniform[i].c.data()[e]);
        }
    }
}

TEST_CASE("grouped_batched_gemm: survives a rebind", "[ComputeGraph][GroupedBatchedGemm][Rebind]") {
    // The one test a node that baked its pointers in would fail. rebind() and
    // the MemoryPlanning arena both move storage between executions, so the
    // executor has to read through the slot every time.
    auto batch = mixed_batch<double>(false, false);
    auto l     = lists_of(batch);

    cg::Graph graph("grouped_rebind");
    {
        cg::CaptureGuard const guard(graph);
        cg::grouped_batched_gemm(1.0, l.a, l.b, 0.0, l.c);
    }

    // Swap one destination for a different tensor and replay; the result has to
    // land in the NEW tensor and match what the old one would have received.
    auto              expected = reference<double>(batch, 1.0, 0.0, false, false);
    Tensor<double, 2> fresh    = create_zero_tensor<double>("fresh", batch[3].c.dim(0), batch[3].c.dim(1));
    graph.rebind(batch[3].c, fresh);
    graph.execute();

    require_close(fresh, expected[3]);
}

TEST_CASE("grouped_batched_gemm: beta != 0 records the read of every destination", "[ComputeGraph][GroupedBatchedGemm]") {
    // Accumulation reads C before writing it, so the edge from whoever produced
    // C must survive or the sort is free to hoist this above its producer.
    auto batch = mixed_batch<double>(false, false);
    auto l     = lists_of(batch);

    cg::Graph beta_zero("grouped_b0");
    {
        cg::CaptureGuard const guard(beta_zero);
        cg::grouped_batched_gemm(1.0, l.a, l.b, 0.0, l.c);
    }
    cg::Graph beta_one("grouped_b1");
    {
        cg::CaptureGuard const guard(beta_one);
        cg::grouped_batched_gemm(1.0, l.a, l.b, 1.0, l.c);
    }

    REQUIRE(beta_zero.nodes()[0].inputs.size() == 12);
    REQUIRE(beta_one.nodes()[0].inputs.size() == 18);
}

TEST_CASE("grouped_batched_gemm_blocked: blocks of several bases, several shapes", "[ComputeGraph][GroupedBatchedGemm]") {
    // The DLPNO overlap build's shape exactly: one destination tensor per shape
    // class, every member a column range of its class's tensor, and the classes
    // disagreeing on shape. Neither the blocked form nor the grouped form alone
    // expresses it.
    constexpr size_t k = 6;
    // Class 0: 4x5 blocks, three of them. Class 1: 7x3 blocks, two of them.
    auto base0 = create_zero_tensor<double>("base0", 4, 5 * 3);
    auto base1 = create_zero_tensor<double>("base1", 7, 3 * 2);

    std::vector<Tensor<double, 2>>   a_store, b_store;
    std::vector<size_t>              offsets;
    std::vector<Tensor<double, 2> *> bases;
    a_store.reserve(5);
    b_store.reserve(5);
    for (int i = 0; i < 3; i++) {
        a_store.push_back(create_random_tensor<double>(fmt::format("a0{}", i), 4, k));
        b_store.push_back(create_random_tensor<double>(fmt::format("b0{}", i), k, 5));
    }
    for (int i = 0; i < 2; i++) {
        a_store.push_back(create_random_tensor<double>(fmt::format("a1{}", i), 7, k));
        b_store.push_back(create_random_tensor<double>(fmt::format("b1{}", i), k, 3));
    }

    std::vector<Tensor<double, 2> const *> a_list, b_list;
    for (size_t i = 0; i < a_store.size(); i++) {
        a_list.push_back(&a_store[i]);
        b_list.push_back(&b_store[i]);
        bases.push_back(i < 3 ? &base0 : &base1);
        // Column-major: block `slot` of a (rows, cols*N) tensor starts at
        // slot * cols * rows and inherits the parent's leading dimension.
        offsets.push_back(i < 3 ? i * 5 * 4 : (i - 3) * 3 * 7);
    }

    cg::Graph graph("grouped_blocked");
    {
        cg::CaptureGuard const guard(graph);
        cg::grouped_batched_gemm_blocked(1.0, a_list, b_list, 0.0, bases, offsets);
    }

    REQUIRE(graph.num_nodes() == 1);
    REQUIRE(graph.nodes()[0].kind == cg::OpKind::GroupedBatchedGemm);
    // One output per DISTINCT base, which is what orders a later read of the
    // whole base against this write.
    REQUIRE(graph.nodes()[0].outputs.size() == 2);

    auto const *d = std::get_if<cg::GroupedBatchedGemmDescriptor>(&graph.nodes()[0].op_data);
    REQUIRE(d != nullptr);
    REQUIRE(d->groups.size() == 2);
    REQUIRE(d->total == 5);

    graph.execute();

    for (size_t i = 0; i < a_store.size(); i++) {
        INFO("member " << i);
        auto const  rows = a_store[i].dim(0);
        auto const  cols = b_store[i].dim(1);
        auto const *got  = (i < 3 ? base0 : base1).data() + offsets[i];
        for (size_t c = 0; c < cols; c++) {
            for (size_t r = 0; r < rows; r++) {
                double ref = 0.0;
                for (size_t p = 0; p < k; p++) {
                    ref += a_store[i](r, p) * b_store[i](p, c);
                }
                REQUIRE(std::abs(got[r + c * rows] - ref) < 1e-11);
            }
        }
    }
}

TEST_CASE("grouped_batched_gemm_blocked: rejects overlapping blocks of one base", "[ComputeGraph][GroupedBatchedGemm]") {
    // Merging two blocked batches whose blocks collide is the same silent race
    // as two members sharing a destination tensor, one indirection further away.
    auto base = create_zero_tensor<double>("base", 4, 10);
    auto a0   = create_random_tensor<double>("a0", 4, 6);
    auto b0   = create_random_tensor<double>("b0", 6, 5);
    auto a1   = create_random_tensor<double>("a1", 4, 6);
    auto b1   = create_random_tensor<double>("b1", 6, 5);

    std::vector<Tensor<double, 2> const *> const a_list{&a0, &a1};
    std::vector<Tensor<double, 2> const *> const b_list{&b0, &b1};
    std::vector<Tensor<double, 2> *> const       bases{&base, &base};

    // Columns [0,5) and [4,9): one column of overlap.
    std::vector<size_t> const bad{0, 4 * 4};
    REQUIRE_THROWS_AS(cg::grouped_batched_gemm_blocked(1.0, a_list, b_list, 0.0, bases, bad), std::invalid_argument);

    // Columns [0,5) and [5,10): adjacent, which is the layout the overlap build
    // actually uses and must keep working.
    std::vector<size_t> const good{0, 5 * 4};
    REQUIRE_NOTHROW(cg::grouped_batched_gemm_blocked(1.0, a_list, b_list, 0.0, bases, good));
}

TEST_CASE("grouped_batched_gemm: group profiling computes the same answer", "[ComputeGraph][GroupedBatchedGemm]") {
    // `einsums:graph:profile-groups` deliberately runs a DIFFERENT execution -
    // one uniform call per group under its own profiler zone, so the per-shape
    // breakdown is recoverable. Different timing is the point; a different
    // answer would be a bug, and this is the only thing that would catch it.
    auto batch = mixed_batch<double>(true, true);
    auto ref   = reference<double>(batch, 2.0, -1.0, true, true);

    auto      &gc   = GlobalConfigMap::get_singleton();
    bool const prev = gc.get_bool("graph-profile-groups", false);
    gc.set_bool("graph-profile-groups", true);

    auto      l = lists_of(batch);
    cg::Graph graph("grouped_profiled");
    {
        cg::CaptureGuard const guard(graph);
        cg::grouped_batched_gemm(2.0, l.a, l.b, -1.0, l.c, true, true);
    }
    graph.execute();
    gc.set_bool("graph-profile-groups", prev);

    for (size_t i = 0; i < batch.size(); i++) {
        INFO("member " << i);
        require_close(batch[i].c, ref[i]);
    }
}

TEST_CASE("grouped_batched_gemm: rejects malformed batches", "[ComputeGraph][GroupedBatchedGemm]") {
    auto batch = mixed_batch<double>(false, false);
    auto l     = lists_of(batch);

    SECTION("empty") {
        std::vector<Tensor<double, 2> const *> const none_a, none_b;
        std::vector<Tensor<double, 2> *> const       none_c;
        REQUIRE_THROWS_AS(cg::grouped_batched_gemm(1.0, none_a, none_b, 0.0, none_c), std::invalid_argument);
    }

    SECTION("mismatched list lengths") {
        auto shorter = l.b;
        shorter.pop_back();
        REQUIRE_THROWS_AS(cg::grouped_batched_gemm(1.0, l.a, shorter, 0.0, l.c), std::invalid_argument);
    }

    SECTION("null operand") {
        auto holed = l.a;
        holed[2]   = nullptr;
        REQUIRE_THROWS_AS(cg::grouped_batched_gemm(1.0, holed, l.b, 0.0, l.c), std::invalid_argument);
    }

    SECTION("two members sharing a destination") {
        // The mistake this entry point makes newly reachable: merging two calls
        // that accumulate into the same C. They would race, silently.
        auto shared = l.c;
        shared[4]   = shared[1];
        REQUIRE_THROWS_AS(cg::grouped_batched_gemm(1.0, l.a, l.b, 1.0, shared), std::invalid_argument);
    }

    SECTION("a member whose own operands do not compose") {
        // B's link dimension disagrees with A's. The uniform form catches this
        // by comparing against member 0; here there is nothing to compare
        // against, so the check has to be internal to the member.
        auto wrong = create_random_tensor<double>("wrong", 3, 5);
        auto bad   = l.b;
        bad[1]     = &wrong;
        REQUIRE_THROWS_AS(cg::grouped_batched_gemm(1.0, l.a, bad, 0.0, l.c), std::invalid_argument);
    }
}
