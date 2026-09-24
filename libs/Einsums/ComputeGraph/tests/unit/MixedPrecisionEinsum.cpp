//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

// cg::einsum over operands of different element types, checked against the brute-force reference.

#include <Einsums/ComputeGraph.hpp>
#include <Einsums/ComputeGraph/Passes/ScratchPrivatization.hpp>
#include <Einsums/Tensor/RuntimeTensor.hpp>
#include <Einsums/TensorUtilities/CreateRandomTensor.hpp>
#include <Einsums/TensorUtilities/CreateZeroTensor.hpp>
#include <Einsums/Testing/ReferenceEinsum.hpp>

#include <cmath>
#include <complex>
#include <stdexcept>
#include <tuple>
#include <type_traits>

#include <Einsums/Testing.hpp>

using namespace einsums;
namespace cg = einsums::compute_graph;
using einsums::testing::reference_einsum;

namespace {

template <typename T>
constexpr bool single_precision = std::is_same_v<T, float> || std::is_same_v<T, std::complex<float>>;

/// Tolerance: a float anywhere in the chain limits the agreement to a float's precision.
template <typename TC, typename TA, typename TB>
constexpr double tolerance = (single_precision<TC> || single_precision<TA> || single_precision<TB>) ? 1.0e-5 : 1.0e-12;

template <typename TC, typename TA, typename TB, typename CT>
void check_against(CT const &got, CT const &want) {
    for (size_t n = 0; n < want.size(); ++n) {
        CAPTURE(n);
        CHECK(std::abs(got.data()[n] - want.data()[n]) <= tolerance<TC, TA, TB> * (1.0 + std::abs(want.data()[n])));
    }
}

} // namespace

// Every combination the rules allow whose types are not all the same: 36 of them. A complex
// operand with a real output is the combination left out, because it does not compile.
using MixedTriples =
    std::tuple<std::tuple<float, float, double>, std::tuple<float, double, float>, std::tuple<float, double, double>,
               std::tuple<double, float, float>, std::tuple<double, float, double>, std::tuple<double, double, float>,
               std::tuple<std::complex<float>, float, float>, std::tuple<std::complex<float>, float, double>,
               std::tuple<std::complex<float>, float, std::complex<float>>, std::tuple<std::complex<float>, float, std::complex<double>>,
               std::tuple<std::complex<float>, double, float>, std::tuple<std::complex<float>, double, double>,
               std::tuple<std::complex<float>, double, std::complex<float>>, std::tuple<std::complex<float>, double, std::complex<double>>,
               std::tuple<std::complex<float>, std::complex<float>, float>, std::tuple<std::complex<float>, std::complex<float>, double>,
               std::tuple<std::complex<float>, std::complex<float>, std::complex<double>>,
               std::tuple<std::complex<float>, std::complex<double>, float>, std::tuple<std::complex<float>, std::complex<double>, double>,
               std::tuple<std::complex<float>, std::complex<double>, std::complex<float>>,
               std::tuple<std::complex<float>, std::complex<double>, std::complex<double>>, std::tuple<std::complex<double>, float, float>,
               std::tuple<std::complex<double>, float, double>, std::tuple<std::complex<double>, float, std::complex<float>>,
               std::tuple<std::complex<double>, float, std::complex<double>>, std::tuple<std::complex<double>, double, float>,
               std::tuple<std::complex<double>, double, double>, std::tuple<std::complex<double>, double, std::complex<float>>,
               std::tuple<std::complex<double>, double, std::complex<double>>, std::tuple<std::complex<double>, std::complex<float>, float>,
               std::tuple<std::complex<double>, std::complex<float>, double>,
               std::tuple<std::complex<double>, std::complex<float>, std::complex<float>>,
               std::tuple<std::complex<double>, std::complex<float>, std::complex<double>>,
               std::tuple<std::complex<double>, std::complex<double>, float>,
               std::tuple<std::complex<double>, std::complex<double>, double>,
               std::tuple<std::complex<double>, std::complex<double>, std::complex<float>>>;

TEMPLATE_LIST_TEST_CASE("mixed precision einsum - every allowed combination, eagerly", "[ComputeGraph][MixedPrecision]", MixedTriples) {
    using TC = std::tuple_element_t<0, TestType>;
    using TA = std::tuple_element_t<1, TestType>;
    using TB = std::tuple_element_t<2, TestType>;
    using TR = cg::detail::PromoteT<TA, TB>;

    auto const A        = create_random_tensor<TA>("A", 4, 5);
    auto const B        = create_random_tensor<TB>("B", 5, 3);
    auto       C        = create_random_tensor<TC>("C", 4, 3);
    auto       expected = C;

    cg::einsum("ij <- ik ; kj", TC{0.5}, &C, TR{1.5}, A, B);
    CHECK(std::string(cg::dispatch::last_dispatch_route()) == "generic_loop_mixed_precision");

    reference_einsum("ij <- ik ; kj", TC{0.5}, &expected, TR{1.5}, A, B);
    check_against<TC, TA, TB>(C, expected);
}

TEST_CASE("mixed precision einsum - complex<float> times double keeps both the imaginary part and the precision",
          "[ComputeGraph][MixedPrecision]") {
    // The pair has no operator* between them, and BiggestTypeT would name double for it. The
    // accumulator here is complex<double>.
    using cf = std::complex<float>;
    using cd = std::complex<double>;
    auto a   = create_zero_tensor<cf>("a", 2);
    auto b   = create_zero_tensor<double>("b", 2);
    auto c   = create_zero_tensor<cd>("c", 1);
    a(0)     = cf{1.0f, 2.0f};
    a(1)     = cf{3.0f, -1.0f};
    b(0)     = 1.0 / 3.0;
    b(1)     = 2.0;
    cg::einsum(" <- i ; i", &c, a, b);
    CHECK(c(0).real() == Catch::Approx(1.0 / 3.0 + 6.0).epsilon(1e-15));
    CHECK(c(0).imag() == Catch::Approx(2.0 / 3.0 - 2.0).epsilon(1e-15));
}

TEST_CASE("mixed precision einsum - conjugation, runtime-rank operands and other specs", "[ComputeGraph][MixedPrecision]") {
    using cd = std::complex<double>;
    using cf = std::complex<float>;

    SECTION("conjugating a complex operand, and a no-op on a real one") {
        auto const A        = create_random_tensor<cf>("A", 3, 4);
        auto const B        = create_random_tensor<double>("B", 4, 2);
        auto       C        = create_zero_tensor<cd>("C", 3, 2);
        auto       expected = C;
        cg::einsum("ij <- ik ; kj", cd{0}, &C, cd{1}, A, B, /*conj_a=*/true, /*conj_b=*/true);
        reference_einsum("ij <- ik ; kj", cd{0}, &expected, cd{1}, A, B, true, true);
        check_against<cd, cf, double>(C, expected);
    }

    SECTION("runtime-rank operands") {
        RuntimeTensor<double> A        = create_random_tensor<double>("A", {3, 4});
        RuntimeTensor<float>  B        = create_random_tensor<float>("B", {4, 5});
        RuntimeTensor<double> C        = create_zero_tensor<double>("C", {3, 5});
        auto                  expected = C;
        cg::einsum("ij <- ik ; kj", &C, A, B);
        reference_einsum("ij <- ik ; kj", &expected, A, B);
        check_against<double, double, float>(C, expected);
    }

    SECTION("an outer product, a diagonal and a lone summed letter") {
        auto const A = create_random_tensor<float>("A", 3, 3);
        auto const B = create_random_tensor<double>("B", 3, 4);
        auto const x = create_random_tensor<double>("x", 3);
        for (auto spec : {"ijk <- ij ; k", "i <- ii ; i", "i <- ij ; i"}) {
            CAPTURE(spec);
            std::string const s = spec;
            if (s == "ijk <- ij ; k") {
                auto C = create_zero_tensor<double>("C", 3, 3, 3), expected = C;
                cg::einsum(cg::EinsumFormatString(s), &C, A, x);
                reference_einsum(s, &expected, A, x);
                check_against<double, float, double>(C, expected);
            } else {
                auto C = create_zero_tensor<double>("C", 3), expected = C;
                cg::einsum(cg::EinsumFormatString(s), &C, A, x);
                reference_einsum(s, &expected, A, x);
                check_against<double, float, double>(C, expected);
            }
        }
    }
}

TEST_CASE("mixed precision einsum - the rules string_einsum enforces still hold", "[ComputeGraph][MixedPrecision]") {
    SECTION("an empty contraction scales C once") {
        auto const A = create_zero_tensor<float>("A", 3, 0);
        auto const B = create_zero_tensor<double>("B", 0, 2);
        auto       C = create_zero_tensor<double>("C", 3, 2);
        C.set_all(4.0);
        cg::einsum("ij <- ik ; kj", 0.5, &C, 1.0, A, B);
        CHECK(C(2, 1) == 2.0);
        CHECK(std::string(cg::dispatch::last_dispatch_route()) == "empty_input_scale_only");
    }

    SECTION("an output overlapping an input is rejected") {
        auto C = create_random_tensor<double>("C", 4, 4);
        auto B = create_random_tensor<float>("B", 4, 4);
        CHECK_THROWS_AS(cg::einsum("ij <- ik ; kj", &C, C, B), std::invalid_argument);
    }

    SECTION("permutation operators are not supported") {
        auto const A = create_random_tensor<float>("A", 3, 3);
        auto const B = create_random_tensor<double>("B", 3, 3);
        auto       C = create_zero_tensor<double>("C", 3, 3);
        CHECK_THROWS_AS(cg::einsum("ij <- P(ij) ik ; kj", &C, A, B), std::invalid_argument);
    }
}

// ── Captured ────────────────────────────────────────────────────────────────

namespace {

/// Overwrite @p T with fresh random values in place, so a captured graph that holds its storage
/// sees them on the next replay.
template <typename T>
void refill(Tensor<T, 2> &target) {
    auto const fresh = create_random_tensor<T>("fresh", target.dim(0), target.dim(1));
    for (size_t i = 0; i < target.dim(0); ++i) {
        for (size_t j = 0; j < target.dim(1); ++j) {
            target(i, j) = fresh(i, j);
        }
    }
}

cg::ParsedEinsumSpec gemm_spec() {
    cg::ParsedEinsumSpec s;
    s.c_indices = {"i", "j"};
    s.a_indices = {"i", "k"};
    s.b_indices = {"k", "j"};
    s.raw       = "i,j <- i,k ; k,j";
    return s;
}

} // namespace

TEMPLATE_LIST_TEST_CASE("mixed precision einsum - every allowed combination, captured and replayed", "[ComputeGraph][MixedPrecision]",
                        MixedTriples) {
    using TC = std::tuple_element_t<0, TestType>;
    using TA = std::tuple_element_t<1, TestType>;
    using TB = std::tuple_element_t<2, TestType>;
    using TR = cg::detail::PromoteT<TA, TB>;

    auto A = create_random_tensor<TA>("A", 4, 5);
    auto B = create_random_tensor<TB>("B", 5, 3);
    auto C = create_random_tensor<TC>("C", 4, 3);

    cg::Graph graph("mixed");
    {
        cg::CaptureGuard const guard(graph);
        cg::einsum("ij <- ik ; kj", TC{0.5}, &C, TR{1.5}, A, B);
    }

    // The second replay reads new inputs: the node reads its operands live rather than baking them.
    for (int round = 0; round < 2; ++round) {
        CAPTURE(round);
        if (round == 1) {
            refill(A);
            refill(B);
        }
        auto expected = C;
        graph.execute();
        reference_einsum("ij <- ik ; kj", TC{0.5}, &expected, TR{1.5}, A, B);
        check_against<TC, TA, TB>(C, expected);
    }
}

TEST_CASE("mixed precision einsum - capture rejects the permutation operators", "[ComputeGraph][MixedPrecision]") {
    auto const             A = create_random_tensor<float>("A", 3, 3);
    auto const             B = create_random_tensor<double>("B", 3, 3);
    auto                   C = create_zero_tensor<double>("C", 3, 3);
    cg::Graph              graph("mixed");
    cg::CaptureGuard const guard(graph);
    CHECK_THROWS_AS(cg::einsum("ij <- P(ij) ik ; kj", &C, A, B), std::invalid_argument);
}

TEST_CASE("mixed precision einsum - a node a pass builds", "[ComputeGraph][MixedPrecision]") {
    // Passes synthesize contractions through Graph::make_einsum_node, which used to throw whenever
    // the operands' element types differed.
    RuntimeTensor<float>  A(create_random_tensor<float>("A", 3, 4));
    RuntimeTensor<double> B(create_random_tensor<double>("B", 4, 2));
    RuntimeTensor<double> C("C", std::vector<size_t>{3, 2});
    C.zero();

    cg::Graph  graph("pass_built");
    auto const a_id = graph.register_tensor(cg::make_handle(A, 0));
    auto const b_id = graph.register_tensor(cg::make_handle(B, 0));
    auto const c_id = graph.register_tensor(cg::make_handle(C, 0));

    SECTION("contracts") {
        auto node = graph.make_einsum_node(a_id, b_id, c_id, gemm_spec(), 0.0, 2.0);
        // No BLAS batching hint: GEMMBatching would fold the node into a GEMM that reads every
        // operand as C's type. Deriving one used to read A's floats as doubles.
        auto const *desc = std::get_if<cg::EinsumDescriptor>(&node.op_data);
        REQUIRE(desc != nullptr);
        CHECK(desc->gemm_hint == nullptr);
        graph.add_node(std::move(node));
        graph.execute();

        RuntimeTensor<double> expected("E", std::vector<size_t>{3, 2});
        expected.zero();
        reference_einsum("ij <- ik ; kj", 0.0, &expected, 2.0, A, B);
        check_against<double, float, double>(C, expected);
    }

    SECTION("rejects a complex prefactor into a real output") {
        CHECK_THROWS_AS(graph.make_einsum_node(a_id, b_id, c_id, gemm_spec(), 0.0, std::complex<double>{1.0, 1.0}), std::invalid_argument);
    }

    SECTION("rejects a complex contraction into a real output") {
        RuntimeTensor<std::complex<float>> Z(create_random_tensor<std::complex<float>>("Z", 3, 4));
        auto const                         z_id = graph.register_tensor(cg::make_handle(Z, 0));
        CHECK_THROWS_AS(graph.make_einsum_node(z_id, b_id, c_id, gemm_spec(), 0.0, 1.0), std::invalid_argument);
    }

    SECTION("rejects the permutation operators") {
        auto spec      = gemm_spec();
        spec.operators = cg::parse_einsum_spec("ij <- P(ij) ik ; kj").value().operators;
        CHECK_THROWS_AS(graph.make_einsum_node(a_id, b_id, c_id, spec, 0.0, 1.0), std::invalid_argument);
    }
}

TEST_CASE("an executor that reads an operand as the wrong type fails instead of reinterpreting it", "[ComputeGraph][MixedPrecision]") {
    // OperandAccessor::impl<T>() was an unchecked cast: an executor built for the wrong element type
    // read the tensor's bytes as that type and computed garbage without a word.
    RuntimeTensor<double> D(create_random_tensor<double>("D", 2, 2));
    cg::Graph             graph("accessor");
    auto const            id       = graph.register_tensor(cg::make_handle(D, 0));
    auto const            accessor = cg::resolve_operand(graph, id, "test", "D");
    CHECK(accessor.dtype() == einsums::packed_gemm::ScalarType::Float64);
    CHECK_NOTHROW(accessor.impl<double>());
    CHECK_THROWS_AS(accessor.impl<float>(), std::logic_error);
}

TEST_CASE("redirect_slot refuses tensors of different element types", "[ComputeGraph][MixedPrecision]") {
    // An executor reads a redirected operand as the type it was built for, and its accessor recorded
    // that type before the redirect, so nothing after it would notice the bytes are another type's.
    RuntimeTensor<float>  F(create_random_tensor<float>("F", 2, 2));
    RuntimeTensor<double> D(create_random_tensor<double>("D", 2, 2));
    RuntimeTensor<double> E(create_random_tensor<double>("E", 2, 2));
    cg::Graph             graph("redirect");
    auto const            f_id = graph.register_tensor(cg::make_handle(F, 0));
    auto const            d_id = graph.register_tensor(cg::make_handle(D, 0));
    auto const            e_id = graph.register_tensor(cg::make_handle(E, 0));
    CHECK_THROWS_AS(graph.redirect_slot(d_id, f_id), std::logic_error);
    CHECK_NOTHROW(graph.redirect_slot(d_id, e_id));
}

TEST_CASE("mixed precision einsum - scratch privatization rebuilds a mixed node onto its clone", "[ComputeGraph][MixedPrecision]") {
    // ScratchPrivatization renames each interior write->read episode of a reused scratch onto a
    // clone and rebuilds the nodes through Graph::make_einsum_node, which used to throw on operands
    // of different types. The clone takes the scratch's own type, so the rebuilt node is the same
    // mixed contraction. The shape of test_privatization_splits_generations, with float operands.
    std::vector<RuntimeTensor<float>> ops;
    for (int k = 0; k < 3; ++k) {
        ops.emplace_back(create_random_tensor<float>("A", 12, 12));
    }
    RuntimeTensor<double> B(create_random_tensor<double>("B", 12, 12));
    RuntimeTensor<double> acc("acc", std::vector<size_t>{12, 12});
    acc.zero();

    cg::Graph graph("privatize_mixed");
    auto     &tmp = graph.declare_zero_runtime_tensor<double>("tmp", {12, 12}, true);
    {
        cg::CaptureGuard const guard(graph);
        for (auto const &A : ops) {
            cg::einsum("ij <- ik ; kj", 0.0, &tmp, 1.0, A, B);
            cg::axpby(1.0, tmp, 1.0, &acc);
        }
    }

    auto pass = std::make_shared<cg::passes::ScratchPrivatization>();
    pass->set_require_executor(false);
    cg::PassManager manager;
    manager.add(pass);
    graph.apply(manager);
    CHECK(pass->num_tensors_privatized() == 1);
    CHECK(pass->num_copies_created() == 2);
    CHECK(pass->num_nodes_rebuilt() == 4);

    graph.optimize(); // the default pipeline, which materializes the clones
    graph.execute();

    RuntimeTensor<double> expected("expected", std::vector<size_t>{12, 12});
    expected.zero();
    for (auto const &A : ops) {
        reference_einsum("ij <- ik ; kj", 1.0, &expected, 1.0, A, B);
    }
    check_against<double, float, double>(acc, expected);
}
