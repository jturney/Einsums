//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

// LayoutAssignment: storage order chosen so contractions read their operands flat.
//
// The shape every test below is built on, and why it is the one:
//
//   W[i,j,x] = A[i,k]   B[k,x,j]      // producer: C's free groups disagree with B's
//   R[i,x,y] = W[i,j,x] D[j,y]        // consumer: W's contracted letter sits BETWEEN its free ones
//
// As captured that costs two operand copies per replay - B at the producer, W at the consumer -
// and neither is visible as a node, because both happen inside the contraction kernel. Storing W
// as (i,x,j) removes both at once, which is what makes this a layout question rather than two
// unrelated ones: no local rewrite of either contraction can see the other's cost.

#include <Einsums/ComputeGraph.hpp>
#include <Einsums/ComputeGraph/Passes/LayoutAssignment.hpp>
#include <Einsums/Tensor/RuntimeTensor.hpp>
#include <Einsums/Tensor/Tensor.hpp>
#include <Einsums/TensorUtilities/CreateRandomTensor.hpp>
#include <Einsums/TensorUtilities/CreateZeroTensor.hpp>
#include <Einsums/Testing/ReferenceEinsum.hpp>

#include <cmath>
#include <complex>
#include <limits>
#include <memory>
#include <string>
#include <string_view>
#include <vector>

#include <Einsums/Testing.hpp>

using namespace einsums;
namespace cg = einsums::compute_graph;

namespace {

constexpr size_t kI = 6, kJ = 5, kX = 7, kK = 3, kY = 4;

/// The two-node chain of the file note, with `W` declared in @p w_dims order.
/// The caller supplies the letters so a test can capture the same arithmetic under a different
/// storage order without duplicating the builder.
template <typename T>
void capture_chain(cg::Graph &graph, RuntimeTensor<T> &R, RuntimeTensor<T> const &A, RuntimeTensor<T> const &B, RuntimeTensor<T> const &D,
                   std::string const &w_spec, size_t d0, size_t d1, size_t d2) {
    auto &W = graph.declare_runtime_tensor<T>("W", {d0, d1, d2}, /*intermediate=*/true);

    // Built at run time so one builder serves both storage orders, which means the runtime
    // string_view constructor rather than the consteval literal one.
    std::string const produce = w_spec + " <- i,k ; k,x,j";
    std::string const consume = "i,x,y <- " + w_spec + " ; j,y";

    cg::CaptureGuard const guard(graph);
    cg::einsum(std::string_view{produce}, 0.0, &W, 1.0, A, B);
    cg::einsum(std::string_view{consume}, 0.0, &R, 1.0, W, D);
}

/// @brief The norm-relative gap between two results of the same program.
///
/// What a re-associating pass promises to stay inside, and deliberately NOT an element-wise
/// relative check. A near-cancellation element is small against the tensor's norm, so its
/// RELATIVE error is large while the answer as a whole is fine, and it is exactly that element
/// which fails first when a different toolchain orders the fused multiply-adds differently. The
/// element-wise form of this test passed on two platforms and failed on Windows at 2.8e-12, on an
/// element of magnitude 3.5e-4, for a rewrite that was behaving.
template <typename T>
double norm_relative_gap(RuntimeTensor<T> const &got, RuntimeTensor<T> const &want) {
    REQUIRE(got.size() == want.size());
    double error = 0.0, reference = 0.0;
    for (size_t i = 0; i < want.size(); i++) {
        error += std::norm(got.data()[i] - want.data()[i]);
        reference += std::norm(want.data()[i]);
    }
    return reference > 0.0 ? std::sqrt(error) / std::sqrt(reference) : std::sqrt(error);
}

/// @brief The bound this tier declares, which is the number the pass is validated against.
template <typename T>
double re_associating_bound() {
    return cg::tier_bound(cg::PassTier::ReAssociating, std::numeric_limits<RemoveComplexT<T>>::epsilon());
}

std::shared_ptr<cg::passes::LayoutAssignment> only_layout(cg::Graph &graph) {
    cg::PassManager pm;
    auto            pass = std::make_shared<cg::passes::LayoutAssignment>();
    pm.add(pass);
    graph.apply(pm);
    return pass;
}

/// The index lists of the @p which-th einsum node, as (C, A, B).
std::vector<std::vector<std::string>> spec_of(cg::Graph const &graph, size_t which) {
    size_t seen = 0;
    for (auto const &node : graph.nodes()) {
        if (node.kind != cg::OpKind::Einsum) {
            continue;
        }
        if (seen++ != which) {
            continue;
        }
        auto const *desc = node.op_data.get_if<cg::EinsumDescriptor>();
        REQUIRE(desc != nullptr);
        return {desc->spec.c_indices, desc->spec.a_indices, desc->spec.b_indices};
    }
    FAIL("no einsum node at that position");
    return {};
}

/// Run the chain to completion and hand back R.
template <typename T>
RuntimeTensor<T> run_chain(std::string const &w_spec, size_t d0, size_t d1, size_t d2, RuntimeTensor<T> const &A, RuntimeTensor<T> const &B,
                           RuntimeTensor<T> const &D, bool with_layout) {
    RuntimeTensor<T> R = create_zero_tensor<T>("R", kI, kX, kY);
    cg::Graph        graph("chain");
    capture_chain(graph, R, A, B, D, w_spec, d0, d1, d2);

    cg::PassManager pm;
    if (with_layout) {
        pm.add<cg::passes::LayoutAssignment>();
    }
    pm.add<cg::passes::Materialization>();
    graph.apply(pm);
    graph.execute();
    return R;
}

} // namespace

TEMPLATE_LIST_TEST_CASE("LayoutAssignment - both contractions are made to read flat", "[ComputeGraph][LayoutAssignment]",
                        testing::AllScalarTypes) {
    using T            = TestType;
    RuntimeTensor<T> A = create_random_tensor<T>("A", kI, kK);
    RuntimeTensor<T> B = create_random_tensor<T>("B", kK, kX, kJ);
    RuntimeTensor<T> D = create_random_tensor<T>("D", kJ, kY);
    RuntimeTensor<T> R = create_zero_tensor<T>("R", kI, kX, kY);

    cg::Graph graph("flat");
    capture_chain(graph, R, A, B, D, "i,j,x", kI, kJ, kX);

    auto pass = only_layout(graph);

    CHECK(pass->num_relaid_out() == 1);
    // Both copies, not one: the producer's B and the consumer's W. A pass that only looked at
    // the consumer would find (i,x,j) too and report one.
    CHECK(pass->num_copies_removed() == 2);
    CHECK(pass->estimated_saving_us() > 0.0);

    // W is stored (i,x,j) now, and BOTH index lists say so.
    CHECK(spec_of(graph, 0)[0] == std::vector<std::string>{"i", "x", "j"});
    CHECK(spec_of(graph, 1)[1] == std::vector<std::string>{"i", "x", "j"});
    // Nothing else moved: A, B, D and R are the caller's.
    CHECK(spec_of(graph, 0)[1] == std::vector<std::string>{"i", "k"});
    CHECK(spec_of(graph, 0)[2] == std::vector<std::string>{"k", "x", "j"});
    CHECK(spec_of(graph, 1)[0] == std::vector<std::string>{"i", "x", "y"});
    CHECK(spec_of(graph, 1)[2] == std::vector<std::string>{"j", "y"});

    // The tensor itself, not just the index lists that name it.
    cg::TensorHandle const *w = nullptr;
    for (auto const &[id, handle] : graph.tensors_map()) {
        if (handle.name == "W") {
            w = &handle;
        }
    }
    REQUIRE(w != nullptr);
    CHECK(w->dims == std::vector<size_t>{kI, kX, kJ});
}

TEMPLATE_LIST_TEST_CASE("LayoutAssignment - the answer does not change", "[ComputeGraph][LayoutAssignment]", testing::AllScalarTypes) {
    using T            = TestType;
    RuntimeTensor<T> A = create_random_tensor<T>("A", kI, kK);
    RuntimeTensor<T> B = create_random_tensor<T>("B", kK, kX, kJ);
    RuntimeTensor<T> D = create_random_tensor<T>("D", kJ, kY);

    auto const plain  = run_chain("i,j,x", kI, kJ, kX, A, B, D, /*with_layout=*/false);
    auto const relaid = run_chain("i,j,x", kI, kJ, kX, A, B, D, /*with_layout=*/true);
    // And, as a third opinion that the model is describing the same arithmetic rather than the
    // same bug twice: the chain captured directly in the order the pass chose.
    auto const by_hand = run_chain("i,x,j", kI, kX, kJ, A, B, D, /*with_layout=*/false);

    // Norm-relative against the tier's own bound; see norm_relative_gap for why an element-wise
    // relative check is the wrong instrument for a pass in this tier.
    CHECK(norm_relative_gap(relaid, plain) <= re_associating_bound<T>());
    CHECK(norm_relative_gap(by_hand, plain) <= re_associating_bound<T>());

    // And against the naive oracle, so the three agreeing is not the engine agreeing with itself.
    RuntimeTensor<T> W = create_zero_tensor<T>("W", kI, kJ, kX);
    RuntimeTensor<T> R = create_zero_tensor<T>("R", kI, kX, kY);
    testing::reference_einsum("i,j,x <- i,k ; k,x,j", &W, A, B);
    testing::reference_einsum("i,x,y <- i,j,x ; j,y", &R, W, D);
    CHECK(norm_relative_gap(plain, R) <= re_associating_bound<T>());
}

TEMPLATE_LIST_TEST_CASE("LayoutAssignment - the chosen order is already the best one", "[ComputeGraph][LayoutAssignment]",
                        testing::AllScalarTypes) {
    using T            = TestType;
    RuntimeTensor<T> A = create_random_tensor<T>("A", kI, kK);
    RuntimeTensor<T> B = create_random_tensor<T>("B", kK, kX, kJ);
    RuntimeTensor<T> D = create_random_tensor<T>("D", kJ, kY);
    RuntimeTensor<T> R = create_zero_tensor<T>("R", kI, kX, kY);

    cg::Graph graph("already-flat");
    capture_chain(graph, R, A, B, D, "i,x,j", kI, kX, kJ);

    auto pass = only_layout(graph);
    CHECK(pass->num_relaid_out() == 0);
    CHECK(pass->explain().empty());
}

TEMPLATE_LIST_TEST_CASE("LayoutAssignment - a caller's tensor keeps the caller's axes", "[ComputeGraph][LayoutAssignment]",
                        testing::AllScalarTypes) {
    using T            = TestType;
    RuntimeTensor<T> A = create_random_tensor<T>("A", kI, kK);
    RuntimeTensor<T> B = create_random_tensor<T>("B", kK, kX, kJ);
    RuntimeTensor<T> D = create_random_tensor<T>("D", kJ, kY);
    RuntimeTensor<T> R = create_zero_tensor<T>("R", kI, kX, kY);
    // The same chain, except W is the caller's tensor rather than the graph's. Its axis order is
    // part of what the caller asked for, so the copies stay.
    RuntimeTensor<T> W = create_zero_tensor<T>("W", kI, kJ, kX);

    cg::Graph graph("user-owned");
    {
        cg::CaptureGuard const guard(graph);
        cg::einsum("i,j,x <- i,k ; k,x,j", 0.0, &W, 1.0, A, B);
        cg::einsum("i,x,y <- i,j,x ; j,y", 0.0, &R, 1.0, W, D);
    }

    auto pass = only_layout(graph);
    CHECK(pass->num_relaid_out() == 0);
    CHECK(spec_of(graph, 0)[0] == std::vector<std::string>{"i", "j", "x"});
}

TEMPLATE_LIST_TEST_CASE("LayoutAssignment - rank two is left alone", "[ComputeGraph][LayoutAssignment]", testing::AllScalarTypes) {
    using T            = TestType;
    RuntimeTensor<T> A = create_random_tensor<T>("A", kI, kK);
    RuntimeTensor<T> B = create_random_tensor<T>("B", kK, kJ);
    RuntimeTensor<T> D = create_random_tensor<T>("D", kJ, kY);
    RuntimeTensor<T> R = create_zero_tensor<T>("R", kI, kY);

    cg::Graph graph("rank-two");
    {
        auto                  &Tt = graph.declare_runtime_tensor<T>("T", {kI, kJ}, /*intermediate=*/true);
        cg::CaptureGuard const guard(graph);
        cg::einsum("i,j <- i,k ; k,j", 0.0, &Tt, 1.0, A, B);
        cg::einsum("i,y <- i,j ; j,y", 0.0, &R, 1.0, Tt, D);
    }

    // A matrix has two readings and BLAS takes either one through `transa`, so there is nothing
    // here for a layout to win - and a GemmHint that a rewrite would have to keep in step.
    auto pass = only_layout(graph);
    CHECK(pass->num_relaid_out() == 0);
}

TEMPLATE_LIST_TEST_CASE("LayoutAssignment - a use it cannot rewrite pins the tensor", "[ComputeGraph][LayoutAssignment]",
                        testing::AllScalarTypes) {
    using T             = TestType;
    RuntimeTensor<T> A  = create_random_tensor<T>("A", kI, kK);
    RuntimeTensor<T> B  = create_random_tensor<T>("B", kK, kX, kJ);
    RuntimeTensor<T> D  = create_random_tensor<T>("D", kJ, kY);
    RuntimeTensor<T> R  = create_zero_tensor<T>("R", kI, kX, kY);
    RuntimeTensor<T> Wc = create_zero_tensor<T>("Wc", kI, kJ, kX);

    cg::Graph graph("pinned");
    {
        auto                  &W = graph.declare_runtime_tensor<T>("W", {kI, kJ, kX}, /*intermediate=*/true);
        cg::CaptureGuard const guard(graph);
        cg::einsum("i,j,x <- i,k ; k,x,j", 0.0, &W, 1.0, A, B);
        cg::einsum("i,x,y <- i,j,x ; j,y", 0.0, &R, 1.0, W, D);
        // A third reader with no index list to rewrite. Moving W's axes underneath it would
        // leave it copying the same bytes into a different meaning.
        cg::axpby(1.0, W, 0.0, &Wc);
    }

    auto pass = only_layout(graph);
    CHECK(pass->num_relaid_out() == 0);
    CHECK(spec_of(graph, 0)[0] == std::vector<std::string>{"i", "j", "x"});
}

TEMPLATE_LIST_TEST_CASE("LayoutAssignment - per-axis annotations follow their axes", "[ComputeGraph][LayoutAssignment]",
                        testing::AllScalarTypes) {
    using T            = TestType;
    RuntimeTensor<T> A = create_random_tensor<T>("A", kI, kK);
    RuntimeTensor<T> B = create_random_tensor<T>("B", kK, kX, kJ);
    RuntimeTensor<T> D = create_random_tensor<T>("D", kJ, kY);
    RuntimeTensor<T> R = create_zero_tensor<T>("R", kI, kX, kY);

    cg::Graph    graph("symbols");
    cg::TensorId w_id{0};
    {
        auto                  &W = graph.declare_runtime_tensor<T>("W", {kI, kJ, kX}, /*intermediate=*/true);
        cg::CaptureGuard const guard(graph);
        cg::einsum("i,j,x <- i,k ; k,x,j", 0.0, &W, 1.0, A, B);
        cg::einsum("i,x,y <- i,j,x ; j,y", 0.0, &R, 1.0, W, D);
        w_id = graph.find_tensor_id_by_ptr(&W);
    }
    REQUIRE(w_id != 0);
    graph.annotate_dims(w_id, {"nI", "nJ", "nX"});

    CHECK(only_layout(graph)->num_relaid_out() == 1);

    // The failure this guards is not a wrong number today: it is a dim symbol naming the wrong
    // extent at the next bind, which is exactly what the symbolic-extent machinery is for.
    cg::TensorHandle const *w = graph.find_tensor(w_id);
    REQUIRE(w != nullptr);
    CHECK(w->dim_symbols == std::vector<std::string>{"nI", "nX", "nJ"});
    CHECK(w->dims == std::vector<size_t>{kI, kX, kJ});
}

TEMPLATE_LIST_TEST_CASE("LayoutAssignment - the same graph gets the same assignment", "[ComputeGraph][LayoutAssignment]",
                        testing::AllScalarTypes) {
    using T            = TestType;
    RuntimeTensor<T> A = create_random_tensor<T>("A", kI, kK);
    RuntimeTensor<T> B = create_random_tensor<T>("B", kK, kX, kJ);
    RuntimeTensor<T> D = create_random_tensor<T>("D", kJ, kY);

    // A search whose answer varies between runs makes every layout downstream of it vary too,
    // which is the Kahn-FIFO lesson applied to a pass that picks one candidate out of several.
    std::vector<std::string> first;
    for (int trial = 0; trial < 3; trial++) {
        RuntimeTensor<T> R = create_zero_tensor<T>("R", kI, kX, kY);
        cg::Graph        graph("determinism");
        capture_chain(graph, R, A, B, D, "i,j,x", kI, kJ, kX);
        only_layout(graph);
        auto const chosen = spec_of(graph, 0)[0];
        if (trial == 0) {
            first = chosen;
        }
        CHECK(chosen == first);
    }
}

// ── Folding an explicit permute away ──────────────────────────────────────────────────────────
//
// The second shape this pass reasons about, and the one where its decision removes a node:
//
//   T[j,x,i] = permute(X[i,j,x])       // an explicit copy, in an order capture chose
//   R1[i,y]  = T[j,x,i] U[j,x,y]
//   R2[i,z]  = T[j,x,i] V[j,x,z]
//
// Storing T the way X is stored makes the permute copy its source into its source's own order,
// which computes nothing, so the node goes and the contractions read X. PermuteFusion declines
// this exact shape because the copy has two readers and a peephole has no way to choose between
// them; the layout question does not care how many there are.

namespace {

constexpr size_t kZ = 2;

size_t count_permutes(cg::Graph const &graph) {
    size_t seen = 0;
    for (auto const &node : graph.nodes()) {
        if (node.kind == cg::OpKind::Permute) {
            seen++;
        }
    }
    return seen;
}

/// The permuted chain above, with @p copy_is_the_callers deciding who owns T.
template <typename T>
void capture_permuted_chain(cg::Graph &graph, RuntimeTensor<T> &R1, RuntimeTensor<T> &R2, RuntimeTensor<T> const &X,
                            RuntimeTensor<T> const &U, RuntimeTensor<T> const &V, RuntimeTensor<T> *callers_copy = nullptr) {
    RuntimeTensor<T> *copy =
        callers_copy != nullptr ? callers_copy : &graph.declare_runtime_tensor<T>("T", {kJ, kX, kI}, /*intermediate=*/true);

    cg::CaptureGuard const guard(graph);
    cg::permute("jxi <- ijx", copy, X);
    cg::einsum("iy <- jxi ; jxy", 0.0, &R1, 1.0, *copy, U);
    cg::einsum("iz <- jxi ; jxz", 0.0, &R2, 1.0, *copy, V);
}

} // namespace

TEMPLATE_LIST_TEST_CASE("LayoutAssignment - a permute the layout choice makes redundant is deleted", "[ComputeGraph][LayoutAssignment]",
                        testing::AllScalarTypes) {
    using T             = TestType;
    RuntimeTensor<T> X  = create_random_tensor<T>("X", kI, kJ, kX);
    RuntimeTensor<T> U  = create_random_tensor<T>("U", kJ, kX, kY);
    RuntimeTensor<T> V  = create_random_tensor<T>("V", kJ, kX, kZ);
    RuntimeTensor<T> R1 = create_zero_tensor<T>("R1", kI, kY);
    RuntimeTensor<T> R2 = create_zero_tensor<T>("R2", kI, kZ);

    cg::Graph graph("fold");
    capture_permuted_chain(graph, R1, R2, X, U, V);
    REQUIRE(count_permutes(graph) == 1);

    auto pass = only_layout(graph);

    CHECK(pass->num_permutes_folded() == 1);
    CHECK(pass->num_relaid_out() == 1);
    CHECK(count_permutes(graph) == 0);
    // Both readers index X directly now, in the order X is stored.
    CHECK(spec_of(graph, 0)[1] == std::vector<std::string>{"i", "j", "x"});
    CHECK(spec_of(graph, 1)[1] == std::vector<std::string>{"i", "j", "x"});

    cg::TensorId const x_id = graph.find_tensor_id_by_ptr(&X);
    REQUIRE(x_id != 0);
    for (auto const &node : graph.nodes()) {
        if (node.kind == cg::OpKind::Einsum) {
            CHECK(node.inputs[0] == x_id);
        }
    }
}

TEMPLATE_LIST_TEST_CASE("LayoutAssignment - the folded chain computes what the captured one did", "[ComputeGraph][LayoutAssignment]",
                        testing::AllScalarTypes) {
    using T            = TestType;
    RuntimeTensor<T> X = create_random_tensor<T>("X", kI, kJ, kX);
    RuntimeTensor<T> U = create_random_tensor<T>("U", kJ, kX, kY);
    RuntimeTensor<T> V = create_random_tensor<T>("V", kJ, kX, kZ);

    auto run = [&](bool with_layout) {
        RuntimeTensor<T> R1 = create_zero_tensor<T>("R1", kI, kY);
        RuntimeTensor<T> R2 = create_zero_tensor<T>("R2", kI, kZ);
        cg::Graph        graph("fold-run");
        capture_permuted_chain(graph, R1, R2, X, U, V);
        cg::PassManager pm;
        if (with_layout) {
            pm.add<cg::passes::LayoutAssignment>();
        }
        pm.add<cg::passes::Materialization>();
        graph.apply(pm);
        graph.execute();
        return std::pair{R1, R2};
    };

    auto const [plain1, plain2]   = run(false);
    auto const [folded1, folded2] = run(true);
    CHECK(norm_relative_gap(folded1, plain1) <= re_associating_bound<T>());
    CHECK(norm_relative_gap(folded2, plain2) <= re_associating_bound<T>());
}

TEMPLATE_LIST_TEST_CASE("LayoutAssignment - a permute that already copies in place is deleted", "[ComputeGraph][LayoutAssignment]",
                        testing::AllScalarTypes) {
    using T            = TestType;
    RuntimeTensor<T> X = create_random_tensor<T>("X", kI, kJ, kX);
    RuntimeTensor<T> U = create_random_tensor<T>("U", kJ, kX, kY);
    RuntimeTensor<T> R = create_zero_tensor<T>("R", kI, kY);

    cg::Graph graph("identity-permute");
    {
        auto                  &Tt = graph.declare_runtime_tensor<T>("T", {kI, kJ, kX}, /*intermediate=*/true);
        cg::CaptureGuard const guard(graph);
        cg::permute("ijx <- ijx", &Tt, X);
        cg::einsum("iy <- ijx ; jxy", 0.0, &R, 1.0, Tt, U);
    }

    auto pass = only_layout(graph);

    // Nothing moves: the copy is already stored the way its source is. The saving is the node.
    CHECK(pass->num_relaid_out() == 0);
    CHECK(pass->num_permutes_folded() == 1);
    CHECK(count_permutes(graph) == 0);
    CHECK(pass->estimated_saving_us() > 0.0);
}

TEMPLATE_LIST_TEST_CASE("LayoutAssignment - a copy the caller owns is not dissolved", "[ComputeGraph][LayoutAssignment]",
                        testing::AllScalarTypes) {
    using T             = TestType;
    RuntimeTensor<T> X  = create_random_tensor<T>("X", kI, kJ, kX);
    RuntimeTensor<T> U  = create_random_tensor<T>("U", kJ, kX, kY);
    RuntimeTensor<T> V  = create_random_tensor<T>("V", kJ, kX, kZ);
    RuntimeTensor<T> R1 = create_zero_tensor<T>("R1", kI, kY);
    RuntimeTensor<T> R2 = create_zero_tensor<T>("R2", kI, kZ);
    // The caller holds T and expects the permuted copy in it, so the node stays even though the
    // contractions would be no worse without it. This is one of the cases PermuteFusion still
    // owns rather than a case both passes decline.
    RuntimeTensor<T> Tt = create_zero_tensor<T>("T", kJ, kX, kI);

    cg::Graph graph("user-copy");
    capture_permuted_chain(graph, R1, R2, X, U, V, &Tt);

    auto pass = only_layout(graph);
    CHECK(pass->num_permutes_folded() == 0);
    CHECK(count_permutes(graph) == 1);
}

TEMPLATE_LIST_TEST_CASE("LayoutAssignment - a source written after the copy pins the permute", "[ComputeGraph][LayoutAssignment]",
                        testing::AllScalarTypes) {
    using T             = TestType;
    RuntimeTensor<T> X  = create_random_tensor<T>("X", kI, kJ, kX);
    RuntimeTensor<T> U  = create_random_tensor<T>("U", kJ, kX, kY);
    RuntimeTensor<T> A  = create_random_tensor<T>("A", kI, kK);
    RuntimeTensor<T> B  = create_random_tensor<T>("B", kK, kJ, kX);
    RuntimeTensor<T> R1 = create_zero_tensor<T>("R1", kI, kY);

    cg::Graph graph("source-rewritten");
    {
        auto                  &Tt = graph.declare_runtime_tensor<T>("T", {kJ, kX, kI}, /*intermediate=*/true);
        cg::CaptureGuard const guard(graph);
        cg::permute("jxi <- ijx", &Tt, X);
        cg::einsum("iy <- jxi ; jxy", 0.0, &R1, 1.0, Tt, U);
        // A copy is a snapshot and its source is not. Reading X where the program reads T would
        // hand the contraction the value written here.
        cg::einsum("ijx <- ik ; kjx", 0.0, &X, 1.0, A, B);
    }

    auto pass = only_layout(graph);
    CHECK(pass->num_permutes_folded() == 0);
    CHECK(count_permutes(graph) == 1);
}

TEMPLATE_LIST_TEST_CASE("LayoutAssignment - a reader with no index list pins the copy", "[ComputeGraph][LayoutAssignment]",
                        testing::AllScalarTypes) {
    using T             = TestType;
    RuntimeTensor<T> X  = create_random_tensor<T>("X", kI, kJ, kX);
    RuntimeTensor<T> U  = create_random_tensor<T>("U", kJ, kX, kY);
    RuntimeTensor<T> R1 = create_zero_tensor<T>("R1", kI, kY);
    RuntimeTensor<T> Tc = create_zero_tensor<T>("Tc", kJ, kX, kI);

    cg::Graph graph("pinned-copy");
    {
        auto                  &Tt = graph.declare_runtime_tensor<T>("T", {kJ, kX, kI}, /*intermediate=*/true);
        cg::CaptureGuard const guard(graph);
        cg::permute("jxi <- ijx", &Tt, X);
        cg::einsum("iy <- jxi ; jxy", 0.0, &R1, 1.0, Tt, U);
        cg::axpby(1.0, Tt, 0.0, &Tc);
    }

    auto pass = only_layout(graph);
    CHECK(pass->num_permutes_folded() == 0);
    CHECK(count_permutes(graph) == 1);
}

TEMPLATE_LIST_TEST_CASE("LayoutAssignment - it reports what it did", "[ComputeGraph][LayoutAssignment]", testing::AllScalarTypes) {
    using T            = TestType;
    RuntimeTensor<T> A = create_random_tensor<T>("A", kI, kK);
    RuntimeTensor<T> B = create_random_tensor<T>("B", kK, kX, kJ);
    RuntimeTensor<T> D = create_random_tensor<T>("D", kJ, kY);
    RuntimeTensor<T> R = create_zero_tensor<T>("R", kI, kX, kY);

    cg::Graph graph("report");
    capture_chain(graph, R, A, B, D, "i,j,x", kI, kJ, kX);

    cg::PassManager pm;
    pm.add<cg::passes::LayoutAssignment>();
    graph.apply(pm);

    auto const report = pm.explain();
    INFO(report);
    CHECK(report.find("LayoutAssignment") != std::string::npos);
    CHECK(report.find("structural-algebraic") != std::string::npos);
}
