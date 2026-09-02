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
void capture_chain(cg::Graph &graph, RuntimeTensor<double> &R, RuntimeTensor<double> const &A, RuntimeTensor<double> const &B,
                   RuntimeTensor<double> const &D, std::string const &w_spec, size_t d0, size_t d1, size_t d2) {
    auto &W = graph.declare_runtime_tensor<double>("W", {d0, d1, d2}, /*intermediate=*/true);

    // Built at run time so one builder serves both storage orders, which means the runtime
    // string_view constructor rather than the consteval literal one.
    std::string const produce = w_spec + " <- i,k ; k,x,j";
    std::string const consume = "i,x,y <- " + w_spec + " ; j,y";

    cg::CaptureGuard const guard(graph);
    cg::einsum(std::string_view{produce}, 0.0, &W, 1.0, A, B);
    cg::einsum(std::string_view{consume}, 0.0, &R, 1.0, W, D);
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
        auto const *desc = std::get_if<cg::EinsumDescriptor>(&node.op_data);
        REQUIRE(desc != nullptr);
        return {desc->spec.c_indices, desc->spec.a_indices, desc->spec.b_indices};
    }
    FAIL("no einsum node at that position");
    return {};
}

/// Run the chain to completion and hand back R.
RuntimeTensor<double> run_chain(std::string const &w_spec, size_t d0, size_t d1, size_t d2, RuntimeTensor<double> const &A,
                                RuntimeTensor<double> const &B, RuntimeTensor<double> const &D, bool with_layout) {
    RuntimeTensor<double> R = create_zero_tensor<double>("R", kI, kX, kY);
    cg::Graph             graph("chain");
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

TEST_CASE("LayoutAssignment - both contractions are made to read flat", "[ComputeGraph][LayoutAssignment]") {
    RuntimeTensor<double> A = create_random_tensor<double>("A", kI, kK);
    RuntimeTensor<double> B = create_random_tensor<double>("B", kK, kX, kJ);
    RuntimeTensor<double> D = create_random_tensor<double>("D", kJ, kY);
    RuntimeTensor<double> R = create_zero_tensor<double>("R", kI, kX, kY);

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

TEST_CASE("LayoutAssignment - the answer does not change", "[ComputeGraph][LayoutAssignment]") {
    RuntimeTensor<double> A = create_random_tensor<double>("A", kI, kK);
    RuntimeTensor<double> B = create_random_tensor<double>("B", kK, kX, kJ);
    RuntimeTensor<double> D = create_random_tensor<double>("D", kJ, kY);

    auto const plain  = run_chain("i,j,x", kI, kJ, kX, A, B, D, /*with_layout=*/false);
    auto const relaid = run_chain("i,j,x", kI, kJ, kX, A, B, D, /*with_layout=*/true);
    // And, as a third opinion that the model is describing the same arithmetic rather than the
    // same bug twice: the chain captured directly in the order the pass chose.
    auto const by_hand = run_chain("i,x,j", kI, kX, kJ, A, B, D, /*with_layout=*/false);

    for (size_t i = 0; i < kI; i++) {
        for (size_t x = 0; x < kX; x++) {
            for (size_t y = 0; y < kY; y++) {
                INFO("R[" << i << "," << x << "," << y << "]");
                CHECK_THAT(relaid(i, x, y), Catch::Matchers::WithinRel(plain(i, x, y), 1.0e-13));
                CHECK_THAT(by_hand(i, x, y), Catch::Matchers::WithinRel(plain(i, x, y), 1.0e-13));
            }
        }
    }
}

TEST_CASE("LayoutAssignment - the chosen order is already the best one", "[ComputeGraph][LayoutAssignment]") {
    RuntimeTensor<double> A = create_random_tensor<double>("A", kI, kK);
    RuntimeTensor<double> B = create_random_tensor<double>("B", kK, kX, kJ);
    RuntimeTensor<double> D = create_random_tensor<double>("D", kJ, kY);
    RuntimeTensor<double> R = create_zero_tensor<double>("R", kI, kX, kY);

    cg::Graph graph("already-flat");
    capture_chain(graph, R, A, B, D, "i,x,j", kI, kX, kJ);

    auto pass = only_layout(graph);
    CHECK(pass->num_relaid_out() == 0);
    CHECK(pass->explain().empty());
}

TEST_CASE("LayoutAssignment - a caller's tensor keeps the caller's axes", "[ComputeGraph][LayoutAssignment]") {
    RuntimeTensor<double> A = create_random_tensor<double>("A", kI, kK);
    RuntimeTensor<double> B = create_random_tensor<double>("B", kK, kX, kJ);
    RuntimeTensor<double> D = create_random_tensor<double>("D", kJ, kY);
    RuntimeTensor<double> R = create_zero_tensor<double>("R", kI, kX, kY);
    // The same chain, except W is the caller's tensor rather than the graph's. Its axis order is
    // part of what the caller asked for, so the copies stay.
    RuntimeTensor<double> W = create_zero_tensor<double>("W", kI, kJ, kX);

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

TEST_CASE("LayoutAssignment - rank two is left alone", "[ComputeGraph][LayoutAssignment]") {
    RuntimeTensor<double> A = create_random_tensor<double>("A", kI, kK);
    RuntimeTensor<double> B = create_random_tensor<double>("B", kK, kJ);
    RuntimeTensor<double> D = create_random_tensor<double>("D", kJ, kY);
    RuntimeTensor<double> R = create_zero_tensor<double>("R", kI, kY);

    cg::Graph graph("rank-two");
    {
        auto                  &T = graph.declare_runtime_tensor<double>("T", {kI, kJ}, /*intermediate=*/true);
        cg::CaptureGuard const guard(graph);
        cg::einsum("i,j <- i,k ; k,j", 0.0, &T, 1.0, A, B);
        cg::einsum("i,y <- i,j ; j,y", 0.0, &R, 1.0, T, D);
    }

    // A matrix has two readings and BLAS takes either one through `transa`, so there is nothing
    // here for a layout to win - and a GemmHint that a rewrite would have to keep in step.
    auto pass = only_layout(graph);
    CHECK(pass->num_relaid_out() == 0);
}

TEST_CASE("LayoutAssignment - a use it cannot rewrite pins the tensor", "[ComputeGraph][LayoutAssignment]") {
    RuntimeTensor<double> A  = create_random_tensor<double>("A", kI, kK);
    RuntimeTensor<double> B  = create_random_tensor<double>("B", kK, kX, kJ);
    RuntimeTensor<double> D  = create_random_tensor<double>("D", kJ, kY);
    RuntimeTensor<double> R  = create_zero_tensor<double>("R", kI, kX, kY);
    RuntimeTensor<double> Wc = create_zero_tensor<double>("Wc", kI, kJ, kX);

    cg::Graph graph("pinned");
    {
        auto                  &W = graph.declare_runtime_tensor<double>("W", {kI, kJ, kX}, /*intermediate=*/true);
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

TEST_CASE("LayoutAssignment - per-axis annotations follow their axes", "[ComputeGraph][LayoutAssignment]") {
    RuntimeTensor<double> A = create_random_tensor<double>("A", kI, kK);
    RuntimeTensor<double> B = create_random_tensor<double>("B", kK, kX, kJ);
    RuntimeTensor<double> D = create_random_tensor<double>("D", kJ, kY);
    RuntimeTensor<double> R = create_zero_tensor<double>("R", kI, kX, kY);

    cg::Graph    graph("symbols");
    cg::TensorId w_id{0};
    {
        auto                  &W = graph.declare_runtime_tensor<double>("W", {kI, kJ, kX}, /*intermediate=*/true);
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

TEST_CASE("LayoutAssignment - the same graph gets the same assignment", "[ComputeGraph][LayoutAssignment]") {
    RuntimeTensor<double> A = create_random_tensor<double>("A", kI, kK);
    RuntimeTensor<double> B = create_random_tensor<double>("B", kK, kX, kJ);
    RuntimeTensor<double> D = create_random_tensor<double>("D", kJ, kY);

    // A search whose answer varies between runs makes every layout downstream of it vary too,
    // which is the Kahn-FIFO lesson applied to a pass that picks one candidate out of several.
    std::vector<std::string> first;
    for (int trial = 0; trial < 3; trial++) {
        RuntimeTensor<double> R = create_zero_tensor<double>("R", kI, kX, kY);
        cg::Graph             graph("determinism");
        capture_chain(graph, R, A, B, D, "i,j,x", kI, kJ, kX);
        only_layout(graph);
        auto const chosen = spec_of(graph, 0)[0];
        if (trial == 0) {
            first = chosen;
        }
        CHECK(chosen == first);
    }
}

TEST_CASE("LayoutAssignment - it reports what it did", "[ComputeGraph][LayoutAssignment]") {
    RuntimeTensor<double> A = create_random_tensor<double>("A", kI, kK);
    RuntimeTensor<double> B = create_random_tensor<double>("B", kK, kX, kJ);
    RuntimeTensor<double> D = create_random_tensor<double>("D", kJ, kY);
    RuntimeTensor<double> R = create_zero_tensor<double>("R", kI, kX, kY);

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
