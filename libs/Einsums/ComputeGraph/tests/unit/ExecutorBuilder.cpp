//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

// ExecutorBuilder: a node's executor is derived from its data rather than baked
// at the capture site. Three properties are under test here.
//
//  1. IDENTITY. Rebuilding a captured node's executor from (kind, dtype, rank,
//     descriptor, operand ids) alone reproduces the captured one BIT FOR BIT.
//     That is the gate on every conversion: a builder that computes almost the
//     same thing is a silent numerical change, not a refactor.
//  2. NO DESYNC. A prefactor rewritten on the descriptor after capture is what
//     the next execute() applies. The executors these kinds used to carry baked
//     their scalars into a closure and ignored the descriptor entirely.
//  3. REDIRECT AND REBIND. A built executor resolves its operands through the
//     graph's slots, so redirect_slot() and rebind() reach it, exactly as they
//     reach a capture-baked lambda.
//
// Plus the monotonicity test that pins which kinds are reconstructible today.

#include <Einsums/ComputeGraph.hpp>
#include <Einsums/LinearAlgebra.hpp>
#include <Einsums/Tensor/RuntimeTensor.hpp>
#include <Einsums/Tensor/Tensor.hpp>
#include <Einsums/TensorUtilities/CreateRandomTensor.hpp>
#include <Einsums/TensorUtilities/CreateZeroTensor.hpp>

#include <complex>
#include <cstdint>
#include <cstring>
#include <functional>
#include <span>
#include <string>
#include <utility>
#include <variant>
#include <vector>

#include <Einsums/Testing.hpp>

using namespace einsums;
namespace cg = einsums::compute_graph;

namespace {

/// The (dtype, rank) key a manifest would record for @p node.
///
/// The node's own destination, which is what a manifest records per slot -
/// except for WriteParam, whose destination is a ParamTable entry no TensorId
/// names, so its key comes from the source scalar instead, and the two
/// control-flow kinds, which name no tensor at all.
std::pair<packed_gemm::ScalarType, size_t> node_key(cg::Graph const &graph, cg::Node const &node) {
    if (node.outputs.empty() && node.inputs.empty()) {
        return {packed_gemm::ScalarType::Unknown, size_t{0}};
    }
    cg::TensorId const id     = node.outputs.empty() ? node.inputs.front() : node.outputs.front();
    auto const        &handle = graph.tensor(id);
    return {handle.dtype, node.outputs.empty() ? size_t{0} : handle.rank};
}

/// Rebuild every reconstructible node's executor from data alone.
///
/// Deliberately reads nothing but what a saved file would hold: the node's
/// kind, its operands' dtype and rank, its descriptor, and its operand ids.
/// @return How many nodes were rebuilt.
size_t rebuild_executors(cg::Graph &graph) {
    size_t rebuilt = 0;
    for (auto &node : graph.nodes()) {
        if (!cg::reconstruction_blocker(node).empty()) {
            continue;
        }
        auto const [dtype, rank] = node_key(graph, node);
        node.execute = cg::build_executor(node.kind, dtype, rank, node.op_data, graph, std::span<cg::TensorId const>{node.inputs},
                                          std::span<cg::TensorId const>{node.outputs});
        ++rebuilt;
    }
    return rebuilt;
}

/// A copy of @p data with every live-params handle dropped.
///
/// What is left is exactly what a saved file would carry: the descriptor's
/// snapshot scalars and nothing shared with a running executor. Rebuilding
/// from this is the honest reconstruction test.
cg::OpData strip_live_params(cg::OpData data) {
    if (auto *d = std::get_if<cg::ScaleDescriptor>(&data)) {
        d->params.reset();
    } else if (auto *d = std::get_if<cg::PermuteDescriptor>(&data)) {
        d->params.reset();
    } else if (auto *d = std::get_if<cg::AxpbyDescriptor>(&data)) {
        d->params.reset();
    } else if (auto *d = std::get_if<cg::ElementwiseBinaryDescriptor>(&data)) {
        d->params.reset();
    } else if (std::holds_alternative<cg::DotDescriptor>(data) || std::holds_alternative<cg::TraceDescriptor>(data) ||
               std::holds_alternative<cg::GemmDescriptor>(data) || std::holds_alternative<cg::WriteParamDescriptor>(data)) {
        // Nothing to strip. These four hold snapshots only: a dot records one
        // bool, a trace records nothing, a write_param records a name and a
        // storage type, and a gemm's prefactors have no live block because no
        // pass rewrites them (see GemmDescriptor). What capture built is
        // already exactly what a file would carry.
    } else if (auto *d = std::get_if<cg::EinsumDescriptor>(&data)) {
        // An einsum carries three live handles, not one: the scalars, the index
        // lists, and the packed-GEMM memo. A saved file holds none of them - the
        // scalars and the indices are recoverable from the descriptor's own
        // snapshot fields, and the memo is pure cache - so all three go.
        d->params.reset();
        d->indices.reset();
        d->site.reset();
    }
    return data;
}

/// Rebuild every reconstructible node from its SNAPSHOT scalars alone.
size_t rebuild_from_snapshots(cg::Graph &graph) {
    size_t rebuilt = 0;
    for (auto &node : graph.nodes()) {
        if (!cg::reconstruction_blocker(node).empty()) {
            continue;
        }
        cg::OpData const bare    = strip_live_params(node.op_data);
        auto const [dtype, rank] = node_key(graph, node);
        node.execute             = cg::build_executor(node.kind, dtype, rank, bare, graph, std::span<cg::TensorId const>{node.inputs},
                                                      std::span<cg::TensorId const>{node.outputs});
        ++rebuilt;
    }
    return rebuilt;
}

/// A byte-for-byte snapshot of an OWNING tensor's storage.
template <typename TensorType>
std::vector<unsigned char> bytes_of(TensorType const &t) {
    using T = typename TensorType::ValueType;
    std::vector<unsigned char> out(t.size() * sizeof(T));
    std::memcpy(out.data(), t.data(), out.size());
    return out;
}

/// Overwrite a tensor's storage from a snapshot taken by @ref bytes_of.
template <typename TensorType>
void restore(TensorType *t, std::vector<unsigned char> const &snapshot) {
    using T = typename TensorType::ValueType;
    REQUIRE(snapshot.size() == t->size() * sizeof(T));
    std::memcpy(t->data(), snapshot.data(), snapshot.size());
}

/// The first node of @p kind, or null.
cg::Node *find_node(cg::Graph &graph, cg::OpKind kind) {
    for (auto &node : graph.nodes()) {
        if (node.kind == kind) {
            return &node;
        }
    }
    return nullptr;
}

/// A byte-for-byte snapshot of one scalar.
///
/// The point of comparing BYTES rather than values: a dot's summation order is
/// what a conversion could change, and two orders agree to every tolerance a
/// value comparison would use while landing on different bits.
template <typename T>
std::vector<unsigned char> bytes_of_scalar(T const &value) {
    std::vector<unsigned char> out(sizeof(T));
    std::memcpy(out.data(), &value, sizeof(T));
    return out;
}

/// A value of @p T that is awkward enough to expose a rounding difference:
/// none of these is representable in binary, so an executor that reaches a
/// different kernel will not land on the same bits.
template <typename T>
T awkward(double re, double im) {
    if constexpr (IsComplexV<T>) {
        return T{static_cast<typename T::value_type>(re), static_cast<typename T::value_type>(im)};
    } else {
        return static_cast<T>(re);
    }
}

} // namespace

// ── Reconstructible set: monotonicity ───────────────────────────────────────

TEST_CASE("ExecutorBuilder - the reconstructible set holds every kind converted so far", "[ComputeGraph][ExecutorBuilder]") {
    // RULE: kinds may be ADDED to this list, never removed.
    //
    // The design's guarantee is that the reconstructible set only ever grows.
    // Removing a kind means a graph that could be saved yesterday cannot be
    // saved today, which is a regression rather than a refactor, and it is the
    // kind of regression a newly added op kind can cause by accident - by
    // reintroducing a baked lambda under a kind that had stopped needing one.
    // If a change to this list is deliberate, the argument for it belongs in
    // the commit message, not in a quiet edit here.
    std::vector<cg::OpKind> const converted{
        cg::OpKind::Scale,          cg::OpKind::Permute,          cg::OpKind::Transpose,   cg::OpKind::Axpby, cg::OpKind::DirectProduct,
        cg::OpKind::DirectDivision, cg::OpKind::Einsum,           cg::OpKind::Dot,         cg::OpKind::Trace, cg::OpKind::WriteParam,
        cg::OpKind::Gemm,           cg::OpKind::ElementTransform, cg::OpKind::Conditional, cg::OpKind::Loop,
    };

    for (auto kind : converted) {
        INFO("kind: " << cg::op_kind_name(kind));
        REQUIRE(cg::is_reconstructible(kind));
    }
}

TEST_CASE("ExecutorBuilder - the predicate and the builder's dispatch agree", "[ComputeGraph][ExecutorBuilder]") {
    // Two sources of truth for "can this be rebuilt" would drift, and the way
    // they would drift is a builder entry landing without its bit, so a save
    // refuses a node it could actually reconstruct. Walk EVERY kind and require
    // the two to answer the same.
    cg::Graph graph("agreement");

    for (int raw = 0; raw <= static_cast<int>(cg::OpKind::Custom); ++raw) {
        auto const kind = static_cast<cg::OpKind>(raw);
        INFO("kind: " << cg::op_kind_name(kind));

        // Called with no operands and no descriptor, so a kind WITH an entry
        // still throws - but about its operands or its descriptor, never about
        // the entry being absent.
        std::string message;
        try {
            cg::OpData const empty{};
            (void)cg::build_executor(kind, packed_gemm::ScalarType::Float64, 2, empty, graph, {}, {});
        } catch (std::exception const &e) {
            message = e.what();
        }
        REQUIRE_FALSE(message.empty());

        bool const says_no_entry = message.find("has no builder entry") != std::string::npos;
        REQUIRE(says_no_entry == !cg::is_reconstructible(kind));
    }
}

TEST_CASE("ExecutorBuilder - an unconvertible kind names itself when refused", "[ComputeGraph][ExecutorBuilder]") {
    cg::Graph        graph("refusal");
    cg::OpData const empty{};
    REQUIRE_THROWS_WITH(cg::build_executor(cg::OpKind::HPTTPermute, packed_gemm::ScalarType::Float64, 2, empty, graph, {}, {}),
                        Catch::Matchers::ContainsSubstring("HPTTPermute"));
}

// ── Identity: rebuilt executors reproduce captured ones bit for bit ─────────

TEMPLATE_TEST_CASE("ExecutorBuilder - rebuilt Scale is bitwise identical", "[ComputeGraph][ExecutorBuilder]", float, double,
                   std::complex<float>, std::complex<double>) {
    using T = TestType;

    auto A       = create_random_tensor<T>("A", 5, 4);
    auto initial = bytes_of(A);

    cg::Graph graph("builder_scale");
    {
        cg::CaptureGuard const guard(graph);
        cg::scale(awkward<T>(0.1, -0.3), &A);
    }

    graph.execute();
    auto const captured = bytes_of(A);
    REQUIRE(captured != initial);

    restore(&A, initial);
    REQUIRE(rebuild_executors(graph) == 1);
    graph.execute();

    REQUIRE(bytes_of(A) == captured);
}

TEMPLATE_TEST_CASE("ExecutorBuilder - rebuilt Permute is bitwise identical", "[ComputeGraph][ExecutorBuilder]", float, double,
                   std::complex<float>, std::complex<double>) {
    using T = TestType;

    auto A = create_random_tensor<T>("A", 5, 4);
    auto C = create_random_tensor<T>("C", 4, 5);

    auto const a_initial = bytes_of(A);
    auto const c_initial = bytes_of(C);

    cg::Graph graph("builder_permute");
    {
        cg::CaptureGuard const guard(graph);
        // beta != 0 so the destination's prior contents participate: an
        // executor that zeroed instead of scaling would show up here.
        cg::permute(cg::PermuteFormatString{"ji <- ij"}, awkward<T>(0.7, 0.2), &C, awkward<T>(0.3, -0.9), A);
    }

    graph.execute();
    auto const captured = bytes_of(C);
    REQUIRE(captured != c_initial);

    restore(&A, a_initial);
    restore(&C, c_initial);
    REQUIRE(rebuild_executors(graph) == 1);
    graph.execute();

    REQUIRE(bytes_of(C) == captured);
}

TEMPLATE_TEST_CASE("ExecutorBuilder - rebuilt Permute is bitwise identical at rank 3", "[ComputeGraph][ExecutorBuilder]", float, double,
                   std::complex<float>, std::complex<double>) {
    using T = TestType;

    auto A = create_random_tensor<T>("A", 3, 4, 5);
    auto C = create_random_tensor<T>("C", 5, 3, 4);

    auto const a_initial = bytes_of(A);
    auto const c_initial = bytes_of(C);

    cg::Graph graph("builder_permute3");
    {
        cg::CaptureGuard const guard(graph);
        cg::permute(cg::PermuteFormatString{"kij <- ijk"}, awkward<T>(0.25, 0.5), &C, awkward<T>(1.1, -0.4), A);
    }

    graph.execute();
    auto const captured = bytes_of(C);

    restore(&A, a_initial);
    restore(&C, c_initial);
    REQUIRE(rebuild_executors(graph) == 1);
    graph.execute();

    REQUIRE(bytes_of(C) == captured);
}

TEST_CASE("ExecutorBuilder - rebuilt Transpose is bitwise identical", "[ComputeGraph][ExecutorBuilder]") {
    auto A = create_random_tensor<double>("A", 5, 4);
    auto C = create_zero_tensor<double>("C", 4, 5);

    auto const c_initial = bytes_of(C);

    cg::Graph graph("builder_transpose");
    {
        cg::CaptureGuard const guard(graph);
        cg::transpose(&C, A);
    }

    graph.execute();
    auto const captured = bytes_of(C);
    REQUIRE(captured != c_initial);

    restore(&C, c_initial);
    REQUIRE(rebuild_executors(graph) == 1);
    graph.execute();

    REQUIRE(bytes_of(C) == captured);
    for (size_t i = 0; i < 4; i++) {
        for (size_t j = 0; j < 5; j++) {
            REQUIRE(C(i, j) == A(j, i));
        }
    }
}

TEMPLATE_TEST_CASE("ExecutorBuilder - rebuilt Axpby is bitwise identical", "[ComputeGraph][ExecutorBuilder]", float, double,
                   std::complex<float>, std::complex<double>) {
    using T = TestType;

    auto X = create_random_tensor<T>("X", 6, 3);
    auto Y = create_random_tensor<T>("Y", 6, 3);

    auto const x_initial = bytes_of(X);
    auto const y_initial = bytes_of(Y);

    cg::Graph graph("builder_axpby");
    {
        cg::CaptureGuard const guard(graph);
        cg::axpby(awkward<T>(0.3, 0.1), X, awkward<T>(-0.7, 0.4), &Y);
    }

    graph.execute();
    auto const captured = bytes_of(Y);

    restore(&X, x_initial);
    restore(&Y, y_initial);
    REQUIRE(rebuild_executors(graph) == 1);
    graph.execute();

    REQUIRE(bytes_of(Y) == captured);
}

TEST_CASE("ExecutorBuilder - rebuilt Axpy keeps the beta == 1 fast path", "[ComputeGraph][ExecutorBuilder]") {
    auto X = create_random_tensor<double>("X", 6, 3);
    auto Y = create_random_tensor<double>("Y", 6, 3);

    auto const x_initial = bytes_of(X);
    auto const y_initial = bytes_of(Y);

    cg::Graph graph("builder_axpy");
    {
        cg::CaptureGuard const guard(graph);
        cg::axpy(0.3, X, &Y);
    }

    graph.execute();
    auto const captured = bytes_of(Y);

    restore(&X, x_initial);
    restore(&Y, y_initial);
    REQUIRE(rebuild_executors(graph) == 1);
    graph.execute();

    REQUIRE(bytes_of(Y) == captured);
}

TEMPLATE_TEST_CASE("ExecutorBuilder - rebuilt DirectProduct is bitwise identical", "[ComputeGraph][ExecutorBuilder]", float, double,
                   std::complex<float>, std::complex<double>) {
    using T = TestType;

    auto A = create_random_tensor<T>("A", 4, 4);
    auto B = create_random_tensor<T>("B", 4, 4);
    auto C = create_random_tensor<T>("C", 4, 4);

    auto const a_initial = bytes_of(A);
    auto const b_initial = bytes_of(B);
    auto const c_initial = bytes_of(C);

    cg::Graph graph("builder_direct_product");
    {
        cg::CaptureGuard const guard(graph);
        cg::direct_product(awkward<T>(0.9, -0.2), A, B, awkward<T>(0.4, 0.6), &C);
    }

    graph.execute();
    auto const captured = bytes_of(C);

    restore(&A, a_initial);
    restore(&B, b_initial);
    restore(&C, c_initial);
    REQUIRE(rebuild_executors(graph) == 1);
    graph.execute();

    REQUIRE(bytes_of(C) == captured);
}

TEMPLATE_TEST_CASE("ExecutorBuilder - rebuilt DirectDivision is bitwise identical", "[ComputeGraph][ExecutorBuilder]", float, double,
                   std::complex<float>, std::complex<double>) {
    using T = TestType;

    auto A = create_random_tensor<T>("A", 4, 4);
    auto B = create_random_tensor<T>("B", 4, 4);
    auto C = create_random_tensor<T>("C", 4, 4);

    // Keep the denominator away from zero so the comparison is about the
    // kernel and not about how two paths spell an infinity.
    for (size_t i = 0; i < 4; i++) {
        for (size_t j = 0; j < 4; j++) {
            B(i, j) = B(i, j) + awkward<T>(2.0, 0.5);
        }
    }

    auto const a_initial = bytes_of(A);
    auto const b_initial = bytes_of(B);
    auto const c_initial = bytes_of(C);

    cg::Graph graph("builder_direct_division");
    {
        cg::CaptureGuard const guard(graph);
        cg::direct_division(awkward<T>(0.9, -0.2), A, B, awkward<T>(0.4, 0.6), &C);
    }

    graph.execute();
    auto const captured = bytes_of(C);

    restore(&A, a_initial);
    restore(&B, b_initial);
    restore(&C, c_initial);
    REQUIRE(rebuild_executors(graph) == 1);
    graph.execute();

    REQUIRE(bytes_of(C) == captured);
}

TEMPLATE_TEST_CASE("ExecutorBuilder - a whole graph rebuilt from descriptor snapshots is bitwise identical",
                   "[ComputeGraph][ExecutorBuilder]", float, double, std::complex<float>, std::complex<double>) {
    using T = TestType;

    // Nothing shared with the captured executors survives strip_live_params,
    // so this is the reconstruction a LOADER performs: descriptor snapshots
    // plus operand ids, with no in-memory state carried over.
    auto A = create_random_tensor<T>("A", 4, 4);
    auto B = create_random_tensor<T>("B", 4, 4);
    auto C = create_random_tensor<T>("C", 4, 4);
    auto D = create_random_tensor<T>("D", 4, 4);

    for (size_t i = 0; i < 4; i++) {
        for (size_t j = 0; j < 4; j++) {
            B(i, j) = B(i, j) + awkward<T>(3.0, 0.25);
        }
    }

    auto const a0 = bytes_of(A);
    auto const b0 = bytes_of(B);
    auto const c0 = bytes_of(C);
    auto const d0 = bytes_of(D);

    cg::Graph graph("builder_snapshot_roundtrip");
    {
        cg::CaptureGuard const guard(graph);
        cg::direct_product(awkward<T>(0.7, 0.3), A, B, awkward<T>(0.2, -0.1), &C);
        cg::direct_division(awkward<T>(1.3, -0.6), C, B, awkward<T>(0.5, 0.25), &D);
        cg::scale(awkward<T>(0.9, 0.4), &D);
        cg::permute(cg::PermuteFormatString{"ji <- ij"}, awkward<T>(0.6, 0.1), &C, awkward<T>(0.8, -0.2), D);
        cg::axpby(awkward<T>(0.35, 0.15), C, awkward<T>(-0.45, 0.05), &D);
    }

    graph.execute();
    auto const captured_c = bytes_of(C);
    auto const captured_d = bytes_of(D);

    restore(&A, a0);
    restore(&B, b0);
    restore(&C, c0);
    restore(&D, d0);
    REQUIRE(rebuild_from_snapshots(graph) == 5);
    graph.execute();

    REQUIRE(bytes_of(C) == captured_c);
    REQUIRE(bytes_of(D) == captured_d);
}

// ── The desync this replaces ────────────────────────────────────────────────

TEST_CASE("ExecutorBuilder - a Scale factor rewritten after capture is applied", "[ComputeGraph][ExecutorBuilder]") {
    // The old executor closed over the factor and the descriptor was a
    // write-only record: a pass could set it to anything and the replay would
    // still apply the captured value.
    auto A  = create_random_tensor<double>("A", 3, 3);
    auto A0 = Tensor<double, 2>(A);

    cg::Graph graph("scale_desync");
    {
        cg::CaptureGuard const guard(graph);
        cg::scale(2.0, &A);
    }

    auto *node = find_node(graph, cg::OpKind::Scale);
    REQUIRE(node != nullptr);
    auto *desc = std::get_if<cg::ScaleDescriptor>(&node->op_data);
    REQUIRE(desc != nullptr);
    REQUIRE(desc->params != nullptr);

    desc->params->alpha = cg::PrefactorScalar{5.0};
    desc->factor        = desc->params->alpha;

    graph.execute();

    for (size_t i = 0; i < 3; i++) {
        for (size_t j = 0; j < 3; j++) {
            REQUIRE(A(i, j) == 5.0 * A0(i, j));
        }
    }
}

TEST_CASE("ExecutorBuilder - Permute prefactors rewritten after capture are applied", "[ComputeGraph][ExecutorBuilder]") {
    auto A  = create_random_tensor<double>("A", 3, 2);
    auto C  = create_random_tensor<double>("C", 2, 3);
    auto C0 = Tensor<double, 2>(C);

    cg::Graph graph("permute_desync");
    {
        cg::CaptureGuard const guard(graph);
        cg::permute(cg::PermuteFormatString{"ji <- ij"}, 0.0, &C, 1.0, A);
    }

    auto *node = find_node(graph, cg::OpKind::Permute);
    REQUIRE(node != nullptr);
    auto *desc = std::get_if<cg::PermuteDescriptor>(&node->op_data);
    REQUIRE(desc != nullptr);
    REQUIRE(desc->params != nullptr);

    // alpha 1 -> 3, beta 0 -> 2: both must reach the replay, and the beta
    // rewrite is the sharper half, since it turns an overwrite into an
    // accumulation.
    desc->params->alpha = cg::PrefactorScalar{3.0};
    desc->params->beta  = cg::PrefactorScalar{2.0};
    desc->alpha         = std::complex<double>{3.0, 0.0};
    desc->beta          = std::complex<double>{2.0, 0.0};

    graph.execute();

    for (size_t i = 0; i < 2; i++) {
        for (size_t j = 0; j < 3; j++) {
            REQUIRE_THAT(C(i, j), Catch::Matchers::WithinAbs(2.0 * C0(i, j) + 3.0 * A(j, i), 1e-12));
        }
    }
}

TEST_CASE("ExecutorBuilder - a complex Scale factor survives capture", "[ComputeGraph][ExecutorBuilder]") {
    // ScaleDescriptor::factor was a plain double filled from `factor.real()`,
    // so this read back as 2.0 and every pass that consulted it reasoned about
    // the wrong operation.
    using T = std::complex<double>;
    auto A  = create_random_tensor<T>("A", 3, 3);

    cg::Graph graph("scale_complex_descriptor");
    {
        cg::CaptureGuard const guard(graph);
        cg::scale(T{2.0, 3.0}, &A);
    }

    auto *node = find_node(graph, cg::OpKind::Scale);
    REQUIRE(node != nullptr);
    auto *desc = std::get_if<cg::ScaleDescriptor>(&node->op_data);
    REQUIRE(desc != nullptr);
    REQUIRE(std::holds_alternative<T>(desc->factor));
    REQUIRE(std::get<T>(desc->factor) == T{2.0, 3.0});
}

// ── Redirects and rebinds reach a built executor ────────────────────────────

TEST_CASE("ExecutorBuilder - a rebuilt executor follows redirect_slot", "[ComputeGraph][ExecutorBuilder][CSE]") {
    // Executor lambdas resolve operands through their captured TensorSlot, not
    // through Node::inputs, which is why CSE has to call redirect_slot at all
    // (see Pass_CSE). A data-built executor must resolve the same way, or every
    // redirect-based pass silently stops working on the kinds it converts.
    auto X     = create_random_tensor<double>("X", 4, 3);
    auto Y     = create_random_tensor<double>("Y", 4, 3);
    auto out   = create_zero_tensor<double>("out", 4, 3);
    auto spare = create_zero_tensor<double>("spare", 4, 3);

    cg::Graph graph("builder_redirect");
    {
        cg::CaptureGuard const guard(graph);
        cg::axpby(1.0, X, 0.0, &out);   // out = X
        cg::axpby(1.0, Y, 0.0, &spare); // only here so Y gets a slot to redirect to
    }

    REQUIRE(rebuild_executors(graph) == 2);

    cg::TensorId const x_id = graph.find_tensor_id_by_ptr(&X);
    cg::TensorId const y_id = graph.find_tensor_id_by_ptr(&Y);
    REQUIRE(x_id != 0);
    REQUIRE(y_id != 0);

    graph.redirect_slot(x_id, y_id);
    graph.execute();

    for (size_t i = 0; i < 4; i++) {
        for (size_t j = 0; j < 3; j++) {
            REQUIRE(out(i, j) == Y(i, j));
        }
    }
}

TEST_CASE("ExecutorBuilder - a rebuilt executor writes the rebound storage", "[ComputeGraph][ExecutorBuilder][Rebind]") {
    auto X    = create_random_tensor<double>("X", 4, 3);
    auto out  = create_zero_tensor<double>("out", 4, 3);
    auto out2 = create_zero_tensor<double>("out2", 4, 3);

    cg::Graph graph("builder_rebind");
    {
        cg::CaptureGuard const guard(graph);
        cg::axpby(1.0, X, 0.0, &out);
    }

    REQUIRE(rebuild_executors(graph) == 1);

    graph.rebind(out, out2);
    graph.execute();

    for (size_t i = 0; i < 4; i++) {
        for (size_t j = 0; j < 3; j++) {
            REQUIRE(out2(i, j) == X(i, j));
            REQUIRE(out(i, j) == 0.0); // the whole node moved, not half of it
        }
    }
}

// ── serializability_report ─────────────────────────────────────────────────

TEST_CASE("ExecutorBuilder - a fully converted graph reports no blockers", "[ComputeGraph][ExecutorBuilder]") {
    auto A = create_random_tensor<double>("A", 4, 4);
    auto B = create_random_tensor<double>("B", 4, 4);
    auto C = create_zero_tensor<double>("C", 4, 4);
    auto D = create_zero_tensor<double>("D", 4, 4);

    cg::Graph graph("report_clean");
    {
        cg::CaptureGuard const guard(graph);
        cg::direct_product(1.0, A, B, 0.0, &C);
        cg::scale(2.0, &C);
        cg::transpose(&D, C);
        cg::axpby(1.0, D, 1.0, &C);
    }

    auto const report = graph.serializability_report();
    if (!report.empty()) {
        INFO("first blocker: " << report.front().kind_name << " - " << report.front().reason);
        FAIL("a graph of converted kinds should report no blockers");
    }
    REQUIRE(report.empty());
}

TEST_CASE("ExecutorBuilder - the report names the node and the field that blocks a save", "[ComputeGraph][ExecutorBuilder]") {
    auto A = create_random_tensor<double>("A", 4, 3);
    auto B = create_random_tensor<double>("B", 3, 5);
    auto C = create_zero_tensor<double>("C", 4, 5);
    auto D = create_zero_tensor<double>("D", 4, 5);

    cg::Graph graph("report_blocked");
    {
        cg::CaptureGuard const guard(graph);
        cg::scale(2.0, &A);                // reconstructible
        cg::einsum("ik;kj->ij", &C, A, B); // reconstructible since B2
        cg::element_transform(&D, [](double v) { return v + 1.0; });
    }

    auto const report = graph.serializability_report();
    REQUIRE(report.size() == 1);
    REQUIRE(report.front().kind_name == "ElementTransform");
    // The message names the FIX, not the missing field: "no descriptor" is not
    // something a user can act on, and "register a named op" is.
    REQUIRE_THAT(report.front().reason, Catch::Matchers::ContainsSubstring("anonymous closure"));
    REQUIRE_THAT(report.front().reason, Catch::Matchers::ContainsSubstring("register_op"));
    REQUIRE_FALSE(report.front().label.empty());
    REQUIRE(report.front().node_id != 0);
}

TEST_CASE("ExecutorBuilder - an einsum no longer blocks a save", "[ComputeGraph][ExecutorBuilder][Einsum]") {
    // The gemm_hint used to hold three std::functions, which is a closure a
    // file cannot contain; the hint is data now and the node reports clean.
    auto A = create_random_tensor<double>("A", 4, 3);
    auto B = create_random_tensor<double>("B", 3, 5);
    auto C = create_zero_tensor<double>("C", 4, 5);

    cg::Graph graph("report_einsum");
    {
        cg::CaptureGuard const guard(graph);
        cg::einsum("ik;kj->ij", &C, A, B);
    }

    auto const *node = find_node(graph, cg::OpKind::Einsum);
    REQUIRE(node != nullptr);
    auto const *desc = std::get_if<cg::EinsumDescriptor>(&node->op_data);
    REQUIRE(desc != nullptr);
    REQUIRE(desc->gemm_hint != nullptr);

    REQUIRE(cg::reconstruction_blocker(*node).empty());
    REQUIRE(graph.serializability_report().empty());
}

TEST_CASE("ExecutorBuilder - a tiled direct division is reported despite a reconstructible kind",
          "[ComputeGraph][ExecutorBuilder][Tiled]") {
    // OpKind::DirectDivision is reconstructible, but the TILED variant records
    // under the same kind with a TiledElementwiseDescriptor and has no builder
    // entry. The per-kind bit cannot see that; the per-node verdict must.
    using Grid = std::vector<std::vector<int>>;
    TiledRuntimeTensor<double> const A("A", Grid{{2, 2}, {2, 2}});
    TiledRuntimeTensor<double> const B("B", Grid{{2, 2}, {2, 2}});
    TiledRuntimeTensor<double>       C("C", Grid{{2, 2}, {2, 2}});

    cg::Graph graph("report_tiled");
    {
        cg::CaptureGuard const guard(graph);
        cg::direct_division(1.0, A, B, 0.0, &C);
    }

    auto const report = graph.serializability_report();
    REQUIRE(report.size() == 1);
    REQUIRE(report.front().kind_name == "DirectDivision");
    REQUIRE_THAT(report.front().reason, Catch::Matchers::ContainsSubstring("tiled"));
}

// ── Einsum ─────────────────────────────────────────────────────────────────
//
// The riskiest conversion of the milestone: an einsum's executor used to be
// baked at the capture site over the operands' STATIC types, and the dispatch
// cascade below it has a dozen routes. Rebuilding it from the descriptor has to
// land on the same route AND the same bits, or a graph replays differently than
// it captured.

TEMPLATE_TEST_CASE("ExecutorBuilder - rebuilt Einsum is bitwise identical for a GEMM shape", "[ComputeGraph][ExecutorBuilder][Einsum]",
                   float, double, std::complex<float>, std::complex<double>) {
    using T = TestType;

    // Rank-2 x rank-2 -> rank-2 with one link index: the shape that qualifies
    // for a GemmHint, so this covers the hint-eligible path as well.
    auto A = create_random_tensor<T>("A", 5, 4);
    auto B = create_random_tensor<T>("B", 4, 3);
    auto C = create_random_tensor<T>("C", 5, 3);

    auto const a0 = bytes_of(A);
    auto const b0 = bytes_of(B);
    auto const c0 = bytes_of(C);

    cg::Graph graph("builder_einsum_gemm");
    {
        cg::CaptureGuard const guard(graph);
        // beta != 0 so C's prior contents participate.
        cg::einsum("ij <- ik ; kj", awkward<T>(0.4, -0.2), &C, awkward<T>(0.9, 0.3), A, B);
    }

    auto const *node = find_node(graph, cg::OpKind::Einsum);
    REQUIRE(node != nullptr);
    auto const *desc = std::get_if<cg::EinsumDescriptor>(&node->op_data);
    REQUIRE(desc != nullptr);
    REQUIRE(desc->gemm_hint != nullptr);

    graph.execute();
    auto const captured = bytes_of(C);
    REQUIRE(captured != c0);

    restore(&A, a0);
    restore(&B, b0);
    restore(&C, c0);
    REQUIRE(rebuild_executors(graph) == 1);
    graph.execute();

    REQUIRE(bytes_of(C) == captured);
}

TEMPLATE_TEST_CASE("ExecutorBuilder - rebuilt Einsum is bitwise identical for a rank-3 contraction",
                   "[ComputeGraph][ExecutorBuilder][Einsum]", float, double, std::complex<float>, std::complex<double>) {
    using T = TestType;

    // Three index letters on the first operand and a shared link: no BLAS fast
    // path takes this, so it exercises the PackedGemm / generic end of the
    // cascade rather than the GEMM end above.
    auto A = create_random_tensor<T>("A", 3, 4, 5);
    auto B = create_random_tensor<T>("B", 5, 2);
    auto C = create_random_tensor<T>("C", 3, 4, 2);

    auto const a0 = bytes_of(A);
    auto const b0 = bytes_of(B);
    auto const c0 = bytes_of(C);

    cg::Graph graph("builder_einsum_rank3");
    {
        cg::CaptureGuard const guard(graph);
        cg::einsum("abc <- abd ; dc", awkward<T>(0.25, 0.75), &C, awkward<T>(1.1, -0.4), A, B);
    }

    graph.execute();
    auto const captured = bytes_of(C);
    REQUIRE(captured != c0);

    restore(&A, a0);
    restore(&B, b0);
    restore(&C, c0);
    REQUIRE(rebuild_executors(graph) == 1);
    graph.execute();

    REQUIRE(bytes_of(C) == captured);
}

TEMPLATE_TEST_CASE("ExecutorBuilder - rebuilt Einsum honors the conjugation flags", "[ComputeGraph][ExecutorBuilder][Einsum]",
                   std::complex<float>, std::complex<double>) {
    using T = TestType;

    // conj_a / conj_b live on EinsumParams AND on the descriptor's snapshot.
    // A rebuild that read neither would compute the unconjugated product, which
    // is numerically plausible and completely wrong.
    auto A = create_random_tensor<T>("A", 4, 3);
    auto B = create_random_tensor<T>("B", 3, 4);
    auto C = create_random_tensor<T>("C", 4, 4);

    auto const a0 = bytes_of(A);
    auto const b0 = bytes_of(B);
    auto const c0 = bytes_of(C);

    cg::Graph graph("builder_einsum_conj");
    {
        cg::CaptureGuard const guard(graph);
        cg::einsum("ij <- ik ; kj", awkward<T>(0.3, 0.2), &C, awkward<T>(0.8, -0.5), A, B, /*conj_a=*/true, /*conj_b=*/false);
    }

    graph.execute();
    auto const captured = bytes_of(C);

    restore(&A, a0);
    restore(&B, b0);
    restore(&C, c0);
    REQUIRE(rebuild_from_snapshots(graph) == 1);
    graph.execute();

    REQUIRE(bytes_of(C) == captured);
}

TEMPLATE_TEST_CASE("ExecutorBuilder - rebuilt Einsum keeps the repeated-letter route", "[ComputeGraph][ExecutorBuilder][Einsum]", float,
                   double, std::complex<float>, std::complex<double>) {
    using T = TestType;

    // "ij <- ii ; jj" is a diagonal access on both inputs. Every fast path in
    // the cascade assumes a letter appears at most once per operand, so this
    // spec is claimed by the repeat-aware generic loop before any of them; a
    // rebuild that landed on a fast path would silently compute something else.
    auto A = create_random_tensor<T>("A", 4, 4);
    auto B = create_random_tensor<T>("B", 5, 5);
    auto C = create_random_tensor<T>("C", 4, 5);

    auto const a0 = bytes_of(A);
    auto const b0 = bytes_of(B);
    auto const c0 = bytes_of(C);

    cg::Graph graph("builder_einsum_diagonal");
    {
        cg::CaptureGuard const guard(graph);
        cg::einsum("ij <- ii ; jj", awkward<T>(0.5, 0.1), &C, awkward<T>(0.7, -0.3), A, B);
    }

    graph.execute();
    auto const captured = bytes_of(C);
    auto const route    = std::string{cg::dispatch::last_dispatch_route()};
    REQUIRE(route == "generic_loop_repeated_indices");

    restore(&A, a0);
    restore(&B, b0);
    restore(&C, c0);
    REQUIRE(rebuild_executors(graph) == 1);
    graph.execute();

    REQUIRE(bytes_of(C) == captured);
    REQUIRE(std::string{cg::dispatch::last_dispatch_route()} == route);
}

TEMPLATE_TEST_CASE("ExecutorBuilder - an Einsum rebuilt from snapshots alone is bitwise identical",
                   "[ComputeGraph][ExecutorBuilder][Einsum]", float, double, std::complex<float>, std::complex<double>) {
    using T = TestType;

    // The LOADER case: params, indices and the packed-GEMM memo are all gone,
    // and the builder has to synthesize them from the descriptor's own
    // snapshot fields. Nothing in memory is carried over.
    auto A = create_random_tensor<T>("A", 4, 6);
    auto B = create_random_tensor<T>("B", 6, 4);
    auto C = create_random_tensor<T>("C", 4, 4);
    auto D = create_random_tensor<T>("D", 4, 4);

    auto const a0 = bytes_of(A);
    auto const b0 = bytes_of(B);
    auto const c0 = bytes_of(C);
    auto const d0 = bytes_of(D);

    cg::Graph graph("builder_einsum_snapshot");
    {
        cg::CaptureGuard const guard(graph);
        cg::einsum("ij <- ik ; kj", awkward<T>(0.2, -0.4), &C, awkward<T>(1.3, 0.6), A, B);
        cg::einsum("ij <- ik ; kj", awkward<T>(0.5, 0.25), &D, awkward<T>(0.35, -0.15), C, C);
        cg::scale(awkward<T>(0.9, 0.4), &D);
    }

    graph.execute();
    auto const captured_c = bytes_of(C);
    auto const captured_d = bytes_of(D);

    restore(&A, a0);
    restore(&B, b0);
    restore(&C, c0);
    restore(&D, d0);
    REQUIRE(rebuild_from_snapshots(graph) == 3);
    graph.execute();

    REQUIRE(bytes_of(C) == captured_c);
    REQUIRE(bytes_of(D) == captured_d);
}

TEST_CASE("ExecutorBuilder - Einsum prefactors rewritten after capture are applied", "[ComputeGraph][ExecutorBuilder][Einsum]") {
    // Graph::update_prefactors writes the live params AND the descriptor
    // snapshot; a rebuilt executor reads the live block, so the rewrite lands
    // on the next replay whether the node was rebuilt before or after it.
    auto A = create_random_tensor<double>("A", 3, 3);
    auto B = create_random_tensor<double>("B", 3, 3);
    auto C = create_zero_tensor<double>("C", 3, 3);

    cg::Graph graph("einsum_desync");
    {
        cg::CaptureGuard const guard(graph);
        cg::einsum("ij <- ik ; kj", 0.0, &C, 1.0, A, B);
    }

    REQUIRE(rebuild_executors(graph) == 1);

    auto *node = find_node(graph, cg::OpKind::Einsum);
    REQUIRE(node != nullptr);
    graph.update_prefactors(node->id, cg::PrefactorScalar{0.0}, cg::PrefactorScalar{2.5});
    graph.execute();

    for (size_t i = 0; i < 3; i++) {
        for (size_t j = 0; j < 3; j++) {
            double want = 0.0;
            for (size_t k = 0; k < 3; k++) {
                want += A(i, k) * B(k, j);
            }
            REQUIRE_THAT(C(i, j), Catch::Matchers::WithinAbs(2.5 * want, 1e-12));
        }
    }
}

TEST_CASE("ExecutorBuilder - a rebuilt Einsum follows redirect_slot", "[ComputeGraph][ExecutorBuilder][Einsum][CSE]") {
    auto X   = create_random_tensor<double>("X", 4, 4);
    auto Y   = create_random_tensor<double>("Y", 4, 4);
    auto B   = create_random_tensor<double>("B", 4, 4);
    auto out = create_zero_tensor<double>("out", 4, 4);
    auto aux = create_zero_tensor<double>("aux", 4, 4);

    cg::Graph graph("einsum_redirect");
    {
        cg::CaptureGuard const guard(graph);
        cg::einsum("ij <- ik ; kj", 0.0, &out, 1.0, X, B);
        cg::einsum("ij <- ik ; kj", 0.0, &aux, 1.0, Y, B); // only so Y gets a slot
    }

    REQUIRE(rebuild_executors(graph) == 2);

    cg::TensorId const x_id = graph.find_tensor_id_by_ptr(&X);
    cg::TensorId const y_id = graph.find_tensor_id_by_ptr(&Y);
    REQUIRE(x_id != 0);
    REQUIRE(y_id != 0);

    graph.redirect_slot(x_id, y_id);
    graph.execute();

    for (size_t i = 0; i < 4; i++) {
        for (size_t j = 0; j < 4; j++) {
            REQUIRE_THAT(out(i, j), Catch::Matchers::WithinAbs(aux(i, j), 1e-12));
        }
    }
}

TEST_CASE("ExecutorBuilder - a rebuilt Einsum writes the rebound storage", "[ComputeGraph][ExecutorBuilder][Einsum][Rebind]") {
    auto A    = create_random_tensor<double>("A", 4, 4);
    auto B    = create_random_tensor<double>("B", 4, 4);
    auto out  = create_zero_tensor<double>("out", 4, 4);
    auto out2 = create_zero_tensor<double>("out2", 4, 4);

    cg::Graph graph("einsum_rebind");
    {
        cg::CaptureGuard const guard(graph);
        cg::einsum("ij <- ik ; kj", 0.0, &out, 1.0, A, B);
    }

    REQUIRE(rebuild_executors(graph) == 1);

    graph.rebind(out, out2);
    graph.execute();

    for (size_t i = 0; i < 4; i++) {
        for (size_t j = 0; j < 4; j++) {
            double want = 0.0;
            for (size_t k = 0; k < 4; k++) {
                want += A(i, k) * B(k, j);
            }
            REQUIRE_THAT(out2(i, j), Catch::Matchers::WithinAbs(want, 1e-12));
            REQUIRE(out(i, j) == 0.0); // the whole node moved, not half of it
        }
    }
}

TEST_CASE("ExecutorBuilder - a rebind that changes an lda is honored, and the hint stays a snapshot",
          "[ComputeGraph][ExecutorBuilder][Einsum][Rebind]") {
    // GemmHint::*::leading_dim is recorded when the hint is derived, and
    // Graph::rebind accepts any tensor of matching rank and dims - a column
    // range of a wider store has the same shape and a different leading
    // dimension. The staleness rule is: the recorded value is a PLANNING hint
    // that nothing re-derives, and execution reads the live impl instead, so
    // the answer is right even though the record is stale.
    constexpr size_t n = 4;

    auto A = create_random_tensor<double>("A", n, n);
    auto B = create_random_tensor<double>("B", n, n);
    auto C = create_zero_tensor<double>("C", n, n);

    // A wider parent, so the slice's leading dimension differs from A's.
    auto wide = create_random_tensor<double>("wide", 3 * n, n);
    auto tall = wide(Range{0, n}, AllT{});
    for (size_t i = 0; i < n; i++) {
        for (size_t j = 0; j < n; j++) {
            tall(i, j) = A(i, j) + 0.5;
        }
    }

    cg::Graph graph("einsum_rebind_lda");
    {
        cg::CaptureGuard const guard(graph);
        cg::einsum("ij <- ik ; kj", 0.0, &C, 1.0, A, B);
    }

    auto *node = find_node(graph, cg::OpKind::Einsum);
    REQUIRE(node != nullptr);
    auto *desc = std::get_if<cg::EinsumDescriptor>(&node->op_data);
    REQUIRE(desc != nullptr);
    REQUIRE(desc->gemm_hint != nullptr);
    int const recorded_lda = desc->gemm_hint->a.leading_dim;

    REQUIRE(rebuild_executors(graph) == 1);
    // By id, because the slice is a TensorView and the two-tensor overload
    // wants both operands to be the same type.
    cg::TensorId const a_id = graph.find_tensor_id_by_ptr(&A);
    REQUIRE(a_id != cg::TensorId{0});
    graph.rebind(a_id, tall);
    int const live_lda = static_cast<int>(tall.impl().get_lda());
    REQUIRE(live_lda != recorded_lda);

    graph.execute();

    // Right answer, from the REBOUND operand and its real leading dimension.
    for (size_t i = 0; i < n; i++) {
        for (size_t j = 0; j < n; j++) {
            double want = 0.0;
            for (size_t k = 0; k < n; k++) {
                want += tall(i, k) * B(k, j);
            }
            REQUIRE_THAT(C(i, j), Catch::Matchers::WithinAbs(want, 1e-12));
        }
    }

    // And the recorded value did not move: it is a plan-time record, not a
    // second source of truth for execution.
    REQUIRE(desc->gemm_hint->a.leading_dim == recorded_lda);
}

// ── Dot ────────────────────────────────────────────────────────────────────
//
// The conversion that removes the last raw pointer from a built executor. A
// dot's destination is ONE number, and the two capture surfaces disagree about
// what holds it: ``cg::dot(&e, A, B)`` registers a bare ``T *`` (a rank-0
// handle with no TensorImpl and no slot), while ``cg::dot_python`` hands the
// graph a rank-1 tensor so a Python caller has a scalar it can go on to scale.
// One ScalarAccessor covers both, and these tests pin that the write still
// lands in the caller's own variable.

TEMPLATE_TEST_CASE("ExecutorBuilder - rebuilt Dot is bitwise identical", "[ComputeGraph][ExecutorBuilder][Dot]", float, double,
                   std::complex<float>, std::complex<double>) {
    using T = TestType;

    auto A = create_random_tensor<T>("A", 7, 5);
    auto B = create_random_tensor<T>("B", 7, 5);

    T result{};

    cg::Graph graph("builder_dot");
    {
        cg::CaptureGuard const guard(graph);
        cg::dot(&result, A, B);
    }

    graph.execute();
    T const captured = result;
    REQUIRE(captured != T{});

    result = T{};
    REQUIRE(rebuild_executors(graph) == 1);
    graph.execute();

    // Bitwise, and into the CALLER'S variable: the raw-pointer semantics the
    // slot route replaces are preserved exactly.
    REQUIRE(bytes_of_scalar(result) == bytes_of_scalar(captured));
}

TEMPLATE_TEST_CASE("ExecutorBuilder - rebuilt Dot is bitwise identical through a tensor destination",
                   "[ComputeGraph][ExecutorBuilder][Dot]", float, double, std::complex<float>, std::complex<double>) {
    using T = TestType;

    // ``dot_python``'s shape: the destination is a rank-1 tensor with a slot of
    // its own, so the accessor takes the tensor route rather than the raw one.
    auto A = create_random_tensor<T>("A", 6, 6);
    auto B = create_random_tensor<T>("B", 6, 6);
    auto R = create_zero_tensor<T>("R", 1);

    cg::Graph graph("builder_dot_python");
    {
        cg::CaptureGuard const guard(graph);
        cg::dot_python(&R, A, B);
    }

    graph.execute();
    auto const captured = bytes_of(R);

    R(0) = T{};
    REQUIRE(rebuild_executors(graph) == 1);
    graph.execute();

    REQUIRE(bytes_of(R) == captured);
}

TEMPLATE_TEST_CASE("ExecutorBuilder - rebuilt Dot keeps dot and dotc apart", "[ComputeGraph][ExecutorBuilder][Dot]", std::complex<float>,
                   std::complex<double>) {
    using T = TestType;

    // ``conjugated`` is the descriptor's only field, and getting it wrong is
    // numerically plausible and completely wrong: for a complex operand the
    // bilinear sum and the Hermitian one are different numbers. The test
    // requires both that each rebuild matches its own capture AND that the two
    // still differ, so a builder that ignored the flag cannot pass by
    // accidentally agreeing with itself.
    auto A = create_random_tensor<T>("A", 5, 4);
    auto B = create_random_tensor<T>("B", 5, 4);
    auto R = create_zero_tensor<T>("R", 1);
    auto S = create_zero_tensor<T>("S", 1);

    cg::Graph graph("builder_dot_conj");
    {
        cg::CaptureGuard const guard(graph);
        cg::dot_python(&R, A, B);
        cg::dotc_python(&S, A, B);
    }

    graph.execute();
    auto const captured_plain = bytes_of(R);
    auto const captured_conj  = bytes_of(S);
    REQUIRE(captured_plain != captured_conj);

    R(0) = T{};
    S(0) = T{};
    REQUIRE(rebuild_from_snapshots(graph) == 2);
    graph.execute();

    REQUIRE(bytes_of(R) == captured_plain);
    REQUIRE(bytes_of(S) == captured_conj);
}

TEST_CASE("ExecutorBuilder - a real dotc agrees with dot", "[ComputeGraph][ExecutorBuilder][Dot]") {
    // Conjugation is a no-op on a real dtype, so the two builders must land on
    // the same number. ``true_dot`` and ``dot`` are different kernels even
    // there, so this is a real check rather than a tautology.
    auto A = create_random_tensor<double>("A", 9, 1);
    auto B = create_random_tensor<double>("B", 9, 1);
    auto R = create_zero_tensor<double>("R", 1);
    auto S = create_zero_tensor<double>("S", 1);

    cg::Graph graph("builder_dotc_real");
    {
        cg::CaptureGuard const guard(graph);
        cg::dot_python(&R, A, B);
        cg::dotc_python(&S, A, B);
    }

    REQUIRE(rebuild_executors(graph) == 2);
    graph.execute();

    REQUIRE_THAT(R(0), Catch::Matchers::WithinAbs(S(0), 1e-14));
}

TEST_CASE("ExecutorBuilder - a rebuilt Dot follows redirect_slot", "[ComputeGraph][ExecutorBuilder][Dot][CSE]") {
    auto X   = create_random_tensor<double>("X", 4, 3);
    auto Y   = create_random_tensor<double>("Y", 4, 3);
    auto Z   = create_random_tensor<double>("Z", 4, 3);
    auto out = create_zero_tensor<double>("out", 1);
    auto aux = create_zero_tensor<double>("aux", 1);

    cg::Graph graph("dot_redirect");
    {
        cg::CaptureGuard const guard(graph);
        cg::dot_python(&out, X, Z);
        cg::dot_python(&aux, Y, Z); // only so Y gets a slot to redirect to
    }

    REQUIRE(rebuild_executors(graph) == 2);

    cg::TensorId const x_id = graph.find_tensor_id_by_ptr(&X);
    cg::TensorId const y_id = graph.find_tensor_id_by_ptr(&Y);
    REQUIRE(x_id != 0);
    REQUIRE(y_id != 0);

    graph.redirect_slot(x_id, y_id);
    graph.execute();

    REQUIRE(out(0) == aux(0));
}

TEST_CASE("ExecutorBuilder - a rebuilt Dot writes the rebound destination", "[ComputeGraph][ExecutorBuilder][Dot][Rebind]") {
    auto A    = create_random_tensor<double>("A", 5, 2);
    auto B    = create_random_tensor<double>("B", 5, 2);
    auto out  = create_zero_tensor<double>("out", 1);
    auto out2 = create_zero_tensor<double>("out2", 1);

    cg::Graph graph("dot_rebind");
    {
        cg::CaptureGuard const guard(graph);
        cg::dot_python(&out, A, B);
    }

    REQUIRE(rebuild_executors(graph) == 1);

    graph.rebind(out, out2);
    graph.execute();

    double want = 0.0;
    for (size_t ii = 0; ii < 5; ii++) {
        for (size_t jj = 0; jj < 2; jj++) {
            want += A(ii, jj) * B(ii, jj);
        }
    }
    REQUIRE_THAT(out2(0), Catch::Matchers::WithinAbs(want, 1e-12));
    REQUIRE(out(0) == 0.0); // the whole node moved, not half of it
}

TEST_CASE("ExecutorBuilder - a tiled dot is reported despite a reconstructible kind", "[ComputeGraph][ExecutorBuilder][Dot][Tiled]") {
    // The DirectDivision precedent, one kind over: OpKind::Dot is
    // reconstructible, and the TILED dot records under it with a
    // TiledDotDescriptor whose per-tile reduction has no builder entry.
    using Grid = std::vector<std::vector<int>>;
    TiledRuntimeTensor<double> const A("A", Grid{{2, 2}, {2, 2}});
    TiledRuntimeTensor<double> const B("B", Grid{{2, 2}, {2, 2}});
    auto                             R = create_zero_tensor<double>("R", 1);

    cg::Graph graph("dot_tiled_report");
    {
        cg::CaptureGuard const guard(graph);
        cg::dot_python(&R, A, B);
    }

    auto const report = graph.serializability_report();
    REQUIRE(report.size() == 1);
    REQUIRE(report.front().kind_name == "Dot");
    REQUIRE_THAT(report.front().reason, Catch::Matchers::ContainsSubstring("tiled"));
}

// ── Trace ──────────────────────────────────────────────────────────────────

TEMPLATE_TEST_CASE("ExecutorBuilder - rebuilt Trace is bitwise identical", "[ComputeGraph][ExecutorBuilder][Trace]", float, double,
                   std::complex<float>, std::complex<double>) {
    using T = TestType;

    // Complex traces ARE supported: the diagonal sum is defined for any dtype
    // the graph carries, so all four are under test rather than the reals only.
    auto A = create_random_tensor<T>("A", 6, 6);

    T result{};

    cg::Graph graph("builder_trace");
    {
        cg::CaptureGuard const guard(graph);
        cg::trace(&result, A);
    }

    graph.execute();
    T const captured = result;
    REQUIRE(captured != T{});

    result = T{};
    REQUIRE(rebuild_executors(graph) == 1);
    graph.execute();

    REQUIRE(bytes_of_scalar(result) == bytes_of_scalar(captured));

    // The sum is sequential and in index order, which is what makes the bit
    // comparison above meaningful; spell it out once.
    T reference{};
    for (size_t ii = 0; ii < 6; ii++) {
        reference += A(ii, ii);
    }
    REQUIRE(bytes_of_scalar(result) == bytes_of_scalar(reference));
}

TEMPLATE_TEST_CASE("ExecutorBuilder - rebuilt Trace is bitwise identical through a tensor destination",
                   "[ComputeGraph][ExecutorBuilder][Trace]", float, double, std::complex<float>, std::complex<double>) {
    using T = TestType;

    // ``trace_python`` is the runtime-rank surface: it asks its operand for
    // ``rank()``, which a compile-time Tensor does not have.
    RuntimeTensor<T> A("A", std::vector<size_t>{5, 5});
    for (size_t ii = 0; ii < 5; ii++) {
        for (size_t jj = 0; jj < 5; jj++) {
            A(ii, jj) = awkward<T>(0.3 * static_cast<double>(ii + 1) - 0.125 * static_cast<double>(jj), 0.2 * static_cast<double>(jj + 1));
        }
    }
    RuntimeTensor<T> R("R", std::vector<size_t>{1});
    R.zero();

    cg::Graph graph("builder_trace_python");
    {
        cg::CaptureGuard const guard(graph);
        cg::trace_python(&R, A);
    }

    graph.execute();
    auto const captured = bytes_of(R);

    R(0) = T{};
    REQUIRE(rebuild_from_snapshots(graph) == 1);
    graph.execute();

    REQUIRE(bytes_of(R) == captured);
}

TEST_CASE("ExecutorBuilder - a rebuilt Trace reads a view's diagonal, not its parent's", "[ComputeGraph][ExecutorBuilder][Trace]") {
    // A strided operand is where a rank-erased diagonal walk can go wrong: the
    // impl's subscript has to apply BOTH strides, not assume a contiguous
    // square. A slice of a wider parent has neither.
    auto wide = create_random_tensor<double>("wide", 6, 6);
    auto sub  = wide(Range{1, 4}, Range{2, 5});

    double result = 0.0;

    cg::Graph graph("trace_view");
    {
        cg::CaptureGuard const guard(graph);
        cg::trace(&result, sub);
    }

    REQUIRE(rebuild_executors(graph) == 1);
    graph.execute();

    double want = 0.0;
    for (size_t ii = 0; ii < 3; ii++) {
        want += wide(1 + ii, 2 + ii);
    }
    REQUIRE_THAT(result, Catch::Matchers::WithinAbs(want, 1e-14));
}

TEST_CASE("ExecutorBuilder - a rebuilt Trace refuses a non-square operand at execute", "[ComputeGraph][ExecutorBuilder][Trace]") {
    // The square check travels with the executor rather than being settled at
    // capture, because a rebind validates against the SLOT and can seat a
    // different shape under the node between replays.
    auto square = create_random_tensor<double>("square", 4, 4);
    auto oblong = create_random_tensor<double>("oblong", 4, 2);

    double result = 0.0;

    cg::Graph graph("trace_square_guard");
    {
        cg::CaptureGuard const guard(graph);
        cg::trace(&result, square);
    }

    REQUIRE(rebuild_executors(graph) == 1);
    graph.execute();
    REQUIRE(result != 0.0);

    // Reach past rebind()'s dim validation by repointing the slot directly, the
    // way a pass that reshapes an intermediate would. Same static type, so the
    // slot's geometry accessor still decodes what it is handed.
    cg::TensorId const id = graph.find_tensor_id_by_ptr(&square);
    REQUIRE(id != 0);
    cg::TensorSlot *slot = graph.find_slot(id);
    REQUIRE(slot != nullptr);
    slot->ptr = &oblong;

    REQUIRE_THROWS_WITH(graph.execute(), Catch::Matchers::ContainsSubstring("square"));
}

// ── WriteParam ─────────────────────────────────────────────────────────────
//
// The first kind whose reconstructible BIT is true while one of its arms is
// blocked. The tensor arm names a graph scalar and rebuilds; the callback arm
// holds a std::function, which is one of the four closures Part 3.2 of the
// design inventories and which Part 3.3's BoundExpr work retires.

TEST_CASE("ExecutorBuilder - a rebuilt WriteParam reads the graph's source, not a captured reference",
          "[ComputeGraph][ExecutorBuilder][WriteParam]") {
    std::int64_t first  = 3;
    std::int64_t second = 1;

    cg::Pipeline pipe("wp_rebuild");
    pipe.set_param("n", 0);

    cg::Graph *stage = nullptr;
    {
        stage = &pipe.add_stage("s");
        cg::CaptureGuard const guard(*stage);
        cg::write_param("n", first);
    }

    pipe.execute();
    REQUIRE(pipe.get_param("n") == 3);

    REQUIRE(rebuild_executors(*stage) == 1);
    pipe.execute();
    REQUIRE(pipe.get_param("n") == 3);

    // Repoint the graph's handle for the source scalar. A registered scalar has
    // no slot, so this IS what a rebind of one looks like - and it is exactly
    // what the captured executor, holding a ``T &`` to the caller's variable,
    // could not see.
    auto const *node = find_node(*stage, cg::OpKind::WriteParam);
    REQUIRE(node != nullptr);
    auto const *desc = std::get_if<cg::WriteParamDescriptor>(&node->op_data);
    REQUIRE(desc != nullptr);
    REQUIRE(desc->source_id != 0);
    REQUIRE(desc->source_type == cg::ParamSourceType::Int64);

    stage->tensor(desc->source_id).tensor_ptr = &second;
    pipe.execute();
    REQUIRE(pipe.get_param("n") == 1);
}

TEST_CASE("ExecutorBuilder - a rebuilt WriteParam decodes a narrower source", "[ComputeGraph][ExecutorBuilder][WriteParam]") {
    // ``write_param`` takes any arithmetic scalar and narrows it to the int64 a
    // ParamTable holds. TensorHandle::dtype reports Unknown for every integral
    // type, so the descriptor has to carry the storage type itself or a rebuilt
    // executor would decode the wrong width.
    int source = 5;

    cg::Pipeline pipe("wp_narrow");
    pipe.set_param("n", 0);

    cg::Graph *stage = nullptr;
    {
        stage = &pipe.add_stage("s");
        cg::CaptureGuard const guard(*stage);
        cg::write_param("n", source);
    }

    auto const *node = find_node(*stage, cg::OpKind::WriteParam);
    REQUIRE(node != nullptr);
    REQUIRE(std::get_if<cg::WriteParamDescriptor>(&node->op_data)->source_type == cg::ParamSourceType::Int);

    REQUIRE(rebuild_executors(*stage) == 1);
    pipe.execute();
    REQUIRE(pipe.get_param("n") == 5);
}

TEST_CASE("ExecutorBuilder - a WriteParam tensor arm reports no blocker", "[ComputeGraph][ExecutorBuilder][WriteParam]") {
    std::int64_t source = 2;

    cg::Pipeline pipe("wp_clean");
    pipe.set_param("n", 0);

    cg::Graph *stage = nullptr;
    {
        stage = &pipe.add_stage("s");
        cg::CaptureGuard const guard(*stage);
        cg::write_param("n", source);
    }

    auto const report = stage->serializability_report();
    if (!report.empty()) {
        INFO("first blocker: " << report.front().kind_name << " - " << report.front().reason);
        FAIL("a write_param reading a graph scalar should report no blocker");
    }
    REQUIRE(report.empty());
}

TEST_CASE("ExecutorBuilder - the callback arm of write_param blocks a save the kind does not",
          "[ComputeGraph][ExecutorBuilder][WriteParam]") {
    cg::Pipeline pipe("wp_callback");
    pipe.set_param("n", 0);

    cg::Graph *stage = nullptr;
    {
        stage = &pipe.add_stage("s");
        cg::CaptureGuard const guard(*stage);
        cg::write_param("n", std::function<std::int64_t()>([] { return std::int64_t{7}; }));
    }

    // The kind's bit says yes and this node still cannot be saved. That split
    // is the point: only the per-node verdict can see an arm.
    REQUIRE(cg::is_reconstructible(cg::OpKind::WriteParam));

    auto const report = stage->serializability_report();
    REQUIRE(report.size() == 1);
    REQUIRE(report.front().kind_name == "WriteParam");
    REQUIRE_THAT(report.front().reason, Catch::Matchers::ContainsSubstring("callback"));

    // And it still runs: the callback arm keeps its capture-baked executor.
    pipe.execute();
    REQUIRE(pipe.get_param("n") == 7);
}

// ── Gemm ───────────────────────────────────────────────────────────────────

TEMPLATE_TEST_CASE("ExecutorBuilder - rebuilt Gemm is bitwise identical", "[ComputeGraph][ExecutorBuilder][Gemm]", float, double,
                   std::complex<float>, std::complex<double>) {
    using T = TestType;

    auto A = create_random_tensor<T>("A", 5, 4);
    auto B = create_random_tensor<T>("B", 4, 6);
    auto C = create_random_tensor<T>("C", 5, 6);

    auto const a0 = bytes_of(A);
    auto const b0 = bytes_of(B);
    auto const c0 = bytes_of(C);

    cg::Graph graph("builder_gemm");
    {
        cg::CaptureGuard const guard(graph);
        // beta != 0 so C's prior contents participate, which also exercises the
        // RMW input list (C repeated as inputs[2]).
        cg::gemm<false, false>(awkward<T>(0.6, -0.25), A, B, awkward<T>(0.35, 0.4), &C);
    }

    graph.execute();
    auto const captured = bytes_of(C);
    REQUIRE(captured != c0);

    restore(&A, a0);
    restore(&B, b0);
    restore(&C, c0);
    REQUIRE(rebuild_from_snapshots(graph) == 1);
    graph.execute();

    REQUIRE(bytes_of(C) == captured);
}

TEST_CASE("ExecutorBuilder - rebuilt Gemm covers every transpose combination", "[ComputeGraph][ExecutorBuilder][Gemm]") {
    // The descriptor records BLAS chars, and getting one wrong reads an operand
    // in the other orientation - which still runs, and on a square operand
    // still produces plausible numbers. All four combinations, against the same
    // mathematical product.
    constexpr size_t m = 4;
    constexpr size_t k = 3;
    constexpr size_t n = 5;

    auto const check = [&]<bool TA, bool TB>() {
        auto A = TA ? create_random_tensor<double>("A", k, m) : create_random_tensor<double>("A", m, k);
        auto B = TB ? create_random_tensor<double>("B", n, k) : create_random_tensor<double>("B", k, n);
        auto C = create_zero_tensor<double>("C", m, n);

        cg::Graph graph("builder_gemm_trans");
        {
            cg::CaptureGuard const guard(graph);
            cg::gemm<TA, TB>(1.0, A, B, 0.0, &C);
        }

        graph.execute();
        auto const captured = bytes_of(C);

        C.zero();
        REQUIRE(rebuild_executors(graph) == 1);
        graph.execute();
        REQUIRE(bytes_of(C) == captured);

        // Against the arithmetic, so a builder that dropped BOTH flags cannot
        // pass by agreeing with a capture that dropped them too.
        for (size_t ii = 0; ii < m; ii++) {
            for (size_t jj = 0; jj < n; jj++) {
                double want = 0.0;
                for (size_t kk = 0; kk < k; kk++) {
                    want += (TA ? A(kk, ii) : A(ii, kk)) * (TB ? B(jj, kk) : B(kk, jj));
                }
                INFO("trans_a=" << TA << " trans_b=" << TB << " at (" << ii << "," << jj << ")");
                REQUIRE_THAT(C(ii, jj), Catch::Matchers::WithinAbs(want, 1e-12));
            }
        }
    };

    check.template operator()<false, false>();
    check.template operator()<false, true>();
    check.template operator()<true, false>();
    check.template operator()<true, true>();
}

TEMPLATE_TEST_CASE("ExecutorBuilder - rebuilt Gemm honors a conjugate transpose", "[ComputeGraph][ExecutorBuilder][Gemm]",
                   std::complex<float>, std::complex<double>) {
    using T = TestType;

    // The runtime-char overload exists to reach BLAS 'c', which no pair of
    // bools can spell. The descriptor records chars for exactly this case.
    RuntimeTensor<T> A("A", std::vector<size_t>{3, 4});
    RuntimeTensor<T> B("B", std::vector<size_t>{4, 2});
    RuntimeTensor<T> C("C", std::vector<size_t>{3, 2});
    for (size_t ii = 0; ii < 3; ii++) {
        for (size_t jj = 0; jj < 4; jj++) {
            A(ii, jj) = awkward<T>(0.1 * static_cast<double>(ii + 1), 0.3 * static_cast<double>(jj + 1));
        }
    }
    for (size_t ii = 0; ii < 4; ii++) {
        for (size_t jj = 0; jj < 2; jj++) {
            B(ii, jj) = awkward<T>(0.7 * static_cast<double>(jj + 1), -0.2 * static_cast<double>(ii + 1));
        }
    }
    C.zero();

    cg::Graph graph("builder_gemm_conj");
    {
        cg::CaptureGuard const guard(graph);
        cg::gemm(T{1}, A, B, T{0}, &C, linear_algebra::Transpose::N, linear_algebra::Transpose::N);
    }

    auto const *node = find_node(graph, cg::OpKind::Gemm);
    REQUIRE(node != nullptr);
    auto const *desc = std::get_if<cg::GemmDescriptor>(&node->op_data);
    REQUIRE(desc != nullptr);
    // Transpose::N is the BLAS character 'N'; the descriptor records it as given.
    REQUIRE(desc->trans_a == 'N');

    graph.execute();
    auto const plain = bytes_of_scalar(C(0, 0));

    // Now the conjugate-transposed form of the same product, which must differ
    // and must survive a rebuild.
    RuntimeTensor<T> Ah("Ah", std::vector<size_t>{4, 3});
    for (size_t ii = 0; ii < 3; ii++) {
        for (size_t jj = 0; jj < 4; jj++) {
            Ah(jj, ii) = std::conj(A(ii, jj));
        }
    }
    RuntimeTensor<T> D("D", std::vector<size_t>{3, 2});
    D.zero();

    cg::Graph conj_graph("builder_gemm_conj_c");
    {
        cg::CaptureGuard const guard(conj_graph);
        cg::gemm(T{1}, Ah, B, T{0}, &D, linear_algebra::Transpose::C, linear_algebra::Transpose::N);
    }

    auto const *conj_node = find_node(conj_graph, cg::OpKind::Gemm);
    REQUIRE(conj_node != nullptr);
    REQUIRE(std::get_if<cg::GemmDescriptor>(&conj_node->op_data)->trans_a == 'C');

    conj_graph.execute();
    auto const captured = bytes_of(D);

    D.zero();
    REQUIRE(rebuild_executors(conj_graph) == 1);
    conj_graph.execute();

    REQUIRE(bytes_of(D) == captured);
    // conj(conj(A))^T^T is A, so the two products agree - which is the check
    // that 'c' really was applied rather than silently read as 't'.
    REQUIRE(bytes_of_scalar(D(0, 0)) == plain);
}

TEST_CASE("ExecutorBuilder - a rebuilt Gemm follows redirect_slot", "[ComputeGraph][ExecutorBuilder][Gemm][CSE]") {
    auto X   = create_random_tensor<double>("X", 4, 4);
    auto Y   = create_random_tensor<double>("Y", 4, 4);
    auto B   = create_random_tensor<double>("B", 4, 4);
    auto out = create_zero_tensor<double>("out", 4, 4);
    auto aux = create_zero_tensor<double>("aux", 4, 4);

    cg::Graph graph("gemm_redirect");
    {
        cg::CaptureGuard const guard(graph);
        cg::gemm<false, false>(1.0, X, B, 0.0, &out);
        cg::gemm<false, false>(1.0, Y, B, 0.0, &aux); // only so Y gets a slot
    }

    REQUIRE(rebuild_executors(graph) == 2);

    cg::TensorId const x_id = graph.find_tensor_id_by_ptr(&X);
    cg::TensorId const y_id = graph.find_tensor_id_by_ptr(&Y);
    REQUIRE(x_id != 0);
    REQUIRE(y_id != 0);

    graph.redirect_slot(x_id, y_id);
    graph.execute();

    REQUIRE(bytes_of(out) == bytes_of(aux));
}

TEST_CASE("ExecutorBuilder - a rebuilt Gemm writes the rebound storage", "[ComputeGraph][ExecutorBuilder][Gemm][Rebind]") {
    auto A    = create_random_tensor<double>("A", 4, 4);
    auto B    = create_random_tensor<double>("B", 4, 4);
    auto out  = create_zero_tensor<double>("out", 4, 4);
    auto out2 = create_zero_tensor<double>("out2", 4, 4);

    cg::Graph graph("gemm_rebind");
    {
        cg::CaptureGuard const guard(graph);
        cg::gemm<false, false>(1.0, A, B, 0.0, &out);
    }

    REQUIRE(rebuild_executors(graph) == 1);

    graph.rebind(out, out2);
    graph.execute();

    for (size_t ii = 0; ii < 4; ii++) {
        for (size_t jj = 0; jj < 4; jj++) {
            double want = 0.0;
            for (size_t kk = 0; kk < 4; kk++) {
                want += A(ii, kk) * B(kk, jj);
            }
            REQUIRE_THAT(out2(ii, jj), Catch::Matchers::WithinAbs(want, 1e-12));
            REQUIRE(out(ii, jj) == 0.0); // the whole node moved, not half of it
        }
    }
}

TEST_CASE("ExecutorBuilder - a Gemm prefactor rewritten after capture is applied", "[ComputeGraph][ExecutorBuilder][Gemm]") {
    // Gemm prefactors are snapshots, with no live block, because no pass
    // rewrites them today (see GemmDescriptor). The rule that applies until one
    // does is the header's: rewrite the descriptor and rebuild.
    auto A = create_random_tensor<double>("A", 3, 3);
    auto B = create_random_tensor<double>("B", 3, 3);
    auto C = create_zero_tensor<double>("C", 3, 3);

    cg::Graph graph("gemm_rewrite");
    {
        cg::CaptureGuard const guard(graph);
        cg::gemm<false, false>(1.0, A, B, 0.0, &C);
    }

    auto *node = find_node(graph, cg::OpKind::Gemm);
    REQUIRE(node != nullptr);
    std::get_if<cg::GemmDescriptor>(&node->op_data)->alpha = cg::PrefactorScalar{2.5};

    REQUIRE(rebuild_executors(graph) == 1);
    graph.execute();

    for (size_t ii = 0; ii < 3; ii++) {
        for (size_t jj = 0; jj < 3; jj++) {
            double want = 0.0;
            for (size_t kk = 0; kk < 3; kk++) {
                want += A(ii, kk) * B(kk, jj);
            }
            REQUIRE_THAT(C(ii, jj), Catch::Matchers::WithinAbs(2.5 * want, 1e-12));
        }
    }
}

TEST_CASE("ExecutorBuilder - make_gemm_executor and the builder are one kernel", "[ComputeGraph][ExecutorBuilder][Gemm]") {
    // The Graph method is a forwarder onto build_executor now. It used to cast
    // TensorHandle::tensor_ptr straight to Tensor<T, 2>* whatever the tensor
    // was, so a RuntimeTensor operand was type confusion; going through the
    // builder it is just another impl.
    RuntimeTensor<double> A("A", std::vector<size_t>{3, 3});
    RuntimeTensor<double> B("B", std::vector<size_t>{3, 3});
    RuntimeTensor<double> C("C", std::vector<size_t>{3, 3});
    for (size_t ii = 0; ii < 3; ii++) {
        for (size_t jj = 0; jj < 3; jj++) {
            A(ii, jj) = 0.5 * static_cast<double>(ii + 1) + 0.25 * static_cast<double>(jj);
            B(ii, jj) = 1.5 - 0.125 * static_cast<double>(ii * 3 + jj);
        }
    }
    C.zero();

    cg::Graph graph("make_gemm_executor");
    {
        cg::CaptureGuard const guard(graph);
        cg::gemm<false, false>(1.0, A, B, 0.0, &C);
    }
    graph.execute();
    auto const captured = bytes_of_scalar(C(2, 2));

    cg::TensorId const a_id = graph.find_tensor_id_by_ptr(&A);
    cg::TensorId const b_id = graph.find_tensor_id_by_ptr(&B);
    cg::TensorId const c_id = graph.find_tensor_id_by_ptr(&C);
    REQUIRE(a_id != 0);
    REQUIRE(b_id != 0);
    REQUIRE(c_id != 0);

    C.zero();
    auto direct = graph.make_gemm_executor(a_id, b_id, c_id, 1.0, 0.0);
    direct();

    REQUIRE(bytes_of_scalar(C(2, 2)) == captured);
}

TEST_CASE("ExecutorBuilder - a graph of gemms and dots reports no blockers", "[ComputeGraph][ExecutorBuilder][Gemm][Dot]") {
    auto A = create_random_tensor<double>("A", 4, 4);
    auto B = create_random_tensor<double>("B", 4, 4);
    auto C = create_zero_tensor<double>("C", 4, 4);
    auto R = create_zero_tensor<double>("R", 1);

    double e = 0.0;
    double t = 0.0;

    cg::Graph graph("report_gemm_dot");
    {
        cg::CaptureGuard const guard(graph);
        cg::gemm<false, true>(1.0, A, B, 0.0, &C);
        cg::dot(&e, A, C);
        cg::dot_python(&R, A, B);
        cg::trace(&t, C);
    }

    auto const report = graph.serializability_report();
    if (!report.empty()) {
        INFO("first blocker: " << report.front().kind_name << " - " << report.front().reason);
        FAIL("a graph of converted kinds should report no blockers");
    }
    REQUIRE(report.empty());
    REQUIRE(rebuild_from_snapshots(graph) == 4);
}

// ── Pass-emitted Gemm nodes ────────────────────────────────────────────────

TEST_CASE("ExecutorBuilder - ContractionPlanning's Gemm nodes are data now", "[ComputeGraph][ExecutorBuilder][Gemm][CP]") {
    // The pass emitted OpKind::Gemm nodes whose executor was a closure baked at
    // pass time with the dims, the transpose flags and the scalars all inside
    // it - a node wearing a kind that promised a descriptor it did not have.
    // They carry a GemmDescriptor and go through the builder now, so they are
    // reconstructible; what must not change is the numbers.
    //
    // The cost model is chosen (as the pass's own tests do) so the cheap
    // parenthesization wins and a fold is guaranteed.
    cg::CostModel model;
    model.cpu.peak_gflops_fp64          = 100.0;
    model.cpu.mem_bandwidth_gbps        = 40.0;
    model.cpu.kernel_launch_overhead_us = 0.1;
    model.cpu.name                      = "TestCPU";

    auto const build = [](cg::Graph &g, Tensor<double, 2> const &A, Tensor<double, 2> const &B, Tensor<double, 2> const &D,
                          Tensor<double, 2> &out) {
        auto                  &mid = g.create_zero_tensor<double, 2>("mid", 100, 100);
        cg::CaptureGuard const guard(g);
        cg::einsum("ik;kj->ij", 0.0, &mid, 1.0, A, B);
        cg::einsum("ik;kj->ij", 0.0, &out, 1.0, mid, D);
    };

    auto A = create_random_tensor<double>("A", 100, 1);
    auto B = create_random_tensor<double>("B", 1, 100);
    auto D = create_random_tensor<double>("D", 100, 2);

    // Reference: the same graph, left unfolded.
    auto      reference = create_zero_tensor<double>("reference", 100, 2);
    cg::Graph plain("cp_reference");
    build(plain, A, B, D, reference);
    plain.execute();

    auto      folded = create_zero_tensor<double>("folded", 100, 2);
    cg::Graph graph("cp_folded");
    build(graph, A, B, D, folded);

    cg::passes::ContractionPlanning pass(model);
    pass.run(graph);
    REQUIRE(pass.chains_restructured() == 1);

    size_t gemms = 0;
    for (auto const &node : graph.nodes()) {
        if (node.kind != cg::OpKind::Gemm) {
            continue;
        }
        ++gemms;
        INFO("node: " << node.label);
        // Absent from the report: the descriptor is there and it is the right one.
        REQUIRE(cg::reconstruction_blocker(node).empty());
        REQUIRE(std::holds_alternative<cg::GemmDescriptor>(node.op_data));
    }
    REQUIRE(gemms > 0);

    for (auto const &blocker : graph.serializability_report()) {
        INFO("blocker: " << blocker.kind_name << " - " << blocker.reason);
        REQUIRE(blocker.kind_name != "Gemm");
    }

    graph.execute();
    for (size_t ii = 0; ii < 100; ii++) {
        for (size_t jj = 0; jj < 2; jj++) {
            REQUIRE_THAT(folded(ii, jj), Catch::Matchers::WithinAbs(reference(ii, jj), 1e-9));
        }
    }

    // And a rebuild from the descriptor reproduces the folded chain exactly.
    auto const first = bytes_of(folded);
    folded.zero();
    REQUIRE(rebuild_from_snapshots(graph) > 0);
    graph.execute();
    REQUIRE(bytes_of(folded) == first);
}

// ── ElementTransform: the named kernel ─────────────────────────────────────
//
// The kind's descriptor is a NAME, and the registry resolves it at build time.
// What has to hold is that a rebuild lands on the same kernel: an element map
// reached through a different route would be a silent numerical change exactly
// as a differently routed einsum would.

TEMPLATE_TEST_CASE("ExecutorBuilder - a rebuilt named ElementTransform is bitwise identical", "[ComputeGraph][ExecutorBuilder]", float,
                   double, std::complex<float>, std::complex<double>) {
    using T = TestType;

    auto A       = create_random_tensor<T>("A", 5, 3);
    auto initial = bytes_of(A);

    cg::Graph graph("builder_element_transform");
    {
        cg::CaptureGuard const guard(graph);
        cg::element_transform(&A, "square");
    }

    graph.execute();
    auto const captured = bytes_of(A);
    REQUIRE(captured != initial);

    restore(&A, initial);
    REQUIRE(rebuild_executors(graph) == 1);
    graph.execute();

    REQUIRE(bytes_of(A) == captured);
}

TEST_CASE("ExecutorBuilder - a named ElementTransform rebuilds from its snapshot alone", "[ComputeGraph][ExecutorBuilder]") {
    // Nothing but the op name is recorded, so "strip the live handles" is a
    // no-op here and the rebuild is honest by construction. The test is that
    // the name really is enough.
    auto A       = create_random_tensor<double>("A", 4, 4);
    auto initial = bytes_of(A);

    cg::Graph graph("element_transform_snapshot");
    {
        cg::CaptureGuard const guard(graph);
        cg::element_transform(&A, "negate");
    }

    graph.execute();
    auto const captured = bytes_of(A);

    restore(&A, initial);
    REQUIRE(rebuild_from_snapshots(graph) == 1);
    graph.execute();
    REQUIRE(bytes_of(A) == captured);
}

TEST_CASE("ExecutorBuilder - an anonymous ElementTransform still runs and still blocks a save", "[ComputeGraph][ExecutorBuilder]") {
    auto named     = create_random_tensor<double>("named", 3, 3);
    auto anonymous = named;

    cg::Graph graph("element_transform_mixed");
    {
        cg::CaptureGuard const guard(graph);
        cg::element_transform(&named, "square");
        cg::element_transform(&anonymous, [](double v) { return v * v; });
    }
    graph.execute();

    // Same math from both spellings...
    REQUIRE(bytes_of(named) == bytes_of(anonymous));

    // ...and only the anonymous one blocks.
    auto const report = graph.serializability_report();
    REQUIRE(report.size() == 1);
    REQUIRE(report.front().kind_name == "ElementTransform");
    REQUIRE_THAT(report.front().reason, Catch::Matchers::ContainsSubstring("anonymous closure"));
    REQUIRE(rebuild_executors(graph) == 1);
}

TEST_CASE("ExecutorBuilder - a rebuilt named ElementTransform follows a redirect", "[ComputeGraph][ExecutorBuilder]") {
    // The property every converted kind has to have: operands are resolved
    // through the graph's slots, so repointing one moves what the executor
    // touches.
    auto first  = create_random_tensor<double>("first", 3, 3);
    auto second = create_random_tensor<double>("second", 3, 3);

    auto const first_before  = bytes_of(first);
    auto const second_before = bytes_of(second);

    cg::Graph graph("element_transform_redirect");
    {
        cg::CaptureGuard const guard(graph);
        cg::element_transform(&first, "negate");
        cg::element_transform(&second, "square"); // only here so `second` gets a slot to redirect to
    }

    REQUIRE(rebuild_executors(graph) == 2);

    cg::TensorId const first_id  = graph.find_tensor_id_by_ptr(&first);
    cg::TensorId const second_id = graph.find_tensor_id_by_ptr(&second);
    REQUIRE(first_id != 0);
    REQUIRE(second_id != 0);

    graph.redirect_slot(first_id, second_id);
    graph.execute();

    // Both nodes now write `second`, and nothing touches `first`.
    REQUIRE(bytes_of(first) == first_before);
    REQUIRE(bytes_of(second) != second_before);
}

// ── PredExpr: a conditional and a loop become data ──────────────────────────
//
// The three closures Part 3.2 of the design inventoried beside the kernels are
// a conditional's predicate, a loop's condition and a write_param's source.
// All three are BoundExpr-shaped now, so the arm a node holds is what decides
// whether it can be saved - and a callback arm still runs, unchanged.

TEST_CASE("PredExpr - the arms resolve to what they say", "[ComputeGraph][ExecutorBuilder][PredExpr]") {
    cg::ParamTable params;
    params.set("n", 7);

    REQUIRE(cg::PredExpr{}.resolve(&params)); // an unset predicate is true
    REQUIRE(cg::PredExpr{true}.resolve(&params));
    REQUIRE_FALSE(cg::PredExpr{false}.resolve(&params));

    REQUIRE(cg::PredExpr::compare(cg::BoundExpr{"n"}, cg::CmpOp::Eq, cg::BoundExpr{7}).resolve(&params));
    REQUIRE(cg::PredExpr::compare(cg::BoundExpr{"n"}, cg::CmpOp::Gt, cg::BoundExpr{6}).resolve(&params));
    REQUIRE_FALSE(cg::PredExpr::compare(cg::BoundExpr{"n"}, cg::CmpOp::Lt, cg::BoundExpr{7}).resolve(&params));
    REQUIRE(cg::PredExpr::compare(cg::BoundExpr{"n"}, cg::CmpOp::Le, cg::BoundExpr{7}).resolve(&params));
    REQUIRE(cg::PredExpr::compare(cg::BoundExpr{"n"}, cg::CmpOp::Ne, cg::BoundExpr{0}).resolve(&params));
    REQUIRE(cg::PredExpr::compare(cg::BoundExpr{"n"}, cg::CmpOp::Ge, cg::BoundExpr{7}).resolve(&params));

    // The iteration arm reads the loop's counter, not the table.
    REQUIRE(cg::PredExpr::iteration(cg::CmpOp::Lt, cg::BoundExpr{3}).resolve(&params, 2));
    REQUIRE_FALSE(cg::PredExpr::iteration(cg::CmpOp::Lt, cg::BoundExpr{3}).resolve(&params, 3));

    cg::GateFlags flags(4, false);
    flags.set(2, true);
    REQUIRE(cg::PredExpr::flag(flags, 2).resolve(&params));
    REQUIRE_FALSE(cg::PredExpr::flag(flags, 1).resolve(&params));
    REQUIRE_FALSE(cg::PredExpr::flag(flags, 99).resolve(&params)); // past the end reads false

    REQUIRE(cg::PredExpr::callback(std::function<bool()>{[] { return true; }}).resolve(&params));
    REQUIRE(cg::PredExpr::callback(std::function<bool(std::size_t)>{[](std::size_t it) { return it < 2; }}).resolve(&params, 1));
}

TEST_CASE("PredExpr - a named parameter with no table bound is an error, not a default", "[ComputeGraph][ExecutorBuilder][PredExpr]") {
    auto const pred = cg::PredExpr::compare(cg::BoundExpr{"n"}, cg::CmpOp::Gt, cg::BoundExpr{0});
    REQUIRE_THROWS_WITH(pred.resolve(nullptr), Catch::Matchers::ContainsSubstring("n"));

    // A predicate that names nothing resolves happily without one.
    REQUIRE(cg::PredExpr::compare(cg::BoundExpr{2}, cg::CmpOp::Gt, cg::BoundExpr{1}).resolve(nullptr));
}

TEST_CASE("PredExpr - only a closure blocks a save", "[ComputeGraph][ExecutorBuilder][PredExpr]") {
    REQUIRE_FALSE(cg::PredExpr{true}.names_a_closure());
    REQUIRE_FALSE(cg::PredExpr::compare(cg::BoundExpr{"n"}, cg::CmpOp::Gt, cg::BoundExpr{0}).names_a_closure());
    REQUIRE_FALSE(cg::PredExpr::iteration(cg::CmpOp::Lt, cg::BoundExpr{3}).names_a_closure());
    REQUIRE(cg::PredExpr::callback(std::function<bool()>{[] { return true; }}).names_a_closure());

    // A Compare arm whose OPERAND is a callback is a closure too: the arm is
    // data and its content is not, which is exactly the case a per-arm check
    // that only looked at the top level would miss.
    auto const sneaky =
        cg::PredExpr::compare(cg::BoundExpr{std::function<std::int64_t()>{[] { return 1; }}}, cg::CmpOp::Gt, cg::BoundExpr{0});
    REQUIRE_FALSE(sneaky.is_callback());
    REQUIRE(sneaky.names_a_closure());
}

TEST_CASE("ExecutorBuilder - a Compare conditional picks a branch from the parameter table", "[ComputeGraph][ExecutorBuilder][PredExpr]") {
    auto value = create_zero_tensor<double>("value", 2, 2);

    cg::Graph graph("compare_conditional");
    {
        auto [then_g, else_g] =
            graph.add_conditional("gate", cg::PredExpr::compare(cg::BoundExpr{"take_then"}, cg::CmpOp::Ne, cg::BoundExpr{0}));
        {
            cg::CaptureGuard const guard(then_g);
            cg::scale(2.0, &value);
        }
        {
            cg::CaptureGuard const guard(else_g);
            cg::scale(3.0, &value);
        }
    }

    // Neither the node nor its predicate blocks a save.
    auto const *node = find_node(graph, cg::OpKind::Conditional);
    REQUIRE(node != nullptr);
    REQUIRE(cg::reconstruction_blocker(*node).empty());

    // The same replayed graph takes both branches as the parameter flips.
    value(0, 0) = 1.0;
    graph.params_ptr()->set("take_then", 1);
    graph.execute();
    REQUIRE(value(0, 0) == 2.0);

    graph.params_ptr()->set("take_then", 0);
    graph.execute();
    REQUIRE(value(0, 0) == 6.0);
}

TEST_CASE("ExecutorBuilder - a rebuilt Compare conditional replays its subgraph identically", "[ComputeGraph][ExecutorBuilder][PredExpr]") {
    auto A = create_random_tensor<double>("A", 4, 3);
    auto B = create_random_tensor<double>("B", 3, 5);
    auto C = create_zero_tensor<double>("C", 4, 5);

    cg::Graph graph("rebuilt_conditional");
    {
        auto [then_g, else_g] = graph.add_conditional("gate", cg::PredExpr::compare(cg::BoundExpr{"go"}, cg::CmpOp::Gt, cg::BoundExpr{0}));
        cg::CaptureGuard const guard(then_g);
        cg::einsum("ik;kj->ij", &C, A, B);
    }
    graph.params_ptr()->set("go", 1);
    graph.execute();
    auto const captured = bytes_of(C);
    REQUIRE(captured != bytes_of(create_zero_tensor<double>("zero", 4, 5)));

    C.zero();
    REQUIRE(rebuild_executors(graph) == 1);
    graph.execute();
    REQUIRE(bytes_of(C) == captured);

    // And the rebuilt node still honors a false predicate.
    C.zero();
    graph.params_ptr()->set("go", 0);
    graph.execute();
    REQUIRE(bytes_of(C) == bytes_of(create_zero_tensor<double>("zero", 4, 5)));
}

TEST_CASE("ExecutorBuilder - add_conditional_flag records the FlagTest arm", "[ComputeGraph][ExecutorBuilder][PredExpr][GateFlags]") {
    // The gate-flag conditional used to close over the shared buffer in a
    // lambda. It is the same load expressed as data now, so the behavior is
    // unchanged and the node stops blocking a save.
    auto value  = create_zero_tensor<double>("value", 2, 2);
    value(0, 0) = 1.0;

    cg::GateFlags flags(2, false);

    cg::Graph graph("flag_conditional");
    {
        auto [then_g, else_g] = graph.add_conditional_flag("gate", flags, 0);
        cg::CaptureGuard const guard(then_g);
        cg::scale(2.0, &value);
    }

    auto const *node = find_node(graph, cg::OpKind::Conditional);
    REQUIRE(node != nullptr);
    auto const *desc = std::get_if<cg::ConditionalDescriptor>(&node->op_data);
    REQUIRE(desc != nullptr);
    REQUIRE(desc->predicate.is_flag_test());
    REQUIRE(cg::reconstruction_blocker(*node).empty());

    graph.execute();
    REQUIRE(value(0, 0) == 1.0); // flag clear: the branch is skipped

    flags.set(0, true);
    graph.execute();
    REQUIRE(value(0, 0) == 2.0);

    // Writes to the array reach a REBUILT executor too, because the buffer is
    // shared rather than copied.
    REQUIRE(rebuild_executors(graph) == 1);
    flags.set(0, false);
    graph.execute();
    REQUIRE(value(0, 0) == 2.0);
}

TEST_CASE("ExecutorBuilder - a callback conditional runs and reports its arm", "[ComputeGraph][ExecutorBuilder][PredExpr]") {
    auto value  = create_zero_tensor<double>("value", 2, 2);
    value(0, 0) = 1.0;
    bool take   = true;

    cg::Graph graph("callback_conditional");
    {
        auto [then_g, else_g] = graph.add_conditional("gate", [&take] { return take; });
        cg::CaptureGuard const guard(then_g);
        cg::scale(2.0, &value);
    }

    graph.execute();
    REQUIRE(value(0, 0) == 2.0);
    take = false;
    graph.execute();
    REQUIRE(value(0, 0) == 2.0);

    auto const report = graph.serializability_report();
    REQUIRE(report.size() == 1);
    REQUIRE(report.front().kind_name == "Conditional");
    REQUIRE_THAT(report.front().reason, Catch::Matchers::ContainsSubstring("callback predicate"));
}

TEST_CASE("ExecutorBuilder - a loop condition can compare the iteration index", "[ComputeGraph][ExecutorBuilder][PredExpr]") {
    auto value  = create_zero_tensor<double>("value", 2, 2);
    value(0, 0) = 1.0;

    cg::Graph graph("iteration_loop");
    {
        // Continue while the iteration that just finished is below 2, so the
        // body runs for iterations 0, 1 and 2.
        auto                  &body = graph.add_loop("iter", 100, cg::PredExpr::iteration(cg::CmpOp::Lt, cg::BoundExpr{2}));
        cg::CaptureGuard const guard(body);
        cg::scale(2.0, &value);
    }

    auto const *node = find_node(graph, cg::OpKind::Loop);
    REQUIRE(node != nullptr);
    REQUIRE(cg::reconstruction_blocker(*node).empty());

    graph.execute();
    REQUIRE(value(0, 0) == 8.0);
}

TEST_CASE("ExecutorBuilder - a loop condition can compare parameters", "[ComputeGraph][ExecutorBuilder][PredExpr]") {
    auto value  = create_zero_tensor<double>("value", 2, 2);
    value(0, 0) = 1.0;

    cg::Graph graph("compare_loop");
    {
        auto &body = graph.add_loop("iter", 10, cg::PredExpr::compare(cg::BoundExpr{"keep_going"}, cg::CmpOp::Ne, cg::BoundExpr{0}));
        cg::CaptureGuard const guard(body);
        cg::scale(2.0, &value);
    }

    // A loop whose condition names a parameter is ordered against whatever
    // writes it, which is what param_reads exists to express.
    auto const *node = find_node(graph, cg::OpKind::Loop);
    REQUIRE(node != nullptr);
    REQUIRE(cg::param_reads(*node) == std::vector<std::string>{"keep_going"});

    graph.params_ptr()->set("keep_going", 0);
    graph.execute();
    REQUIRE(value(0, 0) == 2.0); // the body always runs at least once
}

TEST_CASE("ExecutorBuilder - a loop's iteration count is observable on the node", "[ComputeGraph][ExecutorBuilder][PredExpr]") {
    // It was not: the executor wrote the count into its own COPY of the
    // descriptor, so the field on the node stayed 0 no matter how many
    // iterations ran.
    auto value = create_zero_tensor<double>("value", 2, 2);

    cg::Graph graph("iteration_count");
    {
        auto                  &body = graph.add_loop("iter", 100, cg::PredExpr::iteration(cg::CmpOp::Lt, cg::BoundExpr{4}));
        cg::CaptureGuard const guard(body);
        cg::scale(1.0, &value);
    }

    // Looked up FRESH each time: execute() sorts the node vector, so a
    // descriptor pointer taken before it does not survive.
    auto const count = [&graph]() -> size_t {
        auto const *node = find_node(graph, cg::OpKind::Loop);
        REQUIRE(node != nullptr);
        auto const *desc = std::get_if<cg::LoopDescriptor>(&node->op_data);
        REQUIRE(desc != nullptr);
        return desc->last_iteration_count();
    };

    REQUIRE(count() == 0);

    graph.execute();
    REQUIRE(count() == 5);

    // A second run overwrites the count rather than accumulating it.
    graph.execute();
    REQUIRE(count() == 5);
}

TEST_CASE("ExecutorBuilder - a callback loop condition runs and reports its arm", "[ComputeGraph][ExecutorBuilder][PredExpr]") {
    auto value  = create_zero_tensor<double>("value", 2, 2);
    value(0, 0) = 1.0;

    cg::Graph graph("callback_loop");
    {
        auto                  &body = graph.add_loop("iter", 100, [](size_t iter) { return iter < 2; });
        cg::CaptureGuard const guard(body);
        cg::scale(2.0, &value);
    }

    graph.execute();
    REQUIRE(value(0, 0) == 8.0);

    auto const report = graph.serializability_report();
    REQUIRE(report.size() == 1);
    REQUIRE(report.front().kind_name == "Loop");
    REQUIRE_THAT(report.front().reason, Catch::Matchers::ContainsSubstring("callback condition"));
}

TEST_CASE("ExecutorBuilder - an absent loop condition still runs to the safety limit", "[ComputeGraph][ExecutorBuilder][PredExpr]") {
    // The behavior an empty std::function<bool(size_t)> has always had, now
    // spelled as a default PredExpr rather than as a null check in the executor.
    auto value  = create_zero_tensor<double>("value", 2, 2);
    value(0, 0) = 1.0;

    cg::Graph graph("unconditional_loop");
    {
        auto                  &body = graph.add_loop("iter", 3, std::function<bool(size_t)>{});
        cg::CaptureGuard const guard(body);
        cg::scale(2.0, &value);
    }

    graph.execute();
    REQUIRE(value(0, 0) == 8.0);
}

TEST_CASE("ExecutorBuilder - a rebuilt loop replays its body identically", "[ComputeGraph][ExecutorBuilder][PredExpr]") {
    auto A = create_random_tensor<double>("A", 4, 3);
    auto B = create_random_tensor<double>("B", 3, 5);
    auto C = create_zero_tensor<double>("C", 4, 5);

    cg::Graph graph("rebuilt_loop");
    {
        auto                  &body = graph.add_loop("iter", 10, cg::PredExpr::iteration(cg::CmpOp::Lt, cg::BoundExpr{2}));
        cg::CaptureGuard const guard(body);
        cg::einsum("ik;kj->ij", 1.0, &C, 1.0, A, B);
    }

    graph.execute();
    auto const captured = bytes_of(C);

    C.zero();
    REQUIRE(rebuild_executors(graph) == 1);
    graph.execute();
    REQUIRE(bytes_of(C) == captured);

    auto const *node = find_node(graph, cg::OpKind::Loop);
    REQUIRE(node != nullptr);
    auto const *desc = std::get_if<cg::LoopDescriptor>(&node->op_data);
    REQUIRE(desc != nullptr);
    REQUIRE(desc->last_iteration_count() == 3);
}

// ── write_param: the last of the four closures ─────────────────────────────

TEST_CASE("ExecutorBuilder - write_param takes a literal", "[ComputeGraph][ExecutorBuilder][PredExpr]") {
    cg::Graph graph("write_param_const");
    {
        cg::CaptureGuard const guard(graph);
        cg::write_param("n", cg::BoundExpr{11});
    }

    auto const *node = find_node(graph, cg::OpKind::WriteParam);
    REQUIRE(node != nullptr);
    REQUIRE(cg::reconstruction_blocker(*node).empty());
    REQUIRE(cg::param_reads(*node).empty()); // a literal names nothing
    REQUIRE(cg::param_writes(*node) == std::vector<std::string>{"n"});

    graph.execute();
    REQUIRE(graph.params_ptr()->get("n") == 11);

    // And it rebuilds from the descriptor alone.
    graph.params_ptr()->set("n", 0);
    REQUIRE(rebuild_executors(graph) == 1);
    graph.execute();
    REQUIRE(graph.params_ptr()->get("n") == 11);
}

TEST_CASE("ExecutorBuilder - write_param copies another parameter", "[ComputeGraph][ExecutorBuilder][PredExpr]") {
    cg::Graph graph("write_param_param");
    {
        cg::CaptureGuard const guard(graph);
        cg::write_param("dst", cg::BoundExpr{"src"});
    }

    auto const *node = find_node(graph, cg::OpKind::WriteParam);
    REQUIRE(node != nullptr);
    REQUIRE(cg::reconstruction_blocker(*node).empty());
    // Reads one name and writes another, so the scheduler can order it between
    // the two.
    REQUIRE(cg::param_reads(*node) == std::vector<std::string>{"src"});
    REQUIRE(cg::param_writes(*node) == std::vector<std::string>{"dst"});

    graph.params_ptr()->set("src", 5);
    graph.execute();
    REQUIRE(graph.params_ptr()->get("dst") == 5);

    graph.params_ptr()->set("src", 9);
    REQUIRE(rebuild_executors(graph) == 1);
    graph.execute();
    REQUIRE(graph.params_ptr()->get("dst") == 9);
}

TEST_CASE("ExecutorBuilder - a callback write_param runs and reports its arm", "[ComputeGraph][ExecutorBuilder][PredExpr]") {
    std::int64_t source = 3;

    cg::Graph graph("write_param_callback");
    {
        cg::CaptureGuard const guard(graph);
        cg::write_param("n", std::function<std::int64_t()>{[&source] { return source; }});
    }

    graph.execute();
    REQUIRE(graph.params_ptr()->get("n") == 3);

    source = 4;
    graph.execute();
    REQUIRE(graph.params_ptr()->get("n") == 4);

    auto const report = graph.serializability_report();
    REQUIRE(report.size() == 1);
    REQUIRE(report.front().kind_name == "WriteParam");
    REQUIRE_THAT(report.front().reason, Catch::Matchers::ContainsSubstring("callback arm"));

    // It still REBUILDS, because the closure is in the descriptor: only a SAVE
    // is blocked, and the two questions are deliberately separate.
    graph.params_ptr()->set("n", 0);
    for (auto &node : graph.nodes()) {
        node.execute = cg::build_executor(node.kind, packed_gemm::ScalarType::Unknown, 0, node.op_data, graph,
                                          std::span<cg::TensorId const>{node.inputs}, std::span<cg::TensorId const>{node.outputs});
    }
    graph.execute();
    REQUIRE(graph.params_ptr()->get("n") == 4);
}
