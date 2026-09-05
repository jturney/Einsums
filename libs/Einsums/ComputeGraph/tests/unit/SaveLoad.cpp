//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

/// @file SaveLoad.cpp
/// @brief The graph IR round trip, its refusals, and its hash.
///
/// The central property is BITWISE identity: capture a program, execute it,
/// snapshot the answer; save; load; bind the same tensors; execute; require the
/// same bits. Values-within-tolerance is deliberately not the bar. A loader that
/// reaches a different kernel - a different summation order, a different BLAS
/// arm because a prefactor's dtype was widened - agrees to every tolerance a
/// value comparison would use while landing on different bits, and that is the
/// class of bug this feature is most able to introduce.
///
/// Four tiers here:
///
///  1. **Round trip per kind family**, so every reconstructible descriptor is
///     exercised end to end rather than only in the writer.
///  2. **Refusals**, one case per unresolvable-name class and per closure arm.
///     A save that quietly dropped a callback, or a load that quietly ignored an
///     unknown op kind, is the failure mode the strict document model exists to
///     prevent, so each one is pinned.
///  3. **Hash invariants**: provenance-independent, structurally sensitive, and
///     renumbering-deterministic (two captures of one program produce
///     byte-identical files).
///  4. **Goldens**, the forever-compatibility promise as a test.

#include <Einsums/ComputeGraph.hpp>
#include <Einsums/ComputeGraph/GraphIR.hpp>
#include <Einsums/Tensor/RuntimeTensor.hpp>
#include <Einsums/Tensor/TiledRuntimeTensor.hpp>
#include <Einsums/TensorUtilities/CreateIdentity.hpp>
#include <Einsums/TensorUtilities/CreateRandomTensor.hpp>
#include <Einsums/TensorUtilities/CreateZeroTensor.hpp>

#include <fmt/format.h>

#include <complex>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

#include <Einsums/Testing.hpp>

using namespace einsums;
namespace cg = einsums::compute_graph;

namespace {

/// A byte-for-byte snapshot of an owning tensor's storage.
template <typename TensorType>
std::vector<unsigned char> bytes_of(TensorType const &t) {
    using T = typename TensorType::ValueType;
    std::vector<unsigned char> out(t.size() * sizeof(T));
    std::memcpy(out.data(), t.data(), out.size());
    return out;
}

template <typename T>
std::vector<unsigned char> bytes_of_scalar(T const &value) {
    std::vector<unsigned char> out(sizeof(T));
    std::memcpy(out.data(), &value, sizeof(T));
    return out;
}

/// Save @p graph, requiring success, and return the document.
std::string must_save(cg::Graph const &graph, cg::SaveOptions const &options = {}) {
    auto text = cg::save_graph_string(graph, options);
    INFO((text ? std::string{} : text.error().message));
    REQUIRE(text.has_value());
    return *text;
}

/// Load @p text, requiring success.
cg::Graph must_load(std::string const &text) {
    auto graph = cg::load_graph_string(text);
    INFO((graph ? std::string{} : graph.error().message));
    REQUIRE(graph.has_value());
    return std::move(*graph);
}

/// The message a refused save produced.
std::string save_refusal(cg::Graph const &graph) {
    auto text = cg::save_graph_string(graph);
    REQUIRE_FALSE(text.has_value());
    return text.error().message;
}

/// The message a refused load produced.
std::string load_refusal(std::string const &text) {
    auto graph = cg::load_graph_string(text);
    REQUIRE_FALSE(graph.has_value());
    return graph.error().message;
}

/// Replace the first occurrence of @p from with @p to, for the seeded-corruption
/// cases. Fails the test when @p from is absent, so a schema change that renames
/// a key breaks the test loudly instead of silently making it assert nothing.
std::string patched(std::string text, std::string_view from, std::string_view to) {
    auto const at = text.find(from);
    INFO("looking for '" << from << "'");
    REQUIRE(at != std::string::npos);
    return text.replace(at, from.size(), to);
}

} // namespace

// ── Tier 1: round trip per kind family ─────────────────────────────────────

TEST_CASE("SaveLoad - dense element-wise kinds round-trip bitwise", "[ComputeGraph][SaveLoad]") {
    auto A = create_random_tensor<double>("A", 4, 5);
    auto B = create_random_tensor<double>("B", 4, 5);
    auto C = create_zero_tensor<double>("C", 4, 5);
    auto D = create_zero_tensor<double>("D", 5, 4);

    auto const A0 = bytes_of(A);
    auto const B0 = bytes_of(B);

    cg::Graph graph("elementwise");
    {
        cg::CaptureGuard const guard(graph);
        cg::scale(0.375, &A);
        cg::axpby(0.25, A, 0.5, &B);
        cg::direct_product(1.5, A, B, 0.0, &C);
        cg::direct_division(2.0, A, B, 0.25, &C);
        cg::permute("ji <- ij", 0.0, &D, 3.25, C);
    }
    graph.execute();
    auto const expected_a = bytes_of(A);
    auto const expected_b = bytes_of(B);
    auto const expected_c = bytes_of(C);
    auto const expected_d = bytes_of(D);

    std::string const text   = must_save(graph);
    cg::Graph         loaded = must_load(text);

    // Fresh storage, seeded exactly as the captured run started.
    auto A2 = create_zero_tensor<double>("A2", 4, 5);
    auto B2 = create_zero_tensor<double>("B2", 4, 5);
    auto C2 = create_zero_tensor<double>("C2", 4, 5);
    auto D2 = create_zero_tensor<double>("D2", 5, 4);
    std::memcpy(A2.data(), A0.data(), A0.size());
    std::memcpy(B2.data(), B0.data(), B0.size());

    loaded.bind("A", A2, "B", B2, "C", C2, "D", D2);
    REQUIRE(loaded.unbound_manifest_entries().empty());
    loaded.execute();

    REQUIRE(bytes_of(A2) == expected_a);
    REQUIRE(bytes_of(B2) == expected_b);
    REQUIRE(bytes_of(C2) == expected_c);
    REQUIRE(bytes_of(D2) == expected_d);
}

TEMPLATE_TEST_CASE("SaveLoad - einsum round-trips bitwise", "[ComputeGraph][SaveLoad]", double, std::complex<double>) {
    using T = TestType;

    auto A = create_random_tensor<T>("A", 6, 4);
    auto B = create_random_tensor<T>("B", 4, 5);
    auto C = create_zero_tensor<T>("C", 6, 5);
    auto E = create_zero_tensor<T>("E", 6, 4, 5);

    auto const A0 = bytes_of(A);
    auto const B0 = bytes_of(B);

    cg::Graph graph("einsum");
    {
        cg::CaptureGuard const guard(graph);
        // A GEMM-shaped contraction, so the node carries a gemm_hint; and an
        // outer product, which does not.
        cg::einsum("ij <- ik ; kj", T{0}, &C, T{1}, A, B);
        cg::einsum("ikj <- ik ; kj", T{0}, &E, T{1}, A, B);
    }
    graph.execute();
    auto const expected_c = bytes_of(C);
    auto const expected_e = bytes_of(E);

    // The GEMM hint is part of what the file carries; check it survived rather
    // than assuming, since nothing else in the round trip would notice.
    std::string const text = must_save(graph);
    REQUIRE_THAT(text, Catch::Matchers::ContainsSubstring("gemm_hint"));
    cg::Graph loaded = must_load(text);

    bool hinted = false;
    for (auto const &node : loaded.nodes()) {
        if (auto const *desc = std::get_if<cg::EinsumDescriptor>(&node.op_data); desc != nullptr && desc->gemm_hint != nullptr) {
            hinted = true;
            REQUIRE(desc->gemm_hint->m == 6);
            REQUIRE(desc->gemm_hint->n == 5);
            REQUIRE(desc->gemm_hint->k == 4);
        }
    }
    REQUIRE(hinted);

    auto A2 = create_zero_tensor<T>("A2", 6, 4);
    auto B2 = create_zero_tensor<T>("B2", 4, 5);
    auto C2 = create_zero_tensor<T>("C2", 6, 5);
    auto E2 = create_zero_tensor<T>("E2", 6, 4, 5);
    std::memcpy(A2.data(), A0.data(), A0.size());
    std::memcpy(B2.data(), B0.data(), B0.size());

    loaded.bind("A", A2, "B", B2, "C", C2, "E", E2);
    loaded.execute();

    REQUIRE(bytes_of(C2) == expected_c);
    REQUIRE(bytes_of(E2) == expected_e);
}

TEST_CASE("SaveLoad - gemm, transpose, dot and trace round-trip bitwise", "[ComputeGraph][SaveLoad]") {
    auto A = create_random_tensor<double>("A", 4, 4);
    auto B = create_random_tensor<double>("B", 4, 4);
    auto C = create_zero_tensor<double>("C", 4, 4);

    auto const A0 = bytes_of(A);
    auto const B0 = bytes_of(B);

    double captured_dot   = 0.0;
    double captured_trace = 0.0;

    cg::Graph graph("blas");
    {
        cg::CaptureGuard const guard(graph);
        cg::gemm<false, false>(1.25, A, B, 0.0, &C);
        cg::dot(&captured_dot, A, B);
        cg::trace(&captured_trace, C);
    }
    graph.execute();
    auto const expected_c     = bytes_of(C);
    auto const expected_dot   = bytes_of_scalar(captured_dot);
    auto const expected_trace = bytes_of_scalar(captured_trace);

    cg::Graph loaded = must_load(must_save(graph));

    auto A2 = create_zero_tensor<double>("A2", 4, 4);
    auto B2 = create_zero_tensor<double>("B2", 4, 4);
    auto C2 = create_zero_tensor<double>("C2", 4, 4);
    std::memcpy(A2.data(), A0.data(), A0.size());
    std::memcpy(B2.data(), B0.data(), B0.size());

    double loaded_dot   = 0.0;
    double loaded_trace = 0.0;

    loaded.bind("A", A2, "B", B2, "C", C2);
    // A raw scalar destination is rank 0 and has no tensor to bind; bind_scalar
    // is what hands it back, and it works because ScalarAccessor reads the
    // handle's pointer on every call rather than baking it at build time.
    loaded.bind_scalar("dot_result", &loaded_dot);
    loaded.bind_scalar("trace_result", &loaded_trace);
    loaded.execute();

    REQUIRE(bytes_of(C2) == expected_c);
    REQUIRE(bytes_of_scalar(loaded_dot) == expected_dot);
    REQUIRE(bytes_of_scalar(loaded_trace) == expected_trace);
}

namespace {

/// A symmetric matrix with well-separated eigenvalues, so LAPACK's ordering of
/// the eigenvectors is not a coin toss between two runs of the same build.
Tensor<double, 2> symmetric_matrix() {
    auto out = create_zero_tensor<double>("A", 4, 4);
    for (size_t i = 0; i < 4; ++i) {
        for (size_t j = i; j < 4; ++j) {
            double const value = 1.0 / (1.0 + static_cast<double>(i + j)) + (i == j ? 3.0 * static_cast<double>(i + 1) : 0.0);
            out(i, j)          = value;
            out(j, i)          = value;
        }
    }
    return out;
}

} // namespace

TEST_CASE("SaveLoad - a symmetric eigendecomposition round-trips bitwise", "[ComputeGraph][SaveLoad]") {
    // A is decomposed IN PLACE, so the node lists it as an input AND an output, and the
    // matrix the loaded graph must be handed is the ORIGINAL rather than the eigenvectors
    // the capture-time execute left in it. The snapshot below is taken before any execute
    // for exactly that reason.
    auto A = symmetric_matrix();
    auto W = create_zero_tensor<double>("W", 4);

    auto const A0 = bytes_of(A);

    cg::Graph graph("eigen");
    {
        cg::CaptureGuard const guard(graph);
        cg::syev(&A, &W);
    }
    graph.execute();
    auto const expected_vectors = bytes_of(A);
    auto const expected_values  = bytes_of(W);

    std::string const text = must_save(graph);
    REQUIRE_THAT(text, Catch::Matchers::ContainsSubstring("\"Syev\""));
    REQUIRE_THAT(text, Catch::Matchers::ContainsSubstring("compute_eigenvectors"));

    cg::Graph loaded = must_load(text);

    auto A2 = create_zero_tensor<double>("A2", 4, 4);
    auto W2 = create_zero_tensor<double>("W2", 4);
    std::memcpy(A2.data(), A0.data(), A0.size());
    loaded.bind("A", A2, "W", W2);
    loaded.execute();

    REQUIRE(bytes_of(A2) == expected_vectors);
    REQUIRE(bytes_of(W2) == expected_values);
}

TEST_CASE("SaveLoad - an eigenvalues-only decomposition stays eigenvalues-only", "[ComputeGraph][SaveLoad]") {
    // ComputeEigenvectors is a TEMPLATE argument at the capture site, so it is the one part
    // of a syev that a file has to carry explicitly. Getting it wrong is not a slower answer:
    // LAPACK's jobz='n' leaves A holding the tridiagonal reduction's scratch, so a loaded
    // graph that guessed 'v' would hand every downstream consumer a different matrix.
    auto A = symmetric_matrix();
    auto W = create_zero_tensor<double>("W", 4);

    auto const A0 = bytes_of(A);

    cg::Graph graph("eigenvalues_only");
    {
        cg::CaptureGuard const guard(graph);
        cg::syev<false>(&A, &W);
    }
    graph.execute();
    auto const expected_values = bytes_of(W);

    cg::Graph loaded = must_load(must_save(graph));

    // The flag itself, read back off the node rather than inferred from the numbers, so the
    // assertion still means something if LAPACK's two jobs ever agree on A for small inputs.
    REQUIRE(loaded.num_nodes() == 1);
    auto const *descriptor = std::get_if<cg::SyevDescriptor>(&loaded.nodes()[0].op_data);
    REQUIRE(descriptor != nullptr);
    REQUIRE_FALSE(descriptor->compute_eigenvectors);

    auto A2 = create_zero_tensor<double>("A2", 4, 4);
    auto W2 = create_zero_tensor<double>("W2", 4);
    std::memcpy(A2.data(), A0.data(), A0.size());
    loaded.bind("A", A2, "W", W2);
    loaded.execute();

    REQUIRE(bytes_of(W2) == expected_values);

    // And the eigenvectors are NOT what came back, which is what makes the flag observable
    // in the numbers as well as in the descriptor.
    auto      V  = symmetric_matrix();
    auto      Wv = create_zero_tensor<double>("Wv", 4);
    cg::Graph vectors("with_vectors");
    {
        cg::CaptureGuard const guard(vectors);
        cg::syev<true>(&V, &Wv);
    }
    vectors.execute();
    REQUIRE(bytes_of(A2) != bytes_of(V));
}

TEST_CASE("SaveLoad - a named element transform round-trips", "[ComputeGraph][SaveLoad]") {
    auto A = create_random_tensor<double>("A", 3, 3);
    for (size_t i = 0; i < A.size(); ++i) {
        A.data()[i] += 2.0; // keep the reciprocal well away from zero
    }
    auto const A0 = bytes_of(A);

    cg::Graph graph("named_kernel");
    {
        cg::CaptureGuard const guard(graph);
        cg::element_transform(&A, "recip");
    }
    graph.execute();
    auto const expected = bytes_of(A);

    std::string const text = must_save(graph);
    REQUIRE_THAT(text, Catch::Matchers::ContainsSubstring("\"recip\""));

    cg::Graph loaded = must_load(text);
    auto      A2     = create_zero_tensor<double>("A2", 3, 3);
    std::memcpy(A2.data(), A0.data(), A0.size());
    loaded.bind("A", A2);
    loaded.execute();
    REQUIRE(bytes_of(A2) == expected);
}

TEST_CASE("SaveLoad - write_param and a parameter table round-trip", "[ComputeGraph][SaveLoad]") {
    auto A = create_random_tensor<double>("A", 2, 2);
    auto B = create_zero_tensor<double>("B", 2, 2);

    cg::Graph graph("params");
    graph.params_ptr()->set("n_occ", 3);
    {
        cg::CaptureGuard const guard(graph);
        cg::write_param("derived", cg::BoundExpr{std::string("n_occ")});
        cg::write_param("literal", cg::BoundExpr{std::int64_t{7}});
        cg::axpby(1.0, A, 0.0, &B);
    }
    graph.execute();

    std::string const text = must_save(graph);
    REQUIRE_THAT(text, Catch::Matchers::ContainsSubstring("\"n_occ\""));

    cg::Graph loaded = must_load(text);
    REQUIRE(loaded.params_ptr()->get("n_occ") == 3);
    auto A2 = create_random_tensor<double>("A2", 2, 2);
    auto B2 = create_zero_tensor<double>("B2", 2, 2);
    loaded.bind("A", A2, "B", B2);
    loaded.execute();
    REQUIRE(loaded.params_ptr()->get("derived") == 3);
    REQUIRE(loaded.params_ptr()->get("literal") == 7);
}

TEST_CASE("SaveLoad - a loop and a conditional round-trip with their bodies", "[ComputeGraph][SaveLoad]") {
    auto A = create_zero_tensor<double>("A", 2, 2);
    auto B = create_zero_tensor<double>("B", 2, 2);
    for (size_t i = 0; i < A.size(); ++i) {
        A.data()[i] = 1.0 + static_cast<double>(i);
    }
    auto const A0 = bytes_of(A);

    cg::GateFlags gates(2, true);
    gates.set(1, false);

    cg::Graph graph("control_flow");
    graph.name_gate_flags("blocks", gates);
    graph.params_ptr()->set("limit", 3);
    {
        // Three iterations: the loop condition is a comparison against the
        // iteration index, which is the data-shaped spelling of a convergence
        // test.
        cg::Graph &body = graph.add_loop("iters", 10, cg::PredExpr::iteration(cg::CmpOp::Lt, cg::BoundExpr{std::string("limit")}));
        {
            cg::CaptureGuard const guard(body);
            cg::axpby(1.0, A, 1.0, &B);
        }
    }
    {
        auto [taken, skipped] = graph.add_conditional_flag("gate0", gates, 0);
        {
            cg::CaptureGuard const guard(taken);
            cg::scale(2.0, &B);
        }
        {
            cg::CaptureGuard const guard(skipped);
            cg::scale(0.0, &B);
        }
    }
    {
        auto [taken, skipped] = graph.add_conditional_flag("gate1", gates, 1);
        {
            cg::CaptureGuard const guard(taken);
            cg::scale(100.0, &B);
        }
        {
            cg::CaptureGuard const guard(skipped);
            cg::scale(0.5, &B);
        }
    }
    graph.execute();
    auto const expected_b = bytes_of(B);

    std::string const text   = must_save(graph);
    cg::Graph         loaded = must_load(text);

    // The gate array comes back by name, sharing the graph's buffer, so writing
    // through it is what the next replay reads.
    cg::GateFlags loaded_gates = loaded.gate_flags("blocks");
    REQUIRE(loaded_gates.size() == 2);
    loaded_gates.set(0, true);
    loaded_gates.set(1, false);
    REQUIRE(loaded.params_ptr()->get("limit") == 3);

    auto A2 = create_zero_tensor<double>("A2", 2, 2);
    auto B2 = create_zero_tensor<double>("B2", 2, 2);
    std::memcpy(A2.data(), A0.data(), A0.size());
    loaded.bind("A", A2, "B", B2);
    loaded.execute();
    INFO(text);
    REQUIRE(bytes_of(B2) == expected_b);
}

TEST_CASE("SaveLoad - spaces, dim symbols and a re-bind at a new size", "[ComputeGraph][SaveLoad]") {
    // register_space is idempotent for an identical declaration and hands back
    // the id, so the ids come from the registration rather than from a lookup
    // that would then have to be checked.
    auto             &registry = cg::global_space_registry();
    cg::SpaceId const occ      = registry.register_space(cg::IndexSpace{.name = "saveload_occ"});
    cg::SpaceId const virt     = registry.register_space(cg::IndexSpace{.name = "saveload_virt"});

    auto F = create_random_tensor<double>("F", 3, 4);
    auto G = create_random_tensor<double>("G", 4, 3);
    auto H = create_zero_tensor<double>("H", 3, 3);

    cg::Graph          graph("symbolic");
    cg::TensorId const f_id = graph.register_operand(F);
    cg::TensorId const g_id = graph.register_operand(G);
    cg::TensorId const h_id = graph.register_operand(H);
    graph.annotate_spaces(f_id, {occ, virt});
    graph.annotate_spaces(g_id, {virt, occ});
    graph.annotate_spaces(h_id, {occ, occ});
    graph.annotate_dims(f_id, {"no", "nv"});
    graph.annotate_dims(g_id, {"nv", "no"});
    graph.annotate_dims(h_id, {"no", "no"});
    {
        cg::CaptureGuard const guard(graph);
        cg::einsum("ij <- ia ; aj", 0.0, &H, 1.0, F, G);
    }
    graph.execute();

    std::string const text = must_save(graph);
    REQUIRE_THAT(text, Catch::Matchers::ContainsSubstring("saveload_occ"));
    REQUIRE_THAT(text, Catch::Matchers::ContainsSubstring("symbol_ties"));

    cg::Graph loaded = must_load(text);
    // Annotations survive: the letters of the contraction carry their spaces,
    // and the slots carry their symbols.
    REQUIRE(loaded.tensor_dim_symbols(loaded.manifest().find("F")->id) == std::vector<std::string>{"no", "nv"});
    REQUIRE(loaded.symbol_spaces().size() == 2);

    // The payoff: a saved graph is valid for a FAMILY of sizes, not for the one
    // geometry it was captured at.
    auto F2 = create_random_tensor<double>("F2", 5, 6);
    auto G2 = create_random_tensor<double>("G2", 6, 5);
    auto H2 = create_zero_tensor<double>("H2", 5, 5);
    loaded.bind("F", F2, "G", G2, "H", H2);
    loaded.execute();

    auto expected = create_zero_tensor<double>("expected", 5, 5);
    einsums::tensor_algebra::einsum(0.0, std::tuple{einsums::index::i, einsums::index::j}, &expected, 1.0,
                                    std::tuple{einsums::index::i, einsums::index::a}, F2, std::tuple{einsums::index::a, einsums::index::j},
                                    G2);
    REQUIRE(bytes_of(H2) == bytes_of(expected));
}

TEST_CASE("SaveLoad - a deferred intermediate comes back deferred and rebinds", "[ComputeGraph][SaveLoad][Symbolic]") {
    // The sibling test above proves a re-bind at a new size for a graph whose every tensor is
    // a manifest entry. Every real graph also has SCRATCH, and that case was broken in a way
    // no test could see: AllocState was not serialized, so a loaded intermediate came back
    // materialized and the extent-changing bind was refused for storage reasons.
    auto             &registry = cg::global_space_registry();
    cg::SpaceId const occ      = registry.register_space(cg::IndexSpace{.name = "sl_occ", .scale_symbol = "o", .dim_symbol = "sl_no"});
    cg::SpaceId const virt     = registry.register_space(cg::IndexSpace{.name = "sl_virt", .scale_symbol = "v", .dim_symbol = "sl_nv"});

    RuntimeTensor<double> amp("amp", std::vector<std::size_t>{4, 6});
    RuntimeTensor<double> out("out", std::vector<std::size_t>{4, 4});
    for (std::size_t i = 0; i < 4; ++i) {
        for (std::size_t a = 0; a < 6; ++a) {
            amp(i, a) = 0.25 * static_cast<double>(i + 1) - 0.125 * static_cast<double>(a);
        }
    }

    cg::Graph graph("scratch_reuse");
    graph.annotate_spaces(amp, {occ, virt});
    graph.annotate_dims(amp, {"sl_no", "sl_nv"});
    graph.annotate_spaces(out, {occ, occ});
    graph.annotate_dims(out, {"sl_no", "sl_no"});
    auto &tmp = graph.declare_zero_runtime_tensor<double>("tmp", {cg::SpaceDim{occ}, cg::SpaceDim{virt}}, true);
    {
        cg::CaptureGuard const guard(graph);
        cg::einsum("ia;ia->ia", &tmp, amp, amp);
        cg::einsum("ia;ja->ij", &out, tmp, amp);
    }

    std::string const text = must_save(graph);
    REQUIRE_THAT(text, Catch::Matchers::ContainsSubstring("\"alloc\""));
    REQUIRE_THAT(text, Catch::Matchers::ContainsSubstring("deferred"));

    cg::Graph loaded = must_load(text);

    // Bind the loaded graph to a DIFFERENT problem and let it re-derive the scratch.
    RuntimeTensor<double> amp2("amp2", std::vector<std::size_t>{3, 5});
    RuntimeTensor<double> out2("out2", std::vector<std::size_t>{3, 3});
    for (std::size_t i = 0; i < 3; ++i) {
        for (std::size_t a = 0; a < 5; ++a) {
            amp2(i, a) = 0.5 * static_cast<double>(i) - 0.0625 * static_cast<double>(a + 2);
        }
    }
    REQUIRE_NOTHROW(loaded.bind("amp", amp2, "out", out2));

    cg::PassManager pm;
    pm.add<cg::passes::Materialization>();
    loaded.apply(pm);
    loaded.execute();

    // The reference: capture the same thing at the new size and compare bitwise, which is
    // what "valid for a family of problems" has to mean.
    RuntimeTensor<double> ref("ref", std::vector<std::size_t>{3, 3});
    cg::Graph             fresh("scratch_reuse_fresh");
    auto                 &fresh_tmp = fresh.declare_zero_runtime_tensor<double>("tmp", std::vector<std::size_t>{3, 5}, true);
    {
        cg::CaptureGuard const guard(fresh);
        cg::einsum("ia;ia->ia", &fresh_tmp, amp2, amp2);
        cg::einsum("ia;ja->ij", &ref, fresh_tmp, amp2);
    }
    cg::PassManager fresh_pm;
    fresh_pm.add<cg::passes::Materialization>();
    fresh.apply(fresh_pm);
    fresh.execute();

    REQUIRE(bytes_of(out2) == bytes_of(ref));
}

// ── Tier 2: refusals ───────────────────────────────────────────────────────

TEST_CASE("SaveLoad - a graph with blockers refuses, carrying the report", "[ComputeGraph][SaveLoad]") {
    auto A = create_random_tensor<double>("A", 3, 3);

    cg::Graph graph("anonymous_kernel");
    {
        cg::CaptureGuard const guard(graph);
        cg::element_transform(&A, [](double x) { return x * x; });
    }
    std::string const message = save_refusal(graph);
    REQUIRE_THAT(message, Catch::Matchers::ContainsSubstring("anonymous closure"));
    REQUIRE_THAT(message, Catch::Matchers::ContainsSubstring("element_ops::register_op"));
}

TEST_CASE("SaveLoad - a callback predicate refuses at save, naming the node", "[ComputeGraph][SaveLoad]") {
    auto A = create_zero_tensor<double>("A", 2, 2);

    cg::Graph graph("callback_predicate");
    {
        cg::Graph             &body = graph.add_loop("spin", 2, [](size_t it) { return it < 1; });
        cg::CaptureGuard const guard(body);
        cg::scale(2.0, &A);
    }
    std::string const message = save_refusal(graph);
    REQUIRE_THAT(message, Catch::Matchers::ContainsSubstring("spin"));
    REQUIRE_THAT(message, Catch::Matchers::ContainsSubstring("callback condition"));
}

TEST_CASE("SaveLoad - an unnamed gate-flag array refuses at save", "[ComputeGraph][SaveLoad]") {
    auto A = create_zero_tensor<double>("A", 2, 2);

    cg::GateFlags const gates(1, true);
    cg::Graph           graph("nameless_flags");
    {
        auto [taken, skipped] = graph.add_conditional_flag("gate", gates, 0);
        cg::CaptureGuard const guard(taken);
        cg::scale(2.0, &A);
    }
    std::string const message = save_refusal(graph);
    REQUIRE_THAT(message, Catch::Matchers::ContainsSubstring("UNNAMED gate-flag array"));
    REQUIRE_THAT(message, Catch::Matchers::ContainsSubstring("name_gate_flags"));

    // And naming it is all it takes.
    graph.name_gate_flags("gate", gates);
    REQUIRE(cg::save_graph_string(graph).has_value());
}

TEST_CASE("SaveLoad - a callback view bound refuses at save, naming the field", "[ComputeGraph][SaveLoad]") {
    auto A = create_zero_tensor<double>("A", 4, 4);

    cg::Graph graph("callback_source");
    {
        cg::CaptureGuard const guard(graph);
        cg::write_param("n", [] { return std::int64_t{3}; });
        cg::scale(2.0, &A);
    }
    std::string const message = save_refusal(graph);
    REQUIRE_THAT(message, Catch::Matchers::ContainsSubstring("callback arm"));
}

TEST_CASE("SaveLoad - an integral scalar operand refuses with the fix named", "[ComputeGraph][SaveLoad]") {
    auto A = create_zero_tensor<double>("A", 2, 2);
    int  n = 5;

    cg::Graph graph("integral_scalar");
    {
        cg::CaptureGuard const guard(graph);
        cg::write_param("n", n);
        cg::scale(2.0, &A);
    }
    std::string const message = save_refusal(graph);
    REQUIRE_THAT(message, Catch::Matchers::ContainsSubstring("no BLAS element type"));
    REQUIRE_THAT(message, Catch::Matchers::ContainsSubstring("BoundExpr"));
}

TEST_CASE("SaveLoad - every unresolvable name refuses at load, naming the string", "[ComputeGraph][SaveLoad]") {
    auto A = create_random_tensor<double>("A", 3, 3);
    auto B = create_zero_tensor<double>("B", 3, 3);

    cg::Graph graph("names");
    {
        cg::CaptureGuard const guard(graph);
        cg::axpby(1.0, A, 0.0, &B);
        cg::element_transform(&B, "recip");
    }
    std::string const text = must_save(graph, cg::SaveOptions{.pretty = false});

    SECTION("op kind") {
        auto const message = load_refusal(patched(text, R"("kind":"Axpby")", R"("kind":"Frobnicate")"));
        REQUIRE_THAT(message, Catch::Matchers::ContainsSubstring("'Frobnicate' is not a known op kind"));
    }
    SECTION("element op, with the registered set listed") {
        auto const message = load_refusal(patched(text, R"("op":"recip")", R"("op":"no_such_kernel")"));
        REQUIRE_THAT(message, Catch::Matchers::ContainsSubstring("no_such_kernel"));
        REQUIRE_THAT(message, Catch::Matchers::ContainsSubstring("Registered:"));
        REQUIRE_THAT(message, Catch::Matchers::ContainsSubstring("recip"));
    }
    SECTION("dtype") {
        auto const message = load_refusal(patched(text, R"("dtype":"float64")", R"("dtype":"float128")"));
        REQUIRE_THAT(message, Catch::Matchers::ContainsSubstring("'float128' is not a known dtype"));
    }
    SECTION("manifest direction") {
        auto const message = load_refusal(patched(text, R"("direction":"input")", R"("direction":"sideways")"));
        REQUIRE_THAT(message, Catch::Matchers::ContainsSubstring("'sideways' is not a known manifest direction"));
    }
    SECTION("ownership scope") {
        auto const message = load_refusal(patched(text, R"("scope":"graph")", R"("scope":"galaxy")"));
        REQUIRE_THAT(message, Catch::Matchers::ContainsSubstring("'galaxy' is not a known ownership scope"));
    }
    SECTION("index space") {
        auto const message = load_refusal(patched(text, R"("names":[])", R"("names":["no_such_space"])"));
        REQUIRE_THAT(message, Catch::Matchers::ContainsSubstring("no_such_space"));
        REQUIRE_THAT(message, Catch::Matchers::ContainsSubstring("Registered:"));
    }
    SECTION("an unconsumed key") {
        auto const message = load_refusal(patched(text, R"("name":"names")", R"("surprise":1,"name":"names")"));
        REQUIRE_THAT(message, Catch::Matchers::ContainsSubstring("$.surprise"));
        REQUIRE_THAT(message, Catch::Matchers::ContainsSubstring("does not understand"));
    }
}

TEST_CASE("SaveLoad - a comparison operator and a parameter source type resolve by name", "[ComputeGraph][SaveLoad]") {
    auto A = create_zero_tensor<double>("A", 2, 2);

    cg::Graph graph("named_enums");
    graph.params_ptr()->set("limit", 1);
    {
        cg::Graph &body = graph.add_loop("iters", 4, cg::PredExpr::iteration(cg::CmpOp::Lt, cg::BoundExpr{std::string("limit")}));
        cg::CaptureGuard const guard(body);
        cg::scale(2.0, &A);
    }
    std::string const text = must_save(graph, cg::SaveOptions{.pretty = false});

    SECTION("comparison operator") {
        auto const message = load_refusal(patched(text, R"("op":"lt")", R"("op":"approximately")"));
        REQUIRE_THAT(message, Catch::Matchers::ContainsSubstring("'approximately' is not a known comparison operator"));
    }
}

TEST_CASE("SaveLoad - the version gate refuses a newer file, naming both versions", "[ComputeGraph][SaveLoad]") {
    auto A = create_zero_tensor<double>("A", 2, 2);

    cg::Graph graph("versioned");
    {
        cg::CaptureGuard const guard(graph);
        cg::scale(2.0, &A);
    }
    std::string const text = must_save(graph, cg::SaveOptions{.pretty = false});

    // Patched against the version this build WRITES rather than against a literal, which is
    // a version number in a test that a schema bump has to remember to update. It did not.
    std::string const current = fmt::format(R"("einsums_graph_ir":"{}")", cg::graph_ir_schema_version);

    SECTION("a newer schema is refused with both versions named") {
        auto const message = load_refusal(patched(text, current, R"("einsums_graph_ir":"9.0.0")"));
        REQUIRE_THAT(message, Catch::Matchers::ContainsSubstring("9.0.0"));
        REQUIRE_THAT(message, Catch::Matchers::ContainsSubstring(std::string{cg::graph_ir_schema_version}));
        REQUIRE_THAT(message, Catch::Matchers::ContainsSubstring("never the reverse"));
    }
    SECTION("a malformed version is refused before anything else is read") {
        auto const message = load_refusal(patched(text, current, R"("einsums_graph_ir":"one")"));
        REQUIRE_THAT(message, Catch::Matchers::ContainsSubstring("major.minor.patch"));
    }
}

TEST_CASE("SaveLoad - validate-only reports every seeded problem, not the first", "[ComputeGraph][SaveLoad]") {
    auto A = create_random_tensor<double>("A", 3, 3);
    auto B = create_zero_tensor<double>("B", 3, 3);

    cg::Graph graph("multi_error");
    {
        cg::CaptureGuard const guard(graph);
        cg::axpby(1.0, A, 0.0, &B);
        cg::element_transform(&B, "recip");
        cg::scale(2.0, &B);
    }
    std::string text = must_save(graph, cg::SaveOptions{.pretty = false});

    // Three independent problems, in three different sections.
    text = patched(text, R"("kind":"Axpby")", R"("kind":"Frobnicate")");
    text = patched(text, R"("op":"recip")", R"("op":"no_such_kernel")");
    text = patched(text, R"("scope":"graph")", R"("scope":"galaxy")");

    auto const result = cg::validate_graph_ir_string(text);
    REQUIRE_FALSE(result.has_value());
    INFO(result.error().message);
    // ALL THREE are reported, which is the property; the count is deliberately
    // not pinned, because one seeded problem legitimately cascades into more
    // (an unknown op kind leaves its descriptor unreadable, and that
    // descriptor's keys then read as content this build does not understand).
    // Reporting the consequences alongside the cause is what a tool wants; what
    // it must never do is stop at the first.
    REQUIRE_THAT(result.error().message, Catch::Matchers::ContainsSubstring("problem(s)"));
    REQUIRE_THAT(result.error().message, Catch::Matchers::ContainsSubstring("Frobnicate"));
    REQUIRE_THAT(result.error().message, Catch::Matchers::ContainsSubstring("no_such_kernel"));
    REQUIRE_THAT(result.error().message, Catch::Matchers::ContainsSubstring("galaxy"));

    // A clean file validates.
    REQUIRE(cg::validate_graph_ir_string(must_save(graph, cg::SaveOptions{.pretty = false})).has_value());
}

// ── Tier 3: content hash and renumbering ───────────────────────────────────

TEST_CASE("SaveLoad - content_hash covers structure and nothing else", "[ComputeGraph][SaveLoad]") {
    auto const build = [](std::string const &name, double factor) {
        auto      A = create_zero_tensor<double>("A", 3, 3);
        cg::Graph graph(name);
        {
            cg::CaptureGuard const guard(graph);
            cg::scale(factor, &A);
        }
        return std::pair{graph.content_hash(), must_save(graph)};
    };

    SECTION("provenance does not enter the hash") {
        auto      A = create_zero_tensor<double>("A", 3, 3);
        cg::Graph graph("hashed");
        {
            cg::CaptureGuard const guard(graph);
            cg::scale(2.0, &A);
        }
        std::uint64_t const bare = graph.content_hash();

        cg::SaveOptions options;
        options.structural_passes     = {"CSE", "ScaleAbsorption"};
        std::string const with_passes = must_save(graph, options);
        std::string const without     = must_save(graph);
        REQUIRE(with_passes != without);       // the files differ
        REQUIRE(graph.content_hash() == bare); // the hash does not
    }

    SECTION("a structural change moves the hash") {
        auto const [hash_a, text_a] = build("same", 2.0);
        auto const [hash_b, text_b] = build("same", 2.5);
        REQUIRE(hash_a != hash_b);
    }

    SECTION("the graph name is structure") {
        auto const [hash_a, text_a] = build("one", 2.0);
        auto const [hash_b, text_b] = build("two", 2.0);
        REQUIRE(hash_a != hash_b);
    }
}

TEST_CASE("SaveLoad - two captures of one program produce byte-identical files", "[ComputeGraph][SaveLoad]") {
    auto const capture = []() {
        // Fresh tensors each time, so the graph's own TensorIds and every
        // address differ between the two runs. Only the renumbering makes the
        // files agree.
        auto      A = create_zero_tensor<double>("A", 3, 4);
        auto      B = create_zero_tensor<double>("B", 4, 2);
        auto      C = create_zero_tensor<double>("C", 3, 2);
        cg::Graph graph("determinism");
        {
            cg::CaptureGuard const guard(graph);
            cg::einsum("ij <- ik ; kj", 0.0, &C, 1.0, A, B);
            cg::scale(0.5, &C);
        }
        return std::pair{must_save(graph), graph.content_hash()};
    };

    auto const [first, first_hash]   = capture();
    auto const [second, second_hash] = capture();
    REQUIRE(first == second);
    REQUIRE(first_hash == second_hash);

    // And a save of a LOADED graph reproduces the same bytes, which is the
    // fixpoint that makes the format an interchange format rather than an
    // export.
    cg::Graph const loaded = must_load(first);
    REQUIRE(must_save(loaded) == first);
    REQUIRE(loaded.content_hash() == first_hash);
}

TEST_CASE("SaveLoad - content_hash refuses a graph that cannot be written", "[ComputeGraph][SaveLoad]") {
    auto A = create_zero_tensor<double>("A", 2, 2);

    cg::Graph graph("unhashable");
    {
        cg::CaptureGuard const guard(graph);
        cg::element_transform(&A, [](double x) { return x + 1.0; });
    }
    REQUIRE_THROWS_WITH(graph.content_hash(), Catch::Matchers::ContainsSubstring("anonymous closure"));
}

TEST_CASE("SaveLoad - an element transform's parameter travels with the node", "[ComputeGraph][SaveLoad][ElementOps]") {
    // A threshold that stayed in the capturing process would make a saved graph mean
    // something different everywhere else: the same file would drop different directions
    // depending on what the loading process happened to think the policy was. So the number
    // is in the node, and this is the test that says so.
    auto values     = create_zero_tensor<double>("values", 3);
    values(0)       = 4.0;
    values(1)       = 1.0e-17;
    values(2)       = -1.0;
    auto const seed = bytes_of(values);

    cg::Graph graph("element_param");
    {
        cg::CaptureGuard const guard(graph);
        cg::element_transform(&values, "inv_sqrt_or_zero", 1.0e-10);
    }
    graph.execute();
    auto const expected = bytes_of(values);

    std::string const text = must_save(graph);
    REQUIRE_THAT(text, Catch::Matchers::ContainsSubstring(R"("param":1e-10)"));

    cg::Graph loaded = must_load(text);
    REQUIRE(loaded.num_nodes() == 1);
    auto const *desc = std::get_if<cg::ElementTransformDescriptor>(&loaded.nodes()[0].op_data);
    REQUIRE(desc != nullptr);
    REQUIRE(desc->param.has_value());
    REQUIRE(*desc->param == 1.0e-10);

    std::memcpy(values.data(), seed.data(), seed.size());
    loaded.bind("values", values);
    loaded.execute();
    REQUIRE(bytes_of(values) == expected);
}

TEST_CASE("SaveLoad - an element transform with no parameter writes no key and runs the op's default",
          "[ComputeGraph][SaveLoad][ElementOps]") {
    // The other half of the compatibility promise: a node that named no number writes no key,
    // which is exactly what every file older than 1.3.0 looks like, and it has to keep
    // computing what it computed. For this op that default is the bare x>0 guard, so the
    // eigenvalue at 1e-17 survives as an enormous reciprocal square root rather than being
    // quietly dropped by whatever threshold the loading build now prefers.
    auto values = create_zero_tensor<double>("values", 2);
    values(0)   = 4.0;
    values(1)   = 1.0e-17;

    cg::Graph graph("element_no_param");
    {
        cg::CaptureGuard const guard(graph);
        cg::element_transform(&values, "inv_sqrt_or_zero");
    }

    std::string const text = must_save(graph);
    // The key, not the word: the document has a ``params`` section, and a substring match on
    // "param" would pass for the wrong reason.
    REQUIRE_THAT(text, !Catch::Matchers::ContainsSubstring(R"("param":)"));

    cg::Graph   loaded = must_load(text);
    auto const *desc   = std::get_if<cg::ElementTransformDescriptor>(&loaded.nodes()[0].op_data);
    REQUIRE(desc != nullptr);
    REQUIRE_FALSE(desc->param.has_value());

    loaded.bind("values", values);
    loaded.execute();
    REQUIRE_THAT(values(0), Catch::Matchers::WithinAbs(0.5, 1.0e-12));
    REQUIRE(values(1) > 1.0e8);
}

TEST_CASE("SaveLoad - an approximation record's measured-parameter key round-trips", "[ComputeGraph][SaveLoad]") {
    // A record is written once, at optimize time, so its bound is what the structure claims. What
    // a bind can leave behind is a parameter, and the record names which one; that is what lets a
    // caller holding a record read what a particular bind's fit was worth, and it is the whole of
    // the rule for a fit re-fitted inside a loop, where the parameter's value when the solver
    // stops is the last iteration's.
    auto A = create_random_tensor<double>("A", 4, 3);
    auto B = create_random_tensor<double>("B", 3, 5);
    auto C = create_zero_tensor<double>("C", 4, 5);

    cg::Graph graph("measured_record");
    {
        cg::CaptureGuard const guard(graph);
        cg::einsum("i,k ; k,j -> i,j", &C, A, B);
    }
    graph.note_approximation(cg::make_approximation_record("Thc", cg::ApproximationEffect::NormRelative, 1.0e-4, 1.0e-4, {}, {}, "Thc(T2)",
                                                           cg::ApproximationOrigin::Asserted, "Thc.T2.residual_squared"));

    auto const saved = cg::save_graph_string(graph);
    REQUIRE(saved.has_value());
    if (std::getenv("EINSUMS_WRITE_GOLDEN") != nullptr) {
        std::ofstream out(std::filesystem::path{EINSUMS_GRAPH_IR_GOLDEN_DIR} / "v1_5_0_measured_record.eig.json", std::ios::binary);
        out << *saved;
    }

    cg::Graph const back = must_load(*saved);
    REQUIRE(back.approximations().size() == 1);
    CHECK(back.approximations().front().measurement == "Thc.T2.residual_squared");
    CHECK(back.approximations().front().setup == "Thc(T2)");
}

// ── Tier 4: goldens ────────────────────────────────────────────────────────

TEST_CASE("SaveLoad - every checked-in golden still loads", "[ComputeGraph][SaveLoad]") {
    // THE NEVER-BREAK RULE. Every file under tests/unit/goldens/ was written by
    // some earlier build of this library, and every future build must still read
    // it. A golden is never edited and never regenerated: if a change makes one
    // fail, the change is what is wrong, not the file. A schema addition gets a
    // NEW golden beside the old ones, so the corpus records the whole history
    // rather than only the present.
    std::filesystem::path const directory{EINSUMS_GRAPH_IR_GOLDEN_DIR};
    REQUIRE(std::filesystem::is_directory(directory));

    size_t loaded_count = 0;
    for (auto const &entry : std::filesystem::directory_iterator(directory)) {
        if (!entry.is_regular_file() || entry.path().extension() != ".json") {
            continue;
        }
        INFO("golden: " << entry.path().string());
        std::ifstream     file(entry.path(), std::ios::binary);
        std::string const text{std::istreambuf_iterator<char>(file), std::istreambuf_iterator<char>()};
        REQUIRE_FALSE(text.empty());

        auto const validated = cg::validate_graph_ir_string(text);
        INFO((validated ? std::string{} : validated.error().message));
        REQUIRE(validated.has_value());

        cg::Graph const graph = must_load(text);
        REQUIRE(graph.num_nodes() > 0);
        ++loaded_count;
    }
    REQUIRE(loaded_count > 0);
}

TEST_CASE("SaveLoad - a tiled eigendecomposition refuses although its kind is reconstructible", "[ComputeGraph][SaveLoad][TiledRuntime]") {
    // Syev is per KIND, and the kind covers two operations. The dense one is reconstructible;
    // the tiled one diagonalizes each diagonal block of a grid of buffers, has no builder
    // entry, and records no descriptor at all. So the per-node verdict is the only one that
    // can tell them apart, and the message has to say that the absent descriptor is what a
    // tiled node looks like rather than a field the writer dropped.
    using Grid = std::vector<std::vector<int>>;
    TiledRuntimeTensor<double> A("A", Grid{{2, 3}, {2, 3}});
    TiledRuntimeTensor<double> W("W", Grid{{2, 3}});

    A.add_tile({0, 0});
    A.add_tile({1, 1});
    A.materialize();

    cg::Graph graph("tiled_eigen");
    {
        cg::CaptureGuard const guard(graph);
        cg::syev(&A, &W);
    }

    auto const report = graph.serializability_report();
    REQUIRE(report.size() == 1);
    CHECK(report.front().kind_name == "Syev");
    CHECK_THAT(report.front().reason, Catch::Matchers::ContainsSubstring("tiled variant records none"));

    CHECK_THAT(save_refusal(graph), Catch::Matchers::ContainsSubstring("Syev"));
}

TEST_CASE("SaveLoad - a batched form refuses to save and says why", "[ComputeGraph][SaveLoad][Batched]") {
    // "Not yet reconstructible" reads as an oversight. This one is a decision, and the
    // refusal has to carry it: a grouped batch partitions its members by SHAPE at capture
    // and reorders the operand lists to match, so its descriptor is a function of one
    // problem's extents. Saving it would freeze that problem into the file, which is what
    // the structure/tuning split forbids. Making it saveable means capturing the
    // algebraic form and grouping on load, not teaching the descriptor to serialize itself.
    auto a1 = create_random_tensor<double>("a1", 3, 4);
    auto b1 = create_random_tensor<double>("b1", 4, 5);
    auto c1 = create_zero_tensor<double>("c1", 3, 5);
    auto a2 = create_random_tensor<double>("a2", 2, 6);
    auto b2 = create_random_tensor<double>("b2", 6, 7);
    auto c2 = create_zero_tensor<double>("c2", 2, 7);

    cg::Graph graph("grouped_refusal");
    {
        cg::CaptureGuard const                       guard(graph);
        std::vector<Tensor<double, 2> const *> const as{&a1, &a2};
        std::vector<Tensor<double, 2> const *> const bs{&b1, &b2};
        std::vector<Tensor<double, 2> *> const       cs{&c1, &c2};
        cg::grouped_batched_gemm(1.0, as, bs, 0.0, cs);
    }

    auto const report = graph.serializability_report();
    REQUIRE(report.size() == 1);
    CHECK(report.front().kind_name == "GroupedBatchedGemm");
    CHECK_THAT(report.front().reason, Catch::Matchers::ContainsSubstring("resource decision"));
    CHECK_THAT(report.front().reason, Catch::Matchers::ContainsSubstring("let the resource phase group them"));

    auto const saved = cg::save_graph_string(graph);
    REQUIRE_FALSE(saved.has_value());
    CHECK_THAT(saved.error().message, Catch::Matchers::ContainsSubstring("resource decision"));
}

TEST_CASE("SaveLoad - a load resolves spaces against a registry the caller owns", "[ComputeGraph][SaveLoad][Spaces]") {
    // A saved graph carries space NAMES, because a SpaceId means nothing outside the registry
    // that issued it. Resolving them needs a registry, and load_graph used the process-global
    // one unconditionally: a caller keeping their own registry got told the space "is not
    // registered in this process" with an empty list of what IS, having registered everything.
    cg::SpaceRegistry mine;
    cg::SpaceId const occ  = mine.register_space(cg::IndexSpace{.name = "private_occ", .scale_symbol = "o", .dim_symbol = "p_no"});
    cg::SpaceId const virt = mine.register_space(cg::IndexSpace{.name = "private_virt", .scale_symbol = "v", .dim_symbol = "p_nv"});

    auto F = create_random_tensor<double>("F", 3, 4);
    auto G = create_random_tensor<double>("G", 4, 3);
    auto H = create_zero_tensor<double>("H", 3, 3);

    cg::Graph graph("private_registry");
    graph.set_space_registry(mine);
    graph.annotate_spaces(F, {occ, virt});
    graph.annotate_dims(F, {"p_no", "p_nv"});
    graph.annotate_spaces(G, {virt, occ});
    graph.annotate_dims(G, {"p_nv", "p_no"});
    graph.annotate_spaces(H, {occ, occ});
    graph.annotate_dims(H, {"p_no", "p_no"});
    {
        cg::CaptureGuard const guard(graph);
        cg::einsum("ia;aj->ij", &H, F, G);
    }

    std::string const text = must_save(graph);

    // The global registry has never heard of these names, so the default overload must fail
    // and say so. That is the behaviour a caller was stuck with.
    auto const against_global = cg::load_graph_string(text);
    REQUIRE_FALSE(against_global.has_value());
    CHECK_THAT(against_global.error().message, Catch::Matchers::ContainsSubstring("private_occ"));

    // Handed the right registry, the same document loads.
    auto const against_mine = cg::load_graph_string(text, mine);
    REQUIRE(against_mine.has_value());

    // And the loaded graph USES it, so the ids read back are the ones it was built from
    // rather than whatever sits at those indices in the global registry.
    cg::Graph &loaded = const_cast<cg::Graph &>(*against_mine);
    CHECK(&loaded.space_registry() == &mine);
    // The manifest is held in a local on purpose: `manifest()` returns BY VALUE, so
    // `loaded.manifest().find("F")` leaves the pointer dangling once the temporary dies at
    // the semicolon. Reading it inside the same full-expression is safe; storing it is not.
    auto const  contract = loaded.manifest();
    auto const *entry    = contract.find("F");
    REQUIRE(entry != nullptr);
    CHECK(loaded.tensor_spaces(entry->id) == std::vector<cg::SpaceId>{occ, virt});

    // validate_graph_ir_string takes one too, for the same reason.
    CHECK_FALSE(cg::validate_graph_ir_string(text).has_value());
    CHECK(cg::validate_graph_ir_string(text, mine).has_value());
}

TEST_CASE("a save records the structural passes that shaped the graph", "[ComputeGraph][SaveLoad][Provenance]") {
    // The provenance block used to take its pass list from whoever called save_graph, which
    // meant it was accurate exactly as often as someone remembered to fill it in - and the
    // block exists to answer "what shaped this file" for a graph whose numbers turn out wrong,
    // which is the moment nobody has that to hand.
    auto A     = create_random_tensor<double>("A", 4, 5);
    auto delta = create_identity_tensor<double>("delta", 5, 5);
    auto C     = create_zero_tensor<double>("C", 4, 5);

    cg::Graph graph("recorded");
    graph.annotate_tag(delta, cg::ProvenanceTag{.name = std::string(cg::provenance_identity)});
    {
        cg::CaptureGuard const guard(graph);
        cg::einsum("ik;kj->ij", &C, A, delta);
    }
    CHECK(graph.structural_passes().empty()); // nothing has run yet

    cg::PassManager pm;
    pm.add(std::make_shared<cg::passes::DeltaElimination>());
    REQUIRE(graph.apply(pm));

    REQUIRE(graph.structural_passes().size() == 1);
    CHECK(graph.structural_passes()[0] == "DeltaElimination");

    // Applying again records nothing new: the list says what shaped the graph, not how many
    // times a caller ran a manager.
    cg::PassManager again;
    again.add(std::make_shared<cg::passes::DeltaElimination>());
    graph.apply(again);
    CHECK(graph.structural_passes().size() == 1);

    auto const text = cg::save_graph_string(graph);
    REQUIRE(text.has_value());
    CHECK(text->find("DeltaElimination") != std::string::npos);

    // And it survives the round trip, so a load-optimize-resave does not silently drop the
    // history of everything that shaped the file the first time.
    auto loaded = cg::load_graph_string(*text);
    REQUIRE(loaded.has_value());
    REQUIRE(loaded->structural_passes().size() == 1);
    CHECK(loaded->structural_passes()[0] == "DeltaElimination");
}

TEST_CASE("a caller-supplied pass list still wins", "[ComputeGraph][SaveLoad][Provenance]") {
    // A tool assembling a file from pieces knows things the graph does not, so an explicit list
    // overrides the recorded one rather than being merged with it: a merge would produce a
    // history that is neither what the tool said nor what happened.
    auto A = create_random_tensor<double>("A", 4, 3);
    auto B = create_random_tensor<double>("B", 3, 5);
    auto C = create_zero_tensor<double>("C", 4, 5);

    cg::Graph graph("supplied");
    {
        cg::CaptureGuard const guard(graph);
        cg::einsum("ik;kj->ij", &C, A, B);
    }

    cg::SaveOptions options;
    options.structural_passes = {"SomeOfflineOptimizer"};
    auto const text           = cg::save_graph_string(graph, options);
    REQUIRE(text.has_value());
    CHECK(text->find("SomeOfflineOptimizer") != std::string::npos);
}
