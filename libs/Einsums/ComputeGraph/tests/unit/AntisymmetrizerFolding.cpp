//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#include <Einsums/ComputeGraph.hpp>
#include <Einsums/ComputeGraph/Passes/AntisymmetrizerFolding.hpp>
#include <Einsums/ComputeGraph/Passes/AntisymmetryDetection.hpp>
#include <Einsums/ComputeGraph/Passes/AntisymmetryInference.hpp>
#include <Einsums/Tensor/RuntimeTensor.hpp>
#include <Einsums/Tensor/SymmetryOps.hpp>
#include <Einsums/Tensor/Tensor.hpp>
#include <Einsums/TensorUtilities/CreateRandomTensor.hpp>

#include <memory>
#include <stdexcept>
#include <vector>

#include <Einsums/Testing.hpp>

using namespace einsums;
namespace cg = einsums::compute_graph;

namespace {

/// Capture `result = dot( P[spec](wsrc), P[spec](vsrc) )`, which is the shape
/// the toy's naive (T) energy has: an antisymmetrized quantity contracted
/// against another antisymmetrized quantity.
void capture_energy(cg::Graph &graph, std::string const &spec, std::vector<size_t> const &dims, RuntimeTensor<double> const &wsrc,
                    RuntimeTensor<double> const &vsrc, RuntimeTensor<double> &result) {
    auto                  &W = graph.create_zero_runtime_tensor<double>("W", dims, true);
    auto                  &V = graph.create_zero_runtime_tensor<double>("V", dims, true);
    cg::CaptureGuard const capture(graph);
    cg::permute(cg::PermuteFormatString(spec), 0.0, &W, 1.0, wsrc);
    cg::permute(cg::PermuteFormatString(spec), 0.0, &V, 1.0, vsrc);
    cg::dot_python(&result, W, V);
}

double run(cg::Graph &graph, RuntimeTensor<double> &result) {
    graph.execute();
    return result(std::vector<size_t>{0});
}

} // namespace

// THE test. The fold discards N-1 of the operator's terms and multiplies by N,
// which is only the same number if the identity holds. Comparing a folded run
// against an unfolded one on the same data is the whole claim.
TEST_CASE("AntisymmetrizerFolding - the folded contraction gives the same value", "[ComputeGraph][AntisymmetrizerFolding]") {
    size_t const              n = 4;
    std::vector<size_t> const dims{n, n, n};
    auto                      w_typed = create_random_tensor<double>("wsrc", n, n, n);
    auto                      v_typed = create_random_tensor<double>("vsrc", n, n, n);
    RuntimeTensor<double>     wsrc(w_typed);
    RuntimeTensor<double>     vsrc(v_typed);

    std::string const spec = "i,j,k <- P(i/j/k) i,j,k";

    RuntimeTensor<double> plain_result("plain", {1});
    cg::Graph             plain("unfolded");
    capture_energy(plain, spec, dims, wsrc, vsrc, plain_result);
    double const unfolded = run(plain, plain_result);

    RuntimeTensor<double> folded_result("folded", {1});
    cg::Graph             folded("folded");
    capture_energy(folded, spec, dims, wsrc, vsrc, folded_result);

    auto            inference = std::make_shared<cg::passes::AntisymmetryInference>();
    auto            fold      = std::make_shared<cg::passes::AntisymmetrizerFolding>();
    cg::PassManager manager;
    manager.add(inference);
    manager.add(fold);
    folded.apply(manager);

    REQUIRE(inference->num_tagged() == 2); // both antisymmetrizer outputs
    REQUIRE(fold->num_folded() == 1);

    double const value = run(folded, folded_result);
    INFO("unfolded=" << unfolded << " folded=" << value);
    // ReAssociating, not bitwise: the discarded terms were equal in exact
    // arithmetic and merely close in floating point.
    REQUIRE_THAT(value, Catch::Matchers::WithinRel(unfolded, 1e-12));
    REQUIRE(std::abs(unfolded) > 1e-6); // the comparison is not two zeros
}

// The coset operator the triples correction uses. Its output is antisymmetric
// under nothing, so AntisymmetryInference tags neither operand and the fold has
// no premise. A pass that fired here would be wrong, not merely optimistic.
TEST_CASE("AntisymmetrizerFolding - a coset operator is not folded", "[ComputeGraph][AntisymmetrizerFolding]") {
    size_t const              n = 3;
    std::vector<size_t> const dims{n, n, n};
    auto                      w_typed = create_random_tensor<double>("wsrc", n, n, n);
    auto                      v_typed = create_random_tensor<double>("vsrc", n, n, n);
    RuntimeTensor<double>     wsrc(w_typed);
    RuntimeTensor<double>     vsrc(v_typed);

    RuntimeTensor<double> result("r", {1});
    cg::Graph             graph("coset");
    capture_energy(graph, "i,j,k <- P(i/jk) i,j,k", dims, wsrc, vsrc, result);

    auto            inference = std::make_shared<cg::passes::AntisymmetryInference>();
    auto            fold      = std::make_shared<cg::passes::AntisymmetrizerFolding>();
    cg::PassManager manager;
    manager.add(inference);
    manager.add(fold);
    graph.apply(manager);

    CHECK(inference->num_tagged() == 0);
    CHECK(fold->num_folded() == 0);
}

// Without the inference pass there is no hint, so the fold must decline even
// though the operator is one it could otherwise collapse. The premise has to be
// established, not assumed from the operator's presence.
TEST_CASE("AntisymmetrizerFolding - declines without an established premise", "[ComputeGraph][AntisymmetrizerFolding]") {
    size_t const              n = 3;
    std::vector<size_t> const dims{n, n, n};
    auto                      w_typed = create_random_tensor<double>("wsrc", n, n, n);
    auto                      v_typed = create_random_tensor<double>("vsrc", n, n, n);
    RuntimeTensor<double>     wsrc(w_typed);
    RuntimeTensor<double>     vsrc(v_typed);

    RuntimeTensor<double> result("r", {1});
    cg::Graph             graph("no_inference");
    capture_energy(graph, "i,j,k <- P(i/j/k) i,j,k", dims, wsrc, vsrc, result);

    auto            fold = std::make_shared<cg::passes::AntisymmetrizerFolding>();
    cg::PassManager manager;
    manager.add(fold);
    graph.apply(manager);

    CHECK(fold->num_candidates() >= 1);
    CHECK(fold->num_folded() == 0);
    CHECK(fold->explain().empty());
}

TEST_CASE("AntisymmetrizerFolding - a graph with no operator is untouched", "[ComputeGraph][AntisymmetrizerFolding]") {
    auto                  a_typed = create_random_tensor<double>("a", 6);
    auto                  b_typed = create_random_tensor<double>("b", 6);
    RuntimeTensor<double> A(a_typed);
    RuntimeTensor<double> B(b_typed);
    RuntimeTensor<double> result("r", {1});

    cg::Graph graph("plain");
    {
        cg::CaptureGuard const capture(graph);
        cg::dot_python(&result, A, B);
    }
    std::size_t const before = graph.num_nodes();

    auto            fold = std::make_shared<cg::passes::AntisymmetrizerFolding>();
    cg::PassManager manager;
    manager.add(fold);
    graph.apply(manager);

    CHECK(fold->num_candidates() == 0);
    CHECK(fold->num_folded() == 0);
    CHECK(graph.num_nodes() == before);
}

// THE chain this whole line of work exists for, in miniature. It is the shape of
// the toy's (T) energy: an antisymmetrized quantity, divided by an invariant
// denominator, contracted against another antisymmetrized quantity. Nothing here
// is structural. Every link rests on a fact read out of the data:
//
//   detection : wsrc is antisymmetric within P(i/jk)'s group, D is invariant
//   R1 (cond) : so W = P(i/jk)(wsrc) is fully antisymmetric
//   R2        : so Wd = W / D keeps it
//   fold      : so dot(Wd, P(i/jk)(vsrc)) collapses to 3 * dot(Wd, vsrc)
TEST_CASE("AntisymmetrizerFolding - the detected chain reaches a fold", "[ComputeGraph][AntisymmetrizerFolding]") {
    size_t const n = 4;

    // Antisymmetric in axes (1,2), the group P(i/jk) does not permute within.
    //
    // Built by antisymmetrizing RANDOM data rather than from a closed form. The
    // first attempt used (j-k)*(1+i), which is antisymmetric in (j,k) and which
    // P(i/jk) annihilates exactly: expanding
    // f(ijk) - f(jik) - f(kji) cancels term by term and W came out identically
    // zero, so the test compared two zeros and would have passed with the fold
    // computing anything at all.
    auto const r_w = create_random_tensor<double>("rw", n, n, n);
    auto const r_v = create_random_tensor<double>("rv", n, n, n);

    RuntimeTensor<double> wsrc("wsrc", {n, n, n});
    RuntimeTensor<double> vsrc("vsrc", {n, n, n});
    RuntimeTensor<double> den("den", {n, n, n});
    for (size_t i = 0; i < n; ++i) {
        for (size_t j = 0; j < n; ++j) {
            for (size_t k = 0; k < n; ++k) {
                wsrc(std::vector<size_t>{i, j, k}) = r_w(i, j, k) - r_w(i, k, j);
                vsrc(std::vector<size_t>{i, j, k}) = r_v(i, j, k) - r_v(i, k, j);
                // Invariant under every permutation of the three axes, and never
                // zero, which is what an energy denominator looks like.
                den(std::vector<size_t>{i, j, k}) = 2.0 + static_cast<double>(i) + static_cast<double>(j) + static_cast<double>(k);
            }
        }
    }

    auto const capture_chain = [&](cg::Graph &graph, RuntimeTensor<double> &result) {
        auto                  &W  = graph.create_zero_runtime_tensor<double>("W", {n, n, n}, true);
        auto                  &Wd = graph.create_zero_runtime_tensor<double>("Wd", {n, n, n}, true);
        auto                  &V  = graph.create_zero_runtime_tensor<double>("V", {n, n, n}, true);
        cg::CaptureGuard const capture(graph);
        cg::permute("i,j,k <- P(i/jk) i,j,k", 0.0, &W, 1.0, wsrc);
        cg::direct_division(1.0, W, den, 0.0, &Wd);
        cg::permute("i,j,k <- P(i/jk) i,j,k", 0.0, &V, 1.0, vsrc);
        cg::dot_python(&result, Wd, V);
    };

    RuntimeTensor<double> plain_result("plain", {1});
    cg::Graph             plain("chain_unfolded");
    capture_chain(plain, plain_result);
    plain.execute();
    double const unfolded = plain_result(std::vector<size_t>{0});

    RuntimeTensor<double> folded_result("folded", {1});
    cg::Graph             folded("chain_folded");
    capture_chain(folded, folded_result);

    auto            detection = std::make_shared<cg::passes::AntisymmetryDetection>();
    auto            inference = std::make_shared<cg::passes::AntisymmetryInference>();
    auto            fold      = std::make_shared<cg::passes::AntisymmetrizerFolding>();
    cg::PassManager manager;
    manager.add(detection);
    manager.add(inference);
    manager.add(fold);
    folded.apply(manager);

    INFO("detected " << detection->num_found() << " generator(s), tagged " << inference->num_tagged() << ", folded " << fold->num_folded());
    CHECK(detection->num_found() > 0);
    CHECK(inference->num_tagged() > 0);
    REQUIRE(fold->num_folded() == 1);

    folded.execute();
    double const value = folded_result(std::vector<size_t>{0});
    INFO("unfolded=" << unfolded << " folded=" << value);
    REQUIRE(std::abs(unfolded) > 1e-6);
    REQUIRE_THAT(value, Catch::Matchers::WithinRel(unfolded, 1e-12));
}

// THE guard. A fold justified by reading the bound tensors is only valid for
// those tensors, and rebind() repoints a graph at new ones without re-running
// the pipeline. Without this the graph would go on computing the folded form
// against data that does not satisfy the identity, and the failure would be a
// converged wrong number rather than an error.
TEST_CASE("AntisymmetrizerFolding - a rebind past a detected fold is refused", "[ComputeGraph][AntisymmetrizerFolding]") {
    size_t const n = 4;

    auto const            r = create_random_tensor<double>("r", n, n, n);
    RuntimeTensor<double> wsrc("wsrc", {n, n, n});
    RuntimeTensor<double> vsrc("vsrc", {n, n, n});
    for (size_t i = 0; i < n; ++i) {
        for (size_t j = 0; j < n; ++j) {
            for (size_t k = 0; k < n; ++k) {
                wsrc(std::vector<size_t>{i, j, k}) = r(i, j, k) - r(i, k, j);
                vsrc(std::vector<size_t>{i, j, k}) = r(k, j, i) - r(j, k, i);
            }
        }
    }

    RuntimeTensor<double> result("res", {1});
    cg::Graph             graph("guarded");
    {
        auto                  &W = graph.create_zero_runtime_tensor<double>("W", {n, n, n}, true);
        auto                  &V = graph.create_zero_runtime_tensor<double>("V", {n, n, n}, true);
        cg::CaptureGuard const capture(graph);
        cg::permute("i,j,k <- P(i/jk) i,j,k", 0.0, &W, 1.0, wsrc);
        cg::permute("i,j,k <- P(i/jk) i,j,k", 0.0, &V, 1.0, vsrc);
        cg::dot_python(&result, W, V);
    }

    auto            detection = std::make_shared<cg::passes::AntisymmetryDetection>();
    auto            inference = std::make_shared<cg::passes::AntisymmetryInference>();
    auto            fold      = std::make_shared<cg::passes::AntisymmetrizerFolding>();
    cg::PassManager manager;
    manager.add(detection);
    manager.add(inference);
    manager.add(fold);
    graph.apply(manager);
    REQUIRE(fold->num_folded() == 1);

    // On the data the fold was justified by, everything is fine and the guard
    // costs one sweep on the first execute and a boolean thereafter.
    REQUIRE_NOTHROW(graph.execute());
    REQUIRE_NOTHROW(graph.execute());

    // Rebind the antisymmetric source to data with no symmetry at all. The
    // rewrite is now invalid, and the graph has to say so rather than compute.
    auto const            junk_typed = create_random_tensor<double>("junk", n, n, n);
    RuntimeTensor<double> junk(junk_typed);
    REQUIRE_FALSE(check_symmetry(junk, SymmetryDescriptor::antisymmetric_pair(1, 2)));

    graph.rebind(wsrc, junk);
    REQUIRE_THROWS_AS(graph.execute(), std::runtime_error);
}

// The toy's (T) chain in miniature, rooted where a residual's antisymmetry
// actually comes from: the AMPLITUDES, reached through a contraction.
//
//   detection : t2 is antisymmetric in the slots carrying (j,k)
//   R3        : so Xc = t2 . g is antisymmetric in the output's (j,k)
//   R1 (cond) : so W = P(i/jk)(Xc) is fully antisymmetric
//   R2        : so Wd = W / D keeps it
//   fold      : so dot(Wd, P(i/jk)(Xd)) collapses to 3 * dot(Wd, Xd)
//
// Every link rests on one fact read out of one tensor. Nothing is declared and
// nothing is structural.
TEST_CASE("AntisymmetrizerFolding - a chain rooted at the amplitudes", "[ComputeGraph][AntisymmetrizerFolding]") {
    size_t const n = 4;

    // t2(j,k,m) antisymmetric in (j,k), built by projecting random data so the
    // operator does not annihilate it the way a closed form would.
    auto const            r = create_random_tensor<double>("r", n, n, n);
    RuntimeTensor<double> t2("t2", {n, n, n});
    for (size_t j = 0; j < n; ++j) {
        for (size_t k = 0; k < n; ++k) {
            for (size_t m = 0; m < n; ++m) {
                t2(std::vector<size_t>{j, k, m}) = r(j, k, m) - r(k, j, m);
            }
        }
    }
    REQUIRE(check_symmetry(t2, SymmetryDescriptor::antisymmetric_pair(0, 1)));

    auto const            g_typed = create_random_tensor<double>("g", n, n);
    RuntimeTensor<double> g(g_typed); // g(m,i): carries neither j nor k
    auto const            x_typed = create_random_tensor<double>("x", n, n, n);
    RuntimeTensor<double> xd(x_typed);

    RuntimeTensor<double> den("den", {n, n, n});
    for (size_t i = 0; i < n; ++i) {
        for (size_t j = 0; j < n; ++j) {
            for (size_t k = 0; k < n; ++k) {
                den(std::vector<size_t>{i, j, k}) = 2.0 + static_cast<double>(i) + static_cast<double>(j) + static_cast<double>(k);
            }
        }
    }

    auto const capture_chain = [&](cg::Graph &graph, RuntimeTensor<double> &result) {
        auto                  &Xc = graph.create_zero_runtime_tensor<double>("Xc", {n, n, n}, true);
        auto                  &W  = graph.create_zero_runtime_tensor<double>("W", {n, n, n}, true);
        auto                  &Wd = graph.create_zero_runtime_tensor<double>("Wd", {n, n, n}, true);
        auto                  &V  = graph.create_zero_runtime_tensor<double>("V", {n, n, n}, true);
        cg::CaptureGuard const capture(graph);
        // Xc(i,j,k) = sum_m t2(j,k,m) g(m,i). The carrier holds j and k; the
        // other operand holds neither, which is what R3 requires.
        cg::einsum("i,j,k <- j,k,m ; m,i", 0.0, &Xc, 1.0, t2, g);
        cg::permute("i,j,k <- P(i/jk) i,j,k", 0.0, &W, 1.0, Xc);
        cg::direct_division(1.0, W, den, 0.0, &Wd);
        cg::permute("i,j,k <- P(i/jk) i,j,k", 0.0, &V, 1.0, xd);
        cg::dot_python(&result, Wd, V);
    };

    RuntimeTensor<double> plain_result("plain", {1});
    cg::Graph             plain("amplitude_chain_unfolded");
    capture_chain(plain, plain_result);
    plain.execute();
    double const unfolded = plain_result(std::vector<size_t>{0});

    RuntimeTensor<double> folded_result("folded", {1});
    cg::Graph             folded("amplitude_chain_folded");
    capture_chain(folded, folded_result);

    auto            detection = std::make_shared<cg::passes::AntisymmetryDetection>();
    auto            inference = std::make_shared<cg::passes::AntisymmetryInference>();
    auto            fold      = std::make_shared<cg::passes::AntisymmetrizerFolding>();
    cg::PassManager manager;
    manager.add(detection);
    manager.add(inference);
    manager.add(fold);
    folded.apply(manager);

    INFO("detected " << detection->num_found() << ", tagged " << inference->num_tagged() << ", folded " << fold->num_folded());
    CHECK(detection->num_found() > 0);
    REQUIRE(fold->num_folded() == 1);

    folded.execute();
    double const value = folded_result(std::vector<size_t>{0});
    INFO("unfolded=" << unfolded << " folded=" << value);
    REQUIRE(std::abs(unfolded) > 1e-6);
    REQUIRE_THAT(value, Catch::Matchers::WithinRel(unfolded, 1e-12));
}

// The guard's BOUNDARY, pinned in both directions so it is a documented
// limitation rather than a surprise. A bind clears the Setup body's computed
// flag; an in-place overwrite of a validated input does not, and the graph cannot
// see one - the tensor belongs to the caller, no node writes it, and its address
// is unchanged. invalidate_setup is the way across.
TEST_CASE("AntisymmetrizerFolding - the guard runs per bind, and invalidate_setup forces it", "[ComputeGraph][AntisymmetrizerFolding]") {
    size_t const n = 4;
    auto const   r = create_random_tensor<double>("r", n, n, n);

    RuntimeTensor<double> wsrc("wsrc", {n, n, n});
    RuntimeTensor<double> vsrc("vsrc", {n, n, n});
    for (size_t i = 0; i < n; ++i) {
        for (size_t j = 0; j < n; ++j) {
            for (size_t k = 0; k < n; ++k) {
                wsrc(std::vector<size_t>{i, j, k}) = r(i, j, k) - r(i, k, j);
                vsrc(std::vector<size_t>{i, j, k}) = r(k, j, i) - r(j, k, i);
            }
        }
    }

    RuntimeTensor<double> result("res", {1});
    cg::Graph             graph("boundary");
    {
        auto                  &W = graph.create_zero_runtime_tensor<double>("W", {n, n, n}, true);
        auto                  &V = graph.create_zero_runtime_tensor<double>("V", {n, n, n}, true);
        cg::CaptureGuard const capture(graph);
        cg::permute("i,j,k <- P(i/jk) i,j,k", 0.0, &W, 1.0, wsrc);
        cg::permute("i,j,k <- P(i/jk) i,j,k", 0.0, &V, 1.0, vsrc);
        cg::dot_python(&result, W, V);
    }

    auto            detection = std::make_shared<cg::passes::AntisymmetryDetection>();
    auto            inference = std::make_shared<cg::passes::AntisymmetryInference>();
    auto            fold      = std::make_shared<cg::passes::AntisymmetrizerFolding>();
    cg::PassManager manager;
    manager.add(detection);
    manager.add(inference);
    manager.add(fold);
    graph.apply(manager);
    REQUIRE(fold->num_folded() == 1);
    REQUIRE_NOTHROW(graph.execute());

    // Destroy the symmetry the fold rests on, IN PLACE, without rebinding.
    wsrc(std::vector<size_t>{0, 1, 2}) += 17.0;
    REQUIRE_FALSE(check_symmetry(wsrc, SymmetryDescriptor::antisymmetric_pair(1, 2)));

    // The graph cannot see that, and says so by continuing. This is the
    // documented boundary, asserted so a change to it is deliberate.
    REQUIRE_NOTHROW(graph.execute());

    // The caller's way across: put the guard back to work.
    graph.invalidate_setup();
    REQUIRE_THROWS_AS(graph.execute(), std::runtime_error);
}
