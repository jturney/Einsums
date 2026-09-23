//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

/// @file TagsAndProviders.cpp
/// @brief Saying what a tensor IS, registering something that knows how to factor it, and
///        reading what the result is worth.
///
/// Index spaces say how big an axis is and how it grows. They cannot say that a tensor is an
/// electron repulsion integral, or a Kronecker delta, or an energy denominator, and a pass
/// that wants to RECOGNIZE something has to ask that question instead. A provenance tag is the
/// answer: an open vocabulary name plus free-form attributes, declared by the caller and never
/// inferred from a tensor's contents.
///
/// Five sections, on synthetic tensors so this runs anywhere:
///   1. A tag is a declaration. ``DeltaElimination`` acts on one.
///   2. A provider registry. The metric fit claims a tagged tensor and the pass re-associates
///      the contraction around its factors.
///   3. The record the rewrite left on the graph, and what each field is for.
///   4. Composition. Two records on one output do not add unless their units say so.
///   5. The accuracy budget, which is where a pass refuses rather than spends.

#include <Einsums/ComputeGraph.hpp>
#include <Einsums/ComputeGraph/Factorization.hpp>
#include <Einsums/ComputeGraph/MetricFitFactorization.hpp>
#include <Einsums/ComputeGraph/Passes/FactorizationPass.hpp>
#include <Einsums/LinearAlgebra.hpp>
#include <Einsums/Print.hpp>
#include <Einsums/Runtime.hpp>
#include <Einsums/TensorUtilities/CreateIdentity.hpp>
#include <Einsums/TensorUtilities/CreateRandomTensor.hpp>
#include <Einsums/TensorUtilities/CreateZeroTensor.hpp>

#include <cmath>
#include <cstddef>
#include <iostream>
#include <memory>
#include <string>

namespace cg = einsums::compute_graph;

namespace {

std::size_t const nbf  = 6;
std::size_t const naux = 5;

/// A symmetric positive-definite metric, built as R R^T plus a shift so nothing is singular.
einsums::Tensor<double, 2> make_metric() {
    auto seed = einsums::create_random_tensor<double>("seed", naux, naux);
    auto out  = einsums::create_zero_tensor<double>("J", naux, naux);
    for (std::size_t p = 0; p < naux; ++p) {
        for (std::size_t q = 0; q < naux; ++q) {
            double sum = 0.0;
            for (std::size_t r = 0; r < naux; ++r) {
                sum += seed(p, r) * seed(q, r);
            }
            out(p, q) = sum + (p == q ? 2.0 : 0.0);
        }
    }
    return out;
}

/// The four-index tensor a metric fit reproduces exactly: sum_PQ R[P,m,n] Jinv[P,Q] R[Q,p,q].
///
/// Built by inverting the metric directly, so the tensor the fit is compared against owes
/// nothing to the code path being demonstrated.
einsums::Tensor<double, 4> exact_fit(einsums::Tensor<double, 3> const &R, einsums::Tensor<double, 2> const &J) {
    auto inverse = einsums::Tensor<double, 2>("Jinv", naux, naux);
    inverse      = J;
    einsums::linear_algebra::invert(&inverse);

    auto out = einsums::create_zero_tensor<double>("(mn|pq)", nbf, nbf, nbf, nbf);
    for (std::size_t m = 0; m < nbf; ++m) {
        for (std::size_t n = 0; n < nbf; ++n) {
            for (std::size_t p = 0; p < nbf; ++p) {
                for (std::size_t q = 0; q < nbf; ++q) {
                    double sum = 0.0;
                    for (std::size_t a = 0; a < naux; ++a) {
                        for (std::size_t b = 0; b < naux; ++b) {
                            sum += R(a, m, n) * inverse(a, b) * R(b, p, q);
                        }
                    }
                    out(m, n, p, q) = sum;
                }
            }
        }
    }
    return out;
}

/// The largest elementwise difference between two rank-two tensors.
double largest_difference(einsums::Tensor<double, 2> const &left, einsums::Tensor<double, 2> const &right) {
    double worst = 0.0;
    for (std::size_t i = 0; i < nbf; ++i) {
        for (std::size_t j = 0; j < nbf; ++j) {
            worst = std::max(worst, std::abs(left(i, j) - right(i, j)));
        }
    }
    return worst;
}

/// Add @p pass to @p pm without handing over ownership of a stack object.
std::shared_ptr<cg::OptimizerPass> borrow(cg::OptimizerPass &pass) {
    return std::shared_ptr<cg::OptimizerPass>(&pass, [](cg::OptimizerPass *) {});
}

} // namespace

int einsums_main() {
    using namespace einsums;

    // ═══════════════════════════════════════════════════════════════════════
    // 1. A tag is a declaration
    // ═══════════════════════════════════════════════════════════════════════
    println("=== A tag says what a tensor IS ===\n");
    {
        // A contraction against a Kronecker delta is a rename, so DeltaElimination does the
        // rename and drops the delta. What it cannot do is notice that a tensor happens to
        // hold an identity: a structural rewrite is what a saved graph keeps, and a later
        // bind may put a different tensor behind the same name. So the caller declares it.
        auto A     = create_random_tensor<double>("A", 4, 5);
        auto delta = create_identity_tensor<double>("delta", 5, 5);
        auto D     = create_random_tensor<double>("D", 5, 3);
        auto C     = create_zero_tensor<double>("C", 4, 3);

        cg::Graph graph("delta");
        graph.annotate_tag(delta, cg::ProvenanceTag{.name = std::string(cg::provenance_identity)});
        auto &tmp = graph.create_zero_runtime_tensor<double>("tmp", {4, 5}, true);
        {
            cg::CaptureGuard const guard(graph);
            cg::einsum("ik;kj->ij", &tmp, A, delta);
            cg::einsum("ij;jl->il", &C, tmp, D);
        }

        auto const                   before = graph.num_nodes();
        cg::passes::DeltaElimination elimination;
        cg::PassManager              pm;
        pm.add(borrow(elimination));
        pm.run(graph);
        println("  nodes {} -> {}, eliminated {}, intermediates dissolved {}", before, graph.num_nodes(), elimination.num_eliminated(),
                elimination.num_dissolved());
        println("  The rewrite is bitwise exact: a sum with one nonzero term and exact zeros");
        println("  everywhere else owes the same float, not a nearby one.\n");

        // A tag may carry attributes, which are how one vocabulary name qualifies itself.
        // They are sorted on the way in, so two tags built by setting the same keys in a
        // different order compare equal and a saved graph's bytes do not depend on call order.
        auto       eri = create_zero_tensor<double>("(mn|pq)", 2, 2, 2, 2);
        cg::Graph  qualified("qualified");
        auto const tag = cg::ProvenanceTag::make_with_attributes("eri", {{"basis", "cc-pvdz"}, {"kind", "coulomb"}});
        qualified.annotate_tag(eri, tag);
        println("  a qualified tag: name '{}', basis '{}'", tag.name, tag.attribute("basis").value_or("(none)"));
        println("  A tag crosses an operation only where the output holds the same elements as");
        println("  the input, which today is the axis reorderings and nothing else. A view does");
        println("  NOT inherit one: a slice of a delta off the diagonal is an ordinary matrix of");
        println("  ones and zeros, and eliminating a contraction against it is a wrong answer.");
    }

    // ═══════════════════════════════════════════════════════════════════════
    // 2. A provider registry
    // ═══════════════════════════════════════════════════════════════════════
    println("\n=== A provider claims a tag and states a factored form ===\n");

    auto R = create_random_tensor<double>("(Q|mn)", naux, nbf, nbf);
    auto J = make_metric();
    auto T = exact_fit(R, J);
    auto D = create_random_tensor<double>("density", nbf, nbf);

    // The reference: the four-index contraction, run with nothing rewritten.
    auto      reference_result = create_zero_tensor<double>("C", nbf, nbf);
    cg::Graph reference("reference");
    {
        cg::CaptureGuard const guard(reference);
        cg::einsum("m,n,p,q ; p,q -> m,n", &reference_result, T, D);
    }
    reference.execute();

    auto      C = create_zero_tensor<double>("C", nbf, nbf);
    cg::Graph graph("fitted");
    {
        cg::CaptureGuard const guard(graph);
        cg::einsum("m,n,p,q ; p,q -> m,n", &C, T, D);
    }
    graph.annotate_tag(T, cg::ProvenanceTag{.name = "eri"});

    // A provider registers ON A TAG. Nothing in this class knows what an integral is:
    // density fitting is this class registered on "eri", and the chemistry lives in the
    // registration rather than in the library.
    cg::FactorizationRegistry registry;
    // The fourth argument is the bound the provider ASSERTS, which is what the record carries
    // and what a later pass composes against. It is a claim rather than a measurement, for the
    // reason section 3 gives.
    registry.add(std::make_shared<cg::MetricFitFactorization>("eri", R, J, 1e-6));

    cg::passes::FactorizationPass factorization(registry);
    {
        auto const      captured = graph.num_nodes();
        cg::PassManager pm;
        pm.add(borrow(factorization));
        bool const fired = graph.apply(pm);
        println("  fired: {}, factorized {}", fired, factorization.num_factorized());
        println("  nodes {} -> {}", captured, graph.num_nodes());
        for (auto const &[reason, count] : factorization.skip_reasons()) {
            println("  declined ({}): {}", count, reason);
        }
        println("  The decline is the SETUP BODY the rewrite just emitted, which the pass visits");
        println("  like any other graph and which holds no tagged tensor of its own.");
        std::cout << '\n';
        println("  The substitution alone would make the arithmetic WORSE: replacing one tensor");
        println("  by two factors and contracting in the captured order is more work, not less.");
        println("  What pays is the re-association the pass does afterwards, and it is costed");
        println("  twice before anything is emitted: symbolically, which is the claim about the");
        println("  family, and at the extents this graph holds, which is a veto on a rewrite");
        println("  that is slower at the size in front of it.");
    }

    // The fitting is a SETUP body: it runs once per bound problem rather than once per
    // replay, and a graph that is bound again refits before the next execute.
    auto defaults = cg::PassManager::create_default();
    graph.apply(defaults);
    graph.execute();
    println("\n  largest deviation from the unfitted contraction: {:.3e}", largest_difference(C, reference_result));

    // ═══════════════════════════════════════════════════════════════════════
    // 3. The record
    // ═══════════════════════════════════════════════════════════════════════
    println("\n=== What the graph now says about itself ===\n");
    for (auto const &record : graph.approximations()) {
        println("  pass      {}", record.pass_name);
        println("  asked for {:.3e}", record.tolerance);
        println("  bound     {:.3e} ({}, {})", record.bound, cg::approximation_effect_name(record.effect),
                cg::approximation_origin_name(record.origin));
        println("  setup     {}", record.setup.empty() ? std::string("(none)") : record.setup);
        println("  outputs   {}", record.outputs.empty() ? std::string("(every output)") : record.outputs.front());
    }
    std::cout << '\n';
    println("  ASSERTED, not measured, and the distinction is the useful part of the field. The");
    println("  error of a metric fit is its difference from the exact four-index tensor, and a");
    println("  caller holding that tensor had no reason to fit it. A truncated decomposition");
    println("  holding its own discarded singular values genuinely knows, and records MEASURED.");
    std::cout << '\n';
    println("  What CAN be measured here is reported per bind instead: how many auxiliary");
    println("  directions the guarded inverse square root threw away, which is a specific and");
    println("  common way for accuracy to degrade quietly.");

    // ═══════════════════════════════════════════════════════════════════════
    // 4. Composition
    // ═══════════════════════════════════════════════════════════════════════
    println("\n=== Two approximations on one output ===\n");
    {
        // A second record, written by hand here because the passes that would write one need
        // a denominator or a grid. Everything below is what happens whoever wrote them.
        graph.note_approximation(cg::ApproximationRecord{.pass_name = "Illustration",
                                                         .tolerance = 1e-3,
                                                         .effect    = cg::ApproximationEffect::NormRelative,
                                                         .bound     = 1e-3,
                                                         .origin    = cg::ApproximationOrigin::Asserted,
                                                         .outputs   = {"C"}});
        auto const tolerance = graph.approximation_tolerance("C");
        println("  relative {:.12e}, absolute {:.12e}", tolerance.relative, tolerance.absolute);
        println("  1e-6 and 1e-3 composed, and the last digits are the product term.");
        std::cout << '\n';
        println("  Two RELATIVE bounds do not add. The second rewrite's error is relative to the");
        println("  already perturbed result, so the composed bound is e1 + e2 + e1*e2, and");
        println("  dropping the product term makes a budget quietly optimistic, which is the one");
        println("  direction an accuracy contract must never fail in. Two bounds in different");
        println("  units do not combine at all: a norm bound says nothing about the worst");
        println("  element, so they are reported as two numbers and a comparison widens by both.");
    }

    // ═══════════════════════════════════════════════════════════════════════
    // 5. The budget
    // ═══════════════════════════════════════════════════════════════════════
    println("\n=== The budget is where a pass refuses ===\n");
    {
        auto      fresh_result = create_zero_tensor<double>("C", nbf, nbf);
        cg::Graph budgeted("budgeted");
        {
            cg::CaptureGuard const guard(budgeted);
            cg::einsum("m,n,p,q ; p,q -> m,n", &fresh_result, T, D);
        }
        budgeted.annotate_tag(T, cg::ProvenanceTag{.name = "eri"});
        // A budget caps ONE kind of error, and it is a property of the run rather than of the
        // graph, so it is never saved: a loaded graph's caller states their own.
        budgeted.set_accuracy_budget(cg::ApproximationEffect::NormRelative, 1e-16);

        cg::passes::FactorizationPass refused(registry);
        cg::PassManager               pm;
        pm.add(borrow(refused));
        bool const fired = budgeted.apply(pm);
        println("  fired: {}, factorized {}", fired, refused.num_factorized());
        for (auto const &[reason, count] : refused.skip_reasons()) {
            println("  declined ({}): {}", count, reason);
        }
        std::cout << '\n';
        println("  The refusal happens BEFORE anything is rewritten, so a graph is never left");
        println("  half-rewritten with an unrecorded approximation in it.");
    }

    println("\nDone.");
    finalize();
    return EXIT_SUCCESS;
}

int main(int argc, char **argv) {
    return einsums::start(einsums_main, argc, argv);
}
