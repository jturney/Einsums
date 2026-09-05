//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#include <Einsums/ComputeGraph.hpp>
#include <Einsums/ComputeGraph/GraphIR.hpp>
#include <Einsums/ComputeGraph/LaplaceQuadrature.hpp>
#include <Einsums/ComputeGraph/Passes/LaplaceTransform.hpp>
#include <Einsums/TensorUtilities/CreateRandomTensor.hpp>
#include <Einsums/TensorUtilities/CreateZeroTensor.hpp>

#include <cmath>
#include <memory>
#include <string>
#include <vector>

#include <Einsums/Testing.hpp>

using namespace einsums;
namespace cg = einsums::compute_graph;

namespace {

std::size_t const nocc = 4;
std::size_t const nvir = 5;
std::size_t const nlnk = 3;

/// Occupied energies, well below the virtual ones, so the gap never approaches zero.
template <typename T>
Tensor<T, 1> occupied(std::string name = "eps_o") {
    Tensor<T, 1> out(std::move(name), nocc);
    for (std::size_t i = 0; i < nocc; ++i) {
        out(i) = static_cast<T>(-0.9 + 0.15 * static_cast<double>(i));
    }
    return out;
}

/// Virtual energies. The smallest gap is about 0.35 and the largest about 2.9, which is a
/// spectral range ratio of roughly eight, close to what a small molecule actually has.
template <typename T>
Tensor<T, 1> virtuals(std::string name = "eps_v") {
    Tensor<T, 1> out(std::move(name), nvir);
    for (std::size_t i = 0; i < nvir; ++i) {
        out(i) = static_cast<T>(0.05 + 0.5 * static_cast<double>(i));
    }
    return out;
}

/// The reciprocal denominator the caller tags: 1 / (eps_v[a] - eps_o[i]).
template <typename T>
Tensor<T, 2> denominator(Tensor<T, 1> const &eo, Tensor<T, 1> const &ev, std::string name = "denom") {
    Tensor<T, 2> out(std::move(name), nocc, nvir);
    for (std::size_t i = 0; i < nocc; ++i) {
        for (std::size_t a = 0; a < nvir; ++a) {
            out(i, a) = static_cast<T>(1) / (ev(a) - eo(i));
        }
    }
    return out;
}

/// The program both arms run: P = (A B) * D, with the numerator a graph-owned intermediate.
///
/// Returned as a lambda rather than written twice, because the whole point of the comparison
/// is that the two arms capture the SAME program and differ only in which passes run.
template <typename T>
void capture_program(cg::Graph &graph, Tensor<T, 2> const &A, Tensor<T, 2> const &B, Tensor<T, 2> const &D, Tensor<T, 2> *P) {
    auto &numerator = graph.scratch<T, 2>("numerator", nocc, nvir);
    {
        cg::CaptureGuard const guard(graph);
        cg::einsum("i,k ; k,a -> i,a", &numerator, A, B);
        cg::direct_product(T{1}, numerator, D, T{0}, P);
    }
}

/// The same vector as a runtime-rank tensor, which is what ``cg::outer_sum`` needs: it asks its
/// operands for a ``rank()`` and a compile-time-rank tensor answers with a constant instead.
template <typename T>
RuntimeTensor<T> as_runtime(Tensor<T, 1> const &vector) {
    RuntimeTensor<T> out(vector.name(), std::vector<std::size_t>{vector.dim(0)});
    for (std::size_t i = 0; i < vector.dim(0); ++i) {
        out(std::array<std::size_t, 1>{i}) = vector(i);
    }
    return out;
}

/// Give the pass the two energy vectors every case here names in its tags.
template <typename T>
void supply_energies(cg::passes::LaplaceTransform &laplace, Tensor<T, 1> const &eo, Tensor<T, 1> const &ev) {
    laplace.add_energy("eps_o", eo);
    laplace.add_energy("eps_v", ev);
}

/// Add the pass to a manager without handing it ownership, so the caller can read its counters.
std::shared_ptr<cg::OptimizerPass> borrow(cg::OptimizerPass &pass) {
    return {&pass, [](cg::OptimizerPass *) {}};
}

/// Run the default pipeline. A named local because ``apply`` takes a reference and the
/// factory returns by value.
void apply_defaults(cg::Graph &graph) {
    auto defaults = cg::PassManager::create_default();
    graph.apply(defaults);
}

std::string must_save(cg::Graph const &graph) {
    auto text = cg::save_graph_string(graph);
    INFO((text ? std::string{} : text.error().message));
    REQUIRE(text.has_value());
    return *text;
}

cg::Graph must_load(std::string const &text) {
    auto graph = cg::load_graph_string(text);
    INFO((graph ? std::string{} : graph.error().message));
    REQUIRE(graph.has_value());
    return std::move(*graph);
}

} // namespace

// ── The rule itself ─────────────────────────────────────────────────────────

TEST_CASE("LaplaceQuadrature - the measured error is inside the tolerance it was built for", "[ComputeGraph][Laplace]") {
    // Across four decades of tolerance and two spectral ranges, because the bound's three
    // terms are traded off against each other and a rule that met its target on one range
    // by accident would not meet it on another.
    for (double epsilon : {1.0e-3, 1.0e-4, 1.0e-6, 1.0e-8}) {
        for (auto const [low, high] : {std::pair{0.5, 5.0}, std::pair{0.1, 100.0}}) {
            cg::laplace::SpectralRange const range{.low = low, .high = high};
            std::size_t const                count = cg::laplace::quadrature_point_count(range, epsilon);
            INFO("epsilon " << epsilon << " over [" << low << ", " << high << "] takes " << count << " points");
            REQUIRE(count >= 2);
            REQUIRE(cg::laplace::measure_quadrature_error(cg::laplace::build_quadrature(range, epsilon, count)) <= epsilon);
        }
    }
}

TEST_CASE("LaplaceQuadrature - tightening the tolerance grows the point count and shrinks the error", "[ComputeGraph][Laplace]") {
    cg::laplace::SpectralRange const range{.low = 0.35, .high = 2.9};

    std::size_t const loose = cg::laplace::quadrature_point_count(range, 1.0e-3);
    std::size_t const tight = cg::laplace::quadrature_point_count(range, 1.0e-7);
    REQUIRE(tight > loose);

    double const loose_error = cg::laplace::measure_quadrature_error(cg::laplace::build_quadrature(range, 1.0e-3, loose));
    double const tight_error = cg::laplace::measure_quadrature_error(cg::laplace::build_quadrature(range, 1.0e-7, tight));
    REQUIRE(tight_error < loose_error);
}

TEST_CASE("LaplaceQuadrature - a negative denominator is represented, and one straddling zero is refused", "[ComputeGraph][Laplace]") {
    // A denominator of orbital energies is negative as often as it is positive, depending only
    // on which way round the caller wrote the difference; refusing one of the two would be an
    // arbitrary convention to trip over.
    cg::laplace::SpectralRange const negative{.low = -3.0, .high = -0.4};
    auto const grid = cg::laplace::build_quadrature(negative, 1.0e-6, cg::laplace::quadrature_point_count(negative, 1.0e-6));
    REQUIRE(grid.negated);
    REQUIRE(cg::laplace::measure_quadrature_error(grid) <= 1.0e-6);

    // A range containing zero has an unbounded reciprocal and no quadrature at all.
    REQUIRE_THROWS(cg::laplace::quadrature_point_count(cg::laplace::SpectralRange{.low = -1.0, .high = 1.0}, 1.0e-6));
    REQUIRE_THROWS(cg::laplace::quadrature_point_count(cg::laplace::SpectralRange{.low = 0.0, .high = 1.0}, 1.0e-6));
}

TEST_CASE("LaplaceQuadrature - the spectral range of a signed sum is the sum of the extremes", "[ComputeGraph][Laplace]") {
    auto const range = cg::laplace::spectral_range({-0.9, 0.05}, {-0.45, 2.05}, {-1, 1});
    REQUIRE(range.low == Catch::Approx(0.05 + 0.45));
    REQUIRE(range.high == Catch::Approx(2.05 + 0.9));
}

// ── The rewrite ─────────────────────────────────────────────────────────────

TEMPLATE_TEST_CASE("LaplaceTransform - the quadrature reproduces the exact denominator", "[ComputeGraph][Laplace]", float, double) {
    using T = TestType;
    // Float carries about seven digits, so a tolerance below its own rounding would measure
    // the dtype rather than the rule.
    double const tolerance = std::is_same_v<T, float> ? 1.0e-4 : 1.0e-8;

    auto const eo = occupied<T>();
    auto const ev = virtuals<T>();
    auto const D  = denominator<T>(eo, ev);
    auto const A  = create_random_tensor<T>("A", nocc, nlnk);
    auto const B  = create_random_tensor<T>("B", nlnk, nvir);

    auto      exact = create_zero_tensor<T>("P", nocc, nvir);
    cg::Graph reference("reference");
    capture_program<T>(reference, A, B, D, &exact);
    apply_defaults(reference);
    reference.execute();

    auto      transformed = create_zero_tensor<T>("P", nocc, nvir);
    cg::Graph graph("laplace");
    capture_program<T>(graph, A, B, D, &transformed);
    graph.annotate_tag(D, cg::passes::LaplaceTransform::denominator_tag({"eps_o", "eps_v"}, "-+"));

    cg::passes::LaplaceTransform laplace;
    supply_energies(laplace, eo, ev);
    laplace.set_epsilon(tolerance);
    cg::PassManager pm;
    pm.add(borrow(laplace));
    REQUIRE(graph.apply(pm));
    REQUIRE(laplace.num_transformed() == 1);
    REQUIRE(laplace.last_point_count() >= 2);

    apply_defaults(graph);
    graph.execute();

    // The record, and then the oracle held to it. The bound is what the pass MEASURED, so a
    // comparison at that bound is the record being asked to earn its place rather than a
    // tolerance chosen to make the test pass.
    REQUIRE(graph.approximations().size() == 1);
    auto const &record = graph.approximations()[0];
    REQUIRE(record.pass_name == "LaplaceTransform");
    REQUIRE(record.origin == cg::ApproximationOrigin::Measured);
    REQUIRE(record.effect == cg::ApproximationEffect::NormRelative);
    REQUIRE(record.tolerance == Catch::Approx(tolerance));
    REQUIRE(record.bound <= tolerance);

    double worst = 0.0;
    double scale = 0.0;
    for (std::size_t i = 0; i < nocc; ++i) {
        for (std::size_t a = 0; a < nvir; ++a) {
            worst = std::max(worst, static_cast<double>(std::abs(transformed(i, a) - exact(i, a))));
            scale = std::max(scale, static_cast<double>(std::abs(exact(i, a))));
        }
    }
    INFO("worst " << worst << " against a recorded bound of " << record.bound << " on a scale of " << scale);
    // The recorded bound plus the dtype's own rounding, which the record does not claim to
    // cover and which a float comparison is dominated by.
    double const rounding = static_cast<double>(std::numeric_limits<T>::epsilon()) * 64.0;
    REQUIRE(worst <= (record.bound + rounding) * scale);
}

TEST_CASE("LaplaceTransform - a tighter tolerance costs more points and buys a smaller error", "[ComputeGraph][Laplace]") {
    auto const eo = occupied<double>();
    auto const ev = virtuals<double>();
    auto const D  = denominator<double>(eo, ev);
    auto const A  = create_random_tensor<double>("A", nocc, nlnk);
    auto const B  = create_random_tensor<double>("B", nlnk, nvir);

    auto const run = [&](double epsilon) {
        auto      out = create_zero_tensor<double>("P", nocc, nvir);
        cg::Graph graph("laplace");
        capture_program<double>(graph, A, B, D, &out);
        graph.annotate_tag(D, cg::passes::LaplaceTransform::denominator_tag({"eps_o", "eps_v"}, "-+"));
        cg::passes::LaplaceTransform laplace;
        supply_energies(laplace, eo, ev);
        laplace.set_epsilon(epsilon);
        cg::PassManager pm;
        pm.add(borrow(laplace));
        REQUIRE(graph.apply(pm));
        return std::pair{laplace.last_point_count(), laplace.last_measured_error()};
    };

    auto const [loose_points, loose_error] = run(1.0e-3);
    auto const [tight_points, tight_error] = run(1.0e-9);
    INFO("1e-3 took " << loose_points << " points at " << loose_error << "; 1e-9 took " << tight_points << " at " << tight_error);
    REQUIRE(tight_points > loose_points);
    REQUIRE(tight_error < loose_error);
    REQUIRE(loose_error <= 1.0e-3);
    REQUIRE(tight_error <= 1.0e-9);
}

TEST_CASE("LaplaceTransform - an explicit point count overrides the derivation", "[ComputeGraph][Laplace]") {
    auto const eo = occupied<double>();
    auto const ev = virtuals<double>();
    auto const D  = denominator<double>(eo, ev);
    auto const A  = create_random_tensor<double>("A", nocc, nlnk);
    auto const B  = create_random_tensor<double>("B", nlnk, nvir);
    auto       P  = create_zero_tensor<double>("P", nocc, nvir);

    cg::Graph graph("laplace");
    capture_program<double>(graph, A, B, D, &P);
    graph.annotate_tag(D, cg::passes::LaplaceTransform::denominator_tag({"eps_o", "eps_v"}, "-+"));

    cg::passes::LaplaceTransform laplace;
    supply_energies(laplace, eo, ev);
    laplace.set_epsilon(1.0e-6);
    laplace.set_points(7);
    cg::PassManager pm;
    pm.add(borrow(laplace));
    REQUIRE(graph.apply(pm));
    REQUIRE(laplace.last_point_count() == 7);
    // Fewer points than the tolerance asks for stretches the step, and the record says so
    // rather than repeating the target: what the graph carries is what the rule achieves.
    REQUIRE(laplace.last_measured_error() > 1.0e-6);
    REQUIRE(graph.approximations()[0].bound == Catch::Approx(laplace.last_measured_error()));

    REQUIRE_THROWS(laplace.set_points(1));
    REQUIRE_THROWS(laplace.set_points(-3));
    REQUIRE_THROWS(laplace.set_epsilon(0.0));
    REQUIRE_THROWS(laplace.set_epsilon(2.0));
}

TEST_CASE("LaplaceTransform - a deliberately wrong weight fails the oracle", "[ComputeGraph][Laplace]") {
    // The rule is only evidence if a wrong rule would be caught, and the quadrature's weights
    // are the half of it a typo would be invisible in: the points still span the interval and
    // the answer is still smooth and plausible.
    cg::laplace::SpectralRange const range{.low = 0.35, .high = 2.9};
    auto grid = cg::laplace::build_quadrature(range, 1.0e-8, cg::laplace::quadrature_point_count(range, 1.0e-8));
    REQUIRE(cg::laplace::measure_quadrature_error(grid) <= 1.0e-8);

    grid.weights[grid.weights.size() / 2] *= 1.05;
    REQUIRE(cg::laplace::measure_quadrature_error(grid) > 1.0e-8);
}

// ── The refusals ────────────────────────────────────────────────────────────

namespace {

/// Run the pass over a program the caller sets up, and report what it did and what it said.
struct Outcome {
    bool                     modified{false};
    std::size_t              transformed{0};
    std::vector<std::string> skips;
};

template <typename T>
Outcome run_pass(cg::Graph &graph, Tensor<T, 1> const &eo, Tensor<T, 1> const &ev) {
    cg::passes::LaplaceTransform laplace;
    supply_energies(laplace, eo, ev);
    cg::PassManager pm;
    pm.add(borrow(laplace));
    Outcome out;
    out.modified    = graph.apply(pm);
    out.transformed = laplace.num_transformed();
    for (auto const &[reason, count] : laplace.skip_reasons()) {
        (void)count;
        out.skips.push_back(reason);
    }
    return out;
}

bool mentions(std::vector<std::string> const &lines, std::string_view fragment) {
    return std::ranges::any_of(lines, [fragment](std::string const &line) { return line.find(fragment) != std::string::npos; });
}

} // namespace

TEST_CASE("LaplaceTransform - a sliced denominator carries no tag and is reported, not guessed", "[ComputeGraph][Laplace]") {
    // Provenance does not cross a view, and a slice of a reciprocal is not the reciprocal of a
    // slice of anything this pass could name, so the tag simply does not reach the operand.
    auto const eo = occupied<double>();
    auto const ev = virtuals<double>();
    auto       D  = denominator<double>(eo, ev);
    auto const A  = create_random_tensor<double>("A", nocc, nlnk);
    auto const B  = create_random_tensor<double>("B", nlnk, nvir - 1);
    auto       P  = create_zero_tensor<double>("P", nocc, nvir - 1);

    cg::Graph graph("sliced");
    auto     &numerator = graph.scratch<double, 2>("numerator", nocc, nvir - 1);
    {
        cg::CaptureGuard const guard(graph);
        auto                  &slice = cg::view(D, cg::ViewAxis::full(), cg::ViewAxis::range(0, static_cast<std::int64_t>(nvir - 1)));
        cg::einsum("i,k ; k,a -> i,a", &numerator, A, B);
        cg::direct_product(1.0, numerator, slice, 0.0, &P);
    }
    graph.annotate_tag(D, cg::passes::LaplaceTransform::denominator_tag({"eps_o", "eps_v"}, "-+"));

    Outcome const outcome = run_pass(graph, eo, ev);
    REQUIRE(outcome.transformed == 0);
    REQUIRE(mentions(outcome.skips, "has no direct-product consumer"));
}

TEST_CASE("LaplaceTransform - a folded-axis denominator is declined with the reason", "[ComputeGraph][Laplace]") {
    // The pair-driven MP2 form folds two occupied energies into a scalar prefactor, so its
    // denominator is a two-axis object naming four energies. Carrying the folded pair as an
    // attribute would bake a bound value into structure.
    auto const eo = occupied<double>();
    auto const ev = virtuals<double>();
    auto const D  = denominator<double>(eo, ev);
    auto const A  = create_random_tensor<double>("A", nocc, nlnk);
    auto const B  = create_random_tensor<double>("B", nlnk, nvir);
    auto       P  = create_zero_tensor<double>("P", nocc, nvir);

    cg::Graph graph("folded");
    capture_program<double>(graph, A, B, D, &P);
    graph.annotate_tag(D, cg::passes::LaplaceTransform::denominator_tag({"eps_o", "eps_v", "eps_o", "eps_v"}, "-+-+"));

    Outcome const outcome = run_pass(graph, eo, ev);
    REQUIRE(outcome.transformed == 0);
    REQUIRE(mentions(outcome.skips, "names more energies than the tagged tensor has axes"));
}

TEST_CASE("LaplaceTransform - a consumer that is not a direct product is declined", "[ComputeGraph][Laplace]") {
    auto const eo = occupied<double>();
    auto const ev = virtuals<double>();
    auto const D  = denominator<double>(eo, ev);
    auto const A  = create_random_tensor<double>("A", nocc, nlnk);
    auto       P  = create_zero_tensor<double>("P", nocc, nlnk);

    cg::Graph graph("contracted");
    auto     &numerator = graph.scratch<double, 2>("numerator", nocc, nvir);
    {
        cg::CaptureGuard const guard(graph);
        cg::einsum("i,k ; k,a -> i,a", &numerator, A, create_random_tensor<double>("B", nlnk, nvir));
        // The denominator is CONTRACTED with the numerator rather than multiplied elementwise
        // by it, so there is no per-axis factor for an exponential to ride on.
        cg::einsum("i,a ; i,a -> i,k", &P, numerator, D);
    }
    graph.annotate_tag(D, cg::passes::LaplaceTransform::denominator_tag({"eps_o", "eps_v"}, "-+"));

    Outcome const outcome = run_pass(graph, eo, ev);
    REQUIRE(outcome.transformed == 0);
    REQUIRE(mentions(outcome.skips, "has no direct-product consumer"));
}

TEST_CASE("LaplaceTransform - a numerator observed from outside the region is declined", "[ComputeGraph][Laplace]") {
    auto const eo = occupied<double>();
    auto const ev = virtuals<double>();
    auto const D  = denominator<double>(eo, ev);
    auto const A  = create_random_tensor<double>("A", nocc, nlnk);
    auto const B  = create_random_tensor<double>("B", nlnk, nvir);
    auto       P  = create_zero_tensor<double>("P", nocc, nvir);
    // The numerator is the CALLER's tensor rather than graph-owned scratch, so it is
    // observable and the rewrite may not dissolve it.
    auto numerator = create_zero_tensor<double>("numerator", nocc, nvir);

    cg::Graph graph("escaping");
    {
        cg::CaptureGuard const guard(graph);
        cg::einsum("i,k ; k,a -> i,a", &numerator, A, B);
        cg::direct_product(1.0, numerator, D, 0.0, &P);
    }
    graph.annotate_tag(D, cg::passes::LaplaceTransform::denominator_tag({"eps_o", "eps_v"}, "-+"));

    Outcome const outcome = run_pass(graph, eo, ev);
    REQUIRE(outcome.transformed == 0);
    REQUIRE(mentions(outcome.skips, "observed from outside the region"));
}

TEST_CASE("LaplaceTransform - a complex denominator is declined", "[ComputeGraph][Laplace]") {
    using C       = std::complex<double>;
    auto const eo = occupied<double>();
    auto const ev = virtuals<double>();
    auto const A  = create_random_tensor<C>("A", nocc, nlnk);
    auto const B  = create_random_tensor<C>("B", nlnk, nvir);
    auto       P  = create_zero_tensor<C>("P", nocc, nvir);
    auto       D  = create_zero_tensor<C>("denom", nocc, nvir);
    for (std::size_t i = 0; i < nocc; ++i) {
        for (std::size_t a = 0; a < nvir; ++a) {
            D(i, a) = C{1.0 / (ev(a) - eo(i)), 0.0};
        }
    }

    cg::Graph graph("complex");
    auto     &numerator = graph.scratch<C, 2>("numerator", nocc, nvir);
    {
        cg::CaptureGuard const guard(graph);
        cg::einsum("i,k ; k,a -> i,a", &numerator, A, B);
        cg::direct_product(C{1}, numerator, D, C{0}, &P);
    }
    graph.annotate_tag(D, cg::passes::LaplaceTransform::denominator_tag({"eps_o", "eps_v"}, "-+"));

    Outcome const outcome = run_pass(graph, eo, ev);
    REQUIRE(outcome.transformed == 0);
    REQUIRE(mentions(outcome.skips, "is complex"));
}

TEST_CASE("LaplaceTransform - a denominator with a writer the pass cannot verify is declined", "[ComputeGraph][Laplace]") {
    auto const eo = occupied<double>();
    auto const ev = virtuals<double>();
    auto const A  = create_random_tensor<double>("A", nocc, nlnk);
    auto const B  = create_random_tensor<double>("B", nlnk, nvir);

    // Each arm writes the denominator with something the pass cannot read as the recipe the tag
    // describes, and the pass has to decline rather than fit a quadrature to a value the graph
    // will recompute into something else.
    auto const eo_rt = as_runtime(eo);
    auto const ev_rt = as_runtime(ev);

    auto const declines = [&](std::string const &label, auto &&write) {
        auto      P = create_zero_tensor<double>("P", nocc, nvir);
        cg::Graph graph(label);
        auto     &D         = graph.declare_runtime_tensor<double>("denom", {nocc, nvir}, /*intermediate=*/true);
        auto     &numerator = graph.scratch<double, 2>("numerator", nocc, nvir);
        {
            cg::CaptureGuard const guard(graph);
            write(D);
            cg::einsum("i,k ; k,a -> i,a", &numerator, A, B);
            cg::direct_product(1.0, numerator, D, 0.0, &P);
        }
        graph.annotate_tag(D, cg::passes::LaplaceTransform::denominator_tag({"eps_o", "eps_v"}, "-+"));

        Outcome const outcome = run_pass(graph, eo, ev);
        INFO(label);
        REQUIRE(outcome.transformed == 0);
        REQUIRE(mentions(outcome.skips, "has a writer this pass cannot verify"));
    };

    // A third writer, which is more of the recipe than the tag describes.
    declines("a scale on top of the chain", [&](RuntimeTensor<double> &D) {
        cg::outer_sum(&D, std::vector<RuntimeTensor<double> const *>{&eo_rt, &ev_rt}, {-1.0, 1.0});
        cg::element_transform(&D, "recip");
        cg::scale(1.0, &D);
    });

    // The right shape with the wrong arithmetic: the tag says the occupied energy is
    // subtracted and the node adds it.
    declines("a sign the tag does not say", [&](RuntimeTensor<double> &D) {
        cg::outer_sum(&D, std::vector<RuntimeTensor<double> const *>{&eo_rt, &ev_rt}, {1.0, 1.0});
        cg::element_transform(&D, "recip");
    });

    // The right shape with the wrong element operation: the tag is on the RECIPROCAL, and a
    // tensor that is the sum itself is a different quantity.
    declines("an element operation that is not the reciprocal", [&](RuntimeTensor<double> &D) {
        cg::outer_sum(&D, std::vector<RuntimeTensor<double> const *>{&eo_rt, &ev_rt}, {-1.0, 1.0});
        cg::element_transform(&D, "negate");
    });

    // And an outer sum over a vector the tag does not name.
    auto const other = as_runtime(occupied<double>("eps_other"));
    declines("an energy the tag does not name", [&](RuntimeTensor<double> &D) {
        cg::outer_sum(&D, std::vector<RuntimeTensor<double> const *>{&other, &ev_rt}, {-1.0, 1.0});
        cg::element_transform(&D, "recip");
    });
}

TEST_CASE("LaplaceTransform - a verified writer chain is accepted and dissolved with the tensor", "[ComputeGraph][Laplace]") {
    auto const eo = occupied<double>();
    auto const ev = virtuals<double>();
    auto const A  = create_random_tensor<double>("A", nocc, nlnk);
    auto const B  = create_random_tensor<double>("B", nlnk, nvir);

    // The exact answer, from the same program with the denominator built eagerly.
    auto const D_exact = denominator<double>(eo, ev);
    auto       exact   = create_zero_tensor<double>("P_exact", nocc, nvir);
    {
        cg::Graph reference("reference");
        capture_program<double>(reference, A, B, D_exact, &exact);
        apply_defaults(reference);
        reference.execute();
    }

    auto      P = create_zero_tensor<double>("P", nocc, nvir);
    cg::Graph graph("verified chain");
    // DEFERRED, which is the point: a denominator the rewrite dissolves is never allocated, so
    // the full-axis form does not pay for the four-index tensor it writes down.
    auto      &D         = graph.declare_runtime_tensor<double>("denom", {nocc, nvir}, /*intermediate=*/true);
    auto      &numerator = graph.scratch<double, 2>("numerator", nocc, nvir);
    auto const eo_rt     = as_runtime(eo);
    auto const ev_rt     = as_runtime(ev);
    {
        cg::CaptureGuard const guard(graph);
        cg::outer_sum(&D, std::vector<RuntimeTensor<double> const *>{&eo_rt, &ev_rt}, {-1.0, 1.0});
        cg::element_transform(&D, "recip");
        cg::einsum("i,k ; k,a -> i,a", &numerator, A, B);
        cg::direct_product(1.0, numerator, D, 0.0, &P);
    }
    graph.annotate_tag(D, cg::passes::LaplaceTransform::denominator_tag({"eps_o", "eps_v"}, "-+"));

    cg::passes::LaplaceTransform laplace;
    laplace.set_epsilon(1.0e-8);
    supply_energies(laplace, eo, ev);
    cg::PassManager pm;
    pm.add(borrow(laplace));
    REQUIRE(graph.apply(pm));
    REQUIRE(laplace.num_transformed() == 1);

    // The chain is gone: nothing forms the denominator any more.
    for (auto const &node : graph.nodes()) {
        REQUIRE(node.kind != cg::OpKind::ElementTransform);
        REQUIRE(node.label != "outer_sum");
    }

    apply_defaults(graph);
    // And nothing allocates it either, which is what an unused deferred intermediate gets.
    for (auto const &node : graph.nodes()) {
        if (node.kind != cg::OpKind::Alloc) {
            continue;
        }
        for (cg::TensorId const out : node.outputs) {
            REQUIRE(graph.tensor(out).name != "denom");
        }
    }

    graph.execute();
    for (std::size_t i = 0; i < nocc; ++i) {
        for (std::size_t a = 0; a < nvir; ++a) {
            REQUIRE(P(i, a) == Catch::Approx(exact(i, a)).epsilon(1.0e-6));
        }
    }
}

TEST_CASE("LaplaceTransform - a tag naming an energy this graph cannot use is declined", "[ComputeGraph][Laplace]") {
    auto const eo = occupied<double>();
    auto const ev = virtuals<double>();
    auto const D  = denominator<double>(eo, ev);
    auto const A  = create_random_tensor<double>("A", nocc, nlnk);
    auto const B  = create_random_tensor<double>("B", nlnk, nvir);
    auto       P  = create_zero_tensor<double>("P", nocc, nvir);

    cg::Graph graph("missing_energy");
    capture_program<double>(graph, A, B, D, &P);
    graph.annotate_tag(D, cg::passes::LaplaceTransform::denominator_tag({"eps_o", "not_a_tensor"}, "-+"));

    Outcome const outcome = run_pass(graph, eo, ev);
    REQUIRE(outcome.transformed == 0);
    REQUIRE(mentions(outcome.skips, "cannot use"));
}

TEST_CASE("LaplaceTransform - the tag helper refuses a list it cannot mean", "[ComputeGraph][Laplace]") {
    REQUIRE_THROWS(cg::passes::LaplaceTransform::denominator_tag({}, ""));
    REQUIRE_THROWS(cg::passes::LaplaceTransform::denominator_tag({"eps_o"}, "-+"));
    REQUIRE_THROWS(cg::passes::LaplaceTransform::denominator_tag({"eps_o"}, "x"));
    REQUIRE(cg::passes::LaplaceTransform::tag_name() == "laplace_denominator");

    // Two vectors under one name would make which of them the quadrature reads depend on
    // registration order, and a manifest binds by name.
    auto const                   eo = occupied<double>();
    cg::passes::LaplaceTransform laplace;
    laplace.add_energy("eps_o", eo);
    REQUIRE_THROWS(laplace.add_energy("eps_o", eo));
    laplace.clear_energies();
    laplace.add_energy("eps_o", eo);
}

// ── Save, load, rebind, refit ───────────────────────────────────────────────

TEST_CASE("LaplaceTransform - a transformed graph saves, loads and refits at a new geometry", "[ComputeGraph][Laplace]") {
    auto const eo = occupied<double>();
    auto const ev = virtuals<double>();
    auto const D  = denominator<double>(eo, ev);
    auto const A  = create_random_tensor<double>("A", nocc, nlnk);
    auto const B  = create_random_tensor<double>("B", nlnk, nvir);
    auto       P  = create_zero_tensor<double>("P", nocc, nvir);

    cg::Graph graph("laplace_roundtrip");
    capture_program<double>(graph, A, B, D, &P);
    graph.annotate_tag(D, cg::passes::LaplaceTransform::denominator_tag({"eps_o", "eps_v"}, "-+"));

    cg::passes::LaplaceTransform laplace;
    supply_energies(laplace, eo, ev);
    laplace.set_epsilon(1.0e-8);
    cg::PassManager pm;
    pm.add(borrow(laplace));
    REQUIRE(graph.apply(pm));

    // Nothing in the rewrite is a closure, which is what makes the round trip possible at all.
    REQUIRE(graph.serializability_report().empty());
    std::string const text = must_save(graph);

    cg::Graph loaded = must_load(text);
    REQUIRE(loaded.approximations().size() == 1);
    REQUIRE(loaded.approximations()[0].pass_name == "LaplaceTransform");

    // A different geometry, and deliberately one where the OLD rule is badly wrong rather than
    // merely suboptimal. The captured gaps span [0.35, 2.9]; these span [0.02, 0.59], so the
    // exponents the old grid reaches are far too small for the new smallest gap and a stale
    // rule is off by about a quarter of the answer. A geometry that merely widened the range
    // would have passed this test without refitting anything, which was the first version.
    Tensor<double, 1> eo2("eps_o", nocc);
    Tensor<double, 1> ev2("eps_v", nvir);
    for (std::size_t i = 0; i < nocc; ++i) {
        eo2(i) = eo(i);
    }
    for (std::size_t a = 0; a < nvir; ++a) {
        ev2(a) = -0.43 + 0.03 * static_cast<double>(a);
    }
    auto D2 = denominator<double>(eo2, ev2);
    auto A2 = create_random_tensor<double>("A", nocc, nlnk);
    auto B2 = create_random_tensor<double>("B", nlnk, nvir);
    auto P2 = create_zero_tensor<double>("P", nocc, nvir);

    // No "denom": the rewrite DISSOLVED it. A transformed graph reads the energies and never
    // the reciprocal, so the denominator has left the interface, which is the substitution
    // showing up where a caller can see it.
    loaded.bind_begin();
    loaded.bind_add("eps_o", eo2);
    loaded.bind_add("eps_v", ev2);
    loaded.bind_add("A", A2);
    loaded.bind_add("B", B2);
    loaded.bind_add("P", P2);
    loaded.bind_commit();

    apply_defaults(loaded);
    loaded.execute();

    // The oracle at the NEW geometry, from a fresh capture with no pass.
    auto      expected = create_zero_tensor<double>("P", nocc, nvir);
    cg::Graph reference("reference");
    capture_program<double>(reference, A2, B2, D2, &expected);
    apply_defaults(reference);
    reference.execute();

    double worst = 0.0;
    double scale = 0.0;
    for (std::size_t i = 0; i < nocc; ++i) {
        for (std::size_t a = 0; a < nvir; ++a) {
            worst = std::max(worst, std::abs(P2(i, a) - expected(i, a)));
            scale = std::max(scale, std::abs(expected(i, a)));
        }
    }
    INFO("worst after the rebind: " << worst << " on a scale of " << scale);
    // A stale rule is off by about 0.24 here, so a bound of 1e-6 is the refit being asserted
    // rather than a tolerance chosen to accommodate whatever came out.
    REQUIRE(worst <= 1.0e-6 * scale);
}
