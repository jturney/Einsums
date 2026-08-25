//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

// ElementOps: the named-kernel registry that makes an ElementTransform node
// data rather than a closure. Three properties are under test here.
//
//  1. REGISTRATION IS A CONTRACT. A repeated identical registration is a
//     no-op, a conflicting one is an error, and an unknown lookup names the op
//     it could not find. Anything looser and two hosts registering the same
//     name would resolve to whichever ran first.
//  2. ONE REGISTRATION COVERS FOUR DTYPES. A generic kernel is instantiated per
//     BLAS element type at registration; a CONSTRAINED one is instantiated only
//     where it compiles, which is what makes "real-only" a property the
//     registry can see instead of a comment.
//  3. NAMED AND ANONYMOUS AGREE. cg::element_transform(C, "square") computes
//     what cg::element_transform(C, [](auto x){ return x*x; }) computes, eagerly
//     and under capture, for every dtype.

#include <Einsums/ComputeGraph.hpp>
#include <Einsums/Tensor/RuntimeTensor.hpp>
#include <Einsums/Tensor/Tensor.hpp>
#include <Einsums/TensorUtilities/CreateRandomTensor.hpp>

#include <cmath>
#include <complex>
#include <concepts>
#include <string>
#include <vector>

#include <Einsums/Testing.hpp>

using namespace einsums;
namespace cg  = einsums::compute_graph;
namespace ops = einsums::compute_graph::element_ops;

namespace {

/// A tolerance appropriate to @p T for a single-element-map comparison.
template <typename T>
RemoveComplexT<T> tol_for() {
    if constexpr (std::is_same_v<RemoveComplexT<T>, float>) {
        return 1.0e-5F;
    } else {
        return 1.0e-12;
    }
}

} // namespace

// ── Registration ───────────────────────────────────────────────────────────

TEST_CASE("ElementOps - a fresh registry starts empty and grows by registration", "[ComputeGraph][ElementOps]") {
    ops::ElementOpRegistry registry;
    REQUIRE(registry.size() == 0);
    REQUIRE_FALSE(registry.contains("bump"));

    registry.register_op(
        "bump", []<typename T>(T x) { return x + T{1}; }, ops::ElementOpSignature{.description = "x + 1"});

    REQUIRE(registry.size() == 1);
    REQUIRE(registry.contains("bump"));
    REQUIRE(registry.names() == std::vector<std::string>{"bump"});
    REQUIRE(registry.kernel<double>("bump")(2.0) == 3.0);
}

TEST_CASE("ElementOps - the same registration made twice is a no-op", "[ComputeGraph][ElementOps]") {
    // Idempotence is not a convenience: a header-scope initializer seeding an op
    // is compiled into every translation unit that includes it, and the second
    // one through must not fail.
    ops::ElementOpRegistry registry;
    auto const             kernel = []<typename T>(T x) { return x + T{1}; };

    registry.register_op("bump", kernel);
    registry.register_op("bump", kernel);
    registry.register_op("bump", kernel);

    REQUIRE(registry.size() == 1);
    REQUIRE(registry.kernel<float>("bump")(1.0F) == 2.0F);
}

TEST_CASE("ElementOps - a different kernel under a registered name is refused", "[ComputeGraph][ElementOps]") {
    ops::ElementOpRegistry registry;
    registry.register_op("bump", []<typename T>(T x) { return x + T{1}; });

    REQUIRE_THROWS_WITH(registry.register_op("bump", []<typename T>(T x) { return x + T{2}; }),
                        Catch::Matchers::ContainsSubstring("bump") && Catch::Matchers::ContainsSubstring("different content"));

    // And the FIRST registration is what survives; a refused conflict must not
    // half-replace an entry.
    REQUIRE(registry.kernel<double>("bump")(0.0) == 1.0);
}

TEST_CASE("ElementOps - the same kernel under a different signature is refused", "[ComputeGraph][ElementOps]") {
    // The declared domain is part of what a saved graph carries, so two
    // registrations that disagree about it are two different ops.
    ops::ElementOpRegistry registry;
    auto const             kernel = []<typename T>(T x) { return x + T{1}; };

    registry.register_op("bump", kernel, ops::ElementOpSignature{.domain = ops::ElementOpDomain::AllDtypes});
    REQUIRE_THROWS_WITH(registry.register_op("bump", kernel, ops::ElementOpSignature{.domain = ops::ElementOpDomain::RealOnly}),
                        Catch::Matchers::ContainsSubstring("different content"));
}

TEST_CASE("ElementOps - an unknown name is refused with the name in the message", "[ComputeGraph][ElementOps]") {
    ops::ElementOpRegistry const registry;
    REQUIRE_THROWS_WITH(registry.kernel<double>("mp2_denominator"), Catch::Matchers::ContainsSubstring("mp2_denominator"));
    REQUIRE_THROWS_WITH(registry.signature("mp2_denominator"), Catch::Matchers::ContainsSubstring("mp2_denominator"));
}

TEST_CASE("ElementOps - an empty name and a non-unary arity are refused", "[ComputeGraph][ElementOps]") {
    ops::ElementOpRegistry registry;
    auto const             kernel = []<typename T>(T x) { return x; };

    REQUIRE_THROWS_WITH(registry.register_op("", kernel), Catch::Matchers::ContainsSubstring("must not be empty"));
    REQUIRE_THROWS_WITH(registry.register_op("pair", kernel, ops::ElementOpSignature{.arity = 2}),
                        Catch::Matchers::ContainsSubstring("unary"));
}

TEST_CASE("ElementOps - a kernel that does not cover its declared domain is refused", "[ComputeGraph][ElementOps]") {
    // The registry cannot instantiate a real-only kernel for complex, and the
    // message has to say WHICH dtype is missing rather than "it did not work".
    ops::ElementOpRegistry registry;
    REQUIRE_THROWS_WITH(registry.register_op(
                            "real_only_kernel", []<std::floating_point T>(T x) { return x + T{1}; },
                            ops::ElementOpSignature{.domain = ops::ElementOpDomain::AllDtypes}),
                        Catch::Matchers::ContainsSubstring("complex<float>"));
}

TEST_CASE("ElementOps - a real-only op refuses a complex lookup by name", "[ComputeGraph][ElementOps]") {
    ops::ElementOpRegistry registry;
    registry.register_op(
        "halve", []<std::floating_point T>(T x) { return x / T{2}; }, ops::ElementOpSignature{.domain = ops::ElementOpDomain::RealOnly});

    REQUIRE(registry.kernel<double>("halve")(4.0) == 2.0);
    REQUIRE_THROWS_WITH(registry.kernel<std::complex<double>>("halve"),
                        Catch::Matchers::ContainsSubstring("halve") && Catch::Matchers::ContainsSubstring("complex<double>"));
}

TEST_CASE("ElementOps - a declared real-only domain masks a kernel that would compile for complex", "[ComputeGraph][ElementOps]") {
    // The declaration wins over what the kernel happens to support: it is the
    // declaration a saved graph carries, so it has to be the thing that binds.
    ops::ElementOpRegistry registry;
    registry.register_op(
        "bump", []<typename T>(T x) { return x + T{1}; }, ops::ElementOpSignature{.domain = ops::ElementOpDomain::RealOnly});

    REQUIRE(registry.signature("bump").domain == ops::ElementOpDomain::RealOnly);
    REQUIRE_THROWS_WITH(registry.kernel<std::complex<float>>("bump"), Catch::Matchers::ContainsSubstring("not defined for"));
}

// ── The starter set ────────────────────────────────────────────────────────

TEST_CASE("ElementOps - the process registry ships the starter ops", "[ComputeGraph][ElementOps]") {
    auto const &registry = ops::global_element_op_registry();
    for (auto const *name : {"recip", "square", "negate", "sqrt_or_zero"}) {
        INFO("op: " << name);
        REQUIRE(registry.contains(name));
    }
    REQUIRE(registry.signature("sqrt_or_zero").domain == ops::ElementOpDomain::RealOnly);
    REQUIRE(registry.signature("recip").domain == ops::ElementOpDomain::AllDtypes);
    REQUIRE(registry.signature("recip").arity == 1);
}

TEMPLATE_TEST_CASE("ElementOps - the dtype-generic starter ops compute the right thing", "[ComputeGraph][ElementOps]", float, double,
                   std::complex<float>, std::complex<double>) {
    using T = TestType;

    auto const &registry = ops::global_element_op_registry();
    T const     x        = T{2};

    REQUIRE_THAT(std::abs(registry.kernel<T>("recip")(x) - T{1} / x), Catch::Matchers::WithinAbs(0.0, tol_for<T>()));
    REQUIRE_THAT(std::abs(registry.kernel<T>("square")(x) - x * x), Catch::Matchers::WithinAbs(0.0, tol_for<T>()));
    REQUIRE_THAT(std::abs(registry.kernel<T>("negate")(x) + x), Catch::Matchers::WithinAbs(0.0, tol_for<T>()));
}

TEMPLATE_TEST_CASE("ElementOps - sqrt_or_zero guards the negative branch", "[ComputeGraph][ElementOps]", float, double) {
    using T = TestType;

    auto const kernel = ops::global_element_op_registry().kernel<T>("sqrt_or_zero");
    REQUIRE_THAT(kernel(T{4}), Catch::Matchers::WithinAbs(2.0, tol_for<T>()));
    REQUIRE(kernel(T{0}) == T{0});
    REQUIRE(kernel(T{-1}) == T{0});
}

// ── The named capture path ─────────────────────────────────────────────────

TEMPLATE_TEST_CASE("ElementOps - a named element_transform matches the anonymous one, eagerly", "[ComputeGraph][ElementOps]", float, double,
                   std::complex<float>, std::complex<double>) {
    using T = TestType;

    auto named     = create_random_tensor<T>("named", 4, 3);
    auto anonymous = named;

    cg::element_transform(&named, "square");
    cg::element_transform(&anonymous, [](T value) { return value * value; });

    for (size_t i = 0; i < 4; i++) {
        for (size_t j = 0; j < 3; j++) {
            REQUIRE_THAT(std::abs(named(i, j) - anonymous(i, j)), Catch::Matchers::WithinAbs(0.0, tol_for<T>()));
        }
    }
}

TEMPLATE_TEST_CASE("ElementOps - a named element_transform matches the anonymous one under capture", "[ComputeGraph][ElementOps]", float,
                   double, std::complex<float>, std::complex<double>) {
    using T = TestType;

    auto named     = create_random_tensor<T>("named", 5, 2);
    auto anonymous = named;

    cg::Graph graph("named_element_transform");
    {
        cg::CaptureGuard const guard(graph);
        cg::element_transform(&named, "negate");
    }
    graph.execute();

    cg::element_transform(&anonymous, [](T value) { return -value; });

    for (size_t i = 0; i < 5; i++) {
        for (size_t j = 0; j < 2; j++) {
            REQUIRE_THAT(std::abs(named(i, j) - anonymous(i, j)), Catch::Matchers::WithinAbs(0.0, tol_for<T>()));
        }
    }
}

TEST_CASE("ElementOps - a named element_transform records a descriptor, an anonymous one does not", "[ComputeGraph][ElementOps]") {
    auto named     = create_random_tensor<double>("named", 3, 3);
    auto anonymous = create_random_tensor<double>("anonymous", 3, 3);

    cg::Graph graph("descriptor_shape");
    {
        cg::CaptureGuard const guard(graph);
        cg::element_transform(&named, "recip");
        cg::element_transform(&anonymous, [](double v) { return 1.0 / v; });
    }

    REQUIRE(graph.num_nodes() == 2);
    auto const *desc = std::get_if<cg::ElementTransformDescriptor>(&graph.nodes()[0].op_data);
    REQUIRE(desc != nullptr);
    REQUIRE(desc->op_name == "recip");
    REQUIRE(std::holds_alternative<std::monostate>(graph.nodes()[1].op_data));
}

TEST_CASE("ElementOps - a named element_transform over a runtime-rank tensor works", "[ComputeGraph][ElementOps]") {
    // The closure overload needs a compile-time rank; the named one walks the
    // operand's TensorImpl and so covers the runtime-rank types too.
    RuntimeTensor<double> A("A", std::vector<size_t>{2, 3});
    A.zero();
    for (size_t item = 0; item < A.size(); item++) {
        A.data()[item] = static_cast<double>(item + 1);
    }

    cg::Graph graph("runtime_named");
    {
        cg::CaptureGuard const guard(graph);
        cg::element_transform(&A, "square");
    }
    graph.execute();

    for (size_t item = 0; item < A.size(); item++) {
        auto const expected = static_cast<double>(item + 1);
        REQUIRE_THAT(A.data()[item], Catch::Matchers::WithinAbs(expected * expected, 1.0e-12));
    }
}

TEST_CASE("ElementOps - a named element_transform over a strided view touches only the view", "[ComputeGraph][ElementOps]") {
    // The general walk is the odometer over dims and strides; a non-contiguous
    // destination is exactly what it exists for, and the elements OUTSIDE the
    // view have to come through untouched.
    auto       parent = create_random_tensor<double>("parent", 4, 4);
    auto const before = parent;

    auto column = parent(All, Range{1, 2});

    cg::Graph graph("strided_named");
    {
        cg::CaptureGuard const guard(graph);
        cg::element_transform(&column, "negate");
    }
    graph.execute();

    for (size_t i = 0; i < 4; i++) {
        for (size_t j = 0; j < 4; j++) {
            double const expected = j == 1 ? -before(i, j) : before(i, j);
            INFO("element (" << i << ", " << j << ")");
            REQUIRE_THAT(parent(i, j), Catch::Matchers::WithinAbs(expected, 1.0e-12));
        }
    }
}

TEST_CASE("ElementOps - an unregistered op name is refused at capture, naming the op", "[ComputeGraph][ElementOps]") {
    // At CAPTURE, not at replay. A name this process cannot resolve is a caller
    // error, and the design's rule is that it fails where the name is, not
    // part-way through an execute().
    auto A = create_random_tensor<double>("A", 2, 2);

    cg::Graph graph("unknown_named");
    {
        cg::CaptureGuard const guard(graph);
        REQUIRE_THROWS_WITH(cg::element_transform(&A, "no_such_op"), Catch::Matchers::ContainsSubstring("no_such_op"));
    }
    REQUIRE(graph.num_nodes() == 0);
}

TEST_CASE("ElementOps - a real-only op refuses a complex destination by name", "[ComputeGraph][ElementOps]") {
    auto A = create_random_tensor<std::complex<double>>("A", 2, 2);

    REQUIRE_THROWS_WITH(cg::element_transform(&A, "sqrt_or_zero"),
                        Catch::Matchers::ContainsSubstring("sqrt_or_zero") && Catch::Matchers::ContainsSubstring("complex<double>"));
}
