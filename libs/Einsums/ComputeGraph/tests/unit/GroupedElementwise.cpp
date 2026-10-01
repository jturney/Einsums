//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

// The grouped element-wise forms against the emission they replace: the SAME
// single-tensor call, once per member, in the same order. Their contract is
// bit-identity rather than agreement to roundoff, because an element-wise
// kernel's result cannot depend on how the run was divided, so every
// comparison between the two is on the exact bytes. The per-member result is
// checked against an independent oracle as well, since bit-identity with it
// alone would pass if the single-tensor kernel were wrong.

#include <Einsums/ComputeGraph.hpp>
#include <Einsums/Tensor/RuntimeTensor.hpp>
#include <Einsums/Testing/ReferenceEinsum.hpp>
#include <Einsums/Testing/TensorCompare.hpp>

#include <complex>
#include <cstddef>
#include <cstring>
#include <limits>
#include <random>
#include <vector>

#include <Einsums/Testing.hpp>

using namespace einsums;
namespace cg = einsums::compute_graph;

namespace {

/// Values in [0.5, 1.5], in both parts for a complex type, so no divisor is near zero.
template <typename T>
RuntimeTensor<T> randt(std::string const &name, std::vector<std::size_t> const &dims, std::mt19937 &gen) {
    RuntimeTensor<T>                       out(name, dims);
    std::uniform_real_distribution<double> dist(0.5, 1.5);
    T                                     *p = out.data();
    for (std::size_t i = 0; i < out.size(); i++) {
        if constexpr (IsComplexV<T>) {
            double const re = dist(gen);
            p[i]            = T(re, dist(gen));
        } else {
            p[i] = T(dist(gen));
        }
    }
    return out;
}

using einsums::testing::bytes_of;

/// Member extents chosen to span the cases that break a grouped form: several
/// sizes so no member's shape is the run's, and a zero extent so an empty
/// member is carried rather than special-cased.
std::vector<std::vector<std::size_t>> const kShapes = {{4, 3, 5}, {2, 2, 2}, {0, 3, 4}, {7, 1, 2}, {3, 6, 1}, {5, 5, 5}};

/// The per-member result is itself engine output, so bit-identity against it alone would pass if
/// the single-tensor kernel were wrong too. Each test anchors it to an oracle that shares no code
/// with either: reference_permute, or a plain loop over the elements. A few roundings of values
/// near one, so a hundred epsilon of the element type.
template <typename T>
einsums::testing::Tolerance oracle_tol() {
    double const eps = 100.0 * std::numeric_limits<RemoveComplexT<T>>::epsilon();
    return {.rtol = eps, .atol = eps};
}

/// A prefactor for @p T: complex with a nonzero imaginary part for a complex type, so a dropped or
/// conjugated imaginary part changes the result.
template <typename T>
T pf(double re, double im) {
    if constexpr (IsComplexV<T>) {
        return T(static_cast<RemoveComplexT<T>>(re), static_cast<RemoveComplexT<T>>(im));
    } else {
        return T(re);
    }
}

/// C = alpha * op(A, B) + beta * C element by element, beta == 0 assigning, for op a product or quotient.
template <typename T, typename Op>
RuntimeTensor<T> elementwise_oracle(T alpha, RuntimeTensor<T> const &A, RuntimeTensor<T> const &B, T beta, RuntimeTensor<T> C, Op op) {
    for (std::size_t n = 0; n < C.size(); n++) {
        T const value = alpha * op(A.data()[n], B.data()[n]);
        C.data()[n]   = beta == T(0) ? value : value + beta * C.data()[n];
    }
    return C;
}

} // namespace

// grouped_permute takes real prefactors, so its complex runs differ from the real ones in the data.
TEMPLATE_LIST_TEST_CASE("GroupedPermute - matches the per-member permute bit for bit", "[ComputeGraph][GroupedElementwise]",
                        testing::AllScalarTypes) {
    using T = TestType;
    std::mt19937 gen(7);

    std::vector<RuntimeTensor<T>> a, want, got;
    std::vector<double>           c_pfs, a_pfs;
    for (std::size_t i = 0; i < kShapes.size(); i++) {
        auto const &d = kShapes[i];
        a.push_back(randt<T>("a", d, gen));
        // "abc <- acb" keeps the leading extent and swaps the other two.
        want.push_back(randt<T>("want", {d[0], d[2], d[1]}, gen));
        got.push_back(want.back());
        c_pfs.push_back(i % 2 == 0 ? 0.0 : 1.0);
        a_pfs.push_back(0.25 * static_cast<double>(i) - 0.5);
    }

    for (std::size_t i = 0; i < a.size(); i++) {
        RuntimeTensor<T> oracle = want[i];
        einsums::testing::reference_permute("abc <- acb", T(c_pfs[i]), &oracle, T(a_pfs[i]), a[i]);
        cg::string_permute<RuntimeTensor<T>, RuntimeTensor<T>>("abc <- acb", &want[i], a[i], T(c_pfs[i]), T(a_pfs[i]));
        einsums::testing::require_tensors_close(want[i], oracle, oracle_tol<T>());
    }

    std::vector<RuntimeTensor<T> *>       c_list;
    std::vector<RuntimeTensor<T> const *> a_list;
    for (std::size_t i = 0; i < a.size(); i++) {
        c_list.push_back(&got[i]);
        a_list.push_back(&a[i]);
    }
    cg::grouped_permute<RuntimeTensor<T>, RuntimeTensor<T>>("abc <- acb", c_list, a_list, c_pfs, a_pfs);

    for (std::size_t i = 0; i < a.size(); i++) {
        REQUIRE(bytes_of(got[i]) == bytes_of(want[i]));
    }
}

TEMPLATE_LIST_TEST_CASE("GroupedPermute - capture replays match eager and each other", "[ComputeGraph][GroupedElementwise]",
                        testing::AllScalarTypes) {
    using T = TestType;
    std::mt19937 gen(11);

    std::vector<RuntimeTensor<T>> a, want, got;
    std::vector<double>           c_pfs, a_pfs;
    for (auto const &d : kShapes) {
        a.push_back(randt<T>("a", d, gen));
        want.push_back(randt<T>("want", {d[0], d[2], d[1]}, gen));
        got.push_back(want.back());
        c_pfs.push_back(1.0);
        a_pfs.push_back(-2.0);
    }

    std::vector<RuntimeTensor<T> *>       w_list, g_list;
    std::vector<RuntimeTensor<T> const *> a_list;
    for (std::size_t i = 0; i < a.size(); i++) {
        w_list.push_back(&want[i]);
        g_list.push_back(&got[i]);
        a_list.push_back(&a[i]);
    }
    std::vector<RuntimeTensor<T>> oracles = want;
    for (std::size_t i = 0; i < a.size(); i++) {
        einsums::testing::reference_permute("abc <- acb", T(c_pfs[i]), &oracles[i], T(a_pfs[i]), a[i]);
    }
    cg::grouped_permute<RuntimeTensor<T>, RuntimeTensor<T>>("abc <- acb", w_list, a_list, c_pfs, a_pfs);
    for (std::size_t i = 0; i < a.size(); i++) {
        einsums::testing::require_tensors_close(want[i], oracles[i], oracle_tol<T>());
    }

    cg::Graph g("grouped permute");
    {
        cg::CaptureGuard guard(g);
        cg::grouped_permute<RuntimeTensor<T>, RuntimeTensor<T>>("abc <- acb", g_list, a_list, c_pfs, a_pfs);
    }
    REQUIRE(g.num_nodes() == 1);
    g.execute();
    for (std::size_t i = 0; i < a.size(); i++) {
        REQUIRE(bytes_of(got[i]) == bytes_of(want[i]));
    }

    // A second replay accumulates again; the eager loop is stepped alongside it
    // so the comparison stays against the emission rather than against a
    // remembered value.
    cg::grouped_permute<RuntimeTensor<T>, RuntimeTensor<T>>("abc <- acb", w_list, a_list, c_pfs, a_pfs);
    g.execute();
    for (std::size_t i = 0; i < a.size(); i++) {
        REQUIRE(bytes_of(got[i]) == bytes_of(want[i]));
    }
}

TEMPLATE_LIST_TEST_CASE("GroupedDirectProduct - matches the per-member product bit for bit", "[ComputeGraph][GroupedElementwise]",
                        testing::AllScalarTypes) {
    using T = TestType;
    std::mt19937 gen(13);

    std::vector<RuntimeTensor<T>> a, b, want, got;
    std::vector<T>                alphas, betas;
    for (std::size_t i = 0; i < kShapes.size(); i++) {
        auto const &d = kShapes[i];
        a.push_back(randt<T>("a", d, gen));
        b.push_back(randt<T>("b", d, gen));
        want.push_back(randt<T>("want", d, gen));
        got.push_back(want.back());
        alphas.push_back(pf<T>(0.5 + 0.125 * static_cast<double>(i), 0.25));
        betas.push_back(i % 3 == 0 ? T(0) : pf<T>(1.0, -0.5));
    }

    for (std::size_t i = 0; i < a.size(); i++) {
        auto const oracle = elementwise_oracle(alphas[i], a[i], b[i], betas[i], want[i], [](T x, T y) { return x * y; });
        cg::direct_product<T, RuntimeTensor<T>, RuntimeTensor<T>, RuntimeTensor<T>>(alphas[i], a[i], b[i], betas[i], &want[i]);
        einsums::testing::require_tensors_close(want[i], oracle, oracle_tol<T>());
    }

    std::vector<RuntimeTensor<T> const *> a_list, b_list;
    std::vector<RuntimeTensor<T> *>       c_list;
    for (std::size_t i = 0; i < a.size(); i++) {
        a_list.push_back(&a[i]);
        b_list.push_back(&b[i]);
        c_list.push_back(&got[i]);
    }
    cg::grouped_direct_product<T, RuntimeTensor<T>, RuntimeTensor<T>, RuntimeTensor<T>>(alphas, a_list, b_list, betas, c_list);

    for (std::size_t i = 0; i < a.size(); i++) {
        REQUIRE(bytes_of(got[i]) == bytes_of(want[i]));
    }
}

TEMPLATE_LIST_TEST_CASE("GroupedDirectDivision - matches the per-member division, eager and replayed", "[ComputeGraph][GroupedElementwise]",
                        testing::AllScalarTypes) {
    using T = TestType;
    std::mt19937 gen(17);

    std::vector<RuntimeTensor<T>> a, b, want, got;
    std::vector<T>                alphas, betas;
    for (std::size_t i = 0; i < kShapes.size(); i++) {
        auto const &d = kShapes[i];
        a.push_back(randt<T>("a", d, gen));
        b.push_back(randt<T>("b", d, gen));
        want.push_back(randt<T>("want", d, gen));
        got.push_back(want.back());
        alphas.push_back(pf<T>(-1.0, 0.5));
        betas.push_back(i % 2 == 0 ? T(0) : pf<T>(1.0, 0.25));
    }

    for (std::size_t i = 0; i < a.size(); i++) {
        auto const oracle = elementwise_oracle(alphas[i], a[i], b[i], betas[i], want[i], [](T x, T y) { return x / y; });
        cg::direct_division<T, RuntimeTensor<T>, RuntimeTensor<T>, RuntimeTensor<T>>(alphas[i], a[i], b[i], betas[i], &want[i]);
        einsums::testing::require_tensors_close(want[i], oracle, oracle_tol<T>());
    }

    std::vector<RuntimeTensor<T> const *> a_list, b_list;
    std::vector<RuntimeTensor<T> *>       c_list;
    for (std::size_t i = 0; i < a.size(); i++) {
        a_list.push_back(&a[i]);
        b_list.push_back(&b[i]);
        c_list.push_back(&got[i]);
    }

    cg::Graph g("grouped division");
    {
        cg::CaptureGuard guard(g);
        cg::grouped_direct_division<T, RuntimeTensor<T>, RuntimeTensor<T>, RuntimeTensor<T>>(alphas, a_list, b_list, betas, c_list);
    }
    REQUIRE(g.num_nodes() == 1);
    g.execute();

    for (std::size_t i = 0; i < a.size(); i++) {
        REQUIRE(bytes_of(got[i]) == bytes_of(want[i]));
    }
}

TEST_CASE("GroupedElementwise - rejects the runs it cannot compute", "[ComputeGraph][GroupedElementwise]") {
    std::mt19937 gen(19);

    auto x = randt<double>("x", {3, 4, 2}, gen);
    auto y = randt<double>("y", {3, 4, 2}, gen);
    auto z = randt<double>("z", {3, 2, 4}, gen);

    std::vector<RuntimeTensor<double> const *> a_list{&x, &y};
    std::vector<RuntimeTensor<double> *>       c_list{&z, &z};

    // Two members writing one tensor: the run threads, so the node has no
    // ordering to give them.
    REQUIRE_THROWS_AS(
        (cg::grouped_permute<RuntimeTensor<double>, RuntimeTensor<double>>("abc <- acb", c_list, a_list, {0.0, 0.0}, {1.0, 1.0})),
        std::invalid_argument);

    // Lists of different lengths.
    std::vector<RuntimeTensor<double> *> one{&z};
    REQUIRE_THROWS_AS((cg::grouped_permute<RuntimeTensor<double>, RuntimeTensor<double>>("abc <- acb", one, a_list, {0.0}, {1.0})),
                      std::invalid_argument);

    // An empty run is a caller mistake, not a no-op: the emission it replaces
    // had nothing to merge.
    std::vector<RuntimeTensor<double> *>       none_c;
    std::vector<RuntimeTensor<double> const *> none_a;
    REQUIRE_THROWS_AS((cg::grouped_permute<RuntimeTensor<double>, RuntimeTensor<double>>("abc <- acb", none_c, none_a, {}, {})),
                      std::invalid_argument);

    // Members whose operands disagree on shape.
    std::vector<RuntimeTensor<double> const *> mixed_a{&x};
    std::vector<RuntimeTensor<double> const *> mixed_b{&z};
    std::vector<RuntimeTensor<double> *>       mixed_c{&y};
    REQUIRE_THROWS_AS((cg::grouped_direct_product<double, RuntimeTensor<double>, RuntimeTensor<double>, RuntimeTensor<double>>(
                          {1.0}, mixed_a, mixed_b, {0.0}, mixed_c)),
                      DimensionError);
}
