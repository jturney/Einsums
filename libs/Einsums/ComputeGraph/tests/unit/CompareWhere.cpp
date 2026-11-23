//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

// compare and where.
//
// Expected values come from the brute-force loops here, written with the language's own
// comparison operators and a plain branch, never through the library's kernels. Each case runs
// eagerly and captured, and a captured graph replays twice.

#include <Einsums/ComputeGraph.hpp>
#include <Einsums/ComputeGraph/GraphIR.hpp>
#include <Einsums/Tensor/RuntimeTensor.hpp>
#include <Einsums/Tensor/Tensor.hpp>
#include <Einsums/TensorUtilities/CreateRandomTensor.hpp>
#include <Einsums/TensorUtilities/CreateZeroTensor.hpp>

#include <array>
#include <complex>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

#include <Einsums/Testing.hpp>

using namespace einsums;
namespace cg = einsums::compute_graph;

namespace {

constexpr std::array<std::string_view, 6> kOps{"gt", "ge", "lt", "le", "eq", "ne"};

template <typename T>
bool ref_compare(std::string_view op, T a, T b) {
    if (op == "gt") {
        return a > b;
    }
    if (op == "ge") {
        return a >= b;
    }
    if (op == "lt") {
        return a < b;
    }
    if (op == "le") {
        return a <= b;
    }
    if (op == "eq") {
        return a == b;
    }
    return a != b;
}

/// Small integers with NaN, both infinities and both zeros planted, so ties and IEEE cases occur.
template <typename T>
Tensor<T, 3> mask_input(std::string const &name, size_t n0, size_t n1, size_t n2, unsigned seed) {
    Tensor<T, 3> t(name, n0, n1, n2);
    for (size_t i = 0; i < t.size(); ++i) {
        t.data()[i] = static_cast<T>(static_cast<int>((i * 7 + seed * 3) % 5) - 2);
    }
    t.data()[seed % t.size()]       = std::numeric_limits<T>::quiet_NaN();
    t.data()[(seed + 3) % t.size()] = std::numeric_limits<T>::infinity();
    t.data()[(seed + 5) % t.size()] = -std::numeric_limits<T>::infinity();
    t.data()[(seed + 7) % t.size()] = -T{0};
    return t;
}

template <typename Body>
void run(bool captured, Body body) {
    if (!captured) {
        body();
        return;
    }
    cg::Graph graph("compare_where");
    {
        cg::CaptureGuard const guard(graph);
        body();
    }
    graph.execute();
    graph.execute();
}

} // namespace

// ── compare ──────────────────────────────────────────────────────────────────

TEMPLATE_TEST_CASE("CompareWhere - compare of two tensors matches the language's operators", "[ComputeGraph][CompareWhere]", float,
                   double) {
    using T             = TestType;
    bool const captured = GENERATE(false, true);

    auto       A      = mask_input<T>("A", 3, 4, 2, 1);
    auto       parent = mask_input<T>("parent", 5, 4, 2, 2);
    auto       B      = parent(Range{1, 4}, All, All);
    auto const B0     = Tensor<T, 3>(B);
    for (auto const op : kOps) {
        INFO("op " << op << " captured " << captured);
        auto out = create_zero_tensor<T>("out", 3, 4, 2);
        run(captured, [&] { cg::compare(&out, A, B, op); });
        for (size_t i = 0; i < out.size(); ++i) {
            INFO("element " << i);
            REQUIRE(out.data()[i] == (ref_compare(op, A.data()[i], B0.data()[i]) ? T{1} : T{0}));
        }
    }
}

TEMPLATE_TEST_CASE("CompareWhere - compare against a scalar matches the language's operators", "[ComputeGraph][CompareWhere]", float,
                   double) {
    using T               = TestType;
    bool const   captured = GENERATE(false, true);
    double const scalar   = GENERATE(0.0, 1.0, -std::numeric_limits<double>::infinity(), std::numeric_limits<double>::quiet_NaN());

    auto       parent = mask_input<T>("parent", 5, 4, 2, 3);
    auto       A      = parent(Range{1, 4}, All, All);
    auto const A0     = Tensor<T, 3>(A);
    for (auto const op : kOps) {
        INFO("op " << op << " scalar " << scalar << " captured " << captured);
        auto out = create_zero_tensor<T>("out", 3, 4, 2);
        run(captured, [&] { cg::compare(&out, A, scalar, op); });
        for (size_t i = 0; i < out.size(); ++i) {
            INFO("element " << i);
            REQUIRE(out.data()[i] == (ref_compare(op, A0.data()[i], static_cast<T>(scalar)) ? T{1} : T{0}));
        }
    }
}

TEST_CASE("CompareWhere - every comparison with NaN is false except ne", "[ComputeGraph][CompareWhere]") {
    double const          nan = std::numeric_limits<double>::quiet_NaN();
    RuntimeTensor<double> A("A", std::vector<size_t>{3});
    RuntimeTensor<double> B("B", std::vector<size_t>{3});
    RuntimeTensor<double> out("out", std::vector<size_t>{3});
    A.data()[0] = nan, A.data()[1] = nan, A.data()[2] = 1.0;
    B.data()[0] = nan, B.data()[1] = 1.0, B.data()[2] = nan;
    for (auto const op : kOps) {
        INFO("op " << op);
        cg::compare(&out, A, B, op);
        for (size_t i = 0; i < 3; ++i) {
            REQUIRE(out.data()[i] == (op == "ne" ? 1.0 : 0.0));
        }
    }
}

TEST_CASE("CompareWhere - an unknown comparison is refused at capture, listing the valid names", "[ComputeGraph][CompareWhere]") {
    RuntimeTensor<double> A("A", std::vector<size_t>{2});
    RuntimeTensor<double> out("out", std::vector<size_t>{2});
    A.zero();
    cg::Graph graph("bad_op");
    {
        cg::CaptureGuard const guard(graph);
        REQUIRE_THROWS_AS(cg::compare(&out, A, A, "greater"), std::invalid_argument);
        REQUIRE_THROWS_WITH(cg::compare(&out, A, 0.0, "greater"),
                            Catch::Matchers::ContainsSubstring("greater") && Catch::Matchers::ContainsSubstring("gt, ge, lt, le, eq, ne"));
    }
    REQUIRE(graph.num_nodes() == 0);
}

TEST_CASE("CompareWhere - shape mismatches and partial overlaps are refused", "[ComputeGraph][CompareWhere]") {
    auto A   = create_random_tensor<double>("A", 2, 3);
    auto B   = create_random_tensor<double>("B", 3, 2);
    auto out = create_zero_tensor<double>("out", 2, 3);
    REQUIRE_THROWS_AS(cg::compare(&out, A, B, "gt"), DimensionError);
    REQUIRE_THROWS_AS(cg::where(&out, A, A, B), DimensionError);
    RuntimeTensor<double> R3("R3", std::vector<size_t>{2, 3, 1});
    RuntimeTensor<double> R2("R2", std::vector<size_t>{2, 3});
    REQUIRE_THROWS_AS(cg::compare(&R3, R2, 0.0, "gt"), RankError);

    auto line = create_random_tensor<double>("line", 10);
    auto lo   = line(Range{0, 6});
    auto hi   = line(Range{2, 8});
    auto C    = create_random_tensor<double>("C", 6);
    REQUIRE_THROWS_AS(cg::compare(&hi, lo, C, "gt"), std::invalid_argument);
    REQUIRE_THROWS_AS(cg::where(&hi, C, lo, C), std::invalid_argument);
    REQUIRE_THROWS_AS(cg::where(&hi, lo, C, C), std::invalid_argument);
}

namespace {
template <typename C, typename A>
concept CompareAccepts = requires(C *c, A const &a) { cg::compare(c, a, a, std::string_view{"eq"}); };
template <typename C, typename A>
concept CompareScalarAccepts = requires(C *c, A const &a) { cg::compare(c, a, 0.0, std::string_view{"eq"}); };
template <typename C, typename A>
concept WhereAccepts = requires(C *c, A const &a) { cg::where(c, a, a, a); };
} // namespace

TEST_CASE("CompareWhere - complex operands are refused at the API", "[ComputeGraph][CompareWhere]") {
    using Z = RuntimeTensor<std::complex<double>>;
    STATIC_REQUIRE_FALSE(CompareAccepts<Z, Z>);
    STATIC_REQUIRE_FALSE(CompareScalarAccepts<Z, Z>);
    STATIC_REQUIRE_FALSE(WhereAccepts<Z, Z>);
    using D = RuntimeTensor<double>;
    STATIC_REQUIRE(CompareAccepts<D, D>);
    STATIC_REQUIRE(CompareScalarAccepts<D, D>);
    STATIC_REQUIRE(WhereAccepts<D, D>);
}

// ── where ────────────────────────────────────────────────────────────────────

TEMPLATE_TEST_CASE("CompareWhere - where selects, element by element, over views", "[ComputeGraph][CompareWhere]", float, double) {
    using T             = TestType;
    bool const captured = GENERATE(false, true);

    auto       mask_parent = mask_input<T>("mask_parent", 5, 4, 2, 4);
    auto       mask        = mask_parent(Range{1, 4}, All, All);
    auto const M0          = Tensor<T, 3>(mask);
    auto const A           = create_random_tensor<T>("A", 3, 4, 2);
    auto const B           = create_random_tensor<T>("B", 3, 4, 2);
    auto       out         = create_zero_tensor<T>("out", 3, 4, 2);

    run(captured, [&] { cg::where(&out, mask, A, B); });
    for (size_t i = 0; i < out.size(); ++i) {
        INFO("element " << i);
        // NaN compares unequal to zero, so a NaN mask entry selects A, as numpy's truthiness does.
        T const expected = M0.data()[i] != T{0} ? A.data()[i] : B.data()[i];
        REQUIRE(out.data()[i] == expected);
    }
}

TEST_CASE("CompareWhere - where never lets the unselected branch's inf or NaN through", "[ComputeGraph][CompareWhere]") {
    // mask*A + (1-mask)*B would give NaN at every one of these: 0 * inf and 0 * NaN are NaN.
    double const                inf = std::numeric_limits<double>::infinity();
    double const                nan = std::numeric_limits<double>::quiet_NaN();
    RuntimeTensor<double>       M("M", std::vector<size_t>{4});
    RuntimeTensor<double>       A("A", std::vector<size_t>{4});
    RuntimeTensor<double>       B("B", std::vector<size_t>{4});
    RuntimeTensor<double>       out("out", std::vector<size_t>{4});
    std::array<double, 4> const m{1.0, 1.0, 0.0, 0.0}, a{1.0, 2.0, inf, nan}, b{inf, nan, 3.0, 4.0};
    for (size_t i = 0; i < 4; ++i) {
        M.data()[i] = m[i], A.data()[i] = a[i], B.data()[i] = b[i];
    }
    cg::Graph graph("select");
    {
        cg::CaptureGuard const guard(graph);
        cg::where(&out, M, A, B);
    }
    auto passes = cg::PassManager::create_default();
    graph.apply(passes);
    graph.execute();
    REQUIRE(out.data()[0] == 1.0);
    REQUIRE(out.data()[1] == 2.0);
    REQUIRE(out.data()[2] == 3.0);
    REQUIRE(out.data()[3] == 4.0);
}

TEST_CASE("CompareWhere - where may write in place over any of its operands", "[ComputeGraph][CompareWhere]") {
    for (int alias = 0; alias < 3; ++alias) {
        INFO("alias " << alias);
        auto M = create_zero_tensor<double>("M", 6);
        auto A = create_random_tensor<double>("A", 6);
        auto B = create_random_tensor<double>("B", 6);
        for (size_t i = 0; i < 6; ++i) {
            M(i) = (i % 2 == 0) ? 1.0 : 0.0;
        }
        auto const M0 = M, A0 = A, B0 = B;
        auto      &out = alias == 0 ? M : (alias == 1 ? A : B);
        cg::where(&out, M, A, B);
        for (size_t i = 0; i < 6; ++i) {
            REQUIRE(out(i) == (M0(i) != 0.0 ? A0(i) : B0(i)));
        }
    }
}

// ── passes, loops and GraphIR ────────────────────────────────────────────────

TEST_CASE("CompareWhere - a mask feeds a select inside a loop body under the passes", "[ComputeGraph][CompareWhere]") {
    auto const            A = create_random_tensor<double>("A", 4, 3);
    auto const            B = create_random_tensor<double>("B", 4, 3);
    RuntimeTensor<double> M("M", std::vector<size_t>{4, 3});
    RuntimeTensor<double> out("out", std::vector<size_t>{4, 3});
    cg::Graph             graph("compare_where_loop");
    auto                 &body = graph.add_loop("iter", 3, [](size_t iter) -> bool { return iter + 1 < 3; });
    {
        cg::CaptureGuard const guard(body);
        cg::compare(&M, A, B, "gt");
        cg::where(&out, M, A, B);
    }
    auto passes = cg::PassManager::create_default();
    graph.apply(passes);
    graph.execute();
    for (size_t i = 0; i < 4; ++i) {
        for (size_t j = 0; j < 3; ++j) {
            REQUIRE(out.data()[i + 4 * j] == (A(i, j) > B(i, j) ? A(i, j) : B(i, j)));
        }
    }
}

TEMPLATE_TEST_CASE("CompareWhere - compare and where round-trip through GraphIR bitwise", "[ComputeGraph][CompareWhere][SaveLoad]", float,
                   double) {
    using T = TestType;
    auto A  = create_random_tensor<T>("A", 3, 4);
    auto B  = create_random_tensor<T>("B", 3, 4);
    auto M  = create_zero_tensor<T>("M", 3, 4);
    auto N  = create_zero_tensor<T>("N", 3, 4);
    auto W  = create_zero_tensor<T>("W", 3, 4);

    cg::Graph graph("compare_where_ir");
    {
        cg::CaptureGuard const guard(graph);
        cg::compare(&M, A, B, "le");
        cg::compare(&N, A, 0.25, "gt");
        cg::where(&W, N, A, B);
    }
    graph.execute();

    auto text = cg::save_graph_string(graph);
    INFO((text ? std::string{} : text.error().message));
    REQUIRE(text.has_value());
    REQUIRE_THAT(*text, Catch::Matchers::ContainsSubstring("\"Compare\""));
    REQUIRE_THAT(*text, Catch::Matchers::ContainsSubstring("\"Select\""));

    auto loaded = cg::load_graph_string(*text);
    INFO((loaded ? std::string{} : loaded.error().message));
    REQUIRE(loaded.has_value());
    auto M2 = create_zero_tensor<T>("M2", 3, 4);
    auto N2 = create_zero_tensor<T>("N2", 3, 4);
    auto W2 = create_zero_tensor<T>("W2", 3, 4);
    loaded->bind("A", A);
    loaded->bind("B", B);
    loaded->bind("M", M2);
    loaded->bind("N", N2);
    loaded->bind("W", W2);
    loaded->execute();
    REQUIRE(std::memcmp(M.data(), M2.data(), M.size() * sizeof(T)) == 0);
    REQUIRE(std::memcmp(N.data(), N2.data(), N.size() * sizeof(T)) == 0);
    REQUIRE(std::memcmp(W.data(), W2.data(), W.size() * sizeof(T)) == 0);
}
