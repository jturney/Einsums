//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

// Prefix scans along one axis: scan and the cumsum, cumprod, cummax and cummin wrappers.
//
// Expected values come from a brute-force fold written here, one line at a time with nested
// loops, never through the library's kernel. Each case runs eagerly and captured, and a captured
// graph replays twice, so an accumulate-instead-of-assign bug shows as a wrong second answer.

#include <Einsums/ComputeGraph.hpp>
#include <Einsums/ComputeGraph/GraphIR.hpp>
#include <Einsums/Tensor/RuntimeTensor.hpp>
#include <Einsums/Tensor/Tensor.hpp>
#include <Einsums/TensorUtilities/CreateRandomTensor.hpp>
#include <Einsums/TensorUtilities/CreateZeroTensor.hpp>

#include <array>
#include <cmath>
#include <complex>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

#ifdef _OPENMP
#    include <omp.h>
#endif

#include <Einsums/Testing.hpp>

using namespace einsums;
namespace cg = einsums::compute_graph;

namespace {

/// The scan's binary operator, written independently: NaN wins max and min, and of two equal
/// zeros the maximum is +0 unless both are -0 (and the minimum the reverse).
template <typename T>
T ref_combine(std::string_view op, T acc, T x) {
    if (op == "sum") {
        return acc + x;
    }
    if (op == "prod") {
        return acc * x;
    }
    if constexpr (!IsComplexV<T>) {
        bool const max = op == "max";
        if (std::isnan(acc) || std::isnan(x)) {
            return std::numeric_limits<T>::quiet_NaN();
        }
        if (acc == T{0} && x == T{0}) {
            bool const negative = max ? (std::signbit(acc) && std::signbit(x)) : (std::signbit(acc) || std::signbit(x));
            return negative ? -T{0} : T{0};
        }
        return max ? (acc > x ? acc : x) : (acc < x ? acc : x);
    }
    FAIL("no reference for op " << op);
    return acc;
}

template <typename T>
T ref_identity(std::string_view op) {
    if (op == "sum") {
        return T{0};
    }
    if (op == "prod") {
        return T{1};
    }
    if constexpr (!IsComplexV<T>) {
        return op == "max" ? -std::numeric_limits<T>::infinity() : std::numeric_limits<T>::infinity();
    }
    return T{0};
}

/// The brute-force scan of a rank-3 tensor along @p axis, into a fresh tensor.
template <typename T>
Tensor<T, 3> ref_scan(Tensor<T, 3> const &A, size_t axis, std::string_view op, bool exclusive, bool reverse) {
    Tensor<T, 3>                out("ref", A.dim(0), A.dim(1), A.dim(2));
    std::array<size_t, 3> const dims{A.dim(0), A.dim(1), A.dim(2)};
    std::array<size_t, 3>       other{};
    size_t                      o = 0;
    for (size_t a = 0; a < 3; ++a) {
        if (a != axis) {
            other[o++] = a;
        }
    }
    for (size_t p = 0; p < dims[other[0]]; ++p) {
        for (size_t q = 0; q < dims[other[1]]; ++q) {
            T acc = ref_identity<T>(op);
            for (size_t step = 0; step < dims[axis]; ++step) {
                size_t const          k = reverse ? dims[axis] - 1 - step : step;
                std::array<size_t, 3> idx{};
                idx[other[0]] = p;
                idx[other[1]] = q;
                idx[axis]     = k;
                T const x     = A(idx[0], idx[1], idx[2]);
                if (exclusive) {
                    out(idx[0], idx[1], idx[2]) = acc;
                    acc                         = ref_combine(op, acc, x);
                } else {
                    acc                         = step == 0 ? x : ref_combine(op, acc, x);
                    out(idx[0], idx[1], idx[2]) = acc;
                }
            }
        }
    }
    return out;
}

template <typename T>
bool same_value(T a, T b, RemoveComplexT<T> tol) {
    if constexpr (IsComplexV<T>) {
        return std::abs(a - b) <= tol * (1 + std::abs(b));
    } else {
        if (std::isnan(a) || std::isnan(b)) {
            return std::isnan(a) && std::isnan(b);
        }
        if (std::isinf(b)) {
            return a == b;
        }
        if (a == T{0} && b == T{0}) {
            return std::signbit(a) == std::signbit(b);
        }
        return std::abs(a - b) <= tol * (1 + std::abs(b));
    }
}

template <typename T>
RemoveComplexT<T> tol_for() {
    if constexpr (std::is_same_v<RemoveComplexT<T>, float>) {
        return 1e-5F;
    } else {
        return 1e-12;
    }
}

template <typename Body>
void run(bool captured, Body body) {
    if (!captured) {
        body();
        return;
    }
    cg::Graph graph("scan");
    {
        cg::CaptureGuard const guard(graph);
        body();
    }
    graph.execute();
    graph.execute();
}

template <typename T>
std::vector<std::string_view> ops_for() {
    if constexpr (IsComplexV<T>) {
        return {"sum", "prod"};
    } else {
        return {"sum", "prod", "max", "min"};
    }
}

} // namespace

// ── semantics over views, axes and flags ─────────────────────────────────────

TEMPLATE_LIST_TEST_CASE("Scan - every op, axis and flag matches a brute-force fold over a strided view", "[ComputeGraph][Scan]",
                        testing::AllScalarTypes) {
    using T                = TestType;
    bool const   captured  = GENERATE(false, true);
    size_t const axis      = GENERATE(0, 1, 2);
    bool const   exclusive = GENERATE(false, true);
    bool const   reverse   = GENERATE(false, true);

    // A slice of a larger parent, so the scan axis and the line axes all walk by stride. Values
    // near 1 keep a product of a few of them well scaled.
    auto parent = create_random_tensor<T>("parent", 5, 4, 3);
    for (size_t i = 0; i < parent.size(); ++i) {
        parent.data()[i] = T{1} + parent.data()[i] * T{0.25};
    }
    auto       A  = parent(Range{1, 4}, All, All);
    auto const A0 = Tensor<T, 3>(A);

    for (auto const op : ops_for<T>()) {
        INFO("op " << op << " axis " << axis << " exclusive " << exclusive << " reverse " << reverse << " captured " << captured);
        auto out = create_zero_tensor<T>("out", 3, 4, 3);
        run(captured, [&] { cg::scan(&out, A, axis, op, exclusive, reverse); });
        auto const expected = ref_scan(A0, axis, op, exclusive, reverse);
        for (size_t i = 0; i < out.size(); ++i) {
            INFO("element " << i);
            REQUIRE(same_value(out.data()[i], expected.data()[i], tol_for<T>()));
        }
    }
}

TEMPLATE_TEST_CASE("Scan - the named wrappers are scan with their op fixed", "[ComputeGraph][Scan]", float, double) {
    using T      = TestType;
    auto const A = create_random_tensor<T>("A", 6, 3);
    auto       a = create_zero_tensor<T>("a", 6, 3);
    auto       b = create_zero_tensor<T>("b", 6, 3);
    for (auto const op : ops_for<T>()) {
        INFO("op " << op);
        cg::scan(&a, A, 0, op);
        if (op == "sum") {
            cg::cumsum(&b, A, 0);
        } else if (op == "prod") {
            cg::cumprod(&b, A, 0);
        } else if (op == "max") {
            cg::cummax(&b, A, 0);
        } else {
            cg::cummin(&b, A, 0);
        }
        REQUIRE(std::memcmp(a.data(), b.data(), a.size() * sizeof(T)) == 0);
    }
}

TEST_CASE("Scan - max and min propagate NaN as numpy's accumulate does", "[ComputeGraph][Scan]") {
    double const                nan = std::numeric_limits<double>::quiet_NaN();
    RuntimeTensor<double>       A("A", std::vector<size_t>{5});
    RuntimeTensor<double>       out("out", std::vector<size_t>{5});
    std::array<double, 5> const a{1.0, 3.0, nan, 5.0, 0.0};
    for (size_t i = 0; i < 5; ++i) {
        A.data()[i] = a[i];
    }
    cg::cummax(&out, A, 0);
    REQUIRE(out.data()[0] == 1.0);
    REQUIRE(out.data()[1] == 3.0);
    for (size_t i = 2; i < 5; ++i) {
        REQUIRE(std::isnan(out.data()[i])); // once NaN is seen, every later element is NaN
    }

    // An all-NaN line stays NaN, and so does every op over it.
    for (size_t i = 0; i < 5; ++i) {
        A.data()[i] = nan;
    }
    for (auto const op : {"sum", "prod", "max", "min"}) {
        INFO("op " << op);
        cg::scan(&out, A, 0, op);
        for (size_t i = 0; i < 5; ++i) {
            REQUIRE(std::isnan(out.data()[i]));
        }
    }
}

TEST_CASE("Scan - an exclusive scan starts from the operator's identity", "[ComputeGraph][Scan]") {
    double const          inf = std::numeric_limits<double>::infinity();
    RuntimeTensor<double> A("A", std::vector<size_t>{3});
    RuntimeTensor<double> out("out", std::vector<size_t>{3});
    A.data()[0] = 2.0, A.data()[1] = 3.0, A.data()[2] = 4.0;
    cg::scan(&out, A, 0, "sum", true);
    REQUIRE((out.data()[0] == 0.0 && out.data()[1] == 2.0 && out.data()[2] == 5.0));
    cg::scan(&out, A, 0, "prod", true);
    REQUIRE((out.data()[0] == 1.0 && out.data()[1] == 2.0 && out.data()[2] == 6.0));
    cg::scan(&out, A, 0, "max", true);
    REQUIRE((out.data()[0] == -inf && out.data()[1] == 2.0 && out.data()[2] == 3.0));
    cg::scan(&out, A, 0, "min", true, true);
    REQUIRE((out.data()[2] == inf && out.data()[1] == 4.0 && out.data()[0] == 3.0));
}

TEST_CASE("Scan - a rank-1 tensor and zero extents", "[ComputeGraph][Scan]") {
    RuntimeTensor<double> A("A", std::vector<size_t>{4});
    RuntimeTensor<double> out("out", std::vector<size_t>{4});
    for (size_t i = 0; i < 4; ++i) {
        A.data()[i] = static_cast<double>(i + 1);
    }
    cg::cumsum(&out, A, 0);
    REQUIRE((out.data()[0] == 1.0 && out.data()[1] == 3.0 && out.data()[2] == 6.0 && out.data()[3] == 10.0));

    // A zero-extent scan axis, or a zero extent elsewhere, is a no-op: nothing to write.
    for (auto const &shape : {std::vector<size_t>{0, 3}, std::vector<size_t>{3, 0}}) {
        RuntimeTensor<double> E("E", shape);
        RuntimeTensor<double> F("F", shape);
        for (size_t axis = 0; axis < 2; ++axis) {
            REQUIRE_NOTHROW(cg::cumprod(&F, E, axis));
            cg::Graph graph("empty_scan");
            {
                cg::CaptureGuard const guard(graph);
                cg::cummax(&F, E, axis);
            }
            REQUIRE_NOTHROW(graph.execute());
        }
    }
}

// ── aliasing ─────────────────────────────────────────────────────────────────

TEST_CASE("Scan - an in-place scan over the same tensor or view is exact", "[ComputeGraph][Scan]") {
    for (bool const exclusive : {false, true}) {
        INFO("exclusive " << exclusive);
        auto       parent = create_random_tensor<double>("parent", 5, 4);
        auto const P0     = parent;
        auto       view   = parent(Range{1, 4}, All);
        auto const V0     = Tensor<double, 2>(view);

        cg::Graph graph("in_place_scan");
        {
            cg::CaptureGuard const guard(graph);
            cg::scan(&view, view, 1, "sum", exclusive);
        }
        graph.execute();

        for (size_t i = 0; i < 3; ++i) {
            double acc = 0.0;
            for (size_t j = 0; j < 4; ++j) {
                double const x = V0(i, j);
                if (exclusive) {
                    REQUIRE(view(i, j) == acc);
                    acc += x;
                } else {
                    acc += x;
                    REQUIRE(view(i, j) == acc);
                }
            }
        }
        // The rows outside the view are untouched.
        for (size_t j = 0; j < 4; ++j) {
            REQUIRE(parent(0, j) == P0(0, j));
            REQUIRE(parent(4, j) == P0(4, j));
        }
    }
}

TEST_CASE("Scan - a partially overlapping destination is refused", "[ComputeGraph][Scan]") {
    auto line = create_random_tensor<double>("line", 10);
    auto A    = line(Range{0, 6});
    auto C    = line(Range{2, 8});
    REQUIRE_THROWS_AS(cg::cumsum(&C, A, 0), std::invalid_argument);

    RuntimeTensor<double> M("M", std::vector<size_t>{3, 3});
    M.zero();
    auto Mt = M.permute_view({1, 0});
    REQUIRE_THROWS_AS(cg::cumsum(&Mt, M, 0), std::invalid_argument);
}

// ── rejections ───────────────────────────────────────────────────────────────

TEST_CASE("Scan - a bad axis, op, shape or complex max is refused at the API", "[ComputeGraph][Scan]") {
    RuntimeTensor<double> A("A", std::vector<size_t>{2, 3});
    RuntimeTensor<double> out("out", std::vector<size_t>{2, 3});
    A.zero();

    cg::Graph graph("bad_scan");
    {
        cg::CaptureGuard const guard(graph);
        REQUIRE_THROWS_AS(cg::scan(&out, A, 2, "sum"), std::out_of_range);
        REQUIRE_THROWS_WITH(cg::scan(&out, A, 0, "mean"),
                            Catch::Matchers::ContainsSubstring("mean") && Catch::Matchers::ContainsSubstring("sum, prod, max, min"));
        RuntimeTensor<double> wrong("wrong", std::vector<size_t>{3, 2});
        REQUIRE_THROWS_AS(cg::scan(&wrong, A, 0, "sum"), DimensionError);
        RuntimeTensor<double> rank3("rank3", std::vector<size_t>{2, 3, 1});
        REQUIRE_THROWS_AS(cg::scan(&rank3, A, 0, "sum"), RankError);

        RuntimeTensor<std::complex<double>> Z("Z", std::vector<size_t>{2, 3});
        RuntimeTensor<std::complex<double>> zout("zout", std::vector<size_t>{2, 3});
        Z.zero();
        REQUIRE_THROWS_WITH(cg::scan(&zout, Z, 0, "max"), Catch::Matchers::ContainsSubstring("real"));
        REQUIRE_THROWS_AS(cg::scan(&zout, Z, 0, "min"), std::invalid_argument);
        REQUIRE_NOTHROW(cg::scan(&zout, Z, 0, "sum"));
    }
    REQUIRE(graph.num_nodes() == 1); // only the complex sum was recorded
}

namespace {
template <typename C, typename A>
concept CummaxAccepts = requires(C *c, A const &a) { cg::cummax(c, a, size_t{0}); };
} // namespace

TEST_CASE("Scan - cummax and cummin have no complex overload", "[ComputeGraph][Scan]") {
    STATIC_REQUIRE_FALSE(CummaxAccepts<RuntimeTensor<std::complex<double>>, RuntimeTensor<std::complex<double>>>);
    STATIC_REQUIRE(CummaxAccepts<RuntimeTensor<double>, RuntimeTensor<double>>);
}

// ── reproducibility, passes, loops and GraphIR ───────────────────────────────

TEST_CASE("Scan - cumsum is bit-identical with one thread and with many", "[ComputeGraph][Scan]") {
    // Many short lines, so the many-thread run actually divides lines among threads.
    RuntimeTensor<double> A("A", std::vector<size_t>{257, 64, 9});
    for (size_t i = 0; i < A.size(); ++i) {
        A.data()[i] = std::sin(static_cast<double>(i)) * 1e3 + 1e-3 * static_cast<double>(i % 7);
    }
    RuntimeTensor<double> serial("serial", std::vector<size_t>{257, 64, 9});
    RuntimeTensor<double> wide("wide", std::vector<size_t>{257, 64, 9});

#ifdef _OPENMP
    int const prior = omp_get_max_threads();
    omp_set_num_threads(1);
#endif
    cg::cumsum(&serial, A, 2);
#ifdef _OPENMP
    omp_set_num_threads(std::max(prior, 4));
#endif
    cg::cumsum(&wide, A, 2);
#ifdef _OPENMP
    omp_set_num_threads(prior);
#endif
    REQUIRE(std::memcmp(serial.data(), wide.data(), serial.size() * sizeof(double)) == 0);
}

TEST_CASE("Scan - results hold under the default passes and inside a loop body", "[ComputeGraph][Scan]") {
    auto const            A = create_random_tensor<double>("A", 4, 5);
    RuntimeTensor<double> S("S", std::vector<size_t>{4, 5});
    RuntimeTensor<double> M("M", std::vector<size_t>{4, 5});
    cg::Graph             graph("scan_loop");
    auto                 &body = graph.add_loop("iter", 3, [](size_t iter) -> bool { return iter + 1 < 3; });
    {
        cg::CaptureGuard const guard(body);
        cg::cumsum(&S, A, 1);
        cg::cummax(&M, S, 1);
    }
    auto passes = cg::PassManager::create_default();
    graph.apply(passes);
    graph.execute();
    for (size_t i = 0; i < 4; ++i) {
        double acc = 0.0, hi = -std::numeric_limits<double>::infinity();
        for (size_t j = 0; j < 5; ++j) {
            acc += A(i, j);
            hi = std::max(hi, acc);
            REQUIRE_THAT(S.data()[i + 4 * j], Catch::Matchers::WithinAbs(acc, 1e-12));
            REQUIRE_THAT(M.data()[i + 4 * j], Catch::Matchers::WithinAbs(hi, 1e-12));
        }
    }
}

TEMPLATE_LIST_TEST_CASE("Scan - a scan round-trips through GraphIR bitwise", "[ComputeGraph][Scan][SaveLoad]", testing::AllScalarTypes) {
    using T = TestType;
    auto A  = create_random_tensor<T>("A", 3, 4);
    auto S  = create_zero_tensor<T>("S", 3, 4);
    auto P  = create_zero_tensor<T>("P", 3, 4);

    cg::Graph graph("scan_ir");
    {
        cg::CaptureGuard const guard(graph);
        cg::scan(&S, A, 1, "sum", true, false);
        cg::scan(&P, A, 0, "prod", false, true);
    }
    graph.execute();

    auto text = cg::save_graph_string(graph);
    INFO((text ? std::string{} : text.error().message));
    REQUIRE(text.has_value());
    REQUIRE_THAT(*text, Catch::Matchers::ContainsSubstring("\"Scan\""));

    auto loaded = cg::load_graph_string(*text);
    INFO((loaded ? std::string{} : loaded.error().message));
    REQUIRE(loaded.has_value());
    auto S2 = create_zero_tensor<T>("S2", 3, 4);
    auto P2 = create_zero_tensor<T>("P2", 3, 4);
    loaded->bind("A", A);
    loaded->bind("S", S2);
    loaded->bind("P", P2);
    loaded->execute();
    REQUIRE(std::memcmp(S.data(), S2.data(), S.size() * sizeof(T)) == 0);
    REQUIRE(std::memcmp(P.data(), P2.data(), P.size() * sizeof(T)) == 0);
}
