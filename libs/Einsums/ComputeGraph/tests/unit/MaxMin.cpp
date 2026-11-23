//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

// maximum / minimum, max_axes / min_axes and the scalar min.
//
// Expected values come from the brute-force loops in this file, which share no code with the
// kernels: a loop over every index tuple and a fold written the obvious way. Each case runs
// eagerly and captured, and a captured graph is replayed twice, so an accumulate-instead-of-
// assign bug shows as a wrong second answer.

#include <Einsums/ComputeGraph.hpp>
#include <Einsums/ComputeGraph/GraphIR.hpp>
#include <Einsums/Tensor/RuntimeTensor.hpp>
#include <Einsums/Tensor/Tensor.hpp>
#include <Einsums/TensorUtilities/CreateRandomTensor.hpp>
#include <Einsums/TensorUtilities/CreateZeroTensor.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <complex>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

#include <Einsums/Testing.hpp>

using namespace einsums;
namespace cg = einsums::compute_graph;

namespace {

/// NumPy's maximum and minimum, written independently of the library's kernel: NaN wins, and of
/// two equal zeros the maximum is +0 unless both are -0 (and the minimum the reverse).
template <typename T>
T ref_extremum(bool max, T a, T b) {
    if (std::isnan(a) || std::isnan(b)) {
        return std::numeric_limits<T>::quiet_NaN();
    }
    if (a == T{0} && b == T{0}) {
        bool const both_negative = std::signbit(a) && std::signbit(b);
        bool const any_negative  = std::signbit(a) || std::signbit(b);
        bool const negative      = max ? both_negative : any_negative;
        return negative ? -T{0} : T{0};
    }
    return max ? (a > b ? a : b) : (a < b ? a : b);
}

/// Whether two values are the same value, NaN equal to NaN and the zeros told apart.
template <typename T>
bool same_value(T a, T b) {
    if (std::isnan(a) || std::isnan(b)) {
        return std::isnan(a) && std::isnan(b);
    }
    return a == b && std::signbit(a) == std::signbit(b);
}

/// A random rank-3 tensor with a few special values planted at fixed positions.
template <typename T>
Tensor<T, 3> special_tensor(std::string const &name, size_t n0, size_t n1, size_t n2, bool plant) {
    auto t = create_random_tensor<T>(name, n0, n1, n2);
    if (plant) {
        t(0, 0, 0)           = std::numeric_limits<T>::quiet_NaN();
        t(1, 1, 1)           = -std::numeric_limits<T>::infinity();
        t(n0 - 1, 0, n2 - 1) = -T{0};
        t(0, n1 - 1, 0)      = T{0};
    }
    return t;
}

/// Run @p body eagerly, or captured and replayed twice.
template <typename Body>
void run(bool captured, Body body) {
    if (!captured) {
        body();
        return;
    }
    cg::Graph graph("max_min");
    {
        cg::CaptureGuard const guard(graph);
        body();
    }
    graph.execute();
    graph.execute();
}

} // namespace

// ── elementwise maximum / minimum ────────────────────────────────────────────

TEMPLATE_TEST_CASE("MaxMin - elementwise maximum and minimum match the reference over views", "[ComputeGraph][MaxMin]", float, double) {
    using T             = TestType;
    bool const captured = GENERATE(false, true);
    bool const max      = GENERATE(true, false);
    INFO("captured " << captured << " max " << max);

    auto A          = special_tensor<T>("A", 3, 4, 2, true);
    auto parent     = special_tensor<T>("parent", 5, 4, 2, false);
    parent(2, 2, 1) = T{0};
    auto       B    = parent(Range{1, 4}, All, All); // a strided slice
    auto       C    = create_zero_tensor<T>("C", 3, 4, 2);
    auto const A0   = A;
    auto const P0   = parent;

    run(captured, [&] {
        if (max) {
            cg::maximum(&C, A, B);
        } else {
            cg::minimum(&C, A, B);
        }
    });

    for (size_t i = 0; i < 3; ++i) {
        for (size_t j = 0; j < 4; ++j) {
            for (size_t k = 0; k < 2; ++k) {
                INFO("(" << i << ", " << j << ", " << k << ")");
                REQUIRE(same_value(C(i, j, k), ref_extremum(max, A0(i, j, k), P0(i + 1, j, k))));
            }
        }
    }
    // The slice's parent is read, never written.
    for (size_t i = 0; i < parent.size(); ++i) {
        REQUIRE(same_value(parent.data()[i], P0.data()[i]));
    }
}

TEST_CASE("MaxMin - the signed zeros and NaN follow numpy in either operand order", "[ComputeGraph][MaxMin]") {
    double const                nan = std::numeric_limits<double>::quiet_NaN();
    RuntimeTensor<double>       A("A", std::vector<size_t>{6});
    RuntimeTensor<double>       B("B", std::vector<size_t>{6});
    RuntimeTensor<double>       C("C", std::vector<size_t>{6});
    std::array<double, 6> const a{0.0, -0.0, nan, 1.0, -0.0, nan};
    std::array<double, 6> const b{-0.0, 0.0, 1.0, nan, -0.0, nan};
    for (size_t i = 0; i < 6; ++i) {
        A.data()[i] = a[i];
        B.data()[i] = b[i];
    }
    for (bool const max : {true, false}) {
        if (max) {
            cg::maximum(&C, A, B);
        } else {
            cg::minimum(&C, A, B);
        }
        for (size_t i = 0; i < 6; ++i) {
            INFO("max " << max << " element " << i);
            REQUIRE(same_value(C.data()[i], ref_extremum(max, a[i], b[i])));
        }
    }
}

TEST_CASE("MaxMin - an elementwise destination may be one of its operands", "[ComputeGraph][MaxMin]") {
    auto       A  = create_random_tensor<double>("A", 4, 3);
    auto const B  = create_random_tensor<double>("B", 4, 3);
    auto const A0 = A;

    cg::Graph graph("in_place");
    {
        cg::CaptureGuard const guard(graph);
        cg::maximum(&A, A, B);
    }
    graph.execute();
    for (size_t i = 0; i < 4; ++i) {
        for (size_t j = 0; j < 3; ++j) {
            REQUIRE(A(i, j) == ref_extremum(true, A0(i, j), B(i, j)));
        }
    }
}

TEST_CASE("MaxMin - a partially overlapping destination is refused", "[ComputeGraph][MaxMin]") {
    auto line = create_random_tensor<double>("line", 10);
    auto A    = line(Range{0, 6});
    auto C    = line(Range{2, 8});
    auto B    = create_random_tensor<double>("B", 6);
    REQUIRE_THROWS_AS(cg::maximum(&C, A, B), std::invalid_argument);
    REQUIRE_THROWS_AS(cg::minimum(&C, B, A), std::invalid_argument);

    // Same base pointer, different layout: the transpose of the operand.
    RuntimeTensor<double> M("M", std::vector<size_t>{3, 3});
    RuntimeTensor<double> N("N", std::vector<size_t>{3, 3});
    M.zero();
    N.zero();
    auto Mt = M.permute_view({1, 0});
    REQUIRE_THROWS_AS(cg::maximum(&Mt, M, N), std::invalid_argument);
}

TEST_CASE("MaxMin - elementwise shape mismatches are refused", "[ComputeGraph][MaxMin]") {
    auto A = create_random_tensor<double>("A", 2, 3);
    auto B = create_random_tensor<double>("B", 3, 2);
    auto C = create_zero_tensor<double>("C", 2, 3);
    REQUIRE_THROWS_AS(cg::maximum(&C, A, B), DimensionError);
    auto D = create_zero_tensor<double>("D", 3, 2);
    REQUIRE_THROWS_AS(cg::minimum(&D, A, A), DimensionError);

    RuntimeTensor<double> R("R", std::vector<size_t>{2, 3, 1});
    RuntimeTensor<double> S("S", std::vector<size_t>{2, 3});
    REQUIRE_THROWS_AS(cg::maximum(&R, S, S), RankError);
}

namespace {
template <typename C, typename A>
concept MaximumAccepts = requires(C *c, A const &a) { cg::maximum(c, a, a); };
template <typename C, typename A>
concept MaxAxesAccepts = requires(C *c, A const &a) { cg::max_axes(c, a, std::vector<size_t>{}); };
template <typename C, typename A>
concept MinAccepts = requires(C *c, A const &a) { cg::min_python(c, a); };
} // namespace

TEST_CASE("MaxMin - complex operands are refused at the API", "[ComputeGraph][MaxMin]") {
    // A complex number has no order, so no complex overload exists: the refusal is a compile-time
    // one, and Python sees it as a call no overload accepts.
    using Z = RuntimeTensor<std::complex<double>>;
    STATIC_REQUIRE_FALSE(MaximumAccepts<Z, Z>);
    STATIC_REQUIRE_FALSE(MaxAxesAccepts<Z, Z>);
    STATIC_REQUIRE_FALSE(MinAccepts<Z, Z>);
    using D = RuntimeTensor<double>;
    STATIC_REQUIRE(MaximumAccepts<D, D>);
    STATIC_REQUIRE(MaxAxesAccepts<D, D>);
    STATIC_REQUIRE(MinAccepts<D, D>);
}

// ── max_axes / min_axes ──────────────────────────────────────────────────────

namespace {

/// The brute-force axis reduction: for each kept index, fold every reduced one in turn.
template <typename T>
std::vector<T> ref_reduce_axes(bool max, Tensor<T, 3> const &A, std::vector<size_t> const &axes) {
    std::array<bool, 3> reduced{false, false, false};
    for (size_t const ax : axes) {
        reduced[ax] = true;
    }
    std::array<size_t, 3> const dims{A.dim(0), A.dim(1), A.dim(2)};
    std::array<size_t, 3>       out_dims{};
    size_t                      out_rank = 0;
    for (size_t a = 0; a < 3; ++a) {
        if (!reduced[a]) {
            out_dims[out_rank++] = dims[a];
        }
    }
    size_t out_size = 1;
    for (size_t a = 0; a < out_rank; ++a) {
        out_size *= out_dims[a];
    }
    std::vector<T>    out(out_size);
    std::vector<bool> seen(out_size, false);
    for (size_t i = 0; i < dims[0]; ++i) {
        for (size_t j = 0; j < dims[1]; ++j) {
            for (size_t k = 0; k < dims[2]; ++k) {
                std::array<size_t, 3> const idx{i, j, k};
                // Column-major linear position in the output, first kept axis fastest.
                size_t pos = 0, scale = 1, slot = 0;
                for (size_t a = 0; a < 3; ++a) {
                    if (!reduced[a]) {
                        pos += idx[a] * scale;
                        scale *= out_dims[slot++];
                    }
                }
                out[pos]  = seen[pos] ? ref_extremum(max, out[pos], A(i, j, k)) : A(i, j, k);
                seen[pos] = true;
            }
        }
    }
    return out;
}

/// The value of @p out at column-major linear position @p pos, through its own strides.
template <typename T>
T at_linear(RuntimeTensor<T> const &out, size_t pos) {
    size_t offset = 0;
    for (size_t a = 0; a < out.rank(); ++a) {
        offset += (pos % out.dim(a)) * out.stride(a);
        pos /= out.dim(a);
    }
    return out.data()[offset];
}

} // namespace

TEMPLATE_TEST_CASE("MaxMin - max_axes and min_axes match the reference for every axis subset", "[ComputeGraph][MaxMin]", float, double) {
    using T             = TestType;
    bool const captured = GENERATE(false, true);
    bool const max      = GENERATE(true, false);
    auto const axes     = GENERATE(std::vector<size_t>{0}, std::vector<size_t>{1}, std::vector<size_t>{2}, std::vector<size_t>{0, 2},
                                   std::vector<size_t>{2, 1}, std::vector<size_t>{0, 1, 2}, std::vector<size_t>{});
    INFO("captured " << captured << " max " << max << " axes " << axes.size());

    // A strided slice of a parent with NaN and the zeros planted, so the reduction walks by stride.
    auto parent     = special_tensor<T>("parent", 5, 4, 3, true);
    parent(2, 0, 1) = std::numeric_limits<T>::quiet_NaN();
    parent(3, 2, 0) = -T{0};
    parent(1, 2, 0) = T{0};
    auto       A    = parent(Range{1, 4}, All, All);
    auto const A0   = Tensor<T, 3>(A);

    std::vector<size_t> out_dims;
    for (size_t a = 0; a < 3; ++a) {
        if (std::find(axes.begin(), axes.end(), a) == axes.end()) {
            out_dims.push_back(A.dim(a));
        }
    }
    RuntimeTensor<T> out("out", out_dims);
    out.zero();

    run(captured, [&] {
        if (max) {
            cg::max_axes(&out, A, axes);
        } else {
            cg::min_axes(&out, A, axes);
        }
    });

    auto const expected = ref_reduce_axes(max, A0, axes);
    REQUIRE(out.size() == expected.size());
    for (size_t pos = 0; pos < expected.size(); ++pos) {
        INFO("output position " << pos);
        REQUIRE(same_value(at_linear(out, pos), expected[pos]));
    }
}

TEST_CASE("MaxMin - an all -inf line reduces to -inf, never to a finite seed", "[ComputeGraph][MaxMin]") {
    double const          inf = std::numeric_limits<double>::infinity();
    RuntimeTensor<double> A("A", std::vector<size_t>{3, 2});
    for (size_t i = 0; i < A.size(); ++i) {
        A.data()[i] = -inf;
    }
    A.data()[3] = 1.0; // second column holds -inf, 1, -inf
    RuntimeTensor<double> out("out", std::vector<size_t>{2});
    cg::max_axes(&out, A, {0});
    REQUIRE(out.data()[0] == -inf);
    REQUIRE(out.data()[1] == 1.0);

    for (size_t i = 0; i < A.size(); ++i) {
        A.data()[i] = inf;
    }
    cg::min_axes(&out, A, {0});
    REQUIRE(out.data()[0] == inf);
}

TEST_CASE("MaxMin - reducing over an empty axis is refused", "[ComputeGraph][MaxMin]") {
    for (bool const max : {true, false}) {
        INFO("max " << max);
        RuntimeTensor<double> A("A", std::vector<size_t>{0, 3});
        RuntimeTensor<double> out("out", std::vector<size_t>{3});
        auto const            call = [&](RuntimeTensor<double> &o, RuntimeTensor<double> const &a, std::vector<size_t> axes) {
            if (max) {
                cg::max_axes(&o, a, std::move(axes));
            } else {
                cg::min_axes(&o, a, std::move(axes));
            }
        };
        REQUIRE_THROWS_AS(call(out, A, {0}), std::invalid_argument);

        cg::Graph graph("empty_reduction");
        {
            cg::CaptureGuard const guard(graph);
            REQUIRE_THROWS_AS(call(out, A, {0}), std::invalid_argument);
        }
        REQUIRE(graph.num_nodes() == 0);

        // An empty KEPT axis is an empty result, not an empty reduction.
        RuntimeTensor<double> B("B", std::vector<size_t>{3, 0});
        RuntimeTensor<double> empty("empty", std::vector<size_t>{0});
        REQUIRE_NOTHROW(call(empty, B, {0}));
    }
}

TEST_CASE("MaxMin - max_axes validates its axes and its destination as sum_axes does", "[ComputeGraph][MaxMin]") {
    RuntimeTensor<double> A("A", std::vector<size_t>{2, 3});
    RuntimeTensor<double> o3("o3", std::vector<size_t>{3});
    RuntimeTensor<double> o2("o2", std::vector<size_t>{2});
    RuntimeTensor<double> o23("o23", std::vector<size_t>{2, 3});
    REQUIRE_THROWS_AS(cg::max_axes(&o3, A, {0, 0}), std::invalid_argument);
    REQUIRE_THROWS_AS(cg::max_axes(&o3, A, {2}), std::out_of_range);
    REQUIRE_THROWS_AS(cg::min_axes(&o23, A, {0}), RankError);
    REQUIRE_THROWS_AS(cg::min_axes(&o2, A, {0}), DimensionError);
}

// ── scalar min ───────────────────────────────────────────────────────────────

TEMPLATE_TEST_CASE("MaxMin - the scalar min and max match a brute-force fold", "[ComputeGraph][MaxMin]", float, double) {
    using T             = TestType;
    bool const captured = GENERATE(false, true);

    auto             parent = special_tensor<T>("parent", 5, 4, 3, false);
    auto             A      = parent(Range{1, 4}, All, All);
    auto const       A0     = Tensor<T, 3>(A);
    RuntimeTensor<T> rmin("rmin", std::vector<size_t>{});
    RuntimeTensor<T> rmax("rmax", std::vector<size_t>{});
    run(captured, [&] {
        cg::min_python(&rmin, A);
        cg::max_python(&rmax, A);
    });

    T lo = A0.data()[0], hi = A0.data()[0];
    for (size_t i = 1; i < A0.size(); ++i) {
        lo = ref_extremum(false, lo, A0.data()[i]);
        hi = ref_extremum(true, hi, A0.data()[i]);
    }
    REQUIRE(rmin.data()[0] == lo);
    REQUIRE(rmax.data()[0] == hi);
}

TEST_CASE("MaxMin - the scalar reductions keep infinities and NaN", "[ComputeGraph][MaxMin]") {
    double const          inf = std::numeric_limits<double>::infinity();
    RuntimeTensor<double> A("A", std::vector<size_t>{2});
    RuntimeTensor<double> r("r", std::vector<size_t>{});

    // A lowest() seed would leave max(-inf, -inf) at -1.8e308.
    A.data()[0] = -inf;
    A.data()[1] = -inf;
    cg::max_python(&r, A);
    REQUIRE(r.data()[0] == -inf);

    A.data()[0] = inf;
    A.data()[1] = inf;
    cg::min_python(&r, A);
    REQUIRE(r.data()[0] == inf);

    A.data()[0] = std::numeric_limits<double>::quiet_NaN();
    A.data()[1] = -inf;
    cg::min_python(&r, A);
    REQUIRE(std::isnan(r.data()[0]));

    RuntimeTensor<double> empty("empty", std::vector<size_t>{0});
    REQUIRE_THROWS_AS(cg::min_python(&r, empty), std::invalid_argument);
}

// ── passes, loops and GraphIR ────────────────────────────────────────────────

TEST_CASE("MaxMin - results hold under the default passes and inside a loop body", "[ComputeGraph][MaxMin]") {
    auto const            A = create_random_tensor<double>("A", 4, 3);
    auto const            B = create_random_tensor<double>("B", 4, 3);
    RuntimeTensor<double> C("C", std::vector<size_t>{4, 3});
    RuntimeTensor<double> R("R", std::vector<size_t>{3});
    RuntimeTensor<double> S("S", std::vector<size_t>{4});

    cg::Graph graph("max_min_loop");
    auto     &body = graph.add_loop("iter", 3, [](size_t iter) -> bool { return iter + 1 < 3; });
    {
        cg::CaptureGuard const guard(body);
        cg::maximum(&C, A, B);
        cg::min_axes(&R, C, {0});
        cg::max_axes(&S, C, {1});
    }
    auto passes = cg::PassManager::create_default();
    graph.apply(passes);
    graph.execute();

    for (size_t j = 0; j < 3; ++j) {
        double lo = std::numeric_limits<double>::infinity();
        for (size_t i = 0; i < 4; ++i) {
            double const c = ref_extremum(true, A(i, j), B(i, j));
            REQUIRE(C.data()[i + 4 * j] == c);
            lo = std::min(lo, c);
        }
        REQUIRE(R.data()[j] == lo);
    }
    for (size_t i = 0; i < 4; ++i) {
        double hi = -std::numeric_limits<double>::infinity();
        for (size_t j = 0; j < 3; ++j) {
            hi = std::max(hi, ref_extremum(true, A(i, j), B(i, j)));
        }
        REQUIRE(S.data()[i] == hi);
    }
}

TEMPLATE_TEST_CASE("MaxMin - every max/min kind round-trips through GraphIR bitwise", "[ComputeGraph][MaxMin][SaveLoad]", float, double) {
    using T = TestType;
    auto A  = create_random_tensor<T>("A", 3, 4);
    auto B  = create_random_tensor<T>("B", 3, 4);
    auto C  = create_zero_tensor<T>("C", 3, 4);
    auto D  = create_zero_tensor<T>("D", 3, 4);
    auto R  = create_zero_tensor<T>("R", 4);
    auto S  = create_zero_tensor<T>("S", 3);

    cg::Graph graph("max_min_ir");
    {
        cg::CaptureGuard const guard(graph);
        cg::maximum(&C, A, B);
        cg::minimum(&D, A, B);
        cg::max_axes(&R, C, {0});
        cg::min_axes(&S, D, {1});
    }
    graph.execute();

    auto text = cg::save_graph_string(graph);
    INFO((text ? std::string{} : text.error().message));
    REQUIRE(text.has_value());
    REQUIRE_THAT(*text, Catch::Matchers::ContainsSubstring("\"ElementwiseMaxMin\""));
    REQUIRE_THAT(*text, Catch::Matchers::ContainsSubstring("\"ReduceAxes\""));

    auto loaded = cg::load_graph_string(*text);
    INFO((loaded ? std::string{} : loaded.error().message));
    REQUIRE(loaded.has_value());

    auto C2 = create_zero_tensor<T>("C2", 3, 4);
    auto D2 = create_zero_tensor<T>("D2", 3, 4);
    auto R2 = create_zero_tensor<T>("R2", 4);
    auto S2 = create_zero_tensor<T>("S2", 3);
    loaded->bind("A", A);
    loaded->bind("B", B);
    loaded->bind("C", C2);
    loaded->bind("D", D2);
    loaded->bind("R", R2);
    loaded->bind("S", S2);
    loaded->execute();

    REQUIRE(std::memcmp(C.data(), C2.data(), C.size() * sizeof(T)) == 0);
    REQUIRE(std::memcmp(D.data(), D2.data(), D.size() * sizeof(T)) == 0);
    REQUIRE(std::memcmp(R.data(), R2.data(), R.size() * sizeof(T)) == 0);
    REQUIRE(std::memcmp(S.data(), S2.data(), S.size() * sizeof(T)) == 0);
}
