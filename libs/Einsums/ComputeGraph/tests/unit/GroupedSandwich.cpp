//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

// The grouped sandwich against a plain-loop oracle that shares no code with
// the kernel: the same dressed accumulation written as four nested loops.
// Tolerances are roundoff-scale rather than zero, because the kernel
// accumulates per q slice and the oracle per scalar product - the documented
// contract is agreement to accumulation roundoff, plus bit-identical replays.

#include <Einsums/ComputeGraph.hpp>
#include <Einsums/Tensor/RuntimeTensor.hpp>
#include <Einsums/Testing/TensorCompare.hpp>

#include <cstddef>
#include <cstring>
#include <limits>
#include <random>
#include <vector>

#include <Einsums/Testing.hpp>

using namespace einsums;
namespace cg = einsums::compute_graph;

namespace {

template <typename T>
RuntimeTensor<T> randt(std::string const &name, std::vector<std::size_t> const &dims, std::mt19937 &gen) {
    RuntimeTensor<T>                       out(name, dims);
    std::uniform_real_distribution<double> dist(-1.0, 1.0);
    T                                     *p = out.data();
    for (std::size_t i = 0; i < out.size(); i++) {
        p[i] = static_cast<T>(dist(gen));
    }
    return out;
}

/// The oracle: C += sum_q (A[q] - P^T M[q]) S (A[q] - P^T M[q])^T, all loops, accumulated in
/// double whatever the element type, so a float run is held to a more accurate reference.
template <typename T>
void oracle(RuntimeTensor<T> &C, RuntimeTensor<T> const &A, RuntimeTensor<T> const &M, RuntimeTensor<T> const &P, RuntimeTensor<T> const &S,
            std::size_t nq, std::size_t nk, std::size_t na) {
    std::vector<double> B(na * na), W(na * na);
    for (std::size_t q = 0; q < nq; q++) {
        for (std::size_t b = 0; b < na; b++) {
            for (std::size_t a = 0; a < na; a++) {
                double v = A.data()[q + nq * (a + na * b)];
                for (std::size_t k = 0; k < nk; k++) {
                    v -= static_cast<double>(P.data()[k + nk * a]) * static_cast<double>(M.data()[q + nq * (k + nk * b)]);
                }
                B[a + na * b] = v;
            }
        }
        for (std::size_t d = 0; d < na; d++) {
            for (std::size_t a = 0; a < na; a++) {
                double v = 0.0;
                for (std::size_t c = 0; c < na; c++) {
                    v += B[a + na * c] * static_cast<double>(S.data()[c + na * d]);
                }
                W[a + na * d] = v;
            }
        }
        for (std::size_t b = 0; b < na; b++) {
            for (std::size_t a = 0; a < na; a++) {
                double v = 0.0;
                for (std::size_t d = 0; d < na; d++) {
                    v += W[a + na * d] * B[b + na * d];
                }
                C.data()[a + na * b] = static_cast<T>(static_cast<double>(C.data()[a + na * b]) + v);
            }
        }
    }
}

/// Accumulation roundoff per unit of work, about 1e-12 in double.
template <typename T>
double roundoff() {
    return 5000.0 * std::numeric_limits<T>::epsilon();
}

using einsums::testing::bytes_of;

} // namespace

// grouped_sandwich requires a real floating-point element type, so complex is not a case.
TEMPLATE_TEST_CASE("GroupedSandwich - eager matches the loop oracle over mixed members", "[ComputeGraph][GroupedSandwich]", float, double) {
    using T = TestType;
    std::mt19937 gen(31);

    // Mixed extents, including an nq large enough to span several q blocks
    // when na is small, an nk = 0 member with the (1, na) placeholder P the
    // DLPNO emission uses, and a zero-extent member.
    struct Shape {
        std::size_t nq, nk, na;
    };
    std::vector<Shape> const shapes = {{40, 4, 6}, {97, 3, 5}, {13, 0, 4}, {0, 2, 3}, {21, 5, 1}};

    std::vector<RuntimeTensor<T>> A, M, P, S, C, expected;
    for (auto const &[nq, nk, na] : shapes) {
        A.push_back(randt<T>("A", {nq, na, na}, gen));
        M.push_back(randt<T>("M", {nq, nk, na}, gen));
        P.push_back(randt<T>("P", {std::max<std::size_t>(nk, 1), na}, gen));
        S.push_back(randt<T>("S", {na, na}, gen));
        C.push_back(randt<T>("C", {na, na}, gen));
        expected.push_back(C.back());
    }
    for (std::size_t i = 0; i < shapes.size(); i++) {
        oracle(expected[i], A[i], M[i], P[i], S[i], shapes[i].nq, shapes[i].nk, shapes[i].na);
    }

    std::vector<RuntimeTensor<T> *>       c_list;
    std::vector<RuntimeTensor<T> const *> a_list, m_list, p_list, s_list;
    for (std::size_t i = 0; i < shapes.size(); i++) {
        c_list.push_back(&C[i]);
        a_list.push_back(&A[i]);
        m_list.push_back(&M[i]);
        p_list.push_back(&P[i]);
        s_list.push_back(&S[i]);
    }
    cg::grouped_sandwich(c_list, a_list, m_list, p_list, s_list);

    for (std::size_t i = 0; i < shapes.size(); i++) {
        auto const [nq, nk, na] = shapes[i];
        double const scale      = 1.0 + static_cast<double>(nq) * static_cast<double>(na);
        for (std::size_t e = 0; e < C[i].size(); e++) {
            REQUIRE_THAT(C[i].data()[e], Catch::Matchers::WithinAbs(expected[i].data()[e], roundoff<T>() * scale));
        }
    }
}

// grouped_sandwich requires a real floating-point element type, so complex is not a case.
TEMPLATE_TEST_CASE("GroupedSandwich - a strided A view matches the oracle on its slice", "[ComputeGraph][GroupedSandwich]", float, double) {
    using T = TestType;
    std::mt19937      gen(7);
    std::size_t const nq = 33, nk = 3, na = 5, pad = 2;

    // A lives inside a padded parent, the way a pair block lives inside a
    // bucketed store: dims (nq, na, na) with strides (1, nq, nq * (na + pad)).
    auto                       parent = randt<T>("parent", {nq, na + pad, na}, gen);
    RuntimeTensorView<T> const a_view(parent, std::vector<std::size_t>{nq, na, na}, std::vector<std::size_t>{1, nq, nq * (na + pad)},
                                      std::vector<std::size_t>{0, 0, 0});

    auto M = randt<T>("M", {nq, nk, na}, gen);
    auto P = randt<T>("P", {nk, na}, gen);
    auto S = randt<T>("S", {na, na}, gen);
    auto C = randt<T>("C", {na, na}, gen);

    // The oracle wants a dense copy of the viewed slice.
    RuntimeTensor<T> a_dense("A dense", {nq, na, na});
    for (std::size_t b = 0; b < na; b++) {
        for (std::size_t a = 0; a < na; a++) {
            for (std::size_t q = 0; q < nq; q++) {
                a_dense.data()[q + nq * (a + na * b)] = parent.data()[q + nq * (a + (na + pad) * b)];
            }
        }
    }
    auto expected = C;
    oracle(expected, a_dense, M, P, S, nq, nk, na);

    std::vector<RuntimeTensor<T> *>           c_list{&C};
    std::vector<RuntimeTensorView<T> const *> a_list{&a_view};
    std::vector<RuntimeTensor<T> const *>     m_list{&M}, p_list{&P}, s_list{&S};
    cg::grouped_sandwich(c_list, a_list, m_list, p_list, s_list);

    for (std::size_t e = 0; e < C.size(); e++) {
        REQUIRE_THAT(C.data()[e], Catch::Matchers::WithinAbs(expected.data()[e], 10.0 * roundoff<T>()));
    }
}

// grouped_sandwich requires a real floating-point element type, so complex is not a case.
TEMPLATE_TEST_CASE("GroupedSandwich - capture replays match eager and are bit-identical", "[ComputeGraph][GroupedSandwich]", float,
                   double) {
    using T = TestType;
    std::mt19937      gen(19);
    std::size_t const nq = 57, nk = 4, na = 6;

    auto A  = randt<T>("A", {nq, na, na}, gen);
    auto M  = randt<T>("M", {nq, nk, na}, gen);
    auto P  = randt<T>("P", {nk, na}, gen);
    auto S  = randt<T>("S", {na, na}, gen);
    auto C0 = randt<T>("C", {na, na}, gen);

    // Eager reference.
    auto eager = C0;
    {
        std::vector<RuntimeTensor<T> *>       c{&eager};
        std::vector<RuntimeTensor<T> const *> a{&A}, m{&M}, p{&P}, s{&S};
        cg::grouped_sandwich(c, a, m, p, s);
    }

    auto      C = C0;
    cg::Graph graph("sandwich");
    {
        cg::CaptureGuard const                guard(graph);
        std::vector<RuntimeTensor<T> *>       c{&C};
        std::vector<RuntimeTensor<T> const *> a{&A}, m{&M}, p{&P}, s{&S};
        cg::grouped_sandwich(c, a, m, p, s);
    }
    REQUIRE(graph.num_nodes() == 1);

    graph.execute();
    REQUIRE(bytes_of(C) == bytes_of(eager));
    auto const first = bytes_of(C);

    // A replay accumulates again; a second pair of replays from the same
    // starting bytes must land on the same bits (deterministic q order).
    graph.execute();
    auto const second = bytes_of(C);
    REQUIRE(second != first);

    std::memcpy(C.data(), C0.data(), C0.size() * sizeof(T));
    graph.execute();
    REQUIRE(bytes_of(C) == first);
    graph.execute();
    REQUIRE(bytes_of(C) == second);
}
