//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

// The grouped gather-rotate against two oracles that share no code with the
// kernel: a plain-loop transcription of its definition, and the emission it
// replaces (a gather into a whole block, then the two contractions). Tolerances
// are roundoff-scale rather than zero, because the kernel blocks its reductions
// as GEMMs and the loops do not - the documented contract is agreement to
// roundoff, plus bit-identical replays.

#include <Einsums/ComputeGraph.hpp>
#include <Einsums/Tensor/RuntimeTensor.hpp>

#include <cstddef>
#include <cstring>
#include <random>
#include <vector>

#include <Einsums/Testing.hpp>

using namespace einsums;
namespace cg = einsums::compute_graph;

namespace {

RuntimeTensor<double> randt(std::string const &name, std::vector<std::size_t> const &dims, std::mt19937 &gen) {
    RuntimeTensor<double>                  out(name, dims);
    std::uniform_real_distribution<double> dist(-1.0, 1.0);
    double                                *p = out.data();
    for (std::size_t i = 0; i < out.size(); i++) {
        p[i] = dist(gen);
    }
    return out;
}

/// The oracle: C[q, a, b] = sum_uv src[qs[q], us[u], us[v]] X[u, a] X[v, b],
/// every index a loop, reading the source through its own strides.
void oracle(RuntimeTensor<double> &C, double const *src, std::size_t sq, std::size_t su, std::size_t sv, std::vector<std::size_t> const &qs,
            std::vector<std::size_t> const &us, RuntimeTensor<double> const &X) {
    std::size_t const nq = qs.size(), nu = us.size(), nt = X.dim(1);
    for (std::size_t q = 0; q < nq; q++) {
        for (std::size_t b = 0; b < nt; b++) {
            for (std::size_t a = 0; a < nt; a++) {
                double acc = 0.0;
                for (std::size_t v = 0; v < nu; v++) {
                    for (std::size_t u = 0; u < nu; u++) {
                        acc += src[qs[q] * sq + us[u] * su + us[v] * sv] * X.data()[u + nu * a] * X.data()[v + nu * b];
                    }
                }
                C.data()[q + nq * (a + nt * b)] = acc;
            }
        }
    }
}

std::vector<unsigned char> bytes_of(RuntimeTensor<double> const &t) {
    std::vector<unsigned char> out(t.size() * sizeof(double));
    std::memcpy(out.data(), t.data(), out.size());
    return out;
}

} // namespace

TEST_CASE("GroupedGatherRotate - eager matches the loop oracle over mixed members", "[ComputeGraph][GroupedGatherRotate]") {
    std::mt19937      gen(31);
    std::size_t const NQ = 61, NU = 12;

    auto src = randt("src", {NQ, NU, NU}, gen);

    // Mixed extents: an nq large enough to span several q tiles, a selection
    // that is one ascending run and one that is neither ordered nor unique, a
    // single-column transform, an empty PAO selection, an empty auxiliary
    // selection, and a member whose transform has no columns at all.
    struct Member {
        std::vector<std::size_t> qs, us;
        std::size_t              nt;
    };
    std::vector<Member> members;
    members.push_back({{}, {}, 3});                                    // nq == 0
    members.push_back({{0, 1, 2, 3, 4, 5, 6, 7, 8, 9}, {2, 3, 4}, 2}); // one ascending run
    members.push_back({{9, 0, 4, 4, 17}, {7, 1, 11, 0}, 3});           // out of order, repeated
    members.push_back({{3, 5}, {}, 2});                                // nu == 0: writes zeros
    members.push_back({{1, 2}, {0, 5}, 0});                            // nt == 0
    {
        // Enough auxiliary functions to make the tiling run several times.
        std::vector<std::size_t> qs(NQ);
        for (std::size_t i = 0; i < NQ; i++) {
            qs[i] = i;
        }
        members.push_back({qs, {0, 2, 4, 6, 8, 10}, 4});
    }
    members.push_back({{6}, {1}, 1}); // everything degenerate but nonzero

    std::vector<RuntimeTensor<double>> X, C, expected;
    for (auto const &m : members) {
        X.push_back(randt("X", {m.us.size(), m.nt}, gen));
        // Deliberately filled with garbage: the operation ASSIGNS, so nothing
        // may survive from before the call.
        C.push_back(randt("C", {m.qs.size(), m.nt, m.nt}, gen));
        expected.push_back(C.back());
    }
    for (std::size_t i = 0; i < members.size(); i++) {
        oracle(expected[i], src.data(), src.stride(0), src.stride(1), src.stride(2), members[i].qs, members[i].us, X[i]);
    }

    std::vector<RuntimeTensor<double> *>       c_list;
    std::vector<RuntimeTensor<double> const *> x_list;
    std::vector<std::vector<std::size_t>>      q_list, u_list;
    for (std::size_t i = 0; i < members.size(); i++) {
        c_list.push_back(&C[i]);
        x_list.push_back(&X[i]);
        q_list.push_back(members[i].qs);
        u_list.push_back(members[i].us);
    }
    cg::grouped_gather_rotate(c_list, src, q_list, u_list, x_list);

    for (std::size_t i = 0; i < members.size(); i++) {
        double const scale = 1.0 + static_cast<double>(members[i].us.size() * members[i].us.size());
        for (std::size_t e = 0; e < C[i].size(); e++) {
            REQUIRE_THAT(C[i].data()[e], Catch::Matchers::WithinAbs(expected[i].data()[e], 1e-13 * scale));
        }
    }
}

TEST_CASE("GroupedGatherRotate - matches the gather-plus-two-contractions emission", "[ComputeGraph][GroupedGatherRotate]") {
    // The reference is the emission this node replaces, run through the same
    // library entry points the DLPNO phase used to call.
    std::mt19937      gen(5);
    std::size_t const NQ = 40, NU = 9;
    std::size_t const nq = 23, nu = 5, nt = 3;

    auto src = randt("src", {NQ, NU, NU}, gen);
    auto X   = randt("X", {nu, nt}, gen);

    std::vector<std::size_t> qs, us{6, 0, 3, 8, 1};
    for (std::size_t i = 0; i < nq; i++) {
        qs.push_back(2 * i % NQ);
    }

    RuntimeTensor<double> uv("uv", {nq, nu, nu});
    RuntimeTensor<double> half("half", {nq, nt, nu});
    RuntimeTensor<double> reference("reference", {nq, nt, nt});
    cg::gather(&uv, src, {qs, us, us});
    cg::einsum("Qav <- Quv ; ua", &half, uv, X);
    cg::einsum("Qab <- Qav ; vb", &reference, half, X);

    auto                                       C = randt("C", {nq, nt, nt}, gen);
    std::vector<RuntimeTensor<double> *>       c_list{&C};
    std::vector<RuntimeTensor<double> const *> x_list{&X};
    cg::grouped_gather_rotate(c_list, src, {qs}, {us}, x_list);

    for (std::size_t e = 0; e < C.size(); e++) {
        REQUIRE_THAT(C.data()[e], Catch::Matchers::WithinAbs(reference.data()[e], 1e-12));
    }
}

TEST_CASE("GroupedGatherRotate - strided source and destination views", "[ComputeGraph][GroupedGatherRotate]") {
    std::mt19937      gen(7);
    std::size_t const NQ = 37, NU = 6, pad = 3;
    std::size_t const nq = 19, nu = 4, nt = 3;

    // The source lives inside a padded parent, the way a block lives inside a
    // bucketed store: dims (NQ, NU, NU), strides (1, NQ, NQ * (NU + pad)).
    auto                            parent = randt("parent", {NQ, NU + pad, NU}, gen);
    RuntimeTensorView<double> const s_view(parent, std::vector<std::size_t>{NQ, NU, NU}, std::vector<std::size_t>{1, NQ, NQ * (NU + pad)},
                                           std::vector<std::size_t>{0, 0, 0});

    auto X = randt("X", {nu, nt}, gen);

    std::vector<std::size_t> qs, us{5, 2, 0, 3};
    for (std::size_t i = 0; i < nq; i++) {
        qs.push_back(i + 4);
    }

    // The destination is a column-slab of a wider store, so its q stride is 1
    // but its outermost stride is not the product of the inner extents.
    auto                      dest_parent = randt("dest parent", {nq, nt, nt + 2}, gen);
    RuntimeTensorView<double> c_view(dest_parent, std::vector<std::size_t>{nq, nt, nt}, std::vector<std::size_t>{1, nq, nq * nt},
                                     std::vector<std::size_t>{0, 0, 0});

    std::vector<double> const outside(dest_parent.data() + nq * nt * nt, dest_parent.data() + dest_parent.size());

    RuntimeTensor<double> expected("expected", {nq, nt, nt});
    oracle(expected, s_view.data(), s_view.stride(0), s_view.stride(1), s_view.stride(2), qs, us, X);

    std::vector<RuntimeTensorView<double> *>   c_list{&c_view};
    std::vector<RuntimeTensor<double> const *> x_list{&X};
    cg::grouped_gather_rotate(c_list, s_view, {qs}, {us}, x_list);

    for (std::size_t b = 0; b < nt; b++) {
        for (std::size_t a = 0; a < nt; a++) {
            for (std::size_t q = 0; q < nq; q++) {
                REQUIRE_THAT(dest_parent.data()[q + nq * (a + nt * b)],
                             Catch::Matchers::WithinAbs(expected.data()[q + nq * (a + nt * b)], 1e-12));
            }
        }
    }
    // The slabs past the viewed one are untouched.
    for (std::size_t e = nq * nt * nt; e < dest_parent.size(); e++) {
        REQUIRE(dest_parent.data()[e] == outside[e - nq * nt * nt]);
    }
}

TEST_CASE("GroupedGatherRotate - capture replays match eager and are bit-identical", "[ComputeGraph][GroupedGatherRotate]") {
    std::mt19937      gen(19);
    std::size_t const NQ = 53, NU = 10;
    std::size_t const nq = 29, nu = 6, nt = 4;

    auto src = randt("src", {NQ, NU, NU}, gen);
    auto X0  = randt("X0", {nu, nt}, gen);
    auto X1  = randt("X1", {nu, nt}, gen);

    std::vector<std::size_t> q0, q1, u0{1, 4, 9, 0, 6, 2}, u1{0, 1, 2, 3, 4, 5};
    for (std::size_t i = 0; i < nq; i++) {
        q0.push_back(i);
        q1.push_back(NQ - 1 - i);
    }

    auto eager0 = randt("eager0", {nq, nt, nt}, gen);
    auto eager1 = randt("eager1", {nq, nt, nt}, gen);
    {
        std::vector<RuntimeTensor<double> *>       c{&eager0, &eager1};
        std::vector<RuntimeTensor<double> const *> x{&X0, &X1};
        cg::grouped_gather_rotate(c, src, {q0, q1}, {u0, u1}, x);
    }

    RuntimeTensor<double> C0("C0", {nq, nt, nt}), C1("C1", {nq, nt, nt});
    cg::Graph             graph("gather_rotate");
    {
        cg::CaptureGuard const                     guard(graph);
        std::vector<RuntimeTensor<double> *>       c{&C0, &C1};
        std::vector<RuntimeTensor<double> const *> x{&X0, &X1};
        cg::grouped_gather_rotate(c, src, {q0, q1}, {u0, u1}, x);
    }
    REQUIRE(graph.num_nodes() == 1);

    graph.execute();
    REQUIRE(bytes_of(C0) == bytes_of(eager0));
    REQUIRE(bytes_of(C1) == bytes_of(eager1));

    // The node assigns, so a second replay from anywhere lands on the same
    // bits: the tiled axis indexes no sum and nothing reads the destination.
    auto const first = bytes_of(C0);
    std::memset(C0.data(), 0x5a, C0.size() * sizeof(double));
    graph.execute();
    REQUIRE(bytes_of(C0) == first);
    graph.execute();
    REQUIRE(bytes_of(C0) == first);
}

TEST_CASE("GroupedGatherRotate - rejects the shapes it cannot compute", "[ComputeGraph][GroupedGatherRotate]") {
    std::mt19937      gen(23);
    std::size_t const NQ = 8, NU = 5;

    auto src = randt("src", {NQ, NU, NU}, gen);
    auto X   = randt("X", {3, 2}, gen);
    auto C   = randt("C", {4, 2, 2}, gen);

    std::vector<RuntimeTensor<double> *>       c_list{&C};
    std::vector<RuntimeTensor<double> const *> x_list{&X};
    std::vector<std::size_t> const             qs{0, 1, 2, 3}, us{0, 1, 4};

    // An empty run is a caller mistake, not an empty loop.
    {
        std::vector<RuntimeTensor<double> *>       none_c;
        std::vector<RuntimeTensor<double> const *> none_x;
        REQUIRE_THROWS_AS(cg::grouped_gather_rotate(none_c, src, {}, {}, none_x), std::invalid_argument);
    }
    // Lists of different lengths.
    REQUIRE_THROWS_AS(cg::grouped_gather_rotate(c_list, src, {qs, qs}, {us}, x_list), std::invalid_argument);
    // An index past the end of the source.
    REQUIRE_THROWS_AS(cg::grouped_gather_rotate(c_list, src, {std::vector<std::size_t>{0, 1, 2, NQ}}, {us}, x_list), std::out_of_range);
    REQUIRE_THROWS_AS(cg::grouped_gather_rotate(c_list, src, {qs}, {std::vector<std::size_t>{0, 1, NU}}, x_list), std::out_of_range);
    // A destination whose extents do not follow from the lists.
    REQUIRE_THROWS_AS(cg::grouped_gather_rotate(c_list, src, {std::vector<std::size_t>{0, 1, 2}}, {us}, x_list), dimension_error);
    // Two members writing one destination would race.
    {
        std::vector<RuntimeTensor<double> *>       both_c{&C, &C};
        std::vector<RuntimeTensor<double> const *> both_x{&X, &X};
        REQUIRE_THROWS_AS(cg::grouped_gather_rotate(both_c, src, {qs, qs}, {us, us}, both_x), std::invalid_argument);
    }
    // A rank-2 source is not a (Q|u v) block.
    {
        auto flat = randt("flat", {NQ, NU}, gen);
        REQUIRE_THROWS_AS(cg::grouped_gather_rotate(c_list, flat, {qs}, {us}, x_list), rank_error);
    }
}
