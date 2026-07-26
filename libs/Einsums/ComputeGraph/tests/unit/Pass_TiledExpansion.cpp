//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

// Coverage: TiledExpansion lowers one opaque tiled-einsum node into per-tile
// DENSE nodes. The bar is that the expanded graph produces bit-for-bit what the
// unexpanded (opaque) path produces, including the semantics that are easy to get
// silently wrong: c_pf applied exactly once, structural zeros emitting nothing,
// and pre-existing output tiles that receive no contribution still being scaled.

#include <Einsums/ComputeGraph.hpp>
#include <Einsums/ComputeGraph/Passes/TiledExpansion.hpp>
#include <Einsums/Tensor/TiledRuntimeTensor.hpp>

#include <cmath>
#include <memory>
#include <vector>

#include <Einsums/Testing.hpp>

using namespace einsums;
namespace cg = einsums::compute_graph;

namespace {

using Grid = std::vector<std::vector<int>>;

/// Build a 2-D tiled tensor over `grid`, populating only the listed tile coords.
TiledRuntimeTensor<double> make_tiled(std::string name, Grid const &grid, std::vector<std::vector<int>> const &coords) {
    TiledRuntimeTensor<double> t(std::move(name), grid);
    for (auto const &c : coords) {
        t.tile(c).materialize();
    }
    return t;
}

std::vector<std::vector<int>> full_coords(Grid const &grid) {
    std::vector<std::vector<int>> out;
    for (int i = 0; i < static_cast<int>(grid[0].size()); ++i) {
        for (int j = 0; j < static_cast<int>(grid[1].size()); ++j) {
            out.push_back({i, j});
        }
    }
    return out;
}

/// Fill every populated tile from a global (row, col) function.
template <typename F>
void fill(TiledRuntimeTensor<double> &T, F &&f) {
    auto const &off = T.tile_offsets();
    auto const &sz  = T.tile_sizes();
    for (int ti = 0; ti < static_cast<int>(sz[0].size()); ++ti) {
        for (int tj = 0; tj < static_cast<int>(sz[1].size()); ++tj) {
            if (!T.has_tile({ti, tj}))
                continue;
            auto &tile = T.tile({ti, tj});
            tile.materialize();
            for (int lr = 0; lr < sz[0][ti]; ++lr) {
                for (int lc = 0; lc < sz[1][tj]; ++lc) {
                    tile(std::vector<size_t>{static_cast<size_t>(lr), static_cast<size_t>(lc)}) = f(off[0][ti] + lr, off[1][tj] + lc);
                }
            }
        }
    }
}

/// Gather a dense R x C picture, absent tiles reading as zero.
std::vector<double> gather(TiledRuntimeTensor<double> const &T, size_t R, size_t C) {
    std::vector<double> M(R * C, 0.0);
    auto const         &off = T.tile_offsets();
    auto const         &sz  = T.tile_sizes();
    for (int ti = 0; ti < static_cast<int>(sz[0].size()); ++ti) {
        for (int tj = 0; tj < static_cast<int>(sz[1].size()); ++tj) {
            if (!T.has_tile({ti, tj}))
                continue;
            auto const &tile = T.tile({ti, tj});
            for (int lr = 0; lr < sz[0][ti]; ++lr) {
                for (int lc = 0; lc < sz[1][tj]; ++lc) {
                    M[(off[0][ti] + lr) * C + off[1][tj] + lc] =
                        tile(std::vector<size_t>{static_cast<size_t>(lr), static_cast<size_t>(lc)});
                }
            }
        }
    }
    return M;
}

/// Every tile coordinate of a grid, in a deterministic order (the unordered_map
/// behind tiles() must never drive anything two instances have to agree on).
std::vector<std::vector<int>> enumerate_coords(Grid const &grid) {
    std::vector<std::vector<int>> out{{}};
    for (auto const &axis : grid) {
        std::vector<std::vector<int>> next;
        for (auto const &prefix : out) {
            for (int t = 0; t < static_cast<int>(axis.size()); ++t) {
                auto c = prefix;
                c.push_back(t);
                next.push_back(std::move(c));
            }
        }
        out = std::move(next);
    }
    return out;
}

/// Build a tiled tensor of any rank, populating the listed coords.
TiledRuntimeTensor<double> make_ndim(std::string name, Grid const &grid, std::vector<std::vector<int>> const &coords) {
    TiledRuntimeTensor<double> t(std::move(name), grid);
    for (auto const &c : coords) {
        t.tile(c).materialize();
    }
    return t;
}

/// Fill deterministically from the tile coordinate and the element's offset, so
/// two independently built copies hold identical data.
void fill_det(TiledRuntimeTensor<double> &T, double salt) {
    for (auto const &coord : enumerate_coords(T.tile_sizes())) {
        if (!T.has_tile(coord)) {
            continue;
        }
        auto &tile = T.tile(coord);
        tile.materialize();
        double seed = salt;
        for (int c : coord) {
            seed = seed * 3.0 + static_cast<double>(c);
        }
        for (size_t i = 0; i < tile.size(); ++i) {
            tile.data()[i] = std::sin(seed + static_cast<double>(i) * 0.25);
        }
    }
}

/// Same populated tile set and identical contents. Compares every element of
/// every tile, which is stronger than gathering to a dense picture and works at
/// any rank.
void require_tiles_match(TiledRuntimeTensor<double> const &got, TiledRuntimeTensor<double> const &want) {
    // Guard against a vacuous pass: if the contraction produced no output tiles at
    // all, every comparison below is trivially satisfied and proves nothing.
    REQUIRE(want.num_filled_tiles() > 0);
    REQUIRE(got.num_filled_tiles() == want.num_filled_tiles());
    for (auto const &coord : enumerate_coords(want.tile_sizes())) {
        REQUIRE(got.has_tile(coord) == want.has_tile(coord));
        if (!want.has_tile(coord)) {
            continue;
        }
        auto const &g = got.tile(coord);
        auto const &w = want.tile(coord);
        REQUIRE(g.size() == w.size());
        for (size_t i = 0; i < w.size(); ++i) {
            REQUIRE(std::abs(g.data()[i] - w.data()[i]) < 1e-11);
        }
    }
}

size_t nodes_of_kind(cg::Graph const &g, cg::OpKind k) {
    return static_cast<size_t>(std::ranges::count_if(g.nodes(), [k](cg::Node const &n) { return n.kind == k; }));
}

} // namespace

TEST_CASE("TiledExpansion - expanded result matches the opaque path", "[ComputeGraph][Passes][Tiled]") {
    Grid const gA{{2, 3}, {4, 5}};
    Grid const gB{{4, 5}, {3, 4}};
    Grid const gC{{2, 3}, {3, 4}};

    auto aval = [](int r, int c) { return 1.0 + static_cast<double>(r * 9 + c); };
    auto bval = [](int r, int c) { return 2.0 - static_cast<double>(r * 7 + c); };

    // Reference: the existing opaque tiled path.
    auto A_ref = make_tiled("A", gA, full_coords(gA));
    auto B_ref = make_tiled("B", gB, full_coords(gB));
    auto C_ref = make_tiled("C", gC, {});
    fill(A_ref, aval);
    fill(B_ref, bval);
    {
        cg::Graph              gref("tiled_ref");
        cg::CaptureGuard const guard(gref);
        cg::einsum("ij <- ik ; kj", &C_ref, A_ref, B_ref);
        const_cast<cg::Graph &>(gref).execute();
    }

    // Expanded.
    auto A = make_tiled("A2", gA, full_coords(gA));
    auto B = make_tiled("B2", gB, full_coords(gB));
    auto C = make_tiled("C2", gC, {});
    fill(A, aval);
    fill(B, bval);

    cg::Graph graph("tiled_expand");
    {
        cg::CaptureGuard const guard(graph);
        cg::einsum("ij <- ik ; kj", &C, A, B);
    }
    REQUIRE(nodes_of_kind(graph, cg::OpKind::Custom) == 1);
    REQUIRE(nodes_of_kind(graph, cg::OpKind::Einsum) == 0);

    cg::PassManager pm;
    auto            pass = std::make_shared<cg::passes::TiledExpansion>();
    pm.add(pass);
    REQUIRE(graph.apply(pm));

    CHECK(pass->num_expanded() == 1);
    CHECK(pass->num_declined() == 0);
    // 2x2 output tiles, each accumulating over a 2-wide link: 8 contractions.
    CHECK(pass->num_tile_nodes() == 8);
    CHECK(nodes_of_kind(graph, cg::OpKind::Custom) == 0);
    CHECK(nodes_of_kind(graph, cg::OpKind::Einsum) == 8);

    graph.execute();

    auto const got  = gather(C, 5, 7);
    auto const want = gather(C_ref, 5, 7);
    REQUIRE(got.size() == want.size());
    for (size_t i = 0; i < got.size(); ++i) {
        REQUIRE(std::abs(got[i] - want[i]) < 1e-11);
    }
}

TEST_CASE("TiledExpansion - absent operand tiles emit no node", "[ComputeGraph][Passes][Tiled]") {
    // Block-diagonal A: only (0,0) and (1,1) present, so half the tile pairs are
    // structural zeros and must produce no work at all.
    Grid const g{{2, 3}, {2, 3}};

    auto A_ref = make_tiled("A", g, {{0, 0}, {1, 1}});
    auto B_ref = make_tiled("B", g, full_coords(g));
    auto C_ref = make_tiled("C", g, {});
    fill(A_ref, [](int r, int c) { return 1.0 + r + c; });
    fill(B_ref, [](int r, int c) { return 2.0 + r - c; });
    {
        cg::Graph              gref("sparse_ref");
        cg::CaptureGuard const guard(gref);
        cg::einsum("ij <- ik ; kj", &C_ref, A_ref, B_ref);
        const_cast<cg::Graph &>(gref).execute();
    }

    auto A = make_tiled("A2", g, {{0, 0}, {1, 1}});
    auto B = make_tiled("B2", g, full_coords(g));
    auto C = make_tiled("C2", g, {});
    fill(A, [](int r, int c) { return 1.0 + r + c; });
    fill(B, [](int r, int c) { return 2.0 + r - c; });

    cg::Graph graph("sparse_expand");
    {
        cg::CaptureGuard const guard(graph);
        cg::einsum("ij <- ik ; kj", &C, A, B);
    }
    cg::PassManager pm;
    auto            pass = std::make_shared<cg::passes::TiledExpansion>();
    pm.add(pass);
    REQUIRE(graph.apply(pm));

    // Only k == i contributes, so 2 output tiles x 1 contribution each = 4 nodes
    // for a 2x2 output grid (i,j) with a single surviving k per i.
    CHECK(pass->num_tile_nodes() == 4);
    CHECK(nodes_of_kind(graph, cg::OpKind::Einsum) == 4);

    graph.execute();
    auto const got  = gather(C, 5, 5);
    auto const want = gather(C_ref, 5, 5);
    for (size_t i = 0; i < got.size(); ++i) {
        REQUIRE(std::abs(got[i] - want[i]) < 1e-11);
    }
}

TEST_CASE("TiledExpansion - a pre-existing output tile with no contribution is still scaled", "[ComputeGraph][Passes][Tiled]") {
    // The runtime scales EVERY pre-existing output tile by c_pf up front, even one
    // that no (A,B) pair reaches. Expanding only the contributing pairs would leave
    // it untouched - a silent numerical difference from the opaque path.
    //
    // A holds only tile (0,0), so only output tiles (0,*) receive contributions,
    // while C's pre-existing tile (1,1) receives none.
    Grid const g{{2, 2}, {2, 2}};

    auto A_ref = make_tiled("Ar", g, {{0, 0}});
    auto B_ref = make_tiled("Br", g, {{0, 0}, {0, 1}});
    auto C_ref = make_tiled("Cr", g, {{0, 0}, {1, 1}});
    fill(A_ref, [](int, int) { return 1.0; });
    fill(B_ref, [](int, int) { return 1.0; });
    fill(C_ref, [](int, int) { return 7.0; }); // pre-existing content
    {
        cg::Graph              gref("scale_ref");
        cg::CaptureGuard const guard(gref);
        cg::einsum("ij <- ik ; kj", 0.5, &C_ref, 1.0, A_ref, B_ref);
        const_cast<cg::Graph &>(gref).execute();
    }

    auto A = make_tiled("Ae", g, {{0, 0}});
    auto B = make_tiled("Be", g, {{0, 0}, {0, 1}});
    auto C = make_tiled("Ce", g, {{0, 0}, {1, 1}});
    fill(A, [](int, int) { return 1.0; });
    fill(B, [](int, int) { return 1.0; });
    fill(C, [](int, int) { return 7.0; });

    cg::Graph graph("scale_expand");
    {
        cg::CaptureGuard const guard(graph);
        cg::einsum("ij <- ik ; kj", 0.5, &C, 1.0, A, B);
    }
    cg::PassManager pm;
    auto            pass = std::make_shared<cg::passes::TiledExpansion>();
    pm.add(pass);
    REQUIRE(graph.apply(pm));
    // At least one Scale node, for the untouched pre-existing tile (1,1).
    CHECK(nodes_of_kind(graph, cg::OpKind::Scale) >= 1);

    graph.execute();
    auto const got  = gather(C, 4, 4);
    auto const want = gather(C_ref, 4, 4);
    for (size_t i = 0; i < got.size(); ++i) {
        REQUIRE(std::abs(got[i] - want[i]) < 1e-11);
    }
    // Concretely: the untouched tile held 7.0 and must now hold 3.5.
    CHECK(std::abs(got[2 * 4 + 2] - 3.5) < 1e-11);
}

TEST_CASE("TiledExpansion - declines over the node budget and leaves the graph alone", "[ComputeGraph][Passes][Tiled]") {
    Grid const g{{2, 2, 2}, {2, 2, 2}};
    auto       A = make_tiled("A", g, full_coords(g));
    auto       B = make_tiled("B", g, full_coords(g));
    auto       C = make_tiled("C", g, {});
    fill(A, [](int r, int c) { return 1.0 + r + c; });
    fill(B, [](int r, int c) { return 1.0 - r + c; });

    cg::Graph graph("budget");
    {
        cg::CaptureGuard const guard(graph);
        cg::einsum("ij <- ik ; kj", &C, A, B);
    }

    cg::PassManager pm;
    auto            pass = std::make_shared<cg::passes::TiledExpansion>(/*max_nodes=*/4); // 3x3x3 = 27 combinations
    pm.add(pass);
    CHECK_FALSE(graph.apply(pm));

    CHECK(pass->num_expanded() == 0);
    CHECK(pass->num_declined() == 1);
    // The opaque node survives, so the graph still computes the right answer.
    CHECK(nodes_of_kind(graph, cg::OpKind::Custom) == 1);

    graph.execute();
    auto const got = gather(C, 6, 6);
    CHECK(std::abs(got[0]) > 0.0);
}

// ── Higher rank ───────────────────────────────────────────────────────────
// The enumeration is rank-general by construction (it walks the unique-index
// grid exactly as detail::tiled_runtime_einsum does), so these are coverage
// rather than a separate code path. Both compare against the opaque path
// element-by-element rather than a hand-derived answer.

TEST_CASE("TiledExpansion - rank-3 contraction matches the opaque path", "[ComputeGraph][Passes][Tiled]") {
    // C[i,j,k] = sum_l A[i,j,l] B[l,k]; the contracted l partition must align.
    Grid const gA{{2, 3}, {2}, {3, 4}};
    Grid const gB{{3, 4}, {2, 3}};
    Grid const gC{{2, 3}, {2}, {2, 3}};

    auto A_ref = make_ndim("A", gA, enumerate_coords(gA));
    auto B_ref = make_ndim("B", gB, enumerate_coords(gB));
    auto C_ref = make_ndim("C", gC, {});
    fill_det(A_ref, 1.0);
    fill_det(B_ref, 2.0);
    {
        cg::Graph              gref("r3_ref");
        cg::CaptureGuard const guard(gref);
        cg::einsum("ijk <- ijl ; lk", &C_ref, A_ref, B_ref);
        const_cast<cg::Graph &>(gref).execute();
    }

    auto A = make_ndim("A2", gA, enumerate_coords(gA));
    auto B = make_ndim("B2", gB, enumerate_coords(gB));
    auto C = make_ndim("C2", gC, {});
    fill_det(A, 1.0);
    fill_det(B, 2.0);

    cg::Graph graph("r3_expand");
    {
        cg::CaptureGuard const guard(graph);
        cg::einsum("ijk <- ijl ; lk", &C, A, B);
    }
    cg::PassManager pm;
    auto            pass = std::make_shared<cg::passes::TiledExpansion>();
    pm.add(pass);
    REQUIRE(graph.apply(pm));
    CHECK(pass->num_expanded() == 1);
    CHECK(pass->num_declined() == 0);
    CHECK(nodes_of_kind(graph, cg::OpKind::Custom) == 0);
    CHECK(nodes_of_kind(graph, cg::OpKind::Einsum) > 0);

    graph.execute();
    require_tiles_match(C, C_ref);
}

TEST_CASE("TiledExpansion - CCSD-shaped rank-4 contraction matches the opaque path", "[ComputeGraph][Passes][Tiled]") {
    // C[i,j,a,b] = sum_{c,d} A[i,j,c,d] B[c,d,a,b] -- the particle-ladder shape,
    // with two contracted indices rather than one.
    std::vector<int> const ip{2, 3}, jp{2}, ap{3}, bp{2, 2}, cp{2, 3}, dp{3};
    Grid const             gA{ip, jp, cp, dp};
    Grid const             gB{cp, dp, ap, bp};
    Grid const             gC{ip, jp, ap, bp};

    auto A_ref = make_ndim("A", gA, enumerate_coords(gA));
    auto B_ref = make_ndim("B", gB, enumerate_coords(gB));
    auto C_ref = make_ndim("C", gC, {});
    fill_det(A_ref, 0.5);
    fill_det(B_ref, 1.5);
    {
        cg::Graph              gref("r4_ref");
        cg::CaptureGuard const guard(gref);
        cg::einsum("ijab <- ijcd ; cdab", &C_ref, A_ref, B_ref);
        const_cast<cg::Graph &>(gref).execute();
    }

    auto A = make_ndim("A2", gA, enumerate_coords(gA));
    auto B = make_ndim("B2", gB, enumerate_coords(gB));
    auto C = make_ndim("C2", gC, {});
    fill_det(A, 0.5);
    fill_det(B, 1.5);

    cg::Graph graph("r4_expand");
    {
        cg::CaptureGuard const guard(graph);
        cg::einsum("ijab <- ijcd ; cdab", &C, A, B);
    }
    cg::PassManager pm;
    auto            pass = std::make_shared<cg::passes::TiledExpansion>();
    pm.add(pass);
    REQUIRE(graph.apply(pm));
    CHECK(pass->num_expanded() == 1);
    CHECK(nodes_of_kind(graph, cg::OpKind::Custom) == 0);

    graph.execute();
    require_tiles_match(C, C_ref);
}

TEST_CASE("TiledExpansion - sparse rank-3 operands still match", "[ComputeGraph][Passes][Tiled]") {
    // Drop half of A's tiles so a large share of the rank-3 combinations are
    // structural zeros; the expanded graph must agree with the opaque path on
    // both the surviving values AND which output tiles come into existence.
    Grid const gA{{2, 3}, {2}, {3, 4}};
    Grid const gB{{3, 4}, {2, 3}};
    Grid const gC{{2, 3}, {2}, {2, 3}};

    std::vector<std::vector<int>> sparse_a;
    for (auto const &c : enumerate_coords(gA)) {
        if ((c[0] + c[2]) % 2 == 0) {
            sparse_a.push_back(c);
        }
    }
    REQUIRE(sparse_a.size() < enumerate_coords(gA).size());

    auto A_ref = make_ndim("A", gA, sparse_a);
    auto B_ref = make_ndim("B", gB, enumerate_coords(gB));
    auto C_ref = make_ndim("C", gC, {});
    fill_det(A_ref, 3.0);
    fill_det(B_ref, 4.0);
    {
        cg::Graph              gref("r3s_ref");
        cg::CaptureGuard const guard(gref);
        cg::einsum("ijk <- ijl ; lk", &C_ref, A_ref, B_ref);
        const_cast<cg::Graph &>(gref).execute();
    }

    auto A = make_ndim("A2", gA, sparse_a);
    auto B = make_ndim("B2", gB, enumerate_coords(gB));
    auto C = make_ndim("C2", gC, {});
    fill_det(A, 3.0);
    fill_det(B, 4.0);

    cg::Graph graph("r3s_expand");
    {
        cg::CaptureGuard const guard(graph);
        cg::einsum("ijk <- ijl ; lk", &C, A, B);
    }
    cg::PassManager pm;
    auto            pass = std::make_shared<cg::passes::TiledExpansion>();
    pm.add(pass);
    REQUIRE(graph.apply(pm));

    graph.execute();
    require_tiles_match(C, C_ref);
}

TEST_CASE("TiledExpansion - tiled scale expands to one dense scale per tile", "[ComputeGraph][Passes][Tiled]") {
    Grid const                          g{{2, 3}, {4, 5}};
    std::vector<std::vector<int>> const populated{{0, 0}, {1, 1}, {0, 1}};

    // Reference: the opaque per-tile path.
    auto A_ref = make_tiled("A", g, populated);
    fill_det(A_ref, 1.5);
    {
        cg::Graph              gref("scale_ref");
        cg::CaptureGuard const guard(gref);
        cg::scale(-2.25, &A_ref);
        const_cast<cg::Graph &>(gref).execute();
    }

    auto A = make_tiled("A2", g, populated);
    fill_det(A, 1.5);

    cg::Graph graph("scale_expand");
    {
        cg::CaptureGuard const guard(graph);
        cg::scale(-2.25, &A);
    }
    REQUIRE(nodes_of_kind(graph, cg::OpKind::Custom) == 1);

    cg::PassManager pm;
    auto            pass = std::make_shared<cg::passes::TiledExpansion>();
    pm.add(pass);
    REQUIRE(graph.apply(pm));

    CHECK(pass->num_expanded() == 1);
    CHECK(pass->num_tile_nodes() == populated.size());
    CHECK(nodes_of_kind(graph, cg::OpKind::Custom) == 0);
    CHECK(nodes_of_kind(graph, cg::OpKind::Scale) == populated.size());

    graph.execute();
    require_tiles_match(A, A_ref);
}

TEST_CASE("TiledExpansion - tiled axpy expands to one dense axpy per stored X tile", "[ComputeGraph][Passes][Tiled]") {
    Grid const g{{2, 3}, {4, 5}};
    // (0,1) is in X but not Y, so Y must infer-and-create it; (1,0) is in Y but
    // not X, so it must be left exactly as it was.
    std::vector<std::vector<int>> const x_tiles{{0, 0}, {0, 1}};
    std::vector<std::vector<int>> const y_tiles{{0, 0}, {1, 0}};

    auto X_ref = make_tiled("X", g, x_tiles);
    auto Y_ref = make_tiled("Y", g, y_tiles);
    fill_det(X_ref, 0.5);
    fill_det(Y_ref, 2.5);
    {
        cg::Graph              gref("axpy_ref");
        cg::CaptureGuard const guard(gref);
        cg::axpy(1.75, X_ref, &Y_ref);
        const_cast<cg::Graph &>(gref).execute();
    }

    auto X = make_tiled("X2", g, x_tiles);
    auto Y = make_tiled("Y2", g, y_tiles);
    fill_det(X, 0.5);
    fill_det(Y, 2.5);

    cg::Graph graph("axpy_expand");
    {
        cg::CaptureGuard const guard(graph);
        cg::axpy(1.75, X, &Y);
    }
    REQUIRE(nodes_of_kind(graph, cg::OpKind::Custom) == 1);

    cg::PassManager pm;
    auto            pass = std::make_shared<cg::passes::TiledExpansion>();
    pm.add(pass);
    REQUIRE(graph.apply(pm));

    CHECK(pass->num_expanded() == 1);
    CHECK(pass->num_tile_nodes() == x_tiles.size());
    CHECK(nodes_of_kind(graph, cg::OpKind::Axpy) == x_tiles.size());

    // Every emitted node must declare its read of the destination, or the
    // accumulation is invisible to the scheduler.
    for (auto const &nd : graph.nodes()) {
        if (nd.kind != cg::OpKind::Axpy) {
            continue;
        }
        REQUIRE(nd.outputs.size() == 1);
        REQUIRE(std::ranges::find(nd.inputs, nd.outputs[0]) != nd.inputs.end());
    }

    graph.execute();
    require_tiles_match(Y, Y_ref);
}

TEST_CASE("TiledExpansion - declines a tiled scale whose tensor another node writes", "[ComputeGraph][Passes][Tiled]") {
    // The runtime scales whichever tiles exist when it runs. Here the einsum
    // creates C's tiles, so freezing C's tile set at pass time would scale the
    // wrong set. The scale must stay opaque, and the answer must be unchanged.
    Grid const gA{{2, 3}, {4, 5}};
    Grid const gB{{4, 5}, {3, 4}};
    Grid const gC{{2, 3}, {3, 4}};

    auto A_ref = make_tiled("A", gA, full_coords(gA));
    auto B_ref = make_tiled("B", gB, full_coords(gB));
    auto C_ref = make_tiled("C", gC, {});
    fill_det(A_ref, 1.0);
    fill_det(B_ref, 2.0);
    {
        cg::Graph              gref("scale_after_ref");
        cg::CaptureGuard const guard(gref);
        cg::einsum("ij <- ik ; kj", &C_ref, A_ref, B_ref);
        cg::scale(3.0, &C_ref);
        const_cast<cg::Graph &>(gref).execute();
    }

    auto A = make_tiled("A2", gA, full_coords(gA));
    auto B = make_tiled("B2", gB, full_coords(gB));
    auto C = make_tiled("C2", gC, {});
    fill_det(A, 1.0);
    fill_det(B, 2.0);

    cg::Graph graph("scale_after_expand");
    {
        cg::CaptureGuard const guard(graph);
        cg::einsum("ij <- ik ; kj", &C, A, B);
        cg::scale(3.0, &C);
    }

    cg::PassManager pm;
    auto            pass = std::make_shared<cg::passes::TiledExpansion>();
    pm.add(pass);
    CHECK_FALSE(graph.apply(pm));

    // The scale declines because the einsum may add tiles to C. That leaves a node
    // still naming C's whole-tensor id, which in turn strands the einsum: expanding
    // it would move every write of C down to per-tile ids and drop the edge the
    // scale depends on. Both decline and the graph is untouched.
    //
    // Expanding BOTH would be valid and is the better answer, but it needs the
    // scale to enumerate the tiles the einsum is going to create, which is not
    // known until emission. Left as future work; declining is the honest outcome.
    CHECK(pass->num_expanded() == 0);
    CHECK(pass->num_declined() == 2);
    CHECK(nodes_of_kind(graph, cg::OpKind::Custom) == 2);

    graph.execute();
    require_tiles_match(C, C_ref);
}

TEST_CASE("TiledExpansion - declines when an unexpandable node shares the tiled tensor", "[ComputeGraph][Passes][Tiled]") {
    // A tiled element_transform has no descriptor and can never expand, so it goes
    // on naming C's whole-tensor id. Expanding the einsum would replace every write
    // of C with writes to per-tile ids, leaving that reader with no writer at all
    // and free to be scheduled before the tiles are filled.
    Grid const gA{{2, 3}, {4, 5}};
    Grid const gB{{4, 5}, {3, 4}};
    Grid const gC{{2, 3}, {3, 4}};

    auto square = [](double x) { return x * x; };

    auto A_ref = make_tiled("A", gA, full_coords(gA));
    auto B_ref = make_tiled("B", gB, full_coords(gB));
    auto C_ref = make_tiled("C", gC, {});
    fill_det(A_ref, 1.0);
    fill_det(B_ref, 2.0);
    {
        cg::Graph              gref("strand_ref");
        cg::CaptureGuard const guard(gref);
        cg::einsum("ij <- ik ; kj", &C_ref, A_ref, B_ref);
        cg::element_transform(&C_ref, square);
        const_cast<cg::Graph &>(gref).execute();
    }

    auto A = make_tiled("A2", gA, full_coords(gA));
    auto B = make_tiled("B2", gB, full_coords(gB));
    auto C = make_tiled("C2", gC, {});
    fill_det(A, 1.0);
    fill_det(B, 2.0);

    cg::Graph graph("strand_expand");
    {
        cg::CaptureGuard const guard(graph);
        cg::einsum("ij <- ik ; kj", &C, A, B);
        cg::element_transform(&C, square);
    }

    cg::PassManager pm;
    auto            pass = std::make_shared<cg::passes::TiledExpansion>();
    pm.add(pass);
    CHECK_FALSE(graph.apply(pm));

    CHECK(pass->num_expanded() == 0);
    CHECK(pass->num_declined() == 1);

    graph.execute();
    require_tiles_match(C, C_ref);
}

TEST_CASE("TiledExpansion - declines a tiled axpy over mismatched tile grids", "[ComputeGraph][Passes][Tiled]") {
    // Same 5 x 9 shape, different partitions. The runtime throws; declining keeps
    // the opaque node so it still throws rather than expanding to something that
    // silently pairs up mismatched tiles.
    auto X = make_tiled("X", Grid{{2, 3}, {4, 5}}, full_coords(Grid{{2, 3}, {4, 5}}));
    auto Y = make_tiled("Y", Grid{{5}, {9}}, full_coords(Grid{{5}, {9}}));

    cg::Graph graph("axpy_mismatch");
    {
        cg::CaptureGuard const guard(graph);
        cg::axpy(1.0, X, &Y);
    }

    cg::PassManager pm;
    auto            pass = std::make_shared<cg::passes::TiledExpansion>();
    pm.add(pass);
    graph.apply(pm);

    CHECK(pass->num_expanded() == 0);
    CHECK(pass->num_declined() == 1);
    CHECK(nodes_of_kind(graph, cg::OpKind::Custom) == 1);
    REQUIRE_THROWS(graph.execute());
}
