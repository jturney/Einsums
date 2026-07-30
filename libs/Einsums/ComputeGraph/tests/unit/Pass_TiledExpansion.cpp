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

#include <fmt/format.h>

#include <cmath>
#include <memory>
#include <tuple>
#include <vector>

#include <Einsums/Testing.hpp>

using namespace einsums;
namespace cg = einsums::compute_graph;

/// Densification (small tiles lowered to gather + one dense einsum + scatter) is a
/// separate lowering with its own cases at the end of this file. Everything above
/// is testing the PER-TILE lowering -- node counts, screening, batching -- on tiles
/// deliberately too small to be worth dispatching, which is exactly what the
/// densify gate exists to catch. So these cases pin it off rather than fight it.
constexpr auto kPerTile = cg::passes::Densify::Never;
/// Likewise for the elementwise ops: these tiles are far too small for a dispatch
/// to pay for itself, so the fusion gate would collapse every scale, axpy and
/// divide group into one node. The per-tile cases below pin it off; the fusion
/// cases at the end of this file turn it back on.
constexpr auto kNoFuse = cg::passes::FuseTiles::Never;

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
    auto            pass = std::make_shared<cg::passes::TiledExpansion>(4096, -1.0, kPerTile, kNoFuse);
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
    auto            pass = std::make_shared<cg::passes::TiledExpansion>(4096, -1.0, kPerTile, kNoFuse);
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
    auto            pass = std::make_shared<cg::passes::TiledExpansion>(4096, -1.0, kPerTile, kNoFuse);
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
    auto pass = std::make_shared<cg::passes::TiledExpansion>(/*max_nodes=*/4, -1.0, kPerTile, kNoFuse); // 3x3x3 = 27 combinations
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
    auto            pass = std::make_shared<cg::passes::TiledExpansion>(4096, -1.0, kPerTile, kNoFuse);
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
    auto            pass = std::make_shared<cg::passes::TiledExpansion>(4096, -1.0, kPerTile, kNoFuse);
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
    auto            pass = std::make_shared<cg::passes::TiledExpansion>(4096, -1.0, kPerTile, kNoFuse);
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
    auto            pass = std::make_shared<cg::passes::TiledExpansion>(4096, -1.0, kPerTile, kNoFuse);
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
    auto            pass = std::make_shared<cg::passes::TiledExpansion>(4096, -1.0, kPerTile, kNoFuse);
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

TEST_CASE("TiledExpansion - expands a tiled tensor that is produced then scaled", "[ComputeGraph][Passes][Tiled]") {
    // C is created by the einsum and then scaled. The scale cannot read C's tile
    // set off the object at pass time, because those tiles do not exist yet; it
    // reads the set the einsum is predicted to write. Both expand.
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
    auto            pass = std::make_shared<cg::passes::TiledExpansion>(4096, -1.0, kPerTile, kNoFuse);
    pm.add(pass);
    REQUIRE(graph.apply(pm));

    CHECK(pass->num_expanded() == 2);
    CHECK(pass->num_declined() == 0);
    CHECK(nodes_of_kind(graph, cg::OpKind::Custom) == 0);
    // One scale per C tile, and the contraction did not degenerate into scales.
    CHECK(nodes_of_kind(graph, cg::OpKind::Scale) == C_ref.num_filled_tiles());
    CHECK(nodes_of_kind(graph, cg::OpKind::Einsum) > 0);

    graph.execute();
    require_tiles_match(C, C_ref);
}

TEST_CASE("TiledExpansion - expands a chain where one contraction feeds the next", "[ComputeGraph][Passes][Tiled]") {
    // D = (A*B)*E. C is both an output and an input, so the second contraction has
    // to take C's sparsity from what the first is predicted to write.
    Grid const gA{{2, 3}, {4, 5}};
    Grid const gB{{4, 5}, {3, 4}};
    Grid const gC{{2, 3}, {3, 4}};
    Grid const gE{{3, 4}, {2, 2}};
    Grid const gD{{2, 3}, {2, 2}};

    auto build = [&](char const *tag, TiledRuntimeTensor<double> &C_out, TiledRuntimeTensor<double> &D_out) {
        auto A = make_tiled(std::string("A") + tag, gA, full_coords(gA));
        auto B = make_tiled(std::string("B") + tag, gB, full_coords(gB));
        auto E = make_tiled(std::string("E") + tag, gE, full_coords(gE));
        fill_det(A, 1.0);
        fill_det(B, 2.0);
        fill_det(E, 3.0);
        return std::make_tuple(std::move(A), std::move(B), std::move(E), &C_out, &D_out);
    };

    auto C_ref = make_tiled("C", gC, {});
    auto D_ref = make_tiled("D", gD, {});
    {
        auto [A, B, E, Cp, Dp] = build("r", C_ref, D_ref);
        cg::Graph              gref("chain_ref");
        cg::CaptureGuard const guard(gref);
        cg::einsum("ij <- ik ; kj", Cp, A, B);
        cg::einsum("il <- ij ; jl", Dp, *Cp, E);
        const_cast<cg::Graph &>(gref).execute();
    }

    auto C = make_tiled("C2", gC, {});
    auto D = make_tiled("D2", gD, {});
    auto A = make_tiled("A2", gA, full_coords(gA));
    auto B = make_tiled("B2", gB, full_coords(gB));
    auto E = make_tiled("E2", gE, full_coords(gE));
    fill_det(A, 1.0);
    fill_det(B, 2.0);
    fill_det(E, 3.0);

    cg::Graph graph("chain_expand");
    {
        cg::CaptureGuard const guard(graph);
        cg::einsum("ij <- ik ; kj", &C, A, B);
        cg::einsum("il <- ij ; jl", &D, C, E);
    }

    cg::PassManager pm;
    auto            pass = std::make_shared<cg::passes::TiledExpansion>(4096, -1.0, kPerTile, kNoFuse);
    pm.add(pass);
    REQUIRE(graph.apply(pm));

    CHECK(pass->num_expanded() == 2);
    CHECK(pass->num_declined() == 0);
    CHECK(nodes_of_kind(graph, cg::OpKind::Custom) == 0);

    graph.execute();
    require_tiles_match(C, C_ref);
    require_tiles_match(D, D_ref);
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
    auto            pass = std::make_shared<cg::passes::TiledExpansion>(4096, -1.0, kPerTile, kNoFuse);
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
    auto            pass = std::make_shared<cg::passes::TiledExpansion>(4096, -1.0, kPerTile, kNoFuse);
    pm.add(pass);
    graph.apply(pm);

    CHECK(pass->num_expanded() == 0);
    CHECK(pass->num_declined() == 1);
    CHECK(nodes_of_kind(graph, cg::OpKind::Custom) == 1);
    REQUIRE_THROWS(graph.execute());
}

TEST_CASE("TiledExpansion - the emitted tile GEMMs are batchable", "[ComputeGraph][Passes][Tiled]") {
    // The measurable payoff of expanding. A uniform tile grid makes every tile
    // GEMM the same shape, and the ones writing distinct C tiles are independent,
    // so GEMMBatching can collapse them into gemm_batch calls. If this ever stops
    // firing, expansion is paying node-count cost for nothing.
    Grid const g{{2, 2}, {2, 2}}; // 4 x 4, every tile 2 x 2

    auto A_ref = make_tiled("A", g, full_coords(g));
    auto B_ref = make_tiled("B", g, full_coords(g));
    auto C_ref = make_tiled("C", g, {});
    fill_det(A_ref, 1.0);
    fill_det(B_ref, 2.0);
    {
        cg::Graph              gref("batch_ref");
        cg::CaptureGuard const guard(gref);
        cg::einsum("ij <- ik ; kj", &C_ref, A_ref, B_ref);
        const_cast<cg::Graph &>(gref).execute();
    }

    auto A = make_tiled("A2", g, full_coords(g));
    auto B = make_tiled("B2", g, full_coords(g));
    auto C = make_tiled("C2", g, {});
    fill_det(A, 1.0);
    fill_det(B, 2.0);

    cg::Graph graph("batch_expand");
    {
        cg::CaptureGuard const guard(graph);
        cg::einsum("ij <- ik ; kj", &C, A, B);
    }

    cg::PassManager pm;
    auto            expand = std::make_shared<cg::passes::TiledExpansion>(4096, -1.0, kPerTile, kNoFuse);
    auto            batch  = std::make_shared<cg::passes::GEMMBatching>(cg::CostModel::detect_default());
    pm.add(expand);
    pm.add(batch);
    REQUIRE(graph.apply(pm));

    CHECK(expand->num_expanded() == 1);
    // Two batches: the first write to each of the 4 output tiles (c_pf=0), then the
    // second accumulation into each (c_pf=1). All 8 tile GEMMs are absorbed.
    CHECK(batch->num_batches() == 2);
    CHECK(batch->total_batched() == 8);
    CHECK(nodes_of_kind(graph, cg::OpKind::BatchedGemm) == 2);
    CHECK(nodes_of_kind(graph, cg::OpKind::Einsum) == 0);

    graph.execute();
    require_tiles_match(C, C_ref);
}

TEST_CASE("TiledExpansion - the default pipeline lowers a tiled contraction", "[ComputeGraph][Passes][Tiled]") {
    // End to end through populate_default: nothing opaque survives and the answer is
    // unchanged. These tiles are 2x2, so the shared cost model prices four 2x2 GEMMs
    // above one gather/contract/scatter and the default pipeline densifies. Batching
    // of the per-tile form is covered by the cases that pin densification off.
    Grid const g{{2, 2}, {2, 2}};

    auto A_ref = make_tiled("A", g, full_coords(g));
    auto B_ref = make_tiled("B", g, full_coords(g));
    auto C_ref = make_tiled("C", g, {});
    fill_det(A_ref, 1.0);
    fill_det(B_ref, 2.0);
    {
        cg::Graph              gref("pipeline_ref");
        cg::CaptureGuard const guard(gref);
        cg::einsum("ij <- ik ; kj", &C_ref, A_ref, B_ref);
        const_cast<cg::Graph &>(gref).execute();
    }

    auto A = make_tiled("A2", g, full_coords(g));
    auto B = make_tiled("B2", g, full_coords(g));
    auto C = make_tiled("C2", g, {});
    fill_det(A, 1.0);
    fill_det(B, 2.0);

    cg::Graph graph("pipeline_expand");
    {
        cg::CaptureGuard const guard(graph);
        cg::einsum("ij <- ik ; kj", &C, A, B);
    }
    REQUIRE(nodes_of_kind(graph, cg::OpKind::Custom) == 1);

    auto pm = cg::PassManager::create_default();
    REQUIRE(graph.apply(pm));

    CHECK(nodes_of_kind(graph, cg::OpKind::Custom) == 0);
    CHECK(nodes_of_kind(graph, cg::OpKind::Einsum) == 1);
    CHECK(nodes_of_kind(graph, cg::OpKind::TileGather) == 2);
    CHECK(nodes_of_kind(graph, cg::OpKind::TileScatter) == 1);

    graph.execute();
    require_tiles_match(C, C_ref);
}

namespace {

/// Zero one stored tile in place, leaving it present but numerically zero -- the
/// symmetry-forbidden block case: stored because the grid is uniform, exactly zero
/// because the irrep product does not contain the totally symmetric rep.
void zero_tile(TiledRuntimeTensor<double> &T, std::vector<int> const &coord) {
    auto &tile = T.tile(coord);
    tile.materialize();
    for (size_t i = 0; i < tile.size(); ++i) {
        tile.data()[i] = 0.0;
    }
}

/// Scale one stored tile down to a small but nonzero magnitude.
void shrink_tile(TiledRuntimeTensor<double> &T, std::vector<int> const &coord, double factor) {
    auto &tile = T.tile(coord);
    tile.materialize();
    for (size_t i = 0; i < tile.size(); ++i) {
        tile.data()[i] *= factor;
    }
}

} // namespace

TEST_CASE("TiledExpansion - a stored-but-zero operand tile is screened out", "[ComputeGraph][Passes][Tiled]") {
    Grid const g{{2, 2}, {2, 2}};

    auto build = [&](std::string tag, TiledRuntimeTensor<double> & /*C*/) {
        auto A = make_tiled("A" + tag, g, full_coords(g));
        auto B = make_tiled("B" + tag, g, full_coords(g));
        fill_det(A, 1.0);
        fill_det(B, 2.0);
        zero_tile(A, {0, 0}); // present, exactly zero
        return std::make_pair(std::move(A), std::move(B));
    };

    // Reference: no screening, so every present tile pair gets a node.
    auto C_ref    = make_tiled("C", g, {});
    auto [Ar, Br] = build("r", C_ref);
    {
        cg::Graph              gref("screen_ref");
        cg::CaptureGuard const guard(gref);
        cg::einsum("ij <- ik ; kj", &C_ref, Ar, Br);
        const_cast<cg::Graph &>(gref).execute();
    }

    auto C      = make_tiled("C2", g, {});
    auto [A, B] = build("2", C);
    cg::Graph graph("screen_expand");
    {
        cg::CaptureGuard const guard(graph);
        cg::einsum("ij <- ik ; kj", &C, A, B);
    }

    cg::PassManager pm;
    // Tolerance 0: exact. Only tiles that are identically zero are pruned, so the
    // numbers must match the unscreened reference exactly.
    auto pass = std::make_shared<cg::passes::TiledExpansion>(4096, 0.0, kPerTile, kNoFuse);
    pm.add(pass);
    REQUIRE(graph.apply(pm));

    // A(0,0) pairs with B(0,j) for j in {0,1}: two contributions pruned.
    CHECK(pass->num_screened() == 2);
    CHECK(pass->num_tile_nodes() == 6);

    graph.execute();
    // Compare gathered values, not tile sets: screening is allowed to leave an
    // output tile uncreated, which is the whole point.
    CHECK(gather(C, 4, 4) == gather(C_ref, 4, 4));
}

TEST_CASE("TiledExpansion - screening is off unless a tolerance is given", "[ComputeGraph][Passes][Tiled]") {
    Grid const g{{2, 2}, {2, 2}};
    auto       A = make_tiled("A", g, full_coords(g));
    auto       B = make_tiled("B", g, full_coords(g));
    auto       C = make_tiled("C", g, {});
    fill_det(A, 1.0);
    fill_det(B, 2.0);
    zero_tile(A, {0, 0});

    cg::Graph graph("screen_default_off");
    {
        cg::CaptureGuard const guard(graph);
        cg::einsum("ij <- ik ; kj", &C, A, B);
    }

    cg::PassManager pm;
    auto            pass = std::make_shared<cg::passes::TiledExpansion>(4096, -1.0, kPerTile, kNoFuse); // default ctor
    pm.add(pass);
    REQUIRE(graph.apply(pm));

    // No tile inspected, no contribution dropped: the default must not silently
    // change the emitted node set, since screening is a numerical decision.
    CHECK(pass->num_screened() == 0);
    CHECK(pass->num_tile_nodes() == 8);
}

TEST_CASE("TiledExpansion - a positive tolerance screens small tiles approximately", "[ComputeGraph][Passes][Tiled]") {
    Grid const g{{2, 2}, {2, 2}};
    auto       A = make_tiled("A", g, full_coords(g));
    auto       B = make_tiled("B", g, full_coords(g));
    auto       C = make_tiled("C", g, {});
    fill_det(A, 1.0);
    fill_det(B, 2.0);
    shrink_tile(A, {1, 1}, 1e-14); // small but not zero

    cg::Graph graph("screen_threshold");
    {
        cg::CaptureGuard const guard(graph);
        cg::einsum("ij <- ik ; kj", &C, A, B);
    }

    cg::PassManager pm;
    auto            pass = std::make_shared<cg::passes::TiledExpansion>(4096, 1e-10, kPerTile, kNoFuse);
    pm.add(pass);
    REQUIRE(graph.apply(pm));

    CHECK(pass->num_screened() == 2); // A(1,1) x B(1,j), j in {0,1}
    CHECK(pass->num_tile_nodes() == 6);
}

TEST_CASE("TiledExpansion - a produced operand is never screened on value", "[ComputeGraph][Passes][Tiled]") {
    // C is written by the first contraction, so at pass time its tiles hold
    // whatever was in them beforehand -- here, deliberately zeros. Screening on
    // that would prune the second contraction's work against values the graph has
    // not computed yet.
    Grid const g{{2, 2}, {2, 2}};
    auto       A = make_tiled("A", g, full_coords(g));
    auto       B = make_tiled("B", g, full_coords(g));
    auto       E = make_tiled("E", g, full_coords(g));
    auto       C = make_tiled("C", g, full_coords(g)); // pre-created AND zero
    auto       D = make_tiled("D", g, {});
    fill_det(A, 1.0);
    fill_det(B, 2.0);
    fill_det(E, 3.0);
    for (auto const &co : full_coords(g)) {
        zero_tile(C, co);
    }

    auto C_ref = make_tiled("Cr", g, full_coords(g));
    auto D_ref = make_tiled("Dr", g, {});
    auto A_ref = make_tiled("Ar", g, full_coords(g));
    auto B_ref = make_tiled("Br", g, full_coords(g));
    auto E_ref = make_tiled("Er", g, full_coords(g));
    fill_det(A_ref, 1.0);
    fill_det(B_ref, 2.0);
    fill_det(E_ref, 3.0);
    for (auto const &co : full_coords(g)) {
        zero_tile(C_ref, co);
    }
    {
        cg::Graph              gref("produced_screen_ref");
        cg::CaptureGuard const guard(gref);
        cg::einsum("ij <- ik ; kj", &C_ref, A_ref, B_ref);
        cg::einsum("il <- ij ; jl", &D_ref, C_ref, E_ref);
        const_cast<cg::Graph &>(gref).execute();
    }

    cg::Graph graph("produced_screen");
    {
        cg::CaptureGuard const guard(graph);
        cg::einsum("ij <- ik ; kj", &C, A, B);
        cg::einsum("il <- ij ; jl", &D, C, E);
    }

    cg::PassManager pm;
    auto            pass = std::make_shared<cg::passes::TiledExpansion>(4096, 0.0, kPerTile, kNoFuse);
    pm.add(pass);
    REQUIRE(graph.apply(pm));

    CHECK(pass->num_expanded() == 2);
    CHECK(pass->num_screened() == 0); // C is written here, so it is never measured

    graph.execute();
    CHECK(gather(D, 4, 4) == gather(D_ref, 4, 4));
}

TEST_CASE("TiledExpansion - screened sparsity propagates to the next contraction", "[ComputeGraph][Passes][Tiled]") {
    // Every contribution to one C tile screens out, so that tile is never created.
    // The second contraction then sees it as ABSENT and skips its work too, without
    // being told anything: the propagation falls out of the predicted tile sets.
    Grid const g{{2, 2}, {2, 2}};
    auto       A = make_tiled("A", g, full_coords(g));
    auto       B = make_tiled("B", g, full_coords(g));
    auto       E = make_tiled("E", g, full_coords(g));
    auto       C = make_tiled("C", g, {}); // empty: infer-and-create
    auto       D = make_tiled("D", g, {});
    fill_det(A, 1.0);
    fill_det(B, 2.0);
    fill_det(E, 3.0);
    // Zero the whole first row of A, so C(0,*) receives nothing at all.
    zero_tile(A, {0, 0});
    zero_tile(A, {0, 1});

    cg::Graph graph("propagate");
    {
        cg::CaptureGuard const guard(graph);
        cg::einsum("ij <- ik ; kj", &C, A, B);
        cg::einsum("il <- ij ; jl", &D, C, E);
    }

    cg::PassManager pm;
    auto            pass = std::make_shared<cg::passes::TiledExpansion>(4096, 0.0, kPerTile, kNoFuse);
    pm.add(pass);
    REQUIRE(graph.apply(pm));

    graph.execute();

    // C's first tile row was never created.
    CHECK_FALSE(C.has_tile({0, 0}));
    CHECK_FALSE(C.has_tile({0, 1}));
    CHECK(C.has_tile({1, 0}));

    // And D's first tile row is zero as a result, whether or not it was created.
    auto const d = gather(D, 4, 4);
    for (size_t r = 0; r < 2; ++r) {
        for (size_t c = 0; c < 4; ++c) {
            CHECK(d[r * 4 + c] == 0.0);
        }
    }
    // The rest is real work, not an accidentally empty graph.
    CHECK(std::ranges::any_of(d, [](double v) { return v != 0.0; }));
}

TEST_CASE("TiledExpansion - tiled direct_division expands per tile", "[ComputeGraph][Passes][Tiled]") {
    // The amplitude update. It has to expand, not merely run: an unexpandable
    // node sharing the amplitudes would strand them and decline the whole graph.
    Grid const                          g{{2, 2}, {2, 2}};
    std::vector<std::vector<int>> const num{{0, 0}, {1, 1}};
    std::vector<std::vector<int>> const den = full_coords(g);
    std::vector<std::vector<int>> const dst = full_coords(g);

    auto A_ref = make_tiled("A", g, num);
    auto B_ref = make_tiled("B", g, den);
    auto C_ref = make_tiled("C", g, dst);
    fill_det(A_ref, 1.0);
    fill_det(B_ref, 7.0); // fill_det uses sin(); shift below keeps it away from 0
    fill_det(C_ref, 2.0);
    for (auto const &co : den) {
        auto &t = B_ref.tile(co);
        for (size_t i = 0; i < t.size(); ++i) {
            t.data()[i] += 3.0;
        }
    }
    {
        cg::Graph              gref("divide_ref");
        cg::CaptureGuard const guard(gref);
        cg::direct_division(1.5, A_ref, B_ref, 0.25, &C_ref);
        const_cast<cg::Graph &>(gref).execute();
    }

    auto A = make_tiled("A2", g, num);
    auto B = make_tiled("B2", g, den);
    auto C = make_tiled("C2", g, dst);
    fill_det(A, 1.0);
    fill_det(B, 7.0);
    fill_det(C, 2.0);
    for (auto const &co : den) {
        auto &t = B.tile(co);
        for (size_t i = 0; i < t.size(); ++i) {
            t.data()[i] += 3.0;
        }
    }

    cg::Graph graph("divide_expand");
    {
        cg::CaptureGuard const guard(graph);
        cg::direct_division(1.5, A, B, 0.25, &C);
    }
    REQUIRE(nodes_of_kind(graph, cg::OpKind::Custom) == 0); // records as DirectDivision

    cg::PassManager pm;
    auto            pass = std::make_shared<cg::passes::TiledExpansion>(4096, -1.0, kPerTile, kNoFuse);
    pm.add(pass);
    REQUIRE(graph.apply(pm));

    CHECK(pass->num_expanded() == 1);
    // Two numerator tiles divide; the other two destination tiles are only scaled
    // by beta, which is the same leftover rule the contraction path uses.
    CHECK(nodes_of_kind(graph, cg::OpKind::DirectDivision) == num.size());
    CHECK(nodes_of_kind(graph, cg::OpKind::Scale) == dst.size() - num.size());

    graph.execute();
    require_tiles_match(C, C_ref);
}

TEST_CASE("TiledExpansion - a batched group writes every element of every destination", "[ComputeGraph][Passes][Tiled]") {
    // Non-uniform occupied blocks with uniform virtual blocks: the (1,1),(2,2),(3,3)
    // output tiles are all 2x4, so GEMMBatching groups them while (0,0) at 4x4
    // stays alone. The destinations are poisoned first, so any element the batch
    // fails to write stays poisoned instead of depending on what the heap held.
    Grid const                          ov{{4, 2, 2, 2}, {4, 4, 4, 4}};
    Grid const                          vv{{4, 4, 4, 4}, {4, 4, 4, 4}};
    std::vector<std::vector<int>> const diag{{0, 0}, {1, 1}, {2, 2}, {3, 3}};

    auto poison = [](TiledRuntimeTensor<double> &T) {
        for (auto const &kv : T.tiles()) {
            auto &t = const_cast<RuntimeTensor<double> &>(kv.second);
            t.materialize();
            for (size_t i = 0; i < t.size(); ++i) {
                t.data()[i] = 1.0e30;
            }
        }
    };

    auto A     = make_tiled("A", ov, diag);
    auto F     = make_tiled("F", vv, diag);
    auto C_ref = make_tiled("Cr", ov, diag);
    auto C     = make_tiled("C", ov, diag);
    fill_det(A, 1.0);
    fill_det(F, 2.0);

    poison(C_ref);
    {
        cg::Graph              gref("batch_ref");
        cg::CaptureGuard const guard(gref);
        cg::einsum("ia <- ie ; ae", &C_ref, A, F);
        const_cast<cg::Graph &>(gref).execute();
    }

    poison(C);
    cg::Graph graph("batch_poison");
    {
        cg::CaptureGuard const guard(graph);
        cg::einsum("ia <- ie ; ae", &C, A, F);
    }

    cg::PassManager pm;
    auto            expand = std::make_shared<cg::passes::TiledExpansion>(4096, -1.0, kPerTile, kNoFuse);
    auto            batch  = std::make_shared<cg::passes::GEMMBatching>(cg::CostModel::detect_default());
    pm.add(expand);
    pm.add(batch);
    REQUIRE(graph.apply(pm));
    REQUIRE(batch->num_batches() > 0);

    graph.execute();
    require_tiles_match(C, C_ref);
}

TEST_CASE("TiledExpansion - an overwrite batch followed by an accumulate batch", "[ComputeGraph][Passes][Tiled]") {
    // The T1 residual shape: one contraction overwrites the destination tiles
    // (c_pf=0) and a second accumulates into them (c_pf=1). Both expand into
    // groups of three 2x4 tiles, so both batch, and the second reads what the
    // first wrote. Destinations are poisoned so an unwritten element is visible.
    Grid const                          ov{{4, 2, 2, 2}, {4, 4, 4, 4}};
    Grid const                          vv{{4, 4, 4, 4}, {4, 4, 4, 4}};
    Grid const                          oo{{4, 2, 2, 2}, {4, 2, 2, 2}};
    std::vector<std::vector<int>> const diag{{0, 0}, {1, 1}, {2, 2}, {3, 3}};

    auto poison = [](TiledRuntimeTensor<double> &T) {
        for (auto const &kv : T.tiles()) {
            auto &t = const_cast<RuntimeTensor<double> &>(kv.second);
            t.materialize();
            for (size_t i = 0; i < t.size(); ++i) {
                t.data()[i] = 1.0e30;
            }
        }
    };

    auto A     = make_tiled("A", ov, diag);
    auto F     = make_tiled("F", vv, diag);
    auto M     = make_tiled("M", oo, diag);
    auto C_ref = make_tiled("Cr", ov, diag);
    auto C     = make_tiled("C", ov, diag);
    fill_det(A, 1.0);
    fill_det(F, 2.0);
    fill_det(M, 3.0);

    poison(C_ref);
    {
        cg::Graph              gref("chain_ref");
        cg::CaptureGuard const guard(gref);
        cg::einsum("ia <- ie ; ae", 0.0, &C_ref, 1.0, A, F);
        cg::einsum("ia <- ma ; mi", 1.0, &C_ref, -1.0, A, M);
        const_cast<cg::Graph &>(gref).execute();
    }

    poison(C);
    cg::Graph graph("chain_poison");
    {
        cg::CaptureGuard const guard(graph);
        cg::einsum("ia <- ie ; ae", 0.0, &C, 1.0, A, F);
        cg::einsum("ia <- ma ; mi", 1.0, &C, -1.0, A, M);
    }

    cg::PassManager pm;
    auto            expand = std::make_shared<cg::passes::TiledExpansion>(4096, -1.0, kPerTile, kNoFuse);
    auto            batch  = std::make_shared<cg::passes::GEMMBatching>(cg::CostModel::detect_default());
    pm.add(expand);
    pm.add(batch);
    REQUIRE(graph.apply(pm));
    REQUIRE(batch->num_batches() > 0);
    graph.execute();
    require_tiles_match(C, C_ref);
}

// ── Densified lowering ──────────────────────────────────────────────────────
// Below min_tile_flops the pass stops emitting one node per tile pair and instead
// gathers, contracts once, and scatters back. The cases above pin that off; these
// exercise it.

TEST_CASE("TiledExpansion - densified result matches the per-tile lowering", "[ComputeGraph][Passes][Tiled]") {
    Grid const gA{{2, 3}, {4, 5}};
    Grid const gB{{4, 5}, {3, 4}};
    Grid const gC{{2, 3}, {3, 4}};

    auto aval = [](int r, int c) { return 1.0 + static_cast<double>(r * 9 + c); };
    auto bval = [](int r, int c) { return 2.0 - static_cast<double>(r * 7 + c); };

    auto run = [&](cg::passes::Densify densify, size_t &n_einsum, size_t &n_gather) {
        auto A = make_tiled("A", gA, full_coords(gA));
        auto B = make_tiled("B", gB, full_coords(gB));
        auto C = make_tiled("C", gC, {});
        fill(A, aval);
        fill(B, bval);
        cg::Graph graph("g");
        {
            cg::CaptureGuard const guard(graph);
            cg::einsum("ij <- ik ; kj", &C, A, B);
        }
        cg::PassManager pm;
        pm.add(std::make_shared<cg::passes::TiledExpansion>(4096, -1.0, densify, kNoFuse));
        REQUIRE(graph.apply(pm));
        n_einsum = nodes_of_kind(graph, cg::OpKind::Einsum);
        n_gather = nodes_of_kind(graph, cg::OpKind::TileGather);
        graph.execute();
        return gather(C, 5, 7);
    };

    size_t     per_tile_einsums = 0, per_tile_gathers = 0, dense_einsums = 0, dense_gathers = 0;
    auto const reference = run(kPerTile, per_tile_einsums, per_tile_gathers);
    auto const densified = run(cg::passes::Densify::Always, dense_einsums, dense_gathers);

    // The lowering really did change: many einsums and no marshalling, versus one
    // einsum fed by two gathers.
    CHECK(per_tile_einsums > 1);
    CHECK(per_tile_gathers == 0);
    CHECK(dense_einsums == 1);
    CHECK(dense_gathers == 2);

    // Same answer. Not bitwise: the dense contraction sums a link index in one
    // sweep where the per-tile form sums it in per-tile chunks, so the two differ
    // by floating-point association only.
    REQUIRE(reference.size() == densified.size());
    for (size_t i = 0; i < reference.size(); ++i) {
        REQUIRE_THAT(densified[i], Catch::Matchers::WithinRel(reference[i], 1e-12));
    }
}

TEST_CASE("TiledExpansion - densifying creates no tile the per-tile path would not", "[ComputeGraph][Passes][Tiled]") {
    // A sparse operand pair: only the diagonal blocks are stored, so only the
    // diagonal output blocks may be written. Densifying computes the full product
    // internally, and the scatter must still discard everything else rather than
    // materializing output tiles the per-tile lowering never creates.
    Grid const gA{{2, 3}, {4, 5}};
    Grid const gB{{4, 5}, {3, 4}};
    Grid const gC{{2, 3}, {3, 4}};

    auto A = make_tiled("A", gA, {{0, 0}, {1, 1}});
    auto B = make_tiled("B", gB, {{0, 0}, {1, 1}});
    auto C = make_tiled("C", gC, {});
    fill(A, [](int r, int c) { return 1.0 + static_cast<double>(r + c); });
    fill(B, [](int r, int c) { return 2.0 - static_cast<double>(r + c); });

    cg::Graph graph("sparse");
    {
        cg::CaptureGuard const guard(graph);
        cg::einsum("ij <- ik ; kj", &C, A, B);
    }
    cg::PassManager pm;
    auto            pass = std::make_shared<cg::passes::TiledExpansion>(4096, -1.0, cg::passes::Densify::Always, kNoFuse);
    pm.add(pass);
    REQUIRE(graph.apply(pm));
    REQUIRE(pass->num_densified() == 1);
    graph.execute();

    CHECK(C.has_tile({0, 0}));
    CHECK(C.has_tile({1, 1}));
    CHECK_FALSE(C.has_tile({0, 1}));
    CHECK_FALSE(C.has_tile({1, 0}));
}

TEST_CASE("TiledExpansion - Auto keeps the per-tile lowering for large sparse tiles", "[ComputeGraph][Passes][Tiled]") {
    // The other side of the cost comparison. Only the diagonal blocks are stored, so
    // densifying would contract the off-diagonal blocks the tiled form skips
    // entirely -- and at this tile size the per-tile contractions are already large
    // enough to run near peak, so there is no throughput left to buy with that extra
    // arithmetic. Auto must decline. Small tiles reverse this (see above), which is
    // precisely why the decision is a time comparison and not a size threshold.
    Grid const g{{64, 64}, {64, 64}};

    auto A = make_tiled("A", g, {{0, 0}, {1, 1}});
    auto B = make_tiled("B", g, {{0, 0}, {1, 1}});
    auto C = make_tiled("C", g, {});
    fill(A, [](int r, int c) { return 1.0 + static_cast<double>((r + c) % 7); });
    fill(B, [](int r, int c) { return 2.0 - static_cast<double>((r * 3 + c) % 5); });

    cg::Graph graph("large_sparse");
    {
        cg::CaptureGuard const guard(graph);
        cg::einsum("ij <- ik ; kj", &C, A, B);
    }
    cg::PassManager pm;
    auto            pass = std::make_shared<cg::passes::TiledExpansion>(4096, -1.0, cg::passes::Densify::Auto, kNoFuse);
    pm.add(pass);
    REQUIRE(graph.apply(pm));
    CHECK(pass->num_densified() == 0);
    CHECK(pass->num_expanded() == 1);
    CHECK(nodes_of_kind(graph, cg::OpKind::TileGather) == 0);
    CHECK(nodes_of_kind(graph, cg::OpKind::Einsum) > 1);
}

// ── Fused elementwise lowering ──────────────────────────────────────────────
// The tiled scale, axpy and divide collapse into a single TileElementwise node
// when their tiles are too small to be worth dispatching one at a time. The cases
// above pin that off with kNoFuse; these exercise it.

TEST_CASE("TiledExpansion - a fused scale matches the per-tile lowering", "[ComputeGraph][Passes][Tiled]") {
    Grid const                          g{{2, 3}, {4, 5}};
    std::vector<std::vector<int>> const populated{{0, 0}, {0, 1}, {1, 1}};

    auto run = [&](cg::passes::FuseTiles fuse, TiledRuntimeTensor<double> &A) {
        cg::Graph graph("scale_fuse");
        {
            cg::CaptureGuard const guard(graph);
            cg::scale(-2.25, &A);
        }
        cg::PassManager pm;
        auto            pass = std::make_shared<cg::passes::TiledExpansion>(4096, -1.0, kPerTile, fuse);
        pm.add(pass);
        REQUIRE(graph.apply(pm));
        graph.execute();
        return std::make_tuple(pass->num_fused(), nodes_of_kind(graph, cg::OpKind::Scale),
                               nodes_of_kind(graph, cg::OpKind::TileElementwise));
    };

    auto A_ref = make_tiled("A", g, populated);
    fill_det(A_ref, 1.5);
    auto const [ref_fused, ref_scales, ref_tile_ew] = run(kNoFuse, A_ref);
    CHECK(ref_fused == 0);
    CHECK(ref_scales == populated.size());
    CHECK(ref_tile_ew == 0);

    auto A = make_tiled("A2", g, populated);
    fill_det(A, 1.5);
    auto const [fused, scales, tile_ew] = run(cg::passes::FuseTiles::Always, A);
    CHECK(fused == 1);
    CHECK(scales == 0);
    CHECK(tile_ew == 1);

    // Same operation on the same tiles, so this is bit-for-bit, not merely close.
    require_tiles_match(A, A_ref);
}

TEST_CASE("TiledExpansion - a fused axpy matches the per-tile lowering and declares its destination", "[ComputeGraph][Passes][Tiled]") {
    Grid const g{{2, 3}, {4, 5}};
    // (0,1) is in X but not Y, so Y must infer-and-create it; (1,0) is in Y but not
    // X, so the fused node must leave it exactly as it was.
    std::vector<std::vector<int>> const x_tiles{{0, 0}, {0, 1}};
    std::vector<std::vector<int>> const y_tiles{{0, 0}, {1, 0}};

    auto run = [&](cg::passes::FuseTiles fuse, TiledRuntimeTensor<double> &X, TiledRuntimeTensor<double> &Y, cg::Graph &graph) {
        {
            cg::CaptureGuard const guard(graph);
            cg::axpy(1.75, X, &Y);
        }
        cg::PassManager pm;
        auto            pass = std::make_shared<cg::passes::TiledExpansion>(4096, -1.0, kPerTile, fuse);
        pm.add(pass);
        REQUIRE(graph.apply(pm));
        graph.execute();
        return pass->num_fused();
    };

    auto      X_ref = make_tiled("X", g, x_tiles);
    auto      Y_ref = make_tiled("Y", g, y_tiles);
    cg::Graph gref("axpy_per_tile");
    fill_det(X_ref, 0.5);
    fill_det(Y_ref, 2.5);
    CHECK(run(kNoFuse, X_ref, Y_ref, gref) == 0);

    auto      X = make_tiled("X2", g, x_tiles);
    auto      Y = make_tiled("Y2", g, y_tiles);
    cg::Graph graph("axpy_fused");
    fill_det(X, 0.5);
    fill_det(Y, 2.5);
    CHECK(run(cg::passes::FuseTiles::Always, X, Y, graph) == 1);
    CHECK(nodes_of_kind(graph, cg::OpKind::Axpy) == 0);
    REQUIRE(nodes_of_kind(graph, cg::OpKind::TileElementwise) == 1);

    // The accumulation reads every tile it writes. Dropping that from the fused
    // node would let the scheduler move a producer of Y past it.
    for (auto const &nd : graph.nodes()) {
        if (nd.kind != cg::OpKind::TileElementwise) {
            continue;
        }
        CHECK(nd.outputs.size() == x_tiles.size());
        for (auto out : nd.outputs) {
            CHECK(std::ranges::find(nd.inputs, out) != nd.inputs.end());
        }
    }

    require_tiles_match(Y, Y_ref);
}

TEST_CASE("TiledExpansion - a fused divide matches the per-tile lowering, leftovers included", "[ComputeGraph][Passes][Tiled]") {
    Grid const                          g{{2, 2}, {2, 2}};
    std::vector<std::vector<int>> const num{{0, 0}, {1, 1}};
    std::vector<std::vector<int>> const den = full_coords(g);
    std::vector<std::vector<int>> const dst = full_coords(g);

    auto build = [&](TiledRuntimeTensor<double> &A, TiledRuntimeTensor<double> &B, TiledRuntimeTensor<double> &C) {
        fill_det(A, 1.0);
        fill_det(B, 7.0); // fill_det uses sin(); the shift below keeps it away from 0
        fill_det(C, 2.0);
        for (auto const &co : den) {
            auto &t = B.tile(co);
            for (size_t i = 0; i < t.size(); ++i) {
                t.data()[i] += 3.0;
            }
        }
    };

    auto A_ref = make_tiled("A", g, num);
    auto B_ref = make_tiled("B", g, den);
    auto C_ref = make_tiled("C", g, dst);
    build(A_ref, B_ref, C_ref);
    {
        cg::Graph graph("divide_per_tile");
        {
            cg::CaptureGuard const guard(graph);
            cg::direct_division(1.5, A_ref, B_ref, 0.25, &C_ref);
        }
        cg::PassManager pm;
        pm.add(std::make_shared<cg::passes::TiledExpansion>(4096, -1.0, kPerTile, kNoFuse));
        REQUIRE(graph.apply(pm));
        graph.execute();
    }

    auto A = make_tiled("A2", g, num);
    auto B = make_tiled("B2", g, den);
    auto C = make_tiled("C2", g, dst);
    build(A, B, C);
    cg::Graph graph("divide_fused");
    {
        cg::CaptureGuard const guard(graph);
        cg::direct_division(1.5, A, B, 0.25, &C);
    }
    cg::PassManager pm;
    auto            pass = std::make_shared<cg::passes::TiledExpansion>(4096, -1.0, kPerTile, cg::passes::FuseTiles::Always);
    pm.add(pass);
    REQUIRE(graph.apply(pm));

    // The divides became one node and the two leftover beta-scales another.
    CHECK(pass->num_fused() == 2);
    CHECK(nodes_of_kind(graph, cg::OpKind::DirectDivision) == 0);
    CHECK(nodes_of_kind(graph, cg::OpKind::Scale) == 0);
    CHECK(nodes_of_kind(graph, cg::OpKind::TileElementwise) == 2);

    graph.execute();
    require_tiles_match(C, C_ref);
}

TEST_CASE("TiledExpansion - Auto fuses tiles too small to dispatch and leaves large ones alone", "[ComputeGraph][Passes][Tiled]") {
    // Decided against an explicit cost model rather than the detected one, so the
    // threshold this pins is the modelled one and not the host's bandwidth. With
    // the default profile a node costs 2.5 us to enter and memory runs at 40 GB/s,
    // so an in-place scale pays for its own dispatch at about 50 KB per tile.
    cg::CostModel const model;

    auto run = [&model](Grid const &g) {
        auto A = make_tiled("A", g, full_coords(g));
        fill_det(A, 1.5);
        cg::Graph graph("auto_fuse");
        {
            cg::CaptureGuard const guard(graph);
            cg::scale(-2.25, &A);
        }
        cg::PassManager pm;
        auto            pass = std::make_shared<cg::passes::TiledExpansion>(4096, -1.0, kPerTile, cg::passes::FuseTiles::Auto, model);
        pm.add(pass);
        REQUIRE(graph.apply(pm));
        return pass->num_fused();
    };

    CHECK(run(Grid{{8, 8}, {8, 8}}) == 1);         // 512 B a tile: all dispatch
    CHECK(run(Grid{{256}, {256}}) == 0);           // 512 KB in one tile: nothing to fuse anyway
    CHECK(run(Grid{{256, 256}, {256, 256}}) == 0); // 512 KB a tile: the dispatch is noise
}

TEST_CASE("TiledExpansion - a contraction's leftover scales fuse", "[ComputeGraph][Passes][Tiled]") {
    // C holds four tiles but only the diagonal ones receive a contribution, so the
    // other two are merely scaled by the output prefactor. Those are elementwise and
    // just as small as any other tile op, so they collapse the same way.
    Grid const gA{{2, 3}, {4, 5}};
    Grid const gB{{4, 5}, {3, 4}};
    Grid const gC{{2, 3}, {3, 4}};

    auto A_ref = make_tiled("A", gA, {{0, 0}, {1, 1}});
    auto B_ref = make_tiled("B", gB, {{0, 0}, {1, 1}});
    auto C_ref = make_tiled("C", gC, full_coords(gC));
    fill_det(A_ref, 1.0);
    fill_det(B_ref, 2.0);
    fill_det(C_ref, 3.0);
    {
        cg::Graph graph("leftover_per_tile");
        {
            cg::CaptureGuard const guard(graph);
            cg::einsum("ij <- ik ; kj", 0.5, &C_ref, 1.0, A_ref, B_ref);
        }
        cg::PassManager pm;
        pm.add(std::make_shared<cg::passes::TiledExpansion>(4096, -1.0, kPerTile, kNoFuse));
        REQUIRE(graph.apply(pm));
        graph.execute();
    }

    auto A = make_tiled("A2", gA, {{0, 0}, {1, 1}});
    auto B = make_tiled("B2", gB, {{0, 0}, {1, 1}});
    auto C = make_tiled("C2", gC, full_coords(gC));
    fill_det(A, 1.0);
    fill_det(B, 2.0);
    fill_det(C, 3.0);
    cg::Graph graph("leftover_fused");
    {
        cg::CaptureGuard const guard(graph);
        cg::einsum("ij <- ik ; kj", 0.5, &C, 1.0, A, B);
    }
    cg::PassManager pm;
    auto            pass = std::make_shared<cg::passes::TiledExpansion>(4096, -1.0, kPerTile, cg::passes::FuseTiles::Always);
    pm.add(pass);
    REQUIRE(graph.apply(pm));

    CHECK(pass->num_fused() == 1);
    CHECK(nodes_of_kind(graph, cg::OpKind::Scale) == 0);
    CHECK(nodes_of_kind(graph, cg::OpKind::TileElementwise) == 1);
    CHECK(nodes_of_kind(graph, cg::OpKind::Einsum) > 0);

    graph.execute();
    require_tiles_match(C, C_ref);
}

// ── Gather reuse ────────────────────────────────────────────────────────────
// Densifying gathers each tiled operand into a dense buffer. One operand is
// typically contracted many times in a row, and re-copying it each time is waste
// while nothing has written it.

TEST_CASE("TiledExpansion - an operand gathered twice is gathered once", "[ComputeGraph][Passes][Tiled]") {
    Grid const gA{{2, 3}, {4, 5}};
    Grid const gB{{4, 5}, {3, 4}};
    Grid const gC{{2, 3}, {3, 4}};

    auto aval = [](int r, int c) { return 1.0 + static_cast<double>(r * 9 + c); };
    auto bval = [](int r, int c) { return 2.0 - static_cast<double>(r * 7 + c); };

    // Two contractions over the same operands into different outputs. Nothing
    // writes A or B in between, so both gathers of each may be shared.
    auto run = [&](cg::passes::Densify densify, size_t &n_gather, size_t &reused) {
        auto A = make_tiled("A", gA, full_coords(gA));
        auto B = make_tiled("B", gB, full_coords(gB));
        auto C = make_tiled("C", gC, {});
        auto D = make_tiled("D", gC, {});
        fill(A, aval);
        fill(B, bval);
        cg::Graph graph("reuse");
        {
            cg::CaptureGuard const guard(graph);
            cg::einsum("ij <- ik ; kj", &C, A, B);
            cg::einsum("ij <- ik ; kj", 2.0, &D, 1.0, A, B);
        }
        cg::PassManager pm;
        auto            pass = std::make_shared<cg::passes::TiledExpansion>(4096, -1.0, densify, kNoFuse);
        pm.add(pass);
        REQUIRE(graph.apply(pm));
        n_gather = nodes_of_kind(graph, cg::OpKind::TileGather);
        reused   = pass->num_gathers_reused();
        graph.execute();
        return std::make_pair(gather(C, 5, 7), gather(D, 5, 7));
    };

    size_t     per_tile_gathers = 0, per_tile_reused = 0, dense_gathers = 0, dense_reused = 0;
    auto const reference = run(kPerTile, per_tile_gathers, per_tile_reused);
    auto const densified = run(cg::passes::Densify::Always, dense_gathers, dense_reused);

    CHECK(per_tile_gathers == 0);
    CHECK(per_tile_reused == 0);
    // Four operand gathers were called for; two were served from the first pair.
    CHECK(dense_gathers == 2);
    CHECK(dense_reused == 2);

    REQUIRE(reference.first.size() == densified.first.size());
    for (size_t i = 0; i < reference.first.size(); ++i) {
        REQUIRE_THAT(densified.first[i], Catch::Matchers::WithinRel(reference.first[i], 1e-12));
        REQUIRE_THAT(densified.second[i], Catch::Matchers::WithinRel(reference.second[i], 1e-12));
    }
}

TEST_CASE("TiledExpansion - a gathered operand that is written is gathered again", "[ComputeGraph][Passes][Tiled]") {
    // The invalidation that makes the reuse above safe. A is scaled between the two
    // contractions, so the second must see the scaled values -- reusing the first
    // gather would silently contract the stale copy.
    Grid const gA{{2, 3}, {4, 5}};
    Grid const gB{{4, 5}, {3, 4}};
    Grid const gC{{2, 3}, {3, 4}};

    auto aval = [](int r, int c) { return 1.0 + static_cast<double>(r * 9 + c); };
    auto bval = [](int r, int c) { return 2.0 - static_cast<double>(r * 7 + c); };

    auto run = [&](cg::passes::Densify densify, size_t &n_gather, size_t &reused) {
        auto A = make_tiled("A", gA, full_coords(gA));
        auto B = make_tiled("B", gB, full_coords(gB));
        auto C = make_tiled("C", gC, {});
        auto D = make_tiled("D", gC, {});
        fill(A, aval);
        fill(B, bval);
        cg::Graph graph("invalidate");
        {
            cg::CaptureGuard const guard(graph);
            cg::einsum("ij <- ik ; kj", &C, A, B);
            cg::scale(-3.0, &A);
            cg::einsum("ij <- ik ; kj", &D, A, B);
        }
        cg::PassManager pm;
        auto            pass = std::make_shared<cg::passes::TiledExpansion>(4096, -1.0, densify, kNoFuse);
        pm.add(pass);
        REQUIRE(graph.apply(pm));
        n_gather = nodes_of_kind(graph, cg::OpKind::TileGather);
        reused   = pass->num_gathers_reused();
        graph.execute();
        return gather(D, 5, 7);
    };

    size_t     per_tile_gathers = 0, per_tile_reused = 0, dense_gathers = 0, dense_reused = 0;
    auto const reference = run(kPerTile, per_tile_gathers, per_tile_reused);
    auto const densified = run(cg::passes::Densify::Always, dense_gathers, dense_reused);

    // A is gathered twice; only B is shared.
    CHECK(dense_gathers == 3);
    CHECK(dense_reused == 1);

    REQUIRE(reference.size() == densified.size());
    for (size_t i = 0; i < reference.size(); ++i) {
        REQUIRE_THAT(densified[i], Catch::Matchers::WithinRel(reference[i], 1e-12));
    }
}
