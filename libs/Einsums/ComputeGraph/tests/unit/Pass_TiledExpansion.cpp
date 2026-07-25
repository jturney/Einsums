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
