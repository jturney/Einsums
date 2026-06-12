//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

// Tier-B (B1) coverage: cg::einsum over TiledRuntimeTensor operands. The whole
// tiled tensor becomes a graph operand; the contraction walks the tile grid and
// composes dense per-tile einsums. Tests cover eager + captured execution, the
// non-symmetric (off-diagonal/rectangular) sparsity with infer-and-create
// output tiles, and the tile-partition alignment guard.

#include <Einsums/ComputeGraph.hpp>
#include <Einsums/Tensor/TiledRuntimeTensor.hpp>

#include <cmath>
#include <vector>

#include <Einsums/Testing.hpp>

using namespace einsums;
namespace cg = einsums::compute_graph;

namespace {

using Grid = std::vector<std::vector<int>>;

// Fill every grid tile of a 2-D tiled tensor from a global (row, col) function.
template <typename F>
void fill_tiled(TiledRuntimeTensor<double> &T, F &&f) {
    auto const &off = T.tile_offsets();
    auto const &sz  = T.tile_sizes();
    for (int ti = 0; ti < static_cast<int>(sz[0].size()); ++ti) {
        for (int tj = 0; tj < static_cast<int>(sz[1].size()); ++tj) {
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

// Reconstruct a dense R×C matrix from a 2-D tiled tensor (absent tiles read 0).
std::vector<std::vector<double>> gather(TiledRuntimeTensor<double> const &T, int R, int C) {
    std::vector<std::vector<double>> M(R, std::vector<double>(C, 0.0));
    auto const                      &off = T.tile_offsets();
    auto const                      &sz  = T.tile_sizes();
    for (auto const &[coord, tile] : T.tiles()) {
        int const ti = coord[0];
        int const tj = coord[1];
        for (int lr = 0; lr < sz[0][ti]; ++lr) {
            for (int lc = 0; lc < sz[1][tj]; ++lc) {
                M[off[0][ti] + lr][off[1][tj] + lc] = tile(std::vector<size_t>{static_cast<size_t>(lr), static_cast<size_t>(lc)});
            }
        }
    }
    return M;
}

} // namespace

TEST_CASE("TiledRuntimeTensor - tiled einsum GEMM (eager) matches dense", "[ComputeGraph][TiledRuntime]") {
    // i:{1,2}=3, k:{2,1}=3, j:{1,1}=2 ; contracted k partition matches in A and B.
    TiledRuntimeTensor<double> A("A", Grid{{1, 2}, {2, 1}});
    TiledRuntimeTensor<double> B("B", Grid{{2, 1}, {1, 1}});
    TiledRuntimeTensor<double> C("C", Grid{{1, 2}, {1, 1}});

    auto af = [](int r, int c) { return 1.0 + 3 * r + c; };
    auto bf = [](int r, int c) { return 0.5 + r - 2 * c; };
    fill_tiled(A, af);
    fill_tiled(B, bf);

    cg::einsum("ik;kj->ij", &C, A, B); // eager (c_pf=0, ab_pf=1)

    auto Cg = gather(C, 3, 2);
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 2; ++j) {
            double expected = 0.0;
            for (int k = 0; k < 3; ++k) {
                expected += af(i, k) * bf(k, j);
            }
            REQUIRE(std::abs(Cg[i][j] - expected) < 1e-12);
        }
    }
    // Infer-and-create: every output grid cell received a contribution.
    REQUIRE(C.num_filled_tiles() == 4);
}

TEST_CASE("TiledRuntimeTensor - tiled einsum GEMM inside a captured graph", "[ComputeGraph][TiledRuntime]") {
    TiledRuntimeTensor<double> A("A", Grid{{1, 2}, {2, 1}});
    TiledRuntimeTensor<double> B("B", Grid{{2, 1}, {1, 1}});
    TiledRuntimeTensor<double> C("C", Grid{{1, 2}, {1, 1}});

    auto af = [](int r, int c) { return 2.0 - r + 0.25 * c; };
    auto bf = [](int r, int c) { return 1.0 + 0.5 * r * c; };
    fill_tiled(A, af); // inputs must be materialized before execute
    fill_tiled(B, bf);

    cg::Graph g("tiled_gemm");
    {
        cg::CaptureGuard const guard(g);
        cg::einsum("ik;kj->ij", &C, A, B);
    }
    g.execute();

    auto Cg = gather(C, 3, 2);
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 2; ++j) {
            double expected = 0.0;
            for (int k = 0; k < 3; ++k) {
                expected += af(i, k) * bf(k, j);
            }
            REQUIRE(std::abs(Cg[i][j] - expected) < 1e-12);
        }
    }
}

TEST_CASE("TiledRuntimeTensor - sparse inputs: missing tiles contribute zero", "[ComputeGraph][TiledRuntime]") {
    TiledRuntimeTensor<double> A("A", Grid{{1, 2}, {2, 1}});
    TiledRuntimeTensor<double> B("B", Grid{{2, 1}, {1, 1}});
    TiledRuntimeTensor<double> C("C", Grid{{1, 2}, {1, 1}});

    auto af = [](int r, int c) { return 1.0 + 3 * r + c; };
    auto bf = [](int r, int c) { return 0.5 + r - 2 * c; };

    // Populate only a subset of A's tiles (off-diagonal/rectangular pattern):
    // present: (0,0)=1x2, (1,1)=2x1.  Absent: (0,1), (1,0).
    auto const &asz  = A.tile_sizes();
    auto const &aoff = A.tile_offsets();
    for (std::vector<int> coord : {std::vector<int>{0, 0}, std::vector<int>{1, 1}}) {
        auto &tile = A.tile(coord);
        tile.materialize();
        for (int lr = 0; lr < asz[0][coord[0]]; ++lr) {
            for (int lc = 0; lc < asz[1][coord[1]]; ++lc) {
                tile(std::vector<size_t>{static_cast<size_t>(lr), static_cast<size_t>(lc)}) =
                    af(aoff[0][coord[0]] + lr, aoff[1][coord[1]] + lc);
            }
        }
    }
    fill_tiled(B, bf);

    cg::einsum("ik;kj->ij", &C, A, B);

    // Reference: A is zero wherever its tile is absent.
    auto a_present = [](int i, int k) { return (i == 0 && k < 2) || (i >= 1 && k >= 2); };
    auto Cg        = gather(C, 3, 2);
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 2; ++j) {
            double expected = 0.0;
            for (int k = 0; k < 3; ++k) {
                if (a_present(i, k)) {
                    expected += af(i, k) * bf(k, j);
                }
            }
            REQUIRE(std::abs(Cg[i][j] - expected) < 1e-12);
        }
    }
}

TEST_CASE("TiledRuntimeTensor - misaligned contracted partition throws", "[ComputeGraph][TiledRuntime]") {
    TiledRuntimeTensor<double> A("A", Grid{{1, 2}, {2, 1}}); // k partition {2,1}
    TiledRuntimeTensor<double> B("B", Grid{{1, 2}, {1, 1}}); // k partition {1,2}, mismatch
    TiledRuntimeTensor<double> C("C", Grid{{1, 2}, {1, 1}});

    fill_tiled(A, [](int, int) { return 1.0; });
    fill_tiled(B, [](int, int) { return 1.0; });

    REQUIRE_THROWS(cg::einsum("ik;kj->ij", &C, A, B));
}
