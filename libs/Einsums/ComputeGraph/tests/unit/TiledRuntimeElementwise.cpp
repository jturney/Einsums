//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

// Elementwise tiled ops: cg::scale / cg::element_transform / cg::axpy over
// TiledRuntimeTensor operands, composed per tile. Covers eager + captured
// execution against a dense reference, including an off-diagonal/rectangular
// tile layout.

#include <Einsums/ComputeGraph.hpp>
#include <Einsums/Tensor/TiledRuntimeTensor.hpp>

#include <cmath>
#include <vector>

#include <Einsums/Testing.hpp>

using namespace einsums;
namespace cg = einsums::compute_graph;

namespace {

using Grid = std::vector<std::vector<int>>;

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

TEST_CASE("TiledRuntimeTensor - tiled scale (eager + captured)", "[ComputeGraph][TiledRuntime]") {
    auto f = [](int r, int c) { return 1.0 + 2 * r - c; };

    // Eager.
    TiledRuntimeTensor<double> A("A", Grid{{2, 3}, {4, 5}}); // 5 x 9
    fill_tiled(A, f);
    cg::scale(2.0, &A);
    auto Ag = gather(A, 5, 9);
    for (int i = 0; i < 5; ++i) {
        for (int j = 0; j < 9; ++j) {
            REQUIRE(std::abs(Ag[i][j] - 2.0 * f(i, j)) < 1e-12);
        }
    }

    // Captured.
    TiledRuntimeTensor<double> B("B", Grid{{2, 3}, {4, 5}});
    fill_tiled(B, f);
    cg::Graph g("tiled_scale");
    {
        cg::CaptureGuard const guard(g);
        cg::scale(-3.0, &B);
    }
    g.execute();
    auto Bg = gather(B, 5, 9);
    for (int i = 0; i < 5; ++i) {
        for (int j = 0; j < 9; ++j) {
            REQUIRE(std::abs(Bg[i][j] - (-3.0) * f(i, j)) < 1e-12);
        }
    }
}

TEST_CASE("TiledRuntimeTensor - tiled element_transform", "[ComputeGraph][TiledRuntime]") {
    auto f = [](int r, int c) { return 0.5 + r + 0.25 * c; };

    TiledRuntimeTensor<double> A("A", Grid{{2, 3}, {4, 5}});
    fill_tiled(A, f);

    cg::element_transform(&A, [](double x) { return x * x; });

    auto Ag = gather(A, 5, 9);
    for (int i = 0; i < 5; ++i) {
        for (int j = 0; j < 9; ++j) {
            REQUIRE(std::abs(Ag[i][j] - f(i, j) * f(i, j)) < 1e-12);
        }
    }
}

TEST_CASE("TiledRuntimeTensor - tiled axpy (eager + captured)", "[ComputeGraph][TiledRuntime]") {
    auto xf = [](int r, int c) { return 1.0 + r - c; };
    auto yf = [](int r, int c) { return 3.0 - r + 2 * c; };

    // Eager.
    TiledRuntimeTensor<double> X("X", Grid{{2, 3}, {4, 5}});
    TiledRuntimeTensor<double> Y("Y", Grid{{2, 3}, {4, 5}});
    fill_tiled(X, xf);
    fill_tiled(Y, yf);
    cg::axpy(1.5, X, &Y);
    auto Yg = gather(Y, 5, 9);
    for (int i = 0; i < 5; ++i) {
        for (int j = 0; j < 9; ++j) {
            REQUIRE(std::abs(Yg[i][j] - (yf(i, j) + 1.5 * xf(i, j))) < 1e-12);
        }
    }

    // Captured.
    TiledRuntimeTensor<double> X2("X2", Grid{{2, 3}, {4, 5}});
    TiledRuntimeTensor<double> Y2("Y2", Grid{{2, 3}, {4, 5}});
    fill_tiled(X2, xf);
    fill_tiled(Y2, yf);
    cg::Graph g("tiled_axpy");
    {
        cg::CaptureGuard const guard(g);
        cg::axpy(-2.0, X2, &Y2);
    }
    g.execute();
    auto Y2g = gather(Y2, 5, 9);
    for (int i = 0; i < 5; ++i) {
        for (int j = 0; j < 9; ++j) {
            REQUIRE(std::abs(Y2g[i][j] - (yf(i, j) - 2.0 * xf(i, j))) < 1e-12);
        }
    }
}
