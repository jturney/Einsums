//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

// Elementwise tiled ops: cg::scale / cg::element_transform / cg::axpy over
// TiledRuntimeTensor operands, composed per tile. Covers eager + captured
// execution against a dense reference, including an off-diagonal/rectangular
// tile layout.

#include <Einsums/ComputeGraph.hpp>
#include <Einsums/Tensor/Tensor.hpp>
#include <Einsums/Tensor/TiledRuntimeTensor.hpp>

#include <algorithm>
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

TEST_CASE("TiledRuntimeTensor - tiled axpy declares its read of the destination", "[ComputeGraph][TiledRuntime]") {
    // tiled_axpy computes Y += alpha*X, so Y is read as well as written and must
    // appear in the node's INPUTS (the RMW convention the dense axpy follows).
    // Listing only X made the node look like a pure overwrite, which is what the
    // loop test below turns into a wrong answer.
    TiledRuntimeTensor<double> X("X", Grid{{2, 3}, {4, 5}});
    TiledRuntimeTensor<double> Y("Y", Grid{{2, 3}, {4, 5}});
    fill_tiled(X, [](int r, int c) { return 1.0 + r - c; });
    fill_tiled(Y, [](int r, int c) { return 3.0 - r + 2 * c; });

    cg::Graph g("tiled_axpy_rmw");
    {
        cg::CaptureGuard const guard(g);
        cg::axpy(1.5, X, &Y);
    }

    REQUIRE(g.num_nodes() == 1);
    auto const &node = g.nodes()[0];
    REQUIRE(node.outputs.size() == 1);
    REQUIRE(std::ranges::find(node.inputs, node.outputs[0]) != node.inputs.end());
}

TEST_CASE("TiledRuntimeTensor - tiled axpy in a loop is not hoisted", "[ComputeGraph][TiledRuntime]") {
    // A tiled accumulation in a loop body with a loop-invariant X. LIH decides
    // "self-modifying" from reads_destination() plus an input==output scan; a
    // tiled axpy is an OpKind::Custom node with no descriptor, so it fails the
    // first check and used to fail the second as well by omitting Y from its
    // inputs. The node then looked invariant and was hoisted OUT of the loop,
    // accumulating once instead of once per iteration.
    //
    // Checked differentially against the identical dense loop rather than a
    // hand-computed trip count, so the test does not encode add_loop's
    // iteration semantics.
    auto xf = [](int r, int c) { return 1.0 + r - c; };
    auto yf = [](int r, int c) { return 3.0 - r + 2 * c; };

    auto build_loop = [](cg::Graph &g, auto &X, auto &Y) {
        auto                  &body = g.add_loop("loop", 4, [](size_t iter) { return iter < 3; });
        cg::CaptureGuard const guard(body);
        cg::axpy(1.5, X, &Y);
    };

    // Dense oracle: OpKind::Axpy, which reads_destination() already protects.
    Tensor<double, 2> Xd("Xd", 5, 9);
    Tensor<double, 2> Yd("Yd", 5, 9);
    for (int i = 0; i < 5; ++i) {
        for (int j = 0; j < 9; ++j) {
            Xd(i, j) = xf(i, j);
            Yd(i, j) = yf(i, j);
        }
    }
    cg::Graph gd("dense_axpy_loop");
    build_loop(gd, Xd, Yd);
    auto [dense_modified, dense_pass] = gd.apply<cg::passes::LoopInvariantHoisting>();
    CHECK(dense_pass.num_hoisted() == 0);
    gd.execute();

    // The loop must run more than once, or the comparison below is vacuous.
    REQUIRE(std::abs(Yd(0, 0) - (yf(0, 0) + 1.5 * xf(0, 0))) > 1e-12);

    TiledRuntimeTensor<double> Xt("Xt", Grid{{2, 3}, {4, 5}});
    TiledRuntimeTensor<double> Yt("Yt", Grid{{2, 3}, {4, 5}});
    fill_tiled(Xt, xf);
    fill_tiled(Yt, yf);
    cg::Graph gt("tiled_axpy_loop");
    build_loop(gt, Xt, Yt);
    auto [tiled_modified, tiled_pass] = gt.apply<cg::passes::LoopInvariantHoisting>();
    REQUIRE(tiled_pass.num_hoisted() == 0);
    gt.execute();

    auto Ytg = gather(Yt, 5, 9);
    for (int i = 0; i < 5; ++i) {
        for (int j = 0; j < 9; ++j) {
            REQUIRE(std::abs(Ytg[i][j] - Yd(i, j)) < 1e-12);
        }
    }
}

TEST_CASE("TiledRuntimeTensor - tiled direct_division (eager + captured)", "[ComputeGraph][TiledRuntime]") {
    auto af = [](int r, int c) { return 1.0 + r - c; };
    auto bf = [](int r, int c) { return 2.5 + 0.5 * r + c; }; // never zero
    auto cf = [](int r, int c) { return 0.25 * r - c; };

    // Eager, with beta != 0 so the destination is read.
    TiledRuntimeTensor<double> A("A", Grid{{2, 3}, {4, 5}});
    TiledRuntimeTensor<double> B("B", Grid{{2, 3}, {4, 5}});
    TiledRuntimeTensor<double> C("C", Grid{{2, 3}, {4, 5}});
    fill_tiled(A, af);
    fill_tiled(B, bf);
    fill_tiled(C, cf);
    cg::direct_division(2.0, A, B, 0.5, &C);
    auto Cg = gather(C, 5, 9);
    for (int i = 0; i < 5; ++i) {
        for (int j = 0; j < 9; ++j) {
            REQUIRE(std::abs(Cg[i][j] - (2.0 * af(i, j) / bf(i, j) + 0.5 * cf(i, j))) < 1e-12);
        }
    }

    // Captured.
    TiledRuntimeTensor<double> A2("A2", Grid{{2, 3}, {4, 5}});
    TiledRuntimeTensor<double> B2("B2", Grid{{2, 3}, {4, 5}});
    TiledRuntimeTensor<double> C2("C2", Grid{{2, 3}, {4, 5}});
    fill_tiled(A2, af);
    fill_tiled(B2, bf);
    fill_tiled(C2, cf);
    cg::Graph g("tiled_divide");
    {
        cg::CaptureGuard const guard(g);
        cg::direct_division(2.0, A2, B2, 0.5, &C2);
    }
    // beta != 0 reads C, so C must appear among the inputs.
    REQUIRE(g.num_nodes() == 1);
    REQUIRE(std::ranges::find(g.nodes()[0].inputs, g.nodes()[0].outputs[0]) != g.nodes()[0].inputs.end());

    g.execute();
    auto C2g = gather(C2, 5, 9);
    for (int i = 0; i < 5; ++i) {
        for (int j = 0; j < 9; ++j) {
            REQUIRE(std::abs(C2g[i][j] - Cg[i][j]) < 1e-12);
        }
    }
}

TEST_CASE("TiledRuntimeTensor - tiled direct_division rejects a missing denominator tile", "[ComputeGraph][TiledRuntime]") {
    // An absent denominator block is a structurally zero divisor. The dense op
    // would produce infinities; the tiled one says so instead.
    TiledRuntimeTensor<double> A("A", Grid{{2, 3}, {4, 5}});
    TiledRuntimeTensor<double> B("B", Grid{{2, 3}, {4, 5}});
    TiledRuntimeTensor<double> C("C", Grid{{2, 3}, {4, 5}});
    fill_tiled(A, [](int r, int c) { return 1.0 + r - c; });
    // B gets only one tile, so most of A's tiles have no denominator.
    B.tile({0, 0}).materialize();
    REQUIRE_THROWS(cg::direct_division(1.0, A, B, 0.0, &C));
}
