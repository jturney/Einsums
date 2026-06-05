//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

// Per-block eigendecomposition over a block-diagonal TiledRuntimeTensor:
// cg::syev independently diagonalizes each diagonal tile, writing that block's
// eigenvalues into the matching rank-1 tile of W. (syev/heev are not bound to
// Python, so this coverage is C++-only.)

#include <Einsums/ComputeGraph.hpp>
#include <Einsums/Tensor/TiledRuntimeTensor.hpp>

#include <cmath>
#include <vector>

#include <Einsums/Testing.hpp>

using namespace einsums;
namespace cg = einsums::compute_graph;

namespace {
using Grid = std::vector<std::vector<int>>;
double &el(RuntimeTensor<double> &t, size_t i, size_t j) {
    return t(std::vector<size_t>{i, j});
}
} // namespace

TEST_CASE("TiledRuntimeTensor - tiled syev (block-diagonal)", "[ComputeGraph][TiledRuntime]") {
    // Block-diagonal 5x5: block0 2x2, block1 3x3. Row/col partitions match.
    TiledRuntimeTensor<double> A("A", Grid{{2, 3}, {2, 3}});
    TiledRuntimeTensor<double> W("W", Grid{{2, 3}}); // rank-1, eigenvalues per block

    A.add_tile({0, 0});
    A.add_tile({1, 1});
    A.materialize(); // tiles zero-initialized

    // block 0: [[2,1],[1,2]] -> eigenvalues {1, 3}
    auto &a0     = A.tile({0, 0});
    el(a0, 0, 0) = 2.0;
    el(a0, 0, 1) = 1.0;
    el(a0, 1, 0) = 1.0;
    el(a0, 1, 1) = 2.0;
    // block 1: diag(4,5,6) -> eigenvalues {4,5,6}
    auto &a1     = A.tile({1, 1});
    el(a1, 0, 0) = 4.0;
    el(a1, 1, 1) = 5.0;
    el(a1, 2, 2) = 6.0;

    cg::syev(&A, &W); // eager

    // LAPACK syev returns eigenvalues in ascending order, per block.
    auto &w0 = W.tile({0});
    REQUIRE(std::abs(w0(std::vector<size_t>{0}) - 1.0) < 1e-10);
    REQUIRE(std::abs(w0(std::vector<size_t>{1}) - 3.0) < 1e-10);

    auto &w1 = W.tile({1});
    REQUIRE(std::abs(w1(std::vector<size_t>{0}) - 4.0) < 1e-10);
    REQUIRE(std::abs(w1(std::vector<size_t>{1}) - 5.0) < 1e-10);
    REQUIRE(std::abs(w1(std::vector<size_t>{2}) - 6.0) < 1e-10);
}

TEST_CASE("TiledRuntimeTensor - tiled syev inside a captured graph", "[ComputeGraph][TiledRuntime]") {
    TiledRuntimeTensor<double> A("A", Grid{{2}, {2}});
    TiledRuntimeTensor<double> W("W", Grid{{2}});
    A.add_tile({0, 0});
    A.materialize();
    auto &a0     = A.tile({0, 0});
    el(a0, 0, 0) = 5.0;
    el(a0, 0, 1) = 2.0;
    el(a0, 1, 0) = 2.0;
    el(a0, 1, 1) = 5.0; // eigenvalues {3, 7}

    cg::Graph g("tiled_syev");
    {
        cg::CaptureGuard const guard(g);
        cg::syev(&A, &W);
    }
    g.execute();

    auto &w0 = W.tile({0});
    REQUIRE(std::abs(w0(std::vector<size_t>{0}) - 3.0) < 1e-10);
    REQUIRE(std::abs(w0(std::vector<size_t>{1}) - 7.0) < 1e-10);
}

TEST_CASE("TiledRuntimeTensor - tiled syev rejects off-diagonal tiles", "[ComputeGraph][TiledRuntime]") {
    TiledRuntimeTensor<double> A("A", Grid{{2, 3}, {2, 3}});
    TiledRuntimeTensor<double> W("W", Grid{{2, 3}});
    A.add_tile({0, 0});
    A.add_tile({0, 1}); // off-diagonal -> not block-diagonal
    A.materialize();
    REQUIRE_THROWS(cg::syev(&A, &W));
}
