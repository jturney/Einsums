//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

// Coverage for TiledRuntimeTensor: a runtime-rank, tile-wise sparse
// tensor that the ComputeGraph can capture for lifecycle orchestration
// (materialize / release / zero) without understanding its tile structure.
// These tests exercise the integration seam (make_handle) and the captured
// lifecycle callbacks the passes drive, plus the class's own behavior. They
// deliberately include an off-diagonal, *rectangular* tile to prove the
// general (non-symmetric) case works, not just block-diagonal.

#include <Einsums/ComputeGraph.hpp>
#include <Einsums/Tensor/RuntimeTensor.hpp>
#include <Einsums/Tensor/TiledRuntimeTensor.hpp>

#include <vector>

#include <Einsums/Testing.hpp>

using namespace einsums;
namespace cg = einsums::compute_graph;

TEST_CASE("TiledRuntimeTensor - grid shape and general tiles", "[ComputeGraph][TiledRuntime]") {
    // axis 0 split into tiles of size {2, 3}; axis 1 into {4, 5}.
    // Global shape is 5 x 9, with a 2x2 grid of possible tiles.
    TiledRuntimeTensor<double> t("t", {{2, 3}, {4, 5}});

    REQUIRE(t.rank() == 2);
    REQUIRE(t.dim(0) == 5);
    REQUIRE(t.dim(1) == 9);
    REQUIRE(t.dim(-1) == 9); // negative indexing
    REQUIRE(t.grid_size() == 4);
    REQUIRE(t.num_filled_tiles() == 0);

    // A diagonal tile (square) and an off-diagonal tile (rectangular), the
    // latter is exactly the non-symmetric case BlockTensor cannot represent.
    t.add_tile({0, 0}); // 2 x 4
    t.add_tile({0, 1}); // 2 x 5  (off-diagonal, rectangular)
    t.add_tile({1, 1}); // 3 x 5

    REQUIRE(t.num_filled_tiles() == 3);
    REQUIRE(t.has_tile({0, 0}));
    REQUIRE(t.has_tile({0, 1}));
    REQUIRE_FALSE(t.has_tile({1, 0}));

    REQUIRE(t.tile({0, 1}).dim(0) == 2);
    REQUIRE(t.tile({0, 1}).dim(1) == 5);
    REQUIRE(t.tile({1, 1}).dim(0) == 3);
    REQUIRE(t.tile({1, 1}).dim(1) == 5);

    // Wrong-rank coordinate is rejected.
    REQUIRE_THROWS(t.add_tile({0}));
}

TEST_CASE("TiledRuntimeTensor - data() null for multi-tile, set for single cell", "[ComputeGraph][TiledRuntime]") {
    TiledRuntimeTensor<double> multi("multi", {{2, 3}, {4, 5}});
    multi.add_tile({0, 0});
    multi.add_tile({1, 1});
    multi.materialize();
    // No single contiguous buffer -> data() is null.
    REQUIRE(multi.data() == nullptr);

    // Degenerate single-cell grid: one possible tile -> data() exposes it.
    TiledRuntimeTensor<double> single("single", {{4}, {5}});
    single.add_tile({0, 0});
    REQUIRE(single.grid_size() == 1);
    single.materialize();
    REQUIRE(single.data() != nullptr);
}

TEST_CASE("TiledRuntimeTensor - deferred lifecycle materialize/zero/release", "[ComputeGraph][TiledRuntime]") {
    TiledRuntimeTensor<double> t("t", {{2, 3}, {4, 5}});
    t.add_tile({0, 0});
    t.add_tile({0, 1});

    // Tiles are declared but not yet backed by storage.
    REQUIRE_FALSE(t.is_materialized());

    t.materialize();
    REQUIRE(t.is_materialized());
    REQUIRE(t.tile({0, 0}).is_materialized());
    REQUIRE(t.tile({0, 1}).is_materialized());

    // zero() touches every populated tile.
    t.tile({0, 0})(std::vector<size_t>{0, 0}) = 7.0;
    t.zero();
    REQUIRE(t.tile({0, 0})(std::vector<size_t>{0, 0}) == 0.0);

    t.release();
    REQUIRE_FALSE(t.is_materialized());
    // Sparsity pattern (which tiles exist) survives a release.
    REQUIRE(t.num_filled_tiles() == 2);
}

TEST_CASE("TiledRuntimeTensor - ComputeGraph make_handle integration", "[ComputeGraph][TiledRuntime]") {
    TiledRuntimeTensor<double> t("tiled", {{2, 3}, {4, 5}});
    t.add_tile({0, 0});
    t.add_tile({0, 1});

    cg::TensorHandle h = cg::make_handle(t, 0);

    // The handle is flagged tiled and carries the honest global shape, but no
    // single data buffer, and starts deferred (tiles not yet allocated).
    REQUIRE(h.is_tiled);
    REQUIRE(h.rank == 2);
    REQUIRE(h.dims == std::vector<size_t>{5, 9});
    REQUIRE(h.data_ptr == nullptr);
    REQUIRE(h.alloc_state == cg::AllocState::Deferred);

    // The captured lifecycle callbacks (what MaterializationPass / FreeInsertion
    // drive) operate tile-by-tile on the backing tensor.
    REQUIRE(static_cast<bool>(h.materialize_fn));
    REQUIRE(static_cast<bool>(h.release_fn));

    h.materialize_fn();
    REQUIRE(t.is_materialized());

    h.release_fn();
    REQUIRE_FALSE(t.is_materialized());
    REQUIRE(t.num_filled_tiles() == 2);
}

TEST_CASE("TiledRuntimeTensor - dense RuntimeTensor handle is not tiled", "[ComputeGraph][TiledRuntime]") {
    // Control: an ordinary dense runtime tensor must not be flagged tiled.
    RuntimeTensor<double> const dense("dense", std::vector<size_t>{3, 3}, /*row_major=*/true);
    cg::TensorHandle            h = cg::make_handle(dense, 0);
    REQUIRE_FALSE(h.is_tiled);
    REQUIRE(h.data_ptr != nullptr);
}
