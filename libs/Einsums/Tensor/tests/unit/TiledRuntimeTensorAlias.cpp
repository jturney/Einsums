//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#include <Einsums/Tensor/TiledRuntimeTensor.hpp>

#include <vector>

#include <Einsums/Testing.hpp>

// Prototype: a TiledRuntimeTensor whose tiles ALIAS external row-major buffers
// (e.g. psi4 Matrix irrep blocks) with no copy. This is the zero-copy precursor
// to making such tensors the storage backend for psi4 Matrix/Vector.
TEST_CASE("TiledRuntimeTensor aliases external buffers (zero-copy)", "[tensor][tiled][alias]") {
    using namespace einsums;

    // Two irreps per axis: block (0,0) is 2x2, block (1,1) is 3x3, both row-major
    // — exactly how psi4 stores contiguous irrep blocks.
    std::vector<int> const     axis{2, 3};
    TiledRuntimeTensor<double> T("aliased", {axis, axis}, /*row_major=*/true);

    std::vector<double> blk00 = {1, 2, 3, 4};                         // 2x2
    std::vector<double> blk11 = {10, 11, 12, 13, 14, 15, 16, 17, 18}; // 3x3

    T.add_alias_tile({0, 0}, blk00.data(), /*row_major=*/true);
    T.add_alias_tile({1, 1}, blk11.data(), /*row_major=*/true);

    SECTION("tile data() points at the external buffer — no copy") {
        REQUIRE(T.tile({0, 0}).data() == blk00.data());
        REQUIRE(T.tile({1, 1}).data() == blk11.data());
    }

    SECTION("reads see the external data, row-major") {
        REQUIRE(T.tile({0, 0})(std::vector<size_t>{0, 0}) == 1.0);
        REQUIRE(T.tile({0, 0})(std::vector<size_t>{0, 1}) == 2.0); // row-major: next element
        REQUIRE(T.tile({0, 0})(std::vector<size_t>{1, 0}) == 3.0);
        REQUIRE(T.tile({1, 1})(std::vector<size_t>{2, 2}) == 18.0);
    }

    SECTION("zero-copy is live in both directions") {
        blk00[3] = 99.0; // mutate the external buffer
        REQUIRE(T.tile({0, 0})(std::vector<size_t>{1, 1}) == 99.0);
        T.tile({0, 0})(std::vector<size_t>{0, 0}) = -7.0; // write through the tensor
        REQUIRE(blk00[0] == -7.0);                        // shows in the external buffer
    }

    SECTION("aliased tiles are materialized and release/materialize-proof") {
        REQUIRE(T.tile({0, 0}).is_materialized());
        T.tile({0, 0}).materialize();                   // no-op
        REQUIRE(T.tile({0, 0}).data() == blk00.data()); // still pointing at the buffer
        T.tile({0, 0}).release();                       // no-op — must not free foreign memory
        REQUIRE(T.tile({0, 0}).data() == blk00.data());
        REQUIRE(blk00[0] == 1.0); // buffer untouched
    }

    SECTION("global shape reflects the grid") {
        REQUIRE(T.rank() == 2);
        REQUIRE(T.dims() == std::vector<size_t>{5, 5}); // 2 + 3 per axis
        REQUIRE(T.num_filled_tiles() == 2);
    }
}
