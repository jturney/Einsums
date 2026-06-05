//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

/// @file TensorFileSliceWrite.cpp
/// @brief Unit tests for TensorFile::write_slice + RuntimeTensor overloads.
///
/// Covers the round-trip pattern: reserve a tensor, fill it block-by-block
/// via write_slice, then read the full tensor back and verify every element.
/// Also exercises the new RuntimeTensor overloads end-to-end so the Python
/// bindings have the same behavior they will inherit.

#include <Einsums/Tensor/RuntimeTensor.hpp>
#include <Einsums/Tensor/Tensor.hpp>
#include <Einsums/TensorIO/TensorFile.hpp>
#include <Einsums/TensorUtilities/CreateRandomTensor.hpp>
#include <Einsums/TensorUtilities/CreateZeroTensor.hpp>

#include <cstdio>

#include <Einsums/Testing.hpp>

using namespace einsums;
using namespace einsums::tensor_io;

namespace {
struct TempFile {
    std::string path;
    explicit TempFile(std::string name) : path("/tmp/einsums_slice_" + std::move(name)) {}
    ~TempFile() { std::remove(path.c_str()); }
};
} // namespace

// ── Static-rank: write full, then write_slice patches a slab ─────────────

TEST_CASE("TensorFile::write_slice patches a slab of an existing entry", "[TensorIO][slice]") {
    TempFile const tmp("static_slab.etn");

    // Build a 4x4 tensor of zeros, then patch [1:3, 1:3] with a 2x2 of 7's.
    auto              base = create_zero_tensor<double>("A", 4, 4);
    Tensor<double, 2> patch("patch", 2, 2);
    patch.zero();
    for (size_t i = 0; i < 2; ++i) {
        for (size_t j = 0; j < 2; ++j) {
            patch(i, j) = 7.0;
        }
    }

    {
        TensorFile out(tmp.path, TensorFile::Mode::Write);
        out.write("A", base);
        out.write_slice("A", patch, {{{1, 3}, {1, 3}}});
    }

    Tensor<double, 2> roundtrip("rt", 4, 4);
    {
        TensorFile in(tmp.path, TensorFile::Mode::Read);
        in.read("A", roundtrip);
    }

    for (size_t i = 0; i < 4; ++i) {
        for (size_t j = 0; j < 4; ++j) {
            double const expected = (i >= 1 && i < 3 && j >= 1 && j < 3) ? 7.0 : 0.0;
            REQUIRE(roundtrip(i, j) == expected);
        }
    }
}

TEST_CASE("TensorFile::write_slice round-trips through read_slice", "[TensorIO][slice]") {
    TempFile const tmp("slice_roundtrip.etn");

    auto base = create_random_tensor<double>("A", 6, 6);

    Tensor<double, 2> sub("sub", 2, 3);
    for (size_t i = 0; i < 2; ++i) {
        for (size_t j = 0; j < 3; ++j) {
            sub(i, j) = j + 100.0 + 10.0 * i;
        }
    }

    {
        TensorFile out(tmp.path, TensorFile::Mode::Write);
        out.write("A", base);
        out.write_slice("A", sub, {{{2, 4}, {1, 4}}});
    }

    Tensor<double, 2> read_back("rb", 2, 3);
    {
        TensorFile in(tmp.path, TensorFile::Mode::Read);
        in.read_slice("A", read_back, {{{2, 4}, {1, 4}}});
    }

    for (size_t i = 0; i < 2; ++i) {
        for (size_t j = 0; j < 3; ++j) {
            REQUIRE(read_back(i, j) == 100.0 + 10.0 * i + j);
        }
    }
}

TEST_CASE("TensorFile::write_slice rejects mismatched dims", "[TensorIO][slice]") {
    TempFile const          tmp("mismatch.etn");
    auto                    base = create_zero_tensor<double>("A", 4, 4);
    Tensor<double, 2> const wrong("wrong", 3, 2); // doesn't match any slab of [1:3, 1:3]

    TensorFile out(tmp.path, TensorFile::Mode::Write);
    out.write("A", base);
    REQUIRE_THROWS_AS(out.write_slice("A", wrong, {{{1, 3}, {1, 3}}}), std::runtime_error);
}

// ── reserve + write_slice — fill an entry block-by-block ────────────────

TEST_CASE("TensorFile::reserve + write_slice fills a tensor incrementally", "[TensorIO][slice]") {
    TempFile const tmp("reserve.etn");

    constexpr size_t N = 4;
    constexpr size_t B = 2;

    // Reserve space for the full tensor, then fill 2x2 quadrants one at a time.
    {
        TensorFile out(tmp.path, TensorFile::Mode::ReadWrite);
        out.reserve<double>("A", {N, N});

        // Fill each quadrant with a unique value so we can verify placement.
        for (size_t const bi : {std::size_t{0}, std::size_t{1}}) {
            for (size_t const bj : {std::size_t{0}, std::size_t{1}}) {
                Tensor<double, 2> block("blk", B, B);
                double const      fill = 10.0 * bi + bj;
                for (size_t i = 0; i < B; ++i) {
                    for (size_t j = 0; j < B; ++j) {
                        block(i, j) = fill;
                    }
                }
                out.write_slice("A", block, {{{bi * B, (bi + 1) * B}, {bj * B, (bj + 1) * B}}});
            }
        }
    }

    Tensor<double, 2> roundtrip("rt", N, N);
    {
        TensorFile in(tmp.path, TensorFile::Mode::Read);
        in.read("A", roundtrip);
    }

    for (size_t bi = 0; bi < 2; ++bi) {
        for (size_t bj = 0; bj < 2; ++bj) {
            double const expected = 10.0 * bi + bj;
            for (size_t i = 0; i < B; ++i) {
                for (size_t j = 0; j < B; ++j) {
                    REQUIRE(roundtrip(bi * B + i, bj * B + j) == expected);
                }
            }
        }
    }
}

// ── RuntimeTensor overloads ──────────────────────────────────────────────

TEST_CASE("TensorFile RuntimeTensor read/write round-trip", "[TensorIO][runtime]") {
    TempFile const tmp("rt_full.etn");

    GeneralRuntimeTensor<double, std::allocator<double>> rt("A", std::vector<size_t>{3, 4});
    for (size_t i = 0; i < rt.size(); ++i) {
        rt.data()[i] = static_cast<double>(i);
    }

    {
        TensorFile out(tmp.path, TensorFile::Mode::Write);
        out.write("A", rt);
    }

    GeneralRuntimeTensor<double, std::allocator<double>> back("back", std::vector<size_t>{0});
    {
        TensorFile in(tmp.path, TensorFile::Mode::Read);
        in.read("A", back);
    }

    REQUIRE(back.rank() == 2);
    REQUIRE(back.dim(0) == 3);
    REQUIRE(back.dim(1) == 4);
    for (size_t i = 0; i < back.size(); ++i) {
        REQUIRE(back.data()[i] == static_cast<double>(i));
    }
}

TEST_CASE("TensorFile RuntimeTensor read_slice + write_slice", "[TensorIO][runtime][slice]") {
    TempFile const tmp("rt_slab.etn");

    // Reserve a 4x4 tensor.
    {
        TensorFile out(tmp.path, TensorFile::Mode::ReadWrite);
        out.reserve<double>("A", {4, 4});

        // Fill with zeros first.
        GeneralRuntimeTensor<double, std::allocator<double>> zeros("z", std::vector<size_t>{4, 4});
        zeros.zero();
        out.write_slice("A", zeros, {{0, 4}, {0, 4}});

        // Patch [1:3, 1:3] with 9's.
        GeneralRuntimeTensor<double, std::allocator<double>> patch("p", std::vector<size_t>{2, 2});
        patch.set_all(9.0);
        out.write_slice("A", patch, {{1, 3}, {1, 3}});
    }

    GeneralRuntimeTensor<double, std::allocator<double>> slab("s", std::vector<size_t>{0});
    {
        TensorFile in(tmp.path, TensorFile::Mode::Read);
        in.read_slice("A", slab, {{1, 3}, {1, 3}});
    }
    REQUIRE(slab.rank() == 2);
    REQUIRE(slab.dim(0) == 2);
    REQUIRE(slab.dim(1) == 2);
    for (size_t i = 0; i < slab.size(); ++i) {
        REQUIRE(slab.data()[i] == 9.0);
    }
}
