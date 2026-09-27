//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

/// @file TensorFileLayout.cpp
/// @brief An .etn data region is column-major whatever layout the tensor had in memory.
///
/// The transfers used to copy a tensor's memory to and from the file byte for byte. A row-major
/// tensor (built with the row_major constructors, or in a process run with --einsums:row-major)
/// therefore wrote its elements transposed: a column-major reader got the transpose back, and a
/// slice read or write, whose file offsets assume column-major data, touched the wrong elements
/// even within one row-major process. Every case here fills tensors by logical index and checks
/// by logical index, so each one fails if any transfer copies memory order instead of file order.

#include <Einsums/Comm/Runtime.hpp>
#include <Einsums/Options/Get.hpp>
#include <Einsums/Tensor/RuntimeTensor.hpp>
#include <Einsums/Tensor/Tensor.hpp>
#include <Einsums/TensorBase/Options.hpp>
#include <Einsums/TensorIO/DistributedTensorFile.hpp>
#include <Einsums/TensorIO/TensorFile.hpp>

#include <cstdio>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <string>
#include <vector>

#include <Einsums/Testing.hpp>

using namespace einsums;
using namespace einsums::tensor_io;

namespace {

struct TempFile {
    std::string path;
    explicit TempFile(std::string const &name, bool shared = false)
        : path((std::filesystem::temp_directory_path() /
                ("einsums_layout_" + (shared ? std::string{} : "r" + std::to_string(comm::world_rank()) + "_") + name))
                   .string()) {}
    ~TempFile() { std::remove(path.c_str()); }
};

/// A distinct value for every logical index, with an imaginary part for complex types, so a
/// transposed, shifted, or half-swapped element shows.
template <typename T>
T value_at(size_t i, size_t j, size_t k) {
    double const x = 1.0 + static_cast<double>(i) + 1000.0 * static_cast<double>(j) + 1000000.0 * static_cast<double>(k);
    if constexpr (IsComplexV<T>) {
        using R = RemoveComplexT<T>;
        return T{static_cast<R>(x), static_cast<R>(-0.5 * x)};
    } else {
        return static_cast<T>(x);
    }
}

template <typename TensorType>
void fill(TensorType &t, size_t i0 = 0, size_t j0 = 0, size_t k0 = 0) {
    using T = typename TensorType::ValueType;
    for (size_t i = 0; i < static_cast<size_t>(t.dim(0)); ++i)
        for (size_t j = 0; j < static_cast<size_t>(t.dim(1)); ++j)
            for (size_t k = 0; k < static_cast<size_t>(t.dim(2)); ++k)
                t(i, j, k) = value_at<T>(i0 + i, j0 + j, k0 + k);
}

/// Count the elements of @p t that differ from the pattern starting at (i0, j0, k0).
template <typename TensorType>
size_t mismatches(TensorType const &t, size_t i0 = 0, size_t j0 = 0, size_t k0 = 0) {
    using T      = typename TensorType::ValueType;
    size_t wrong = 0;
    for (size_t i = 0; i < static_cast<size_t>(t.dim(0)); ++i)
        for (size_t j = 0; j < static_cast<size_t>(t.dim(1)); ++j)
            for (size_t k = 0; k < static_cast<size_t>(t.dim(2)); ++k)
                wrong += t(i, j, k) == value_at<T>(i0 + i, j0 + j, k0 + k) ? 0 : 1;
    return wrong;
}

auto file_bytes(std::string const &path) -> std::vector<char> {
    std::ifstream in(path, std::ios::binary);
    return {std::istreambuf_iterator<char>(in), std::istreambuf_iterator<char>()};
}

template <typename T>
using RT = GeneralRuntimeTensor<T, std::allocator<T>>;

/// Restores the storage-order option however the case exits.
struct RestoreRowMajor {
    bool previous = config::get(option::RowMajor);
    ~RestoreRowMajor() { config::set(option::RowMajor, previous); }
};

} // namespace

TEMPLATE_LIST_TEST_CASE("TensorFile - a file is the same whichever layout wrote it", "[TensorIO][Layout]", testing::AllScalarTypes) {
    TempFile const from_col("col.etn");
    TempFile const from_row("row.etn");

    Tensor<TestType, 3> col(false, "A", 4, 3, 5);
    Tensor<TestType, 3> row(true, "A", 4, 3, 5);
    REQUIRE(row.is_row_major());
    fill(col);
    fill(row);

    {
        TensorFile out(from_col.path, TensorFile::Mode::Write);
        out.write("A", col);
    }
    {
        TensorFile out(from_row.path, TensorFile::Mode::Write);
        out.write("A", row);
    }

    REQUIRE(file_bytes(from_col.path) == file_bytes(from_row.path));
}

TEMPLATE_LIST_TEST_CASE("TensorFile - static-rank transfers keep logical order across layouts", "[TensorIO][Layout]",
                        testing::AllScalarTypes) {
    auto const [write_row_major, read_row_major] = GENERATE(table<bool, bool>({{true, false}, {false, true}, {true, true}}));
    CAPTURE(write_row_major, read_row_major);
    TempFile const tmp("static.etn");

    Tensor<TestType, 3> A(write_row_major, "A", 6, 4, 5);
    fill(A);

    SECTION("write, read") {
        {
            TensorFile out(tmp.path, TensorFile::Mode::Write);
            out.write("A", A);
        }
        Tensor<TestType, 3> B(read_row_major, "B", 1, 1, 1);
        TensorFile          in(tmp.path, TensorFile::Mode::Read);
        in.read("A", B);
        REQUIRE(B.is_row_major() == read_row_major);
        REQUIRE(B.dim(0) == 6);
        REQUIRE(B.dim(2) == 5);
        REQUIRE(mismatches(B) == 0);
    }

    SECTION("write_local, read_local") {
        {
            TensorFile out(tmp.path, TensorFile::Mode::Write);
            out.write_local("A", A, 0, 1);
        }
        Tensor<TestType, 3> B(read_row_major, "B", 6, 4, 5);
        TensorFile          in(tmp.path, TensorFile::Mode::Read);
        in.read_local("A", B, 0);
        REQUIRE(mismatches(B) == 0);
    }

    SECTION("read_slice") {
        {
            TensorFile out(tmp.path, TensorFile::Mode::Write);
            out.write("A", A);
        }
        Tensor<TestType, 3> S(read_row_major, "S", 1, 1, 1);
        TensorFile          in(tmp.path, TensorFile::Mode::Read);
        in.read_slice<TestType, 3>("A", S, {{{1, 5}, {2, 4}, {1, 4}}});
        REQUIRE(S.dim(0) == 4);
        REQUIRE(S.dim(1) == 2);
        REQUIRE(S.dim(2) == 3);
        REQUIRE(mismatches(S, 1, 2, 1) == 0);
    }

    SECTION("reserve, write_slice") {
        {
            TensorFile out(tmp.path, TensorFile::Mode::Write);
            out.reserve<TestType>("A", {6, 4, 5});
            // Two slabs that tile the entry, each a view of the pattern at its own origin.
            Tensor<TestType, 3> lo(write_row_major, "lo", 6, 4, 2);
            Tensor<TestType, 3> hi(write_row_major, "hi", 6, 4, 3);
            fill(lo);
            fill(hi, 0, 0, 2);
            out.write_slice<TestType, 3>("A", lo, {{{0, 6}, {0, 4}, {0, 2}}});
            out.write_slice<TestType, 3>("A", hi, {{{0, 6}, {0, 4}, {2, 5}}});
        }
        Tensor<TestType, 3> B(read_row_major, "B", 6, 4, 5);
        TensorFile          in(tmp.path, TensorFile::Mode::Read);
        in.read("A", B);
        REQUIRE(mismatches(B) == 0);
    }
}

TEMPLATE_LIST_TEST_CASE("TensorFile - RuntimeTensor transfers keep logical order across layouts", "[TensorIO][Layout][runtime]",
                        testing::AllScalarTypes) {
    auto const [write_row_major, read_row_major] = GENERATE(table<bool, bool>({{true, false}, {false, true}, {true, true}}));
    CAPTURE(write_row_major, read_row_major);
    TempFile const tmp("runtime.etn");

    RT<TestType> A("A", std::vector<size_t>{6, 4, 5}, write_row_major);
    fill(A);

    SECTION("write, read") {
        {
            TensorFile out(tmp.path, TensorFile::Mode::Write);
            out.write("A", A);
        }
        RT<TestType> B("B", std::vector<size_t>{0}, read_row_major);
        TensorFile   in(tmp.path, TensorFile::Mode::Read);
        in.read("A", B);
        REQUIRE(B.rank() == 3);
        REQUIRE(B.is_row_major() == read_row_major);
        REQUIRE(mismatches(B) == 0);
    }

    SECTION("read_slice") {
        {
            TensorFile out(tmp.path, TensorFile::Mode::Write);
            out.write("A", A);
        }
        RT<TestType> S("S", std::vector<size_t>{0}, read_row_major);
        TensorFile   in(tmp.path, TensorFile::Mode::Read);
        in.read_slice("A", S, {{1, 5}, {2, 4}, {1, 4}});
        REQUIRE(S.dim(0) == 4);
        REQUIRE(S.dim(2) == 3);
        REQUIRE(mismatches(S, 1, 2, 1) == 0);
    }

    SECTION("reserve, write_slice") {
        {
            TensorFile out(tmp.path, TensorFile::Mode::Write);
            out.reserve<TestType>("A", {6, 4, 5});
            RT<TestType> lo("lo", std::vector<size_t>{6, 4, 2}, write_row_major);
            RT<TestType> hi("hi", std::vector<size_t>{6, 4, 3}, write_row_major);
            fill(lo);
            fill(hi, 0, 0, 2);
            out.write_slice("A", lo, {{0, 6}, {0, 4}, {0, 2}});
            out.write_slice("A", hi, {{0, 6}, {0, 4}, {2, 5}});
        }
        RT<TestType> B("B", std::vector<size_t>{0}, read_row_major);
        TensorFile   in(tmp.path, TensorFile::Mode::Read);
        in.read("A", B);
        REQUIRE(mismatches(B) == 0);
    }
}

TEMPLATE_LIST_TEST_CASE("TensorFile - reordering spans several staging chunks", "[TensorIO][Layout]", testing::AllScalarTypes) {
    // Large enough that a row-major transfer stages its elements more than once, so each chunk
    // after the first has to resume the index walk where the previous one stopped.
    size_t const d0 = 130, d1 = 70, d2 = 1 + (3 * tensor_io::detail::REORDER_CHUNK_BYTES / (130 * 70 * sizeof(TestType)));
    REQUIRE(d0 * d1 * d2 * sizeof(TestType) > 2 * tensor_io::detail::REORDER_CHUNK_BYTES);
    TempFile const tmp("chunks.etn");

    Tensor<TestType, 3> A(true, "A", d0, d1, d2);
    fill(A);
    {
        TensorFile out(tmp.path, TensorFile::Mode::Write);
        out.write("A", A);
    }

    Tensor<TestType, 3> col(false, "col", 1, 1, 1);
    Tensor<TestType, 3> row(true, "row", 1, 1, 1);
    TensorFile          in(tmp.path, TensorFile::Mode::Read);
    in.read("A", col);
    in.read("A", row);
    REQUIRE(mismatches(col) == 0);
    REQUIRE(mismatches(row) == 0);
}

TEST_CASE("TensorFile - a row-major process reads a column-major process's file", "[TensorIO][Layout]") {
    // The reported case: the storage order comes from --einsums:row-major, not from the call.
    TempFile const tmp("option.etn");

    Tensor<double, 3> A("A", 3, 4, 2);
    REQUIRE_FALSE(A.is_row_major());
    fill(A);
    {
        TensorFile out(tmp.path, TensorFile::Mode::Write);
        out.write("A", A);
    }

    RestoreRowMajor const restore;
    config::set(option::RowMajor, true);
    Tensor<double, 3> B("B", 1, 1, 1);
    RT<double>        C("C", std::vector<size_t>{0});
    REQUIRE(B.is_row_major());
    REQUIRE(C.is_row_major());

    TensorFile in(tmp.path, TensorFile::Mode::Read);
    in.read("A", B);
    in.read("A", C);
    REQUIRE(mismatches(B) == 0);
    REQUIRE(mismatches(C) == 0);
}

TEMPLATE_LIST_TEST_CASE("DistributedTensorFile - transfers keep logical order across layouts", "[TensorIO][Layout][Distributed]",
                        testing::AllScalarTypes) {
    TempFile const tmp("dist.etn", true);

    // Every rank builds the same pattern, so the replicated write needs no broadcast.
    Tensor<TestType, 3> A(true, "A", 5, 3, 4);
    fill(A);
    Tensor<TestType, 3> local(true, "local", 2 + static_cast<size_t>(comm::world_rank()), 3, 2);
    fill(local);
    {
        DistributedTensorFile out(tmp.path, DistributedTensorFile::Mode::Write);
        out.write("A", A);
        out.write_local("local", local);
    }

    Tensor<TestType, 3> B(false, "B", 5, 3, 4);
    Tensor<TestType, 3> C(true, "C", 5, 3, 4);
    Tensor<TestType, 3> back(false, "back", 1, 1, 1);
    {
        DistributedTensorFile in(tmp.path, DistributedTensorFile::Mode::Read);
        in.read("A", B);
        in.read("A", C);
        in.read_local("local", back);
    }
    REQUIRE(mismatches(B) == 0);
    REQUIRE(mismatches(C) == 0);
    REQUIRE(back.dim(0) == local.dim(0));
    REQUIRE(mismatches(back) == 0);
}
