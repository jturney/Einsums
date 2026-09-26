//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

/// @file TensorFileBasic.cpp
/// @brief Unit tests for the .etn tensor file format and TensorFile class.

#include <Einsums/Comm/Runtime.hpp>
#include <Einsums/Tensor/RuntimeTensor.hpp>
#include <Einsums/Tensor/Tensor.hpp>
#include <Einsums/TensorIO/TensorFile.hpp>
#include <Einsums/TensorUtilities/CreateRandomTensor.hpp>
#include <Einsums/TensorUtilities/CreateZeroTensor.hpp>

#include <cstddef>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <tuple>
#include <type_traits>

#include <Einsums/Testing.hpp>

using namespace einsums;
using namespace einsums::tensor_io;

// Helper: create a temp file path and clean up after test.
// Uses per-rank paths for serial TensorFile tests to avoid contention under MPI.
namespace {
struct TempFile {
    std::string path;
    bool        distributed; ///< True for distributed tests (shared path across ranks)
    TempFile(std::string const &name = "test.etn", bool dist = false) : distributed(dist) {
        if (dist) {
            // Shared path: all ranks use the same file
            path = (std::filesystem::temp_directory_path() / ("einsums_test_" + name)).string();
        } else {
            // Per-rank path to avoid contention
            path = (std::filesystem::temp_directory_path() / ("einsums_test_r" + std::to_string(comm::world_rank()) + "_" + name)).string();
        }
    }
    ~TempFile() { std::remove(path.c_str()); }
};

/// Call @p fn with a value of each supported element type in turn.
template <typename Fn>
void for_each_scalar_type(Fn &&fn) {
    std::apply([&](auto... tags) { (fn(tags), ...); }, testing::AllScalarTypes{});
}

/// Overwrite one field of the named entry's record in a closed .etn file, to stand in for a damaged file.
template <typename Field>
void patch_entry_field(std::string const &path, std::string const &name, size_t field_offset, Field value) {
    std::fstream f(path, std::ios::in | std::ios::out | std::ios::binary);
    REQUIRE(f.good());
    FileHeader header{};
    f.read(reinterpret_cast<char *>(&header), sizeof(header));
    for (uint32_t i = 0; i < header.num_tensors; ++i) {
        auto const  record = static_cast<std::streamoff>(header.entry_table_offset + i * sizeof(TensorEntry));
        TensorEntry entry{};
        f.seekg(record);
        f.read(reinterpret_cast<char *>(&entry), sizeof(entry));
        if (entry.get_name() == name) {
            f.seekp(record + static_cast<std::streamoff>(field_offset));
            f.write(reinterpret_cast<char const *>(&value), sizeof(value));
            REQUIRE(f.good());
            return;
        }
    }
    FAIL("entry " << name << " not found in " << path);
}
} // namespace

// ═══════════════════════════════════════════════════════════════════════════════
// Format sanity
// ═══════════════════════════════════════════════════════════════════════════════

TEST_CASE("Format - header and entry sizes", "[TensorIO][Format]") {
    CHECK(sizeof(FileHeader) == 64);
    CHECK(sizeof(TensorEntry) == 160);
    CHECK(ETN_DATA_ALIGNMENT == 64);
}

TEST_CASE("Format - header init and validate", "[TensorIO][Format]") {
    FileHeader h;
    h.init();
    CHECK(h.is_valid());
    CHECK(h.version == ETN_VERSION);
    CHECK(h.num_tensors == 0);
}

TEST_CASE("Format - dtype mapping", "[TensorIO][Format]") {
    CHECK(dtype_for<float>() == DType::Float32);
    CHECK(dtype_for<double>() == DType::Float64);
    CHECK(dtype_for<std::complex<float>>() == DType::Complex64);
    CHECK(dtype_for<std::complex<double>>() == DType::Complex128);
    CHECK(dtype_for<int32_t>() == DType::Int32);
    CHECK(dtype_for<int64_t>() == DType::Int64);

    CHECK(dtype_size(DType::Float32) == 4);
    CHECK(dtype_size(DType::Float64) == 8);
    CHECK(dtype_size(DType::Complex128) == 16);
}

// ═══════════════════════════════════════════════════════════════════════════════
// Write and read round-trip
// ═══════════════════════════════════════════════════════════════════════════════

TEMPLATE_LIST_TEST_CASE("TensorFile - write and read matrix", "[TensorIO][RoundTrip]", testing::AllScalarTypes) {
    TempFile const tmp("matrix.etn");

    auto A = create_random_tensor<TestType>("A", 10, 8);

    // Write
    {
        TensorFile out(tmp.path, TensorFile::Mode::Write);
        out.write("A", A);
        CHECK(out.num_tensors() == 1);
    }

    // Read back
    auto B = create_zero_tensor<TestType>("B", 10, 8);
    {
        TensorFile in(tmp.path, TensorFile::Mode::Read);
        CHECK(in.contains("A"));
        CHECK(in.num_tensors() == 1);
        CHECK(in.dims("A") == std::vector<size_t>{10, 8});
        CHECK(in.dtype("A") == dtype_for<TestType>());

        in.read("A", B);
    }

    // Verify
    for (size_t i = 0; i < A.size(); i++) {
        CHECK(A.data()[i] == B.data()[i]);
    }
}

TEMPLATE_LIST_TEST_CASE("TensorFile - write and read vector", "[TensorIO][RoundTrip]", testing::AllScalarTypes) {
    TempFile const tmp("vector.etn");

    auto V = create_random_tensor<TestType>("V", 100);

    {
        TensorFile out(tmp.path, TensorFile::Mode::Write);
        out.write("V", V);
    }

    // Start from a different extent: read resizes the destination to the stored dims.
    auto W = create_zero_tensor<TestType>("W", 7);
    {
        TensorFile in(tmp.path, TensorFile::Mode::Read);
        in.read("V", W);
    }

    REQUIRE(W.dim(0) == 100);
    for (size_t i = 0; i < V.size(); i++) {
        CHECK(V.data()[i] == W.data()[i]);
    }
}

TEMPLATE_LIST_TEST_CASE("TensorFile - write and read rank-4 tensor", "[TensorIO][RoundTrip]", testing::AllScalarTypes) {
    TempFile const tmp("rank4.etn");

    auto T = create_random_tensor<TestType>("T", 4, 3, 5, 2);

    {
        TensorFile out(tmp.path, TensorFile::Mode::Write);
        out.write("T", T);
    }

    auto U = create_zero_tensor<TestType>("U", 4, 3, 5, 2);
    {
        TensorFile in(tmp.path, TensorFile::Mode::Read);
        in.read("T", U);
    }

    for (size_t i = 0; i < T.size(); i++) {
        CHECK(T.data()[i] == U.data()[i]);
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// Multiple tensors in one file
// ═══════════════════════════════════════════════════════════════════════════════

TEST_CASE("TensorFile - multiple tensors", "[TensorIO][Multi]") {
    TempFile const tmp("multi.etn");

    auto A = create_random_tensor<double>("A", 10, 10);
    auto B = create_random_tensor<float>("B", 20);
    auto C = create_random_tensor<double>("C", 3, 3, 3);

    {
        TensorFile out(tmp.path, TensorFile::Mode::Write);
        out.write("A", A);
        out.write("B", B);
        out.write("C", C);
        CHECK(out.num_tensors() == 3);
    }

    {
        TensorFile in(tmp.path, TensorFile::Mode::Read);
        CHECK(in.num_tensors() == 3);
        CHECK(in.contains("A"));
        CHECK(in.contains("B"));
        CHECK(in.contains("C"));
        CHECK_FALSE(in.contains("D"));

        auto names = in.tensor_names();
        CHECK(names.size() == 3);

        auto A2 = create_zero_tensor<double>("A2", 10, 10);
        auto B2 = create_zero_tensor<float>("B2", 20);
        auto C2 = create_zero_tensor<double>("C2", 3, 3, 3);

        in.read("A", A2);
        in.read("B", B2);
        in.read("C", C2);

        for (size_t i = 0; i < A.size(); i++)
            CHECK(A.data()[i] == A2.data()[i]);
        for (size_t i = 0; i < B.size(); i++)
            CHECK(B.data()[i] == B2.data()[i]);
        for (size_t i = 0; i < C.size(); i++)
            CHECK(C.data()[i] == C2.data()[i]);
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// Slice reads
// ═══════════════════════════════════════════════════════════════════════════════

TEMPLATE_LIST_TEST_CASE("TensorFile - read_slice 2D", "[TensorIO][Slice]", testing::AllScalarTypes) {
    TempFile const tmp("slice2d.etn");

    auto A = create_random_tensor<TestType>("A", 20, 16);

    {
        TensorFile out(tmp.path, TensorFile::Mode::Write);
        out.write("A", A);
    }

    // Read a 5x8 slice starting at (3, 4)
    auto slice = Tensor<TestType, 2>("slice", 5, 8);
    {
        TensorFile in(tmp.path, TensorFile::Mode::Read);
        in.read_slice<TestType, 2>("A", slice, {{{3, 8}, {4, 12}}});
    }

    for (size_t i = 0; i < 5; i++)
        for (size_t j = 0; j < 8; j++)
            CHECK(slice(i, j) == A(3 + i, 4 + j));
}

TEMPLATE_LIST_TEST_CASE("TensorFile - read_slice 1D", "[TensorIO][Slice]", testing::AllScalarTypes) {
    TempFile const tmp("slice1d.etn");

    auto V = create_random_tensor<TestType>("V", 100);

    {
        TensorFile out(tmp.path, TensorFile::Mode::Write);
        out.write("V", V);
    }

    auto slice = Tensor<TestType, 1>("slice", 20);
    {
        TensorFile in(tmp.path, TensorFile::Mode::Read);
        in.read_slice<TestType, 1>("V", slice, {{{10, 30}}});
    }

    for (size_t i = 0; i < 20; i++)
        CHECK(slice(i) == V(10 + i));
}

// ═══════════════════════════════════════════════════════════════════════════════
// Error handling
// ═══════════════════════════════════════════════════════════════════════════════

TEST_CASE("TensorFile - read nonexistent tensor throws", "[TensorIO][Error]") {
    TempFile const tmp("empty.etn");

    {
        TensorFile const out(tmp.path, TensorFile::Mode::Write);
        // Don't write anything
    }

    auto       A = Tensor<double, 2>("A", 3, 3);
    TensorFile in(tmp.path, TensorFile::Mode::Read);
    REQUIRE_THROWS(in.read("nonexistent", A));
}

// Every typed copy out of (or into) an entry compares the stored dtype with the tensor's, and a
// compile-time rank with the stored rank, before it touches memory. Without the check a
// complex128 entry read into a float32 tensor wrote four times the destination's size, and
// same-width pairs (float64 and complex64) silently reinterpreted the bits.
TEMPLATE_LIST_TEST_CASE("TensorFile - every read rejects a dtype mismatch", "[TensorIO][Error][DType]", testing::AllScalarTypes) {
    TempFile const tmp("dtype_mismatch.etn");

    auto A = create_random_tensor<TestType>("A", 4, 3);
    {
        TensorFile out(tmp.path, TensorFile::Mode::Write);
        out.write("A", A);
        out.write_local("L", A, 0, 1);
    }

    TensorFile        in(tmp.path, TensorFile::Mode::Read);
    std::string const stored(dtype_name(dtype_for<TestType>()));

    for_each_scalar_type([&](auto tag) {
        using U = decltype(tag);
        if constexpr (!std::is_same_v<U, TestType>) {
            std::string const requested(dtype_name(dtype_for<U>()));
            INFO("stored " << stored << ", requested " << requested);

            auto dst = create_zero_tensor<U>("dst", 2, 2);
            REQUIRE_THROWS_AS(in.read("A", dst), std::invalid_argument);
            REQUIRE_THROWS_WITH(in.read("A", dst), Catch::Matchers::ContainsSubstring("TensorFile::read") &&
                                                       Catch::Matchers::ContainsSubstring("'A'") &&
                                                       Catch::Matchers::ContainsSubstring("holds " + stored + " data") &&
                                                       Catch::Matchers::ContainsSubstring("the tensor holds " + requested));
            REQUIRE_THROWS_AS((in.read_slice<U, 2>("A", dst, {{{0, 2}, {0, 2}}})), std::invalid_argument);
            REQUIRE_THROWS_AS(in.read_local("L", dst, 0), std::invalid_argument);

            // The rejection happens before the destination is resized or written.
            REQUIRE(dst.dim(0) == 2);
            REQUIRE(dst.dim(1) == 2);
            for (size_t i = 0; i < dst.size(); ++i) {
                CHECK(dst.data()[i] == U{0});
            }

            GeneralRuntimeTensor<U, std::allocator<U>> rt("rt", std::vector<size_t>{2, 2});
            rt.zero();
            REQUIRE_THROWS_AS(in.read("A", rt), std::invalid_argument);
            REQUIRE_THROWS_AS(in.read_slice("A", rt, {{0, 2}, {0, 2}}), std::invalid_argument);
            REQUIRE(rt.rank() == 2);
            REQUIRE(rt.dim(0) == 2);
            REQUIRE(rt.dim(1) == 2);
        }
    });
}

TEMPLATE_LIST_TEST_CASE("TensorFile - write_slice rejects a dtype mismatch and leaves the entry intact", "[TensorIO][Error][DType]",
                        testing::AllScalarTypes) {
    TempFile const tmp("dtype_mismatch_write.etn");

    auto A = create_random_tensor<TestType>("A", 4, 3);
    {
        TensorFile out(tmp.path, TensorFile::Mode::Write);
        out.write("A", A);
    }

    {
        TensorFile rw(tmp.path, TensorFile::Mode::ReadWrite);
        for_each_scalar_type([&](auto tag) {
            using U = decltype(tag);
            if constexpr (!std::is_same_v<U, TestType>) {
                INFO("requested " << dtype_name(dtype_for<U>()));
                auto patch = create_random_tensor<U>("patch", 2, 2);
                REQUIRE_THROWS_AS((rw.write_slice<U, 2>("A", patch, {{{0, 2}, {0, 2}}})), std::invalid_argument);

                GeneralRuntimeTensor<U, std::allocator<U>> rt("rt", std::vector<size_t>{2, 2});
                rt.zero();
                REQUIRE_THROWS_AS(rw.write_slice("A", rt, {{0, 2}, {0, 2}}), std::invalid_argument);
            }
        });
    }

    auto back = create_zero_tensor<TestType>("back", 4, 3);
    {
        TensorFile in(tmp.path, TensorFile::Mode::Read);
        in.read("A", back);
    }
    for (size_t i = 0; i < A.size(); ++i) {
        CHECK(back.data()[i] == A.data()[i]);
    }
}

TEST_CASE("TensorFile - static-rank reads and writes reject a rank mismatch", "[TensorIO][Error][Rank]") {
    TempFile const tmp("rank_mismatch.etn");

    auto A = create_random_tensor<double>("A", 4, 3);
    {
        TensorFile out(tmp.path, TensorFile::Mode::Write);
        out.write("A", A);
        out.write_local("L", A, 0, 1);
    }

    {
        TensorFile in(tmp.path, TensorFile::Mode::Read);

        auto three = create_zero_tensor<double>("three", 2, 2, 2);
        REQUIRE_THROWS_AS(in.read("A", three), std::invalid_argument);
        REQUIRE_THROWS_WITH(in.read("A", three), Catch::Matchers::ContainsSubstring("'A'") &&
                                                     Catch::Matchers::ContainsSubstring("has rank 2") &&
                                                     Catch::Matchers::ContainsSubstring("the tensor has rank 3"));
        REQUIRE_THROWS_AS((in.read_slice<double, 3>("A", three, {{{0, 1}, {0, 1}, {0, 1}}})), std::invalid_argument);
        REQUIRE_THROWS_AS(in.read_local("L", three, 0), std::invalid_argument);
        REQUIRE(three.dim(0) == 2);
        REQUIRE(three.dim(2) == 2);

        auto one = create_zero_tensor<double>("one", 5);
        REQUIRE_THROWS_AS(in.read("A", one), std::invalid_argument);
        REQUIRE_THROWS_AS((in.read_slice<double, 1>("A", one, {{{0, 2}}})), std::invalid_argument);
        REQUIRE_THROWS_AS(in.read_local("L", one, 0), std::invalid_argument);
        REQUIRE(one.dim(0) == 5);

        // Runtime-rank reads take the stored rank, but the ranges must still name every dimension.
        GeneralRuntimeTensor<double, std::allocator<double>> rt("rt", std::vector<size_t>{2, 2});
        REQUIRE_THROWS_AS(in.read_slice("A", rt, {{0, 1}, {0, 1}, {0, 1}}), std::invalid_argument);
    }

    {
        TensorFile rw(tmp.path, TensorFile::Mode::ReadWrite);
        auto       patch = create_random_tensor<double>("patch", 2);
        REQUIRE_THROWS_AS((rw.write_slice<double, 1>("A", patch, {{{0, 2}}})), std::invalid_argument);
    }
}

TEST_CASE("TensorFile - slice ranges must lie inside the stored entry", "[TensorIO][Error][Slice]") {
    TempFile const tmp("slab_bounds.etn");

    auto A = create_random_tensor<double>("A", 4, 3);
    {
        TensorFile out(tmp.path, TensorFile::Mode::Write);
        out.write("A", A);
    }

    {
        TensorFile        in(tmp.path, TensorFile::Mode::Read);
        Tensor<double, 2> slab("slab", 1, 1);
        REQUIRE_THROWS_AS((in.read_slice<double, 2>("A", slab, {{{0, 5}, {0, 3}}})), std::out_of_range);
        REQUIRE_THROWS_AS((in.read_slice<double, 2>("A", slab, {{{0, 4}, {2, 4}}})), std::out_of_range);
        REQUIRE_THROWS_AS((in.read_slice<double, 2>("A", slab, {{{3, 1}, {0, 3}}})), std::invalid_argument);
        REQUIRE(slab.dim(0) == 1);

        GeneralRuntimeTensor<double, std::allocator<double>> rt("rt", std::vector<size_t>{1, 1});
        REQUIRE_THROWS_AS(in.read_slice("A", rt, {{0, 4}, {1, 4}}), std::out_of_range);
        REQUIRE_THROWS_AS(in.read_slice("A", rt, {{2, 1}, {0, 1}}), std::invalid_argument);
    }

    {
        TensorFile rw(tmp.path, TensorFile::Mode::ReadWrite);
        auto       patch = create_random_tensor<double>("patch", 2, 2);
        REQUIRE_THROWS_AS((rw.write_slice<double, 2>("A", patch, {{{3, 5}, {0, 2}}})), std::out_of_range);

        GeneralRuntimeTensor<double, std::allocator<double>> rt("rt", std::vector<size_t>{2, 2});
        rt.zero();
        REQUIRE_THROWS_AS(rw.write_slice("A", rt, {{0, 2}, {2, 4}}), std::out_of_range);
    }

    auto back = create_zero_tensor<double>("back", 4, 3);
    {
        TensorFile in(tmp.path, TensorFile::Mode::Read);
        in.read("A", back);
    }
    for (size_t i = 0; i < A.size(); ++i) {
        CHECK(back.data()[i] == A.data()[i]);
    }
}

TEST_CASE("TensorFile - read rejects a damaged entry record", "[TensorIO][Error]") {
    TempFile const tmp("damaged.etn");

    auto A = create_random_tensor<double>("A", 4, 3);
    {
        TensorFile out(tmp.path, TensorFile::Mode::Write);
        out.write("A", A);
        out.write("B", A);
    }

    // "A" claims twice the bytes its dims hold; "B" carries a dtype code outside the enum.
    patch_entry_field<uint64_t>(tmp.path, "A", offsetof(TensorEntry, data_size), uint64_t{4 * 3 * sizeof(double) * 2});
    patch_entry_field<uint8_t>(tmp.path, "B", offsetof(TensorEntry, dtype), uint8_t{200});

    TensorFile in(tmp.path, TensorFile::Mode::Read);
    auto       dst = create_zero_tensor<double>("dst", 4, 3);
    REQUIRE_THROWS_AS(in.read("A", dst), std::runtime_error);
    REQUIRE_THROWS_WITH(in.read("A", dst), Catch::Matchers::ContainsSubstring("damaged") && Catch::Matchers::ContainsSubstring("192"));
    REQUIRE_THROWS_WITH(in.read("B", dst), Catch::Matchers::ContainsSubstring("unknown dtype code 200"));

    GeneralRuntimeTensor<double, std::allocator<double>> rt("rt", std::vector<size_t>{1});
    REQUIRE_THROWS_AS(in.read("A", rt), std::runtime_error);
    REQUIRE(rt.dim(0) == 1);
}

// ═══════════════════════════════════════════════════════════════════════════════
// Distributed I/O (works with both mock and real MPI)
// ═══════════════════════════════════════════════════════════════════════════════

#include <Einsums/Comm/Collectives.hpp>
#include <Einsums/Comm/Runtime.hpp>
#include <Einsums/TensorIO/DistributedTensorFile.hpp>

#include <span>

TEST_CASE("DistributedTensorFile - write and read replicated", "[TensorIO][Distributed]") {
    TempFile const tmp("dist_repl.etn", true);

    auto A = create_random_tensor<double>("A", 8, 6);
    // NOLINTNEXTLINE(bugprone-unused-return-value)
    (void)comm::broadcast<double>(std::span<double>(A.data(), A.size()), 0);

    // Write (collective)
    {
        DistributedTensorFile out(tmp.path, DistributedTensorFile::Mode::Write);
        out.write("A", A);
    }

    // Read back (collective)
    auto B = create_zero_tensor<double>("B", 8, 6);
    {
        DistributedTensorFile in(tmp.path, DistributedTensorFile::Mode::Read);
        CHECK(in.contains("A"));
        in.read("A", B);
    }

    for (size_t i = 0; i < A.size(); i++) {
        CHECK(A.data()[i] == B.data()[i]);
    }
}

TEST_CASE("DistributedTensorFile - write_local and read_local", "[TensorIO][Distributed]") {
    TempFile const tmp("dist_local.etn", true);
    int const      rank   = comm::world_rank();
    int const      nprocs = comm::world_size();

    // Each rank has a different-sized local tensor
    size_t const local_rows = 10 + static_cast<size_t>(rank) * 2; // rank 0: 10, rank 1: 12, etc.
    auto         local      = create_random_tensor<double>("local", local_rows, 8);

    // Write (collective)
    {
        DistributedTensorFile out(tmp.path, DistributedTensorFile::Mode::Write);
        out.write_local("data", local);
    }

    // Read back (collective)
    auto restored = create_zero_tensor<double>("restored", local_rows, 8);
    {
        DistributedTensorFile in(tmp.path, DistributedTensorFile::Mode::Read);
        in.read_local("data", restored);
    }

    for (size_t i = 0; i < local.size(); i++) {
        CHECK(local.data()[i] == restored.data()[i]);
    }
}

TEST_CASE("DistributedTensorFile - mixed replicated and local", "[TensorIO][Distributed]") {
    TempFile const tmp("dist_mixed.etn", true);
    int const      rank = comm::world_rank();

    auto global = create_random_tensor<double>("global", 5, 5);
    // NOLINTNEXTLINE(bugprone-unused-return-value)
    (void)comm::broadcast<double>(std::span<double>(global.data(), global.size()), 0);
    auto local = create_random_tensor<double>("local", 3 + static_cast<size_t>(rank), 4);

    {
        DistributedTensorFile out(tmp.path, DistributedTensorFile::Mode::Write);
        out.write("global_tensor", global);
        out.write_local("local_tensor", local);
    }

    auto global2 = create_zero_tensor<double>("global2", 5, 5);
    auto local2  = create_zero_tensor<double>("local2", 3 + static_cast<size_t>(rank), 4);

    {
        DistributedTensorFile in(tmp.path, DistributedTensorFile::Mode::Read);
        CHECK(in.contains("global_tensor"));
        in.read("global_tensor", global2);
        in.read_local("local_tensor", local2);
    }

    for (size_t i = 0; i < global.size(); i++)
        CHECK(global.data()[i] == global2.data()[i]);
    for (size_t i = 0; i < local.size(); i++)
        CHECK(local.data()[i] == local2.data()[i]);
}

TEST_CASE("DistributedTensorFile - reads reject dtype and rank mismatches", "[TensorIO][Distributed][Error]") {
    TempFile const tmp("dist_mismatch.etn", true);

    auto global = create_random_tensor<double>("global", 5, 4);
    auto local  = create_random_tensor<double>("local", 3, 4);
    {
        DistributedTensorFile out(tmp.path, DistributedTensorFile::Mode::Write);
        out.write("G", global);
        out.write_local("L", local);
    }

    DistributedTensorFile in(tmp.path, DistributedTensorFile::Mode::Read);

    auto wrong_type = create_zero_tensor<std::complex<double>>("wrong_type", 2, 2);
    REQUIRE_THROWS_AS(in.read("G", wrong_type), std::invalid_argument);
    REQUIRE_THROWS_WITH(in.read("G", wrong_type), Catch::Matchers::ContainsSubstring("DistributedTensorFile::read") &&
                                                      Catch::Matchers::ContainsSubstring("holds float64 data") &&
                                                      Catch::Matchers::ContainsSubstring("the tensor holds complex128"));
    REQUIRE_THROWS_AS(in.read_local("L", wrong_type), std::invalid_argument);
    REQUIRE(wrong_type.dim(0) == 2);

    auto narrow = create_zero_tensor<float>("narrow", 2, 2);
    REQUIRE_THROWS_AS(in.read("G", narrow), std::invalid_argument);
    REQUIRE_THROWS_AS(in.read_local("L", narrow), std::invalid_argument);

    auto wrong_rank = create_zero_tensor<double>("wrong_rank", 2, 2, 2);
    REQUIRE_THROWS_AS(in.read("G", wrong_rank), std::invalid_argument);
    REQUIRE_THROWS_AS(in.read_local("L", wrong_rank), std::invalid_argument);
    REQUIRE(wrong_rank.dim(2) == 2);
}

// ═══════════════════════════════════════════════════════════════════════════════
// Checkpoint (ComputeGraph integration)
// ═══════════════════════════════════════════════════════════════════════════════

#include <Einsums/ComputeGraph.hpp>
#include <Einsums/TensorAlgebra/Detail/Index.hpp>
#include <Einsums/TensorIO/Checkpoint.hpp>

namespace cg = einsums::compute_graph;
using namespace einsums::index;

TEMPLATE_LIST_TEST_CASE("Checkpoint - save and restore graph tensors", "[TensorIO][Checkpoint]", testing::AllScalarTypes) {
    TempFile const tmp("checkpoint.etn");

    auto A = create_random_tensor<TestType>("A", 6, 8);
    auto B = create_random_tensor<TestType>("B", 8, 4);
    auto C = create_zero_tensor<TestType>("C", 6, 4);

    // Build a graph and execute
    cg::Graph graph("ckpt_test");
    {
        cg::CaptureGuard const guard(graph);
        cg::einsum("ik;kj->ij", &C, A, B);
    }
    graph.execute();

    // Save checkpoint
    checkpoint::save(tmp.path, graph);

    // Corrupt C to verify restore works
    TestType const saved_val = C(0, 0);
    C(0, 0)                  = TestType{-999};
    CHECK(C(0, 0) == TestType{-999});

    // Restore from checkpoint
    checkpoint::restore(tmp.path, graph);
    CHECK(C(0, 0) == saved_val);
}

TEMPLATE_LIST_TEST_CASE("Checkpoint - save subset of tensors", "[TensorIO][Checkpoint]", testing::AllScalarTypes) {
    TempFile const tmp("checkpoint_subset.etn");

    auto A = create_random_tensor<TestType>("A", 5, 5);
    auto B = create_random_tensor<TestType>("B", 5, 5);
    auto C = create_zero_tensor<TestType>("C", 5, 5);

    cg::Graph graph("ckpt_subset");
    {
        cg::CaptureGuard const guard(graph);
        cg::einsum("ik;kj->ij", &C, A, B);
    }
    graph.execute();

    // Save only C
    checkpoint::save(tmp.path, graph, {"C"});

    // Verify only C is in the file
    TensorFile const in(tmp.path, TensorFile::Mode::Read);
    CHECK(in.contains("C"));
    CHECK_FALSE(in.contains("A"));
    CHECK_FALSE(in.contains("B"));
}

TEST_CASE("Checkpoint - restore rejects a checkpoint of another dtype", "[TensorIO][Checkpoint][Error]") {
    TempFile const tmp("checkpoint_dtype.etn");

    auto A = create_random_tensor<double>("A", 3, 4);
    auto B = create_random_tensor<double>("B", 4, 2);
    auto C = create_zero_tensor<double>("C", 3, 2);

    cg::Graph graph("ckpt_double");
    {
        cg::CaptureGuard const guard(graph);
        cg::einsum("ik;kj->ij", &C, A, B);
    }
    graph.execute();
    checkpoint::save(tmp.path, graph);

    // A graph over the same names in single precision cannot take the double checkpoint.
    auto Af = create_zero_tensor<float>("A", 3, 4);
    auto Bf = create_zero_tensor<float>("B", 4, 2);
    auto Cf = create_zero_tensor<float>("C", 3, 2);

    cg::Graph single("ckpt_float");
    {
        cg::CaptureGuard const guard(single);
        cg::einsum("ik;kj->ij", &Cf, Af, Bf);
    }
    REQUIRE_THROWS_AS(checkpoint::restore(tmp.path, single), std::invalid_argument);
    for (size_t i = 0; i < Cf.size(); ++i) {
        CHECK(Cf.data()[i] == 0.0F);
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// GraphIO integration (read_etn/write_etn in ComputeGraph capture)
// ═══════════════════════════════════════════════════════════════════════════════

#include <Einsums/TensorIO/GraphIO.hpp>

TEMPLATE_LIST_TEST_CASE("GraphIO - read_etn and write_etn in graph", "[TensorIO][GraphIO]", testing::AllScalarTypes) {
    TempFile const tmp_in("graphio_in.etn");
    TempFile const tmp_out("graphio_out.etn");

    // Create input file with tensor A
    auto A = create_random_tensor<TestType>("A", 6, 4);
    {
        TensorFile out(tmp_in.path, TensorFile::Mode::Write);
        out.write("A", A);
    }

    // Build a graph: read A from .etn → compute C = A * B → write C to .etn
    auto B = create_random_tensor<TestType>("B", 4, 5);
    auto C = create_zero_tensor<TestType>("C", 6, 5);

    cg::Graph graph("graphio_test");
    {
        cg::CaptureGuard const guard(graph);
        tensor_io::read_etn(tmp_in.path, "A", &A);
        cg::einsum("ik;kj->ij", &C, A, B);
        tensor_io::write_etn(tmp_out.path, "C", &C);
    }

    graph.execute();

    // Verify C was written to the output file
    auto C2 = create_zero_tensor<TestType>("C2", 6, 5);
    {
        TensorFile in(tmp_out.path, TensorFile::Mode::Read);
        CHECK(in.contains("C"));
        in.read("C", C2);
    }

    // C and C2 should match
    for (size_t idx = 0; idx < C.size(); idx++) {
        CHECK(C.data()[idx] == C2.data()[idx]);
    }
}

TEST_CASE("GraphIO - read_etn rejects a dtype mismatch", "[TensorIO][GraphIO][Error]") {
    TempFile const tmp("graphio_dtype.etn");

    auto A = create_random_tensor<std::complex<double>>("A", 2, 3);
    {
        TensorFile out(tmp.path, TensorFile::Mode::Write);
        out.write("A", A);
    }

    // Eager: the read runs at the call.
    auto Y = create_zero_tensor<float>("Y", 2, 3);
    REQUIRE_THROWS_AS(tensor_io::read_etn(tmp.path, "A", &Y), std::invalid_argument);

    // Captured: the DiskRead node's executor runs the same checked read at replay.
    cg::Graph graph("graphio_dtype");
    {
        cg::CaptureGuard const guard(graph);
        tensor_io::read_etn(tmp.path, "A", &Y);
    }
    REQUIRE_THROWS_AS(graph.execute(), std::invalid_argument);
    for (size_t i = 0; i < Y.size(); ++i) {
        CHECK(Y.data()[i] == 0.0F);
    }
}

TEMPLATE_LIST_TEST_CASE("GraphIO - read_etn outside capture executes immediately", "[TensorIO][GraphIO]", testing::AllScalarTypes) {
    TempFile const tmp("graphio_imm.etn");

    auto A = create_random_tensor<TestType>("A", 3, 3);
    {
        TensorFile out(tmp.path, TensorFile::Mode::Write);
        out.write("A", A);
    }

    // Zero A, then read back from .etn (outside capture = immediate)
    A.zero();
    CHECK(A(0, 0) == TestType{0});

    // This should work outside capture mode
    {
        TensorFile in(tmp.path, TensorFile::Mode::Read);
        in.read("A", A);
    }

    CHECK(A(0, 0) != TestType{0}); // Restored from file
}

// ═══════════════════════════════════════════════════════════════════════════════
// Additional coverage
// ═══════════════════════════════════════════════════════════════════════════════

TEMPLATE_LIST_TEST_CASE("Checkpoint - save and restore Workspace", "[TensorIO][Checkpoint]", testing::AllScalarTypes) {
    TempFile const tmp("ckpt_ws.etn");

    cg::Workspace ws("test_ws");
    auto         &A = ws.declare_tensor<TestType, 2>(std::string("A"), 4, 4);
    auto         &B = ws.declare_tensor<TestType, 2>(std::string("B"), 3, 5);
    A.materialize();
    B.materialize();
    for (size_t idx = 0; idx < A.size(); idx++)
        A.data()[idx] = static_cast<TestType>(idx);
    for (size_t idx = 0; idx < B.size(); idx++)
        B.data()[idx] = static_cast<TestType>(idx * 10);

    checkpoint::save(tmp.path, ws);

    // Corrupt
    A.zero();
    B.zero();
    CHECK(A(1, 0) == TestType{0});

    checkpoint::restore(tmp.path, ws);
    CHECK(A(0, 0) == TestType{0}); // First element is index 0
    CHECK(A(1, 0) == TestType{1}); // Second element (column-major: data[1])
    CHECK(B(0, 0) == TestType{0});
    CHECK(B(1, 0) == TestType{10}); // Column-major: data[1] = 1 * 10
}

TEMPLATE_LIST_TEST_CASE("TensorFile - ReadWrite mode appends tensors", "[TensorIO][ReadWrite]", testing::AllScalarTypes) {
    TempFile const tmp("readwrite.etn");

    auto A = create_random_tensor<TestType>("A", 5, 5);
    auto B = create_random_tensor<TestType>("B", 3, 3);

    // Write A first
    {
        TensorFile out(tmp.path, TensorFile::Mode::Write);
        out.write("A", A);
    }

    // Append B in ReadWrite mode
    {
        TensorFile rw(tmp.path, TensorFile::Mode::ReadWrite);
        CHECK(rw.contains("A"));
        CHECK(rw.num_tensors() == 1);
        rw.write("B", B);
        CHECK(rw.num_tensors() == 2);
    }

    // Verify both are readable
    {
        TensorFile in(tmp.path, TensorFile::Mode::Read);
        CHECK(in.contains("A"));
        CHECK(in.contains("B"));
        CHECK(in.num_tensors() == 2);

        auto A2 = create_zero_tensor<TestType>("A2", 5, 5);
        auto B2 = create_zero_tensor<TestType>("B2", 3, 3);
        in.read("A", A2);
        in.read("B", B2);

        for (size_t idx = 0; idx < A.size(); idx++)
            CHECK(A.data()[idx] == A2.data()[idx]);
        for (size_t idx = 0; idx < B.size(); idx++)
            CHECK(B.data()[idx] == B2.data()[idx]);
    }
}

TEST_CASE("TensorFile - tensor_names lists all tensors", "[TensorIO][Query]") {
    TempFile const tmp("query.etn");

    auto X = create_random_tensor<double>("X", 2, 2);
    auto Y = create_random_tensor<float>("Y", 3);
    auto Z = create_random_tensor<double>("Z", 4, 4, 4);

    {
        TensorFile out(tmp.path, TensorFile::Mode::Write);
        out.write("X", X);
        out.write("Y", Y);
        out.write("Z", Z);
    }

    TensorFile const in(tmp.path, TensorFile::Mode::Read);
    auto             names = in.tensor_names();
    CHECK(names.size() == 3);

    // Check dims query
    CHECK(in.dims("X") == std::vector<size_t>{2, 2});
    CHECK(in.dims("Y") == std::vector<size_t>{3});
    CHECK(in.dims("Z") == std::vector<size_t>{4, 4, 4});

    CHECK(in.dtype("X") == DType::Float64);
    CHECK(in.dtype("Y") == DType::Float32);
}
