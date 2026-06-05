//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

/// @file TensorFileBasic.cpp
/// @brief Unit tests for the .etn tensor file format and TensorFile class.

#include <Einsums/Comm/Runtime.hpp>
#include <Einsums/Tensor/Tensor.hpp>
#include <Einsums/TensorIO/TensorFile.hpp>
#include <Einsums/TensorUtilities/CreateRandomTensor.hpp>
#include <Einsums/TensorUtilities/CreateZeroTensor.hpp>

#include <cstdio>

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
            // Shared path — all ranks use the same file
            path = "/tmp/einsums_test_" + name;
        } else {
            // Per-rank path to avoid contention
            path = "/tmp/einsums_test_r" + std::to_string(comm::world_rank()) + "_" + name;
        }
    }
    ~TempFile() { std::remove(path.c_str()); }
};
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

TEST_CASE("TensorFile - write and read double matrix", "[TensorIO][RoundTrip]") {
    TempFile const tmp("matrix.etn");

    auto A = create_random_tensor<double>("A", 10, 8);

    // Write
    {
        TensorFile out(tmp.path, TensorFile::Mode::Write);
        out.write("A", A);
        CHECK(out.num_tensors() == 1);
    }

    // Read back
    auto B = create_zero_tensor<double>("B", 10, 8);
    {
        TensorFile in(tmp.path, TensorFile::Mode::Read);
        CHECK(in.contains("A"));
        CHECK(in.num_tensors() == 1);
        CHECK(in.dims("A") == std::vector<size_t>{10, 8});
        CHECK(in.dtype("A") == DType::Float64);

        in.read("A", B);
    }

    // Verify
    for (size_t i = 0; i < A.size(); i++) {
        CHECK(A.data()[i] == B.data()[i]);
    }
}

TEST_CASE("TensorFile - write and read float vector", "[TensorIO][RoundTrip]") {
    TempFile const tmp("vector.etn");

    auto V = create_random_tensor<float>("V", 100);

    {
        TensorFile out(tmp.path, TensorFile::Mode::Write);
        out.write("V", V);
    }

    auto W = create_zero_tensor<float>("W", 100);
    {
        TensorFile in(tmp.path, TensorFile::Mode::Read);
        in.read("V", W);
    }

    for (size_t i = 0; i < V.size(); i++) {
        CHECK(V.data()[i] == W.data()[i]);
    }
}

TEST_CASE("TensorFile - write and read complex tensor", "[TensorIO][RoundTrip]") {
    TempFile const tmp("complex.etn");

    auto C = Tensor<std::complex<double>, 2>("C", 5, 5);
    for (size_t i = 0; i < 5; i++)
        for (size_t j = 0; j < 5; j++)
            C(i, j) = {static_cast<double>(i), static_cast<double>(j)};

    {
        TensorFile out(tmp.path, TensorFile::Mode::Write);
        out.write("C", C);
    }

    auto D = Tensor<std::complex<double>, 2>("D", 5, 5);
    {
        TensorFile in(tmp.path, TensorFile::Mode::Read);
        in.read("C", D);
    }

    for (size_t i = 0; i < 5; i++)
        for (size_t j = 0; j < 5; j++) {
            CHECK(D(i, j).real() == C(i, j).real());
            CHECK(D(i, j).imag() == C(i, j).imag());
        }
}

TEST_CASE("TensorFile - write and read rank-4 tensor", "[TensorIO][RoundTrip]") {
    TempFile const tmp("rank4.etn");

    auto T = create_random_tensor<double>("T", 4, 3, 5, 2);

    {
        TensorFile out(tmp.path, TensorFile::Mode::Write);
        out.write("T", T);
    }

    auto U = create_zero_tensor<double>("U", 4, 3, 5, 2);
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

TEST_CASE("TensorFile - read_slice 2D", "[TensorIO][Slice]") {
    TempFile const tmp("slice2d.etn");

    auto A = create_random_tensor<double>("A", 20, 16);

    {
        TensorFile out(tmp.path, TensorFile::Mode::Write);
        out.write("A", A);
    }

    // Read a 5x8 slice starting at (3, 4)
    auto slice = Tensor<double, 2>("slice", 5, 8);
    {
        TensorFile in(tmp.path, TensorFile::Mode::Read);
        in.read_slice<double, 2>("A", slice, {{{3, 8}, {4, 12}}});
    }

    for (size_t i = 0; i < 5; i++)
        for (size_t j = 0; j < 8; j++)
            CHECK(slice(i, j) == A(3 + i, 4 + j));
}

TEST_CASE("TensorFile - read_slice 1D", "[TensorIO][Slice]") {
    TempFile const tmp("slice1d.etn");

    auto V = create_random_tensor<double>("V", 100);

    {
        TensorFile out(tmp.path, TensorFile::Mode::Write);
        out.write("V", V);
    }

    auto slice = Tensor<double, 1>("slice", 20);
    {
        TensorFile in(tmp.path, TensorFile::Mode::Read);
        in.read_slice<double, 1>("V", slice, {{{10, 30}}});
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

// ═══════════════════════════════════════════════════════════════════════════════
// Checkpoint (ComputeGraph integration)
// ═══════════════════════════════════════════════════════════════════════════════

#include <Einsums/ComputeGraph.hpp>
#include <Einsums/TensorAlgebra/Detail/Index.hpp>
#include <Einsums/TensorIO/Checkpoint.hpp>

namespace cg = einsums::compute_graph;
using namespace einsums::index;

TEST_CASE("Checkpoint - save and restore graph tensors", "[TensorIO][Checkpoint]") {
    TempFile const tmp("checkpoint.etn");

    auto A = create_random_tensor<double>("A", 6, 8);
    auto B = create_random_tensor<double>("B", 8, 4);
    auto C = create_zero_tensor<double>("C", 6, 4);

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
    double const saved_val = C(0, 0);
    C(0, 0)                = -999.0;
    CHECK(C(0, 0) == -999.0);

    // Restore from checkpoint
    checkpoint::restore(tmp.path, graph);
    CHECK(C(0, 0) == saved_val);
}

TEST_CASE("Checkpoint - save subset of tensors", "[TensorIO][Checkpoint]") {
    TempFile const tmp("checkpoint_subset.etn");

    auto A = create_random_tensor<double>("A", 5, 5);
    auto B = create_random_tensor<double>("B", 5, 5);
    auto C = create_zero_tensor<double>("C", 5, 5);

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

// ═══════════════════════════════════════════════════════════════════════════════
// GraphIO integration (read_etn/write_etn in ComputeGraph capture)
// ═══════════════════════════════════════════════════════════════════════════════

#include <Einsums/TensorIO/GraphIO.hpp>

TEST_CASE("GraphIO - read_etn and write_etn in graph", "[TensorIO][GraphIO]") {
    TempFile const tmp_in("graphio_in.etn");
    TempFile const tmp_out("graphio_out.etn");

    // Create input file with tensor A
    auto A = create_random_tensor<double>("A", 6, 4);
    {
        TensorFile out(tmp_in.path, TensorFile::Mode::Write);
        out.write("A", A);
    }

    // Build a graph: read A from .etn → compute C = A * B → write C to .etn
    auto B = create_random_tensor<double>("B", 4, 5);
    auto C = create_zero_tensor<double>("C", 6, 5);

    cg::Graph graph("graphio_test");
    {
        cg::CaptureGuard const guard(graph);
        tensor_io::read_etn(tmp_in.path, "A", &A);
        cg::einsum("ik;kj->ij", &C, A, B);
        tensor_io::write_etn(tmp_out.path, "C", &C);
    }

    graph.execute();

    // Verify C was written to the output file
    auto C2 = create_zero_tensor<double>("C2", 6, 5);
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

TEST_CASE("GraphIO - read_etn outside capture executes immediately", "[TensorIO][GraphIO]") {
    TempFile const tmp("graphio_imm.etn");

    auto A = create_random_tensor<double>("A", 3, 3);
    {
        TensorFile out(tmp.path, TensorFile::Mode::Write);
        out.write("A", A);
    }

    // Zero A, then read back from .etn (outside capture = immediate)
    A.zero();
    CHECK(A(0, 0) == 0.0);

    // This should work outside capture mode
    {
        TensorFile in(tmp.path, TensorFile::Mode::Read);
        in.read("A", A);
    }

    CHECK(A(0, 0) != 0.0); // Restored from file
}

// ═══════════════════════════════════════════════════════════════════════════════
// Additional coverage
// ═══════════════════════════════════════════════════════════════════════════════

TEST_CASE("Checkpoint - save and restore Workspace", "[TensorIO][Checkpoint]") {
    TempFile const tmp("ckpt_ws.etn");

    cg::Workspace ws("test_ws");
    auto         &A = ws.declare_tensor<double, 2>(std::string("A"), 4, 4);
    auto         &B = ws.declare_tensor<double, 2>(std::string("B"), 3, 5);
    A.materialize();
    B.materialize();
    for (size_t idx = 0; idx < A.size(); idx++)
        A.data()[idx] = static_cast<double>(idx);
    for (size_t idx = 0; idx < B.size(); idx++)
        B.data()[idx] = static_cast<double>(idx * 10);

    checkpoint::save(tmp.path, ws);

    // Corrupt
    A.zero();
    B.zero();
    CHECK(A(0, 0) == 0.0);

    checkpoint::restore(tmp.path, ws);
    CHECK(A(0, 0) == 0.0); // First element is index 0
    CHECK(A(1, 0) == 1.0); // Second element (column-major: data[1])
    CHECK(B(0, 0) == 0.0);
    CHECK(B(1, 0) == 10.0); // Column-major: data[1] = 1 * 10
}

TEST_CASE("TensorFile - ReadWrite mode appends tensors", "[TensorIO][ReadWrite]") {
    TempFile const tmp("readwrite.etn");

    auto A = create_random_tensor<double>("A", 5, 5);
    auto B = create_random_tensor<double>("B", 3, 3);

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

        auto A2 = create_zero_tensor<double>("A2", 5, 5);
        auto B2 = create_zero_tensor<double>("B2", 3, 3);
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
