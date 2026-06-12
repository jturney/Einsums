//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

/**
 * @file TensorIOExample.cpp
 * @brief Example: .etn tensor file I/O with checkpointing.
 *
 * Demonstrates:
 * 1. Writing tensors to a .etn file
 * 2. Reading part of a tensor back with a slice
 * 3. Checkpointing a ComputeGraph and restoring
 * 4. Using read_etn/write_etn inside graph capture
 */

#include <Einsums/ComputeGraph.hpp>
#include <Einsums/Print.hpp>
#include <Einsums/Runtime.hpp>
#include <Einsums/Tensor/Tensor.hpp>
#include <Einsums/TensorIO/Checkpoint.hpp>
#include <Einsums/TensorIO/GraphIO.hpp>
#include <Einsums/TensorIO/TensorFile.hpp>
#include <Einsums/TensorUtilities/CreateRandomTensor.hpp>
#include <Einsums/TensorUtilities/CreateZeroTensor.hpp>

using namespace einsums;
using namespace einsums::index;
namespace cg   = einsums::compute_graph;
namespace tio  = einsums::tensor_io;
namespace ckpt = einsums::tensor_io::checkpoint;

int einsums_main() {
    constexpr size_t N = 20;

    // ── 1. Basic TensorFile I/O ──────────────────────────────────────────
    einsums::println("=== 1. Basic TensorFile I/O ===");

    auto A = create_random_tensor<double>("A", N, N);
    auto B = create_random_tensor<double>("B", N, N);

    {
        tio::TensorFile out("/tmp/einsums_example.etn", tio::TensorFile::Mode::Write);
        out.write("A", A);
        out.write("B", B);
        einsums::println("  Wrote {} tensors to /tmp/einsums_example.etn", out.num_tensors());
    }

    {
        tio::TensorFile in("/tmp/einsums_example.etn", tio::TensorFile::Mode::Read);
        einsums::println("  File contains: {}", fmt::join(in.tensor_names(), ", "));
        einsums::println("  A dims: {}x{}", in.dims("A")[0], in.dims("A")[1]);
    }

    // ── 2. Slice read ────────────────────────────────────────────────────
    einsums::println("\n=== 2. Slice read ===");

    auto slice = Tensor<double, 2>("slice", 5, 5);
    {
        tio::TensorFile in("/tmp/einsums_example.etn", tio::TensorFile::Mode::Read);
        in.read_slice<double, 2>("A", slice, {{{0, 5}, {0, 5}}});
    }
    einsums::println("  Read 5x5 slice of A: A(0,0) = {:.6f}, slice(0,0) = {:.6f}", A(0, 0), slice(0, 0));
    einsums::println("  Match: {}", A(0, 0) == slice(0, 0) ? "YES" : "NO");

    // ── 3. Checkpoint a ComputeGraph ─────────────────────────────────────
    einsums::println("\n=== 3. Checkpoint ===");

    auto C = create_zero_tensor<double>("C", N, N);

    cg::Graph graph("example");
    {
        cg::CaptureGuard const guard(graph);
        cg::einsum("ik;kj->ij", &C, A, B);
    }
    graph.execute();

    einsums::println("  C(0,0) = {:.6f} (after compute)", C(0, 0));

    // Save checkpoint
    ckpt::save("/tmp/einsums_checkpoint.etn", graph);
    einsums::println("  Saved checkpoint");

    // Corrupt C, then restore
    double const saved = C(0, 0);
    C.zero();
    einsums::println("  C(0,0) = {:.6f} (after zeroing)", C(0, 0));

    ckpt::restore("/tmp/einsums_checkpoint.etn", graph);
    einsums::println("  C(0,0) = {:.6f} (after restore)", C(0, 0));
    einsums::println("  Restored correctly: {}", C(0, 0) == saved ? "YES" : "NO");

    // ── 4. GraphIO: read/write in capture ────────────────────────────────
    einsums::println("\n=== 4. GraphIO ===");

    auto D = create_zero_tensor<double>("D", N, N);

    cg::Graph graph2("graphio_example");
    {
        cg::CaptureGuard const guard(graph2);
        // Read A from file → compute D = A * B → write D to file
        tio::read_etn("/tmp/einsums_example.etn", "A", &A);
        cg::einsum("ik;kj->ij", &D, A, B);
        tio::write_etn("/tmp/einsums_result.etn", "D", &D);
    }
    graph2.execute();

    einsums::println("  Executed graph with read_etn → einsum → write_etn");
    einsums::println("  D(0,0) = {:.6f}", D(0, 0));

    // Cleanup
    std::remove("/tmp/einsums_example.etn");
    std::remove("/tmp/einsums_checkpoint.etn");
    std::remove("/tmp/einsums_result.etn");

    einsums::println("\nDone!");
    return 0;
}

int main(int argc, char **argv) {
    return einsums::start(einsums_main, argc, argv);
}
