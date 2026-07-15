//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

/// @file OptimizationPasses.cpp
/// @brief Demonstrates all optimization passes available for computation graphs.
///
/// Passes demonstrated:
///   1. ScaleAbsorption (einsum), removes a scale the einsum overwrites
///   2. MemoryPlanning: analyzes tensor liveness and reports memory savings
///   3. Reorder: memory-aware topological sort
///   4. CSE: common subexpression elimination
///   5. ContractionPlanning: cost-model chain restructuring
///   6. ConstantFolding: folds constant-input operations at optimization time
///   7. ScaleAbsorption (permute), removes a scale a downstream permute overwrites
///   8. GEMMBatching: detects independent GEMMs that can be batched
///   9. LoopInvariantHoisting: moves invariant operations out of loops
///  10. DeadNodeElimination: removes nodes whose outputs are unused
///  11. ElementWiseFusion: fuses consecutive element-wise operations
///  12. DistributiveFactoring: rewrites A*B1 + A*B2 → A*(B1+B2)
///  13. InplaceOptimization: detects in-place operation candidates (analysis)
///  14. PermuteFusion: detects transpose-into-GEMM fusion (analysis)
///  15. create_default(): applies all safe passes in one call

#include <Einsums/ComputeGraph.hpp>
#include <Einsums/Print.hpp>
#include <Einsums/Runtime.hpp>
#include <Einsums/TensorAlgebra.hpp>
#include <Einsums/TensorUtilities/CreateRandomTensor.hpp>
#include <Einsums/TensorUtilities/CreateZeroTensor.hpp>

#include <iostream>

namespace cg = einsums::compute_graph;

int einsums_main() {
    using namespace einsums;
    using namespace einsums::index;

    // ═══════════════════════════════════════════════════════════════════════
    // 1. ScaleAbsorption (einsum case)
    // ═══════════════════════════════════════════════════════════════════════
    println("=== ScaleAbsorption (einsum) Pass ===\n");
    {
        auto A = create_random_tensor<double>("A", 6, 4);
        auto B = create_random_tensor<double>("B", 4, 5);
        auto C = create_random_tensor<double>("C", 6, 5);

        cg::Graph graph("fuse_example");
        {
            cg::CaptureGuard guard(graph);

            // Pattern: Scale C, then overwrite C with einsum (c_pf=0)
            // The scale can be absorbed into the einsum as c_pf=α
            cg::scale(3.0, &C);
            cg::einsum("ik;kj->ij", 0.0, &C, 1.0, A, B);
        }

        println("Before fusion:");
        graph.print_summary(std::cout);

        auto [modified, _p] = graph.apply<cg::passes::ScaleAbsorption>();

        println("\nAfter fusion (modified={}):", modified);
        graph.print_summary(std::cout);
        println("  -> Scale node absorbed into einsum: c_prefactor is now 3.0");

        graph.execute();
        println("Result C:");
        println(C);
    }

    // ═══════════════════════════════════════════════════════════════════════
    // 2. MemoryPlanning
    // ═══════════════════════════════════════════════════════════════════════
    println("\n=== MemoryPlanning Pass ===\n");
    {
        auto A  = create_random_tensor<double>("A", 100, 100);
        auto B  = create_random_tensor<double>("B", 100, 100);
        auto T1 = create_zero_tensor<double>("T1", 100, 100);
        auto T2 = create_zero_tensor<double>("T2", 100, 100);
        auto T3 = create_zero_tensor<double>("T3", 100, 100);

        cg::Graph graph("memory_example");
        {
            cg::CaptureGuard guard(graph);

            // Chain of operations with intermediates
            // T1 only needed for the second einsum
            // T2 only needed for the third
            cg::einsum("ik;kj->ij", &T1, A, B);
            cg::einsum("ik;kj->ij", &T2, T1, A);
            cg::einsum("ik;kj->ij", &T3, T2, B);
        }

        auto [_m1, mem] = graph.apply<cg::passes::MemoryPlanning>();

        mem.print_report(std::cout);
        println("  Total memory includes all 5 tensors ({} bytes each)", 100 * 100 * sizeof(double));
        println("  Peak memory is lower because not all are live simultaneously");
    }

    // ═══════════════════════════════════════════════════════════════════════
    // 3. Reorder (memory-aware topological sort)
    // ═══════════════════════════════════════════════════════════════════════
    println("\n=== Reorder Pass ===\n");
    {
        auto A = create_random_tensor<double>("A", 10, 10);
        auto B = create_random_tensor<double>("B", 10, 10);
        auto C = create_zero_tensor<double>("C", 10, 10);
        auto D = create_zero_tensor<double>("D", 10, 10);

        cg::Graph graph("reorder_example");
        {
            cg::CaptureGuard guard(graph);

            // Two independent operations, can be reordered
            cg::einsum("ik;kj->ij", &C, A, B);
            cg::einsum("ik;kj->ij", &D, B, A);
        }

        println("Before reorder:");
        graph.print_summary(std::cout);

        auto [modified, _p] = graph.apply<cg::passes::Reorder>();

        println("\nAfter reorder (modified={}):", modified);
        graph.print_summary(std::cout);

        graph.execute();
    }

    // ═══════════════════════════════════════════════════════════════════════
    // 4. CSE (Common Subexpression Elimination)
    // ═══════════════════════════════════════════════════════════════════════
    println("\n=== CSE Pass ===\n");
    {
        auto A = create_random_tensor<double>("A", 8, 6);
        auto B = create_random_tensor<double>("B", 6, 4);
        auto C = create_zero_tensor<double>("C", 8, 4);

        cg::Graph graph("cse_example");
        // Graph-owned scratch: CSE only folds duplicates whose outputs the
        // user can't observe directly. A user-visible duplicate (e.g. a
        // second create_zero_tensor above) keeps its producer - writes to
        // user tensors are a contract.
        auto &D = graph.create_tensor<double, 2>("D", 8, 4);
        {
            cg::CaptureGuard guard(graph);

            // Two identical computations, CSE eliminates the duplicate
            cg::einsum("ik;kj->ij", &C, A, B);
            cg::einsum("ik;kj->ij", &D, A, B);
        }

        println("Before CSE: {} nodes", graph.num_nodes());
        graph.print_summary(std::cout);

        auto [modified, _p] = graph.apply<cg::passes::CSE>();

        println("\nAfter CSE (modified={}): {} nodes", modified, graph.num_nodes());
        graph.print_summary(std::cout);
        println("  -> Duplicate einsum eliminated; D's computation reuses C's result");

        graph.execute();
    }

    // ═══════════════════════════════════════════════════════════════════════
    // 5. ContractionPlanning (chain restructuring)
    // ═══════════════════════════════════════════════════════════════════════
    println("\n=== ContractionPlanning Pass ===\n");
    {
        // Classic matrix chain example where parenthesization matters enormously:
        // A(100x1) * B(1x100) * C(100x1)
        //
        // Left-to-right: (A*B)*C
        //   Step 1: A*B → 100x100, costs 2*100*1*100 = 20,000 FLOPs
        //   Step 2: (A*B)*C → 100x1, costs 2*100*100*1 = 20,000 FLOPs
        //   Total: 40,000 FLOPs
        //
        // Optimal: A*(B*C)
        //   Step 1: B*C → 1x1, costs 2*1*100*1 = 200 FLOPs
        //   Step 2: A*(B*C) → 100x1, costs 2*100*1*1 = 200 FLOPs
        //   Total: 400 FLOPs  (100x improvement!)

        auto A  = create_random_tensor<double>("A", 100, 1);
        auto B  = create_random_tensor<double>("B", 1, 100);
        auto C  = create_random_tensor<double>("C", 100, 1);
        auto T1 = create_zero_tensor<double>("T1", 100, 100);
        auto T2 = create_zero_tensor<double>("T2", 100, 1);

        cg::Graph graph("chain_example");
        {
            cg::CaptureGuard guard(graph);
            cg::einsum("ik;kj->ij", &T1, A, B);  // T1 = A*B
            cg::einsum("ik;kj->ij", &T2, T1, C); // T2 = T1*C
        }

        auto [modified5, chain] = graph.apply<cg::passes::ContractionPlanning>();

        println("Matrix chain: A(100x1) * B(1x100) * C(100x1)");
        println("  Chains restructured: {} ({} intermediates created)", chain.chains_restructured(), chain.intermediates_created());
        for (auto const &rep : chain.chain_reports()) {
            println("  Estimated: {:.1f}us -> {:.1f}us ({:.2f}x speedup)", rep.original_time_us, rep.optimal_time_us, rep.speedup);
        }
        (void)modified5;
    }

    // ═══════════════════════════════════════════════════════════════════════
    // Combined: multiple passes on one graph
    // ═══════════════════════════════════════════════════════════════════════
    println("\n=== Combined Passes ===\n");
    {
        auto A = create_random_tensor<double>("A", 20, 20);
        auto B = create_random_tensor<double>("B", 20, 20);
        auto C = create_random_tensor<double>("C", 20, 20);

        cg::Graph graph("combined");
        {
            cg::CaptureGuard guard(graph);
            cg::scale(2.0, &C);
            cg::einsum("ik;kj->ij", 0.0, &C, 1.0, A, B);
        }

        println("Before passes: {} nodes", graph.num_nodes());

        cg::PassManager pm;
        pm.add<cg::passes::ScaleAbsorption>().add<cg::passes::Reorder>();
        graph.apply(pm);
        auto [_m3, mem2] = graph.apply<cg::passes::MemoryPlanning>();

        println("After passes: {} nodes", graph.num_nodes());
        mem2.print_report(std::cout);

        graph.execute();
        println("Result C (with profiling):");
        println(C);
    }

    // ═══════════════════════════════════════════════════════════════════════
    // 6. ConstantFolding
    // ═══════════════════════════════════════════════════════════════════════
    println("\n=== ConstantFolding ===\n");
    {
        // ConstantFolding only folds graph-owned intermediate tensors that are
        // never written by any node. User-owned tensors are NOT assumed constant
        // since they may change between loop iterations or execute() calls.
        // This makes the pass safe for both one-shot graphs and loop bodies.

        auto A = create_random_tensor<double>("A", 5, 5);
        auto B = create_random_tensor<double>("B", 5, 5);
        auto C = create_zero_tensor<double>("C", 5, 5);

        cg::Graph graph("constant_fold");
        {
            cg::CaptureGuard guard(graph);
            // A and B are user-owned, the pass will NOT fold this.
            // To fold, use graph-owned intermediates.
            cg::einsum("ik;kj->ij", &C, A, B);
        }

        auto [_m4, fold] = graph.apply<cg::passes::ConstantFolding>();
        println("Folded {} constant nodes (0 expected: A and B are user-owned)", fold.num_folded());

        graph.execute();
        println("C[0,0] = {:.4f} (computed during execute, not during folding)", C(0, 0));
    }

    // ═══════════════════════════════════════════════════════════════════════
    // 7. ScaleAbsorption (generalized)
    // ═══════════════════════════════════════════════════════════════════════
    println("\n=== ScaleAbsorption ===\n");
    {
        auto A = create_random_tensor<double>("A", 4, 6);
        auto C = create_random_tensor<double>("C", 6, 4);

        cg::Graph graph("absorb");
        {
            cg::CaptureGuard guard(graph);
            cg::scale(3.0, &C);
            cg::permute("ji <- ij", 0.0, &C, 1.0, A);
        }

        println("Before: {} nodes", graph.num_nodes());
        auto [_m5, absorb] = graph.apply<cg::passes::ScaleAbsorption>();
        println("After: {} nodes (absorbed {} scale ops)", graph.num_nodes(), absorb.num_absorbed());
    }

    // ═══════════════════════════════════════════════════════════════════════
    // 8. GEMMBatching (analysis)
    // ═══════════════════════════════════════════════════════════════════════
    println("\n=== GEMMBatching ===\n");
    {
        auto A1 = create_random_tensor<double>("A1", 10, 8);
        auto A2 = create_random_tensor<double>("A2", 10, 8);
        auto A3 = create_random_tensor<double>("A3", 10, 8);
        auto B1 = create_random_tensor<double>("B1", 8, 6);
        auto B2 = create_random_tensor<double>("B2", 8, 6);
        auto B3 = create_random_tensor<double>("B3", 8, 6);
        auto C1 = create_zero_tensor<double>("C1", 10, 6);
        auto C2 = create_zero_tensor<double>("C2", 10, 6);
        auto C3 = create_zero_tensor<double>("C3", 10, 6);

        cg::Graph graph("batch");
        {
            cg::CaptureGuard guard(graph);
            cg::einsum("ik;kj->ij", &C1, A1, B1);
            cg::einsum("ik;kj->ij", &C2, A2, B2);
            cg::einsum("ik;kj->ij", &C3, A3, B3);
        }

        auto [_m6, batch] = graph.apply<cg::passes::GEMMBatching>();
        println("Found {} batch groups with {} total batched GEMMs", batch.num_batches(), batch.total_batched());
    }

    // ═══════════════════════════════════════════════════════════════════════
    // 9. LoopInvariantHoisting
    // ═══════════════════════════════════════════════════════════════════════
    println("\n=== LoopInvariantHoisting ===\n");
    {
        auto A = create_random_tensor<double>("A", 4, 4);
        auto B = create_random_tensor<double>("B", 4, 4);
        auto C = create_zero_tensor<double>("C", 4, 4);

        cg::Graph graph("hoist");

        auto &body = graph.add_loop("iter", 10, [](size_t iter) { return iter < 9; });
        {
            cg::CaptureGuard guard(body);
            cg::einsum("ik;kj->ij", &C, A, B); // Invariant!
            cg::scale(0.99, &C);               // NOT invariant
        }

        auto [_m7, hoist] = graph.apply<cg::passes::LoopInvariantHoisting>();
        println("Hoisted {} invariant operations out of loop", hoist.num_hoisted());
    }

    // ═══════════════════════════════════════════════════════════════════════
    // 10. DeadNodeElimination
    // ═══════════════════════════════════════════════════════════════════════
    println("\n=== DeadNodeElimination ===\n");
    {
        auto A = create_random_tensor<double>("A", 4, 4);
        auto B = create_random_tensor<double>("B", 4, 4);
        auto C = create_zero_tensor<double>("C", 4, 4);

        cg::Graph graph("dne_demo");
        // Graph-owned intermediate T, if nothing reads it, it's dead.
        auto &T = graph.create_zero_tensor<double, 2>("T", 4, 4);

        {
            cg::CaptureGuard guard(graph);
            cg::einsum("ik;kj->ij", &T, A, B);
            cg::einsum("ik;kj->ij", &C, A, B);
        }

        println("Before: {} nodes", graph.num_nodes());
        auto [_m10, dne] = graph.apply<cg::passes::DeadNodeElimination>();
        println("After: {} nodes (eliminated {} dead nodes)", graph.num_nodes(), dne.num_eliminated());
        println("  -> T is intermediate with no reader, so its producer was removed");
    }

    // ═══════════════════════════════════════════════════════════════════════
    // 11. ElementWiseFusion
    // ═══════════════════════════════════════════════════════════════════════
    println("\n=== ElementWiseFusion ===\n");
    {
        auto A = create_random_tensor<double>("A", 5, 5);

        cg::Graph graph("ewf_demo");
        {
            cg::CaptureGuard guard(graph);
            // Three consecutive scales on the same tensor can fuse
            cg::scale(2.0, &A);
            cg::scale(3.0, &A);
            cg::scale(4.0, &A);
        }

        println("Before: {} nodes (3 separate scale ops)", graph.num_nodes());
        auto [_m11, ewf] = graph.apply<cg::passes::ElementWiseFusion>();
        println("After: {} nodes (fused {} pairs → single scale by 24.0)", graph.num_nodes(), ewf.num_fused());
    }

    // ═══════════════════════════════════════════════════════════════════════
    // 12. DistributiveFactoring
    // ═══════════════════════════════════════════════════════════════════════
    println("\n=== DistributiveFactoring ===\n");
    {
        // R += A*B1, R += A*B2 can be rewritten as R += A*(B1+B2)
        // Saves one matrix multiply when there are many terms sharing A.
        auto A  = create_random_tensor<double>("A", 6, 4);
        auto B1 = create_random_tensor<double>("B1", 4, 6);
        auto B2 = create_random_tensor<double>("B2", 4, 6);
        auto R  = create_zero_tensor<double>("R", 6, 6);

        cg::Graph graph("factor_demo");
        {
            cg::CaptureGuard guard(graph);
            cg::einsum("ik;kj->ij", 1.0, &R, 1.0, A, B1);
            cg::einsum("ik;kj->ij", 1.0, &R, 1.0, A, B2);
        }

        println("Before: {} nodes (2 einsums with shared operand A)", graph.num_nodes());
        auto [_m12, factor] = graph.apply<cg::passes::DistributiveFactoring>();
        println("After: {} nodes (found {} factoring groups, eliminated {} nodes)", graph.num_nodes(), factor.num_groups(),
                factor.num_eliminated());
    }

    // ═══════════════════════════════════════════════════════════════════════
    // 13. InplaceOptimization (analysis)
    // ═══════════════════════════════════════════════════════════════════════
    println("\n=== InplaceOptimization ===\n");
    {
        auto A = create_random_tensor<double>("A", 4, 3);
        auto B = create_random_tensor<double>("B", 3, 5);
        auto D = create_random_tensor<double>("D", 5, 3);
        auto C = create_zero_tensor<double>("C", 4, 3);

        cg::Graph graph("inplace_demo");
        auto     &T = graph.create_zero_tensor<double, 2>("T", 4, 5);

        {
            cg::CaptureGuard guard(graph);
            cg::einsum("ik;kj->ij", &T, A, B);
            cg::einsum("ik;kj->ij", &C, T, D);
        }

        auto [_m13, inplace] = graph.apply<cg::passes::InplaceOptimization>();
        println("In-place candidates: {} (intermediates with 1 writer + 1 reader)", inplace.num_candidates());
    }

    // ═══════════════════════════════════════════════════════════════════════
    // 14. PermuteFusion (analysis)
    // ═══════════════════════════════════════════════════════════════════════
    println("\n=== PermuteFusion ===\n");
    {
        auto A = create_random_tensor<double>("A", 4, 4);
        auto B = create_random_tensor<double>("B", 4, 4);

        cg::Graph graph("permfuse_demo");
        auto     &At = graph.create_zero_tensor<double, 2>("At", 4, 4);
        auto     &C  = graph.create_zero_tensor<double, 2>("C", 4, 4);

        {
            cg::CaptureGuard guard(graph);
            // Transpose A, then use it in a GEMM → fusion opportunity
            cg::permute("ji <- ij", 0.0, &At, 1.0, A);
            cg::gemm<false, false>(1.0, At, B, 0.0, &C);
        }

        auto [_m14, pf] = graph.apply<cg::passes::PermuteFusion>();
        println("Permute-GEMM fusion candidates: {} (transpose can be absorbed into GEMM flags)", pf.num_candidates());
    }

    // ═══════════════════════════════════════════════════════════════════════
    // 15. PassManager::create_default(): all safe passes in one call
    // ═══════════════════════════════════════════════════════════════════════
    println("\n=== create_default() ===\n");
    {
        auto A = create_random_tensor<double>("A", 10, 8);
        auto B = create_random_tensor<double>("B", 8, 6);
        auto C = create_zero_tensor<double>("C", 10, 6);

        cg::Graph graph("default_demo");
        {
            cg::CaptureGuard guard(graph);
            cg::scale(2.0, &C);
            cg::einsum("ik;kj->ij", 0.0, &C, 1.0, A, B);
            cg::scale(3.0, &C);
        }

        println("Before: {} nodes", graph.num_nodes());

        // create_default() applies all safe optimization and analysis passes:
        //   ConstantFolding, ScaleAbsorption, CSE, DeadNodeElimination,
        //   ElementWiseFusion, LoopInvariantHoisting, Reorder,
        //   GPU passes (when available), MemoryPlanning,
        //   InplaceOptimization, GEMMBatching, PermuteFusion
        auto pm = cg::PassManager::create_default();
        graph.apply(pm);

        println("After create_default(): {} nodes", graph.num_nodes());
        println("  -> ScaleAbsorption removed the dead scale(2.0) the einsum overwrites");
        println("  -> ConstantFolding is safe (only folds graph-owned intermediates)");
        println("  -> GPU passes included but won't place double-precision ops on MPS");

        graph.execute();
        println("Result C[0,0] = {:.4f}", C(0, 0));
    }

    finalize();
    return EXIT_SUCCESS;
}

int main(int argc, char **argv) {
    return einsums::start(einsums_main, argc, argv);
}
