//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

/**
 * @file ComputeGraph.hpp
 * @brief Umbrella header for the ComputeGraph module.
 *
 * The ComputeGraph module provides a deferred-execution computation graph for Einsums,
 * inspired by CUDA Graphs and PyTorch FX. It allows users to:
 *
 * 1. **Capture** a sequence of tensor operations (einsum, gemm, scale, svd, etc.)
 *    into a graph instead of executing them immediately.
 * 2. **Optimize** the captured graph with passes (fusion, CSE, memory planning, etc.).
 * 3. **Execute** or **replay** the graph, amortizing overhead for iterative algorithms.
 * 4. **Profile** execution with per-node instrumentation and annotations.
 *
 * @par Quick Start
 * @code
 * #include <Einsums/ComputeGraph/ComputeGraph.hpp>
 *
 * namespace cg = einsums::compute_graph;
 *
 * // Create tensors
 * auto A = create_random_tensor<double>("A", N, N);
 * auto B = create_random_tensor<double>("B", N, N);
 *
 * // Graph with owned intermediate
 * cg::Graph graph("example");
 * auto &C = graph.create_zero_tensor<double, 2>("C", N, N);
 *
 * // Capture
 * {
 *     cg::CaptureGuard guard(graph);
 *     cg::einsum("ik;kj->ij", &C, A, B);
 *     cg::scale(2.0, &C);
 * }
 *
 * // Optimize
 * auto pm = cg::PassManager::create_default();
 * graph.apply(pm);
 *
 * // Execute
 * graph.execute();
 * @endcode
 *
 * @par Key Classes
 * - Graph: DAG container with capture, execution, and optimization
 * - Pipeline: Multi-stage workflow with loops and early exit
 * - CaptureGuard: RAII guard for capture scoping
 * - OptimizerPass: Base class for graph optimization passes
 *
 * @par Operations (einsums::compute_graph namespace)
 * Graph-aware wrappers for all Einsums operations: einsum, scale, permute,
 * transpose, element_transform, gemm, gemv, ger, axpy, axpby, dot, norm,
 * symm_gemm, direct_product, syev, heev, gesv, invert, svd, qr, pow, det, etc.
 *
 * @par Optimization Passes (einsums::compute_graph::passes namespace)
 * Graph-transforming:
 * - ScaleAbsorption: Absorb scale into subsequent operations' prefactors
 * - CSE: Common subexpression elimination
 * - ConstantFolding: Execute constant computations at optimization time
 * - DeadNodeElimination: Remove nodes with no consumers
 * - Reorder: Memory-aware topological sort
 * - LoopInvariantHoisting: Move invariant ops out of loops
 *
 * Analysis-only:
 * - MemoryPlanning: Tensor liveness analysis
 * - ChainParenthesization: Optimal matrix chain order
 * - InplaceOptimization: Detect in-place candidates
 * - GEMMBatching: Detect batchable independent GEMMs
 * - PermuteFusion: Detect transpose-into-GEMM candidates
 *
 * @par Tensor Lifetime Safety
 * Tensors referenced by captured operations must outlive the graph. Two mechanisms:
 * - **Graph::create_tensor()**: Creates graph-owned tensors with Alloc node
 * - **Runtime validation**: execute() checks tensor pointers before running
 */

#pragma once

#include <Einsums/ComputeGraph/Blueprints.hpp>
#include <Einsums/ComputeGraph/CaptureContext.hpp>
#include <Einsums/ComputeGraph/EinsumSpec.hpp>
#include <Einsums/ComputeGraph/Executor.hpp>
#include <Einsums/ComputeGraph/Graph.hpp>
#include <Einsums/ComputeGraph/Node.hpp>
#include <Einsums/ComputeGraph/Operations.hpp>
#include <Einsums/ComputeGraph/Optimizer.hpp>
#include <Einsums/ComputeGraph/Passes/CSE.hpp>
#include <Einsums/ComputeGraph/Passes/ChainParenthesization.hpp>
#include <Einsums/ComputeGraph/Passes/CommunicationElimination.hpp>
#include <Einsums/ComputeGraph/Passes/CommunicationInsertion.hpp>
#include <Einsums/ComputeGraph/Passes/CommunicationScheduling.hpp>
#include <Einsums/ComputeGraph/Passes/ConstantFolding.hpp>
#include <Einsums/ComputeGraph/Passes/ContractionPlanning.hpp>
#include <Einsums/ComputeGraph/Passes/DeadNodeElimination.hpp>
#include <Einsums/ComputeGraph/Passes/DistributionPlanning.hpp>
#include <Einsums/ComputeGraph/Passes/DistributiveFactoring.hpp>
#include <Einsums/ComputeGraph/Passes/ElementWiseFusion.hpp>
#include <Einsums/ComputeGraph/Passes/FreeInsertion.hpp>
#include <Einsums/ComputeGraph/Passes/GEMMBatching.hpp>
#include <Einsums/ComputeGraph/Passes/GPUDiagnostics.hpp>
#include <Einsums/ComputeGraph/Passes/GPUPlacement.hpp>
#include <Einsums/ComputeGraph/Passes/IOPrefetch.hpp>
#include <Einsums/ComputeGraph/Passes/InplaceOptimization.hpp>
#include <Einsums/ComputeGraph/Passes/LinearCombinationContractionFolding.hpp>
#include <Einsums/ComputeGraph/Passes/LoopInvariantHoisting.hpp>
#include <Einsums/ComputeGraph/Passes/Materialization.hpp>
#include <Einsums/ComputeGraph/Passes/MemoryPlanning.hpp>
#include <Einsums/ComputeGraph/Passes/PermuteFusion.hpp>
#include <Einsums/ComputeGraph/Passes/Reorder.hpp>
#include <Einsums/ComputeGraph/Passes/ScaleAbsorption.hpp>
#include <Einsums/ComputeGraph/Passes/StreamAssignment.hpp>
#include <Einsums/ComputeGraph/Passes/SymmetryPropagation.hpp>
#include <Einsums/ComputeGraph/Passes/TransferElimination.hpp>
#include <Einsums/ComputeGraph/Passes/TransferInsertion.hpp>
#include <Einsums/ComputeGraph/Pipeline.hpp>
#include <Einsums/ComputeGraph/TensorHandle.hpp>
#include <Einsums/ComputeGraph/View.hpp>
#include <Einsums/ComputeGraph/Workspace.hpp>
