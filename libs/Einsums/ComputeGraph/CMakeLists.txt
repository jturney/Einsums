#----------------------------------------------------------------------------------------------
# Copyright (c) The Einsums Developers. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
#----------------------------------------------------------------------------------------------

set(ComputeGraphHeaders
    Einsums/ComputeGraph.hpp
    Einsums/ComputeGraph/Approximation.hpp
    Einsums/ComputeGraph/Blueprints.hpp
    Einsums/ComputeGraph/Factorization.hpp
    Einsums/ComputeGraph/Blueprints/Orthogonalization.hpp
    Einsums/ComputeGraph/Blueprints/TensorAlgebra.hpp
    Einsums/ComputeGraph/BoundExpr.hpp
    Einsums/ComputeGraph/CaptureContext.hpp
    Einsums/ComputeGraph/Options.hpp
    Einsums/ComputeGraph/DeviceShadowMap.hpp
    Einsums/ComputeGraph/Diis.hpp
    Einsums/ComputeGraph/ElementOps.hpp
    Einsums/ComputeGraph/Error.hpp
    Einsums/ComputeGraph/ExecutorBuilder.hpp
    Einsums/ComputeGraph/ABILayout.hpp
    Einsums/ComputeGraph/GateFlags.hpp
    Einsums/ComputeGraph/Graph.hpp
    Einsums/ComputeGraph/GraphIR.hpp
    Einsums/ComputeGraph/InterfaceManifest.hpp
    Einsums/ComputeGraph/LuPivots.hpp
    Einsums/ComputeGraph/Detail/GroupedBatchedGemm.hpp
    Einsums/ComputeGraph/Detail/Json.hpp
    Einsums/ComputeGraph/CostModel.hpp
    Einsums/ComputeGraph/Moldability.hpp
    Einsums/ComputeGraph/Node.hpp
    Einsums/ComputeGraph/Operations.hpp
    Einsums/ComputeGraph/Optimizer.hpp
    Einsums/ComputeGraph/UsageAnalysis.hpp
    Einsums/ComputeGraph/EscapeAnalysis.hpp
    Einsums/ComputeGraph/TensorExpr.hpp
    Einsums/ComputeGraph/Passes/ContractionPlanning.hpp
    Einsums/ComputeGraph/Passes/DeltaElimination.hpp
    Einsums/ComputeGraph/Passes/ProvenancePropagation.hpp
    Einsums/ComputeGraph/Passes/CommunicationElimination.hpp
    Einsums/ComputeGraph/Passes/CommunicationInsertion.hpp
    Einsums/ComputeGraph/Passes/CommunicationScheduling.hpp
    Einsums/ComputeGraph/Passes/ConstantFolding.hpp
    Einsums/ComputeGraph/Passes/DistributionPlanning.hpp
    Einsums/ComputeGraph/Passes/CSE.hpp
    Einsums/ComputeGraph/Passes/DeadNodeElimination.hpp
    Einsums/ComputeGraph/Passes/DistributiveFactoring.hpp
    Einsums/ComputeGraph/Passes/ElementWiseFusion.hpp
    Einsums/ComputeGraph/Passes/FreeInsertion.hpp
    Einsums/ComputeGraph/Passes/GEMMBatching.hpp
    Einsums/ComputeGraph/Passes/InplaceOptimization.hpp
    Einsums/ComputeGraph/Passes/LinearCombinationContractionFolding.hpp
    Einsums/ComputeGraph/Passes/StreamContractionFusion.hpp
    Einsums/ComputeGraph/Passes/SymmetrizedAccumulation.hpp
    Einsums/ComputeGraph/Passes/TiledExpansion.hpp
    Einsums/ComputeGraph/Passes/StreamKernel.hpp
    Einsums/ComputeGraph/Passes/LoopInvariantHoisting.hpp
    Einsums/ComputeGraph/Passes/MemoryPlanning.hpp
    Einsums/ComputeGraph/Passes/InputSlicing.hpp
    Einsums/ComputeGraph/Passes/IOPrefetch.hpp
    Einsums/ComputeGraph/Passes/Materialization.hpp
    Einsums/ComputeGraph/Passes/PermuteFusion.hpp
    Einsums/ComputeGraph/Passes/RegionRewrite.hpp
    Einsums/ComputeGraph/Passes/Reorder.hpp
    Einsums/ComputeGraph/Passes/SUMMAExpansion.hpp
    Einsums/ComputeGraph/Passes/CrossSpaceValidation.hpp
    Einsums/ComputeGraph/Passes/GPUDiagnostics.hpp
    Einsums/ComputeGraph/Passes/GPUPlacement.hpp
    Einsums/ComputeGraph/Passes/ScalingAnalysis.hpp
    Einsums/ComputeGraph/Passes/ScaleAbsorption.hpp
    Einsums/ComputeGraph/Passes/ScratchPrivatization.hpp
    Einsums/ComputeGraph/Passes/SpacePropagation.hpp
    Einsums/ComputeGraph/Passes/StreamAssignment.hpp
    Einsums/ComputeGraph/Passes/SymmetryPropagation.hpp
    Einsums/ComputeGraph/Passes/ThreadPlanning.hpp
    Einsums/ComputeGraph/Passes/TransferElimination.hpp
    Einsums/ComputeGraph/Passes/TransferInsertion.hpp
    Einsums/ComputeGraph/Pipeline.hpp
    Einsums/ComputeGraph/PredExpr.hpp
    Einsums/ComputeGraph/SpaceRegistryAccess.hpp
    Einsums/ComputeGraph/SymbolicCost.hpp
    Einsums/ComputeGraph/EinsumSpec.hpp
    Einsums/ComputeGraph/Executor.hpp
    Einsums/ComputeGraph/StringDispatch.hpp
    Einsums/ComputeGraph/TensorHandle.hpp
    Einsums/ComputeGraph/TensorSlot.hpp
    Einsums/ComputeGraph/View.hpp
    Einsums/ComputeGraph/Workspace.hpp
)

set(ComputeGraphSources
    Approximation.cpp
    CaptureContext.cpp
    Factorization.cpp
    Options.cpp
    EinsumSpec.cpp
    ElementOps.cpp
    Executor.cpp
    StringDispatch.cpp
    ExecutorBuilder.cpp
    ABILayout.cpp
    Graph.cpp
    GraphIR.cpp
    InterfaceManifest.cpp
    Json.cpp
    SymbolicExtents.cpp
    GroupedBatchedGemm.cpp
    UsageAnalysis.cpp
    EscapeAnalysis.cpp
    TensorExpr.cpp
    CostModel.cpp
    SymbolicCost.cpp
    Node.cpp
    Optimizer.cpp
    Passes/CommunicationElimination.cpp
    Passes/CommunicationInsertion.cpp
    Passes/CommunicationScheduling.cpp
    Passes/ContractionPlanning.cpp
    Passes/DeltaElimination.cpp
    Passes/ProvenancePropagation.cpp
    Passes/ConstantFolding.cpp
    Passes/DistributionPlanning.cpp
    Passes/CSE.cpp
    Passes/DeadNodeElimination.cpp
    Passes/DistributiveFactoring.cpp
    Passes/ElementWiseFusion.cpp
    Passes/FreeInsertion.cpp
    Passes/GEMMBatching.cpp
    Passes/InplaceOptimization.cpp
    Passes/LinearCombinationContractionFolding.cpp
    Passes/StreamContractionFusion.cpp
    Passes/StreamKernelDispatch.cpp
    Passes/LoopInvariantHoisting.cpp
    Passes/MemoryPlanning.cpp
    Passes/InputSlicing.cpp
    Passes/IOPrefetch.cpp
    Passes/Materialization.cpp
    Passes/PermuteFusion.cpp
    Passes/RegionRewrite.cpp
    Passes/Reorder.cpp
    Passes/SUMMAExpansion.cpp
    Passes/CrossSpaceValidation.cpp
    Passes/GPUDiagnostics.cpp
    Passes/GPUPlacement.cpp
    Passes/ScalingAnalysis.cpp
    Passes/ScaleAbsorption.cpp
    Passes/ScratchPrivatization.cpp
    Passes/SpacePropagation.cpp
    Passes/StreamAssignment.cpp
    Passes/SymmetrizedAccumulation.cpp
    Passes/TiledExpansion.cpp
    Passes/SymmetryPropagation.cpp
    Passes/ThreadPlanning.cpp
    Passes/TransferElimination.cpp
    Passes/TransferInsertion.cpp
    Pipeline.cpp
    SpaceRegistry.cpp
    ThreadSweep.cpp
)

# StreamKernelImpl.cpp is the arch-generic stream-fusion inner kernel: it is
# compiled once per SIMD-dispatch rung (each copy in its own namespace) rather
# than as a regular module source. StreamKernelDispatch.cpp (a normal source)
# bridges the copies to a runtime-selected entry point. See
# Einsums/SIMD/RuntimeFeatures.hpp and Einsums_AddSIMDDispatch.cmake. No `sme`
# rung is requested: the kernel is a bandwidth-bound streaming FMA that gains
# nothing from the matrix unit.
include(Einsums_AddSIMDDispatch)
einsums_add_simd_dispatch_sources(
  ComputeGraphStreamKernelRungs IMPL src/Passes/StreamKernelImpl.cpp RUNGS baseline v2 v3 v4
)

include(Einsums_AddModule)
einsums_add_module(
  Einsums ComputeGraph
  PYBIND
  # This module's bindings are by far the largest in the tree - around 1,030
  # binding statements, an order of magnitude past the next module - and as one
  # translation unit they peak near 7.5 GB under the sanitizer legs' Debug
  # flags, which is what forces those legs to -j1 and what OOM-killed the
  # coverage leg outright. Split eight ways each shard peaks under 4.7 GB.
  # Going finer buys little: the einsum overload set is a single indivisible
  # emit unit that costs 4.3 GB on its own, and every extra shard re-parses the
  # whole header set, a fixed 3.4 s and 0.75 GB apiece.
  PYBIND_NUM_TU 8
  SOURCES ${ComputeGraphSources}
  HEADERS ${ComputeGraphHeaders}
  MODULE_DEPENDENCIES
    Einsums_BLAS
    Einsums_Comm
    Einsums_Options
    Einsums_ComputeGraphTypes
    Einsums_Concepts
    Einsums_Config
    Einsums_CXX23
    Einsums_GPU
    Einsums_Hardware
    Einsums_LinearAlgebra
    Einsums_Logging
    Einsums_PackedGemm
    Einsums_Profile
    Einsums_SIMD
    Einsums_TaskPool
    Einsums_Tensor
    Einsums_TensorAlgebra
  CMAKE_SUBDIRS examples tests tools
)

# Attach the generated per-rung wrapper sources (and the EINSUMS_SIMD_HAS_RUNG_*
# definitions StreamKernelDispatch.cpp keys off) to the module target.
target_sources(Einsums_ComputeGraph PRIVATE ${ComputeGraphStreamKernelRungs})
target_compile_definitions(Einsums_ComputeGraph PRIVATE ${ComputeGraphStreamKernelRungs_DEFINITIONS})
