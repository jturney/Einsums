#----------------------------------------------------------------------------------------------
# Copyright (c) The Einsums Developers. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
#----------------------------------------------------------------------------------------------

set(ComputeGraphHeaders
    Einsums/ComputeGraph.hpp
    Einsums/ComputeGraph/Blueprints.hpp
    Einsums/ComputeGraph/Blueprints/Orthogonalization.hpp
    Einsums/ComputeGraph/Blueprints/TensorAlgebra.hpp
    Einsums/ComputeGraph/BoundExpr.hpp
    Einsums/ComputeGraph/CaptureContext.hpp
    Einsums/ComputeGraph/DeviceShadowMap.hpp
    Einsums/ComputeGraph/Error.hpp
    Einsums/ComputeGraph/Graph.hpp
    Einsums/ComputeGraph/HardwareProfile.hpp
    Einsums/ComputeGraph/Node.hpp
    Einsums/ComputeGraph/Operations.hpp
    Einsums/ComputeGraph/Optimizer.hpp
    Einsums/ComputeGraph/UsageAnalysis.hpp
    Einsums/ComputeGraph/Passes/ContractionPlanning.hpp
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
    Einsums/ComputeGraph/Passes/StreamKernel.hpp
    Einsums/ComputeGraph/Passes/LoopInvariantHoisting.hpp
    Einsums/ComputeGraph/Passes/MemoryPlanning.hpp
    Einsums/ComputeGraph/Passes/InputSlicing.hpp
    Einsums/ComputeGraph/Passes/IOPrefetch.hpp
    Einsums/ComputeGraph/Passes/Materialization.hpp
    Einsums/ComputeGraph/Passes/PermuteFusion.hpp
    Einsums/ComputeGraph/Passes/Reorder.hpp
    Einsums/ComputeGraph/Passes/SUMMAExpansion.hpp
    Einsums/ComputeGraph/Passes/GPUDiagnostics.hpp
    Einsums/ComputeGraph/Passes/GPUPlacement.hpp
    Einsums/ComputeGraph/Passes/ScaleAbsorption.hpp
    Einsums/ComputeGraph/Passes/StreamAssignment.hpp
    Einsums/ComputeGraph/Passes/SymmetryPropagation.hpp
    Einsums/ComputeGraph/Passes/TransferElimination.hpp
    Einsums/ComputeGraph/Passes/TransferInsertion.hpp
    Einsums/ComputeGraph/Pipeline.hpp
    Einsums/ComputeGraph/EinsumSpec.hpp
    Einsums/ComputeGraph/Executor.hpp
    Einsums/ComputeGraph/StringDispatch.hpp
    Einsums/ComputeGraph/TensorHandle.hpp
    Einsums/ComputeGraph/TensorSlot.hpp
    Einsums/ComputeGraph/View.hpp
    Einsums/ComputeGraph/Workspace.hpp
)

set(ComputeGraphSources
    CaptureContext.cpp
    EinsumSpec.cpp
    Executor.cpp
    Graph.cpp
    UsageAnalysis.cpp
    HardwareProfile.cpp
    Node.cpp
    Optimizer.cpp
    Passes/CommunicationElimination.cpp
    Passes/CommunicationInsertion.cpp
    Passes/CommunicationScheduling.cpp
    Passes/ContractionPlanning.cpp
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
    Passes/Reorder.cpp
    Passes/SUMMAExpansion.cpp
    Passes/GPUDiagnostics.cpp
    Passes/GPUPlacement.cpp
    Passes/ScaleAbsorption.cpp
    Passes/StreamAssignment.cpp
    Passes/SymmetrizedAccumulation.cpp
    Passes/SymmetryPropagation.cpp
    Passes/TransferElimination.cpp
    Passes/TransferInsertion.cpp
    Pipeline.cpp
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
  SOURCES ${ComputeGraphSources}
  HEADERS ${ComputeGraphHeaders}
  MODULE_DEPENDENCIES
    Einsums_Comm
    Einsums_ComputeGraphTypes
    Einsums_Concepts
    Einsums_Config
    Einsums_CXX23
    Einsums_GPU
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
