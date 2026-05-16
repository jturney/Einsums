#----------------------------------------------------------------------------------------------
# Copyright (c) The Einsums Developers. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
#----------------------------------------------------------------------------------------------

set(ComputeGraphTests
    Blueprints.cpp
    ControlFlow.cpp
    DependencyOrder.cpp
    DistributedIntegration.cpp
    DistributiveFactoring.cpp
    EinsumSpec.cpp
    Executor.cpp
    GEMMBatching.cpp
    GPUPasses.cpp
    GraphCapture.cpp
    GraphFeatures.cpp
    HardwareProfile.cpp
    MemoryNodes.cpp
    MixedOps.cpp
    OperationsCoverage.cpp
    ParallelForGraph.cpp
    Pass_ChainParenthesization.cpp
    Pass_ConstantFolding.cpp
    Pass_CSE.cpp
    Pass_DeadNodeElimination.cpp
    Pass_ElementWiseFusion.cpp
    Pass_InplaceOptimization.cpp
    Pass_LoopInvariantHoisting.cpp
    Pass_MemoryPlanning.cpp
    Pass_Reorder.cpp
    Pass_ScaleAbsorption.cpp
    Pass_SymmetryPropagation.cpp
    PassManager.cpp
    PermuteFusion.cpp
    PipelineLoop.cpp
    Rebind.cpp
    RuntimeTensorEinsum.cpp
    RuntimeTensorLAPACK.cpp
    RuntimeTensorOwnership.cpp
    RuntimeTensorView.cpp
    StridedBatchedGemm.cpp
    StringEinsum.cpp
    TensorLifetime.cpp
    TensorRankConcept.cpp
    View.cpp
    Workspace.cpp
)

foreach(test ${ComputeGraphTests})
  get_filename_component(TestName ${test} NAME_WE)

  einsums_add_executable(
    ${TestName}_test INTERNAL_FLAGS
    SOURCES ${test}
    NOINSTALL
  )

  einsums_add_unit_test(
    "Modules.ComputeGraph" ${TestName}
  )
endforeach()

einsums_add_python_unit_test(
  "Modules.ComputeGraph" EinsumPython
  SCRIPT test_einsum_python.py
)

einsums_add_python_unit_test(
  "Modules.ComputeGraph" ScalePython
  SCRIPT test_scale_python.py
)

einsums_add_python_unit_test(
  "Modules.ComputeGraph" PermutePython
  SCRIPT test_permute_python.py
)

einsums_add_python_unit_test(
  "Modules.ComputeGraph" GraphPython
  SCRIPT test_graph_python.py
)

einsums_add_python_unit_test(
  "Modules.ComputeGraph" BlasPython
  SCRIPT test_blas_python.py
)

einsums_add_python_unit_test(
  "Modules.ComputeGraph" LapackPython
  SCRIPT test_lapack_python.py
)

einsums_add_python_unit_test(
  "Modules.ComputeGraph" MixedOpsPython
  SCRIPT test_mixed_ops_python.py
)

einsums_add_python_unit_test(
  "Modules.ComputeGraph" ExecutorPython
  SCRIPT test_executor_python.py
)

einsums_add_python_unit_test(
  "Modules.ComputeGraph" CapturePython
  SCRIPT test_capture_python.py
)

einsums_add_python_unit_test(
  "Modules.ComputeGraph" ErrorReportingPython
  SCRIPT test_error_reporting_python.py
)
