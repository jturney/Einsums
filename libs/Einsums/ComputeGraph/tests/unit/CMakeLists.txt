#----------------------------------------------------------------------------------------------
# Copyright (c) The Einsums Developers. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
#----------------------------------------------------------------------------------------------

# Convert ``snake_case`` to ``CamelCase`` so we can derive a test name
# from a ``test_<snake>_python.py`` source filename without keeping a
# parallel name list. Lowercase parts get their first letter uppercased;
# the rest of each part is left untouched, so ``mixed_ops`` becomes
# ``MixedOps`` and ``error_reporting`` becomes ``ErrorReporting``.
function(_einsums_snake_to_camel out_var snake)
  set(_camel "")
  string(REPLACE "_" ";" _parts "${snake}")
  foreach(_part IN LISTS _parts)
    if(_part STREQUAL "")
      continue()
    endif()
    string(SUBSTRING "${_part}" 0 1 _head)
    string(TOUPPER "${_head}" _head)
    string(SUBSTRING "${_part}" 1 -1 _tail)
    string(APPEND _camel "${_head}${_tail}")
  endforeach()
  set(${out_var} "${_camel}" PARENT_SCOPE)
endfunction()

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
    LoopOptimizationCorrectness.cpp
    MemoryNodes.cpp
    MixedOps.cpp
    OperationsCoverage.cpp
    ParallelForGraph.cpp
    Pass_ChainParenthesization.cpp
    Pass_ConstantFolding.cpp
    Pass_CSE.cpp
    Pass_DeadNodeElimination.cpp
    Pass_ElementWiseFusion.cpp
    Pass_FreeInsertion.cpp
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
    RecursionPlumbing.cpp
    RuntimeTensorEinsum.cpp
    RuntimeTensorLAPACK.cpp
    RuntimeTensorOwnership.cpp
    RuntimeTensorView.cpp
    StridedBatchedGemm.cpp
    StringEinsum.cpp
    TensorLifetime.cpp
    TiledRuntimeEig.cpp
    TiledRuntimeEinsum.cpp
    TiledRuntimeElementwise.cpp
    TensorRankConcept.cpp
    TiledRuntimeTensorLifetime.cpp
    View.cpp
    Workspace.cpp
    test_auto_view_python.py
    test_axpy_axpby_views_python.py
    test_blas_python.py
    test_block_copy_python.py
    test_capture_python.py
    test_direct_division_python.py
    test_direct_product_views_python.py
    test_einsum_python.py
    test_einsum_views_python.py
    test_element_transform_python.py
    test_error_reporting_python.py
    test_executor_python.py
    test_fuzz_differential_python.py
    test_fuzz_ergonomics_python.py
    test_gemm_views_python.py
    test_gemv_views_python.py
    test_ger_views_python.py
    test_graph_python.py
    test_lapack_python.py
    test_linalg_new_python.py
    test_loop_python.py
    test_mixed_ops_python.py
    test_mp2_python.py
    test_outer_sum_python.py
    test_pass_chain_parenthesization_python.py
    test_pass_constant_folding_python.py
    test_pass_cse_python.py
    test_pass_dead_node_elimination_python.py
    test_pass_element_wise_fusion_python.py
    test_pass_inplace_optimization_python.py
    test_pass_loop_invariant_hoisting_python.py
    test_pass_memory_planning_python.py
    test_pass_reorder_python.py
    test_pass_scale_absorption_python.py
    test_pass_symmetry_propagation_python.py
    test_permute_python.py
    test_scalar_writers_python.py
    test_scf_python.py
    test_scale_python.py
    test_symm_gemm_views_python.py
    test_tiled_ops_python.py
    test_view_python.py
)

foreach(test ${ComputeGraphTests})
  get_filename_component(TestName ${test} NAME_WE)
  get_filename_component(Extension ${test} EXT)

  if (Extension STREQUAL ".cpp")
    einsums_add_executable(
      ${TestName}_test INTERNAL_FLAGS
      SOURCES ${test}
      NOINSTALL
    )

    einsums_add_unit_test(
      "Modules.ComputeGraph" ${TestName}
    )
  elseif (Extension STREQUAL ".py")
    # Naming convention: ``test_<snake>_python.py`` -> ``<Camel>Python``,
    # so e.g. ``test_mixed_ops_python.py`` registers as
    # ``Tests.Unit.Modules.ComputeGraph.MixedOpsPython``.
    string(REGEX REPLACE "^test_"   "" _stem "${TestName}")
    string(REGEX REPLACE "_python$" "" _stem "${_stem}")
    _einsums_snake_to_camel(_camel "${_stem}")
    einsums_add_python_unit_test(
      "Modules.ComputeGraph" "${_camel}Python"
      SCRIPT ${test}
    )
  endif ()
endforeach()
