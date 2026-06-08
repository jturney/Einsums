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
    test_hyp_controlflow_diff_python.py
    test_hyp_einsum_diff_python.py
    test_hyp_elementwise_diff_python.py
    test_hyp_lapack_diff_python.py
    test_hyp_linalg_diff_python.py
    test_hyp_program_diff_python.py
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
    test_view_lifetime_python.py
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

    # FuzzDifferentialPython runs ~6340 pytest sub-cases in a single
    # process. Under TSan instrumentation, the lock-order graph
    # accumulates per-mutex / per-cross-edge state for every test case
    # — by end-of-process the graph traversal in libtsan's exit-time
    # bookkeeping overruns the worker thread's stack. The deadly signal
    # fires AFTER our finalize() has completed cleanly, and TSan
    # reports it as "nested bug in the same thread, aborting" because
    # the crash happens inside its own reporting machinery (CI run
    # 26692242448). The test logic itself passes (6340 passed, 244
    # skipped in 771s). Disable deadlock detection just for this test
    # so the exit-time stack overflow doesn't kill the process; race
    # detection (the part that found the real bugs in d235b490 /
    # 7b288ea9 / 4d496cd2) stays enabled. Other Python tests are
    # tiny by comparison and keep full TSan settings.
    if(EINSUMS_BUILD_PYTHON
       AND EINSUMS_WITH_SANITIZERS
       AND "${EINSUMS_WITH_SANITIZERS}" MATCHES "thread"
       AND "${_camel}Python" STREQUAL "FuzzDifferentialPython"
    )
      set_property(
        TEST "Tests.Unit.Modules.ComputeGraph.FuzzDifferentialPython"
        APPEND PROPERTY ENVIRONMENT
        "TSAN_OPTIONS=ignore_noninstrumented_modules=1:detect_deadlocks=0:suppressions=${PROJECT_SOURCE_DIR}/devtools/sanitizers/tsan.supp"
      )
      # Bump the per-test timeout from ctest's 1500 sec default to one hour
      # for the TSan leg. With 6340 fuzz sub-cases × TSan's 5-15x
      # instrumentation slowdown, this test legitimately needs more than
      # 25 minutes (CI run 26710235787: 1500.17 sec timeout while still
      # making forward progress, 0 TSan warnings — the test would have
      # finished cleanly given a few more minutes).
      set_property(
        TEST "Tests.Unit.Modules.ComputeGraph.FuzzDifferentialPython"
        PROPERTY TIMEOUT 3600
      )
    endif()
  endif ()
endforeach()
