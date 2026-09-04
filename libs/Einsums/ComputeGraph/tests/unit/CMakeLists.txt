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
    Approximation.cpp
    BisectDriver.cpp
    Blueprints.cpp
    Factorization.cpp
    MetricFit.cpp
    PooledOperands.cpp
    ControlFlow.cpp
    DeltaElimination.cpp
    DependencyOrder.cpp
    DiisAccelerator.cpp
    DistributedIntegration.cpp
    DistributiveFactoring.cpp
    AxpbyDescriptor.cpp
    EinsumSpec.cpp
    ElementOps.cpp
    EscapeAnalysis.cpp
    Executor.cpp
    ExecutorBuilder.cpp
    GEMMBatching.cpp
    GPUPasses.cpp
    GateFlags.cpp
    GraphCapture.cpp
    GraphFeatures.cpp
    GraphRegistry.cpp
    InterfaceManifest.cpp
    Json.cpp
    LayoutAssignment.cpp
    MultiTermFactorization.cpp
    SaveLoad.cpp
    SetupSubgraph.cpp
    CostModel.cpp
    LoopOptimizationCorrectness.cpp
    LuFactorization.cpp
    MemoryNodes.cpp
    MetadataBoundary.cpp
    MixedOps.cpp
    MixedWidthDifferential.cpp
    MixedWidthVendorSafety.cpp
    OperationsCoverage.cpp
    ParallelForGraph.cpp
    Pass_CommunicationPasses.cpp
    Pass_ConstantFolding.cpp
    Pass_ContractionPlanning.cpp
    Pass_CSE.cpp
    Pass_DistributionPasses.cpp
    Pass_DeadNodeElimination.cpp
    Pass_ElementWiseFusion.cpp
    Pass_FreeInsertion.cpp
    Pass_InplaceOptimization.cpp
    Pass_LinearCombinationContractionFolding.cpp
    Pass_StreamContractionFusion.cpp
    SymmetrizedAccumulationMatch.cpp
    SymmetrizedAccumulationRewrite.cpp
    Pass_LoopInvariantHoisting.cpp
    Pass_Materialization.cpp
    AliasBoxDerivation.cpp
    AliasStructural.cpp
    AliasOrderBatchedViews.cpp
    AliasOrderSharedScratch.cpp
    Pass_MemoryPlanning.cpp
    Pass_ProgramOrderGuard.cpp
    Pass_Reorder.cpp
    Pass_ScaleAbsorption.cpp
    Pass_TiledExpansion.cpp
    Pass_CrossSpaceValidation.cpp
    Pass_ScalingAnalysis.cpp
    Pass_SpacePropagation.cpp
    Pass_SymmetryPropagation.cpp
    PassManager.cpp
    PassPhases.cpp
    PassSwitches.cpp
    PermuteFusion.cpp
    PostReplayValidation.cpp
    PipelineLoop.cpp
    OperandOwnership.cpp
    Rebind.cpp
    RegionIdentity.cpp
    RecursionPlumbing.cpp
    EagerParityGaps.cpp
    Gather.cpp
    RuntimeTensorEinsum.cpp
    RuntimeTensorLAPACK.cpp
    RuntimeTensorOwnership.cpp
    ParametricLoop.cpp
    RuntimeTensorView.cpp
    StridedBatchedGemm.cpp
    GroupedBatchedGemm.cpp
    GroupedSandwich.cpp
    GroupedGatherRotate.cpp
    GroupedElementwise.cpp
    GroupedScalarOps.cpp
    SpaceAnnotation.cpp
    SpaceRegistry.cpp
    SymbolicCost.cpp
    SpaceShapedDeclare.cpp
    SymbolicExtents.cpp
    StringEinsum.cpp
    TensorLifetime.cpp
    ThreadCostModel.cpp
    ThreadPlanning.cpp
    ThreadWidth.cpp
    TiledRuntimeEig.cpp
    TiledRuntimeEinsum.cpp
    TiledRuntimeElementwise.cpp
    TensorRankConcept.cpp
    UsageAnalysis.cpp
    VerifyLevelIndependence.cpp
    TiledRuntimeTensorLifetime.cpp
    View.cpp
    WidthBudget.cpp
    Workspace.cpp
    test_auto_view_python.py
    test_axpy_axpby_views_python.py
    test_batched_views_python.py
    test_blas_python.py
    test_gate_flags_python.py
    test_gather_python.py
    test_grouped_gather_rotate_python.py
    test_block_copy_python.py
    test_capture_parity_gaps_python.py
    test_binding_reachability_python.py
    test_bisect_driver_python.py
    test_capture_python.py
    test_approximation_python.py
    test_ccsd_reuse_python.py
    test_setup_subgraph_python.py
    test_delta_elimination_python.py
    test_diis_python.py
    test_diis_differential_python.py
    test_diis_cpp_differential_python.py
    test_direct_division_python.py
    test_direct_product_views_python.py
    test_einsum_conj_passes_python.py
    test_einsum_conj_properties_python.py
    test_einsum_python.py
    test_einsum_views_python.py
    test_element_transform_python.py
    test_layout_assignment_python.py
    test_multi_term_factorization_python.py
    test_error_reporting_python.py
    test_executor_python.py
    test_einsum_rejection_python.py
    test_fuzz_alias_equiv_python.py
    test_fuzz_diff_flat_python.py
    test_fuzz_diff_control_flow_python.py
    test_fuzz_diff_deep_nesting_python.py
    test_fuzz_diff_cross_executor_flat_python.py
    test_fuzz_diff_cross_executor_nested_python.py
    test_fuzz_diff_parallel_stress_python.py
    test_fuzz_diff_graph_scratch_python.py
    test_fuzz_diff_regressions_python.py
    test_fuzz_diff_roundtrip_python.py
    test_fuzz_diff_region_identity_python.py
    test_fuzz_diff_cse_python.py
    test_fuzz_diff_replay_python.py
    test_fuzz_diff_random_pipeline_python.py
    test_fuzz_diff_random_pipeline_complex_python.py
    test_fuzz_diff_deferred_python.py
    test_fuzz_diff_pipeline_python.py
    test_fuzz_diff_gemm_batching_python.py
    test_fuzz_diff_optin_python.py
    test_tier_measurement_python.py
    test_tier_opportunities_python.py
    test_pass_operand_lifetime_after_rewrite_python.py
    test_factorization_python.py
    test_fuzz_ergonomics_python.py
    test_gemm_views_python.py
    test_gemv_views_python.py
    test_ger_views_python.py
    test_graph_python.py
    test_grouped_scalar_ops_python.py
    test_hyp_conjugation_python.py
    test_hyp_controlflow_diff_python.py
    test_hyp_dotc_python.py
    test_hyp_einsum_diff_python.py
    test_hyp_elementwise_diff_python.py
    test_hyp_gemm_transpose_python.py
    test_hyp_gemv_gerc_python.py
    test_hyp_lapack_diff_python.py
    test_hyp_largedim_diff_python.py
    test_hyp_largerank_diff_python.py
    test_hyp_einsum_conj_python.py
    test_hyp_linalg_diff_python.py
    test_hyp_metamorphic_python.py
    test_hyp_program_diff_python.py
    test_hyp_region_passes_python.py
    test_hyp_symmgemm_diff_python.py
    test_hyp_tiled_diff_python.py
    test_lapack_python.py
    test_linalg_new_python.py
    test_loop_python.py
    test_lu_factorization_python.py
    test_mixed_ops_python.py
    test_mp2_python.py
    test_outer_sum_python.py
    test_pass_constant_folding_python.py
    test_pass_cse_python.py
    test_pass_dead_node_elimination_python.py
    test_pass_element_wise_fusion_python.py
    test_pass_inplace_optimization_python.py
    test_pass_linear_combination_contraction_folding_python.py
    test_pass_loop_invariant_hoisting_python.py
    test_pass_memory_planning_python.py
    test_pass_operand_lifetime_python.py
    test_pass_reorder_python.py
    test_pass_cross_space_validation_python.py
    test_pass_scale_absorption_python.py
    test_pass_scaling_analysis_python.py
    test_pass_space_propagation_python.py
    test_pass_symmetry_propagation_python.py
    test_permute_python.py
    test_scalar_writers_python.py
    test_scf_python.py
    test_scale_python.py
    test_spaces_python.py
    test_symbolic_extents_python.py
    test_special_values_python.py
    test_symm_gemm_views_python.py
    test_symmetrized_accumulation_python.py
    test_tensor_conj_methods_python.py
    test_tiled_conj_python.py
    test_tiled_ops_python.py
    test_tiled_parity_python.py
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
    # NOTE: the fuzz / property tests (FuzzDiff*, FuzzErgonomics, Hyp*) bound
    # their sample counts under a sanitizer run themselves (see
    # tests/unit/_sanitizer_scaling.py). That keeps each process small enough
    # that the ThreadSanitizer deadlock detector's exit-time lock-order-graph
    # traversal no longer overruns the stack, so no per-test detect_deadlocks=0
    # override is needed here and deadlock detection stays on for every test.
    einsums_add_python_unit_test(
      "Modules.ComputeGraph" "${_camel}Python"
      SCRIPT ${test}
    )
  endif ()
endforeach()

# SaveLoad's golden corpus lives beside the test rather than being generated:
# the promise it encodes is that a file written by an earlier build still loads,
# and a generated file cannot make that promise. The directory is passed in so
# the test can walk it without a working-directory assumption.
if(TARGET SaveLoad_test)
  target_compile_definitions(
    SaveLoad_test PRIVATE EINSUMS_GRAPH_IR_GOLDEN_DIR="${CMAKE_CURRENT_SOURCE_DIR}/goldens"
  )
endif()
