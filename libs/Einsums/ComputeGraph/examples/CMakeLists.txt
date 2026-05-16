#----------------------------------------------------------------------------------------------
# Copyright (c) The Einsums Developers. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
#----------------------------------------------------------------------------------------------

if(EINSUMS_WITH_EXAMPLES)
  einsums_add_pseudo_target(Examples.Modules.Einsums.ComputeGraph)
  einsums_add_pseudo_dependencies(Examples.Modules Examples.Modules.Einsums.ComputeGraph)
  if(EINSUMS_WITH_TESTS AND EINSUMS_WITH_TESTS_EXAMPLES)
    einsums_add_pseudo_target(Tests.Examples.Modules.Einsums.ComputeGraph)
    einsums_add_pseudo_dependencies(
      Tests.Examples.Modules Tests.Examples.Modules.Einsums.ComputeGraph
    )
  endif()
else()
  return()
endif()

set(example_programs
    AsyncIO
    BasicCapture
    ControlFlow
    DeferredAllocation
    GraphRebind
    MixedOperations
    OptimizationPasses
    GEMMBatchingDemo
    ParallelExecution
    PipelineWorkflow
    SCFSimulation
    StringEinsum
    TaskPoolIntegration
    FockBuildTaskPool
    IntegralTransform
    ContinuationPipeline
    DistributedGEMM
    GPUOffload
    Views
)

foreach(example_program ${example_programs})
  set(sources ${example_program}.cpp)

  source_group("Source Files" FILES ${sources})

  einsums_add_executable(
    CG_${example_program} INTERNAL_FLAGS
    SOURCES ${sources}
    FOLDER "Examples/Modules/ComputeGraph"
    NOINSTALL
  )
  target_include_directories(CG_${example_program} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})

  einsums_add_example_target_dependencies("Modules.Einsums.ComputeGraph" CG_${example_program})

  if(EINSUMS_WITH_TESTS AND EINSUMS_WITH_TESTS_EXAMPLES)
    einsums_add_example_test(
      "Modules.Einsums.ComputeGraph" CG_${example_program}
    )
  endif()
endforeach()
