#----------------------------------------------------------------------------------------------
# Copyright (c) The Einsums Developers. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
#----------------------------------------------------------------------------------------------

set(ComputeGraphPerfTests
    BenchmarkDispatchRoutes.cpp
    BenchmarkEagerCallOverhead.cpp
    BenchmarkExecutors.cpp
    BenchmarkGraphOverhead.cpp
    BenchmarkOuterProduct.cpp
)

# The machinery around contractions; everything else here is the contraction suite.
set(ComputeGraphInfrastructureBenchmarks BenchmarkEagerCallOverhead BenchmarkExecutors BenchmarkGraphOverhead)

foreach(test ${ComputeGraphPerfTests})
  get_filename_component(TestName ${test} NAME_WE)
  set(_suite contraction)
  if(TestName IN_LIST ComputeGraphInfrastructureBenchmarks)
    set(_suite infrastructure)
  endif()

  einsums_add_executable(
    ${TestName}_test INTERNAL_FLAGS
    SOURCES ${test}
    NOINSTALL
  )

  einsums_add_performance_test(
    "Modules.ComputeGraph" ${TestName} SUITE ${_suite}
  )
endforeach()

# The Tensor Contraction Benchmark's shapes (tcb_cases.csv, from the head-to-head harness). A
# full run of every case in both precisions takes several minutes, past the runner's default
# per-test timeout.
einsums_add_executable(BenchmarkTCB_test INTERNAL_FLAGS SOURCES BenchmarkTCB.cpp NOINSTALL)
target_compile_definitions(BenchmarkTCB_test PRIVATE EINSUMS_TCB_CASES="${CMAKE_CURRENT_SOURCE_DIR}/tcb_cases.csv")
einsums_add_performance_test("Modules.ComputeGraph" BenchmarkTCB SUITE contraction TIMEOUT 3600)
