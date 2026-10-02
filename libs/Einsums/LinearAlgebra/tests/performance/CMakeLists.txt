#----------------------------------------------------------------------------------------------
# Copyright (c) The Einsums Developers. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
#----------------------------------------------------------------------------------------------

set(LinearAlgebraPerfTests
    BenchmarkBLAS.cpp
    BenchmarkBlockTileReduction.cpp
    BenchmarkBandwidth.cpp
    BenchmarkLAPACK.cpp
    BenchmarkPeakFlops.cpp
    BenchmarkSmallElementwise.cpp
    BenchmarkThreadScaling.cpp
)

# The machinery around contractions; everything else here is the contraction suite.
set(LinearAlgebraInfrastructureBenchmarks BenchmarkBlockTileReduction BenchmarkBandwidth BenchmarkPeakFlops BenchmarkSmallElementwise BenchmarkThreadScaling)

foreach(test ${LinearAlgebraPerfTests})
  get_filename_component(TestName ${test} NAME_WE)
  set(_suite contraction)
  if(TestName IN_LIST LinearAlgebraInfrastructureBenchmarks)
    set(_suite infrastructure)
  endif()

  einsums_add_executable(${TestName}_test INTERNAL_FLAGS SOURCES ${test} NOINSTALL)
  einsums_add_performance_test("Modules.LinearAlgebra" ${TestName} SUITE ${_suite})
endforeach()
