#----------------------------------------------------------------------------------------------
# Copyright (c) The Einsums Developers. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
#----------------------------------------------------------------------------------------------

set(LinearAlgebraPerfTests
    BenchmarkBLAS.cpp
    BenchmarkBandwidth.cpp
    BenchmarkLAPACK.cpp
    BenchmarkPeakFlops.cpp
    BenchmarkThreadScaling.cpp
)

foreach(test ${LinearAlgebraPerfTests})
  get_filename_component(TestName ${test} NAME_WE)

  einsums_add_executable(${TestName}_test INTERNAL_FLAGS SOURCES ${test} NOINSTALL)
  einsums_add_performance_test("Modules.LinearAlgebra" ${TestName})
endforeach()
