#----------------------------------------------------------------------------------------------
# Copyright (c) The Einsums Developers. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
#----------------------------------------------------------------------------------------------

set(HPTTPerfTests
    BenchmarkTranspose.cpp
    BenchmarkPlanCache.cpp
)

foreach(test ${HPTTPerfTests})
  get_filename_component(TestName ${test} NAME_WE)

  einsums_add_executable(${TestName}_test INTERNAL_FLAGS SOURCES ${test} NOINSTALL)
  einsums_add_performance_test("Modules.HPTT" ${TestName})
endforeach()
