#----------------------------------------------------------------------------------------------
# Copyright (c) The Einsums Developers. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
#----------------------------------------------------------------------------------------------

# What these measure exists only in a build with the profiler.
set(ProfilePerfTests)
if(EINSUMS_WITH_PROFILER)
  list(APPEND ProfilePerfTests BenchmarkZoneCost.cpp)
endif()

foreach(test ${ProfilePerfTests})
  get_filename_component(TestName ${test} NAME_WE)

  einsums_add_executable(
    ${TestName}_test INTERNAL_FLAGS
    SOURCES ${test}
    NOINSTALL
  )

  einsums_add_performance_test("Modules.Profile" ${TestName} SUITE infrastructure)
endforeach()
