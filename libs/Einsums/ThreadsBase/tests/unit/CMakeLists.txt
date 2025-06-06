#----------------------------------------------------------------------------------------------
# Copyright (c) The Einsums Developers. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
#----------------------------------------------------------------------------------------------

#
# Example of a more involved unit test file that includes standard C++ and a HIP test. Example1.cpp
# requires an additional library to successfully link. Example2.hip should only run on one thread.
#
# set(ExampleTests Example1.cpp Example2.hip )
#
# set(Example1_cpp_FLAGS DEPENDENCIES SomeOtherGreatLibrary) set(Example2_hip_PARAMETERS THREADS 1)

# Name of test file. Include .cpp, .hip, etc. extension.
set(ThreadsBaseTests)

foreach(test ${ThreadsBaseTests})
  # Extract base name of test Example1.cpp -> Example1, Example.2.hip -> Example
  get_filename_component(TestName ${test} NAME_WE)
  # Extract extension of the test Example1.cpp -> .cpp, Example.2.hip -> .2.hip
  get_filename_component(Extension ${test} EXT)

  # Remove the leading dot and replace any periods with underscores
  string(SUBSTRING ${Extension} 1 -1 EXT_NO_DOT) # Remove the leading dot
  string(REPLACE "." "_" TestExtension ${EXT_NO_DOT})

  # Check if the cleaned extension is "cu" or "hip"
  if(NOT EINSUMS_WITH_GPU_SUPPORT AND (TestExtension STREQUAL "cu" OR TestExtension STREQUAL "hip"))
    # Skip this iteration
    continue()
  endif()

  set(Sources ${test})

  source_group("Source Files" FILES ${Sources})

  einsums_add_executable(
    ${TestName}_test INTERNAL_FLAGS SOURCES ${Sources} ${${TestName}_${TestExtension}_FLAGS}
                                            NOINSTALL
  )

  einsums_add_unit_test(
    "Modules.Einsums.ThreadsBase" ${TestName} ${${TestName}_${TestExtension}_PARAMETERS}
  )
endforeach()
