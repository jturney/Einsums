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
set(RuntimeTests SignalHandlers.cpp # InitializeFinalize.cpp
)

foreach(test ${RuntimeTests})
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
    ${TestName}_test INTERNAL_FLAGS
    SOURCES ${Sources} ${${TestName}_${TestExtension}_FLAGS}
    NOINSTALL
  )

  einsums_add_unit_test("Modules.Einsums.Runtime" ${TestName})
endforeach()

# Where the profiler session is exported has to be asked from outside the process, so this one
# drives the HelloWorld2 example rather than linking against the runtime it is testing.
einsums_add_python_unit_test(
  "Modules.Einsums.Runtime" ProfileSessionExport SCRIPT test_profile_session_export_python.py
)
# Every zone the runtime records around start-up and teardown is a
# LabeledSectionInternal, so a build with EINSUMS_WITH_PROFILER_INTERNAL off has
# no teardown zone for the export to be checked against - HelloWorld2 records no
# zones at all there. Say which build this is rather than letting the test
# assert on something the preprocessor removed.
#
# The EINSUMS_BUILD_PYTHON term is not redundant: einsums_add_python_unit_test
# returns without registering anything when Python is off, and set_property on a
# test that was never added is a hard configure error.
if(EINSUMS_BUILD_PYTHON AND EINSUMS_WITH_PROFILER_INTERNAL)
  set_property(
    TEST "Tests.Unit.Modules.Einsums.Runtime.ProfileSessionExport"
    APPEND
    PROPERTY ENVIRONMENT "EINSUMS_TEST_PROFILER_INTERNAL=1"
  )
endif()

# And which arm refuses a session request. With the profiler off the export is compiled
# out down to the refusal, which then names the build option rather than the server one.
# Set in BOTH configurations rather than only when on, so that an unset value means the
# test was run by hand instead of through ctest and can accept either diagnostic; guessing
# from silence would fail the one build the guess got wrong.
if(EINSUMS_BUILD_PYTHON)
  if(EINSUMS_WITH_PROFILER)
    set(_einsums_test_profiler 1)
  else()
    set(_einsums_test_profiler 0)
  endif()
  set_property(
    TEST "Tests.Unit.Modules.Einsums.Runtime.ProfileSessionExport"
    APPEND
    PROPERTY ENVIRONMENT "EINSUMS_TEST_PROFILER=${_einsums_test_profiler}"
  )
  unset(_einsums_test_profiler)
endif()
