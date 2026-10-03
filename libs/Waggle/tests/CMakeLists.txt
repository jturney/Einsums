#----------------------------------------------------------------------------------------------
# Copyright (c) The Einsums Developers. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
#----------------------------------------------------------------------------------------------

# Waggle's tests link the collector and Catch2's main, nothing else, so they build and run without
# any library that uses Waggle. `waggle_tests` builds them all.

set(_waggle_unit_tests
    Annotations
    Consumer
    Producer
    RingBuffer
    Server
    Settings
    Sites
    Snapshot
    StringTable
)

add_custom_target(waggle_tests)

foreach(_name IN LISTS _waggle_unit_tests)
  set(_target waggle_${_name}_test)
  add_executable(${_target} unit/${_name}.cpp)
  # Waggle::internal: the tests reach the collector's private headers.
  target_link_libraries(${_target} PRIVATE Waggle::internal Catch2::Catch2WithMain)
  add_dependencies(waggle_tests ${_target})

  add_test(NAME ${WAGGLE_TEST_PREFIX}${_name} COMMAND ${_target})
  # No report at exit: it would bury the test output.
  set_tests_properties(${WAGGLE_TEST_PREFIX}${_name} PROPERTIES ENVIRONMENT "WAGGLE_REPORT=false" TIMEOUT 300)
endforeach()

# A plain program, not a Catch2 case: what it checks happens after main returns.
add_executable(waggle_ExitWithoutFinalize_test unit/ExitWithoutFinalize.cpp)
target_link_libraries(waggle_ExitWithoutFinalize_test PRIVATE Waggle::waggle)
add_dependencies(waggle_tests waggle_ExitWithoutFinalize_test)
add_test(NAME ${WAGGLE_TEST_PREFIX}ExitWithoutFinalize COMMAND waggle_ExitWithoutFinalize_test)
set_tests_properties(${WAGGLE_TEST_PREFIX}ExitWithoutFinalize PROPERTIES ENVIRONMENT "WAGGLE_REPORT=false" TIMEOUT 300)
