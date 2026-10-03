#----------------------------------------------------------------------------------------------
# Copyright (c) The Einsums Developers. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
#----------------------------------------------------------------------------------------------

# Waggle's tests link the collector and Catch2's main, nothing else, so they build and run without
# any library that uses Waggle. `waggle_tests` builds them all.

set(_waggle_unit_tests
    Annotations
    Consumer
    Domains
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

# A second, separately built copy of the collector, for the test that loads two at once. Its sources
# and settings are the collector's own; only the file differs.
get_target_property(_waggle_sources waggle SOURCES)
list(TRANSFORM _waggle_sources PREPEND ${PROJECT_SOURCE_DIR}/)
add_library(waggle_second_copy SHARED EXCLUDE_FROM_ALL ${_waggle_sources})
target_compile_features(waggle_second_copy PUBLIC cxx_std_20)
target_include_directories(waggle_second_copy PRIVATE ${PROJECT_SOURCE_DIR}/include ${PROJECT_SOURCE_DIR}/src)
target_compile_definitions(waggle_second_copy PRIVATE WAGGLE_EXPORTS)
target_link_libraries(waggle_second_copy PRIVATE fmt::fmt Threads::Threads ${CMAKE_DL_LIBS})
set_target_properties(waggle_second_copy PROPERTIES CXX_VISIBILITY_PRESET hidden VISIBILITY_INLINES_HIDDEN ON)

if(NOT WIN32)
  add_executable(waggle_Duplicates_test unit/Duplicates.cpp)
  target_link_libraries(waggle_Duplicates_test PRIVATE Waggle::internal Catch2::Catch2WithMain ${CMAKE_DL_LIBS})
  target_compile_definitions(waggle_Duplicates_test PRIVATE WAGGLE_SECOND_COPY_PATH="$<TARGET_FILE:waggle_second_copy>")
  add_dependencies(waggle_Duplicates_test waggle_second_copy)
  add_dependencies(waggle_tests waggle_Duplicates_test)
  add_test(NAME ${WAGGLE_TEST_PREFIX}Duplicates COMMAND waggle_Duplicates_test)
  set_tests_properties(${WAGGLE_TEST_PREFIX}Duplicates PROPERTIES ENVIRONMENT "WAGGLE_REPORT=false" TIMEOUT 300)
endif()
