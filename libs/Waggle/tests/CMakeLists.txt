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

# The C interface from C: the header must compile as strict C, without the C++ layer.
enable_language(C)
add_executable(waggle_CApi_test unit/CApi.c)
target_link_libraries(waggle_CApi_test PRIVATE Waggle::waggle)
set_target_properties(waggle_CApi_test PROPERTIES C_STANDARD 11 C_STANDARD_REQUIRED ON C_EXTENSIONS OFF)
if(NOT MSVC)
  target_compile_options(waggle_CApi_test PRIVATE -Wall -Wextra -pedantic -Werror)
endif()
add_dependencies(waggle_tests waggle_CApi_test)
add_test(NAME ${WAGGLE_TEST_PREFIX}CApi COMMAND waggle_CApi_test)
set_tests_properties(${WAGGLE_TEST_PREFIX}CApi PROPERTIES ENVIRONMENT "WAGGLE_REPORT=false" TIMEOUT 300)

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

# The viewer's tests, run from the source tree. They need pytest; the app's need Textual too, and
# skip themselves without it.
find_package(Python3 COMPONENTS Interpreter)
if(Python3_Interpreter_FOUND)
  execute_process(
    COMMAND ${Python3_EXECUTABLE} -c "import pytest"
    RESULT_VARIABLE _waggle_no_pytest
    OUTPUT_QUIET ERROR_QUIET
  )
  if(_waggle_no_pytest EQUAL 0)
    add_test(NAME ${WAGGLE_TEST_PREFIX}ViewerPython COMMAND ${Python3_EXECUTABLE} -m pytest -q -p no:cacheprovider
                                                            ${CMAKE_CURRENT_SOURCE_DIR}/python
    )
    set_tests_properties(
      ${WAGGLE_TEST_PREFIX}ViewerPython PROPERTIES ENVIRONMENT "PYTHONPATH=${PROJECT_SOURCE_DIR}/python" LABELS PYTHON TIMEOUT 600
    )
  else()
    message(STATUS "Waggle: pytest not found; the viewer's tests are not registered")
  endif()
endif()

# What one zone costs, whole and in pieces. Not a test: run it by hand, paired against another build,
# on a quiet machine on mains power.
add_executable(waggle_ZoneCost_benchmark EXCLUDE_FROM_ALL performance/ZoneCost.cpp)
target_link_libraries(waggle_ZoneCost_benchmark PRIVATE Waggle::internal Catch2::Catch2WithMain)
add_custom_target(waggle_benchmarks DEPENDS waggle_ZoneCost_benchmark)
