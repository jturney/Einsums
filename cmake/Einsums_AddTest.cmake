#----------------------------------------------------------------------------------------------
# Copyright (c) The Einsums Developers. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
#----------------------------------------------------------------------------------------------

if (EINSUMS_WITH_TESTS_VALGRIND)
  find_program(VALGRIND_EXECUTABLE valgrind REQUIRED)
endif ()

function(einsums_add_test category name)
  set(options FAILURE_EXPECTED RUN_SERIAL TESTING PROFORMANCE_TESTING MPIWRAPPER)
  set(one_value_args COST EXECUTABLE RANKS THREADS TIMEOUT WRAPPER)
  set(multi_value_args ARGS)
  cmake_parse_arguments(${name} "${options}" "${one_value_args}" "${multi_value_args}" ${ARGN})

  if (NOT ${name}_RANKS)
    set(${name}_RANKS 1)
  endif ()

  if (NOT ${name}_THREADS)
    set(${name}_THREADS 1)
  elseif (EINSUMS_WITH_TESTS_MAX_THREADS GREATER 0 AND ${name}_THREADS GREATER
      EINSUMS_WITH_TESTS_MAX_THREADS
  )
    set(${name}_THREADS ${CALIBIRI_WITH_TESTS_MAX_THREADS})
  endif ()

  if (NOT ${name}_EXECUTABLE)
    set(${name}_EXECUTABLE ${name})
  endif ()

  if (TARGET ${${name}_EXECUTABLE}_test)
    set(_exe "$<TARGET_FILE:${${name}_EXECUTABLE}_test>")
  elseif (TARGET ${${name}_EXECUTABLE})
    set(_exe "$<TARGET_FILE:${${name}_EXECUTABLE}>")
  else ()
    set(_exe "${${name}_EXECUTABLE}")
  endif ()

  if (${name}_RUN_SERIAL)
    set(run_serial TRUE)
  endif ()

  # If --einsums:threads=cores or all
  if (${name}_THREADS LESS_EQUAL 0)
    set(run_serial TRUE)
    if (${name}_THREADS EQUAL -1)
      set(${name}_THREADS "all")
    elseif (${name}_THREADS EQUAL -2)
      set(${name}_THREADS "cores")
    endif ()
  endif ()

  set(args "--einsums:threads=${${name}_THREADS}")
  if (EINSUMS_WITH_TESTS_DEBUG_LOG)
    set(args ${args} "--einsums:debug-einsums-log=${EINSUMS_WITH_TESTS_DEBUG_LOG_DESTINATION}")
  endif ()

  if (EINSUMS_WITH_PARALLEL_TESTS_BIND_NONE
      AND NOT run_serial
      AND NOT "${name}_MPIWRAPPER"
  )
    set(args ${args} "--einsums:bind=none")
  endif ()

  set(args "${${name}_ARGS}" "${${name}_UNPARSED_ARGUMENTS}" ${args})

  set(_script_location ${PROJECT_BINARY_DIR})

  set(cmd ${_exe})

  if (${name}_WRAPPER)
    list(PREPEND cmd "${${name}_WRAPPER}" ${_preflags_list_})
  endif ()

  if (${name}_MPIWRAPPER)
    set(_preflags_list_ ${MPIEXEC_PREFLAGS})
    separate_arguments(_preflags_list_)
    list(PREPEND cmd "${MPIEXEC_EXECUTABLE}" "${MPIEXEC_NUMPROC_FLAG}" "${${name}_RANKS}"
        ${_preflags_list_}
    )
  endif ()

  if (EINSUMS_WITH_TESTS_VALGRIND)
    set(valgrind_cmd ${VALGRIND_EXECUTABLE} ${EINSUMS_WITH_TESTS_VALGRIND_OPTIONS})
  endif ()

  set(_full_name "${category}.${name}")
  add_test(NAME "${category}.${name}" COMMAND ${valgrind_cmd} ${cmd} ${args})
  if (${run_serial})
    set_tests_properties("${_full_name}" PROPERTIES RUN_SERIAL TRUE)
  endif ()
  if (${name}_COST)
    set_tests_properties("${_full_name}" PROPERTIES COST ${${name}_COST})
  endif ()
  if (${name}_TIMEOUT)
    set_tests_properties("${_full_name}" PROPERTIES TIMEOUT ${${name}_TIMEOUT})
  endif ()
  if (${name}_FAILURE_EXPECTED)
    set_tests_properties("${_full_name}" PROPERTIES WILL_FAIL TRUE)
  endif ()

  # Only real tests, i.e. executables ending in _test, link to einsums_testing
  if (TARGET ${${name}_EXECUTABLE}_test AND ${name}_TESTING)
    target_link_libraries(${${name}_EXECUTABLE}_test PRIVATE snitch::snitch)
    #    target_link_libraries(${${name}_EXECUTABLE}_test PRIVATE einsums_testing)
  endif ()

  if (TARGET ${${name}_EXECUTABLE}_test AND ${name}_PERFORMANCE_TESTING)
    #    target_link_libraries(${${name}_EXECUTABLE}_test PRIVATE einsums_performance_testing)
  endif ()

endfunction(einsums_add_test)

function(einsums_add_test_target_dependencies category name)
  set(one_value_args PSEUDO_DEPS_NAME)
  cmake_parse_arguments(${name} "${options}" "${one_value_args}" "${multi_value_args}" ${ARGN})
  # default target_extension is _test but for examples.* target, it may vary
  if (NOT ("${category}" MATCHES "tests.examples*"))
    set(_ext "_test")
  endif ()
  # Add a custom target for this example
  einsums_add_pseudo_target(${category}.${name})
  # Make pseudo-targets depend on master pseudo-target
  einsums_add_pseudo_dependencies(${category} ${category}.${name})
  # Add dependencies to pseudo-target
  if (${name}_PSEUDO_DEPS_NAME)
    # When the test depend on another executable name
    einsums_add_pseudo_dependencies(${category}.${name} ${${name}_PSEUDO_DEPS_NAME}${_ext})
  else ()
    einsums_add_pseudo_dependencies(${category}.${name} ${name}${_ext})
  endif ()
endfunction(einsums_add_test_target_dependencies)

# To add test to the category root as in tests/regressions/ with correct name
function(einsums_add_test_and_deps_test category subcategory name)
  if ("${subcategory}" STREQUAL "")
    einsums_add_test(tests.${category} ${name} ${ARGN})
    einsums_add_test_target_dependencies(tests.${category} ${name} ${ARGN})
  else ()
    einsums_add_test(tests.${category}.${subcategory} ${name} ${ARGN})
    einsums_add_test_target_dependencies(tests.${category}.${subcategory} ${name} ${ARGN})
  endif ()
endfunction(einsums_add_test_and_deps_test)

# Only unit and regression tests link to the testing library. Performance tests and examples don't
# link to the testing library. Performance tests link to the performance_testing library.
function(einsums_add_unit_test subcategory name)
  einsums_add_test_and_deps_test("unit" "${subcategory}" ${name} ${ARGN} TESTING)
endfunction(einsums_add_unit_test)

function(einsums_add_regression_test subcategory name)
  # ARGN needed in case we add a test with the same executable
  einsums_add_test_and_deps_test("regressions" "${subcategory}" ${name} ${ARGN} TESTING)
endfunction(einsums_add_regression_test)

function(einsums_add_performance_test subcategory name)
  einsums_add_test_and_deps_test(
      "performance" "${subcategory}" ${name} ${ARGN} RUN_SERIAL PERFORMANCE_TESTING
  )
endfunction(einsums_add_performance_test)

function(einsums_add_example_test subcategory name)
  einsums_add_test_and_deps_test("examples" "${subcategory}" ${name} ${ARGN})
endfunction(einsums_add_example_test)

# To create target examples.<name> when calling make examples need 2 distinct rules for examples and
# tests.examples
function(einsums_add_example_target_dependencies subcategory name)
  set(options DEPS_ONLY)
  cmake_parse_arguments(${name} "${options}" "${one_value_args}" "${multi_value_args}" ${ARGN})
  if (NOT ${name}_DEPS_ONLY)
    # Add a custom target for this example
    einsums_add_pseudo_target(examples.${subcategory}.${name})
  endif ()
  # Make pseudo-targets depend on master pseudo-target
  einsums_add_pseudo_dependencies(examples.${subcategory} examples.${subcategory}.${name})
  # Add dependencies to pseudo-target
  einsums_add_pseudo_dependencies(examples.${subcategory}.${name} ${name})
endfunction(einsums_add_example_target_dependencies)
