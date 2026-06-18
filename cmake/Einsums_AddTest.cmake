#----------------------------------------------------------------------------------------------
# Copyright (c) The Einsums Developers. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
#----------------------------------------------------------------------------------------------

if(EINSUMS_WITH_TESTS_VALGRIND)
  find_program(VALGRIND_EXECUTABLE valgrind REQUIRED)
endif()

#:
#: einsums_add_test
#:
#:    Register a runtime test with optional wrappers, Valgrind, and test properties.
#:
#:    Creates a CTest entry named ``<category>.<name>`` that invokes an executable (or target),
#:    optionally through a custom wrapper and/or Valgrind. Supports serial execution, cost,
#:    timeout, and expected‑failure semantics. If the resolved executable target ends with ``_test``,
#:    it can be linked to Einsums testing utilities when ``TESTING`` is set.
#:
#:    **Signature**
#:    ``einsums_add_test(<category> <name>
#:        [EXECUTABLE <exe-or-target>]
#:        [ARGS <...>]
#:        [RANKS <int>] [THREADS <int>]
#:        [COST <int>] [TIMEOUT <seconds>]
#:        [WRAPPER <cmd>]
#:        [RUN_SERIAL]
#:        [FAILURE_EXPECTED]
#:        [TESTING]
#:        [PERFORMANCE_TESTING]
#:        [MPIWRAPPER]
#:    )``
#:
#:    **Behavior**
#:    - Resolves ``EXECUTABLE`` in this order: ``<name>_test`` target → ``<name>`` target → raw path.
#:    - Prepends internal runtime args ``--einsums:no-install-signal-handlers`` and
#:      ``--einsums:no-profiler-report``.
#:    - Optionally prefixes the command with ``WRAPPER`` and/or Valgrind when
#:      ``EINSUMS_WITH_TESTS_VALGRIND`` is ON.
#:    - Sets CTest properties: ``RUN_SERIAL``, ``COST``, ``TIMEOUT``, ``WILL_FAIL`` as requested.
#:    - For real test targets ``<exe>_test`` with ``TESTING``: links ``Catch2::Catch2`` and
#:      ``einsums_testing``.
#:
#:    **Notes**
#:    - ``RANKS`` and ``THREADS`` are parsed for future use; ``THREADS`` is clamped by
#:      ``EINSUMS_WITH_TESTS_MAX_THREADS`` when > 0.
#:    - When Valgrind is enabled globally, it is discovered via ``find_program(valgrind)``.
#:
#:    **Example**
#:    .. code-block:: cmake
#:
#:       einsums_add_test(Tests.Unit TensorOps EXECUTABLE tensor_ops_test RUN_SERIAL TIMEOUT 120)
function(einsums_add_test category name)
  set(options FAILURE_EXPECTED RUN_SERIAL TESTING PERFORMANCE_TESTING MPIWRAPPER)
  set(one_value_args COST EXECUTABLE RANKS THREADS TIMEOUT WRAPPER)
  set(multi_value_args ARGS)
  cmake_parse_arguments(${name} "${options}" "${one_value_args}" "${multi_value_args}" ${ARGN})

  if(NOT ${name}_RANKS)
    set(${name}_RANKS 1)
  endif()

  if(NOT ${name}_THREADS)
    set(${name}_THREADS 1)
  elseif(EINSUMS_WITH_TESTS_MAX_THREADS GREATER 0 AND ${name}_THREADS GREATER
                                                      EINSUMS_WITH_TESTS_MAX_THREADS
  )
    set(${name}_THREADS ${CALIBIRI_WITH_TESTS_MAX_THREADS})
  endif()

  if(NOT ${name}_EXECUTABLE)
    set(${name}_EXECUTABLE ${name})
  endif()

  if(TARGET ${${name}_EXECUTABLE}_test)
    set(_exe "$<TARGET_FILE:${${name}_EXECUTABLE}_test>")
  elseif(TARGET ${${name}_EXECUTABLE})
    set(_exe "$<TARGET_FILE:${${name}_EXECUTABLE}>")
  else()
    set(_exe "${${name}_EXECUTABLE}")
  endif()

  if(${name}_RUN_SERIAL)
    set(run_serial TRUE)
  endif()

  set(args "--einsums:no-install-signal-handlers")
  set(args ${args} "--einsums:no-profiler-report")
  set(args "${${name}_ARGS}" "${${name}_UNPARSED_ARGUMENTS}" ${args})

  set(_script_location ${PROJECT_BINARY_DIR})

  set(cmd ${_exe})

  if(${name}_WRAPPER)
    list(PREPEND cmd "${${name}_WRAPPER}" ${_preflags_list_})
  endif()

  if(EINSUMS_WITH_TESTS_VALGRIND)
    set(valgrind_cmd ${VALGRIND_EXECUTABLE} ${EINSUMS_WITH_TESTS_VALGRIND_OPTIONS})
  endif()

  if(${name}_FAILURE_EXPECTED)
    set(precommand ${CMAKE_COMMAND} -E env)
  endif()

  set(_full_name "${category}.${name}")
  add_test(NAME "${category}.${name}" COMMAND ${precommand} ${valgrind_cmd} ${cmd} ${args})
  if(${run_serial})
    set_tests_properties("${_full_name}" PROPERTIES RUN_SERIAL TRUE)
  endif()
  if(${name}_COST)
    set_tests_properties("${_full_name}" PROPERTIES COST ${${name}_COST})
  endif()
  if(${name}_TIMEOUT)
    set_tests_properties("${_full_name}" PROPERTIES TIMEOUT ${${name}_TIMEOUT})
  endif()
  if(${name}_FAILURE_EXPECTED)
    set_tests_properties("${_full_name}" PROPERTIES WILL_FAIL TRUE)
  endif()

  if(TARGET ${${name}_EXECUTABLE}_test)
    set_target_properties(${${name}_EXECUTABLE}_test PROPERTIES RUNTIME_OUTPUT_DIRECTORY "")
  endif()

  # Only real tests, i.e. executables ending in _test, link to einsums_testing
  if(TARGET ${${name}_EXECUTABLE}_test AND ${name}_TESTING)
    target_link_libraries(${${name}_EXECUTABLE}_test PRIVATE Catch2::Catch2)
    target_link_libraries(${${name}_EXECUTABLE}_test PRIVATE einsums_testing)
  endif()

  if(TARGET ${${name}_EXECUTABLE}_test AND ${name}_PERFORMANCE_TESTING)
    # target_link_libraries(${${name}_EXECUTABLE}_test PRIVATE einsums_performance_testing)
  endif()

endfunction(einsums_add_test)

#:
#: einsums_add_test_target_dependencies
#:
#:    Create a pseudo‑target for a test and wire it into its category group.
#:
#:    Adds a buildable convenience target ``<category>.<name>`` and makes it depend on the
#:    appropriate test executable target (by default ``<name>_test``). Also hooks it under the
#:    category‑level pseudo‑target ``<category>``.
#:
#:    **Signature**
#:    ``einsums_add_test_target_dependencies(<category> <name> [PSEUDO_DEPS_NAME <target-base>])``
#:
#:    **Behavior**
#:    - If the category does **not** match ``Tests.Examples*``, the dependent test target name is
#:      assumed to end with ``_test``; otherwise no suffix is used.
#:    - Creates pseudo‑target ``<category>.<name>`` via ``einsums_add_pseudo_target``.
#:    - Attaches it to the category root via ``einsums_add_pseudo_dependencies``.
#:    - Makes the pseudo‑target depend on either ``<PSEUDO_DEPS_NAME><suffix>`` or ``<name><suffix>``.
#:
#:    **Example**
#:    .. code-block:: cmake
#:
#:       einsums_add_test_target_dependencies(Tests.Unit TensorOps)
function(einsums_add_test_target_dependencies category name)
  set(one_value_args PSEUDO_DEPS_NAME)
  cmake_parse_arguments(${name} "${options}" "${one_value_args}" "${multi_value_args}" ${ARGN})
  # default target_extension is _test but for examples.* target, it may vary
  if(NOT ("${category}" MATCHES "Tests.Examples*"))
    set(_ext "_test")
  endif()
  # Add a custom target for this example
  einsums_add_pseudo_target(${category}.${name})
  # Make pseudo-targets depend on master pseudo-target
  einsums_add_pseudo_dependencies(${category} ${category}.${name})
  # Add dependencies to pseudo-target
  if(${name}_PSEUDO_DEPS_NAME)
    # When the test depend on another executable name
    einsums_add_pseudo_dependencies(${category}.${name} ${${name}_PSEUDO_DEPS_NAME}${_ext})
  else()
    einsums_add_pseudo_dependencies(${category}.${name} ${name}${_ext})
  endif()
endfunction(einsums_add_test_target_dependencies)

#:
#: einsums_add_test_and_deps_test
#:
#:    Convenience wrapper to define a test and its pseudo‑target within the Tests hierarchy.
#:
#:    Constructs the full category path as ``Tests.<category>[.<subcategory>]`` and invokes
#:    ``einsums_add_test`` and
#:    ``einsums_add_test_target_dependencies`` with the same arguments.
#:
#:    **Signature**
#:    ``einsums_add_test_and_deps_test(<category> <subcategory> <name> [ARGS ...])``
#:
#:    **Example**
#:    .. code-block:: cmake
#:
#:       einsums_add_test_and_deps_test(Unit "" TensorOps EXECUTABLE tensor_ops_test RUN_SERIAL)
function(einsums_add_test_and_deps_test category subcategory name)
  if("${subcategory}" STREQUAL "")
    einsums_add_test(Tests.${category} ${name} ${ARGN})
    einsums_add_test_target_dependencies(Tests.${category} ${name} ${ARGN})
  else()
    einsums_add_test(Tests.${category}.${subcategory} ${name} ${ARGN})
    einsums_add_test_target_dependencies(Tests.${category}.${subcategory} ${name} ${ARGN})
  endif()
endfunction(einsums_add_test_and_deps_test)

#:
#: einsums_set_test_properties
#:
#:    Apply standard Einsums labels and sanitizer/profiler environment to a test.
#:
#:    Sets CTest properties on a given test name, including labels and environment variables used
#:    for profiling and sanitizer behavior.
#:
#:    **Signature**
#:    ``einsums_set_test_properties(<name> <labels>)``
#:
#:    **Behavior**
#:    - Sets ``LABELS`` to the provided list/string.
#:    - Sets ``ENVIRONMENT`` with:
#:      ``LLVM_PROFILE_FILE=<name>.profraw``,
#:      ``TSAN_OPTIONS=ignore_noninstrumented_modules=1``,
#:      ``LSAN_OPTIONS=suppression=${PROJECT_SOURCE_DIR}/devtools/lsan.supp``.
#:
#:    **Example**
#:    .. code-block:: cmake
#:
#:       einsums_set_test_properties(Tests.Unit.TensorOps "UNIT_ONLY")
function(einsums_set_test_properties name labels)
  set_tests_properties(
    ${name}
    PROPERTIES
      LABELS
      ${labels}
      ENVIRONMENT
      "LLVM_PROFILE_FILE=${name}.profraw;TSAN_OPTIONS=ignore_noninstrumented_modules=1;LSAN_OPTIONS=suppression=${PROJECT_SOURCE_DIR}/devtools/lsan.supp"
  )
endfunction()

#:
#: einsums_add_unit_test
#:
#:    Define a unit test under ``Tests.Unit.<subcategory>`` and wire its pseudo‑target.
#:
#:    Forwards to ``einsums_add_test_and_deps_test`` with ``TESTING`` enabled and then
#:    ``einsums_set_test_properties`` with the label ``UNIT_ONLY``.
#:
#:    **Signature**
#:    ``einsums_add_unit_test(<subcategory> <name> [ARGS ...])``
#:
#:    **Example**
#:    .. code-block:: cmake
#:
#:       einsums_add_unit_test(Math tensor_add EXECUTABLE tensor_add_test)
function(einsums_add_unit_test subcategory name)
  einsums_add_test_and_deps_test("Unit" "${subcategory}" ${name} ${ARGN} TESTING)
  einsums_set_test_properties("Tests.Unit.${subcategory}.${name}" "UNIT_ONLY")
endfunction(einsums_add_unit_test)

#:
#: einsums_add_regression_test
#:
#:    Define a regression test under ``Tests.Regressions.<subcategory>`` and wire its pseudo‑target.
#:
#:    Forwards to ``einsums_add_test_and_deps_test`` with ``TESTING`` enabled and then
#:    ``einsums_set_test_properties`` with the label ``REGRESSION_ONLY``.
#:
#:    **Signature**
#:    ``einsums_add_regression_test(<subcategory> <name> [ARGS ...])``
#:
#:    **Example**
#:    .. code-block:: cmake
#:
#:       einsums_add_regression_test(Parser bad_header EXECUTABLE parser_bad_header_test FAILURE_EXPECTED)
function(einsums_add_regression_test subcategory name)
  # ARGN needed in case we add a test with the same executable
  einsums_add_test_and_deps_test("Regressions" "${subcategory}" ${name} ${ARGN} TESTING)
  einsums_set_test_properties("Tests.Regressions.${subcategory}.${name}" "REGRESSION_ONLY")
endfunction(einsums_add_regression_test)

#:
#: einsums_add_performance_test
#:
#:    Define a performance test under ``Tests.Performance.<subcategory>`` and wire its pseudo‑target.
#:
#:    Forwards to ``einsums_add_test_and_deps_test`` with ``RUN_SERIAL`` and
#:    ``PERFORMANCE_TESTING`` enabled, then sets label ``PERFORMANCE_ONLY``.
#:
#:    **Signature**
#:    ``einsums_add_performance_test(<subcategory> <name> [ARGS ...])``
#:
#:    **Example**
#:    .. code-block:: cmake
#:
#:       einsums_add_performance_test(Benchmarks gemm_perf EXECUTABLE gemm_bench TIMEOUT 600)
function(einsums_add_performance_test subcategory name)
  einsums_add_test_and_deps_test(
    "Performance" "${subcategory}" ${name} ${ARGN} RUN_SERIAL PERFORMANCE_TESTING
  )
  einsums_set_test_properties("Tests.Performance.${subcategory}.${name}" "PERFORMANCE_ONLY")
endfunction(einsums_add_performance_test)

#:
#: einsums_add_example_test
#:
#:    Register an example as a test under ``Tests.Examples.<subcategory>`` and wire its pseudo‑target.
#:
#:    Forwards to ``einsums_add_test_and_deps_test`` and labels the test ``EXAMPLES_ONLY``.
#:
#:    **Signature**
#:    ``einsums_add_example_test(<subcategory> <name> [ARGS ...])``
#:
#:    **Example**
#:    .. code-block:: cmake
#:
#:       einsums_add_example_test(IO read_write_example EXECUTABLE read_write_example)
function(einsums_add_example_test subcategory name)
  einsums_add_test_and_deps_test("Examples" "${subcategory}" ${name} ${ARGN})
  einsums_set_test_properties("Tests.Examples.${subcategory}.${name}" "EXAMPLES_ONLY")
endfunction(einsums_add_example_test)

#:
#: einsums_add_example_target_dependencies
#:
#:    Create an ``Examples.<subcategory>.<name>`` pseudo‑target and attach it to its category.
#:
#:    This enables category‑scoped builds like ``cmake --build . --target Examples.<subcategory>``.
#:
#:    **Signature**
#:    ``einsums_add_example_target_dependencies(<subcategory> <name> [DEPS_ONLY])``
#:
#:    **Behavior**
#:    - Unless ``DEPS_ONLY`` is set, defines the pseudo‑target ``Examples.<subcategory>.<name>``.
#:    - Always wires it under the category target ``Examples.<subcategory>``.
#:    - Adds a dependency from the pseudo‑target to the real executable target ``<name>``.
#:
#:    **Example**
#:    .. code-block:: cmake
#:
#:       einsums_add_example_target_dependencies(IO convert_image)
function(einsums_add_example_target_dependencies subcategory name)
  set(options DEPS_ONLY)
  cmake_parse_arguments(${name} "${options}" "${one_value_args}" "${multi_value_args}" ${ARGN})
  if(NOT ${name}_DEPS_ONLY)
    # Add a custom target for this example
    einsums_add_pseudo_target(Examples.${subcategory}.${name})
  endif()
  # Make pseudo-targets depend on master pseudo-target
  einsums_add_pseudo_dependencies(Examples.${subcategory} Examples.${subcategory}.${name})
  # Add dependencies to pseudo-target
  einsums_add_pseudo_dependencies(Examples.${subcategory}.${name} ${name})
endfunction(einsums_add_example_target_dependencies)
