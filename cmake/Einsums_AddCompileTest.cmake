#----------------------------------------------------------------------------------------------
# Copyright (c) The Einsums Developers. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
#----------------------------------------------------------------------------------------------

#:
#: einsums_add_compile_test
#:
#:    Add a build-only test that verifies whether a source compiles.
#:
#:    This function defines a CMake test whose only purpose is to ensure that a
#:    source file (or set of files) can be compiled successfully — without
#:    executing any code. It is typically used to test whether headers, flags,
#:    or compile-time configurations are valid in isolation.
#:
#:    **Signature**
#:    ``einsums_add_compile_test(<category> <name> [OPTIONS...])``
#:
#:    **Positional Arguments**
#:
#:    - ``category`` *(required)*:
#:      A test category (e.g., `compile_flags`, `diagnostics`, `headers`). Used to namespace the test name.
#:
#:    - ``name`` *(required)*:
#:      A unique name for this test within the category. Combined with `category`
#:      to form the actual test name as `<category>.<name>`.
#:
#:    **Keyword Arguments**
#:
#:    - ``SOURCES`` *(required, multi-value)*:
#:      Source file(s) to be compiled. Can be a relative or absolute path.
#:
#:    - ``SOURCE_ROOT`` *(optional)*:
#:      Root directory for the sources (used by Einsums functions like `einsums_add_executable`).
#:
#:    - ``FOLDER`` *(optional)*:
#:      IDE folder or logical grouping (e.g., for Visual Studio or Xcode projects).
#:
#:    - ``DEPENDENCIES`` *(optional, multi-value)*:
#:      Additional CMake targets or libraries that this test depends on.
#:
#:    - ``FAILURE_EXPECTED`` *(optional flag)*:
#:      If set, the test will pass only if the compilation **fails** (e.g., to test compiler diagnostics).
#:
#:    - ``NOLIBS`` *(optional flag)*:
#:      Prevents linking against standard Einsums libraries. Use for lightweight compile-only checks.
#:
#:    - ``OBJECT`` *(optional flag)*:
#:      Compiles the source into an object file via `einsums_add_library` instead of an executable.
#:
#:    **Behavior**
#:
#:    This function:
#:
#:    1. Registers a test named `<category>.<name>` using CMake’s ``add_test``.
#:
#:    2. Adds a compile target using either:
#:       - ``einsums_add_library`` (if `OBJECT` is set), or
#:       - ``einsums_add_executable`` (default).
#:
#:    3. Sets the test to trigger a build of the target (but does not run it).
#:
#:    4. Tags the test with the label ``COMPILE_ONLY``.
#:
#:    5. If `FAILURE_EXPECTED` is enabled, sets `WILL_FAIL TRUE` so the test passes only on failure.
#:
#:    **Test Result**
#:
#:    The test **passes** if the target compiles successfully (unless `FAILURE_EXPECTED` is set).
#:
#:    **Example**
#:
#:    .. code-block:: cmake
#:
#:       einsums_add_compile_test(
#:         flags -march_native
#:         SOURCES test/compile/march_native.cpp
#:         CONFIGURATIONS Release
#:         PUBLIC
#:         LANGUAGES CXX
#:       )
#:
#:       einsums_add_compile_test(
#:         diagnostics invalid_api_usage
#:         SOURCES test/compile/invalid_api.cpp
#:         FAILURE_EXPECTED
#:       )
#:
#:    **See also**
#:
#:    - ``einsums_add_executable``
#:    - ``einsums_add_library``
#:    - ``add_test``
#:    - ``set_tests_properties``
function(einsums_add_compile_test category name)
  set(options FAILURE_EXPECTED NOLIBS OBJECT)
  set(one_value_args SOURCE_ROOT FOLDER)
  set(multi_value_args SOURCES DEPENDENCIES)
  cmake_parse_arguments(${name} "${options}" "${one_value_args}" "${multi_value_args}" ${ARGN})

  set(_additional_flags)
  if(${name}_NOLIBS)
    set(_additional_flags ${_additional_flags} NOLIBS)
  endif()
  if(${name}_OBJECT)
    set(_additional_flags ${_additional_flags} OBJECT)
  endif()

  string(REGEX REPLACE "\\." "_" test_name "${category}.${name}")

  if(${name}_OBJECT)
    einsums_add_library(
      ${test_name}
      SOURCE_ROOT ${${name}_SOURCE_ROOT}
      SOURCES ${${name}_SOURCES}
      EXCLUDE_FROM_ALL EXCLUDE_FROM_DEFAULT_BUILD
      FOLDER ${${name}_FOLDER}
      DEPENDENCIES ${${name}_DEPENDENCIES} ${_additional_flags}
    )
  else()
    einsums_add_executable(
      ${test_name}
      SOURCE_ROOT ${${name}_SOURCE_ROOT}
      SOURCES ${${name}_SOURCES}
      EXCLUDE_FROM_ALL EXCLUDE_FROM_DEFAULT_BUILD
      FOLDER ${${name}_FOLDER}
      DEPENDENCIES ${${name}_DEPENDENCIES} ${_additional_flags}
    )
  endif()

  add_test(
    NAME "${category}.${name}"
    COMMAND ${CMAKE_COMMAND} --build ${PROJECT_BINARY_DIR} --target ${test_name} --config
            $<CONFIGURATION>
    WORKING_DIRECTORY ${PROJECT_BINARY_DIR}
  )

  set_tests_properties("${category}.${name}" PROPERTIES LABELS "COMPILE_ONLY")

  if(${name}_FAILURE_EXPECTED)
    set_tests_properties("${category}.${name}" PROPERTIES WILL_FAIL TRUE)
  endif()
endfunction()

#:
#: einsums_add_compile_test_target_dependencies
#:
#:    Register a pseudo-target for a compile test and link it to its category group.
#:
#:    This function creates a CMake pseudo-target (an alias-like target that doesn’t build anything)
#:    for a compile-only test, and wires it into a higher-level category target. It enables convenient
#:    group-based builds (e.g., `make compile_flags` builds all compile tests in the `compile_flags` category).
#:
#:    **Signature**
#:    ``einsums_add_compile_test_target_dependencies(<category> <name>)``
#:
#:    **Arguments**
#:
#:    - ``category`` *(required)*:
#:      The category of the compile test (e.g., `headers`, `flags`, `diagnostics`). This also names the master pseudo-target.
#:
#:    - ``name`` *(required)*:
#:      The specific test name within the category. A pseudo-target named `<category>.<name>` is created.
#:
#:    **Behavior**
#:
#:    This function:
#:
#:    1. Creates a pseudo-target named `<category>.<name>` via ``einsums_add_pseudo_target``.
#:
#:    2. Registers it as a dependency of the category-wide pseudo-target `<category>` via
#:       ``einsums_add_pseudo_dependencies``.
#:
#:    3. Enables easy invocation of category-wide test targets, e.g.:
#:       - `ninja compile_flags`
#:       - `cmake --build . --target headers.diagnostics`
#:
#:    **Example**
#:
#:    .. code-block:: cmake
#:
#:       einsums_add_compile_test_target_dependencies(headers header_stability)
#:
#:    This would:
#:
#:    - Create a target `headers.header_stability`
#:    - Make it part of the `headers` pseudo-target
#:
#:    **See also**
#:
#:    - ``einsums_add_compile_test``
#:    - ``einsums_add_pseudo_target``
#:    - ``einsums_add_pseudo_dependencies``
#:    - ``add_custom_target``
function(einsums_add_compile_test_target_dependencies category name)
  einsums_add_pseudo_target(${category}.${name})
  # make pseudo-targets depend on master pseudo-target
  einsums_add_pseudo_dependencies(${category} ${category}.${name})
endfunction()

#:
#: einsums_add_test_and_deps_compile_test
#:
#:    Register a compile-only test along with its pseudo-target and category grouping.
#:
#:    This function is a convenience wrapper that defines a compile-time test and associates it
#:    with a structured naming hierarchy (`Tests.<category>[.<subcategory>]`) and a pseudo-target
#:    dependency group for easy batch builds.
#:
#:    **Signature**
#:    ``einsums_add_test_and_deps_compile_test(<category> <subcategory> <name> [OPTIONS...])``
#:
#:    **Positional Arguments**
#:
#:    - ``category`` *(required)*:
#:      Top-level grouping of the test (e.g., `headers`, `flags`, `features`). Used to form test and target names.
#:
#:    - ``subcategory`` *(optional, pass empty string if unused)*:
#:      Optional second-level grouping (e.g., `CUDA`, `GCC`, `MSVC`). If omitted or empty, no subcategory is added.
#:
#:    - ``name`` *(required)*:
#:      Unique name for the test within the given category (and optional subcategory).
#:
#:    **Forwarded Arguments**
#:
#:    All additional keyword arguments (`SOURCES`, `SOURCE_ROOT`, `DEPENDENCIES`, `CONFIGURATIONS`, etc.)
#:    are forwarded directly to ``einsums_add_compile_test``.
#:
#:    **Behavior**
#:
#:    - Constructs a hierarchical test name of the form:
#:      - ``Tests.<category>.<subcategory>.<name>`` (if subcategory is non-empty)
#:      - ``Tests.<category>.<name>`` (if subcategory is empty)
#:
#:    - Registers the test via ``einsums_add_compile_test``.
#:
#:    - Creates a pseudo-target for the test via ``einsums_add_compile_test_target_dependencies``,
#:      enabling grouped builds like:
#:        - `cmake --build . --target Tests.headers`
#:        - `ninja Tests.compile_flags.GNU`
#:
#:    **Example**
#:
#:    .. code-block:: cmake
#:
#:       einsums_add_test_and_deps_compile_test(
#:         flags
#:         GCC
#:         no_unused_macros
#:         SOURCES test/compile/unused_macros.cpp
#:         CONFIGURATIONS Debug
#:       )
#:
#:       einsums_add_test_and_deps_compile_test(
#:         headers
#:         ""
#:         memory_safety
#:         SOURCES test/compile/memory_safe.cpp
#:         FAILURE_EXPECTED
#:       )
#:
#:    These define:
#:    - `Tests.flags.GCC.no_unused_macros`
#:    - `Tests.headers.memory_safety`
#:
#:    And register pseudo-targets for:
#:    - `Tests.flags.GCC`
#:    - `Tests.headers`
#:
#:    **See also**
#:
#:    - ``einsums_add_compile_test``
#:    - ``einsums_add_compile_test_target_dependencies``
#:    - ``add_test``
function(einsums_add_test_and_deps_compile_test category subcategory name)
  if("${subcategory}" STREQUAL "")
    einsums_add_compile_test(Tests.${category} ${name} ${ARGN})
    einsums_add_compile_test_target_dependencies(Tests.${category} ${name})
  else()
    einsums_add_compile_test(Tests.${category}.${subcategory} ${name} ${ARGN})
    einsums_add_compile_test_target_dependencies(Tests.${category}.${subcategory} ${name})
  endif()
endfunction(einsums_add_test_and_deps_compile_test)

#:
#: einsums_add_unit_compile_test
#:
#:    Register a compile-only unit test under the `Unit` test category.
#:
#:    This is a wrapper around ``einsums_add_test_and_deps_compile_test``
#:    that hardcodes the top-level category to `Unit`, enabling you to more concisely
#:    define compile-only tests that check low-level correctness or API usage in isolation.
#:
#:    **Signature**
#:    ``einsums_add_unit_compile_test(<subcategory> <name> [OPTIONS...])``
#:
#:    **Arguments**
#:
#:    - ``subcategory`` *(required)*:
#:      Second-level grouping to organize the test (e.g., `Math`, `Tensor`, `Backend`).
#:
#:    - ``name`` *(required)*:
#:      Unique name for the test within the `Unit.<subcategory>` namespace.
#:
#:    - Additional keyword arguments are forwarded to
#:      ``einsums_add_test_and_deps_compile_test``, such as:
#:
#:      - ``SOURCES``: Source files for the compile test.
#:      - ``SOURCE_ROOT``: Source root directory for Einsums.
#:      - ``DEPENDENCIES``: Extra libraries or targets to link against.
#:      - ``FAILURE_EXPECTED``: Whether the test is expected to fail.
#:      - ``OBJECT`` or ``NOLIBS``: Flags to control how the test is compiled.
#:
#:    **Behavior**
#:
#:    - Constructs a full test name: ``Tests.Unit.<subcategory>.<name>``.
#:
#:    - Adds a CMake test that triggers compilation only (no execution).
#:
#:    - Creates a pseudo-target for the test and wires it into its category:
#:      - E.g., `Tests.Unit.Tensor` → `Tests.Unit.Tensor.my_test`
#:
#:    **Example**
#:
#:    .. code-block:: cmake
#:
#:       einsums_add_unit_compile_test(
#:         Tensor
#:         contraction_validity
#:         SOURCES test/compile/unit/tensor_contraction.cpp
#:         DEPENDENCIES einsums_core
#:       )
#:
#:       einsums_add_unit_compile_test(
#:         Math
#:         constexpr_overflow_check
#:         SOURCES test/compile/unit/constexpr_guard.cpp
#:         FAILURE_EXPECTED
#:       )
#:
#:    These define:
#:    - `Tests.Unit.Tensor.contraction_validity`
#:    - `Tests.Unit.Math.constexpr_overflow_check`
#:
#:    **See also**
#:
#:    - ``einsums_add_test_and_deps_compile_test``
#:    - ``einsums_add_compile_test``
#:    - ``einsums_add_compile_test_target_dependencies``
#:    - ``add_test``
function(einsums_add_unit_compile_test subcategory name)
  einsums_add_test_and_deps_compile_test("Unit" "${subcategory}" ${name} ${ARGN})
endfunction(einsums_add_unit_compile_test)

#:
#: einsums_add_regression_compile_test
#:
#:    Register a compile-only regression test under the `Regressions` category.
#:
#:    This function is a convenience wrapper around
#:    ``einsums_add_test_and_deps_compile_test``, hardcoding the category
#:    to `Regressions` to help organize compile-time regression tests.
#:
#:    These tests are typically used to ensure that previously reported issues do not
#:    reappear — for example, checking that invalid code no longer compiles, or that
#:    formerly broken headers now compile cleanly.
#:
#:    **Signature**
#:    ``einsums_add_regression_compile_test(<subcategory> <name> [OPTIONS...])``
#:
#:    **Arguments**
#:
#:    - ``subcategory`` *(required)*:
#:      Secondary grouping for the regression test (e.g., `CUDA`, `GCC13`, `MSVC`, `HeaderGuards`).
#:
#:    - ``name`` *(required)*:
#:      Name of the specific regression test case.
#:
#:    - All additional keyword arguments are forwarded to
#:      ``einsums_add_test_and_deps_compile_test``, including:
#:
#:      - ``SOURCES``: Required source files.
#:      - ``SOURCE_ROOT``: Einsums project source root.
#:      - ``DEPENDENCIES``: Dependencies required to compile.
#:      - ``FAILURE_EXPECTED``: Mark test as expected to fail (e.g., to check for bad API usage).
#:      - ``OBJECT`` or ``NOLIBS``: Compilation mode flags.
#:
#:    **Behavior**
#:
#:    - Registers a compile-only test with the name: ``Tests.Regressions.<subcategory>.<name>``
#:    - Adds a CMake test that compiles the source (but does not run it).
#:    - Adds a pseudo-target: `Tests.Regressions.<subcategory>` for grouped builds.
#:
#:    **Example**
#:
#:    .. code-block:: cmake
#:
#:       einsums_add_regression_compile_test(
#:         CUDA
#:         missing_kernels_guard
#:         SOURCES test/compile/regressions/cuda_missing_guard.cu
#:         FAILURE_EXPECTED
#:       )
#:
#:       einsums_add_regression_compile_test(
#:         HeaderGuards
#:         double_include_fix
#:         SOURCES test/compile/regressions/fixed_double_include.cpp
#:       )
#:
#:    These produce:
#:    - `Tests.Regressions.CUDA.missing_kernels_guard`
#:    - `Tests.Regressions.HeaderGuards.double_include_fix`
#:
#:    **See also**
#:
#:    - ``einsums_add_test_and_deps_compile_test``
#:    - ``einsums_add_compile_test``
#:    - ``einsums_add_compile_test_target_dependencies``
#:    - ``add_test``
function(einsums_add_regression_compile_test subcategory name)
  einsums_add_test_and_deps_compile_test("Regressions" "${subcategory}" ${name} ${ARGN})
endfunction(einsums_add_regression_compile_test)

#:
#: einsums_add_headers_compile_test
#:
#:    Register a compile-only test for validating header inclusion.
#:
#:    This function defines a test that checks whether one or more header files
#:    can be compiled in isolation — i.e., without relying on transitive includes
#:    or implicit ordering. It is a wrapper around
#:    ``einsums_add_test_and_deps_compile_test`` with the category
#:    fixed to `Headers` and `OBJECT` mode enabled.
#:
#:    **Signature**
#:    ``einsums_add_headers_compile_test(<subcategory> <name> [OPTIONS...])``
#:
#:    **Arguments**
#:
#:    - ``subcategory`` *(required — pass "" if none)*:
#:      Logical grouping (e.g., `Core`, `IO`, `CUDA`). This is used to structure the test name
#:      as `Tests.Headers.<subcategory>.<name>`. Pass `""` to omit the subcategory.
#:
#:    - ``name`` *(required)*:
#:      The name of the test case within the category.
#:
#:    - Additional keyword arguments are forwarded to
#:      ``einsums_add_test_and_deps_compile_test``, including:
#:
#:      - ``SOURCES``: The source file that includes the header(s) under test.
#:      - ``SOURCE_ROOT``: Einsums source root directory.
#:      - ``DEPENDENCIES``: Any required interface targets or libraries.
#:      - ``FAILURE_EXPECTED``: If set, marks the test as passing only on failure.
#:      - ``FOLDER``: Optional grouping in IDEs.
#:
#:    **Behavior**
#:
#:    - Creates a test named: `Tests.Headers[.<subcategory>].<name>`
#:    - Compiles the given source in `OBJECT` mode using ``einsums_add_library``
#:      (ensures linking is skipped — this is for compile-only validation).
#:    - Adds a test via ``add_test`` that builds the target.
#:    - Adds the test to a pseudo-target via ``einsums_add_compile_test_target_dependencies``.
#:
#:    **Note**
#:
#:    The use of `OBJECT` ensures the test validates compilation only, without requiring
#:    a full `main()` or linking context.
#:
#:    **Example**
#:
#:    .. code-block:: cmake
#:
#:       einsums_add_headers_compile_test(
#:         Core
#:         einsums_tensor
#:         SOURCES test/compile/headers/einsums_tensor.cpp
#:       )
#:
#:       einsums_add_headers_compile_test(
#:         ""
#:         top_level_includes
#:         SOURCES test/compile/headers/all_headers.cpp
#:       )
#:
#:    These define:
#:    - `Tests.Headers.Core.einsums_tensor`
#:    - `Tests.Headers.top_level_includes`
#:
#:    **See also**
#:
#:    - ``einsums_add_test_and_deps_compile_test``
#:    - ``einsums_add_compile_test``
#:    - ``einsums_add_compile_test_target_dependencies``
#:    - ``einsums_add_library``
#:    - ``add_test``
function(einsums_add_headers_compile_test subcategory name)
  # Important to keep the double quotes around subcategory otherwise it doesn't consider empty
  # argument but just removes it
  einsums_add_test_and_deps_compile_test("Headers" "${subcategory}" ${name} ${ARGN} OBJECT)
endfunction(einsums_add_headers_compile_test)

#:
#: einsums_add_header_tests
#:
#:    Generate compile‑only tests for a set of header files (plus an aggregate “all headers” test).
#:
#:    This helper scans a provided list of headers, auto‑creates tiny translation units that
#:    include each header, and registers per‑header compile tests under the
#:    ``Tests.Headers.<category>`` namespace. It also creates a single aggregate test that
#:    includes all non‑excluded headers together to catch ODR / include‑ordering issues.
#:
#:    **Signature**
#:    ``einsums_add_header_tests(<category> [HEADER_ROOT <dir>] [HEADERS <h...>] [EXCLUDE <h...>] [EXCLUDE_FROM_ALL <h...>] [DEPENDENCIES <t...>] [NOLIBS])``
#:
#:    **Positional Arguments**
#:    - ``category`` *(required)*:
#:      Logical group name for the header tests (e.g., ``Core``, ``IO``, ``CUDA``). Used to form
#:      test and pseudo‑target names like ``Tests.Headers.<category>``.
#:
#:    **Keyword Arguments**
#:    - ``HEADER_ROOT`` *(optional, one value)*:
#:      Root directory used to compute a project‑relative include like ``#include <rel/path.hpp>``.
#:
#:    - ``HEADERS`` *(required, multi‑value)*:
#:      The list of header files to validate.
#:
#:    - ``EXCLUDE`` *(optional, multi‑value)*:
#:      Specific headers to skip entirely (no individual test generated).
#:
#:    - ``EXCLUDE_FROM_ALL`` *(optional, multi‑value)*:
#:      Headers that should be **excluded only from the aggregate** “AllHeaders” test, while still
#:      receiving individual per‑header tests.
#:
#:    - ``DEPENDENCIES`` *(optional, multi‑value)*:
#:      Extra targets to depend on for compilation (interfaces, flags, etc.).
#:
#:    - ``NOLIBS`` *(optional flag)*:
#:      Forwarded to lower‑level helpers to avoid linking Einsums libraries (compile‑only).
#:
#:    **Behavior**
#:    1. Creates a category pseudo‑target: ``add_custom_target(Tests.Headers.<category>)``.
#:    2. Iterates over ``HEADERS`` and **skips** any that:
#:       - Are listed in ``EXCLUDE``, or
#:       - Have a path segment containing the literal substrings ``detail1`` or ``Detail1``.
#:    3. For each remaining header:
#:       - Computes a relative path from ``HEADER_ROOT`` and writes a generated TU
#:         (``.cpp``) in the binary dir that does: ``#include <rel/path.hpp>`` and a guardable
#:         ``main()``.
#:       - Registers an individual compile‑only test via
#:         ``einsums_add_headers_compile_test`` with:
#:           - Category ``"<category>"`` and a test name derived from the header path
#:             (slashes → underscores, ``.hpp`` → ``_hpp``).
#:           - ``SOURCE_ROOT`` set to the header’s directory (for include resolution).
#:           - ``DEPENDENCIES`` including any provided targets and the global
#:             ``einsums_private_flags`` / ``einsums_public_flags`` interfaces.
#:           - Propagated ``NOLIBS`` when set.
#:       - If the header is **not** listed in ``EXCLUDE_FROM_ALL``, its include line is appended
#:         to the aggregate “all headers” source buffer.
#:    4. Writes the aggregate TU ``AllHeaders.cpp`` (concatenation of selected ``#include`` lines,
#:       with a guardable ``main()``) and registers an aggregate compile‑only test
#:       ``Tests.Headers.<category>.AllHeaders`` via
#:       ``einsums_add_headers_compile_test``.
#:
#:    **Notes**
#:    - Tests are compile‑only (OBJECT mode) and wired into the appropriate pseudo‑targets so you
#:      can build by group (e.g., ``cmake --build . --target Tests.Headers.<category>``).
#:    - The “detail” filter uses literal substrings ``detail1``/``Detail1`` as implemented.
#:
#:    **Example**
#:    .. code-block:: cmake
#:
#:       file(GLOB EINSUMS_CORE_HEADERS
#:            "${CMAKE_SOURCE_DIR}/include/einsums/core/*.hpp")
#:
#:       einsums_add_header_tests(Core
#:         HEADER_ROOT "${CMAKE_SOURCE_DIR}/include"
#:         HEADERS ${EINSUMS_CORE_HEADERS}
#:         EXCLUDE "${CMAKE_SOURCE_DIR}/include/einsums/core/internal_only.hpp"
#:         EXCLUDE_FROM_ALL "${CMAKE_SOURCE_DIR}/include/einsums/core/heavy_header.hpp"
#:         DEPENDENCIES einsums_public_flags einsums_private_flags
#:         NOLIBS
#:       )
#:
#:    This generates:
#:    - Per‑header compile tests like ``Tests.Headers.Core.einsums_core_matrix_hpp``
#:    - An aggregate test ``Tests.Headers.Core.AllHeaders`` that includes all non‑excluded headers
#:      together.
#:
#:    **See also**
#:    - ``einsums_add_headers_compile_test``
#:    - ``einsums_add_test_and_deps_compile_test``
#:    - ``add_custom_target``
function(einsums_add_header_tests category)
  set(options NOLIBS)
  set(one_value_args HEADER_ROOT)
  set(multi_value_args HEADERS EXCLUDE EXCLUDE_FROM_ALL DEPENDENCIES)

  cmake_parse_arguments(${category} "${options}" "${one_value_args}" "${multi_value_args}" ${ARGN})

  set(_additional_flags)
  if(${category}_NOLIBS)
    set(_additional_flags ${_additional_flags} NOLIBS)
  endif()

  set(all_headers)
  add_custom_target(Tests.Headers.${category})

  foreach(header ${${category}_HEADERS})

    # skip all headers in directories containing 'detail'
    string(FIND "${header}" "detail1" detail_pos)
    list(FIND ${category}_EXCLUDE "${header}" exclude_pos)

    if(${detail_pos} EQUAL -1)
      string(FIND "${header}" "Detail1" detail_pos)
    endif()

    if(${detail_pos} EQUAL -1 AND ${exclude_pos} EQUAL -1)
      # extract relative path of header
      string(REGEX REPLACE "${${category}_HEADER_ROOT}/" "" relpath "${header}")

      # .hpp --> .cpp
      string(REGEX REPLACE ".hpp" ".cpp" full_test_file "${relpath}")
      # remove extension, '/' --> '_'
      string(REGEX REPLACE ".hpp" "_hpp" test_file "${relpath}")
      string(REGEX REPLACE "/" "_" test_name "${test_file}")

      # generate the test
      file(WRITE ${CMAKE_CURRENT_BINARY_DIR}/${full_test_file}
           "#include <${relpath}>\n" "#ifndef EINSUMS_MAIN_DEFINED\n" "int main() { return 0; }\n"
           "#endif\n"
      )

      set(exclude_all_pos -1)
      list(FIND ${category}_EXCLUDE_FROM_ALL "${header}" exclude_all_pos)
      if(${exclude_all_pos} EQUAL -1)
        set(all_headers ${all_headers} "#include <${relpath}>\n")
      endif()

      get_filename_component(header_dir "${relpath}" DIRECTORY)

      einsums_add_headers_compile_test(
        "${category}" ${test_name}
        SOURCES "${CMAKE_CURRENT_BINARY_DIR}/${full_test_file}"
        SOURCE_ROOT "${CMAKE_CURRENT_BINARY_DIR}/${header_dir}"
        FOLDER "Tests/Headers/${header_dir}"
        DEPENDENCIES ${${category}_DEPENDENCIES} einsums_private_flags einsums_public_flags
                     ${_additional_flags}
      )

    endif()
  endforeach()

  set(test_name "AllHeaders")
  set(all_headers_test_file "${CMAKE_CURRENT_BINARY_DIR}/${test_name}.cpp")
  file(WRITE ${all_headers_test_file} ${all_headers} "#ifndef EINSUMS_MAIN_DEFINED\n"
                                      "int main() { return 0; }\n" "#endif\n"
  )

  einsums_add_headers_compile_test(
    "${category}" ${test_name}
    SOURCES "${all_headers_test_file}"
    SOURCE_ROOT "${CMAKE_CURRENT_BINARY_DIR}"
    FOLDER "Tests/Headers"
    DEPENDENCIES ${${category}_DEPENDENCIES} ${_additional_flags}
  )
endfunction()
