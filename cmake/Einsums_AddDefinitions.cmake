#----------------------------------------------------------------------------------------------
# Copyright (c) The Einsums Developers. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
#----------------------------------------------------------------------------------------------

#:
#: einsums_add_config_define
#:
#:    Append a compile-time preprocessor definition to the global Einsums config property.
#:
#:    This function registers a preprocessor definition (e.g., `EINSUMS_HAS_FOO`) into the
#:    global property `EINSUMS_CONFIG_DEFINITIONS`. These are typically emitted later into a
#:    generated `einsums/config.hpp` or passed to public interface targets.
#:
#:    **Signature**
#:    ``einsums_add_config_define(<definition> [<value>])``
#:
#:    **Arguments**
#:
#:    - ``definition`` *(required)*:
#:      The name of the macro to define (e.g., ``EINSUMS_HAVE_STD_BIT_CAST``).
#:
#:    - ``value`` *(optional)*:
#:      An optional value to assign to the macro, such as `1`, `"string"`, or another symbol.
#:      If present, the macro will be defined as:
#:
#:      .. code-block:: c
#:
#:         #define EINSUMS_HAVE_STD_BIT_CAST 1
#:
#:      Otherwise, it will be treated as a presence-only macro:
#:
#:      .. code-block:: c
#:
#:         #define EINSUMS_HAVE_STD_BIT_CAST
#:
#:    **Behavior**
#:
#:    - Appends the formatted macro (with or without value) to the CMake global property
#:      ``EINSUMS_CONFIG_DEFINITIONS``.
#:
#:    - This property is used during config header or compile option generation.
#:
#:    - Takes care to preserve a value of `0` (unlike `if(ARGN)` which would treat it as false).
#:
#:    **Example**
#:
#:    .. code-block:: cmake
#:
#:       einsums_add_config_define(EINSUMS_HAS_STD_BIT_CAST 1)
#:
#:       einsums_add_config_define(EINSUMS_USE_CUDA)
#:
#:    These result in the following entries in the config macro list:
#:
#:    - ``EINSUMS_HAS_STD_BIT_CAST 1``
#:    - ``EINSUMS_USE_CUDA``
#:
#:    **See also**
#:
#:    - ``einsums_add_config_test``
#:    - ``set_property(GLOBAL APPEND PROPERTY ...)``
#:    - Global property: ``EINSUMS_CONFIG_DEFINITIONS``
function(einsums_add_config_define definition)
  # if(ARGN) ignores an argument "0"
  set(Args ${ARGN})
  list(LENGTH Args ArgsLen)
  if(ArgsLen GREATER 0)
    set_property(GLOBAL APPEND PROPERTY EINSUMS_CONFIG_DEFINITIONS "${definition} ${ARGN}")
  else()
    set_property(GLOBAL APPEND PROPERTY EINSUMS_CONFIG_DEFINITIONS "${definition}")
  endif()
endfunction()

#:
#: einsums_add_config_cond_define
#:
#:    Append a **conditional** preprocessor definition to the Einsums configuration property list.
#:
#:    This function works similarly to ``einsums_add_config_define``, but stores
#:    the result in the global property `EINSUMS_CONFIG_COND_DEFINITIONS`, which is intended for
#:    **conditionally emitted** macros — e.g., those gated by platform, compiler, GPU support,
#:    or user build options.
#:
#:    **Signature**
#:    ``einsums_add_config_cond_define(<definition> [<value>])``
#:
#:    **Arguments**
#:
#:    - ``definition`` *(required)*:
#:      The name of the macro to define (e.g., ``EINSUMS_USE_CUDA_BACKEND``).
#:      If the string begins with `-D`, this prefix will be stripped and a warning will be issued.
#:
#:    - ``value`` *(optional)*:
#:      Value assigned to the macro (e.g., `1`, `0`, `"some_string"`).
#:      If omitted, the macro is treated as a presence-only define.
#:
#:    **Behavior**
#:
#:    - If ``definition`` starts with ``-D``, a warning is printed and the prefix is removed.
#:      This helps users avoid passing full compiler-style definitions like ``-DFOO=1``.
#:
#:    - The macro (with or without a value) is appended to the global property
#:      ``EINSUMS_CONFIG_COND_DEFINITIONS``.
#:
#:    - This property is later consumed when generating conditional config headers or selectively
#:      passing compile definitions to interface targets.
#:
#:    - Values like `0` are preserved (guarding against CMake's default behavior of ignoring falsy strings).
#:
#:    **Example**
#:
#:    .. code-block:: cmake
#:
#:       einsums_add_config_cond_define(EINSUMS_USE_CUDA_BACKEND 1)
#:
#:       einsums_add_config_cond_define(EINSUMS_NO_ASSERTIONS)
#:
#:       einsums_add_config_cond_define(-DOLD_STYLE_DEFINE 0)
#:
#:    The third example logs a warning and registers the macro as ``OLD_STYLE_DEFINE 0``.
#:
#:    **See also**
#:
#:    - ``einsums_add_config_define``
#:    - Global properties:
#:      - ``EINSUMS_CONFIG_DEFINITIONS``
#:      - ``EINSUMS_CONFIG_COND_DEFINITIONS``
#:    - ``set_property(GLOBAL APPEND PROPERTY ...)``
function(einsums_add_config_cond_define definition)

  # Check if the definition starts with a -D if so, post a warning and remove it.
  string(FIND ${definition} "-D" _pos)
  if(NOT ${_pos} EQUAL -1 AND ${_pos} EQUAL 0)
    message(
      WARNING "einsums_add_config_cond_define: definition should not start with -D, removing it."
    )
    string(SUBSTRING ${definition} 2 -1 definition)
  endif()

  # if(ARGN) ignores an argument "0"
  set(Args ${ARGN})
  list(LENGTH Args ArgsLen)
  if(ArgsLen GREATER 0)
    set_property(GLOBAL APPEND PROPERTY EINSUMS_CONFIG_COND_DEFINITIONS "${definition} ${ARGN}")
  else()
    set_property(GLOBAL APPEND PROPERTY EINSUMS_CONFIG_COND_DEFINITIONS "${definition}")
  endif()

endfunction()

#:
#: einsums_add_config_define_namespace
#:
#:    Register a compile-time macro under a **named definition namespace**.
#:
#:    This function allows logically grouping configuration defines into named namespaces
#:    (e.g., for use in separate header files or modular components) by appending them to a
#:    custom global property: ``EINSUMS_CONFIG_DEFINITIONS_<NAMESPACE>``.
#:
#:    **Signature**
#:    ``einsums_add_config_define_namespace(DEFINE <macro> [VALUE <val>] NAMESPACE <ns>)``
#:
#:    **Arguments**
#:
#:    - ``DEFINE`` *(required)*:
#:      The name of the macro to define (e.g., ``EINSUMS_ENABLE_ASSERTS``).
#:
#:    - ``VALUE`` *(optional, multi‑value)*:
#:      Optional value(s) to assign to the macro. If provided, the define will be written as:
#:      ``EINSUMS_ENABLE_ASSERTS 1`` or ``FOO "bar baz"``, etc.
#:
#:    - ``NAMESPACE`` *(required)*:
#:      A string used to identify the logical grouping of the define. This name determines
#:      the global CMake property that receives the define:
#:      ``EINSUMS_CONFIG_DEFINITIONS_<NAMESPACE>``.
#:
#:    **Behavior**
#:
#:    - Appends a formatted string (either ``DEFINE VALUE...`` or just ``DEFINE``) to the global
#:      property: ``EINSUMS_CONFIG_DEFINITIONS_<NAMESPACE>``.
#:
#:    - Ensures that trailing whitespace is avoided if no value is provided.
#:
#:    - These namespace-specific definitions can be used when generating custom config headers,
#:      modular build targets, or conditionally scoped compile flags.
#:
#:    **Example**
#:
#:    .. code-block:: cmake
#:
#:       einsums_add_config_define_namespace(
#:         DEFINE EINSUMS_ENABLE_ASSERTS
#:         VALUE 1
#:         NAMESPACE Debug
#:       )
#:
#:       einsums_add_config_define_namespace(
#:         DEFINE EINSUMS_DISABLE_WARNING_PRAGMA
#:         NAMESPACE Minimal
#:       )
#:
#:    These create:
#:    - ``EINSUMS_CONFIG_DEFINITIONS_Debug`` with value ``EINSUMS_ENABLE_ASSERTS 1``
#:    - ``EINSUMS_CONFIG_DEFINITIONS_Minimal`` with value ``EINSUMS_DISABLE_WARNING_PRAGMA``
#:
#:    **See also**
#:
#:    - ``einsums_add_config_define``
#:    - ``set_property(GLOBAL APPEND PROPERTY ...)``
#:    - Global properties:
#:      - ``EINSUMS_CONFIG_DEFINITIONS_<NAMESPACE>``
#:      - ``EINSUMS_CONFIG_DEFINITIONS``
function(einsums_add_config_define_namespace)
  set(options)
  set(one_value_args DEFINE NAMESPACE)
  set(multi_value_args VALUE)
  cmake_parse_arguments(OPTION "${options}" "${one_value_args}" "${multi_value_args}" ${ARGN})

  set(DEF_VAR EINSUMS_CONFIG_DEFINITIONS_${OPTION_NAMESPACE})

  # to avoid extra trailing spaces (no value), use an if check
  if(OPTION_VALUE)
    set_property(GLOBAL APPEND PROPERTY ${DEF_VAR} "${OPTION_DEFINE} ${OPTION_VALUE}")
  else()
    set_property(GLOBAL APPEND PROPERTY ${DEF_VAR} "${OPTION_DEFINE}")
  endif()

endfunction()

#:
#: einsums_add_config_cond_define_namespace
#:
#:    Register a **conditionally scoped** macro inside a named configuration namespace.
#:
#:    This function is the conditional counterpart to
#:    ``einsums_add_config_define_namespace``, storing the definition in the
#:    global property `EINSUMS_CONFIG_COND_DEFINITIONS_<NAMESPACE>` instead of the always-defined one.
#:
#:    Use this for config macros that should be emitted only under certain conditions or build
#:    modes (e.g., platform-dependent or feature-dependent defines).
#:
#:    **Signature**
#:    ``einsums_add_config_cond_define_namespace(DEFINE <macro> [VALUE <val>] NAMESPACE <ns>)``
#:
#:    **Arguments**
#:
#:    - ``DEFINE`` *(required)*:
#:      The macro name to define (e.g., `EINSUMS_USE_CUDA_BACKEND`).
#:
#:    - ``VALUE`` *(optional, multi-value)*:
#:      Value(s) to assign to the macro. Supports things like `1`, `0`, `"some_string"`.
#:
#:      - If **no value** is given, the macro is treated as presence-only (e.g., `#define FOO`).
#:
#:      - Values such as `0` are explicitly accepted via a string comparison to avoid being ignored.
#:
#:    - ``NAMESPACE`` *(required)*:
#:      The logical group in which to store the define. This creates or appends to the global
#:      property `EINSUMS_CONFIG_COND_DEFINITIONS_<NAMESPACE>`.
#:
#:    **Behavior**
#:
#:    - Builds a property name from the given namespace:
#:      ``EINSUMS_CONFIG_COND_DEFINITIONS_<NAMESPACE>``.
#:
#:    - If a value is provided (including `"0"`), the macro is recorded as:
#:      ``DEFINE VALUE``. Otherwise, it is recorded as ``DEFINE``.
#:
#:    - Stores the define using ``set_property(GLOBAL APPEND PROPERTY ...)``.
#:
#:    **Use case**
#:
#:    These defines are typically later emitted into generated config headers or applied
#:    selectively to targets depending on build conditions.
#:
#:    **Examples**
#:
#:    .. code-block:: cmake
#:
#:       einsums_add_config_cond_define_namespace(
#:         DEFINE EINSUMS_USE_CUDA_BACKEND
#:         VALUE 1
#:         NAMESPACE GPU
#:       )
#:
#:       einsums_add_config_cond_define_namespace(
#:         DEFINE EINSUMS_AVOID_STD_SPAN
#:         VALUE 0
#:         NAMESPACE Legacy
#:
#:       einsums_add_config_cond_define_namespace(
#:         DEFINE EINSUMS_ENABLE_SANITIZERS
#:         NAMESPACE Debug
#:       )
#:
#:    These populate:
#:    - ``EINSUMS_CONFIG_COND_DEFINITIONS_GPU`` with ``EINSUMS_USE_CUDA_BACKEND 1``
#:    - ``EINSUMS_CONFIG_COND_DEFINITIONS_Legacy`` with ``EINSUMS_AVOID_STD_SPAN 0``
#:    - ``EINSUMS_CONFIG_COND_DEFINITIONS_Debug`` with ``EINSUMS_ENABLE_SANITIZERS``
#:
#:    **See also**
#:
#:    - ``einsums_add_config_define_namespace``
#:    - ``einsums_add_config_cond_define``
#:    - Global property: ``EINSUMS_CONFIG_COND_DEFINITIONS_<NAMESPACE>``
function(einsums_add_config_cond_define_namespace)
  set(one_value_args DEFINE NAMESPACE)
  set(multi_value_args VALUE)
  cmake_parse_arguments(OPTION "${options}" "${one_value_args}" "${multi_value_args}" ${ARGN})

  set(DEF_VAR EINSUMS_CONFIG_COND_DEFINITIONS_${OPTION_NAMESPACE})

  # to avoid extra trailing spaces (no value), use an if check
  if(OPTION_VALUE OR "${OPTION_VALUE}" STREQUAL "0")
    set_property(GLOBAL APPEND PROPERTY ${DEF_VAR} "${OPTION_DEFINE} ${OPTION_VALUE}")
  else()
    set_property(GLOBAL APPEND PROPERTY ${DEF_VAR} "${OPTION_DEFINE}")
  endif()

endfunction()

#:
#: einsums_write_config_defines_file
#:
#:    Emit a header (or templated file) containing configured preprocessor macros.
#:
#:    This function collects previously registered config defines (both unconditional and
#:    conditional) — from either the default/global namespace or a named namespace — then
#:    generates a file with the appropriate ``#define`` lines. Conditional entries are guarded
#:    against redefinition and, when named with version hints, wrapped in C++ standard checks.
#:
#:    **Signature**
#:    ``einsums_write_config_defines_file(NAMESPACE <ns|default> FILENAME <path> [TEMPLATE <in-file>])``
#:
#:    **Arguments**
#:
#:    - ``NAMESPACE`` *(required)*:
#:      Which definition namespace to use. Pass ``default`` to use the global properties
#:      (``EINSUMS_CONFIG_DEFINITIONS`` and ``EINSUMS_CONFIG_COND_DEFINITIONS``). Otherwise,
#:      the function reads:
#:      - ``EINSUMS_CONFIG_DEFINITIONS_<NAMESPACE>``
#:      - ``EINSUMS_CONFIG_COND_DEFINITIONS_<NAMESPACE>``
#:
#:    - ``FILENAME`` *(required)*:
#:      Output path to write. If ``TEMPLATE`` is **not** provided, a complete header is generated.
#:      If ``TEMPLATE`` **is** provided, that template is processed via ``configure_file``
#:      and written here.
#:
#:    - ``TEMPLATE`` *(optional)*:
#:      An input template file to be processed with ``@ONLY`` replacement. The variable
#:      ``@einsums_config_defines@`` is available for inclusion of the generated ``#define`` block.
#:
#:    **Input sources**
#:    - Unconditional defines are read from the selected ``EINSUMS_CONFIG_DEFINITIONS[...]`` property
#:      and emitted as:
#:      ``#define <NAME [VALUE]>``
#:
#:    - Conditional defines are read from ``EINSUMS_CONFIG_COND_DEFINITIONS[...]`` and emitted with
#:      include guards so they do not overwrite prior user/toolchain defines:
#:      ``#if !defined(<NAME>)\n#define <NAME [VALUE]>\n#endif``.
#:      Additionally, names containing the substrings ``HAVE_CXX20`` or ``HAVE_CXX17`` are wrapped
#:      in a C++ standard check:
#:      - ``HAVE_CXX20`` → ``#if __cplusplus >= 202002``
#:      - ``HAVE_CXX17`` → ``#if __cplusplus >= 201500``  (as implemented)
#:
#:    **Behavior**
#:    1. Load the appropriate global properties based on ``NAMESPACE``.
#:    2. Sort and de‑duplicate both define lists.
#:    3. Build a single string variable ``einsums_config_defines`` containing newline‑terminated
#:       ``#define`` lines and conditional guarded blocks.
#:    4. If ``TEMPLATE`` is **absent**:
#:       - Generate a self‑contained C++ header with a license banner, ``#pragma once``, and the
#:         contents of ``einsums_config_defines``; then copy it to ``FILENAME``.
#:    5. If ``TEMPLATE`` is **present**:
#:       - Call ``configure_file`` with ``@ONLY`` so that the template can reference
#:         ``@einsums_config_defines@`` (and other standard CMake variables) and write the result
#:         to ``FILENAME``.
#:
#:    **Template usage**
#:    In your template, insert:
#:
#:    .. code-block:: c
#:
#:       /* Generated config defines */
#:       @einsums_config_defines@
#:
#:    **Example — generate a default header**
#:    .. code-block:: cmake
#:
#:       einsums_write_config_defines_file(
#:         NAMESPACE default
#:         FILENAME "${CMAKE_BINARY_DIR}/include/einsums/config.hpp"
#:       )
#:
#:    **Example — use a custom template and namespace**
#:    .. code-block:: cmake
#:
#:       einsums_write_config_defines_file(
#:         NAMESPACE GPU
#:         TEMPLATE  "${CMAKE_SOURCE_DIR}/cmake/templates/config_defs.in.hpp"
#:         FILENAME  "${CMAKE_BINARY_DIR}/include/einsums/config_gpu.hpp"
#:       )
#:
#:    **See also**
#:    - ``einsums_add_config_define``
#:    - ``einsums_add_config_cond_define``
#:    - ``einsums_add_config_define_namespace``
#:    - ``einsums_add_config_cond_define_namespace``
#:    - Global properties:
#:      ``EINSUMS_CONFIG_DEFINITIONS*``, ``EINSUMS_CONFIG_COND_DEFINITIONS*``
function(einsums_write_config_defines_file)
  set(options)
  set(one_value_args TEMPLATE NAMESPACE FILENAME)
  set(multi_value_args)
  cmake_parse_arguments(OPTION "${options}" "${one_value_args}" "${multi_value_args}" ${ARGN})

  if(${OPTION_NAMESPACE} STREQUAL "default")
    get_property(DEFINITIONS_VAR GLOBAL PROPERTY EINSUMS_CONFIG_DEFINITIONS)
    get_property(COND_DEFINITIONS_VAR GLOBAL PROPERTY EINSUMS_CONFIG_COND_DEFINITIONS)
  else()
    get_property(DEFINITIONS_VAR GLOBAL PROPERTY EINSUMS_CONFIG_DEFINITIONS_${OPTION_NAMESPACE})
    get_property(
      COND_DEFINITIONS_VAR GLOBAL PROPERTY EINSUMS_CONFIG_COND_DEFINITIONS_${OPTION_NAMESPACE}
    )
  endif()

  if(DEFINED DEFINITIONS_VAR)
    list(SORT DEFINITIONS_VAR)
    list(REMOVE_DUPLICATES DEFINITIONS_VAR)
  endif()

  set(einsums_config_defines "\n")
  foreach(def ${DEFINITIONS_VAR})
    set(einsums_config_defines "${einsums_config_defines}#define ${def}\n")
  endforeach()

  if(DEFINED COND_DEFINITIONS_VAR)
    list(SORT COND_DEFINITIONS_VAR)
    list(REMOVE_DUPLICATES COND_DEFINITIONS_VAR)
    set(einsums_config_defines "${einsums_config_defines}\n")
  endif()
  foreach(def ${COND_DEFINITIONS_VAR})
    string(FIND ${def} " " _pos)
    if(NOT ${_pos} EQUAL -1)
      string(SUBSTRING ${def} 0 ${_pos} defname)
    else()
      set(defname ${def})
      string(STRIP ${defname} defname)
    endif()

    # C++20 specific variable
    string(FIND ${def} "HAVE_CXX20" _pos)
    if(NOT ${_pos} EQUAL -1)
      set(einsums_config_defines
          "${einsums_config_defines}#if __cplusplus >= 202002 && !defined(${defname})\n#define ${def}\n#endif\n"
      )
    else()
      # C++17 specific variable
      string(FIND ${def} "HAVE_CXX17" _pos)
      if(NOT ${_pos} EQUAL -1)
        set(einsums_config_defines
            "${einsums_config_defines}#if __cplusplus >= 201500 && !defined(${defname})\n#define ${def}\n#endif\n"
        )
      else()
        set(einsums_config_defines
            "${einsums_config_defines}#if !defined(${defname})\n#define ${def}\n#endif\n"
        )
      endif()
    endif()
  endforeach()

  # if the user has not specified a template, generate a proper header file
  if(NOT OPTION_TEMPLATE)
    # Uppercase namespace
    string(TOUPPER ${OPTION_NAMESPACE} NAMESPACE_UPPER)

    # Make PREAMBLE a single string using bracket syntax
    set(PREAMBLE [[
//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

// Do not edit this file! It has been generated by the cmake configuration step.

#pragma once
]])

    # Paths
    set(TEMP_FILENAME "${PROJECT_BINARY_DIR}${CMAKE_FILES_DIRECTORY}/${NAMESPACE_UPPER}.tmp")
    set(OUT_FILENAME "${OPTION_FILENAME}")

    # Build full content as a single string (avoid list expansion artifacts)
    set(_content "${PREAMBLE}\n${einsums_config_defines}\n")

    # Write candidate content to temp
    file(WRITE "${TEMP_FILENAME}" "${_content}")

    # Ensure output dir exists
    get_filename_component(_out_dir "${OUT_FILENAME}" DIRECTORY)
    if (_out_dir AND NOT IS_DIRECTORY "${_out_dir}")
      file(MAKE_DIRECTORY "${_out_dir}")
    endif ()

    # Compare & replace only if different (or if output doesn't exist)
    execute_process(
            COMMAND ${CMAKE_COMMAND} -E compare_files "${OUT_FILENAME}" "${TEMP_FILENAME}"
            RESULT_VARIABLE _cmp_res
            ERROR_QUIET
    )

    if (_cmp_res)
      file(COPY "${TEMP_FILENAME}" DESTINATION "${_out_dir}")
      file(RENAME "${_out_dir}/${NAMESPACE_UPPER}.tmp" "${OUT_FILENAME}")
      einsums_debug("Updated ${OUT_FILENAME} (content changed).")
    else ()
      einsums_debug("${OUT_FILENAME} unchanged (content identical).")
    endif ()

    # Clean up
    file(REMOVE "${TEMP_FILENAME}")
  else()
    configure_file("${OPTION_TEMPLATE}" "${OPTION_FILENAME}" @ONLY)
  endif()
endfunction()
