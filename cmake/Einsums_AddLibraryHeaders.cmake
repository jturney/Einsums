#----------------------------------------------------------------------------------------------
# Copyright (c) The Einsums Developers. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
#----------------------------------------------------------------------------------------------

#:
#: einsums_add_library_headers
#:
#:    Collect header files for a library target (IDE visibility on MSVC), using CMake globs.
#:
#:    On **MSVC generators only**, this helper populates the cache variable
#:    ``<name>_HEADERS`` with absolute paths to headers discovered by one or more glob patterns.
#:    This list is later attached to the library target for IDE browsing/filtering.
#:
#:    **Signature**
#:    ``einsums_add_library_headers(<name> <globtype> [APPEND] [EXCLUDE <regex>] [GLOBS <...>])``
#:
#:    **Positional Arguments**
#:    - ``name`` *(required)*: Logical target name whose header cache variable
#:      ``<name>_HEADERS`` will be filled.
#:    - ``globtype`` *(required)*: Passed to ``file`` (e.g., ``GLOB`` or
#:      ``GLOB_RECURSE``).
#:
#:    **Options**
#:    - ``APPEND``: Append to the existing ``<name>_HEADERS`` list. If **omitted**, the
#:      list is reset before adding.
#:
#:    **Keywords**
#:    - ``EXCLUDE <regex>``: Exclude headers whose **absolute path** matches the regex.
#:    - ``GLOBS <...>``: One or more glob patterns passed to ``file(<globtype> ...)``.
#:
#:    **Behavior**
#:    1. On MSVC, runs ``file(<globtype> headers <GLOBS...>)`` to discover headers.
#:    2. For each header, computes an absolute path and filters it against ``EXCLUDE`` (if any).
#:    3. Populates the cache variable ``<name>_HEADERS`` (type: ``INTERNAL``) with the results
#:       (reset unless ``APPEND`` is given).
#:    4. No‑op on non‑MSVC generators (function returns without changes).
#:
#:    **Example**
#:    .. code-block:: cmake
#:
#:       einsums_add_library_headers(einsums_core GLOB_RECURSE
#:         GLOBS "${CMAKE_SOURCE_DIR}/include/einsums/*.hpp"
#:         EXCLUDE ".*/detail/.*"
#:       )
#:
#:    After this call, ``einsums_core_HEADERS`` contains filtered absolute header paths
#:    (MSVC only), suitable for attaching to the target for IDE display.
#:
#:    **See also**
#:    - ``file``
#:    - ``einsums_add_library_headers_noglob``
function(einsums_add_library_headers name globtype)
  if(MSVC)
    set(options APPEND)
    set(one_value_args)
    set(multi_value_args EXCLUDE GLOBS)
    cmake_parse_arguments(HEADERS "${options}" "${one_value_args}" "${multi_value_args}" ${ARGN})

    if(NOT HEADERS_APPEND)
      set(${name}_HEADERS
          ""
          CACHE INTERNAL "Headers for lib${name}." FORCE
      )
    endif()

    file(${globtype} headers ${HEADERS_GLOBS})

    foreach(header ${headers})
      get_filename_component(absolute_path ${header} ABSOLUTE)

      set(add_flag ON)

      if(HEADERS_EXCLUDE)
        if(${absolute_path} MATCHES ${HEADERS_EXCLUDE})
          set(add_flag OFF)
        endif()
      endif()

      if(add_flag)
        einsums_debug(
          "add_library_headers.${name}" "Adding ${absolute_path} to header list for lib${name}"
        )
        set(${name}_HEADERS
            ${${name}_HEADERS} ${absolute_path}
            CACHE INTERNAL "Headers for lib${name}." FORCE
        )
      endif()
    endforeach()
  endif()
endfunction()

#:
#: einsums_add_library_headers_noglob
#:
#:    Collect header files for a library target (IDE visibility on MSVC), from an explicit list.
#:
#:    This is the non‑glob counterpart to ``einsums_add_library_headers``. On **MSVC
#:    generators only**, it filters an explicit list of headers and stores absolute paths in the
#:    ``<name>_HEADERS`` cache variable for later attachment to the library target.
#:
#:    **Signature**
#:    ``einsums_add_library_headers_noglob(<name> [APPEND] [EXCLUDE <regex>] [HEADERS <...>])``
#:
#:    **Positional Arguments**
#:    - ``name`` *(required)*: Logical target name whose header cache variable
#:      ``<name>_HEADERS`` will be filled.
#:
#:    **Options**
#:    - ``APPEND``: Append to the existing ``<name>_HEADERS`` list. If **omitted**, the
#:      list is reset before adding.
#:
#:    **Keywords**
#:    - ``EXCLUDE <regex>``: Exclude headers whose **absolute path** matches the regex.
#:    - ``HEADERS <...>``: Explicit list of header file paths.
#:
#:    **Behavior**
#:    1. On MSVC, iterates over ``HEADERS``, resolves each to an absolute path, filters using
#:       ``EXCLUDE`` (if provided), and writes the result into the ``INTERNAL`` cache variable
#:       ``<name>_HEADERS`` (reset unless ``APPEND`` is given).
#:    2. No‑op on non‑MSVC generators.
#:
#:    **Example**
#:    .. code-block:: cmake
#:
#:       einsums_add_library_headers_noglob(einsums_utils
#:         HEADERS
#:           "${CMAKE_SOURCE_DIR}/include/einsums/utils/log.hpp"
#:           "${CMAKE_SOURCE_DIR}/include/einsums/utils/fs.hpp"
#:         EXCLUDE ".*/experimental/.*"
#:       )
#:
#:    **See also**
#:    - ``einsums_add_library_headers``
function(einsums_add_library_headers_noglob name)
  if(MSVC)
    set(options APPEND)
    set(one_value_args)
    set(multi_value_args EXCLUDE HEADERS)
    cmake_parse_arguments(HEADERS "${options}" "${one_value_args}" "${multi_value_args}" ${ARGN})

    einsums_print_list(
      "DEBUG" "einsums_add_library_sources_noglob.${name}" "Sources for ${name}" HEADERS_HEADERS
    )

    set(headers ${HEADERS_HEADERS})

    if(NOT HEADERS_APPEND)
      set(${name}_HEADERS
          ""
          CACHE INTERNAL "Headers for lib${name}." FORCE
      )
    endif()

    foreach(header ${headers})
      get_filename_component(absolute_path ${header} ABSOLUTE)

      set(add_flag ON)

      if(HEADERS_EXCLUDE)
        if(${absolute_path} MATCHES ${HEADERS_EXCLUDE})
          set(add_flag OFF)
        endif()
      endif()

      if(add_flag)
        einsums_debug(
          "einsums_add_library_headers_noglob.${name}"
          "Adding ${absolute_path} to header list for lib${name}"
        )
        set(${name}_HEADERS
            ${${name}_HEADERS} ${absolute_path}
            CACHE INTERNAL "Headers for lib${name}." FORCE
        )
      endif()
    endforeach()
  endif()
endfunction()
