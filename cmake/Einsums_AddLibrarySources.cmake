#----------------------------------------------------------------------------------------------
# Copyright (c) The Einsums Developers. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
#----------------------------------------------------------------------------------------------

#:
#: einsums_add_library_sources
#:
#:    Collect source files for a library target using CMake globs.
#:
#:    Populates the cache variable ``<name>_SOURCES`` with **absolute paths** discovered by
#:    ``file`` globs, optionally filtering with a regex and/or appending to an
#:    existing list. This list is typically consumed by ``einsums_add_library``.
#:
#:    **Signature**
#:    ``einsums_add_library_sources(<name> <globtype> [APPEND] [EXCLUDE <regex>] [GLOBS <...>])``
#:
#:    **Positional Arguments**
#:    - ``name`` *(required)*: Logical library name. The function writes to the
#:      ``INTERNAL`` cache variable ``<name>_SOURCES``.
#:    - ``globtype`` *(required)*: Passed to ``file``; usually ``GLOB`` or
#:      ``GLOB_RECURSE``.
#:
#:    **Options**
#:    - ``APPEND``: If set, append to the existing ``<name>_SOURCES`` list instead of replacing it.
#:
#:    **Keywords**
#:    - ``EXCLUDE <regex>``: Skip files whose **absolute path** matches this regex.
#:    - ``GLOBS <...>``: One or more glob patterns forwarded to ``file(<globtype> ...)``.
#:
#:    **Behavior**
#:    1. Calls ``file(<globtype> sources <GLOBS...>)`` to gather matches.
#:    2. Resets ``<name>_SOURCES`` unless ``APPEND`` is provided.
#:    3. Converts each match to an absolute path, filters via ``EXCLUDE`` if given,
#:       and appends approved paths to ``<name>_SOURCES`` (cache ``INTERNAL``).
#:
#:    **Example**
#:    .. code-block:: cmake
#:
#:       einsums_add_library_sources(einsums_core GLOB_RECURSE
#:         GLOBS
#:           "${CMAKE_SOURCE_DIR}/src/einsums/*.cpp"
#:           "${CMAKE_SOURCE_DIR}/src/common/*.cc"
#:         EXCLUDE ".*/experimental/.*"
#:       )
#:
#:    **See also**
#:    - ``file``
#:    - ``einsums_add_library``
#:    - ``einsums_add_library_sources_noglob``
function(einsums_add_library_sources name globtype)
  set(options APPEND)
  set(one_value_args)
  set(multi_value_args EXCLUDE GLOBS)
  cmake_parse_arguments(SOURCES "${options}" "${one_value_args}" "${multi_value_args}" ${ARGN})

  file(${globtype} sources ${SOURCES_GLOBS})

  if(NOT SOURCES_APPEND)
    set(${name}_SOURCES
        ""
        CACHE INTERNAL "Sources for lib${name}." FORCE
    )
  endif()

  foreach(source ${sources})
    get_filename_component(absolute_path ${source} ABSOLUTE)

    set(add_flag ON)

    if(SOURCES_EXCLUDE)
      if(${absolute_path} MATCHES ${SOURCES_EXCLUDE})
        set(add_flag OFF)
      endif()
    endif()

    if(add_flag)
      einsums_debug(
        "add_library_sources.${name}" "Adding ${absolute_path} to source list for lib${name}"
      )
      set(${name}_SOURCES
          ${${name}_SOURCES} ${absolute_path}
          CACHE INTERNAL "Sources for lib${name}." FORCE
      )
    endif()
  endforeach()
endfunction()

#:
#: einsums_add_library_sources_noglob
#:
#:    Collect source files for a library target from an explicit list (no globs).
#:
#:    Non‑glob counterpart to ``einsums_add_library_sources``. Populates the
#:    ``INTERNAL`` cache variable ``<name>_SOURCES`` with **absolute paths** from a provided list,
#:    with optional regex filtering and append semantics.
#:
#:    **Signature**
#:    ``einsums_add_library_sources_noglob(<name> [APPEND] [EXCLUDE <regex>] [SOURCES <...>])``
#:
#:    **Positional Arguments**
#:    - ``name`` *(required)*: Logical library name. Results are written to ``<name>_SOURCES``.
#:
#:    **Options**
#:    - ``APPEND``: If set, append to the existing ``<name>_SOURCES`` list instead of replacing it.
#:
#:    **Keywords**
#:    - ``EXCLUDE <regex>``: Skip files whose **absolute path** matches the regex.
#:    - ``SOURCES <...>``: Explicit list of source files to include.
#:
#:    **Behavior**
#:    1. Resets ``<name>_SOURCES`` unless ``APPEND`` is given.
#:    2. Converts each listed source to an absolute path, filters with ``EXCLUDE`` if provided,
#:       and appends surviving entries to ``<name>_SOURCES`` (cache ``INTERNAL``).
#:
#:    **Example**
#:    .. code-block:: cmake
#:
#:       einsums_add_library_sources_noglob(einsums_utils
#:         SOURCES
#:           "${CMAKE_SOURCE_DIR}/src/utils/fs.cpp"
#:           "${CMAKE_SOURCE_DIR}/src/utils/log.cc"
#:         EXCLUDE ".*/legacy/.*"
#:       )
#:
#:    **See also**
#:    - ``einsums_add_library_sources``
#:    - ``einsums_add_library``
function(einsums_add_library_sources_noglob name)
  set(options APPEND)
  set(one_value_args)
  set(multi_value_args EXCLUDE SOURCES)
  cmake_parse_arguments(SOURCES "${options}" "${one_value_args}" "${multi_value_args}" ${ARGN})

  # einsums_print_list("DEBUG" "einsums_add_library_sources_noglob.${name}" "Sources for ${name}"
  # ${SOURCES_SOURCES})

  set(sources ${SOURCES_SOURCES})

  if(NOT SOURCES_APPEND)
    set(${name}_SOURCES
        ""
        CACHE INTERNAL "Sources for lib${name}." FORCE
    )
  endif()

  foreach(source ${sources})
    get_filename_component(absolute_path ${source} ABSOLUTE)

    set(add_flag ON)

    if(SOURCES_EXCLUDE)
      if(${absolute_path} MATCHES ${SOURCES_EXCLUDE})
        set(add_flag OFF)
      endif()
    endif()

    if(add_flag)
      einsums_debug(
        "add_library_sources.${name}" "Adding ${absolute_path} to source list for lib${name}"
      )
      set(${name}_SOURCES
          ${${name}_SOURCES} ${absolute_path}
          CACHE INTERNAL "Sources for lib${name}." FORCE
      )
    endif()
  endforeach()
endfunction()
