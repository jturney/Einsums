#----------------------------------------------------------------------------------------------
# Copyright (c) The Einsums Developers. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
#----------------------------------------------------------------------------------------------

include(CMakeDependentOption)
include(CMakeParseArguments)

set(EINSUMS_OPTION_CATEGORIES "Generic" "Build Targets" "Profiling" "Debugging")

#:
#: einsums_option
#:
#:    Declare a project option with optional dependency, category, advanced flag, and string choices.
#:
#:    A unified wrapper around ``option``, ``cmake_dependent_option``, and
#:    cache variables for non‑BOOL types. It also records the option in Einsums’ metadata and stores
#:    its category for later documentation/UX.
#:
#:    **Signature**
#:    ``einsums_option(<option> <type> <description> <default>
#:        [ADVANCED]
#:        [CATEGORY <name>]
#:        [DEPENDS <expr>]
#:        [STRINGS <v1;v2;...>]
#:    )``
#:
#:    **Positional arguments**
#:    - ``option`` *(required)*: Cache variable name (e.g., ``EINSUMS_WITH_CUDA``).
#:    - ``type`` *(required)*: One of CMake’s cache types (e.g., ``BOOL``, ``STRING``, ``PATH``, ``FILEPATH``).
#:    - ``description`` *(required)*: Help text shown in GUIs and cache tools.
#:    - ``default`` *(required)*: Default value for the cache entry.
#:
#:    **Options / keywords**
#:    - ``ADVANCED`` *(switch)*: Marks the cache entry as advanced.
#:    - ``CATEGORY <name>``: Logical grouping for this option (defaults to ``Generic``). Stored in
#:      the internal cache variable ``<option>Category``.
#:    - ``DEPENDS <expr>``: For ``BOOL`` options, creates a ``cmake_dependent_option``
#:      gated by the given CMake expression ; for non‑BOOL types this is **invalid** and triggers an error.
#:    - ``STRINGS <v1;v2;...>``: For ``STRING`` type only, sets the allowed values list shown in GUIs.
#:
#:    **Behavior**
#:    - If ``type`` is ``BOOL``:
#:      - Without ``DEPENDS`` → defines a normal ``option``.
#:      - With ``DEPENDS`` → defines a ``cmake_dependent_option`` with the provided guard.
#:    - If ``type`` is **not** ``BOOL``:
#:      - Fails if ``DEPENDS`` was provided.
#:      - Ensures a cache entry of the given ``type`` exists with ``description`` and ``default``.
#:        If a non‑cache variable with the same name exists, it is replaced by a cache entry.
#:      - If ``STRINGS`` is given and ``type`` is ``STRING``, applies the choices to the cache entry; otherwise errors.
#:    - Applies ``ADVANCED`` (if present) to the cache entry.
#:    - Appends the option name to the global property ``EINSUMS_MODULE_CONFIG_EINSUMS``.
#:    - Stores the option’s category in the internal cache variable ``<option>Category`` (default ``Generic`` or provided ``CATEGORY``).
#:
#:    **Examples**
#:    .. code-block:: cmake
#:
#:       # Simple boolean option
#:       einsums_option(EINSUMS_WITH_CUDA BOOL "Enable CUDA backends" OFF)
#:
#:       # Dependent boolean option (only visible when profiling is enabled)
#:       einsums_option(EINSUMS_WITH_PAPI BOOL "Enable PAPI profiling" OFF DEPENDS EINSUMS_WITH_PROFILING)
#:
#:       # String option with choices, marked advanced, categorized under "Build Targets"
#:       einsums_option(EINSUMS_BACKEND STRING "Select default backend" "cpu"
#:         STRINGS "cpu;cuda;hip" CATEGORY "Build Targets" ADVANCED)
macro(einsums_option option type description default)
  set(options ADVANCED)
  set(one_value_args CATEGORY DEPENDS)
  set(multi_value_args STRINGS)
  cmake_parse_arguments(
    EINSUMS_OPTION "${options}" "${one_value_args}" "${multi_value_args}" ${ARGN}
  )

  if("${type}" STREQUAL "BOOL")
    # Use regular CMake options for booleans
    if(NOT EINSUMS_OPTION_DEPENDS)
      option(${option} "${description}" ${default})
    else()
      cmake_dependent_option(${option} "${description}" ${default} "${EINSUMS_OPTION_DEPENDS}" OFF)
    endif()
  else()
    if(EINSUMS_OPTION_DEPENDS)
      message(FATAL_ERROR "einsums_option DEPENDS keyword can only be used with BOOL options")
    endif()
    # Use custom cache variables for other types
    if(NOT DEFINED ${option})
      set(${option}
          ${default}
          CACHE ${type} "${description}" FORCE
      )
    else()
      get_property(
        _option_is_cache_property
        CACHE "${option}"
        PROPERTY TYPE
        SET
      )
      if(NOT _option_is_cache_property)
        set(${option}
            ${default}
            CACHE ${type} "${description}" FORCE
        )
        if(EINSUMS_OPTION_ADVANCED)
          mark_as_advanced(${option})
        endif()
      else()
        set_property(CACHE "${option}" PROPERTY HELPSTRING "${description}")
        set_property(CACHE "${option}" PROPERTY TYPE ${type})
      endif()
    endif()

    if(EINSUMS_OPTION_STRINGS)
      if("${type}" STREQUAL "STRING")
        set_property(CACHE "${option}" PROPERTY STRINGS "${EINSUMS_OPTION_STRINGS}")
      else()
        message(FATAL_ERROR "einsums_option STRINGS keyword can only be used with STRING type")
      endif()
    endif()
  endif()

  if(EINSUMS_OPTION_ADVANCED)
    mark_as_advanced(${option})
  endif()

  set_property(GLOBAL APPEND PROPERTY EINSUMS_MODULE_CONFIG_EINSUMS ${option})

  set(_category "Generic")
  if(EINSUMS_OPTION_CATEGORY)
    set(_category "${EINSUMS_OPTION_CATEGORY}")
  endif()
  set(${option}Category
      ${_category}
      CACHE INTERNAL ""
  )
endmacro()
