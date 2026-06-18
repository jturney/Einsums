#----------------------------------------------------------------------------------------------
# Copyright (c) The Einsums Developers. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
#----------------------------------------------------------------------------------------------

#:
#: einsums_add_library
#:
#:    Define an Einsums library target (SHARED/STATIC/OBJECT) with project‑standard setup.
#:
#:    A convenience wrapper around ``add_library`` that normalizes sources/headers,
#:    applies GPU language tweaks for HIP/NVHPC, configures output directories (optionally with a
#:    suffix), and then delegates policy‑specific wiring to ``einsums_setup_target``.
#:
#:    **Signature**
#:    ``einsums_add_library(<name>
#:        [EXCLUDE_FROM_ALL]
#:        [INTERNAL_FLAGS]
#:        [NOLIBS]
#:        [NOEXPORT]
#:        [STATIC]
#:        [OBJECT]
#:        [NONAMEPREFIX]
#:        [UNITY_BUILD]
#:        [FOLDER <ide-folder>]
#:        [SOURCE_ROOT <dir>] [HEADER_ROOT <dir>]
#:        [SOURCE_GLOB <pattern>] [HEADER_GLOB <pattern>]
#:        [OUTPUT_SUFFIX <subdir>]
#:        [INSTALL_SUFFIX <dir>]
#:        [SOURCES <...>] [HEADERS <...>] [AUXILIARY <...>]
#:        [DEPENDENCIES <targets...>]
#:        [COMPILER_FLAGS <...>] [LINK_FLAGS <...>]
#:    )``
#:
#:    **Positional**
#:    - ``name`` *(required)*: Target name.
#:
#:    **Boolean options**
#:    - ``EXCLUDE_FROM_ALL``: Create the target but omit it from the default build.
#:    - ``INTERNAL_FLAGS``: Forward a project‑internal flag to ``einsums_setup_target``.
#:    - ``NOLIBS``: Forward to `einsums_setup_target` to avoid linking default Einsums libs.
#:    - ``NOEXPORT``: Suppress export setup (if omitted, export is enabled).
#:    - ``STATIC``: Build as a static library.
#:    - ``OBJECT``: Build as an object library.
#:    - ``NONAMEPREFIX``: (Reserved) Request no name prefix (toolchain‑specific).
#:    - ``UNITY_BUILD``: Opt into unity builds via `einsums_setup_target`.
#:
#:    **One‑value keywords**
#:    - ``FOLDER``: IDE grouping (e.g., “Libs/Core”).
#:    - ``SOURCE_ROOT`` / ``HEADER_ROOT``: Roots for source grouping (default ``"."``).
#:    - ``SOURCE_GLOB`` / ``HEADER_GLOB``: (Reserved) Glob patterns (expansion handled upstream).
#:    - ``OUTPUT_SUFFIX``: Extra subdirectory under runtime/lib/archive output dirs.
#:    - ``INSTALL_SUFFIX``: Override install destination directory (lib/bin); prepared internally.
#:
#:    **Multi‑value keywords**
#:    - ``SOURCES``: Source files.
#:    - ``HEADERS``: Header files (for IDE visibility).
#:    - ``AUXILIARY``: Extra files to attach to the target.
#:    - ``DEPENDENCIES``: Link/usage dependencies.
#:    - ``COMPILER_FLAGS``: Extra compile options (forwarded).
#:    - ``LINK_FLAGS``: Extra link options (forwarded).
#:
#:    **Behavior**
#:    1. Validates that ``STATIC`` and ``OBJECT`` are **not** both set; errors if so.
#:    2. Defaults ``SOURCE_ROOT``/``HEADER_ROOT`` to ``"."`` and logs inputs.
#:    3. Normalizes source lists via ``einsums_add_library_sources_noglob`` and groups
#:       files with ``einsums_add_source_group``.
#:    4. **GPU source handling**
#:       - If ``EINSUMS_WITH_HIP`` is ON, ``.cu`` sources are marked ``LANGUAGE HIP``.
#:       - With NVHPC, ``.cu`` sources receive compiler‑specific CUDA flags via
#:         ``einsums_add_nvhpc_cuda_flags``.
#:    5. Chooses link type: ``STATIC`` → static, ``OBJECT`` → object, otherwise **shared**.
#:    6. Invokes ``add_library`` with the chosen type, optional
#:       ``EXCLUDE_FROM_ALL``, and the collected sources/headers/aux files.
#:    7. **Output directories** — If ``OUTPUT_SUFFIX`` is set, adjusts
#:       ``RUNTIME/LIBRARY/ARCHIVE_OUTPUT_DIRECTORY`` (per‑config on MSVC; single‑config otherwise)
#:       under ``${EINSUMS_WITH_BINARY_DIR}``.
#:    8. Accumulates target policy flags (``NOLIBS``, export on by default unless ``NOEXPORT``,
#:       ``INTERNAL_FLAGS``, ``UNITY_BUILD``) and calls:
#:       ::
#:          einsums_setup_target(
#:            <name>
#:            TYPE LIBRARY
#:            NAME <name>
#:            FOLDER <...>
#:            COMPILE_FLAGS <...>
#:            LINK_FLAGS <...>
#:            DEPENDENCIES <...>
#:          )
#:
#:    **Notes**
#:    - Installation destinations and certain flags (e.g., PDB install on MSVC, NONAMEPREFIX) are
#:      prepared internally but must be honored by ``einsums_setup_target`` (or other
#:      project logic). If you expect install rules to be applied here, ensure
#:      `einsums_setup_target` consumes these settings.
#:    - When both ``STATIC`` and ``OBJECT`` are requested, the function aborts with a clear error.
#:
#:    **Examples**
#:    Build a shared library with IDE grouping and deps:
#:    .. code-block:: cmake
#:
#:       einsums_add_library(einsums_core
#:         FOLDER "Libs/Core"
#:         SOURCE_ROOT "${CMAKE_SOURCE_DIR}/src"
#:         HEADER_ROOT "${CMAKE_SOURCE_DIR}/include"
#:         SOURCES src/core/a.cpp src/core/b.cpp
#:         HEADERS include/einsums/a.hpp include/einsums/b.hpp
#:         DEPENDENCIES fmt::fmt
#:       )
#:
#:    Static lib into a custom output tree, unity build:
#:    .. code-block:: cmake
#:
#:       einsums_add_library(einsums_utils
#:         STATIC
#:         OUTPUT_SUFFIX "tools"
#:         UNITY_BUILD
#:         SOURCES src/utils/x.cpp src/utils/y.cpp
#:       )
#:
#:    Object library with no default build and no export:
#:    .. code-block:: cmake
#:
#:       einsums_add_library(einsums_objs
#:         OBJECT
#:         EXCLUDE_FROM_ALL
#:         NOEXPORT
#:         SOURCES src/lowlevel/part1.cpp src/lowlevel/part2.cpp
#:       )
#:
#:    **See also**
#:    - ``add_library``
#:    - ``einsums_setup_target``
#:    - ``einsums_add_library_sources_noglob``
#:    - ``einsums_add_source_group``
function(einsums_add_library name)
  set(options
      EXCLUDE_FROM_ALL
      INTERNAL_FLAGS
      NOLIBS
      NOEXPORT
      STATIC
      OBJECT
      NONAMEPREFIX
      UNITY_BUILD
  )
  set(one_value_args
      FOLDER
      SOURCE_ROOT
      HEADER_ROOT
      SOURCE_GLOB
      HEADER_GLOB
      OUTPUT_SUFFIX
      INSTALL_SUFFIX
  )
  set(multi_value_args SOURCES HEADERS AUXILIARY DEPENDENCIES COMPILER_FLAGS LINK_FLAGS)
  cmake_parse_arguments(${name} "${options}" "${one_value_args}" "${multi_value_args}" ${ARGN})

  if(${name}_OBJECT AND ${name}_STATIC)
    einsums_error(
      "Trying to create ${name} library with both STATIC and OBJECT. Only one can be used at the same time."
    )
  endif()

  if(NOT ${name}_SOURCE_ROOT)
    set(${name}_SOURCE_ROOT ".")
  endif()
  einsums_debug("add_library.${name}" "${name}_SOURCE_ROOT: ${${name}_SOURCE_ROOT}")

  if(NOT ${name}_HEADER_ROOT)
    set(${name}_HEADER_ROOT ".")
  endif()
  einsums_debug("add_library.${name}" "${name}_HEADER_ROOT: ${${name}_HEADER_ROOT}")

  einsums_add_library_sources_noglob(${name} SOURCES "${${name}_SOURCES}")

  einsums_add_source_group(
    NAME ${name}
    CLASS "Source Files"
    ROOT ${${name}_SOURCE_ROOT}
    TARGETS ${${name}_SOURCES}
  )

  einsums_print_list("DEBUG" "add_library.${name}" "Sources for ${name}" ${name}_SOURCES)
  einsums_print_list("DEBUG" "add_library.${name}" "Headers for ${name}" ${name}_HEADERS)
  einsums_print_list("DEBUG" "add_library.${name}" "Dependencies for ${name}" ${name}_DEPENDENCIES)

  set(exclude_from_all)

  if(${name}_EXCLUDE_FROM_ALL)
    set(exclude_from_all EXCLUDE_FROM_ALL)
  else()
    if(MSVC)
      set(library_install_destination ${CMAKE_INSTALL_BINDIR})
    else()
      set(library_install_destination ${CMAKE_INSTALL_LIBDIR})
    endif()
    set(archive_install_destination ${CMAKE_INSTALL_LIBDIR})
    set(runtime_install_destination ${CMAKE_INSTALL_BINDIR})
    if(${name}_INSTALL_SUFFIX)
      set(library_install_destination ${${name}_INSTALL_SUFFIX})
      set(archive_install_destination ${${name}_INSTALL_SUFFIX})
      set(runtime_install_destination ${${name}_INSTALL_SUFFIX})
    endif()
    # cmake-format: off
    set(_target_flags
        INSTALL INSTALL_FLAGS
        LIBRARY DESTINATION ${library_install_destination}
        ARCHIVE DESTINATION ${archive_install_destination}
        RUNTIME DESTINATION ${runtime_install_destination}
    )
    # cmake-format: on

    # install PDB if needed
    if(MSVC AND NOT ${name}_STATIC)
      # cmake-format: off
      set(_target_flags
          ${_target_flags}
          INSTALL_PDB $<TARGET_PDB_FILE:${name}>
          DESTINATION ${runtime_install_destination}
          CONFIGURATIONS Debug RelWithDebInfo
          OPTIONAL
      )
      # cmake-format: on
    endif()
  endif()

  if(${name}_NONAMEPREFIX)
    set(_target_flags ${_target_flags} NONAMEPREFIX)
  endif()

  if(${name}_STATIC)
    set(${name}_linktype STATIC)
  elseif(${name}_OBJECT)
    set(${name}_linktype OBJECT)
  else()
    set(${name}_linktype SHARED)
  endif()

  if(EINSUMS_WITH_HIP)
    foreach(source ${${name}_SOURCES})
      get_filename_component(extension ${source} EXT)
      if(${extension} STREQUAL ".cu")
        set_source_files_properties(${source} PROPERTIES LANGUAGE HIP)
      endif()
    endforeach()
  endif()

  if(CMAKE_CXX_COMPILER_ID STREQUAL "NVHPC")
    foreach(source ${${name}_SOURCES})
      get_filename_component(extension ${source} EXT)
      if(${extension} STREQUAL ".cu")
        einsums_add_nvhpc_cuda_flags(${source})
      endif()
    endforeach()
  endif()

  add_library(
    ${name} ${${name}_linktype} ${exclude_from_all} ${${name}_SOURCES} ${${name}_HEADERS}
            ${${name}_AUXILIARY}
  )

  if(${name}_OUTPUT_SUFFIX)
    if(MSVC)
      set_target_properties(
        ${name}
        PROPERTIES RUNTIME_OUTPUT_DIRECTORY_RELEASE
                   "${EINSUMS_WITH_BINARY_DIR}/Release/bin/${${name}_OUTPUT_SUFFIX}"
                   LIBRARY_OUTPUT_DIRECTORY_RELEASE
                   "${EINSUMS_WITH_BINARY_DIR}/Release/bin/${${name}_OUTPUT_SUFFIX}"
                   ARCHIVE_OUTPUT_DIRECTORY_RELEASE
                   "${EINSUMS_WITH_BINARY_DIR}/Release/lib/${${name}_OUTPUT_SUFFIX}"
                   RUNTIME_OUTPUT_DIRECTORY_DEBUG
                   "${EINSUMS_WITH_BINARY_DIR}/Debug/bin/${${name}_OUTPUT_SUFFIX}"
                   LIBRARY_OUTPUT_DIRECTORY_DEBUG
                   "${EINSUMS_WITH_BINARY_DIR}/Debug/bin/${${name}_OUTPUT_SUFFIX}"
                   ARCHIVE_OUTPUT_DIRECTORY_DEBUG
                   "${EINSUMS_WITH_BINARY_DIR}/Debug/lib/${${name}_OUTPUT_SUFFIX}"
                   RUNTIME_OUTPUT_DIRECTORY_MINSIZEREL
                   "${EINSUMS_WITH_BINARY_DIR}/MinSizeRel/bin/${${name}_OUTPUT_SUFFIX}"
                   LIBRARY_OUTPUT_DIRECTORY_MINSIZEREL
                   "${EINSUMS_WITH_BINARY_DIR}/MinSizeRel/bin/${${name}_OUTPUT_SUFFIX}"
                   ARCHIVE_OUTPUT_DIRECTORY_MINSIZEREL
                   "${EINSUMS_WITH_BINARY_DIR}/MinSizeRel/lib/${${name}_OUTPUT_SUFFIX}"
                   RUNTIME_OUTPUT_DIRECTORY_RELWITHDEBINFO
                   "${EINSUMS_WITH_BINARY_DIR}/RelWithDebInfo/bin/${${name}_OUTPUT_SUFFIX}"
                   LIBRARY_OUTPUT_DIRECTORY_RELWITHDEBINFO
                   "${EINSUMS_WITH_BINARY_DIR}/RelWithDebInfo/bin/${${name}_OUTPUT_SUFFIX}"
                   ARCHIVE_OUTPUT_DIRECTORY_RELWITHDEBINFO
                   "${EINSUMS_WITH_BINARY_DIR}/RelWithDebInfo/lib/${${name}_OUTPUT_SUFFIX}"
      )
    else()
      set_target_properties(
        ${name}
        PROPERTIES RUNTIME_OUTPUT_DIRECTORY
                   "${EINSUMS_WITH_BINARY_DIR}/bin/${${name}_OUTPUT_SUFFIX}"
                   LIBRARY_OUTPUT_DIRECTORY
                   "${EINSUMS_WITH_BINARY_DIR}/lib/${${name}_OUTPUT_SUFFIX}"
                   ARCHIVE_OUTPUT_DIRECTORY
                   "${EINSUMS_WITH_BINARY_DIR}/lib/${${name}_OUTPUT_SUFFIX}"
      )
    endif()
  endif()

  # get public and private compile options that einsums needs
  if(${${name}_NOLIBS})
    set(_target_flags ${_target_flags} NOLIBS)
  endif()

  if(NOT ${${name}_NOEXPORT})
    set(_target_flags ${_target_flags} EXPORT)
  endif()

  if(${name}_INTERNAL_FLAGS)
    set(_target_flags ${_target_flags} INTERNAL_FLAGS)
  endif()

  if(${name}_UNITY_BUILD)
    set(_target_flags ${_target_flags} UNITY_BUILD)
  endif()

  einsums_setup_target(
    ${name}
    TYPE LIBRARY
    NAME ${name}
    FOLDER ${${name}_FOLDER}
    COMPILE_FLAGS ${${name}_COMPILE_FLAGS}
    LINK_FLAGS ${${name}_LINK_FLAGS}
    DEPENDENCIES ${${name}_DEPENDENCIES}
  )
endfunction()
