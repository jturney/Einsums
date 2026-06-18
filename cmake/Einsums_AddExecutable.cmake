#----------------------------------------------------------------------------------------------
# Copyright (c) The Einsums Developers. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
#----------------------------------------------------------------------------------------------

#:
#: einsums_add_executable
#:
#:    Define an Einsums executable target with rich options (install/layout, GPU language, flags, unity, etc.).
#:
#:    A convenience wrapper around ``add_executable`` plus project‑standard setup:
#:    source/headers discovery, GPU language handling (CUDA/HIP/NVHPC), macOS dSYM generation,
#:    custom output names/suffixes, optional install rules, and forwarding of build policy flags to
#:    ``einsums_setup_target``.
#:
#:    **Signature**
#:    ``einsums_add_executable(<name>
#:        [GPU]
#:        [EXCLUDE_FROM_ALL]
#:        [EXCLUDE_FROM_DEFAULT_BUILD]
#:        [INTERNAL_FLAGS]
#:        [NOLIBS]
#:        [UNITY_BUILD]
#:        [NOINSTALL]
#:        [INI <file>]
#:        [FOLDER <ide-folder>]
#:        [SOURCE_ROOT <dir>]
#:        [HEADER_ROOT <dir>]
#:        [SOURCE_GLOB <pattern>]
#:        [HEADER_GLOB <pattern>]
#:        [OUTPUT_NAME <basename>]
#:        [OUTPUT_SUFFIX <subdir>]
#:        [INSTALL_SUFFIX <dir>]
#:        [LANGUAGE <C|CXX|CUDA|HIP>]
#:        [SOURCES <...>] [HEADERS <...>] [AUXILIARY <...>]
#:        [DEPENDENCIES <targets...>]
#:        [COMPILE_FLAGS <...>] [LINK_FLAGS <...>]
#:    )``
#:
#:    **Positional**
#:    - ``name`` *(required)*: Target name.
#:
#:    **Boolean options**
#:    - ``GPU``: Mark the target as GPU code. Sets target language to CUDA/HIP (or NVHPC CUDA link).
#:    - ``EXCLUDE_FROM_ALL``: Do not build by default (opt‑in only).
#:    - ``EXCLUDE_FROM_DEFAULT_BUILD``: Exclude from default config build on multi‑config generators.
#:    - ``INTERNAL_FLAGS``: Forwarded to ``einsums_setup_target`` to enable internal policy flags.
#:    - ``NOLIBS``: Forwarded to `einsums_setup_target`; avoid linking standard Einsums libs.
#:    - ``UNITY_BUILD``: Forwarded to `einsums_setup_target` to opt into unity builds.
#:    - ``NOINSTALL``: Suppress install rules for this executable.
#:
#:    **One‑value keywords**
#:    - ``INI``: (reserved) Path to an INI file for target‑specific settings.
#:    - ``FOLDER``: IDE folder/group (e.g., “Apps/Tools”).
#:    - ``SOURCE_ROOT``: Base path for source grouping (default ``"."``).
#:    - ``HEADER_ROOT``: Base path for header grouping (default ``"."``).
#:    - ``SOURCE_GLOB``, ``HEADER_GLOB``: (reserved) Glob patterns—actual expansion handled upstream.
#:    - ``OUTPUT_NAME``: Basename of produced artifact (prefixed by `EINSUMS_WITH_EXECUTABLE_PREFIX`).
#:    - ``OUTPUT_SUFFIX``: Extra path component under the runtime output directory (e.g., ``tools``).
#:    - ``INSTALL_SUFFIX``: Override install destination (defaults to ``CMAKE_INSTALL_BINDIR``).
#:    - ``LANGUAGE``: Nominal language (default ``CXX``). GPU/compilers may override per‑file.
#:
#:    **Multi‑value keywords**
#:    - ``SOURCES``: Source files for the executable.
#:    - ``HEADERS``: Header files to attach for IDE organization.
#:    - ``AUXILIARY``: Extra files to add to the target (scripts, data).
#:    - ``DEPENDENCIES``: Link/usage dependencies (targets).
#:    - ``COMPILE_FLAGS``: Extra compile options (forwarded to `einsums_setup_target`).
#:    - ``LINK_FLAGS``: Extra link options (forwarded).
#:
#:    **Behavior (high‑level)**
#:    1. **Defaults & roots** — Ensures ``LANGUAGE=CXX``, ``SOURCE_ROOT``/``HEADER_ROOT`` default to ``"."``.
#:    2. **Source staging** — Uses project helper ``einsums_add_library_sources_noglob``
#:       to normalize provided ``SOURCES`` (and attach headers); groups files with
#:       ``einsums_add_source_group`` under ``FOLDER``/roots for IDEs.
#:    3. **GPU language tweaks**
#:       - When ``EINSUMS_WITH_HIP``: `.cu` sources are coerced to ``LANGUAGE HIP``.
#:       - With NVHPC: `.cu` sources get compiler‑specific flags via
#:         ``einsums_add_nvhpc_cuda_flags``; if GPU content is present, link with ``-cuda``.
#:       - With ``GPU`` option:
#:         * HIP available → target ``LANGUAGE HIP``,
#:         * NVHPC → set link flags ``-cuda``,
#:         * otherwise → target ``LANGUAGE CUDA``.
#:    4. **Target creation** — Calls ``add_executable`` with sources/headers/auxiliary.
#:    5. **macOS dSYM** — On Apple with Debug/RelWithDebInfo, post‑build ``dsymutil`` runs
#:       to generate a `.dSYM` bundle.
#:    6. **Output naming/layout**
#:       - ``OUTPUT_NAME`` sets the runtime name (prefixed).
#:       - ``OUTPUT_SUFFIX`` adjusts runtime output directories per‑config (MSVC) or single‑config.
#:    7. **Install rules**
#:       - Skipped if ``NOINSTALL`` **or** ``EXCLUDE_FROM_ALL`` is set.
#:       - Otherwise installs the exe to ``CMAKE_INSTALL_BINDIR`` (or ``INSTALL_SUFFIX``).
#:       - On MSVC, installs PDBs for Debug/RelWithDebInfo.
#:    8. **Exclusion controls**
#:       - Applies ``EXCLUDE_FROM_ALL`` and/or ``EXCLUDE_FROM_DEFAULT_BUILD`` properties to the target.
#:    9. **Forwarding to project policy**
#:       - Constructs `_target_flags` from ``NOLIBS``, ``INTERNAL_FLAGS``, ``UNITY_BUILD`` and calls
#:         ``einsums_setup_target``:
#:         ``einsums_setup_target(<name> TYPE EXECUTABLE FOLDER <...> COMPILE_FLAGS <...> LINK_FLAGS <...> DEPENDENCIES <...> <_target_flags>)``.
#:
#:    **Notes**
#:    - If ``EINSUMS_WITH_HIP`` is enabled, `.cu` files can be compiled as HIP sources; this allows
#:      mixed codebases to compile under HIP toolchains.
#:    - NVHPC CUDA linkage is handled by setting target ``LINK_FLAGS`` to ``-cuda`` when GPU sources are detected.
#:    - File grouping/log output is produced via project helpers (`einsums_print_list`) for traceability.
#:
#:    **Examples**
#:    Minimal:
#:    .. code-block:: cmake
#:
#:       einsums_add_executable(einsums-tool
#:         SOURCES tools/main.cpp
#:         DEPENDENCIES Einsums::einsums
#:       )
#:
#:    CUDA (NVHPC) app with custom output dir and no install:
#:    .. code-block:: cmake
#:
#:       einsums_add_executable(simulator
#:         GPU
#:         SOURCES src/sim.cu src/util.cpp
#:         OUTPUT_SUFFIX "gpu"
#:         NOINSTALL
#:         DEPENDENCIES Einsums::einsums_cuda
#:         COMPILE_FLAGS -DUSE_FAST_MATH
#:       )
#:
#:    HIP build, IDE grouping, unity:
#:    .. code-block:: cmake
#:
#:       einsums_add_executable(bench
#:         FOLDER "Apps/Benchmarks"
#:         SOURCE_ROOT "${CMAKE_SOURCE_DIR}/apps/bench"
#:         HEADER_ROOT "${CMAKE_SOURCE_DIR}/apps/bench"
#:         SOURCES apps/bench/main.cpp apps/bench/kernels.cu
#:         UNITY_BUILD
#:         DEPENDENCIES Einsums::einsums_hip
#:       )
#:
#:    **See also**
#:    - ``add_executable``
#:    - ``einsums_setup_target``
#:    - ``einsums_add_library_sources_noglob``, ``einsums_add_source_group``
#:    - ``dsymutil``
function(einsums_add_executable name)
  set(options
      GPU
      EXCLUDE_FROM_ALL
      EXCLUDE_FROM_DEFAULT_BUILD
      INTERNAL_FLAGS
      NOLIBS
      UNITY_BUILD
      NOINSTALL
  )
  set(one_value_args
      INI
      FOLDER
      SOURCE_ROOT
      HEADER_ROOT
      SOURCE_GLOB
      HEADER_GLOB
      OUTPUT_NAME
      OUTPUT_SUFFIX
      INSTALL_SUFFIX
      LANGUAGE
  )
  set(multi_value_args SOURCES HEADERS AUXILIARY DEPENDENCIES COMPILE_FLAGS LINK_FLAGS)
  cmake_parse_arguments(${name} "${options}" "${one_value_args}" "${multi_value_args}" ${ARGN})

  if(NOT ${name}_LANGUAGE)
    set(${name}_LANGUAGE CXX)
  endif()

  if(NOT ${name}_SOURCE_ROOT)
    set(${name}_SOURCE_ROOT ".")
  endif()
  einsums_debug("Add executable ${name}: ${name}_SOURCE_ROOT: ${${name}_SOURCE_ROOT}")

  if(NOT ${name}_HEADER_ROOT)
    set(${name}_HEADER_ROOT ".")
  endif()
  einsums_debug("Add executable ${name}: ${name}_HEADER_ROOT: ${${name}_HEADER_ROOT}")

  einsums_add_library_sources_noglob(${name}_executable SOURCES "${${name}_SOURCES}")

  einsums_add_source_group(
    NAME ${name}
    CLASS "Source Files"
    ROOT ${${name}_SOURCE_ROOT}
    TARGETS ${${name}_executable_SOURCES}
  )

  set(${name}_SOURCES ${${name}_executable_SOURCES})
  set(${name}_HEADERS ${${name}_executable_HEADERS})

  einsums_print_list("DEBUG" "Add executable ${name}: Sources for ${name}" ${name}_SOURCES)
  einsums_print_list("DEBUG" "Add executable ${name}: Headers for ${name}" ${name}_HEADERS)
  einsums_print_list(
    "DEBUG" "Add executable ${name}: Dependencies for ${name}" ${name}_DEPENDENCIES
  )

  set(_target_flags)

  # add the executable build target
  if(${name}_EXCLUDE_FROM_ALL)
    set(exclude_from_all EXCLUDE_FROM_ALL TRUE)
  elseif(NOT ${name}_NOINSTALL)
    set(install_destination ${CMAKE_INSTALL_BINDIR})
    if(${name}_INSTALL_SUFFIX)
      set(install_destination ${${name}_INSTALL_SUFFIX})
    endif()
    set(_target_flags INSTALL INSTALL_FLAGS DESTINATION ${install_destination})
    if(MSVC)
      set(_target_flags
          ${_target_flags}
          INSTALL_PDB
          $<TARGET_PDB_FILE:${name}>
          DESTINATION
          ${install_destination}
          CONFIGURATIONS
          Debug
          RelWithDebInfo
          OPTIONAL
      )
    endif()
  endif()

  if(${name}_EXCLUDE_FROM_DEFAULT_BUILD)
    set(exclude_from_all ${exclude_from_all} EXCLUDE_FROM_DEFAULT_BUILD TRUE)
  endif()

  if(EINSUMS_WITH_HIP)
    foreach(source ${${name}_SOURCES})
      get_filename_component(extension ${source} EXT)
      if(${extension} STREQUAL ".cu")
        set_source_file_properties(${source} PROPERTIES LANGUAGE HIP)
      endif()
    endforeach()
  endif()

  set(target_has_gpu_sources FALSE)
  if(CMAKE_CXX_COMPILER_ID STREQUAL "NVHPC")
    foreach(source ${${name}_SOURCES})
      get_filename_component(extension ${source} EXT)
      if(${extension} STREQUAL ".cu")
        einsums_add_nvhpc_cuda_flags(${source})
        set(target_has_gpu_sources TRUE)
      endif()
    endforeach()
  endif()

  add_executable(${name} ${${name}_SOURCES} ${${name}_HEADERS} ${${name}_AUXILIARY})

  # Create a .dSYM file on macOS
  if (APPLE AND dsymutil_EXECUTABLE AND (${CMAKE_BUILD_TYPE} STREQUAL "Debug" OR ${CMAKE_BUILD_TYPE} STREQUAL "RelWithDebInfo"))
    add_custom_command(
      TARGET ${name}
      POST_BUILD
      COMMENT "Running dsymutil on: $<TARGET_FILE:${name}>"
      VERBATIM
      COMMAND dsymutil $<TARGET_FILE:${name}>
    )
  endif()

  if(${name}_OUTPUT_SUFFIX)
    if(MSVC)
      set_target_properties(
        ${name}
        PROPERTIES RUNTIME_OUTPUT_DIRECTORY_RELEASE
                   "${EINSUMS_WITH_BINARY_DIR}/Release/bin/${${name}_OUTPUT_SUFFIX}"
                   RUNTIME_OUTPUT_DIRECTORY_DEBUG
                   "${EINSUMS_WITH_BINARY_DIR}/Debug/bin/${${name}_OUTPUT_SUFFIX}"
                   RUNTIME_OUTPUT_DIRECTORY_MINSIZEREL
                   "${EINSUMS_WITH_BINARY_DIR}/MinSizeRel/bin/${${name}_OUTPUT_SUFFIX}"
                   RUNTIME_OUTPUT_DIRECTORY_RELWITHDEBINFO
                   "${EINSUMS_WITH_BINARY_DIR}/RelWithDebInfo/bin/${${name}_OUTPUT_SUFFIX}"
      )
    else()
      set_target_properties(
        ${name} PROPERTIES RUNTIME_OUTPUT_DIRECTORY
                           "${EINSUMS_WITH_BINARY_DIR}/bin/${${name}_OUTPUT_SUFFIX}"
      )
    endif()
  endif()

  if(CMAKE_CXX_COMPILER_ID STREQUAL "NVHPC" AND target_has_gpu_sources)
    set_target_properties(${name} PROPERTIES LINK_FLAGS "-cuda")
  endif()

  if(${name}_GPU)
    if(EINSUMS_WITH_HIP)
      set_target_properties(${name} PROPERTIES LANGUAGE HIP)
    elseif(CMAKE_CXX_COMPILER_ID STREQUAL "NVHPC")
      set_target_properties(${name} PROPERTIES LINK_FLAGS "-cuda")
    else()
      set_target_properties(${name} PROPERTIES LANGUAGE CUDA)
    endif()
  endif()

  if(${name}_OUTPUT_NAME)
    set_target_properties(
      ${name} PROPERTIES OUTPUT_NAME "${EINSUMS_WITH_EXECUTABLE_PREFIX}${${name}_OUTPUT_NAME}"
    )
  else()
    set_target_properties(${name} PROPERTIES OUTPUT_NAME "${EINSUMS_WITH_EXECUTABLE_PREFIX}${name}")
  endif()

  if(exclude_from_all)
    set_target_properties(${name} PROPERTIES ${exclude_from_all})
  endif()

  if(${${name}_NOLIBS})
    set(_target_flags ${_target_flags} NOLIBS)
  endif()

  if(${name}_INTERNAL_FLAGS)
    set(_target_flags ${_target_flags} INTERNAL_FLAGS)
  endif()

  if(${name}_UNITY_BUILD)
    set(_target_flags ${_target_flags} UNITY_BUILD)
  endif()

  einsums_setup_target(
    ${name}
    TYPE EXECUTABLE
    FOLDER ${${name}_FOLDER}
    COMPILE_FLAGS ${${name}_COMPILE_FLAGS}
    LINK_FLAGS ${${name}_LINK_FLAGS}
    DEPENDENCIES ${${name}_DEPENDENCIES} ${_target_flags}
  )
endfunction()
