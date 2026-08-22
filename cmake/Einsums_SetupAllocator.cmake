#----------------------------------------------------------------------------------------------
# Copyright (c) The Einsums Developers. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
#----------------------------------------------------------------------------------------------

include(Einsums_AddDefinitions)

# mimalloc is a hard requirement: the library's own allocation layer
# (einsums::memory::aligned_alloc and MemoryPool) is written against
# mi_malloc_aligned / mi_heap_new_in_arena, with no system-allocator fallback.
# EINSUMS_WITH_MALLOC survives, but it now only chooses whether the GLOBAL
# operator new/malloc traffic is redirected as well.
if(NOT TARGET einsums_dependencies_mimalloc)

  # Prefer an installed mimalloc (conda-forge carries one on every platform we
  # build); fall back to a source build so a bare environment still configures,
  # mirroring how fmt/Catch2/spdlog are handled. The fallback tag stays at the
  # version the arena behavior in DESIGN-memory-pool.md was verified against;
  # what an INSTALLED mimalloc has to satisfy is checked below, and is wider.
  include(FetchContent)
  fetchcontent_declare(
    mimalloc
    GIT_REPOSITORY https://github.com/microsoft/mimalloc.git
    GIT_TAG v3.3.2
    FIND_PACKAGE_ARGS
    CONFIG
  )

  set(MI_BUILD_TESTS OFF CACHE BOOL "" FORCE)
  set(MI_BUILD_OBJECT OFF CACHE BOOL "" FORCE)
  set(MI_OVERRIDE OFF CACHE BOOL "" FORCE)

  fetchcontent_makeavailable(mimalloc)

  # An installed mimalloc exports ``mimalloc``; a source build exports
  # ``mimalloc`` (shared) and ``mimalloc-static``.
  if(TARGET mimalloc)
    set(EINSUMS_MIMALLOC_TARGET mimalloc)
  elseif(TARGET mimalloc-static)
    set(EINSUMS_MIMALLOC_TARGET mimalloc-static)
  else()
    einsums_error("mimalloc is required but neither the ``mimalloc`` nor the ``mimalloc-static`` target exists after setup.")
  endif()

  # Which mimalloc the environment actually supplied, read from the header
  # rather than from find_package. Two reasons the version has to be checked
  # here instead of being requested above:
  #
  # * mimalloc's installed config file was generated with AnyNewerVersion and
  #   predates version-range support, so CMake silently keeps only the lower
  #   endpoint of a range and 3.4 satisfies a request for 3.3.
  # * that config reports the version as "3.4" with no patch component, and the
  #   release that matters here differs from its neighbour in the patch digit.
  #
  # MI_MALLOC_VERSION carries the full number (30402 == 3.4.2), so the header is
  # the only source precise enough to express the rule.
  #
  # The rule is >=3.3, <3.6, excluding 3.4.0 and 3.4.1. Each bound was measured
  # against the full test suite, one mimalloc source build per release:
  #
  # * Below 3.3 the arena entry points MemoryPool is written against
  #   (mi_heap_new_in_arena) do not exist.
  #
  # * 3.4.0 and 3.4.1 moved mimalloc's fixed macOS TLS slots from 108/109 to
  #   126/127, where they collide with slots whose pthread destructor is the
  #   system free. Every thread that allocated then aborts as it unwinds -
  #
  #     BUG_IN_CLIENT_OF_LIBMALLOC: POINTER_BEING_FREED_WAS_NOT_ALLOCATED
  #
  #   - with mimalloc's own per-thread heap as the pointer. It surfaces as 13
  #   tests dying at process exit, every one AFTER its last assertion passed, so
  #   the report is a bare SIGTRAP under a green log. Upstream reverted the
  #   slots in 3.4.2 (mimalloc issue #1333); 3.4.2 through 3.4.5 are clean.
  #
  # * 3.5 raised mi_arena_min_alignment() from 64 KiB to 256 MiB. That one was
  #   ours rather than upstream's, and is fixed: the alignment constrains an
  #   arena's BASE, not its size, so MemoryPool no longer rounds the size to it
  #   (see arena_size_for). 3.5 passes the suite.
  #
  # * The ceiling is a minor version above the newest series tested, not a
  #   known failure. Both breakages so far arrived on a minor bump and both
  #   were silent - a SIGTRAP after the last assertion, and a memory ceiling
  #   rejecting pools that used to fit - so a new series gets looked at before
  #   it is trusted rather than after.
  set(_einsums_mimalloc_header "")
  get_target_property(_einsums_mi_incdirs ${EINSUMS_MIMALLOC_TARGET} INTERFACE_INCLUDE_DIRECTORIES)
  foreach(_dir ${_einsums_mi_incdirs})
    if(EXISTS "${_dir}/mimalloc.h")
      set(_einsums_mimalloc_header "${_dir}/mimalloc.h")
      break()
    endif()
  endforeach()
  if(NOT _einsums_mimalloc_header AND EXISTS "${mimalloc_SOURCE_DIR}/include/mimalloc.h")
    set(_einsums_mimalloc_header "${mimalloc_SOURCE_DIR}/include/mimalloc.h")
  endif()

  if(_einsums_mimalloc_header)
    file(STRINGS "${_einsums_mimalloc_header}" _einsums_mi_ver_line REGEX "define[ \t]+MI_MALLOC_VERSION")
    string(REGEX MATCH "[0-9]+" _einsums_mi_ver "${_einsums_mi_ver_line}")
  endif()

  if(_einsums_mi_ver AND (_einsums_mi_ver LESS 30300
                          OR _einsums_mi_ver GREATER_EQUAL 30600
                          OR (_einsums_mi_ver GREATER_EQUAL 30400 AND _einsums_mi_ver LESS 30402)))
    einsums_error(
      "Found mimalloc ${_einsums_mi_ver} (MI_MALLOC_VERSION), which Einsums cannot use. "
      "Supported: >=3.3 and <3.6, excluding 3.4.0 and 3.4.1 (macOS TLS slot collision, "
      "mimalloc issue #1333, fixed in 3.4.2). "
      "In a conda environment: conda install 'mimalloc>=3.3,<3.6,!=3.4.0,!=3.4.1'."
    )
  endif()
  if(NOT _einsums_mi_ver)
    einsums_warn("Could not read MI_MALLOC_VERSION from the mimalloc headers; version compatibility is unchecked.")
  endif()

  add_library(einsums_dependencies_mimalloc INTERFACE)
  target_link_libraries(einsums_dependencies_mimalloc INTERFACE ${EINSUMS_MIMALLOC_TARGET})
  if(MSVC)
    target_compile_options(einsums_dependencies_mimalloc INTERFACE /INCLUDE:mi_version)
  endif()

  install(
    TARGETS einsums_dependencies_mimalloc
    EXPORT einsums_internal_targets
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR} COMPONENT einsums_dependencies_mimalloc
  )
  einsums_export_internal_targets(einsums_dependencies_mimalloc)
endif()

if(NOT TARGET einsums_dependencies_allocator)

  if(NOT EINSUMS_WITH_MALLOC)
    set(EINSUMS_WITH_MALLOC
        CACHE STRING "Redirect the global allocator. Supported values are mimalloc and system."
              ${DEFAULT_MALLOC}
    )
  endif()

  string(TOUPPER "${EINSUMS_WITH_MALLOC}" EINSUMS_WITH_MALLOC_UPPER)

  if(NOT EINSUMS_WITH_MALLOC_UPPER MATCHES "^(SYSTEM|MIMALLOC)$")
    einsums_error(
      "EINSUMS_WITH_MALLOC was set to ${EINSUMS_WITH_MALLOC}. Valid options for EINSUMS_WITH_MALLOC are: system and mimalloc."
    )
  endif()

  add_library(einsums_dependencies_allocator INTERFACE)
  target_link_libraries(einsums_dependencies_allocator INTERFACE einsums_dependencies_mimalloc)

  if("${EINSUMS_WITH_MALLOC_UPPER}" STREQUAL "MIMALLOC")
    set(EINSUMS_MALLOC_LIBRARY mimalloc)
    einsums_warn(
      "einsums is using mimalloc as the global allocator. Typically, exporting the following environment variables will further improve performance: MIMALLOC_EAGER_COMMIT_DELAY=0 and MIMALLOC_ALLOW_LARGE_OS_PAGES=1."
    )
  endif()

  einsums_info("Using ${EINSUMS_WITH_MALLOC} as the global allocator. The einsums allocation layer always uses mimalloc.")

  # convey selected allocator type to the build configuration
  if(NOT EINSUMS_FIND_PACKAGE)
    einsums_add_config_define(EINSUMS_HAVE_MALLOC "\"${EINSUMS_WITH_MALLOC}\"")
    einsums_add_config_define(EINSUMS_HAVE_MALLOC_${EINSUMS_WITH_MALLOC_UPPER})
  endif()

  install(
          TARGETS einsums_dependencies_allocator
          EXPORT einsums_internal_targets
          LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
          ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
          RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR} COMPONENT einsums_dependencies_allocator
  )
  einsums_export_internal_targets(einsums_dependencies_allocator)

endif()
