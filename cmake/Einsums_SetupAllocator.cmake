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
  # mirroring how fmt/Catch2/spdlog are handled. v3.3.2 is the version the
  # arena behavior in DESIGN-memory-pool.md was verified against.
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
