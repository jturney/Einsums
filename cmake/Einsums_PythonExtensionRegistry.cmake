#----------------------------------------------------------------------------------------------
# Copyright (c) The Einsums Developers. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
#----------------------------------------------------------------------------------------------

# Guard rail for the one build shape that cannot work: a STATIC libEinsums
# hosting more than one Python extension module.
#
# Extension modules are linked independently, so a static libEinsums is copied
# into each one, and every process-unique thing the library owns is copied with
# it. That is not a short list: roughly 67 ``::instance()`` and 110
# ``get_singleton()`` call sites, plus the sealed-world identity, the
# stage-module registry, the profiler and its string table, and the
# configuration maps. Two modules in one process then disagree about all of it.
#
# One module is fine, because there is exactly one copy. That is the normal
# static build and nothing here objects to it.
#
# Two or more is a different program, and it fails in ways that do not name
# their cause. The sealed handshake refuses modules that are not skewed, because
# they really did bind to a different copy. The profiler resolved a string id
# against a table that never issued it and took the consumer thread down with a
# SIGSEGV that ctest reported as a bare crash with no output. Both of those were
# real, both were paid for in CI time, and neither pointed at linkage.
#
# So count the modules and refuse the combination while the build is still being
# configured, where the message can say what is wrong.

# Record that ``_target`` is a Python extension module built by this project.
#
# Call once per module, right after it is created. Modules that are skipped by
# their own guards must not register: the count is of what will actually be
# built.
function(einsums_register_python_extension _target)
  set_property(GLOBAL APPEND PROPERTY EINSUMS_PYTHON_EXTENSIONS "${_target}")
endfunction()

# Refuse a static build that would produce more than one extension module.
#
# Call once, after every subdirectory that might create one has been added.
function(einsums_assert_python_extensions_can_share_a_library)
  if(BUILD_SHARED_LIBS)
    return()
  endif()

  get_property(_extensions GLOBAL PROPERTY EINSUMS_PYTHON_EXTENSIONS)
  list(LENGTH _extensions _count)
  if(_count LESS 2)
    return()
  endif()

  list(JOIN _extensions "\n    " _listed)
  message(
    FATAL_ERROR
      "BUILD_SHARED_LIBS=OFF builds a STATIC libEinsums, and this configuration "
      "would produce ${_count} Python extension modules:\n"
      "    ${_listed}\n\n"
      "Each one links its own copy of the library, so each gets a private "
      "profiler, string table, sealed-world identity, stage-module registry and "
      "configuration map. They cannot see each other's state, and the failures "
      "that follow do not name linkage as the cause: the sealed handshake "
      "refuses modules that are not actually skewed, and the profiler can "
      "resolve a string id against a table that never issued it.\n\n"
      "Either build shared (BUILD_SHARED_LIBS=ON, the default), or reduce this "
      "build to a single extension module. A static libEinsums hosts exactly "
      "one."
  )
endfunction()
