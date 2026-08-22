#----------------------------------------------------------------------------------------------
# Copyright (c) The Einsums Developers. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
#----------------------------------------------------------------------------------------------

include(FetchContent)

# The spdlog Einsums compiles against, named once. Two things read it: the tag
# fetched below, and the version an out-of-tree consumer is required to match in
# EinsumsConfig.cmake. Keeping one source for both is what stops the exported
# package from asking for a spdlog the library was never built with.
#
# Pinned rather than tracking the v1.x BRANCH it used to follow: a branch makes
# the headers folded into libEinsums a function of the day the worktree was
# configured, so two developers a month apart get different ones.
set(EINSUMS_SPDLOG_VERSION 1.17.0)

set(SPDLOG_INSTALL TRUE)
set(SPDLOG_FMT_EXTERNAL TRUE)

# Build spdlog as a STATIC library folded into libEinsums instead of shipping a
# separate shared spdlog. A shared spdlog on Windows lands in
# build/_deps/spdlog-build/, which is neither beside the test executables nor on
# PATH, so every test exe fails to start with 0xC0000135 (STATUS_DLL_NOT_FOUND).
# Static removes that runtime dependency entirely.
#
# spdlog is always built from source (no FIND_PACKAGE_ARGS): conda-forge ships
# only a shared spdlog, so preferring the system package would defeat the static
# choice on the platforms that have it. Building from source keeps the
# static/shared decision ours on every platform.
set(SPDLOG_BUILD_SHARED OFF CACHE BOOL "" FORCE)
block()
  # spdlog's add_library() honors BUILD_SHARED_LIBS; force it off inside this
  # scope so spdlog is STATIC even though the rest of the tree is shared.
  set(BUILD_SHARED_LIBS OFF)

  # ...and give it the visibility every other translation unit in the tree
  # already has. -fvisibility=hidden is carried by einsums_private_flags, which
  # only Einsums' own targets link, so spdlog was the single set of sources we
  # compile at default visibility. The mismatch is not cosmetic on Mach-O: an
  # object compiled hidden binds a weak symbol like std::piecewise_construct
  # directly, and when the same symbol also has a default-visibility definition
  # in libspdlog.a the linker reports every such reference -
  #
  #   ld: warning: direct access in function '...' to global weak symbol
  #   'std::__1::piecewise_construct' ... means the weak symbol cannot be
  #   overridden at runtime
  #
  # - which was 378 warnings on the macOS leg, all naming spdlog.cpp.o. Hidden
  # visibility only affects the dynamic symbol table of the final shared object,
  # and spdlog is folded statically into consumers, so nothing that resolves
  # today stops resolving.
  if(EINSUMS_WITH_HIDDEN_VISIBILITY)
    set(CMAKE_C_VISIBILITY_PRESET hidden)
    set(CMAKE_CXX_VISIBILITY_PRESET hidden)
    set(CMAKE_VISIBILITY_INLINES_HIDDEN ON)
  endif()

  # SYSTEM: spdlog's headers are compiled here, not merely consumed, so without
  # it they are held to einsums_private_flags' warning level and every
  # translation unit that reaches Logging.hpp reports spdlog's own diagnostics
  # as if they were ours. fmt 12.2 deprecated the string_view conversion that
  # spdlog 1.17's logger::log goes through, which is several warnings per TU
  # across most of the tree, for a line no Einsums change can affect. Upstream
  # has already replaced that call with fmt.get() on the v1.x branch; the fix
  # is simply not in a release yet.
  fetchcontent_declare(
    spdlog
    GIT_REPOSITORY https://github.com/gabime/spdlog.git
    GIT_TAG v${EINSUMS_SPDLOG_VERSION}
    SYSTEM
  )
  fetchcontent_makeavailable(spdlog)
endblock()

# PIC so the static archive links into the shared libEinsums on ELF/Mach-O.
if(TARGET spdlog)
  set_target_properties(spdlog PROPERTIES POSITION_INDEPENDENT_CODE ON)
endif()
