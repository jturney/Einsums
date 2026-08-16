#----------------------------------------------------------------------------------------------
# Copyright (c) The Einsums Developers. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
#----------------------------------------------------------------------------------------------

include(FetchContent)
fetchcontent_declare(
  cpptrace
  GIT_REPOSITORY https://github.com/jeremy-rifkin/cpptrace.git
  GIT_TAG v1.0.1 # <HASH or TAG>
  FIND_PACKAGE_ARGS
  1.0
)

# Build it static on Windows, whatever BUILD_SHARED_LIBS says.
#
# cpptrace is fetched and built here rather than found, so under the project's
# BUILD_SHARED_LIBS=ON it produces a cpptrace.dll - and that DLL lands in the
# FetchContent build tree, because RUNTIME_OUTPUT_DIRECTORY is set by
# einsums_add_library/einsums_add_executable and a third-party target never goes
# through either. Every executable linking libEinsums then fails to load with
# STATUS_DLL_NOT_FOUND (0xc0000135). Catch2 and fmt avoid this only by coming from the
# conda environment, already on PATH.
#
# Windows-only because there is nothing to fix elsewhere: ELF and Mach-O resolve the
# shared build through RPATH, which is why the Linux leg has always run with
# EINSUMS_WITH_BACKTRACES on and never noticed. Static also suits what this dependency
# is - small, with no ABI surface of ours to preserve - and removes the deployment
# question rather than answering it.
if(WIN32)
  set(_einsums_cpptrace_saved_shared "${BUILD_SHARED_LIBS}")
  set(BUILD_SHARED_LIBS OFF)
endif()

fetchcontent_makeavailable(cpptrace)

if(WIN32)
  set(BUILD_SHARED_LIBS "${_einsums_cpptrace_saved_shared}")
  unset(_einsums_cpptrace_saved_shared)
endif()
