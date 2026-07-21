#----------------------------------------------------------------------------------------------
# Copyright (c) The Einsums Developers. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
#----------------------------------------------------------------------------------------------

include(FetchContent)

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
  fetchcontent_declare(
    spdlog
    GIT_REPOSITORY https://github.com/gabime/spdlog.git
    GIT_TAG v1.x
  )
  fetchcontent_makeavailable(spdlog)
endblock()

# PIC so the static archive links into the shared libEinsums on ELF/Mach-O.
if(TARGET spdlog)
  set_target_properties(spdlog PROPERTIES POSITION_INDEPENDENT_CODE ON)
endif()
