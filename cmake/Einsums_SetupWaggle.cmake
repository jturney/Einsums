#----------------------------------------------------------------------------------------------
# Copyright (c) The Einsums Developers. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
#----------------------------------------------------------------------------------------------

# Waggle (https://github.com/Einsums/Waggle), the profiler that began as Einsums' Profile module and
# that Einsums shares with the other libraries in a process: the collector library, the headers its
# zone macros come from, and the waggle Python package behind `einsums profiler`.
#
# An installed Waggle is used when found; otherwise the pinned commit is fetched and built here. To
# work on both at once, configure with -DFETCHCONTENT_SOURCE_DIR_WAGGLE=<Waggle checkout>.

include(FetchContent)

# For a Waggle built here. Its tests run with Einsums' unit tests, named to sit beside them.
set(WAGGLE_BUILD_TESTS ${EINSUMS_WITH_TESTS_UNIT})
set(WAGGLE_TEST_PREFIX "Tests.Unit.Waggle.")
if(Python_EXECUTABLE)
  # The interpreter Einsums found, for Waggle's viewer tests.
  set(Python3_EXECUTABLE "${Python_EXECUTABLE}")
endif()
if(EINSUMS_BUILD_PYTHON)
  # The waggle package beside the einsums package, in the build tree and in the install:
  # `einsums profiler` runs its viewer, and einsums.profile is its waggle._core.
  set(WAGGLE_PYTHON_STAGING_DIR "${CMAKE_BINARY_DIR}/lib")
  set(WAGGLE_PYTHON_INSTALL_DIR "${EINSUMS_INSTALL_PYMODDIR}")
  set(WAGGLE_BUILD_PYTHON ON)
endif()

fetchcontent_declare(
  Waggle
  GIT_REPOSITORY https://github.com/Einsums/Waggle.git
  GIT_TAG 05f5e5676e460013e0681b2d0e4beecf7aa99174 # main, 2026-10-03
  FIND_PACKAGE_ARGS
  0.1
  CONFIG
)
fetchcontent_makeavailable(Waggle)
