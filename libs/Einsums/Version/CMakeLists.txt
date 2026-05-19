#----------------------------------------------------------------------------------------------
# Copyright (c) The Einsums Developers. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
#----------------------------------------------------------------------------------------------

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

set(VersionHeaders Einsums/Version.hpp)

set(VersionSources Version.cpp)

include(Einsums_AddModule)
einsums_add_module(
  Einsums Version
  PYBIND
  SOURCES ${VersionSources}
  HEADERS ${VersionHeaders}
  DEPENDENCIES
  MODULE_DEPENDENCIES Einsums_Config
  CMAKE_SUBDIRS examples tests
  BASE_LIBNAME Einsums
)
