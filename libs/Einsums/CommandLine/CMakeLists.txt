#----------------------------------------------------------------------------------------------
# Copyright (c) The Einsums Developers. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
#----------------------------------------------------------------------------------------------

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

set(CommandLineHeaders Einsums/CommandLine/CommandLine.hpp)

set(CommandLineSources)

include(Einsums_AddModule)
einsums_add_module(
  Einsums CommandLine
  SOURCES ${CommandLineSources}
  HEADERS ${CommandLineHeaders}
  DEPENDENCIES
  MODULE_DEPENDENCIES Einsums_Config Einsums_Assertion Einsums_Logging
  CMAKE_SUBDIRS examples tests
  BASE_LIBNAME Einsums
)
