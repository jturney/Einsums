#----------------------------------------------------------------------------------------------
# Copyright (c) The Einsums Developers. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
#----------------------------------------------------------------------------------------------

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

set(RuntimeConfigurationHeaders Einsums/RuntimeConfiguration/RuntimeConfiguration.hpp)

set(RuntimeConfigurationSources RuntimeConfiguration.cpp)

include(Einsums_AddModule)
einsums_add_module(
  Einsums RuntimeConfiguration
  SOURCES ${RuntimeConfigurationSources}
  HEADERS ${RuntimeConfigurationHeaders}
  DEPENDENCIES
  MODULE_DEPENDENCIES
    Einsums_Config
    Einsums_Options
    Einsums_Errors
    Einsums_Logging
    Einsums_TypeSupport
    Einsums_Version
    Einsums_Print
  CMAKE_SUBDIRS examples tests
  BASE_LIBNAME Einsums
)
