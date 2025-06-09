#----------------------------------------------------------------------------------------------
# Copyright (c) The Einsums Developers. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
#----------------------------------------------------------------------------------------------

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

set(RuntimeConfigurationHeaders
    Einsums/RuntimeConfiguration/StartupFunction.hpp
    Einsums/RuntimeConfiguration/ShutdownFunction.hpp
    Einsums/RuntimeConfiguration/RuntimeConfiguration.hpp
)

set(RuntimeConfigurationSources RuntimeConfiguration.cpp)

find_package(argparse 3 REQUIRED)

include(Einsums_AddModule)
einsums_add_module(
  Einsums RuntimeConfiguration
  SOURCES ${RuntimeConfigurationSources}
  HEADERS ${RuntimeConfigurationHeaders}
  DEPENDENCIES argparse::argparse
  MODULE_DEPENDENCIES Einsums_Config Einsums_Errors Einsums_Logging Einsums_TypeSupport
  CMAKE_SUBDIRS examples tests
  BASE_LIBNAME Einsums
)
