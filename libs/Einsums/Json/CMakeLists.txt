#----------------------------------------------------------------------------------------------
# Copyright (c) The Einsums Developers. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
#----------------------------------------------------------------------------------------------

set(JsonHeaders Einsums/Json/Json.hpp)

set(JsonSources Json.cpp)

include(Einsums_AddModule)
einsums_add_module(
  Einsums Json
  SOURCES ${JsonSources}
  HEADERS ${JsonHeaders}
  DEPENDENCIES
  MODULE_DEPENDENCIES Einsums_Config Einsums_CXX23 Einsums_Errors
  CMAKE_SUBDIRS examples tests
)
